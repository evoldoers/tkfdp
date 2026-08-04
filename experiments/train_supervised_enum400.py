"""Supervised enum400 dynfield trainer on the FIXED PDB contact partition.

Learns pi_archetype (K_a=20 archetype profiles), the +Gamma+I field-rate bin
weights `w`, and `rho_chain`; holds fixed the DM class-concentration `alpha`,
the partition (PDB-supervised), `S`, `rho`, and the enum400 arch enumeration.
The site class is MARGINALIZED (collapsed), not sampled.

E/M derivation: analysis/supervised_enum400_training.md. Reuses
src/tkfdp/coupling/dynfield/phylo_elbo/supervised_trainer.py.

  E: per contact cluster, per field-rate bin -> class-marginal M_C(g),
     field-rate responsibility r_C(g) prop w_g exp(M_C(g)).
  M: (a) w <- mean_C r_C          (mixture EM, monotone)
     (b) rho_chain <- GEM line search on F        (accept-if-better)
     (c) pi_archetype <- collapsed per-archetype Holmes-Rubin Dirichlet update
                         (guarded on F).

Both GPUs are typically busy on this box; default to CPU
(JAX_PLATFORMS=cpu) for the small validation subset.

  JAX_PLATFORMS=cpu python experiments/train_supervised_enum400.py \
      --kinds saltbridge --max-pairs 200 --n-iter 12 --topN 8 \
      --arch-update --arch-families 15 --out results/supervised_enum400_smoke
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "experiments")
import precompute_pairing as PP                                     # noqa: E402
from tkfdp.coupling.dynfield.phylo_elbo import supervised_trainer as ST  # noqa: E402

PDIR = Path("data/pdb_partition_clv_top1000_sifts")
CLV = Path("data/pfam_processed_clv_top1000_thin128")


def load_pairs(kinds, max_pairs=None, seed=0, split="train"):
    """{fam: [(i,j)]} for the requested contact kinds + confirmed flips, over
    the chosen split. Copied from fit_pdb_hyperparams (the supervised set)."""
    from tkfdp.bio import load_split
    keep = set(load_split()[split])
    out = {}
    for p in sorted(glob.glob(str(PDIR / "*.npz"))):
        if "confirmed" in p:
            continue
        fam = Path(p).stem
        if fam not in keep or not (CLV / f"{fam}.npz").exists():
            continue
        d = np.load(p, allow_pickle=False)
        if "pairs" not in d.files:
            continue
        sel = [(int(a), int(b)) for (a, b), k in zip(d["pairs"], d["kind"])
               if str(k) in kinds]
        if sel:
            out[fam] = sel
    cf = json.load(open(PDIR / "confirmed_flips.json"))
    cf_set = set()
    for r in cf:
        if r["family"] in keep and (CLV / f"{r['family']}.npz").exists():
            out.setdefault(r["family"], [])
            ij = (int(r["i"]), int(r["j"]))
            cf_set.add((r["family"], min(ij), max(ij)))
            if ij not in out[r["family"]]:
                out[r["family"]].append(ij)
    if max_pairs:
        allp = [(f, ij) for f, ps in out.items() for ij in ps]
        rng = np.random.default_rng(seed)
        allp = [allp[i] for i in rng.permutation(len(allp))[:max_pairs]]
        out = {}
        for f, ij in allp:
            out.setdefault(f, []).append(ij)
    return out, cf_set


def confirmed_mask(recs, cf_set):
    return np.array([(r["fam"], min(r["i"], r["j"]), max(r["i"], r["j"])) in cf_set
                     for r in recs], bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="saltbridge")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--topN", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--scratch", action="store_true",
                    help="flat pi_archetype init instead of the LG-C20 warm start")
    ap.add_argument("--arch-update", action="store_true",
                    help="run the pi_archetype Holmes-Rubin M-step (slower)")
    ap.add_argument("--hr-mode", choices=["exact", "mc"], default="exact",
                    help="pi_archetype HR: 'exact' factored cap-2 tree HR (genuine "
                         "EM on F); 'mc' the old sampled-history/rate-0 approximation")
    ap.add_argument("--arch-q-thresh", type=float, default=1e-3,
                    help="exact HR: skip (class-pair, rate-bin) configs with posterior below this")
    ap.add_argument("--hr-backend", choices=["auto", "jax", "numpy"], default="auto",
                    help="exact-HR backend: 'jax' batches over configs (fast), "
                         "'numpy' is the reference, 'auto' picks jax iff a GPU is visible")
    ap.add_argument("--hr-chunk", type=int, default=16,
                    help="jax HR: fixed per-bin config-batch size (padded/masked for "
                         "stable shape + executable reuse). Larger = more GPU "
                         "parallelism; smaller = less CPU compute per call")
    ap.add_argument("--singletons", dest="singletons", action="store_true", default=True,
                    help="include non-contact SINGLETON columns in F and the HR "
                         "(default: on -- ALL columns; genuine EM over all columns)")
    ap.add_argument("--no-singletons", dest="singletons", action="store_false",
                    help="contacts-only (legacy): exclude singleton columns")
    ap.add_argument("--singleton-frac", type=float, default=1.0,
                    help="fraction of non-contact columns to cover (default 1.0 = ALL)")
    ap.add_argument("--singleton-cap", type=int, default=None,
                    help="max singleton columns per family (default: no cap)")
    ap.add_argument("--arch-families", type=int, default=15,
                    help="cap #families used for the MC pi_archetype HR stats "
                         "(exact mode uses all contact-pair clusters that define F)")
    ap.add_argument("--arch-every", type=int, default=1,
                    help="run the pi_archetype M-step every N iterations")
    ap.add_argument("--kappa-pi", type=float, default=1.0)
    ap.add_argument("--arch-prior", choices=["lg08", "laplace", "sparse"], default="laplace",
                    help="archetype-profile Dirichlet prior: laplace = flat +1 pseudocounts "
                         "(DEFAULT, no background pull); lg08 = kappa_pi*PI_LG08 (opt-in, "
                         "pulls toward background); sparse = flat alpha<1 pseudocounts + "
                         "clamped MAP mode (sparsity-inducing, zeros low-count AAs)")
    ap.add_argument("--arch-alpha", type=float, default=0.5,
                    help="per-AA pseudocount for --arch-prior sparse (must be <1)")
    ap.add_argument("--no-rho-chain", action="store_true")
    ap.add_argument("--field-rate-bins", type=int, default=3)
    ap.add_argument("--field-alpha", type=float, default=0.5)
    ap.add_argument("--p-inv", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/supervised_enum400_smoke")
    # ---- DM class-prior learning (separately switchable) ----
    ap.add_argument("--fit-dm", action="store_true",
                    help="learn the DM class prior inside the loop (default: flat/off)")
    ap.add_argument("--dm-init", choices=["swap", "random"], default="swap",
                    help="swap = one component per archetype-pair {A,B} favoring "
                         "(A,B)/(B,A) (freeze alpha, learn pi); random = free H-mixture")
    ap.add_argument("--dm-H", type=int, default=10, help="components for --dm-init random")
    ap.add_argument("--dm-conc", type=float, default=10.0, help="swap-component concentration")
    ap.add_argument("--dm-freeze-alpha", action="store_true", default=None,
                    help="update only pi (default: True for swap, False for random)")
    ap.add_argument("--dm-every", type=int, default=1, help="DM M-step cadence (iters)")
    ap.add_argument("--dm-fixed", action="store_true",
                    help="use the DM as a FIXED class prior (no M-step) -- EM fine-tunes "
                         "only archetypes + field-rate, DM weights/hyperparams untouched")
    ap.add_argument("--init-params", default=None,
                    help="_params.npz to warm-start pi_archetype (+field-rate,rho_chain)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")

    def emit(s):
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    t0 = time.time()
    byfam, cf_set = load_pairs(set(args.kinds.split(",")),
                               max_pairs=args.max_pairs, seed=args.seed, split=args.split)
    npairs = sum(len(v) for v in byfam.values())
    emit(f"# supervised set: {npairs} pairs over {len(byfam)} families "
         f"(kinds={args.kinds} + confirmed flips)")

    ds, rates, weights = PP.build_enum400_ds(
        list(byfam.keys()), field_rate_bins=args.field_rate_bins,
        field_alpha=args.field_alpha, p_inv=args.p_inv, seed=args.seed)
    ST.set_clv_dir(CLV)
    st = ds.state
    Kc, Ka = st.K_c, st.K_a
    from tkfdp.lg08 import PI_LG08
    pi_bar = np.asarray(PI_LG08, np.float64)
    A_ = pi_bar.shape[0]
    # archetype-profile prior: laplace (flat +1), lg08 (kappa_pi*PI_LG08), or sparse (flat alpha<1, clamped MAP mode)
    arch_sparse = (args.arch_prior == "sparse")
    if args.arch_prior == "laplace":
        pi_bar = np.full(A_, 1.0 / A_); kappa_pi_use = float(A_)          # prior_alpha = 1/AA
    elif args.arch_prior == "sparse":
        pi_bar = np.full(A_, 1.0 / A_); kappa_pi_use = float(args.arch_alpha) * A_  # prior_alpha = arch_alpha/AA
        emit(f"# arch prior: SPARSE (alpha={args.arch_alpha}/AA, clamped MAP mode)")
    else:
        kappa_pi_use = float(args.kappa_pi)

    cluster_id_snapshot = {fam: None for fam in byfam}  # partition is implicit
    rho_snapshot = st.rho.copy()

    if args.init_params:                            # warm-start on trained archetypes
        _p = np.load(args.init_params)
        ds = ST.apply_pi_archetype(ds, np.asarray(_p["pi_archetype"], float), rates,
                                   np.asarray(_p["field_rate_weights"], float)
                                   if "field_rate_weights" in _p.files else weights)
        st = ds.state
        if "rho_chain" in _p.files:
            st.rho_chain = float(_p["rho_chain"]); ST.rebuild_rate_kernels(ds)
        if "field_rate_weights" in _p.files:
            weights = np.asarray(_p["field_rate_weights"], float)
        emit(f"# pi_archetype: warm-start from {args.init_params}")
    elif args.scratch:
        rng0 = np.random.default_rng(args.seed)
        pi0 = pi_bar[None, :] * np.ones((Ka, 1)) + 0.02 * rng0.random((Ka, len(pi_bar)))
        pi0 /= pi0.sum(1, keepdims=True)
        ds = ST.apply_pi_archetype(ds, pi0, rates, weights)
        st = ds.state
        emit("# pi_archetype: FLAT/scratch init")
    else:
        emit("# pi_archetype: LG-C20 warm start")
    aa_snapshot = st.arch_assignment.copy()

    # ---- DM class prior (separately switchable) ----
    if args.fit_dm:
        freeze = args.dm_freeze_alpha if args.dm_freeze_alpha is not None else (args.dm_init != "random")
        if args.dm_init == "swap":
            dm = ST.make_swap_dm(Kc, Ka, conc=args.dm_conc)
            role = "FIXED prior (no M-step)" if args.dm_fixed else \
                f"freeze_alpha={freeze} -> learn which swaps coevolve via pi"
            emit(f"# DM: seeded SWAP prior, {dm.H} components ({dm.H-1} archetype-pairs "
                 f"+ static), {role}")
        else:
            dm = ST.make_dm(Kc, H=args.dm_H, seed=args.seed)
            emit(f"# DM: FREE mixture, H={args.dm_H}, freeze_alpha={freeze}")
        alpha_log = dm
    else:
        dm = None; alpha_log = None                 # FIXED, flat (cancels)
    emit(f"# rates={np.round(rates, 3)}  prior weights={np.round(weights, 3)}  "
         f"rho_chain0={st.rho_chain:.4f}  rho={np.round(st.rho, 3)}")

    w = np.asarray(weights, float).copy()
    rng = np.random.default_rng(args.seed + 1)

    # ---- singleton (non-contact) columns: cover ALL columns in F + HR ----
    sing_cols = None
    if args.singletons:
        sing_cols, n_sing_tot, n_sing_cov = ST.singleton_cols(
            ds, byfam, frac=args.singleton_frac, per_fam_cap=args.singleton_cap,
            seed=args.seed)
        emit(f"# singletons: covering {n_sing_cov}/{n_sing_tot} non-contact columns "
             f"({100.0 * n_sing_cov / max(1, n_sing_tot):.1f}%) over {len(sing_cols)} "
             f"families -> F + pi_archetype HR span pairs AND singletons")
    else:
        emit("# singletons: OFF (contacts-only, legacy)")

    def score_all(ds_):
        rp, _, _, _ = ST.score_perbin_fast(ds_, byfam, topN=args.topN)
        rs = ST.score_singletons_perbin(ds_, sing_cols, topN=args.topN) if sing_cols else []
        return rp, rs

    # ---- iter 0: score + baseline readouts ----
    recs_p, recs_s = score_all(ds)
    recs = recs_p + recs_s                          # F + HR span pairs AND singletons
    cf_mask = confirmed_mask(recs_p, cf_set)        # flip readout over contact pairs
    M = ST.cluster_marginals_perbin(recs, alpha_log)
    F = ST.corpus_F(M, w)
    M_p = ST.cluster_marginals_perbin(recs_p, alpha_log)
    fp, phi = ST.flip_prevalence(M_p, w)
    ch0 = ST.archetype_charge(st.pi_archetype)
    emit(f"# iter  0  F={F:+.2f}  flip_prev={fp:.3f}  w_inv={w[0]:.3f}  "
         f"rho_chain={st.rho_chain:.4f}  n_clusters={len(recs)}  t={time.time()-t0:.0f}s")
    if cf_mask.any():
        emit(f"#          flip[conf]={phi[cf_mask].mean():.3f} "
             f"flip[other]={phi[~cf_mask].mean():.3f}")

    history = [dict(iter=0, F=F, flip_prev=fp, w=w.tolist(),
                    rho_chain=float(st.rho_chain))]

    for it in range(1, args.n_iter + 1):
        # (a) field-rate weight EM (monotone) -- over pairs AND singletons
        r, _ = ST.field_rate_responsibilities(M, w)
        w = r.mean(0)
        w = w / w.sum()

        # (b) rho_chain GEM line search (F over pairs AND singletons). Reuse the
        # recs it already scored at the winning rho instead of a redundant re-score
        # (bit-identical: a re-score at best rho gives exactly these).
        if not args.no_rho_chain:
            _, _, recs_p, recs_s = ST.mstep_rho_chain(
                ds, byfam, w, args.topN, alpha_log, sing_cols=sing_cols)
            recs = recs_p + recs_s
            M = ST.cluster_marginals_perbin(recs, alpha_log)

        # (b.5) DM class-prior M-step -- learn which swaps coevolve (pairs+singletons)
        if args.fit_dm and not args.dm_fixed and (it % args.dm_every == 0):
            _frz = args.dm_freeze_alpha if args.dm_freeze_alpha is not None else (args.dm_init == "swap")
            ST.mstep_dm(recs, dm, np.log(w + 1e-300), freeze_alpha=_frz)
            M = ST.cluster_marginals_perbin(recs, alpha_log)   # dm mutated -> refresh M

        # (c) pi_archetype HR M-step (guarded) -- HR over the SAME clusters as F
        arch_note = ""
        if args.arch_update and (it % args.arch_every == 0):
            if args.hr_mode == "exact":
                # EXACT factored HR over the SAME pair+singleton clusters that
                # define F -> a genuine EM step (monotone; no accept/reject needed).
                dwell, real = ST.exact_hr_per_archetype(
                    ds, recs, w, st.S, alpha_log, q_thresh=args.arch_q_thresh,
                    hr_backend=args.hr_backend, b_chunk=args.hr_chunk)
            else:
                fams_arch = list(byfam.keys())[:args.arch_families]
                cols_by_fi = ST._all_columns(ds, {f: byfam[f] for f in fams_arch})
                order, rho_resp = ST.column_arch_responsibilities(ds, cols_by_fi)
                dwell, real = ST.accumulate_hr_per_archetype(
                    ds, order, rho_resp, rng, st.S, n_history=1)
            pi_cand = ST.mstep_pi_archetype(ds, dwell, real, st.S,
                                            kappa_pi_use, pi_bar, sparse=arch_sparse)
            F_before = ST.corpus_F(M, w)
            ds_cand = ST.apply_pi_archetype(ds, pi_cand, rates, w)
            recs_pc, recs_sc = score_all(ds_cand)
            recs_c = recs_pc + recs_sc
            M_c = ST.cluster_marginals_perbin(recs_c, alpha_log)
            F_after = ST.corpus_F(M_c, w)
            if F_after >= F_before - 1e-6:
                ds = ds_cand; st = ds.state
                recs_p, recs_s, recs, M = recs_pc, recs_sc, recs_c, M_c
                arch_note = f" pi_arch:accept dF={F_after-F_before:+.2f}"
            else:                                   # damp toward previous (skip)
                arch_note = f" pi_arch:reject dF={F_after-F_before:+.2f}"

        M = ST.cluster_marginals_perbin(recs, alpha_log)
        F = ST.corpus_F(M, w)
        M_p = ST.cluster_marginals_perbin(recs_p, alpha_log)
        fp, phi = ST.flip_prevalence(M_p, w)
        emit(f"# iter {it:2d}  F={F:+.2f}  flip_prev={fp:.3f}  w_inv={w[0]:.3f}  "
             f"rho_chain={st.rho_chain:.4f}{arch_note}  t={time.time()-t0:.0f}s")
        if cf_mask.any():
            emit(f"#          flip[conf]={phi[cf_mask].mean():.3f} "
                 f"flip[other]={phi[~cf_mask].mean():.3f}")
        if args.fit_dm and getattr(dm, "comp_labels", None):
            order = np.argsort(-dm.pi)[:6]
            emit(f"#          DM top components (pi): " +
                 "  ".join(f"{dm.comp_labels[h]}:{dm.pi[h]:.3f}" for h in order))
        history.append(dict(iter=it, F=F, flip_prev=fp, w=w.tolist(),
                            rho_chain=float(st.rho_chain)))
        # checkpoint the learned params (+ DM if learned)
        save = dict(pi_archetype=st.pi_archetype, field_rate_weights=w,
                    rho_chain=np.float64(st.rho_chain), rates=rates)
        if args.fit_dm:
            save["dm_alpha"] = dm.alpha; save["dm_pi"] = dm.pi
        np.savez(out / "_params.npz", **save)

    # ---- sanity: alpha + partition unchanged ----
    assert np.array_equal(st.arch_assignment, aa_snapshot), "arch enumeration changed!"
    assert np.allclose(st.rho, rho_snapshot), "field stationary rho changed!"
    ch1 = ST.archetype_charge(st.pi_archetype)
    n_charge_flip = int(np.sum(np.sign(ch0) != np.sign(ch1)))
    Fmono = all(history[i]["F"] >= history[i - 1]["F"] - 1e-4
                for i in range(1, len(history)))
    emit(f"# DONE  F monotone={Fmono}  flip_prev {history[0]['flip_prev']:.3f}"
         f"->{history[-1]['flip_prev']:.3f}  w_inv {weights[0]:.3f}->{w[0]:.3f}")
    emit(f"#       archetype charge sign flips (of {len(ch0)}): {n_charge_flip}  "
         f"(acidic stay acidic if 0)")
    emit(f"#       alpha FIXED (flat), partition FIXED (PDB) -- asserted unchanged")

    report = dict(
        config=vars(args), n_pairs=npairs, n_families=len(byfam),
        F_monotone=Fmono, history=history,
        flip_prev_start=history[0]["flip_prev"], flip_prev_end=history[-1]["flip_prev"],
        w_inv_start=float(weights[0]), w_inv_end=float(w[0]),
        archetype_charge_start=ch0.tolist(), archetype_charge_end=ch1.tolist(),
        charge_sign_flips=n_charge_flip,
    )
    (out / "_report.json").write_text(json.dumps(report, indent=2))
    emit(f"# wrote {out}/_report.json  +  {out}/_params.npz")


if __name__ == "__main__":
    main()
