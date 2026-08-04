"""Train the class-marginalised, field-rate-marginalised (+Gamma+I) dynfield
model on the PDB-supervised partition, and test whether the field-rate flip
posterior phi(C) separates data-confirmed acid-base flips from other contacts.

Frozen LG-Cxx archetypes; learned target = arch_assignment via MH over the
class+rate-marginal evidence. See marginal_scorer.py / marginal_trainer.py and
appendix-tkfdp.tex "Rate heterogeneity (+Gamma+I)".

Usage (GPU 1, baseline on GPU 0):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python experiments/train_marginal_dynfield.py \
    --n-families 150 --pairs-only --n-sweeps 8 --out-dir results/marginal_c20_smoke
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path.home() / "tkf-mixdom" / "python"))

from tkfdp.bio import load_split
from tkfdp.lg08 import S_LG08
from tkfdp.coupling.dynfield.phylo_elbo.corpus_state import (
    build_corpus_state, apply_pdb_partition, _cluster_columns_by_id)
from tkfdp.coupling.dynfield.phylo_elbo import marginal_trainer as mt
from tkfdp.coupling.dynfield.phylo_elbo import field_rate_trainer as frt
from tkfdp.coupling.dynfield.phylo_elbo.rate_hetero import gamma_plus_inv_rates


def _lg_c20():
    from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c20
    p = np.asarray(le_gascuel_c20()[0], dtype=np.float64)
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clv-dir", default="data/pfam_processed_clv_top1000_thin128")
    ap.add_argument("--partition-dir", default="data/pdb_partition_clv_top1000_sifts")
    ap.add_argument("--confirmed-flips",
                    default="data/pdb_partition_clv_top1000_sifts/confirmed_flips.json")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-families", type=int, default=150)
    ap.add_argument("--pairs-only", action="store_true", default=False)
    ap.add_argument("--K-c", type=int, default=20)
    ap.add_argument("--max-cluster-size", type=int, default=2,
                    help="Cluster cap. >2 = higher-order coevolution: m columns "
                         "share one field trajectory (m-way synchronised flip). "
                         "m>2 forward is moment-matching (approx); enum400 fast "
                         "path handles any m.")
    ap.add_argument("--rho-chain", type=float, default=0.15)
    ap.add_argument("--field-rate-bins", type=int, default=3)
    ap.add_argument("--field-alpha", type=float, default=0.5)
    ap.add_argument("--p-inv", type=float, default=0.5)
    ap.add_argument("--n-sweeps", type=int, default=8)
    ap.add_argument("--scalable", action="store_true", default=False,
                    help="Field-rate-marginal + class-SAMPLING (G x cost, full "
                         "K_c). Default is exact class marginalisation (K_c^2).")
    ap.add_argument("--enum400", action="store_true", default=False,
                    help="Enumerated model: L_field=2, K_c=K_a^2 classes = ALL "
                         "(arch@0, arch@1) pairs; arch_assignment FIXED to that "
                         "enumeration and NOT sampled -- cn over the K_a^2 classes "
                         "does everything. Implies --scalable, forces K_c/L_field.")
    ap.add_argument("--field-rho0", type=float, default=0.6,
                    help="enum400: prior weight of field state 0 (MUST be > 0.5). "
                         "Asymmetric field prior breaks the theta0<->theta1 "
                         "relabeling symmetry so (A,B) != (B,A) is identifiable.")
    ap.add_argument("--dm-prior", action="store_true", default=False,
                    help="Mixture-of-Dirichlet-multinomials base prior over each "
                         "cluster's class distribution (appendix sec:dm-mixture). "
                         "Exchangeable/cap>2-safe class-combination preference; "
                         "components = coevolutionary cluster types.")
    ap.add_argument("--dm-components", type=int, default=20)
    ap.add_argument("--dm-alpha-pi", type=float, default=5.0)
    ap.add_argument("--site-rate-bins", type=int, default=0,
                    help="Per-site substitution-rate bins (Gamma), 0 = off. "
                         "Adds an invariant bin -> G_s = this+1; folded into "
                         "the class axis (base_class x rate_bin).")
    ap.add_argument("--site-alpha", type=float, default=0.5)
    ap.add_argument("--site-p-inv", type=float, default=0.3)
    ap.add_argument("--arch-passes", type=int, default=1,
                    help="Discovery: arch_assignment MH passes per sweep (arch "
                         "is cheap vs the z+cn pass; more helps it track the "
                         "shifting partition).")
    ap.add_argument("--discover", action="store_true", default=False,
                    help="UNSUPERVISED: discover the cap-2 partition (CRP z-move, "
                         "alpha_z fixed at the corpus default 100) from all-"
                         "singletons instead of using PDB labels. Implies "
                         "--scalable; reports flip posterior on discovered pairs.")
    ap.add_argument("--site-invariant", action="store_true", default=False,
                    help="Include the rate-0 invariant SUBSTITUTION bin (+Gamma+I; "
                         "only 100%%-constant columns can take it, -inf otherwise). "
                         "Off = +Gamma only (all rates > 0).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="Resume from <out-dir>/_chkpt.npz (rolling sweep-level "
                         "checkpoint). Requires --out-dir and a matching config "
                         "(same families, K_c/K_a/L_field).")
    ap.add_argument("--ckpt-every-sec", type=int, default=900,
                    help="Discovery: also checkpoint this often WITHIN a sweep "
                         "(0=only at sweep boundaries). Sweeps on the full corpus "
                         "run hours, so the default caps crash exposure at ~15min.")
    ap.add_argument("--audit-elbo", type=int, default=0, metavar="N",
                    help="Instead of training: build the state (use with --resume "
                         "to load a checkpoint), score every cluster by the "
                         "moment-matching forward AND the exact cap-2 reference, "
                         "report the moment-projection gap, and exit. N>0 audits a "
                         "random N-cluster sample; N<0 audits all clusters.")
    ap.add_argument("--init-arch", default="",
                    help="Load arch_assignment (the global field map; families "
                         "need NOT match) from this checkpoint before running. "
                         "For TRANSFER: train arch supervised on one split, then "
                         "discover on another. Combine with --arch-passes 0 to "
                         "freeze the transferred field map during discovery.")
    args = ap.parse_args()

    if args.resume and not args.out_dir:
        sys.exit("--resume requires --out-dir (that is where _chkpt.npz lives)")

    rng = np.random.default_rng(args.seed)
    clv = Path(args.clv_dir)
    fams_all = json.loads((clv / "index.json").read_text())["families"]
    keep = set(load_split()[args.split])
    pdir = Path(args.partition_dir)
    fams = [f for f in fams_all if f in keep and (pdir / f"{f}.npz").exists()]
    fams = fams[:args.n_families]
    paths = [str(clv / f"{f}.npz") for f in fams]
    print(f"# {len(paths)} {args.split} families with PDB partitions", flush=True)

    K_a = 20
    L_field = 4
    if args.enum400:                          # enumerated (arch@0, arch@1) classes
        args.scalable = True
        args.arch_passes = 0                  # arch is FIXED; never sampled
        L_field = 2
        args.K_c = K_a * K_a                  # 400
    pa = _lg_c20()
    t0 = time.time()
    st = build_corpus_state(paths, K_c=args.K_c, K_a=K_a, L_field=L_field,
                            pi_archetype=pa, S=np.asarray(S_LG08),
                            rho_chain=args.rho_chain, rng=rng, n_tau_bins=32,
                            max_cluster_size=args.max_cluster_size, verbose=False)
    if args.enum400:
        if args.field_rho0 <= 0.5:
            sys.exit("--field-rho0 must be > 0.5 (else the theta0<->theta1 "
                     "relabeling stays unidentifiable)")
        # class c == (arch@theta0, arch@theta1) = (c // K_a, c % K_a); FIXED,
        # never sampled -- cn over the K_a^2 classes carries all field structure.
        aa = np.stack([np.arange(args.K_c) // K_a, np.arange(args.K_c) % K_a], 1)
        st.arch_assignment = aa.astype(np.int32)
        # asymmetric field prior rho0 > rho1 -> (A,B) != (B,A) identifiable.
        st.rho = np.array([args.field_rho0, 1.0 - args.field_rho0], np.float64)
        st.refresh_pi_field()
        print(f"# enum400: K_c={args.K_c} classes = all ({K_a}x{K_a}) arch pairs, "
              f"L_field=2, arch FIXED (no arch moves), rho={st.rho.round(3)}",
              flush=True)
    if args.discover:
        args.scalable = True                 # discovery uses the scalable path
    else:
        apply_pdb_partition(st, str(pdir), verbose=False)
    print(f"# corpus{'' if args.discover else '+partition'} built in "
          f"{time.time()-t0:.1f}s", flush=True)

    # Resume: overwrite partition/classes/arch + rng from the rolling checkpoint
    # (must precede build_*_state / init_ll so the derived tables are rebuilt
    # from the resumed state). load_checkpoint verifies family order + config.
    start_sweep = 0
    if args.resume:
        from tkfdp.coupling.dynfield.phylo_elbo import corpus_state as cs
        ckpt = Path(args.out_dir) / "_chkpt.npz"
        if ckpt.exists():
            start_sweep, rng = cs.load_checkpoint(st, ckpt, rng)
            npair = sum(1 for fam in st.families
                        for _, c in _cluster_columns_by_id(fam.cluster_id).items()
                        if len(c) == 2)
            print(f"# ===== RESUME =====", flush=True)
            print(f"#   checkpoint : {ckpt}", flush=True)
            print(f"#   at sweep   : {start_sweep} (will continue to "
                  f"{args.n_sweeps})", flush=True)
            print(f"#   recovered  : {len(st.families)} families, {npair} pairs, "
                  f"arch {st.arch_assignment.shape}", flush=True)
            print(f"# ==================", flush=True)
        else:
            print(f"# --resume set but {ckpt} missing; starting fresh at sweep 0",
                  flush=True)

    # Transfer a trained field map: overwrite arch_assignment (class x field ->
    # archetype; family-independent) from another checkpoint. Must precede
    # build_*_state so the derived pi_field/tables reflect it. Freeze with
    # --arch-passes 0.
    if args.init_arch:
        da = np.load(args.init_arch, allow_pickle=False)
        aa = np.asarray(da["arch_assignment"], np.int32)
        if aa.shape != st.arch_assignment.shape:
            sys.exit(f"--init-arch shape {aa.shape} != corpus "
                     f"{st.arch_assignment.shape} (K_c/L_field must match)")
        st.arch_assignment = aa.copy()
        st.refresh_pi_field()
        print(f"# init-arch: transferred arch_assignment {aa.shape} from "
              f"{args.init_arch}", flush=True)

    # confirmed-flip pair set: {(family, i, j)}
    cf = {(r["family"], int(r["i"]), int(r["j"]))
          for r in json.loads(Path(args.confirmed_flips).read_text())}

    # Supervised: fixed cluster list from the partition. Discovery: none (starts
    # all-singleton; the z-move finds the pairs).
    clusters, flip_mask = [], None
    if not args.discover:
        kinds_flip = []
        for fi, fam in enumerate(st.families):
            for cid, cols in _cluster_columns_by_id(fam.cluster_id).items():
                m = len(cols)
                if m == 2:
                    i, j = sorted(int(x) for x in cols)
                    clusters.append((fi, np.asarray(cols, np.int32)))
                    kinds_flip.append((fam.family_id, i, j) in cf)
                elif m == 1 and not args.pairs_only:
                    clusters.append((fi, np.asarray(cols, np.int32)))
                    kinds_flip.append(False)
        flip_mask = np.asarray(kinds_flip, bool)
        n_pairs = sum(1 for (_, c) in clusters if len(c) == 2)
        print(f"# {len(clusters)} clusters ({n_pairs} pairs, "
              f"{flip_mask.sum()} confirmed-flip pairs)", flush=True)

    rates, weights = gamma_plus_inv_rates(args.field_rate_bins, args.field_alpha,
                                          args.p_inv)
    print(f"# field-rate bins: rates={np.round(rates,3)} w={np.round(weights,3)}",
          flush=True)

    log_stream = None
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        log_stream = open(Path(args.out_dir) / "training.log", "w")

    if args.scalable:
        site_rate_bin, rates_sub = None, None
        if args.site_rate_bins > 0:
            from tkfdp.lg08 import PI_LG08
            from tkfdp.coupling.dynfield.phylo_elbo.rate_hetero import (
                estimate_site_rate_bins, discrete_gamma_rates)
            if args.site_invariant:
                # +Gamma+I: the rate-0 invariant bin is available, but the EB
                # fit assigns it ONLY to 100%-constant columns (non-constant ->
                # log-lik -inf, the true value, NOT a floor). The fit holds the
                # field at zero (pure LG08), so it can't make a varying column
                # look frozen.
                rates_sub, w_sub = gamma_plus_inv_rates(
                    args.site_rate_bins, args.site_alpha, args.site_p_inv)
            else:
                rates_sub = discrete_gamma_rates(args.site_rate_bins,
                                                 args.site_alpha)
                w_sub = np.ones(len(rates_sub)) / len(rates_sub)
            t0 = time.time()
            site_rate_bin = estimate_site_rate_bins(
                paths, np.asarray(S_LG08), np.asarray(PI_LG08), rates_sub, w_sub)
            allb = np.concatenate(list(site_rate_bin.values()))
            print(f"# per-site rates: G_s={len(rates_sub)} "
                  f"rates={np.round(rates_sub,3)} bins="
                  f"{ {int(b):int((allb==b).sum()) for b in range(len(rates_sub))} } "
                  f"in {time.time()-t0:.1f}s", flush=True)
        dm = None
        if args.dm_prior:
            from tkfdp.coupling.dynfield.phylo_elbo.dm_prior import DMPrior
            dm = DMPrior(args.K_c, H=args.dm_components, alpha_pi=args.dm_alpha_pi)
            print(f"# DM prior: mixture of {args.dm_components} Dirichlet-"
                  f"multinomials over K_c={args.K_c} classes", flush=True)
        if args.discover:
            from tkfdp.coupling.dynfield.phylo_elbo import field_rate_discovery as fd
            ds = fd.build_discovery_state(st, rates, weights,
                                          site_rate_bin=site_rate_bin,
                                          rates_sub=rates_sub)
            ds.enum400 = args.enum400        # invariant-bin fast path in cn
            ds.dm = dm                       # mixture-of-DM class-composition prior
            if args.resume and dm is not None:
                ckpt = Path(args.out_dir) / "_chkpt.npz"
                if ckpt.exists():
                    fd.restore_dm(ds, ckpt)
                    print(f"# DM resume: restored mixture + {len(dm.h)} cluster "
                          f"components", flush=True)
            ncol = sum(f.L for f in st.families)
            print(f"# DISCOVER: {ncol} columns (all singletons), alpha_z="
                  f"{st.alpha_z}, G={len(rates)} G_s={ds.G_s}; init ...",
                  flush=True)
            t0 = time.time(); fd.init_ll(ds)
            print(f"# init in {time.time()-t0:.1f}s", flush=True)
            if args.audit_elbo != 0:
                from tkfdp.coupling.dynfield.phylo_elbo.elbo_audit import run_audit
                run_audit(ds, 'ds', sample=(args.audit_elbo if args.audit_elbo > 0
                                            else None), rng=rng)
                return
            fd.run_interleaved(ds, args.n_sweeps, rng,
                               arch_passes=args.arch_passes,
                               confirmed_flip=cf, log_stream=log_stream,
                               out_dir=args.out_dir or None,
                               start_sweep=start_sweep,
                               ckpt_every_sec=args.ckpt_every_sec)
        else:
            fr = frt.build_field_rate_state(st, clusters, rates, weights,
                                            site_rate_bin=site_rate_bin,
                                            rates_sub=rates_sub)
            fr.enum400 = args.enum400        # invariant-bin fast path in cn
            if args.resume and dm is not None:
                ckpt = Path(args.out_dir) / "_chkpt.npz"
                if ckpt.exists():
                    frt.restore_dm(fr, dm, ckpt)
                    print(f"# DM resume: restored mixture + {len(dm.h)} cluster "
                          f"components", flush=True)
            print(f"# scalable: {len(clusters)} clusters, class-sampling + "
                  f"field-rate-marginal (G={len(rates)}, G_s={fr.G_s}); init ...",
                  flush=True)
            t0 = time.time(); frt.init_ll(fr)
            print(f"# init in {time.time()-t0:.1f}s", flush=True)
            if args.audit_elbo != 0:
                from tkfdp.coupling.dynfield.phylo_elbo.elbo_audit import run_audit
                run_audit(fr, 'fr', sample=(args.audit_elbo if args.audit_elbo > 0
                                            else None), rng=rng)
                return
            frt.run_training(fr, args.n_sweeps, rng, do_arch=not args.enum400,
                             confirmed_flip_mask=flip_mask, log_stream=log_stream,
                             out_dir=args.out_dir or None,
                             start_sweep=start_sweep, dm=dm,
                             ckpt_every_sec=args.ckpt_every_sec)
    else:
        ms = mt.build_marginal_state(st, clusters, rates, weights)
        print(f"# {len(ms.specs)} (cluster,labeling) specs; init ...", flush=True)
        t0 = time.time(); mt.init_ll_table(ms)
        print(f"# init in {time.time()-t0:.1f}s", flush=True)
        mt.run_marginal_training(ms, args.n_sweeps, rng,
                                 confirmed_flip_mask=flip_mask,
                                 log_stream=log_stream,
                                 out_dir=args.out_dir or None,
                                 start_sweep=start_sweep)
    if log_stream:
        log_stream.close()


if __name__ == "__main__":
    main()
