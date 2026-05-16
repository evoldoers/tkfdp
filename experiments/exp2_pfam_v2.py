"""New Pfam SVI driver — composes items 1-6 of the 2026-05-08 reparam.

Per outer iteration:
  1. Build per-(c1, c2) log P cache from current Potts DP atoms + pi_class.
  2. Joint partition + class Gibbs on each MSA (gibbs_sweep_K).
  3. Update per-class pi via Dirichlet conjugacy (secret-destination ghost).
  4. Update per-site eta via Gamma posterior mean.
  5. Update Potts atoms via per-atom Laplace MAP.
  6. (Periodic) Potts DP CRP-Gibbs over h_{c, c'} assignments + alpha_H Escobar-West.

Cluster-1 default (no partition pairs). Use --init-pair-fraction > 0 to
seed pairs and exercise the pair-side machinery. With pairs, the Potts
atoms get observations and the Laplace updates have signal.

Example:
  python3 experiments/exp2_pfam_v2.py --families PF00027 --K 2 \
      --init-pair-fraction 0.4 --n-outer 6 --out-dir results/...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import jax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tkfdp.lg08 import PI_LG08, S_LG08_F81, ALPHA_ORDER
from tkfdp.partition_K import gibbs_sweep_K, n_pairs_K
from tkfdp.pfam_data import families_from_list
from tkfdp.svi import (SVIState, accumulate_real_counts,
                            build_log_P_cache_K_atoms,
                            init_svi_state,
                            per_column_log_marginal_class_specific,
                            potts_dp_crp_sweep,
                            update_eta_per_col_diagnostic,
                            update_pi_class,
                            update_potts_atoms_jit,
                            hr_per_class_per_msa)
from tkfdp.potts_dp import escobar_west_alpha_H_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=str, default="",
                     help="Comma-separated Pfam family IDs to load via "
                          "Stockholm parsing. Ignored when --processed-dir is "
                          "set. Either --families or --processed-dir must be "
                          "supplied.")
    ap.add_argument("--min-cherries", type=int, default=8)
    ap.add_argument("--min-aa-fraction", type=float, default=0.0)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--n-outer", type=int, default=100)
    ap.add_argument("--n-gibbs-per-outer", type=int, default=1)
    ap.add_argument("--n-laplace-steps", type=int, default=20)
    ap.add_argument("--init-pair-fraction", type=float, default=0.4)
    ap.add_argument("--a-eta", type=float, default=2.0)
    ap.add_argument("--b-eta", type=float, default=2.0)
    ap.add_argument("--kappa-pi", type=float, default=4.0)
    ap.add_argument("--alpha-c", type=float, default=10.0)
    ap.add_argument("--alpha-z", type=float, default=100.0,
                     help="Ewens concentration on the partition. "
                          "Per-pair log-prior cost is -log(alpha_z) (size-{1,2} Ewens). "
                          "alpha_z > 1 favors singletons; alpha_z < 1 favors pairs. "
                          "Default 100 -> per-pair penalty ~4.6 nats (matches the "
                          "old -2*log(alpha_c=10) heuristic).")
    ap.add_argument("--alpha-H", type=float, default=1.0)
    ap.add_argument("--n-crp-every", type=int, default=4,
                    help="Run Potts DP CRP-Gibbs every this many outer iters.")
    ap.add_argument("--anchor-families", type=str, default="",
                    help="Comma-separated subset of families whose partition is "
                         "FIXED to PDB Cα<8Å contacts during training (only c_s "
                         "is resampled; partner_s is locked). Provides a touch "
                         "of supervision for the Potts atom learning.")
    ap.add_argument("--restrict-anchor-families", type=str, default="",
                    help="Comma-separated subset of families whose partition "
                         "GIBBS PROPOSAL is RESTRICTED to PDB Cα<8Å contact "
                         "candidates (every column-pair below the distance "
                         "threshold; not greedy-matched). Both partner and "
                         "class are resampled, but the partner draw is "
                         "supported only on the candidate set. Mutually "
                         "exclusive with --anchor-families. Strict-restrict: "
                         "families whose PDB contacts can't be loaded are "
                         "treated as full-pair-pool non-anchors with a "
                         "warning. Use a manifest builder to pre-validate.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/exp2_pfam_v2"))
    ap.add_argument("--use-elbo", action="store_true",
                    help="Replace the exact 400-state log-P pair loss with the "
                         "Holmes-Rubin closed-form variational ELBO at constant "
                         "Q_hat (geometric-mean construction with damped "
                         "(bar_p_1, bar_p_2) fixed-point inner iteration). "
                         "Strict lower bound on log P, exact at H=0. Validation "
                         "log-likelihood is always computed via the exact path.")
    ap.add_argument("--checkpoint-every", type=int, default=5,
                    help="Dump rolling checkpoint to <out_dir>/_chkpt every N "
                         "outer iters (atomic write; survives kill mid-run). "
                         "0 disables.")
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="Resume from a checkpoint dir (typically "
                         "<out_dir>/_chkpt). Family list and K_c must match.")
    ap.add_argument("--resume-globals-from", type=Path, default=None,
                    help="Load ONLY the global params (pi_class + Potts DP "
                         "atoms/assignments/counts/TSB) from a checkpoint dir, "
                         "and re-initialize per-family latents fresh. Lets a "
                         "K=4 substitution-trained checkpoint seed a run on a "
                         "different corpus (e.g. the PDB-anchored subset). "
                         "Mutually exclusive with --resume-from.")
    ap.add_argument("--val-families", type=str, default="",
                    help="Comma-separated held-out family IDs for periodic val "
                         "log-likelihood. Empty = no val LL = no early stop.")
    ap.add_argument("--val-every", type=int, default=5,
                    help="Compute val LL every N outer iters (only when "
                         "--val-families is non-empty).")
    ap.add_argument("--val-burnin", type=int, default=20)
    ap.add_argument("--val-samples", type=int, default=10)
    ap.add_argument("--processed-dir", type=Path, default=None,
                    help="Path to a preprocessed Pfam directory (with "
                         "index.json + per-family .npz, from "
                         "experiments/preprocess_pfam_topN.py). When set, "
                         "loads the first --n-families families from the index "
                         "instead of parsing Stockholm files. Faster and "
                         "supports much larger corpora.")
    ap.add_argument("--n-families", type=int, default=None,
                    help="When --processed-dir is set, take this many families "
                         "from the front of the index (which is sorted by "
                         "cherry count). Default: all in the index.")
    ap.add_argument("--em-warmup-iters", type=int, default=500,
                    help="Max iterations of soft EM on the column→site-class "
                         "assignments BEFORE the SVI loop. Default 500; set "
                         "to 0 to disable. Uses the Dirichlet-multinomial "
                         "likelihood under each class's pi (no Potts "
                         "coupling, no partner moves). Stops early when "
                         "L1(delta pi_class) < --em-warmup-tol.")
    ap.add_argument("--em-warmup-tol", type=float, default=1e-5,
                    help="Convergence tolerance: max-class L1 change in "
                         "pi_class between EM iters.")
    ap.add_argument("--em-warmup-seeds", type=int, default=50,
                    help="Multi-seed soft EM: run from this many random "
                         "Dirichlet inits, pick the one with the highest "
                         "training data log-likelihood. EM is fast (~150 ms/"
                         "seed at 1000 families) so 50 seeds is ~10s.")
    ap.add_argument("--K-H-max", type=int, default=10,
                    help="Cap on the number of Potts atoms allocated by TSB. "
                         "Default 10 (no-op for K_c <= 4 since 10 = 4*5/2; "
                         "an active cap for K_c >= 5). When < K_c(K_c+1)/2, "
                         "class-pairs are round-robin assigned to atoms at "
                         "init; TSB resampling specializes.")
    ap.add_argument("--use-side-potentials", action="store_true",
                    help="Add per-class-pair singleton side-potential "
                         "vectors h_a, h_b ~ N(0, h_prior_tau^-1) per AA, "
                         "MAP-fit alongside H atoms via Adam; self-pairs "
                         "(c, c) tied h_a = h_b for joint-Q reversibility. "
                         "See main.tex Remark 'Per-class-pair side "
                         "potentials'. Adds ~K_c(K_c+1)/2 x 2 x A params "
                         "per atom and roughly doubles the per-Adam-step "
                         "cost; OFF by default.")
    ap.add_argument("--h-prior-tau", type=float, default=4.0,
                    help="Gaussian-prior precision for the per-class-pair "
                         "side-potential vectors h_a, h_b (centered at zero).")
    ap.add_argument("--n-tau", type=int, default=50,
                    help="Number of geometric-spaced tau bins for log_P "
                         "caching. Geomspace gives fine resolution at small t "
                         "where exp(Q*t) varies fastest. Smaller values trade "
                         "tau-precision for cache memory + faster eigh chain.")
    ap.add_argument("--patience", type=int, default=6,
                    help="Stop early if val LL fails to improve for this many "
                         "consecutive val-LL evaluations. Set 0 to disable. "
                         "Each val cycle is --val-every outer iters apart, so "
                         "patience=6 with val-every=5 means 30 outers of "
                         "no improvement triggers stop.")
    args = ap.parse_args()
    if args.processed_dir is None and not args.families.strip():
        ap.error("Either --processed-dir or --families must be supplied.")
    if args.resume_from is not None and args.resume_globals_from is not None:
        ap.error("--resume-from and --resume-globals-from are mutually exclusive.")
    if args.anchor_families and args.restrict_anchor_families:
        ap.error("--anchor-families and --restrict-anchor-families are mutually exclusive.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.processed_dir is not None:
        from tkfdp.pfam_data_fast import families_from_processed
        print(f"Loading preprocessed Pfam corpus from {args.processed_dir} ...")
        families = families_from_processed(
            args.processed_dir, n_families=args.n_families,
            min_cherries=args.min_cherries,
        )
        print(f"  Loaded {len(families)} families "
                f"({sum(fc.n_cherries for fc in families)} cherries, "
                f"{sum(fc.L for fc in families)} columns)")
    else:
        family_ids = [f.strip() for f in args.families.split(",") if f.strip()]
        print(f"Loading {len(family_ids)} Pfam families ...")
        families = families_from_list(
            family_ids, min_cherries=args.min_cherries,
            min_aa_fraction=args.min_aa_fraction
        )
    for fc in families:
        print(f"  {fc.family}: L={fc.L}, cherries={fc.n_cherries}")

    rng = np.random.default_rng(args.seed)
    per_family_data = []
    all_t = []
    for fc in families:
        per_family_data.append(dict(
            family=fc.family, L=fc.L, n_cherries=fc.n_cherries, tau=fc.tau,
            aa_a=fc.aa_a, aa_b=fc.aa_b, both_aa=fc.both_aa_mask(),
        ))
        all_t.append(fc.tau)
    all_t = np.concatenate(all_t)
    # Quantize tau to 0.01 to bound the unique-t cache
    # Geomspace tau quantization: bin into args.n_tau geometric bins between
    # min(0.01, t.min()) and max(t.max(), 2.74). Snap each cherry tau to the
    # nearest bin via log-distance. Geomspace concentrates resolution at small
    # t where exp(Q*t) varies fastest; n_tau=50 gives ~5-10% relative tau
    # precision and a 5x smaller log_P cache vs the linear-step variant.
    t_lo = max(0.005, float(all_t.min()))
    t_hi = max(2.74, float(all_t.max()))
    unique_t = np.geomspace(t_lo, t_hi, args.n_tau).astype(np.float64)
    log_unique_t = np.log(unique_t)
    log_all_t = np.log(np.clip(all_t, t_lo, t_hi))
    # nearest-in-log-space binning
    inv = np.argmin(np.abs(log_all_t[:, None] - log_unique_t[None, :]), axis=1)
    inv = inv.astype(np.int64)
    inv_t_dict = {}
    cursor = 0
    for fd in per_family_data:
        n = fd['n_cherries']
        inv_t_dict[fd['family']] = inv[cursor: cursor + n].astype(np.int64)
        cursor += n
    print(f"Unique tau (geomspace, n={len(unique_t)}): "
            f"[{unique_t[0]:.4f}, {unique_t[-1]:.3f}], "
            f"min/max ratio {unique_t[1] / unique_t[0]:.3f}")

    state = init_svi_state(per_family_data, K_c=args.K,
                              init_pair_fraction=args.init_pair_fraction,
                              K_H_max=args.K_H_max,
                              use_side_potentials=args.use_side_potentials,
                              rng=rng)

    # Anchor support: load PDB contacts for the requested families and
    # override their initial partition with those pairs. The Gibbs sweep
    # for anchor families uses fix_partition=True (only c_s resampled).
    anchor_set = set(f.strip() for f in args.anchor_families.split(",") if f.strip())
    is_anchor = [fc.family in anchor_set for fc in families]
    if anchor_set:
        from tkfdp.bio import PFAM_SEED_DIR, parse_stockholm
        from tkfdp.pdb_contacts import pdb_contacts_for_family
        from tkfdp.partition_K import init_from_pairs_K
        print(f"\nAnchor families ({sorted(anchor_set)}): loading PDB contacts ...")
        for fam_idx, fc in enumerate(families):
            if not is_anchor[fam_idx]: continue
            seqs = parse_stockholm(PFAM_SEED_DIR / f"{fc.family}.sto")
            info = pdb_contacts_for_family(fc.family, seqs,
                                             distance_threshold=8.0,
                                             min_separation=4)
            if info is None or not info.contact_pairs:
                print(f"  {fc.family}: no PDB contacts; treating as non-anchor")
                is_anchor[fam_idx] = False
                continue
            # PDB contacts use the unfiltered Stockholm column index. Our
            # pipeline at min_aa_fraction=0 also uses unfiltered indexing,
            # so contacts apply directly.
            pairs_in = [(int(i), int(j)) for (i, j) in info.contact_pairs
                          if 0 <= i < fc.L and 0 <= j < fc.L]
            print(f"  {fc.family}: PDB={info.pdb_id}, "
                  f"{len(pairs_in)} contacts as fixed partition")
            state.states_per_msa[fam_idx] = init_from_pairs_K(
                fc.family, fc.L, args.K, pairs_in, rng)

    # Restrict-anchor support: load PDB candidates (raw, NOT greedy-matched)
    # and constrain the Gibbs partner-proposal to that candidate set. Per
    # family we build an (L, L) bool allowed_partner_mask. The chain may
    # add/remove pairs freely subject to (a) Ewens size-{1,2} prior (one
    # pair per column) and (b) the candidate-set restriction.
    restrict_set = set(f.strip() for f in args.restrict_anchor_families.split(",")
                       if f.strip())
    is_restrict = [fc.family in restrict_set for fc in families]
    allowed_partner_masks: list[np.ndarray | None] = [None] * len(families)
    if restrict_set:
        from tkfdp.bio import PFAM_SEED_DIR, parse_stockholm
        from tkfdp.pdb_contacts import pdb_contacts_for_family
        from tkfdp.partition_K import init_from_pairs_K
        print(f"\nRestrict-anchor families ({len(restrict_set)}): loading PDB "
              "candidates ...")
        n_drop = 0
        for fam_idx, fc in enumerate(families):
            if not is_restrict[fam_idx]: continue
            seqs = parse_stockholm(PFAM_SEED_DIR / f"{fc.family}.sto")
            info = pdb_contacts_for_family(fc.family, seqs,
                                             distance_threshold=8.0,
                                             min_separation=4)
            if info is None or not info.candidate_pairs:
                print(f"  {fc.family}: no PDB candidates; treating as non-restrict")
                is_restrict[fam_idx] = False
                n_drop += 1
                continue
            cands = [(int(i), int(j)) for (i, j) in info.candidate_pairs
                       if 0 <= i < fc.L and 0 <= j < fc.L]
            mask = np.zeros((fc.L, fc.L), dtype=bool)
            for (i, j) in cands:
                mask[i, j] = True; mask[j, i] = True
            allowed_partner_masks[fam_idx] = mask
            # Warm-start the partition with the greedy matching of the same
            # candidate set (= what the fix run would have used). The chain
            # then explores from there.
            seed_pairs = [(int(i), int(j)) for (i, j) in info.contact_pairs
                            if 0 <= i < fc.L and 0 <= j < fc.L]
            print(f"  {fc.family}: PDB={info.pdb_id}, "
                  f"{len(cands)} candidate pairs, {len(seed_pairs)} greedy-init")
            state.states_per_msa[fam_idx] = init_from_pairs_K(
                fc.family, fc.L, args.K, seed_pairs, rng)
        print(f"Restrict-anchor: {sum(is_restrict)} families with allowed-pair masks; "
              f"{n_drop} fell through to non-anchor.")
    state.alpha_c = args.alpha_c
    state.a_eta = args.a_eta
    state.b_eta = args.b_eta
    state.kappa_pi = args.kappa_pi

    pi_bar = np.asarray(PI_LG08)
    S = np.asarray(S_LG08_F81)

    # Partial resume: load only the global params (pi_class + Potts DP)
    # from a checkpoint and graft them onto the fresh state. Per-family
    # latents stay freshly initialized (or anchor-pinned). The mismatched
    # family-list error from validate_resume does NOT apply here.
    if args.resume_globals_from is not None:
        from tkfdp.checkpoint import load_globals_from_checkpoint
        mu_p_for_grafting = np.zeros((20, 20))
        tau_p_for_grafting = np.full((20, 20), 4.0)
        pi_class_g, potts_dp_g, meta_g = load_globals_from_checkpoint(
            args.resume_globals_from, mu_p_for_grafting, tau_p_for_grafting,
        )
        if pi_class_g.shape[0] != args.K:
            raise SystemExit(
                f"--resume-globals-from K_c mismatch: "
                f"checkpoint K_c={pi_class_g.shape[0]} vs run --K {args.K}")
        state.pi_class = pi_class_g
        state.potts_dp = potts_dp_g
        state.alpha_H = float(meta_g.get("alpha_H", args.alpha_H))
        print(f"Resumed globals from {args.resume_globals_from}: "
              f"K_c={pi_class_g.shape[0]}, atoms={potts_dp_g.atoms.shape}, "
              f"alpha_H={state.alpha_H}")

    # Resume from checkpoint if requested. The checkpoint must come from a
    # run with matching --families and --K (else the SVIState shape mismatches).
    from tkfdp.checkpoint import (
        BEST_CHKPT_NAME, EarlyStoppingState, load_checkpoint, save_checkpoint,
        update_early_stopping, validate_resume,
    )
    es = EarlyStoppingState()
    it_start = 0
    trace_loaded = None
    if args.resume_from is not None:
        mu_p_for_resume = np.zeros((20, 20))
        tau_p_for_resume = np.full((20, 20), 4.0)
        loaded_state, trace_loaded, loaded_rng, es, meta = load_checkpoint(
            args.resume_from, per_family_data, mu_p_for_resume, tau_p_for_resume
        )
        validate_resume(meta, [fc.family for fc in families], args.K)
        state = loaded_state
        rng = loaded_rng
        it_start = int(meta["iter"])
        print(f"Resumed from {args.resume_from}: iter={it_start}, "
                f"best_val_LL={es.best_val_LL:.2f} (best_iter={es.best_iter})")

    # Validation families (held-out) for periodic val LL + early stopping.
    val_fcs = []
    if args.val_families:
        val_ids = [f.strip() for f in args.val_families.split(",") if f.strip()]
        print(f"\nVal families: {val_ids}")
        val_fcs = families_from_list(
            val_ids, min_cherries=args.min_cherries,
            min_aa_fraction=args.min_aa_fraction,
        )
    A = 20
    # Size-{1,2}-restricted Ewens partition prior. With singletons + pairs
    # only, Γ(1) = Γ(2) = 1 so block-size factors collapse and
    # P(π) ∝ alpha_z^{|π|}. Pair option in the Gibbs proposal has one fewer
    # block than the singleton alternative, so the per-pair log cost is
    # -log(alpha_z).
    log_pair_offset = -np.log(args.alpha_z)

    # Optional pre-SVI EM warm-up on column → site-class assignments.
    # No coupling, no partner moves — just settle the column partition +
    # pi_class via Dirichlet-multinomial EM before introducing H atoms.
    if args.em_warmup_iters > 0 and args.resume_from is None:
        from tkfdp.svi import em_warmup_site_classes
        print(f"\nEM warmup: {args.em_warmup_iters} iters of column-class "
                f"assignment (Dirichlet-multinomial likelihood, no coupling)")
        t_em0 = time.time()
        state = em_warmup_site_classes(
            state, per_family_data,
            kappa_pi=args.kappa_pi, pi_bar=pi_bar,
            n_iters=args.em_warmup_iters, rng=rng,
            tol=args.em_warmup_tol, n_seeds=args.em_warmup_seeds,
            verbose=True,
        )
        print(f"  EM warmup total: {time.time() - t_em0:.1f}s")

    print(f"\nSVI K_c={args.K}, n_outer={args.n_outer}, "
          f"init_pair_frac={args.init_pair_fraction}\n")

    if trace_loaded is None:
        trace = dict(elapsed=[], log_l=[], n_pairs=[], pi_diff=[], H_norm=[],
                       val_LL=[])
    else:
        trace = trace_loaded
        trace.setdefault("val_LL", [])
    t0_total = time.time()
    early_stop = False
    for it in range(it_start, args.n_outer):
        t0 = time.time()

        # 1. Build log_P caches for partition Gibbs.
        # log_P_cache: (K_c, K_c, n_t, 400, 400) — already per-(c1, c2)
        #   with the atom looked up via state.potts_dp.assignments[c1, c2]
        #   inside build_log_P_cache_K_atoms.
        # log_P_single_cache: (K_c, n_t, 20, 20) for per-class singleton
        #   evidence. Built ONCE per outer (depends only on pi_class +
        #   unique_t), reused across all families. Replaces the per-MSA
        #   per-class per-tau jsl.expm loop which alone was costing 200K
        #   uncached JIT calls per outer at 1000 families × K_c=4 × n_t=50.
        log_P_cache = build_log_P_cache_K_atoms(state, unique_t, S)
        import jax.numpy as jnp
        import jax.scipy.linalg as jsl

        @jax.jit
        def _build_log_P_single(pi_class, unique_t, S):
            K_c = pi_class.shape[0]
            S_off = S - jnp.diag(jnp.diag(S))
            # Q_c[a, a'] = S_off[a, a'] * pi_class[c, a']  for a != a'
            Q = S_off[None, :, :] * pi_class[:, None, :]    # (K_c, A, A)
            Q = Q.at[:, jnp.arange(20), jnp.arange(20)].set(0)
            row_sums = Q.sum(axis=-1)
            Q = Q.at[:, jnp.arange(20), jnp.arange(20)].set(-row_sums)
            # exp(Q * t) per (c, t) via vmap-of-vmap on jax.scipy.linalg.expm
            def per_t_per_c(Q_c, t):
                return jsl.expm(Q_c * t)
            P = jax.vmap(jax.vmap(per_t_per_c, in_axes=(None, 0)),
                          in_axes=(0, None))(Q, unique_t)
            return jnp.log(jnp.clip(P, 1e-300, 1.0))

        log_P_single_cache = np.asarray(_build_log_P_single(
            jnp.asarray(state.pi_class), jnp.asarray(unique_t), jnp.asarray(S)
        ))

        # 2. Joint partition + class Gibbs per MSA
        n_pairs_total = 0
        for fam_idx, fd in enumerate(per_family_data):
            st = state.states_per_msa[fam_idx]
            tau_idx = inv_t_dict[fd['family']]
            both_aa = fd['both_aa']

            def make_pair_fn(fd=fd, tau_idx=tau_idx, both_aa=both_aa, st=st):
                # Vectorized version: for column s, compute pair_lik[k_s, k_t, t]
                # for all t in one fancy-indexed gather over cherries.
                # Eliminates the cherry x K_c x K_c Python loop that dominated
                # the partition-Gibbs sweep cost (~80 s/outer at K_c=4, 50 fam).
                aa_a_full = fd['aa_a'].astype(np.int64)   # (C, L)
                aa_b_full = fd['aa_b'].astype(np.int64)
                ba = both_aa                              # (C, L) bool
                K_c = state.K_c

                def pair_fn(s):
                    L = fd['L']
                    out = np.zeros((K_c, K_c, L), dtype=np.float64)
                    valid_s_mask = ba[:, s]
                    if not valid_s_mask.any():
                        return out
                    valid_idx = np.flatnonzero(valid_s_mask)   # (M_v,) cherry indices
                    a_s_v = aa_a_full[valid_idx, s]            # (M_v,)
                    b_s_v = aa_b_full[valid_idx, s]
                    aa_a_v = aa_a_full[valid_idx, :]           # (M_v, L)
                    aa_b_v = aa_b_full[valid_idx, :]
                    ti_v = tau_idx[valid_idx]                  # (M_v,)
                    valid_t_mask = ba[valid_idx, :].astype(np.float64)  # (M_v, L)
                    # Clip gap AAs (encoded as 20) to a safe in-range value
                    # so the joint-state-index gather doesn't IndexError.
                    # The valid_t_mask zeroes out their contribution anyway.
                    aa_a_v_safe = np.minimum(aa_a_v, 19)
                    aa_b_v_safe = np.minimum(aa_b_v, 19)
                    # Joint state indices per (cherry, t):
                    start_idx = a_s_v[:, None] * 20 + aa_a_v_safe   # (M_v, L)
                    end_idx = b_s_v[:, None] * 20 + aa_b_v_safe
                    # Gather log_P_cache[k_s, k_t, ti_v[:, None], start_idx, end_idx]
                    # via numpy fancy indexing for each (k_s, k_t).
                    for k_s in range(K_c):
                        for k_t in range(K_c):
                            P = log_P_cache[k_s, k_t]          # (n_t, A^2, A^2)
                            log_p_v = P[ti_v[:, None], start_idx, end_idx]  # (M_v, L)
                            log_p_v = log_p_v * valid_t_mask
                            out[k_s, k_t, :] = log_p_v.sum(axis=0)
                    return out
                return pair_fn

            pair_fn = make_pair_fn()
            # Per-(column, class) singleton evidence: sum_c log P_single_class_c[a, b](τ_c).
            # Vectorized across all columns and cherries via gather from the
            # outer-shared log_P_single_cache (K_c, n_t, 20, 20).
            #
            # Replaces the per-MSA per-class per-tau Python loop that called
            # jsl.expm 200K times at 1000 families. New cost per family is
            # one numpy fancy-index gather sized (K_c, C, L).
            ba_f = both_aa.astype(np.float64)              # (C, L)
            tau_idx_arr = tau_idx                            # (C,)
            aa_a_arr = np.minimum(fd['aa_a'].astype(np.int64), 19)  # (C, L), gap clipped
            aa_b_arr = np.minimum(fd['aa_b'].astype(np.int64), 19)
            # log_P_single_cache: (K_c, n_t, 20, 20)
            # For each cherry c and column s: index into cache at
            # (k, tau_idx[c], aa_a[c, s], aa_b[c, s]).
            # Broadcast to (K_c, C, L), masked by both_aa, summed over C.
            tile_ti = np.broadcast_to(tau_idx_arr[:, None],
                                          aa_a_arr.shape)             # (C, L)
            # Per-class gather via stacking K_c slices.
            sll = np.zeros((fd['L'], state.K_c))
            for c_idx in range(state.K_c):
                # log_P_single_cache[c_idx, tile_ti, aa_a_arr, aa_b_arr] -> (C, L)
                gathered = log_P_single_cache[c_idx, tile_ti,
                                                  aa_a_arr, aa_b_arr]
                gathered = gathered * ba_f                              # mask
                sll[:, c_idx] = gathered.sum(axis=0)                    # sum over C

            fix_part = is_anchor[fam_idx]
            allowed_mask = allowed_partner_masks[fam_idx]
            for _ in range(args.n_gibbs_per_outer):
                gibbs_sweep_K(st, pair_fn, sll, rng,
                                temperature=1.0,
                                log_pair_prior_offset=log_pair_offset,
                                alpha_c=args.alpha_c,
                                fix_partition=fix_part,
                                allowed_partner_mask=allowed_mask)
            n_pairs_total += n_pairs_K(st)

        # 3. Aggregate per-class HR sufficient stats and update pi
        # (eta_s is INTEGRATED OUT via the closed-form NB marginal — we
        # don't track eta as a parameter; HR is computed at unit-rate Q.)
        K_c = state.K_c
        dwell_total = np.zeros((K_c, A))
        real_counts = np.zeros((K_c, A))
        for fam_idx, fd in enumerate(per_family_data):
            cls = state.states_per_msa[fam_idx].cls
            N_acc, dwell, _ = hr_per_class_per_msa(
                fd['aa_a'], fd['aa_b'], fd['tau'], fd['both_aa'],
                cls, K_c, state.pi_class, S
            )
            dwell_total += dwell
            real_counts += accumulate_real_counts(
                fd['aa_a'], fd['aa_b'], fd['both_aa'], cls, K_c, A
            )
        state.pi_class = update_pi_class(
            K_c, state.pi_class, dwell_total, real_counts,
            S, state.kappa_pi, pi_bar
        )

        # 5. Update each Potts atom via JIT-hoisted Laplace MAP
        mu_prior = np.zeros((A, A))
        tau_prior = np.full((A, A), 4.0)
        try:
            state = update_potts_atoms_jit(
                state, per_family_data, unique_t, inv_t_dict, S,
                mu_prior, tau_prior,
                n_steps=args.n_laplace_steps, lr=0.05,
                loss_kind=("elbo" if args.use_elbo else "exact"),
                h_prior_tau=args.h_prior_tau,
            )
        except Exception as e:
            print(f"  WARN: atom-Laplace update failed: {e}")

        # 6. Periodic Potts TSB resample over h_{c, c'} assignments + rho.
        # (Replaces the CRP-Gibbs variant; K_H_max = K_c(K_c+1)/2 atoms
        # are always allocated, assignments are resampled via Categorical
        # (rho * lik), and stick weights rho are conjugate-Beta-updated
        # from per-atom counts.)
        if (it + 1) % args.n_crp_every == 0:
            from tkfdp.svi import potts_tsb_sweep
            state = potts_tsb_sweep(
                state, per_family_data, unique_t, inv_t_dict, S, rng,
                loss_kind=("elbo" if args.use_elbo else "exact"),
            )

        K_H = state.potts_dp.atoms.shape[0]

        # Diagnostic: closed-form per-column NB marginal at the new state
        # (== rate-side log evidence for each column under its current class)
        nb_total = 0.0
        for fam_idx, fd in enumerate(per_family_data):
            cls = state.states_per_msa[fam_idx].cls
            log_marg = per_column_log_marginal_class_specific(
                fd['aa_a'], fd['aa_b'], fd['tau'], fd['both_aa'], cls,
                K_c, state.pi_class, S, state.a_eta, state.b_eta
            )
            nb_total += float(log_marg.sum())

        elapsed = time.time() - t0
        n_per_class = np.zeros(K_c, dtype=int)
        for st in state.states_per_msa:
            n_per_class += np.bincount(st.cls, minlength=K_c)
        pi_diff = np.linalg.norm(state.pi_class - pi_bar[None, :], axis=1)
        H_norm = np.linalg.norm(state.potts_dp.atoms[0])
        print(f"[outer {it+1:2d}/{args.n_outer}] "
              f"pairs={n_pairs_total}  K_H={K_H}  ||H_0||={H_norm:.2f}  "
              f"class_counts={n_per_class.tolist()}  pi_diff={[f'{d:.2f}' for d in pi_diff]}  "
              f"NB_total={nb_total:.1f}  ({elapsed:.1f}s)")

        trace['elapsed'].append(elapsed)
        trace['n_pairs'].append(int(n_pairs_total))
        trace['pi_diff'].append([float(d) for d in pi_diff])
        trace['H_norm'].append(float(H_norm))
        trace['log_l'].append(float(nb_total))

        # Periodic checkpoint dump
        if args.checkpoint_every > 0 and (it + 1) % args.checkpoint_every == 0:
            save_checkpoint(state, trace, rng, args.out_dir, it + 1, es)

        # Periodic val LL + early stopping
        if val_fcs and (it + 1) % args.val_every == 0:
            from tkfdp.val_loglik_v2 import val_log_likelihood
            t_val0 = time.time()
            val_score, _ = val_log_likelihood(
                state, val_fcs,
                n_burnin=args.val_burnin, n_samples=args.val_samples,
                seed=args.seed + it,
            )
            t_val = time.time() - t_val0
            trace['val_LL'].append([int(it + 1), float(val_score)])
            improved = val_score > es.best_val_LL
            update_early_stopping(es, val_score, it + 1)
            print(f"  val_LL={val_score:.2f}  best={es.best_val_LL:.2f} "
                    f"(iter {es.best_iter})  no-improvement={es.n_evals_since_improvement}/"
                    f"{args.patience if args.patience > 0 else 'inf'}  ({t_val:.1f}s)")
            # Compact param report — fired every val cycle so the trajectory
            # of pi_class top AAs and Potts top pairs is visible in the log.
            from tkfdp.inspect_params import format_params_summary
            print(format_params_summary(
                state.pi_class, state.potts_dp.atoms,
                state.potts_dp.assignments, top_aa=5, top_pairs=10,
                indent="    ",
            ))
            if improved:
                # Snapshot the best-so-far state to a separate dir; this
                # survives the rolling chkpt overwrite + early-stop.
                save_checkpoint(state, trace, rng, args.out_dir, it + 1, es,
                                  subdir=BEST_CHKPT_NAME)
            if args.patience > 0 and es.n_evals_since_improvement >= args.patience:
                print(f"\nEarly stop: {es.n_evals_since_improvement} consecutive "
                        f"val cycles without improvement (patience={args.patience}). "
                        f"Best val_LL={es.best_val_LL:.2f} at iter {es.best_iter} "
                        f"snapshotted in {args.out_dir / BEST_CHKPT_NAME}/")
                early_stop = True
                break

    if early_stop:
        # Force a final checkpoint dump on early stop so the saved chkpt
        # reflects the iter that triggered the stop, not the prior chkpt.
        save_checkpoint(state, trace, rng, args.out_dir, it + 1, es)
    total = time.time() - t0_total
    n_done = it + 1 - it_start if it_start <= args.n_outer else args.n_outer
    print(f"\nTotal: {total:.1f}s ({total/max(n_done, 1):.1f}s/outer "
            f"over {n_done} iters; started at it={it_start})")

    # Save artifacts
    np.save(args.out_dir / "pi_class.npy", state.pi_class)
    np.save(args.out_dir / "potts_atoms.npy", state.potts_dp.atoms)
    np.save(args.out_dir / "potts_assignments.npy", state.potts_dp.assignments)
    for fam_idx, fd in enumerate(per_family_data):
        np.save(args.out_dir / f"eta_{fd['family']}.npy", state.eta_per_msa[fam_idx])
        np.save(args.out_dir / f"cls_{fd['family']}.npy",
                state.states_per_msa[fam_idx].cls)
        np.save(args.out_dir / f"partner_{fd['family']}.npy",
                state.states_per_msa[fam_idx].partner)
    with open(args.out_dir / "trace.json", "w") as f:
        json.dump(trace, f, indent=2)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"\nArtifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
