"""Dynamic-latent-field variant trainer for Pfam corpora.

Mirrors `experiments/exp2_pfam_v2.py` at the orchestration level but
drives the dynfield variant: F81-on-DP field selector, per-(class, field)
stationary `pi_field`, soft-EM atom updates with the CRP-style
variable-size partition Gibbs. See `docs/dynfield_runbook.md` for the
API surface; this script is the operator-facing CLI driver.

Quick start (tiny smoke):

  python3 experiments/train_dynfield.py \\
      --processed-dir data/pfam_processed_top1000 \\
      --n-families 10 \\
      --out-dir results/dynfield_smoke \\
      --K-c 1 --L-max 4 --rho-chain 0.5 \\
      --n-outer-iters 5

For the runbook's "preflight warnings" convention, this script also
warns loudly when:
  - --val-families is missing (no val LL trajectory recorded).
  - --log-file is missing AND output is not a tty (probably tailed but
    not flushed; auto-flushes anyway).

Output layout (`--out-dir`):
  _chkpt/         rolling checkpoint (state.npz + meta.json + trace.json)
  _best_chkpt/    best-so-far snapshot (by val LL if val_families set,
                  else by training LL)
  train.log       full stdout (if --log-file)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src/ to sys.path so `from tkfdp...` works when invoked directly.
THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
sys.path.insert(0, str(REPO / "src"))

# Enable JAX on-disk compilation cache BEFORE any jax import triggers a
# compile. Sits under ~/.jax_cache by default (override with env var
# TKF_JAX_CACHE_DIR). Aggressive thresholds: cache everything that
# compiles, not just long-compile kernels. Cross-process, cross-session
# hit rate matters more here than local cache hygiene.
_JAX_CACHE_DIR = os.environ.get('TKF_JAX_CACHE_DIR',
                                    str(Path.home() / '.jax_cache'))
try:
    import jax as _jax_early
    _jax_early.config.update('jax_compilation_cache_dir', _JAX_CACHE_DIR)
    _jax_early.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
    _jax_early.config.update('jax_persistent_cache_min_compile_time_secs',
                                0.0)
    # NOTE: Do NOT set 'jax_persistent_cache_enable_xla_caches' = 'all'.
    # It forces XLA to route final CUBIN through nvlink for on-disk
    # serialisation, which on this box invokes system nvlink 12.2 while
    # JAX's internal ptxas targets 12.9 — the mismatch aborts every
    # non-trivial GPU compile. The other three cache flags are sufficient
    # for cross-process amortisation and don't touch nvlink.
    Path(_JAX_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[jax cache] enabled at {_JAX_CACHE_DIR}", flush=True)
except Exception as _e:
    print(f"[jax cache] setup skipped: {_e!r}", flush=True)

from tkfdp.svi import (init_svi_state_dynfield,
                          em_warmup_site_classes,
                          train_dynfield_full_iter,
                          class_gibbs_sweep_all_dynfield)
from tkfdp.checkpoint import (save_checkpoint, load_checkpoint,
                                 EarlyStoppingState, validate_resume)
from tkfdp.lg08 import PI_LG08


# ---------------------------------------------------------------------------
# Preflight warnings (per the project-wide convention).
# ---------------------------------------------------------------------------

def _uniform_tsb_betas(K: int) -> np.ndarray:
    """Return tsb_betas (K-1,) whose stick-breaking construction gives
    a uniform 1/K distribution over K atoms. Used when the field or
    archetype DP updates are disabled (--update-field-tsb=False and
    --update-arch-tsb=False, default) so that (rho, tsb_betas) and
    (rho_arch, tsb_betas_arch) stay in a consistent state that also
    round-trips through the checkpoint code path.

    Derivation: rho[i] = (∏_{j<i}(1 − β_j)) · β_i for i < K − 1, rho[K−1]
    = ∏_{j<K−1}(1 − β_j). Setting rho[i] = 1/K uniformly and solving
    recursively gives β_i = 1/(K − i)."""
    if K < 1:
        raise ValueError(f"K must be positive, got {K}")
    tsb = np.zeros(K - 1, dtype=np.float64)
    for i in range(K - 1):
        tsb[i] = 1.0 / float(K - i)
    return tsb


def _reset_field_tsb_uniform(state):
    L_max = int(state.dyn_field.L_max)
    state.dyn_field.rho = np.full(L_max, 1.0 / L_max, dtype=np.float64)
    state.dyn_field.tsb_betas = _uniform_tsb_betas(L_max)


def _reset_arch_tsb_uniform(state):
    K_a = int(state.dyn_field.K_a)
    state.dyn_field.rho_arch = np.full(K_a, 1.0 / K_a, dtype=np.float64)
    state.dyn_field.tsb_betas_arch = _uniform_tsb_betas(K_a)


def _preflight_warn(args):
    warnings = []
    if not args.val_families:
        warnings.append(
            "--val-families not set: no held-out validation LL will be "
            "recorded, so early stopping has no signal. The best-so-far "
            "checkpoint will track the best training LL instead.")
    if args.log_file is None and not sys.stdout.isatty():
        warnings.append(
            "stdout is not a tty and --log-file is unset. If you are "
            "tailing the run output, stdout will be auto-flushed each "
            "iter so progress is visible; persist the trace via "
            "--log-file <path> if you want a saved copy.")
    if args.K_c > 1 and args.em_warmup_iters == 0:
        warnings.append(
            f"K_c={args.K_c} > 1 but --em-warmup-iters=0: pi_class will "
            f"start at uniform LG08 across all K_c classes, so the soft "
            f"EM init has no class structure to break symmetry from. "
            f"Consider --em-warmup-iters >= 20.")
    if args.pi_arch_init == "c10" and int(args.n_archetypes) != 10:
        warnings.append(
            f"--pi-arch-init c10 requires --n-archetypes 10 (LG-C10 has "
            f"exactly 10 categories); got {args.n_archetypes}. Refusing "
            f"to launch.")
        sys.stderr.write("\n".join(warnings) + "\n")
        sys.stderr.flush()
        raise SystemExit(2)
    if args.freeze_pi_arch and args.pi_arch_init == "random":
        warnings.append(
            "--freeze-pi-arch with --pi-arch-init random will freeze the "
            "random-Dirichlet initial vocabulary forever. This is almost "
            "certainly not what you want. Combine with --pi-arch-init c10.")
    if args.freeze_pi_arch and args.split_merge_every > 0:
        warnings.append(
            "--freeze-pi-arch is incompatible with --split-merge-every > 0 "
            "(split-merge mutates pi_arch). Force-disabling split-merge.")
        args.split_merge_every = 0
    if warnings:
        sep = "\n  - "
        msg = "[PREFLIGHT WARNINGS]" + sep + sep.join(warnings)
        print(msg, flush=True)
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Args.
# ---------------------------------------------------------------------------

def _build_argparser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train the dynamic-latent-field variant on a Pfam corpus.")
    p.add_argument("--processed-dir", type=Path, default=None,
                   help="Preprocessed Pfam corpus (produced by "
                         "experiments/preprocess_pfam_topN.py).")
    p.add_argument("--pswm-corpus-dir", type=Path, default=None,
                   help="Preprocessed PSWM corpus (produced by "
                         "experiments/preprocess_pfam_pswm.py). Each family "
                         "is stored as per-branch PSWM pairs from LG08 "
                         "Felsenstein peeling. Hard aa_a/aa_b are MC-sampled "
                         "from PSWM at load time; the sampler is re-invoked "
                         "at the start of every SVI outer iter so E-step "
                         "and HR M-step see fresh samples across iters "
                         "(soft-observation posterior in expectation).")
    p.add_argument("--families", type=str, default=None,
                   help="Comma-separated list of Pfam family IDs "
                         "(alternative to --processed-dir).")
    p.add_argument("--n-families", type=int, default=None,
                   help="Take first N families from corpus (default: all).")
    p.add_argument("--min-cherries", type=int, default=2,
                   help="Drop families with fewer cherries than this.")
    p.add_argument("--val-families", type=str, default="",
                   help="Comma-separated Pfam families to hold out for val LL.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output dir for checkpoint + log.")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Optional log file (tee stdout to this).")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Resume from a dynfield checkpoint.")
    p.add_argument("--seed", type=int, default=0)

    # Model.
    p.add_argument("--K-c", type=int, default=4,
                   help="Number of site classes.")
    p.add_argument("--L-max", type=int, default=8,
                   help="DP truncation cap for the field selector.")
    p.add_argument("--rho-chain", type=float, default=1.0,
                   help="F81-on-DP rate multiplier; 0 -> per-class GTR.")
    p.add_argument("--alpha-field", type=float, default=1.0,
                   help="DP concentration on the field selector.")
    p.add_argument("--alpha-z", type=float, default=1.0,
                   help="Ewens concentration on cluster partitions.")
    p.add_argument("--alpha-c", type=float, default=1.0,
                   help="Symmetric Dirichlet-Multinomial concentration on "
                         "site-class assignments.")
    p.add_argument("--alpha-prior", type=float, default=0.5,
                   help="Dirichlet base measure concentration on pi_arch "
                         "(used in the Newton MAP update). Default 0.5 is a "
                         "sparse simplex-corner prior encouraging peaky "
                         "archetype distributions; the M-step clamps pi at "
                         "1e-30 to avoid subnormal float64. Use alpha_prior "
                         ">= 1 (e.g. 2) for Laplace-style smoothing.")
    p.add_argument("--lg08-is-temp", type=float, default=1.0,
                   help="Tempering exponent on the LG08 importance-sample "
                         "weight w' = w^beta (par:arch-lg08-is). "
                         "1.0 (default) is proper IS; smaller values (0.1-0.5) "
                         "trade correction bias for reduced weight variance "
                         "when ESS/K is low; 0.0 recovers unweighted training.")
    p.add_argument("--batch-size", type=int, default=0,
                   help="Mini-batch size for the tied-θ E-step + M-step. "
                         "0 (default) = full batch (all --n-families each "
                         "iter). Set >0 to enable Robbins-Monro mini-batch "
                         "SVI: each outer iter samples a subset of "
                         "batch_size families, runs the θ-MCMC + HR SS on "
                         "them only, and damps the pi_arch/ρ_chain M-step "
                         "with α_t = --m-step-alpha-init · (t+1)^"
                         "(−--m-step-alpha-decay). N_theta_sum and T_sum "
                         "are scaled by n_train/batch_size for unbiased "
                         "corpus-scale ρ_chain estimation.")
    p.add_argument("--m-step-alpha-init", type=float, default=1.0,
                   help="Robbins-Monro damping step size at iter 0 for the "
                         "mini-batch SVI M-step. α_t = α_init · "
                         "(t+1)^(−α_decay). Default 1.0 (undamped at "
                         "iter 0). Only used when --batch-size > 0.")
    p.add_argument("--m-step-alpha-decay", type=float, default=0.5,
                   help="Robbins-Monro exponent. Classical choice is 0.5 "
                         "(α_t = 1/√(t+1)). Only used when --batch-size > 0.")
    p.add_argument("--theta-mcmc-n-draws", type=int, default=1,
                   help="Number of independent (X, θ, M) draws to "
                         "accumulate SS over per outer iter under "
                         "--sample-theta-mcmc. Default 1 (one draw per "
                         "iter). Multiple draws reduce Monte-Carlo "
                         "variance of the M-step SS — approaching the "
                         "posterior-expectation SS that standard soft-EM "
                         "would use in the limit K → ∞. Empirically, "
                         "K=1 (default previously) drives archetype delta "
                         "collapse in low-signal regimes (e.g. cap-1) "
                         "because a single hard-sample chain (X → θ → "
                         "arch_assignment) attributes each observation "
                         "to exactly one archetype with no counter-"
                         "weighting from posterior mass on other atoms. "
                         "K > 1 averages over the hard-sample noise.")
    p.add_argument("--sample-theta-mcmc", action="store_true", default=False,
                   help="Enable tied-theta MCMC (par:arch-lg08-is): "
                         "per (family, cluster) sample theta_V at every "
                         "internal node V of the tree via Felsenstein-on-theta "
                         "conditional on the sampled residues X, plus a binary "
                         "M_v jump indicator per branch for rho_chain "
                         "identifiability. HR SS is computed on the "
                         "fully-observed (X, theta, M) trajectory rather than "
                         "marginalising over theta at each cherry. Requires "
                         "--pswm-corpus-dir (CLV path).")
    p.add_argument("--max-cluster-size", type=int, default=16,
                   help="CRP truncation cap.")
    p.add_argument("--rho-chain-mh-steps", type=int, default=0,
                   help="Per outer iter, run this many MH steps to "
                         "update rho_chain under a Gamma prior. 0 (default) "
                         "keeps rho_chain fixed.")
    p.add_argument("--rho-chain-prior-a", type=float, default=1.5,
                   help="Gamma prior shape on rho_chain "
                         "(prior mean = a/b).")
    p.add_argument("--rho-chain-prior-b", type=float, default=5.0,
                   help="Gamma prior rate on rho_chain "
                         "(default 5.0; prior mean 0.3 with default a=1.5).")
    p.add_argument("--rho-chain-step-size", type=float, default=0.3,
                   help="Std-dev of the log-space MH proposal on rho_chain.")
    p.add_argument("--alpha-z-mh-steps", type=int, default=0,
                   help="Per outer iter, run this many MH steps to "
                         "estimate alpha_z under a Gamma prior. 0 (default) "
                         "keeps alpha_z fixed at --alpha-z.")
    p.add_argument("--alpha-z-prior-a", type=float, default=100.0,
                   help="Gamma prior shape on alpha_z (mean a/b). "
                         "Default a=100.0 encodes a strong prior preference "
                         "for many small clusters -- the design intent of "
                         "the CRP prior over column partitions in this "
                         "model. Empirically the data pulls alpha_z up "
                         "toward 20 even under a weak Gamma(3, 1) prior, "
                         "so the default is set high to actively encourage "
                         "the intended regime.")
    p.add_argument("--alpha-z-prior-b", type=float, default=1.0,
                   help="Gamma prior rate on alpha_z "
                         "(default 1.0; mean 100.0 with a=100.0, std=10). "
                         "Prior mass roughly on [80, 120]; MH still lets "
                         "the data move alpha_z within a sensible range.")
    p.add_argument("--alpha-z-step-size", type=float, default=0.5,
                   help="Std-dev of the log-space MH proposal on alpha_z.")
    # Archetype hyperparameters (archetype vocabulary is mandatory).
    p.add_argument("--n-archetypes", type=int, default=16,
                   help="Number of archetype slots K_a (TSB truncation "
                         "of the archetype DP). pi_field[c, theta, :] = "
                         "pi_archetype[arch_assignment[c, theta], :] with "
                         "pi_archetype (K_a, A) a shared vocabulary.")
    p.add_argument("--alpha-arch", type=float, default=1.0,
                   help="DP concentration on the archetype vocabulary. "
                         "Low (0.1-1.0) concentrates on few distinct "
                         "archetypes.")
    p.add_argument("--pi-arch-init", type=str, default="random",
                   choices=("random", "c10"),
                   help="Initial pi_archetype vocabulary. 'random' "
                         "(default) samples from the class-marginal "
                         "mixture after EM warmup. 'c10' loads the "
                         "LG-C10 profiles (biologically-anchored 10-"
                         "category vocabulary with 2 acidic + 2 basic "
                         "categories). Requires --n-archetypes 10. "
                         "Combined with --freeze-pi-arch: model learns "
                         "arch_assignment only; vocabulary is fixed and "
                         "eigendecomps/expm caches are precomputable, "
                         "which also removes the archetype-collapse "
                         "pathology observed with random init.")
    p.add_argument("--freeze-pi-arch", action="store_true", default=False,
                   help="Skip the Newton HR M-step for pi_archetype. "
                         "arch_assignment, rho_chain, and other params "
                         "still update. Intended pairing: "
                         "--pi-arch-init c10 for a fixed LG-C10 "
                         "vocabulary.")
    p.add_argument("--diagonal-arch-init", action="store_true", default=False,
                   help="Initialise arch_assignment[c, theta] = c for "
                         "all theta and c in range(min(K_c, K_a)). "
                         "Under --pi-arch-init c10 with K_c=K_a, this "
                         "makes pi_class[c] = pi_arch[c] = LG-C10 "
                         "category c exactly at init (pure C10 "
                         "mixture, no-coevolution baseline). Training "
                         "moves off-diagonal to discover coevolution "
                         "signal. Skips the EM warmup redraw of "
                         "arch_assignment. Intended pairing: "
                         "--pi-arch-init c10 --freeze-pi-arch "
                         "--em-warmup-iters 0.")
    p.add_argument("--arch-update-mode", type=str, default="none",
                   choices=("none", "argmax", "swap", "gibbs"),
                   help="How to update arch_assignment[c, theta] each "
                         "outer iter under --sample-theta-mcmc. "
                         "'none' (default): don't update — arch_"
                         "assignment stays at whatever init set it to "
                         "(useful as static-mixture baseline under "
                         "--diagonal-arch-init). 'argmax': pointwise "
                         "argmax of the soft posterior over K_a "
                         "categories per (c, theta), matching the "
                         "composite-path update in svi.py. 'swap': "
                         "permutation-only MH move — at each (theta, "
                         "c1<c2) pair per sweep, propose swapping "
                         "arch_assignment[c1, theta] <-> arch_"
                         "assignment[c2, theta], accept via posterior "
                         "ratio. Preserves the multiset of archetypes "
                         "used at each theta (identity of usage never "
                         "changes); no longer a DP, effectively a "
                         "K_c! permutation search per theta. 'gibbs': "
                         "categorical Gibbs sample of arch_assignment"
                         "[c, theta] from the multinomial-only full "
                         "conditional prop rho_arch[k] · prod_a "
                         "pi_arch[k, a]^{V[c, theta, a]}. Uses the "
                         "sampled HR V tensor from the tied-θ trajectory "
                         "as sufficient stats. Under --fix-theta0-"
                         "diagonal, theta=0 is reset to c=k identity "
                         "after each Gibbs (or swap/argmax) update.")
    p.add_argument("--fix-theta0-diagonal", action='store_true', default=False,
                   help="After every arch_assignment update, reset "
                         "arch_assignment[c, 0] = c for c < min(K_c, K_a). "
                         "Keeps the modal field (theta=0) locked at the "
                         "LG-C10 identity mapping. Recommended when "
                         "--diagonal-arch-init is set and arch_update_mode "
                         "!= 'none'.")
    p.add_argument("--arch-swap-sweeps", type=int, default=1,
                   help="Number of full swap sweeps per outer iter "
                         "under --arch-update-mode swap. Each sweep "
                         "proposes every (theta, c1<c2) pair in "
                         "random order. Cost is O(sweeps · L_max · "
                         "K_c^2) which is tiny.")
    p.add_argument("--update-field-tsb", action='store_true', default=False,
                   help="Update the field DP's (rho, tsb_betas) each iter "
                         "from the sampled θ trajectory. Default False: "
                         "rho fixed at uniform 1/L_max, tsb_betas set to "
                         "the corresponding stick-breaking values. "
                         "Rationale: at finite L_max the TSB(α~1) prior "
                         "biases rho toward geometric decay, and combined "
                         "with the tied-θ MCMC's hard θ samples (low-"
                         "variance r stat at corpus scale), the MAP or "
                         "Beta-posterior TSB update creates a positive-"
                         "feedback collapse toward a single dominant θ "
                         "atom (documented in "
                         "results/tied_theta_K8_n10_2026-07-03_full_mstep_"
                         "map_collapsed). Uniform is the symmetric-"
                         "Dirichlet limit of the truncated TSB and avoids "
                         "the collapse.")
    p.add_argument("--update-arch-tsb", action='store_true', default=False,
                   help="Update the archetype DP's (rho_arch, "
                         "tsb_betas_arch) each iter from arch_assignment "
                         "slot counts, as done inside split_merge_step. "
                         "Default False: rho_arch fixed at uniform 1/K_a. "
                         "Split-merge still runs and detects dead "
                         "archetypes via low pi_arch entropy (the second "
                         "dead-detector) alone; the weight-based dead "
                         "detector (rho_arch < 0.01) never fires under "
                         "the uniform override. Same rationale as "
                         "--update-field-tsb.")
    p.add_argument("--use-hr-mstep", action='store_true', default=True,
                   help="Use the Holmes-Rubin closed-form M-step for "
                         "pi_archetype (GTR fixed-point) and rho_chain "
                         "(Gamma posterior). Default True.")
    p.add_argument("--K-rate-bins", type=int, default=0,
                   help="Enable Yang +Gamma+I rate heterogeneity on the "
                         "field-flip rate with this many Gamma quantile "
                         "bins (0 disables). Standard choice is 4 "
                         "(Yang 1996). Requires --use-hr-mstep.")
    p.add_argument("--alpha-gamma", type=float, default=1.0,
                   help="Gamma(alpha, alpha) shape parameter for the "
                         "per-cluster rate multiplier. Mean rate = 1, "
                         "variance = 1/alpha. Fixed at init in this "
                         "implementation.")
    p.add_argument("--p-inv-init", type=float, default=0.2,
                   help="Initial fraction of clusters in the invariant "
                         "(zero-rate) bin. Learned via Beta-conjugate "
                         "posterior each iter.")

    # Per-site +Gamma+I (par:arch-gamma-plus-I-persite in appendix).
    p.add_argument("--K-rate-bins-site", type=int, default=0,
                   help="Enable Yang +Gamma+I rate heterogeneity on the "
                         "per-SITE substitution rate with this many Gamma "
                         "quantile bins (0 disables). Independent from "
                         "--K-rate-bins (per-cluster field-flip). Standard "
                         "choice is 4.")
    p.add_argument("--alpha-gamma-site", type=float, default=1.0,
                   help="Shape parameter of the Gamma(alpha, alpha) prior "
                         "on the per-site rate multiplier.")
    p.add_argument("--p-inv-site-init", type=float, default=0.2,
                   help="Initial fraction of sites in the invariant bin.")
    p.add_argument("--site-resample-every", type=int, default=0,
                   help="Resample per-site bins via Rao-Blackwellised "
                         "Gibbs every N outer iters. 0 (default) = no "
                         "resample, bins fixed at prior draw (RAxML-style "
                         "empirical Bayes; see par:arch-gamma-plus-I-persite).")

    # Training-mode dispatch (see par:arch-phylo-elbo and
    # src/tkfdp/coupling/dynfield/backends.py).
    p.add_argument("--training-mode", type=str, default="composite",
                   choices=("composite", "phylo-elbo"),
                   help="Selects the E-step backend. 'composite' is the "
                         "existing cherry-pair regime (default). "
                         "'phylo-elbo' would use variational message-"
                         "passing on a tree per par:arch-phylo-elbo; "
                         "backend class exists as a documented stub in "
                         "backends.py but the tree preprocessing + "
                         "message-passing implementation is NOT YET "
                         "LANDED. Passing this mode raises "
                         "NotImplementedError at the first E-step call. "
                         "See task #109 for progress.")

    # Split-merge on the archetype block (see par:arch-split-merge).
    p.add_argument("--split-merge-every", type=int, default=0,
                   help="Apply hill-climbing split-merge moves on the "
                         "archetype block every N outer iters. 0 (default) = "
                         "off. Recommended: 5-10 for K_a >= 8. See "
                         "par:arch-split-merge.")
    p.add_argument("--split-merge-dead-threshold", type=float, default=0.01,
                   help="Archetypes with rho_arch below this are considered "
                         "dead and eligible to receive one arm of a split.")
    p.add_argument("--split-merge-tvd-threshold", type=float, default=0.05,
                   help="Archetype pairs with TVD(pi_arch[k1], pi_arch[k2]) "
                         "below this are eligible to be merged.")
    p.add_argument("--split-merge-max-moves", type=int, default=2,
                   help="Max split+merge moves accepted per invocation.")

    # Loop.
    p.add_argument("--n-outer-iters", type=int, default=20,
                   help="Number of outer iterations.")
    p.add_argument("--em-warmup-iters", type=int, default=50,
                   help="Number of soft EM warmup iters on pi_class "
                         "before the main SVI loop.")
    p.add_argument("--em-warmup-tol", type=float, default=1e-5,
                   help="EM warmup convergence tol on max-class L1(delta pi).")
    p.add_argument("--n-cluster-sweeps-per-outer", type=int, default=1,
                   help="Number of CRP cluster Gibbs sweeps per outer iter.")
    p.add_argument("--n-class-sweeps-per-outer", type=int, default=1,
                   help="Number of class-label Gibbs sweeps per outer iter.")
    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="Save rolling checkpoint every N outer iters.")
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stop after this many non-improving iters. "
                         "0 (default) disables early stopping. Uses "
                         "log_lik_after_atom as the improvement metric.")
    return p


# ---------------------------------------------------------------------------
# Data loading helpers.
# ---------------------------------------------------------------------------

def _load_per_family_data(args, family_filter=None):
    if getattr(args, 'pswm_corpus_dir', None) is not None:
        # CLV corpus: par:arch-lg08-is. Each family holds bottom-up
        # CLVs from LG08 peeling; the outer loop draws a fresh joint
        # tree history per iter (see _resample_clv_history below).
        # Initial aa_a / aa_b are filled from the first draw so
        # downstream shape probes see valid arrays.
        from tkfdp.pfam_data import families_from_clv_dir
        clv_families = families_from_clv_dir(
            args.pswm_corpus_dir, family_ids=None,
            min_columns=30, max_columns=200)
        if args.n_families is not None:
            clv_families = clv_families[: args.n_families]
        if family_filter is not None:
            clv_families = [fc for fc in clv_families
                              if fc.family in family_filter]
        init_rng = np.random.default_rng(args.seed ^ 0xC0FFEE)
        out = []
        for fc in clv_families:
            X = fc.sample_history(init_rng)
            fc_hard = fc.extract_branch_cherries(X)
            fd_entry = dict(
                family=fc.family, L=fc.L, n_cherries=fc_hard.n_cherries,
                tau=np.asarray(fc_hard.tau, dtype=np.float64),
                aa_a=fc_hard.aa_a, aa_b=fc_hard.aa_b,
                both_aa=fc_hard.both_aa_mask(),
                K=max(1, fc.L // 4),
                _clv=fc,                               # keep FamilyCLV for resample
            )
            out.append(fd_entry)
        return out

    if args.processed_dir is not None:
        from tkfdp.pfam_data_fast import families_from_processed
        families = families_from_processed(
            args.processed_dir, n_families=args.n_families,
            min_cherries=args.min_cherries)
    else:
        from tkfdp.pfam_data import families_from_list
        ids = [f.strip() for f in args.families.split(",") if f.strip()]
        families = families_from_list(ids, min_cherries=args.min_cherries)
    if family_filter is not None:
        families = [fc for fc in families if fc.family in family_filter]
    out = []
    for fc in families:
        fd_entry = dict(
            family=fc.family, L=fc.L, n_cherries=fc.n_cherries,
            tau=np.asarray(fc.tau, dtype=np.float64),
            aa_a=np.asarray(fc.aa_a), aa_b=np.asarray(fc.aa_b),
            both_aa=fc.both_aa_mask(),
            K=max(1, fc.L // 4),
        )
        out.append(fd_entry)
    return out


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    args = _build_argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout to log file if requested.
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log_file, "w", buffering=1)
        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, s):
                for st in self.streams: st.write(s); st.flush()
            def flush(self):
                for st in self.streams: st.flush()
        sys.stdout = _Tee(sys.__stdout__, log_f)

    _preflight_warn(args)

    # Validate the E-step backend selection up front so a phylo-elbo
    # request fails fast (before corpus load + JIT compile) with the
    # useful "not yet implemented" message from backends.py rather than
    # much deeper into the training loop.
    from tkfdp.coupling.dynfield.backends import make_backend
    _backend = make_backend(args.training_mode)
    print(f"[dynfield trainer] backend: {_backend.name}", flush=True)
    if args.training_mode == "phylo-elbo":
        # Trigger the NotImplementedError with the useful docstring
        # message; do not proceed to corpus load.
        _backend.soft_stats(None, [])

    print(f"\n[dynfield trainer] args: {vars(args)}", flush=True)
    rng = np.random.default_rng(args.seed)

    val_fams = set(f.strip() for f in args.val_families.split(",") if f.strip())
    print(f"Loading corpus from {args.processed_dir or '(family list)'} ...",
            flush=True)
    per_family_data = _load_per_family_data(
        args, family_filter=None if not val_fams else lambda fc: True)
    # Split train / val.
    train_per_fam = [fd for fd in per_family_data
                       if fd['family'] not in val_fams]
    val_per_fam = [fd for fd in per_family_data
                     if fd['family'] in val_fams]
    print(f"  Train: {len(train_per_fam)} families, "
            f"{sum(fd['n_cherries'] for fd in train_per_fam)} cherries.")
    if val_per_fam:
        print(f"  Val:   {len(val_per_fam)} families, "
                f"{sum(fd['n_cherries'] for fd in val_per_fam)} cherries.")

    # Initialise (or resume) dynfield state.
    A_dummy = np.zeros((20, 20))   # checkpoint API takes Potts mu/tau priors
    resume_meta = None
    # Auto-resume: if --resume-from is not given but the out-dir already
    # has a rolling checkpoint from a prior interrupted run, resume from
    # it. Explicit --resume-from still wins.
    if args.resume_from is None:
        auto_chkpt = args.out_dir / "_chkpt"
        if (auto_chkpt / "state.npz").exists() \
                and (auto_chkpt / "meta.json").exists():
            args.resume_from = auto_chkpt
            print(f"[auto-resume] found rolling checkpoint at "
                    f"{auto_chkpt} — resuming there. Pass "
                    f"--resume-from '' or delete the dir to start fresh.",
                    flush=True)
    if args.resume_from is not None:
        print(f"\nResuming from {args.resume_from} ...", flush=True)
        state, trace_resumed, rng, es, meta = load_checkpoint(
            args.resume_from, train_per_fam,
            mu_prior=A_dummy, tau_prior=A_dummy)
        validate_resume(meta,
                          [fd['family'] for fd in train_per_fam],
                          args.K_c)
        start_iter = int(meta.get('iter', 0))
        print(f"  Resumed at iter {start_iter}, "
                f"variant={meta.get('coupling_variant', 'potts')}, "
                f"rho_chain={state.dyn_field.rho_chain:.3f} "
                f"(was {args.rho_chain:.3f} from CLI; CLI ignored on resume)",
                flush=True)
        if state.coupling_variant != 'dynamic_field':
            raise ValueError(
                f"Resume checkpoint variant is {state.coupling_variant!r}, "
                f"not 'dynamic_field'. Use exp2_pfam_v2.py for Potts.")
        trace = trace_resumed
        resume_meta = meta
        # Skip EM warmup on resume; assume state is past it.
    else:
        state = init_svi_state_dynfield(
            train_per_fam, K_c=args.K_c, A=20,
            L_max=args.L_max, alpha_field=args.alpha_field,
            rho_chain=args.rho_chain,
            K_a=int(args.n_archetypes),
            alpha_arch=float(args.alpha_arch),
            alpha_prior=args.alpha_prior,
            rng=rng)
        # Force all-singleton init when max-cluster-size is 1: the
        # default init_random_K uses n_pairs=L//4 which contradicts a
        # strict cap-1 experiment. Reset partner arrays to -1 (all
        # unpaired) and invalidate cluster_id so it re-derives to
        # per-column singletons.
        if args.max_cluster_size == 1:
            for st_fam in state.states_per_msa:
                st_fam.partner = -np.ones(int(st_fam.L), dtype=np.int32)
                st_fam.cluster_id = None
            print("[cap-1] forced all-singleton init (partner=−1 for "
                    "every column)", flush=True)
        start_iter = 0
        trace = {'outer_iter': [], 'log_lik_total': [],
                  'n_clusters_total': [], 'mean_cluster_size': [],
                  'max_cluster_size': [], 'wall_secs': [],
                  'rho_chain': []}
        es = EarlyStoppingState()
        print(f"[archetypes] init: K_a={args.n_archetypes}, "
                f"alpha_arch={args.alpha_arch:.3f}", flush=True)
        if args.pi_arch_init == "c10":
            sys.path.insert(
                0, str(Path.home() / "tkf-mixdom" / "python"))
            from tkfmixdom.jax.core.site_class_profiles import (
                le_gascuel_c10)
            _c10_profiles, _c10_weights, _ = le_gascuel_c10()
            _c10_profiles = np.asarray(_c10_profiles, dtype=np.float64)
            _c10_profiles = _c10_profiles / _c10_profiles.sum(
                axis=1, keepdims=True)
            state.dyn_field.pi_archetype = _c10_profiles.copy()
            print(f"[archetypes] pi_arch initialised from LG-C10 "
                    f"(10 categories, alphabetical order). Freeze: "
                    f"{bool(args.freeze_pi_arch)}", flush=True)
        if args.diagonal_arch_init:
            _K_a = int(state.dyn_field.K_a)
            _K_c = int(args.K_c)
            _L_max = int(state.dyn_field.L_max)
            _aa = rng.integers(
                0, _K_a, size=(_K_c, _L_max)).astype(np.int32)
            _n_diag = min(_K_c, _K_a)
            _aa[:_n_diag, :] = np.arange(
                _n_diag, dtype=np.int32)[:, None]
            state.dyn_field.arch_assignment = _aa
            state.dyn_field.materialise_pi_field()
            state.dyn_field.pi_class = np.einsum(
                't,cta->ca',
                state.dyn_field.rho, state.dyn_field.pi_field)
            state.pi_class = state.dyn_field.pi_class
            print(f"[archetypes] arch_assignment: fully diagonal "
                    f"(class N→archetype N for N<{_n_diag} at every θ). "
                    f"pi_class[c] = pi_arch[c] at init.", flush=True)
        # +Gamma+I rate heterogeneity init (Yang 1996 style).
        if args.K_rate_bins > 0:
            state.dyn_field.K_rate_bins = int(args.K_rate_bins)
            state.dyn_field.alpha_gamma = float(args.alpha_gamma)
            state.dyn_field.p_inv = float(args.p_inv_init)
            from tkfdp.coupling.dynfield.hr_jax import gamma_quantile_means
            _m = gamma_quantile_means(args.alpha_gamma, args.K_rate_bins)
            print(f"[+Γ+I] init: K_r={args.K_rate_bins}, "
                    f"alpha_gamma={args.alpha_gamma:.3f}, "
                    f"p_inv={args.p_inv_init:.3f}, "
                    f"bin means={_m.round(3).tolist()}", flush=True)
        # Per-SITE +Gamma+I on the substitution rate
        # (par:arch-gamma-plus-I-persite).
        if args.K_rate_bins_site > 0:
            state.dyn_field.K_rate_bins_site = int(args.K_rate_bins_site)
            state.dyn_field.alpha_gamma_site = float(args.alpha_gamma_site)
            state.dyn_field.p_inv_site = float(args.p_inv_site_init)
            from tkfdp.coupling.dynfield.hr_jax import gamma_quantile_means
            _ms = gamma_quantile_means(args.alpha_gamma_site,
                                            args.K_rate_bins_site)
            print(f"[+Γ+I persite] init: K_r_site={args.K_rate_bins_site}, "
                    f"alpha_gamma_site={args.alpha_gamma_site:.3f}, "
                    f"p_inv_site={args.p_inv_site_init:.3f}, "
                    f"bin means={_ms.round(3).tolist()}, "
                    f"resample_every={args.site_resample_every}",
                    flush=True)
            from tkfdp.svi import init_site_rate_bins
            init_site_rate_bins(state, rng,
                                     per_family_data=train_per_fam)

    if args.resume_from is None and args.em_warmup_iters > 0:
        print(f"\nEM warmup: {args.em_warmup_iters} iters of column-class "
                f"assignment soft EM ...", flush=True)
        state = em_warmup_site_classes(
            state, train_per_fam,
            kappa_pi=1.0, pi_bar=np.asarray(PI_LG08),
            n_iters=args.em_warmup_iters, rng=rng,
            tol=args.em_warmup_tol, verbose=False)
        # Re-seed archetypes from the learned per-class marginals so that
        # the first outer iter sees a data-consistent starting point.
        # Each archetype is sampled from a mixture over the K_c class
        # marginals; arch_assignment is redrawn uniformly over K_a.
        # When pi_arch_init == 'c10' the vocabulary was already set to
        # LG-C10 above; skip the reseed (only redraw arch_assignment).
        K_a = state.dyn_field.K_a
        if args.pi_arch_init != "c10":
            kappa = 50.0
            pi_arch_new = np.zeros((K_a, state.A), dtype=np.float64)
            for k in range(K_a):
                c_pick = int(rng.integers(args.K_c))
                pi_arch_new[k] = rng.dirichlet(
                    kappa * state.pi_class[c_pick])
            state.dyn_field.pi_archetype = pi_arch_new
        if not args.diagonal_arch_init:
            state.dyn_field.arch_assignment = rng.integers(
                0, K_a, size=(args.K_c, args.L_max)).astype(np.int32)
        state.dyn_field.materialise_pi_field()
        state.dyn_field.pi_class = np.einsum(
            't,cta->ca', state.dyn_field.rho, state.dyn_field.pi_field)
        state.pi_class = state.dyn_field.pi_class
        if args.pi_arch_init == "c10":
            _aa_msg = ("preserved (diagonal at θ=0)"
                          if args.diagonal_arch_init
                          else "redrawn uniformly")
            print(f"  pi_class learned; pi_arch kept at LG-C10; "
                    f"arch_assignment {_aa_msg}.", flush=True)
        else:
            print(f"  pi_class learned; archetypes re-seeded from "
                    f"marginals.", flush=True)

    # DP TSB freezes (default: both frozen at uniform to avoid the
    # tied-θ MCMC positive-feedback collapse — see --update-field-tsb
    # and --update-arch-tsb helptext).
    if not args.update_field_tsb:
        _reset_field_tsb_uniform(state)
        state.dyn_field.materialise_pi_field()
        state.dyn_field.pi_class = np.einsum(
            't,cta->ca', state.dyn_field.rho, state.dyn_field.pi_field)
        state.pi_class = state.dyn_field.pi_class
        print(f"[dp] field TSB frozen at uniform rho=1/L_max="
                f"{1.0/state.dyn_field.L_max:.3f}", flush=True)
    if not args.update_arch_tsb:
        _reset_arch_tsb_uniform(state)
        print(f"[dp] archetype TSB frozen at uniform rho_arch=1/K_a="
                f"{1.0/state.dyn_field.K_a:.3f}", flush=True)

    # Outer loop.
    # alpha_z may be MH-updated each iter; track it locally so resume can
    # pick up the latest value via extra_meta.
    alpha_z_state = float(args.alpha_z if resume_meta is None
                            else resume_meta.get('alpha_z', args.alpha_z))
    print(f"\nDynfield SVI: {args.n_outer_iters} outer iters "
            f"(starting at iter {start_iter}; alpha_z={alpha_z_state:.3f})",
            flush=True)
    # CLV/LG08 importance-sampling training (par:arch-lg08-is): if any
    # family carries a FamilyCLV, draw a fresh joint tree history under
    # LG08 at the start of every outer, extract per-branch cherries,
    # and refresh aa_a/aa_b in-place. This preserves branch-length
    # correlations that the earlier marginal-PSWM sampler discarded.
    _clv_rng = np.random.default_rng(args.seed ^ 0xBAD54FE)
    _batch_rng = np.random.default_rng(args.seed ^ 0xC0DEFACE)

    def _sample_batch_ids(n_train, batch_size, rng):
        if batch_size <= 0 or batch_size >= n_train:
            return list(range(n_train))
        return list(rng.choice(n_train, size=batch_size, replace=False))
    def _resample_clv_history(state, per_fam_data, rng, batch_ids=None):
        """Class-conditional Felsenstein IS (par:arch-lg08-is).

        For each family with a CLV bundle:
          1. Build class-conditional GTR transition cache using the
             current pi_class + per-site class assignments c_s.
          2. Peel the tree top-down under this proposal to get a fresh
             CLV.
          3. Draw a whole-tree history via the standard stochastic-
             mapping algorithm.
          4. Refresh aa_a, aa_b, both_aa in place; stash X and the
             proposal caches for the IS weight computation later.
        """
        from tkfdp.pswm_peeling import (
            build_class_conditional_transition_cache,
            compute_class_conditional_clv,
            sample_class_conditional_history)
        from tkfdp.lg08 import S_LG08, PI_LG08 as _PI_LG08
        S = np.asarray(S_LG08, dtype=np.float64)
        pi_class = state.pi_class
        # Ancestor-sampling proposal: use a FIXED LG08 stationary
        # (broadcast into pi_field shape) rather than the currently-
        # trained pi_field. Under the trained proposal there is a
        # self-reinforcing feedback loop — pi_arch peaks on residue r
        # → P_cache favors r at internal nodes → sampled X_v = r at
        # more nodes → SS reinforces the peak. Empirically drove the
        # cap-1 archetypes past the LG-C10 sparsity floor (ent min
        # 0.00 vs LG-C10 ent min 1.55). Freezing the proposal at LG08
        # breaks the loop; the IS weight computation later
        # (_compute_cluster_correction_weights) already handles the
        # target/proposal ratio, so this just changes the proposal.
        pi_field_shape = state.dyn_field.pi_field.shape  # (K_c, L_max, A)
        pi_field = np.broadcast_to(
            np.asarray(_PI_LG08, dtype=np.float64),
            pi_field_shape).copy()
        rho = np.asarray(state.dyn_field.rho, dtype=np.float64)
        # Γ+I bin means: if enabled, feed to the proposal cache so the
        # Felsenstein sampler at invariant columns (m=0) uses P=I, at
        # slow columns uses P=expm(m·Q·τ) with m<1, etc. Without this,
        # X was drawn under an m=1 proposal at every column and the
        # step-3 bin Gibbs then punished bin 0 with -inf on any X≠Y
        # produced by the m=1 sampler — a self-defeating inconsistency
        # that empirically drove frac_invariant to 0.
        _bin_means_prop = None
        _K_r_site = int(getattr(state.dyn_field,
                                     'K_rate_bins_site', 0) or 0)
        if _K_r_site > 0:
            from tkfdp.coupling.dynfield.hr_jax import gamma_quantile_means
            _alpha_g = float(getattr(state.dyn_field,
                                          'alpha_gamma_site', 1.0))
            _bin_means_prop = np.concatenate(
                [[0.0], gamma_quantile_means(_alpha_g, _K_r_site)]
            ).astype(np.float64)
        _batch_set = None if batch_ids is None else set(int(i) for i in batch_ids)
        for fam_idx, fd in enumerate(per_fam_data):
            if _batch_set is not None and fam_idx not in _batch_set:
                continue
            fc = fd.get('_clv')
            if fc is None:
                continue
            cls = state.states_per_msa[fam_idx].cls
            st_fam = state.states_per_msa[fam_idx]
            _srb = getattr(st_fam, 'site_rate_bin', None)
            _use_bins = _bin_means_prop is not None and _srb is not None
            _srb_col = (np.asarray(_srb, dtype=np.int64)
                          if _use_bins else None)
            P_cache, tau_idx, log_P_prop = build_class_conditional_transition_cache(
                pi_field, rho, S, fc.parent, fc.tau,
                bin_means=(_bin_means_prop if _use_bins else None))
            clv_cc, _ = compute_class_conditional_clv(
                fc.leaf_msa, fc.parent, cls, pi_class, P_cache, tau_idx,
                site_rate_bin=_srb_col)
            X = sample_class_conditional_history(
                clv_cc, fc.parent, cls, fc.leaf_msa,
                pi_class, P_cache, tau_idx, rng,
                site_rate_bin=_srb_col)
            fd['_X'] = X
            fd['_log_P_proposal_cache'] = log_P_prop
            fd['_tau_idx'] = tau_idx
            fc_hard = fc.extract_branch_cherries(X)
            fd['aa_a'] = fc_hard.aa_a
            fd['aa_b'] = fc_hard.aa_b
            fd['both_aa'] = fc_hard.both_aa_mask()

    def _compute_cluster_correction_weights(state, per_fam_data,
                                                    temperature: float = 1.0):
        """Per-(family, cluster, cherry) IS weight comparing the
        Interp 2 cluster joint (target) to the per-site class-marginal
        product (proposal) — the field-coupling correction discussed
        in par:arch-lg08-is.

        For each cluster observation tuple (classes, X_C, Y_C, tau_q):
          log_w = log P_cluster(X_C, Y_C | tau_q, classes)                  (target)
                  - sum_{s in C} log P^{(c_s, marg)}(X_s, Y_s; tau_q)         (proposal)

        Target is `cluster_emission_batched` (Interp 2 with shared θ_P
        per cherry). Proposal is my per-site field-marginal peeling
        (each site sampled independently marginalising θ). Ratio is
        trivially 1 for size-1 clusters at rho_chain = 0; nontrivial
        otherwise, quantifying the intra-cluster field-coupling signal.

        Returns:
          weights: list of floats, aligned per-tuple with
                     extract_cluster_observations output.
          diag: dict of ESS/K plus per-cluster-size log_w statistics.
        """
        import numpy as _np
        from tkfdp.coupling.dynfield import emission as _em
        from tkfdp.coupling.dynfield import DynamicFieldCouplingModel
        from tkfdp.partition_K import (cluster_id_from_partner,
                                              clusters_from_cluster_id)
        from tkfdp.pswm_peeling import enumerate_branches

        model = DynamicFieldCouplingModel.from_svi_state(state)
        weights: 'list[float]' = []
        log_w_all: 'list[float]' = []
        log_w_by_size: 'dict[int, list[float]]' = {}
        n_singleton = 0
        n_multi = 0
        for fam_idx, fd in enumerate(per_fam_data):
            st = state.states_per_msa[fam_idx]
            if st.cluster_id is None:
                st.cluster_id = cluster_id_from_partner(st.partner)
            clusters = clusters_from_cluster_id(st.cluster_id)
            aa_a = fd['aa_a']
            aa_b = fd['aa_b']
            both_aa = fd['both_aa']
            tau_family = fd['tau']                             # (n_cherries,)
            cls = st.cls

            log_P_prop_node = fd.get('_log_P_proposal_cache')   # (K_c, n_tau_uniq, A, A)
            tau_idx_node = fd.get('_tau_idx')                    # (n_nodes,) -> tau_uniq bin
            fc = fd.get('_clv')

            if log_P_prop_node is None or tau_idx_node is None or fc is None:
                # No CLV bundle: skip correction, emit weight 1 per tuple.
                for _, members in clusters.items():
                    mem_a = _np.asarray(members, dtype=_np.int64)
                    n_obs = both_aa[:, mem_a].sum(axis=1)
                    for _ in range(int((n_obs > 0).sum())):
                        weights.append(1.0)
                continue

            # Map per-cherry index in fd -> tau bin. Cherry order matches
            # enumerate_branches, so cherry q's child node id is branches[q, 1].
            branches = enumerate_branches(fc.parent)
            tau_idx_cherry = tau_idx_node[branches[:, 1]]        # (n_cherries,)

            per_cherry_cache = _em.precompute_cluster_emission_per_cherry(
                tau=tau_family,
                rho=model.dyn_field.rho,
                pi_field=model.dyn_field.pi_field,
                rho_chain=float(model.dyn_field.rho_chain),
            )

            for cid, members in clusters.items():
                mem = _np.asarray(members, dtype=_np.int64)
                m = int(mem.shape[0])
                classes_c = cls[mem].astype(_np.int64)          # (m,)
                X_batch = aa_a[:, mem]                          # (n_cherries, m)
                Y_batch = aa_b[:, mem]
                mask_X = X_batch < 20
                mask_Y = Y_batch < 20

                # Target: Interp 2 cluster joint per cherry.
                totals = _em.cluster_emission_batched(
                    classes=classes_c,
                    X_batch=X_batch, Y_batch=Y_batch,
                    mask_X=mask_X, mask_Y=mask_Y,
                    per_cherry=per_cherry_cache,
                )                                                # (n_cherries,)
                log_p_target = _np.log(_np.maximum(totals, 1e-300))

                # Proposal: sum over cluster sites of log P^{(c_s, marg)}.
                # log_P_prop_node has shape (K_c, n_tau_uniq, A, A).
                # For each cherry q and site s in cluster:
                #   log_P_prop_node[classes_c[s], tau_idx_cherry[q],
                #                    X_batch[q, s], Y_batch[q, s]]
                n_cherries = int(X_batch.shape[0])
                if m == 0 or n_cherries == 0:
                    continue
                # Sanitise gather indices at gap positions (they'll be
                # masked out below via both_aa).
                X_safe = _np.where(mask_X, X_batch, 0)
                Y_safe = _np.where(mask_Y, Y_batch, 0)
                # Index: (n_cherries, m) into (K_c, n_tau_uniq, A, A).
                cls_bcast = classes_c[None, :]                    # (1, m)
                tau_bcast = tau_idx_cherry[:, None]                # (n_cherries, 1)
                # Fancy index: broadcast to (n_cherries, m).
                gathered = log_P_prop_node[cls_bcast, tau_bcast,
                                                  X_safe, Y_safe]   # (n_cherries, m)
                both = mask_X & mask_Y
                gathered = _np.where(both, gathered, 0.0)
                log_q_proposal = gathered.sum(axis=1)              # (n_cherries,)

                log_w_c = log_p_target - log_q_proposal            # (n_cherries,)

                # Emit weight per observed cherry (aligns with
                # extract_cluster_observations, which iterates flat
                # indices where n_obs_per_cherry > 0).
                obs = both_aa[:, mem]                              # (n_cherries, m)
                n_obs_per_cherry = obs.sum(axis=1)
                for q in _np.flatnonzero(n_obs_per_cherry > 0):
                    lw = float(log_w_c[q])
                    log_w_all.append(lw)
                    weights.append(lw)                              # convert to unnormed later
                    log_w_by_size.setdefault(m, []).append(lw)
                if m == 1:
                    n_singleton += int((n_obs_per_cherry > 0).sum())
                else:
                    n_multi += int((n_obs_per_cherry > 0).sum())

        # Tempering + corpus-max normalisation for numerical stability.
        if log_w_all:
            log_w_arr = _np.asarray(log_w_all, dtype=_np.float64)
            if temperature != 1.0:
                log_w_arr = temperature * log_w_arr
            max_log = float(log_w_arr.max())
            unnorm = _np.exp(log_w_arr - max_log)
            wn = unnorm / max(float(unnorm.sum()), 1e-30)
            ess = 1.0 / max(float(_np.sum(wn * wn)), 1e-30)
            ess_frac = ess / float(len(log_w_all))
            # Overlay onto per-tuple weight list.
            final_weights: 'list[float]' = []
            idx_w = 0
            for w in weights:
                if isinstance(w, float) and w == 1.0 and idx_w >= len(unnorm):
                    final_weights.append(1.0)
                else:
                    final_weights.append(float(unnorm[idx_w]))
                    idx_w += 1
            weights = final_weights
        else:
            ess_frac = 1.0

        # Per-cluster-size log_w summary.
        size_summary: 'dict[int, tuple[float, float]]' = {}
        for sz, lws in log_w_by_size.items():
            arr = _np.asarray(lws, dtype=_np.float64)
            size_summary[sz] = (float(arr.mean()), float(arr.std()))
        diag = dict(ess_frac=ess_frac,
                        n_tuples=len(log_w_all),
                        n_singleton_tuples=n_singleton,
                        n_multi_tuples=n_multi,
                        log_w_by_size=size_summary,
                        log_w_max_min_gap=(float(log_w_arr.max() - log_w_arr.min())
                                                if log_w_all else 0.0))
        return weights, diag


    def _compute_is_cluster_weights(state, per_fam_data,
                                          temperature: float = 1.0):
        """Per-cluster SNIS weights + ESS diagnostic for par:arch-lg08-is.

        Iterates over families in the same order that
        extract_cluster_observations does. Emits a parallel list of
        floats (one per (family, cluster, cherry) tuple) suitable to
        pass as `weight_per_cluster` into train_dynfield_one_iter.

        Uses the class-marginal F81-on-DP transition against the LG08
        LG08 transition to compute log P_TKFDP / log P_LG08 per site;
        aggregates over sites in each cluster to get per-cluster log_w;
        then subtracts corpus-wide max and exponentiates. The M-step is
        invariant to a global weight scaling (all V, U, W, T scale
        together and cancel in the pi_arch / rho_chain updates) so
        SNIS self-normalisation is not further applied here.
        """
        import numpy as _np
        from tkfdp.lg08_is import (compute_lg08_log_p_per_site,
                                          compute_tkfdp_log_p_per_site,
                                          build_log_p_class_marg,
                                          build_log_p_lg08, _tau_to_idx)
        from tkfdp.lg08 import S_LG08
        from tkfdp.partition_K import (cluster_id_from_partner,
                                              clusters_from_cluster_id)
        S = _np.asarray(S_LG08, dtype=_np.float64)
        pi_field = state.dyn_field.pi_field
        rho = _np.asarray(state.dyn_field.rho, dtype=_np.float64)

        per_cluster_log_w = []                              # (fam, cid, obs_indices)
        ordered_tuples = []                                 # aligns w/ extract_cluster_observations
        for fam_idx, fd in enumerate(per_fam_data):
            fc = fd.get('_clv')
            X = fd.get('_X')
            if fc is None or X is None:
                # Non-CLV family: emit weight = 1 for every tuple that
                # will come out of extract_cluster_observations.
                st = state.states_per_msa[fam_idx]
                if st.cluster_id is None:
                    st.cluster_id = cluster_id_from_partner(st.partner)
                clusters = clusters_from_cluster_id(st.cluster_id)
                both_aa = fd['both_aa']
                for _, members in clusters.items():
                    mem = _np.asarray(members, dtype=_np.int64)
                    obs = both_aa[:, mem]
                    n_obs = obs.sum(axis=1)
                    for _ in range(int((n_obs > 0).sum())):
                        ordered_tuples.append(1.0)
                continue

            # Target: F81-on-DP field-marginal per-branch transition.
            valid = fc.parent >= 0
            unique_tau = _np.unique(_np.round(fc.tau[valid] / 1e-6) * 1e-6)
            log_P_TK = build_log_p_class_marg(pi_field, rho, S, unique_tau)
            tau_idx_tk = _tau_to_idx(_np.round(fc.tau / 1e-6) * 1e-6,
                                          unique_tau)

            cls = state.states_per_msa[fam_idx].cls
            log_p_TK = compute_tkfdp_log_p_per_site(
                X, fc.parent, fc.tau, cls, state.pi_class,
                log_P_TK, tau_idx_tk)
            # Proposal: class-conditional GTR cache stashed in
            # _resample_clv_history above. Compute log-p per site using
            # the same helper, which is generic in the transition cache.
            log_P_prop = fd.get('_log_P_proposal_cache')
            tau_idx_prop = fd.get('_tau_idx')
            log_p_PROP = compute_tkfdp_log_p_per_site(
                X, fc.parent, fc.tau, cls, state.pi_class,
                log_P_prop, tau_idx_prop)
            log_ratio = log_p_TK - log_p_PROP              # (L,)

            st = state.states_per_msa[fam_idx]
            if st.cluster_id is None:
                st.cluster_id = cluster_id_from_partner(st.partner)
            clusters = clusters_from_cluster_id(st.cluster_id)
            both_aa = fd['both_aa']
            for cid, members in clusters.items():
                mem = _np.asarray(members, dtype=_np.int64)
                cluster_log_w = float(log_ratio[mem].sum())
                obs = both_aa[:, mem]
                n_obs = obs.sum(axis=1)
                n_tuples = int((n_obs > 0).sum())
                for _ in range(n_tuples):
                    per_cluster_log_w.append(cluster_log_w)
                    ordered_tuples.append(None)             # placeholder

        # Tempering: w' = w^temperature (β = 0 → uniform, 1 → proper IS).
        # Corpus-wide log-max normalisation for numerical stability.
        if temperature != 1.0:
            per_cluster_log_w = [float(temperature) * lw
                                    for lw in per_cluster_log_w]
        if per_cluster_log_w:
            max_log = float(_np.max(per_cluster_log_w))
        else:
            max_log = 0.0
        # Overlay the CLV weights onto the ordered tuple list.
        clv_idx = 0
        final_weights = []
        # Second pass — reassemble in the ordered_tuples order, replacing
        # the placeholder Nones with normalised weights.
        # (In the loop above, non-CLV families produced float(1.0)
        # directly, while CLV families produced None + a log_w stored
        # in per_cluster_log_w.)
        for slot in ordered_tuples:
            if slot is None:
                w = float(_np.exp(per_cluster_log_w[clv_idx] - max_log))
                clv_idx += 1
                final_weights.append(w)
            else:
                final_weights.append(float(slot))
        # Diagnostic: ESS across CLV clusters only.
        if per_cluster_log_w:
            log_w = _np.asarray(per_cluster_log_w, dtype=_np.float64)
            w = _np.exp(log_w - max_log)
            wn = w / max(w.sum(), 1e-30)
            ess = 1.0 / max(_np.sum(wn * wn), 1e-30)
            ess_frac = ess / float(len(per_cluster_log_w))
        else:
            ess_frac = 1.0
        return final_weights, ess_frac
    t0 = time.time()
    n_train = len(train_per_fam)
    _batch_size = int(getattr(args, 'batch_size', 0) or 0)
    _use_batch = _batch_size > 0 and _batch_size < n_train
    for it in range(start_iter, start_iter + args.n_outer_iters):
        t_it = time.time()
        # Mini-batch: sample which families are refreshed + contribute
        # SS this iter. If disabled, batch_ids is the full corpus.
        batch_ids = _sample_batch_ids(n_train, _batch_size, _batch_rng)
        if _use_batch:
            print(f"    [batch] iter {it+1} batch={len(batch_ids)}/"
                    f"{n_train} ids[:8]={batch_ids[:8]}", flush=True)
        _resample_clv_history(state, train_per_fam, _clv_rng,
                                 batch_ids=(batch_ids if _use_batch else None))
        if val_per_fam:
            _resample_clv_history(state, val_per_fam, _clv_rng)
        # Cluster + class sweeps.
        for _ in range(args.n_cluster_sweeps_per_outer):
            from tkfdp.svi import cluster_gibbs_sweep_all
            state = cluster_gibbs_sweep_all(
                state, train_per_fam, rng,
                alpha_z=alpha_z_state,
                max_cluster_size=args.max_cluster_size,
                batch_ids=(batch_ids if _use_batch else None))
        # Optional alpha_z MH update (after the sweep that produced the
        # current partitions).
        az_mh_accept = None
        if args.alpha_z_mh_steps > 0:
            from tkfdp.partition_K import (update_alpha_z_mh,
                                              clusters_from_cluster_id)
            partitions = []
            for st in state.states_per_msa:
                cmap = clusters_from_cluster_id(st.cluster_id)
                partitions.append((len(cmap), int(st.L)))
            alpha_z_state, az_info = update_alpha_z_mh(
                alpha_z_state, partitions,
                prior_a=args.alpha_z_prior_a,
                prior_b=args.alpha_z_prior_b,
                n_steps=args.alpha_z_mh_steps,
                step_size=args.alpha_z_step_size,
                rng=rng)
            az_mh_accept = az_info['n_steps_accept']
        for _ in range(args.n_class_sweeps_per_outer):
            if args.K_c > 1:
                state = class_gibbs_sweep_all_dynfield(
                    state, train_per_fam, rng, alpha_c=args.alpha_c,
                    batch_ids=(batch_ids if _use_batch else None))
        # Cluster extraction + atom update + optional rho_chain MH.
        from tkfdp.svi import (extract_cluster_observations,
                                  train_dynfield_one_iter,
                                  extract_cluster_bins,
                                  gibbs_resample_site_bins)
        clusters = extract_cluster_observations(state, train_per_fam)
        # Per-site +Γ+I Gibbs resample every args.site_resample_every iters
        # (par:arch-gamma-plus-I-persite). Skip when disabled or on the
        # very first outer iter (where pi_arch is still at random init;
        # a Gibbs step here samples against a nonsense conditional and
        # can inject enough invariant bins to bias the first atom step).
        # First fire is after iter site_resample_every.
        site_gibbs_info = {}
        elapsed = it - start_iter
        if (args.K_rate_bins_site > 0 and args.site_resample_every > 0
                and elapsed > 0
                and (elapsed % args.site_resample_every == 0)):
            site_gibbs_info = gibbs_resample_site_bins(
                state, train_per_fam, rng,
                batch_ids=(batch_ids if _use_batch else None))
            if site_gibbs_info:
                print(f"    [persite Gibbs @ iter {it+1}] "
                        f"n_columns={site_gibbs_info.get('n_columns_resampled', 0)} "
                        f"flipped={site_gibbs_info.get('n_columns_flipped', 0)} "
                        f"frac_invariant={site_gibbs_info.get('frac_invariant', 0.0):.3f}",
                        flush=True)
        bins_per_cluster = extract_cluster_bins(state, train_per_fam)
        # Tied-theta MCMC path (par:arch-lg08-is): sample theta_V at
        # every internal node per cluster + binary M_v per branch, then
        # accumulate HR SS on the fully-observed (X, theta, M) trajectory.
        # Bypasses train_dynfield_one_iter's composite HR pass.
        do_sm = (args.split_merge_every > 0
                    and (elapsed % args.split_merge_every == 0))
        theta_mcmc_diag = {}
        clv_present = any(fd.get('_clv') is not None for fd in train_per_fam)
        if getattr(args, 'sample_theta_mcmc', False) and any(
                fd.get('_clv') is not None and fd.get('_X') is not None
                for fd in train_per_fam):
            import time as _time_dbg
            _t_theta_start = _time_dbg.time()
            from tkfdp.theta_mcmc import (sample_theta_cluster,
                                                  build_shared_P_cache)
            from tkfdp.theta_mcmc_hr import accumulate_hr_ss_tied_theta
            from tkfdp.coupling.dynfield.hr import (
                update_pi_archetype_gtr, update_rho_chain_gamma)
            from tkfdp.coupling.dynfield.hr_jax import aggregate_by_arch
            from tkfdp.lg08 import S_LG08
            from tkfdp.partition_K import (cluster_id_from_partner,
                                                  clusters_from_cluster_id)
            from tkfdp.pswm_peeling import enumerate_branches
            S_np = np.asarray(S_LG08, dtype=np.float64)
            pi_field_np = np.asarray(state.dyn_field.pi_field,
                                          dtype=np.float64)
            rho_np = np.asarray(state.dyn_field.rho, dtype=np.float64)
            rho_chain_f = float(state.dyn_field.rho_chain)
            # Per-site Γ+I: if enabled, bin_means_full[0] = 0 (invariant)
            # plus K_rate_bins_site Yang Γ-quantile means.
            _bin_means_full = None
            _K_r_site = int(getattr(state.dyn_field, 'K_rate_bins_site', 0) or 0)
            if _K_r_site > 0:
                from tkfdp.coupling.dynfield.hr_jax import gamma_quantile_means
                _alpha_g = float(getattr(state.dyn_field,
                                              'alpha_gamma_site', 1.0))
                _bin_means_full = np.concatenate(
                    [[0.0], gamma_quantile_means(_alpha_g, _K_r_site)])

            _n_draws = int(getattr(args, 'theta_mcmc_n_draws', 1) or 1)
            _hr_agg = None   # accumulated over draws
            n_M1_total = 0
            n_M0_total = 0
            _t_sample_total = 0.0
            _t_pcache_total = 0.0
            _t_hr = _time_dbg.time()
            _batch_set = set(int(i) for i in batch_ids)
            for _draw_i in range(_n_draws):
                if _draw_i > 0:
                    _resample_clv_history(
                        state, train_per_fam, _clv_rng,
                        batch_ids=(batch_ids if _use_batch else None))
                theta_samples: 'dict[int, dict[int, np.ndarray]]' = {}
                M_samples: 'dict[int, dict[int, np.ndarray]]' = {}
                cluster_columns_map: 'dict[int, dict[int, np.ndarray]]' = {}
                cluster_classes_map: 'dict[int, dict[int, np.ndarray]]' = {}
                cluster_branches_map: 'dict[int, dict[int, np.ndarray]]' = {}
                for fam_idx, fd in enumerate(train_per_fam):
                    if fam_idx not in _batch_set:
                        continue
                    fc = fd.get('_clv')
                    X = fd.get('_X')
                    if fc is None or X is None:
                        continue
                    st = state.states_per_msa[fam_idx]
                    cls = st.cls
                    if st.cluster_id is None:
                        st.cluster_id = cluster_id_from_partner(st.partner)
                    clusters_dict = clusters_from_cluster_id(st.cluster_id)
                    theta_samples[fam_idx] = {}
                    M_samples[fam_idx] = {}
                    cluster_columns_map[fam_idx] = {}
                    cluster_classes_map[fam_idx] = {}
                    cluster_branches_map[fam_idx] = {}
                    branches = enumerate_branches(fc.parent)
                    _t0 = _time_dbg.time()
                    P_cache_fam, tau_idx_fam = build_shared_P_cache(
                        pi_field_np, S_np, fc.parent, fc.tau)
                    _t_pcache_total += _time_dbg.time() - _t0
                    for cid, members in clusters_dict.items():
                        mem = np.asarray(members, dtype=np.int64)
                        classes_c = cls[mem].astype(np.int64)
                        _t1 = _time_dbg.time()
                        theta_sampled, M_v, _ = sample_theta_cluster(
                            X, mem, classes_c, fc.parent, fc.tau,
                            pi_field_np, rho_np, S_np, rho_chain_f,
                            _clv_rng,
                            P_cache_full=P_cache_fam,
                            tau_idx_full=tau_idx_fam)
                        _t_sample_total += _time_dbg.time() - _t1
                        theta_samples[fam_idx][cid] = theta_sampled
                        M_samples[fam_idx][cid] = M_v
                        cluster_columns_map[fam_idx][cid] = mem
                        cluster_classes_map[fam_idx][cid] = classes_c
                        cluster_branches_map[fam_idx][cid] = branches
                        n_M1_total += int(M_v.sum())
                        n_M0_total += int((M_v == 0).sum()) - 1
                _hr_draw = accumulate_hr_ss_tied_theta(
                    state, train_per_fam, theta_samples, M_samples,
                    cluster_columns_map, cluster_classes_map,
                    cluster_branches_map, S_np,
                    bin_means_full=_bin_means_full)
                if _hr_agg is None:
                    _hr_agg = {k: (v.copy() if hasattr(v, 'copy') else v)
                                  for k, v in _hr_draw.items()}
                else:
                    _hr_agg['V'] += _hr_draw['V']
                    _hr_agg['U'] += _hr_draw['U']
                    _hr_agg['W'] += _hr_draw['W']
                    _hr_agg['N_theta_sum'] += _hr_draw['N_theta_sum']
                    _hr_agg['T_sum'] += _hr_draw['T_sum']
                    _hr_agg['n_clust'] += _hr_draw['n_clust']
            hr_stats = _hr_agg
            _t_hr_dur = _time_dbg.time() - _t_hr
            print(f"    [theta-timing] P_cache={_t_pcache_total:.1f}s "
                    f"sample={_t_sample_total:.1f}s HR_SS={_t_hr_dur:.1f}s "
                    f"total_theta={_time_dbg.time() - _t_theta_start:.1f}s",
                    flush=True)
            # Mini-batch scaling: batch SS estimates full-corpus SS
            # under uniform inclusion probability. Scale by n_train /
            # batch_size for ρ_chain's Gamma MAP so the prior weight
            # stays comparable to the scaled data term. (pi_arch Newton
            # is scale-invariant so no scaling needed there.)
            _scale = float(n_train) / max(len(batch_ids), 1) if _use_batch else 1.0
            # Robbins-Monro damped M-step. α_t = α_init · (t+1)^(−decay).
            _alpha_init = float(getattr(args, 'm_step_alpha_init', 1.0))
            _alpha_decay = float(getattr(args, 'm_step_alpha_decay', 0.5))
            _alpha_t = (_alpha_init * (float(it + 1) ** (-_alpha_decay))
                          if _use_batch else 1.0)
            # arch_assignment update (before pi_arch M-step so
            # aggregate_by_arch uses fresh assignment). HR V (K_c, L_max,
            # A) plays the role of N for the archetype conditional.
            _arch_mode = getattr(args, 'arch_update_mode', 'none')
            _arch_info = {'mode': _arch_mode}
            if _arch_mode == 'argmax':
                from tkfdp.coupling.dynfield.archetypes import (
                    soft_arch_posterior)
                _probs = soft_arch_posterior(
                    hr_stats['V'], state.dyn_field.pi_archetype,
                    state.dyn_field.rho_arch)
                state.dyn_field.arch_assignment = np.argmax(
                    _probs, axis=2).astype(np.int32)
            elif _arch_mode == 'swap':
                from tkfdp.coupling.dynfield.archetypes import (
                    swap_arch_step)
                _new_aa, _swap_info = swap_arch_step(
                    state.dyn_field.arch_assignment,
                    hr_stats['V'],
                    state.dyn_field.pi_archetype,
                    rng,
                    n_sweeps=int(getattr(args, 'arch_swap_sweeps', 1)))
                state.dyn_field.arch_assignment = _new_aa
                _arch_info.update(_swap_info)
                print(f"    [arch-swap] proposed={_swap_info['n_proposed']} "
                        f"accepted={_swap_info['n_accepted']} "
                        f"rate={_swap_info['accept_rate']:.2f}",
                        flush=True)
            elif _arch_mode == 'gibbs':
                from tkfdp.coupling.dynfield.archetypes import (
                    sample_arch_assignment)
                _new_aa = sample_arch_assignment(
                    hr_stats['V'], state.dyn_field.pi_archetype,
                    state.dyn_field.rho_arch, rng)
                state.dyn_field.arch_assignment = _new_aa
                _n_changed = int(
                    (_new_aa != np.asarray(state.dyn_field.arch_assignment)).sum())
                print(f"    [arch-gibbs] resampled all (c, θ) from multinomial "
                        f"conditional (V-only)", flush=True)
            # Fix theta=0 diagonal if requested (after any update above).
            if (getattr(args, 'fix_theta0_diagonal', False)
                    and _arch_mode in ('argmax', 'swap', 'gibbs')):
                _n_diag = int(min(state.dyn_field.K_c, state.dyn_field.K_a))
                _aa_fixed = np.asarray(
                    state.dyn_field.arch_assignment, dtype=np.int32).copy()
                _aa_fixed[:_n_diag, 0] = np.arange(_n_diag, dtype=np.int32)
                state.dyn_field.arch_assignment = _aa_fixed
            # pi_archetype update (gated by --freeze-pi-arch).
            if not args.freeze_pi_arch:
                V_agg, U_agg, W_agg = aggregate_by_arch(
                    hr_stats['V'], hr_stats['U'], hr_stats['W'],
                    state.dyn_field.arch_assignment,
                    state.dyn_field.K_a)
                pi_arch_target = update_pi_archetype_gtr(
                    V_agg, U_agg, W_agg, S_np,
                    alpha_prior=args.alpha_prior)
                _pi_arch_old = np.asarray(
                    state.dyn_field.pi_archetype, dtype=np.float64)
                pi_arch_new = ((1.0 - _alpha_t) * _pi_arch_old
                                  + _alpha_t * pi_arch_target)
                pi_arch_new = pi_arch_new / np.maximum(
                    pi_arch_new.sum(axis=1, keepdims=True), 1e-300)
                state.dyn_field.pi_archetype = pi_arch_new
            # rho_chain always updates.
            rc_target = update_rho_chain_gamma(
                _scale * hr_stats['N_theta_sum'],
                _scale * hr_stats['T_sum'],
                prior_a=args.rho_chain_prior_a,
                prior_b=args.rho_chain_prior_b,
                mode='map')
            _rc_old = float(state.dyn_field.rho_chain)
            rc_new = (1.0 - _alpha_t) * _rc_old + _alpha_t * float(rc_target)
            state.dyn_field.rho_chain = float(rc_new)
            if _use_batch:
                _pi_msg = ("frozen" if args.freeze_pi_arch
                              else "updated")
                print(f"    [svi] α_t={_alpha_t:.3f} rc_target={rc_target:.3f} "
                        f"→ rc={rc_new:.3f}  pi_arch={_pi_msg}", flush=True)

            # Field-DP TSB M-step (gated by --update-field-tsb; default
            # off, in which case rho stays frozen at uniform).
            if args.update_field_tsb:
                r_theta = np.zeros(int(state.dyn_field.L_max),
                                    dtype=np.float64)
                for _fi, _fd in enumerate(train_per_fam):
                    _tbc = theta_samples.get(_fi, {})
                    if not _tbc:
                        continue
                    _both_aa = _fd['both_aa']
                    _cc_map = cluster_columns_map[_fi]
                    _br_map = cluster_branches_map[_fi]
                    for _cid, _ts in _tbc.items():
                        _cols = _cc_map[_cid]
                        _br = _br_map[_cid]
                        _child = _br[:, 1]
                        _theta_c = _ts[_child]
                        _has_obs = _both_aa[:, _cols].any(axis=1)
                        _mask = _has_obs
                        _bc = np.bincount(
                            _theta_c[_mask].astype(np.int64),
                            minlength=int(state.dyn_field.L_max))
                        r_theta += _bc.astype(np.float64)
                from tkfdp.coupling.dynfield.updates import update_rho_tsb
                class _ModelWrap:
                    def __init__(_s, ss):
                        _s.dyn_field = ss.dyn_field
                _wrap = _ModelWrap(state)
                # Beta-posterior sample (not MAP): MAP TSB on hard-count
                # r collapses to one-atom (see --update-field-tsb
                # helptext). Sample mode is empirically enough to hold
                # against the pathology.
                tsb_new, rho_new = update_rho_tsb(_wrap, r_theta,
                                                        mode='sample', rng=rng)
                state.dyn_field.tsb_betas = tsb_new
                state.dyn_field.rho = rho_new

            state.dyn_field.materialise_pi_field()
            state.dyn_field.pi_class = np.einsum(
                't,cta->ca', state.dyn_field.rho, state.dyn_field.pi_field)

            # Split-merge on the archetype block (bypasses composite
            # train_dynfield_one_iter, so must be invoked explicitly).
            # V_total (segment starts + M=1 attributions) plays the role
            # of N_ctheta for the multinomial LL scoring; SM internally
            # rebuilds rho_arch and tsb_betas_arch from the new
            # arch_assignment.
            sm_info = {}
            if do_sm:
                from tkfdp.coupling.dynfield.split_merge import (
                    split_merge_step)
                sm_info = split_merge_step(
                    state, hr_stats['V'], rng,
                    max_moves_per_call=int(args.split_merge_max_moves),
                    dead_threshold=float(args.split_merge_dead_threshold),
                    merge_tvd_threshold=float(args.split_merge_tvd_threshold),
                    verbose=False)
                if sm_info.get('n_moves_accepted', 0) > 0:
                    state.dyn_field.materialise_pi_field()
                    state.dyn_field.pi_class = np.einsum(
                        't,cta->ca', state.dyn_field.rho,
                        state.dyn_field.pi_field)
            # Undo SM's implicit rho_arch update if the archetype TSB
            # is meant to stay uniform.
            if not args.update_arch_tsb:
                _reset_arch_tsb_uniform(state)

            # Per-iter DP + archetype diagnostics.
            _pa = state.dyn_field.pi_archetype
            _K_a = int(_pa.shape[0])
            _usage = np.bincount(
                state.dyn_field.arch_assignment.ravel().astype(np.int64),
                minlength=_K_a)
            _ent = -np.sum(_pa * np.log(np.maximum(_pa, 1e-30)), axis=1)
            _ra = np.asarray(state.dyn_field.rho_arch, dtype=np.float64) \
                if getattr(state.dyn_field, 'rho_arch', None) is not None \
                else np.zeros(_K_a)
            _dead = int(((_usage == 0) | (_ra < 0.01)).sum())
            print(f"    [dp] rho=["
                    f"{','.join(f'{v:.3f}' for v in state.dyn_field.rho)}]  "
                    f"arch_used={int((_usage > 0).sum())}/{_K_a}  "
                    f"arch_dead={_dead}  "
                    f"arch_ent(min/med/max)="
                    f"{float(_ent.min()):.2f}/"
                    f"{float(np.median(_ent)):.2f}/"
                    f"{float(_ent.max()):.2f}  "
                    f"rho_arch=["
                    f"{','.join(f'{v:.2f}' for v in _ra)}]",
                    flush=True)
            _n_splits = len(sm_info.get('splits', [])) if sm_info else 0
            _n_merges = len(sm_info.get('merges', [])) if sm_info else 0
            print(f"    [split-merge] splits={_n_splits} merges={_n_merges} "
                    f"accepted={sm_info.get('n_moves_accepted', 0) if sm_info else 0}",
                    flush=True)

            atom_info = {
                'log_lik_total': 0.0,
                'log_lik_after_atom': 0.0,
                'log_lik_after_sm': 0.0,
                'hr_N_theta_sum': float(hr_stats['N_theta_sum']),
                'hr_T_sum': float(hr_stats['T_sum']),
                'hr_rho_chain': float(rc_new),
            }
            if sm_info:
                atom_info['split_merge'] = sm_info
            theta_mcmc_diag = dict(
                n_branches_M0=n_M0_total, n_branches_M1=n_M1_total,
                N_theta_sum=hr_stats['N_theta_sum'],
                T_sum=hr_stats['T_sum'],
                fam_families=len(theta_samples))
            is_weights, ess_frac = None, 1.0
            is_diag = {}
        else:
            if clv_present:
                is_weights, is_diag = _compute_cluster_correction_weights(
                    state, train_per_fam,
                    temperature=getattr(args, 'lg08_is_temp', 1.0))
                ess_frac = float(is_diag['ess_frac'])
            else:
                is_weights, ess_frac = None, 1.0
                is_diag = {}
            state, atom_info = train_dynfield_one_iter(
                state, clusters, alpha_prior=args.alpha_prior, rng=rng,
                use_hr_mstep=bool(args.use_hr_mstep),
                rho_chain_prior_a=args.rho_chain_prior_a,
                rho_chain_prior_b=args.rho_chain_prior_b,
                bins_per_cluster=bins_per_cluster,
                weight_per_cluster=is_weights,
                do_split_merge=do_sm,
                split_merge_dead_threshold=args.split_merge_dead_threshold,
                split_merge_tvd_threshold=args.split_merge_tvd_threshold,
                split_merge_max_moves=args.split_merge_max_moves)
        rho_mh_accept = None
        # When use_hr_mstep is on, rho_chain is updated by the closed-form
        # Gamma posterior inside train_dynfield_one_iter -- skip MH.
        if args.rho_chain_mh_steps > 0 and not args.use_hr_mstep:
            from tkfdp.coupling.dynfield import updates as _up
            from tkfdp.coupling.dynfield import DynamicFieldCouplingModel
            model = DynamicFieldCouplingModel.from_svi_state(state)
            c_obs = [(cls, X, Y) for (cls, X, Y, _) in clusters]
            c_t = np.asarray([t for (_, _, _, t) in clusters],
                              dtype=np.float64)
            new_rc, rc_info = _up.update_rho_chain_mh(
                model, c_obs, c_t,
                prior_a=args.rho_chain_prior_a,
                prior_b=args.rho_chain_prior_b,
                n_steps=args.rho_chain_mh_steps,
                step_size=args.rho_chain_step_size,
                rng=rng)
            state.dyn_field.rho_chain = new_rc
            rho_mh_accept = rc_info['n_steps_accept']
        # Cluster size summary.
        from tkfdp.partition_K import clusters_from_cluster_id
        sizes = []
        for st in state.states_per_msa:
            cmap = clusters_from_cluster_id(st.cluster_id)
            sizes.extend(len(v) for v in cmap.values())
        sizes_arr = np.asarray(sizes) if sizes else np.zeros(1)
        elapsed = time.time() - t_it
        trace['outer_iter'].append(it + 1)
        trace['log_lik_total'].append(atom_info['log_lik_total'])
        trace.setdefault('log_lik_after_atom', []).append(
            atom_info.get('log_lik_after_atom', float('nan')))
        trace.setdefault('log_lik_after_sm', []).append(
            atom_info.get('log_lik_after_sm',
                            atom_info.get('log_lik_after_atom', float('nan'))))
        trace['n_clusters_total'].append(int(len(sizes)))
        trace['mean_cluster_size'].append(float(sizes_arr.mean()))
        trace['max_cluster_size'].append(int(sizes_arr.max()))
        trace['wall_secs'].append(float(elapsed))
        trace['rho_chain'].append(float(state.dyn_field.rho_chain))
        rc_str = (f"rho_chain={state.dyn_field.rho_chain:.3f}"
                   + (f" ({rho_mh_accept}/{args.rho_chain_mh_steps} MH)"
                       if rho_mh_accept is not None else ""))
        az_str = (f"alpha_z={alpha_z_state:.3f}"
                    + (f" ({az_mh_accept}/{args.alpha_z_mh_steps} MH)"
                        if az_mh_accept is not None else ""))
        ll_before = atom_info['log_lik_total']
        ll_after_atom = atom_info.get('log_lik_after_atom', float('nan'))
        ll_after_sm = atom_info.get('log_lik_after_sm', ll_after_atom)
        d_atom = ll_after_atom - ll_before
        d_sm = ll_after_sm - ll_after_atom
        # dLL_crp: change from previous iter's END-of-iter LL (post-split-
        # merge if any) to this iter's pre-atom LL -- captures the CRP +
        # class Gibbs effect on the pure LL.
        prev_ll_end = (trace['log_lik_after_sm'][-2]
                          if len(trace['log_lik_after_sm']) >= 2 else float('nan'))
        d_crp = ll_before - prev_ll_end
        d_atom_str = f"dLL_atom={d_atom:+.1f}" if not np.isnan(d_atom) else ""
        d_crp_str = (f"dLL_crp={d_crp:+.1f}"
                       if not np.isnan(d_crp) else "")
        d_sm_str = f"  dLL_sm={d_sm:+.1f}" if abs(d_sm) > 0.5 else ""
        patience_str = (f"  patience={es.n_evals_since_improvement}"
                            if es.n_evals_since_improvement > 0 else "")
        sd_size = float(sizes_arr.std())
        p50, p90, p99 = np.percentile(sizes_arr, [50, 90, 99])
        n_gt1 = int((sizes_arr > 1).sum())
        n_gte10 = int((sizes_arr >= 10).sum())
        size_str = (f"size(mean={sizes_arr.mean():.2f} sd={sd_size:.2f} "
                       f"p50={p50:.0f} p90={p90:.0f} p99={p99:.0f} "
                       f"max={int(sizes_arr.max())} "
                       f"n>1={n_gt1} n≥10={n_gte10})")
        ess_str = (f"  ESS/K={ess_frac:.3f}" if clv_present else "")
        print(f"  iter {it+1:3d}/{start_iter + args.n_outer_iters}  "
                f"LL={ll_after_sm:+.2f}  {d_crp_str}  {d_atom_str}{d_sm_str}  "
                f"clusters={len(sizes)}  {size_str}  "
                f"{rc_str}  {az_str}{patience_str}{ess_str}  "
                f"({elapsed:.1f}s)",
                flush=True)
        # Tied-theta MCMC diagnostics.
        if theta_mcmc_diag:
            print(f"    [theta-MCMC] families={theta_mcmc_diag['fam_families']} "
                    f"M=0/M=1 branches: {theta_mcmc_diag['n_branches_M0']}/{theta_mcmc_diag['n_branches_M1']} "
                    f"N_theta_sum={theta_mcmc_diag['N_theta_sum']:.0f} "
                    f"T_sum={theta_mcmc_diag['T_sum']:.1f}",
                    flush=True)
        # Cluster-correction (par:arch-lg08-is) diagnostics.
        if is_diag:
            size_str_lw = " ".join(
                f"m={sz}:{mu:+.2f}pm{sd:.2f}"
                for sz, (mu, sd) in sorted(is_diag.get('log_w_by_size', {}).items())
                if sz <= 8)
            print(f"    [IS diag] n_tup={is_diag['n_tuples']} "
                    f"(single={is_diag['n_singleton_tuples']} "
                    f"multi={is_diag['n_multi_tuples']}) "
                    f"log_w_gap={is_diag['log_w_max_min_gap']:.2f}nats  "
                    f"{size_str_lw}",
                    flush=True)
        extra_meta = {'alpha_z': float(alpha_z_state)}
        # Track best-so-far by end-of-iter (post-split-merge) LL — this
        # is what the persisted state actually achieves.
        if not val_fams:
            ll = atom_info.get('log_lik_after_sm',
                                  atom_info.get('log_lik_after_atom',
                                                   atom_info['log_lik_total']))
            if ll > es.best_val_LL:
                es.best_val_LL = ll
                es.best_iter = it + 1
                es.n_evals_since_improvement = 0
                save_checkpoint(state, trace, rng, args.out_dir,
                                  it=it + 1, es=es,
                                  subdir="_best_chkpt",
                                  extra_meta=extra_meta)
            else:
                es.n_evals_since_improvement += 1
        if (it + 1) % args.checkpoint_every == 0:
            save_checkpoint(state, trace, rng, args.out_dir,
                              it=it + 1, es=es, subdir="_chkpt",
                              extra_meta=extra_meta)
        if args.patience > 0 and es.n_evals_since_improvement >= args.patience:
            print(f"  early stop: {es.n_evals_since_improvement} iters "
                    f"without LL improvement (patience={args.patience}); "
                    f"best iter {es.best_iter} with LL={es.best_val_LL:+.2f}",
                    flush=True)
            break
    # Final save.
    save_checkpoint(state, trace, rng, args.out_dir,
                      it=args.n_outer_iters, es=es, subdir="_chkpt",
                      extra_meta={'alpha_z': float(alpha_z_state)})
    print(f"\nTotal wall: {time.time() - t0:.1f}s", flush=True)
    print(f"Final LL: {trace['log_lik_total'][-1]:+.2f}", flush=True)
    print(f"Best LL:  {es.best_val_LL:+.2f} at iter {es.best_iter}", flush=True)


if __name__ == "__main__":
    main()
