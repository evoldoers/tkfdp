"""BAliBASE 3 evaluation harness.

Compares five methods on the 386 BAliBASE 3 benchmarks (and optionally
PREFAB / OXBENCH / SABRE bundled in the same drive5 tarball):

  - baseline_fsa     : tkfmixdom FSA with TKF92, no Potts.
  - tkfdp_precorr    : baseline + TKF-DP pre-correction (fold boost into
                       Match emission, re-run F/B). Default alpha_z=100.
  - tkfdp_coupled    : baseline + coupled-pair greedy annealing.
  - muscle           : MUSCLE 5 (`muscle -align in.fa -output out.fa`).
  - mafft            : MAFFT 7 with --auto.

MUSCLE and MAFFT are sanity-comparison points: they confirm we're using
the same scoring conventions as the wider literature, and they catch
infrastructure regressions on our side that would otherwise silently
trash our scores.

Reads BAliBASE from $BIO_DATASETS_HOME/balibase/bench1.0/<set>/{in,ref}
(default ~/bio-datasets/data/balibase/bench1.0). Writes one row per
(benchmark, method) to <out_dir>/results.csv plus a summary table.

Run:  python experiments/eval_balibase.py --bench bali3 --out results/balibase_eval
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# Hook into ~/tkf-mixdom for FSA + scoring.
TKFMIXDOM_ROOT = Path.home() / "tkf-mixdom" / "python"
sys.path.insert(0, str(TKFMIXDOM_ROOT))
from tkfmixdom.jax.evaluate.metrics import sp_score, tc_score                     # noqa: E402
from tkfmixdom.jax.util.io import AA_TO_INT, read_fasta                            # noqa: E402

# Hook into local TKF-DP code.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from test_balibase_postprocess import (                                            # noqa: E402
    _correct_pair_via_fb_rerun, load_minimal_state, _msa_from_col_assignments,
)
from tkfmixdom.jax.tree.fsa_anneal import (                                       # noqa: E402
    compute_pairwise_posteriors, sequence_annealing as _sa_inner,
    select_pairs_full, _score_alignment as _sa_score_alignment,
)


def sequence_annealing(n_seqs, seq_lens, pair_post,
                       n_iterations: int = 5, verbose: bool = False,
                       _fsa_n_restarts: int = 10,
                       _fsa_base_seed: int = 0):
    """Multi-start wrapper around tkfmixdom.jax.tree.fsa_anneal.sequence_annealing.

    Default: n_iterations=5, _fsa_n_restarts=10 — runs 10 independent
    refinement trials with seeded RNG and returns the trial with the
    highest internal `_score_alignment`. Reproducible AND multi-modal-
    aware.

    Background: the upstream code includes a "refinement" remove-and-
    reinsert loop (added in tkfmixdom commit 60ce2ace, 2026-04-04, by
    an earlier AI agent) that calls `np.random.permutation(n_seqs)`
    per iteration WITHOUT seeding the global RNG. We observed SP
    swings of up to 30 percentage points (BB11002 tkfdp_aug:
    {0.272, 0.361, 0.584}) across unseeded runs.

    Empirically the refinement loop IS helpful (disabling it lost
    ~20% SP on BB11002 — the priority-queue greedy alone gets stuck
    on hard cases that the random refinement rescues). The non-
    determinism is the only real problem.

    Strategy: seed each of N=_fsa_n_restarts trials deterministically,
    pick the best by FSA's internal score. Reproducible AND robust.

    Pass `_fsa_n_restarts=1` for legacy single-trial behaviour
    (still reproducible). Pass `n_iterations=0` to skip refinement
    entirely (matches the original FSA paper algorithm but loses
    SP on hard cases).
    """
    pair_post_dict = dict(pair_post)
    if n_iterations <= 0:
        return _sa_inner(n_seqs, seq_lens, pair_post_dict,
                         n_iterations=0, verbose=verbose)
    best_score = -np.inf
    best_col = None
    best_msa_len = None
    for restart in range(_fsa_n_restarts):
        np.random.seed(_fsa_base_seed + restart)
        col_assignments, msa_length = _sa_inner(
            n_seqs, seq_lens, pair_post_dict,
            n_iterations=n_iterations, verbose=verbose)
        score = _sa_score_alignment(col_assignments, seq_lens, pair_post_dict)
        if score > best_score:
            best_score = score
            best_col = col_assignments
            best_msa_len = msa_length
    return best_col, best_msa_len
from tkfmixdom.jax.core.protein import rate_matrix_lg                              # noqa: E402


def default_balibase_root() -> Path:
    home = Path(os.environ.get("BIO_DATASETS_HOME",
                                  Path.home() / "bio-datasets")) / "data"
    return home / "balibase" / "bench1.0"


# ============================================================================
# Data loading
# ============================================================================

def load_seqs(path: Path) -> dict:
    """Load .fa as {name: int array, wildcards mapped to 20}."""
    out = {}
    for name, seq in read_fasta(str(path)):
        # Strip non-letter characters first (bench1.0 input is plain FASTA).
        clean = "".join(c for c in seq if c.isalpha())
        out[name] = np.array([AA_TO_INT.get(c.upper(), 20) for c in clean],
                                dtype=np.int32)
    return out


def load_ref(path: Path) -> tuple[dict, dict]:
    """Load .ref alignment. Returns (msa, core_mask) where:
      msa[name] = (L_aln,) int array, -1 for gap.
      core_mask[name] = (L_aln,) bool array, True only on uppercase positions.
    BAliBASE convention: lowercase = unaligned 'insert' regions outside the
    core, dots/dashes = gap. Strict SP/TC scoring restricts to core columns."""
    msa, core = {}, {}
    for name, seq in read_fasta(str(path)):
        L = len(seq)
        row = np.full(L, -1, dtype=np.int32)
        is_core = np.zeros(L, dtype=bool)
        for k, c in enumerate(seq):
            if c == '.' or c == '-':
                row[k] = -1
            else:
                row[k] = AA_TO_INT.get(c.upper(), 20)
                is_core[k] = c.isupper()
        msa[name] = row; core[name] = is_core
    return msa, core


def restrict_to_core(msa: dict, core: dict) -> dict:
    """DEPRECATED -- broken. Trimming the ref MSA renumbers residue
    indices, which breaks the per-residue-index pair-set logic inside
    sp_score / tc_score in tkfmixdom: pi-th non-gap residue in the
    trimmed ref no longer matches pi-th non-gap residue in the
    untrimmed pred MSA.

    The right "strict-core" implementation is to count only ref pairs
    where both residue endpoints are core (uppercase), without trimming
    the ref. That requires a strict-aware sp_score variant; we have not
    written one. For now, drop --strict-core and use non-strict scoring,
    which matches the most common literature convention.

    Kept here as a tombstone in case someone tries to re-enable
    --strict-core; the call site raises rather than producing wrong
    numbers."""
    raise NotImplementedError(
        "--strict-core is broken; trimming the ref MSA renumbers "
        "residue indices and breaks sp_score's per-residue-index pair-"
        "set logic. Drop the flag and use non-strict scoring, OR fix "
        "by writing a sp_score_strict that filters ref pairs by core "
        "membership without trimming.")


# ============================================================================
# Method runners
# ============================================================================

# --- Per-benchmark shared compute caches -----------------------------------
# All TKF-DP-aware methods share these inputs across a single benchmark:
#   pair_post, pair_taus  : output of compute_pairwise_posteriors
#   boost_states_by_tag   : output of build_boost_state for each model_tag
# The driver loop builds them lazily on demand and passes them as kwargs.
# Methods that don't need a particular cache ignore it via **_.

def _get_pair_post(seqs, cache):
    """Lazy-compute (pair_post, pair_taus); store in cache dict so all
    methods on the same benchmark share the result."""
    if "pair_post" in cache:
        return cache["pair_post"], cache["pair_taus"]
    Q_lg, pi_lg = rate_matrix_lg()
    n = len(seqs)
    pairs = select_pairs_full(n)
    pair_post, pair_taus = compute_pairwise_posteriors(
        seqs, pairs, model='tkf92', Q=Q_lg, pi=pi_lg)
    cache["pair_post"] = pair_post; cache["pair_taus"] = pair_taus
    cache["Q_lg"] = Q_lg; cache["pi_lg"] = pi_lg
    return pair_post, pair_taus


def _get_boost_state(seqs, state, model_tag, cache):
    """Lazy-compute boost_states for a given model. Cached per model_tag
    so coupled and coupled_b sharing the same model only build once."""
    boost_cache = cache.setdefault("boost_states_by_tag", {})
    if model_tag in boost_cache:
        return boost_cache[model_tag]
    from tkfdp.coupled_annealing import build_boost_state                       # type: ignore
    pair_post, pair_taus = _get_pair_post(seqs, cache)
    names = list(seqs.keys())
    seqs_int = [np.asarray(seqs[nm]) for nm in names]
    pair_post_np = {k: np.asarray(v) for k, v in pair_post.items()}
    bs = build_boost_state(pair_post_np, pair_taus, seqs_int, state)
    boost_cache[model_tag] = bs
    return bs


def run_baseline_fsa(seqs: dict, _bm_cache=None, **_) -> dict:
    """Baseline FSA: TKF92 + sequence_annealing on shared pair_post."""
    if _bm_cache is None:
        _bm_cache = {}
    pair_post, _ = _get_pair_post(seqs, _bm_cache)
    n = len(seqs); seq_lens = [len(seqs[k]) for k in seqs]
    names = list(seqs.keys())
    col_assignments, msa_length = sequence_annealing(
        n, seq_lens, dict(pair_post), n_iterations=5, verbose=False)
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def run_tkfdp_precorr(seqs: dict, state, alpha_z: float = 100.0,
                          _bm_cache=None, **_) -> dict:
    """FSA + TKF-DP pre-correction via fb_rerun. Uses shared pair_post +
    pair_taus; the boost-corrected per-pair Q is computed afresh per
    method (depends on alpha_z which is method-specific)."""
    if _bm_cache is None:
        _bm_cache = {}
    pair_post, pair_taus = _get_pair_post(seqs, _bm_cache)
    Q_lg, pi_lg = _bm_cache["Q_lg"], _bm_cache["pi_lg"]
    names = list(seqs.keys()); n = len(names)
    seq_lens = [len(seqs[k]) for k in names]
    pair_post_corr = {}
    for (i, j), Q in pair_post.items():
        x_arr = np.asarray(seqs[names[i]]); y_arr = np.asarray(seqs[names[j]])
        Q_corr = _correct_pair_via_fb_rerun(
            x_arr, y_arr, np.asarray(Q), float(pair_taus[(i, j)]),
            0.02, 0.05, 0.5, Q_lg, pi_lg, state, alpha_z=alpha_z,
        )
        pair_post_corr[(i, j)] = jnp.asarray(Q_corr)
    col_assignments, msa_length = sequence_annealing(
        n, seq_lens, pair_post_corr, n_iterations=5, verbose=False)
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def _run_tkfdp_coupled_inner(seqs, state, model_tag, scoring_mode, q_min,
                                  mu_min, max_pairs_per_anchor, n_anneal_iters,
                                  prior_coup, lambda_pair, _bm_cache):
    from tkfdp.coupled_annealing import coupled_sequence_annealing             # type: ignore
    if _bm_cache is None:
        _bm_cache = {}
    pair_post, pair_taus = _get_pair_post(seqs, _bm_cache)
    boost_states = _get_boost_state(seqs, state, model_tag, _bm_cache)
    names = list(seqs.keys()); n = len(names)
    seq_lens = [len(seqs[k]) for k in names]
    pair_post_np = {k: np.asarray(v) for k, v in pair_post.items()}
    col_assignments, msa_length = coupled_sequence_annealing(
        n, seq_lens, pair_post_np, boost_states=boost_states,
        n_iterations=n_anneal_iters, q_min=q_min, mu_min=mu_min,
        max_pairs_per_anchor=max_pairs_per_anchor,
        scoring_mode=scoring_mode,
        prior_coup=prior_coup, lambda_pair=lambda_pair,
        verbose=False,
    )
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def run_tkfdp_coupled(seqs: dict, state, q_min: float = 0.1,
                          mu_min: float = 0.1, max_pairs_per_anchor: int = 32,
                          n_anneal_iters: int = 5, _bm_cache=None,
                          _model_tag: str = "default", **_) -> dict:
    """FSA + TKF-DP coupled-pair greedy annealing (Design A in
    analysis/coupled_fsa_design_alternatives.md): per-quadruple log-M
    boost as multiplicative sqrt(M) on the average TGF weight."""
    return _run_tkfdp_coupled_inner(
        seqs, state, _model_tag, scoring_mode="log_M",
        q_min=q_min, mu_min=mu_min,
        max_pairs_per_anchor=max_pairs_per_anchor,
        n_anneal_iters=n_anneal_iters,
        prior_coup=0.01, lambda_pair=1.0, _bm_cache=_bm_cache,
    )


def run_tkfdp_coupled_b(seqs: dict, state, q_min: float = 0.1,
                            mu_min: float = 0.1, max_pairs_per_anchor: int = 32,
                            n_anneal_iters: int = 5, prior_coup: float = 0.01,
                            lambda_pair: float = 1.0, _bm_cache=None,
                            _model_tag: str = "default", **_) -> dict:
    """FSA + TKF-DP coupled-pair greedy annealing (Design B): additive
    lambda_pair * p_coup bonus on the average TGF weight, with p_coup
    the Bayesian posterior on the column-pair being coupled."""
    return _run_tkfdp_coupled_inner(
        seqs, state, _model_tag, scoring_mode="posterior",
        q_min=q_min, mu_min=mu_min,
        max_pairs_per_anchor=max_pairs_per_anchor,
        n_anneal_iters=n_anneal_iters,
        prior_coup=prior_coup, lambda_pair=lambda_pair, _bm_cache=_bm_cache,
    )


def run_tkfdp_scfg(seqs: dict, state, alpha_z: float = 100.0,
                       chunk_size: int = 8, q_min: float = 0.0,
                       _bm_cache=None, _model_tag: str = "default",
                       **_) -> dict:
    """FSA + TKF-DP F2-SCFG exact 0-or-1-edge correction.

    For each sequence pair (i, j), runs ``f2_scfg.scfg_corrected_posterior``
    to compute the corrected pair posterior Q'_{ij} (the third pathway
    of the TKF-DP postprocessing trade-off, see appendix of main.tex).
    Then feeds Q'_{ij} into the standard FSA assembler.

    Cost is O(L^4) per sequence pair; chunk_size bounds the per-pair
    memory at chunk_size * L^2 floats.
    """
    from tkfdp.f2_scfg import scfg_corrected_posterior                       # type: ignore
    if _bm_cache is None:
        _bm_cache = {}
    pair_post, pair_taus = _get_pair_post(seqs, _bm_cache)
    Q_lg, pi_lg = _bm_cache["Q_lg"], _bm_cache["pi_lg"]
    boost_states = _get_boost_state(seqs, state, _model_tag, _bm_cache)
    names = list(seqs.keys()); n = len(names)
    seq_lens = [len(seqs[k]) for k in names]
    pair_post_corr = {}
    for (i, j), Q in pair_post.items():
        # Use the clamped sequences from boost_state to match the boost's
        # 20-letter alphabet convention; the TKF92 emission machinery
        # accepts wildcards (index 20) but the F2-SCFG boost-tensor lookup
        # demands 0..19.
        bs = boost_states[(i, j)]
        x_arr = bs.x_seq
        y_arr = bs.y_seq
        t = float(pair_taus[(i, j)])
        Q_corr, _, _, _ = scfg_corrected_posterior(
            x_arr, y_arr, t, 0.02, 0.05, 0.5,
            Q_lg, pi_lg, bs,
            alpha_z=alpha_z, q_min=q_min, chunk_size=chunk_size,
        )
        pair_post_corr[(i, j)] = jnp.asarray(Q_corr)
    col_assignments, msa_length = sequence_annealing(
        n, seq_lens, pair_post_corr, n_iterations=5, verbose=False)
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def run_tkfdp_aug_phmm(seqs: dict, state, alpha_z: float = 100.0,
                        _bm_cache=None, _model_tag: str = "default",
                        **_) -> dict:
    """FSA + TKF-DP memory-augmented PHMM exact 0-or-1-edge correction.

    Equivalent to ``run_tkfdp_scfg`` (the F2-SCFG O(L^4) implementation)
    but uses the augmented Pair HMM in O(L^2 * A^2) time. See
    ``src/tkfdp/aug_phmm.py``.
    """
    from tkfdp.aug_phmm import aug_phmm_corrected_posterior              # type: ignore
    if _bm_cache is None:
        _bm_cache = {}
    pair_post, pair_taus = _get_pair_post(seqs, _bm_cache)
    Q_lg, pi_lg = _bm_cache["Q_lg"], _bm_cache["pi_lg"]
    boost_states = _get_boost_state(seqs, state, _model_tag, _bm_cache)
    names = list(seqs.keys()); n = len(names)
    seq_lens = [len(seqs[k]) for k in names]
    pair_post_corr = {}
    for (i, j), Q in pair_post.items():
        bs = boost_states[(i, j)]
        x_arr = bs.x_seq
        y_arr = bs.y_seq
        t = float(pair_taus[(i, j)])
        Q_corr, _, _, _ = aug_phmm_corrected_posterior(
            x_arr, y_arr, t, 0.02, 0.05, 0.5,
            Q_lg, pi_lg, bs, alpha_z=alpha_z, q_min=0.0,
        )
        pair_post_corr[(i, j)] = jnp.asarray(Q_corr)
    col_assignments, msa_length = sequence_annealing(
        n, seq_lens, pair_post_corr, n_iterations=5, verbose=False)
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def run_tkfdp_aug_antidiag(seqs: dict, state, alpha_z: float = 100.0,
                            _bm_cache=None, _model_tag: str = "default",
                            **_) -> dict:
    """FSA + TKF-DP memory-augmented PHMM with antidiagonal-wavefront DP.

    Mathematically identical to ``run_tkfdp_aug_phmm`` (same Q'_{ij},
    same L_exact); the only difference is the DP traversal order. The
    antidiagonal version processes O(min(Lx, Ly)) cells per scan step
    in parallel via vmap, giving better GPU throughput at large L.
    See ``src/tkfdp/aug_phmm_antidiag.py``.
    """
    from tkfdp.aug_phmm_antidiag import (                                # type: ignore
        aug_phmm_antidiag_corrected_posterior,
    )
    if _bm_cache is None:
        _bm_cache = {}
    pair_post, pair_taus = _get_pair_post(seqs, _bm_cache)
    Q_lg, pi_lg = _bm_cache["Q_lg"], _bm_cache["pi_lg"]
    boost_states = _get_boost_state(seqs, state, _model_tag, _bm_cache)
    names = list(seqs.keys()); n = len(names)
    seq_lens = [len(seqs[k]) for k in names]
    pair_post_corr = {}
    for (i, j), Q in pair_post.items():
        bs = boost_states[(i, j)]
        x_arr = bs.x_seq
        y_arr = bs.y_seq
        t = float(pair_taus[(i, j)])
        Q_corr, _, _, _ = aug_phmm_antidiag_corrected_posterior(
            x_arr, y_arr, t, 0.02, 0.05, 0.5,
            Q_lg, pi_lg, bs, alpha_z=alpha_z, q_min=0.0,
        )
        pair_post_corr[(i, j)] = jnp.asarray(Q_corr)
    col_assignments, msa_length = sequence_annealing(
        n, seq_lens, pair_post_corr, n_iterations=5, verbose=False)
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def run_tkfdp_mcmc(seqs: dict, state, alpha_z: float = 100.0,
                   mcmc_n_sweeps: int = 1000, mcmc_n_burnin: int = 200,
                   mcmc_n_chains: int = 1, mcmc_k_max: int = -1,
                   mcmc_seed: int = 0,
                   mcmc_alpha_z_init: float = -1.0,
                   mcmc_alpha_z_final: float = -1.0,
                   mcmc_anneal_fraction: float = 0.0,
                   mcmc_alpha_z_ladder: str = "",
                   mcmc_swap_every: int = 10,
                   _bm_cache=None, _model_tag: str = "default",
                   **_) -> dict:
    """FSA + TKF-DP MCMC sampler from the infinite Pair HMM.

    Pure-Gibbs path resample between adjacent edge anchors + MH edge
    add/remove with per-edge weight eps = 1/alpha_z.

    Three modes (in order of preference for difficult-to-mix cases):

      (a) DEFAULT (single chain):
          mcmc_alpha_z_ladder == "" and mcmc_anneal_fraction == 0
          Single chain at the target alpha_z.

      (b) Simulated annealing on alpha_z:
          mcmc_alpha_z_init > 0, mcmc_anneal_fraction > 0
          Single chain; alpha_z ramps from init to final over the first
          `anneal_fraction * n_sweeps` sweeps, then held at final.

      (c) Replica exchange on alpha_z:
          mcmc_alpha_z_ladder = "100,200,500,1000,5000,50000"
          K parallel chains, one per ladder rung. Cold rung = min(ladder)
          is the target. Hot rungs explore freely. Swaps proposed every
          mcmc_swap_every sweeps. Cold-rung Q' is returned.
    """
    from tkfdp.mcmc_infinite_phmm import mcmc_corrected_posterior  # type: ignore
    if _bm_cache is None:
        _bm_cache = {}
    pair_post, pair_taus = _get_pair_post(seqs, _bm_cache)
    Q_lg, pi_lg = _bm_cache["Q_lg"], _bm_cache["pi_lg"]
    boost_states = _get_boost_state(seqs, state, _model_tag, _bm_cache)
    names = list(seqs.keys()); n = len(names)
    seq_lens = [len(seqs[k]) for k in names]
    pair_post_corr = {}
    a0 = None if mcmc_alpha_z_init <= 0 else float(mcmc_alpha_z_init)
    af = None if mcmc_alpha_z_final <= 0 else float(mcmc_alpha_z_final)
    ladder = (None if not mcmc_alpha_z_ladder
              else [float(x) for x in mcmc_alpha_z_ladder.split(",")])
    for (i, j), Q in pair_post.items():
        bs = boost_states[(i, j)]
        x_arr = bs.x_seq
        y_arr = bs.y_seq
        t = float(pair_taus[(i, j)])
        Q_corr, _, _, _, _ = mcmc_corrected_posterior(
            x_arr, y_arr, t, 0.02, 0.05, 0.5,
            Q_lg, pi_lg, bs, alpha_z=alpha_z,
            n_sweeps=mcmc_n_sweeps, n_burnin=mcmc_n_burnin,
            n_chains=mcmc_n_chains, k_max=mcmc_k_max,
            seed=mcmc_seed,
            alpha_z_init=a0, alpha_z_final=af,
            anneal_fraction=mcmc_anneal_fraction,
            alpha_z_ladder=ladder, swap_every=mcmc_swap_every,
        )
        pair_post_corr[(i, j)] = jnp.asarray(Q_corr)
    col_assignments, msa_length = sequence_annealing(
        n, seq_lens, pair_post_corr, n_iterations=5, verbose=False)
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def run_tkfdp_aug_2edge(seqs: dict, state, alpha_z: float = 100.0,
                         _bm_cache=None, _model_tag: str = "default",
                         **_) -> dict:
    """FSA + TKF-DP memory-augmented PHMM exact 0-or-1-or-2-edge correction.

    Generalises ``run_tkfdp_aug_phmm`` to allow up to two coupled
    column-pairs per alignment via the size-{0, 1, 2}-truncated
    Ewens (CRP-truncated) prior. See ``src/tkfdp/aug_phmm_2edge.py``.

    Cost is O(L^2 * |T|) where the tag space |T| ~ A^4 / 2 ~ 80k for
    A=20 -- about 200x more memory than the 1-edge variant. Practical
    only on short sequences without the antidiagonal optimisation.
    """
    from tkfdp.aug_phmm_2edge import aug_phmm_2edge_corrected_posterior  # type: ignore
    if _bm_cache is None:
        _bm_cache = {}
    pair_post, pair_taus = _get_pair_post(seqs, _bm_cache)
    Q_lg, pi_lg = _bm_cache["Q_lg"], _bm_cache["pi_lg"]
    boost_states = _get_boost_state(seqs, state, _model_tag, _bm_cache)
    names = list(seqs.keys()); n = len(names)
    seq_lens = [len(seqs[k]) for k in names]
    pair_post_corr = {}
    for (i, j), Q in pair_post.items():
        bs = boost_states[(i, j)]
        x_arr = bs.x_seq
        y_arr = bs.y_seq
        t = float(pair_taus[(i, j)])
        Q_corr, _, _, _ = aug_phmm_2edge_corrected_posterior(
            x_arr, y_arr, t, 0.02, 0.05, 0.5,
            Q_lg, pi_lg, bs, alpha_z=alpha_z, q_min=0.0,
        )
        pair_post_corr[(i, j)] = jnp.asarray(Q_corr)
    col_assignments, msa_length = sequence_annealing(
        n, seq_lens, pair_post_corr, n_iterations=5, verbose=False)
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def run_muscle(seqs: dict, **_) -> dict:
    return _run_external("muscle", ["-align", "{in}", "-output", "{out}"], seqs)


def run_mafft(seqs: dict, **_) -> dict:
    return _run_external("mafft", ["--auto", "--quiet", "{in}"], seqs,
                            mafft_stdout=True)


def _run_external(prog: str, argv_template: list[str], seqs: dict,
                       mafft_stdout: bool = False) -> dict:
    """Run an external aligner on a temp FASTA, return parsed MSA dict."""
    from tkfmixdom.jax.util.io import INT_TO_AA
    if shutil.which(prog) is None:
        raise FileNotFoundError(
            f"{prog} not on PATH. Run scripts/fetch_aligners.sh.")
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.fa"; out_path = Path(td) / "out.fa"
        with open(in_path, "w") as f:
            for name, arr in seqs.items():
                seq = "".join(INT_TO_AA.get(int(i), "X") for i in arr)
                f.write(f">{name}\n{seq}\n")
        argv = [prog] + [a.format(**{"in": str(in_path), "out": str(out_path)})
                                  for a in argv_template]
        if mafft_stdout:
            with open(out_path, "w") as f:
                subprocess.run(argv, stdout=f, check=True)
        else:
            subprocess.run(argv, check=True, capture_output=True)
        out = {}
        for name, seq in read_fasta(str(out_path)):
            row = np.full(len(seq), -1, dtype=np.int32)
            for k, c in enumerate(seq):
                if c not in '.-':
                    row[k] = AA_TO_INT.get(c.upper(), 20)
            out[name] = row
    return out


METHODS = {
    "baseline_fsa": run_baseline_fsa,
    "tkfdp_precorr": run_tkfdp_precorr,
    "tkfdp_coupled": run_tkfdp_coupled,        # Design A (per-quadruple log-M)
    "tkfdp_coupled_b": run_tkfdp_coupled_b,    # Design B (per-pair posterior + lambda_pair)
    "tkfdp_scfg": run_tkfdp_scfg,              # F2-SCFG exact 0-or-1-edge
    "tkfdp_aug": run_tkfdp_aug_phmm,           # Aug-PHMM row-scan, O(L^2 A^2)
    "tkfdp_aug_antidiag": run_tkfdp_aug_antidiag,  # Aug-PHMM antidiagonal-wavefront
    "tkfdp_aug_2edge": run_tkfdp_aug_2edge,    # Aug-PHMM 0-or-1-or-2-edge, O(L^2 |T|), |T| ~ A^4 / 2
    "tkfdp_mcmc": run_tkfdp_mcmc,              # MCMC infinite Pair HMM, O(L^4) setup + O(L) per sweep
    "muscle": run_muscle,
    "mafft": run_mafft,
}

# Methods that consume a TKF-DP state; the rest are model-independent and
# their result rows carry model="-".
MODEL_DEPENDENT = {"tkfdp_precorr", "tkfdp_coupled", "tkfdp_coupled_b",
                   "tkfdp_scfg", "tkfdp_aug", "tkfdp_aug_antidiag",
                   "tkfdp_aug_2edge", "tkfdp_mcmc"}

CSV_FIELDS = ["benchmark", "method", "model", "sp", "tc",
                  "seconds", "n_seqs", "error"]


# ============================================================================
# Idempotence: load existing rows, dedupe key (benchmark, method, model)
# ============================================================================

def load_existing(csv_path: Path) -> tuple[list, set]:
    """Read existing CSV (if any). Return (rows, done_keys) where:
      - rows: deduplicated to the most-recent row per (benchmark, method,
        model) cell (last write wins -- so a successful retry supersedes
        a previous error row).
      - done_keys: set of cells that already have a finite, non-error SP
        score in their latest row. Cells that previously errored or NaN-ed
        are treated as MISSING -- the next invocation retries them
        automatically (idempotent gap-filling).

    The dedup makes the in-memory rows match what a clean rewrite of the
    CSV would look like, so the run-end summary's failure count reflects
    *current* cell state rather than historical attempts."""
    if not csv_path.exists():
        return [], set()
    by_cell: dict[tuple[str, str, str], dict] = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            key = (r["benchmark"], r["method"], r.get("model", "-"))
            by_cell[key] = r       # last row for the cell wins
    rows = list(by_cell.values())
    done = set()
    for r in rows:
        if not r.get("error", "").strip() and r.get("sp", ""):
            try:
                sp = float(r["sp"])
                if not np.isnan(sp):
                    done.add((r["benchmark"], r["method"],
                                   r.get("model", "-")))
            except ValueError:
                pass
    return rows, done


def rewrite_csv(csv_path: Path, rows: list) -> None:
    """Atomically rewrite the CSV deduped to one row per cell. Called at
    end of run to clean up after retries that left stale error rows."""
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    os.replace(tmp, csv_path)


def append_row(csv_path: Path, row: dict, write_header: bool) -> None:
    """Append a single row to the CSV (idempotent: callers pass
    write_header=True only on the first write to a fresh file)."""
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        # Coerce values to plain (str|float|int|None) for CSV portability.
        out = {k: row.get(k, "") for k in CSV_FIELDS}
        w.writerow(out)


def parse_checkpoints(specs: list[str]) -> list[tuple[str, Path]]:
    """Parse `--checkpoints "tag:path" "tag2:path2" ...`. The tag is
    used as the model column in the result CSV; the path is loaded with
    load_minimal_state. If a spec has no colon the tag is the path's
    basename."""
    out = []
    for s in specs:
        if ":" in s:
            tag, path = s.split(":", 1)
        else:
            path = s
            tag = Path(path).parent.name or Path(path).name
        out.append((tag, Path(path)))
    return out


# ============================================================================
# Driver
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bali-root", type=Path, default=default_balibase_root())
    ap.add_argument("--bench", default="bali3",
                       help="Subdirectory under bench1.0/ (bali3, prefab4, ox, sabre, ...)")
    ap.add_argument("--checkpoints", nargs="+",
                       default=["K4_emwarm:results/exp2_v2_K4_top1000_tsb_emwarm/_best_chkpt"],
                       help="One or more 'tag:path' specs. Each TKF-DP "
                            "model is run alongside the model-independent "
                            "methods (baseline / muscle / mafft) and "
                            "tagged in the 'model' CSV column.")
    ap.add_argument("--out-dir", type=Path,
                       default=Path("results/balibase_eval"))
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--alpha-z", type=float, default=100.0)
    ap.add_argument("--n-benchmarks", type=int, default=None,
                       help="Cap on number of benchmarks (debug; default all)")
    ap.add_argument("--strict-core", action="store_true",
                       help="BAliBASE convention: score only on all-core columns of the ref.")
    # MCMC-specific config (used only by tkfdp_mcmc).
    ap.add_argument("--mcmc-n-sweeps", type=int, default=1000)
    ap.add_argument("--mcmc-n-burnin", type=int, default=200)
    ap.add_argument("--mcmc-n-chains", type=int, default=1)
    ap.add_argument("--mcmc-k-max", type=int, default=-1,
                       help="Max edges (negative = infinity)")
    ap.add_argument("--mcmc-seed", type=int, default=0)
    ap.add_argument("--mcmc-alpha-z-init", type=float, default=-1.0,
                       help="Annealing start alpha_z; pass <=0 to disable "
                            "annealing.")
    ap.add_argument("--mcmc-alpha-z-final", type=float, default=-1.0,
                       help="Annealing end alpha_z; pass <=0 to use --alpha-z.")
    ap.add_argument("--mcmc-anneal-fraction", type=float, default=0.0,
                       help="Fraction of n_sweeps to anneal over; 0 = no "
                            "annealing.")
    ap.add_argument("--mcmc-alpha-z-ladder", default="",
                       help="Replica-exchange alpha_z ladder, comma-separated. "
                            "Smallest = cold (target). Empty = single chain. "
                            "Example: 100,200,500,1000,5000,50000")
    ap.add_argument("--mcmc-swap-every", type=int, default=10,
                       help="Replica-exchange swap proposal frequency in sweeps.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = args.bali_root / args.bench / "in"
    ref_dir = args.bali_root / args.bench / "ref"
    bench_ids = sorted(p.name for p in in_dir.iterdir() if p.is_file())
    if args.n_benchmarks:
        bench_ids = bench_ids[:args.n_benchmarks]
    print(f"BAliBASE eval on {args.bench}: {len(bench_ids)} benchmarks")

    # Load all requested TKF-DP models up front (small; ~KB each).
    needs_models = any(m in MODEL_DEPENDENT for m in args.methods)
    models: list[tuple[str, object]] = []
    if needs_models:
        for tag, path in parse_checkpoints(args.checkpoints):
            print(f"  Loading TKF-DP model '{tag}' from {path}")
            try:
                state = load_minimal_state(path)
                models.append((tag, state))
                print(f"    K_c={state.K_c}, "
                        f"K_H={state.potts_dp.atoms.shape[0]}, "
                        f"side potentials="
                        f"{'on' if state.potts_dp.h_pairs is not None else 'off'}")
            except Exception as e:
                print(f"    SKIP: {e}")
    if not models:
        models = [("-", None)]   # so the loop below runs once for indep methods

    # Idempotence: read existing rows, build the done-set. Cells with a
    # finite SP are considered done; cells that errored or NaN-ed (or
    # are absent entirely) are considered missing and will be re-run.
    csv_path = args.out_dir / f"{args.bench}_results.csv"
    existing_rows, done = load_existing(csv_path)
    write_header = not existing_rows
    if existing_rows:
        n_err = sum(1 for r in existing_rows if r.get("error", "").strip())
        print(f"Found {len(existing_rows)} existing rows: "
                f"{len(done)} successful (skipping), "
                f"{n_err} errored (will retry).")

    # Build the iteration plan: every (benchmark, method, model_tag) cell.
    plan = []
    for bid in bench_ids:
        for method in args.methods:
            if method in MODEL_DEPENDENT:
                for tag, _state in models:
                    plan.append((bid, method, tag))
            else:
                plan.append((bid, method, "-"))
    todo = [c for c in plan if c not in done]
    print(f"Plan: {len(plan)} cells total, {len(todo)} to run, "
            f"{len(plan) - len(todo)} already done.")

    state_by_tag = {tag: st for tag, st in models}
    last_seqs: dict | None = None; last_ref_for_score: dict | None = None
    last_bid: str | None = None
    # Per-benchmark shared-compute cache: pair_post + pair_taus + boost_states
    # are computed lazily by the first method that needs them and reused by
    # all subsequent methods on the same benchmark. Reset on each new bid.
    _bm_cache: dict = {}
    # JAX JIT-cache OOM safety: clear compilation caches every N benchmarks
    # so the LLVM in-memory section doesn't grow unbounded across diverse
    # sequence-length bin combinations. With 386 BAliBASE 3 benchmarks and
    # sequence lengths from ~30 to ~600, the JIT cache otherwise OOMs at
    # ~17 benchmarks on a 32 GB box.
    CLEAR_CACHE_EVERY = 5
    benchmarks_seen_this_run = 0
    # Index existing rows by cell so that a retry overwrites the previous
    # row for that cell rather than appending a stale duplicate.
    rows_by_cell: dict[tuple[str, str, str], dict] = {
        (r["benchmark"], r["method"], r.get("model", "-")): r
        for r in existing_rows
    }
    for cell_idx, (bid, method, model_tag) in enumerate(todo):
        # Load benchmark inputs once per bid (todo is sorted by plan order
        # = bid then method then model, so consecutive cells share a bid).
        if bid != last_bid:
            try:
                last_seqs = load_seqs(in_dir / bid)
                ref_msa, core = load_ref(ref_dir / bid)
                last_ref_for_score = (restrict_to_core(ref_msa, core)
                                              if args.strict_core else ref_msa)
                last_bid = bid
                _bm_cache.clear()       # new BM => fresh shared cache
                benchmarks_seen_this_run += 1
                # Periodic JIT-cache clear to bound memory.
                if (benchmarks_seen_this_run % CLEAR_CACHE_EVERY == 0):
                    jax.clear_caches()
            except Exception as e:
                print(f"  [{cell_idx + 1}/{len(todo)}] {bid}: SKIP load ({e})")
                last_bid = None
                continue
            if len(last_seqs) < 2:
                last_bid = None
                continue
        if last_bid is None:
            continue
        state = state_by_tag.get(model_tag)
        t0 = time.time()
        try:
            pred = METHODS[method](last_seqs, state=state,
                                            alpha_z=args.alpha_z,
                                            _bm_cache=_bm_cache,
                                            _model_tag=model_tag,
                                            mcmc_n_sweeps=args.mcmc_n_sweeps,
                                            mcmc_n_burnin=args.mcmc_n_burnin,
                                            mcmc_n_chains=args.mcmc_n_chains,
                                            mcmc_k_max=args.mcmc_k_max,
                                            mcmc_seed=args.mcmc_seed,
                                            mcmc_alpha_z_init=args.mcmc_alpha_z_init,
                                            mcmc_alpha_z_final=args.mcmc_alpha_z_final,
                                            mcmc_anneal_fraction=args.mcmc_anneal_fraction,
                                            mcmc_alpha_z_ladder=args.mcmc_alpha_z_ladder,
                                            mcmc_swap_every=args.mcmc_swap_every)
            sp = sp_score(pred, last_ref_for_score)
            tc = tc_score(pred, last_ref_for_score)
            err = ""
        except Exception as e:
            sp = float('nan'); tc = float('nan'); err = repr(e)[:120]
        elapsed = time.time() - t0
        row = dict(benchmark=bid, method=method, model=model_tag,
                       sp=sp, tc=tc, seconds=elapsed,
                       n_seqs=len(last_seqs), error=err)
        rows_by_cell[(bid, method, model_tag)] = row
        # Append-as-we-go: crash-safe partial progress. The CSV may
        # accumulate duplicate rows for retried cells; rewrite_csv at
        # end-of-run dedupes (and load_existing on the next invocation
        # also dedupes by last-write-wins).
        append_row(csv_path, row, write_header=write_header)
        write_header = False
        print(f"  [{cell_idx + 1}/{len(todo)}] {bid} / "
                f"{method:14s} / {model_tag:>14s}  "
                f"SP={sp:.3f} TC={tc:.3f} t={elapsed:.1f}s {err}")

    # End-of-run cleanup: dedupe + atomic rewrite.
    rows = list(rows_by_cell.values())
    rewrite_csv(csv_path, rows)
    print(f"\nWrote {csv_path}")

    # Summary by (method, model).
    print(f"\n{'method':>16s} {'model':>16s}  {'mean SP':>8s} "
            f" {'mean TC':>8s}  {'median SP':>9s}  {'n':>4s}  "
            f"{'failures':>8s}")
    keys = sorted({(r["method"], r.get("model", "-")) for r in rows})
    for method, model_tag in keys:
        cell_rows = [r for r in rows if r["method"] == method
                            and r.get("model", "-") == model_tag]
        sps = [float(r["sp"]) for r in cell_rows
                  if not r.get("error", "").strip()
                  and not np.isnan(float(r["sp"]))]
        tcs = [float(r["tc"]) for r in cell_rows
                  if not r.get("error", "").strip()
                  and not np.isnan(float(r["tc"]))]
        n_fail = sum(1 for r in cell_rows if r.get("error", "").strip())
        if sps:
            print(f"{method:>16s} {model_tag:>16s}  "
                    f"{np.mean(sps):8.4f}  {np.mean(tcs):8.4f}  "
                    f"{np.median(sps):9.4f}  {len(sps):4d}  {n_fail:8d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
