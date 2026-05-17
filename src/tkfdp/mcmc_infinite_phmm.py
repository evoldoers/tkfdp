"""MCMC sampler for the infinite Pair HMM (TKF-DP).

This module implements the principled MCMC formulation of the
TKF-DP coevolutionary correction (``main.tex`` sec:infinite-hmm).
The bounded-edge dynamic-programming approximations in
``aug_phmm.py`` (1-edge) and ``aug_phmm_2edge.py`` (2-edge) are
finite truncations of this model; the MCMC sampler is exact in
expectation up to MC error, with no edge-count truncation.

Phase 1 implementation: scan-over-scan (no antidiagonal-wavefront
optimisation). Setup is O(L^4); per-sweep cost is O(L) at convergence.

The algorithm:

1.  **Setup phase** (per sequence pair, cached for the chain):
    -  Build the TKF92 Pair HMM (log_trans, state_types, sub, pi).
    -  Compute the baseline (no-edge) Forward and Backward tables
       alpha and beta in log space.
    -  Compute the partial-Forward tensor F^partial[i, j; k, l]
       (= F_2(i, j; k, l) of f2_scfg.py): the no-edge Pair HMM
       Forward probability summed over alignments that visit Match
       at both anchor cells (i, j) and (k, l).
    -  Pre-tabulate the AA-marginal Potts boost at observed AAs:
       M_obs[i, j, k, l] = M_AA[X_i, Y_j, X_k, Y_l] (dense fp32).

2.  **Initial state**: Viterbi alignment from the no-edge Pair HMM,
    no edges (E = empty).

3.  **MCMC sweep**:
    a.  **Combined segment-resample** (per H7 user spec): partition
        the alignment into 2|E| + 1 segments by the edge anchor
        Match cells (plus virtual start/end), then RESAMPLE each
        inter-edge-anchor segment from the conditional baseline
        Pair HMM Forward distribution. Under Strategy S-1
        (no edges inside the segment, by construction), the
        Hastings ratio collapses to the CRP-prior path-length
        factor (canonical CRP) or 1 (bounded_eps). This is a
        single combined move per sweep.
    b.  **Edge add/remove moves**: standard CRP add and remove
        proposals. Add: pick two unpaired Match cells uniformly
        and propose an edge between them (Q_add-uniform per H2).
        Remove: pick an existing edge uniformly. Hastings ratios
        balance against the canonical CRP prior or bounded_eps
        prior.

The MCMC output is the per-cell running mean Q'[i, j] of the
indicator 1[(i, j) is Match in current alignment], over post-burn-in
sweeps. This is the same Q' that aug_phmm.py / aug_phmm_2edge.py
return analytically (under their respective bounded approximations).

For verification:
- E.1: MCMC under bounded_eps mode + k_max=1 matches aug_phmm to
       within MC standard error.
- E.2: MCMC under bounded_eps mode + k_max=2 matches aug_phmm_2edge
       (modulo the small "non-overlapping nested edges" combinatorial
       difference; passes for L_x, L_y <= 4).
- E.3: at small Lx, Ly: MCMC under canonical CRP matches a brute-force
       enumeration of all (alignment, edge_set) configurations.
- E.4: with M=1, no-data: MCMC's empirical |E|-given-N_M histogram
       matches the analytical CRP marginal.
- E.5: tight-proposal MH simplification check (the load-bearing claim
       that the MH ratio collapses to the CRP-prior path-length factor).

The chain alternates two move types per sweep, both well-defined
on the canonical infinite-HMM target with size-{1,2}-truncated Ewens
prior on edges:

  1.  Path resample (Gibbs, accept rate = 1).  For each pair of
      adjacent edge anchors, draw a new path between them from
      pi_TKF92 conditional on the bounding anchors being Match cells.
      Because edge endpoint cells are FIXED during this step, the
      Potts boost factors are constant; the conditional reduces to
      the baseline TKF92 path conditional, which is exactly what the
      precomputed F_partial traceback samples.

  2.  Edge add / remove (MH).  Standalone proposals to add or remove
      a coupled edge between two unpaired Match cells. The MH ratio
      uses the per-edge prior factor eps = 1/alpha_z and the M boost.

To address the "high-|E| at standard alpha_z" issue (the size-{1,2}
canonical Ewens at alpha_z=100, N_M~75 has E[|E|]~13), the run loop
supports a SIMULATED ANNEALING schedule on alpha_z: ramp from
`alpha_z_init` (often small; favours pairs, helps the chain explore
the edge graph aggressively) up to `alpha_z_final` (large; congeals
into few strong edges) over the first `anneal_fraction` of sweeps,
then sample from the final temperature for the rest.
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Tkfmixdom Pair HMM primitives.
TKFMIXDOM_ROOT = Path.home() / "tkf-mixdom" / "python"
if str(TKFMIXDOM_ROOT) not in sys.path:
    sys.path.insert(0, str(TKFMIXDOM_ROOT))
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm  # noqa: E402
from tkfmixdom.jax.dp.hmm import (                                  # noqa: E402
    pair_hmm_emissions, _pad_to_bin, _pad_seq, _emit_mask,
    _find_e_idx, NEG_INF,
)
from tkfmixdom.jax.core.params import S, M, I, D, E                  # noqa: E402

from .aug_phmm import (build_M_tensor_aa_marginal,                    # noqa: E402
                        build_M_tensor_classmarg)
from .block_likelihoods import (build_singlet_emission,               # noqa: E402
                                 build_M_tensor as build_M_tensor_unified,
                                 empirical_pi_c_from_checkpoint)
from .f2_scfg import (                                               # noqa: E402
    forward_pair_hmm, backward_pair_hmm, _restart_forward_jit,
)

A = 20  # amino acid alphabet


# ---------------------------------------------------------------------------
# JIT-cached primitives (keyed by Lx_pad, Ly_pad so reused across pairs).
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _chunk_mu_M_jit(log_trans, state_types, emit, ai, aj, Lx_pad, Ly_pad):
    """vmap'd restart-Forward; returns mu_M of shape (chunk, Lx_pad+1, Ly_pad+1).

    For each anchor (ai[k], aj[k]), runs the restart-Forward starting from
    M-state at that anchor and returns the M-state slice mu[..., M].
    """
    def kernel(i_a, j_a):
        mu = _restart_forward_jit(
            log_trans, state_types, emit, Lx_pad, Ly_pad, i_a, j_a)
        return mu[:, :, M]
    return jax.vmap(kernel)(ai, aj)


@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _chunk_mu_full_jit(log_trans, state_types, emit, ai, aj, Lx_pad, Ly_pad):
    """vmap'd restart-Forward; returns full 5-state mu of shape
    (chunk, Lx_pad+1, Ly_pad+1, n_states). Used by the cache-prepopulate
    path so the per-anchor scores match what the on-demand path stores."""
    def kernel(i_a, j_a):
        return _restart_forward_jit(
            log_trans, state_types, emit, Lx_pad, Ly_pad, i_a, j_a)
    return jax.vmap(kernel)(ai, aj)


# ===========================================================================
# Phase-B JIT'd stochastic-traceback kernel.
#
# Profiling identified _stochastic_traceback_segment as 84% of sampler time,
# dominated by ~32-48 ms/call of Python-interpreter overhead in the per-cell
# loop body (1700 calls per pair * ~22 cells/call * ~1.5 ms/cell of Python
# branching, tuple/list manipulation, closure invocation, np reductions).
#
# This kernel reimplements the inner backward walk as a jax.lax.scan over a
# fixed-size max_iter (bin'd geometrically for JIT cache reuse). The walk
# terminates early via a `done` mask carried in the scan state -- Patrick
# Kidger's bounded_while_loop pattern (jax-ml/jax#8239). Iterations after
# termination produce sentinel cells (state == -1) which the host filters.
#
# Targets the CPU backend: the score slabs are np.ndarray (host memory) and
# the kernel work is O(max_iter * n_states) per call, i.e. a few hundred
# flops -- way too small to benefit from GPU, and host->device transfer
# of the (Lx_pad+1, Ly_pad+1, n_states) score slab would dominate. The
# kernel is compiled once per (Lx_pad, Ly_pad, max_iter) bin signature
# and reused across all anchors / sweeps / rungs / replicates.
# ===========================================================================


def _select_cpu_device():
    """Cache the CPU device for jit pinning. None on first call if jax has
    not initialised yet; lazy-resolved on first kernel call."""
    try:
        return jax.devices('cpu')[0]
    except (RuntimeError, IndexError):
        return None


@partial(jax.jit,
         static_argnames=('max_iter', 'n_states', 'is_start_anchor'),
         backend='cpu')
def _scan_traceback_kernel_jit(
        rng_key,            # PRNGKey
        scores,             # fp32 (Lx_pad+1, Ly_pad+1, n_states); padding == -inf
        log_trans,          # fp32 (n_states, n_states)
        stop_i, stop_j,     # int32: entry-anchor stop coordinates
        init_state,         # int32: state at the cell where the walk starts
        init_i, init_j,     # int32: cell where the walk starts
        max_iter,           # static int -- geomspace bin
        n_states,           # static int (== 5 in current model)
        is_start_anchor):   # static bool
    """Backward stochastic traceback as a lax.scan over fixed max_iter.

    Mirrors the semantics of the original Python _stochastic_traceback_segment
    body (lines ~664-687): walks backward through cells, sampling predecessor
    states. Terminates without emitting the entry-anchor/start-corner cell.

    Returns (path_states, path_is, path_js): three (max_iter,) int arrays.
    Iterations on or after termination produce sentinel values (-1) which the
    host wrapper filters.
    """
    UNREACHABLE_THRESH = -1e25
    S_S, M_S, I_S, D_S = 0, 1, 2, 3

    def step(carry, key):
        cur_state, i_cur, j_cur, done = carry
        # Compile-time-distinct termination check (is_start_anchor is static)
        if is_start_anchor:
            at_stop = ((cur_state == S_S)
                       & (i_cur == jnp.int32(0))
                       & (j_cur == jnp.int32(0)))
        else:
            at_stop = ((cur_state == M_S)
                       & (i_cur == stop_i)
                       & (j_cur == stop_j))
        # Defensive: if we somehow land in S without being at (0,0), stop too
        terminate = at_stop | (cur_state == S_S)
        new_done = done | terminate
        # Output cell (sentinel when masked)
        emit_mask = ~new_done
        out_state = jnp.where(emit_mask, cur_state, jnp.int32(-1))
        out_i = jnp.where(emit_mask, i_cur, jnp.int32(-1))
        out_j = jnp.where(emit_mask, j_cur, jnp.int32(-1))
        # Predecessor cell direction from cur_state
        i_prev = jnp.where(cur_state == M_S, i_cur - 1,
                  jnp.where(cur_state == I_S, i_cur,
                   jnp.where(cur_state == D_S, i_cur - 1, i_cur)))
        j_prev = jnp.where(cur_state == M_S, j_cur - 1,
                  jnp.where(cur_state == I_S, j_cur - 1,
                   jnp.where(cur_state == D_S, j_cur, j_cur)))
        # Clip to valid range for safe gather (values not used when done)
        Lx_max = jnp.int32(scores.shape[0] - 1)
        Ly_max = jnp.int32(scores.shape[1] - 1)
        i_prev_safe = jnp.clip(i_prev, 0, Lx_max)
        j_prev_safe = jnp.clip(j_prev, 0, Ly_max)
        # Logits over predecessor states, conditioned on successor = cur_state
        tlogits = scores[i_prev_safe, j_prev_safe, :] + log_trans[:, cur_state]
        # Detect unreachable: if max logit too low, freeze
        unreachable = jnp.max(tlogits) < UNREACHABLE_THRESH
        s_next = jax.random.categorical(key, tlogits)
        # Update carry: freeze on termination/unreachable
        freeze = new_done | unreachable
        next_state = jnp.where(freeze, cur_state, s_next.astype(jnp.int32))
        next_i = jnp.where(freeze, i_cur, i_prev)
        next_j = jnp.where(freeze, j_cur, j_prev)
        return (next_state, next_i, next_j, freeze), (out_state, out_i, out_j)

    keys = jax.random.split(rng_key, max_iter)
    init_carry = (jnp.int32(init_state),
                  jnp.int32(init_i),
                  jnp.int32(init_j),
                  jnp.bool_(False))
    _, (path_states, path_is, path_js) = jax.lax.scan(
        step, init_carry, keys)
    # Pack into one (3, max_iter) array so the host pays a single device
    # sync per traceback rather than three.
    return jnp.stack([path_states, path_is, path_js], axis=0)


def _sample_state_at_np(scores, log_trans, e_idx, i_pos, j_pos, succ_state,
                        n_states, rng):
    """Pure-numpy single-step predecessor-state categorical (Python side).
    Used only for the initial sample at the segment boundary; the bulk
    backward walk is jit'd. Returns -1 if unreachable."""
    UNREACHABLE_THRESH = -1e25
    if succ_state == E:
        tlogits = scores[i_pos, j_pos, :] + log_trans[:, e_idx]
    else:
        tlogits = scores[i_pos, j_pos, :] + log_trans[:, succ_state]
    msk = (np.isfinite(tlogits) & (tlogits > UNREACHABLE_THRESH))
    if not msk.any():
        return -1
    m_max = tlogits[msk].max()
    w = np.where(msk, np.exp(tlogits - m_max), 0.0)
    s = w.sum()
    if s <= 0:
        return -1
    w = w / s
    return int(rng.choice(n_states, p=w))


# ===========================================================================
# Setup-phase data structures
# ===========================================================================


@dataclass
class MCMCSetup:
    """Pre-computed per-pair tables consumed by the sampler.

    Attributes:
      Lx, Ly:           real sequence lengths.
      Lx_pad, Ly_pad:   geometric-bin-padded lengths.
      x_seq, y_seq:     real (unpadded) AA sequences (np.int32).
      x_pad, y_pad:     padded AA sequences (np.int32).
      log_trans, state_types, sub_matrix, pi_out: TKF92 Pair HMM tables.
      log_alpha, log_beta: padded baseline Forward / Backward tables
                           (Lx_pad+1, Ly_pad+1, 5).
      F_partial:        dense (Lx_pad+1, Ly_pad+1, Lx_pad+1, Ly_pad+1)
                        F_2-style partial-Forward in LOG space at the
                        Match-state of both anchors. Includes both
                        anchors' M-emissions. Padded / unreachable
                        entries are NEG_INF.
      M_obs:            dense (Lx_pad+1, Ly_pad+1, Lx_pad+1, Ly_pad+1)
                        AA-marginal log-Potts-boost at observed AAs:
                        M_obs[i, j, k, l] = log M_AA[X_i, Y_j, X_k, Y_l].
                        For positions outside the real region we use 0
                        (log 1 = no boost), which is harmless because
                        edges only point at valid Match cells.
      log_F0:           log of the baseline partition function.
      mu_cache:         dict-keyed cache of per-anchor restart-Forward
                        full-state tensor mu[(i_a, j_a)] of shape
                        (Lx_pad+1, Ly_pad+1, 5). Populated on demand
                        and reused across sweeps; the cache is bounded
                        by the number of edge anchors in flight at any
                        time (bounded by 2 * |E_max| ~ 2 * alpha_z).

    Sizes at L=200: F_partial and M_obs are each ~6.4 GB fp32. Use
    smaller L for development/testing; at L=100 each is 400 MB.
    """

    Lx: int
    Ly: int
    Lx_pad: int
    Ly_pad: int
    x_seq: np.ndarray
    y_seq: np.ndarray
    x_pad: jnp.ndarray
    y_pad: jnp.ndarray
    log_trans: jnp.ndarray
    state_types: jnp.ndarray
    sub_matrix: jnp.ndarray
    pi_out: jnp.ndarray
    emit: jnp.ndarray         # padded log-emission table (Lx_pad+1, Ly_pad+1, 5)
    log_alpha: jnp.ndarray
    log_beta: jnp.ndarray
    F_partial: np.ndarray     # log space; padded; (Lx_pad+1)^2 (Ly_pad+1)^2
    M_obs: np.ndarray         # log space; padded; (Lx_pad+1)^2 (Ly_pad+1)^2
    log_F0: float
    mu_cache: dict = field(default_factory=dict)

    # Phase-A invariant cache. These are the np.asarray-converted versions of
    # JAX-side tables that the sampler hot loops would otherwise re-convert on
    # every call. Profiling identified _stochastic_traceback_segment as 84% of
    # total sampler time, with redundant `np.asarray(setup.log_trans)` and
    # `_find_e_idx(setup.state_types)` calls inside the hot loop. Populated
    # once at the end of precompute_partial_forward.
    log_trans_np: np.ndarray = None
    state_types_np: np.ndarray = None
    log_alpha_np: np.ndarray = None
    log_beta_np: np.ndarray = None
    e_idx: int = -1

    # Edge prior concentration (CURRENT effective value; the run loop
    # mutates this when annealing is on). The prior is the size-{1,2}-
    # truncated Ewens / equivalently a per-edge weight eps = 1/alpha_z;
    # large alpha_z discourages edges, small alpha_z favours them.
    alpha_z: float = 100.0

    # Setup phase wall-time (informational).
    setup_seconds: float = 0.0
    # Per-phase setup breakdown: keys include 'pair_hmm_build', 'forward_backward',
    # 'm_obs_gather', 'table_convert'. All in seconds. Total = setup_seconds.
    setup_breakdown: dict = field(default_factory=dict)
    # Live counters mutated by the sampler hot loop (restart-Forward cache).
    # Reset to zero at run start in run_replica_exchange_chain / run_mcmc_chain.
    rf_seconds: float = 0.0       # cumulative wall time in _restart_forward_jit
    rf_n_misses: int = 0          # mu_cache misses (one restart-Forward per miss)
    rf_n_hits: int = 0            # mu_cache hits (no restart-Forward; just lookup)
    tb_seconds: float = 0.0       # cumulative wall time in inner traceback
    tb_n_calls: int = 0           # number of inner-traceback calls


def precompute_partial_forward(
        x_seq: np.ndarray, y_seq: np.ndarray, t: float,
        ins_rate: float, del_rate: float, ext: float,
        Q_lg, pi_lg, boost_state,
        alpha_z: float = 100.0,
        prepop_top_k: int = -1,
        prepop_chunk: int = 256,
        prepop_mem_budget_mib: float = 2048.0) -> MCMCSetup:
    """Build the MCMCSetup for one sequence pair.

    Setup cost: O(L^4) flops for F_partial; O(L^2) for alpha/beta;
    O(L^4) for M_obs (just gather, fast).

    Memory: ~2 * (L_pad+1)^4 fp32 floats. At L=100 padded ~128, that
    is 2 * 128^4 * 4 = 2.1 GB total. Use small L for dev.
    """
    t0 = time.time()
    breakdown = {}

    Lx = int(x_seq.shape[0]); Ly = int(y_seq.shape[0])
    Lx_pad = _pad_to_bin(Lx); Ly_pad = _pad_to_bin(Ly)

    # 1) Build TKF92 Pair HMM + padded emission table.
    #
    # log_trans + state_types depend ONLY on indel rates / t / ext (not on
    # the substitution model), so we always call make_tkf92_pair_hmm for
    # those, then OVERRIDE sub_matrix + pi_out with the canonical
    # class-summed singleton emission from block_likelihoods.build_singlet_emission
    # IF the boost_state carries the K=4 state + branch length + empirical pi_c.
    #
    # The class-summed Match emission satisfies pi_out_eff[a] *
    # sub_matrix_eff[a, b] = P_singlet(a, b; t) = sum_c pi_c * pi^(c)(a)
    # * P_c(a -> b; t), so substitution into the existing TKF92 PHMM API
    # (forward_pair_hmm, backward_pair_hmm, pair_hmm_emissions) is
    # transparent. This brings the inference-time singleton emission into
    # agreement with what was trained (svi.py uses the same per-class Q_c
    # construction; see block_likelihoods.py module docstring).
    t_phase = time.time()
    log_trans, state_types, sub_matrix_lg, pi_out_lg = make_tkf92_pair_hmm(
        ins_rate, del_rate, t, ext, Q_lg, pi_lg)
    if (getattr(boost_state, 'tkf_state', None) is not None
            and getattr(boost_state, 'pi_c', None) is not None):
        _, pi_out_np, sub_matrix_np = build_singlet_emission(
            boost_state.tkf_state, t, pi_c=np.asarray(boost_state.pi_c))
        sub_matrix = jnp.asarray(sub_matrix_np, dtype=sub_matrix_lg.dtype)
        pi_out = jnp.asarray(pi_out_np, dtype=pi_out_lg.dtype)
    else:
        # Fallback for old callers: plain LG08 baseline. Warn -- this
        # produces a model-spec mismatch with the trained K=4 emwarm
        # checkpoint (which has trained per-class profiles). See
        # mcmc_infinite_phmm_audit.md.
        print("[mcmc_infinite_phmm] WARNING: boost_state lacks tkf_state / "
              "pi_c; baseline FB falls back to plain LG08 single-class "
              "emission. Edge boost will then disagree with the trained "
              "K=4 model's singleton convention.", flush=True)
        sub_matrix, pi_out = sub_matrix_lg, pi_out_lg
    x_pad = jnp.asarray(_pad_seq(jnp.asarray(x_seq), Lx_pad))
    y_pad = jnp.asarray(_pad_seq(jnp.asarray(y_seq), Ly_pad))
    emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi_out)
    mask = _emit_mask(jnp.asarray(Lx), jnp.asarray(Ly), Lx_pad, Ly_pad,
                      state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)
    emit.block_until_ready()
    breakdown['pair_hmm_build'] = time.time() - t_phase

    # 2) Baseline Forward / Backward (O(L^2)).
    t_phase = time.time()
    log_alpha = forward_pair_hmm(
        log_trans, state_types, sub_matrix, pi_out,
        x_pad, y_pad, jnp.asarray(Lx), jnp.asarray(Ly))
    log_beta = backward_pair_hmm(
        log_trans, state_types, sub_matrix, pi_out,
        x_pad, y_pad, jnp.asarray(Lx), jnp.asarray(Ly))
    log_alpha.block_until_ready(); log_beta.block_until_ready()
    breakdown['forward_backward'] = time.time() - t_phase

    # 3) Baseline log_F0.
    e_idx = _find_e_idx(state_types)
    log_F0 = float(jax.nn.logsumexp(log_alpha[Lx, Ly, :] + log_trans[:, e_idx]))

    # 5) F_partial: legacy O(L^4) precompute. Identified as dead code
    #    (grep over tkf-dp tree confirms it is written but never read by
    #    the sampler; middle-segment traceback uses _restart_forward_jit
    #    via setup.mu_cache instead). Replaced by a sentinel shape-0
    #    array to expose any latent consumer immediately.
    F_partial = np.empty((0, 0, 0, 0), dtype=np.float32)

    # 6) M_obs[i, j, k, l] = log M_AA[X_i, Y_j, X_k, Y_l].  THIS IS THE
    #    LAST STRUCTURALLY-O(L^4) PIECE OF SETUP (memory-bandwidth-bound
    #    fancy gather; ~400 MB at L=100, ~6.4 GB at L=200).
    #
    # Build M_AA from the *class-marginal* construction (proper, post-
    # 2026-05-15). The earlier build_M_tensor_aa_marginal used a
    # gamma-weighted denominator (boost_state.denom) which is the
    # pre-MCMC first-order correction relic and violated the x=y, t=0
    # identity M_obs == M_solo by a 2-3x cell-dependent bias.
    t_phase = time.time()
    if (getattr(boost_state, 'tkf_state', None) is not None
            and getattr(boost_state, 'branch_length', None) is not None
            and getattr(boost_state, 'pi_c', None) is not None):
        # Unified path (post-2026-05-15): block_likelihoods.build_M_tensor
        # with the canonical (LG08 pair background, empirical pi_c)
        # convention -- matches the trained K=4 emwarm release. Doublet
        # uses sum_{c1, c2} pi_c(c1) pi_c(c2) pi_joint(LG08-bg)(a, c)
        # P_joint(... ; t) at eta=1, divided by the singlet emission
        # product -- consistent with the new baseline FB above.
        M_AA = build_M_tensor_unified(
            boost_state.tkf_state,
            boost_state.branch_length,
            pi_c=np.asarray(boost_state.pi_c),
            pair_background=getattr(boost_state, 'pair_background', 'lg08'))
    else:
        # Fallback for old callers. Emit a warning + use the relic.
        print("[mcmc_infinite_phmm] WARNING: boost_state lacks tkf_state / "
              "branch_length / pi_c; falling back to gamma-weighted M "
              "(relic). Results will be biased relative to the canonical "
              "target.", flush=True)
        M_AA = build_M_tensor_aa_marginal(boost_state)  # (A, A, A, A)
    log_M_AA = np.log(np.clip(M_AA, 1e-300, None)).astype(np.float32)
    # x_pad / y_pad as np.int32; clamp wildcards (index 20) to 19 for the
    # AA-tensor lookup (matching aug_phmm convention; H13 default).
    x_np = np.asarray(x_pad).astype(np.int32)
    y_np = np.asarray(y_pad).astype(np.int32)
    x_clip = np.minimum(x_np, A - 1)
    y_clip = np.minimum(y_np, A - 1)
    # Build a (Lx_pad, Ly_pad, Lx_pad, Ly_pad) gather of log_M_AA at observed AAs.
    # Then pad with one extra row/col of zeros (to make the shape (Lx_pad+1)^2
    # (Ly_pad+1)^2 and align with F_partial's 1-based residue convention).
    M_obs = np.zeros((Lx_pad + 1, Ly_pad + 1, Lx_pad + 1, Ly_pad + 1),
                     dtype=np.float32)
    # M_obs[i_a + 1, j_a + 1, k_a + 1, l_a + 1] = log_M_AA[x[i_a], y[j_a], x[k_a], y[l_a]]
    # Vectorise the gather:
    #   log_M_AA[x_clip[:, None, None, None],
    #            y_clip[None, :, None, None],
    #            x_clip[None, None, :, None],
    #            y_clip[None, None, None, :]]
    # This is (Lx_pad, Ly_pad, Lx_pad, Ly_pad).
    gathered = log_M_AA[x_clip[:, None, None, None],
                        y_clip[None, :, None, None],
                        x_clip[None, None, :, None],
                        y_clip[None, None, None, :]]
    # Store at positions [1..Lx_pad, 1..Ly_pad, 1..Lx_pad, 1..Ly_pad]
    # so that M_obs[i, j, k, l] uses 1-based residue position indexing
    # consistent with F_partial.
    M_obs[1:, 1:, 1:, 1:] = gathered
    breakdown['m_obs_gather'] = time.time() - t_phase

    # Phase-A: pre-convert the JAX-side tables that the sampler hot loops
    # would otherwise re-fetch on every call (~1700 calls per pair at L=100).
    t_phase = time.time()
    log_trans_np = np.asarray(log_trans)
    state_types_np = np.asarray(state_types)
    log_alpha_np = np.asarray(log_alpha)
    log_beta_np = np.asarray(log_beta)
    e_idx_int = int(_find_e_idx(state_types))
    breakdown['table_convert'] = time.time() - t_phase

    # 7) Optional cache prepopulate: rank Match cells by baseline posterior
    #    log_alpha[i,j,M] + log_beta[i,j,M] - log_F0, take the top
    #    prepop_top_k anchors, and batch-restart-Forward at all of them
    #    in one (or a few) vmap'd GPU launches. The chain is then
    #    overwhelmingly likely to find these anchors in cache the first
    #    time it visits them, instead of paying L^{3/2} sequential GPU
    #    launches during sampling.
    #
    #    Defaults: prepop_top_k = -1 means "use ceil(L^{3/2})" (with a
    #    memory-budget cap of prepop_mem_budget_mib MiB). Pass 0 to
    #    disable prepopulation entirely.
    mu_cache = {}
    prepop_seconds = 0.0
    prepop_n_anchors = 0
    if prepop_top_k != 0:
        t_phase = time.time()
        # Default K = ceil(L^{3/2}) where L = max(Lx, Ly).
        L_eff = max(Lx, Ly)
        if prepop_top_k < 0:
            K_default = int(np.ceil(L_eff * np.sqrt(L_eff)))
        else:
            K_default = int(prepop_top_k)
        # Memory cap. Each cached entry is (Lx_pad+1)(Ly_pad+1)*n_states fp32.
        n_states = log_trans_np.shape[0]
        per_entry_mib = ((Lx_pad + 1) * (Ly_pad + 1) * n_states * 4) / (1024 * 1024)
        K_mem_cap = max(1, int(prepop_mem_budget_mib / max(per_entry_mib, 1e-9)))
        # Also cap at the number of valid Match cells (Lx * Ly).
        K_pos_cap = Lx * Ly
        K = max(0, min(K_default, K_mem_cap, K_pos_cap))

        if K > 0:
            # Posterior at each Match cell, restricted to the real region.
            # log_F0 is the partition function; subtract for normalisation.
            log_post = log_alpha_np[1:Lx + 1, 1:Ly + 1, M] \
                       + log_beta_np[1:Lx + 1, 1:Ly + 1, M] \
                       - log_F0  # (Lx, Ly)
            log_post_flat = log_post.reshape(-1)
            # Mask out -inf cells (unreachable / off-band).
            finite_mask = np.isfinite(log_post_flat)
            n_finite = int(finite_mask.sum())
            K_eff = min(K, n_finite)
            if K_eff > 0:
                # Take the top-K_eff by log_post.
                order = np.argpartition(-log_post_flat, K_eff - 1)[:K_eff]
                # Convert flat indices to (i, j) in 1-based residue coords
                # (matching the anchor-coords used in the sampler).
                ai_np = (order // Ly).astype(np.int32) + 1
                aj_np = (order % Ly).astype(np.int32) + 1
                # Batch in chunks to bound peak GPU memory.
                for s in range(0, K_eff, prepop_chunk):
                    e = min(s + prepop_chunk, K_eff)
                    ai_chunk = jnp.asarray(ai_np[s:e])
                    aj_chunk = jnp.asarray(aj_np[s:e])
                    mu_chunk = _chunk_mu_full_jit(
                        log_trans, state_types, emit,
                        ai_chunk, aj_chunk, Lx_pad, Ly_pad)
                    mu_chunk_np = np.asarray(mu_chunk)  # (chunk, Lx_pad+1, Ly_pad+1, n_states)
                    for k in range(e - s):
                        ka = (int(ai_np[s + k]), int(aj_np[s + k]))
                        mu_cache[ka] = mu_chunk_np[k]
                prepop_n_anchors = K_eff
        breakdown['mu_cache_prepop'] = time.time() - t_phase
        prepop_seconds = breakdown['mu_cache_prepop']
    breakdown['mu_cache_prepop_n_anchors'] = prepop_n_anchors

    setup_seconds = time.time() - t0

    return MCMCSetup(
        Lx=Lx, Ly=Ly, Lx_pad=Lx_pad, Ly_pad=Ly_pad,
        x_seq=np.asarray(x_seq, dtype=np.int32),
        y_seq=np.asarray(y_seq, dtype=np.int32),
        x_pad=x_pad, y_pad=y_pad,
        log_trans=log_trans, state_types=state_types,
        sub_matrix=sub_matrix, pi_out=pi_out, emit=emit,
        log_alpha=log_alpha, log_beta=log_beta,
        F_partial=F_partial, M_obs=M_obs,
        log_F0=log_F0,
        alpha_z=float(alpha_z),
        setup_seconds=setup_seconds,
        setup_breakdown=breakdown,
        mu_cache=mu_cache,
        log_trans_np=log_trans_np,
        state_types_np=state_types_np,
        log_alpha_np=log_alpha_np,
        log_beta_np=log_beta_np,
        e_idx=e_idx_int,
    )


# ===========================================================================
# Alignment representation.
# ===========================================================================
#
# An alignment is a list of (state, i, j) triples where state in {S, M, I, D, E},
# i in 0..Lx, j in 0..Ly are the sequence positions consumed up through
# this cell. The path begins at (S, 0, 0), advances to a residue-state
# (M, I, or D) at each step (consuming residues), and terminates at (E, Lx, Ly).
#
# This is a direct match-cell representation; we don't carry edges in the
# alignment object (edges live in a separate set E of (i, j) pairs).
#
# State successor rules:
#   M from (i-1, j-1, *), consumes (X_i, Y_j)        -- adds Match cell at (i, j)
#   I from (i, j-1, *),   consumes Y_j                -- inserts gap in X
#   D from (i-1, j, *),   consumes X_i                -- inserts gap in Y


def _stochastic_traceback_full(
        rng_key, setup: MCMCSetup) -> List[Tuple[int, int, int]]:
    """Sample a full alignment from the baseline pair-HMM Forward
    distribution via stochastic traceback through log_alpha.

    Returns: list of (state, i, j) triples (state in {M, I, D}).
    The alignment starts implicitly at (S, 0, 0) and ends at E.
    """
    log_trans = setup.log_trans_np
    state_types = setup.state_types_np
    log_alpha = setup.log_alpha_np
    e_idx = setup.e_idx
    Lx, Ly = setup.Lx, setup.Ly
    rng = np.random.default_rng(int(jax.random.randint(
        rng_key, (), 0, 2**31 - 1)))

    # Final step: pick the predecessor state of E.
    end_logits = log_alpha[Lx, Ly, :] + log_trans[:, e_idx]
    end_logits = end_logits - np.max(end_logits)
    end_w = np.exp(end_logits)
    end_w = end_w / end_w.sum()
    cur_state = int(rng.choice(log_trans.shape[0], p=end_w))
    i, j = Lx, Ly
    path: List[Tuple[int, int, int]] = [(cur_state, i, j)]

    # Walk back through alpha. At each step we sample the (predecessor state,
    # decrement) given the current (state, i, j). For M we go to (i-1, j-1);
    # for I to (i, j-1); for D to (i-1, j).
    while not (cur_state == S and i == 0 and j == 0):
        if cur_state == M:
            i_prev, j_prev = i - 1, j - 1
        elif cur_state == I:
            i_prev, j_prev = i, j - 1
        elif cur_state == D:
            i_prev, j_prev = i - 1, j
        elif cur_state == S and i == 0 and j == 0:
            break
        else:
            raise RuntimeError(
                f"Unknown state {cur_state} at ({i}, {j})")
        # Pick the predecessor state at (i_prev, j_prev).
        logits = log_alpha[i_prev, j_prev, :] + log_trans[:, cur_state]
        logits = logits - np.max(logits[np.isfinite(logits)])
        w = np.exp(logits); w[~np.isfinite(w)] = 0.0
        if w.sum() <= 0:
            # Defensive: if all logits are -inf, place all weight on S
            # (can only happen if the alignment is degenerate).
            w = np.zeros_like(w); w[S] = 1.0
        w = w / w.sum()
        prev_state = int(rng.choice(log_trans.shape[0], p=w))
        i, j = i_prev, j_prev
        cur_state = prev_state
        if not (cur_state == S and i == 0 and j == 0):
            path.append((cur_state, i, j))

    path.reverse()
    return path


def _initial_alignment(rng_key, setup: MCMCSetup,
                       init_mode: str = "viterbi") -> List[Tuple[int, int, int]]:
    """Initialise the chain's alignment.

    init_mode in {"viterbi", "forward_sample"}.

    "viterbi" runs a small numpy DP for the maximum-probability alignment
    (which equals the modal Pair HMM alignment under the baseline). For
    medium L (<= 500) this is fine.

    "forward_sample" draws one sample from the baseline Forward
    distribution via stochastic traceback. This is the natural drop-in
    for cases where Viterbi might be a poor mode (multimodal posterior).
    """
    if init_mode == "forward_sample":
        return _stochastic_traceback_full(rng_key, setup)
    elif init_mode == "viterbi":
        # numpy Viterbi (small).
        log_trans = setup.log_trans_np
        state_types = setup.state_types_np
        emit = np.asarray(setup.emit)
        Lx, Ly = setup.Lx, setup.Ly
        ns = log_trans.shape[0]
        e_idx = setup.e_idx

        is_M = state_types == M; is_I = state_types == I; is_D = state_types == D

        # V[i, j, k] = max log-prob of any path ending at state k at (i, j),
        # including the emission at the final state.
        # TB[i, j, k] = predecessor state index (an int).
        V = np.full((Lx + 1, Ly + 1, ns), NEG_INF)
        TB = np.full((Lx + 1, Ly + 1, ns), -1, dtype=np.int32)
        V[0, 0, S] = 0.0

        # Row 0 (j = 1..Ly, only I-types reachable).
        for j in range(1, Ly + 1):
            for k in range(ns):
                if not is_I[k]:
                    continue
                scores = V[0, j - 1, :] + log_trans[:, k]
                if not np.any(np.isfinite(scores)):
                    continue
                best = int(np.argmax(scores))
                V[0, j, k] = scores[best] + emit[0, j, k]
                TB[0, j, k] = best
        # Column 0 (i = 1..Lx, only D-types).
        for i in range(1, Lx + 1):
            for k in range(ns):
                if not is_D[k]:
                    continue
                scores = V[i - 1, 0, :] + log_trans[:, k]
                if not np.any(np.isfinite(scores)):
                    continue
                best = int(np.argmax(scores))
                V[i, 0, k] = scores[best] + emit[i, 0, k]
                TB[i, 0, k] = best
        # Interior.
        for i in range(1, Lx + 1):
            for j in range(1, Ly + 1):
                for k in range(ns):
                    if is_M[k]:
                        scores = V[i - 1, j - 1, :] + log_trans[:, k]
                    elif is_I[k]:
                        scores = V[i, j - 1, :] + log_trans[:, k]
                    elif is_D[k]:
                        scores = V[i - 1, j, :] + log_trans[:, k]
                    else:
                        continue
                    if not np.any(np.isfinite(scores)):
                        continue
                    best = int(np.argmax(scores))
                    V[i, j, k] = scores[best] + emit[i, j, k]
                    TB[i, j, k] = best
        # Terminate.
        end_scores = V[Lx, Ly, :] + log_trans[:, e_idx]
        if not np.any(np.isfinite(end_scores)):
            raise RuntimeError("Viterbi: no valid terminal path")
        cur = int(np.argmax(end_scores))
        i, j = Lx, Ly
        path: List[Tuple[int, int, int]] = [(cur, i, j)]
        while not (cur == S and i == 0 and j == 0):
            prev = int(TB[i, j, cur])
            if prev < 0:
                break
            if cur == M:
                i, j = i - 1, j - 1
            elif cur == I:
                i, j = i, j - 1
            elif cur == D:
                i, j = i - 1, j
            else:
                break
            if not (prev == S and i == 0 and j == 0):
                path.append((prev, i, j))
            cur = prev
        path.reverse()
        return path
    else:
        raise ValueError(f"unknown init_mode: {init_mode}")


# ===========================================================================
# Segment-resample MH move.
# ===========================================================================


def _match_cells_of(path: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    """Return list of (i, j) Match cells in path order."""
    return [(i, j) for (st, i, j) in path if st == M]


def _edge_anchors_in_path_order(
        path: List[Tuple[int, int, int]],
        edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]
        ) -> List[Tuple[int, int]]:
    """Return the union of edge endpoint Match cells in path order."""
    edge_set = set()
    for (a, b) in edges:
        edge_set.add(a); edge_set.add(b)
    out = []
    for (st, i, j) in path:
        if st == M and (i, j) in edge_set:
            out.append((i, j))
    return out


def _path_log_prob(path: List[Tuple[int, int, int]],
                   setup: MCMCSetup) -> float:
    """Compute log pi_TKF92(A) for an alignment A under the baseline.

    Includes path transitions and emissions, plus the trailing transition
    to E.
    """
    log_trans = setup.log_trans_np
    emit = np.asarray(setup.emit)
    e_idx = setup.e_idx
    lp = 0.0
    prev = S
    for (st, i, j) in path:
        lp += log_trans[prev, st]
        lp += emit[i, j, st]
        prev = st
    lp += log_trans[prev, e_idx]
    return float(lp)


def _stochastic_traceback_segment(
        rng: np.random.Generator,
        setup: MCMCSetup,
        i_a: int, j_a: int,        # entry anchor (1-based pos), or (0, 0) for start
        i_b: int, j_b: int,        # exit anchor  (1-based pos), or (Lx, Ly) for end
        is_start_anchor: bool,     # True: i_a/j_a are virtual start (0, 0)
        is_end_anchor: bool        # True: exit is the E state at (Lx, Ly)
        ) -> List[Tuple[int, int, int]]:
    """Sample a fragment between two anchors from the baseline Pair HMM
    Forward distribution restricted to passing through Match at both
    endpoints (or at start/end where appropriate).

    Returns: list of (state, i, j) representing all cells STRICTLY
    BETWEEN the two anchors. Anchor cells themselves are excluded.

    Approach:
      - Build the "scoring grid" scores[i, j, k] giving the log-Forward
        weight of reaching (i, j, k) along any path that started at the
        entry anchor:
          * is_start_anchor=True  ->  scores = log_alpha (start at (0, 0, S))
          * else                   ->  scores = restart-Forward from M
                                        at (i_a, j_a)  (the mu tensor)
      - Determine the trace start: the cell just before the exit anchor:
          * is_end_anchor=True   ->  exit is E at (Lx, Ly); pick the
                                      penultimate state by sampling from
                                      scores[Lx, Ly, :] + log_trans[:, E].
          * else                  ->  exit is M at (i_b, j_b); the
                                      penultimate cell is sampled from
                                      scores[i_prev, j_prev, :] + log_trans[:, M]
                                      where (i_prev, j_prev) is the
                                      diagonal predecessor (i_b - 1, j_b - 1).
      - Walk back cell by cell, sampling predecessor (state, i, j) until
        we reach the entry anchor.
    """
    # Phase-A: use pre-converted numpy tables from setup (avoids re-running
    # np.asarray on a JAX device array 1700 times per pair).
    log_trans = setup.log_trans_np
    Lx, Ly = setup.Lx, setup.Ly
    e_idx = setup.e_idx
    ns = log_trans.shape[0]

    # Build scores grid.
    _t_seg0 = time.time()
    if is_start_anchor:
        scores = setup.log_alpha_np
    else:
        cache_key = (int(i_a), int(j_a))
        if cache_key in setup.mu_cache:
            scores = setup.mu_cache[cache_key]
            setup.rf_n_hits += 1
        else:
            _t_rf0 = time.time()
            mu = _restart_forward_jit(
                setup.log_trans, setup.state_types, setup.emit,
                setup.Lx_pad, setup.Ly_pad,
                jnp.int32(i_a), jnp.int32(j_a))
            scores = np.asarray(mu)
            setup.rf_seconds += time.time() - _t_rf0
            setup.rf_n_misses += 1
            setup.mu_cache[cache_key] = scores

    # Stop coordinate: when we step back to (i_a, j_a) we stop.
    # For start segment, stop is (0, 0).
    stop_i, stop_j = i_a, j_a

    # --- Initialise the trace (in Python; cheap single categorical). ---
    if is_end_anchor:
        # Exit is E at (Lx, Ly). Sample predecessor state.
        s_init = _sample_state_at_np(scores, log_trans, e_idx,
                                     Lx, Ly, E, ns, rng)
        if s_init < 0:
            return []
        i_cur, j_cur = Lx, Ly
        # Segment-empty edge case: entry anchor is the (Lx, Ly) M cell
        if i_cur == stop_i and j_cur == stop_j and s_init == M:
            return []
    else:
        # Exit is M at (i_b, j_b). Predecessor cell is on the diagonal.
        i_prev, j_prev = i_b - 1, j_b - 1
        if (i_prev, j_prev) == (stop_i, stop_j):
            return []  # adjacent anchors, empty segment
        s_init = _sample_state_at_np(scores, log_trans, e_idx,
                                     i_prev, j_prev, M, ns, rng)
        if s_init < 0:
            return []
        i_cur, j_cur = i_prev, j_prev

    # --- Walk back via Phase-B JIT kernel. ---
    # max_iter is bounded by the no-match path length (each cell consumes at
    # least one residue from x or y). Bin to geomspace for JIT cache reuse.
    max_iter_bin = _pad_to_bin(Lx + Ly + 2)
    rng_key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    # Pass scalar args as numpy int32 (avoids jnp.int32 conversion primitives;
    # JAX will absorb them into the jit'd kernel via traced arguments).
    stacked = _scan_traceback_kernel_jit(
        rng_key, scores, log_trans,
        np.int32(stop_i), np.int32(stop_j),
        np.int32(s_init), np.int32(i_cur), np.int32(j_cur),
        max_iter=int(max_iter_bin),
        n_states=ns,
        is_start_anchor=bool(is_start_anchor))
    # One device sync, then filter sentinels and reverse to forward order.
    arr = np.asarray(stacked)               # (3, max_iter)
    valid = arr[0] != -1
    out = [tuple(row) for row in arr[:, valid].T.tolist()]
    out.reverse()
    setup.tb_seconds += time.time() - _t_seg0
    setup.tb_n_calls += 1
    return out


def _resample_alignment_given_anchors(
        rng: np.random.Generator,
        setup: MCMCSetup,
        anchor_positions: List[Tuple[int, int]]
        ) -> List[Tuple[int, int, int]]:
    """Resample every inter-edge-anchor segment of the alignment from
    the baseline conditional Forward distribution.

    anchor_positions is the list of EDGE-ANCHOR Match cell positions
    in path order (1-based residue coords). Per the user's spec (H7),
    the alignment is partitioned into 2|E| + 1 segments by these anchors
    plus virtual start (0, 0) and end (just-before-E).

    Each anchor cell is FIXED. The returned new alignment passes through
    Match at every anchor.

    Returns: the new full alignment as a list of (state, i, j) triples.
    """
    new_path: List[Tuple[int, int, int]] = []
    Lx, Ly = setup.Lx, setup.Ly
    if len(anchor_positions) == 0:
        # Single segment from (0, 0) to E.
        seg = _stochastic_traceback_segment(
            rng, setup, 0, 0, Lx, Ly,
            is_start_anchor=True, is_end_anchor=True)
        new_path.extend(seg)
        return new_path
    # Multiple segments.
    # Segment 0: from (0, 0) to first anchor.
    (i_b0, j_b0) = anchor_positions[0]
    seg0 = _stochastic_traceback_segment(
        rng, setup, 0, 0, i_b0, j_b0,
        is_start_anchor=True, is_end_anchor=False)
    new_path.extend(seg0)
    new_path.append((M, i_b0, j_b0))
    # Middle segments: anchor k to anchor k+1.
    for k in range(len(anchor_positions) - 1):
        (i_a, j_a) = anchor_positions[k]
        (i_b, j_b) = anchor_positions[k + 1]
        seg = _stochastic_traceback_segment(
            rng, setup, i_a, j_a, i_b, j_b,
            is_start_anchor=False, is_end_anchor=False)
        new_path.extend(seg)
        new_path.append((M, i_b, j_b))
    # Last segment: last anchor to E.
    (i_aL, j_aL) = anchor_positions[-1]
    segE = _stochastic_traceback_segment(
        rng, setup, i_aL, j_aL, Lx, Ly,
        is_start_anchor=False, is_end_anchor=True)
    new_path.extend(segE)
    return new_path


def _resample_one_segment(
        rng: np.random.Generator,
        setup: MCMCSetup,
        path: List[Tuple[int, int, int]],
        edge_anchor_positions: List[Tuple[int, int]],
        seg_idx: int,
        ) -> Tuple[List[Tuple[int, int, int]], int, int]:
    """Resample exactly one segment (between adjacent edge-anchor anchors,
    or between the virtual start/end and an edge anchor).

    seg_idx in {0, 1, ..., len(edge_anchor_positions)} (so 0 = start
    segment, |E_anchors| = end segment).

    Returns (new_path, N_M_old_seg, N_M_new_seg).
    """
    Lx, Ly = setup.Lx, setup.Ly
    n_anchors = len(edge_anchor_positions)
    # Determine segment endpoints.
    if seg_idx == 0:
        # Start segment.
        if n_anchors == 0:
            # Single segment from start to end.
            i_a, j_a = 0, 0; i_b, j_b = Lx, Ly
            is_start, is_end = True, True
        else:
            i_a, j_a = 0, 0
            i_b, j_b = edge_anchor_positions[0]
            is_start, is_end = True, False
    elif seg_idx == n_anchors:
        # End segment (after last anchor).
        i_a, j_a = edge_anchor_positions[-1]
        i_b, j_b = Lx, Ly
        is_start, is_end = False, True
    else:
        # Middle segment between anchor seg_idx-1 and seg_idx.
        i_a, j_a = edge_anchor_positions[seg_idx - 1]
        i_b, j_b = edge_anchor_positions[seg_idx]
        is_start, is_end = False, False

    # Identify the cells in the OLD segment (between i_a/j_a and i_b/j_b).
    # These are the cells of `path` strictly after the entry anchor and
    # strictly before the exit anchor.
    # We split the path into segments at edge-anchor cells.
    old_segments = _split_path_into_segments(path, edge_anchor_positions)
    if seg_idx >= len(old_segments):
        # Shouldn't happen; defensive.
        return path, 0, 0
    old_seg = old_segments[seg_idx]
    N_M_old = sum(1 for (st, _, _) in old_seg if st == M)

    # Sample new segment.
    new_seg = _stochastic_traceback_segment(
        rng, setup, i_a, j_a, i_b, j_b, is_start, is_end)
    N_M_new = sum(1 for (st, _, _) in new_seg if st == M)

    # Assemble new path: replace old_seg with new_seg.
    new_path: List[Tuple[int, int, int]] = []
    for k, seg in enumerate(old_segments):
        if k == seg_idx:
            new_path.extend(new_seg)
        else:
            new_path.extend(seg)
        # Add the trailing anchor cell (if not the last segment).
        if k < n_anchors:
            new_path.append((M,) + edge_anchor_positions[k])

    return new_path, N_M_old, N_M_new


def _split_path_into_segments(
        path: List[Tuple[int, int, int]],
        edge_anchor_positions: List[Tuple[int, int]]
        ) -> List[List[Tuple[int, int, int]]]:
    """Split a path into 2|E| + 1 segments by edge-anchor Match cells.

    Returns a list of len(edge_anchor_positions) + 1 segments. Segment k
    contains all cells STRICTLY BETWEEN anchor k-1 and anchor k (with
    anchor -1 = virtual start, anchor |E_anchors| = virtual end).

    Anchor cells themselves are NOT included in any segment.
    """
    if len(edge_anchor_positions) == 0:
        return [list(path)]
    anchor_set = set(edge_anchor_positions)
    # Determine the path index of each anchor.
    out: List[List[Tuple[int, int, int]]] = [[]]
    anchors_left = list(edge_anchor_positions)
    next_anchor_idx = 0
    for cell in path:
        st, i, j = cell
        if (st == M and next_anchor_idx < len(edge_anchor_positions)
                and (i, j) == edge_anchor_positions[next_anchor_idx]):
            # This is the next expected edge anchor; start a new segment.
            out.append([])
            next_anchor_idx += 1
        else:
            out[-1].append(cell)
    # Pad to expected length (in case the alignment didn't include all
    # expected anchors -- shouldn't happen if anchors are in path-order).
    while len(out) < len(edge_anchor_positions) + 1:
        out.append([])
    return out


def _crp_log_prior_pathlen(N_M: int, alpha_z: float) -> float:
    """log of the CRP-prior path-length factor:
        sum_{m=1..N_M} -log(m - 1 + alpha_z)
    (Equivalently, -log of the rising factorial of alpha_z up to N_M.)
    """
    if N_M <= 0:
        return 0.0
    # Use math.lgamma trick: prod_{m=1..N_M}(m - 1 + alpha_z) =
    #   prod_{j=0..N_M-1}(alpha_z + j) = Gamma(alpha_z + N_M) / Gamma(alpha_z)
    from math import lgamma
    return float(lgamma(alpha_z) - lgamma(alpha_z + N_M))


def _segment_resample_move(
        rng: np.random.Generator,
        setup: MCMCSetup,
        path: List[Tuple[int, int, int]],
        edge_anchor_positions: List[Tuple[int, int]],
        ) -> Tuple[List[Tuple[int, int, int]], int, int]:
    """Apply the combined segment-resample move (per-segment MH version).

    Edge anchors are FIXED Match cells. Per H4 (systematic scan) and H7
    (sample every adjacent segment between edge anchors), we visit each
    inter-edge-anchor segment and run a per-segment MH move with the
    tight-proposal simplification (Hastings ratio = CRP-prior factor
    only).

    Each per-segment move:
      - Sample new_seg via stochastic traceback.
      - Compute N_M change (delta N_M).
      - Accept w.p. min(1, CRP_ratio) where CRP_ratio depends on the
        change in N_M total.

    Returns (new_path, n_propose, n_accept).
    """
    cur_match_set = set(_match_cells_of(path))
    for ap in edge_anchor_positions:
        assert ap in cur_match_set, \
            f"edge anchor {ap} not in current Match cells {cur_match_set}"

    n_segments = len(edge_anchor_positions) + 1
    n_propose = 0; n_accept = 0
    for seg_idx in range(n_segments):
        proposed_path, _N_M_old_seg, _N_M_new_seg = _resample_one_segment(
            rng, setup, path, edge_anchor_positions, seg_idx)
        n_propose += 1
        # Pure Gibbs: edge endpoints are FIXED during this move, so the
        # M boost product and the per-edge prior weight eps^|E| are
        # constant. The conditional pi(A | E) reduces to baseline TKF92
        # path conditional, which is exactly what _resample_one_segment
        # samples from F_partial. Accept rate = 1 by construction.
        path = proposed_path
        n_accept += 1
    return path, n_propose, n_accept


# ===========================================================================
# Edge add / remove MH moves.
# ===========================================================================


def _log_M_obs(setup: MCMCSetup, p1: Tuple[int, int],
               p2: Tuple[int, int]) -> float:
    """log M(p1, p2) at observed AAs."""
    return float(setup.M_obs[p1[0], p1[1], p2[0], p2[1]])


def _edge_add_move(
        rng: np.random.Generator,
        setup: MCMCSetup,
        path: List[Tuple[int, int, int]],
        edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        k_max: int,
        ) -> Tuple[List, bool]:
    """Propose adding one edge between two Match cells.

    Q_add-uniform: pick two distinct Match cells uniformly from the
    currently-unpaired Match cells (per H6: matchings only; no shared
    endpoints).

    Hastings ratio (canonical CRP):
      target ratio = alpha_z * M(e ; t)
      q_add(E_new | E_old) = 1 / [N_unpaired choose 2]
      q_remove(E_old | E_new) = 1 / |E_new|
      H = alpha_z * M * [N_unpaired choose 2] / |E_new|

    Hastings ratio (bounded_eps):
      target ratio = eps * M(e ; t) [for opening at one cell + close at another]
      [Specifically the bounded prior weight is eps^{2|E|} M-product,
       so adding an edge gives eps^2 * M(e); but to match the aug_phmm/
       aug_phmm_2edge convention where the per-cell spawn weight is eps,
       the "edge addition" target ratio is eps^2 * M(e)].
      q ratio same as canonical.

    Returns (new_edges, accepted).
    """
    matches = _match_cells_of(path)
    paired = set()
    for (a, b) in edges:
        paired.add(a); paired.add(b)
    unpaired = [m for m in matches if m not in paired]
    n_unpaired = len(unpaired)
    if n_unpaired < 2:
        return edges, False
    if k_max >= 0 and len(edges) >= k_max:
        return edges, False
    # Pick two unpaired Match cells uniformly.
    n_pairs_unp = n_unpaired * (n_unpaired - 1) // 2
    flat_idx = int(rng.integers(0, n_pairs_unp))
    # Decode flat_idx -> (a, b) with a < b.
    a = 0
    while flat_idx >= n_unpaired - 1 - a:
        flat_idx -= n_unpaired - 1 - a
        a += 1
    b = a + 1 + flat_idx
    p_a = unpaired[a]; p_b = unpaired[b]
    # Canonicalise edge as a 2-tuple of unordered pairs.
    new_edge = (p_a, p_b) if p_a <= p_b else (p_b, p_a)

    # Target ratio: per-edge weight eps = 1/alpha_z (size-{1,2} Ewens
    # / equivalently bounded-eps; mathematically identical formulations).
    log_M = _log_M_obs(setup, p_a, p_b)
    eps = 1.0 / setup.alpha_z
    log_target = float(np.log(eps)) + log_M
    # Proposal ratio: q_remove(old | new) / q_add(new | old).
    # q_add(new | old) = 1 / n_pairs_unp.
    # q_remove(old | new) = 1 / |E_new| = 1 / (|E_old| + 1).
    log_qratio = float(np.log(n_pairs_unp)) - float(np.log(len(edges) + 1))
    log_H = log_target + log_qratio
    log_H = min(log_H, 0.0)
    u = float(rng.random())
    if np.log(max(u, 1e-300)) < log_H:
        new_edges = list(edges) + [new_edge]
        return new_edges, True
    return edges, False


def _edge_remove_move(
        rng: np.random.Generator,
        setup: MCMCSetup,
        path: List[Tuple[int, int, int]],
        edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        ) -> Tuple[List, bool]:
    """Propose removing one edge.

    Hastings ratio (canonical CRP):
      target ratio = 1 / (alpha_z * M(e ; t))
      q_remove(E_new | E_old) = 1 / |E_old|
      q_add(E_old | E_new) = 1 / [(N_unpaired_after) choose 2 + 1]
                            (1 more "unpaired" cell pair becomes available
                             after the removal).
      H = (1 / (alpha_z * M)) * |E_old| / [N_unpaired_after choose 2]

    Returns (new_edges, accepted).
    """
    if len(edges) == 0:
        return edges, False
    idx = int(rng.integers(0, len(edges)))
    e_to_remove = edges[idx]
    p_a, p_b = e_to_remove
    # Compute n_unpaired after removal.
    matches = _match_cells_of(path)
    paired_after = set()
    for k, e in enumerate(edges):
        if k == idx:
            continue
        paired_after.add(e[0]); paired_after.add(e[1])
    n_unpaired_after = sum(1 for m in matches if m not in paired_after)
    n_pairs_unp_after = n_unpaired_after * (n_unpaired_after - 1) // 2
    if n_pairs_unp_after <= 0:
        # Defensive (shouldn't happen since we have at least 2 unpaired
        # Match cells after removal).
        return edges, False
    log_M = _log_M_obs(setup, p_a, p_b)
    eps = 1.0 / setup.alpha_z
    log_target = -(float(np.log(eps)) + log_M)
    log_qratio = float(np.log(len(edges))) - float(np.log(n_pairs_unp_after))
    log_H = log_target + log_qratio
    log_H = min(log_H, 0.0)
    u = float(rng.random())
    if np.log(max(u, 1e-300)) < log_H:
        new_edges = list(edges); new_edges.pop(idx)
        return new_edges, True
    return edges, False



# ===========================================================================
# Main MCMC loop.
# ===========================================================================


def _attach_timing_to_diag(diag: 'MCMCDiagnostics', setup: 'MCMCSetup') -> None:
    """Copy the live timing/cache counters from the setup object onto the
    finalised diag. Call once at the end of each chain run."""
    diag.setup_seconds = float(getattr(setup, 'setup_seconds', 0.0))
    diag.setup_breakdown = dict(getattr(setup, 'setup_breakdown', {}) or {})
    diag.rf_seconds = float(getattr(setup, 'rf_seconds', 0.0))
    diag.rf_n_misses = int(getattr(setup, 'rf_n_misses', 0))
    diag.rf_n_hits = int(getattr(setup, 'rf_n_hits', 0))
    diag.mu_cache_size = int(len(getattr(setup, 'mu_cache', {}) or {}))
    diag.tb_seconds = float(getattr(setup, 'tb_seconds', 0.0))
    diag.tb_n_calls = int(getattr(setup, 'tb_n_calls', 0))
    diag.Lx = int(getattr(setup, 'Lx', 0))
    diag.Ly = int(getattr(setup, 'Ly', 0))


@dataclass
class MCMCDiagnostics:
    """Slim diagnostics."""
    n_sweeps: int = 0
    n_burnin: int = 0
    n_accept_seg: int = 0
    n_accept_add: int = 0
    n_accept_remove: int = 0
    n_propose_seg: int = 0
    n_propose_add: int = 0
    n_propose_remove: int = 0
    log_pi_trace: List[float] = field(default_factory=list)
    n_edges_trace: List[int] = field(default_factory=list)
    n_match_trace: List[int] = field(default_factory=list)
    runtime_seconds: float = 0.0
    # Timing breakdown (populated at end of run from setup counters):
    setup_seconds: float = 0.0          # one-shot setup wall-time
    setup_breakdown: dict = field(default_factory=dict)  # per-phase setup
    rf_seconds: float = 0.0             # cumulative restart-Forward wall time
    rf_n_misses: int = 0                # mu_cache misses (= restart-Forward calls)
    rf_n_hits: int = 0                  # mu_cache hits (no restart-Forward)
    mu_cache_size: int = 0              # |unique anchors touched| at end of run
    tb_seconds: float = 0.0             # cumulative inner-traceback wall time
    tb_n_calls: int = 0                 # inner-traceback call count
    Lx: int = 0                         # for K_unique vs L^2 ratio postproc
    Ly: int = 0
    # Per-sweep cache-growth trace: cumulative mu_cache_size after each
    # sweep. Starts at prepop_n_anchors (after Forward+Backward F-B
    # populates the top L^{3/2} cells). Grows by 1 each time the chain
    # visits a new anchor not in the prepop set. Length = n_sweeps.
    mu_cache_size_trace: List[int] = field(default_factory=list)
    # Edge marginal posterior accumulators (post-burnin only). For each
    # recorded sweep we walk the current edge set and increment:
    #   edge_pos_x_counts[i] += 1 for every endpoint (i, *) in either
    #     edge anchor (so each edge contributes 2 counts on X).
    #   edge_pos_y_counts[j] += 1 likewise on Y.
    #   edge_cell_counts[(i, j)] += 1 for both endpoint cells of every
    #     edge in the current set.
    # Divide by n_recorded_for_edges to get marginal P(position is an
    # edge endpoint | data). This is signal that can be compared to PDB
    # contacts (e.g. C-C disulfide bonds): the alignment is marginalised
    # out, so this asks "does the sampler put edges at residues that are
    # in 3D contact regardless of which residue they align to?".
    # Stored 1-based: index 0 unused; indices 1..Lx (resp 1..Ly) valid.
    edge_pos_x_counts: List[int] = field(default_factory=list)
    edge_pos_y_counts: List[int] = field(default_factory=list)
    edge_cell_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    n_recorded_for_edges: int = 0
    # X-X / Y-Y unordered-pair edge marginal posteriors. For each edge
    # ((i1, j1), (i2, j2)) recorded in a post-burnin sweep, contribute
    # +1 to edge_pair_x_counts[(min(i1, i2), max(i1, i2))] and similarly
    # on Y. These project the joint sampler's coupled-pair distribution
    # onto each sequence axis separately, allowing comparison against a
    # single-sequence edge-marginal baseline (which by construction has no
    # cross-sequence covariation evidence and should be flat on signal
    # families).
    edge_pair_x_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    edge_pair_y_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)


def _unnormalised_log_target(
        path: List[Tuple[int, int, int]],
        edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        setup: MCMCSetup) -> float:
    """log pi(A, E) (unnormalised) for diagnostics. NOT used in the MH
    inner loop (which uses ratios).

    pi(A, E) propto pi_TKF92(A) * eps^|E| * prod_{e in E} M(e).
    """
    log_baseline = _path_log_prob(path, setup)
    log_boost = 0.0
    for (a, b) in edges:
        log_boost += _log_M_obs(setup, a, b)
    eps = 1.0 / setup.alpha_z
    log_prior = len(edges) * float(np.log(eps))
    return log_baseline + log_boost + log_prior

def run_mcmc_chain(
        setup: MCMCSetup,
        n_sweeps: int = 1000,
        n_burnin: int = 200,
        k_max: int = -1,
        n_edge_moves_per_sweep: int = 8,
        init_mode: str = "viterbi",
        seed: int = 0,
        record_every: int = 1,
        verbose: bool = False,
        alpha_z_init: Optional[float] = None,
        alpha_z_final: Optional[float] = None,
        anneal_fraction: float = 0.0,
        ) -> Tuple[np.ndarray, MCMCDiagnostics]:
    """Run a single MCMC chain.

    The kernel is Gibbs+MH on the infinite-Pair-HMM target:

      pi(A, E | X, Y)  propto  pi_TKF92(A | X, Y) * eps^|E| * prod_e M(e)

    Per sweep:
      1. Pure-Gibbs path resample between every adjacent pair of edge
         anchors via stochastic traceback through F^partial. With edge
         endpoints fixed, the M and eps factors are constant, so the
         conditional pi(A | E) reduces to baseline TKF92. Accept = 1.
      2. n_edge_moves_per_sweep MH attempts to add or remove an edge.
         Per-edge target factor eps = 1/alpha_z; M boost at the chosen
         (cell_a, cell_b) pair via the precomputed M_obs tensor.

    SIMULATED ANNEALING on alpha_z (optional). If `alpha_z_init` is
    given, alpha_z is linearly interpolated from `alpha_z_init` (small
    -> favours pairs, helps explore the edge graph) at sweep 0 to
    `alpha_z_final` (or setup.alpha_z if final not given) at sweep
    `int(anneal_fraction * n_sweeps)`, then held at the final value
    for the remaining sweeps. The setup's alpha_z is overwritten as
    the chain progresses; restored to the original on exit.

    Returns:
      Q_prime: (Lx, Ly) per-cell running mean of 1[(i, j) is Match
               in current alignment], over post-burn-in sweeps.
      diagnostics: MCMCDiagnostics.
    """
    rng = np.random.default_rng(seed)
    rng_key = jax.random.PRNGKey(seed)
    path = _initial_alignment(rng_key, setup, init_mode=init_mode)
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    diag = MCMCDiagnostics()
    diag.n_sweeps = n_sweeps
    diag.n_burnin = n_burnin

    Lx, Ly = setup.Lx, setup.Ly
    Q_acc = np.zeros((Lx, Ly), dtype=np.float64)
    n_recorded = 0
    # Pre-size edge-position accumulators (1-based indexing).
    diag.edge_pos_x_counts = [0] * (Lx + 1)
    diag.edge_pos_y_counts = [0] * (Ly + 1)

    # Annealing schedule. If not specified, just hold at setup.alpha_z.
    a0 = setup.alpha_z
    if alpha_z_final is None:
        alpha_z_final_eff = setup.alpha_z
    else:
        alpha_z_final_eff = float(alpha_z_final)
    do_anneal = (alpha_z_init is not None) and (anneal_fraction > 0.0)
    if do_anneal:
        anneal_end_sweep = max(1, int(anneal_fraction * n_sweeps))
    else:
        anneal_end_sweep = 0

    t0 = time.time()
    for sweep in range(n_sweeps):
        # Annealing: update setup.alpha_z in-place for this sweep.
        if do_anneal and sweep < anneal_end_sweep:
            f = sweep / anneal_end_sweep
            setup.alpha_z = float(alpha_z_init) * (1 - f) + alpha_z_final_eff * f
        else:
            setup.alpha_z = alpha_z_final_eff

        # 1) Pure-Gibbs path resample sweep (accept = 1 by construction).
        edge_anchor_positions = _edge_anchors_in_path_order(path, edges)
        path, n_p, n_a = _segment_resample_move(
            rng, setup, path, edge_anchor_positions)
        diag.n_propose_seg += n_p
        diag.n_accept_seg += n_a

        # 2) Edge add / remove MH.
        for _ in range(n_edge_moves_per_sweep):
            if rng.random() < 0.5:
                edges, acc = _edge_add_move(rng, setup, path, edges, k_max)
                diag.n_propose_add += 1
                if acc:
                    diag.n_accept_add += 1
            else:
                edges, acc = _edge_remove_move(rng, setup, path, edges)
                diag.n_propose_remove += 1
                if acc:
                    diag.n_accept_remove += 1

        # 3) Diagnostics + accumulator (only after burn-in).
        if sweep >= n_burnin and (sweep % record_every == 0):
            n_recorded += 1
            for (i, j) in _match_cells_of(path):
                if 1 <= i <= Lx and 1 <= j <= Ly:
                    Q_acc[i - 1, j - 1] += 1.0
            diag.log_pi_trace.append(
                _unnormalised_log_target(path, edges, setup))
            diag.n_edges_trace.append(len(edges))
            diag.n_match_trace.append(len(_match_cells_of(path)))
            # Edge marginal posterior accumulator: each edge endpoint
            # (cell on X x Y) contributes one count to its sequence-
            # position marginals AND to its (i, j) cell. With the
            # alignment marginalised out across sweeps, these become
            # P(position is on any edge | data) and P(cell is an edge
            # endpoint | data).
            diag.n_recorded_for_edges += 1
            for (a, b) in edges:
                ai, aj = a
                bi, bj = b
                if 1 <= ai <= Lx:
                    diag.edge_pos_x_counts[ai] += 1
                if 1 <= bi <= Lx:
                    diag.edge_pos_x_counts[bi] += 1
                if 1 <= aj <= Ly:
                    diag.edge_pos_y_counts[aj] += 1
                if 1 <= bj <= Ly:
                    diag.edge_pos_y_counts[bj] += 1
                diag.edge_cell_counts[(ai, aj)] = (
                    diag.edge_cell_counts.get((ai, aj), 0) + 1)
                diag.edge_cell_counts[(bi, bj)] = (
                    diag.edge_cell_counts.get((bi, bj), 0) + 1)
                # X-X / Y-Y unordered-pair projections.
                if 1 <= ai <= Lx and 1 <= bi <= Lx:
                    key_x = (min(ai, bi), max(ai, bi))
                    diag.edge_pair_x_counts[key_x] = (
                        diag.edge_pair_x_counts.get(key_x, 0) + 1)
                if 1 <= aj <= Ly and 1 <= bj <= Ly:
                    key_y = (min(aj, bj), max(aj, bj))
                    diag.edge_pair_y_counts[key_y] = (
                        diag.edge_pair_y_counts.get(key_y, 0) + 1)
        # Per-sweep cache-size snapshot (post-burnin records only;
        # uses the same record_every cadence as log_pi_trace above).
        if sweep >= n_burnin and (sweep % record_every == 0):
            diag.mu_cache_size_trace.append(len(setup.mu_cache))
        if verbose and (sweep + 1) % 100 == 0:
            mean_E = (np.mean(diag.n_edges_trace[-100:])
                      if diag.n_edges_trace else 0.0)
            print(f"  sweep {sweep + 1}/{n_sweeps}: alpha_z={setup.alpha_z:.1f} "
                  f"|E|={len(edges)} <|E|>={mean_E:.2f} "
                  f"acc_seg={diag.n_accept_seg / max(1, diag.n_propose_seg):.2f} "
                  f"acc_add={diag.n_accept_add / max(1, diag.n_propose_add):.2f} "
                  f"acc_rm={diag.n_accept_remove / max(1, diag.n_propose_remove):.2f}")

    # Restore setup.alpha_z so the caller's view is preserved.
    setup.alpha_z = a0
    diag.runtime_seconds = time.time() - t0
    _attach_timing_to_diag(diag, setup)
    if n_recorded == 0:
        Q_prime = np.zeros((Lx, Ly), dtype=np.float64)
    else:
        Q_prime = Q_acc / n_recorded
    return Q_prime, diag


# ===========================================================================
# Replica exchange (parallel tempering) on the alpha_z ladder.
# ===========================================================================
#
# Standard parallel tempering: K chains run in parallel, each at a different
# value of alpha_z (the size-{1,2}-truncated Ewens edge concentration). Cold
# rung (smallest alpha_z, target) prefers many edges; hot rungs (large
# alpha_z, hot = "sparse-edge") prefer fewer. Periodically propose to swap
# states between adjacent rungs.
#
# The hot rung at alpha_z -> infinity degenerates to baseline TKF92 sampling
# (no edges essentially), giving the cold chain access to alignment
# proposals that are "pure-baseline-FSA-style" and can shake it out of edge-
# induced multimodality.
#
# Swap MH ratio derivation. For two rungs at alpha_a < alpha_b with current
# states (path_a, edges_a) and (path_b, edges_b):
#   pi_k(state) propto pi_TKF92(path) * (1/alpha_k)^|edges| * prod_e M(e)
#
# The pi_TKF92 and prod_M factors cancel in the swap MH ratio (they don't
# depend on alpha_k); only the (1/alpha_k)^|E| factor enters:
#
#   alpha_swap = (alpha_b / alpha_a)^(|edges_b| - |edges_a|)
#
# Equivalently  log alpha_swap = (|edges_b| - |edges_a|) * (log alpha_b - log alpha_a)
#
# For typical equilibrium configurations (cold has more edges, hot has fewer),
# log alpha_swap < 0 and swaps are usually rejected. Rare swaps shuttle
# information between rungs.


def _swap_proposal(
        rng: np.random.Generator,
        setups: List["MCMCSetup"],
        states: List[Tuple[List, List]],
        ) -> Tuple[int, int, bool]:
    """Propose a swap between a random adjacent pair of rungs.

    Returns (rung_a, rung_b, accepted). If K < 2, returns (-1, -1, False).
    """
    K = len(setups)
    if K < 2:
        return -1, -1, False
    a = int(rng.integers(0, K - 1))
    b = a + 1
    _, edges_a = states[a]
    _, edges_b = states[b]
    log_alpha_a = float(np.log(setups[a].alpha_z))
    log_alpha_b = float(np.log(setups[b].alpha_z))
    log_ratio = (len(edges_b) - len(edges_a)) * (log_alpha_b - log_alpha_a)
    log_ratio = min(0.0, log_ratio)
    u = float(rng.random())
    if np.log(max(u, 1e-300)) < log_ratio:
        states[a], states[b] = states[b], states[a]
        return a, b, True
    return a, b, False


def run_replica_exchange_chain(
        setup_template: "MCMCSetup",
        alpha_z_ladder: List[float],
        n_sweeps: int = 1000,
        n_burnin: int = 200,
        k_max: int = -1,
        n_edge_moves_per_sweep: int = 8,
        init_mode: str = "viterbi",
        seed: int = 0,
        record_every: int = 1,
        verbose: bool = False,
        swap_every: int = 10,
        ) -> Tuple[np.ndarray, Dict]:
    """Replica-exchange MCMC on the alpha_z ladder.

    Args:
      setup_template: MCMCSetup whose alpha_z field will be overridden per rung.
      alpha_z_ladder: list of alpha_z values; smallest = cold (target), largest = hot.
      swap_every: sweep frequency for swap proposals (default every 10 sweeps).
      Other args: same as run_mcmc_chain (cold-rung sweep budget; per-rung
      sweeps run on each).

    Returns:
      Q_prime: cold-rung per-cell Q' over post-burn-in samples.
      diagnostics: dict with per-rung MCMCDiagnostics, swap stats.
    """
    K = len(alpha_z_ladder)
    if K < 1:
        raise ValueError("alpha_z_ladder must have at least 1 entry")
    alpha_z_sorted = sorted(alpha_z_ladder)  # cold first
    # Per-rung setup (shares F_partial / M_obs; only alpha_z differs).
    setups: List[MCMCSetup] = []
    for alpha in alpha_z_sorted:
        # Shallow copy via dataclass replace; share underlying jnp arrays.
        # mu_cache: each rung gets its own dict starting from a SHALLOW
        # COPY of the prepopulated cache. Cache entries are immutable
        # numpy arrays, so the copy is cheap and per-rung writes are
        # isolated. Sharing entries across rungs is correct because
        # restart-Forward depends only on the Pair-HMM tables (which all
        # rungs share) and the anchor coords -- not on alpha_z.
        from dataclasses import replace as _dc_replace
        s = _dc_replace(setup_template, alpha_z=float(alpha),
                        mu_cache=dict(setup_template.mu_cache),
                        rf_seconds=0.0, rf_n_misses=0, rf_n_hits=0,
                        tb_seconds=0.0, tb_n_calls=0)
        setups.append(s)
    # Per-rung initial state.
    rng = np.random.default_rng(seed)
    rng_keys = jax.random.split(jax.random.PRNGKey(seed), K)
    states: List[Tuple[List, List]] = []
    diags: List[MCMCDiagnostics] = []
    Lx, Ly = setup_template.Lx, setup_template.Ly
    for k in range(K):
        path = _initial_alignment(rng_keys[k], setups[k], init_mode=init_mode)
        edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        states.append((path, edges))
        d = MCMCDiagnostics()
        d.n_sweeps = n_sweeps
        d.n_burnin = n_burnin
        # Pre-size edge-position accumulators on every rung; we record
        # only into the cold-rung diag below, but allocating on each
        # makes per-rung post-hoc analysis symmetric should we ever
        # extend recording to other rungs.
        d.edge_pos_x_counts = [0] * (Lx + 1)
        d.edge_pos_y_counts = [0] * (Ly + 1)
        diags.append(d)
    Q_acc = np.zeros((Lx, Ly), dtype=np.float64)
    n_recorded = 0
    swap_n_propose = [0] * max(K - 1, 0)
    swap_n_accept = [0] * max(K - 1, 0)
    # Label-by-rung trajectory: rung_of_label[k] is the current rung that
    # label k occupies. Initially label k starts at rung k. On accepted
    # swap (a, a+1) we swap the labels at those rung positions, which is
    # equivalent to flipping the rung index of the two affected labels.
    labels_at_rung = list(range(K))   # labels_at_rung[r] = label at rung r
    rung_traj: List[List[int]] = []   # one row per sweep; rung_traj[s][l] = rung of label l at sweep s
    t0 = time.time()
    for sweep in range(n_sweeps):
        # 1) Per-rung Gibbs+MH sweep.
        for k in range(K):
            path, edges = states[k]
            edge_anchors = _edge_anchors_in_path_order(path, edges)
            path, n_p, n_a = _segment_resample_move(
                rng, setups[k], path, edge_anchors)
            diags[k].n_propose_seg += n_p
            diags[k].n_accept_seg += n_a
            for _ in range(n_edge_moves_per_sweep):
                if rng.random() < 0.5:
                    edges, acc = _edge_add_move(
                        rng, setups[k], path, edges, k_max)
                    diags[k].n_propose_add += 1
                    if acc:
                        diags[k].n_accept_add += 1
                else:
                    edges, acc = _edge_remove_move(
                        rng, setups[k], path, edges)
                    diags[k].n_propose_remove += 1
                    if acc:
                        diags[k].n_accept_remove += 1
            states[k] = (path, edges)
        # 2) Swap proposal.
        if K > 1 and (sweep + 1) % swap_every == 0:
            a, b, accepted = _swap_proposal(rng, setups, states)
            if a >= 0:
                swap_n_propose[a] += 1
                if accepted:
                    swap_n_accept[a] += 1
                    # Permute the rung-label tracking so we can later
                    # compute round-trip times.
                    labels_at_rung[a], labels_at_rung[a + 1] = (
                        labels_at_rung[a + 1], labels_at_rung[a])
        # 2b) Record rung-of-each-label after any swap. Cheap: K ints.
        if K > 1:
            rung_of_label = [0] * K
            for r_idx, lab in enumerate(labels_at_rung):
                rung_of_label[lab] = r_idx
            rung_traj.append(rung_of_label)
        # 3) Cold-rung accumulator + diagnostics (post-burn-in only).
        if sweep >= n_burnin and (sweep % record_every == 0):
            n_recorded += 1
            cold_path, cold_edges = states[0]
            for (i, j) in _match_cells_of(cold_path):
                if 1 <= i <= Lx and 1 <= j <= Ly:
                    Q_acc[i - 1, j - 1] += 1.0
            diags[0].log_pi_trace.append(
                _unnormalised_log_target(cold_path, cold_edges, setups[0]))
            diags[0].n_edges_trace.append(len(cold_edges))
            diags[0].n_match_trace.append(len(_match_cells_of(cold_path)))
            # Cold-rung edge marginal posterior accumulator. Same logic
            # as run_mcmc_chain above; we marginalise the cold-chain
            # samples over alignment to get P(position has an edge |
            # data) and P(cell is an edge endpoint | data).
            diags[0].n_recorded_for_edges += 1
            for (a, b) in cold_edges:
                ai, aj = a
                bi, bj = b
                if 1 <= ai <= Lx:
                    diags[0].edge_pos_x_counts[ai] += 1
                if 1 <= bi <= Lx:
                    diags[0].edge_pos_x_counts[bi] += 1
                if 1 <= aj <= Ly:
                    diags[0].edge_pos_y_counts[aj] += 1
                if 1 <= bj <= Ly:
                    diags[0].edge_pos_y_counts[bj] += 1
                diags[0].edge_cell_counts[(ai, aj)] = (
                    diags[0].edge_cell_counts.get((ai, aj), 0) + 1)
                diags[0].edge_cell_counts[(bi, bj)] = (
                    diags[0].edge_cell_counts.get((bi, bj), 0) + 1)
                # X-X / Y-Y unordered-pair projections (cold-rung only).
                if 1 <= ai <= Lx and 1 <= bi <= Lx:
                    key_x = (min(ai, bi), max(ai, bi))
                    diags[0].edge_pair_x_counts[key_x] = (
                        diags[0].edge_pair_x_counts.get(key_x, 0) + 1)
                if 1 <= aj <= Ly and 1 <= bj <= Ly:
                    key_y = (min(aj, bj), max(aj, bj))
                    diags[0].edge_pair_y_counts[key_y] = (
                        diags[0].edge_pair_y_counts.get(key_y, 0) + 1)
            # Per-rung cache-size snapshot: shows how the prepop set's
            # targeting holds up as the chain explores. mu_cache_size[t]
            # = prepop_n_anchors + (chain-driven additions so far on
            # this rung).
            for k in range(K):
                diags[k].mu_cache_size_trace.append(len(setups[k].mu_cache))
        if verbose and (sweep + 1) % 100 == 0:
            mean_E_cold = (np.mean(diags[0].n_edges_trace[-100:])
                           if diags[0].n_edges_trace else 0.0)
            swap_acc = [n_a / max(1, n_p)
                        for n_a, n_p in zip(swap_n_accept, swap_n_propose)]
            print(f"  sweep {sweep+1}/{n_sweeps}: "
                  f"|E|=cold:{len(states[0][1])} hot:{len(states[-1][1])} "
                  f"<|E_cold|>={mean_E_cold:.2f} "
                  f"swap_acc={[f'{x:.2f}' for x in swap_acc]}")
    runtime = time.time() - t0
    if n_recorded == 0:
        Q_prime = np.zeros((Lx, Ly), dtype=np.float64)
    else:
        Q_prime = Q_acc / n_recorded
    # Wire per-rung runtime + setup/cache instrumentation into each rung's
    # MCMCDiagnostics so they surface in the JSON output. The setups[k]
    # objects hold the per-rung mu_cache / rf_seconds / tb_seconds counters
    # mutated by the hot loop. Without this, the per-rung diag fields
    # remain at their dataclass defaults (zero) -- bug noticed
    # 2026-05-15 after first BB12041 family flush.
    for k in range(K):
        diags[k].runtime_seconds = runtime
        _attach_timing_to_diag(diags[k], setups[k])
    diagnostics = dict(
        per_rung=diags,
        alpha_z_ladder=alpha_z_sorted,
        swap_n_propose=swap_n_propose,
        swap_n_accept=swap_n_accept,
        rung_traj=rung_traj,
        runtime_seconds=runtime,
    )
    return Q_prime, diagnostics


# ===========================================================================
# Multi-chain convergence-aware driver.
# ===========================================================================


def run_mcmc_multi_chain(
        setup: MCMCSetup,
        n_sweeps: int = 1000,
        n_burnin: int = 200,
        n_chains: int = 1,
        k_max: int = -1,
        n_edge_moves_per_sweep: int = 8,
        init_mode: str = "viterbi",
        seed: int = 0,
        verbose: bool = False,
        auto_burnin: bool = False,
        burnin_window: int = 100,
        burnin_tol: float = 1e-2,
        alpha_z_init: Optional[float] = None,
        alpha_z_final: Optional[float] = None,
        anneal_fraction: float = 0.0,
        ) -> Tuple[np.ndarray, Dict]:
    """Run n_chains independent MCMC chains; return averaged Q_prime
    and a combined diagnostics dict.

    auto_burnin (H3): if True, choose burn-in adaptively per chain by
    monitoring when the running per-cell Q' stabilises (window of size
    burnin_window). Otherwise use the fixed n_burnin.

    Annealing parameters (alpha_z_init, alpha_z_final, anneal_fraction)
    are forwarded to run_mcmc_chain.
    """
    Q_chains = []
    diags = []
    for c in range(n_chains):
        chain_seed = seed * 1000003 + c
        if auto_burnin:
            Q, d = _run_chain_auto_burnin(
                setup, max_sweeps=n_sweeps + n_burnin,
                k_max=k_max, n_edge_moves_per_sweep=n_edge_moves_per_sweep,
                init_mode=init_mode, seed=chain_seed,
                window=burnin_window, tol=burnin_tol, verbose=verbose)
        else:
            Q, d = run_mcmc_chain(
                setup, n_sweeps=n_sweeps, n_burnin=n_burnin,
                k_max=k_max, n_edge_moves_per_sweep=n_edge_moves_per_sweep,
                init_mode=init_mode, seed=chain_seed, verbose=verbose,
                alpha_z_init=alpha_z_init, alpha_z_final=alpha_z_final,
                anneal_fraction=anneal_fraction)
        Q_chains.append(Q)
        diags.append(d)
    Q_mean = np.mean(np.stack(Q_chains, axis=0), axis=0)
    Q_var = np.var(np.stack(Q_chains, axis=0), axis=0) if n_chains > 1 else np.zeros_like(Q_mean)

    out_diag = {
        "n_chains": n_chains,
        "per_chain": diags,
        "Q_chain_var": Q_var,
        "Q_chain_max": np.max(np.stack(Q_chains, axis=0), axis=0)
                        if n_chains > 0 else None,
        "Q_chain_min": np.min(np.stack(Q_chains, axis=0), axis=0)
                        if n_chains > 0 else None,
    }
    return Q_mean, out_diag


def _run_chain_auto_burnin(
        setup: MCMCSetup,
        max_sweeps: int,
        k_max: int = -1,
        n_edge_moves_per_sweep: int = 8,
        init_mode: str = "viterbi",
        seed: int = 0,
        window: int = 100,
        tol: float = 1e-2,
        verbose: bool = False,
        ) -> Tuple[np.ndarray, MCMCDiagnostics]:
    """Run a chain for at most max_sweeps; chop off auto-detected burn-in.

    Convergence criterion: total log-prob's running mean over window W
    changes by < tol * |running_mean| between consecutive windows.
    """
    rng = np.random.default_rng(seed)
    rng_key = jax.random.PRNGKey(seed)
    path = _initial_alignment(rng_key, setup, init_mode=init_mode)
    edges = []
    diag = MCMCDiagnostics()
    diag.n_sweeps = max_sweeps

    Lx, Ly = setup.Lx, setup.Ly
    Q_acc = np.zeros((Lx, Ly), dtype=np.float64)
    log_pi_trace = []
    n_edges_trace = []
    n_match_trace = []
    n_recorded = 0
    burnin_done = False
    burnin_idx = max_sweeps  # default if never converges

    t0 = time.time()
    for sweep in range(max_sweeps):
        edge_anchor_positions = _edge_anchors_in_path_order(path, edges)
        path, n_p, n_a = _segment_resample_move(
            rng, setup, path, edge_anchor_positions)
        diag.n_propose_seg += n_p
        diag.n_accept_seg += n_a
        for _ in range(n_edge_moves_per_sweep):
            if rng.random() < 0.5:
                edges, acc = _edge_add_move(rng, setup, path, edges, k_max)
                diag.n_propose_add += 1
                if acc:
                    diag.n_accept_add += 1
            else:
                edges, acc = _edge_remove_move(rng, setup, path, edges)
                diag.n_propose_remove += 1
                if acc:
                    diag.n_accept_remove += 1
        # Trace.
        log_pi_trace.append(_unnormalised_log_target(path, edges, setup))
        n_edges_trace.append(len(edges))
        n_match_trace.append(len(_match_cells_of(path)))
        # Convergence check.
        if not burnin_done and len(log_pi_trace) >= 2 * window:
            recent = np.array(log_pi_trace[-window:])
            prev = np.array(log_pi_trace[-2 * window:-window])
            if (np.abs(recent.mean() - prev.mean())
                    < tol * max(abs(recent.mean()), 1.0)):
                burnin_done = True
                burnin_idx = sweep + 1
                if verbose:
                    print(f"  auto-burnin at sweep {sweep + 1}")
        if burnin_done:
            n_recorded += 1
            for (i, j) in _match_cells_of(path):
                if 1 <= i <= Lx and 1 <= j <= Ly:
                    Q_acc[i - 1, j - 1] += 1.0
        if verbose and (sweep + 1) % 100 == 0:
            print(f"  sweep {sweep + 1}/{max_sweeps}: "
                  f"burnin_done={burnin_done} recorded={n_recorded}")
    diag.runtime_seconds = time.time() - t0
    diag.n_burnin = burnin_idx
    diag.log_pi_trace = log_pi_trace
    diag.n_edges_trace = n_edges_trace
    diag.n_match_trace = n_match_trace
    _attach_timing_to_diag(diag, setup)
    if n_recorded == 0:
        Q_prime = np.zeros((Lx, Ly), dtype=np.float64)
    else:
        Q_prime = Q_acc / n_recorded
    return Q_prime, diag


# ===========================================================================
# Public end-to-end API (mirrors aug_phmm_corrected_posterior).
# ===========================================================================


def mcmc_corrected_posterior(
        x_seq: np.ndarray, y_seq: np.ndarray, t: float,
        ins_rate: float, del_rate: float, ext: float,
        Q_lg, pi_lg, boost_state,
        alpha_z: float = 100.0,
        n_sweeps: int = 1000,
        n_burnin: int = 200,
        n_chains: int = 1,
        k_max: int = -1,
        seed: int = 0,
        init_mode: str = "viterbi",
        n_edge_moves_per_sweep: int = 8,
        auto_burnin: bool = False,
        verbose: bool = False,
        q_min: float = 0.0,    # ignored; included for API parity.
        alpha_z_init: Optional[float] = None,
        alpha_z_final: Optional[float] = None,
        anneal_fraction: float = 0.0,
        alpha_z_ladder: Optional[List[float]] = None,
        swap_every: int = 10,
        prepop_top_k: int = -1,
        prepop_chunk: int = 256,
        prepop_mem_budget_mib: float = 2048.0,
        ) -> Tuple[np.ndarray, Optional[float], np.ndarray, float, Dict]:
    """End-to-end MCMC sampler API.

    Same per-pair signature style as
    ``aug_phmm.aug_phmm_corrected_posterior``, plus MCMC-specific kwargs.

    Kernel: pure-Gibbs path resample between adjacent edge anchors +
    MH edge add/remove with per-edge weight eps = 1/alpha_z (the
    size-{1,2}-truncated Ewens / equivalently bounded-eps prior; both
    parameterisations are mathematically identical for this prior).

    Annealing (optional):
      alpha_z_init: starting alpha_z (often small; favours pairs).
      alpha_z_final: ending alpha_z; defaults to `alpha_z`.
      anneal_fraction: fraction of n_sweeps spent ramping. 0 = no anneal.

    Returns:
      Q_prime: (Lx, Ly) MCMC running mean of 1[Match at (i, j)].
      L_exact_est: None (per H14, partition function not estimated).
      Q_baseline: (Lx, Ly) F1/F0 baseline from the no-edge model.
      log_F0: scalar baseline log-partition.
      mcmc_diag: dict of MCMC diagnostics.
    """
    setup = precompute_partial_forward(
        x_seq, y_seq, t, ins_rate, del_rate, ext, Q_lg, pi_lg, boost_state,
        alpha_z=alpha_z,
        prepop_top_k=prepop_top_k,
        prepop_chunk=prepop_chunk,
        prepop_mem_budget_mib=prepop_mem_budget_mib)

    if alpha_z_ladder is not None and len(alpha_z_ladder) > 1:
        # Replica-exchange path. Cold rung is min(ladder) (target).
        # Note: n_chains is ignored under RE — output is the cold rung only.
        Q_prime, diag = run_replica_exchange_chain(
            setup, alpha_z_ladder=alpha_z_ladder,
            n_sweeps=n_sweeps, n_burnin=n_burnin, k_max=k_max,
            n_edge_moves_per_sweep=n_edge_moves_per_sweep,
            init_mode=init_mode, seed=seed, verbose=verbose,
            swap_every=swap_every)
    else:
        Q_prime, diag = run_mcmc_multi_chain(
            setup, n_sweeps=n_sweeps, n_burnin=n_burnin,
            n_chains=n_chains, k_max=k_max,
            n_edge_moves_per_sweep=n_edge_moves_per_sweep,
            init_mode=init_mode, seed=seed,
            auto_burnin=auto_burnin, verbose=verbose,
            alpha_z_init=alpha_z_init, alpha_z_final=alpha_z_final,
            anneal_fraction=anneal_fraction)

    # Baseline Q from the setup.
    log_alpha = np.asarray(setup.log_alpha)
    log_beta = np.asarray(setup.log_beta)
    log_F1 = log_alpha[1:setup.Lx + 1, 1:setup.Ly + 1, M] \
        + log_beta[1:setup.Lx + 1, 1:setup.Ly + 1, M]
    Q_baseline = np.exp(log_F1 - setup.log_F0)
    return Q_prime, None, Q_baseline, setup.log_F0, diag
