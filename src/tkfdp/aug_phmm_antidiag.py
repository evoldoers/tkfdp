"""Antidiagonal-wavefront variant of the memory-augmented Pair HMM.

Mathematically identical to ``src/tkfdp/aug_phmm.py``: same Q'_{ij},
same L_exact, same posterior formula
``Q'_{ij} = sum_t alpha[i, j, M, t] * beta[i, j, M, t] / L_exact``.
The only difference is the DP traversal order. The row-scan version has
limited intra-row parallelism (one cell at a time inside a row); this
version processes all O(min(Lx, Ly)) cells along each antidiagonal in
parallel via ``jax.vmap``, giving O(Lx + Ly) sequential ``scan`` steps
on GPU.

Public API mirrors ``aug_phmm`` exactly. Both the forward and backward
DP carry only the *two* most recent antidiagonals; the full
``(Lx_pad+1, Ly_pad+1, 5, N_TAGS)`` table is reconstructed by scattering
the per-diagonal scan outputs back through a ``fori_loop``. Cross-
validated against ``aug_phmm.aug_phmm_corrected_posterior`` to ~1e-10.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Pull tkfmixdom's TKF92 Pair HMM builder + state codes (do not modify).
TKFMIXDOM_ROOT = Path.home() / "tkf-mixdom" / "python"
if str(TKFMIXDOM_ROOT) not in sys.path:
    sys.path.insert(0, str(TKFMIXDOM_ROOT))
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm  # noqa: E402
from tkfmixdom.jax.dp.hmm import (                                  # noqa: E402
    pair_hmm_emissions, _pad_to_bin, _pad_seq, _emit_mask,
    _find_e_idx, NEG_INF,
)
from tkfmixdom.jax.core.params import S, M, I, D, E                  # noqa: E402

# Re-export the tag-layout constants and the M-tensor builder so callers
# can use this module as a drop-in replacement for `aug_phmm`.
from .aug_phmm import (                                              # noqa: E402
    A, TAG_NO_EDGE, N_AB_TAGS, TAG_DONE, N_TAGS,
    _tag_ab, build_M_tensor_aa_marginal,
)


# ---------------------------------------------------------------------------
# Antidiagonal indexing helpers
# ---------------------------------------------------------------------------
#
# Each antidiagonal d (= i + j) is parameterised by its local index k in
# [0, D_max) where D_max = min(Lx_pad, Ly_pad) + 1. The cell at slot k of
# diagonal d is (i, j) = (i_min(d) + k, d - i_min(d) - k) with
#   i_min(d) = max(0, d - Ly_pad).
# Cells with k beyond the actual diagonal length are masked NEG_INF.


def _i_min(d, Ly_pad):
    return jnp.maximum(0, d - Ly_pad)


# ---------------------------------------------------------------------------
# Forward DP (antidiagonal wavefront).
# ---------------------------------------------------------------------------


def _aug_forward_core_antidiag(log_trans, state_types, emit, log_M_tensor,
                                x_pad, y_pad, log_eps, Lx_pad, Ly_pad):
    """Augmented Forward in log space using antidiagonal scan + vmap.

    Same arguments and return signature as
    ``aug_phmm._aug_forward_core``. ``Lx_pad``/``Ly_pad`` must be Python
    ints (static argnames at the JIT layer).
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)
    D_max = min(Lx_pad, Ly_pad) + 1
    n_diags = Lx_pad + Ly_pad  # diagonals 1..Lx_pad+Ly_pad

    # Diagonal-d=0 init: only cell (0, 0) with alpha[S, no_edge] = 0.
    diag0 = jnp.full((D_max, ns, N_TAGS), NEG_INF)
    diag0 = diag0.at[0, S, TAG_NO_EDGE].set(0.0)

    # --- Per-cell update -----------------------------------------------------
    def step_from(prev):
        """prev shape (ns, N_TAGS) -> (ns, N_TAGS): logsumexp of
        prev[k, t] + log_trans[k, s'] over predecessor state k."""
        return jax.nn.logsumexp(
            prev[:, None, :] + log_trans[:, :, None], axis=0)

    def compute_cell(prev_diag, prev_prev_diag, d, k):
        """Compute alpha at diagonal d, slot k. Returns (cell, extra_cd,
        extra_done) where ``cell`` is shape (ns, N_TAGS) and the two
        extras are scalar log-values (the per-Match-cell left/right
        edge contributions; NEG_INF at non-Match positions).
        """
        i_min_d = _i_min(d, Ly_pad)
        i = i_min_d + k
        j = d - i

        # M predecessor: (i-1, j-1) on diagonal d-2.
        m_k = (i - 1) - _i_min(d - 2, Ly_pad)
        m_k_safe = jnp.clip(m_k, 0, D_max - 1)
        prev_M_cell = prev_prev_diag[m_k_safe]                     # (ns, N_TAGS)
        step_M_pred = step_from(prev_M_cell)                       # (ns, N_TAGS)

        # I predecessor: (i, j-1) on diagonal d-1.
        i_k = i - _i_min(d - 1, Ly_pad)
        i_k_safe = jnp.clip(i_k, 0, D_max - 1)
        prev_I_cell = prev_diag[i_k_safe]
        step_I_pred = step_from(prev_I_cell)

        # D predecessor: (i-1, j) on diagonal d-1.
        d_k = (i - 1) - _i_min(d - 1, Ly_pad)
        d_k_safe = jnp.clip(d_k, 0, D_max - 1)
        prev_D_cell = prev_diag[d_k_safe]
        step_D_pred = step_from(prev_D_cell)

        emit_cell = emit[i, j]                                     # (ns,)
        full_M = step_M_pred + emit_cell[:, None]                  # (ns, N_TAGS)
        full_I = step_I_pred + emit_cell[:, None]
        full_D = step_D_pred + emit_cell[:, None]

        # Boundary masks: a destination state is reachable from a given
        # predecessor only if that predecessor cell exists.
        m_ok = (i >= 1) & (j >= 1)
        i_ok = (j >= 1)
        d_ok = (i >= 1)
        full_M = jnp.where(m_ok, full_M, NEG_INF)
        full_I = jnp.where(i_ok, full_I, NEG_INF)
        full_D = jnp.where(d_ok, full_D, NEG_INF)

        # Each destination state s' gets its single "from" contribution
        # determined by its type (M / I / D / S / E).
        cell = jnp.where(is_M[:, None], full_M,
                 jnp.where(is_I[:, None], full_I,
                   jnp.where(is_D[:, None], full_D, NEG_INF)))

        # Match-cell tag extras (only when (i, j) is a Match cell —
        # i.e., i >= 1 and j >= 1; otherwise (c_idx, d_idx) is undefined
        # and we mask the extras to NEG_INF).
        i_aa = jnp.clip(i - 1, 0, Lx_pad - 1)
        j_aa = jnp.clip(j - 1, 0, Ly_pad - 1)
        c_idx = x_pad[i_aa]
        d_idx = y_pad[j_aa]
        is_match_cell = m_ok                                       # i, j >= 1

        emit_M_cd = emit_cell[M]                                   # scalar
        match_step_per_tag = step_M_pred[M, :]                     # (N_TAGS,)
        cd_tag = 1 + c_idx * A + d_idx                             # scalar
        extra_cd = emit_M_cd + log_eps + match_step_per_tag[TAG_NO_EDGE]
        ab_steps = match_step_per_tag[1:1 + N_AB_TAGS]             # (A*A,)
        log_M_cd_flat = log_M_tensor[:, :, c_idx, d_idx].reshape(N_AB_TAGS)
        extra_done = emit_M_cd + jax.nn.logsumexp(ab_steps + log_M_cd_flat)

        # Mask extras to NEG_INF at non-Match cells (i==0 or j==0).
        extra_cd = jnp.where(is_match_cell, extra_cd, NEG_INF)
        extra_done = jnp.where(is_match_cell, extra_done, NEG_INF)

        # Apply extras to the M-state row (only if this is a Match cell).
        # Build a per-tag delta that's NEG_INF except at cd_tag and TAG_DONE.
        delta = jnp.full((N_TAGS,), NEG_INF)
        delta = delta.at[TAG_DONE].set(extra_done)
        # cd_tag contribution; if not a Match cell, extra_cd is already
        # NEG_INF so the dynamic-update is a no-op in log-add.
        delta = delta.at[cd_tag].set(
            jnp.logaddexp(delta[cd_tag], extra_cd))
        new_M_row = jnp.logaddexp(cell[M, :], delta)
        cell = cell.at[M, :].set(new_M_row)
        return cell, extra_cd, extra_done

    def scan_fn(carry, d):
        prev_diag, prev_prev_diag = carry                          # (D_max, ns, N_TAGS)
        ks = jnp.arange(D_max)
        i_vals = _i_min(d, Ly_pad) + ks
        j_vals = d - i_vals
        valid = (i_vals <= Lx_pad) & (j_vals >= 0) & (j_vals <= Ly_pad)

        cells, extras_cd, extras_done = jax.vmap(
            lambda k: compute_cell(prev_diag, prev_prev_diag, d, k))(ks)

        # cells: (D_max, ns, N_TAGS). Mask invalid slots.
        cells = jnp.where(valid[:, None, None], cells, NEG_INF)
        extras_cd = jnp.where(valid, extras_cd, NEG_INF)
        extras_done = jnp.where(valid, extras_done, NEG_INF)
        return (cells, prev_diag), (cells, extras_cd, extras_done)

    init_carry = (diag0, jnp.full((D_max, ns, N_TAGS), NEG_INF))
    (_, _), (all_diags, all_extras_cd, all_extras_done) = jax.lax.scan(
        scan_fn, init_carry, jnp.arange(1, n_diags + 1))
    # all_diags:        (n_diags, D_max, ns, N_TAGS) for d = 1..Lx_pad+Ly_pad
    # all_extras_cd:    (n_diags, D_max)
    # all_extras_done:  (n_diags, D_max)

    # --- Scatter back into the full grid -------------------------------------
    # alpha shape (Lx_pad+1, Ly_pad+1, ns, N_TAGS); seed (0, 0).
    alpha = jnp.full((Lx_pad + 1, Ly_pad + 1, ns, N_TAGS), NEG_INF)
    alpha = alpha.at[0, 0, S, TAG_NO_EDGE].set(0.0)

    n_flat = (Lx_pad + 1) * (Ly_pad + 1)
    alpha_flat = jnp.concatenate([
        alpha.reshape(n_flat, ns, N_TAGS),
        jnp.full((1, ns, N_TAGS), NEG_INF),
    ], axis=0)
    left_flat = jnp.full((n_flat + 1,), NEG_INF)
    right_flat = jnp.full((n_flat + 1,), NEG_INF)

    def _scatter_one_diag(idx, carry):
        af, lf, rf = carry
        d = idx + 1                                                # 1..n_diags
        diag_data = all_diags[idx]                                 # (D_max, ns, N_TAGS)
        ex_cd = all_extras_cd[idx]                                 # (D_max,)
        ex_done = all_extras_done[idx]                             # (D_max,)
        i_min_d = jnp.maximum(0, d - Ly_pad)
        ks = jnp.arange(D_max)
        i_vals = i_min_d + ks
        j_vals = d - i_vals
        valid = (i_vals <= Lx_pad) & (j_vals >= 0) & (j_vals <= Ly_pad)
        lin = i_vals * (Ly_pad + 1) + j_vals
        lin_safe = jnp.where(valid, lin, n_flat)                   # invalid -> dummy
        af = af.at[lin_safe].set(diag_data)
        lf = lf.at[lin_safe].set(ex_cd)
        rf = rf.at[lin_safe].set(ex_done)
        return (af, lf, rf)

    alpha_flat, left_flat, right_flat = jax.lax.fori_loop(
        0, n_diags, _scatter_one_diag, (alpha_flat, left_flat, right_flat))
    alpha = alpha_flat[:n_flat].reshape(Lx_pad + 1, Ly_pad + 1, ns, N_TAGS)
    alpha = alpha.at[0, 0, S, TAG_NO_EDGE].set(0.0)
    alpha_left_edge = left_flat[:n_flat].reshape(Lx_pad + 1, Ly_pad + 1)
    alpha_right_edge = right_flat[:n_flat].reshape(Lx_pad + 1, Ly_pad + 1)
    return alpha, alpha_left_edge, alpha_right_edge


# ---------------------------------------------------------------------------
# Backward DP (antidiagonal wavefront).
# ---------------------------------------------------------------------------


def _aug_backward_core_antidiag(log_trans, state_types, emit, log_M_tensor,
                                 x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                                 real_Lx, real_Ly):
    """Augmented Backward in log space using antidiagonal scan + vmap.

    Mirror of ``_aug_forward_core_antidiag`` walking d from (Lx_pad+Ly_pad)
    down to 0. Carries the next two antidiagonals.

    Same convention as ``aug_phmm._aug_backward_core``:
    ``beta[i, j, s, t]`` = log P(suffix | currently at (i, j, s) carrying
    outgoing tag t), excluding the emission at (i, j, s) itself.
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    e_idx = _find_e_idx(state_types)
    D_max = min(Lx_pad, Ly_pad) + 1
    n_diags_total = Lx_pad + Ly_pad + 1                            # 0..Lx_pad+Ly_pad

    # Terminal: at (real_Lx, real_Ly), beta[s, t] = log_trans[s, E] for
    # t in {no_edge, done}; NEG_INF elsewhere (orphan (a, b) tags
    # forbidden under the SCFG grammar).
    term_d = real_Lx + real_Ly
    term_k = real_Lx - jnp.maximum(0, term_d - Ly_pad)
    term_mask = jnp.zeros(N_TAGS, dtype=bool).at[TAG_NO_EDGE].set(True
                                              ).at[TAG_DONE].set(True)
    beta_term_st = jnp.where(
        term_mask[None, :],
        log_trans[:, e_idx, None] + jnp.zeros((1, N_TAGS)),
        NEG_INF)                                                   # (ns, N_TAGS)

    def _make_term_diag(d):
        """Build a per-diagonal slot tensor that places beta_term at slot
        term_k iff d == term_d, else all NEG_INF."""
        empty = jnp.full((D_max, ns, N_TAGS), NEG_INF)
        # slot index per k: zeros vector shape (D_max,), set [term_k] when d matches.
        match_d = (d == term_d)
        # Build: for each slot k, emit beta_term_st if (k == term_k && match_d).
        ks = jnp.arange(D_max)
        is_term_slot = (ks == term_k) & match_d
        diag = jnp.where(is_term_slot[:, None, None],
                          beta_term_st[None, :, :],
                          empty)
        return diag

    # Top-of-scan init: we walk d from (Lx_pad + Ly_pad) down to 0. The
    # first d processed is (Lx_pad + Ly_pad). Its successors are at
    # diagonals (d+1) and (d+2), both beyond range, so the "next" and
    # "next_next" carry diagonals are empty — except that the terminal
    # condition for d == term_d must be applied INSIDE the cell update
    # (we do this by overlaying the terminal condition at the end of
    # each step rather than via a successor read).

    def back_step_match(succ_M_cell, succ_emit_M, c_succ, d_succ):
        """Backward contribution from M-successor at (i+1, j+1).

        succ_M_cell: (N_TAGS,) -- next-diag-cell[M, :].
        succ_emit_M: scalar -- emit at successor cell, M-state.
        Returns shape (ns, N_TAGS): contribution to source cell.
        """
        cd_succ_tag = 1 + c_succ * A + d_succ
        contrib = jnp.full((N_TAGS,), NEG_INF)
        # Tag no_edge: stay-no_edge OR spawn left edge into (c', d').
        contrib_no_edge = jnp.logaddexp(
            succ_M_cell[TAG_NO_EDGE],
            log_eps + succ_M_cell[cd_succ_tag])
        contrib = contrib.at[TAG_NO_EDGE].set(contrib_no_edge)
        # Tag done: carry through.
        contrib = contrib.at[TAG_DONE].set(succ_M_cell[TAG_DONE])
        # Tag (a, b): carry through OR resolve via M(a, b; c', d') -> done.
        ab_next = succ_M_cell[1:1 + N_AB_TAGS]
        log_M_succ = log_M_tensor[:, :, c_succ, d_succ].reshape(N_AB_TAGS)
        contrib_ab = jnp.logaddexp(
            ab_next, log_M_succ + succ_M_cell[TAG_DONE])
        contrib = jax.lax.dynamic_update_slice(contrib, contrib_ab, (1,))
        # Add log_trans[s, M] + emit[M] to lift to source side.
        return log_trans[:, M:M + 1] + succ_emit_M + contrib[None, :]

    def compute_cell_bwd(next_diag, next_next_diag, d, k):
        """Compute beta at diagonal d, slot k. Returns (ns, N_TAGS)."""
        i_min_d = _i_min(d, Ly_pad)
        i = i_min_d + k
        j = d - i

        # M successor: (i+1, j+1) on d+2.
        m_k = (i + 1) - _i_min(d + 2, Ly_pad)
        m_k_safe = jnp.clip(m_k, 0, D_max - 1)
        succ_M_full = next_next_diag[m_k_safe]                     # (ns, N_TAGS)
        succ_M_cell = succ_M_full[M, :]                            # (N_TAGS,)
        i_succ = jnp.clip(i + 1, 0, Lx_pad)
        j_succ = jnp.clip(j + 1, 0, Ly_pad)
        succ_emit_M = emit[i_succ, j_succ, M]                      # scalar
        # AAs at the successor Match cell:
        i_aa = jnp.clip((i + 1) - 1, 0, Lx_pad - 1)
        j_aa = jnp.clip((j + 1) - 1, 0, Ly_pad - 1)
        c_succ = x_pad[i_aa]
        d_succ = y_pad[j_aa]
        contrib_M = back_step_match(succ_M_cell, succ_emit_M, c_succ, d_succ)
        # Mask: M-successor only valid when (i+1, j+1) is in-bounds.
        m_ok = (i + 1 <= Lx_pad) & (j + 1 <= Ly_pad)
        contrib_M = jnp.where(m_ok, contrib_M, NEG_INF)

        # I successor: (i, j+1) on d+1.
        i_k = i - _i_min(d + 1, Ly_pad)
        i_k_safe = jnp.clip(i_k, 0, D_max - 1)
        succ_I_full = next_diag[i_k_safe]                          # (ns, N_TAGS)
        succ_emit_I = emit[jnp.clip(i, 0, Lx_pad),
                            jnp.clip(j + 1, 0, Ly_pad)]            # (ns,)
        contrib_I_sp = log_trans + succ_emit_I[None, :]            # (ns, ns)
        contrib_I_sp = jnp.where(is_I[None, :], contrib_I_sp, NEG_INF)
        # contrib_I shape (ns, N_TAGS): sum over s' of
        #   log_trans[s, s'] + succ_emit_I[s'] + succ_I_full[s', t].
        contrib_I = jax.nn.logsumexp(
            contrib_I_sp[:, :, None] + succ_I_full[None, :, :], axis=1)
        contrib_I = jnp.where((j + 1 <= Ly_pad), contrib_I, NEG_INF)

        # D successor: (i+1, j) on d+1.
        d_k = (i + 1) - _i_min(d + 1, Ly_pad)
        d_k_safe = jnp.clip(d_k, 0, D_max - 1)
        succ_D_full = next_diag[d_k_safe]                          # (ns, N_TAGS)
        succ_emit_D = emit[jnp.clip(i + 1, 0, Lx_pad),
                            jnp.clip(j, 0, Ly_pad)]                # (ns,)
        contrib_D_sp = log_trans + succ_emit_D[None, :]
        contrib_D_sp = jnp.where(is_D[None, :], contrib_D_sp, NEG_INF)
        contrib_D = jax.nn.logsumexp(
            contrib_D_sp[:, :, None] + succ_D_full[None, :, :], axis=1)
        contrib_D = jnp.where((i + 1 <= Lx_pad), contrib_D, NEG_INF)

        cell = jnp.logaddexp(contrib_M, jnp.logaddexp(contrib_I, contrib_D))
        return cell

    def scan_fn(carry, d):
        next_diag, next_next_diag = carry                          # (D_max, ns, N_TAGS)
        ks = jnp.arange(D_max)
        i_vals = _i_min(d, Ly_pad) + ks
        j_vals = d - i_vals
        in_pad = (i_vals <= Lx_pad) & (j_vals >= 0) & (j_vals <= Ly_pad)
        is_term_cell = (i_vals == real_Lx) & (j_vals == real_Ly)

        cells = jax.vmap(
            lambda k: compute_cell_bwd(next_diag, next_next_diag, d, k))(ks)

        # Overlay terminal condition where (i, j) == (real_Lx, real_Ly).
        term_cell = jnp.broadcast_to(beta_term_st, (D_max, ns, N_TAGS))
        cells = jnp.where(is_term_cell[:, None, None], term_cell, cells)
        # Mask out-of-padding cells.
        cells = jnp.where(in_pad[:, None, None], cells, NEG_INF)
        return (cells, next_diag), cells

    # Init "next" and "next_next" for the very first scan step (d = Lx_pad + Ly_pad):
    # next is diag (Lx_pad + Ly_pad + 1) — empty. next_next is diag
    # (Lx_pad + Ly_pad + 2) — also empty.
    empty_diag = jnp.full((D_max, ns, N_TAGS), NEG_INF)
    init_carry = (empty_diag, empty_diag)

    (_, _), all_diags_bwd = jax.lax.scan(
        scan_fn, init_carry,
        jnp.arange(n_diags_total)[::-1])
    # all_diags_bwd[idx] corresponds to d = (n_diags_total - 1) - idx
    #                                     = (Lx_pad + Ly_pad) - idx.

    # Scatter into full grid.
    n_flat = (Lx_pad + 1) * (Ly_pad + 1)
    beta = jnp.full((Lx_pad + 1, Ly_pad + 1, ns, N_TAGS), NEG_INF)
    beta_flat = jnp.concatenate([
        beta.reshape(n_flat, ns, N_TAGS),
        jnp.full((1, ns, N_TAGS), NEG_INF),
    ], axis=0)

    def _scatter_one_diag(idx, bf):
        d = (n_diags_total - 1) - idx                              # Lx_pad+Ly_pad..0
        diag_data = all_diags_bwd[idx]                             # (D_max, ns, N_TAGS)
        i_min_d = jnp.maximum(0, d - Ly_pad)
        ks = jnp.arange(D_max)
        i_vals = i_min_d + ks
        j_vals = d - i_vals
        valid = (i_vals <= Lx_pad) & (j_vals >= 0) & (j_vals <= Ly_pad)
        lin = i_vals * (Ly_pad + 1) + j_vals
        lin_safe = jnp.where(valid, lin, n_flat)
        return bf.at[lin_safe].set(diag_data)

    beta_flat = jax.lax.fori_loop(
        0, n_diags_total, _scatter_one_diag, beta_flat)
    beta = beta_flat[:n_flat].reshape(Lx_pad + 1, Ly_pad + 1, ns, N_TAGS)
    return beta


# ---------------------------------------------------------------------------
# JIT-cached wrappers.
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _aug_forward_antidiag_jit(log_trans, state_types, emit, log_M_tensor,
                                x_pad, y_pad, log_eps, Lx_pad, Ly_pad):
    return _aug_forward_core_antidiag(
        log_trans, state_types, emit, log_M_tensor,
        x_pad, y_pad, log_eps, Lx_pad, Ly_pad)


@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _aug_backward_antidiag_jit(log_trans, state_types, emit, log_M_tensor,
                                 x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                                 real_Lx, real_Ly):
    return _aug_backward_core_antidiag(
        log_trans, state_types, emit, log_M_tensor,
        x_pad, y_pad, log_eps, Lx_pad, Ly_pad, real_Lx, real_Ly)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def forward_aug_phmm_antidiag(log_trans, state_types, sub_matrix, pi,
                                x_seq, y_seq, real_Lx, real_Ly,
                                log_M_tensor, log_eps,
                                log_emit_table=None):
    """Antidiagonal Forward DP wrapper. Same return shape as
    ``aug_phmm.forward_aug_phmm``.
    """
    Lx_pad = _pad_to_bin(int(x_seq.shape[0]))
    Ly_pad = _pad_to_bin(int(y_seq.shape[0]))
    if log_emit_table is None:
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
        emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi)
    else:
        ns = state_types.shape[0]
        Lx_arr = x_seq.shape[0]; Ly_arr = y_seq.shape[0]
        emit = jnp.full((Lx_pad + 1, Ly_pad + 1, ns), NEG_INF)
        emit = emit.at[:Lx_arr + 1, :Ly_arr + 1, :].set(log_emit_table)
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
    mask = _emit_mask(real_Lx, real_Ly, Lx_pad, Ly_pad,
                      state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)
    return _aug_forward_antidiag_jit(
        log_trans, state_types, emit, log_M_tensor,
        x_pad, y_pad, log_eps, Lx_pad, Ly_pad)


def backward_aug_phmm_antidiag(log_trans, state_types, sub_matrix, pi,
                                 x_seq, y_seq, real_Lx, real_Ly,
                                 log_M_tensor, log_eps,
                                 log_emit_table=None):
    """Antidiagonal Backward DP wrapper. Same return shape as
    ``aug_phmm.backward_aug_phmm``.
    """
    Lx_pad = _pad_to_bin(int(x_seq.shape[0]))
    Ly_pad = _pad_to_bin(int(y_seq.shape[0]))
    if log_emit_table is None:
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
        emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi)
    else:
        ns = state_types.shape[0]
        Lx_arr = x_seq.shape[0]; Ly_arr = y_seq.shape[0]
        emit = jnp.full((Lx_pad + 1, Ly_pad + 1, ns), NEG_INF)
        emit = emit.at[:Lx_arr + 1, :Ly_arr + 1, :].set(log_emit_table)
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
    mask = _emit_mask(real_Lx, real_Ly, Lx_pad, Ly_pad,
                      state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)
    return _aug_backward_antidiag_jit(
        log_trans, state_types, emit, log_M_tensor,
        x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
        jnp.asarray(real_Lx), jnp.asarray(real_Ly))


def aug_phmm_antidiag_corrected_posterior(
        x_seq, y_seq, t: float,
        ins_rate: float, del_rate: float, ext: float,
        Q_lg, pi_lg, boost_state,
        alpha_z: float = 100.0,
        q_min: float = 0.0) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """End-to-end Q' computation under the antidiagonal augmented PHMM.

    Drop-in replacement for
    ``aug_phmm.aug_phmm_corrected_posterior`` with identical math but
    antidiagonal-wavefront DP traversal. Returns the same 4-tuple:

        (Q_prime, L_exact, Q_baseline, log_F0)
    """
    Lx = x_seq.shape[0]; Ly = y_seq.shape[0]
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)

    log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
        ins_rate, del_rate, t, ext, Q_lg, pi_lg)

    x_pad = jnp.asarray(_pad_seq(jnp.asarray(x_seq), Lx_pad))
    y_pad = jnp.asarray(_pad_seq(jnp.asarray(y_seq), Ly_pad))

    M_tensor_np = build_M_tensor_aa_marginal(boost_state)
    log_M_tensor = jnp.log(jnp.clip(jnp.asarray(M_tensor_np), 1e-300, None))

    log_eps = float(np.log(1.0 / float(alpha_z)))
    log_eps_j = jnp.asarray(log_eps)

    emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi_out)
    mask = _emit_mask(jnp.asarray(Lx), jnp.asarray(Ly), Lx_pad, Ly_pad,
                      state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)

    alpha, alpha_left_edge, alpha_right_edge = _aug_forward_antidiag_jit(
        log_trans, state_types, emit, log_M_tensor,
        x_pad, y_pad, log_eps_j, Lx_pad, Ly_pad)
    beta = _aug_backward_antidiag_jit(
        log_trans, state_types, emit, log_M_tensor,
        x_pad, y_pad, log_eps_j, Lx_pad, Ly_pad,
        jnp.asarray(Lx), jnp.asarray(Ly))

    e_idx = _find_e_idx(state_types)
    end_alpha = alpha[Lx, Ly, :, :]                                # (ns, N_TAGS)
    end_no_edge = end_alpha[:, TAG_NO_EDGE] + log_trans[:, e_idx]
    end_done = end_alpha[:, TAG_DONE] + log_trans[:, e_idx]
    log_L_exact = jax.nn.logsumexp(jnp.concatenate([end_no_edge, end_done]))

    log_post_M = jax.nn.logsumexp(alpha[:, :, M, :] + beta[:, :, M, :],
                                       axis=-1)                    # (Lx_pad+1, Ly_pad+1)
    log_Q_prime_pad = log_post_M - log_L_exact
    log_Q_prime = log_Q_prime_pad[1:Lx + 1, 1:Ly + 1]
    Q_prime = np.asarray(jnp.exp(log_Q_prime))

    # Baseline F1/F0 — identical to aug_phmm reference.
    from .f2_scfg import compute_F0_F1
    F0_b, log_F0_b, F1_b, _, _ = compute_F0_F1(
        log_trans, state_types, sub_matrix, pi_out,
        x_pad, y_pad, jnp.asarray(Lx), jnp.asarray(Ly))
    Q_baseline = np.asarray(F1_b) / float(max(F0_b, 1e-300))
    log_F0_baseline = log_F0_b
    L_exact_val = float(jnp.exp(log_L_exact))
    log_F0_val = float(log_F0_baseline)

    _ = q_min
    return Q_prime, L_exact_val, Q_baseline, log_F0_val
