"""F2-SCFG: exact 0-or-1-edge Pair-SCFG marginal posteriors for TKF-DP.

Implements the third pathway of the TKF-DP postprocessing trade-off
(see appendix "Exact 0-or-1-edge marginal posteriors via Pair-SCFG
inside-outside" of ``main.tex``):

    Q'_{ij} = [ F1(i, j)
                + eps * sum_{(k, l), k != i} F2(i, j; k, l) * M(i, j; k, l; t) ]
              / L_exact

where

    L_exact = F0
              + eps * sum_{(i, j; k, l), i < k}
                   F2(i, j; k, l) * M(i, j; k, l; t)

and eps = 1 / alpha_z is the per-pair partner prior under the
size-{1, 2} Ewens partition with concentration alpha_z.

The three pair-HMM marginal moments F0, F1, F2 are defined in eq:F0,
eq:F1, eq:F2 of main.tex; M is the four-residue Potts coupling boost
of eq:M-marginal (already implemented in ``src/tkfdp/postprocessing.py``
and exposed per (sequence-pair) via ``src/tkfdp/coupled_annealing.py``).

The implementation is GPU-friendly: padded-and-cached so JIT shapes are
stable across heterogeneous-length sequence pairs, batched/chunked over
the anchor axis to keep the F2 memory footprint bounded by the chunk
size, and explicitly masks padded positions out of every partition sum.

Public API:

  forward_pair_hmm(log_trans, state_types, sub_matrix, pi,
                   x_seq, y_seq, real_Lx, real_Ly)
      -> log_alpha (Lx_pad+1, Ly_pad+1, 5)
      Forward DP returning the alpha tensor in log space. Entries for
      padded (i > real_Lx) or (j > real_Ly) positions are NEG_INF.
      Convention matches the main-text spec: alpha[i, j, k] is the total
      log-prob of paths reaching state k after consuming exactly i
      residues of X and j of Y, INCLUDING all emissions up to and
      including the emission at state k itself.

  backward_pair_hmm(...)
      -> log_beta (Lx_pad+1, Ly_pad+1, 5)
      Backward DP returning beta in log space. beta[i, j, k] is the
      total log-prob of paths starting from state k at (i, j), ending at
      E, INCLUDING all emissions AFTER state k but EXCLUDING the
      emission at state k itself. Terminal: beta[real_Lx, real_Ly, k]
      = log_trans[k, E].

  scfg_corrected_posterior(x_seq, y_seq, ins_rate, del_rate, ext,
                           Q_lg, pi_lg, boost_state, alpha_z=100.0,
                           q_min=0.0, chunk_size=8)
      -> (Q_prime (Lx, Ly), L_exact (scalar))
      The end-to-end pipeline: build the TKF92 Pair HMM, compute alpha
      and beta, then chunk over anchors (i_a, j_a) computing the
      restart-Forward mu and accumulating the F2 contributions to both
      the per-(i_a, j_a) numerator of Q' and the global L_exact
      denominator.

Conventions:
  - alpha and beta are returned in log space; F1, F2 contributions are
    converted to linear space before accumulation (the partition sums
    are over real numbers, not log space).
  - We use (i, j) for the row anchor (i.e., the column-pair where the
    coupled-edge is anchored when summed over candidate partners
    (k, l)) and (k, l) for the partner column-pair.
  - The L_exact "i < k" ordered restriction comes straight from
    eq:exact-01; we enforce it by computing F2 only for k >= i (and
    inside the i_a == i case, only for l > j_a; equivalently, only for
    (k, l) > (i_a, j_a) in lex order).
  - Q' uses an UNORDERED partner sum over k != i (eq:exact-Q), so in
    the per-(i, j) numerator we sum over all partner (k, l) with the
    column index k != i and any l (equivalently: every alignment that
    pairs (i, j) AND (k, l) under the coupled Potts kernel). To avoid
    double-counting we exploit the lex-order F2 storage and reflect
    it back in the numerator (see _accumulate_chunk).
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

A = 20  # amino acid alphabet


# ---------------------------------------------------------------------------
# Forward and Backward returning alpha and beta directly.
# ---------------------------------------------------------------------------


def _forward_alpha_core(log_trans, state_types, emit, Lx_pad, Ly_pad):
    """Forward DP returning the full alpha tensor (log space).

    Same recursion as tkfmixdom's _forward_2d_core but the output is the
    forward table itself; total log-prob is recovered downstream from
    alpha[real_Lx, real_Ly, :].

    All shapes are static at JIT trace time (Lx_pad, Ly_pad).
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    # Row 0: F[0, 0, S] = 0, F[0, j, I-type] = ... for j >= 1.
    row0 = jnp.full((Ly_pad + 1, ns), NEG_INF)
    row0 = row0.at[0, S].set(0.0)

    def row0_step(prev_cell, j):
        raw = jax.nn.logsumexp(prev_cell[:, None] + log_trans, axis=0) \
            + emit[0, j]
        cell = jnp.where(is_I, raw, NEG_INF)
        return cell, cell

    _, row0_rest = jax.lax.scan(row0_step, row0[0],
                                jnp.arange(1, Ly_pad + 1))
    row0 = jnp.concatenate([row0[0:1], row0_rest], axis=0)

    def row_step(prev_row, i):
        # Column 0 (j=0): only D-type predecessor (i-1, 0).
        raw0 = jax.nn.logsumexp(prev_row[0][:, None] + log_trans, axis=0) \
            + emit[i, 0]
        cell0 = jnp.where(is_D, raw0, NEG_INF)

        def col_step(prev_cell, j):
            m_val = jax.nn.logsumexp(
                prev_row[j - 1][:, None] + log_trans, axis=0) + emit[i, j]
            i_val = jax.nn.logsumexp(
                prev_cell[:, None] + log_trans, axis=0) + emit[i, j]
            d_val = jax.nn.logsumexp(
                prev_row[j][:, None] + log_trans, axis=0) + emit[i, j]
            cell = jnp.where(
                is_M, m_val,
                jnp.where(is_I, i_val,
                          jnp.where(is_D, d_val, NEG_INF)))
            return cell, cell

        _, row_rest = jax.lax.scan(col_step, cell0,
                                   jnp.arange(1, Ly_pad + 1))
        curr_row = jnp.concatenate([cell0[None], row_rest], axis=0)
        return curr_row, curr_row

    _, all_rows = jax.lax.scan(row_step, row0,
                               jnp.arange(1, Lx_pad + 1))
    F = jnp.concatenate([row0[None], all_rows], axis=0)
    return F


def _backward_beta_core(log_trans, state_types, emit, Lx_pad, Ly_pad,
                        real_Lx, real_Ly):
    """Backward DP returning the full beta tensor (log space).

    Terminal condition is placed at (real_Lx, real_Ly): beta[real_Lx,
    real_Ly, k] = log_trans[k, E_idx]. Beta cells at (i > real_Lx) or
    (j > real_Ly) are NEG_INF.

    Same recursion as tkfmixdom's _backward_2d_core, just with the
    terminal placed at the (possibly traced) real lengths and extra
    masking for padded positions.
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    e_idx = _find_e_idx(state_types)
    beta_term = log_trans[:, e_idx]                        # (ns,)

    # Row Lx_pad: backward DP runs from (Lx_pad, Ly_pad) DOWN to (0, 0).
    # We initialize the backward grid with NEG_INF and place the terminal
    # at (real_Lx, real_Ly). Beyond (real_Lx, real_Ly) all cells stay
    # NEG_INF. Recursion is then a column- and row-scan.

    # Pre-mask emissions BEYOND (real_Lx, real_Ly): the input emit
    # already has NEG_INF at padded positions, so the recursion will
    # not propagate any positive probability through padding. Defensive
    # only.

    # Build an (Lx_pad+1, Ly_pad+1, ns) mask that's True ONLY at
    # (real_Lx, real_Ly).
    i_eq = jnp.arange(Lx_pad + 1) == real_Lx               # (Lx_pad+1,)
    j_eq = jnp.arange(Ly_pad + 1) == real_Ly               # (Ly_pad+1,)
    is_term = i_eq[:, None] & j_eq[None, :]                # (Lx_pad+1, Ly_pad+1)

    def row_step_bwd(next_row, i):
        # j = Ly_pad: only D-type successor at (i+1, Ly_pad). Use safe
        # index for the i+1 lookup (out-of-bounds reads NEG_INF).
        i_succ = jnp.minimum(i + 1, Lx_pad)
        succ_emit_D = emit[i_succ, Ly_pad]
        contrib_D = log_trans + succ_emit_D[None, :] \
            + next_row[Ly_pad][None, :]
        contrib_D = jnp.where(is_D[None, :], contrib_D, NEG_INF)
        cell_Ly_pad = jax.nn.logsumexp(contrib_D, axis=1)
        # Override with terminal if (i, Ly_pad) == (real_Lx, real_Ly).
        cell_Ly_pad = jnp.where(is_term[i, Ly_pad],
                                beta_term, cell_Ly_pad)

        def col_step_bwd(beta_right, j):
            j_succ = jnp.minimum(j + 1, Ly_pad)
            i_succ_local = jnp.minimum(i + 1, Lx_pad)
            # M successor at (i+1, j+1)
            succ_emit_M = emit[i_succ_local, j_succ]
            contrib_M = log_trans + succ_emit_M[None, :] \
                + next_row[j_succ][None, :]
            contrib_M = jnp.where(is_M[None, :], contrib_M, NEG_INF)
            # I successor at (i, j+1)
            succ_emit_I = emit[i, j_succ]
            contrib_I = log_trans + succ_emit_I[None, :] \
                + beta_right[None, :]
            contrib_I = jnp.where(is_I[None, :], contrib_I, NEG_INF)
            # D successor at (i+1, j)
            succ_emit_D = emit[i_succ_local, j]
            contrib_D = log_trans + succ_emit_D[None, :] \
                + next_row[j][None, :]
            contrib_D = jnp.where(is_D[None, :], contrib_D, NEG_INF)
            all_contrib = jnp.logaddexp(contrib_M,
                                        jnp.logaddexp(contrib_I, contrib_D))
            cell = jax.nn.logsumexp(all_contrib, axis=1)
            cell = jnp.where(is_term[i, j], beta_term, cell)
            return cell, cell

        _, row_rest = jax.lax.scan(
            col_step_bwd, cell_Ly_pad,
            jnp.arange(Ly_pad - 1, -1, -1))
        curr_row = jnp.concatenate([row_rest[::-1],
                                    cell_Ly_pad[None]], axis=0)
        return curr_row, curr_row

    # Initialize last row buffer (i = Lx_pad). Only "stay" at I (since
    # there's no row beyond) -- actually for the recursion's first
    # invocation we need a "next_row" at i = Lx_pad+1 (which doesn't
    # exist). Trick: start the scan at i = Lx_pad with a NEG_INF buffer
    # and force the terminal value via is_term inside col_step_bwd.
    init_next_row = jnp.full((Ly_pad + 1, ns), NEG_INF)
    # Place the terminal in init_next_row at the column real_Ly IF
    # i = real_Lx + 1 (handled by row_step_bwd). For our purposes the
    # init_next_row corresponds to i=Lx_pad; the recursion will produce
    # the correct values via terminal-overrides at (real_Lx, real_Ly).

    # We scan i = Lx_pad, Lx_pad-1, ..., 0. After the scan, all rows are
    # available. The "last_row" (i = Lx_pad) is the first scan output.
    _, all_rows_bwd = jax.lax.scan(
        row_step_bwd, init_next_row,
        jnp.arange(Lx_pad, -1, -1))
    # all_rows_bwd[0] corresponds to i = Lx_pad, ..., all_rows_bwd[-1] to i=0.
    # Reverse to get (Lx_pad+1, Ly_pad+1, ns) in i-ascending order.
    B = all_rows_bwd[::-1]
    return B


# We expose JIT-cached wrappers keyed on the static Lx_pad / Ly_pad.

@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _forward_alpha_jit(log_trans, state_types, emit, Lx_pad, Ly_pad):
    return _forward_alpha_core(log_trans, state_types, emit,
                               Lx_pad, Ly_pad)


@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _backward_beta_jit(log_trans, state_types, emit, Lx_pad, Ly_pad,
                       real_Lx, real_Ly):
    return _backward_beta_core(log_trans, state_types, emit,
                               Lx_pad, Ly_pad, real_Lx, real_Ly)


def forward_pair_hmm(log_trans, state_types, sub_matrix, pi,
                     x_seq, y_seq, real_Lx, real_Ly,
                     log_emit_table=None):
    """Forward DP wrapper. Pads to geometric bins; returns log-alpha.

    See module docstring for the alpha convention.
    """
    Lx_pad = _pad_to_bin(int(x_seq.shape[0]))
    Ly_pad = _pad_to_bin(int(y_seq.shape[0]))
    if log_emit_table is None:
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
        emit = pair_hmm_emissions(state_types, x_pad, y_pad,
                                  sub_matrix, pi)
    else:
        ns = state_types.shape[0]
        Lx_arr = x_seq.shape[0]; Ly_arr = y_seq.shape[0]
        emit = jnp.full((Lx_pad + 1, Ly_pad + 1, ns), NEG_INF)
        emit = emit.at[:Lx_arr + 1, :Ly_arr + 1, :].set(log_emit_table)
    mask = _emit_mask(real_Lx, real_Ly, Lx_pad, Ly_pad,
                      state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)
    return _forward_alpha_jit(log_trans, state_types, emit,
                              Lx_pad, Ly_pad)


def backward_pair_hmm(log_trans, state_types, sub_matrix, pi,
                      x_seq, y_seq, real_Lx, real_Ly,
                      log_emit_table=None):
    """Backward DP wrapper. Pads to geometric bins; returns log-beta.

    See module docstring for the beta convention.
    """
    Lx_pad = _pad_to_bin(int(x_seq.shape[0]))
    Ly_pad = _pad_to_bin(int(y_seq.shape[0]))
    if log_emit_table is None:
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
        emit = pair_hmm_emissions(state_types, x_pad, y_pad,
                                  sub_matrix, pi)
    else:
        ns = state_types.shape[0]
        Lx_arr = x_seq.shape[0]; Ly_arr = y_seq.shape[0]
        emit = jnp.full((Lx_pad + 1, Ly_pad + 1, ns), NEG_INF)
        emit = emit.at[:Lx_arr + 1, :Ly_arr + 1, :].set(log_emit_table)
    mask = _emit_mask(real_Lx, real_Ly, Lx_pad, Ly_pad,
                      state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)
    return _backward_beta_jit(log_trans, state_types, emit,
                              Lx_pad, Ly_pad,
                              jnp.asarray(real_Lx), jnp.asarray(real_Ly))


# ---------------------------------------------------------------------------
# Restart-Forward: mu[(i_a, j_a) -> (k, l), state] in log space.
# ---------------------------------------------------------------------------


def _restart_forward_core(log_trans, state_types, emit, Lx_pad, Ly_pad,
                          i_a, j_a):
    """Restart Forward starting from state M at (i_a, j_a).

    Initial condition: mu[i_a, j_a, M] = 0 (no emission factor, since
    the M-emission at (i_a, j_a) is already counted in the outer
    alpha[i_a, j_a, M]). All other initial cells are NEG_INF.

    Forward recursion is identical to the standard Forward; the result
    mu[k, l, M] is in log space and represents the partial probability
    of an alignment fragment from (i_a, j_a) [no emission there] to
    (k, l) [WITH the M-emission at (k, l)].

    For (k < i_a) or (l < j_a) the cell is unreachable and remains
    NEG_INF. For (k = i_a, l = j_a, k_state = M) the cell is initialized
    to 0; for any other state at (i_a, j_a) it's NEG_INF.

    All shapes are static (Lx_pad, Ly_pad) at JIT trace time. The
    anchor coordinates i_a, j_a are TRACED jnp scalars so we can vmap.
    """
    ns = log_trans.shape[0]
    is_M_st = (state_types == M)
    is_I_st = (state_types == I)
    is_D_st = (state_types == D)

    # Build the initial cell-at-(i_a, j_a) by placing 0.0 at index M
    # (and NEG_INF elsewhere). Use a mask for indices.
    init_cell = jnp.where(is_M_st, 0.0, NEG_INF)

    # Now run a full Forward over all (i, j) but with the start condition
    # placed at (i_a, j_a) instead of (0, 0). Cells with i < i_a or
    # j < j_a are NEG_INF; the cell at (i_a, j_a) is init_cell; other
    # cells are computed by the standard recursion.
    #
    # We handle this by running the standard recursion but explicitly
    # overriding cells at (i_a, j_a) to init_cell and zeroing cells
    # outside the [i_a..Lx_pad] x [j_a..Ly_pad] reachable region.

    is_anchor_i = jnp.arange(Lx_pad + 1) == i_a            # (Lx_pad+1,)
    is_anchor_j = jnp.arange(Ly_pad + 1) == j_a            # (Ly_pad+1,)
    in_region_i = jnp.arange(Lx_pad + 1) >= i_a            # (Lx_pad+1,)
    in_region_j = jnp.arange(Ly_pad + 1) >= j_a            # (Ly_pad+1,)

    # Row 0: only meaningful if i_a == 0. Handle inside the row scans
    # with the override.

    def row0_step(prev_cell, j):
        raw = jax.nn.logsumexp(prev_cell[:, None] + log_trans, axis=0) \
            + emit[0, j]
        cell = jnp.where(is_I_st, raw, NEG_INF)
        # If this is the anchor cell (and the anchor sits in row 0),
        # override with init_cell.
        is_anchor = is_anchor_i[0] & is_anchor_j[j]
        cell = jnp.where(is_anchor, init_cell, cell)
        return cell, cell

    row0 = jnp.full((Ly_pad + 1, ns), NEG_INF)
    # Force start-of-row-0 cell:
    is_anchor_00 = is_anchor_i[0] & is_anchor_j[0]
    row0 = row0.at[0].set(jnp.where(is_anchor_00, init_cell, row0[0]))
    _, row0_rest = jax.lax.scan(row0_step, row0[0],
                                jnp.arange(1, Ly_pad + 1))
    row0 = jnp.concatenate([row0[0:1], row0_rest], axis=0)

    def row_step(prev_row, i):
        # Column 0 (j=0): D-type successor only.
        raw0 = jax.nn.logsumexp(prev_row[0][:, None] + log_trans, axis=0) \
            + emit[i, 0]
        cell0 = jnp.where(is_D_st, raw0, NEG_INF)
        is_anchor = is_anchor_i[i] & is_anchor_j[0]
        cell0 = jnp.where(is_anchor, init_cell, cell0)

        def col_step(prev_cell, j):
            m_val = jax.nn.logsumexp(
                prev_row[j - 1][:, None] + log_trans, axis=0) + emit[i, j]
            i_val = jax.nn.logsumexp(
                prev_cell[:, None] + log_trans, axis=0) + emit[i, j]
            d_val = jax.nn.logsumexp(
                prev_row[j][:, None] + log_trans, axis=0) + emit[i, j]
            cell = jnp.where(
                is_M_st, m_val,
                jnp.where(is_I_st, i_val,
                          jnp.where(is_D_st, d_val, NEG_INF)))
            is_anchor = is_anchor_i[i] & is_anchor_j[j]
            cell = jnp.where(is_anchor, init_cell, cell)
            return cell, cell

        _, row_rest = jax.lax.scan(col_step, cell0,
                                   jnp.arange(1, Ly_pad + 1))
        curr_row = jnp.concatenate([cell0[None], row_rest], axis=0)
        return curr_row, curr_row

    _, all_rows = jax.lax.scan(row_step, row0,
                               jnp.arange(1, Lx_pad + 1))
    mu = jnp.concatenate([row0[None], all_rows], axis=0)
    # Mask cells outside the (i, j) >= (i_a, j_a) reachable region.
    in_region = in_region_i[:, None] & in_region_j[None, :]
    mu = jnp.where(in_region[:, :, None], mu, NEG_INF)
    return mu


@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _restart_forward_jit(log_trans, state_types, emit, Lx_pad, Ly_pad,
                         i_a, j_a):
    return _restart_forward_core(log_trans, state_types, emit,
                                 Lx_pad, Ly_pad, i_a, j_a)


# ---------------------------------------------------------------------------
# Per-anchor F2 kernel + chunk accumulation.
# ---------------------------------------------------------------------------


def _build_log_M_field_jax(boost_state_jax, i_a, j_a, Lx_pad, Ly_pad):
    """log M(i_a, j_a; k, l) over all (k, l) in (Lx_pad, Ly_pad), JAX.

    boost_state_jax is a tuple (gamma, denom, joint_per_cp, x_pad, y_pad)
    of jnp arrays of static shapes:
      gamma: (Lx_pad, Ly_pad, K_c)  -- 0 outside real region
      denom: (Lx_pad, Ly_pad)        -- 1.0 outside real region (so
                                         log(denom) = 0; values there
                                         are masked downstream anyway)
      joint_per_cp: (K_c, K_c, A, A, A, A)
      x_pad: (Lx_pad,) int  -- padded amino-acid indices, 0 outside
      y_pad: (Ly_pad,) int

    Returns a (Lx_pad, Ly_pad) log-M field. Note we use (Lx_pad,
    Ly_pad) not (Lx_pad+1, Ly_pad+1): the M field is indexed by
    *residue* coordinates (1..Lx, 1..Ly in 1-based) while alpha/beta
    are indexed by *position* (0..Lx, 0..Ly). We use 0-based residue
    indices here, consistent with x_pad/y_pad.

    M values at padded positions are masked (set to log(1)=0) so the
    F2 accumulator can ignore them via a separate mask without leaking
    log-arithmetic NaNs.
    """
    gamma, denom, joint_per_cp, x_pad, y_pad = boost_state_jax
    K_c = gamma.shape[-1]
    g_a = gamma[i_a, j_a]                                  # (K_c,)
    Xi_a = x_pad[i_a]                                       # scalar
    Yj_a = y_pad[j_a]                                       # scalar
    # T[c', a', b'] = sum_c g_a[c] * J[c, c', Xi_a, a', Yj_a, b']
    J_slice = joint_per_cp[:, :, Xi_a, :, Yj_a, :]         # (K_c, K_c, A, A)
    T = jnp.einsum('c,cdpb->dpb', g_a, J_slice)            # (K_c, A, A)
    # numer[k, l] = sum_c' gamma[k, l, c'] * T[c', x_pad[k], y_pad[l]]
    Xb = x_pad                                              # (Lx_pad,)
    Yb = y_pad                                              # (Ly_pad,)
    # T_at[k, l, c'] = T[c', x_pad[k], y_pad[l]]
    T_at = T[:, Xb, :][:, :, Yb]                           # (K_c, Lx_pad, Ly_pad)
    T_at = jnp.transpose(T_at, (1, 2, 0))                  # (Lx_pad, Ly_pad, K_c)
    numer = jnp.sum(gamma * T_at, axis=-1)                 # (Lx_pad, Ly_pad)
    denom_field = denom[i_a, j_a] * denom                  # (Lx_pad, Ly_pad)
    safe_num = jnp.clip(numer, 1e-300, None)
    safe_den = jnp.clip(denom_field, 1e-300, None)
    return jnp.log(safe_num) - jnp.log(safe_den)           # (Lx_pad, Ly_pad)


def _per_anchor_kernel(log_trans, state_types, emit, Lx_pad, Ly_pad,
                        log_alpha, log_beta, log_M_at_anchor,
                        boost_state_jax, real_Lx, real_Ly, i_a, j_a):
    """Compute the F2 contributions for ONE anchor (i_a, j_a).

    Returns:
      anchor_numer: scalar -- the contribution to the numerator of
        Q'_{i_a, j_a} from F2-with-i_a-as-the-fixed-(i, j) edge:
          eps * sum_{(k, l), k != i_a} F2(i_a, j_a; k, l) * M.
        Note: this implements the eq:exact-Q convention (over UNORDERED
        partner column with k != i_a; both k > i_a and k < i_a allowed
        but k == i_a forbidden).
      partial_numer_field: (Lx_pad, Ly_pad) -- the contribution from
        this anchor to OTHER (k, l) numerators, where (i_a, j_a) is the
        partner column. Specifically partial_numer_field[k, l] gets
        F2(i_a, j_a; k, l) * M(i_a, j_a; k, l) for k != i_a (so the
        outer accumulation builds the symmetric sum).
        Implemented only for k >= i_a (i.e., (i_a, j_a) is the
        smaller-index anchor in the pair); the caller's main loop
        guarantees we hit each unordered pair exactly twice (once with
        each endpoint as anchor) by NOT pre-restricting to k >= i_a.
        Actually we DO restrict to k > i_a OR (k == i_a AND l > j_a)
        for storage efficiency, then we add the contribution to BOTH
        (i_a, j_a)'s numerator AND (k, l)'s numerator. See below.
      l_exact_contribution: scalar -- contribution to L_exact:
          eps * sum_{(k, l), (k, l) > (i_a, j_a) lex AND k > i_a}
                F2(i_a, j_a; k, l) * M.
        Note: per main.tex eq:exact-01, the L_exact sum is over
        (i, j; k, l) with i < k. Since we anchor on (i_a, j_a) and
        partner (k, l), the "i < k" condition becomes "k > i_a".

    Convention recap:
      log_alpha[i, j, M]: includes M-emission at (i, j).
      log_beta[i, j, M]: excludes M-emission at (i, j).
      mu[i_a, j_a -> k, l, M]: starts at (i_a, j_a) with no emission
        added; ends at (k, l) WITH the M-emission at (k, l).
      So F2(i_a, j_a; k, l) = exp(log_alpha[i_a, j_a, M] +
            mu[k, l, M] + log_beta[k, l, M]).
    """
    # Restart Forward from M at (i_a, j_a).
    mu = _restart_forward_core(log_trans, state_types, emit,
                               Lx_pad, Ly_pad, i_a, j_a)
    # mu[i, j, k] for state k. We only care about M-state at (k, l).
    # Note mu is indexed at *position* (k, l) i.e. (Lx_pad+1, Ly_pad+1);
    # match-state entries at (k, l) for k >= 1, l >= 1 are the relevant
    # log-prob factors. We slice [1:, 1:, M] to get a (Lx_pad, Ly_pad)
    # array indexed by 0-based residue coordinates.
    mu_M = mu[1:, 1:, M]                                    # (Lx_pad, Ly_pad)

    # log_alpha and log_beta likewise are at *position* coords; the
    # M-cell at (i, j) in 1-based residue coordinates is at position
    # (i, j) in alpha/beta. So log_alpha[1:, 1:, M] is the (Lx_pad,
    # Ly_pad) match-cell array indexed by (i-1, j-1) for residue (i, j).
    log_alpha_M = log_alpha[1:, 1:, M]                      # (Lx_pad, Ly_pad)
    log_beta_M = log_beta[1:, 1:, M]                        # (Lx_pad, Ly_pad)

    # The anchor scalar log_alpha[i_a, j_a, M] (using 1-based residue
    # coordinates: i_a = position i, so (i_a-1, j_a-1) in our
    # _M arrays). But we work in *position* coords for i_a/j_a inside
    # _restart_forward_core, so i_a/j_a here refer to positions
    # 0..Lx_pad / 0..Ly_pad. Match cell at position (i_a, j_a) lives at
    # (i_a, j_a, M) -- which is at our log_alpha_M[i_a-1, j_a-1] when
    # i_a >= 1, j_a >= 1.
    # We will only ever invoke this kernel with i_a >= 1, j_a >= 1
    # (anchors are residues, not the silent-start position).
    anchor_log_alpha = log_alpha[i_a, j_a, M]               # scalar
    anchor_log_beta = log_beta[i_a, j_a, M]                 # scalar (unused)

    # Build the F2 log-table over the partner residue (k, l).
    # F2[k, l] (linear) for k, l 1..Lx, 1..Ly.
    log_F2 = anchor_log_alpha + mu_M + log_beta_M           # (Lx_pad, Ly_pad)

    # Multiply by M(i_a, j_a; k, l).  M field is also (Lx_pad, Ly_pad).
    log_F2M = log_F2 + log_M_at_anchor                     # (Lx_pad, Ly_pad)

    # Mask: real positions only. Residue index k in 1..real_Lx is k_idx
    # in 0..real_Lx-1 in the (Lx_pad, Ly_pad) array.
    k_idx = jnp.arange(Lx_pad)
    l_idx = jnp.arange(Ly_pad)
    real_mask = ((k_idx[:, None] < real_Lx)
                 & (l_idx[None, :] < real_Ly))            # (Lx_pad, Ly_pad)
    # Exclude the diagonal cell (k = i_a, l = j_a) entirely from F2:
    # F2(i, j; i, j) is degenerate (the same column twice).
    # i_a (position) corresponds to residue index i_a - 1 in (Lx_pad,
    # Ly_pad) arrays.
    self_mask = ~((k_idx[:, None] == (i_a - 1))
                  & (l_idx[None, :] == (j_a - 1)))
    base_mask = real_mask & self_mask                      # (Lx_pad, Ly_pad)

    # --- Q'_{i_a, j_a} numerator contribution ---
    # eps factor is applied OUTSIDE this kernel.  We return the raw
    # sum_{(k, l): k != i_a} F2 * M.
    # k_pos_mask: residue k != i_a means k != i_a - 0; in our (Lx_pad,
    # Ly_pad) array this is k_idx != i_a - 1.
    k_diff_mask = k_idx != (i_a - 1)                       # (Lx_pad,)
    numer_mask = base_mask & k_diff_mask[:, None]
    F2M = jnp.where(numer_mask, jnp.exp(log_F2M), 0.0)     # (Lx_pad, Ly_pad)
    anchor_numer = jnp.sum(F2M)                            # scalar

    # --- The "partial_numer_field" gets returned so the caller can
    # add F2(i_a, j_a; k, l) * M to OTHER (k, l) numerators. We return
    # the same F2M masked to k != i_a (positions where this F2 entry
    # is a valid SCFG contribution under k != i_a). Since the caller
    # walks every anchor (i_a, j_a) in turn, and the unordered-partner
    # convention for Q' demands sum over (k, l) with k != i_a, the
    # "partner-side numerator" for output Q'[k, l] is similarly built
    # by aggregating F2(i_anchor, j_anchor; k, l) * M from all
    # anchors with i_anchor != k. We cover this by accumulating
    # partial_numer_field across anchors and indexing by partner (k, l).
    partner_numer_field = F2M                              # (Lx_pad, Ly_pad)

    # --- L_exact contribution. Here the constraint is i < k (ordered
    # in the i axis). So in our (Lx_pad, Ly_pad) array, k_idx > i_a - 1
    # i.e. k_idx >= i_a.
    k_gt_mask = k_idx > (i_a - 1)                          # (Lx_pad,)
    l_exact_mask = base_mask & k_gt_mask[:, None]
    F2M_lex = jnp.where(l_exact_mask, jnp.exp(log_F2M), 0.0)
    l_exact_contrib = jnp.sum(F2M_lex)                     # scalar

    return anchor_numer, partner_numer_field, l_exact_contrib


@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _process_anchor_chunk(log_trans, state_types, emit,
                          log_alpha, log_beta, boost_state_jax,
                          real_Lx, real_Ly, anchor_is, anchor_js,
                          Lx_pad, Ly_pad):
    """vmap'd per-anchor F2 kernel over a chunk of anchor positions.

    Args:
      anchor_is, anchor_js: (chunk,) int arrays of POSITION coordinates
        in 1..Lx_pad / 1..Ly_pad (not residue indices). Anchors with
        i_a > real_Lx or j_a > real_Ly are valid placeholders; the
        kernel masks their contributions to zero via real_mask.
      ...

    Returns:
      anchor_numers: (chunk,) per-anchor numerator contribution.
      partner_numer_fields: (chunk, Lx_pad, Ly_pad) per-anchor
        partner-numer fields.
      l_exact_contribs: (chunk,) per-anchor L_exact contributions.
    """
    def kernel_for_one(i_a, j_a):
        log_M = _build_log_M_field_jax(boost_state_jax, i_a - 1, j_a - 1,
                                       Lx_pad, Ly_pad)
        return _per_anchor_kernel(log_trans, state_types, emit,
                                  Lx_pad, Ly_pad,
                                  log_alpha, log_beta, log_M,
                                  boost_state_jax, real_Lx, real_Ly,
                                  i_a, j_a)
    return jax.vmap(kernel_for_one)(anchor_is, anchor_js)


# ---------------------------------------------------------------------------
# Boost-state precomputation (delegates to coupled_annealing.build_boost_state
# but produces JAX-friendly padded tensors keyed by (Lx_pad, Ly_pad)).
# ---------------------------------------------------------------------------


def _pad_boost_state_to_jax(boost_state_one, Lx_pad, Ly_pad,
                            x_seq, y_seq):
    """Convert a PairBoostState (np arrays) to padded JAX tensors.

    Pads gamma, denom, x_seq, y_seq to (Lx_pad, Ly_pad) shape so the
    JIT-cached F2 kernel sees stable shapes. Padded gamma is set to a
    uniform-class distribution; padded denom is set to 1.0 so log(denom)
    contributes 0; padded x/y residues are set to 0 so the joint-emit
    lookup hits a defined cell. Padded positions are EXCLUDED downstream
    by the real_mask in _per_anchor_kernel.
    """
    Lx = x_seq.shape[0]; Ly = y_seq.shape[0]
    K_c = boost_state_one.gamma.shape[-1]

    gamma_pad = np.zeros((Lx_pad, Ly_pad, K_c), dtype=np.float64)
    gamma_pad[:Lx, :Ly, :] = boost_state_one.gamma
    # uniform distribution at padded cells (irrelevant but well-defined)
    if Lx < Lx_pad or Ly < Ly_pad:
        gamma_pad[Lx:, :, :] = 1.0 / K_c
        gamma_pad[:Lx, Ly:, :] = 1.0 / K_c

    denom_pad = np.ones((Lx_pad, Ly_pad), dtype=np.float64)
    denom_pad[:Lx, :Ly] = boost_state_one.denom

    x_pad = np.zeros((Lx_pad,), dtype=np.int32)
    x_pad[:Lx] = boost_state_one.x_seq
    y_pad = np.zeros((Ly_pad,), dtype=np.int32)
    y_pad[:Ly] = boost_state_one.y_seq

    return (
        jnp.asarray(gamma_pad),
        jnp.asarray(denom_pad),
        jnp.asarray(boost_state_one.joint_per_cp),
        jnp.asarray(x_pad),
        jnp.asarray(y_pad),
    )


# ---------------------------------------------------------------------------
# Top-level: scfg_corrected_posterior.
# ---------------------------------------------------------------------------


def compute_F0_F1(log_trans, state_types, sub_matrix, pi,
                  x_seq, y_seq, real_Lx, real_Ly):
    """Compute F0 (scalar partition fn) and F1[i, j] (linear-space match
    posterior NUMERATOR, F1 = alpha * beta at M).

    Returns:
      F0: scalar.
      log_F0: scalar (log F0).
      F1: (real_Lx, real_Ly) match-state per-residue F1 numerator.
      log_alpha, log_beta: padded log-tensors.
    """
    log_alpha = forward_pair_hmm(log_trans, state_types, sub_matrix, pi,
                                 x_seq, y_seq, real_Lx, real_Ly)
    log_beta = backward_pair_hmm(log_trans, state_types, sub_matrix, pi,
                                 x_seq, y_seq, real_Lx, real_Ly)

    e_idx = _find_e_idx(state_types)
    log_F0 = jax.nn.logsumexp(log_alpha[real_Lx, real_Ly, :]
                              + log_trans[:, e_idx])

    # F1[i, j] = exp(log_alpha[i, j, M] + log_beta[i, j, M]); the
    # M-emission is in alpha and the post-M alignment continuation is
    # in beta. Slice to real shape.
    log_F1_pad = log_alpha[1:real_Lx + 1, 1:real_Ly + 1, M] \
        + log_beta[1:real_Lx + 1, 1:real_Ly + 1, M]

    return jnp.exp(log_F0), log_F0, jnp.exp(log_F1_pad), log_alpha, log_beta


def scfg_corrected_posterior(
        x_seq, y_seq, t: float,
        ins_rate: float, del_rate: float, ext: float,
        Q_lg, pi_lg, boost_state,
        alpha_z: float = 100.0, q_min: float = 0.0,
        chunk_size: int = 8) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """End-to-end Q' computation under the F2-SCFG.

    The boost_state must have been built at the same per-pair branch
    length t (see ``coupled_annealing.build_boost_state``).

    Args:
      x_seq, y_seq: (Lx,), (Ly,) int residue arrays. Wildcards at index
        20 are clamped to 19 inside the boost lookup (matching the
        coupled-annealing convention) but kept as-is for the Pair HMM
        emissions (which know how to handle the wildcard via
        pair_hmm_emissions's 21-padded log-tables).
      t: branch length for this pair.
      ins_rate, del_rate, ext: TKF92 parameters.
      Q_lg, pi_lg: substitution generator + stationary (typically
        rate_matrix_lg() output).
      boost_state: PairBoostState (from coupled_annealing.build_boost_state)
        for THIS sequence pair.
      alpha_z: Ewens partition concentration. eps = 1 / alpha_z.
      q_min: anchor pruning threshold. Anchors (i_a, j_a) with baseline
        match posterior Q[i_a, j_a] = F1[i_a, j_a] / F0 < q_min are
        SKIPPED for the F2 inner sum (they contribute negligibly to
        L_exact). Default 0.0 (no pruning, full anchor sweep).
      chunk_size: process this many anchors at a time. Bounds the
        per-chunk F2 memory at chunk_size * Lx_pad * Ly_pad floats.

    Returns:
      Q_prime: (Lx, Ly) corrected match posterior (numpy).
      L_exact: scalar (float) partition function.
      Q_baseline: (Lx, Ly) F1/F0 baseline (numpy).
      log_F0: scalar (float).
    """
    Lx = x_seq.shape[0]; Ly = y_seq.shape[0]
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)

    # Build TKF92 Pair HMM at the matched branch length.
    log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
        ins_rate, del_rate, t, ext, Q_lg, pi_lg)

    x_pad = jnp.asarray(_pad_seq(jnp.asarray(x_seq), Lx_pad))
    y_pad = jnp.asarray(_pad_seq(jnp.asarray(y_seq), Ly_pad))

    # Compute F0 and F1.
    F0, log_F0, F1, log_alpha, log_beta = compute_F0_F1(
        log_trans, state_types, sub_matrix, pi_out,
        x_pad, y_pad, jnp.asarray(Lx), jnp.asarray(Ly))

    # Build padded boost-state JAX tensors.
    boost_state_jax = _pad_boost_state_to_jax(
        boost_state, Lx_pad, Ly_pad, x_seq, y_seq)

    # Reuse the masked emission table for the restart-Forward.
    emit_template = pair_hmm_emissions(state_types, x_pad, y_pad,
                                       sub_matrix, pi_out)
    mask = _emit_mask(jnp.asarray(Lx), jnp.asarray(Ly),
                      Lx_pad, Ly_pad, state_types.shape[0])
    emit_template = jnp.where(mask, emit_template, NEG_INF)

    eps = 1.0 / float(alpha_z)

    # Build the list of anchors: positions (i_a, j_a) for i_a in
    # 1..Lx, j_a in 1..Ly. Optionally pruned by Q_baseline >= q_min.
    Q_baseline_np = np.asarray(F1) / max(float(F0), 1e-300)
    anchor_mask = Q_baseline_np >= q_min                    # (Lx, Ly)
    anchor_idx = np.argwhere(anchor_mask)                   # (n_anchors, 2)
    if anchor_idx.size == 0:
        # Nothing to do; return baseline.
        return Q_baseline_np, float(F0), Q_baseline_np, float(log_F0)

    n_anchors = anchor_idx.shape[0]

    # Anchor positions are 1-based (residue position == array index + 1).
    anchor_is = anchor_idx[:, 0] + 1
    anchor_js = anchor_idx[:, 1] + 1

    # Pad to multiple of chunk_size with placeholder (Lx_pad, Ly_pad)
    # which the kernel masks via real_mask.
    n_chunks = (n_anchors + chunk_size - 1) // chunk_size
    pad_to = n_chunks * chunk_size
    anchor_is_pad = np.full(pad_to, Lx_pad, dtype=np.int32)
    anchor_js_pad = np.full(pad_to, Ly_pad, dtype=np.int32)
    anchor_is_pad[:n_anchors] = anchor_is
    anchor_js_pad[:n_anchors] = anchor_js

    # Allocate accumulators.
    anchor_numer_acc = np.zeros((Lx, Ly), dtype=np.float64)
    l_exact_acc = 0.0
    # Numer from a partner perspective: when (i_a, j_a) is the "smaller"
    # anchor and (k, l) is the partner, F2(i_a, j_a; k, l) * M is added
    # to BOTH (i_a, j_a)'s and (k, l)'s numerators (both are the
    # column-pair under the unordered-partner convention of eq:exact-Q).
    # Our per-anchor kernel returns the partner-numer field
    # (Lx_pad, Ly_pad) which contains F2 * M for every (k, l) with
    # k != i_a -- including k > i_a AND k < i_a. Summing this directly
    # over anchors gives, for output Q'[k, l],
    #   sum_{(i_a, j_a), i_a != k} F2(i_a, j_a; k, l) * M(i_a, j_a; k, l)
    # which is exactly the numerator of eq:exact-Q at (k, l). Each
    # unordered pair {(i_a, j_a), (k, l)} thus contributes its F2*M to
    # both sides via the two anchor visits, as required.
    partner_numer_acc = np.zeros((Lx, Ly), dtype=np.float64)

    real_Lx_t = jnp.asarray(Lx)
    real_Ly_t = jnp.asarray(Ly)

    for chunk_idx in range(n_chunks):
        s = chunk_idx * chunk_size
        e = s + chunk_size
        ai = jnp.asarray(anchor_is_pad[s:e])
        aj = jnp.asarray(anchor_js_pad[s:e])
        anchor_numers, partner_fields, l_exacts = _process_anchor_chunk(
            log_trans, state_types, emit_template,
            log_alpha, log_beta, boost_state_jax,
            real_Lx_t, real_Ly_t, ai, aj,
            Lx_pad, Ly_pad)
        anchor_numers = np.asarray(anchor_numers)
        partner_fields = np.asarray(partner_fields)
        l_exacts = np.asarray(l_exacts)
        # Each chunk slot's contribution.
        for slot in range(chunk_size):
            global_idx = s + slot
            if global_idx >= n_anchors:
                break
            i_a = int(anchor_is_pad[global_idx])    # position
            j_a = int(anchor_js_pad[global_idx])
            i_res = i_a - 1                         # 0-based residue idx
            j_res = j_a - 1
            anchor_numer_acc[i_res, j_res] += float(anchor_numers[slot])
            l_exact_acc += float(l_exacts[slot])
            # partner_fields[slot] is (Lx_pad, Ly_pad); its real region
            # 0..Lx-1, 0..Ly-1 contains the F2*M entries indexed by
            # partner (k, l) (in 0-based residue coords). Add to
            # partner_numer_acc.
            partner_numer_acc += partner_fields[slot, :Lx, :Ly]

    # Combine the two numerator-accumulators. Q'_{ij} numerator (eq:exact-Q):
    #   F1(i, j) + eps * sum_{(k, l), k != i} F2(i, j; k, l) * M(i, j; k, l)
    #
    # Reachability of F2 in our restart-Forward (mu starts at M(i_a, j_a)):
    # mu[k, l, M] is nonzero only for (k, l) >= (i_a, j_a) componentwise
    # (and in fact for k > i_a AND l > j_a strict, since revisiting the
    # same X or Y column would imply multiple Match emissions for one
    # residue, which a Pair HMM forbids). So the per-anchor kernel only
    # produces F2 entries for partner (k, l) with k > i_a.
    #
    # F2 is symmetric: F2(i, j; k, l) = F2(k, l; i, j). And M is
    # symmetric: M(i, j; k, l) = M(k, l; i, j) (by the symmetry of
    # joint_per_cp under simultaneous swap of class indices and
    # residue indices). So the numerator at output cell (i, j) splits
    # naturally into two halves under the lex order:
    #   sum_{k > i, l} F2(i, j; k, l) * M     [anchor at (i, j)]
    #   + sum_{k < i, l} F2(i, j; k, l) * M
    #   = sum_{k > i, l} F2(i, j; k, l) * M
    #   + sum_{k < i, l} F2(k, l; i, j) * M(k, l; i, j)  [by symmetry]
    #
    # The first half is anchor_numer_acc[i, j] (this anchor's
    # contribution from partners "above" it). The second half is
    # partner_numer_acc[i, j] (where some other anchor (k, l) with
    # k < i had (i, j) as one of its partners, accumulated via the
    # partner_numer_field returned by _per_anchor_kernel).
    #
    # The partner_numer_field already restricts to k > anchor_i (not
    # k != anchor_i), so each unordered pair contributes its F2*M to
    # exactly ONE entry of partner_numer_acc at the (i, j) end and to
    # exactly ONE entry of anchor_numer_acc at the (k, l) end --
    # never double-counted.
    numer_total = anchor_numer_acc + partner_numer_acc

    F1_np = np.asarray(F1)
    Q_prime_unnorm = F1_np + eps * numer_total              # (Lx, Ly)

    # L_exact accumulator was constrained to (k > i_a), i.e. each
    # unordered (i, j) < (k, l) pair counted exactly once per the
    # eq:exact-01 convention (i < k). Add to F0 to form the partition.
    L_exact_val = float(F0) + eps * l_exact_acc

    Q_prime = Q_prime_unnorm / L_exact_val
    Q_baseline = Q_baseline_np                              # F1 / F0

    return Q_prime, L_exact_val, Q_baseline, float(log_F0)
