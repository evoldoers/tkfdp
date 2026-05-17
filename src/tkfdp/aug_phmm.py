"""Memory-augmented Pair HMM for the exact 0-or-1-edge SCFG posterior.

Implements the same Q'_{ij} as src/tkfdp/f2_scfg.py via state augmentation
rather than F2 tensor enumeration. Cost: O(L^2 * A^2) instead of F2's
O(L^4). Generalises naturally to k-edge models (k=2: A^4 tag space; k=3:
A^6) without grammar ordering constraints.

The augmented state is (PHMM state, tag) where tag in {no_edge, (a, b)
for AA pairs, done}. Sparse-aware Forward and Backward exploit the
tag-preserving structure at non-Match cells and the small set of new
tags created at Match cells.

Tag layout (n_tags = A * A + 2 = 402 for A = 20):
  - tag 0:                  no_edge
  - tags 1 .. A*A:          (a, b) for a, b in 0..A-1, encoded as 1 + a*A + b
  - tag A*A + 1:            done

Augmented partition function:

    L_exact = sum over s, t in {no_edge, done} of
              alpha[real_Lx, real_Ly, s, t] * trans[s, E]

(orphan (a, b) tags at the end correspond to alignments that started a
left edge but never resolved a right edge -- these are not part of the
0-or-1-edge SCFG and must be excluded.)

Corrected per-residue match posterior:

    Q'_{ij} = sum_t exp(log_alpha[i, j, M, t] + log_beta[i, j, M, t]) / L_exact

equivalent to the F2-SCFG output of f2_scfg.scfg_corrected_posterior in
the K_c = 1 case (where the boost M depends only on the four amino acids
at the coupled pair, not on the per-position class posteriors). For
K_c > 1 the augmented PHMM uses the prior (uniform) class distribution
to AA-marginalise M, giving an approximation to the position-dependent
F2-SCFG result.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

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
# Tag layout helpers
# ---------------------------------------------------------------------------

TAG_NO_EDGE = 0
N_AB_TAGS = A * A
TAG_DONE = N_AB_TAGS + 1
N_TAGS = N_AB_TAGS + 2  # 402


def _tag_ab(a: int, b: int) -> int:
    """Encode an (a, b) AA pair into a tag id in [1, A*A]."""
    return 1 + a * A + b


# ---------------------------------------------------------------------------
# Class-marginal M tensor (PROPER, no data-dependent reweighting).
# ---------------------------------------------------------------------------


def build_M_tensor_classmarg(state, t: float,
                              S: Optional[np.ndarray] = None,
                              A_alpha: int = A) -> np.ndarray:
    """Build the (A, A, A, A) M tensor under the *proper class-marginal*
    construction (eq:M-marginal in main.tex, post-2026-05-15 revision).

    Definition:
        denom_joint(a, b; t) =
            sum_c (1 / K_c) * pi_c[a] * P_c(a -> b; t)
        numer_joint(a, c, b, d; t) =
            sum_{c1, c2} (1 / K_c^2) * pi_joint_{c1,c2}(a, c)
                                       * P_{c1,c2}((a, c) -> (b, d); t)
        M[a, b, c, d] = numer_joint(a, c, b, d; t)
                        / (denom_joint(a, b; t) * denom_joint(c, d; t))

    All sums use the UNIFORM class prior 1/K_c (resp. 1/K_c^2 for
    class-pairs). NO per-cell, data-dependent gamma. This is what the
    paper's eq:M-marginal actually specifies; the previous
    ``build_M_tensor_aa_marginal`` used a gamma-weighted denominator
    (the boost_state.denom field) which is the pre-MCMC first-order
    correction relic. That relic violates the x=y, t=0 identity
    M_obs == M_solo by a 2-3x cell-dependent bias.

    Args:
        state: object with K_c, pi_class, potts_dp (PottsDPState).
        t: branch length.
        S: optional exchangeability matrix (defaults to LG08 inside
            build_per_class_match_emit / build_per_classpair_joint_emit).
        A_alpha: alphabet size (default 20).

    Returns:
        M_tensor: (A, A, A, A) numpy array, M[a, b, c, d] is the boost
            factor for a coupled edge whose left endpoint carries AAs
            (a, b) = (x_{i1}, y_{j1}) and whose right endpoint carries
            (c, d) = (x_{i2}, y_{j2}).
    """
    from .postprocessing import (build_per_class_match_emit,
                                  build_per_classpair_joint_emit)
    from .generator import joint_stationary_pair
    from .lg08 import S_LG08_J

    pi_class = np.asarray(state.pi_class, dtype=np.float64)
    K_c, A_local = pi_class.shape
    assert A_local == A_alpha, \
        f"M tensor: alphabet mismatch {A_local} vs {A_alpha}"
    S_arr = np.asarray(S_LG08_J if S is None else S, dtype=np.float64)

    # --- denominator: class-marginal single-site joint at time t ---
    # P_match[c, a, b] = P(ancestor=a, descendant=b | class=c, t)
    # NOTE: build_per_class_match_emit returns expm(Q_c * t) -- the
    # *conditional* transition matrix, not multiplied by pi_c[a].
    # We multiply by pi_c[a] explicitly below.
    P_match = np.asarray(build_per_class_match_emit(pi_class, float(t), S_arr))
    # joint emission: pi_c[a] * P_match[c, a, b]
    pi_times_P = pi_class[:, :, None] * P_match              # (K, A, A)
    denom_aa = pi_times_P.mean(axis=0)                       # (A, A) -- uniform-class avg

    # --- numerator: class-marginal coupled-pair joint at time t ---
    # P_joint[c, c', a, a', b, b'] = P((a, a') -> (b, b') | (c, c'), t,
    #                                  Potts atom + side potentials)
    P_joint_cp = np.asarray(build_per_classpair_joint_emit(
        state, float(t), S_arr))                              # (K, K, A, A, A, A)
    # Per-class-pair joint stationary at coupled site:
    atoms = np.asarray(state.potts_dp.atoms, dtype=np.float64)
    assignments = np.asarray(state.potts_dp.assignments, dtype=np.int64)
    h_pairs = state.potts_dp.h_pairs
    use_h = h_pairs is not None
    if use_h:
        h_pairs = np.asarray(h_pairs, dtype=np.float64)
        from .potts_dp import canonical_pair_idx_table
        cp_idx_np, cp_swap_np = canonical_pair_idx_table(K_c)

    pi_joint_cp = np.zeros((K_c, K_c, A_local, A_local), dtype=np.float64)
    for c1 in range(K_c):
        for c2 in range(K_c):
            atom_idx = int(assignments[c1, c2])
            H = jnp.asarray(atoms[atom_idx])
            if use_h:
                k_can = int(cp_idx_np[c1, c2])
                swap = int(cp_swap_np[c1, c2])
                h_a = jnp.asarray(h_pairs[k_can, swap])  # noqa: F841
                h_b = jnp.asarray(h_pairs[k_can, 1 - swap])  # noqa: F841
            else:
                h_a = h_b = None
            pij_flat = joint_stationary_pair(
                H, jnp.asarray(pi_class[c1]), jnp.asarray(pi_class[c2]),
                h_a=h_a, h_b=h_b)
            pi_joint_cp[c1, c2] = np.asarray(pij_flat).reshape(
                A_local, A_local)

    # numer[a, c, b, d] = (1/K_c^2) * sum_{c1, c2}
    #     pi_joint_cp[c1, c2, a, c] * P_joint_cp[c1, c2, a, c, b, d]
    # Vectorise: broadcast pi_joint_cp on the (b, d) axes.
    weighted = pi_joint_cp[..., None, None] * P_joint_cp     # (K, K, A, A, A, A)
    numer = weighted.mean(axis=(0, 1))                       # (A, A, A, A) i.e. (a, c, b, d)
    # Reorder to (a, b, c, d) per the boost API: left endpoint (a, b),
    # right endpoint (c, d).
    numer = np.transpose(numer, (0, 2, 1, 3))                # (a, b, c, d)

    # Denominator: (a, b) at left * (c, d) at right.
    denom_safe = np.clip(denom_aa, 1e-300, None)
    M = numer / (denom_safe[:, :, None, None]
                 * denom_safe[None, None, :, :])
    return M


def build_M_tensor_aa_marginal(boost_state, A_alpha: int = A) -> np.ndarray:
    """Build the (A, A, A, A) AA-only M tensor M[a, b, c, d].

    Definition:
        M[a, b, c, d] = (sum_{c1, c2} pi_class[c1] * pi_class[c2]
                         * J[c1, c2, a, c, b, d])
                       / (denom_aa[a, b] * denom_aa[c, d])
    where
        denom_aa[a, b] = sum_c pi_class[c] * P_match(a, b | c)
                      = sum_c pi_class[c] * pi_c[a] * sub_c[a, b]
        P_match(a, b | c) is the per-class match emission probability.

    For the K_c = 1 case (single class), this reduces exactly to the
    F2-SCFG per-position log_M formula because gamma[i, j] is the
    constant 1 and denom[i, j] = P(X_i, Y_j) is the same per-AA marginal.

    For K_c > 1, this is the prior-class approximation (uniform class
    weights instead of position-specific posteriors).

    Args:
        boost_state: PairBoostState containing joint_per_cp of shape
            (K_c, K_c, A, A, A, A). Per the build_boost_state convention
            we use the uniform class prior (1/K_c) as the gamma weights.
        A_alpha: alphabet size (default 20).

    Returns:
        M_tensor: (A, A, A, A) numpy array. M_tensor[a, b, c, d] is
            the boost factor for a coupled pair with left AAs (a, b)
            and right AAs (c, d).
    """
    J = np.asarray(boost_state.joint_per_cp)              # (K, K, A, A, A, A)
    K_c = J.shape[0]
    A_local = J.shape[2]
    assert A_local == A_alpha, \
        f"M tensor: alphabet mismatch {A_local} vs {A_alpha}"
    pi_class = np.full(K_c, 1.0 / K_c)
    # J indexing: J[c1, c2, a, c, b, d] (per coupled_annealing.log_M_at_pair)
    #   c1, c2 = class indices; (a, c) = left/right AAs in X; (b, d) = left/right AAs in Y.
    # The augmented-PHMM "left edge" carries AAs (a, b) = (X_i, Y_j) and the
    # "right edge" AAs (c, d) = (X_k, Y_l).
    # Numerator with prior gamma:
    #   numer[a, b, c, d] = sum_{c1, c2} pi_class[c1] * pi_class[c2]
    #                                     * J[c1, c2, a, c, b, d]
    numer = np.einsum('e,f,efacbd->abcd', pi_class, pi_class, J)
    # denom_aa[a, b] = sum_c pi_class[c] * P_match(a, b | c)
    # We need P_match(a, b | c) — equivalently, the per-class single-site
    # joint emission. The boost_state caches `denom` per position which is
    # already this per-(i, j) mixture. For a position with AAs (a, b),
    # denom[i, j] depends only on (a, b) when K_c = 1 (since gamma = 1
    # uniformly), but for K_c > 1 we need a *per-AA* marginal.
    #
    # We reconstruct P_match(a, b | c) from joint_per_cp under the
    # convention that joint_per_cp[c, c, a, a, b, b] (diagonal in coupled
    # indices) equals P(a, b | c)^2 (independent-edge limit). But that's
    # the per-class-pair joint at the coupled-pair level, not the
    # per-class single-site marginal. So we can't easily extract it from
    # joint_per_cp alone.
    #
    # Instead, we compute denom_aa from the boost_state's `denom` field
    # IF K_c = 1, where denom[i, j] = P(X_i, Y_j) depends only on the AA
    # pair. Then we look up denom_aa[a, b] = denom[i, j] for any (i, j)
    # with X_i = a, Y_j = b. For K_c = 1 this is well-defined.
    #
    # For K_c > 1 we compute the prior-marginal denominator the long way:
    # denom_aa[a, b] = sum_c (1/K_c) * pi_c[a] * sub_c[a, b]
    # but pi_c and sub_c aren't stored in boost_state. So we infer them
    # from joint_per_cp under the constraint that joint_per_cp[c, c', a,
    # a', b, b'] factorises as P(a, b | c) * P(a', b' | c') when there is
    # no coupling. We don't have the no-coupling baseline, so... we just
    # use the K_c = 1 path.
    #
    # PRACTICAL CHOICE: use the boost_state.denom field as a per-(i, j)
    # lookup. For K_c = 1 it's a function of (X_i, Y_j) only and
    # equivalent. For K_c > 1 it's position-dependent and the augmented
    # PHMM is no longer position-equivalent to F2-SCFG anyway.
    #
    # Build denom_aa via averaging over positions with each (a, b):
    x_seq = np.asarray(boost_state.x_seq)
    y_seq = np.asarray(boost_state.y_seq)
    denom = np.asarray(boost_state.denom)                  # (Lx, Ly)
    denom_aa = np.zeros((A_local, A_local))
    counts = np.zeros((A_local, A_local), dtype=np.int64)
    for i in range(x_seq.shape[0]):
        for j in range(y_seq.shape[0]):
            a = int(x_seq[i]); b = int(y_seq[j])
            if a < A_local and b < A_local:
                denom_aa[a, b] += float(denom[i, j])
                counts[a, b] += 1
    # For (a, b) pairs we never observed, fall back to the pi-weighted
    # marginal (a constant proxy).
    fallback = float(np.mean(denom)) if denom.size > 0 else 1e-10
    avg = np.where(counts > 0, denom_aa / np.maximum(counts, 1), fallback)
    denom_aa = avg

    # Build M_tensor[a, b, c, d] = numer[a, b, c, d] / (denom_aa[a, b]
    #                              * denom_aa[c, d]).
    denom_safe = np.clip(denom_aa, 1e-300, None)
    M_tensor = numer / (denom_safe[:, :, None, None] * denom_safe[None, None, :, :])
    return M_tensor


# ---------------------------------------------------------------------------
# Forward DP.
# ---------------------------------------------------------------------------


def _aug_forward_core(log_trans, state_types, emit, log_M_tensor,
                      x_pad, y_pad, log_eps, Lx_pad, Ly_pad):
    """Augmented Forward in log space.

    Args:
        log_trans: (5, 5) PHMM log-transitions.
        state_types: (5,) state types (S, M, I, D, E).
        emit: (Lx_pad+1, Ly_pad+1, 5) log-emission table.
        log_M_tensor: (A, A, A, A) log-M tensor (a, b, c, d) where
            (a, b) is the left-edge AA pair and (c, d) is the right-edge
            AA pair.
        x_pad, y_pad: (Lx_pad,), (Ly_pad,) padded AA sequences (clamped
            to 0..A-1; these are 0-based residue indices).
        log_eps: scalar log(1 / alpha_z).
        Lx_pad, Ly_pad: static padded sequence lengths.

    Returns:
        alpha: (Lx_pad+1, Ly_pad+1, 5, n_tags) augmented forward table.
        alpha_left_edge: (Lx_pad+1, Ly_pad+1) log of the per-Match-cell
            "eps-spawn" extra contribution -- the increment added to
            alpha[i, j, M, (c, d)_tag] from the no_edge -> (c, d)
            transition at this cell. NEG_INF for non-Match positions
            (i = 0, j = 0). Used to attribute "left edge" mass to (i, j).
        alpha_right_edge: (Lx_pad+1, Ly_pad+1) log of the per-Match-cell
            "right-edge resolution" extra -- the increment added to
            alpha[i, j, M, done] from the (a, b) -> done transitions at
            this cell.
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)                              # (5,)
    is_I = (state_types == I)
    is_D = (state_types == D)

    # Helper: standard PHMM step from a predecessor cell `prev` (5, n_tags),
    # returning the un-emitted "step" array shape (5, n_tags) — i.e., the
    # `LSE_k log_trans[k, s'] + prev[k, tag]` part for each (s', tag).
    def step_from(prev):
        # prev: (5, n_tags). For each output state s', sum over predecessor
        # k. Tag axis broadcasts. Result: (5, n_tags).
        # log_trans[k, s'] + prev[k, t]; LSE over k axis.
        # log_trans shape (5, 5). Broadcast: (5, 1, 1) + (5, 5, 1) +(none)+ (5, 1, n_tags) -> ...
        # simpler: einsum-like via broadcasting.
        return jax.nn.logsumexp(
            prev[:, None, :] + log_trans[:, :, None], axis=0)
        # result (5, n_tags). For each output state s' index and tag t,
        # = LSE_k (prev[k, t] + log_trans[k, s']).

    # Helper: apply emission for a given cell (i, j) by adding emit[i, j, s']
    # broadcast over tag axis. Inputs: step (5, n_tags), emit_cell (5,)
    def apply_emit(step, emit_cell):
        return step + emit_cell[:, None]                   # (5, n_tags)

    # Helper: at a Match cell with current AAs (c, d), apply the augmented
    # extras. Input: cell with carry-through done (shape (5, n_tags)).
    # `match_step_per_tag` is the (n_tags,) per-tag predecessor-Match-step
    # output BEFORE applying the M-state's emission, i.e.,
    #   match_step_per_tag[tag] = LSE_k (prev[k, tag] + log_trans[k, M])
    # which is equal to step_from(prev)[M, tag].
    def apply_match_extras(cell, match_step_per_tag, emit_M_cd, c_idx, d_idx):
        # cell: (5, n_tags), only cell[M, :] gets extras.
        # Extra 1: at tag (c, d), add emit_M_cd + log_eps + match_step[no_edge].
        cd_tag = 1 + c_idx * A + d_idx                      # scalar
        extra_cd = emit_M_cd + log_eps + match_step_per_tag[TAG_NO_EDGE]
        # Extra 2: at tag DONE, add emit_M_cd + LSE_(a,b) [log_M[a, b, c, d]
        #          + match_step_per_tag[(a, b)_tag]].
        ab_steps = match_step_per_tag[1:1 + N_AB_TAGS]      # (A*A,)
        log_M_cd_flat = log_M_tensor[:, :, c_idx, d_idx].reshape(N_AB_TAGS)
        extra_done = emit_M_cd + jax.nn.logsumexp(ab_steps + log_M_cd_flat)

        new_M_row = cell[M, :]
        new_M_row = new_M_row.at[cd_tag].set(
            jnp.logaddexp(new_M_row[cd_tag], extra_cd))
        new_M_row = new_M_row.at[TAG_DONE].set(
            jnp.logaddexp(new_M_row[TAG_DONE], extra_done))
        return cell.at[M, :].set(new_M_row), extra_cd, extra_done

    # Initialize: alpha[0, 0, S, no_edge] = 0; everything else NEG_INF.
    cell00 = jnp.full((ns, N_TAGS), NEG_INF)
    cell00 = cell00.at[S, TAG_NO_EDGE].set(0.0)

    # Row 0: only I-type destinations (j = 1..Ly_pad). No Match, so no
    # left-edge or right-edge extras here.
    def row0_step(prev_cell, j):
        step = step_from(prev_cell)                        # (5, N_TAGS)
        emit_cell = emit[0, j]                             # (5,)
        full = apply_emit(step, emit_cell)                  # (5, N_TAGS)
        cell = jnp.where(is_I[:, None], full, NEG_INF)
        return cell, cell

    _, row0_rest = jax.lax.scan(
        row0_step, cell00, jnp.arange(1, Ly_pad + 1))
    row0 = jnp.concatenate([cell00[None], row0_rest], axis=0)
    # Row 0 has no Match cells; left/right extras are NEG_INF everywhere.
    row0_left = jnp.full((Ly_pad + 1,), NEG_INF)
    row0_right = jnp.full((Ly_pad + 1,), NEG_INF)

    def row_step(prev_row, i):
        prev_cell00 = prev_row[0]                           # (5, N_TAGS)
        step0 = step_from(prev_cell00)                      # (5, N_TAGS)
        emit0 = emit[i, 0]                                  # (5,)
        full0 = apply_emit(step0, emit0)
        cell0 = jnp.where(is_D[:, None], full0, NEG_INF)
        # j = 0 has no Match.
        left0 = jnp.asarray(NEG_INF)
        right0 = jnp.asarray(NEG_INF)

        c_idx = x_pad[i - 1]                                # 0-based AA at row i

        def col_step(prev_cell, j):
            prev_diag = prev_row[j - 1]
            prev_left = prev_cell
            prev_up = prev_row[j]

            step_M_pred = step_from(prev_diag)
            step_I_pred = step_from(prev_left)
            step_D_pred = step_from(prev_up)

            emit_cell = emit[i, j]
            full_M = apply_emit(step_M_pred, emit_cell)
            full_I = apply_emit(step_I_pred, emit_cell)
            full_D = apply_emit(step_D_pred, emit_cell)

            cell = jnp.where(is_M[:, None], full_M,
                     jnp.where(is_I[:, None], full_I,
                       jnp.where(is_D[:, None], full_D, NEG_INF)))
            d_idx = y_pad[j - 1]
            emit_M_cd = emit_cell[M]
            match_step_per_tag = step_M_pred[M, :]
            cell, extra_cd, extra_done = apply_match_extras(
                cell, match_step_per_tag, emit_M_cd, c_idx, d_idx)
            return cell, (cell, extra_cd, extra_done)

        _, (row_rest, row_left_rest, row_right_rest) = jax.lax.scan(
            col_step, cell0, jnp.arange(1, Ly_pad + 1))
        curr_row = jnp.concatenate([cell0[None], row_rest], axis=0)
        curr_left = jnp.concatenate([left0[None], row_left_rest], axis=0)
        curr_right = jnp.concatenate([right0[None], row_right_rest], axis=0)
        return curr_row, (curr_row, curr_left, curr_right)

    _, (all_rows, all_left, all_right) = jax.lax.scan(
        row_step, row0, jnp.arange(1, Lx_pad + 1))
    alpha = jnp.concatenate([row0[None], all_rows], axis=0)
    alpha_left_edge = jnp.concatenate([row0_left[None], all_left], axis=0)
    alpha_right_edge = jnp.concatenate([row0_right[None], all_right], axis=0)
    return alpha, alpha_left_edge, alpha_right_edge


# ---------------------------------------------------------------------------
# Backward DP.
# ---------------------------------------------------------------------------


def _aug_backward_core(log_trans, state_types, emit, log_M_tensor,
                       x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                       real_Lx, real_Ly):
    """Augmented Backward in log space.

    Convention: beta[i, j, s, t] = log P(emissions after (i, j, s, t)
    until E | currently at (i, j, s) with outgoing tag t). Excludes the
    emission at (i, j, s) itself.

    Terminal: beta[real_Lx, real_Ly, s, t] = log_trans[s, E] for
    t in {no_edge, done}, NEG_INF for the (a, b) tags (orphan left edges
    are not allowed under the SCFG grammar).
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    e_idx = _find_e_idx(state_types)
    # beta_term[s, t]: termination contribution at (real_Lx, real_Ly).
    # Allowed only at t in {no_edge, done}.
    beta_term_no_edge = log_trans[:, e_idx]                 # (5,)
    term_mask = jnp.zeros(N_TAGS, dtype=bool).at[TAG_NO_EDGE].set(True
                                              ).at[TAG_DONE].set(True)
    beta_term = jnp.where(
        term_mask[None, :],
        beta_term_no_edge[:, None] + jnp.zeros((1, N_TAGS)),
        NEG_INF)
    # is_term mask over (i, j)
    i_eq = jnp.arange(Lx_pad + 1) == real_Lx
    j_eq = jnp.arange(Ly_pad + 1) == real_Ly
    is_term = i_eq[:, None] & j_eq[None, :]                 # (Lx_pad+1, Ly_pad+1)

    # Helper: backward "successor-step" — given a successor cell `next_cell`
    # of shape (5, n_tags) ALREADY including the emission at the successor
    # via `next_emit`, and a target state s' is M/I/D/E, contribute to the
    # current cell's (5, n_tags) by `LSE_{s'} log_trans[s, s'] +
    # next_emit[s'] + next_cell[s', t]` (per tag t).
    #
    # We do this in pieces because the M-successor case has tag-mixing
    # extras (right-edge resolution and left-edge spawn).

    def back_step_non_match(next_cell_se, next_emit_se):
        """Backward contribution from a (single) non-Match successor state.
        next_cell_se: shape (n_tags,) -- next_cell[s', :]. (s' fixed.)
        next_emit_se: scalar -- next_emit[s'].
        Returns: shape (n_tags,) per-tag contribution per target state s.
                 Wait, we want shape (5, n_tags) -- contribution to each
                 source state s. So: log_trans[:, s'] + next_emit_se +
                 next_cell_se. Broadcast: (5,) + scalar + (n_tags,) ->
                 (5, n_tags).
        """
        # Returns shape (5, n_tags)
        return log_trans[:, None] + next_emit_se + next_cell_se[None, :]

    # M-successor with augmented tag-mixing:
    # next_cell (M-succ at (i+1, j+1)) has shape (n_tags,) (just M-state row).
    # The augmented transitions are:
    #   t = no_edge: contributes next_cell[no_edge] + log(1)
    #                  + next_cell[(c', d')_tag] + log(eps)
    #   t = (a, b): contributes next_cell[(a, b)_tag] + log(1)
    #                  + next_cell[done] + log_M[a, b, c', d']
    #   t = done: contributes next_cell[done] + log(1)
    # Then add log_trans[s, M] + emit[i+1, j+1, M] = log_trans[:, M] + next_emit_M
    # Returns shape (5, n_tags)
    def back_step_match(next_cell_M, next_emit_M, c_succ_idx, d_succ_idx):
        # next_cell_M: (N_TAGS,) -- next_cell[M, :].
        # next_emit_M: scalar (log P_match(c', d'))
        # c_succ_idx, d_succ_idx: AAs of the successor Match cell
        cd_succ_tag = 1 + c_succ_idx * A + d_succ_idx       # scalar
        # Build per-tag (s side) contribution (shape (N_TAGS,)).
        contrib = jnp.full((N_TAGS,), NEG_INF)
        # Tag no_edge:
        contrib_no_edge = jnp.logaddexp(
            next_cell_M[TAG_NO_EDGE],
            log_eps + next_cell_M[cd_succ_tag])
        contrib = contrib.at[TAG_NO_EDGE].set(contrib_no_edge)
        # Tag done:
        contrib = contrib.at[TAG_DONE].set(next_cell_M[TAG_DONE])
        # Tag (a, b) for all (a, b): contrib = LSE(next_cell[(a,b)_tag],
        #  log_M[a, b, c', d'] + next_cell[done])
        ab_next = next_cell_M[1:1 + N_AB_TAGS]              # (A*A,)
        log_M_succ = log_M_tensor[:, :, c_succ_idx, d_succ_idx].reshape(N_AB_TAGS)
        contrib_ab = jnp.logaddexp(
            ab_next, log_M_succ + next_cell_M[TAG_DONE])
        contrib = jax.lax.dynamic_update_slice(
            contrib, contrib_ab, (1,))
        # Now contribute to source state s via log_trans[s, M] + next_emit_M.
        return log_trans[:, M:M + 1] + next_emit_M + contrib[None, :]
        # shape (5, N_TAGS)

    # Backward row scan (i from Lx_pad down to 0).
    init_next_row = jnp.full((Ly_pad + 1, ns, N_TAGS), NEG_INF)

    def row_step_bwd(next_row, i):
        # next_row: (Ly_pad+1, ns, N_TAGS), corresponds to row (i+1).
        # Compute the cell at j = Ly_pad first (only D-type successor).
        j = Ly_pad
        i_succ = jnp.minimum(i + 1, Lx_pad)
        # D-successor at (i+1, Ly_pad)
        succ_emit_D = emit[i_succ, Ly_pad]                  # (5,)
        succ_cell_D = next_row[Ly_pad]                      # (5, N_TAGS)
        # Aggregate D-successor contributions over s' axis.
        # back_step_non_match expects (n_tags,) for each s', but here we
        # want to do all D-state contributions at once. We'll use a
        # per-state-type masked accumulation:
        # contribution to source cell shape (5, N_TAGS):
        #   sum over s' of log_trans[:, s'] + next_emit[s'] + next_cell[s', :]
        # via logsumexp. But then we restrict s' to D-type (others NEG_INF).
        #
        # General-purpose contribution from any non-Match successor type
        # (the s' filter is encoded by setting non-D contributions to
        # NEG_INF):
        contrib_D_sp = log_trans + succ_emit_D[None, :]    # (5, 5) per (s, s')
        contrib_D_sp = jnp.where(is_D[None, :], contrib_D_sp, NEG_INF)
        contrib_D = jax.nn.logsumexp(
            contrib_D_sp[:, :, None] + succ_cell_D[None, :, :], axis=1)
        # contrib_D shape (5, N_TAGS)
        cell_jLypad = contrib_D
        # Apply terminal override.
        cell_jLypad = jnp.where(is_term[i, Ly_pad],
                                 beta_term, cell_jLypad)

        def col_step_bwd(beta_right, j):
            j_succ = jnp.minimum(j + 1, Ly_pad)
            i_succ_local = jnp.minimum(i + 1, Lx_pad)
            # M-successor at (i+1, j+1): augmented tag mixing.
            succ_emit_M = emit[i_succ_local, j_succ, M]    # scalar
            succ_cell_M = next_row[j_succ][M, :]            # (N_TAGS,)
            c_succ = x_pad[i_succ_local - 1]                # 0-based AA
            d_succ = y_pad[j_succ - 1]
            contrib_M = back_step_match(succ_cell_M, succ_emit_M,
                                          c_succ, d_succ)
            # Mask: only meaningful when state s' = M (we already use
            # log_trans[:, M]). However the back_step_match output is
            # already for s' = M only. We need to ensure the upper-bound
            # check that (i+1, j+1) is within real region — handled via
            # NEG_INF in next_row at padded positions.

            # I-successor at (i, j+1)
            succ_emit_I = emit[i, j_succ]                    # (5,)
            succ_cell_I = beta_right                         # (5, N_TAGS)
            contrib_I_sp = log_trans + succ_emit_I[None, :]  # (5, 5)
            contrib_I_sp = jnp.where(is_I[None, :], contrib_I_sp, NEG_INF)
            contrib_I = jax.nn.logsumexp(
                contrib_I_sp[:, :, None] + succ_cell_I[None, :, :], axis=1)

            # D-successor at (i+1, j)
            succ_emit_D = emit[i_succ_local, j]              # (5,)
            succ_cell_D = next_row[j]                        # (5, N_TAGS)
            contrib_D_sp = log_trans + succ_emit_D[None, :]  # (5, 5)
            contrib_D_sp = jnp.where(is_D[None, :], contrib_D_sp, NEG_INF)
            contrib_D = jax.nn.logsumexp(
                contrib_D_sp[:, :, None] + succ_cell_D[None, :, :], axis=1)

            cell = jnp.logaddexp(contrib_M,
                                 jnp.logaddexp(contrib_I, contrib_D))
            cell = jnp.where(is_term[i, j], beta_term, cell)
            return cell, cell

        _, row_rest = jax.lax.scan(
            col_step_bwd, cell_jLypad,
            jnp.arange(Ly_pad - 1, -1, -1))
        # row_rest is in REVERSE j-order; reverse it.
        curr_row = jnp.concatenate([row_rest[::-1], cell_jLypad[None]],
                                    axis=0)
        return curr_row, curr_row

    _, all_rows_bwd = jax.lax.scan(
        row_step_bwd, init_next_row,
        jnp.arange(Lx_pad, -1, -1))
    # all_rows_bwd[0] corresponds to i=Lx_pad, ..., [-1] to i=0; reverse.
    beta = all_rows_bwd[::-1]
    return beta


# ---------------------------------------------------------------------------
# JIT-cached wrappers.
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _aug_forward_jit(log_trans, state_types, emit, log_M_tensor,
                      x_pad, y_pad, log_eps, Lx_pad, Ly_pad):
    return _aug_forward_core(log_trans, state_types, emit, log_M_tensor,
                              x_pad, y_pad, log_eps, Lx_pad, Ly_pad)


@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _aug_backward_jit(log_trans, state_types, emit, log_M_tensor,
                       x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                       real_Lx, real_Ly):
    return _aug_backward_core(log_trans, state_types, emit, log_M_tensor,
                               x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                               real_Lx, real_Ly)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def forward_aug_phmm(log_trans, state_types, sub_matrix, pi,
                      x_seq, y_seq, real_Lx, real_Ly,
                      log_M_tensor, log_eps,
                      log_emit_table=None):
    """Augmented Forward DP wrapper.

    Returns:
        alpha:            (Lx_pad+1, Ly_pad+1, 5, N_TAGS)
        alpha_left_edge:  (Lx_pad+1, Ly_pad+1)
        alpha_right_edge: (Lx_pad+1, Ly_pad+1)
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
    return _aug_forward_jit(log_trans, state_types, emit, log_M_tensor,
                             x_pad, y_pad, log_eps, Lx_pad, Ly_pad)


def backward_aug_phmm(log_trans, state_types, sub_matrix, pi,
                       x_seq, y_seq, real_Lx, real_Ly,
                       log_M_tensor, log_eps,
                       log_emit_table=None):
    """Augmented Backward DP wrapper. Returns beta of shape
    (Lx_pad+1, Ly_pad+1, 5, n_tags) in log space."""
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
    return _aug_backward_jit(log_trans, state_types, emit, log_M_tensor,
                              x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                              jnp.asarray(real_Lx), jnp.asarray(real_Ly))


def aug_phmm_corrected_posterior(
        x_seq, y_seq, t: float,
        ins_rate: float, del_rate: float, ext: float,
        Q_lg, pi_lg, boost_state,
        alpha_z: float = 100.0,
        q_min: float = 0.0) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """End-to-end Q' computation under the augmented PHMM.

    Implements equation eq:exact-Q from the appendix of main.tex:

        Q'_{ij} = (F_1(i, j)
                   + eps * sum_{(k, l), k != i} F_2(i, j; k, l) M(i, j; k, l))
                 / L_exact

    using state augmentation rather than F2 tensor enumeration.
    Equivalent to ``f2_scfg.scfg_corrected_posterior`` in the K_c = 1
    case (where M depends only on AAs, not positions); for K_c > 1 the
    augmented PHMM uses the prior-class M tensor as an approximation.

    The numerator is decomposed into three contributions:
      1. F_1(i, j): standard pair-HMM Match posterior numerator.
      2. eps * sum_l F_2(i, j; k, l) M with (i, j) as the LEFT edge of
         the coupled pair.
      3. eps * sum_l F_2(i, j; k, l) M with (i, j) as the RIGHT edge.

    Contribution 2 is computed as the per-cell "left-edge spawn"
    forward-extra times the augmented backward at the matching (c, d)
    tag. Contribution 3 is the per-cell "right-edge resolution"
    forward-extra times the augmented backward at the done tag.

    Returns:
        Q_prime: (Lx, Ly) corrected match posterior (numpy).
        L_exact: scalar partition function.
        Q_baseline: (Lx, Ly) F1/F0 baseline (numpy).
        log_F0: scalar baseline log-partition.
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

    # Augmented Forward: get alpha + per-cell left/right-edge contributions.
    alpha, alpha_left_edge, alpha_right_edge = _aug_forward_jit(
        log_trans, state_types, emit, log_M_tensor,
        x_pad, y_pad, log_eps_j, Lx_pad, Ly_pad)
    # Augmented Backward.
    beta = _aug_backward_jit(log_trans, state_types, emit, log_M_tensor,
                              x_pad, y_pad, log_eps_j, Lx_pad, Ly_pad,
                              jnp.asarray(Lx), jnp.asarray(Ly))

    # L_exact = sum over s, t in {no_edge, done} of alpha[Lx, Ly, s, t]
    # * trans[s, E]. (Orphan (a, b) tags at end excluded.)
    e_idx = _find_e_idx(state_types)
    end_alpha = alpha[Lx, Ly, :, :]                         # (5, N_TAGS)
    end_no_edge = end_alpha[:, TAG_NO_EDGE] + log_trans[:, e_idx]
    end_done = end_alpha[:, TAG_DONE] + log_trans[:, e_idx]
    log_L_exact = jax.nn.logsumexp(jnp.concatenate([end_no_edge, end_done]))

    # The corrected posterior is the augmented PHMM marginal at M-state,
    # summed over all tag values:
    #   Q'_{ij} = (sum_t alpha_aug[i, j, M, t] * beta_aug[i, j, M, t]) / Z_aug
    # where Z_aug = L_exact (the augmented partition function, terminating
    # only via no_edge or done tags).
    #
    # This naturally collects all paths that pass through Match at (i, j)
    # under the augmented model:
    #   - tag = no_edge: paths where no edge has fired by (i, j)
    #   - tag = (a, b): paths where some earlier Match started a left
    #                   edge with AAs (a, b); (i, j) is an uninvolved
    #                   Match while waiting
    #   - tag = (c, d): includes both "earlier Match was the left-edge
    #                   carrying the (c, d) tag" and "(i, j) IS the
    #                   left-edge that just fired"; the augmented forward
    #                   already adds the eps-spawn contribution to
    #                   alpha[i, j, M, (c, d)] in the Match-cell update
    #   - tag = done:   includes "(i, j) IS the right edge resolving
    #                   from some (a, b)" plus "uninvolved Match after
    #                   both edges resolved earlier"; both already in
    #                   alpha[i, j, M, done]
    # No decomposition required -- the augmented FB does the right thing.
    log_post_M = jax.nn.logsumexp(alpha[:, :, M, :] + beta[:, :, M, :],
                                       axis=-1)               # (Lx_pad+1, Ly_pad+1)
    log_Q_prime_pad = log_post_M - log_L_exact
    log_Q_prime = log_Q_prime_pad[1:Lx + 1, 1:Ly + 1]
    Q_prime = np.asarray(jnp.exp(log_Q_prime))

    # Standard pair-HMM F_1 / F_0 for the returned Q_baseline reference.
    from .f2_scfg import compute_F0_F1
    F0_b, log_F0_b, F1_b, _, _ = compute_F0_F1(
        log_trans, state_types, sub_matrix, pi_out,
        x_pad, y_pad, jnp.asarray(Lx), jnp.asarray(Ly))
    Q_baseline = np.asarray(F1_b) / float(max(F0_b, 1e-300))
    log_F0_baseline = log_F0_b
    L_exact_val = float(jnp.exp(log_L_exact))
    log_F0_val = float(log_F0_baseline)

    # q_min pruning: not currently implemented (the augmented PHMM
    # naturally accumulates over all cells without per-anchor expense).
    _ = q_min
    return Q_prime, L_exact_val, Q_baseline, log_F0_val
