"""TKF-DP postprocessing for pairwise residue alignment posteriors.

Implements the first-order mean-field correction described in
Section "Pairwise Alignment Postprocessing" of main.tex:

    Q'_{ij} ∝ Q_{ij} * exp(boost_{ij})

where

    boost_{ij} = eps * sum_{(i',j') != (i,j)} Q_{i'j'} * (M(i, j; i', j'; t) - 1)

and M is the class-marginalized Potts boost tensor (eq. M-marginal in the
paper). The output is a per-residue-pair multiplicative factor on the
Match-state log-emission table; renormalization to the pair-HMM marginal
constraints is produced by a fresh forward--backward pass downstream.

Public API:
  build_per_class_match_emit(state, t, S=None) -> (K_c, A, A)
      Per-class single-site joint emission P(a, b | c; t) from the
      trained TKF-DP state at branch length t.
  build_per_classpair_joint_emit(state, t, S=None) -> (K_c, K_c, A, A, A, A)
      Per-class-pair joint emission tensor
      P_joint((a, a') -> (b, b'); t, H_{c,c'}) for every ordered class-pair
      (c, c'). For c <= c' the canonical Potts atom + side potentials are
      used; for c > c' the (a <-> a', b <-> b') swap of the canonical entry
      is materialized so the tensor is symmetric under the joint swap.
  class_posteriors_from_baseline(per_class_emit, x_seq, y_seq, pi_class)
      -> (L_X, L_Y, K_c)  gamma_{ij}(c) of eq. gamma in the paper.
  pair_emission_boost(Q_baseline, gamma, per_class_emit, joint_per_cp,
                      x_seq, y_seq, alpha_z, L_aln=None)
      -> (L_X, L_Y) log-boost  to add to the Match-state log-emission
      table before re-running the Pair HMM forward--backward.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from .generator import (
    A,
    A2,
    PI_LG08_J,
    S_LG08_J,
    build_joint_Q_pair,
    joint_stationary_pair,
    symmetrize_eigh,
    transition_matrices,
)
from .potts_dp import (
    PottsDPState,
    canonical_pair_idx_table,
    canonical_pair_is_diag,
)


# --- Per-class single-site emissions ---------------------------------------

def _f81_unnormalized_Q(pi: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """F81-form rate matrix S(off-diag) * pi, UN-normalized.

    Matches the per-class single-site Q used in build_joint_Q_pair, which
    constructs the joint generator from the unnormalized F81 (per-class
    rate scaling is supplied separately via eta). Using the same
    convention here is what makes M(a, b, a', b'; t) -> 1 in the H = 0
    limit (joint Q factorizes as a Kronecker sum, exp factorizes as a
    Kronecker product, so per-class single-site emission marginalizes
    site 2 of the joint exactly).
    """
    Q = (S - jnp.diag(jnp.diag(S))) * pi[None, :]
    Q = Q - jnp.diag(Q.sum(axis=1))
    return Q


def build_per_class_match_emit(pi_class: np.ndarray, t: float,
                                  S: Optional[np.ndarray] = None
                                  ) -> jnp.ndarray:
    """Per-class single-site joint emission P(a, b | c; t) at branch length t.

    Returns a (K_c, A, A) tensor where entry [c, a, b] is the probability
    that an ancestral residue of class c, type a, evolves to type b over
    branch length t. This matches the emission used to construct the
    baseline Pair HMM Match-state emissions in the no-Potts mixture model.
    """
    pi_j = jnp.asarray(pi_class)
    S_j = jnp.asarray(S_LG08_J if S is None else S)
    t_j = jnp.asarray(t)

    def per_class(pi_c):
        Q_c = _f81_unnormalized_Q(pi_c, S_j)
        return jsl.expm(Q_c * t_j)

    return jax.vmap(per_class)(pi_j)


# --- Per-class-pair joint emissions (with Potts coupling + side potentials)


def _per_classpair_canonical_joint(pi_a: jnp.ndarray, pi_b: jnp.ndarray,
                                       H: jnp.ndarray, t: float,
                                       S: jnp.ndarray,
                                       h_a: Optional[jnp.ndarray] = None,
                                       h_b: Optional[jnp.ndarray] = None
                                       ) -> jnp.ndarray:
    """exp(Q_joint * t) for a single ordered class-pair (a, b), with Potts
    atom H and optional side potentials. Returns (A^2, A^2) in the same
    state-ordering convention as build_joint_Q_pair (state index = a*A + b).
    """
    Q = build_joint_Q_pair(H, pi_a, pi_b, S=S, h_a=h_a, h_b=h_b)
    pi_j = joint_stationary_pair(H, pi_a, pi_b, h_a=h_a, h_b=h_b)
    Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
    P = transition_matrices(jnp.asarray([t]), Lambda, U_sym, sqrt_pij)[0]
    return P  # (A^2, A^2)


def build_per_classpair_joint_emit(state, t: float,
                                      S: Optional[np.ndarray] = None
                                      ) -> jnp.ndarray:
    """Per-class-pair joint emission tensor at branch length t.

    Returns a (K_c, K_c, A, A, A, A) tensor where
        P_joint[c, c', a, a', b, b'] = P((a, a') -> (b, b'); t, H_{c, c'})
    is the joint-substitution probability that an ancestral *unordered*
    column-pair of classes (c, c') with state (a at site 1 of c, a' at site
    1 of c') evolves to (b, b') over branch length t under the trained
    Potts atom assigned to canonical class-pair {c, c'} + the matching
    side potentials.

    For ordered (c, c') with c > c' we use the canonical (c', c) joint and
    swap site indices so the tensor is symmetric under simultaneous swap
    of (c, c') and (a, a') and (b, b'); this is consistent with the
    canonical-pair indexing used everywhere else in the codebase.
    """
    K_c = state.K_c
    pi_class = jnp.asarray(state.pi_class)
    atoms = jnp.asarray(state.potts_dp.atoms)             # (K_H, A, A)
    assignments = state.potts_dp.assignments              # (K_c, K_c) int
    h_pairs = state.potts_dp.h_pairs                       # may be None
    cp_idx_np, cp_swap_np = canonical_pair_idx_table(K_c)
    S_j = jnp.asarray(S_LG08_J if S is None else S)
    use_h = h_pairs is not None
    if use_h:
        h_pairs_j = jnp.asarray(h_pairs)

    P_table = np.zeros((K_c, K_c, A, A, A, A), dtype=np.float64)
    for c in range(K_c):
        for cp in range(K_c):
            atom_idx = int(assignments[c, cp])
            H = atoms[atom_idx]
            if use_h:
                k_can = int(cp_idx_np[c, cp])
                swap = int(cp_swap_np[c, cp])
                h_a = h_pairs_j[k_can, swap]
                h_b = h_pairs_j[k_can, 1 - swap]
            else:
                h_a = h_b = None
            P_flat = _per_classpair_canonical_joint(
                pi_class[c], pi_class[cp], H, t, S_j, h_a=h_a, h_b=h_b
            )                                              # (A^2, A^2)
            P_6d = np.asarray(P_flat).reshape(A, A, A, A)  # (a, a', b, b')
            # Note: build_joint_Q_pair encodes site-1 = a, site-2 = a' and
            # the ordered ROW state index is a*A + a'; column state index
            # is b*A + b'. So P_flat[a*A + a', b*A + b'] = P((a, a')->(b, b')).
            P_table[c, cp] = P_6d
    return jnp.asarray(P_table)


# --- Class posteriors gamma_{ij}(c) -----------------------------------------


def class_posteriors_from_baseline(per_class_emit: jnp.ndarray,
                                       x_seq: np.ndarray,
                                       y_seq: np.ndarray,
                                       pi_class: np.ndarray) -> jnp.ndarray:
    """gamma_{ij}(c) = pi_c * P(X_i, Y_j | c; t) / sum_c' pi_c' * P(...).

    Args:
      per_class_emit: (K_c, A, A) from build_per_class_match_emit.
      x_seq: (L_X,) int array of amino-acid indices.
      y_seq: (L_Y,) int array of amino-acid indices.
      pi_class: (K_c,) per-class mixing weights (sum to 1).

    Returns: (L_X, L_Y, K_c) per-(i,j) class posterior.
    """
    pi_c = jnp.asarray(pi_class)
    e = per_class_emit                                     # (K_c, A, A)
    # Gather per-(i, j, c) emission e[c, X_i, Y_j], shape (L_X, L_Y, K_c).
    e_ij = e[:, x_seq, :][:, :, y_seq]                     # (K_c, L_X, L_Y)
    e_ij = jnp.transpose(e_ij, (1, 2, 0))                  # (L_X, L_Y, K_c)
    weighted = pi_c[None, None, :] * e_ij
    norm = jnp.sum(weighted, axis=-1, keepdims=True)
    return weighted / jnp.clip(norm, 1e-300, None)


# --- The boost field boost_{ij} ---------------------------------------------


def pair_emission_boost(Q_baseline: jnp.ndarray,
                          gamma: jnp.ndarray,
                          per_class_emit: jnp.ndarray,
                          joint_per_cp: jnp.ndarray,
                          x_seq: np.ndarray,
                          y_seq: np.ndarray,
                          alpha_z: float,
                          L_aln: Optional[float] = None) -> jnp.ndarray:
    """Compute the per-(i, j) log-boost to add to the Match-state log-emission
    table of the Pair HMM, implementing equation eq:Q-prime of main.tex.

    boost_{ij} = log( 1 + eps * sum_{(i',j')} Q_{i'j'} * (M(i,j;i',j') - 1) )

    where eps = 1 / (alpha_z + L_aln - 1) and M is the class-marginalized
    Potts boost tensor (eq:M-marginal). The implementation never
    materializes the (L_X, L_Y, L_X, L_Y) tensor M; instead we factor the
    inner sum into an O(K_c^2 A^4) reduction across the sequence pair plus
    O(L_X^2 L_Y^2 K_c^2 A^2) at the per-(i,j) outer level.

    Args:
      Q_baseline: (L_X, L_Y) baseline match-state posterior from the
          forward--backward of the no-Potts Pair HMM. Linear (not log).
      gamma: (L_X, L_Y, K_c) class posteriors from
          class_posteriors_from_baseline.
      per_class_emit: (K_c, A, A) per-class single-site emission.
      joint_per_cp: (K_c, K_c, A, A, A, A) per-class-pair joint emission.
      x_seq, y_seq: (L_X,), (L_Y,) sequence index arrays.
      alpha_z: Ewens partition concentration on the column partition.
      L_aln: optional estimated number of aligned columns (defaults to
          sum of Q_baseline, the expected aligned-column count under the
          baseline marginals).

    Returns: (L_X, L_Y) log-boost. Add this to the Pair HMM Match-state
    log-emission table and re-run forward--backward to obtain Q'.
    """
    Q = jnp.asarray(Q_baseline)
    K_c = gamma.shape[-1]
    L_X = x_seq.shape[0]; L_Y = y_seq.shape[0]
    L_aln = float(jnp.sum(Q)) if L_aln is None else float(L_aln)
    eps = 1.0 / (alpha_z + max(L_aln - 1.0, 0.0))

    # Denominator of M (per-column denominator of eq:M-marginal):
    #   Denom(i, j) = sum_c gamma[i, j, c] * P(X_i, Y_j | c; t)
    e_ij = per_class_emit[:, x_seq, :][:, :, y_seq]        # (K_c, L_X, L_Y)
    e_ij = jnp.transpose(e_ij, (1, 2, 0))                  # (L_X, L_Y, K_c)
    Denom = jnp.sum(gamma * e_ij, axis=-1)                 # (L_X, L_Y)
    Denom = jnp.clip(Denom, 1e-300, None)

    # Build the per-(c', a', b') accumulator over the candidate-partner
    # sequence side:
    #   W[c', a', b'] = sum_{i',j'} (Q[i',j'] / Denom[i',j']) * gamma[i',j',c']
    #                 * 1{X_{i'} = a'} * 1{Y_{j'} = b'}
    R = Q / Denom                                          # (L_X, L_Y)
    Xoh = jax.nn.one_hot(jnp.asarray(x_seq), A)            # (L_X, A)
    Yoh = jax.nn.one_hot(jnp.asarray(y_seq), A)            # (L_Y, A)
    W = jnp.einsum('ijc,ij,ia,jb->cab',
                     gamma, R, Xoh, Yoh)                   # (K_c, A, A)

    # Inner contraction with the class-pair joint emission:
    #   Z[c, a, b] = sum_{d, p, q} J[c, d, a, p, b, q] * W[d, p, q]
    # where J = joint_per_cp has axes
    #   (first class, second class, site-1 anc, site-2 anc,
    #    site-1 desc, site-2 desc)
    # so we contract W's (partner class, partner X-AA = site-2 anc,
    # partner Y-AA = site-2 desc) against J's axes (1, 3, 5). The output
    # Z[c, a, b] is indexed by (first class, site-1 anc = X of fixed
    # column, site-1 desc = Y of fixed column).
    Z = jnp.einsum('cdapbq,dpq->cab', joint_per_cp, W)     # (K_c, A, A)

    # Outer contraction at the (i, j) level. Numerator of the inner sum:
    #   Numer(i, j) = sum_c gamma[i, j, c] * Z[c, X_i, Y_j]
    # so the inner sum_{(i', j')} Q_{i'j'} M(i, j; i', j') equals
    #   Numer(i, j) / Denom(i, j).
    Z_ij = Z[:, x_seq, :][:, :, y_seq]                     # (K_c, L_X, L_Y)
    Z_ij = jnp.transpose(Z_ij, (1, 2, 0))                  # (L_X, L_Y, K_c)
    Numer = jnp.sum(gamma * Z_ij, axis=-1)                 # (L_X, L_Y)
    inner_sum_full = Numer / Denom                         # (L_X, L_Y)

    # Subtract the diagonal self-contribution Q[i, j] * M(i, j; i, j) — the
    # exclusion (i', j') != (i, j) in the formula. M(i, j; i, j) at the
    # diagonal would re-use the same (i, j) twice, which is not a
    # well-defined unordered pair; the cleanest treatment is to exclude it.
    # For simplicity and because the contribution is O(1/L^2) at most, we
    # subtract the linearized estimate Q[i,j] * (M_self - 1) where M_self is
    # approximated by 1 (i.e., we just subtract Q[i, j] from inner_sum and
    # the offset below).
    inner_sum_excl = inner_sum_full - Q                    # remove self-Q

    # Subtract the "1" baseline:
    #   sum_{(i',j') != (i,j)} Q_{i'j'} * (M - 1)
    #     = inner_sum_excl - (sum Q - Q[i, j])
    total_Q_excl = jnp.sum(Q) - Q                          # (L_X, L_Y)
    delta = inner_sum_excl - total_Q_excl                  # (L_X, L_Y)

    # Use the log(1 + eps * delta) form to handle large delta gracefully
    # (saturates rather than blowing up under strong individual edges).
    return jnp.log1p(eps * delta)


# --- Convenience: the full pipeline ----------------------------------------


def correct_pair_posterior(Q_baseline: np.ndarray,
                              x_seq: np.ndarray,
                              y_seq: np.ndarray,
                              t: float,
                              state,
                              alpha_z: float,
                              S: Optional[np.ndarray] = None,
                              return_boost: bool = False):
    """Convenience wrapper: takes a baseline Q matrix and a trained TKF-DP
    state, returns either (Q' renormalized in the Match emission sense)
    or the log-boost field for caller-driven re-run of forward--backward.

    Returns:
        log_boost: (L_X, L_Y) — add to the Pair HMM Match-state log-emit
            table and re-run forward--backward to get the renormalized Q'.

    If return_boost=False, returns Q_baseline * exp(log_boost) WITHOUT
    re-renormalizing through forward--backward — useful for quick tests
    and diagnostics; not the production path (which should renormalize via
    the Pair HMM marginal constraints).
    """
    pi_class_j = jnp.asarray(state.pi_class)
    pi_c = jnp.full(state.K_c, 1.0 / state.K_c)            # uniform K-prior

    per_class_emit = build_per_class_match_emit(state.pi_class, t, S=S)
    joint_per_cp = build_per_classpair_joint_emit(state, t, S=S)

    gamma = class_posteriors_from_baseline(per_class_emit, x_seq, y_seq,
                                                pi_c)
    log_boost = pair_emission_boost(Q_baseline, gamma, per_class_emit,
                                       joint_per_cp, x_seq, y_seq,
                                       alpha_z=alpha_z)
    if return_boost:
        return log_boost
    return jnp.asarray(Q_baseline) * jnp.exp(log_boost)
