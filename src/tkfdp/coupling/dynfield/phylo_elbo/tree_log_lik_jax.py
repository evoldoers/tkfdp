"""Level-by-level JIT'd tree log-likelihood (Milestone 3).

Reads a `PaddedTree` (from tree_padded.py) and computes the marginal
log-likelihood via the phylo-ELBO forward sweep, using the JAX
primitives from mm_clv_jax.py.

The forward sweep is level-by-level:
  Level 0:  leaf CLVs (from leaf_obs).
  Level l >= 1:  for each slot (2 children at prev level):
    left_msg  = mm_edge_jax(left_clv,  P_left,  beta_left,  W_left)
    right_msg = mm_edge_jax(right_clv, P_right, beta_right, W_right)
    combined  = mm_combine_jax(left_msg, right_msg, pi_arch)
    -- for phantom identity slots, use the left child's CLV directly.
  Level D root:  mm_mass_one_jax(root_clv, rho).

Because different PaddedTree buckets (N_bucket, D_bucket) have
different level counts and different n_slots per level, the JIT
compiles ONE function per bucket. Within a bucket, all trees share
the shape and can be vmapped across batch.

Milestone 3 provides the single-tree function; Milestone 4 adds
vmap-over-batch batching.
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from .mm_clv_jax import (
    leaf_clv_jax, mm_combine_jax, mm_edge_jax, mm_mass_one_jax,
    mm_mass_two_jax,
)


# ---------------------------------------------------------------------------
# Edge and substitution kernel builders (JAX).
# ---------------------------------------------------------------------------


def edge_kernel_from_tau(rho: jnp.ndarray, rho_chain: float,
                             tau: float) -> 'tuple[jnp.ndarray, jnp.ndarray]':
    """Compute (beta, W) at branch length tau under Interp-2 F81-on-DP
    field CTMC.

    beta[l]  = exp(-rho_chain * (1 - rho[l]) * tau)
    W[l, l'] = P_F(l -> l'; tau) - delta * beta
    where P_F(l -> l'; tau) = exp(-rho_chain * tau) * delta_{l,l'}
                             + (1 - exp(-rho_chain * tau)) * rho[l']
    (Interp 1 on the marginal; combined with Interp 2 exit rate
    for beta -- matches mm_clv.py's convention.)
    """
    L = rho.shape[0]
    beta = jnp.exp(-rho_chain * (1.0 - rho) * tau)  # (L,)
    et = jnp.exp(-rho_chain * tau)                    # scalar
    # P_F[l, l'] = et * delta + (1 - et) * rho[l']
    PF = (1.0 - et) * rho[None, :] + et * jnp.eye(L)  # (L, L)
    beta_diag = jnp.diag(beta)
    W = PF - beta_diag
    return beta, W


def subst_kernel_from_tau(xi: jnp.ndarray, U: jnp.ndarray,
                             Uinv: jnp.ndarray, classes: jnp.ndarray,
                             tau: float) -> jnp.ndarray:
    """Per-site substitution kernel P_sub[m, L, A, A] at branch length tau.

    xi:      (K_c, L, A)     eigenvalues per (class, theta)
    U:       (K_c, L, A, A)  right eigenvectors
    Uinv:    (K_c, L, A, A)  inverse eigenvectors (left eigenvectors)
    classes: (m,)            class index per site
    """
    # Gather per-site: (m, L, A) eigenvalues + (m, L, A, A) eigenvectors
    xi_m = xi[classes]                  # (m, L, A)
    U_m = U[classes]                    # (m, L, A, A)
    Uinv_m = Uinv[classes]              # (m, L, A, A)
    # exp(xi * tau)
    exp_xt = jnp.exp(xi_m * tau)        # (m, L, A)
    # P = U @ diag(exp_xt) @ Uinv
    # = einsum('mlab,mlb,mlbc->mlac', U_m, exp_xt, Uinv_m)
    P = jnp.einsum('mlab,mlb,mlbc->mlac', U_m, exp_xt, Uinv_m)
    return P


# ---------------------------------------------------------------------------
# Per-level building block (identity-lift-aware).
# ---------------------------------------------------------------------------


def _lookup_clv(clvs_prev: dict, idx: jnp.ndarray) -> dict:
    """Gather CLVs at positions `idx` from clvs_prev (batched over
    the slot dimension)."""
    return {
        'r': clvs_prev['r'][idx],       # (n_slots, L)
        's': clvs_prev['s'][idx],       # (n_slots, L)
        'A': clvs_prev['A'][idx],       # (n_slots, m, L, A)
    }


def _mm_edge_vmap(F_batch: dict, P_sub_batch: jnp.ndarray,
                    beta_batch: jnp.ndarray, W_batch: jnp.ndarray) -> dict:
    """vmap of mm_edge_jax over a leading `n_slots` batch axis.

    F_batch['r']: (n_slots, L); F_batch['A']: (n_slots, m, L, A);
    P_sub_batch: (n_slots, m, L, A, A); beta_batch: (n_slots, L);
    W_batch: (n_slots, L, L).
    """
    return jax.vmap(mm_edge_jax, in_axes=(
        {'r': 0, 's': 0, 'A': 0}, 0, 0, 0))(
        F_batch, P_sub_batch, beta_batch, W_batch)


def _mm_combine_vmap(F_L_batch: dict, F_R_batch: dict,
                        pi_arch: jnp.ndarray) -> dict:
    """vmap of mm_combine_jax over a leading `n_slots` batch axis.
    pi_arch (m, L, A) broadcasts across the batch."""
    return jax.vmap(mm_combine_jax, in_axes=(
        {'r': 0, 's': 0, 'A': 0}, {'r': 0, 's': 0, 'A': 0}, None))(
        F_L_batch, F_R_batch, pi_arch)


def _propagate_level(clvs_prev: dict,
                        child_pos: jnp.ndarray,       # (n_slots, 2) int
                        child_branch: jnp.ndarray,    # (n_slots, 2) float
                        is_identity: jnp.ndarray,     # (n_slots,) float 1/0
                        pi_arch: jnp.ndarray,         # (m, L, A)
                        rho: jnp.ndarray,             # (L,)
                        rho_chain: float,
                        xi: jnp.ndarray, U: jnp.ndarray, Uinv: jnp.ndarray,
                        classes: jnp.ndarray) -> dict:
    """Compute CLVs at one level from clvs_prev.

    For identity-lift slots (is_identity == 1): pass the left child's
    CLV through unchanged (branches are 0-length by construction).
    For real combine slots: mm_edge each child then mm_combine.
    """
    left_idx = child_pos[:, 0]              # (n_slots,)
    right_idx = child_pos[:, 1]
    left_tau = child_branch[:, 0]           # (n_slots,)
    right_tau = child_branch[:, 1]

    # Gather child CLVs.
    left_child = _lookup_clv(clvs_prev, left_idx)
    right_child = _lookup_clv(clvs_prev, right_idx)

    # Build per-slot P_sub, beta, W for each side.
    def _kernels_at_tau(tau_scalar):
        beta, W = edge_kernel_from_tau(rho, rho_chain, tau_scalar)
        P_sub = subst_kernel_from_tau(xi, U, Uinv, classes, tau_scalar)
        return beta, W, P_sub

    beta_L, W_L, P_L = jax.vmap(_kernels_at_tau)(left_tau)
    beta_R, W_R, P_R = jax.vmap(_kernels_at_tau)(right_tau)

    # mm_edge each side.
    msg_L = _mm_edge_vmap(left_child, P_L, beta_L, W_L)
    msg_R = _mm_edge_vmap(right_child, P_R, beta_R, W_R)

    # mm_combine.
    combined = _mm_combine_vmap(msg_L, msg_R, pi_arch)

    # Blend: for identity slots, pass left_child through directly (branches
    # were 0-length, both children equal, no combine).
    is_id = is_identity[:, None]                    # (n_slots, 1) for r, s
    is_id_A = is_identity[:, None, None, None]      # (n_slots, 1, 1, 1) for A
    out = {
        'r': jnp.where(is_id > 0.5, left_child['r'], combined['r']),
        's': jnp.where(is_id > 0.5, left_child['s'], combined['s']),
        'A': jnp.where(is_id_A > 0.5, left_child['A'], combined['A']),
    }
    return out


# ---------------------------------------------------------------------------
# Full tree forward pass (single tree).
# ---------------------------------------------------------------------------


def tree_log_lik_jax(
    leaf_obs: jnp.ndarray,               # (N_bucket, m) int32; -1=gap
    leaf_mask: jnp.ndarray,              # (N_bucket,) float64
    child_pos_by_level: 'tuple[jnp.ndarray, ...]',
    child_branch_by_level: 'tuple[jnp.ndarray, ...]',
    is_identity_by_level: 'tuple[jnp.ndarray, ...]',
    root_slot: int,
    pi_arch: jnp.ndarray,                # (m, L, A)
    rho: jnp.ndarray,                    # (L,)
    rho_chain: float,
    xi: jnp.ndarray,                     # (K_c, L, A)
    U: jnp.ndarray,                      # (K_c, L, A, A)
    Uinv: jnp.ndarray,                   # (K_c, L, A, A)
    classes: jnp.ndarray,                # (m,) int
) -> jnp.ndarray:
    """Single-tree forward tree log-lik under moment-matching phylo-ELBO.

    Bucket shape is baked in via the length of *_by_level tuples and the
    shapes of leaf_obs etc. To batch across trees at the same bucket,
    call jax.vmap on this function (M4).
    """
    # Level 0: leaves.
    leaf_clvs = jax.vmap(leaf_clv_jax, in_axes=(0, None))(
        leaf_obs, pi_arch)
    # Structure of leaf_clvs: 'r' (N, L), 's' (N, L), 'A' (N, m, L, A).

    clvs = leaf_clvs
    for level in range(len(child_pos_by_level)):
        clvs = _propagate_level(
            clvs,
            child_pos_by_level[level],
            child_branch_by_level[level],
            is_identity_by_level[level],
            pi_arch, rho, rho_chain, xi, U, Uinv, classes,
        )

    # Root: mm_mass_one at root_slot.
    root_clv = {
        'r': clvs['r'][root_slot],
        's': clvs['s'][root_slot],
        'A': clvs['A'][root_slot],
    }
    mass = mm_mass_one_jax(root_clv, rho)
    return jnp.log(jnp.maximum(mass, 1e-300))


# ---------------------------------------------------------------------------
# Helpers to build eigendecompositions and pack padded_tree tensors.
# ---------------------------------------------------------------------------


def gtr_eigendecomp_batch_jax(pi_arch: jnp.ndarray,
                                 S: jnp.ndarray
                                 ) -> 'tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]':
    """Per-(K_a, L) eigendecomposition of GTR Q^{c, theta} matrices.

    Q_{ij}(c, theta) = S_{ij} pi_arch[c, theta, j]  for i != j
                     = -sum_{j!=i} Q_{ij}           for i == j

    Args:
      pi_arch:   (K_c, L, A)
      S:         (A, A) symmetric zero-diagonal exchangeability
    Returns:
      xi:   (K_c, L, A)   eigenvalues (real; sorted descending)
      U:    (K_c, L, A, A) right eigenvectors (columns)
      Uinv: (K_c, L, A, A) inverse
    """
    K_c, L, A = pi_arch.shape
    # Build Q per (c, l).
    Q_off = S[None, None, :, :] * pi_arch[:, :, None, :]  # (K_c, L, A, A)
    Q = Q_off - jnp.eye(A)[None, None, :, :] * Q_off.sum(axis=-1)[..., None]
    # Symmetrise via similarity to a symmetric matrix for numerical stability,
    # then diagonalise: Q_sym = D^{1/2} Q D^{-1/2} where D = diag(pi_arch).
    # But eig on non-symmetric matrix works fine for these small dims.
    xi, U = jnp.linalg.eig(Q)  # complex; take real part (Q is diagonalisable)
    xi = jnp.real(xi)
    U = jnp.real(U)
    Uinv = jnp.linalg.inv(U)
    return xi, U, Uinv


def padded_tree_to_jax(pt) -> dict:
    """Convert a PaddedTree (numpy) into JAX arrays plus derived
    per-level is_identity flag.

    An identity-lift slot is one whose two child positions coincide
    AND whose branch lengths are both zero (built as phantom lift by
    tree_padded.py).
    """
    n_levels = pt.D_bucket
    child_pos_j = tuple(jnp.asarray(pt.child_pos[l], dtype=jnp.int32)
                            for l in range(n_levels))
    child_branch_j = tuple(jnp.asarray(pt.child_branch[l], dtype=jnp.float64)
                                for l in range(n_levels))
    is_identity_j = []
    for l in range(n_levels):
        cp = pt.child_pos[l]                    # (n_slots, 2)
        cb = pt.child_branch[l]                 # (n_slots, 2)
        # identity: both children same slot AND both branches 0.
        is_id = ((cp[:, 0] == cp[:, 1]) & (cb[:, 0] == 0.0)
                    & (cb[:, 1] == 0.0)).astype('float64')
        # But the phantom pad slots (slot_mask=0) also have (0, 0, 0, 0)
        # which trips the identity check; that's fine because the outputs
        # are dead-end (not referenced downstream).
        is_identity_j.append(jnp.asarray(is_id))
    is_identity_j = tuple(is_identity_j)

    return {
        'leaf_obs': jnp.asarray(pt.leaf_obs, dtype=jnp.int32),
        'leaf_mask': jnp.asarray(pt.leaf_mask, dtype=jnp.float64),
        'child_pos_by_level': child_pos_j,
        'child_branch_by_level': child_branch_j,
        'is_identity_by_level': is_identity_j,
        'root_slot': int(pt.root_slot),
    }
