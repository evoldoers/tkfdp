"""K-class multi-site-class extension of the simplified pipeline.

Storage convention: a stack of K(K+1)/2 unique Potts slices `H_slices`
of shape (K_pairs, A, A), each slice symmetric in the AA indices.
The class-pair (a, b) shares the same slice as (b, a), so K(K+1)/2
slices cover all class pairs.

`class_pair_idx(K, a, b)` maps class pair to the unique slice index.
Encoding: with p = min(a, b), q = max(a, b),
    idx = p * K - p * (p - 1) // 2 + (q - p).
For K = 2: idx(0,0)=0, idx(0,1)=idx(1,0)=1, idx(1,1)=2.
For K = 3: idx(0,0)=0, idx(0,1)=1, idx(0,2)=2, idx(1,1)=3, idx(1,2)=4, idx(2,2)=5.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .generator import (
    A, A2,
    build_joint_Q,
    joint_stationary,
    log_transition_matrices,
    symmetrize_eigh,
    transition_matrices,
)


def n_class_pairs(K: int) -> int:
    return K * (K + 1) // 2


def class_pair_idx_table(K: int) -> np.ndarray:
    """(K, K) int32 table mapping (a, b) -> unique slice idx."""
    tbl = np.zeros((K, K), dtype=np.int32)
    for a in range(K):
        for b in range(K):
            p, q = (a, b) if a <= b else (b, a)
            tbl[a, b] = p * K - p * (p - 1) // 2 + (q - p)
    return tbl


def project_slices_symmetric_zero_trace(H_slices: jnp.ndarray) -> jnp.ndarray:
    """Project each (A, A) slice to symmetric, zero-trace.
    H_slices: (K_pairs, A, A)."""
    H_sym = 0.5 * (H_slices + jnp.swapaxes(H_slices, -1, -2))
    diag = jnp.einsum('kii->k', H_sym) / A
    return H_sym - diag[:, None, None] * jnp.eye(A)[None, :, :]


def build_joint_Q_all(H_slices: jnp.ndarray) -> jnp.ndarray:
    """For each H slice, build the 400x400 joint generator. Returns
    shape (K_pairs, A2, A2)."""
    return jax.vmap(build_joint_Q)(H_slices)


def joint_stationary_all(H_slices: jnp.ndarray) -> jnp.ndarray:
    """(K_pairs, A2) stationary per slice."""
    return jax.vmap(joint_stationary)(H_slices)


def log_P_unique_K(H_slices: jnp.ndarray, unique_t: jnp.ndarray) -> jnp.ndarray:
    """Per-slice log P matrices at all unique cherry distances.
    Returns shape (K_pairs, n_t, A2, A2)."""
    def one_slice(H_):
        Q = build_joint_Q(H_)
        pi_j = joint_stationary(H_)
        Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
        return log_transition_matrices(unique_t, Lambda, U_sym, sqrt_pij)
    return jax.vmap(one_slice)(H_slices)


def composite_log_likelihood_K(H_slices: jnp.ndarray,
                                unique_t: jnp.ndarray,
                                obs: jnp.ndarray) -> jnp.ndarray:
    """Sum of log P over observations.

    obs columns: (t_idx, class_pair_idx, start_state, end_state).
    """
    # Build all slices' transition matrices in one vmap.
    def per_slice_P(H_):
        Q = build_joint_Q(H_)
        pi_j = joint_stationary(H_)
        Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
        return transition_matrices(unique_t, Lambda, U_sym, sqrt_pij)
    P_all = jax.vmap(per_slice_P)(H_slices)   # (K_pairs, n_t, 400, 400)

    t_idx = obs[:, 0]
    cp_idx = obs[:, 1]
    start = obs[:, 2]
    end = obs[:, 3]
    p_obs = P_all[cp_idx, t_idx, start, end]
    return jnp.sum(jnp.log(jnp.clip(p_obs, 1e-300, 1.0)))
