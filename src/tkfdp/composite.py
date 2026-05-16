"""Composite log-likelihood under the F81 400-state joint generator.

Cherry observations are gathered into:
- unique_t: (K,) array of unique cherry distances
- obs: (M, 3) int32 [t_idx, start_state, end_state]

Per main.tex \S2 (post-2026-05-08 reparameterization), the generator
takes an explicit pi (per-class stationary) and eta_pair (per-site
rate multipliers); both default to LG08 / unit values for backwards
compatibility with the simplified pipeline (K=1, eta=1).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .generator import (
    build_joint_Q,
    joint_stationary,
    symmetrize_eigh,
    transition_matrices,
)
from .lg08 import PI_LG08_J, S_LG08_F81_J


def composite_log_likelihood(H: jnp.ndarray,
                             unique_t: jnp.ndarray,
                             obs: jnp.ndarray,
                             pi: jnp.ndarray = PI_LG08_J,
                             S: jnp.ndarray = S_LG08_F81_J,
                             eta_pair: tuple[float, float] = (1.0, 1.0)) -> jnp.ndarray:
    """Sum of log P_(start, end)(t; H, pi, S, eta_pair) over all observations.

    obs columns: (t_idx, start_state, end_state).
    """
    Q = build_joint_Q(H, pi=pi, S=S, eta_pair=eta_pair)
    pi_j = joint_stationary(H, pi=pi)
    Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
    P_all = transition_matrices(unique_t, Lambda, U_sym, sqrt_pij)

    t_idx = obs[:, 0]
    start = obs[:, 1]
    end = obs[:, 2]
    p_obs = P_all[t_idx, start, end]
    log_p = jnp.log(jnp.clip(p_obs, 1e-300, 1.0))
    return jnp.sum(log_p)


def project_to_symmetric_zero_trace(H: jnp.ndarray) -> jnp.ndarray:
    """Project H onto the symmetric, zero-trace subspace.
    The zero-trace pinning removes the indistinguishable add-a-constant-to-diag
    invariance discussed in implementation_notes.md Section 8.
    """
    H_sym = 0.5 * (H + H.T)
    return H_sym - jnp.trace(H_sym) / H_sym.shape[0] * jnp.eye(H_sym.shape[0])
