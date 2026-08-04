"""Branch-kernel primitives for the phylo-ELBO forward mm-edge.

Reference: appendix par:arch-phylo-elbo, eq:arch-elbo-K-decomp
line ~2603.

The compound (x, θ) branch kernel factors into a no-jump and jump
component:

  K(x_v, θ_v | x_p, θ_p; τ)
    = β(θ_p, τ) * δ(θ_v = θ_p) * prod_n P^{(c_n, θ_p)}(x_{p,n}, x_{v,n}; τ)
    + W(θ_p, θ_v; τ) * prod_n π^{arch[c_n, θ_v]}(x_{v,n})

Under Interp 2, the θ chain has state-dependent exit rate
ρ_chain(1 − ρ[θ]). This module computes:

  β(θ, τ)          = exp(-ρ_chain (1 - ρ[θ]) τ)   [no-jump prob]
  K_field(θ_v |θ_p; τ)                            [marginal θ kernel]
  W(θ_p, θ_v; τ)   = K_field - β(θ_p, τ) δ_{θ_v = θ_p}
                                                  [≥1-jump weight]

The field-CTMC generator has entries:
  Q_field(θ_p, θ_v) = ρ_chain * ρ[θ_v]           for θ_v ≠ θ_p
                    = -ρ_chain (1 - ρ[θ_p])       for θ_v = θ_p

so that the exit rate at θ_p is ρ_chain(1 - ρ[θ_p]) and, conditional
on a jump, the destination distribution is ρ[θ_v] / (1 - ρ[θ_p]) for
θ_v ≠ θ_p (Interp 2). Note that under detailed balance the field
CTMC is reversible with stationary ρ[·], since ρ[θ_p] Q(θ_p, θ_v)
= ρ_chain ρ[θ_p] ρ[θ_v] is symmetric in (θ_p, θ_v) for the off-
diagonal.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def beta_no_jump(theta: 'int | np.ndarray',
                 rho_chain: float,
                 rho: np.ndarray,
                 tau: float) -> 'float | np.ndarray':
    """Per-branch no-jump probability β(θ, τ) = exp(-ρ_chain(1 - ρ[θ])τ).

    Vectorised over theta (accepts int or array).
    """
    return np.exp(-float(rho_chain) * (1.0 - rho[theta]) * float(tau))


def field_generator(rho_chain: float, rho: np.ndarray) -> np.ndarray:
    """Field-CTMC generator Q_field of shape (L, L).

    Off-diagonal: Q(θ_p, θ_v) = ρ_chain * ρ[θ_v]  for θ_p ≠ θ_v.
    Diagonal:     Q(θ, θ)     = -ρ_chain (1 - ρ[θ]).

    Row sums vanish. Stationary is ρ (verifies via ρ^T Q = 0).
    """
    L = int(rho.shape[0])
    Q = rho_chain * np.broadcast_to(rho[None, :], (L, L)).copy()
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def field_transition(rho_chain: float, rho: np.ndarray,
                     tau: float) -> np.ndarray:
    """Field-CTMC marginal transition K_field(θ_p, θ_v; τ) shape (L, L).

    K_field[i, j] = P(θ_v = j | θ_p = i, τ) = expm(Q_field τ)[i, j].
    Row stochastic. τ = 0 gives identity.
    """
    Q = field_generator(rho_chain, rho)
    return expm(Q * float(tau))


def jump_weight(rho_chain: float, rho: np.ndarray, tau: float) -> np.ndarray:
    """W(θ_p, θ_v; τ) = K_field(θ_p, θ_v; τ) - β(θ_p, τ) δ_{θ_v = θ_p}.

    Interpretation: the "at least one jump" mass of the branch kernel
    at the field level. Row-sums = 1 - β (per row θ_p). Off-diagonal
    entries equal K_field's off-diagonal (all jump mass). Diagonal
    entry is K_field(θ, θ) - β(θ, τ), the probability that θ was
    jumped away and eventually returned to θ.
    """
    K = field_transition(rho_chain, rho, tau)
    L = K.shape[0]
    beta_vec = np.exp(-rho_chain * (1.0 - rho) * tau)
    W = K.copy()
    W[np.arange(L), np.arange(L)] -= beta_vec
    return W
