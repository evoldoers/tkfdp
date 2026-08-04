"""F81-on-DP field-selector CTMC primitives for the dynamic-latent-field
coupling variant.

The latent field theta evolves on the tree as an F81-shaped CTMC over the
DP stick-breaking atoms truncated at L_max:

    Q[theta -> theta'] = rho_chain * rho_{theta'}     for theta != theta'
    Q[theta -> theta]  = -rho_chain * (1 - rho_theta)                    (1)

where `rho_chain` is the field-chain rate multiplier (default 1.0; see
state.DynamicFieldState docstring). Equivalently
`Q = rho_chain * (1 rho^T - I)`. The spectrum is
`{0, -rho_chain, -rho_chain, ..., -rho_chain}` (one zero eigenvalue with
right eigenvector `1`; the other L-1 eigenvalues are -rho_chain in the
rank-(L-1) subspace orthogonal to rho). Hence the closed-form transition
kernel is independent of the source state's identity, decaying uniformly
at rate rho_chain:

    P[theta -> theta'; t] = rho_{theta'}
                            + (delta_{theta theta'} - rho_{theta'}) * exp(-rho_chain * t)
                                                                          (2)

Detailed balance: rho_theta * P[theta -> theta'; t] = rho_{theta'} *
P[theta' -> theta; t] (proved by direct substitution into (2); the F81
form is the unique reversible chain with constant-in-source-state
off-diagonal). Stationary partition over realised tree-labelings is the
Ewens / CRP-on-tree.

Strong lumpability: the chain is *exactly* lumpable onto any partition
of the state space (in particular, the "occupied atoms plus tail" cut
used at inference time when an MCMC state has assigned the field at
some clusters and the rest live in the tail). The proof: row sums into
each block are equal across all sources within the block since the
off-diagonal Q[theta -> theta'] depends only on the destination
rho_{theta'} (Burke-Rosenblatt necessary condition).

Module surface:

  - `stick_breaking_to_rho(betas)`: convert TSB Beta(1, alpha) draws to a
    normalised (L,) stick-breaking weight vector.
  - `f81_dp_generator(rho)`: build the (L, L) F81-on-DP generator (1).
  - `f81_dp_transition(rho, t)`: closed-form (L, L) transition matrix (2).
  - `f81_dp_transitions_vmap(rho, ts)`: vectorised over a (n_t,) vector
    of branch lengths; returns (n_t, L, L).
  - `lumped_rho(rho, occupied)`: rho on the "occupied + tail" lumped
    state space.
"""
from __future__ import annotations

import numpy as np


def stick_breaking_to_rho(betas: np.ndarray) -> np.ndarray:
    """Truncated-stick-breaking weights for a DP atom set.

    Given (L - 1,) Beta(1, alpha) draws, returns the (L,) normalised
    stick-breaking weight vector

        rho_l = beta_l * prod_{m < l} (1 - beta_m)    for l < L - 1
        rho_{L-1} = prod_{m < L-1} (1 - beta_m)       (tail catches the rest)

    so that sum_l rho_l = 1 exactly.
    """
    betas = np.asarray(betas, dtype=np.float64)
    L = betas.shape[0] + 1
    rho = np.empty(L, dtype=np.float64)
    remaining = 1.0
    for i in range(L - 1):
        rho[i] = remaining * betas[i]
        remaining *= 1.0 - betas[i]
    rho[L - 1] = remaining
    return rho


def f81_dp_generator(rho: np.ndarray, rho_chain: float = 1.0) -> np.ndarray:
    """(L, L) F81-on-DP generator Q[theta -> theta'] = rho_chain * rho_{theta'}.

    Diagonal chosen so each row sums to 0; reversible w.r.t. rho.
    """
    rho = np.asarray(rho, dtype=np.float64)
    L = rho.shape[0]
    Q = np.tile(rho[None, :], (L, 1)) * float(rho_chain)
    np.fill_diagonal(Q, 0.0)
    Q -= np.diag(Q.sum(axis=1))
    return Q


def f81_dp_transition(rho: np.ndarray, t: float,
                       rho_chain: float = 1.0) -> np.ndarray:
    """Closed-form (L, L) F81-on-DP transition matrix at time t.

        P[theta -> theta'; t] = rho_{theta'}
                                + (delta_{theta theta'} - rho_{theta'}) * exp(-rho_chain * t)

    Source-state-independent decay at uniform rate rho_chain; matches
    expm(Q * t) exactly to machine precision (verified in
    tests/dynfield/test_math_precompute.py::test_f81_dp_detailed_balance
    at rho_chain=1).
    """
    rho = np.asarray(rho, dtype=np.float64)
    L = rho.shape[0]
    decay = float(np.exp(-float(rho_chain) * float(t)))
    return rho[None, :] + (np.eye(L) - rho[None, :]) * decay


def f81_dp_transitions_vmap(rho: np.ndarray, ts: np.ndarray,
                              rho_chain: float = 1.0) -> np.ndarray:
    """Vectorised closed-form transitions at a (n_t,) array of branch
    lengths. Returns (n_t, L, L) without building L^2 worth of expm calls.
    """
    rho = np.asarray(rho, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    L = rho.shape[0]
    decay = np.exp(-float(rho_chain) * ts)              # (n_t,)
    eye_minus_rho = np.eye(L) - rho[None, :]            # (L, L)
    out = (rho[None, None, :]
           + eye_minus_rho[None, :, :] * decay[:, None, None])
    return out


def lumped_rho(rho: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    """Lumped stationary on the "occupied + tail" reduced state space.

    Given (L,) rho and a boolean (L,) mask `occupied`, returns a vector
    of length |occupied| + 1: the occupied-atom masses in order followed
    by their tail-sum.
    """
    rho = np.asarray(rho, dtype=np.float64)
    occupied = np.asarray(occupied, dtype=bool)
    occ_mass = rho[occupied]
    tail_mass = float(rho[~occupied].sum())
    return np.concatenate([occ_mass, [tail_mass]])


def no_jump_prob(t: float, halve: bool = False,
                  rho_chain: float = 1.0) -> float:
    """No-jump probability on a branch of length t under the F81-on-DP
    field chain (carrier-vs-stationary mixture interpretation; see
    `coupling.dynfield.state.DynamicFieldState` docstring for Interp 1 vs
    Interp 2). With `halve=True` returns exp(-rho_chain * t / 2), the
    per-half-edge no-jump probability used in the cap-2 cherry geometry
    (cherry diameter t = two half-edges of t/2 each).
    """
    s = float(t) / 2.0 if halve else float(t)
    return float(np.exp(-float(rho_chain) * s))
