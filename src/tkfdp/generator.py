"""400-state joint generator Q(eta, S, pi, H) on (x_i, x_j) pairs.

F81 form per main.tex \S2 (post-2026-05-08 reparameterization):

    Q^s(x -> x') = eta_s * S[x, x'] * pi(x') * exp(-0.5 * dH_s)

where eta_s is the per-site rate multiplier, S is the (symmetric)
exchangeability, pi is the (per-class) stationary distribution, and
dH_s = H[x', y] - H[x, y] is the Potts cost differential against the
partner column. The joint generator on (x, y) places site-1 jumps at
rate `eta_1 * S * pi_1 * exp(-0.5 dH_1)` and site-2 jumps at rate
`eta_2 * S * pi_2 * exp(-0.5 dH_2)` symmetrically. Simultaneous
two-site jumps are forbidden. State indexing: idx = x * 20 + y.

Reversibility w.r.t. pi_joint(x, y) ∝ pi_1(x) pi_2(y) exp(-H(x, y))
holds with the same exp(-0.5 dH) Metropolis factor as before; the
F81 form is one of two natural reversible instances of GTR (see
main.tex \S2 'F81 vs. symmetric-Metropolis form' remark). Using F81
combined with secret-destination augmentation (main.tex \S7.4) yields
strict Dirichlet--multinomial conjugacy on pi^(c).

Per-class eigendecomposition: the symmetrized similarity transform
D^{1/2} Q D^{-1/2} = eta_pair * S * sqrt(pi_x * pi_y) is now pi-
dependent, so the eigh has to run per pi (per class). At A=20 each
A^2 x A^2 eigh is small.
"""

from __future__ import annotations

import sys
import warnings

import jax
import jax.numpy as jnp

from .lg08 import PI_LG08_J, Q_LG08_J, S_LG08_F81_J as S_LG08_J

A = 20
A2 = A * A
JITTER = 1e-6  # diagonal jitter on symmetrized Q before eigh to avoid degenerate-eigenvalue NaNs in the JVP


_LEGACY_SINGLE_PI_BANNER = (
    "\n"
    "============================================================\n"
    " WARNING: legacy single-pi joint Q called ({fn}).\n"
    " This path uses ONE stationary distribution (default LG08) for\n"
    " BOTH sites of a coupled column-pair, ignoring per-site-class\n"
    " stationaries. It is NOT on the canonical v2 training graph.\n"
    " Use joint_stationary_pair / build_joint_Q_pair with explicit\n"
    " (pi_a, pi_b) instead.\n"
    "============================================================\n"
)


def _warn_legacy_single_pi(fn_name: str) -> None:
    # Loud, impossible-to-miss: both a Python UserWarning (so test runners
    # and IDEs surface it) and a stderr banner (so it shows in log files
    # even when warning filters silence the UserWarning).
    warnings.warn(
        f"{fn_name} uses a single pi (default LG08) for both sites; "
        f"use build_joint_Q_pair / joint_stationary_pair with per-class pi.",
        UserWarning, stacklevel=3,
    )
    sys.stderr.write(_LEGACY_SINGLE_PI_BANNER.format(fn=fn_name))
    sys.stderr.flush()


def joint_stationary(H: jnp.ndarray,
                     pi: jnp.ndarray = PI_LG08_J) -> jnp.ndarray:
    """LEGACY single-pi joint stationary. Emits a loud warning at every call.

    Computes pi_joint[x, y] = pi[x] * pi[y] * exp(-H[x, y]) / Z with a SINGLE
    pi (default LG08) for both sites. Retained only for v1 callers (composite
    likelihood, sim, val_loglik); the canonical v2 substitution path uses
    `joint_stationary_pair(H, pi_a, pi_b)` with per-class pi for each site.
    """
    _warn_legacy_single_pi("joint_stationary")
    H_sym = 0.5 * (H + H.T)
    log_w = jnp.log(pi)[:, None] + jnp.log(pi)[None, :] - H_sym
    log_w = log_w - jax.scipy.special.logsumexp(log_w)
    return jnp.exp(log_w).reshape(A2)


def joint_stationary_pair(H: jnp.ndarray,
                          pi_a: jnp.ndarray,
                          pi_b: jnp.ndarray) -> jnp.ndarray:
    """For pair (s, t) with classes (a, b): pi_joint[x, y] =
    pi_a[x] * pi_b[y] * exp(-H[x, y]) / Z."""
    H_sym = 0.5 * (H + H.T)
    log_w = jnp.log(pi_a)[:, None] + jnp.log(pi_b)[None, :] - H_sym
    log_w = log_w - jax.scipy.special.logsumexp(log_w)
    return jnp.exp(log_w).reshape(A2)


def build_joint_Q(H: jnp.ndarray,
                  pi: jnp.ndarray = PI_LG08_J,
                  S: jnp.ndarray = S_LG08_J,
                  eta_pair: tuple[float, float] = (1.0, 1.0)) -> jnp.ndarray:
    """LEGACY single-pi joint generator. Emits a loud warning at every call.

    Site-1 flip rate (x, y) -> (x', y):  eta_1 * S[x, x'] * pi(x') * exp(-0.5 dH_1)
    Site-2 flip rate (x, y) -> (x, y'):  eta_2 * S[y, y'] * pi(y') * exp(-0.5 dH_2)

    where dH_1 = H[x', y] - H[x, y] and dH_2 = H[x, y'] - H[x, y].

    At H = 0, eta_pair = (1, 1), pi = PI_LG08, S = S_LG08, this reduces
    to Q_LG08 ⊗ I + I ⊗ Q_LG08 (two independent LG08 chains). For the
    symmetric pi_a = pi_b = pi case (within-class cluster), the joint
    generator is reversible w.r.t. the joint Potts stationary
    pi(x) pi(y) exp(-H(x, y)) / Z.

    LEGACY: This is the single-pi path predating the F81-form per-class-pi
    reparam. Use ``build_joint_Q_pair(H, pi_a, pi_b, S, eta_pair)`` with
    per-class pi for the canonical v2 training path.
    """
    _warn_legacy_single_pi("build_joint_Q")
    eta_1, eta_2 = eta_pair
    H_sym = 0.5 * (H + H.T)
    S_off = S - jnp.diag(jnp.diag(S))   # zero diagonal

    # Site-1 base rate: S[x, x'] * pi(x'). Shape (A, A) -> broadcast over y axis.
    base1 = S_off * pi[None, :]   # (x, x') -> S[x, x'] * pi[x']
    # Modulate by H exponential (site-1 destination is x'): dH_1 = H[x', y] - H[x, y]
    R1 = eta_1 * base1[:, None, :] * jnp.exp(
        -0.5 * (H_sym.T[None, :, :] - H_sym[:, :, None])
    )   # R1[x, y, x']

    # Site-2 base rate: S[y, y'] * pi(y'). Note: pi is the PER-CLASS stationary
    # for site-2 here too (within-class case). For inter-class pairs use
    # build_joint_Q_pair which takes pi_a, pi_b separately.
    base2 = S_off * pi[None, :]   # (y, y') -> S[y, y'] * pi[y']
    R2 = eta_2 * base2[None, :, :] * jnp.exp(
        -0.5 * (H_sym[:, None, :] - H_sym[:, :, None])
    )   # R2[x, y, y']

    eye = jnp.eye(A)
    Q4 = (R1[:, :, :, None] * eye[None, :, None, :]
          + R2[:, :, None, :] * eye[:, None, :, None])
    Q = Q4.reshape(A2, A2)
    row_sums = Q.sum(axis=1)
    Q = Q - jnp.diag(row_sums)
    return Q


def build_joint_Q_pair(H: jnp.ndarray,
                        pi_a: jnp.ndarray, pi_b: jnp.ndarray,
                        S: jnp.ndarray = S_LG08_J,
                        eta_pair: tuple[float, float] = (1.0, 1.0)) -> jnp.ndarray:
    """Inter-class variant of build_joint_Q: site-1 uses pi_a, site-2 uses pi_b.
    Reversible w.r.t. pi_a(x) pi_b(y) exp(-H(x, y)) / Z.
    """
    eta_1, eta_2 = eta_pair
    H_sym = 0.5 * (H + H.T)
    S_off = S - jnp.diag(jnp.diag(S))
    R1 = eta_1 * (S_off * pi_a[None, :])[:, None, :] * jnp.exp(
        -0.5 * (H_sym.T[None, :, :] - H_sym[:, :, None])
    )
    R2 = eta_2 * (S_off * pi_b[None, :])[None, :, :] * jnp.exp(
        -0.5 * (H_sym[:, None, :] - H_sym[:, :, None])
    )
    eye = jnp.eye(A)
    Q4 = (R1[:, :, :, None] * eye[None, :, None, :]
          + R2[:, :, None, :] * eye[:, None, :, None])
    Q = Q4.reshape(A2, A2)
    row_sums = Q.sum(axis=1)
    Q = Q - jnp.diag(row_sums)
    return Q


def symmetrize_eigh(Q: jnp.ndarray, pi_joint: jnp.ndarray):
    """Symmetrize via Q_sym = D^{1/2} Q D^{-1/2} with D = diag(pi_joint),
    eigendecompose, and return (Lambda, U_sym, sqrt_pi_joint).

    Reconstruction:  exp(Q t) = D^{-1/2} U_sym diag(exp(Lambda t)) U_sym^T D^{1/2}.

    Under F81 the symmetrized matrix is eta_pair * S * sqrt(pi_x * pi_y),
    which depends on pi (per-class) and so cannot be cached across
    classes (unlike the symmetric-Metropolis form previously used).
    Each per-class eigh is O(A^4) on the 400x400 joint generator and is
    not the bottleneck.
    """
    sqrt = jnp.sqrt(pi_joint)
    inv = 1.0 / sqrt
    Q_sym = sqrt[:, None] * Q * inv[None, :]
    Q_sym = 0.5 * (Q_sym + Q_sym.T)
    Q_sym_j = Q_sym + JITTER * jnp.eye(Q_sym.shape[0])
    Lambda, U_sym = jnp.linalg.eigh(Q_sym_j)
    Lambda = Lambda - JITTER
    return Lambda, U_sym, sqrt


def transition_matrices(t_values: jnp.ndarray,
                        Lambda: jnp.ndarray,
                        U_sym: jnp.ndarray,
                        sqrt_pi_joint: jnp.ndarray) -> jnp.ndarray:
    """Vectorize exp(Q t) over a batch of t values."""
    inv = 1.0 / sqrt_pi_joint
    def one(t):
        expL = jnp.exp(Lambda * t)
        M_sym = (U_sym * expL[None, :]) @ U_sym.T
        return inv[:, None] * M_sym * sqrt_pi_joint[None, :]
    return jax.vmap(one)(t_values)


def log_transition_matrices(t_values: jnp.ndarray,
                            Lambda: jnp.ndarray,
                            U_sym: jnp.ndarray,
                            sqrt_pi_joint: jnp.ndarray) -> jnp.ndarray:
    P = transition_matrices(t_values, Lambda, U_sym, sqrt_pi_joint)
    return jnp.log(jnp.clip(P, 1e-300, 1.0))


def single_site_transition(t: float,
                           pi: jnp.ndarray = PI_LG08_J,
                           S: jnp.ndarray = S_LG08_J,
                           eta: float = 1.0) -> jnp.ndarray:
    """exp(eta * Q_F81(S, pi) * t) as a (20, 20) matrix.

    For default (pi = PI_LG08, S = S_LG08, eta = 1.0) this matches LG08
    timing (mean rate = 1 sub/site/unit time at the LG08 stationary).
    """
    import numpy as np
    import jax.scipy.linalg as jsl
    Q = (S - jnp.diag(jnp.diag(S))) * pi[None, :]
    row_sums = Q.sum(axis=1)
    Q = Q - jnp.diag(row_sums)
    # Normalize so that the mean rate at pi is 1 (LG08 convention).
    mean_rate = -jnp.sum(pi * jnp.diag(Q))
    Q = Q / mean_rate
    return jsl.expm(eta * Q * t)
