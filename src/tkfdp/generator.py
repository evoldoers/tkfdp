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


# ---------------------------------------------------------------------------
# A1 (Sinkhorn) reversibility correction. See psb-paper/supplement.tex
# `sec:rev-suppl` for the math; in short, the naive joint
#     pi_joint_naive(a,b) ∝ pi_a(a) pi_b(b) exp(-H(a,b))
# has marginals different from (pi_a, pi_b) whenever H != 0, which makes the
# augmented (substitution + indel) generator non-reversible on partial-
# presence pairs. The single-site Sinkhorn fix introduces side potentials
# (h_a, h_b) so that
#     pi_joint_a1(a,b) ∝ pi_a(a) pi_b(b) exp(-h_a(a) - h_b(b) - H(a,b))
# has marginals exactly (pi_a, pi_b). The interaction is preserved exactly
# (every log-odds-ratio of H survives the scaling), so H still carries
# correlation; the side potentials are a deterministic readout of (pi, H).
# ---------------------------------------------------------------------------

def sinkhorn_pair(H: jnp.ndarray,
                  pi_a: jnp.ndarray, pi_b: jnp.ndarray,
                  max_iter: int = 200, tol: float = 1e-12
                  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sinkhorn / IPF matrix scaling: find side potentials (h_a, h_b) so the
    corrected joint
        pi_joint_a1(a,b) ∝ pi_a(a) pi_b(b) exp(-h_a(a) - h_b(b) - H(a,b))
    has marginals exactly pi_a (rows) and pi_b (cols).

    Gauge: we normalize so that pi_a[0]-weighted h_a sums to zero (one
    arbitrary additive constant per row potential and per column potential
    is absorbed by the joint normalizer). Iteration runs in log space for
    numerical stability and is implemented as a fixed-iter jax.lax.while_loop
    so it stays JIT-compatible.

    Returns (h_a, h_b), both shape (A,).
    """
    H_sym = 0.5 * (H + H.T)
    log_pi_a = jnp.log(pi_a)
    log_pi_b = jnp.log(pi_b)
    # log K0(a, b) = log pi_a + log pi_b - H_sym; Sinkhorn iterates
    # log_u, log_v such that logsumexp_b (log K0 + log_v) = log pi_a + log_u^{-1}
    # equivalently rescales rows then cols repeatedly.
    log_K0 = log_pi_a[:, None] + log_pi_b[None, :] - H_sym

    def body(state):
        log_u, log_v, _it, _err = state
        # row rescale
        log_u_new = log_pi_a - jax.scipy.special.logsumexp(
            log_K0 + log_v[None, :], axis=1)
        # col rescale
        log_v_new = log_pi_b - jax.scipy.special.logsumexp(
            log_K0 + log_u_new[:, None], axis=0)
        err = jnp.maximum(
            jnp.max(jnp.abs(log_u_new - log_u)),
            jnp.max(jnp.abs(log_v_new - log_v)))
        return (log_u_new, log_v_new, _it + 1, err)

    def cond(state):
        _u, _v, it, err = state
        return (it < max_iter) & (err > tol)

    init = (jnp.zeros(A), jnp.zeros(A), 0, jnp.inf)
    log_u, log_v, _, _ = jax.lax.while_loop(cond, body, init)
    # h_a = -log_u, h_b = -log_v (so pi_joint_a1 ∝ K0 * exp(log_u + log_v))
    h_a = -log_u
    h_b = -log_v
    # Gauge-fix: center each potential by subtracting its (pi-weighted) mean.
    h_a = h_a - jnp.sum(pi_a * h_a)
    h_b = h_b - jnp.sum(pi_b * h_b)
    return h_a, h_b


def joint_stationary_pair_a1(H: jnp.ndarray,
                              pi_a: jnp.ndarray,
                              pi_b: jnp.ndarray) -> jnp.ndarray:
    """Sinkhorn-corrected joint stationary (the A1 amendment).

    Marginal-consistency: sum_b pi_joint(a,b) = pi_a(a) and sum_a = pi_b(b)
    hold exactly, restoring reversibility of the augmented generator.
    """
    H_sym = 0.5 * (H + H.T)
    h_a, h_b = sinkhorn_pair(H_sym, pi_a, pi_b)
    log_pi = (jnp.log(pi_a)[:, None] + jnp.log(pi_b)[None, :]
              - h_a[:, None] - h_b[None, :] - H_sym)
    log_pi = log_pi - jax.scipy.special.logsumexp(log_pi)
    return jnp.exp(log_pi).reshape(A2)


def build_joint_Q_pair_a1(H: jnp.ndarray,
                           pi_a: jnp.ndarray, pi_b: jnp.ndarray,
                           S: jnp.ndarray = S_LG08_J,
                           eta_pair: tuple[float, float] = (1.0, 1.0)
                           ) -> jnp.ndarray:
    """A1 (Sinkhorn-corrected) joint generator.

    The pair CTMC stays reversible w.r.t. the marginal-consistent joint
    pi_joint_a1 (so detailed balance holds across the indel seam, where
    the surviving site continues under the lone pi_a/pi_b). Construction:
    apply the same F81-form pattern as `build_joint_Q_pair` but with each
    per-site stationary multiplied by the Sinkhorn factor exp(-h_*).

    Site-1 flip rate: eta_1 S[a,a'] * pi_a(a') * exp(-h_a(a')) * exp(-0.5 dH_1)
    Site-2 flip rate: eta_2 S[b,b'] * pi_b(b') * exp(-h_b(b')) * exp(-0.5 dH_2)

    This satisfies the detailed-balance condition with pi_joint_a1 (see
    psb-paper/supplement.tex sec:rev-suppl).
    """
    eta_1, eta_2 = eta_pair
    H_sym = 0.5 * (H + H.T)
    h_a, h_b = sinkhorn_pair(H_sym, pi_a, pi_b)
    pi_a_tilde = pi_a * jnp.exp(-h_a)
    pi_b_tilde = pi_b * jnp.exp(-h_b)
    S_off = S - jnp.diag(jnp.diag(S))
    R1 = eta_1 * (S_off * pi_a_tilde[None, :])[:, None, :] * jnp.exp(
        -0.5 * (H_sym.T[None, :, :] - H_sym[:, :, None])
    )
    R2 = eta_2 * (S_off * pi_b_tilde[None, :])[None, :, :] * jnp.exp(
        -0.5 * (H_sym[:, None, :] - H_sym[:, :, None])
    )
    eye = jnp.eye(A)
    Q4 = (R1[:, :, :, None] * eye[None, :, None, :]
          + R2[:, :, None, :] * eye[:, None, :, None])
    Q = Q4.reshape(A2, A2)
    row_sums = Q.sum(axis=1)
    Q = Q - jnp.diag(row_sums)
    return Q


def conditional_boltzmann_insertion(pi_joint_flat: jnp.ndarray,
                                      given_axis: int) -> jnp.ndarray:
    """Return the conditional P(other_axis | given_axis) at a coupled pair
    under joint stationary pi_joint_flat (shape (A^2,)).

    Used at indel-seam insertion: when one site of a pair is born next to
    a still-alive partner at AA `a`, the new residue's AA must be drawn
    from pi_joint(other | a) to satisfy detailed balance with the
    cross-level deletion edge (Thm. `thm:revcond` of the supplement).

    given_axis: 0 if the surviving site is at axis 0 (rows), 1 if axis 1.

    Returns (A, A) matrix where row `a` is P(other | given=a).
    """
    pi = pi_joint_flat.reshape(A, A)
    if given_axis == 0:
        marg = pi.sum(axis=1, keepdims=True)
    else:
        marg = pi.sum(axis=0, keepdims=True).T  # (A, 1) of column sums
    cond = pi / jnp.clip(marg, 1e-300, None)
    if given_axis == 1:
        cond = cond.T  # so row=given index, col=other index uniformly
    return cond


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
