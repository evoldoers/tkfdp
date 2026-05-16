"""Holmes-Rubin closed-form strict ELBO at constant rate (2-site case).

Implements the derivation in `refs/holmes_rubin_elbo.md`. The bound is:
- exact at H = 0,
- always ≤ log P_exact (Jensen),
- closed-form (no MC noise, no boundary divergence in log p),
- O(A^6) per cell at A = 20 — milliseconds.

Public API:
    fit_constant_rate_then_elbo(H, x_a, x_b, t, ...) -> dict

This iterates the geometric-mean fixed point on time-averaged bar_p (cheap)
to find Q_hat^1, Q_hat^2, then evaluates the strict closed-form ELBO at
that Q_hat via the Holmes-Rubin sums.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .lg08 import Q_LG08, PI_LG08

A = 20
EPS = 1e-30
JITTER = 1e-8


# -- Eigendecomposition helpers --------------------------------------------

def _build_Q_hat(H: jnp.ndarray, p_other: jnp.ndarray) -> jnp.ndarray:
    """Geometric-mean variational rate at fixed (constant) p_other.
    Q_hat[x, x'] = Q_LG08[x, x'] * exp(-0.5 * (H_eff[x'] - H_eff[x]))
    """
    H_eff = H @ p_other
    factor = jnp.exp(-0.5 * (H_eff[None, :] - H_eff[:, None]))
    Q_off = jnp.where(jnp.eye(A, dtype=bool), 0.0, jnp.asarray(Q_LG08) * factor)
    return Q_off - jnp.diag(Q_off.sum(axis=-1))


def _eigh_via_symmetrization(Q: jnp.ndarray, pi: jnp.ndarray):
    """Symmetrize Q via D^{1/2} Q D^{-1/2} (D = diag(pi)), eigh, return
    (Lambda, U, Vinv) such that Q = U diag(Lambda) Vinv with U Vinv = I.
    """
    sqrt = jnp.sqrt(pi)
    inv = 1.0 / jnp.clip(sqrt, EPS, None)
    Q_sym = sqrt[:, None] * Q * inv[None, :]
    Q_sym = 0.5 * (Q_sym + Q_sym.T) + JITTER * jnp.eye(A)
    Lambda, U_sym = jnp.linalg.eigh(Q_sym)
    Lambda = Lambda - JITTER
    U = inv[:, None] * U_sym                    # (A, A)
    Vinv = U_sym.T * sqrt[None, :]              # (A, A)
    return Lambda, U, Vinv


def _stationary(Q_hat: jnp.ndarray) -> jnp.ndarray:
    """Stationary of Q_hat (the eigenvector with eigenvalue 0 of Q_hat^T)."""
    # Q_hat is row-sum-zero, has 1 zero eigenvalue. Compute via SVD or eigh of Q^T Q.
    # For our use, we know pi_hat ∝ pi_LG08 * exp(-H_eff). Use that closed form.
    # Caller will provide pi.
    raise NotImplementedError("provide pi externally")


# -- Holmes-Rubin elementary integral --------------------------------------

def _hr_pair(alpha: jnp.ndarray, beta: jnp.ndarray, t: float) -> jnp.ndarray:
    """HR(alpha, beta, t) = integral_0^t exp(alpha u) exp(beta (t-u)) du.
    Vectorised over alpha, beta of compatible broadcast shapes.
    """
    e_a = jnp.exp(alpha * t)
    e_b = jnp.exp(beta * t)
    diff = alpha - beta
    safe_diff = jnp.where(jnp.abs(diff) < 1e-12, 1.0, diff)
    eqv = jnp.abs(diff) < 1e-12
    off = (e_a - e_b) / safe_diff
    diag = t * e_a
    return jnp.where(eqv, diag, off)


# -- Closed-form ELBO (the centerpiece) ------------------------------------

def _elbo_at_Qhat(H: jnp.ndarray,
                   x_a: tuple[int, int], x_b: tuple[int, int],
                   t: float,
                   Q_hat_1: jnp.ndarray, Q_hat_2: jnp.ndarray,
                   pi_hat_1: jnp.ndarray, pi_hat_2: jnp.ndarray):
    """Closed-form strict ELBO at given (constant) Q_hat^1, Q_hat^2.

    Steps follow refs/holmes_rubin_elbo.md.
    """
    # Eigendecomposition of each per-site Q_hat (via symmetrization)
    Lambda1, U1, V1inv = _eigh_via_symmetrization(Q_hat_1, pi_hat_1)
    Lambda2, U2, V2inv = _eigh_via_symmetrization(Q_hat_2, pi_hat_2)

    # Endpoint marginals P_s
    expL1_t = jnp.exp(Lambda1 * t)
    expL2_t = jnp.exp(Lambda2 * t)
    P1 = jnp.dot(U1[x_a[0], :] * expL1_t, V1inv[:, x_b[0]])
    P2 = jnp.dot(U2[x_a[1], :] * expL2_t, V2inv[:, x_b[1]])
    log_P1 = jnp.log(jnp.clip(P1, EPS, None))
    log_P2 = jnp.log(jnp.clip(P2, EPS, None))

    # Holmes-Rubin 4-tensor M[k_1, l_1, k_2, l_2] = HR(Lambda1_k1 + Lambda2_k2, Lambda1_l1 + Lambda2_l2, t)
    # Build via outer sums:
    L1_k = Lambda1[:, None, None, None]    # (k_1, _, _, _)
    L1_l = Lambda1[None, :, None, None]    # (_, l_1, _, _)
    L2_k = Lambda2[None, None, :, None]    # (_, _, k_2, _)
    L2_l = Lambda2[None, None, None, :]    # (_, _, _, l_2)
    alpha_grid = L1_k + L2_k               # (k_1, _, k_2, _)  (will broadcast)
    beta_grid = L1_l + L2_l                # (_, l_1, _, l_2)
    M = _hr_pair(alpha_grid + jnp.zeros_like(beta_grid),
                 beta_grid + jnp.zeros_like(alpha_grid), t)
    # Cleanest: explicitly broadcast
    alpha = (L1_k + L2_k) * jnp.ones((1, A, 1, A))                   # (k_1, l_1, k_2, l_2) but l_1 const along l_1 axis
    beta = (L1_l + L2_l) * jnp.ones((A, 1, A, 1))
    # The above has bugs from broadcasting weirdness; redo cleanly:
    alpha_full = Lambda1[:, None, None, None] + Lambda2[None, None, :, None]    # (k_1, _, k_2, _)
    beta_full = Lambda1[None, :, None, None] + Lambda2[None, None, None, :]     # (_, l_1, _, l_2)
    # Need shape (k_1, l_1, k_2, l_2) with alpha = Lambda1_k1 + Lambda2_k2 and beta = Lambda1_l1 + Lambda2_l2
    # Broadcasting:
    alpha_b = (Lambda1[:, None] + Lambda2[None, :])[:, None, :, None]     # (k_1, _, k_2, _) -> broadcast over l's
    beta_b = (Lambda1[:, None] + Lambda2[None, :])[None, :, None, :]      # (_, l_1, _, l_2)
    M = _hr_pair(alpha_b * jnp.ones((1, A, 1, A)),
                 beta_b * jnp.ones((A, 1, A, 1)), t)   # shape (A, A, A, A)

    # S_1[x_1, k_1, l_1] = U1[xa1, k1] V1inv[k1, x_1] U1[x_1, l1] V1inv[l1, xb1]
    S1_J = (U1[x_a[0], :, None, None]                       # (k_1, _, _)
            * V1inv[:, None, :].transpose(2, 0, 1))         # need (x_1, k_1, l_1)
    # Cleaner via einsum:
    S1_J = jnp.einsum('k,kx,xl,l->xkl',
                      U1[x_a[0], :], V1inv, U1, V1inv[:, x_b[0]])    # (x_1, k_1, l_1)
    S2_J = jnp.einsum('k,kx,xl,l->xkl',
                      U2[x_a[1], :], V2inv, U2, V2inv[:, x_b[1]])    # (x_2, k_2, l_2)

    # J(x_1, x_2) = (1/(P_1 P_2)) * sum S_1 S_2 M
    # Best contraction: T1[x_2, k_1, l_1] = sum_{k_2, l_2} M S_2; J = sum_{k_1, l_1} S_1 T1 / (P_1 P_2)
    T1_J = jnp.einsum('klmn,xmn->xkl', M, S2_J)                       # (x_2, k_1, l_1)
    J = jnp.einsum('xkl,ykl->xy', T1_J, S1_J) / (P1 * P2)             # J[x_2, x_1] -> swap to (x_1, x_2)
    J = J.T   # now (x_1, x_2)

    # Sanity: sum_{x_1, x_2} J = t
    # (we don't enforce; Jensen bound holds anyway)

    # K(x_1, y_1, x_2): same structure but with V1inv[l_1, x_1] -> V1inv[l_1, xb1]; U1[x_1, l_1] -> U1[y_1, l_1]
    # S1_K[x_1, y_1, k_1, l_1] = U1[xa1, k1] V1inv[k1, x_1] U1[y_1, l1] V1inv[l1, xb1]
    S1_K = jnp.einsum('k,kx,yl,l->xykl',
                      U1[x_a[0], :], V1inv, U1, V1inv[:, x_b[0]])     # (x_1, y_1, k_1, l_1)
    # K = (Qhat1[x_1, y_1] / (P_1 P_2)) * sum_{k_1, l_1, k_2, l_2} S1_K S2_J M
    # Use T1_J already computed: T1_J[x_2, k_1, l_1]
    K = jnp.einsum('xykl,zkl->xyz', S1_K, T1_J) * (Q_hat_1 / (P1 * P2))[:, :, None]   # (x_1, y_1, x_2)

    # Symmetric for site 2:
    S1_K2 = jnp.einsum('k,kx,yl,l->xykl',
                       U2[x_a[1], :], V2inv, U2, V2inv[:, x_b[1]])    # (x_2, y_2, k_2, l_2)
    T2_J = jnp.einsum('klmn,xkl->xmn', M, S1_J)                       # (x_1, k_2, l_2)
    K2 = jnp.einsum('xykl,zkl->xyz', S1_K2, T2_J) * (Q_hat_2 / (P1 * P2))[:, :, None]  # (x_2, y_2, x_1)
    # Swap (x_1, x_2) for site-2's K:  K2[x_2, y_2, x_1] -> we use it as K2(x_2, y_2, x_1)

    # I_gamma^1(x_1, y_1) = (Qhat1[x_1, y_1] / P_1) * sum_{k, l} U1[xa1, k] V1inv[k, x_1] U1[y_1, l] V1inv[l, xb1] * HR(Lambda1_k, Lambda1_l, t)
    HR_1 = _hr_pair(Lambda1[:, None] * jnp.ones((1, A)),
                    Lambda1[None, :] * jnp.ones((A, 1)), t)            # (k, l)
    I_gamma_1 = (Q_hat_1 / P1) * jnp.einsum('k,kx,yl,l,kl->xy',
                                              U1[x_a[0], :], V1inv,
                                              U1, V1inv[:, x_b[0]], HR_1)
    HR_2 = _hr_pair(Lambda2[:, None] * jnp.ones((1, A)),
                    Lambda2[None, :] * jnp.ones((A, 1)), t)
    I_gamma_2 = (Q_hat_2 / P2) * jnp.einsum('k,kx,yl,l,kl->xy',
                                              U2[x_a[1], :], V2inv,
                                              U2, V2inv[:, x_b[1]], HR_2)

    # bar_p_s(x) = (1/t) sum_other J  — for site 1, bar_p_1(x_1) = J(x_1, .).sum() / t
    bar_p_1 = J.sum(axis=1) / t              # (A,)
    bar_p_2 = J.sum(axis=0) / t              # (A,)

    # Off-diagonal masks
    mask_off_1 = ~jnp.eye(A, dtype=bool)
    mask_off_2 = ~jnp.eye(A, dtype=bool)

    Q_LG_off = jnp.asarray(Q_LG08) - jnp.diag(jnp.diag(jnp.asarray(Q_LG08)))

    # KL_diag^1
    # = sum_{x_1, x_2, y_1 != x_1} J(x_1, x_2) Q_LG08(x_1, y_1) exp(-0.5 dH_1)
    #   - t * sum_{x_1, y_1 != x_1} bar_p_1(x_1) Q_hat_1(x_1, y_1)
    # dH_1[x_1, y_1, x_2] = H[y_1, x_2] - H[x_1, x_2]
    dH_1 = H[None, :, :] - H[:, None, :]                              # (x_1, y_1, x_2)
    AM_1 = Q_LG_off[:, :, None] * jnp.exp(-0.5 * dH_1)                # (x_1, y_1, x_2)
    KL_diag_1 = jnp.einsum('xz,xyz->', J, AM_1) \
                - t * jnp.einsum('x,xy->', bar_p_1, Q_hat_1 * mask_off_1)
    # KL_jump^1
    # = -0.5 sum_{x_1, x_2, y_1 != x_1} K(x_1, y_1, x_2) dH_1(x_1, y_1, x_2)
    #   - sum_{x_1, y_1 != x_1} log G_1(x_1, y_1) I_gamma_1(x_1, y_1)
    log_G_1 = jnp.where(mask_off_1,
                          jnp.log(jnp.clip(Q_hat_1 / jnp.clip(jnp.asarray(Q_LG08), EPS, None), EPS, None)),
                          0.0)
    # KL_jump = -Girsanov_jump = -∫ p_other γ_self log(Q_unc/Q_var)
    #         = -∫ p_other γ_self (-0.5 dH - log G)
    #         = +0.5 sum K dH + sum log_G I_gamma   (signs flipped from earlier draft)
    KL_jump_1 = 0.5 * jnp.einsum('xyz,xyz->', K, dH_1) \
                + jnp.einsum('xy,xy->', log_G_1, I_gamma_1 * mask_off_1)

    # KL_diag^2 (symmetric: site 2 jumps, conditional on site 1)
    # dH_2[x_2, y_2, x_1] = H[y_2, x_1] - H[x_2, x_1] = H.T[x_1, y_2] - H.T[x_1, x_2]
    dH_2 = H.T[None, :, :] - H.T[:, None, :]                          # (x_2, y_2, x_1)... wait need to think
    # Actually H[y_2, x_1] - H[x_2, x_1] indexed by (x_2, y_2, x_1):
    # H[y_2, x_1] depends on (y_2, x_1)
    # H[x_2, x_1] depends on (x_2, x_1)
    H_T = H.T                                                          # H_T[a, b] = H[b, a]
    # dH_2[x_2, y_2, x_1] = H[y_2, x_1] - H[x_2, x_1]
    dH_2 = H_T[None, :, :] - H_T[:, None, :]
    # H_T[None, :, :] has shape (_, y_2, x_1), broadcasting to (x_2, y_2, x_1) -> H[x_1, y_2]?? Hmm
    # We want H[y_2, x_1]. H_T[a, b] = H[b, a]. So H[y_2, x_1] = H_T[x_1, y_2].
    # So we want: for each (x_2, y_2, x_1): H_T[x_1, y_2] - H_T[x_1, x_2]
    dH_2 = H_T[:, None, None, :].swapaxes(0, 3).swapaxes(0, 2)        # ugh, let me just do indexing
    # Cleaner: build via explicit broadcasting
    # H[y_2, x_1] indexed by (x_2, y_2, x_1): broadcast H[y_2, x_1] over x_2
    H_y2_x1 = jnp.broadcast_to(H[None, :, :], (A, A, A)).transpose(0, 1, 2)  # (x_2, y_2, x_1) with values H[y_2, x_1] independent of x_2
    H_x2_x1 = H[:, None, :]                                                  # (x_2, _, x_1) -> H[x_2, x_1]
    dH_2 = H_y2_x1 - H_x2_x1                                                  # (x_2, y_2, x_1)

    AM_2 = Q_LG_off[:, :, None] * jnp.exp(-0.5 * dH_2)                # (x_2, y_2, x_1)
    KL_diag_2 = jnp.einsum('xz,xyz->', J.T, AM_2) \
                - t * jnp.einsum('y,yx->', bar_p_2, Q_hat_2 * mask_off_2)
    # K2 has shape (x_2, y_2, x_1)
    log_G_2 = jnp.where(mask_off_2,
                          jnp.log(jnp.clip(Q_hat_2 / jnp.clip(jnp.asarray(Q_LG08), EPS, None), EPS, None)),
                          0.0)
    KL_jump_2 = 0.5 * jnp.einsum('xyz,xyz->', K2, dH_2) \
                + jnp.einsum('xy,xy->', log_G_2, I_gamma_2 * mask_off_2)

    KL_total = KL_diag_1 + KL_jump_1 + KL_diag_2 + KL_jump_2
    elbo = log_P1 + log_P2 - KL_total
    return dict(
        elbo=float(elbo),
        log_P1=float(log_P1), log_P2=float(log_P2),
        KL_diag_1=float(KL_diag_1), KL_diag_2=float(KL_diag_2),
        KL_jump_1=float(KL_jump_1), KL_jump_2=float(KL_jump_2),
        KL_total=float(KL_total),
        bar_p_1=np.asarray(bar_p_1), bar_p_2=np.asarray(bar_p_2),
    )


# -- Driver: find Q_hat via constant-rate fixed point, then evaluate ELBO --

def fit_constant_rate_then_elbo(H: np.ndarray,
                                  x_a: tuple[int, int],
                                  x_b: tuple[int, int],
                                  t: float,
                                  n_iter: int = 30,
                                  damping: float = 0.5,
                                  tol: float = 1e-9):
    """Run the constant-rate geometric-mean fixed-point iteration (using
    the closed-form ELBO machinery to compute bar_p at each step), then
    return the strict closed-form ELBO at the converged Q_hat.
    """
    H_j = jnp.asarray(H)
    pi_LG08_j = jnp.asarray(PI_LG08)

    # Initialize bar_p ~ pi_LG08
    bar_p_1 = pi_LG08_j
    bar_p_2 = pi_LG08_j

    for it in range(n_iter):
        Q_hat_1 = _build_Q_hat(H_j, bar_p_2)
        Q_hat_2 = _build_Q_hat(H_j, bar_p_1)
        # pi_hat_s ∝ pi_LG08 * exp(-H @ p_other)
        log_pi_1 = jnp.log(pi_LG08_j) - H_j @ bar_p_2
        log_pi_2 = jnp.log(pi_LG08_j) - H_j @ bar_p_1
        pi_hat_1 = jax.nn.softmax(log_pi_1)
        pi_hat_2 = jax.nn.softmax(log_pi_2)

        # Evaluate ELBO components to extract bar_p
        info = _elbo_at_Qhat(H_j, x_a, x_b, t, Q_hat_1, Q_hat_2, pi_hat_1, pi_hat_2)
        new_bp_1 = jnp.asarray(info["bar_p_1"])
        new_bp_2 = jnp.asarray(info["bar_p_2"])
        new_bp_1 = jnp.clip(new_bp_1, 1e-12, None)
        new_bp_2 = jnp.clip(new_bp_2, 1e-12, None)
        new_bp_1 = new_bp_1 / new_bp_1.sum()
        new_bp_2 = new_bp_2 / new_bp_2.sum()

        new_bp_1 = damping * new_bp_1 + (1 - damping) * bar_p_1
        new_bp_2 = damping * new_bp_2 + (1 - damping) * bar_p_2
        delta = float(max(jnp.abs(new_bp_1 - bar_p_1).max(),
                          jnp.abs(new_bp_2 - bar_p_2).max()))
        bar_p_1, bar_p_2 = new_bp_1, new_bp_2
        if delta < tol:
            break

    Q_hat_1 = _build_Q_hat(H_j, bar_p_2)
    Q_hat_2 = _build_Q_hat(H_j, bar_p_1)
    log_pi_1 = jnp.log(pi_LG08_j) - H_j @ bar_p_2
    log_pi_2 = jnp.log(pi_LG08_j) - H_j @ bar_p_1
    pi_hat_1 = jax.nn.softmax(log_pi_1)
    pi_hat_2 = jax.nn.softmax(log_pi_2)
    info = _elbo_at_Qhat(H_j, x_a, x_b, t, Q_hat_1, Q_hat_2, pi_hat_1, pi_hat_2)
    info["n_iter"] = it + 1
    return info
