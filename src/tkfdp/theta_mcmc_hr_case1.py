"""Case M >= 1 SS attribution for tied-theta MCMC HR accumulator.

Given a per-branch sampled (theta_p, theta_c, X_p, X_c, tau_v) with
M_v = 1 (>= 1 field jump), returns per-(theta, A) contributions to
V/W and per-(theta, A, A) to U for the site's row (c_s) in
(K_c, L, A) / (K_c, L, A, A) accumulators, plus per-branch E_N_jump.

Extracted from src/tkfdp/coupling/dynfield/hr_jax.py::_case_jump. The
substantive difference: theta_p, theta_c are FIXED (sampled), so we
skip the L*L sweep and the field-side / residue-side weighting
(rho[theta_X] * P_jump * L_res_j) that applies when theta_X, theta_Y
are marginalised. The per-site conditional SS derived here are what
the tied-theta M-step needs.

Attribution structure (per site n at class c_s):
  V[c_s, theta_p, X_p]     += 1                           (first-segment start)
  V[c_s, theta_c, :]       += P_avg_last                  (last-jump destination)
  V[c_s, theta_int, :]     += E_non_last * pi_arch[k_int] (middle arrivals)
  W[c_s, theta_p, :]       += W_first                     (first-segment dwell)
  W[c_s, theta_c, :]       += W_last                      (last-segment dwell)
  W[c_s, theta_int, :]     += middle_time * pi_arch[k_int]
  U[c_s, theta_p, :, :]    += U_first
  U[c_s, theta_c, :, :]    += pi_arch[k_c] * S * W_last^T (off-diag)
  U[c_s, theta_int, :, :]  += middle_time * S * (pi outer pi) (off-diag)

with all first/last SS averaged over the exact mixture density for
(tau_1, tau_v - tau_M) as in par:arch-hr, and middle-segment SS from
the field bridge conditional on (theta_p, theta_c, M >= 1) endpoints.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tkfdp.coupling.dynfield.hr_jax import (
    field_bridge_stats,
    gtr_free_end_hr_avg_case1_jax,
    gtr_transition_avg_case1_jax,
)


def _case_jump_per_site(
    pi_arch, xi, U_arch, D_arch, S,
    rho, rho_chain, arch_assignment,
    c_s, theta_p, theta_c,
    X_p, X_c, tau_v, m_site,
):
    """Return (V_LA (L, A), W_LA (L, A), U_LAA (L, A, A), E_N_jump).

    Under jit + vmap over per-call scalars/ints.
    """
    L = rho.shape[0]
    A = pi_arch.shape[1]

    P_XY, E_T, E_N_all, E_arr = field_bridge_stats(
        rho, rho_chain, theta_p, theta_c, tau_v)
    g_t = jnp.exp(-rho_chain * tau_v)
    beta_t = jnp.exp(-rho_chain * (1.0 - rho[theta_p]) * tau_v)
    same = (theta_p == theta_c).astype(jnp.float64)
    P_no_jump = jnp.where(theta_p == theta_c, beta_t, 0.0)
    P_jump = jnp.maximum(0.0, P_XY - P_no_jump)

    E_T_all_prod = E_T * P_XY
    E_T_all_prod = jnp.where(
        theta_p == theta_c,
        E_T_all_prod.at[theta_p].add(-P_no_jump * tau_v),
        E_T_all_prod)
    # E_T_jump[i] = E[time at θ=i | bridge, M ≥ 1]. Bounded by τ_v.
    # Under self-loop with vanishing ε the numerator and P_jump both
    # cancel to O(ε²) and precision loss can blow the ratio; clamp to
    # the physical range [0, τ_v] to catch it (same rationale as the
    # L'Hopital treatment of E_N_jump).
    E_T_jump_raw = E_T_all_prod / jnp.maximum(P_jump, 1e-300)
    E_T_jump = jnp.clip(E_T_jump_raw, 0.0, tau_v)
    # E_N_jump likewise physical-bounded: for well-conditioned branches
    # ~O(1); loose clamp at 1000. (Separate _E_N_jump_scalar with
    # L'Hopital handles the ρ_chain SS; here we only use E_N_jump
    # indirectly through E_arrivals sums, so the clamp is a
    # belt-and-suspenders.)
    E_N_jump = jnp.clip(
        E_N_all * P_XY / jnp.maximum(P_jump, 1e-300), 0.0, 1000.0)
    # E_arrivals_jump[i] ≤ E_N_jump total; clamp to 1000 per θ.
    E_arrivals_jump = jnp.clip(
        E_arr * P_XY / jnp.maximum(P_jump, 1e-300), 0.0, 1000.0)

    r_X = rho_chain * (1.0 - rho[theta_p])
    r_Y = rho_chain * (1.0 - rho[theta_c])
    delta_1_f = -r_X
    delta_2_f = rho_chain * rho[theta_p]
    beta_1_f = r_X * rho[theta_c]
    beta_2_f = (r_X * (rho[theta_p]
                        / jnp.maximum(1.0 - rho[theta_p], 1e-300))
                  * (rho[theta_c] - same) * g_t)
    delta_1_l = -r_Y
    delta_2_l = rho_chain * rho[theta_c]
    beta_1_l = r_Y * rho[theta_p]
    beta_2_l = (r_Y * (rho[theta_c]
                        / jnp.maximum(1.0 - rho[theta_c], 1e-300))
                  * (rho[theta_p] - same) * g_t)

    def _Z(delta, T):
        d_safe = jnp.where(jnp.abs(delta) < 1e-12, 1.0, delta)
        return jnp.where(jnp.abs(delta) < 1e-12, T,
                            (jnp.exp(delta * T) - 1.0) / d_safe)

    def _E_tau(delta, T):
        d_safe = jnp.where(jnp.abs(delta) < 1e-12, 1.0, delta)
        exp_dT = jnp.exp(delta * T)
        return jnp.where(
            jnp.abs(delta) < 1e-12,
            T / 2.0,
            T * exp_dT / (exp_dT - 1.0) - 1.0 / d_safe)

    Z_1_f = _Z(delta_1_f, tau_v); Z_2_f = _Z(delta_2_f, tau_v)
    Z_tot_f = jnp.maximum(beta_1_f * Z_1_f + beta_2_f * Z_2_f, 1e-300)
    a_1_f = beta_1_f * Z_1_f / Z_tot_f
    a_2_f = beta_2_f * Z_2_f / Z_tot_f
    Z_1_l = _Z(delta_1_l, tau_v); Z_2_l = _Z(delta_2_l, tau_v)
    Z_tot_l = jnp.maximum(beta_1_l * Z_1_l + beta_2_l * Z_2_l, 1e-300)
    a_1_l = beta_1_l * Z_1_l / Z_tot_l
    a_2_l = beta_2_l * Z_2_l / Z_tot_l

    E_tau1 = a_1_f * _E_tau(delta_1_f, tau_v) + a_2_f * _E_tau(delta_2_f, tau_v)
    E_tauM_to_t = (a_1_l * _E_tau(delta_1_l, tau_v)
                      + a_2_l * _E_tau(delta_2_l, tau_v))

    k_by_theta = arch_assignment[c_s]                    # (L,)
    k_p = k_by_theta[theta_p]
    k_c = k_by_theta[theta_c]
    pi_kp = pi_arch[k_p]
    pi_kc = pi_arch[k_c]
    # Per-site rate scaling: xi_scaled = m_site * xi[k]. m_site = 0
    # → invariant site (no substitution between jumps, residue only
    # resamples at jumps via Interp 2). Matches the m-scaling used by
    # _cluster_stats_inner in hr_jax.py for the composite path.
    xi_p = m_site * xi[k_p]
    xi_c = m_site * xi[k_c]
    Up = U_arch[k_p]
    Uc = U_arch[k_c]
    Dp = D_arch[k_p]
    Dc = D_arch[k_c]

    # --- V ---
    V_first_A = jax.nn.one_hot(X_p, A)

    P_avg_last_1 = gtr_transition_avg_case1_jax(
        pi_kc, xi_c, Uc, Dc, S, X_c, tau_v, delta_1_l)
    P_avg_last_2 = gtr_transition_avg_case1_jax(
        pi_kc, xi_c, Uc, Dc, S, X_c, tau_v, delta_2_l)
    P_avg_last = a_1_l * P_avg_last_1 + a_2_l * P_avg_last_2

    def _V_mid(theta_int):
        k_i = k_by_theta[theta_int]
        pi_k_i = pi_arch[k_i]
        E_non_last = E_arrivals_jump[theta_int] - jnp.where(
            theta_int == theta_c, 1.0, 0.0)
        E_non_last = jnp.maximum(E_non_last, 0.0)
        return E_non_last * pi_k_i
    V_mid_LA = jax.vmap(_V_mid)(jnp.arange(L))

    oh_p = jax.nn.one_hot(theta_p, L)
    oh_c = jax.nn.one_hot(theta_c, L)
    V_LA = (oh_p[:, None] * V_first_A[None, :]
              + oh_c[:, None] * P_avg_last[None, :]
              + V_mid_LA)

    # --- W ---
    W_first_1, U_first_1 = gtr_free_end_hr_avg_case1_jax(
        pi_kp, xi_p, Up, Dp, S, X_p, tau_v, delta_1_f)
    W_first_2, U_first_2 = gtr_free_end_hr_avg_case1_jax(
        pi_kp, xi_p, Up, Dp, S, X_p, tau_v, delta_2_f)
    W_first = a_1_f * W_first_1 + a_2_f * W_first_2
    U_first = a_1_f * U_first_1 + a_2_f * U_first_2

    W_last_1, _ = gtr_free_end_hr_avg_case1_jax(
        pi_kc, xi_c, Uc, Dc, S, X_c, tau_v, delta_1_l)
    W_last_2, _ = gtr_free_end_hr_avg_case1_jax(
        pi_kc, xi_c, Uc, Dc, S, X_c, tau_v, delta_2_l)
    W_last = a_1_l * W_last_1 + a_2_l * W_last_2

    def _W_mid(theta_int):
        k_i = k_by_theta[theta_int]
        pi_k_i = pi_arch[k_i]
        first_time = jnp.where(theta_int == theta_p, E_tau1, 0.0)
        last_time = jnp.where(theta_int == theta_c, E_tauM_to_t, 0.0)
        middle_time = jnp.maximum(0.0, E_T_jump[theta_int] - first_time - last_time)
        return middle_time * pi_k_i
    W_mid_LA = jax.vmap(_W_mid)(jnp.arange(L))
    W_LA = (oh_p[:, None] * W_first[None, :]
              + oh_c[:, None] * W_last[None, :]
              + W_mid_LA)

    # --- U ---
    # All U contributions get an overall m_site multiplier: for the
    # first/last-segment primitives, U primitives return (1/m) times
    # the physical Q^{k, m}·W due to the m·ξ scaling in the J
    # denominator (see comment in hr_jax.py::_cluster_stats_inner);
    # multiplying by m at attribution corrects. For the middle segment
    # (Dynkin on stationary chain), m_site·S·(π outer π) is directly
    # the physical Q^{k, m}·W_mid form.
    U_last_ij = pi_kc[:, None] * S * W_last[None, :]
    U_last_ij = U_last_ij - jnp.diag(jnp.diag(U_last_ij))

    def _U_mid(theta_int):
        k_i = k_by_theta[theta_int]
        pi_k_i = pi_arch[k_i]
        first_time = jnp.where(theta_int == theta_p, E_tau1, 0.0)
        last_time = jnp.where(theta_int == theta_c, E_tauM_to_t, 0.0)
        middle_time = jnp.maximum(0.0, E_T_jump[theta_int] - first_time - last_time)
        outer = pi_k_i[:, None] * pi_k_i[None, :]
        U_int = middle_time * S * outer
        U_int = U_int - jnp.diag(jnp.diag(U_int))
        return U_int
    U_mid_LAA = jax.vmap(_U_mid)(jnp.arange(L))
    U_LAA = m_site * (
        oh_p[:, None, None] * U_first[None, :, :]
        + oh_c[:, None, None] * U_last_ij[None, :, :]
        + U_mid_LAA)

    return V_LA, W_LA, U_LAA, E_N_jump


def _E_N_jump_scalar(rho, rho_chain, theta_p, theta_c, tau_v):
    """Return E[N_jumps | M >= 1, theta_p, theta_c, tau_v] under the
    F81-on-DP field bridge. Scalar in / scalar out.

    Cancellation regime: for theta_p == theta_c with vanishing
    epsilon = rho_chain * tau, both the numerator EN_times_P and the
    denominator P_jump = P_XY - beta_t are O(epsilon^2), and float64
    subtraction destroys the tail. Direct evaluation gives spurious
    E_N_jump ~ 10^283 whenever the O(epsilon^2) cancellation
    underflows the 1e-300 floor.

    Analytic limit (independent of the rho distribution): both
    numerator and denominator carry a factor of rho[theta] *
    (1 - rho[theta]) * epsilon^2. The leading-order coefficients give
    EN_times_P / P_jump -> 2 as epsilon -> 0. Higher-order correction
    is O(epsilon), so using the literal 2 whenever epsilon is below
    the float64 cancellation threshold is exact to ~1e-6 for the
    threshold below (well within any downstream accumulation
    tolerance).

    Cross-theta case has no cancellation and uses the direct formula.
    A loose upper clamp (1000) is retained as belt+suspenders for any
    other edge case (e.g. numerical noise pushing E_N_all itself into
    the tail).
    """
    P_XY, _E_T, E_N_all, _E_arr = field_bridge_stats(
        rho, rho_chain, theta_p, theta_c, tau_v)
    beta_t = jnp.exp(-rho_chain * (1.0 - rho[theta_p]) * tau_v)
    P_no_jump = jnp.where(theta_p == theta_c, beta_t, 0.0)
    P_jump = jnp.maximum(0.0, P_XY - P_no_jump)
    raw = E_N_all * P_XY / jnp.maximum(P_jump, 1e-300)
    # L'Hopital limit: EN_times_P and P_jump both O(ε^2) for self-loop;
    # ratio → 2 exactly as ε → 0. Threshold 1e-6 keeps the general
    # formula whenever it is numerically well-conditioned.
    eps = rho_chain * tau_v
    same = (theta_p == theta_c)
    lhopital = same & (eps < 1e-6)
    val = jnp.where(lhopital, 2.0, raw)
    return jnp.minimum(jnp.maximum(val, 0.0), 1000.0)


_case_jump_kernel = None
_E_N_jump_kernel = None


def get_case_jump_kernel():
    global _case_jump_kernel
    if _case_jump_kernel is None:
        batched = jax.vmap(
            _case_jump_per_site,
            in_axes=(None, None, None, None, None, None, None, None,
                       0, 0, 0, 0, 0, 0, 0))
        _case_jump_kernel = jax.jit(batched)
    return _case_jump_kernel


def get_E_N_jump_kernel():
    """Return jitted vmap of _E_N_jump_scalar over per-branch (theta_p,
    theta_c, tau_v) arrays."""
    global _E_N_jump_kernel
    if _E_N_jump_kernel is None:
        batched = jax.vmap(
            _E_N_jump_scalar,
            in_axes=(None, None, 0, 0, 0))
        _E_N_jump_kernel = jax.jit(batched)
    return _E_N_jump_kernel
