"""JAX-vectorized cluster-level Holmes-Rubin sufficient statistics.

Port of hr.py::hr_cluster_stats to a jit/vmap-friendly form. The pure-numpy
version costs ~450 ms per cluster for K_a=2, L=4, A=20, N=2 (mostly Python
overhead across the L^2 x N inner loops and the ~4 sub-primitives per
inner iteration). This module hoists all per-cluster shapes out of Python
and exposes:

  * hr_cluster_stats_jax_single(rho, rho_chain, eig_all, arch_assignment,
                                classes, X_obs, Y_obs, t, S)
      -> (P_obs, V, U, W, N_theta) for a single cluster of fixed size N.

  * hr_accumulate_batch(rho, rho_chain, eig_all, arch_assignment,
                          classes_batch, X_batch, Y_batch, t_batch, mask, S)
      -> (V_sum, U_sum, W_sum, N_theta_sum, T_sum, log_lik_sum) reduced over
      a padded batch, jit + vmapped over the batch axis.

The primitives closed-forms are those verified by
tests/dynfield/test_hr_stats_closed_form.py against Gillespie / compound-
CTMC references; only the packaging is changed.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# Field-chain primitives, jax-vectorised.
# ---------------------------------------------------------------------------


def field_bridge_stats(rho, rho_chain, theta_X, theta_Y, t):
    """Return (P_XY, E_T (L,), E_N_theta (scalar), E_arrivals (L,)) for the
    F81-on-DP field bridge. Closed forms from hr.py, expressed with
    jnp.where instead of python if/else so we can jit."""
    L = rho.shape[0]
    g_t = jnp.exp(-rho_chain * t)
    I1 = (1.0 - g_t) / rho_chain
    delta_XY = (theta_X == theta_Y).astype(jnp.float64)
    P_XY = g_t * delta_XY + (1.0 - g_t) * rho[theta_Y]

    # E_T[i] = num / P_XY.
    idx = jnp.arange(L)
    d_iY = (idx == theta_Y).astype(jnp.float64) - rho[theta_Y]
    d_Xi = (theta_X == idx).astype(jnp.float64) - rho
    num = (rho * rho[theta_Y] * t
             + (rho * d_iY + rho[theta_Y] * d_Xi) * I1
             + d_Xi * d_iY * t * g_t)
    E_T = num / jnp.maximum(P_XY, 1e-300)

    # E[N | bridge] via spectral collapse.
    sigma = jnp.sum(rho ** 2)
    Sum1 = rho_chain * t * rho[theta_Y] * (1.0 - sigma)
    Sum23 = (1.0 - g_t) * rho[theta_Y] * (2.0 * sigma - rho[theta_X] - rho[theta_Y])
    Sum4 = -rho_chain * g_t * t * (
        delta_XY * rho[theta_X] - rho[theta_X] * rho[theta_Y]
        - rho[theta_Y] ** 2 + rho[theta_Y] * sigma)
    EN_times_P = Sum1 + Sum23 + Sum4
    E_N_theta = EN_times_P / jnp.maximum(P_XY, 1e-300)

    # Per-destination arrivals.
    def arr(i):
        rho_i = rho[i]
        d_ai = (theta_X == i).astype(jnp.float64) - rho_i
        d_ib = (i == theta_Y).astype(jnp.float64) - rho[theta_Y]
        integrand = ((1.0 - rho_i) * rho[theta_Y] * t
                       + (1.0 - rho_i) * d_ib * I1
                       - d_ai * rho[theta_Y] * I1
                       - d_ai * d_ib * t * g_t)
        return rho_chain * rho_i * integrand / jnp.maximum(P_XY, 1e-300)
    E_arrivals = jax.vmap(arr)(idx)
    return P_XY, E_T, E_N_theta, E_arrivals


# ---------------------------------------------------------------------------
# GTR residue-chain primitives, jax-vectorised. eig = (xi (A,), U (A, A),
# D_half (A,)) per archetype, stacked to (K_a, A) / (K_a, A, A) / (K_a, A)
# for batched dispatch.
# ---------------------------------------------------------------------------


def gtr_eigendecomp_batch(pi_arch, S):
    """Return (xi (K_a, A), U (K_a, A, A), D_half (K_a, A)) for the K_a
    archetype rate matrices Q^k[i,j] = S[i,j] pi[k,j] for i!=j."""
    K_a, A = pi_arch.shape

    def _one(pi_k):
        # Q_k[i,j] for i!=j = S[i,j] pi_k[j]; diag = -row sum.
        Q_off = S * pi_k[None, :]
        Q_off = Q_off - jnp.diag(jnp.diag(Q_off))
        Q_diag = -Q_off.sum(axis=1)
        Q = Q_off + jnp.diag(Q_diag)
        D_half = jnp.sqrt(jnp.maximum(pi_k, 1e-300))
        Q_sym = (D_half[:, None] * Q) / D_half[None, :]
        Q_sym = 0.5 * (Q_sym + Q_sym.T)
        xi, U = jnp.linalg.eigh(Q_sym)
        return xi, U, D_half
    xi, U, D_half = jax.vmap(_one)(pi_arch)
    return xi, U, D_half


def _P_arch(xi, U, D_half, tau):
    """P^k(tau) = D_half[j]/D_half[i] * sum_alpha U[i,a] U[j,a] exp(xi_a tau).
    Given single-arch eig arrays."""
    P_sym = (U * jnp.exp(xi * tau)[None, :]) @ U.T
    return (P_sym * D_half[None, :]) / D_half[:, None]


def gtr_bridge_hr_jax(pi_k, xi, U_k, D_k, S, X, Y, t):
    """Bridge HR at a single site, for a specific archetype.
    Returns (P_XY, W_i (A,), U_ij (A, A))."""
    A = pi_k.shape[0]
    P_all = _P_arch(xi, U_k, D_k, t)
    P_XY = P_all[X, Y]
    exp_xi_t = jnp.exp(xi * t)
    # J[a, b] = (exp - exp) / (xi_a - xi_b) with L'Hopital limit.
    diff = xi[:, None] - xi[None, :]
    diff_safe = jnp.where(jnp.abs(diff) < 1e-12, 1.0, diff)
    J = jnp.where(jnp.abs(diff) < 1e-12,
                    t * exp_xi_t[:, None],
                    (exp_xi_t[:, None] - exp_xi_t[None, :]) / diff_safe)
    factor_Y_over_X = D_k[Y] / D_k[X]
    # W_i * P_XY = factor * sum_{alpha, beta} U[X, a] U[i, a] U[i, b] U[Y, b] J[a, b]
    UX = U_k[X, :]      # (A,)
    UY = U_k[Y, :]
    # term[i, a, b] = UX[a] U[i,a] U[i,b] UY[b] J[a,b]
    # sum over a, b -> per-i
    # We can rewrite: (UX * J).T   is (A_b, A_a), times (U[i,a] U[i,b]) reduced
    # Alternative: compute A_i = sum_a UX[a] U[i, a] * (sum_b U[i, b] UY[b] J[a, b])
    # Let M_ab = U[X, a] * UY[b] * J[a, b]  (A, A). Then W_i * P = factor * sum_{a,b} M[a,b] U[i,a] U[i,b].
    M_ab = UX[:, None] * UY[None, :] * J           # (A_a, A_b)
    # sum_{a,b} U[i,a] U[i,b] M[a,b] = U @ M @ U^T diag[i,i]
    UMUt = (U_k @ M_ab) @ U_k.T                     # (A_i, A_i)
    W_i = factor_Y_over_X * jnp.diag(UMUt) / jnp.maximum(P_XY, 1e-300)

    # U_ij = S[i,j] * D_k[i] * D_k[j] * factor_Y_over_X * s / P_XY
    # where s[i, j] = sum_{a, b} U[X, a] U[i, a] U[j, b] U[Y, b] J[a, b]
    # = (U @ (UX * J transposed)) @ ... hmm let's just do it
    # Actually: s[i, j] = U[i, :] @ (M_ab) @ U[j, :]^T? No.
    # s[i, j] = sum_a U[X, a] U[i, a] sum_b U[j, b] U[Y, b] J[a, b]
    # Let A_a_j = sum_b U[j, b] UY[b] J[a, b] = (U * UY[None, :]) @ J^T [j, a] hmm
    # Actually: let J_row[a] be a vector over b; then sum_b U[j, b] UY[b] J[a, b] = (U * UY)[j, :] @ J[a, :]
    # We do a 4-index tensor contract:
    #   s[i, j] = sum_{a, b} UX[a] U[i, a] U[j, b] UY[b] J[a, b]
    # via einsum
    s = jnp.einsum('a,ia,jb,b,ab->ij', UX, U_k, U_k, UY, J)
    U_ij = (S * D_k[:, None] * D_k[None, :] * factor_Y_over_X
              * s / jnp.maximum(P_XY, 1e-300))
    U_ij = U_ij - jnp.diag(jnp.diag(U_ij))
    return P_XY, W_i, U_ij


def gtr_free_end_hr_avg_case1_jax(pi_k, xi, U_k, D_k, S, X, t, delta):
    """W_i (dwell) and U_ij averaged over case-1 tau_1 on [0, t].
    delta = rate of the tau_1 truncated-exp density."""
    A = pi_k.shape[0]
    Z = jnp.where(jnp.abs(delta) < 1e-12, t, (jnp.exp(delta * t) - 1.0)
                    / jnp.where(jnp.abs(delta) < 1e-12, 1.0, delta))
    # E[h(xi_alpha, tau_1)] under p(tau_1) = exp(delta tau_1) / Z on [0, t]
    # For xi_alpha != 0:
    #   E[h(xi_a, tau_1)] = (1/(Z xi_a)) [ (exp((xi_a + delta) t) - 1)/(xi_a + delta) - Z ]
    # For xi_a == 0: E[tau_1 truncated exp] = E[tau_1].
    def _E_h(xa):
        xa_plus_d = xa + delta
        # Compute both branches and select
        # Branch xa == 0:
        E_tau1 = jnp.where(jnp.abs(delta) < 1e-12,
                              t / 2.0,
                              t * jnp.exp(delta * t) / (jnp.exp(delta * t) - 1.0)
                                - 1.0 / jnp.where(jnp.abs(delta) < 1e-12, 1.0, delta))
        # Branch xa != 0 and xa + delta != 0:
        val_generic = ((jnp.exp(xa_plus_d * t) - 1.0)
                          / jnp.where(jnp.abs(xa_plus_d) < 1e-12, 1.0, xa_plus_d)
                          - Z)
        val_generic = val_generic / (Z * jnp.where(jnp.abs(xa) < 1e-12, 1.0, xa))
        # Branch xa != 0 but xa + delta == 0 (L'Hopital):
        val_lhopital = (t + (jnp.exp(-xa * t) - 1.0)
                            / jnp.where(jnp.abs(xa) < 1e-12, 1.0, xa)) / (
                            Z * jnp.where(jnp.abs(xa) < 1e-12, 1.0, xa))
        return jnp.where(jnp.abs(xa) < 1e-12, E_tau1,
                            jnp.where(jnp.abs(xa_plus_d) < 1e-12, val_lhopital,
                                        val_generic))
    E_h = jax.vmap(_E_h)(xi)                        # (A,)
    factor = 1.0 / D_k[X]
    # W_i = factor * D_k[i] * sum_alpha U[X, a] U[i, a] E_h[a]
    W_i = factor * D_k * (U_k @ (U_k[X, :] * E_h))
    Q_off = S * pi_k[None, :]
    Q_off = Q_off - jnp.diag(jnp.diag(Q_off))
    U_ij = Q_off * W_i[:, None]
    return W_i, U_ij


def gtr_transition_avg_case1_jax(pi_k, xi, U_k, D_k, S, X, t, delta):
    """Target-state distribution E[P^k(X -> a; tau)] under case-1 density."""
    A = pi_k.shape[0]
    Z = jnp.where(jnp.abs(delta) < 1e-12, t,
                    (jnp.exp(delta * t) - 1.0)
                    / jnp.where(jnp.abs(delta) < 1e-12, 1.0, delta))
    exp_delta_t_over_Z = jnp.exp(delta * t) / jnp.maximum(Z, 1e-300)

    def _E_exp(xa):
        d = xa - delta
        d_safe = jnp.where(jnp.abs(d) < 1e-12, 1.0, d)
        return jnp.where(jnp.abs(d) < 1e-12,
                            exp_delta_t_over_Z * t,
                            exp_delta_t_over_Z * (jnp.exp(d * t) - 1.0) / d_safe)
    E_exp = jax.vmap(_E_exp)(xi)                    # (A,)
    # P_avg[a] = (D_k[a]/D_k[X]) sum_alpha U[X, a] U[a, alpha] E_exp[alpha]
    P_avg = (D_k / D_k[X]) * (U_k @ (U_k[X, :] * E_exp))
    return P_avg


# ---------------------------------------------------------------------------
# Cluster-level accumulator.
# ---------------------------------------------------------------------------


def _cluster_stats_inner(rho, rho_chain, pi_arch, xi, U_arch, D_arch,
                            arch_assignment, classes, X_obs, Y_obs,
                            site_mask, m_per_site, t, S, K_c):
    """Single-cluster stats, jit-friendly. Returns
    (P_obs, V (K_c, L, A), U (K_c, L, A, A), W (K_c, L, A), N_theta (scalar)).

    Per-(c, theta) HR sufficient stats -- callers aggregate over
    (c, theta) with the current arch_assignment to get per-archetype
    stats for the pi_arch M-step, and use V_ctheta directly to score
    the arch_assignment Gibbs conditional.

    m_per_site: (N,) per-site substitution rate multiplier for the
    dynfield +Γ+I persite extension (par:arch-gamma-plus-I-persite).
    Applied as xi_effective_n = m_per_site[n] * xi[k_n] in every
    substitution primitive call; m=0 encodes the invariant bin, in
    which case the residue is preserved (I[X=Y] in both no-jump and
    jump cases; jump-case SS contributions are zeroed). Pass ones(N)
    to disable per-site rate heterogeneity.
    """
    L = rho.shape[0]
    K_a, A = pi_arch.shape
    N = classes.shape[0]

    beta_t = jnp.exp(-rho_chain * (1.0 - rho) * t)

    # Precompute per-site archetype indices per theta.
    # k_by_theta[n, theta] = arch_assignment[classes[n], theta]
    k_by_theta = arch_assignment[classes]           # (N, L)
    # Per-site rate mask: 1 if variant (m > 0), 0 if invariant. Used to
    # zero out substitution SS contributions from invariant sites in the
    # M>=1 case (where the site's L_res is forced to I[X=Y]).
    site_variant = (m_per_site > 0.0).astype(jnp.float64)  # (N,)

    # Helper: scatter a per-site (A,) value into shape (K_c, L, A) at
    # position (c_n, theta_slot). Vectorised-friendly: uses outer products.
    def _scatter_A(c_n, theta_slot, val_A):
        oh_c = jnp.eye(K_c)[c_n]                     # (K_c,)
        oh_t = jnp.eye(L)[theta_slot]                # (L,)
        return oh_c[:, None, None] * oh_t[None, :, None] * val_A[None, None, :]

    def _scatter_AA(c_n, theta_slot, val_AA):
        oh_c = jnp.eye(K_c)[c_n]
        oh_t = jnp.eye(L)[theta_slot]
        return oh_c[:, None, None, None] * oh_t[None, :, None, None] * val_AA[None, None, :, :]

    # Sweep (theta_X, theta_Y).
    def _pair(theta_X, theta_Y):
        # Field-side bridge.
        P_XY, E_T, E_N_all, E_arr = field_bridge_stats(
            rho, rho_chain, theta_X, theta_Y, t)
        P_no_jump = jnp.where(theta_X == theta_Y, beta_t[theta_X], 0.0)
        P_jump = jnp.maximum(0.0, P_XY - P_no_jump)

        # ---- Case M=0 (only if theta_X == theta_Y).
        def _case0():
            def _site(n):
                c_n = classes[n]
                k_n = k_by_theta[n, theta_X]
                pi_kn = pi_arch[k_n]
                # Per-site substitution-rate scaling: xi_effective = m * xi.
                m_n = m_per_site[n]
                xi_n = m_n * xi[k_n]
                _, W_i, U_ij = gtr_bridge_hr_jax(
                    pi_kn, xi_n, U_arch[k_n], D_arch[k_n], S,
                    X_obs[n], Y_obs[n], t)
                # Per-site rate scaling of U (physical: U^{k, m} =
                # m·Q^k·W^{k, m}). See derivation: gtr_bridge_hr_jax with
                # xi_scaled = m·xi returns W_prim = W^{k, m}(T)
                # correctly, but U_prim = (1/m)·U^{k, m}(T) because the
                # eigenvalue-difference denominator in J acquires an
                # extra 1/m factor that only exp(xi·t) does not. So U
                # must be multiplied by m at the call site. Verified
                # analytically (J identity) and numerically via
                # Gillespie for A=4 DNA.
                V_c = _scatter_A(c_n, theta_X, jnp.eye(A)[X_obs[n]])
                W_c = _scatter_A(c_n, theta_X, W_i)
                U_c = _scatter_AA(c_n, theta_X, m_n * U_ij)
                P_res = _P_arch(xi_n, U_arch[k_n], D_arch[k_n], t)[X_obs[n], Y_obs[n]]
                return V_c, W_c, U_c, P_res
            V_sites, W_sites, U_sites, P_res_per_site = jax.vmap(_site)(
                jnp.arange(N))
            P_res_per_site = jnp.where(site_mask > 0.5, P_res_per_site, 1.0)
            L_res_0 = jnp.prod(jnp.maximum(P_res_per_site, 1e-300))
            weight_0 = rho[theta_X] * P_no_jump * L_res_0
            # site_mask (N,) applied over leading N axis (site).
            sm = site_mask[:, None, None, None]
            V0 = weight_0 * (V_sites * sm).sum(axis=0)   # (K_c, L, A)
            W0 = weight_0 * (W_sites * sm).sum(axis=0)
            U0 = weight_0 * (U_sites * site_mask[:, None, None, None, None]).sum(axis=0)
            return weight_0, V0, W0, U0
        weight_0, V0, W0, U0 = jax.lax.cond(
            theta_X == theta_Y, _case0,
            lambda: (0.0,
                        jnp.zeros((K_c, L, A)),
                        jnp.zeros((K_c, L, A)),
                        jnp.zeros((K_c, L, A, A))))

        # ---- Case M >= 1.
        def _case_jump():
            # Per-site residue-likelihood factor under Interp 1
            # ("resample from stationary" on jump). Applies uniformly to
            # variant *and* invariant per-site bins: an invariant-rate
            # site (m = 0) still resamples its residue at field jumps
            # (see par:arch-gamma-plus-I-persite; the "witness site"
            # interpretation). Between jumps the invariant site keeps
            # its residue -- that shows up in Case 0 where P^{k, m=0}
            # correctly collapses to I[X = Y] via the ξ · m = 0
            # eigenvalue scaling. Truly-100%-conserved sites are
            # captured through a different mechanism: a singleton
            # cluster whose per-cluster rate bin is invariant (no field
            # jumps in that cluster).
            def _res_j(n):
                k_Y_n = k_by_theta[n, theta_Y]
                return pi_arch[k_Y_n, Y_obs[n]]
            res_j = jax.vmap(_res_j)(jnp.arange(N))
            res_j = jnp.where(site_mask > 0.5, res_j, 1.0)
            L_res_j = jnp.prod(res_j)
            weight_j = rho[theta_X] * P_jump * L_res_j

            # E_T_jump[i] and E_arrivals_jump[i] (bridge conditional to M>=1).
            E_T_all_prod = E_T * P_XY
            E_T_all_prod = jnp.where(
                theta_X == theta_Y,
                E_T_all_prod.at[theta_X].add(-P_no_jump * t),
                E_T_all_prod)
            E_T_jump = E_T_all_prod / jnp.maximum(P_jump, 1e-300)
            E_N_jump = E_N_all * P_XY / jnp.maximum(P_jump, 1e-300)
            E_arrivals_jump = E_arr * P_XY / jnp.maximum(P_jump, 1e-300)

            # Exact M ≥ 1 τ_1 density (see appendix par:arch-hr):
            #   p(τ_1 | M≥1, θ_X, θ_Y, t) ∝ β_1 · exp(δ_1 · τ_1)
            #                              + β_2 · exp(δ_2 · τ_1)
            # with δ_1 = -r_X, δ_2 = ρ_chain · ρ[θ_X], and coefficients
            # β_1 = r_X · ρ[θ_Y],
            # β_2 = r_X · (ρ[θ_X] / (1 - ρ[θ_X])) · (ρ[θ_Y] - 𝟙[θ_X=θ_Y]) · g_T.
            # (Derivation: sum over intermediate θ_1 of jump prob times
            # continuation bridge P_field(θ_1 → θ_Y; t-τ_1), simplified via
            # stationarity Σ ρ[θ_1] P_field(θ_1 → θ_Y; s) = ρ[θ_Y]. Exact
            # under Interp 1 for the F81-on-DP field CTMC.)
            #
            # Same mixture structure for the last segment τ_M via time
            # reversal (swap θ_X ↔ θ_Y).
            #
            # Both terms are handled by gtr_free_end_hr_avg_case1_jax
            # and gtr_transition_avg_case1_jax with the appropriate δ;
            # we call each primitive twice and take the weighted average
            # a_1 · (·|δ_1) + a_2 · (·|δ_2) where a_i = β_i Z_i / Σ β_j Z_j.
            r_X = rho_chain * (1.0 - rho[theta_X])
            r_Y = rho_chain * (1.0 - rho[theta_Y])
            g_T = jnp.exp(-rho_chain * t)
            _same_XY = (theta_X == theta_Y).astype(jnp.float64)
            # First-segment mixture parameters.
            delta_1_f = -r_X
            delta_2_f = rho_chain * rho[theta_X]
            beta_1_f = r_X * rho[theta_Y]
            beta_2_f = (r_X * (rho[theta_X]
                                    / jnp.maximum(1.0 - rho[theta_X], 1e-300))
                          * (rho[theta_Y] - _same_XY) * g_T)
            # Last-segment mixture parameters (swap θ_X ↔ θ_Y).
            delta_1_l = -r_Y
            delta_2_l = rho_chain * rho[theta_Y]
            beta_1_l = r_Y * rho[theta_X]
            beta_2_l = (r_Y * (rho[theta_Y]
                                    / jnp.maximum(1.0 - rho[theta_Y], 1e-300))
                          * (rho[theta_X] - _same_XY) * g_T)

            def _Z(delta, T):
                """Z_δ = ∫_0^T exp(δτ) dτ, closed form with δ→0 limit T."""
                d_safe = jnp.where(jnp.abs(delta) < 1e-12, 1.0, delta)
                return jnp.where(jnp.abs(delta) < 1e-12, T,
                                    (jnp.exp(delta * T) - 1.0) / d_safe)

            def _E_tau(delta, T):
                """E[τ] under truncated-exp density p(τ)=exp(δτ)/Z on [0, T]."""
                d_safe = jnp.where(jnp.abs(delta) < 1e-12, 1.0, delta)
                exp_dT = jnp.exp(delta * T)
                return jnp.where(
                    jnp.abs(delta) < 1e-12,
                    T / 2.0,
                    T * exp_dT / (exp_dT - 1.0) - 1.0 / d_safe)

            Z_1_f = _Z(delta_1_f, t); Z_2_f = _Z(delta_2_f, t)
            Z_tot_f = jnp.maximum(beta_1_f * Z_1_f + beta_2_f * Z_2_f, 1e-300)
            a_1_f = beta_1_f * Z_1_f / Z_tot_f
            a_2_f = beta_2_f * Z_2_f / Z_tot_f

            Z_1_l = _Z(delta_1_l, t); Z_2_l = _Z(delta_2_l, t)
            Z_tot_l = jnp.maximum(beta_1_l * Z_1_l + beta_2_l * Z_2_l, 1e-300)
            a_1_l = beta_1_l * Z_1_l / Z_tot_l
            a_2_l = beta_2_l * Z_2_l / Z_tot_l

            # E[τ_1] and E[t - τ_M] under exact densities — used for
            # middle-segment time attribution below.
            E_tau1 = a_1_f * _E_tau(delta_1_f, t) + a_2_f * _E_tau(delta_2_f, t)
            E_tauM_to_t = (a_1_l * _E_tau(delta_1_l, t)
                              + a_2_l * _E_tau(delta_2_l, t))

            # Per site: first / last / middle contributions.
            def _site(n):
                c_n = classes[n]
                k_X_n = k_by_theta[n, theta_X]
                k_Y_n = k_by_theta[n, theta_Y]
                pi_kx = pi_arch[k_X_n]
                pi_kY = pi_arch[k_Y_n]
                # Per-site rate multiplier and eigenvalue scaling.
                # Under Q^{k, m} = m · Q^k, expected substitution jumps
                # over an interval are:  E[# jumps i→j] = m · Q^k[i,j] · W_i
                # (physical CTMC fact). All U primitives here — Case 0
                # bridge, Case ≥1 first/last free-end, and Case ≥1
                # middle-segment — compute the *unscaled* Q^k · W_i via
                # eigenvalue formulas that acquire an extra 1/m factor
                # from the J = (exp(m ξ_a t) − exp(m ξ_b t)) / (m(ξ_a −
                # ξ_b)) denominator. So we multiply U by m at every
                # attribution site. W (dwell time) does *not* need this
                # correction: the 1/m factor in J cancels the m
                # accumulated in exp(m·ξ·t), yielding the correct
                # m-scaled dwell. Verified algebraically (change-of-
                # variable s' = m·s) and numerically via Gillespie on
                # A=4 DNA.
                #
                # Consequence for invariant sites (m = 0): all U
                # contributions are naturally zero, matching "no
                # continuous substitution" without a special-case mask.
                m_n = m_per_site[n]
                xi_Xn = m_n * xi[k_X_n]
                xi_Yn = m_n * xi[k_Y_n]

                # V t=0 boundary at (c_n, theta_X, X_n).
                V_site = _scatter_A(c_n, theta_X, jnp.eye(A)[X_obs[n]])

                # V last-jump resample distribution at (c_n, theta_Y, ·).
                # Weighted mixture of two truncated-exp densities per the
                # exact M ≥ 1 τ_M density.
                P_avg_last_1 = gtr_transition_avg_case1_jax(
                    pi_kY, xi_Yn, U_arch[k_Y_n], D_arch[k_Y_n], S,
                    Y_obs[n], t, delta_1_l)
                P_avg_last_2 = gtr_transition_avg_case1_jax(
                    pi_kY, xi_Yn, U_arch[k_Y_n], D_arch[k_Y_n], S,
                    Y_obs[n], t, delta_2_l)
                P_avg_last = a_1_l * P_avg_last_1 + a_2_l * P_avg_last_2
                V_site = V_site + _scatter_A(c_n, theta_Y, P_avg_last)

                # First segment W, U at (c_n, theta_X, ·). Two calls,
                # weighted by the exact-density mixture coefficients.
                W_first_1, U_first_1 = gtr_free_end_hr_avg_case1_jax(
                    pi_kx, xi_Xn, U_arch[k_X_n], D_arch[k_X_n], S,
                    X_obs[n], t, delta_1_f)
                W_first_2, U_first_2 = gtr_free_end_hr_avg_case1_jax(
                    pi_kx, xi_Xn, U_arch[k_X_n], D_arch[k_X_n], S,
                    X_obs[n], t, delta_2_f)
                W_first = a_1_f * W_first_1 + a_2_f * W_first_2
                U_first = a_1_f * U_first_1 + a_2_f * U_first_2
                W_site = _scatter_A(c_n, theta_X, W_first)
                U_site = _scatter_AA(c_n, theta_X, m_n * U_first)

                # Last segment W, U at (c_n, theta_Y, ·). Two calls,
                # weighted by the last-segment mixture coefficients.
                W_last_1, _ = gtr_free_end_hr_avg_case1_jax(
                    pi_kY, xi_Yn, U_arch[k_Y_n], D_arch[k_Y_n], S,
                    Y_obs[n], t, delta_1_l)
                W_last_2, _ = gtr_free_end_hr_avg_case1_jax(
                    pi_kY, xi_Yn, U_arch[k_Y_n], D_arch[k_Y_n], S,
                    Y_obs[n], t, delta_2_l)
                W_last = a_1_l * W_last_1 + a_2_l * W_last_2
                W_site = W_site + _scatter_A(c_n, theta_Y, W_last)
                U_last_ij = pi_kY[:, None] * S * W_last[None, :]
                U_last_ij = U_last_ij - jnp.diag(jnp.diag(U_last_ij))
                U_site = U_site + _scatter_AA(c_n, theta_Y, m_n * U_last_ij)

                # Middle-segment contributions scatter to (c_n, theta_int, ·).
                def _mid_theta(theta_int):
                    k_i = k_by_theta[n, theta_int]
                    pi_k_i = pi_arch[k_i]
                    E_non_last = E_arrivals_jump[theta_int] - jnp.where(
                        theta_int == theta_Y, 1.0, 0.0)
                    E_non_last = jnp.maximum(E_non_last, 0.0)
                    V_int = _scatter_A(c_n, theta_int, E_non_last * pi_k_i)
                    first_time = jnp.where(theta_int == theta_X, E_tau1, 0.0)
                    last_time = jnp.where(theta_int == theta_Y, E_tauM_to_t, 0.0)
                    middle_time = jnp.maximum(0.0, E_T_jump[theta_int] - first_time - last_time)
                    W_int = _scatter_A(c_n, theta_int, middle_time * pi_k_i)
                    outer = pi_k_i[:, None] * pi_k_i[None, :]
                    U_int_ij = middle_time * S * outer
                    U_int_ij = U_int_ij - jnp.diag(jnp.diag(U_int_ij))
                    U_int = _scatter_AA(c_n, theta_int, m_n * U_int_ij)
                    return V_int, W_int, U_int
                V_mid_s, W_mid_s, U_mid_s = jax.vmap(_mid_theta)(jnp.arange(L))
                V_site = V_site + V_mid_s.sum(axis=0)
                W_site = W_site + W_mid_s.sum(axis=0)
                U_site = U_site + U_mid_s.sum(axis=0)

                return V_site, W_site, U_site
            V_s, W_s, U_s = jax.vmap(_site)(jnp.arange(N))
            # Under the witness-site convention (par:arch-gamma-plus-I-persite),
            # invariant-rate sites (m = 0) still emit via the Case >=1
            # resample formula L_res_j = π_arch(k(c, θ_Y), Y), so their
            # SS contribution to π_arch (boundary V at (c, θ_Y, Y) from
            # the resample event) is real and must not be zeroed. The
            # per-site substitution primitives receive xi_scaled = 0 for
            # invariant, which correctly makes W_i concentrate at the
            # starting residue and zeroes U_ij; the middle-segment
            # `pi_k_i * middle_time` attribution assumes fast
            # equilibration (Interp 1) and is a known approximation for
            # both invariant and variant sites -- kept consistent here.
            sm3 = site_mask[:, None, None, None]
            sm4 = site_mask[:, None, None, None, None]
            Vj = weight_j * (V_s * sm3).sum(axis=0)
            Wj = weight_j * (W_s * sm3).sum(axis=0)
            Uj = weight_j * (U_s * sm4).sum(axis=0)
            return weight_j, Vj, Wj, Uj, weight_j * E_N_jump
        weight_j, Vj, Wj, Uj, Nj = jax.lax.cond(
            P_jump > 0.0, _case_jump,
            lambda: (0.0,
                        jnp.zeros((K_c, L, A)),
                        jnp.zeros((K_c, L, A)),
                        jnp.zeros((K_c, L, A, A)),
                        0.0))

        P_pair = weight_0 + weight_j
        V_pair = V0 + Vj
        W_pair = W0 + Wj
        U_pair = U0 + Uj
        N_pair = Nj
        return P_pair, V_pair, W_pair, U_pair, N_pair

    # vmap over all L*L pairs and reduce.
    tx_grid, ty_grid = jnp.meshgrid(jnp.arange(L), jnp.arange(L), indexing='ij')
    P_grid, V_grid, W_grid, U_grid, N_grid = jax.vmap(
        jax.vmap(_pair, in_axes=(0, 0)),
        in_axes=(0, 0))(tx_grid, ty_grid)
    P_obs = jnp.sum(P_grid)
    V = jnp.sum(V_grid, axis=(0, 1))
    W = jnp.sum(W_grid, axis=(0, 1))
    U = jnp.sum(U_grid, axis=(0, 1))
    N_theta = jnp.sum(N_grid)
    return P_obs, V, U, W, N_theta


hr_cluster_stats_jit = jax.jit(_cluster_stats_inner,
                                    static_argnames=('K_c',))


# ---------------------------------------------------------------------------
# Batch accumulator: vmap over a padded cluster batch of a fixed N (sites)
# and reduce to corpus-level sums.
# ---------------------------------------------------------------------------


def _batch_reduce(rho, rho_chain, pi_arch, xi, U_arch, D_arch, arch_assignment,
                    classes_b, X_b, Y_b, site_mask_b, m_per_site_b,
                    t_b, mask_b, S, K_c):
    """Reduce over a batch of clusters. Returns per-(c, theta) HR stats:
      V (K_c, L, A), U (K_c, L, A, A), W (K_c, L, A)  -- summed over batch.
    plus scalars N_theta_sum, T_sum, log_lik_sum, P_obs_sum.

    m_per_site_b: (batch, N) per-site substitution rate multipliers for
    the +Γ+I persite extension. Pass jnp.ones((batch, N)) to disable.
    """

    def _one(classes, X, Y, site_mask, m_per_site, t, m):
        P_obs, V, U, W, N_theta = _cluster_stats_inner(
            rho, rho_chain, pi_arch, xi, U_arch, D_arch,
            arch_assignment, classes, X, Y, site_mask, m_per_site,
            t, S, K_c)
        # Divide by P_obs to get CONDITIONAL sums (EM M-step target).
        P_safe = jnp.maximum(P_obs, 1e-300)
        V_cond = V / P_safe
        U_cond = U / P_safe
        W_cond = W / P_safe
        N_theta_cond = N_theta / P_safe
        return (m * P_obs, m * V_cond, m * U_cond, m * W_cond,
                  m * N_theta_cond, m * t,
                  m * jnp.log(jnp.maximum(P_obs, 1e-300)))
    P_s, V_s, U_s, W_s, N_s, T_s, LL_s = jax.vmap(_one)(
        classes_b, X_b, Y_b, site_mask_b, m_per_site_b, t_b, mask_b)
    return (jnp.sum(V_s, axis=0),
              jnp.sum(U_s, axis=0),
              jnp.sum(W_s, axis=0),
              jnp.sum(N_s),
              jnp.sum(T_s),
              jnp.sum(LL_s),
              jnp.sum(P_s))


batch_reduce_jit = jax.jit(_batch_reduce, static_argnames=('K_c',))


# ---------------------------------------------------------------------------
# Numpy-friendly wrapper: bucket clusters by N (# sites), pad each bucket to
# a fixed shape, run the jitted batch reduce, sum across buckets.
# ---------------------------------------------------------------------------


def _sqrt2_buckets(max_N: int):
    """Return sqrt(2)-geometrically-spaced bucket sizes: 1, 2, 3, 4, 6, 8,
    12, 16, ... up to and including max_N."""
    import math
    buckets = []
    x = 1.0
    while True:
        b = int(math.ceil(x))
        if not buckets or b > buckets[-1]:
            buckets.append(b)
        if b >= max_N:
            break
        x *= math.sqrt(2)
    return buckets


def _bucket_for(n: int, buckets):
    for b in buckets:
        if b >= n:
            return b
    return buckets[-1]


def _bins_to_m(bins_array, bin_means_full):
    """Convert an integer bin-index array to a rate-multiplier array.
    bin_means_full[0] = 0 (invariant), bin_means_full[1..K_r_site] =
    Γ quantile means. Returns None if bins_array is None."""
    import numpy as _np
    if bins_array is None or bin_means_full is None:
        return None
    return bin_means_full[_np.asarray(bins_array, dtype=_np.int64)]


def accumulate_cluster_stats_hr_jax(model, cluster_observations, *,
                                       t_per_cluster,
                                       bins_per_cluster=None,
                                       bin_means_full=None,
                                       weight_per_cluster=None,
                                       S=None,
                                       batch_size: int = 512,
                                       enable_x64: bool = True):
    # Auto-shrink batch to keep GPU memory below ~8GB per bucket. The per-
    # cluster intermediate in the (θ_X, θ_Y) grid scales as
    # batch × L² × N × K_c × L × A², so for large N_bucket a big batch
    # blows the GPU allocator. Cap the effective batch by:
    #   ~ 8e9 bytes / (L² × N × K_c × L × A² × 8 bytes per float)
    def _auto_batch(N_bucket: int) -> int:
        import math
        try:
            _K_c, _L = model.dyn_field.arch_assignment.shape
            _A = model.dyn_field.pi_archetype.shape[1]
        except Exception:
            _K_c, _L, _A = 8, 4, 20
        target_bytes = 8e9  # ~8 GB soft cap on intermediates
        per_cluster_bytes = _L * _L * N_bucket * _K_c * _L * _A * _A * 8
        if per_cluster_bytes <= 0:
            return batch_size
        return max(1, min(batch_size, int(target_bytes // per_cluster_bytes)))
    """Corpus-level HR sufficient stat accumulation via jit + vmap.

    Clusters are padded up to a sqrt(2)-geometrically-spaced bucket size in
    N (site count), so a corpus with cluster sizes up to 256 shares a
    fixed set of ~17 compiled shapes (1, 2, 3, 4, 6, 8, 12, 16, 23, 32, 46,
    64, 91, 128, 181, 256) rather than the raw 256. Padded (dummy) sites in
    each cluster reuse the first observed site's classes/X/Y so no
    branching is needed inside the jit; padded slots inside a batch use
    mask=0 so they contribute nothing to the reductions. (Padding to a
    larger N inflates per-cluster compute but the compiled cache reuse
    across the corpus dominates in practice.)

    Returns {'V': (K_a, A), 'U': (K_a, A, A), 'W': (K_a, A),
             'N_theta_sum': float, 'T_sum': float, 'n_clust': int,
             'log_lik': float}.
    """
    import numpy as np
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
    dyn = model.dyn_field
    if getattr(dyn, 'pi_archetype', None) is None:
        raise ValueError("HR accumulator requires archetype variant")
    if S is None:
        from tkfdp.lg08 import S_LG08
        S = np.asarray(S_LG08, dtype=np.float64)
    S_j = jnp.asarray(S, dtype=jnp.float64)
    rho = jnp.asarray(dyn.rho, dtype=jnp.float64)
    rho_chain = jnp.float64(dyn.rho_chain)
    pi_arch = jnp.asarray(dyn.pi_archetype, dtype=jnp.float64)
    arch_assignment = jnp.asarray(dyn.arch_assignment, dtype=jnp.int32)

    # Precompute eigendecomps.
    xi, U_arch, D_arch = gtr_eigendecomp_batch(pi_arch, S_j)

    # Bucket clusters by sqrt(2)-spaced N.
    max_N = max(len(obs[0]) for obs in cluster_observations) if cluster_observations else 1
    buckets = _sqrt2_buckets(max_N)
    by_bucket: dict[int, list[int]] = {b: [] for b in buckets}
    for i, obs in enumerate(cluster_observations):
        classes, X_obs, Y_obs = obs
        N = int(len(classes))
        b = _bucket_for(N, buckets)
        by_bucket[b].append(i)

    K_a, A = pi_arch.shape
    K_c = int(arch_assignment.shape[0])
    L = int(rho.shape[0])
    V_total = np.zeros((K_c, L, A), dtype=np.float64)
    U_total = np.zeros((K_c, L, A, A), dtype=np.float64)
    W_total = np.zeros((K_c, L, A), dtype=np.float64)
    N_theta_total = 0.0
    T_total = 0.0
    log_lik_total = 0.0
    n_clust_total = 0
    for N_bucket, idxs in by_bucket.items():
        if not idxs:
            continue
        eff_batch = _auto_batch(N_bucket)
        # Build padded batches of `eff_batch` at a time.
        for start in range(0, len(idxs), eff_batch):
            batch_idxs = idxs[start:start + eff_batch]
            actual = len(batch_idxs)
            classes_b = np.zeros((eff_batch, N_bucket), dtype=np.int32)
            X_b = np.zeros((eff_batch, N_bucket), dtype=np.int32)
            Y_b = np.zeros((eff_batch, N_bucket), dtype=np.int32)
            site_mask_b = np.zeros((eff_batch, N_bucket), dtype=np.float64)
            # Default per-site rate multiplier = 1.0 (no per-site
            # heterogeneity). When bins_per_cluster is provided, per-site
            # m is looked up via bin_means_full[bin_id].
            m_per_site_b = np.ones((eff_batch, N_bucket), dtype=np.float64)
            t_b = np.ones(eff_batch, dtype=np.float64) * 0.1
            mask_b = np.zeros(eff_batch, dtype=np.float64)
            for j, ci in enumerate(batch_idxs):
                classes, X_obs, Y_obs = cluster_observations[ci]
                N_actual = int(len(classes))
                classes_b[j, :N_actual] = np.asarray(classes, dtype=np.int32)
                X_b[j, :N_actual] = np.asarray(X_obs, dtype=np.int32)
                Y_b[j, :N_actual] = np.asarray(Y_obs, dtype=np.int32)
                # Pad remaining slots with duplicates of site 0 so array
                # indices stay valid; site_mask=0 on padding drops those
                # sites from V/U/W and residue-likelihood products.
                for k in range(N_actual, N_bucket):
                    classes_b[j, k] = classes_b[j, 0]
                    X_b[j, k] = X_b[j, 0]
                    Y_b[j, k] = Y_b[j, 0]
                site_mask_b[j, :N_actual] = 1.0
                t_b[j] = float(t_per_cluster[ci])
                # mask_b doubles as the per-cluster importance weight
                # under the LG08 IS scheme (par:arch-lg08-is). Absent a
                # weight vector this is a plain 0/1 valid-cluster mask.
                mask_b[j] = (1.0 if weight_per_cluster is None
                              else float(weight_per_cluster[ci]))
                if bins_per_cluster is not None and bin_means_full is not None:
                    m_site = _bins_to_m(bins_per_cluster[ci], bin_means_full)
                    if m_site is not None:
                        m_per_site_b[j, :N_actual] = m_site[:N_actual]
            V_b, U_b, W_b, Nt_b, T_b, LL_b, _ = batch_reduce_jit(
                rho, rho_chain, pi_arch, xi, U_arch, D_arch,
                arch_assignment,
                jnp.asarray(classes_b), jnp.asarray(X_b), jnp.asarray(Y_b),
                jnp.asarray(site_mask_b), jnp.asarray(m_per_site_b),
                jnp.asarray(t_b),
                jnp.asarray(mask_b), S_j, K_c)
            V_total += np.asarray(V_b, dtype=np.float64)
            U_total += np.asarray(U_b, dtype=np.float64)
            W_total += np.asarray(W_b, dtype=np.float64)
            N_theta_total += float(Nt_b)
            T_total += float(T_b)
            log_lik_total += float(LL_b)
            n_clust_total += actual
    return {'V': V_total, 'U': U_total, 'W': W_total,
              'N_theta_sum': N_theta_total, 'T_sum': T_total,
              'n_clust': n_clust_total, 'log_lik': log_lik_total}


def _batch_reduce_percluster(rho, rho_chain, pi_arch, xi, U_arch, D_arch,
                                 arch_assignment, classes_b, X_b, Y_b,
                                 site_mask_b, m_per_site_b, t_b, mask_b,
                                 S, K_c):
    """Same as _batch_reduce but returns PER-CLUSTER arrays instead of
    corpus-summed. Used by the +Gamma+I loop that needs per-cluster
    log P_obs at each rate bin.

    m_per_site_b: (batch, N) per-site substitution rate multipliers.
    Pass ones to disable per-site heterogeneity.
    """
    def _one(classes, X, Y, site_mask, m_per_site, t, m):
        P_obs, V, U, W, N_theta = _cluster_stats_inner(
            rho, rho_chain, pi_arch, xi, U_arch, D_arch,
            arch_assignment, classes, X, Y, site_mask, m_per_site,
            t, S, K_c)
        P_safe = jnp.maximum(P_obs, 1e-300)
        V_cond = V / P_safe
        U_cond = U / P_safe
        W_cond = W / P_safe
        N_theta_cond = N_theta / P_safe
        return (m * P_obs, m * V_cond, m * U_cond, m * W_cond,
                  m * N_theta_cond, m * t,
                  m * jnp.log(jnp.maximum(P_obs, 1e-300)))
    return jax.vmap(_one)(classes_b, X_b, Y_b, site_mask_b, m_per_site_b,
                              t_b, mask_b)


batch_reduce_percluster_jit = jax.jit(_batch_reduce_percluster,
                                          static_argnames=('K_c',))


def accumulate_cluster_stats_hr_jax_gammaI(model, cluster_observations, *,
                                               t_per_cluster,
                                               bins_per_cluster=None,
                                               bin_means_full_site=None,
                                               weight_per_cluster=None,
                                               S=None,
                                               batch_size: int = 512,
                                               enable_x64: bool = True):
    """+Gamma+I corpus accumulator: runs the HR pass at each of K_r+1
    effective rho_chain values (invariant, plus K_r Gamma quantile bins)
    on the same set of clusters, then computes per-cluster bin posteriors
    q_q(b) and weight-aggregates V, U, W across bins.

    Requires model.dyn_field.K_rate_bins, alpha_gamma, p_inv to be set.

    Returns {'V', 'U', 'W', 'N_theta_sum', 'T_sum', 'n_clust', 'log_lik',
             'q_inv_mean', 'q_bin_mean', 'log_lik_by_bin_mean'}.
    """
    import numpy as np
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
    dyn = model.dyn_field
    if getattr(dyn, 'K_rate_bins', None) is None or dyn.K_rate_bins <= 0:
        raise ValueError("+Gamma+I accumulator requires K_rate_bins > 0")
    if S is None:
        from tkfdp.lg08 import S_LG08
        S = np.asarray(S_LG08, dtype=np.float64)
    S_j = jnp.asarray(S, dtype=jnp.float64)
    rho = jnp.asarray(dyn.rho, dtype=jnp.float64)
    rho_chain_base = float(dyn.rho_chain)
    pi_arch = jnp.asarray(dyn.pi_archetype, dtype=jnp.float64)
    arch_assignment = jnp.asarray(dyn.arch_assignment, dtype=jnp.int32)
    xi, U_arch, D_arch = gtr_eigendecomp_batch(pi_arch, S_j)

    K_r = int(dyn.K_rate_bins)
    alpha_g = float(dyn.alpha_gamma) if dyn.alpha_gamma else 1.0
    p_inv = float(dyn.p_inv) if dyn.p_inv is not None else 0.0
    bin_means = gamma_quantile_means(alpha_g, K_r)          # (K_r,)
    # Full bin set: [inv, γ_1, ..., γ_K_r].
    effective_rates = np.concatenate([[0.0], rho_chain_base * bin_means])
    # Prior weights.
    prior_p = np.concatenate([[p_inv],
                                np.full(K_r, (1.0 - p_inv) / K_r)])
    log_prior = np.log(np.maximum(prior_p, 1e-300))

    K_a, A = pi_arch.shape
    K_c = int(arch_assignment.shape[0])
    L = int(rho.shape[0])
    max_N = max(len(o[0]) for o in cluster_observations) if cluster_observations else 1
    buckets = _sqrt2_buckets(max_N)
    by_bucket: 'dict[int, list[int]]' = {b: [] for b in buckets}
    for i, o in enumerate(cluster_observations):
        by_bucket[_bucket_for(len(o[0]), buckets)].append(i)

    def _auto_batch(N_bucket: int) -> int:
        import math
        try:
            _K_c, _L = model.dyn_field.arch_assignment.shape
            _A = model.dyn_field.pi_archetype.shape[1]
        except Exception:
            _K_c, _L, _A = 8, 4, 20
        target_bytes = 8e9
        per_cluster_bytes = _L * _L * N_bucket * _K_c * _L * _A * _A * 8
        return max(1, min(batch_size, int(target_bytes // per_cluster_bytes)))

    V_total = np.zeros((K_c, L, A), dtype=np.float64)
    U_total = np.zeros((K_c, L, A, A), dtype=np.float64)
    W_total = np.zeros((K_c, L, A), dtype=np.float64)
    N_theta_total = 0.0
    T_total = 0.0
    log_lik_total = 0.0
    n_clust_total = 0
    q_inv_sum = 0.0
    q_bins_sum = np.zeros(K_r, dtype=np.float64)
    log_lik_by_bin_sum = np.zeros(K_r + 1, dtype=np.float64)

    for N_bucket, idxs in by_bucket.items():
        if not idxs:
            continue
        eff_batch = _auto_batch(N_bucket)
        for start in range(0, len(idxs), eff_batch):
            batch_idxs = idxs[start:start + eff_batch]
            actual = len(batch_idxs)
            classes_b = np.zeros((eff_batch, N_bucket), dtype=np.int32)
            X_b = np.zeros((eff_batch, N_bucket), dtype=np.int32)
            Y_b = np.zeros((eff_batch, N_bucket), dtype=np.int32)
            site_mask_b = np.zeros((eff_batch, N_bucket), dtype=np.float64)
            # Per-site rate multipliers (site +Γ+I). Ones = disabled.
            m_per_site_b = np.ones((eff_batch, N_bucket), dtype=np.float64)
            t_b = np.ones(eff_batch, dtype=np.float64) * 0.1
            mask_b = np.zeros(eff_batch, dtype=np.float64)
            for j, ci in enumerate(batch_idxs):
                classes, X_obs, Y_obs = cluster_observations[ci]
                N_actual = int(len(classes))
                classes_b[j, :N_actual] = np.asarray(classes, dtype=np.int32)
                X_b[j, :N_actual] = np.asarray(X_obs, dtype=np.int32)
                Y_b[j, :N_actual] = np.asarray(Y_obs, dtype=np.int32)
                for k in range(N_actual, N_bucket):
                    classes_b[j, k] = classes_b[j, 0]
                    X_b[j, k] = X_b[j, 0]
                    Y_b[j, k] = Y_b[j, 0]
                site_mask_b[j, :N_actual] = 1.0
                t_b[j] = float(t_per_cluster[ci])
                mask_b[j] = (1.0 if weight_per_cluster is None
                              else float(weight_per_cluster[ci]))
                if bins_per_cluster is not None and bin_means_full_site is not None:
                    m_site = _bins_to_m(bins_per_cluster[ci], bin_means_full_site)
                    if m_site is not None:
                        m_per_site_b[j, :N_actual] = m_site[:N_actual]

            # Run K_r+1 HR passes over this batch, one per bin.
            V_by_bin = np.zeros((K_r + 1, eff_batch, K_c, L, A), dtype=np.float64)
            U_by_bin = np.zeros((K_r + 1, eff_batch, K_c, L, A, A), dtype=np.float64)
            W_by_bin = np.zeros((K_r + 1, eff_batch, K_c, L, A), dtype=np.float64)
            N_by_bin = np.zeros((K_r + 1, eff_batch), dtype=np.float64)
            log_lik_by_bin = np.full((K_r + 1, eff_batch), -np.inf,
                                          dtype=np.float64)
            for bi, rate_bi in enumerate(effective_rates):
                out = batch_reduce_percluster_jit(
                    rho, jnp.float64(rate_bi), pi_arch, xi, U_arch, D_arch,
                    arch_assignment,
                    jnp.asarray(classes_b), jnp.asarray(X_b),
                    jnp.asarray(Y_b), jnp.asarray(site_mask_b),
                    jnp.asarray(m_per_site_b),
                    jnp.asarray(t_b), jnp.asarray(mask_b), S_j, K_c)
                # out: (P, V, U, W, N_theta, T, LL) each shape (eff_batch, ...)
                V_by_bin[bi] = np.asarray(out[1])
                U_by_bin[bi] = np.asarray(out[2])
                W_by_bin[bi] = np.asarray(out[3])
                N_by_bin[bi] = np.asarray(out[4])
                log_lik_by_bin[bi] = np.asarray(out[6])

            # Compute per-cluster bin posterior q_q(b).
            # log-score b for cluster q: log_prior[b] + log_lik_by_bin[b, q]
            # Only real clusters (mask=1) contribute to sums.
            log_score = log_prior[:, None] + log_lik_by_bin       # (K_r+1, eff_batch)
            log_score_max = np.max(log_score, axis=0, keepdims=True)
            unnorm = np.exp(log_score - log_score_max)
            q = unnorm / np.maximum(unnorm.sum(axis=0, keepdims=True), 1e-300)
            # Weight-average V, U, W over bins per cluster.
            # V_agg_cluster[q, c, θ, a] = sum_b q[b, q] * V_by_bin[b, q, c, θ, a]
            V_agg_batch = np.einsum('bq,bqcla->qcla', q, V_by_bin)
            U_agg_batch = np.einsum('bq,bqclij->qclij', q, U_by_bin)
            W_agg_batch = np.einsum('bq,bqcla->qcla', q, W_by_bin)
            N_agg_batch = np.einsum('bq,bq->q', q, N_by_bin)
            # Marginal log-lik per cluster = log sum_b p(b) P(obs|b).
            # Numerically stable: log_score_max + log(sum unnorm).
            log_lik_batch = (log_score_max[0]
                                + np.log(np.maximum(unnorm.sum(axis=0), 1e-300)))
            # Effective rate per cluster (for the rho_chain M-step denominator):
            # E[m | q_q] = sum_b q[b, q] * (0 if b==inv else bin_means[b-1])
            eff_rate_multipliers = np.concatenate([[0.0], bin_means])
            eff_rate_per_q = eff_rate_multipliers @ q               # (eff_batch,)

            # Accumulate corpus-level sums for real clusters only.
            for j in range(actual):
                V_total += V_agg_batch[j]
                U_total += U_agg_batch[j]
                W_total += W_agg_batch[j]
                N_theta_total += float(N_agg_batch[j])
                T_total += float(eff_rate_per_q[j] * t_b[j])
                log_lik_total += float(log_lik_batch[j])
                q_inv_sum += float(q[0, j])
                q_bins_sum += q[1:, j]
                for bi in range(K_r + 1):
                    log_lik_by_bin_sum[bi] += float(log_lik_by_bin[bi, j])
            n_clust_total += actual

    return {'V': V_total, 'U': U_total, 'W': W_total,
              'N_theta_sum': N_theta_total, 'T_sum': T_total,
              'n_clust': n_clust_total, 'log_lik': log_lik_total,
              'q_inv_mean': q_inv_sum / max(n_clust_total, 1),
              'q_bin_mean': q_bins_sum / max(n_clust_total, 1),
              'log_lik_by_bin_mean': log_lik_by_bin_sum
                                        / max(n_clust_total, 1),
              'bin_means': bin_means,
              'effective_rates': effective_rates}


# ---------------------------------------------------------------------------
# Gibbs sample of arch_assignment given per-(c, theta) HR sufficient stats.
# ---------------------------------------------------------------------------


def gibbs_sample_arch_from_hr(V_ctheta, U_ctheta, W_ctheta, pi_arch,
                                  rho_arch, S, rng, stochastic: bool = True):
    """Update arch_assignment[c, theta] from its HR-based conditional
        P(A[c, θ] = k | ·) ∝ ρ_arch[k] · exp( L_{c,θ,k}(pi_arch) )
    where L_{c,θ,k} is the per-(c, θ) complete-data trajectory
    log-likelihood contribution under archetype k:

        L_{c,θ,k}(pi) = Σ_a V'[c,θ,a] log pi_k[a]
                       − Σ_a r[c,θ,a] pi_k[a]  + (const in k)

    with V'[c, θ, a] = V[c, θ, a] + Σ_i U[c, θ, i, a] (state-entry count)
    and r[c, θ, a] = Σ_{i≠a} W[c, θ, i] · S[i, a]
    (dwell-weighted opportunity).

    stochastic=True: Gumbel-max Categorical sampling (stochastic-EM).
    stochastic=False: argmax (hard-EM). rng ignored in hard-EM branch.

    Returns arch_new (K_c, L_max) int32.
    """
    import numpy as np
    V = np.asarray(V_ctheta, dtype=np.float64)
    U = np.asarray(U_ctheta, dtype=np.float64)
    W = np.asarray(W_ctheta, dtype=np.float64)
    pi = np.asarray(pi_arch, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    log_rho_arch = np.log(np.maximum(np.asarray(rho_arch), 1e-300))

    K_c, L, A = V.shape
    K_a = pi.shape[0]
    # V'[c, θ, a] = V + sum_i U[·, i, a]
    Vprime = V + U.sum(axis=2)                    # (K_c, L, A)
    # r[c, θ, a] = sum_{i != a} W[c, θ, i] * S[i, a]
    # = W · (S − diag(S)); since S has zero diagonal (off-only convention),
    # r = W · S along the i axis.
    r = W @ S                                      # (K_c, L, A)
    # log pi (K_a, A)
    log_pi = np.log(np.maximum(pi, 1e-300))
    # For each (c, θ, k): compute L_{c, θ, k} = Σ_a V'[·, ·, a] log_pi[k, a]
    # − Σ_a r[·, ·, a] pi[k, a].
    # (K_c, L, K_a) via einsum.
    Vprime_flat = Vprime.reshape(K_c * L, A)       # (K_c*L, A)
    r_flat = r.reshape(K_c * L, A)
    log_score = Vprime_flat @ log_pi.T             # (K_c*L, K_a)
    log_score = log_score - (r_flat @ pi.T)        # subtract dwell penalty
    # Add prior.
    log_score = log_score + log_rho_arch[None, :]  # (K_c*L, K_a)
    if stochastic:
        g = rng.gumbel(size=log_score.shape)
        arch_flat = np.argmax(log_score + g, axis=1)
    else:
        arch_flat = np.argmax(log_score, axis=1)
    return arch_flat.astype(np.int32).reshape(K_c, L)


def gamma_quantile_means(alpha: float, K: int) -> 'np.ndarray':
    """Return the K equal-probability quantile means of a Gamma(alpha, alpha)
    distribution (mean 1, variance 1/alpha), following Yang 1994/1996.
    Used as the discretised rate multipliers m_1, ..., m_K.
    """
    import numpy as np
    from scipy.stats import gamma as _gamma
    # Cut points at k/K quantiles for k in 1..K-1.
    edges = _gamma.ppf(np.arange(1, K) / K, a=alpha, scale=1.0 / alpha)
    # Boundary edges at 0 and inf; use ppf(0)=0 and ppf(1)=inf handled below.
    lo = np.concatenate([[0.0], edges])
    hi = np.concatenate([edges, [np.inf]])
    # Bin mean under Gamma(alpha, alpha): E[X | lo <= X < hi] = alpha * (F_{alpha+1}(hi) - F_{alpha+1}(lo)) / P(bin)
    # where F_{alpha+1} is the Gamma CDF with shape alpha+1.
    Fhi = _gamma.cdf(hi, a=alpha + 1.0, scale=1.0 / alpha)
    Flo = _gamma.cdf(lo, a=alpha + 1.0, scale=1.0 / alpha)
    # P(bin) = 1/K by construction.
    means = (Fhi - Flo) * K
    return means


def compute_arch_probs_from_hr(V_ctheta, U_ctheta, W_ctheta, pi_arch,
                                   rho_arch, S):
    """Compute the mean-field posterior arch_probs[c, θ, k] under the
    HR-based conditional:
        log q(A[c, θ] = k) ∝ log ρ_arch[k] + L_{c,θ,k}(pi_arch)
    Normalised via softmax over k.  Deterministic; no sampling.

    Returns arch_probs (K_c, L, K_a).
    """
    import numpy as np
    from scipy.special import logsumexp
    V = np.asarray(V_ctheta, dtype=np.float64)
    U = np.asarray(U_ctheta, dtype=np.float64)
    W = np.asarray(W_ctheta, dtype=np.float64)
    pi = np.asarray(pi_arch, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    log_rho_arch = np.log(np.maximum(np.asarray(rho_arch), 1e-300))

    K_c, L, A = V.shape
    K_a = pi.shape[0]
    Vprime = V + U.sum(axis=2)                      # (K_c, L, A)
    r = W @ S                                        # (K_c, L, A)
    log_pi = np.log(np.maximum(pi, 1e-300))

    Vprime_flat = Vprime.reshape(K_c * L, A)
    r_flat = r.reshape(K_c * L, A)
    log_score = Vprime_flat @ log_pi.T               # (K_c*L, K_a)
    log_score = log_score - (r_flat @ pi.T)
    log_score = log_score + log_rho_arch[None, :]

    # Softmax stabilised via logsumexp.
    log_norm = logsumexp(log_score, axis=1, keepdims=True)
    arch_probs = np.exp(log_score - log_norm)         # (K_c*L, K_a)
    return arch_probs.reshape(K_c, L, K_a)


def aggregate_by_arch_soft(V_ctheta, U_ctheta, W_ctheta, arch_probs, K_a):
    """Soft-EM aggregation: per-arch V, U, W are weighted sums of per-(c, θ)
    stats via arch_probs (mean-field posterior on A)."""
    import numpy as np
    K_c, L, A = V_ctheta.shape
    ap = np.asarray(arch_probs, dtype=np.float64)   # (K_c, L, K_a)
    # V[k, a] = sum_{c, theta} ap[c, theta, k] * V_ctheta[c, theta, a]
    V = np.einsum('clk,cla->ka', ap, V_ctheta)
    W = np.einsum('clk,cla->ka', ap, W_ctheta)
    U = np.einsum('clk,clij->kij', ap, U_ctheta)
    return V, U, W


def aggregate_by_arch(V_ctheta, U_ctheta, W_ctheta, arch_assignment, K_a):
    """Given per-(c, θ) HR stats and an arch_assignment, aggregate to
    per-archetype sums for the M-step.  V (K_a, A), U (K_a, A, A),
    W (K_a, A)."""
    import numpy as np
    K_c, L, A = V_ctheta.shape
    # scatter K_c * L → K_a
    arch_flat = np.asarray(arch_assignment).reshape(-1).astype(np.int64)
    V = np.zeros((K_a, A), dtype=np.float64)
    U = np.zeros((K_a, A, A), dtype=np.float64)
    W = np.zeros((K_a, A), dtype=np.float64)
    for idx in range(K_c * L):
        k = int(arch_flat[idx])
        c = idx // L; th = idx % L
        V[k] += V_ctheta[c, th]
        U[k] += U_ctheta[c, th]
        W[k] += W_ctheta[c, th]
    return V, U, W


# ---------------------------------------------------------------------------
# Per-site bin Rao-Blackwellised Gibbs sampler
# (par:arch-gamma-plus-I-persite in appendix-tkfdp.tex).
# ---------------------------------------------------------------------------


def _eigendecomp_np(pi_arch, S):
    import numpy as _np
    K_a, A = pi_arch.shape
    xi = _np.zeros((K_a, A))
    U = _np.zeros((K_a, A, A))
    D_half = _np.zeros((K_a, A))
    for k in range(K_a):
        pi_k = pi_arch[k]
        Q_off = S * pi_k[None, :]
        Q_off = Q_off - _np.diag(_np.diag(Q_off))
        Q_diag = -Q_off.sum(axis=1)
        Q = Q_off + _np.diag(Q_diag)
        D_half[k] = _np.sqrt(_np.maximum(pi_k, 1e-300))
        Q_sym = (D_half[k][:, None] * Q) / D_half[k][None, :]
        Q_sym = 0.5 * (Q_sym + Q_sym.T)
        xi[k], U[k] = _np.linalg.eigh(Q_sym)
    return xi, U, D_half


def _logsumexp_np(x):
    import numpy as _np
    m = _np.max(x)
    if not _np.isfinite(m):
        return m
    return m + _np.log(_np.sum(_np.exp(x - m)))


def gibbs_sample_persite_bins(model, cluster_observations, t_per_cluster,
                                 bins_per_cluster, bin_means_full,
                                 log_prior, rng, *, S=None,
                                 allow_invariant: bool = False):
    """Rao-Blackwellised per-site bin Gibbs step; see
    par:arch-gamma-plus-I-persite (eq:arch-persite-Gibbs).

    Iterates over sites within each cluster sequentially. For each
    site n and candidate bin b:
      LL_n(b) = logsumexp_pair( log_A_pair(b at n) OR log_B_pair(b at n) )
    where log_A_pair captures the no-jump case (diagonal in θ_X = θ_Y)
    and log_B_pair captures the jump case (θ_X != θ_Y and θ_X == θ_Y
    with intermediate jumps). Posterior over bins is
      P(b_n = b | rest) ∝ Prior(b) · exp(LL_n(b)).

    Invariant bin (bin index 0): under the "witness-site" convention
    (par:arch-gamma-plus-I-persite), an invariant-rate site has no
    continuous substitution but still resamples its residue at each
    field jump from stationary at θ_Y (same Case ≥1 emission as
    variant bins). Case 0 (no jump) gives I[X = Y] via the m = 0
    eigenvalue collapse to identity. This makes invariant a safe
    per-cherry Gibbs option — it never drives the cluster likelihood
    to zero. Truly-100%-conserved sites are captured via a *different*
    mechanism: a singleton cluster whose per-cluster rate bin is
    invariant (no field jumps in that cluster). Set
    `allow_invariant=False` to force Γ-only sampling for A/B
    comparison against the pre-change Γ-only model.

    Args:
      model: DynamicFieldCouplingModel (reads pi_arch, arch_assignment,
        rho, rho_chain).
      cluster_observations: list of (classes, X_obs, Y_obs) tuples (the
        `c_obs` used by the HR pass).
      t_per_cluster: (Q,) branch length per cluster.
      bins_per_cluster: list of per-cluster (N,) int arrays with values
        in {0..K_r_site} (bin 0 = invariant).
      bin_means_full: (K_r_site + 1,) array; [0] = 0.0 (invariant),
        [1..K_r_site] = Γ quantile means.
      log_prior: (K_r_site + 1,) log-prior over bins.
      rng: np.random.Generator.
      S: (A, A) exchangeability matrix; defaults to LG08.

    Returns: new bins_per_cluster (list of new (N,) int32 arrays).
    """
    import numpy as _np
    from tkfdp.lg08 import S_LG08
    if S is None:
        S = _np.asarray(S_LG08, dtype=_np.float64)
    dyn = model.dyn_field
    rho = _np.asarray(dyn.rho, dtype=_np.float64)
    rho_chain = float(dyn.rho_chain)
    pi_arch = _np.asarray(dyn.pi_archetype, dtype=_np.float64)
    arch_assignment = _np.asarray(dyn.arch_assignment, dtype=_np.int32)
    L = int(rho.shape[0])
    K_bins = int(bin_means_full.shape[0])
    NEG_INF = -1e18

    # Precompute per-archetype eigendecomp (cached across all clusters).
    xi, U_arch, D_arch = _eigendecomp_np(pi_arch, S)

    new_bins = []
    for (obs_tuple, bins_cur, tau) in zip(
            cluster_observations, bins_per_cluster, t_per_cluster):
        classes, X_obs, Y_obs = obs_tuple
        classes = _np.asarray(classes, dtype=_np.int32)
        X_obs = _np.asarray(X_obs, dtype=_np.int32)
        Y_obs = _np.asarray(Y_obs, dtype=_np.int32)
        bins_new = _np.asarray(bins_cur, dtype=_np.int32).copy()
        N = int(len(classes))
        if N == 0:
            new_bins.append(bins_new)
            continue
        t = float(tau)

        # Field weights.
        beta_t = _np.exp(-rho_chain * (1.0 - rho) * t)  # (L,)
        g_t = float(_np.exp(-rho_chain * t))
        # P_field[tx, ty] = g_t·δ + (1-g_t)·ρ[ty].
        P_field = (g_t * _np.eye(L)
                     + (1.0 - g_t) * rho[None, :] * _np.ones((L, 1)))
        P_no_jump = _np.diag(beta_t)
        P_jump = _np.maximum(P_field - P_no_jump, 0.0)
        # Weights for Case 0 (diagonal only) and Case >=1.
        log_w_A_diag = _np.log(_np.maximum(rho * beta_t, 1e-300))  # (L,)
        log_w_B = _np.log(_np.maximum(rho[:, None] * P_jump, 1e-300))  # (L, L)

        # Per-site per-θ per-bin emissions.
        # log_P0[n, θ, b] = log P^{k_n(θ), m_b}(X_n, Y_n; t)  (Case 0)
        # log_LJ[n, θ, b] = log residue-emit under Case >=1 at θ_Y = θ, bin b
        #   variant: log pi_arch[k_n(θ), Y_n]; invariant: log I[X_n = Y_n]
        k_by_theta = arch_assignment[classes]                 # (N, L)
        log_P0 = _np.full((N, L, K_bins), NEG_INF)
        log_LJ = _np.full((N, L, K_bins), NEG_INF)
        for n in range(N):
            Xn = int(X_obs[n]); Yn = int(Y_obs[n])
            X_eq_Y = (Xn == Yn)
            for th in range(L):
                k = int(k_by_theta[n, th])
                pi_kY = float(pi_arch[k, Yn])
                log_pi_kY = _np.log(max(pi_kY, 1e-300))
                for b in range(K_bins):
                    m_b = float(bin_means_full[b])
                    if m_b == 0.0:  # invariant
                        # Case 0 (no jump): residue preserved -> I[X = Y].
                        log_P0[n, th, b] = 0.0 if X_eq_Y else NEG_INF
                        # Case >=1 (jumps): residue resamples at each jump
                        # from stationary at θ_Y (witness-site model;
                        # par:arch-gamma-plus-I-persite). Same formula as
                        # variant bins: log π_arch[k(c, θ_Y), Y].
                        log_LJ[n, th, b] = log_pi_kY
                    else:
                        # P^{k, m_b}[X, Y] via eigenvalue scaling.
                        exp_xi_t = _np.exp(m_b * xi[k] * t)
                        P_sym = (U_arch[k] * exp_xi_t[None, :]) @ U_arch[k].T
                        P_XY = (D_arch[k][Yn] / max(D_arch[k][Xn], 1e-300)
                                    * P_sym[Xn, Yn])
                        log_P0[n, th, b] = _np.log(max(P_XY, 1e-300))
                        log_LJ[n, th, b] = log_pi_kY

        # Sequential Gibbs sweep over sites in this cluster.
        # Per-θ sum of Case 0 site log-emissions at current bins.
        sum_log_P0 = _np.zeros(L)  # over θ
        sum_log_LJ = _np.zeros(L)  # over θ_Y
        for n in range(N):
            b_n = int(bins_new[n])
            sum_log_P0 += log_P0[n, :, b_n]
            sum_log_LJ += log_LJ[n, :, b_n]

        for n_target in range(N):
            b_cur = int(bins_new[n_target])
            # Contribution of "others" at each θ.
            other_log_P0 = sum_log_P0 - log_P0[n_target, :, b_cur]  # (L,)
            other_log_LJ = sum_log_LJ - log_LJ[n_target, :, b_cur]  # (L,)

            LL_n = _np.full(K_bins, NEG_INF)
            for b in range(K_bins):
                # log_A[tx, ty] non-inf only on diagonal.
                log_A_diag = (log_w_A_diag + other_log_P0
                                + log_P0[n_target, :, b])  # (L,)
                # log_B[tx, ty] = log_w_B[tx, ty] + other_log_LJ[ty]
                #                + log_LJ[n_target, ty, b]
                addend_ty = other_log_LJ + log_LJ[n_target, :, b]  # (L,)
                log_B = log_w_B + addend_ty[None, :]  # (L, L)
                # Combine: diagonal is logaddexp(log_A_diag, log_B_diag);
                # off-diagonal is log_B.
                log_P_pair = log_B.copy()
                for tx in range(L):
                    log_P_pair[tx, tx] = _np.logaddexp(
                        log_A_diag[tx], log_B[tx, tx])
                LL_n[b] = _logsumexp_np(log_P_pair.reshape(-1))

            # Posterior sample.
            log_post = log_prior + LL_n
            if not allow_invariant:
                # Mask out invariant (bin 0). See docstring: bin-0 assignment
                # is inconsistent across cherries at the same column when
                # applied per-cluster.
                log_post[0] = -_np.inf
            else:
                # Per-cherry consistency check for the invariant bin.
                # Bins are persisted per-column, but the Gibbs iterates
                # per-(cluster, cherry) — the same column can be visited
                # by multiple cherries with different (X_n, Y_n) pairs.
                # If this cherry has X_n != Y_n at this site, assigning
                # invariant here (bin 0) will kill this cluster's Case 0
                # factor (L_res_0 = I[X = Y] = 0) even under the
                # witness-site convention; that is a hard local cost.
                # Only permit invariant on cherries where X_n == Y_n at
                # this site — the strict per-cherry compatibility rule.
                # Other cherries at the same column will not clobber a
                # consistent bin-0 assignment because they would leave
                # a non-invariant bin standing under overwrite semantics.
                if int(X_obs[n_target]) != int(Y_obs[n_target]):
                    log_post[0] = -_np.inf
            log_post -= _logsumexp_np(log_post)
            p = _np.exp(log_post)
            s = float(p.sum())
            if s <= 0.0 or not _np.isfinite(s):
                # Fall back to uniform over the Γ bins.
                p = _np.zeros(K_bins)
                p[1:] = 1.0 / max(K_bins - 1, 1)
                s = 1.0
            p = p / s
            b_new = int(rng.choice(K_bins, p=p))

            # Update running sums to reflect the swap.
            sum_log_P0 += log_P0[n_target, :, b_new] - log_P0[n_target, :, b_cur]
            sum_log_LJ += log_LJ[n_target, :, b_new] - log_LJ[n_target, :, b_cur]
            bins_new[n_target] = b_new

        new_bins.append(bins_new)
    return new_bins
