"""Closed-form Holmes-Rubin sufficient statistics for the hierarchical
archetype variant of the dynfield model.

Derivation and notation follow math-paper/appendix-tkfdp.tex
sec:archetype-suppl. Provides:

  - Field-chain HR primitives (F81-on-DP with rate rho_chain):
      field_case_probs             : P(M=0), P(M=1, theta_1) per theta_X
      field_expected_dwell_jumps   : E[T_i], E[N_theta] given endpoints

  - GTR residue-chain HR primitives (per archetype eigendecomposition):
      gtr_eigendecomp              : symmetrised eigendecomp of Q^k
      gtr_bridge_hr                : E[W_i], E[U_ij] given (X, Y, t)
      gtr_free_end_hr              : W_i, U_ij at time tau, no end cond'g
      gtr_free_end_hr_avg          : ditto averaged over case-1 tau_1 density

  - Cluster-level accumulation:
      hr_cluster_stats             : (V[k, a], U[k, i, j], W[k, i]) per site

Verified against direct spectral evaluation of the compound
CTMC on state space F * A^N in tests/dynfield/test_hr_stats_closed_form.py.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Field chain: F81-on-DP with rate rho_chain, stationary rho.
# Rank-1 Q_theta = rho_chain (1 rho^T - I) with spectrum {0, -rho_chain}.
# ---------------------------------------------------------------------------

def field_case_probs(rho: np.ndarray, rho_chain: float, theta_X: int,
                       t: float) -> 'tuple[float, np.ndarray, float]':
    """Case probabilities P(M=0), P(M=1, theta_1) for each candidate
    theta_1 != theta_X, and P(M>=2) as residual.

    Uses the closed forms from appendix eq:arch-P0 and eq:arch-P1-marg,
    with the L'Hopital limit at rho[theta_1] == rho[theta_X].
    """
    L_max = int(rho.shape[0])
    r = rho_chain * (1.0 - rho)                    # exit rates r_i
    beta_t = np.exp(-r * t)                         # beta_t(i)

    P0 = float(beta_t[theta_X])

    # P(M=1, theta_1 = j | theta_X, t) marginalising over tau_1
    P1 = np.zeros(L_max, dtype=np.float64)
    beta_X = beta_t[theta_X]
    rho_X = float(rho[theta_X])
    for j in range(L_max):
        if j == theta_X:
            continue
        drho = float(rho[j] - rho_X)
        if abs(drho) < 1e-12:
            # L'Hopital limit
            P1[j] = float(rho[j]) * rho_chain * t * beta_X
        else:
            P1[j] = float(rho[j]) * (beta_t[j] - beta_X) / drho

    P1_sum = float(P1.sum())
    P2plus = max(0.0, 1.0 - P0 - P1_sum)
    return P0, P1, P2plus


def field_expected_dwell_jumps(rho: np.ndarray, rho_chain: float,
                                  theta_X: int, theta_Y: int,
                                  t: float) -> 'tuple[float, np.ndarray, float]':
    """Bridged HR for the field chain given endpoints (theta_X, theta_Y)
    and branch length t.

    Returns:
      P_total   : P(theta_Y | theta_X, t)
      E_T[i]    : E[dwell in field state i | theta_X, theta_Y, t]
      E_N_theta : E[# field jumps | theta_X, theta_Y, t]

    E_T[i] uses the standard bridge integral
      int_0^t P_uncond(a -> i, s) * P_uncond(i -> b, t-s) ds / P_XY,
    which under the rank-1 spectral form P(a -> b, tau) =
    rho[b] + g_tau * (delta_{a,b} - rho[b]) collapses to a closed form.

    E_N_theta uses the correct bridge-conditional spectral integral
      E[N | bridge] * P_XY = sum_{i != j} Q[i, j] * int P(a -> i, s)
                                              * P(j -> b, t-s) ds,
    which is *not* equal to rho_chain * sum (1 - rho) * E_T[bridge]
    (that unconditional-rate * conditional-dwell shortcut is only valid
    when the trajectory is at field-stationarity throughout, i.e. when
    both bridge endpoints are drawn from rho). Under the rank-1 spectral
    form the sum-integral collapses to a closed form in
    {rho, rho_chain, t, g_t, sigma := sum rho^2}. See derivation and
    Gillespie verification in tests/dynfield/test_hr_stats_closed_form.py.
    """
    L_max = int(rho.shape[0])
    g_t = np.exp(-rho_chain * t)

    delta_XY = 1.0 if theta_X == theta_Y else 0.0
    P_total = g_t * delta_XY + (1.0 - g_t) * float(rho[theta_Y])

    # E[T_i | bridge]: closed-form spectral integral (unchanged, correct).
    E_T = np.zeros(L_max, dtype=np.float64)
    I1 = (1.0 - g_t) / rho_chain if rho_chain > 0 else t
    for i in range(L_max):
        d_iY = (1.0 if i == theta_Y else 0.0) - float(rho[theta_Y])
        d_Xi = (1.0 if theta_X == i else 0.0) - float(rho[i])
        num = (float(rho[i]) * float(rho[theta_Y]) * t
                + (float(rho[i]) * d_iY + float(rho[theta_Y]) * d_Xi) * I1
                + d_Xi * d_iY * t * g_t)
        E_T[i] = num / max(P_total, 1e-300)

    # E[N | bridge (a, b)] via spectral collapse. Derivation: substitute
    # P(u -> v, tau) = rho[v] + g_tau * (delta_{uv} - rho[v]) into
    #   Q[i,j] = rho_chain * rho[j]  (i != j)
    # and integrate  Q[i,j] * P(a -> i, s) * P(j -> b, t-s)  ds over
    # s in [0, t], summing over i != j.  Four term types survive:
    #
    #   Sum1: rho_chain * t * rho[b] * (1 - sigma)                 [const]
    #   Sum2+3: rho_chain * I1 * rho[b] * (2 sigma - rho[a] - rho[b])
    #   Sum4: -rho_chain * g_t * t * (delta_{a,b} * rho[a]
    #             - rho[a] * rho[b] - rho[b]^2 + rho[b] * sigma)
    #
    # where sigma := sum_i rho[i]^2 and I1 := (1 - g_t) / rho_chain.
    # Then E[N | bridge] * P_XY = Sum1 + Sum2+3 + Sum4.
    a = int(theta_X); b = int(theta_Y)
    sigma = float(np.sum(np.asarray(rho, dtype=np.float64) ** 2))
    rho_a = float(rho[a]); rho_b = float(rho[b])
    Sum1 = rho_chain * t * rho_b * (1.0 - sigma)
    Sum23 = rho_chain * I1 * rho_b * (2.0 * sigma - rho_a - rho_b)
    Sum4 = -rho_chain * g_t * t * (
        delta_XY * rho_a - rho_a * rho_b - rho_b * rho_b + rho_b * sigma)
    EN_times_P = Sum1 + Sum23 + Sum4
    E_N_theta = EN_times_P / max(P_total, 1e-300)

    return float(P_total), E_T, float(E_N_theta)


def field_expected_arrivals(rho: np.ndarray, rho_chain: float,
                              theta_X: int, theta_Y: int,
                              t: float) -> np.ndarray:
    """Bridge-conditional E[# arrivals at each field state theta_int | (theta_X,
    theta_Y), t] as an (L,) array.  Sums to E[N_theta | bridge] over theta_int.

    Derivation (analogous to E_N_theta but not marginalised over destination):
      E[N_int | (a, b), t] * P_XY = sum_{j != int} Q[j, int]
                                    * int P(a -> j, s) * P(int -> b, t-s) ds
                                  = rho_chain * rho[int]
                                    * int (1 - P(a -> int, s)) * P(int -> b, t-s) ds
    Under the rank-1 spectral form P(u -> v, tau) = rho[v] + g_tau (delta_{uv} - rho[v]),
    the integral collapses to a closed form in {rho, g_t, t, rho_chain,
    delta_{a, int}, delta_{int, b}}. Sum over theta_int matches the
    aggregate E_N_theta returned by field_expected_dwell_jumps.

    Verified against Gillespie in tests/dynfield/test_hr_stats_closed_form.py.
    """
    L_max = int(rho.shape[0])
    g_t = np.exp(-rho_chain * t)
    I1 = (1.0 - g_t) / rho_chain if rho_chain > 0 else t
    delta_XY = 1.0 if theta_X == theta_Y else 0.0
    P_XY = g_t * delta_XY + (1.0 - g_t) * float(rho[theta_Y])
    rho_b = float(rho[theta_Y])

    E_arr = np.zeros(L_max, dtype=np.float64)
    for i in range(L_max):
        rho_i = float(rho[i])
        d_ai = (1.0 if theta_X == i else 0.0) - rho_i
        d_ib = (1.0 if i == theta_Y else 0.0) - rho_b
        integrand = ((1.0 - rho_i) * rho_b * t
                       + (1.0 - rho_i) * d_ib * I1
                       - d_ai * rho_b * I1
                       - d_ai * d_ib * t * g_t)
        E_arr[i] = rho_chain * rho_i * integrand / max(P_XY, 1e-300)
    return E_arr


# ---------------------------------------------------------------------------
# GTR residue chain per archetype: Q^k(y, z) = S[y, z] * pi[k, z] for y != z.
# Reversibility: pi[k, y] Q^k(y, z) = pi[k, z] Q^k(z, y).
# ---------------------------------------------------------------------------

def gtr_eigendecomp(pi_k: np.ndarray, S: np.ndarray,
                     ) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Spectral decomposition of the GTR rate matrix Q^k with stationary
    pi_k and symmetric exchangeabilities S.

    Returns (xi, U, D_half) where:
      xi        : (A,) real eigenvalues (xi[0] = 0 stationary).
      U         : (A, A) orthonormal eigenvectors of the symmetrised
                  matrix Q_sym = D^{1/2} Q^k D^{-1/2}.
      D_half    : (A,) sqrt(pi_k).

    Then Q^k = D_half^{-1} U diag(xi) U^T D_half and expm(Q^k t)[i, j] =
    D_half[j] / D_half[i] * sum_alpha U[i, alpha] U[j, alpha] exp(xi[alpha] t).
    """
    A = pi_k.shape[0]
    Q = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            if i != j:
                Q[i, j] = S[i, j] * pi_k[j]
        Q[i, i] = -Q[i].sum()
    D_half = np.sqrt(np.maximum(pi_k, 1e-300))
    Q_sym = (D_half[:, None] * Q) / D_half[None, :]
    # Symmetrise for numerical safety.
    Q_sym = 0.5 * (Q_sym + Q_sym.T)
    xi, U = np.linalg.eigh(Q_sym)
    return xi, U, D_half


def gtr_transition_prob(xi: np.ndarray, U: np.ndarray, D_half: np.ndarray,
                         t: float) -> np.ndarray:
    """Full transition matrix P[i, j](t) = expm(Q^k t)[i, j] from
    eigendecomp `gtr_eigendecomp`."""
    A = xi.shape[0]
    # P_sym = U diag(exp(xi t)) U^T
    P_sym = (U * np.exp(xi * t)[None, :]) @ U.T
    # De-symmetrise: P[i, j] = D_half[j] / D_half[i] * P_sym[i, j].
    return (P_sym * D_half[None, :]) / D_half[:, None]


def gtr_bridge_hr(pi_k: np.ndarray, S: np.ndarray, X: int, Y: int,
                   t: float,
                   eig: 'tuple[np.ndarray, np.ndarray, np.ndarray] | None' = None,
                   ) -> 'tuple[float, np.ndarray, np.ndarray]':
    """Standard GTR bridged HR: expected dwell in state i and expected
    transitions i -> j on a branch of length t with observed endpoints
    (X, Y).

    Returns (P_XY, W_i, U_ij) where:
      P_XY  : P(Y | X, t) = expm(Q^k t)[X, Y]
      W_i   : (A,) E[dwell in i | X, Y, t]
      U_ij  : (A, A) E[# transitions i -> j | X, Y, t]

    Uses the standard J^{alpha, beta} spectral formula
    (Holmes & Rubin 2002); for F81 it reduces to Bruno 1996.
    """
    if eig is None:
        eig = gtr_eigendecomp(pi_k, S)
    xi, U, D_half = eig
    A = pi_k.shape[0]

    # P(Y | X, t)
    P_all = gtr_transition_prob(xi, U, D_half, t)
    P_XY = float(P_all[X, Y])

    # J^{alpha, beta}(t) = int_0^t exp(xi_alpha s + xi_beta (t-s)) ds
    #                    = (exp(xi_alpha t) - exp(xi_beta t)) / (xi_alpha - xi_beta)   [alpha != beta]
    #                    = t * exp(xi_alpha t)                                          [alpha == beta]
    exp_xi_t = np.exp(xi * t)
    J = np.zeros((A, A), dtype=np.float64)
    for a in range(A):
        for b in range(A):
            d = xi[a] - xi[b]
            if abs(d) < 1e-12:
                J[a, b] = t * exp_xi_t[a]
            else:
                J[a, b] = (exp_xi_t[a] - exp_xi_t[b]) / d

    # E[dwell in state i | X, Y, t] * P_XY
    #   = sqrt(pi[Y] / pi[X]) * sum_{alpha, beta} U[X, alpha] U[i, alpha]
    #                                             U[i, beta]  U[Y, beta] J[alpha, beta]
    # W_i = W_i_num / P_XY
    W_i = np.zeros(A, dtype=np.float64)
    factor_Y_over_X = D_half[Y] / D_half[X]
    for i in range(A):
        s = 0.0
        for a in range(A):
            for b in range(A):
                s += U[X, a] * U[i, a] * U[i, b] * U[Y, b] * J[a, b]
        W_i[i] = factor_Y_over_X * s / max(P_XY, 1e-300)

    # E[# transitions i -> j | X, Y, t] * P_XY = Q[i, j] * integral of
    # P(X -> i; s) * P(j -> Y; t - s) ds.  Under the symmetric-eigenvector
    # form,
    #   P(X -> i; s) * P(j -> Y; t - s)
    #     = (D_half[i] / D_half[X]) * (D_half[Y] / D_half[j])
    #       * sum U[X,a] U[i,a] U[j,b] U[Y,b] exp(xi_a s + xi_b (t-s)),
    # and Q[i, j] = S[i, j] * pi[j] = S[i, j] * D_half[j]^2. The
    # D_half[j] factors combine as D_half[i] * D_half[j], not
    # D_half[j]^2 as a naive Q[i, j] * factor_Y_over_X would give.
    U_ij = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            if i == j:
                continue
            s = 0.0
            for a in range(A):
                for b in range(A):
                    s += U[X, a] * U[i, a] * U[j, b] * U[Y, b] * J[a, b]
            U_ij[i, j] = (S[i, j] * D_half[i] * D_half[j]
                            * factor_Y_over_X * s
                            / max(P_XY, 1e-300))

    return P_XY, W_i, U_ij


def gtr_free_end_hr(pi_k: np.ndarray, S: np.ndarray, X: int, tau: float,
                     eig: 'tuple[np.ndarray, np.ndarray, np.ndarray] | None' = None,
                     ) -> 'tuple[np.ndarray, np.ndarray]':
    """GTR one-endpoint-observed HR: given start X and duration tau,
    with no end-state conditioning, expected dwell and transitions.

    Returns (W_i, U_ij) where:
      W_i[i]    = E[dwell in state i on [0, tau] | X_0 = X]
      U_ij[i,j] = E[# i -> j transitions on [0, tau] | X_0 = X]

    Formulas:
      W_i(X; tau)      = integrate expm(Q^k s)[X, i] over s in [0, tau]
      U_ij(X; tau)     = Q[i, j] * W_i(X; tau)   (transitions occur at
                        rate Q[i, j] while in state i)

    Spectral form:
      W_i(X; tau) = sqrt(pi[i] / pi[X]) * sum_alpha U[X, alpha] U[i, alpha]
                    * h(xi[alpha], tau)
      where h(0, tau) = tau, h(xi, tau) = (exp(xi tau) - 1) / xi.
    """
    if eig is None:
        eig = gtr_eigendecomp(pi_k, S)
    xi, U, D_half = eig
    A = pi_k.shape[0]

    # h(xi, tau)
    h = np.where(np.abs(xi) < 1e-12, tau, (np.exp(xi * tau) - 1.0) / np.where(np.abs(xi) < 1e-12, 1.0, xi))
    factor = 1.0 / D_half[X]                       # sqrt(1 / pi[X])
    W_i = np.zeros(A, dtype=np.float64)
    for i in range(A):
        s = 0.0
        for a in range(A):
            s += U[X, a] * U[i, a] * h[a]
        W_i[i] = factor * D_half[i] * s

    Q_off = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            if i != j:
                Q_off[i, j] = S[i, j] * pi_k[j]
    U_ij = Q_off * W_i[:, None]
    return W_i, U_ij


def gtr_free_end_hr_avg_case1(pi_k: np.ndarray, S: np.ndarray, X: int,
                                 t: float, delta: float,
                                 eig: 'tuple[np.ndarray, np.ndarray, np.ndarray] | None' = None,
                                 ) -> 'tuple[np.ndarray, np.ndarray]':
    """Free-end HR averaged over jump-time tau_1 under the case-1
    truncated-exponential density on [0, t] with rate `delta`.

    p(tau_1 | case (1), theta_X, theta_1) propto exp(delta * tau_1),
    normalised over [0, t]. The average is
      E[W_i(X; tau_1)]
        = int_0^t p(tau_1) * W_i(X; tau_1) dtau_1.

    Under the spectral form of W_i(X; tau) = C_{X, i} + sum_alpha
    c_{X, alpha, i} h(xi_alpha, tau), the averaging integrates
    h(xi_alpha, tau_1) against the truncated exponential in tau_1 to
    give a closed-form linear combination.
    """
    if eig is None:
        eig = gtr_eigendecomp(pi_k, S)
    xi, U, D_half = eig
    A = pi_k.shape[0]

    # Normalisation of the truncated exponential.
    if abs(delta) < 1e-12:
        Z = t  # uniform limit
    else:
        Z = (np.exp(delta * t) - 1.0) / delta

    # E[h(xi_alpha, tau_1)] under p(tau_1) = exp(delta * tau_1) / Z
    # For xi_alpha != 0:
    #   E[h(xi_a, tau_1)] = (1 / (Z * xi_a))
    #     * [ (exp((xi_a + delta) * t) - 1) / (xi_a + delta)   [if xi_a + delta != 0]
    #         - Z ]
    # For xi_a == 0: E[h(0, tau_1)] = E[tau_1] under the truncated exp.
    E_h = np.zeros(A, dtype=np.float64)
    for a in range(A):
        xa = float(xi[a])
        if abs(xa) < 1e-12:
            # E[tau_1] under truncated exponential rate delta on [0, t].
            if abs(delta) < 1e-12:
                E_h[a] = t / 2.0
            else:
                E_h[a] = (t * np.exp(delta * t) / (np.exp(delta * t) - 1.0)
                            - 1.0 / delta)
        else:
            xa_plus_d = xa + delta
            if abs(xa_plus_d) < 1e-12:
                # L'Hopital limit: int_0^t (e^{xa s} - 1) * e^{-xa s} / xa / Z ds
                # e^{delta s} = e^{-xa s}, and (e^{xa s} - 1) e^{-xa s} = 1 - e^{-xa s}
                # int_0^t (1 - e^{-xa s}) ds = t + (e^{-xa t} - 1)/xa
                E_h[a] = (t + (np.exp(-xa * t) - 1.0) / xa) / (Z * xa)
            else:
                num_bracket = (np.exp(xa_plus_d * t) - 1.0) / xa_plus_d - Z
                E_h[a] = num_bracket / (Z * xa)

    factor = 1.0 / D_half[X]
    W_i = np.zeros(A, dtype=np.float64)
    for i in range(A):
        s = 0.0
        for a in range(A):
            s += U[X, a] * U[i, a] * E_h[a]
        W_i[i] = factor * D_half[i] * s

    Q_off = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            if i != j:
                Q_off[i, j] = S[i, j] * pi_k[j]
    U_ij = Q_off * W_i[:, None]
    return W_i, U_ij


def gtr_transition_avg_case1(pi_k: np.ndarray, S: np.ndarray, X: int,
                                t: float, delta: float,
                                eig: 'tuple[np.ndarray, np.ndarray, np.ndarray] | None' = None,
                                ) -> np.ndarray:
    """Target-state distribution E[P^k(X -> a; tau)] averaged over tau under
    the case-1 truncated-exponential density p(tau) ~ exp(delta * tau) on
    [0, t], normalised.

    Returns P_avg (A,) with sum_a P_avg[a] = 1.

    Purpose: under the archetype dynfield case-M=1, the second segment
    [tau_1, t] has resampled start z at tau_1 and observed end Y_n at t.
    By reversibility of GTR, the residue-conditioned distribution of z at
    tau_1^+ equals P^{k_1}(Y_n -> z; t - tau_1). Averaging this over the
    case-1 tau_1 posterior gives the V-attribution weight for the second
    segment's start. See appendix sec:archetype-suppl (case M=1, second
    segment "symmetric under time reversal").

    Setup: after substituting tau' = t - tau_1, the density on tau' is
        p(tau') propto exp(-delta * tau') on [0, t]
    (still normalised). Under the spectral form
        P^k(X -> a; tau') = (D_half[a] / D_half[X])
                            * sum_alpha U[X, alpha] U[a, alpha] * exp(xi_alpha * tau')
    the average factorises into per-alpha exponential integrals.
    """
    if eig is None:
        eig = gtr_eigendecomp(pi_k, S)
    xi, U, D_half = eig
    A = pi_k.shape[0]

    # p(tau_1) = exp(delta tau_1) / Z_1 on [0, t] with Z_1 = (exp(delta t) - 1)/delta.
    # Under tau' = t - tau_1: p(tau') = exp(delta t) * exp(-delta tau') / Z_1.
    # E[exp(xi_alpha tau')] under p(tau'):
    #   = (exp(delta t) / Z_1) * int_0^t exp((xi_alpha - delta) tau') dtau'
    #   = (exp(delta t) / Z_1) * (exp((xi_alpha - delta) t) - 1) / (xi_alpha - delta)
    #     [for xi_alpha - delta != 0; use t*exp(delta t)/Z_1 in the L'Hopital limit]
    if abs(delta) < 1e-12:
        Z1 = t
    else:
        Z1 = (np.exp(delta * t) - 1.0) / delta
    exp_delta_t_over_Z1 = np.exp(delta * t) / max(Z1, 1e-300)

    E_exp = np.zeros(A, dtype=np.float64)
    for alpha in range(A):
        d = float(xi[alpha]) - delta
        if abs(d) < 1e-12:
            E_exp[alpha] = exp_delta_t_over_Z1 * t
        else:
            E_exp[alpha] = exp_delta_t_over_Z1 * (np.exp(d * t) - 1.0) / d

    P_avg = np.zeros(A, dtype=np.float64)
    for a in range(A):
        s = 0.0
        for alpha in range(A):
            s += U[X, alpha] * U[a, alpha] * E_exp[alpha]
        P_avg[a] = (D_half[a] / D_half[X]) * s
    return P_avg


# ---------------------------------------------------------------------------
# M-steps that consume HR sufficient statistics.
# ---------------------------------------------------------------------------


def update_pi_archetype_gtr(V: np.ndarray, U: np.ndarray, W: np.ndarray,
                              S: np.ndarray, *,
                              alpha_prior: float = 0.5,
                              n_newton: int = 32,
                              tol: float = 1e-10,
                              pi_clamp: float = 1e-30,
                              ) -> np.ndarray:
    """GTR M-step for pi_archetype from HR sufficient statistics.

    Given V[k, a] (segment starts), U[k, i, j] (i→j transitions), W[k, i]
    (dwell time) per archetype k, and fixed exchangeabilities S[i, j],
    the complete-data log-likelihood in pi[k, :] is
        ell_k(pi) = sum_a V'[k, a] log pi[a]  -  sum_i W[k, i]  r_k[a] pi[a]
    with
        V'[k, a] = V[k, a] + sum_i U[k, i, a]    (state-entry count),
        r_k[a]   = sum_{b != a} S[a, b] W[k, b]  (dwell-weighted opportunity).
    Under a Dirichlet(alpha_prior) prior on pi[k, :] and Lagrange multiplier
    lambda for sum_a pi[k, a] = 1, the stationarity condition is
        (V'[k, a] + alpha_prior - 1) / pi[k, a] = r_k[a] + lambda,
    i.e.  pi[k, a] = (V'[k, a] + alpha_prior - 1) / (r_k[a] + lambda),
    with lambda the unique root of  g(lambda) := sum_a c[a] / (r[a] + lambda)
    - 1 = 0 on lambda > -min_a r_k[a]. g is strictly decreasing on this
    interval, so Newton on log(lambda + min_r + eps) or on lambda directly
    converges quadratically.

    Sub-1 Dirichlet prior (default alpha_prior=0.5) is a sparse
    simplex-corner prior. The MAP numerator c = V' + (alpha - 1) can
    be negative for entries with V' <= 0.5; we clamp to `pi_clamp`
    (default 1e-30) so the Newton solver operates on positive c, and
    clamp the output pi elementwise to `pi_clamp` before renormalising.
    This preserves sparsity (unobserved entries land at ~pi_clamp not
    ~1/A) while avoiding subnormal float64 in downstream D = sqrt(pi):
    sqrt(1e-30) = 1e-15, comfortably above the 1e-308 subnormal
    threshold. Set alpha_prior >= 1 for Laplace-style smoothing (the
    old default); alpha_prior=0 recovers the pure MLE.

    Returns pi_archetype_new (K_a, A).
    """
    V = np.asarray(V, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    K_a, A = V.shape

    Vprime = V + U.sum(axis=1)                      # (K_a, A)
    # Clamp c > 0 so Newton is well-posed even under sparse (alpha < 1)
    # priors. Sparsity is preserved: entries with V' below the pseudocount
    # threshold land at c=pi_clamp, driving pi[a] to ~pi_clamp before
    # renormalisation (not exactly 0, but negligible).
    c = np.maximum(Vprime + (alpha_prior - 1.0), pi_clamp)

    pi_new = np.zeros((K_a, A), dtype=np.float64)
    for k in range(K_a):
        Wk = W[k]                                     # (A,)
        # r_k[a] = sum_{b != a} S[a, b] * W_k[b]
        r = S @ Wk                                    # symmetric S, diag zero
        c_k = c[k]                                    # (A,)
        # Fallback: if all c_k <= 0 (unused arch), use uniform.
        if float(c_k.sum()) <= 0.0:
            pi_new[k] = 1.0 / A
            continue

        # We want lambda > -min r (all denominators positive).
        # g(lambda) = sum c[a] / (r[a] + lambda) - 1
        # dg/dlambda = -sum c[a] / (r[a] + lambda)^2  (strictly negative
        # when any c[a] > 0).
        # g is decreasing AND convex on (−r_min, ∞) (g'' = 2 sum c/(r+λ)^3
        # > 0), so Newton starting *above* the root λ* is monotonically
        # decreasing without overshoot -- pick a safe upper bracket:
        # λ_hi such that sum c/(r+λ_hi) ≤ 1. Taking λ_hi = Σc − r_min:
        # then denom[a] = r[a] + Σc − r_min ≥ Σc for a = argmin r, so
        # sum c/(r+λ_hi) ≤ Σc/Σc = 1, i.e.\ g(λ_hi) ≤ 0 as required.
        # (Old default lam = max(0, -r_min + 1e-8) starts *below* the root
        # when Σc ≫ 1, giving g(λ_init) ≈ Σc/1e-8 which overflows in gp
        # when Σc gets even mildly peaked -- observed at rho_chain ≈ 0.08
        # with heavily concentrated pi_arch in mid-training.)
        r_min = float(r.min())
        total_c = float(c_k.sum())
        lam = max(total_c - r_min + 1e-8, 1e-8)
        lam_floor = -r_min + 1e-30
        denom_floor = 1e-30
        # Newton on g(λ) = Σ c/(r+λ) − 1. Compute gp = Σ c/(r+λ)^2 as
        # Σ (c/denom) * (1/denom) rather than Σ c/(denom*denom): the
        # latter overflows when denom < ~1e-154 (denom*denom > 1e308)
        # even when the correct term is well-scaled. Reformulating as
        # (c/denom)/denom keeps intermediates in-range whenever
        # c/denom itself is (which it must be for g to be finite).
        # Suppress the residual overflow warnings from adversarial
        # inputs — the finite-check below catches them defensively.
        with np.errstate(over='ignore', invalid='ignore'):
            for _ in range(int(n_newton)):
                denom = np.maximum(r + lam, denom_floor)
                c_over_denom = c_k / denom
                g = float(c_over_denom.sum()) - 1.0
                if abs(g) < tol:
                    break
                gp = -float((c_over_denom / denom).sum())
                if (not np.isfinite(g) or not np.isfinite(gp)
                        or gp == 0.0):
                    break
                step = g / gp
                new_lam = lam - step
                if new_lam <= lam_floor:
                    new_lam = 0.5 * (lam + lam_floor)
                if abs(new_lam - lam) < 1e-15:
                    break
                lam = new_lam
        denom = np.maximum(r + lam, denom_floor)
        pi_k = c_k / denom
        if not np.all(np.isfinite(pi_k)) or float(pi_k.sum()) <= 0.0:
            # Fallback: Dirichlet posterior mean (ignores dwell/transition
            # sufficient statistics, but numerically safe).
            pi_k = c_k / max(total_c, 1e-30)
        pi_k = np.clip(pi_k, pi_clamp, None)
        pi_k = pi_k / pi_k.sum()
        pi_new[k] = pi_k
    return pi_new


def update_rho_chain_gamma(N_theta_sum: float, T_sum: float, *,
                             prior_a: float = 1.5,
                             prior_b: float = 5.0,
                             mode: str = "map",
                             ) -> float:
    """Closed-form M-step for rho_chain from bridge-conditional HR
    sufficient statistics (Sum_q E[N_theta_q | obs], Sum_q T_q).

    Under Gamma(prior_a, prior_b) prior on rho_chain and the field-chain
    likelihood
        p(field trajectories | rho_chain) prop_to
            rho_chain^{Sum N_theta} exp(-rho_chain * Sum T),
    the posterior is Gamma(prior_a + Sum N_theta, prior_b + Sum T).

    mode='map' returns (prior_a - 1 + Sum N_theta) / (prior_b + Sum T)
    (posterior mode).  mode='mean' returns (prior_a + Sum N_theta) /
    (prior_b + Sum T) (posterior mean).  Default is 'map' to match the
    appendix eq:arch-Mstep-rho.
    """
    N_theta_sum = float(N_theta_sum)
    T_sum = float(T_sum)
    prior_a = float(prior_a); prior_b = float(prior_b)
    if mode == "map":
        num = prior_a - 1.0 + N_theta_sum
    elif mode == "mean":
        num = prior_a + N_theta_sum
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    den = prior_b + T_sum
    return max(1e-12, num / max(den, 1e-300))


# ---------------------------------------------------------------------------
# Cluster-level accumulator. Composes the primitives to reproduce (V, U, W,
# N_theta) for a (cluster, cherry) pair. Verified against
# compound_ctmc_bridge_hr in tests/dynfield/test_hr_stats_closed_form.py.
# ---------------------------------------------------------------------------


def hr_cluster_stats(rho: np.ndarray, rho_chain: float,
                       pi_archetype: np.ndarray,
                       arch_assignment: np.ndarray,
                       classes: np.ndarray,
                       X_obs: np.ndarray, Y_obs: np.ndarray,
                       t: float, S: np.ndarray,
                       n_tau1_grid: int = 32,
                       ) -> 'tuple[float, np.ndarray, np.ndarray, np.ndarray, float]':
    """Cluster-level Holmes-Rubin sufficient statistics for the archetype
    dynfield model, composing the case-decomposition primitives.

    The residue at each field jump is resampled from pi_arch of the new
    archetype. Marginalising unobserved intermediate residue values
    collapses (by stationarity):
      Σ_{x'} pi_arch[k, x'] · P^k(x' → Y_n; τ) = pi_arch[k, Y_n]
    so the compound residue-endpoint likelihood factorises as

      P(X_obs at 0, Y_obs at t | θ_X, θ_Y, t) =
          δ_{θ_X, θ_Y} · β_t(θ_X) · prod_n P^{k_X_n}(X_n → Y_n; t)
          + (P_θ(θ_X → θ_Y; t) - δ_{θ_X, θ_Y} · β_t(θ_X)) · prod_n pi_arch[k_Y_n, Y_n],

    the first term being the M=0 (no field-jump) contribution and the second
    the M ≥ 1 mass at endpoint pair (θ_X, θ_Y).

    HR decomposition:
      - M=0 contribution: standard GTR bridge HR per site.
      - M ≥ 1 contribution: field-side bridge HR from field_expected_dwell_jumps,
        residues stationary in whatever arch the field is in (first-segment
        [0,τ_1] free-end contribution from X_n and last-segment [τ_M, t]
        time-reversed free-end contribution from Y_n are added as corrections
        via the primitive gtr_free_end_hr_avg_case1 for M=1; higher M
        contributions collapse to stationary middle-segment HR).

    Returns
    -------
    P_obs   : marginal joint P(X_obs, Y_obs; t) marginalizing theta_X ~ rho.
    V       : (K_a, A) expected # segment starts, by (arch, residue).
    U       : (K_a, A, A) expected # residue substitutions, by (arch, i, j).
    W       : (K_a, A) expected dwell time, by (arch, residue).
    N_theta : expected total # field jumps.
    """
    rho = np.asarray(rho, dtype=np.float64)
    pi_archetype = np.asarray(pi_archetype, dtype=np.float64)
    arch_assignment = np.asarray(arch_assignment, dtype=np.int64)
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)
    X_obs = np.asarray(X_obs, dtype=np.int64).reshape(-1)
    Y_obs = np.asarray(Y_obs, dtype=np.int64).reshape(-1)
    S = np.asarray(S, dtype=np.float64)

    L = int(rho.shape[0])
    K_a, A = int(pi_archetype.shape[0]), int(pi_archetype.shape[1])
    N = int(classes.shape[0])

    eigs = [gtr_eigendecomp(pi_archetype[k], S) for k in range(K_a)]
    beta_t = np.exp(-rho_chain * (1.0 - rho) * t)
    g_t = np.exp(-rho_chain * t)

    V = np.zeros((K_a, A), dtype=np.float64)
    W = np.zeros((K_a, A), dtype=np.float64)
    U_stats = np.zeros((K_a, A, A), dtype=np.float64)
    N_theta = 0.0
    P_obs = 0.0

    # Field transition prob P_θ(θ_X → θ_Y; t) from the rank-1 spectral form:
    #   P_θ(i → j; t) = g_t · δ_{ij} + (1 - g_t) · ρ[j]
    P_field = np.full((L, L), (1.0 - g_t) * 0.0, dtype=np.float64)
    for i in range(L):
        for j in range(L):
            P_field[i, j] = g_t * (1.0 if i == j else 0.0) + (1.0 - g_t) * float(rho[j])

    # =========================================================================
    # Sweep (theta_X, theta_Y) pairs. For each:
    #   M=0 contribution:  θ_X = θ_Y only, β_t(θ_X) * prod_n P^k(X_n → Y_n; t)
    #   M ≥ 1 contribution: (P_θ(θ_X → θ_Y; t) - δ_{θ_X, θ_Y} β_t(θ_X))
    #                       * prod_n pi_arch[k_Y_n, Y_n]
    # (see docstring; unobserved intermediate residues marginalise via
    # stationarity of the resample distribution.)
    # =========================================================================

    # Field-side dwell HR bridge: E[T_i | θ_X, θ_Y, t] and E[N_θ | θ_X, θ_Y, t].
    #   field_expected_dwell_jumps returns (P_XY, E_T[i], E_N_theta).

    for theta_X in range(L):
        for theta_Y in range(L):
            P_XY = float(P_field[theta_X, theta_Y])
            P_no_jump = float(beta_t[theta_X]) if theta_X == theta_Y else 0.0
            P_jump = max(0.0, P_XY - P_no_jump)

            # ------- M=0 contribution (θ_X = θ_Y only) -------
            if theta_X == theta_Y and P_no_jump > 0.0:
                L_res_0 = 1.0
                per_site_hr = []
                for n in range(N):
                    k_n = int(arch_assignment[int(classes[n]), theta_X])
                    P_res, W_i, U_ij = gtr_bridge_hr(
                        pi_archetype[k_n], S, int(X_obs[n]),
                        int(Y_obs[n]), t, eig=eigs[k_n])
                    L_res_0 *= max(P_res, 1e-300)
                    per_site_hr.append((k_n, W_i, U_ij))
                weight_0 = float(rho[theta_X]) * P_no_jump * L_res_0
                P_obs += weight_0
                for n, (k_n, W_i, U_ij) in enumerate(per_site_hr):
                    V[k_n, int(X_obs[n])] += weight_0
                    W[k_n] += weight_0 * W_i
                    U_stats[k_n] += weight_0 * U_ij

            # ------- M ≥ 1 contribution -------
            if P_jump <= 0.0:
                continue

            L_res_j = 1.0
            for n in range(N):
                k_Y_n = int(arch_assignment[int(classes[n]), theta_Y])
                L_res_j *= float(pi_archetype[k_Y_n, int(Y_obs[n])])
            weight_j = float(rho[theta_X]) * P_jump * L_res_j
            P_obs += weight_j

            # Field HR bridge: E[T_i | θ_X, θ_Y, t] (over all M inclusive).
            # Restrict to M ≥ 1 by subtracting M=0 contribution.
            _, E_T_all, E_N_all = field_expected_dwell_jumps(
                rho, rho_chain, theta_X, theta_Y, t)
            # Bridge-conditional per-θ_int arrival counts (for V attribution).
            E_arrivals_all = field_expected_arrivals(
                rho, rho_chain, theta_X, theta_Y, t)
            # For M ≥ 1 conditional dwell: E[T_i | θ_X, θ_Y, t] * P_XY
            #   - δ_{θ_X, θ_Y} * β_t(θ_X) * (t if i == θ_X else 0)
            # gives E[T_i · I(M≥1) · P_XY].  Divide by P_jump for conditional.
            T_all_prod = E_T_all * P_XY   # E[T_i | θ_X, θ_Y, t] · P_XY
            if theta_X == theta_Y:
                T_all_prod = T_all_prod.copy()
                T_all_prod[theta_X] -= P_no_jump * t
            E_T_jump = T_all_prod / max(P_jump, 1e-300)
            # E[N_θ | M ≥ 1] = E[N_θ] / P(M ≥ 1) since M=0 contributes 0.
            E_N_jump = float(E_N_all * P_XY) / max(P_jump, 1e-300)
            # Per-θ arrival counts under M ≥ 1: same subtraction (no arrivals
            # under M=0).  E_arrivals_all is bridge-conditional (all M).
            E_arrivals_jump = E_arrivals_all * P_XY / max(P_jump, 1e-300)

            # N_θ accumulates
            N_theta += weight_j * E_N_jump

            # Residue-side HR under M ≥ 1: first / middle / last decomposition.
            # First segment [0, τ_1] in arch k_X_n, from X_n (unobserved end).
            # Last  segment [τ_M, t] in arch k_Y_n, to Y_n (time-reversed:
            #                                    from Y_n, unobserved end).
            # Middle segments (only under M ≥ 2): both endpoints resampled ~
            #                                    pi_arch, residues at stationarity.
            #
            # Approximation: use the case-M=1 truncated-exp density for both
            # τ_1 and τ_M. Exact when M=1 dominates (short branches); adds an
            # O((ρ_chain · t)^2) error at higher M. Middle-segment time then
            # residualises from E_T_jump minus first / last times.
            delta_first = (rho_chain * (1.0 - float(rho[theta_X]))
                             - rho_chain * (1.0 - float(rho[theta_Y])))
            # delta_first: τ_1 rate (r_1 - r_X) with θ_1 = θ_Y under M=1.
            # Actually delta_first = r_Y - r_X (jump-time density in τ_1).
            delta_first = -delta_first
            delta_last = -delta_first
            # E[τ_1 | case-1] via truncated-exp mean.
            if abs(delta_first) < 1e-12:
                E_tau1 = t / 2.0
            else:
                exp_dt = float(np.exp(delta_first * t))
                E_tau1 = t * exp_dt / (exp_dt - 1.0) - 1.0 / delta_first
            E_tauM_to_t = t - E_tau1   # E[t - τ_M | case-1]

            for n in range(N):
                # V at t=0 boundary (X_n at θ_X):
                k_X_n = int(arch_assignment[int(classes[n]), theta_X])
                V[k_X_n, int(X_obs[n])] += weight_j

                pi_kx = pi_archetype[k_X_n]
                k_Y_n = int(arch_assignment[int(classes[n]), theta_Y])
                pi_kY = pi_archetype[k_Y_n]

                # First segment W, U: free-end HR from X_n, τ_1 avg.
                W_first, U_first = gtr_free_end_hr_avg_case1(
                    pi_kx, S, int(X_obs[n]), t, delta_first,
                    eig=eigs[k_X_n])
                W[k_X_n] += weight_j * W_first
                U_stats[k_X_n] += weight_j * U_first

                # Last segment W, U (time-reversed free-end from Y_n, over
                # duration t - τ_M):
                #   E[dwell in i]  = W^{free-end}(Y_n; τ'; k_Y)
                #   E[# i→j jumps] = pi_arch[k_Y, i] · S[i, j] · W^{free-end}_j(Y_n; τ')
                # where τ' = t - τ_M ~ p(τ') on [0, t] with rate delta_last.
                W_last, _ = gtr_free_end_hr_avg_case1(
                    pi_kY, S, int(Y_obs[n]), t, delta_last,
                    eig=eigs[k_Y_n])
                W[k_Y_n] += weight_j * W_last
                U_last = pi_kY[:, None] * S * W_last[None, :]
                np.fill_diagonal(U_last, 0.0)
                U_stats[k_Y_n] += weight_j * U_last

                # Middle-segment stationary contribution: middle time in θ_int
                # = E_T_jump[θ_int] - (first-seg time in θ_int, if θ_X = θ_int)
                #                   - (last-seg time in θ_int, if θ_Y = θ_int).
                for theta_int in range(L):
                    k = int(arch_assignment[int(classes[n]), theta_int])
                    pi_k = pi_archetype[k]
                    first_time = E_tau1 if theta_int == theta_X else 0.0
                    last_time = E_tauM_to_t if theta_int == theta_Y else 0.0
                    middle_time = float(E_T_jump[theta_int]) - first_time - last_time
                    if middle_time <= 0:
                        continue
                    W[k] += weight_j * middle_time * pi_k
                    outer = pi_k[:, None] * pi_k[None, :]
                    contrib_U = weight_j * middle_time * S * outer
                    np.fill_diagonal(contrib_U, 0.0)
                    U_stats[k] += contrib_U
                # V for post-jump segment starts. Decomposition:
                # (a) Last jump (destination = θ_Y): 1 arrival per trajectory.
                #     By reversibility (appendix eq:arch-Vsecond-closed) the
                #     residue post-resample is distributed as
                #     P^{k_Y}(Y_n -> a; t - τ_M) averaged over τ_M's
                #     bridge-conditional density. Approximate τ_M's density
                #     by the case-M=1 truncated-exp form -- exact for M=1
                #     (which dominates at short branches) and approximate for
                #     M >= 2 where τ_M has a different exponential-like form.
                # (b) Non-last jumps (destinations θ_int, including
                #     re-entries into θ_Y under M >= 2): drawn i.i.d. from
                #     pi_arch[k]. E[# non-last jumps to θ_int] =
                #     E_arrivals_jump[θ_int] - δ_{θ_int, θ_Y}.
                k_Y_n = int(arch_assignment[int(classes[n]), theta_Y])
                pi_k_Y = pi_archetype[k_Y_n]
                delta_last = (rho_chain * (1.0 - float(rho[theta_Y]))
                                - rho_chain * (1.0 - float(rho[theta_X])))
                P_avg_last = gtr_transition_avg_case1(
                    pi_k_Y, S, int(Y_obs[n]), t, delta_last,
                    eig=eigs[k_Y_n])
                V[k_Y_n] += weight_j * P_avg_last

                # Non-last jumps: subtract 1 from θ_Y's arrival count.
                for theta_int in range(L):
                    k = int(arch_assignment[int(classes[n]), theta_int])
                    pi_k = pi_archetype[k]
                    E_non_last = float(E_arrivals_jump[theta_int]) - (
                        1.0 if theta_int == theta_Y else 0.0)
                    if E_non_last <= 0:
                        continue
                    V[k] += weight_j * E_non_last * pi_k

    return P_obs, V, U_stats, W, N_theta
