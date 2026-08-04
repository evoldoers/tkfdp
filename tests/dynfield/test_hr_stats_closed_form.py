"""Verify closed-form HR sufficient statistics against a direct
spectral evaluation of the full compound CTMC on state space
F * A^N, for small (F, A, N) and random parameters.

The full compound state is (theta, x_1, ..., x_N). We build the
compound rate matrix explicitly, compute expm and the HR bridge
integrals directly (via the F81-on-DP + GTR joint), then aggregate the
compound-state expectations by (archetype, residue) and (field-state)
to obtain the same (V, U, W, N_theta) that the closed-form primitives
in dynfield/hr.py produce.

The closed-form field-side helpers can be tested independently of
GTR because the F81-on-DP chain is self-contained.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.linalg import expm

from tkfdp.coupling.dynfield.hr import (
    field_case_probs, field_expected_dwell_jumps, field_expected_arrivals,
    gtr_eigendecomp, gtr_bridge_hr, gtr_free_end_hr, gtr_transition_prob,
    gtr_free_end_hr_avg_case1, gtr_transition_avg_case1,
    hr_cluster_stats,
)


# ---------------------------------------------------------------------------
# Field chain: direct spectral evaluation via matrix exponential.
# ---------------------------------------------------------------------------

def _build_Q_theta(rho: np.ndarray, rho_chain: float) -> np.ndarray:
    L = rho.shape[0]
    Q = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            if i != j:
                Q[i, j] = rho_chain * rho[j]
        Q[i, i] = -Q[i].sum()
    return Q


def _direct_field_bridge_hr(rho: np.ndarray, rho_chain: float,
                              theta_X: int, theta_Y: int,
                              t: float,
                              n_grid: int = 4001,
                              ) -> 'tuple[float, np.ndarray, float]':
    """Direct numerical evaluation of the bridge HR via trapezoidal
    integration -- independent of the closed form.

      E[T_i | bridge] * P_XY = int_0^t P(a -> i, s) * P(i -> b, t-s) ds
      E[N   | bridge] * P_XY = sum_{i != j} Q[i,j] * int P(a -> i, s)
                                            * P(j -> b, t-s) ds
    """
    Q = _build_Q_theta(rho, rho_chain)
    P_full = expm(Q * t)
    P_XY = float(P_full[theta_X, theta_Y])

    ss = np.linspace(0.0, t, n_grid)
    L = rho.shape[0]
    P_by_s = np.stack([expm(Q * s) for s in ss], axis=0)
    P_by_tms = np.stack([expm(Q * (t - s)) for s in ss], axis=0)

    E_T = np.zeros(L, dtype=np.float64)
    for i in range(L):
        integrand = P_by_s[:, theta_X, i] * P_by_tms[:, i, theta_Y]
        E_T[i] = np.trapz(integrand, ss) / max(P_XY, 1e-300)

    EN_times_P = 0.0
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            integrand = P_by_s[:, theta_X, i] * Q[i, j] * P_by_tms[:, j, theta_Y]
            EN_times_P += float(np.trapz(integrand, ss))
    E_N_theta = EN_times_P / max(P_XY, 1e-300)

    return P_XY, E_T, float(E_N_theta)


def _rand_field(rng, L: int, seed_shift: int = 0
                 ) -> 'tuple[np.ndarray, float]':
    r = rng.gamma(2.0, size=L)
    return r / r.sum(), float(rng.uniform(0.5, 2.0))


def test_field_bridge_hr_matches_direct():
    rng = np.random.default_rng(0)
    for L in (2, 3, 4):
        rho, rho_chain = _rand_field(rng, L)
        for t in (0.1, 0.5, 2.0):
            for theta_X in range(L):
                for theta_Y in range(L):
                    P_c, ET_c, EN_c = field_expected_dwell_jumps(
                        rho, rho_chain, theta_X, theta_Y, t)
                    P_d, ET_d, EN_d = _direct_field_bridge_hr(
                        rho, rho_chain, theta_X, theta_Y, t)
                    assert np.isclose(P_c, P_d, atol=1e-8), (
                        f"P mismatch L={L} t={t} X={theta_X} Y={theta_Y}: "
                        f"closed={P_c} direct={P_d}")
                    assert np.allclose(ET_c, ET_d, atol=1e-6), (
                        f"E[T] mismatch: closed={ET_c} direct={ET_d}")
                    assert np.isclose(EN_c, EN_d, atol=1e-6), (
                        f"E[N_theta] mismatch: closed={EN_c} direct={EN_d}")


def test_field_case_probs_sum_to_one():
    rng = np.random.default_rng(2)
    for L in (2, 3, 4):
        rho, rho_chain = _rand_field(rng, L)
        for t in (0.1, 1.0, 5.0):
            for theta_X in range(L):
                P0, P1, P2plus = field_case_probs(
                    rho, rho_chain, theta_X, t)
                total = P0 + float(P1.sum()) + P2plus
                assert abs(total - 1.0) < 1e-10, (
                    f"case probs sum to {total} != 1 for L={L} t={t}")


def _gillespie_field_bridge_EN(
        rho: np.ndarray, rho_chain: float, theta_X: int, theta_Y: int,
        t: float, n_samples: int = 200000, rng=None,
        ) -> 'tuple[float, float]':
    """Bridge-conditional E[N_theta] via Gillespie sampling with endpoint
    acceptance/rejection. Returns (bridge_acceptance_rate, mean N).
    Standard error of mean N over N_matches samples is sqrt(Var/N_matches).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    L = rho.shape[0]
    r_out = rho_chain * (1.0 - rho)
    N_matches = 0
    N_sum = 0
    for _ in range(n_samples):
        theta = theta_X
        elapsed = 0.0
        n = 0
        while r_out[theta] > 1e-15:
            dt = rng.exponential(1.0 / r_out[theta])
            if elapsed + dt >= t:
                break
            elapsed += dt
            n += 1
            p = rho.copy(); p[theta] = 0.0; p = p / p.sum()
            theta = int(rng.choice(L, p=p))
        if theta == theta_Y:
            N_matches += 1
            N_sum += n
    if N_matches == 0:
        return 0.0, 0.0
    return N_matches / n_samples, N_sum / N_matches


def _direct_field_case_probs_via_gillespie(
        rho: np.ndarray, rho_chain: float, theta_X: int, t: float,
        n_samples: int = 200000, rng=None):
    """Gillespie-based verification of P(M=0), P(M=1), P(M>=2)."""
    if rng is None:
        rng = np.random.default_rng(0)
    L = rho.shape[0]
    r = rho_chain * (1.0 - rho)                    # exit rates
    counts = np.zeros(3, dtype=np.int64)            # M=0, M=1, M>=2
    for _ in range(n_samples):
        theta = theta_X
        M = 0
        elapsed = 0.0
        while True:
            if r[theta] < 1e-15:
                break
            dt = rng.exponential(1.0 / r[theta])
            if elapsed + dt >= t:
                break
            elapsed += dt
            M += 1
            # Jump to new state theta' != theta with prob rho[j]/(1-rho[theta]).
            p = rho.copy()
            p[theta] = 0.0
            p = p / p.sum()
            theta = int(rng.choice(L, p=p))
            if M >= 2:
                break
        counts[min(M, 2)] += 1
    total = float(counts.sum())
    return counts[0] / total, counts[1] / total, counts[2] / total


def test_field_bridge_EN_matches_gillespie():
    """Bridge-conditional E[N_theta | (theta_X, theta_Y)] against Gillespie.
    This is the check that was missing from the initial primitives commit --
    the previous test_field_bridge_hr_matches_direct only verified the primitive
    against a helper that used the same (incorrect for the bridge) formula
    rho_chain * sum((1-rho) * E_T[bridge]), which mixes the unconditional
    exit rate with bridge-conditional dwell and does not equal E[N | bridge]
    outside the fully-stationary case.
    """
    rng = np.random.default_rng(41)
    for (L, t, tX, tY) in [
        (2, 0.15, 0, 1),   # M=1 dominates
        (2, 0.15, 0, 0),   # M=0 dominates, E[N] ~ 0.01
        (2, 0.5, 1, 0),
        (3, 0.3, 0, 2),
    ]:
        r = rng.gamma(2.0, size=L)
        rho = r / r.sum()
        rc = float(rng.uniform(0.5, 2.0))
        _, _, EN_closed = field_expected_dwell_jumps(rho, rc, tX, tY, t)
        _, EN_gillespie = _gillespie_field_bridge_EN(
            rho, rc, tX, tY, t, n_samples=100000,
            rng=np.random.default_rng(123))
        # Stderr of the mean over ~N_matches samples of a Poisson-like variate
        # is roughly sqrt(EN_gillespie / N_matches). Use loose tolerance.
        assert abs(EN_closed - EN_gillespie) < 0.03, (
            f"E[N | bridge] mismatch L={L} t={t} ({tX}, {tY}): "
            f"closed={EN_closed:.4f} gillespie={EN_gillespie:.4f}")


def test_field_case_probs_matches_gillespie():
    rng = np.random.default_rng(11)
    L = 3
    rho, rho_chain = _rand_field(rng, L)
    t = 0.7
    theta_X = 0
    P0_c, P1_c, P2plus_c = field_case_probs(rho, rho_chain, theta_X, t)
    P0_g, P1_g, P2plus_g = _direct_field_case_probs_via_gillespie(
        rho, rho_chain, theta_X, t, n_samples=100000,
        rng=np.random.default_rng(42))
    # Gillespie std ~ sqrt(p (1-p) / N) ~ 0.002 with N=1e5.
    assert abs(P0_c - P0_g) < 0.01, f"P0 closed={P0_c}, gillespie={P0_g}"
    assert abs(P1_c.sum() - P1_g) < 0.01, (
        f"P1 sum closed={P1_c.sum()}, gillespie={P1_g}")
    assert abs(P2plus_c - P2plus_g) < 0.01, (
        f"P2+ closed={P2plus_c}, gillespie={P2plus_g}")


# ---------------------------------------------------------------------------
# GTR bridge HR: check against direct spectral evaluation and Bruno 1996
# F81 special case.
# ---------------------------------------------------------------------------

def _rand_pi_S(rng, A: int) -> 'tuple[np.ndarray, np.ndarray]':
    raw = rng.gamma(2.0, size=A)
    pi = raw / raw.sum()
    S_raw = rng.gamma(2.0, size=(A, A))
    S = 0.5 * (S_raw + S_raw.T)
    np.fill_diagonal(S, 0.0)
    return pi, S


def _direct_gtr_bridge_hr(pi_k, S, X, Y, t, n_grid=2001):
    """Direct numerical evaluation of the bridge HR via matrix
    exponentiation and trapezoidal integration."""
    A = pi_k.shape[0]
    Q = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            if i != j:
                Q[i, j] = S[i, j] * pi_k[j]
        Q[i, i] = -Q[i].sum()
    P_all = expm(Q * t)
    P_XY = float(P_all[X, Y])
    ss = np.linspace(0.0, t, n_grid)
    P_s = np.stack([expm(Q * s) for s in ss], axis=0)
    P_tms = np.stack([expm(Q * (t - s)) for s in ss], axis=0)
    W_i = np.zeros(A, dtype=np.float64)
    for i in range(A):
        W_i[i] = np.trapz(P_s[:, X, i] * P_tms[:, i, Y], ss)
    W_i /= max(P_XY, 1e-300)

    U_ij = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            if i == j:
                continue
            U_ij[i, j] = np.trapz(
                P_s[:, X, i] * Q[i, j] * P_tms[:, j, Y], ss)
    U_ij /= max(P_XY, 1e-300)
    return P_XY, W_i, U_ij


def test_gtr_bridge_hr_matches_direct():
    rng = np.random.default_rng(13)
    for A in (3, 4):
        pi, S = _rand_pi_S(rng, A)
        for t in (0.1, 1.0):
            for X in (0, A - 1):
                for Y in (0, A - 1):
                    P_c, W_c, U_c = gtr_bridge_hr(pi, S, X, Y, t)
                    P_d, W_d, U_d = _direct_gtr_bridge_hr(pi, S, X, Y, t)
                    assert np.isclose(P_c, P_d, atol=1e-8), (
                        f"P_XY mismatch")
                    assert np.allclose(W_c, W_d, atol=1e-5), (
                        f"W_i mismatch A={A} t={t}: max diff "
                        f"{np.max(np.abs(W_c - W_d))}")
                    assert np.allclose(U_c, U_d, atol=1e-5), (
                        f"U_ij mismatch A={A} t={t}: max diff "
                        f"{np.max(np.abs(U_c - U_d))}")


def test_gtr_free_end_hr_matches_direct():
    rng = np.random.default_rng(17)
    A = 3
    pi, S = _rand_pi_S(rng, A)
    tau = 0.7
    X = 1
    W_c, U_c = gtr_free_end_hr(pi, S, X, tau)
    # Direct: int_0^tau P(X -> i, s) ds
    Q = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            if i != j:
                Q[i, j] = S[i, j] * pi[j]
        Q[i, i] = -Q[i].sum()
    ss = np.linspace(0.0, tau, 2001)
    P_s = np.stack([expm(Q * s) for s in ss], axis=0)
    W_d = np.zeros(A, dtype=np.float64)
    for i in range(A):
        W_d[i] = np.trapz(P_s[:, X, i], ss)
    assert np.allclose(W_c, W_d, atol=1e-6)


def test_gtr_transition_avg_case1_matches_direct():
    """E[P^k(X -> a; tau)] under case-1 truncated exp density on [0, t]:
    closed-form spectral vs outer numerical integration over tau with
    inner spectral P^k."""
    rng = np.random.default_rng(29)
    for A in (3, 4):
        pi, S = _rand_pi_S(rng, A)
        eig = gtr_eigendecomp(pi, S)
        for delta in (-0.5, 0.0, 0.6):
            for X in (0, A - 1):
                for t in (0.2, 1.0):
                    P_c = gtr_transition_avg_case1(pi, S, X, t, delta, eig=eig)
                    # Direct: outer trapezoidal over tau in [0, t]
                    tau_grid = np.linspace(1e-8, t, 601)
                    if abs(delta) < 1e-12:
                        Z = t
                        p_tau = np.ones_like(tau_grid) / Z
                    else:
                        Z = (np.exp(delta * t) - 1.0) / delta
                        p_tau = np.exp(delta * tau_grid) / Z
                    # For each tau, compute P^k(X -> a; t - tau) since the
                    # primitive averages over tau' = t - tau (the second-
                    # segment duration).
                    P_target = np.stack(
                        [gtr_transition_prob(*eig, float(t - tau))[X]
                         for tau in tau_grid], axis=0)
                    P_d = np.trapz(P_target * p_tau[:, None], tau_grid, axis=0)
                    assert np.allclose(P_c, P_d, atol=1e-5), (
                        f"A={A} X={X} t={t} delta={delta}: "
                        f"max diff {np.max(np.abs(P_c - P_d))}")
                    # Sanity: sums to 1
                    assert abs(P_c.sum() - 1.0) < 1e-5, (
                        f"P_avg does not sum to 1: {P_c.sum()}")


def _gillespie_biased_start_second_segment(
        pi_k, S, Y, t, delta, n_samples=200000, rng=None):
    """Gillespie: sample tau_1 from case-1 density; run GTR chain backward
    from Y over duration (t - tau_1); return empirical distribution of
    the *start* state after backward run.

    By reversibility, this equals the case-1 posterior distribution of
    the resampled z at tau_1^+, given end state Y and duration t - tau_1.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    A = pi_k.shape[0]
    # For GTR chain in arch k: rates Q[i, j] = S[i, j] * pi_k[j] for i != j.
    Q_off = S * pi_k[None, :]
    np.fill_diagonal(Q_off, 0.0)
    exit_rate = Q_off.sum(axis=1)  # rate out of each state
    counts = np.zeros(A, dtype=np.int64)
    if abs(delta) < 1e-12:
        Z = t
    else:
        Z = (np.exp(delta * t) - 1.0) / delta
    for _ in range(n_samples):
        # Sample tau_1 by inverse CDF: F(tau) = (exp(delta tau) - 1)/(delta Z)
        u = rng.random()
        if abs(delta) < 1e-12:
            tau_1 = u * t
        else:
            tau_1 = np.log(1 + u * delta * Z) / delta
        # Forward GTR chain from Y for duration (t - tau_1).
        state = int(Y)
        elapsed = 0.0
        seg = t - tau_1
        while elapsed < seg and exit_rate[state] > 1e-15:
            dt = rng.exponential(1.0 / exit_rate[state])
            if elapsed + dt >= seg:
                break
            elapsed += dt
            # Jump to new state.
            p = Q_off[state].copy()
            p = p / p.sum()
            state = int(rng.choice(A, p=p))
        counts[state] += 1
    return counts / n_samples


def test_gtr_transition_avg_case1_matches_gillespie():
    """Verify the reversibility-based biased start distribution against
    Gillespie samples of a backward chain from Y over case-1 tau."""
    rng = np.random.default_rng(53)
    A = 4
    pi, S = _rand_pi_S(rng, A)
    for delta in (-0.2, 0.3):
        for Y in (0, 2):
            for t in (0.3, 1.0):
                P_c = gtr_transition_avg_case1(pi, S, Y, t, delta)
                P_g = _gillespie_biased_start_second_segment(
                    pi, S, Y, t, delta, n_samples=100000,
                    rng=np.random.default_rng(71))
                # Stderr ~ sqrt(p(1-p)/N) ~ 0.002 with N=1e5.
                assert np.max(np.abs(P_c - P_g)) < 0.01, (
                    f"delta={delta} Y={Y} t={t}: "
                    f"closed={P_c}, gillespie={P_g}")


def test_gtr_free_end_avg_case1_matches_direct():
    """Compare the case-1 tau_1 averaged W_i against direct numerical
    quadrature of the same average (trap on tau_1 outer, spectral on
    W_i(X; tau_1) inner)."""
    rng = np.random.default_rng(19)
    A = 3
    pi, S = _rand_pi_S(rng, A)
    delta = 0.4
    t = 1.0
    X = 0
    W_c, _ = gtr_free_end_hr_avg_case1(pi, S, X, t, delta)
    # Direct outer integral over tau_1.
    tau_grid = np.linspace(1e-6, t, 501)
    if abs(delta) < 1e-12:
        Z = t
    else:
        Z = (np.exp(delta * t) - 1.0) / delta
    p_tau = np.exp(delta * tau_grid) / Z
    # For each tau_1 in the grid, compute W_i(X; tau_1) via closed form.
    W_by_tau = np.stack([
        gtr_free_end_hr(pi, S, X, float(tau))[0]
        for tau in tau_grid], axis=0)
    W_d = np.trapz(W_by_tau * p_tau[:, None], tau_grid, axis=0)
    assert np.allclose(W_c, W_d, atol=1e-5), (
        f"case-1 averaged W mismatch: closed={W_c} direct={W_d}")


# ---------------------------------------------------------------------------
# Compound-CTMC direct evaluation of cluster HR stats: used as gold standard
# for the closed-form cluster accumulator (which composes the primitives
# above with the case-decomposition posteriors).
# ---------------------------------------------------------------------------

def _build_compound_Q(rho, rho_chain, pi_archetype, arch_assignment,
                        classes, S):
    """Build the full compound CTMC generator on state space L * A^N.

    State encoding: idx = theta * (A^N) + sum_{n=0..N-1} x_n * A^(N-1-n).
    """
    L = rho.shape[0]
    N = classes.shape[0]
    K_a, A = pi_archetype.shape
    total = L * (A ** N)

    def encode(theta, xs):
        idx = int(theta)
        for x in xs:
            idx = idx * A + int(x)
        return idx

    def decode(idx):
        xs = []
        for _ in range(N):
            xs.append(idx % A)
            idx //= A
        return int(idx), tuple(reversed(xs))

    Q = np.zeros((total, total), dtype=np.float64)
    for src in range(total):
        theta, xs = decode(src)
        # (a) Residue substitutions per site (theta unchanged).
        for n in range(N):
            k_n = int(arch_assignment[int(classes[n]), theta])
            xn = xs[n]
            for y in range(A):
                if y == xn:
                    continue
                xs_new = list(xs); xs_new[n] = y
                dst = encode(theta, xs_new)
                Q[src, dst] += S[xn, y] * pi_archetype[k_n, y]
        # (b) Field jumps: theta -> theta', all residues resample from
        #     pi_archetype[arch[c_n, theta']]  independently.
        for theta_new in range(L):
            if theta_new == theta:
                continue
            rate_field = rho_chain * rho[theta_new]
            # Enumerate all target residue tuples (A^N of them).
            for xs_new_flat in range(A ** N):
                xs_new = []
                tmp = xs_new_flat
                for _ in range(N):
                    xs_new.append(tmp % A)
                    tmp //= A
                xs_new = tuple(reversed(xs_new))
                # Density: prod_n pi_arch[arch[c_n, theta_new]][x_new_n]
                w = 1.0
                for n in range(N):
                    k_n = int(arch_assignment[int(classes[n]), theta_new])
                    w *= float(pi_archetype[k_n, xs_new[n]])
                dst = encode(theta_new, xs_new)
                Q[src, dst] += rate_field * w
        # Diagonal from row-sum.
        Q[src, src] = -Q[src, :src].sum() - Q[src, src+1:].sum()
    return Q, encode


def compound_ctmc_bridge_hr(rho, rho_chain, pi_archetype, arch_assignment,
                              classes, X_obs, Y_obs, t, S):
    """Direct bridge HR on the compound CTMC. Returns (P_obs, V, U, W,
    N_theta) with per-archetype aggregations.

    P_obs is the joint observation probability marginalising theta_X, theta_Y
    (under X-rooted stationary theta_X ~ rho).
    """
    L = rho.shape[0]
    N = classes.shape[0]
    K_a, A = pi_archetype.shape

    Q, encode = _build_compound_Q(
        rho, rho_chain, pi_archetype, arch_assignment, classes, S)
    total = Q.shape[0]

    # Compute bridge weight matrix.
    P_all = expm(Q * t)

    # Enumerate source states (theta_X, X) with X = observed X_obs
    #    (marginal over theta_X with weight rho[theta_X]).
    src_indices_and_weights = []
    for theta_X in range(L):
        s_idx = encode(theta_X, tuple(int(x) for x in X_obs))
        src_indices_and_weights.append((theta_X, s_idx, float(rho[theta_X])))

    # Enumerate destination states (theta_Y, Y_obs).
    dst_indices = []
    for theta_Y in range(L):
        d_idx = encode(theta_Y, tuple(int(y) for y in Y_obs))
        dst_indices.append((theta_Y, d_idx))

    # P(observations) = sum src rho[theta_X] * sum dst P_all[src, dst]
    P_obs = 0.0
    for _, s_idx, w in src_indices_and_weights:
        for _, d_idx in dst_indices:
            P_obs += w * float(P_all[s_idx, d_idx])

    # Direct HR: E[dwell in state z | X, Y, t] via numerical integration.
    # E[T_z | s_X, s_Y, t] = int_0^t P_all_s(s_X, z) P_all_(t-s)(z, s_Y) ds / P_all_t(s_X, s_Y)
    n_grid = 51
    ss = np.linspace(0.0, t, n_grid)
    P_by_s = np.stack([expm(Q * s) for s in ss], axis=0)      # (n_grid, total, total)
    P_by_tms = np.stack([expm(Q * (t - s)) for s in ss], axis=0)

    V = np.zeros((K_a, A), dtype=np.float64)
    U = np.zeros((K_a, A, A), dtype=np.float64)
    W = np.zeros((K_a, A), dtype=np.float64)
    N_theta = 0.0

    # For W, U, N_theta we need to compute over each (theta, x_state).
    # Aggregate by (archetype-at-site-n, residue-at-site-n).
    for theta in range(L):
        for x_flat in range(A ** N):
            xs = []
            tmp = x_flat
            for _ in range(N):
                xs.append(tmp % A)
                tmp //= A
            xs = tuple(reversed(xs))
            z_idx = encode(theta, xs)
            # E[T_z | observations] * P_obs = sum over (s_idx, w, d_idx) w
            #   * int_0^t P_all_s(s_X, z) P_all_(t-s)(z, s_Y) ds
            E_T_z_times_Pobs = 0.0
            for _, s_idx, w in src_indices_and_weights:
                for _, d_idx in dst_indices:
                    integrand = P_by_s[:, s_idx, z_idx] * P_by_tms[:, z_idx, d_idx]
                    E_T_z_times_Pobs += w * np.trapz(integrand, ss)
            E_T_z = E_T_z_times_Pobs / max(P_obs, 1e-300)
            # Contribute to W[k(c_n, theta), x_n] for each site n.
            for n in range(N):
                k_n = int(arch_assignment[int(classes[n]), theta])
                W[k_n, xs[n]] += E_T_z

    # For U (transitions): sum over (z_src, z_dst) w P_all_s(sX, z_src) *
    # Q[z_src, z_dst] * P_all_(t-s)(z_dst, sY).
    # We aggregate:
    #   (a) Residue transitions (theta unchanged, one site x_n changes):
    #       contribute to U[k(c_n, theta), x_n_before, x_n_after].
    #   (b) Field transitions (theta changes): contribute to N_theta.
    for theta in range(L):
        for x_flat in range(A ** N):
            xs = []
            tmp = x_flat
            for _ in range(N):
                xs.append(tmp % A)
                tmp //= A
            xs = tuple(reversed(xs))
            src_z = encode(theta, xs)
            # Residue transitions per site.
            for n in range(N):
                k_n = int(arch_assignment[int(classes[n]), theta])
                for y in range(A):
                    if y == xs[n]:
                        continue
                    xs_new = list(xs); xs_new[n] = y
                    dst_z = encode(theta, xs_new)
                    q = float(Q[src_z, dst_z])
                    if q == 0.0:
                        continue
                    # E[N(src_z -> dst_z) | X, Y, t] = q * int_0^t
                    #   P_s(sX, src_z) * P_tms(dst_z, sY) ds / P_obs
                    e_n = 0.0
                    for _, s_idx, w in src_indices_and_weights:
                        for _, d_idx in dst_indices:
                            integrand = P_by_s[:, s_idx, src_z] * P_by_tms[:, dst_z, d_idx]
                            e_n += w * np.trapz(integrand, ss)
                    e_n = q * e_n / max(P_obs, 1e-300)
                    U[k_n, xs[n], y] += e_n
            # Field transitions from (theta, xs) to (theta_new, xs_new).
            for theta_new in range(L):
                if theta_new == theta:
                    continue
                # Sum over target xs_new.
                for xs_new_flat in range(A ** N):
                    xs_new = []
                    tmp = xs_new_flat
                    for _ in range(N):
                        xs_new.append(tmp % A)
                        tmp //= A
                    xs_new = tuple(reversed(xs_new))
                    dst_z = encode(theta_new, xs_new)
                    q = float(Q[src_z, dst_z])
                    if q == 0.0:
                        continue
                    e_n = 0.0
                    for _, s_idx, w in src_indices_and_weights:
                        for _, d_idx in dst_indices:
                            integrand = P_by_s[:, s_idx, src_z] * P_by_tms[:, dst_z, d_idx]
                            e_n += w * np.trapz(integrand, ss)
                    e_n = q * e_n / max(P_obs, 1e-300)
                    N_theta += e_n
                    # V: post-jump residue state contribution
                    #   For each site n: V[k(c_n, theta_new), xs_new[n]] += e_n
                    for n in range(N):
                        k_n_new = int(arch_assignment[int(classes[n]), theta_new])
                        V[k_n_new, xs_new[n]] += e_n

    # V also has t=0 boundary contribution: V[arch(c_n, theta_X), X_n] += P(theta_X | obs)
    for theta_X, s_idx, w in src_indices_and_weights:
        # P(theta_X | obs) = w * sum_dst P_all[s_idx, d_idx] / P_obs
        prob_theta_X = 0.0
        for _, d_idx in dst_indices:
            prob_theta_X += float(P_all[s_idx, d_idx])
        prob_theta_X *= w / max(P_obs, 1e-300)
        for n in range(N):
            k_n = int(arch_assignment[int(classes[n]), theta_X])
            V[k_n, int(X_obs[n])] += prob_theta_X

    return P_obs, V, U, W, float(N_theta)


def test_compound_ctmc_hr_small_N1():
    """Sanity check: compound CTMC direct evaluation returns finite,
    conservation-satisfying stats for a single-site cluster."""
    rng = np.random.default_rng(23)
    L = 2; A = 3; N = 1; K_a = 2
    rho, rho_chain = _rand_field(rng, L)
    pi_arch = rng.dirichlet(np.full(A, 2.0), size=K_a)
    arch_assignment = rng.integers(0, K_a, size=(1, L), dtype=np.int32)
    S_raw = rng.gamma(2.0, size=(A, A))
    S = 0.5 * (S_raw + S_raw.T)
    np.fill_diagonal(S, 0.0)
    classes = np.array([0], dtype=np.int64)
    X_obs = np.array([0], dtype=np.int64)
    Y_obs = np.array([1], dtype=np.int64)
    t = 0.5
    P_obs, V, U, W, N_theta = compound_ctmc_bridge_hr(
        rho, rho_chain, pi_arch, arch_assignment,
        classes, X_obs, Y_obs, t, S)
    # Sanity: P_obs > 0, W sum ~ t (dwell time)
    assert P_obs > 0
    assert abs(W.sum() - t) < 1e-3, f"W sum {W.sum()} != t {t}"
    # V should be non-negative
    assert (V >= -1e-9).all()
    # U diagonal is 0
    for k in range(K_a):
        assert np.allclose(np.diag(U[k]), 0.0)


def _cluster_fixture(rng, L, A, N, K_c, K_a, t):
    """Random small cluster fixture for comparison tests."""
    rho, rho_chain = _rand_field(rng, L)
    pi_arch = rng.dirichlet(np.full(A, 2.0), size=K_a)
    arch_assignment = rng.integers(0, K_a, size=(K_c, L), dtype=np.int64)
    S_raw = rng.gamma(2.0, size=(A, A))
    S = 0.5 * (S_raw + S_raw.T)
    np.fill_diagonal(S, 0.0)
    classes = rng.integers(0, K_c, size=N, dtype=np.int64)
    X_obs = rng.integers(0, A, size=N, dtype=np.int64)
    Y_obs = rng.integers(0, A, size=N, dtype=np.int64)
    return dict(rho=rho, rho_chain=rho_chain, pi_arch=pi_arch,
                arch_assignment=arch_assignment, S=S, classes=classes,
                X_obs=X_obs, Y_obs=Y_obs, t=t, A=A, L=L, N=N,
                K_a=K_a, K_c=K_c)


def _rel_err(a, b, eps=1e-9):
    denom = max(abs(a), abs(b), eps)
    return abs(a - b) / denom


def test_hr_cluster_stats_smoke_N1():
    """Sanity: cluster HR runs, W sums to ~ t * P_obs, U is off-diagonal only."""
    rng = np.random.default_rng(31)
    fx = _cluster_fixture(rng, L=2, A=3, N=1, K_c=1, K_a=2, t=0.3)
    P_obs, V, U_hr, W_hr, N_theta = hr_cluster_stats(
        fx['rho'], fx['rho_chain'], fx['pi_arch'], fx['arch_assignment'],
        fx['classes'], fx['X_obs'], fx['Y_obs'], fx['t'], fx['S'])
    assert P_obs > 0
    # W sums (over all archetypes/residues) should equal ~t * P_obs for a
    # single-site cluster (total dwell time). The case (2+) middle-stationary
    # residual is an approximation; within 5% is acceptable at short t.
    N = fx['N']
    assert abs(W_hr.sum() - fx['t'] * P_obs * N) < 0.05 * fx['t'] * P_obs * N, (
        f"W sum {W_hr.sum()} vs t*P_obs*N {fx['t'] * P_obs * N}")
    # U diagonal is 0
    for k in range(fx['K_a']):
        assert np.allclose(np.diag(U_hr[k]), 0.0)
    assert N_theta >= 0


def test_hr_cluster_stats_matches_compound_N1_short_t():
    """Compare hr_cluster_stats to compound_ctmc_bridge_hr for a small
    single-site cluster at short branch length. compound_ctmc_bridge_hr
    returns *conditional* HR (V, U, W, N_theta divided by P_obs);
    hr_cluster_stats returns *joint* (V, U, W, N_theta * P_obs). Normalise
    before comparing."""
    rng = np.random.default_rng(37)
    fx = _cluster_fixture(rng, L=2, A=3, N=1, K_c=1, K_a=2, t=0.15)
    P_c, V_c, U_c, W_c, Nt_c = hr_cluster_stats(
        fx['rho'], fx['rho_chain'], fx['pi_arch'], fx['arch_assignment'],
        fx['classes'], fx['X_obs'], fx['Y_obs'], fx['t'], fx['S'])
    P_r, V_r, U_r, W_r, Nt_r = compound_ctmc_bridge_hr(
        fx['rho'], fx['rho_chain'], fx['pi_arch'], fx['arch_assignment'],
        fx['classes'], fx['X_obs'], fx['Y_obs'], fx['t'], fx['S'])
    # Normalize closed-form to match compound-CTMC convention.
    V_c_n = V_c / max(P_c, 1e-300)
    U_c_n = U_c / max(P_c, 1e-300)
    W_c_n = W_c / max(P_c, 1e-300)
    Nt_c_n = Nt_c / max(P_c, 1e-300)
    assert _rel_err(P_c, P_r) < 0.02, f"P mismatch closed={P_c} ref={P_r}"
    for k in range(fx['K_a']):
        for a in range(fx['A']):
            if V_r[k, a] > 1e-3:
                assert _rel_err(V_c_n[k, a], V_r[k, a]) < 0.20, (
                    f"V[{k},{a}] closed={V_c_n[k, a]} ref={V_r[k, a]}")
            if W_r[k, a] > 1e-3:
                assert _rel_err(W_c_n[k, a], W_r[k, a]) < 0.20, (
                    f"W[{k},{a}] closed={W_c_n[k, a]} ref={W_r[k, a]}")
    assert _rel_err(Nt_c_n, Nt_r) < 0.05, (
        f"N_theta closed={Nt_c_n} ref={Nt_r}")


def test_hr_cluster_stats_matches_compound_sweep():
    """Sweep across several small fixtures to confirm the closed-form
    accumulator matches compound_ctmc within accuracy bounds at short-to-
    moderate branches (where the M=1 approximation for τ_M dominates).
    """
    rng = np.random.default_rng(101)
    for L in (2, 3):
        for K_a in (1, 3):
            for t in (0.1, 0.3):
                for seed in (0, 1):
                    fx_rng = np.random.default_rng(seed * 17 + L * 3 + K_a)
                    fx = _cluster_fixture(fx_rng, L=L, A=4, N=1, K_c=2,
                                            K_a=K_a, t=t)
                    P_c, V_c, U_c, W_c, Nt_c = hr_cluster_stats(
                        fx['rho'], fx['rho_chain'], fx['pi_arch'],
                        fx['arch_assignment'], fx['classes'],
                        fx['X_obs'], fx['Y_obs'], fx['t'], fx['S'])
                    P_r, V_r, U_r, W_r, Nt_r = compound_ctmc_bridge_hr(
                        fx['rho'], fx['rho_chain'], fx['pi_arch'],
                        fx['arch_assignment'], fx['classes'],
                        fx['X_obs'], fx['Y_obs'], fx['t'], fx['S'])
                    tag = f"L={L} K_a={K_a} t={t} seed={seed}"
                    assert _rel_err(P_c, P_r) < 0.05, (
                        f"{tag}: P closed={P_c} ref={P_r}")
                    V_c_n = V_c / max(P_c, 1e-300)
                    W_c_n = W_c / max(P_c, 1e-300)
                    for k in range(K_a):
                        for a in range(4):
                            if V_r[k, a] > 0.02:
                                assert _rel_err(V_c_n[k, a], V_r[k, a]) < 0.25, (
                                    f"{tag} V[{k},{a}] "
                                    f"closed={V_c_n[k, a]} ref={V_r[k, a]}")
                            if W_r[k, a] > 0.005:
                                assert _rel_err(W_c_n[k, a], W_r[k, a]) < 0.25, (
                                    f"{tag} W[{k},{a}] "
                                    f"closed={W_c_n[k, a]} ref={W_r[k, a]}")
                    assert _rel_err(Nt_c / max(P_c, 1e-300), Nt_r) < 0.15, (
                        f"{tag} N_theta closed={Nt_c/P_c} ref={Nt_r}")


def test_update_pi_archetype_gtr_recovers_truth():
    """Synthesize a large synthetic GTR trajectory in a known pi_true,
    accumulate HR sufficient stats analytically (from pi_true and S), then
    verify update_pi_archetype_gtr returns pi_true to high precision.

    Analytic accumulation: for a chain at stationarity in arch k over a
    long total time T, expected sufficient stats are
        V[k, a] = 0                     (no fresh starts if T is dwell only)
        W[k, i] = T * pi_true[i]        (stationary occupancy)
        U[k, i, j] = T * pi_true[i] * S[i, j] * pi_true[j]  (i != j)
    Optimising over pi with these stats must return pi_true exactly (up
    to the alpha_prior shift). We inject V[k, a] = N_starts * pi_true[a]
    as a mild "prior on starts" so V_bullet > 0 -- this is realistic
    since bridged HR always has at least the t=0 boundary V += 1 term.
    """
    from tkfdp.coupling.dynfield.hr import update_pi_archetype_gtr

    rng = np.random.default_rng(203)
    A = 4
    pi_true = rng.dirichlet(np.full(A, 2.0))
    S_raw = rng.gamma(2.0, size=(A, A))
    S = 0.5 * (S_raw + S_raw.T)
    np.fill_diagonal(S, 0.0)

    T_total = 1000.0
    N_starts = 100.0
    V = (N_starts * pi_true)[None, :]                          # (1, A)
    W = (T_total * pi_true)[None, :]                            # (1, A)
    U = np.zeros((1, A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            if i != j:
                U[0, i, j] = T_total * pi_true[i] * S[i, j] * pi_true[j]

    pi_est = update_pi_archetype_gtr(V, U, W, S, alpha_prior=1.0)[0]
    # With alpha_prior = 1, no shift; recovery should be to numerical
    # precision.
    assert np.allclose(pi_est, pi_true, atol=1e-6), (
        f"pi_est={pi_est}, pi_true={pi_true}, "
        f"max diff {np.max(np.abs(pi_est - pi_true))}")

    # Also verify that the solve is normalised.
    assert abs(pi_est.sum() - 1.0) < 1e-10


def test_update_pi_archetype_gtr_prior_shrinks_toward_uniform():
    """A strong Dirichlet(alpha_prior > 1) prior should shrink pi toward
    uniform. With zero data (V = U = W = 0) it should be exactly uniform."""
    from tkfdp.coupling.dynfield.hr import update_pi_archetype_gtr

    rng = np.random.default_rng(211)
    A = 4
    S_raw = rng.gamma(2.0, size=(A, A))
    S = 0.5 * (S_raw + S_raw.T)
    np.fill_diagonal(S, 0.0)
    V = np.zeros((1, A))
    U = np.zeros((1, A, A))
    W = np.zeros((1, A))
    pi_est = update_pi_archetype_gtr(V, U, W, S, alpha_prior=1.0)[0]
    # With zero counts the fallback goes to uniform.
    assert np.allclose(pi_est, np.full(A, 1.0 / A))


def test_update_rho_chain_gamma_map_and_mean():
    """Closed-form Gamma posterior update for rho_chain: MAP is
    (a-1+N)/(b+T); mean is (a+N)/(b+T). Verify both."""
    from tkfdp.coupling.dynfield.hr import update_rho_chain_gamma

    # With prior Gamma(a=1.5, b=5.0), N = 50, T = 25:
    # MAP  = (1.5 - 1 + 50) / (5 + 25) = 50.5 / 30 = 1.6833...
    # Mean = (1.5 + 50) / (5 + 25) = 51.5 / 30 = 1.7166...
    map_val = update_rho_chain_gamma(50.0, 25.0, prior_a=1.5, prior_b=5.0,
                                        mode="map")
    mean_val = update_rho_chain_gamma(50.0, 25.0, prior_a=1.5, prior_b=5.0,
                                         mode="mean")
    assert abs(map_val - 50.5 / 30.0) < 1e-10
    assert abs(mean_val - 51.5 / 30.0) < 1e-10

    # In the no-data limit, MAP should give (a-1)/b = 0.1 for Gamma(1.5, 5).
    assert abs(update_rho_chain_gamma(0.0, 0.0, prior_a=1.5, prior_b=5.0,
                                         mode="map")
                 - 0.5 / 5.0) < 1e-10


def test_update_pi_archetype_gtr_grows_toward_data_with_alpha_prior():
    """Verify concurrent behaviour: as data grows, the posterior mode
    approaches the MLE regardless of the prior."""
    from tkfdp.coupling.dynfield.hr import update_pi_archetype_gtr

    rng = np.random.default_rng(217)
    A = 4
    pi_true = rng.dirichlet(np.full(A, 2.0))
    S_raw = rng.gamma(2.0, size=(A, A))
    S = 0.5 * (S_raw + S_raw.T)
    np.fill_diagonal(S, 0.0)

    for T_total in (100.0, 10000.0):
        N_starts = T_total * 0.1
        V = (N_starts * pi_true)[None, :]
        W = (T_total * pi_true)[None, :]
        U = np.zeros((1, A, A))
        for i in range(A):
            for j in range(A):
                if i != j:
                    U[0, i, j] = T_total * pi_true[i] * S[i, j] * pi_true[j]
        pi_est = update_pi_archetype_gtr(V, U, W, S, alpha_prior=2.0)[0]
        # Effective pseudocount from alpha_prior is A * (alpha - 1) = 4,
        # so at data volume ~ T_total the deviation is ~ 4 / (V_bullet + 4).
        # Loose tolerance scales appropriately.
        assert np.allclose(pi_est, pi_true, atol=10.0 / max(T_total, 1.0))


if __name__ == "__main__":
    test_field_bridge_hr_matches_direct()
    test_field_bridge_EN_matches_gillespie()
    test_field_case_probs_sum_to_one()
    test_field_case_probs_matches_gillespie()
    test_gtr_bridge_hr_matches_direct()
    test_gtr_free_end_hr_matches_direct()
    test_gtr_free_end_avg_case1_matches_direct()
    test_compound_ctmc_hr_small_N1()
    test_hr_cluster_stats_smoke_N1()
    test_hr_cluster_stats_matches_compound_N1_short_t()
    test_hr_cluster_stats_matches_compound_sweep()
    test_update_pi_archetype_gtr_recovers_truth()
    test_update_pi_archetype_gtr_prior_shrinks_toward_uniform()
    test_update_rho_chain_gamma_map_and_mean()
    test_update_pi_archetype_gtr_grows_toward_data_with_alpha_prior()
    print("OK")
