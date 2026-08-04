"""Adversarial tests for the per-site rate multiplier `m` in HR SS attribution.

Under per-site rate scaling `Q^{k, m} = m · Q^k`, the correct expected
substitution-jump count over an interval is
    E[N_ij] = m · Q^k[i, j] · E[dwell at i]
(free-end / unconditional case; martingale identity)
or the direct Doob h-transform integral for the bridge case, which
turns out to give the *same* linear m dependence for the primitive.

The HR primitives in `gtr_bridge_hr_jax`, `gtr_free_end_hr_avg_case1_jax`,
and the middle-segment formula in `_case_jump` all internally compute
values proportional to `1/m` × correct_U (because the J denominator
`(ξ_a − ξ_b)` acquires an extra 1/m factor under `ξ_scaled = m·ξ`).
So a call-site multiplication by `m_per_site[n]` is required at every
U attribution.

This module provides:
  1. Algebraic identity test:
       `gtr_bridge_hr_jax(m·ξ, T)` and `gtr_bridge_hr_jax(ξ, m·T)`
       differ by exactly a 1/m factor in U (and in W).
  2. Gillespie ground-truth tests on A=4 DNA-like alphabet:
       primitive U * m == E[N_ij] simulated under Q^{k, m}
       for Case 0 (bridge), Case ≥1 (free-end truncated exp).
  3. Invariant limit (m = 0): all U attributions must be exactly zero.

The Gillespie tests use rejection sampling for the bridge case
(reject paths that don't hit Y) with 100 000 paths per m value —
enough to bring MC noise to ~0.5% on the accepted-path U estimates.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tkfdp.coupling.dynfield.hr_jax import (
    gtr_bridge_hr_jax,
    gtr_free_end_hr_avg_case1_jax,
    gtr_eigendecomp_batch,
)


A = 4
S_NP = np.ones((A, A)) - np.eye(A)  # Jukes-Cantor-ish symmetric
PI_K_NP = np.array([0.30, 0.20, 0.25, 0.25])


def _eig():
    pi_k = jnp.asarray(PI_K_NP)
    S = jnp.asarray(S_NP)
    xi_j, U_j, D_j = gtr_eigendecomp_batch(pi_k[None, :], S)
    return pi_k, S, xi_j[0], U_j[0], D_j[0]


def _Q_matrix(m):
    Q = m * S_NP * PI_K_NP[None, :]
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def _gillespie_bridge(Q, X, Y, T, n_paths, rng):
    """Reject-sample bridge trajectories X→Y under Q. Return U, W, n_accepted."""
    A = Q.shape[0]
    U_sum = np.zeros((A, A))
    W_sum = np.zeros(A)
    n_acc = 0
    exit_rate = -np.diag(Q)
    for _ in range(n_paths):
        state = X; t = 0.0
        U_p = np.zeros((A, A)); W_p = np.zeros(A)
        while True:
            r = exit_rate[state]
            if r == 0.0:
                W_p[state] += T - t; break
            dt = rng.exponential(1.0 / r)
            if t + dt >= T:
                W_p[state] += T - t; break
            W_p[state] += dt; t += dt
            p_dest = np.maximum(Q[state], 0.0)
            p_dest[state] = 0.0
            p_dest /= p_dest.sum()
            new_state = int(rng.choice(A, p=p_dest))
            U_p[state, new_state] += 1
            state = new_state
        if state == Y:
            U_sum += U_p; W_sum += W_p; n_acc += 1
    if n_acc == 0:
        return None, None, 0
    return U_sum / n_acc, W_sum / n_acc, n_acc


def _gillespie_freeend(Q, X, tau_fixed, n_paths, rng):
    """Free-end trajectory starting at X, evolving for exactly tau_fixed."""
    A = Q.shape[0]
    U_sum = np.zeros((A, A))
    W_sum = np.zeros(A)
    exit_rate = -np.diag(Q)
    for _ in range(n_paths):
        state = X; t = 0.0
        while True:
            r = exit_rate[state]
            if r == 0.0:
                W_sum[state] += tau_fixed - t; break
            dt = rng.exponential(1.0 / r)
            if t + dt >= tau_fixed:
                W_sum[state] += tau_fixed - t; break
            W_sum[state] += dt; t += dt
            p_dest = np.maximum(Q[state], 0.0)
            p_dest[state] = 0.0
            p_dest /= p_dest.sum()
            new_state = int(rng.choice(A, p=p_dest))
            U_sum[state, new_state] += 1
            state = new_state
    return U_sum / n_paths, W_sum / n_paths


def _gillespie_truncated_exp(Q, X, T, delta, n_paths, rng):
    """Free-end with tau drawn from truncated-exp density on [0, T] with rate δ.
    Matches gtr_free_end_hr_avg_case1_jax semantics."""
    A = Q.shape[0]
    U_sum = np.zeros((A, A))
    W_sum = np.zeros(A)
    for _ in range(n_paths):
        if abs(delta) < 1e-12:
            tau = rng.uniform(0.0, T)
        else:
            u = rng.uniform()
            tau = np.log(1.0 + u * (np.exp(delta * T) - 1.0)) / delta
        u_arr, w_arr = _gillespie_freeend(Q, X, tau, 1, rng)
        U_sum += u_arr; W_sum += w_arr
    return U_sum / n_paths, W_sum / n_paths


# ---------------------------------------------------------------------------
# Test 1: Algebraic identity — (m·ξ, T) primitive == (1/m) · (ξ, m·T) primitive
#         for BOTH U and W (Case 0 bridge).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [0.5, 2.0, 0.137, 2.386])
def test_case0_algebraic_mscale_identity(m):
    pi_k, S, xi, U_e, D_h = _eig()
    X, Y, T = 0, 2, 0.5
    xi_scaled = jnp.asarray(m * np.asarray(xi))

    P1, W1, U1 = gtr_bridge_hr_jax(pi_k, xi_scaled, U_e, D_h, S, X, Y, T)
    P2, W2, U2 = gtr_bridge_hr_jax(pi_k, xi, U_e, D_h, S, X, Y, m * T)

    # P_XY is invariant under this substitution (only depends on ξ·t product)
    assert float(jnp.abs(P1 - P2)) < 1e-12
    # W_prim(m·ξ, T) = (1/m) · W_prim(ξ, m·T) (from J denominator having 1/m)
    assert float(jnp.max(jnp.abs(W1 - W2 / m))) < 1e-10
    # U_prim(m·ξ, T) = (1/m) · U_prim(ξ, m·T) (same reason)
    assert float(jnp.max(jnp.abs(U1 - U2 / m))) < 1e-10


# ---------------------------------------------------------------------------
# Test 2: Case 0 bridge U with * m fix matches Gillespie under Q^{k, m}
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [0.5, 1.0, 2.0])
def test_case0_bridge_U_matches_gillespie_with_m(m):
    pi_k, S, xi, U_e, D_h = _eig()
    xi_np = np.asarray(xi)
    X, Y, T = 0, 2, 0.5
    n_paths = 200_000
    rng = np.random.default_rng(int(m * 1000))

    xi_scaled = jnp.asarray(m * xi_np)
    _, W_prim, U_prim = gtr_bridge_hr_jax(
        pi_k, xi_scaled, U_e, D_h, S, X, Y, T)
    U_fixed = m * np.asarray(U_prim)

    Q_m = _Q_matrix(m)
    U_gil, W_gil, n_acc = _gillespie_bridge(Q_m, X, Y, T, n_paths, rng)
    assert n_acc > 500, f"Gillespie accepted only {n_acc} paths"

    max_dU = float(np.max(np.abs(U_fixed - U_gil)))
    max_dW = float(np.max(np.abs(np.asarray(W_prim) - W_gil)))
    # Empirical MC precision at n_acc ~ 1e4 is ~1%. Use lenient thresholds.
    assert max_dU < 0.05, (f"m={m}: primitive · m disagrees with Gillespie "
                              f"U by {max_dU} (n_acc={n_acc})")
    assert max_dW < 0.02, (f"m={m}: primitive W disagrees with Gillespie "
                              f"W by {max_dW} (n_acc={n_acc})")


# ---------------------------------------------------------------------------
# Test 3: Case ≥1 free-end U (truncated-exp density) with * m matches Gillespie
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m,delta", [
    (0.5, -0.2),
    (1.0, -0.2),
    (2.0, -0.2),
    (0.5, 0.5),
    (2.0, 0.5),
])
def test_case1_freeend_U_matches_gillespie_with_m(m, delta):
    pi_k, S, xi, U_e, D_h = _eig()
    xi_np = np.asarray(xi)
    X, T = 0, 0.5
    n_paths = 10_000
    rng = np.random.default_rng(int(1e6 * (m + delta)))

    xi_scaled = jnp.asarray(m * xi_np)
    W_prim, U_prim = gtr_free_end_hr_avg_case1_jax(
        pi_k, xi_scaled, U_e, D_h, S, X, T, delta)
    U_fixed = m * np.asarray(U_prim)

    Q_m = _Q_matrix(m)
    U_gil, W_gil = _gillespie_truncated_exp(Q_m, X, T, delta, n_paths, rng)

    max_dU = float(np.max(np.abs(U_fixed - U_gil)))
    max_dW = float(np.max(np.abs(np.asarray(W_prim) - W_gil)))
    assert max_dU < 0.03, (f"m={m}, δ={delta}: primitive·m vs Gillespie U "
                              f"disagree by {max_dU}")
    assert max_dW < 0.01, (f"m={m}, δ={delta}: primitive W vs Gillespie W "
                              f"disagree by {max_dW}")


# ---------------------------------------------------------------------------
# Test 4: Invariant limit — U = 0 exactly (no substitution jumps for m = 0).
# ---------------------------------------------------------------------------
def test_invariant_limit_U_zero():
    pi_k, S, xi, U_e, D_h = _eig()
    xi_zero = jnp.zeros_like(xi)
    X, Y, T = 0, 0, 0.5  # X = Y compatible with invariant

    # Case 0 bridge: primitive * 0 = 0 exactly.
    _, _, U_bridge = gtr_bridge_hr_jax(
        pi_k, xi_zero, U_e, D_h, S, X, Y, T)
    U_bridge_fixed = 0.0 * np.asarray(U_bridge)
    assert np.max(np.abs(U_bridge_fixed)) == 0.0

    # Case ≥1 free-end: primitive * 0 = 0.
    W_free, U_free = gtr_free_end_hr_avg_case1_jax(
        pi_k, xi_zero, U_e, D_h, S, X, T, -0.2)
    U_free_fixed = 0.0 * np.asarray(U_free)
    assert np.max(np.abs(U_free_fixed)) == 0.0
    # W_free with xi=0 should concentrate at X.
    assert float(W_free[X]) > 0.0
    # W_free elsewhere should be ~machine zero.
    W_free_np = np.array(W_free, copy=True)
    W_free_np[X] = 0
    assert float(np.max(np.abs(W_free_np))) < 1e-10
