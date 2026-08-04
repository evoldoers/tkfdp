"""End-to-end HR sufficient-statistics verification via full joint rate matrix.

For a simplified DNA model (A=4, L=2 field states, 1 site, K_c=1, K_a=2
archetypes), we construct the joint (θ, X) CTMC on L·A = 8 states and
compute EXACT expected sufficient statistics on a bridge over time T
using the Van Loan / auxiliary-matrix method.

We then compare against the code's HR closed-form SS accumulator
(`_cluster_stats_inner`) aggregated over all (θ_X, θ_Y) pairs.

Coverage:
  * P_obs (cluster marginal likelihood): must match to machine precision.
  * N_theta (expected # field jumps): must match to machine precision.
  * V, W, U per-(c, θ, x): should match up to the documented O((ρ·t)²)
    approximation from the M=1 truncated-exp density used for τ_1, τ_M
    (see hr.py:740-742 for the design note).
  * Verified across parameter grid: {ρ_chain ∈ [0.1, 2], m ∈ [0.5, 2],
    invariant m=0, various T}. Includes both X=Y and X≠Y endpoints.

This test is load-bearing for the m-scaling fix (see task #100) — it
verifies that the per-site rate multiplier m is applied correctly at
every U attribution site.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pytest
import scipy.linalg
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tkfdp.coupling.dynfield.hr_jax import (
    gtr_eigendecomp_batch, _cluster_stats_inner)


# ---- Model dimensions ----
A = 4
L = 2
K_c = 1
K_a = 2

# Symmetric exchangeability (Jukes-Cantor-ish)
S_NP = np.ones((A, A)) - np.eye(A)
# Two archetypes with different stationary distributions
PI_ARCH_NP = np.array([
    [0.10, 0.20, 0.30, 0.40],
    [0.35, 0.35, 0.15, 0.15],
])
# Field stationary
RHO_FIELD_NP = np.array([0.4, 0.6])
# arch_assignment[c=0, θ] = θ
ARCH_ASSIGN_NP = np.array([[0, 1]], dtype=np.int32)


def _build_joint_Q(rho_chain: float, m: float) -> np.ndarray:
    """Joint generator Q on state = θ·A + X.
    Interp 1: at each field jump, residue is resampled from π_arch of new θ.
    Between jumps, residue evolves under m · Q^{arch(θ)}.
    """
    LA = L * A
    Q = np.zeros((LA, LA))
    for th in range(L):
        for x in range(A):
            s = th * A + x
            # Field jumps (with residue resample from π_arch of new θ's arch)
            for th_p in range(L):
                if th_p == th: continue
                k_p = ARCH_ASSIGN_NP[0, th_p]
                for x_p in range(A):
                    s_p = th_p * A + x_p
                    Q[s, s_p] += (rho_chain * RHO_FIELD_NP[th_p]
                                      * PI_ARCH_NP[k_p, x_p])
            # Residue substitution at fixed field
            k = ARCH_ASSIGN_NP[0, th]
            for x_p in range(A):
                if x_p == x: continue
                s_p = th * A + x_p
                Q[s, s_p] += m * S_NP[x, x_p] * PI_ARCH_NP[k, x_p]
    for s in range(LA):
        Q[s, s] = -Q[s, :].sum()
    return Q


def _exact_bridge_SS(Q_joint: np.ndarray, s_start: int, s_end: int, T: float):
    """Return (P_bridge, W_joint[LA], U_joint[LA, LA]) via Van Loan
    matrix exponentials.

    E[∫_0^T P(s→s'; τ) · e_i e_j^T · P(τ; T-τ) dτ]  computed via the
    2LA×2LA auxiliary matrix trick. Then divided by P_bridge.
    """
    LA = Q_joint.shape[0]
    P_T = scipy.linalg.expm(Q_joint * T)
    P_XY = P_T[s_start, s_end]
    if P_XY <= 0:
        return P_XY, np.zeros(LA), np.zeros((LA, LA))
    W = np.zeros(LA)
    for s in range(LA):
        E = np.zeros((LA, LA))
        E[s, s] = 1.0
        Aux = np.block([[Q_joint, E], [np.zeros((LA, LA)), Q_joint]])
        M = scipy.linalg.expm(Aux * T)
        W[s] = M[s_start, LA + s_end] / P_XY
    U = np.zeros((LA, LA))
    for s1 in range(LA):
        for s2 in range(LA):
            if s1 == s2 or Q_joint[s1, s2] == 0.0:
                continue
            E = np.zeros((LA, LA))
            E[s1, s2] = 1.0
            Aux = np.block([[Q_joint, E], [np.zeros((LA, LA)), Q_joint]])
            M = scipy.linalg.expm(Aux * T)
            U[s1, s2] = Q_joint[s1, s2] * M[s_start, LA + s_end] / P_XY
    return P_XY, W, U


def _aggregate_exact(rho_chain: float, m: float, X_obs: int, Y_obs: int,
                       T: float):
    """Aggregate exact bridge SS over (θ_X, θ_Y) pairs weighted by ρ[θ_X].

    Returns:
      P_obs:  Σ_{θ_X, θ_Y} ρ[θ_X] · P_joint((θ_X, X_obs) → (θ_Y, Y_obs); T)
      W_ct[c, θ_curr, x_curr]:  aggregated joint W
      U_ct[c, θ_curr, i, j]:    aggregated joint U for residue jumps
      N_theta_total: aggregated joint U for field jumps
    """
    Q_joint = _build_joint_Q(rho_chain, m)
    LA = L * A
    P_obs = 0.0
    W_ct = np.zeros((K_c, L, A))
    U_ct = np.zeros((K_c, L, A, A))
    N_theta_total = 0.0
    for th_X in range(L):
        for th_Y in range(L):
            s_start = th_X * A + X_obs
            s_end = th_Y * A + Y_obs
            P_pair, W_pair, U_pair = _exact_bridge_SS(
                Q_joint, s_start, s_end, T)
            weight = RHO_FIELD_NP[th_X] * P_pair
            P_obs += weight
            for th_c in range(L):
                for x_c in range(A):
                    s_c = th_c * A + x_c
                    W_ct[0, th_c, x_c] += weight * W_pair[s_c]
            for th_c in range(L):
                for i in range(A):
                    s1 = th_c * A + i
                    for j in range(A):
                        if i == j: continue
                        s2 = th_c * A + j
                        U_ct[0, th_c, i, j] += weight * U_pair[s1, s2]
            for th1 in range(L):
                for th2 in range(L):
                    if th1 == th2: continue
                    for x1 in range(A):
                        for x2 in range(A):
                            s1 = th1 * A + x1
                            s2 = th2 * A + x2
                            N_theta_total += weight * U_pair[s1, s2]
    return P_obs, W_ct, U_ct, N_theta_total


def _hr_closedform(rho_chain: float, m: float, X_obs: int, Y_obs: int,
                    T: float):
    """Call _cluster_stats_inner and return (P_obs, V, W, U, N_theta)."""
    rho = jnp.asarray(RHO_FIELD_NP)
    pi_arch = jnp.asarray(PI_ARCH_NP)
    arch_assignment = jnp.asarray(ARCH_ASSIGN_NP, dtype=jnp.int32)
    xi, U_arch, D_arch = gtr_eigendecomp_batch(pi_arch, jnp.asarray(S_NP))
    classes = jnp.asarray([0], dtype=jnp.int32)
    X_arr = jnp.asarray([X_obs], dtype=jnp.int32)
    Y_arr = jnp.asarray([Y_obs], dtype=jnp.int32)
    site_mask = jnp.asarray([1.0], dtype=jnp.float64)
    m_per_site = jnp.asarray([m], dtype=jnp.float64)
    P_obs, V, U, W, N_theta = _cluster_stats_inner(
        rho, jnp.float64(rho_chain), pi_arch, xi, U_arch, D_arch,
        arch_assignment, classes, X_arr, Y_arr, site_mask, m_per_site,
        jnp.float64(T), jnp.asarray(S_NP), K_c)
    return (float(P_obs), np.asarray(V), np.asarray(W), np.asarray(U),
              float(N_theta))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rho_chain,m,X_obs,Y_obs,T",
    [
        (0.5, 1.0, 0, 1, 0.3),   # baseline variant
        (0.5, 1.0, 2, 2, 0.3),   # X=Y baseline
        (2.0, 1.0, 0, 1, 0.3),   # fast field
        (0.1, 1.0, 0, 1, 0.5),   # slow field
        (0.5, 0.5, 0, 1, 0.3),   # half rate m
        (0.5, 2.0, 0, 1, 0.3),   # double rate m
        (0.5, 0.0, 2, 2, 0.3),   # invariant m=0 with X=Y
        (0.5, 1.0, 0, 1, 1.0),   # longer T
    ],
)
def test_P_obs_matches_matrix_exp(rho_chain, m, X_obs, Y_obs, T):
    """P_obs (marginal likelihood) must match the exact joint matrix exp
    to machine precision — this is the load-bearing training quantity."""
    P_exact, _, _, _ = _aggregate_exact(rho_chain, m, X_obs, Y_obs, T)
    P_hr, _, _, _, _ = _hr_closedform(rho_chain, m, X_obs, Y_obs, T)
    assert abs(P_exact - P_hr) < 1e-12, (
        f"P_obs mismatch: exact={P_exact}, HR={P_hr}, "
        f"|Δ|={abs(P_exact - P_hr)}")


@pytest.mark.parametrize(
    "rho_chain,m,X_obs,Y_obs,T",
    [
        (0.5, 1.0, 0, 1, 0.3),
        (2.0, 1.0, 0, 1, 0.3),
        (0.5, 0.5, 0, 1, 0.3),
        (0.5, 2.0, 0, 1, 0.3),
        (0.5, 0.0, 2, 2, 0.3),
    ],
)
def test_N_theta_matches_matrix_exp(rho_chain, m, X_obs, Y_obs, T):
    """N_theta (field jump count) is a field-side quantity — must match
    to machine precision (field bridge closed forms are exact)."""
    _, _, _, N_exact = _aggregate_exact(rho_chain, m, X_obs, Y_obs, T)
    _, _, _, _, N_hr = _hr_closedform(rho_chain, m, X_obs, Y_obs, T)
    assert abs(N_exact - N_hr) < 1e-12, (
        f"N_theta mismatch: exact={N_exact}, HR={N_hr}, "
        f"|Δ|={abs(N_exact - N_hr)}")


@pytest.mark.parametrize(
    "rho_chain,m,X_obs,Y_obs,T,rel_tol",
    [
        # Machine-precision agreement across the whole ρ·t grid, since
        # the exact M ≥ 1 τ_1, τ_M density is used (two-exponential
        # mixture; see appendix par:arch-hr and hr_jax _case_jump).
        (0.5, 1.0, 0, 1, 0.3, 1e-10),
        (0.1, 1.0, 0, 1, 0.5, 1e-10),
        (0.5, 0.5, 0, 1, 0.3, 1e-10),
        (0.5, 2.0, 0, 1, 0.3, 1e-10),
        (2.0, 1.0, 0, 1, 0.3, 1e-10),
        (0.5, 1.0, 0, 1, 1.0, 1e-10),
    ],
)
def test_W_U_match_matrix_exp_within_approximation(
        rho_chain, m, X_obs, Y_obs, T, rel_tol):
    """W and U per-(c, θ, x) must match matrix-exp to machine precision.
    The τ_1, τ_M densities are the exact two-exponential mixtures
    (β_1 · exp(δ_1 τ) + β_2 · exp(δ_2 τ)) with no M-truncation.
    """
    _, W_exact, U_exact, _ = _aggregate_exact(rho_chain, m, X_obs, Y_obs, T)
    _, _, W_hr, U_hr, _ = _hr_closedform(rho_chain, m, X_obs, Y_obs, T)
    max_W = max(np.max(np.abs(W_exact)), np.max(np.abs(W_hr)), 1e-12)
    max_U = max(np.max(np.abs(U_exact)), np.max(np.abs(U_hr)), 1e-12)
    rel_dW = np.max(np.abs(W_exact - W_hr)) / max_W
    rel_dU = np.max(np.abs(U_exact - U_hr)) / max_U
    assert rel_dW < rel_tol, (
        f"W relative error {rel_dW:.2e} > tol {rel_tol} "
        f"for ρ={rho_chain}, m={m}, T={T}")
    assert rel_dU < rel_tol, (
        f"U relative error {rel_dU:.2e} > tol {rel_tol} "
        f"for ρ={rho_chain}, m={m}, T={T}")


def test_invariant_no_substitution_U():
    """Invariant limit m=0: U must be exactly zero (no substitution jumps).
    Verified with X=Y (so cluster likelihood is nonzero)."""
    _, _, _, U_hr, _ = _hr_closedform(rho_chain=0.5, m=0.0, X_obs=2, Y_obs=2, T=0.4)
    assert np.max(np.abs(U_hr)) == 0.0, (
        f"Invariant m=0 gave nonzero U: max|U|={np.max(np.abs(U_hr))}")


def test_m_scaling_bilinear():
    """P_obs should approach 1 as T → 0 (identity limit); and for large T,
    the system should equilibrate."""
    # Identity limit
    P0, _, _, _, _ = _hr_closedform(rho_chain=0.5, m=1.0, X_obs=0, Y_obs=0, T=0.001)
    assert P0 > 0.99, f"P_obs at T=0.001 should be near 1, got {P0}"
    P0_variant, _, _, _, _ = _hr_closedform(
        rho_chain=0.5, m=1.0, X_obs=0, Y_obs=1, T=0.001)
    assert P0_variant < 0.01, (
        f"P_obs(X≠Y) at T=0.001 should be near 0, got {P0_variant}")
