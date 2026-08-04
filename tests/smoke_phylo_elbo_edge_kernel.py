"""Smoke tests for phylo_elbo.edge_kernel (Phase 1).

Verifies:
  - β(θ, τ) has the exp form and hits the τ=0 and τ→∞ limits.
  - Q_field row sums vanish; ρ is left eigenvector with eigenvalue 0.
  - K_field is row stochastic; τ=0 is identity; τ→∞ marginals converge
    to ρ.
  - W = K_field - β · I recovers the ≥1-jump component; W row sums =
    1 - β; off-diagonal W matches K_field's off-diagonal.
  - Detailed balance: ρ[i] K_field[i, j] = ρ[j] K_field[j, i].
"""
from __future__ import annotations

import numpy as np

from tkfdp.phylo_elbo.edge_kernel import (
    beta_no_jump, field_generator, field_transition, jump_weight)


def test_beta_limits():
    rho = np.array([0.3, 0.5, 0.2])
    rho_chain = 0.4

    # τ = 0 → β = 1 for every θ.
    b0 = np.array([beta_no_jump(k, rho_chain, rho, 0.0) for k in range(3)])
    assert np.allclose(b0, 1.0), b0

    # τ = 1e6 → β → 0 (as long as 1 - ρ[θ] > 0).
    b_inf = np.array([beta_no_jump(k, rho_chain, rho, 1e6) for k in range(3)])
    assert np.allclose(b_inf, 0.0), b_inf

    # Analytic form: β = exp(-ρ_chain (1 - ρ[θ]) τ).
    tau = 2.5
    b = np.array([beta_no_jump(k, rho_chain, rho, tau) for k in range(3)])
    expected = np.exp(-rho_chain * (1.0 - rho) * tau)
    assert np.allclose(b, expected), (b, expected)
    print("beta OK")


def test_generator_stationarity():
    rho = np.array([0.4, 0.35, 0.25])
    rho_chain = 0.3
    Q = field_generator(rho_chain, rho)

    # Rows sum to 0.
    assert np.allclose(Q.sum(axis=1), 0.0), Q.sum(axis=1)

    # ρ^T Q = 0 (ρ is left eigenvector of Q with eigenvalue 0).
    lhs = rho @ Q
    assert np.allclose(lhs, 0.0, atol=1e-12), lhs

    # Diagonal entries match the exit-rate formula.
    for k in range(3):
        assert np.isclose(Q[k, k], -rho_chain * (1.0 - rho[k])), Q[k, k]

    # Off-diagonal matches ρ_chain * ρ[dest].
    for i in range(3):
        for j in range(3):
            if i != j:
                assert np.isclose(Q[i, j], rho_chain * rho[j])
    print("generator OK")


def test_field_transition_limits():
    rho = np.array([0.4, 0.35, 0.25])
    rho_chain = 0.3

    # τ = 0 → identity.
    K0 = field_transition(rho_chain, rho, 0.0)
    assert np.allclose(K0, np.eye(3)), K0

    # τ = 100 → each row → ρ.
    K_inf = field_transition(rho_chain, rho, 100.0)
    for i in range(3):
        assert np.allclose(K_inf[i], rho, atol=1e-6), (i, K_inf[i], rho)

    # Row stochasticity at any τ.
    for tau in (0.01, 0.5, 2.0, 10.0):
        K = field_transition(rho_chain, rho, tau)
        assert np.allclose(K.sum(axis=1), 1.0, atol=1e-10)
        assert (K >= -1e-12).all()

    # Detailed balance: ρ[i] K[i, j] = ρ[j] K[j, i].
    K = field_transition(rho_chain, rho, 1.0)
    for i in range(3):
        for j in range(3):
            lhs = rho[i] * K[i, j]
            rhs = rho[j] * K[j, i]
            assert np.isclose(lhs, rhs, atol=1e-12), (i, j, lhs, rhs)
    print("field_transition OK")


def test_jump_weight():
    rho = np.array([0.4, 0.35, 0.25])
    rho_chain = 0.3
    tau = 0.7

    K = field_transition(rho_chain, rho, tau)
    W = jump_weight(rho_chain, rho, tau)
    beta_vec = np.exp(-rho_chain * (1.0 - rho) * tau)

    # W diagonal = K_diag - β.
    for k in range(3):
        assert np.isclose(W[k, k], K[k, k] - beta_vec[k]), (k, W[k, k])
    # W off-diagonal = K off-diagonal.
    for i in range(3):
        for j in range(3):
            if i != j:
                assert np.isclose(W[i, j], K[i, j])

    # Row sums of W = 1 - β.
    assert np.allclose(W.sum(axis=1), 1.0 - beta_vec)

    # τ = 0 → W = 0.
    W0 = jump_weight(rho_chain, rho, 0.0)
    assert np.allclose(W0, 0.0), W0
    print("jump_weight OK")


if __name__ == "__main__":
    test_beta_limits()
    test_generator_stationarity()
    test_field_transition_limits()
    test_jump_weight()
    print("all Phase 1 phylo-ELBO smoke tests passed")
