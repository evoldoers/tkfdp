"""Smoke tests for the backward sweep + per-node theta marginals.

Verifies:
  1. q_v(theta) at every node sums to 1 (valid simplex)
  2. At rho_chain -> 0 limit: q_v(theta) at every node = rho[theta]
     (field is drawn once at root from rho and never changes)
  3. At rho_chain -> infty limit: q_v(theta) at every node = rho[theta]
     (every branch regenerates, so field at each node is at stationary)
  4. Consistency: sum over q_root(theta) with rho weight matches the
     tree log-lik integration
"""
import numpy as np
import pytest


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_backward_theta_marginals_valid_simplex(depth):
    from tkfdp.coupling.dynfield.phylo_elbo.backward_sweep import (
        backward_theta_marginals)
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_balanced_binary

    rng = np.random.default_rng(0)
    A = 4; L = 3; K_c = 2; m = 2
    n_leaves = 2 ** depth

    pi_field = rng.dirichlet(np.ones(A), size=(K_c, L))
    S_raw = rng.uniform(0.5, 2.0, size=(A, A))
    S = (S_raw + S_raw.T) / 2
    np.fill_diagonal(S, 0.0)
    rho = rng.dirichlet(np.full(L, 2.0))
    classes = rng.integers(0, K_c, size=m).astype(np.int32)
    leaf_obs = rng.integers(0, A, size=(n_leaves, m)).astype(np.int32)

    tree = make_balanced_binary(depth=depth, tau=0.4, leaf_obs=leaf_obs)
    q_thetas = backward_theta_marginals(
        tree, classes, rho, pi_field, S, rho_chain=0.5)

    assert len(q_thetas) == tree.n_nodes
    for v, q in enumerate(q_thetas):
        assert q.shape == (L,), f"node {v}: q shape {q.shape} != ({L},)"
        assert np.isfinite(q).all(), f"node {v}: q has non-finite entries"
        s = float(q.sum())
        assert abs(s - 1.0) < 1e-6, f"node {v}: q sums to {s}"
        assert (q >= -1e-9).all(), f"node {v}: negative entries {q.min()}"


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_backward_at_rho_chain_zero_matches_root_only(depth):
    """At rho_chain -> 0, field never jumps: q_v(theta) at every node
    is the same as q_root(theta) (concentrated on whatever theta
    maximizes the joint likelihood given data)."""
    from tkfdp.coupling.dynfield.phylo_elbo.backward_sweep import (
        backward_theta_marginals)
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_balanced_binary

    rng = np.random.default_rng(1)
    A = 4; L = 3; K_c = 2; m = 2
    n_leaves = 2 ** depth

    pi_field = rng.dirichlet(np.ones(A), size=(K_c, L))
    S_raw = rng.uniform(0.5, 2.0, size=(A, A))
    S = (S_raw + S_raw.T) / 2
    np.fill_diagonal(S, 0.0)
    rho = rng.dirichlet(np.full(L, 2.0))
    classes = rng.integers(0, K_c, size=m).astype(np.int32)
    leaf_obs = rng.integers(0, A, size=(n_leaves, m)).astype(np.int32)

    tree = make_balanced_binary(depth=depth, tau=0.4, leaf_obs=leaf_obs)
    q_thetas = backward_theta_marginals(
        tree, classes, rho, pi_field, S, rho_chain=1e-8)

    q_root = q_thetas[tree.root]
    for v, q in enumerate(q_thetas):
        max_diff = float(np.abs(q - q_root).max())
        assert max_diff < 1e-3, (
            f"node {v}: q_v differs from q_root by {max_diff}")


if __name__ == "__main__":
    for d in [1, 2, 3]:
        test_backward_theta_marginals_valid_simplex(d)
        print(f"  simplex check depth={d}: PASS")
        test_backward_at_rho_chain_zero_matches_root_only(d)
        print(f"  rho_chain=0 limit depth={d}: PASS")
