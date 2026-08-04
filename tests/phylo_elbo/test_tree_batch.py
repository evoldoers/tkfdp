"""Agreement test: bucketed_tree_log_lik matches per-cluster numpy path.

Builds a small corpus of mixed-shape clusters (cherries, depth-2
balanced, unbalanced ((A,B),C), varying m widths) and verifies the
JAX bucketed pipeline matches per-cluster numpy tree_log_lik_mm to
machine precision.
"""
from __future__ import annotations

import numpy as np
import pytest


def _random_model(rng, L, m_max, A_alpha, K_c):
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        per_class_field_Q)
    pi_field = rng.dirichlet(np.ones(A_alpha), size=(K_c, L))
    S_raw = rng.uniform(0.5, 2.0, size=(A_alpha, A_alpha))
    S = (S_raw + S_raw.T) / 2
    np.fill_diagonal(S, 0.0)
    rho = rng.dirichlet(np.full(L, 2.0))
    Q = per_class_field_Q(pi_field, S)
    return {'pi_field': pi_field, 'S': S, 'rho': rho, 'Q': Q}


def test_bucketed_matches_per_cluster():
    from tkfdp.coupling.dynfield.phylo_elbo.tree import (
        build_tree, make_balanced_binary, make_cherry)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_log_lik import (
        tree_log_lik_mm)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_batch import (
        bucketed_tree_log_lik)

    rng = np.random.default_rng(0)
    L = 3
    A_alpha = 4
    K_c = 3
    rho_chain = 0.5
    M = _random_model(rng, L, m_max=5, A_alpha=A_alpha, K_c=K_c)

    clusters: 'list[tuple]' = []

    # A few cherries at various m widths.
    for _ in range(3):
        m = int(rng.integers(1, 5))
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
        leaf_obs = rng.integers(0, A_alpha, size=(2, m)).astype(np.int32)
        tree = make_cherry(tau=float(rng.uniform(0.1, 0.8)), leaf_obs=leaf_obs)
        clusters.append((tree, classes))

    # Balanced depth-2 (4 leaves).
    for _ in range(2):
        m = int(rng.integers(1, 4))
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
        leaf_obs = rng.integers(0, A_alpha, size=(4, m)).astype(np.int32)
        tree = make_balanced_binary(
            depth=2, tau=float(rng.uniform(0.1, 0.6)), leaf_obs=leaf_obs)
        clusters.append((tree, classes))

    # Unbalanced ((A,B), C).
    for _ in range(2):
        m = int(rng.integers(1, 3))
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
        parent = [3, 3, 4, 4, -1]
        branch = [0.4, 0.5, 0.8, 0.3, 0.0]
        leaf_obs = rng.integers(0, A_alpha, size=(3, m)).astype(np.int32)
        tree = build_tree(parent, branch, leaf_obs)
        clusters.append((tree, classes))

    # Per-cluster numpy reference.
    ll_np = np.array([
        tree_log_lik_mm(tree, classes, M['rho'], M['pi_field'],
                             M['S'], rho_chain)
        for tree, classes in clusters
    ])

    # Bucketed JAX.
    ll_jax = bucketed_tree_log_lik(
        clusters, M['rho'], M['pi_field'], M['S'], rho_chain)

    diffs = ll_jax - ll_np
    max_diff = float(np.abs(diffs).max())
    print(f"N_clusters={len(clusters)}  max |diff|={max_diff:.2e}")
    for i, d in enumerate(diffs):
        print(f"  cluster {i}: ll_np={ll_np[i]:+.10f} "
                f"ll_jax={ll_jax[i]:+.10f} diff={d:+.2e}")
    assert max_diff < 1e-9, f"max_diff={max_diff}"


if __name__ == "__main__":
    test_bucketed_matches_per_cluster()
    print("all M4 tests PASS")
