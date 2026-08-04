"""Correctness + wall-clock: K_a-vmap arch Gibbs equals sequential (M10b).

Runs one arch Gibbs sweep in both padded (K-vmapped) and non-padded
(K-sequential) modes on the same synthetic corpus and confirms
identical log-lik values across candidates. Also reports wall-clock.
"""
from __future__ import annotations

import time

import numpy as np
import pytest


def test_kvmap_matches_sequential():
    from tkfdp.coupling.dynfield.phylo_elbo.tree import (
        make_balanced_binary, make_cherry)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_batch import (
        bucketed_tree_log_lik, bucketed_tree_log_lik_padded_kvmap)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
        build_padded_tree)

    rng = np.random.default_rng(0)
    L = 2; K_c = 2; K_a = 3; A = 4
    pi_archetype = rng.dirichlet(np.ones(A), size=K_a)
    rho = np.full(L, 1.0 / L)
    S = np.ones((A, A)) - np.eye(A)
    rho_chain = 0.1

    # Small synthetic corpus.
    clusters = []
    for _ in range(6):
        m = int(rng.integers(1, 3))
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
        leaf_obs = rng.integers(0, A, size=(4, m)).astype(np.int32)
        tree = make_balanced_binary(depth=2, tau=0.2, leaf_obs=leaf_obs)
        clusters.append((tree, classes))

    padded_clusters = [(build_padded_tree(t), c) for t, c in clusters]

    # Build K_a variants of pi_field for a specific (c*, theta*).
    aa = np.tile(np.arange(min(K_c, K_a))[:, None],
                    (1, L)).astype(np.int32)
    c_star, theta_star = 0, 1
    pi_field_base = pi_archetype[aa]
    pi_field_variants = np.broadcast_to(
        pi_field_base[None, :, :, :], (K_a, K_c, L, A)).copy()
    for k in range(K_a):
        pi_field_variants[k, c_star, theta_star] = pi_archetype[k]

    # Sequential path.
    t0 = time.time()
    ll_seq = np.zeros((K_a, len(clusters)))
    for k in range(K_a):
        ll = bucketed_tree_log_lik(
            clusters, rho, pi_field_variants[k], S, rho_chain)
        ll_seq[k] = ll
    dt_seq = time.time() - t0

    # K-vmapped path.
    t0 = time.time()
    ll_kv = bucketed_tree_log_lik_padded_kvmap(
        padded_clusters, rho, pi_field_variants, S, rho_chain)
    dt_kv = time.time() - t0

    max_diff = float(np.max(np.abs(ll_seq - ll_kv)))
    print(f"seq  ll[0] = {ll_seq[0].round(4).tolist()}  dt={dt_seq:.2f}s")
    print(f"kv   ll[0] = {ll_kv[0].round(4).tolist()}  dt={dt_kv:.2f}s")
    print(f"max abs diff = {max_diff:.2e}")
    print(f"speedup: {dt_seq / max(dt_kv, 1e-9):.2f}x")
    assert max_diff < 1e-10, f"K-vmap mismatch: {max_diff}"


if __name__ == "__main__":
    test_kvmap_matches_sequential()
    print("M10b K-vmap test PASS")
