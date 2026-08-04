"""Test BucketBatchCache (M10a): repeated calls hit the cache and
return identical results to no-cache path."""
from __future__ import annotations

import time

import numpy as np


def test_cache_agreement_and_reuse():
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_cherry
    from tkfdp.coupling.dynfield.phylo_elbo.tree_batch import (
        BucketBatchCache, bucketed_tree_log_lik_padded)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
        build_padded_tree)

    rng = np.random.default_rng(3)
    L = 2; K_c = 2; A = 4
    pi_field = rng.dirichlet(np.ones(A), size=(K_c, L))
    rho = np.full(L, 1.0 / L)
    S = np.ones((A, A)) - np.eye(A)
    rho_chain = 0.05

    clusters = []
    for _ in range(4):
        m = int(rng.integers(1, 3))
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
        leaf_obs = rng.integers(0, A, size=(2, m)).astype(np.int32)
        clusters.append((build_padded_tree(make_cherry(0.1, leaf_obs)),
                              classes))

    cache = BucketBatchCache()

    # Two calls, same clusters — second should hit the cache.
    ll_1 = bucketed_tree_log_lik_padded(
        clusters, rho, pi_field, S, rho_chain, cache=cache)
    ll_2 = bucketed_tree_log_lik_padded(
        clusters, rho, pi_field, S, rho_chain, cache=cache)
    ll_nocache = bucketed_tree_log_lik_padded(
        clusters, rho, pi_field, S, rho_chain)

    assert np.allclose(ll_1, ll_2, atol=1e-15)
    assert np.allclose(ll_1, ll_nocache, atol=1e-15)
    assert len(cache._entries) >= 1, \
        f"cache should hold >=1 entry, got {len(cache._entries)}"
    print(f"  cache holds {len(cache._entries)} entries; ll matches")


if __name__ == "__main__":
    test_cache_agreement_and_reuse()
    print("M10a cache test PASS")
