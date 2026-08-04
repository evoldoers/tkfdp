"""Correctness: K_c-vmap c_n Gibbs equals sequential (M10c).

Runs one c_n Gibbs sweep in both padded (K_c-batched-copies) and
non-padded (K_c-sequential) modes on a small synthetic corpus.
Under a deterministic RNG, the sampled c_n values must match.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_kc_vmap_matches_sequential():
    from tkfdp.coupling.dynfield.phylo_elbo.gibbs_cn import (
        gibbs_cn_sweep)
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_cherry
    from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
        build_padded_tree)

    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    L = 2; K_c = 3; A = 4
    pi_field = rng_a.dirichlet(np.ones(A), size=(K_c, L))
    # Consume the same number of RNG draws so both rngs proceed in step.
    _ = rng_b.dirichlet(np.ones(A), size=(K_c, L))
    rho = np.full(L, 1.0 / L)
    S = np.ones((A, A)) - np.eye(A)
    rho_chain = 0.05

    # Small synthetic corpus.
    n_clusters = 6
    m = 3
    clusters_raw = []
    for _ in range(n_clusters):
        leaf_obs = rng_a.integers(0, A, size=(2, m)).astype(np.int32)
        _ = rng_b.integers(0, A, size=(2, m))
        classes = rng_a.integers(0, K_c, size=m).astype(np.int32)
        _ = rng_b.integers(0, K_c, size=m)
        tree = make_cherry(tau=0.1, leaf_obs=leaf_obs)
        clusters_raw.append((tree, classes))

    clusters_seq = [(t, c.copy()) for t, c in clusters_raw]
    clusters_pad = [(build_padded_tree(t), c.copy()) for t, c in clusters_raw]

    rng_seq = np.random.default_rng(99)
    rng_pad = np.random.default_rng(99)

    new_seq, _info_seq = gibbs_cn_sweep(
        clusters_seq, pi_field, rho, S, rho_chain, K_c, rng_seq,
        padded=False)
    new_pad, _info_pad = gibbs_cn_sweep(
        clusters_pad, pi_field, rho, S, rho_chain, K_c, rng_pad,
        padded=True)

    for i in range(n_clusters):
        assert np.array_equal(new_seq[i], new_pad[i]), \
            f"cluster {i}: seq={new_seq[i]} pad={new_pad[i]}"
        print(f"  cluster {i}: {new_seq[i].tolist()}")


if __name__ == "__main__":
    test_kc_vmap_matches_sequential()
    print("M10c K_c-vmap test PASS")
