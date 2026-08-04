"""Correctness + timing: bucketed_tree_log_lik_padded_binned vs
bucketed_tree_log_lik_padded on a multi-cluster synthetic corpus.
"""
from __future__ import annotations

import time

import numpy as np
import pytest


def test_bucketed_binned_matches_padded_at_zero_bin_error():
    """When every branch tau is on the geomspace grid exactly (or
    when n_bins is very large), the binned path should match the
    unbinned path to within numeric precision on the P_sub matmuls."""
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import per_class_field_Q
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_cherry
    from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
        build_padded_tree)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_batch import (
        bucketed_tree_log_lik_padded,
        bucketed_tree_log_lik_padded_binned)
    from tkfdp.coupling.dynfield.phylo_elbo.tau_binning import (
        build_tau_bins, collect_all_taus, precompute_kernel_tables,
        rebuild_padded_trees_with_bins)

    rng = np.random.default_rng(3)
    L = 2; K_c = 2; A = 4
    pi_field = rng.dirichlet(np.ones(A), size=(K_c, L))
    rho = np.full(L, 1.0 / L)
    S = np.ones((A, A)) - np.eye(A)
    rho_chain = 0.1

    # Cherries all at same tau (single unique value -> lands exactly on grid).
    n_clusters = 5
    m = 2
    clusters_raw = []
    for _ in range(n_clusters):
        leaf_obs = rng.integers(0, A, size=(2, m)).astype(np.int32)
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
        tree = make_cherry(tau=0.4, leaf_obs=leaf_obs)
        clusters_raw.append((tree, classes))
    padded_clusters = [(build_padded_tree(t), c) for t, c in clusters_raw]

    # Unbinned reference.
    ll_ref = bucketed_tree_log_lik_padded(
        padded_clusters, rho, pi_field, S, rho_chain)

    # Binned path with lots of bins so grid contains 0.4 exactly.
    pts = [pt for pt, _ in padded_clusters]
    all_taus = collect_all_taus(pts)
    bin_centers = build_tau_bins(all_taus, n_bins=64,
                                          t_lo=0.4, t_hi=0.4)
    # With t_lo == t_hi, all bins collapse; force n_bins=1.
    bin_centers = np.array([0.4])
    _, bin_idx_per_tree = rebuild_padded_trees_with_bins(pts, bin_centers)
    tables = precompute_kernel_tables(
        bin_centers, rho, rho_chain, pi_field, S)

    ll_bin = bucketed_tree_log_lik_padded_binned(
        padded_clusters, bin_idx_per_tree, rho, tables)

    max_diff = float(np.max(np.abs(ll_ref - ll_bin)))
    print(f"unbinned: {ll_ref.round(4).tolist()}")
    print(f"binned:   {ll_bin.round(4).tolist()}")
    print(f"max abs diff: {max_diff:.2e}")
    assert max_diff < 1e-10, f"n_bins=1 exact-tau mismatch: {max_diff}"


def test_bucketed_binned_wall_clock():
    """Time both paths on a bigger corpus, mixed tau."""
    from tkfdp.coupling.dynfield.phylo_elbo.tree import (
        make_balanced_binary, make_cherry)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
        build_padded_tree)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_batch import (
        bucketed_tree_log_lik_padded,
        bucketed_tree_log_lik_padded_binned)
    from tkfdp.coupling.dynfield.phylo_elbo.tau_binning import (
        build_tau_bins, collect_all_taus, precompute_kernel_tables,
        rebuild_padded_trees_with_bins)

    rng = np.random.default_rng(4)
    L = 3; K_c = 3; A = 4
    pi_field = rng.dirichlet(np.ones(A), size=(K_c, L))
    rho = np.full(L, 1.0 / L)
    S = np.ones((A, A)) - np.eye(A)
    rho_chain = 0.1

    # Mixed synthetic corpus.
    n_clusters = 10
    m = 3
    clusters_raw = []
    for _ in range(n_clusters):
        leaf_obs = rng.integers(0, A, size=(4, m)).astype(np.int32)
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
        tau = float(rng.uniform(0.1, 1.0))
        tree = make_balanced_binary(depth=2, tau=tau, leaf_obs=leaf_obs)
        clusters_raw.append((tree, classes))
    padded_clusters = [(build_padded_tree(t), c) for t, c in clusters_raw]

    pts = [pt for pt, _ in padded_clusters]
    all_taus = collect_all_taus(pts)
    bin_centers = build_tau_bins(all_taus, n_bins=16)
    _, bin_idx_per_tree = rebuild_padded_trees_with_bins(pts, bin_centers)
    tables = precompute_kernel_tables(
        bin_centers, rho, rho_chain, pi_field, S)

    # Warm-up JIT for both paths (first call includes compile).
    _ = bucketed_tree_log_lik_padded(
        padded_clusters, rho, pi_field, S, rho_chain)
    _ = bucketed_tree_log_lik_padded_binned(
        padded_clusters, bin_idx_per_tree, rho, tables)

    # Timed.
    dts_ref = []; dts_bin = []
    for _ in range(3):
        t0 = time.time()
        ll_ref = bucketed_tree_log_lik_padded(
            padded_clusters, rho, pi_field, S, rho_chain)
        dts_ref.append(time.time() - t0)
    for _ in range(3):
        t0 = time.time()
        ll_bin = bucketed_tree_log_lik_padded_binned(
            padded_clusters, bin_idx_per_tree, rho, tables)
        dts_bin.append(time.time() - t0)

    max_diff = float(np.max(np.abs(ll_ref - ll_bin)))
    print(f"unbinned: {np.mean(dts_ref)*1000:.1f}ms")
    print(f"binned:   {np.mean(dts_bin)*1000:.1f}ms")
    print(f"speedup: {np.mean(dts_ref)/max(np.mean(dts_bin), 1e-9):.2f}x")
    print(f"max abs LL diff (binning approx): {max_diff:.2e}")


def _rand_binary_tree(n_leaves, m, A, rng):
    """Random binary tree (coalescent-style) with `n_leaves` leaves and
    `m` alignment columns. Leaves are nodes 0..n_leaves-1; internal nodes
    are created in order so the final (root) node is index n_nodes-1, as
    build_tree requires."""
    from tkfdp.coupling.dynfield.phylo_elbo.tree import build_tree
    n_nodes = 2 * n_leaves - 1
    parent = -np.ones(n_nodes, dtype=np.int32)
    bl = np.zeros(n_nodes, dtype=np.float64)
    active = list(range(n_leaves))
    nxt = n_leaves
    while len(active) > 1:
        i, j = rng.choice(len(active), size=2, replace=False)
        a, b = active[i], active[j]
        parent[a] = nxt
        parent[b] = nxt
        bl[a] = rng.uniform(0.1, 0.9)
        bl[b] = rng.uniform(0.1, 0.9)
        active = [x for k, x in enumerate(active) if k not in (i, j)] + [nxt]
        nxt += 1
    leaf_obs = rng.integers(0, A, size=(n_leaves, m)).astype(np.int32)
    return build_tree(parent, bl, leaf_obs)


def test_slot_padding_is_inert():
    """A cluster's binned log-lik must be identical whether it is scored
    alone or batched alongside larger clusters.

    This is the invariant the sqrt2 slot-bucket padding in
    `_bucketed_n_slots_per_level` relies on: rounding each level's slot
    count up to the sqrt2 grid (so the JIT variant set stays bounded) only
    ever ADDS zero-filled pad slots to the SMALLER trees in a bucket, and
    those pad slots must be numerically inert. Batching a small tree with
    larger ones (and letting the bucketer build the pad region the way
    real training does, via `_bucket_shape_batch`) is the faithful way to
    exercise that padding — NOT hand-appending slots to a single tree's own
    per-level arrays, which produces a slot layout the batched forward
    never generates.
    """
    from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
        build_padded_tree)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_batch import (
        bucketed_tree_log_lik_padded_binned, bucket_key_from_padded)
    from tkfdp.coupling.dynfield.phylo_elbo.tau_binning import (
        build_tau_bins, collect_all_taus, precompute_kernel_tables,
        rebuild_padded_trees_with_bins)

    rng = np.random.default_rng(7)
    L = 2; K_c = 2; A = 4; m = 4
    pi_field = rng.dirichlet(np.ones(A), size=(K_c, L))
    rho = np.full(L, 1.0 / L)
    S = np.ones((A, A)) - np.eye(A)
    rho_chain = 0.2

    # Mixed-size corpus: varied leaf counts -> clusters co-bucket at
    # different actual per-level slot counts, so batching pads the smaller
    # ones up to the bucket (and sqrt2) grid.
    clusters = []
    for _ in range(14):
        n_leaves = int(rng.integers(4, 10))
        tree = _rand_binary_tree(n_leaves, m, A, rng)
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
        clusters.append((build_padded_tree(tree), classes))

    pts = [pt for pt, _ in clusters]
    bin_centers = build_tau_bins(collect_all_taus(pts), n_bins=16)
    _, bin_idx = rebuild_padded_trees_with_bins(pts, bin_centers)
    tables = precompute_kernel_tables(bin_centers, rho, rho_chain,
                                              pi_field, S)
    if 'pi_field' not in tables:                 # older kernel-table dict
        tables = dict(tables); tables['pi_field'] = pi_field

    # Guard against a vacuous test: at least one bucket must contain
    # clusters with DIFFERING per-level slot counts, so real padding fires.
    by_bucket = {}
    for pt, cls in clusters:
        key = bucket_key_from_padded(pt, cls)
        prof = tuple(pt.n_slots(l + 1) for l in range(pt.D_bucket))
        by_bucket.setdefault(key, set()).add(prof)
    assert any(len(profiles) > 1 for profiles in by_bucket.values()), (
        "test is vacuous: no bucket mixes clusters of differing slot "
        "counts, so no padding is exercised")

    # Scored all together (small clusters get padded up to the bucket).
    ll_together = np.asarray(
        bucketed_tree_log_lik_padded_binned(clusters, bin_idx, rho, tables))

    # Scored one at a time (each cluster in isolation).
    ll_alone = np.array([
        float(bucketed_tree_log_lik_padded_binned(
            [clusters[i]], [bin_idx[i]], rho, tables)[0])
        for i in range(len(clusters))])

    max_diff = float(np.max(np.abs(ll_together - ll_alone)))
    print(f"max |batched - alone| diff over {len(clusters)} clusters: "
          f"{max_diff:.2e}")
    assert max_diff < 1e-9, f"slot padding perturbed LL: {max_diff}"


if __name__ == "__main__":
    test_bucketed_binned_matches_padded_at_zero_bin_error()
    test_bucketed_binned_wall_clock()
    test_slot_padding_is_inert()
    print("M12 bucketed binned tests PASS")
