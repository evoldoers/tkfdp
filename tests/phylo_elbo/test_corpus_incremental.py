"""Equivalence tests for the incremental arch-move kernel updates in
`corpus_state` against the full-rebuild reference paths.

The arch moves (atomic_arch_swap / gswap / gibbs) were changed to reuse the
invariant beta/W tables and recompute only the perturbed P_sub slices
(n_bins expm) instead of rebuilding the full K_c*L*n_bins table per
candidate. These tests pin that the fast path reproduces the slow path:

  - `_score_affected_clusters_under_overrides` must equal
    `_score_affected_clusters_under_pi` on the corresponding full pi_field.
  - `apply_arch_slices` after mutating arch_assignment must leave the kernel
    tables bit-equal to a full `refresh_pi_field`.

Requires JAX (the forward is JAX). Skipped automatically if unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")


def _mk_family(family_id, L, depth, A, rng):
    """Build a synthetic FamilyState from a balanced binary tree."""
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_balanced_binary
    from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
        build_padded_tree)
    from tkfdp.coupling.dynfield.phylo_elbo.corpus_state import FamilyState

    n_leaves = 2 ** depth
    leaf_obs = rng.integers(0, A, size=(n_leaves, L)).astype(np.int32)
    tree = make_balanced_binary(depth=depth, tau=0.4, leaf_obs=leaf_obs)
    pt = build_padded_tree(tree)
    # cluster_id: first two columns form one cluster, rest singletons.
    cluster_id = np.arange(L, dtype=np.int32)
    if L >= 2:
        cluster_id[1] = cluster_id[0]
    return FamilyState(
        family_id=family_id, L=L,
        N_bucket=pt.N_bucket, D_bucket=pt.D_bucket,
        n_leaves_actual=pt.n_leaves_actual, depth_actual=pt.depth_actual,
        child_pos=pt.child_pos, child_branch=pt.child_branch,
        slot_mask=pt.slot_mask, root_slot=pt.root_slot,
        leaf_mask=pt.leaf_mask, leaf_obs_full=pt.leaf_obs,
        cluster_id=cluster_id,
        classes=rng.integers(0, 3, size=L).astype(np.int32))


def _mk_corpus(rng, K_c=3, K_a=4, L_field=3, A=4):
    from tkfdp.coupling.dynfield.phylo_elbo.corpus_state import (
        CorpusState, _sqrt2_buckets_up_to)
    from tkfdp.coupling.dynfield.phylo_elbo.tau_binning import build_tau_bins

    families = [_mk_family(f"F{i}", L=5, depth=2, A=A, rng=rng)
                for i in range(3)]
    # tau bins from all branch lengths.
    all_taus = []
    for f in families:
        for lvl in f.child_branch:
            for row in lvl:
                for v in row:
                    if float(v) > 0:
                        all_taus.append(float(v))
    bin_centers = build_tau_bins(np.asarray(all_taus), n_bins=8)
    bin_idx_by_family = []
    for f in families:
        per_level = []
        for lvl_branches in f.child_branch:
            idx = np.zeros_like(lvl_branches, dtype=np.int32)
            for i in range(lvl_branches.shape[0]):
                for j in range(2):
                    tau = float(lvl_branches[i, j])
                    idx[i, j] = int(np.argmin(np.abs(bin_centers - tau)))
            per_level.append(idx)
        bin_idx_by_family.append(per_level)
    pi_archetype = rng.dirichlet(np.ones(A), size=K_a)
    S = np.ones((A, A)) - np.eye(A)
    arch_assignment = rng.integers(0, K_a, size=(K_c, L_field)).astype(np.int32)
    rho = np.full(L_field, 1.0 / L_field)
    state = CorpusState(
        families=families, K_c=K_c, K_a=K_a, L_field=L_field,
        pi_archetype=pi_archetype, arch_assignment=arch_assignment,
        rho=rho, rho_chain=0.5, S=S,
        bin_centers=bin_centers, bin_idx_by_family=bin_idx_by_family,
        m_buckets=_sqrt2_buckets_up_to(8), max_cluster_size=8, alpha_z=100.0)
    state.refresh_pi_field()
    return state


def test_override_scorer_matches_full_pi():
    from tkfdp.coupling.dynfield.phylo_elbo.corpus_state import (
        compute_all_cluster_lls, _score_affected_clusters_under_pi,
        _score_affected_clusters_under_overrides)

    rng = np.random.default_rng(0)
    state = _mk_corpus(rng)
    compute_all_cluster_lls(state)

    c, theta, k = 1, 2, int(state.arch_assignment[1, 2])
    k = (k + 1) % state.K_a                         # a genuine change
    pi_row = state.pi_archetype[k]

    # Fast path: per-slice override (overrides map (c, theta) -> archetype
    # INDEX; the cached P_sub slice for k equals gtr_P_bins(pi_archetype[k])).
    fast = _score_affected_clusters_under_overrides(
        state, {c}, {(c, theta): k})

    # Reference: full pi_field rebuild.
    pi_full = np.array(state.pi_field, copy=True)
    pi_full[c, theta] = pi_row
    ref = _score_affected_clusters_under_pi(state, {c}, pi_full)

    assert set(fast.keys()) == set(ref.keys())
    for key in ref:
        assert abs(fast[key] - ref[key]) < 1e-8, (key, fast[key], ref[key])


def test_apply_arch_slices_matches_full_refresh():
    rng = np.random.default_rng(1)
    state = _mk_corpus(rng)

    c, theta = 0, 1
    k_new = (int(state.arch_assignment[c, theta]) + 2) % state.K_a
    state.arch_assignment[c, theta] = k_new
    state.apply_arch_slices([(c, theta)])
    P_inc = np.asarray(state.kernel_tables['P_sub'])
    pi_inc = np.array(state.pi_field, copy=True)

    # Full rebuild from the same (already-mutated) arch_assignment.
    state.refresh_pi_field()
    P_full = np.asarray(state.kernel_tables['P_sub'])
    pi_full = np.asarray(state.pi_field)

    assert np.allclose(P_inc, P_full, atol=1e-12), \
        float(np.max(np.abs(P_inc - P_full)))
    assert np.allclose(pi_inc, pi_full, atol=1e-14)


if __name__ == "__main__":
    test_override_scorer_matches_full_pi()
    test_apply_arch_slices_matches_full_refresh()
    print("corpus incremental-kernel equivalence PASS")
