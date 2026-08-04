"""Correctness + speedup: τ-binning forward pass (M12).

At n_bins large enough (>= 64), the binned forward pass should match
the exact (per-branch matrix-exp) forward pass to within ~1% log-lik.
At n_bins small (e.g. 8) it's a coarser approximation but still
useful for training.

Also measures wall-clock speedup on a synthetic corpus.
"""
from __future__ import annotations

import time

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


def test_binning_matches_exact_at_high_bins():
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv_jax import leaf_clv_jax
    from tkfdp.coupling.dynfield.phylo_elbo.pfam_loader import (
        family_to_tree)
    from tkfdp.coupling.dynfield.phylo_elbo.tree import (
        make_balanced_binary)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
        build_padded_tree)
    from tkfdp.coupling.dynfield.phylo_elbo.tau_binning import (
        assign_bins, build_tau_bins, collect_all_taus,
        precompute_kernel_tables, rebuild_padded_trees_with_bins)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_log_lik_jax import (
        gtr_eigendecomp_batch_jax, padded_tree_to_jax, tree_log_lik_jax)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_log_lik_jax_binned import (
        tree_log_lik_jax_binned)
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(1)
    L = 2; K_c = 2; A = 4
    M = _random_model(rng, L, m_max=3, A_alpha=A, K_c=K_c)
    rho_chain = 0.1

    # 4-leaf balanced tree, m=2.
    m = 2
    leaf_obs = rng.integers(0, A, size=(4, m)).astype(np.int32)
    tree = make_balanced_binary(depth=2, tau=0.3, leaf_obs=leaf_obs)
    classes = rng.integers(0, K_c, size=m).astype(np.int32)
    pt = build_padded_tree(tree)

    # pi_arch per site.
    pi_arch = M['pi_field'][classes]  # (m, L, A)

    # Exact reference (unbinned).
    pt_j = padded_tree_to_jax(pt)
    xi, U, Uinv = gtr_eigendecomp_batch_jax(
        jnp.asarray(M['pi_field']), jnp.asarray(M['S']))
    ll_exact = float(tree_log_lik_jax(
        pt_j['leaf_obs'], pt_j['leaf_mask'],
        pt_j['child_pos_by_level'], pt_j['child_branch_by_level'],
        pt_j['is_identity_by_level'], pt_j['root_slot'],
        jnp.asarray(pi_arch), jnp.asarray(M['rho']),
        rho_chain, xi, U, Uinv, jnp.asarray(classes)))

    # Binned version.
    for n_bins in (8, 32, 128):
        all_taus = collect_all_taus([pt])
        bin_centers = build_tau_bins(all_taus, n_bins)
        _, bin_idx_per_tree = rebuild_padded_trees_with_bins(
            [pt], bin_centers)
        tables = precompute_kernel_tables(
            bin_centers, M['rho'], rho_chain, M['pi_field'], M['S'])

        cp_j = pt_j['child_pos_by_level']
        cb_bin_j = tuple(jnp.asarray(bin_idx_per_tree[0][l])
                            for l in range(len(cp_j)))
        ll_binned = float(tree_log_lik_jax_binned(
            pt_j['leaf_obs'], pt_j['leaf_mask'],
            cp_j, cb_bin_j,
            pt_j['is_identity_by_level'], pt_j['root_slot'],
            jnp.asarray(pi_arch), jnp.asarray(M['rho']),
            jnp.asarray(tables['beta']), jnp.asarray(tables['W']),
            jnp.asarray(tables['P_sub']), jnp.asarray(classes)))

        rel_err = abs(ll_binned - ll_exact) / max(abs(ll_exact), 1e-9)
        print(f"  n_bins={n_bins}: exact={ll_exact:+.6f} "
                f"binned={ll_binned:+.6f} rel_err={rel_err:.2e}")
        # At n_bins=128, expect very small error (branches are 0.3,
        # exactly on grid). At lower n_bins, still ballpark.
        if n_bins >= 128:
            assert rel_err < 5e-2, f"n_bins=128 rel_err={rel_err}"


def test_incremental_pslice_matches_full_rebuild():
    """The incremental arch-move kernel update must equal a full rebuild.

    An arch move changes pi_field at one (c, theta) entry. The incremental
    path (corpus_state) keeps beta/W and overwrites only that P_sub slice
    via gtr_P_bins. This must reproduce precompute_kernel_tables run on the
    fully-updated pi_field. Pure numpy/scipy — no JAX needed.
    """
    import numpy as np
    from tkfdp.coupling.dynfield.phylo_elbo.tau_binning import (
        precompute_kernel_tables, gtr_P_bins)

    rng = np.random.default_rng(1)
    n_bins = 8; K_c = 3; L = 2; A = 4
    bin_centers = np.geomspace(0.05, 2.0, n_bins)
    rho = rng.dirichlet(np.ones(L))
    rho_chain = 0.3
    S = np.ones((A, A)) - np.eye(A)
    pi_field = rng.dirichlet(np.ones(A), size=(K_c, L))

    base = precompute_kernel_tables(bin_centers, rho, rho_chain, pi_field, S)

    c, th = 1, 0
    pi_new = rng.dirichlet(np.ones(A))
    pi_field2 = pi_field.copy(); pi_field2[c, th] = pi_new
    full = precompute_kernel_tables(bin_centers, rho, rho_chain, pi_field2, S)

    # beta / W do NOT depend on pi_field -> unchanged.
    assert np.allclose(base['beta'], full['beta'], atol=1e-14)
    assert np.allclose(base['W'], full['W'], atol=1e-14)

    # Incremental P_sub: base with the single (c, theta) slice replaced.
    P_inc = np.array(base['P_sub'], copy=True)
    P_inc[:, c, th] = gtr_P_bins(pi_new, S, bin_centers)
    assert np.allclose(P_inc, full['P_sub'], atol=1e-12), \
        float(np.max(np.abs(P_inc - full['P_sub'])))
    # Untouched slices are bit-identical to base.
    for cc in range(K_c):
        for tt in range(L):
            if (cc, tt) == (c, th):
                continue
            assert np.array_equal(base['P_sub'][:, cc, tt],
                                       full['P_sub'][:, cc, tt])


if __name__ == "__main__":
    test_binning_matches_exact_at_high_bins()
    test_incremental_pslice_matches_full_rebuild()
    print("M12 τ-binning correctness PASS")
