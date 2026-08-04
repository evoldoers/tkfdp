"""Agreement test: tree_log_lik_jax vs the recursive numpy tree_log_lik_mm.

Verifies:
  - Cherry (depth 1)
  - Balanced depth-2 (4 leaves)
  - Balanced depth-3 (8 leaves)
  - Unbalanced ((A,B), C) with a phantom identity lift
Cherry: exact under MM (both paths). Depth 2+: agreement to double
precision (both paths use the SAME projection formulas, so numerical
identity is expected).
"""
from __future__ import annotations

import numpy as np
import pytest


def _random_model(rng, L, m, A_alpha, K_c):
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        per_class_field_Q)
    pi_field = rng.dirichlet(np.ones(A_alpha), size=(K_c, L))
    S_raw = rng.uniform(0.5, 2.0, size=(A_alpha, A_alpha))
    S = (S_raw + S_raw.T) / 2
    np.fill_diagonal(S, 0.0)
    rho = rng.dirichlet(np.full(L, 2.0))
    if K_c == 1:
        classes = np.zeros(m, dtype=np.int32)
    else:
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
    Q = per_class_field_Q(pi_field, S)
    pi_arch = pi_field[classes]     # (m, L, A)
    return {'pi_field': pi_field, 'S': S, 'rho': rho, 'classes': classes,
              'Q': Q, 'pi_arch': pi_arch}


def _run_pair(tree, rho_chain, seed):
    import jax
    jax.config.update("jax_enable_x64", True)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        per_class_field_Q)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_log_lik import (
        tree_log_lik_mm)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
        build_padded_tree)
    from tkfdp.coupling.dynfield.phylo_elbo.tree_log_lik_jax import (
        gtr_eigendecomp_batch_jax, padded_tree_to_jax, tree_log_lik_jax)
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    m = int(tree.m)
    L = 3
    A_alpha = 4
    K_c = 2
    M = _random_model(rng, L, m, A_alpha, K_c)

    # Numpy reference.
    ll_np = tree_log_lik_mm(
        tree, M['classes'], M['rho'], M['pi_field'], M['S'], rho_chain)

    # JAX version.
    pt = build_padded_tree(tree)
    pt_j = padded_tree_to_jax(pt)
    xi, U, Uinv = gtr_eigendecomp_batch_jax(
        jnp.asarray(M['pi_field']), jnp.asarray(M['S']))
    ll_jax = float(tree_log_lik_jax(
        pt_j['leaf_obs'], pt_j['leaf_mask'],
        pt_j['child_pos_by_level'], pt_j['child_branch_by_level'],
        pt_j['is_identity_by_level'], pt_j['root_slot'],
        jnp.asarray(M['pi_arch']), jnp.asarray(M['rho']),
        rho_chain, xi, U, Uinv, jnp.asarray(M['classes'])))

    print(f"  ll_np={ll_np:+.10f} ll_jax={ll_jax:+.10f} diff={ll_jax - ll_np:+.2e}")
    return ll_np, ll_jax


@pytest.mark.parametrize("rho_chain", [0.05, 0.5, 1.5])
def test_cherry_jax_matches_numpy(rho_chain):
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_cherry
    m = 3
    leaf_obs = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int32) % 4
    tree = make_cherry(tau=0.5, leaf_obs=leaf_obs)
    ll_np, ll_jax = _run_pair(tree, rho_chain, seed=0)
    assert np.isclose(ll_np, ll_jax, atol=1e-10, rtol=1e-10), \
        f"cherry ll mismatch at rho_chain={rho_chain}"


@pytest.mark.parametrize("rho_chain", [0.1, 0.5])
def test_balanced_depth2_jax_matches_numpy(rho_chain):
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_balanced_binary
    m = 2
    leaf_obs = np.array([[0, 1], [2, 3], [1, 0], [3, 2]], dtype=np.int32) % 4
    tree = make_balanced_binary(depth=2, tau=0.3, leaf_obs=leaf_obs)
    ll_np, ll_jax = _run_pair(tree, rho_chain, seed=1)
    assert np.isclose(ll_np, ll_jax, atol=1e-10, rtol=1e-10), \
        f"depth-2 ll mismatch at rho_chain={rho_chain}"


@pytest.mark.parametrize("rho_chain", [0.1, 0.5])
def test_balanced_depth3_jax_matches_numpy(rho_chain):
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_balanced_binary
    m = 2
    leaf_obs = np.arange(16).reshape(8, 2).astype(np.int32) % 4
    tree = make_balanced_binary(depth=3, tau=0.2, leaf_obs=leaf_obs)
    ll_np, ll_jax = _run_pair(tree, rho_chain, seed=2)
    assert np.isclose(ll_np, ll_jax, atol=1e-9, rtol=1e-9), \
        f"depth-3 ll mismatch at rho_chain={rho_chain}"


@pytest.mark.parametrize("rho_chain", [0.1, 0.5])
def test_unbalanced_ab_c_jax_matches_numpy(rho_chain):
    from tkfdp.coupling.dynfield.phylo_elbo.tree import build_tree
    # ((A, B), C) — one internal node ((A,B)), then root ((A,B), C).
    parent = [3, 3, 4, 4, -1]
    branch = [0.5, 0.5, 1.0, 0.3, 0.0]
    leaf_obs = np.array([[0, 1], [2, 3], [1, 2]], dtype=np.int32) % 4
    tree = build_tree(parent, branch, leaf_obs)
    ll_np, ll_jax = _run_pair(tree, rho_chain, seed=3)
    assert np.isclose(ll_np, ll_jax, atol=1e-10, rtol=1e-10), \
        f"unbalanced ll mismatch at rho_chain={rho_chain}"


if __name__ == "__main__":
    print("== cherry ==")
    for rc in [0.05, 0.5, 1.5]:
        test_cherry_jax_matches_numpy(rc)
    print("== balanced depth-2 ==")
    for rc in [0.1, 0.5]:
        test_balanced_depth2_jax_matches_numpy(rc)
    print("== balanced depth-3 ==")
    for rc in [0.1, 0.5]:
        test_balanced_depth3_jax_matches_numpy(rc)
    print("== unbalanced ((A,B),C) ==")
    for rc in [0.1, 0.5]:
        test_unbalanced_ab_c_jax_matches_numpy(rc)
    print("all M3 tests PASS")
