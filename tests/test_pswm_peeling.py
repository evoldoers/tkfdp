"""Smoke tests for per-site LG08 Felsenstein peeling.

1. Constructs a small cherry tree (2 leaves + root) and verifies:
   - PSWM at leaves matches observed deltas (or fallback for gaps)
   - PSWM at root matches direct sum-over-x_root marginalisation
2. Deeper (depth-2) balanced binary tree.
"""
import numpy as np
import pytest
from scipy.linalg import expm

from tkfdp.bio import Node
from tkfdp.lg08 import Q_LG08, PI_LG08
from tkfdp.pswm_peeling import compute_pswm_family


def _cherry(tau_a: float, tau_b: float) -> Node:
    """Two-leaf cherry with named leaves 'a' and 'b'."""
    a = Node(); a.name = 'a'; a.branch_length = tau_a; a.children = []
    b = Node(); b.name = 'b'; b.branch_length = tau_b; b.children = []
    root = Node(); root.name = ''; root.branch_length = None
    root.children = [a, b]
    return root


def _balanced_depth2(tau: float) -> Node:
    """4-leaf balanced binary tree: root -> {L, R}; L -> {a, b}; R -> {c, d}."""
    a = Node(); a.name = 'a'; a.branch_length = tau; a.children = []
    b = Node(); b.name = 'b'; b.branch_length = tau; b.children = []
    c = Node(); c.name = 'c'; c.branch_length = tau; c.children = []
    d = Node(); d.name = 'd'; d.branch_length = tau; d.children = []
    L = Node(); L.name = 'L'; L.branch_length = tau; L.children = [a, b]
    R = Node(); R.name = 'R'; R.branch_length = tau; R.children = [c, d]
    root = Node(); root.name = ''; root.branch_length = None
    root.children = [L, R]
    return root


def test_cherry_pswm_matches_direct_marginalisation():
    """Verify per-site PSWM matches direct enumeration on a cherry."""
    tau = 0.5
    root = _cherry(tau, tau)
    # 1-site MSA with 2 leaves observing residues x_a=3 (D) and x_b=5 (E).
    msa = np.array([[3], [5]], dtype=np.int8)
    leaf_names = ['a', 'b']
    out = compute_pswm_family(msa, leaf_names, root)
    n_leaves = 2
    n_nodes = 3
    assert out['n_nodes'] == n_nodes
    assert out['n_leaves'] == n_leaves

    # Leaf PSWMs: delta at observed residues.
    pswm_leaf_a = out['pswm'][0, 0, :]
    pswm_leaf_b = out['pswm'][1, 0, :]
    assert pswm_leaf_a[3] > 0.99, f"leaf a not delta at D: {pswm_leaf_a}"
    assert pswm_leaf_b[5] > 0.99, f"leaf b not delta at E: {pswm_leaf_b}"

    # Root PSWM: direct marginal.
    # P(x_root = a | x_a=3, x_b=5) proportional to
    #   pi(a) * P(a, 3; tau) * P(a, 5; tau)
    P = expm(Q_LG08 * tau)
    unnorm = PI_LG08 * P[:, 3] * P[:, 5]
    direct = unnorm / unnorm.sum()
    root_id = int(out['root_id'])
    pswm_root = out['pswm'][root_id, 0, :]
    diff = float(np.abs(pswm_root - direct).max())
    print(f"cherry root PSWM max diff vs direct: {diff:.2e}")
    assert diff < 1e-10, f"cherry root PSWM mismatch: {diff}"


def test_depth2_pswm_matches_direct_marginalisation():
    """Verify per-site PSWM at every node matches brute-force
    enumeration on a depth-2 balanced binary tree with 4 leaves."""
    tau = 0.4
    root = _balanced_depth2(tau)
    # 1-site observations at leaves a, b, c, d.
    x_obs = {'a': 0, 'b': 5, 'c': 10, 'd': 12}
    msa = np.array([[x_obs['a']], [x_obs['b']], [x_obs['c']], [x_obs['d']]],
                       dtype=np.int8)
    leaf_names = ['a', 'b', 'c', 'd']
    out = compute_pswm_family(msa, leaf_names, root)

    # Direct enumeration over x_L, x_R, x_root of joint prob.
    P = expm(Q_LG08 * tau)
    joint = 0.0
    from itertools import product
    Amarg = 20
    # For each node we want P(x_v | leaves) proportional to
    #   sum over other hidden of pi(x_root) * P(x_root, x_L) * P(x_root, x_R)
    #     * P(x_L, x_a) * P(x_L, x_b) * P(x_R, x_c) * P(x_R, x_d)
    # subject to fixed leaf observations x_a=0, x_b=5, x_c=10, x_d=12.

    def direct_marginal(target_node: str) -> np.ndarray:
        result = np.zeros(Amarg)
        for x_root in range(Amarg):
            for x_L in range(Amarg):
                for x_R in range(Amarg):
                    p = (PI_LG08[x_root]
                         * P[x_root, x_L] * P[x_root, x_R]
                         * P[x_L, x_obs['a']] * P[x_L, x_obs['b']]
                         * P[x_R, x_obs['c']] * P[x_R, x_obs['d']])
                    if target_node == 'root':
                        result[x_root] += p
                    elif target_node == 'L':
                        result[x_L] += p
                    elif target_node == 'R':
                        result[x_R] += p
        return result / result.sum()

    # Find node ids: leaves 0..3 (a, b, c, d); internals L, R; root.
    # In our post-order assignment, leaves first (a, b, c, d in order?).
    # Check by name.
    root_id = int(out['root_id'])
    # Find L and R by their children (each has 2 leaf children with those names).
    # For simplicity we just find which internal node has children a, b vs c, d.
    parent = out['parent']
    # Leaves indices for a, b, c, d.
    leaf_idx = {}
    # Nodes were leaves-first per _assign_node_ids.
    # We know msa rows are 0=a, 1=b, 2=c, 3=d, and leaf_msa_row matches that
    # order via the traversal.
    leaf_msa_row = out['leaf_msa_row']
    for i, row in enumerate(leaf_msa_row):
        name = leaf_names[row]
        leaf_idx[name] = i
    # Parent of leaf a and b is node L; parent of leaf c and d is node R.
    L_id = int(parent[leaf_idx['a']])
    R_id = int(parent[leaf_idx['c']])
    assert L_id != R_id

    direct_root = direct_marginal('root')
    direct_L = direct_marginal('L')
    direct_R = direct_marginal('R')

    diff_root = float(np.abs(out['pswm'][root_id, 0] - direct_root).max())
    diff_L = float(np.abs(out['pswm'][L_id, 0] - direct_L).max())
    diff_R = float(np.abs(out['pswm'][R_id, 0] - direct_R).max())
    print(f"depth-2 root diff: {diff_root:.2e}")
    print(f"depth-2 L    diff: {diff_L:.2e}")
    print(f"depth-2 R    diff: {diff_R:.2e}")
    assert diff_root < 1e-10
    assert diff_L < 1e-10
    assert diff_R < 1e-10


def test_deep_tree_leaves_remain_delta():
    """Regression: bottom-up CLV underflows to 0 on deep trees without
    per-node rescaling. Leaves must remain delta on their observed
    residue regardless of tree depth."""
    # Build a chain-like binary tree: 128 leaves via 7 levels of pairing.
    def _build(n_levels: int, tau: float = 0.3) -> Node:
        leaves = []
        for i in range(2 ** n_levels):
            L_i = Node(); L_i.name = f'l{i}'
            L_i.branch_length = tau; L_i.children = []
            leaves.append(L_i)
        current = leaves
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                p = Node(); p.branch_length = tau
                p.children = [current[i], current[i + 1]]
                next_level.append(p)
            current = next_level
        root = current[0]
        root.branch_length = None
        return root, leaves

    n_levels = 7                                    # 128 leaves
    root, leaves = _build(n_levels)
    n_leaves = len(leaves)
    rng = np.random.default_rng(0)
    L = 3
    msa = rng.integers(0, 20, size=(n_leaves, L)).astype(np.int8)
    leaf_names = [f'l{i}' for i in range(n_leaves)]
    out = compute_pswm_family(msa, leaf_names, root)

    # Verify every leaf PSWM is delta on the observed residue.
    for i in range(n_leaves):
        row = int(out['leaf_msa_row'][i])
        for site in range(L):
            x_obs = int(msa[row, site])
            argmax_p = int(out['pswm'][i, site].argmax())
            max_p = float(out['pswm'][i, site].max())
            assert argmax_p == x_obs, (
                f"leaf {i}, site {site}: expected argmax={x_obs}, "
                f"got {argmax_p}")
            assert max_p > 0.99, (
                f"leaf {i}, site {site}: expected max~1, got {max_p}")


def test_sampler_root_marginal_matches_analytic():
    """The joint LG08 sampler must draw the root residue from
    pi * clv[root] / Z (analytic tree posterior marginal at the root)."""
    tau_v = 0.4
    root = _balanced_depth2(tau_v)
    x_obs = {'a': 0, 'b': 5, 'c': 10, 'd': 12}
    msa = np.array([[x_obs['a']], [x_obs['b']], [x_obs['c']], [x_obs['d']]],
                       dtype=np.int8)
    leaf_names = ['a', 'b', 'c', 'd']
    out = compute_pswm_family(msa, leaf_names, root)
    clv = out['clv']
    parent = out['parent']
    tau = out['tau']
    n_leaves = out['n_leaves']
    root_id = out['root_id']
    leaf_msa = msa

    from tkfdp.pswm_peeling import sample_joint_tree_history
    K = 10_000
    counts = np.zeros(20, dtype=np.int64)
    for k in range(K):
        rng = np.random.default_rng(k)
        X = sample_joint_tree_history(clv, parent, tau, leaf_msa, rng)
        counts[X[root_id, 0]] += 1
    emp = counts / K
    analytic = PI_LG08 * clv[root_id, 0]
    analytic = analytic / analytic.sum()
    diff = np.abs(emp - analytic).max()
    print(f"root marginal max diff (K={K}): {diff:.4f}")
    assert diff < 0.02, f"root marginal off by {diff}"


def test_sampler_leaves_respected():
    """Sampler must return the observed residue at leaf positions."""
    root = _balanced_depth2(0.4)
    x_obs = {'a': 0, 'b': 5, 'c': 10, 'd': 12}
    msa = np.array([[x_obs['a']], [x_obs['b']], [x_obs['c']], [x_obs['d']]],
                       dtype=np.int8)
    out = compute_pswm_family(msa, ['a', 'b', 'c', 'd'], root)
    from tkfdp.pswm_peeling import sample_joint_tree_history
    for k in range(50):
        rng = np.random.default_rng(k)
        X = sample_joint_tree_history(out['clv'], out['parent'],
                                            out['tau'], msa, rng)
        # leaves are indices 0..3 (n_leaves=4)
        leaf_msa_row = out['leaf_msa_row']
        for i in range(4):
            r = int(leaf_msa_row[i])
            assert int(X[i, 0]) == int(msa[r, 0]), (
                f"leaf {i} sample={X[i, 0]} != observed={msa[r, 0]}")


if __name__ == "__main__":
    test_cherry_pswm_matches_direct_marginalisation()
    print("cherry test PASS")
    test_depth2_pswm_matches_direct_marginalisation()
    print("depth-2 test PASS")
    test_deep_tree_leaves_remain_delta()
    print("deep-tree leaves-delta test PASS")
    test_sampler_root_marginal_matches_analytic()
    print("sampler root-marginal test PASS")
    test_sampler_leaves_respected()
    print("sampler leaves-respected test PASS")
