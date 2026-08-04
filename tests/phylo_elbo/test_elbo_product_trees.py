"""Product-of-trees structured mean-field ELBO: valid bound, exactness regimes,
monotone coordinate ascent.

elbo_product_trees uses q = q_Theta(theta,Delta) * prod_n q_n(x_n) with the
residue factors independent of the field (full along-tree residue chains). This
buys O(m N A^2) cost (linear in m) at the price of the field<->residue seam, so
-- unlike the seam-keeping elbo_persite/elbo_treestruct -- it is NOT exact for
m=1 or at rho_chain=0 with field-dependent emissions. Here we pin what MUST
hold: (a) it is a genuine lower bound (ELBO <= exact) on small trees; (b) it is
exact in the provable regimes (L=1 any rho_chain; field-independent emissions in
the rho_chain->0 limit); (c) coordinate ascent is monotone and warm-start <=
converged. The exact reference is exact_peel.exact_ll_tree_general, itself
cross-checked against exact_cap2.exact_pair_ll_tree (the validated m=2 peel).
"""
import numpy as np
import pytest

from tkfdp.coupling.dynfield.phylo_elbo.elbo_product_trees import elbo_product_trees
from tkfdp.coupling.dynfield.phylo_elbo.exact_peel import exact_ll_tree_general
from tkfdp.coupling.dynfield.phylo_elbo.exact_cap2 import exact_pair_ll_tree
from tkfdp.coupling.dynfield.phylo_elbo.tree import (
    make_cherry, make_balanced_binary)


def _model(rng, L=2, A=3, Kc=1):
    pi = rng.dirichlet(np.ones(A), size=(Kc, L))
    Sr = rng.uniform(0.5, 2.0, (A, A)); S = (Sr + Sr.T) / 2; np.fill_diagonal(S, 0.0)
    return pi, S, rng.dirichlet(np.full(L, 2.0))


# --------------------------------------------------------------------------- (0)
def test_exact_peel_matches_validated_pair_peel():
    """exact_ll_tree_general (m-general) == exact_pair_ll_tree (validated m=2)."""
    rng = np.random.default_rng(1)
    A = 3; L = 2
    pi, S, rho = _model(rng, L, A, Kc=2)
    obs = np.array([[0, 1], [1, 2], [2, 0], [0, 1]], np.int32)
    tree = make_balanced_binary(2, 0.4, obs)
    cls = np.array([0, 1], np.int32)
    pi_arch = pi.reshape(2 * L, A)
    a1 = np.array([t for t in range(L)]); a2 = np.array([L + t for t in range(L)])
    leaf_pair = np.full((tree.n_nodes, 2), -1, np.int32)
    for v in range(tree.n_leaves):
        leaf_pair[v] = obs[v]
    for rc in (0.15, 0.7, 2.0):
        g = exact_ll_tree_general(tree, cls, rho, pi, S, rc)
        gp = exact_pair_ll_tree(tree.parent, tree.branch_length, leaf_pair,
                                tree.root, a1, a2, pi_arch, S, rho, rc)
        assert abs(g - gp) < 1e-9, f"peel mismatch rc={rc}: {g} vs {gp}"


# --------------------------------------------------------------------------- (a)
@pytest.mark.parametrize("rc", [0.15, 0.7, 2.0])
@pytest.mark.parametrize("name,m,mk", [
    ("cherry", 1, lambda: make_cherry(0.5, np.array([[0], [2]], np.int32))),
    ("cherry", 2, lambda: make_cherry(0.5, np.array([[0, 1], [2, 0]], np.int32))),
    ("cherry", 3, lambda: make_cherry(0.5, np.array([[0, 1, 2], [2, 0, 1]], np.int32))),
    ("depth2", 1, lambda: make_balanced_binary(
        2, 0.4, np.array([[0], [1], [2], [0]], np.int32))),
    ("depth2", 2, lambda: make_balanced_binary(
        2, 0.4, np.array([[0, 1], [1, 2], [2, 0], [0, 1]], np.int32))),
])
def test_valid_lower_bound_and_above_floor(rc, name, m, mk):
    rng = np.random.default_rng(m * 11 + int(rc * 10))
    pi, S, rho = _model(rng, Kc=max(1, m))
    cls = (np.arange(m) % pi.shape[0]).astype(np.int32)
    tree = mk()
    exact = exact_ll_tree_general(tree, cls, rho, pi, S, rc)
    d = elbo_product_trees(tree, cls, rho, pi, S, rc, return_terms=True)
    assert d["elbo"] <= exact + 1e-7, (
        f"product-of-trees ELBO {d['elbo']} > exact {exact} ({name} m={m} rc={rc})")
    # NOTE: no ordering vs the node-mean-field floor (elbo.py) is asserted --
    # the two families are incomparable (PoT keeps along-tree residue
    # correlation but drops the per-node field<->residue seam that the floor
    # keeps), so neither dominates. See the eval writeup.


# --------------------------------------------------------------------------- (b)
@pytest.mark.parametrize("rc", [0.15, 1.0, 3.0])
def test_exact_when_L1_any_rho_chain(rc):
    """L=1 (trivial field): residues are m independent tree chains -> exact."""
    rng = np.random.default_rng(int(rc * 10) + 5)
    A = 3
    pi = rng.dirichlet(np.ones(A), size=(2, 1))            # (Kc=2, L=1, A)
    Sr = rng.uniform(0.5, 2.0, (A, A)); S = (Sr + Sr.T) / 2; np.fill_diagonal(S, 0.0)
    rho = np.array([1.0])
    cls = np.array([0, 1], np.int32)
    tree = make_balanced_binary(2, 0.4, np.array(
        [[0, 1], [1, 2], [2, 0], [0, 1]], np.int32))
    exact = exact_ll_tree_general(tree, cls, rho, pi, S, rc)
    el = elbo_product_trees(tree, cls, rho, pi, S, rc)
    assert abs(exact - el) < 1e-7, f"L=1 not exact rc={rc}: {el} vs {exact}"


def test_exact_field_independent_in_no_jump_limit():
    """Field-independent emissions: gap -> 0 as rho_chain -> 0 (scales linearly)."""
    rng = np.random.default_rng(21)
    A = 3; L = 2
    Sr = rng.uniform(0.5, 2.0, (A, A)); S = (Sr + Sr.T) / 2; np.fill_diagonal(S, 0.0)
    rho = rng.dirichlet([2.0, 2.0])
    base = rng.dirichlet(np.ones(A), size=(2, 1))
    pi_ind = np.repeat(base, L, axis=1)                    # identical across theta
    cls = np.array([0, 1], np.int32)
    tree = make_balanced_binary(2, 0.4, np.array(
        [[0, 1], [1, 2], [2, 0], [0, 1]], np.int32))
    gaps = []
    for rc in (1e-3, 1e-4, 1e-5):
        exact = exact_ll_tree_general(tree, cls, rho, pi_ind, S, rc)
        el = elbo_product_trees(tree, cls, rho, pi_ind, S, rc)
        gaps.append(exact - el)
        assert el <= exact + 1e-9
    # vanishing and roughly linear in rho_chain
    assert gaps[-1] < 1e-4, f"no-jump limit not exact: {gaps}"
    assert gaps[0] > gaps[-1], f"gap should shrink with rho_chain: {gaps}"


def test_field_dependent_rho_chain0_is_strict_bound():
    """Honest counter-regime: field-DEPENDENT emissions at rho_chain->0 stay
    strictly loose (the frozen-field mixture the family cannot represent)."""
    rng = np.random.default_rng(22)
    pi, S, rho = _model(rng, Kc=2)
    cls = np.array([0, 1], np.int32)
    tree = make_balanced_binary(2, 0.4, np.array(
        [[0, 1], [1, 2], [2, 0], [0, 1]], np.int32))
    exact = exact_ll_tree_general(tree, cls, rho, pi, S, 1e-5)
    el = elbo_product_trees(tree, cls, rho, pi, S, 1e-5)
    assert el <= exact + 1e-9
    assert exact - el > 1e-3, "expected a strictly loose bound here"


# --------------------------------------------------------------------------- (c)
@pytest.mark.parametrize("rc", [0.15, 1.0])
def test_monotone_coordinate_ascent_and_warm_start(rc):
    rng = np.random.default_rng(int(rc * 10) + 30)
    pi, S, rho = _model(rng, Kc=2)
    cls = np.array([0, 1], np.int32)
    tree = make_balanced_binary(3, 0.4, np.array(
        [[i % 3, (i + 1) % 3] for i in range(8)], np.int32))
    d = elbo_product_trees(tree, cls, rho, pi, S, rc, return_terms=True)
    tr = d["elbo_trace"]
    for i in range(len(tr) - 1):
        assert tr[i + 1] >= tr[i] - 1e-9, f"non-monotone at {i}: {tr}"
    assert d["warm_elbo"] <= d["elbo"] + 1e-9
    exact = exact_ll_tree_general(tree, cls, rho, pi, S, rc)
    assert d["elbo"] <= exact + 1e-7
