"""Tractable tree-structured ELBO matches the enumerated reference + exactness.

elbo_treestruct.elbo_treestruct computes L(q) for the residue-correlated q via a
rooted downward sweep with per-node A^m marginals (no whole-tree enumeration). It
must (a) equal elbo_rescorr_bruteforce (same q, tractable vs enumerated) to
machine precision, and (b) hit the full exactness set: cherries at any m, all
trees at m=1, and field-rate 0. (This module uses O(A^{2m}) node/edge tables --
correct and cheap for the cap-2/cap-3 clusters in use; the per-site O(N m A^2)
form is elbo_treestruct_persite.)
"""
import itertools
import numpy as np
import pytest

from tkfdp.coupling.dynfield.phylo_elbo.elbo_treestruct import elbo_treestruct
from tkfdp.coupling.dynfield.phylo_elbo.elbo_rescorr import (
    elbo_rescorr_bruteforce, branch_factor)
from tkfdp.coupling.dynfield.phylo_elbo.tree import make_cherry, make_balanced_binary


def _exact_logP(tree, cls, rho, pi, S, rc):
    L = rho.shape[0]; A = pi.shape[2]; m = len(cls); nn = tree.n_nodes; root = tree.root
    isl = [tree.is_leaf(v) for v in range(nn)]
    internal = [v for v in range(nn) if not isl[v]]
    xsp = list(itertools.product(range(A), repeat=m)); tot = 0.0
    for th in itertools.product(range(L), repeat=nn):
        for xi in itertools.product(xsp, repeat=len(internal)):
            xa = {}
            for i, v in enumerate(internal):
                xa[v] = xi[i]
            for v in range(nn):
                if isl[v]:
                    xa[v] = tuple(int(tree.leaf_obs[v, n]) for n in range(m))
            p = rho[th[root]]
            for n in range(m):
                p *= pi[int(cls[n]), th[root], xa[root][n]]
            for v in range(nn):
                if v == root:
                    continue
                u = int(tree.parent[v])
                p *= branch_factor(xa[u], th[u], xa[v], th[v], cls,
                                   float(tree.branch_length[v]), pi, S, rho, rc)
            tot += p
    return float(np.log(tot))


def _model(rng, Kc=1):
    pi = rng.dirichlet(np.ones(3), size=(Kc, 2))
    Sr = rng.uniform(0.5, 2.0, (3, 3)); S = (Sr + Sr.T) / 2; np.fill_diagonal(S, 0.0)
    return pi, S, rng.dirichlet([2.0, 2.0])


def _cherry(m):
    return make_cherry(0.5, np.array([[(i + n) % 3 for n in range(m)]
                                      for i in range(2)], np.int32))


@pytest.mark.parametrize("rc", [0.15, 0.5, 2.0])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_matches_reference_and_exact_on_cherry(rc, m):
    rng = np.random.default_rng(m * 5 + int(rc * 10))
    pi, S, rho = _model(rng, Kc=max(1, m))
    cls = (np.arange(m) % pi.shape[0]).astype(np.int32)
    tree = _cherry(m)
    exact = _exact_logP(tree, cls, rho, pi, S, rc)
    ts = elbo_treestruct(tree, cls, rho, pi, S, rc)
    ref = elbo_rescorr_bruteforce(tree, cls, rho, pi, S, rc)
    assert abs(ts - ref) < 1e-9, f"tractable != enumerated ref: {ts} vs {ref}"
    assert abs(ts - exact) < 1e-7, f"cherry m={m} not exact: {ts} vs {exact}"


@pytest.mark.parametrize("rc", [1e-8, 0.15, 0.5, 2.0])
def test_singleton_exact_at_depth(rc):
    rng = np.random.default_rng(int(rc * 100) + 3)
    pi, S, rho = _model(rng)
    cls = np.zeros(1, np.int32)
    tree = make_balanced_binary(depth=2, tau=0.4,
                                leaf_obs=np.array([[0], [1], [2], [0]], np.int32))
    exact = _exact_logP(tree, cls, rho, pi, S, rc)
    ts = elbo_treestruct(tree, cls, rho, pi, S, rc)
    assert abs(ts - exact) < 1e-6, f"m=1 depth2 rc={rc} not exact: {ts} vs {exact}"


def test_depth2_m2_is_certified_bound():
    rng = np.random.default_rng(11)
    pi, S, rho = _model(rng, Kc=2)
    cls = np.array([0, 1], np.int32)
    tree = make_balanced_binary(depth=2, tau=0.4, leaf_obs=np.array(
        [[0, 1], [1, 2], [2, 0], [0, 1]], np.int32))
    for rc in (0.15, 1.0):
        exact = _exact_logP(tree, cls, rho, pi, S, rc)
        ts = elbo_treestruct(tree, cls, rho, pi, S, rc)
        ref = elbo_rescorr_bruteforce(tree, cls, rho, pi, S, rc)
        assert ts <= exact + 1e-7, "tree-structured ELBO violates the bound"
        assert abs(ts - ref) < 1e-9, "tractable != enumerated ref"
