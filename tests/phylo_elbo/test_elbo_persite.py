"""Per-site O(A^2) tree-structured ELBO (field tree-Markov + residues) is a valid
lower bound, tighter than the node-mean-field floor.

This is the factored (field tree-Markov) family, so -- unlike the joint
elbo_treestruct -- it is not exact even at a cherry; it trades that for
O(N(L^2 + m L A^2)) with no A^m. Here we pin the two guarantees that must hold:
L(q) <= exact log P, and L(q) > the node-mean-field floor (elbo.elbo).
"""
import itertools
import numpy as np
import pytest

from tkfdp.coupling.dynfield.phylo_elbo.elbo_persite import elbo_persite
from tkfdp.coupling.dynfield.phylo_elbo.elbo import elbo as floor
from tkfdp.coupling.dynfield.phylo_elbo.elbo_rescorr import branch_factor
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


@pytest.mark.parametrize("rc", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("name,m,mk", [
    ("cherry", 1, lambda: make_cherry(0.5, np.array([[0], [2]], np.int32))),
    ("cherry", 2, lambda: make_cherry(0.5, np.array([[0, 1], [2, 0]], np.int32))),
    ("depth2", 1, lambda: make_balanced_binary(
        depth=2, tau=0.4, leaf_obs=np.array([[0], [1], [2], [0]], np.int32))),
])
def test_persite_is_valid_bound_tighter_than_floor(rc, name, m, mk):
    rng = np.random.default_rng(m * 7 + int(rc * 10))
    pi, S, rho = _model(rng, Kc=max(1, m))
    cls = (np.arange(m) % pi.shape[0]).astype(np.int32)
    tree = mk()
    exact = _exact_logP(tree, cls, rho, pi, S, rc)
    ps = elbo_persite(tree, cls, rho, pi, S, rc)
    fl = floor(tree, cls, rho, pi, S, rc)
    assert ps <= exact + 1e-6, f"per-site ELBO {ps} > exact {exact}"
    assert ps > fl - 1e-9, f"per-site ELBO {ps} not >= floor {fl}"
