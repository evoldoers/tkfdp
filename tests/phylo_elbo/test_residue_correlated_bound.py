"""The residue-correlated ELBO is a certified bound, tighter than the floor.

Validates elbo_rescorr.elbo_rescorr_bruteforce (the correct-by-construction
reference: builds the genuine residue-correlated q -- rooted BP posterior with
the root combine kept unprojected -- and evaluates L(q)=E_q[log P]+H(q) by
enumeration) against a brute-force exact log P on small trees:

  * EQUALITY (not just <=) for cherries at any m and singletons (m=1) at any
    depth -- where q is the exact posterior;
  * a certified bound (L <= log P, Zq = 1) for m>=2 at depth>=2;
  * strictly tighter than the residue-mean-field floor (elbo.elbo) everywhere.
"""
import itertools
import numpy as np
import pytest

from tkfdp.coupling.dynfield.phylo_elbo.elbo_rescorr import (
    elbo_rescorr_bruteforce, branch_factor)
from tkfdp.coupling.dynfield.phylo_elbo.elbo import elbo as elbo_meanfield
from tkfdp.coupling.dynfield.phylo_elbo.tree import make_cherry, make_balanced_binary


def _exact_logP(tree, classes, rho, pi, S, rc):
    L = rho.shape[0]; A = pi.shape[2]; m = len(classes)
    nn = tree.n_nodes; root = tree.root
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
                p *= pi[int(classes[n]), th[root], xa[root][n]]
            for v in range(nn):
                if v == root:
                    continue
                u = int(tree.parent[v])
                p *= branch_factor(xa[u], th[u], xa[v], th[v], classes,
                                   float(tree.branch_length[v]), pi, S, rho, rc)
            tot += p
    return float(np.log(tot))


def _model(rng, L=2, A=3, Kc=1):
    pi = rng.dirichlet(np.ones(A), size=(Kc, L))
    Sr = rng.uniform(0.5, 2.0, (A, A)); S = (Sr + Sr.T) / 2; np.fill_diagonal(S, 0.0)
    return pi, S, rng.dirichlet(np.full(L, 2.0))


def _cherry(m):
    obs = np.array([[(i + n) % 3 for n in range(m)] for i in range(2)], np.int32)
    return make_cherry(0.5, obs)


def _depth2(m):
    obs = np.array([[(i + n) % 3 for n in range(m)] for i in range(4)], np.int32)
    return make_balanced_binary(depth=2, tau=0.4, leaf_obs=obs)


@pytest.mark.parametrize("rc", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_cherry_bound_is_exact(rc, m):
    rng = np.random.default_rng(m * 5 + int(rc * 10))
    pi, S, rho = _model(rng, Kc=max(1, m))
    cls = (np.arange(m) % pi.shape[0]).astype(np.int32)
    tree = _cherry(m)
    exact = _exact_logP(tree, cls, rho, pi, S, rc)
    d = elbo_rescorr_bruteforce(tree, cls, rho, pi, S, rc, return_terms=True)
    assert abs(d["Zq"] - 1.0) < 1e-8
    assert abs(d["elbo"] - exact) < 1e-7, f"cherry m={m} rc={rc}: {d['elbo']} vs {exact}"


@pytest.mark.parametrize("rc", [0.5, 1.0, 2.0])
def test_singleton_bound_is_exact_at_depth(rc):
    rng = np.random.default_rng(int(rc * 10) + 3)
    pi, S, rho = _model(rng)
    cls = np.zeros(1, np.int32)
    tree = _depth2(1)
    exact = _exact_logP(tree, cls, rho, pi, S, rc)
    d = elbo_rescorr_bruteforce(tree, cls, rho, pi, S, rc, return_terms=True)
    assert abs(d["Zq"] - 1.0) < 1e-8
    assert abs(d["elbo"] - exact) < 1e-7, f"m=1 depth2 rc={rc}: {d['elbo']} vs {exact}"


@pytest.mark.parametrize("rc", [0.5, 1.0, 2.0])
def test_depth2_m2_certified_and_tighter_than_floor(rc):
    rng = np.random.default_rng(int(rc * 10) + 9)
    pi, S, rho = _model(rng, Kc=2)
    cls = np.array([0, 1], np.int32)
    tree = _depth2(2)
    exact = _exact_logP(tree, cls, rho, pi, S, rc)
    rescorr = elbo_rescorr_bruteforce(tree, cls, rho, pi, S, rc)
    floor = elbo_meanfield(tree, cls, rho, pi, S, rc)
    assert rescorr <= exact + 1e-7, "residue-correlated ELBO violates the bound"
    assert floor <= exact + 1e-7, "mean-field floor violates the bound"
    assert rescorr > floor + 1e-3, "residue-correlated should be strictly tighter"
    # and materially tighter: closes most of the floor's gap
    assert (exact - rescorr) < 0.25 * (exact - floor)
