"""Exactness of the residue-correlated tree-structured variational form.

The mm_clv rank-1+scalar message IS the residue-correlated form (r*prod A =
Delta=0 tracking with per-site residue correlation; s = Delta=1 renewal). The
appendix (eq:res-corr-branch) predicts its maximised ELBO -- which the forward
log-likelihood realises wherever no combine-projection is lossy -- is EXACT
(= log P) in two regimes this test pins down against independent oracles:

  * singletons (m=1) at ANY depth   -- oracle: exact (theta,x) L*A Felsenstein;
  * cherries at ANY m               -- oracle: brute-force over theta + x_root.

Equality (not just <=) is asserted to machine precision. This is the headline
property of the residue-correlated family: it closes the m=1-at-depth gap that
the residue mean-field left open.
"""
import itertools
import numpy as np
import pytest
from scipy.linalg import expm

from tkfdp.coupling.dynfield.phylo_elbo.tree_log_lik import tree_log_lik_mm
from tkfdp.coupling.dynfield.phylo_elbo.tree import make_cherry, make_balanced_binary


def _sub_P(pi_field, S, c, theta, tau):
    A = pi_field.shape[2]
    Q = np.zeros((A, A))
    for a in range(A):
        for b in range(A):
            if a != b:
                Q[a, b] = S[a, b] * pi_field[c, theta, b]
        Q[a, a] = -Q[a, :].sum()
    return expm(Q * tau)


def _branch_factor(xu, thu, xv, thv, classes, tau, pi_field, S, rho, rho_chain):
    """K(x_v_vec, th_v | x_u_vec, th_u) for an m-site cluster on one branch."""
    m = len(classes)
    beta = np.exp(-rho_chain * (1 - rho[thu]) * tau)
    et = np.exp(-rho_chain * tau)
    P_theta = (et if thu == thv else 0.0) + (1 - et) * rho[thv]
    W = P_theta - (beta if thv == thu else 0.0)
    # Delta=0 (no jump): th_v == th_u, per-site substitution
    nojump = 0.0
    if thv == thu:
        nojump = beta
        for n in range(m):
            nojump *= _sub_P(pi_field, S, int(classes[n]), thu, tau)[xu[n], xv[n]]
    # Delta=1 (jump): renewal from stationary at th_v
    jump = W
    for n in range(m):
        jump *= pi_field[int(classes[n]), thv, xv[n]]
    return nojump + jump


def _exact_m1(tree, classes, rho, pi_field, S, rho_chain):
    """Exact m=1 log P via the (theta,x) joint L*A-state Felsenstein (any depth)."""
    L = rho.shape[0]; A = pi_field.shape[2]; c = int(classes[0])
    msg = [None] * tree.n_nodes
    for v in tree.post_order:
        v = int(v)
        if tree.is_leaf(v):
            mm = np.zeros((L, A)); mm[:, int(tree.leaf_obs[v, 0])] = 1.0; msg[v] = mm
        else:
            mm = np.ones((L, A))
            for ch in tree.children[v]:
                tau = float(tree.branch_length[ch])
                K = np.array([[[[_branch_factor([xp], tp, [xv], tv, classes, tau,
                                                 pi_field, S, rho, rho_chain)
                                 for xv in range(A)] for tv in range(L)]
                               for xp in range(A)] for tp in range(L)])  # [tp,xp,tv,xv]
                mm *= np.einsum('pqvw,vw->pq', K, msg[ch])
            msg[v] = mm
    r = tree.root
    prior = rho[:, None] * pi_field[c]      # (L,A)
    return float(np.log((prior * msg[r]).sum()))


def _exact_cherry(tree, classes, rho, pi_field, S, rho_chain):
    """Exact log P on a cherry (one internal node) for any m: brute force over
    theta (all 3 nodes) and the root residue vector x_root in A^m."""
    L = rho.shape[0]; A = pi_field.shape[2]; m = len(classes)
    r = tree.root
    leaves = [v for v in range(tree.n_nodes) if tree.is_leaf(v)]
    assert len(leaves) == 2 and not tree.is_leaf(r)
    total = 0.0
    for thr, tl0, tl1 in itertools.product(range(L), repeat=3):
        thetas = {r: thr, leaves[0]: tl0, leaves[1]: tl1}
        for xr in itertools.product(range(A), repeat=m):
            p = rho[thr]
            for n in range(m):
                p *= pi_field[int(classes[n]), thr, xr[n]]
            for lv in leaves:
                obs = [int(tree.leaf_obs[lv, n]) for n in range(m)]
                p *= _branch_factor(list(xr), thr, obs, thetas[lv], classes,
                                    float(tree.branch_length[lv]), pi_field, S,
                                    rho, rho_chain)
            total += p
    return float(np.log(total))


def _model(rng, L=2, A=3, K_c=1):
    pi_field = rng.dirichlet(np.ones(A), size=(K_c, L))
    Sr = rng.uniform(0.5, 2.0, (A, A)); S = (Sr + Sr.T) / 2; np.fill_diagonal(S, 0.0)
    rho = rng.dirichlet(np.full(L, 2.0))
    return pi_field, S, rho


@pytest.mark.parametrize("rho_chain", [0.15, 0.5, 1.0, 3.0])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_singleton_exact_any_depth(rho_chain, depth):
    """m=1: the mm_clv forward equals the exact (theta,x) log P at any depth."""
    rng = np.random.default_rng(depth * 7 + int(rho_chain * 10))
    pi_field, S, rho = _model(rng, L=2)
    classes = np.zeros(1, np.int32)
    n_leaves = 2 ** depth
    leaf_obs = np.array([[i % 3] for i in range(n_leaves)], np.int32)
    tree = (make_cherry(0.5, leaf_obs) if depth == 1
            else make_balanced_binary(depth=depth, tau=0.4, leaf_obs=leaf_obs))
    exact = _exact_m1(tree, classes, rho, pi_field, S, rho_chain)
    mm = float(tree_log_lik_mm(tree, classes, rho, pi_field, S, rho_chain))
    assert abs(exact - mm) < 1e-9, f"m=1 depth={depth} rc={rho_chain}: {mm} vs {exact}"


@pytest.mark.parametrize("rho_chain", [0.15, 0.5, 1.0, 3.0])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_cherry_exact_any_m(rho_chain, m):
    """Cherry: the mm_clv forward equals the exact log P for any cluster width m."""
    rng = np.random.default_rng(m * 11 + int(rho_chain * 10))
    pi_field, S, rho = _model(rng, L=2, K_c=max(1, m))
    classes = np.arange(m, dtype=np.int32) % pi_field.shape[0]
    leaf_obs = np.array([[(i + n) % 3 for n in range(m)] for i in range(2)], np.int32)
    tree = make_cherry(0.5, leaf_obs)
    exact = _exact_cherry(tree, classes, rho, pi_field, S, rho_chain)
    mm = float(tree_log_lik_mm(tree, classes, rho, pi_field, S, rho_chain))
    assert abs(exact - mm) < 1e-9, f"cherry m={m} rc={rho_chain}: {mm} vs {exact}"
