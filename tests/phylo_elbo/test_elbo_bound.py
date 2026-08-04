"""Certification: the dynfield ELBO functional is a genuine lower bound.

For small trees (m=1, so the full hidden state can be enumerated) we compute the
EXACT tree marginal likelihood log P(data) by brute force, the moment-matching
forward's log Z_MM, and the certified ELBO from elbo.elbo(). The load-bearing
assertion is

    ELBO <= log P(data)      (for every tree, depth, rho_chain)

which is what makes the bound "certified" -- unlike log Z_MM, which is an
EP-style estimate and is NOT constrained to sit below the truth (we exhibit
cases where it does not). At depth-1 cherries the family is exact, so the ELBO
matches log P (up to the there-tight Jensen terms).
"""
import itertools
import numpy as np
import pytest

from tkfdp.coupling.dynfield.phylo_elbo.elbo import elbo
from tkfdp.coupling.dynfield.phylo_elbo.tree_log_lik import tree_log_lik_mm
from tkfdp.coupling.dynfield.phylo_elbo.tree import make_cherry, make_balanced_binary


def _compound_kernel(x_p, th_p, x_v, th_v, c, tau, pi_field, S, rho, rho_chain):
    """K(x_v,th_v | x_p,th_p; tau) for one site (parent p -> child v)."""
    A = pi_field.shape[2]
    beta = np.exp(-rho_chain * (1 - rho[th_p]) * tau)
    Q = np.zeros((A, A))
    for a in range(A):
        for b in range(A):
            if a != b:
                Q[a, b] = S[a, b] * pi_field[c, th_p, b]
        Q[a, a] = -Q[a, :].sum()
    from scipy.linalg import expm
    P = expm(Q * tau)
    no_jump = (beta if th_v == th_p else 0.0) * P[x_p, x_v]
    et = np.exp(-rho_chain * tau)
    K_field = (et if th_p == th_v else 0.0) + (1 - et) * rho[th_v]
    W = K_field - (beta if th_v == th_p else 0.0)
    return no_jump + W * pi_field[c, th_v, x_v]


def _exact_log_marginal(tree, classes, rho, pi_field, S, rho_chain):
    """Brute-force exact log P(data) for m=1 (enumerate theta everywhere + x at
    internal nodes)."""
    L = int(rho.shape[0]); A = pi_field.shape[2]; c = int(classes[0])
    n = tree.n_nodes; root = tree.root
    is_leaf = [tree.is_leaf(v) for v in range(n)]
    internal = [v for v in range(n) if not is_leaf[v]]
    total = 0.0
    for th in itertools.product(range(L), repeat=n):
        for xi in itertools.product(range(A), repeat=len(internal)):
            x = np.zeros(n, np.int64)
            for i, v in enumerate(internal):
                x[v] = xi[i]
            for v in range(n):
                if is_leaf[v]:
                    x[v] = int(tree.leaf_obs[v, 0])
            p = rho[th[root]] * pi_field[c, th[root], x[root]]
            for v in range(n):
                if v == root:
                    continue
                par = int(tree.parent[v])
                p *= _compound_kernel(x[par], th[par], x[v], th[v], c,
                                      float(tree.branch_length[v]),
                                      pi_field, S, rho, rho_chain)
            total += p
    return float(np.log(total))


def _random_model(rng, L=2, A=3):
    pi_field = rng.dirichlet(np.ones(A), size=(1, L))
    S_raw = rng.uniform(0.5, 2.0, size=(A, A)); S = (S_raw + S_raw.T) / 2
    np.fill_diagonal(S, 0.0)
    rho = rng.dirichlet(np.full(L, 2.0))
    return pi_field, S, rho


def _trees():
    yield "cherry", make_cherry(tau=0.5, leaf_obs=np.array([[0], [2]], np.int32))
    yield "depth2", make_balanced_binary(
        depth=2, tau=0.4, leaf_obs=np.array([[0], [1], [2], [0]], np.int32))


# rho_chain > 0 only: at exactly 0 the field is frozen and the mean-field-over-
# nodes family cannot represent a soft-but-perfectly-correlated field (the honest
# value there is -inf / vacuous), and that frozen bin is handled exactly by other
# means (the invariant-rate fast path). The bound is what we certify for rc > 0.
@pytest.mark.parametrize("rho_chain", [0.15, 0.3, 1.0, 3.0])
@pytest.mark.parametrize("L", [2])
def test_elbo_is_lower_bound(rho_chain, L):
    rng = np.random.default_rng(int(rho_chain * 10) + 100 * L)
    pi_field, S, rho = _random_model(rng, L=L)
    classes = np.zeros(1, np.int32)
    for name, tree in _trees():
        exact = _exact_log_marginal(tree, classes, rho, pi_field, S, rho_chain)
        lb = elbo(tree, classes, rho, pi_field, S, rho_chain)
        assert lb <= exact + 1e-7, (
            f"{name} L={L} rho_chain={rho_chain}: ELBO {lb:.6f} > exact "
            f"{exact:.6f} (violates lower-bound guarantee)")


def test_field_factorization_gap():
    """The mean-field-over-nodes q factorizes the field across nodes, so the
    certified bound is valid but NOT tight -- there is a strictly positive gap
    even at a cherry, and it grows with depth. This documents that a
    field-correlated q (tree-structured / exact-field) is needed for tightness.
    """
    rng = np.random.default_rng(7)
    pi_field, S, rho = _random_model(rng, L=2)
    classes = np.zeros(1, np.int32)
    gaps = {}
    for name, tree in _trees():
        exact = _exact_log_marginal(tree, classes, rho, pi_field, S, 0.5)
        lb = elbo(tree, classes, rho, pi_field, S, 0.5)
        gaps[name] = exact - lb
        assert lb <= exact + 1e-7            # still a valid bound
        assert gaps[name] > 1e-3             # but not tight (field factorized)
    # gap compounds with depth
    assert gaps["depth2"] > gaps["cherry"]
