"""Stage-A bit-exact reductions for the archetype-orbit scorer
(orbit_scorer.score_cluster_orbit), each against an INDEPENDENT ground truth:

  1. static  : all-singleton orbits  ==  product of plain GTR Felsensteins
  2. coupled : one pair-orbit {a,b}  ==  compound-generator brute force (pair)
  3. covarion: singleton in a pair-orbit == compound-generator brute force
               (single); and a different-orbit pair == sum of the two singles.
"""
import numpy as np
from scipy.linalg import expm

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from tkfdp.coupling.dynfield.phylo_elbo.exact_cap2 import (
    gtr_Q, _postorder, cherry_ll_bruteforce)
from tkfdp.coupling.dynfield.phylo_elbo.orbit_scorer import score_cluster_orbit

A, KA = 5, 4                                   # small alphabet + archetypes


def _model(seed=0):
    rng = np.random.default_rng(seed)
    pi = rng.random((KA, A)) + 0.2; pi /= pi.sum(1, keepdims=True)
    S = rng.random((A, A)); S = S + S.T; np.fill_diagonal(S, 0.0)
    return pi, S, rng


def _cherry(tauL, tauR):
    return (np.array([-1, 0, 0]), np.array([0.0, tauL, tauR]), 0)   # parent,tau,root


def plain_felsenstein(parent, tau, leaf_obs, root, k, pi_arch, S):
    """Independent plain GTR Felsenstein for a single column at fixed archetype k."""
    Q = gtr_Q(pi_arch[k], S)
    from collections import defaultdict
    ch = defaultdict(list)
    for v in range(len(parent)):
        if parent[v] >= 0:
            ch[parent[v]].append(v)
    M, ls = {}, {}
    for v in _postorder(root, ch):
        if not ch[v]:
            x = int(leaf_obs[v]); M[v] = np.ones(A) if x < 0 else np.eye(A)[x]; ls[v] = 0.0
        else:
            m = np.ones(A); s = 0.0
            for c in ch[v]:
                m = m * (expm(Q * float(tau[c])) @ M[c]); s += ls[c]
            mx = m.max(); m = m / mx; s += np.log(mx); M[v] = m; ls[v] = s
    return float(np.log(pi_arch[k] @ M[root]) + ls[root])


def single_cherry_bruteforce(oL, oR, tauL, tauR, a1, pi_arch, S, rho, rho_chain):
    """Compound (theta,x) generator brute force for ONE covarion column."""
    L = len(rho); N = L * A
    idx = lambda t, x: t * A + x
    Qk = {k: gtr_Q(pi_arch[k], S) for k in set(int(x) for x in a1)}
    Q = np.zeros((N, N))
    for t in range(L):
        for x in range(A):
            i = idx(t, x)
            for xp in range(A):
                if xp != x:
                    Q[i, idx(t, xp)] += Qk[int(a1[t])][x, xp]
            for tp in range(L):
                if tp != t:
                    for xp in range(A):
                        Q[i, idx(tp, xp)] += rho_chain * rho[tp] * pi_arch[int(a1[tp])][xp]
    for i in range(N):
        Q[i, i] = -(Q[i].sum() - Q[i, i])
    PL, PR = expm(Q * tauL), expm(Q * tauR)
    tot = 0.0
    for t in range(L):
        for x in range(A):
            pri = rho[t] * pi_arch[int(a1[t])][x]; i = idx(t, x)
            pl = sum(PL[i, idx(tp, oL)] for tp in range(L))
            pr = sum(PR[i, idx(tp, oR)] for tp in range(L))
            tot += pri * pl * pr
    return float(np.log(tot))


def test_static_reduction():
    """All-singleton orbits: a pair cluster == two independent GTR columns,
    for any rate (no field to move)."""
    pi, S, rng = _model(1)
    parent, tau, root = _cherry(0.3, 0.7)
    orbit_id = np.arange(KA)                    # all singletons
    a, b = 0, 2
    leaf_cols = np.array([[-1, -1], [1, 3], [4, 0]])   # root, leafL(i,j), leafR(i,j)
    ref = (plain_felsenstein(parent, tau, leaf_cols[:, 0], root, a, pi, S)
           + plain_felsenstein(parent, tau, leaf_cols[:, 1], root, b, pi, S))
    for rate in (0.0, 1.3, 5.0):
        got = score_cluster_orbit(parent, tau, root, leaf_cols, [a, b],
                                  orbit_id, pi, S, rho_chain=0.4, rate=rate)
        assert abs(got - ref) < 1e-9, (rate, got, ref)


def test_coupled_pair_vs_bruteforce():
    """One pair-orbit {a,b}, pair cluster (a,b): coupled forward == compound
    generator brute force with a1=[a,b], a2=[b,a], rho=uniform."""
    pi, S, rng = _model(2)
    parent, tau, root = _cherry(0.5, 0.9)
    a, b = 1, 3
    orbit_id = np.array([0, 1, 2, 1])          # archetypes 1 and 3 share orbit 1
    leaf_cols = np.array([[-1, -1], [2, 0], [4, 1]])
    oL, oR = (leaf_cols[1, 0], leaf_cols[1, 1]), (leaf_cols[2, 0], leaf_cols[2, 1])
    rc = 0.6
    for rate in (0.0, 1.0, 2.5):
        got = score_cluster_orbit(parent, tau, root, leaf_cols, [a, b],
                                  orbit_id, pi, S, rho_chain=rc, rate=rate)
        ref = cherry_ll_bruteforce(oL, oR, tau[1], tau[2],
                                   np.array([a, b]), np.array([b, a]),
                                   pi, S, np.array([0.5, 0.5]), rc * rate)
        assert abs(got - ref) < 1e-9, (rate, got, ref)


def test_covarion_single_and_factorised_pair():
    """A singleton in a pair-orbit == single compound brute force; and a pair
    whose columns live in DIFFERENT pair-orbits factorises into two singles."""
    pi, S, rng = _model(3)
    parent, tau, root = _cherry(0.4, 0.6)
    # orbits: {0,1} and {2,3}
    orbit_id = np.array([0, 0, 1, 1])
    rc, rate = 0.7, 1.5
    # (a) singleton cluster, class 0 in orbit {0,1}
    leaf_single = np.array([[-1], [2], [4]])
    got_s = score_cluster_orbit(parent, tau, root, leaf_single, [0],
                                orbit_id, pi, S, rho_chain=rc, rate=rate)
    ref_s = single_cherry_bruteforce(2, 4, tau[1], tau[2], np.array([0, 1]),
                                     pi, S, np.array([0.5, 0.5]), rc * rate)
    assert abs(got_s - ref_s) < 1e-9, (got_s, ref_s)
    # (b) DIFFERENT orbits {0,1} and {2,3}: INDEPENDENT async per-orbit fields,
    # so the cluster FACTORISES into two single-column (covarion) forwards
    # ({AC,AD,BC,BD} collapses to two unsynchronised singletons).
    leaf_cols = np.array([[-1, -1], [2, 3], [4, 1]])
    got_p = score_cluster_orbit(parent, tau, root, leaf_cols, [0, 2],
                                orbit_id, pi, S, rho_chain=rc, rate=rate)
    s0 = single_cherry_bruteforce(2, 4, tau[1], tau[2], np.array([0, 1]),
                                  pi, S, np.array([0.5, 0.5]), rc * rate)
    s1 = single_cherry_bruteforce(3, 1, tau[1], tau[2], np.array([2, 3]),
                                  pi, S, np.array([0.5, 0.5]), rc * rate)
    assert abs(got_p - (s0 + s1)) < 1e-9, (got_p, s0 + s1)


if __name__ == "__main__":
    test_static_reduction(); print("static OK")
    test_coupled_pair_vs_bruteforce(); print("coupled OK")
    test_covarion_single_and_factorised_pair(); print("covarion+factorised OK")
    print("ALL STAGE-A REDUCTIONS PASS")


# --------------------------------------------------------------- Stage B: JAX == numpy
def _bin(branch, n_bins=16):
    from tkfdp.coupling.dynfield.phylo_elbo.tau_binning import build_tau_bins, assign_bins
    pos = branch[branch > 0]
    bc = build_tau_bins(np.asarray(pos), n_bins=n_bins)
    return bc, bc[assign_bins(branch, bc)]


def test_jax_matches_numpy():
    """Stage B: the JAX positional scorer reproduces the numpy Stage-A scorer to
    machine precision on a 4-leaf tree (same binned taus) across L in {1,2,4} --
    including the COUPLED different-orbit 4-state case."""
    from tkfdp.coupling.dynfield.phylo_elbo.tree import build_tree
    from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import build_padded_tree
    from tkfdp.coupling.dynfield.phylo_elbo import orbit_scorer_jax as oj
    from tkfdp.coupling.dynfield.phylo_elbo.orbit_scorer import score_cluster_orbit

    pi, S, rng = _model(7)
    parent = np.array([4, 4, 5, 5, 6, 6, -1])           # leaves 0..3, internals 4,5, root 6
    branch = np.array([0.3, 0.5, 0.2, 0.7, 0.4, 0.6, 0.1])
    bc, taub = _bin(branch)
    leaf2 = np.array([[1, 3], [4, 0], [2, 1], [0, 4]], np.int32)
    lc = np.full((7, 2), -1, np.int32); lc[:4] = leaf2
    rho_chain = 0.55
    T = oj.build_arch_P(pi, S, bc)
    pt2 = build_padded_tree(build_tree(parent, branch, leaf2))
    pt0 = build_padded_tree(build_tree(parent, branch, leaf2[:, [0]]))
    pt1 = build_padded_tree(build_tree(parent, branch, leaf2[:, [1]]))

    # oj.score_cluster is a per-ORBIT-GROUP forward (columns sharing one orbit).
    # Single-group cases: same-orbit pair (L2), static single (L1), covarion (L2).
    cases = [
        ("coupled  L2", [1, 3], np.array([0, 1, 2, 1]), False),   # 1,3 share orbit
        ("static   L1", [0],    np.arange(4),           True),    # singleton orbit
        ("covarion L2", [0],    np.array([0, 0, 1, 1]), True),    # pair-orbit {0,1}
    ]
    for rate in (0.0, 1.4, 3.0):
        for name, cls, oid, single in cases:
            pt = pt0 if single else pt2
            lcx = lc[:, [0]] if single else lc
            got = oj.score_cluster(pt, cls, oid, T, rho_chain, rate)
            ref = score_cluster_orbit(parent, taub, 6, lcx, cls, oid, pi, S,
                                      rho_chain, rate)
            assert abs(got - ref) < 1e-8, (name, rate, got, ref)

        # Different-orbit pair FACTORISES: JAX per-group sum == numpy (model B).
        oid = np.array([0, 0, 1, 1])                   # orbits {0,1}, {2,3}
        got = (oj.score_cluster(pt0, [0], oid, T, rho_chain, rate)
               + oj.score_cluster(pt1, [2], oid, T, rho_chain, rate))
        ref = score_cluster_orbit(parent, taub, 6, lc, [0, 2], oid, pi, S,
                                  rho_chain, rate)
        assert abs(got - ref) < 1e-8, ("factorised", rate, got, ref)


def test_batched_matches_per_tree():
    """Stage-B batching: score_units_batched == per-unit score_cluster (1e-8),
    across a mixed bucket of m/L units on a shared tree shape."""
    from tkfdp.coupling.dynfield.phylo_elbo.tree import build_tree
    from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import build_padded_tree
    from tkfdp.coupling.dynfield.phylo_elbo import orbit_scorer_jax as oj

    pi, S, rng = _model(11)
    parent = np.array([4, 4, 5, 5, 6, 6, -1])
    branch = np.array([0.3, 0.5, 0.2, 0.7, 0.4, 0.6, 0.1])
    bc, _ = _bin(branch)
    T = oj.build_arch_P(pi, S, bc)
    rho_chain = 0.5

    # (leaf-obs 2-col, classes, orbit_id) -- columns of each unit share an orbit
    specs = [
        (np.array([[1, -1], [3, -1], [0, -1], [2, -1]]), [0], np.arange(4)),           # m1 L1
        (np.array([[2, -1], [1, -1], [4, -1], [0, -1]]), [0], np.array([0, 0, 1, 1])), # m1 L2
        (np.array([[1, 3], [4, 0], [2, 1], [0, 4]]),     [1, 3], np.array([0, 1, 2, 1])), # m2 L2
        (np.array([[0, 2], [3, 1], [4, 4], [2, 0]]),     [1, 3], np.array([0, 1, 2, 1])), # m2 L2
        (np.array([[3, -1], [0, -1], [1, -1], [4, -1]]), [2], np.array([0, 0, 1, 1])), # m1 L2
    ]
    for rate in (0.0, 1.7):
        units, refs = [], []
        for leaf, cls, oid in specs:
            m = len(cls)
            pt = build_padded_tree(build_tree(parent, branch, leaf[:, :m]))
            _, a_cols = oj.joint_field(cls, oid)
            units.append((pt, a_cols))
            refs.append(oj.score_cluster(pt, cls, oid, T, rho_chain, rate))
        got = oj.score_units_batched(units, T, rho_chain, rate)
        assert np.allclose(got, refs, atol=1e-8), (rate, got, np.array(refs))
