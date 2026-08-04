"""Numpy reference scorer for the archetype-orbit permutation-field model
(docs/orbit_field_implementation.md, appendix par:archetype-orbits).

Frozen-diagonal base: a column's class *is* its base archetype. The orbit
partition `orbit_id` groups archetypes; a cluster's field is the permutation
of the relevant orbit, so the field alphabet is Sym(orbit), enumerated as
permutations of that orbit's archetypes (identity first).

Per-cluster likelihood FACTORISES over orbits (only a same-orbit pair is
coupled); we reuse the validated exact_cap2 branch operators, which already
take per-field-state archetype-index arrays a1[phi], a2[phi]:

  * singleton cluster          -> single-column forward over Sym(orbit)
  * pair, same orbit           -> coupled pair forward over Sym(orbit)
  * pair, different orbits     -> product of two single-column forwards

This module is the Stage-A reference; a JAX-batched version follows once the
reductions here are pinned bit-exact.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import permutations

import numpy as np
from scipy.linalg import expm

from .exact_cap2 import gtr_Q, branch_single, _postorder


# ------------------------------------------------------------------ orbits

def orbit_members(orbit_id: np.ndarray) -> 'dict[int, list[int]]':
    """orbit label -> sorted archetype indices."""
    d: 'dict[int, list[int]]' = defaultdict(list)
    for k, b in enumerate(np.asarray(orbit_id).tolist()):
        d[int(b)].append(int(k))
    return {b: sorted(v) for b, v in d.items()}


def orbit_perms(members: 'list[int]') -> 'list[dict[int, int]]':
    """Enumerate Sym(members) as archetype->image maps, IDENTITY FIRST."""
    members = list(members)
    idp = {a: a for a in members}
    out = [idp]
    for p in permutations(members):
        pm = {a: p[i] for i, a in enumerate(members)}
        if pm != idp:
            out.append(pm)
    return out                                   # len = |members|!


def _a_of(perms, cls):
    """a[phi] = perm(cls) for each enumerated field state phi."""
    return np.asarray([pm[cls] for pm in perms], dtype=np.int64)


# ------------------------------------------------- single-column tree forward
# (exact_cap2 has the PAIR tree forward; the orbit factorisation also needs the
# single-column one. Same postorder + per-slot log-scaling, message (L, A).)

def exact_single_ll_tree(parent, tau, leaf_obs, root, a1, pi_arch, S,
                         rho, rho_chain) -> float:
    """Exact single-column log-lik on a general tree via the factored branch
    operator. leaf_obs (n,) int: observed residue at leaves, -1 internal/gap.
    a1 (L,) archetype index per field state; rho (L,) field stationary."""
    L, A = rho.shape[0], pi_arch.shape[1]
    n = len(parent)
    children = defaultdict(list)
    for v in range(n):
        if parent[v] >= 0:
            children[parent[v]].append(v)
    ks = set(int(x) for x in a1.tolist())
    Qk = {k: gtr_Q(pi_arch[k], S) for k in ks}
    P_cache: dict = {}

    def P_at(t):
        key = round(float(t), 12)
        if key not in P_cache:
            P_cache[key] = {k: expm(Qk[k] * t) for k in ks}
        return P_cache[key]

    def leaf_msg(x):
        M = np.zeros((L, A))
        M[:] = (np.ones(A) if x < 0 else np.eye(A)[x])[None]
        return M

    order = _postorder(root, children)
    M, logscale = {}, {}
    for v in order:
        if not children[v]:
            Mv = leaf_msg(int(leaf_obs[v])); lsc = 0.0
        else:
            Mv = np.ones((L, A)); lsc = 0.0
            for c in children[v]:
                Mv = Mv * branch_single(M[c], float(tau[c]), a1, P_at(float(tau[c])),
                                        pi_arch, rho, rho_chain)
                lsc += logscale[c]
            mx = Mv.max()
            if mx > 0:
                Mv = Mv / mx; lsc += np.log(mx)
        M[v] = Mv; logscale[v] = lsc
    m_root = np.array([pi_arch[a1[t]] @ M[root][t] for t in range(L)])
    return float(np.log(rho @ m_root) + logscale[root])


# ------------------------------------------------------------- cluster scorer

def score_cluster_orbit(parent, tau, root, leaf_cols, classes, orbit_id,
                        pi_arch, S, rho_chain, rate) -> float:
    """Log-lik of one cluster under INDEPENDENT (asynchronous) per-orbit fields.

    Each orbit's permutation evolves as its own F81-on-DP chain, so the cluster
    likelihood FACTORISES over the distinct orbits its columns touch: columns
    that SHARE an orbit stay coupled (a 2-cycle {AB,BA} or {AA,BB}), while
    columns in different orbits are independent (the {AC,AD,BC,BD} case
    collapses to two async singletons, one oscillating A<->B, one C<->D). Each
    forward therefore runs on Sym(one orbit), L = |B|! <= s_max! = 2.

      leaf_cols : (n, m) int observed residue per node per column (-1 gap/internal)
      classes   : (m,) int base archetype (== class) of each column
      orbit_id  : (K,) int archetype -> orbit label
      rate      : field-rate multiplier r (r=0 -> static/invariant bin)
    """
    from collections import defaultdict
    from .exact_cap2 import exact_pair_ll_tree
    classes = [int(c) for c in classes]
    members = orbit_members(orbit_id)
    rc = float(rho_chain) * float(rate)
    leaf_cols = np.asarray(leaf_cols)

    groups = defaultdict(list)                         # orbit label -> column positions
    for pos, c in enumerate(classes):
        groups[int(orbit_id[c])].append(pos)

    total = 0.0
    for b, positions in groups.items():
        perms = orbit_perms(members[b]); L = len(perms)
        rho = np.full(L, 1.0 / L)
        if len(positions) == 1:
            p = positions[0]
            total += exact_single_ll_tree(parent, tau, leaf_cols[:, p], root,
                                          _a_of(perms, classes[p]), pi_arch, S, rho, rc)
        else:                                          # >=2 columns share orbit b
            p0, p1 = positions[0], positions[1]
            total += exact_pair_ll_tree(parent, tau, leaf_cols[:, [p0, p1]], root,
                                        _a_of(perms, classes[p0]),
                                        _a_of(perms, classes[p1]),
                                        pi_arch, S, rho, rc)
    return total
