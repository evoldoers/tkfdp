"""Mood-light field CTMC on the symmetric group S_C (permutations of C
archetypes) -- the simpler model of paper 2b Sec. 4, replacing the enum400/DM/
renewal machinery of ``coupling.dynfield``.

Model
-----
Each archetype is an LG08-style GTR process with its own equilibrium pi^c
(shared exchangeability S_LG08, archetype-specific stationary). A site in class
c uses, under field state theta, archetype theta(c). The field theta in S_C is a
continuous-time Markov chain whose jumps are TRANSPOSITIONS (swap two
archetypes a,b): theta' = (a b) o theta. Every jump therefore changes the
transposition distance from the identity

    d(theta) = C - #cycles(theta)          (the Cayley distance on S_C)

by exactly +-1, so the off-diagonal support is C(C-1)/2 per row -- a sparsity
constraint on the field exchangeabilities.

The reversible generator is a GTR form on the transposition edges,
    Q[i,j] = s_{min(d_i,d_j)} * w_{a,b} * pi_field[j],
where theta_j = (a b) o theta_i, and:
  * stationary          pi_field(theta) = p[d(theta)] / Z          (p_0..p_{C-1});
  * distance factor     s_d on edges between distances d and d+1   (s_0..s_{C-2});
  * archetype-pair factor  w_{a,b}, symmetric/unordered, one per swapped pair
                        {a,b}, so the jump rate may depend on which archetypes
                        are swapped as well as on d(source), d(dest).
The GTR form makes the chain reversible w.r.t. pi_field for any p, s, w >= 0.
Zeroing p_d for d>=2 recovers a star of single-swaps around the identity, i.e.
something close to the enum400 single-transposition field.

Free field parameters: p_0..p_{C-1} (C-1 after normalisation);
s_0..s_{C-2} (C-1) and w over the C(C-1)/2 archetype pairs, sharing one overall
scale (fixed by mean-rate-1), so the exchangeability contributes
(C-1) + C(C-1)/2 - 1 free rates. Archetypes add C*19.

Validated for C=2,3,4 (rows sum to 0; pi_field Q = 0; detailed balance;
C(C-1)/2 swaps per row; rate depends only on ({a,b}, d_i, d_j); |d_i-d_j|=1 on
every edge; mean rate 1 after normalisation).
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np


def n_cycles(perm) -> int:
    """Number of cycles of a permutation given as a tuple perm[i] = image of i."""
    C = len(perm)
    seen = [False] * C
    c = 0
    for i in range(C):
        if not seen[i]:
            c += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = perm[j]
    return c


def transposition_distance(perm) -> int:
    """Cayley distance from the identity on S_C = C - #cycles (minimum number of
    transpositions whose product is perm)."""
    return len(perm) - n_cycles(perm)


def build_field(C: int, p, s=None, w=None, normalize_rate: bool = True):
    """Build the C!-state transposition field generator.

    p : (C,) nonnegative stationary weights by transposition distance p_0..p_{C-1}.
    s : (C-1,) nonnegative distance-level factors s_0..s_{C-2} (edge between
        distances d and d+1); default all-ones.
    w : (C(C-1)/2,) nonnegative archetype-pair factors in the pair order
        [(0,1),(0,2),...,(C-2,C-1)]; symmetric/unordered; default all-ones.
    normalize_rate : scale Q so the stationary mean exit rate is 1.

    Returns (states, Q, pi, dist):
      states : list of C! permutation tuples (theta(c) = archetype of class c)
      Q      : (C!, C!) generator, off-diagonal support = transposition edges
      pi     : (C!,) stationary = p[dist]/sum
      dist   : (C!,) transposition distance of each state
    """
    states = list(itertools.permutations(range(C)))
    n = len(states)
    idx = {t: i for i, t in enumerate(states)}
    dist = np.array([transposition_distance(t) for t in states])
    p = np.asarray(p, float)
    pi = p[dist] / p[dist].sum()
    s = np.ones(C - 1) if s is None else np.asarray(s, float)
    pairs = [(a, b) for a in range(C) for b in range(a + 1, C)]
    w = np.ones(len(pairs)) if w is None else np.asarray(w, float)
    wmap = {ab: w[k] for k, ab in enumerate(pairs)}
    Q = np.zeros((n, n))
    for i, t in enumerate(states):
        for (a, b) in pairs:                                 # theta' = (a b) o theta
            tp = tuple(b if x == a else a if x == b else x for x in t)
            j = idx[tp]
            Q[i, j] = s[min(dist[i], dist[j])] * wmap[(a, b)] * pi[j]   # GTR (reversible)
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(n)] = -Q.sum(1)
    if normalize_rate:
        Q = Q / max(-(pi * np.diag(Q)).sum(), 1e-30)   # guard degenerate divisor
    return states, Q, pi, dist


def build_truncated_field(C: int, p, s0: float = 1.0, w=None,
                          normalize_rate: bool = True):
    """Cayley<=1 TRUNCATED field: only the identity (d=0) and the C(C-1)/2 single
    transpositions (d=1) -- 1 + C(C-1)/2 states instead of C!. Since composing two
    distinct transpositions has Cayley distance 2 (dropped), the only surviving
    edges are id <-> each transposition: a STAR. Reversible GTR form (matches
    build_field restricted to d<=1):

        Q[id, tau_ab] = s0 * w_ab * pi_field[tau_ab]
        Q[tau_ab, id] = s0 * w_ab * pi_field[id]           (no tau<->tau edges)

    p : (2,) stationary weights (p0 for identity, p1 shared by all transpositions).
    s0: single distance-0<->1 factor. w: (C(C-1)/2,) archetype-pair swap factors.

    Returns (states, Q, pi, dist, arch) with arch[i,c] = archetype of class c in
    state i.  Free field params here: p1/p0 (1), w over C(C-1)/2 pairs (shared
    scale) -> the "archetype-swap rates"; archetypes add C*19 as usual.
    """
    ident = tuple(range(C))
    pairs = [(a, b) for a in range(C) for b in range(a + 1, C)]
    states = [ident] + [tuple(b if x == a else a if x == b else x for x in ident)
                        for (a, b) in pairs]
    K = len(pairs); n = 1 + K
    dist = np.array([0] + [1] * K)
    p = np.asarray(p, float)
    pi = np.concatenate([[p[0]], np.full(K, p[1])]); pi = pi / pi.sum()
    w = np.ones(K) if w is None else np.asarray(w, float)
    Q = np.zeros((n, n))
    for k in range(K):
        j = 1 + k
        Q[0, j] = s0 * w[k] * pi[j]
        Q[j, 0] = s0 * w[k] * pi[0]
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(n)] = -Q.sum(1)
    if normalize_rate:
        Q = Q / max(-(pi * np.diag(Q)).sum(), 1e-30)   # guard degenerate divisor
    arch = np.array([list(t) for t in states])
    return states, Q, pi, dist, arch


def field_dims(C: int) -> dict:
    """Parameter-count summary for a field of C archetypes."""
    return {
        "n_states": _fact(C),
        "swaps_per_row": comb(C, 2),
        "stationary_free": C - 1,                    # p_0..p_{C-1}, normalised
        "exchangeability": (C - 1) + comb(C, 2) - 1,  # s_d + w_{ab}, one shared scale
        "archetype_free": 19 * C,                    # C stationaries pi^c, 19 free each
    }


def _fact(k):
    r = 1
    for i in range(2, k + 1):
        r *= i
    return r
