#!/usr/bin/env python3
"""permfield mood-light model: exact joint (theta,x) composite-likelihood fit
(JAX autodiff), with a self-consistent synthetic recovery check.

Model (docs/permutation_field_model.md): C archetypes (GTR Q^c = S_LG08 (x) pi^c);
field theta on S_C (permfield.build_field), a C!-state transposition CTMC with
Cayley-distance stationary p_d, distance factors s_d, archetype-pair factors
w_{a,b}. A column of class c evolves its residue under archetype theta(c),
Markov-modulated by the field; the joint (theta,x) generator G_c has C!*20 states
and expm(G_c tau) integrates within-branch field jumps exactly. Each column draws
its class c ~ rho.

Composite (per-column) likelihood: family LL = sum_cols log sum_c rho_c P(col|c),
each P(col|c) an exact joint (theta,x) Felsenstein over the tree. Parameters
{pi^c, p_d, s_d, w_ab, rho} are fit by gradient ascent (optax Adam) through the
differentiable likelihood -- no hand-derived M-steps. The composite shares
parameters across columns but not the field trajectory (the shared-field
product-of-trees ELBO is the documented refinement).

`--synthetic` simulates from the per-column joint model (exactly the fitted
model) and checks parameter recovery; the real-data path fits a CLV family.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import numpy as np

sys.path.insert(0, "src")
import jax                                              # noqa: E402
import jax.numpy as jnp                                 # noqa: E402
jax.config.update("jax_enable_x64", True)
from jax.scipy.linalg import expm as jexpm             # noqa: E402
from jax.scipy.special import logsumexp as jlse        # noqa: E402
import optax                                            # noqa: E402
from tkfdp.lg08 import S_LG08, PI_LG08                  # noqa: E402
from tkfdp.permfield import transposition_distance     # noqa: E402

A = 20
S = jnp.asarray(np.asarray(S_LG08, float))
PI0 = np.asarray(PI_LG08, float)


# ---------- permutation-state bookkeeping ----------
def perm_states(C):
    states = list(itertools.permutations(range(C)))
    dist = np.array([transposition_distance(t) for t in states])
    arch = np.array([list(t) for t in states])           # arch[i,c] = theta_i(c)
    idx = {t: i for i, t in enumerate(states)}
    pairs = [(a, b) for a in range(C) for b in range(a + 1, C)]
    # transposition edge list: (i, j, pair_index, dmin)
    edges = []
    for i, t in enumerate(states):
        for k, (a, b) in enumerate(pairs):
            tp = tuple(b if x == a else a if x == b else x for x in t)
            j = idx[tp]
            edges.append((i, j, k, min(dist[i], dist[j])))
    edges = np.array(edges)                              # (nE, 4)
    return states, dist, arch, np.array(pairs), edges


# ---------- differentiable generators ----------
def gtr_Q(pi):
    Q = S * pi[None, :]
    Q = Q - jnp.diag(jnp.diag(Q))
    Q = Q - jnp.diag(Q.sum(1))
    return Q / (-(pi * jnp.diag(Q)).sum())               # mean rate 1


def field_Q(p, s, w, dist, edges, n):
    pi = p[dist]; pi = pi / pi.sum()
    i, j, k, dmin = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]
    vals = s[dmin] * w[k] * pi[j]                         # GTR reversible
    Q = jnp.zeros((n, n)).at[i, j].set(vals)
    Q = Q - jnp.diag(Q.sum(1))
    return Q / (-(pi * jnp.diag(Q)).sum()), pi


def joint_generator(Qc, Qf, arch_c, n):
    """Joint (theta,x) generator; arch_c[i] = archetype used by this class under
    field state i. Block structure: residue block-diagonal Qc[arch_c[i]], field
    transitions Qf[i,j] * I_A off the diagonal blocks."""
    nA = n * A
    # residue blocks
    res = Qc[arch_c]                                      # (n,A,A)
    G = jnp.zeros((nA, nA))
    rr = (jnp.arange(n)[:, None, None] * A + jnp.arange(A)[None, :, None])
    cc = (jnp.arange(n)[:, None, None] * A + jnp.arange(A)[None, None, :])
    G = G.at[jnp.broadcast_to(rr, (n, A, A)).reshape(-1),
             jnp.broadcast_to(cc, (n, A, A)).reshape(-1)].add(res.reshape(-1))
    # field transitions (x unchanged): block (i,j) = Qf[i,j] * I_A = kron(Qf_offdiag, I_A)
    Gf = jnp.kron(Qf - jnp.diag(jnp.diag(Qf)), jnp.eye(A))
    G = G + Gf
    G = G - jnp.diag(jnp.diag(G))
    G = G - jnp.diag(G.sum(1))
    return G


# ---------- vectorized joint Felsenstein (all columns at once) ----------
def make_felsenstein(parent, tau, nl, N):
    order = _postorder(parent)
    kids = [np.array(_children(parent, v)) for v in range(N)]
    branch_nodes = [v for v in range(N) if parent[v] >= 0]

    def felsenstein(B, rd, leaf_msg):
        """B: (N,nS,nS) branch ops (identity where no parent); rd: (nS,);
        leaf_msg: (nl, nS, L). Returns per-column loglik (L,)."""
        msg = [None] * N
        for v in range(nl):
            msg[v] = leaf_msg[v]                          # (nS,L)
        for v in order:
            if v < nl:
                continue
            m = None
            for u in kids[v]:
                contrib = B[u] @ msg[u]                  # (nS,L)
                m = contrib if m is None else m * contrib
            msg[v] = m
        root = order[-1]
        return jnp.log(jnp.maximum(jnp.einsum("s,sL->L", rd, msg[root]), 1e-300))
    return felsenstein, branch_nodes


def _children(parent, v):
    return [u for u in range(len(parent)) if parent[u] == v]


def _postorder(parent):
    root = int(np.where(parent < 0)[0][0])
    order, stack = [], [root]
    while stack:
        v = stack.pop(); order.append(v); stack.extend(_children(parent, v))
    return order[::-1]


# ---------- family composite log-likelihood ----------
def build_loglik(parent, tau, msa, C):
    states, dist, arch, pairs, edges = perm_states(C)
    n = len(states); N = len(parent); nl, L = msa.shape
    tau_j = jnp.asarray(tau)
    dist_j = jnp.asarray(dist)
    arch_j = jnp.asarray(arch)
    edges_j = jnp.asarray(edges)
    felsenstein, branch_nodes = make_felsenstein(parent, tau, nl, N)
    bn = np.array(branch_nodes)

    # leaf messages (nl, nS, L): 1 at (theta,x=r) for observed r, all-ones if gap
    leaf = np.zeros((nl, n * A, L))
    for v in range(nl):
        for j in range(L):
            r = int(msa[v, j])
            if r < A:
                leaf[v, np.arange(n) * A + r, j] = 1.0
            else:
                leaf[v, :, j] = 1.0
    leaf = jnp.asarray(leaf)

    def loglik(u):
        pis = jax.nn.softmax(u["pis"], axis=1)            # (C,A)
        rho = jax.nn.softmax(u["rho"])                    # (C,)
        p = jax.nn.softplus(u["p"]) + 1e-3
        s = jax.nn.softplus(u["s"]) + 1e-3
        w = jax.nn.softplus(u["w"]) + 1e-3
        Qc = jnp.stack([gtr_Q(pis[c]) for c in range(C)])
        Qf, pif = field_Q(p, s, w, dist_j, edges_j, n)
        col_ll = []                                       # (C, L)
        for c in range(C):
            arch_c = arch_j[:, c]                         # (n,)
            G = joint_generator(Qc, Qf, arch_c, n)
            Bfull = jnp.zeros((N, n * A, n * A)).at[jnp.arange(N)].set(jnp.eye(n * A))
            Bbr = jax.vmap(lambda t: jexpm(G * t))(tau_j[bn])   # (nbr,nS,nS)
            Bfull = Bfull.at[bn].set(Bbr)
            rd = (pif[:, None] * pis[arch_c]).reshape(-1)       # (nS,)
            col_ll.append(felsenstein(Bfull, rd, leaf))
        col_ll = jnp.stack(col_ll)                        # (C,L)
        mix = jlse(jnp.log(rho)[:, None] + col_ll, axis=0)   # (L,)
        return mix.sum()
    return loglik, (states, dist, arch, pairs, edges, n)


# ---------- self-consistent simulator (per-column joint model) ----------
def simulate(parent, tau, L, C, pis, p, s, w, rho, seed=1):
    rng = np.random.default_rng(seed)
    states, dist, arch, pairs, edges = perm_states(C)
    n = len(states)
    from tkfdp.permfield import build_field
    _, Qf_np, pif_np, _ = build_field(C, p, s, w)
    Qc_np = [np.asarray(gtr_Q(jnp.asarray(pis[c]))) for c in range(C)]
    order = _postorder(parent)[::-1]                      # preorder
    root = order[0]
    nl = sum(len(_children(parent, v)) == 0 for v in range(len(parent)))
    # per-class joint branch ops
    from scipy.linalg import expm as sexpm
    msa = np.full((nl, L), 20, np.int8)
    cls = rng.choice(C, size=L, p=rho)
    for c in np.unique(cls):
        arch_c = arch[:, c]
        # joint generator (numpy)
        G = np.zeros((n * A, n * A))
        for i in range(n):
            G[i * A:(i + 1) * A, i * A:(i + 1) * A] += Qc_np[arch_c[i]]
        G += np.kron(Qf_np - np.diag(np.diag(Qf_np)), np.eye(A))
        np.fill_diagonal(G, 0.0); G[np.diag_indices(n * A)] = -G.sum(1)
        Bc = {v: sexpm(G * tau[v]) for v in range(len(parent)) if parent[v] >= 0}
        rd = np.concatenate([pif_np[i] * pis[arch_c[i]] for i in range(n)])
        cols = np.where(cls == c)[0]
        for j in cols:
            st = np.zeros(len(parent), int)
            st[root] = rng.choice(n * A, p=rd)
            for v in order[1:]:
                st[v] = rng.choice(n * A, p=Bc[v][st[parent[v]]])
            for lf in range(nl):
                msa[lf, j] = st[lf] % A
    return msa, cls


# ---------- fit driver ----------
def init_params(C, n_edges_pairs, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "pis": jnp.asarray(np.log(PI0)[None, :] + 0.1 * rng.standard_normal((C, A))),
        "rho": jnp.zeros(C),
        "p": jnp.zeros(C),
        "s": jnp.zeros(max(C - 1, 1)),
        "w": jnp.zeros(n_edges_pairs),
    }


def fit(loglik, C, n_pairs, steps=400, lr=0.05, seed=0, verbose=True):
    u = init_params(C, n_pairs, seed)
    opt = optax.adam(lr); st = opt.init(u)
    vg = jax.jit(jax.value_and_grad(lambda z: -loglik(z)))
    hist = []
    for it in range(steps):
        nll, g = vg(u)
        upd, st = opt.update(g, st); u = optax.apply_updates(u, upd)
        hist.append(float(-nll))
        if verbose and (it < 3 or it % 50 == 0 or it == steps - 1):
            rho = np.asarray(jax.nn.softmax(u["rho"]))
            p = np.asarray(jax.nn.softplus(u["p"]) + 1e-3)
            print(f"  it {it:4d}  LL={-nll:12.2f}  rho={np.round(rho,3)}  "
                  f"p~={np.round(p/p.sum(),3)}", flush=True)
    return u, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--family", default="PF00013")
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--max-leaves", type=int, default=48)
    args = ap.parse_args()

    d = np.load(f"data/pfam_processed_clv_top1000_thin128/{args.family}.npz",
                allow_pickle=True)
    parent = d["parent"].astype(int); tau = d["tau"].astype(float)
    # subsample leaves to keep the joint state space cheap for validation
    parent, tau, keep_leaves = _prune_leaves(parent, tau, args.max_leaves)
    C = args.C
    _, _, _, pairs, _ = perm_states(C)
    n_pairs = len(pairs)

    if args.synthetic:
        rng = np.random.default_rng(0)
        pis_t = np.clip(PI0[None, :] * (1 + 0.7 * rng.standard_normal((C, A))), 1e-3, None)
        pis_t /= pis_t.sum(1, keepdims=True)
        p_t = np.array([2.5] + [0.6] * (C - 1))
        rho_t = rng.dirichlet(np.ones(C) * 2.0)
        msa, cls = simulate(parent, tau, args.L, C, pis_t, p_t,
                            np.ones(max(C - 1, 1)), np.ones(n_pairs), rho_t, seed=1)
        print(f"# synthetic C={C}: {msa.shape[0]} leaves x {args.L} cols  "
              f"true rho={np.round(rho_t,3)}  true p={np.round(p_t/p_t.sum(),3)}", flush=True)
        t0 = time.time()
        loglik, meta = build_loglik(parent, tau, msa, C)
        u, hist = fit(loglik, C, n_pairs, steps=args.steps)
        rho = np.asarray(jax.nn.softmax(u["rho"]))
        p = np.array(jax.nn.softplus(u["p"]) + 1e-3); p = p / p.sum()
        pis = np.asarray(jax.nn.softmax(u["pis"], axis=1))
        mono = bool(np.all(np.diff(hist[5:]) > -1.0))
        # match recovered archetypes to truth by profile L1 (label permutation)
        cost = np.abs(pis[:, None, :] - pis_t[None, :, :]).sum(-1)
        match = _hungarian(cost)
        prof_err = float(np.mean([cost[i, match[i]] for i in range(C)]))
        print(f"# fit {time.time()-t0:.1f}s  monotone(LL)={mono}")
        print(f"# rho recovered {np.round(rho,3)}  vs true {np.round(rho_t,3)}")
        print(f"# p   recovered {np.round(p,3)}  vs true {np.round(p_t/p_t.sum(),3)}")
        print(f"# mean archetype-profile L1 error (label-matched) = {prof_err:.3f}")
    else:
        msa = _leaf_msa(d, keep_leaves)
        print(f"# fit C={C} on {args.family}: {msa.shape[0]} leaves x {msa.shape[1]} cols",
              flush=True)
        loglik, meta = build_loglik(parent, tau, msa, C)
        u, hist = fit(loglik, C, n_pairs, steps=args.steps)
        print(f"# final LL={hist[-1]:.2f}")


def _prune_leaves(parent, tau, max_leaves):
    N = len(parent)
    leaves = [v for v in range(N) if len(_children(parent, v)) == 0]
    if len(leaves) <= max_leaves:
        return parent, tau, np.array(leaves)
    rng = np.random.default_rng(0)
    keep = set(rng.choice(leaves, size=max_leaves, replace=False).tolist())
    # iteratively drop unkept leaves and collapse degree-2 internal nodes
    par = parent.copy(); t = tau.copy(); alive = np.ones(N, bool)
    changed = True
    while changed:
        changed = False
        for v in range(N):
            if not alive[v]:
                continue
            kids = [u for u in range(N) if alive[u] and par[u] == v]
            if len(kids) == 0 and v not in keep and par[v] >= 0:
                alive[v] = False; changed = True
            elif len(kids) == 1 and par[v] >= 0:          # collapse
                u = kids[0]; t[u] = t[u] + t[v]; par[u] = par[v]
                alive[v] = False; changed = True
    # reindex
    old = [v for v in range(N) if alive[v]]
    new_leaves = [v for v in old if len([u for u in old if par[u] == v]) == 0]
    remap = {}
    # leaves first
    for v in new_leaves:
        remap[v] = len(remap)
    for v in old:
        if v not in remap:
            remap[v] = len(remap)
    M = len(old); npar = np.full(M, -1, int); ntau = np.zeros(M)
    for v in old:
        npar[remap[v]] = remap[par[v]] if par[v] in remap else -1
        ntau[remap[v]] = t[v]
    keep_leaves = np.array([v for v in new_leaves])       # original ids preserved order
    # map kept original leaf ids -> position
    orig_leaf_ids = [v for v in new_leaves]
    return npar, ntau, np.array(orig_leaf_ids)


def _leaf_msa(d, keep_leaves):
    msa = d["leaf_msa"]
    # keep_leaves holds ORIGINAL node ids that are leaves; leaf_msa is indexed by
    # leaf order 0..n_leaves-1 which equals original leaf node ids in CLV format
    ids = [v for v in keep_leaves if v < msa.shape[0]]
    return msa[np.array(ids)].astype(np.int8)


def _hungarian(cost):
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return {int(r[i]): int(c[i]) for i in range(len(r))}
    except Exception:
        return {i: i for i in range(cost.shape[0])}


if __name__ == "__main__":
    main()
