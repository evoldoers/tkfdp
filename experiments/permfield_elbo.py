#!/usr/bin/env python3
"""permfield mood-light model: shared-field product-of-trees ELBO with exact HR.

Structured mean-field variational fit (docs/permutation_field_model.md steps 1-4;
paper 2b Sec. 5-6): one family-level field posterior q_Theta shared across all
columns (EXACT belief propagation over the C!-state field tree), each column a
20-state residue factor q_j (inside-outside) under a field-averaged branch
generator, class latent c_j summed with responsibilities gamma_j(c). M-steps use
EXACT Holmes-Rubin bridge statistics (tkfdp.permfield.hr).

  q = q_Theta(field, C!-state tree-Markov)  x  prod_j q_j(residue, A-state tree-Markov)

Coordinate ascent:
  1. columns | field : per class c, per-branch averaged generator
        Qbar_{c,u} = sum_theta b_u(theta) Q^{theta(c)} = sum_a beta_{c,u}(a) Q^a;
     inside-outside -> per-column L_j(c) and gamma-weighted edge marginals;
     gamma_j(c) prop rho_c L_j(c).
  2. field | columns : node potentials phi_u(theta) = column expected-log-lik on
     u's out-branches under archetype theta(.); EXACT BP over the C!-state tree.
  3. M-steps (exact HR): archetype pi^a from HR (N^a,T^a) flux gradient; field
     p,s,w from HR field bridge stats (flux gradient); rho_c = mean_j gamma_j(c).

Contrast with experiments/permfield_fit.py (per-column COMPOSITE, independent
field per column, gradient fit): this ties ONE field trajectory across a family's
columns. C<=4: field BP exact; C>4: field factor would be approximated.

Archetype generators Q^a = S_LG08 (x) pi^a and the field generator Q_field are
left UNNORMALISED (fixed branch lengths tau carry the timescale); pi_field is the
stationary p[dist]/Z. --synthetic simulates ONE shared field trajectory per family
then columns from it, and checks the objective climbs + parameter recovery.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import numpy as np
from scipy.linalg import expm as sexpm

sys.path.insert(0, "src")
from tkfdp.lg08 import S_LG08, PI_LG08                      # noqa: E402
from tkfdp.permfield import build_field, transposition_distance  # noqa: E402
from tkfdp.permfield.hr import bridge, eig_rev              # noqa: E402

A = 20
S = np.asarray(S_LG08, float)
PI0 = np.asarray(PI_LG08, float)


# ---------- bookkeeping ----------
def perm_states(C):
    states = list(itertools.permutations(range(C)))
    dist = np.array([transposition_distance(t) for t in states])
    arch = np.array([list(t) for t in states])              # arch[i,c]=theta_i(c)
    pairs = [(a, b) for a in range(C) for b in range(a + 1, C)]
    return states, dist, arch, pairs


def ewens_p(C, alpha):
    """Field stationary as the Ewens distribution over S_C: P(theta) prop
    alpha^{#cycles(theta)}. Since the Cayley distance d(theta) = C - #cycles, this
    is exactly the pi_field(theta) = p[d]/Z form with weights-by-distance
    p_d prop alpha^{-d} (concentration alpha; alpha>1 concentrates on the identity
    = the canonical class k -> archetype k map, which is what breaks the
    mean-field archetype-relabelling symmetry). One parameter, replacing the free
    p_0..p_{C-1}; matches the project's Ewens/DP (alpha_z) backbone."""
    d = np.arange(C, dtype=float)
    return np.power(float(alpha), -d)


def gtr_Q(pi, normalize=False):
    Q = S * pi[None, :]
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(A)] = -Q.sum(1)
    if normalize:
        Q = Q / (-(pi * np.diag(Q)).sum())
    return Q


def children(parent):
    ch = [[] for _ in range(len(parent))]
    for v in range(len(parent)):
        if parent[v] >= 0:
            ch[parent[v]].append(v)
    return ch


def orders(parent):
    root = int(np.where(parent < 0)[0][0])
    ch = children(parent)
    pre, stack = [], [root]
    while stack:
        v = stack.pop(); pre.append(v); stack.extend(ch[v])
    return pre, pre[::-1], root, ch


# ---------- field factor: exact BP over the C!-state tree ----------
def field_bp(parent, Pf, pif, phi, pre, post, root, ch):
    """Sum-product. Pf[v]:(nS,nS) parent->child on branch v; pif:(nS,) root prior;
    phi:(N,nS) LOG node potentials. Returns b (N,nS), xi {v:(nS,nS)}, log Z."""
    N = len(parent); nS = len(pif)
    ephi = np.exp(phi - phi.max(1, keepdims=True))
    lsc = phi.max(1)
    up = np.zeros((N, nS)); logZup = np.zeros(N)
    for v in post:
        m = ephi[v].copy()
        for w in ch[v]:
            m = m * (Pf[w] @ up[w]); logZup[v] += logZup[w]
        s = m.sum(); up[v] = m / max(s, 1e-300)
        logZup[v] += np.log(max(s, 1e-300)) + lsc[v]
    rootvec = pif * up[root]
    logZ = np.log(max(rootvec.sum(), 1e-300)) + logZup[root]
    b = np.zeros((N, nS)); xi = {}
    b[root] = rootvec / rootvec.sum()
    down = np.zeros((N, nS)); down[root] = pif.copy()
    for v in pre:
        if v == root:
            continue
        u = parent[v]
        sib = ephi[u] * down[u]
        for w in ch[u]:
            if w != v:
                sib = sib * (Pf[w] @ up[w])
        sib = sib / max(sib.sum(), 1e-300)
        joint = sib[:, None] * Pf[v] * up[v][None, :]
        joint = joint / max(joint.sum(), 1e-300)
        xi[v] = joint; b[v] = joint.sum(0)
        d = sib @ Pf[v]; down[v] = d / max(d.sum(), 1e-300)
    return b, xi, logZ


# ---------- column E-step: inside-outside under field-averaged generators ----------
def column_estep(parent, tau, msa, C, arch, Qc, pis, rho, b_field,
                 pre, post, root, ch, nl):
    N = len(parent); L = msa.shape[1]; nS = b_field.shape[1]
    branches = [v for v in range(N) if parent[v] >= 0]
    # beta_{c,u}(a) = P(archetype of class c under field at u = a)
    beta = np.zeros((C, N, C))
    for c in range(C):
        for a in range(C):
            beta[c, :, a] = b_field[:, arch[:, c] == a].sum(1)
    Peff, logPa = {}, {}
    for a in range(C):
        for v in branches:
            logPa[(a, v)] = np.log(np.clip(sexpm(Qc[a] * tau[v]), 1e-300, None))
    for c in range(C):
        for v in branches:
            Qb = sum(beta[c, parent[v], a] * Qc[a] for a in range(C))
            Peff[(c, v)] = sexpm(Qb * tau[v])
    obs = msa[:nl]

    # --- inside pass per class -> col_ll, gamma ---
    inside, col_ll = {}, np.zeros((L, C))
    for c in range(C):
        up = np.zeros((N, A, L)); lsc = np.zeros(L)
        for v in post:
            if v < nl:
                m = np.zeros((A, L)); r = obs[v]; val = r < A
                m[:, ~val] = 1.0; m[r[val], np.where(val)[0]] = 1.0
                up[v] = m
            else:
                m = np.ones((A, L))
                for w in ch[v]:
                    m = m * (Peff[(c, w)] @ up[w])
                s = m.sum(0); up[v] = m / np.maximum(s, 1e-300)
                lsc += np.log(np.maximum(s, 1e-300))
        rp = sum(b_field[root, i] * pis[arch[i, c]] for i in range(nS))
        col_ll[:, c] = lsc + np.log(np.maximum((rp[:, None] * up[root]).sum(0), 1e-300))
        inside[c] = up
    lr = np.log(rho)[None, :] + col_ll
    mx = lr.max(1, keepdims=True)
    gamma = np.exp(lr - mx); gamma /= gamma.sum(1, keepdims=True)
    obj = float(np.sum(mx[:, 0] + np.log(np.exp(lr - mx).sum(1))))

    # --- outside pass per class -> gamma-weighted edge marginals ---
    EdgeAcc = {}          # (c,v) -> (A,A) sum_j gamma_j(c) q_j(x_u,x_v)
    RootAcc = np.zeros((C, A))
    for c in range(C):
        up = inside[c]
        down = np.zeros((N, A, L)); down[root] = 1.0
        # root residue prior message: pi^{theta(c)} averaged over field root marginal
        rp = sum(b_field[root, i] * pis[arch[i, c]] for i in range(nS))
        down[root] = np.repeat(rp[:, None], L, 1)
        RootAcc[c] = (gamma[:, c][None, :] * (down[root] * up[root]) /
                      np.maximum((down[root] * up[root]).sum(0), 1e-300)).sum(1)
        for v in pre:
            if v == root:
                continue
            u = parent[v]
            sib = down[u].copy()
            for w in ch[u]:
                if w != v:
                    sib = sib * (Peff[(c, w)] @ up[w])
            # edge marginal on branch v: sib(x_u) * Peff[x_u,x_v] * up_v(x_v)
            Pe = Peff[(c, v)]
            # (A,A,L): x_u,x_v,col
            ev = sib[:, None, :] * Pe[:, :, None] * up[v][None, :, :]
            Z = np.maximum(ev.sum((0, 1)), 1e-300)
            ev = ev / Z[None, None, :]
            EdgeAcc[(c, v)] = (ev * gamma[:, c][None, None, :]).sum(2)
            down[v] = (Pe.T @ sib)
            down[v] = down[v] / np.maximum(down[v].sum(0), 1e-300)
    return dict(gamma=gamma, col_ll=col_ll, obj=obj, beta=beta,
                EdgeAcc=EdgeAcc, RootAcc=RootAcc, logPa=logPa, branches=branches)


# ---------- field node potentials from column evidence ----------
def field_potentials(parent, C, arch, es, pis, N, nS, ch, root):
    logpi = np.log(np.clip(pis, 1e-300, None))              # (C,A)
    # EL[(c,v,a)] = <EdgeAcc[(c,v)], logPa[(a,v)]>
    EL = {}
    for (c, v), E in es["EdgeAcc"].items():
        for a in range(C):
            EL[(c, v, a)] = float((E * es["logPa"][(a, v)]).sum())
    phi = np.zeros((N, nS))
    for u in range(N):
        for v in ch[u]:
            for c in range(C):
                for i in range(nS):
                    phi[u, i] += EL[(c, v, arch[i, c])]
    for i in range(nS):                                     # root residue prior
        for c in range(C):
            phi[root, i] += float((es["RootAcc"][c] * logpi[arch[i, c]]).sum())
    return phi


# ---------- HR M-steps: per-cluster accumulate + global solve ----------
def accum_arch(C, arch, es, tau, Qc, eig_a, parent, Na, Ta, roota):
    """Add this cluster's HR archetype bridge stats (dwell Ta[a], usage Na[a],
    root counts roota[a]) in place. Sums over the cluster's members, classes,
    branches, weighted by the field-archetype incidence beta and gamma."""
    for a in range(C):
        for c in range(C):
            for v in es["branches"]:
                bw = es["beta"][c, parent[v], a]
                if bw < 1e-9:
                    continue
                E = es["EdgeAcc"].get((c, v))
                if E is None:
                    continue
                T, Nxy, _ = bridge(Qc[a], None, tau[v], E, want_N=True, eig=eig_a[a])
                Ta[a] += bw * T; Na[a] += bw * Nxy
        root = int(np.where(np.asarray(parent) < 0)[0][0])
        for c in range(C):                                   # field-POSTERIOR root incidence
            roota[a] += es["RootAcc"][c] * es["beta"][c, root, a]


def solve_arch(C, Na, Ta, roota, pis, steps=40, lr=0.2):
    """Global archetype M-step: maximise the expected-complete-data LL for pi^a
    (Q^a = S (x) pi^a) given aggregated HR counts, by softmax gradient ascent."""
    z = [np.log(np.clip(pis[a], 1e-6, None)) for a in range(C)]
    for _ in range(steps):
        for a in range(C):
            p = np.exp(z[a] - z[a].max()); p = p / p.sum()
            num = roota[a] + Na[a].sum(0)                    # root + incoming usage
            g_p = num / np.maximum(p, 1e-12) - (Ta[a][:, None] * S).sum(0)
            g_z = p * (g_p - (p * g_p).sum())
            z[a] = z[a] + lr * g_z / max(num.sum(), 1.0)
    return np.array([np.exp(z[a] - z[a].max()) / np.exp(z[a] - z[a].max()).sum()
                     for a in range(C)])


def accum_field(C, tau, xi, b_field, Qf, pif, eig_f, root, Wf, Uf, rootc):
    """Add this cluster's HR field bridge stats (dwell Wf, usage Uf) and its root
    stationary counts (rootc) in place."""
    for v, E in xi.items():
        T, Nn, _ = bridge(Qf, pif, tau[v], E, want_N=True, eig=eig_f)
        Wf += T; Uf += Nn
    rootc += b_field[root]


def solve_field(C, Wf, Uf, rootc, alpha, s, w, steps=25, lr=0.15, damp=0.5,
                learn_alpha=False):
    """Global field M-step: fit exchangeabilities s,w (and optionally the Ewens
    concentration alpha) to the aggregated HR field stats. Stationary is the Ewens
    distribution pi_field(theta) prop alpha^{-d} (geometric in Cayley distance);
    alpha is a PRIOR held FIXED by default (learning runs away -- the weakly-
    identified field-activity dimension; --learn-alpha overrides)."""
    occ = rootc                          # log-pi coeff is root occupancy only; the
    #  incoming-usage log-pi term is already carried by (Uf*log Q) since Q=s*w*pi[j].
    #  (Wf+rootc double-counted a spurious dwell*log-pi term -- wrong under learn-alpha.)
    la = np.array([np.log(max(alpha, 1e-3))])
    ls = np.log(np.clip(s, 1e-6, None)); lw = np.log(np.clip(w, 1e-6, None))

    def ll(la, ls, lw):
        p = ewens_p(C, np.exp(la[0]))
        _, Qf, pif, _ = build_field(C, p, np.exp(ls), np.exp(lw), normalize_rate=False)
        pif = np.clip(pif, 1e-12, None)
        lQ = np.log(np.clip(Qf, 1e-300, None)); np.fill_diagonal(lQ, 0.0)
        return (occ * np.log(pif)).sum() + (Uf * lQ).sum() \
            - (Wf * (Qf.sum(1) - np.diag(Qf))).sum()

    def grad(vec):
        base = ll(*vec); flat = np.concatenate(vec); sizes = [len(x) for x in vec]
        eps = 1e-4; gflat = np.zeros_like(flat)
        for k in range(len(flat)):
            f2 = flat.copy(); f2[k] += eps
            gflat[k] = (ll(*np.split(f2, np.cumsum(sizes)[:-1])) - base) / eps
        return np.split(gflat, np.cumsum(sizes)[:-1])

    vec = [la.copy(), ls.copy(), lw.copy()]
    gn = max(len(Wf), 1)
    for _ in range(steps):
        g = grad(vec)
        if not learn_alpha:
            g[0] = np.zeros_like(g[0])
        vec = [np.clip(vec[i] + lr * g[i] / gn, -8, 8) for i in range(3)]
    return (damp * alpha + (1 - damp) * float(np.exp(vec[0][0])),
            damp * s + (1 - damp) * np.exp(vec[1]),
            damp * w + (1 - damp) * np.exp(vec[2]))


# ---------- clustered simulator ----------
def simulate(parent, tau, C, pis, p, s, w, rho, clusters, seed=1):
    """Simulate a clustered corpus: `clusters` is a list of cluster sizes m_k.
    Each cluster gets ONE shared field trajectory; its m members each draw a class
    ~ rho and evolve under that shared field (so members of a cluster are coupled
    only through the field). Returns (msa (nl, total_cols), partition)."""
    rng = np.random.default_rng(seed)
    states, dist, arch, pairs = perm_states(C)
    nS = len(states)
    _, Qf, pif, _ = build_field(C, p, s, w, normalize_rate=False)
    Pf = {v: sexpm(Qf * tau[v]) for v in range(len(parent)) if parent[v] >= 0}
    Qc = [gtr_Q(pis[c]) for c in range(C)]
    Pc = {(a, v): sexpm(Qc[a] * tau[v]) for a in range(C)
          for v in range(len(parent)) if parent[v] >= 0}
    pre, post, root, ch = orders(parent)
    nl = sum(len(ch[v]) == 0 for v in range(len(parent)))
    total = int(sum(clusters))
    msa = np.full((nl, total), 20, np.int8)
    partition, col = [], 0
    for m in clusters:
        th = np.zeros(len(parent), int); th[root] = rng.choice(nS, p=pif)
        for v in pre[1:]:
            th[v] = rng.choice(nS, p=Pf[v][th[parent[v]]])   # one field for the cluster
        cols_here = []
        for _ in range(m):
            c = rng.choice(C, p=rho); x = np.zeros(len(parent), int)
            x[root] = rng.choice(A, p=pis[arch[th[root], c]])
            for v in pre[1:]:
                a = arch[th[parent[v]], c]
                x[v] = rng.choice(A, p=Pc[(a, v)][x[parent[v]]])
            msa[:, col] = x[:nl]; cols_here.append(col); col += 1
        partition.append(np.array(cols_here))
    return msa, partition


def warn_no_structure(L):
    """Loud warning: this model is meant to be run WITH structure supervision."""
    bar = "!" * 74
    print(f"\n{bar}\n"
          "!! permfield: NO cluster structure supplied (partition=None).\n"
          "!! This model is DESIGNED to run with STRUCTURE SUPERVISION -- m=2\n"
          "!! contact-pair clusters (e.g. experiments/build_pdb_partition.py).\n"
          f"!! Falling back to {L} SINGLETONS (each column its own field).\n"
          "!! Singletons are the composite baseline; they CANNOT discover\n"
          "!! coupling and rho/archetypes will stay ~uninformative.\n"
          f"{bar}\n", file=sys.stderr, flush=True)


# ---------- fit: product of (m+1) trees per cluster ----------
def fit(parent, tau, msa, C, partition=None, n_iter=25, seed=0, alpha0=20.0,
        learn_alpha=False, freeze_field=False, verbose=True):
    """Product-of-trees ELBO over a PARTITION of columns into clusters. Each
    cluster of size m carries (m+1) tree factors: one field tree q_Theta (shared
    by the cluster's m members) plus m residue trees. Members are factored
    (coupled only through their shared field), so the cost is linear in m -- no
    A^m. Global params {pi^a, alpha, s, w, rho}; per-cluster field posteriors.
    partition=None -> SINGLETONS (each column its own field) with a loud warning:
    the model is meant to be run WITH structure supervision (contact-pair m=2
    clusters); singletons are the composite baseline and cannot discover coupling."""
    rng = np.random.default_rng(seed)
    N = len(parent); nl, L = msa.shape
    states, dist, arch, pairs = perm_states(C)
    nS = len(states)
    pre, post, root, ch = orders(parent)
    branches = [v for v in range(N) if parent[v] >= 0]
    if partition is None:
        warn_no_structure(L)
        partition = [np.array([j]) for j in range(L)]       # singletons, NOT one giant cluster
    partition = [np.asarray(cl, int) for cl in partition]
    pis = np.clip(PI0[None, :] * (1 + 0.1 * rng.standard_normal((C, A))), 1e-4, None)
    pis /= pis.sum(1, keepdims=True)
    alpha = float(alpha0)
    s = np.ones(max(C - 1, 1)); w = np.ones(len(pairs))
    rho = np.ones(C) / C
    # each cluster's field marginal initialised to the EWENS prior (geometric in
    # Cayley distance) -- identity-centred, breaking the archetype-relabel symmetry.
    _, _, pif0, _ = build_field(C, ewens_p(C, alpha), s, w, normalize_rate=False)
    if freeze_field:
        # DIAGNOSTIC: pin the field to the identity permutation (state 0) with no
        # switching -> class c deterministically uses archetype c, i.e. a plain
        # C-class phylogenetic profile mixture. Tests whether the substitution-class
        # learning differentiates at all, isolated from the field dynamics.
        onehot = np.zeros(nS); onehot[0] = 1.0
        b_fields = [np.tile(onehot, (N, 1)) for _ in partition]
    else:
        b_fields = [np.tile(pif0, (N, 1)) for _ in partition]
    hist = []
    for it in range(n_iter):
        Qc = [gtr_Q(pis[a]) for a in range(C)]
        eig_a = [eig_rev(Qc[a], pis[a]) for a in range(C)]
        _, Qf, pif, _ = build_field(C, ewens_p(C, alpha), s, w, normalize_rate=False)
        eig_f = eig_rev(Qf, np.clip(pif, 1e-10, None))
        Pf = {v: np.clip(sexpm(Qf * tau[v]), 1e-300, None) for v in branches}
        Na = [np.zeros((A, A)) for _ in range(C)]; Ta = [np.zeros(A) for _ in range(C)]
        roota = [np.zeros(A) for _ in range(C)]
        Wf = np.zeros(nS); Uf = np.zeros((nS, nS)); rootc = np.zeros(nS)
        gsum = np.zeros(C); nmemb = 0; obj = 0.0
        for ci, cl in enumerate(partition):
            sub = msa[:, cl]
            es = column_estep(parent, tau, sub, C, arch, Qc, pis, rho, b_fields[ci],
                              pre, post, root, ch, nl)
            if freeze_field:
                obj += es["obj"]                            # field pinned; no field factor
            else:
                phi = field_potentials(parent, C, arch, es, pis, N, nS, ch, root)
                b_fields[ci], xi, logZf = field_bp(parent, Pf, pif, phi, pre, post, root, ch)
                obj += es["obj"] + logZf
                accum_field(C, tau, xi, b_fields[ci], Qf, pif, eig_f, root, Wf, Uf, rootc)
            accum_arch(C, arch, es, tau, Qc, eig_a, parent, Na, Ta, roota)
            gsum += es["gamma"].sum(0); nmemb += len(cl)
        hist.append(obj)
        pis = solve_arch(C, Na, Ta, roota, pis)
        if not freeze_field:
            alpha, s, w = solve_field(C, Wf, Uf, rootc, alpha, s, w, learn_alpha=learn_alpha)
        rho = np.clip(gsum / max(nmemb, 1), 1e-6, None); rho /= rho.sum()
        if verbose and (it < 3 or it % 5 == 0 or it == n_iter - 1):
            print(f"  it {it:3d}  obj={obj:12.2f}  rho={np.round(rho,3)}  "
                  f"alpha={alpha:6.3f}  pfield={np.round(pif,3)}", flush=True)
    return dict(pis=pis, alpha=alpha, s=s, w=w, rho=rho, b_fields=b_fields,
                hist=hist, partition=partition)


def _prune_leaves(parent, tau, max_leaves):
    from experiments.permfield_fit import _prune_leaves as pl
    return pl(parent, tau, max_leaves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--family", default="PF00013")
    ap.add_argument("--max-leaves", type=int, default=24)
    ap.add_argument("--L", type=int, default=120)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=20.0,
                    help="Ewens field concentration (prior, fixed unless --learn-alpha)")
    ap.add_argument("--learn-alpha", action="store_true",
                    help="fit the Ewens concentration too (runs away; weakly identified)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--m", type=int, default=1, help="cluster size m for --synthetic")
    ap.add_argument("--n-clusters", type=int, default=120,
                    help="number of size-m clusters for --synthetic")
    args = ap.parse_args()
    d = np.load(f"data/pfam_processed_clv_top1000_thin128/{args.family}.npz",
                allow_pickle=True)
    parent, tau, keep = _prune_leaves(d["parent"].astype(int),
                                      d["tau"].astype(float), args.max_leaves)
    C = args.C
    if args.synthetic:
        rng = np.random.default_rng(0)
        _, _, arch, pairs = perm_states(C)
        pis_t = np.clip(PI0[None, :] * (1 + 0.7 * rng.standard_normal((C, A))), 1e-3, None)
        pis_t /= pis_t.sum(1, keepdims=True)
        alpha_t = 3.0                                       # Ewens true field
        p_t = ewens_p(C, alpha_t)
        rho_t = rng.dirichlet(np.ones(C) * 2.0)
        clusters = [args.m] * args.n_clusters               # K clusters of size m
        msa, partition = simulate(parent, tau, C, pis_t, p_t,
                                  np.ones(max(C - 1, 1)), np.ones(len(pairs)),
                                  rho_t, clusters, seed=1)
        print(f"# synthetic(Ewens alpha={alpha_t}) C={C}, {args.n_clusters} clusters "
              f"of size m={args.m} ({msa.shape[1]} cols, {msa.shape[0]} leaves)  "
              f"true rho={np.round(rho_t,3)}", flush=True)
        t0 = time.time()
        res = fit(parent, tau, msa, C, partition=partition, n_iter=args.iters,
                  alpha0=args.alpha, learn_alpha=args.learn_alpha)
        mono = bool(np.all(np.diff(res["hist"][3:]) > -1.0))
        cost = np.abs(res["pis"][:, None, :] - pis_t[None, :, :]).sum(-1)
        from scipy.optimize import linear_sum_assignment
        r, cix = linear_sum_assignment(cost)
        perr = float(np.mean([cost[r[i], cix[i]] for i in range(C)]))
        print(f"# fit {time.time()-t0:.1f}s  monotone(obj)={mono}")
        print(f"# rho {np.round(res['rho'],3)} vs true {np.round(rho_t[cix],3)} "
              f"(gauge-aligned)  L1={np.abs(res['rho']-rho_t[cix]).sum():.3f}")
        print(f"# alpha {res['alpha']:.3f} vs true {alpha_t}  (Ewens concentration)")
        print(f"# archetype-profile L1 (matched) = {perr:.3f}")
    else:
        from experiments.permfield_fit import _leaf_msa
        msa = _leaf_msa(d, keep)
        print(f"# ELBO fit C={C} on {args.family}: {msa.shape[0]} leaves x {msa.shape[1]} cols",
              flush=True)
        res = fit(parent, tau, msa, C, n_iter=args.iters, alpha0=args.alpha,
                  learn_alpha=args.learn_alpha)
        print(f"# final obj={res['hist'][-1]:.2f}  monotone={bool(np.all(np.diff(res['hist'][3:])>-1.0))}")


if __name__ == "__main__":
    main()
