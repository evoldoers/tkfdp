"""Numerical verification: factored cap-2 cluster Holmes-Rubin == dense 800-state HR.

See analysis/cluster_ctmc_simplification.md section 4 for the claim.

The cap-2 cluster CTMC has state (theta, x1, x2), theta in {0,1}, residues in
A=20 letters -> L*A*A = 800 states. Its generator (exact_cap2._compound_generator_pair)
is D (per-theta Kronecker sum of two GTR residue generators) + a rank-1 field-jump
renewal (resample theta ~ rho and BOTH residues from the new archetype stationary).

We check that the EXACT endpoint-conditioned Holmes-Rubin sufficient statistics of
the dense 800-state CTMC equal a FACTORED assembly:
  * per-archetype expected substitution counts N^k[a,b] and dwell T^k[a],
  * the expected field-jump count,
computed from a 2-state field posterior x per-residue 20-state Holmes-Rubin.

Ground truth: dense reversible tree HR on the 800-state generator (Van Loan /
Hobolth-Jensen bridge integral, up-down over a small tree). The compound generator
is reversible w.r.t. p(t,x,y) = rho[t] pi^{a1(t)}[x] pi^{a2(t)}[y] (detailed balance
holds for residue moves because each GTR is reversible, and for field jumps because
the renewal target factorises as rho x pi x pi) -- so a real symmetric
eigendecomposition suffices, exactly as analysis/scripts/tree_hr_charge.py does for
the single 20-state chain.

Pure CPU / numpy.  Run:  python analysis/scripts/check_cluster_hr.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tkfdp.lg08 import get_lg08
from tkfdp.coupling.dynfield.phylo_elbo.exact_cap2 import (
    gtr_Q, field_kernels, _compound_generator_pair,
    cherry_ll_bruteforce, cherry_ll_exact, exact_pair_ll_tree,
)

A = 20


# ------------------------------------------------------------------ model setup
def make_archetypes():
    """A few real Le-Gascuel C10 profiles (charge-crossing) + LG08 background.

    Returns pi_arch (K_a, A) with archetype 0 = LG08 background, and a couple of
    charge-shifted archetypes so a1 != a2 genuinely uses different profiles.
    """
    S, pi_bg = get_lg08()
    try:
        sys.path.insert(0, os.path.expanduser("~/tkf-mixdom/python"))
        from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c10
        prof, _, _ = le_gascuel_c10()          # (10, 20) alphabetical
        prof = np.asarray(prof)
        # pick an acid-rich and a base-rich component by net charge
        ACID = [2, 3]           # D, E
        BASE = [8, 14, 6]       # K, R, H
        net = prof[:, BASE].sum(1) - prof[:, ACID].sum(1)
        base_rich = int(np.argmax(net)); acid_rich = int(np.argmin(net))
        pi_arch = np.stack([pi_bg, prof[acid_rich], prof[base_rich], prof[0]])
    except Exception as e:               # pragma: no cover - fallback if profiles absent
        print(f"[warn] C10 profiles unavailable ({e}); using synthetic archetypes")
        acid = pi_bg.copy(); acid[[2, 3]] *= 4.0; acid /= acid.sum()
        base = pi_bg.copy(); base[[8, 14, 6]] *= 4.0; base /= base.sum()
        pi_arch = np.stack([pi_bg, acid, base, pi_bg])
    pi_arch = pi_arch / pi_arch.sum(1, keepdims=True)
    return S, pi_arch


def compound_stationary(a1, a2, pi_arch, rho):
    L = rho.shape[0]
    p = np.zeros(L * A * A)
    for t in range(L):
        for x in range(A):
            for y in range(A):
                p[(t * A + x) * A + y] = rho[t] * pi_arch[a1[t]][x] * pi_arch[a2[t]][y]
    return p


def cidx(t, x, y):
    return (t * A + x) * A + y


# ----------------------------------------------------- reversible-chain HR core
def reversible_eigh(Q, p):
    """Symmetric eigendecomposition of a reversible generator Q (stationary p)."""
    sqrt = np.sqrt(p)
    inv = 1.0 / np.clip(sqrt, 1e-300, None)
    Qs = sqrt[:, None] * Q * inv[None, :]
    Qs = 0.5 * (Qs + Qs.T)
    lam, V = np.linalg.eigh(Qs)
    return lam, V, sqrt, inv


def _I_kl(lam, t):
    e = np.exp(lam * t)
    d = lam[:, None] - lam[None, :]
    safe = np.where(np.abs(d) < 1e-12, 1.0, d)
    off = (e[:, None] - e[None, :]) / safe
    return np.where(np.abs(d) < 1e-12, t * e[:, None], off)


def _Pt(lam, V, sqrt, inv, t):
    e = np.exp(lam * t)
    return inv[:, None] * (V * e[None, :]) @ V.T * sqrt[None, :]


def single_branch_hr(out_vec, in_vec, t, Q, eig=None, p=None):
    """Unnormalised single-branch endpoint-conditioned HR contribution:
      EN[c,d]  = sum_{i,j} out[i] in[j] P(t)[i,j] E[#c->d | i,j]
      dwell[c] = sum_{i,j} out[i] in[j] P(t)[i,j] E[T_c | i,j]
      Z        = sum_{i,j} out[i] in[j] P(t)[i,j]
    (no normalisation; caller divides by Z if a conditional expectation is wanted)."""
    N = Q.shape[0]
    lam, V, sqrt, inv = eig if eig is not None else reversible_eigh(Q, p)
    scaled = Q * (sqrt[:, None] * inv[None, :])
    Ikl = _I_kl(lam, t)
    g = (out_vec * inv) @ V
    h = (in_vec * sqrt) @ V
    M = g[:, None] * Ikl * h[None, :]
    VMV = V @ M @ V.T
    EN = scaled * VMV
    np.fill_diagonal(EN, 0.0)
    dwell = np.diag(VMV).copy()
    Pt = _Pt(lam, V, sqrt, inv, t)
    Z = float(out_vec @ Pt @ in_vec)
    return EN, dwell, Z


def _mc_single_branch(i0, j0, t, Q, n_samp=200000, seed=0):
    """Endpoint-conditioned uniformisation bridge MC (Hobolth-Stone) for a single
    branch conditioned on start=i0, end=j0.  Returns (E[EN], E[dwell]) over the
    bridge (per-transition expected counts and per-state dwell)."""
    rng = np.random.default_rng(seed)
    N = Q.shape[0]
    mu = float(np.max(-np.diag(Q))) * 1.05
    R = np.eye(N) + Q / mu                     # uniformised DTMC
    Pt = expm(Q * t)
    Pij = Pt[i0, j0]
    # n | endpoints ~ Pois(mu t; n) R^n[i0,j0] / Pt[i0,j0]
    nmax = int(mu * t + 12 * np.sqrt(mu * t + 1)) + 8
    # R^n[i0,:] via iteration
    row = np.zeros(N); row[i0] = 1.0
    logpois = -mu * t
    pn = np.zeros(nmax + 1)
    from math import lgamma
    Rn_i0_j0 = np.zeros(nmax + 1)
    cur = row.copy()
    for n in range(nmax + 1):
        Rn_i0_j0[n] = cur[j0]
        cur = cur @ R
    for n in range(nmax + 1):
        lp = -mu * t + n * np.log(mu * t + 1e-300) - lgamma(n + 1)
        pn[n] = np.exp(lp) * Rn_i0_j0[n]
    pn = pn / pn.sum()
    EN = np.zeros((N, N)); dwell = np.zeros(N)
    # precompute R rows for sampling forward-filtered bridge
    for _ in range(n_samp):
        n = rng.choice(nmax + 1, p=pn)
        # sample virtual-jump chain states s0=i0,...,sn=j0 via forward-backward on R
        states = [i0]
        if n > 0:
            # backward messages b_k[state] = R^{n-k}[state, j0]
            b = np.zeros((n + 1, N)); b[n, j0] = 1.0
            for k in range(n - 1, -1, -1):
                b[k] = R @ b[k + 1]
            cur_s = i0
            for k in range(1, n + 1):
                if k == n:
                    states.append(j0); break
                w = R[cur_s] * b[k]
                w = w / w.sum()
                cur_s = rng.choice(N, p=w)
                states.append(cur_s)
        # times of the n virtual jumps: uniform order stats -> dwell in each state
        if n > 0:
            ts = np.sort(rng.uniform(0, t, size=n))
            seg = np.diff(np.concatenate([[0.0], ts, [t]]))
        else:
            seg = np.array([t])
        for k, s in enumerate(states):
            dwell[s] += seg[k]
        for k in range(n):
            a, b_ = states[k], states[k + 1]
            if a != b_:
                EN[a, b_] += 1.0
    return EN / n_samp, dwell / n_samp


def _children(parent):
    kids = defaultdict(list)
    root = None
    for v, u in enumerate(parent):
        if u < 0:
            root = v
        else:
            kids[u].append(v)
    return kids, root


def tree_hr(parent, tau, leaf_msgs, Q, p, eig=None):
    """Endpoint-conditioned Holmes-Rubin on a reversible CTMC over a tree.

    parent: (n,) int, parent[root] < 0.  tau: (n,) branch length above each node.
    leaf_msgs: dict leaf -> (N,) evidence vector (indicator for observed states).
    Q, p: (N,N) reversible generator and (N,) stationary (root prior).

    Returns (EN, dwell, logZ):
      EN[i,j]  = expected number of i->j transitions over the whole tree,
      dwell[i] = expected total time in state i over the whole tree,
      logZ     = tree log-likelihood.
    Per-branch normalisation makes any per-node message scaling cancel (the
    standard up-down trick, cf. tree_hr_charge.py).
    """
    N = Q.shape[0]
    lam, V, sqrt, inv = eig if eig is not None else reversible_eigh(Q, p)
    kids, root = _children(parent)
    n = len(parent)
    scaled = Q * (sqrt[:, None] * inv[None, :])   # Q_ij sqrt_i inv_j for contrib

    # inside messages (normalised per node), track log-scale for logZ
    inside = {}
    logscale = {}
    Pc = {}
    order = []
    stack = [root]
    while stack:
        v = stack.pop()
        order.append(v)
        for c in kids[v]:
            stack.append(c)
    for v in reversed(order):
        if not kids[v]:
            m = np.asarray(leaf_msgs[v], float).copy()
            ls = 0.0
        else:
            m = np.ones(N)
            ls = 0.0
            for c in kids[v]:
                Pc[c] = _Pt(lam, V, sqrt, inv, float(tau[c]))
                m = m * (Pc[c] @ inside[c])
                ls += logscale[c]
        s = m.sum()
        if s <= 0:
            s = 1.0
        inside[v] = m / s
        logscale[v] = ls + np.log(s)
    logZ = float(np.log(p @ inside[root]) + logscale[root])

    # outside messages: O[root] = p ; propagate down
    EN = np.zeros((N, N))
    dwell = np.zeros(N)
    O = {root: p.copy()}
    dq = [root]
    while dq:
        u = dq.pop()
        ch = kids[u]
        mprod = np.ones(N)
        mv = {}
        for c in ch:
            mv[c] = Pc[c] @ inside[c]
            mprod = mprod * mv[c]
        for v in ch:
            sib = np.divide(mprod, mv[v], out=np.zeros(N), where=mv[v] > 1e-300)
            combined = O[u] * sib          # outside message at u for branch u->v
            Z = float(combined @ mv[v])
            if Z > 1e-300:
                t = float(tau[v])
                Ikl = _I_kl(lam, t)
                g = (combined * inv) @ V
                h = (inside[v] * sqrt) @ V
                M = g[:, None] * Ikl * h[None, :]
                VMV = V @ M @ V.T
                EN += (scaled * VMV) / Z
                dwell += np.diag(VMV) / Z
            O[v] = combined @ Pc[v]
            dq.append(v)
    # zero the (spurious) diagonal of EN
    np.fill_diagonal(EN, 0.0)
    return EN, dwell, logZ


# --------------------------------------------- aggregate dense EN,dwell -> stats
def aggregate_dense(EN, dwell, a1, a2, L, Ka):
    """Fold the 800-state EN/dwell into per-archetype N^k[a,b], T^k[a], jumps."""
    Nk = np.zeros((Ka, A, A))
    Tk = np.zeros((Ka, A))
    jumps = 0.0
    for t in range(L):
        k1 = a1[t]
        k2 = a2[t]
        for x in range(A):
            for y in range(A):
                i = cidx(t, x, y)
                dwell_i = dwell[i]
                Tk[k1, x] += dwell_i
                Tk[k2, y] += dwell_i
                # residue-1 substitution (t,x,y)->(t,x',y)
                for xp in range(A):
                    if xp != x:
                        Nk[k1, x, xp] += EN[i, cidx(t, xp, y)]
                # residue-2 substitution (t,x,y)->(t,x,y')
                for yp in range(A):
                    if yp != y:
                        Nk[k2, y, yp] += EN[i, cidx(t, x, yp)]
                # field jump (t,..)->(t',..)
                for tp in range(L):
                    if tp != t:
                        for xp in range(A):
                            for yp in range(A):
                                jumps += EN[i, cidx(tp, xp, yp)]
    return Nk, Tk, jumps


# ----------------------------------------------- single-chain (20-state) tree HR
def residue_tree_hr(parent, tau, leaf_letters, Qk, pik):
    """Standard 20-state endpoint-conditioned HR for one residue under a FIXED
    archetype k.  leaf_letters: dict leaf -> observed letter (int) or None."""
    msgs = {}
    for v, u in enumerate(parent):
        pass
    kids, root = _children(parent)
    leaves = [v for v in range(len(parent)) if not kids[v]]
    for v in leaves:
        ll = leaf_letters.get(v)
        e = np.ones(A) if ll is None else np.eye(A)[ll]
        msgs[v] = e
    return tree_hr(parent, tau, msgs, Qk, pik)


# ============================================================================
#  Regime helpers
# ============================================================================
def build_dense(a1, a2, pi_arch, S, rho, rho_chain):
    Q, _ = _compound_generator_pair(a1, a2, pi_arch, S, rho, rho_chain)
    p = compound_stationary(a1, a2, pi_arch, rho)
    return Q, p


def leaf_pair_msg(x, y):
    """Compound leaf evidence: indicator on (x,y) across all theta."""
    m = np.zeros(L_GLOBAL * A * A)
    for t in range(L_GLOBAL):
        m[cidx(t, x, y)] = 1.0
    return m


L_GLOBAL = 2


# ----------------------------------------------------------------------- checks
def check_felsenstein(pi_arch, S):
    print("\n=== 1. Felsenstein re-assert (dense expm  vs  factored branch op) ===")
    rho = np.array([0.55, 0.45])
    worst = 0.0
    for (a1, a2) in [(np.array([0, 0]), np.array([0, 0])),
                     (np.array([1, 2]), np.array([2, 1])),
                     (np.array([1, 3]), np.array([2, 0]))]:
        for rho_chain in [0.0, 0.4, 1.3]:
            for (tL, tR) in [(0.3, 0.7), (1.1, 0.2)]:
                obsL, obsR = (2, 8), (3, 14)
                bf = cherry_ll_bruteforce(obsL, obsR, tL, tR, a1, a2, pi_arch, S,
                                          rho, rho_chain)
                ex = cherry_ll_exact(obsL, obsR, tL, tR, a1, a2, pi_arch, S,
                                     rho, rho_chain)
                worst = max(worst, abs(bf - ex))
    print(f"  max |LL_dense - LL_factored| over configs = {worst:.3e}")
    return worst


def factored_rho0(parent, tau, leaves_xy, a1, a2, pi_arch, S, rho, Ka):
    """Factored assembly at rho_chain=0 (field frozen, no jumps).

    Field posterior phi(theta) ~ rho[theta] * L1(theta) * L2(theta), then
    per-residue fixed-archetype 20-state tree HR weighted by phi(theta).
    """
    L = rho.shape[0]
    Qk = {k: gtr_Q(pi_arch[k], S) for k in range(Ka)}
    # residue leaf-letter dicts
    ll1 = {v: leaves_xy[v][0] for v in leaves_xy}
    ll2 = {v: leaves_xy[v][1] for v in leaves_xy}
    # per-theta residue likelihoods and HR
    logL = np.zeros(L)
    hr1 = {}
    hr2 = {}
    for th in range(L):
        EN1, dw1, lZ1 = residue_tree_hr(parent, tau, ll1, Qk[a1[th]], pi_arch[a1[th]])
        EN2, dw2, lZ2 = residue_tree_hr(parent, tau, ll2, Qk[a2[th]], pi_arch[a2[th]])
        logL[th] = np.log(rho[th]) + lZ1 + lZ2
        hr1[th] = (EN1, dw1)
        hr2[th] = (EN2, dw2)
    phi = np.exp(logL - logL.max())
    phi /= phi.sum()
    Nk = np.zeros((Ka, A, A))
    Tk = np.zeros((Ka, A))
    for th in range(L):
        EN1, dw1 = hr1[th]
        EN2, dw2 = hr2[th]
        Nk[a1[th]] += phi[th] * EN1
        Tk[a1[th]] += phi[th] * dw1
        Nk[a2[th]] += phi[th] * EN2
        Tk[a2[th]] += phi[th] * dw2
    return Nk, Tk, 0.0


def _single_gen(a_n, pi_arch, S, rho, rho_chain):
    """Compound SINGLETON generator on (theta, x): field jump theta->theta' (!=theta)
    resamples x ~ pi^{a_n(theta')}; residue GTR within field.  40-state, reversible."""
    L = rho.shape[0]
    N = L * A
    Q = np.zeros((N, N))
    Qk = {k: gtr_Q(pi_arch[k], S) for k in set(a_n.tolist())}
    def si(t, x):
        return t * A + x
    for t in range(L):
        for x in range(A):
            i = si(t, x)
            for xp in range(A):
                if xp != x:
                    Q[i, si(t, xp)] += Qk[a_n[t]][x, xp]
            for tp in range(L):
                if tp != t:
                    rate = rho_chain * rho[tp]
                    for xp in range(A):
                        Q[i, si(tp, xp)] += rate * pi_arch[a_n[tp]][xp]
    for i in range(N):
        Q[i, i] = -(Q[i].sum() - Q[i, i])
    p = np.zeros(N)
    for t in range(L):
        for x in range(A):
            p[si(t, x)] = rho[t] * pi_arch[a_n[t]][x]
    return Q, p


def _agg40(EN40, dw40, a_n, Ka, L):
    """Fold a 40-state (theta,x) EN/dwell into per-arch N^k[a,b], T^k[a], jumps."""
    Nk = np.zeros((Ka, A, A)); Tk = np.zeros((Ka, A)); jumps = 0.0
    for t in range(L):
        k = a_n[t]
        for x in range(A):
            i = t * A + x
            Tk[k, x] += dw40[i]
            for xp in range(A):
                if xp != x:
                    Nk[k, x, xp] += EN40[i, t * A + xp]
            for tp in range(L):
                if tp != t:
                    for xp in range(A):
                        jumps += EN40[i, tp * A + xp]
    return Nk, Tk, jumps


def factored_rho_pos(parent, tau, leaves_xy, a1, a2, pi_arch, S, rho, Ka,
                     rho_chain=None):
    """Factored assembly for rho_chain > 0.

    Enumerate field configs (theta at every node, jump-indicator Delta at every
    edge; residues are conditionally independent given (theta-nodes, Delta-edges)).
    For each config: field weight wf, per-residue message-passing likelihood, and
    per-branch residue HR kernels
      Delta=0: 20-state GTR endpoint HR under the frozen archetype;
      Delta=1: (40-state (theta,x) endpoint HR) - (Delta=0 part) -> the reset/renewal.
    Field-jump count uses the residue-independent 2-state field HR, config-weighted.
    Everything is O(K_a A^3); no 800-state object is ever formed.
    """
    L = rho.shape[0]
    n = len(parent)
    kids, root = _children(parent)
    leaves = [v for v in range(n) if not kids[v]]
    edges = [v for v in range(n) if v != root]        # edge identified by child v

    Qk = {k: gtr_Q(pi_arch[k], S) for k in range(Ka)}
    eig_gtr = {k: reversible_eigh(Qk[k], pi_arch[k]) for k in range(Ka)}
    P_gtr = {k: {} for k in range(Ka)}
    # 40-state per-residue chains
    Q40_1, p40_1 = _single_gen(a1, pi_arch, S, rho, rho_chain)
    Q40_2, p40_2 = _single_gen(a2, pi_arch, S, rho, rho_chain)
    eig40 = {1: reversible_eigh(Q40_1, p40_1), 2: reversible_eigh(Q40_2, p40_2)}
    a_of = {1: a1, 2: a2}
    # 2-state field chain
    Qf = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            if i != j:
                Qf[i, j] = rho_chain * rho[j]
        Qf[i, i] = -(Qf[i].sum())
    eigf = reversible_eigh(Qf, rho)

    def Pk(k, t):
        key = round(t, 12)
        if key not in P_gtr[k]:
            lam, V, sq, iv = eig_gtr[k]
            P_gtr[k][key] = _Pt(lam, V, sq, iv, t)
        return P_gtr[k][key]

    # field kernels per unique tau
    fk = {}
    def field_kern(t):
        key = round(t, 12)
        if key not in fk:
            fk[key] = field_kernels(rho, rho_chain, t)
        return fk[key]

    # accumulators
    Nk = np.zeros((Ka, A, A)); Tk = np.zeros((Ka, A)); jumps = 0.0
    Ztot = 0.0

    # enumerate theta at nodes
    for theta_bits in range(L ** n):
        theta = [(theta_bits // (L ** v)) % L for v in range(n)]
        # enumerate Delta at edges; Delta=0 only allowed if theta_u==theta_v
        valid_edges = []
        for v in edges:
            u = int(parent[v])
            opts = [1] if theta[u] != theta[v] else [0, 1]
            valid_edges.append((v, u, opts))
        n_delta = 1
        for (_, _, opts) in valid_edges:
            n_delta *= len(opts)
        for dbits in range(n_delta):
            Delta = {}
            r = dbits
            for (v, u, opts) in valid_edges:
                Delta[v] = opts[r % len(opts)]
                r //= len(opts)
            # field weight
            _, beta_c = None, None
            wf = rho[theta[root]]
            for (v, u, opts) in valid_edges:
                t = float(tau[v])
                P_theta, beta, J = field_kern(t)
                wf *= beta[theta[u]] if Delta[v] == 0 else J[theta[u], theta[v]]
            if wf <= 0:
                continue
            # per-residue message passing (inside + outside) under B_n operators
            res_msgs = {}
            liks = {}
            for rn in (1, 2):
                a_n = a_of[rn]
                # branch operator per edge
                Bop = {}
                for (v, u, opts) in valid_edges:
                    t = float(tau[v])
                    if Delta[v] == 0:
                        Bop[v] = Pk(a_n[theta[u]], t)
                    else:
                        Bop[v] = np.broadcast_to(pi_arch[a_n[theta[v]]][None, :],
                                                 (A, A))
                # inside
                inside = {}
                order = []
                st = [root]
                while st:
                    w = st.pop(); order.append(w)
                    for c in kids[w]:
                        st.append(c)
                for v in reversed(order):
                    if not kids[v]:
                        x = leaves_xy[v][rn - 1]
                        inside[v] = np.eye(A)[x]
                    else:
                        m = np.ones(A)
                        for c in kids[v]:
                            m = m * (Bop[c] @ inside[c])
                        inside[v] = m
                root_prior = pi_arch[a_n[theta[root]]]
                lik = float(root_prior @ inside[root])
                # outside: O[node] (message into node's own children) and per-edge
                # cavity cav[v] = O[parent] * (siblings' upward messages) -- this is
                # the outside message the HR branch operator needs for edge (u,v).
                Onode = {root: root_prior.copy()}
                cav = {}
                dq = [root]
                while dq:
                    uu = dq.pop()
                    ch = kids[uu]
                    mprod = np.ones(A); mv = {}
                    for c in ch:
                        mv[c] = Bop[c] @ inside[c]
                        mprod = mprod * mv[c]
                    for v in ch:
                        sib = np.divide(mprod, mv[v], out=np.zeros(A),
                                        where=mv[v] > 1e-300)
                        cav[v] = Onode[uu] * sib
                        Onode[v] = cav[v] @ Bop[v]
                        dq.append(v)
                res_msgs[rn] = (inside, cav, Bop)
                liks[rn] = lik
            Ztot += wf * liks[1] * liks[2]

            # residue HR: for residue rn, weight = wf * lik_other * (rn messages)
            for rn in (1, 2):
                a_n = a_of[rn]
                other = 2 if rn == 1 else 1
                w_out = wf * liks[other]
                if w_out == 0:
                    continue
                inside, cav, Bop = res_msgs[rn]
                lam40, V40, sq40, iv40 = eig40[rn]
                for (v, u, opts) in valid_edges:
                    t = float(tau[v])
                    out_u = cav[v]               # branch cavity (outside at u \ v)
                    in_v = inside[v]             # inside at child
                    if Delta[v] == 0:
                        k = a_n[theta[u]]
                        EN20, dw20, _ = single_branch_hr(out_u, in_v, t, Qk[k],
                                                         eig=eig_gtr[k])
                        Nk[k] += w_out * EN20
                        Tk[k] += w_out * dw20
                    else:
                        # 40-state full endpoint HR, embed messages in theta-blocks
                        tu, tv = theta[u], theta[v]
                        o40 = np.zeros(L * A); o40[tu * A:tu * A + A] = out_u
                        i40 = np.zeros(L * A); i40[tv * A:tv * A + A] = in_v
                        EN40, dw40, _ = single_branch_hr(o40, i40, t, Q40_1 if rn == 1
                                                         else Q40_2, eig=eig40[rn])
                        Nkc, Tkc, _ = _agg40(EN40, dw40, a_n, Ka, L)
                        # subtract Delta=0 part (only if tu==tv): beta * GTR HR
                        Jw = field_kern(t)[2][tu, tv]
                        if tu == tv:
                            k = a_n[tu]
                            _, beta, _ = field_kern(t)
                            EN20, dw20, _ = single_branch_hr(out_u, in_v, t, Qk[k],
                                                             eig=eig_gtr[k])
                            Nkc[k] -= beta[tu] * EN20
                            Tkc[k] -= beta[tu] * dw20
                        Nk += (w_out / Jw) * Nkc
                        Tk += (w_out / Jw) * Tkc

            # field-jump count: residue-independent 2-state field HR, config-weighted
            w_field = wf * liks[1] * liks[2]
            for (v, u, opts) in valid_edges:
                if Delta[v] == 1:
                    t = float(tau[v])
                    tu, tv = theta[u], theta[v]
                    e_u = np.eye(L)[tu]; e_v = np.eye(L)[tv]
                    EN2, _, _ = single_branch_hr(e_u, e_v, t, Qf, eig=eigf)
                    Jw = field_kern(t)[2][tu, tv]
                    ejumps = (EN2.sum() - np.trace(EN2)) / Jw   # E[#jumps | tu,tv,Delta1]
                    jumps += w_field * ejumps

    Nk /= Ztot; Tk /= Ztot; jumps /= Ztot
    return Nk, Tk, jumps


def _reldiff(a, b, floor=1e-9):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    denom = np.maximum(np.abs(a) + np.abs(b), floor)
    return float(np.max(np.abs(a - b) / denom))


def run_regime(name, a1, a2, pi_arch, S, rho, rho_chain, trees, factored_fn, Ka):
    print(f"\n=== {name}  (rho_chain={rho_chain}, a1={a1.tolist()}, a2={a2.tolist()}) ===")
    Q, p = build_dense(a1, a2, pi_arch, S, rho, rho_chain)
    eig = reversible_eigh(Q, p)
    for tname, (parent, tau, leaves_xy) in trees.items():
        msgs = {v: leaf_pair_msg(*leaves_xy[v]) for v in leaves_xy}
        EN, dwell, logZ = tree_hr(parent, tau, msgs, Q, p, eig=eig)
        Nk_d, Tk_d, jumps_d = aggregate_dense(EN, dwell, a1, a2, rho.shape[0], Ka)
        Nk_f, Tk_f, jumps_f = factored_fn(parent, tau, leaves_xy, a1, a2, pi_arch,
                                          S, rho, Ka)
        dN = _reldiff(Nk_d, Nk_f)
        dT = _reldiff(Tk_d, Tk_f)
        dJ = abs(jumps_d - jumps_f)
        # LL cross check vs exact_cap2
        lp = np.full((len(parent), 2), -1, dtype=int)
        for v, (x, y) in leaves_xy.items():
            lp[v] = (x, y)
        root = int(np.where(np.asarray(parent) < 0)[0][0])
        ll_ref = exact_pair_ll_tree(np.asarray(parent), np.asarray(tau), lp, root,
                                    a1, a2, pi_arch, S, rho, rho_chain)
        print(f"  [{tname}] logZ_dense={logZ:.6f}  logZ_exact_cap2={ll_ref:.6f}  "
              f"dLL={abs(logZ - ll_ref):.2e}")
        print(f"          reldiff N^k[a,b]={dN:.3e}   T^k[a]={dT:.3e}   "
              f"|jumps_dense-jumps_fac|={dJ:.3e}  (jumps_dense={jumps_d:.4f})")
    return


def make_trees():
    # cherry: node 0 = root (internal), nodes 1,2 = leaves
    cherry_parent = np.array([-1, 0, 0])
    cherry_tau = np.array([0.0, 0.35, 0.8])
    cherry_leaves = {1: (2, 8), 2: (3, 14)}     # (D,K) and (E,R): charge-crossing
    # 3-leaf: root 0 -> internal 1 and leaf 2 ; internal 1 -> leaves 3,4
    tri_parent = np.array([-1, 0, 0, 1, 1])
    tri_tau = np.array([0.0, 0.4, 0.9, 0.3, 0.6])
    tri_leaves = {2: (8, 3), 3: (2, 14), 4: (3, 8)}
    return {
        "cherry": (cherry_parent, cherry_tau, cherry_leaves),
        "3-leaf": (tri_parent, tri_tau, tri_leaves),
    }


def validate_dense_primitive(pi_arch, S):
    """Cross-check the reversible single-branch HR primitive on the 800-state
    compound chain against an endpoint-conditioned uniformisation MC.  This
    exercises field-jump attribution + residue substitutions under jumps."""
    print("\n=== 0. Dense HR primitive vs Monte-Carlo (single branch, 800-state) ===")
    a1 = np.array([1, 2]); a2 = np.array([2, 1])
    rho = np.array([0.55, 0.45]); rho_chain = 1.1
    Q, p = build_dense(a1, a2, pi_arch, S, rho, rho_chain)
    eig = reversible_eigh(Q, p)
    t = 0.6
    # endpoints: start (theta=0, x=D=2, y=K=8); end (theta=1, x=E=3, y=R=14)
    i0 = cidx(0, 2, 8); j0 = cidx(1, 3, 14)
    out = np.zeros(800); out[i0] = 1.0
    inn = np.zeros(800); inn[j0] = 1.0
    EN_a, dw_a, Z = single_branch_hr(out, inn, t, Q, eig=eig)
    EN_a /= Z; dw_a /= Z
    EN_m, dw_m = _mc_single_branch(i0, j0, t, Q, n_samp=120000, seed=1)
    # aggregate both into (field jumps, total residue subs, total dwell)
    def agg(EN, dwell):
        jumps = 0.0; subs = 0.0
        for tt in range(2):
            for x in range(A):
                for y in range(A):
                    ii = cidx(tt, x, y)
                    for xp in range(A):
                        if xp != x:
                            subs += EN[ii, cidx(tt, xp, y)]
                    for yp in range(A):
                        if yp != y:
                            subs += EN[ii, cidx(tt, x, yp)]
                    for tp in range(2):
                        if tp != tt:
                            for xp in range(A):
                                for yp in range(A):
                                    jumps += EN[ii, cidx(tp, xp, yp)]
        return jumps, subs, dwell.sum()
    ja, sa, da = agg(EN_a, dw_a)
    jm, sm, dm = agg(EN_m, dw_m)
    print(f"  field jumps  analytic={ja:.4f}  MC={jm:.4f}  (rel {abs(ja-jm)/max(ja,1e-9):.2%})")
    print(f"  resid subs   analytic={sa:.4f}  MC={sm:.4f}  (rel {abs(sa-sm)/max(sa,1e-9):.2%})")
    print(f"  total dwell  analytic={da:.4f}  MC={dm:.4f}  (=t={t}; rel {abs(da-dm)/t:.2%})")
    return


def main():
    S, pi_arch = make_archetypes()
    Ka = pi_arch.shape[0]
    rho = np.array([0.55, 0.45])
    trees = make_trees()

    w = check_felsenstein(pi_arch, S)
    validate_dense_primitive(pi_arch, S)

    # Regime 1: rho_chain = 0 (pure per-residue HR, Delta=0 only). a1 != a2.
    run_regime("2. rho_chain=0  (no jumps; pure per-residue HR)",
               np.array([1, 2]), np.array([2, 1]), pi_arch, S, rho, 0.0,
               trees, factored_rho0, Ka)

    # also a1 == a2 (both residues same archetype map) at rho_chain=0
    run_regime("2b. rho_chain=0  (a1==a2)",
               np.array([0, 1]), np.array([0, 1]), pi_arch, S, rho, 0.0,
               trees, factored_rho0, Ka)

    from functools import partial
    # Regime 3: rho_chain > 0 (jumps active), charge-crossing a1 != a2
    for rc in (0.4, 1.1):
        fn = partial(factored_rho_pos, rho_chain=rc)
        run_regime(f"3. rho_chain={rc}  (jumps active; Delta=1 reset)",
                   np.array([1, 2]), np.array([2, 1]), pi_arch, S, rho, rc,
                   trees, fn, Ka)

    # Regime 4: rho_chain > 0, a1 == a2
    fn = partial(factored_rho_pos, rho_chain=0.7)
    run_regime("4. rho_chain=0.7  (a1==a2, jumps active)",
               np.array([0, 1]), np.array([0, 1]), pi_arch, S, rho, 0.7,
               trees, fn, Ka)


if __name__ == "__main__":
    main()
