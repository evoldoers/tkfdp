"""Exact factored cap-2 cluster Holmes-Rubin, scalable to real family trees.

This is the importable, tree-scalable sibling of the validated numerical check
`analysis/scripts/check_cluster_hr.py` (see `analysis/cluster_ctmc_simplification.md`).
The check validates -- to eigendecomposition round-off, in every regime -- that
the endpoint-conditioned Holmes-Rubin sufficient statistics of the dense
L*A*A (=800) cap-2 cluster CTMC FACTOR into a per-archetype substitution count
N^k[a,b] and dwell T^k[a] at O(K_a*A^3), Monte-Carlo-free.  The check does that
by enumerating the full field configuration (theta at every node, jump-indicator
Delta at every edge), which is O(L^n_nodes) and only feasible on tiny trees.

For a REAL family tree (up to ~256 nodes) L^n_nodes enumeration is impossible.
This module does the same exact statistics by a COMPOUND up-down message pass
(the (L,A,A) messages of `exact_cap2`, which are already validated for the tree
log-likelihood), extracting the per-branch residue HR in factored O(L*A^3) form:

  * Delta=0 (no field jump on the branch): residue n evolves under the frozen
    archetype a_n(theta); its endpoint-conditioned HR is the 20-state GTR HR,
    with the OTHER residue and the outside/inside folded into an A x A weight
    matrix (the generalised `single_branch_hr` -> `_genhr_matrix`).
  * Delta>=1 (>=1 field jump): both residues reset to stationary at every jump,
    which DECOUPLES them on the branch -- given >=1 jump with field endpoints
    (theta_u, theta_v), the other residue's start letter is forgotten and its end
    letter is exactly stationary pi(theta_v).  So residue n's >=1-jump HR is the
    40-state (theta,x) endpoint HR of `check_cluster_hr` with the other residue
    marginalised (row-sum of the outside) and projected onto its stationary
    (pi-weighted inside), minus the Delta=0 GTR part when theta_u==theta_v.

The compound partition Z normalises every branch (the standard up-down trick).
This reproduces the dense 800-state tree HR exactly (validated by
`analysis/scripts/validate_cluster_hr_exact.py`) at O(n_branches * L * A^3), no
Monte-Carlo, no 800-state object.

Reused verbatim from the validated `check_cluster_hr.py`: `reversible_eigh`,
`_I_kl`, `_Pt`, `single_branch_hr`, `tree_hr`, `_agg40`, `build_single_gen`
(there `_single_gen`), `aggregate_dense`.  The dense reference path
`pair_tree_hr_dense` is kept here so callers can self-check.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .exact_cap2 import (gtr_Q, field_kernels, branch_pair,
                         _compound_generator_pair)

A_DIM = 20


# ============================================================================
#  Reversible-chain HR primitives  (verbatim from check_cluster_hr.py)
# ============================================================================
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
      Z        = sum_{i,j} out[i] in[j] P(t)[i,j]."""
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


def _genhr_matrix(W, t, Q, eig):
    """Generalised `single_branch_hr` with a WEIGHT MATRIX W[i,j] in place of the
    separable outer product out[i] in[j].  Returns (EN, dwell), unnormalised:
      EN[c,d]  = sum_{i,j} W[i,j] P(t)[i,j] E[#c->d | i,j]
      dwell[c] = sum_{i,j} W[i,j] P(t)[i,j] E[T_c | i,j].
    With W = out (x) in this reduces exactly to `single_branch_hr`."""
    lam, V, sqrt, inv = eig
    scaled = Q * (sqrt[:, None] * inv[None, :])
    Ikl = _I_kl(lam, t)
    Wtil = (inv[:, None] * W) * sqrt[None, :]
    M = (V.T @ Wtil @ V) * Ikl
    VMV = V @ M @ V.T
    EN = scaled * VMV
    np.fill_diagonal(EN, 0.0)
    dwell = np.diag(VMV).copy()
    return EN, dwell


def _children(parent):
    kids = defaultdict(list)
    root = None
    for v, u in enumerate(parent):
        if u < 0:
            root = v
        else:
            kids[int(u)].append(v)
    return kids, root


def tree_hr(parent, tau, leaf_msgs, Q, p, eig=None):
    """Endpoint-conditioned Holmes-Rubin on a reversible CTMC over a tree.
    (Verbatim from check_cluster_hr.py; used for the dense reference path.)"""
    N = Q.shape[0]
    lam, V, sqrt, inv = eig if eig is not None else reversible_eigh(Q, p)
    kids, root = _children(parent)
    scaled = Q * (sqrt[:, None] * inv[None, :])

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
            combined = O[u] * sib
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
    np.fill_diagonal(EN, 0.0)
    return EN, dwell, logZ


def aggregate_dense(EN, dwell, a1, a2, L, Ka, A=A_DIM):
    """Fold the L*A*A EN/dwell into per-archetype N^k[a,b], T^k[a], jumps."""
    def cidx(t, x, y):
        return (t * A + x) * A + y
    Nk = np.zeros((Ka, A, A))
    Tk = np.zeros((Ka, A))
    jumps = 0.0
    for t in range(L):
        k1 = a1[t]; k2 = a2[t]
        for x in range(A):
            for y in range(A):
                i = cidx(t, x, y)
                dwell_i = dwell[i]
                Tk[k1, x] += dwell_i
                Tk[k2, y] += dwell_i
                for xp in range(A):
                    if xp != x:
                        Nk[k1, x, xp] += EN[i, cidx(t, xp, y)]
                for yp in range(A):
                    if yp != y:
                        Nk[k2, y, yp] += EN[i, cidx(t, x, yp)]
                for tp in range(L):
                    if tp != t:
                        for xp in range(A):
                            for yp in range(A):
                                jumps += EN[i, cidx(tp, xp, yp)]
    return Nk, Tk, jumps


def build_single_gen(a_n, pi_arch, S, rho, rho_chain, A=A_DIM):
    """Compound SINGLETON generator on (theta, x): field jump theta->theta'
    resamples x ~ pi^{a_n(theta')}; residue GTR within field.  L*A-state,
    reversible.  (check_cluster_hr._single_gen, generalised A.)"""
    L = rho.shape[0]
    N = L * A
    Q = np.zeros((N, N))
    Qk = {k: gtr_Q(pi_arch[k], S) for k in set(int(v) for v in a_n)}

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


def _agg40(EN40, dw40, a_n, Ka, L, A=A_DIM):
    """Fold an L*A-state (theta,x) EN/dwell into per-arch N^k[a,b], T^k[a].
    Vectorised: per field state theta the residue substitutions are the A x A
    diagonal block of EN40; jumps are the off-diagonal (theta != theta') mass."""
    Nk = np.zeros((Ka, A, A)); Tk = np.zeros((Ka, A))
    jumps = float(EN40.sum())
    for t in range(L):
        k = int(a_n[t])
        blk = EN40[t * A:t * A + A, t * A:t * A + A]
        Nk[k] += blk                                 # diagonal already zeroed
        Tk[k] += dw40[t * A:t * A + A]
        jumps -= float(blk.sum())                    # within-field mass is not a jump
    return Nk, Tk, jumps


# ============================================================================
#  Dense reference path (pair) -- for validation only
# ============================================================================
def _compound_stationary(a1, a2, pi_arch, rho, A=A_DIM):
    L = rho.shape[0]
    p = np.zeros(L * A * A)
    for t in range(L):
        for x in range(A):
            for y in range(A):
                p[(t * A + x) * A + y] = rho[t] * pi_arch[a1[t]][x] * pi_arch[a2[t]][y]
    return p


def _leaf_pair_compound_msg(x, y, L, A=A_DIM):
    """Compound leaf evidence indicator on (x,y) across theta; gap (>=A or <0)
    -> uninformative (all ones) on that residue."""
    xr = np.ones(A) if (x is None or x < 0 or x >= A) else np.eye(A)[x]
    yr = np.ones(A) if (y is None or y < 0 or y >= A) else np.eye(A)[y]
    blk = np.outer(xr, yr).reshape(-1)
    return np.concatenate([blk for _ in range(L)])


def pair_tree_hr_dense(parent, tau, xcol, ycol, a1, a2, pi_arch, S, rho,
                       rho_chain, A=A_DIM):
    """DENSE reference: build the L*A*A compound generator and run the reversible
    tree HR, then aggregate.  O(n_branches * (L*A*A)^3) -- validation only."""
    a1 = np.asarray(a1); a2 = np.asarray(a2)
    L = rho.shape[0]; Ka = pi_arch.shape[0]
    Q, _ = _compound_generator_pair(a1, a2, pi_arch, S, rho, rho_chain)
    p = _compound_stationary(a1, a2, pi_arch, rho, A)
    kids, _ = _children(parent)
    leaf_msgs = {}
    for v in range(len(parent)):
        if not kids[v]:
            leaf_msgs[v] = _leaf_pair_compound_msg(int(xcol[v]), int(ycol[v]), L, A)
    EN, dwell, logZ = tree_hr(parent, tau, leaf_msgs, Q, p)
    Nk, Tk, jumps = aggregate_dense(EN, dwell, a1, a2, L, Ka, A)
    return Nk, Tk, jumps, logZ


# ============================================================================
#  Exact factored PAIR HR via compound up-down  (scalable)
# ============================================================================
def _pair_leaf_msg(x, y, L, A=A_DIM):
    """(L,A,A) compound leaf message; gaps -> uninformative on that residue."""
    xr = np.ones(A) if (x is None or x < 0 or x >= A) else np.eye(A)[x]
    yr = np.ones(A) if (y is None or y < 0 or y >= A) else np.eye(A)[y]
    blk = np.outer(xr, yr)
    return np.repeat(blk[None], L, axis=0).copy()


class SharedArchEig:
    """Corpus-wide reusable per-archetype GTR generator + reversible eigh, plus a
    memoised 40-state (theta,x) chain cache keyed by (a_n tuple, rho_chain).
    Rebuilt whenever pi_archetype changes (once per M-step), shared across every
    cluster/config so the O(K_a A^3) eigendecompositions are not repeated."""
    def __init__(self, pi_arch, S, rho, A=A_DIM):
        self.pi_arch = pi_arch; self.S = S; self.rho = np.asarray(rho, float)
        self.A = A; self.Ka = pi_arch.shape[0]
        self.gtr_Q = {k: gtr_Q(pi_arch[k], S) for k in range(self.Ka)}
        self.gtr_eig = {k: reversible_eigh(self.gtr_Q[k], pi_arch[k])
                        for k in range(self.Ka)}
        self._q40 = {}

    def q40(self, a_n, rho_chain):
        key = (tuple(int(v) for v in a_n), round(float(rho_chain), 12))
        if key not in self._q40:
            Q40, p40 = build_single_gen(a_n, self.pi_arch, self.S, self.rho,
                                        float(rho_chain), self.A)
            self._q40[key] = (Q40, reversible_eigh(Q40, p40))
        return self._q40[key]


class _PairKernels:
    """Per-config kernels shared across all branches of a tree: per-archetype GTR
    eig + P(tau); the two residue 40-state (theta,x) chains.  Reuses a
    `SharedArchEig` when supplied (avoids rebuilding the K_a eigendecompositions
    and the 40-state chains for every cluster/config)."""
    def __init__(self, a1, a2, pi_arch, S, rho, rho_chain, A=A_DIM, shared=None):
        self.a1 = np.asarray(a1); self.a2 = np.asarray(a2)
        self.pi_arch = pi_arch; self.S = S; self.rho = np.asarray(rho, float)
        self.rho_chain = float(rho_chain); self.A = A
        self.L = self.rho.shape[0]
        self.Ka = pi_arch.shape[0]
        ks = set(int(k) for k in self.a1.tolist()) | set(int(k) for k in self.a2.tolist())
        self.ks = ks
        if shared is None:
            shared = SharedArchEig(pi_arch, S, self.rho, A)
        self.gtr_Q = {k: shared.gtr_Q[k] for k in ks}
        self.gtr_eig = {k: shared.gtr_eig[k] for k in ks}
        self._P_cache = {k: {} for k in ks}
        # residue 40-state chains (memoised in the shared cache)
        self.Q40 = {}
        self.eig40 = {}
        for rn, a_n in ((1, self.a1), (2, self.a2)):
            Q40, eig40 = shared.q40(a_n, self.rho_chain)
            self.Q40[rn] = Q40
            self.eig40[rn] = eig40
        # 2-state field chain (for the field-jump count; rho_chain-dependent)
        Qf = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(self.L):
                if i != j:
                    Qf[i, j] = self.rho_chain * self.rho[j]
            Qf[i, i] = -Qf[i].sum()
        self.Qf = Qf
        self.eigf = reversible_eigh(Qf, self.rho) if self.rho_chain > 0 else None

    def P(self, k, t):
        key = round(float(t), 12)
        c = self._P_cache[k]
        if key not in c:
            c[key] = _Pt(*self.gtr_eig[k], t)
        return c[key]

    def P_dict(self, t):
        return {k: self.P(k, t) for k in self.ks}


def _branch_pair_hr(O_u, I_v, t, K, Nk, Tk, mvI=None, want_jumps=True):
    """Accumulate the UNnormalised per-archetype residue HR of ONE branch into
    Nk (Ka,A,A) / Tk (Ka,A).  Returns the branch compound partition Z and the
    unnormalised field-jump mass (0 if want_jumps is False).

    O_u, I_v: (L,A,A) compound outside(parent)/inside(child).  K: _PairKernels.
    mvI: optional precomputed branch_pair(I_v) (reused from the outside pass).
    """
    A = K.A; L = K.L
    a1 = K.a1; a2 = K.a2; pi_arch = K.pi_arch
    Pth, beta, J = field_kernels(K.rho, K.rho_chain, t)
    Pd = K.P_dict(t)

    # branch partition Z = <O_u, branch_pair(I_v)>
    if mvI is None:
        mvI = branch_pair(I_v, t, a1, a2, Pd, pi_arch, K.rho, K.rho_chain)
    Z = float((O_u * mvI).sum())

    # ---- Delta = 0 : frozen field theta, residues independent -------------
    for th in range(L):
        k1 = int(a1[th]); k2 = int(a2[th])
        P1 = Pd[k1]; P2 = Pd[k2]
        Ou = O_u[th]; Iv = I_v[th]
        # residue 1 weight matrix over (x_u, x_v); residue 2 folded via P2
        W1 = beta[th] * (Ou @ P2 @ Iv.T)
        EN1, dw1 = _genhr_matrix(W1, t, K.gtr_Q[k1], K.gtr_eig[k1])
        Nk[k1] += EN1; Tk[k1] += dw1
        # residue 2 weight matrix over (y_u, y_v); residue 1 folded via P1
        W2 = beta[th] * (Ou.T @ P1 @ Iv)
        EN2, dw2 = _genhr_matrix(W2, t, K.gtr_Q[k2], K.gtr_eig[k2])
        Nk[k2] += EN2; Tk[k2] += dw2

    # ---- Delta >= 1 : reset/renewal decouples the residues ----------------
    # rho_chain == 0 -> no field jumps: J == 0, all Delta>=1 terms vanish.
    if K.rho_chain <= 0:
        return Z, 0.0
    for rn, a_n, other_pi_side in ((1, a1, 2), (2, a2, 1)):
        lam, V, sqrt, inv = K.eig40[rn]
        Q40 = K.Q40[rn]
        scaled40 = Q40 * (sqrt[:, None] * inv[None, :])
        Ikl40 = _I_kl(lam, t)
        for tu in range(L):
            # marginalise the OTHER residue's start (row/col sum of O_u)
            if rn == 1:
                outv = O_u[tu].sum(axis=1)                   # (A,) over x_u
            else:
                outv = O_u[tu].sum(axis=0)                   # (A,) over y_u
            bu = slice(tu * A, tu * A + A)
            g = (outv * inv[bu]) @ V[bu]                     # (L*A,)
            for tv in range(L):
                # project the OTHER residue's end onto its stationary pi(tv)
                if rn == 1:
                    inv_end = I_v[tv] @ pi_arch[int(a2[tv])]
                else:
                    inv_end = pi_arch[int(a1[tv])] @ I_v[tv]
                bv = slice(tv * A, tv * A + A)
                h = (inv_end * sqrt[bv]) @ V[bv]             # (L*A,)
                M = g[:, None] * Ikl40 * h[None, :]
                VMV = V @ M @ V.T
                EN40 = scaled40 * VMV
                np.fill_diagonal(EN40, 0.0)
                dw40 = np.diag(VMV)
                Nkc, Tkc, _ = _agg40(EN40, dw40, a_n, K.Ka, L, A)
                if tu == tv:                                 # remove the 0-jump part
                    k = int(a_n[tu])
                    EN0, dw0 = _genhr_matrix(np.outer(outv, inv_end), t,
                                             K.gtr_Q[k], K.gtr_eig[k])
                    Nkc[k] -= beta[tu] * EN0
                    Tkc[k] -= beta[tu] * dw0
                Nk += Nkc; Tk += Tkc

    # ---- field-jump mass (residue-independent, config-weighted) -----------
    # JMP branch operator contributes, per (tu,tv): (sum O_u[tu]) * J[tu,tv]*m(tv)
    # with m(tv)=pi1(tv).I_v[tv].pi2(tv).  Conditional E[jumps|tu,tv]=ejumps.
    jmass = 0.0
    if want_jumps and K.rho_chain > 0:
        m_c = np.array([pi_arch[int(a1[tp])] @ I_v[tp] @ pi_arch[int(a2[tp])]
                        for tp in range(L)])
        Ou_mass = O_u.sum(axis=(1, 2))                       # (L,) over x,y at parent
        for tu in range(L):
            for tv in range(L):
                if J[tu, tv] <= 0:
                    continue
                e_u = np.eye(L)[tu]; e_v = np.eye(L)[tv]
                EN2, _, _ = single_branch_hr(e_u, e_v, t, K.Qf, eig=K.eigf)
                ejumps = (EN2.sum() - np.trace(EN2)) / J[tu, tv]
                jmass += Ou_mass[tu] * J[tu, tv] * m_c[tv] * ejumps
    return Z, jmass


def pair_tree_hr(parent, tau, xcol, ycol, a1, a2, pi_arch, S, rho, rho_chain,
                 A=A_DIM, shared=None, want_jumps=True):
    """EXACT factored cap-2 pair Holmes-Rubin over a real family tree.

    parent (n,), tau (n,): rooted tree, parent[root] < 0.
    xcol, ycol: (n,) residue letters for the two paired columns; leaves observe
      an int in [0,A) (gap or internal -> value >=A or <0, uninformative).
    a1[theta], a2[theta]: archetype index for column 1 / column 2 at field theta.
    shared: optional `SharedArchEig` reused across clusters/configs.

    Returns (Nk (Ka,A,A), Tk (Ka,A), jumps).  Matches `pair_tree_hr_dense`
    (the dense 800-state tree HR) to eigendecomposition precision.
    """
    a1 = np.asarray(a1); a2 = np.asarray(a2)
    L = np.asarray(rho, float).shape[0]
    Ka = pi_arch.shape[0]
    K = _PairKernels(a1, a2, pi_arch, S, rho, rho_chain, A, shared=shared)
    kids, root = _children(parent)
    n = len(parent)

    # ---- compound inside pass (postorder), per-node normalised -----------
    order = []
    stack = [root]
    while stack:
        v = stack.pop(); order.append(v)
        for c in kids[v]:
            stack.append(c)
    inside = {}
    logscale = {}
    mv_cache = {}                                    # branch_pair(inside[c]) per child
    for v in reversed(order):
        if not kids[v]:
            m = _pair_leaf_msg(int(xcol[v]), int(ycol[v]), L, A)
            ls = 0.0
        else:
            m = np.ones((L, A, A)); ls = 0.0
            for c in kids[v]:
                mvc = branch_pair(inside[c], float(tau[c]), a1, a2,
                                  K.P_dict(float(tau[c])), pi_arch, K.rho,
                                  K.rho_chain)
                mv_cache[c] = mvc
                m = m * mvc
                ls += logscale[c]
        s = m.sum()
        if s <= 0:
            s = 1.0
        inside[v] = m / s
        # rescale the cached child messages consistency is not needed (we recompute)
        logscale[v] = ls + np.log(s)

    p_comp = np.zeros((L, A, A))
    for th in range(L):
        p_comp[th] = rho[th] * np.outer(pi_arch[int(a1[th])], pi_arch[int(a2[th])])

    # ---- compound outside pass + per-branch HR ---------------------------
    Nk = np.zeros((Ka, A, A)); Tk = np.zeros((Ka, A)); jumps = 0.0
    O = {root: p_comp.copy()}
    dq = [root]
    while dq:
        u = dq.pop()
        ch = kids[u]
        if not ch:
            continue
        mv = {}
        mprod = np.ones((L, A, A))
        for c in ch:
            mv[c] = mv_cache[c]                       # reuse from inside pass
            mprod = mprod * mv[c]
        for v in ch:
            sib = np.divide(mprod, mv[v], out=np.zeros_like(mprod),
                            where=mv[v] > 1e-300)
            combined = O[u] * sib                    # (L,A,A) outside cavity at u
            t = float(tau[v])
            Nk_b = np.zeros((Ka, A, A)); Tk_b = np.zeros((Ka, A))
            Z, jmass = _branch_pair_hr(combined, inside[v], t, K, Nk_b, Tk_b,
                                       mvI=mv[v], want_jumps=want_jumps)
            if Z > 1e-300:
                Nk += Nk_b / Z
                Tk += Tk_b / Z
                jumps += jmass / Z
            # propagate outside to child via reversibility:
            #   O[v] = p_comp (*) branch_pair(combined / p_comp)
            ceff = np.divide(combined, p_comp, out=np.zeros_like(combined),
                             where=p_comp > 1e-300)
            O[v] = p_comp * branch_pair(ceff, t, a1, a2, K.P_dict(t), pi_arch,
                                        K.rho, K.rho_chain)
            dq.append(v)
    return Nk, Tk, jumps


# ============================================================================
#  Exact SINGLETON HR via the 40-state (theta,x) tree  (scalable, dense-cheap)
# ============================================================================
def single_tree_hr(parent, tau, xcol, a_n, pi_arch, S, rho, rho_chain, A=A_DIM):
    """EXACT per-archetype HR for a single (non-contact) column, via the
    L*A-state (theta,x) compound chain tree HR.  a_n[theta] = archetype at field
    theta.  Returns (Nk (Ka,A,A), Tk (Ka,A), jumps)."""
    a_n = np.asarray(a_n)
    L = np.asarray(rho, float).shape[0]
    Ka = pi_arch.shape[0]
    Q40, p40 = build_single_gen(a_n, pi_arch, S, np.asarray(rho, float),
                                float(rho_chain), A)
    kids, _ = _children(parent)
    leaf_msgs = {}
    for v in range(len(parent)):
        if not kids[v]:
            x = int(xcol[v])
            e = np.ones(A) if (x < 0 or x >= A) else np.eye(A)[x]
            msg = np.zeros(L * A)
            for th in range(L):
                msg[th * A:th * A + A] = e
            leaf_msgs[v] = msg
    EN, dwell, logZ = tree_hr(parent, tau, leaf_msgs, Q40, p40)
    Nk, Tk, jumps = _agg40(EN, dwell, a_n, Ka, L, A)
    return Nk, Tk, jumps
