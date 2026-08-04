#!/usr/bin/env python3
r"""First-order coupling reachability of two-sided-lumpable pair chains (the "product trap").

For a reversible pair CTMC on [n]x[n] built over a single-site generator W (reversible
w.r.t. u, eigensystem W psi_a = lam_a psi_a, lam_0=0, psi_0=const), we linearise the
two-sided-lumpable REVERSIBLE family at the product chain pi0 = u (x) u and ask which
stationary-coupling directions can be turned on at first order.

THEOREM (verified here to machine precision).  The first-order reachable coupling tangent is
    span{ psi_a (x) psi_b : lam_a = lam_b, a,b >= 1 }        (the resonant eigen-mode pairs),
of dimension   sum_{lam != 0} m_lam^2   (non-exchangeable; m_lam = multiplicity of eigenvalue lam).
Generic spectrum (all distinct) -> n-1 (only a=b).  Fully degenerate (JC69) -> (n-1)^2 (full).

This script computes (a) the reachable tangent dimension by direct linear algebra, (b) the
closed-form prediction, and (c) the principal angles between the reachable tangent and the
eigen-mode-agreement span, for generic backgrounds and the DNA progression JC69/K2P/HKY85.
numpy only.  Run:  python experiments/lumpable_tangent_rank.py
"""
import numpy as np, itertools


def _modes(r, u):
    n = len(u)
    W = r * u[None, :]; np.fill_diagonal(W, -W.sum(1))
    su = np.sqrt(u); Ws = su[:, None] * W / su[None, :]; Ws = (Ws + Ws.T) / 2
    lam, V = np.linalg.eigh(Ws); psi = V / su[:, None]
    o = np.argsort(-lam)
    return W, lam[o], psi[:, o]


def reachable_tangent(r, u):
    """Dimension + orthonormal basis of the first-order reachable coupling (non-exchangeable)."""
    n = len(u); states = list(itertools.product(range(n), range(n))); N = n * n
    sidx = {s: k for k, s in enumerate(states)}
    pi0 = np.array([u[i] * u[j] for (i, j) in states])
    W = r * u[None, :]; np.fill_diagonal(W, -W.sum(1))
    S0 = np.zeros((N, N))                                   # product flux conductances Q0_xy=S0_xy*pi0_y
    for a, (i, j) in enumerate(states):
        for b, (k, l) in enumerate(states):
            if a == b:
                continue
            if l == j and k != i:   S0[a, b] = W[i, k] / pi0[b]     # = W_ik/(u_k u_j)
            elif k == i and l != j: S0[a, b] = W[j, l] / pi0[b]
    pairs = [(a, b) for a in range(N) for b in range(N) if a < b]; pid = {}
    for t, (a, b) in enumerate(pairs): pid[(a, b)] = t; pid[(b, a)] = t
    npr = len(pairs); nz = npr + N
    od = [(a, b) for a in range(N) for b in range(N) if a != b]
    G = np.zeros((len(od), nz))                            # (sigma, dpi) -> dQ,  dQ=sigma*pi0 + S0*dpi
    for t, (a, b) in enumerate(od):
        pr = (a, b) if a < b else (b, a); G[t, pid[pr]] += pi0[b]; G[t, npr + b] += S0[a, b]

    def blk(a, coord, val):
        f = np.zeros(len(od))
        for t, (x, y) in enumerate(od):
            if x == a and states[y][coord] == val: f[t] = 1
        return f

    rows = []                                              # two-sided lumpability: block rates context-free
    for i in range(n):
        for k in range(n):
            if k == i: continue
            js = [sidx[(i, j)] for j in range(n)]; base = blk(js[0], 0, k)
            for jj in js[1:]: rows.append(blk(jj, 0, k) - base)
    for j in range(n):
        for l in range(n):
            if l == j: continue
            ks = [sidx[(i, j)] for i in range(n)]; base = blk(ks[0], 1, l)
            for kk in ks[1:]: rows.append(blk(kk, 1, l) - base)
    B = np.array(rows) @ G
    e = np.zeros(nz); e[npr:] = 1; B = np.vstack([B, e])   # sum(dpi)=0
    _, s_, vt = np.linalg.svd(B); ker = vt[np.sum(s_ > 1e-9):].T
    coup = []
    for c in range(ker.shape[1]):
        DP = ker[npr:, c].reshape(n, n); am = DP.sum(1); bm = DP.sum(0)
        coup.append((DP - np.outer(am, u) - np.outer(u, bm)).ravel())    # strip marginals
    coup = np.array(coup); _, sv, cv = np.linalg.svd(coup)
    return cv[:int(np.sum(sv > 1e-8))]


def eigenmode_span(lam, psi, u, tol=1e-5):
    """span{ psi_a (x) psi_b : lam_a=lam_b, a,b>=1 } as an orthonormal basis of couplings."""
    n = len(u); B = []
    for a in range(1, n):
        for b in range(1, n):
            if abs(lam[a] - lam[b]) < tol:
                d = np.outer(u * psi[:, a], u * psi[:, b])
                if np.linalg.norm(d) > 1e-9: B.append(d.ravel() / np.linalg.norm(d))
    return np.linalg.qr(np.array(B).T)[0].T if B else np.zeros((0, n * n))


def resonance_count(lam, tol=1e-5):
    nz = [l for l in lam if abs(l) > tol]
    s = 0.0
    seen = []
    for l in nz:
        if any(abs(l - v) < tol for v in seen): continue
        seen.append(l); m = sum(abs(l - v) < tol for v in nz); s += m * m
    return int(round(s))


def principal_angles(A, Bm):
    if A.shape[0] == 0 or Bm.shape[0] == 0: return np.array([])
    Qa = np.linalg.qr(A.T)[0]; Qb = np.linalg.qr(Bm.T)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.round(np.degrees(np.arccos(np.clip(s, -1, 1))), 1)


def _dna_r(kappa, n=4):
    A, G, C, T = 0, 1, 2, 3
    r = np.ones((n, n))
    for (a, b) in [(A, G), (C, T)]: r[a, b] = r[b, a] = kappa
    np.fill_diagonal(r, 0); return r


def report(name, r, u):
    W, lam, psi = _modes(r, u)
    Tb = reachable_tangent(r, u); Es = eigenmode_span(lam, psi, u)
    n = len(u)
    print(f"{name:<22} eig(W)={np.round(lam,3)}  u={np.round(u,3)}")
    print(f"{'':22} reachable dim={Tb.shape[0]:2d}   sum m_lam^2={resonance_count(lam):2d}   "
          f"[n-1={n-1}, full={(n-1)**2}]   angles(tangent,eigenspan)={principal_angles(Tb,Es)}")


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    print("== generic random backgrounds (distinct eigenvalues -> n-1) ==")
    for n in (3, 4, 5):
        r = np.abs(rng.randn(n, n)) + 0.6; r = (r + r.T) / 2; np.fill_diagonal(r, 0)
        u = np.abs(rng.randn(n)) + 0.5; u /= u.sum()
        report(f"random n={n}", r, u)
    print("\n== DNA progression: the trap is set by the DYNAMICS (exchangeability), not composition ==")
    _ren = np.ones((4, 4)); np.fill_diagonal(_ren, 0)     # F81/renewal: r = const (memoryless)
    report("F81/renewal unif",  _ren,        np.ones(4) / 4)
    report("F81/renewal skewed", _ren,       np.array([0.1, 0.2, 0.3, 0.4]))   # full even when skewed
    report("JC69",            _dna_r(1.0), np.ones(4) / 4)
    report("K2P (kappa=4)",   _dna_r(4.0), np.ones(4) / 4)
    report("HKY85 mild-skew", _dna_r(4.0), np.array([0.22, 0.28, 0.28, 0.22]))
    report("HKY85 skewed",    _dna_r(4.0), np.array([0.1, 0.2, 0.3, 0.4]))
