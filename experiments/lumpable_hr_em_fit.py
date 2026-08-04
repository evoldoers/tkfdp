#!/usr/bin/env python3
r"""EXACT-lumpability HR-EM fit of the two-sided-lumpable reversible pair chain at
n=4, and the controlled product-trap / renewal-init experiment.

WHY THIS FILE EXISTS.  `analysis/lumpable_product_trap.md` proves that, linearised
at the PRODUCT chain, a reversible two-sided-lumpable perturbation can turn on a
stationary coupling only in the RESONANT tangent span{psi_a (x) psi_b: lam_a=lam_b}.
For a structured single-site background W with DISTINCT eigenvalues the tangent is
only the (n-1) agreement modes, so a likelihood-ascent trainer started at the
product should STALL (MI~0) for any coupling in the non-resonant complement.  The
paper's own operational check used a soft-Lagrangian L-BFGS on the non-convex
observed-data likelihood -- too noisy to trust.  This file replaces it with a
RELIABLE exact-lumpability HR-EM fit and settles, at n=4, (a) whether the trap
bites a real trainer, and (b) whether renewal initialisation cures it.

THE FITTER (route (ii), exact -- NOT a penalty).
  * State space: ordered pairs x=(i,j) in [n]^2 (n^2 states).  Reversible
    exchangeable generator Q_xy = F_xy/pi_x with F symmetric+Klein-4-orbit-constant
    (so component-exchangeable and reversible w.r.t. an exchangeable joint pi).
  * E-step: EXACT Holmes-Rubin endpoint-conditioned bridge (`tkfdp.permfield.hr`),
    summed over the endpoint-joint data, giving expected usage N_xy and dwell T_x
    (same machinery as `fit_pair_models.estep`, generalised to any n).
  * Flux M-step (EXACT lumpable, positivity automatic): at fixed pi, the two-sided
    strong-lumpability constraint on the orbit flux phi is HOMOGENEOUS LINEAR,
    C(pi) phi = 0 (sum_l Q[(i,j)->(k,l)] independent of j).  The complete-data
    auxiliary sum_o[C_o log phi_o - H_o phi_o] is concave; its KKT point is
    phi_o = C_o/(H_o + (C^T mu)_o), solved for mu by a damped, feasibility-guarded
    Newton on the convex dual Psi(mu) = -sum_o C_o log(H_o+(C^T mu)_o) (the rational
    dual / Sinkhorn analogue).  phi_o = C_o/den >= 0 automatically, and C phi -> 0
    to ~1e-13, so lumpability is EXACT (verified: sum_l Q indep of j to ~1e-11).
  * pi-step (constrained, exact): the coupled exchangeable stationary is a free
    block.  We PROFILE the flux out -- m(pi) = max_{phi lumpable,>=0} sum_o[...]
    - sum_x N_x^row log pi_x -- and locally maximise m over pi with a warm-started
    L-BFGS.  Because the flux is re-solved on null(C(pi)) at every trial pi, the
    (pi, flux) pair is EXACTLY lumpable at every point; the update is a generalised
    EM M-step (Q(theta_new) >= Q(theta_old)), so the observed-data LL is monotone.

The Synchronized (no-lumpability) variant drops C(pi): flux M-step phi_o=C_o/H_o.

Run:  OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONPATH=src \
      python3 experiments/lumpable_hr_em_fit.py [--verify] [--trap]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from tkfdp.permfield.hr import bridge, eig_rev            # noqa: E402
import lumpable_tangent_rank as ltr                       # noqa: E402


# ============================================================================
# generic-n pair-chain machinery
# ============================================================================
def build_orbits_n(n):
    """Klein-4 (V4) orbit id over directed off-diagonal transitions on [n]^2."""
    NS = n * n
    ii, jj = np.divmod(np.arange(NS), n)
    orbit_id = np.full((NS, NS), -1, np.int64)
    canon = {}
    nxt = 0
    for x in range(NS):
        i, j = int(ii[x]), int(jj[x])
        for y in range(NS):
            if x == y:
                continue
            k, l = int(ii[y]), int(jj[y])
            imgs = ((i, j, k, l), (k, l, i, j), (j, i, l, k), (l, k, j, i))
            key = min(imgs)
            o = canon.get(key)
            if o is None:
                o = nxt
                canon[key] = o
                nxt += 1
            orbit_id[x, y] = o
    ch1 = ii[:, None] != ii[None, :]
    ch2 = jj[:, None] != jj[None, :]
    is_single = ch1 ^ ch2
    return orbit_id, nxt, is_single


def Q_from_flux_n(F, pi):
    n2 = F.shape[0]
    Q = F / np.maximum(pi[:, None], 1e-300)
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(n2)] = -Q.sum(1)
    return Q


def flux_from_orbit_n(phi, orbit_id):
    n2 = orbit_id.shape[0]
    off = ~np.eye(n2, dtype=bool)
    F = np.zeros((n2, n2))
    F[off] = phi[orbit_id[off]]
    return F


def estep_n(Q, pi, tau, npair):
    n2 = Q.shape[0]
    eig = eig_rev(Q, np.clip(pi, 1e-12, None))
    N = np.zeros((n2, n2))
    T = np.zeros(n2)
    for t in range(tau.shape[0]):
        edge = npair[:, :, t]
        if edge.sum() == 0:
            continue
        Tt, Nt, _ = bridge(Q, pi, tau[t], edge, want_N=True, eig=eig)
        N += Nt
        T += Tt
    return N, T


def loglik_n(Q, pi, tau, npair):
    lam, U, Uinv = eig_rev(Q, np.clip(pi, 1e-12, None))
    ll = 0.0
    for t in range(tau.shape[0]):
        edge = npair[:, :, t]
        if edge.sum() == 0:
            continue
        P = (U * np.exp(lam * tau[t])[None, :]) @ Uinv
        ll += float((edge * np.log(np.clip(P, 1e-300, None))).sum())
    return ll


def empirical_pi_n(npair, n):
    """Symmetrised start+end occupancy -> empirical (coupled) joint stationary."""
    tot = npair.sum(2)
    occ = 0.5 * (tot.sum(1) + tot.sum(0))
    pi = occ.reshape(n, n)
    pi = 0.5 * (pi + pi.T)
    pi = pi / pi.sum()
    return pi.reshape(n * n)


def mutual_information(pi, n):
    """MI (nats) of the joint stationary between the two coordinates."""
    P = np.clip(pi.reshape(n, n), 0, None)
    P = P / P.sum()
    ri = P.sum(1)
    rj = P.sum(0)
    out = 0.0
    for a in range(n):
        for b in range(n):
            if P[a, b] > 0:
                out += P[a, b] * np.log(P[a, b] / (ri[a] * rj[b]))
    return float(out)


def lumpability_residual(Q, n):
    """Strong two-sided lumpability residual: max over off-block (i,k) of
    range_j(sum_l Q[(i,j)->(k,l)]), and the Y-symmetric version.  ~0 iff lumpable."""
    Q4 = Q.reshape(n, n, n, n)                    # [i,j,k,l]
    AX = Q4.sum(axis=3)                           # [i,j,k] = sum_l
    resX = 0.0
    for i in range(n):
        for k in range(n):
            if k == i:
                continue
            v = AX[i, :, k]
            resX = max(resX, float(v.max() - v.min()))
    AY = Q4.sum(axis=2)                           # [i,j,l] = sum_k
    resY = 0.0
    for j in range(n):
        for l in range(n):
            if l == j:
                continue
            v = AY[:, j, l]
            resY = max(resY, float(v.max() - v.min()))
    return max(resX, resY)


# ============================================================================
# orbit aggregation + exact lumpable flux M-step (rational-dual Newton)
# ============================================================================
def _off_index(n):
    n2 = n * n
    off = ~np.eye(n2, dtype=bool)
    src = np.repeat(np.arange(n2), n2).reshape(n2, n2)[off]
    return off, src


def precompute_lump_constraint(orbit_id, n):
    """COO structure of the homogeneous X-lumpability constraint C(pi) phi = 0.

    Rows: for each i, k != i, j in 1..n-1,  b_{i,j,k} - b_{i,0,k} = 0 with
    b_{i,j,k} = sum_l phi[orbit((i,j),(k,l))]/pi_{(i,j)}.  (Y-lumpability follows
    from component-exchangeability.)  Returns (row,col,pistate,sign,n_rows); build
    the dense C at a given pi by scattering sign/pi[pistate] into (row,col)."""
    rr, cc, ps, sg = [], [], [], []
    r = 0
    for i in range(n):
        for k in range(n):
            if k == i:
                continue
            for j in range(1, n):
                xj = i * n + j
                x0 = i * n + 0
                for l in range(n):
                    yk = k * n + l
                    rr.append(r); cc.append(int(orbit_id[xj, yk])); ps.append(xj); sg.append(1.0)
                    rr.append(r); cc.append(int(orbit_id[x0, yk])); ps.append(x0); sg.append(-1.0)
                r += 1
    return (np.array(rr), np.array(cc), np.array(ps), np.array(sg), r)


def build_Cmat(pre, pi, n_orbits):
    """Dense homogeneous lumpability constraint C(pi).  ROW-NORMALISED: each
    constraint row carries entries ~1/pi (up to ~1/min(pi)), so on small-u
    backgrounds (hky85) the raw C is badly scaled and the dual Newton ill-
    conditioned.  Scaling each row to unit norm leaves null(C) -- the lumpable flux
    family and its optimum -- unchanged, but conditions the solve."""
    rr, cc, ps, sg, nr = pre
    data = sg / np.maximum(pi[ps], 1e-300)
    C = np.zeros((nr, n_orbits))
    np.add.at(C, (rr, cc), data)
    rn = np.sqrt((C * C).sum(1))
    C = C / np.maximum(rn[:, None], 1e-300)
    return C


# warm-start cache for the flux-dual multipliers.  Consecutive L-BFGS pi-evals in
# the M-step have nearly identical pi, so the previous mu is a near-solution and the
# dual Newton converges in 1-2 steps (vs dozens with heavy backtracking cold).
_MU_CACHE = {"mu": None}


def mstep_flux_lumpable(C_o, H_o, Cmat, iters=60, tol=1e-12, ridge=1e-10,
                        use_cache=False):
    """max_phi sum_o[C_o log phi_o - H_o phi_o] s.t. C phi = 0, phi >= 0.
    phi_o = C_o/(H_o+(C^T mu)_o); solve C phi(mu)=0 by damped Newton on the convex
    dual Psi(mu) = -sum_o C_o log(H_o+(C^T mu)_o).  Positivity is automatic.

    C_o, H_o are SCALE-NORMALISED by the total usage before the solve (phi is
    invariant to this joint scaling) -- without it the count-vs-rate scale gap
    (C_o~1e5, H_o~1e7) makes the dual Hessian ~1e-9 and the Newton steps blow up.
    Warm-started from _MU_CACHE (consecutive L-BFGS pi-evals share a near solution)."""
    active = C_o > 0
    scale = float(C_o[active].sum()) if active.any() else 1.0
    Ca = np.where(active, C_o / scale, 0.0)
    He = H_o / scale
    nr = Cmat.shape[0]
    warm = _MU_CACHE["mu"] if use_cache else None
    mu = warm.copy() if (warm is not None and warm.shape[0] == nr) else np.zeros(nr)

    def den(m):
        return He + Cmat.T @ m

    def Psi(m):
        d = den(m)
        if np.any(d[active] <= 0):
            return np.inf
        return float(-(Ca * np.log(np.maximum(d, 1e-300))).sum())

    if not np.isfinite(Psi(mu)):                         # warm start infeasible: reset
        mu = np.zeros(nr)
    for _ in range(iters):
        d = np.maximum(den(mu), 1e-100)
        phi = np.where(active, Ca / d, 0.0)
        R = Cmat @ phi                                   # residual (want 0)
        if np.linalg.norm(R) < tol * (np.linalg.norm(phi) + 1e-12):
            break                                        # relative-violation stop
        w = np.where(active, Ca / (d * d), 0.0)
        Hs = (Cmat * w[None, :]) @ Cmat.T
        Hs[np.diag_indices(nr)] += ridge
        try:
            step = np.linalg.solve(Hs, R)                # min Psi: mu += Hs^{-1} R
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Hs, R, rcond=None)[0]
        f0 = Psi(mu)
        gdot = -(R @ step)                               # grad Psi . step  (<= 0)
        a = 1.0
        for _ in range(50):
            ft = Psi(mu + a * step)
            if np.isfinite(ft) and ft <= f0 + 1e-4 * a * gdot:
                break
            a *= 0.5
        mu = mu + a * step
    if use_cache:
        _MU_CACHE["mu"] = mu.copy()
    d = np.maximum(den(mu), 1e-100)
    return np.where(active, C_o / (scale * d), 0.0)     # unscale: phi = C_o/(H_o+C^T mu_raw)


def mstep_flux_free(C_o, H_o):
    return np.where(H_o > 0, C_o / np.maximum(H_o, 1e-300), 0.0)


# ============================================================================
# coupled-exchangeable stationary parametrisation + profiled pi-step
# ============================================================================
def _triu(n):
    return np.triu_indices(n)


def pi_from_theta(theta, n, triu, floor=1e-9):
    M = np.zeros((n, n))
    M[triu] = theta
    d = np.diag(M).copy()
    M = M + M.T
    M[np.diag_indices(n)] = d
    M = M - M.max()
    p = np.exp(M)
    p = p / p.sum()
    p = np.maximum(p, floor)                   # keep pi numerically sane (1/pi bounded)
    p = p / p.sum()
    return p.reshape(n * n)


def theta_from_pi(pi, n, triu):
    M = np.log(np.clip(pi.reshape(n, n), 1e-12, None))
    return M[triu]


def _profile_negm(theta, n, triu, C_o, Nrow, oid_off, src_off, pre,
                  n_orbits, model):
    pi = pi_from_theta(theta, n, triu)
    H_o = np.bincount(oid_off, weights=T_glob[src_off] / np.maximum(pi[src_off], 1e-300),
                      minlength=n_orbits)
    if model == "lumpable":
        Cmat = build_Cmat(pre, pi, n_orbits)
        phi = mstep_flux_lumpable(C_o, H_o, Cmat)
    else:
        phi = mstep_flux_free(C_o, H_o)
    act = C_o > 0
    val = float((C_o[act] * np.log(np.maximum(phi[act], 1e-300)) - H_o[act] * phi[act]).sum())
    val -= float((Nrow * np.log(np.clip(pi, 1e-300, None))).sum())
    return -val


# module-global handoff for T within the profiled objective (kept simple + fast)
T_glob = None


# ============================================================================
# the fit
# ============================================================================
def fit(npair, tau, n, model="lumpable", init="product", n_em=40,
        m_maxiter=40, tol_frac=1e-8, verbose=False, mu_renewal=1.0,
        interior_eps=1e-2):
    """EXACT-lumpability HR-EM fit.  model in {lumpable, synchronized};
    init in {product, renewal}."""
    global T_glob
    _MU_CACHE["mu"] = None                            # fresh warm-start cache per fit
    n2 = n * n
    orbit_id, n_orbits, is_single = build_orbits_n(n)
    off, src_off = _off_index(n)
    oid_off = orbit_id[off]
    pre = precompute_lump_constraint(orbit_id, n)
    triu = _triu(n)

    rho = empirical_pi_n(npair, n).reshape(n, n).sum(1)
    rho = rho / rho.sum()
    if init == "product":
        pi = np.outer(rho, rho).reshape(n2)
        # strictly-interior small symmetric flux (incl. double-transition orbits) so
        # the HR E-step sees nonzero usage and is not pinned at the absorbing 0 flux.
        F = interior_eps * np.sqrt(pi[:, None] * pi[None, :])
        F[~off] = 0.0
        Q = Q_from_flux_n(F, pi)
    elif init == "renewal":
        pi = empirical_pi_n(npair, n)               # coupled empirical joint
        Q = mu_renewal * np.tile(pi, (n2, 1))       # Q_xy = mu pi_y (renewal, lumpable)
        np.fill_diagonal(Q, 0.0)
        Q[np.diag_indices(n2)] = -Q.sum(1)
    else:
        raise ValueError(init)

    hist = []
    F = None
    phi = None
    Ntot = float(npair.sum())
    for it in range(n_em):
        N, T = estep_n(Q, pi, tau, npair)
        T_glob = T
        C_o = np.bincount(oid_off, weights=N[off], minlength=n_orbits)
        Nrow = N.sum(1)
        theta0 = theta_from_pi(pi, n, triu)
        bounds = [(-20.0, 20.0)] * theta0.shape[0]      # keep pi in a sane range
        res = minimize(_profile_negm, theta0, method="L-BFGS-B", bounds=bounds,
                       args=(n, triu, C_o, Nrow, oid_off, src_off, pre, n_orbits, model),
                       options={"maxiter": m_maxiter, "ftol": 1e-12, "gtol": 1e-9})
        pi = pi_from_theta(res.x, n, triu)
        # final flux at the accepted pi
        H_o = np.bincount(oid_off, weights=T[src_off] / np.maximum(pi[src_off], 1e-300),
                          minlength=n_orbits)
        if model == "lumpable":
            phi = mstep_flux_lumpable(C_o, H_o, build_Cmat(pre, pi, n_orbits))
        else:
            phi = mstep_flux_free(C_o, H_o)
        F = flux_from_orbit_n(phi, orbit_id)
        Q = Q_from_flux_n(F, pi)
        ll = loglik_n(Q, pi, tau, npair)
        hist.append(ll)
        if verbose and (it < 3 or it % 5 == 0 or it == n_em - 1):
            print(f"      [{model}/{init}] it {it:2d} LL={ll:.2f} "
                  f"MI={mutual_information(pi, n):.4f}", flush=True)
        if it > 4 and abs(hist[-1] - hist[-2]) < tol_frac * Ntot:
            break
    lumpres = lumpability_residual(Q, n)
    diffs = np.diff(hist) if len(hist) > 1 else np.array([0.0])
    return dict(Q=Q, pi=pi, F=F, phi=phi, hist=[float(h) for h in hist],
                ll=float(hist[-1]), mi=mutual_information(pi, n),
                lump_resid=float(lumpres),
                monotone=bool(diffs.min() > -1e-6 * max(Ntot, 1.0)),
                max_decrease=float(max(0.0, -diffs.min())) if len(diffs) else 0.0)


# ============================================================================
# backgrounds, couplings, and data generators
# ============================================================================
def modes(r, u):
    """(W, lam desc, psi u-orthonormal columns).  Reuses lumpable_tangent_rank."""
    return ltr._modes(r, u)


def generic_background(seed, n=4):
    rng = np.random.RandomState(seed)
    r = np.abs(rng.randn(n, n)) + 0.6
    r = (r + r.T) / 2
    np.fill_diagonal(r, 0.0)
    u = np.abs(rng.randn(n)) + 0.5
    u = u / u.sum()
    return r, u


def hky_background(kappa=4.0, u=(0.1, 0.2, 0.3, 0.4)):
    return ltr._dna_r(kappa), np.array(u, float)


def coupling_dir(psi, u, a, b):
    """Zero-marginal exchangeable stationary-coupling direction for eigen-modes a,b."""
    da = u * psi[:, a]
    db = u * psi[:, b]
    D = np.outer(da, db)
    if a != b:
        D = D + np.outer(db, da)
    return D


def scale_for_mi(u, D, n, target=0.2):
    base = np.outer(u, u)
    neg = D < 0
    smax = 0.98 * float(np.min(base[neg] / (-D[neg]))) if neg.any() else 20.0

    def mi(s):
        return mutual_information((base + s * D).ravel(), n)

    if mi(smax) < target:
        s = 0.9 * smax
        return s, mi(s)
    lo, hi = 0.0, smax
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mi(mid) < target:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return s, mi(s)


def _endpoints(Q, pi, tau, Ntot):
    n2 = Q.shape[0]
    lam, U, Uinv = eig_rev(Q, np.clip(pi, 1e-12, None))
    npair = np.zeros((n2, n2, len(tau)))
    for t in range(len(tau)):
        P = (U * np.exp(lam * tau[t])[None, :]) @ Uinv
        J = pi[:, None] * np.clip(P, 0, None)
        J = J / J.sum()
        npair[:, :, t] = Ntot * J
    return npair


def _rate_normalise(Q, pi):
    rate = -float((pi * np.diag(Q)).sum())
    return Q / rate


def gen_conditional(pi, r, n, tau, Ntot):
    """CONDITIONAL data: Metropolis-sqrt SINGLE-transition chain reversible w.r.t.
    the coupled pi (proposal exchangeability S=r, acceptance sqrt(pi_dest/pi_src)).
    One coordinate changes at a time -- NO concerted double substitutions."""
    n2 = n * n
    piM = pi.reshape(n, n)
    Q = np.zeros((n2, n2))
    for i in range(n):
        for j in range(n):
            x = i * n + j
            for k in range(n):
                if k != i:
                    Q[x, k * n + j] = r[i, k] * np.sqrt(piM[k, j] / piM[i, j])
            for l in range(n):
                if l != j:
                    Q[x, i * n + l] = r[j, l] * np.sqrt(piM[i, l] / piM[i, j])
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(n2)] = -Q.sum(1)
    Q = _rate_normalise(Q, pi)
    return _endpoints(Q, pi, tau, Ntot), Q


def gen_renewal(pi, n, tau, Ntot, mu=1.0):
    """LUMPABLE data: joint-renewal chain Q_xy = mu pi_y (jump to a fresh draw from
    the coupled joint pi).  Two-sided lumpable; carries concerted double-sub signal."""
    n2 = n * n
    Q = mu * np.tile(pi, (n2, 1))
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(n2)] = -Q.sum(1)
    Q = _rate_normalise(Q, pi)
    return _endpoints(Q, pi, tau, Ntot), Q


def resonant_fraction(D, r, u, n=4, tol=1e-6):
    """Exact fraction of coupling energy in the RESONANT eigen-mode span
    span{psi_a (x) psi_b: lam_a=lam_b} = the first-order reachable tangent (proven
    equal in analysis/lumpable_product_trap.md).  In eigen-coordinates c=psi^T D psi
    (u-orthonormal psi), = sum_{lam_a=lam_b} c_ab^2 / sum_{a,b>=1} c_ab^2.  Exactly 1
    for an agreement mode (a,a), exactly 0 for a cross mode (a,b), lam_a!=lam_b."""
    W, lam, psi = modes(r, u)
    c = psi.T @ D @ psi
    tot = res = 0.0
    for a in range(1, n):
        for b in range(1, n):
            tot += c[a, b] ** 2
            if abs(lam[a] - lam[b]) < tol:
                res += c[a, b] ** 2
    return float(res / max(tot, 1e-30))


def select_directions(r, u, n=4, target=0.20):
    """Per background: delta_IN = the agreement mode (a,a) with the highest
    positivity-feasible MI (resonant); delta_OUT = the cross mode (a,b), a<b, with
    the highest feasible MI (non-resonant, since the spectrum is distinct).  Each
    scaled to `target` MI (capped by positivity)."""
    W, lam, psi = modes(r, u)
    ins = [(a, a) for a in range(1, n)]
    outs = [(a, b) for a in range(1, n) for b in range(a + 1, n)]
    best_in = max(ins, key=lambda ab: scale_for_mi(u, coupling_dir(psi, u, *ab), n, target)[1])
    best_out = max(outs, key=lambda ab: scale_for_mi(u, coupling_dir(psi, u, *ab), n, target)[1])
    out = {}
    for lab, ab in (("IN", best_in), ("OUT", best_out)):
        D = coupling_dir(psi, u, *ab)
        s, mi = scale_for_mi(u, D, n, target)
        pi = (np.outer(u, u) + s * D).ravel()
        out[lab] = dict(modes=ab, D=D, s=s, mi_true=mi, pi=pi,
                        resonant_frac=resonant_fraction(D, r, u, n))
    return out


# ============================================================================
# drivers: verification + trap
# ============================================================================
def run_verification(tau, Ntot, seed=0):
    print("\n# ============ VERIFICATION ============", flush=True)
    n = 4
    out = {}

    # ---- V1: recover a KNOWN lumpable chain (joint renewal, coupled pi) ----
    r, u = generic_background(seed, n)
    sel = select_directions(r, u, n, target=0.25)
    pi_true = sel["IN"]["pi"]
    mi_true = sel["IN"]["mi_true"]
    npair, Qtrue = gen_renewal(pi_true, n, tau, Ntot)
    res = fit(npair, tau, n, model="lumpable", init="renewal", n_em=60)
    lr_true = lumpability_residual(Qtrue, n)
    print(f"# V1 lumpable-recovery: MI_true={mi_true:.4f} MI_fit={res['mi']:.4f} "
          f"| lump_resid(fit)={res['lump_resid']:.2e} lump_resid(true)={lr_true:.2e} "
          f"| monotone={res['monotone']} maxdec={res['max_decrease']:.2e}", flush=True)
    out["V1_lumpable_recovery"] = dict(mi_true=mi_true, mi_fit=res["mi"],
                                       lump_resid_fit=res["lump_resid"],
                                       lump_resid_true=lr_true,
                                       monotone=res["monotone"],
                                       max_decrease=res["max_decrease"],
                                       ll=res["ll"])

    # ---- V2: Synchronized recovers a KNOWN general reversible chain ----
    orbit_id, n_orbits, is_single = build_orbits_n(n)
    rng = np.random.RandomState(seed + 7)
    phi_g = np.abs(rng.randn(n_orbits)) * 0.5 + 0.05   # random symmetric-exchangeable flux
    pi_g = select_directions(r, u, n, target=0.15)["OUT"]["pi"]   # coupled stationary
    F_g = flux_from_orbit_n(phi_g, orbit_id)
    Q_g = Q_from_flux_n(F_g, pi_g)
    Q_g = _rate_normalise(Q_g, pi_g)
    npair2 = _endpoints(Q_g, pi_g, tau, Ntot)
    mi_true2 = mutual_information(pi_g, n)
    res2 = fit(npair2, tau, n, model="synchronized", init="renewal", n_em=60)
    ll_true2 = loglik_n(Q_g, pi_g, tau, npair2)
    print(f"# V2 synchronized-recovery: MI_true={mi_true2:.4f} MI_fit={res2['mi']:.4f} "
          f"| LL_true/N={ll_true2/npair2.sum():.5f} LL_fit/N={res2['ll']/npair2.sum():.5f} "
          f"| monotone={res2['monotone']} maxdec={res2['max_decrease']:.2e}", flush=True)
    out["V2_synchronized_recovery"] = dict(mi_true=mi_true2, mi_fit=res2["mi"],
                                           ll_true_per=ll_true2 / npair2.sum(),
                                           ll_fit_per=res2["ll"] / npair2.sum(),
                                           monotone=res2["monotone"],
                                           max_decrease=res2["max_decrease"])
    return out


def run_trap(tau, Ntot, mi_target=0.25, save_dir="results/lumpable_trap"):
    print("\n# ============ TRAP / RENEWAL-INIT EXPERIMENT ============", flush=True)
    n = 4
    os.makedirs(save_dir, exist_ok=True)
    backgrounds = {
        "generic": generic_background(0, n),
        "hky85skew": hky_background(4.0, (0.1, 0.2, 0.3, 0.4)),
    }
    cells = []
    for bg_name, (r, u) in backgrounds.items():
        W, lam, psi = modes(r, u)
        sel = select_directions(r, u, n, target=mi_target)
        for dir_name in ("IN", "OUT"):
            D = sel[dir_name]["D"]
            ov = sel[dir_name]["resonant_frac"]
            mi_true = sel[dir_name]["mi_true"]
            pi_true = sel[dir_name]["pi"]
            print(f"\n# --- bg={bg_name} dir={dir_name} modes={sel[dir_name]['modes']} "
                  f"resonant_frac={ov:.2e} MI_true={mi_true:.4f} "
                  f"eig(W)={np.round(lam,3)} ---", flush=True)
            data = {
                "conditional": gen_conditional(pi_true, r, n, tau, Ntot)[0],
                "lumpable": gen_renewal(pi_true, n, tau, Ntot)[0],
            }
            for data_name, npair in data.items():
                for init in ("product", "renewal"):
                    t0 = time.time()
                    res = fit(npair, tau, n, model="lumpable", init=init, n_em=60)
                    dt = time.time() - t0
                    row = dict(bg=bg_name, direction=dir_name, data=data_name,
                               init=init, resonant_frac=ov, mi_true=mi_true,
                               mi_fit=res["mi"], ll_per=res["ll"] / npair.sum(),
                               lump_resid=res["lump_resid"], monotone=res["monotone"],
                               max_decrease=res["max_decrease"], n_iter=len(res["hist"]),
                               time_s=dt)
                    cells.append(row)
                    print(f"#   {data_name:<11} {init:<8} MI_fit={res['mi']:.4f} "
                          f"LL/N={row['ll_per']:.5f} lumpres={res['lump_resid']:.1e} "
                          f"mono={res['monotone']} ({dt:.1f}s)", flush=True)
                    tag = f"{bg_name}_{dir_name}_{data_name}_{init}"
                    np.savez(os.path.join(save_dir, f"fit_{tag}.npz"),
                             Q=res["Q"], pi=res["pi"], F=res["F"],
                             mi_fit=res["mi"], mi_true=mi_true, ll=res["ll"])
    return cells


def print_table(cells):
    print("\n# ==== MI / LL table ====", flush=True)
    hdr = (f"# {'bg':<10}{'dir':<5}{'data':<12}{'init':<9}"
           f"{'MI_true':>9}{'MI_fit':>9}{'LL/N':>11}{'lumpres':>10}")
    print(hdr, flush=True)
    for c in cells:
        print(f"# {c['bg']:<10}{c['direction']:<5}{c['data']:<12}{c['init']:<9}"
              f"{c['mi_true']:>9.4f}{c['mi_fit']:>9.4f}{c['ll_per']:>11.5f}"
              f"{c['lump_resid']:>10.1e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--trap", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ntot", type=float, default=4e5)
    ap.add_argument("--mi-target", type=float, default=0.20)
    ap.add_argument("--out", default="results/lumpable_trap/results.json")
    args = ap.parse_args()

    tau = np.array([0.15, 0.4, 0.8, 1.5])
    Ntot = args.ntot
    result = {"tau": tau.tolist(), "ntot_per_bin": Ntot}
    t0 = time.time()
    if args.verify or args.all:
        result["verification"] = run_verification(tau, Ntot)
    if args.trap or args.all:
        cells = run_trap(tau, Ntot, mi_target=args.mi_target)
        result["trap_cells"] = cells
        print_table(cells)
    print(f"\n# total {time.time()-t0:.1f}s", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2, default=float)
    print(f"# wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
