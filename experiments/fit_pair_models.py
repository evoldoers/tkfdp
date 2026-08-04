#!/usr/bin/env python3
"""Fit the four 2-component pair-substitution models of paper 2b Sec. 3 to a
time-binned coupled count tensor, by HR-EM.

Models (all reversible + component-exchangeable, fit from n(i,j;k,l;t)):
  Synchronized : general pair chain, simultaneous transitions allowed (all Klein
                 orbits) -- most parameters.
  Coupled      : general SINGLE-transition chain (CTBN); one component changes at
                 a time, exchangeabilities free (context-dependent rates).
  Lumpable     : single-transition + strong lumpability of either coordinate
                 (rational-dual flux update, em_lumpable_reversible_ctmc.tex eq 3).
  metropolis_* : single-transition, one SHARED single-site exchangeability S with a
                 Metropolis-family acceptance kernel Q_xy = S_xy f(pi_x,pi_y):
                 sqrt f=sqrt(pi_y/pi_x), gtr f=pi_y, barker f=pi_y/(pi_x+pi_y),
                 hastings f=min(1,pi_y/pi_x). Fewest parameters (190).

State space: ordered pairs x=(i,j) in [20]^2 (400 states); index x=i*20+j.
Reversible generator Q_xy = F_xy/pi_x with F symmetric on Klein-4 orbits
(group V4 = {e, R:(x,y)->(y,x), E:swap components, RE}).

E-step (exact Holmes-Rubin, tkfdp.permfield.hr): for each time bin t, the count
slice n(.,.,t) is the joint endpoint weight; HR returns expected transition usage
N_xy and dwell T_x summed over the counts. M-step: orbit-aggregate
C_o=sum N_xy, H_o=sum T_x/pi_x; flux phi_o = C_o/H_o (Synchronized over all
orbits, Coupled over single-transition orbits only). The stationary pi_ij is
ML-FIT by default (symmetric pi_ij=pi_ji): closed-form for the flux/GTR forms,
a damped gradient step for the sqrt/barker/hastings kernels; --fix-pi holds it at
the empirical pair frequency. Reports observed-data log-likelihood
sum_{x,y,t} n log expm(Q*t)[x,y] and free-parameter count per model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "src")
from tkfdp.permfield.hr import bridge, eig_rev              # noqa: E402

NA = 20
NS = NA * NA                                                # 400 pair states


# ---------- counts ----------
def load_counts(path):
    d = np.load(f"{path}/counts.npz", allow_pickle=True)
    npair = d["n_pair"].astype(float).reshape(NS, NS, -1)   # (400,400,T)
    tau = d["tau_centers"].astype(float)
    return npair, tau, str(d["alphabet"])


def n_parts(path):
    import os
    i = 0
    while os.path.exists(f"{path}/part_{i}.npz"):
        i += 1
    return i


def load_parts(path, ids):
    """Sum the n_pair count tensors over the given part shards (disjoint family
    splits). Returns (npair (400,400,T), tau, alphabet)."""
    npair = None
    for i in ids:
        d = np.load(f"{path}/part_{i}.npz", allow_pickle=True)
        p = d["n_pair"].astype(float).reshape(NS, NS, -1)
        npair = p if npair is None else npair + p
    d0 = np.load(f"{path}/counts.npz", allow_pickle=True)
    return npair, d0["tau_centers"].astype(float), str(d0["alphabet"])


def empirical_pi(npair):
    tot = npair.sum(2)                                      # (400,400) over time
    occ = 0.5 * (tot.sum(1) + tot.sum(0))                  # start+end occupancy
    pi = occ.reshape(NA, NA)
    pi = 0.5 * (pi + pi.T)                                  # component-exchangeable
    pi = pi / pi.sum()
    return pi.reshape(NS)


# ---------- Klein-4 orbit machinery ----------
def build_orbits():
    """orbit_id (400,400) int over off-diagonal directed transitions; -1 on the
    diagonal. is_single (400,400) bool. Returns (orbit_id, n_orbits, is_single)."""
    ii, jj = np.divmod(np.arange(NS), NA)                   # (i,j) of each state
    orbit_id = np.full((NS, NS), -1, np.int64)
    canon = {}
    nxt = 0
    for x in range(NS):
        i, j = ii[x], jj[x]
        for y in range(NS):
            if x == y:
                continue
            k, l = ii[y], jj[y]
            # Klein-4 images of directed transition (i,j;k,l)
            imgs = [(i, j, k, l), (k, l, i, j), (j, i, l, k), (l, k, j, i)]
            key = min(imgs)
            o = canon.get(key)
            if o is None:
                o = nxt; canon[key] = o; nxt += 1
            orbit_id[x, y] = o
    is_single = np.zeros((NS, NS), bool)
    ch1 = ii[:, None] != ii[None, :]
    ch2 = jj[:, None] != jj[None, :]
    is_single = ch1 ^ ch2                                   # exactly one component changes
    return orbit_id, nxt, is_single


# ---------- reversible generator from orbit fluxes ----------
def Q_from_flux(F, pi):
    Q = F / np.maximum(pi[:, None], 1e-300)
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(NS)] = -Q.sum(1)
    return Q


# ---------- HR E-step over the 400-state pair chain ----------
def estep(Q, pi, tau, npair):
    eig = eig_rev(Q, np.clip(pi, 1e-12, None))
    N = np.zeros((NS, NS)); T = np.zeros(NS)
    for t in range(tau.shape[0]):
        edge = npair[:, :, t]
        if edge.sum() == 0:
            continue
        Tt, Nt, _ = bridge(Q, pi, tau[t], edge, want_N=True, eig=eig)
        N += Nt; T += Tt
    return N, T


def mstep_flux(N, T, pi, orbit_id, n_orbits, mask):
    """phi_o = C_o / H_o(pi); C_o = sum_{orbit} N_xy, H_o = sum_{orbit} T_x/pi_x.
    `mask` (400,400 bool) restricts which directed transitions may carry flux
    (all off-diagonal for Synchronized; single-transition for Coupled)."""
    off = ~np.eye(NS, dtype=bool)
    use = off & mask
    oid = orbit_id[use]
    Cx = N[use]
    Hx = (T[:, None] / np.maximum(pi[:, None], 1e-300) * np.ones((1, NS)))[use]
    C_o = np.bincount(oid, weights=Cx, minlength=n_orbits)
    H_o = np.bincount(oid, weights=Hx, minlength=n_orbits)
    phi = np.where(H_o > 0, C_o / np.maximum(H_o, 1e-300), 0.0)
    F = np.zeros((NS, NS))
    F[use] = phi[oid]
    return F


# (the Metropolis-family single-site-exchangeability chains -- sqrt/gtr/barker/
# hastings -- are defined together near fit_model as mstep_metropolis.)


def build_lump_rows(orbit_id):
    """Lumpability constraint rows (i,j,k), i!=k: (Bphi)_ijk = sum_l F_ij,kl.
    Returns (row_i, row_j, row_k, row_orbs) with row_orbs (n_rows, NA) the orbit
    of each (ij->kl), l=0..NA-1."""
    ri, rj, rk, ro = [], [], [], []
    for i in range(NA):
        for j in range(NA):
            x = i * NA + j
            for k in range(NA):
                if i == k:
                    continue
                ri.append(i); rj.append(j); rk.append(k)
                ro.append([orbit_id[x, k * NA + l] for l in range(NA)])
    return np.array(ri), np.array(rj), np.array(rk), np.array(ro)


def _perp(v, pr, ik, nik):
    """Project out range(D): for each edge (i,k) remove the pi-weighted mean of v
    over the spectator j. range(D)'s basis vectors have disjoint (i,k) supports so
    this is the exact orthogonal projector onto range(D)^perp = null(D^T)."""
    num = np.bincount(ik, weights=pr * v, minlength=nik)
    den = np.bincount(ik, weights=pr * pr, minlength=nik)
    coeff = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
    return v - pr * coeff[ik]


def mstep_lumpable(N, T, pi, orbit_id, n_orbits, rows, piM, rho,
                   outer=70, inner=10, cg_iter=400, rho0=1e11, growth=1.5,
                   rho_max=1e18, verbose=False):
    """Lumpable M-step (em_lumpable_reversible_ctmc.tex): maximise the expected
    complete-data log-likelihood over the general (all-orbit) reversible-exchangeable
    flux subject to strong lumpability B*phi = D(pi)*g. With g free, range(D)'s
    orthogonal basis (disjoint (i,k) supports) reduces the constraint to
    c(phi) = P_perp(B*phi) = 0. Solved by an AUGMENTED LAGRANGIAN in log-flux
    psi = log phi (so phi>0 with no boundary): inner max by a diag-preconditioned
    Gauss-Newton (step-capped, backtracked), outer multiplier update mu += rho*c.

    Lumpability is FEASIBLE (c can be driven to 0), but the flux/count scale
    mismatch (fluxes ~1e-4, counts ~1e6) makes it ill-conditioned; this reaches a
    relative violation ~0.02 in practical time (machine-zero needs far more work).
    Empirically the coupled process is NEARLY lumpable, so the Lumpable fit lands
    close to Synchronized in log-likelihood."""
    ri, rj, rk, ro = rows
    off = ~np.eye(NS, dtype=bool); oid = orbit_id[off]
    C_o = np.bincount(oid, weights=N[off], minlength=n_orbits)
    H_o = np.bincount(oid, weights=(T[:, None] / np.maximum(pi[:, None], 1e-300)
                                    * np.ones((1, NS)))[off], minlength=n_orbits)
    pr = piM[ri, rj] / np.maximum(rho[ri], 1e-300)
    ik = ri * NA + rk; nik = NA * NA
    BtB_diag = np.bincount(ro.ravel(), minlength=n_orbits).astype(float)  # rows per orbit

    def Bf(v): return v[ro].sum(1)                          # B phi (rows)
    def BT(v):
        o = np.zeros(n_orbits); np.add.at(o, ro.ravel(), np.repeat(v, NA)); return o
    def Pp(v): return _perp(v, pr, ik, nik)

    psi = np.log(np.maximum(C_o / np.maximum(H_o, 1e-9), 1e-12))
    mu = np.zeros(len(ri))
    rho_pen = rho0
    for _o in range(outer):
        for _ in range(inner):                             # inner: augmented-Lag max
            phi = np.exp(psi); c = Pp(Bf(phi))
            grad = (C_o - H_o * phi) - phi * BT(Pp(mu + rho_pen * c))
            Hd = np.maximum(H_o * phi + rho_pen * phi * phi * BtB_diag, 1e-30)
            Minv = 1.0 / Hd

            def Hv(v):
                return H_o * phi * v + rho_pen * phi * BT(Pp(Bf(phi * v)))

            d = np.zeros_like(grad); r = grad.copy(); z = Minv * r; p = z.copy()
            rz = r @ z; g0 = grad @ grad
            for _ in range(cg_iter):
                Hp = Hv(p); a = rz / max(p @ Hp, 1e-30)
                d += a * p; r -= a * Hp
                if r @ r < 1e-16 * g0:
                    break
                z = Minv * r; rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
            d = d * min(1.0, 3.0 / max(np.abs(d).max(), 1e-30))  # cap log-step

            def AL(ps):
                ph = np.exp(ps); cc = Pp(Bf(ph))
                return (C_o * ps - H_o * ph).sum() - mu @ cc - 0.5 * rho_pen * (cc @ cc)

            f0 = AL(psi); al = 1.0
            for _ in range(30):
                if np.isfinite(AL(psi + al * d)) and AL(psi + al * d) > f0:
                    break
                al *= 0.5
            psi = psi + al * d
        mu = mu + rho_pen * Pp(Bf(np.exp(psi)))
        rho_pen = min(rho_pen * growth, rho_max)           # penalty continuation
        if verbose and (_o % 10 == 0 or _o == outer - 1):
            phi = np.exp(psi); rv = np.linalg.norm(Pp(Bf(phi))) / max(
                np.linalg.norm(Bf(phi)), 1e-30)
            print(f"      [lumpable ALM] outer {_o:3d} relviol={rv:.2e}", flush=True)
    phi = np.exp(psi)
    F = np.zeros((NS, NS)); F[off] = phi[oid]
    return F


def loglik(Q, pi, tau, npair):
    lam, U, Uinv = eig_rev(Q, np.clip(pi, 1e-12, None))
    ll = 0.0
    for t in range(tau.shape[0]):
        edge = npair[:, :, t]
        if edge.sum() == 0:
            continue
        P = (U * np.exp(lam * tau[t])[None, :]) @ Uinv
        ll += float((edge * np.log(np.clip(P, 1e-300, None))).sum())
    return ll


# ---------- fit ----------
def mstep_pi_lumpable(N, T, F, pi, rows, n_total, outer=30, inner=80):
    """Constrained stationary CM-step for the (half-)lumpable models
    (em_lumpable_reversible_ctmc.tex, eq. pistep): maximise the expected complete-data
    stationary objective Q_pi(pi) = -sum_x N_{x+} log pi_x - sum_x T_x r_x/pi_x
    (r_x = sum_y F_xy, F fixed) subject to the lumpability feasibility
    P_perp^{(pi)}(B phi) = 0, by an augmented Lagrangian with multipliers.  BOTH sides are
    rescaled to O(1) -- per-count objective, relative-violation constraint -- because the
    flux (~1e-4) vs count (~1e8) scale gap otherwise makes mu.c and rho|c|^2 negligible and
    the constraint invisible (that is the whole reason the old GTR mstep_pi drifted off the
    lumpable manifold and limit-cycled).  Replaces mstep_pi for model in {lumpable}."""
    ri, rj, rk, ro = rows
    ik = ri * NA + rk; nik = NA * NA
    Bphi = F.reshape(NA, NA, NA, NA).sum(3)[ri, rj, rk]
    Bn = max(float(np.linalg.norm(Bphi)), 1e-30)
    N_out = N.sum(1); Tr = T * F.sum(1)

    def _sympi(u):
        M = 0.5 * (u.reshape(NA, NA) + u.reshape(NA, NA).T)
        M = M - M.max(); p = np.exp(M); return (p / p.sum()).reshape(NS)

    def cons(p):
        pm = p.reshape(NA, NA); rho = pm.sum(1)
        pr = pm[ri, rj] / np.maximum(rho[ri], 1e-300)
        return _perp(Bphi, pr, ik, nik) / Bn

    def negAL(u, mu, rp):
        p = np.clip(_sympi(u), 1e-12, None)
        Qpi = (-(N_out * np.log(p)).sum() - (Tr / p).sum()) / n_total
        c = cons(p)
        return -(Qpi - mu @ c - 0.5 * rp * (c @ c))

    u = np.log(np.clip(pi, 1e-12, None)); mu = np.zeros(len(ri)); rp = 1.0
    for _o in range(outer):
        u = minimize(negAL, u, args=(mu, rp), method="L-BFGS-B",
                     options=dict(maxiter=inner)).x
        mu = mu + rp * cons(np.clip(_sympi(u), 1e-12, None)); rp = min(rp * 2.0, 1e6)
    return _sympi(u)


def mstep_pi(F, pi, N, T):
    """ML stationary update for a reversible-exchangeable chain, given the fitted
    symmetric flux F and HR stats: with exchangeability s = F/(pi_x pi_y),
    pi_y ∝ N_{+y} / sum_x T_x s_xy. Symmetrise (pi_ij=pi_ji) and renormalise."""
    s = F / np.maximum(pi[:, None] * pi[None, :], 1e-300)   # symmetric exchangeability
    num = N.sum(0)                                          # N_{+y} in-transitions to y
    den = (T[:, None] * s).sum(0)                           # sum_x T_x s_xy
    pi_new = np.where(den > 0, num / np.maximum(den, 1e-300), pi)
    pm = pi_new.reshape(NA, NA); pm = 0.5 * (pm + pm.T)     # component-exchangeable
    pm = np.maximum(pm, 1e-12); pm /= pm.sum()
    return pm.reshape(NS), s


# ---------- Metropolis family: single-transition, shared exchangeability S,
# reversible Q_xy = S_xy * f(pi_x, pi_y). Four acceptance kernels f. ----------
def _met_f(src, dest, kernel):
    """Acceptance f(pi_src, pi_dest) = Q/S (elementwise)."""
    if kernel == "sqrt":
        return np.sqrt(dest / np.maximum(src, 1e-300))
    if kernel == "gtr":
        return dest
    if kernel == "barker":
        return dest / np.maximum(src + dest, 1e-300)
    if kernel == "hastings":
        return np.minimum(1.0, dest / np.maximum(src, 1e-300))
    raise ValueError(kernel)


def _met_sym(src, dest, kernel):
    """Symmetric flux kernel sym(pi_x,pi_y) = pi_x f(pi_x,pi_y) = F/S (symmetric)."""
    if kernel == "sqrt":
        return np.sqrt(src * dest)
    if kernel == "gtr":
        return src * dest
    if kernel == "barker":
        return src * dest / np.maximum(src + dest, 1e-300)
    if kernel == "hastings":
        return np.minimum(src, dest)
    raise ValueError(kernel)


def _met_dlogf(src, dest, kernel):
    """(d log f / d log-none... ) return (df/ds, df/dd)-as-dlog: d log f / d pi_src,
    d log f / d pi_dest, elementwise."""
    if kernel == "sqrt":
        return -0.5 / np.maximum(src, 1e-300), 0.5 / np.maximum(dest, 1e-300)
    if kernel == "gtr":
        return np.zeros_like(src), 1.0 / np.maximum(dest, 1e-300)
    if kernel == "barker":
        sd = np.maximum(src + dest, 1e-300)
        return -1.0 / sd, 1.0 / np.maximum(dest, 1e-300) - 1.0 / sd
    if kernel == "hastings":
        m = dest < src
        return np.where(m, -1.0 / np.maximum(src, 1e-300), 0.0), \
            np.where(m, 1.0 / np.maximum(dest, 1e-300), 0.0)
    raise ValueError(kernel)


def _met_build_from_S(S, piM, kernel, fun):
    """Build a single-transition (A^2,A^2) matrix with block value S_ik * fun(pi_ij,pi_kj)."""
    M4 = np.zeros((NA, NA, NA, NA))
    for j in range(NA):                                    # comp 1: (i,j)->(k,j)
        M4[:, j, :, j] = S * fun(piM[:, j][:, None], piM[:, j][None, :], kernel)
    for i in range(NA):                                    # comp 2: (i,j)->(i,l)
        M4[i, :, i, :] += S * fun(piM[i, :][:, None], piM[i, :][None, :], kernel)
    return M4.reshape(NS, NS)


def mstep_metropolis(N, T, pi, kernel):
    """S-step for the Metropolis family (pi fixed within this step). Shared
    exchangeability S_ab = usage / exposure, exposure = sum_spectator T * f(kernel).
    Returns the symmetric flux F = S * sym(pi_x,pi_y)."""
    piM = np.maximum(pi.reshape(NA, NA), 1e-300); T2 = T.reshape(NA, NA)
    N4 = N.reshape(NA, NA, NA, NA)
    Cdir = np.einsum("ijkj->ik", N4) + np.einsum("ijil->jl", N4)
    Cnum = Cdir + Cdir.T
    f1 = _met_f(piM[:, None, :], piM[None, :, :], kernel)   # (a,b,j) comp1 exposure
    H1 = (T2[:, None, :] * f1).sum(2)
    f2 = _met_f(piM[:, :, None], piM[:, None, :], kernel)   # (i,a,b) comp2 exposure
    H2 = (T2[:, :, None] * f2).sum(0)
    Hden = (H1 + H2); Hden = Hden + Hden.T
    S = np.where(Hden > 0, Cnum / np.maximum(Hden, 1e-300), 0.0); np.fill_diagonal(S, 0.0)
    return _met_build_from_S(S, piM, kernel, _met_sym)


def _met_S_from_F(F, pi, kernel):
    piM = np.maximum(pi.reshape(NA, NA), 1e-300)
    num = np.einsum("ijkj->ik", F.reshape(NA, NA, NA, NA))
    den = np.zeros((NA, NA))
    for j in range(NA):
        den += _met_sym(piM[:, j][:, None], piM[:, j][None, :], kernel)
    return num / np.maximum(den, 1e-300)


def _met_Q(S, piM, kernel):
    Q = _met_build_from_S(S, piM, kernel, _met_f)
    np.fill_diagonal(Q, 0.0); Q[np.diag_indices(NS)] = -Q.sum(1)
    return Q


def mstep_pi_metropolis(S, pi, N, T, kernel, steps=40, lr=0.5):
    """Symmetric-pi M-step for the Metropolis family: damped gradient ascent on
    log pi_ij of the complete-data LL (pi enters nonlinearly through f). Unified
    over kernels via d log f / d pi (see _met_dlogf)."""
    b = np.log(np.maximum(pi, 1e-12)).reshape(NA, NA); b = 0.5 * (b + b.T)
    scale = max(T.sum(), 1.0)
    for _ in range(steps):
        pm = np.exp(b - b.max()); pm = pm / pm.sum()
        piv = pm.reshape(NS); Q = _met_Q(S, pm.reshape(NA, NA), kernel)
        R = N - T[:, None] * Q                             # residual N_xy - T_x Q_xy
        ds, dd = _met_dlogf(piv[:, None], piv[None, :], kernel)
        grad = piv * ((R * ds).sum(1) + (R * dd).sum(0))   # d ll / d log pi_z
        gM = grad.reshape(NA, NA); gM = gM + gM.T
        b = b + lr * gM / scale; b = 0.5 * (b + b.T)
    pm = np.exp(b - b.max()); return (pm / pm.sum()).reshape(NS)


def renewal_init(pj, mu=1.0):
    """EXACTLY-lumpable renewal init for the Lumpable fit: F81 on the joint pair
    stationary pj.  Q_xy = mu*pj_y (y!=x) -- at each event the whole pair is resampled
    from pj -- reversible w.r.t. pj with the FACTORIZED symmetric flux F_xy = mu*pj_x*pj_y.
    Because F factorizes, the lumpability residual is identically zero for ANY pj:
    (B phi)_{ij,k} = sum_l F_{ij,kl} = pj_ij * rho_k = (pj_ij/rho_i)*(rho_i*rho_k), which
    lies in range(D) exactly (rho_k = sum_l pj_kl the cpt1 marginal).  So this point sits
    ON the two-sided lumpable manifold and lumps to F81 on each marginal -- the highly-
    degenerate exactly-lumpable renewal seed.  The constrained Lumpable EM then enriches
    the flux beyond F81 while the constrained pi-/flux-steps preserve lumpability.

    pj is component-symmetrized over the swap R (i,j)<->(j,i) so the init is component-
    exchangeable (empirical / mixture pj are already symmetric; this is a no-op there)."""
    pj = np.asarray(pj, float).reshape(NS)
    sw = np.arange(NS).reshape(NA, NA).T.reshape(NS)
    pj = 0.5 * (pj + pj[sw]); pj = pj / pj.sum()
    F = mu * np.outer(pj, pj); np.fill_diagonal(F, 0.0)     # F81 flux, exactly lumpable
    return pj, F


def fit_model(npair, tau, pi, orbit_id, n_orbits, is_single, model,
              n_iter=25, verbose=True, lump_rows=None, fit_pi=True,
              lump_product_init=False, init_pi=None, init_F=None):
    off = ~np.eye(NS, dtype=bool)
    is_met = model.startswith("metropolis_")
    # Synchronized & Lumpable use all orbits; Coupled & Metropolis single-transition
    mask = is_single if (model == "coupled" or is_met) else off
    pi = pi.copy()                                          # local; ML-updated if fit_pi
    if init_pi is not None:
        # explicit warm start (e.g. renewal_init): (pi, F) supplied, skip default init
        pi = np.asarray(init_pi, float).copy()
        piM = pi.reshape(NA, NA); rho = piM.sum(1)
        F = np.asarray(init_F, float).copy() if init_F is not None else np.zeros((NS, NS))
        Q = Q_from_flux(F, pi)
    else:
      # Two-sided lumpable is nonconvex; the coupled empirical pair stationary is a poor
      # basin (the local pi-CM-step stalls there).  Start from the independent-marginal
      # PRODUCT rho(x)rho -- a well-conditioned start that the constrained pi-step is free to
      # leave but empirically holds; it reaches a much higher optimum than the empirical start.
      if lump_product_init:
        r1 = pi.reshape(NA, NA).sum(1); r1 = r1 / r1.sum()
        pi = (r1[:, None] * r1[None, :]).reshape(NS)
      piM = pi.reshape(NA, NA); rho = piM.sum(1)
      # init: independent single-site process (empirical marginal), small flux
      F = np.zeros((NS, NS)); F[off & mask] = 1e-2 * np.sqrt(
        pi[:, None] * pi[None, :])[off & mask]
      Q = Q_from_flux(F, pi)
    hist = []
    for it in range(n_iter):
        N, T = estep(Q, pi, tau, npair)
        kernel = model.split("_", 1)[1] if is_met else None
        if is_met:
            F = mstep_metropolis(N, T, pi, kernel)
        elif model == "lumpable":
            F = mstep_lumpable(N, T, pi, orbit_id, n_orbits, lump_rows, piM, rho)
        else:
            F = mstep_flux(N, T, pi, orbit_id, n_orbits, mask)
        Q = Q_from_flux(F, pi)
        if fit_pi:
            if model == "lumpable":                        # CONSTRAINED pi-step (preserves
                pi = mstep_pi_lumpable(N, T, F, pi, lump_rows, float(npair.sum()))
                Q = Q_from_flux(F, pi)                      # lumpability; GTR mstep_pi does not)
            elif is_met and kernel != "gtr":               # nonlinear exposure: gradient pi-step
                S = _met_S_from_F(F, pi, kernel)
                pi = mstep_pi_metropolis(S, pi, N, T, kernel)
                Q = _met_Q(S, pi.reshape(NA, NA), kernel)
            else:                                          # flux/GTR form: closed-form pi-step
                pi, s = mstep_pi(F, pi, N, T)              # (gtr exchangeability S is pi-invariant)
                Q = s * pi[None, :]
                np.fill_diagonal(Q, 0.0); Q[np.diag_indices(NS)] = -Q.sum(1)
            piM = pi.reshape(NA, NA); rho = piM.sum(1)      # keep lumpable/metropolis pi in sync
        ll = loglik(Q, pi, tau, npair)
        hist.append(ll)
        if verbose and (it < 2 or it % 5 == 0 or it == n_iter - 1):
            print(f"    [{model}] it {it:2d}  LL={ll:.1f}", flush=True)
        # converge on PER-COUNT LL change (the 0.1%-relative rule stopped the
        # high-parameter models far short of their optimum, e.g. Synchronized).
        if it > 5 and (hist[-1] - hist[-2]) < 1e-6 * npair.sum():
            break
    return dict(Q=Q, F=F, pi=pi, hist=hist, ll=hist[-1])


def n_params(model, n_orbits, n_single_orbits):
    if model == "synchronized":
        return n_orbits
    if model == "coupled":
        return n_single_orbits
    if model == "lumpable":
        # general chain (N_phi orbits) + g (N_g) - lumpability constraints (N_c)
        n = NA; Nc = n * (n - 1) * (2 * n - 1) // 2
        return n_orbits + n * (n - 1) // 2 - Nc
    if model.startswith("metropolis_"):
        return 190                       # shared single-site exchangeability S
    return n_orbits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="path to data/cherry_counts_* dir")
    ap.add_argument("--models", default="synchronized,coupled",
                    help="comma list of {synchronized,coupled,lumpable,potts}")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--val-part", type=int, default=-1,
                    help="held-out family shard (default: last part if shards exist)")
    ap.add_argument("--fix-pi", action="store_true",
                    help="hold pi_ij at the empirical pair frequency (default: ML-fit pi)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # train/val split over disjoint family shards part_*.npz (held-out per-count LL)
    np_parts = n_parts(args.corpus)
    if np_parts >= 2:
        val = args.val_part if args.val_part >= 0 else np_parts - 1
        train_ids = [i for i in range(np_parts) if i != val]
        npair, tau, alph = load_parts(args.corpus, train_ids)
        vpair, _, _ = load_parts(args.corpus, [val])
        print(f"# corpus {args.corpus}: {np_parts} family shards; train={train_ids} "
              f"(pair {npair.sum():.3e}), val=part_{val} (pair {vpair.sum():.3e})", flush=True)
    else:
        npair, tau, alph = load_counts(args.corpus)
        vpair = None
        print(f"# corpus {args.corpus}: no shards; train=val=full (pair "
              f"{npair.sum():.3e}) -- NO held-out", flush=True)
    pi = empirical_pi(npair)
    t0 = time.time()
    orbit_id, n_orbits, is_single = build_orbits()
    n_single_orbits = len(np.unique(orbit_id[(~np.eye(NS, dtype=bool)) & is_single]))
    print(f"# orbits: {n_orbits} total, {n_single_orbits} single-transition "
          f"(built in {time.time()-t0:.1f}s)", flush=True)
    model_list = [m.strip() for m in args.models.split(",")]
    lump_rows = build_lump_rows(orbit_id) if "lumpable" in model_list else None

    results = {}
    params = {}
    for model in model_list:
        print(f"# fitting {model} ...", flush=True)
        t1 = time.time()
        r = fit_model(npair, tau, pi, orbit_id, n_orbits, is_single, model,
                      n_iter=args.iters, lump_rows=lump_rows, fit_pi=not args.fix_pi,
                      lump_product_init=(model == "lumpable"))
        params[f"{model}__pi"] = r["pi"]        # persist fitted params -- never re-fit
        params[f"{model}__Q"] = r["Q"]          # just to read pi/Q/flux back out
        params[f"{model}__F"] = r["F"]
        npar = n_params(model, n_orbits, n_single_orbits)
        per = r["ll"] / npair.sum()
        val_per = (loglik(r["Q"], r["pi"], tau, vpair) / vpair.sum()
                   if vpair is not None else float("nan"))   # r["pi"] (fitted), NOT
        #  the empirical pi: eig_rev needs Q reversible w.r.t. the pi it is given, else
        #  the symmetrisation is wrong and P != expm(Qt) (corrupted GTR/Lumpable val).
        results[model] = dict(ll=r["ll"], per_count_ll=per, val_per_count_ll=val_per,
                              n_params=npar, time_s=time.time() - t1,
                              monotone=bool(np.all(np.diff(r["hist"]) > -1.0)))
        print(f"# {model}: LL={r['ll']:.1f}  per-count={per:.4f}  "
              f"params={npar}  {time.time()-t1:.1f}s  "
              f"monotone={results[model]['monotone']}", flush=True)

    print("\n# ==== summary (per-count log-likelihood) ====")
    print(f"# {'model':<14}{'train':>12}{'val(held-out)':>16}{'params':>10}")
    for m, r in results.items():
        print(f"# {m:<14}{r['per_count_ll']:>12.4f}{r['val_per_count_ll']:>16.4f}"
              f"{r['n_params']:>10}")
    if args.out:
        json.dump({"corpus": args.corpus, "results": results},
                  open(args.out, "w"), indent=2)
        print(f"# wrote {args.out}")
        pp = args.out.rsplit(".", 1)[0] + "_params.npz"
        np.savez(pp, alphabet=np.array(list("ACDEFGHIKLMNPQRSTVWY")), **params)
        print(f"# wrote {pp} (fitted pi/Q/F per model -- read back, no re-fit)")


if __name__ == "__main__":
    main()
