#!/usr/bin/env python3
r"""Correct two-sided Lumpable EM with the DERIVATION's constrained stationary step
(em_lumpable_reversible_ctmc.tex eq:pistep), replacing the buggy GTR mstep_pi.

ECM per iter:
  E-step -> flux CM-step (mstep_lumpable, correct) given pi
  -> constrained pi-step: maximise Q_pi(pi) = -sum_x N_{x+} log pi_x - sum_x T_x r_x/pi_x
     (conditional-transition model, M_x=0, r_x = sum_y F_xy fixed) subject to the
     lumpability feasibility P_perp^{(pi)}(B phi) = 0, via a penalty on symmetric-pi logits.
Init pi at the product rho(x)rho so the family is feasible; the pi-step is FREE (209-dof
symmetric), so if a NON-product pi raises the likelihood it will move there -- resolving
whether the two-sided lumpable MLE is product or not.
"""
import sys, glob, argparse
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "experiments")
import fit_pair_models as FP

NA, NS = 20, 400


def sym_pi(u):
    M = 0.5 * (u.reshape(NA, NA) + u.reshape(NA, NA).T)
    M = M - M.max(); p = np.exp(M); p /= p.sum()
    return p.reshape(NS)


def run(corpus, n_iter, seed_prod=True, out=None, verbose=True, warm=None):
    nparts = len(glob.glob(f"{corpus}/part_*.npz")); val = nparts - 1
    tr, tau, _ = FP.load_parts(corpus, list(range(val)))
    va, _, _ = FP.load_parts(corpus, [val])
    orbit_id, n_orbits, is_single = FP.build_orbits()
    ri, rj, rk, ro = FP.build_lump_rows(orbit_id)
    ik = ri * NA + rk; nik = NA * NA

    def Bphi_from_F(F):
        return F.reshape(NA, NA, NA, NA).sum(3)[ri, rj, rk]

    def cons(pi, Bphi):
        piM = pi.reshape(NA, NA); rho = piM.sum(1)
        pr = piM[ri, rj] / np.maximum(rho[ri], 1e-300)
        return FP._perp(Bphi, pr, ik, nik)

    off = ~np.eye(NS, dtype=bool)
    pe = FP.empirical_pi(tr); rho1 = pe.reshape(NA, NA).sum(1); rho1 /= rho1.sum()
    if warm:                                              # warm start from Synchronized (pi + flux)
        wd = np.load(warm); pi = wd["synchronized__pi"].copy(); F = wd["synchronized__F"].copy()
        print(f"# WARM init from {warm}: pi_nonproduct_dev="
              f"{np.abs(pi.reshape(NA,NA)-np.outer(rho1,rho1)).max()/np.outer(rho1,rho1).max():.3f}", flush=True)
    else:
        pi = (rho1[:, None] * rho1[None, :]).reshape(NS) if seed_prod else pe
        F = np.zeros((NS, NS)); F[off] = 1e-2 * np.sqrt(pi[:, None] * pi[None, :])[off]
    Q = FP.Q_from_flux(F, pi)
    hist = []
    for it in range(n_iter):
        N, T = FP.estep(Q, pi, tau, tr)
        piM = pi.reshape(NA, NA); rho = piM.sum(1)
        F = FP.mstep_lumpable(N, T, pi, orbit_id, n_orbits, (ri, rj, rk, ro), piM, rho)
        Bphi = Bphi_from_F(F)
        N_out = N.sum(1); Tr = T * F.sum(1)                 # a_x=N_{x+}, c_x=T_x r_x

        # constrained pi-step: augmented Lagrangian, BOTH sides RESCALED to O(1) so the
        # multipliers bite (objective -> per-count; constraint -> relative violation --
        # the flux/count ~1e12 scale gap otherwise drowns the constraint out entirely).
        Ntot = float(tr.sum()); Bn = max(float(np.linalg.norm(Bphi)), 1e-30)
        def negAL(u, mu, rho_pen):
            p = np.clip(sym_pi(u), 1e-12, None)
            Qpi = (-(N_out * np.log(p)).sum() - (Tr / p).sum()) / Ntot   # O(1)
            c = cons(p, Bphi) / Bn                                        # O(1) relative
            return -(Qpi - mu @ c - 0.5 * rho_pen * (c @ c))
        u = np.log(np.clip(pi, 1e-12, None))
        mu = np.zeros(len(ri)); rho_pen = 1.0
        for _o in range(30):                                # outer AL: multiplier updates
            u = minimize(negAL, u, args=(mu, rho_pen), method="L-BFGS-B",
                         options=dict(maxiter=80)).x
            c = cons(np.clip(sym_pi(u), 1e-12, None), Bphi) / Bn
            mu = mu + rho_pen * c
            rho_pen = min(rho_pen * 2.0, 1e6)
        pi = sym_pi(u)
        Q = FP.Q_from_flux(F, pi)
        ll = FP.loglik(Q, pi, tau, tr) / tr.sum()
        vl = FP.loglik(Q, pi, tau, va) / va.sum()
        cviol = np.linalg.norm(cons(pi, Bphi)) / max(np.linalg.norm(Bphi), 1e-30)
        pdev = np.abs(pi.reshape(NA, NA) - np.outer(rho1, rho1)).max() / np.outer(rho1, rho1).max()
        hist.append(ll)
        if verbose:
            print(f"it {it:2d}: train={ll:.4f} val={vl:.4f}  relviol={cviol:.1e} "
                  f"pi_nonproduct_dev={pdev:.3f}", flush=True)
    mono = bool(np.all(np.diff(hist) > -1e-4))
    print(f"# {corpus}: FINAL train={hist[-1]:.4f} val={vl:.4f}  monotone={mono}  "
          f"pi_nonproduct_dev={pdev:.3f}", flush=True)
    if out:
        np.savez(out, Q=Q, pi=pi, F=F)
    return dict(train=hist[-1], val=vl, mono=mono, pdev=pdev, Q=Q, pi=pi, F=F)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/cherry_counts_trrosetta")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--warm", default=None, help="npz with synchronized__pi/__F for warm init")
    a = ap.parse_args()
    run(a.corpus, a.iters, out=a.out, warm=a.warm)
