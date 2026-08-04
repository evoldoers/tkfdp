#!/usr/bin/env python3
r"""Report coupling MI (stationary AND dynamic) per component, weighted-mean, and max, for
the Metropolis coupling mixtures and the exact-kernel lumpable mixtures.

  MI_stat  = MI(pi_c) of the joint stationary -- the coupling the Metropolis components carry,
             but IDENTICALLY 0 for the lumpable mixtures (product stationary rho_c(x)rho_c;
             their coupling lives entirely in the double-transition DYNAMICS).
  MI_dyn   = coupling MI in the two-point branch joint J(x,y)=pi_x P(t)_{xy}: reshaping the
             end-pair by site, MI between site-1's trajectory (i,k) and site-2's (j,l),
             averaged over the corpus branch-length (tau) distribution.  This is the common
             currency -- the coupling BOTH families actually encode over a branch.  An
             independent chain has MI_dyn = 0.

Reads the Metropolis components (results/mixture_component_char/components_K{K}.npz) and the
lumpable-mixture checkpoints (results/pair_models/lumpable_mixture/_resume_gi_K{K}.npz, or a
--params npz with lumpable__* keys).  Run with PYTHONPATH=src from the repo root."""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import fit_pair_models as FP                                    # noqa: E402
from fit_pair_models import NA, NS                              # noqa: E402
import fit_coupling_mixture_freeS as FS                         # noqa: E402  (FS.mi)
from tkfdp.permfield.hr import eig_rev                          # noqa: E402


def tau_weights(corpus):
    z = np.load(corpus, allow_pickle=True)
    tb = z["tb"]; cnt = z["cnt"].astype(float); tau = z["tau_centers"].astype(float)
    w = np.bincount(tb, weights=cnt, minlength=len(tau)).astype(float)
    return tau, w / max(w.sum(), 1e-30)


def dyn_mi(Q, pi, tau, wtau):
    """tau-averaged coupling MI between the two sites' branch trajectories under J=pi_x P(t)."""
    pic = np.clip(pi, 1e-12, None); lam, U, Uinv = eig_rev(Q, pic)
    tot = 0.0
    for t in range(len(tau)):
        if wtau[t] <= 0:
            continue
        P = (U * np.exp(lam * tau[t])) @ Uinv
        J = np.maximum(pi[:, None] * P, 0.0); J = J / max(J.sum(), 1e-30)
        Js = J.reshape(NA, NA, NA, NA).transpose(0, 2, 1, 3).reshape(NS, NS)  # [(i,k),(j,l)]
        m1 = Js.sum(1); m2 = Js.sum(0)
        tot += wtau[t] * float(np.sum(Js * np.log(
            np.maximum(Js, 1e-300) / np.maximum(np.outer(m1, m2), 1e-300))))
    return tot


def report(label, Qs, pis, w, tau, wtau):
    K = len(w); w = np.asarray(w, float); w = w / w.sum(); order = np.argsort(w)[::-1]
    smi = [FS.mi(pis[c]) for c in range(K)]
    dmi = [dyn_mi(Qs[c], pis[c], tau, wtau) for c in range(K)]
    print(f"\n## {label}")
    print(f"# {'c':>2} {'w':>6} {'MI_stat':>9} {'MI_dyn':>9}")
    for c in order:
        print(f"# {c:>2} {w[c]:>6.3f} {smi[c]:>9.4f} {dmi[c]:>9.4f}")
    wstat = sum(w[c] * smi[c] for c in range(K)); wdyn = sum(w[c] * dmi[c] for c in range(K))
    print(f"#  w-mean  MI_stat={wstat:.4f}  MI_dyn={wdyn:.4f}")
    print(f"#  max     MI_stat={max(smi):.4f}  MI_dyn={max(dmi):.4f}")
    return dict(w_stat=wstat, w_dyn=wdyn, max_stat=max(smi), max_dyn=max(dmi),
                per_stat=[smi[c] for c in order], per_dyn=[dmi[c] for c in order])


def metro_qs(z):
    S = np.asarray(z["S"], float); pis = np.asarray(z["pis"], float)
    Qs = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(len(pis))]
    return Qs, [pis[c] for c in range(len(pis))], np.asarray(z["weights"], float)


def lump_qs(z):
    pis = [np.asarray(p, float) for p in z["pis"]]; Fs = [np.asarray(f, float) for f in z["Fs"]]
    Qs = [FP.Q_from_flux(Fs[c], pis[c]) for c in range(len(pis))]
    return Qs, pis, np.asarray(z["w"], float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--K", type=int, nargs="+", default=[2, 3, 4, 8])
    ap.add_argument("--corpus", default="data/per_contact_trrosetta/counts.npz")
    ap.add_argument("--metro-dir", default="results/mixture_component_char")
    ap.add_argument("--lump-dir", default="results/pair_models/lumpable_mixture")
    a = ap.parse_args()
    tau, wtau = tau_weights(a.corpus)
    for K in a.K:
        mp = f"{a.metro_dir}/components_K{K}.npz"
        if os.path.exists(mp):
            Qs, pis, w = metro_qs(np.load(mp, allow_pickle=True))
            report(f"Metropolis mixture K={K}", Qs, pis, w, tau, wtau)
        lp = f"{a.lump_dir}/_resume_gi_K{K}.npz"
        if os.path.exists(lp):
            Qs, pis, w = lump_qs(np.load(lp, allow_pickle=True))
            report(f"Lumpable (exact-kernel) mixture K={K}", Qs, pis, w, tau, wtau)


if __name__ == "__main__":
    main()
