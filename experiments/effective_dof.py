#!/usr/bin/env python3
"""Empirical effective identifiable DOF of a fitted reversible pair model under
component-swap-symmetrised counts, via finite-difference Fisher curvature at the
MLE, split into swap-SYMMETRIC and swap-ANTISYMMETRIC flux subspaces.

Why this works: the swap-symmetrised log-likelihood depends on the model only
through the swap-symmetric part of log P (it is sum_n log[P(ijkl) P(jilk)]). So a
flux perturbation whose effect on log P is swap-ANTISYMMETRIC is EXACTLY flat --
unidentified by the augmented data. At the MLE the first-order term vanishes, so
the local curvature  c(dF) = -(LL(Q+eps dQ)+LL(Q-eps dQ)-2 LL0)/eps^2  estimates
dQ^T F dQ (F = Fisher). We probe random unit flux perturbations restricted to the
symmetric subspace (dF = dF^swap) and the antisymmetric subspace (dF = -dF^swap),
and compare their curvature distributions. antisym ~ 0 confirms those directions
are tied by the augmentation; the identified fraction * nominal dimension is the
effective DOF estimate.

Usage:
  python experiments/effective_dof.py --fit results/pair_models/half_lumpable_trrosetta_fit.npz \
      --corpus trrosetta --nominal 72789
  python experiments/effective_dof.py --params results/pair_models/trrosetta_converged_params.npz \
      --model synchronized --corpus trrosetta --nominal 40299
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from fit_pair_models import load_parts, n_parts, loglik, NA, NS       # noqa: E402


def curvature(Q0, ll0, pi, tau, nsym, dF, eps):
    """Second-difference curvature ~ dQ^T F dQ for a unit flux perturbation dF."""
    dQ = dF / np.maximum(pi[:, None], 1e-300)
    np.fill_diagonal(dQ, 0.0)
    dQ[np.diag_indices(NS)] = -dQ.sum(1)
    llp = loglik(Q0 + eps * dQ, pi, tau, nsym)
    llm = loglik(Q0 - eps * dQ, pi, tau, nsym)
    return -(llp + llm - 2.0 * ll0) / eps ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", default=None, help="npz with Q, pi, tau")
    ap.add_argument("--params", default=None, help="fit_pair_models *_params.npz")
    ap.add_argument("--model", default=None, help="model key inside --params")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--nominal", type=float, default=None, help="nominal DOF for the estimate")
    ap.add_argument("--n-probes", type=int, default=64)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.fit:
        z = np.load(args.fit); Q0 = z["Q"]; pi = z["pi"]; tau = z["tau"]
    else:
        z = np.load(args.params)
        Q0 = z[f"{args.model}__Q"]; pi = z[f"{args.model}__pi"]
        _, tau, _ = load_parts(f"data/cherry_counts_{args.corpus}", [0])

    path = f"data/cherry_counts_{args.corpus}"; npar = n_parts(path)
    npair, tau2, _ = load_parts(path, list(range(npar - 1)))
    if tau is None:
        tau = tau2
    sw = np.arange(NS).reshape(NA, NA).T.reshape(NS)
    nsym = npair + npair[sw][:, sw]                                   # symmetrised train
    ll0 = loglik(Q0, pi, tau, nsym)

    off = ~np.eye(NS, dtype=bool)
    rng = np.random.default_rng(args.seed)
    csym, casym = [], []
    for _ in range(args.n_probes):
        R = rng.standard_normal((NS, NS)); R = np.triu(R, 1); R = R + R.T   # symmetric flux
        R *= off
        Rs = 0.5 * (R + R[sw][:, sw])                                # swap-symmetric part
        Ra = 0.5 * (R - R[sw][:, sw])                                # swap-antisymmetric part
        for store, dF in ((csym, Rs), (casym, Ra)):
            nrm = np.sqrt((dF * dF).sum())
            if nrm < 1e-12:
                continue
            store.append(curvature(Q0, ll0, pi, tau, nsym, dF / nrm, args.eps))
    csym = np.array(csym); casym = np.array(casym)
    med_s = np.median(csym); med_a = np.median(casym)
    ratio = med_a / max(med_s, 1e-300)
    print(f"# model={args.model or args.fit}  corpus={args.corpus}")
    print(f"#   swap-SYMMETRIC  curvature: median={med_s:.3e}  "
          f"[q10={np.quantile(csym,.1):.2e} q90={np.quantile(csym,.9):.2e}]")
    print(f"#   swap-ANTISYMM   curvature: median={med_a:.3e}  "
          f"[q10={np.quantile(casym,.1):.2e} q90={np.quantile(casym,.9):.2e}]")
    print(f"#   antisym/sym curvature ratio = {ratio:.3e}  "
          f"(=> antisym directions {'FLAT (tied by symmetrisation)' if ratio<0.05 else 'partly identified'})")
    if args.nominal:
        # crude effective-DOF: nominal * (identified fraction). With antisym ~ flat,
        # only the symmetric half is identified.
        frac = 1.0 / (1.0 + med_s / max(med_a, 1e-300))   # ~ sym share of identified energy
        print(f"#   nominal DOF={args.nominal:.0f}; antisym flat => effective ~ symmetric "
              f"subspace; identified-energy sym-fraction ~ {1-ratio/(1+ratio):.3f}")


if __name__ == "__main__":
    main()
