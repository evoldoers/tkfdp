#!/usr/bin/env python3
r"""Renewal-initialised Lumpable fit: multi-start the two-sided-lumpable HR-EM from the
HIGHLY-DEGENERATE RENEWAL parameterization (fit_pair_models.renewal_init) instead of the
default independent-marginal PRODUCT start, to test the renewal-init cure for the product
optimization stall (analysis/lumpable_trap_optimization.md) on the REAL empirical corpora.

Candidate seeds (each -> a component-exchangeable reversible (pi, F) init):
  product            : rho(x)rho(y)  (the current default Lumpable init; baseline)
  renewal_emp        : renewal from the empirical pair pi (the "one-size-fits-all"
                       synchronized stationary) + LG08 exchangeability
  renewal_K{K}_c{c}  : renewal from Metropolis-mixture component c of the K-component
                       shared-free-S coupling mixture (pi_c + that mixture's shared S),
                       loaded from results/mixture_component_char/components_K{K}.npz

Default: SCAN only -- report each seed's STARTING per-count log-likelihood (cheap: one
400-state eig + a P(t) sweep per seed) plus the current stored Lumpable optimum, so we can
see which basins start high before spending EM time.  With --run-em, run the constrained
Lumpable EM (fit_pair_models.fit_model, model=lumpable) from each seed and report the final
train/val per-count LL, flagging the best.

Run with PYTHONPATH=src from the repo root."""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
import fit_pair_models as FP                                    # noqa: E402
from fit_pair_models import NA, NS                              # noqa: E402
from tkfdp.lg08 import Q_LG08 as _QLG, PI_LG08 as _PILG         # noqa: E402


def lg08_S():
    S = np.asarray(_QLG, float) / np.maximum(np.asarray(_PILG, float)[None, :], 1e-300)
    S = 0.5 * (S + S.T); np.fill_diagonal(S, 0.0)
    return S


def product_init(pi):
    r1 = pi.reshape(NA, NA).sum(1); r1 = r1 / r1.sum()
    pj = (r1[:, None] * r1[None, :]).reshape(NS)
    off = ~np.eye(NS, dtype=bool)
    F = np.zeros((NS, NS))
    F[off] = 1e-2 * np.sqrt(pj[:, None] * pj[None, :])[off]     # matches fit_model default
    return pj, F


def collect_seeds(pi_emp, comp_paths):
    """[(label, pi_init, F_init)] for every candidate warm start.  The renewal seeds are
    F81 on the given joint pi (exactly lumpable); the empirical pi is the one-size-fits-all
    synchronized stationary, the mixture pi_c are the Metropolis coupling components."""
    seeds = [("product", *product_init(pi_emp)),
             ("renewal_emp", *FP.renewal_init(pi_emp))]
    for path in comp_paths:
        z = np.load(path, allow_pickle=True)
        K = int(z["K"]); pis = np.asarray(z["pis"], float); w = np.asarray(z["weights"], float)
        for c in range(K):
            pi_ci, F_ci = FP.renewal_init(pis[c])
            seeds.append((f"renewal_K{K}_c{c}(w={w[c]:.2f})", pi_ci, F_ci))
    return seeds


def lump_residual(pi_init, F_init):
    """Relative two-sided-lumpability residual ||P_perp(B phi)|| / ||B phi|| of an init:
    B phi_{ij,k} = sum_l F[ij,kl] (i!=k); P_perp removes, per (i,k), the pr-weighted mean
    over spectator j with pr_ij = pi_ij/rho_i.  F81 (renewal) => identically 0."""
    F4 = F_init.reshape(NA, NA, NA, NA)
    pm = pi_init.reshape(NA, NA); rho = pm.sum(1)
    ri, rj, rk = [], [], []
    for i in range(NA):
        for j in range(NA):
            for k in range(NA):
                if i != k:
                    ri.append(i); rj.append(j); rk.append(k)
    ri, rj, rk = np.array(ri), np.array(rj), np.array(rk)
    Bphi = F4.sum(3)[ri, rj, rk]
    pr = pm[ri, rj] / np.maximum(rho[ri], 1e-300)
    ik = ri * NA + rk
    res = FP._perp(Bphi, pr, ik, NA * NA)
    return float(np.linalg.norm(res) / max(np.linalg.norm(Bphi), 1e-30))


def start_ll(pi_init, F_init, tau, npair):
    Q = FP.Q_from_flux(F_init, pi_init)
    return FP.loglik(Q, pi_init, tau, npair)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, help="data/cherry_counts_* dir")
    ap.add_argument("--components", default="",
                    help="comma list of components_K*.npz to seed mixture-component renewals")
    ap.add_argument("--val-part", type=int, default=-1)
    ap.add_argument("--run-em", action="store_true", help="run the Lumpable EM from each seed")
    ap.add_argument("--em-seeds", default="",
                    help="comma list of substrings; only seeds whose label matches get an EM "
                         "run (empty = all). e.g. 'product,renewal_emp'")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    np_parts = FP.n_parts(a.corpus)
    if np_parts >= 2:
        val = a.val_part if a.val_part >= 0 else np_parts - 1
        train_ids = [i for i in range(np_parts) if i != val]
        npair, tau, _ = FP.load_parts(a.corpus, train_ids)
        vpair, _, _ = FP.load_parts(a.corpus, [val])
    else:
        npair, tau, _ = FP.load_counts(a.corpus); vpair = None
    tot = float(npair.sum()); vtot = float(vpair.sum()) if vpair is not None else float("nan")
    print(f"# corpus {a.corpus}: train pair {tot:.3e}"
          + (f", val pair {vtot:.3e}" if vpair is not None else " (no held-out)"), flush=True)

    pi_emp = FP.empirical_pi(npair)
    comp_paths = [p.strip() for p in a.components.split(",") if p.strip()]
    seeds = collect_seeds(pi_emp, comp_paths)

    # reference: current stored Lumpable optimum (product-init constrained fit)
    ref = {}
    try:
        pp = np.load(f"results/pair_models/lumpable_{a.corpus.split('cherry_counts_')[-1]}"
                     f"_params.npz", allow_pickle=True)
        Qr, pir = pp["lumpable__Q"], pp["lumpable__pi"]
        ref = {"train": FP.loglik(Qr, pir, tau, npair) / tot,
               "val": (FP.loglik(Qr, pir, tau, vpair) / vtot) if vpair is not None else float("nan")}
        print(f"# stored Lumpable optimum: train/count={ref['train']:.4f}"
              + (f"  val/count={ref['val']:.4f}" if vpair is not None else ""), flush=True)
    except Exception as e:
        print(f"# (no stored Lumpable optimum to compare: {e})", flush=True)

    print(f"\n# ==== starting per-count LL + lumpability residual by seed ====\n"
          f"# {'seed':<26}{'start train':>14}{'lump_resid':>14}", flush=True)
    rows = {}
    for label, pi0, F0 in seeds:
        s = start_ll(pi0, F0, tau, npair) / tot
        lr = lump_residual(pi0, F0)
        rows[label] = {"start_train": s, "lump_resid": lr}
        print(f"# {label:<26}{s:>14.4f}{lr:>14.2e}", flush=True)

    if a.run_em:
        orbit_id, n_orbits, is_single = FP.build_orbits()
        lump_rows = FP.build_lump_rows(orbit_id)
        print(f"\n# ==== Lumpable EM from each seed ({a.iters} iters) ====", flush=True)
        pats = [p.strip() for p in a.em_seeds.split(",") if p.strip()]
        em_seeds = [s for s in seeds if not pats or any(p in s[0] for p in pats)]
        best = None
        for label, pi0, F0 in em_seeds:
            t1 = time.time()
            r = FP.fit_model(npair, tau, pi_emp, orbit_id, n_orbits, is_single, "lumpable",
                             n_iter=a.iters, verbose=False, lump_rows=lump_rows, fit_pi=True,
                             init_pi=pi0, init_F=F0)
            tr = r["ll"] / tot
            va = (FP.loglik(r["Q"], r["pi"], tau, vpair) / vtot) if vpair is not None else float("nan")
            mono = bool(np.all(np.diff(r["hist"]) > -1.0))
            rows[label].update(em_train=tr, em_val=va, monotone=mono, time_s=time.time() - t1)
            print(f"# {label:<26} EM train/count={tr:.4f}  val/count={va:.4f}  "
                  f"mono={mono}  [{time.time()-t1:.0f}s]", flush=True)
            key = va if vpair is not None else tr
            if best is None or key > best[0]:
                best = (key, label)
        print(f"\n# BEST seed: {best[1]}  ({'val' if vpair is not None else 'train'}/count={best[0]:.4f})",
              flush=True)

    if a.out:
        json.dump({"corpus": a.corpus, "reference_lumpable": ref, "seeds": rows},
                  open(a.out, "w"), indent=2)
        print(f"# wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
