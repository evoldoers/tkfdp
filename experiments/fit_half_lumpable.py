#!/usr/bin/env python3
"""Fit Half-lumpable = the one-sided (cpt1) lumpable reversible pair chain, and score
it held-out (same metric as fit_pair_models). See
analysis/dependent_onesided_lumpable_derivation.md.

SUPERSEDED for the paper table by experiments/fit_half_lumpable_kernel.py.  This ALM
fitter freezes pi at the *empirical correlated* pair stationary (pi = empirical_pi(npair))
and solves the flux by the penalty ALM.  Because half-lumpable's feasible set CONTAINS
two-sided Lumpable's, its optimum must be >= Lumpable's -- but frozen at a correlated pi
it cannot reach Lumpable's product-pi optimum, so it scored *below* Lumpable (a pure
convergence artifact of the fixed-pi choice, not a modelling fact).  fit_half_lumpable_kernel
fits at the product pi with the exact convex kernel M-step and correctly lands at or above
Lumpable (both settle at the product stationary here; the coupling washes out either way).
Use the kernel fitter for tab:pairfit; this file is kept as the ALM reference.

Half-lumpable is the MOST PERMISSIVE reversible 400-state chain strongly lumpable to
cpt1 (cpt1's marginal is an exact Markov/GTR process), with NOTHING imposed on cpt2.
Mechanically it is our existing exact-lumpable ALM (em_lumpable_reversible_ctmc.tex,
fit_pair_models.mstep_lumpable) with ONE change: the flux is parameterised on
TIME-REVERSAL-only orbits (phi_{ab,cd}=phi_{cd,ab}) instead of Klein-4
(component-swap x time-reversal). build_lump_rows already constrains only the cpt1
row-marginal Sum_l F[(i,j),(k,l)] = pi_{ij} g_{ik}; on Klein-4 orbits the swap tie
makes that two-sided (our "Lumpable"), on time-reversal orbits it stays one-sided.

Counts are component-swap-symmetrised (arbitrary field label), as for Renewal; the
stationary stays symmetric while the DYNAMICS are non-exchangeable (that is the
lever over 2-sided Lumpable). Predicted to beat Lumpable (strictly fewer
constraints). Effective identifiable DOF (Fisher spectrum) is a separate diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
import fit_pair_models as FP                                          # noqa: E402
from fit_pair_models import NA, NS                                    # noqa: E402


def build_orbits_timerev():
    """orbit_id (400,400) over off-diagonal transitions, tying only time-reversal
    images (i,j;k,l)~(k,l;i,j) -- NOT the component swap. -> ~C(400,2)=79,800 orbits."""
    ii, jj = np.divmod(np.arange(NS), NA)
    oid = np.full((NS, NS), -1, np.int64)
    canon = {}
    nxt = 0
    for x in range(NS):
        i, j = int(ii[x]), int(jj[x])
        for y in range(NS):
            if x == y:
                continue
            k, l = int(ii[y]), int(jj[y])
            key = min((i, j, k, l), (k, l, i, j))
            o = canon.get(key)
            if o is None:
                o = nxt; canon[key] = o; nxt += 1
            oid[x, y] = o
    return oid, nxt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpora", default="trrosetta,trrosetta_rate,af_full")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--out", default="results/pair_models/half_lumpable_eval.json")
    args = ap.parse_args()

    sw = np.arange(NS).reshape(NA, NA).T.reshape(NS)
    t0 = time.time()
    oid, no = build_orbits_timerev()
    rows = FP.build_lump_rows(oid)
    print(f"# time-reversal orbits: {no} ({time.time()-t0:.0f}s)", flush=True)

    out = {}
    for name in [c.strip() for c in args.corpora.split(",")]:
        path = f"data/cherry_counts_{name}"
        npar = FP.n_parts(path)
        if npar < 2:
            continue
        val = npar - 1
        npair, tau, _ = FP.load_parts(path, [i for i in range(npar) if i != val])
        vpair, _, _ = FP.load_parts(path, [val])
        npair = npair + npair[sw][:, sw]                             # symmetrise counts
        vpair = vpair + vpair[sw][:, sw]
        pi = FP.empirical_pi(npair)
        t1 = time.time()
        r = FP.fit_model(npair, tau, pi, oid, no, None, "lumpable",
                         n_iter=args.iters, lump_rows=rows, verbose=True)
        tr = r["ll"] / npair.sum()
        va = FP.loglik(r["Q"], r["pi"], tau, vpair) / vpair.sum()
        out[name] = dict(train=float(tr), val=float(va), n_orbits=int(no),
                         time_s=time.time() - t1)
        # persist fitted Q/pi for the Fisher effective-DOF diagnostic
        np.savez(f"results/pair_models/half_lumpable_{name}_fit.npz", Q=r["Q"], pi=r["pi"],
                 tau=tau)
        print(f"# Half-lumpable (one-sided lumpable) {name}: train={tr:.4f} val={va:.4f}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()
