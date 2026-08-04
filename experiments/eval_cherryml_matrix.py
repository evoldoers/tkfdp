#!/usr/bin/env python3
"""Evaluate CherryML's published 400x400 co-evolution rate matrix (Q2) as an
external baseline on our coupled cherry-count corpora, on the SAME held-out
transition-likelihood metric as experiments/fit_pair_models.py.

Q2 (Zenodo 7830072, rate_matrices/Q2.txt) is a fixed matrix trained on
CherryML's own FastTree cherries over the trRosetta families. Its pair states
use the single-AA order 'ARNDCQEGHILKMFPSTWYV'; we remap to our tensor order
'ACDEFGHIKLMNPQRSTVWY'. Because Q2's time axis need not match our LG-ML tau, we
grant it its ONE natural free parameter -- a global rate scale c -- fit by ML on
the training shards (report c=1 too). Likelihood is the pure transition score
sum_t n[:,:,t] . log expm(c*Q2*tau_t) (no initial-state term), identical to
fit_pair_models.loglik, so the numbers drop straight into the model table.

Caveat: on trRosetta, Q2 trained on those very families (different cherry
extraction), so its trRosetta val is not truly out-of-sample; af_full (deep
Pfam, disjoint alignments) is the cleaner external test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from fit_pair_models import load_parts, n_parts                       # noqa: E402

CHERRY_ALPHA = "ARNDCQEGHILKMFPSTWYV"     # CherryML / LG single-AA order
OUR_ALPHA = "ACDEFGHIKLMNPQRSTVWY"        # our tensor order
NA, NS = 20, 400


def load_q2(path):
    """Parse Q2.txt (header row of 400 pair labels, then 400 labelled rows) and
    remap both axes from CherryML pair order to our pair order -> (400,400)."""
    lines = [ln.rstrip("\n") for ln in Path(path).read_text().splitlines() if ln.strip()]
    Q = np.array([[float(x) for x in ln.split()[1:]] for ln in lines[1:]], float)
    assert Q.shape == (NS, NS), Q.shape
    # permutation: our pair index -> CherryML pair index
    cpos = {a: i for i, a in enumerate(CHERRY_ALPHA)}
    perm = np.empty(NS, int)
    for xo in range(NA):
        for yo in range(NA):
            xc = cpos[OUR_ALPHA[xo]]; yc = cpos[OUR_ALPHA[yo]]
            perm[xo * NA + yo] = xc * NA + yc
    Q = Q[np.ix_(perm, perm)]
    Q = Q - np.diag(Q.sum(1))              # re-clean the generator (rows sum to 0)
    return Q


def transition_ll(Q, c, tau, npair):
    """sum_t n[:,:,t] . log expm(c*Q*tau_t) -- same metric as fit_pair_models."""
    ll = 0.0
    for t in range(tau.shape[0]):
        edge = npair[:, :, t]
        if edge.sum() == 0:
            continue
        P = expm(c * Q * tau[t])
        ll += float((edge * np.log(np.clip(P, 1e-300, None))).sum())
    return ll


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--q2", default="data/external/cherryml/rate_matrices/Q2.txt")
    ap.add_argument("--corpora", default="trrosetta,trrosetta_rate,af_full")
    ap.add_argument("--out", default="results/pair_models/cherryml_q2_eval.json")
    args = ap.parse_args()

    Q = load_q2(args.q2)
    np.savez("data/external/cherryml/Q2_ours.npz", Q=Q,
             alphabet=np.array(list(OUR_ALPHA)))
    print(f"# loaded + remapped Q2 -> our order; |rowsum|max={np.abs(Q.sum(1)).max():.1e}",
          flush=True)

    out = {}
    for name in [c.strip() for c in args.corpora.split(",")]:
        path = f"data/cherry_counts_{name}"
        npar = n_parts(path)
        if npar < 2:
            print(f"# {name}: <2 shards, skip"); continue
        val = npar - 1
        npair, tau, _ = load_parts(path, [i for i in range(npar) if i != val])
        vpair, _, _ = load_parts(path, [val])
        # ML global scale on train
        res = minimize_scalar(lambda lc: -transition_ll(Q, np.exp(lc), tau, npair),
                              bounds=(np.log(0.05), np.log(20.0)), method="bounded")
        c = float(np.exp(res.x))
        tr = transition_ll(Q, c, tau, npair) / npair.sum()
        va = transition_ll(Q, c, tau, vpair) / vpair.sum()
        tr1 = transition_ll(Q, 1.0, tau, npair) / npair.sum()
        va1 = transition_ll(Q, 1.0, tau, vpair) / vpair.sum()
        out[name] = dict(scale=c, train=tr, val=va, train_c1=tr1, val_c1=va1)
        print(f"# {name:16s} c*={c:.3f}  train={tr:.4f} val={va:.4f}  "
              f"(c=1: train={tr1:.4f} val={va1:.4f})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()
