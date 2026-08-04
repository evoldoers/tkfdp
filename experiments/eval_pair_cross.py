#!/usr/bin/env python3
"""Cross-corpus generalisation of the pair-substitution models: train each model
on each corpus (full counts) and score per-count log-likelihood on every corpus.
The off-diagonal (train A -> eval B) tests whether the coupling structure is
universal or corpus-specific. Prints a per-count-LL matrix.

Only the EXACT-M-step models are reported (synchronized, coupled, potts); the
Lumpable rational-dual is approximate (its damped Newton-Jacobi under-enforces
lumpability at scale) and is excluded here.
"""
from __future__ import annotations

import argparse
import sys
import numpy as np

sys.path.insert(0, "src")
import experiments.fit_pair_models as F                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="trrosetta,af_full")
    ap.add_argument("--models", default="synchronized,coupled,potts")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    corp = args.corpora.split(",")
    models = args.models.split(",")
    data = {c: F.load_counts(f"data/cherry_counts_{c}") for c in corp}
    oid, no, iss = F.build_orbits()

    trained = {}
    for tc in corp:
        npair, tau, _ = data[tc]
        pi = F.empirical_pi(npair)
        for m in models:
            r = F.fit_model(npair, tau, pi, oid, no, iss, m, n_iter=args.iters,
                            verbose=False)
            trained[(tc, m)] = (r["Q"], pi, tau)
            print(f"# trained {m} on {tc}", flush=True)

    print("\n# per-count log-likelihood  [rows = train corpus, cols = eval corpus]")
    for m in models:
        print(f"\n## {m}")
        hdr = "  train\\eval  " + "".join(f"{c:>12}" for c in corp)
        print(hdr)
        for tc in corp:
            Q, pi, tau = trained[(tc, m)]
            cells = []
            for ec in corp:
                ne, _, _ = data[ec]
                cells.append(F.loglik(Q, pi, tau, ne) / ne.sum())
            print(f"  {tc:<11}" + "".join(f"{v:>12.4f}" for v in cells))


if __name__ == "__main__":
    main()
