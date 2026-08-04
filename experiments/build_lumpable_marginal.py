#!/usr/bin/env python3
"""Fit the exactly-lumpable pair chain on a coupled cherry-count corpus and save
its 400x400 generator Q together with the extracted single-site marginal
generator A. Lumpability = the marginals are Markov, so A_ik = mean_j sum_l
Q[(i,j),(k,l)] is well-defined; the independent-sites null used by the
coupling-responsibility reweighting (experiments/responsibility_reweight_*) is
A (+) A (Kronecker sum), which has the SAME per-site marginals as the paired
chain -- coupling is the only thing removed.

Output npz (default data/lumpable_<corpus>.npz): Q (400,400), A (20,20),
pi (400,), tau, train_ll_per_count.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
import fit_pair_models as F                                           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="data/cherry_counts_trrosetta")
    ap.add_argument("--train-parts", default="0,1,2,3")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"data/lumpable_{Path(args.corpus).name.replace('cherry_counts_','')}.npz"

    t0 = time.time()
    parts = [int(x) for x in args.train_parts.split(",")]
    npair, tau, _ = F.load_parts(args.corpus, parts)
    pi = F.empirical_pi(npair)
    oid, no, iss = F.build_orbits()
    rows = F.build_lump_rows(oid)
    r = F.fit_model(npair, tau, pi, oid, no, iss, "lumpable", n_iter=args.iters,
                    verbose=True, lump_rows=rows)
    Q4 = r["Q"].reshape(20, 20, 20, 20)
    A = np.einsum("ijkl->ik", Q4) / 20.0                              # marginal generator
    np.fill_diagonal(A, 0.0); A[np.diag_indices(20)] = -A.sum(1)
    np.savez(out, Q=r["Q"], A=A, pi=r["pi"], tau=tau,
             train_ll_per_count=r["ll"] / npair.sum())
    print(f"# saved {out}: train per-count={r['ll']/npair.sum():.5f}, "
          f"A mean off-diag rate={A[~np.eye(20,dtype=bool)].mean():.4f} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
