#!/usr/bin/env python3
"""Structure-SUPERVISED corpus fit of the permfield mood-light model.

Wires REAL PDB structural-contact clusters (from
experiments/build_pdb_partition.py, precomputed under
data/pdb_partition_clv_top1000_sifts/) into the corpus-scale permfield trainer
`experiments.permfield_corpus.fit_corpus` as the partition_fn.

This model is meaningless without structure supervision: it is DESIGNED to be
run with contact pairs as m=2 clusters. Singletons (the fallback) or one giant
cluster cannot supply the coupling structure the field is meant to explain, and
leave rho ~uniform. Here every stored (col_i, col_j) contact becomes one shared
m=2 field cluster and every remaining column is a singleton, covering all L
columns exactly once.

Column indexing: the partition npz stores `pairs` as column indices into the
raw CLV alignment (0..L-1); the thin128 CLV corpus that permfield_corpus loads
uses the SAME columns (leaf-thinning prunes leaves, never columns), which is
verified by matching each partition file's stored `L` against fam['L'] and
refusing any mismatch. This is exactly the guarantee the dynfield consumer
`corpus_state.apply_pdb_partition` relies on.

Does NOT modify experiments/permfield_corpus.py or experiments/permfield_elbo.py.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, os.getcwd())
# Default to CPU so a stray run never grabs a shared GPU (GPU 0 is in use by another
# user). This is numpy/scipy-bound anyway; to use the free GPU explicitly, set
# CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda before invoking.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from experiments.permfield_corpus import (          # noqa: E402
    load_families, fit_corpus, _entropy, singleton_partition,
)

PART_DIR_DEFAULT = "data/pdb_partition_clv_top1000_sifts"


# ---------- partition provider ----------
def load_partition_index(part_dir):
    """Read every <FAM>.npz in the partition dir. Returns
    {fam_name: dict(L=int, pairs=(n,2) int array, kinds=list[str])}."""
    idx = {}
    for p in sorted(glob.glob(os.path.join(part_dir, "*.npz"))):
        fam = os.path.splitext(os.path.basename(p))[0]
        d = np.load(p, allow_pickle=True)
        pairs = np.asarray(d["pairs"], dtype=np.int64).reshape(-1, 2)
        kinds = (np.asarray(d["kind"]).tolist() if "kind" in d.files
                 else [""] * pairs.shape[0])
        idx[fam] = dict(L=int(d["L"]), pairs=pairs, kinds=kinds)
    return idx


def make_partition_fn(part_index):
    """Build a partition_fn(fam)->list[np.ndarray]. Each contact pair becomes an
    m=2 cluster; every other column is a singleton. Asserts the clusters cover
    0..L-1 exactly once (no overlap, no gap). Pairs that would reuse a column
    already claimed by an earlier pair are dropped (greedy disjoint matching) so
    the cover invariant always holds -- the SIFTS partitions are already disjoint
    matchings in practice, so this drops nothing, but it keeps the invariant
    provable."""
    def partition_fn(fam):
        name = fam["name"]
        L = fam["L"]
        entry = part_index.get(name)
        if entry is None or int(entry["L"]) != L:
            # no structure / column-indexing mismatch -> singletons (defensive;
            # callers should pre-filter to contact families with matching L)
            return [np.array([j], dtype=np.int64) for j in range(L)]
        used = np.zeros(L, dtype=bool)
        clusters = []
        for i, j in entry["pairs"]:
            i, j = int(i), int(j)
            if not (0 <= i < L and 0 <= j < L) or i == j:
                continue
            if used[i] or used[j]:
                continue                                   # keep disjoint
            used[i] = used[j] = True
            clusters.append(np.array([i, j], dtype=np.int64))
        for j in range(L):
            if not used[j]:
                clusters.append(np.array([j], dtype=np.int64))
        # --- cover invariant: union is exactly 0..L-1, no overlap ---
        allcols = np.concatenate(clusters) if clusters else np.array([], int)
        assert allcols.size == L, f"{name}: cover size {allcols.size} != L {L}"
        assert np.array_equal(np.sort(allcols), np.arange(L)), \
            f"{name}: clusters do not cover 0..L-1 exactly once"
        return clusters
    return partition_fn


def select_contact_families(fams, part_index):
    """Keep only loaded families that have >=1 PDB contact pair AND matching L."""
    kept, n_pairs, kind_totals = [], 0, {}
    for fam in fams:
        e = part_index.get(fam["name"])
        if e is None or int(e["L"]) != fam["L"] or e["pairs"].shape[0] == 0:
            continue
        kept.append(fam)
        n_pairs += int(e["pairs"].shape[0])
        for k in e["kinds"]:
            kind_totals[k] = kind_totals.get(k, 0) + 1
    return kept, n_pairs, kind_totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--n-families", type=int, default=200,
                    help="cap on contact families to fit (after filtering)")
    ap.add_argument("--load-scan", type=int, default=1000,
                    help="how many corpus families to scan/load before filtering")
    ap.add_argument("--max-leaves", type=int, default=24)
    ap.add_argument("--sweeps", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--learn-alpha", action="store_true")
    ap.add_argument("--clv-dir", default="data/pfam_processed_clv_top1000_thin128")
    ap.add_argument("--part-dir", default=PART_DIR_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--singleton", action="store_true",
                    help="CONTROL: fit the SAME contact families but with "
                         "singletons (no structure) for a matched contrast")
    ap.add_argument("--init-mode", choices=["perturb", "dirichlet"], default="perturb",
                    help="archetype init: 'dirichlet' breaks the relabelling symmetry harder")
    ap.add_argument("--init-noise", type=float, default=0.1,
                    help="perturb-mode: multiplicative-lognormal noise scale")
    ap.add_argument("--init-kappa", type=float, default=5.0,
                    help="dirichlet-mode: Dirichlet(kappa*pi_LG08); smaller = more distinct")
    ap.add_argument("--alpha-final", type=float, default=None,
                    help="if set, ANNEAL the field prior geometrically from --alpha (frozen) "
                         "to --alpha-final (active) over the sweeps")
    args = ap.parse_args()

    print(f"# STRUCTURE-SUPERVISED permfield corpus fit  C={args.C}  "
          f"sweeps={args.sweeps}  max_leaves={args.max_leaves}  "
          f"alpha0={args.alpha}  part_dir={args.part_dir}", flush=True)

    part_index = load_partition_index(args.part_dir)
    print(f"# partition index: {len(part_index)} families with PDB partitions",
          flush=True)

    fams_all = load_families(args.clv_dir, args.load_scan, args.max_leaves,
                             verbose=False)
    fams, n_pairs, kind_totals = select_contact_families(fams_all, part_index)
    fams = fams[:args.n_families]
    # recompute pair/kind totals over the (possibly capped) selection
    n_pairs = sum(int(part_index[f["name"]]["pairs"].shape[0]) for f in fams)
    ncols = sum(f["L"] for f in fams); nleaf = sum(f["nl"] for f in fams)
    print(f"# loaded {len(fams_all)} corpus families; "
          f"{len(fams)} kept (>=1 contact, L-matched, capped)", flush=True)
    print(f"# fitting {len(fams)} families: {ncols} columns, {nleaf} leaves, "
          f"{n_pairs} contact m=2 clusters", flush=True)
    print(f"# contact-kind totals over selection = {kind_totals}", flush=True)

    if args.singleton:
        print("# CONTROL MODE: singletons on the same contact families "
              "(no structure supervision)", flush=True)
        partition_fn = singleton_partition
    else:
        partition_fn = make_partition_fn(part_index)

    t0 = time.time()
    res = fit_corpus(fams, args.C, sweeps=args.sweeps, alpha0=args.alpha,
                     learn_alpha=args.learn_alpha, seed=args.seed,
                     partition_fn=partition_fn, verbose=True,
                     init_mode=args.init_mode, init_noise=args.init_noise,
                     init_kappa=args.init_kappa, alpha_final=args.alpha_final)
    dt = time.time() - t0

    hist = np.asarray(res["hist"]); diffs = np.diff(hist)
    mono = bool(np.all(diffs >= -1e-4))
    worst = float(diffs.min()) if len(diffs) else float("nan")
    print(f"\n# ===== STRUCTURE-SUPERVISED RESULT (C={args.C}) =====")
    print(f"# fit {dt:.1f}s over {args.sweeps} sweeps, {len(fams)} families")
    print(f"# objective monotone non-decreasing = {mono}  "
          f"(worst delta = {worst:+.4f})")
    print(f"# final corpus obj = {hist[-1]:.2f}")
    print(f"# pooled rho    = {np.round(res['rho'], 4)}")
    print(f"# pooled alpha  = {res['alpha']:.4f}")
    print(f"# pooled pfield = {np.round(res['pif'], 4)}")
    print(f"# field s = {np.round(res['s'], 4)}  w = {np.round(res['w'], 4)}")
    ent = [_entropy(res["pis"][a]) for a in range(args.C)]
    print(f"# per-archetype pi entropy (nats) = {np.round(ent, 4)}  "
          f"(uniform-20 = {np.log(20):.4f})")


if __name__ == "__main__":
    main()
