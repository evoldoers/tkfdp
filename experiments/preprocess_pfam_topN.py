"""Preprocess the N deepest Pfam seed families into ready-to-load .npz files.

Phase 1 (cheap): scan all family .sto files, count sequences (= seed depth).
Phase 2 (medium): for the top-N by depth, parse alignment + tree, extract
cherries, save (aa_a, aa_b, tau, family, L, n_cherries) per family.
Phase 3 (manifest): index.json with the picked family list + depths.

Output layout:
  data/pfam_processed_topN/<family>.npz
  data/pfam_processed_topN/index.json

Each .npz: aa_a (int8 [C, L]), aa_b (int8 [C, L]), tau (float64 [C]),
L (int), n_cherries (int), family (str).

Usage:
  python3 experiments/preprocess_pfam_topN.py --top-n 1000 \
      --out-dir data/pfam_processed_top1000

The seed-depth heuristic is a proxy for cherry count; deeper alignments
generally yield more cherries. Some folds have unusual seed/tree
relationships, so the actual cherry count after parsing may differ from
the depth rank — the manifest records both.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PARENT, "src"))

from tkfdp.bio import (PFAM_SEED_DIR, PFAM_TREE_DIR, has_family,
                          load_family, parse_stockholm)
from tkfdp.pfam_data import family_cherries


def count_sequences(sto_path: Path) -> int:
    """Cheap sequence count: lines starting with a non-comment, non-blank
    word that is not the // terminator."""
    count = 0
    opener = gzip.open if str(sto_path).endswith(".gz") else open
    try:
        with opener(sto_path, "rt") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line: continue
                if line.startswith("#"): continue
                if line.startswith("//"): continue
                count += 1
    except Exception:
        return 0
    return count


def survey_depths(seed_dir: Path) -> list[tuple[str, int]]:
    """Return [(family, depth)] sorted by depth desc."""
    out = []
    for sto in sorted(seed_dir.glob("PF*.sto")):
        fam = sto.stem
        n = count_sequences(sto)
        if n > 0:
            out.append((fam, n))
    out.sort(key=lambda t: -t[1])
    return out


def process_family(fam: str, out_dir: Path) -> dict:
    """Parse + cherry-extract a family, save .npz, return metadata dict."""
    target = out_dir / f"{fam}.npz"
    if target.exists():
        # Skip already-processed (idempotent re-runs).
        arrs = np.load(target, allow_pickle=False)
        return dict(family=fam, status="skipped", L=int(arrs["L"]),
                       n_cherries=int(arrs["n_cherries"]))
    try:
        if not has_family(fam):
            return dict(family=fam, status="missing")
        fd = load_family(fam)
        fc = family_cherries(fd)
        if fc.n_cherries == 0:
            return dict(family=fam, status="no_cherries")
        np.savez(target,
                   aa_a=fc.aa_a, aa_b=fc.aa_b, tau=fc.tau,
                   L=np.int32(fc.L), n_cherries=np.int32(fc.n_cherries),
                   family=np.array(fam, dtype=str))
        return dict(family=fam, status="ok", L=int(fc.L),
                       n_cherries=int(fc.n_cherries))
    except Exception as e:
        return dict(family=fam, status="error", error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path,
                       default=Path("data/pfam_processed_top1000"))
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--max-L", type=int, default=200,
                       help="Skip families with L > max-L (Gibbs is O(L^2)).")
    ap.add_argument("--depth-cache", type=Path, default=None,
                       help="Optional path to cache the depth survey JSON.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Surveying seed depths in {PFAM_SEED_DIR} ...")
    if args.depth_cache and args.depth_cache.exists():
        depths = json.load(open(args.depth_cache))
        depths = [(d[0], int(d[1])) for d in depths]
        print(f"  Loaded {len(depths)} depth entries from {args.depth_cache}")
    else:
        depths = survey_depths(PFAM_SEED_DIR)
        print(f"  {len(depths)} families with non-empty seed alignments")
        if args.depth_cache:
            with open(args.depth_cache, "w") as f:
                json.dump(depths, f)
            print(f"  Cached depths to {args.depth_cache}")

    # Pick top N (deepest) candidates; we'll filter further by L during processing.
    candidates = depths[: args.top_n * 3]
    print(f"  Processing top {len(candidates)} candidates (will filter to "
            f"top {args.top_n} after L < {args.max_L} filter)")

    # Process in parallel. process_family is CPU-bound parsing, no GPU.
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(process_family, fam, args.out_dir): fam
                     for fam, _ in candidates}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            results.append(r)
            if (i + 1) % 100 == 0:
                ok = sum(1 for r in results if r["status"] in ("ok", "skipped"))
                print(f"  Processed {i+1}/{len(candidates)}: ok/skipped={ok}")

    # Filter by status + L.
    good = [r for r in results if r["status"] in ("ok", "skipped")]
    good_with_L = [r for r in good if r["L"] <= args.max_L]
    good_with_L.sort(key=lambda r: -r["n_cherries"])
    chosen = good_with_L[: args.top_n]

    print(f"\nResults:")
    print(f"  total processed:          {len(results)}")
    print(f"  successful (ok+skipped):  {len(good)}")
    print(f"  with L <= {args.max_L}:           {len(good_with_L)}")
    print(f"  picked top {args.top_n}:           {len(chosen)}")
    print(f"  total cherries:           {sum(r['n_cherries'] for r in chosen)}")
    print(f"  total columns:            {sum(r['L'] for r in chosen)}")

    # Write index.
    index = dict(
        families=[r["family"] for r in chosen],
        L=[r["L"] for r in chosen],
        n_cherries=[r["n_cherries"] for r in chosen],
        max_L=args.max_L,
        top_n=args.top_n,
        out_dir=str(args.out_dir),
    )
    with open(args.out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nWrote {args.out_dir}/index.json with {len(chosen)} families")
    print(f"Per-family .npz files in {args.out_dir}/")


if __name__ == "__main__":
    main()
