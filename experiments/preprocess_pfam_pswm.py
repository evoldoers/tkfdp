"""Preprocess Pfam families into per-node LG08 Felsenstein CLVs.

For each family with an existing FastTree guide tree, run per-site LG08
Felsenstein pruning to compute the bottom-up conditional-likelihood
vector at every node. The CLV lets the training loop draw whole-tree
joint histories under LG08 as an importance-sampling proposal for the
TKF-DP posterior (see math-paper appendix par:arch-lg08-is).

Output schema:

  <out_dir>/<family>.npz:
    - family:              str
    - L:                   int (alignment length)
    - n_nodes:             int (2N-1 for a rooted binary N-leaf tree)
    - n_leaves:            int
    - parent:              (n_nodes,) int32; -1 at root
    - tau:                 (n_nodes,) float64; branch length to parent
    - clv:                 (n_nodes, L, A=20) float32; rescaled CLV
                              (max=1 per node/site for fp32 stability)
    - log_scale:           (n_nodes, L) float32; log-scale accumulated
                              during rescaling. True CLV entry is
                              clv[v, s, a] * exp(log_scale[v, s]).
    - log_p_lg_per_site:   (L,) float64; LG08 tree marginal log-lik
                              log P^LG(y_s | tree, LG08) = log(pi @
                              clv[root, s]) + log_scale[root, s].
    - leaf_msa:            (n_leaves, L) int8; observed leaf residues,
                              gap = 20 (aligns with clv-encoded deltas).
    - root_id:             int

Cost per family: O(L * N_nodes * A^2) for peeling (a few seconds each).
Storage: ~12-15 MB per family; 1000 families ~= 13 GB.

Retires the earlier marginal-PSWM output (pswm_a / pswm_b): those
tensors were the per-node PSWM = normalize(outside * clv), which is
correct as a marginal but discards the branch-length correlation
needed for training. See par:arch-lg08-is footnote in the appendix.

Usage:
  python3 experiments/preprocess_pfam_pswm.py \\
      --seed-index data/pfam_processed_top1000/index.json \\
      --out-dir data/pfam_processed_clv_top1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PARENT, "src"))

from tkfdp.bio import (has_family, load_family, PFAM_SEED_DIR,
                          PFAM_TREE_DIR)
from tkfdp.pswm_peeling import compute_pswm_family


def process_family(fam: str, out_dir: 'str | Path', max_L: int = 0) -> dict:
    """Process one family; return metadata dict.

    Idempotent: skip if output already exists AND loads cleanly.
    If max_L > 0, families with L > max_L are skipped with an error.
    """
    out_dir = Path(out_dir)
    target = out_dir / f"{fam}.npz"
    if target.exists():
        try:
            arrs = np.load(target, allow_pickle=False)
            return dict(family=fam,
                          status="skipped",
                          L=int(arrs['L']),
                          n_nodes=int(arrs['n_nodes']),
                          n_leaves=int(arrs['n_leaves']))
        except Exception:
            pass

    if not has_family(fam):
        return dict(family=fam, status="missing")

    try:
        fd = load_family(fam)
    except Exception as e:
        return dict(family=fam, status="error",
                     error=f"load_family: {e}")
    L = int(fd.msa.shape[1])
    if L == 0 or fd.msa.shape[0] < 2:
        return dict(family=fam, status="error",
                     error=f"empty/tiny MSA (L={L}, n_seq={fd.msa.shape[0]})")
    if max_L > 0 and L > max_L:
        return dict(family=fam, status="skip_L",
                     L=L, msg=f"L={L} > max_L={max_L}")

    try:
        peel = compute_pswm_family(fd.msa, fd.names, fd.tree)
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return dict(family=fam, status="error",
                     error=f"peeling: {e}\n{tb}")

    n_nodes = peel['n_nodes']
    n_leaves = peel['n_leaves']

    # Observed leaf residues (aligns with clv-encoded deltas at leaves).
    leaf_msa = np.full((n_leaves, L), 20, dtype=np.int8)   # 20 = gap
    for i in range(n_leaves):
        row = int(peel['leaf_msa_row'][i])
        if row >= 0:
            leaf_msa[i] = fd.msa[row].astype(np.int8)

    np.savez(target,
              family=np.array(fam, dtype=str),
              L=np.int64(L),
              n_nodes=np.int64(n_nodes),
              n_leaves=np.int64(n_leaves),
              root_id=np.int64(peel['root_id']),
              parent=peel['parent'].astype(np.int32),
              tau=peel['tau'].astype(np.float64),
              clv=peel['clv'].astype(np.float32),
              log_scale=peel['log_scale'].astype(np.float32),
              log_p_lg_per_site=peel['log_p_lg_per_site'].astype(np.float64),
              leaf_msa=leaf_msa)

    return dict(family=fam,
                  status="ok",
                  L=L,
                  n_nodes=n_nodes,
                  n_leaves=n_leaves)


def _list_families_with_trees() -> 'list[str]':
    fams = []
    for sto in sorted(PFAM_SEED_DIR.glob("PF*.sto")):
        fam = sto.stem
        if (PFAM_TREE_DIR / f"{fam}.nwk").exists():
            fams.append(fam)
    return fams


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", type=str, default=None,
                     help="comma-separated family list; overrides --seed-index")
    ap.add_argument("--seed-index", type=Path, default=None,
                     help="reuse family list from an existing "
                          "pfam_processed_top*/index.json")
    ap.add_argument("--all", action="store_true",
                     help="process all families with .sto + .nwk (careful: 25k)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-L", type=int, default=200,
                     help="skip families with L > max_L (0 = no cap)")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.families:
        fams = [f.strip() for f in args.families.split(",") if f.strip()]
    elif args.seed_index:
        idx = json.load(open(args.seed_index))
        fams = list(idx['families'])
    elif args.all:
        fams = _list_families_with_trees()
    else:
        print("either --families, --seed-index, or --all required",
              file=sys.stderr)
        sys.exit(2)

    print(f"[preprocess_pswm] processing {len(fams)} families -> {args.out_dir} "
          f"(max_L={args.max_L}, workers={args.n_workers})", flush=True)

    results: 'list[dict]' = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(process_family, fam, str(args.out_dir),
                              args.max_L): fam for fam in fams}
        report_every = max(1, len(fams) // 40)
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            results.append(r)
            if args.verbose or (i % report_every == 0):
                summary = {k: r[k] for k in ('family', 'status', 'L',
                                                'n_nodes', 'n_leaves')
                             if k in r}
                if 'error' in r:
                    summary['error'] = r['error'][:120]
                if 'msg' in r:
                    summary['msg'] = r['msg']
                print(f"  [{i+1}/{len(fams)}] {summary}", flush=True)

    ok_results = [r for r in results if r.get('status') in ('ok', 'skipped')]

    manifest = dict(
        n_requested=len(fams),
        n_ok=len(ok_results),
        max_L=args.max_L,
        source=(str(args.seed_index) if args.seed_index else
                  ("--families" if args.families else "--all")),
        families=[r['family'] for r in ok_results],
        L=[r['L'] for r in ok_results],
        n_nodes=[r['n_nodes'] for r in ok_results],
        n_leaves=[r['n_leaves'] for r in ok_results],
        errors=[r for r in results if r.get('status') not in ('ok', 'skipped')],
    )
    with open(args.out_dir / "index.json", "w") as f:
        json.dump(manifest, f, indent=1)

    total_nodes = sum(r.get('n_nodes', 0) for r in ok_results)
    total_columns = sum(r.get('L', 0) for r in ok_results)
    print(f"[preprocess_pswm] done: {len(ok_results)}/{len(fams)} succeeded, "
          f"total_nodes={total_nodes}, total_columns={total_columns}",
          flush=True)


if __name__ == "__main__":
    main()
