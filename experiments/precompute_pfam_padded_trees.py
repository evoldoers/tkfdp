"""Precompute PaddedTree objects for a Pfam corpus and save to disk (M9).

For each family in a FamilyCLV corpus, chunk the L alignment columns
into fixed-width clusters, build the PaddedTree per cluster (once),
and save to a per-family npz. Training then loads these precomputed
PaddedTrees instead of rebuilding them inside every JAX call.

Output schema:
  <out-dir>/index.json           corpus manifest
  <out-dir>/<family>.npz         all clusters for this family, packed as
                                 cluster_0000_leaf_obs, cluster_0000_child_pos_l0,
                                 ..., cluster_XXXX_...

Usage:
    python experiments/precompute_pfam_padded_trees.py \\
        --clv-dir data/pfam_processed_clv_top1000 \\
        --out-dir data/pfam_precomputed_padded \\
        --cluster-m 8 \\
        [--n-families 10]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tkfdp.coupling.dynfield.phylo_elbo.pfam_loader import (
    family_to_clusters)
from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
    PaddedTree, build_padded_tree)
from tkfdp.pfam_data import load_clv_family


def save_family_padded_trees(family_id: str, clv_path: Path,
                                    out_path: Path, cluster_m: int,
                                    rng: np.random.Generator) -> dict:
    """Precompute and save all PaddedTrees for one family.

    Returns a dict with per-cluster shape info (for the corpus index).
    """
    fd = load_clv_family(str(clv_path))
    # K_c=1 here is a placeholder for classes init (which is discarded);
    # classes are re-initialised at load-time by the trainer.
    clusters = family_to_clusters(fd, cluster_m, K_c=1, rng=rng)
    d: 'dict[str, np.ndarray]' = {}
    per_cluster_info = []
    for i, (tree, _classes) in enumerate(clusters):
        pt = build_padded_tree(tree)
        prefix = f"cluster_{i:04d}_"
        d[prefix + 'N_bucket'] = np.int32(pt.N_bucket)
        d[prefix + 'D_bucket'] = np.int32(pt.D_bucket)
        d[prefix + 'n_leaves_actual'] = np.int32(pt.n_leaves_actual)
        d[prefix + 'depth_actual'] = np.int32(pt.depth_actual)
        d[prefix + 'm'] = np.int32(pt.m)
        d[prefix + 'leaf_obs'] = pt.leaf_obs.astype(np.int32)
        d[prefix + 'leaf_mask'] = pt.leaf_mask.astype(np.float64)
        d[prefix + 'root_slot'] = np.int32(pt.root_slot)
        for ell in range(pt.D_bucket):
            d[prefix + f'child_pos_l{ell}'] = pt.child_pos[ell].astype(np.int32)
            d[prefix + f'child_branch_l{ell}'] = pt.child_branch[ell].astype(np.float64)
            d[prefix + f'slot_mask_l{ell}'] = pt.slot_mask[ell].astype(np.float64)
        per_cluster_info.append({
            'cluster_id': i,
            'N_bucket': int(pt.N_bucket),
            'D_bucket': int(pt.D_bucket),
            'm': int(pt.m),
            'n_leaves_actual': int(pt.n_leaves_actual),
            'depth_actual': int(pt.depth_actual),
        })
    d['n_clusters'] = np.int32(len(clusters))
    d['family_id'] = np.str_(family_id)
    d['cluster_m'] = np.int32(cluster_m)
    np.savez_compressed(out_path, **d)
    return {
        'family_id': family_id,
        'n_clusters': len(clusters),
        'cluster_m': cluster_m,
        'clusters': per_cluster_info,
    }


def load_family_padded_trees(npz_path) -> 'list[PaddedTree]':
    """Load all PaddedTrees for one family from a precomputed npz."""
    d = np.load(npz_path, allow_pickle=False)
    n_clusters = int(d['n_clusters'])
    out = []
    for i in range(n_clusters):
        prefix = f"cluster_{i:04d}_"
        D = int(d[prefix + 'D_bucket'])
        cp = [np.asarray(d[prefix + f'child_pos_l{ell}'], dtype=np.int32)
                 for ell in range(D)]
        cb = [np.asarray(d[prefix + f'child_branch_l{ell}'], dtype=np.float64)
                 for ell in range(D)]
        sm = [np.asarray(d[prefix + f'slot_mask_l{ell}'], dtype=np.float64)
                 for ell in range(D)]
        pt = PaddedTree(
            N_bucket=int(d[prefix + 'N_bucket']),
            D_bucket=D,
            n_leaves_actual=int(d[prefix + 'n_leaves_actual']),
            depth_actual=int(d[prefix + 'depth_actual']),
            m=int(d[prefix + 'm']),
            leaf_obs=np.asarray(d[prefix + 'leaf_obs'], dtype=np.int32),
            leaf_mask=np.asarray(d[prefix + 'leaf_mask'], dtype=np.float64),
            child_pos=cp,
            child_branch=cb,
            slot_mask=sm,
            root_slot=int(d[prefix + 'root_slot']),
        )
        out.append(pt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument("--clv-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--cluster-m", type=int, default=8)
    ap.add_argument("--n-families", type=int, default=0,
                       help="0 = all in index")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    clv_dir = Path(args.clv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (clv_dir / "index.json").open() as f:
        clv_index = json.load(f)
    family_ids = list(clv_index['families'])
    if args.n_families > 0:
        family_ids = family_ids[:args.n_families]

    corpus_manifest = {
        'clv_dir': args.clv_dir,
        'cluster_m': args.cluster_m,
        'n_families_processed': 0,
        'families': [],
    }
    t_start = time.time()
    for family_id in family_ids:
        clv_path = clv_dir / f"{family_id}.npz"
        if not clv_path.exists():
            print(f"  [skip] {family_id}: no CLV npz", flush=True)
            continue
        out_path = out_dir / f"{family_id}.npz"
        t0 = time.time()
        try:
            info = save_family_padded_trees(
                family_id, clv_path, out_path, args.cluster_m, rng)
            corpus_manifest['families'].append(info)
            corpus_manifest['n_families_processed'] += 1
            print(f"  [ok]   {family_id}: {info['n_clusters']} clusters "
                    f"in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  [fail] {family_id}: {e}", flush=True)

    (out_dir / "index.json").write_text(
        json.dumps(corpus_manifest, indent=2))
    print(f"# Done: {corpus_manifest['n_families_processed']}/{len(family_ids)}"
            f" families in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
