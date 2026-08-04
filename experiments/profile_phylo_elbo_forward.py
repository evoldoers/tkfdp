"""Profile a single bucketed_tree_log_lik_padded_binned call to find
where the ~30s / forward comes from on 2315-leaf trees.

Instruments:
  - PaddedTree build time
  - shape batch build time
  - jax.numpy transfer (numpy -> jnp.asarray)
  - JAX compile time (first call vs subsequent)
  - Actual JAX execution time (block_until_ready)

Reports each in seconds. Prints as we go with flush=True.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from tkfdp.coupling.dynfield.phylo_elbo.pfam_loader import (
    load_pfam_clusters)
from tkfdp.coupling.dynfield.phylo_elbo.tau_binning import (
    build_tau_bins, collect_all_taus, precompute_kernel_tables,
    rebuild_padded_trees_with_bins)
from tkfdp.coupling.dynfield.phylo_elbo.tree_batch import (
    BucketBatchCache, _bucket_shape_batch, _fill_pi_arch_classes,
    _stack_padded_batch_binned, bucket_key_from_padded,
    bucketed_tree_log_lik_padded_binned)
from tkfdp.coupling.dynfield.phylo_elbo.tree_log_lik_jax_binned import (
    tree_log_lik_jax_binned)
from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
    build_padded_tree)
from tkfdp.lg08 import S_LG08


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-families", type=int, default=1)
    ap.add_argument("--cluster-m", type=int, default=8)
    ap.add_argument("--K-c", type=int, default=3)
    ap.add_argument("--n-tau-bins", type=int, default=8)
    ap.add_argument("--limit-clusters", type=int, default=3)
    ap.add_argument("--n-repeats", type=int, default=5,
                       help="Number of times to call the forward (to see "
                             "warm-up vs steady-state timing).")
    args = ap.parse_args()

    print(f"# JAX devices: {jax.devices()}", flush=True)
    jax.config.update("jax_enable_x64", True)

    K_c = args.K_c
    L = 2
    A = 20
    rho_chain = 0.1

    rng = np.random.default_rng(0)

    print(f"# Loading {args.n_families} families ...", flush=True)
    t0 = time.time()
    import json
    from pathlib import Path
    clv_dir = Path("data/pfam_processed_clv_top1000")
    with (clv_dir / "index.json").open() as f:
        idx = json.load(f)
    fam_ids = idx['families'][:args.n_families]
    clv_paths = [str(clv_dir / f"{f}.npz") for f in fam_ids
                    if (clv_dir / f"{f}.npz").exists()]
    clusters = load_pfam_clusters(clv_paths, args.cluster_m, K_c, rng)
    if args.limit_clusters > 0:
        clusters = clusters[:args.limit_clusters]
    print(f"  loaded {len(clusters)} clusters in {time.time()-t0:.2f}s",
            flush=True)

    print("# Building PaddedTrees ...", flush=True)
    t0 = time.time()
    clusters = [(build_padded_tree(t), c) for (t, c) in clusters]
    print(f"  {time.time()-t0:.2f}s", flush=True)
    for i, (pt, c) in enumerate(clusters):
        print(f"  cluster {i}: n_leaves={pt.n_leaves_actual} "
                f"D_bucket={pt.D_bucket} m={pt.m}", flush=True)

    print("# tau binning ...", flush=True)
    t0 = time.time()
    pts_only = [pt for pt, _ in clusters]
    all_taus = collect_all_taus(pts_only)
    bin_centers = build_tau_bins(all_taus, n_bins=args.n_tau_bins)
    _, bin_idx_per_cluster = rebuild_padded_trees_with_bins(pts_only, bin_centers)
    print(f"  {time.time()-t0:.2f}s "
            f"(range {bin_centers[0]:.4f}..{bin_centers[-1]:.3f})", flush=True)

    rng2 = np.random.default_rng(1)
    pi_field = rng2.dirichlet(np.ones(A), size=(K_c, L))
    rho = np.full(L, 1.0/L)
    S = np.asarray(S_LG08, dtype=np.float64)

    print("# precompute_kernel_tables ...", flush=True)
    t0 = time.time()
    tables = precompute_kernel_tables(bin_centers, rho, rho_chain, pi_field, S)
    print(f"  {time.time()-t0:.2f}s "
            f"(P_sub shape {tables['P_sub'].shape})", flush=True)

    # ---- INSTRUMENTED FORWARD ----
    # Break bucketed_tree_log_lik_padded_binned into pieces.
    bucket = bucket_key_from_padded(clusters[0][0], clusters[0][1])
    print(f"# bucket key = {bucket}", flush=True)

    # 1. Shape batch build (numpy -> jnp).
    print("# _bucket_shape_batch (numpy stack + jnp transfer) ...", flush=True)
    t0 = time.time()
    shape_batch = _bucket_shape_batch(clusters, bucket)
    print(f"  {time.time()-t0:.2f}s", flush=True)

    # 2. pi_arch/classes fill.
    t0 = time.time()
    vary_batch = _fill_pi_arch_classes(clusters, bucket, pi_field)
    print(f"# _fill_pi_arch_classes: {time.time()-t0:.2f}s", flush=True)

    # 3. bin arrays.
    t0 = time.time()
    child_bin_by_level = _stack_padded_batch_binned(
        clusters, bucket, bin_idx_per_cluster)
    print(f"# _stack_padded_batch_binned: {time.time()-t0:.2f}s", flush=True)

    # 4. Kernel table transfer.
    t0 = time.time()
    rho_j = jnp.asarray(rho, dtype=jnp.float64)
    beta_j = jnp.asarray(tables['beta'], dtype=jnp.float64)
    W_j = jnp.asarray(tables['W'], dtype=jnp.float64)
    P_sub_j = jnp.asarray(tables['P_sub'], dtype=jnp.float64)
    jax.block_until_ready(beta_j)
    print(f"# kernel table transfer: {time.time()-t0:.2f}s", flush=True)

    # 5. Actual vmapped forward. Time the compile + N calls.
    def _forward_one(leaf_obs_i, leaf_mask_i, cp_i, cb_i, id_i,
                        root_i, pi_arch_i, classes_i):
        return tree_log_lik_jax_binned(
            leaf_obs_i, leaf_mask_i, cp_i, cb_i, id_i, root_i,
            pi_arch_i, rho_j, beta_j, W_j, P_sub_j, classes_i)

    vmapped = jax.vmap(_forward_one, in_axes=(
        0, 0, tuple([0] * len(shape_batch['child_pos_by_level'])),
        tuple([0] * len(child_bin_by_level)),
        tuple([0] * len(shape_batch['is_identity_by_level'])),
        0, 0, 0))

    for i in range(args.n_repeats):
        t0 = time.time()
        ll_batch = vmapped(
            shape_batch['leaf_obs'], shape_batch['leaf_mask'],
            shape_batch['child_pos_by_level'], child_bin_by_level,
            shape_batch['is_identity_by_level'], shape_batch['root_slot'],
            vary_batch['pi_arch'], vary_batch['classes'])
        jax.block_until_ready(ll_batch)
        dt = time.time() - t0
        label = "compile+run" if i == 0 else "run"
        print(f"# vmapped forward call {i+1}/{args.n_repeats} ({label}): "
                f"{dt:.2f}s  ll={np.asarray(ll_batch).round(4).tolist()}",
                flush=True)


if __name__ == "__main__":
    main()
