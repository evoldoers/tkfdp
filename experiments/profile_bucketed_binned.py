"""Profile bucketed_tree_log_lik_padded_binned end-to-end after M13."""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tkfdp.coupling.dynfield.phylo_elbo.pfam_loader import (
    load_pfam_clusters)
from tkfdp.coupling.dynfield.phylo_elbo.tau_binning import (
    build_tau_bins, collect_all_taus, precompute_kernel_tables,
    rebuild_padded_trees_with_bins)
from tkfdp.coupling.dynfield.phylo_elbo.tree_batch import (
    BucketBatchCache, bucketed_tree_log_lik_padded_binned)
from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
    build_padded_tree)
from tkfdp.lg08 import S_LG08


def main():
    print(f"# JAX devices: {jax.devices()}")
    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(0)
    K_c = 3; L = 2; A = 20; rho_chain = 0.1

    clv_dir = Path("data/pfam_processed_clv_top1000")
    with (clv_dir / "index.json").open() as f:
        idx = json.load(f)
    clv_paths = [str(clv_dir / f"{idx['families'][0]}.npz")]
    clusters = load_pfam_clusters(clv_paths, 8, K_c, rng)[:3]
    clusters = [(build_padded_tree(t), c) for (t, c) in clusters]
    print(f"# {len(clusters)} clusters, first: n_leaves="
            f"{clusters[0][0].n_leaves_actual} D={clusters[0][0].D_bucket}")

    pts_only = [pt for pt, _ in clusters]
    bin_centers = build_tau_bins(collect_all_taus(pts_only), 8)
    _, bin_idx = rebuild_padded_trees_with_bins(pts_only, bin_centers)

    rng2 = np.random.default_rng(1)
    pi_field = rng2.dirichlet(np.ones(A), size=(K_c, L))
    rho = np.full(L, 1.0/L)
    S = np.asarray(S_LG08, dtype=np.float64)
    tables = precompute_kernel_tables(bin_centers, rho, rho_chain, pi_field, S)

    cache = BucketBatchCache()

    for i in range(5):
        t0 = time.time()
        ll = bucketed_tree_log_lik_padded_binned(
            clusters, bin_idx, rho, tables, cache=cache)
        dt = time.time() - t0
        label = "compile+run" if i == 0 else "run"
        print(f"# call {i+1}/5 ({label}): {dt:.2f}s  ll={ll.round(4).tolist()}")


if __name__ == "__main__":
    main()
