"""Diagnostic: is `bucketed_tree_log_lik_padded_binned` retracing every
call because of fresh closures?

Test: wrap the vmapped forward with an EXPLICIT jax.jit so JAX caches
by function id. Time 5 successive calls.

If call 2..5 are near-instant, the original design was retracing.
If they're still ~60s, something else is the bottleneck.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

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
    _bucket_shape_batch, _fill_pi_arch_classes,
    _stack_padded_batch_binned, bucket_key_from_padded)
from tkfdp.coupling.dynfield.phylo_elbo.tree_log_lik_jax_binned import (
    tree_log_lik_jax_binned)
from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
    build_padded_tree)
from tkfdp.lg08 import S_LG08


def main():
    print(f"# JAX devices: {jax.devices()}")
    jax.config.update("jax_enable_x64", True)

    K_c = 3
    L = 2
    A = 20
    rho_chain = 0.1

    rng = np.random.default_rng(0)

    import json
    from pathlib import Path
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

    bucket = bucket_key_from_padded(clusters[0][0], clusters[0][1])
    shape_batch = _bucket_shape_batch(clusters, bucket)
    vary_batch = _fill_pi_arch_classes(clusters, bucket, pi_field)
    child_bin = _stack_padded_batch_binned(clusters, bucket, bin_idx)

    rho_j = jnp.asarray(rho, dtype=jnp.float64)
    beta_j = jnp.asarray(tables['beta'], dtype=jnp.float64)
    W_j = jnp.asarray(tables['W'], dtype=jnp.float64)
    P_sub_j = jnp.asarray(tables['P_sub'], dtype=jnp.float64)

    def _forward_one(leaf_obs_i, leaf_mask_i, cp_i, cb_i, id_i,
                        root_i, pi_arch_i, classes_i,
                        rho_j, beta_j, W_j, P_sub_j):
        return tree_log_lik_jax_binned(
            leaf_obs_i, leaf_mask_i, cp_i, cb_i, id_i, root_i,
            pi_arch_i, rho_j, beta_j, W_j, P_sub_j, classes_i)

    # Version 1: fresh vmap per call (matches current implementation).
    print("\n### Test A: fresh jax.vmap per call ###")
    for i in range(3):
        t0 = time.time()
        vmapped = jax.vmap(_forward_one, in_axes=(
            0, 0, tuple([0] * len(shape_batch['child_pos_by_level'])),
            tuple([0] * len(child_bin)),
            tuple([0] * len(shape_batch['is_identity_by_level'])),
            0, 0, 0, None, None, None, None))
        ll = vmapped(
            shape_batch['leaf_obs'], shape_batch['leaf_mask'],
            shape_batch['child_pos_by_level'], child_bin,
            shape_batch['is_identity_by_level'], shape_batch['root_slot'],
            vary_batch['pi_arch'], vary_batch['classes'],
            rho_j, beta_j, W_j, P_sub_j)
        jax.block_until_ready(ll)
        print(f"  call {i+1}: {time.time()-t0:.2f}s")

    # Version 2: JIT once, call many times.
    print("\n### Test B: jax.jit'd vmap, cached ###")
    vmapped_jit = jax.jit(jax.vmap(_forward_one, in_axes=(
        0, 0, tuple([0] * len(shape_batch['child_pos_by_level'])),
        tuple([0] * len(child_bin)),
        tuple([0] * len(shape_batch['is_identity_by_level'])),
        0, 0, 0, None, None, None, None)))
    for i in range(3):
        t0 = time.time()
        ll = vmapped_jit(
            shape_batch['leaf_obs'], shape_batch['leaf_mask'],
            shape_batch['child_pos_by_level'], child_bin,
            shape_batch['is_identity_by_level'], shape_batch['root_slot'],
            vary_batch['pi_arch'], vary_batch['classes'],
            rho_j, beta_j, W_j, P_sub_j)
        jax.block_until_ready(ll)
        print(f"  call {i+1}: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
