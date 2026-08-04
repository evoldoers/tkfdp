"""Gibbs sampling of site-class assignments c_n (M6).

For each cluster and each site n within it, sample c_n from its
full conditional under the phylo-ELBO marginal likelihood:
  P(c_n = c | ...) prop alpha_c[c] * exp(marginal_ll(cluster
    | classes with position n set to c))
where alpha_c is a per-class prior (uniform by default) and
marginal_ll is via bucketed_tree_log_lik.

Cost per cluster per site: K_c bucketed_tree_log_lik evaluations
(each on that ONE cluster). For a corpus with sum(cluster.m) total
sites, that's sum_i m_i * K_c single-cluster forwards per sweep.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .tree import Tree
from .tree_batch import (
    BucketBatchCache, bucketed_tree_log_lik,
    bucketed_tree_log_lik_padded, bucketed_tree_log_lik_padded_binned)
from .tau_binning import precompute_kernel_tables


def gibbs_cn_sweep(
    clusters,
    pi_field: np.ndarray,
    rho: np.ndarray,
    S: np.ndarray,
    rho_chain: float,
    K_c: int,
    rng: np.random.Generator,
    alpha_c: 'np.ndarray | None' = None,
    verbose: bool = False,
    padded: bool = False,
    cache: 'BucketBatchCache | None' = None,
    binned: bool = False,
    bin_idx_per_cluster: 'list[list[np.ndarray]] | None' = None,
    bin_centers: 'np.ndarray | None' = None,
) -> 'tuple[list[np.ndarray], dict]':
    """One Gibbs sweep over all site-class assignments c_n.

    Args:
      clusters: list of (Tree, classes) pairs. classes is mutated
        in the copy returned.
      pi_field: (K_c, L, A) — already derived from pi_archetype +
        arch_assignment.
      rho, S, rho_chain: rest of the model.
      K_c: number of classes to consider.
      rng: np.random.Generator.
      alpha_c: (K_c,) prior weights; default uniform.
      verbose: per-site printing.

    Returns:
      (new_classes_list, info) where new_classes_list[i] is the updated
      classes for cluster i, and info reports counts.
    """
    if alpha_c is None:
        alpha_c = np.ones(K_c, dtype=np.float64) / K_c
    log_prior = np.log(np.maximum(alpha_c, 1e-300))
    _forward = (bucketed_tree_log_lik_padded if padded
                    else bucketed_tree_log_lik)
    if binned:
        assert padded, "binned mode requires padded=True"
        assert bin_idx_per_cluster is not None and bin_centers is not None, \
            "binned mode needs bin_idx_per_cluster + bin_centers"
        # pi_field is constant across c_n candidates; precompute tables
        # ONCE per sweep.
        tables = precompute_kernel_tables(
            bin_centers, rho, rho_chain, pi_field, S)

    new_classes = [np.asarray(cl[1], dtype=np.int32).copy()
                    for cl in clusters]

    n_sites_total = 0
    n_kept = 0
    n_changed = 0

    for ci in range(len(clusters)):
        tree, _ = clusters[ci]
        classes = new_classes[ci]
        m = int(classes.shape[0])
        # Random order of sites to avoid systematic bias.
        site_order = rng.permutation(m)
        for n in site_order:
            n_sites_total += 1
            if padded:
                # M10c: build K_c copies of the SAME tree with variants of
                # classes (only entry n differs per copy). Single batched
                # forward returns K_c log-liks in one JIT call.
                classes_variants = np.tile(classes[None, :], (K_c, 1))
                for c in range(K_c):
                    classes_variants[c, n] = c
                batch_clusters = [(tree, classes_variants[c])
                                     for c in range(K_c)]
                if binned:
                    # K_c copies share the same bin indices.
                    bin_list = [bin_idx_per_cluster[ci]] * K_c
                    ll = bucketed_tree_log_lik_padded_binned(
                        batch_clusters, bin_list, rho, tables,
                        cache=cache)
                else:
                    ll = _forward(
                        batch_clusters, rho, pi_field, S, rho_chain,
                        cache=cache)
                log_L = np.asarray(ll, dtype=np.float64)
            else:
                # Sequential fallback for Tree-based path.
                log_L = np.zeros(K_c, dtype=np.float64)
                for c in range(K_c):
                    trial = classes.copy()
                    trial[n] = c
                    ll = _forward(
                        [(tree, trial)], rho, pi_field, S, rho_chain)
                    log_L[c] = float(ll[0])
            log_L += log_prior
            log_L -= log_L.max()
            p = np.exp(log_L)
            p /= p.sum()
            c_new = int(rng.choice(K_c, p=p))
            if c_new == int(classes[n]):
                n_kept += 1
            else:
                n_changed += 1
            if verbose:
                print(f"    cluster={ci}, site={n}: p={p.round(3).tolist()}"
                        f" → c={c_new}", flush=True)
            classes[n] = c_new
        new_classes[ci] = classes

    info = {
        'n_sites_total': n_sites_total,
        'n_kept': n_kept,
        'n_changed': n_changed,
    }
    return new_classes, info
