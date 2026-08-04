"""Gibbs sampling of arch_assignment[c, theta] under the phylo-ELBO
marginal likelihood (M5).

For each (c*, theta*) with theta* != 0 (optionally), sample
arch_assignment[c*, theta*] from its full conditional:
  P(arch[c*, theta*] = k | ...) prop rho_arch[k]
    * exp(sum_clusters marginal_ll(cluster | arch[c*, theta*] = k))
where marginal_ll is via `bucketed_tree_log_lik` (forward phylo-ELBO
with theta marginalised, pi_archetype frozen).

Efficiency: for each (c*, theta*), only clusters containing at least
one site with c_n == c* are affected. Non-affected clusters' log-lik
is invariant under arch_assignment[c*, theta*] changes, so their
contribution cancels in the softmax.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from .tree import Tree
from .tree_batch import (
    BucketBatchCache, bucketed_tree_log_lik,
    bucketed_tree_log_lik_padded, bucketed_tree_log_lik_padded_binned,
    bucketed_tree_log_lik_padded_kvmap)
from .tau_binning import precompute_kernel_tables
from .tree_padded import PaddedTree


def _clusters_containing_class(clusters,
                                    ) -> 'dict[int, list[int]]':
    """Precompute per-class cluster index list.

    Returns dict {c: [cluster_indices where any site has class c]}.
    """
    per_class: 'dict[int, list[int]]' = defaultdict(list)
    for i, (_, classes) in enumerate(clusters):
        for c in np.unique(np.asarray(classes)):
            per_class[int(c)].append(i)
    return per_class


def _corpus_ll_by_cluster(
    clusters, pi_field, rho, S, rho_chain,
    bin_idx_per_cluster, bin_centers, cache,
) -> np.ndarray:
    """Full-corpus per-cluster LL under current pi_field (binned path)."""
    tables = precompute_kernel_tables(
        bin_centers, rho, rho_chain, pi_field, S)
    return np.asarray(bucketed_tree_log_lik_padded_binned(
        clusters, bin_idx_per_cluster, rho, tables, cache=cache),
        dtype=np.float64)


def swap_arch_assignment_sweep_binned(
    clusters,
    pi_archetype: np.ndarray,
    arch_assignment: np.ndarray,
    rho: np.ndarray,
    S: np.ndarray,
    rho_chain: float,
    bin_idx_per_cluster,
    bin_centers: np.ndarray,
    rng: np.random.Generator,
    fix_theta0: bool = True,
    cache: 'BucketBatchCache | None' = None,
    verbose: bool = False,
) -> 'tuple[np.ndarray, dict]':
    """Permutation-only MH sweep on arch_assignment (M14 + swap flavour).

    For each theta_star (skipping theta=0 if fix_theta0), iterate over
    unordered class pairs (c1 < c2) in random order. Propose swapping
    arch_assignment[c1, theta_star] <-> arch_assignment[c2, theta_star]
    and MH-accept via the marginal-LL ratio (computed on the union of
    clusters containing class c1 or c2). Preserves the multiset of
    archetypes used at each theta.

    Caches full-corpus per-cluster LL under current aa; on accept, only
    the affected slice is refreshed with the ll_swap values we just
    computed (they become the new baseline). Cuts per-pair cost from
    two forward passes to one.
    """
    K_a, A = pi_archetype.shape
    K_c, L = arch_assignment.shape
    aa = np.asarray(arch_assignment, dtype=np.int32).copy()
    per_class = _clusters_containing_class(clusters)

    # Cache full-corpus per-cluster LL under current aa.
    ll_by_cluster = _corpus_ll_by_cluster(
        clusters, pi_archetype[aa], rho, S, rho_chain,
        bin_idx_per_cluster, bin_centers, cache)

    n_proposed = 0
    n_accepted = 0
    theta_range = np.arange(1, L) if fix_theta0 else np.arange(L)

    for theta_star in rng.permutation(theta_range):
        theta_star = int(theta_star)
        pairs = [(c1, c2) for c1 in range(K_c) for c2 in range(c1 + 1, K_c)]
        rng.shuffle(pairs)
        for (c1, c2) in pairs:
            k1 = int(aa[c1, theta_star])
            k2 = int(aa[c2, theta_star])
            if k1 == k2:
                continue
            affected_ci = sorted(
                set(per_class.get(c1, [])) | set(per_class.get(c2, [])))
            if not affected_ci:
                continue
            affected_clusters = [clusters[i] for i in affected_ci]
            affected_bins = [bin_idx_per_cluster[i] for i in affected_ci]
            affected_arr = np.asarray(affected_ci)

            ll_curr_sum = float(ll_by_cluster[affected_arr].sum())

            # Proposed LL (swap) on affected slice only.
            aa_swap = aa.copy()
            aa_swap[c1, theta_star] = k2
            aa_swap[c2, theta_star] = k1
            pi_field_swap = pi_archetype[aa_swap]
            tables_swap = precompute_kernel_tables(
                bin_centers, rho, rho_chain, pi_field_swap, S)
            ll_swap_per_cluster = np.asarray(
                bucketed_tree_log_lik_padded_binned(
                    affected_clusters, affected_bins, rho, tables_swap,
                    cache=cache),
                dtype=np.float64)
            ll_swap_sum = float(ll_swap_per_cluster.sum())

            log_ratio = ll_swap_sum - ll_curr_sum
            n_proposed += 1
            if log_ratio >= 0.0 or rng.random() < np.exp(log_ratio):
                aa[c1, theta_star] = k2
                aa[c2, theta_star] = k1
                n_accepted += 1
                ll_by_cluster[affected_arr] = ll_swap_per_cluster
                if verbose:
                    print(f"    swap (c1={c1}, c2={c2}, theta={theta_star}) "
                            f"log_ratio={log_ratio:+.2f} ACCEPT",
                            flush=True)

    info = {
        'n_proposed': int(n_proposed),
        'n_accepted': int(n_accepted),
        'accept_rate': (float(n_accepted) / n_proposed
                          if n_proposed > 0 else 0.0),
    }
    return aa, info


def gibbs_swap_arch_assignment_sweep_binned(
    clusters,
    pi_archetype: np.ndarray,
    arch_assignment: np.ndarray,
    rho: np.ndarray,
    S: np.ndarray,
    rho_chain: float,
    bin_idx_per_cluster,
    bin_centers: np.ndarray,
    rng: np.random.Generator,
    fix_theta0: bool = True,
    cache: 'BucketBatchCache | None' = None,
    verbose: bool = False,
) -> 'tuple[np.ndarray, dict]':
    """Gibbs-style swap sweep on arch_assignment.

    For each theta_star and each class A_c (in random order), evaluate
    all K_c candidate partners B_c in {0,...,K_c-1} — where B_c == A_c
    is the null (no swap) — via marginal LL, softmax, sample. This is
    Gibbs on the neighbourhood of A_c's transposition orbit at fixed
    theta_star. Permutation-preserving; ergodic on S_{K_c} within each
    theta because transpositions {(A_c B_c) : B_c} generate all of it
    over the sweep.

    Cost per (A_c, theta_star): up to K_c-1 sequential forward passes
    on the affected clusters — a candidate is skipped for free when
    (a) it's the null, (b) the swap is a no-op because arch entries
    match, or (c) no clusters contain either class.

    Uses ll_by_cluster caching: baseline LL computed once per sweep,
    updated in-place when a non-null candidate is accepted.
    """
    K_a, A = pi_archetype.shape
    K_c, L = arch_assignment.shape
    aa = np.asarray(arch_assignment, dtype=np.int32).copy()
    per_class = _clusters_containing_class(clusters)

    ll_by_cluster = _corpus_ll_by_cluster(
        clusters, pi_archetype[aa], rho, S, rho_chain,
        bin_idx_per_cluster, bin_centers, cache)

    n_class_updates = 0
    n_null = 0
    n_moved = 0
    theta_range = np.arange(1, L) if fix_theta0 else np.arange(L)

    for theta_star in rng.permutation(theta_range):
        theta_star = int(theta_star)
        for A_c in rng.permutation(K_c):
            A_c = int(A_c)
            log_L_baseline = float(ll_by_cluster.sum())
            log_L = np.full(K_c, log_L_baseline, dtype=np.float64)
            # Per-candidate cached per-cluster LL (for accept-path reuse).
            cand_ll: 'dict[int, tuple[np.ndarray, np.ndarray]]' = {}
            k_A = int(aa[A_c, theta_star])

            for B_c in range(K_c):
                if B_c == A_c:
                    continue
                k_B = int(aa[B_c, theta_star])
                if k_A == k_B:
                    continue                                  # no-op swap
                affected_ci = sorted(
                    set(per_class.get(A_c, []))
                    | set(per_class.get(B_c, [])))
                if not affected_ci:
                    continue
                affected_clusters = [clusters[i] for i in affected_ci]
                affected_bins = [bin_idx_per_cluster[i] for i in affected_ci]
                affected_arr = np.asarray(affected_ci)

                aa_swap = aa.copy()
                aa_swap[A_c, theta_star] = k_B
                aa_swap[B_c, theta_star] = k_A
                pi_field_swap = pi_archetype[aa_swap]
                tables_swap = precompute_kernel_tables(
                    bin_centers, rho, rho_chain, pi_field_swap, S)
                ll_swap_per_cluster = np.asarray(
                    bucketed_tree_log_lik_padded_binned(
                        affected_clusters, affected_bins, rho, tables_swap,
                        cache=cache),
                    dtype=np.float64)
                delta = float(
                    ll_swap_per_cluster.sum()
                    - ll_by_cluster[affected_arr].sum())
                log_L[B_c] = log_L_baseline + delta
                cand_ll[B_c] = (affected_arr, ll_swap_per_cluster)

            log_L -= log_L.max()
            p = np.exp(log_L)
            p /= p.sum()
            B_new = int(rng.choice(K_c, p=p))
            n_class_updates += 1
            if B_new == A_c or B_new not in cand_ll:
                n_null += 1
                if verbose:
                    print(f"    gswap (A={A_c}, theta={theta_star}) "
                            f"-> null  p_null={p[A_c]:.3f}", flush=True)
                continue
            # Apply the sampled swap.
            k_B = int(aa[B_new, theta_star])
            aa[A_c, theta_star] = k_B
            aa[B_new, theta_star] = k_A
            affected_arr, ll_new = cand_ll[B_new]
            ll_by_cluster[affected_arr] = ll_new
            n_moved += 1
            if verbose:
                print(f"    gswap (A={A_c}, theta={theta_star}) -> "
                        f"B={B_new}  p={p[B_new]:.3f}", flush=True)

    info = {
        'n_class_updates': int(n_class_updates),
        'n_null': int(n_null),
        'n_moved': int(n_moved),
        'move_rate': (float(n_moved) / n_class_updates
                        if n_class_updates > 0 else 0.0),
    }
    return aa, info


def gibbs_arch_assignment_sweep(
    clusters,
    pi_archetype: np.ndarray,
    arch_assignment: np.ndarray,
    rho_arch: np.ndarray,
    rho: np.ndarray,
    S: np.ndarray,
    rho_chain: float,
    rng: np.random.Generator,
    fix_theta0: bool = True,
    theta_order: 'np.ndarray | None' = None,
    class_order: 'np.ndarray | None' = None,
    verbose: bool = False,
    padded: bool = False,
    cache: 'BucketBatchCache | None' = None,
    binned: bool = False,
    bin_idx_per_cluster: 'list[list[np.ndarray]] | None' = None,
    bin_centers: 'np.ndarray | None' = None,
) -> 'tuple[np.ndarray, dict]':
    """One sweep over (c*, theta*) pairs (theta* != 0 if fix_theta0),
    sampling arch_assignment[c*, theta*] from its full conditional
    marginal-lik posterior.

    Args:
      clusters: list of (Tree, classes) pairs, classes.shape = (m,).
      pi_archetype: (K_a, A) frozen archetype simplex.
      arch_assignment: (K_c, L) current assignment; updated in-place
        (a copy is returned as well).
      rho_arch: (K_a,) prior weights over archetypes.
      rho, S, rho_chain: rest of the model.
      rng: np.random.Generator.
      fix_theta0: if True, skip theta=0 (keeps c=k identity).
      theta_order, class_order: optional permutation of the sweep;
        default random permutations.

    Returns:
      (new_arch_assignment, info) where info reports acceptance
      and total_forwards.
    """
    K_a, A = pi_archetype.shape
    K_c, L = arch_assignment.shape
    aa = np.asarray(arch_assignment, dtype=np.int32).copy()

    per_class = _clusters_containing_class(clusters)
    _forward = (bucketed_tree_log_lik_padded if padded
                    else bucketed_tree_log_lik)
    # M10b: under padded mode, use the K_a-vmapped variant that
    # evaluates all candidates in a single JIT call.
    # M12b: binned mode is currently sequential (K_a candidates each
    # get their own P_sub_tab); K-vmap+binned is future work.
    use_kvmap = padded and not binned
    if binned:
        assert padded, "binned mode requires padded=True"
        assert bin_idx_per_cluster is not None and bin_centers is not None, \
            "binned mode requires bin_idx_per_cluster + bin_centers"

    theta_range = np.arange(1, L) if fix_theta0 else np.arange(L)
    if theta_order is None:
        theta_order = rng.permutation(theta_range)
    if class_order is None:
        class_order = rng.permutation(K_c)

    n_forward_calls = 0
    n_kept_same = 0
    n_changed = 0

    for c_star in class_order:
        c_star = int(c_star)
        affected = per_class.get(c_star, [])
        if not affected:
            continue
        affected_clusters = [clusters[i] for i in affected]
        for theta_star in theta_order:
            theta_star = int(theta_star)
            # Compute per-candidate log-lik sum over affected clusters.
            if use_kvmap:
                # Build K_a variant pi_fields: only entry
                # (c_star, theta_star) changes per candidate.
                pi_field_base = pi_archetype[aa]                  # (K_c, L, A)
                pi_field_variants = np.broadcast_to(
                    pi_field_base[None, :, :, :],
                    (K_a, K_c, L, A)).copy()
                for k in range(K_a):
                    pi_field_variants[k, c_star, theta_star] = pi_archetype[k]
                ll_k = bucketed_tree_log_lik_padded_kvmap(
                    affected_clusters, rho, pi_field_variants, S, rho_chain,
                    cache=cache)
                log_L = ll_k.sum(axis=1)                          # (K_a,)
                n_forward_calls += 1                              # single JIT call
            elif binned:
                # M12b: sequential K_a candidates, each with a fresh
                # kernel_tables precompute for its candidate pi_field.
                # bin_idx_per_cluster is corpus-wide; slice per affected.
                affected_ci_set = set(per_class.get(c_star, []))
                affected_bins = [bin_idx_per_cluster[i]
                                    for i in affected_ci_set]
                # (index alignment: affected_clusters must be built in
                # the same order as we iterate affected_ci_set)
                affected_bins = [bin_idx_per_cluster[i]
                                    for i in per_class.get(c_star, [])]
                log_L = np.zeros(K_a, dtype=np.float64)
                for k in range(K_a):
                    aa_trial = aa.copy()
                    aa_trial[c_star, theta_star] = k
                    pi_field_trial = pi_archetype[aa_trial]
                    tables = precompute_kernel_tables(
                        bin_centers, rho, rho_chain, pi_field_trial, S)
                    ll = bucketed_tree_log_lik_padded_binned(
                        affected_clusters, affected_bins, rho, tables,
                        cache=cache)
                    log_L[k] = float(ll.sum())
                    n_forward_calls += 1
            else:
                log_L = np.zeros(K_a, dtype=np.float64)
                for k in range(K_a):
                    aa_trial = aa.copy()
                    aa_trial[c_star, theta_star] = k
                    pi_field_trial = pi_archetype[aa_trial]
                    ll = _forward(
                        affected_clusters, rho, pi_field_trial, S, rho_chain,
                        cache=cache) if padded else _forward(
                            affected_clusters, rho, pi_field_trial, S, rho_chain)
                    log_L[k] = float(ll.sum())
                    n_forward_calls += 1
            # Prior contribution.
            log_L += np.log(np.maximum(rho_arch, 1e-300))
            # Softmax.
            log_L -= log_L.max()
            p = np.exp(log_L)
            p /= p.sum()
            # Sample.
            k_new = int(rng.choice(K_a, p=p))
            if k_new == int(aa[c_star, theta_star]):
                n_kept_same += 1
            else:
                n_changed += 1
            if verbose:
                print(f"    (c={c_star}, θ={theta_star}) affected="
                        f"{len(affected)}  p={p.round(3).tolist()} → "
                        f"k={k_new}", flush=True)
            aa[c_star, theta_star] = k_new

    info = {
        'n_forward_calls': n_forward_calls,
        'n_kept_same': n_kept_same,
        'n_changed': n_changed,
        'total_updates': n_kept_same + n_changed,
    }
    return aa, info
