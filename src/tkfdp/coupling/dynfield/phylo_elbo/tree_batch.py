"""Bucketed, batched tree log-likelihood evaluation (Milestone 4).

Given a list of (Tree, classes) clusters and model state (rho, pi_field,
S, rho_chain), bucket by (m_bucket, N_bucket, D_bucket), pad each
cluster to its bucket shape, and vmap `tree_log_lik_jax` across the
batch dim per bucket. Return a per-cluster log-lik array.

Bucketing:
  - m_bucket: sqrt2_bucket(cluster.m)  — cluster width padding.
  - N_bucket: sqrt2_bucket(cluster.tree.n_leaves)  — leaf-count.
  - D_bucket: sqrt2_bucket(cluster.tree.depth)  — tree depth.

Each unique bucket triple compiles ONE JIT function; the corpus reuses
compiled shapes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .tree import Tree
from .tree_padded import (
    PaddedTree, build_padded_tree, compute_node_levels, sqrt2_bucket)
from .tree_log_lik_jax import (
    gtr_eigendecomp_batch_jax, padded_tree_to_jax, tree_log_lik_jax)


def bucket_key(tree: Tree, classes: np.ndarray) -> 'tuple[int, int, int]':
    """(m_bucket, N_bucket, D_bucket) — the sqrt(2)-spaced bucket triple."""
    depth_actual = int(compute_node_levels(tree)[tree.root])
    return (sqrt2_bucket(int(classes.shape[0])),
              sqrt2_bucket(int(tree.n_leaves)),
              sqrt2_bucket(max(1, depth_actual)))


def bucket_key_from_padded(pt: PaddedTree,
                                classes: np.ndarray
                                ) -> 'tuple[int, int, int]':
    """(m_bucket, N_bucket, D_bucket) — for a pre-built PaddedTree.

    m is NOT sqrt2-bucketed because PaddedTrees are built with
    m = m_actual by default; stacking a bucket batch requires
    identical leaf_obs shapes, which requires identical m. Use pt.m
    (equal to m_actual for un-padded trees) so each bucket batch has
    same-m clusters.

    N and D ARE sqrt2-bucketed since PaddedTree.N_bucket and
    PaddedTree.D_bucket already equal sqrt2_bucket of actual.
    """
    # Key on the PADDED (N_bucket, D_bucket) -- with global tree padding these
    # are corpus constants, so all trees share one compiled shape per m.
    return (int(pt.m), int(pt.N_bucket), int(pt.D_bucket))


def _bucketed_n_slots_per_level(
        padded: 'list[tuple[PaddedTree, np.ndarray]]',
        D_bucket: int) -> 'list[int]':
    """Per-level slot count for a bucket batch, padded UP to the sqrt2
    grid so the JIT sees a BOUNDED set of argument shapes.

    Without this, the slot dimension is the raw max-over-batch slot count,
    which varies with the exact set of trees co-occurring in a batch — so
    every novel combination triggers a fresh, never-evicted XLA compile and
    the executable cache grows without bound (the host-RAM leak).

    Correctness: slots in [real_count, padded_count) are zero-filled
    downstream (child_pos=0, child_branch=0). Such a slot reads slot 0 of
    the level below and is never referenced by any upper level or by
    root_slot, so it is inert — it does not change any tree's likelihood.
    This is exactly the padding already applied when a bucket mixes trees
    of different slot counts; here we round the target up to the grid so
    the compiled shape is reused across batches.
    """
    return [sqrt2_bucket(max(pt.n_slots(l + 1) for pt, _ in padded))
            for l in range(D_bucket)]


def _bucket_shape_batch(padded: 'list[tuple[PaddedTree, np.ndarray]]',
                             bucket: 'tuple[int, int, int]') -> dict:
    """Shape-only bucket batch (leaf_obs, per-level arrays, etc.).

    Pi_field / classes / pi_arch INDEPENDENT: this can be cached across
    Gibbs sweeps because the tree topology + observations don't change.
    """
    m_bucket, N_bucket, D_bucket = bucket
    B = len(padded)
    n_slots_per_level = _bucketed_n_slots_per_level(padded, D_bucket)

    leaf_obs_batch = np.stack([pt.leaf_obs for pt, _ in padded])
    leaf_mask_batch = np.stack([pt.leaf_mask for pt, _ in padded])

    child_pos_by_level = []
    child_branch_by_level = []
    is_identity_by_level = []
    for l in range(D_bucket):
        n_slots_l = n_slots_per_level[l]
        cp_b = np.zeros((B, n_slots_l, 2), dtype=np.int32)
        cb_b = np.zeros((B, n_slots_l, 2), dtype=np.float64)
        id_b = np.zeros((B, n_slots_l), dtype=np.float64)
        for i, (pt, _) in enumerate(padded):
            n_actual = pt.n_slots(l + 1)
            cp_b[i, :n_actual, :] = pt.child_pos[l]
            cb_b[i, :n_actual, :] = pt.child_branch[l]
            id_mask = ((pt.child_pos[l][:, 0] == pt.child_pos[l][:, 1])
                          & (pt.child_branch[l][:, 0] == 0.0)
                          & (pt.child_branch[l][:, 1] == 0.0))
            id_b[i, :n_actual] = id_mask.astype(np.float64)
        child_pos_by_level.append(jnp.asarray(cp_b))
        child_branch_by_level.append(jnp.asarray(cb_b))
        is_identity_by_level.append(jnp.asarray(id_b))

    root_slots = np.array([pt.root_slot for pt, _ in padded], dtype=np.int32)

    return {
        'leaf_obs': jnp.asarray(leaf_obs_batch),
        'leaf_mask': jnp.asarray(leaf_mask_batch),
        'child_pos_by_level': tuple(child_pos_by_level),
        'child_branch_by_level': tuple(child_branch_by_level),
        'is_identity_by_level': tuple(is_identity_by_level),
        'root_slot': jnp.asarray(root_slots),
        'B': B,
    }


def _fill_pi_arch_classes(padded: 'list[tuple[PaddedTree, np.ndarray]]',
                              bucket: 'tuple[int, int, int]',
                              pi_field: np.ndarray) -> dict:
    """pi_arch and classes for one bucket batch (varies per Gibbs call)."""
    m_bucket, N_bucket, D_bucket = bucket
    K_c, L, A_alph = pi_field.shape
    B = len(padded)
    pi_arch_batch = np.full((B, m_bucket, L, A_alph),
                                    1.0 / A_alph, dtype=np.float64)
    classes_batch = np.zeros((B, m_bucket), dtype=np.int32)
    for i, (_, classes) in enumerate(padded):
        m_actual = int(classes.shape[0])
        for n in range(m_actual):
            c = int(classes[n])
            pi_arch_batch[i, n] = pi_field[c]
            classes_batch[i, n] = c
    return {
        'pi_arch': jnp.asarray(pi_arch_batch),
        'classes': jnp.asarray(classes_batch),
    }


def _stack_padded_batch(padded: 'list[tuple[PaddedTree, np.ndarray]]',
                             bucket: 'tuple[int, int, int]',
                             pi_field: np.ndarray) -> dict:
    """Backward-compatible: shape + pi_arch/classes in one call."""
    out = _bucket_shape_batch(padded, bucket)
    out.update(_fill_pi_arch_classes(padded, bucket, pi_field))
    return out


class BucketBatchCache:
    """Cache of shape-only bucket batches keyed by cluster-id tuple.

    Usage:
      cache = BucketBatchCache()
      shape = cache.get_or_build(padded_list, bucket)
      # ... use shape + fresh pi_arch/classes each call ...
    """
    def __init__(self, maxsize: int = 128):
        self._entries: 'dict' = {}
        self._maxsize = maxsize

    def get_or_build(self, padded, bucket) -> dict:
        key = (bucket, tuple(id(pt) for pt, _ in padded))
        if key not in self._entries:
            if len(self._entries) >= self._maxsize:
                # LRU-ish: drop the first-inserted.
                self._entries.pop(next(iter(self._entries)))
            self._entries[key] = _bucket_shape_batch(padded, bucket)
        return self._entries[key]

    def clear(self) -> None:
        self._entries.clear()


def build_batch(cluster_indices: 'list[int]',
                    clusters: 'list[tuple[Tree, np.ndarray]]',
                    bucket: 'tuple[int, int, int]',
                    pi_field: np.ndarray) -> dict:
    """Build batched JAX tensors for one bucket.

    Args:
      cluster_indices: indices into `clusters` for this bucket.
      clusters: full corpus list of (Tree, classes) pairs.
      bucket: (m_bucket, N_bucket, D_bucket).
      pi_field: (K_c, L, A) model tensor.

    Returns dict with:
      leaf_obs        (B, N_bucket, m_bucket) int32
      leaf_mask       (B, N_bucket) float64
      child_pos       tuple of D_bucket arrays, each (B, n_slots_l, 2) int32
      child_branch    tuple of D_bucket arrays, each (B, n_slots_l, 2) float64
      is_identity     tuple of D_bucket arrays, each (B, n_slots_l) float64
      root_slot       (B,) int32
      pi_arch         (B, m_bucket, L, A) float64
      classes         (B, m_bucket) int32
    """
    m_bucket, N_bucket, D_bucket = bucket
    padded = []
    for ci in cluster_indices:
        tree, classes = clusters[ci]
        pt = build_padded_tree(tree, N_bucket=N_bucket, D_bucket=D_bucket,
                                     m_bucket=m_bucket)
        padded.append((pt, classes))
    return _stack_padded_batch(padded, bucket, pi_field)


def _tree_log_lik_batched(
    leaf_obs, leaf_mask, child_pos_by_level, child_branch_by_level,
    is_identity_by_level, root_slot, pi_arch, classes,
    rho, rho_chain, xi, U, Uinv,
):
    """vmap tree_log_lik_jax over a leading batch axis. Structural in-axes
    are 0 for per-cluster tensors and None for shared model params."""
    def _one(leaf_obs_i, leaf_mask_i, cp_i, cb_i, id_i, root_i,
                 pi_arch_i, classes_i):
        return tree_log_lik_jax(
            leaf_obs_i, leaf_mask_i, cp_i, cb_i, id_i, int(root_i),
            pi_arch_i, rho, rho_chain, xi, U, Uinv, classes_i)
    # For root_slot per-cluster, we can't pass an int; need to gather at
    # runtime inside tree_log_lik_jax. Rewrite: use jnp.take.
    # Simpler: build a helper that does dynamic gather.

    def _one_dynamic_root(leaf_obs_i, leaf_mask_i, cp_i, cb_i, id_i,
                                root_slot_i, pi_arch_i, classes_i):
        # Duplicate the tree_log_lik_jax body but use dynamic root_slot.
        from .mm_clv_jax import (
            leaf_clv_jax, mm_combine_jax, mm_edge_jax, mm_mass_one_jax)
        from .tree_log_lik_jax import _propagate_level

        leaf_clvs = jax.vmap(leaf_clv_jax, in_axes=(0, None))(
            leaf_obs_i, pi_arch_i)
        clvs = leaf_clvs
        for level in range(len(cp_i)):
            clvs = _propagate_level(
                clvs, cp_i[level], cb_i[level], id_i[level],
                pi_arch_i, rho, rho_chain, xi, U, Uinv, classes_i,
            )
        # Dynamic gather of the root.
        root_clv = {
            'r': jnp.take(clvs['r'], root_slot_i, axis=0),
            's': jnp.take(clvs['s'], root_slot_i, axis=0),
            'A': jnp.take(clvs['A'], root_slot_i, axis=0),
        }
        mass = mm_mass_one_jax(root_clv, rho)
        return jnp.log(jnp.maximum(mass, 1e-300))

    vmapped = jax.vmap(_one_dynamic_root, in_axes=(
        0, 0, tuple([0] * len(child_pos_by_level)),
        tuple([0] * len(child_branch_by_level)),
        tuple([0] * len(is_identity_by_level)),
        0, 0, 0))
    return vmapped(
        leaf_obs, leaf_mask, child_pos_by_level, child_branch_by_level,
        is_identity_by_level, root_slot, pi_arch, classes)


def bucketed_tree_log_lik_padded_kvmap(
    clusters: 'list[tuple[PaddedTree, np.ndarray]]',
    rho: np.ndarray, pi_field_variants: np.ndarray,
    S: np.ndarray, rho_chain: float,
    cache: 'BucketBatchCache | None' = None,
) -> np.ndarray:
    """Batched-over-K per-cluster tree log-lik under K different
    pi_field variants (M10b).

    Args:
      clusters: list of (PaddedTree, classes) pairs.
      rho, S, rho_chain: model tensors (same across all K).
      pi_field_variants: (K, K_c, L, A) — K candidate pi_field tensors.

    Returns (K, n_clusters) log-likelihoods.

    The K candidate axis is vmapped inside a SINGLE JIT call per bucket,
    versus K separate calls in bucketed_tree_log_lik_padded. Since the
    tree structure and observations are identical across candidates,
    only pi_arch / substitution eigendecomps change per candidate. The
    total FLOPs scale linearly with K, but the JIT compile is amortised
    once and JAX may parallelise on GPU.
    """
    jax.config.update("jax_enable_x64", True)
    rho_j = jnp.asarray(rho, dtype=jnp.float64)
    pi_field_variants_j = jnp.asarray(pi_field_variants, dtype=jnp.float64)
    S_j = jnp.asarray(S, dtype=jnp.float64)
    K = int(pi_field_variants.shape[0])

    # Batched eigendecomps across K.
    def _eig_one(pi_field_k):
        return gtr_eigendecomp_batch_jax(pi_field_k, S_j)
    xi_k, U_k, Uinv_k = jax.vmap(_eig_one)(pi_field_variants_j)
    # Shapes: (K, K_c, L, A), (K, K_c, L, A, A), (K, K_c, L, A, A)

    log_liks = np.zeros((K, len(clusters)), dtype=np.float64)

    by_bucket: 'dict[tuple[int, int, int], list[int]]' = defaultdict(list)
    for i, (pt, classes) in enumerate(clusters):
        by_bucket[bucket_key_from_padded(pt, classes)].append(i)

    for bucket, idxs in by_bucket.items():
        padded = [clusters[i] for i in idxs]
        # Shape batch (cacheable) + pi_arch/classes (variant-specific).
        if cache is not None:
            shape_batch = cache.get_or_build(padded, bucket)
        else:
            shape_batch = _bucket_shape_batch(padded, bucket)
        vary_batch = _fill_pi_arch_classes(
            padded, bucket, pi_field_variants[0])
        batch = {**shape_batch, **vary_batch}
        # Now build pi_arch per candidate: (K, B, m_bucket, L, A).
        # For each candidate k, look up pi_field_variants[k, classes[b, n]].
        # classes shape: (B, m_bucket). Add K leading axis.
        classes_np = np.asarray(batch['classes'])       # (B, m_bucket)
        B, m_bucket = classes_np.shape
        K_c, L, A_alph = pi_field_variants.shape[1:]
        # pi_arch_kb[k, b, n, l, a] = pi_field_variants[k, classes[b, n], l, a]
        pi_arch_kb = pi_field_variants[:, classes_np, :, :]     # (K, B, m, L, A)
        pi_arch_kb_j = jnp.asarray(pi_arch_kb)

        # Vmap the batched forward over the K axis. In-axes: pi_arch,
        # xi, U, Uinv get K axis; everything else is shared.
        def _forward_one_k(pi_arch_k, xi_k_one, U_k_one, Uinv_k_one):
            return _tree_log_lik_batched(
                batch['leaf_obs'], batch['leaf_mask'],
                batch['child_pos_by_level'], batch['child_branch_by_level'],
                batch['is_identity_by_level'], batch['root_slot'],
                pi_arch_k, batch['classes'],
                rho_j, rho_chain, xi_k_one, U_k_one, Uinv_k_one)
        ll_kb = jax.vmap(_forward_one_k)(
            pi_arch_kb_j, xi_k, U_k, Uinv_k)      # (K, B)
        ll_kb_np = np.asarray(ll_kb, dtype=np.float64)
        for k in range(K):
            for pos, i in enumerate(idxs):
                log_liks[k, i] = ll_kb_np[k, pos]

    return log_liks


# -----------------------------------------------------------------
# Module-level cache of jit'd binned vmap forwards, keyed by n_levels.
# NO CLOSURES — every dep passed as an explicit arg so JAX caches
# by function id + shape signature and doesn't retrace per call.
# -----------------------------------------------------------------


def _forward_one_binned(
    leaf_obs_i, leaf_mask_i, cp_i, cb_i, id_i, root_i,
    pi_arch_i, classes_i, rho_j, beta_j, W_j, P_sub_j,
):
    """Module-level (no closures) single-tree binned forward."""
    from .tree_log_lik_jax_binned import tree_log_lik_jax_binned
    return tree_log_lik_jax_binned(
        leaf_obs_i, leaf_mask_i, cp_i, cb_i, id_i, root_i,
        pi_arch_i, rho_j, beta_j, W_j, P_sub_j, classes_i)


_BINNED_FORWARD_VMAP_CACHE: 'dict[int, callable]' = {}


def _get_binned_forward_vmap(n_levels: int):
    """Return a jax.jit'd vmap over the batch axis of
    _forward_one_binned, with `n_levels` per-level tuple arguments.
    Cached in-process by n_levels so subsequent calls with the same
    D_bucket reuse the compilation."""
    if n_levels not in _BINNED_FORWARD_VMAP_CACHE:
        vmapped = jax.vmap(_forward_one_binned, in_axes=(
            0, 0,
            tuple([0] * n_levels),   # child_pos_by_level
            tuple([0] * n_levels),   # child_bin_by_level
            tuple([0] * n_levels),   # is_identity_by_level
            0,                       # root_slot (per-cluster)
            0, 0,                    # pi_arch, classes (per-cluster)
            None, None, None, None,  # rho, beta_tab, W_tab, P_sub_tab (shared)
        ))
        _BINNED_FORWARD_VMAP_CACHE[n_levels] = jax.jit(vmapped)
    return _BINNED_FORWARD_VMAP_CACHE[n_levels]


def jit_cache_stats() -> 'tuple[int, int]':
    """Diagnostic: (n_jit_functions, total_compiled_variants) across the
    binned-forward JIT cache.

    Each entry in `_BINNED_FORWARD_VMAP_CACHE` is one `jax.jit` object
    (keyed by n_levels); each such object holds an INTERNAL, never-evicted
    compiled executable per distinct argument-shape signature it is called
    with. `total_variants` sums those internal counts — it is the quantity
    that, if it climbs without bound across a run, indicates the executable
    cache is the source of monotonic host/device memory growth. It should
    plateau once every (n_levels, slot-bucket, m_bucket, B_pad) shape combo
    has been seen once.
    """
    n_fns = len(_BINNED_FORWARD_VMAP_CACHE)
    total = 0
    for fn in _BINNED_FORWARD_VMAP_CACHE.values():
        try:
            total += int(fn._cache_size())
        except Exception:
            try:
                total += len(fn._cache)          # older jax fallback
            except Exception:
                pass
    return n_fns, total


def _stack_padded_batch_binned(padded: 'list[tuple[PaddedTree, np.ndarray]]',
                                        bucket: 'tuple[int, int, int]',
                                        bin_idx_per_tree: 'list[list[np.ndarray]]'
                                        ) -> 'tuple[jnp.ndarray, ...]':
    """Per-level child-branch bin index arrays for a bucket of padded
    trees. Same slot padding as _bucket_shape_batch."""
    m_bucket, N_bucket, D_bucket = bucket
    B = len(padded)
    # MUST match _bucket_shape_batch's slot padding exactly so child_pos
    # (shape batch) and child_bin (this batch) share the same slot axis.
    n_slots_per_level = _bucketed_n_slots_per_level(padded, D_bucket)
    child_bin_by_level = []
    for l in range(D_bucket):
        n_slots_l = n_slots_per_level[l]
        cbin_b = np.zeros((B, n_slots_l, 2), dtype=np.int32)
        for i, (pt, _) in enumerate(padded):
            n_actual = pt.n_slots(l + 1)
            cbin_b[i, :n_actual, :] = bin_idx_per_tree[i][l]
        child_bin_by_level.append(jnp.asarray(cbin_b))
    return tuple(child_bin_by_level)


def bucketed_tree_log_lik_padded_binned(
    clusters: 'list[tuple[PaddedTree, np.ndarray]]',
    bin_idx_per_cluster: 'list[list[np.ndarray]]',
    rho: np.ndarray,
    kernel_tables: dict,
    cache: 'BucketBatchCache | None' = None,
) -> np.ndarray:
    """Per-cluster tree log-lik via binned forward (M12).

    Args:
      clusters: list of (PaddedTree, classes).
      bin_idx_per_cluster: list of per-level bin-index arrays;
        bin_idx_per_cluster[i][l] is the (n_slots_l, 2) int32 array
        for cluster i at level l.
      rho: (L,) TSB weights.
      kernel_tables: dict from precompute_kernel_tables:
        {'beta': (n_bins, L), 'W': (n_bins, L, L),
         'P_sub': (n_bins, K_c, L, A, A)}.
      cache: optional BucketBatchCache for shape reuse.

    Returns (n_clusters,) log-likelihoods.
    """
    from .tree_log_lik_jax_binned import tree_log_lik_jax_binned
    jax.config.update("jax_enable_x64", True)
    rho_j = jnp.asarray(rho, dtype=jnp.float64)
    beta_j = jnp.asarray(kernel_tables['beta'], dtype=jnp.float64)
    W_j = jnp.asarray(kernel_tables['W'], dtype=jnp.float64)
    P_sub_j = jnp.asarray(kernel_tables['P_sub'], dtype=jnp.float64)

    # Derive pi_field from P_sub_tab's stationary would be roundabout;
    # kernel_tables should carry it too. Assume caller precomputes
    # tables from a specific pi_field and stores it here for pi_arch.
    # Simpler: pi_field must be passed alongside. Add it via
    # kernel_tables['pi_field'].
    pi_field = kernel_tables['pi_field']

    log_liks = np.zeros(len(clusters), dtype=np.float64)

    by_bucket: 'dict[tuple[int, int, int], list[int]]' = defaultdict(list)
    for i, (pt, classes) in enumerate(clusters):
        by_bucket[bucket_key_from_padded(pt, classes)].append(i)

    for bucket, idxs in by_bucket.items():
        padded = [clusters[i] for i in idxs]
        if cache is not None:
            shape_batch = cache.get_or_build(padded, bucket)
        else:
            shape_batch = _bucket_shape_batch(padded, bucket)
        vary_batch = _fill_pi_arch_classes(padded, bucket, pi_field)
        bin_lists = [bin_idx_per_cluster[i] for i in idxs]
        child_bin_by_level = _stack_padded_batch_binned(
            padded, bucket, bin_lists)

        n_levels = len(shape_batch['child_pos_by_level'])
        vmapped = _get_binned_forward_vmap(n_levels)
        ll_batch = vmapped(
            shape_batch['leaf_obs'], shape_batch['leaf_mask'],
            shape_batch['child_pos_by_level'], child_bin_by_level,
            shape_batch['is_identity_by_level'], shape_batch['root_slot'],
            vary_batch['pi_arch'], vary_batch['classes'],
            rho_j, beta_j, W_j, P_sub_j)
        ll_np = np.asarray(ll_batch, dtype=np.float64)
        for k, i in enumerate(idxs):
            log_liks[i] = ll_np[k]

    return log_liks


def bucketed_tree_log_lik_padded(
    clusters: 'list[tuple[PaddedTree, np.ndarray]]',
    rho: np.ndarray, pi_field: np.ndarray,
    S: np.ndarray, rho_chain: float,
    cache: 'BucketBatchCache | None' = None,
) -> np.ndarray:
    """Per-cluster tree log-lik from pre-built PaddedTrees.

    Same semantics as bucketed_tree_log_lik but takes PaddedTrees
    directly, skipping the per-call build_padded_tree work. Groups
    by (m_actual, n_leaves_actual, depth_actual)-derived bucket and
    stacks + vmaps per bucket.

    Args:
      clusters: list of (PaddedTree, classes) pairs.
      rho, pi_field, S, rho_chain: model tensors.

    Returns (n_clusters,) log-likelihoods.
    """
    jax.config.update("jax_enable_x64", True)
    rho_j = jnp.asarray(rho, dtype=jnp.float64)
    pi_field_j = jnp.asarray(pi_field, dtype=jnp.float64)
    S_j = jnp.asarray(S, dtype=jnp.float64)
    xi, U, Uinv = gtr_eigendecomp_batch_jax(pi_field_j, S_j)

    log_liks = np.zeros(len(clusters), dtype=np.float64)

    by_bucket: 'dict[tuple[int, int, int], list[int]]' = defaultdict(list)
    for i, (pt, classes) in enumerate(clusters):
        by_bucket[bucket_key_from_padded(pt, classes)].append(i)

    for bucket, idxs in by_bucket.items():
        padded = [clusters[i] for i in idxs]
        if cache is not None:
            shape_batch = cache.get_or_build(padded, bucket)
        else:
            shape_batch = _bucket_shape_batch(padded, bucket)
        vary_batch = _fill_pi_arch_classes(padded, bucket, pi_field)
        batch = {**shape_batch, **vary_batch}
        ll_batch = _tree_log_lik_batched(
            batch['leaf_obs'], batch['leaf_mask'],
            batch['child_pos_by_level'], batch['child_branch_by_level'],
            batch['is_identity_by_level'], batch['root_slot'],
            batch['pi_arch'], batch['classes'],
            rho_j, rho_chain, xi, U, Uinv)
        ll_np = np.asarray(ll_batch, dtype=np.float64)
        for k, i in enumerate(idxs):
            log_liks[i] = ll_np[k]

    return log_liks


def bucketed_tree_log_lik(clusters: 'list[tuple[Tree, np.ndarray]]',
                              rho: np.ndarray, pi_field: np.ndarray,
                              S: np.ndarray, rho_chain: float,
                              ) -> np.ndarray:
    """Per-cluster tree log-lik for a whole corpus via bucketed JAX.

    Args:
      clusters: list of (Tree, classes) pairs; classes.shape = (m,).
      rho: (L,) field TSB weights.
      pi_field: (K_c, L, A) archetype-materialised stationary.
      S: (A, A) exchangeability.
      rho_chain: scalar.

    Returns (n_clusters,) log-likelihoods.
    """
    jax.config.update("jax_enable_x64", True)
    rho_j = jnp.asarray(rho, dtype=jnp.float64)
    pi_field_j = jnp.asarray(pi_field, dtype=jnp.float64)
    S_j = jnp.asarray(S, dtype=jnp.float64)
    xi, U, Uinv = gtr_eigendecomp_batch_jax(pi_field_j, S_j)

    log_liks = np.zeros(len(clusters), dtype=np.float64)

    # Bucket clusters.
    by_bucket = defaultdict(list)
    for i, (tree, classes) in enumerate(clusters):
        by_bucket[bucket_key(tree, classes)].append(i)

    for bucket, idxs in by_bucket.items():
        batch = build_batch(idxs, clusters, bucket, pi_field)
        ll_batch = _tree_log_lik_batched(
            batch['leaf_obs'], batch['leaf_mask'],
            batch['child_pos_by_level'], batch['child_branch_by_level'],
            batch['is_identity_by_level'], batch['root_slot'],
            batch['pi_arch'], batch['classes'],
            rho_j, rho_chain, xi, U, Uinv)
        ll_np = np.asarray(ll_batch, dtype=np.float64)
        for k, i in enumerate(idxs):
            log_liks[i] = ll_np[k]

    return log_liks
