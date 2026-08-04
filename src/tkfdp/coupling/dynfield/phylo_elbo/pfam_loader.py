"""Adapter from Pfam FamilyCLV bundles (tkfdp.pfam_data) to the
phylo_elbo Tree + cluster format (M8).

Given a preprocessed FamilyCLV npz (from
`experiments/preprocess_pfam_pswm.py`), produce:
  - `phylo_elbo.tree.Tree` with the family's topology, branch lengths,
    and per-column leaf observations
  - list of (Tree, classes) cluster tuples for the Gibbs pipeline

Cluster construction: chunk the L alignment columns into equal-sized
sub-clusters of width `cluster_m`. Each sub-cluster's tree is the
FULL family tree (no per-cluster pruning at this stage). Sites within
a sub-cluster share a theta trajectory; sites in different sub-clusters
have independent theta trajectories.

Rationale for one-tree-per-cluster (no pruning): keeping the full
family tree makes per-cluster forward-pass cost O(N_leaves * m * A^2)
regardless of gap patterns. Pruning per cluster to only leaves with
non-gap observations at those columns would reduce work but adds
non-trivial preprocessing (topology re-simplification, degree-2
contractions); a follow-up if compute becomes the bottleneck.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tkfdp.bio import GAP_INDEX
from tkfdp.pfam_data import FamilyCLV, load_clv_family

from .tree import Tree, build_tree


def _binarize_parent_tau(parent: np.ndarray, tau: np.ndarray,
                             n_leaves: int
                             ) -> 'tuple[np.ndarray, np.ndarray, int]':
    """Convert a rooted tree (possibly with multifurcations) to strictly
    binary by inserting cascading phantom internal nodes with 0-length
    branches.

    Approach: build a fresh (parent, tau) node list from scratch. Emit
    leaves 0..n_leaves-1 first (preserving their original ids), then
    emit internals in post-order, cascading multifurcations into
    binaries via new phantom internal nodes. Root ends up last.
    """
    n_nodes_in = int(parent.shape[0])
    children_in: 'list[list[int]]' = [[] for _ in range(n_nodes_in)]
    root_in = -1
    for v in range(n_nodes_in):
        p = int(parent[v])
        if p >= 0:
            children_in[p].append(v)
        else:
            root_in = v
    assert root_in >= 0

    # If already binary, return original.
    if all(len(c) <= 2 for c in children_in):
        # Ensure root is last, otherwise re-emit.
        if root_in == n_nodes_in - 1:
            return (np.asarray(parent, dtype=np.int32).copy(),
                      np.asarray(tau, dtype=np.float64).copy(),
                      n_nodes_in)

    # Emit fresh node list. First slot 0..n_leaves-1 are the leaves;
    # then internals, then root last.
    # Use a post-order traversal, and for each visited internal, emit
    # cascading phantoms as needed.
    #
    # out_parent[new_id] = new_id of parent (or -1 root).
    # out_tau[new_id] = branch length to that parent.
    #
    # Track old_id -> new_id for the "top" of each subtree (i.e. the
    # node whose id in the new emission represents that subtree). For
    # multifurcating internal, the "top" is the last-emitted node in
    # the cascade for it.

    out_parent: 'list[int]' = [-1] * n_leaves          # leaves placeholder
    out_tau: 'list[float]' = [0.0] * n_leaves
    # Preserve leaf tau (branch to their old parent, which we'll rewire
    # after post-order).
    for l in range(n_leaves):
        out_tau[l] = float(tau[l])

    subtree_top_new_id: 'dict[int, int]' = {l: l for l in range(n_leaves)}

    # Post-order over old ids.
    post: 'list[int]' = []
    stack = [(root_in, iter(children_in[root_in]))]
    while stack:
        v, it = stack[-1]
        try:
            c = next(it)
            stack.append((c, iter(children_in[c])))
        except StopIteration:
            post.append(v)
            stack.pop()

    # For each old internal in post-order, emit cascade.
    # Cascade for children [c1, c2, ..., cn]:
    #   phantom_1 = new internal, children (c1, c2), tau 0 for phantom_1.
    #   phantom_2 = new internal, children (phantom_1, c3), tau 0.
    #   ...
    #   phantom_(n-1) = TOP for v (has children (phantom_(n-2), cn), tau =
    #                   original tau[v] to v's parent).
    # For n == 2: single phantom (which is just "v" itself).
    for v in post:
        if v < n_leaves:
            continue
        cs = children_in[v]
        # Look up new ids for children.
        child_new_ids = [subtree_top_new_id[c] for c in cs]
        child_tau = [float(tau[c]) for c in cs]
        # Rewire the child taus in out_tau (in case they haven't been
        # set yet after some cascades).
        for cid_new, ct in zip(child_new_ids, child_tau):
            out_tau[cid_new] = ct
        # Cascade.
        if len(cs) == 2:
            # Emit one new node as v's new id.
            new_id = len(out_parent)
            out_parent.append(-1)
            out_tau.append(float(tau[v]))
            # Wire children to this new_id.
            out_parent[child_new_ids[0]] = new_id
            out_parent[child_new_ids[1]] = new_id
            subtree_top_new_id[v] = new_id
        else:
            # Cascade of (n - 1) phantoms.
            prev_top = child_new_ids[0]
            for i in range(1, len(cs)):
                new_id = len(out_parent)
                out_parent.append(-1)
                # Tau for the top phantom (the last one) is v's original
                # tau; intermediates are 0.
                is_top = (i == len(cs) - 1)
                out_tau.append(float(tau[v]) if is_top else 0.0)
                # Wire prev_top and child i to this new node.
                out_parent[prev_top] = new_id
                out_parent[child_new_ids[i]] = new_id
                prev_top = new_id
            subtree_top_new_id[v] = prev_top

    # Root: the last-emitted node for the old root.
    new_root = subtree_top_new_id[root_in]
    # Root's parent must be -1; verify.
    assert out_parent[new_root] == -1
    # Ensure root is LAST (swap with the current last if not).
    n_out = len(out_parent)
    if new_root != n_out - 1:
        # Swap positions new_root and n_out - 1.
        last = n_out - 1
        # Any node whose parent == new_root -> should point to last.
        # Any node whose parent == last -> should point to new_root.
        # Then swap the pair themselves.
        for v in range(n_out):
            p = out_parent[v]
            if p == new_root:
                out_parent[v] = last
            elif p == last:
                out_parent[v] = new_root
        out_parent[new_root], out_parent[last] = out_parent[last], out_parent[new_root]
        out_tau[new_root], out_tau[last] = out_tau[last], out_tau[new_root]
        new_root = last
    return (np.asarray(out_parent, dtype=np.int32),
              np.asarray(out_tau, dtype=np.float64),
              n_out)


def _prune_tree_to_leaves(parent: np.ndarray, tau: np.ndarray,
                              n_leaves_orig: int,
                              keep_leaves: np.ndarray,
                              ) -> 'tuple[np.ndarray, np.ndarray, int, np.ndarray]':
    """Prune a rooted binary tree to a subset of leaves.

    Applies degree-2 contractions: any internal node with only 1 kept
    descendant edge becomes a passthrough (its child adopts the parent's
    parent, with branch lengths summed).

    Args:
      parent: (n_nodes,) int; -1 at root.
      tau: (n_nodes,) float.
      n_leaves_orig: original leaf count (leaf ids 0..n_leaves_orig-1).
      keep_leaves: bool mask (n_leaves_orig,) or int array of kept leaf
        ids.

    Returns:
      (new_parent, new_tau, new_n_nodes, kept_leaf_orig_ids)
      kept_leaf_orig_ids: (new_n_leaves,) int, the original leaf id
        for each new leaf slot (0..new_n_leaves-1). Used by callers to
        remap observations.
    """
    n_nodes = int(parent.shape[0])
    if keep_leaves.dtype == bool:
        keep_mask_leaf = keep_leaves.astype(bool)
    else:
        keep_mask_leaf = np.zeros(n_leaves_orig, dtype=bool)
        keep_mask_leaf[keep_leaves] = True

    # Compute keep[v] = True if v is a kept leaf or has a kept descendant.
    keep = np.zeros(n_nodes, dtype=bool)
    keep[:n_leaves_orig] = keep_mask_leaf
    # Post-order.
    children: 'list[list[int]]' = [[] for _ in range(n_nodes)]
    root = -1
    for v in range(n_nodes):
        p = int(parent[v])
        if p >= 0:
            children[p].append(v)
        else:
            root = v
    post: 'list[int]' = []
    stack = [(root, iter(children[root]))]
    while stack:
        v, it = stack[-1]
        try:
            c = next(it)
            stack.append((c, iter(children[c])))
        except StopIteration:
            post.append(v)
            stack.pop()
    for v in post:
        if v >= n_leaves_orig:
            keep[v] = any(keep[c] for c in children[v])

    # Nearest-kept-ancestor pointer + branch-length sum along the path.
    # Walk parent chain from each kept node up until we hit another kept
    # node (or root).
    new_parent_of: 'dict[int, int]' = {}
    new_tau_of: 'dict[int, float]' = {}
    for v in range(n_nodes):
        if v == root or not keep[v]:
            continue
        p = int(parent[v])
        acc_tau = float(tau[v])
        while p >= 0 and not keep[p]:
            acc_tau += float(tau[p])
            p = int(parent[p])
        new_parent_of[v] = p
        new_tau_of[v] = acc_tau

    # Contract degree-2 internals: nodes with exactly 1 kept child.
    # Count kept children per kept node.
    kept_child_count: 'dict[int, int]' = {v: 0 for v in range(n_nodes)
                                             if keep[v]}
    for v, p in new_parent_of.items():
        if p >= 0 and p in kept_child_count:
            kept_child_count[p] += 1

    to_contract = {v for v, cnt in kept_child_count.items()
                       if cnt == 1 and v >= n_leaves_orig}
    # For each contract-node, redirect its single-kept-child's parent
    # around it and add its branch length. Iterate until no more.
    while to_contract:
        # Find any contract-node whose new_parent isn't itself
        # to-contract (process leafward-most first).
        progress = False
        for v in list(to_contract):
            # v's single-kept-child(ren) — find them.
            v_kids = [w for w, p in new_parent_of.items() if p == v]
            if len(v_kids) != 1:
                to_contract.discard(v)
                continue
            w = v_kids[0]
            # If w itself is to-contract, we can still process v first
            # (v becomes irrelevant after w gets a new parent above v).
            new_parent_of[w] = new_parent_of.get(v, -1)
            new_tau_of[w] = new_tau_of[w] + new_tau_of.get(v, 0.0)
            # Remove v from the graph.
            new_parent_of.pop(v, None)
            new_tau_of.pop(v, None)
            to_contract.discard(v)
            progress = True
            break
        if not progress:
            break

    # After contraction, identify remaining kept nodes: leaves (with
    # keep_mask_leaf True) + internals with >=2 kept children.
    kept_child_count = {v: 0 for v in range(n_nodes) if keep[v]
                            and v in new_parent_of or v == root}
    # Rebuild kept_child_count from remaining new_parent_of.
    kept_child_count = {}
    all_nodes_in_pruned = set()
    for v in list(new_parent_of.keys()):
        all_nodes_in_pruned.add(v)
        p = new_parent_of[v]
        if p >= 0:
            all_nodes_in_pruned.add(p)
            kept_child_count[p] = kept_child_count.get(p, 0) + 1

    # Assemble new (parent, tau) with convention: leaves 0..new_n_leaves-1,
    # then internals in some order, then root last.
    kept_leaves_list = sorted(
        v for v in all_nodes_in_pruned if v < n_leaves_orig)
    kept_internals_list = sorted(
        v for v in all_nodes_in_pruned if v >= n_leaves_orig
        and v != root)
    # The root of the pruned tree: the node with no parent in new_parent_of.
    # It's the nearest-kept-ancestor of everyone else. Could be the original
    # root, OR a contracted-away root's successor.
    new_root_orig_id = None
    for v in all_nodes_in_pruned:
        if v not in new_parent_of or new_parent_of[v] == -1:
            new_root_orig_id = v
            break
    assert new_root_orig_id is not None, "no root in pruned tree"
    # Convention: root last.
    ordered = kept_leaves_list + [
        v for v in kept_internals_list if v != new_root_orig_id
    ] + [new_root_orig_id]
    remap = {old: new for new, old in enumerate(ordered)}
    n_new = len(ordered)
    out_parent = np.full(n_new, -1, dtype=np.int32)
    out_tau = np.zeros(n_new, dtype=np.float64)
    for old_id in ordered:
        new_id = remap[old_id]
        if old_id == new_root_orig_id:
            out_parent[new_id] = -1
            out_tau[new_id] = 0.0
        else:
            old_p = new_parent_of[old_id]
            out_parent[new_id] = remap[old_p] if old_p != -1 else -1
            out_tau[new_id] = new_tau_of[old_id]
    kept_leaf_orig_ids = np.array(kept_leaves_list, dtype=np.int32)
    return out_parent, out_tau, n_new, kept_leaf_orig_ids


def family_to_tree(family_data: FamilyCLV,
                       column_indices: np.ndarray) -> Tree:
    """Build a phylo_elbo Tree for a subset of alignment columns.

    Args:
      family_data: FamilyCLV bundle.
      column_indices: (m,) int indices into the L-column alignment.

    Returns a Tree with:
      - topology from family_data.parent / family_data.tau
      - leaf_obs (N_leaves, m) int32; -1 marks gaps (family gap-index 20
        is remapped to -1 here to match the phylo_elbo convention).
    """
    N_leaves = int(family_data.n_leaves)
    m = int(column_indices.shape[0])
    # Slice leaf_msa to the selected columns; remap gap (20) -> -1.
    leaf_msa_slice = family_data.leaf_msa[:, column_indices]  # (N, m) int8
    leaf_obs = np.where(leaf_msa_slice == GAP_INDEX, -1,
                            leaf_msa_slice).astype(np.int32)
    # Binarize the topology (FastTree may produce multifurcating root
    # or unresolved nodes). Preserves leaf ids 0..N_leaves-1.
    parent_b, tau_b, n_nodes_b = _binarize_parent_tau(
        family_data.parent, family_data.tau, N_leaves)
    return build_tree(
        parent=parent_b,
        branch_length=tau_b,
        leaf_obs=leaf_obs,
    )


def family_to_tree_pruned(family_data: FamilyCLV,
                                column_indices: np.ndarray) -> Tree:
    """Build a Tree pruned to only leaves with any non-gap observation
    at any of the given columns.

    Massively smaller than family_to_tree when the cluster's columns
    are sparse (typical Pfam pattern). Degree-2 internal contractions
    keep the tree strict-binary.
    """
    N_leaves_orig = int(family_data.n_leaves)
    m = int(column_indices.shape[0])
    leaf_msa_slice = family_data.leaf_msa[:, column_indices]  # (N, m)
    # Keep leaves with at least one non-gap observation.
    any_obs = (leaf_msa_slice != GAP_INDEX).any(axis=1)  # (N,)
    if not any_obs.any():
        # No observations at any leaf; fall back to full tree (rare).
        return family_to_tree(family_data, column_indices)
    parent_b, tau_b, _ = _binarize_parent_tau(
        family_data.parent, family_data.tau, N_leaves_orig)
    parent_p, tau_p, n_nodes_p, kept_ids = _prune_tree_to_leaves(
        parent_b, tau_b, N_leaves_orig, any_obs)
    # Build leaf_obs for the pruned leaves.
    # Remap: new_leaf_id -> original_leaf_id via kept_ids.
    n_kept = int(kept_ids.shape[0])
    leaf_obs = np.where(
        leaf_msa_slice[kept_ids] == GAP_INDEX, -1,
        leaf_msa_slice[kept_ids]).astype(np.int32)
    return build_tree(parent=parent_p, branch_length=tau_p,
                          leaf_obs=leaf_obs)


def family_to_clusters(family_data: FamilyCLV,
                          cluster_m: int,
                          K_c: int,
                          rng: np.random.Generator,
                          prune: bool = False,
                          ) -> 'list[tuple[Tree, np.ndarray]]':
    """Chunk the family's L columns into clusters of width cluster_m.

    The last chunk may be shorter than cluster_m. Per-site class
    assignments (c_n) initialised uniformly at random over [0, K_c).

    Args:
      family_data: FamilyCLV.
      cluster_m: target cluster width in columns.
      K_c: number of classes (for random init).
      rng: np.random.Generator for c_n init.

    Returns list of (Tree, classes) tuples.
    """
    L = int(family_data.L)
    clusters = []
    _build = family_to_tree_pruned if prune else family_to_tree
    for start in range(0, L, cluster_m):
        stop = min(start + cluster_m, L)
        cols = np.arange(start, stop, dtype=np.int32)
        m = int(cols.shape[0])
        tree = _build(family_data, cols)
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
        clusters.append((tree, classes))
    return clusters


def load_pfam_clusters(clv_paths: 'list[str]',
                          cluster_m: int, K_c: int,
                          rng: np.random.Generator,
                          prune: bool = False,
                          ) -> 'list[tuple[Tree, np.ndarray]]':
    """Load a list of FamilyCLV npz files and produce a flat cluster list.

    Args:
      clv_paths: list of npz paths (from data/pfam_processed_clv_*/).
      cluster_m: cluster width.
      K_c: number of classes.
      rng: rng for c_n init.

    Returns concatenated (Tree, classes) list.
    """
    all_clusters = []
    for path in clv_paths:
        fd = load_clv_family(path)
        all_clusters.extend(family_to_clusters(
            fd, cluster_m, K_c, rng, prune=prune))
    return all_clusters
