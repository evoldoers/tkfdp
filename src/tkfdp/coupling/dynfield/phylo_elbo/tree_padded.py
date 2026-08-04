"""Padded tree layout for JIT'd level-by-level tree forward passes.

Milestone 2 of the phylo-ELBO Gibbs plan
(`docs/phylo_elbo_gibbs_jax_plan.md`).

Represents a tree as a fixed-shape (per bucket) structure suitable for
JIT compilation via level-by-level explicit sibling-pair gathers
(option 2 from the design discussion). Handles arbitrary binary
topologies including deeply unbalanced trees by inserting 0-length
"phantom" identity edges to align every internal node's children at
exactly one level below.

A `PaddedTree` at bucket (N_bucket, D_bucket) has:
  - leaf_obs:    (N_bucket, m) int          padded leaf observations
                                              (-1 for gap; padded leaves
                                              all -1)
  - leaf_mask:   (N_bucket,) float          1 real, 0 phantom
  - For each level l in [1, D_bucket] (leaves at 0, root at D_bucket):
      child_pos[l]:      (max_slots_l, 2) int   position of each child at
                                                  level l-1 that this slot
                                                  combines
      child_branch[l]:   (max_slots_l, 2) float branch lengths child -> slot
      slot_mask[l]:      (max_slots_l,) float   1 real internal node,
                                                  0 phantom
  - root_slot: (,) int      the position of the root at level D_bucket
                              (typically 0; 0 by construction of build)

Level definition: every leaf is at level 0. An internal node v with
children c1, c2 is at level max(level(c1), level(c2)) + 1, with
phantom identity nodes inserted for any child whose level is more
than 1 below v. This ensures every internal node's children are
exactly one level below.

Padding conventions:
  - Padded leaves have leaf_mask=0. Their leaf_obs is all -1 (gap).
    Downstream leaf_clv makes their contribution neutral (constant 1).
  - Padded internal slots have slot_mask=0. Downstream propagation
    should short-circuit these (or set their branch_lengths=0 so the
    mm_edge is identity).
  - The bucket max_slots_l is chosen so all trees in the bucket fit;
    per-level padding is separate (different levels may have different
    max_slots).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .tree import Tree


# ---------------------------------------------------------------------------
# Data structure.
# ---------------------------------------------------------------------------


@dataclass
class PaddedTree:
    """Fixed-shape padded representation of one tree."""
    N_bucket: int                                 # padded leaf count
    D_bucket: int                                 # padded depth
    n_leaves_actual: int
    depth_actual: int
    m: int                                        # cluster width
    leaf_obs: np.ndarray                          # (N_bucket, m) int32; -1 = gap
    leaf_mask: np.ndarray                         # (N_bucket,) float64
    # Per-level (1..D_bucket) arrays, indexed as list[l - 1]:
    child_pos: 'list[np.ndarray]'                # each (max_slots_l, 2) int32
    child_branch: 'list[np.ndarray]'             # each (max_slots_l, 2) float64
    slot_mask: 'list[np.ndarray]'                # each (max_slots_l,) float64
    root_slot: int                                # position of root at level D

    def n_slots(self, level: int) -> int:
        """Max slots at level l. l = 0 -> N_bucket; l >= 1 -> len(slot_mask[l-1])."""
        if level == 0:
            return self.N_bucket
        return int(self.slot_mask[level - 1].shape[0])

    def check(self) -> None:
        assert self.leaf_obs.shape == (self.N_bucket, self.m)
        assert self.leaf_mask.shape == (self.N_bucket,)
        assert self.n_leaves_actual <= self.N_bucket
        assert self.depth_actual <= self.D_bucket
        assert len(self.child_pos) == self.D_bucket
        assert len(self.child_branch) == self.D_bucket
        assert len(self.slot_mask) == self.D_bucket
        for l in range(1, self.D_bucket + 1):
            n = self.n_slots(l)
            n_prev = self.n_slots(l - 1)
            assert self.child_pos[l - 1].shape == (n, 2)
            assert self.child_branch[l - 1].shape == (n, 2)
            assert self.slot_mask[l - 1].shape == (n,)
            # Child positions are in valid range.
            active = self.slot_mask[l - 1] > 0.5
            if np.any(active):
                cp = self.child_pos[l - 1][active]
                assert (cp >= 0).all() and (cp < n_prev).all(), \
                    f"child_pos out of range at level {l}: min={cp.min()}, " \
                    f"max={cp.max()}, n_prev={n_prev}"
        assert 0 <= self.root_slot < self.n_slots(self.D_bucket)

    def save(self, path) -> None:
        """Save this PaddedTree to disk as a single .npz.

        Per-level arrays are packed as `child_pos_l{ell}`,
        `child_branch_l{ell}`, `slot_mask_l{ell}` for ell = 0..D-1.
        """
        d: 'dict[str, np.ndarray]' = {
            'N_bucket': np.int32(self.N_bucket),
            'D_bucket': np.int32(self.D_bucket),
            'n_leaves_actual': np.int32(self.n_leaves_actual),
            'depth_actual': np.int32(self.depth_actual),
            'm': np.int32(self.m),
            'leaf_obs': self.leaf_obs.astype(np.int32),
            'leaf_mask': self.leaf_mask.astype(np.float64),
            'root_slot': np.int32(self.root_slot),
        }
        for ell in range(self.D_bucket):
            d[f'child_pos_l{ell}'] = self.child_pos[ell].astype(np.int32)
            d[f'child_branch_l{ell}'] = self.child_branch[ell].astype(np.float64)
            d[f'slot_mask_l{ell}'] = self.slot_mask[ell].astype(np.float64)
        np.savez_compressed(path, **d)

    @staticmethod
    def load(path) -> 'PaddedTree':
        d = np.load(path, allow_pickle=False)
        D = int(d['D_bucket'])
        child_pos = [np.asarray(d[f'child_pos_l{ell}'], dtype=np.int32)
                        for ell in range(D)]
        child_branch = [np.asarray(d[f'child_branch_l{ell}'],
                                          dtype=np.float64)
                            for ell in range(D)]
        slot_mask = [np.asarray(d[f'slot_mask_l{ell}'], dtype=np.float64)
                        for ell in range(D)]
        return PaddedTree(
            N_bucket=int(d['N_bucket']),
            D_bucket=D,
            n_leaves_actual=int(d['n_leaves_actual']),
            depth_actual=int(d['depth_actual']),
            m=int(d['m']),
            leaf_obs=np.asarray(d['leaf_obs'], dtype=np.int32),
            leaf_mask=np.asarray(d['leaf_mask'], dtype=np.float64),
            child_pos=child_pos,
            child_branch=child_branch,
            slot_mask=slot_mask,
            root_slot=int(d['root_slot']),
        )


# ---------------------------------------------------------------------------
# Bucketing helpers.
# ---------------------------------------------------------------------------


def _sqrt2_sequence(max_x: int) -> 'list[int]':
    """Generate the increasing sqrt(2)-spaced integer sequence up to max_x.

    Sequence: 1, 2, 3, 4, 6, 8, 12, 16, 23, 32, ... Uses ceil with a
    1e-9 fudge to avoid float-precision loss of exact powers of 2
    (sqrt(2)^2 = 2.0000000000000004 would otherwise ceil to 3).
    """
    buckets: 'list[int]' = []
    x = 1.0
    while True:
        b = int(math.ceil(x - 1e-9))
        if b < 1:
            b = 1
        if not buckets or b > buckets[-1]:
            buckets.append(b)
        if b >= max_x:
            break
        x *= math.sqrt(2)
    return buckets


def sqrt2_bucket(x: int) -> int:
    """Smallest bucket in the sqrt(2)-geomspace sequence >= x."""
    if x <= 1:
        return 1
    for b in _sqrt2_sequence(x):
        if b >= x:
            return b
    return x


def compute_node_levels(tree: Tree) -> np.ndarray:
    """For each node v, level(v) = 0 if leaf, else 1 + max(level(c) for c in
    children(v)). Root's level = tree depth (max distance to a leaf).
    """
    n = tree.n_nodes
    levels = np.full(n, -1, dtype=np.int32)
    # Post-order guarantees children processed before parents.
    for v in tree.post_order:
        v = int(v)
        if tree.is_leaf(v):
            levels[v] = 0
        else:
            child_levels = [int(levels[int(c)]) for c in tree.children[v]]
            levels[v] = 1 + max(child_levels)
    return levels


# ---------------------------------------------------------------------------
# Builder.
# ---------------------------------------------------------------------------


def build_padded_tree(tree: Tree,
                        N_bucket: 'int | None' = None,
                        D_bucket: 'int | None' = None,
                        m_bucket: 'int | None' = None,
                        ) -> PaddedTree:
    """Build a PaddedTree from a Tree.

    Args:
      tree: source Tree (with tree.check_consistency having passed).
      N_bucket: leaf-count padding target. Default = sqrt2_bucket(n_leaves).
      D_bucket: depth padding target. Default = sqrt2_bucket(depth_actual).
      m_bucket: cluster-width padding target. Default = tree.m (no
        m-padding). When set, leaf_obs is padded to (N_bucket, m_bucket)
        with -1 (gap marker) for padded columns. Downstream pi_arch
        and classes tensors must also be padded to m_bucket.

    Returns a PaddedTree with the level-by-level arrays and phantom
    identity nodes for any subtree that "waits" between levels.
    """
    # 1. Node levels + bucket sizes.
    levels = compute_node_levels(tree)
    depth_actual = int(levels[tree.root])
    if N_bucket is None:
        N_bucket = sqrt2_bucket(tree.n_leaves)
    if D_bucket is None:
        D_bucket = sqrt2_bucket(max(1, depth_actual))
    assert N_bucket >= tree.n_leaves, \
        f"N_bucket={N_bucket} < n_leaves={tree.n_leaves}"
    assert D_bucket >= depth_actual, \
        f"D_bucket={D_bucket} < depth={depth_actual}"
    m_actual = tree.m
    if m_bucket is None:
        m_bucket = m_actual
    assert m_bucket >= m_actual, \
        f"m_bucket={m_bucket} < m_actual={m_actual}"

    m = m_bucket

    # 2. Build a REPRESENTATION at each level as a list of (node_kind,
    # source, branch_length_to_next_level). node_kind is either
    # 'real' (node_id) or 'phantom_chain' (base_leaf_id) indicating
    # an identity chain lifted from a leaf.
    #
    # Semantics:
    #   level_slots[0]:  the leaves (padded to N_bucket). Real leaves fill
    #                     first n_leaves slots; padded slots are phantom.
    #   level_slots[l] for l >= 1: internal nodes at level l, PLUS phantom
    #                     lifts for subtree roots whose parent is at a
    #                     level > l.
    #
    # For each internal node v at level lv >= 1, position it at level lv.
    # For each of its children c, if level(c) == lv - 1, use c directly at
    # level lv - 1. If level(c) < lv - 1, insert an identity chain from
    # level(c) up to level lv - 1 (this is the "wait" logic).

    # Level slots. Use tuples (kind, data) where kind is 'real' or 'ident'.
    # For real: data = source_node_id at prev level (same or lifted).
    # For phantom identity chain: data = source of chain (the base node id).
    #
    # Simpler: track slot_id -> (source_slot_id_at_prev_level,
    # branch_length_from_prev_to_this). Real internal node's children come
    # from their children's slot_ids at prev level.

    # Assignment strategy: BFS by level starting from level 0.
    # At level 0: leaves 0..n_leaves-1 in order.
    # At level l:
    #   For each internal node v with level(v) == l:
    #     For each child c: if level(c) == l - 1, use c's slot at level l-1.
    #                        if level(c) < l - 1, need a phantom identity chain.
    #   For each subtree root ("thing to lift") whose parent is at level > l:
    #     Add a phantom identity slot at level l pointing to its slot at
    #     level l - 1 with 0-length branch.

    # Phase A: compute level assignments and slot positions.
    #
    # For each real node v, define slot_at[v, lift_level] for lift_level in
    # [level(v), parent_level]. slot_at[v, level(v)] = v's own slot on its
    # native level. slot_at[v, level(v) + k] = k-th phantom lift (identity
    # from v's native slot).

    parent_level = np.zeros(tree.n_nodes, dtype=np.int32)
    for v in range(tree.n_nodes):
        if v == tree.root:
            # Give root a virtual parent one level above D_bucket so the
            # lift condition (nl < level < pl) covers levels through
            # D_bucket for the root. That way root has a slot at every
            # level from its native level up through D_bucket.
            parent_level[v] = D_bucket + 1
        else:
            parent_level[v] = int(levels[int(tree.parent[v])])

    # For each level, list of (real_or_phantom, source_id).
    # Convention: source_id is the node_id whose data lives at this slot.
    # For a phantom lift, source_id = the leaf/internal-node being lifted.
    slots_by_level: 'list[list[tuple[int, bool]]]' = [
        [] for _ in range(D_bucket + 1)
    ]
    # slot_at_level[node_id][level] = slot index at that level (or -1)
    slot_at: 'dict[tuple[int, int], int]' = {}

    # Level 0: leaves in id order, plus phantom padding at the end.
    for leaf_id in range(tree.n_leaves):
        slot_at[(leaf_id, 0)] = leaf_id
        slots_by_level[0].append((leaf_id, True))
    # Phantom leaves fill up to N_bucket. Their "source" is a fake id -1
    # (we'll skip them in the leaf_obs and mask them out).
    for _ in range(tree.n_leaves, N_bucket):
        slots_by_level[0].append((-1, False))

    # Higher levels: process nodes in level order (BFS).
    # For each node v: assign its own slot at level(v), plus any lifts
    # up to parent_level(v).
    for level in range(1, D_bucket + 1):
        # Find nodes whose level == this.
        for v in range(tree.n_nodes):
            if int(levels[v]) == level:
                idx = len(slots_by_level[level])
                slot_at[(v, level)] = idx
                slots_by_level[level].append((v, True))
        # Lift any node whose parent_level > level and native level < level.
        # These are "waiters" that need identity chain from prev level UP to
        # (but not including) the parent's level. Condition: nl < level < pl
        # (strict both sides) - at level == pl the parent consumes the child
        # directly from level-1, no need to make an intermediate slot.
        for v in range(tree.n_nodes):
            nl = int(levels[v])
            pl = int(parent_level[v])
            if nl < level < pl:
                # v has a slot at level - 1 (either its own or a phantom lift).
                # Lift it to level via 0-length identity.
                if (v, level - 1) not in slot_at:
                    continue  # shouldn't happen with correct algorithm
                idx = len(slots_by_level[level])
                slot_at[(v, level)] = idx
                slots_by_level[level].append((v, False))  # phantom slot

    # Phase B: construct per-level arrays.
    max_slots_by_level = [len(slots_by_level[l])
                             for l in range(D_bucket + 1)]

    # leaf_obs and leaf_mask. m padded to m_bucket with -1 (gap) in
    # extra columns; the extra columns' A stays uniform through the
    # pipeline (neutral under mm_edge and mm_combine since
    # <pi_arch, 1> = 1 per site).
    leaf_obs = np.full((N_bucket, m_bucket), -1, dtype=np.int32)
    leaf_mask = np.zeros(N_bucket, dtype=np.float64)
    for slot_idx in range(len(slots_by_level[0])):
        src, is_real = slots_by_level[0][slot_idx]
        if is_real:
            leaf_obs[slot_idx, :m_actual] = tree.leaf_obs[src]
            leaf_mask[slot_idx] = 1.0

    # child_pos, child_branch, slot_mask per level l >= 1.
    child_pos: 'list[np.ndarray]' = []
    child_branch: 'list[np.ndarray]' = []
    slot_mask: 'list[np.ndarray]' = []
    for level in range(1, D_bucket + 1):
        n_slots = max_slots_by_level[level]
        cp = np.zeros((n_slots, 2), dtype=np.int32)
        cb = np.zeros((n_slots, 2), dtype=np.float64)
        sm = np.zeros(n_slots, dtype=np.float64)
        for slot_idx in range(n_slots):
            src, is_real = slots_by_level[level][slot_idx]
            if src == -1:
                # Phantom padding slot. Point at self via prev level (no valid
                # source). Use position 0 as a placeholder; slot_mask=0 mutes.
                cp[slot_idx, :] = 0
                cb[slot_idx, :] = 0.0
                sm[slot_idx] = 0.0
            elif not is_real:
                # Phantom identity lift of node src from level - 1.
                # Both "children" are the same: src's slot at level - 1.
                # With 0-length branches, both mm_edge are identity, then
                # mm_combine of a CLV with itself.
                # Instead we use a single-child path: make left = src's
                # slot at prev level, right = dummy pointing at 0 with
                # branch_length 0 and combine multiplies by identity.
                # Simpler: use left = right = same slot, branches = 0.
                src_slot_prev = slot_at[(src, level - 1)]
                cp[slot_idx, 0] = src_slot_prev
                cp[slot_idx, 1] = src_slot_prev
                cb[slot_idx, :] = 0.0
                sm[slot_idx] = 1.0  # Active, but marked as phantom via
                                    # zero-branch (see notes below).
            else:
                # Real internal node at this level. Its 2 children are at
                # level - 1 (either their native slot or a phantom lift).
                real_v = src
                assert not tree.is_leaf(real_v)
                child_ids = tree.children[real_v]
                # For non-binary internal nodes, we currently only support
                # binary; multifurcations should be pre-decomposed.
                assert len(child_ids) == 2, \
                    f"non-binary internal node {real_v} (children " \
                    f"{child_ids}); pre-decompose into cascading binaries"
                for k, cid in enumerate(child_ids):
                    cp[slot_idx, k] = slot_at[(cid, level - 1)]
                    cb[slot_idx, k] = float(tree.branch_length[cid])
                sm[slot_idx] = 1.0
        child_pos.append(cp)
        child_branch.append(cb)
        slot_mask.append(sm)

    root_slot = slot_at[(tree.root, D_bucket)]

    pt = PaddedTree(
        N_bucket=N_bucket, D_bucket=D_bucket,
        n_leaves_actual=tree.n_leaves, depth_actual=depth_actual,
        m=m,
        leaf_obs=leaf_obs, leaf_mask=leaf_mask,
        child_pos=child_pos, child_branch=child_branch,
        slot_mask=slot_mask, root_slot=int(root_slot))
    pt.check()
    return pt
