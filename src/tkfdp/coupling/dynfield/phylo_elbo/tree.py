"""Tree data structure for phylo-ELBO message passing.

A tree over N total nodes (leaves + internals + root) is represented as:

  n_nodes: int          total number of nodes
  n_leaves: int         number of leaves (indices 0..n_leaves-1)
  parent: (n_nodes,) int32     parent index per node; parent[root] = -1
  children: 'list[list[int]]'  child indices per node (leaves empty)
  branch_length: (n_nodes,) float64
                        length of the branch from node -> parent[node];
                        branch_length[root] = 0
  leaf_obs: (n_leaves, m) int32
                        per-leaf residue observations; -1 marks
                        unobserved / gap positions
  post_order: (n_nodes,) int32
                        post-order traversal (children before parents;
                        leaves first, root last)
  pre_order: (n_nodes,) int32
                        pre-order traversal (parents before children;
                        root first, leaves last)

Leaves are indexed 0..n_leaves-1 by convention; internal nodes are
indexed n_leaves..n_nodes-2; root is index n_nodes-1. Non-binary
internal nodes are permitted (variable-arity children list); the
sweeps handle them by sequential binary combines with intermediate
projection (see moment_match.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Tree:
    n_nodes: int
    n_leaves: int
    parent: np.ndarray                     # (n_nodes,) int32
    children: 'list[list[int]]'            # per-node child list
    branch_length: np.ndarray              # (n_nodes,) float64
    leaf_obs: np.ndarray                   # (n_leaves, m) int32; -1 = gap
    post_order: np.ndarray                 # (n_nodes,) int32
    pre_order: np.ndarray                  # (n_nodes,) int32

    @property
    def root(self) -> int:
        return int(self.n_nodes - 1)

    @property
    def m(self) -> int:
        return int(self.leaf_obs.shape[1])

    def is_leaf(self, v: int) -> bool:
        return v < self.n_leaves

    def is_internal(self, v: int) -> bool:
        return v >= self.n_leaves

    def check_consistency(self) -> None:
        """Assert tree structural invariants."""
        assert self.parent.shape == (self.n_nodes,)
        assert self.branch_length.shape == (self.n_nodes,)
        assert self.parent[self.root] == -1, "root must have parent=-1"
        # Every non-root has a valid parent pointing at some other node.
        for v in range(self.n_nodes):
            if v == self.root:
                continue
            p = int(self.parent[v])
            assert 0 <= p < self.n_nodes, f"parent of {v} out of range: {p}"
            assert v != p, f"self-loop at {v}"
            assert v in self.children[p], (
                f"child list of {p} missing {v}: {self.children[p]}")
        # Leaves have no children; internals may have any arity.
        for v in range(self.n_leaves):
            assert not self.children[v], f"leaf {v} has children"
        # Traversal orderings.
        assert self.post_order.shape == (self.n_nodes,)
        assert set(self.post_order.tolist()) == set(range(self.n_nodes))
        assert set(self.pre_order.tolist()) == set(range(self.n_nodes))
        # Post-order: children before parents.
        seen = set()
        for v in self.post_order:
            for c in self.children[int(v)]:
                assert c in seen, (
                    f"post_order visits {v} before its child {c}")
            seen.add(int(v))
        assert self.pre_order[0] == self.root
        assert self.post_order[-1] == self.root


def _traversal_orders(children: 'list[list[int]]', root: int, n_nodes: int
                        ) -> 'tuple[np.ndarray, np.ndarray]':
    """Compute post-order and pre-order via iterative DFS."""
    # Post-order.
    post = []
    stack = [(root, iter(children[root]))]
    seen_children = {root: 0}
    while stack:
        v, it = stack[-1]
        try:
            c = next(it)
            stack.append((c, iter(children[c])))
        except StopIteration:
            post.append(v)
            stack.pop()
    # Pre-order.
    pre = []
    stack = [root]
    while stack:
        v = stack.pop()
        pre.append(v)
        # Push children in reverse so they come out in-order.
        for c in reversed(children[v]):
            stack.append(c)
    assert len(post) == n_nodes and len(pre) == n_nodes
    return (np.asarray(post, dtype=np.int32),
              np.asarray(pre, dtype=np.int32))


def build_tree(parent: 'list[int] | np.ndarray',
                 branch_length: 'list[float] | np.ndarray',
                 leaf_obs: np.ndarray) -> Tree:
    """Build a Tree from parent-pointer + branch-length arrays plus
    leaf observations.

    Args:
      parent: (n_nodes,) int; parent index per node; -1 for root.
      branch_length: (n_nodes,) float; branch to parent[node];
        arbitrary for root.
      leaf_obs: (n_leaves, m) int32; leaf residue observations, -1
        marking gaps. n_leaves is inferred from the shape.

    Returns a Tree with children list, traversal orderings, and
    invariants checked.
    """
    parent = np.asarray(parent, dtype=np.int32)
    branch_length = np.asarray(branch_length, dtype=np.float64)
    leaf_obs = np.asarray(leaf_obs, dtype=np.int32)
    n_leaves = int(leaf_obs.shape[0])
    n_nodes = int(parent.shape[0])
    assert branch_length.shape == (n_nodes,)
    root = int(np.where(parent == -1)[0][0])
    assert root == n_nodes - 1, (
        f"convention violation: root must be last node (root={root}, "
        f"n_nodes-1={n_nodes-1})")

    # Build children list.
    children: 'list[list[int]]' = [[] for _ in range(n_nodes)]
    for v in range(n_nodes):
        p = int(parent[v])
        if p >= 0:
            children[p].append(v)
    post_order, pre_order = _traversal_orders(children, root, n_nodes)
    tree = Tree(n_nodes=n_nodes, n_leaves=n_leaves,
                  parent=parent, children=children,
                  branch_length=branch_length,
                  leaf_obs=leaf_obs,
                  post_order=post_order, pre_order=pre_order)
    tree.check_consistency()
    return tree


# ---------------------------------------------------------------------------
# Constructors for common topologies used in tests + development.
# ---------------------------------------------------------------------------


def make_cherry(tau: float, leaf_obs: np.ndarray) -> Tree:
    """Depth-1 cherry: two leaves sharing a single parent (root).

    parent = [1, 1, -1]         (2 leaves, 1 root)
    branch = [tau, tau, 0]      (equal branch from each leaf to root)
    leaf_obs: (2, m) int32
    """
    leaf_obs = np.asarray(leaf_obs, dtype=np.int32)
    assert leaf_obs.shape[0] == 2, "cherry must have exactly 2 leaves"
    # Nodes: 0 = leaf A, 1 = leaf B, 2 = root.
    parent = [2, 2, -1]
    branch = [tau, tau, 0.0]
    return build_tree(parent, branch, leaf_obs)


def make_star(taus: 'list[float]', leaf_obs: np.ndarray) -> Tree:
    """Star: n leaves all sharing a single parent (root). n = len(taus).
    """
    leaf_obs = np.asarray(leaf_obs, dtype=np.int32)
    n_leaves = int(leaf_obs.shape[0])
    assert len(taus) == n_leaves
    root = n_leaves
    parent = [root] * n_leaves + [-1]
    branch = list(taus) + [0.0]
    return build_tree(parent, branch, leaf_obs)


def make_balanced_binary(depth: int, tau: float,
                             leaf_obs: np.ndarray) -> Tree:
    """Balanced binary tree of given depth. Number of leaves 2**depth.
    All internal branch lengths equal to `tau`.

    Nodes are ordered leaves-first (0..2**depth-1), then internal
    nodes in post-order, then root last.
    """
    assert depth >= 1
    n_leaves = 2 ** depth
    leaf_obs = np.asarray(leaf_obs, dtype=np.int32)
    assert leaf_obs.shape[0] == n_leaves

    # Build children by processing levels bottom-up.
    n_internal = n_leaves - 1
    n_nodes = n_leaves + n_internal
    parent = [-1] * n_nodes
    branch = [0.0] * n_nodes

    # At level d = depth-1 (immediate parents of leaves), we have
    # 2**(depth-1) internal nodes. Assign next internal index starting
    # from n_leaves.
    next_internal = n_leaves
    prev_layer = list(range(n_leaves))
    # Set branches of leaves to their parent-to-be.
    for leaf in prev_layer:
        branch[leaf] = tau
    while len(prev_layer) > 1:
        new_layer = []
        for i in range(0, len(prev_layer), 2):
            u1 = prev_layer[i]
            u2 = prev_layer[i + 1]
            p = next_internal
            next_internal += 1
            parent[u1] = p
            parent[u2] = p
            new_layer.append(p)
        # Set branches of the new-layer internal nodes to their
        # upcoming parent.
        for v in new_layer:
            branch[v] = tau
        prev_layer = new_layer
    # The last remaining node is the root.
    root = prev_layer[0]
    assert root == n_nodes - 1
    parent[root] = -1
    branch[root] = 0.0
    return build_tree(parent, branch, leaf_obs)
