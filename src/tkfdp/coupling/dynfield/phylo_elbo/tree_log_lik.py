"""Tree log-likelihood via moment-matching CLVs.

Mirrors evolmoves ts/cluster/dynfield-emission.ts::dynfieldTreeBlockLogLikMM.

Post-order sweep: build CLVs bottom-up, propagate through edges, combine
via moment-matching projection at internal nodes, and integrate against
rho * prod pi_field at the root via the EXACT mm_mass_two (no projection
at the root, so cherries land exactly).

At depth 1 (cherry / star with tree.n_leaves == 2 children of root) the
result is exact. At the rho_chain -> 0 and rho_chain -> infty limits it
is also exact for any depth. At intermediate rho_chain on deeper trees
the projection is a controlled approximation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .mm_clv import (
    MMClv, leaf_clv, mm_edge, mm_combine, mm_rescale,
    mm_mass_two, mm_mass_one, per_class_field_Q)
from .tree import Tree


def tree_log_lik_mm(tree: Tree,
                       classes: np.ndarray,
                       rho: np.ndarray,
                       pi_field: np.ndarray,
                       S: np.ndarray,
                       rho_chain: float,
                       eta: float = 1.0) -> float:
    """Tree block log-likelihood under the moment-matching variational
    ELBO.

    Args:
      tree: Tree data structure. leaf_obs of shape (n_leaves, m) with
        -1 marking gaps.
      classes: (m,) int; per-column site-class assignment.
      rho: (L,) TSB weights.
      pi_field: (K_c, L, A) archetype-materialised stationary.
      S: (A, A) exchangeability.
      rho_chain: field-flip rate.
      eta: substitution-rate scaling (default 1.0). Used for +Gamma+I
        rate categories.

    Returns the block log-likelihood (a scalar).
    """
    Q = per_class_field_Q(pi_field, S)
    m = int(tree.m)
    L = int(rho.shape[0])

    def node_clv(v: int) -> MMClv:
        if tree.is_leaf(v):
            return leaf_clv(tree.leaf_obs[v], classes, pi_field)
        # Internal: propagate each child's CLV through the edge to v,
        # then moment-match combine.
        msgs: 'list[MMClv]' = []
        for c in tree.children[v]:
            child_clv = node_clv(c)
            tau_c = float(tree.branch_length[c])
            msg = mm_edge(child_clv, classes, tau_c, rho, pi_field, Q,
                            rho_chain, eta=eta)
            msg = mm_rescale(msg)
            msgs.append(msg)
        # Combine all messages via moment-matching projection.
        # At INTERNAL non-root nodes this is the projected form.
        acc = msgs[0]
        for msg in msgs[1:]:
            acc = mm_rescale(mm_combine(acc, msg, classes, pi_field))
        return acc

    # Root handling: combine all-but-last child, then exact-integrate
    # the final combine via mm_mass_two (no projection at root).
    root = tree.root
    if not tree.children[root]:
        # Degenerate: single-node "tree" whose root is a leaf.
        leaf = leaf_clv(tree.leaf_obs[root], classes, pi_field)
        mass, log_scale = mm_mass_one(leaf, classes, rho, pi_field)
        return float(np.log(max(mass, 1e-300))) + float(log_scale)

    root_msgs: 'list[MMClv]' = []
    for c in tree.children[root]:
        child_clv = node_clv(c)
        tau_c = float(tree.branch_length[c])
        msg = mm_edge(child_clv, classes, tau_c, rho, pi_field, Q,
                        rho_chain, eta=eta)
        msg = mm_rescale(msg)
        root_msgs.append(msg)

    if len(root_msgs) == 1:
        mass, log_scale = mm_mass_one(root_msgs[0], classes, rho, pi_field)
        return float(np.log(max(mass, 1e-300))) + float(log_scale)

    acc = root_msgs[0]
    for msg in root_msgs[1:-1]:
        acc = mm_rescale(mm_combine(acc, msg, classes, pi_field))
    mass, log_scale = mm_mass_two(acc, root_msgs[-1],
                                       classes, rho, pi_field)
    return float(np.log(max(mass, 1e-300))) + float(log_scale)
