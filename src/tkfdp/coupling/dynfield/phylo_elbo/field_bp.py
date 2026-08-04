"""Exact sum-product belief propagation on an L-state field tree MRF.

Building block for the per-site (Delta-explicit) tree-structured ELBO: the
field-and-jumps factor of the residue-correlated q is a tree Markov random field
over theta_v in {0..L-1} with a root node potential and per-edge potentials that
carry a jump index Delta in {0,1}. This module does exact BP on it, returning
log Z, node marginals q_v(theta), and edge marginals q_uv(theta_u,theta_v,Delta).
It is m-independent (the residue expectations are folded into the potentials by
the caller). Validated against brute-force enumeration.
"""
from __future__ import annotations

import numpy as np


def _logsumexp(a, axis=None):
    a = np.asarray(a, np.float64)
    mx = np.max(a, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    out = mx + np.log(np.sum(np.exp(a - mx), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else float(np.squeeze(out))


def field_logbp(tree, log_node_pot, log_edge_pot):
    """Exact BP on the field tree MRF (log domain).

    Args:
      tree: numpy Tree (root, children, parent, post_order, pre_order, n_nodes).
      log_node_pot: dict v -> (L,) log node potential (0 where none).
      log_edge_pot: dict v -> (L, L, n_delta) log edge potential for the edge
        parent(v)->v, indexed [theta_parent, theta_child, Delta].

    Returns (logZ, q_node, q_edge) where q_node[v] is (L,) and q_edge[v] is
    (L, L, n_delta) for the edge parent(v)->v (normalized joint marginal).
    """
    nn = tree.n_nodes; root = tree.root
    any_v = next(iter(log_edge_pot)) if log_edge_pot else None
    L = int(log_node_pot.get(root, log_edge_pot[any_v][:, 0, 0]).shape[0]) \
        if any_v is not None else int(log_node_pot[root].shape[0])

    def npot(v):
        return log_node_pot.get(v, np.zeros(L))
    # edge potential summed over Delta -> (L,L) for the field marginals
    logE = {v: _logsumexp(log_edge_pot[v], axis=2) for v in log_edge_pot}

    # ---- upward: up[v](theta_v) = sum of child messages (no node potential) ---
    up = [np.zeros(L) for _ in range(nn)]
    msg_from = {}
    for v in tree.post_order:
        v = int(v)
        acc = np.zeros(L)
        for c in tree.children[v]:
            c = int(c)
            # message to v(theta_v) = logsumexp_{theta_c}( logE[c][theta_v,theta_c] + npot(c) + up[c] )
            m = _logsumexp(logE[c] + (npot(c) + up[c])[None, :], axis=1)
            msg_from[c] = m
            acc = acc + m
        up[v] = acc
    logZ = _logsumexp(up[root] + npot(root))

    # ---- downward: down[v](theta_v) = message into v from the rest of tree ----
    down = [np.zeros(L) for _ in range(nn)]
    for v in tree.pre_order:
        v = int(v)
        for c in tree.children[v]:
            c = int(c)
            cavity = up[v] - msg_from[c] + down[v] + npot(v)   # exclude c
            down[c] = _logsumexp(logE[c] + cavity[:, None], axis=0)  # (L,) over theta_c

    # ---- node marginals -------------------------------------------------------
    q_node = []
    for v in range(nn):
        lg = up[v] + down[v] + npot(v)
        q_node.append(np.exp(lg - _logsumexp(lg)))

    # ---- edge marginals q_uv(theta_u,theta_v,Delta) ---------------------------
    q_edge = [None] * nn
    for v in range(nn):
        if v == root:
            continue
        u = int(tree.parent[v])
        cavity_u = up[u] - msg_from[v] + down[u] + npot(u)      # (L,) over theta_u
        below_v = npot(v) + up[v]                                # (L,) over theta_v
        # joint over (theta_u, theta_v, Delta) up to normalization
        lg = (cavity_u[:, None, None] + log_edge_pot[v]
              + below_v[None, :, None])                          # (L,L,nd)
        q_edge[v] = np.exp(lg - _logsumexp(lg))
    return logZ, q_node, q_edge
