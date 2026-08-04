"""Per-site LG08 Felsenstein pruning + posterior marginals at every tree node.

Given a Pfam family's MSA and guide tree, compute the Position-Specific
Weight Matrix (PSWM) at every node: M_v[site, aa] = P(residue at node v,
site s = aa | leaf observations at site s under IID LG08 substitution).

Two-pass sum-product on the tree per site:
  Bottom-up: compute conditional likelihood vector (CLV) at each node
    from the leaves toward the root.
  Top-down: compute the outside partial at each node from the root
    down; combine with the CLV to get the marginal.

Result: for a tree with 2N-1 nodes and MSA of L sites, an (N_nodes, L, A)
PSWM tensor where A = 20 (aa alphabet).

Complexity: per site, O(N_nodes * A^2) via matrix exp per branch (cached).
Whole family: O(L * N_nodes * A^2). For a Pfam family with N=500 leaves,
L=200 sites: ~500 * 200 * 400 = 40M ops. Trivial.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import expm

from .bio import Node
from .lg08 import Q_LG08, PI_LG08


A_ALPH = 20                              # amino acid alphabet size
GAP_IDX = 20                             # gap marker in .msa


def _assign_node_ids(root: Node) -> 'tuple[list[Node], np.ndarray]':
    """Depth-first traversal; returns (nodes_in_postorder, parent_id_array).

    Leaves come first (indices 0..n_leaves-1), then internals in post-order,
    root last. parent[v] = -1 for root.
    """
    # First pass: collect leaves and internals separately in post-order.
    postorder: 'list[Node]' = []
    stack: 'list[tuple[Node, bool]]' = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            postorder.append(node)
        else:
            stack.append((node, True))
            for c in reversed(node.children or []):
                stack.append((c, False))
    # Reorder so leaves come first, then internals in post-order.
    leaves = [n for n in postorder if not n.children]
    internals = [n for n in postorder if n.children]
    nodes_ordered = leaves + internals
    node_to_id = {id(n): i for i, n in enumerate(nodes_ordered)}
    parent = np.full(len(nodes_ordered), -1, dtype=np.int32)
    parent_from_child: 'dict[int, Node]' = {}

    # Walk once more to fill parent map.
    def walk(v: Node):
        for c in (v.children or []):
            parent_from_child[id(c)] = v
            walk(c)
    walk(root)
    for i, v in enumerate(nodes_ordered):
        p = parent_from_child.get(id(v))
        if p is not None:
            parent[i] = node_to_id[id(p)]
    return nodes_ordered, parent


def _children_ids(nodes: 'list[Node]', parent: np.ndarray
                     ) -> 'list[list[int]]':
    """For each node id, list its children's ids."""
    n = len(nodes)
    children: 'list[list[int]]' = [[] for _ in range(n)]
    for i in range(n):
        p = int(parent[i])
        if p >= 0:
            children[p].append(i)
    return children


def _pack_branch_matrices(taus: np.ndarray, Q: np.ndarray,
                             cache_grid: float = 1e-6
                             ) -> 'dict[float, np.ndarray]':
    """Return {rounded_tau: expm(Q * tau)} for the unique branch lengths.
    Rounds tau to `cache_grid` to dedupe near-identical branches."""
    unique = np.unique(np.round(taus / cache_grid) * cache_grid)
    return {float(u): expm(Q * float(u)) for u in unique}


def compute_pswm_family(msa: np.ndarray,
                            leaf_names: 'list[str]',
                            tree: Node,
                            pi: np.ndarray = PI_LG08,
                            Q: np.ndarray = Q_LG08,
                            ) -> dict:
    """Run per-site LG08 Felsenstein pruning + posterior marginals.

    Args:
      msa: (N_msa, L) int8 array; leaf residue observations. Gap = GAP_IDX = 20.
      leaf_names: list of leaf-name strings for msa rows.
      tree: root Node of the guide tree (leaf.name matches leaf_names).
      pi: (A,) LG08 stationary; default from lg08.PI_LG08.
      Q: (A, A) LG08 rate matrix.

    Returns dict:
      n_nodes: int
      n_leaves: int
      parent: (n_nodes,) int32; -1 for root
      tau: (n_nodes,) float64; branch length to parent (0 for root)
      pswm: (n_nodes, L, A) float64; per-site per-node marginal
      leaf_msa_row: (n_leaves,) int32; row index into `msa` for each leaf
      root_id: int
    """
    nodes, parent = _assign_node_ids(tree)
    n_nodes = len(nodes)
    n_leaves = sum(1 for n in nodes if not n.children)
    L = int(msa.shape[1])

    # Branch length from each node to its parent (tau[root] = 0).
    tau = np.zeros(n_nodes, dtype=np.float64)
    for i, v in enumerate(nodes):
        if v.branch_length is not None:
            tau[i] = float(v.branch_length)

    # Leaf name -> MSA row lookup.
    name_to_row = {name: i for i, name in enumerate(leaf_names)}
    leaf_msa_row = np.full(n_leaves, -1, dtype=np.int32)
    for i in range(n_leaves):
        v = nodes[i]
        if v.name in name_to_row:
            leaf_msa_row[i] = int(name_to_row[v.name])

    root_id = int(np.where(parent == -1)[0][0])

    # Cache expm per unique branch length.
    tau_cache = _pack_branch_matrices(tau, Q)

    def P(tau_v: float) -> np.ndarray:
        return tau_cache[float(np.round(tau_v / 1e-6) * 1e-6)]

    children = _children_ids(nodes, parent)
    post_order = np.arange(n_nodes)                # already in post-order
    pre_order = post_order[::-1]

    pswm = np.zeros((n_nodes, L, A_ALPH), dtype=np.float64)
    # CLV storage: bottom-up conditional-likelihood vector L_v(x, s) at
    # every node. Rescaled per-node to max=1 so it fits fp32; the
    # accumulated log-scale (base e) is tracked separately so the true
    # unnormalised CLV is `clv_out[v, s] * exp(log_scale_out[v, s])`.
    # Only internal nodes accumulate scale (leaves have delta CLV of
    # max=1 already). The LG08 tree marginal per site is
    #   log P^{LG}(y_s | tree) = log( pi @ clv_out[root, s] ) + log_scale_out[root, s]
    # used as the importance-weight denominator (par:arch-lg08-is).
    clv_out = np.zeros((n_nodes, L, A_ALPH), dtype=np.float64)
    log_scale_out = np.zeros((n_nodes, L), dtype=np.float64)

    # Per-site: two-pass sum-product with per-node rescaling.
    # Without rescaling, clv at internal nodes underflows to zero on
    # deep trees (~700 leaves): each child contributes a probability
    # factor and their product collapses below fp64 precision. We
    # rescale clv[i] by its max after each internal update; the scale
    # cancels in the final marginal (pswm[i, site] proportional to
    # outside[i] * clv[i], normalised).
    for site in range(L):
        # Bottom-up CLV.
        clv = np.zeros((n_nodes, A_ALPH), dtype=np.float64)
        log_scale = np.zeros(n_nodes, dtype=np.float64)
        for i in range(n_leaves):
            row = int(leaf_msa_row[i])
            if row < 0:
                clv[i] = 1.0                        # missing leaf -> uninformative
                continue
            x = int(msa[row, site])
            if x >= 0 and x < A_ALPH:
                clv[i, x] = 1.0
            else:
                clv[i] = 1.0                        # gap: uninformative
        for i in post_order[n_leaves:]:              # internals
            v_clv = np.ones(A_ALPH, dtype=np.float64)
            for c in children[i]:
                P_c = P(tau[c])
                v_clv *= P_c @ clv[c]
                log_scale[i] += log_scale[c]         # inherit child scales
            m = v_clv.max()
            if m > 0:
                v_clv = v_clv / m
                log_scale[i] += np.log(m)
            clv[i] = v_clv
        clv_out[:, site, :] = clv
        log_scale_out[:, site] = log_scale

        # Top-down outside partial including root prior.
        # outside[v](x_v) proportional to sum_{outside subtree at v} pi(x_root)
        #                     * prod_{branches} P(x_child | x_parent)
        # marginalised over states outside v's subtree, keeping the branch
        # transitions into v.
        #
        # Init: outside[root] = pi (the "everything outside root" is just
        # the root prior since root has no ancestor).
        outside = np.zeros((n_nodes, A_ALPH), dtype=np.float64)
        outside[root_id] = pi.copy()
        for i in pre_order:
            if i == root_id: continue
            p = int(parent[i])
            P_v = P(tau[i])                            # P_v[a, b] = P(state_b at v | state_a at p)
            # g_p(x_p) = outside[p](x_p) * prod_{siblings s of v under p}
            #                    (P_s @ clv[s])(x_p)
            # where (P_s @ clv[s])(x_p) = sum_{x_s} P(x_s | x_p) * clv[s](x_s)
            #                            = P(obs at s's subtree | x_p)
            g_p = outside[p].copy()
            for s in children[p]:
                if s == i: continue
                P_s = P(tau[s])
                g_p *= P_s @ clv[s]
            # outside[v](x_v) = sum_{x_p} g_p(x_p) * P_v(x_p, x_v)
            out_i = g_p @ P_v
            # Rescale (same underflow issue as CLV for deep trees).
            m = out_i.max()
            if m > 0:
                out_i = out_i / m
            outside[i] = out_i

        # Marginal at v: normalise outside * clv (outside already includes pi).
        for i in range(n_nodes):
            marg = outside[i] * clv[i]
            s = float(marg.sum())
            if s > 0:
                pswm[i, site] = marg / s
            else:
                pswm[i, site] = pi                    # fallback

    # LG08 tree marginal per site:
    #   log P^{LG}(y_s | tree) = log( pi @ clv_out[root, s] ) + log_scale[root, s]
    root_clv = clv_out[root_id]                             # (L, A)
    pi_dot_root = np.maximum(root_clv @ pi, 1e-300)         # (L,) via (L,A)@(A,)
    log_p_lg_per_site = np.log(pi_dot_root) + log_scale_out[root_id]

    return dict(
        n_nodes=n_nodes,
        n_leaves=n_leaves,
        parent=parent,
        tau=tau,
        pswm=pswm,
        clv=clv_out,                                        # (n_nodes, L, A)
        log_scale=log_scale_out,                            # (n_nodes, L)
        log_p_lg_per_site=log_p_lg_per_site,                # (L,)
        leaf_msa_row=leaf_msa_row,
        root_id=root_id,
    )


def enumerate_branches(parent: np.ndarray) -> np.ndarray:
    """Return (n_branches, 2) int32 array of (parent_id, child_id) tuples.
    A branch exists for every non-root node."""
    branches = []
    for i in range(len(parent)):
        p = int(parent[i])
        if p >= 0:
            branches.append((p, i))
    return np.asarray(branches, dtype=np.int32)


def _preorder_from_parent(parent: np.ndarray) -> np.ndarray:
    """Return an ordering of nodes such that every parent comes before its
    children. Uses a simple DFS from the root."""
    n = len(parent)
    root = int(np.where(parent == -1)[0][0])
    # Build children lists.
    children_of = [[] for _ in range(n)]
    for i in range(n):
        p = int(parent[i])
        if p >= 0:
            children_of[p].append(i)
    order = []
    stack = [root]
    while stack:
        v = stack.pop()
        order.append(v)
        for c in children_of[v]:
            stack.append(c)
    return np.asarray(order, dtype=np.int32)


def build_class_conditional_transition_cache(
        pi_field: np.ndarray,
        rho: np.ndarray,
        S: np.ndarray,
        parent: np.ndarray,
        tau: np.ndarray,
        cache_grid: float = 1e-6,
        bin_means: 'np.ndarray | None' = None,
        ) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Build the transition cache for the class-conditional
    field-marginal proposal used in par:arch-lg08-is:

        P^{(c, m)}_{marg}(t)
          = sum_theta rho[theta] * expm( m · Q^{(c, theta)} · t )
        Q^{(c, theta)}[i, j] = S[i, j] * pi_field[c, theta, j]  (i != j)
        Q^{(c, theta)}[i, i] = -sum_{j!=i} Q^{(c, theta)}[i, j]

    where m is a per-column rate multiplier drawn from a Yang Γ+I
    discretisation (m = 0 for the invariant bin, m > 0 otherwise).
    When `bin_means` is None, the cache is built without m — a single
    m=1 slot — matching the pre-Γ+I default.

    Args:
      pi_field:  (K_c, L_field, A) — per-(class, field) stationary.
      rho:       (L_field,) — TSB field prior.
      S:         (A, A) — LG08 exchangeabilities.
      parent:    (n_nodes,) int32 — used only to enumerate unique tau.
      tau:       (n_nodes,) float64.
      bin_means: (K_bins,) float — rate multipliers. `bin_means[0]`
                   must be 0.0 for the invariant bin; the remaining
                   entries are the Yang Γ-quantile means. If None or
                   K_bins == 1 with mean 1, no bin axis is added.

    Returns (P_cache, tau_idx, log_P_cache):
      P_cache:     (K_c, n_tau, A, A) if bin_means is None,
                     (K_c, K_bins, n_tau, A, A) otherwise.
      tau_idx:     (n_nodes,) int64 — bin index into the tau axis.
      log_P_cache: same shape as P_cache — element-wise log.
    """
    K_c, L_field, A = pi_field.shape
    unique_tau_map: 'dict[float, int]' = {}
    unique_tau_list: 'list[float]' = []
    for v in range(len(parent)):
        if int(parent[v]) < 0:
            continue
        key = float(np.round(float(tau[v]) / cache_grid) * cache_grid)
        if key not in unique_tau_map:
            unique_tau_map[key] = len(unique_tau_list)
            unique_tau_list.append(key)
    n_tau = len(unique_tau_list)
    idx_diag = np.arange(A)
    tau_idx = np.zeros(len(parent), dtype=np.int64)
    for v in range(len(parent)):
        if int(parent[v]) < 0:
            continue
        key = float(np.round(float(tau[v]) / cache_grid) * cache_grid)
        tau_idx[v] = unique_tau_map[key]

    if bin_means is None:
        P_cache = np.zeros((K_c, n_tau, A, A), dtype=np.float64)
        for c in range(K_c):
            for th in range(L_field):
                pi_ct = pi_field[c, th]
                Q = S * pi_ct[None, :]
                Q[idx_diag, idx_diag] = 0.0
                Q[idx_diag, idx_diag] = -Q.sum(axis=1)
                weight = float(rho[th])
                for t_i, t in enumerate(unique_tau_list):
                    P_cache[c, t_i] += weight * expm(Q * float(t))
        # Root-fix: scipy expm on a stiff generator (which arises when
        # pi_field has entries at pi_clamp = 1e-30 after archetype
        # collapse) produces occasional small negative eigen-basis
        # residuals. Left in place, they propagate through the peeling
        # via v_clv *= contrib into -inf/NaN in a downstream division.
        # Clip non-negative and re-normalise rows to unity so P_cache
        # is a valid stochastic matrix regardless of the generator's
        # conditioning.
        P_cache = np.maximum(P_cache, 0.0)
        row_sums = P_cache.sum(axis=-1, keepdims=True)
        P_cache = P_cache / np.maximum(row_sums, 1e-300)
        log_P_cache = np.log(np.maximum(P_cache, 1e-300))
        return P_cache, tau_idx, log_P_cache

    bin_means = np.asarray(bin_means, dtype=np.float64)
    K_bins = int(bin_means.shape[0])
    P_cache = np.zeros((K_c, K_bins, n_tau, A, A), dtype=np.float64)
    eye = np.eye(A, dtype=np.float64)
    for c in range(K_c):
        for th in range(L_field):
            pi_ct = pi_field[c, th]
            Q = S * pi_ct[None, :]
            Q[idx_diag, idx_diag] = 0.0
            Q[idx_diag, idx_diag] = -Q.sum(axis=1)
            weight = float(rho[th])
            for b in range(K_bins):
                m_b = float(bin_means[b])
                for t_i, t in enumerate(unique_tau_list):
                    if m_b == 0.0:
                        # Invariant bin: expm(0) = I regardless of τ.
                        P_cache[c, b, t_i] += weight * eye
                    else:
                        P_cache[c, b, t_i] += weight * expm(
                            Q * (m_b * float(t)))
    # Same root-fix as the no-bin branch: expm precision residuals can
    # slip small negatives into the mixture; clip and renormalise so
    # every row of every P_cache[c, b, t] is a valid distribution.
    P_cache = np.maximum(P_cache, 0.0)
    row_sums = P_cache.sum(axis=-1, keepdims=True)
    P_cache = P_cache / np.maximum(row_sums, 1e-300)
    log_P_cache = np.log(np.maximum(P_cache, 1e-300))
    return P_cache, tau_idx, log_P_cache


def compute_class_conditional_clv(leaf_msa: np.ndarray,
                                        parent: np.ndarray,
                                        cls: np.ndarray,
                                        pi_class: np.ndarray,
                                        P_cache: np.ndarray,
                                        tau_idx: np.ndarray,
                                        site_rate_bin: 'np.ndarray | None' = None,
                                        ) -> 'tuple[np.ndarray, np.ndarray]':
    """Per-site bottom-up Felsenstein under class-conditional GTR
    (par:arch-lg08-is): site s uses the class-c_s transition
    P^{(c_s)}(tau_v) at branch v.

    Args:
      leaf_msa:  (n_leaves, L) int8 — observed leaf residues (gap = 20).
      parent:    (n_nodes,) int32.
      cls:       (L,) int64 — per-site class label c_s.
      pi_class:  (K_c, A) — class stationaries (used only to size K_c).
      P_cache:   (K_c, n_tau, A, A) — from
                   build_class_conditional_transition_cache.
      tau_idx:   (n_nodes,) int64 — as above.

    Returns:
      clv:       (n_nodes, L, A) float64, rescaled to max=1 per (v, s).
      log_scale: (n_nodes, L) float64.
    """
    n_leaves = int(leaf_msa.shape[0])
    L = int(leaf_msa.shape[1])
    n_nodes = int(len(parent))
    K_c, A = pi_class.shape
    cls = np.asarray(cls, dtype=np.int64)
    assert cls.shape[0] == L

    clv = np.zeros((n_nodes, L, A), dtype=np.float64)
    log_scale = np.zeros((n_nodes, L), dtype=np.float64)
    for i in range(n_leaves):
        for s in range(L):
            x = int(leaf_msa[i, s])
            if 0 <= x < A:
                clv[i, s, x] = 1.0
            else:
                clv[i, s, :] = 1.0

    # Children lookup.
    children: 'list[list[int]]' = [[] for _ in range(n_nodes)]
    for v in range(n_nodes):
        p = int(parent[v])
        if p >= 0:
            children[p].append(v)

    # Post-order: leaves first (already there), then any node whose
    # children are all processed. Simple iterative scheme by DFS from root.
    root = int(np.where(parent == -1)[0][0])
    post_order: 'list[int]' = []
    stack: 'list[tuple[int, bool]]' = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            post_order.append(node)
        else:
            stack.append((node, True))
            for c in children[node]:
                stack.append((c, False))

    # Peel internals. All the clamped divisions below are numerically
    # valid but numpy fires RuntimeWarnings when the intermediates hit
    # subnormals; suppress the noise. A finite-guard at the end
    # replaces any site whose CLV somehow went non-finite (e.g. via a
    # trap-state archetype at pi=1e-30 * huge τ pushing expm entries
    # around subnormals) with a uniform distribution — a degraded but
    # non-poisoning value that keeps the peeling running.
    with np.errstate(over='ignore', under='ignore',
                        invalid='ignore', divide='ignore'):
        for v in post_order:
            if not children[v]:
                continue
            v_clv = np.ones((L, A), dtype=np.float64)
            for c in children[v]:
                if site_rate_bin is None:
                    P_c = P_cache[cls, tau_idx[c]]
                else:
                    P_c = P_cache[cls, site_rate_bin, tau_idx[c]]
                contrib = np.einsum('sab,sb->sa', P_c, clv[c])
                v_clv *= contrib
                log_scale[v] += log_scale[c]
            m = v_clv.max(axis=-1)
            m = np.maximum(m, 1e-300)
            v_clv = v_clv / m[:, None]
            # Finite-guard.
            bad = ~np.isfinite(v_clv).all(axis=-1) | (
                v_clv.sum(axis=-1) <= 0)
            if bad.any():
                v_clv[bad] = 1.0 / A
                m[bad] = 1.0
            log_scale[v] += np.log(m)
            clv[v] = v_clv
    return clv, log_scale


def sample_class_conditional_history(clv: np.ndarray,
                                             parent: np.ndarray,
                                             cls: np.ndarray,
                                             leaf_msa: np.ndarray,
                                             pi_class: np.ndarray,
                                             P_cache: np.ndarray,
                                             tau_idx: np.ndarray,
                                             rng: np.random.Generator,
                                             site_rate_bin: 'np.ndarray | None' = None,
                                             ) -> np.ndarray:
    """Top-down conditional sample of a whole-tree history under the
    class-conditional GTR proposal (par:arch-lg08-is).

    Returns X: (n_nodes, L) int8.
    """
    n_nodes, L, A = clv.shape
    n_leaves = int(leaf_msa.shape[0])
    cls = np.asarray(cls, dtype=np.int64)
    root_id = int(np.where(parent == -1)[0][0])

    X = np.zeros((n_nodes, L), dtype=np.int8)
    for i in range(n_leaves):
        for s in range(L):
            x_obs = int(leaf_msa[i, s])
            if 0 <= x_obs < A:
                X[i, s] = x_obs

    # Suppress benign subnormal-arithmetic warnings; finite-guard the
    # per-site sampling distributions so a rare zero row falls back to
    # uniform rather than tripping rng.choice with NaN.
    with np.errstate(over='ignore', under='ignore',
                        invalid='ignore', divide='ignore'):
        # Root: sample from pi_class[cls[s]] * clv[root, s] / Z per site.
        pi_at_sites = pi_class[cls]                          # (L, A)
        root_w = pi_at_sites * clv[root_id]
        s_root = root_w.sum(axis=-1, keepdims=True)
        root_w_norm = root_w / np.maximum(s_root, 1e-300)
        bad_root = ~np.isfinite(root_w_norm).all(axis=-1) | (
            np.squeeze(s_root, axis=-1) <= 0)
        if bad_root.any():
            root_w_norm[bad_root] = 1.0 / A
        cum = root_w_norm.cumsum(axis=-1)
        u = rng.random(size=(L, 1))
        X[root_id] = (u < cum).argmax(axis=-1).astype(np.int8)

        order = _preorder_from_parent(parent)
        for v in order:
            v = int(v)
            p = int(parent[v])
            if p < 0:
                continue
            if site_rate_bin is None:
                P_v = P_cache[cls, tau_idx[v]]
            else:
                P_v = P_cache[cls, site_rate_bin, tau_idx[v]]
            x_pa = X[p]
            idx_s = np.arange(L)
            w = P_v[idx_s, x_pa] * clv[v]
            s_w = w.sum(axis=-1, keepdims=True)
            w_norm = w / np.maximum(s_w, 1e-300)
            bad_w = ~np.isfinite(w_norm).all(axis=-1) | (
                np.squeeze(s_w, axis=-1) <= 0)
            if bad_w.any():
                w_norm[bad_w] = 1.0 / A
            cum = w_norm.cumsum(axis=-1)
            u = rng.random(size=(L, 1))
            sampled = (u < cum).argmax(axis=-1).astype(np.int8)
            if v < n_leaves:
                gap_mask = (leaf_msa[v] >= A) | (leaf_msa[v] < 0)
                X[v] = np.where(gap_mask, sampled, X[v])
            else:
                X[v] = sampled
    return X


def sample_joint_tree_history(clv: np.ndarray,
                                 parent: np.ndarray,
                                 tau: np.ndarray,
                                 leaf_msa: np.ndarray,
                                 rng: np.random.Generator,
                                 pi: np.ndarray = PI_LG08,
                                 Q: np.ndarray = Q_LG08,
                                 ) -> np.ndarray:
    """Top-down conditional sample of internal residues at every site
    under the LG08 tree posterior (par:arch-lg08-is).

    At each site s:
      - x_root^s ~ pi(a) * clv[root, s, a] / Z
      - x_v^s | x_pa(v)^s ~ P_LG08(τ_v)[x_pa, a] * clv[v, s, a] / Z
        in pre-order.

    Leaves are copied from `leaf_msa`; if a leaf position is a gap
    (leaf_msa == 20) the sampled residue is drawn from the CLV directly
    (which is uniform at gap positions by construction).

    Args:
      clv:      (n_nodes, L, A) float — rescaled bottom-up CLV.
      parent:   (n_nodes,) int32 — parent id per node; -1 at root.
      tau:      (n_nodes,) float64 — branch length to parent (0 at root).
      leaf_msa: (n_leaves, L) int8 — observed leaf residues (gap = 20).
      rng:      numpy Generator.
      pi, Q:    LG08 stationary + rate matrix.

    Returns:
      X: (n_nodes, L) int8 — sampled residue at every node/site.
    """
    n_nodes, L, A = clv.shape
    n_leaves = int(leaf_msa.shape[0])
    root_id = int(np.where(parent == -1)[0][0])
    clv64 = clv.astype(np.float64, copy=False)

    # Cache expm(Q * tau) per unique branch length (excluding root's 0).
    tau_cache: 'dict[float, np.ndarray]' = {}
    for v in range(n_nodes):
        if int(parent[v]) < 0:
            continue
        key = float(np.round(float(tau[v]) / 1e-6) * 1e-6)
        if key not in tau_cache:
            tau_cache[key] = expm(Q * key)
    def P_of(tau_v: float) -> np.ndarray:
        return tau_cache[float(np.round(tau_v / 1e-6) * 1e-6)]

    X = np.zeros((n_nodes, L), dtype=np.int8)

    # Leaves: copy observed. At gap positions we defer to CLV-sampling
    # below (leaves are visited in pre-order like any node), which under
    # a uniform gap-CLV gives a uniform draw.
    for i in range(n_leaves):
        for s in range(L):
            x_obs = int(leaf_msa[i, s])
            if 0 <= x_obs < A:
                X[i, s] = x_obs
            # else: sampled below via CLV @ P_LG conditional

    order = _preorder_from_parent(parent)

    # Root: sample from pi * clv[root] / Z per site.
    root_w = pi[None, :] * clv64[root_id]           # (L, A)
    root_w_norm = root_w / np.maximum(root_w.sum(axis=-1, keepdims=True), 1e-300)
    # Categorical sample per site.
    cum = root_w_norm.cumsum(axis=-1)
    u = rng.random(size=(L, 1))
    X[root_id] = (u < cum).argmax(axis=-1).astype(np.int8)

    # Non-root: conditional sample given parent.
    for v in order:
        v = int(v)
        p = int(parent[v])
        if p < 0:
            continue
        # For LEAF nodes with a non-gap observation, skip (already set).
        # But we still need to visit their gap positions.
        P_v = P_of(float(tau[v]))                       # (A, A) LG08 transition
        x_pa = X[p]                                     # (L,) parent residues
        # w[s, a] = P_v[x_pa[s], a] * clv[v, s, a]
        w = P_v[x_pa] * clv64[v]                        # (L, A)
        w_norm = w / np.maximum(w.sum(axis=-1, keepdims=True), 1e-300)
        cum = w_norm.cumsum(axis=-1)
        u = rng.random(size=(L, 1))
        sampled = (u < cum).argmax(axis=-1).astype(np.int8)
        if v < n_leaves:
            # Leaf: keep observed residues where non-gap, use sampled at gaps.
            gap_mask = (leaf_msa[v] >= A) | (leaf_msa[v] < 0)
            X[v] = np.where(gap_mask, sampled, X[v])
        else:
            X[v] = sampled

    return X
