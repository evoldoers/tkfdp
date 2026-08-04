"""Per-cluster Felsenstein-on-theta MCMC sampler.

For each (family, cluster) we sample:
  - theta_v at every internal node v of the tree
  - a binary jump indicator M_v on every branch (needed for rho_chain
    identifiability)
conditional on the already-sampled residue trajectory X.

Under Interp 2 with F81-on-DP the per-branch cluster emission factors
into (par:arch-lg08-is discussion continuation, 2026-07):

    E_cluster(theta_p, theta_c) = A(theta_p) * delta_{theta_p=theta_c}
                                 + B(theta_p) * C(theta_c)

    A(theta)   = beta(theta) * Sigma_prod(theta)
    B(theta)   = 1 - beta(theta)
    C(theta)   = rho[theta] * pi_prod_c(theta)
    beta(theta) = exp(-rho_chain * (1 - rho[theta]) * tau)
    Sigma_prod(theta) = prod_{s in C} expm(Q^{(c_s, theta)} tau)[X_p^s, X_c^s]
    pi_prod_c(theta)  = prod_{s in C} pi^{(c_s, theta)}(X_c^s)

Felsenstein-on-theta per branch is then O(L * m) rather than O(L^2 * m).

Top-down sampling and per-branch M sampling follow directly.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

A_ALPH = 20


def _preorder_from_parent(parent: np.ndarray) -> np.ndarray:
    n = len(parent)
    root = int(np.where(parent == -1)[0][0])
    children_of: 'list[list[int]]' = [[] for _ in range(n)]
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


def _postorder_from_parent(parent: np.ndarray) -> np.ndarray:
    n = len(parent)
    root = int(np.where(parent == -1)[0][0])
    children_of: 'list[list[int]]' = [[] for _ in range(n)]
    for i in range(n):
        p = int(parent[i])
        if p >= 0:
            children_of[p].append(i)
    postorder = []
    stack: 'list[tuple[int, bool]]' = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            postorder.append(node)
        else:
            stack.append((node, True))
            for c in children_of[node]:
                stack.append((c, False))
    return np.asarray(postorder, dtype=np.int32)


def _children_of(parent: np.ndarray) -> 'list[list[int]]':
    n = len(parent)
    children: 'list[list[int]]' = [[] for _ in range(n)]
    for i in range(n):
        p = int(parent[i])
        if p >= 0:
            children[p].append(i)
    return children


def sample_theta_per_cluster(X: np.ndarray,
                                    parent: np.ndarray,
                                    tau: np.ndarray,
                                    classes: np.ndarray,
                                    pi_field: np.ndarray,
                                    rho: np.ndarray,
                                    S: np.ndarray,
                                    rho_chain: float,
                                    rng: np.random.Generator,
                                    ) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Draw theta_v at every internal node and a binary M_v on every
    branch for ONE (family, cluster) pair.

    Args:
      X:        (n_nodes, L_cols) int8 residue at every node/column.
      parent:   (n_nodes,) int32.
      tau:      (n_nodes,) float64 branch length to parent (0 at root).
      classes:  (m,) int64 per-site class labels c_s for sites in the
                cluster (indexes into the columns of X).
      cluster_columns: (m,) int64 — the column indices in X that belong
                to this cluster. classes[s] = c_{cluster_columns[s]}.
      pi_field: (K_c, L_theta, A) per-(class, field) stationary.
      rho:      (L_theta,) TSB field prior.
      S:        (A, A) LG08 exchangeabilities.
      rho_chain: float — F81-on-DP field rate.
      rng:      numpy Generator.

    Returns:
      theta_sampled: (n_nodes,) int32 — theta at every node (values only
                       meaningful at internal nodes; leaves get theta_c).
      M_per_branch:  (n_nodes,) int8 — 0/1 jump indicator for the branch
                       above each node (0 at root).
      log_N_theta:   (n_nodes,) float — per-branch expected jumps
                       (= M_per_branch here) for aggregation into
                       N_theta_sum.
    """
    raise NotImplementedError("Use sample_theta_per_cluster_columns instead.")


def _build_per_branch_kernels(X: np.ndarray,
                                    cluster_columns: np.ndarray,
                                    classes: np.ndarray,
                                    parent: np.ndarray,
                                    tau: np.ndarray,
                                    pi_field: np.ndarray,
                                    rho: np.ndarray,
                                    S: np.ndarray,
                                    rho_chain: float,
                                    P_cache_full: 'np.ndarray | None' = None,
                                    tau_idx_full: 'np.ndarray | None' = None,
                                    log_pi_field: 'np.ndarray | None' = None,
                                    ) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Compute per-branch A(theta), B(theta), C(theta) shape (n_nodes, L_theta).

    Vectorised: no per-site or per-theta python loops.

    Optional shared caches (built once per family per iter):
      P_cache_full: (K_c, L_theta, n_tau_uniq, A, A) transition matrices.
      tau_idx_full: (n_nodes,) int64 bin index into the tau axis of
                    P_cache_full for each node's incoming-branch tau.
      log_pi_field: (K_c, L_theta, A) precomputed logs.
    When omitted these are built locally (slower on repeated calls).
    """
    n_nodes = int(parent.shape[0])
    K_c, L_theta, A = pi_field.shape
    m = int(classes.shape[0])
    is_root = parent < 0

    # Build shared caches if not supplied.
    if P_cache_full is None or tau_idx_full is None:
        P_cache_full, tau_idx_full = build_shared_P_cache(
            pi_field, S, parent, tau)
    if log_pi_field is None:
        log_pi_field = np.log(np.maximum(pi_field, 1e-300))

    # Gather residues at parent and child (child == this node) at cluster columns.
    parent_safe = np.where(is_root, 0, parent)
    X_p = X[parent_safe][:, cluster_columns].astype(np.int64)     # (n_nodes, m)
    X_c = X[:, cluster_columns].astype(np.int64)                  # (n_nodes, m)

    # Fancy index into P_cache_full [K_c, L_theta, n_tau, A, A]
    # for shape (n_nodes, L_theta, m):
    #   entries[v, th, s] = P_cache_full[classes[s], th, tau_idx_full[v], X_p[v,s], X_c[v,s]]
    cls_b = classes[None, None, :]                                # (1, 1, m)
    th_b = np.arange(L_theta)[None, :, None]                       # (1, L_theta, 1)
    tau_b = tau_idx_full[:, None, None]                            # (n_nodes, 1, 1)
    Xp_b = X_p[:, None, :]                                         # (n_nodes, 1, m)
    Xc_b = X_c[:, None, :]                                         # (n_nodes, 1, m)
    entries = P_cache_full[cls_b, th_b, tau_b, Xp_b, Xc_b]         # (n_nodes, L_theta, m)
    # Sigma_prod[v, theta] = prod_s entries[v, theta, s]
    sigma_prod = np.prod(np.maximum(entries, 1e-300), axis=2)      # (n_nodes, L_theta)

    # pi_prod_c[v, theta] = prod_s pi_field[classes[s], theta, X_c[v, s]]
    pi_entries = pi_field[cls_b, th_b, Xc_b]                       # (n_nodes, L_theta, m)
    pi_prod_c = np.prod(np.maximum(pi_entries, 1e-300), axis=2)     # (n_nodes, L_theta)

    # beta(theta) per branch: (n_nodes, L_theta).
    beta = np.exp(-rho_chain * (1.0 - rho)[None, :] * tau[:, None])   # (n_nodes, L_theta)
    A_arr = beta * sigma_prod
    B_arr = 1.0 - beta
    C_arr = rho[None, :] * pi_prod_c

    # Zero out root's row (no branch above root).
    A_arr[is_root] = 0.0
    B_arr[is_root] = 0.0
    C_arr[is_root] = 0.0
    return A_arr, B_arr, C_arr


def build_shared_P_cache(pi_field: np.ndarray,
                              S: np.ndarray,
                              parent: np.ndarray,
                              tau: np.ndarray,
                              cache_grid: float = 1e-6,
                              ) -> 'tuple[np.ndarray, np.ndarray]':
    """Build one shared (K_c, L_theta, n_tau_uniq, A, A) transition-matrix
    table per family per outer iter, reused across all clusters. scipy
    expm in a loop; on small A=20 matrices scipy is competitive with JAX
    even on GPU (measured 1.2x GPU speedup only).

    Returns:
      P_cache_full: (K_c, L_theta, n_tau_uniq, A, A) numpy float64.
      tau_idx_full: (n_nodes,) int64 bin index into the tau axis for each
                    node's incoming branch (0 at root).
    """
    n_nodes = int(parent.shape[0])
    K_c, L_theta, A = pi_field.shape
    unique_tau_map: 'dict[float, int]' = {}
    unique_tau_list: 'list[float]' = []
    for v in range(n_nodes):
        if int(parent[v]) < 0:
            continue
        key = float(np.round(float(tau[v]) / cache_grid) * cache_grid)
        if key not in unique_tau_map:
            unique_tau_map[key] = len(unique_tau_list)
            unique_tau_list.append(key)
    n_tau = len(unique_tau_list)
    P_cache = np.zeros((K_c, L_theta, n_tau, A, A), dtype=np.float64)
    idx_diag = np.arange(A)
    for c in range(K_c):
        for th in range(L_theta):
            pi_ct = pi_field[c, th]
            Q = S * pi_ct[None, :]
            Q[idx_diag, idx_diag] = 0.0
            Q[idx_diag, idx_diag] = -Q.sum(axis=1)
            for t_i, t in enumerate(unique_tau_list):
                P_cache[c, th, t_i] = expm(Q * float(t))
    tau_idx = np.zeros(n_nodes, dtype=np.int64)
    for v in range(n_nodes):
        if int(parent[v]) < 0:
            continue
        key = float(np.round(float(tau[v]) / cache_grid) * cache_grid)
        tau_idx[v] = unique_tau_map[key]
    return P_cache, tau_idx


def _felsenstein_on_theta_up(A_arr: np.ndarray,
                                    B_arr: np.ndarray,
                                    C_arr: np.ndarray,
                                    parent: np.ndarray,
                                    ) -> 'tuple[np.ndarray, np.ndarray]':
    """Bottom-up CLV_theta per node.

    Returns (clv_theta, log_scale). clv_theta[v, theta] rescaled to max=1
    per node; log_scale[v] tracks accumulated log-scale (for numerical
    stability on deep trees).
    """
    n_nodes, L_theta = A_arr.shape
    children = _children_of(parent)
    post_order = _postorder_from_parent(parent)
    clv = np.ones((n_nodes, L_theta), dtype=np.float64)
    log_scale = np.zeros(n_nodes, dtype=np.float64)

    for v in post_order:
        v = int(v)
        if not children[v]:
            # Leaf: CLV_theta_leaf(theta) = 1 (no theta-observation at leaf)
            continue
        acc = np.ones(L_theta, dtype=np.float64)
        for c in children[v]:
            # Per child c: factor = A(theta_v)*clv_c(theta_v) + B(theta_v)*<C_c, clv_c>
            #                                   [same-theta term]      [jump term]
            C_dot = float(np.sum(C_arr[c] * clv[c]))          # <C_c, clv_c> scalar
            factor = A_arr[c] * clv[c] + B_arr[c] * C_dot     # (L_theta,)
            acc *= np.maximum(factor, 1e-300)
            log_scale[v] += log_scale[c]
        m_scale = float(acc.max())
        if m_scale > 0.0:
            acc = acc / m_scale
            log_scale[v] += np.log(m_scale)
        clv[v] = acc

    return clv, log_scale


def _sample_theta_topdown(clv: np.ndarray,
                                A_arr: np.ndarray,
                                B_arr: np.ndarray,
                                C_arr: np.ndarray,
                                parent: np.ndarray,
                                rho: np.ndarray,
                                rng: np.random.Generator,
                                ) -> np.ndarray:
    """Top-down sample theta_v at every node.

    Root: theta_root ~ rho * clv_root(theta) / Z
    Non-root v: theta_v | theta_pa(v) ~ E_branch(theta_pa, theta_v) *
                    clv_v(theta_v) / Z
    """
    n_nodes, L_theta = clv.shape
    root = int(np.where(parent == -1)[0][0])
    theta_sampled = np.zeros(n_nodes, dtype=np.int32)

    def _safe_normalize(w: np.ndarray, tag: str) -> np.ndarray:
        """Return a proper probability vector or a uniform fallback.

        The Felsenstein CLV can underflow to 0 on deep trees + large
        clusters when every pi_field*P_cache entry is at the 1e-300
        clamp: prod_m clamp * clv[c] * B*C -> 0 exactly. When that
        happens (or NaN sneaks in through some other path), fall
        back to uniform rather than crashing `rng.choice`.
        """
        s = float(w.sum())
        if not np.isfinite(s) or s < 1e-100:
            return np.full(L_theta, 1.0 / L_theta, dtype=np.float64)
        w = w / s
        # Round-off safety: rng.choice checks sum in float64 with
        # tolerance ~1e-8. Re-normalise the same way numpy does.
        w = w / w.sum()
        return w

    # Root
    w = rho * clv[root]
    w = _safe_normalize(w, 'root')
    theta_sampled[root] = int(rng.choice(L_theta, p=w))

    pre_order = _preorder_from_parent(parent)
    for v in pre_order:
        v = int(v)
        if v == root:
            continue
        p_id = int(parent[v])
        theta_p = int(theta_sampled[p_id])
        # E_branch(theta_p, theta_v) = A[v, theta_p]*delta_{theta_v=theta_p}
        #                            + B[v, theta_p]*C[v, theta_v]
        # unnormalised weight over theta_v:
        w_v = B_arr[v, theta_p] * C_arr[v]                    # (L_theta,) — jump term
        w_v[theta_p] += A_arr[v, theta_p]                     # add same-theta term
        w_v = w_v * clv[v]
        w_v = _safe_normalize(w_v, f'v={v}')
        theta_sampled[v] = int(rng.choice(L_theta, p=w_v))
    return theta_sampled


def _sample_M_per_branch(theta_sampled: np.ndarray,
                                A_arr: np.ndarray,
                                B_arr: np.ndarray,
                                C_arr: np.ndarray,
                                parent: np.ndarray,
                                rng: np.random.Generator,
                                ) -> np.ndarray:
    """Binary M_v per branch given sampled theta endpoints.

    theta_p != theta_c: M = 1
    theta_p == theta_c: sample from posterior via Bernoulli.
    """
    n_nodes = theta_sampled.shape[0]
    M_per_branch = np.zeros(n_nodes, dtype=np.int8)
    for v in range(n_nodes):
        p_id = int(parent[v])
        if p_id < 0:
            continue
        theta_p = int(theta_sampled[p_id])
        theta_c = int(theta_sampled[v])
        if theta_p != theta_c:
            M_per_branch[v] = 1
            continue
        # theta_p == theta_c: two contributions
        #   M=0 term: A[v, theta_p]
        #   M>=1 term (returning to same theta): B[v, theta_p] * C[v, theta_p]
        p_M0 = float(A_arr[v, theta_p])
        p_M1 = float(B_arr[v, theta_p] * C_arr[v, theta_p])
        total = p_M0 + p_M1
        if total <= 0.0:
            M_per_branch[v] = 0
        else:
            u = float(rng.random())
            M_per_branch[v] = 0 if u * total < p_M0 else 1
    return M_per_branch


def sample_theta_cluster(X: np.ndarray,
                                cluster_columns: np.ndarray,
                                classes: np.ndarray,
                                parent: np.ndarray,
                                tau: np.ndarray,
                                pi_field: np.ndarray,
                                rho: np.ndarray,
                                S: np.ndarray,
                                rho_chain: float,
                                rng: np.random.Generator,
                                P_cache_full: 'np.ndarray | None' = None,
                                tau_idx_full: 'np.ndarray | None' = None,
                                ) -> 'tuple[np.ndarray, np.ndarray, dict]':
    """Sample theta_v (n_nodes,) and M_v (n_nodes,) for one cluster.

    Optional shared caches (P_cache_full, tau_idx_full) — pass them in to
    amortise the expm build across all clusters in the same family.

    Returns:
      theta_sampled: (n_nodes,) int32
      M_per_branch: (n_nodes,) int8
      diag:         dict with per-branch A, B, C caches and diagnostics
    """
    A_arr, B_arr, C_arr = _build_per_branch_kernels(
        X, cluster_columns, classes, parent, tau,
        pi_field, rho, S, rho_chain,
        P_cache_full=P_cache_full, tau_idx_full=tau_idx_full)
    clv, log_scale = _felsenstein_on_theta_up(A_arr, B_arr, C_arr, parent)
    theta_sampled = _sample_theta_topdown(
        clv, A_arr, B_arr, C_arr, parent, rho, rng)
    M_per_branch = _sample_M_per_branch(
        theta_sampled, A_arr, B_arr, C_arr, parent, rng)
    diag = dict(
        A_arr=A_arr, B_arr=B_arr, C_arr=C_arr,
        clv_theta=clv, log_scale=log_scale,
    )
    return theta_sampled, M_per_branch, diag
