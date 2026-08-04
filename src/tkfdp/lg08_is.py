"""Per-site importance weights for the LG08-IID proposal on tree
histories (appendix par:arch-lg08-is).

Given a sampled joint history X (n_nodes, L) drawn from
q(X | y, tree) = P_LG08(X | y, tree), the SNIS weight for a cluster C
under the current TKF-DP model parameters Theta is (up to a
family-wise constant that cancels in self-normalisation):

    w_C  proportional to  P_TKFDP(X_C, y_C | tree, Theta)
                                 / P_LG08(X_C, y_C | tree)

Both densities factor over branches; the composite training regime
treats sites in a cluster as conditionally independent given the
per-site class label c_s, so

    log w_C  =  sum_{s in C}  log P_TKFDP(X^s, y^s | tree, c_s)
                            - log P_LG08(X^s, y^s | tree)

Each per-site log-probability is a sum over the tree's branches of a
GTR transition-matrix entry evaluated at the sampled residues.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .lg08 import PI_LG08, Q_LG08


A_ALPH = 20


def compute_lg08_log_p_per_site(X: np.ndarray,
                                     parent: np.ndarray,
                                     tau: np.ndarray,
                                     pi: np.ndarray = PI_LG08,
                                     Q: np.ndarray = Q_LG08,
                                     ) -> np.ndarray:
    """Total log P_LG08(X^s, y^s | tree) per site.

    log P = log pi_LG[x_root^s] + sum_v log P_LG08(x_pa(v)^s -> x_v^s;
    tau_v).

    Args:
      X:      (n_nodes, L) int8 — full node/site residue assignment
              (leaves already fixed to their observed values).
      parent: (n_nodes,) int32 — parent index; -1 at root.
      tau:    (n_nodes,) float64 — branch length to parent (0 at root).
      pi, Q:  LG08 stationary and rate matrix.

    Returns:
      log_p_per_site: (L,) float64.
    """
    n_nodes, L = X.shape
    root_id = int(np.where(parent == -1)[0][0])

    log_pi = np.log(np.maximum(pi, 1e-300))
    log_p = np.zeros(L, dtype=np.float64)
    log_p += log_pi[X[root_id]]                         # root prior

    # Cache expm(Q * tau_v) per unique branch length.
    tau_cache: 'dict[float, np.ndarray]' = {}
    for v in range(n_nodes):
        if int(parent[v]) < 0:
            continue
        key = float(np.round(float(tau[v]) / 1e-6) * 1e-6)
        if key not in tau_cache:
            tau_cache[key] = np.log(
                np.maximum(expm(Q * key), 1e-300))
    def logP_of(tau_v: float) -> np.ndarray:
        return tau_cache[float(np.round(tau_v / 1e-6) * 1e-6)]

    for v in range(n_nodes):
        p = int(parent[v])
        if p < 0:
            continue
        log_P_v = logP_of(float(tau[v]))                # (A, A)
        log_p += log_P_v[X[p], X[v]]                    # (L,) via fancy index
    return log_p


def compute_tkfdp_log_p_per_site(X: np.ndarray,
                                      parent: np.ndarray,
                                      tau: np.ndarray,
                                      cls: np.ndarray,
                                      pi_class: np.ndarray,
                                      log_P_single_cache: np.ndarray,
                                      tau_idx: np.ndarray,
                                      ) -> np.ndarray:
    """Total log P_TKFDP(X^s, y^s | tree, c_s) per site under the
    class-marginal F81-on-DP transition.

    log P = log pi_class[c_s, x_root^s]
          + sum_v log P^{c_s}(x_pa(v)^s -> x_v^s; tau_v)

    where `log_P_single_cache[c, t_idx, a, b] = log P^{c}(a -> b; tau_t)`
    is the pre-built cache from the SVI outer loop (K_c, n_tau, A, A).

    Args:
      X:                   (n_nodes, L) int8.
      parent:              (n_nodes,) int32.
      tau:                 (n_nodes,) float64 (unused except for shape;
                              transitions come from log_P_single_cache).
      cls:                 (L,) int64 — per-site class label c_s.
      pi_class:            (K_c, A) float — class-marginal residue prior.
      log_P_single_cache:  (K_c, n_tau, A, A) — precomputed transition
                              log-probs at each unique tau bin.
      tau_idx:             (n_nodes,) int64 — index into unique-tau axis
                              for each branch's tau_v (0 at root).

    Returns:
      log_p_per_site: (L,) float64.
    """
    n_nodes, L = X.shape
    root_id = int(np.where(parent == -1)[0][0])
    cls = np.asarray(cls, dtype=np.int64)
    x_root = X[root_id].astype(np.int64)                # (L,)

    log_pi_c = np.log(np.maximum(pi_class, 1e-300))
    log_p = log_pi_c[cls, x_root]                       # (L,)

    for v in range(n_nodes):
        p = int(parent[v])
        if p < 0:
            continue
        t_bin = int(tau_idx[v])
        # log P^{c_s}(X[p, s] -> X[v, s]; tau_bin[v]) for each site s.
        x_p = X[p].astype(np.int64)
        x_v = X[v].astype(np.int64)
        # Fancy index into (K_c, n_tau, A, A).
        log_p += log_P_single_cache[cls, t_bin, x_p, x_v]
    return log_p


def build_log_p_class_marg(pi_field: np.ndarray,
                                rho: np.ndarray,
                                S: np.ndarray,
                                unique_tau: np.ndarray,
                                ) -> np.ndarray:
    """Class-marginal GTR transition matrices under the current F81-on-DP
    parameters.

    For every (class c, tau in unique_tau) build
        P^c_{marg}(tau)[a, b]
          = sum_theta rho[theta] * expm(Q^{c, theta} * tau)[a, b]
        Q^{c, theta}[i, j] = S[i, j] * pi_field[c, theta, j]  (i != j)
        Q^{c, theta}[i, i] = -sum_{j!=i} Q^{c, theta}[i, j]

    Args:
      pi_field:   (K_c, L, A) — per-(class, field) stationary.
      rho:        (L,) — TSB field prior.
      S:          (A, A) — LG08 exchangeabilities.
      unique_tau: (n_tau,) float — unique branch lengths in the family.

    Returns:
      log_P: (K_c, n_tau, A, A) float64 — element-wise log.
    """
    K_c, L, A = pi_field.shape
    n_tau = int(unique_tau.shape[0])
    log_P = np.zeros((K_c, n_tau, A, A), dtype=np.float64)
    idx_diag = np.arange(A)
    for c in range(K_c):
        for th in range(L):
            pi_ct = pi_field[c, th]
            Q = S * pi_ct[None, :]                          # (A, A)
            Q[idx_diag, idx_diag] = 0.0
            Q[idx_diag, idx_diag] = -Q.sum(axis=1)
            for t_i in range(n_tau):
                P = expm(Q * float(unique_tau[t_i]))
                # Accumulate into log-space at the end.
                log_P[c, t_i] += rho[th] * P
    log_P = np.log(np.maximum(log_P, 1e-300))
    return log_P


def build_log_p_lg08(unique_tau: np.ndarray,
                          pi: np.ndarray = PI_LG08,
                          Q: np.ndarray = Q_LG08,
                          ) -> np.ndarray:
    """LG08 transition matrices, one per unique branch length.

    Returns log_P: (n_tau, A, A) float64.
    """
    n_tau = int(unique_tau.shape[0])
    log_P = np.zeros((n_tau, A_ALPH, A_ALPH), dtype=np.float64)
    for t_i in range(n_tau):
        P = expm(Q * float(unique_tau[t_i]))
        log_P[t_i] = np.log(np.maximum(P, 1e-300))
    return log_P


def _tau_to_idx(tau: np.ndarray, unique_tau: np.ndarray) -> np.ndarray:
    """Return (n,) int64 index into unique_tau for each tau_v."""
    order = {float(u): i for i, u in enumerate(unique_tau)}
    return np.array([order.get(float(t), 0) for t in tau],
                       dtype=np.int64)


def per_cluster_log_weights(log_p_tkfdp_per_site: np.ndarray,
                                 log_p_lg08_per_site: np.ndarray,
                                 cluster_of_site: 'list[np.ndarray]',
                                 ) -> np.ndarray:
    """Aggregate per-site log-density ratios into per-cluster log weights.

    Args:
      log_p_tkfdp_per_site: (L,) float64.
      log_p_lg08_per_site:  (L,) float64.
      cluster_of_site: list of (site_indices,) int64 arrays — one entry
        per cluster, giving the site indices belonging to that cluster.

    Returns:
      log_w: (n_clusters,) float64.
    """
    log_ratio = log_p_tkfdp_per_site - log_p_lg08_per_site  # (L,)
    return np.array([log_ratio[idxs].sum() for idxs in cluster_of_site],
                       dtype=np.float64)


def self_normalise(log_w: np.ndarray) -> 'tuple[np.ndarray, float]':
    """Turn per-cluster log weights into a self-normalised weight vector
    plus the effective sample size (par:arch-lg08-is eq. ESS).

    Uses log-sum-exp for stability. Returns (w, ess) with
    w.sum() == 1 and 1 <= ess <= len(w).
    """
    if log_w.size == 0:
        return np.zeros_like(log_w), 0.0
    log_w_max = float(log_w.max())
    unnorm = np.exp(log_w - log_w_max)
    total = float(unnorm.sum())
    if total <= 0.0:
        return np.full_like(log_w, 1.0 / log_w.size), 0.0
    w = unnorm / total
    ess = float(1.0 / np.sum(w * w))
    return w, ess
