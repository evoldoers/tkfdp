"""Rate heterogeneity for the class-marginalised dynfield scorer (+Gamma+I).

Two independent discretised Gamma+Invariant mixtures (Yang 1994; Yang 1996):

  * Per-SITE substitution rate m_n on the residue GTR. Estimated ONCE at
    init by empirical Bayes from the fixed LG08 per-site conditional
    likelihoods (per-column conservation), then held constant. Absorbs
    across-site rate variation so it is not misread as field dynamics.

  * Per-CLUSTER field-rate r_C on the F81-on-DP field chain
    (rho_chain -> rho_chain * r_C). SHARED-GLOBAL Gamma+I quantiles;
    marginalised (not fixed) during scoring. The invariant bin r_C = 0
    is the "static / no-flip" regime: g_tau = 1, theta constant on the
    tree. Flip posterior phi(C) = P(r_C != 0 | C).

Both share the same discretisation primitive. See appendix-tkfdp.tex
"Rate heterogeneity (+Gamma+I)".
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def discrete_gamma_rates(K: int, alpha: float) -> np.ndarray:
    """Yang's mean-normalised discrete-Gamma rate multipliers: K bins, each
    the mean rate over an equal-probability quantile of Gamma(alpha, alpha)
    (mean 1). Returns (K,) rates with mean exactly 1.

    Uses the standard "mean of quantile" discretisation (Yang 1994), not the
    median, so the K rates average to 1 by construction.
    """
    if K <= 1:
        return np.ones(1, dtype=np.float64)
    # equal-probability boundaries in Gamma(alpha, scale=1/alpha) space
    beta = alpha  # rate param so mean = alpha/beta = 1
    edges = stats.gamma.ppf(np.linspace(0, 1, K + 1), a=alpha, scale=1.0 / beta)
    edges[0], edges[-1] = 0.0, np.inf
    # mean of Gamma over each quantile via the incomplete-gamma identity:
    #   E[X | a<X<b] = (F_{alpha+1}(b) - F_{alpha+1}(a)) / (1/K) * (alpha/beta)
    ga1 = stats.gamma.cdf(edges, a=alpha + 1, scale=1.0 / beta)
    rates = (ga1[1:] - ga1[:-1]) * K * (alpha / beta)
    return rates.astype(np.float64)


def gamma_plus_inv_rates(K_gamma: int, alpha: float,
                         p_inv: float) -> tuple[np.ndarray, np.ndarray]:
    """(rates, weights) for a Gamma(alpha)+I mixture with K_gamma variable
    bins plus one invariant (rate 0) bin. Variable rates are RESCALED so the
    overall mean rate is 1 (the invariant bin contributes 0), matching the
    standard +Gamma+I convention: mean = (1 - p_inv) * mean(gamma_scaled) = 1.

    Returns:
      rates:   (K_gamma + 1,) with rates[0] = 0 (invariant), rest > 0.
      weights: (K_gamma + 1,) prior; weights[0] = p_inv.
    """
    g = discrete_gamma_rates(K_gamma, alpha)          # mean 1
    scaled = g / max(1.0 - p_inv, 1e-8)               # so (1-p_inv)*mean = 1
    rates = np.concatenate([[0.0], scaled])
    w_var = (1.0 - p_inv) / K_gamma
    weights = np.concatenate([[p_inv], np.full(K_gamma, w_var)])
    return rates.astype(np.float64), weights.astype(np.float64)


def _lg08_persite_ll_at_rates(leaf_msa, parent, tau, root, S, pi, rates):
    """Per-site LG08 log-lik under each branch-rate multiplier, via a compact
    Felsenstein pruning. leaf_msa (n_leaves, L) int (gap=20); parent/tau (n,);
    rates (G,). Returns (L, G)."""
    from scipy.linalg import expm
    A = pi.shape[0]; L = leaf_msa.shape[1]; n = len(parent)
    Q = S * pi[None, :]; np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(A)] = -Q.sum(1)
    from collections import defaultdict
    ch = defaultdict(list)
    for v in range(n):
        if parent[v] >= 0:
            ch[parent[v]].append(v)
    order, st = [], [root]
    while st:
        v = st.pop(); order.append(v); st.extend(ch[v])
    order = order[::-1]
    n_leaves = leaf_msa.shape[0]
    out = np.zeros((L, len(rates)))
    for gi, m in enumerate(rates):
        if m <= 0:                    # invariant (rate 0 = frozen substitution)
            # Under a truly frozen site every leaf equals the root, so a
            # NON-constant column has likelihood 0 -> -inf (never eligible for
            # the invariant bin). Only 100%-conserved columns survive. Field
            # rate is held at zero here (pure LG08 fit), so the field selector
            # cannot pick up the slack and make a varying column look frozen.
            for s in range(L):
                col = leaf_msa[:, s]; obs = col[col < 20]
                out[s, gi] = (np.log(pi[obs[0]])
                              if len(obs) and np.all(obs == obs[0])
                              else -np.inf)
            continue
        Pc = {}
        clv = {}; lsc = {}
        for v in order:
            if not ch[v]:
                M = np.ones((L, A))
                if v < n_leaves:
                    for s in range(L):
                        x = int(leaf_msa[v, s])
                        if x < 20:
                            e = np.zeros(A); e[x] = 1.0; M[s] = e
                clv[v] = M; lsc[v] = np.zeros(L)
            else:
                M = np.ones((L, A)); sc = np.zeros(L)
                for c in ch[v]:
                    t = float(tau[c]) * m
                    if t not in Pc:
                        Pc[t] = expm(Q * t)
                    M = M * (clv[c] @ Pc[t].T)
                    sc = sc + lsc[c]
                mx = M.max(1, keepdims=True); mx[mx <= 0] = 1.0
                clv[v] = M / mx; sc = sc + np.log(mx[:, 0])
                lsc[v] = sc
        out[:, gi] = np.log(clv[root] @ pi) + lsc[root]
    return out


def estimate_site_rate_bins(clv_paths, S, pi_lg, rates, weights):
    """Per-column FIXED substitution-rate-bin index, estimated ONCE by
    empirical Bayes from LG08 per-site likelihoods (argmax posterior over the
    G rate bins) and held fixed. rates/weights define the shared Gamma+I grid
    (rates[0]=0 = invariant/constant sites). Returns {family: (L,) int in [0,G)}.

    This is the efficient "extra classes" route: the discrete bin is folded
    into the class index (base_class x rate_bin) so per-site rate needs no
    forward change -- just a G-fold-wider P_sub / pi_field table."""
    from tkfdp.pfam_data import load_clv_family
    logw = np.log(np.asarray(weights, float))
    out = {}
    for p in clv_paths:
        fd = load_clv_family(p)
        llk = _lg08_persite_ll_at_rates(
            np.asarray(fd.leaf_msa), np.asarray(fd.parent),
            np.asarray(fd.tau), int(fd.root_id), S, np.asarray(pi_lg), rates)
        out[str(fd.family)] = np.argmax(llk + logw[None, :],
                                        axis=1).astype(np.int32)
    return out


def init_site_rates(log_lik_per_site_by_rate: np.ndarray,
                    rates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Empirical-Bayes posterior-MEAN per-site rate multiplier, held fixed.

    Args:
      log_lik_per_site_by_rate: (L, G) log P(column s | LG08, branch-scale
        rates[g]) -- the per-site conditional likelihood under each rate bin
        (invariant bin included, rate 0 => only constant-column sites survive).
      rates:   (G,) rate multipliers (rates[0] = 0 invariant).
      weights: (G,) rate prior.

    Returns:
      m: (L,) posterior-mean rate multiplier per column.
    """
    lp = log_lik_per_site_by_rate + np.log(weights)[None, :]
    lp -= lp.max(axis=1, keepdims=True)
    post = np.exp(lp)
    post /= post.sum(axis=1, keepdims=True)
    return (post * rates[None, :]).sum(axis=1)
