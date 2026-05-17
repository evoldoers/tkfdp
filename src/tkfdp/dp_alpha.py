"""DP-concentration update for the symmetric finite-K Dirichlet-Multinomial.

For class counts {n_{k,m}} across MSAs m and a Gamma prior α_c ~ Gamma(a, b),
the posterior is

    log p(α_c | counts) ∝ (a-1) log α_c - b α_c
        + Σ_m [ log Γ(α_c) - log Γ(α_c + L_m)
                + Σ_k (log Γ(α_c/K + n_{k,m}) - log Γ(α_c/K)) ]

where L_m = Σ_k n_{k,m}. Not conjugate with the Dirichlet-Multinomial
likelihood, but the log-posterior is smooth and concave on log-α_c for
typical priors and easily optimised via 1D Brent search.

We use scipy.optimize.minimize_scalar in log-space to find the MAP
value, robust over α_c ∈ (1e-3, 1e3).

For the truncated-DP / TSB upgrade described in
implementation_notes.md §13, this routine should be replaced by the
Escobar-West auxiliary-variable Gibbs update on the underlying
DP — that's the principled path. The finite-K MAP here is a stopgap.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


def map_update_alpha_c(class_counts_per_msa: list[np.ndarray],
                        K: int,
                        prior_a: float = 2.0, prior_b: float = 1.0,
                        bracket: tuple[float, float] = (1e-3, 1e3)) -> float:
    """MAP estimate of α_c given class counts. Gamma(a, b) prior, optimised
    in log-space. Returns the optimal α_c."""
    counts_arrs = [np.asarray(c, dtype=np.float64) for c in class_counts_per_msa
                   if np.asarray(c).sum() > 0]
    if not counts_arrs:
        return prior_a / max(prior_b, 1e-12)

    def neg_log_post(log_alpha):
        alpha = float(np.exp(log_alpha))
        per_class = alpha / K
        ll = (prior_a - 1) * log_alpha - prior_b * alpha
        for n in counts_arrs:
            L_m = float(n.sum())
            # log Γ(α) - log Γ(α + L) + Σ_k [log Γ(α/K + n_k) - log Γ(α/K)]
            ll += gammaln(alpha) - gammaln(alpha + L_m)
            ll += np.sum(gammaln(per_class + n) - gammaln(per_class))
        return -ll

    res = minimize_scalar(neg_log_post,
                            bounds=(np.log(bracket[0]), np.log(bracket[1])),
                            method='bounded',
                            options=dict(xatol=1e-3))
    return float(np.exp(res.x))
