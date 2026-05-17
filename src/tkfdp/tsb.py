"""Truncated stick-breaking site-class DP.

Per main.tex §7.4 (Ishwaran & James 2001; Blei & Jordan 2006), we
truncate the site-class stick at level K_c, set β_{K_c} = 1 so ρ_k = 0
for k > K_c, and treat the K_c-1 "free" stick proportions {β_k} plus
the concentration α_c as latent variables to be inferred from the
class assignments {c_s}.

This module supplies a *Gibbs-style* update (rather than the full
Beta CAVI of main.tex §7.4) for compatibility with the hard-assignment
infrastructure in `partition_K.py`. Per outer iter, after the joint
(partner, class) Gibbs sweep across all MSAs:

  1. Pool class counts n_k = Σ_m n_{k,m} across all training MSAs.
  2. Sample (or MAP) β_k from Beta(1 + n_k, α_c + Σ_{j>k} n_j).
  3. Optionally MAP-update α_c from its conditional given {β_k}.
  4. Convert {β_k} to weights ρ_k for the next outer iter's class prior.

The stick weights {ρ_k} replace the symmetric Dirichlet-Multinomial
prior used in the finite-K stopgap. The truncation density-error
decays as 4N exp(-(K_c-1)/α_c); for α_c ∈ [3, 10] and N moderate,
K_c = 16-32 leaves negligible slack.

Future upgrade per main.tex §7.4: full Beta-mean-field CAVI on stick
posteriors and Categorical-mean-field on assignments, with importance-
weighted particles for per-class profiles {(η_c, π^(c))}. Not needed
yet since our v0 keeps profiles fixed at LG08.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, betaln


def stick_to_weights(beta: np.ndarray) -> np.ndarray:
    """Convert stick proportions β_1..β_{K_c-1} to weights ρ_1..ρ_{K_c}.
    Sets β_{K_c} = 1 so the weights sum to 1 exactly."""
    K_c = len(beta) + 1
    rho = np.zeros(K_c, dtype=np.float64)
    log_remaining = 0.0
    for k in range(K_c - 1):
        rho[k] = beta[k] * np.exp(log_remaining)
        log_remaining += np.log1p(-beta[k])
    rho[K_c - 1] = np.exp(log_remaining)
    return rho


def update_betas_from_counts(class_counts_global: np.ndarray,
                              alpha_c: float,
                              rng: np.random.Generator | None = None,
                              mode: str = 'sample') -> np.ndarray:
    """Conjugate posterior on stick proportions given total class counts.

    Posterior: β_k ~ Beta(1 + n_k, α_c + Σ_{j>k} n_j),  for k = 1..K_c-1.

    `mode` ∈ {'sample', 'map'}: 'sample' draws from the Beta posterior,
    'map' returns the Beta mean (the natural CAVI update).
    """
    K_c = len(class_counts_global)
    beta = np.zeros(K_c - 1, dtype=np.float64)
    cum_after = class_counts_global[::-1].cumsum()[::-1]   # cum_after[k] = Σ_{j>=k} n_j
    if rng is None and mode == 'sample':
        rng = np.random.default_rng()
    for k in range(K_c - 1):
        a = 1.0 + class_counts_global[k]
        b = alpha_c + (cum_after[k + 1] if k + 1 < K_c else 0.0)
        if mode == 'sample':
            beta[k] = float(rng.beta(a, b))
        else:  # MAP / mean
            beta[k] = a / (a + b)
        # numerical guard
        beta[k] = float(np.clip(beta[k], 1e-9, 1.0 - 1e-9))
    return beta


def map_update_alpha_c_from_betas(beta: np.ndarray,
                                    prior_a: float = 2.0, prior_b: float = 1.0,
                                    bracket: tuple[float, float] = (1e-3, 1e3)) -> float:
    """MAP update of α_c given {β_k}.

    Conditional likelihood on α_c: each β_k ~ Beta(1, α_c) gives
    log p(β_k | α_c) = log α_c + (α_c - 1) * log(1 - β_k).

    With Gamma(prior_a, prior_b) prior, the log-posterior is:
       (prior_a - 1) log α_c - prior_b * α_c
       + (K_c - 1) * log α_c
       + (α_c - 1) * Σ log(1 - β_k)

    Concave on log α_c; optimised by 1D Brent search.
    """
    K_minus = len(beta)
    log1m = np.sum(np.log1p(-beta))
    def neg_log_post(log_a):
        a = float(np.exp(log_a))
        return -((prior_a - 1) * log_a - prior_b * a
                  + K_minus * log_a + (a - 1) * log1m)
    res = minimize_scalar(neg_log_post,
                            bounds=(np.log(bracket[0]), np.log(bracket[1])),
                            method='bounded',
                            options=dict(xatol=1e-3))
    return float(np.exp(res.x))
