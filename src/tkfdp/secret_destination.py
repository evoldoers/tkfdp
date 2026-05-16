"""Secret-destination augmentation for the F81 generator (main.tex \S7.4).

Under the F81 form Q^s_{xy} = eta_s * S_{xy} * pi^(c)(y) * exp(-0.5 dH_s),
the dynamics decompose into:
- An S-driven proposal Poisson clock at rate eta_s * S_{xy} for each
  proposal x -> y;
- A Bernoulli filter that accepts the proposal with probability
  pi^(c)(y) * exp(-0.5 dH_s).

Rejected proposals are silent (a "ghost"). To recover full multinomial
structure on pi^(c), each rejected proposal is augmented with a
"secret destination" j != y drawn from pi[j] / (1 - pi[y]). Every
proposal then casts exactly one Categorical(pi^(c)) vote on its final
destination state. The augmented count vector

    N^(c)_y = (real accepted moves at destination y, summed over branches)
            + (secret-destination ghost contributions at y)

is Multinomial(N^(c)_total, pi^(c)) by construction, and the closed-
form Dirichlet posterior is

    pi^(c) | N^(c) ~ Dirichlet(kappa_pi * pi_bar + N^(c)).

In EM/VBEM the augmentation is replaced by its conditional expectation:
each proposal contributes pi^(c)(y) to E[N^(c)_y] regardless of the
proposed destination (the accept-and-vote-y case contributes pi(y); the
reject-with-secret-y case contributes (1 - pi(y'))*pi(y)/(1 - pi(y'))
= pi(y) for any y' != y), so

    E[N^(c)_y]
        = pi^(c)(y) * sum_b sum_x T^(b, s)_x * (sum_{y'} S_{x, y'} - S_{x, y})
        = pi^(c)(y) * (T_S(b, s) - S_dot(b, s, y))

computable in O(A^2) per branch as one matrix-vector product against S.

This module exposes:
  expected_ghost_counts(pi, S, dwell_per_state) -> N_ghost  (A-vector)
  expected_real_counts(transition_count_matrix) -> N_real  (A-vector)
  dirichlet_posterior(prior_alpha, real, ghost) -> posterior_alpha
  dirichlet_log_marginal(prior_alpha, real, ghost) -> log B(post)/B(prior)

The Dirichlet log marginal is the closed-form integral
  ∫ L(data | pi') Dir(pi' | prior_alpha) dpi'
  = log B(prior_alpha + N_total) - log B(prior_alpha)

(here N_total = real + ghost, with the augmentation-conditional-
expectation gloss for VBEM use).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def expected_ghost_counts(pi: np.ndarray, S: np.ndarray,
                           dwell_per_state: np.ndarray,
                           eta: float = 1.0) -> np.ndarray:
    """Closed-form expected ghost destination counts under EM.

    `pi`: (A,) class-stationary distribution.
    `S`: (A, A) symmetric exchangeability.
    `dwell_per_state`: (A,) total expected dwell time per state on this
        branch/site (Holmes-Rubin sufficient statistic, summed over
        branches if you accumulate before the call).
    `eta`: per-site rate multiplier (multiplied into the proposal rates).

    Returns N_ghost: (A,) vector of expected ghost counts at each
    destination state. Computed as
        N_ghost[y] = pi[y] * eta * (T_S - sum_x dwell_x * S[x, y])
    where T_S = sum_x dwell_x * S_row_x (off-diagonal sum).
    """
    A = len(pi)
    S_off = S.copy()
    np.fill_diagonal(S_off, 0.0)
    S_row = S_off.sum(axis=1)
    T_S = float(np.sum(dwell_per_state * S_row))
    # sum_x dwell_x * S[x, y] -> dwell_per_state @ S_off  -> (A,)
    Sx_dot_y = dwell_per_state @ S_off
    return eta * pi * (T_S - Sx_dot_y)


def expected_real_counts(transition_count_matrix: np.ndarray) -> np.ndarray:
    """Sum the off-diagonal columns of a (A, A) Holmes-Rubin expected
    transition-count matrix to get per-destination accepted-jump totals.

    `transition_count_matrix[x, y]` = E[# x -> y jumps] (off-diag; diag
    ignored).
    Returns N_real: (A,) vector with N_real[y] = sum_x T_xy."""
    return transition_count_matrix.sum(axis=0) - np.diag(transition_count_matrix)


def dirichlet_posterior(prior_alpha: np.ndarray,
                          N_real: np.ndarray,
                          N_ghost: np.ndarray) -> np.ndarray:
    """Dirichlet posterior parameter vector under the augmentation:
        post_alpha = prior_alpha + N_real + N_ghost.
    """
    return prior_alpha + N_real + N_ghost


def dirichlet_posterior_mean(prior_alpha: np.ndarray,
                               N_real: np.ndarray,
                               N_ghost: np.ndarray) -> np.ndarray:
    """Mean of the Dirichlet posterior on pi (= MAP for VBEM)."""
    post = dirichlet_posterior(prior_alpha, N_real, N_ghost)
    return post / post.sum()


def log_multivariate_beta(alpha: np.ndarray) -> float:
    """log B(alpha) = sum log Γ(alpha_i) - log Γ(sum alpha_i)."""
    return float(np.sum(gammaln(alpha)) - gammaln(np.sum(alpha)))


def dirichlet_log_marginal(prior_alpha: np.ndarray,
                             N_real: np.ndarray,
                             N_ghost: np.ndarray) -> float:
    """Closed-form integral
        log ∫ Multinomial(data | pi) * Dir(pi | prior) dpi
        = log B(prior + N_total) - log B(prior)
    where N_total = N_real + N_ghost. This is the new-class branch
    of the cluster-assignment Gibbs predictive, evaluated in O(A) time."""
    post = prior_alpha + N_real + N_ghost
    return log_multivariate_beta(post) - log_multivariate_beta(prior_alpha)


# --- Iterated EM update for pi^(c) given current (S, eta, dwell, real_counts) ---

def em_pi_update(prior_alpha: np.ndarray, N_real: np.ndarray,
                  S: np.ndarray, dwell_per_state: np.ndarray,
                  eta: float = 1.0, n_iters: int = 5,
                  tol: float = 1e-6) -> np.ndarray:
    """One EM round on pi^(c): repeatedly compute expected ghost counts at
    the current pi estimate and update the Dirichlet posterior. Returns
    the converged posterior mean.

    Returns pi_post: (A,) posterior-mean estimate of pi^(c).
    """
    A = len(prior_alpha)
    pi_curr = prior_alpha / prior_alpha.sum()  # init at prior mean
    for _ in range(n_iters):
        N_ghost = expected_ghost_counts(pi_curr, S, dwell_per_state, eta=eta)
        pi_new = dirichlet_posterior_mean(prior_alpha, N_real, N_ghost)
        if np.abs(pi_new - pi_curr).max() < tol:
            return pi_new
        pi_curr = pi_new
    return pi_curr
