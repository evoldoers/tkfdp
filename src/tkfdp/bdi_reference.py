"""Closed-form TKF91 BDI reference: P_ij(T), E[B], E[D], E[S].

The closed-form expressions are reproduced from gravestone_evaluation.md
Section 2 (themselves derived from ~/tkf-mixdom/tkf/body-tkf91.tex).

Conventions:
- TKF91: birth rate per fragment = lambda; death rate per fragment = mu;
  immigration rate = lambda (so the ancestral "blank" position acts as a
  perpetual birth source).
- For the BDI process used in the gravestone evaluation we work with the
  *non-immortal* count formulation: i fragments at t=0, j at t=T, with
  lambda < mu so the equilibrium is finite.
- alpha, beta, gamma are the standard TKF91 finite-time factors.

We compute P_ij(T) and the score-function-derived expectations
E[B|i,j,T], E[D|i,j,T], E[S|i,j,T] using JAX autodiff so the d log P / d lambda
and d log P / d mu pieces come for free.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def alpha_beta_gamma(lam: jnp.ndarray, mu: jnp.ndarray, T: jnp.ndarray):
    """TKF91 finite-time factors. Assumes lam, mu, T positive and lam != mu.
    Returns (alpha, beta, gamma).

    Definitions (Thorne, Kishino, Felsenstein 1991):
        alpha = exp(-mu * T)
        beta  = lam * (1 - exp((lam - mu)*T)) / (mu - lam * exp((lam - mu)*T))
        gamma = 1 - mu * beta / (lam * (1 - alpha))

    To avoid catastrophic cancellation when lam ~ mu, write
    e_mu = exp(-mu T); e_diff = exp((lam-mu) T); use the closed form directly.
    """
    e_mu = jnp.exp(-mu * T)
    e_diff = jnp.exp((lam - mu) * T)
    alpha = e_mu
    beta = lam * (1 - e_diff) / (mu - lam * e_diff)
    one_minus_alpha = 1.0 - alpha
    gamma = 1.0 - mu * beta / (lam * one_minus_alpha)
    return alpha, beta, gamma


def _comb(n, k):
    """Binomial coefficient via lgamma; works for nonneg integer n,k as
    plain numbers (we'll vectorize externally)."""
    return jnp.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def log_P_ij(i: int, j: int, T: float, lam: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """log P((N(0)=i) -> (N(T)=j); lam, mu) for the TKF91 BDI process.

    Uses the closed-form sum (TKF91 eq. 14 / gravestone_evaluation.md eq.):

      P_ij(T) = sum_{m, n} C(i, m) C(m, n) C(j, i - m + n)
                  * alpha^(i - m) (1 - alpha)^m
                  * beta^(j - i + m - n) (1 - beta)^(i + 1 - m + n)
                  * gamma^n (1 - gamma)^(m - n)

    Index ranges: m runs 0..i, n runs 0..min(m, i + j - m), with the C(j, .)
    binomial vanishing where its lower index is negative or > j.

    Implementation: dense double loop in Python over (m, n) — fine since
    i, j ≤ ~30 in our parameter grid. Returns a JAX scalar so we can grad.
    """
    alpha, beta, gamma_ = alpha_beta_gamma(lam, mu, T)

    one_minus_alpha = 1.0 - alpha
    one_minus_beta = 1.0 - beta
    one_minus_gamma = 1.0 - gamma_

    P = jnp.array(0.0)
    for m in range(0, i + 1):
        for n in range(0, m + 1):
            k = j - i + n  # the C(j, i - m + n) lower index becomes (i - m + n)
                            # but the indexing form usually wants C(j, i - m + n)
            low = i - m + n
            if low < 0 or low > j:
                continue
            term = (
                _comb(i, m) * _comb(m, n) * _comb(j, low)
                * alpha ** (i - m) * one_minus_alpha ** m
                * beta ** (j - i + m - n) * one_minus_beta ** (i + 1 - m + n)
                * gamma_ ** n * one_minus_gamma ** (m - n)
            )
            P = P + term

    return jnp.log(P)


def expected_B_D_S(i: int, j: int, T: float,
                   lam: jnp.ndarray, mu: jnp.ndarray):
    """Closed-form posterior expectations E[B|i,j,T], E[D|i,j,T], E[S|i,j,T].

    Uses score-function identity from gravestone_evaluation.md Section 2:

        E[S]   = (j - i + mu * dlogP/dmu - lam * dlogP/dlam - lam * T) / (lam - mu)
        E[B]   = lam * dlogP/dlam + lam * E[S] + lam * T
        E[D]   = mu  * dlogP/dmu  + mu  * E[S]
        E[B-D] = j - i  (conservation; cross-check)

    We get the log-derivatives by jax.grad on log_P_ij.
    """
    grad_lam = jax.grad(lambda l, m: log_P_ij(i, j, T, l, m), argnums=0)(lam, mu)
    grad_mu = jax.grad(lambda l, m: log_P_ij(i, j, T, l, m), argnums=1)(lam, mu)

    E_S = (j - i + mu * grad_mu - lam * grad_lam - lam * T) / (lam - mu)
    E_B = lam * grad_lam + lam * E_S + lam * T
    E_D = mu * grad_mu + mu * E_S
    return E_B, E_D, E_S
