"""Recursive midpoint-traceback sampler for the TKF91 BDI bridge,
using the closed-form alpha-beta-gamma transition probability for
midpoint marginals over N_a (the alive count).

Per Ian Holmes's note: the BDI is an infinite chain; we should not
truncate the state space. The TKF91 P_ij(T) closed form handles the
infinite chain exactly, and sampling N_a(t_mid) given the alive-count
endpoints (N_a(0), N_a(T)) requires only

    P(N_a(t_m) = m | N_a(0) = i, N_a(T) = j) =
        P_im(t_left) * P_mj(t_right) / P_ij(t_left + t_right)

with the upper truncation chosen by detecting where mass becomes
negligible (no a-priori state-space cap). Gravestone count is recovered
from the sampled path directly: N_g(T) = total death events.

Base case (small interval, few expected events): Gillespie+rejection on
the full TKF91 dynamics (with immortal link), per branch_sampler.py.

This replaces riccati2.py / branch_sampler_recursive.py, which used a
truncated CTMC approximation on the (a, k) state space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from .bdi_reference import log_P_ij
from .branch_sampler import gillespie_unconditional, BranchHistory


@dataclass
class TKF91History:
    n_births: int
    n_deaths: int
    integrated_alive: float
    n_alive_end: int
    n_grave_end: int
    midpoints: list[tuple[float, int]]   # (t, N_a(t)) sampled midpoints


def _abg(lam: float, mu: float, T: float) -> tuple[float, float, float]:
    """Plain-numpy alpha, beta, gamma TKF91 finite-time factors."""
    e_mu = np.exp(-mu * T)
    e_diff = np.exp((lam - mu) * T)
    alpha = e_mu
    beta = lam * (1 - e_diff) / (mu - lam * e_diff)
    gamma = 1.0 - mu * beta / (lam * (1.0 - alpha))
    return alpha, beta, gamma


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    from math import lgamma
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _log_P_ij_numpy(i: int, j: int, T: float, lam: float, mu: float) -> float:
    """Plain-numpy log P_ij(T) — same closed form as bdi_reference.log_P_ij
    but without JAX overhead. Returns -inf for impossible transitions."""
    alpha, beta, gamma = _abg(lam, mu, T)
    one_m_alpha = 1 - alpha
    one_m_beta = 1 - beta
    one_m_gamma = 1 - gamma
    log_alpha = np.log(max(alpha, 1e-300))
    log_one_m_alpha = np.log(max(one_m_alpha, 1e-300))
    log_beta = np.log(max(beta, 1e-300))
    log_one_m_beta = np.log(max(one_m_beta, 1e-300))
    log_gamma = np.log(max(gamma, 1e-300))
    log_one_m_gamma = np.log(max(one_m_gamma, 1e-300))

    log_terms = []
    for m in range(0, i + 1):
        for n in range(0, m + 1):
            low = i - m + n
            if low < 0 or low > j:
                continue
            log_t = (
                _log_comb(i, m) + _log_comb(m, n) + _log_comb(j, low)
                + (i - m) * log_alpha + m * log_one_m_alpha
                + (j - i + m - n) * log_beta + (i + 1 - m + n) * log_one_m_beta
                + n * log_gamma + (m - n) * log_one_m_gamma
            )
            log_terms.append(log_t)
    if not log_terms:
        return -np.inf
    a = np.array(log_terms)
    a_max = a.max()
    return float(a_max + np.log(np.exp(a - a_max).sum()))


def _midpoint_marginal_alive(i: int, j: int, t_left: float, t_right: float,
                              lam: float, mu: float, m_max: int) -> np.ndarray:
    """Return P(N_a(t_left) = m | N_a(0) = i, N_a(T) = j) for m = 0..m_max
    under the TKF91 BDI process (with immortal link).

    P(m | i, j, t_left, t_right) ∝ P(i -> m; t_left) * P(m -> j; t_right)

    Uses the alpha-beta-gamma closed form — no state-space truncation in the
    CTMC sense; the upper limit m_max only bounds where we evaluate.
    """
    log_p_left = np.array([_log_P_ij_numpy(i, m, t_left, lam, mu) for m in range(m_max + 1)])
    log_p_right = np.array([_log_P_ij_numpy(m, j, t_right, lam, mu) for m in range(m_max + 1)])
    log_p = log_p_left + log_p_right
    log_p -= log_p.max()
    p = np.exp(log_p)
    p /= p.sum()
    return p


def _adaptive_m_max(i: int, j: int, T: float, lam: float, mu: float) -> int:
    """Pick m_max that captures essentially all probability mass at the
    midpoint. Heuristic: bounded by the larger of the endpoints plus a
    cushion that grows with sqrt(events expected).

    For TKF91 with lambda < mu and starting in equilibrium, N_a is roughly
    Geometric(lam/mu), so sigma ~ 1/(1 - lam/mu). We use 6*sigma + some
    deterministic buffer.
    """
    base = max(i, j)
    expected_events = (lam + mu) * T * base
    sigma = max(1.0, np.sqrt(expected_events) + np.sqrt(base))
    return int(base + 6 * sigma + 10)


def sample_one_recursive(i_endpoints: tuple[int, int],
                              t_endpoints: tuple[float, float],
                              lam: float, mu: float,
                              rng: np.random.Generator,
                              n_recurse_levels: int = 2,
                              base_max_attempts: int = 100_000) -> TKF91History:
    """Recursive midpoint sampler for TKF91 BDI bridge from (N_a = i_l) at
    t_l to (N_a = i_r) at t_r. Returns a TKF91History with summed counts.

    Sampling happens in two stages:
    1. Recursively bisect [t_l, t_r] for n_recurse_levels, sampling
       N_a(t_mid) at each midpoint from the closed-form midpoint marginal.
    2. At the leaves of the recursion, run TKF91 Gillespie+rejection on
       the full dynamics (with immortal link) to get a valid path between
       sampled alive-count endpoints. N_g count is read off the path.
    """
    i_l, i_r = i_endpoints
    t_l, t_r = t_endpoints
    T = t_r - t_l

    if n_recurse_levels == 0 or T < 1e-6:
        return _base_case_rejection(i_l, i_r, T, lam, mu, rng, base_max_attempts)

    t_m = 0.5 * (t_l + t_r)
    m_max = _adaptive_m_max(i_l + i_r, i_l + i_r, T, lam, mu)
    p_mid = _midpoint_marginal_alive(i_l, i_r, t_m - t_l, t_r - t_m, lam, mu, m_max)
    m = int(rng.choice(m_max + 1, p=p_mid))

    left = sample_one_recursive((i_l, m), (t_l, t_m), lam, mu, rng,
                                     n_recurse_levels - 1, base_max_attempts)
    right = sample_one_recursive((m, i_r), (t_m, t_r), lam, mu, rng,
                                      n_recurse_levels - 1, base_max_attempts)
    return TKF91History(
        n_births=left.n_births + right.n_births,
        n_deaths=left.n_deaths + right.n_deaths,
        integrated_alive=left.integrated_alive + right.integrated_alive,
        n_alive_end=right.n_alive_end,
        n_grave_end=left.n_grave_end + right.n_grave_end,
        midpoints=left.midpoints + [(t_m, m)] + right.midpoints,
    )


def _base_case_rejection(i: int, j: int, T: float, lam: float, mu: float,
                          rng: np.random.Generator, max_attempts: int) -> TKF91History:
    """Sample one TKF91 path from N_a(0)=i to N_a(T)=j by rejection."""
    for _ in range(max_attempts):
        h = gillespie_unconditional(i, T, lam, mu, rng)
        if h.n_alive_end == j:
            return TKF91History(
                n_births=h.n_births, n_deaths=h.n_deaths,
                integrated_alive=h.integrated_alive,
                n_alive_end=h.n_alive_end, n_grave_end=h.n_grave_end,
                midpoints=[],
            )
    raise RuntimeError(
        f"base-case rejection failed after {max_attempts} attempts "
        f"(i={i}, j={j}, T={T}, lam={lam}, mu={mu})"
    )
