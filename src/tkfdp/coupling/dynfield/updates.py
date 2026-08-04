"""Phase D.1 atom-update layer for the dynamic-latent-field variant.

Standalone helpers (the SVI-loop integration is Phase D.4); each operates
on a `DynamicFieldCouplingModel` plus aggregated sufficient statistics
and returns the parameters to overwrite on `model.dyn_field`.

The update suite mirrors the Potts-side updates in
`tkfdp.potts_dp` / `tkfdp.svi`:

  PER-CLUSTER POSTERIOR (the key new piece of math):
    `cluster_log_posterior_theta_doublet(...)`: for a coupled-column
    cherry observation, log p(theta_P | (X_i, X_j, Y_i, Y_j); t, c_i, c_j)
    up to normalisation.

  SUFFICIENT STATS (accumulated across a corpus):
    `accumulate_doublet_stats(...)`: per-(c, theta, residue) destination
    counts N + per-cluster theta_P count vector r.

  PARAMETER UPDATES:
    `update_pi_field_dirichlet(...)`: Dirichlet-conjugate posterior
    mean update of `pi_field[c, theta, :]` given counts N and a
    Dirichlet base measure prior.
    `update_rho_tsb(...)`: truncated-stick-breaking Beta posterior on
    `tsb_betas` given cluster-theta counts r; refreshes `rho` from
    `tsb_betas`.

APPROXIMATIONS (deliberate; to refine in Phase D.3):

  1. The accumulator uses a HARD-MAP theta_P per cluster (one Gibbs
     sample per pass), not soft-EM over the per-cluster posterior. This
     simplifies the count accumulation at the cost of MC variance.
  2. The "destination" residue per leaf is the OBSERVED leaf residue,
     regardless of which 4-case case actually generated it. Under
     stationarity of Q^(c, theta_P), the leaf marginal IS
     pi_field[c, theta_P] in the no-jump case; in the >=1-jump case the
     leaf is at pi_field[c, theta_endpoint] for some theta_endpoint ~ rho.
     Charging both to (c, theta_P) ignores the post-jump field; this
     introduces a small bias that disappears as `rho_chain → 0` (the
     covariance-decoupled limit) and as the model converges.
  3. Singleton (uncoupled) columns are NOT yet handled by this module --
     the SVI scaffold passes singletons through a different code path
     (per-class HR + secret-destination Dirichlet on pi_class). The
     dynfield singleton update is deferred to Phase D.3; for the smoke
     run we rely on the field-marginal pi_class being held fixed.

These approximations are tracked in docs/dynfield_design.md.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import expm as _expm

from ...lg08 import S_LG08_F81_J as S_LG08_J
from .emission import (per_class_field_Q,
                        per_class_field_P_half,
                        per_class_field_cherry_sigma,
                        field_marginal_pi,
                        field_marginal_joint,
                        field_marginal_joint_per_theta,
                        jump_weights)


A = 20


# ---------------------------------------------------------------------------
# Per-(c, theta) precompute caches (shared by the per-cluster posterior and
# the sufficient-stat accumulators).
# ---------------------------------------------------------------------------

def precompute_per_cluster_caches(t: float, rho: np.ndarray,
                                    pi_field: np.ndarray,
                                    S: Optional[np.ndarray] = None,
                                    eta: float = 1.0,
                                    rho_chain: float = 1.0,
                                    ) -> dict:
    """Precompute the per-(c, theta) tensors used by every per-cluster
    posterior at the given cherry diameter t (Interp 2). Returns a dict::

      {
        'Sigma':    (K_c, L_max, A, A),      # per-(c, theta) cherry joint
        'pi_field': (K_c, L_max, A),         # echoed back for convenience
        'pi_marg':  (K_c, A),                # field-marginal lone-pi
        'J':        (K_c, K_c, A, A),        # field-marginal cluster joint
        'J_prime':  (K_c, K_c, L_max, A, A), # per-theta_P post-jump joint
        'pi_prime': (K_c, L_max, A),         # per-theta_P post-jump singleton
        'beta':     (L_max,),                # state-dep no-jump prob
        'rho':      (L_max,),
        't':        float,
        'rho_chain': float,
      }

    beta(theta_P) = exp(-rho_chain * (1 - rho[theta_P]) * t/2) is the
    Interp 2 state-dependent no-jump probability; J_prime and pi_prime
    are the per-theta_P post-jump field-marginal emissions (convex
    combinations of (J, pi^(c, theta_P)) and (pi_marg, pi^(c, theta_P))
    respectively; see emission.field_marginal_joint_per_theta).
    """
    rho = np.asarray(rho, dtype=np.float64)
    pi_field = np.asarray(pi_field, dtype=np.float64)
    Q_cf = per_class_field_Q(pi_field, S=S)
    P_half = per_class_field_P_half(Q_cf, t=t, eta=eta)
    Sigma = per_class_field_cherry_sigma(P_half, pi_field)
    pi_marg = field_marginal_pi(rho, pi_field)
    J = field_marginal_joint(rho, pi_field)
    beta, w_J, w_pi, _ = jump_weights(rho, t=t, rho_chain=rho_chain)
    J_prime = field_marginal_joint_per_theta(
        rho, pi_field, t=t, rho_chain=rho_chain)
    pi_prime = (np.einsum('t,ca->cta', w_J, pi_marg)
                + np.einsum('t,cta->cta', w_pi, pi_field))
    return {'Sigma': Sigma, 'pi_field': pi_field, 'pi_marg': pi_marg,
            'J': J, 'J_prime': J_prime, 'pi_prime': pi_prime,
            'beta': beta, 'w_J': w_J, 'w_pi': w_pi, 'rho': rho,
            't': float(t), 'rho_chain': float(rho_chain)}


# ---------------------------------------------------------------------------
# Per-cluster theta posterior (the key new piece of math for training).
# ---------------------------------------------------------------------------

def cluster_log_posterior_theta_doublet(caches: dict,
                                          c_i: int, c_j: int,
                                          Xi: int, Xj: int, Yi: int, Yj: int
                                          ) -> np.ndarray:
    """log p(theta_P | obs) up to normalisation, for a cap-2 cluster
    (Interp 2). Shape (L_max,).

    The per-theta_P doublet likelihood (4-case sum *without* the theta_P
    rho weight) under Interp 2 is

      P^(c_i, c_j, theta_P)(X_i, X_j, Y_i, Y_j; t)
        = beta(theta_P)^2          * Sigma[c_i, theta_P, X_i, Y_i]
                                   * Sigma[c_j, theta_P, X_j, Y_j]
        + beta(theta_P)(1-beta(theta_P)) * pi_field[c_i, theta_P, X_i]
                                          * pi_field[c_j, theta_P, X_j]
                                          * J_prime[c_i, c_j, theta_P, Y_i, Y_j]
        + (1-beta(theta_P))*beta(theta_P) * J_prime[c_i, c_j, theta_P, X_i, X_j]
                                          * pi_field[c_i, theta_P, Y_i]
                                          * pi_field[c_j, theta_P, Y_j]
        + (1-beta(theta_P))^2      * J_prime[c_i, c_j, theta_P, X_i, X_j]
                                   * J_prime[c_i, c_j, theta_P, Y_i, Y_j]

    Multiplied by rho[theta_P] and taken in log space gives the
    unnormalised log posterior.
    """
    Sigma = caches['Sigma']; pi_field = caches['pi_field']
    J_prime = caches['J_prime']; rho = caches['rho']
    beta = caches['beta']                                          # (L_max,)
    one_m_beta = 1.0 - beta
    # Per-theta_P terms.
    t1 = Sigma[c_i, :, Xi, Yi] * Sigma[c_j, :, Xj, Yj]              # (L_max,)
    t2 = (pi_field[c_i, :, Xi] * pi_field[c_j, :, Xj]
          * J_prime[c_i, c_j, :, Yi, Yj])
    t3 = (J_prime[c_i, c_j, :, Xi, Xj]
          * pi_field[c_i, :, Yi] * pi_field[c_j, :, Yj])
    t4 = J_prime[c_i, c_j, :, Xi, Xj] * J_prime[c_i, c_j, :, Yi, Yj]
    P_per_theta = (beta * beta * t1
                   + beta * one_m_beta * t2
                   + one_m_beta * beta * t3
                   + one_m_beta * one_m_beta * t4)
    log_post = np.log(np.clip(rho * P_per_theta, 1e-300, None))
    return log_post


def cluster_log_posterior_theta_singlet(caches: dict, c: int,
                                          Xi: int, Yi: int) -> np.ndarray:
    """log p(theta_P | obs) up to normalisation, for a singleton column
    (Interp 2). Returns shape (L_max,)."""
    Sigma = caches['Sigma']; pi_field = caches['pi_field']
    pi_prime = caches['pi_prime']; rho = caches['rho']
    beta = caches['beta']
    one_m_beta = 1.0 - beta
    t1 = Sigma[c, :, Xi, Yi]                                    # (L_max,)
    t2 = pi_field[c, :, Xi] * pi_prime[c, :, Yi]
    t3 = pi_prime[c, :, Xi] * pi_field[c, :, Yi]
    t4 = pi_prime[c, :, Xi] * pi_prime[c, :, Yi]
    P_per_theta = (beta * beta * t1
                   + beta * one_m_beta * t2
                   + one_m_beta * beta * t3
                   + one_m_beta * one_m_beta * t4)
    log_post = np.log(np.clip(rho * P_per_theta, 1e-300, None))
    return log_post


def _sample_categorical(log_p: np.ndarray, rng: np.random.Generator) -> int:
    """Sample from a categorical with unnormalised log-probabilities."""
    log_p = np.asarray(log_p)
    p = np.exp(log_p - log_p.max())
    p /= p.sum()
    return int(rng.choice(p.shape[0], p=p))


# ---------------------------------------------------------------------------
# Soft EM attribution helpers (Phase D.3).
# ---------------------------------------------------------------------------

def attribute_cherry_doublet_soft(caches: dict,
                                    c_i: int, c_j: int,
                                    Xi: int, Xj: int, Yi: int, Yj: int,
                                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """Soft-EM attribution for one cap-2 cherry under Interp 2.

    Computes the per-(theta_P, case) posterior P(theta_P, case | obs),
    and the per-(case) attribution of leaf residues to (c, theta, residue)
    bins using the J' = w_J * J + w_pi * pi(theta_P)(x)pi(theta_P)
    decomposition: yj-side leaves are split between the J-component
    (theta_end ~ rho posterior) and the theta_P-retention component
    (theta_end = theta_P).

    Returns:
      N_inc: (K_c, L_max, A) fractional destination counts.
      r_inc: (L_max,) per-theta_P cluster-count contribution
              (= sum over cases of P(theta_P, case | obs); sums to 1).
      log_lik: float, log p(obs).
    """
    rho = caches['rho']; beta = caches['beta']
    w_J_vec = caches['w_J']; w_pi_vec = caches['w_pi']
    pi_field = caches['pi_field']; Sigma = caches['Sigma']
    J = caches['J']; J_prime = caches['J_prime']
    K_c, L_max, A_ = pi_field.shape
    one_m_beta = 1.0 - beta

    # Per-theta_P 4-case likelihoods (no rho weight yet).
    T1_vec = Sigma[c_i, :, Xi, Yi] * Sigma[c_j, :, Xj, Yj]
    T2_vec = pi_field[c_i, :, Xi] * pi_field[c_j, :, Xj] * J_prime[c_i, c_j, :, Yi, Yj]
    T3_vec = J_prime[c_i, c_j, :, Xi, Xj] * pi_field[c_i, :, Yi] * pi_field[c_j, :, Yj]
    T4_vec = J_prime[c_i, c_j, :, Xi, Xj] * J_prime[c_i, c_j, :, Yi, Yj]
    L_T1 = beta * beta * T1_vec
    L_T2 = beta * one_m_beta * T2_vec
    L_T3 = one_m_beta * beta * T3_vec
    L_T4 = one_m_beta * one_m_beta * T4_vec
    P_T1 = rho * L_T1
    P_T2 = rho * L_T2
    P_T3 = rho * L_T3
    P_T4 = rho * L_T4
    total = float(P_T1.sum() + P_T2.sum() + P_T3.sum() + P_T4.sum())
    if total <= 0:
        return (np.zeros((K_c, L_max, A_), dtype=np.float64),
                np.zeros(L_max, dtype=np.float64),
                -np.inf)
    P_T1n = P_T1 / total
    P_T2n = P_T2 / total
    P_T3n = P_T3 / total
    P_T4n = P_T4 / total
    r_inc = P_T1n + P_T2n + P_T3n + P_T4n
    log_lik = float(np.log(total))

    # Sub-component splits for the yj cases.
    J_val_X = float(J[c_i, c_j, Xi, Xj])
    J_val_Y = float(J[c_i, c_j, Yi, Yj])
    J_prime_X = J_prime[c_i, c_j, :, Xi, Xj]
    J_prime_Y = J_prime[c_i, c_j, :, Yi, Yj]
    pi_pi_X = pi_field[c_i, :, Xi] * pi_field[c_j, :, Xj]
    pi_pi_Y = pi_field[c_i, :, Yi] * pi_field[c_j, :, Yj]

    def _safe_div(num, den):
        return np.where(den > 1e-300, num / np.where(den > 1e-300, den, 1.0), 0.0)

    sub_J_share_X = _safe_div(w_J_vec * J_val_X, J_prime_X)
    sub_pi_share_X = _safe_div(w_pi_vec * pi_pi_X, J_prime_X)
    sub_J_share_Y = _safe_div(w_J_vec * J_val_Y, J_prime_Y)
    sub_pi_share_Y = _safe_div(w_pi_vec * pi_pi_Y, J_prime_Y)

    # Sub-J theta_end posterior, given the cluster's (X_i, X_j) or (Y_i, Y_j).
    pst_X = rho * pi_pi_X
    pst_X = pst_X / max(float(pst_X.sum()), 1e-300)
    pst_Y = rho * pi_pi_Y
    pst_Y = pst_Y / max(float(pst_Y.sum()), 1e-300)

    N_inc = np.zeros((K_c, L_max, A_), dtype=np.float64)

    # T1 (nn, nn): all 4 residues at theta_P.
    N_inc[c_i, :, Xi] += P_T1n
    N_inc[c_j, :, Xj] += P_T1n
    N_inc[c_i, :, Yi] += P_T1n
    N_inc[c_j, :, Yj] += P_T1n

    # T2 (nn, yj): X residues at theta_P; Y residues split.
    N_inc[c_i, :, Xi] += P_T2n
    N_inc[c_j, :, Xj] += P_T2n
    N_inc[c_i, :, Yi] += P_T2n * sub_pi_share_Y
    N_inc[c_j, :, Yj] += P_T2n * sub_pi_share_Y
    sub_J_Y_T2 = float((P_T2n * sub_J_share_Y).sum())
    N_inc[c_i, :, Yi] += sub_J_Y_T2 * pst_Y
    N_inc[c_j, :, Yj] += sub_J_Y_T2 * pst_Y

    # T3 (yj, nn): symmetric to T2.
    N_inc[c_i, :, Yi] += P_T3n
    N_inc[c_j, :, Yj] += P_T3n
    N_inc[c_i, :, Xi] += P_T3n * sub_pi_share_X
    N_inc[c_j, :, Xj] += P_T3n * sub_pi_share_X
    sub_J_X_T3 = float((P_T3n * sub_J_share_X).sum())
    N_inc[c_i, :, Xi] += sub_J_X_T3 * pst_X
    N_inc[c_j, :, Xj] += sub_J_X_T3 * pst_X

    # T4 (yj, yj): both sides split.
    N_inc[c_i, :, Xi] += P_T4n * sub_pi_share_X
    N_inc[c_j, :, Xj] += P_T4n * sub_pi_share_X
    sub_J_X_T4 = float((P_T4n * sub_J_share_X).sum())
    N_inc[c_i, :, Xi] += sub_J_X_T4 * pst_X
    N_inc[c_j, :, Xj] += sub_J_X_T4 * pst_X
    N_inc[c_i, :, Yi] += P_T4n * sub_pi_share_Y
    N_inc[c_j, :, Yj] += P_T4n * sub_pi_share_Y
    sub_J_Y_T4 = float((P_T4n * sub_J_share_Y).sum())
    N_inc[c_i, :, Yi] += sub_J_Y_T4 * pst_Y
    N_inc[c_j, :, Yj] += sub_J_Y_T4 * pst_Y

    return N_inc, r_inc, log_lik


def attribute_cherry_singlet_soft(caches: dict, c: int,
                                    Xi: int, Yi: int,
                                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """Soft-EM attribution for one singleton cherry under Interp 2.

    Same decomposition as the doublet but specialised to one column.
    Returns (N_inc, r_inc, log_lik).
    """
    rho = caches['rho']; beta = caches['beta']
    w_J_vec = caches['w_J']; w_pi_vec = caches['w_pi']
    pi_field = caches['pi_field']; Sigma = caches['Sigma']
    pi_marg = caches['pi_marg']; pi_prime = caches['pi_prime']
    K_c, L_max, A_ = pi_field.shape
    one_m_beta = 1.0 - beta

    T1_vec = Sigma[c, :, Xi, Yi]
    T2_vec = pi_field[c, :, Xi] * pi_prime[c, :, Yi]
    T3_vec = pi_prime[c, :, Xi] * pi_field[c, :, Yi]
    T4_vec = pi_prime[c, :, Xi] * pi_prime[c, :, Yi]
    L_T1 = beta * beta * T1_vec
    L_T2 = beta * one_m_beta * T2_vec
    L_T3 = one_m_beta * beta * T3_vec
    L_T4 = one_m_beta * one_m_beta * T4_vec
    P_T1 = rho * L_T1
    P_T2 = rho * L_T2
    P_T3 = rho * L_T3
    P_T4 = rho * L_T4
    total = float(P_T1.sum() + P_T2.sum() + P_T3.sum() + P_T4.sum())
    if total <= 0:
        return (np.zeros((K_c, L_max, A_), dtype=np.float64),
                np.zeros(L_max, dtype=np.float64),
                -np.inf)
    P_T1n = P_T1 / total; P_T2n = P_T2 / total
    P_T3n = P_T3 / total; P_T4n = P_T4 / total
    r_inc = P_T1n + P_T2n + P_T3n + P_T4n
    log_lik = float(np.log(total))

    pi_marg_X = float(pi_marg[c, Xi])
    pi_marg_Y = float(pi_marg[c, Yi])
    pi_prime_X = pi_prime[c, :, Xi]
    pi_prime_Y = pi_prime[c, :, Yi]
    pi_field_X = pi_field[c, :, Xi]
    pi_field_Y = pi_field[c, :, Yi]

    def _safe_div(num, den):
        return np.where(den > 1e-300, num / np.where(den > 1e-300, den, 1.0), 0.0)

    sub_J_share_X = _safe_div(w_J_vec * pi_marg_X, pi_prime_X)
    sub_pi_share_X = _safe_div(w_pi_vec * pi_field_X, pi_prime_X)
    sub_J_share_Y = _safe_div(w_J_vec * pi_marg_Y, pi_prime_Y)
    sub_pi_share_Y = _safe_div(w_pi_vec * pi_field_Y, pi_prime_Y)

    pst_X = rho * pi_field_X
    pst_X = pst_X / max(float(pst_X.sum()), 1e-300)
    pst_Y = rho * pi_field_Y
    pst_Y = pst_Y / max(float(pst_Y.sum()), 1e-300)

    N_inc = np.zeros((K_c, L_max, A_), dtype=np.float64)

    N_inc[c, :, Xi] += P_T1n
    N_inc[c, :, Yi] += P_T1n

    N_inc[c, :, Xi] += P_T2n
    N_inc[c, :, Yi] += P_T2n * sub_pi_share_Y
    sub_J_Y_T2 = float((P_T2n * sub_J_share_Y).sum())
    N_inc[c, :, Yi] += sub_J_Y_T2 * pst_Y

    N_inc[c, :, Yi] += P_T3n
    N_inc[c, :, Xi] += P_T3n * sub_pi_share_X
    sub_J_X_T3 = float((P_T3n * sub_J_share_X).sum())
    N_inc[c, :, Xi] += sub_J_X_T3 * pst_X

    N_inc[c, :, Xi] += P_T4n * sub_pi_share_X
    sub_J_X_T4 = float((P_T4n * sub_J_share_X).sum())
    N_inc[c, :, Xi] += sub_J_X_T4 * pst_X
    N_inc[c, :, Yi] += P_T4n * sub_pi_share_Y
    sub_J_Y_T4 = float((P_T4n * sub_J_share_Y).sum())
    N_inc[c, :, Yi] += sub_J_Y_T4 * pst_Y

    return N_inc, r_inc, log_lik


def accumulate_doublet_stats_soft(model, cherry_observations: list,
                                    *, t_per_cherry: Optional[np.ndarray] = None,
                                    t: Optional[float] = None,
                                    ) -> dict:
    """Soft-EM analogue of accumulate_doublet_stats.

    Returns {'N': (K_c, L_max, A), 'r': (L_max,), 'n_clust': int,
              'log_lik': float}.
    """
    K_c = model.K_c
    L_max = model.dyn_field.L_max
    if t_per_cherry is None and t is None:
        raise ValueError("specify either t (scalar) or t_per_cherry (array)")
    N = np.zeros((K_c, L_max, A), dtype=np.float64)
    r = np.zeros(L_max, dtype=np.float64)
    n_clust = 0
    log_lik_total = 0.0

    if t is not None:
        caches = precompute_per_cluster_caches(
            t=t, rho=model.dyn_field.rho,
            pi_field=model.dyn_field.pi_field,
            rho_chain=float(model.dyn_field.rho_chain))
        cached_t = float(t)
    else:
        cached_t = None
        caches = None

    for idx, (c_i, c_j, Xi, Xj, Yi, Yj) in enumerate(cherry_observations):
        if t_per_cherry is not None:
            t_here = float(t_per_cherry[idx])
            if cached_t is None or abs(cached_t - t_here) > 1e-12:
                caches = precompute_per_cluster_caches(
                    t=t_here, rho=model.dyn_field.rho,
                    pi_field=model.dyn_field.pi_field,
                    rho_chain=float(model.dyn_field.rho_chain))
                cached_t = t_here
        N_inc, r_inc, log_lik = attribute_cherry_doublet_soft(
            caches, c_i, c_j, Xi, Xj, Yi, Yj)
        N += N_inc
        r += r_inc
        log_lik_total += log_lik
        n_clust += 1
    return {'N': N, 'r': r, 'n_clust': n_clust, 'log_lik': log_lik_total}


def accumulate_singlet_stats_soft(model, singleton_observations: list,
                                    *, t_per_cherry: Optional[np.ndarray] = None,
                                    t: Optional[float] = None,
                                    ) -> dict:
    """Soft-EM analogue of accumulate_singlet_stats."""
    K_c = model.K_c
    L_max = model.dyn_field.L_max
    N = np.zeros((K_c, L_max, A), dtype=np.float64)
    r = np.zeros(L_max, dtype=np.float64)
    n_clust = 0
    log_lik_total = 0.0
    cached_t = None
    caches = None
    if t is not None:
        caches = precompute_per_cluster_caches(
            t=t, rho=model.dyn_field.rho,
            pi_field=model.dyn_field.pi_field,
            rho_chain=float(model.dyn_field.rho_chain))
        cached_t = float(t)
    for idx, (c, Xi, Yi) in enumerate(singleton_observations):
        if t_per_cherry is not None:
            t_here = float(t_per_cherry[idx])
            if cached_t is None or abs(cached_t - t_here) > 1e-12:
                caches = precompute_per_cluster_caches(
                    t=t_here, rho=model.dyn_field.rho,
                    pi_field=model.dyn_field.pi_field,
                    rho_chain=float(model.dyn_field.rho_chain))
                cached_t = t_here
        N_inc, r_inc, log_lik = attribute_cherry_singlet_soft(
            caches, c, Xi, Yi)
        N += N_inc
        r += r_inc
        log_lik_total += log_lik
        n_clust += 1
    return {'N': N, 'r': r, 'n_clust': n_clust, 'log_lik': log_lik_total}


# ---------------------------------------------------------------------------
# Sufficient statistics accumulator (hard MAP / Gibbs; kept for sanity).
# ---------------------------------------------------------------------------

def accumulate_doublet_stats(model, cherry_observations: list,
                               *, t_per_cherry: Optional[np.ndarray] = None,
                               t: Optional[float] = None,
                               rng: Optional[np.random.Generator] = None,
                               sample_theta: bool = True,
                               ) -> dict:
    """Accumulate per-(c, theta, residue) destination counts and
    per-theta cluster counts across a batch of coupled-column cherries.

    Args:
      model: DynamicFieldCouplingModel.
      cherry_observations: list of tuples
        (c_i, c_j, Xi, Xj, Yi, Yj) -- (class pair, four observed
        leaf residues). For singletons, pass via
        `accumulate_singlet_stats` instead.
      t_per_cherry: (n_cherries,) array of branch lengths. Required if
        cherries have different t. If None, `t` must be a scalar.
      t: scalar branch length applied to all cherries (precompute once).
      rng: numpy Generator for sampling theta_P when `sample_theta=True`.
      sample_theta: if True, sample theta_P from the per-cluster
        posterior (Gibbs); if False, take the MAP. Default True.

    Returns dict::

      {
        'N':       (K_c, L_max, A),  # destination counts
        'r':       (L_max,),         # cluster -> theta counts
        'n_clust': int,              # total clusters processed
      }
    """
    if rng is None:
        rng = np.random.default_rng(0)
    K_c = model.K_c
    L_max = model.dyn_field.L_max
    if t_per_cherry is None and t is None:
        raise ValueError("specify either t (scalar) or t_per_cherry (array)")
    N = np.zeros((K_c, L_max, A), dtype=np.float64)
    r = np.zeros(L_max, dtype=np.float64)
    n_clust = 0

    if t is not None:
        caches = precompute_per_cluster_caches(
            t=t, rho=model.dyn_field.rho,
            pi_field=model.dyn_field.pi_field,
            rho_chain=float(model.dyn_field.rho_chain))
        cached_t = float(t)
    else:
        cached_t = None
        caches = None

    for idx, (c_i, c_j, Xi, Xj, Yi, Yj) in enumerate(cherry_observations):
        # Refresh cache if per-cherry t changes.
        if t_per_cherry is not None:
            t_here = float(t_per_cherry[idx])
            if cached_t is None or abs(cached_t - t_here) > 1e-12:
                caches = precompute_per_cluster_caches(
                    t=t_here, rho=model.dyn_field.rho,
                    pi_field=model.dyn_field.pi_field,
                    rho_chain=float(model.dyn_field.rho_chain))
                cached_t = t_here
        log_post = cluster_log_posterior_theta_doublet(
            caches, c_i, c_j, Xi, Xj, Yi, Yj)
        if sample_theta:
            theta_P = _sample_categorical(log_post, rng)
        else:
            theta_P = int(np.argmax(log_post))
        N[c_i, theta_P, Xi] += 1.0
        N[c_i, theta_P, Yi] += 1.0
        N[c_j, theta_P, Xj] += 1.0
        N[c_j, theta_P, Yj] += 1.0
        r[theta_P] += 1.0
        n_clust += 1
    return {'N': N, 'r': r, 'n_clust': n_clust}


def accumulate_singlet_stats(model, singleton_observations: list,
                              *, t_per_cherry: Optional[np.ndarray] = None,
                              t: Optional[float] = None,
                              rng: Optional[np.random.Generator] = None,
                              sample_theta: bool = True,
                              ) -> dict:
    """Like `accumulate_doublet_stats` but for uncoupled (singleton)
    cherries: each observation is (c, Xi, Yi)."""
    if rng is None:
        rng = np.random.default_rng(0)
    K_c = model.K_c
    L_max = model.dyn_field.L_max
    N = np.zeros((K_c, L_max, A), dtype=np.float64)
    r = np.zeros(L_max, dtype=np.float64)
    n_clust = 0
    cached_t = None
    caches = None
    if t is not None:
        caches = precompute_per_cluster_caches(
            t=t, rho=model.dyn_field.rho,
            pi_field=model.dyn_field.pi_field,
            rho_chain=float(model.dyn_field.rho_chain))
        cached_t = float(t)
    for idx, (c, Xi, Yi) in enumerate(singleton_observations):
        if t_per_cherry is not None:
            t_here = float(t_per_cherry[idx])
            if cached_t is None or abs(cached_t - t_here) > 1e-12:
                caches = precompute_per_cluster_caches(
                    t=t_here, rho=model.dyn_field.rho,
                    pi_field=model.dyn_field.pi_field,
                    rho_chain=float(model.dyn_field.rho_chain))
                cached_t = t_here
        log_post = cluster_log_posterior_theta_singlet(caches, c, Xi, Yi)
        if sample_theta:
            theta_P = _sample_categorical(log_post, rng)
        else:
            theta_P = int(np.argmax(log_post))
        N[c, theta_P, Xi] += 1.0
        N[c, theta_P, Yi] += 1.0
        r[theta_P] += 1.0
        n_clust += 1
    return {'N': N, 'r': r, 'n_clust': n_clust}


def merge_stats(*stats: dict) -> dict:
    """Sum N and r across multiple sufficient-stat dicts (e.g., doublet
    + singlet accumulators on the same corpus)."""
    out = None
    for s in stats:
        if out is None:
            out = {'N': s['N'].copy(), 'r': s['r'].copy(),
                   'n_clust': int(s['n_clust'])}
        else:
            out['N'] += s['N']; out['r'] += s['r']
            out['n_clust'] += int(s['n_clust'])
    return out


# ---------------------------------------------------------------------------
# Parameter updates.
# ---------------------------------------------------------------------------

def update_rho_tsb(model, r: np.ndarray, *,
                    alpha_field: Optional[float] = None,
                    mode: str = 'map',
                    rng: Optional[np.random.Generator] = None,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """TSB Beta posterior update of (tsb_betas, rho).

    For i in 0..L_max-2:
      beta_i | n ~ Beta(1 + r[i], alpha_field + sum_{j > i} r[j])

    `mode='map'` returns the Beta mean
        beta_i_hat = (1 + r[i]) / (1 + alpha_field + sum_{j >= i} r[j]).
    `mode='sample'` draws from the Beta posterior (requires rng).

    Returns (tsb_betas_new, rho_new).
    """
    if alpha_field is None:
        alpha_field = float(model.dyn_field.alpha_field)
    L_max = model.dyn_field.L_max
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (L_max,):
        raise ValueError(f"r shape {r.shape} != ({L_max},)")
    # Suffix sums r[j > i].
    suffix = np.zeros(L_max - 1, dtype=np.float64)
    cumul = float(r[L_max - 1])
    for i in range(L_max - 2, -1, -1):
        suffix[i] = cumul
        cumul += float(r[i])
    if mode == 'map':
        a = 1.0 + r[:L_max - 1]
        b = alpha_field + suffix
        tsb_betas = a / (a + b)
    elif mode == 'sample':
        if rng is None:
            rng = np.random.default_rng(0)
        a = 1.0 + r[:L_max - 1]
        b = alpha_field + suffix
        tsb_betas = rng.beta(a, b)
    else:
        raise ValueError(f"mode must be 'map' or 'sample', got {mode!r}")
    # Build rho from tsb_betas.
    rho_new = np.empty(L_max, dtype=np.float64)
    remaining = 1.0
    for i in range(L_max - 1):
        rho_new[i] = remaining * tsb_betas[i]
        remaining *= 1.0 - tsb_betas[i]
    rho_new[L_max - 1] = remaining
    return tsb_betas, rho_new


def apply_updates_inplace(model, *,
                            tsb_betas_new: Optional[np.ndarray] = None,
                            rho_new: Optional[np.ndarray] = None):
    """Overwrite model.dyn_field.rho / tsb_betas in-place. After this,
    downstream emission tensors should be rebuilt (the existing caches
    in DynamicFieldCouplingModel are stateless, so any new call uses
    the fresh state).

    pi_field itself is a derived quantity (pi_archetype[arch_assignment])
    and is refreshed via model.dyn_field.materialise_pi_field() after
    the archetype M-step, not here.
    """
    if tsb_betas_new is not None:
        model.dyn_field.tsb_betas = np.asarray(tsb_betas_new, dtype=np.float64)
    if rho_new is not None:
        model.dyn_field.rho = np.asarray(rho_new, dtype=np.float64)
    # Refresh model.pi_class as the field-marginal stationary.
    if rho_new is not None:
        pi_class_new = np.einsum(
            't,cta->ca', model.dyn_field.rho, model.dyn_field.pi_field)
        model.dyn_field.pi_class = pi_class_new
        model.pi_class = pi_class_new


# ---------------------------------------------------------------------------
# Variable-size cluster soft EM attribution (Phase D.5).
# ---------------------------------------------------------------------------

def attribute_cluster_soft(caches: dict,
                             classes: np.ndarray,
                             X_obs: np.ndarray, Y_obs: np.ndarray,
                             ) -> tuple[np.ndarray, np.ndarray, float]:
    """Soft-EM attribution for one cluster of arbitrary size m >= 1.

    Generalises attribute_cherry_doublet_soft (m=2) and
    attribute_cherry_singlet_soft (m=1). Per the 4-case structure:

      T1 (nn, nn): all 2m residues attributed to theta_P (per-site, per-class).
      T2 (nn, yj): X-side residues at theta_P; Y-side split between
                   theta_P-retention (sub-pi) and post-jump-end-theta
                   (sub-J, with theta_end posterior over X_obs / Y_obs).
      T3 (yj, nn): symmetric.
      T4 (yj, yj): both sides split independently.

    The sub-J theta_end posterior given the cluster's observations is
      P(theta_end | sub-J, X_obs) prop_to rho[theta] * prod_i pi[c_i, theta, X_i]
    (and analogously Y), evaluated as a per-(theta, cluster_obs) vector.
    """
    from .emission import cluster_emission_per_theta

    rho = caches['rho']
    K_c, L_max, A_ = caches['pi_field'].shape
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)
    X_obs = np.asarray(X_obs, dtype=np.int64).reshape(-1)
    Y_obs = np.asarray(Y_obs, dtype=np.int64).reshape(-1)
    m = classes.shape[0]

    P_per_theta_case, info = cluster_emission_per_theta(
        t=caches['t'], rho=rho, pi_field=caches['pi_field'],
        classes=classes, X_obs=X_obs, Y_obs=Y_obs,
        rho_chain=caches['rho_chain'],
        precomputed_Sigma=caches['Sigma'])

    total = float(P_per_theta_case.sum())
    if total <= 0:
        return (np.zeros((K_c, L_max, A_), dtype=np.float64),
                np.zeros(L_max, dtype=np.float64),
                -np.inf)
    P_norm = P_per_theta_case / total
    log_lik = float(np.log(total))
    # Per-theta_P, per-case normalised posteriors.
    P_T1n = P_norm[:, 0]
    P_T2n = P_norm[:, 1]
    P_T3n = P_norm[:, 2]
    P_T4n = P_norm[:, 3]
    # r increment (cluster -> theta_P count).
    r_inc = P_norm.sum(axis=1)

    # Sub-J vs sub-pi share for X and Y sides, per theta_P.
    pi_prod_X = info['pi_prod_X']; pi_prod_Y = info['pi_prod_Y']
    J_X = info['J_X']; J_Y = info['J_Y']
    J_prime_X = info['J_prime_X']; J_prime_Y = info['J_prime_Y']
    w_J = info['w_J']; w_pi = info['w_pi']

    def _safe_div(num, den):
        return np.where(den > 1e-300, num / np.where(den > 1e-300, den, 1.0), 0.0)

    sub_J_share_X = _safe_div(w_J * J_X, J_prime_X)
    sub_pi_share_X = _safe_div(w_pi * pi_prod_X, J_prime_X)
    sub_J_share_Y = _safe_div(w_J * J_Y, J_prime_Y)
    sub_pi_share_Y = _safe_div(w_pi * pi_prod_Y, J_prime_Y)

    # Sub-J theta_end posterior given cluster observations.
    pst_X = rho * pi_prod_X
    pst_X = pst_X / max(float(pst_X.sum()), 1e-300)
    pst_Y = rho * pi_prod_Y
    pst_Y = pst_Y / max(float(pst_Y.sum()), 1e-300)

    N_inc = np.zeros((K_c, L_max, A_), dtype=np.float64)

    # Iterate sites; per-site attribution mirrors the size-2 case.
    # T1 (nn, nn): X_i, Y_i at theta_P (weight P_T1n) for each site.
    for i in range(m):
        c = int(classes[i])
        N_inc[c, :, int(X_obs[i])] += P_T1n
        N_inc[c, :, int(Y_obs[i])] += P_T1n
    # T2 (nn, yj): X at theta_P; Y split.
    sub_J_total_Y_T2 = float((P_T2n * sub_J_share_Y).sum())
    for i in range(m):
        c = int(classes[i])
        N_inc[c, :, int(X_obs[i])] += P_T2n
        N_inc[c, :, int(Y_obs[i])] += P_T2n * sub_pi_share_Y
        N_inc[c, :, int(Y_obs[i])] += sub_J_total_Y_T2 * pst_Y
    # T3 (yj, nn): symmetric.
    sub_J_total_X_T3 = float((P_T3n * sub_J_share_X).sum())
    for i in range(m):
        c = int(classes[i])
        N_inc[c, :, int(Y_obs[i])] += P_T3n
        N_inc[c, :, int(X_obs[i])] += P_T3n * sub_pi_share_X
        N_inc[c, :, int(X_obs[i])] += sub_J_total_X_T3 * pst_X
    # T4 (yj, yj): both split.
    sub_J_total_X_T4 = float((P_T4n * sub_J_share_X).sum())
    sub_J_total_Y_T4 = float((P_T4n * sub_J_share_Y).sum())
    for i in range(m):
        c = int(classes[i])
        N_inc[c, :, int(X_obs[i])] += P_T4n * sub_pi_share_X
        N_inc[c, :, int(X_obs[i])] += sub_J_total_X_T4 * pst_X
        N_inc[c, :, int(Y_obs[i])] += P_T4n * sub_pi_share_Y
        N_inc[c, :, int(Y_obs[i])] += sub_J_total_Y_T4 * pst_Y

    return N_inc, r_inc, log_lik


def accumulate_cluster_stats_soft(model,
                                    cluster_observations: list,
                                    *, t_per_cluster: Optional[np.ndarray] = None,
                                    t: Optional[float] = None,
                                    ) -> dict:
    """Variable-cluster-size analogue of accumulate_doublet_stats_soft.

    Inputs:
      cluster_observations: list of (classes, X_obs, Y_obs) tuples --
        each is a cluster of arbitrary size m. classes, X_obs, Y_obs are
        m-tuples or 1-D arrays of int.
      t_per_cluster: (n_clusters,) or scalar t.

    Returns {'N': (K_c, L_max, A), 'r': (L_max,), 'n_clust': int,
              'log_lik': float}.
    """
    K_c = model.K_c
    L_max = model.dyn_field.L_max
    if t_per_cluster is None and t is None:
        raise ValueError("specify either t (scalar) or t_per_cluster (array)")
    N = np.zeros((K_c, L_max, A), dtype=np.float64)
    r = np.zeros(L_max, dtype=np.float64)
    n_clust = 0
    log_lik_total = 0.0
    # Cache the per-cluster caches by t across the whole corpus. Under the
    # gap-fix cluster_observations produces one tuple per (cluster, cherry)
    # -> ~10-40k tuples per iter with mostly-distinct t values. A single
    # slot cache thrashed on every tuple; keying a dict on quantised t
    # avoids the ~64 expm(20x20) calls per tuple. Quantise to 1e-12 to
    # match the old single-slot tolerance.
    caches_by_t: dict = {}

    def _get_caches(t_val: float):
        key = round(float(t_val), 12)
        c = caches_by_t.get(key)
        if c is None:
            c = precompute_per_cluster_caches(
                t=float(t_val), rho=model.dyn_field.rho,
                pi_field=model.dyn_field.pi_field,
                rho_chain=float(model.dyn_field.rho_chain))
            caches_by_t[key] = c
        return c

    if t is not None:
        caches = _get_caches(t)
    for idx, obs in enumerate(cluster_observations):
        classes, X_obs, Y_obs = obs
        if t_per_cluster is not None:
            caches = _get_caches(float(t_per_cluster[idx]))
        N_inc, r_inc, log_lik = attribute_cluster_soft(
            caches, classes, X_obs, Y_obs)
        N += N_inc
        r += r_inc
        log_lik_total += log_lik
        n_clust += 1
    return {'N': N, 'r': r, 'n_clust': n_clust, 'log_lik': log_lik_total}


def accumulate_cluster_stats_hr(model,
                                  cluster_observations: list,
                                  *, t_per_cluster: Optional[np.ndarray] = None,
                                  t: Optional[float] = None,
                                  S: Optional[np.ndarray] = None,
                                  ) -> dict:
    """Accumulate HR sufficient statistics across the corpus using
    hr_cluster_stats. Requires the archetype variant (pi_archetype and
    arch_assignment on the dyn_field state).

    Returns {'V': (K_a, A), 'U': (K_a, A, A), 'W': (K_a, A),
             'N_theta_sum': float, 'T_sum': float, 'n_clust': int,
             'log_lik': float}.

    V, U, W are the JOINT sufficient stats (product with P_obs); N_theta_sum
    is Sum_q E[N_theta_q | obs] * P_obs_q; T_sum is Sum_q t_q * P_obs_q.
    For the M-step, the P_obs weight cancels out.
    """
    from .hr import hr_cluster_stats
    dyn = model.dyn_field
    if getattr(dyn, 'pi_archetype', None) is None:
        raise ValueError(
            "accumulate_cluster_stats_hr requires archetype variant "
            "(pi_archetype and arch_assignment on dyn_field)")
    if S is None:
        from tkfdp.lg08 import S_LG08
        S = np.asarray(S_LG08, dtype=np.float64)
    if t_per_cluster is None and t is None:
        raise ValueError("specify either t (scalar) or t_per_cluster (array)")

    K_a, A_ = dyn.pi_archetype.shape
    V = np.zeros((K_a, A_), dtype=np.float64)
    U = np.zeros((K_a, A_, A_), dtype=np.float64)
    W = np.zeros((K_a, A_), dtype=np.float64)
    N_theta_sum = 0.0
    T_sum = 0.0
    n_clust = 0
    log_lik_total = 0.0
    for idx, obs in enumerate(cluster_observations):
        classes, X_obs, Y_obs = obs
        t_q = float(t if t_per_cluster is None else t_per_cluster[idx])
        P_obs, V_q, U_q, W_q, Nt_q = hr_cluster_stats(
            dyn.rho, float(dyn.rho_chain), dyn.pi_archetype,
            dyn.arch_assignment, classes, X_obs, Y_obs, t_q, S)
        if P_obs <= 0:
            n_clust += 1
            continue
        # hr_cluster_stats returns JOINT sufficient stats (E[X | obs] * P_obs).
        # EM M-step needs CONDITIONAL sums. Divide by P_obs per cluster.
        V += V_q / float(P_obs)
        U += U_q / float(P_obs)
        W += W_q / float(P_obs)
        N_theta_sum += float(Nt_q) / float(P_obs)
        T_sum += t_q                       # unweighted branch length
        log_lik_total += float(np.log(P_obs))
        n_clust += 1
    return {'V': V, 'U': U, 'W': W,
              'N_theta_sum': N_theta_sum, 'T_sum': T_sum,
              'n_clust': n_clust, 'log_lik': log_lik_total}


# ---------------------------------------------------------------------------
# rho_chain learning via MH random walk with Gamma prior (Phase D.7).
# ---------------------------------------------------------------------------

def _corpus_log_likelihood_at_rho_chain(model, clusters: list,
                                          t_per_cluster: np.ndarray,
                                          rho_chain_val: float) -> float:
    """Sum of log p(cluster_obs | rho_chain_val) over all clusters in
    the corpus. Sigma per unique t is independent of rho_chain so we
    cache it once."""
    from .emission import (per_class_field_Q, per_class_field_P_half,
                              per_class_field_cherry_sigma,
                              cluster_emission_per_theta)
    Q_cf = per_class_field_Q(model.dyn_field.pi_field)
    unique_t = np.unique(np.asarray(t_per_cluster, dtype=np.float64))
    Sigma_per_t = {}
    for t_val in unique_t:
        P_half = per_class_field_P_half(Q_cf, t=float(t_val))
        Sigma_per_t[float(t_val)] = per_class_field_cherry_sigma(
            P_half, model.dyn_field.pi_field)
    ll = 0.0
    for i, (classes, X_obs, Y_obs) in enumerate(clusters):
        t_q = float(t_per_cluster[i])
        P_per_theta_case, _ = cluster_emission_per_theta(
            t=t_q, rho=model.dyn_field.rho,
            pi_field=model.dyn_field.pi_field,
            classes=classes, X_obs=X_obs, Y_obs=Y_obs,
            rho_chain=float(rho_chain_val),
            precomputed_Sigma=Sigma_per_t[t_q])
        total = float(P_per_theta_case.sum())
        ll += float(np.log(max(total, 1e-300)))
    return ll


def update_rho_chain_mh(model, clusters: list, t_per_cluster: np.ndarray,
                          *, prior_a: float = 1.5, prior_b: float = 5.0,
                          n_steps: int = 5, step_size: float = 0.3,
                          rng: 'np.random.Generator | None' = None,
                          ) -> tuple[float, dict]:
    """Metropolis-Hastings random walk on log(rho_chain) with Gamma(a, b)
    prior on rho_chain (in natural space). The Gamma(prior_a, prior_b)
    has mean `prior_a / prior_b`; defaults Gamma(1.5, 5) (mean 0.3, std
    0.24) nudge rho_chain to be slower than the unit substitution rate.

    Target in log space (proposal symmetric in log space; Jacobian
    absorbed into the prior exponent):

      log target(log rho)
        = log p(data | rho) + (prior_a - 1) log(rho) - prior_b * rho
                                    + log(rho)            (Jacobian)
        = log p(data | rho) + prior_a * log(rho) - prior_b * rho + const

    Returns (new_rho_chain, info_dict). info has 'n_steps_accept',
    'final_log_lik', 'log_lik_curve'.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if not clusters:
        return float(model.dyn_field.rho_chain), {
            'n_steps_accept': 0, 'final_log_lik': 0.0, 'log_lik_curve': []}
    rho_curr = float(model.dyn_field.rho_chain)
    log_rho_curr = float(np.log(rho_curr))
    ll_curr = _corpus_log_likelihood_at_rho_chain(
        model, clusters, t_per_cluster, rho_curr)
    log_lik_curve = [ll_curr]
    n_accept = 0
    for _ in range(int(n_steps)):
        log_rho_prop = log_rho_curr + float(rng.normal(0.0, float(step_size)))
        rho_prop = float(np.exp(log_rho_prop))
        ll_prop = _corpus_log_likelihood_at_rho_chain(
            model, clusters, t_per_cluster, rho_prop)
        log_ratio = ((ll_prop - ll_curr)
                      + float(prior_a) * (log_rho_prop - log_rho_curr)
                      - float(prior_b) * (rho_prop - rho_curr))
        if log_ratio > 0 or rng.random() < float(np.exp(log_ratio)):
            rho_curr = rho_prop
            log_rho_curr = log_rho_prop
            ll_curr = ll_prop
            n_accept += 1
        log_lik_curve.append(ll_curr)
    model.dyn_field.rho_chain = rho_curr
    return float(rho_curr), {
        'n_steps_accept': int(n_accept),
        'final_log_lik': float(ll_curr),
        'log_lik_curve': log_lik_curve,
    }
