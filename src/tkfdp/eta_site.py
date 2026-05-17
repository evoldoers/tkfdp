"""Per-site rate multiplier eta_s ~ Gamma(a_eta, b_eta) with closed-form
Gamma--Poisson conjugacy on the F81 substitution likelihood.

Per main.tex \S2 (F81 reparam) and \S7.4 / 7.5 (corrected by the user
on 2026-05-08 to use the pi-weighted dwell-time integral, not the
S-clock integral): the F81 path likelihood factors as

   L(path | eta_s, pi, S) = eta_s^{N_accepted_s} * exp(-eta_s * T̃_s) * (pi-and-S factors).

This is Gamma-conjugate on eta_s directly: with Gamma(a_eta, b_eta)
prior the posterior is the closed-form

   eta_s | path ~ Gamma(a_eta + N_accepted_s, b_eta + T̃_s),

and the marginal of N_accepted given T̃_s and the prior is the
closed-form Negative-Binomial:

   log p(N | T̃, a, b)
     = lgamma(a + N) - lgamma(a) - lgamma(N + 1)
       + a log b + N log T̃ - (a + N) log(b + T̃).

No uniformization, no quadrature, no silent-failure sampling needed
for eta_s. The per-site sufficient statistics from Holmes--Rubin are:

  N_accepted_s = sum_b sum_{x != y} N_xy^{(b, s)}    (substitution count)
  T̃_s         = sum_b sum_x T_x^{(b, s)} * sum_{y != x} S[x, y] * pi[y]
              = sum_b sum_x T_x^{(b, s)} * (-Q[x, x])
                under the F81 form Q[x, y] = S[x, y] pi[y] (off-diag).

The secret-destination augmentation (`secret_destination.py`) is needed
for the pi^(c) Dirichlet posterior, NOT for eta_s.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

from .lg08 import PI_LG08, Q_LG08, S_LG08_F81


# --- Closed-form Gamma-Poisson posterior + Negative-Binomial marginal ------

def posterior_eta_gamma(N_acc: float, T_tilde: float,
                          a_eta: float, b_eta: float) -> tuple[float, float]:
    """Closed-form Gamma posterior on eta_s:
       Gamma(a_eta + N_accepted, b_eta + T̃_s)."""
    return float(a_eta + N_acc), float(b_eta + T_tilde)


def posterior_eta_mean(N_acc: float, T_tilde: float,
                        a_eta: float, b_eta: float) -> float:
    """Posterior mean (a + N) / (b + T̃)."""
    return (a_eta + N_acc) / (b_eta + T_tilde)


def negative_binomial_log_marginal(N_acc: float, T_tilde: float,
                                     a_eta: float, b_eta: float) -> float:
    """Closed-form per-site rate marginal log p(N_acc | T̃, a, b):

       lgamma(a + N) - lgamma(a) - lgamma(N + 1)
       + a log b + N log T̃ - (a + N) log(b + T̃).

    N_acc may be fractional (Holmes--Rubin conditional expectation),
    so gammaln of a non-integer is used."""
    return float(gammaln(a_eta + N_acc) - gammaln(a_eta) - gammaln(N_acc + 1)
                  + a_eta * np.log(b_eta)
                  + N_acc * np.log(max(T_tilde, 1e-300))
                  - (a_eta + N_acc) * np.log(b_eta + T_tilde))


# --- Holmes-Rubin sufficient statistics per cherry -------------------------

def _eigh_F81(Q: np.ndarray, pi: np.ndarray):
    """Eigendecompose the symmetrized F81 generator
       Q_sym = D^{1/2} Q D^{-1/2}, D = diag(pi)."""
    sqrt = np.sqrt(pi); inv = 1.0 / sqrt
    Q_sym = sqrt[:, None] * Q * inv[None, :]
    Q_sym = 0.5 * (Q_sym + Q_sym.T)
    Lambda, U = np.linalg.eigh(Q_sym)
    return Lambda, U, sqrt


def hr_per_cherry(aa_a: int, aa_b: int, tau: float,
                   Q: np.ndarray, pi: np.ndarray
                   ) -> tuple[float, float, np.ndarray]:
    """Holmes--Rubin sufficient statistics for a single cherry.

    Returns:
      N_acc       = E[# substitutions in [0, tau] | a, b, tau, Q]
                  = sum_x dwell_x * (-Q[x, x]) + tau * (Q P)[a, b] / P[a, b]
                  (correct E[N_off_diag] formula)
      T̃           = sum_x dwell_x * (-Q[x, x])
                  = pi-weighted dwell-time integral under F81
      dwell_x     = E[T_x | a, b, tau, Q]    (A,) array

    The two derivations:
      E[T_x] = (sqrt(pi[b]) / sqrt(pi[a])) sum_{k, l} U[a, k] U[x, k] U[x, l] U[b, l] I_{kl}(tau) / P[a, b]
      where I_{kl}(tau) = (exp(L_k tau) - exp(L_l tau)) / (L_k - L_l) for k != l,
            I_{kk}(tau) = tau * exp(L_k tau).
      P[a, b](tau) = (1/sqrt(pi[a])) sum_k U[a, k] exp(L_k tau) U[b, k] sqrt(pi[b]).

    The substitution count: N_acc = E[N_off_diag] integrated over the
    coupled bridge:
      E[N_off_diag] = sum_{x != y} Q[x, y] * J(x, y; a, b, tau) / P[a, b]
                    = (Q is reversible)  sum_x dwell_x * (-Q[x, x])
                                         + tau * (Q P(tau))[a, b] / P[a, b]
    where the second term is the off-diagonal correction
      tau * (Q P)[a, b] / P[a, b] = tau * sum_k U[a, k] L_k exp(L_k tau) U[b, k] sqrt(pi[b]) / (sqrt(pi[a]) P[a, b]).
    """
    Lambda, U, sqrt = _eigh_F81(Q, pi)
    inv = 1.0 / sqrt
    expL = np.exp(Lambda * tau)
    P_at = float(inv[aa_a] * (U[aa_a] * expL @ U[aa_b]) * sqrt[aa_b])
    if P_at < 1e-300:
        return 0.0, 0.0, np.zeros(Q.shape[0])

    # Eigenvalue-pair integrals for dwell formula
    expL_k = np.exp(Lambda * tau)[:, None]
    expL_l = np.exp(Lambda * tau)[None, :]
    diff = Lambda[:, None] - Lambda[None, :]
    safe = np.where(diff == 0, 1.0, diff)
    off = (expL_k - expL_l) / safe
    same = np.eye(Lambda.size, dtype=bool)
    I_kl = np.where(same, tau * expL_k, off)

    Ua = U[aa_a]; Ub = U[aa_b]
    M_kl = (Ua[:, None] * Ub[None, :]) * I_kl
    K_x = np.einsum('xk,xl,kl->x', U, U, M_kl)
    K_x = K_x * (sqrt[aa_b] / sqrt[aa_a])
    E_T_x = K_x / P_at

    # T̃ = pi-weighted dwell-time integral = sum_x dwell_x * (-Q[x, x])
    T_tilde = float(np.sum(E_T_x * (-np.diag(Q))))

    # N_accepted = T̃ + tau * (Q P)[a, b] / P[a, b]
    QP_ab = float(inv[aa_a] * (U[aa_a] * (Lambda * np.exp(Lambda * tau))) @ U[aa_b] * sqrt[aa_b])
    N_acc = T_tilde + tau * QP_ab / P_at
    return N_acc, T_tilde, E_T_x


def _hr_per_cherry_at_eigh(Lambda, U, sqrt_pi, a, b, tau):
    """JAX-traceable version of hr_per_cherry given pre-computed eigh.
    Returns (N_acc, T_tilde, dwell). For vmap/JIT.
    """
    import jax.numpy as jnp
    inv = 1.0 / sqrt_pi
    expL = jnp.exp(Lambda * tau)
    Ua = U[a]; Ub = U[b]
    P_at = inv[a] * jnp.sum(Ua * expL * Ub) * sqrt_pi[b]
    P_at = jnp.clip(P_at, 1e-300, None)
    expL_k = expL[:, None]; expL_l = expL[None, :]
    diff = Lambda[:, None] - Lambda[None, :]
    safe = jnp.where(jnp.abs(diff) < 1e-12, 1.0, diff)
    off = (expL_k - expL_l) / safe
    same = jnp.eye(Lambda.shape[0], dtype=bool)
    I_kl = jnp.where(same, tau * expL_k, off)
    M_kl = (Ua[:, None] * Ub[None, :]) * I_kl
    K_x = jnp.einsum('xk,xl,kl->x', U, U, M_kl)
    K_x = K_x * (sqrt_pi[b] / sqrt_pi[a])
    E_T_x = K_x / P_at
    # T_tilde uses -Q diagonal which we don't have here; reconstruct as
    # sum_x dwell_x * sum_{y != x} S_off[x, y] * pi[y] (== -Q[x, x]).
    # But we can pass diag(-Q) externally. Simpler: caller passes neg_diag_Q.
    # For now compute: T_tilde uses Lambda eigh isn't enough; need Q.
    # Caller must pass neg_diag_Q separately.
    return P_at, E_T_x, expL


def hr_batch_jax(Q, pi, neg_diag_Q, a_arr, b_arr, tau_arr):
    """JAX-vmap'd Holmes-Rubin sufficient stats over a batch of (a, b, tau).
    Q, pi, neg_diag_Q are SHARED across the batch (per-class).
    Returns (N_arr, T_tilde_arr, dwell_arr) with shapes (B,), (B,), (B, A).

    Eigh is computed ONCE; per-cherry ops are vmap-batched.
    """
    import jax
    import jax.numpy as jnp

    sqrt = jnp.sqrt(pi)
    inv = 1.0 / jnp.clip(sqrt, 1e-30, None)
    Q_sym = sqrt[:, None] * Q * inv[None, :]
    Q_sym = 0.5 * (Q_sym + Q_sym.T)
    Lambda, U = jnp.linalg.eigh(Q_sym)

    def per_cherry(a, b, tau):
        expL = jnp.exp(Lambda * tau)
        Ua = U[a]; Ub = U[b]
        P_at = inv[a] * jnp.sum(Ua * expL * Ub) * sqrt[b]
        P_at = jnp.clip(P_at, 1e-300, None)
        expL_k = expL[:, None]; expL_l = expL[None, :]
        diff = Lambda[:, None] - Lambda[None, :]
        safe = jnp.where(jnp.abs(diff) < 1e-12, 1.0, diff)
        off = (expL_k - expL_l) / safe
        same = jnp.eye(Lambda.shape[0], dtype=bool)
        I_kl = jnp.where(same, tau * expL_k, off)
        M_kl = (Ua[:, None] * Ub[None, :]) * I_kl
        K_x = jnp.einsum('xk,xl,kl->x', U, U, M_kl)
        K_x = K_x * (sqrt[b] / sqrt[a])
        E_T_x = K_x / P_at
        T_tilde = jnp.sum(E_T_x * neg_diag_Q)
        QP_ab = inv[a] * jnp.sum(Ua * (Lambda * expL) * Ub) * sqrt[b]
        N_acc = T_tilde + tau * QP_ab / P_at
        return N_acc, T_tilde, E_T_x

    return jax.vmap(per_cherry)(a_arr, b_arr, tau_arr)


# JIT-cache the batch routine: shape signature is (B, A) — recompiled per
# unique batch size. Caller can pad batches to a common B_max for cache hits.
import jax as _jax_local
hr_batch_jit = _jax_local.jit(hr_batch_jax)


def per_column_sufficient_stats(aa_a: np.ndarray, aa_b: np.ndarray,
                                  tau: np.ndarray, both_aa: np.ndarray,
                                  Q: np.ndarray | None = None,
                                  pi: np.ndarray | None = None
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate (N_accepted_s, T̃_s) per column from cherry observations
    via the Holmes--Rubin formulas.

    Q and pi default to LG08 (cluster-1 case); for per-class profiles
    pass the per-class F81 Q (built from S * pi[None, :] off-diag, with
    diag set to row-sum-zero) and pi.
    Returns (N_acc_per_col, T_tilde_per_col), both (L,) arrays.
    """
    if Q is None: Q = np.asarray(Q_LG08)
    if pi is None: pi = np.asarray(PI_LG08)
    L = aa_a.shape[1]
    N_acc = np.zeros(L); T_tilde = np.zeros(L)
    for s in range(L):
        v = both_aa[:, s]
        if not v.any():
            continue
        for c in np.flatnonzero(v):
            a = int(aa_a[c, s]); b = int(aa_b[c, s])
            t = float(tau[c])
            n_c, t_c, _ = hr_per_cherry(a, b, t, Q, pi)
            N_acc[s] += n_c; T_tilde[s] += t_c
    return N_acc, T_tilde


def per_column_log_marginal(aa_a: np.ndarray, aa_b: np.ndarray,
                              tau: np.ndarray, both_aa: np.ndarray,
                              a_eta: float = 2.0, b_eta: float = 2.0,
                              Q: np.ndarray | None = None,
                              pi: np.ndarray | None = None
                              ) -> np.ndarray:
    """Per-column closed-form Negative-Binomial marginal log-likelihood
    log p(N_acc_s | T̃_s, a_eta, b_eta).
    """
    N_acc, T_tilde = per_column_sufficient_stats(aa_a, aa_b, tau, both_aa, Q=Q, pi=pi)
    return np.array([negative_binomial_log_marginal(N_acc[s], T_tilde[s], a_eta, b_eta)
                      for s in range(len(N_acc))])


def per_column_eta_post_mean(aa_a: np.ndarray, aa_b: np.ndarray,
                                tau: np.ndarray, both_aa: np.ndarray,
                                a_eta: float = 2.0, b_eta: float = 2.0,
                                Q: np.ndarray | None = None,
                                pi: np.ndarray | None = None
                                ) -> np.ndarray:
    """Posterior-mean point estimate per column: (a + N) / (b + T̃)."""
    N_acc, T_tilde = per_column_sufficient_stats(aa_a, aa_b, tau, both_aa, Q=Q, pi=pi)
    return (a_eta + N_acc) / (b_eta + T_tilde)
