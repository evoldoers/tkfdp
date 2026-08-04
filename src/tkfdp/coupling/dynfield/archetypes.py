"""Hierarchical archetype extension of the dynamic-latent-field variant.

The flat dynfield model has each `pi_field[c, theta, :]` be a free
Dirichlet-distributed simplex point. This gives no structural incentive
for theta to act as a **coupling mediator** -- the model can happily
choose each per-(c, theta) distribution independently, so a "field
flip" often produces a near-identical emission and no salt-bridge-style
covariation ever emerges.

The hierarchical extension shares a small **archetype vocabulary**
`pi_archetype: (K_a, A)` across all (c, theta) pairs. Each archetype
is a full simplex point drawn from a DP with a Dirichlet base measure.
A per-(c, theta) `arch_assignment[c, theta]` maps the class-and-field
index to one of the archetypes. Then

    pi_field[c, theta, :] = pi_archetype[arch_assignment[c, theta], :]

Semantics:

- A "site class" is a mapping `theta -> archetype_index`. Some classes
  are `(hyd, hyd, hyd, hyd)` (specializer: theta flip is a no-op). Others
  are `(pos, neg, pos, neg)` (mediator: theta flip is a salt-bridge swap).
- Within an archetype the full 20-simplex mass is available, so
  substitution physics stays intact -- the model does **not** collapse
  to corner-only atoms.
- The DP prior on archetypes concentrates on ~5-10 distinct simplex
  points, giving a discrete "biochemical vocabulary" (aromatic,
  positive, negative, hydrophobic-core, ...).

Inference primitives implemented here:
- Conjugate Dirichlet posterior update on `pi_archetype` given
  aggregated per-archetype counts.
- Categorical Gibbs update on `arch_assignment[c, theta]` given per-
  (c, theta) aggregated counts.
- TSB stick update on `rho_arch` (optional; simple mean-field variant
  for now).

Not yet: alpha_arch MH -- alpha_arch is fixed at construction for the
first pass; add later.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def init_archetype_state(K_c: int, L_max: int, A: int, K_a: int,
                          rng: np.random.Generator,
                          alpha_prior: float = 1.0,
                          alpha_arch: float = 1.0,
                          ) -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    """Initialise a hierarchical archetype state.

    Returns (pi_archetype, arch_assignment, rho_arch, tsb_betas_arch) where:
      pi_archetype: (K_a, A) random Dirichlet(alpha_prior) draws.
      arch_assignment: (K_c, L_max) int32, uniformly random assignments.
      rho_arch: (K_a,) TSB weights from tsb_betas_arch.
      tsb_betas_arch: (K_a - 1,) initialised at the Beta(1, alpha_arch)
        prior mean (1 / (1 + alpha_arch)) so rho_arch decays smoothly.
    """
    K_a = int(K_a)
    pi_archetype = rng.dirichlet(np.full(A, alpha_prior), size=K_a)
    arch_assignment = rng.integers(0, K_a, size=(K_c, L_max),
                                     dtype=np.int32)
    tsb_betas_arch = np.full(K_a - 1, 1.0 / (1.0 + alpha_arch),
                              dtype=np.float64)
    rho_arch = _stick_breaking(tsb_betas_arch, K_a)
    return pi_archetype, arch_assignment, rho_arch, tsb_betas_arch


def _stick_breaking(tsb_betas: np.ndarray, K_a: int) -> np.ndarray:
    """Truncated stick-breaking: rho[i] = beta_i * prod_{j<i} (1 - beta_j)
    for i in 0..K_a-2 and rho[K_a-1] = remaining mass."""
    rho = np.empty(K_a, dtype=np.float64)
    remaining = 1.0
    for i in range(K_a - 1):
        rho[i] = remaining * tsb_betas[i]
        remaining *= 1.0 - tsb_betas[i]
    rho[K_a - 1] = remaining
    return rho


def aggregate_counts_by_archetype(N: np.ndarray,
                                    arch_assignment: np.ndarray,
                                    K_a: int) -> np.ndarray:
    """Aggregate per-(c, theta, a) counts into per-archetype counts via
    a HARD assignment. Deprecated in favour of `expected_N_arch` for the
    soft-EM update; kept for the stochastic variant.

    Returns (K_a, A) archetype-level counts where
    N_arch[k, :] = sum over (c, theta) with arch_assignment[c, theta] == k
                    of N[c, theta, :].
    """
    K_c, L_max, A = N.shape
    N_arch = np.zeros((K_a, A), dtype=np.float64)
    flat_N = N.reshape(K_c * L_max, A)
    flat_arch = arch_assignment.reshape(-1)
    for i in range(flat_arch.shape[0]):
        k = int(flat_arch[i])
        N_arch[k] += flat_N[i]
    return N_arch


def expected_N_arch(N: np.ndarray, arch_probs: np.ndarray,
                     ) -> 'tuple[np.ndarray, np.ndarray]':
    """Soft-EM aggregation: given N (K_c, L_max, A) and arch_probs
    (K_c, L_max, K_a) posterior over archetype indices, return

      E[N_arch[k, a]] = sum_{c, theta} arch_probs[c, theta, k] * N[c, theta, a]
      E[n_k]           = sum_{c, theta} arch_probs[c, theta, k]

    Returns (N_arch, n_k) of shapes (K_a, A) and (K_a,).
    """
    K_c, L_max, A = N.shape
    K_a = arch_probs.shape[2]
    N_flat = N.reshape(K_c * L_max, A)               # (K*L, A)
    p_flat = arch_probs.reshape(K_c * L_max, K_a)    # (K*L, K_a)
    N_arch = p_flat.T @ N_flat                        # (K_a, A)
    n_k = p_flat.sum(axis=0)                          # (K_a,)
    return N_arch, n_k


def update_pi_archetype(N_arch: np.ndarray, alpha_prior: float = 1.0,
                         ) -> np.ndarray:
    """Conjugate Dirichlet posterior mean update on pi_archetype.

    Given N_arch: (K_a, A) aggregated counts and a Dirichlet(alpha_prior)
    prior, returns the posterior mean

      pi_archetype[a, :] = (N_arch[a, :] + alpha_prior)
                          / (N_arch[a, :].sum() + A * alpha_prior)

    Handles unused archetypes (N_arch[a] == 0) by falling back to the
    Dirichlet prior mean (uniform over the alphabet).
    """
    K_a, A = N_arch.shape
    row_sum = N_arch.sum(axis=1, keepdims=True)
    return (N_arch + alpha_prior) / (row_sum + A * alpha_prior)


def soft_arch_posterior(N: np.ndarray,
                          pi_archetype: np.ndarray,
                          rho_arch: np.ndarray,
                          ) -> np.ndarray:
    """Compute the full soft posterior P(arch[c, theta] = k | N, ...)
    for every (c, theta).

    Full conditional:
      P(arch[c, theta] = k | N, pi_archetype, rho_arch)
        propto rho_arch[k] * prod_x pi_archetype[k, x]^{N[c, theta, x]}
        =  exp( log rho_arch[k] + sum_x N[c, theta, x] * log pi_archetype[k, x] )

    Returns (K_c, L_max, K_a) probabilities that sum to 1 along the last
    axis. Used by the soft-EM archetype update:
      E[N_arch[k, a]] = sum_{c, theta} P(arch[c, theta] = k) * N[c, theta, a]
      E[n_k]           = sum_{c, theta} P(arch[c, theta] = k)
    """
    K_c, L_max, A = N.shape
    K_a = pi_archetype.shape[0]
    assert pi_archetype.shape == (K_a, A)
    assert rho_arch.shape == (K_a,)

    log_pi = np.log(np.maximum(pi_archetype, 1e-300))         # (K_a, A)
    N_flat = N.reshape(K_c * L_max, A)                         # (K*L, A)
    log_lik = N_flat @ log_pi.T                                # (K*L, K_a)
    log_prior = np.log(np.maximum(rho_arch, 1e-300))[None, :]  # (1, K_a)
    log_p = log_lik + log_prior                                # (K*L, K_a)
    log_p -= log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p)
    p /= p.sum(axis=1, keepdims=True)
    return p.reshape(K_c, L_max, K_a)


def sample_arch_assignment(N: np.ndarray,
                             pi_archetype: np.ndarray,
                             rho_arch: np.ndarray,
                             rng: np.random.Generator,
                             ) -> np.ndarray:
    """Gibbs sample new arch_assignment[c, theta] per (c, theta) via
    Gumbel-max Categorical. Provided for the stochastic variant; the
    default trainer path uses `soft_arch_posterior` + expected-count EM
    instead (see `archetype_em_step`)."""
    K_c, L_max, A = N.shape
    K_a = pi_archetype.shape[0]
    log_pi = np.log(np.maximum(pi_archetype, 1e-300))
    N_flat = N.reshape(K_c * L_max, A)
    log_lik = N_flat @ log_pi.T
    log_prior = np.log(np.maximum(rho_arch, 1e-300))[None, :]
    log_p = log_lik + log_prior
    g = rng.gumbel(size=log_p.shape)
    new_arch = np.argmax(log_p + g, axis=1)
    return new_arch.astype(np.int32).reshape(K_c, L_max)


def swap_arch_step(arch_assignment: np.ndarray,
                     N: np.ndarray,
                     pi_archetype: np.ndarray,
                     rng: np.random.Generator,
                     n_sweeps: int = 1,
                     ) -> 'tuple[np.ndarray, dict]':
    """Permutation-only MH update on `arch_assignment`.

    Move: pick a field state theta, pick two distinct classes (c1, c2),
    propose swapping arch_assignment[c1, theta] <-> arch_assignment[c2,
    theta]. Accept via Metropolis-Hastings. The proposal is symmetric,
    so acceptance = min(1, exp(log ratio)), and rho_arch cancels because
    the multiset of archetypes used at theta is preserved by any swap:

      log ratio = sum_a (N[c1, theta, a] - N[c2, theta, a]) *
                       (log pi[k2, a] - log pi[k1, a])

    where k1 = arch_assignment[c1, theta], k2 = arch_assignment[c2,
    theta] before the swap. One sweep proposes every (theta, c1<c2)
    pair in a random order.

    Returns:
      new_arch_assignment: (K_c, L_max) int32.
      info: dict with 'n_proposed', 'n_accepted', 'accept_rate' per sweep.
    """
    K_c, L_max, A = N.shape
    K_a = pi_archetype.shape[0]
    assert arch_assignment.shape == (K_c, L_max)
    log_pi = np.log(np.maximum(pi_archetype, 1e-300))  # (K_a, A)
    A_new = np.asarray(arch_assignment, dtype=np.int32).copy()

    n_proposed = 0
    n_accepted = 0
    pairs = [(c1, c2) for c1 in range(K_c) for c2 in range(c1 + 1, K_c)]
    for _sw in range(int(n_sweeps)):
        order = list(range(L_max))
        rng.shuffle(order)
        for theta in order:
            perm = list(pairs)
            rng.shuffle(perm)
            for (c1, c2) in perm:
                k1 = int(A_new[c1, theta])
                k2 = int(A_new[c2, theta])
                if k1 == k2:
                    continue
                # log p(swap) - log p(current)
                # = sum_a (N[c1, theta, a] - N[c2, theta, a]) *
                #        (log_pi[k2, a] - log_pi[k1, a])
                diff_N = N[c1, theta, :] - N[c2, theta, :]
                diff_lp = log_pi[k2, :] - log_pi[k1, :]
                log_ratio = float(np.dot(diff_N, diff_lp))
                n_proposed += 1
                if log_ratio >= 0.0 or rng.random() < np.exp(log_ratio):
                    A_new[c1, theta] = k2
                    A_new[c2, theta] = k1
                    n_accepted += 1
    info = {
        'n_proposed': int(n_proposed),
        'n_accepted': int(n_accepted),
        'accept_rate': (float(n_accepted) / n_proposed
                            if n_proposed > 0 else 0.0),
    }
    return A_new, info


def update_rho_arch_tsb(arch_assignment: np.ndarray, K_a: int,
                          alpha_arch: float = 1.0,
                          ) -> 'tuple[np.ndarray, np.ndarray]':
    """TSB Beta posterior mean update of (tsb_betas_arch, rho_arch).

    Matches the existing `update_rho_tsb` convention for the field
    selector: under stick-breaking with Beta(1, alpha_arch) prior, the
    posterior on the i-th stick given assignment counts n_a is

      beta_i | n ~ Beta(1 + n_i, alpha_arch + sum_{j > i} n_j)

    The posterior mean is
      beta_i_hat = (1 + n_i) / (1 + alpha_arch + sum_{j >= i} n_j).
    rho_arch is then reconstructed by stick-breaking from beta.
    """
    counts = np.bincount(arch_assignment.reshape(-1).astype(np.int64),
                          minlength=K_a).astype(np.float64)
    # Suffix sums count[j > i].
    suffix = np.zeros(K_a - 1, dtype=np.float64)
    cumul = float(counts[K_a - 1])
    for i in range(K_a - 2, -1, -1):
        suffix[i] = cumul
        cumul += float(counts[i])
    a = 1.0 + counts[:K_a - 1]
    b = alpha_arch + suffix
    tsb_betas = a / (a + b)
    rho = _stick_breaking(tsb_betas, K_a)
    return tsb_betas, rho


def _update_rho_arch_tsb_expected(n_k: np.ndarray, K_a: int,
                                     alpha_arch: float = 1.0,
                                     ) -> 'tuple[np.ndarray, np.ndarray]':
    """TSB Beta posterior mean update on (tsb_betas_arch, rho_arch) given
    EXPECTED assignment counts n_k[k] = E[# (c, theta) with arch = k]."""
    n_k = np.asarray(n_k, dtype=np.float64)
    suffix = np.zeros(K_a - 1, dtype=np.float64)
    cumul = float(n_k[K_a - 1])
    for i in range(K_a - 2, -1, -1):
        suffix[i] = cumul
        cumul += float(n_k[i])
    a = 1.0 + n_k[:K_a - 1]
    b = alpha_arch + suffix
    tsb_betas = a / (a + b)
    rho = _stick_breaking(tsb_betas, K_a)
    return tsb_betas, rho


def archetype_em_step(N: np.ndarray,
                       pi_archetype: np.ndarray,
                       rho_arch: np.ndarray,
                       *,
                       alpha_prior: float = 1.0,
                       alpha_arch: float = 1.0,
                       ) -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    """Soft-EM archetype update given per-(c, theta, a) expected
    attribution counts N (already marginalised over field selector
    histories by the upstream soft-EM in `attribute_cluster_soft`).

    E step:
      1. arch_probs[c, theta, k] = P(arch[c, theta] = k | N, pi_arch,
         rho_arch)
      2. E[N_arch[k, a]] = sum_{c, theta} arch_probs[c, theta, k] * N[c, theta, a]
      3. E[n_k]           = sum_{c, theta} arch_probs[c, theta, k]

    M step:
      4. pi_archetype[k, :] posterior mean under Dirichlet(alpha_prior)
         given E[N_arch[k, :]].
      5. (tsb_betas_arch, rho_arch) posterior mean under
         Beta(1, alpha_arch) TSB given E[n_k].

    For logging / interpretation we also emit
      arch_assignment_map[c, theta] = argmax_k arch_probs[c, theta, k]

    The M-projected update is guaranteed monotone in the archetype-only
    ELBO given fixed N.

    Returns (pi_archetype_new, arch_assignment_map, rho_arch_new,
    tsb_betas_arch_new).
    """
    K_a = int(pi_archetype.shape[0])
    arch_probs = soft_arch_posterior(N, pi_archetype, rho_arch)
    N_arch_exp, n_k_exp = expected_N_arch(N, arch_probs)
    pi_archetype_new = update_pi_archetype(N_arch_exp, alpha_prior)
    tsb_betas_arch_new, rho_arch_new = _update_rho_arch_tsb_expected(
        n_k_exp, K_a, alpha_arch)
    arch_assignment_map = np.argmax(arch_probs, axis=2).astype(np.int32)
    return (pi_archetype_new, arch_assignment_map, rho_arch_new,
             tsb_betas_arch_new)


def archetype_gibbs_step(N: np.ndarray,
                          pi_archetype: np.ndarray,
                          arch_assignment: np.ndarray,
                          rho_arch: np.ndarray,
                          rng: np.random.Generator,
                          *,
                          alpha_prior: float = 1.0,
                          alpha_arch: float = 1.0,
                          ) -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    """Stochastic Gibbs variant of the archetype step: samples
    `arch_assignment` from its full conditional, aggregates counts by
    the sample, then posterior mean on `pi_archetype` and TSB update on
    `rho_arch`.

    Kept for reference. The trainer path uses `archetype_em_step`, which
    is monotone / does not fluctuate under the stochastic assignment
    draw.
    """
    arch_new = sample_arch_assignment(
        N, pi_archetype, rho_arch, rng)
    K_a = int(pi_archetype.shape[0])
    N_arch = aggregate_counts_by_archetype(N, arch_new, K_a)
    pi_archetype_new = update_pi_archetype(N_arch, alpha_prior)
    tsb_betas_arch_new, rho_arch_new = update_rho_arch_tsb(
        arch_new, K_a, alpha_arch)
    return pi_archetype_new, arch_new, rho_arch_new, tsb_betas_arch_new
