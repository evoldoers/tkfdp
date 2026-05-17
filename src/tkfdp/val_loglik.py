"""Validation log-likelihood estimator: marginalise the partition latent
via partition Gibbs MCMC and report a comparison-friendly score.

For each held-out MSA n with cherries C_n, given a (trained) global H:

    log p(D_n | H)  =  log sum_{z_n} p(D_n | z_n, H) p(z_n)

where z_n is the MSA's size-2 partition. We don't compute this exactly —
the partition space is huge — but we want a *comparable* score across
trained H's. We provide three estimators:

1. **mean log-likelihood at MCMC samples** ("ELBO-like"):
       score = (1/K) sum_k log p(D_n | z_k, H)
   where z_k ~ posterior(z_n | D_n, H) via Gibbs MCMC.
   By Jensen's inequality this is a *lower bound* on log p(D_n | H).
   Comparable across H's: if H_1 is a better model, its posterior
   typically concentrates on partitions that score higher under H_1
   than the analogous partitions under H_2.

2. **logsumexp at MCMC samples** ("IWAE-like"):
       score = logsumexp_k log p(D_n | z_k, H) - log K
   This is the importance estimate with the MCMC stationary as proposal.
   It's *biased* (proposal = posterior, so weights are p / posterior =
   p(D) / p(D|z), and the estimator becomes the harmonic mean of
   likelihoods — known unstable). But it rewards H's whose posterior
   has high-likelihood modes.

3. **best partition log-likelihood** ("MAP"):
       score = max_k log p(D_n | z_k, H)
   The MCMC's best partition. Insensitive to chain mixing.

For cross-H comparison the user picks one. We default to (1) (most
defensible).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from .composite import composite_log_likelihood
from .generator import (
    A,
    build_joint_Q,
    joint_stationary,
    log_transition_matrices,
    symmetrize_eigh,
)
from .lg08 import Q_LG08_J
from .partition import (
    FamilyPartitionState,
    edges_of,
    gibbs_sweep,
    init_random_pairs,
)
from .pfam_data import FamilyCherries


# ----------------------------------------------------------------------------
# Per-MSA helpers (same as in exp2_pfam.py — kept here to make this module
# self-contained for evaluation only).
# ----------------------------------------------------------------------------

@jax.jit
def _log_P_unique_jit(H_, unique_t_):
    Q = build_joint_Q(H_)
    pi_j = joint_stationary(H_)
    Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
    return log_transition_matrices(unique_t_, Lambda, U_sym, sqrt_pij)


@jax.jit
def _log_P_LG08_unique_jit(unique_t_):
    def one(t):
        return jnp.log(jnp.clip(jsl.expm(Q_LG08_J * t), 1e-300, 1.0))
    return jax.vmap(one)(unique_t_)


def _per_msa_setup(fc: FamilyCherries):
    """Pre-compute per-MSA aux tensors needed by Gibbs."""
    return dict(
        family=fc.family,
        L=fc.L,
        n_cherries=fc.n_cherries,
        tau=fc.tau,
        aa_a=fc.aa_a,
        aa_b=fc.aa_b,
        both_aa=fc.both_aa_mask(),
    )


def _gather_pair_loglik(family_data, log_P_unique, tau_idx, s):
    aa_a = family_data["aa_a"]; aa_b = family_data["aa_b"]
    both_aa = family_data["both_aa"]
    L = family_data["L"]; C = family_data["n_cherries"]
    a_s = aa_a[:, s].astype(np.int64)
    b_s = aa_b[:, s].astype(np.int64)
    valid_s = both_aa[:, s]

    aa_a_64 = aa_a.astype(np.int64); aa_b_64 = aa_b.astype(np.int64)
    start_idx = a_s[:, None] * 20 + aa_a_64
    end_idx = b_s[:, None] * 20 + aa_b_64
    log_p_per_obs = np.zeros((C, L), dtype=np.float64)
    for c in range(C):
        if not valid_s[c]:
            continue
        v = both_aa[c, :]
        if v.any():
            P = log_P_unique[tau_idx[c]]
            log_p_per_obs[c, v] = P[start_idx[c, v], end_idx[c, v]]
    return log_p_per_obs.sum(axis=0)


def _precompute_singleton_loglik(family_data, log_P_LG08_unique, tau_idx):
    aa_a = family_data["aa_a"]; aa_b = family_data["aa_b"]
    both_aa = family_data["both_aa"]
    L = family_data["L"]; C = family_data["n_cherries"]
    out = np.zeros(L, dtype=np.float64)
    for c in range(C):
        v = both_aa[c, :]
        if not v.any():
            continue
        a = aa_a[c, v].astype(np.int64); b = aa_b[c, v].astype(np.int64)
        out[v] += log_P_LG08_unique[tau_idx[c]][a, b]
    return out


def _composite_log_likelihood_at_partition(family_data, tau_idx, state: FamilyPartitionState,
                                            log_P_unique: np.ndarray,
                                            single_loglik: np.ndarray) -> float:
    """Sum log-likelihood over (cherry, edge) pairs PLUS singleton contributions
    for unpaired columns. Note: gives *full* log-likelihood for this MSA at
    the current partition (not just the H-dependent part)."""
    aa_a = family_data["aa_a"]; aa_b = family_data["aa_b"]
    both_aa = family_data["both_aa"]
    L = family_data["L"]
    total = 0.0
    paired_cols = set()
    # Pair contributions
    for s in range(L):
        t = int(state.partner[s])
        if t <= s:
            continue
        valid = both_aa[:, s] & both_aa[:, t]
        if not valid.any():
            continue
        tau_idx_c = tau_idx[valid]
        a_s = aa_a[valid, s].astype(np.int64); a_t = aa_a[valid, t].astype(np.int64)
        b_s = aa_b[valid, s].astype(np.int64); b_t = aa_b[valid, t].astype(np.int64)
        start = a_s * 20 + a_t; end = b_s * 20 + b_t
        for c, st, en in zip(tau_idx_c, start, end):
            total += float(log_P_unique[c, st, en])
        paired_cols.add(s); paired_cols.add(t)
    # Singleton contributions (columns NOT in any pair)
    for s in range(L):
        if s not in paired_cols:
            total += float(single_loglik[s])
    return total


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

@dataclass
class ValLogLikResult:
    family: str
    score_mean: float        # mean log-likelihood at MCMC samples (lower bound on log p(D|H))
    score_logsumexp: float   # logsumexp - log K (biased IWAE estimate)
    score_best: float        # max log-likelihood (MAP partition LL)
    n_samples: int
    n_pairs_mean: float


def val_log_likelihood(H: np.ndarray,
                        val_families: list[FamilyCherries],
                        n_burnin: int = 50,
                        n_samples: int = 30,
                        dp_alpha: float = 1.0,
                        init_pair_fraction: float = 0.4,
                        init_partitions: dict[str, FamilyPartitionState] | None = None,
                        seed: int = 0,
                        verbose: bool = False) -> tuple[float, list[ValLogLikResult]]:
    """For each val MSA, run partition Gibbs MCMC at the given H starting
    from random-pairs init (or `init_partitions[family]` if provided) and
    return three comparison-friendly scores per family + an aggregate.

    Returns (sum_score_mean_over_families, per_family_results).
    """
    rng = np.random.default_rng(seed)

    # Pool unique t values across val
    all_t = np.concatenate([fc.tau for fc in val_families])
    unique_t, inv = np.unique(all_t, return_inverse=True)
    log_P_unique = np.asarray(_log_P_unique_jit(jnp.asarray(H), jnp.asarray(unique_t)))
    log_P_LG08 = np.asarray(_log_P_LG08_unique_jit(jnp.asarray(unique_t)))
    log_prior_offset = -2.0 * np.log(dp_alpha)

    cursor = 0
    out: list[ValLogLikResult] = []
    sum_score_mean = 0.0
    for fc in val_families:
        fd = _per_msa_setup(fc)
        tau_idx = inv[cursor: cursor + fc.n_cherries].astype(np.int64)
        cursor += fc.n_cherries

        # Initialize
        if init_partitions is not None and fc.family in init_partitions:
            state = init_partitions[fc.family]
        else:
            n_pairs_init = int(fc.L * init_pair_fraction / 2)
            state = init_random_pairs(fc.family, fc.L, n_pairs_init, rng)

        sll = _precompute_singleton_loglik(fd, log_P_LG08, tau_idx)

        def pair_fn(s):
            return _gather_pair_loglik(fd, log_P_unique, tau_idx, s)

        log_l_samples = []
        n_pairs_samples = []
        for it in range(n_burnin + n_samples):
            gibbs_sweep(state, pair_fn, sll, rng,
                          temperature=1.0, log_pair_prior_offset=log_prior_offset)
            if it >= n_burnin:
                ll = _composite_log_likelihood_at_partition(fd, tau_idx, state, log_P_unique, sll)
                log_l_samples.append(ll)
                n_pairs_samples.append(int((state.partner >= 0).sum() // 2))

        log_l = np.array(log_l_samples)
        score_mean = float(log_l.mean())
        score_logsumexp = float(jax.scipy.special.logsumexp(log_l) - np.log(len(log_l)))
        score_best = float(log_l.max())
        sum_score_mean += score_mean

        if verbose:
            print(f"  {fc.family}: mean={score_mean:.2f}  lse={score_logsumexp:.2f}  "
                  f"best={score_best:.2f}  pairs_mean={np.mean(n_pairs_samples):.1f}")
        out.append(ValLogLikResult(
            family=fc.family,
            score_mean=score_mean, score_logsumexp=score_logsumexp,
            score_best=score_best, n_samples=len(log_l),
            n_pairs_mean=float(np.mean(n_pairs_samples)),
        ))

    return sum_score_mean, out
