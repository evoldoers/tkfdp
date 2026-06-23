"""Validation log-likelihood under the post-2026-05-08 reparameterization.

Given a trained `SVIState` (with per-class pi, Potts DP atoms +
assignments) and a list of held-out val FamilyCherries, run partition +
class Gibbs at the trained model on each val MSA and report the
log-likelihood at MCMC samples.

The per-MSA log-likelihood at a (partner, cls) state decomposes as:

  log p(D_n | state, trained params)
    = SUM over pair edges (s, t) in the partition:
        log P_pair[(a_s, a_t), (b_s, b_t)](τ; H_atom_{c_s, c_t}, π_{c_s}, π_{c_t})
    + SUM over singleton columns s:
        log NB(N_acc_s ; T̃_s, a_eta, b_eta)            [eta marginalized]
        - log [unconditional Pi_singleton] (so it's a CONDITIONAL log L,
          not a joint with pi prior).

For a singleton, the NB marginal already integrates eta out, giving the
data-conditional log evidence per site under the Gamma(a, b) prior on
eta and the trained per-class pi. Pair contributions use the trained
H atom for the assigned (c_s, c_t) class-pair.

Three reported scores per family (mirroring the legacy val_loglik):
- score_mean: mean log L across MCMC samples (Jensen lower bound).
- score_logsumexp: logsumexp - log K (biased IWAE-style).
- score_best: max log L (MAP partition LL).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from .eta_site import (hr_per_cherry, negative_binomial_log_marginal)
from .generator import (
    A as A_const,
    build_joint_Q_pair,
    joint_stationary_pair,
    log_transition_matrices,
    symmetrize_eigh,
)
from .lg08 import PI_LG08, S_LG08_F81
from .partition import (
    FamilyPartitionState,
    gibbs_sweep as gibbs_sweep_partition_only,
    init_all_singletons,
    init_random_pairs,
)
from .partition_K import FamilyKState, gibbs_sweep_K, init_random_K, n_pairs_K
from .pfam_data import FamilyCherries
from .svi import SVIState, build_log_P_cache_K_atoms

# Numerical safety floor for log(0) when normalising prior weights.
_LOG_EPS = -1e300


@dataclass
class ValLogLikResultV2:
    family: str
    score_mean: float
    score_logsumexp: float
    score_best: float
    n_samples: int
    n_pairs_mean: float
    class_balance_mean: list
    log_l_samples: list | None = None   # per-post-burnin-sweep LLs (for noise diagnostics)


def _per_msa_setup(fc: FamilyCherries) -> dict:
    return dict(
        family=fc.family, L=fc.L, n_cherries=fc.n_cherries,
        tau=fc.tau, aa_a=fc.aa_a, aa_b=fc.aa_b,
        both_aa=fc.both_aa_mask(),
    )


def _build_singleton_NB_per_class(fd: dict, K_c: int, pi_class: np.ndarray,
                                     S: np.ndarray, a_eta: float, b_eta: float
                                     ) -> np.ndarray:
    """Per-(column, class) NB marginal log-likelihood:
       log NB(N_acc_s | T̃_s, a, b)
    computed with the F81 generator at pi_class[c]. Returns (L, K_c)."""
    L = fd['L']
    aa_a = fd['aa_a']; aa_b = fd['aa_b']; tau = fd['tau']
    both_aa = fd['both_aa']
    out = np.full((L, K_c), -np.inf)
    Qs = []
    for c in range(K_c):
        Q = (S - np.diag(np.diag(S))) * pi_class[c][None, :]
        np.fill_diagonal(Q, -Q.sum(axis=1))
        Qs.append(Q)
    for s in range(L):
        v = both_aa[:, s]
        if not v.any():
            out[s, :] = 0.0; continue
        for c in range(K_c):
            Q = Qs[c]; pi_c = pi_class[c]
            N_acc = 0.0; T_tilde = 0.0
            for c_idx in np.flatnonzero(v):
                a = int(aa_a[c_idx, s]); b = int(aa_b[c_idx, s])
                t = float(tau[c_idx])
                N_c, T_c, _ = hr_per_cherry(a, b, t, Q, pi_c)
                N_acc += N_c; T_tilde += T_c
            out[s, c] = negative_binomial_log_marginal(
                N_acc, T_tilde, a_eta, b_eta
            )
    return out


def _build_log_P_single_cache(state: SVIState,
                                  unique_t: np.ndarray,
                                  S: np.ndarray) -> np.ndarray:
    """Per-class single-site transition log-probability cache.

    Returns (K_c, n_t, A, A) float64 where
        log_P_single[c, ti, a, b] = log P(a -> b | class c, tau=unique_t[ti], F81).

    Used for the partial-overlap fallback inside _precompute_pair_LL:
    cherries where only one of (s, t) is observed contribute the single-
    site evolution log-prob at that column's class, not the joint pair
    transition log-prob.
    """
    import jax.numpy as jnp
    import jax.scipy.linalg as jsl
    K_c = state.K_c
    A = state.A
    pi_class = np.asarray(state.pi_class, dtype=np.float64)
    S_np = np.asarray(S, dtype=np.float64)
    n_t = len(unique_t)
    log_P_single = np.zeros((K_c, n_t, A, A), dtype=np.float64)
    for c in range(K_c):
        Q = (S_np - np.diag(np.diag(S_np))) * pi_class[c][None, :]
        np.fill_diagonal(Q, -Q.sum(axis=1))
        Q_j = jnp.asarray(Q)
        for ti, t in enumerate(unique_t):
            P = np.asarray(jsl.expm(Q_j * float(t)))
            log_P_single[c, ti] = np.log(np.clip(P, 1e-300, 1.0))
    return log_P_single


def _precompute_pair_LL(fd: dict, K_c: int,
                          tau_idx: np.ndarray,
                          log_P_cache: np.ndarray,
                          log_P_single_cache: np.ndarray | None = None
                          ) -> np.ndarray:
    """Per-MSA precompute of pair_LL[s, t, k_s, k_t] = composite-likelihood
    sum over cherries c, where each cherry contributes:

      - log P_joint[k_s, k_t, tau_c, (a_s*20+a_t), (b_s*20+b_t)]  if both s, t valid
      - log P_single[k_s, tau_c, a_s, b_s]                        if only s valid
      - log P_single[k_t, tau_c, a_t, b_t]                        if only t valid
      - 0                                                           if neither valid

    The partial-overlap fallback (single-site contribution for cherries
    where only one column is observed) is on by default whenever
    log_P_single_cache is supplied. Without it the fallback contribution
    is omitted (legacy "shared cherries only" behaviour, retained for
    benchmarking / reproducibility).

    Returns a (L, L, K_c, K_c) float64 array.
    """
    L = fd['L']; C = fd['n_cherries']
    aa_a = fd['aa_a'].astype(np.int64)            # (C, L)
    aa_b = fd['aa_b'].astype(np.int64)
    both_aa = fd['both_aa']

    pair_LL = np.zeros((L, L, K_c, K_c), dtype=np.float64)
    for c in range(C):
        valid = both_aa[c, :]
        if not valid.any():
            continue
        idx_v = np.where(valid)[0]                # (m_v,)
        a = aa_a[c, idx_v]; b = aa_b[c, idx_v]

        # (a) both s, t valid: joint contribution on the (idx_v, idx_v)
        # block.
        start_mat = a[:, None] * 20 + a[None, :]
        end_mat   = b[:, None] * 20 + b[None, :]
        P_slice = log_P_cache[:, :, tau_idx[c]]
        contrib = P_slice[:, :, start_mat, end_mat]
        contrib = np.transpose(contrib, (2, 3, 0, 1))
        pair_LL[np.ix_(idx_v, idx_v)] += contrib

        # (b)/(c) partial-overlap: only s valid (or only t valid). For
        # each c_s in K_c, the contribution to pair_LL[s, t, c_s, c_t]
        # (s in idx_v, t in idx_nv) is log_P_single[c_s, tau_c, a_s, b_s],
        # independent of c_t.
        if log_P_single_cache is None:
            continue
        idx_nv = np.where(~valid)[0]
        if len(idx_nv) == 0:
            continue
        # log_P_single_cache: (K_c, n_t, A, A). For cherry c, gather
        # log_P_single[:, tau_c, a, b] at the m_v valid columns.
        Ps_c = log_P_single_cache[:, tau_idx[c]]   # (K_c, A, A)
        # Gather per-(class, valid-column) entry:
        sing_vals_v = Ps_c[:, a, b]                # (K_c, m_v)
        # Add to pair_LL[idx_v, idx_nv, c_s, c_t]: broadcast c_t (axis 3).
        pair_LL[np.ix_(idx_v, idx_nv)] += sing_vals_v.T[:, None, :, None]
        # And to pair_LL[idx_nv, idx_v, c_s, c_t]: broadcast c_s (axis 2).
        pair_LL[np.ix_(idx_nv, idx_v)] += sing_vals_v.T[None, :, None, :]
    return pair_LL


def _full_log_likelihood_at(state: SVIState, fd: dict, st: FamilyKState,
                              singleton_log_NB: np.ndarray,
                              log_P_pair: np.ndarray,
                              tau_idx: np.ndarray) -> float:
    """Full log p(D | partner, cls, trained params) at the current state.

    Pair contributions: gather from the precomputed log_P_pair cache
    (shape (K_c, K_c, n_t, 400, 400)), summed over cherries with both
    columns valid.
    Singleton contributions: use the per-(column, class) NB marginal
    table for the column's current class assignment.
    """
    K_c = state.K_c
    aa_a = fd['aa_a']; aa_b = fd['aa_b']; both_aa = fd['both_aa']
    L = fd['L']
    total = 0.0
    paired_cols = set()
    for s in range(L):
        t = int(st.partner[s])
        if t <= s: continue
        valid = both_aa[:, s] & both_aa[:, t]
        if not valid.any(): continue
        c_s = int(st.cls[s]); c_t = int(st.cls[t])
        ti = tau_idx[valid]
        a_s = aa_a[valid, s].astype(np.int64); a_t = aa_a[valid, t].astype(np.int64)
        b_s = aa_b[valid, s].astype(np.int64); b_t = aa_b[valid, t].astype(np.int64)
        st_idx = a_s * 20 + a_t; en_idx = b_s * 20 + b_t
        for ci, sti, eni in zip(ti, st_idx, en_idx):
            total += float(log_P_pair[c_s, c_t, ci, sti, eni])
        paired_cols.add(s); paired_cols.add(t)
    # Singletons
    for s in range(L):
        if s not in paired_cols:
            c_s = int(st.cls[s])
            total += float(singleton_log_NB[s, c_s])
    return total


def _marginalize_pair_LL_over_classes(pair_LL: np.ndarray,
                                          log_pi_c: np.ndarray) -> np.ndarray:
    """Class-marginalized pair LL:
        out[s, t] = log sum_{c_s, c_t}
                       pi_c[c_s] * pi_c[c_t] * exp(pair_LL[s, t, c_s, c_t])

    pair_LL: (L, L, K_c, K_c) — cherry-summed conditional pair LL.
        Entries are -inf for (s, t, c_s, c_t) cells where a class-pair
        had a zero-probability transition; scipy.special.logsumexp
        handles those gracefully.
    log_pi_c: (K_c,) — log of trained empirical class prior

    Returns (L, L) float64.
    """
    from scipy.special import logsumexp
    weighted = pair_LL + log_pi_c[None, None, :, None] + log_pi_c[None, None, None, :]
    flat = weighted.reshape(*weighted.shape[:-2], -1)
    return logsumexp(flat, axis=-1)


def _marginalize_sing_NB_over_classes(sing_log_NB: np.ndarray,
                                          log_pi_c: np.ndarray) -> np.ndarray:
    """Class-marginalized per-column NB singleton LL:
        out[s] = log sum_c pi_c[c] * exp(sing_log_NB[s, c])
    sing_log_NB may have -inf entries for columns with no valid cherries.
    """
    from scipy.special import logsumexp
    weighted = sing_log_NB + log_pi_c[None, :]
    return logsumexp(weighted, axis=-1)


def val_log_likelihood_class_marginal(
        state: SVIState,
        val_families: list[FamilyCherries],
        pi_c: np.ndarray,
        n_burnin: int = 50, n_samples: int = 30,
        dp_alpha: float = 10.0,
        init_pair_fraction: float = 0.4,
        seed: int = 0,
        verbose: bool = False
        ) -> tuple[float, list[ValLogLikResultV2]]:
    """Val LL with site classes marginalized analytically.

    At eval time the trained parameters (pi_class, Potts atoms,
    assignments) are frozen, so the column-class assignment c_s and the
    pair class assignment (c_s, c_t) can be summed out before running
    MCMC. The Gibbs chain then samples only the size-{1,2} partition,
    which has a much smaller state space than the joint
    (partition, c_s)_{s=1..L} and mixes correspondingly faster.

    Math (per-MSA, given partition π and trained model):
        L(π) = SUM over singleton columns s of
                 log SUM_c pi_c[c] * exp(sing_log_NB[s, c])
             + SUM over pair (s, t) in π of
                 log SUM_{c_s, c_t} pi_c[c_s] * pi_c[c_t]
                                     * exp(pair_LL[s, t, c_s, c_t])

    Each inner log-sum-exp folds the K_c (or K_c^2) class axis into a
    scalar per (column) / per (pair). The partition Gibbs then runs on
    a single (L,) singleton vector and an (L, L) pair matrix.

    Args:
        pi_c: (K_c,) empirical class prior. Get via
            block_likelihoods.empirical_pi_c_from_checkpoint(...).
    """
    K_c = state.K_c
    rng = np.random.default_rng(seed)
    log_pair_offset = -2.0 * np.log(dp_alpha)
    log_pi_c = np.log(np.clip(np.asarray(pi_c), 1e-300, None))

    # Pool unique tau across val (same as the class-aware variant)
    all_t = np.concatenate([fc.tau for fc in val_families])
    all_t_q = np.round(all_t / 0.01) * 0.01
    unique_t, inv = np.unique(all_t_q, return_inverse=True)
    inv_t_dict = {}
    cursor = 0
    for fc in val_families:
        n = fc.n_cherries
        inv_t_dict[fc.family] = inv[cursor: cursor + n].astype(np.int64)
        cursor += n

    log_P_cache = build_log_P_cache_K_atoms(state, unique_t, S_LG08_F81)
    log_P_single_cache = _build_log_P_single_cache(state, unique_t,
                                                      np.asarray(S_LG08_F81))

    out: list[ValLogLikResultV2] = []
    sum_score_mean = 0.0
    for fc in val_families:
        fd = _per_msa_setup(fc)
        tau_idx = inv_t_dict[fc.family]

        sing_log_NB = _build_singleton_NB_per_class(
            fd, K_c, state.pi_class, np.asarray(S_LG08_F81),
            state.a_eta, state.b_eta
        )
        sing_marg = _marginalize_sing_NB_over_classes(sing_log_NB, log_pi_c)  # (L,)

        # Per-cherry composite-likelihood pair LL: shared cherries
        # contribute joint pair transitions; partial-overlap cherries
        # (only one of s, t valid) contribute the single-site transition
        # at the observed column's class. This is the correct
        # composite-likelihood treatment per the per-cherry decomposition
        # of P_pair(D_st) = product_c [joint if both | single_s if only s |
        # single_t if only t | 1 if neither].
        pair_LL = _precompute_pair_LL(
            fd, K_c, tau_idx, log_P_cache,
            log_P_single_cache=log_P_single_cache,
        )                                                                    # (L, L, K_c, K_c)
        pair_marg = _marginalize_pair_LL_over_classes(pair_LL, log_pi_c)     # (L, L)

        # Partition-only Gibbs (no cls).
        n_pairs_init = int(fc.L * init_pair_fraction / 2)
        if n_pairs_init > 0:
            st = init_random_pairs(fc.family, fc.L, n_pairs_init, rng)
        else:
            st = init_all_singletons(fc.family, fc.L)

        def pair_loglik_fn(s, pair_marg=pair_marg):
            return pair_marg[s]                                                # (L,)

        log_l_samples = []
        n_pairs_samples = []
        for it in range(n_burnin + n_samples):
            gibbs_sweep_partition_only(st, pair_loglik_fn, sing_marg, rng,
                                            temperature=1.0,
                                            log_pair_prior_offset=log_pair_offset)
            if it >= n_burnin:
                # Total marginal LL at current partition: singletons + pairs.
                partner = st.partner
                paired = partner >= 0
                # Singletons contribute sing_marg[s] each.
                # Each pair (s, t) contributes pair_marg[s, t] once
                # (sum over s of pair_marg[s, partner[s]] / 2 since each pair
                # is double-counted; but using s < t is cleaner).
                singleton_idx = np.where(~paired)[0]
                ll = float(sing_marg[singleton_idx].sum())
                pair_pairs = [(s, int(partner[s])) for s in range(len(partner))
                                if partner[s] > s]
                for s, t in pair_pairs:
                    ll += float(pair_marg[s, t])
                log_l_samples.append(ll)
                n_pairs_samples.append(len(pair_pairs))

        log_l = np.array(log_l_samples)
        score_mean = float(log_l.mean())
        score_logsumexp = float(jax.scipy.special.logsumexp(log_l) - np.log(len(log_l)))
        score_best = float(log_l.max())
        sum_score_mean += score_mean
        if verbose:
            print(f"  {fc.family}: mean={score_mean:.2f}  best={score_best:.2f}  "
                    f"pairs_mean={np.mean(n_pairs_samples):.1f}  (class-marginal)")
        out.append(ValLogLikResultV2(
            family=fc.family,
            score_mean=score_mean, score_logsumexp=score_logsumexp,
            score_best=score_best, n_samples=len(log_l),
            n_pairs_mean=float(np.mean(n_pairs_samples)),
            # class_balance: always uniform (classes marginalized analytically)
            class_balance_mean=[1.0 / K_c] * K_c,
            log_l_samples=[float(x) for x in log_l],
        ))
    return sum_score_mean, out


def val_log_likelihood(state: SVIState,
                            val_families: list[FamilyCherries],
                            n_burnin: int = 50, n_samples: int = 30,
                            dp_alpha: float = 10.0, alpha_c: float = 1.0,
                            init_pair_fraction: float = 0.4,
                            seed: int = 0,
                            verbose: bool = False
                            ) -> tuple[float, list[ValLogLikResultV2]]:
    """For each val MSA, run partition + class Gibbs at the trained
    `state` and report log-likelihood scores at the MCMC samples.

    The Gibbs uses the trained pi_class for the singleton-side prior on
    c_s (class assignment) and the trained Potts atoms for the pair
    contributions. Eta is marginalized via the NB marginal in the
    singleton score.
    """
    K_c = state.K_c
    rng = np.random.default_rng(seed)
    log_pair_offset = -2.0 * np.log(dp_alpha)

    # Pool unique tau across val
    all_t = np.concatenate([fc.tau for fc in val_families])
    all_t_q = np.round(all_t / 0.01) * 0.01
    unique_t, inv = np.unique(all_t_q, return_inverse=True)
    inv_t_dict = {}
    cursor = 0
    for fc in val_families:
        n = fc.n_cherries
        inv_t_dict[fc.family] = inv[cursor: cursor + n].astype(np.int64)
        cursor += n

    # Build the (K_c, K_c, n_t, 400, 400) log_P cache from trained state.
    log_P_cache = build_log_P_cache_K_atoms(state, unique_t, S_LG08_F81)

    out: list[ValLogLikResultV2] = []
    sum_score_mean = 0.0
    for fc in val_families:
        fd = _per_msa_setup(fc)
        tau_idx = inv_t_dict[fc.family]

        # Precompute per-(column, class) NB marginal table once per MSA
        sing_log_NB = _build_singleton_NB_per_class(
            fd, K_c, state.pi_class, np.asarray(S_LG08_F81),
            state.a_eta, state.b_eta
        )

        # Initialize state
        n_pairs_init = int(fc.L * init_pair_fraction / 2)
        st = init_random_K(fc.family, fc.L, K_c, n_pairs_init, rng)

        # Precompute pair_LL[s, t, k_s, k_t] once per MSA. At eval time the
        # trained log_P_cache and the residue data are both fixed, so the
        # pair log-likelihood for any (s, t, k_s, k_t) is constant across
        # the entire Gibbs run. Building it once turns each pair_fn(s) call
        # from O(C * K_c^2 * L) into an O(K_c^2 * L) slice + transpose.
        pair_LL = _precompute_pair_LL(fd, K_c, tau_idx, log_P_cache)

        def make_pair_fn(pair_LL=pair_LL):
            def pair_fn(s):
                # pair_LL[s, t, k_s, k_t] -> return (k_s, k_t, t).
                return np.transpose(pair_LL[s], (1, 2, 0))
            return pair_fn

        pair_fn = make_pair_fn()
        # Class-conditional singleton evidence (L, K_c) — gibbs_sweep_K
        # consumes this directly per the post-audit fix.
        sll = sing_log_NB

        log_l_samples = []
        n_pairs_samples = []
        cls_balance_samples = []
        for it in range(n_burnin + n_samples):
            gibbs_sweep_K(st, pair_fn, sll, rng,
                            temperature=1.0,
                            log_pair_prior_offset=log_pair_offset,
                            alpha_c=alpha_c)
            if it >= n_burnin:
                ll = _full_log_likelihood_at(
                    state, fd, st, sing_log_NB, log_P_cache, tau_idx
                )
                log_l_samples.append(ll)
                n_pairs_samples.append(n_pairs_K(st))
                cls_balance_samples.append(np.bincount(st.cls, minlength=K_c).tolist())

        log_l = np.array(log_l_samples)
        score_mean = float(log_l.mean())
        score_logsumexp = float(jax.scipy.special.logsumexp(log_l) - np.log(len(log_l)))
        score_best = float(log_l.max())
        sum_score_mean += score_mean
        cb_mean = np.mean(np.asarray(cls_balance_samples), axis=0).tolist()
        if verbose:
            print(f"  {fc.family}: mean={score_mean:.2f}  best={score_best:.2f}  "
                    f"pairs_mean={np.mean(n_pairs_samples):.1f}  "
                    f"cls_mean={[round(x, 1) for x in cb_mean]}")
        out.append(ValLogLikResultV2(
            family=fc.family,
            score_mean=score_mean, score_logsumexp=score_logsumexp,
            score_best=score_best, n_samples=len(log_l),
            n_pairs_mean=float(np.mean(n_pairs_samples)),
            class_balance_mean=cb_mean,
            log_l_samples=[float(x) for x in log_l],
        ))

    return sum_score_mean, out
