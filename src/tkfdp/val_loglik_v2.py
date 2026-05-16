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
from .partition_K import FamilyKState, gibbs_sweep_K, init_random_K, n_pairs_K
from .pfam_data import FamilyCherries
from .svi import SVIState, build_log_P_cache_K_atoms


@dataclass
class ValLogLikResultV2:
    family: str
    score_mean: float
    score_logsumexp: float
    score_best: float
    n_samples: int
    n_pairs_mean: float
    class_balance_mean: list


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

        # pair_loglik_fn for gibbs_sweep_K: returns (K, K, L) table
        # of pair-loglik(s, t; (k_s, k_t)) under the cached log_P.
        def make_pair_fn(fd=fd, tau_idx=tau_idx, st=st):
            def pair_fn(s):
                K = K_c; L = fd['L']; C = fd['n_cherries']
                out_arr = np.zeros((K, K, L), dtype=np.float64)
                a_s = fd['aa_a'][:, s].astype(np.int64)
                b_s = fd['aa_b'][:, s].astype(np.int64)
                valid_s = fd['both_aa'][:, s]
                aa_a = fd['aa_a'].astype(np.int64); aa_b = fd['aa_b'].astype(np.int64)
                start = a_s[:, None] * 20 + aa_a
                end = b_s[:, None] * 20 + aa_b
                for c in range(C):
                    if not valid_s[c]: continue
                    v = fd['both_aa'][c, :]
                    if not v.any(): continue
                    si = start[c, v]; ei = end[c, v]
                    ti = tau_idx[c]
                    for k_s in range(K):
                        for k_t in range(K):
                            P = log_P_cache[k_s, k_t, ti]
                            out_arr[k_s, k_t, v] += P[si, ei]
                return out_arr
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
        ))

    return sum_score_mean, out
