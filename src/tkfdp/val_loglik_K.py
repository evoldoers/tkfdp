"""K-class extension of val_loglik.py.

For each held-out MSA n, given trained H_slices and cp_table, run joint
(partner_s, c_s) Gibbs MCMC starting from random init at the val time
α and α_c. Report mean / lse / best log-likelihood at MCMC samples.

Comparable across K's because the score is the *full* log p(D | z, c, H)
including singleton contributions (LG08 for now), so adding more
classes can only help on Pfam if the additional H slices fit something.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from .lg08 import Q_LG08_J
from .multiclass import (
    class_pair_idx_table,
    log_P_unique_K,
)
from .partition_K import (
    FamilyKState,
    gibbs_sweep_K,
    init_random_K,
    n_pairs_K,
)
from .pfam_data import FamilyCherries


@jax.jit
def _log_P_LG08_jit(unique_t_):
    return jax.vmap(lambda t: jnp.log(jnp.clip(jsl.expm(Q_LG08_J * t), 1e-300, 1.0)))(unique_t_)


def _per_msa_setup(fc: FamilyCherries):
    return dict(family=fc.family, L=fc.L, n_cherries=fc.n_cherries,
                tau=fc.tau, aa_a=fc.aa_a, aa_b=fc.aa_b,
                both_aa=fc.both_aa_mask())


def _gather_pair_K(fd, log_P_K, cp_table, cls, tau_idx, s):
    """Returns (K, K, L) pair-loglik table for column s."""
    aa_a = fd["aa_a"]; aa_b = fd["aa_b"]; both_aa = fd["both_aa"]
    L = fd["L"]; C = fd["n_cherries"]
    K = cp_table.shape[0]
    a_s = aa_a[:, s].astype(np.int64); b_s = aa_b[:, s].astype(np.int64)
    valid_s = both_aa[:, s]
    aa_a_64 = aa_a.astype(np.int64); aa_b_64 = aa_b.astype(np.int64)
    start = a_s[:, None] * 20 + aa_a_64
    end = b_s[:, None] * 20 + aa_b_64
    out = np.zeros((K, K, L), dtype=np.float64)
    cp_idx_arr = cp_table[:, :, None]
    for c in range(C):
        if not valid_s[c]: continue
        v = both_aa[c, :]
        if not v.any(): continue
        si = start[c, v]; ei = end[c, v]
        ti = tau_idx[c]
        gather = log_P_K[cp_idx_arr, ti, si[None, None, :], ei[None, None, :]]
        v_idx = np.flatnonzero(v)
        out[:, :, v_idx] += gather
    return out


def _single_loglik(fd, log_P_LG08, tau_idx):
    aa_a = fd["aa_a"]; aa_b = fd["aa_b"]; both_aa = fd["both_aa"]
    L = fd["L"]; C = fd["n_cherries"]
    out = np.zeros(L, dtype=np.float64)
    for c in range(C):
        v = both_aa[c, :]
        if not v.any(): continue
        a = aa_a[c, v].astype(np.int64); b = aa_b[c, v].astype(np.int64)
        out[v] += log_P_LG08[tau_idx[c]][a, b]
    return out


def _full_log_likelihood_at(fd, tau_idx, state: FamilyKState,
                              log_P_K: np.ndarray, cp_table: np.ndarray,
                              single_loglik: np.ndarray) -> float:
    """Compute full log p(D | z, c, H) at the current state."""
    aa_a = fd["aa_a"]; aa_b = fd["aa_b"]; both_aa = fd["both_aa"]
    L = fd["L"]
    total = 0.0
    paired_cols = set()
    for s in range(L):
        t = int(state.partner[s])
        if t <= s: continue
        valid = both_aa[:, s] & both_aa[:, t]
        if not valid.any(): continue
        c_s = int(state.cls[s]); c_t = int(state.cls[t])
        cp = int(cp_table[c_s, c_t])
        ti = tau_idx[valid]
        a_s = aa_a[valid, s].astype(np.int64); a_t = aa_a[valid, t].astype(np.int64)
        b_s = aa_b[valid, s].astype(np.int64); b_t = aa_b[valid, t].astype(np.int64)
        st = a_s * 20 + a_t; en = b_s * 20 + b_t
        for c, sti, eni in zip(ti, st, en):
            total += float(log_P_K[cp, c, sti, eni])
        paired_cols.add(s); paired_cols.add(t)
    for s in range(L):
        if s not in paired_cols:
            total += float(single_loglik[s])
    return total


@dataclass
class ValLogLikResultK:
    family: str
    score_mean: float
    score_logsumexp: float
    score_best: float
    n_samples: int
    n_pairs_mean: float
    class_balance_mean: list


def val_log_likelihood_K(H_slices: np.ndarray, K: int,
                           val_families: list[FamilyCherries],
                           n_burnin: int = 50, n_samples: int = 30,
                           dp_alpha: float = 1.0, alpha_c: float = 1.0,
                           init_pair_fraction: float = 0.4,
                           seed: int = 0,
                           verbose: bool = False,
                           tsb_log_weights: np.ndarray | None = None) -> tuple[float, list[ValLogLikResultK]]:
    """If `tsb_log_weights` is supplied (shape (K,)), it's used as the class
    log-prior in val Gibbs (matches TSB-trained H). Otherwise the symmetric
    finite-K Dirichlet-Multinomial is used at concentration `alpha_c`."""
    cp_table = class_pair_idx_table(K)
    rng = np.random.default_rng(seed)

    all_t = np.concatenate([fc.tau for fc in val_families])
    unique_t, inv = np.unique(all_t, return_inverse=True)
    log_P_K = np.asarray(log_P_unique_K(jnp.asarray(H_slices), jnp.asarray(unique_t)))
    log_P_LG08 = np.asarray(_log_P_LG08_jit(jnp.asarray(unique_t)))
    log_pair_offset = -2.0 * np.log(dp_alpha)

    cursor = 0
    out = []
    sum_score_mean = 0.0
    for fc in val_families:
        fd = _per_msa_setup(fc)
        tau_idx = inv[cursor: cursor + fc.n_cherries].astype(np.int64)
        cursor += fc.n_cherries

        n_pairs_init = int(fc.L * init_pair_fraction / 2)
        state = init_random_K(fc.family, fc.L, K, n_pairs_init, rng)
        sll = _single_loglik(fd, log_P_LG08, tau_idx)

        def pair_fn(s):
            return _gather_pair_K(fd, log_P_K, cp_table, state.cls, tau_idx, s)

        log_l_samples = []
        n_pairs_samples = []
        cls_balance_samples = []
        for it in range(n_burnin + n_samples):
            gibbs_sweep_K(state, pair_fn, sll, rng,
                            temperature=1.0, log_pair_prior_offset=log_pair_offset,
                            alpha_c=alpha_c,
                            class_log_weights=tsb_log_weights)
            if it >= n_burnin:
                ll = _full_log_likelihood_at(fd, tau_idx, state, log_P_K, cp_table, sll)
                log_l_samples.append(ll)
                n_pairs_samples.append(n_pairs_K(state))
                cls_balance_samples.append(np.bincount(state.cls, minlength=K).tolist())

        log_l = np.array(log_l_samples)
        score_mean = float(log_l.mean())
        score_logsumexp = float(jax.scipy.special.logsumexp(log_l) - np.log(len(log_l)))
        score_best = float(log_l.max())
        sum_score_mean += score_mean
        cb_mean = np.mean(np.asarray(cls_balance_samples), axis=0).tolist()
        if verbose:
            print(f"  {fc.family}: mean={score_mean:.2f}  best={score_best:.2f}  "
                    f"pairs_mean={np.mean(n_pairs_samples):.1f}  cls_mean={[round(x,1) for x in cb_mean]}")
        out.append(ValLogLikResultK(
            family=fc.family,
            score_mean=score_mean, score_logsumexp=score_logsumexp,
            score_best=score_best, n_samples=len(log_l),
            n_pairs_mean=float(np.mean(n_pairs_samples)),
            class_balance_mean=cb_mean,
        ))

    return sum_score_mean, out
