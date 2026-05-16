"""Per-MSA partition MCMC: matchings on alignment columns with size-2 cap.

State per family: int array `partner` of shape (L,) where partner[s] = -1
means "singleton" and partner[s] = t means "paired with t". Symmetric:
partner[t] = s.

Gibbs update on column s:
  Candidates = {None} ∪ {current_partner of s, if any}
                       ∪ {all currently unpaired columns t != s}
  delta_t = pair_loglik(s, t) - single_loglik(s) - single_loglik(t)   for t != None
  delta_None = 0
  Sample t with probability ∝ exp(delta_t)

Update partner array consistently: if s was paired with u and now pairs with t,
then partner[u] -> -1, partner[t] -> s, partner[s] -> t.

The DP prior with size-2 cap is a uniform-over-matchings prior; we omit it
(it cancels in this single-column Gibbs proposal under the given cap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FamilyPartitionState:
    family: str
    L: int
    partner: np.ndarray  # (L,) int32, -1 for singleton, otherwise the partner index


def init_all_singletons(family: str, L: int) -> FamilyPartitionState:
    return FamilyPartitionState(family=family, L=L,
                                 partner=-np.ones(L, dtype=np.int32))


def init_random_pairs(family: str, L: int, n_pairs: int,
                      rng: np.random.Generator) -> FamilyPartitionState:
    state = init_all_singletons(family, L)
    cols = list(range(L))
    rng.shuffle(cols)
    for i in range(min(n_pairs, len(cols) // 2)):
        s, t = cols[2 * i], cols[2 * i + 1]
        state.partner[s] = t
        state.partner[t] = s
    return state


def init_from_pairs(family: str, L: int,
                    pairs: list[tuple[int, int]]) -> FamilyPartitionState:
    """Initialize from an explicit list of (i, j) pairs (e.g. PDB contacts).
    Pairs that share a column with an already-set pair are silently skipped.
    Pairs with out-of-range columns are also skipped.
    """
    state = init_all_singletons(family, L)
    for s, t in pairs:
        if not (0 <= s < L and 0 <= t < L) or s == t:
            continue
        if state.partner[s] != -1 or state.partner[t] != -1:
            continue
        state.partner[s] = t
        state.partner[t] = s
    return state


def n_pairs_in(state: FamilyPartitionState) -> int:
    return int((state.partner >= 0).sum() // 2)


def edges_of(state: FamilyPartitionState) -> list[tuple[int, int]]:
    """Return list of unique edge tuples (s, t) with s < t."""
    out = []
    for s in range(state.L):
        t = int(state.partner[s])
        if 0 <= t and s < t:
            out.append((s, t))
    return out


def gibbs_sweep(state: FamilyPartitionState,
                pair_loglik_fn,
                single_loglik: np.ndarray,
                rng: np.random.Generator,
                temperature: float = 1.0,
                log_pair_prior_offset: float = 0.0) -> FamilyPartitionState:
    """One Gibbs sweep over all columns (in random order).

    `pair_loglik_fn(s)` should return a (L,) array of pair_loglik(s, t) for
    every t (with the s-th entry ignored). It must use the *current* H but
    NOT condition on the rest of the partition (each pair_loglik is just the
    cherry-summed log-likelihood under the joint generator at H).

    `log_pair_prior_offset` adds a constant log-odds bias to *every* pair
    move (positive = favour pairs, negative = favour singletons). For the
    DP-with-size-2-cap prior, the natural choice is

        log_pair_prior_offset = log(1) - 2 * log(α) = -2 log α,

    which makes a pair "cost" α^2 relative to two singletons (each
    singleton costs α under CRP). Larger α => more singletons. With the
    default 0.0 this reduces to the uniform-over-matchings prior.
    """
    L = state.L
    order = rng.permutation(L)
    for s in order:
        u = int(state.partner[s])  # current partner, or -1
        # candidates: -1 ("None"), and every column t such that
        #   t != s, and (t is currently singleton, or t == u)
        is_unpaired = (state.partner == -1)
        # u, if exists, is currently paired with s (not "available" by the
        # general rule); we explicitly include it.
        cand_mask = is_unpaired.copy()
        cand_mask[s] = False
        if u >= 0:
            cand_mask[u] = True   # explicitly include current partner

        cand_idx = np.flatnonzero(cand_mask)  # column indices
        # delta_t = pair_loglik(s, t) - single_loglik[s] - single_loglik[t]
        #         + log_pair_prior_offset    (CRP-like edge cost)
        all_pair = pair_loglik_fn(s)  # (L,)
        deltas = (all_pair[cand_idx] - single_loglik[s] - single_loglik[cand_idx]
                  + log_pair_prior_offset)

        # Append the "singleton" option with delta = 0
        all_deltas = np.concatenate([[0.0], deltas])
        all_options = np.concatenate([[-1], cand_idx])

        # Categorical sample with temperature
        log_probs = all_deltas / temperature
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs /= probs.sum()
        choice_idx = rng.choice(len(all_options), p=probs)
        new_partner = int(all_options[choice_idx])

        # Update partner array consistently.
        # 1. If s was paired with u, free u.
        if u >= 0 and u != new_partner:
            state.partner[u] = -1
        # 2. If new_partner != -1 and was paired with someone (shouldn't happen
        #    since new_partner ∈ unpaired ∪ {u}), free that someone.
        if new_partner >= 0 and new_partner != u:
            old_partner_of_new = int(state.partner[new_partner])
            if old_partner_of_new >= 0 and old_partner_of_new != s:
                state.partner[old_partner_of_new] = -1
        # 3. Wire s ↔ new_partner.
        state.partner[s] = new_partner
        if new_partner >= 0:
            state.partner[new_partner] = s

    return state
