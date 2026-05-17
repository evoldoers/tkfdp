"""K-class extension of partition.py: each column carries both a partner
(size-2 partition) and a class label c_s ∈ {0, ..., K-1}.

Joint Gibbs proposal on column s:
  Candidates = {(-1, k) for k in 0..K-1}
              ∪ {(t, k) for t != s, current_partner[t] in {-1, s},
                              k in 0..K-1}

The conditional log-prob includes:
  - pair-likelihood under H_{k, c_t} if pair candidate
  - singleton-likelihood (under shared LG08 baseline) if singleton
  - log_pair_prior_offset: per-pair Ewens log-prior cost. For a size-{1,2}
        Ewens partition with concentration alpha_z, P(π) ∝ alpha_z^{|π|}
        (the Γ(|B|) block-size factors collapse since Γ(1) = Γ(2) = 1).
        A pair option has one fewer block than the singleton alternative,
        so the proper per-pair log-prior cost is -log(alpha_z). Pass that
        value in via log_pair_prior_offset; the Ewens normalization
        (Pochhammer (alpha_z)_L) is constant across moves and drops out
        of the Gibbs ratio.
  - finite-K Dirichlet-Multinomial class prior on c_s:
        log p(c_s = k | c_{-s}) = log(n_k^{-s} + alpha_c / K) - log(L - 1 + alpha_c)
    or stick-breaking weights (TSB) if class_log_weights is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FamilyKState:
    family: str
    L: int
    K: int
    partner: np.ndarray  # (L,) int32, -1 for singleton
    cls: np.ndarray      # (L,) int32, class label in {0..K-1}


def init_random_K(family: str, L: int, K: int,
                  n_pairs: int,
                  rng: np.random.Generator) -> FamilyKState:
    """Random partition (n_pairs random pairs, rest singletons) and uniform
    random class labels."""
    partner = -np.ones(L, dtype=np.int32)
    cols = list(range(L))
    rng.shuffle(cols)
    for i in range(min(n_pairs, L // 2)):
        s, t = cols[2 * i], cols[2 * i + 1]
        partner[s] = t; partner[t] = s
    cls = rng.integers(0, K, size=L).astype(np.int32)
    return FamilyKState(family=family, L=L, K=K, partner=partner, cls=cls)


def init_from_pairs_K(family: str, L: int, K: int,
                       pairs: list[tuple[int, int]],
                       rng: np.random.Generator) -> FamilyKState:
    """Init partition from explicit pairs (e.g. PDB contacts), classes uniform random."""
    partner = -np.ones(L, dtype=np.int32)
    for s, t in pairs:
        if not (0 <= s < L and 0 <= t < L) or s == t: continue
        if partner[s] != -1 or partner[t] != -1: continue
        partner[s] = t; partner[t] = s
    cls = rng.integers(0, K, size=L).astype(np.int32)
    return FamilyKState(family=family, L=L, K=K, partner=partner, cls=cls)


def n_pairs_K(state: FamilyKState) -> int:
    return int((state.partner >= 0).sum() // 2)


def gibbs_sweep_K(state: FamilyKState,
                   pair_loglik_fn,
                   single_loglik: np.ndarray,
                   rng: np.random.Generator,
                   temperature: float = 1.0,
                   log_pair_prior_offset: float = 0.0,
                   alpha_c: float = 1.0,
                   fix_partition: bool = False,
                   class_log_weights: np.ndarray | None = None,
                   allowed_partner_mask: np.ndarray | None = None,
                   ) -> FamilyKState:
    """One Gibbs sweep over all columns. For each column s, the proposal is
    over (new_partner, new_class) jointly.

    `pair_loglik_fn(s)` returns a (K, K, L) array `pair_lk[k_s, k_t, t]` =
    cherry-summed log P_{(x_s, x_t), (y_s, y_t)}(τ; H_{k_s, k_t}). The
    s-th entry along the last axis is ignored.

    `single_loglik[s]` is the LG08 single-site contribution at column s
    (class-independent in the v0 minimal version).

    `alpha_c` is the symmetric Dirichlet-Multinomial concentration on
    class assignments.

    If `fix_partition=True`, the partition is held fixed (no partner moves)
    and only the class label `c_s` is resampled per column. Useful for
    "anchor MSA" mode where pairs are pinned to known PDB contacts but
    class labels remain latent.

    If `class_log_weights` (shape (K,)) is supplied, it overrides the
    symmetric finite-K Dirichlet-Multinomial class prior. Pass log ρ_k
    here when running TSB (truncated stick-breaking, see
    `tkfdp.tsb.stick_to_weights`).

    If `allowed_partner_mask` (shape (L, L), bool) is supplied, the
    partner draw for column s is restricted to {t : allowed_partner_mask[s, t]}.
    Singleton always remains an option (the restriction only narrows the
    pair branch). This gives a "PDB-restrict" mode where the chain
    explores valid size-{1,2} partitions whose pairs are a subset of a
    known candidate set (e.g. all Cα < 8 Å contacts), without committing
    to any one greedy assignment. The matrix should be symmetric (i.e.
    `allowed_partner_mask[s, t] == allowed_partner_mask[t, s]`); the diagonal
    is ignored. Has no effect when `fix_partition=True`.
    """
    L = state.L; K = state.K
    order = rng.permutation(L)
    for s in order:
        u = int(state.partner[s])  # current partner or -1

        # Class prior — symmetric finite-K (default) or stick-breaking weights (TSB)
        if class_log_weights is not None:
            log_class_prior = class_log_weights   # already log ρ_k under TSB
        else:
            counts = np.bincount(state.cls, minlength=K).astype(np.float64)
            counts[state.cls[s]] -= 1.0  # exclude s
            log_class_prior = np.log(counts + alpha_c / K) - np.log(L - 1 + alpha_c)

        # Pair-loglik table for column s: shape (K, K, L)
        pl = pair_loglik_fn(s)

        # Singleton evidence at column s. If single_loglik is (L, K) then
        # singleton_evidence(s, k) is class-conditional; if it's (L,) it's
        # broadcast as class-independent (legacy behavior).
        if single_loglik.ndim == 2:
            sll_s_per_k = single_loglik[s]                # (K,)
        else:
            sll_s_per_k = np.full(K, single_loglik[s])

        if fix_partition:
            # Class-only resample: keep partner fixed, sample c_s | rest
            if u >= 0:
                # paired: log p(c_s = k) ∝ pl[k, c_t, t] + log_class_prior[k]
                c_t = int(state.cls[u])
                deltas = pl[:, c_t, u] + log_class_prior   # (K,)
            else:
                # singleton: + class-conditional singleton evidence
                deltas = log_class_prior + sll_s_per_k     # (K,)
            log_probs = deltas / temperature
            log_probs -= log_probs.max()
            probs = np.exp(log_probs); probs /= probs.sum()
            state.cls[s] = int(rng.choice(K, p=probs))
            continue

        is_unpaired = (state.partner == -1)
        cand_mask = is_unpaired.copy()
        cand_mask[s] = False
        if u >= 0:
            cand_mask[u] = True
        if allowed_partner_mask is not None:
            cand_mask &= allowed_partner_mask[s]
        cand_idx = np.flatnonzero(cand_mask)  # eligible partner columns

        # Singleton option in class k_s gets the column's class-conditional
        # singleton-evidence sll_s_per_k[k_s] + log_class_prior[k_s].
        # The pair option for partner t in class k_s gets pl[k_s, c_t, t]
        # MINUS the singleton-baseline at the (s, t) class-pair (s in class
        # k_s, t fixed at its current class c_t), so that the comparison is
        # absolute rather than relative.
        if single_loglik.ndim == 2:
            sll_t_per_ct = single_loglik[cand_idx, state.cls[cand_idx]]   # (M,)
        else:
            sll_t_per_ct = single_loglik[cand_idx]                         # (M,)

        # Singleton options: (K,)
        single_deltas = log_class_prior + sll_s_per_k

        # Pair options: pair_loglik(s, t; k_s, c_t) + log_class_prior[k_s]
        #              + log_pair_prior_offset
        # No singleton-baseline subtraction needed since we made the
        # singleton branch ABSOLUTE above. But to keep the same scale we
        # subtract sll_t_per_ct (t's singleton evidence at its current
        # class) — this is a constant that doesn't affect s's choice but
        # keeps numerical magnitudes balanced.
        c_t = state.cls[cand_idx]                                           # (M,)
        pair_term = pl[:, c_t, cand_idx]                                    # (K, M)
        pair_deltas = (pair_term
                        - sll_t_per_ct[None, :]
                        + log_pair_prior_offset
                        + log_class_prior[:, None])                         # (K, M)

        # Stack all options
        # Singleton: option_id = (-1, k); count K options. delta = single_deltas[k]
        # Pair:      option_id = (t,  k); count K * M options. delta = pair_deltas[k, m]
        # We want a flat list of deltas in a known canonical order so we can
        # decode the choice cleanly.
        M = len(cand_idx)
        flat_deltas = np.concatenate([single_deltas, pair_deltas.reshape(-1)])
        # Categorical sample with temperature
        log_probs = flat_deltas / temperature
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs /= probs.sum()
        choice = rng.choice(len(flat_deltas), p=probs)

        if choice < K:
            new_partner = -1; new_class = int(choice)
        else:
            idx = choice - K
            new_class = int(idx // M)
            partner_idx = int(idx % M)
            new_partner = int(cand_idx[partner_idx])

        # Update partner consistently
        if u >= 0 and u != new_partner:
            state.partner[u] = -1
        if new_partner >= 0 and new_partner != u:
            old_partner_of_new = int(state.partner[new_partner])
            if old_partner_of_new >= 0 and old_partner_of_new != s:
                state.partner[old_partner_of_new] = -1
        state.partner[s] = new_partner
        if new_partner >= 0:
            state.partner[new_partner] = s
        state.cls[s] = new_class

    return state
