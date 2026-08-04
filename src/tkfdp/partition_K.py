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
from scipy.special import gammaln


@dataclass
class FamilyKState:
    family: str
    L: int
    K: int
    partner: np.ndarray  # (L,) int32, -1 for singleton (Potts / size-2 path)
    cls: np.ndarray      # (L,) int32, class label in {0..K-1}
    # Variable-size cluster support (Phase D.5 step 2). When set, this is
    # the (L,) int32 array of cluster ids; columns with the same cluster_id
    # are in the same cluster. Cluster ids are integers in [0, n_clusters);
    # they do not need to be contiguous or sorted (canonical_cluster_ids
    # normalises if desired). For backward compatibility, this field is
    # optional and defaults to None; the Potts trainer uses `partner` only.
    cluster_id: 'np.ndarray | None' = None
    # Per-site rate bin under the dynfield +Γ+I persite extension
    # (par:arch-gamma-plus-I-persite in appendix-tkfdp.tex). Values in
    # {0, 1, ..., K_rate_bins_site} where 0 encodes the invariant bin and
    # 1..K_rate_bins_site encode the Gamma quantile bins. None disables.
    site_rate_bin: 'np.ndarray | None' = None


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


# ---------------------------------------------------------------------------
# Variable-size cluster utilities (Phase D.5 step 2).
# ---------------------------------------------------------------------------

def cluster_id_from_partner(partner: np.ndarray) -> np.ndarray:
    """Derive a `cluster_id` array from a Potts-style `partner` array.

    Each pair `(s, t)` (where partner[s] = t and partner[t] = s) becomes
    one cluster; each singleton column becomes its own cluster. Cluster
    ids are assigned in column order with no gaps (canonical / sorted).
    """
    L = partner.shape[0]
    cluster_id = -np.ones(L, dtype=np.int32)
    next_id = 0
    for s in range(L):
        if cluster_id[s] >= 0:
            continue
        cluster_id[s] = next_id
        t = int(partner[s])
        if t >= 0 and cluster_id[t] < 0:
            cluster_id[t] = next_id
        next_id += 1
    return cluster_id


def partner_from_cluster_id(cluster_id: np.ndarray) -> np.ndarray:
    """Derive a Potts-compatible `partner` array from `cluster_id`.

    For each cluster of size 2, the two columns are made partners. For
    singletons, partner = -1. Clusters of size > 2 raise ValueError --
    Potts only supports size-{1, 2} partitions.
    """
    L = cluster_id.shape[0]
    partner = -np.ones(L, dtype=np.int32)
    clusters = clusters_from_cluster_id(cluster_id)
    for cid, members in clusters.items():
        if len(members) == 1:
            continue
        elif len(members) == 2:
            partner[members[0]] = members[1]
            partner[members[1]] = members[0]
        else:
            raise ValueError(
                f"cluster {cid} has size {len(members)} > 2; "
                f"partner array only supports size-{{1, 2}} clusters")
    return partner


def clusters_from_cluster_id(cluster_id: np.ndarray) -> dict[int, list[int]]:
    """Group columns by cluster id. Returns dict mapping cluster_id ->
    list of column indices (in column-order)."""
    out: dict[int, list[int]] = {}
    for s, cid in enumerate(np.asarray(cluster_id, dtype=np.int64)):
        out.setdefault(int(cid), []).append(int(s))
    return out


def canonical_cluster_ids(cluster_id: np.ndarray) -> np.ndarray:
    """Rewrite cluster ids so they are contiguous in [0, n_clusters), in
    order of first appearance."""
    L = cluster_id.shape[0]
    remap: dict[int, int] = {}
    out = np.empty(L, dtype=np.int32)
    next_id = 0
    for s in range(L):
        cid = int(cluster_id[s])
        if cid not in remap:
            remap[cid] = next_id
            next_id += 1
        out[s] = remap[cid]
    return out


def n_clusters_K(state: FamilyKState) -> int:
    """Number of distinct clusters in `state.cluster_id` (or implied by
    `state.partner` if cluster_id is None)."""
    if state.cluster_id is not None:
        return int(len(set(state.cluster_id.tolist())))
    # Fall back to partner-derived.
    return int(len(clusters_from_cluster_id(
        cluster_id_from_partner(state.partner))))


# ---------------------------------------------------------------------------
# CRP-style variable-size partition Gibbs (Phase D.5 step 2).
# ---------------------------------------------------------------------------

def gibbs_sweep_cluster(state: FamilyKState,
                          cluster_loglik_fn,
                          rng: np.random.Generator,
                          *,
                          alpha_z: float = 1.0,
                          max_cluster_size: int = 16,
                          temperature: float = 1.0,
                          batched_score_fn=None,
                          ) -> FamilyKState:
    """One CRP-Gibbs sweep over column -> cluster assignments (Neal 2000
    Algorithm 3, finite-truncated to `max_cluster_size`).

    For each column s in random order:
      1. Identify the current cluster c_curr that contains s.
      2. For each candidate move (stay in c_curr, leave to a new
         singleton cluster, join any other existing cluster), compute
         the marginal log-likelihood ratio and the CRP-on-blocks prior
         ratio.
      3. Sample s's new cluster id from the categorical.

    `cluster_loglik_fn(columns)` must return a scalar log-likelihood
    `log p(observations | cluster columns, classes[columns], t per cherry)`
    given a python list / tuple / 1-D int array of column indices.
    It is called O(n_clusters) times per column per sweep, so callers
    should cache per-cluster components where possible.

    Optionally, callers may pass `batched_score_fn(list_of_column_lists)
    -> np.ndarray[B]` that batch-scores all candidates for a column in a
    single call. When provided, the sweep uses it in preference to
    `cluster_loglik_fn`, which becomes a fallback for the empty-list
    edge case.

    The CRP prior on partitions: P(pi) propto alpha_z^{|pi|} * prod_B (|B|-1)!
    (Ewens with parameter alpha_z). For a move that takes column s from
    cluster c_curr (size n) to cluster c_new (size m), the prior ratio
    contribution depends only on the affected cluster sizes; constants
    drop out of the Gibbs categorical.

    `max_cluster_size` caps cluster size to prevent runaway aggregation
    in the early iterations (where the model has not yet learned to
    distinguish coupled from uncoupled column pairs); default 16 is
    well above typical Pfam coupling structure.
    """
    L = state.L
    if state.cluster_id is None:
        state.cluster_id = cluster_id_from_partner(state.partner)
    cluster_id = state.cluster_id.copy()

    order = rng.permutation(L)
    for s in order:
        s = int(s)
        c_curr = int(cluster_id[s])
        clusters = clusters_from_cluster_id(cluster_id)
        curr_members = clusters[c_curr]
        n_curr = len(curr_members)
        curr_minus = [c for c in curr_members if c != s] if n_curr > 1 else []

        # Build the list of column-lists we need to score, plus an index
        # map back to candidate options.
        # Indices: 0 = curr_members, 1 = curr_minus, 2 = [s] (singleton).
        # Then for each "other" cluster: (other_members, other_members+[s]).
        score_requests: 'list[list[int]]' = [
            list(curr_members), curr_minus, [s],
        ]
        c_other_ids: 'list[int]' = []
        for c_other, other_members in clusters.items():
            if c_other == c_curr:
                continue
            if len(other_members) + 1 > max_cluster_size:
                continue
            c_other_ids.append(c_other)
            score_requests.append(list(other_members))
            score_requests.append(list(other_members) + [s])

        if batched_score_fn is not None:
            scores = batched_score_fn(score_requests)
        else:
            scores = np.array(
                [cluster_loglik_fn(r) for r in score_requests],
                dtype=np.float64)

        loglik_curr = float(scores[0])
        loglik_curr_minus_s = float(scores[1]) if n_curr > 1 else 0.0
        loglik_singleton = float(scores[2]) if n_curr > 1 else 0.0

        cand_ids = []
        cand_log_p = []

        # (1) Stay in c_curr.
        cand_ids.append(c_curr)
        cand_log_p.append(0.0)

        # (2) Move to a brand-new singleton cluster. Only meaningful if
        # n_curr > 1 (otherwise this is a no-op).
        if n_curr > 1:
            lik_delta = (-loglik_curr + loglik_curr_minus_s
                          + loglik_singleton)
            # Ewens prior delta: c_curr shrinks (|B|-1)! factor n-1 -> n-2,
            # contributing -log(n-1); a new size-1 block adds 0! = 1 and one
            # more block factor alpha_z -> +log alpha_z.
            prior_delta = float(np.log(alpha_z)) - float(np.log(n_curr - 1))
            cand_ids.append(-1)
            cand_log_p.append(lik_delta + prior_delta)

        # (3) Move to each other existing cluster.
        for i, c_other in enumerate(c_other_ids):
            other_members = clusters[c_other]
            m = len(other_members)
            loglik_other = float(scores[3 + 2 * i])
            loglik_other_plus = float(scores[3 + 2 * i + 1])
            lik_delta = (-loglik_curr + loglik_curr_minus_s
                          - loglik_other + loglik_other_plus)
            if n_curr > 1:
                # c_curr shrinks (-log(n-1)); c_other grows (+log m).
                prior_delta = -float(np.log(n_curr - 1)) + float(np.log(m))
            else:
                # c_curr disappears (lose one block: -log alpha_z); c_other
                # grows (+log m).
                prior_delta = -float(np.log(alpha_z)) + float(np.log(m))
            cand_ids.append(c_other)
            cand_log_p.append(lik_delta + prior_delta)

        # Categorical sample.
        log_p = np.asarray(cand_log_p, dtype=np.float64) / temperature
        log_p -= log_p.max()
        p = np.exp(log_p)
        p /= p.sum()
        idx = int(rng.choice(len(p), p=p))
        new_cid = cand_ids[idx]

        if new_cid == -1:
            # Allocate a fresh cluster id (one larger than current max).
            new_cid = int(cluster_id.max()) + 1
        if new_cid == c_curr:
            continue   # no move
        cluster_id[s] = new_cid

    state.cluster_id = canonical_cluster_ids(cluster_id)
    # Keep `partner` in sync where possible (size <=2 only):
    try:
        state.partner = partner_from_cluster_id(state.cluster_id)
    except ValueError:
        # cluster_id contains size > 2 clusters; partner has no
        # well-defined value. Leave it stale.
        pass
    return state


# ---------------------------------------------------------------------------
# CRP concentration alpha_z: random-walk MH on log alpha_z under Gamma prior.
# ---------------------------------------------------------------------------

def update_alpha_z_mh(alpha_z: float,
                       partitions_per_msa: 'list[tuple[int, int]]',
                       *,
                       prior_a: float = 1.5,
                       prior_b: float = 2.0,
                       n_steps: int = 5,
                       step_size: float = 0.5,
                       rng: 'np.random.Generator | None' = None,
                       ) -> 'tuple[float, dict]':
    """Random-walk MH on log alpha_z under a Gamma(prior_a, prior_b) prior.

    Gamma(prior_a, prior_b) here uses the rate parameterisation: density
    proportional to alpha^(a-1) * exp(-b * alpha); mean = a / b.

    For each MSA m with `n_m` columns partitioned into `K_m` clusters, the
    Ewens density contributes

      log P(pi_m | alpha_z) = K_m log alpha_z + sum_{B in pi_m} log Gamma(|B|)
                              - log Gamma(alpha_z + n_m) + log Gamma(alpha_z)

    The block-size term `sum_B log Gamma(|B|)` is constant in alpha_z and
    cancels in the MH ratio, so we only need (K_m, n_m) per MSA.

    Proposal: symmetric Gaussian in log alpha_z. Accept ratio
      [ log P(pi | alpha') + log prior(alpha') ]
        - [ log P(pi | alpha)  + log prior(alpha)  ].
    The Jacobian of the log transform cancels in the symmetric proposal.

    Args:
      alpha_z: current value (positive).
      partitions_per_msa: list of (K_m, n_m) per MSA. K_m = number of
        clusters in MSA m's partition; n_m = number of columns.
      prior_a, prior_b: Gamma shape, rate parameters.
      n_steps: number of MH attempts to make.
      step_size: standard deviation of the log-space Gaussian proposal.

    Returns:
      (new_alpha_z, info) where info has 'n_steps_accept', 'final_log_lik',
      'final_log_post'.
    """
    if rng is None:
        rng = np.random.default_rng()
    assert alpha_z > 0
    assert prior_a > 0 and prior_b > 0
    partitions_per_msa = [(int(K), int(n)) for K, n in partitions_per_msa]

    def log_lik(alpha: float) -> float:
        if alpha <= 0:
            return -np.inf
        ll = 0.0
        log_alpha = np.log(alpha)
        gammaln_alpha = gammaln(alpha)
        for K, n in partitions_per_msa:
            ll += (K * log_alpha
                    + gammaln_alpha
                    - gammaln(alpha + n))
        return float(ll)

    def log_prior(alpha: float) -> float:
        if alpha <= 0:
            return -np.inf
        return float((prior_a - 1.0) * np.log(alpha) - prior_b * alpha)

    log_cur = log_lik(alpha_z) + log_prior(alpha_z)
    n_accept = 0
    for _ in range(int(n_steps)):
        prop = float(np.exp(np.log(alpha_z) + step_size * rng.normal()))
        log_prop = log_lik(prop) + log_prior(prop)
        log_ratio = log_prop - log_cur
        if np.log(rng.uniform()) < log_ratio:
            alpha_z = prop
            log_cur = log_prop
            n_accept += 1
    info = {
        'n_steps_accept': int(n_accept),
        'final_log_lik': log_lik(alpha_z),
        'final_log_post': log_cur,
    }
    return float(alpha_z), info
