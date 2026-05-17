"""Sequence annealing with coevolutionary scoring of coupled merges.

Implements the variant described in Section "Sequence annealing with
coevolutionary scoring" of main.tex: instead of folding the Potts boost
into the per-residue posterior matrices Q_{ij} via a mean-field correction
(see ``src/tkfdp/postprocessing.py``), the boost is applied DURING the
greedy column-merging step. Single-edge candidates and coupled-pair
candidates share a single priority queue; when a coupled candidate is
popped, we attempt both component merges atomically, falling back to a
single-edge commit if one half has been invalidated by an intervening
merge.

The DAG bookkeeping (Pearce--Kelly online topological ordering for cycle
detection, same-sequence-exclusion checks, TGF dynamic-recalculation)
mirrors ``tkfmixdom.jax.tree.fsa_anneal._amap_align`` essentially line for
line; only the candidate-set construction and the pop-loop are extended.

Reuses the per-class-pair joint emission tensor of
``src/tkfdp/postprocessing.py``; the Potts machinery is identical.

Public API:
    coupled_sequence_annealing(...)
        Drop-in replacement for fsa_anneal.sequence_annealing that takes
        an additional argument ``boost_tensors`` (per sequence-pair joint
        emission state) used to score coupled candidates.
    build_boost_state(...)
        Convenience builder: from a dict of pair posteriors, sequence
        arrays, branch lengths, and a trained TKF-DP state, precompute the
        per-(seq-pair) inputs required by ``coupled_sequence_annealing``.
"""

from __future__ import annotations

import heapq
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Pull the score / refinement helpers from tkfmixdom (do not modify them).
TKFMIXDOM_ROOT = Path.home() / "tkf-mixdom" / "python"
if str(TKFMIXDOM_ROOT) not in sys.path:
    sys.path.insert(0, str(TKFMIXDOM_ROOT))
from tkfmixdom.jax.tree.fsa_anneal import (                                       # noqa: E402
    _score_alignment, _refine_one_sequence,
)

from .postprocessing import (
    build_per_class_match_emit, build_per_classpair_joint_emit,
    class_posteriors_from_baseline,
)


# --- Per-pair boost-state precomputation ----------------------------------


@dataclass
class PairBoostState:
    """Precomputed quantities for fast coupled-candidate scoring on a
    fixed (X, Y) sequence pair.

    Fields:
        x_seq, y_seq: (L_X,), (L_Y,) int arrays (clamped to amino-acid
            alphabet 0..A-1).
        gamma: (L_X, L_Y, K_c) per-(i, j) class posterior gamma_{ij}(c)
            from eq:gamma in the paper. RELIC -- the gamma-weighted M
            construction was the pre-MCMC first-order correction; the
            current MCMC chain uses the class-marginal M tensor (see
            tkf_state + branch_length below) and ignores gamma/denom.
        denom: (L_X, L_Y) per-column denominator (also relic, see gamma).
        joint_per_cp: (K_c, K_c, A, A, A, A) per-class-pair joint emission
            tensor at branch length t (shared across all pairs at the same
            branch length, but cached per pair here for simplicity).
        tkf_state: the K=4 TKF-DP state (object with K_c, pi_class,
            potts_dp.atoms/assignments/h_pairs). Used by the class-marginal
            M-tensor builder.
        branch_length: float tau, used by build_M_tensor_classmarg.
    """
    x_seq: np.ndarray
    y_seq: np.ndarray
    gamma: np.ndarray
    denom: np.ndarray
    joint_per_cp: np.ndarray
    tkf_state: object = None
    branch_length: float = 0.0
    # Class prior (empirical from training counts) and pair-background
    # convention used by the unified block_likelihoods builders. These
    # are set by the caller (e.g. sweep_infinite_phmm_balibase.py) so
    # the MCMC chain in mcmc_infinite_phmm.py can use the canonical
    # singlet emission + M tensor.
    pi_c: np.ndarray = None
    pair_background: str = 'lg08'


def build_boost_state(pair_posteriors: Dict[Tuple[int, int], np.ndarray],
                      pair_taus: Dict[Tuple[int, int], float],
                      sequences_int: List[np.ndarray],
                      tkfdp_state, S=None,
                      A: int = 20,
                      pi_c=None,
                      pair_background: str = 'lg08'
                      ) -> Dict[Tuple[int, int], PairBoostState]:
    """Build the per-pair boost state for every pair in pair_posteriors.

    Args:
        pair_posteriors: {(i, j): (L_i, L_j) Q_{ij}}, the BASELINE pair-HMM
            match-state posteriors (no Potts correction).
        pair_taus: {(i, j): float} branch length used for that pair.
        sequences_int: list of integer-coded sequences (wildcards may map
            to >= A; we clamp to A - 1 for the boost calculation since the
            trained Potts atoms live on a length-A alphabet).
        tkfdp_state: a state object exposing K_c, pi_class, potts_dp.
        S: optional exchangeability matrix (defaults to LG08 inside
            ``build_per_class_match_emit``).
        A: alphabet size (default 20).

    Returns: {(i, j): PairBoostState}.
    """
    out: Dict[Tuple[int, int], PairBoostState] = {}
    pi_uniform = jnp.full(tkfdp_state.K_c, 1.0 / tkfdp_state.K_c)
    # Cache (joint_per_cp, per_class_emit) by branch-length key, since
    # multiple pairs may share a tau.
    joint_cache: Dict[float, jnp.ndarray] = {}
    emit_cache: Dict[float, jnp.ndarray] = {}

    for (i, j), Q in pair_posteriors.items():
        t = float(pair_taus[(i, j)])
        if t not in joint_cache:
            joint_cache[t] = build_per_classpair_joint_emit(
                tkfdp_state, t, S=S)
            emit_cache[t] = build_per_class_match_emit(
                tkfdp_state.pi_class, t, S=S)
        joint_per_cp = joint_cache[t]
        per_class_emit = emit_cache[t]

        x_clamp = np.minimum(np.asarray(sequences_int[i], dtype=np.int64),
                             A - 1)
        y_clamp = np.minimum(np.asarray(sequences_int[j], dtype=np.int64),
                             A - 1)

        gamma = class_posteriors_from_baseline(per_class_emit,
                                               x_clamp, y_clamp,
                                               pi_uniform)
        # denom_{i,j} = sum_c gamma[i,j,c] * P(X_i, Y_j | c)
        e_ij = per_class_emit[:, x_clamp, :][:, :, y_clamp]      # (K, Lx, Ly)
        e_ij = jnp.transpose(e_ij, (1, 2, 0))                    # (Lx, Ly, K)
        denom = jnp.sum(gamma * e_ij, axis=-1)                   # (Lx, Ly)
        denom = jnp.clip(denom, 1e-300, None)

        out[(i, j)] = PairBoostState(
            x_seq=x_clamp,
            y_seq=y_clamp,
            gamma=np.asarray(gamma),
            denom=np.asarray(denom),
            joint_per_cp=np.asarray(joint_per_cp),
            tkf_state=tkfdp_state,
            branch_length=float(t),
            pi_c=(np.asarray(pi_c, dtype=np.float64) if pi_c is not None
                  else None),
            pair_background=pair_background,
        )
    return out


# --- The pairwise log-M query ----------------------------------------------


def log_M_at_pair(boost_state: PairBoostState,
                  i: int, j: int, ip: int, jp: int) -> float:
    """log M(i, j; i', j'; t) for one specific quartet (i, j, i', j').

    Uses the class-marginalized definition (eq:M-marginal in main.tex).
    Cost is O(K_c^2) per query plus a single A^2 lookup into joint_per_cp.
    """
    g_ij = boost_state.gamma[i, j]                               # (K,)
    g_ip = boost_state.gamma[ip, jp]                             # (K,)
    Xi = int(boost_state.x_seq[i]); Yj = int(boost_state.y_seq[j])
    Xip = int(boost_state.x_seq[ip]); Yjp = int(boost_state.y_seq[jp])
    # joint_per_cp[c, c', a, a', b, b'] -> we want J[:, :, Xi, Xip, Yj, Yjp]
    # which is a (K, K) slice.
    J = boost_state.joint_per_cp[:, :, Xi, Xip, Yj, Yjp]         # (K, K)
    numer = float(np.einsum('c,d,cd->', g_ij, g_ip, J))
    denom = float(boost_state.denom[i, j] * boost_state.denom[ip, jp])
    if denom <= 0.0:
        return 0.0
    return float(np.log(max(numer, 1e-300)) - np.log(denom))


def log_M_field_for_pair(boost_state: PairBoostState,
                         i: int, j: int) -> np.ndarray:
    """Vectorized: log M(i, j; i', j') over all (i', j') with the
    fixed (i, j). Returns a (L_X, L_Y) array.

    Used by the candidate-generation step to enumerate boosts cheaply.
    """
    g_ij = boost_state.gamma[i, j]                               # (K,)
    Xi = int(boost_state.x_seq[i]); Yj = int(boost_state.y_seq[j])
    Lx, Ly, _ = boost_state.gamma.shape
    # Build T[c', a', b'] = sum_c gamma[i,j,c] * J[c, c', Xi, a', Yj, b'].
    J_slice = boost_state.joint_per_cp[:, :, Xi, :, Yj, :]       # (K, K, A, A)
    T = np.einsum('c,cdpb->dpb', g_ij, J_slice)                  # (K, A, A)
    # Now numer[i', j'] = sum_c' gamma[i', j', c'] * T[c', X_{i'}, Y_{j'}]
    Xa = boost_state.x_seq                                       # (Lx,)
    Yb = boost_state.y_seq                                       # (Ly,)
    T_ip_jp = T[:, Xa, :][:, :, Yb]                              # (K, Lx, Ly)
    T_ip_jp = np.transpose(T_ip_jp, (1, 2, 0))                   # (Lx, Ly, K)
    numer = np.sum(boost_state.gamma * T_ip_jp, axis=-1)         # (Lx, Ly)
    denom = boost_state.denom[i, j] * boost_state.denom           # (Lx, Ly)
    safe_num = np.clip(numer, 1e-300, None)
    safe_den = np.clip(denom, 1e-300, None)
    return np.log(safe_num) - np.log(safe_den)


# --- The coupled annealer --------------------------------------------------


def coupled_sequence_annealing(
        n_seqs: int,
        seq_lengths: List[int],
        pair_posteriors: Dict[Tuple[int, int], np.ndarray],
        boost_states: Optional[Dict[Tuple[int, int], PairBoostState]] = None,
        n_iterations: int = 5,
        verbose: bool = False,
        gap_factor: float = 1.0,
        edge_weight_threshold: float = 0.0,
        q_min: float = 0.1,
        mu_min: float = 0.1,
        max_pairs_per_anchor: int = 32,
        scoring_mode: str = "log_M",
        prior_coup: float = 0.01,
        lambda_pair: float = 1.0,
):
    """Build MSA via AMAP sequence annealing with coupled-pair coevolutionary
    scoring of merges.

    Mirrors the signature of fsa_anneal.sequence_annealing, with extra
    arguments controlling the coupled extension:
        boost_states: per (seq_i, seq_j) PairBoostState. If None, falls
            back to the baseline single-edge AMAP (equivalent to running
            fsa_anneal.sequence_annealing).
        q_min: minimum baseline posterior to consider a residue pair as
            either anchor or partner of a coupled candidate (threshold
            pruning, default 0.1).
        mu_min: minimum |log M| to consider a coupled candidate worth
            scoring jointly (boost pruning, default 0.1 nat).
        max_pairs_per_anchor: hard cap on the number of partner candidates
            per anchor, retained by largest |log M| (default 32).
        scoring_mode: "log_M" (default; Design A in
            analysis/coupled_fsa_design_alternatives.md): the multiplicative
            sqrt(M) boost on the average TGF weight. "posterior"
            (Design B): the additive lambda_pair * p_coup bonus, with
            p_coup the Bayesian posterior on the column-pair being
            coupled given the prior_coup pre-coupling probability.
        prior_coup: prior probability that a candidate column-pair is
            coupled (Design B only). Default 0.01 = 1/alpha_z at our
            default alpha_z=100.
        lambda_pair: bonus weight on the coupling posterior (Design B
            only). Default 1.0; analogous in scale to FSA's gap_factor.

    Returns:
        col_assignments: list of (L_i,) int arrays.
        msa_length: number of columns.
    """
    col_assignments = _coupled_amap_align(
        n_seqs, seq_lengths, pair_posteriors, boost_states,
        gap_factor=gap_factor,
        edge_weight_threshold=edge_weight_threshold,
        q_min=q_min, mu_min=mu_min,
        max_pairs_per_anchor=max_pairs_per_anchor,
        scoring_mode=scoring_mode,
        prior_coup=prior_coup,
        lambda_pair=lambda_pair,
        verbose=verbose,
    )

    if all(len(ca) == 0 for ca in col_assignments):
        return col_assignments, 0

    n_cols = max(max(ca) for ca in col_assignments if len(ca) > 0) + 1

    if verbose:
        score = _score_alignment(col_assignments, seq_lengths,
                                 pair_posteriors)
        print(f"  Coupled AMAP build: {n_cols} cols, score={score:.4f}")

    # Refinement (same as upstream).
    for iteration in range(n_iterations):
        improved = False
        best_score = _score_alignment(col_assignments, seq_lengths,
                                      pair_posteriors)
        order = np.random.permutation(n_seqs)
        for seq_idx in order:
            if seq_lengths[seq_idx] == 0:
                continue
            col_assignments_new, n_cols_new = _refine_one_sequence(
                col_assignments, seq_idx, n_seqs, seq_lengths,
                pair_posteriors)
            new_score = _score_alignment(col_assignments_new, seq_lengths,
                                         pair_posteriors)
            if new_score > best_score + 1e-10:
                col_assignments = col_assignments_new
                n_cols = n_cols_new
                best_score = new_score
                improved = True
        if verbose:
            print(f"  Refinement {iteration+1}: {n_cols} cols, "
                  f"score={best_score:.4f}, improved={improved}")
        if not improved:
            break

    return col_assignments, n_cols


# --- Coupled AMAP core ----------------------------------------------------


def _coupled_amap_align(n_seqs: int,
                        seq_lengths: List[int],
                        pair_posteriors: Dict[Tuple[int, int], np.ndarray],
                        boost_states: Optional[Dict[Tuple[int, int],
                                                    PairBoostState]],
                        gap_factor: float = 1.0,
                        edge_weight_threshold: float = 0.0,
                        q_min: float = 0.1,
                        mu_min: float = 0.1,
                        max_pairs_per_anchor: int = 32,
                        scoring_mode: str = "log_M",
                        prior_coup: float = 0.01,
                        lambda_pair: float = 1.0,
                        verbose: bool = False):
    """Core AMAP DAG column-merging algorithm with coupled-pair candidates.

    The DAG / Pearce--Kelly bookkeeping is adapted from
    ``tkfmixdom.jax.tree.fsa_anneal._amap_align`` (DART
    ``MultiSequenceDag::AlignDag``); only the candidate-set construction
    and the pop-loop are different.
    """
    if sum(seq_lengths) == 0:
        return [np.array([], dtype=int) for _ in range(n_seqs)]

    # ---- Gap posteriors --------------------------------------------------
    gap_post_0 = {}  # (i, j) -> (L_i,) gap posterior for seq i vs seq j
    gap_post_1 = {}  # (i, j) -> (L_j,) gap posterior for seq j vs seq i
    for (i, j), post in pair_posteriors.items():
        post = np.asarray(post)
        gp0 = np.maximum(1.0 - np.sum(post, axis=1), 1e-4)
        gp1 = np.maximum(1.0 - np.sum(post, axis=0), 1e-4)
        gap_post_0[(i, j)] = gp0
        gap_post_1[(i, j)] = gp1

    # ---- Column / topological state -------------------------------------
    total_residues = sum(seq_lengths)
    col_seqs: Dict[int, Dict[int, int]] = {}
    seqpos_to_col: Dict[Tuple[int, int], int] = {}
    col_index: Dict[int, int] = {}
    merged_into: Dict[int, int] = {}

    cid = 0
    idx = 0
    for si in range(n_seqs):
        for k in range(seq_lengths[si]):
            pos = k + 1
            col_seqs[cid] = {si: pos}
            seqpos_to_col[(si, pos)] = cid
            col_index[cid] = idx
            merged_into[cid] = cid
            cid += 1; idx += 1

    live_cols = set(range(total_residues))

    def find_col(c):
        while merged_into[c] != c:
            merged_into[c] = merged_into[merged_into[c]]
            c = merged_into[c]
        return c

    # ---- Pearce--Kelly cycle detection (verbatim from upstream) ---------
    node_visited: set = set()

    def dfs_forward(node, upper_bound, r_forward):
        node_visited.add(node)
        r_forward.append(node)
        for seq, pos in col_seqs[node].items():
            nxt_key = (seq, pos + 1)
            if nxt_key not in seqpos_to_col:
                continue
            w = find_col(seqpos_to_col[nxt_key])
            if w not in live_cols or w == node:
                continue
            if col_index[w] == col_index[upper_bound]:
                return True
            if (w not in node_visited and
                    col_index[w] < col_index[upper_bound]):
                if dfs_forward(w, upper_bound, r_forward):
                    return True
        return False

    def dfs_backward(node, lower_bound, r_backward):
        node_visited.add(node)
        r_backward.append(node)
        for seq, pos in col_seqs[node].items():
            prev_key = (seq, pos - 1)
            if prev_key not in seqpos_to_col:
                continue
            w = find_col(seqpos_to_col[prev_key])
            if w not in live_cols or w == node:
                continue
            if (w not in node_visited and
                    col_index[lower_bound] < col_index[w]):
                dfs_backward(w, lower_bound, r_backward)

    def reorder(r_forward, r_backward):
        all_indices = sorted(
            [col_index[c] for c in r_backward] +
            [col_index[c] for c in r_forward])
        r_backward.sort(key=lambda c: col_index[c])
        r_forward.sort(key=lambda c: col_index[c])
        it = iter(all_indices)
        for c in r_backward:
            col_index[c] = next(it)
        for c in r_forward:
            col_index[c] = next(it)

    def try_add_edge(col1, col2):
        nonlocal node_visited
        col1 = find_col(col1); col2 = find_col(col2)
        if col1 == col2:
            return 0
        if set(col_seqs[col1].keys()) & set(col_seqs[col2].keys()):
            return 1
        if col_index[col1] < col_index[col2]:
            l_bound, u_bound = col1, col2
        else:
            l_bound, u_bound = col2, col1
        node_visited = set()
        r_forward = []
        if dfs_forward(l_bound, u_bound, r_forward):
            return 2
        r_backward = []
        dfs_backward(u_bound, l_bound, r_backward)
        node_visited = set()
        if len(r_forward) == 1:
            col1, col2 = u_bound, l_bound
        elif len(r_backward) == 1:
            col1, col2 = l_bound, u_bound
        else:
            reorder(r_forward, r_backward)
            col1, col2 = l_bound, u_bound
        for seq, pos in list(col_seqs[col2].items()):
            col_seqs[col1][seq] = pos
            seqpos_to_col[(seq, pos)] = col1
        merged_into[col2] = col1
        live_cols.discard(col2)
        del col_seqs[col2]
        return 0

    # ---- Coupled-pair composite-likelihood state ------------------------
    # For each currently-committed coupled pair, track the running tree-
    # shaped composite log-likelihood ratio log[ ∏_cherries P_joint /
    # ∏_cherries P_indep ] across the cherries that have been committed
    # to this pair so far. Each coupled-merge commit adds exactly one
    # cherry's log_M (the cherry between the two sequences whose residues
    # were the merge endpoints). Single-edge commits add NO cherry to a
    # coupled pair (a single-edge merge can grow at most one column of
    # the pair; a cherry contributes only when both columns gain the
    # same sequence, which only happens via a coupled commit).
    #
    # Keys are frozenset({col_a_rep, col_b_rep}); reps update via the
    # union-find, but a coupled-pair commit always passes its first arg
    # as the survivor (try_add_edge(col1, col2) absorbs col2 into col1),
    # so once a pair is committed its key stays stable.
    coupled_pair_state: Dict[frozenset, Dict] = {}    # {key: {'log_M_total': float}}
    # Size-2-component invalidation: a column that's already in a coupled
    # pair can't be in another. Maps current rep -> its partner rep.
    col_in_pair: Dict[int, int] = {}

    def get_pair_log_M_total(c1, c2) -> float:
        """Current running composite log-M for the (c1, c2) pair."""
        rc1, rc2 = find_col(c1), find_col(c2)
        key = frozenset((rc1, rc2)) if rc1 != rc2 else None
        if key is None:
            return 0.0
        st = coupled_pair_state.get(key)
        return st["log_M_total"] if st is not None else 0.0

    def commit_pair_cherry(c1, pc1, log_M_increment: float) -> None:
        """Called after a coupled commit succeeds. Updates the pair's
        composite log-M total and the col_in_pair index."""
        rc1, rpc1 = find_col(c1), find_col(pc1)
        key = frozenset((rc1, rpc1)) if rc1 != rpc1 else None
        if key is None:
            return
        st = coupled_pair_state.setdefault(
            key, {"log_M_total": 0.0})
        st["log_M_total"] += log_M_increment
        # Bind both reps to the pair. (Re-binding on extension is a no-op
        # since the same rep stays bound to the same partner rep.)
        col_in_pair[rc1] = rpc1
        col_in_pair[rpc1] = rc1

    def coupled_size_2_blocked(c1, pc1) -> bool:
        """True if either col is already in a different coupled pair, in
        which case committing (c1, pc1) would create a size-3+ component
        and is forbidden."""
        rc1, rpc1 = find_col(c1), find_col(pc1)
        partner_of_c1 = col_in_pair.get(rc1)
        partner_of_pc1 = col_in_pair.get(rpc1)
        if partner_of_c1 is not None and partner_of_c1 != rpc1:
            return True
        if partner_of_pc1 is not None and partner_of_pc1 != rc1:
            return True
        return False

    def coupled_score(sum_w: float, log_M_increment: float, c1, pc1) -> float:
        """Compose a coupled candidate's priority weight using the
        running composite likelihood of the (c1, pc1) pair plus this
        candidate's own cherry contribution."""
        composite = get_pair_log_M_total(c1, pc1) + log_M_increment
        if scoring_mode == "log_M":
            return sum_w * float(np.exp(0.5 * composite))
        elif scoring_mode == "posterior":
            M_val = float(np.exp(composite))
            p_coup = (prior_coup * M_val /
                          (prior_coup * M_val + (1.0 - prior_coup)))
            return sum_w + lambda_pair * p_coup
        else:
            raise ValueError(f"Unknown scoring_mode: {scoring_mode!r}")

    # ---- TGF dynamic-recalculation weight (verbatim from upstream) ------
    INVALID = -1e10

    def calc_tgf_weight(src_col, tgt_col):
        src_col = find_col(src_col); tgt_col = find_col(tgt_col)
        if src_col == tgt_col:
            return INVALID, INVALID
        c1pos = col_seqs.get(src_col, {})
        c2pos = col_seqs.get(tgt_col, {})
        sum_pmatch = 0.0; sum_pgap = 0.0
        for seq_i, pos_i in c1pos.items():
            for seq_j, pos_j in c2pos.items():
                if seq_i == seq_j:
                    return INVALID, INVALID
                if (seq_i, seq_j) in pair_posteriors:
                    post = pair_posteriors[(seq_i, seq_j)]
                    pmatch = float(post[pos_i - 1, pos_j - 1])
                    gp_i = float(gap_post_0[(seq_i, seq_j)][pos_i - 1])
                    gp_j = float(gap_post_1[(seq_i, seq_j)][pos_j - 1])
                elif (seq_j, seq_i) in pair_posteriors:
                    post = pair_posteriors[(seq_j, seq_i)]
                    pmatch = float(post[pos_j - 1, pos_i - 1])
                    gp_i = float(gap_post_1[(seq_j, seq_i)][pos_i - 1])
                    gp_j = float(gap_post_0[(seq_j, seq_i)][pos_j - 1])
                else:
                    continue
                sum_pmatch += 2 * pmatch
                sum_pgap += gp_i + gp_j
        if sum_pgap < 1e-10:
            if sum_pmatch > 0:
                return 1e10, 1e10
            return INVALID, INVALID
        return sum_pmatch / sum_pgap, sum_pmatch - sum_pgap

    # ---- Candidate generation -------------------------------------------
    edge_counter = 0
    edges: List[Tuple] = []

    # Single-edge candidates: indexed by sequence-pair, residue-pair.
    # We also keep an index into the per-pair structure for coupled
    # candidate generation.
    per_pair_anchors: Dict[Tuple[int, int, bool, Tuple[int, int]],
                            List[Tuple[int, int, float, float]]] = {}

    for si in range(n_seqs):
        for sj in range(si + 1, n_seqs):
            if (si, sj) in pair_posteriors:
                post = pair_posteriors[(si, sj)]
                # gp_i (length L_si, index by ri), gp_j (length L_sj, by rj).
                gp_i = gap_post_0[(si, sj)]; gp_j = gap_post_1[(si, sj)]
                key = (si, sj); transposed = False
            elif (sj, si) in pair_posteriors:
                post = np.asarray(pair_posteriors[(sj, si)]).T
                gp_i = gap_post_1[(sj, si)]; gp_j = gap_post_0[(sj, si)]
                key = (sj, si); transposed = True
            else:
                continue
            post = np.asarray(post)
            Li, Lj = post.shape
            anchors: List[Tuple[int, int, float, float]] = []
            for ri in range(Li):
                pgap_i = float(gp_i[ri])
                for rj in range(Lj):
                    pmatch = float(post[ri, rj])
                    if pmatch < 0.01:
                        continue
                    pgap_j = float(gp_j[rj])
                    denom = pgap_i + pgap_j
                    if denom < 1e-10:
                        weight = 1e10 if pmatch > 0 else 0.0
                    else:
                        weight = 2 * pmatch / denom
                    if (weight < edge_weight_threshold or
                            weight < gap_factor):
                        continue
                    col_i = seqpos_to_col[(si, ri + 1)]
                    col_j = seqpos_to_col[(sj, rj + 1)]
                    heapq.heappush(edges,
                                   (-weight, edge_counter, 'S',
                                    col_i, col_j, weight, None))
                    edge_counter += 1
                    if pmatch >= q_min:
                        # Store anchor with its base TGF weight as fourth slot.
                        anchors.append((ri, rj, pmatch, weight))
            per_pair_anchors[(si, sj, transposed, key)] = anchors

    # Coupled-pair candidates per pair (only if boost_states provided).
    n_coupled_added = 0
    if boost_states is not None:
        for (si, sj, transposed, key), anchors in per_pair_anchors.items():
            bs = boost_states.get(key, None)
            if bs is None or len(anchors) < 2:
                continue
            # If we accessed the pair via transposition, the boost state
            # uses the (key) ordering, but anchors are in (si, sj) ordering.
            # Map anchor (ri, rj) [si, sj coords] back to (rk, rl)
            # [key coords].
            def to_key_coords(ri, rj):
                if not transposed:
                    return ri, rj
                return rj, ri  # key = (sj, si): swap

            Lx_bs, Ly_bs = bs.gamma.shape[0], bs.gamma.shape[1]
            # For each anchor a, compute log_M field and pick the top-K
            # other anchors by |log M| with both above q_min.
            # To keep cost tractable we cap at max_pairs_per_anchor per
            # anchor.
            anchor_key_coords = [to_key_coords(ri, rj)
                                 for ri, rj, _, _ in anchors]
            # pre-extract all anchor coordinates as arrays for fast lookup
            anchor_kx = np.array([k[0] for k in anchor_key_coords],
                                 dtype=int)
            anchor_ky = np.array([k[1] for k in anchor_key_coords],
                                 dtype=int)
            anchor_q = np.array([a[2] for a in anchors], dtype=float)
            anchor_w = np.array([a[3] for a in anchors], dtype=float)
            for idx_a, (ri_a, rj_a, q_a, w_a) in enumerate(anchors):
                kx_a, ky_a = anchor_key_coords[idx_a]
                # log_M field for the anchor (only need it at other
                # anchor positions).
                # Build a fast 1-row computation:
                g_a = bs.gamma[kx_a, ky_a]                        # (K,)
                Xi_a = int(bs.x_seq[kx_a]); Yj_a = int(bs.y_seq[ky_a])
                # T[c', a', b'] = sum_c g_a[c] * J[c, c', Xi_a, a', Yj_a, b']
                J_slice = bs.joint_per_cp[:, :, Xi_a, :, Yj_a, :]
                T = np.einsum('c,cdpb->dpb', g_a, J_slice)        # (K, A, A)
                # Evaluate at the OTHER anchors
                Xb = bs.x_seq[anchor_kx]                          # (Na,)
                Yb = bs.y_seq[anchor_ky]                          # (Na,)
                # gamma at other anchors: (Na, K)
                gamma_b = bs.gamma[anchor_kx, anchor_ky]          # (Na, K)
                # T_at_anchors[idx_b, c'] = T[c', Xb, Yb]
                T_at = T[:, Xb, Yb].T                              # (Na, K)
                numer = np.sum(gamma_b * T_at, axis=1)             # (Na,)
                denom_field = bs.denom[kx_a, ky_a] * \
                    bs.denom[anchor_kx, anchor_ky]
                safe_num = np.clip(numer, 1e-300, None)
                safe_den = np.clip(denom_field, 1e-300, None)
                logM = np.log(safe_num) - np.log(safe_den)         # (Na,)
                # Filter: |logM| >= mu_min, ne self, q >= q_min.
                mask = (np.abs(logM) >= mu_min) & (anchor_q >= q_min)
                mask[idx_a] = False
                idxs = np.where(mask)[0]
                if idxs.size == 0:
                    continue
                if idxs.size > max_pairs_per_anchor:
                    # keep top-K by |logM|
                    order = np.argsort(-np.abs(logM[idxs]))
                    idxs = idxs[order[:max_pairs_per_anchor]]
                # Push coupled-pair entries.
                col_i_a = seqpos_to_col[(si, ri_a + 1)]
                col_j_a = seqpos_to_col[(sj, rj_a + 1)]
                base_w_a = float(w_a)
                for idx_b in idxs:
                    ri_b, rj_b, q_b, w_b = anchors[idx_b]
                    if (ri_a, rj_a) == (ri_b, rj_b):
                        continue
                    # Avoid double-pushing: only when (anchor_a < anchor_b)
                    # in lex order on (ri, rj).
                    if (ri_a, rj_a) >= (ri_b, rj_b):
                        continue
                    base_w_b = float(w_b)
                    sum_w = base_w_a + base_w_b
                    col_i_b = seqpos_to_col[(si, ri_b + 1)]
                    col_j_b = seqpos_to_col[(sj, rj_b + 1)]
                    # Score with composite log-M (initially 0; pair_state
                    # is empty at enqueue time, so this is just the
                    # candidate's own cherry contribution).
                    joint_w = coupled_score(sum_w, float(logM[idx_b]),
                                                 col_i_a, col_i_b)
                    if (joint_w < edge_weight_threshold or
                            joint_w < gap_factor):
                        continue
                    # Coupled entry: ('C', neg_priority, counter,
                    #                  col_i_a, col_j_a, base_w_a,
                    #                  (col_i_b, col_j_b, base_w_b,
                    #                   logM_val))
                    heapq.heappush(edges,
                                   (-joint_w, edge_counter, 'C',
                                    col_i_a, col_j_a, base_w_a,
                                    (col_i_b, col_j_b, base_w_b,
                                     float(logM[idx_b]))))
                    edge_counter += 1
                    n_coupled_added += 1

    if verbose:
        print(f"  Coupled AMAP: {len(edges)} candidates "
              f"({n_coupled_added} coupled), starting greedy merge")

    # ---- Pop loop --------------------------------------------------------
    n_merged = 0
    n_coupled_committed = 0
    n_coupled_demoted = 0

    while edges:
        item = heapq.heappop(edges)
        neg_w = item[0]; kind = item[2]
        col1 = item[3]; col2 = item[4]; base_w = item[5]
        partner = item[6]

        c1 = find_col(col1); c2 = find_col(col2)
        if c1 == c2:
            # First half already merged
            if kind == 'C' and partner is not None:
                # Demote: try the partner alone.
                pcol_i, pcol_j, pbase_w, _ = partner
                pc1 = find_col(pcol_i); pc2 = find_col(pcol_j)
                if pc1 != pc2 and pc1 in live_cols and pc2 in live_cols:
                    nw, _ = calc_tgf_weight(pc1, pc2)
                    if nw > -1e9 and nw >= edge_weight_threshold and \
                            nw >= gap_factor:
                        heapq.heappush(edges,
                                       (-nw, edge_counter, 'S',
                                        pc1, pc2, nw, None))
                        edge_counter += 1
                        n_coupled_demoted += 1
            continue
        if c1 not in live_cols or c2 not in live_cols:
            continue

        new_w, _ = calc_tgf_weight(c1, c2)
        if new_w <= -1e9:
            continue
        if new_w < edge_weight_threshold or new_w < gap_factor:
            continue

        # For coupled candidates, also recompute partner TGF for the
        # joint priority decision, AND look up the current composite
        # log-M of the (column1, column2) pair (which may have grown
        # since this candidate was enqueued, due to other coupled
        # commits extending the pair).
        if kind == 'C' and partner is not None:
            pcol_i, pcol_j, pbase_w, logM_val = partner
            pc1 = find_col(pcol_i); pc2 = find_col(pcol_j)
            if (pc1 == pc2 or pc1 not in live_cols or
                    pc2 not in live_cols):
                # Partner already invalid: demote to single-edge.
                heapq.heappush(edges,
                               (-new_w, edge_counter, 'S',
                                c1, c2, new_w, None))
                edge_counter += 1
                n_coupled_demoted += 1
                continue
            # Size-2-component check: forbid coupling if either column
            # is already in a different coupled pair.
            if coupled_size_2_blocked(c1, pc1):
                heapq.heappush(edges,
                               (-new_w, edge_counter, 'S',
                                c1, c2, new_w, None))
                edge_counter += 1
                n_coupled_demoted += 1
                continue
            new_pw, _ = calc_tgf_weight(pc1, pc2)
            if new_pw <= -1e9 or new_pw < edge_weight_threshold or \
                    new_pw < gap_factor:
                heapq.heappush(edges,
                               (-new_w, edge_counter, 'S',
                                c1, c2, new_w, None))
                edge_counter += 1
                n_coupled_demoted += 1
                continue
            # Recompute joint priority with current TGF estimates AND
            # the current composite log-M from any prior coupled commits
            # that extended this same (col1, col2) pair.
            joint_w_now = coupled_score(new_w + new_pw, logM_val, c1, pc1)
            # Standard dynamic-recalc check: re-insert if priority
            # dropped below the next item.
            if edges and joint_w_now < -edges[0][0]:
                heapq.heappush(edges,
                               (-joint_w_now, edge_counter, 'C',
                                c1, c2, new_w,
                                (pc1, pc2, new_pw, logM_val)))
                edge_counter += 1
                continue
            # Commit both halves atomically.
            r1 = try_add_edge(c1, c2)
            if r1 != 0:
                # First half rejected by DAG/conflict. Try partner alone.
                pc1_now = find_col(pcol_i); pc2_now = find_col(pcol_j)
                if pc1_now != pc2_now and pc1_now in live_cols and \
                        pc2_now in live_cols:
                    r2 = try_add_edge(pc1_now, pc2_now)
                    if r2 == 0:
                        n_merged += 1
                continue
            n_merged += 1
            pc1_now = find_col(pcol_i); pc2_now = find_col(pcol_j)
            if pc1_now == pc2_now:
                continue
            if pc1_now not in live_cols or pc2_now not in live_cols:
                continue
            r2 = try_add_edge(pc1_now, pc2_now)
            if r2 == 0:
                n_merged += 1
                n_coupled_committed += 1
                # Update the running composite log-M for this pair, and
                # bind both columns into a coupled pair (size-2 lock).
                # The merge of (c1, c2) results in c1 as the survivor,
                # similarly (pc1_now, pc2_now) -> pc1_now.
                commit_pair_cherry(c1, pc1_now, logM_val)
            continue

        # Single-edge candidate: standard dynamic-recalc check.
        if edges and new_w < -edges[0][0]:
            heapq.heappush(edges,
                           (-new_w, edge_counter, 'S',
                            c1, c2, new_w, None))
            edge_counter += 1
            continue
        result = try_add_edge(c1, c2)
        if result == 0:
            n_merged += 1

    if verbose:
        print(f"  Coupled AMAP: merged {n_merged} edges "
              f"({n_coupled_committed} as part of a coupled pair, "
              f"{n_coupled_demoted} demoted), "
              f"{len(live_cols)} columns remain")

    # ---- Topological sort of remaining columns --------------------------
    adj = {c: set() for c in live_cols}
    in_degree = {c: 0 for c in live_cols}
    for c in live_cols:
        for seq, pos in col_seqs[c].items():
            nxt_key = (seq, pos + 1)
            if nxt_key in seqpos_to_col:
                sc = find_col(seqpos_to_col[nxt_key])
                if sc in live_cols and sc != c and sc not in adj[c]:
                    adj[c].add(sc)
                    in_degree[sc] += 1
    topo_queue = []
    for c in live_cols:
        if in_degree[c] == 0:
            heapq.heappush(topo_queue, (col_index[c], c))
    topo_order = []
    while topo_queue:
        _, node = heapq.heappop(topo_queue)
        topo_order.append(node)
        for succ in adj[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                heapq.heappush(topo_queue, (col_index[succ], succ))
    if len(topo_order) != len(live_cols):
        topo_order = sorted(live_cols, key=lambda c: col_index[c])
    col_remap = {c: i for i, c in enumerate(topo_order)}

    col_assignments = []
    for si in range(n_seqs):
        ca = np.zeros(seq_lengths[si], dtype=int)
        for k in range(seq_lengths[si]):
            pos = k + 1
            live_c = find_col(seqpos_to_col[(si, pos)])
            ca[k] = col_remap[live_c]
        col_assignments.append(ca)
    return col_assignments
