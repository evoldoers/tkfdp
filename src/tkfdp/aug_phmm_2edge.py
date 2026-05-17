"""2-edge memory-augmented Pair HMM for the size-{0, 1, 2}-truncated
SCFG posterior. Generalises ``src/tkfdp/aug_phmm.py`` to allow up to
two coupled column-pairs per alignment.

The principled prior is the size-{0, 1, 2}-truncated Ewens partition
(equivalently a CRP-truncated stick on coupled-edge endpoints).
Under our static-tag-space approximation (per-cell encoding of the
spawn weight epsilon = 1/alpha_z, applied independently at every
Match cell where a spawn happens), this gives:
  P(0 edges) propto 1
  P(1 edge)  propto eps  (with combinatorial factor: # ways to place
                          the spawn + close cells)
  P(2 edges) propto eps^2

This is an approximation to the principled infinite-HMM CRP rule
P(spawn|m) = alpha_z / (m + alpha_z) where m counts matches consumed;
see ``main.tex`` sec:infinite-hmm for the exact form.

----------------------------------------------------------------------
Tag layout (n_tags = 81002 for A=20)
----------------------------------------------------------------------

Conceptually the tag is the multiset of "in-flight left endpoint AAs"
plus a "closures-so-far" counter in {0, 1, 2}. The flat layout is:

  range            count        meaning
  -----            -----        -------
  [0, 1)           1            no_edge:        0 spawns, 0 closes
  [1, 1+A^2)       400          singletons:     1 spawn, 0 closes,
                                                in-flight (a, b)
  [401, 80601)     A^2*(A^2+1)/2 = 80200
                                pairs:          2 spawns, 0 closes,
                                                in-flight unordered
                                                multiset {(a1,b1),
                                                          (a2,b2)}
  [80601, 81001)   400          closed_singles: 2 spawns, 1 close,
                                                1 in-flight (a, b)
  [81001, 81002)   1            closed_done:    >= 1 closes; either
                                                (1 spawn 1 close) or
                                                (2 spawns 2 closes);
                                                terminal-eligible

The "no_edge" and "closed_done" tags are the ONLY terminating tags
(orphan in-flight endpoints are excluded from L_exact).

Pair tag indexing (multiset of two AA-pairs):
  Each AA-pair (a, b) is mapped to a flat index ab = a*A + b in
  [0, A^2). For a multiset {(a1,b1), (a2,b2)} with flat indices
  i1, i2 (sorted so i_low <= i_high), the multiset index in
  [0, A^2*(A^2+1)/2) is:
        pair_mset_idx(i_low, i_high) = i_high*(i_high+1)//2 + i_low
  This is the standard "lower-triangle including diagonal" indexing
  on an A^2 x A^2 matrix. The pair-tag is then PAIR_BASE +
  pair_mset_idx(i_low, i_high).

----------------------------------------------------------------------
Match-cell transitions (current AAs (c, d))
----------------------------------------------------------------------

At a Match cell at position (i, j) with current AAs (c, d) =
(X[i-1], Y[j-1]):

  carry-through (any tag):                weight 1
  spawn from no_edge to single{(c,d)}:    weight eps
  spawn from single{X} to pair{X,(c,d)}:  weight eps  (multiset
                                                       construction)
  close from single{X} to closed_done:    weight M(X; c, d)
  close from pair{X, Y} to                BOTH branches fire:
        closed_single{Y}                  - weight M(X; c, d) to
                                            closed_single{Y}
                                          - weight M(Y; c, d) to
                                            closed_single{X}
  close from closed_single{X} to          weight M(X; c, d)
        closed_done

For the same-element pair multiset {X, X}, BOTH closure branches
still fire and BOTH go to closed_single{X}, giving total weight
2 * M(X; c, d) * step[pair{X,X}] to closed_single{X}. This is the
correct multiplicity for the SCFG: the multiset {X, X} represents
two physically-distinguishable in-flight edges (spawned at different
positions but with the same AAs); either one may close at this Match
cell, giving two distinct SCFG configurations.

NOTE: there is no "spawn from closed_single" transition (the
2-edge budget is exhausted once we've spawned twice). And there is
no "single to closed_single" transition (closing the only
in-flight edge from single just goes to closed_done; the
closed_single tag arises only from the pair to closed_single
closure step).

The carry-through interpretation of singleton{X} (without firing a
close) means the single edge is still in-flight and the alignment
hasn't yet committed to its right endpoint; the carry-through of
pair{X, Y} is the analogous "neither edge has fired its right
endpoint yet". Both can carry through without ever firing close,
but then the alignment terminates at a non-terminating tag (orphan
in-flight) and is excluded from L_exact.

----------------------------------------------------------------------
Cost
----------------------------------------------------------------------

Tag space is ~81k vs 1-edge's 402 (~200x more memory). Forward DP
table is (Lx_pad+1) * (Ly_pad+1) * 5 * 81002. At L=100, fp32, that's
~16 GB -- memory tight. For development use small L (<= 30); for
larger L the antidiagonal stretch goal would help.

The dominant per-Match-cell op is the pair-to-closed_single closure
contraction at O(A^4) ops/cell (vectorised as a (A^2, A^2) matmul-
like operation).
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Pull tkfmixdom's TKF92 Pair HMM builder + state codes (do not modify).
TKFMIXDOM_ROOT = Path.home() / "tkf-mixdom" / "python"
if str(TKFMIXDOM_ROOT) not in sys.path:
    sys.path.insert(0, str(TKFMIXDOM_ROOT))
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm  # noqa: E402
from tkfmixdom.jax.dp.hmm import (                                  # noqa: E402
    pair_hmm_emissions, _pad_to_bin, _pad_seq, _emit_mask,
    _find_e_idx, NEG_INF,
)
from tkfmixdom.jax.core.params import S, M, I, D, E                  # noqa: E402

from .aug_phmm import build_M_tensor_aa_marginal                     # noqa: E402

A = 20  # amino acid alphabet
A2 = A * A                                  # 400
N_PAIRS_MSET = A2 * (A2 + 1) // 2           # 80200

# Tag offsets:
TAG_NO_EDGE = 0
SINGLE_BASE = 1
PAIR_BASE = SINGLE_BASE + A2                # 401
CLOSED_SINGLE_BASE = PAIR_BASE + N_PAIRS_MSET   # 80601
TAG_CLOSED_DONE = CLOSED_SINGLE_BASE + A2       # 81001
N_TAGS = TAG_CLOSED_DONE + 1                # 81002


# ---------------------------------------------------------------------------
# Tag layout helpers (numpy / pure python -- used in brute force and to
# precompute index tables for the DP).
# ---------------------------------------------------------------------------


def single_tag(a: int, b: int) -> int:
    """Encode an in-flight singleton with AAs (a, b)."""
    return SINGLE_BASE + a * A + b


def closed_single_tag(a: int, b: int) -> int:
    """Encode a 1-closed + in-flight (a, b) singleton state."""
    return CLOSED_SINGLE_BASE + a * A + b


def pair_mset_idx(i_low: int, i_high: int) -> int:
    """Lower-triangle (incl. diag) index for unordered multiset of two
    flat AA-pair indices in [0, A^2). Requires i_low <= i_high.
    """
    assert 0 <= i_low <= i_high < A2
    return (i_high * (i_high + 1)) // 2 + i_low


def pair_tag_from_aa(a1: int, b1: int, a2: int, b2: int) -> int:
    """Encode an unordered multiset {(a1, b1), (a2, b2)} as a tag id."""
    i1 = a1 * A + b1
    i2 = a2 * A + b2
    if i1 <= i2:
        return PAIR_BASE + pair_mset_idx(i1, i2)
    return PAIR_BASE + pair_mset_idx(i2, i1)


def decode_tag(t: int) -> Tuple[str, tuple]:
    """Decode a tag id into ('no_edge'|'single'|'pair'|'closed_single'
    |'closed_done', payload).

    payload is:
      - () for no_edge / closed_done
      - (a, b) for single / closed_single
      - ((a1, b1), (a2, b2)) for pair (in canonical order i_low <= i_high)
    """
    if t == TAG_NO_EDGE:
        return ('no_edge', ())
    if t == TAG_CLOSED_DONE:
        return ('closed_done', ())
    if SINGLE_BASE <= t < PAIR_BASE:
        ab = t - SINGLE_BASE
        return ('single', (ab // A, ab % A))
    if CLOSED_SINGLE_BASE <= t < TAG_CLOSED_DONE:
        ab = t - CLOSED_SINGLE_BASE
        return ('closed_single', (ab // A, ab % A))
    if PAIR_BASE <= t < CLOSED_SINGLE_BASE:
        idx = t - PAIR_BASE
        # Invert pair_mset_idx: idx = i_high*(i_high+1)/2 + i_low
        # so i_high = floor((sqrt(8*idx + 1) - 1) / 2)... let's just
        # search.
        # i_high satisfies i_high*(i_high+1)/2 <= idx < (i_high+1)*(i_high+2)/2
        i_high = int((np.sqrt(8 * idx + 1) - 1) / 2)
        # adjust for floating-point edge cases
        while (i_high + 1) * (i_high + 2) // 2 <= idx:
            i_high += 1
        while i_high * (i_high + 1) // 2 > idx:
            i_high -= 1
        i_low = idx - i_high * (i_high + 1) // 2
        return ('pair', ((i_low // A, i_low % A), (i_high // A, i_high % A)))
    raise ValueError(f"unknown tag id {t}")


# Index tables, precomputed once (small, usable inside the JIT'd DP via
# jnp.asarray). All shapes are A x A (and A^2 x A^2) numpy arrays.
def _build_index_tables() -> dict:
    """Precompute index tables for the Forward DP closure step.

    Returns a dict of np.ndarray with:
      single_idx_table:  (A, A) -> single_tag(a, b)
      closed_single_idx_table: (A, A) -> closed_single_tag(a, b)
      pair_tag_table_AA: (A^2, A^2) -> pair tag id for ordered (i, j)
                          flat indices (we sort internally)
      pair_canon_table:  (N_PAIRS_MSET, 2) -> two flat AA-pair indices
                          (i_low, i_high) for each mset index
      pair_low_AA:       (N_PAIRS_MSET, 2) -> (a, b) for the i_low element
      pair_high_AA:      (N_PAIRS_MSET, 2) -> (a, b) for the i_high element
    """
    sing = np.zeros((A, A), dtype=np.int32)
    csing = np.zeros((A, A), dtype=np.int32)
    for a in range(A):
        for b in range(A):
            sing[a, b] = single_tag(a, b)
            csing[a, b] = closed_single_tag(a, b)
    # Pair-tag table for ordered AA-pair indices (i, j) -> pair tag.
    pt = np.zeros((A2, A2), dtype=np.int32)
    canon = np.zeros((N_PAIRS_MSET, 2), dtype=np.int32)
    for i in range(A2):
        for j in range(A2):
            lo, hi = (i, j) if i <= j else (j, i)
            pt[i, j] = PAIR_BASE + pair_mset_idx(lo, hi)
    # Build canon table by iterating multiset indices.
    for i_high in range(A2):
        for i_low in range(i_high + 1):
            idx = pair_mset_idx(i_low, i_high)
            canon[idx, 0] = i_low
            canon[idx, 1] = i_high
    pair_low_AA = np.stack([canon[:, 0] // A, canon[:, 0] % A], axis=1)
    pair_high_AA = np.stack([canon[:, 1] // A, canon[:, 1] % A], axis=1)
    return {
        'single_idx_table': sing,
        'closed_single_idx_table': csing,
        'pair_tag_table_AA': pt,
        'pair_canon': canon,
        'pair_low_AA': pair_low_AA,
        'pair_high_AA': pair_high_AA,
    }


_INDEX_TABLES_CACHE = None


def _index_tables() -> dict:
    global _INDEX_TABLES_CACHE
    if _INDEX_TABLES_CACHE is None:
        _INDEX_TABLES_CACHE = _build_index_tables()
    return _INDEX_TABLES_CACHE


# ---------------------------------------------------------------------------
# Forward DP step at a Match cell.
# ---------------------------------------------------------------------------
#
# Given the predecessor cell's per-tag step output for the M-state output:
#   step_M[t] = LSE_k (prev_diag[k, t] + log_trans[k, M])    for t in [0, N_TAGS)
# and the M-emission log P(c, d), we must:
#   1. Output the carry-through: cell_M[t] = step_M[t] + emit_M(c, d)
#      for ALL t.
#   2. Add augmented contributions to specific output tags:
#      - cell_M[single_tag(c, d)] += eps * step_M[no_edge] * emit_M(c, d)
#      - For each (a, b): cell_M[pair_tag_from_aa(a, b, c, d)] +=
#            eps * step_M[single_tag(a, b)] * emit_M(c, d)
#      - cell_M[closed_done] += sum_{(a,b)} M(a, b; c, d) *
#            step_M[single_tag(a, b)] * emit_M(c, d)
#      - For each pair tag T = pair{X, Y}: BOTH branches always fire:
#            contribute M(X; c, d) * step_M[T] * emit_M(c, d)
#                       to cell_M[closed_single_tag(Y)],
#            contribute M(Y; c, d) * step_M[T] * emit_M(c, d)
#                       to cell_M[closed_single_tag(X)].
#          For same-element {X, X}: both branches contribute equal
#          weight M(X; c, d) * step_M[T] * emit_M(c, d) to the SAME
#          target cell_M[closed_single_tag(X)], summing to multiplicity
#          2 -- the multiset {X, X} represents 2 physically-
#          distinguishable in-flight edges.
#      - cell_M[closed_done] += sum_{(a,b)} M(a, b; c, d) *
#            step_M[closed_single_tag(a, b)] * emit_M(c, d)
#
# We vectorise the pair-closure step (the dominant op) by iterating
# over the multiset's (i_low, i_high) decomposition: for each pair_tag
# T with elements (low, high), branch A contributes to closed_single
# {high} (closing low) and branch B to closed_single{low} (closing
# high). Both branches always fire. The same-element multiplicity-2 is
# implicit: when low == high, branches A and B BOTH contribute to
# closed_single{low} = closed_single{high}. The contributions are
# segment-LSE'd into the closed_single output range (size A^2) using
# jax.ops.segment_max + segment_sum for stable scatter-LSE on the GPU
# (since multiple pair tags may map to the same closed_single target).

# ---------------------------------------------------------------------------
# JAX core: forward DP.
# ---------------------------------------------------------------------------


def _build_jax_index_tables():
    tab = _index_tables()
    return {
        'single_idx': jnp.asarray(tab['single_idx_table']),
        'closed_single_idx': jnp.asarray(tab['closed_single_idx_table']),
        'pair_tag_AA': jnp.asarray(tab['pair_tag_table_AA']),
        'pair_canon': jnp.asarray(tab['pair_canon']),
        'pair_low_AA': jnp.asarray(tab['pair_low_AA']),
        'pair_high_AA': jnp.asarray(tab['pair_high_AA']),
    }


def _aug2_match_extras(cell, step_M_per_tag, emit_M_cd, c_idx, d_idx,
                        log_M_tensor, log_eps,
                        single_idx_table, closed_single_idx_table,
                        pair_tag_table_AA, pair_low_AA, pair_high_AA):
    """Apply the augmented Match-cell transitions.

    Args:
      cell: (5, N_TAGS) the candidate-cell output, ALREADY containing
        the carry-through (step + emit) for ALL tags. We will only
        write to cell[M, :].
      step_M_per_tag: (N_TAGS,) the per-tag M-output of the standard
        forward step from the diagonal predecessor: step_M[t] =
        LSE_k (prev_diag[k, t] + log_trans[k, M]). NOTE: this is
        WITHOUT the emit_M factor; we apply emit_M separately to the
        augmented contributions.
      emit_M_cd: scalar log P_match(c, d).
      c_idx, d_idx: AA indices at this Match cell (0-based).
      log_M_tensor: (A, A, A, A) log-M tensor; log_M_tensor[a, b, c, d]
        is the log-coupling-boost for left-edge AAs (a, b) closing at
        right-edge AAs (c, d).
      log_eps: scalar log(eps) = -log(alpha_z).
      single_idx_table: (A, A) -> single tag id (jnp).
      closed_single_idx_table: (A, A) -> closed single tag id (jnp).
      pair_tag_table_AA: (A^2, A^2) -> pair tag id, indexed by ordered
        (X_flat, Y_flat). Self-canonicalised internally.
      pair_low_AA: (N_PAIRS_MSET, 2) low-element AA (a, b) per pair tag.
      pair_high_AA: (N_PAIRS_MSET, 2) high-element AA per pair tag.

    Returns: updated (5, N_TAGS) cell.
    """
    cd_single_tag = single_idx_table[c_idx, d_idx]
    log_M_at_cd = log_M_tensor[:, :, c_idx, d_idx]  # (A, A) M(., .; c, d)

    # 1. Spawn no_edge -> single{(c, d)}.
    log_spawn_no_edge = log_eps + step_M_per_tag[TAG_NO_EDGE] + emit_M_cd
    new_M = cell[M, :]
    new_M = new_M.at[cd_single_tag].set(
        jnp.logaddexp(new_M[cd_single_tag], log_spawn_no_edge))

    # 2. Spawn single{(a, b)} -> pair{(a, b), (c, d)}.
    # For each (a, b), pair_tag = pair_tag_table_AA[a*A+b, c*A+d]
    cd_flat = c_idx * A + d_idx
    pair_tags_for_spawn = pair_tag_table_AA[:, cd_flat]  # (A^2,) pair tag for each (a, b)
    # Get step_M_per_tag at single_tag(a, b) for all (a, b).
    single_tags_flat = single_idx_table.reshape(A2)  # (A^2,)
    step_M_at_single = step_M_per_tag[single_tags_flat]  # (A^2,)
    log_spawn_single = log_eps + step_M_at_single + emit_M_cd  # (A^2,)
    # Scatter-add to pair tags. Multiple (a, b) values may map to
    # different pair tags, but a single (a, b) maps to exactly one
    # pair tag (no collision). Use segment_sum-style accumulation via
    # logaddexp.
    # Build a delta vector of shape (N_TAGS,) initialised to NEG_INF
    # and scatter the contributions. Then logaddexp into new_M.
    delta = jnp.full((N_TAGS,), NEG_INF)
    # NOTE: pair_tags_for_spawn may have duplicate target tags for
    # different (a, b)? Check: pair_tag(a, b, c, d) is determined by
    # the unordered multiset {(a, b), (c, d)}. Different (a, b) values
    # give different multisets (since (c, d) is fixed). So unique.
    # However the scatter via .at[].set() with duplicates would
    # behave as last-write-wins; we use the segment-sum-style
    # logsumexp via .add() for safety.
    # In log-space, we replace empty cells with NEG_INF and use
    # jnp.logaddexp on the sum; for unique indices this is equivalent
    # to .at[].set().
    delta = delta.at[pair_tags_for_spawn].set(log_spawn_single)
    new_M = jnp.logaddexp(new_M, delta)

    # 3. Close single{(a, b)} -> closed_done.
    # contribution to closed_done: LSE_(a,b) M(a,b;c,d) *
    #     step_M[single_tag(a,b)] * emit_M(c, d)
    log_close_from_single = (log_M_at_cd.reshape(A2)
                              + step_M_at_single
                              + emit_M_cd)  # (A^2,)
    # Sum over (a, b):
    log_close_from_single_lse = jax.nn.logsumexp(log_close_from_single)
    new_M = new_M.at[TAG_CLOSED_DONE].set(
        jnp.logaddexp(new_M[TAG_CLOSED_DONE], log_close_from_single_lse))

    # 4. Close pair{X, Y} -> closed_single{Y} (closing X) and -> closed_single{X}
    # (closing Y). For X = Y: only one branch.
    # Approach: iterate over the (i_low, i_high) decomposition of each
    # pair tag.
    pair_step = step_M_per_tag[PAIR_BASE:PAIR_BASE + N_PAIRS_MSET]  # (N_PAIRS_MSET,)
    # AAs at low/high elements: pair_low_AA / pair_high_AA both (N_PAIRS_MSET, 2)
    a_low = pair_low_AA[:, 0]  # (N_PAIRS_MSET,)
    b_low = pair_low_AA[:, 1]
    a_high = pair_high_AA[:, 0]
    b_high = pair_high_AA[:, 1]
    log_M_low = log_M_at_cd[a_low, b_low]   # (N_PAIRS_MSET,)
    log_M_high = log_M_at_cd[a_high, b_high]
    # closed_single tag IDs:
    cs_tag_low = closed_single_idx_table[a_low, b_low]   # (N_PAIRS_MSET,)
    cs_tag_high = closed_single_idx_table[a_high, b_high]

    # Branch A: close X=low, leave Y=high -> closed_single{high}, weight M(low; c, d).
    # Branch B: close Y=high, leave X=low -> closed_single{low}, weight M(high; c, d).
    # Both branches always fire (including for the same-element {X, X}
    # multiset, where both branches contribute equal weight M(X; c, d)
    # to closed_single{X} -- this is the correct multiplicity-2 since
    # the multiset {X, X} represents 2 distinguishable spawn events
    # at different positions, either of which can close at this cell).
    contrib_A = log_M_low + pair_step + emit_M_cd  # (N_PAIRS_MSET,)
    contrib_B = log_M_high + pair_step + emit_M_cd

    # Scatter both contributions into the closed_single output range.
    # For Branch A: target tag = cs_tag_high.
    # For Branch B: target tag = cs_tag_low.
    # Multiple pair tags may target the SAME closed_single tag; we need
    # logsumexp accumulation, not set-overwrite. Use segment-sum-style
    # via jax.ops.segment_logsumexp through manual log-add.
    # Approach: build delta over the closed_single range only (size A^2),
    # then merge into new_M.
    delta_cs = jnp.full((A2,), NEG_INF)
    # Branch A: target index in [0, A^2) is cs_tag_high - CLOSED_SINGLE_BASE.
    cs_idx_A = cs_tag_high - CLOSED_SINGLE_BASE  # (N_PAIRS_MSET,)
    cs_idx_B = cs_tag_low - CLOSED_SINGLE_BASE   # (N_PAIRS_MSET,)
    # Use a manual scatter-LSE: logsumexp for each cs target index over
    # all pair tags that map to it. We'll vmap a per-target gather:
    # for cs_idx in 0..A^2, sum_{pt: cs_idx_A[pt] == cs_idx} exp(contrib_A[pt])
    #                       + sum_{pt: cs_idx_B[pt] == cs_idx} exp(contrib_B[pt])
    # Use segment_max-like pattern via segment-style.

    # Implement as: build a (A^2 + 1)-bucket scatter via segment_logsumexp.
    # We'll do it manually with jax.ops.segment_max + logsumexp trick;
    # or use a (A^2 x N_PAIRS_MSET) one-hot matmul (160k x 80k = 12.8 GB
    # of fp32; too big). Alternative: use scatter add in linear space
    # with stable rescaling.
    #
    # Cleanest approach: use jax.ops.segment_sum after rescaling by
    # max. Compute max over relevant pair_tags per cs_idx (a max-reduce
    # via segment_max), then logsumexp via:
    #   M[cs] = max_{pt: target=cs} contrib[pt]
    #   delta_cs[cs] = M[cs] + log(sum_{pt} exp(contrib[pt] - M[cs]))
    #
    # For 1 cs target receiving contributions from <= 2*A^2 pair_tags
    # via Branch A or B, we can use segment_max + segment_sum.

    # Combine the two branches into one (target_idx, log_value) list.
    target_idx = jnp.concatenate([cs_idx_A, cs_idx_B])  # (2*N_PAIRS_MSET,)
    log_vals = jnp.concatenate([contrib_A, contrib_B])

    # Numerical-stable segment LSE:
    # 1. compute per-segment max using jax.ops.segment_max.
    seg_max = jax.ops.segment_max(log_vals, target_idx, num_segments=A2)
    # If a segment has no entries, segment_max returns -inf-like;
    # protect from -inf - -inf = NaN.
    seg_max_safe = jnp.where(jnp.isfinite(seg_max), seg_max, 0.0)
    # 2. exp shifted, segment_sum, log + add back.
    shifted = jnp.exp(log_vals - seg_max_safe[target_idx])
    seg_sum = jax.ops.segment_sum(shifted, target_idx, num_segments=A2)
    seg_lse = jnp.where(seg_sum > 0,
                         jnp.log(jnp.maximum(seg_sum, 1e-300)) + seg_max_safe,
                         NEG_INF)
    delta_cs = seg_lse  # (A^2,)
    # Merge into new_M at the closed_single range.
    cs_range = jnp.arange(CLOSED_SINGLE_BASE, CLOSED_SINGLE_BASE + A2)
    new_M = new_M.at[cs_range].set(
        jnp.logaddexp(new_M[cs_range], delta_cs))

    # 5. Close closed_single{(a, b)} -> closed_done.
    cs_step = step_M_per_tag[CLOSED_SINGLE_BASE:CLOSED_SINGLE_BASE + A2]  # (A^2,)
    log_close_from_cs = (log_M_at_cd.reshape(A2)
                          + cs_step
                          + emit_M_cd)  # (A^2,)
    log_close_from_cs_lse = jax.nn.logsumexp(log_close_from_cs)
    new_M = new_M.at[TAG_CLOSED_DONE].set(
        jnp.logaddexp(new_M[TAG_CLOSED_DONE], log_close_from_cs_lse))

    return cell.at[M, :].set(new_M)


def _aug2_forward_core(log_trans, state_types, emit, log_M_tensor,
                       x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                       single_idx_table, closed_single_idx_table,
                       pair_tag_table_AA, pair_low_AA, pair_high_AA):
    """Augmented Forward DP for the 2-edge model. Returns alpha
    of shape (Lx_pad+1, Ly_pad+1, 5, N_TAGS) in log space."""
    ns = log_trans.shape[0]
    is_M_st = (state_types == M)
    is_I_st = (state_types == I)
    is_D_st = (state_types == D)

    def step_from(prev):
        # prev: (5, N_TAGS). Output: (5, N_TAGS) where
        #   out[s', t] = LSE_k (prev[k, t] + log_trans[k, s'])
        return jax.nn.logsumexp(
            prev[:, None, :] + log_trans[:, :, None], axis=0)

    def apply_emit(step, emit_cell):
        return step + emit_cell[:, None]

    def apply_match(cell, match_step_per_tag, emit_M_cd, c_idx, d_idx):
        return _aug2_match_extras(
            cell, match_step_per_tag, emit_M_cd, c_idx, d_idx,
            log_M_tensor, log_eps,
            single_idx_table, closed_single_idx_table,
            pair_tag_table_AA, pair_low_AA, pair_high_AA)

    # Initialize alpha[0, 0, S, no_edge] = 0; everything else NEG_INF.
    cell00 = jnp.full((ns, N_TAGS), NEG_INF)
    cell00 = cell00.at[S, TAG_NO_EDGE].set(0.0)

    # Row 0.
    def row0_step(prev_cell, j):
        step = step_from(prev_cell)
        emit_cell = emit[0, j]
        full = apply_emit(step, emit_cell)
        cell = jnp.where(is_I_st[:, None], full, NEG_INF)
        return cell, cell

    _, row0_rest = jax.lax.scan(
        row0_step, cell00, jnp.arange(1, Ly_pad + 1))
    row0 = jnp.concatenate([cell00[None], row0_rest], axis=0)

    def row_step(prev_row, i):
        prev_cell00 = prev_row[0]
        step0 = step_from(prev_cell00)
        emit0 = emit[i, 0]
        full0 = apply_emit(step0, emit0)
        cell0 = jnp.where(is_D_st[:, None], full0, NEG_INF)

        c_idx = x_pad[i - 1]

        def col_step(prev_cell, j):
            prev_diag = prev_row[j - 1]
            prev_left = prev_cell
            prev_up = prev_row[j]

            step_M_pred = step_from(prev_diag)
            step_I_pred = step_from(prev_left)
            step_D_pred = step_from(prev_up)

            emit_cell = emit[i, j]
            full_M = apply_emit(step_M_pred, emit_cell)
            full_I = apply_emit(step_I_pred, emit_cell)
            full_D = apply_emit(step_D_pred, emit_cell)

            cell = jnp.where(is_M_st[:, None], full_M,
                     jnp.where(is_I_st[:, None], full_I,
                       jnp.where(is_D_st[:, None], full_D, NEG_INF)))
            d_idx = y_pad[j - 1]
            emit_M_cd = emit_cell[M]
            match_step_per_tag = step_M_pred[M, :]
            cell = apply_match(cell, match_step_per_tag, emit_M_cd,
                               c_idx, d_idx)
            return cell, cell

        _, row_rest = jax.lax.scan(
            col_step, cell0, jnp.arange(1, Ly_pad + 1))
        curr_row = jnp.concatenate([cell0[None], row_rest], axis=0)
        return curr_row, curr_row

    _, all_rows = jax.lax.scan(
        row_step, row0, jnp.arange(1, Lx_pad + 1))
    alpha = jnp.concatenate([row0[None], all_rows], axis=0)
    return alpha


# ---------------------------------------------------------------------------
# Backward DP for the 2-edge model.
# ---------------------------------------------------------------------------
#
# Convention matches the 1-edge module:
#   beta[i, j, s, t] = log P(emissions after (i, j, s, t) until E),
#   excluding the emission AT (i, j, s, t).
#
# Terminal: beta[real_Lx, real_Ly, s, t] = log_trans[s, E] for
# t in {no_edge, closed_done}, NEG_INF for all other tags
# (orphan in-flight endpoints excluded).
#
# Backward step from a successor M-cell (i+1, j+1) with AAs (c', d')
# at successor:
#   - For source tag no_edge: contribution = next[no_edge] + carry-through
#     + log_eps + next[single_tag(c', d')]   (spawn no_edge -> single)
#   - For source tag single{(a, b)}:
#       contribution = next[single_tag(a, b)] (carry-through)
#                    + log_eps + next[pair_tag(a, b, c', d')] (spawn to pair)
#                    + M(a, b; c', d') + next[closed_done]    (close to done)
#   - For source tag pair{X, Y}:
#       contribution = next[pair_tag(X, Y)] (carry-through)
#                    + (X != Y branches handled symmetrically)
#       Specifically: the close transition pair{X, Y} -> closed_single{Y}
#       (closing X) gives weight M(X; c', d'); the symmetric branch
#       pair{X, Y} -> closed_single{X} (closing Y) gives weight
#       M(Y; c', d'). For X = Y, only one branch.
#       So contribution = next[pair_tag(X, Y)] (carry-through)
#                       + LSE over branches of M(branch_AA; c', d') +
#                                              next[closed_single_tag(other)]
#   - For source tag closed_single{(a, b)}:
#       contribution = next[closed_single_tag(a, b)] (carry-through)
#                    + M(a, b; c', d') + next[closed_done]   (close to done)
#   - For source tag closed_done:
#       contribution = next[closed_done] (carry-through; no transition out)
#
# Then add log_trans[s, M] + emit_M(c', d') to source state s for each
# of the above source-tag contributions.
#
# Backward I- and D- successors are STANDARD (no tag mixing) since they
# only carry the tag through unchanged.


def _aug2_back_match_contrib(next_cell_M, c_succ_idx, d_succ_idx,
                              log_M_tensor, log_eps,
                              single_idx_table, closed_single_idx_table,
                              pair_tag_table_AA, pair_low_AA, pair_high_AA):
    """Per-source-tag contribution (BEFORE emit/trans) from an M
    successor. Returns shape (N_TAGS,).

    contrib[t] is the augmented contribution to a source cell with
    output tag t, accumulated over:
      - carry-through (source tag t survives the M transition with
        the same tag)
      - augmented "fan-in" cases as listed above.

    The output state (s' = M) factor log_trans[s, M] + emit_M(c', d')
    is applied OUTSIDE this function.
    """
    cd_flat = c_succ_idx * A + d_succ_idx
    cd_single_tag = single_idx_table[c_succ_idx, d_succ_idx]
    log_M_at_cd = log_M_tensor[:, :, c_succ_idx, d_succ_idx]  # (A, A)
    log_M_flat = log_M_at_cd.reshape(A2)  # (A^2,)

    contrib = jnp.full((N_TAGS,), NEG_INF)

    # source = no_edge:
    val_no = jnp.logaddexp(
        next_cell_M[TAG_NO_EDGE],                   # carry-through
        log_eps + next_cell_M[cd_single_tag])       # spawn no_edge -> single
    contrib = contrib.at[TAG_NO_EDGE].set(val_no)

    # source = single{(a, b)}: 3 contributions (carry, spawn-to-pair, close-to-done)
    # Carry-through: next_cell_M[single_tag(a, b)]
    single_tags_flat = single_idx_table.reshape(A2)
    next_at_single = next_cell_M[single_tags_flat]  # (A^2,)
    # Spawn single{(a, b)} -> pair{(a, b), (c', d')}: log_eps + next_cell_M[pair_tag]
    pair_tags_for_spawn = pair_tag_table_AA[:, cd_flat]  # (A^2,) for each (a, b)
    next_at_pair = next_cell_M[pair_tags_for_spawn]      # (A^2,)
    spawn_to_pair = log_eps + next_at_pair
    # Close single{(a, b)} -> closed_done: M(a, b; c', d') + next_cell_M[closed_done]
    close_to_done = log_M_flat + next_cell_M[TAG_CLOSED_DONE]
    val_single = jnp.logaddexp(next_at_single,
                                jnp.logaddexp(spawn_to_pair, close_to_done))  # (A^2,)
    contrib = contrib.at[single_tags_flat].set(val_single)

    # source = pair{X, Y}: 2 contributions (carry, close-to-closed_single).
    # Carry-through: next_cell_M[pair_tag]
    pair_tags_full = jnp.arange(PAIR_BASE, PAIR_BASE + N_PAIRS_MSET)
    next_at_pair_full = next_cell_M[pair_tags_full]  # (N_PAIRS_MSET,)
    # Close pair{X, Y} -> closed_single{Y} (closing X), weight M(X; c', d').
    # And -> closed_single{X} (closing Y), weight M(Y; c', d') (only if X != Y).
    a_low = pair_low_AA[:, 0]
    b_low = pair_low_AA[:, 1]
    a_high = pair_high_AA[:, 0]
    b_high = pair_high_AA[:, 1]
    log_M_low = log_M_at_cd[a_low, b_low]  # (N_PAIRS_MSET,) M(low; c', d')
    log_M_high = log_M_at_cd[a_high, b_high]  # (N_PAIRS_MSET,) M(high; c', d')
    cs_tag_low = closed_single_idx_table[a_low, b_low]    # (N_PAIRS_MSET,)
    cs_tag_high = closed_single_idx_table[a_high, b_high]
    # Branch A: close X=low, leave Y=high -> contribution
    #   M(low; c', d') + next_cell_M[closed_single_tag(high)]
    # Branch B: close Y=high, leave X=low -> contribution
    #   M(high; c', d') + next_cell_M[closed_single_tag(low)]
    # Both branches always fire (mirroring the forward; for same-element
    # pair{X, X} both branches contribute equal weight to the same
    # closed_single{X} successor, giving multiplicity 2).
    branchA = log_M_low + next_cell_M[cs_tag_high]  # (N_PAIRS_MSET,)
    branchB = log_M_high + next_cell_M[cs_tag_low]
    close_pair_to_cs = jnp.logaddexp(branchA, branchB)  # (N_PAIRS_MSET,)
    val_pair = jnp.logaddexp(next_at_pair_full, close_pair_to_cs)
    contrib = contrib.at[pair_tags_full].set(val_pair)

    # source = closed_single{(a, b)}: 2 contributions (carry, close-to-done).
    cs_tags_flat = closed_single_idx_table.reshape(A2)
    next_at_cs = next_cell_M[cs_tags_flat]  # (A^2,)
    close_cs_to_done = log_M_flat + next_cell_M[TAG_CLOSED_DONE]
    val_cs = jnp.logaddexp(next_at_cs, close_cs_to_done)
    contrib = contrib.at[cs_tags_flat].set(val_cs)

    # source = closed_done: only carry-through.
    contrib = contrib.at[TAG_CLOSED_DONE].set(next_cell_M[TAG_CLOSED_DONE])

    return contrib


def _aug2_backward_core(log_trans, state_types, emit, log_M_tensor,
                        x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                        real_Lx, real_Ly,
                        single_idx_table, closed_single_idx_table,
                        pair_tag_table_AA, pair_low_AA, pair_high_AA):
    """Augmented Backward DP for the 2-edge model."""
    ns = log_trans.shape[0]
    is_M_st = (state_types == M)
    is_I_st = (state_types == I)
    is_D_st = (state_types == D)
    e_idx = _find_e_idx(state_types)

    # Terminal at (real_Lx, real_Ly): allowed only at t in {no_edge, closed_done}.
    beta_term_no_edge = log_trans[:, e_idx]  # (5,)
    term_mask = jnp.zeros(N_TAGS, dtype=bool).at[TAG_NO_EDGE].set(True
                                          ).at[TAG_CLOSED_DONE].set(True)
    beta_term = jnp.where(
        term_mask[None, :],
        beta_term_no_edge[:, None] + jnp.zeros((1, N_TAGS)),
        NEG_INF)
    i_eq = jnp.arange(Lx_pad + 1) == real_Lx
    j_eq = jnp.arange(Ly_pad + 1) == real_Ly
    is_term = i_eq[:, None] & j_eq[None, :]

    init_next_row = jnp.full((Ly_pad + 1, ns, N_TAGS), NEG_INF)

    def row_step_bwd(next_row, i):
        # j = Ly_pad: only D-successor.
        j = Ly_pad
        i_succ = jnp.minimum(i + 1, Lx_pad)
        succ_emit_D = emit[i_succ, Ly_pad]
        succ_cell_D = next_row[Ly_pad]
        contrib_D_sp = log_trans + succ_emit_D[None, :]
        contrib_D_sp = jnp.where(is_D_st[None, :], contrib_D_sp, NEG_INF)
        contrib_D = jax.nn.logsumexp(
            contrib_D_sp[:, :, None] + succ_cell_D[None, :, :], axis=1)
        cell_jLypad = contrib_D
        cell_jLypad = jnp.where(is_term[i, Ly_pad],
                                 beta_term, cell_jLypad)

        def col_step_bwd(beta_right, j):
            j_succ = jnp.minimum(j + 1, Ly_pad)
            i_succ_local = jnp.minimum(i + 1, Lx_pad)
            # M-successor at (i+1, j+1): augmented tag mixing.
            succ_emit_M = emit[i_succ_local, j_succ, M]
            succ_cell_M = next_row[j_succ][M, :]
            c_succ = x_pad[i_succ_local - 1]
            d_succ = y_pad[j_succ - 1]
            contrib_per_tag = _aug2_back_match_contrib(
                succ_cell_M, c_succ, d_succ,
                log_M_tensor, log_eps,
                single_idx_table, closed_single_idx_table,
                pair_tag_table_AA, pair_low_AA, pair_high_AA)  # (N_TAGS,)
            # Contribute to source state s via log_trans[s, M] + emit_M.
            contrib_M = log_trans[:, M:M + 1] + succ_emit_M + contrib_per_tag[None, :]
            # I-successor at (i, j+1): standard (no tag mixing).
            succ_emit_I = emit[i, j_succ]
            succ_cell_I = beta_right
            contrib_I_sp = log_trans + succ_emit_I[None, :]
            contrib_I_sp = jnp.where(is_I_st[None, :], contrib_I_sp, NEG_INF)
            contrib_I = jax.nn.logsumexp(
                contrib_I_sp[:, :, None] + succ_cell_I[None, :, :], axis=1)
            # D-successor at (i+1, j): standard.
            succ_emit_D = emit[i_succ_local, j]
            succ_cell_D = next_row[j]
            contrib_D_sp = log_trans + succ_emit_D[None, :]
            contrib_D_sp = jnp.where(is_D_st[None, :], contrib_D_sp, NEG_INF)
            contrib_D = jax.nn.logsumexp(
                contrib_D_sp[:, :, None] + succ_cell_D[None, :, :], axis=1)
            cell = jnp.logaddexp(contrib_M,
                                 jnp.logaddexp(contrib_I, contrib_D))
            cell = jnp.where(is_term[i, j], beta_term, cell)
            return cell, cell

        _, row_rest = jax.lax.scan(
            col_step_bwd, cell_jLypad,
            jnp.arange(Ly_pad - 1, -1, -1))
        curr_row = jnp.concatenate([row_rest[::-1], cell_jLypad[None]],
                                    axis=0)
        return curr_row, curr_row

    _, all_rows_bwd = jax.lax.scan(
        row_step_bwd, init_next_row,
        jnp.arange(Lx_pad, -1, -1))
    beta = all_rows_bwd[::-1]
    return beta


# ---------------------------------------------------------------------------
# JIT wrappers.
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _aug2_forward_jit(log_trans, state_types, emit, log_M_tensor,
                      x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                      single_idx_table, closed_single_idx_table,
                      pair_tag_table_AA, pair_low_AA, pair_high_AA):
    return _aug2_forward_core(log_trans, state_types, emit, log_M_tensor,
                              x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                              single_idx_table, closed_single_idx_table,
                              pair_tag_table_AA, pair_low_AA, pair_high_AA)


@partial(jax.jit, static_argnames=('Lx_pad', 'Ly_pad'))
def _aug2_backward_jit(log_trans, state_types, emit, log_M_tensor,
                       x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                       real_Lx, real_Ly,
                       single_idx_table, closed_single_idx_table,
                       pair_tag_table_AA, pair_low_AA, pair_high_AA):
    return _aug2_backward_core(log_trans, state_types, emit, log_M_tensor,
                               x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
                               real_Lx, real_Ly,
                               single_idx_table, closed_single_idx_table,
                               pair_tag_table_AA, pair_low_AA, pair_high_AA)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def forward_aug2_phmm(log_trans, state_types, sub_matrix, pi,
                       x_seq, y_seq, real_Lx, real_Ly,
                       log_M_tensor, log_eps,
                       log_emit_table=None):
    """2-edge augmented Forward wrapper. Returns alpha (Lx_pad+1,
    Ly_pad+1, 5, N_TAGS)."""
    Lx_pad = _pad_to_bin(int(x_seq.shape[0]))
    Ly_pad = _pad_to_bin(int(y_seq.shape[0]))
    if log_emit_table is None:
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
        emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi)
    else:
        ns = state_types.shape[0]
        Lx_arr = x_seq.shape[0]; Ly_arr = y_seq.shape[0]
        emit = jnp.full((Lx_pad + 1, Ly_pad + 1, ns), NEG_INF)
        emit = emit.at[:Lx_arr + 1, :Ly_arr + 1, :].set(log_emit_table)
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
    mask = _emit_mask(real_Lx, real_Ly, Lx_pad, Ly_pad,
                      state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)
    tab = _build_jax_index_tables()
    return _aug2_forward_jit(
        log_trans, state_types, emit, log_M_tensor,
        x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
        tab['single_idx'], tab['closed_single_idx'],
        tab['pair_tag_AA'], tab['pair_low_AA'], tab['pair_high_AA'])


def backward_aug2_phmm(log_trans, state_types, sub_matrix, pi,
                        x_seq, y_seq, real_Lx, real_Ly,
                        log_M_tensor, log_eps,
                        log_emit_table=None):
    """2-edge augmented Backward wrapper. Returns beta (Lx_pad+1,
    Ly_pad+1, 5, N_TAGS)."""
    Lx_pad = _pad_to_bin(int(x_seq.shape[0]))
    Ly_pad = _pad_to_bin(int(y_seq.shape[0]))
    if log_emit_table is None:
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
        emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi)
    else:
        ns = state_types.shape[0]
        Lx_arr = x_seq.shape[0]; Ly_arr = y_seq.shape[0]
        emit = jnp.full((Lx_pad + 1, Ly_pad + 1, ns), NEG_INF)
        emit = emit.at[:Lx_arr + 1, :Ly_arr + 1, :].set(log_emit_table)
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
    mask = _emit_mask(real_Lx, real_Ly, Lx_pad, Ly_pad,
                      state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)
    tab = _build_jax_index_tables()
    return _aug2_backward_jit(
        log_trans, state_types, emit, log_M_tensor,
        x_pad, y_pad, log_eps, Lx_pad, Ly_pad,
        jnp.asarray(real_Lx), jnp.asarray(real_Ly),
        tab['single_idx'], tab['closed_single_idx'],
        tab['pair_tag_AA'], tab['pair_low_AA'], tab['pair_high_AA'])


def aug_phmm_2edge_corrected_posterior(
        x_seq, y_seq, t: float,
        ins_rate: float, del_rate: float, ext: float,
        Q_lg, pi_lg, boost_state,
        alpha_z: float = 100.0,
        q_min: float = 0.0) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """End-to-end 2-edge augmented PHMM corrected posterior.

    Same signature as ``aug_phmm_corrected_posterior`` (1-edge module).

    Returns:
      Q_prime: (Lx, Ly) corrected match posterior.
      L_exact: scalar 2-edge partition function.
      Q_baseline: (Lx, Ly) F1/F0 baseline.
      log_F0: scalar baseline log-partition.
    """
    Lx = x_seq.shape[0]; Ly = y_seq.shape[0]
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)

    log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
        ins_rate, del_rate, t, ext, Q_lg, pi_lg)

    x_pad = jnp.asarray(_pad_seq(jnp.asarray(x_seq), Lx_pad))
    y_pad = jnp.asarray(_pad_seq(jnp.asarray(y_seq), Ly_pad))

    M_tensor_np = build_M_tensor_aa_marginal(boost_state)
    log_M_tensor = jnp.log(jnp.clip(jnp.asarray(M_tensor_np), 1e-300, None))

    log_eps = float(np.log(1.0 / float(alpha_z)))
    log_eps_j = jnp.asarray(log_eps)

    emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi_out)
    mask = _emit_mask(jnp.asarray(Lx), jnp.asarray(Ly), Lx_pad, Ly_pad,
                      state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)

    tab = _build_jax_index_tables()
    alpha = _aug2_forward_jit(
        log_trans, state_types, emit, log_M_tensor,
        x_pad, y_pad, log_eps_j, Lx_pad, Ly_pad,
        tab['single_idx'], tab['closed_single_idx'],
        tab['pair_tag_AA'], tab['pair_low_AA'], tab['pair_high_AA'])
    beta = _aug2_backward_jit(
        log_trans, state_types, emit, log_M_tensor,
        x_pad, y_pad, log_eps_j, Lx_pad, Ly_pad,
        jnp.asarray(Lx), jnp.asarray(Ly),
        tab['single_idx'], tab['closed_single_idx'],
        tab['pair_tag_AA'], tab['pair_low_AA'], tab['pair_high_AA'])

    # L_exact = sum over s, t in {no_edge, closed_done} of
    # alpha[Lx, Ly, s, t] + log_trans[s, E].
    e_idx = _find_e_idx(state_types)
    end_alpha = alpha[Lx, Ly, :, :]
    end_no_edge = end_alpha[:, TAG_NO_EDGE] + log_trans[:, e_idx]
    end_done = end_alpha[:, TAG_CLOSED_DONE] + log_trans[:, e_idx]
    log_L_exact = jax.nn.logsumexp(jnp.concatenate([end_no_edge, end_done]))

    # Q'_{ij} = (sum_t alpha[i, j, M, t] * beta[i, j, M, t]) / L_exact
    log_post_M = jax.nn.logsumexp(alpha[:, :, M, :] + beta[:, :, M, :],
                                       axis=-1)
    log_Q_prime_pad = log_post_M - log_L_exact
    log_Q_prime = log_Q_prime_pad[1:Lx + 1, 1:Ly + 1]
    Q_prime = np.asarray(jnp.exp(log_Q_prime))

    # Standard pair-HMM F1 / F0 baseline.
    from .f2_scfg import compute_F0_F1
    F0_b, log_F0_b, F1_b, _, _ = compute_F0_F1(
        log_trans, state_types, sub_matrix, pi_out,
        x_pad, y_pad, jnp.asarray(Lx), jnp.asarray(Ly))
    Q_baseline = np.asarray(F1_b) / float(max(F0_b, 1e-300))
    log_F0_val = float(log_F0_b)
    L_exact_val = float(jnp.exp(log_L_exact))

    _ = q_min  # unused; full sweep
    return Q_prime, L_exact_val, Q_baseline, log_F0_val


# ---------------------------------------------------------------------------
# Brute-force reference for verification (numpy, slow, small cases only).
# ---------------------------------------------------------------------------


def brute_force_2edge_posterior(
        x_seq, y_seq, t: float,
        ins_rate: float, del_rate: float, ext: float,
        Q_lg, pi_lg, boost_state,
        alpha_z: float = 100.0):
    """Direct enumeration of the 2-edge SCFG-equivalent partition for
    very small sequence pairs (Lx, Ly <= 4 strongly recommended).

    Reference for the DP. Enumerates every PHMM alignment, then for
    each alignment with M Match cells:
      - 0-edge contribution: pi(A) * 1
      - 1-edge contribution: pi(A) * eps * sum over unordered pairs
        of distinct Match cells {p1, p2} (p1 < p2 in path order) of
        M(AAs(p1); AAs(p2))
      - 2-edge contribution: pi(A) * eps^2 * sum over unordered sets
        of 4 distinct Match cells {m1 < m2 < m3 < m4 in path order}
        of [ M(AAs(m1); AAs(m3)) * M(AAs(m2); AAs(m4))
           + M(AAs(m1); AAs(m4)) * M(AAs(m2); AAs(m3)) ]
        (the two valid edge-assignments, with both spawns at m1, m2
        preceding both closures at m3, m4 -- matches the DP encoding
        which forbids "spawn after a previous closure").

    Returns: (Q_prime (Lx, Ly), L_exact, Q_baseline (Lx, Ly), log_F0).
    """
    from itertools import combinations
    Lx = int(x_seq.shape[0]); Ly = int(y_seq.shape[0])
    log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
        ins_rate, del_rate, t, ext, Q_lg, pi_lg)
    log_trans_np = np.asarray(log_trans)
    pi_np = np.asarray(pi_out)
    sub_np = np.asarray(sub_matrix)
    eps = 1.0 / float(alpha_z)

    # M tensor (AA-marginal).
    M_tensor = build_M_tensor_aa_marginal(boost_state)  # (A, A, A, A)

    # State codes.
    S_st, M_st, I_st, D_st, E_st = 0, 1, 2, 3, 4

    log_emit_M = np.log(pi_np[x_seq[:, None]] *
                         sub_np[x_seq[:, None], y_seq[None, :]] + 1e-300)
    log_emit_I = np.log(pi_np[y_seq] + 1e-300)
    log_emit_D = np.log(pi_np[x_seq] + 1e-300)

    # Enumerate paths.
    def enumerate_paths(i, j, prev_state, log_p, path, results):
        if i == Lx and j == Ly:
            log_p_e = log_p + log_trans_np[prev_state, E_st]
            # Capture: list of Match positions (i_idx, j_idx) and
            # their AAs.
            match_cells = [(ii, jj, int(x_seq[ii]), int(y_seq[jj]))
                           for (st, ii, jj) in path if st == M_st]
            results.append((np.exp(log_p_e), match_cells))
            return
        if i < Lx and j < Ly:
            new_lp = log_p + log_trans_np[prev_state, M_st] + log_emit_M[i, j]
            path.append((M_st, i, j))
            enumerate_paths(i + 1, j + 1, M_st, new_lp, path, results)
            path.pop()
        if j < Ly:
            new_lp = log_p + log_trans_np[prev_state, I_st] + log_emit_I[j]
            path.append((I_st, i, j))
            enumerate_paths(i, j + 1, I_st, new_lp, path, results)
            path.pop()
        if i < Lx:
            new_lp = log_p + log_trans_np[prev_state, D_st] + log_emit_D[i]
            path.append((D_st, i, j))
            enumerate_paths(i + 1, j, D_st, new_lp, path, results)
            path.pop()

    results = []
    enumerate_paths(0, 0, S_st, 0.0, [], results)

    F0 = 0.0
    L_exact = 0.0
    F1 = np.zeros((Lx, Ly))
    Q_prime_numer = np.zeros((Lx, Ly))

    for prob, cells in results:
        n_cells = len(cells)
        # 0-edge:
        w0 = 1.0
        # 1-edge:
        w1 = 0.0
        # iterate unordered pairs of distinct Match cells.
        for p1, p2 in combinations(range(n_cells), 2):
            (i1, j1, a1, b1) = cells[p1]
            (i2, j2, a2, b2) = cells[p2]
            w1 += M_tensor[a1, b1, a2, b2]
        w1 *= eps
        # 2-edge: iterate 4 distinct Match cells in increasing path order.
        w2 = 0.0
        for m1, m2, m3, m4 in combinations(range(n_cells), 4):
            (i1, j1, a1, b1) = cells[m1]
            (i2, j2, a2, b2) = cells[m2]
            (i3, j3, a3, b3) = cells[m3]
            (i4, j4, a4, b4) = cells[m4]
            # Two assignments of closures to spawns:
            # spawn-1 = m1, spawn-2 = m2; close-1 = m3 or m4 (paired
            # with spawn-1), the other = m4 or m3 (paired with spawn-2).
            assign_A = M_tensor[a1, b1, a3, b3] * M_tensor[a2, b2, a4, b4]
            assign_B = M_tensor[a1, b1, a4, b4] * M_tensor[a2, b2, a3, b3]
            w2 += assign_A + assign_B
        w2 *= eps ** 2

        total_w = w0 + w1 + w2
        F0 += prob * w0
        L_exact += prob * total_w
        # F1[i, j] increments for each Match cell at (i, j):
        for (ii, jj, _, _) in cells:
            F1[ii, jj] += prob
            Q_prime_numer[ii, jj] += prob * total_w

    Q_baseline = F1 / max(F0, 1e-300)
    Q_prime = Q_prime_numer / max(L_exact, 1e-300)
    log_F0 = np.log(max(F0, 1e-300))
    return Q_prime, L_exact, Q_baseline, log_F0
