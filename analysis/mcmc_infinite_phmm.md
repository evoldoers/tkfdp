# MCMC sampler from the infinite Pair HMM

This document specifies the algorithm and implementation plan for a
Markov-chain Monte Carlo sampler that draws alignments and edge sets
from the **infinite Pair HMM** of `main.tex` Section
`sec:infinite-hmm`. This is the principled formulation that the
bounded-edge dynamic-programming approximations in
`src/tkfdp/aug_phmm.py` (1-edge) and `src/tkfdp/aug_phmm_2edge.py`
(2-edge) are approximating.

The high-level paragraph in `main.tex` `\paragraph{MCMC sampler from
the infinite Pair HMM}` (around line 798) is the authoritative
high-level spec. This document refines that paragraph to a precise
algorithm + implementation plan. **No code is written yet.** The user
will review this plan before authorising implementation.

---

## A. Mathematical setup

### A.1 Target distribution

We sample `(A, E)` from
```
pi(A, E | X, Y) ∝ P_baseline(A | X, Y) · CRP(E ; A, alpha_z) · BoostProd(E ; A, X, Y, t).
```
The three factors are:

1. **Baseline pair-HMM joint probability** of the alignment:
   `P_baseline(A | X, Y) = pi_TKF92(X, Y, A)` — the no-edge TKF92
   Pair HMM joint probability of `A` aligning `X` and `Y`. This is
   the same `pi(X, Y, A)` summand that defines `F_0`, `F_1`, `F_2`
   in `main.tex` (eqs F0, F1, F2). It includes both the path-
   transition factors and the per-cell single-site emissions.

2. **CRP prior on the edge set, conditioned on the Match cells of A.**
   Let `N_M(A)` be the number of Match cells in `A`, indexed by
   their *Match-cell rank* `m ∈ {1, 2, …, N_M(A)}` (rank along the
   alignment path, in order of consumption). For each Match cell the
   infinite-HMM model decides "open a new coupled-edge endpoint
   here?" with probability `alpha_z / (m - 1 + alpha_z)` (canonical
   CRP) — equivalently, an *opening* indicator `o_m ∈ {0, 1}` with
   `P(o_m = 1 | m) = alpha_z / (m - 1 + alpha_z)`. The total number
   of openings over a path of length `N_M` is exactly the number of
   coupled-edge *endpoints*, and each endpoint must be paired
   (opening + closure) for the path to terminate cleanly. Let
   `|E|` be the number of *edges* (pairs of endpoints); the number
   of openings is `2|E|`. Under the standard CRP-table view this
   corresponds to: for each Match cell, draw it as a "new table"
   (open) with prob `alpha_z / (m - 1 + alpha_z)` or as "join an
   existing open table" (close one of the in-flight openings) with
   prob `(m - 1 - 2|E_open|) / (m - 1 + alpha_z)` — but because the
   sampler explicitly proposes the *edge graph* `E` rather than the
   table assignment, we encode the prior more directly as

   ```
   CRP(E ; A, alpha_z)
     = ∏_{e ∈ E} alpha_z
       / ∏_{m=1..N_M(A)} (m - 1 + alpha_z)        … "new-table" factor
   ```

   times a combinatorial factor accounting for which Match cells
   the openings landed on and which opening pairs with which closure.
   For concreteness we use the **two-cells-per-edge** placement
   convention: an edge `e` is an unordered pair of distinct Match
   cells `{(i_1, j_1), (i_2, j_2)}` (its two endpoints). The CRP
   prior is then

   ```
   CRP(E ; A, alpha_z)
     = alpha_z^{|E|}
       * ∏_{m=1..N_M(A)} 1 / (m - 1 + alpha_z)
       * (combinatorial weight from the placements).
   ```

   The combinatorial weight cancels in the MH ratios because (i) it
   is the same in every "set of `|E|` distinct edges with these
   2|E| Match-cell positions," and (ii) the segment-resample move
   keeps the *positions* of edges fixed (or it does not — see
   B.1.b), so for the MH ratio only the `alpha_z^{|E|} / ∏(m - 1 +
   alpha_z)` factor contributes. **In the MH derivations below we
   carry only this factor.**

   The convention matches eq:infinite-hmm-edge of `main.tex`:
   marginalising the per-cell `P(open | m)` over a fixed alignment
   gives precisely this prior.

3. **Edge-boost product** at each edge: for each edge `e =
   {(i_1, j_1), (i_2, j_2)} ∈ E`,

   ```
   BoostProd(E ; A, X, Y, t)
     = ∏_{e ∈ E} M(i_1, j_1; i_2, j_2 ; t)
   ```

   where `M(...; t)` is the four-residue Potts coupling boost of
   `main.tex` eq:M-marginal, evaluated at the four observed amino
   acids `(X_{i_1}, Y_{j_1}, X_{i_2}, Y_{j_2})` and the per-pair
   inferred branch length `t`. This is the same `M` tensor used in
   `aug_phmm.build_M_tensor_aa_marginal` (AA-marginal version under
   the prior class distribution; see G.3 for the K_c > 1 caveat).
   When two edges share an endpoint, this product factorises by
   independence — see open question H.6.

The unnormalised target is therefore
```
pi(A, E | X, Y)
  ∝ P_baseline(A | X, Y)
    · alpha_z^{|E|} / ∏_{m=1..N_M(A)} (m - 1 + alpha_z)
    · ∏_{e ∈ E} M(e ; t).
```

### A.2 Partial-Forward tensor `F^partial[i, j; k, l]`

Define the partial-Forward tensor as the **baseline (no-edge) pair-HMM
Forward probability of any alignment that visits Match at both anchors
`(i, j)` and `(k, l)`** with `(i, j) <_{lex} (k, l)`:
```
F^partial[i, j; k, l]
  = sum_{A : X_i ~_A Y_j AND X_k ~_A Y_l}  pi_TKF92(X, Y, A)
  = F_2(X, Y; i, j; k, l)
```
i.e. it is **identical to** the `F_2` tensor of `main.tex` eq:F2 and
of `src/tkfdp/f2_scfg.py`. We re-use the same setup kernel.

**Precise inclusion/sign convention.** Both endpoints `(i, j)` and
`(k, l)` are in M-state. The Match-cell emissions at *both* anchors
are *included* (the sum is over alignments, not segments, so every
emission in the alignment is counted exactly once). This matches the
F_2 convention in `f2_scfg.compute_F0_F1` — the M-emission at the
anchor is in `alpha[i, j, M]`, the M-emission at the partner is in
`mu[k, l, M] + beta[k, l, M]` via the restart-Forward `mu`.

We will **also need** boundary-anchored variants for the segment-
resample proposal (see B.1):

- `F^partial_open[i, j ; k, l]` — like `F^partial`, but the alignment
  *opens* at `(i, j)` (the entire prefix `[1..i-1]×[1..j-1]` is
  unconstrained, i.e. a standard Forward up to `(i, j)`) and
  *closes* at `(k, l)` (the entire suffix `[k+1..L_x]×[l+1..L_y]` is
  unconstrained, i.e. a standard Backward from `(k, l)`).
- `F^partial_segment[i, j ; k, l]` — the **fragment-only** version:
  the Forward probability of the fragment from immediately after
  `(i, j)` to immediately before `(k, l)`, with both endpoints in
  M-state but their emissions *excluded* and the entry/exit
  transitions `M→…` and `…→M` *included*. This is the proposal
  kernel for the segment-resample move (the proposal samples a
  fragment between two fixed anchors).

The relationship is `F^partial[i, j; k, l] = alpha[i, j, M] *
F^partial_segment[i, j; k, l] * (M-emission at k, l) * beta[k, l,
M]`. For the segment-resample move we want the conditional
distribution of the fragment given the two anchors, so
`F^partial_segment` is the natural object.

**Memory layout.** Dense fp32 4-tensor of shape
`(L_x_pad+1, L_y_pad+1, L_x_pad+1, L_y_pad+1)`. At `L_x = L_y = 200`
this is `200^4 * 4 bytes ≈ 6.4 GB`, prohibitive on most GPUs. We
mitigate via:

- **Lex-order reduction**: store only `(i, j) <_lex (k, l)` entries,
  saving a factor 2.
- **Reachability mask**: entries with `k < i` or `l < j` are zero;
  store as a banded array along the second-anchor axis for memory.
- **Anchor bracketing**: in practice the segment-resample move only
  ever queries `(i_a, j_a)` and `(i_b, j_b)` that are adjacent
  Match anchors in the *current* alignment, so the second-anchor
  axis is sparse in practice. We store the full `F^partial`
  precomputed once (when memory permits, `L ≤ 100`) and recompute on
  demand for larger `L` (anchor-by-anchor restart-Forward, exactly
  as `f2_scfg._restart_forward_core` already does). See G.4.
- **Padding for JIT bin reuse**: same `_pad_to_bin` geometric bin
  scheme as `aug_phmm.py` and `f2_scfg.py`, on all four anchor axes.
  Padded entries are masked to zero (linear) / `NEG_INF` (log).

**Boundary cases.** We require `1 ≤ i ≤ k ≤ L_x` and similarly for
`j ≤ l ≤ L_y` for valid Match anchors. The "virtual" boundary cells
`(0, 0)` (start) and `(L_x + 1, L_y + 1)` (end-pseudo) are handled
separately: the segment from `(0, 0)` to the first anchor uses the
standard `alpha[i, j, M]` (no left boundary anchor); the segment
from the last anchor to `(L_x + 1, L_y + 1)` uses the standard
`beta[i, j, M]`.

### A.3 Convention recap

- Match cells are in 1-based residue coordinates `(i, j)` with
  `1 ≤ i ≤ L_x` and `1 ≤ j ≤ L_y`.
- Match-cell rank `m` along an alignment path is the count of Match
  cells consumed up to and including the cell.
- Edge endpoints are Match cells; an edge is an unordered pair of
  distinct Match cells.
- An edge whose two endpoints are *the same* Match cell is forbidden
  (handled by enforcing `(i_1, j_1) ≠ (i_2, j_2)`).
- Edges may share endpoints with other edges in the unbounded model;
  the Boost product still factorises (see H.6).

---

## B. Move definitions

Each MCMC sweep alternates two move classes: **(1) segment-resample**
moves and **(2) edge add/remove** moves. Sweep schedule is open
question H.4.

### B.1 Segment-resample MH move

#### B.1.a Choice of segment

Given the current alignment `A` with Match cells
`{a_1, a_2, …, a_{N_M}}` in path order, pick two **adjacent** Match
anchors `a_p = (i_a, j_a)` and `a_{p+1} = (i_b, j_b)` with `p ∈ {0,
1, …, N_M}`. We treat `a_0 = (0, 0)` and `a_{N_M + 1} = (L_x + 1,
L_y + 1)` as virtual sentinel anchors (start and end). The
*segment* is the sub-alignment between these two anchors,
*excluding* the anchor cells themselves.

The choice of `p` is uniform over `{0, …, N_M}` — `N_M + 1` options
per move. (Open question H.4 — random subset vs all.)

#### B.1.b Proposal `q_seg(A_new | A_old)`

Sample a new segment between `a_p = (i_a, j_a)` and `a_{p+1} = (i_b,
j_b)` from the **baseline (no-edge) pair-HMM Forward distribution
restricted to passing through M-state at both anchors and consuming
exactly the residues in the bracketing window**:

```
q_seg(A_new | A_old)
  = pi_TKF92(segment | M at a_p, M at a_{p+1})
  = pi_TKF92(fragment from a_p+1 to a_{p+1}-1)
    / F^partial_segment[i_a, j_a; i_b, j_b]
```

Implementation: traceback from the `F^partial_segment[i_a, j_a; i_b,
j_b]` table, sampling each cell from the categorical of
predecessor-state-and-position weighted by Forward partials. This is
a standard Forward-then-stochastic-traceback procedure; the
restart-Forward kernel from `f2_scfg._restart_forward_core` is
directly reusable.

The proposal **does not include** edge boosts or CRP factors. Those
all live in the MH acceptance.

#### B.1.c Reverse proposal `q_seg(A_old | A_new)`

Symmetric: from the new alignment `A_new`, the same segment-choice
indexing `p` selects the same pair of anchor positions `(i_a, j_a)`,
`(i_b, j_b)` (because anchors at index `p` are the same; the segment
between them was just resampled), and the *reverse* proposal
samples the old segment from the same `F^partial_segment[i_a, j_a;
i_b, j_b]` distribution. Hence

```
q_seg(A_old | A_new) / q_seg(A_new | A_old)
  = pi_TKF92(old_segment) / pi_TKF92(new_segment)
```

up to the `F^partial_segment` denominator which **cancels** because
both proposals use the same anchors and therefore the same
denominator. **This is the central simplification** that makes the
move tight.

#### B.1.d Edge handling under segment resampling

The segment-resample move *changes the set of Match cells in the
segment* — possibly losing or gaining Match cells relative to the old
segment. Edges whose endpoints fall inside the segment must be
addressed:

- **If both endpoints of an edge are inside the segment**: the edge
  in `A_new` references Match-cell positions that may no longer
  exist (the new segment's Match cells are in different positions).
  We *delete* such edges from `E_new` as part of the proposal.
  Specifically, on segment resample we partition `E_old` into
  `E_old = E_outside ∪ E_inside` where `E_inside` has both
  endpoints inside the segment, and propose `E_new = E_outside`.
  The segment-resample move thus *only deletes edges*, never adds
  them. (Edges with one endpoint inside and one outside the segment:
  see H.7; the simplest convention is that the segment-resample
  move is *rejected* in that case — equivalently, those edges are
  cross-segment and must be removed by edge-remove moves before
  segment resampling can succeed across them.)

- We **augment the move's Hastings ratio** with the CRP-prior factor
  for the deleted edges. The reverse proposal needs to *add back*
  edges to recover `A_old`'s edge set, which is not a symmetric
  proposal — it is one-directional unless we also extend the move
  to *propose new edges within the new segment*.

  **Two implementation strategies**:
  - **Strategy S-1 ("conservative segment")**: the segment-resample
    move only operates when `E_inside = ∅` (no edges to delete).
    Then `E_new = E_old` always. The MH ratio is purely the path-
    weight ratio with the CRP-prior depending on `N_M^{seg}`. Edge
    moves alone handle creating/destroying edges. **Simpler;
    recommended default.**
  - **Strategy S-2 ("integrated segment+edge")**: the segment move
    is allowed to delete `E_inside` and propose new edges within the
    new segment; the proposal includes a forward draw of new edges
    via the standard CRP-add proposal *within the new segment*. The
    reverse proposal removes those new edges and adds back the old
    `E_inside` via CRP-add proposals on the old segment. The MH
    ratio gains contributions from both directions; the algebra is
    routine but tedious.

  **Decision: implement Strategy S-1 by default.** Add Strategy S-2
  as an experimental flag (see H.8).

#### B.1.e Target ratio under Strategy S-1

With `E_new = E_old = E` (no edges in the segment), the target ratio
factorises into:

- **Baseline path ratio**:
  `pi_TKF92(A_new) / pi_TKF92(A_old) = (pi_TKF92(new_seg) /
  pi_TKF92(old_seg))` because all out-of-segment alignment factors
  cancel.

- **CRP-prior ratio**:
  ```
  CRP(E ; A_new, alpha_z) / CRP(E ; A_old, alpha_z)
    = ∏_{m=1..N_M(A_new)} 1/(m - 1 + alpha_z)
      / ∏_{m=1..N_M(A_old)} 1/(m - 1 + alpha_z)
  ```
  Let `N_M^{old, seg}` and `N_M^{new, seg}` be the number of Match
  cells in the old and new segments. The Match-cell ranks *outside*
  the segment are unchanged before the segment, but *shifted* by
  `(N_M^{new, seg} - N_M^{old, seg})` after the segment. The
  product simplifies to
  ```
  CRP_ratio
    = [∏_{m = m_0 + 1}^{m_0 + N_M^{new, seg}} (m - 1 + alpha_z)]^{-1}
      * [∏_{m = m_0 + 1}^{m_0 + N_M^{old, seg}} (m - 1 + alpha_z)]
      * [∏_{m = m_0 + N_M^{new, seg} + 1}^{N_M(A_new)} (m - 1 + alpha_z)]^{-1}
      * [∏_{m = m_0 + N_M^{old, seg} + 1}^{N_M(A_old)} (m - 1 + alpha_z)]
  ```
  where `m_0` is the running Match-count entering the segment (the
  rank of the last Match cell *before* the segment, i.e. of `a_p`,
  or 0 if `p = 0`). The full product can be computed efficiently in
  O(L) by storing prefix products of `(m - 1 + alpha_z)`.

  The `alpha_z^{|E|}` factor is unchanged (E is unchanged), and
  cancels.

- **Boost-product ratio**: unchanged because `E` is unchanged, but
  *the Match-cell positions of edges may have changed in 1-D rank*
  (their absolute coords are unchanged). Since the boost depends
  only on `(i_1, j_1, i_2, j_2)` and not on rank, the ratio is 1.

**Hastings ratio (Strategy S-1)**:
```
H_seg = min(1,
            [pi_TKF92(new_seg) / pi_TKF92(old_seg)]      ← target path ratio
            * CRP_ratio                                  ← CRP-prior factor
            * [q_seg(A_old | A_new) / q_seg(A_new | A_old)])
```
With the proposal substitution
```
q_seg(A_new | A_old) = pi_TKF92(new_seg) / F^partial_segment[a_p, a_{p+1}],
q_seg(A_old | A_new) = pi_TKF92(old_seg) / F^partial_segment[a_p, a_{p+1}],
```
the path-ratio and proposal-ratio cancel, leaving
```
H_seg = min(1, CRP_ratio).
```

This is the user's claim: **the proposal is tight; the only
uncorrected factor is the CRP-prior path-length term**, which is
small whenever the per-cell spawn weight `eps · L_aln ≪ 1`.

#### B.1.f Computational cost

- Per move: O(`segment length`) to traceback + O(`segment length`)
  to compute the CRP-prior ratio.
- Average segment length is O(1) at high-anchor-density regimes,
  but worst-case O(L) for sparse anchors (e.g. an empty initial
  alignment).
- One sweep: O(`N_M + 1`) ~ O(L) such moves per sweep.
- Total per sweep: O(L^2) in the worst case, O(L) in the well-mixed
  regime.

The O(L^2) bound is dominated by the much-slower setup phase and is
the asymptotic-limit case rarely hit at convergence; in practice the
MCMC sweep cost is dominated by O(L) per sweep.

### B.2 CRP add-edge MH move

#### B.2.a Proposal `q_add(E_new | E_old, A)`

Pick a Match-cell rank `m ∈ {1, …, N_M(A)}` for the *opening* with
**probability proportional to the per-cell M-weight at the opening
position**, conditioned on `m` not already participating in `|E_open
|` edges (or with no such conditioning — see H.6). Pick a *partner*
Match cell from the remaining Match cells weighted by the Boost
factor `M((i_m, j_m); (i_n, j_n); t)` (so the proposal includes the
edge boost — making it a "weighted CRP-add" rather than a uniform
proposal).

**Soft cap `k_max`**: if `|E_old| ≥ k_max`, the add proposal is
**rejected with acceptance probability 0**. Default `k_max = ∞`.

Two specific proposals to consider:

- **Q_add-uniform**: `q_add` picks both endpoints **uniformly at
  random from the Match cells of A**. Cost O(1) sampling. Tighter
  MH algebra (the proposal probability is `1 / [N_M(N_M - 1) / 2]`
  for unordered pairs of distinct Match cells).

- **Q_add-boosted**: weight by `M((m, n); t)`. Tighter acceptance
  rate (the proposal "looks" like the target). Cost O(L) (compute
  one row of M-weights then sample). **Recommended** because the
  boost can vary by orders of magnitude across pairs.

We default to **Q_add-uniform** for maximum conceptual simplicity
(and because the user specifies "sampled from the M-weighted
distribution" — see H.5; we treat this as a mild weighting on the
M-marginal).

#### B.2.b Reverse proposal `q_remove(E_old | E_new, A)`

Pick an edge uniformly at random from `E_new`. Probability
`1 / |E_new|`. Cost O(1).

#### B.2.c Target ratio

```
pi(A, E_new) / pi(A, E_old)
  = alpha_z * (1 / (m_max - 1 + alpha_z))^0 * M((i_1, j_1); (i_2, j_2); t)
                  ↑ no Match cells added / removed; no path change ↑
```

Let me re-derive. With `E_new = E_old ∪ {e}` for some new edge `e`:
- `P_baseline(A)` is unchanged.
- `CRP(E_new, A) / CRP(E_old, A) = alpha_z` (one extra edge factor;
  the `∏_m (m - 1 + alpha_z)` denominator is unchanged because A is
  unchanged).
- `BoostProd(E_new) / BoostProd(E_old) = M(e ; t)`.

So `pi(A, E_new) / pi(A, E_old) = alpha_z * M(e ; t)`.

#### B.2.d Hastings ratio

For Q_add-uniform proposal:
- `q_add(E_new | E_old) = 1 / [N_M(N_M - 1) / 2 - |E_old|]` (assuming
  we never propose an edge already in `E_old`; or if we permit
  duplicates, see H.6).
- `q_remove(E_old | E_new) = 1 / |E_new| = 1 / (|E_old| + 1)`.

```
H_add = min(1,
   alpha_z * M(e ; t)
   * q_remove(E_old | E_new) / q_add(E_new | E_old))

      = min(1,
   alpha_z * M(e ; t)
   * [N_M(N_M - 1) / 2 - |E_old|] / (|E_old| + 1)).
```

The `alpha_z` factor in the target is the source of the "easy
edges" — for `alpha_z` large, `H_add` is high and edges are
plentiful (which seems backwards; recall the convention: `alpha_z`
is the CRP concentration, *small* → many tables, *large* → few
tables). Wait — let me double check. In the standard CRP, large
`alpha_z` ⇒ many "new tables" ⇒ many distinct atoms. Here, "new
table" ⇒ open a coupled-edge endpoint. So large `alpha_z` ⇒ many
endpoints ⇒ many edges. In the bounded-edge construction, `eps =
1 / alpha_z` is the per-cell spawn weight, so *small* `eps` ⇒ *large*
`alpha_z` ⇒ few endpoints relative to N_M. The `alpha_z` factor in
the target ratio is dimensionally consistent: the per-edge prior
mass `alpha_z` is matched by the per-cell normalisation `(m - 1 +
alpha_z)` in the denominator (which sums to roughly `N_M log
alpha_z` in expectation; the marginal over the Match-count counter
gives expected edges proportional to `alpha_z` rather than `N_M ·
alpha_z`).

This gives the desired "expected edges = O(alpha_z)" prior in the
infinite-HMM model, as stated in `main.tex` eq:infinite-hmm-edge.

#### B.2.e Cost

O(L) for Q_add-uniform (constant-time sampling but O(L) for the
boost factor M lookup if we did Q_add-boosted; for Q_add-uniform
it's O(1)). The user specifies "O(L) per add"; we adopt that as
budget for both variants.

### B.3 CRP remove-edge MH move

#### B.3.a Proposal `q_remove(E_new | E_old, A)`

Pick an edge uniformly at random from `E_old`. Probability
`1 / |E_old|`. Cost O(1).

If `|E_old| = 0`, the move is a no-op (acceptance probability 0).

#### B.3.b Reverse proposal `q_add(E_old | E_new, A)`

For consistency with B.2: `q_add(E_old | E_new) = 1 / [N_M(N_M - 1) /
2 - |E_new|]`.

#### B.3.c Target ratio

`pi(A, E_new) / pi(A, E_old) = 1 / (alpha_z * M(e ; t))`.

#### B.3.d Hastings ratio

```
H_remove = min(1,
   1 / [alpha_z * M(e ; t)]
   * q_add(E_old | E_new) / q_remove(E_new | E_old))

         = min(1,
   1 / [alpha_z * M(e ; t)]
   * (|E_old|) / [N_M(N_M - 1) / 2 - |E_old| + 1]).
```

#### B.3.e Cost

O(1) per move.

### B.4 Sweep schedule and budget

One **MCMC sweep** consists of:
1. **Segment-resample sweep**: visit each adjacent anchor pair
   `(a_p, a_{p+1})` for `p = 0, …, N_M`. (See H.4 for "all vs
   random subset".)
2. **Edge-modification sweep**: propose `n_edge_moves` edge
   add/remove moves. Default: `n_edge_moves = max(8, 2 * (|E| +
   1))`, balanced so that on average each edge gets visited a few
   times per sweep. Each individual move is randomly add (50% prob)
   or remove (50% prob).

This gives O(L) work per sweep total (excluding the O(L^4)
one-time setup).

---

## C. Initial alignment

We initialise the chain with `A_0 = ` Viterbi alignment from the
**baseline (no-edge) pair-HMM**, and `E_0 = ∅`. Justification:

- Viterbi is the modal alignment under the no-edge model and is
  a reasonable starting point for the no-edge target.
- Starting with `E_0 = ∅` is consistent with "no information beyond
  the baseline" and lets the chain build up edges incrementally via
  MH-accept.
- An alternative is to use a single sample from the baseline pair-HMM
  Forward (a Forward + stochastic traceback). This adds noise to the
  initial state and may help mixing. We expose both as options
  (`init_mode = "viterbi"` / `init_mode = "forward_sample"`) with
  Viterbi as the default.

**Empty alignment (gaps only)** is rejected as initialisation because
it has `N_M = 0` and therefore no anchors to segment-resample
between; the chain would have to bootstrap entirely via the empty-
segment-to-full-segment proposal, which is a rare draw. (The first
segment-resample move from `A_0 = ` empty *does* sample a full
alignment from the baseline Forward — equivalent to `init_mode =
"forward_sample"` — but it costs an extra sweep.)

**Burn-in**. Default `n_burnin = max(50, L / 4)` sweeps, conditional
on convergence diagnostics passing. This is a heuristic budget that
scales sub-linearly with sequence length (because each segment-
resample is O(L) work but only resamples O(1) Match cells per move,
so to "rotate" through the alignment requires O(L / per-move
information)). See open question H.3.

---

## D. Convergence diagnostics

We track three diagnostic quantities at sweep granularity (not
per-move):

1. **Per-cell `Q'_{ij}` running estimate stability**. Maintain a
   running mean and running second-moment of the `1[(i, j) is
   Match]` indicator over post-burn-in samples (Welford's online
   algorithm). The standard error of the mean for cell `(i, j)`
   after `n` samples is approximately
   `sqrt(Q'_{ij}(1 - Q'_{ij}) / n)` for a target `Q'_{ij}` near
   `0.5`. We declare convergence on cell `(i, j)` when the
   windowed standard error over the last `w = 100` samples is
   below `tol = 0.01`. Aggregate: declare global convergence when
   `≥ 95%` of cells have converged.

2. **Effective sample size (ESS) for global statistics**. Compute
   ESS via the autocorrelation-based estimator on at least:
   - `log pi(A, E)` (the joint log-target).
   - `|E|` (the edge count).
   - `N_M(A)` (the Match-cell count).

   Report `ESS / n_samples` ratio per sweep. Acceptable: `≥ 0.05`
   (i.e. autocorrelation length `≤ 20` sweeps).

3. **Multi-chain `R-hat` (potential scale reduction factor)**. Run
   `n_chains = 4` independent chains with different initial seeds.
   Compute `R-hat` for `log pi(A, E)` and for `|E|` after burn-in.
   Acceptable: `R-hat ≤ 1.05`. Multi-chain comparison is feasible
   because each chain is independent (parallel chains just need
   distinct PRNG keys); per-chain cost is identical so the cost
   scaling is `n_chains × n_sweeps` work for `n_chains` independent
   `Q'` estimates.

**Stopping criterion**: combination of fixed-budget upper bound
(default `n_sweeps = 1000` post-burn-in per chain) **and**
convergence-driven early stop (if all three diagnostics pass before
the budget exhausts). This is the standard PyMC / Stan combination.

---

## E. Verification protocol

We verify correctness via four cross-checks, listed in increasing
order of strength. All four must pass before the implementation is
declared correct.

### E.1 Cross-validate against `aug_phmm` (1-edge, large `alpha_z`)

At large `alpha_z` (e.g. `alpha_z = 10^4`, so `eps = 10^{-4}`),
the expected number of edges is small: `E[|E|] ≈ alpha_z / (N_M /
N_M)` approximately... actually under the infinite-HMM prior `E[|E|]
≈ alpha_z` (from `main.tex` eq:infinite-hmm-edge marginalisation).
So at `alpha_z = 10^{-4}` we get `E[|E|] ≈ 10^{-4}` and the 1-edge
truncation is essentially the entire posterior.

**Procedure**:
1. Pick a small-to-medium pair (`L_x = L_y = 30`, real residues).
2. Compute `Q'_aug = aug_phmm.aug_phmm_corrected_posterior(...,
   alpha_z=alpha_z)`.
3. Run the MCMC sampler with the same parameters, `k_max = 1`.
4. Compute `Q'_mcmc` averaged over `n_samples ≥ 5000` post-burn-in
   samples.
5. **Acceptance criterion**: `max |Q'_mcmc - Q'_aug| ≤ 5 σ_MC`
   where `σ_MC = sqrt(Q'_aug(1 - Q'_aug) / n_samples)` is the
   per-cell MC noise floor. Equivalently: cell-wise z-score
   `|Q'_mcmc - Q'_aug| / σ_MC ≤ 5` for `≥ 99%` of cells. (Five
   sigma allows for a few outliers in the cell distribution
   without false alarms.)

### E.2 Cross-validate against `aug_phmm_2edge` (2-edge, moderate `alpha_z`)

At moderate `alpha_z` (e.g. `alpha_z = 100`, so `eps = 0.01`), and
`L_aln ≈ 50`, the expected edges is `eps · L_aln ≈ 0.5` under the
bounded-edge model and `alpha_z * (something)` under the infinite-
HMM model. Since the two models differ in their edge-count prior,
we **cannot** expect MCMC at `k_max = ∞` to match
`aug_phmm_2edge`. We can however expect MCMC at `k_max = 2` with the
*bounded-edge* version of the CRP prior to match. (Defining the
bounded-edge CRP prior requires a separate approximation flag in
the sampler; see H.10.)

For the simpler, cleaner version of E.2 we **change the sampler
prior to the bounded-edge prior** (per-cell spawn weight `eps`
instead of canonical CRP) and verify against `aug_phmm_2edge`.

**Procedure**:
1. Same setup as E.1 but with `alpha_z = 100`, `k_max = 2`,
   and **`prior_mode = "bounded_eps"`** (a flag that switches the
   sampler to the bounded-edge per-cell-spawn prior).
2. Compute `Q'_2edge = aug_phmm_2edge.aug_phmm_2edge_corrected_posterior(
   ..., alpha_z=alpha_z)`.
3. Run MCMC; compare `Q'_mcmc` vs `Q'_2edge` with the same
   acceptance criterion as E.1.

### E.3 Cross-validate against brute-force enumeration

For very small problems (`L_x ≤ 4`, `L_y ≤ 4`), we can enumerate:
- all alignments `A` (there are O(`Lx Ly` choose `min(Lx, Ly)`)) ≈
  6-70 alignments at this scale),
- all edge sets `E` (with `|E| ≤ k_max`).

For each `(A, E)`, compute the unnormalised target
`pi(A, E)` by direct evaluation of the three factors. Sum to get
the normalising constant `Z`. Compute `Q'_{ij} = sum over (A, E)
with (i, j) Match in A of pi(A, E) / Z`.

**Procedure**:
1. Run brute-force enumeration to get the exact `Q'_BF`.
2. Run MCMC sampler to get `Q'_mcmc`.
3. **Acceptance criterion**: same as E.1 — z-score ≤ 5 for ≥ 99% of
   cells.

This is the *strongest* cross-check because it directly verifies the
MCMC is sampling from the correct target, including the CRP prior
(not just an approximate version of it). All three move types are
exercised non-trivially even on small problems.

### E.4 Detailed-balance sanity check (no-data case)

Run the MCMC chain on a "no-data" problem: e.g. `L_x = L_y = 5`,
sequences of all-X (the wildcard amino acid) so that all emissions
are `pi[X] = 1/A`. The Boost factor `M = 1` for all edges (because
`pi_joint(X, *) = pi(X) * pi(*)` factorises, hence the Boost is 1).
Under these conditions:
- `pi(A | X, Y) ∝ pi_TKF92(A)` is the prior on alignments under
  TKF92.
- `pi(E | A) = CRP(E ; A, alpha_z)`: the *empirical* edge-count
  histogram from MCMC should match the analytical CRP prior on
  `|E|` over the random `N_M`-distribution induced by TKF92.

**Procedure**:
1. Run MCMC for `≥ 10^5` samples (after burn-in).
2. Compute the empirical `|E|` histogram conditional on `N_M`
   bin (e.g. `N_M = 1, 2, 3, 4, 5`).
3. Compute the analytical CRP marginal for each `N_M`:
   `P(|E| = k | N_M) = (multinomial / Stirling integral for fixed N_M)`,
   from the standard Antoniak formula or by explicit enumeration.
4. **Acceptance criterion**: total variation distance ≤ 0.05.

This verifies the CRP-prior factor is implemented correctly and
the chain is in detailed balance.

---

## F. Implementation plan

### F.1 File layout

- `src/tkfdp/mcmc_infinite_phmm.py` — main module:
  - `precompute_partial_forward(...)` — O(L^4) setup; reuses
    `f2_scfg._restart_forward_core` per anchor row.
  - `mcmc_sampler(...)` — main loop: state init, burn-in, sweep
    loop, diagnostic logging.
  - `_segment_resample_move(...)` — Strategy S-1 resample.
  - `_edge_add_move(...)`, `_edge_remove_move(...)` — CRP moves.
  - `mcmc_corrected_posterior(...)` — public API; same signature
    style as `aug_phmm_corrected_posterior`. Returns `(Q', L_exact_est,
    Q_baseline, log_F0, mcmc_diagnostics_dict)`.
  - `init_alignment_viterbi(...)` — initial alignment via Viterbi.
  - `init_alignment_forward_sample(...)` — alternative.
- `tests/smoke_mcmc_infinite_phmm.py`:
  - `test_E1_cross_aug_phmm_at_large_alpha_z` (E.1)
  - `test_E2_cross_aug_2edge_at_moderate_alpha_z_bounded_prior` (E.2)
  - `test_E3_brute_force_small` (E.3) — over `Lx, Ly ∈ {2, 3, 4}`.
  - `test_E4_detailed_balance_no_data` (E.4)
  - `test_kmax_truncation` — verify `k_max = 1` matches `aug_phmm`,
    `k_max = 2` matches `aug_phmm_2edge` at sufficiently small
    `eps · L_aln`.
  - `test_initial_state_options` — verify Viterbi vs Forward-sample
    converge to the same `Q'`.
- `experiments/eval_balibase.py` — register a new method
  `tkfdp_mcmc` that calls `mcmc_corrected_posterior` per pair, mirrors
  `run_tkfdp_aug_phmm` exactly. The MCMC budget is exposed via
  CLI args on the harness (`--mcmc-n-sweeps`, `--mcmc-n-burnin`,
  `--mcmc-n-chains`, `--mcmc-k-max`, `--mcmc-seed`).

### F.2 JIT structure

- `precompute_partial_forward` — JIT-compiled per `(Lx_pad, Ly_pad)`.
  Internally vmap over the anchor axis, scan over rows. Same shape-
  caching strategy as `f2_scfg._process_anchor_chunk`.
- `_segment_resample_move` — JIT'd inner loop (the traceback). The
  acceptance test is a single scalar comparison so can be JIT'd
  inside the move; no Python overhead per move.
- `_edge_add_move` / `_edge_remove_move` — JIT'd per-move kernels.
- The outer `mcmc_sampler` Python loop holds a `for`-loop over
  sweeps with a `jax.lax.scan` over the sub-loop (segment moves +
  edge moves). The full sweep is JIT'd as a single `scan` call, so
  the per-sweep Python overhead is amortised across `n_sweeps`.

`static_argnames` for the sweep:
- `Lx_pad`, `Ly_pad` (from sequence length pads, geometric bin).
- `n_state_types = 5` (S, M, I, D, E; same as the upstream PHMM).
- `prior_mode ∈ {"crp", "bounded_eps"}` (selector for the prior
  factor in MH ratios).
- `k_max` (`-1` for `∞`, otherwise positive int).

JIT shape caching: 5 unique `(Lx_pad, Ly_pad)` pairs at typical
BAliBASE sequence lengths (≤ 200). Each precompiled once, reused
across every pair of that bin-size.

### F.3 Vectorisation

- **Vmap** over independent MCMC chains: `n_chains = 4` chains run
  in parallel via `jax.vmap` over the chain axis. This is a 4x
  speedup at constant per-chain cost.
- **Scan** over MCMC sweeps: the inner loop is a `lax.scan`. Each
  step's body is a single sweep; the carry is `(rng_key, state)`
  with `state = (alignment_path, edge_set, diagnostics)`.
- **Vmap** over chunked anchors during setup: same as
  `f2_scfg._process_anchor_chunk`.

### F.4 Padding/bins/masking

Same scheme as `aug_phmm.py` and `f2_scfg.py`:
- `_pad_to_bin` from `tkfmixdom.jax.dp.hmm` for sequence lengths
  (geometric bins).
- Padded positions masked to `NEG_INF` in log-emission tables; to
  zero in linear partial-Forward tensor.
- Padded anchors in segment-resample masked with the standard
  `_emit_mask` pattern.

### F.5 RNG handling

- The chain seed is exposed as `mcmc_seed: int = 0`. We split the
  master key into `n_chains` chain keys, then within each chain
  split into per-sweep keys (one key per sweep) and within each
  sweep into per-move keys (one key per move).
- `jax.random.split` is the standard primitive; we use the
  `(key, n_keys)` form for clarity.

### F.6 Storage of `F^partial`

- **Default mode**: dense fp32 4-tensor in GPU memory. Practical for
  `L ≤ 100` (200 MB).
- **Compressed mode**: lex-order reduction (factor 2). Practical for
  `L ≤ 130` (200 MB after compression).
- **Streaming mode**: anchor-by-anchor restart-Forward (no precompute).
  Each segment-resample move recomputes the per-anchor restart-Forward
  on demand. Cost per sweep becomes O(L^3) (one restart-Forward per
  segment, O(L^2) cost), total O(L^4) per sweep — slower than the
  precompute version for many sweeps but bounded memory. Practical
  for `L > 130`.
- **Sliding-window mode** (compromise): precompute `F^partial[i, j;
  k, l]` only for `(k, l)` within a window `(i + 1, j + 1) ≤ (k, l)
  ≤ (i + W, j + W)` for some window `W`. Segments within the window
  use the precomputed table; segments outside do a streaming
  restart-Forward. Default `W = 30`.

The implementation will support all four modes via a `storage_mode`
flag with `default = "auto"` that picks based on `L`.

---

## G. Performance plan

### G.1 Setup cost

- The O(L^4) partial-Forward precompute is the bottleneck.
- We **reuse** `f2_scfg._restart_forward_core` and
  `f2_scfg._per_anchor_kernel`, just dropping the F2-tensor
  accumulation and instead saving `mu[k, l, M]` per anchor.
- Precise cost for the `L = 100` case: 100^4 / chunk_size = 100^3 = 10^6
  per-cell operations, × 5 states × ~10 flops per cell = 5 · 10^7
  flops. On a GPU this is < 0.1 s.
- For `L = 200` the cost is ~16x: ~1 s setup. Memory is the binding
  constraint, not flops.

### G.2 Sweep cost

- Per sweep: O(L) segment-resample moves × O(L) per-move work +
  O(8 + 2|E|) edge moves × O(1) work = **O(L^2) per sweep** worst
  case, **O(L) per sweep** at convergence (when the typical segment
  length is O(1)).
- Target throughput: 1000 sweeps/s at `L = 100` (fully GPU-bound,
  ~10 µs/sweep with vectorisation/scan), 100 sweeps/s at `L = 200`.
  These are JIT-compiled scan loops; the Python overhead is one-time.

### G.3 Boost-tensor caveat for `K_c > 1`

The Boost `M` tensor is built via
`aug_phmm.build_M_tensor_aa_marginal(boost_state)` which uses the
prior class distribution (uniform `1 / K_c`) instead of the per-
position posterior class distribution. For `K_c > 1` this is an
approximation; the F2-SCFG implementation uses the per-position
posterior (`gamma[i, j]`). For the MCMC sampler we **default to the
prior-class M tensor for parity with `aug_phmm`**, but expose a flag
`use_position_M = True` that builds the per-position M field on
demand (same `_build_log_M_field_jax` from `f2_scfg.py`). The latter
is more accurate but has overhead per Match-cell visit.

### G.4 Cache reuse

- `F^partial` is invariant across sweeps within one MCMC run on one
  pair. Storage is in GPU memory for the duration of `mcmc_sampler`.
- The boost-state JAX tensor, log-trans, state-types, emission table
  are also cached for the chain.
- Across pairs, the JIT-compiled functions are cached by shape; the
  `F^partial` tensor and emission table are recomputed per pair.

### G.5 Estimated implementation effort

- **Core module** (`mcmc_infinite_phmm.py`): ~700 lines (comparable
  to `aug_phmm.py`). Includes the four storage modes, two prior
  modes, two add-proposal flavours, the diagnostic logging, the
  multi-chain runner.
- **Smoke tests**: ~400 lines (5 tests, each requiring a brute-force
  reference, a comparison harness, and a tolerance derivation).
- **Eval-harness registration**: ~30 lines added to `eval_balibase.py`.
- **Total**: ~1100 lines new + ~30 lines edits.
- **Agent-hours**: ~16 hours of careful implementation + ~8 hours of
  testing/debugging across the four cross-validation protocols. Total
  ~24 agent-hours (3 work-days at the worktree-level granularity).

This budget assumes no surprises with the F2 reuse path; if
`f2_scfg._restart_forward_core` doesn't lift cleanly to per-anchor
storage, add ~4 hours for a custom kernel.

---

## H. Open questions / decisions for the user

The plan above makes reasonable defaults for several decision points
where multiple paths exist. The user should review and confirm or
override before implementation begins.

### H.1 Strategy for segment-resample edge handling (B.1.d)

**Default: Strategy S-1 (conservative)** — segment-resample is
rejected if any edge falls inside the segment. Edges are only
created/destroyed by edge-modification moves.

Alternative: Strategy S-2 (integrated) — segment-resample also
proposes edges within the new segment. More efficient but more
complex MH algebra.

**Question**: confirm S-1, or request S-2?

### H.2 Add-proposal weighting (B.2.a)

**Default: Q_add-uniform** — pick both endpoints uniformly from
Match cells. Cost O(1) per move.

Alternative: Q_add-boosted — weight by `M`. Tighter acceptance, cost
O(L) per move.

**Question**: confirm Q_add-uniform, or request Q_add-boosted? The
user spec says "sampled from the M-weighted distribution" which
sounds like Q_add-boosted. Need to clarify whether "M-weighted"
means "weighted by the per-edge M factor" (Q_add-boosted) or
"weighted by the marginal Match probability `Q_{ij}`"
(Q_add-marginal-weighted, a third option).

### H.3 Burn-in budget (C)

**Default: `n_burnin = max(50, L / 4)`** — heuristic, sub-linear in
L.

Alternative: tied to convergence diagnostics (run until R-hat ≤ 1.05
and ESS ≥ 50, no fixed budget).

**Question**: fixed-budget burn-in OK, or convergence-driven?

### H.4 Segment-resample sweep schedule (B.1.a, B.4)

**Default: visit all `N_M + 1` adjacent anchor pairs per sweep**
(systematic scan).

Alternative: random subset (e.g. `n_seg_moves = max(8, N_M)`
sampled with replacement). Lower per-sweep cost; possibly worse
mixing.

**Question**: systematic vs random? My default leans systematic for
convergence guarantees.

### H.5 Precise definition of "M-weighted" in user spec (B.2.a)

**Question**: in the user-supplied algorithm spec, "Propose adding
a coupled edge between two random Match cells (sampled from the
M-weighted distribution)" — clarify whether
- Q_add-uniform: just two random Match cells, uniform.
- Q_add-boosted: weighted by `M(e ; t)` (this is closest to "M-
  weighted" but cost O(L) per pick).
- Q_add-marginal-Q-weighted: weighted by the baseline pair-HMM
  per-cell match posterior `Q_{ij}` (no edge boost involved).

The default I've coded is Q_add-uniform, but I think Q_add-marginal-
Q-weighted is the most natural reading of the spec. Need confirmation.

### H.6 Edges sharing endpoints (A.1, B.2)

**Question**: in the unbounded model, can two distinct edges share
an endpoint? The CRP itself allows it (each Match cell can be a
"new table" multiple times, but the table-membership semantics break
down for our edge-graph encoding).

The simplest convention is: **edges are unordered pairs of distinct
Match cells, and no two edges share an endpoint** (i.e. `E` is a
*matching* in graph-theoretic terms). This matches the size-{1, 2}
Ewens partition convention used elsewhere in TKF-DP.

If we instead allow edges to share endpoints (so a Match cell can be
the endpoint of multiple edges), the Boost product factorises but
the proposal needs to be careful (don't propose duplicate edges).
The aug_phmm_2edge.py code uses a *multiset* of in-flight endpoints
for closure, suggesting that two edges can have the same endpoint
amino-acids (but at different positions); see line 49 of
aug_phmm_2edge.py for the multiplicity-2 case. This is *positional*
multiplicity, not endpoint-sharing.

**Default**: edges are matchings (no shared endpoints). Distinct
positions, distinct edges.

**Question**: confirm matching constraint, or allow shared endpoints?

### H.7 Cross-segment edges (B.1.d)

**Question**: an edge with one endpoint inside the segment-resample
target and one outside — what does the segment-resample move do?

**Default**: segment-resample is **rejected** if any edge has one
endpoint inside and one outside (a "cross-segment" edge). This means
the chain must first edge-remove the cross-segment edge, then
segment-resample, then edge-add a (possibly different) edge.

Alternative: Strategy S-2 in B.1.d, which generalises this cleanly.

**Question**: confirm rejection, or request integrated handling?

### H.8 Strategy S-2 as experimental flag

**Question**: even if the default is S-1, do we want S-2 implemented
as an opt-in flag? Estimated extra effort: ~6 agent-hours. Will
likely be needed for very long sequences where the segment-resample
move is the bottleneck and can't easily be augmented with edge
moves.

### H.9 Multi-chain count (D)

**Default: `n_chains = 4`**.

**Question**: adjust? More chains improve `R-hat` reliability but
multiply GPU memory cost (each chain holds its own `F^partial`).

### H.10 Bounded-eps prior mode (E.2)

**Default: implement `prior_mode = {"crp", "bounded_eps"}`** as a
selector flag, defaulting to `"crp"` (the canonical CRP prior).

The `bounded_eps` mode replaces the per-Match-cell CRP factor with
the per-cell spawn weight `eps = 1 / alpha_z`, recovering the
bounded-edge model from `aug_phmm`. This is needed for the E.2
verification protocol against `aug_phmm_2edge`.

**Question**: confirm `bounded_eps` is in scope for the implementation.

### H.11 Output diagnostics format

**Default: return per-pair `(Q', L_exact_est, Q_baseline, log_F0,
diagnostics)` where `diagnostics` is a dict with keys**
`{n_chains, n_sweeps, n_burnin, ess_log_pi, ess_E, ess_NM, rhat_log_pi,
rhat_E, n_accept_seg, n_accept_add, n_accept_remove, mean_E, var_E,
mean_NM, var_NM, runtime_seconds}`.

**Question**: any additional diagnostics required? E.g. per-cell
running variance of `Q'` (would let downstream code do uncertainty-
aware FSA assembly)?

### H.12 Eval-harness defaults (F.1)

**Default**: `tkfdp_mcmc` method in `eval_balibase.py` defaults to
`n_sweeps = 1000`, `n_burnin = 200`, `n_chains = 1`, `k_max = ∞`,
`prior_mode = "crp"`. Single-chain to keep BAliBASE eval per-pair
cost similar to `tkfdp_aug` (a few seconds per pair).

**Question**: confirm these defaults, or specify alternatives?

### H.13 Wildcard amino acids (X = 20)

**Default**: same as `aug_phmm.py` — the wildcard is clamped to 19
inside the boost lookup but kept as 20 for the Pair HMM emissions.
The Boost factor at a wildcard is `M = 1` by convention (no
information).

**Question**: confirm.

### H.14 Should MCMC also estimate `L_exact`?

**Question**: should the sampler estimate the partition function
`L_exact = sum_{A, E} P_baseline · CRP · BoostProd`? This requires
either thermodynamic integration or a stepping-stone estimator —
substantially more code (~200 lines) and more agent-time. The eval
harness only consumes `Q'`, so this is mostly a curiosity / sanity
check.

**Default**: do not estimate `L_exact`. Return `L_exact_est = None`
or an obvious sentinel.

**Question**: confirm.

---

## Estimated implementation effort

- **Core module + tests + harness registration**: ~1100 lines new
  code, ~24 agent-hours. (See G.5 for the breakdown.)
- **Optional Strategy S-2 extension**: +6 agent-hours.
- **Optional `L_exact` estimator**: +6 agent-hours.

Total assuming defaults: **~24 agent-hours** (3 working days).
Total with optionals: **~36 agent-hours**.

---

## References within the codebase

- `main.tex` `\paragraph{MCMC sampler from the infinite Pair HMM}`
  (~line 798) — high-level spec.
- `main.tex` `sec:f2-scfg` (~line 713) — F_0, F_1, F_2 definitions.
- `main.tex` `sec:aug-phmm` (~line 781) — bounded-edge augmentation.
- `main.tex` `sec:infinite-hmm` (~line 788) — infinite-HMM theory.
- `src/tkfdp/f2_scfg.py` — O(L^4) anchor-pair Forward (reuse target).
- `src/tkfdp/aug_phmm.py` — 1-edge bounded approximation.
- `src/tkfdp/aug_phmm_2edge.py` — 2-edge bounded approximation
  (multiset-tag reference).
- `experiments/eval_balibase.py` — eval harness; mirror
  `run_tkfdp_aug_phmm` for `tkfdp_mcmc` registration.
