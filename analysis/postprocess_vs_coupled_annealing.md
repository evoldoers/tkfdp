# Precorrection vs. coupled annealing: a critical comparison

We have two distinct mechanisms for incorporating the trained TKF-DP
Potts coupling into a multiple-sequence-alignment pipeline that
otherwise consumes pair-HMM posteriors:

1. **Precorrection** (Section "Pairwise Alignment Postprocessing" of
   `main.tex`, implemented in `src/tkfdp/postprocessing.py`). Each
   pair posterior matrix `Q_{ij}` is corrected once via a mean-field
   formula that folds the marginal Potts boost `M(i,j;i',j')` into a
   per-residue-pair multiplicative factor on the Match-state log-
   emission table; a fresh forward--backward then renormalizes back
   to a valid pair-HMM posterior `Q'_{ij}`. The downstream sequence
   annealer is unchanged.

2. **Coupled annealing** (Section "Sequence annealing with
   coevolutionary scoring", implemented in
   `src/tkfdp/coupled_annealing.py`). The pair posteriors are left
   uncorrected. The greedy column-merging step inside the annealer
   is extended to consider pairs of merges scored jointly with
   `log Q_1 + log Q_2 + log M(...)`. Single-edge candidates and
   coupled candidates share one priority queue; coupled candidates
   demote to single-edge candidates when one half is invalidated by
   an intervening commit.

Both methods consume the same trained per-class-pair joint emission
tensor (`build_per_classpair_joint_emit` from
`src/tkfdp/postprocessing.py`); only the application is different.

## Statistical efficiency

Precorrection is a linearization of the size-{1,2} Ewens partition
prior at strength `eps = 1 / (alpha_z + L_aln - 1)`. Marginalizing
the perturbed log-posterior factor `log(1 + eps (M - 1))` over the
mean-field approximation `<A_{i'j'}> ~ Q_{i'j'}` produces a single
boost field that is correct to first order in `eps` and to mean-
field order in the partner-occupancy distribution. Its principal
failure mode is at columns participating in a single strong
coevolutionary partnership of magnitude `|M - 1| = O(1)`, where the
linear expansion underestimates the saturation of `log(1 + eps (M -
1))`. Iterating the mean-field map for two or three steps recovers
most of the missing mass; for the residual gap on a small set of
dominant edges, a Bethe-style cluster correction at the dominant
edge would be required. The estimator is downstream-agnostic: any
consistency-based assembler that consumes pair posteriors gets the
benefit.

Coupled annealing keeps `M` on its native four-residue domain — no
marginalization is performed before the greedy commit. For a
candidate pair whose isolated TGF weights are both large and whose
`log M` is also large, the joint priority is amplified by exactly
the right factor. The scheme is exact at the level of a single
coupled commit and never linearizes. Its principal failure mode is
that the greedy schedule is myopic: the priority queue only sees
edges and pairs already known to it, and an early commit on a
high-priority coupled candidate can lock out a globally better
arrangement that would have been reached by a different commit
order. The myopia is the standard cost of any greedy-pairwise
assembler and is shared by the baseline AMAP algorithm; the
coupled extension amplifies it because each commit now affects
two columns at once and the combinatorics of conflict-driven
demotion are richer.

Neither method is a strict approximation of the other; they are
two distinct posterior surrogates that happen to share an
underlying Potts construction. Empirically, on weakly-coupled
families (small `|log M|` everywhere), the precorrection's
linearization is tight and a downstream assembler is hard to
distort, so precorrection should slightly outperform; on
sharply-coupled families (a few large `|log M|` edges with low
overall coupling density), coupled annealing's exact joint score
should outperform.

## Computational cost

Precorrection costs `O(L_X^2 L_Y^2 K_c^2 A^2)` per sequence pair
(equation `eq:Q-prime` in the paper), one-shot per pair, plus the
fresh forward--backward at `O(L_X L_Y)`. The `O(L^4)` factor
dominates and is the same as the worst-case enumeration of all
`(i, j, i', j')` quartets, but the contraction is vectorized into
two batched einsums and runs on the GPU in milliseconds for
`L ~ 100`.

Coupled annealing has worst-case complexity `O(L^4)` candidate
pairs per sequence pair, multiplied by a `K_c^2 A^4` per-candidate
score evaluation (precomputed from the same joint emission tensor),
multiplied by `O(L)` greedy commits with TGF dynamic recalculation,
giving a worst-case `O(L^5)` cost. Two prunings restore
tractability:

- *Threshold pruning* (`q_min`): drop coupled candidates unless
  both component posteriors exceed `q_min`. In practice for
  alignments where the pair posterior is concentrated on `O(L)`
  high-confidence cells, this restricts the candidate set to
  `O(L^2)` and the per-step cost to `O(L^2)` amortized.

- *Boost pruning* (`mu_min`): drop coupled candidates whose
  `|log M| < mu_min`. Pairs that fail this test contribute
  negligibly above their decoupled score and are processed as
  ordinary single-edge candidates.

Under both prunings the implementation runs in seconds per pair
on a CPU for `L ~ 100`. The big-O scaling difference between the
two methods is small in this regime; we expect precorrection to
become substantially cheaper at `L >> 100` because its cost is
matrix-shaped and GPU-friendly while coupled annealing's pop loop
is irreducibly sequential.

## Failure modes

| failure mode                | precorrection                          | coupled annealing                                |
|-----------------------------|----------------------------------------|--------------------------------------------------|
| saturated single edge       | underestimates by `eps` linearization  | exact at the edge, but greedy near it            |
| many weak edges             | accurate (mean-field is appropriate)   | noise-amplifying via coupled-priority distortion |
| sharp single contact pair   | requires fixed-point iteration / Bethe | scored exactly, committed atomically             |
| inconsistent coevolution    | broadcast: averages over partners      | locks in early winner, may be wrong              |
| sparse posterior support    | unaffected (boost field still defined) | candidate set may collapse                       |

The clearest distinction is at the boundary where the coupled
extension's *exactness* and the precorrection's *globality* point
in opposite directions. On a family with two genuinely strong
coevolutionary edges that compete for the same columns,
precorrection's mean-field assigns a softer boost to each and lets
the assembler decide; coupled annealing commits one of them
greedily and forecloses the other. If the greedy choice is
correct, coupled annealing wins decisively; if not, it loses by a
larger margin than precorrection ever can.

## When each method is likely to win

We expect the following empirical ordering on Pfam families:

- **Compact globular domains with many resolved contacts**
  (e.g.\ small all-beta folds, hyperthermophile-stabilized
  scaffolds): coupled annealing should slightly outperform
  precorrection because each contact is sharp and the joint
  scoring is exact. The risk of greedy-locking is mitigated by
  redundant contacts that all push the same direction.

- **Diffuse coevolution from secondary-structure constraints**
  (e.g.\ alpha-helical bundles, intrinsically disordered domains):
  precorrection should outperform because the per-edge `|log M|`
  is small, the linearization is tight, and the global
  consistency assembler can integrate the broadcast boost without
  greedy distortion.

- **Families with no detectable coevolution** (e.g.\ very short
  motifs, signal peptides, families where the TKF-DP H atom
  reduces to near-zero post-fitting): both methods should reduce
  to baseline, with coupled annealing slightly noisier in
  practice because it admits coupled candidates whose `|log M|`
  is genuine but small and statistically meaningless.

## Empirical results on BB11001 (BAliBASE Reference 1.1)

The integration test `tests/test_balibase_coupled_annealing.py`
runs all three methods on the BB11001 family (4 sequences,
~83--91 aa, short DNA-binding helices). The `K=4` checkpoint
loaded is the best of the experiment 2 emwarm run on the top-1000
Pfam families.

```
                                  method        SP        TC
                                baseline    0.8683    0.8816
                       precorr alpha_z=1    0.8395    0.8421
                      precorr alpha_z=10    0.8642    0.8816
                     precorr alpha_z=100    0.8663    0.8816
                    precorr alpha_z=1000    0.8683    0.8816
            coupled q_min=0.1 mu_min=0.1    0.8025    0.7763
            coupled q_min=0.1 mu_min=0.5    0.8025    0.7763
              coupled q_min=0.1 mu_min=1    0.8395    0.8421
              coupled q_min=0.1 mu_min=2    0.8683    0.8816
            coupled q_min=0.2 mu_min=0.1    0.8189    0.8026
            coupled q_min=0.2 mu_min=0.5    0.8189    0.8026
              coupled q_min=0.2 mu_min=1    0.8621    0.8816
              coupled q_min=0.2 mu_min=2    0.8683    0.8816
            coupled q_min=0.4 mu_min=0.1    0.8169    0.8026
            coupled q_min=0.4 mu_min=0.5    0.8066    0.7895
              coupled q_min=0.4 mu_min=1    0.8086    0.7895
              coupled q_min=0.4 mu_min=2    0.8683    0.8816
            coupled q_min=0.6 mu_min=0.1    0.7901    0.7500
            coupled q_min=0.6 mu_min=0.5    0.7901    0.7500
              coupled q_min=0.6 mu_min=1    0.8107    0.7895
              coupled q_min=0.6 mu_min=2    0.8683    0.8816
```

BB11001 is a "no detectable coevolution" case — neither method
improves on the baseline; precorrection is neutral at large
`alpha_z` (where `eps` shrinks the boost to nothing), and coupled
annealing is neutral at large `mu_min` (where the boost-pruning
filter eliminates all coupled candidates). At smaller `mu_min`,
coupled annealing actively hurts by 5--8 SP points, which is the
predicted noise-amplification mode for a no-coevolution family.
The asymmetry — precorrection degrades gracefully toward baseline
while coupled annealing degrades sharply with more aggressive
pruning thresholds — is the central empirical signature: the
greedy schedule is unforgiving when the underlying signal is
weak.

This single-family result is consistent with the predictions
above, but is not a definitive characterization. A larger
benchmark sweep (BAliBASE Reference 1.1 + 3, contact-rich
domains) is needed to confirm that on contact-rich families the
ordering reverses. That benchmark is left as future work.
