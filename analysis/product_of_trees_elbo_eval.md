# Product-of-trees structured mean-field ELBO — evaluation

Derivation: `math-paper/draft-product-trees-elbo.tex`.
Implementation: `src/tkfdp/coupling/dynfield/phylo_elbo/elbo_product_trees.py`.
Exact reference: `src/tkfdp/coupling/dynfield/phylo_elbo/exact_peel.py`
(`exact_ll_tree_general`, the m-general `L*A^m` peel, machine-precision-matched
to the validated m=2 peel `exact_cap2.exact_pair_ll_tree`).
Harness: `analysis/scripts/eval_product_of_trees.py` →
`analysis/results/product_of_trees_eval.json`.
Tests: `tests/phylo_elbo/test_elbo_product_trees.py` (all pass).

Model matches the live runs: `rho=[0.6,0.4]`, `rho_chain=0.15`, LG08 `(S, pi)`,
C20 archetypes, enum400 class→arch mapping `c → (c//K_a, c%K_a)`. Classes for
each family from `results/enum400dm_discover_train470/_chkpt.npz`. Real tree =
`data/pfam_processed_clv_top1000_thin128/PF02457.npz` (128 leaves, 254 nodes),
leaf residue = `leaf_msa` (gap → −1).

## What this bound is

`q(θ,Δ,x_1..x_m) = q_Θ(θ,Δ) · ∏_n q_n(x_n)`: a **product of independent
tree-structured factors** — one tree-Markov law over the field-and-jumps
configuration, and, per site, a **full** A-state tree chain over that site's
residue trajectory (not per-node marginals). The factors are independent
(residues **not** conditioned on the field); coordinate ascent synchronises
them. Cost is `O(m·N·A² + N·L²)` — **linear in m**, never `A^m`. The one
approximation is that it **severs the field↔residue seam** (and the
jump-reset coupling residues have to Δ). It is a genuine lower bound always,
and exact only when emissions are field-independent **and** there are no jumps
(ρ_chain=0, or the degenerate L=1); see the derivation.

## Headline

- **It is a valid bound** everywhere tested (`ELBO ≤ exact`, verified against the
  exact peel on every case; unit tests enforce it).
- On the real 128-leaf tree it is **~15–20× tighter in absolute nats than the
  previous per-site bound** (`elbo_persite`). On the confirmed-flip pair
  PF02457 (19,140) the previous bound is **72.15 nats** loose (reproducing the
  documented ~72-nat baseline exactly); product-of-trees is **3.32 nats** loose.
- **Decisive metric** (pairing-difference error, PART B): see below.

---

## PART A — absolute exact−ELBO gap on the PF02457 128-leaf tree

Head-to-head against `elbo_persite` (the ~72-nat baseline), same cases.

### m = 1 singletons (gap = exact − ELBO, nats)

| col | exact | PoT gap | persite gap |
|----:|------:|--------:|------------:|
|  19 | −222.59 | **2.18** | 8.37 |
| 140 | −152.92 | **0.97** | 54.45 |
|  30 |  −47.36 | **4.96** | 96.51 |
|  90 | −281.12 | **1.31** | 0.70 |
| 120 |  −70.20 | **8.48** | 87.49 |
| **mean** | | **3.58** | **49.50** |

### m = 2 pairs (gap = exact − ELBO, nats)

| (i,j) | exact | PoT gap | persite gap |
|:-----:|------:|--------:|------------:|
| (19,140) | −369.04 | **3.32** | 72.15 |
| (90,120) | −349.93 | **5.00** | 46.50 |
| **mean** | | **4.16** | **59.32** |

**Reading.** `elbo_persite` is catastrophically loose on deep real trees
(49–96 nats): it effectively crushes the residue trajectory toward per-node
marginals, so the substitution likelihood along ~250 branches is badly bounded.
Product-of-trees keeps the **full** residue tree chain per site, making the
per-site residue marginal likelihood essentially exact; the only residual
looseness (a few nats) is the dropped field↔residue seam. One case (col 90)
is a tie — where the field posterior is near-certain, the seam costs little and
`elbo_persite`'s per-node residue crush also happens to be mild.

---

## PART B — pairing-difference error (the decisive metric)

`Δ = LL(pair, shared field) − LL(i alone) − LL(j alone)` is the model's
coevolutionary coupling score for a column pair. The error that matters for a
scorer is `Δ_ELBO − Δ_exact`, over the confirmed-flip pairs
(`confirmed_flips.json`) that have both a thinned tree and discovered classes.

58 confirmed-flip pairs (those with both a thinned tree and discovered classes).

| method | mean | std | \|mean\| | \|max\| |
|:--|--:|--:|--:|--:|
| **product-of-trees** | +1.19 | **2.09** | **1.81** | **10.00** |
| `elbo_persite` (this run) | −0.46 | 9.35 | 6.72 | 26.47 |
| `elbo_persite` (brief baseline) | −3.3 | 10.6 | — | 17.6 |

**Product-of-trees is the clearly better coupling scorer**: its coupling-score
error has **std 2.1 nats** versus `elbo_persite`'s **9.3** — a ~4.5× reduction
in spread — and a typical (|mean|) error of 1.8 vs 6.7 nats. The large absolute
per-site gaps in PART A largely **cancel in the difference** (pair and
singletons are loose by similar amounts), so what survives is a small, low-
variance coupling error. My `elbo_persite` reproduction (std 9.3, |max| 26.5)
sits in the same ballpark as the documented baseline (std 10.6, |max| 17.6);
the subset of pairs differs, which explains the |max| gap.

**The bad cases (not cherry-picked).** Product-of-trees' worst pair is
PF02561 (77,102), error −10.0 nats (this is the |max|), where dropping the seam
under-counts a strong coupling (`Δ_exact = −14.3`); on that pair `elbo_persite`
happens to be near-exact (+1.5). So the seam does occasionally matter for the
coupling itself — product-of-trees wins on spread, not on every pair.

---

## PART C — m = 1..4 tightness, runtime, convergence (reduced 6-letter model)

To get a tractable exact `L·A^m` reference at m=3,4 we reduce the alphabet to
**A=6** physicochemical groups (a pure tractability device: the real PF02457
tree topology and branch lengths are kept; leaf residues are the real columns
collapsed to 6 groups; reduced archetypes are the group-summed C20 profiles;
`S` is flat/F81). All three methods and the exact peel run on the identical
reduced model, so the comparison is apples-to-apples.

Columns [19,140,30,90], nested (m=1 uses [19]; m=4 uses all four).

| m | exact | PoT gap | persite gap | warm gap | conv gap | sweeps | t(PoT) | t(persite) |
|--:|------:|--------:|------------:|---------:|---------:|-------:|-------:|-----------:|
| 1 | −139.34 | **1.37** |   2.02 | 1.88 | 1.37 | 13 | 0.51s | 0.96s |
| 2 | −273.15 | **0.06** |   2.48 | 1.91 | 0.06 |  5 | 0.29s | 1.68s |
| 3 | −310.60 | **0.19** |  76.05 | 7.76 | 0.19 |  6 | 0.51s | 4.04s |
| 4 | −520.50 | **2.10** |  61.15 | 10.09 | 2.10 |  6 | 0.69s | 4.42s |

- **Tightness stays bounded as m grows** (PoT gap ≤ 2.1 nats through m=4), while
  `elbo_persite` **blows up at m≥3** (76 nats): the per-node residue crush
  compounds across sites. Note the exact `L·A^m` peel is only tractable here
  because A=6; at the real A=20 it is `2·20^4 = 320k` states/node (m=4) —
  product-of-trees remains `O(m·N·A²)`.
- **Runtime is essentially flat/linear in m** for product-of-trees
  (0.3–0.7 s), versus `elbo_persite` growing 1→4.4 s. The exact peel is cheap
  only at A=6.
- **Convergence is monotone** on the real 128-leaf tree (verified for every m),
  in 5–13 sweeps.
- **Warm start is a loose starting point at larger m** (warm gap 7.8–10.1 nats
  at m=3,4 vs converged 0.2–2.1): the field-prior-averaged residue posteriors
  ignore the residue evidence's pull on the field, so coordinate ascent is
  **essential**, recovering ~8 nats over 5–6 sweeps. At m=1,2 the warm start is
  already within ~2 nats.

---

## Honest assessment

**It is a usable scorer, not just a paper trophy — with caveats.**

- **Tight where it counts.** On real 128-leaf trees the absolute gap is a few
  nats and, decisively, the *pairing-difference* error (the quantity a
  coevolution scorer actually uses) has std 2.1 nats and |max| 10 — a 4.5×
  improvement over `elbo_persite`, whose corresponding error (std 9.3) makes it
  useless as the brief noted. The gap stays bounded through m=4 where
  `elbo_persite` blows up (76 nats), which is the whole point: it is the ELBO
  that stays tractable **and** tight for m>2.

- **Where it is loose.** (1) It is inexact even at m=1 and at ρ_chain=0 with
  field-dependent emissions — it drops the field↔residue seam, unlike the
  seam-keeping `elbo_persite`/`elbo_treestruct`. Empirically that seam costs a
  few nats per site. (2) The *sign* of the miss is systematic on the coupling:
  PoT's mean pairing error is +1.2 (it slightly **over**-scores independence /
  **under**-scores coupling on average), and on the single worst pair it
  under-counts a strong coupling by 10 nats. (3) The warm start is poor at
  m≥3, so it must be run to convergence (cheap: <1 s, 5–6 sweeps).

- **Incomparable to the node-mean-field floor.** Product-of-trees and `elbo.py`
  are not ordered: PoT keeps along-tree residue correlation but drops the
  per-node seam; the floor keeps the seam but drops the correlation. On deep
  real trees the along-tree correlation dominates, so PoT is far tighter in
  practice, but there is no theorem that PoT ≥ floor (and it fails on small
  high-ρ_chain trees).

- **Verdict.** Good enough to be a scorer for the cap-2 flip task (std-2-nat
  coupling error) and the *only* tractable option with controlled tightness for
  m>2. It is not a drop-in replacement for the exact cap-2 peel where that peel
  is affordable (A=20, m=2: the exact peel is ~0.1 s and exact). Its niche is
  m≥3 clusters and as a fast, tight, monotone lower bound for training-time
  scoring — which is exactly the regime the exact peel cannot reach.

## Reproduce

```
JAX_PLATFORMS=cpu PYTHONPATH=src python3 analysis/scripts/eval_product_of_trees.py
JAX_PLATFORMS=cpu PYTHONPATH=src python3 -m pytest tests/phylo_elbo/test_elbo_product_trees.py -q
```
