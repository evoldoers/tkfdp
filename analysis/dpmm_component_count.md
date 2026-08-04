# How many coupling components (pairing archetypes) does the stationary count data support?

A cheap, stationary-only probe. The number of components is a stationary
(rate-invariant) question, so it needs only the tree-weighted per-contact
amino-acid-**pair** count tables, not the CTMC. We fit a Dirichlet-process
mixture of Dirichlet-multinomials (and, as a robust cross-check, the finite
Holmes-Harris-Quince finite DMM with evidence-based K selection) to the
symmetric 210-unordered-pair count vector of each structural contact, and read
off the supported number of components.

Code: `experiments/dpmm_column_pairs.py` (`build`, `fit`, `overdisp`
subcommands). Reports: `results/dpmm_column_pairs/report.json` (main sweep +
DP) and `.../overdisp.json` (the overdispersion regime study). Run with
`PYTHONPATH=src`.

## TL;DR headline

**The stationary pair-count data do not resolve a clean number of archetypes at
all -- the composition landscape is a continuum, and the apparent "many
components" is largely a within-archetype overdispersion artifact.** Three
findings:

1. **Overdispersion <-> K trade off, and it dominates the answer.** Component
   count and per-component Dirichlet concentration (overdispersion) are
   confounded. Forcing near-multinomial components (concentration B = 1e4)
   pins K* at the sweep ceiling (>=50), doubles the DP occupancy
   (occ_K ~ 94 vs 52 at matched alpha), and makes the between-archetype
   variance *look* 4x larger. When the concentration is **fit freely** the data
   choose a strongly overdispersed solution (B ~ 40, ~10x the multinomial
   variance) -- yet K* still does not turn over below 50, because the
   composition is a continuum: only when we *impose* excess overdispersion
   (B = 8) does held-out K* finally drop (to ~32), at a fit cost.

2. **The honest between-vs-within split says "no discrete archetypes".** At the
   free (data-chosen) overdispersion, between-archetype means explain only
   **~8%** of the per-contact composition spread at K=50 (vs 0.33 forced
   near-multinomial), and that fraction **keeps rising with K with no plateau**
   -- the other ~92% is within-archetype overdispersion. There is no natural
   cluster count; the data tile a continuum.

3. **The components separate by amino-acid COMPOSITION, not by COUPLING.** The
   interpretable archetypes are composition classes (salt-bridge-enriched,
   hydrophobic core, aromatic, glycine/small, proline), each with **low coupling
   MI** (median 0.016, max 0.071 nats at K=20; weighted-mean wMI ~ 0.02-0.035,
   <15% of the bias-corrected per-contact coupling of ~0.25 nats). Pooling
   single-sequence pair frequencies into composition classes washes coupling
   out, exactly as the pooled-tensor experiments already found.

**Bottom line: "how many components" has no clean answer from stationary counts.**
Forced near-multinomial => "O(50+), unbounded" (the inflated artifact);
overdispersion fit => a continuum (no turnover for the data's own B ~ 40; a
small imposed concentration is needed to force even K ~ 32), dominated
~92% by within-archetype scatter. And none of these are *coupling* archetypes --
coupling stays washed out. The clean coupling-component count needs the dynamic
(transition) data, not stationary counts.

---

## 1. Data: tree-weighted per-contact pair-count tables

Built from the CherryML/trRosetta 15,051-family set (contact definition reused
from `build_cherry_counts_trrosetta`: Cbeta < 8 A, |i-j| >= 7, greedy maximal
matching so each column is in <=1 contact). For each family we subsample the
MSA to <=512 sequences, compute the **theta=0.8 identity reweighting**
(w_s = 1/#{seqs >= 80% identical to s}; the cheap tree-weighting proxy for row
novelty), then for each contact (colA,colB) tally the weighted joint counts of
the 20x20 amino-acid pairs over sequences and **fold to the 210 unordered
pairs** (exchangeable: C[a,b] and C[b,a] land in the same bin). Contacts with
< 25 effective counts are dropped.

Corpus (2,971 families, `--max-fam 3000`): **240,632 contacts**.

| quantity | value |
|---|---|
| contacts | 240,632 |
| families | 2,971 |
| effective counts N_eff (weighted): median / mean | 402 / 356 |
| N_eff 5% / 10% / 90% quantiles | 66 / 112 / 475 |
| fraction of contacts with N_eff >= 50 / >= 100 | 0.97 / 0.91 |
| redundancy N_raw / N_eff: median / mean / p90 | 1.15 / 1.23 / 1.43 |

The redundancy factor (>1) is exactly the tree-weighting effect: raw sequence
counts overstate the independent evidence by ~15-23% at the median (up to ~43%
in the most redundant families), which is what inflates apparent components if
left uncorrected. (The subsample-to-512 caps how much redundancy is even
visible; on full MSAs it would be larger.)

The fit is run on an 80,000-contact random subsample (the evidence-optimal K
grows only ~log N, so a subsample answers the count question at a fraction of
the cost; the `overdisp` study uses 30,000).

Per-contact coupling MI (grounding the ~0.16 figure from the dynamic-model
work), computed the same way as the component MIs (`mi_of_joint`):

| estimator | weighted | note |
|---|---|---|
| plugin | 0.525 | finite-sample biased UP |
| independent-null floor (resample at same N_eff) | 0.278 | the bias |
| **bias-corrected (plugin - null)** | **0.245** | honest per-contact coupling |
| Miller-Madow | 0.402 | undercorrects (observed-support only) |

So the honest per-contact coupling is ~0.25 nats -- same order as the ~0.16
quoted previously (the difference is estimator/units; single-sequence pair
frequencies are more biased than the pooled/transition estimates).

---

## 2. Finite DMM sweep and DP occupancy (free concentration)

Finite DMM by EM (Minka fixed-point M-step = joint MLE of mean **and**
concentration), family-split held-out predictive, BIC, and an HHQ-style
Laplace evidence. Weighted counts, 80,000 contacts:

| K | held-out / contact | BIC | wMI (coupling captured) | K_eff |
|---:|---:|---:|---:|---:|
| 1 | -1230.4 | 157,056,874 | 0.0045 | 1.0 |
| 2 | -1216.6 | 155,282,796 | 0.0039 | 2.0 |
| 4 | -1209.3 | 154,350,054 | 0.0083 | 4.0 |
| 8 | -1205.2 | 153,831,727 | 0.0120 | 7.3 |
| 12 | -1202.8 | 153,518,689 | 0.0141 | 10.4 |
| 16 | -1200.7 | 153,268,019 | 0.0181 | 14.7 |
| 20 | -1199.7 | 153,156,346 | 0.0195 | 17.4 |
| 25 | -1198.7 | 153,027,075 | 0.0227 | 21.5 |
| 30 | -1197.9 | 152,931,923 | 0.0267 | 25.6 |
| 40 | -1196.4 | 152,761,489 | 0.0314 | 34.7 |
| 50 | -1195.3 | 152,646,610 | 0.0346 | 43.6 |

**K* = 50 by all three selectors (held-out, BIC, Laplace) -- i.e. it never
turns over inside the tested range.** The reason is diagnostic: the BIC penalty
per added component is only ~210 log(80000) ~ 2,400 (in 2*NLL units), while the
data likelihood keeps improving by ~28,000 per component -- more than 10x the
penalty. With free concentration this is *not* a near-multinomial artifact (see
Section 3), it is a genuine composition continuum. Held-out per-contact gains
shrink smoothly (13.8 -> 7.3 -> 4.1 -> ... -> 0.1 per group) but stay positive.

DP mixture (collapsed Gibbs, fixed Dirichlet base measure, MAP concentration
refit; weighted, 8,000-contact subsample), occupied-K vs alpha:

| alpha | 0.3 | 1 | 3 | 10 | 30 | Gamma(2,1) hyperprior |
|---|---:|---:|---:|---:|---:|---|
| occupied K | 30.9 | 37.8 | 59.6 | 64.3 | 85.8 | 58.3 (inferred alpha ~ 7.2) |

The DP agrees with the DMM that the count landscape supports **many** occupied
components (tens), growing with alpha and with N (occupied-K ~ alpha log N under
the CRP; an 8,000-contact subsample understates the full-corpus count). But
Section 3 shows this "many" is the overdispersion-confounded number, not a count
of distinct archetypes.

---

## 3. The overdispersion <-> K confound (the real answer)

Component count and per-component overdispersion trade off directly: a
near-multinomial (under-dispersed) component cannot absorb per-contact
variability, so the model splits it; a broad (overdispersed) component absorbs
that scatter without splitting. We sweep K = 1..50 under four concentration regimes
(weighted, 30,000 contacts). `frac_between` is the between-archetype fraction of
the (sampling-noise-corrected) composition spread at K=50; `wMI` is the
weighted-mean per-component coupling MI at K=50:

| regime | forced B | K*_heldout | K*_BIC | fitted B (med) | frac_between (@K=50) | wMI (@K=50) |
|---|---:|---:|---:|---:|---:|---:|
| near-multinomial | 10,000 | **50** (ceiling) | 50 | 10,000 | 0.329 | 0.123 |
| intermediate | 300 | **50** (ceiling) | 50 | 300 | 0.297 | 0.112 |
| **free (data-chosen)** | -- | **50** (ceiling) | 50 | **41** | **0.084** | **0.035** |
| high-overdispersion | 8 | **32** | 32 | 8 | 0.133 | 0.049 |

Free-regime between-fraction vs K (weighted): 0.008 (K=2), 0.019 (K=4),
0.030 (K=8), 0.051 (K=16), 0.066 (K=32), 0.084 (K=50) -- **monotone, no
plateau**.

- **The freely-fit concentration is B ~ 40**, ~10x the multinomial variance
  (rho = 1/(B+1) ~ 0.024; per-contact fitted B_k range 3-111 at K=20). The data
  choose strong overdispersion on their own; the components are nowhere near
  multinomial. So K* = 50 is **not** a near-multinomial artifact -- it is a
  genuine composition continuum that a single per-component Dirichlet cannot
  absorb, so splitting keeps helping.
- **Only imposing excess overdispersion collapses K.** Free (B ~ 40) and
  everything more concentrated pin K* at the ceiling; only B = 8 (more
  overdispersed than the data's own choice) makes held-out K* turn over, at 32,
  and it costs held-out likelihood. So overdispersion and K trade off
  continuously and there is no clean K at the data's ML overdispersion.
- **Variance decomposition (the honest "how many archetypes").** Under the
  **free** fit the between-archetype means explain only **~8%** of the composition
  spread at K=50, and that fraction is still climbing (no plateau) -- the rest is
  within-archetype overdispersion. Force **near-multinomial** and the same
  decomposition attributes **~4x more** to "between" (0.33), because the
  overdispersion that was absorbing the scatter is now forbidden -- a pure
  artifact of the constraint.
- **DP occupancy inflates under near-multinomial refit**: at alpha = 1 on 8,000
  contacts the DP occupies **52** components with the overdispersion fit vs
  **94** forced near-multinomial (~1.8x). (Absolute DP occupancy is
  base-measure- and alpha-dependent; the *ratio* is the robust signal.)

### Tree-weighting

Redundancy makes counts look under-dispersed (near-identical rows sharpen the
count vector), which pushes the fit toward the near-multinomial / high-K regime.
In principle proper theta=0.8 weighting lowers the effective N, raising the
fitted overdispersion and lowering K. **Here the effect is small**, because the
512-sequence MSA subsample already caps redundancy to a median 1.15x
(Section 1):

| | mean N | free-fit B (median) | overdispersion (N+B)/(B+1) | K*_heldout | DP occ_K (alpha=0.3 / 10) |
|---|---:|---:|---:|---:|---:|
| weighted (theta=0.8) | 357 | 31-34 | ~11-12x | 50 (ceiling) | 30.9 / 64.3 |
| unweighted | 422 | 29-30 | ~14-15x | 50 (ceiling) | 38.4 / 80.9 |

The fitted concentration barely moves (B ~ 30 both) and K* pins at the ceiling in
both, but the **DP occupancy trends higher un-weighted** (38 vs 31 at alpha=0.3;
81 vs 64 at alpha=10) -- redundancy does inflate apparent components, in the
predicted direction, though modestly here. The effect is muted because the
512-sequence subsample already caps redundancy to ~1.15x median; on full
(un-subsampled) MSAs it would be larger, and the extreme of this direction is the
near-multinomial regime, which doubles DP occupancy (above). Net: tree-weighting
matters for honesty (it removes ~15-23% phantom evidence at the median and lowers
DP occupancy ~15-25%) but does not, on its own, resolve the component count.

---

## 4. Interpretability of the archetypes

Component means are symmetric 20x20 joints; we score each for coupling MI and
biophysical signatures (salt bridge D/E<->K/R, aromatic F/W/Y, disulfide C-C,
hydrophobic core, size matching). Weighted fit at K=20 (representative;
component MI: min 0.005, median 0.016, max 0.071 nats; wMI = 0.020):

| weight | fitted B | MI | signature | top pairs |
|---:|---:|---:|---|---|
| 0.095 | 110 | 0.005 | salt-bridge (+/-) | EK DK ES DR DS |
| 0.075 | 17 | 0.007 | hydrophobic core | IL LV IV LL II |
| 0.057 | 22 | 0.039 | hydrophobic core (A-rich) | AV AL AI LV IV |
| 0.044 | 53 | 0.017 | salt-bridge (+/-) | EK ER DR KS KT |
| 0.043 | 20 | 0.027 | aromatic + hydrophobic | FL LY FV FI VY |
| 0.038 | 58 | 0.015 | proline | AP PS EP PT DP |
| 0.036 | 60 | 0.016 | aromatic (Tyr) | AY RY LY KY SY |
| 0.025 | 18 | 0.048 | aromatic (Trp) | AW SW LW AF AY |
| (others) | 3-111 | 0.005-0.071 | glycine/small, polar/surface, charged-mixed | AG GS ..., AS AT ... |

The recurring, interpretable axes are **amino-acid composition** classes
(buried hydrophobic core; aromatic; salt-bridge-enriched; glycine/small loops;
proline; polar/surface), each with **low internal coupling MI**. The archetypes
are physically sensible, but even the most-coupled component's MI is small
(<=0.07 nats): pooling contacts by composition does not isolate a coupling
*direction*, because within any composition class the specific coupling pairs
point different ways and average toward the marginal chemistry. Notably there is
**no disulfide (C-C) archetype** -- cysteine-cysteine contacts are too rare to
survive as their own component and are absorbed into the small/polar classes;
the clearest "coupling-interpretable" class (salt bridge) still has MI only
~0.005-0.017 because its mean is dominated by the marginal charged composition,
not the complementary pairing.

---

## 5. Comparison to the dynamic (substitution-side) mixture

The dynamic free-S mixture over the same size-2 contacts reaches
**MI(pi_c) 0.03 -> 0.29 over K = 1..8** (and was still rising) because it fits
**transition** dynamics (i,j)->(k,l), which resolve coupling directly, and each
pi_c is a full 400-state coupled stationary. This stationary DMM's per-component
coupling stays **wMI ~ 0.02-0.035** at K up to 50 -- an order of magnitude less
-- because single-sequence pair *frequencies* pooled into composition classes
barely express coupling.

**So the two probes answer different questions.** The stationary count data say
the *composition* landscape is a continuum -- no clean component count at the
data's own overdispersion (K* pinned at 50+; between-archetype variance small
and non-saturating), inflating to "O(50-100+)" only if you forbid overdispersion.
They do **not** resolve the number of *coupling* archetypes at all -- that
requires the transition/dynamic data, where the free-S mixture shows the coupling
MI climbing 0.03 -> 0.29 and still rising at K=8. Stationary single-sequence pair
counts are simply the wrong instrument for counting coupling components: their
component structure is composition, and coupling is washed out.
