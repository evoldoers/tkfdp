# Does the K_c=16 "all-fix" dynfield use theta dynamically? Follow-up

Diagnostics on `results/dynfield_top100_K16_L4_allfix_2026-06-30/_best_chkpt/state.npz`
(K_c=16, L_max=4, learned rho_chain=0.211, learned alpha_z=17.17,
gap-marginalised cluster likelihood; top-100 Pfam, best val LL
-4.65e6). Same six diagnostics as `dynfield_field_flip_diagnostics.md`;
K_c=4/8 rows reproduce that writeup under its strict criterion.

## Headline finding

**The K_c=16 model has moved a real but modest step toward dynamic
theta-use, and the effective evidence base is 10x-25x larger.** Median
P(nn=static) drops from 0.999 (K_c=4) / 1.000 (K_c=8) to 0.981; the
fraction of records posting P(static) >= 0.99 falls from 70-80% to
42%; 84 size-2 clusters now show flipping compositions with mean
non-static mass > 0.1 (vs 8 at K_c=4, 1 at K_c=8). **Strict-valid
records go from 6368 / 2747 to 71256, and gap-marginalised valid
records reach 373595 across 1807 clusters** -- the model is now
constrained by a real corpus. But no record anywhere posts P >= 0.5
on any single jump case, and median cluster theta_P entropy at small
clusters sits at the H(rho) prior. Static-specializer regime,
weakened.

## A. Per-(cluster, cherry) 4-case posterior

Under gap-marginalisation the K_c=16 model has 373595 valid records
(mean P(nn, ny, yn, yy) = (0.906, 0.045, 0.045, 0.004); ny/yn identical
by cherry time-symmetry).

| model | n gap-marg | median P(nn) | 25% | 75% | %>=0.95 | %>=0.99 | %(1-P)>=0.20 |
|---|---|---|---|---|---|---|---|
| K_c=4  | 385970 | 0.9994 | 0.979 | 1.000 | 80.3% | 70.1% | 11.0% |
| K_c=8  | 239323 | 1.0000 | 0.996 | 1.000 | 86.3% | 79.6% |  8.2% |
| K_c=16 | 373595 | 0.9806 | 0.890 | 0.998 | 63.4% | 41.7% | 16.0% |

K_c=16 is the first with a non-degenerate lower quartile (0.890 vs
~1.000 at K_c=4/8). Under the strict "all cluster columns observed"
criterion used by the original writeup: K_c=4 has 6368 records
(median 0.9987), K_c=8 has 2747 (median 0.9996), K_c=16 has **71256**
(median 0.9892, %>=0.99 = 48.9%). No record anywhere puts >= 0.5 mass
on a single jump case.

## B. Per-cluster theta_P posterior entropy

Prior entropies: H(rho_K4)=1.764, H(rho_K8)=1.920, H(rho_K16)=1.986
bits (uniform=2.0). The K_c=16 rho [0.200, 0.296, 0.256, 0.249] is the
flattest: the smallest atom carries 0.200, versus 0.053 at K_c=4.

| model | n clusters | mean H | median H | %H<0.5 | %H<1.0 |
|---|---|---|---|---|---|
| K_c=4  | 1755 | 1.345 | 1.417 |  2.7% | 14.7% |
| K_c=8  | 1072 | 1.290 | 1.335 |  5.2% | 22.8% |
| K_c=16 | 1807 | 1.697 | 1.888 |  4.9% |  9.6% |

By size (K_c=16 medians): size 2 -> 1.94 bits (n=565); 3-4 -> 1.91
(n=558); 5-8 -> 1.80 (n=411); 9-16 -> 1.25 (n=181); 17+ -> 0.32 bits
(n=92, 30% of size-16 locked). Small clusters sit near the prior,
consistent with the estimated rho_chain=0.211 (was fixed at 0.5): a
slower field chain gives weaker per-site theta_P discrimination.

## C. Salt-bridge / flipping-pair hunt (size-2 clusters)

| model | size-2 obs | flipping | flipping + non-static>0.1 | charge complementary |
|---|---|---|---|---|
| K_c=4  |  59 |  10 |  8 |  9 |
| K_c=8  |  28 |   2 |  1 |  1 |
| K_c=16 | 565 | 222 | 84 | 203 |

K_c=16 has 10x-20x as many observable size-2 clusters. 84 (~15%)
show both flipping AND mean non-static > 0.1. Top examples:

- PF11929 cols (6, 113) classes (6, 11): mean P-case
  (0.492, 0.226, 0.227, 0.055), n=3.
- PF13927 cols (94, 96) classes (11, 11): (0.687, 0.147, 0.147, 0.019),
  n=204.
- PF12158 cols (73, 139) and (38, 179): mean P-static ~0.71, n>200.

Most-dynamic single record anywhere: PF01250 cluster 1 (size 22),
cherry 130, P(nn, ny, yn, yy) = (0.000, 0.494, 0.492, 0.014). Genuine
one-side-jump posterior, singular.

## D. Per-family medians

All 100 families in each corpus contribute >= 2 records. Median-of-
medians P(static): 0.9998 (K_c=4), 1.0000 (K_c=8), 0.9867 (K_c=16).
Bottom five families at K_c=16:

| family | K_c=4 | K_c=8 | K_c=16 |
|---|---|---|---|
| PF12158 | 0.817 | 0.889 | 0.789 |
| PF00571 | 0.917 | 0.983 | 0.844 |
| PF13475 | 0.943 | 0.973 | 0.884 |
| PF13927 |     - |     - | 0.893 |
| PF13545 |     - |     - | 0.898 |

PF12158 and PF00571 top the "most dynamic" list across all three
models (both are SH3-like signalling domains with well-documented
cross-column correlations).

## E. Cross-model comparison

| metric | K_c=4 | K_c=8 | K_c=16 |
|---|---|---|---|
| rho_chain (learned/fixed)      |  0.500 |  0.500 |  0.211 |
| H(rho) [bits]                  |  1.764 |  1.920 |  1.986 |
| n size>=2 clusters total       |   1779 |   1084 |   1871 |
| **strict valid records**       | **6368** | **2747** | **71256** |
| **gap-marg valid records**     |**385970**|**239323**|**373595**|
| strict clusters w/ records     |     54 |     20 |    610 |
| gap-marg %(P>=0.99)            |  70.1% |  79.6% |  41.7% |
| gap-marg %(1-P>=0.20)          |  11.0% |   8.2% |  16.0% |
| gap-marg %(1-P>=0.50)          |   5.0% |   4.1% |   4.0% |
| median cluster H(theta) [bits] |  1.417 |  1.335 |  1.888 |
| size-2 flipping + non-static>0.1 |    8 |     1 |     84 |

Aligning with the parameter-side "less field-as-specializer" observation:
strict valid records 6368 -> 71256, strict clusters 54 -> 610, median
P(static) 0.999 -> 0.981, %(P>=0.99) 70-80% -> 42%, size-2 flipping-
with-dynamic-mass 8 -> 84, smallest rho atom 0.05 -> 0.20. Not moving:
%(non-static)>=0.5 stays 4-5% and no record posts P>=0.5 on any single
jump case; median theta_P entropy at size 2 sits at the prior (1.94
vs 1.986 bits) -- theta_P locks in only around size >= 12.

## F. Verdict on static-vs-dynamic

**Static-specializer regime, weakened but not overturned.** The K_c=16
all-fix model is less locked-in than either predecessor and is now
constrained by an 11x-26x larger strict-valid corpus. The most
compelling change is the theta_P entropy shift: median moves from ~1.4
bits (below prior, "cluster picks one theta") to ~1.9 bits (at prior,
"theta belief tracks evidence"). Combined with the balanced rho and
the biophysically clean 16 per-class marginals, the model is now
specialising via K_c rather than by trapping theta_P per cluster -- a
healthier factorisation.

But median P(static) is still 0.98, no record puts P >= 0.5 on any
single jump case, and %(non-static)>=0.5 remains ~4% (unchanged across
all three models). The pattern that overrides everything else --
typical cherry diameters are short enough that beta(theta_P) is near 1
for common thetas -- persists. Next steps:

1. **Push the corpus.** Strict-valid clusters grew ~50 -> ~610;
   top-1000 Pfam should push into the thousands and let dynamic tails
   saturate.
2. **Focus on the large-cluster tail.** Theta_P locks in meaningfully
   only around size >= 12 (178 such clusters at K_c=16, of which 92
   have size >= 17 and median H = 0.32 bits). Downstream consumers
   that use mean-field theta_P should look there.
3. **Warmup schedule for rho_chain.** Fixing rho_chain high early
   (say 1.0) would let residue coupling drive theta_P attribution
   during warmup; annealing back to the learned 0.21 later would
   preserve the fit while giving the model a chance to discover which
   clusters ought to be dynamic before specialization.
