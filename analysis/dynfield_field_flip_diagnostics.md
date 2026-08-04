# Does dynfield use theta dynamically or as a static specializer?

Diagnostics on the two trained dynfield checkpoints
`results/dynfield_top100_K4_L4_2026-06-30/_best_chkpt/state.npz`
(K_c=4, L_max=4, max_cluster_size=8, val-LL -139402.79) and
`results/dynfield_top100_K8_L4_M16_2026-06-30/_best_chkpt/state.npz`
(K_c=8, L_max=4, max_cluster_size=16, val-LL -95232.36). Both used
rho_chain=0.5 (fixed) on top-100 Pfam. Model spec
`math-paper/appendix-tkfdp.tex`; kernel
`src/tkfdp/coupling/dynfield/emission.py::cluster_emission_per_theta`.

## Headline finding

**Both trained models use theta as an effectively static, per-cluster
specializer rather than as a dynamic coupling mediator.** Across 6,368
(cluster, cherry) records in K_c=4 and 2,747 in K_c=8, the median
posterior on "no jump on either half-edge" is >= 0.999, the IQR upper
bound is exactly 1.0, and not a single posterior assigns >= 0.5 mass
to any of the three jump cases (ny, yn, yy). 80-82% of records put
>= 0.95 mass on the static case; 67-72% put >= 0.99. The user's
hypothesis is confirmed: each cluster effectively locks into one
(c, theta) configuration, gaining specialization from theta but not
exercising the dynamic-coupling structure the model formally permits.

Moving from K_c=4 to K_c=8 does **not** shift the model toward
dynamic-theta behavior; the marginal-case distributions are
statistically indistinguishable, and K_c=8 actually has a slightly
larger fraction of "locked-in" (low-entropy) clusters.

## Posterior 4-case marginal P(case | data)

Means across all valid (cluster, cherry) records (size >= 2 clusters,
all cluster columns observed in that cherry):

| model | n records | mean P(nn=static) | mean P(ny) | mean P(yn) | mean P(yy) |
|---|---|---|---|---|---|
| K_c=4 | 6368 | 0.9375 | 0.0271 | 0.0271 | 0.0083 |
| K_c=8 | 2747 | 0.9388 | 0.0256 | 0.0256 | 0.0100 |

P(static) summary:

| model | median | 25% | 75% | %>=0.95 | %>=0.99 |
|---|---|---|---|---|---|
| K_c=4 | 0.9987 | 0.974 | 1.0000 | 80.3% | 67.0% |
| K_c=8 | 0.9996 | 0.982 | 1.0000 | 82.4% | 72.0% |

In both models, max(P(ny), P(yn), P(yy)) never exceeds 0.5 on any
record, and only 4.7-4.9% of records have any dynamic-case
probability >= 0.2. Real jump events in the cap-2 chain are
effectively never inferred.

## Per-cluster theta_P posterior entropy

Per cluster we averaged P(theta_P | data) across its valid cherries
and computed Shannon entropy H in bits. The data-free prior gives
H(rho_K4) = 1.764 bits, H(rho_K8) = 1.920 bits (uniform max = 2.000
at L_max = 4).

| model | n clusters | mean H | median H | % H < 0.5 ("locked") | % H < 1.0 |
|---|---|---|---|---|---|
| K_c=4 | 54 | 0.710 | 0.706 | 42.6% | 63.0% |
| K_c=8 | 20 | 0.692 | 0.449 | 55.0% | 70.0% |

By cluster size (K_c=4): median H is 1.41 bits at size 2 (near prior;
data not enough to pin theta), 0.03 bits at size 5 (100% locked),
0.32 bits at size 8 (53.6% locked). The trend holds in K_c=8 up to
size 16 (median H = 0.24 bits, 83% locked). Entropy collapses sharply
as more sites per cluster vote on the same theta -- exactly the
behavior of static specialization.

## Salt-bridge / flip hunt on size-2 clusters

For each size-2 cluster with valid cherries we collected the (X_0,
X_1) and (Y_0, Y_1) pairs and asked whether the set contains both
(a, b) and (b, a) for some a != b ("flipping"). We also flagged
"charge complementary" clusters (one column sees K/R/H, the other
D/E).

| model | size-2 obs | flipping | flipping AND mean P(non-static)>0.1 | charge complementary |
|---|---|---|---|---|
| K_c=4 | 14 | 10 | 8 | 13 |
| K_c=8 | 2 | 2 | 1 | 2 |

Flipping is the norm for size-2 cluster compositions, and most have
charge-complementary endpoints. Yet on average the posterior still
concentrates >85% mass on the static case. The clearest
counterexample is K_c=8 / PF00036 cluster {col 11, col 12} (classes
(0, 2)). 170 cherries are valid; the most-dynamic cherries give
P(nn, ny, yn, yy) = (0.008, 0.434, 0.407, 0.150) at tau=1.93 with
residues X = (D, G), Y = (N, C). Mean P(non-static) across the 170 is
~0.20. Genuine dynamic-theta behavior -- but **one cluster out of
1084** in K_c=8.

## Per-family medians and gap-driven clustering

Median P(static) across each family's (cluster, cherry) records.
K_c=4: only 15/100 families contribute any valid cluster-cherry pair;
per-family medians range 0.974 (most dynamic, PF02912) to 0.9999
(PF21982), median-of-medians 0.999, IQR [0.998, 1.000]. K_c=8 covers
7/100 families in the same band (median-of-medians 0.9998, range
0.997 to 1.000). No family breaks the pattern.

A conditioning side observation: only **54/1779 size>=2 clusters in
K_c=4 (3.0%)** and **20/1084 in K_c=8 (1.8%)** have any cherry in
which all cluster columns are simultaneously observed. The remainder
are clusters of mostly-gap columns whose members have observation but
never jointly. The CRP cluster sweep is optimising primarily over
gap-pattern coherence, not residue-level coupling. Cluster sizes are
right-skewed up to max_cluster_size: in K_c=8, 12.5% of clusters sit
at the cap of 16, and the modal size is 15-16.

## Cross-K_c comparison and verdict

| metric | K_c=4 | K_c=8 |
|---|---|---|
| mean P(static) | 0.938 | 0.939 |
| median P(static) | 0.999 | 1.000 |
| % records w/ P(static) >= 0.95 | 80.3% | 82.4% |
| mean cluster theta-entropy | 0.71 bits | 0.69 bits |
| % clusters locked (H < 0.5) | 42.6% | 55.0% |
| valid size-2 clusters | 14 | 2 |
| flipping size-2 clusters | 10/14 | 2/2 |
| H(rho) prior | 1.76 bits | 1.92 bits |
| size>=2 clusters w/ any valid cherry | 3.0% | 1.8% |

Going from 4 to 8 classes does not move the model toward dynamic use
of theta. If anything, the K_c=8 distributions are slightly more
static / more locked-in: the posterior is sharper on the no-jump
case, and a larger fraction of clusters concentrate posterior mass on
a single theta. The model is using both K_c and theta as
specialization axes, not as coupling-mediation axes. K_c=8 mostly
proliferates more clusters at the max-size cap, not more
dynamic-theta clusters.

## Verdict

**Static use, confirmed.** Across both trained models the
field-selector theta serves as a per-cluster specialization knob:
each cluster picks one (rarely two) theta values with high posterior
confidence, then the cap-2 chain almost never fires. Median P(static)
>= 0.999; no record anywhere puts >= 0.5 mass on a jump case. This is
the static-specializer regime, not the dynamic-coupling-mediator
regime the model was designed for, and increasing K_c does not unlock
dynamic theta-use.

Concrete recommendations:

1. **Estimate rho_chain rather than fix it at 0.5.** The current
setting over-encourages no-jump everywhere: cherry diameters are
typically t/2 < 1, so beta = exp(-rho_chain*(1 - rho_theta)*t/2) is
near 1 for most (theta_P, cherry) pairs, and the trained rho
concentrates further on theta values with rho_theta close to 1.
Adding a Gamma prior on rho_chain and estimating it (Section 5.3 of
`math-paper/appendix-tkfdp.tex`) lets the data choose whether the
chain should fire.

2. **Reduce max_cluster_size rather than raise it.** The right-skewed
distribution to the cap signals that large clusters are being formed
by aggregating mostly-gap columns whose joint observation is rare. A
cap of 4 would force the CRP onto more constrained, residue-
overlapping configurations -- where the cap-2 chain would have a
chance to fire. The K_c=8 model currently pays for L_max *
max_cluster_size = 64 emission tensor entries per cluster but uses
<2% of them on residue-level coupling.
