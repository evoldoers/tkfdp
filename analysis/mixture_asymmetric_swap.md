# Asymmetric (directional) swap-pair coupling components vs symmetric ones

**Question.** Every coupling component in `fit_coupling_mixture_freeS.py` carries a
SYMMETRIC joint stationary pi_c over the 210 UNORDERED amino-acid pairs
(pi_c(a,b)=pi_c(b,a), 209 free params) -- an exchangeable component. Does the
site-site coupling in structural contacts have DIRECTIONAL structure that a symmetric
component throws away? Concretely: at a matched parameter/component budget, does the
data prefer FEWER asymmetric (directional) archetypes or MORE symmetric ones? If
asymmetric wins, exchangeability is leaving directional signal on the table.

Code: `experiments/fit_coupling_mixture_asym.py` (new; the symmetric baseline is
`fit_coupling_mixture_freeS.py`, run UNCHANGED as a subprocess). Corpus
`data/per_contact_trrosetta/counts.npz`. Held-out = 20% of families (unseen). seed 0,
em-iters 60, inner 2, shared free S on BOTH sides. Fits in `results/mixture_asym/`.

## The corpus keeps a consistent ORDERED pair-state (checked first -- else moot)

Asymmetry is only meaningful relative to a consistent site ordering. The corpus is
built (`build_per_contact_corpus.py`) by greedy contact matching over
`np.triu_indices(L, k=1)`, so every contact `(colA, colB)` has **colA < colB**, and the
pair-state is `pf = (residue@colA)*20 + (residue@colB)` (likewise `pt`). Verified on the
500,594-cluster corpus:

- fraction with colA < colB = **1.000** (0 with colA == colB) -- a consistent
  **sequence ordering: site 1 = the lower column index, site 2 = the higher**.
- The corpus is **NOT symmetrized / folded**: transitions are stored directionally
  (pf, pt over the 400 ordered states), per-cluster identity preserved.

The pooled start/end occupancy over the 400 ordered states is *nearly* symmetric
(sum|P - P^T| = 0.041, off-diagonal corr(P_ij, P_ji) = 0.996) -- expected, because
sequence order i<j is largely arbitrary relative to the chemistry of a contact, so
across the whole corpus each orientation appears about equally. That is a statement
about the POOLED marginal; individual clusters and fitted components can still be
strongly directional. **All asymmetry reported below is w.r.t. this sequence ordering.**

## Method: exchangeable mixture of directional swap-pairs

**Generator (unchanged).** Each component is the same reversible Metropolis-sqrt pair
chain `Q_c = met_sqrt(S, pi_c)` on the 400 ordered pair-states, built from ONE shared
free single-site exchangeability S (warm-init LG08) and a per-component stationary pi_c.
Metropolis-sqrt with a symmetric S, `Q_xy = S_(single-site) * sqrt(pi_y/pi_x)` on
single-coordinate moves, is reversible w.r.t. ANY target pi -- symmetric OR asymmetric.
So an asymmetric pi_c (a full distribution over the 400 ORDERED states, 399 free params)
is a perfectly valid coupling component; nothing about the construction changes except
that pi_c is no longer forced to satisfy pi_c(a,b)=pi_c(b,a).

**Keeping the mixture exchangeable: swap-pairs.** A lone asymmetric component would make
the mixture depend on which contact site we call "1". To preserve exchangeability
(invariance under the site-swap relabelling sigma:(a,b)->(b,a)), each asymmetric
component enters as a SWAP-PAIR: the pair {pi_c, pi_c^swap} with pi_c^swap(a,b)=pi_c(b,a),
**tied to one free pi_c**, **equal weight W_c/2 each**, appearing as **two** mixture
components. The component set is closed under sigma with equal weights, so

    L_mix(g) = sum_c (W_c/2) [ exp LL(g | pi_c) + exp LL(g | pi_c^swap) ]

is EXACTLY sigma-invariant (verified numerically, below). Because S is symmetric and the
single-site moves map coordinate-1 <-> coordinate-2 under sigma,

    met_sqrt(S, pi_c^swap)[x, y] = met_sqrt(S, pi_c)[sigma x, sigma y],

so the swap component needs NO separate generator or eigendecomposition -- it is pi_c's
own transition grid read at site-swapped state indices. Scoring the swap component on a
cluster equals scoring pi_c's chain on the site-swapped counts.

**Exact parameter / component accounting (matched budget).**

| unit | components | free pi params |
|---|---:|---:|
| 1 symmetric component | 1 | 210 - 1 = 209 |
| 1 asymmetric swap-pair | 2 | 400 - 1 = 399 |
| 2 symmetric components | 2 | 2*209 = 418 |

So **P asymmetric swap-pairs (2P components, 399P params) is matched to 2P free
symmetric components (2P components, 418P params)** in component count and, to within
19P params, in parameter count -- with the asymmetric side being the **leaner** one.
Both sides additionally share the same free S (190 params), which cancels from the
comparison.

**Reversible asymmetric M-step.** The symmetric pi-step (`FP.mstep_pi_metropolis`) does
damped gradient ascent on log pi of the complete-data log-likelihood, then symmetrises
`b = 0.5(b + b^T)`. Dropping ONLY that symmetrisation gives the unrestricted
reversible-chain / Holmes-Rubin M-step over all 400 states, which naturally yields an
asymmetric pi_c (the symmetric case is exactly the restriction pi(a,b)=pi(b,a)). Per
swap-pair the responsibility-weighted evidence is folded into pi_c's frame,

    Ncounts_c = sum_g r_{g,(c,+)} n_g  +  sum_g r_{g,(c,-)} sigma(n_g)

(the "+" component contributes the cluster counts, the "-" component contributes the
site-swapped counts, equivalently its responsibility accumulated at swapped state
indices), followed by the HR estep under Q_c and the asymmetric pi-step. Shared S is
pooled over all pairs exactly as in the symmetric fitter. EM was **monotone** in every
run.

## Degenerate reduction check (validated)

If pi_c is symmetric, pi_c^swap = pi_c, the two tied components coincide, and the
swap-pair collapses to a single symmetric component of weight W_c -- the whole mixture
must reduce EXACTLY to the symmetric mixture. Verified (`degenerate_check`, max abs
discrepancies over held-out clusters):

- model-level reduction, max|LL_swap-pair(g) - LL_symmetric(g)| = **3.98e-13**
- fitted mixture swap-invariance, max|L_mix(g) - L_mix(sigma g)| = **5.68e-14**
- asymmetric M-step preserves the symmetric manifold, max|pi - pi^swap| = **2.08e-17**

(sigma-symmetric N,T + symmetric pi in -> symmetric pi out). The asymmetric fitter is a
strict superset of the symmetric one.

## Matched-budget held-out result

Per-count held-out (unseen-family) log-likelihood; delta = asymmetric minus symmetric
(positive = asymmetric wins). Both sides shared-free-S, identical corpus / family split
/ EM schedule. The symmetric column reproduces the published free-S numbers
(-2.653 / -2.627 / -2.602) exactly, confirming the identical split.

| 2P | asym swap-pairs (val) | symmetric (val) | delta (asym - sym) | asym train | sym train | asym pi-params | sym pi-params |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | -2.6568 | **-2.6527** | -0.0040 | -2.6819 | -2.6779 | 399 | 418 |
| 4 | -2.6338 | **-2.6273** | -0.0065 | -2.6588 | -2.6523 | 798 | 836 |
| 8 | -2.6067 | **-2.6019** | -0.0048 | -2.6317 | -2.6267 | 1596 | 1672 |

**Symmetric wins at every matched size**, by a small but consistent 0.004-0.007
nats/count on held-out -- and it wins while using *more* parameters is NOT the reason
(the asymmetric side is the leaner side, 399P < 418P). At a matched budget the data
prefers MORE symmetric coupling archetypes over FEWER directional ones.

## Is the asymmetry real, or negligible?

The fitted asymmetric components are **strongly directional on the training data**, yet
that directionality buys very little that generalises:

| 2P | weighted sym-KL J(pi_c || pi_c^swap) | weighted TV(pi_c, pi_c^swap) | held-out cost of forcing symmetry |
|---:|---:|---:|---:|
| 2 | 0.298 | 0.313 | +0.0155 |
| 4 | 0.513 | 0.383 | +0.0209 |
| 8 | 0.657 | 0.426 | +0.0229 |

- **sym-KL** = 0.5[KL(pi_c||pi_c^swap) + KL(pi_c^swap||pi_c)], **TV** = 0.5 sum|pi_c -
  pi_c^swap|. Both are far from zero (TV ~0.3-0.5): the components genuinely fit
  directional stationaries, not near-symmetric ones.
- **Cost of forcing symmetry** = held-out per-count LL of the fitted model minus the
  same model with each pi_c replaced by (pi_c + pi_c^swap)/2 (same weights/S). It is
  small and POSITIVE (~0.015-0.023): the directional DOF does help held-out a little,
  over *symmetrising the same components*.

The two facts reconcile the verdict. Take 2P=8 as the example (all four points share the
same S and split):

- symmetrising the asymmetric fit -> ~4 symmetric components -> val ~ -2.630
  (matches free-S K=4 = -2.627);
- making those 4 components DIRECTIONAL (the swap-pairs) -> val -2.6067  (**+0.023** for
  the directional DOF);
- instead spending the same budget on 8 symmetric components (free-S K=8) -> val
  **-2.6019** (**+0.025**, more than the directional DOF bought).

So directional structure is real and weakly generalisable (~0.02 nats/count over
symmetrising), but per unit budget, **doubling the number of symmetric components buys
more than making the components directional** -- hence symmetric wins the matched-budget
comparison by ~0.005.

## What the directional structure looks like (w.r.t. sequence order)

The largest per-pair directional imbalances log[pi_c(a,b)/pi_c(b,a)] (site 1 = lower
column index) are dominated by **glycine at the lower-index contact position** paired
with a bulkier/charged residue at the higher-index position -- e.g. at 2P=8:
G->V, G->T, G->I, G->R, G->K (pair 1); G->P, G->H, G->E (pair 3); plus X->Y motifs
(residue at the lower position, tyrosine at the higher). This is a genuine
sequence-order effect, but modest: because i<j is largely arbitrary relative to a
contact's chemistry, each directional archetype's swap-pair must still carry BOTH
orientations at equal weight, so half of each directional component's capacity is spent
re-representing the mirror -- which is exactly why splitting the budget into more
symmetric components is more efficient.

## Verdict

**No -- at matched budget the data does NOT prefer directional (asymmetric) coupling
archetypes over symmetric ones.** Symmetric components win the held-out comparison at
every matched size (2P = 2, 4, 8) by a small, consistent 0.004-0.007 nats/count, despite
the asymmetric side being the leaner one. There IS a real, weakly generalisable
directional signal in structural contacts (fitted asymmetry TV ~0.3-0.5; forcing
symmetry on a fitted component costs ~0.02 nats/count), but it is worth less than the
equivalent count of extra symmetric components. **Exchangeability is not leaving
meaningful directional signal on the table** in any budget-efficient sense: the
site-site coupling is better described by a few more SYMMETRIC archetypes than by fewer
DIRECTIONAL ones -- consistent with the fact that the sequence ordering (i<j) is nearly
orthogonal to the chemistry that drives the coupling.

*Caveat:* all runs (both sides) hit the 60-iteration cap with tiny tail deltas
(~1e-4/count/iter, as for the published free-S K=16 run), so each val is within ~0.002
of its optimum. The per-size deltas (0.004-0.007) are small, but the sign is consistent
across all three matched sizes and is mechanistically explained by the
force-symmetry decomposition above, so the direction of the verdict is robust.
