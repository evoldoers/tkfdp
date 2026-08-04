# Permutation-field mood-light model (paper 2b Sec. 4, simpler variant)

A deliberately simple version of the weakly-identified "mood light" latent-field
model, backing away from the enum400 / DM / archetype-assignment / renewal
machinery in `coupling.dynfield`. Fit is teed up for C = 2, 3, 4.

## Model

- **C archetypes.** Archetype c is an LG08-style GTR process with a shared
  exchangeability S = S_LG08 and its own equilibrium pi^c:
  `Q^c[x,y] = S[x,y] * pi^c[y]` (x != y). Free: C x 19.

- **Site classes.** Each column draws a class c_s in {1..C} at birth from a
  learned categorical `rho = (rho_1..rho_C)`, `sum_c rho_c = 1` (the site-class
  frequencies; C-1 free). rho is fitted by EM like any mixture weight.

- **Field** theta over S_C (permutations of the C archetypes), a C!-state CTMC.
  A site of class c uses, under field state theta, archetype `theta(c)`; its
  generator is `Q^{theta(c)}`. This is Markov modulation, NOT renewal: at a field
  jump the site keeps its current residue and switches which GTR it follows.

- **Field jumps are transpositions** (swap two archetypes a,b):
  `theta' = (a b) o theta`. Every jump changes the transposition distance from
  the identity `d(theta) = C - #cycles(theta)` (the Cayley distance on S_C) by
  exactly +-1, so each row of Q_field has C(C-1)/2 nonzero off-diagonals -- a
  sparsity constraint on the field exchangeabilities.

- **Reversible GTR generator on the transposition edges:**
  `Q_field[i,j] = s_{min(d_i,d_j)} * w_{a,b} * pi_field[j]`, where
  `theta_j = (a b) o theta_i`, and
  - stationary `pi_field(theta) = p[d(theta)] / Z` depends on theta only through
    the transposition distance d (params p_0..p_{C-1});
  - `s_d` is a distance-level factor on edges between d and d+1 (s_0..s_{C-2});
  - `w_{a,b}` is a symmetric/unordered ARCHETYPE-pair factor (one per swapped
    pair {a,b}, keyed by the two archetypes changing places, not by any class
    pair), so the jump rate may depend on which archetypes are swapped as well
    as on d(source), d(dest).

  The class labelling enters the field ONLY through the Cayley distance d, which
  is centred on the identity theta = id (class k uses archetype k); w_{a,b} and
  the archetype equilibria pi^c are otherwise purely archetype-based. (Verified:
  a given archetype swap {a,b} is charged the same w across all C! source
  states, while the classes it moves vary with theta.)

  Zeroing p_d for d >= 2 leaves mass only on the identity and single swaps, i.e.
  a star of transpositions around identity -- close to the enum400 field.

Free field params: p_0..p_{C-1} (C-1 after normalisation) for the stationary;
s_0..s_{C-2} and w over the C(C-1)/2 pairs share one overall scale (fixed by
mean-rate-1), so the exchangeability contributes (C-1) + C(C-1)/2 - 1 free rates.

| C | states | swaps/row | p (stationary, free) | exch. rates | archetypes |
|---|--------|-----------|----------------------|-------------|------------|
| 2 |   2    |    1      |         1            |     1       |   2 x 19   |
| 3 |   6    |    3      |         2            |     4       |   3 x 19   |
| 4 |  24    |    6      |         3            |     8       |   4 x 19   |

Generator + all checks: `src/tkfdp/permfield/field_ctmc.py` (`build_field`),
validated (rows sum 0; pi_field Q = 0; detailed balance; C(C-1)/2 swaps/row;
rate a function of ({a,b},d_i,d_j) only; |d_i-d_j| = 1 every edge; mean rate 1).

## Fitting (needs a phylogeny + MSA per family)

Data: the thinned CLV corpus `data/pfam_processed_clv_top1000_thin128` (per-family
tree + leaf MSA). Latent per family: each column's class c_s in {1..C}, and the
field trajectory theta(t) on the tree. Parameters: the site-class frequencies
{rho_c}, the archetype equilibria {pi^c}, the field stationary {p_d}, and the
field exchangeabilities {s_d}, {w_ab}.

Given the field trajectory the columns are conditionally independent, so the
product-of-trees ELBO of paper 2b Sec. 5-6 applies with the C!-state field in
place of the 2-state (theta,Delta) field:

1. **E-step (columns | field).** For each column, a Felsenstein pass whose
   per-branch operator is the field-posterior-weighted mixture of the per-field
   substitution matrices `P^{theta(c_s)}(tau)`. Bridge statistics (HR) give the
   per-archetype expected dwell/transition counts. (The 400-state HR probe timed
   at 56 ms, so the per-archetype 20-state HR here is trivial.)
2. **E-step (field | columns).** Belief propagation over the tree-Markov field
   RF on S_C returns the field marginals and expected jump counts by
   distance-edge (d, d+1).
3. **M-step.** Update pi^c by the GTR substitution M-step (secret-destination
   Dirichlet, exact for F81 / biased for LG -- see paper 2b Sec. 2); update p_d
   and the exchangeabilities s_d, w_ab from the field dwell/jump expectations
   under the reversible parameterisation; update the class frequencies
   rho_c = mean_s gamma_s(c) (the mean posterior responsibility, below).
4. **Site classes.** Sum over the C classes per column (cheap for small C): the
   per-column class responsibilities gamma_s(c) = P(c_s = c | data) fall out of
   that sum and both re-weight the per-archetype bridge statistics in step 3 and
   give the rho M-step above. (For large C, Gibbs-sample the classes instead.)

## Tee-up: C = 2, 3, 4 runs

- Reuse: `permfield.build_field` (field generator), the LG08 GTR archetypes, and
  the product-of-trees ELBO / HR machinery (adapted off enum400).
- Build a clean `permfield` trainer that takes (C, corpus) and runs EM to
  convergence, reporting the field stationary p_d, the archetype profiles pi^c,
  and the ELBO. Start on a small family subset, then scale.
- C = 2 is the sanity case (2-state field; a single swap). C = 3, 4 exercise the
  transposition-distance structure (p_2, p_3 and the distance-level rates s_d).

## Status: trainer built + validated (composite likelihood)

`experiments/permfield_fit.py` fits the model by maximising the COMPOSITE
(per-column) likelihood: each column integrates the field on the full tree via
the exact joint (theta, x) generator G_c (a C!*20-state CTMC; expm(G_c tau)
integrates within-branch field jumps exactly), and the family likelihood is
`sum_cols log sum_c rho_c P(column | c)`. All parameters {pi^c, p_d, s_d, w_ab,
rho} are fit by gradient ascent (optax Adam) through the differentiable
likelihood -- no hand-derived M-steps to get wrong. This shares parameters
across columns but NOT the field trajectory (the shared-field product-of-trees
ELBO of steps 1-4 is the documented refinement, still open).

Validation (`--synthetic`, self-consistent: simulate from the per-column joint
model, then fit):
  * log-likelihood is monotone-increasing under Adam;
  * the fitted LL EXCEEDS the LL at the true generating parameters (C=2:
    -13459 fitted vs -13484 at truth), so the optimiser is correct;
  * point recovery of (rho, p) is nonetheless poor for small C -- a genuine
    WEAK IDENTIFIABILITY, not a bug: for C=2 the field swap exchanges which
    archetype each class uses, so a class-c column under an identity-dominant
    field is nearly indistinguishable from a class-(swapped) column under a
    swap-dominant field, trading rho against p. This is exactly the
    "mood-light / weakly-identified" regime paper 2b Sec. 4 describes.

Real-data fits run for C = 2, 3, 4 on a CLV family (leaves subsampled to keep
the C!*20 joint state space cheap: 40 / 120 / 480 states). C=2 on PF00013
converges to an identity-dominant field (p ~ [0.995, 0.005]) with one dominant
class -- the expected near-degenerate mood-light fit.

### Composite landscape (reseeds)

Random-reseed fits of the composite (fixed family + leaf subsample, varying only
the Adam init): C=2 is essentially unimodal (LL range 0.38 over 5 seeds; archetype
profiles agree to L1 0.03, only the field stationary p flips by the label-swap
degeneracy). C=3 is mildly multimodal (LL range 3.8; the best seed finds a
2-active-class solution, others a 1-dominant-class one), so best-of-N reseeding
helps at C>=3.

## Status: shared-field product-of-trees ELBO with exact HR (the C<=4 fit)

`experiments/permfield_elbo.py` implements steps 1-4 above: a structured
mean-field variational fit that ties ONE field trajectory across all of a
family's columns (what the composite drops). Factors
`q = q_Theta(field, C!-state tree)  x  prod_j q_j(residue, 20-state tree)`,
class latent summed with responsibilities gamma_j(c). Coordinate ascent:
  1. columns | field: per class, per-branch field-averaged generator
     `Qbar_{c,u} = sum_a beta_{c,u}(a) Q^a`, inside-outside -> gamma-weighted edge
     marginals; gamma_j(c) prop rho_c L_j(c);
  2. field | columns: node potentials = column expected-log-lik on each node's
     out-branches; EXACT belief propagation over the C!-state field tree;
  3. M-steps by EXACT Holmes-Rubin bridge statistics
     (`src/tkfdp/permfield/hr.py`): archetype pi^a from HR (N^a, T^a) flux
     gradient; field exchangeabilities s,w (rates) from HR field bridge stats;
     rho_c = mean gamma.

The field STATIONARY is the EWENS distribution on S_C -- pi_field(theta) prop
alpha^{#cycles(theta)}, which (since d = C - #cycles) is exactly the geometric-
in-Cayley-distance target pi_field(theta) prop alpha^{-d(theta)}, one concentration
alpha replacing the free p_0..p_{C-1}. This is the project's Ewens/DP (alpha_z)
backbone. It also supplies the symmetry breaking for free: the Ewens prior is
IDENTITY-CENTRED (the identity has the most cycles), so a-priori class k maps to
archetype k -- a uniform field marginal instead makes the field-averaged
generator class-independent and freezes gamma at 1/C. The field marginal is
initialised to this Ewens prior (not a random asymmetry). alpha is a PRIOR
concentration held FIXED by default: learning it runs away (mean-field field-
evidence under-detects field activity, so alpha and the field posterior reinforce
toward identity-collapse) -- the weakly-identified field-activity dimension of the
mood light; `--learn-alpha` fits it anyway.

HR bridge statistics validated against a brute-force time-discretised integral
(dwell + jump expectations match to 5 decimals; sum_x dwell = t exactly;
stationary invariants T=pi*t, N=pi[x]Q_xy*t to machine precision).

ELBO fit validated (shared-field synthetic: one field trajectory per family):
  * objective monotone under coordinate ascent;
  * parameter recovery is GOOD and clearly beats the composite -- with the Ewens
    prior C=2 recovers rho directly (0.139/0.861 vs true 0.112/0.888; the composite
    washed it out to 0.43/0.57) and C=3 recovers rho to a permutation almost
    exactly ({0.144,0.261,0.595} vs true {0.139,0.606,0.254}), archetype-profile
    L1 ~0.14;
  * FAST: the 20-state per-column Felsenstein + C!-state field BP fit C=2/3/4 in
    1-2 s on 20-24 leaves, where the composite's C!*20 joint expm was ~10 s/iter
    at C=4. The C!-state field BP is exact for C<=4; C>4 would approximate the
    field factor (same ELBO, looser field factor) -- the doc-step distinction.

This is the shared-field fit the mood-light model needs; the composite
(`permfield_fit.py`) remains a fast, per-column baseline.

### Product of (m+1) trees per cluster -- general m, any tree, any C

`permfield_elbo.fit` takes a PARTITION of the columns into clusters. A cluster of
size m carries **(m+1) tree-structured variational factors**: one field tree
q_Theta (C!-state, shared by the cluster's m members) plus m residue trees
(A-state, one per member). The members are factored (coupled ONLY through their
shared field), so the cost is **linear in m -- never A^m** -- and the construction
is agnostic to tree shape (Felsenstein/BP) and to C (exact field BP for C<=4).
Global parameters {pi^a, alpha, s, w, rho} are pooled across all clusters
(per-cluster E-step accumulate + global M-step solve: `accum_arch`/`solve_arch`,
`accum_field`/`solve_field`); each cluster keeps its own field posterior.

`partition=None` is one cluster of all columns (the family-global field, maximum
sharing). Singletons (m=1) reduce to independent per-column fields (= the
composite); pairs (m=2) are the coupled-contact case (the two members share one
field, the compensatory salt-bridge flip carried by a shared field swap, no
direct A^2 Potts term). Validated: monotone objective for m=1,2 and C=2,3;
cost scales with (#clusters x (field BP + m residue passes)), no A^m; recovery
improves with sharing (one big cluster L1 ~0.15; many small clusters weaker, as
each field sees only m columns). The clustered simulator (`simulate(...,
clusters=[m,...])`) draws one shared field trajectory per cluster.
