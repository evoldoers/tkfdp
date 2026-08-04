# Dynamic class-chain extension (planning)

**Status**: design sketch, 2026-06-28. Deferred — not yet implemented.

This document plans the next hierarchy level above the field-selector
chain: a per-site dynamic **class** chain (`cls(t)` becomes a trajectory
on the tree instead of a static `cls[s]`). The supplement
(`psb-paper/supplement.tex` §`sec:dynamic-suppl`, "Dynamic site classes:
a nested hierarchy") sketches the construction; this doc translates it
to an impl plan.

## What it adds

The model currently has:

```
class (static per column)  ->  field (per-cluster, evolves on tree)  ->  residue
            cls[s]                       theta_C(t)                       x(t)
```

After this extension:

```
class chain (per-site, evolves on tree)  ->  field chain (per-cluster)  ->  residue
            cls_s(t)                                theta_C(t)                  x(t)
   F81-on-DP rate rho_class_chain          F81-on-DP rate rho_chain
   Gamma prior, mean ~0.05                 Gamma prior, mean ~0.3
```

The supplement's timescale ordering: **residue >> field >> class**
(substitution fastest, coevolutionary regime shifts slower, site
identity slowest). The Gamma prior on `rho_class_chain` should pull
mean to ~0.05 (i.e., ~6x slower than the `rho_chain` prior mean of 0.3,
and ~20x slower than the unit substitution rate).

## Math impact

The cherry doublet emission picks up an extra hierarchy level. For a
single column at class `cls_P` and field `theta_P` at the parent, with
cherry diameter `t`, the leaf residue distribution depends on both
chains' jump statuses on the two half-edges (X-edge, Y-edge). That's
**4 cases for the field chain × 4 cases for the class chain per edge =
16 cases per (theta_P, cls_P) combination per cherry**.

Within each (class_X_jump, class_Y_jump, field_X_jump, field_Y_jump)
case, the residue marginalisation is structurally similar to the
existing 4-case 4-case sum:

- (class no-jump, field no-jump): per-(cls_P, theta_P) GTR cherry joint
  Sigma.
- (class no-jump, field >=1 jump): cls_P fixed, field draws end-state
  from `P(theta_end | >=1 jump, theta_P)`; per-cls_P residue resamples
  from `pi^(cls_P, theta_end)`.
- (class >=1 jump, field no-jump): cls draws end-state from
  `P(cls_end | >=1 jump, cls_P)`; field stays at theta_P; residue
  resamples from `pi^(cls_end, theta_P)`.
- (class >=1 jump, field >=1 jump): both draw end-states
  independently; residue resamples from `pi^(cls_end, theta_end)`.

The two cherry half-edges contribute independent (cls_jump, field_jump)
status pairs, giving the 16-case structure. Within each case, the
per-(cls, theta) intermediates are reused.

**Computational cost**: per cluster of size `m`:

```
O(K_c * L_max * m)         for the per-(cls, theta) products at the parent
+ O(K_c^2 * L_max^2 * m)   for the post-jump J'_class (x) J'_field sums
```

vs. the current dynfield's `O(L_max * m)`. So **K_c^2 L_max factor**
more expensive per cluster. At `K_c=4, L_max=8`: 4^2 * 8 = 128x more
expensive per cluster than the current dynfield. Still tractable on
Pfam corpora (~100K cluster evals per outer iter would take ~1-2
minutes vs 1 second), but a meaningful overhead.

## Implementation outline

### Data structure changes

1. `DynamicFieldState` gains:
   - `pi_class_field: (K_class_max, L_max, A)` — but with dynamic
     classes the class dimension itself is a DP-truncated set of size
     `K_class_max` analogous to `L_max`. So we need a class-DP analogous
     to the field-DP.
   - `rho_class: (K_class_max,)` — class DP stick-breaking weights.
   - `tsb_betas_class: (K_class_max - 1,)`.
   - `alpha_class: float` — class DP concentration.
   - `rho_class_chain: float` — class-chain rate multiplier.
2. `FamilyKState.cls: (L,) int32` becomes a STATIC label of the
   *cluster's class trajectory's initial value* at the parent — but the
   cls trajectory itself is integrated out, not stored. So the
   sufficient stats accumulator carries the work.

### Emission code changes

3. New `coupling/dynfield/class_chain_emission.py`:
   - `class_chain_jump_weights(rho_class, rho_class_chain, t)`: per-cls_P
     state-dependent `beta_class(cls_P)`, and `w_J_class / w_pi_class`
     for the class-end conditional.
   - `cluster_emission_class_field_per_state(...)`: 16-case sum over
     (cls_X_jump, cls_Y_jump, field_X_jump, field_Y_jump). Returns
     `(K_class_max, L_max, 4, 4)` per cluster — the per-(cls_P, theta_P,
     class_case, field_case) likelihood.

### Atom-update changes

4. `attribute_cluster_soft_class_chain(...)`: fractional residue counts
   per (cls, theta, residue). Plus per-cluster cls counts for the
   class-DP TSB update. Plus per-(cluster, cls_P) counts for the
   class-DP concentration update (analogous to the field DP).
5. `update_pi_class_field_dirichlet(...)`: per-(cls, theta) Dirichlet
   posterior mean update (analogous to existing pi_field update).
6. `update_rho_class_tsb(...)`: TSB Beta MAP on class-DP weights.

### Rate-learning changes

7. `update_rho_class_chain_mh(...)`: MH random walk on
   `log(rho_class_chain)` with Gamma prior `Gamma(a_cls, b_cls)`,
   default e.g. `Gamma(1.5, 30)` -> prior mean 0.05.

### Sampler changes

8. Class trajectory inference: the CRP cluster sweep stays unchanged
   (clusters are still about which sites are coupled, not how their
   classes co-evolve). The class-label sweep is REMOVED — `cls` is no
   longer a static label, it's a trajectory, so the per-site
   "current cls" doesn't make sense. Instead the cluster's *cls
   trajectory* is integrated out analytically (via the 16-case sum) and
   the sufficient stats accumulator tracks expected counts.

### CLI changes

9. `experiments/train_dynfield.py`:
   - `--K-class-max` for the class DP truncation.
   - `--alpha-class` for the class DP concentration.
   - `--rho-class-chain` for the class-chain rate initial value.
   - `--rho-class-chain-prior-a / -b` for the Gamma prior.
   - `--n-rho-class-chain-mh-steps`.

### Tests

10. `tests/dynfield/test_class_chain.py`:
    - 16-case closed form vs Gillespie MC.
    - `rho_class_chain -> 0` limit recovers the existing dynfield
      (no class jumps, class stays at parent's cls_P throughout).
    - `rho_class_chain -> infty` limit collapses to a 4-case sum where the
      class is always at stationary `rho_class` (independent of site).

## When to build it

The dynfield core (without the class chain) is already useful — we have
end-to-end training on real Pfam, val LL trajectory, BAliBASE
postprocessing dispatch path. The class chain is a strict generalisation
that adds expressiveness at the cost of `K_c^2 L_max` per-cluster
overhead.

**Reasons to defer**:
- The current `K_c > 1` static path with class Gibbs sweep already gives
  per-site class structure (the difference is just that classes are
  static on the tree, not dynamic).
- A real-Pfam smoke for the static-cls dynfield trainer hasn't been
  run on `K_c > 1` yet; we want to see whether `K_c > 1` even matters
  on top of the dynfield's per-(c, theta) flexibility.
- The 16-case sum + extra DP layer is a substantial code addition
  (~600-800 lines). Should be motivated by a clear empirical need
  rather than a structural completeness argument.

**Reasons to build**:
- The supplement promised it.
- The dynamic class layer is what makes the "nested-iHMM" interpretation
  of the model concrete.
- Some Pfam families have site identity drift over evolutionary time
  scales (gain/loss of functional regions, ND-like fold changes); a
  static cls model can't capture that.

**Recommended trigger for building**: when the current `K_c > 1`
dynfield trainer plateaus and we see evidence (e.g., val LL improves
when we restart training with reshuffled class labels) that static
classes are constraining the model. Until then, defer.

## Open questions

- **Reversibility**: the joint (cls(t), theta(t), x(t)) chain needs to
  be reversible under detailed balance. The supplement claims it is by
  induction on the nesting. Verify explicitly in a small fixture before
  committing to the impl.
- **Identifiability**: with two DP layers nested, the timescale
  ordering (Gamma priors with `rho_class < rho_chain < 1`) is the
  identifiability anchor. How sensitive is the posterior to these
  prior choices? Worth a sensitivity sweep in the first smoke run.
- **Initialisation**: how do we initialise the class-DP at start?
  Probably from the field-marginal pi_class as a single class with
  Dirichlet perturbation per K_class_max slot — analogous to how
  pi_field is currently initialised from pi_class.
