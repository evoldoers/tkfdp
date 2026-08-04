# Variational ELBO for dynfield on general trees

**Status**: design sketch, 2026-06-29. Not yet implemented. Companion
to `docs/dynfield_math.md` (cap-2 cherry closed form, exact) and
`docs/dynfield_runbook.md` (training surface).

## Motivation

Cap-2 cluster training (cherry geometry, one internal parent node, two
leaves) is exactly computable in O(C · F · A²) per cluster -- this is
what `cluster_emission_per_theta` implements via the 4-case
decomposition. For *general* phylogenetic trees (depth > 1), exact
Felsenstein on the dynfield model has cost exponential in tree depth,
because the per-node CLV's "rank-1 + scalar" representation
(Step 2 of the cap-2 derivation) is preserved by per-edge propagation
but *not* by the multiplicative combine at internal nodes: combining
two `(rank-1 + scalar)` CLVs produces three rank-1 terms plus one
scalar, and the rank squares at every internal node of a balanced
binary tree.

This doc plans a structured variational approximation that:

1. Restricts F^v at every node to a low-rank form matching the
   structure that per-edge propagation produces.
2. Is **exact** at cap-2 cherries (training regime) and at the
   rho_chain → 0 and rho_chain → ∞ limits.
3. Gives a tractable O(N · C · F · A²) per coordinate-ascent sweep.
4. Mirrors the Potts variant's Cohn/Girsanov ELBO structurally
   (exact at the training unit, principled extension to deeper trees).

## Generative model

The variational ELBO is model-agnostic with respect to the
**Interp 1 vs Interp 2 question** (whether `theta -> theta` self-events
trigger residue resampling, supplement §sec:reequil-suppl "Carry vs.
resample"). The variational family and coordinate-ascent structure are
identical under both interpretations; only the per-edge kernel
evaluated inside the ELBO's cross-entropy terms differs. Throughout
this doc we use the joint stationary

pi(theta, x) = pi_theta · prod_n pi^(c_n, theta)(x_n)

which is the same in both interpretations.

**Interp 2 (the implemented and recommended model; textbook
F81-on-DP CTMC).** Joint generator:

R_{phi, phi'} = sum_n Q^(c_n, theta)_{x_n, x'_n} prod_{m != n} delta(x_m = x'_m) delta(theta = theta')
             + rho * delta(theta != theta') * pi_{theta'} * prod_n pi^(c_n, theta')(x'_n)
             - rho * (1 - pi_theta) * delta(phi = phi')

Per-edge kernel of length tau:

K_Interp2(phi, phi'; tau) = beta(theta_v, tau) · delta(theta_u = theta_v) · prod_n P_subst^(c_n, theta_v)(x_n, x'_n; tau)
                          + W(theta_v, theta_u; tau) · prod_n pi^(c_n, theta_u)(x'_n)

with state-dependent no-jump probability beta(theta_v, tau) =
exp(-rho · (1 - pi_{theta_v}) · tau) and W(theta_v, theta_u; tau) =
P_field(theta_v -> theta_u; tau) - delta(theta_u = theta_v) ·
beta(theta_v, tau).

**Interp 1 (simplified; self-events count).** Same R but with the
`delta(theta != theta')` removed and the diagonal compensation
uniform (-rho · delta(phi = phi')). Per-edge kernel collapses to a
clean **regeneration form**:

K_Interp1(phi, phi'; tau) = exp(-rho · tau) · K_carry(phi, phi'; tau)
                          + (1 - exp(-rho · tau)) · pi_stationary(phi')

with K_carry the within-theta Q-evolution. The cap-2 cherry doublet
takes the simpler form with uniform beta and J' = J (no per-theta_P
retention term).

**Implementation note**: the dynfield codebase (as of 2026-06-28)
implements Interp 2 throughout (`emission.cherry_doublet_4case`,
`emission.jump_weights`, `dp_field.f81_dp_transition`, etc.). The
variational machinery would be built on top of the Interp 2 kernel.
The Interp 1 form is shown above only because its regeneration
structure makes the variational analysis algebraically cleaner; the
SAME variational family works for Interp 2 with the more involved
kernel.

## Variational ansatz

At every tree node v, restrict the joint (residue, field) marginal to:

q_v(x_v, theta_v) = q_v(theta_v) · [(1 - lambda_v(theta_v)) · prod_n p_n_v(x_v_n | theta_v)
                                   + lambda_v(theta_v) · prod_n pi^(c_n, theta_v)(x_v_n)]

The variational parameters per node v:

- q_v(theta_v): F real numbers (a distribution over theta at v).
- p_n_v(x_v_n | theta_v): F · A per site (per-site, per-theta residue
  marginal under the "tracking" component).
- lambda_v(theta_v): F real numbers in [0, 1] (per-(node, theta)
  mixture weight: probability of being in the "regenerated" component
  rather than the "tracking" component).

**Total per-node storage**: O(F · C · A + 2F). Across N nodes:
O(N · F · C · A).

The mixture has a clear interpretation: lambda_v(theta_v) is the
posterior probability that at least one field event has occurred on
the path from v back to the last "tracking" ancestor, in which case
residues at v are at the stationary; with complementary probability,
the residues are informed by descendants via per-site tracking.

## Recovery of exact answers

The variational family contains the exact F^v in three regimes:

- **Cap-2 cherry** (depth 1, used in training): the exact F^v at the
  parent IS of the form `rank-1 + scalar` with the scalar equal to
  the stationary average. So the variational family contains the
  exact answer at the cherry parent; the ELBO is tight.
- **rho_chain → 0**: no field events ever, so all leaves are
  carry-tracked from the root. lambda_v → 0 for every v; q_v(theta_v)
  concentrates on the initial field; p_n_v carries the per-site
  per-class Q-evolution exactly. ELBO is tight.
- **rho_chain → infty**: every edge regenerates; all nodes are at the
  stationary. lambda_v → 1 for every v; q_v(theta_v) → pi; p_n_v
  irrelevant. ELBO is tight.

At intermediate rho_chain on deeper trees, the ELBO is an
approximation, but a principled one that captures the dominant
structure.

## ELBO

The ELBO over the tree's joint variational distribution
q(theta-trajectory, residues at internal nodes) = prod_edges q_edge ·
prod_nodes q_v:

ELBO(q) = sum_leaves E_{q_leaf}[log P(obs_leaf | x_leaf, theta_leaf)]
        + sum_edges E_q[log K_edge(child, parent)]
        - sum_nodes E_{q_v}[log q_v]
        + log Z_prior

where K_edge is the per-edge transition kernel evaluated at the
endpoint states. Standard mean-field VI structure.

For the variational family above each term factorises:

- **Leaf emission term**: at leaf with observation x^obs, the
  emission is delta(x_leaf = x^obs), so
  E_{q_leaf}[log P(obs | x_leaf, theta_leaf)] becomes the log marginal
  of x^obs under q_leaf, which is sum_{theta_leaf} q_leaf(theta_leaf) ·
  log[(1 - lambda_leaf(theta_leaf)) prod_n p_n_leaf(x^obs_n | theta_leaf)
       + lambda_leaf(theta_leaf) prod_n pi^(c_n, theta_leaf)(x^obs_n)].

  Computable in O(F · C · A).

- **Per-edge cross-entropy term**: E_q[log K_edge]. Under Interp 2 the
  kernel decomposes into the no-jump branch (`delta(theta_u = theta_v)`)
  and the >=1-jump branch (W(theta_v, theta_u; tau) · prod_n pi^(c_n, theta_u)).
  Both factor over n given (theta_v, theta_u), so:

  E_q[log K_edge] = log[ sum_{theta_v, theta_u} q_v(theta_v) q_u(theta_u) · K(theta_v, theta_u; tau, residue-marginals) ]

  with the inner residue-marginal K(theta_v, theta_u; tau, ·) being
  beta(theta_v, tau) · delta(theta_u = theta_v) · prod_n
  E_q[P_subst^(c_n, theta_v)(x_v_n, x_u_n; tau)] + W(theta_v, theta_u; tau) ·
  prod_n E_q[pi^(c_n, theta_u)(x_u_n)]. The state-dependent beta gives an
  extra F factor over the Interp 1 case but the per-site product
  structure is preserved; cost remains O(F² · C · A²) per edge.

  Under Interp 1 the same expression simplifies (uniform exp(-rho * tau)
  prefactor; no W table), giving O(F · C · A² + F²).

- **Per-node entropy term**: H(q_v) = - sum_{x, theta} q_v(x, theta) log q_v(x, theta).
  The mixture form makes this slightly delicate (log of a mixture).
  Standard approach: use the variational lower bound on -H by an
  upper bound on E[log q] obtained via Jensen against the mixture's
  component decomposition (or use a per-mixture-component variational
  inner bound). Closed form, O(F · C · A).

## Per-edge propagation in the variational family

Forward message from child u to parent v, restricted to the
variational form:

The exact one-edge propagation of `(rank-1 + scalar)` form preserves
the form (Step 2 of yesterday's proof). The form is preserved under
either interpretation:

- **Interp 2**: rank-1 multiplier is β_v(θ_v) = β(θ_v, τ) · β_u(θ_v)
  with state-dependent β; scalar γ_v(θ_v) is θ-DEPENDENT (carries the
  W(θ_v, θ_u; τ) integral over θ_u).
- **Interp 1**: rank-1 multiplier scales uniformly by exp(-ρτ);
  scalar γ_v is θ-INDEPENDENT (a single constant).

The variational coordinate-ascent updates have the same closed-form
structure in both cases; only the constants differ. Per edge: O(C · F · A²
+ F²) under either interpretation (the F² term is the field-transition
table evaluated at the (θ_v, θ_u) pair; same in both).

## Internal-node combine via moment-matching projection

This is where the variational approximation kicks in. At a binary
internal node v combining children u_1 and u_2, the **exact** product
of two `(rank-1 + scalar)` CLVs has 4 components (3 rank-1 + 1 scalar);
we **project** to the variational family by moment matching:

For each theta_v:

- q_v(theta_v) ← match the exact joint mass q_{u_1}(theta_v) q_{u_2}(theta_v)
  (normalised across theta_v).
- For each site n, p_n_v(x_v_n | theta_v) ← match the exact
  marginal of x_v_n given theta_v under the joint product (the
  per-site, per-(theta_v) marginal of the exact 4-term product).
- lambda_v(theta_v) ← match the total "regenerated" mass: the
  fraction of the 4-term sum mass that comes from terms involving
  at least one child's scalar component.

Each closed-form. Per binary combine: O(C · F · A) for the per-site
marginal matching, O(F) for q_v, O(F) for lambda_v. Per combine total:
O(C · F · A).

For k-ary nodes (k > 2): apply the binary combine k-1 times in
sequence, projecting after each.

## Coordinate ascent

The ELBO is bilinear-ish in the variational parameters (with the
mixture creating some non-convexity). Standard coordinate ascent:

1. Initialise q_v at uniform p_n_v = uniform over residues,
   q_v(theta) = pi_theta, lambda_v = 0.5 (a-priori we don't know).
2. Sweep over nodes in post-order (leaves up to root): for each node,
   update q_v from children via the projected combine; minimise the
   ELBO terms involving v while holding others fixed.
3. Sweep over nodes in pre-order (root down to leaves): for each node,
   update from parents.
4. Repeat until ELBO converges.

Per sweep: O(N · C · F · A²). Typical convergence: ~10-50 sweeps.

## When to build it

The dynfield core (cap-2 cherry training + IPHMM postprocessing on
sequence pairs) is already useful without this. The variational ELBO
becomes important when we want to:

- Evaluate dynfield log-likelihoods on full phylogenies (instead of
  just cherry sums).
- Use dynfield posteriors for ancestor-residue reconstruction.
- Compare dynfield vs Potts via tree-level held-out validation LL on
  general phylogenies.
- Extend to the dynamic-class-chain layer
  (`docs/dynfield_class_chain.md`), where the variational family
  becomes a 4-way mixture (carry × regenerated for both class and
  field chains).

**Recommended trigger**: when we want to publish a dynfield checkpoint
and demonstrate it competitively on a benchmark that requires more
than cap-2-cherry evidence (e.g., a phylogenetic-inference task on a
non-trivial tree). For now (TKF-DP training + IPHMM pairwise
postprocessing), cap-2 cherry exact suffices.

## Implementation outline

When we build it:

1. `coupling/dynfield/variational/`:
   - `posterior.py` — VariationalPosterior dataclass; per-node
     parameters (q_v, p_n_v, lambda_v) for each tree node.
   - `messages.py` — forward (post-order) and backward (pre-order)
     message-passing sweeps; per-edge propagation that preserves the
     variational form; internal-node moment-matching projection.
   - `elbo.py` — ELBO evaluation; per-edge cross-entropy, per-node
     entropy, leaf-emission contribution.
   - `optimisation.py` — coordinate ascent driver; ELBO convergence
     monitoring.

2. Adapter on the existing CouplingModel protocol:
   `DynamicFieldCouplingModel.evaluate_tree_loglik(tree, observations)`
   returning the converged ELBO. Falls back to exact cap-2 cluster
   emission when tree has depth 1.

3. Tests:
   - ELBO matches exact at cap-2 cherries (tightness test).
   - ELBO matches exact in rho_chain → 0 / → infty limits.
   - Coordinate ascent converges and ELBO is non-decreasing.
   - Forward-backward consistency (the variational posterior at one
     node from a forward sweep matches the backward sweep at
     convergence).

## Per-site class assignments under the variational family

The ansatz above fixes the per-site class `c_n` of each site. At
cap-2 cluster scale, class assignments are marginalised exactly by
the `K_c^2`-term sum that produces the M-tensor in
`coupling.dynfield.emission.class_marginal_doublet`. At general
cluster size `m` the analogous exact marginalisation costs `K_c^m`,
which destroys the linear-in-`m` cost the dynamic field was designed
to deliver. Two paths preserve linearity:

1. **Sampling.** `c_s` is a sampled hard assignment under a separate
   Gibbs sweep on `cls[s]` given the cluster's coupling-side
   observations. This is the path the TKF-DP training loop takes
   (`svi.class_gibbs_sweep_all_dynfield`), and is the right choice
   at training scale where per-cherry signal is too weak to drive
   a mean-field variational marginal to a hard fixed point reliably.

2. **Mean-field per-site marginals.** Add
   `q_n^v(c_n) ∈ Δ^{K_c - 1}` to the per-node variational state and
   marginalise inside the cluster emission:
   `Σ_{c_1,...,c_m} prod_n q_n^v(c_n) · P^(c_1,...,c_m, theta_v)`.
   Cost stays `O(L · m · k²)` per cluster.

Path 2 captures anti-correlation between coupled sites **only at
strong-signal coordinate-ascent fixed points** where each `q_n^v`
collapses to a Dirac on `c_n*`, recovering the hard-assignment
emission that exposes the per-class compositional bias responsible
for anti-correlation. At weak signal (few observations per cluster,
large `K_c`, or symmetric modes such as `(c_1*, c_2*) ↔ (c_2*, c_1*)`
contributing equal likelihood), mean-field marginals stay diffuse,
the emission averages over `K_c^m` class assignments — including
the `K_c^{m-1}` same-class assignments that are typically neutral on
anti-correlation — and the variational emission posterior loses the
anti-correlation structure that motivates the per-site class DP in
the first place.

**Mitigations** (escalating richness, applicable when downstream
metrics show the diffuse-marginal failure mode):

- **Structured-mean-field per cluster pair**: replace
  `prod_n q_n^v(c_n)` with
  `prod_{(i, j) ∈ pairs(C)} q_{ij}^v(c_i, c_j)`. Adds `K_c²` per
  pair, `K_c² · m(m-1)/2` per cluster; captures
  `(c_1*, c_2*) ↔ (c_2*, c_1*)` symmetry exactly.
- **Warm-start from sampled hard `cls`**: run a Gibbs sweep on
  classes first, then initialise `q_n^v` near the Dirac the sample
  defines. Coordinate ascent stays in the strong-signal basin.
- **Entropy tempering on `q_n^v(c_n)`**: dampen the entropy
  contribution early in the optimisation to push toward hard
  assignments, relax once the rest of the variational state has
  settled.

TKF-DP training sidesteps the issue entirely via path 1. The
mean-field class extension is relevant only when extending the ELBO
to general phylogenies, where aggregated per-tree signal at each
site is stronger than the per-cherry signal that defeats it at
training scale.

## Connection to the Potts variant

The Potts variant has a Cohn/Girsanov variational bound on the
coupled-CTMC bridge (supplement §sec:approx-suppl): the joint Q_pair
generator is bounded above by a constant-rate variational family with
closed-form pairwise bridge expectations. The bound is tight at the
cap-2 cluster (where the bridge is the cherry pair).

The dynfield variational ELBO is structurally analogous: tight at the
cap-2 cluster (where the per-node F has rank-1 + scalar form
exactly), principled extension to deeper trees via mean-field on the
per-node mixture. Both variants therefore have:

- Exact training-time inference on cap-2.
- Principled variational extension to phylogenetic inference.

This makes dynfield a proper drop-in replacement for Potts at the
Phase G level (full BAliBASE / phylogenetic comparison), not just at
the cap-2 cherry training level.

## Open questions

- **Posterior consistency at convergence**: after coordinate ascent
  converges, does the resulting q at each node respect the tree's
  conditional independencies? The mean-field-on-residues +
  mixture-with-stationary family is RICHER than strict mean-field, but
  may still leave some posterior bias compared to the exact joint.
  Worth empirically verifying on small trees where exact is feasible.

- **Initialisation**: cold-start (uniform p_n, q_v at the prior) vs
  warm-start (per-site Felsenstein-on-theta as a single forward sweep,
  then project to the variational family). Warm-start probably faster
  but needs verification.

- **Tighter ELBOs**: the moment-matching projection is the simplest
  closed-form; KL-minimisation projection (inner optimisation per
  combine) gives a tighter ELBO but with O(per-combine inner iters)
  overhead. Worth comparing.

- **Bookkeeping for tree topology**: the variational message-passing
  needs to handle non-binary internal nodes (Pfam guide trees have
  some). Standard approach: serialise k-ary combine as k-1 binary
  combines with intermediate projection.

- **Supplement edit**: `prop:hier-fels` in
  `psb-paper/supplement.tex:422` should be restricted to depth-1
  (cherries) or qualified with "approximate / variational on deeper
  trees". This is the LaTeX correction yesterday's audit didn't catch
  because cap-2-only usage hid the issue; flag it when this doc lands.
