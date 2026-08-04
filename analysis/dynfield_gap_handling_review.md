# Dynfield gap handling: math review

## Headline finding

**Confirmed: the dynfield cluster log-likelihood does not marginalise
gaps as missing data; it silently drops every cherry in which any
cluster column is gapped, which under the CRP cluster sweep acts as a
large positive reward for merging columns into co-gappy clusters.**
The bias is exactly `+|sum of (subset) log-likelihoods on cherries
whose evidence was discarded by the all-or-none filter|`. At Pfam
scale (50-150 singleton-observable cherries per column, ~3 nats each)
this is hundreds of nats per merge decision, dwarfing the CRP prior
and explaining the diagnostic-paper observation that only 3.0%
(K_c=4) and 1.8% (K_c=8) of size>=2 clusters have any jointly
observed cherry while clusters still saturate the size-16 cap. The
same skip-on-any-gap convention is reused at three live sites:
`make_cluster_loglik_fn` (`src/tkfdp/svi.py:1268`),
`extract_cluster_observations` (`src/tkfdp/svi.py:1306`), and
`class_gibbs_sweep_all_dynfield` (`src/tkfdp/svi.py:1503`); the
JAX-batched scorer in
`src/tkfdp/coupling/dynfield/emission.py:694-744` reproduces the same
pattern.

## Formal generative model with gaps

Per `math-paper/appendix-tkfdp.tex` Prop. \ref{prop:hier-fels} (l. 960)
and `docs/dynfield_math.md` Interp 2, conditional on the parent field
`theta_P` and the per-cluster classes, leaf residues at observed
positions emit independently per site under the 4-case kernel.
Indels are governed by an independent TKF92 seam; gaps are "site
present in the cherry geometry but the ancestor emission did not
survive on that edge", *not* "AA value unknown".

The per-(c, theta) factors satisfy
(`emission.per_class_field_cherry_sigma`, verified to 5e-17):

  sum_a pi^(c, theta)(a) = 1
  sum_b Sigma^(c, theta)(a, b; t) = pi^(c, theta)(a)
  sum_{a, b} Sigma^(c, theta)(a, b; t) = 1

So **marginalising the AA at any unobserved leaf collapses that factor
to 1**, and the cluster-emission formula reduces term-by-term to the
same 4-case expression on the observed-subset of sites. Numerically:
summing `cluster_emission_per_theta(size-2, X_j, Y_j)` over all 400
`(X_j, Y_j)` matches `cluster_emission_per_theta(size-1, i alone)`
to 7e-19.

## What the kernel evaluates today (current)

`src/tkfdp/svi.py:1263-1280`:

  log p_current(cluster | obs)
      = sum_{q : both_aa[q, cluster].all() } log P^(classes, t_q)(X[q, cluster], Y[q, cluster])

The selection mask `cherry_mask = both_aa[:, cols].all(axis=1)`
discards any cherry where any cluster column is gapped at either leaf.
Cherries with partial observation contribute exactly zero (not
marginalised — *omitted*).

## What the correct kernel should evaluate

Let `S_q = { s in cluster : both_aa[q, s] }` be the observed-subset.

  log p_correct(cluster | obs)
      = sum_{q : |S_q| >= 1} log P^(classes[S_q], t_q)(X[q, S_q], Y[q, S_q])

The 4-case kernel already accepts variable cluster size
(`cluster_emission_per_theta`,
`cluster_emission_batched`); the change is only the *per-cherry
cluster restriction* before scoring.

## The bias term

Let `M_full = {q : |S_q| = m}` and `D = {q : 1 <= |S_q| < m}`. Then

  log p_current(cluster) - log p_correct(cluster)
      = - sum_{q in D} log P^(classes[S_q], t_q)(X[q, S_q], Y[q, S_q])

Since log-probabilities are negative, **the bias is positive**: the
current scorer overscores the cluster relative to the correctly
marginalised value by an amount that grows with `|D|` (the number of
gap-disparate cherries) times the typical |log p| of the per-cherry
subset evidence. The more gap-disparate the cluster, the bigger the
bias.

The CRP-sweep impact (which is the actual training-time misbehaviour)
is sharper. For a candidate "move column s into existing cluster
c_other", the likelihood delta is
`loglik(c_other + s) - loglik(c_other)`. When s is **gap-disjoint**
from c_other — no cherry observes both jointly — the current code
gives `loglik_current(c_other + s) = 0`, so the singleton-on-s
evidence is wiped from the merge candidate. Compare to the
"go-singleton" candidate which scores `+loglik_singleton(s)` on the
same evidence. The merge beats the go-singleton by
`|loglik_singleton(s)|` purely from the gap-aggregation mechanism,
with no residue coupling involved.

## Numerical illustration (independence regime)

Toy: 3 sites (s, c0, c1), `K_c=2`, `L_max=3`, `rho_chain=0.7`,
`tau=0.4`. 50 cherries observe only s, 50 observe only (c0, c1)
jointly, 10 observe all three.

| score | current | correct |
|---|---|---|
| `loglik(c_other = c0+c1)` | -686.40 | -686.40 |
| `loglik(c_other + s)` | -198.01 (10 cherries) | -1123.40 |
| `loglik(singleton s)` | -440.54 | -440.54 |
| CRP delta "join c_other" | **+488.40** | -437.00 |
| CRP delta "go singleton" | -440.54 | -440.54 |

Under correct gap-marginalisation the two candidates differ by 3.5
nats -- the "no signal -> no preference" regime expected when residues
are AA-independent (residual = per-class pi heterogeneity). Under the
current scorer "join the co-gappy cluster" beats "go-singleton" by
**+929 nats** purely from gap-aggregation, matching the
discarded-subset-evidence formula.

## Other call sites and cross-checks

Three dynfield paths use the same `both_aa[:, cols].all(axis=1)`:

- `svi.py:1268` -- `make_cluster_loglik_fn` (CRP cluster Gibbs).
  Rewards merging into co-gappy clusters.
- `svi.py:1306` -- `extract_cluster_observations`, feeds tuples to
  `accumulate_cluster_stats_soft` (`updates.py:826`) which updates
  `pi_field` and `rho`. The atom update sees only the 3% of size>=2
  clusters with any fully observed cherry -- non-representative.
- `svi.py:1503` -- `class_gibbs_sweep_all_dynfield`. On the 97% of
  clusters with no jointly observed cherry the likelihood is 0 for
  every k; the class sweep is prior-only on most columns.
- `emission.py:694-744` -- the JAX scorer reproduces the same
  `both_at.all(axis=2)` + `jnp.where(cherry_mask, log_total, 0.0)`
  pattern; the GPU path has the same bias.

The Potts variant uses `valid = both_aa[:, s] & both_aa[:, t]` at
cherry edges (`_gather_pair_obs`, `update_potts_atoms_jit`). **Correct
for Potts**: the coupling factor is defined only for size-2 clusters
and has no size-1 marginalisation on the coupling side; per-class pi
and per-site eta updates collect their own column / class
contributions separately. The val-LL scorers (`val_loglik*.py`) use
single-column or pair masks; no cluster-level marginalisation.

## Is gap-marginalisation the right convention?

Yes, under the pairwise composite likelihood that `train_dynfield.py`
optimises. The dynfield emission factor is defined per (cluster,
theta_P) and factors over sites conditional on theta_P (Prop.
\ref{prop:hier-fels}); marginalising AA at an unobserved leaf
collapses the per-site factor to 1 by the stochasticity identities
above. The composite likelihood is `sum_{(cluster, cherry)} log P(observed (X, Y) | classes, t)`;
the seam-side log-prob has already been collected separately
(`pi_TKF92(A | t, theta_indel)`, `appendix-tkfdp.tex`
eq. \ref{eq:tkfdp-joint-marginal}, l. 2051). The dynfield factor is
*only* the substitution-side contribution; no other term in the
composite likelihood could be supplying the subset-on-S_q evidence
when `|S_q| < m`. The current silent-drop convention is not a
re-attribution -- it is an outright loss of evidence.

## Recommendation

**Minimal-invasiveness fix.** Change `make_cluster_loglik_fn` and the
matching JAX kernel: per cherry, compute the observed-subset
`S_q = {s in cluster : both_aa[q, s]}`, score the size-|S_q| cluster
`(classes[S_q], X[q, S_q], Y[q, S_q])` via the existing
`cluster_emission_per_theta` / `cluster_emission_batched` (which
already handle arbitrary cluster sizes), and sum log-probs over
`|S_q| >= 1`. Empty `S_q` still contributes 0. Same change at
`extract_cluster_observations` (emit one tuple per (cluster, cherry,
S_q) instead of skipping partials -- so the Dirichlet / TSB updates
see a representative sample) and at `class_gibbs_sweep_all_dynfield`.
The Potts pipeline and the val-LL scorers should remain unchanged.

After the fix, the diagnostic in
`analysis/dynfield_field_flip_diagnostics.md` should show cluster
sizes that no longer saturate the size-16 cap and a substantially
higher fraction of size>=2 clusters with any jointly observed cherry;
the static-vs-dynamic theta posterior may also shift because more
clusters will be subject to actual residue-level pressure rather than
vacuous prior-dominated merges.
