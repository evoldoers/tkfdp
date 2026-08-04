# Single-swap field (Cayley<=1 truncated permutation field) — model + JAX plan

A tractable truncation of the permfield "mood-light" model: the field theta keeps
only the identity and the C(C-1)/2 single transpositions (Cayley distance <=1),
giving **1 + C(C-1)/2** states instead of C!. A transposition tau_ab is exactly a
compensatory archetype swap (acid<->base salt-bridge), so this retains the flip
phenomenon while killing the factorial blow-up.

## Model (DONE + validated)

`src/tkfdp/permfield/field_ctmc.build_truncated_field(C, p, s0, w)` builds it
directly on the 1+C(C-1)/2 states (validated C=3,4,5 against `build_field` with
p_{>=2}=0: off-diagonal match, reversible, rows sum 0, STAR = no tau<->tau edges).

- **States**: id (d=0) + all transpositions tau_ab (d=1). arch[i,c] = archetype of
  class c in state i.
- **Dynamics** (star, reversible GTR): id <-> tau_ab only (tau<->tau' would be
  Cayley 2, dropped). `Q[id,tau_ab]=s0 w_ab pi_field[tau_ab]`,
  `Q[tau_ab,id]=s0 w_ab pi_field[id]`.
- **Residue model**: site of class c under field theta uses archetype theta(c);
  each archetype is an LG08-GTR with shared exchangeability S_LG08 and its OWN
  stationary pi^c. Site-class mixture weights rho (which class a column is).

### Fit targets (given cluster-annotated MSAs from PDB + inclusion criteria)
1. **archetype-swap rates** w_ab (C(C-1)/2) + field stationary ratio p1/p0.
2. **class mixture weights** rho.
3. **archetype stationaries** pi^c (C x 19).

## Product-of-trees ELBO (grounded in experiments/permfield_elbo.py numpy ref)

Per cluster of m columns on a tree: one FIELD tree (CTMC over the 1+C(C-1)/2 states)
+ m residue trees (each an archetype-GTR chain whose per-branch generator is chosen
by the field state at that node). Mean-field factorisation q(theta-tree) prod_j
q(residue-tree_j); the ELBO is the sum of per-factor tree log-partitions minus the
coupling entropy, exact within the product-of-trees family (the headline field-exact
bound, task #11). Structure-supervised: default singletons + loud warning; real PDB
contact pairs are the m=2 clusters ([[feedback_permfield_structure_supervised]]).

## E-step — HR bridge statistics (exact)

On each factor tree, endpoint-conditioned Holmes-Rubin expected dwell times T_x and
transition usages N_xy (tkfdp.permfield.hr.bridge, reversible eigendecomposition +
divided-difference kernel), summed over branches:
- **field tree**: T over the 1+C(C-1)/2 states, N over the id<->tau edges -> the
  sufficient stats for p, w.
- **residue trees**: per-archetype N^a, T^a substitution stats, accumulated with the
  field-state responsibilities (which archetype each class maps to under the field
  posterior) -> the stats for pi^c.
- class responsibilities gamma_j(c) (soft class assignment) -> rho.

## M-step — maximise expected complete-data log-likelihood

- **pi^c**: reversible-GTR stationary update from (N^a,T^a) (flux-gradient / F81-form
  closed step), as in permfield_elbo.
- **w_ab, p**: field GTR flux update from the field bridge stats (id<->tau edges);
  star structure => each w_ab depends only on its own edge's N,T; p1/p0 from the
  id-vs-tau occupancy.
- **rho**: rho_c = mean_j gamma_j(c).

## JAX implementation plan (the build)

Pure JAX, mapped/vectorised, NO internal numpy or python node-loops. **Validate against
the CORRECTED numpy reference `experiments/fit_single_swap_field.py`** (audit fixes:
`occ=rootc`, posterior root incidence, closed-form swap-rate M-step, `normalize_rate=True`
+ geomean(w)=1).

1. **Transition operators -- SPLIT into two classes (audit bug 2):**
   - *Bin-gatherable static tables* (one build per EM iteration, indexed by a geomspaced
     tau-bin): the field-tree `Pf(tau)` per bin (single `Qf`/iter); the pure-archetype
     `logP^a(tau)` per (archetype,bin); and the field/archetype HR divided-difference
     kernels per (generator,bin). All are `expm`/eig of a FIXED generator per iteration,
     so they gather cleanly.
   - *Per-branch recompute (NOT gatherable)*: the residue effective operator
     `Peff[(c,v)] = expm(gtr_Q(pibar_{c,parent(v)}) * tau_v)`, where
     `pibar_{c,u} = sum_a beta[c,u,a] * pi^a` is a field-POSTERIOR-dependent convex
     mixture (beta = the field marginal, recomputed every EM iter). There is NO
     (class,archetype,bin) key to gather this from. Because `gtr_Q` is linear in pi,
     contract `beta.pi -> pibar` once per (class,node) (cheap), then `vmap` a reversible
     eig of the per-mixture GTR (reusing the shared S) over (class,branch).
2. **Postorder-flat scan** (task #16): flat postorder array + parent indices; one
   `lax.scan` for the Felsenstein up-pass and the HR down-pass over masked padded slots.
3. **Logspace** (logsumexp) for the field and residue forwards.
4. **Batch over clusters** with `vmap` (padded to common m<=2, tree shape); the field
   forward vmaps the 1+C(C-1)/2-state chain, the residue forward vmaps per (class,branch).
5. **Mixed precision** (task #17): f64 for eig/divided-difference kernels + the field
   bridge (guard the flagged iter-0 field-HR blowup, ~1e265), f32 for the big matmuls.
6. **Staged validation (audit bug 8)** -- NOT "one full EM step to 1e-6":
   - *E-step gate* (the real 1e-6 gate, optimizer-independent, f64 + binning OFF + no f32):
     per-column `gamma`, `col_ll`, `es["obj"]`, `logZf`, field stats `Wf/Uf/rootc`,
     archetype stats `Na/Ta/roota`, `EdgeAcc/RootAcc`.
   - *M-step gate*: the swap-rate step now has a CLOSED FORM (above), so match it exactly;
     for `pi^a` either port the numpy gradient iteration exactly or match the M-step
     gradient at the numpy input params (do not expect damped post-M-step params to match).
   - *Objective parity*: reproduce `es["obj"]+logZf` bit-for-bit (it is a diagnostic proxy,
     NOT the ELBO -- audit bug 5); compare only basis-invariant eig quantities (P, T, N,
     logZ, sorted lam), never eigenvectors.
   - Binned/f32 forward validated separately at looser tolerance.

Entry point: `experiments/fit_single_swap_field.py` (numpy reference, DONE + audited),
reusing `build_truncated_field`, `tkfdp.permfield.hr`, and the padded-tree machinery from
`coupling/dynfield/phylo_elbo`. Fit for C in {2,3,4,...} on the PDB cluster-annotated
corpus (permfield_corpus.py / run_permfield_pdb.py).

## Known limitation (from the audit)

The swap rates `w_ab` are the model's defining novel parameter but are WEAKLY IDENTIFIED
at realistic data scale: field switches are rare (identity-concentrated stationary), so
per-pair usage `Uf[id,tau_ab]` is sparse and the closed-form `w_k = usage/exposure` is
high-variance (over-spread vs a near-uniform truth; synthetic corr ~0.27 at 200 clusters).
`pi^c` and `rho` recover well; `w` needs many field-switch observations (many contact-pair
clusters / deeper trees) or a weak prior on log-`w`. Report `w` with this caveat; it is a
statistical-power limit, not a fitting bug (the M-step is the exact argmax).
