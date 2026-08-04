# Supervised enum400 dynfield training: E/M derivation and validation

Status: new pipeline (2026-07). Files: `experiments/train_supervised_enum400.py`
(entrypoint), `src/tkfdp/coupling/dynfield/phylo_elbo/supervised_trainer.py`
(reusable E/M module). Model spec: `math-paper/appendix-tkfdp.tex`,
"Rate heterogeneity (+Gamma+I)".

## What is learned vs fixed

On the **fixed PDB contact partition** (size-2 salt-bridge / cation-pi /
disulfide / etc. contacts as clusters, every other column a singleton), we learn

1. `pi_archetype` — the K_a = 20 archetype amino-acid profiles (each a
   distribution over the 20 AAs). Warm-started from LG-C20 (`--scratch` for a
   flat init).
2. the **field rate**: the +Gamma+I field-rate bin weights `w` (including the
   invariant/rate-0 "no-flip" bin) and the field-chain rate `rho_chain`.

Held **fixed**:

- the DM class-concentration `alpha` (never updated — passed in / flat default);
- the cluster labels / partition (PDB-supervised — never sampled or moved);
- the exchangeability `S`, the field stationary `rho`, per-site substitution
  rate bins, the enum400 `arch_assignment` (the identity `c = (k0, k1)`
  enumeration; the archetypes it points at are what we learn).

The **site class** `c_s` (= which archetype pair `(k0, k1)` a column takes) is
**marginalized, not sampled** — the collapsed / Rao-Blackwellized view
(`pairing_cache.py` / `collapsed_discovery.py`). Concretely: for a singleton we
sum over all K_c = K_a^2 classes; for a contact pair we sum over a top-N x top-N
class grid.

## Why

On real salt-bridge contacts the +Gamma+I field-rate prior is badly
mis-specified: fitting the field-rate bin weights (`experiments/fit_pdb_hyperparams.py`)
drives the invariant (rate-0) weight from its 0.50 prior down to ~0.04, i.e. a
flip-prevalence of ~0.96. Under the prior the model reads a compensatory charge
flip as static substitution ("never flips" by default). Learning the field rate
(and the archetypes) supervised on the contacts corrects the charge-under-read.

## Objective

The alpha-fixed, class-and-rate-marginalized corpus log-likelihood

    F(Theta) = sum_C  log sum_g w_g sum_{c in grid(C)} P(c) L_C(c, g; pi_archetype, rho_chain)

where Theta = { pi_archetype, w, rho_chain }, C ranges over clusters, g over the
G field-rate bins (rate multiplier r_g on rho_chain, r_0 = 0 invariant), c over
the class grid, P(c) the fixed DM class prior (flat by default, so it cancels),
and L_C the tree forward log-lik (mm forward, per bin — the sampler's own
scorer, ~21x faster than the 800-state exact peel and >= as accurate as the
tau-binned exact peel; `--exact` keeps the peel).

We fit `w` and `rho_chain` on the **contact clusters** (the supervised
coevolution signal — this is what makes the field-rate readout a statement about
real contacts) and `pi_archetype` over **all columns** (a global stationary).

## E-step (per cluster, per field-rate bin)

Score `L_C(c, g)` on the grid (pairs: top-N^2 chosen by the rate-marginal
singleton LL; singletons: all K_c) with the per-bin mm forward
(`field_rate_discovery._score_specs_rates`, returns `(n_specs, G)`).

Per-bin class-marginal evidence (add the fixed class log-prior `log P(c)`):

    M_C(g) = logsumexp_c [ log P(c) + L_C(c, g) ].

Field-rate responsibility (E over the field rate):

    r_C(g) = w_g exp(M_C(g)) / sum_h w_h exp(M_C(h)).

The per-cluster flip posterior is `phi(C) = 1 - r_C(0)` (mass on the
non-invariant bins).

## M-step

### (a) Field-rate bin weights — mixture EM (monotone)

    w_g  <-  (1/|C|) sum_C r_C(g).

Standard finite-mixture EM on the field-rate bins; monotone in F holding
`pi_archetype`, `rho_chain`, and the per-bin `M_C(g)` fixed. The fitted `w_0`
is the invariant weight; `1 - w_0` (mean over contacts of `phi(C)`) is the
flip-prevalence readout. This is exactly the field-rate M-step in
`fit_pdb_hyperparams.py`.

### (b) rho_chain — generalized-EM line search (accept-if-better)

`rho_chain` enters `L_C` nonlinearly (it scales the field-chain generator), so
there is no closed form. We maximize `F_contacts(rho_chain)` by re-scoring the
contacts on a small multiplicative grid around the current value
(`x {0.5, 0.7, 1.0, 1.4, 2.0}`) and taking the argmax — a GEM step, accepted
only if it does not decrease F. Only the field kernels (`beta`, `W`, built from
`bin_centers`, `rho`, `rho_chain * r_g`) depend on `rho_chain`; `P_sub` does not,
so re-scoring rebuilds only the G rate kernels.

### (c) pi_archetype — collapsed per-archetype Holmes-Rubin M-step

We reuse the F81 secret-destination Dirichlet-conjugate update
(`svi.update_pi_class` / `secret_destination.py`), with the K_c hard classes
replaced by K_a **soft archetype responsibilities**.

**Responsibilities.** For the stationary (emission) update the relevant question
is *which archetype's F81 stationary explains column s's residues*. Under a
frozen field a column emits from a single archetype, and its per-column
frozen-field (invariant-bin) single-archetype forward
`Ss[k] = _col_invariant_static(k)` is exactly the marginal log-likelihood of
column s under archetype k's F81. So

    rho_s(k)  =  softmax_k ( Ss[k] + log pi_prior(k) ),   pi_prior flat over K_a,

a soft assignment of each column to the K_a archetypes (K_a = 20 forwards per
column — cheap; one batched invariant-bin call for all columns). This is the
field-occupancy-collapsed realization of the fuller target

    rho_s(k) = rho_0 q^{(0)}_s(k) + rho_1 q^{(1)}_s(k),
    q^{(theta)}_s(k) = sum_{c: arch[c,theta]=k} q_s(c),

(field occupancy `rho` times the marginal class posterior that field-state
`theta`'s archetype is `k`). We use the frozen-field form because it is
tractable (K_a per column, not K_a^2) and directly targets the stationary; the
dynamic two-archetype (flip) structure is carried by the field-rate part of the
model, not by the stationary estimate.

**Sufficient statistics.** Sample one whole-tree residue history per family from
the LG08 CLV posterior (`FamilyCLV.sample_history` -> `extract_branch_cherries`),
giving per-branch endpoint pairs `(aa_a, aa_b, tau)` per column. For each
archetype k with generator `Q_k = F81(pi_archetype[k], S)`, accumulate
responsibility-weighted Holmes-Rubin stats (`eta_site.hr_batch_jit`) over all
(column s, branch b) with `rho_s(k)` above a small threshold:

    dwell[k, :]   = sum_{s,b} rho_s(k) * E[dwell_x | a,b,tau; Q_k]
    N_real[k, y]  = sum_{s,b} rho_s(k) * [aa_b(s,b) = y]        (destination counts)

**Update.** The secret-destination ghost counts + Dirichlet posterior mean:

    N_ghost[k]   = pi_k * (T_S - dwell[k] @ S_off)              (EM ghost expectation)
    pi_archetype[k] <- Dir-mean( kappa_pi * pi_bar + N_real[k] + N_ghost[k] ),

iterated a few times for ghost/pi consistency (`svi.update_pi_class`, K_c -> K_a).
`pi_bar = PI_LG08`, `kappa_pi` small. Guarded: the candidate `pi_archetype` is
accepted only if it does not decrease F (else damped toward the previous value,
down to a skip) — so the monitored F stays monotone.

### (d) Refresh

After any `pi_archetype` / `rho_chain` change, rebuild the discovery-state
derived tables (`refresh_pi_field` + `build_discovery_state`: K_a x n_bins
`P_sub` slices + G field kernels — the corpus trees/tau-bins are built once).
`alpha` and the partition are never touched.

## Approximations (stated honestly)

1. **Field occupancy** taken at the frozen-field per-column archetype posterior
   rather than endpoint-conditioned per branch — a mean-field factorization of
   the field posterior from the residue posterior.
2. **Substitution histories** sampled from the fixed LG08 tree posterior (the
   CLV), not the dynfield posterior — the plug-in used throughout this pipeline;
   HR endpoint expectations are then computed under each archetype's own `Q_k`.
3. **Per-column class marginal** used for paired columns too (the pair coupling
   drives the partition and field-rate; the per-column marginal is the leading
   term for the emission stationary).

Because each M-step is either a proper EM part (w) or a GEM step accepted only
on improvement (rho_chain, pi_archetype), the monitored F is non-decreasing by
construction.

## Validation

Small subset: salt-bridge contacts + confirmed flips over a few dozen train
families, CPU (`JAX_PLATFORMS=cpu`; both GPUs were busy). Results are filled in
by `train_supervised_enum400.py --report` (see the run log / `_report.json`):

- (i) marginalized log-lik F increases monotonically each EM iteration;
- (ii) the fitted field-rate flip-prevalence rises from the 0.50 prior toward
  the ~0.96 measured on real salt bridges (invariant weight 0.50 -> ~0.04);
- (iii) learned `pi_archetype` stays sane — the acidic archetypes stay acidic
  and the basic archetypes stay basic (net charge = P(basic) - P(acidic) keeps
  its sign; no collapse to a single profile);
- (iv) `alpha` and the partition are unchanged (asserted).

### Measured (validation subset)

FILLED IN BY THE RUN — see `_report.json` written next to the checkpoint and the
summary block echoed at the end of the run log.
