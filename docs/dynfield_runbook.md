# Dynfield variant: runbook

**Status**: 2026-06-28. End-to-end synthetic training works; real-Pfam
CLI driver and IPHMM postprocessing on real data are TODO.

This is the operator-facing companion to `docs/dynfield_design.md` (the
implementation plan) and `docs/dynfield_math.md` (the math derivation).

## What the dynfield variant is

A second `CouplingModel` variant alongside the cap-2 Sinkhorn-corrected
Potts coupling. The paper definition is in `psb-paper/psb2027.tex`
(Definition 2) and the supplement section `sec:dynamic-suppl`.

**Mathematical contract.** Coupling is mediated by a per-cluster shared
field selector `theta_C(t)` evolving on the tree as an F81-on-DP CTMC
with rate `rho_chain`; each coupled site has its own per-(class, field)
stationary `pi_field[c, theta]` and evolves under
`GTR(LG08, pi_field[c, theta])` between field jumps, resampling its
residue at each jump under the "instant re-equilibration" axiom
(Interp 2: textbook F81-on-DP CTMC, only real `theta -> theta'`
transitions count as jumps; see `docs/dynfield_math.md`).

**Computational contract.** Per cluster of size m, all emission
likelihoods cost `O(L_max * m)` — genuinely linear in cluster size, the
hier-fels scaling result. The training corpus need not cap clusters at
size 2 (the IPHMM-side pair-HMM postprocessing still does, by design).

## Top-level API

The variant exposes itself through the existing `CouplingModel` protocol
(`src/tkfdp/coupling/__init__.py`), with the registry key
`'dynamic_field'`. All dispatch happens through
`state.coupling_variant`; both `Potts` and `dynfield` SVIState objects
expose a `.coupling` property that returns the appropriate
`CouplingModel`.

```python
from tkfdp.svi import init_svi_state_dynfield, train_dynfield_full_iter

# per_family_data is the same structure exp2_pfam_v2.py consumes:
#   list of dict with 'family', 'L', 'K', 'aa_a', 'aa_b', 'both_aa', 'tau'.
state = init_svi_state_dynfield(
    per_family_data, K_c=4, A=20,
    L_max=8,                # DP truncation cap for the field selector
    alpha_field=1.0,        # DP concentration on the field
    rho_chain=1.0,          # F81-on-DP rate multiplier; 0 -> per-class GTR
    rng=np.random.default_rng(0))

for it in range(n_iters):
    state, info = train_dynfield_full_iter(
        state, per_family_data, rng,
        alpha_z=1.0,           # Ewens concentration on cluster partitions
        max_cluster_size=16,   # CRP truncation
        alpha_prior=1.0)       # Dirichlet base measure on pi_field
    # info has 'log_lik_total', 'n_clusters_total',
    #         'mean_cluster_size', 'max_cluster_size'.
```

What `train_dynfield_full_iter` does per outer iter:

1. **CRP cluster Gibbs** per MSA. Each column reassigns its cluster id
   under the Ewens prior + dynfield cluster emission likelihood
   (Neal 2000 Algorithm 3, finite-truncated to `max_cluster_size`).
2. **Cluster extraction** across the corpus into `(classes, X_obs, Y_obs, t)`
   tuples — one per `(cluster, cherry)` pair where every cluster member
   is observed.
3. **Soft EM atom update**: per-(theta_P, case) posterior over the 4-case
   cherry decomposition; fractional residue counts distributed to
   `(c, theta, residue)` bins; Dirichlet-conjugate update on
   `pi_field[c, theta]`; TSB Beta MAP update on `rho`.

## Checkpoint format

`save_checkpoint` / `load_checkpoint` (in `src/tkfdp/checkpoint.py`)
dispatch via `state.coupling_variant`. Dynfield checkpoints save:

- `state.npz`: `pi_class`, `pi_field`, `rho`, optional `tsb_betas`, plus
  per-MSA `cls_<fam>`, `partner_<fam>`, `eta_<fam>`.
- `meta.json`: `coupling_variant='dynamic_field'`, `K_c`, `a_eta`,
  `b_eta`, `kappa_pi`, `alpha_c`, `alpha_field`, `rho_chain`, plus the
  usual RNG state / family list / iter / early-stopping bookkeeping.

Legacy Potts checkpoints without `coupling_variant` in meta default to
`'potts'` on load.

## IPHMM postprocessing

`mcmc_infinite_phmm.py:precompute_partial_forward` dispatches through
`boost_state.tkf_state.coupling` (Phase B.3 wiring). With a dynfield
`SVIState` exposed as `boost_state.tkf_state`, the M-tensor build path
"just works": the `state.coupling` accessor returns a
`DynamicFieldCouplingModel` and `build_M_tensor` / `build_M_tensor_typed`
go through dynfield's per-(c, theta) machinery instead of the Potts
Sinkhorn-corrected joint generator.

The `pair_background='lg08'` argument (the IPHMM default, a Potts-side
convention for matching the released K=4 emwarm checkpoint's
training-time pair background) is silently accepted as a no-op for
dynfield. `pi_field` IS the background.

The Sinkhorn step on the indel seam is variant-specific: Potts applies
it; dynfield doesn't need it (the field-marginal joint `J` is
marginal-consistent by construction, verified in
`tests/dynfield/test_coupling_model.py` test 3 to 6e-17).

## What works (end-to-end synthetic)

```
$ python3 tests/dynfield/test_train_dynfield_end_to_end.py
[test 1] end-to-end full iter: cluster Gibbs + atom update -> LL
  iter 1: LL = -1691.98  (10 clusters, mean size = 2.00, max size = 7)
  iter 2: LL = -1330.45  ( 8 clusters, mean size = 2.50, max size = 8)
  ...
  iter 10: LL = -1178.79 (12 clusters, mean size = 1.67, max size = 4)
  LL gain: +513.20 nats over 10 iters
  family 0: recovered cluster sizes = [1, 1, 1, 1, 2, 4]    # == truth
            truth cluster sizes     = [1, 1, 1, 1, 2, 4]
  family 1: recovered cluster sizes = [1, 1, 1, 2, 2, 3]
            truth cluster sizes     = [1, 1, 1, 1, 1, 2, 3]   # close
```

Synthetic data with known cluster structure: from a uniform init, the
CRP sweep + atom update recover the truth cluster structure exactly
for family 0 (one size-4 cluster, one size-2, four singletons) and
close to truth for family 1 over 10 iterations.

## What's deferred

- **Class-label Gibbs sweep**. Currently `state.states_per_msa[*].cls`
  is fixed after init. A class sweep would resample `cls[s]` per column
  given the current cluster structure (analogous to `gibbs_sweep_K`'s
  `fix_partition=True` code path, adapted to `cluster_loglik_fn`).
  Without it, the K_c structure has to be set by the EM warmup on
  `pi_class` (variant-agnostic) and is fixed thereafter.

- **CLI driver script**. `experiments/train_dynfield.py` mirroring
  `exp2_pfam_v2.py` (CLI parsing, data loading, EM warmup, the outer
  loop above, periodic val LL eval, checkpoint save, early stopping).
  Mostly boilerplate; the atoms (data loading, EM warmup, val LL) are
  all variant-agnostic and reusable.

- **rho_chain learning**. The F81-on-DP rate multiplier is currently a
  hyperparameter (CLI default 1.0). For the Potts-limit consistency
  test (`rho_chain -> 0` recovers per-class GTR), it could either stay
  fixed at user-chosen values or be learned via a Gibbs/MH update on
  the cherry sufficient stats.

- **BAliBASE eval integration**. `experiments/eval_balibase.py` already
  dispatches via `state.coupling`; a dynfield checkpoint should work
  end-to-end, but the M-tensor + Sinkhorn-skip + lone-stationary
  consistency on the full BAliBASE pipeline has not been smoke-tested
  yet.

## Quick reference

| Module / function | Where | What it does |
|---|---|---|
| `coupling.dynfield.emission.cluster_emission_per_theta` | `src/tkfdp/coupling/dynfield/emission.py` | Size-m cluster emission (per-theta_P case decomposition) |
| `coupling.dynfield.updates.attribute_cluster_soft` | `src/tkfdp/coupling/dynfield/updates.py` | Soft-EM attribution for one cluster |
| `coupling.dynfield.updates.update_pi_field_dirichlet` | same | Dirichlet posterior mean update of pi_field |
| `coupling.dynfield.updates.update_rho_tsb` | same | TSB Beta MAP update of rho |
| `partition_K.gibbs_sweep_cluster` | `src/tkfdp/partition_K.py` | CRP-Gibbs cluster sweep |
| `partition_K.cluster_id_from_partner` | same | Adapter Potts pair-rep -> cluster_id |
| `svi.init_svi_state_dynfield` | `src/tkfdp/svi.py` | Init an SVIState with dyn_field |
| `svi.train_dynfield_one_iter` | same | Atom-update step (soft EM + Dirichlet + TSB) |
| `svi.train_dynfield_full_iter` | same | Full outer iter (cluster Gibbs + atom update) |
| `svi.extract_cluster_observations` | same | Walk MSAs + cherries -> cluster obs list |
| `svi.make_cluster_loglik_fn` | same | Per-MSA cluster scoring closure |
| `checkpoint.save_checkpoint` | `src/tkfdp/checkpoint.py` | Variant-dispatched save |
| `checkpoint.load_checkpoint` | same | Variant-dispatched load |

## Test inventory

`tests/dynfield/`:

| File | Tests | Coverage |
|---|---|---|
| `test_math_precompute.py` | 5 | F81-on-DP detailed balance, lumping, Kronecker identity at L=1, Interp 2 4-case closed form vs Gillespie MC, marginal consistency |
| `test_coupling_model.py` | 6 | Variant registry, 4-case construction, no-Sinkhorn marginal, M-tensor definition, typed M shape contract, npz round-trip |
| `test_rho_chain.py` | 4 | F81 decay scales linearly, rho_chain=0 / inf limits, default reproduces explicit formula |
| `test_updates.py` | 4 | Per-theta posterior marginalises to doublet, updates change model, training improves LL (hard-MAP), soft EM beats hard-MAP |
| `test_cluster_variable_size.py` | 4 | m=2 == size-2 specialisation, m=1 == singlet specialisation, m=3 training improves LL, per-theta sum == direct eval |
| `test_svi_dynfield.py` | 3 | init shape, coupling dispatches, train_one_iter improves LL |
| `test_checkpoint_dynfield.py` | 2 | save/load round-trip, per-MSA arrays |
| `test_cluster_gibbs.py` | 3 | partner <-> cluster_id, canonical numbering, CRP sweep recovers known structure |
| `test_train_dynfield_end_to_end.py` | 2 | Full iter on synthetic Pfam-shaped data, cluster extraction format |
| `test_iphmm_dispatch.py` | 4 | M-tensor with pair_background='lg08', typed M shape contract, lg08 == per_class, accessor consistency |

**Total**: 37 dynfield-specific tests.
