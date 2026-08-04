# Dynamic latent-field variant: implementation plan

**Status**: design draft 2026-06-28, not yet implemented.

## What we're building

A second coupling model alongside the current cap-2 Potts variant, implementing
the `dynamic latent-field variant` of the model (paper Definition 2;
supplement §3 — `sec:dynamic-suppl`). The deliverables are:

1. **A trainer** (SVI / EM) for the dynamic-latent-field variant on Pfam,
   producing a checkpoint with the same on-disk shape contract as the
   Potts trainer (so downstream tooling — eval_balibase, composite log-LL,
   tkfdp.net release — works uniformly).
2. **An aligner** — i.e., variant support in the Infinite Pair HMM
   sampler — so a trained dynamic-field checkpoint can post-correct
   pairwise alignment posteriors the same way a trained Potts checkpoint
   does.
3. **Maximum factoring**: shared TKF92 indel machinery, shared SVI
   loop scaffold, shared partition Gibbs, shared checkpoint format,
   shared CLI surface. The variant difference should be confined to a
   small `CouplingModel` interface plus its two implementations. (The
   "variational" bit of the Potts variant's variational-EM is
   *Potts-specific*: it bounds the coupled-CTMC bridge with the Cohn
   ELBO. The dynamic-field variant has no coupled-CTMC at the
   substitution layer — emissions factorize given the θ-trajectory —
   so its coupled-emission step is *exact*, not variational. The rest
   of the EM scaffold is unchanged.)

## Math recap

**Potts variant (existing)**. Coupling carried by a residue-space tensor
`H ∈ R^{20×20}`, joint stationary
`pi_joint(a,b) ∝ pi^(c1)(a) pi^(c2)(b) exp(-h_a(a)-h_b(b)-H(a,b))` with the
Sinkhorn-determined side potentials. Per-class pi heterogeneity carries
positional composition; H carries correlation. Reversible at cap-2; needs
order-(n−1) compensatory potentials to scale to larger clusters.

**Dynamic latent-field variant (new)**. Coupling carried by a shared
categorical *field selector* θ_C(t) per key class, evolving on the tree as
an F81-on-DP CTMC:

- Field stationary: a Dirichlet process draw, stick-breaking weights ρ_θ.
- Field transition: Q(θ → θ') = ρ_{θ'} (constant in source — the F81 form);
  reversible w.r.t. ρ; stationary partition is the Ewens / CRP.
- Site emission **given the field**: each coupled site has its own GTR
  generator `Q^(c, θ)` with class-and-field-conditioned stationary
  `pi^(c, θ)`. Within a θ-segment a site evolves under `Q^(c, θ)` toward
  `pi^(c, θ)`; at each θ jump the site *instantly re-equilibrates* (samples
  from the new `pi^(·|c, θ')`). The instant re-equilibration is what keeps
  the chain reversible when θ changes the residue equilibrium (as opposed
  to merely the rate — the Tuffley-Steel covarion regime keeps the
  carried-residue branch reversible).

The key tractability gain (Proposition `prop:hier-fels`): conditional on
the θ-trajectory the per-site emissions factorize, so the per-node
conditional-likelihood vector factorizes over sites at each field value,
and the joint multi-component Felsenstein is *linear* in the number of
coupled sites — no `k^m` blow-up.

There are two independent optional dynamic-DP layers in the general
construction:

- **Dynamic field selector** (this plan): θ evolves over the tree, gives
  evolving coevolutionary regime.
- **Dynamic site classes**: c_s evolves over the tree on its own F81-on-DP,
  giving evolving per-site composition.

We're implementing only the field selector for now. The site-class DP
stays static (matches the current Potts-variant convention).

## Code architecture

### Abstraction: `CouplingModel` protocol

A single thin interface that both variants implement. Lives in
`src/tkfdp/coupling/__init__.py`.

```python
class CouplingModel(Protocol):
    """Abstract over the coupling component of the substitution model.
    Site-class machinery (cls_s, pi_class), TKF92 indel dynamics,
    partition Gibbs, and the eta/site rate hyperparameters are all
    OUTSIDE this protocol — they're shared across variants."""

    variant: str   # 'potts' or 'dynamic_field'
    K_c: int
    A: int

    # ---- training-time API ----
    def cluster_log_emission(
            self, cluster_obs, t,
            pi_class, eta, S
            ) -> np.ndarray:
        """log P(observations at sites in cluster | branch t, class pi).
        Shape (n_cherries,). The cap-2 Potts variant calls a 400×400 expm
        via the per-class-pair joint generator; the dynamic-field variant
        runs the hierarchical Felsenstein at cluster-of-2."""

    def m_tensor(
            self, t, pi_c, S
            ) -> np.ndarray:
        """M-boost tensor for the IPHMM: P_doublet / (P_singlet * P_singlet).
        Shape (A, A, A, A). The Potts version uses build_M_tensor; the
        dynamic-field version sums over the latent field using the
        hierarchical recursion."""

    def m_tensor_typed(
            self, t, pi_c, S
            ) -> dict[str, np.ndarray]:
        """Edge-type-aware variant for allow_id_edges. Same dispatch
        for both variants; both fall through to the typed builder."""

    # ---- updates (training inner loop) ----
    def update_atoms(self, sufficient_stats, ...) -> 'CouplingModel':
        """Atom-update (Laplace for Potts H; DP-Gibbs for field selector).
        Returns the updated model."""

    def update_dp(self, ...) -> 'CouplingModel':
        """DP concentration update (Escobar-West for alpha_H / alpha_field)."""

    # ---- serialization ----
    @classmethod
    def from_npz(cls, arrs, meta) -> 'CouplingModel': ...
    def to_npz(self) -> dict[str, np.ndarray]: ...
```

The protocol is intentionally small: both variants share the EM / SVI
loop structure, the partition Gibbs, the checkpoint frame, the val-LL
computation. Variant-specific math is confined to
`cluster_log_emission` + `m_tensor` + `update_atoms` + serialization.

### File layout

```
src/tkfdp/
├── coupling/
│   ├── __init__.py             # protocol + variant registry
│   ├── potts.py                # existing logic, moved here
│   └── dynfield/
│       ├── __init__.py
│       ├── dp_field.py         # F81-on-DP CTMC, stick-breaking, lumping
│       ├── hierarchical_fels.py # Prop hier-fels recursion
│       ├── emission.py         # pi^(c, θ), per-(c, θ) generator
│       └── state.py            # DynamicFieldState dataclass
├── svi.py                      # SVI loop — refactored to take a CouplingModel
├── mcmc_infinite_phmm.py       # build_M dispatches via CouplingModel
├── block_likelihoods.py        # generic singlet/doublet builders
└── checkpoint.py               # variant-aware load/save
```

### What stays shared

- `SVIState` (extended with a `coupling: CouplingModel` field instead of
  the existing `potts_dp: PottsDPState`)
- TKF92 indel dynamics, `lg08.py` exchangeabilities
- Partition Gibbs (`partition_K.py`)
- Site-class DP, EM warmup (`em_warmup_site_classes`)
- `eta_per_msa`, secret-destination conjugacy on `pi^(c)`
- Variational EM / SVI loop scaffold in `svi.run`
- Validation LL computation (`val_loglik_v2`) — both variants drive it
  through `cluster_log_emission`
- Checkpoint format frame: meta.json carries `coupling_variant`, the
  variant module is dispatched at load time
- Infinite Pair HMM sampler kernel (segment resample, edge add/remove,
  CRP MH correction, swap correction) — all variant-independent. Only the
  M-tensor build differs.

### What's variant-specific

- The atom side: Potts atom Laplace MAP vs DP-field stick-breaking +
  field-assignment Gibbs
- Cluster emission likelihood at coupled sites (Potts expm vs
  hierarchical Felsenstein)
- M-tensor build for the IPHMM
- Joint pair stationary used at indel seams (Sinkhorn-corrected Potts
  joint vs field-marginal joint)

## Per-component implementation tasks

### A. Math precompute & verification

- A.1 Closed-form (or eigendecomposed) F81-on-DP transition. With finite
  field cardinality `L`, the F81 generator
  `Q(θ→θ') = ρ_{θ'}` for `θ ≠ θ'`, with diagonals fixed; the matrix
  exponential has the standard F81 closed form:
  `P(θ→θ', t) = ρ_{θ'} + (δ_{θθ'} − ρ_{θ'}) exp(−t · sum_{θ''} ρ_{θ''})`.
  Lumping onto {occupied atoms} ∪ {tail} preserves the F81 form on the
  truncated state space and is exact for tree inference.
- A.2 Hierarchical Felsenstein recursion at cap-2 (Prop `prop:hier-fels`).
  At cluster size 2 the recursion is: at each node, the CLV is a
  vector over θ; emissions per θ factorize over the two sites; pruning
  multiplies parent CLV by transition × emission, summed over child θ.
  Cost: `O(edges · L^2 + edges · L · 2 · A^2)`. At cap-2 the recursion is
  trivial relative to its general form; matters most for the IPHMM
  M-tensor construction.
- A.3 Indel-seam consistency. Under instant re-equilibration the field
  resamples in concert with both substitution dynamics and indel events;
  the cluster's stationary is `sum_θ ρ_θ · pi^(c, θ)`, and this is the
  marginal used on lone (partial-presence) branches. Verify reversibility
  via the same marginal-consistency criterion (Thm `thm:revcond`).
- A.4 Felsenstein-on-IPHMM. The IPHMM sampler needs P_doublet for the M
  tensor. Under the dynamic field this is a *mixture over no-jump
  statuses* on each branch — for a single edge (IPHMM geometry), 2-case;
  for a cherry of diameter t (SVI training geometry), 4-case. Details and
  the closed forms are in `docs/dynfield_math.md` §3. The key
  factorisation: conditional on (θ_parent, jump_status_on_each_branch)
  the per-site emissions factor over sites — each per-(c, θ) block is
  20×20, so the doublet costs `O(K_c² · L_max · A⁴)` per unique t versus
  Potts's `O(K_c² · A⁴)` 400×400-expm per (c1, c2). Cheaper than Potts at
  small L_max but not free.

Deliverable: short standalone derivation document
(`docs/dynfield_math.md`) plus a numerical-verification Jupyter cell or
script — pick a small (k=4, L=2) fixture, build the joint two ways
(brute via Kronecker, via the hierarchical recursion), assert equal.

### B. Abstraction layer

- B.1 Define `CouplingModel` protocol (above). Keep small.
- B.2 Move existing Potts logic from `svi.py` /
  `block_likelihoods.py` / `mcmc_infinite_phmm.py` into
  `coupling/potts.py`, implementing the protocol. **No behaviour change**
  — existing tests should pass byte-identical on the Potts side.
- B.3 Variant registry in `coupling/__init__.py`:
  `VARIANTS = {'potts': PottsCouplingModel, 'dynamic_field':
  DynamicFieldCouplingModel}`. Checkpoint `meta.json` carries
  `coupling_variant: str`.

### C. Dynamic-latent-field core

- C.1 `coupling/dynfield/dp_field.py`:
  - DP stick-breaking weights (atom rho, tsb_betas analogous to existing
    Potts TSB).
  - F81-on-DP generator + JIT'd transition-matrix vmap at unique
    branch lengths.
  - Strong-lumping onto occupied-atoms-plus-tail. The lumped state
    transition stays F81; correctness check unit test.
- C.2 `coupling/dynfield/emission.py`:
  - Per-(class, field) stationary `pi^(c, θ) ∈ Δ^{A−1}`. Stored as an
    array of shape `(K_c, L_max, A)`.
  - Per-(class, field) generator: GTR(LG08, pi^(c, θ)) F81-form (same
    construction as the existing per-class generator).
  - Cap-2 cluster joint emission via the hierarchical recursion
    (lazy / vmap'd over unique tau).
- C.3 `coupling/dynfield/hierarchical_fels.py`:
  - At cluster size 2, the recursion is a per-θ-marginalized
    per-cherry forward-backward; closed form for cherries (pair of
    leaves with the field constant across the cherry's branch unless
    long enough for a field jump, in which case mix accordingly).
  - For longer trees (postprocessing): generic Felsenstein over the
    lumped θ alphabet. Not needed for the cap-2 cherry training but
    needed for downstream phylogenetic inference.
- C.4 `coupling/dynfield/state.py`:
  - `DynamicFieldState` dataclass mirroring `PottsDPState`:
    `pi_field: (K_c, L_max, A)`, `rho: (L_max,)`,
    `tsb_betas: (L_max - 1,)`, `alpha_field: float`,
    `per_cluster_field_assignment: dict[cluster_key, int]`.
  - npz serialization with `coupling_variant = 'dynamic_field'`
    in meta.

### D. SVI integration

- D.1 `svi.py` extension. The SVI loop currently mutates
  `state.potts_dp` via `update_potts_atoms_jit`. Refactor to call
  `state.coupling.update_atoms(sufficient_stats)`. For the dynamic-field
  variant, this is a Gibbs sweep over field assignments (one per cluster,
  conditional on observations) plus a stick-breaking update on the field
  DP.
- D.2 Sufficient-statistic protocol. The Potts side accumulates
  per-(class-pair) observation tensors for the Laplace MAP. The
  dynamic-field side accumulates per-(class, cluster, field) counts and
  per-(class, field) per-residue counts. Both fit the same "observation
  tensor + index" pattern; expose as a single
  `state.coupling.observation_dtype()` + a per-cherry gather call.
- D.3 EM warmup. Field-side warmup: initialize `pi^(c, θ)` from EM on
  per-class pi (existing), then add small per-θ perturbation per
  Dirichlet draw. Optional separate `--em-warmup-fields` flag for
  variant-aware warmup hyperparameters.
- D.4 Per-cherry log-likelihood
  (`_precompute_pair_LL` and friends). Currently built around the
  per-(c1, c2) class-pair Potts log P cache. Refactor to call
  `state.coupling.cluster_log_emission(...)`, which dispatches the same
  shape `(K_c, K_c, n_t, A^2, A^2)` (or `(K_c, n_t, A, A)` for singletons)
  — Potts version unchanged; dynamic-field version builds via the
  hierarchical recursion.

### E. Infinite Pair HMM integration

- E.1 `precompute_partial_forward`. Currently calls `build_M_tensor` and
  `build_M_tensor_typed` from `block_likelihoods.py`. Refactor to call
  `state.coupling.m_tensor(...)` and
  `state.coupling.m_tensor_typed(...)`. The IPHMM kernel below is
  variant-agnostic.
- E.2 Joint pair stationary for the indel-seam. The IPHMM doesn't
  currently materialize `pi_joint` per (c1, c2) directly — the M tensor
  encodes the boost — but for I/D edges the M-tensor builder does need
  pi_joint. Under dynamic field, `pi_joint(a, b | c1, c2) = Σ_θ ρ_θ
  pi^(c1, θ)(a) pi^(c2, θ)(b)`. This is a field-marginal mixture of rank-1
  joints; reversible without Sinkhorn (the marginal is `Σ_θ ρ_θ pi^(c, θ)`
  by construction). So **the Sinkhorn step is variant-specific**: applied
  for Potts, not needed for dynamic field.
- E.3 The IPHMM kernel itself (segment resample, edge add/remove, CRP
  correction, swap, I/D anchor handling) is unchanged. Tests should pass
  byte-identical on the Potts side under the abstraction.

### F. Trainer / CLI

- F.1 `experiments/exp2_pfam_v2.py`:
  - Add `--coupling-variant {potts,dynamic_field}` (default `potts`).
  - Add `--field-cardinality L_max` (default 8 — the truncation cap for
    the field DP).
  - Add `--alpha-field α_field` (default 1.0).
  - Preflight banner: warn loudly if `--coupling-variant dynamic_field`
    is combined with `--pre-sinkhorn` (incoherent — Sinkhorn doesn't
    apply to the dynamic field).
- F.2 New entry-point script (or option) for variant-aware checkpoint
  inspection: `experiments/show_coupling.py --chkpt <dir>` printing the
  variant, the trained pi_field / rho / atoms, and a small summary
  (top-favored within-field pair emissions etc.). Sanity tool.

### G. Tests

- G.1 Unit tests in `tests/coupling/`:
  - `test_dp_field_reversibility.py`: detailed-balance check on the
    F81-on-DP CTMC at L=4, random stick-breaking.
  - `test_dp_field_lumping.py`: lumped vs un-lumped transition agreement
    to machine precision under arbitrary `occupied + tail` partition.
  - `test_hierarchical_fels_against_brute.py`: cap-2 cluster joint
    emission via hierarchical recursion vs brute Kronecker.
- G.2 SVI smoke under dynamic field: K_c=4 L_max=2 on top-100 Pfam, run
  3 outer iters, assert NB_total improves monotonically. Mirror
  `tests/test_a1_rung.py` style.
- G.3 IPHMM brute-force under dynamic field: extend
  `tests/test_a1_equilibrium.py::test_B_id_edges_brute_force_detailed_balance`
  to dispatch on coupling variant; the brute-force enumeration takes the
  variant-specific M-tensor instead of the typed Potts one.
- G.4 **Per-class-GTR limit consistency** (revised 2026-06-28). The
  reduction to per-class GTR is *not* `L_max=1` — at `L_max=1` the
  F81-on-DP chain still has rate-1 self-jumps which, under instant
  re-equilibration (supplement.tex §sec:reequil-suppl "Carry vs.
  resample"), refresh the residue from `pi_field[c, 0]` at exponentially
  distributed times. So dynfield at `L_max=1` is a covarion-like model,
  not plain per-class GTR.
  The correct reduction is **`rho_chain → 0`** (the scalar rate
  multiplier in the supplement's `Q(theta → theta') = rho_chain *
  pi_{theta'}`). At `rho_chain = 0` the field never jumps, residues
  evolve under `Q^(c, theta_init)` throughout, and dynfield equals
  per-class GTR with the initial-field stationary `pi_field[c, theta_init]`.
  Implementation TODO: expose `rho_chain` as a CLI argument
  (`--rho-chain-field`) and as a parameter on `DynamicFieldState`. The
  test trains
  `--coupling-variant dynamic_field --rho-chain-field 0 --field-cardinality 1`
  and asserts the checkpoint `pi_class` agrees with the
  `--coupling-variant potts --K-H-max 0` per-class GTR baseline to
  numerical noise.

### H. Documentation & paper

- H.1 Update `psb-paper/supplement.tex` §3 (`sec:dynamic-suppl`): we
  already have the prose; add a short subsection on practical truncation
  (L_max as cap) and on the EM updates we implemented.
- H.2 Add `docs/dynfield_runbook.md` mirroring
  `docs/balibase_paired_fsa_quickstart.md` — training command, expected
  wall time, val-LL comparison protocol against a Potts baseline,
  released-checkpoint reproduction recipe.
- H.3 Update `CLAUDE.md` to point at this design doc and the runbook.

### I. Backward compatibility

- I.1 Existing Potts checkpoints (`coupling_variant` absent from meta)
  default to `'potts'` at load.
- I.2 `eval_balibase.py` already accepts model tags; add
  `--coupling-variant` as a per-model attribute or read it from each
  checkpoint's meta.json (preferred — single source of truth).
- I.3 The `--pre-sinkhorn` CLI flag is Potts-only (the dynamic-field
  variant has no Sinkhorn step). Preflight banner already covers the
  incoherent combo.

## Estimated effort

- **Phase A (math)**: 1 day. Mostly a derivation note + the Kronecker
  vs hierarchical cross-check.
- **Phase B (abstraction)**: 2 days. The blast-radius is moderate — 4-5
  files in `src/tkfdp/` touched, no behaviour change on the Potts side.
  Test coverage already adequate to catch regressions
  (E.3 / E.4 / A1 corrections / I/D brute force).
- **Phase C (dynfield core)**: 3 days. Small new module, well-scoped math.
- **Phase D (SVI integration)**: 2 days. Mostly piping; the field
  selector Gibbs is the only non-trivial new training move.
- **Phase E (IPHMM integration)**: 1 day. M-tensor swap-in is small;
  no kernel changes.
- **Phase F (CLI)**: 0.5 day.
- **Phase G (tests)**: 1.5 days. The Potts-limit consistency test is the
  most informative — if it passes, the whole stack is wired correctly.
- **Phase H (docs)**: 0.5 day.

**Total**: ~10–12 working days for a one-person implementation, assuming
no architectural surprises. The abstraction phase is the load-bearing
one: get the `CouplingModel` protocol right and the rest is mechanical.

## Open questions / decisions

- **Field cardinality default**. The DP truncation cap. Start with L_max=8
  (same as the Potts K_c=8 we're re-estimating). CLI flag, not a model
  hyperparameter that needs Escobar-West updates beyond the
  stick-breaking.

- **Field selector scope**. **Decided (2026-06-28): one shared DP, one
  field selector per cluster.** Stick-breaking weights `ρ_θ` and DP
  concentration `α_field` are global; each cluster carries a single
  θ-trajectory shared by its members. Per-class structure enters only
  through the per-(class, field) emission stationary `pi^(c, θ)` (the
  `(K_c, L_max, A)` tensor), which is what tilts classes differently
  under the same field — the necessary and sufficient condition for
  anti-correlation between coupled sites. A class-indexed `ρ^(c)` would
  give each class its own field trajectory, which has no clean
  semantics in a *shared-coupling-mediator* picture (the whole point of
  the latent field is one trajectory per cluster) and is dropped.

- **Trainer wall-time**. The Potts cap-2 cluster emission costs a 400×400
  expm per (c1, c2, t). The dynamic-field cap-2 cluster emission is
  exact in closed form: a 4-case sum over no-jump status on the two cherry
  edges, with the (no-jump, no-jump) case carrying a per-(c, θ_P) GTR
  cherry-joint `Σ^(c, θ_P)(a, b; t)` — i.e., a 20×20 expm per (c, θ_P).
  Doublet cost per cherry is therefore `O(K_c² · L_max · A⁴)` ops per
  unique t versus Potts's `O(K_c² · A^6)` per t (the 400×400 expm is
  `A^6 = 6.4e7`). At K_c=4, L_max=8, A=20: ~2e7 ops per t for dynfield
  vs ~1e9 for Potts — ~50x cheaper, not the 11000x earlier draft
  claimed. Verify in the first smoke run.

## What was reconsidered along the way

- **No ELBO at coupled sites.** Earlier drafts of this plan imported the
  Cohn/Girsanov variational ELBO from the Potts variant by reflex. It
  does not apply. Under instant re-equilibration the conditional
  emissions factorize over sites given the θ-trajectory, the F81-on-DP
  field selector is exactly Felsenstein-computable on a lumped finite
  alphabet, and the per-(c, θ) generator is a 20×20 reversible CTMC
  with a one-time eigendecomposition. The per-cherry joint
    P(X_i, X_j, Y_i, Y_j | t) =
       Σ_{θ_P, θ_X, θ_Y} ρ_{θ_P}
                          · P_F81(θ_P → θ_X; t) · P_F81(θ_P → θ_Y; t)
                          · pi^(c_i, θ_X)(X_i) pi^(c_j, θ_X)(X_j)
                          · pi^(c_i, θ_Y)(Y_i) pi^(c_j, θ_Y)(Y_j)
  is exact; no variational bound is needed at any point in the SVI
  loop on the coupled-substitution side. Conjugacy + Laplace remain
  variant-specific.
