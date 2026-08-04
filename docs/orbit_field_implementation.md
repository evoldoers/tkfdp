# Archetype-orbit permutation-field: implementation plan (QUEUED)

Status: **derivation done** (`math-paper/appendix-tkfdp.tex`,
`par:archetype-orbits`); **code not started**. Execute on user say-so.
This plan is grounded in the current `src/tkfdp/coupling/dynfield/phylo_elbo/`
code so it can be picked up cold.

## What we're building

Replace the free `arch_assignment` `(K_c, L_field)` table with an **orbit
partition** of the `K_a` archetypes; the per-cluster field is a permutation
of one orbit's `Sym(B_b)`. Frozen-diagonal base (`K_c = K_a = K`, class ≡
archetype), so the orbit partition is the *entire* learned field structure.
Goal: make "flip = within-orbit swap" structural (fixes the charge-crossing-
at-chance result), and collapse arch MCMC to split/merge on a K-element
partition.

## The field state space per cluster (independent per-orbit chains)

Each orbit has its **own** F81-on-DP field chain, evolving **independently
(asynchronously)**. So a cluster's likelihood **factorises over the distinct
orbits its columns touch** — the *only* coupling is between columns that
**share** an orbit. This keeps the field local to an orbit, independent of how
the rest of the archetype vocabulary is partitioned (the reason to prefer it).

- **same orbit** `{A,B}`, columns (A,B): 2-cycle **{AB, BA}** — compensatory;
- **same orbit, both class A**: 2-cycle **{AA, BB}**;
- **A + singleton E**: **{AE, BE}** (E static);
- **different orbits** `{A,B},{C,D}`: **factorises** into two async singletons
  (one A↔B, one C↔D) — the `{AC,AD,BC,BD}` product does NOT appear.

So every forward runs on `Sym(one orbit)`, `L = |B|! ≤ s_max! = 2` — the
**max orbit size, never a product**. A per-cluster `+Γ+I` rate drives the
chains (`r=0` = static).

Implementation (`orbit_scorer.py` numpy ref, `orbit_scorer_jax.py` JAX):
group columns by orbit; per group run `orbit_scorer_jax.score_cluster` (a
forward over `Sym(that orbit)`, positional tables `pi=(m,L,A)`,
`P=(nb,m,L,A,A)`, `classes=[0,1]`, reusing `exact_cap2_jax` verbatim for
`L∈{1,2}`); sum group log-LLs. `P_arch` bins are orbit-independent (never
rebuilt on an orbit move). Validated == numpy to 1e-8 (incl. the factorised
different-orbit case).

## Data structures (corpus_state.py)

- Add `orbit_id: np.ndarray` shape `(K_a,)` int — the partition (orbit label
  per archetype). Init all-singletons (`orbit_id = arange(K_a)`).
- Field stationary is **uniform over each orbit's `Sym(B_b)`** (per-orbit
  chains); a learnable swap prior `(1-p,p)` per pair-orbit is a later refinement.
- **Per-cluster** field rate: marginalize the existing `+Γ+I` grid (`rates`,
  `weights`) per cluster exactly as `field_rate_*` does — invariant bin `r=0` =
  static. (Shared across the cluster's per-orbit chains; the chains stay
  independent in their jump *times*.)
- **Drop** `arch_assignment` `(K_c,L_field)` and its `_pi_field` for the
  orbit model (keep the archetype profiles `pi_archetype`, `(K_a,A)`, and the
  per-archetype `P_sub` bins — already built by
  `field_rate_trainer.build_P_sub_aug`).
- Site-rate `+Γ+I` augmentation is orthogonal: keep folding the substitution
  rate bin into the archetype index (`archetype × rate_bin`), exactly as the
  current `_aug` does — the orbit permutation acts on the base-archetype part
  only.

## Per-cluster scorer (implemented + validated)

`orbit_scorer.score_cluster_orbit` (numpy ref) / `orbit_trainer.score_cluster`
(driver): **group the columns by orbit**; per group run one `exact_cap2`
forward over `Sym(that orbit)` — a coupled pair (columns sharing the orbit,
`L≤2`) or a single (`L≤2`) — and **sum** the group log-LLs; per rate bin `r`,
then marginalise the `+Γ+I` grid. The per-group JAX primitive
`orbit_scorer_jax.score_cluster` passes **positional** tables (`pi_arch[a]` →
`(m,L,A)`, `P_arch[:,a]` → `(nb,m,L,A,A)`, `classes=[0,1]`), reusing
`exact_cap2_jax.{exact_pair,exact_single}_tree_ll` verbatim for `L∈{1,2}`.
`P_arch` (per-archetype expm bins) is orbit-independent — built once, never
rebuilt on an orbit move.

Validated (`tests/dynfield/test_orbit_reductions.py`): numpy == independent
ground truth (plain Felsenstein / compound-generator brute force), including
the **factorised** different-orbit case (== sum of two single brute forces);
JAX per-group == numpy to 1e-8 across `L∈{1,2}`, plus a factorised-sum check.

**Correctness subtlety banked:** a singleton orbit is `L=1` (truly static) — a
field over identical profiles would still resample the residue at each jump, so
`L=2`-degenerate ≠ static. The per-orbit enumeration gives `L=1` for singletons
automatically.

**Batching (Stage B optimization, deferred):** group units by `(tree-shape,
L)` and vmap; `L∈{1,2,4}` → few buckets. The per-cluster scorer above is
correct now; batching is a later speed pass.

## MCMC (replaces arch moves)

Orbit **split-merge** (Jain-Neal motif, mirrors `par:arch-split-merge` but on
orbits), in `field_rate_discovery` / `field_rate_trainer`:

- **Merge**(b1,b2): union two orbits (subject to `|B|≤s_max`). Affected
  clusters = those with a column whose class ∈ `B_{b1}∪B_{b2}`. Rescore only
  those; accept on Δ(marginal LL) + DP prior term (merging drops #orbits by
  1 → factor from `α_orbit`).
- **Split**(b): partition a non-singleton orbit. Same affected-set logic.
- **Per-orbit rate**: reuse the `+Γ+I` field-rate update, now per orbit.
- `cn` (column→class/archetype) and `z` (cluster partition, discovery) moves
  are **unchanged**.
- Acceptance uses the same class-marginal / field-rate-marginal cluster
  evidence already in the loop; the affected-cluster delta (not full corpus
  LL) drives it — same optimization as the current `arch_mh` delta.

`α_orbit` high → mostly singletons → flips rare (matches ~9% salt-bridge
flip rate). It is the flip-capacity knob; `--alpha-orbit`, default high.

## Flip posterior (metrics)

`φ(C)` = posterior prob the coupled orbit's field is non-identity on some
edge, computable from the coupled pair forward (ratio of the field-moved mass
to total) — **identically 0** for singleton-orbit clusters (structural, not
emergent). Feeds `analysis/scripts/dynfield_metrics.py` (add an orbit-aware φ
that replaces the current rate-bin φ for orbit checkpoints). Charge-crossing
becomes trivially "does the orbit pair an acid and a base archetype" — report
it, and the null test should now clear.

## Checkpoint / entrypoint

- `save_checkpoint` / `load_checkpoint`: persist `orbit_id`, `rho_orbit`
  (ragged → store as concatenated + offsets), per-orbit rate bins, in place
  of `arch_assignment`. Bump a `model="orbit"` tag; keep back-compat load of
  free-arch checkpoints (reject with a clear message, or map diagonal→
  singletons).
- Entrypoint: add `--orbit-field` to `train_marginal_dynfield.py` (reuses
  discovery/supervised plumbing) + `--s-max` (default 2), `--alpha-orbit`.
  Everything else (`--site-rate-bins`, `--field-rate-bins`, `--discover`,
  `--resume`, `--ckpt-every-sec`, `--init-arch`) carries over. `--init-arch`
  becomes `--init-orbits` (transfer the orbit partition to another split).

## Staging + validation (each stage committed + build/test-gated)

- **Stage A — numpy reference + reductions.** `orbit_scorer` numpy path.
  Bit-exact reduction tests: (1) all-singleton orbits ⇒ every cluster static
  ⇒ LL matches the current model with `L_field=1` / invariant field; (2) a
  hand-built pair orbit `{a,b}` with a same-orbit pair cluster ⇒ LL matches a
  free-arch model with `L_field=2`, `arch=[[a,b],[b,a]]`, uniform `rho`
  (i.e. the orbit forward reproduces the existing forward on the matching
  2-field table). These two nail correctness before any batching.
- **Stage B — JAX batched scorer**, validated against Stage A to machine
  precision on a few families.
- **Stage C — split-merge MCMC** wired into discovery + supervised; smoke on
  ~8 families; confirm moves accept, φ(C)=0 on singletons, and a planted
  acid/base pair orbit gets merged when the data support it.
- **Stage D — full runs + A/B vs free-arch** (`--orbit-field` on GPU when
  free): does charge-crossing now clear the null? does φ separation hold?
  does discovery contact-enrichment / flip-recall improve? Auto-analyzed by
  the existing poller (`watch_and_analyze.py`).

## Files touched (checklist)

- `corpus_state.py` — `orbit_id`, per-orbit stationary/rate, checkpoint
  fields, affected-cluster helper for orbit moves.
- `orbit_scorer.py` (new) — per-cluster orbit forward (numpy + JAX), the
  three-case factorization.
- `field_rate_discovery.py` / `field_rate_trainer.py` — swap `arch_mh` for
  orbit split-merge; route scoring through `orbit_scorer`.
- `tree_batch.py` — positional (per-cluster) variant of the binned forward
  if the exact_cap2 batched path needs it.
- `train_marginal_dynfield.py` — `--orbit-field`, `--s-max`, `--alpha-orbit`,
  `--init-orbits`.
- `analysis/scripts/dynfield_metrics.py` — orbit-aware φ + charge-pairing.
- `tests/dynfield/` — Stage-A reduction tests.

## Open decisions (defaults chosen; flag if you disagree)

- `s_max = 2` (transpositions only) as the default; triplets (`S_3`, 6
  states) supported by the same code but off by default.
- `rho_orbit` uniform to start (swap ⇄ no-swap equal prior); learnable
  `p_b` is a later refinement.
- Keep `field_rate` (+Γ+I) as the per-orbit rate mechanism rather than a
  separate swap-rate parameter — reuses validated machinery; the invariant
  bin = static orbit.
