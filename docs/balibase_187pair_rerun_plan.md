# Workplan: rerunning the BAliBASE 187-pair evaluation

Target: regenerate `tab:aln` of `molbiolevol-paper/paper3-tkfdp/mbe_paper3.tex`
with the coupling supplied by paper 2's best-fitting pair model (the
Metropolis mixture, K=8) instead of the 2026-05 Potts checkpoints.

**Non-negotiable: all 187 pairs.** Every row of the table must be scored on
the same 22 families and 187 sequence pairs. A run that silently drops pairs
is a failed run, not a partial one. See "The coverage trap" below — this has
bitten the project once already.

Status when this plan was written (2026-09-24): steps 0 and 1 are unblocked
and need no new compute. Steps 2-4 need AWS. Step 5 is local.

---

## Background: what the current table rests on

All six rows were verified against committed artifacts on 2026-09-24:

| Row | Source | Verified |
|---|---|---|
| TKF92, TKF92-K20, MAFFT, MUSCLE | `math-paper/results/balibase_summary_l150.md`, FSA1 columns | exact |
| Potts K_c=4 | `~/tkf-mixdom/python/experiments/expected_balibase/expected_balibase_inf_phmm_K4_l150_withsps.json` | recomputed: F1 0.5093 / SP 0.7048 / TC 0.6092, 22 fam / 187 pairs |
| Potts K_c=8, no coupling | `math-paper/results/expected_balibase_tkf92_K8_tkfdp.json` (120 families) | L<150 subset recomputed: F1 0.5174 / SP 0.7290 / TC 0.6287, same 22 fam / 187 pairs |
| Potts K_c=8, coupled | run `v12-aws-k8-top8000`; values recorded in commit `31da1c9e`'s message | **no committed aggregate JSON** — step 0 below |

Coverage of the coupled K8 run is confirmed even though its aggregate is
missing: `s3://tkf-mixdom-gpu-<AWS_ACCOUNT>/balibase-runs/v12-aws-k8-top8000/`
holds 187 real per-pair Q' results plus 54 length-excluded stubs, and the
per-family pair counts match the K4 split family-for-family.

---

## Step 0 — Restitch the missing K8-coupled aggregate (no new compute)

Closes the one provenance gap in the current table. This is an aggregation of
results that already exist, not a rerun.

`analysis/scripts/downstream_fsa_on_cached_qprime.py` reads per-family Q'
arrays from `~/.cache/tkf-mixdom-balibase/<method>/<family>.npz`.

**Most of this is already on disk.** The local cache
`~/.cache/tkf-mixdom-balibase/infinite_phmm_mcmc_K8_coupled_RE/` holds all
**187 per-pair Q' arrays** (`BB*_i_j.npz`), matching the manifest family for
family — so the S3 sync is a fallback, not a prerequisite. What is missing is
the per-family stitch: only **20 of the 22** `BB*.npz` files exist, with
`BB11001` and `BB12041` absent. The coverage guard (below) reports exactly
this when pointed at the cache today.

So step 0 is: stitch the two missing families from their per-pair arrays, then
aggregate.

```bash
# Fallback only, if the local per-pair arrays turn out to be incomplete:
aws s3 sync s3://tkf-mixdom-gpu-<AWS_ACCOUNT>/balibase-runs/v12-aws-k8-top8000/qprime/ \
    ~/.cache/tkf-mixdom-balibase/infinite_phmm_mcmc_K8_coupled_RE/

python analysis/scripts/downstream_fsa_on_cached_qprime.py \
    --method infinite_phmm_mcmc_K8_coupled_RE \
    --params-key <params hash of the v12 run> \
    --out math-paper/results/downstream_fsa_K8coupled_187.json
```

Note that the cache mixes two layouts — 187 per-pair `BB*_i_j.npz` and 20
per-family `BB*.npz` — and the downstream script globs `*.npz` and reads each
stem as a family name. Stitching must produce the per-family files; it will
not read the per-pair ones as families.

The `--params-key` is the sweep's parameter hash; recover it from the v12
per-pair JSONs rather than guessing. Note that
`analysis/scripts/aggregate_infinite_phmm_results.py` takes no arguments and
reads a hard-coded path (`math-paper/results/infinite_phmm_balibase.json`), so
it is not the right tool here — the corpus aggregate written by
`downstream_fsa_on_cached_qprime.py` is what to read.

Acceptance: the output reports `n_families` 22 and 187 pairs summed over
`per_family`, and reproduces F1 0.5131 / SP 0.7301 / TC 0.6399 to three
decimals. If it does not reproduce, the table is wrong and the discrepancy
must be resolved before anything in steps 1-4 is worth starting.

Commit the aggregate JSON **and** the per-family Q' cache under
`math-paper/results/qprime_cache/infinite_phmm_mcmc_K8_coupled_RE/` — that is
the one method directory currently missing from the committed cache mirror.

## Step 1 — Implement the Metropolis mixture as a CouplingModel (no new compute)

The sampler consumes the coupling through exactly one interface. In
`src/tkfdp/mcmc_infinite_phmm.py:582`, `precompute_partial_forward` calls

```python
M_AA = boost_state.tkf_state.coupling.build_M_tensor(
    branch_length, pi_c=..., pair_background=...)
```

and everything downstream sees only the resulting (A,A,A,A) log-boost tensor
`M = P_doublet / (P_singlet x P_singlet)`. Any pair model that can produce a
four-residue joint emission at branch length t therefore drops in without
touching the MCMC.

Add `src/tkfdp/coupling/metropolis_mixture.py`, registered via the
`@register` decorator in `src/tkfdp/coupling/__init__.py`, implementing the
`CouplingModel` protocol: `build_singlet_emission`, `build_doublet_emission`,
`build_M_tensor`, `build_M_tensor_typed`, `to_npz`, `from_npz`. Follow
`src/tkfdp/coupling/potts.py` for structure.

**Parameters.** Paper 2's winning row (val LL -2.5316, MI_stat 0.089 +/- 0.070)
is `results/mixture_rateI_joint/components_K8.npz`:

- `pis` (8, 400) — per-class joint stationary over ordered residue pairs;
  state x = 20*i + j over the alphabet `ACDEFGHIKLMNPQRSTVWY`
- `S` (20, 20) — shared free exchangeability
- `weights` (8,) — mixture weights
- rate grid in the sibling `components_K8.json`: `rate_vals` /
  `rate_weights` (bin 0 = invariant), `alpha` 0.78, `p_inv` 0.0038

Per-class generator: `Q_c = metropolis_sqrt(S, pi_c)`, i.e.
`Q_c[(i,j) -> (k,l)] = S[i,k] * sqrt(pi_c(k,l) / pi_c(i,j))` on
single-residue transitions (exactly one of i->k, j->l), 0 off the
single-substitution support, diagonal set to make rows sum to zero. Helper:
`experiments/fit_pair_models._met_Q(S, pi_c.reshape(20,20), "sqrt")`.

The doublet emission at branch length t is the rate-and-class mixture

```
P_doublet(a,b; c,d; t) = sum_c w_c sum_r rate_weight[r]
                         pi_c(a,b) expm(rate_val[r] * t * Q_c)[(a,b),(c,d)]
```

with the singlet denominator built on the same footing so that the boost is 1
when the coupling is switched off.

Note: `results/pair_models/baseline_rateI/MANIFEST.md` points at
`mixture/components_K8.npz`, which is not a real path — the released tree only
carries the baselines and the transposed mixtures. The symmetric K=8 mixture
lives at `results/mixture_rateI_joint/` as above. Do not chase the manifest
path; it is a stale reference in a tagged release and should not be edited in
place.

**Gates before any AWS spend:**

1. Setting all mixture weight on a single class with a product stationary
   must give M identically 1 (no boost).
2. At t = 0 and x = y, `M_obs` must equal `M_solo` — the identity that caught
   the gamma-weighted-denominator relic documented at
   `mcmc_infinite_phmm.py:560`.
3. Row sums of each `Q_c` are zero to 1e-12; `pi_c Q_c` is a valid
   reversible flux (detailed balance to 1e-10).
4. The exact cap-2 Felsenstein path
   (`src/tkfdp/coupling/dynfield/phylo_elbo/exact_cap2_jax.py`) and the new
   `build_doublet_emission` must agree on a small tree to 1e-6.

## Step 2 — Calibrate on a single family before the sweep

`analysis/scripts/calibrate_infinite_phmm.py` profiles the sampler on the
smallest BAliBASE pair at three length budgets (L ~ 90, 120, 145). It takes no
arguments and currently builds a K_c=1 stub Potts state; point it at the new
coupling and run it on BB11001 (4 sequences, 6 pairs). Confirm acceptance
rates and replica-exchange swap rates are in the same range as the Potts runs
and that peak GPU memory still fits at L = 145. A
mixture boost has a different dynamic range than a Potts boost, so the
alpha_z ladder may need retuning; that is cheap to discover here and
expensive to discover at pair 150 of 187.

## Step 3 — AWS sweep, all 187 pairs

Match the sampler configuration already reported in `sec:balibase` so the new
row is comparable to the existing ones:

- replica exchange over six alpha_z values, 100 to 5,000
- swap proposed every 10 sweeps
- 8,000 sweeps after 2,000 burn-in
- two independent replicates per pair

`analysis/scripts/sweep_infinite_phmm_balibase.py` currently takes the model
as `--checkpoint`, a Potts state loaded by its `build_k4_state` helper. Step 1
must therefore also add a way to supply a `CouplingModel` directly — a
`--coupling-npz` / `--coupling-variant` pair that constructs the state through
the `VARIANTS` registry instead of `build_k4_state`. The existing sampler
flags below are real and should be reused unchanged:

```bash
python analysis/scripts/sweep_infinite_phmm_balibase.py \
    --coupling-variant metropolis_mixture \
    --coupling-npz results/mixture_rateI_joint/components_K8.npz \
    --max-len 150 --n-re-replicates 2 \
    --n-sweeps 8000 --n-burnin 2000 \
    --alpha-z-ladder 100,250,600,1500,3000,5000 --swap-every 10 \
    --label metropmix_K8_RE \
    --out math-paper/results/infinite_phmm_balibase_metropmix_K8.json
```

(`--coupling-variant` and `--coupling-npz` do not exist yet; they are part of
step 1's deliverable. Everything else is an existing flag.) Upload to
`s3://tkf-mixdom-gpu-<AWS_ACCOUNT>/balibase-runs/v15-metropmix-k8/` with the
same layout the v12 run used, so step 4 can restitch identically.

Run under A1 / reversible mode, which has been the sampler default since
2026-06-27 — the new row should be a balanced-model result, not a legacy one.

**Launch-time preflight.** Before committing GPU hours, assert that the
launcher enumerates exactly 187 pairs across 22 families and fail loudly
otherwise. The 2026-05 launcher under-counted `n_seqs` on BB12014, BB20008
and BB30015 and needed a 31-pair recovery batch (recorded in
`results/H_supervised_pdb_rnn_K8_2026-05-31/balibase_aggregate.json`). Do not
rediscover this at the end of the run.

## Step 4 — Aggregate, commit, and update the table

```bash
aws s3 sync s3://tkf-mixdom-gpu-<AWS_ACCOUNT>/balibase-runs/v15-metropmix-k8/qprime/ \
    ~/.cache/tkf-mixdom-balibase/infinite_phmm_mcmc_metropmix_K8_RE/
python analysis/scripts/downstream_fsa_on_cached_qprime.py \
    --method infinite_phmm_mcmc_metropmix_K8_RE \
    --params-key <params hash of the v15 run> \
    --out math-paper/results/downstream_fsa_metropmix_K8_187.json
```

The coverage guard runs automatically and refuses to write a short result, so
a table row cannot be produced from an incomplete stitch. Commit the aggregate
plus the per-family Q' cache, then update `tab:aln` and the provenance comment
beneath it in `mbe_paper3.tex`.

Report F1, SP and TC against the existing K_c=8 Potts row: that is the
comparison the paper is making — same sampler, same pair set, same merge, a
better-fitting coupling.

## Step 5 — Regenerate the two eight-class figure caches (local)

Independent of the alignment table. `fig:lama1` and `fig:pf00053` are drawn
from the eight-class run of commit `00c4025`, whose caches were never
committed; the committed caches are the four-class run of 2026-05-15. The
paper currently quotes the four-class enrichment ratios (17x / 12x / 5x) and
says so in the text, and quotes the eight-class pooled numbers (top 41,
0.087 vs 0.004) in the eight-class paragraph.

Rerunning the tile and triangle analyses under the eight-class model and
committing the caches would let both paragraphs quote the same model. This is
a small local job with no AWS component and no dependency on steps 0-4.

---

## The coverage trap

Commit `72055202` re-stitched the K8-coupled posteriors down to MixFrag's
155-pair coverage and got SP 0.6622, against 0.7301 on the full 187. That
0.068 gap is pure pair selection: BB20008, for instance, contributes
8 sequences / 28 pairs at full coverage but 4 sequences / 6 pairs under the
L<150 filter, and mixing the two conventions across rows silently penalises
whichever row has the harder subset.

The lesson is not "prefer 187" but "state and assert the coverage on every
row". Any comparison table that does not carry its pair count per row is one
refactor away from this bug.

### The guard

The expected set is written down as data in
`analysis/balibase_l150_coverage.json` — 22 families with their `n_seqs` and
`n_pairs` — rather than re-derived and trusted each run. It was built from
four independent sources that agree exactly: a live scan of the BAliBASE
FASTAs, the K4 result JSON, the K8 no-coupling result JSON, and the v12 S3
archive.

`analysis/scripts/balibase_coverage.py` compares against it. Both entry points
are on by default and fatal:

- **Preflight**, in `sweep_infinite_phmm_balibase.py`, right after
  `find_eligible_families()` and before any GPU work. It validates the *full*
  enumeration even when `--fam-subset` narrows the run, so a single-pair AWS
  worker still catches a mis-enumerated corpus. Auto-skips when `--max-len`
  differs from the manifest.
- **Write gate**, in `downstream_fsa_on_cached_qprime.py`: an early check that
  the cache holds every expected family (fails before the FSA work), and a
  final check that refuses to write the result JSON unless per-family pair
  counts match. Auto-skips when `--families` restricts the set.

Both take `--no-coverage-check` for genuinely different corpora. Comparing
per-family counts rather than a bare total is deliberate: it catches the case
where the total is right but the families are wrong.

The guard found a real gap on its first run against live data — the local
`infinite_phmm_mcmc_K8_coupled_RE` cache has per-family stitches for 20 of 22
families. That state had been sitting there unnoticed.

`balibase_coverage.py` also runs standalone against any result JSON, handling
both the `n_pairs` and the `per_pair`-list schemas:

```bash
python analysis/scripts/balibase_coverage.py <result>.json
```

On the known-short 155-pair file it reports the exact shortfall:
`BB11021 2/3, BB12014 10/15, BB20008 6/28, BB30015 6/10`.

---

## Dependencies and estimated effort

| Step | Blocked by | Compute | Effort |
|---|---|---|---|
| 0 Restitch K8 | none | none (S3 download + ~20s) | hours |
| 1 CouplingModel | none | none | 1-2 days incl. gates |
| 2 Calibrate | 1 | 1 GPU, 1 family | hours |
| 3 AWS sweep | 2 | comparable to v12 (187 pairs x 2 replicates x 6 rungs) | 1-2 days wall |
| 4 Aggregate + table | 3 | none | hours |
| 5 K8 figure caches | none | local GPU | hours |

Steps 0, 1 and 5 can proceed in parallel and none of them need AWS.
