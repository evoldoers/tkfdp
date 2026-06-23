# Infinite-Pair-HMM BAliBASE benchmark (AWS fan-out)

This benchmark runs the infinite-Pair-HMM O(L⁴) MCMC sampler on
**187 BAliBASE pairwise comparisons** to produce the per-pair
posterior-F1 sufficient statistics for Table 1 of the
column-paired-FSA paper.  The 187 pairs come from 22 BAliBASE
`bali3pdbm/in` families filtered to `max_seq_len < 150`; each
family contributes C(n, 2) ordered pairs.  Total compute is
fanned out via SkyPilot across multi-region on-demand GPU
instances (g5 / g6 / g4dn xlarge) with a hard concurrency cap.

## What the sampler does

For each pair (anc, des) and a fixed checkpoint of an emission
model, the sampler runs an MCMC chain over alignments under the
infinite Pair HMM (or under a bounded-ε prior that anneals toward
finite Pair HMM as `α_z → ∞`).  The chain produces:

* a posterior `Q'(a, b)` over column pairs (i.e. probability that
  ancestor position `a` and descendant position `b` end up in the
  same alignment column),
* per-chain ESS / Geyer-IACT diagnostics on `n_match` and `log π`,
* Gelman-Rubin r̂ across chains (multi-chain mode only),
* acceptance rates per move type,
* `q_l1_vs_baseline = ‖Q' − Q_baseline‖₁ / n_cells` against the
  trained TKF-DP FB posterior baseline.

Three operating modes (see
`analysis/scripts/sweep_infinite_phmm_balibase.py` docstring):

1. **Single-chain** — one chain at fixed `α_z`.
2. **Multi-chain** — N independent chains at fixed `α_z`; cold-rung
   `Q'` is the across-chain mean; r̂ is well-defined.
3. **Replica exchange** — K chains on an `α_z` ladder with adjacent-rung
   swap proposals every `--swap-every` sweeps.  Cold rung
   = `min(α_z_ladder)`.  r̂ is meaningless across rungs (different
   targets), so per-rung ESS is reported per chain.

`--top-rung-only` pins `|E| = 0` via the bounded-ε prior at very
large `α_z`, reducing the sampler to pure-Gibbs alignment
resampling under the trained TKF92 — used as a convergence-validation
switch (cold `Q'` should match the baseline FB posterior to MCMC
noise).

## Data — 22 BAliBASE families, 187 pairs

From `aws/launch_balibase_sweep.py:FAMILIES`:

| Family  | n | pairs | | Family  | n | pairs | | Family  | n | pairs |
|---------|---|-------|-|---------|---|-------|-|---------|---|-------|
| BB11001 | 6 | 15    | | BB12021 | 5 | 10    | | BB20033 | 4 | 6     |
| BB11013 | 5 | 10    | | BB12032 | 5 | 10    | | BB20038 | 4 | 6     |
| BB11021 | 4 | 6     | | BB12041 | 3 | 3     | | BB30015 | 4 | 6     |
| BB11029 | 4 | 6     | | BB20001 | 5 | 10    | | BB30022 | 4 | 6     |
| BB11035 | 4 | 6     | | BB20008 | 4 | 6     | | BB30025 | 6 | 15    |
| BB12014 | 5 | 10    | | BB20015 | 6 | 15    | | BB40018 | 4 | 6     |
|         |   |       | | BB20030 | 4 | 6     | | BB40029 | 9 | 36    |
|         |   |       | |         |   |       | | BB40038 | 4 | 6     |
|         |   |       | |         |   |       | | BB40045 | 5 | 10    |

**Total: 22 families × C(n, 2) = 187 pairs.**  Eligibility filter is
`max_seq_len < 150`; the BAliBASE 3 `bali3pdbm/in` corpus is in
`~/bio-datasets/data/balibase/bench1.0/`, fetched via
`~/bio-datasets/fetch/balibase/fetch.py` (drive5 mirror, ~21 MB,
fix landed 2026-05-09).

## Released checkpoint (what the sampler conditions on)

GitHub release tag on `${REPO_OWNER}/tkfdp`:

* `results/K4-emwarm-top1000-2026-05-09` — K=4, best val_LL = −298.41
  at outer iter 55.  Default checkpoint in
  `aws/balibase_one_pair_v2.yaml` env `CHECKPOINT`.
* `results/K4-KH4-emwarm-top1000-2026-05-15` — K=4 with KH=4 mixture.
* (Upcoming) `results/K8-KH1-top8000-2026-05-22` — K=8 with KH=1.

Pull a checkpoint with:
```
gh release download <tag> --repo ${REPO_OWNER}/tkfdp -p '*.tar.gz'
```
The release tarball contains `_best_chkpt/state.npz`,
`_best_chkpt/meta.json`, `_best_chkpt/trace.json`.

## AWS fan-out

**Launcher**: `aws/launch_balibase_sweep.py` — submits 187 jobs as
individual `sky jobs launch` calls parameterised by the
`(FAMILY, PAIR_I, PAIR_J)` env triple.  Concurrency cap controlled
by `--concurrency`; default 16.  Spot-evicted jobs are retried up
to `MAX_RETRIES`.

**Worker YAML**: `aws/balibase_one_pair_v2.yaml`
* `resources.any_of` — 4 regions (us-east-1, us-east-2, us-west-2,
  eu-west-1) × {g5.xlarge A10G:1, g6.xlarge L4:1, g4dn.xlarge T4:1}
  = 10 placement options, all on-demand (`use_spot: false`).
* `disk_size: 64`.  No `workdir` / `file_mounts` (eliminates the
  AWS `CreateBucket` concurrent-upload bottleneck that broke the
  v1 launcher at ≥96-way fan-out).  All inputs come from S3.

**Per-pair env defaults**:
```
S3_BUCKET=tkf-mixdom-gpu-<AWS_ACCOUNT>
BUNDLE_PREFIX=balibase-bundle
RESULTS_PREFIX=balibase-runs/v11-aws-canonical
JIT_CACHE_KEY=jax-cache/g5xl-jax-2026-05-15.tar.gz
CHECKPOINT=results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz
N_SWEEPS=8000
N_BURNIN=2000
SEED=0
OUT_SUFFIX=''
N_REPLICATES=2
```

**S3 bundle** (uploaded 2026-05-15, total ~27 MiB):
```
s3://tkf-mixdom-gpu-<AWS_ACCOUNT>/balibase-bundle/
  ├─ bali3pdbm.tar.gz   ~260 KiB  (BAliBASE 3 + fetcher outputs)
  ├─ tkfmixdom.tar.gz   ~1.6 MiB  (tkfmixdom JAX package)
  └─ workdir.tar.gz     ~25 MiB  (tkf-dp workdir slice: src/, analysis/scripts/, configs)
s3://tkf-mixdom-gpu-<AWS_ACCOUNT>/jax-cache/g5xl-jax-2026-05-15.tar.gz
  ~1.4 MiB persistent JIT cache pre-populated by aws/balibase_jit_primer.yaml
s3://tkf-mixdom-gpu-<AWS_ACCOUNT>/balibase-runs/v11-aws-canonical/
  per-pair JSON results land here
```

**Worker setup** (from `balibase_one_pair_v2.yaml setup`):
1. `apt install python3-pip python3-venv`
2. `python3 -m venv ~/venv && pip install awscli`
3. `aws s3 cp` the three bundles and the JIT cache
4. `tar xzf` and reshape into `~/tkf-mixdom/python/` and
   `~/sky_workdir/`
5. `pip install -e ~/tkf-mixdom/python` + `pip install "jax[cuda12]"`
6. Set `JAX_COMPILATION_CACHE_DIR=~/.cache/jax`,
   `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1`,
   `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` — make sure the
   primed JIT cache is actually consumed.

**Worker run** (per pair):
```
cd ~/sky_workdir
timeout 28000 python analysis/scripts/sweep_infinite_phmm_balibase.py \
    --bali3pdbm-only --family $FAMILY --pair-i $PAIR_I --pair-j $PAIR_J \
    --checkpoint $CHECKPOINT --n-sweeps $N_SWEEPS --n-burnin $N_BURNIN \
    --seed $SEED --n-replicates $N_REPLICATES \
    --out /tmp/${FAMILY}_${PAIR_I}_${PAIR_J}${OUT_SUFFIX}.json
aws s3 cp /tmp/${FAMILY}_${PAIR_I}_${PAIR_J}${OUT_SUFFIX}.json \
    s3://${S3_BUCKET}/${RESULTS_PREFIX}/
```

**Merge**: `aws/merge_balibase_results.py` pulls all per-pair JSONs
from `s3://${S3_BUCKET}/${RESULTS_PREFIX}/` and produces the
aggregated table.

## Cost (last observed)

* **K=4 on-demand g5/g6/g4dn**, 16-way concurrency, N=1 replicate
  per pair: ~$30 wall-clock (about 1–2 hours of total compute on
  ~20 GPU-hours).
* Per-pair runtime varies sharply with `n_seqs` and sequence length;
  BB40029 (n=9 → 36 pairs) dominates the wall clock.

## Prerequisites (assuming a fresh machine that has
`~/tkf-dp`, `~/tkf-mixdom`, `~/.aws`)

1. **`sky` CLI**: `uv sync` inside `~/tkf-mixdom/python` (the venv
   has SkyPilot pinned), or `pip install "skypilot[aws]"` directly.
2. **`aws` CLI**: `snap install aws-cli` or apt; the launcher uses
   `aws s3 ls/cp` on `${S3_BUCKET}` and parses outputs.
3. **Verify identity**:
   `aws --profile tkf-gpu sts get-caller-identity` →
   `arn:aws:iam::<AWS_ACCOUNT>:user/claude-orchestrator`.
4. **Verify Sky connectivity**:
   `~/tkf-mixdom/python/.venv/bin/sky check` → must show `AWS: enabled`.
5. **Dry-run**:
   `python aws/launch_balibase_sweep.py --dry-run` → prints all 187
   pairs and the `sky jobs launch` commands without spending.

## Reproducing

```bash
cd ~/tkf-dp
# Sanity-check the pair list:
python aws/launch_balibase_sweep.py --dry-run | head -40

# Actually launch (16-way fan-out, default checkpoint K=4):
python aws/launch_balibase_sweep.py --concurrency 16

# Monitor:
~/tkf-mixdom/python/.venv/bin/sky jobs queue --all | tail -30

# When all jobs land, merge:
python aws/merge_balibase_results.py
```

To re-launch against a different checkpoint (e.g. K8-KH1), update
the YAML's `envs.CHECKPOINT` to the K8-KH1 path inside the bundle,
**re-tar and re-upload the bundle** to include the new
`_best_chkpt/state.npz`, then launch.  Alternative: add a
`gh release download` step to `aws/balibase_one_pair_v2.yaml` setup
so the worker pulls the checkpoint fresh from the GH release.

## Known caveats

* JIT cache `jax-cache/g5xl-jax-2026-05-15.tar.gz` is keyed to a
  specific JAX version / GPU type.  If JAX is upgraded or the
  resource pool shifts away from g5.xlarge, re-prime with
  `aws/balibase_jit_primer.yaml`.
* The 27 MiB S3 bundle is from a particular tkf-dp commit; if the
  worker needs newer code, re-tar `~/tkf-dp` (only `src/`,
  `analysis/scripts/`, plus any new helpers) and upload.
* Spot was disabled in v2 because `use_spot: true` led to too many
  evictions on the long BB40029 jobs; on-demand is the default.
  If wall-clock pressure justifies the eviction risk, flip
  `use_spot: true` in the YAML.
* `n_replicates` of 2 is the default and matches the paper's main
  numbers.  Single-replicate runs are cheaper (~$15) but tighter
  ESS/r̂ require ≥2.
