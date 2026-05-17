# BAliBASE paired-FSA quickstart

Audience: a new agent (or human) who wants to reproduce the
column-paired FSA evaluation on the BAliBASE 3 benchmark using the
released TKF-DP checkpoint.

## What you're running

The TKF-DP postprocessing has two pathways for incorporating Potts
coevolutionary signal into a baseline pair-HMM alignment posterior; both
live in this repo and both are evaluated by the same eval harness.

1. **Pre-correction** (`src/tkfdp/postprocessing.py`).
   For each pair-HMM match-posterior `Q[i, j]`, fold the Potts boost
   into the Match-state log-emission table and re-run the pair-HMM
   forward--backward; downstream FSA assembly is unchanged.
   See `main.tex` Section "Pairwise Alignment Postprocessing".

2. **Coupled-pair greedy ("column-paired FSA")**
   (`src/tkfdp/coupled_annealing.py`).
   Replace FSA's `sequence_annealing` with one that admits **pairs**
   of column merges as candidates, scoring them with the four-residue
   Potts boost tensor `M(i, j; i', j'; t)` *during* greedy assembly.
   Pre-correction is bypassed in this mode -- the boost is consulted
   on its native four-residue domain.
   See `main.tex` subsection "Sequence annealing with coevolutionary
   scoring" and `analysis/postprocess_vs_coupled_annealing.md`.

The two are duals: pre-correction is systematic-but-linearized, coupled
is exact-but-greedy. Run both, score against BAliBASE references, and
compare.

## Prerequisites

### Code

```bash
git clone git@github.com:${REPO_OWNER}/tkfdp.git
cd tkf-dp
git checkout 'results/K4-emwarm-top1000-2026-05-09'   # the released code state
```

### Released checkpoint

Pull the trained K=4 checkpoint that `eval_balibase.py` consumes by
default:

```bash
gh release download \
    'results/K4-emwarm-top1000-2026-05-09' \
    --repo ${REPO_OWNER}/tkfdp \
    -p 'K4-emwarm-top1000-2026-05-09.tar.gz'
tar xzf K4-emwarm-top1000-2026-05-09.tar.gz
mkdir -p results/exp2_v2_K4_top1000_tsb_emwarm/_best_chkpt
cp -r K4-emwarm-top1000-2026-05-09/_best_chkpt/* \
      results/exp2_v2_K4_top1000_tsb_emwarm/_best_chkpt/
```

The `_best_chkpt/` is just three files (`state.npz`, `meta.json`,
`trace.json`); see `MANIFEST.md` inside the tarball for the exact run
configuration that produced it. Best val_LL = -298.41 at outer iter 55.

### BAliBASE benchmark data

Use the `bio-datasets` fetcher (drive5 mirror -- 21 MB, includes
BAliBASE 3 + PREFAB v4 + OXBENCH + SABRE):

```bash
python ~/bio-datasets/fetch/balibase/fetch.py
ln -sfn ~/bio-datasets/data/balibase data/balibase_full   # optional convenience
```

The eval harness reads from `~/bio-datasets/data/balibase/bench1.0/`
by default; override with `--bali-root` or `BIO_DATASETS_HOME`.

### MUSCLE + MAFFT (sanity comparators)

```bash
bash scripts/fetch_aligners.sh
```

Installs static binaries to `~/.local/bin/{muscle,mafft}`. Idempotent.

### tkf-mixdom (the upstream FSA + scoring)

The eval harness imports `compute_pairwise_posteriors`,
`sequence_annealing`, `sp_score`, and `tc_score` from
`~/tkf-mixdom/python/tkfmixdom/jax/...`; clone or symlink that repo
to `~/tkf-mixdom`. The harness adds it to `sys.path` automatically.

## Run the eval

### Pre-correction + baseline + MUSCLE + MAFFT (the default harness)

```bash
PYTHONPATH=src python3 experiments/eval_balibase.py \
    --bench bali3 \
    --strict-core \
    --checkpoints K4:results/exp2_v2_K4_top1000_tsb_emwarm/_best_chkpt \
    --methods baseline_fsa tkfdp_precorr muscle mafft \
    --out-dir results/balibase_eval
```

Writes per-`(benchmark, method, model)` rows to
`results/balibase_eval/bali3_results.csv`. Prints a summary table at
the end. **Idempotent**: re-running with the same args reads the CSV
and only fills missing or errored cells; previous successful cells
are skipped. Append-as-we-go, so partial progress survives a crash.

### Column-paired FSA (the coupled-annealing variant)

The coupled-annealing variant is not yet wired into the main
`eval_balibase.py` runner -- it lives in
`tests/test_balibase_coupled_annealing.py` (a three-way comparison
script that runs baseline + pre-correction + coupled on a single
benchmark, BB11001):

```bash
PYTHONPATH=src python3 tests/test_balibase_coupled_annealing.py
```

For a full BB3 sweep with the coupled variant, factor `run_coupled()`
out of that test file into `experiments/eval_balibase.py` (the
`run_tkfdp_coupled` placeholder is already there) and add it to
`--methods`. The coupled annealer's hyperparams `q_min` (default 0.1
posterior threshold) and `mu_min` (default 0.1 nat boost threshold)
need to be exposed as CLI flags first; the smoke test in
`tests/test_balibase_coupled_annealing.py` shows that aggressive
prunings (`mu_min < 0.5`) hurt SP on low-coevolution families.

### Cross-checking against MUSCLE/MAFFT

If your `baseline_fsa` SP scores diverge from MUSCLE/MAFFT by more than
a few percent on aggregate, you have an infrastructure regression
(scoring, sequence loader, JIT cache, etc.) -- not a model issue. The
two well-tuned production aligners agree to within ~0.003 SP on
BB11001, which is the cross-aligner sanity bound to expect.

## Output and results

| Path | What |
|---|---|
| `results/balibase_eval/bali3_results.csv` | One row per `(benchmark, method, model)`; columns `sp`, `tc`, `seconds`, `n_seqs`, `error` |
| `results/balibase_eval/<bench>_results.csv` | If you sweep PREFAB / OXBENCH / SABRE too |

Summary table printed at end-of-run:
- mean SP, mean TC, median SP per `(method, model)`
- count of finite-score cells + count of failures
- failures with non-empty `error` get retried automatically on the
  next invocation (idempotent gap-fill).

## Troubleshooting

### `gh release download` / `gh release create` picks the wrong repo

Always pass `--repo ${REPO_OWNER}/tkfdp` -- gh's auto-detection of the upstream
repo walks the working directory tree and can pick a sibling repo
(e.g. `~/bio-datasets`) if the cwd is ambiguous. The `--repo` flag
overrides.

### "ModuleNotFoundError: No module named 'tkfmixdom'"

`~/tkf-mixdom/python/` is not on `sys.path`. The eval harness adds it
via `sys.path.insert(0, "~/tkf-mixdom/python")` -- if your
`~/tkf-mixdom` is elsewhere, edit `TKFMIXDOM_ROOT` in the harness.

### "muscle: command not found" / "mafft: command not found"

`~/.local/bin/` is not on PATH or the binaries are missing. Run
`bash scripts/fetch_aligners.sh` (idempotent). For MAFFT specifically,
the wrapper at `~/.local/bin/mafft` exec's
`~/.local/share/mafft/mafft`; verify both exist.

### CUDA-noisy startup

The harness logs a CUDA-init warning when no GPU is available; this is
a harmless `xla_bridge` complaint and falls back to CPU automatically.
Set `CUDA_VISIBLE_DEVICES=""` to suppress GPU detection entirely if
you want a clean log.

## Adding a new trained checkpoint to the eval

When a new training run finishes (e.g. K=8/2000-fam):

1. Stage the checkpoint + log + a `MANIFEST.md` describing the run
   into `/tmp/release_artifacts/<tag-name>/`.
2. Tag the current code state:
   ```bash
   git -c user.email=${MAINTAINER_EMAIL} -c user.name="Ian Holmes" \
       tag -a 'results/<tag-name>' -m '<one-line summary>'
   git push origin 'results/<tag-name>'
   ```
3. Build the tarball (`tar czf ... -C /tmp/release_artifacts <tag-name>`).
4. Attach via the GitHub release:
   ```bash
   gh release create 'results/<tag-name>' \
       /tmp/release_artifacts/<tag-name>.tar.gz \
       --repo ${REPO_OWNER}/tkfdp \
       --title '<title>' \
       --notes-file /tmp/release_artifacts/<tag-name>/MANIFEST.md
   ```
5. Add the new checkpoint to the eval invocation:
   ```bash
   --checkpoints K4:results/exp2_v2_K4_top1000_tsb_emwarm/_best_chkpt \
                 K8:results/exp2_v2_K8_top2000/_best_chkpt
   ```
   The eval harness puts the model tag in the `model` column of the CSV
   and runs all model-dependent methods (`tkfdp_precorr`, eventually
   `tkfdp_coupled`) once per checkpoint.
