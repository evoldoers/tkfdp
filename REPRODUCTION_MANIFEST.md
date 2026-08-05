# TKF-DP reproduction manifest

bioRxiv preprint ID: **BIORXIV/2026/725674**.

This manifest describes the public **drop** served at
[tkfdp.net](https://tkfdp.net) and how to reproduce the models and
evaluations behind it. The published, authoritative spec is
**`math-paper/supplement.pdf`** (built from `supplement.tex` + the
`appendix-*.tex` chain). The two PSB write-ups (`paper1-baumwelch`,
`paper2-coupled`) live in the dev repo under `psb-paper/` but are
**excluded** from the drop.

---

## What the drop contains

Three top-level repos (one is included twice — as a working clone and as a
git submodule of `tkf-dp` so the supplement can resolve
`\input{tkf-mixdom/tkf/...}`):

```
tkfdp.net/
├── tkf-dp/          ← supplement source (math-paper/), coupling code (src/tkfdp/),
│   │                  training/eval entrypoints (experiments/), analysis notes
│   └── math-paper/tkf-mixdom/   ← git submodule → the tkf-mixdom repo below
├── tkf-mixdom/      ← JAX inference/training library (TKF91 / TKF92 / MixFrag / MixDom)
└── bio-datasets/    ← dataset fetch + preprocessing scripts (no data committed)
```

The submodule pin is advanced to the shipped `tkf-mixdom` commit on every
publish (`scripts/publish_drop.sh` re-pins it). Trained parameters are **not**
committed to the trees; they are attached to GitHub releases (below).

**Not in the drop:** `psb-paper/` (the paper working area), the maintainer
publish scripts (`scripts/publish_drop.sh`, `scripts/stage_clean_drop.sh`),
deployment infra (`aws/`, S3 sync/upload scripts), `CLAUDE.md`, build
intermediates, and large caches. AWS account/bucket identifiers are scrubbed
to `<AWS_ACCOUNT>` in the staged copy.

---

## Released parameters and checkpoints

Substitute your own owner when forking:
```bash
RELEASE_OWNER=evoldoers
```
Download with, e.g.:
```bash
gh release download <tag> --repo ${RELEASE_OWNER}/<repo> -p '*.npz' -p '*.tar.gz'
```

### `${RELEASE_OWNER}/tkf-mixdom`
| Tag | Contents |
|---|---|
| `results/mixdom-d3f1-perdomain-2026-08-04` | **MixDom-d3f1** (per-domain substitution matrices), Pfam v3 — current headline substitution-side model |
| `results/mixfrag-F2-pfam-2026-08-04` | **MixFrag F=2** cherry-EM fit (TKF92 + 2 fragtypes) |
| `mixdom-checkpoints-2026-05` | **TKF92-K=20**, **CherryML-C=20** mixture checkpoints (+ the earlier MixDom-d3f1) |

### `${RELEASE_OWNER}/tkfdp`
| Tag | Contents |
|---|---|
| `code-2026-08-04` | Source-code snapshot |
| `results/paper2-coupling-rates-2026-08-04` | Paper-2 coupled amino-acid substitution rates: Exchangeable, Metropolis, Coupled, Metropolis-mixture (+Γ+I) |
| `results/K4-emwarm-top1000-2026-05-09` | TKF-DP K꜀=4 EM-warmup checkpoint (Pfam top-1000); backs the infinite-pair-HMM K=4 results |

> The **per-domain MixDom-d3f1** checkpoint (`svibw_d3f1_perdomain_v3`, released
> as `results/mixdom-d3f1-perdomain-2026-08-04`) is the current headline
> substitution-side model: each domain gets its own substitution matrix; 540
> svi-BW iterations on the v3 corpus.

---

## Reproduction recipe

### 1. Clone + environment
```bash
git clone --recurse-submodules git@github.com:${RELEASE_OWNER}/tkfdp.git
git clone git@github.com:${RELEASE_OWNER}/tkf-mixdom.git
git clone git@github.com:${RELEASE_OWNER}/bio-datasets.git
```
`tkf-mixdom/python/pyproject.toml` is the canonical env (JAX, optax, etc.);
`tkf-dp`'s `src/` layout is imported from the same venv. Install JAX with the
CUDA build for GPU training. MUSCLE 5 + MAFFT 7 (via
`tkf-dp/scripts/fetch_aligners.sh`) are the alignment comparators.

### 2. Data
```bash
python bio-datasets/fetch/pfam/fetch.py          # Pfam (precompiled_v3 shards)
python bio-datasets/fetch/balibase/fetch.py      # BAliBASE 3 (bali3pdbm, drive5 mirror)
```

### 3. Fit the indel/substitution models
- **MixFrag F=2 and TKF92** (cherry-EM / Maraschino), on the v3 corpus:
  `tkf-mixdom/python/experiments/ordsum_val/mixfrag_from_shards.py`.
- **MixDom-d3f1 (per-domain)** by stochastic variational Baum–Welch:
  `tkf-mixdom/python/train_pfam.py --svi-bw --n-dom 3 --n-frag 1 --n-classes 3
  --classdist-init identity --freeze-classdist` (the diagonal frozen classdist
  gives each domain its own substitution matrix). Batched-emission E-step.
- **TKF92-K=20 / CherryML-C=20** mixture baselines: the corresponding
  `distill`/mixture fitters in `tkf-mixdom`.
- **Paper-2 coupled substitution models** (Exchangeable / Metropolis / Coupled /
  Metropolis-mixture): `tkf-dp/src/tkfdp/coupling/`.

### 4. Evaluate
- **Held-out Pfam likelihood** (path-conditioned *and* order-summed), full v3
  validation split: `tkf-mixdom/python/experiments/ordsum_val/score_matchaligned_val.py`.
- **BAliBASE alignment accuracy** (SP / TC / column-F1) via the FSA aligner:
  `tkf-mixdom/python/experiments/expected_pairwise_balibase.py`
  (and `tkf-dp/experiments/eval_balibase.py` for the multi-aligner harness).

### 5. Build the supplement
```bash
cd tkfdp/math-paper && ./build.sh --supp     # → supplement.pdf
```

---

## Publishing (maintainers)

`scripts/publish_drop.sh` stages the three dev trees (applying the exclude
list + credential/account scrub + `ihh/… → ${REPO_OWNER}/…` URL
parameterisation via `scripts/stage_clean_drop.sh`), pushes the three
`evoldoers/*` repos additively, re-pins the `tkfdp` → `tkf-mixdom` submodule,
build-verifies `supplement.pdf`, and refreshes `tkfdp.net`. Pass `--no-push`
to prepare + build-verify without pushing.
