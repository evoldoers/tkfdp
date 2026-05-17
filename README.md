# TKF-DP

Cherry-trained, indel-aware Potts coevolution model — reference
implementation for the paper:

> **Maraschino: TKF-DP** (Holmes & Large)
> bioRxiv preprint: **[BIORXIV/2026/725674](https://www.biorxiv.org/)**
> *(awaiting bioRxiv screening, link will resolve once the
> preprint is posted)*

## Repository

- `math-paper/` — LaTeX source for the manuscript and supplement.
  Build with `cd math-paper && bash build.sh`.
- `src/tkfdp/` — Python reference implementation (TKF-DP MCMC sampler,
  Potts-atom DP, EM warmup).
- `analysis/scripts/` — figure-generation scripts and the recipe for
  reproducing every cell of Tables 1 and 2.
- `aws/` — SkyPilot YAMLs for large sweeps on cloud GPUs.
- `math-paper/tkf-mixdom/` *(git submodule)* — JAX implementation of
  TKF91/TKF92 / MixDom / mixture-class models from the companion paper.
- `bio-datasets/` *(git submodule)* — fetch scripts for Pfam / BAliBase
  / OxBench (no bundled data; corpora pulled at runtime).

## Reproduction

See [`REPRODUCTION_MANIFEST.md`](REPRODUCTION_MANIFEST.md) for the
complete recipe and pointers to released artefacts on this repo's
GitHub Releases page.

## Releases

| Tag | Contents |
|---|---|
| [`results/K4-emwarm-top1000-2026-05-09`](../../releases/tag/results/K4-emwarm-top1000-2026-05-09) | TKF-DP $K_c{=}4$ EM-warmup checkpoint (Pfam top-1000), backs Figs 3-5 and Table 1 row inf-PHMM-$K{=}4$ |
| [`qprime-bali3pdbm-2026-05-16`](../../releases/tag/qprime-bali3pdbm-2026-05-16) | Q' caches (per-pair posteriors) for all 8 methods compared in Tables 1+2, ~3.6 GB across 8 tarballs |

The companion `tkf-mixdom` repo also has a release of trained checkpoints:

| Repo | Tag | Contents |
|---|---|---|
| `evoldoers/tkf-mixdom` | [`mixdom-checkpoints-2026-05`](https://github.com/evoldoers/tkf-mixdom/releases/tag/mixdom-checkpoints-2026-05) | MixDom-d3f1, TKF92-K=20, CherryML-C=20 NPZ ckpts (3 files, ~210 KB total) |

## License

Code: see `LICENSE` (or per-file headers).
Paper and figures: CC-BY 4.0 (consistent with bioRxiv standard).
