# Q' soft-posterior cache (committed mirror of `~/.cache/tkf-mixdom-balibase/`)

Per-pair `Q'[i, j]` arrays for every method evaluated on BAliBASE 3 PDB-M.
Each subdirectory corresponds to one method; each `<family>.{npz,json}`
pair holds the soft (or hard) per-pair posteriors and metadata.

## Schema (uniform across methods)

`<family>.npz`:
- `names`: (n_seqs,) sequence-name array
- `pairs`: (n_pairs, 2) int — sequence-index pairs `[i, j]`
- `post_k`: (Lx, Ly) float32 — pair-k posterior `Q'[i, j]` (or hard 0/1)

`<family>.json`:
- `kind`: `'soft'` or `'hard'`
- `params_key`: 16-hex hash uniquely identifying the (model, params,
  training corpus, sampler config) tuple — cache hits require an exact
  match.

## Methods committed here

| Method | Files | Kind | Notes |
|---|---|---|---|
| `tkf92_lg08` | 22 fams | soft | Baseline TKF92 forward-backward at corpus-fitted rates (ins=0.04581, del=0.04680, ext=0.6835), LG08 substitution. |
| `mixdom_d3f1` | 22 fams | soft | MixDom2 D=3, F=1, C=3 trained on Pfam. |
| `cherryml_C20` | 22 fams | soft | CherryML-distilled mixture of C=20 substitution classes. |
| `tkf92_K20` | 22 fams | soft | K=20-component TKF92 indel mixture. |
| `mafft_auto` | 22 fams | hard | MAFFT FFT-NS-2 alignment, rasterised to (0/1) per-pair indicators. |
| `muscle` | 22 fams | hard | MUSCLE default alignment, same rasterisation. |
| `infinite_phmm_mcmc_K4_coupled_RE` | ≤22 fams (L<150 subset) | soft | TKF-DP K=4 Potts-coupled MCMC sampler (replica-exchange, alpha_z ladder [100,250,700,2000,10⁴], 500 sweeps + 100 burnin). |

## Maintenance

The local cache at `~/.cache/tkf-mixdom-balibase/` is the canonical
working location consulted by the sweep launcher and downstream FSA
tools. This repo directory is a mirror, updated per-family by the
watcher script at `analysis/scripts/qprime_cache_watcher.py` (committed
separately).

To rebuild the cache from scratch for a single method, delete the
corresponding subdirectory of `~/.cache/tkf-mixdom-balibase/<method>/`
and re-run the relevant sweep.
