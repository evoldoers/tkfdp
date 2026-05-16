# K=4 PDB-anchored, Pfam top-1000-with-SCOP (no side potentials)

## What this is

A second TKF-DP K=4 checkpoint, paired with the existing K=4 EM-warmup
release (`K4-emwarm-top1000-2026-05-09`). Same K_c=4, same Potts atom
DP, same LG08 substitution baseline, same single-family validation
(PF00076). The difference is supervision: every training family's
column-pair partition is **fixed to PDB Cα<8 Å contacts** for the
duration of training, so the SVI never resamples edges. The only
latents resampled are per-column site-class labels c_s, the per-class
profile pi_class, the Potts atoms H_t, and the class-pair → atom TSB
assignment.

## Result

- Best `val_LL = -292.42` at outer iter 65 (vs **-298.41** for the
  unsupervised K=4 EM-warm release on the same val family) →
  **+5.99 nats** improvement on PF00076.
- Early-stopped at iter 95 on patience-6.
- Total wall: 11 798 s ≈ 3 h 17 min on one RTX A6000.
- Per-outer mean: 124 s.

## Configuration

| Flag | Value |
|---|---|
| `--processed-dir` | `data/pfam_processed_top1000_pdb` (rebuilt by `analysis/k4_pdbanchor/build_manifest.py`) |
| Manifest size | 1000 (v1.train ∩ valid SCOP id, top-cherry-count) |
| Effective at load (after corpus filters) | 715 families |
| `--K` (K_c) | 4 |
| `--K-H-max` | 10 (default; not reduced) |
| `--alpha-c` / `--alpha-H` | 10.0 / 1.0 |
| `--a-eta` / `--b-eta` / `--kappa-pi` | 2.0 / 2.0 / 4.0 |
| `--resume-globals-from` | `results/exp2_v2_K4_top1000_tsb_emwarm/_best_chkpt` |
| `--anchor-families` | the full 1000-family manifest |
| `--em-warmup-iters` | default 500 (refined c_s on the new corpus over the loaded pi_class) |
| `--val-families` | `PF00076` (matches the K=4 emwarm release for direct comparability) |
| `--patience` | 6 |
| Substitution model | LG08 (S, pi via `tkfdp.lg08`) |
| TKF92 indel rates | corpus-fitted (ins=0.04581, del=0.04680, ext=0.6835) from `~/tkf-mixdom/python/experiments/tkf92_fitted_params.json` |

Of the 1000 manifest families, **32 fell through the anchor at
runtime** ("no PDB contacts; treating as non-anchor") because PDB
fetch / chain-extraction / sequence-alignment failed despite a valid
SCOP cross-ref in the seed Stockholm. Those 32 trained as ordinary
(non-anchor) families. The remaining ~683 (= 715 effective − 32)
ran with their column-pair partition pinned to Cα<8 Å contacts.

## Code provenance

The training-time SVI loop is unchanged vs the K=4 emwarm release.
The only new code paths exercised are:

- `tkfdp.checkpoint.load_globals_from_checkpoint` (commit `6ba1260`)
  — partial-resume helper that loads only pi_class + PottsDPState
  from a checkpoint, leaving per-family latents fresh. Required because
  the PDB-anchored corpus has a different family list from the K=4
  emwarm release; `--resume-from` enforces an exact match.
- `tkfdp.checkpoint.load_globals_from_checkpoint` falls back to the
  `init_svi_state` uniform stick weights when `rho` / `tsb_betas` are
  absent from the source checkpoint (commit `0cb651c`). The K=4
  emwarm checkpoint predates the always-persist fix for those fields;
  without this fallback `tsb_resample_assignments` trips on a None
  several outer iters in.

Plan + corpus manifest (deterministic, regenerable on another machine):
`analysis/k4_pdbanchor/{plan.md, build_manifest.py, manifest.json,
val.json, launch.sh}`. The repo state at the tag is reproduction-equivalent.

## Files

| Path | Size | Notes |
|---|---|---|
| `_best_chkpt/state.npz` | 1.9 MB | best-val snapshot (pi_class, potts_atoms, potts_assignments, potts_counts, per-MSA cls/partner/eta over 715 families) |
| `_best_chkpt/meta.json` | 14 KB | hyperparams, RNG state, family list, best_val_LL=-292.42 |
| `_best_chkpt/trace.json` | 12 KB | per-iter training trace |
| `exp2_v2_K4_top1000_pdbanchor.log` | 166 KB | full training log including per-outer pi_diff / `||H_0||` / class_counts and val-LL trajectory |

## Reproduction

```bash
git checkout 'results/K4-pdbanchor-top1000-2026-05-13'
gh release download 'results/K4-pdbanchor-top1000-2026-05-13' \
    --repo ${REPO_OWNER}/tkfdp -p '*.tar.gz'
tar xzf K4-pdbanchor-top1000-2026-05-13.tar.gz

# (1) Eval the released checkpoint head-to-head against the emwarm release:
PYTHONPATH=src python3 experiments/eval_balibase.py \
    --bench bali3 \
    --checkpoints \
        K4_emwarm:results/exp2_v2_K4_top1000_tsb_emwarm/_best_chkpt \
        K4_pdbanchor:_best_chkpt

# (2) Re-train from scratch on this corpus:
analysis/k4_pdbanchor/build_manifest.py    # rebuilds data/pfam_processed_top1000_pdb/ + manifest.json + val.json
analysis/k4_pdbanchor/launch.sh            # fires the training on GPU 1
```

## Caveats

- 32/1000 anchor-fallthroughs slightly contaminate the "every family
  is PDB-anchored" framing. If you want a strict-anchored re-run, drop
  those families from the manifest and rebuild.
- Effective corpus is 715 not 1000. The drop is a corpus loader filter
  (likely `min_cherries`) downstream of the manifest. Worth verifying
  if you want apples-to-apples cherry counts vs the K=4 emwarm release
  (which used 1000 families, 96k cherries; this run used 715 families,
  ~32k cherries — a 3× cherry-count reduction).
- Val LL is on a single family (PF00076) chosen for comparability with
  the K=4 emwarm release. The +5.99 nat improvement should not be
  taken as a corpus-wide held-out result without a broader val sweep.
