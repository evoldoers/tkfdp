# TKF-DP reproduction manifest

## GitHub release URLs (parameterised — do NOT hardwire to `ihh`)

The reproduction recipe (Section 4 below) uses these placeholder
variables; substitute your own owner / repo name when forking:

```bash
RELEASE_OWNER=ihh      # or "evoldoers", whoever owns the canonical drop
RELEASE_REPO=tkfdp     # or "tkf-dp" (current repo name)
```

**Released assets (current location, as of 2026-05-16):**

| Asset | Tag | URL pattern |
|---|---|---|
| K=4 emwarm checkpoint (paper headline) | `results/K4-emwarm-top1000-2026-05-09` | `https://github.com/${RELEASE_OWNER}/${RELEASE_REPO}/releases/tag/results/K4-emwarm-top1000-2026-05-09` |
| K=4 PDB-anchored variant | `results/K4-pdbanchor-top1000-2026-05-13` | `https://github.com/${RELEASE_OWNER}/${RELEASE_REPO}/releases/tag/results/K4-pdbanchor-top1000-2026-05-13` |
| K=4, K_H=4 atom-sharing test | `results/K4-KH4-emwarm-top1000-2026-05-15` | `https://github.com/${RELEASE_OWNER}/${RELEASE_REPO}/releases/tag/results/K4-KH4-emwarm-top1000-2026-05-15` |
| **Q' matrices** (all 7 methods) | `qprime-bali3pdbm-2026-05-16` *(TBD; release after BB11021 extras land)* | `https://github.com/${RELEASE_OWNER}/${RELEASE_REPO}/releases/tag/qprime-bali3pdbm-2026-05-16` |

The Q' release bundles the 7 tarballs at
`s3://tkf-mixdom-gpu-618647024028/qprime-cache/` (manifest at
`manifest_LATEST.json`); ~1.8 GB total. S3 paths remain a fallback.

---

Drop intended for `tkfdp.net`. Three top-level repos (one git submodule
+ one symlink-via-fetch-scripts):

```
tkfdp.net/
├── tkf-dp/           ← paper sources, K=4 SVI training, K=4 sampler, figure scripts
│   └── math-paper/tkf-mixdom/   ← git submodule pointing at the tkf-mixdom repo below
├── tkf-mixdom/       ← JAX inference / training library (TKF91/TKF92/MixDom)
└── bio-datasets/     ← fetch + preprocessing scripts (no data; ~/.bio-datasets/data is gitignored)
```

The same `tkf-mixdom` checkout is included twice: once as a working
clone (`tkf-mixdom/`) for the `cd tkf-mixdom/python && make`
workflows, and once as a git submodule under `tkf-dp/math-paper/`
because the paper builds rely on `\input{tkf-mixdom/tkf/...}`
relative paths and on relative `\graphicspath{}` references into
`tkf-mixdom/python/experiments/figures/`. Submodule pin is the
`v-slim-2026-05-02` family (currently `b9825d5ce`); the clean drop
should advance both to whatever commit ships.

The released TKF-DP checkpoint
(`results/K4-emwarm-top1000-2026-05-09/`) is included verbatim in
`tkf-dp/` (≈4 MB), since `main.tex`'s figures consume it
directly. Larger artefacts (Q'-cache tarballs, 187 per-pair MCMC
JSONs, per-pair Q' .npz files) live on S3 — referenced by URL, not
copied.

---

## Section 1 — Manifest tree

### `tkf-dp/`

```
tkf-dp/
├── CLAUDE.md                                    # orientation; useful even for human readers
├── README.md                                    # NEW — to be written; see Reproduction Recipe below
├── .gitmodules                                  # math-paper/tkf-mixdom → ${REPO_OWNER}/tkf-mixdom
├── .gitignore                                   # ignores data/, ~/.cache, *.aux, *.bbl, results/legacy*
├── build.sh                                     # NEW — one-liner; wraps math-paper/build.sh
│
├── math-paper/                                  # ← what tkfdp.net serves as the PDF
│   ├── build.sh                                 # pdflatex + bibtex driver (main, supplement, combined)
│   ├── main.tex                                 # 10-pp main paper
│   ├── supplement.tex                           # appendix driver (\input's tkf-mixdom/tkf/*)
│   ├── appendix-ggi.tex                         # GGI moment-matching appendix
│   ├── appendix-tkfdp.tex                       # TKF-DP generative model + inference (Appx D)
│   ├── appendix-infinite-phmm.tex               # infinite Pair HMM + MCMC sampler (Appx E)
│   ├── refs.bib                                 # 85 entries, paper-side; supplement reuses
│   ├── tkf-mixdom/                              # git submodule → ${REPO_OWNER}/tkf-mixdom @ b9825d5c
│   ├── figures/
│   │   ├── tkf_trajectory.py                    # base class for the Galton-Watson figure
│   │   ├── tkf_trajectory_uber.py               # generator for tkf_trajectory_uber.pdf
│   │   ├── tkf_trajectory.tex                   # standalone PGF (optional)
│   │   ├── tkf_trajectory_galton_watson.tex     # standalone PGF (optional)
│   │   ├── fig_bdi_consistency_highmu.py        # regenerator for bdi_consistency_{B,D_highmu,S}.pdf
│   │   ├── AA_EVOLUTION_RECIPE.md               # NOT required (the figures aren't in main.tex)
│   │   ├── tkf_trajectory_uber.pdf              # binary, committed; Fig 1
│   │   ├── bdi_consistency_B.pdf                # binary, committed; Fig 2 row 1 left
│   │   ├── bdi_consistency_D_highmu.pdf         # binary, committed; Fig 2 row 1 middle
│   │   ├── bdi_consistency_S.pdf                # binary, committed; Fig 2 row 1 right
│   │   ├── tkf_bdi_validation_means.pdf         # binary, committed; Fig 2 row 2
│   │   ├── tkf_bdi_validation_shrinkage.pdf     # binary, committed; Fig 2 row 3
│   │   ├── k4_coupling_score.pdf                # binary, committed; Fig 3
│   │   ├── holmes_tile_PF00053_lama1_test.pdf   # binary, committed; Fig 4
│   │   └── holmes_msa_triangle_PF00053.pdf      # binary, committed; Fig 5
│   └── results/
│       ├── infinite_phmm_balibase.json          # rows of Table 1 — inf-PHMM (canonical)
│       ├── infinite_phmm_balibase_k4.json       #   "  K=4 single-chain
│       ├── infinite_phmm_balibase_k4_replicaexchange.json   # "  K=4 RE final
│       ├── infinite_phmm_balibase_k4_top_rung_validation.json
│       ├── infinite_phmm_balibase.aggregate.json
│       ├── balibase_l150_summary.json           # L<150 subset table source
│       ├── balibase_summary_full.csv            # Table 1 (full corpus); also .md
│       ├── balibase_summary_full.md
│       ├── balibase_summary_l150.csv            # Table 2; also .md
│       ├── balibase_summary_l150.md
│       ├── k4_re_downstream_fsa.json            # inf-PHMM after FSA pipe
│       └── qprime_cache/
│           ├── README.md                        # schema + provenance
│           ├── cherryml_C20/    (22 .npz + 22 .json — soft posteriors per family)
│           ├── infinite_phmm_mcmc_K4_pdbanchor_RE/   (22 fams, ≤22 L<150)
│           ├── mafft_auto/                      (hard 0/1)
│           ├── mixdom_d3f1/
│           ├── muscle/                          (hard 0/1)
│           ├── tkf92_K20/
│           └── tkf92_lg08/
│
├── src/tkfdp/                                   # the TKF-DP inference package
│   ├── __init__.py                              # module docstring + AMINO_ACIDS constant
│   ├── lg08.py                                  # LG08 exchangeabilities + F81 form
│   ├── generator.py                             # F81 joint Q(H), eigendecomposition, P(t)
│   ├── secret_destination.py                    # F81 augmentation (Dirichlet conjugacy)
│   ├── partition_K.py                           # CRP-Gibbs over column partitions
│   ├── potts_dp.py                              # Potts atom DP + TSB sweep
│   ├── laplace_potts_v2.py                      # JIT Laplace MAP for atoms (canonical)
│   ├── laplace_potts.py                         # older Laplace (kept for compat with earlier ckpts)
│   ├── eta_site.py                              # per-site gamma rate-multiplier conjugacy
│   ├── multiclass.py                            # per-class profile updates
│   ├── postprocessing.py                        # boost integration → Q' (pre-correction)
│   ├── coupled_annealing.py                     # column-paired FSA (uses tkfmixdom.jax.tree.fsa_anneal)
│   ├── svi.py                                   # main SVI loop (was svi_v2.py)
│   ├── checkpoint.py                            # SVIState save/load .npz + meta.json + trace.json
│   ├── cherries.py                              # tree → cherry list with distances
│   ├── pfam_data.py                             # corpus loader (bio-datasets-aware)
│   ├── pfam_data_fast.py                        # JIT-friendly cached variant
│   ├── bdi_reference.py                         # closed-form TKF91 P_ij(T), E[B], E[D], E[S]
│   ├── branch_sampler.py                        # plain branch sampler (legacy)
│   ├── branch_sampler_recursive.py              # gravestone-augmented branch sampler (Appx D)
│   ├── f2_scfg.py                               # F2 SCFG (referenced in Appx E §1)
│   ├── aug_phmm.py                              # 1-edge augmented Pair HMM (Appx E §2)
│   ├── aug_phmm_2edge.py                        # 2-edge variant
│   ├── aug_phmm_antidiag.py                     # antidiagonal-scan optimised
│   ├── mcmc_infinite_phmm.py                    # block-resample MCMC (Appx E §3) ← Tab 1 source
│   ├── single_seq_edge_mcmc.py                  # 1-seq baselines for Fig 4 (panels a, b)
│   ├── block_likelihoods.py                     # block primitives for the boost tensor
│   ├── composite.py                             # composite log-likelihood
│   ├── composite_partition.py                   # MSA-column composite (Fig 5)
│   ├── partition.py                             # plain (non-K) partition sampler
│   ├── balibase_pdb_contacts.py                 # BAliBASE/PDB contact lookup for plots
│   ├── pdb_contacts.py                          # generic PDB-contact utilities
│   ├── loss_elbo.py                             # ELBO loss (Appx D §1)
│   ├── sim.py                                   # forward simulator (uniformization)
│   ├── dp_alpha.py                              # Escobar-West alpha_H aux Gibbs
│   ├── train.py                                 # Adam loop for H (synthetic experiment)
│   ├── bio.py                                   # Pfam parser (legacy; pfam_data preferred)
│   └── inspect_params.py                        # ckpt → human-readable summary
│
├── experiments/
│   ├── exp2_pfam_v2.py                          # main SVI training entrypoint (produced K4 ckpt)
│   ├── preprocess_pfam_topN.py                  # Stockholm → per-family .npz cherries
│   ├── eval_balibase.py                         # multi-method BAliBASE eval harness
│   └── eval_msa_composite_loglik.py             # composite-likelihood val LL on MSAs
│
├── analysis/scripts/
│   ├── sweep_infinite_phmm_balibase.py          # GENERATOR for infinite_phmm_balibase*.json (Tab 1, Tab 2 inf-PHMM row)
│   ├── qprime_cache_watcher.py                  # background worker; mirrors ~/.cache/tkf-mixdom-balibase/* into results/qprime_cache/
│   ├── downstream_fsa_on_cached_qprime.py       # consumes qprime_cache/* → MSA + SP/TC
│   ├── plot_k4_coupling_and_doublet.py          # GENERATOR for k4_coupling_score.pdf (Fig 3)
│   ├── plot_holmes_tile.py                      # GENERATOR for holmes_tile_PF00053_lama1_test.pdf (Fig 4)
│   ├── plot_holmes_msa_triangle.py              # GENERATOR for holmes_msa_triangle_PF00053.pdf (Fig 5)
│   ├── select_close_cys_pair.py                 # selection helper for Fig 4 family
│   ├── select_pdb_edge_cov_pair.py              # selection helper for PDB-anchor Fig 4 variant
│   ├── select_strongest_cov_pair.py             # selection helper (cov ranking)
│   ├── aggregate_infinite_phmm_results.py       # rolls per-pair JSONs into one .aggregate.json
│   ├── balibase_summary.py                      # writes balibase_summary_*.csv/.md from JSONs
│   ├── l150_subset_summary.py                   # L<150 subset cohort
│   ├── calibrate_infinite_phmm.py               # max-len calibration on 11 GiB GPU
│   ├── check_top_rung_pass.py                   # convergence-validation diagnostic
│   ├── compare_top_rung_to_tkf92.py             # top-rung vs TKF92 baseline
│   └── download_k4_checkpoint.sh                # one-shot ckpt pull from S3/gh
│
├── aws/
│   ├── PLAN.md                                  # 187-pair AWS sweep design
│   ├── balibase_jit_primer.yaml                 # SkyPilot: prime JAX cache, push to S3
│   ├── balibase_one_pair_v2.yaml                # SkyPilot: per-pair worker (current)
│   ├── balibase_one_pair.yaml                   # SkyPilot v1 (kept for archival reproducibility)
│   ├── balibase_one_family.yaml                 # SkyPilot: per-family worker
│   ├── launch_balibase_direct.py                # direct-launch driver (bypasses managed-jobs)
│   ├── launch_balibase_sweep.py                 # managed-jobs driver (v1)
│   └── merge_balibase_results.py                # S3 per-pair JSONs → replicaexchange.json
│
├── scripts/
│   └── fetch_aligners.sh                        # MUSCLE 5 + MAFFT 7 → ~/.local/bin/
│
├── docs/
│   └── balibase_paired_fsa_quickstart.md        # runbook (consumed by README.md)
│
├── refs/
│   ├── cohn2010.tex                             # Cohn et al. 2010 — bibliographic / quote source
│   ├── cohn2010.txt                             # text excerpt
│   └── holmes_rubin_elbo.md                     # working notes (light; can drop if budget tight)
│
├── tests/
│   ├── smoke_aug_phmm.py                        # smoke: 1-edge aug Pair HMM
│   ├── smoke_aug_phmm_2edge.py                  # smoke: 2-edge variant
│   ├── smoke_aug_phmm_antidiag.py
│   ├── smoke_f2_scfg.py                         # smoke: F2 SCFG
│   ├── smoke_mcmc_infinite_phmm.py              # smoke: block-resample sampler
│   ├── smoke_postprocessing.py                  # smoke: boost integration
│   ├── smoke_loss_elbo.py
│   ├── smoke_checkpoint.py
│   ├── smoke_composite_partition.py
│   ├── smoke_chunked_hvp.py
│   ├── test_balibase_postprocess.py             # integration: BB11001 pre-correction
│   ├── test_balibase_coupled_annealing.py       # integration: 3-way comparison on BB11001
│   ├── test_block_likelihoods.py
│   ├── test_edge_pair_projection.py
│   ├── test_msa_column_projection.py
│   ├── test_single_seq_edge_mcmc.py
│   └── test_mcmc_diagnostics.py
│
└── results/
    └── K4-emwarm-top1000-2026-05-09/            # released TKF-DP checkpoint
        ├── MANIFEST.md                          # configuration provenance
        ├── _best_chkpt/
        │   ├── state.npz                        # ← consumed by all figure / eval scripts
        │   ├── meta.json
        │   └── trace.json
        └── logs/                                # training stdout
```

### `tkf-mixdom/` (working clone)

Only what is reachable from the paper appendices + the `expected_balibase`
make targets + the figure generators. Most of `python/experiments/` is
dev-only and excluded.

```
tkf-mixdom/
├── CLAUDE.md
├── README.md                                    # NEW; quickstart for the JAX library alone
│
├── tkf/                                         # LaTeX bodies \input'd by supplement.tex
│   ├── build.sh                                 # builds tkf.tex + mixdom.tex (standalone TKF / MixDom papers)
│   ├── preamble-shared.tex
│   ├── refs.bib
│   ├── body-tkf91.tex                           # supplement App A
│   ├── body-tkf92.tex                           # supplement App A
│   ├── tkf92-wfst-derivation.tex                # supplement App A
│   ├── lhopital-limits.tex                      # supplement App A
│   ├── joint-vs-conditional.tex                 # transitively \input'd by lhopital-limits
│   ├── irreversible.tex                         # transitively \input'd by lhopital-limits
│   ├── score-derivatives.tex                    # supplement App A
│   ├── substitution-mstep.tex                   # supplement App B
│   ├── svb-tkf91.tex                            # supplement App B
│   ├── svb-convergence.tex                      # supplement App B
│   ├── body-maraschino-main.tex                 # supplement App B
│   ├── body-tkf-inference.tex                   # supplement App B
│   ├── varanc-presence.tex                      # supplement App B
│   ├── varanc-bias-appendix.tex                 # supplement App B
│   ├── body-mixdom.tex                          # supplement App C
│   ├── body-mixdom-inference.tex                # supplement App C
│   ├── mixdom-algorithms.tex                    # supplement App C
│   ├── exploded-mixdom.tex                      # supplement App C
│   ├── maraschino.tex                           # supplement App C (renamed in supplement)
│   ├── algebraic-distillation.tex               # supplement App C
│   ├── svb-convergence-mixdom.tex               # supplement App C
│   ├── varanc-vbem.tex                          # supplement App C
│   ├── varanc-presence-mixdom.tex               # supplement App C
│   ├── partition-recon.tex                      # supplement App C
│   ├── mixdom-wfst.tex                          # supplement App C
│   ├── grammar-elaboration.tex                  # supplement App C
│   └── recursive.tex                            # supplement App C (1230 lines)
│   # The following tkf/*.tex files exist in the submodule but are NOT
│   # \input'd by supplement.tex; they are excluded from the clean drop:
│   #   tkf.tex (entrypoint of standalone tkf paper)
│   #   mixdom.tex (entrypoint of standalone MixDom paper)
│   #   body-neural.tex, neural.tex, sim-eval.tex, frontmatter-*.tex,
│   #   models-fitted.tex, progrec.tex, varanc-bias-*.tex (correction/discussion),
│   #   implementations.tex
│   # ...but if the user wants a self-buildable submodule (so they can
│   # `cd tkf-mixdom/tkf && ./build.sh` and get the standalone TKF and
│   # MixDom papers too), include ALL of tkf/.
│
└── python/
    ├── pyproject.toml                           # JAX, optax, mpmath deps
    ├── uv.lock                                  # frozen lock for reproducibility
    ├── tkfmixdom/                               # the importable package
    │   ├── __init__.py
    │   ├── util/                                # bio_datasets, padding, pair_format, sto_index, timing, io
    │   │   ├── bio_datasets.py                  # ← consumed by ALL training scripts
    │   │   ├── pair_format.py, pair_loader.py
    │   │   ├── padding.py, data.py, io.py, sto_index.py, timing.py
    │   │   └── __init__.py
    │   └── jax/
    │       ├── core/                            # types, BDI params, CTMC, alphabets
    │       │   ├── types.py, params.py, bdi.py
    │       │   ├── ctmc.py, ctmc_irreversible.py
    │       │   ├── protein.py, protein_gap.py, protein_gap40.py, rna.py
    │       │   ├── site_class_profiles.py
    │       │   └── __init__.py
    │       ├── grammar/                         # SCFG rules, DP, null removal
    │       │   ├── compile.py, scfg.py, scfg_dp.py
    │       │   └── __init__.py
    │       ├── models/                          # TKF91/92 HMMs, MixDom, distilled compositions
    │       │   ├── left_regular.py              # make_tkf92_pair_hmm — used by tkfdp.coupled_annealing
    │       │   ├── mixdom.py, fully_exploded.py
    │       │   ├── elimination_steps.py, exact_suffstats.py, exact_suffstats_batch.py
    │       │   ├── compiled.py, context_free.py, chi_free.py
    │       │   ├── order1_scfg.py, elaborated.py
    │       │   ├── mixdom_init.py, annabel_mixdom.py, annabel_to_mixdom2.py
    │       │   ├── tkf_grammar.py, rna_grammar.py
    │       │   └── __init__.py
    │       ├── dp/                              # 2D forward-backward, beams
    │       │   ├── hmm.py                       # forward_backward_2d — used by tkfdp
    │       │   ├── hmm_beam.py, scfg_beam.py
    │       │   ├── scfg_factored.py, singlet_forward.py
    │       │   └── __init__.py
    │       ├── train/                           # EM, SVI-BW, Adam, optimizers
    │       │   ├── em.py, vjp.py, constrained.py
    │       │   ├── likelihood.py, fit.py, optimizer.py, adam_train.py
    │       │   ├── tkf92_svi_bw.py, tkf92_vbem.py, tkf92_padded_elbo.py
    │       │   ├── tree_vbem.py, pseudocounts.py
    │       │   ├── restricted_mstep.py, early_stopping.py
    │       │   └── __init__.py
    │       ├── distill/                         # cherry-distill HMM/SCFG → order-1
    │       │   ├── maraschino.py                # load_params helper used by expected_pairwise_balibase
    │       │   ├── maraschino_fit.py            # fit driver
    │       │   ├── hmm.py, scfg.py, wptt.py
    │       │   ├── banded_mixture.py, tkf92_mixture.py
    │       │   └── __init__.py
    │       ├── tree/                            # composition, intersection, FSA, ProgRec, beams
    │       │   ├── fsa_anneal.py                # compute_pairwise_posteriors, sequence_annealing, fsa_align — central to expected_pairwise_balibase + tkfdp.coupled_annealing
    │       │   ├── ancestor.py, compose.py, compose_wptt_rec.py
    │       │   ├── composite_beam.py, composite_beam_jax.py
    │       │   ├── felsenstein.py, progrec_felsenstein.py, progressive.py
    │       │   ├── guide_tree.py, intersect.py, partition_recon.py, partition_recon_jax.py
    │       │   ├── msa_constrained_viterbi.py, profile_compress.py
    │       │   ├── recognizer.py, transducer.py, tree_varanc.py, triad_1d.py, triad_gap_inference.py
    │       │   ├── varanc_presence.py, varanc_presence_mixdom.py
    │       │   └── __init__.py
    │       ├── simulate/                        # plain TKF91/TKF92 simulation (Gillespie)
    │       │   ├── evolve.py, pair_hmm.py, msa.py, simulate.py
    │       │   ├── mixdom_gillespie.py          # consumed by fig_tkf_bdi_validation.py
    │       │   ├── tree_mixdom.py, collapsed_mixdom2.py, mixture_sites.py, labeled_utils.py
    │       │   └── __init__.py
    │       ├── evaluate/                        # io, metrics (sp_score, tc_score)
    │       │   ├── io.py, metrics.py
    │       │   └── __init__.py
    │       └── util/                            # msa_benchmark, balibase_pair_cache, expected_pair_f1
    │           ├── msa_benchmark.py             # parse_fasta, sp_tc_score
    │           ├── expected_pair_f1.py          # the soft pair-F1 accumulator used by every method row of Tab 1, Tab 2
    │           ├── balibase_pair_cache.py
    │           └── __init__.py
    ├── train_pfam.py                            # SVI-BW training driver for MixDom (was used to produce svi_bw_d3f1_postfix*.npz)
    ├── maraschino.py                            # cherry-distillation training driver (CherryML_C20, TKF92_K20)
    ├── fit_tkf92_mixture.py                     # generator for tkf92_mixture_K20_train.npz
    ├── fit_banded_mixdom2_mixture.py            # generator for the banded MixDom variant (not in headline table)
    ├── build_tkf92_cherry_counts.py             # cherries → counts tensors for TKF92 mixture fit
    ├── build_marcounts_parallel.py              # Maraschino counts pipeline
    ├── check_marcounts_integrity.py             # one-shot sanity check
    ├── build_train_trees_subset.py              # Pfam → tree subset selector
    ├── verify_reduced_wfst_routes.py            # WFST consistency check
    │
    ├── experiments/
    │   ├── expected_pairwise_balibase.py        # ← canonical runner for Tab 1 + Tab 2 (all 6 soft + hard rows)
    │   ├── Makefile.expected_balibase           # `make all` runs all 6 method rows
    │   ├── tkf92_fitted_params.json             # corpus-fitted TKF92 anchor params (lambda=0.04581, mu=0.04680, r=0.6835, kappa=0.979)
    │   ├── fsa_tkf92_mixture_balibase.py        # load_mixture_ckpt — helper for expected_pairwise_balibase
    │   ├── fsa_mixdom_pairhmm_oxbench.py        # MixDom pair-HMM builder (re-used for BAliBASE)
    │   ├── fsa_tkf92_oxbench.py                 # TKF92 FSA driver (supplement K-sweep on OxBench)
    │   ├── fig_bdi_consistency.py               # GENERATOR (canonical) for bdi_consistency_{B,D,S}.pdf (Fig 2 row 1)
    │   ├── fig_tkf_bdi_validation.py            # GENERATOR for tkf_bdi_validation_{means,shrinkage}.pdf (Fig 2 rows 2, 3)
    │   ├── fig_bdi_lhopital.py                  # supplement-only validation figure (App A)
    │   ├── fig_bdi_stats.py                     # supplement-only validation figure (App A)
    │   ├── figures/
    │   │   ├── bdi_consistency_B.pdf            # generated by fig_bdi_consistency.py (also re-emitted into math-paper/figures by the _highmu wrapper)
    │   │   ├── bdi_consistency_D.pdf            # canonical (low-mu); the paper uses bdi_consistency_D_highmu.pdf from the wrapper instead
    │   │   ├── bdi_consistency_S.pdf
    │   │   ├── tkf_bdi_validation_means.pdf
    │   │   ├── tkf_bdi_validation_shrinkage.pdf
    │   │   ├── tkf_bdi_validation_shrinkage.json
    │   │   └── (other figures here are dev-only, supplement-only, or for the MixDom paper)
    │   └── expected_balibase/                   # output dir for Makefile.expected_balibase
    │       ├── expected_balibase_tkf92_lg08*.json
    │       ├── expected_balibase_tkf92_K20*.json
    │       ├── expected_balibase_cherryml_C20*.json
    │       ├── expected_balibase_mixdom_d3f1*.json
    │       ├── expected_balibase_mafft*.json
    │       ├── expected_balibase_muscle*.json
    │       └── expected_balibase_inf_phmm_K4_l150_withsps.json
    │
    ├── data/
    │   └── (gitignored; symlinks created by util.bio_datasets at runtime)
    │
    └── pfam/                                    # trained MixDom checkpoints (~1 GB; can be hosted on S3 instead — see Recipe)
        ├── svi_bw_d3f1_postfix_best_val.npz     # ← MixDom-d3f1 row of Tab 1 / Tab 2
        ├── svi_bw_d3f1_postfix.npz              # rolling ckpt (optional, but lets the user resume training)
        ├── tkf92_mixture_K20_train.npz          # ← TKF92-K=20 row
        ├── cherryml_mixture_C20_n5000.npz       # ← CherryML-C=20 row
        # (other ckpts in pfam/ are dev / training-history artefacts; exclude)
```

### `bio-datasets/`

Fetch scripts only; data itself is gitignored.

```
bio-datasets/
├── README.md
├── CLAUDE.md
├── fetch/
│   ├── __init__.py
│   ├── common.py                                # safe_download, safe_outdir, idempotent helpers
│   ├── pfam/
│   │   ├── fetch.py                             # Pfam-A.seed.gz (~20k families) OR --random N
│   │   ├── fetch_full.py                        # full Pfam-A
│   │   ├── prepare.py                           # Pfam .sto → per-family parquet
│   │   ├── prepare_full.py
│   │   ├── preprocess.py                        # tree extraction + cherry counts
│   │   ├── make_split.py                        # train/val/test family splits
│   │   └── splits/                              # pre-computed splits (a few KB; ship them)
│   ├── balibase/
│   │   └── fetch.py                             # drive5 mirror (21 MB; BAliBASE 3 + PREFAB + OXBENCH + SABRE)
│   ├── treefam/fetch.py                         # TreeFam (used by varanc benchmarks, not Tab 1/2)
│   ├── rfam/fetch.py                            # Rfam (RNA; supplement reference only)
│   └── (other fetchers — annevo, crw, gencode, gtrnadb, ncbi, panther, proteingym, silva, tiberius, ucsc, zoonomia — NOT required by this paper; ship them only if you want one canonical bio-datasets repo across all projects)
```

### S3 references (NOT copied; pull at runtime via `aws --profile tkf-gpu s3 cp`)

```
s3://tkf-mixdom-gpu-618647024028/
├── qprime-cache/                                # tarballs of per-family Q' caches (per method)
│   ├── manifest_LATEST.json                     # always-newest pointer
│   ├── manifest_2026-05-16T173404Z.json
│   ├── qprime_tkf92_2026-05-16T173404Z.tar.gz                          (479 MB)
│   ├── qprime_mixdom_d3f1_2026-05-16T173404Z.tar.gz                    (649 MB)
│   ├── qprime_tkf92_K20_2026-05-16T173404Z.tar.gz                      (309 MB)
│   ├── qprime_cherryml_C20_2026-05-16T173404Z.tar.gz                   (323 MB)
│   ├── qprime_mafft_2026-05-16T173404Z.tar.gz                          (464 KB)
│   ├── qprime_muscle_2026-05-16T173404Z.tar.gz                         (480 KB)
│   ├── qprime_infinite_phmm_mcmc_K4_coupled_RE_2026-05-16T173404Z.tar.gz (540 KB)
│   └── tkfmixdom_balibase_cache_2026-05-16_partial.tar.gz              (1.1 GB; superseded by per-method files)
├── balibase-runs/v11-aws-canonical/             # ← 187 per-pair MCMC outputs (Tab 1 inf-PHMM row)
│   ├── BB11001_0_1.json … BB40045_8_*.json     (187 diagnostic JSONs)
│   └── qprime/
│       ├── BB11001_0_1.npz, BB11001_0_1_qpcache.json … (187 .npz + 187 .json)
└── balibase-bundle/                             # what aws/balibase_one_pair_v2.yaml pulls at worker boot (BAliBASE + ckpt + venv + JAX cache)
```

The released TKF-DP checkpoint
(`results/K4-emwarm-top1000-2026-05-09.tar.gz`) is also attached to
GitHub release `results/K4-emwarm-top1000-2026-05-09` on
`${REPO_OWNER}/tkfdp` — use `gh release download` to pull it (see Recipe
step `(d)`).

---

## Section 2 — By-section justification

### 2a. Figures (math-paper/main.tex)

| Figure | File | Generator | Inputs |
| --- | --- | --- | --- |
| Fig 1 — TKF trajectory + lottery tickets | `math-paper/figures/tkf_trajectory_uber.pdf` | `math-paper/figures/tkf_trajectory_uber.py "M(G(G))ID(GG(I)G)" --seed 99 --lottery-seed 99` (uses `tkf_trajectory.py`) | None (synthetic) |
| Fig 2(a–c) — BDI consistency B, D, S | `bdi_consistency_B.pdf`, `bdi_consistency_D_highmu.pdf`, `bdi_consistency_S.pdf` | `math-paper/figures/fig_bdi_consistency_highmu.py` (a wrapper around `tkf-mixdom/python/experiments/fig_bdi_consistency.py`) — output emitted directly into `math-paper/figures/` | `tkfmixdom.jax.simulate.simulate.simulate_bdi_gillespie` |
| Fig 2(d–e) — TKF BDI validation | `tkf_bdi_validation_means.pdf`, `tkf_bdi_validation_shrinkage.pdf` | `tkf-mixdom/python/experiments/fig_tkf_bdi_validation.py` (output: `tkf-mixdom/python/experiments/figures/`) | `tkfmixdom.jax.simulate.mixdom_gillespie.gillespie_bdi_edge` |
| Fig 3 — K=4 coupling log-odds heatmap | `k4_coupling_score.pdf` | `analysis/scripts/plot_k4_coupling_and_doublet.py --ckpt results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz --t 1.0 --out-dir math-paper/figures` | Released ckpt `state.npz`; `tkfdp.block_likelihoods`; LG08 |
| Fig 4 — LAMA1 holmes tile | `holmes_tile_PF00053_lama1_test.pdf` | `analysis/scripts/plot_holmes_tile.py` with `--family PF00053 --pair-lama1-test --ckpt …state.npz --out math-paper/figures/holmes_tile_PF00053_lama1_test` | Released ckpt; UniProt P25391 (LAMA1); pair-cache `holmes_tile_PF00053_lama1_test_cache.json` |
| Fig 5 — PF00053 MSA composite triangle | `holmes_msa_triangle_PF00053.pdf` | `analysis/scripts/plot_holmes_msa_triangle.py --pfam-sto ~/bio-datasets/data/pfam/random100/PF00053.sto --ckpt …state.npz --out math-paper/figures/holmes_msa_triangle_PF00053` | Released ckpt; Pfam PF00053 .sto; pair-cache `holmes_msa_triangle_PF00053_cache.json` |

### 2b. Tables (math-paper/main.tex)

| Table | Source files | Generator chain |
| --- | --- | --- |
| Tab 1 — Headline aln accuracy (L<150 subset, 187 pairs) | `math-paper/results/balibase_summary_l150.{csv,md}` ; `math-paper/results/qprime_cache/<method>/*.npz` ; `math-paper/results/infinite_phmm_balibase_k4_replicaexchange.json` | `tkf-mixdom/python/experiments/Makefile.expected_balibase` (the 5 soft + 2 hard rows) + `analysis/scripts/sweep_infinite_phmm_balibase.py` (inf-PHMM row, run on the 187-pair AWS sweep, see §2d) + `analysis/scripts/l150_subset_summary.py` (rollup) |
| Tab 2 — Full BAliBASE bali3pdbm (120 fams) | `math-paper/results/balibase_summary_full.{csv,md}` ; same qprime_cache (soft methods); no inf-PHMM row | Same as Tab 1 minus the AWS sweep; `analysis/scripts/balibase_summary.py` (rollup) |

### 2c. Released checkpoint (consumed by Figs 3–5 and the AWS inf-PHMM sweep)

- `results/K4-emwarm-top1000-2026-05-09/_best_chkpt/{state.npz, meta.json, trace.json}` — produced by `experiments/exp2_pfam_v2.py` on 1000 Pfam families, K_c=4, EM-warmup (500 iters, 50 seeds), no side potentials, alpha_z=100, K_H_max=10. Best val_LL = −298.41 at outer iter 55. Config + commit-hash provenance in `MANIFEST.md`.

### 2d. AWS 187-pair MCMC sweep (Tab 1 inf-PHMM-K=4 row)

- `aws/PLAN.md` — design doc + spend / quota guidance.
- `aws/balibase_jit_primer.yaml` — one-shot SkyPilot task: spin up g5.xlarge, run one pair with --n-sweeps 100, tar `~/.cache/jax/`, push to S3 as `jax-cache/g5xl-jax-2026-05-15.tar.gz`.
- `aws/balibase_one_pair_v2.yaml` — per-pair worker (current generation). Pulls everything from `s3://${S3_BUCKET}/balibase-bundle/`.
- `aws/balibase_one_pair.yaml` — v1 (kept for archival reproducibility; was used early in the sweep).
- `aws/launch_balibase_direct.py` — direct-launch driver that bounds concurrency, skips pairs already in S3, deduplicates against running clusters.
- `aws/launch_balibase_sweep.py` — managed-jobs driver (v1; wedged at 143 concurrent; superseded by direct launcher).
- `aws/merge_balibase_results.py` — pulls all 187 per-pair JSONs + Q'-cache .npz files from S3, writes `math-paper/results/infinite_phmm_balibase_k4_replicaexchange.json` + per-family caches under `math-paper/results/qprime_cache/infinite_phmm_mcmc_K4_pdbanchor_RE/`.
- `analysis/scripts/qprime_cache_watcher.py` — local-side mirror of `~/.cache/tkf-mixdom-balibase/` → `math-paper/results/qprime_cache/`. Polls + commits per-family.
- `/tmp/ec2_stale_killer.sh` — background watchdog that terminates GPU EC2 instances running >4 hours. SHIP as `aws/ec2_stale_killer.sh` in the clean drop (currently only lives in /tmp; promote it).
- `s3://tkf-mixdom-gpu-618647024028/qprime-cache/manifest_LATEST.json` — pointer to the most recent stash; current contents have 7 methods (tkf92, mixdom_d3f1, tkf92_K20, cherryml_C20, mafft, muscle, infinite_phmm_mcmc_K4_coupled_RE) covering 120 / 22 / 98 families respectively.

### 2e. Supplement appendices (\input chain from supplement.tex)

| Appendix | Files \input'd by `supplement.tex` |
| --- | --- |
| A — BDI + TKF foundations | `tkf-mixdom/tkf/body-tkf91.tex` , `body-tkf92.tex` , `tkf92-wfst-derivation.tex` , `lhopital-limits.tex` (which itself \input's `joint-vs-conditional.tex` and `irreversible.tex`) , `score-derivatives.tex` , `math-paper/appendix-ggi.tex` |
| B — EM + composite likelihoods + variational inference | `substitution-mstep.tex`, `svb-tkf91.tex`, `svb-convergence.tex`, `body-maraschino-main.tex`, `body-tkf-inference.tex`, `varanc-presence.tex`, `varanc-bias-appendix.tex` |
| C — Recursive TKF (MixDom + recursive grammars) | `body-mixdom.tex`, `body-mixdom-inference.tex`, `mixdom-algorithms.tex`, `exploded-mixdom.tex`, `maraschino.tex` (renamed §heading), `algebraic-distillation.tex`, `svb-convergence-mixdom.tex`, `varanc-vbem.tex`, `varanc-presence-mixdom.tex`, `partition-recon.tex`, `mixdom-wfst.tex`, `grammar-elaboration.tex`, `recursive.tex` |
| D — TKF-DP (Dirichlet-process Potts coupling) | `math-paper/appendix-tkfdp.tex` |
| E — infinite Pair HMM + MCMC sampler | `math-paper/appendix-infinite-phmm.tex` |

Cross-references from `appendix-infinite-phmm.tex` to code (each one identifies a `src/tkfdp/*.py` that must be in the drop):

- `f2_scfg.py` — App E §1 reference implementation
- `aug_phmm.py` — App E §2 (1-edge)
- `aug_phmm_2edge.py` — App E §2 (2-edge)
- `mcmc_infinite_phmm.py` — App E §3 (block-resample sampler; Tab 1 row source)

### 2f. Trained MixDom / TKF92-mixture / CherryML checkpoints (Tab 1, Tab 2 soft rows)

| Method | Checkpoint | How it was trained |
| --- | --- | --- |
| TKF92 | `tkf-mixdom/python/experiments/tkf92_fitted_params.json` (also a .npz alternative) | `tkf-mixdom/python/maraschino.py fit-singlet --n-steps 5000` on the corpus |
| TKF92-K=20 mixture | `tkf-mixdom/python/pfam/tkf92_mixture_K20_train.npz` | `tkf-mixdom/python/fit_tkf92_mixture.py --K 20` |
| CherryML-C=20 | `tkf-mixdom/python/pfam/cherryml_mixture_C20_n5000.npz` | `tkf-mixdom/python/maraschino.py fit --K 20 --n-pairs 5000` |
| MixDom-d3f1 | `tkf-mixdom/python/pfam/svi_bw_d3f1_postfix_best_val.npz` | `tkf-mixdom/python/train_pfam.py --arch d3f1 --postfix …` (full Pfam-A SVI-BW run) |

These four files (plus `tkf92_fitted_params.json`) are ~50 MB total; ship in-repo or under a single tag's GitHub Release. If size is a concern, host on the same S3 bucket and add a `download_mixdom_checkpoints.sh` helper.

### 2g. Unified reconstruction benchmarks (user explicitly asked; not in main paper)

Ship the code paths + result JSONs even though the main published paper does not cite them. They live in `tkf-mixdom/python/experiments/` and `tkf-mixdom/python/tkfmixdom/jax/tree/`:

- Specs: `unified_benchmark_test_spec.json`, `unified_benchmark_hard_test_spec.json`, `unified_benchmark_long_test_spec.json`, `unified_benchmark_xhard_test_spec.json` (canonical eval specs; `_test_` files are the canonical val-uncontaminated specs going forward — see `feedback_unified_val_contamination.md` in user memory).
- Generators: `tkf-mixdom/python/experiments/fels21_reconstruction_benchmark.py` , `fels40_reconstruction_benchmark.py` , `composite_beam_benchmark.py` , `ancrec_benchmark.py` , `balibase_reconstruction_benchmark.py` , `build_unified_test_specs.py`, `build_unified_spec_stats_report.py`.
- Results JSONs: `varanc_presence_*.json`, `fels21_reconstruction_unified_*.json`, `fels40_reconstruction_unified_*.json` (under `tkf-mixdom/python/experiments/`).
- Code paths consumed: `tkfmixdom.jax.tree.varanc_presence{,_mixdom}.py` , `tree_varanc.py` , `progrec_felsenstein.py` , `composite_beam{,_jax}.py` , `recognizer.py` , `transducer.py`.
- Convention: every method runs on every entry of the spec; F1 (not accuracy) is headline; always save `pred_seq` per entry. The `${CLAUDE_AGENTS_PATH}` agent enforces this.

### 2h. Dataset fetchers (bio-datasets/)

| Dataset | Required for | Fetcher path |
| --- | --- | --- |
| Pfam | K=4 SVI training (1000 families), MixDom-d3f1 training (full Pfam-A), Fig 5 (PF00053.sto), TKF92 / CherryML cherry-counts | `bio-datasets/fetch/pfam/fetch.py` + `prepare.py` + `make_split.py` |
| BAliBASE | Tab 1, Tab 2 (the BAli3pdbm corpus), Fig 4 (BB12032 candidates), `eval_balibase.py`, `expected_pairwise_balibase.py` | `bio-datasets/fetch/balibase/fetch.py` (drive5 mirror, 21 MB) |
| OxBench | Supplement K-sweep on OxBench (referenced only obliquely in supplement; `tkf-mixdom/python/experiments/fsa_tkf92_oxbench.py` + `oxbench_tkf92*.json`) | NO FETCH SCRIPT EXISTS in bio-datasets/; data is at `~/bio-datasets/data/oxbench/{ox,oxm,oxx,qscore,ref,in,info}/`. WRITE one: a 50-line `bio-datasets/fetch/oxbench/fetch.py` that pulls from the canonical OxBench distribution (`http://www.compbio.dundee.ac.uk/manuals/oxbench/oxbench.tar.gz` or the EBI mirror). See **Open questions / flags** below. |

---

## Section 3 — Excluded categories

| Category | Why excluded |
| --- | --- |
| `.claude/`, `.claude/worktrees/` | sub-agent sandbox copies, not source of truth |
| `_old_*`, `_aborted_*`, `_partial_*`, `_v[0-9]+_` files in `math-paper/results/` and `tkf-mixdom/python/pfam/` | intermediate / superseded artefacts (e.g. `_old_rates_infinite_phmm_*`, `_partial_v2_*`, `_aborted_ladder1_*`) |
| `python/tests/` in tkf-mixdom (level5_gpu, level6_stochastic_fail, all dev smoke tests) | not cited by paper; only `tkf-dp/tests/test_balibase_*.py` are cited integration tests |
| `tkf-mixdom/misc/` (standalone TeX bodies for OTHER papers) | not \input'd by `math-paper/supplement.tex` — `elimination-chain.tex` and `ghost-usage.tex` mentioned in tkf-mixdom CLAUDE.md are NOT cited by the math paper; they describe MixDom-paper-internal proofs |
| `tkf-mixdom/python/.venv/` | bytecode + pip caches; regenerate via uv |
| Removed sims (`tree_sim.py`, `mixdom_sim.py`, `wfst_sim.py`) | already deleted per CLAUDE.md note (April 2026); do NOT resurrect |
| `tkf-dp/data/pdb_cache/` (1000+ PDB files for figure metadata) | only Fig 4 / Fig 5 need a small subset (PF00053 + maybe a few); ship a 5-line `data/fetch_pdb_cache.sh` that pulls on-demand from RCSB instead of shipping 1000 files |
| `tkf-dp/logs/` (training stdout + holmes_tile sharpening run scripts) | not load-bearing; example scripts in `logs/holmes_tile_sharpening/run_all_v*.sh` are useful as ergonomic prior art but inflate the drop |
| `tkf-dp/analysis/` MD documents (`*_evaluation.md`, `inv1/inv2 finalize`, k4_pdbanchor*, k4_pdbrestrict, re_diag) | research notes; not consumed by paper. Keep `analysis/postprocess_vs_coupled_annealing.md` and `analysis/scripts/qprime_cache_watcher.py` (load-bearing) but drop the rest. |
| `tkf-dp/experiments/exp1_*`, `exp2_compare_*`, `exp2_dca_*`, `exp2_heldout_*`, `exp2_pf00027_*`, `exp2_pfam_alpha_sweep.py`, `exp2_pfam_l2_sweep.py`, `exp2_pfam_K.py`, `exp2_potts_dp_synthetic.py`, `exp2_synthetic.py`, `exp2_val_compare*`, `exp3_*`, `em_stochastic_compare.py`, `build_pfam_full_corpus.py`, `preprocess_pfam_full_stress.py` | exploratory / dev-only; the K=4 training pipeline is `exp2_pfam_v2.py` only |
| `tkf-dp/refs/holmes_rubin_elbo.md` | working notes; drop if budget tight |
| All `*.pdf`, `*.png`, `*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.toc`, `*.out`, `*.build.log`, `combined.pdf` in math-paper/ | build outputs; regenerate via `build.sh` |
| `gravestone_evaluation.md`, `gravestone_implementation.md`, `implementation_notes.md`, `pfam_evaluation.md`, `variational_quality_evaluation.md`, `handoff_*.md` at tkf-dp top level | planning / handoff docs; not consumed by paper. Ship as a single archived `PROVENANCE.md` if you want the receipts; otherwise drop. |
| `bio-datasets/fetch/{annevo, crw, gencode, gtrnadb, ncbi, panther, proteingym, silva, tiberius, ucsc, zoonomia}/` | unused by this paper. Only ship if you want one canonical `bio-datasets` repo across all your projects (recommended; the size cost is trivial). |

---

## Section 4 — Reproduction recipe

Save as `tkf-dp/README.md` (or `docs/reproduce.md`) and link from `tkfdp.net`. Tested-in-principle against the in-tree CLAUDE.md + `docs/balibase_paired_fsa_quickstart.md`.

### (a) Clone

```bash
# Three repos. tkf-mixdom is included twice: as a submodule of tkf-dp
# (so math-paper/build.sh can resolve \input{tkf-mixdom/tkf/...}) AND
# as a standalone working clone (so `cd tkf-mixdom/python && make ...`
# works without poking around inside the submodule directory).
git clone git@github.com:${REPO_OWNER}/tkfdp.git
cd tkf-dp && git submodule update --init --recursive && cd ..
git clone git@github.com:${REPO_OWNER}/tkf-mixdom.git
git clone git@github.com:${REPO_OWNER}/bio-datasets.git
```

### (b) Python env

```bash
# tkf-mixdom's pyproject.toml is the canonical env (it pins JAX, optax,
# matplotlib, mpmath). tkf-dp imports tkfmixdom at runtime, so
# everything lives in ONE venv.
cd tkf-mixdom/python
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .                                      # installs tkfmixdom
# JAX with CUDA 12 (skip if CPU-only is OK; some figures fit in CPU memory)
uv pip install -U "jax[cuda12]"                           # https://docs.jax.dev/en/latest/installation.html

# tkf-dp itself is a "src/" layout without pyproject.toml; install as
# editable via a tiny stub:
cd ../../tkf-dp
echo '[project]
name = "tkfdp"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []
[tool.setuptools.packages.find]
where = ["src"]
include = ["tkfdp*"]
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"' > pyproject.toml
uv pip install -e .                                       # installs tkfdp from src/tkfdp/

# MUSCLE 5 + MAFFT 7 (sanity comparators in Tab 1, Tab 2)
bash scripts/fetch_aligners.sh

# AWS CLI + SkyPilot (needed for step (g) only)
uv pip install awscli skypilot[aws]
```

### (c) Fetch datasets

```bash
# Pfam top-1000 (used to retrain the K=4 ckpt from scratch)
cd ../bio-datasets
python fetch/pfam/fetch.py --random 1000 --seed 42
python fetch/pfam/prepare.py        # → ~/bio-datasets/data/pfam/random1000/PF*.sto

# Or: full Pfam-A (for MixDom-d3f1 reproduction)
python fetch/pfam/fetch_full.py
python fetch/pfam/prepare_full.py

# BAliBASE 3 (drive5 mirror, 21 MB)
python fetch/balibase/fetch.py      # → ~/bio-datasets/data/balibase/bench1.0/bali3pdbm/

# OxBench (supplement K-sweep — OPTIONAL; not in main paper)
# [Writes ~/bio-datasets/data/oxbench/]
python fetch/oxbench/fetch.py       # ← NOT YET WRITTEN; see Open questions

# Symlink the data into the project repos
ln -sfn ~/bio-datasets/data/pfam       ~/tkf-mixdom/python/data/pfam
ln -sfn ~/bio-datasets/data/balibase   ~/tkf-dp/data/balibase
```

### (d) Pull the trained checkpoints

```bash
cd ~/tkf-dp

# Released TKF-DP K=4 (consumed by Figs 3-5 and the AWS inf-PHMM sweep)
gh release download 'results/K4-emwarm-top1000-2026-05-09' \
    --repo ${REPO_OWNER}/tkfdp \
    -p 'K4-emwarm-top1000-2026-05-09.tar.gz'
tar xzf K4-emwarm-top1000-2026-05-09.tar.gz
mkdir -p results/exp2_v2_K4_top1000_tsb_emwarm/_best_chkpt
cp -r K4-emwarm-top1000-2026-05-09/_best_chkpt/* \
      results/exp2_v2_K4_top1000_tsb_emwarm/_best_chkpt/

# MixDom-d3f1 + TKF92-K=20 + CherryML-C=20 (consumed by Tab 1, Tab 2 soft rows)
# Option A: pull from a future GitHub release (preferred; we have not cut it yet)
#   gh release download 'mixdom-checkpoints-2026-05' --repo ${REPO_OWNER}/tkf-mixdom -p '*.npz'
# Option B: pull from S3
mkdir -p ~/tkf-mixdom/python/pfam
aws --profile tkf-gpu s3 cp \
    s3://tkf-mixdom-gpu-618647024028/mixdom-checkpoints/svi_bw_d3f1_postfix_best_val.npz \
    ~/tkf-mixdom/python/pfam/svi_bw_d3f1_postfix_best_val.npz
aws --profile tkf-gpu s3 cp \
    s3://tkf-mixdom-gpu-618647024028/mixdom-checkpoints/tkf92_mixture_K20_train.npz \
    ~/tkf-mixdom/python/pfam/tkf92_mixture_K20_train.npz
aws --profile tkf-gpu s3 cp \
    s3://tkf-mixdom-gpu-618647024028/mixdom-checkpoints/cherryml_mixture_C20_n5000.npz \
    ~/tkf-mixdom/python/pfam/cherryml_mixture_C20_n5000.npz
# (`tkf92_fitted_params.json` is committed; no pull needed.)

# 187-pair AWS sweep results (consumed by Tab 1 inf-PHMM-K=4 row)
# Either re-run the sweep (step (g)) or pull the cached outputs:
python aws/merge_balibase_results.py \
    --prefix balibase-runs/v11-aws-canonical \
    --out math-paper/results/infinite_phmm_balibase_k4_replicaexchange.json \
    --qprime-cache-dir math-paper/results/qprime_cache/infinite_phmm_mcmc_K4_pdbanchor_RE
```

### (e) Analytic figures (no MCMC; ~minutes on CPU)

```bash
cd ~/tkf-dp

# Fig 1: TKF trajectory + Galton-Watson + lottery tickets
python math-paper/figures/tkf_trajectory_uber.py "M(G(G))ID(GG(I)G)" \
    --seed 99 --lottery-seed 99 \
    --out math-paper/figures/tkf_trajectory_uber

# Fig 2 row 1: BDI consistency (the wrapper extends the regime grid)
cd math-paper/figures && python fig_bdi_consistency_highmu.py && cd ../..

# Fig 2 rows 2-3: TKF BDI validation (Gillespie, ~5 min CPU at default n_sim)
cd ~/tkf-mixdom/python
JAX_ENABLE_X64=1 .venv/bin/python experiments/fig_tkf_bdi_validation.py
# Copy outputs into math-paper/figures/ (the paper's \graphicspath includes BOTH locations,
# so this is only needed if math-paper/figures/ is the canonical drop folder)
cp experiments/figures/tkf_bdi_validation_{means,shrinkage}.pdf \
    ~/tkf-dp/math-paper/figures/

# Fig 3: K=4 coupling log-odds
cd ~/tkf-dp
python analysis/scripts/plot_k4_coupling_and_doublet.py \
    --ckpt results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz \
    --t 1.0 \
    --out-dir math-paper/figures
```

### (f) BAliBASE soft-method scoring (Tab 1 + Tab 2)

```bash
cd ~/tkf-mixdom/python
export JAX_ENABLE_X64=1

# Run all 6 method rows (5 soft + 2 hard) on the full bali3pdbm corpus:
make -f experiments/Makefile.expected_balibase all \
    BALIBASE_DIR=~/bio-datasets/data/balibase/bali3pdbm \
    TKF92_JSON=experiments/tkf92_fitted_params.json \
    TKF92_MIX_NPZ=pfam/tkf92_mixture_K20_train.npz \
    CHERRYML_NPZ=pfam/cherryml_mixture_C20_n5000.npz \
    MIXDOM_NPZ=pfam/svi_bw_d3f1_postfix_best_val.npz \
    MIX_PAIR_CHUNK=8

# Outputs in experiments/expected_balibase/:
#   expected_balibase_{tkf92,tkf92_K20,cherryml_C20,mixdom_d3f1,mafft,muscle}{.,_l150.,_withsps.}json

# L<150 subset rollup → math-paper/results/balibase_l150_summary.json + .csv/.md
cd ~/tkf-dp
python analysis/scripts/l150_subset_summary.py \
    --indir ~/tkf-mixdom/python/experiments/expected_balibase \
    --out math-paper/results/balibase_summary_l150
python analysis/scripts/balibase_summary.py \
    --indir ~/tkf-mixdom/python/experiments/expected_balibase \
    --out math-paper/results/balibase_summary_full
```

### (g) AWS 187-pair MCMC sweep (Tab 1 inf-PHMM-K=4 row)

```bash
cd ~/tkf-dp
export AWS_PROFILE=tkf-gpu

# (g.1) Bundle BAliBASE + ckpt + venv + JAX cache and push to S3
#   (one-time, ~1-2 GB)
tar czf /tmp/balibase-bundle.tar.gz \
    ~/bio-datasets/data/balibase/bali3pdbm \
    results/K4-emwarm-top1000-2026-05-09/_best_chkpt
aws s3 cp /tmp/balibase-bundle.tar.gz \
    s3://tkf-mixdom-gpu-618647024028/balibase-bundle/

# (g.2) Prime the JAX cache on a g5.xlarge and stash it to S3 (~10 min)
sky launch -c balibase-prime aws/balibase_jit_primer.yaml -y --down

# (g.3) Run the watchdog (terminates GPU EC2 instances running >4h)
nohup bash aws/ec2_stale_killer.sh > /tmp/ec2_stale_killer.log 2>&1 &

# (g.4) Launch all 187 pairs (~4-5 h wall, ~$85 spot spend)
python aws/launch_balibase_direct.py \
    --yaml aws/balibase_one_pair_v2.yaml \
    --max-concurrent 64

# (g.5) Merge S3 outputs into the per-pair JSON + per-family Q'-cache
python aws/merge_balibase_results.py \
    --prefix balibase-runs/v11-aws-canonical \
    --out math-paper/results/infinite_phmm_balibase_k4_replicaexchange.json \
    --qprime-cache-dir math-paper/results/qprime_cache/infinite_phmm_mcmc_K4_pdbanchor_RE

# Kill the watchdog
pkill -f ec2_stale_killer.sh
```

### (h) Stage Q' into FSA pipeline + downstream MSA scoring

```bash
cd ~/tkf-dp
python analysis/scripts/downstream_fsa_on_cached_qprime.py \
    --qprime-dir math-paper/results/qprime_cache/infinite_phmm_mcmc_K4_pdbanchor_RE \
    --out math-paper/results/k4_re_downstream_fsa.json
```

### (i) LAMA1 worked example (Figs 4 + 5)

```bash
cd ~/tkf-dp

# Fig 4: holmes-tile on the LAMA1 EGF-like pair (PF00053 held out from K=4 training)
python analysis/scripts/plot_holmes_tile.py \
    --pfam-sto ~/bio-datasets/data/pfam/random100/PF00053.sto \
    --family PF00053 \
    --pair-name-x LAMA1_HUMAN/1452-1506 \
    --pair-name-y LAMA1_HUMAN/1090-1147 \
    --ckpt results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz \
    --cache-json math-paper/figures/holmes_tile_PF00053_lama1_test_cache.json \
    --out math-paper/figures/holmes_tile_PF00053_lama1_test

# Fig 5: PF00053 MSA composite triangle (8-sequence-pair composite).
# The --seq-subset pins the exact 16 chosen sequence indices so the
# cache hits and the figure renders without re-running MCMC (~5 min);
# omit --seq-subset and --cache-json to re-run from scratch (~30 min).
python analysis/scripts/plot_holmes_msa_triangle.py \
    --pfam-sto ~/bio-datasets/data/pfam/random100/PF00053.sto \
    --ckpt results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz \
    --family-label "PF00053 (Laminin EGF-like)" \
    --n-pairs 8 \
    --seq-subset 26,5,6,50,63,38,35,69,56,70,60,62,53,66,44,13 \
    --cache-json math-paper/figures/holmes_msa_triangle_PF00053_cache.json \
    --out math-paper/figures/holmes_msa_triangle_PF00053
```

### (j) Compile the PDF

```bash
cd ~/tkf-dp/math-paper
./build.sh --combined --open       # main + supplement → combined.pdf
```

---

## Section 5 — Open questions / flags

- **`bio-datasets/fetch/oxbench/fetch.py` does NOT exist.** Per `bio-datasets/README.md` (the contract), every `data/<dataset>/` must have a parallel `fetch/<dataset>/fetch.py`. The OxBench data is present under `~/bio-datasets/data/oxbench/{ox,oxm,oxx,qscore,ref,in,info}/` but its fetcher is missing. The supplement K-sweep on OxBench (`tkf-mixdom/python/experiments/fsa_tkf92_oxbench.py`, `oxbench_tkf92*.json`) WILL break without it. Action item for the clean drop: write a ~50-line fetcher pulling from one of (i) Geoff Barton lab's canonical mirror at `http://www.compbio.dundee.ac.uk/manuals/oxbench/`, (ii) the older EBI mirror, or (iii) one of the curated re-distributions in `bench/oxbench/` of the drive5 BAliBASE bundle (which already gets pulled by `bio-datasets/fetch/balibase/fetch.py`).

- **`aws/ec2_stale_killer.sh` is currently at `/tmp/ec2_stale_killer.sh`** — outside the repo. Promote it into `aws/` before the drop; without it, the AWS sweep recipe step (g.3) breaks.

- **`tkf-mixdom/python/pfam/*` checkpoint hosting strategy is not finalised.** The three named NPZ files (`svi_bw_d3f1_postfix_best_val.npz`, `tkf92_mixture_K20_train.npz`, `cherryml_mixture_C20_n5000.npz`) are ~1 GB total but are NOT under any GitHub release yet. The Recipe step (d) Option B (S3 path) is therefore a placeholder; the user needs to either (1) cut a `tkf-mixdom` release and attach them as artefacts, or (2) push them to S3 at a fixed path. Recommend (1).

- **`bio-datasets/fetch/common.py` lives one level up from where the bio-datasets `README.md` suggests** (it's at `fetch/common.py`, not at the repo root). All the `from common import …` lines in the per-dataset fetchers do `sys.path.insert(0, ...parent)` so this works; just noting it for the clean-drop reader who might look for `bio-datasets/common.py` at the top level and be confused.

- **`scripts/fetch_aligners.sh` references the user's HOME (`~/.local/bin/`, `~/.local/share/mafft/`).** Works on Linux + macOS without sudo; the clean-drop reader on a read-only filesystem (e.g. shared HPC) will need to override `PREFIX` or skip and rely on a system MUSCLE / MAFFT in `$PATH`.

- **The `_best_chkpt/state.npz`-only download path assumes the reader doesn't want the rolling `_chkpt/` snapshots or the per-MSA `.npy` traces.** Reasonable default; flagged in case anyone wants full training-trajectory provenance.

- **The supplement's `\input{tkf-mixdom/tkf/recursive.tex}` block requires the macro-shadowing dance in `supplement.tex`** (the `\let\OrigTok\tok` block). If `tkf-mixdom/tkf/recursive.tex` changes its `\newcommand` list, this block needs updating. Pin the submodule commit explicitly in `.gitmodules` and warn the clean-drop reader before they `git submodule update --remote`.

- **`math-paper/figures/AA_EVOLUTION_RECIPE.md` documents figures that are NOT in `main.tex`.** Five `aa_evolution_*.pdf` figures (and stacked variants) live in `math-paper/figures/` but are not `\includegraphics`'d anywhere. Drop the PDFs from the clean repo; keep the recipe + generator scripts (`plot_aa_evolution*.py`) as a supplementary appendix — they may end up in the supplement in a future revision.

- **`tkf-dp/refs/holmes_rubin_elbo.md`** is a working note, not a paper input. Drop without consequence.

- **No `tkf-dp/pyproject.toml`.** Recipe step (b) writes one inline. If you'd rather not have ad-hoc bootstrapping in the recipe, commit a real `pyproject.toml` into the clean drop. (The reason there isn't one already is that the in-tree project is loose; the user has been running scripts directly with the `tkfmixdom` venv on PYTHONPATH.)

- **`f2_scfg.py`, `aug_phmm.py`, `aug_phmm_2edge.py`** are cited in App E but I have not verified they are actively maintained (their smoke tests exist; the main MCMC sampler `mcmc_infinite_phmm.py` is the actively-used one for the paper's results). Include all three — the appendix's text identifies them by name.
