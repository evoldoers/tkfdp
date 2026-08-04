# Paper-2b pair-substitution models — held-out comparison

Converged held-out per-count transition log-likelihood (Sum_t n[:,:,t] . log P(t),
val = family shard 4), pi ML-fit jointly with the fluxes. Higher = better.

Complexity column = **net free dimension = #flux params + #pi params − #constraints**
(so constrained models like Lumpable are counted honestly, not by their raw orbit count).
pi DOF: 209 for the exchangeable models (symmetric 400-state stationary), 399 for the
asymmetric Renewal / Half-lumpable models. Reference ceilings: full exchangeable reversible flux =
40,090; full NON-exchangeable reversible flux = C(400,2) = 79,800.

| model | net free dim | trRosetta norate | trRosetta +rate | af_full |
|---|---:|---:|---:|---:|
| Synchronized | 40,299 | **-2.2315** | **-2.1535** | **-1.3158** |
| Coupled | 4,009 | -2.2385 | -2.1550 | -1.3186 |
| CherryML Q2 (external) | 1‡ | -2.2327 | -2.1627 | -1.3205 |
| Metropolis (Hastings) | 399 | -2.2422 | -2.1576 | -1.3214 |
| Metropolis (Barker) | 399 | -2.2424 | -2.1572 | -1.3214 |
| Metropolis (sqrt) | 399 | -2.2427 | -2.1572 | -1.3217 |
| Half-lumpable (1-sided) | ~72,599* | -2.2436 | -2.1648 | -1.3255 |
| Metropolis (GTR) | 399 | -2.2581 | -2.1719 | -1.3328 |
| Lumpable (2-sided) | 33,079† | -2.3576 | -2.2817 | -1.4018 |
| Renewal | 4,389 | -2.5509 | -2.4797 | -1.5195 |
| Renewal+GTR | 779 | -2.5533 | -2.4820 | -1.5223 |
| Renewal+GTR+same | 589 | -2.5556 | -2.4829 | -1.5235 |

† Lumpable (2-sided) = 40,090 flux + 190 (g) − 7,410 lumpability constraints + 209 pi
 = 33,079. The lumpability is imposed on EXCHANGEABLE orbits, so the single row-marginal
 constraint set gives lumpability to BOTH coordinates.
‡ CherryML Q2 is a fixed external 400x400 matrix; only 1 param (global rate scale) is fit.
* Half-lumpable (1-sided) = drop exchangeability, keep the cpt1 row-marginal
 (lumpability-to-cpt1) only: 79,800 non-exch flux − 7,410 constraints + 209 (symmetric pi,
 from swap-symmetrised counts) = ~72,599 net. THE KEY RESULT: it beats 2-sided Lumpable by
 +0.107/count (trRosetta) and sits in the Coupled/Metropolis pack -- the SECOND lumpability
 constraint costs ~10x the first (one-sided ~0.012 below Synchronized, two-sided ~0.119).
 It does not beat Synchronized/Coupled because the true coupled process is not lumpable at
 all, so any lumpability constraint costs something. (ALM reaches ~2e-2 relative
 lumpability violation; 6->20 iter gain was 0.0015, so near plateau.)

The net-dim column exposes the real lesson: **Lumpable's poor fit is NOT
under-parameterization** (33k free params, nearly Synchronized's 40k) -- it is the
two-sided lumpability CONSTRAINT forcing a costly structure. Per-count LL is comparable
only *within* a column (each corpus has a different count base / tau structure).

## Correctness note (adversarial audit, 2026-07)

The val column was corrected after an audit found the held-out LL was scored with the
empirical (not ML-fitted) stationary pi at `fit_pair_models.py:490`, corrupting
`eig_rev` for models whose fitted pi drifts from empirical (GTR, Lumpable) -- the
tell-tale being an impossible GTR val>train. Fix: pass `r["pi"]`. The correction makes
the constrained models slightly WORSE (GTR/Lumpable by ~0.007-0.010/count), which
*strengthens* every conclusion, and removes the val>train anomaly. All five headline
conclusions were then independently re-verified against the corrected numbers:
(i) Synchronized wins on all 3 corpora; (ii) one-sided lumpability costs ~9-11x LESS
than two-sided; (iii) Hastings~=Barker~=sqrt > GTR (gap widened to 0.011-0.015);
(iv) CherryML beaten out-of-domain on af_full by Synchronized+Coupled;
(v) Renewal worst everywhere. (Minor audit items, claim-preserving: Lumpable/Half-lumpable
train LL returns the last not best ALM iterate; CherryML uses scipy.expm not the eig
backend; Renewal/Half-lumpable symmetrise val counts -- all pessimistic or negligible.)

## Models

- **Synchronized** — all Klein-4 orbit fluxes free (40,090). Most capacity; wins
  train and held-out everywhere (no overfitting — nesting holds).
- **Coupled** — single-transition orbits only (3,800). Free flux.
- **Metropolis (kernel)** — shared single-site exchangeability S with
  Q_xy = S_xy f(pi_x,pi_y): sqrt `sqrt(pi_y/pi_x)`, gtr `pi_y`, barker
  `pi_y/(pi_x+pi_y)`, hastings `min(1,pi_y/pi_x)`. Hastings≈Barker≈sqrt; GTR worst.
- **CherryML Q2** — external published 400x400 co-evolution matrix (Zenodo
  7830072), fixed, granted one global rate scale. Strong on its home trRosetta
  (near-in-sample), beaten out-of-domain on af_full by Synchronized/Coupled.
- **Lumpable** — single-transition + exact lumpability (marginals Markov). Far
  back: the lumpability constraint is costly.
- **Renewal** family (a.k.a. field+renewal;
  experiments/fit_renewal.py) — cpt1 = autonomous free GTR MASTER (marginal GTR =>
  lumpable-to-cpt1 by construction); cpt2 = subservient partner that INSTANTLY
  resamples from pi(x2|x1) on every cpt1 jump (renewal dual-transition), evolving by
  its own single transitions in between. One-way, reversible w.r.t. an asymmetric
  pi_joint. Three nested variants by cpt2 single-transition freedom:
    - **Renewal** (4,389): cpt2 = free reversible flux per field x1 (20x190=3800).
    - **Renewal+GTR** (779): cpt2 = one GTR, own S2 != S1.
    - **Renewal+GTR+same** (589): cpt2 = GTR reusing cpt1's S1.
  WORST models everywhere, ~0.2 below Lumpable, and the whole family clusters within
  ~0.005/count: freeing cpt2 all the way to 3,800 fluxes buys almost nothing, and
  over-constraining it to +GTR+same costs almost nothing. The binding constraint is
  the ALWAYS-ON, RANK-1 RENEWAL: on each cpt1 jump cpt2 is resampled from a fixed
  distribution (forgetting its pre-jump state), which is FAR stronger than
  lumpability-to-cpt1 requires. Fitted pi ~4-6% asymmetric.
- **Half-lumpable** (one-sided lumpable; experiments/fit_half_lumpable.py) — the
  correct minimal lumpability-to-cpt1 model: the exact-lumpable ALM
  (fit_pair_models.mstep_lumpable) on TIME-REVERSAL orbits (not Klein-4), so the cpt1
  row-marginal constraint (Sum_l Q[(i,j),(k,l)] = g(i,k)) stays one-sided; the
  cpt1-jump kernel on cpt2 is a FREE reversible coupling (not the rank-1 renewal) and
  cpt2's own transitions are free. RESULT: beats 2-sided Lumpable by +0.107/count and
  lands in the Coupled/Metropolis pack -- the second lumpability constraint costs ~10x
  the first. Confirms "fewer constraints -> better" (vs Lumpable) while showing any
  lumpability costs something (the coupled process is not lumpable). Contrast with
  Renewal: SAME lumpability-to-cpt1, differing ONLY in the cpt1-jump kernel on cpt2 --
  free coupling (Half-lumpable, ~-2.24) vs rank-1 renewal (Renewal, ~-2.55). How cpt2
  remodels on a field jump is worth ~0.3/count.

## Reproduce

    # fitted models (pi/Q/flux persisted to *_params.npz)
    python experiments/fit_pair_models.py --corpus data/cherry_counts_trrosetta \
        --models synchronized,coupled,metropolis_barker,metropolis_hastings,metropolis_sqrt,metropolis_gtr \
        --iters 80 --out results/pair_models/trrosetta_converged.json
    python experiments/fit_pair_models.py --corpus data/cherry_counts_trrosetta \
        --models lumpable --iters 15 --out results/pair_models/lumpable_trrosetta.json
    python experiments/eval_cherryml_matrix.py         # CherryML Q2
    python experiments/fit_renewal.py                  # Renewal family
    python analysis/scripts/pair_model_pi_analysis.py \
        results/pair_models/trrosetta_rate_converged_params.npz --outdir analysis/figures
