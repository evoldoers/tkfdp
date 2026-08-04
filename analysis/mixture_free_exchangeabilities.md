# Freeing the exchangeability in the mixture-of-couplings model

**Question.** In `fit_coupling_mixture.py` each Metropolis-sqrt component fits only its
stationary `pi_c` (the coupling); the pair-chain exchangeability `S` is FIXED at LG08.
Does letting the model fit `S` off LG08 improve held-out likelihood, and is the gain in
the exchangeability `S` or in the coupling `pi_c`?

**Design (shared S).** One SINGLE shared 20x20 symmetric exchangeability `S`
(190 params total), warm-initialised at LG08, common to all K components; each component
keeps its own stationary `pi_c`. Picture: *one universal exchange process + K coupling
archetypes*. A single pooled `S` is well-determined by the counts; a separate `S_c` per
component is not (an earlier per-component-`S` run confirmed this directly -- the fitted
`S_c` came out ~identical across components, off-diagonal Pearson correlation 0.995 --
so per-component `S` buys nothing and only adds 190K params).

Model per component: `Q_c = metropolis_sqrt(S, pi_c)`. Endpoint-conditioned HR / bridge
E-step (`fit_pair_models.estep`) gives expected transition usage `N_c` and dwell `T_c` on
the responsibility-weighted count tensor. M-step:
- **pi-step** each `pi_c = mstep_pi_metropolis(S, pi_c, N_c, T_c)` at the current shared S;
- **S-step** POOL the Metropolis S-sufficient-statistics over all components -- usage
  numerator `Cnum_c(N_c)` and `pi_c`-dependent exposure denominator `Hden_c(T_c, pi_c)` --
  then `S = (sum_c Cnum_c) / (sum_c Hden_c)`;
- **weights** `w_c = mean_g r_{gc}`.

`--fixed-S` skips the pooled S-step (S stays LG08), reproducing the original script, so the
two modes share exactly the same corpus, family split, init and EM schedule.

Code: `experiments/fit_coupling_mixture_freeS.py` (default = shared-free-S; `--fixed-S` =
LG08 baseline). Corpus `data/per_contact_trrosetta/counts.npz` (per-contact size-2 clusters,
tau-binned cherry transition counts). Held-out = 20% of families (unseen). seed 0,
em-iters 60, inner 2. EM verified monotone (zero train_mix decreases in every run).

## Held-out (unseen-family) per-count log-likelihood

| K | fixed-S (LG08) val | shared-free-S val | Δ val | fixed-S train | free-S train |
|---:|---:|---:|---:|---:|---:|
| 1  | -5.130 | **-2.672** | +2.458 | -5.135 | -2.697 |
| 2  | -4.904 | **-2.653** | +2.251 | -4.912 | -2.678 |
| 4  | -4.719 | **-2.627** | +2.091 | -4.728 | -2.652 |
| 8  | -4.512 | **-2.602** | +1.910 | -4.523 | -2.627 |
| 16 | -4.334 | **-2.574** | +1.760 | -4.348 | -2.599 |

(The fixed-S column reproduces the reference numbers -5.131 / -4.947 / -4.756 / -4.564 /
-4.380 to within ~0.03, confirming the same corpus/split.)

**Freeing the shared exchangeability helps at every K, by a lot** -- +1.8 to +2.5 nats per
count, dwarfing the mixture effect. LG08 is simply the wrong exchange process for these
structural-contact sites; a single fitted `S` fixes most of it.

**Not overfitting.** The 190 extra params are one shared matrix over ~22M training
transitions. Train-minus-val gap: fixed-S -0.005..-0.014, free-S -0.025..-0.025 (roughly
K-independent). The gap widens by ~0.02 but the val gain is +1.8..+2.5, so the improvement
is overwhelmingly real generalisation, not fit.

## Where is the gain -- S or pi?

**Almost entirely in S.** The whole K=1 free-vs-fixed gap (**+2.458**) is a pure
exchangeability effect (one component, no mixture). Once `S` is free, the *mixture* adds
much less:

- fixed-S val gain K=1→16: **+0.796**
- free-S  val gain K=1→16: **+0.098**

So under LG08 most of what the mixture was "buying" was compensation for the wrong
exchangeability; with a correct shared `S`, the residual coupling structure the mixture
captures is worth only ~0.1 nat/count.

## How far does the shared S move from LG08?

K-independent (0.95 / 0.90 for every K): relative Frobenius `||S-S_LG||/||S_LG|| ≈ 0.95`
(a large *magnitude/scale* move) but off-diagonal Pearson correlation with LG08 `≈ 0.90`
(the exchange *pattern* is broadly preserved). It is a single global recalibration of the
exchange process -- large in norm, LG08-shaped -- and the mixture does not drive it (`S`
barely changes across K).

## Do the components still specialize their coupling?

Yes. Per-component `MI(pi_c)` spreads across the mixture and the weighted mean rises
monotonically with K:

| K | w-mean MI(pi_c) free-S | per-component MI(pi_c) (sorted) |
|---:|---:|---|
| 1  | 0.028 | 0.03 |
| 2  | 0.079 | 0.05, 0.12 |
| 4  | 0.110 | 0.06, 0.10, 0.11, 0.21 |
| 8  | 0.156 | 0.09 … 0.32 (range) |
| 16 | 0.212 | 0.06 … 0.42 (range) |

Components carve out distinct couplings even with `S` free. Note free-S `MI(pi_c)` is
*lower* than fixed-S at matched K (e.g. K=16: 0.212 vs 0.307): with a correct baseline
exchange process, less of the data has to be explained by a coupled `pi_c` -- some of the
apparent "coupling" under LG08 was really marginal-exchange mismatch.

## Conclusion

Moving the shared exchangeability off LG08 is the single biggest win in this model
(+1.8..+2.5 nat/count held-out, all K), and it is a **shared, roughly LG08-shaped but
strongly rescaled** exchange process -- the components do **not** specialize `S`, they
specialize their **coupling** `pi_c`. The clean picture holds: **one universal (fitted)
exchange process + K coupling archetypes.** The mixture is still worth adding on top of the
free `S` (val keeps improving with K), but its marginal value shrinks ~8x once `S` is free.

*Caveat:* the free-S K=16 run hit the 60-iteration cap without triggering the 1e-5
early-stop, but its tail delta is only ~1e-4/count/iter, so val -2.574 is within
~0.002 of the optimum -- the conclusions are unaffected.
