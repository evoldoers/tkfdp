# Paper-2 coupled-substitution rates

Fitted rate matrices for the coupled (two-site) amino-acid substitution models of
**paper 2** (`psb-paper/paper2-coupled.tex`, held-out comparison `tab:pairfit`):
the **Exchangeable**, **Coupled**, **Metropolis** (four acceptance kernels) baselines
and the **Metropolis-mixture** (the paper's coupled coevolution model), all with
Gamma+Invariant rate heterogeneity, fit on the per-contact trRosetta corpus.

Every model is a reversible CTMC on the `A^2 = 400`-state joint of two contacting
sites (`A = 20` amino acids). State `x = 20*i + j` is the ordered pair `(i, j)` with
`i, j` indexing the alphabet `ACDEFGHIKLMNPQRSTVWY`. `Q[x, y]` is the instantaneous
rate from pair-state `x = (i,j)` to `y = (k,l)`; `pi` is the joint stationary.

## What's in the release

| paper name (`tab:pairfit`) | code `--model` | file | free params |
|---|---|---|---|
| Exchangeable (`\Exchm`)          | `synchronized`        | `baselines/synchronized.npz`        | flux 40,090 + pi 209 |
| Coupled (`\Coupledm`)            | `coupled`             | `baselines/coupled.npz`             | flux 3,800 + pi 209 |
| Metropolis (Barker)              | `metropolis_barker`   | `baselines/metropolis_barker.npz`   | S 190 + pi 209 |
| Metropolis (sqrt)                | `metropolis_sqrt`     | `baselines/metropolis_sqrt.npz`     | S 190 + pi 209 |
| Metropolis (Hastings)            | `metropolis_hastings` | `baselines/metropolis_hastings.npz` | S 190 + pi 209 |
| Metropolis (GTR)                 | `metropolis_gtr`      | `baselines/metropolis_gtr.npz`      | S 190 + pi 209 |
| Metropolis mixture (K=8)         | K=8 mixture           | `mixture/components_K8.npz`         | S 190 + 8x pi + weights |
| Metropolis mixture (K=4)         | K=4 mixture           | `mixture/components_K4.npz`         | S 190 + 4x pi + weights |
| Metropolis mixture (K=8, transposed) | 2P=8 swap-pair    | `mixture/asym_gammaI_2P8.npz`       | S 190 + 4x pi + weights |
| Metropolis mixture (K=4, transposed) | 2P=4 swap-pair    | `mixture/asym_gammaI_2P4.npz`       | S 190 + 2x pi + weights |

Each `.npz` has a sibling `.json` with the scalar fit summary (val/train per-count LL,
alpha, p_inv, rate grid, iters).

### `.npz` contents

**Baselines** (`baselines/*.npz`): `Q` (400,400 rate matrix), `F` (400,400 symmetric
flux), `pi` (400,), `S` (20,20 shared exchangeability; zeros for the flux models
`synchronized`/`coupled`), `rate_vals`/`rate_weights` (Gamma+I grid; bin 0 = invariant),
`alpha` (Gamma shape), `p_inv`, `tau` (32 geometric branch-length bin centres),
`alphabet`, `val_per_count_ll`, `train_per_count_ll`.

**Symmetric mixture** (`mixture/components_K*.npz`): `pis` (K,400 per-class stationary),
`S` (20,20 shared free exchangeability), `weights` (K,), `mi_pi` (K, per-class stationary
mutual information), `tau`, `K`, `val_per_count_ll`, `train_per_count_ll`. Each class
generator is `Q_c = metropolis_sqrt(S, pi_c)`.

**Transposed mixture** (`mixture/asym_gammaI_2P*.npz`): `pis` (P,400 per swap-pair),
`S` (20,20), `Wp` (P,) swap-pair weights. Each swap-pair enters the mixture twice, with
the two site orientations `pi_c(a,b)` and `pi_c(b,a)` sharing one parameter set.

To rebuild a generator from a mixture component: `Q_c = S[i,k] * sqrt(pi_c(k,l)/pi_c(i,j))`
on single-residue transitions (one of `i->k` or `j->l`), 0 otherwise; diagonal set so rows
sum to 0. (Helper: `fit_pair_models._met_Q(S, pi_c.reshape(20,20), "sqrt")`.)

## Held-out fit (per-count log-likelihood on the trRosetta validation split)

Nats per transition count on the held-out family split (higher = better). `MI_stat` is the
stationary mutual information of the joint `pi` (mixture rows: mixture-weighted mean ± s.d.
over the classes). All models use Gamma(4)+Invariant rate heterogeneity.

| model | train LL | val LL | MI_stat | free params |
|---|---:|---:|---:|---:|
| Metropolis mixture (K=8)              | −2.5557 | **−2.5316** | 0.089 ± 0.070 | 190 + 8×209 |
| Metropolis mixture (K=8, transposed) | −2.5575 | −2.5332 | 0.046 ± 0.028 | 190 + 4×209 |
| Metropolis mixture (K=4)              | −2.5761 | −2.5515 | 0.044 ± 0.019 | 190 + 4×209 |
| Metropolis mixture (K=4, transposed) | −2.5820 | −2.5573 | 0.039 ± 0.005 | 190 + 2×209 |
| Exchangeable                         | −2.6093 | −2.5847 | 0.025 | 40,090 + 209 |
| Coupled                              | −2.6118 | −2.5866 | 0.025 | 3,800 + 209 |
| Metropolis (Barker)                  | −2.6154 | −2.5900 | 0.026 | 190 + 209 |
| Metropolis (sqrt)                    | −2.6182 | −2.5929 | 0.026 | 190 + 209 |
| Metropolis (Hastings)                | −2.6190 | −2.5937 | 0.022 | 190 + 209 |
| Metropolis (GTR)                     | −2.6306 | −2.6052 | 0.012 | 190 + 209 |

These values are what `tab:pairfit` reports — the mixture rows and the single-component
baselines both match the paper. The baselines are **fit to convergence** here (150 EM
iters, tol 1e-6). (An earlier fixed-iteration fit stopped the slow-converging
high-capacity flux models — Exchangeable/Coupled — and GTR ~0.005–0.007/count short of
their optimum, the caveat documented in `results/pair_models/pair_models_summary.md`;
`tab:pairfit` was updated to these converged values alongside this release.) The model
ranking is unchanged and the mixture wins by ~0.05/count.

## Reproduce

All fits use the **per-contact trRosetta corpus** with a held-out family split
(`--val-frac 0.2 --seed 0`), LG08 warm-init for the shared exchangeability `S`, and a
per-cluster **Gamma(4)+Invariant** rate latent. Run from the repo root.

```bash
# 0. Build the per-contact corpus (structural contacts Cb<8A, |i-j|>=7) from the
#    CherryML trRosetta training set (data/cherryml_trrosetta/training_set).
#    Sharded + resumable; --merge concatenates the shards into counts.npz.
for s in 0 1 2 3 4 5; do
  python experiments/build_per_contact_corpus.py --nshards 6 --shard $s
done
python experiments/build_per_contact_corpus.py --merge   # -> data/per_contact_trrosetta/counts.npz

# 1. Metropolis-mixture, symmetric (the paper's coupled coevolution model):
python experiments/fit_coupling_mixture_rateI.py --K 8 --gamma-cats 4 --seed 0 \
    --val-frac 0.2 --out results/mixture_rateI_joint/components_K8.json
python experiments/fit_coupling_mixture_rateI.py --K 4 --gamma-cats 4 --seed 0 \
    --val-frac 0.2 --out results/mixture_rateI_joint/components_K4.json

# 2. Metropolis-mixture, transposed (non-exchangeable swap-pair variant):
python experiments/fit_coupling_mixture_asym_rateI.py --twoP 8 --gamma-cats 4 --seed 0 \
    --val-frac 0.2 --out results/mixture_asym_rateI/asym_gammaI_2P8.json
python experiments/fit_coupling_mixture_asym_rateI.py --twoP 4 --gamma-cats 4 --seed 0 \
    --val-frac 0.2 --out results/mixture_asym_rateI/asym_gammaI_2P4.json

# 3. Single-component baselines (Exchangeable / Coupled / Metropolis kernels):
for m in synchronized coupled metropolis_barker metropolis_sqrt metropolis_hastings metropolis_gtr; do
  python experiments/fit_pair_baseline_rateI.py --model $m --gamma-cats 4 --seed 0 \
      --val-frac 0.2 --em-iters 150 --out results/pair_models/baseline_rateI/$m.json
done
```

The baseline fitter `experiments/fit_pair_baseline_rateI.py` is a K=1 special case of the
mixture EM: it plugs `fit_pair_models`' per-model M-steps (orbit-flux for
Exchangeable/Coupled; shared-`S` Metropolis for the kernels) into the Gamma+I EM
scaffolding of `fit_coupling_mixture_rateI` (identical corpus / split / rate grid), so all
rows in `tab:pairfit` share one footing.

## Load

```python
import numpy as np
d = np.load("baselines/synchronized.npz")
Q, pi = d["Q"], d["pi"]              # 400x400 rate matrix, 400-vector stationary
# transition probs over branch length t (via eigendecomposition of the reversible Q):
import scipy.linalg as sla
P = sla.expm(Q * t)                  # P[x,y] = P(pair y at time t | pair x at 0)
```
