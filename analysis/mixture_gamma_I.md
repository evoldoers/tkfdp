# Gamma+I rate heterogeneity on top of the shared-free-S coupling mixture

Status: PENDING RESULTS (draft method; tables filled once the sweep completes).

Script: `experiments/fit_coupling_mixture_rateI.py`
Corpus: `data/per_contact_trrosetta/counts.npz` (per-contact size-2 clusters, tau-binned
transition counts n_g(pf, pt, tb)). Held-out split: unseen families (val-frac 0.2, seed 0),
identical to `fit_coupling_mixture_freeS.py`.

## What this adds

`fit_coupling_mixture_freeS.py` fits ONE shared free single-site exchangeability S
(warm-init LG08) plus K pairing classes, each a symmetric joint stationary pi_c (the
coupling), with soft per-cluster responsibilities. This script adds a SECOND latent that
is ORTHOGONAL to the pairing class: a per-cluster evolutionary RATE drawn from a discrete
Gamma+Invariant grid. Both latents stay exchangeable (symmetric S, symmetric pi_c); the
rate simply rescales the branch length shared across the cluster's cherries.

## Model

Per size-2 cluster g the composite likelihood marginalises over BOTH a pairing class
c in {1..K} (weight w_c) AND a rate class r (weight rho_r), independent:

    L(g) = sum_c sum_r  w_c rho_r  prod_e  P_c(rate_r * tau_e)[pf_e -> pt_e]^{n_g^e}

- P_c(t) = expm(Q_c t), Q_c = metropolis_sqrt(S, pi_c): the exchangeable pair generator
  for class c (shared free S; symmetric pi_c). Identical to the free-S mixture.
- rate_r: a discrete Gamma+I grid. (R-1) Gamma categories with rate multipliers of mean 1
  and shape alpha (Yang 1994 mean discretisation, `rate_hetero.discrete_gamma_rates`), plus
  one INVARIANT category (rate 0). The invariant category's transition matrix is the
  identity: probability 1 on no-change transitions (pf == pt), 0 on any substitution, so it
  supports ONLY fully-conserved clusters (all cherries identical at both sites).
- The rate is PER-CLUSTER: the size-2 pair carries one latent rate, constant across its
  cherries; tau varies per cherry so the rate scales each cherry's branch length rate_r*tau.

The class and rate mixtures multiply (w_c rho_r), so they are a-priori independent; whether
their POSTERIORS are independent is measured below (question 3).

## EM (ECM, monotone in the observed-data marginal log-likelihood)

E-step. Per cluster g the responsibility r_{gcr} is the softmax over (c, r) of
log w_c + log rho_r + sum_e n_g^e log P_{c,r}(tb_e), with P_{c,r} evaluated at branch length
rate_r * tau_center[tb]. The invariant bin contributes log P = 0 on the diagonal and -inf
off it, so it only ever competes for clusters with zero substitutions.

M-step (conditional-maximisation blocks, each holding the others fixed => ECM):
- w_c   = sum_{g,r} r_{gcr}            (exact mixture-weight maximiser)
- rho_r = sum_{g,c} r_{gcr};  p_inv = rho_invariant   (exact; invariant responsibility mass)
- pi_c, S : the Holmes-Rubin / bridge M-step of the Metropolis-sqrt chain, run on the
  responsibility-weighted counts placed at the RATE-SCALED branch length rate_r*tau. Because
  the generator Q_c is shared across rates, the endpoint-conditioned usage/dwell statistics
  simply ADD over the Gamma rate bins (N_c = sum_r N_{c,r}, T_c = sum_r T_{c,r}); the
  invariant bin has branch length 0 so contributes zero dwell and drops out. pi_c is updated
  by `mstep_pi_metropolis` and the shared S is pooled over all (class, Gamma-rate) pairs,
  exactly as in the free-S mixture but with rate-scaled branches.
- alpha : a 1-D local search maximising the rate-marginal expected complete-data LL
  Q(alpha) = sum_{g,c,r>0} r_{gcr} LL_{c, rate_r(alpha)}(g), responsibilities and (pi_c, S)
  held fixed. The current alpha is always a candidate, so Q cannot decrease.

Monotonicity of the observed-data marginal LL is asserted every iteration and reported
(`monotone` flag). `--single-rate` collapses the rate grid to a single bin at rate 1 (no
invariant, no alpha), reproducing `fit_coupling_mixture_freeS.py` on the IDENTICAL corpus,
family split and initialisation.

### Identifiability (gauge)

The overall rate scale and the S scale are jointly unidentified: Q is proportional to S and
enters only as Q*rate*tau, so rate -> rate/k, S -> S*k leaves every likelihood unchanged.
Training is therefore run in the canonical mean-1 Yang gauge (the Gamma grid keeps mean 1;
S carries the absolute rate; free rho then shapes the used rate distribution). The gauge is
resolved once, at the end, for reporting: the effective variable-rate distribution is
renormalised to conditional mean 1, and its spread is summarised by a coefficient of
variation (rate_CV). Doing this rescale DURING training instead desynchronises the alpha
CM-step's current-alpha candidate and breaks monotonicity, so it is deliberately avoided.

## Results

Held-out (unseen-family) per-count log-likelihood, trRosetta per-contact corpus
(max-seqs 256; single-rate baselines are the shared-free-S mixture, agent-confirmed):

| K | single-rate | Γ+I | Γ+I gain | alpha | p_inv | rate_CV | class<->rate NMI | max MI(pi_c) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -2.6719 | **-2.5959** | +0.076 | 0.78 | 0.004 | 0.68 | 0.000 | 0.027 |
| 2 | -2.6531 | -2.5775 | +0.076 | 0.78 | 0.004 | 0.67 | 0.029 | 0.136 |
| 4 | -2.6270 | -2.5608 | +0.066 | 0.78 | 0.004 | 0.65 | 0.033 | 0.11 |
| 8 | -2.6021 | -2.5455 | +0.057 | 0.78 | 0.004 | 0.65 | 0.040 | 0.29 |

Findings:
1. **Γ+I helps at every K** (+0.057..+0.076/count). Γ+I at K=1 (one class, -2.596)
   already beats the 8-class single-rate mixture (-2.602): a range of rates with one
   coupling class outscores eight coupling classes at one rate. On the likelihood
   axis, per-cluster rate is the dominant factor.
2. **But it is MI-neutral** (rescaling is stationary-invariant). alpha=0.78 and
   rate_CV~0.65 are flat across K, while MI(pi_c) climbs 0.027 -> 0.29 purely with the
   class count. Coupling recovery tracks classes, not rates.
3. **The two factors separate cleanly**: class<->rate NMI <= 0.04, so a cluster's
   posterior rate and posterior pairing class are essentially independent latents.
4. alpha < 1 = genuine per-cluster rate spread; p_inv ~ 0.004 = negligible invariant
   mass (contacts rarely fully conserved across a whole cluster).

Note: the original agent-launched sweep's background jobs died with the agent turn
(the script writes JSON only on completion), so this table was produced by a
durable session-level relaunch (`scratch_gammaI_sweep.sh`); the single-rate
baselines reproduce the free-S numbers to ~0.003.

## Reproduce

    cd ~/tkf-dp   # (or the worktree); needs data/per_contact_trrosetta/counts.npz
    # single-rate baseline (== fit_coupling_mixture_freeS) and Gamma+I, per K:
    PYTHONPATH=src python experiments/fit_coupling_mixture_rateI.py --K 4 --single-rate
    PYTHONPATH=src python experiments/fit_coupling_mixture_rateI.py --K 4 --gamma-cats 4
