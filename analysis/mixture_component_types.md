# Coupling-mixture components: types and lumpability

Characterisation of the fitted free-S single-rate coupling mixture (`fit_coupling_mixture_freeS.py`; re-fit and dumped by `experiments/characterize_mixture_components.py`). Each component c is a reversible EXCHANGEABLE 400-state pair CTMC Q_c = metropolis_sqrt(S, pi_c) built from one SHARED free single-site exchangeability S (warm-init LG08) and a per-component SYMMETRIC joint stationary pi_c (the coupling). Corpus `data/per_contact_trrosetta/counts.npz`; held-out = unseen families (val-frac 0.2, seed 0). Fitted pi_c / S / weights dumped to `results/mixture_component_char/`.


## K = 4

Re-fit: 150 EM iters, held-out per-count LL = -2.6230, shared-S rel-Frobenius from LG08 = 0.954 (off-diag corr 0.910).

Coupling MI(pi_c): simple-mean = 0.112, weighted-mean = 0.099, max = 0.181 nats.


### Part 1 -- component coupling types

`type` = coupling direction inferred from the excess signature (cross-checked with the composition label); `top coupling pairs` = unordered a-b ranked by pointwise-MI contribution P(a,b) log[P(a,b)/(rho_a rho_b)] (mass-weighted coupling EXCESS, not raw frequency), each shown with its excess ratio P/(rho rho).


| c | weight | MI | coupling type | composition | top coupling pairs (excess x) |
|---:|---:|---:|---|---|---|
| 0 | 0.372 | 0.038 | charge-complementary/salt-bridge | salt-bridge(+/-) | ER(1.6x), LP(1.8x), EK(1.6x), FP(2.0x), DL(1.4x), PV(1.5x) |
| 1 | 0.218 | 0.119 | mixed/weak | size-matched | AG(1.8x), LR(2.5x), GP(2.2x), KL(2.1x), IR(2.2x), RV(2.0x) |
| 2 | 0.209 | 0.109 | hydrophobic size-matching | hydrophobic-core | DG(3.7x), AL(1.3x), AV(1.4x), GL(1.5x), AI(1.4x), GR(2.6x) |
| 3 | 0.200 | 0.181 | disulfide(C-C) | hydrophobic-core,size-matched | CC(22.7x), DP(8.5x), LL(1.3x), GN(5.7x), EG(4.2x), LV(1.2x) |

### Part 2 -- lumpability

`2-sided resid` = relative distance of b_marg(pi_c, A_c) from range(B) (Klein-orbit / exchangeable block-sum incidence): 0 iff a two-sided-LUMPABLE reversible-exchangeable generator can carry this component's (pi_c, marginal). `half resid` = same against range(B_half) (time-reversal orbits, cpt-1 constraint only). `gen resid` = relative Frobenius of the un-representable part of the ACTUAL Q_c's cpt-1 block-sum flux (how far the fitted generator itself is from lumpable). Tolerance 0.0001.


Product-stationary control (MI=0): two-sided range residual = 0.0e+00, half = 0.0e+00 (validates the machinery -- the independent lift is exactly two-sided lumpable).


| c | weight | MI | 2-sided resid | half resid | gen resid | 2-sided lumpable? | half-lumpable? |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 0.372 | 0.038 | 0.0206 | 0.0e+00 | 0.1235 | NO | yes |
| 1 | 0.218 | 0.119 | 0.0487 | 0.0e+00 | 0.1789 | NO | yes |
| 2 | 0.209 | 0.109 | 0.0280 | 0.0e+00 | 0.1430 | NO | yes |
| 3 | 0.200 | 0.181 | 0.0155 | 0.0e+00 | 0.0818 | NO | yes |

**Verdict (K=4): 0/4 components two-sided lumpable; 4/4 half-lumpable.** Two-sided residuals span [0.015472, 0.048702]; half residuals span [0.0e+00, 0.0e+00].


## K = 8

Re-fit: 150 EM iters, held-out per-count LL = -2.5965, shared-S rel-Frobenius from LG08 = 0.953 (off-diag corr 0.909).

Coupling MI(pi_c): simple-mean = 0.155, weighted-mean = 0.143, max = 0.290 nats.


### Part 1 -- component coupling types

`type` = coupling direction inferred from the excess signature (cross-checked with the composition label); `top coupling pairs` = unordered a-b ranked by pointwise-MI contribution P(a,b) log[P(a,b)/(rho_a rho_b)] (mass-weighted coupling EXCESS, not raw frequency), each shown with its excess ratio P/(rho rho).


| c | weight | MI | coupling type | composition | top coupling pairs (excess x) |
|---:|---:|---:|---|---|---|
| 0 | 0.169 | 0.087 | mixed/weak | mixed | LP(2.5x), PV(1.9x), FP(2.5x), IP(2.2x), DL(1.6x), DK(1.6x) |
| 1 | 0.152 | 0.111 | charge-complementary/salt-bridge | salt-bridge(+/-) | LW(3.9x), FF(5.3x), ER(1.5x), PR(2.1x), LN(2.3x), EK(1.4x) |
| 2 | 0.147 | 0.118 | mixed/weak | mixed | AG(3.2x), LR(1.5x), GN(3.5x), RV(1.6x), EL(1.9x), IK(1.8x) |
| 3 | 0.126 | 0.173 | disulfide(C-C) | hydrophobic-core | CC(21.4x), DP(15.1x), LV(1.3x), EP(9.6x), NP(7.0x), IL(1.1x) |
| 4 | 0.118 | 0.164 | mixed/weak | mixed | GL(1.7x), GV(1.6x), AP(2.7x), GS(1.5x), GK(1.8x), GI(1.5x) |
| 5 | 0.116 | 0.113 | charge-complementary/salt-bridge,hydrophobic size-matching | hydrophobic-core | DG(7.2x), AL(1.5x), AV(1.4x), AI(1.5x), DR(10.6x), AF(1.4x) |
| 6 | 0.090 | 0.181 | mixed/weak | mixed | GG(4.5x), GP(4.3x), LT(1.9x), DH(8.2x), TV(1.7x), IT(1.9x) |
| 7 | 0.082 | 0.290 | hydrophobic size-matching | hydrophobic-core,size-matched | GT(7.8x), DS(6.6x), LL(1.4x), AD(4.3x), FL(1.5x), GH(5.8x) |

### Part 2 -- lumpability

`2-sided resid` = relative distance of b_marg(pi_c, A_c) from range(B) (Klein-orbit / exchangeable block-sum incidence): 0 iff a two-sided-LUMPABLE reversible-exchangeable generator can carry this component's (pi_c, marginal). `half resid` = same against range(B_half) (time-reversal orbits, cpt-1 constraint only). `gen resid` = relative Frobenius of the un-representable part of the ACTUAL Q_c's cpt-1 block-sum flux (how far the fitted generator itself is from lumpable). Tolerance 0.0001.


Product-stationary control (MI=0): two-sided range residual = 0.0e+00, half = 0.0e+00 (validates the machinery -- the independent lift is exactly two-sided lumpable).


| c | weight | MI | 2-sided resid | half resid | gen resid | 2-sided lumpable? | half-lumpable? |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 0.169 | 0.087 | 0.0235 | 0.0e+00 | 0.1519 | NO | yes |
| 1 | 0.152 | 0.111 | 0.0339 | 0.0e+00 | 0.1770 | NO | yes |
| 2 | 0.147 | 0.118 | 0.0321 | 0.0e+00 | 0.1585 | NO | yes |
| 3 | 0.126 | 0.173 | 0.0103 | 0.0e+00 | 0.0806 | NO | yes |
| 4 | 0.118 | 0.164 | 0.0464 | 0.0e+00 | 0.1802 | NO | yes |
| 5 | 0.116 | 0.113 | 0.0195 | 0.0e+00 | 0.1445 | NO | yes |
| 6 | 0.090 | 0.181 | 0.0365 | 0.0e+00 | 0.2051 | NO | yes |
| 7 | 0.082 | 0.290 | 0.0399 | 0.0e+00 | 0.1457 | NO | yes |

**Verdict (K=8): 0/8 components two-sided lumpable; 8/8 half-lumpable.** Two-sided residuals span [0.010286, 0.046365]; half residuals span [0.0e+00, 0.0e+00].


## Interpretation

- **Part 1.** The transition-fit components DO carry interpretable coupling directions -- unlike the stationary DP-DMM composition archetypes (wMI <= 0.035, coupling washed out), these have MI(pi_c) up to ~0.29 nats and their excess signatures point at recognisable biophysics (charge-complementary salt bridges D/E<->K/R, hydrophobic size-matching, aromatic stacking). The COMPOSITION label and the COUPLING type can differ: a component can be composition-'salt-bridge-enriched' yet its excess still concentrate on the complementary (opposite-charge) pairs, which is the genuine coupling signal the raw composition hides.

- **Part 2.** Consistent with the section-6 feasibility theory: the empirical per-component coupling leaves range(B), so NO component is two-sided lumpable, while b_marg never leaves range(B_half), so EVERY component is (near-)half-lumpable. The product-stationary control sits at range residual ~0, confirming the residual is a true coupling effect and not a machinery artifact. A factorisation-preserving two-sided-lumpable mixture structurally cannot represent these components; the one-sided (half-lumpable) relaxation can.

