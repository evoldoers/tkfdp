# H-supervised-pdb-rnn-K8-2026-05-31

Supervised fit of the Potts coupling matrix `H` (single shared 20×20)
on PDB-anchored reciprocal-nearest-neighbour (RNN) column-pairs with a
Cα distance threshold of 10 Å.

## Setup

- **Site-class shell**: K_c=8 mixture, π_class taken from the released
  unsupervised checkpoint `K8-KH1-top8000-2026-05-22` (val_LL=-49684.19,
  ‖H‖_F=32.0). The supervised fit re-uses the unsupervised π_class
  and π_c priors; only `H` (and the mixture weight `w_un`) are fit.
- **Supervision dataset**: 1858 Pfam families with PDB anchors →
  72,601 reciprocal-NN <10 Å column-pairs → 2,249,327 cherries.
  Built by `experiments/build_supervised_contacts.py`.
- **Mixture structure** (per column-pair, `experiments/fit_supervised_H_with_null.py`):
  - ζ_{ij}=0 with prior `w_un` → both columns evolve independently
    under the K=8 singlet emissions (no H), pre-marginalised:
    `M(a,d,t) = Σ_c ρ_c π_c(a) P_c(d|a,t)`.
  - ζ_{ij}=1 with prior `1-w_un` → coupled via `H`, conditional on
    `(c1,c2) ∈ {1..K_c}²` with prior `ρ_{c1} ρ_{c2}`.
  Beta(1,1) MAP on `w_un` re-estimated each E-step.
- **EM**: 5 seeds (1 warm-start from `H_unsupervised`, 4 random),
  25 EM iters × 30 inner M-step grad steps, Adam(lr=0.02),
  Gaussian prior `τ=0.1` on off-diagonal `H` entries. All 5 seeds
  converged to the same optimum (max ΔF=0.4 nats on -5.1M nat
  total LL), indicating a highly identifiable objective.

## Winner

`seed0_warm_null/state.npz`:
- `final_marg_LL` = -5,102,601
- `final_w_un` = 0.457 → **46% of PDB-contact column-pairs do not
  coevolve under this model**
- `final_‖H‖_F` = 6.54 (vs unsupervised K=8 ‖H‖=32.0)

## Files

- `contacts.npz` — supervision dataset (cherry pairs + tau bin labels).
- `seed{0,1,2,3,4}_{warm,rand}_null/` — converged EM states for the
  null-mixture variant (with `w_un`). seed0_warm_null is the winner
  used downstream.
- `seed{0,1}_{warm,rand}/` — earlier non-null variants kept for
  reference (assumed all contacts coevolve; superseded by null
  variant after the 46% null-rate finding).
- `balibase_per_pair.csv` / `balibase_aggregate.json` — downstream
  BAliBASE eval results (187 pairs, L<150 subset of bali3pdbm).

## Downstream evaluation

`val_log_likelihood_class_marginal` (49-family held-out Pfam,
n_burnin=50, n_samples=50):
- Supervised: val_LL = -38476.16
- Unsupervised K=8 reference: val_LL = -38385.44
- Δ ≈ -91 nats (within MCMC noise of equivalent).

BAliBASE bali3pdbm L<150 subset (n=187 pairs, infinite-PHMM sampler
`infinite_phmm_mcmc_K8_coupled_RE`, 8000 sweeps + 2000 burn-in per
rung × 6-rung α_z ladder, 2 RE replicates per pair):
- Baseline FSA (no H):      Q' F1 = 0.452, per-pair-mean SP = 0.3130
- **Supervised K=8 (this)**: Q' F1 = 0.452, per-pair-mean SP = 0.3130 (Δ=+0.0001)
- Unsupervised K=8:          Q' F1 = 0.455, per-pair-mean SP = 0.3152 (Δ=+0.0022)

The supervised H gives ~1/20 the BAliBASE lift of the unsupervised H.
Consistent with: (a) ‖H_sup‖ is 5× smaller, (b) 46% of contacts are
in the null mixture component (the null-rate w_un suppresses ‖H‖),
(c) the unsupervised fit also captures coevolutionary patterns that
help anchor MSA alignment but are *not* in the PDB contact set.

## Reproduce

```bash
# Build the contact dataset (1858 PDB-anchored families)
python experiments/build_supervised_contacts.py \
    --pdb-dir ~/bio-datasets/data/pfam/pdb_anchors \
    --out results/H_supervised_pdb_rnn_K8_2026-05-31/contacts.npz \
    --max-dist 10.0 --min-separation 4

# Fit H (winner = seed0 warm-start, null-mixture variant)
python experiments/fit_supervised_H_with_null.py \
    --base-ckpt results/K8-KH1-top8000-2026-05-22/_best_chkpt \
    --contacts results/H_supervised_pdb_rnn_K8_2026-05-31/contacts.npz \
    --out results/H_supervised_pdb_rnn_K8_2026-05-31/seed0_warm_null \
    --seed 0 --warm-start --n-em-iters 25 --prior-tau 0.1 --lr 0.02

# Held-out val LL
python experiments/eval_supervised_H_val_LL.py \
    --supervised-state results/H_supervised_pdb_rnn_K8_2026-05-31/seed0_warm_null/state.npz

# Downstream BAliBASE (AWS sweep; see aws/balibase_one_pair_supH.yaml)
```
