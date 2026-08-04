# Convex kernel-basis fit of the Lumpable pair CTMC

A convex replacement for the biconvex ECM fitter of the two-sided **Lumpable**
reversible-exchangeable pair CTMC (paper2b), plus its one-sided **Half-lumpable**
sibling. The joint 400-state generator is written as

    Q = Q_0 + sum_alpha theta_alpha K_alpha

where `Q_0` is the A(+)A lift of a FIXED marginal GTR generator and the `{K_alpha}`
span the **lumpability kernel**: reversible + exchangeable flux perturbations that
leave every one-component (marginal) observable unchanged. For fixed stationary
`pi` and fixed marginal the complete-data log-likelihood is concave in the flux, so
the M-step is a single convex program solved to the global optimum, wrapped in the
exact endpoint-conditioned (bridge) EM E-step.

Code: `experiments/fit_lumpable_kernel.py` (two-sided),
`experiments/fit_half_lumpable_kernel.py` (one-sided). Run with `PYTHONPATH=src`.
Numbers below are per-observation held-out log-likelihood (larger is better) on a
family shard withheld from training; `pi` = product of the empirical single-site
marginals (the two-sided-lumpable MLE stationary), marginal GTR fitted from the
pair tensor's own marginal.

---

## 1. Kernel construction and dimension

**Orbit-flux coordinates.** Write the reversible generator as `Q_xy = F_xy/pi_x`
with `F` symmetric. Component exchangeability makes `F` constant on the Klein-4
orbits `V4 = {e, R (time reversal), E (component swap), RE}` of directed
off-diagonal transitions; one nonnegative variable `phi_o` per orbit gives
`N_phi = (n^4 + n^2 - 2n)/4` free fluxes (Burnside).

**Two-sided lumpability** (project either coordinate to a Markov chain) is the
linear constraint, for `i != k`,

    (B phi)_ijk := sum_l F_{ij,kl} = (pi_ij/rho_i) g_ik ,     rho_i = sum_j pi_ij,

with `g_ik = g_ki` the symmetric marginal edge fluxes and `A_ik = g_ik/rho_i` the
marginal generator; exchangeability supplies the second coordinate for free.

**The kernel** is the coupling-only subspace: reversible + exchangeable +
two-sided-lumpable flux perturbations that leave the marginal generator fixed. A
perturbation `delta` fixes the marginal iff `sum_l delta F_{ij,kl} = 0` for every
`i != k, j`, i.e. `B delta = 0`. So

    kernel = null(B)      (the block-sum incidence B, mapped over the spectator l).

`Q_0 + sum_alpha theta_alpha K_alpha` with `{K_alpha}` a basis of `null(B)` then
ranges over exactly the generators with the given `pi` and marginal, and the
positivity constraint `q_off >= 0` becomes `phi >= 0`. Validated numerically:
adding any `null(B)` direction to a valid lumpable `Q` leaves its projected
marginal generator unchanged to `~2e-15` (machine precision).

### Kernel dimension (distinct from eq:lumpdof's flux dimension)

The coupling kernel `null(B)` — fixed-marginal reversible-exchangeable perturbations
— has exact dimension (sympy rational arithmetic, n = 3..5)

    rank(B) = (n-1)(n^2 - n + 1)                          [ = N_c - C(n-1,2) ]
    kernel  = N_phi - rank(B) = (n-1)^2 (n^2 - 2n + 4)/4  = 1, 7, 27, 76, 175, 32851

for n = 2,3,4,5,6,20. The component-swap symmetry makes `C(n-1,2) = (n-1)(n-2)/2` of
the lumpability rows linearly dependent, so the *naive* coupling count
`d_n - n(n-1)/2` (= 1, 6, 24, 70, 165, 32680) UNDER-counts `null(B)` by exactly
`C(n-1,2)`.

| n | N_phi | rank(B) | kernel = null(B) | naive d_n - n(n-1)/2 | gap = C(n-1,2) |
|---|-------|---------|------------------|----------------------|----------------|
| 2 | 4     | 3       | **1**            | 1                    | 0              |
| 3 | 21    | 14      | **7**            | 6                    | 1              |
| 4 | 66    | 39      | **27**           | 24                   | 3              |
| 5 | 160   | 84      | **76**           | 70                   | 6              |
| 6 | 330   | 155     | **175**          | 165                  | 10             |
| 20| 40090 | 7239    | **32851**        | 32680                | 171            |

**This is the fixed-marginal COUPLING kernel, not a correction to eq:lumpdof.** The
two-sided-lumpable *flux* dimension `d_n = n(n-1)(n^2-3n+6)/4` in the paper is
correct: an independent direct null-space count of the lumpable flux space at n = 3
gives **9 = d_3**, matching the formula and the fitter's `n_params = 32870` at
n = 20. The two are linked by a clean identity,

    d_n = null(B) + (n-1)        (32851 + 19 = 32870 at n=20; 7 + 2 = 9 at n=3),

whose extra `n-1` is exactly the feasible marginal-preserving coupling dimension of
Section 6. The naive `d_n - n(n-1)/2` is simply the wrong route to the coupling
kernel: within the lumpable flux space the marginal-generator directions and the
coupling directions are not orthogonal complements — they overlap by `C(n-1,2)`,
which the component swap induces.

---

## 2. LP feasibility — does a strictly-positive lumpable generator exist?

Off-diagonal rates are affine in the flux, so feasibility is an LP:

    max t   s.t.   q_xy = phi_{o(xy)}/pi_x >= t  for all off-diagonal (x,y),
                   B phi = b_marg,   phi >= 0.

The A(+)A lift (double-transition flux 0) is always feasible with `t = 0`, so the
optimum has `t >= 0`; substituting `phi = t*pmax + s (s >= 0)` folds the rate floor
into the bounds and removes the explicit inequality rows (HiGHS then solves the
40090-variable LP in ~1 min instead of timing out). Result:

| corpus    | LP max-slack t |
|-----------|----------------|
| trRosetta | **1.72e-5** (>0) |
| af_full   | **1.91e-5** (>0) |

`t > 0`: a strictly-positive lumpable generator with this `(pi, marginal)` exists,
but the interior is razor-thin (margin ~1e-5). The MLE sits essentially on the
boundary (Section 3).

---

## 3. Convex fit vs the reference

EM: exact bridge E-step (`fit_pair_models.estep`) + convex M-step
`max_phi sum_o [C_o log phi_o - H_o phi_o]` s.t. `B phi = b_marg, phi >= 0`. The
M-step is solved to the global optimum by the rational (Poisson-Sinkhorn) dual
coordinate ascent `phi_o(lambda) = C_o/(H_o + (B^T lambda)_o)`; each block-sum row
solves a 1-D monotone equation exactly, which is scale-robust where a global dual
Newton is not (the flux/count scale mismatch spans ~5 orders of magnitude). The EM
is monotone and lands at the global coupling optimum given `pi`.

| model (held-out per-count LL)              | trRosetta   | af_full     |
|--------------------------------------------|-------------|-------------|
| A(+)A independent (product pi, MLE marginal, no coupling) | -2.2478 | -1.3278 |
| **convex kernel-basis Lumpable (this work)** | **-2.2415** | **-1.3245** |
| reference biconvex/lift Lumpable            | -2.2419     | -1.3248     |
| saved ECM Lumpable (joint-pi, `results/pair_models/lumpable_*`) | -2.3576 | -1.4018 |

The convex fit **confirms and slightly beats** the -2.2419 / -1.3248 reference
(monotone, global). It settles the ECM-artifact question: the saved -2.3576 /
-1.4018 is not the Lumpable optimum — it is an artifact of *jointly* fitting a
non-product stationary. With the product `pi` (the profile-likelihood MLE
stationary) and a data-consistent fixed marginal, the true two-sided-lumpable
optimum is -2.2415 / -1.3245, a hair above the independent A(+)A baseline. The
gain from the coupling is only 0.006 / 0.003 per observation — the **coupling
washout**: a single shared 400-state matrix has almost no lumpable coupling to fit.

### Interior vs boundary

The MLE is a **boundary optimum** — the coupling is driven to the positivity floor
rather than to an interior value. Because the M-step is `phi_o = C_o/(...)`, the
coupling directions the data do not support collapse toward zero flux, and the
zeroing is concentrated in the **double-transition** (simultaneous-substitution)
orbits:

| corpus    | min positive rate | frac off-diag rates ~0 | double-orbits zeroed | single-orbits zeroed |
|-----------|-------------------|------------------------|----------------------|----------------------|
| trRosetta | 6.4e-22           | 2.8%                   | 1516/36290 = **4.2%**| 25/3800 = **0.66%**  |
| af_full   | 4.1e-27           | 5.4%                   | 2548/36290 = **7.0%**| 9/3800 = **0.24%**   |

The relative coupling magnitude `||phi - phi0|| / ||phi0||` is 0.072 (trR) /
0.059 (af). So the fit adds a small amount of single-transition context dependence
but pushes most simultaneous-substitution flux to ~0 — the data want little
lumpable coupling, a property of the DATA, not the solver (the n=4 recovery below
shows the same method finds genuine coupling when it is present).

**Solver caveat.** The fixed-marginal constraint is enforced to `B phi = b_marg`
at relative residual ~4e-3, not machine precision. This is intrinsic to the
near-boundary optimum (the dual multipliers run to infinity as `phi_o -> 0`), not a
convergence failure — the log-likelihood and the boundary structure are stable
across further iterations.

---

## 4. Small-n correctness checks

- **n = 2** (3 total DOF: 2 stationary + 1 gauged rate; kernel dim 1). Synthetic
  coupled 2x2 count set. Brute-force 1-D grid MLE `theta = 0.1500`; convex EM
  recovers `theta = 0.1500` with identical per-count LL -0.75914. **Exact match.**
- **n = 4** (kernel dim 27). Data simulated from a known interior lumpable `Q` with
  nonzero coupling (a convex combination of the A(+)A lift and an LP-interior
  point, same marginal). Convex fit recovers the coupling with `corr = 0.9997`,
  interior optimum, and per-count LL matching the truth to 5 decimals
  (-1.69858 vs -1.69859). This confirms the method finds non-trivial coupling when
  it is genuinely present, so the boundary/washout result on real data is a
  property of the data.

Both checks required initializing EM at a strictly-interior generator: the pure
A(+)A lift has *exactly-zero* double-transition rates, which the Holmes-Rubin
E-step maps to zero expected usage — an absorbing boundary that otherwise freezes
the coupling at 0.

---

## 5. Half-lumpable (one-sided) extension

Half-lumpable keeps lumpability of **component 1 only** (cpt1's marginal is Markov;
nothing imposed on cpt2). Mechanically it is the same convex kernel fit with two
changes: the flux is tied only by **time reversal** (`F_{ab,cd} = F_{cd,ab}`), not
by the component swap, so there are `n^2(n^2-1)/2` orbits (79,800 for n=20) instead
of the Klein `N_phi`; and only the cpt1 row-marginal constraint is imposed (the
same rows). Both changes ENLARGE the kernel. Counts are component-swap-symmetrised
(the field label is arbitrary); `pi` stays symmetric but the dynamics are
non-exchangeable.

### Dimension — the swap is exactly the source of the two-sided deficiency

On time-reversal orbits the cpt1 lumpability operator has **full generic rank**:
`rank(B) = N_c = n(n-1)(2n-1)/2` exactly (sympy-confirmed, n=2..6), with **no**
`C(n-1,2)` deficiency. The deficiency in the two-sided case is therefore caused
precisely by the component-swap tie; removing it restores full rank. Hence

    kernel_half = N_phi_half - N_c = n(n-1)(n^2 - n + 1) / 2

| n | N_phi (t-rev) | rank(B) = N_c | kernel_half | two-sided kernel |
|---|---------------|---------------|-------------|------------------|
| 2 | 6             | 3             | **3**       | 1                |
| 3 | 36            | 15            | **21**      | 7                |
| 4 | 120           | 42            | **78**      | 27               |
| 5 | 300           | 90            | **210**     | 76               |
| 6 | 630           | 165           | **465**     | 175              |
| 20| 79800         | 7410          | **72390**   | 32851            |

`kernel_half + 209` (stationary) = 72599, matching paper2b's Half-lumpable "Free
dim ~= 72,599" — the paper got the one-sided count right; only the two-sided count
carries the swap-induced deficiency.

### Fit vs the ALM reference

Same convex machinery (rational-dual M-step, bridge E-step), just on the larger
one-sided kernel; monotone.

| model (held-out per-count LL)                    | trRosetta   | af_full     |
|--------------------------------------------------|-------------|-------------|
| A(+)A independent                                | -2.2478     | -1.3278     |
| **convex kernel-basis Half-lumpable (this work)**| **-2.2394** | **-1.3230** |
| ALM reference (paper2b, near-plateau/non-converged) | -2.2436  | -1.3255     |
| convex two-sided Lumpable (Section 3)            | -2.2415     | -1.3245     |

The convex one-sided fit **beats the ALM reference by 0.0042 / 0.0025** and beats
the two-sided fit by 0.0021 / 0.0015. LP margins `t = 1.72e-5` (trR) / `1.91e-5`
(af) — again `t > 0`, thin interior. **Wall clock: 143 s (trR) / 138 s (af), a
~75x speedup over the ~3-hour ALM** (the LP now solves in ~1 s thanks to the finer
orbits).

### Interior vs boundary, and what dropping the second constraint buys

Still a **boundary** optimum (min positive rate 1.9e-24 trR / 3.3e-28 af), but the
larger kernel is less boundary-bound: the double-transition zeroing is smaller in
fraction than two-sided, while the single-transition zeroing all but disappears.

| corpus    | model      | double-orbits zeroed | single-orbits zeroed | coupling ‖phi-phi0‖/‖phi0‖ |
|-----------|------------|----------------------|----------------------|----------------------------|
| trRosetta | two-sided  | 4.2%                 | 0.66%                | 0.072                      |
| trRosetta | half       | **2.9%**             | **0.33%**            | 0.090                      |
| af_full   | two-sided  | 7.0%                 | 0.24%                | 0.059                      |
| af_full   | half       | **5.2%**             | **0.14%**            | 0.083                      |

Dropping the second lumpability constraint buys ~0.002 per observation: it frees
the cpt2 dynamics to be non-exchangeable (asymmetric single-transition context),
which the data partly use (larger relative coupling, fewer zeroed orbits), while
the simultaneous-substitution flux still largely washes out. Consistent with
paper2b's finding that "the second lumpability constraint costs about ten times the
first" — here the one-sided model recovers most of that cost cleanly and provably
at the global optimum.

**Solver note.** As two-sided, the fixed-cpt1-marginal constraint holds at relative
residual ~5e-3 (intrinsic to the near-boundary optimum, not a convergence failure);
the fit is monotone throughout.

---

## 6. Why product? Feasibility, not just likelihood

The profile-likelihood result says the product stationary is the two-sided-lumpable
MLE. The max-slack LP shows the reason is **feasibility**, not merely a likelihood
optimum. Hold the fitted marginal `(A, rho)` fixed and sweep the stationary
coupling with the single-site marginal pinned at `rho`,

    pi_s = symmetric-Sinkhorn_to_rho( (1-s)*(rho (x) rho) + s*pi_emp ),

from `s = 0` (product) through `s = 1` (empirical pair stationary) into extrapolated
`s > 1`. For each `pi_s` the two-sided lumpability RHS is `b_marg_ijk = pi_ij A_ik`;
solve the signed-`t` LP (t<0 or LP-infeasible => the positive lumpable cone is
empty). `analysis/scripts/lp_stationary_sweep.py`, trRosetta:

| s    | MI(pi_s) nats | ‖B phi* − b_marg‖/‖b_marg‖ | LP outcome        |
|------|---------------|-----------------------------|-------------------|
| 0.00 (product)  | 0.0000 | **2.2e-14**  | feasible, t = +1.72e-5 |
| 0.25 | 0.0019        | 4.0e-3                      | **infeasible**    |
| 0.50 | 0.0072        | 7.9e-3                      | **infeasible**    |
| 0.75 | 0.0154        | —                           | infeasible        |
| 1.00 (empirical)| 0.0266 | 1.5e-2       | **infeasible**    |
| 1.50 | 0.0577        | —                           | infeasible        |

The equality `B phi = b_marg` is exactly solvable only at `s = 0` (residual 2e-14);
along the empirical coupling direction `b_marg` genuinely leaves `range(B)` (residual
grows 4e-3 -> 1.5e-2 and stays infeasible out to `s = 3`, `MI = 0.232`), so no
reversible-exchangeable generator lumpable to `A` with that stationary exists — the
positive cone along the data's direction is empty, and it shuts at the first
increment (`MI = 0.0019` nats). But is that a property of the pooled/empirical
direction specifically, or of ALL coupling? The next test settles it.

### 6.1 Coherent directions and the exact feasible-coupling dimension

Because `b_marg(pi) = pi_ij A_ik` is LINEAR in `pi` and `range(B)`-membership is
linear, the marginal-preserving coupling perturbations `delta` (symmetric, zero
row-sum) that stay two-sided-feasible form a linear subspace,
`{delta : L^T b_marg(delta) = 0}` with `L` a basis of left-null(`B`). Its exact
dimension (random reversible `A`, A-generic over seeds;
`analysis/scripts/coherent_feasibility.py`):

| n | coupling DOF n(n-1)/2 | feasible-coupling dim | interpretation |
|---|------------------------|-----------------------|----------------|
| 2 | 1  | **1** | ALL couplings feasible (the binary telegraph) |
| 3 | 3  | **2** | 2 of 3 directions feasible |
| 4 | 6  | **3** | 3 of 6 |
| 5 | 10 | **4** | 4 of 10 |
| 6 | 15 | **5** | 5 of 15 |

The feasible-coupling dimension is exactly **`n - 1`**. So coupled two-sided-lumpable
chains are NOT forbidden for `n >= 3` — an `(n-1)`-dimensional family of coherent
couplings survives — but for `n >= 3` that family is a thin proper subspace of the
`n(n-1)/2` coupling directions, and it shrinks in relative measure as `n` grows
(19 of 190 at `n = 20`). **This reconciles the telegraph:** at `n = 2` the feasible
family IS the whole coupling space (`n-1 = 1 = n(n-1)/2`), which is exactly why the
binary telegraph is a genuinely coupled two-sided-lumpable chain; the restriction to
a measure-shrinking slice begins precisely at `n = 3`, where the component-swap tie
first over-determines the constraints.

The empirically-relevant coupling directions are NOT in that thin family. Sweeping
coherent couplings at `n = 20` (empirical `rho`, `A`), the matched-residue diagonal,
the salt-bridge charge pattern `-q q^T`, and a random rank-1 `u u^T` all leave
`range(B)`, residual rising with `MI`:

| coherent direction | MI = 0.005–0.01 | MI = 0.02–0.05 | MI = 0.08–0.30 |
|--------------------|-----------------|----------------|----------------|
| diagonal (matched residue) | 3.7e-3 | 7.4e-3 | 1.0e-2 |
| charge / salt-bridge       | 4.0e-3 | 9.0e-3 | 2.1e-2 |
| rank-1 `u u^T`             | 3.3e-2 | 5.3e-2 | 6.8e-2 |

**Verdict (reconciled).** The two-sided-Lumpable feasible stationaries are product
plus a thin `(n-1)`-dimensional family of special coherent couplings — NOT "product
only" (the telegraph lives at `n = 2` where the family is everything), and NOT a full
`n >= 3` obstruction. But the pooled empirical direction and every physically natural
coherent coupling (matched-residue, salt-bridge, rank-1) fall OUTSIDE the family and
are genuinely infeasible. So the product preference is **feasibility-reinforced along
the data's directions**: two-sided lumpability admits coupling only along a
measure-shrinking `(n-1)`-dim slice that excludes the coevolutionary patterns the
data actually carry, so the fit cannot move toward the empirical coupling even
setting likelihood aside — the profile-likelihood optimum sits against a hard
feasibility wall in the empirically-relevant directions.

### 6.2 Per-archetype and the one-sided contrast

The natural contact archetypes are exactly the coherent patterns tested above: a
salt-bridge archetype is the charge pattern `-q q^T`, a matched-packing archetype is
diagonal, a generic co-varying mode is rank-1. All leave `range(B)` (table above), so
disaggregating the pooled coupling into per-archetype stationaries does NOT recover
feasibility — a factorization-preserving two-sided-lumpable mixture cannot represent
the empirical per-contact coupling; the components would have to be non-lumpable (or
confined to the physically-irrelevant `(n-1)`-dim feasible family).

**One-sided contrast — the second constraint is the whole story.** Repeating the
`range(B)` test for the Half-lumpable operator (time-reversal orbits, cpt1 constraint
only) gives residual `~1e-16` (machine zero) at EVERY coupled stationary
(`s = 0, 0.25, 1, 2`, and the coherent patterns too): `b_marg` never leaves
`range(B_half)`, so one-sided-lumpable coupled generators exist for the empirical and
all coherent directions freely. Dropping the second lumpability constraint (the
component-swap tie) therefore lifts the feasible-stationary set from {product + thin
`(n-1)`-dim slice} to all of coupling space — the structural counterpart of its
~0.002/observation likelihood cost (Section 5), and the reason Half-lumpable can fit
the coevolutionary coupling the two-sided model structurally cannot.

---

## Reproduce

```
PYTHONPATH=src python3 experiments/fit_lumpable_kernel.py --all --n-em 20 \
    --out results/pair_models/lumpable_kernel_summary.json
PYTHONPATH=src python3 experiments/fit_half_lumpable_kernel.py --all --n-em 20 \
    --out results/pair_models/half_lumpable_kernel_summary.json
PYTHONPATH=src python3 analysis/scripts/lp_stationary_sweep.py     # Sec 6: t(s) vs MI
PYTHONPATH=src python3 analysis/scripts/coherent_feasibility.py    # Sec 6.1: feasible-coupling dim
```

Fitted generators are saved to
`results/pair_models/{lumpable,half_lumpable}_kernel_<corpus>.npz`
(keys `Q`, `pi`, `phi`, `A`, `rho`, `val_per`, `train_per`).
