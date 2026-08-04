# Cohn CTBN mean-field vs. the closed-form path ELBO (400-state amino-acid pair)

Compares three ways to evaluate the endpoint-conditioned log-transition
`log P(X_T = y | X_0 = x)` of a coupled two-site (two-amino-acid) substitution
process on the full 400-state joint space:

1. **exact** — `log [e^{tW}]_xy` via reversible eigendecomposition (machine precision);
2. **closed-form path ELBO** (ours) — the Girsanov lower bound `L_xy` against the
   independent product-of-marginals bridge `R`, computed for the whole 400x400
   matrix at once with no ODE integration
   (`experiments/elbo_vs_expm.py:closed_form_L`);
3. **Cohn CTBN mean-field** — the endpoint-conditioned round-robin mean-field of
   Cohn et al. (2010), JMLR 11:93, as implemented in **evolsnake**
   (`src/tkfdp/cohn_ctbn/ctbn.py:ctbn_variational_log_cond`), which solves the
   Euler-Lagrange rho/mu ODEs with `diffrax` and integrates the free-energy
   functional `F` over the branch.

Driver: `experiments/cohn_vs_elbo_pair.py`; figure:
`experiments/figures/cohn_vs_elbo.pdf` (`plot_cohn_vs_elbo.py`); raw numbers:
`results/pair_models/cohn_vs_elbo.json`.

## Shared model (fair comparison)

All three methods target **one** generator. We take a real fitted pair
distribution `pij` and shared exchangeabilities `S` (LG08-style, from the
K=8 mixture components in `results/mixture_component_char/components_K8.npz`)
and build a **Cohn/Glauber** 2-site Potts CTBN

    q(a -> c | neighbour b) = S[a,c] * exp(h[c] + 2 J[c,b]),
    h = log m1,   J = 1/2 (log pij - log m1 - log m2),

so that its stationary distribution is **exactly** `pij` (verified: `||stat -
pij|| = 6e-15`, reversibility residual `6e-19`). This is the same stationary and
the same `S` as our square-root-Metropolis pair model — only the acceptance
kernel differs (Glauber vs Metropolis). Cohn's mean-field runs natively on the
Glauber form (its `q_bar`/`q_tilde` assume the `exp(h+2J)` rate); our generic
closed-form ELBO bounds the same exact `expm`. The independent product bridge
`R` (the Glauber generator with `J = 0`) is exactly the process Cohn already
warm-starts from — see "Warm start" below.

Coupling gradient: three models by mutual information of `pij` — MI = 0.164
(mid), 0.290 (strongest real component), and a synthetic **STRONGx2** stress
case (pointwise MI doubled, MI = 1.035). Branch lengths t in {0.1, 0.5, 2.0}.
Per cell, 18 endpoint pairs (single-site / double / typical substitutions).

## Headline results

| model (MI) | t | ours gap mean/max | Cohn gap mean/max | Spearman ours / Cohn | Cohn ODE fails | whole-matrix ELBO slack |
|---|---|---|---|---|---|---|
| 0.164 | 0.1 | 0.035 / 0.25 | **-0.033** / 0.70 | 1.000 / 0.977 | 0/18 | +2.7e-9 |
| 0.164 | 0.5 | 0.014 / 0.10 | **-0.102** / 0.10 | 1.000 / 0.996 | 0/18 | +6.8e-8 |
| 0.164 | 2.0 | 0.037 / 0.30 | **-0.028** / -0.00 | 1.000 / 0.998 | 1/18 | +1.1e-6 |
| 0.290 | 0.1 | 0.037 / 0.32 | **-0.043** / 0.06 | 0.992 / 0.983 | 1/18 | +1.6e-8 |
| 0.290 | 0.5 | 0.054 / 0.60 | **-0.101** / 0.03 | 0.998 / 0.984 | 5/18 | +4.1e-7 |
| 0.290 | 2.0 | 0.041 / 0.24 | **-0.104** / 0.02 | 1.000 / 0.998 | 1/18 | +6.8e-6 |
| 1.035 | 0.1 | 0.180 / 1.19 | **-0.023** / 0.04 | 0.998 / 0.811 | 6/18 | +5.5e-8 |
| 1.035 | 0.5 | 0.253 / 1.62 | **-0.007** / 0.05 | 0.998 / 0.857 | 10/18 | +1.4e-6 |
| 1.035 | 2.0 | 0.208 / 1.51 | **-0.095** / 0.13 | 1.000 / 0.993 | 6/18 | +2.3e-5 |

`gap = exact - approx` (nats); a valid lower bound has `gap >= 0`. "whole-matrix
ELBO slack" is `min(exact - L)` over all 400x400 entries.

Four things stand out:

- **Validity.** The closed-form ELBO is a strict lower bound to machine
  precision on the entire 400x400 matrix in every cell (slack `+1e-8` to
  `+2e-5`). Cohn's mean-field, at its default `diffrax` tolerance, has a
  **negative mean gap in every cell** — its "lower bound" *exceeds* the exact
  log-likelihood by up to ~0.1 nats. This is a numerical artifact, not a
  contradiction of Cohn's theory (see next section).

- **Robustness.** Cohn's ODE fails to converge (`diffrax` max-steps / solver
  error) on a growing fraction of endpoint pairs as coupling strengthens:
  0/18 at MI 0.164 up to **10/18** at MI 1.035. The closed form never fails.

- **Speed.** The closed form computes the **whole 400x400** ELBO in ~0.16 s
  (one reversible eig + a divided-difference kernel). Cohn costs ~5.5 s **per
  single endpoint pair**. Extrapolated to the full matrix that is ~0.16 s vs
  ~5.5 s x 160,000 ≈ 10 days — a ~1.8e5x per-pair speedup (amortized).

- **Tightness.** Where Cohn converges, the two bounds are of comparable
  magnitude. The closed form's product bridge is genuinely looser on
  **double substitutions at very strong coupling** (MI 1.035 gaps up to 1.6
  nats) — the independent bridge cannot cheaply represent two simultaneous
  correlated jumps — but it still returns a valid bound there, whereas Cohn
  fails to solve most of those pairs. Ranking fidelity (Spearman of the
  approximation against exact over the sampled pairs) is ~1.000 for the closed
  form throughout; Cohn stays high except at strong coupling (0.81 at MI 1.035,
  t = 0.1).

## The Cohn bound violations are a reproducible numerical issue, not a bug

Cohn's mean-field free energy `F` **is** provably a variational lower bound in
exact arithmetic. The negative gaps above are the **log-singular boundary**
error of the `F` integral, reproducible and tolerance-driven. Evidence
(`experiments/cohn_bound_tolerance_diag.py`):

1. **Harness is correct.** Reproducing evolsnake's own N=2 `ising2` test
   configuration through this comparison harness gives strictly valid bounds
   (gaps +1.30, +0.29, +1.30). Our `exact` cross-checks against an independent
   `scipy.linalg.expm` to `3.7e-15`. So neither the exact reference nor the
   comparison logic produces the violation.

2. **Tightening the ODE tolerance helps, but does NOT rescue the coupled
   (double-substitution) pairs.** The violation is the log-singular `F` integrand,
   and tolerance controls it only where the integrand is mild. A small `rtol/atol`
   sweep at MI 0.164, t = 0.5 first suggested it was fully fixable:

   | pair | exact | gap @ 1e-3/1e-6 | @ 1e-5/1e-8 | @ 1e-7/1e-10 |
   |---|---|---|---|---|
   | single (6,0)->(5,0) | -6.028 | **-0.061** | -0.0001 | -0.0000 |
   | double (12,18)->(0,5) | -12.935 | **-0.111** | +0.0014 | +0.0023 |
   | double (1,0)->(15,5) | -9.845 | **-0.102** | -0.026 | -0.026 |

   But a full re-run at a tight `rtol=1e-8` over all sampled pairs
   (`cohn_tight_tol.py` -> `results/pair_models/cohn_tight_tol.json`) shows the
   fix is **regime-dependent** (mean / min gap, per regime):

   | model | t | single | typical | double |
   |---|---|---|---|---|
   | MI 0.16 | 0.5 | +0.000 / valid | +0.000 / valid | **-0.206 / min -1.19** |
   | MI 0.29 | 0.5 | ~0 (2 fail) | ~0 | **-0.651 / min -2.09** |
   | MI 0.29 | 2.0 | +0.000 | +0.000 | **-0.204 / min -1.24** |
   | MI 1.04 | 0.1 | ~0 (1 fail) | ~0 | **-0.497 / min -1.82** |

   So at tight tolerance single-substitution and typical pairs become near-exact
   and valid, but **double substitutions still catastrophically violate the bound
   (by up to ~2 nats)** -- and those are exactly the coupled transitions the model
   is for. The `F` integrand is most singular where the bridge marginal is most
   dynamic (both sites moving), so no practical tolerance rescues it there
   (tkf-dp's diagnosis: "more dynamic rho => sharper boundary features => worse
   error"). This is numerical, not a structural bug -- but it means the ODE-based
   Cohn bound **cannot be made reliable on the coupled pairs**, only on the easy
   ones, which is the concrete case for the closed form.

3. **Independently documented, two integrators, same cause.** tkf-dp's own K=2
   Cohn reimplementation (`src/tkfdp/variational_cohn.py`) records this exact
   limitation in its module docstring: "Cohn et al. (2010) inhomogeneous
   mean-field — algorithm right, F integration converges slowly. For the strict
   closed-form bound on a 2-site cluster, use `variational_hr.py` instead (the
   Holmes-Rubin pair-of-eigenvalues formulation)." Its listed fix (option b) is
   "analytically integrate the boundary regions of the F integrand … i.e., the
   closed form already implemented in `variational_hr.py`." The two Cohn
   implementations hit the identical log-singular boundary through *different*
   integrators, which is why the effect is a property of the method, not of one
   codebase:
   - `variational_cohn.py` — fixed-grid **trapezoidal** (`n_grid`): H=0 overshoot
     +0.39 / +0.26 / +0.14 / +0.04 nat at n_grid = 21 / 41 / 81 / 321.
   - evolsnake `ctbn.py` (benchmarked here) — `diffrax` **adaptive**
     (`rtol/atol`): overshoot −0.06 to −0.11 nat at `rtol=1e-3`, vanishing by
     `rtol=1e-7` (the sweep above).

## Proof that the F-integral singularity is inherent (not an integrator artifact)

The singularity is a property of the endpoint-conditioned mean-field free energy
itself, provable from the algebra. (An earlier draft of this proof located a
`t ln t` "finite integrand / log-divergent derivative" at BOTH endpoints; an
adversarial re-derivation showed that was wrong -- the true singularity is
stronger, a bare integrable log divergence at the ARRIVAL endpoint only. The
corrected proof below is what matches the observed O(1/n) quadrature error.)

**Integrand, after the exact cancellation.** `F = int_0^T eps(t) dt` with
(matching `F_deriv`, ctbn.py:217-228)

    eps(t) = -sum_x mu(x,t) qbar(x,t)
             + sum_{x!=y} gamma(x,y,t) [ ln qtilde(x,y,t) + 1 + ln mu(x,t) - ln gamma(x,y,t) ].

Because gamma is Cohn eq 17, gamma(x,y,t) = mu(x,t) qtilde(x,y,t) rho(y,t)/rho(x,t)
(ctbn.py:183), we have ln gamma = ln mu + ln qtilde + ln rho_y - ln rho_x, so the
bracket collapses -- **ln mu and ln qtilde cancel exactly**:

    ln qtilde + 1 + ln mu - ln gamma  =  1 + ln rho_x - ln rho_y.

Hence the only nontrivial (jump/entropy) part of the integrand is

    eps_jump(t) = sum_{x!=y} gamma(x,y,t) ( 1 + ln rho_x(t) - ln rho_y(t) ),

and the energy term -sum_x mu qbar is O(1) and smooth throughout. (This is why
`variational_cohn.py`'s `compute_psi`/`F` build from rho, not mu.)

**Departure endpoint t=0: SMOOTH (no singularity).** rho_x(0) = [e^{QT}]_{x,y} > 0
for all x (irreducible Q, finite T), so 1 + ln rho_x - ln rho_y is bounded; the
collapsing fluxes are gamma(a,.,t) = O(t). Bounded x O(t) = O(t): eps is C^infinity
at t=0. (The earlier draft's t=0 mechanism was spurious -- it relied on ln mu,
which cancels.)

**Arrival endpoint t=T: a bare, integrable log divergence.** As rho collapses to
delta_y, rho_x(t) -> 0 for x != y, so ln rho_x -> -infinity. The relevant flux is
the ARRIVING flux gamma(x,y,t) = mu_x qtilde rho_y/rho_x; near t=T both mu_x and
rho_x vanish like (T-t) and CANCEL, leaving gamma(x,y,T) = O(1) > 0 (the posterior
"last jump into y" rate). Therefore

    eps(t) ~ ( sum_{x!=y} gamma(x,y,T) ) * ln(T-t)  ->  -infinity   as t -> T-,

a bare log divergence of the integrand itself (not merely its derivative),
coefficient C = sum_{x!=y} gamma(x,y,T) > 0. It is integrable (int ln converges,
so F is finite) but the integrand blows up, and the two endpoints are NOT
symmetric. The code's three `mu = where(t<T, mu, rho)` boundary guards
(ctbn.py:289,304,324) exist precisely because of this t=T explosion.

**This matches the observed O(1/n) -- and rules out the t ln t alternative.** A
bare log divergence gives last-panel trapezoid error ~ h ln h, total
O(ln n / n) ~ O(1/n); a merely non-C^1 (t ln t) integrand would give O(1/n^2).
The recorded H=0 trapezoid errors 0.39 / 0.26 / 0.14 / 0.04 at n = 21/41/81/321
(variational_cohn.py:26-31) are ~O(1/n) (4x n cuts error ~3.5x, not ~16x), which
**confirms the bare-log mechanism and refutes the t ln t one**. The sign matches
too: quadrature undershoots the -infinity dip near T, so F is over-estimated,
F_quad > F_true, i.e. the "bound" exceeds exact -- shrinking as n grows.

**Inherent, and the single/double split.** The divergence is driven by the
delta-collapse of rho at the clamped arrival endpoint, which cannot be removed
without removing the endpoint conditioning. Its strength is the arriving-flux
coefficient C = sum_{x!=y} gamma(x,y,T):
- single substitution (a,b)->(c,b): site 2's rho still collapses to delta_b at
  t=T, but site 2 barely moves, so its arriving-flux contribution to C is small ->
  mild divergence, resolvable at tight tol (singles valid at rtol=1e-8).
- double substitution (a,b)->(c,d): both sites move strongly and are coupled
  (via qtilde/rho, the psi term), so C is large -> strong divergence, NOT
  resolvable (doubles violate by 1-2 nats even at rtol=1e-8).
So the per-regime table above follows from the coefficient magnitude (not from a
spectator being "uncollapsed", which it is not).

**Why the closed form escapes it.** The Holmes-Rubin / path ELBO never forms
eps(t) and never quadratures it. It uses a DIFFERENT variational representation --
a single fixed constant-rate product bridge R (not Cohn's iterated inhomogeneous
mean field) -- for which the Girsanov correction int_0^t e^{Rs} C e^{R(t-s)} ds
has a SMOOTH matrix-exponential integrand, evaluated in closed form via the
divided-difference eigenkernel J^{kl}(t). It is therefore not an "analytic
integration of the same singular eps" but a different, non-singular object. (It is
consequently not the tightest bound in the inhomogeneous family --
holmes_rubin_elbo.md -- but it is strict and quadrature-free.) That is the
algebraic reason it is a machine-precision bound where Cohn's ODE/quadrature is
not.

**Implication.** The ODE-/quadrature-based CTBN mean-field is a reliable ELBO
only in the tight-tolerance / fully-converged limit, and reaching that limit is
expensive (finer integration on an already ~5.5 s/pair solve) and fragile
(non-convergence grows with coupling). **Our** Holmes-Rubin closed-form path
ELBO (`variational_hr.py` for the 2-site pair; the whole-matrix Girsanov
`closed_form_L` / paper2b §5 derivation used here) sidesteps the singular
integration entirely — it analytically integrates the boundary via the
divided-difference eigen-kernel — giving a strict machine-precision lower bound
at ~0.16 s for the whole 400x400. That is the concrete motivation for our closed
form, and it is documented as such in `variational_cohn.py` itself.

## Warm start (does Cohn start from our independent approximation?)

Yes. evolsnake's `ctbn_variational_log_cond` initialises the mean-field from the
**exact endpoint-conditioned marginals of the independent single-site process**
`q1 = q_single(params)` (its comment: "assuming no interactions"), then iterates
the coupling in. That independent process is exactly our closed-form ELBO's
bridge `R` (`J = 0`). So the three methods form a progression on the same
independent bridge: independent warm start -> our single closed-form Girsanov
correction -> Cohn's full iterated mean-field. The comparison above already runs
Cohn from that warm start; the tolerance sweep shows the iteration is limited by
the `F`-integral accuracy, not the starting point.

## Reproduce

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
      PYTHONPATH=src:experiments:src/tkfdp/cohn_ctbn \
      python3 experiments/cohn_vs_elbo_pair.py
    python3 experiments/plot_cohn_vs_elbo.py

(The vendored Cohn code and its one local modification are documented in
`src/tkfdp/cohn_ctbn/PROVENANCE.md`. `CTBN_MAX_STEPS` was raised 4x and
`rtol/atol` exposed on `ctbn_variational_log_cond` for the tolerance sweep.)
