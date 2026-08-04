# The product trap as an OPTIMIZATION obstruction: an exact-lumpability HR-EM test (n=4)

Status: SETTLED at n=4 with a reliable exact-lumpability HR-EM fit. This supersedes the
"NOT yet established" caveat in `analysis/lumpable_product_trap.md` Section 6 (which only
had a noisy soft-Lagrangian L-BFGS probe). Code: `experiments/lumpable_hr_em_fit.py`.
Results: `results/lumpable_trap/` (per-cell `fit_*.npz`, `results.json`).

## The question

`analysis/lumpable_product_trap.md` PROVES that, linearised at the product chain, a
reversible two-sided-lumpable perturbation can switch on a stationary coupling only in the
RESONANT tangent span{psi_a (x) psi_b : lam_a = lam_b}. For a structured single-site
background W with DISTINCT eigenvalues the reachable tangent is just the n-1 agreement
modes psi_a (x) psi_a, and every non-resonant ("OUT") coupling is first-order unreachable
(the "trap"). The theorem is about the tangent geometry; it does NOT by itself say whether
a real trainer stalls, nor whether renewal initialisation cures it. This note answers
both, at n=4, with an exact fit.

## The fitter (exact lumpability, not a penalty)

State space: ordered pairs x=(i,j) in [4]^2 (16 states). Reversible, component-
exchangeable generator Q_xy = F_xy / pi_x with F symmetric and constant on Klein-4
(V4) orbits, so Q is reversible w.r.t. an exchangeable joint pi and exchangeable under
coordinate swap.

- E-step: exact Holmes-Rubin endpoint-conditioned bridge (`tkfdp.permfield.hr.bridge`,
  `eig_rev`), summed over the endpoint-joint data, giving expected transition usage N_xy
  and dwell T_x (the same machinery as `fit_pair_models.estep`, generalised to n=4).

- Flux M-step, EXACT lumpable (route (ii)): at fixed pi the two-sided strong-lumpability
  constraint on the orbit flux phi is HOMOGENEOUS LINEAR, C(pi) phi = 0 (it encodes
  "sum_l Q[(i,j)->(k,l)] independent of j"; the Y-side follows by exchangeability). The
  complete-data auxiliary sum_o [C_o log phi_o - H_o phi_o] is concave; its KKT point is

      phi_o = C_o / (H_o + (C^T mu)_o),

  and mu is found by a damped, feasibility-guarded Newton on the convex dual
  Psi(mu) = -sum_o C_o log(H_o + (C^T mu)_o) (the rational-dual / Sinkhorn analogue).
  Positivity phi_o >= 0 is AUTOMATIC (C_o >= 0, denominator > 0). Two conditioning fixes
  make it reliable on skewed backgrounds: (a) each constraint row is normalised to unit
  norm (leaves null(C) unchanged), (b) C_o, H_o are scale-normalised by the total usage
  (phi invariant). The dual then converges in ~10 iterations to C phi = 0 at ~1e-16
  relative, i.e. EXACT lumpability.

- pi-step, EXACT constrained: the coupled exchangeable stationary is a free block. We
  PROFILE the flux out -- m(pi) = max_{phi lumpable, >=0} sum_o[...] - sum_x N_x^row log
  pi_x -- and locally maximise m over pi (10 exchangeable log-weights, bounded) with a
  warm-started L-BFGS. Because the flux is re-solved on null(C(pi)) at every trial pi,
  the (pi, flux) pair is EXACTLY lumpable at every point. This is a generalised EM M-step
  (Q(theta_new) >= Q(theta_old)), so the observed-data log-likelihood is monotone.

The Synchronized variant (used only to validate the E-step/LL) drops C(pi): flux M-step
phi_o = C_o / H_o over all orbits, standard reversible-stationary pi-step.

Observed-data objective (reported LL): sum_{x,y,t} n_xy(t) log P_xy(t), P = expm(Qt).

## Verification (done BEFORE any trap conclusion)

1. Recover a KNOWN lumpable chain -- a joint renewal Q_xy = mu pi_y with a coupled pi
   (MI 0.25): from renewal init the fit returns

       MI_fit = 0.24999  (true 0.25000),  strong-lumpability residual of fitted Q = 5e-11
       (true Q residual 0.0),  EM monotone (max LL decrease 0.0).

2. Synchronized recovers a KNOWN general reversible chain (random exchangeable flux,
   coupled pi, MI 0.15):

       MI_fit = 0.15015  (true 0.15000),  LL_fit/N = -1.44083  (true -1.44078),
       EM monotone.

Both pass: the fitter recovers coupling exactly on representable data, its E-step/LL are
correct, lumpability is exact (~1e-11), and EM is monotone. The trap results below are
therefore trustworthy.

## The controlled trap / renewal-init experiment

Backgrounds (both with DISTINCT eigenvalues -> a real trap, reachable dim = n-1 = 3):
- generic: a random reversible exchangeable W (eig 0, -1.262, -1.311, -1.894).
- hky85skew: HKY85 kappa=4, u=(0.1,0.2,0.3,0.4) (eig 0, -1.0, -1.9, -3.1).

Couplings (exchangeable, zero-marginal), verified by the exact eigen-mode resonant
fraction:
- delta_IN = agreement mode psi_a (x) psi_a  (resonant fraction 1.000, IN-tangent).
- delta_OUT = cross mode psi_a (x) psi_b + psi_b (x) psi_a, lam_a != lam_b
  (resonant fraction 1e-32, OUT of tangent), chosen per background at the highest
  positivity-feasible MI.

Each coupling is scaled to a clear stationary MI (target 0.20, capped by positivity), and
endpoint data J(x,y) = pi_x expm(Qt)_xy are generated over tau = {0.15, 0.4, 0.8, 1.5}
at 4e5 pairs/bin, TWO ways:
- CONDITIONAL: Metropolis-sqrt SINGLE-transition chain reversible w.r.t. pi (proposal
  exchangeability = W's, acceptance sqrt(pi_dest/pi_src)). One coordinate changes at a
  time -- NO concerted double substitutions. NOT lumpable in general.
- LUMPABLE: joint-renewal chain Q_xy = mu pi_y (jump to a fresh joint draw). Two-sided
  lumpable; carries concerted double-substitution signal. Exactly representable.

The lumpable model is fit from TWO inits: product (pi = product of marginals, small
interior flux) and renewal (pi = empirical coupled joint, Q_xy = mu pi_y).

## Results

MI(fitted stationary) and per-site observed-data LL, every cell exact + monotone:

    bg          dir  data         init      MI_true   MI_fit      LL/N    lumpres
    generic     IN   conditional  product    0.2000   0.0458   -1.43155  4.2e-15
    generic     IN   conditional  renewal    0.2000   0.0458   -1.43155  2.0e-15
    generic     IN   lumpable     product    0.2000   0.2000   -1.57266  9.0e-11
    generic     IN   lumpable     renewal    0.2000   0.2000   -1.57266  2.8e-10
    generic     OUT  conditional  product    0.1376   0.0515   -1.46801  8.2e-13
    generic     OUT  conditional  renewal    0.1376   0.0515   -1.46801  8.7e-13
    generic     OUT  lumpable     product    0.1376   0.1376   -1.60262  8.5e-12
    generic     OUT  lumpable     renewal    0.1376   0.1376   -1.60262  2.2e-16
    hky85skew   IN   conditional  product    0.2000   0.0842   -1.34639  1.4e-13
    hky85skew   IN   conditional  renewal    0.2000   0.0842   -1.34639  1.4e-13
    hky85skew   IN   lumpable     product    0.2000   0.2000   -1.55460  4.6e-10
    hky85skew   IN   lumpable     renewal    0.2000   0.2000   -1.55460  1.0e-10
    hky85skew   OUT  conditional  product    0.2000   0.0001   -1.37415  1.1e-12
    hky85skew   OUT  conditional  renewal    0.2000   0.0468   -1.36889  2.3e-15
    hky85skew   OUT  lumpable     product    0.2000   0.2000   -1.55310  1.3e-15
    hky85skew   OUT  lumpable     renewal    0.2000   0.2000   -1.55310  1.7e-16

Product-minus-renewal within each same-data cell (the optimization-trap signal):
- Every LUMPABLE-data cell: dMI = 0.0000, dLL/N = 0.00000 (product == renewal).
- generic IN/OUT conditional, hky IN conditional: dMI = 0.0000, dLL/N = 0.00000.
- hky85skew OUT conditional: dMI = +0.0467, dLL/N = +0.00526 (renewal strictly better).

## The three verdicts

TRAP -- does product-init stall (MI ~ 0) for delta_OUT and not for delta_IN? On which
data type?
- On LUMPABLE (representable) data: NO stall in ANY direction. Product-init recovers the
  FULL true MI for both IN and OUT (generic OUT 0.1376 = true; hky OUT 0.2000 = true).
  The data's concerted double-substitution signal drives the flux off the degenerate
  product point in the first EM step, and pi then moves freely to the coupled optimum. The
  first-order trap does not obstruct a real HR-EM trainer here.
- On CONDITIONAL (single-substitution) data: a genuine product-init stall at MI ~ 0
  appears ONLY for the DEEP-trap cell hky85 OUT (product MI 0.0001 vs a best lumpable fit
  of 0.0468). It does NOT stall for delta_IN (hky IN product = renewal = 0.0842), and it
  does NOT stall for the shallower generic OUT (product = renewal = 0.0515). So the
  textbook "stalls for OUT, not for IN" is realised cleanly on the hky85 background --
  whose OUT mode (1,3) has the largest spectral gap (lam -1 vs -3.1) -- and is absent on
  the small-gap generic OUT mode (2,3) (lam -1.311 vs -1.894), where the second-order
  escape is strong enough to reach the optimum from the product.

RENEWAL FIX -- does renewal-init reach higher MI / better LL than product for delta_OUT,
by how much?
- Only where the trap actually bites, hky85 OUT conditional: renewal reaches MI 0.0468
  vs product 0.0001 (dMI = +0.0467) at a strictly better fit, LL/N -1.36889 vs -1.37415
  (dLL/N = +0.00526 nats/site). Elsewhere renewal and product land at identical MI and LL
  (dMI = dLL = 0) -- renewal neither helps nor hurts (product already reaches the same
  optimum). So renewal init is a real cure exactly in the deep-trap slice and a no-op
  everywhere else.

TRAP vs REPRESENTABILITY (conditional delta_OUT) -- is the best-LL lumpable fit itself
low-MI, or high-MI reachable only from renewal?
- The best-LL lumpable fit is itself LOW MI. Renewal init starts from the true coupled
  pi (MI 0.20) and the EM pulls it DOWN to MI 0.0468 (hky) / 0.0515 (generic) -- far below
  the true MI. So the lumpable family CANNOT represent the conditional, context-dependent
  single-substitution coupling: that is a genuine REPRESENTABILITY limit (a lumpable chain
  has context-free block rates and cannot reproduce the conditional chain's context-
  dependent transitions).
- On TOP of that representability limit, hky85 OUT shows a PURE OPTIMIZATION trap: the
  best lumpable fit (MI 0.0468) is reachable from renewal but NOT from product (product
  stalls at 0.0001 with strictly worse LL). generic OUT shows no such optimization trap --
  both inits reach the same 0.0515 -- only the representability limit.

## Crisp verdict

The product trap is REAL but its operational bite is narrow.

1. It does NOT obstruct a well-behaved exact-lumpability HR-EM trainer on representable
   (lumpable / renewal) data: the concerted double-substitution signal breaks the first-
   order degeneracy and product-init recovers the full coupling, IN or OUT, identically to
   renewal-init.
2. On non-representable (conditional, single-substitution) data the dominant limitation is
   REPRESENTABILITY, not optimization: even a renewal fit that starts at the true coupling
   collapses to a low-MI lumpable compromise (0.047-0.084 vs true 0.14-0.20).
3. The pure OPTIMIZATION trap -- product-init stalling BELOW the best lumpable fit -- is
   realised only when BOTH conditions hold: the data carry no concerted signal AND the
   coupling is non-resonant with a large spectral gap (hky85 OUT). There, and only there,
   renewal init is a genuine cure (MI 0.0001 -> 0.0468, LL/N +0.0053), though it still
   cannot overcome the representability ceiling.

So: renewal initialisation is a correct-but-partial fix. It removes the optimization
stall exactly where the trap bites, but it does not (and cannot) recover a coupling that
the two-sided-lumpable family cannot represent. Any downstream use of a lumpable coevolution
head should therefore worry less about the product-trap optimization pathology (broken by
real concerted signal; renewal-curable otherwise) and more about whether the target
coupling is representable as context-free block dynamics at all.

## Reproduce

    OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONPATH=src \
      python3 experiments/lumpable_hr_em_fit.py --all --out results/lumpable_trap/results.json

Runs the verification (V1, V2) and all 16 trap cells (~3.5 min); writes per-cell
`results/lumpable_trap/fit_*.npz` and the JSON summary. Every fit asserts/records exact
lumpability (residual <= ~1e-10) and EM monotonicity.
