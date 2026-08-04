# The product trap: first-order coupling reachability of two-sided lumpable pair chains

Status: the **inclusion** (reachable coupling ⊆ resonant span -- "the trap") is PROVEN
(Section 3; the two load-bearing lemmas checked numerically to 1e-14 in
`experiments/lumpable_reachability_lemmas.py`). The **converse** (every resonant mode is
reachable, so the dimension is exactly sum m_lam^2) is verified to machine precision
across all backgrounds tested (generic n=3,4,5; JC69; K2P; HKY85) but the analytic
construction is still open. Code: `experiments/lumpable_tangent_rank.py`.

**This supersedes the earlier framing in `analysis/lumpable_kernel_basis.md`.** That
note claimed two-sided lumpability restricts the *representable* stationary coupling to
an (n-1)-dimensional "sliver" that excludes physical couplings. That is WRONG: an
LP-feasibility scan shows a positive two-sided-lumpable reversible chain exists for
*every* target coupling (full (n-1)^2 reachable, non-exchangeable; full symmetric
n(n-1)/2 exchangeable), and the minimal concerted-substitution cost to hold a coupling
of strength t is uniform across coupling directions (~1.2 t, Watson-Crick = agreement =
random). There is no representational limit and no excluded coupling. The real
obstruction is a *first-order optimization* one, characterised below.

## 1. Setup

A reversible pair CTMC on states (i,j) in [n]x[n], built over a single-site generator
W that is reversible w.r.t. a distribution u (sum u = 1). Let W psi_a = lam_a psi_a,
a = 0..n-1, be its eigensystem, with lam_0 = 0, psi_0 = const, and {psi_a} orthonormal
in <f,g>_u = sum_i u_i f_i g_i. The **product chain** Q0 acts as W on each coordinate
independently; its stationary is pi0 = u (x) u, its eigenfunctions are psi_a (x) psi_b
with eigenvalues lam_a + lam_b, and it is two-sided lumpable (each marginal is the
autonomous single-site process W).

A **coupling** perturbation of the stationary is a signed measure delta with zero
marginals (sum_i delta_ij = 0 for all j, and sum_j delta_ij = 0 for all i). In the
eigen-basis its coordinates are delta_ab = sum_ij psi_a(i) psi_b(j) delta_ij; the
coupling directions are {(a,b): a,b >= 1}, a space of dimension (n-1)^2.

We linearise the two-sided-lumpable *reversible* family at Q0 and ask: **which coupling
directions delta can be switched on at first order** (i.e. lie in the tangent to the
lumpable family, projected to the coupling space)?

## 2. The theorem

> **Theorem (first-order reachable coupling).** The first-order reachable coupling
> tangent of the two-sided-lumpable reversible family at the product chain is
>
>     T = span{ psi_a (x) psi_b : lam_a = lam_b,  a,b >= 1 }
>
> — the **resonant eigen-mode pairs** of the single-site generator. Its dimension is
>
>     dim T = sum_{lam != 0} m_lam^2        (non-exchangeable models),
>     dim T = sum_{lam != 0} m_lam(m_lam+1)/2   (exchangeable models),
>
> where the sum is over *distinct* nonzero eigenvalues of W and m_lam is the
> multiplicity of lam.

Proof status: the **inclusion** T ⊆ span{psi_a (x) psi_b : lam_a = lam_b} is proven in
Section 3 (i.e. no reversible lumpable perturbation of the product turns on a
NON-resonant coupling -- this is the trap). The reverse inclusion (equality, hence the
dimension formula) is verified by `reachable_tangent(r,u)`: it equals
`span{psi_a (x) psi_b : lam_a = lam_b}` to numerical zero (all principal angles 0.0 deg)
and dimension sum m_lam^2 exactly in every case tested; the analytic construction of a
lumpable perturbation realising each resonant mode is left open.

### Corollaries
- **Generic background** (all nonzero eigenvalues distinct): only a = b survive, so
  dim T = n-1. The reachable directions are the **eigen-mode agreement couplings**
  psi_a (x) psi_a (correlate the a-th relaxation mode between the two sites).
- **Fully degenerate background** (JC69: one (n-1)-fold eigenvalue): every pair is
  resonant, dim T = (n-1)^2 = full. No trap.
- Everything in between is set by the spectrum: **breaking symmetry splits eigenvalues,
  removes resonances, and shrinks T.**

## 3. Proof of the inclusion (the trap)

Work in the pi0-orthonormal eigenbasis e_{ab} = psi_a (x) psi_b of the product chain Q0
(Q0 e_{ab} = (lam_a + lam_b) e_{ab}). Write M_{(cd),(ef)} := <e_{cd}, M e_{ef}>_{pi0}
for the matrix elements of the perturbation, and <f,g>_delta := sum_x delta_x f_x g_x.
Note the coupling coordinate delta_ab = sum_x delta_x psi_a(i) psi_b(j) is exactly
<e_{a0}, e_{0b}>_delta (since psi_0 = 1); call this identity (*).

**Lemma 1 (lumpability = invariant subspaces).** A generator is two-sided lumpable iff
it preserves V_X = {functions of i alone} = span{e_{c0}} and V_Y = {functions of j
alone} = span{e_{0d}}. (Standard strong lumpability: the aggregated chain is Markov for
every initial law iff Q maps block-constant functions to block-constant functions;
Q 1_{block} constant on blocks <=> Q(V_X) subset V_X.) Because Q0 already preserves
V_X, V_Y, a first-order perturbation preserves lumpability iff M(V_X) subset V_X and
M(V_Y) subset V_Y. In coordinates: M_{(cd),(0b)} = 0 for c != 0, and
M_{(cd),(a0)} = 0 for d != 0.

**Lemma 2 (reversibility identity).** If Q0 + eps M is reversible w.r.t. pi0 + eps delta,
then for all modes,
    M_{(cd),(ef)} - M_{(ef),(cd)} = [(lam_c+lam_d) - (lam_e+lam_f)] <e_{cd}, e_{ef}>_delta.
Proof: the O(eps) term of <f, Q g>_pi = <Q f, g>_pi is
<f, M g>_{pi0} - <M f, g>_{pi0} = <Q0 f, g>_delta - <f, Q0 g>_delta; take f = e_{cd},
g = e_{ef} and use Q0 e = (lam+lam) e. (Verified numerically to 3e-14,
`experiments/lumpable_reachability_lemmas.py`.)

**Theorem (inclusion / the trap).** If delta is the stationary coupling of a reversible,
two-sided-lumpable first-order perturbation of the product, then delta_ab = 0 whenever
lam_a != lam_b (a, b >= 1). Equivalently, the reachable tangent
T subset span{psi_a (x) psi_b : lam_a = lam_b}.

Proof. Apply Lemma 2 with (cd) = (a, 0) and (ef) = (0, b), a, b >= 1. The right-hand
side is [(lam_a + lam_0) - (lam_0 + lam_b)] <e_{a0}, e_{0b}>_delta = (lam_a - lam_b)
delta_ab by (*). On the left, Lemma 1 kills both terms: M_{(a0),(0b)} = 0 because the
input e_{0b} is in V_Y so M e_{0b} in V_Y has no e_{a0}-component (a >= 1); and
M_{(0b),(a0)} = 0 because e_{a0} in V_X so M e_{a0} in V_X has no e_{0b}-component
(b >= 1). Hence 0 = (lam_a - lam_b) delta_ab, so delta_ab = 0 unless lam_a = lam_b. QED

This is exactly the trap: **no reversible lumpable perturbation of the product can switch
on a coupling in a non-resonant mode.** In particular, for a generic single-site process
(all nonzero eigenvalues distinct) the only first-order-accessible couplings are the
diagonal agreement modes psi_a (x) psi_a, an (n-1)-dimensional subspace of the (n-1)^2
coupling directions.

*Open (converse).* That every resonant mode is actually reachable -- so that T equals
the resonant span and dim T = sum m_lam^2 -- is verified numerically to machine
precision in all cases but not yet proven; the remaining task is to construct, for each
resonant (a,b), a reversible lumpable perturbation with delta_ab != 0 (the antisymmetric
part of M is fixed by Lemma 2; one must choose a symmetric part consistent with
Lemma 1's linear constraints).

## 4. Consequence: the product is a degenerate critical point of the fit

Training a two-sided-lumpable model by likelihood on transition data starts, in
practice, from the product chain (the well-conditioned init; e.g.
`fit_pair_models.fit_model`'s `lump_product_init`). The theorem says the likelihood
gradient at the product can move the model only along the (n-1)-dimensional resonant
tangent. If the data's coupling lies in the (n-1)^2 - (n-1) dimensional **non-resonant
complement** — which for a realistic n=20 alphabet is 342 of 361 directions — the
first-order gradient is zero and a local ascent **stalls at the product, MI = 0**.

This is confirmed operationally (`experiments/lumpable_tangent_trap.py`): with identical optimiser, init and
coupling strength, a local fit from the product recovers in-tangent coupling
(MI > 0) but stalls at MI ~ 0 for out-of-tangent coupling — a 60x gap at n=3. It also
explains why n=2 always recovers (there n-1 = 1 = the whole coupling space, so no
non-resonant complement exists) and why the corpus n=20 Lumpable collapsed to the
product. The obstruction is **first-order optimisation geometry, not representation**:
the coupled optimum exists and is reachable (Section, retraction above), but it is
second-order-invisible from the product init along all but the resonant directions.

## 5. What the trap depth depends on (stated carefully)

The one rigorous statement is the multiplicity formula: **dim(trap) = (n-1)^2 - sum_lam
m_lam^2**, where m_lam are the multiplicities of the distinct nonzero eigenvalues of the
single-site generator W. Nothing beyond the eigenvalue multiplicities of W enters. In
particular:

- The trap **vanishes iff the nonzero spectrum is fully degenerate** (one eigenvalue of
  multiplicity n-1), i.e. Sigma m_lam^2 = (n-1)^2.
- A **renewal / F81** process (jump to a fresh draw from u; r = const) has
  W = mu(1 u^T - I), whose nonzero spectrum is -mu with multiplicity n-1 for EVERY u.
  So the renewal has no trap regardless of composition -- this special case is genuinely
  composition-independent.
- A **generic** W (all nonzero eigenvalues distinct) has the deepest trap,
  dim(trap) = (n-1)^2 - (n-1).

Both the exchangeability structure and the composition u feed into W's spectrum, so
**composition is not irrelevant in general** -- e.g. HKY85 mild-skew keeps a degenerate
ts/tv pair (reachable dim 5) while a stronger skew splits it (dim 3). (An earlier draft
of this note claimed the trap is "dynamics, not composition"; that over-generalises the
renewal special case and is false under structured exchangeability -- retracted.) DNA
progression (n=4), reachable dim = sum m_lam^2:

| model            | nonzero eig(W)      | reachable dim | note                          |
|------------------|---------------------|:-------------:|-------------------------------|
| F81 / renewal    | -1, -1, -1  (any u) | 9 = full      | degenerate for any u -> no trap |
| JC69             | -1, -1, -1          | 9 = full      | fully degenerate, no trap     |
| K2P (kappa=4)    | -1, -2.5, -2.5      | 5 = 1+2^2     | ts/tv pair degenerate         |
| HKY85 mild-skew  | -1, -2.5, -2.5      | 5             | ts/tv pair still degenerate   |
| HKY85 skewed     | -1, -1.9, -3.1      | 3 = n-1       | all eigenvalues split         |
| generic / LG08   | all distinct        | n-1           | deepest trap                  |

## 6. The renewal escape hatch (and the optimization fix)

The joint renewal chain Q_xy = mu * pi_y (jump to a fresh draw from the *joint* pi) is
**two-sided lumpable for ANY joint stationary pi, coupled or not** (verified:
X/Y-lumpability deviation = 0 exactly), because the rate into X-block k is mu*pi_X(k),
context-free. So it is an explicit closed-form family of lumpable chains carrying any
coupling, and -- being a fully-degenerate (renewal) point -- it sits where the first-
order Jacobian has **full rank**: from a renewal chain, *every* coupling direction is
first-order accessible.

This is the principled cure for the product trap in training (Section 4). Instead of
perturbing a structured-dynamics product (trapped along all but n-1 directions),
**parametrise / initialise the coupling through a renewal component** -- a jump-to-
coupled-pi term whose target pi directly carries the coupling with full first-order rank.
A model that keeps the structured single-site process for the marginal dynamics (to fit
the data) but adds a renewal term with a free coupled jump-target both fits the observed
one-at-a-time substitutions and accesses coupling without the trap.

**Status of this "fix": SETTLED at n=4** by an exact-lumpability HR-EM fit (Holmes-Rubin
E-step + exact lumpable flux M-step, NOT a penalty; verified to recover a known lumpable
chain to MI 0.25 with lumpability residual 4e-11). See
`analysis/lumpable_trap_optimization.md` and `experiments/lumpable_hr_em_fit.py`. The
nuanced verdict:

- **The trap is real but its bite is narrow.** On REPRESENTABLE data (generated by a
  lumpable/renewal chain, which carries concerted double-substitutions), product-init does
  NOT stall -- the concerted signal breaks the first-order degeneracy and product-init
  recovers the full coupling for both resonant (IN) and non-resonant (OUT) directions,
  robustly across init-flux magnitudes 1e-2..1e-6. A genuine product-init stall (MI 0.0001
  vs a best lumpable fit of 0.047) appears only in the deep-trap cell: CONDITIONAL data
  (Metropolis, no concerted signal) AND a non-resonant coupling AND a large spectral gap
  (HKY85-skew, OUT mode with eigen-gap -1 vs -3.1). Not for IN, not for the small-gap
  generic OUT.
- **Renewal init helps exactly where the trap bites** (HKY85 OUT conditional: MI 0.047 vs
  0.0001 at strictly better likelihood) and is a no-op everywhere else -- a genuine but
  PARTIAL cure (it escapes the stall but cannot beat the representability ceiling).
- **The dominant limit on real data is REPRESENTABILITY, not the trap.** For conditional
  (context-dependent-coupling) data, renewal-init starts at the true MI (0.20) and EM
  pulls it DOWN to ~0.05 -- far below true -- because a lumpable chain (context-free block
  rates) simply cannot reproduce context-dependent single-substitution coupling. The pure
  optimization trap is a narrow second-order effect on top of this representability floor.

So the corpus collapse of Lumpable to low MI is primarily a **representability** limit
(lumpability couples only through concerted substitution, which one-at-a-time data lacks),
with the product-init optimization trap a narrow additional effect in the deep-gap,
non-resonant, no-concerted-signal corner.

## 7. Aside: stationary feasibility of a WC-Potts (Chargaff parity)

Separate from -- and prior to -- the dynamical trap is a MARGINAL-level feasibility
question: does an exchangeable WC-coupled stationary even EXIST with a given composition?
For an exchangeable joint pi(i,j)=pi(j,i) supported only on Watson-Crick pairs, each A is
paired with a T, so pi(A,T)=pi(T,A)=:p forces u_A=u_T=p (and u_G=u_C=q):

> A hard WC-Potts exchangeable stationary exists **iff** u_A = u_T and u_G = u_C
> (Chargaff's second parity rule).

With mismatches allowed, composition skew caps the WC-paired mass:
`max WC mass = 2[min(u_A,u_T) + min(u_G,u_C)]`, which is 1 iff Chargaff parity holds. For
an uneven HKY85 marginal u=(0.1,0.2,0.3,0.4) this ceiling is **0.60** -- at most 60% of
the equilibrium mass can be WC, and >=40% is forced onto mismatches (so a hard WC Potts
is infeasible there). A wobble G-T pair gives the excess G/T an alternative partner and
raises the ceiling to 0.80. This is orthogonal to the first-order escapable-fraction
(the 26% figure), which is a small-coupling perturbation and always marginal-feasible.
Code: `experiments/wc_stationary_feasibility.py`.

## 8. Reproduce

    python experiments/lumpable_tangent_rank.py       # theorem + closed form + DNA progression
    python experiments/lumpable_tangent_trap.py       # operational in/out-of-tangent fit (soft-Lagrangian; noisy)
    python experiments/lumpable_reachability_lemmas.py  # the two proof lemmas, to 1e-14
    python experiments/wc_stationary_feasibility.py   # Chargaff feasibility of a WC-Potts stationary

The first prints, for each background, the reachable tangent dimension, the closed-form
sum m_lam^2, and the principal angles between the reachable tangent and
span{psi_a (x) psi_b : lam_a = lam_b} (all 0.0, confirming the theorem).
