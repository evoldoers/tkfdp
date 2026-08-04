# The Watson-Crick Potts joint two-site model: closed-form CTMC algebra

Status: an algebra-first companion to `psb-paper/paper2b-symmetry.tex` sec:pair. Every
closed form here is either **proven symbolically** (sympy, marked [S]) or **verified
numerically to <1e-9** against a direct build/eigendecompose/Sinkhorn (marked [N]).
Reproduce: `python3 experiments/wc_potts_joint_model.py`. This is the DNA rung between the
binary coupled telegraph (sec:pairbinary) and the amino-acid pair chain: two contacting
nucleotide sites, a proposal with no ts/tv structure, a Potts energy on the base-pair
edges, and the emergent single-site consequences.

## 0. The model (used throughout)

Two sites x,y in {A,C,G,T}. A reversible single-site proposal with NO ts/tv bias -- JC69
(uniform q) or F81 (q = phi). A symmetric Potts energy on the joint state supported only
on the biophysical base-pair edges:

    E(i,j) = E_at  on {A,T}   (weak WC pair, 2 H-bonds)
             E_cg  on {C,G}   (strong WC pair, 3 H-bonds)
             E_gt  on {G,T}   (wobble)
             0     otherwise.

Sign convention (codebase): pi_joint(i,j) ~ exp(-E(i,j)), so a MORE-favored pair has
LOWER energy. Target joint stationary  pi(i,j) = q(i) q(j) exp(-E(i,j)) / Z. Write the
pair weights and their half-Boltzmann roots (these are the natural variables):

    a = e^{-E_at},  c = e^{-E_cg},  w = e^{-E_gt};   alpha = sqrt(a),  gamma = sqrt(c),  omega = sqrt(w).

A key structural fact used everywhere: the three Potts edges A-T, C-G, G-T are ALL
purine-pyrimidine pairs, i.e. all THREE bonds sit on transversion edges; the two
transition edges A-G (purine-purine) and C-T (pyrimidine-pyrimidine) carry zero energy.

**Generator.** Single-site Metropolis-sqrt moves with the F81 proposal (only ONE site
changes per event); with M(i,j) := exp(-E(i,j)),

    Q_{(i,j)->(i',j)} = q(i') * sqrt( M(i',j) / M(i,j) ),   and symmetrically for site y.

This equals proposal(i->i')=q(i') times the sqrt-Hastings acceptance sqrt(pi(i',j)
prop(i'->i) / pi(i,j) prop(i->i')) = sqrt(M(i,j)/M(i',j))... = the rule above. It is
reversible w.r.t. pi with symmetric flux  F_{(i,j),(i',j)} = q(i)q(j)q(i')/Z *
sqrt(M(i,j)M(i',j))  [S, general q and general E]. Stationary/symmetry statements below
are construction-INDEPENDENT; spectrum, W_eff and ts/tv are construction-DEPENDENT (they
use this kernel; JC69 = q uniform unless F81 is named).

---

## Q0. Symmetry breaking, spectrum, and the emergence of a ts/tv bias

**Symmetry group.** A nucleotide permutation g is a model symmetry (applied
simultaneously to both sites) iff it is a label-preserving automorphism of the bond graph
{A-T:E_at, C-G:E_cg, G-T:E_gt} and preserves q. The full symmetry group of the 16-state Q
adds independent per-site relabelings (in the high-symmetry regimes) and the
always-present component exchange E:(i,j)->(j,i). Its irreducible representations are the
eigenvalue degeneracies of Q. Group orders and spectra [N, JC69]:

| regime | condition | diag nuc-auto group | \|G\| | eig(Q) multiplicities |
|---|---|---|---|---|
| (i)   | all E = 0 | S4 (24) | 1152 | 0(x1), -1(x6), -2(x9) |
| (ii)  | E_at=E_cg, E_gt=0 | D4 (8) | 48 | six distinct, mults 1,3,3,1,3,5 |
| (iii) | E_at!=E_cg, E_gt=0 | Klein-4 {e,(AT),(CG),(AT)(CG)} (4) | 8 | 15 distinct (one pair still degenerate) |
| (iv)  | E_gt != 0 | trivial (1) | 2 | 16 distinct (only E survives) |

The concrete permutations: **Watson-Crick complementation sigma = (A T)(C G)** survives
iff E_gt = 0 (it sends the wobble G-T to the zero-edge C-A), and it survives *regardless*
of E_at vs E_cg -- so it lives through regime (iii). The single flips (A T) and (C G) also
survive iff E_gt = 0. The wobble is what kills all of them: at E_gt != 0 only component
exchange E remains.

**Degeneracies lift as consequences of the group.** At E=0 the chain is the independent
product Q0 = W (+) W with W = JC69 (eigenvalues 0, -1, -1, -1), so its 16 eigenvalues are
lam_a + lam_b: the level -1 is 6-fold [(a,0) and (0,a)] and -2 is 9-fold [(a,b), a,b>=1].
Turning on a small WC energy splits these exactly as the group shrinks [N, m=1.001]:

    0(x1),  -1(x6) -> -1(x3) + -1(x3),  -2(x9) -> -2(x1) + -2(x3) + -2(x5).

At regime (ii), m=2, several of the split levels have closed forms, e.g. -(1+sqrt2)/2,
-5 sqrt2/4, -(3+sqrt2)/2 [N]. Each broken symmetry removes a resonance and lifts a
degeneracy; by regime (iv) only the sym/antisym pairing forced by E remains.

**The payoff: a ts/tv bias emerges from the Potts energies alone.** Define the
stationary transition/transversion FLUX ratio (2 ts edges, 4 tv edges):

    kappa_flux = [ F(A-G) + F(C-T) ] / 2   over   [ F(A-C)+F(A-T)+F(C-G)+F(G-T) ] / 4.

For JC69 this collapses to a single closed form, and -- the clean result --

    kappa_flux - 1 = (2 - alpha - gamma)(1 - omega) / [ 2 (1 + alpha + gamma + omega) ]     [S].

So even though the proposal has NO ts/tv structure, the Potts energies induce one, and the
mechanism factorises exactly:

- **The wobble E_gt is the on/off switch.** kappa_flux == 1 for ALL E_at, E_cg whenever
  E_gt = 0 (omega = 1 kills the second factor). No wobble => no net ts/tv flux bias, no
  matter how strong the canonical bonds.
- **The WC bonds E_at, E_cg set the amplitude and sign** through (2 - alpha - gamma).
  Transition-enriched (kappa_flux > 1) exactly when the two factors AGREE in sign: both
  canonical bonds and the wobble favorable (alpha+gamma>2 AND omega>1), or both
  unfavorable. Opposite signs give transversion enrichment.

Mechanism in one line: the equilibrium sits on the canonical WC pairs; from A-T a
transition A->G lands on the wobble G-T, and from C-G a transition C->T also lands on
G-T -- so a favorable wobble makes transitions cheap relative to the mismatch-forming
transversions. Algebraically the wobble enters the transition overlaps as the bridge
terms sqrt(a*w) and sqrt(c*w) through the shared wobble partner (see Q2).

---

## Q1. Marginal distortion and the symmetric Sinkhorn restoration

**(a) Marginal distortion [S, JC69].** The joint marginals deviate from q:

    pi_X(i) ~ q(i) * sum_j q(j) e^{-E(i,j)},   which for JC69 is   pi_X ~ ( 3+a, 3+c, 2+c+w, 2+a+w )  [A,C,G,T].

Read-off: GC-minus-AT content ~ 2(c - a), driven by **E_at - E_cg** (the WC strength
DIFFERENCE); G+T enriched over A+C ~ 2(w - 1), driven by the **wobble**. At E_gt = 0 this
is pi_X(A)=pi_X(T)=3+a and pi_X(C)=pi_X(G)=3+c, i.e. **Chargaff parity (u_A=u_T, u_C=u_G)
holds exactly** -- the marginal footprint of the surviving sigma symmetry. F81 has the
same form with q = phi and G(i) = sum_j phi(j) M(i,j); sigma-parity then also needs phi to
respect parity.

**(b) Sinkhorn / matrix scaling [existence S; factors N-exact].** Seek symmetric positive
d with pi'(i,j) ~ q(i) q(j) d_i d_j M(i,j) having marginals exactly q. The fixed point is

    d_i * ( sum_j q(j) d_j M(i,j) ) = const   for all i,

i.e. the symmetric DAD scaling of the positive symmetric matrix K = diag(q) M diag(q) to
row sums q. Because every M(i,j) = e^{-E} > 0, K is strictly positive and **Sinkhorn's
theorem gives a unique positive symmetric d up to overall scale**. The residual symmetry
collapses the fixed point to 1-2 unknowns and yields closed forms (JC69, up to scale):

    regime (ii)  a=c, E_gt=0 :  d = (1,1,1,1)         -- marginals already uniform, NO correction
    regime (iii) E_gt=0       :  d_A = d_T ~ (1+a)^{-1/2},   d_C = d_G ~ (1+c)^{-1/2}
    only wobble  a=c=1        :  d_A = d_C = 1,               d_G = d_T ~ sqrt( 2/(1+w) )

Each verified to 1e-16 against the iterated Sinkhorn map. General regime (iv) has trivial
symmetry, so no forced equalities; the unique numeric scaling (e.g. d ~
(1,0.873,0.775,0.902) at a2c3w1.5) restores uniform marginals to 1e-17. The corrected
energies are  E'(i,j) = E(i,j) - log d_i - log d_j  (e.g. regime iii:  E'(A,T) = E_at +
log(1 + e^{-E_at}),  with a compensating +½log(1+a)+½log(1+c) leaking onto the previously
neutral off-bond pairs).

**(c) Converse / Chargaff link.** The distortion in (a) is exactly how much an uncorrected
Potts warps a uniform proposal's marginals; E_gt = 0 preserves Chargaff parity. Sinkhorn
is the **soft/interior version** of the hard-support WC-mass LP in
`analysis/lumpable_product_trap.md` sec.7: that LP caps the WC-paired mass under fixed
skewed composition, this scaling instead adjusts a full-support joint to hit target
marginals exactly.

---

## Q2. Best single-component chain W_eff and the emergent HKY85/GTR question

The marginal of the coupled pair chain is not Markov (the product trap). The best
single-component **mean-field / partner-marginalised** generator averages the pair rate
over the partner's conditional law:

    W_eff(i -> i') = sum_j P(y=j | x=i) * Q_{(i,j)->(i',j)}
                   = q(i') * B(i,i') / B(i,i),     B(i,i') := sum_j q(j) sqrt( M(i,j) M(i',j) ).

B is a q-weighted **Gram matrix** of the sqrt-bond vectors v_i(j) = sqrt(M(i,j)), hence
symmetric PSD, and B(i,i) = sum_j q(j) M(i,j) = G(i).

**W_eff is exactly reversible GTR [S].** Its flux F(i,i') = pi_X(i) W_eff(i->i') =
q(i) q(i') B(i,i')/Z is symmetric, so W_eff is reversible w.r.t.

    pi_X(i) ~ q(i) B(i,i),     with exchangeability   S_eff(i,i') = Z * B(i,i') / ( B(i,i) B(i',i') ).

**It is NOT HKY85.** The emergent exchangeability numerators (JC69) expose the structure --
the coupling rides in through B(i,i'):

    ts  A-G : sqrt(a) sqrt(w) + sqrt(c) + 2        (bridge sqrt(a w) via shared wobble partner T)
    ts  C-T : sqrt(a) + sqrt(c) sqrt(w) + 2        (bridge sqrt(c w) via shared wobble partner G)
    tv  A-C : sqrt(a) + sqrt(c) + 2                (pure background)
    tv  A-T : 2 sqrt(a) + sqrt(w) + 1              (weak WC bond)
    tv  C-G : 2 sqrt(c) + sqrt(w) + 1              (strong WC bond)
    tv  G-T : sqrt(a) + sqrt(c) + 2 sqrt(w)        (wobble bond)

The two TRANSITION exchangeabilities are lifted off the background ONLY through the wobble
(terms sqrt(a w), sqrt(c w)); at E_gt = 0 they collapse to the background value and the ts
channel carries no signal -- consistent with kappa_flux == 1 there. E_at, E_cg act on
composition and on the two WC-bond TRANSVERSION exchangeabilities, not on the ts/tv split.
So the Potts does not produce an HKY85 kappa; its natural emergent structure is
**bond-modulation of the three transversion edges plus a wobble-bridged ts term**.

**Symmetry dictates the equalities.** The diagonal symmetry group's orbits on the 6 edges
give the generic S_eff equality pattern [N]:

    (i)  S4        : one orbit  -> W_eff = JC69.
    (ii) D4        : orbits {A-G,C-T,A-C,G-T} and {A-T,C-G}; at a=c these two VALUES also
                     coincide, so W_eff = JC69 STILL (uniform composition) -- the coupling
                     is invisible to a single site at this symmetric point.
    (iii) Klein-4  : orbits {A-G,C-T,A-C,G-T}, {A-T}, {C-G}  -> a 3-exchangeability GTR
                     (the two WC bonds special, the other four equal); parity composition;
                     kappa_flux = 1.
    (iv) trivial   : six distinct exchangeabilities -> generic GTR; parity broken (G,T
                     enriched); kappa_flux != 1.

**How much of GTR is explained.** GTR has 8 free dof (5 exchangeability ratios after a rate
gauge + 3 composition). The map (E_at, E_cg, E_gt) -> (S_eff ratios, pi_X) has Jacobian
rank exactly **3** at every sample point [N] -- an immersion of a 3-dimensional slice, i.e.
**3 of GTR's 8 dimensions**, and that slice does not lie inside the HKY85 (ts/tv)
sub-family. Weak/strong/wobble interpretation: E_at - E_cg controls GC content and the two
WC-bond transversion rates; E_gt controls the G/T composition enrichment, the wobble-bond
rate, and -- uniquely -- the transition channel and hence any net ts/tv bias.

---

## Summary

- **Q0.** Symmetry group order 1152 -> 48 -> 8 -> 2 as
  E=0 -> WC-sym -> E_at!=E_cg -> wobble; WC-complementation sigma=(AT)(CG) survives iff
  E_gt=0. Degeneracy 0,-1(x6),-2(x9) lifts 6->3+3, 9->1+3+5. **ts/tv emergence:**
  kappa_flux - 1 = (2-alpha-gamma)(1-omega) / [2(1+alpha+gamma+omega)] -- the wobble E_gt is
  the ts/tv switch (ratio == 1 when E_gt=0), the WC bonds set amplitude/sign.
- **Q1.** Distortion pi_X ~ (3+a, 3+c, 2+c+w, 2+a+w) [JC69]; GC-AT ~ 2(c-a), G+T ~ 2(w-1);
  E_gt=0 => Chargaff parity. Unique symmetric Sinkhorn d: regime (iii) d_{A,T} ~ (1+a)^{-1/2},
  d_{C,G} ~ (1+c)^{-1/2}; only-wobble d_{G,T} ~ sqrt(2/(1+w)); E' = E - log d_i - log d_j.
- **Q2.** W_eff = reversible GTR with pi_X ~ q(i)B(i,i), S_eff = Z B(i,i')/(B(i,i)B(i',i')),
  B(i,i') = sum_j q(j) sqrt(M(i,j)M(i',j)). kappa_eff drivers: E_gt drives the ts channel,
  E_at-E_cg drives GC composition, E_gt drives G/T enrichment. Image = 3-dim slice of GTR's
  8 dof, NOT HKY85. Verdict: the Potts explains 3 of 8 GTR directions and does not reduce to
  a ts/tv factor.

## Reproduce

    python3 experiments/wc_potts_joint_model.py

Prints the reversibility proofs, the symmetry/spectrum table, the kappa_flux factorization
proof, the Sinkhorn closed-form checks, and the W_eff / GTR-image analysis, each verified
against a direct numerical build to <1e-9.
