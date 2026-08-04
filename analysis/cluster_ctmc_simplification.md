# The cap-2 cluster CTMC simplifies: full Felsenstein + Holmes-Rubin without 800^3

Status: CONSOLIDATION + numerical-check plan (2026-07). The maths is already
written out across the repo; this file only pulls the pieces together and states
the check. Sources:
  * generator + the renewal/reset structure -- math-paper/appendix-tkfdp.tex,
    "Dynamic latent-field variant" and the "Carry vs. resample under a modulator
    jump" paragraph (eq:arch-mapping, eq:arch-Q-field);
  * the NJ/JMP factored branch operator -- src/tkfdp/coupling/dynfield/phylo_elbo/
    exact_cap2.py (branch_pair) and appendix-tkfdp.tex;
  * single-residue Holmes-Rubin (bridge expectations) -- appendix-tkfdp.tex
    rem:bridge-expectations (\citep{HolmesRubin2002}), src/tkfdp/eta_site.py
    (hr_per_cherry), ~/tkf-mixdom/tkf/body-tkf91.tex;
  * the field/residue factorization -- math-paper/draft-product-trees-elbo.tex.

No corner-cutting: for every cluster we keep (top-12 classes per singleton,
top-12^2 class-pairs per pair) we do EXACT Felsenstein and EXACT Holmes-Rubin.
The claim to CHECK is that the exactness costs O(A^3), not O((L*A^m)^3).

Notation: field theta in {0..L-1} (L=2), residues x_1..x_m in A=20 letters.
Cluster state z = (theta, x_1, .., x_m); |z| = L*A^m (= 800 for a pair, 40 for a
singleton). Class c fixes the archetype a_n(theta) each residue takes at each
field state; Q^k = GTR generator of archetype k (F81 form, stationary pi^k);
P^k(t) = exp(Q^k t); rho = field stationary; rho_chain = field-chain rate.

## 1. The generator factorizes (Kronecker-sum + rank-1 renewal)

The cluster generator splits into a field-diagonal "drift" part and a field-jump
"renewal" part:

    Q_clus = D  +  Jmp

D (no field jump; theta held fixed): the residues substitute INDEPENDENTLY under
their current archetype, so on the theta-block D is a Kronecker SUM

    D_theta = sum_{n=1}^m  I ⊗..⊗ Q^{a_n(theta)} ⊗..⊗ I      (nth factor)

i.e. only one residue changes per event. D is block-diagonal in theta.

Jmp (field renewal, F81-on-DP): at the chain's jump the field resamples theta' ~
rho AND every residue resamples from the NEW archetype stationary. So

    Jmp[(theta,x),(theta',x')] = rho_chain * rho[theta'] * prod_n pi^{a_n(theta')}[x_n']   (theta' != theta)

which is RANK ONE in the residues for each (theta,theta') pair: an outer product
of stationaries. (Diagonal fixed by zero row sums.)

Two structural facts do all the work: D is a per-theta Kronecker sum (residues
independent given the field), and Jmp is a rank-1 reset (memoryless renewal).

## 2. Felsenstein: the branch operator is O(L*A^3), not O((L A^m)^3)

Because Jmp is a memoryless reset to prod_n pi, condition a branch of length tau
on whether the field jumps in [0,tau]:

- NO jump (prob-weighted by the survival beta_theta(tau) = exp(-rho_chain(1-rho[theta])tau)):
  theta is constant and D_theta generates INDEPENDENT residue evolution, so the
  transition is the product  beta_theta(tau) * ⊗_n P^{a_n(theta)}(tau).
- >=1 jump: the last renewal resets to prod_n pi^{a_n(theta')} and the residues
  evolve from there; because the reset is rank-1, the whole >=1-jump contribution
  is rank-1 in the residues -- a field-transition kernel J = P_theta - diag(beta)
  applied to the SCALAR renewal marginal m(theta') = prod_n (pi^{a_n(theta')} . M).

For a message M (cluster CLV) this is exactly the cap-2 branch operator already
implemented in `exact_cap2.branch_pair`:

    NJ[theta]  = beta[theta] * P^{a_1(theta)}(tau) M[theta] P^{a_2(theta)}(tau)^T   (pair)
    m[theta']  = pi^{a_1(theta')} . M[theta'] . pi^{a_2(theta')}
    M'[theta]  = NJ[theta] + (J @ m)[theta]                              (broadcast in x,y)

Cost: the NJ term is two A x A mat-muls per field state -> O(L * A^3) per branch,
versus O((L A^m)^3) for a dense exp(Q_clus tau) (5e8 for a pair). This is the
Felsenstein simplification; it is exact and already validated (cherry / tree LL
match the dense compound generator to 1e-12 in exact_cap2's tests).

## 3. Holmes-Rubin: the cluster sufficient statistics factor the same way

We need, for the M-steps: per-archetype expected substitution counts N^k[a,b] and
dwell T^k[a] (for pi_archetype), and the expected number of field jumps (for the
field-rate). Naively HR on Q_clus is an 800-state eigendecomposition + Van Loan
integral per branch, O(800^3). It factors.

KEY. Conditioned on the full field-and-jump trajectory over a branch, the m
residue processes are INDEPENDENT single-residue chains, each a GTR run under the
archetype the field currently selects, cut (reset to stationary) at each field
jump. Therefore, on a branch with the field summary (theta_u, theta_v, Delta):

- Delta = 0 (no field jump on the branch): field is one state theta, residue n
  evolves under GTR^{a_n(theta)} for the whole tau. Its endpoint-conditioned HR
  is the SINGLE-RESIDUE 20-state Holmes-Rubin
      (N, T) = hr_single(a_n(theta); x_u^n, x_v^n, tau),
  the ordinary `eta_site.hr_per_cherry` on Q^{a_n(theta)}. The other residues and
  the field contribute nothing to archetype-a_n(theta)'s counts on this branch.
- Delta = 1 (>=1 field jump): the residue is reset to the new archetype
  stationary at EACH field jump. For M>=1 jumps the branch decomposes into a first
  segment (start X_n observed, end resampled), a (possibly empty) sequence of
  middle segments (resampled -> resampled, i.i.d. from stationary), and a last
  segment (resampled -> end x_v^n observed), integrated over the jump count M and
  the jump times -- exactly the first/middle/last-segment attribution of
  appendix-tkfdp.tex ("Case M >= 1", eq:arch-tau1-exact). NOT a single last-jump
  split (that is only exact for M=1). The numerically-checked shortcut that gives
  the same aggregate is: (40-state (theta,x) endpoint HR) minus the Delta=0 GTR
  part -- reset at every jump, integrated over the whole field path. [Verified: at
  L=2 the single-split and the exact multi-jump form give identical aggregate
  N^k/T^k to 1e-13, but the exact statement is the multi-jump one; the appendix
  already states it correctly, this doc's earlier single-split phrasing did not.]

The residues never enter each other's HR, and never enter the field's -- exactly
because D is a Kronecker sum and Jmp is rank-1. So the cluster HR assembles as

    N^k = sum_{branches} sum_{residues n : reachable arch = k}
             sum_{theta,Delta} q_field(theta,Delta | branch) *
                 E_{residue-endpoints}[ hr_single(k; ...) restricted to k-segments ]

where
  * q_field(theta,Delta | branch) is the endpoint-conditioned field posterior on
    the branch -- the L-state (theta,Delta) tree MRF marginals from the SAME
    factored Felsenstein of section 2 (up-down over the 2-state field, residue
    evidence folded in), computed by `field_bp`;
  * the residue-endpoint posterior on the branch is the single-residue up/down
    message (factored, 20-state);
  * hr_single is the 20-state Holmes-Rubin (`eta_site.hr_per_cherry`), one
    eigendecomposition per archetype (K_a of them), reused across all columns.

Cost: O(K_a * A^3) for the residue HR eigendecompositions (shared corpus-wide),
plus O(L^2) per branch for the field posterior and O(A^2) per residue per branch
for the up/down messages -- NOT O(800^3), and NOT Monte-Carlo. This is the exact
E-step the supervised trainer's archetype update currently ONLY approximates
(rate-0 static responsibilities + a single sampled history); doing it this way
removes both approximations and restores EM monotonicity.

## 4. What to check numerically (section-by-section, against the dense generator)

Build the dense Q_clus (`exact_cap2._compound_generator_pair`) for a pair and:

1. exp(Q_clus tau) == the section-2 factored branch operator applied to each basis
   message  (already covered by exact_cap2's cherry/tree tests; re-assert).
2. Endpoint-conditioned HR on Q_clus (dense eigendecomposition + Van Loan for the
   expected off-diagonal counts and dwell, up-down over a cherry) ==
   the section-3 factored assembly: field posterior (field_bp) x per-residue
   hr_single. Check, per archetype: N^k[a,b] and T^k[a], and the expected field-
   jump count, to ~1e-8 on a cherry and a 3-leaf tree, at rho_chain=0 (no jumps;
   pure per-residue HR) and rho_chain>0 (jumps active). rho_chain=0 isolates the
   Delta=0 path; a single internal branch with a forced field flip isolates
   Delta=1.

Once checked, the trainer's `accumulate_hr_per_archetype` is replaced by this
exact factored HR (drop `sample_history` / the rate-0 responsibility), and the
archetype M-step becomes a genuine EM step.

## Numerical check (results)

Status: CHECKED and CONFIRMED (2026-07). Script:
`analysis/scripts/check_cluster_hr.py` (pure CPU/numpy; run it to reproduce).
Archetypes are real Le-Gascuel C10 profiles (an acid-rich and a base-rich
component, plus LG08 background) so `a1 != a2` genuinely uses charge-crossing
profiles; S is LG08. Trees: a cherry (root + 2 leaves) and a 3-leaf tree, with
charge-crossing leaf residues (D/E vs K/R). The factored assembly matches the
dense 800-state HR to eigendecomposition precision in EVERY regime.

Ground truth. The dense generator is `_compound_generator_pair` (800 states).
It is reversible w.r.t. p(t,x,y) = rho[t] pi^{a1(t)}[x] pi^{a2(t)}[y] (detailed
balance holds for residue moves because each GTR is reversible, and for the
field-jump renewal because the target factorises as rho x pi x pi), so a real
symmetric eigendecomposition + the Van Loan/Hobolth-Jensen bridge integral gives
the exact endpoint-conditioned expected transition counts and dwell over the
tree (up-down pass). This dense HR was independently validated two ways:
  * its tree log-likelihood matches `exact_cap2.exact_pair_ll_tree` to 1e-15;
  * its single-branch endpoint-conditioned expectations (800-state) match an
    endpoint-conditioned uniformisation Monte-Carlo bridge to MC noise
    (field jumps 1.036 vs 1.036, residue subs 1.734 vs 1.733, dwell exact),
    which specifically exercises field-jump attribution and residue
    substitutions under the reset/renewal.

Factored assembly. Field jumps are the residue-independent 2-state field HR,
config-weighted. The per-archetype residue statistics are assembled by
enumerating the field configuration (theta at every node, jump-indicator Delta
at every edge; residues are conditionally independent given (theta-nodes,
Delta-edges) because at a field jump BOTH residues reset to the new archetype
stationary, decoupling their endpoints). Per config and per branch:
  * Delta = 0 : 20-state GTR endpoint-conditioned HR under the frozen archetype
    (`single_branch_hr` on Q^{a_n(theta)});
  * Delta = 1 : (40-state (theta,x) endpoint HR) minus (the Delta=0 GTR part),
    divided by the field jump weight J -- the reset/renewal contribution, which
    correctly accounts for intermediate segments (the "last-jump split" phrasing
    of section 3 is only exact for a single jump; the 40-state (theta,x) chain
    integrates the full multi-jump field path exactly).
No 800-state object is ever formed; the cost is O(K_a A^3) for the residue
eigendecompositions (20-state) plus the 40-state (theta,x) per-residue chains
and the 2-state field peel.

Results (max relative discrepancy over N^k[a,b], T^k[a]; absolute for the field-
jump count), cherry and 3-leaf:

  Regime                                 reldiff N^k    reldiff T^k    |dJumps|
  rho_chain = 0, a1!=a2 (no jumps)       ~1e-10..3e-12  ~1e-10..4e-12  0        (jumps=0)
  rho_chain = 0, a1==a2                  ~1e-11..2e-12  ~1e-11..3e-12  0        (jumps=0)
  rho_chain = 0.4, a1!=a2 (jumps)        ~2e-12..1e-12  ~2e-12..9e-13  <2e-14
  rho_chain = 1.1, a1!=a2 (jumps)        ~2e-13..7e-14  ~2e-13..6e-14  <4e-14
  rho_chain = 0.7, a1==a2 (jumps)        ~3e-12..2e-12  ~6e-12..4e-12  <1e-14

The discrepancies are at eigendecomposition round-off, well under the 1e-8
target, in all regimes -- including active field jumps (Delta=1 reset/renewal)
and charge-crossing archetypes. The Felsenstein re-assert (section 2) matches to
1e-9 as expected.

Conclusion. The exact cap-2 cluster Holmes-Rubin E-step FACTORS as claimed: the
exact endpoint-conditioned N^k[a,b], T^k[a] and field-jump count of the dense
800-state CTMC equal the factored O(K_a A^3) assembly. The supervised trainer's
Monte-Carlo / rate-0 archetype step can therefore be replaced by this exact
factored HR, restoring a genuine EM step.

One derivation note for the appendix: section 3's Delta=1 description as a single
"pre-jump dwell under a_n(theta_u) + post-jump HR under a_n(theta_v)" split is
exact only for exactly one jump. For L=2 with multiple jumps the intermediate
(stationary) segments also contribute residue substitutions; the correct exact
statement is the (40-state (theta,x) endpoint HR) - (Delta=0 GTR part) used here,
which integrates the full field path. The aggregate N^k/T^k are unaffected by the
phrasing (both give the same numbers), but the appendix wording should say
"reset at each field jump, integrated over the field path" rather than implying a
single last-jump split.
