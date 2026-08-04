# Half-lumpable = one-sided (cpt1) lumpable pair chain — derivation

The **Half-lumpable** model is the most permissive reversible 400-state pair chain that
is strongly lumpable to cpt1 (its cpt1 marginal is an exact Markov/GTR process),
without imposing anything on cpt2. It is the corrected "field+renewal": same
foregrounding of cpt1's marginal, but the cpt1-jump no longer *forgets* cpt2.

## 1. Lumpability-to-cpt1 (Kemeny–Snell)

Aggregating states by cpt1 (block B_i = {(i,·)}) gives a Markov process iff the
total rate from any state into each block is block-constant, i.e. **independent of
the spectator cpt2 state**:

    sum_l Q[(i,j),(k,l)] = g(i,k)      for all i != k, all j.        (L1)

`g(i,k)` is then *exactly cpt1's marginal generator*. (L1) is the ONLY constraint.
Nothing constrains cpt2's own transitions or how cpt2 moves during a cpt1-jump.

## 2. cpt1-centered construction (marginal specified, extensible)

- **cpt1 marginal**: a GTR `g(i,k) = S1[i,k] pi1[k]` (S1 symmetric). Specified /
  fit as a GTR; by (L1) it *is* the marginal, foregrounded so extra cpt1-dependent
  components can be tacked on later without disturbing it.
- **stationary**: pi_joint(i,j) = pi1(i) pi(j|i)  (general, asymmetric).
- **cpt2 own transitions** (i fixed): free reversible flux per field,
  Q[(i,j),(i,l)] = E[i](j,l) pi(l|i), E[i] symmetric  (as in Renewal's full variant).
- **cpt1-jump** (i->k): Q[(i,j),(k,l)] = g(i,k) K_{ik}(j -> l), where K_{ik} is a
  row-stochastic kernel on cpt2 (sum_l K_{ik}(j,l) = 1). (L1) holds automatically.

## 3. The cpt1-jump kernel is a COUPLING (the key result)

Reversibility w.r.t. pi_joint + a reversible marginal g (pi1(i)g(i,k)=pi1(k)g(k,i))
forces, for the dual transition,

    pi(j|i) K_{ik}(j,l) = pi(l|k) K_{ki}(l,j).                       (R1)

Define the kernel flux Psi_{ik}(j,l) = pi(j|i) K_{ik}(j,l). Then (R1) is
Psi_{ik}(j,l) = Psi_{ki}(l,j), and the stochasticity of K gives the marginals

    sum_l Psi_{ik}(j,l) = pi(j|i),     sum_j Psi_{ik}(j,l) = pi(l|k).

**So Psi_{ik} is a coupling (transport plan) between the conditional stationaries
pi(.|i) and pi(.|k).** The choice of coupling is exactly the modelling freedom:

- **Renewal** (old "Half-lumpable") = the INDEPENDENT coupling
  Psi_{ik}(j,l) = pi(j|i) pi(l|k): cpt2's post-jump state is drawn fresh from
  pi(.|k), forgetting j. Maximal-entropy / maximal forgetting — this is why cpt2
  over-evolves and Renewal is worst.
- **Half-lumpable** = ANY reversible coupling. Near-diagonal couplings let cpt2 largely
  carry over (preserve its pre-jump identity), which is what the data wants.
- The "cpt2 fully stays" coupling (identity) requires pi(.|i)=pi(.|k) for all i,k,
  i.e. factorized pi — that is the independent-sites / single-transition limit. So a
  *coupled* reversible lumpable-to-cpt1 chain MUST move cpt2 somewhat on cpt1-jumps;
  the freedom is only in HOW (Renewal takes the destructive extreme).

## 4. Why it should beat 2-sided Lumpable

Both are the general reversible chain under a lumpability constraint. Our existing
**Lumpable** imposes (L1) on COMPONENT-EXCHANGEABLE fluxes, so the one row-marginal
constraint set also yields lumpability-to-cpt2 (two-sided). Half-lumpable drops
exchangeability, keeping (L1) for cpt1 only -> strictly fewer constraints ->
strictly larger feasible set -> fits >= Lumpable. It is not comparable to
Synchronized (different symmetry class), so cross-class DOF comparisons are avoided.

## 5. Fitting — reuses the existing reversible-flux ALM

The reversible flux MLE subject to (L1) is *exactly* `mstep_lumpable`
(em_lumpable_reversible_ctmc.tex): maximise the expected complete-data LL over the
reversible flux subject to B*phi = D(pi)g. Two changes make it one-sided:

1. **Orbits**: replace the Klein-4 (component-swap x time-reversal) orbit map with a
   TIME-REVERSAL-ONLY map (phi_{ab,cd}=phi_{cd,ab} but NOT =phi_{ba,dc}). This is the
   whole "remove exchangeability" step; `build_lump_rows` is already cpt1-only, so
   with these orbits it constrains cpt1 alone.
2. **Stationary**: use the asymmetric empirical pi_joint (do NOT symmetrise), and
   component-swap-symmetrise the COUNTS (arbitrary field label) as for Renewal.
3. Optional: pin g to a GTR (S1, pi1) for the clean cpt1-centered / extensible form;
   or leave g as a free reversible marginal (a 20-state GTR either way).

## 6. Degrees of freedom — two distinct quantities

**(a) Model dimension** = size of the parameterization we write down: non-exchangeable
reversible flux C(400,2)=79,800 minus (L1)'s 7,410 net constraints + 399 pi = 72,789.
This needs no fit; it is the honest "model complexity" number for the table (with the
caveat that it is a NON-exchangeable model, so its ceiling is 80,199, not
Synchronized's 40,299 -- do not compare across symmetry classes).

**(b) Effective / identified DOF** = parameter directions the DATA pins down =
rank / smooth `edf(lambda)=tr(F(F+lambda I)^-1)` of the observed information at the MLE.

**Empirical finding (experiments/effective_dof.py): there is NO significant
symmetrisation-induced under-identification.** I had predicted the component-swap
ANTISYMMETRIC flux subspace would be flat (tied by the count symmetrisation) so that
effective DOF collapses toward the exchangeable count. A finite-difference curvature
probe at the fitted Synchronized MLE REFUTES this: antisymmetric-flux curvature
(median 3.70e15) equals symmetric-flux curvature (3.72e15), ratio 0.995 -- both
subspaces are equally curved. The error was conflating "antisymmetric FLUX direction"
with "unidentified direction": symmetrising the counts discards n - n_swap, but the
aggregate gradient vanishing along antisymmetric directions is just the MLE condition,
not lost information -- each observation still has a nonzero score there (the flux->P
map is nonlinear). So with ~3e8 counts the parameters are genuinely identified and
**effective DOF ~ nominal DOF**; no separate effective-DOF column is warranted.

Table policy: report the model dimension (a) only. Clean reference to isolate the
lumpability-to-cpt1 cost: also fit a NON-exchangeable Synchronized (no lumpability),
so Half-lumpable = that minus (L1) and the comparison stays within one symmetry class.
