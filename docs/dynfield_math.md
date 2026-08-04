# Dynamic latent-field variant: math derivations

Companion to `dynfield_design.md`. Tight derivations of the four pieces
the implementation will rely on. Numerical cross-checks live in
`tests/dynfield/test_math_precompute.py`.

## Notation

- Field alphabet (after DP truncation): `Θ = {1, …, L}`.
- Stick-breaking weights: `ρ_θ`, `Σ_θ ρ_θ = 1`. DP concentration:
  `α_field`.
- Amino acid alphabet: `𝒜`, `|𝒜| = k = 20`. LG08 exchangeabilities: `S`.
- Site classes: `c ∈ {1, …, K_c}`. Per-(class, field) stationary:
  `pi^(c, θ) ∈ Δ^{k−1}`. Per-(class, field) generator: `Q^(c, θ) = `
  GTR(`S`, `pi^(c, θ)`) — the same F81-form construction the Potts
  variant already uses, just now indexed by `θ` as well as `c`.
- Cluster: ordered pair `(i, j)` of coupled sites with classes
  `(c_i, c_j)`.
- Field trajectory on a cherry: `θ_P` at the parent, `θ_X` and `θ_Y` at
  the two leaves, branch length `t` on each leaf edge.

## 1. F81-on-DP field selector

The field selector is a CTMC on `Θ` with generator
```
Q_F(θ → θ') = ρ_θ'           for θ ≠ θ'
Q_F(θ → θ) = − (1 − ρ_θ)
```
i.e. the F81 form *with the field stationary*. Detailed balance:
`ρ_θ Q_F(θ → θ') = ρ_θ ρ_θ' = ρ_θ' Q_F(θ' → θ)` — symmetric in (θ, θ').

Spectrum: `Q_F = 1 ρ^T − I` (the rank-1 update minus the identity) has
eigenvalues `{0, −1, −1, …}`, since `Q_F · 1 = (ρ^T · 1) · 1 − 1 = 0`
and any `v` with `ρ^T v = 0` satisfies `Q_F · v = −v`. The transition
kernel therefore takes the standard F81 closed form with uniform decay:
```
P_F(θ → θ'; t) = ρ_θ' + (δ_{θθ'} − ρ_θ') · exp(−t)
```
(The implementation may choose to renormalise the overall rate to give
mean substitution rate 1 at stationary, dividing `Q_F` by
`1 − Σ_θ ρ_θ²`; the closed form's decay constant then becomes that
quantity. We leave the value as 1 for derivation simplicity and pick a
consistent rate convention at implementation time.)

**Stationary**: `ρ` by construction. **Reversibility**: by inspection.

## 2. Strong lumping onto occupied atoms ∪ tail

In SVI we only ever observe the field through its effect on finitely
many coupled clusters. Let `O ⊂ Θ` be the set of *occupied* atoms at
which any cluster's field assignment has been observed under the
variational posterior; let `T = Θ ∖ O` be the *tail*. Define the
lumped state space `O ∪ {*}` where `*` collapses all tail atoms.

**Strong lumpability claim**: under F81, the lumped chain on `O ∪ {*}`
is itself an F81 chain with stationary
```
ρ_o      for o ∈ O
ρ_*  =   Σ_{θ ∈ T} ρ_θ        for the tail block
```
and transition kernel `P_F(o → o'; t) = ρ_o' + (δ_{oo'} − ρ_o') exp(−t)`,
etc. The lumping is **exact** — not "approximate up to leading-order
tail error" as an earlier draft of this doc claimed.

**Proof**. Strong lumpability requires that for every pair of lumped
states `(A, B)` and every original state `x ∈ A`, the aggregated
*rate* `Σ_{y ∈ B} Q[x, y]` is independent of the choice of `x` within
`A`. The F81 generator has `Q[x, y] = ρ_y` for `x ≠ y` and
`Q[x, x] = −(1 − ρ_x)`. For `A ≠ B`:
`Σ_{y ∈ B} Q[x, y] = Σ_{y ∈ B} ρ_y` since `x ∉ B`, so `x ≠ y` everywhere
in the sum — and the result has no `x` dependence. For `A = B`:
`Σ_{y ∈ A} Q[x, y] = Q[x, x] + Σ_{y ∈ A, y ≠ x} ρ_y =
−(1 − ρ_x) + (Σ_{y ∈ A} ρ_y) − ρ_x = −1 + Σ_{y ∈ A} ρ_y`,
again independent of `x`. ∎

The closed-form lumped kernel uses the same uniform `exp(−t)` decay as
the un-lumped one (§1) — the field spectrum `{0, −1, …, −1}` is
inherited by every projection onto an F81 sub-chain.

**Implementation consequence**. Maintain only `|O| + 1` field-state
probabilities at each tree node. New cluster assignments draw from the
tail block with probability `ρ_*`; when the tail draws a fresh atom we
materialise it (rich-get-richer DP semantics) and the lumped state
space grows by one. The conditional likelihood vectors per cluster
stay constant-size in `|O| + 1`, not in the full untruncated `L_max`,
and crucially the lumping introduces *no approximation* in the
F81-on-DP transitions.

## 3. Cap-2 cluster joint emission

Cherry: parent `P`, two leaves `X` and `Y`, branch length `t/2` on each
leaf edge (so total cherry diameter is `t`). Cluster of two coupled
sites `i` and `j` with classes `(c_i, c_j)`. Observations: `(X_i, X_j)`
at leaf `X` and `(Y_i, Y_j)` at leaf `Y`. Per the supplement's
"Carry vs. resample" paragraph, the site between field jumps evolves
under `Q^(c, θ)` (it does *not* sit at stationary mid-segment); only
at a field jump does the residue resample from `pi^(c, θ_new)`. Since
`pi^(c, θ)` is the stationary of `Q^(c, θ)`, post-jump dynamics keeps
the site at `pi^(c, θ)` marginally — but the **no-jump** case retains
the per-(c, θ_P) GTR dynamics from the parent residue, which is what
correlates the two leaves of a cherry through their (shared,
unobserved) parent.

The two coupled sites in the cluster share the **same** field
trajectory on each leaf edge; if a jump happens on the `P → X` edge,
both sites' residues are resampled at that jump. So the two sites'
jump status is coupled — both jump or both don't.

**Interp 2 (no-self-jump CTMC).** The textbook F81-on-DP CTMC counts
only real `θ → θ'` transitions (`θ ≠ θ'`) as jumps; self-jumps are not
events. Under instant re-equilibration, only real jumps trigger residue
resampling. The no-jump probability over a half-edge is therefore
**state-dependent**:

```
β(θ_P) := exp(−ρ_chain · (1 − ρ_{θ_P}) · t/2)
```

(where `ρ_chain` is the F81-on-DP rate multiplier; `ρ_chain = 1` by
default, with the time `t` absorbing the rate). The earlier draft of
this doc used the simplified `p_nj = exp(−ρ_chain · t/2)` *uniformly*
across `θ_P` (Interp 1, the "carrier-vs-stationary mixture" model); the
two interpretations have the same field marginal `P[θ → θ'; t]` but
different (residue, field) joint dynamics. Interp 2 keeps residues
parent-correlated longer at frequent fields, which is the physically
sensible regime (fewer forced resets from the next level up in the
hierarchy).

**Notation:**

- `α := exp(−ρ_chain · t/2)` — the field-marginal decay rate.
- `β(θ_P) := exp(−ρ_chain · (1 − ρ_{θ_P}) · t/2)` — the state-dependent
  no-jump probability over a half-edge starting at field `θ_P`.
- `Σ^(c, θ)(a, b; t) := Σ_p pi^(c, θ)(p) · P^(c, θ)(p → a; t/2) ·
  P^(c, θ)(p → b; t/2)` — the per-(c, θ) GTR cherry joint emission.
- `J^(c_i, c_j)(a, b) := Σ_{θ} ρ_θ · pi^(c_i, θ)(a) · pi^(c_j, θ)(b)`
  — the field-marginal joint stationary at the cluster (same as in §4).
- `J'^(c_i, c_j, θ_P)(a, b)` — the **per-θ_P post-jump joint emission**.
  When the field jumps at least once on a half-edge starting from
  `θ_P`, the end-state `θ_end` has conditional distribution
  ```
  P(θ_end | ≥1 jump, θ_P, t/2)
    = ρ_{θ_end} · (1 − α) / (1 − β(θ_P))                    if θ_end ≠ θ_P
      + δ_{θ_end, θ_P} · (α − β(θ_P)) / (1 − β(θ_P))
  ```
  and the post-jump leaf emits from `pi^(c, θ_end)`. Marginalising over
  `θ_end` gives the convex-combination decomposition:
  ```
  J'^(c_i, c_j, θ_P)(a, b)
    = w_J(θ_P) · J^(c_i, c_j)(a, b)
    + w_pi(θ_P) · pi^(c_i, θ_P)(a) · pi^(c_j, θ_P)(b)
  ```
  with weights
  ```
  w_J(θ_P)  = (1 − α) / (1 − β(θ_P))     (extra mass on the J factor when α small)
  w_pi(θ_P) = (α − β(θ_P)) / (1 − β(θ_P)) (extra mass on θ_P retention)
  ```
  These sum to 1 by algebra. As `ρ_chain → 0` weights are 0/0; the
  yj contribution vanishes anyway (since `1 − β → 0`), so a clipped
  default `(w_J, w_pi) = (1, 0)` is safe.

**Exact cap-2 cherry joint, conditional on `θ_P`** (then averaged with
prior `ρ_{θ_P}`):

```
P((X_i, X_j), (Y_i, Y_j) | c_i, c_j, t, θ_P) =

  β(θ_P)²                                                 [(no-jump on X, no-jump on Y)]
    · Σ^(c_i, θ_P)(X_i, Y_i; t)
    · Σ^(c_j, θ_P)(X_j, Y_j; t)

  + β(θ_P) · (1 − β(θ_P))                                 [(no-jump on X, ≥1 jump on Y)]
    · pi^(c_i, θ_P)(X_i) · pi^(c_j, θ_P)(X_j)
    · J'^(c_i, c_j, θ_P)(Y_i, Y_j)

  + (1 − β(θ_P)) · β(θ_P)                                 [(≥1 jump on X, no-jump on Y)]
    · J'^(c_i, c_j, θ_P)(X_i, X_j)
    · pi^(c_i, θ_P)(Y_i) · pi^(c_j, θ_P)(Y_j)

  + (1 − β(θ_P))²                                         [(≥1 jump on X, ≥1 jump on Y)]
    · J'^(c_i, c_j, θ_P)(X_i, X_j)
    · J'^(c_i, c_j, θ_P)(Y_i, Y_j)
```

The (no-jump, no-jump) term carries the per-(c, θ_P) GTR cherry
dynamics for both sites; the other three carry the per-θ_P post-jump
joint `J'^(c_i, c_j, θ_P)` (which is the field-marginal joint `J` plus
a per-θ_P extra-retention term). Verified by Gillespie MC against the
closed form (test 4 in `tests/dynfield/test_math_precompute.py`).

**Cost**:

- Per (c, θ, t): one 20×20 eigendecomposition of `Q^(c, θ)`, then
  the per-leaf-half-edge transitions and the cherry-joint `Σ` are
  matrix-multiply chains at `O(A³)` per (c, θ, t). Cumulative:
  `O(K_c · L_max · A³ · n_t)` for `Σ`.
- Per (c1, c2, t): the 4-case sum is `O(L_max · A⁴)` per `(c1, c2)`
  ((nn, nn) carries the `θ_P` dependence; the others factorize after
  summing over the leaf θ). Cumulative:
  `O(K_c² · L_max · A⁴ · n_t)` for the cluster doublet emission tensor.
  This is the same order as the Potts variant's `O(K_c² · A⁴ · n_t)`
  (which has an extra constant from the `400³` per-(c1, c2)
  eigendecomposition); the dynamic-field cost is *competitive* with
  Potts but no longer the "strictly cheaper" win an earlier draft of
  this doc claimed.

**Cross-check vs the brute Kronecker form (L=1 limit)**. At `L_max=1`
the field is degenerate (only one atom, no jumps possible); only the
(nn, nn) case survives with weight 1. The cluster emission reduces to
the standard per-(c, θ=0) GTR independent-pair Kronecker product,
which is the Phase A test 3 sanity check.

## 4. M-tensor for the Infinite Pair HMM

The Infinite Pair HMM's edge boost is the multiplicative ratio of
coupled-pair emission to two independent singleton emissions:
```
M(a, b, c, d) =
    P_doublet(a, b, c, d) / [P_singlet(a, b) · P_singlet(c, d)]
```
For the dynamic-field variant on a cherry of branch length `t` per
leaf (so total diameter `t`), the singleton and doublet emissions
each carry the same 4-case structure as §3:

```
P_singlet(a, b) =
    Σ_c   π_c   Σ_{θ_P}   ρ_{θ_P}   · {
        β(θ_P)² · Σ^(c, θ_P)(a, b; t)
      + β(θ_P) · (1 − β(θ_P)) · pi^(c, θ_P)(a) · pi'^(c, θ_P)(b)
      + (1 − β(θ_P)) · β(θ_P) · pi'^(c, θ_P)(a) · pi^(c, θ_P)(b)
      + (1 − β(θ_P))² · pi'^(c, θ_P)(a) · pi'^(c, θ_P)(b)
    }

P_doublet(a, b, c, d) =
    Σ_{c1, c2}  π_c(c1) π_c(c2)  Σ_{θ_P}  ρ_{θ_P}  · {
        β(θ_P)² · Σ^(c1, θ_P)(a, b; t) · Σ^(c2, θ_P)(c, d; t)
      + β(θ_P) · (1 − β(θ_P))
          · pi^(c1, θ_P)(a) · pi^(c2, θ_P)(c)
          · J'^(c1, c2, θ_P)(b, d)
      + (1 − β(θ_P)) · β(θ_P)
          · J'^(c1, c2, θ_P)(a, c)
          · pi^(c1, θ_P)(b) · pi^(c2, θ_P)(d)
      + (1 − β(θ_P))²
          · J'^(c1, c2, θ_P)(a, c) · J'^(c1, c2, θ_P)(b, d)
    }
```
where `β(θ_P) = exp(−ρ_chain · (1 − ρ_{θ_P}) · t/2)`,
`pi'^(c, θ_P)(a) = w_J(θ_P) · pi^(c)_marg(a) + w_pi(θ_P) · pi^(c, θ_P)(a)`,
and `J'^(c1, c2, θ_P)(a, b) = w_J(θ_P) · J^(c1, c2)(a, b) +
w_pi(θ_P) · pi^(c1, θ_P)(a) · pi^(c2, θ_P)(b)` (post-jump per-θ_P
emissions; see §3 for `w_J`, `w_pi`).
(Indexing convention: `(a, c)` at the parent-side / leaf-X end,
`(b, d)` at the child-side / leaf-Y end; same as
`block_likelihoods.build_doublet_emission`.)

The 4-case structure is identical for singlet and doublet; this is
what keeps the M-tensor algebraically clean (and the (yj, yj) and
mixed terms in the numerator and denominator partially cancel,
leaving the multiplicative boost dominated by the (nn, nn)
coupled-cherry term at finite `t`).

**Indel-seam marginal & no Sinkhorn**. Marginal consistency of the
joint stationary is unchanged from §3: the **stationary** of the
coupled-pair cluster is `J^(c1, c2)(a, b)`, with row marginal
`Σ_b J^(c1, c2)(a, b) = Σ_θ ρ_θ · pi^(c1, θ)(a) · 1 = pi^(c1)_marg(a)`.
The dynamic-field variant therefore needs **no Sinkhorn correction**
at the indel seam: the lone-stationary used on a partial-presence
branch *is already* the marginal of the coupled-pair stationary
(Theorem `thm:revcond`). This was the structural motivation in the
reversibility note (§`sec:projectivity`); test 5 confirms it
numerically.

**Implementation consequence**. The `m_tensor` and `m_tensor_typed`
calls for the dynamic-field variant skip the Sinkhorn iteration
entirely. The `--pre-sinkhorn` CLI flag has no effect under this
variant; the preflight banner already covers the incoherent combo.

## What the verification script must show

`tests/dynfield/test_math_precompute.py` must assert, with random
fixtures (small k, small L):

1. F81-on-DP detailed balance: `ρ_θ · P_F(θ→θ'; t) =
   ρ_θ' · P_F(θ'→θ; t)` ≤ 1e-12 for all (θ, θ', t).
2. Lumping consistency: for each candidate `(O, T)` partition the
   lumped chain's transition kernel agrees with the marginalized
   full-chain transition kernel up to the leading-order error
   `max_{x ∈ T} ρ_x` predicted above. At small truncation that error
   is observable; at large truncation it vanishes — both regimes must
   be reproducible from the fixture.
3. Cap-2 joint cross-check at L=1: the rank-1 product agrees with the
   brute Kronecker expm to ≤ 1e-14.
4. Cap-2 joint via the L^3 sum agrees with the brute multi-trajectory
   integral (compute by enumerating all
   ((θ_P, jump times on each leaf edge, end-state)) trajectories
   numerically — small (k=2 or 3, L=2 or 3) so this is tractable).
5. Marginal consistency:
   `sum_c pi_joint(a, c) = sum_θ Σ_c π_c(c) ρ_θ pi^(c, θ)(a) = pi_lone(a)`
   to 1e-14 — i.e., dynamic-field needs no Sinkhorn. Numerical
   confirmation closes the loop on the "reversibility for free" claim
   in the reversibility note.
