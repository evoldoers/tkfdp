# Finding: the moment-matching family is closed under forward log-lik at cap-2 cherry but NOT under backward posterior

**Status**: diagnosed 2026-07-03 via analytic 4-term derivation on m=1, L=2 cherry with ρ_chain=0.5. Verified against brute-force enumeration in
`tests/phylo_elbo/test_backward_vs_exact.py`.

## Summary

The evolmoves derivation (mid-2026) states that the moment-matching (MM)
family for the dynfield block CLV — `r · ∏A(x, θ) + s · ∏π(x, θ)`
with the normalisation `⟨π, A⟩ = 1` — is exact on cap-2 cherries. This
holds for the **forward tree log-likelihood** but NOT for **backward
per-node posteriors** at ρ_chain > 0. The two are computed by
different pipelines:

- **Forward LL** integrates against `ρ · ∏π` at the root via
  `mm_mass_two` — an exact 4-term formula with no projection. At
  cherries with rank-1 leaf messages, this produces exact numbers.
  Verified in `tests/phylo_elbo/test_mm_clv_cherry_exact.py`
  (machine precision, 9/9 configs).
- **Backward posterior** requires downward CLV propagation at every
  non-root edge. `mm_combine` (used to fold `D_parent · ∏siblings U`
  into the family) and `mm_edge` (used to propagate across the branch)
  both project onto the (rank-1 + scalar) form. This introduces
  systematic error at ρ_chain > 0.

Verified residuals at cap-2 cherry (L=2, A=3, m=1):

- ρ_chain = 0.0: exact to machine precision at all nodes
- ρ_chain = 0.5: ~4e-3 at leaves; 8.7e-5 at root
- ρ_chain = 2.0: ~4e-3 at leaves; 4.1e-3 at root

## Analytical mechanism

Consider a cherry with two leaves (x_0, x_1) and one root, branches of
length τ. The exact `D_leaf_0(x, θ) = P(obs at leaf 1 | x_leaf_0 = x,
θ_leaf_0 = θ)` decomposes by field jumps on the leaf_0-to-root branch:

**Case A (no field jump)**: θ_root = θ. The contribution to
`D_leaf_0(x, θ)` is
```
β²(θ) · P²(x, x_1; θ)                — rank-1 shape in x (couples x_leaf to x_1)
+ β(θ) · J(θ, x_1)                    — SCALAR in x (from marginalising θ_leaf_1)
```
where `J(θ, x_1) = ∑_θ' W(θ, θ') · π(x_1, θ')`.

**Case B (≥1 field jump on leaf edge)**: contributes to
`D_leaf_0(x, θ)` under Bayes' inversion using detailed balance
`P(x_root, θ_root | x, θ) = ρ_θ_root π(x_root, θ_root) K^fwd(x, θ | ...) / (ρ_θ π(x, θ))`:
```
∑_θ_r W(θ, θ_r) β(θ_r) π(x_1, θ_r)                  — SCALAR in x
+ ∑_θ_r, θ_1 W(θ, θ_r) W(θ_r, θ_1) π(x_1, θ_1)      — SCALAR in x
```

The key observation: the jump-emission `K_jump(x_leaf | ...) = W(...)
· π(x_leaf, θ_leaf)`. Under Bayes' inversion, the `1/(ρ_θ π(x_leaf,
θ_leaf))` factor from the prior cancels the `π(x_leaf, θ_leaf)` from
the emission, leaving the contribution to `D_leaf_0(x_leaf, θ)` as
**x_leaf-independent** — a constant in x.

## What the MMClv family cannot represent

The MMClv family `r · A(x, θ) + s · π(x, θ)` has two components. To
represent an x-independent constant `t`, one would need
`t = s · π(x, θ)` for all x, which requires `s = t/π(x, θ)` — not a
valid element of the family (s must be a function of θ only).

The projection at `mm_combine` absorbs the x-independent scalar
component into `r · A(x)` — reshaping it into rank-1 form. Concretely
`mm_combine((1, 0, ones), U_1)` returns
```
combined(x, θ) = β P(x, x_1; θ) + jump                    (constant in x)
```
instead of the exact `1 · U_1 = β P(x, x_1; θ) + jump · π(x, θ)`.
The jump-scalar has been reshaped from `π(x)` to `1(x)`.

Downstream, when `joint_marginal_FxD(F_leaf, D_leaf)` computes
`⟨π, A_F · A_D⟩`, this reshaping enters `A_D` and produces a
systematically biased posterior. The bias magnitude scales with the
jump amplitude and the deviation of `π` from uniform.

## Three routes forward

1. **Extended family: `r · A(x, θ) + s · π(x, θ) + t · 1(x, θ)`** —
   introducing a third component for the x-independent scalar. Costs:
   `mm_edge`, `mm_combine`, `mm_mass_one/two` all need to preserve the
   extra term. `mm_combine`'s projection becomes a 9-term product
   rather than 4-term, with corresponding 3 outputs. Probably the
   cleanest fix; requires substantive re-derivation of the linear
   algebra.

2. **Non-MMClv backward for small trees**: compute D exactly via
   full enumeration or matrix exp. Feasible up to ~30-leaf trees at
   K_a=8, m=5 (state space ≤ 2^30 × 20^5). Grows exponentially; not
   scalable but useful for validation.

3. **Accept the bias, document it, validate at ρ_chain=0**. The
   ρ_chain=0 case is exact at all depths (verified) and serves as an
   integration-test lever. For downstream HR extraction, the bias is
   quantified (~1% relative) and might be acceptable depending on
   the SEM's convergence sensitivity.

## Consequence for the composite-likelihood artifact fix

The composite-likelihood artifact (diagnosed in
`analysis/archetype_biophysics.py step 6`) motivated the tree-ELBO
work. The FORWARD LL at cap-2 cherries is exact and the SEM training
loop only needs the forward direction (E-step: HR stats via forward
pass; M-step: update pi_arch, rho_chain). If the tree structure
constrains θ across sibling branches at internal nodes, the
composite-lik freedom is reduced regardless of backward precision.

HR extraction (which currently requires backward for per-branch pair
posteriors) can be reformulated in terms of forward-only quantities:
per-branch (θ_v, θ_u) marginals derived via a forward-only sweep with
in-place edge accumulation, avoiding the backward and its
representational gap. This is a design change worth exploring before
committing to route 1 or 2.

## References

- `math-paper/appendix-tkfdp.tex`, `par:arch-phylo-elbo{,-backward,
  -bridge,-jumps,-hr}` — full derivation attempt
- `src/tkfdp/coupling/dynfield/phylo_elbo/mm_clv.py` — MMClv
  primitives, ported from `~/evolmoves/ts/cluster/dynfield-emission.ts`
- `tests/phylo_elbo/test_backward_vs_exact.py` — brute-force exact
  vs MMClv backward, quantifying the ρ_chain > 0 residual
- Session logs: git commits `342f5ee`, `05c1d5f`, `dad8d4e` (backward
  fixes + diagnosis)
