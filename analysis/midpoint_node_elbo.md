# Midpoint-node variational ELBO vs the closed-form path ELBO

A new variational lower bound on the endpoint-conditioned pair log-transition
`log[e^{tW}]_xy` that places a variational distribution on the **state at the
middle of the branch**, benchmarked the same way as the path ELBO / Cohn
comparison (whole 400x400 matrix, sqrt-Metropolis pair models, branch-length
grid, exact `expm` reference).

Code: `experiments/midpoint_elbo.py` (bound + self-checks),
`experiments/midpoint_vs_path_elbo.py` (benchmark),
`experiments/plot_midpoint_vs_path.py` (figure). Numbers:
`results/pair_models/midpoint_vs_path_elbo.json`. Figure:
`experiments/figures/midpoint_vs_path_elbo.pdf`.

## Why this is new

The `phylo_elbo` package already puts a variational `q` on *phylogenetic-tree*
internal nodes; there was no branch-midpoint variational bound benchmarkable
against exact `expm` on the single 2-site branch. The paper's theoretical
midpoint mesh node `nu` (`appendix-tkfdp.tex:844`) is slack at the optimum
(pinned to the product-bridge marginals, "never leaves the product family"). The
bound here is NOT slack -- see below.

## Construction (no coupled expm required)

Split the branch at `t/2`. Chapman-Kolmogorov is exact:

    [e^{tW}]_xy = sum_z [e^{(t/2)W}]_xz [e^{(t/2)W}]_zy .

Put a variational `q(z)` on the midpoint pair-state `z=(z_i,z_j)`, apply Jensen,
and then bound each **half-transition by its own closed-form path ELBO**
`L_path(.,.;t/2) = closed_form_L(W,R,pi_prod,t/2)` (product-bridge Girsanov bound):

    log[e^{tW}]_xy  >=  sum_z q(z)[ log e^{(t/2)W}_xz + log e^{(t/2)W}_zy ] + H(q)     (Jensen)
                    >=  sum_z q(z)[ L_path(x,z;t/2) + L_path(z,y;t/2) ] + H(q)          (half path ELBOs)

Both steps are lower bounds and compose (q >= 0), so the result is a strict lower
bound. Crucially `L_path` uses ONLY the eigendecomposition of the **product**
bridge `R` plus `W`'s rate-matrix entries in the Girsanov correction -- it never
forms a coupled `expm`. This is `midpoint_elbo_path_edges` -- the honest, scalable
bound.

### Optimising q: coordinate-ascent variational inference (CAVI)

The tractable family for the midpoint is the mean field over the two sites,
`q(z) = q_i(z_i) q_j(z_j)` -- two 20-vectors per endpoint pair. Nothing is
diagonalised inside the optimisation (the half-branch tables `L_path(.,.;t/2)` are
computed once, outside). The ELBO is maximised by **CAVI**: hold one factor fixed
and set the other to its closed-form optimum, then swap:

    q_i(z_i) proportional to exp( sum_zj q_j(z_j) [ L_half(x,(zi,zj)) + L_half((zi,zj),y) ] )
    q_j(z_j) proportional to exp( sum_zi q_i(z_i) [ L_half(x,(zi,zj)) + L_half((zi,zj),y) ] )

Each update is a softmax of a linear function of the other 20-vector (the bracket
is the precomputed edge sum). With two factors this is 2-block coordinate ascent;
it is monotone in the ELBO and converges in ~3-4 sweeps. It is vectorised over all
160,000 `(x,y)` pairs and JIT-compiled as a JAX `fori_loop` (`_cavi_jax`), so the
whole 400x400 bound is one fused GPU kernel (see cost below).

**Diagnostic variant** (`midpoint_elbo_matrix`): replace the half path ELBOs with
the EXACT coupled half-transitions `e^{(t/2)W}`. This needs the eig of the full
coupled `W` (so it does not scale better than exact); it is only used to (i)
self-check the edges/reshape (full-joint `q` reproduces exact to 1e-9..1e-12) and
(ii) show the ceiling reachable if the half-edges were exact. It is not a fair
alternative to the path ELBO because it feeds the bound near-exact half-branch
information.

## Results (sqrt-Metropolis models; gap = exact - approx, meaningful entries)

`mpath` = midpoint with path-ELBO half-edges (product-eig only, the fair bound);
`mexact` = midpoint with exact coupled half-edges (diagnostic; uses coupled expm).

| model | t | path ELBO | mpath (fair) | ratio | mexact (diag) |
|---|---|---|---|---|---|
| trRosetta | 1.0 | 0.0017 | 0.0009 | 1.9x | 0.0003 |
| trRosetta | 2.0 | 0.0067 | 0.0042 | 1.6x | 0.0013 |
| trRosetta | 3.0 | 0.0144 | 0.0099 | 1.5x | 0.0030 |
| MI=0.174 | 1.0 | 0.0181 | 0.0088 | 2.1x | 0.0025 |
| MI=0.174 | 3.0 | 0.1659 | 0.1033 | 1.6x | 0.0306 |
| MI=0.290 | 1.0 | 0.0301 | 0.0161 | 1.9x | 0.0051 |
| MI=0.290 | 2.0 | 0.1233 | 0.0770 | 1.6x | 0.0246 |
| MI=0.290 | 3.0 | 0.2696 | 0.1867 | 1.4x | 0.0576 |

At short branches (`t <= 0.2`) all gaps are ~0. Global Spearman is ~1.000 for both
path and mpath, with mpath slightly ahead at long branches (0.9965 vs 0.9931 at
t=3.0, MI=0.29). Both are strict lower bounds (mpath min-slack `>= 0` to numerical
precision; the small negatives at t<=0.05 are the log-floor on sub-1e-8 entries).
The product baseline is NOT a bound (it overshoots exact).

## Reading

- **A variational midpoint node genuinely tightens the closed-form path ELBO by
  ~1.4-2.1x, using ONLY the product-bridge eig (no coupled expm).** The
  improvement grows with coupling and is largest at intermediate branch lengths.

- **It is not slack** -- contrast the paper's product-family midpoint mesh node
  `nu`, which is. The reason: `nu` is a mesh point ON a single product bridge (the
  bridge is already the optimal product-family path, so a mesh point on it adds
  nothing). This construction instead bounds each HALF by its own independent path
  ELBO and ties them with a discrete `q(z)`; marginalising `z` gives a **mixture
  of product bridges through the midpoint**, a strictly richer family than one
  product bridge -- hence the gain.

- **Cost (JIT'd JAX, GPU; whole 400x400).** The CAVI is a JAX `fori_loop`
  (`_cavi_jax`), NOT numpy -- an earlier numpy version was ~1-5 s purely from
  missing BLAS/fusion + an over-tight tolerance; that was an implementation bug,
  not the method. Measured (t=1, MI=0.29):
    - product-bridge eig + coupling table `C` (t-independent): **~25 ms**
    - path-ELBO per-t kernel: ~7 ms;  half-branch edge kernel: ~9 ms
    - CAVI, 6 sweeps (converges in 3-4): **~30 ms** fp64 / **~2.6 ms** fp32
  So: **path ELBO standalone ~23 ms**; **midpoint standalone ~59 ms (2.6x)** (own
  eig + edges + CAVI); **midpoint as a refinement that shares the path ELBO's eig
  ~42 ms (1.8x)** (edges + CAVI, no second eig -- `eig_rev(R)` is t-independent, so
  `closed_form_L_precompute` is computed once and reused). The floor is the fp64
  CAVI (~30 ms), not the eig; `fp32=True` cuts the CAVI ~12x (refinement -> ~1x the
  path ELBO) at ~4e-3 looseness -- fine for training/ranking, not for this
  tight-gap benchmark. It stays **product-eig only**, so it scales like the path
  ELBO (a coupled `expm` is never formed). The exact-edge diagnostic is ~5-6x
  tighter but needs the coupled `expm`, so it does not scale.

- **Free split fraction (the midpoint "slider").** The split `s` (branch cut into
  `e^{s t W}` and `e^{(1-s)t W}`) is itself a valid free parameter: any `s` is a
  lower bound, so the per-pair MAX over an `s`-grid is too (`midpoint_elbo_path_edges
  (..., s_grid=...)`). Measured, it tightens only **~3-4%** over the fixed `s=0.5`
  (7-point grid, ~7x cost): for these symmetric reversible models the optimal split
  is ~0.5 and the ELBO is flat in `s` near it. Kept as an option -- it should matter
  for asymmetric settings (a cherry with unequal branch lengths, or non-reversible
  `W`), not the symmetric single-branch benchmark.

- **Extensions.** The `mpath -> mexact` gap (e.g. 0.19 vs 0.058 at t=3, MI=0.29)
  is the looseness of the half-branch path ELBOs propagating through. Closing it
  without a coupled `expm` = **recursion**: bound each half with a midpoint node
  too (dyadic refinement). Also: `K>1` interior nodes; a joint (non-mean-field)
  `q` on the midpoint interpolates toward the exact-edge ceiling.

## Reproduce

    # GPU 1 (JIT'd JAX); drop CUDA_VISIBLE_DEVICES + set JAX_PLATFORMS=cpu for CPU
    CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONPATH=src:experiments \
      python3 experiments/midpoint_elbo.py            # self-checks
      python3 experiments/midpoint_vs_path_elbo.py    # benchmark -> JSON (records ms_path, ms_mp)
      python3 experiments/plot_midpoint_vs_path.py    # figure
