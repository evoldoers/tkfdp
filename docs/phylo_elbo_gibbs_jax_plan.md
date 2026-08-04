# Phylo-ELBO Gibbs on `(c, arch_assignment)`, JAX, √2-bucketed

**Status**: design 2026-07-13. Companion to `dynfield_variational_elbo.md`
(variational family) and `mm_backward_family_gap.md` (why we go
forward-only).

## Training scheme

Fixed:
- `pi_archetype` = LG-C10 (frozen). No M-step on it, no HR SS
  extraction needed.

Sampled discretely each iter (Gibbs):
- `c_n` (per-site class assignment), per site of each cluster.
- `arch_assignment[c, theta]`, per (c, theta) with theta != 0.
- `arch_assignment[c, 0]` locked at identity `c=k`.

Marginalised each iter:
- `theta` trajectory across the tree, via forward phylo-ELBO
  (`tree_log_lik_mm`). No `theta` MCMC.

Updated each iter:
- `rho_chain` closed-form Gamma posterior (as usual).

No backward pass. HR SS not needed (nothing to Newton-step).

## Why this configuration

- **Backward pass avoided** (`mm_backward_family_gap.md`): the
  `(r * prod A + s * 1)` MM family is exact under forward mm_edge but
  loses ~1% relative at rho_chain > 0 under backward mm_edge because
  the aggregator `M/pi_stat` isn't in the family. Since we only need
  marginal likelihoods for Gibbs (not per-node posteriors), forward-
  only suffices.

- **`pi_archetype` frozen at LG-C10**: biologically-anchored 10-way
  categorical decomposition (acidic, basic, hydrophobic, ...). Cuts
  the HR/Newton machinery and lets the training loop reduce to
  Gibbs + one scalar M-step for `rho_chain`.

- **Full trees not cherries**: composite training's cherry decomposition
  has a documented "composite-likelihood freedom" that lets each
  cherry independently pick its parent-field state
  (`analysis/archetype_biophysics.py step 6`,
  `math-paper/appendix-tkfdp.tex` par:arch-phylo-elbo motivation). Full
  trees force cross-cherry `theta` consistency at internal nodes,
  eliminating this artifact.

## Per-iter Gibbs step

Per cluster: forward `tree_log_lik_mm(tree, classes, rho, pi_field,
S, rho_chain)` gives marginal log-likelihood `log P(observations |
classes, arch_assignment, pi_arch, ...)`.

Gibbs on `arch_assignment[c*, theta*] = k` (theta* != 0):
```
For k in 1..K_a:
  L_k = sum over affected clusters of tree_log_lik_mm(
      tree_i, classes_i, rho, pi_arch, S, rho_chain)
    where arch_assignment[c*, theta*] is temporarily set to k
p ~ softmax(L)
arch_assignment[c*, theta*] = Categorical(p)
```
"Affected" = clusters containing at least one site with `c_n = c*`.
Cost: `K_a` per-cluster forward evaluations per (c*, theta*), summed
over affected clusters.

Gibbs on `c_n` (per site of each cluster):
```
For c in 1..K_c:
  L_c = tree_log_lik_mm(tree_this_cluster, classes_with_n=c, ...)
p ~ softmax(L)
c_n = Categorical(p)
```
Cost: `K_c` per-cluster forward evaluations per site.

## JAX + bucketing

Two √2-geomspaced bucket dimensions per cluster:
- `m_bucket`: cluster width (number of columns).
- `N_bucket`: number of tree leaves (sequences with observations at
  these columns).

Each (`m_bucket`, `N_bucket`) → one JIT-compiled shape.

**Padded tree layout** (fixed-shape per bucket):
- Trees are decomposed into a **balanced binary of `2^depth` slots**
  where `depth = ceil(log2 N_bucket)`.
- Real leaves fill the first `N_actual` slots.
- Phantom leaves (`mask = 0`) fill the rest — their `leaf_obs = 0`
  and their forward-CLV contribution is neutralized via a `leaf_mask`
  passed through `mm_edge`/`mm_combine` reductions.
- Internal nodes decomposed level-by-level: 2^(depth - level) nodes
  per level. Multifurcating trees are represented as
  cascading binaries with zero-length "phantom" branches.

**Level-by-level forward pass** (JIT'd):
```
for level in [0, ..., depth-1]:  # leaves-first
    # At level 0: each pair of sibling leaves.
    # At level l: pair-combine 2^(depth-l) nodes into 2^(depth-l-1).
    branch_tau_l = branch_lengths[level, :]  # (2^(depth-l),)
    mm_edge_l  (vmap over the batch and over 2^(depth-l))
    mm_combine (vmap over the batch and over 2^(depth-l-1))
```
At the root (level = depth), `mm_mass_two` (or `mm_mass_one` if odd).

vmap axes: `(batch, 2^(depth-l))` per level. Sequence length fixed
by `depth = ceil(log2 N_bucket)`.

**Cluster-column dimension `m`**: appears as a fixed axis in `A_v[m,
L, A]` and `pi_arch[m, L, A]`. Per-site inner products (`dot_AB`,
etc. inside `mm_combine`) reduce over `m`. Padded m-slots (site_mask
= 0) contribute the neutral value (`log 1 = 0` or equivalent).

## Preprocessing

Guide trees already computed by
`experiments/build_pfam_full_corpus.py` (FastTree-LG). Cluster-tree
extraction: for each cluster (subset of columns), take the induced
subtree on the leaves that have any non-gap observation at any of
those columns. Cache per (family, cluster) as
```
{
  'topology': parent-pointer array (2N_leaves - 1,),
  'branch_lengths': (2N_leaves - 1,),
  'leaf_obs': (N_leaves, m) int32 with -1 for gaps,
  'level_index': list per level of node indices at that level
      (level 0 = leaves, level depth = root),
  'sibling_pairs': list per level of (node_a, node_b) pairs
      (for the mm_combine step at level l+1),
}
```

## Milestones (per commit / session)

1. **JAX mm_edge / mm_combine primitives** matching `mm_clv.py` numpy,
   with vmap-friendly signatures. Numpy-agreement smoke tests.
2. **Padded balanced-binary tree layout**: builder from a raw tree
   to the padded (level, sibling_pairs) representation. Test at
   depth 1, 2, 3.
3. **Level-by-level JIT'd forward pass**: `tree_log_lik_jax(tree_pad,
   classes, rho, pi_arch, S, rho_chain)`. Agreement with recursive
   `tree_log_lik_mm` on random trees up to depth 4.
4. **Bucketed batching**: bucket clusters by (m_bucket, N_bucket),
   pad, vmap across batch. Corpus-scale log-lik with wall-clock
   comparison to the numpy path.
5. **Gibbs on arch_assignment[c, theta]**: uses milestone 4 primitive
   with `K_a` batch axis. Empirical convergence test on synthetic
   data with known true assignment.
6. **Gibbs on c_n per site**: analogous with `K_c` batch axis.
   Integration test on Pfam top-10 subset.
7. **Trainer wire-in**: new `--phylo-elbo-gibbs` mode in
   `train_dynfield.py`. Small-scale training run (top-10 Pfam) as
   sanity check.
