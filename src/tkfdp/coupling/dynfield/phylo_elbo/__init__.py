"""Phylogenetic tree-ELBO training backend for dynfield archetypes.

See par:arch-phylo-elbo in appendix-tkfdp.tex for the derivation and
src/tkfdp/coupling/dynfield/backends.py for the training-loop
integration point.

Module layout (Phase 2, in progress):

  tree.py            Tree data structure (topology + leaf obs +
                     internal-node bookkeeping); post-order and
                     pre-order iterators; cherry / star / arbitrary
                     binary trees.
  variational.py     Per-node variational state (q_v, p_{n, v},
                     lambda_v) and construction helpers.
  edge_propagate.py  Per-edge kernel and forward/backward message
                     propagation preserving the (rank-1 + scalar)
                     CLV form.
  moment_match.py    Internal-node combining via moment-matching
                     projection (q_v, per-site marginals, lambda_v
                     all closed form).
  elbo.py            ELBO term computation: leaf emissions, per-edge
                     cross-entropies, per-node entropies.
  sweeps.py          Post-order + pre-order coordinate-ascent sweeps.
  hr_extract.py      Per-branch HR statistics from converged q,
                     aggregated to (V, U, W, N_theta_sum, T_sum) as
                     per par:arch-phylo-elbo-hr.
  preprocessing.py   FastTree wrapper: build guide trees from Pfam
                     MSAs and cache to disk alongside the cherry
                     npz files. Runs once per family.
"""
