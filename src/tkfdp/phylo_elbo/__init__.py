"""Phylogenetic ELBO (par:arch-phylo-elbo).

Moment-matching variational family on a tree of shape (rank-1 + scalar):

  F_v(x_v, θ_v) = r_v(θ_v) * prod_n A_v(x_{v,n} | θ_v)
                + s_v(θ_v) * prod_n π^{arch[c_n, θ_v]}(x_{v,n})

with normalisation <π^{(c_n, θ)}, A_v(· | θ)> = 1 per site n and θ.

Phase 1 (this module skeleton + edge_kernel + smoke tests) implements
the branch-kernel primitives β and W, and the forward per-site
propagation of the tracking marginal A. Later phases add moment
matching, sibling merging, backward propagation, and Gibbs on
arch_assignment.
"""
from __future__ import annotations
