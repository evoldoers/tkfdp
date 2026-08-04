"""Exact field-augmented Felsenstein peel for a dynfield cluster of ANY size m.

The exact likelihood of a cluster is
    P(x_1..x_m) = sum over (theta, Delta) trajectories of
                  P(theta, Delta) * prod_n P(x_n | theta, Delta),
whose exact peel carries an (L * A^m)-state message per node. This module is
the m-general generalisation of exact_cap2.exact_pair_ll_tree (which special-
cases m=2 with an (L, A, A) message). It is the ground-truth reference for the
product-of-trees ELBO at m=3, 4 under a REDUCED alphabet (small A so A^m is
tractable), and it must agree with exact_pair_ll_tree at m=2 and the singleton
forward at m=1.

Branch operator (child -> parent up-message), consistent with the binary-Delta
renewal branch factor (exact_cap2.field_kernels / elbo_rescorr.branch_factor):
  up_v(theta_u, x_u) =
      beta(theta_u) * ( prod_n Psub^{c_n,theta_u}(tau) applied on axis n )(CLV_v(theta_u, .))(x_u)   [Delta=0]
    + sum_{theta_v} W(theta_u, theta_v) * <prod_n pi(c_n,theta_v), CLV_v(theta_v, .)>                 [Delta=1]
Combine at an internal node is the exact elementwise product of up-messages.
Cost per edge O(L * m * A^{m+1}); fine for small A and m.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .exact_cap2 import gtr_Q, field_kernels


def _apply_Psub_all_axes(clv_theta, P_list):
    """clv_theta: (A,)*m tensor for a fixed field state; P_list[n]: (A,A) with
    convention P[x_parent, x_child]. Returns tensor over parent residues x_u:
    out(x_u) = sum_{x_v} prod_n P_n(x_{n,u}, x_{n,v}) clv_theta(x_v)."""
    out = clv_theta
    m = len(P_list)
    for n in range(m):
        # contract axis n of `out` (currently the child index) with P_n's child
        # index -> parent index, then move it back to position n.
        out = np.tensordot(P_list[n], out, axes=([1], [n]))     # new axis at 0
        out = np.moveaxis(out, 0, n)
    return out


def exact_ll_tree_general(tree, classes, rho, pi_field, S, rho_chain, eta=1.0):
    """Exact cluster log-likelihood on a general rooted tree, any m.

    tree: numpy Tree. classes (m,). rho (L,). pi_field (K_c, L, A). S (A,A).
    Leaves clamp residues to tree.leaf_obs (-1 = gap = uninformative)."""
    rho = np.asarray(rho, np.float64)
    L = int(rho.shape[0]); A = int(pi_field.shape[2]); m = int(classes.shape[0])
    nn = tree.n_nodes; root = tree.root
    cls = [int(classes[n]) for n in range(m)]
    # per-(class, theta) GTR generator
    Qk = {c: np.stack([gtr_Q(pi_field[c, th], S) for th in range(L)])
          for c in set(cls)}                                   # c -> (L,A,A)
    shape = (L,) + (A,) * m

    Pcache: dict = {}

    def P_arch(c, th, tau):
        key = (c, th, round(float(tau), 12))
        if key not in Pcache:
            Pcache[key] = expm(Qk[c][th] * float(tau) * float(eta))
        return Pcache[key]

    def leaf_clv(v):
        M = np.zeros(shape)
        # build the outer product of per-site leaf indicators, broadcast over L
        idx = [slice(None)]  # theta axis: all
        sub = np.ones((A,) * m)
        grids = []
        for n in range(m):
            ob = int(tree.leaf_obs[v, n])
            vec = np.ones(A) if ob < 0 else np.eye(A)[ob]
            grids.append(vec)
        # outer product across sites
        og = grids[0]
        for n in range(1, m):
            og = np.multiply.outer(og, grids[n])
        M[:] = og[None, ...]
        return M

    logscale = np.zeros(nn)
    clv = [None] * nn
    for v in tree.post_order:
        v = int(v)
        if tree.is_leaf(v):
            clv[v] = leaf_clv(v); logscale[v] = 0.0
            continue
        Mv = np.ones(shape); lsc = 0.0
        for c in tree.children[v]:
            c = int(c)
            tau = float(tree.branch_length[c])
            _, beta, _ = field_kernels(rho, rho_chain, tau)
            Wm = field_kernels(rho, rho_chain, tau)[0] - np.diag(beta)  # (L,L)
            up = np.zeros(shape)
            # projections proj_c(theta_c) = <prod pi, clv_c(theta_c, .)>
            proj = np.zeros(L)
            for tc in range(L):
                pi_out = np.ones((A,) * m)
                og = pi_field[cls[0], tc]
                for n in range(1, m):
                    og = np.multiply.outer(og, pi_field[cls[n], tc])
                proj[tc] = float((og * clv[c][tc]).sum())
            for tu in range(L):
                # Delta=0: theta_c = tu, apply Psub per site
                P_list = [P_arch(cls[n], tu, tau) for n in range(m)]
                nj = beta[tu] * _apply_Psub_all_axes(clv[c][tu], P_list)
                # Delta=1: constant in x_u
                jmp = float(Wm[tu] @ proj)
                up[tu] = nj + jmp
            Mv = Mv * up
            lsc += logscale[c]
        mx = Mv.max()
        if mx > 0:
            Mv = Mv / mx; lsc += np.log(mx)
        clv[v] = Mv; logscale[v] = lsc
    # root integration: sum_{theta, x} rho(theta) prod_n pi(c_n,theta,x_n) clv_root
    total = 0.0
    for tr in range(L):
        og = pi_field[cls[0], tr]
        for n in range(1, m):
            og = np.multiply.outer(og, pi_field[cls[n], tr])
        total += float(rho[tr]) * float((og * clv[root][tr]).sum())
    return float(np.log(max(total, 1e-300)) + logscale[root])
