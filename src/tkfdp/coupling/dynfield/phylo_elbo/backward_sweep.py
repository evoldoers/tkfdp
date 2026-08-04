"""Belief-propagation backward sweep to extract per-node theta marginal
posteriors from the converged forward CLVs.

Given the forward tree pass (post-order mm_edge + mm_combine sweep from
tree_log_lik.py), each node v has an upward CLV `F_v(x, theta)` that
encodes the LIKELIHOOD of the observations at leaves rooted at v.

The backward sweep computes DOWNWARD CLVs `D_v(x, theta)` = likelihood
of observations OUTSIDE the subtree rooted at v, given (x_v, theta_v).

The posterior at node v is
  q_v(x, theta)  proportional to  F_v(x, theta) * D_v(x, theta)

Marginalising residues via the < pi, A > = 1 convention:
  q_v(theta)     proportional to  rho1_F(theta) * ...  + s_F(theta) * ...

For this MVP we return the theta marginal q_v(theta) only. The
per-site residue marginals p_{n, v}(x | theta) can be computed via
downward extension of the A vectors when HR extraction needs them.

Algorithm:

  Post-order forward sweep (already done in tree_log_lik.py):
    For each node v: `forward_clvs[v] = MMClv(...)`.

  Pre-order backward sweep:
    root: D_root(theta) = rho[theta] * <pi_field-implicit at root>
                        = the joint stationary prior (encoded so that
                          the mass integration matches mm_mass_two).
    For internal node v with parent p, siblings s_i, incoming edge tau_v:
      1. Combine parent's D_p with all siblings' upward messages
         (mm_edge over their branch lengths, then mm_combine into a
         single (rank-1+scalar) form).
      2. Propagate the combined message through v's incoming edge in
         the DOWNWARD direction. Because our F81-on-DP field CTMC is
         REVERSIBLE (detailed balance under rho), the downward edge
         kernel has the same form as the forward one; mm_edge can be
         reused with a swap of endpoint labels.
      3. D_v = combined-downstream after edge propagation.

  Marginal at v:
    q_v(theta) = normalise( F_v(theta) * D_v(theta) )
    where the "theta-marginal" of an MMClv = rho1(theta) + s(theta)
    (the < pi, . > mass integration convention; see mm_mass_one for
    how this ties to the log-lik).

For the tree_log_lik test cases (depth 1-3), the backward sweep at
the root should trivially yield q_root(theta) = rho[theta] * F_root
mass / total-mass (which is just the ROOT posterior).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .mm_clv import (
    MMClv, leaf_clv, mm_edge, mm_combine, mm_rescale,
    mm_mass_two, mm_mass_one, per_class_field_Q)
from .tree import Tree


def forward_clvs(tree: Tree,
                    classes: np.ndarray,
                    rho: np.ndarray,
                    pi_field: np.ndarray,
                    S: np.ndarray,
                    rho_chain: float,
                    eta: float = 1.0) -> 'list[MMClv]':
    """Full forward sweep, returning per-node upward CLV. Post-order.

    Returns a list of MMClv indexed by node id. For a leaf v: leaf_clv
    of tree.leaf_obs[v]. For an internal node v: mm_combine of the
    edge-propagated messages from each child.
    """
    Q = per_class_field_Q(pi_field, S)
    clvs: 'list[Optional[MMClv]]' = [None] * tree.n_nodes
    for v in tree.post_order:
        v = int(v)
        if tree.is_leaf(v):
            clvs[v] = leaf_clv(tree.leaf_obs[v], classes, pi_field)
        else:
            msgs: 'list[MMClv]' = []
            for c in tree.children[v]:
                child_clv = clvs[c]
                tau_c = float(tree.branch_length[c])
                msg = mm_edge(child_clv, classes, tau_c, rho, pi_field,
                                Q, rho_chain, eta=eta)
                msg = mm_rescale(msg)
                msgs.append(msg)
            acc = msgs[0]
            for msg in msgs[1:]:
                acc = mm_rescale(mm_combine(acc, msg, classes, pi_field))
            clvs[v] = acc
    return clvs  # type: ignore


def _joint_marginal_FxD(F: MMClv, D: MMClv, classes: np.ndarray,
                             pi_field: np.ndarray) -> np.ndarray:
    """Per-theta joint marginal <pi_all, F * D>(theta) under the
    (r * prod A + s * 1) CLV form.

    F * D = r_F r_D prod (A_F A_D) + r_F s_D prod A_F + s_F r_D prod A_D
            + s_F s_D * 1

    <pi_all, .>(theta) using invariants <pi, A> = 1 and <pi, 1> = 1:
      = r_F r_D * prodDot_FD  +  r_F s_D  +  s_F r_D  +  s_F s_D

    Same shape as mm_mass_two's contribution at a fixed theta.
    (Previous versions of this code included extra <pi, A * pi> and
    <pi, pi^2> factors on the scalar cross-terms -- artefact of an
    incorrect assumption that the scalar component was prod pi rather
    than the constant 1.)
    """
    L = int(pi_field.shape[1])
    m = int(classes.shape[0])
    out = np.zeros(L, dtype=np.float64)
    for th in range(L):
        prodDot_FD = 1.0
        for n in range(m):
            pi_row = pi_field[int(classes[n]), th, :]
            prodDot_FD *= float(np.sum(pi_row * F.A[n, th] * D.A[n, th]))
        rF, sF = float(F.rho1[th]), float(F.s[th])
        rD, sD = float(D.rho1[th]), float(D.s[th])
        out[th] = rF * rD * prodDot_FD + rF * sD + sF * rD + sF * sD
    return out


def backward_theta_marginals(tree: Tree,
                                classes: np.ndarray,
                                rho: np.ndarray,
                                pi_field: np.ndarray,
                                S: np.ndarray,
                                rho_chain: float,
                                forward: 'Optional[list[MMClv]]' = None,
                                eta: float = 1.0) -> 'list[np.ndarray]':
    """Extract per-node theta marginal posteriors q_v(theta).

    Args:
      tree, classes, rho, pi_field, S, rho_chain, eta: as in
        tree_log_lik_mm.
      forward: precomputed forward CLV list. If None, run forward
        sweep here.

    Returns a list of (L,) numpy arrays indexed by node id. Each q_v
    is a valid simplex on the L field states.
    """
    if forward is None:
        forward = forward_clvs(tree, classes, rho, pi_field, S,
                                     rho_chain, eta=eta)
    Q = per_class_field_Q(pi_field, S)
    L = int(rho.shape[0])
    n_nodes = tree.n_nodes

    # D_root(x, theta) = 1 (constant). No observations outside the root's
    # subtree. In MMClv rep: rho1 = ones per theta, A[n, theta, :] = ones,
    # s = 0. Check: rho1 * prod A + s * prod pi = 1 * prod 1 + 0 = 1 for
    # any (x, theta). The prior rho * prod pi enters explicitly in the
    # posterior computation q_v proportional to
    # rho[theta] * sum_x prod pi(x, theta) * F_v(x, theta) * D_v(x, theta).
    down: 'list[Optional[MMClv]]' = [None] * n_nodes
    root = tree.root
    A_alph = pi_field.shape[2]
    m = tree.m
    D_root_A = np.ones((m, L, A_alph), dtype=np.float64)
    D_root = MMClv(rho1=np.ones(L), A=D_root_A,
                       s=np.zeros(L), log_scale=0.0)
    down[root] = D_root

    # Pre-order sweep from root to leaves.
    for v in tree.pre_order:
        v = int(v)
        if tree.is_leaf(v):
            continue
        parent_down = down[v]
        for c in tree.children[v]:
            combined = parent_down
            for s in tree.children[v]:
                if s == c: continue
                sib_forward = forward[s]
                tau_s = float(tree.branch_length[s])
                sib_msg = mm_edge(sib_forward, classes, tau_s, rho,
                                        pi_field, Q, rho_chain, eta=eta)
                sib_msg = mm_rescale(sib_msg)
                combined = mm_rescale(
                    mm_combine(combined, sib_msg, classes, pi_field))
            # Propagate combined through c's incoming edge downwards.
            # KEY INSIGHT under reversibility of the compound CTMC:
            # backward propagation of D_v through an edge is the SAME
            # operation as forward mm_edge, applied to the parent-side
            # aggregated message. This is because reversibility
            # transforms the reverse-time transition into the forward
            # transition, up to prior-scaling factors that CANCEL
            # exactly against the P_stat weighting in D_v's definition.
            tau_c = float(tree.branch_length[c])
            D_c = mm_edge(combined, classes, tau_c, rho, pi_field,
                              Q, rho_chain, eta=eta)
            D_c = mm_rescale(D_c)
            down[c] = D_c

    # Compute per-node theta marginal via the exact 4-term integral
    # q_v(theta) proportional to rho[theta] * sum_{x_v} prod pi(x_v, theta)
    #                                       * F_v(x_v, theta) * D_v(x_v, theta),
    # expanded to
    #   rF rD prodDot_{<pi, A_F * A_D>}
    #   + rF sD prodDot_{<pi, A_F * pi>}
    #   + sF rD prodDot_{<pi, pi * A_D>}
    #   + sF sD prodDot_{<pi, pi * pi>}
    # per site. Not an approximation; exactly correct under the
    # (rank-1 + scalar) family.
    q_thetas: 'list[np.ndarray]' = []
    for v in range(n_nodes):
        F_v = forward[v]
        D_v = down[v]
        joint_mass = _joint_marginal_FxD(F_v, D_v, classes, pi_field)
        joint = np.asarray(rho, dtype=np.float64) * joint_mass
        total = float(joint.sum())
        if total <= 0.0 or not np.isfinite(total):
            q = np.full(L, 1.0 / L)
        else:
            q = joint / total
        q_thetas.append(q)
    return q_thetas
