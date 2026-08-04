"""Product-of-trees structured mean-field ELBO (draft-product-trees-elbo.tex).

The variational family is a PRODUCT OF INDEPENDENT tree-structured factors

    q(theta, Delta, x_1..x_m) = q_Theta(theta, Delta) * prod_n q_n(x_n)

where q_Theta is a tree-Markov law over the field-and-jumps configuration
(exactly the MRF that field_bp handles) and each q_n is a tree-Markov law over
site n's residue trajectory -- a FULL along-tree A-state chain, NOT crushed to
per-node marginals (the key difference from elbo.py's node-mean-field floor)
and NOT conditioned on the field (the key difference from elbo_persite, which
keeps the field<->residue seam). The factors are independent; coordinate ascent
lets them synchronise.

Cost is O(m * N * A^2) + O(N * L^2) -- LINEAR in m, never A^m -- so it stays
tractable for clusters of any size m, where the exact L*A^m peel explodes.

Coordinate ascent (structured VMP; see the derivation for the boxed formulae):
  * residue update: per site, sum-product on the site-n tree whose per-branch
    operator is the GEOMETRIC mean (exp of the q_Theta-edge-marginal-weighted
    log) of the per-field substitution operators -- NOT the arithmetic mean.
  * field update: fold each site's expected residue evidence into the field
    MRF node/edge potentials, solve with field_logbp.
ELBO = logZ_field + sum_n H[q_n]  (Prop. "field partition function realises the
field-side ELBO"); H[q_n] is the tree-chain (junction-tree) entropy.

It DROPS the field<->residue coupling (the seam between q_Theta and the q_n);
that is the remaining approximation. It is a genuine lower bound always, and
exact under field-independent emissions (Prop. exactness); see the eval writeup
analysis/product_of_trees_elbo_eval.md for measured tightness on real trees.
"""
from __future__ import annotations

import numpy as np

from .mm_clv import (field_transition_row, beta_no_jump, per_class_field_Q,
                     branch_P)
from .field_bp import field_logbp, _logsumexp

_FLOOR = 1e-300
_LOGFLOOR = np.log(_FLOOR)


def _entropy(p, axis=-1):
    p = np.asarray(p, np.float64)
    return -(p * np.log(np.clip(p, _FLOOR, None))).sum(axis=axis)


def elbo_product_trees(tree, classes, rho, pi_field, S, rho_chain, eta=1.0,
                       n_iters=80, tol=1e-9, warm_start=True,
                       return_terms=False):
    """Product-of-trees structured mean-field ELBO.

    Interface mirrors elbo_persite: tree, classes (m,), rho (L,),
    pi_field (K_c, L, A), S (A, A), rho_chain, eta.

    Returns the ELBO (float), or a dict with terms if return_terms=True
    (elbo, logZ_field, H_res, iters, elbo_trace, warm_elbo).
    """
    rho = np.asarray(rho, np.float64)
    L = int(rho.shape[0]); A = int(pi_field.shape[2]); m = int(classes.shape[0])
    nn = tree.n_nodes; root = tree.root
    is_leaf = [tree.is_leaf(v) for v in range(nn)]
    cls = [int(classes[n]) for n in range(m)]
    Q = per_class_field_Q(pi_field, S)

    # log stationary per site/field: logpi[n] (L, A)
    logpi = np.stack([np.log(np.clip(pi_field[cls[n]], _FLOOR, None))
                      for n in range(m)])                       # (m, L, A)

    def leaf_obs(v):
        return [int(tree.leaf_obs[v, n]) for n in range(m)]

    # ---- fixed per-edge kernels (independent of q; precompute once) ----------
    # Pth[v] (L,L) field transition; beta[v] (L,); W[v] (L,L) = Pth - diag(beta)
    # logPsub[v] (m, L, A, A) = log expm(Q(c_n, theta) tau_v)
    Pth = {}; beta = {}; Wm = {}; logPsub = {}
    for v in range(nn):
        if v == root:
            continue
        tau = float(tree.branch_length[v])
        Pth[v] = np.stack([field_transition_row(rho, tu, tau, rho_chain)
                           for tu in range(L)])                 # (L,L)
        beta[v] = np.array([beta_no_jump(rho, tu, tau, rho_chain)
                            for tu in range(L)])                # (L,)
        Wm[v] = Pth[v] - np.diag(beta[v])                       # (L,L)
        logPsub[v] = np.stack(
            [np.log(np.clip(branch_P(Q, cls[n], th, tau, eta), _FLOOR, None))
             for n in range(m) for th in range(L)]
        ).reshape(m, L, A, A)

    # ------------------------------------------------------------------ q_n BP
    def residue_bp(site, log_root_pot, log_node_pot, log_edge_op):
        """Sum-product on the site tree pairwise MRF. Returns (mu, xi, Hqn).
          log_root_pot (A,)            : root node log-potential
          log_node_pot[v] (A,)         : non-root node log-potential (leaves
                                         already restricted to obs by caller)
          log_edge_op[v] (A,A)         : edge log-potential [x_parent, x_child]
        mu[v] (A,) node marginal; xi[v] (A,A) edge marginal [x_u, x_v];
        Hqn scalar rooted tree-chain entropy.
        """
        # upward log-messages: up[v](x_v) = evidence from subtree(v) at v,
        # excluding v's own node potential's contribution above.
        up = [None] * nn
        for v in tree.post_order:
            v = int(v)
            if is_leaf[v]:
                lm = np.full(A, _LOGFLOOR)
                lm[leaf_obs(v)[site]] = 0.0            # clamp; -1 handled below
                if leaf_obs(v)[site] < 0:
                    lm = np.zeros(A)                   # gap: uninformative
                up[v] = lm + log_node_pot[v]
                continue
            acc = log_root_pot.copy() if v == root else log_node_pot[v].copy()
            for c in tree.children[v]:
                c = int(c)
                # message to v(x_v) = logsumexp_{x_c}[ op_c(x_v,x_c) + up[c](x_c) ]
                acc = acc + _logsumexp(log_edge_op[c] + up[c][None, :], axis=1)
            up[v] = acc
        logZ = _logsumexp(up[root])
        # downward log-messages: dn[v](x_v) = evidence from outside subtree(v)
        dn = [None] * nn
        dn[root] = np.zeros(A)
        for v in tree.pre_order:
            v = int(v)
            if is_leaf[v]:
                continue
            base = (log_root_pot if v == root else log_node_pot[v]) + dn[v]
            # precompute sum of child up-messages
            child_up = {}
            for c in tree.children[v]:
                c = int(c)
                child_up[c] = _logsumexp(log_edge_op[c] + up[c][None, :], axis=1)
            tot = base + sum(child_up.values())
            for c in tree.children[v]:
                c = int(c)
                cav = tot - child_up[c]                # cavity at v excl. c
                dn[c] = _logsumexp(log_edge_op[c] + cav[:, None], axis=0)
        # node marginals
        mu = [None] * nn
        for v in range(nn):
            npot = log_root_pot if v == root else log_node_pot[v]
            if is_leaf[v]:
                lm = np.full(A, _LOGFLOOR)
                ob = leaf_obs(v)[site]
                if ob < 0:
                    lm = np.zeros(A)
                else:
                    lm[ob] = 0.0
                lg = lm + npot + dn[v]
            else:
                lg = up[v] + dn[v]
            mu[v] = np.exp(lg - _logsumexp(lg))
        # edge marginals xi[v](x_u, x_v) and entropy
        xi = {}
        Hqn = float(_entropy(mu[root]))
        for v in range(nn):
            if v == root:
                continue
            u = int(tree.parent[v])
            msg_uv = _logsumexp(log_edge_op[v] + up[v][None, :], axis=1)
            # up[u] = npot(u) + sum_children msg, so up[u]+dn[u] is the full
            # belief at u; cavity excluding v subtracts v's upward message.
            cav_u = up[u] + dn[u] - msg_uv
            # up[v] is exactly the upward belief into the edge (clamp+npot for a
            # leaf; npot+children messages for an internal node).
            below_v = up[v]
            lg = cav_u[:, None] + log_edge_op[v] + below_v[None, :]
            e = np.exp(lg - _logsumexp(lg))
            xi[v] = e
            # H(x_v | x_u) = -sum e (log e - log mu_u)
            mu_u = np.clip(mu[u], _FLOOR, None)
            with np.errstate(divide='ignore', invalid='ignore'):
                term = e * (np.log(np.clip(e, _FLOOR, None)) - np.log(mu_u)[:, None])
            Hqn += -float(np.nansum(term))
        return mu, xi, Hqn

    # ------------------------------------------------------ residue block update
    def residue_update(b_nodes, e_edge):
        """Given field node marginals b_nodes[v] (L,) and edge marginals
        e_edge[v] (L,L,2), build the site-n MRF log-potentials and run residue
        BP. Returns (mu_all, xi_all, H_res) where mu_all[n][v], xi_all[n][v]."""
        b_root = b_nodes[root]                                  # (L,)
        mu_all = [None] * m; xi_all = [None] * m; H_res = 0.0
        for n in range(m):
            # root node potential: sum_theta b_root(theta) logpi(theta, x)
            log_root_pot = b_root @ logpi[n]                    # (A,)
            log_node_pot = {}
            log_edge_op = {}
            for v in range(nn):
                if v == root:
                    continue
                w0 = e_edge[v][:, :, 0].diagonal()              # (L,) no-jump joint
                w1 = e_edge[v][:, :, 1].sum(0)                  # (L,) renewal child
                # geometric-mean edge operator: sum_theta w0(theta) logPsub
                log_edge_op[v] = np.einsum('l,lab->ab', w0, logPsub[v][n])
                # node potential from renewal: sum_theta w1(theta) logpi
                log_node_pot[v] = w1 @ logpi[n]                 # (A,)
            mu, xi, Hqn = residue_bp(n, log_root_pot, log_node_pot, log_edge_op)
            mu_all[n] = mu; xi_all[n] = xi; H_res += Hqn
        return mu_all, xi_all, H_res

    # -------------------------------------------------------- field block update
    def build_field_potentials(mu_all, xi_all):
        # root node potential
        lp = np.log(np.clip(rho, _FLOOR, None)).copy()
        for n in range(m):
            lp = lp + mu_all[n][root] @ logpi[n].T              # (L,) sum_x mu logpi
        logphi = {root: lp}
        logpot = {}
        for v in range(nn):
            if v == root:
                continue
            B = np.full((L, L, 2), -np.inf)
            # Delta=1 (renewal): logW + sum_n E_mu[logpi at child]
            e1 = np.zeros(L)
            for n in range(m):
                e1 = e1 + mu_all[n][v] @ logpi[n].T             # (L,) over theta_v
            with np.errstate(divide='ignore'):
                B[:, :, 1] = np.log(np.clip(Wm[v], _FLOOR, None)) + e1[None, :]
            # Delta=0 (no jump, theta_u=theta_v=th): logbeta + sum_n E_xi[logPsub]
            for th in range(L):
                e0 = np.log(max(beta[v][th], _FLOOR))
                for n in range(m):
                    e0 += float((xi_all[n][v] * logPsub[v][n][th]).sum())
                B[th, th, 0] = e0
            logpot[v] = B
        return logphi, logpot

    def field_prior_potentials():
        """Field MRF with no residue evidence (the warm-start field prior)."""
        logphi = {root: np.log(np.clip(rho, _FLOOR, None))}
        logpot = {}
        for v in range(nn):
            if v == root:
                continue
            B = np.full((L, L, 2), -np.inf)
            with np.errstate(divide='ignore'):
                B[:, :, 1] = np.log(np.clip(Wm[v], _FLOOR, None))
            for th in range(L):
                B[th, th, 0] = np.log(max(beta[v][th], _FLOOR))
            logpot[v] = B
        return logphi, logpot

    # ------------------------------------------------------------- warm start
    if warm_start:
        lphi0, lpot0 = field_prior_potentials()
        _, b0, e0 = field_logbp(tree, lphi0, lpot0)
        mu_all, xi_all, H_res = residue_update(b0, e0)
    else:
        # uniform residue marginals
        mu_all = [[np.full(A, 1.0 / A) for _ in range(nn)] for _ in range(m)]
        for n in range(m):
            for v in range(nn):
                if is_leaf[v]:
                    mu_all[n][v] = np.zeros(A)
                    ob = leaf_obs(v)[n]
                    if ob < 0:
                        mu_all[n][v] = np.full(A, 1.0 / A)
                    else:
                        mu_all[n][v][ob] = 1.0
        xi_all = [{v: np.full((A, A), 1.0 / (A * A)) for v in range(nn) if v != root}
                  for _ in range(m)]
        H_res = 0.0
        for n in range(m):
            for v in range(nn):
                if v == root:
                    continue
                mu_u = mu_all[n][int(tree.parent[v])]
                e = mu_u[:, None] * mu_all[n][v][None, :]
                xi_all[n][v] = e
            H_res += float(_entropy(mu_all[n][root]))
            for v in range(nn):
                if v == root:
                    continue
                e = xi_all[n][v]; mu_u = np.clip(mu_all[n][int(tree.parent[v])], _FLOOR, None)
                with np.errstate(divide='ignore', invalid='ignore'):
                    term = e * (np.log(np.clip(e, _FLOOR, None)) - np.log(mu_u)[:, None])
                H_res += -float(np.nansum(term))

    # -------------------------------------------------------- coordinate ascent
    trace = []
    warm_elbo = None
    prev = None
    logZ = 0.0; it = 0
    for it in range(n_iters):
        logphi, logpot = build_field_potentials(mu_all, xi_all)
        logZ, b, e = field_logbp(tree, logphi, logpot)
        elbo = float(logZ) + float(H_res)   # F(q_Theta^it, q_n^{it-1})
        trace.append(elbo)
        if warm_elbo is None:
            warm_elbo = elbo
        if prev is not None and abs(elbo - prev) < tol:
            break
        prev = elbo
        mu_all, xi_all, H_res = residue_update(b, e)

    if return_terms:
        return {"elbo": float(elbo), "logZ_field": float(logZ),
                "H_res": float(H_res), "iters": it + 1,
                "elbo_trace": trace, "warm_elbo": float(warm_elbo)}
    return float(elbo)
