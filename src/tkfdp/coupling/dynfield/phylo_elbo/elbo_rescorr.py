"""Residue-correlated tree-structured ELBO (appendix eq:res-corr-branch).

The certified bound is the ELBO *functional* L(q) = E_q[log P] + H(q) evaluated
at an explicitly-constructed genuine residue-correlated q -- NOT the moment-
matching forward's log Z_MM (which is not a bound). The q is the rooted tree-
Markov law

    q_root(theta,x)      proportional to  prior(theta,x) * F_root(theta,x)
    q(z_v | z_u)         proportional to  K(z_v | z_u)   * F_v(z_v)

where z = (theta, x), F_v is the mm_clv upward message (rank-1+scalar, i.e. the
residue-correlated Delta-mixture: r*prod A = tracking, s = renewal), and K is the
compound branch factor. Because F is exact wherever no combine-projection fires,
q is the exact posterior for m=1 (any depth) and for cherries (any m), so
L(q) = log P there; elsewhere q is a genuine (approximate) distribution, so
L(q) <= log P is a certified bound, and -- keeping the parent->child residue
correlation through K and the downward evidence through F -- strictly tighter
than the residue-mean-field floor (elbo.py).

This module provides the brute-force REFERENCE (`elbo_rescorr_bruteforce`) that
enumerates q on small trees and evaluates the functional exactly -- correct by
construction, used to validate the tractable per-site version and the bound.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.linalg import expm

from .backward_sweep import forward_clvs

_FLOOR = 1e-300


def _sub_P(pi_field, S, c, theta, tau):
    A = pi_field.shape[2]
    Q = np.zeros((A, A))
    for a in range(A):
        for b in range(A):
            if a != b:
                Q[a, b] = S[a, b] * pi_field[c, theta, b]
        Q[a, a] = -Q[a, :].sum()
    return expm(Q * tau)


def branch_factor(xu, thu, xv, thv, classes, tau, pi_field, S, rho, rho_chain):
    """Compound branch factor K(x_v_vec, th_v | x_u_vec, th_u), summed over the
    field-jump indicator Delta (Delta=0 no-jump substitution; Delta=1 renewal)."""
    m = len(classes)
    beta = np.exp(-rho_chain * (1 - rho[thu]) * tau)
    et = np.exp(-rho_chain * tau)
    P_theta = (et if thu == thv else 0.0) + (1 - et) * rho[thv]
    W = P_theta - (beta if thv == thu else 0.0)
    val = W
    for n in range(m):
        val *= pi_field[int(classes[n]), thv, xv[n]]
    if thv == thu:
        nj = beta
        for n in range(m):
            nj *= _sub_P(pi_field, S, int(classes[n]), thu, tau)[xu[n], xv[n]]
        val += nj
    return val


def _F_eval(F, theta, xvec):
    """Evaluate the mm_clv message F(theta, x) = r[theta] prod_n A[n,theta,x_n]
    + s[theta] at a residue vector."""
    v = float(F.rho1[theta])
    for n, a in enumerate(xvec):
        v *= float(F.A[n, theta, a])
    return v + float(F.s[theta])


def elbo_rescorr_bruteforce(tree, classes, rho, pi_field, S, rho_chain,
                            eta=1.0, return_terms=False):
    """Brute-force reference: build the residue-correlated q explicitly and
    evaluate L(q) = E_q[log P] + H(q) by enumeration. Small trees only
    (enumerates theta over all nodes and x over internal nodes)."""
    rho = np.asarray(rho, np.float64)
    L = int(rho.shape[0]); A = int(pi_field.shape[2]); m = int(classes.shape[0])
    n_nodes = tree.n_nodes; root = tree.root
    is_leaf = [tree.is_leaf(v) for v in range(n_nodes)]
    internal = [v for v in range(n_nodes) if not is_leaf[v]]
    fwd = forward_clvs(tree, classes, rho, pi_field, S, rho_chain, eta=eta)

    def prior(thr, xr):
        p = rho[thr]
        for n in range(m):
            p *= pi_field[int(classes[n]), thr, xr[n]]
        return p

    xspace = list(itertools.product(range(A), repeat=m))
    tau_of = {v: float(tree.branch_length[v]) for v in range(n_nodes) if v != root}
    par = {v: int(tree.parent[v]) for v in range(n_nodes) if v != root}

    def child_states(v):
        if is_leaf[v]:
            return [tuple(int(tree.leaf_obs[v, n]) for n in range(m))]
        return xspace

    def N_edge(v, thu, xu):
        """Child->parent message m_{v->u}(z_u) = sum_{z_v} K(z_v|z_u) F_v(z_v)."""
        tot = 0.0
        for thv in range(L):
            for xv in child_states(v):
                tot += branch_factor(xu, thu, xv, thv, classes, tau_of[v],
                                     pi_field, S, rho, rho_chain) \
                    * _F_eval(fwd[v], thv, xv)
        return tot

    # Rooted tree-Markov q = q_root * prod_edges q(z_v|z_u), the exact BP posterior
    #   q_root(z_r)      propto prior(z_r) * F_root_exact(z_r)
    #   q(z_v|z_u)       propto K(z_v|z_u) * F_v(z_v)
    # with F_root_exact = prod_{children c} N_edge(c, .)  -- the root combine kept
    # UNPROJECTED (root residue integrated exactly): single-branch/cherry exact at
    # any m; F exact for m=1 => exact at any depth. Non-root internal F_v are the
    # projected mm_clv messages => a certified bound for m>=2 at depth>=2.
    root_children = [c for c in tree.children[root]]

    def logF_root_exact(thr, xr):
        s = 0.0
        for c in root_children:
            s += np.log(max(N_edge(c, thr, xr), _FLOOR))
        return s

    Z_root = sum(prior(thr, xr) * np.exp(logF_root_exact(thr, xr))
                 for thr in range(L) for xr in xspace)
    logZ_root = np.log(max(Z_root, _FLOOR))
    Ncache = {}

    E_logP = 0.0; E_logq = 0.0; Zq = 0.0
    for th in itertools.product(range(L), repeat=n_nodes):
        for xint in itertools.product(xspace, repeat=len(internal)):
            xass = {}
            for i, v in enumerate(internal):
                xass[v] = xint[i]
            for v in range(n_nodes):
                if is_leaf[v]:
                    xass[v] = tuple(int(tree.leaf_obs[v, n]) for n in range(m))
            lP = np.log(max(prior(th[root], xass[root]), _FLOOR))
            lq = np.log(max(prior(th[root], xass[root]), _FLOOR)) \
                + logF_root_exact(th[root], xass[root]) - logZ_root
            for v in range(n_nodes):
                if v == root:
                    continue
                u = par[v]
                lk = np.log(max(branch_factor(xass[u], th[u], xass[v], th[v],
                                classes, tau_of[v], pi_field, S, rho, rho_chain),
                                _FLOOR))
                lP += lk
                key = (v, th[u], xass[u])
                if key not in Ncache:
                    Ncache[key] = np.log(max(N_edge(v, th[u], xass[u]), _FLOOR))
                lq += lk + np.log(max(_F_eval(fwd[v], th[v], xass[v]), _FLOOR)) \
                    - Ncache[key]
            q = np.exp(lq)
            Zq += q; E_logP += q * lP; E_logq += q * lq
    elbo = E_logP - E_logq
    if return_terms:
        return {"elbo": float(elbo), "Zq": float(Zq),
                "E_logP": float(E_logP), "neg_E_logq": float(-E_logq)}
    return float(elbo)


def _logsumexp(vals):
    a = np.asarray(vals, np.float64)
    mx = a.max()
    return float(mx + np.log(np.exp(a - mx).sum()))
