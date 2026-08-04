"""Tractable tree-structured (residue-correlated) ELBO -- the certified bound.

Computes L(q) = E_q[log P] + H(q) for the GENUINE rooted residue-correlated q
(appendix eq:res-corr-branch): q_root proportional to prior*F_root (root combine
unprojected), branch conditionals q(z_v|z_u) proportional to K(z_v|z_u)*F_v(z_v),
with F_v the mm_clv rank-1+scalar messages (the (theta,x)-joint residue-correlated
form). NOT the moment-matching forward's log Z_MM (not a bound) and NOT a
mean-field surrogate (which decouples the covarion field<->residue coupling).

The node marginals q_v(theta,x) are A^m-dimensional PER NODE (not exponential in
tree size), so a rooted downward sweep with per-node A^m tables computes L(q)
EXACTLY in O(N * A^{2m}) -- m-general (works for any cluster width; no cap-2
assumption / no exact_cap2), cheap for the cap-2/cap-3 clusters actually used.

Because q is the exact posterior wherever no combine-projection fires, L(q) equals
log P for cherries at any m, for all trees at m=1, and at field-rate 0; elsewhere
L(q) <= log P is a certified bound strictly tighter than the mean-field floor.
This matches elbo_rescorr.elbo_rescorr_bruteforce (the enumerated reference) to
machine precision, but without enumerating the whole tree.
"""
from __future__ import annotations

import itertools
import numpy as np

from .backward_sweep import forward_clvs
from .elbo_rescorr import branch_factor, _F_eval

_FLOOR = 1e-300


def _slog(x):
    return np.log(max(float(x), _FLOOR))


def elbo_treestruct(tree, classes, rho, pi_field, S, rho_chain, eta=1.0,
                    return_terms=False):
    rho = np.asarray(rho, np.float64)
    L = int(rho.shape[0]); A = int(pi_field.shape[2]); m = int(classes.shape[0])
    nn = tree.n_nodes; root = tree.root
    is_leaf = [tree.is_leaf(v) for v in range(nn)]
    fwd = forward_clvs(tree, classes, rho, pi_field, S, rho_chain, eta=eta)
    xsp = list(itertools.product(range(A), repeat=m))
    tau = {v: float(tree.branch_length[v]) for v in range(nn) if v != root}

    def prior(th, x):
        p = rho[th]
        for n in range(m):
            p *= pi_field[int(classes[n]), th, x[n]]
        return p

    def states(v):
        if is_leaf[v]:
            obs = tuple(int(tree.leaf_obs[v, n]) for n in range(m))
            return [(th, obs) for th in range(L)]
        return [(th, x) for th in range(L) for x in xsp]

    def K(zu, zv, edge_v):
        (thu, xu), (thv, xv) = zu, zv
        return branch_factor(list(xu), thu, list(xv), thv, classes, tau[edge_v],
                             pi_field, S, rho, rho_chain)

    # N_v(z_u) = sum_{z_v} K(z_v|z_u) F_v(z_v)  (child->parent message), cached.
    Ncache = {}

    def Nedge(v, zu):
        key = (v, zu)
        if key not in Ncache:
            s = 0.0
            for zv in states(v):
                s += K(zu, zv, v) * _F_eval(fwd[v], zv[0], zv[1])
            Ncache[key] = s
        return Ncache[key]

    # ---- downward sweep: exact node marginals q_v(z_v) as {z: prob} -----------
    q = [None] * nn
    # root: q_root propto prior(z_r) * F_root_exact(z_r), F_root_exact = prod_c N_c
    root_children = [int(c) for c in tree.children[root]]
    qr = {}
    for zr in states(root):
        f = prior(zr[0], zr[1])
        for c in root_children:
            f *= Nedge(c, zr)
        qr[zr] = f
    Zr = sum(qr.values())
    q[root] = {z: p / Zr for z, p in qr.items()}
    for v in tree.pre_order:
        v = int(v)
        for c in tree.children[v]:
            c = int(c)
            qc = {}
            for zc in states(c):
                s = 0.0
                Fc = _F_eval(fwd[c], zc[0], zc[1])
                for zu, pu in q[v].items():
                    nv = Nedge(c, zu)
                    if nv > _FLOOR:
                        s += pu * K(zu, zc, c) / nv
                qc[zc] = Fc * s
            Z = sum(qc.values())
            q[c] = {z: (p / Z if Z > _FLOOR else 0.0) for z, p in qc.items()}

    # ---- L(q) = E_q[log P] + H(q), rooted decomposition ----------------------
    # E_q[log P] = E_{q_root}[log prior] + sum_edges E_{q_uv}[log K]
    # H(q)       = H(q_root) + sum_edges E_{q_u}[ H(q(z_v|z_u)) ]
    E_logP = 0.0
    H = 0.0
    for zr, p in q[root].items():
        if p > 0:
            E_logP += p * _slog(prior(zr[0], zr[1]))
            H -= p * _slog(p)
    for v in range(nn):
        if v == root:
            continue
        u = int(tree.parent[v])
        for zu, pu in q[u].items():
            if pu <= 0:
                continue
            nv = Nedge(v, zu)
            # conditional q(z_v|z_u) = K F_v / N_v
            ec = 0.0  # E_{z_v~cond}[log K]
            hc = 0.0  # H(cond)
            for zv in states(v):
                kf = K(zu, zv, v)
                w = kf * _F_eval(fwd[v], zv[0], zv[1]) / nv if nv > _FLOOR else 0.0
                if w > 0:
                    ec += w * _slog(kf)
                    hc -= w * _slog(w)
            E_logP += pu * ec
            H += pu * hc

    elbo = E_logP + H
    if return_terms:
        return {"elbo": float(elbo), "E_logP": float(E_logP), "H": float(H),
                "Zq_root": float(sum(q[root].values()))}
    return float(elbo)
