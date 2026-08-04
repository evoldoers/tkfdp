"""Per-site O(A^2) tree-structured ELBO: field tree-Markov + per-site
branch-correlated residues (appendix eq:res-corr-branch, factored form).

The variational family is
    q = q_Theta(theta, Delta)                      [field-and-jumps, tree-Markov]
        x  prod_n q_res,n(x_n | theta, Delta)      [per-site residue chain]
with, per site n and edge u->v, a per-theta tracking kernel (Delta=0) carrying
the along-branch residue correlation, and renewal from the field stationary
(Delta=1). Everything factorises over sites given (theta,Delta), so the ELBO is
O(N (L^2 + m L A^2)) -- no A^m. It is a genuine lower bound (looser than the
exact joint elbo_treestruct, tighter than the node-marginal field-exact form).

Computed by coordinate ascent:
  (ii) residues fixed  -> field-jump tree MRF potentials -> exact field BP
       (field_bp) -> q_v(theta), q_uv(theta_u,theta_v,Delta), log Z_field.
  (i)  field fixed     -> per-site (theta,x) forward-backward under the field
       posterior edge marginals -> residue node marginals mu and edge pairwise.
ELBO L = log Z_field + residue entropy (junction-tree, per site), which equals
E_q[log P] + H(q) for this factored q.
"""
from __future__ import annotations

import numpy as np

from .mm_clv import field_transition_row, beta_no_jump, per_class_field_Q, branch_P
from .field_bp import field_logbp, _logsumexp

_FLOOR = 1e-300


def _entropy_rows(p, axis=-1):
    p = np.asarray(p, np.float64)
    return -(p * np.log(np.clip(p, _FLOOR, None))).sum(axis=axis)


def elbo_persite(tree, classes, rho, pi_field, S, rho_chain, eta=1.0,
                 n_iters=40, tol=1e-10, return_terms=False):
    rho = np.asarray(rho, np.float64)
    L = int(rho.shape[0]); A = int(pi_field.shape[2]); m = int(classes.shape[0])
    nn = tree.n_nodes; root = tree.root
    is_leaf = [tree.is_leaf(v) for v in range(nn)]
    Q = per_class_field_Q(pi_field, S)
    logpi = np.stack([np.log(np.clip(pi_field[int(classes[n])], _FLOOR, None))
                      for n in range(m)])                     # (m,L,A)
    cls = [int(classes[n]) for n in range(m)]

    # per-edge kernels: field transition P_theta(u->v), beta, W; substitution
    # P_sub[v][n] : (L,A,A) = branch_P for (class n, theta, tau_v)
    Pth = {}; beta = {}; logPsub = {}
    for v in range(nn):
        if v == root:
            continue
        tau = float(tree.branch_length[v])
        Pth[v] = np.stack([field_transition_row(rho, tu, tau, rho_chain)
                           for tu in range(L)])                # (L,L)
        beta[v] = np.array([beta_no_jump(rho, tu, tau, rho_chain) for tu in range(L)])
        logPsub[v] = np.stack([np.log(np.clip(branch_P(Q, cls[n], th, tau, eta),
                                               _FLOOR, None))
                               for n in range(m) for th in range(L)]
                              ).reshape(m, L, A, A)             # (m,L,A,A)

    def leaf_obs(v):
        return [int(tree.leaf_obs[v, n]) for n in range(m)]

    # residue node marginals mu[v] : (m, L, A); leaves are delta(obs) for all theta
    mu = [np.full((m, L, A), 1.0 / A) for _ in range(nn)]
    for v in range(nn):
        if is_leaf[v]:
            ob = leaf_obs(v)
            mu[v] = np.zeros((m, L, A))
            for n in range(m):
                mu[v][n, :, ob[n]] = 1.0
    # variant 2 residue beliefs: node marginals mu[v] (m,L,A), tracking kernels
    # Tk[v][n] (L,A,A) [theta,x_parent,x_child] for Delta=0, renewal marginals
    # rr[v][n] (L,A) for Delta=1. Init tracking to Psub, renewal to pi.
    Tk = {v: np.exp(logPsub[v]).copy() for v in range(nn) if v != root}
    rr = {v: np.stack([np.exp(logpi[n]) for n in range(m)]) for v in range(nn) if v != root}
    for v in range(nn):
        if v == root or not is_leaf[v]:
            continue
        ob = leaf_obs(v)                                    # leaf: x_v = obs deterministically
        for n in range(m):
            Tk[v][n] = 0.0; Tk[v][n][:, :, ob[n]] = 1.0
            rr[v][n] = 0.0; rr[v][n][:, ob[n]] = 1.0

    def build_field_potentials():
        logphi = {root: np.log(np.clip(rho, _FLOOR, None)) +
                  sum(np.einsum('la,la->l', mu[root][n], logpi[n]) for n in range(m))}
        logpot = {}
        for v in range(nn):
            if v == root:
                continue
            u = int(tree.parent[v])
            B = np.full((L, L, 2), -np.inf)
            # Delta=1 (renewal): W(theta_u,theta_v) * exp(sum_n E_rr[log pi])
            W = Pth[v] - np.eye(L) * beta[v][:, None]           # (L,L)
            e1 = np.array([sum(float((rr[v][n, tv] * logpi[n, tv]).sum())
                               for n in range(m)) for tv in range(L)])   # (L,)
            with np.errstate(divide='ignore'):
                B[:, :, 1] = np.log(np.clip(W, _FLOOR, None)) + e1[None, :]
            # Delta=0 (tracking): correlated pairwise mu_u(x_u) * Tk(x_v|x_u)
            for th in range(L):
                e0 = 0.0
                for n in range(m):
                    pair = mu[u][n, th][:, None] * Tk[v][n, th]           # (A,A) x_u,x_v
                    e0 += float((pair * logPsub[v][n, th]).sum())
                B[th, th, 0] = np.log(max(beta[v][th], _FLOOR)) + e0
            logpot[v] = B
        return logphi, logpot

    def residue_update(q_node, q_edge):
        """Per-site (theta,x) forward-backward under q_edge; returns node marginals
        mu, tracking kernels Tk (posterior transition), renewal marginals rr."""
        newmu = [mu[v].copy() for v in range(nn)]
        newTk = {v: Tk[v].copy() for v in Tk}
        newrr = {v: rr[v].copy() for v in rr}
        for n in range(m):
            # upward (theta,x) messages g[v] : (L,A)
            g = [np.zeros((L, A)) for _ in range(nn)]
            for v in tree.post_order:
                v = int(v)
                if is_leaf[v]:
                    gg = np.zeros((L, A)); gg[:, leaf_obs(v)[n]] = 1.0; g[v] = gg
                    continue
                acc = np.ones((L, A))
                for c in tree.children[v]:
                    c = int(c)
                    qc = q_edge[c]                              # (L,L,2)[thv,thc,D]
                    qv = np.clip(q_node[v], _FLOOR, None)
                    r = qc / qv[:, None, None]                  # cond thv->(thc,D)
                    # child contribution to (theta_v, x_v):
                    #  D=0: theta_c=theta_v, x_c ~ Psub(.|x_v); D=1: renewal x_c~pi
                    Psub = np.exp(logPsub[c][n])               # (L,A,A) th,xpar,xchild
                    m0 = np.einsum('lab,lb->la', Psub, g[c])   # (L,A) tracking, th=thv
                    contrib = r[:, :, 0].diagonal()[:, None] * m0  # thv, x_v
                    pic = np.exp(logpi[n])                      # (L,A)
                    renew = np.einsum('vc,ca->v', r[:, :, 1], (pic * g[c]))  # (L,) over thv
                    contrib = contrib + renew[:, None]
                    acc = acc * np.clip(contrib, _FLOOR, None)
                g[v] = acc
            # downward messages h[v] : (L,A) (evidence from outside subtree)
            h = [np.ones((L, A)) for _ in range(nn)]
            pri = np.exp(logpi[n])                              # (L,A) root prior
            h[root] = pri * rho[:, None]
            for v in tree.pre_order:
                v = int(v)
                if is_leaf[v]:
                    continue
                for c in tree.children[v]:
                    c = int(c)
                    # cavity at v excluding c
                    cav = h[v].copy()
                    for c2 in tree.children[v]:
                        if int(c2) == c:
                            continue
                        qc2 = q_edge[int(c2)]; qv = np.clip(q_node[v], _FLOOR, None)
                        r2 = qc2 / qv[:, None, None]
                        Psub2 = np.exp(logPsub[int(c2)][n])
                        m0 = np.einsum('lab,lb->la', Psub2, g[int(c2)])
                        pic = np.exp(logpi[n])
                        renew = np.einsum('vc,ca->v', r2[:, :, 1], (pic * g[int(c2)]))
                        contrib = r2[:, :, 0].diagonal()[:, None] * m0 + renew[:, None]
                        cav = cav * np.clip(contrib, _FLOOR, None)
                    # propagate cavity(theta_v,x_v) down to (theta_c,x_c)
                    qc = q_edge[c]; qv = np.clip(q_node[v], _FLOOR, None)
                    r = qc / qv[:, None, None]                  # thv->(thc,D)
                    Psub = np.exp(logPsub[c][n])               # (L,A,A) th,xpar,xchild
                    # tracking: theta_c=theta_v, x_c from x_v via Psub^T
                    track = r[:, :, 0].diagonal()[:, None] * cav  # (L=thv, x_v)
                    down_track = np.einsum('vb,vba->va', track, Psub)  # (thc=thv, x_c)
                    # renewal: x_c ~ pi(theta_c), weight sum_thv cav . 1
                    wv = cav.sum(1)                             # (L,) over thv
                    down_renew = np.einsum('vc,v->c', r[:, :, 1], wv)[:, None] * np.exp(logpi[n])
                    h[c] = np.clip(down_track + down_renew, _FLOOR, None)
            # node marginals (theta,x) = g*h; tracking kernel Tk(x_v|x_u,th) propto
            # Psub(x_v|x_u;th) g_v(th,x_v); renewal rr(x|th) propto pi(x|th) g_v(th,x).
            pic = np.exp(logpi[n])                              # (L,A)
            for v in range(nn):
                if not is_leaf[v]:
                    b = g[v] * h[v]
                    Z = b.sum(1, keepdims=True)
                    newmu[v][n] = np.where(Z > _FLOOR, b / np.clip(Z, _FLOOR, None), 1.0 / A)
                if v == root:
                    continue
                Psub = np.exp(logPsub[v][n])                    # (L,A,A) th,x_u,x_v
                num = Psub * g[v][:, None, :]                   # (L,A,A)
                Zt = num.sum(2, keepdims=True)
                newTk[v][n] = np.where(Zt > _FLOOR, num / np.clip(Zt, _FLOOR, None), 1.0 / A)
                num2 = pic * g[v]                               # (L,A)
                Zr = num2.sum(1, keepdims=True)
                newrr[v][n] = np.where(Zr > _FLOOR, num2 / np.clip(Zr, _FLOOR, None), 1.0 / A)
        return newmu, newTk, newrr

    def residue_entropy(q_node, q_edge):
        """Rooted conditional (junction-tree) entropy of q_res: H(mu_root) +
        sum_edges E_{q_edge}[ H(x_v | x_u, theta, Delta) ], per site."""
        H = 0.0
        for n in range(m):
            H += float((q_node[root] * _entropy_rows(mu[root][n])).sum())
            for v in range(nn):
                if v == root:
                    continue
                u = int(tree.parent[v]); qe = q_edge[v]        # (L,L,2)
                for th in range(L):                            # Delta=0: theta_u=theta_v=th
                    w0 = float(qe[th, th, 0])
                    if w0 > 0.0:
                        Hrows = _entropy_rows(Tk[v][n][th])    # (A,) H per x_u
                        H += w0 * float((mu[u][n, th] * Hrows).sum())
                w1 = qe[:, :, 1].sum(0)                         # (L,) over theta_v
                for tv in range(L):                            # Delta=1
                    if w1[tv] > 0.0:
                        H += float(w1[tv]) * float(_entropy_rows(rr[v][n][tv]))
        return H

    prev = None
    logZ = 0.0; q_node = None; q_edge = None; H_res = 0.0
    for it in range(n_iters):
        logphi, logpot = build_field_potentials()
        logZ, q_node, q_edge = field_logbp(tree, logphi, logpot)
        H_res = residue_entropy(q_node, q_edge)
        elbo = logZ + H_res
        if prev is not None and abs(elbo - prev) < tol:
            break
        prev = elbo
        mu, Tk, rr = residue_update(q_node, q_edge)

    if return_terms:
        return {"elbo": float(elbo), "logZ_field": float(logZ), "H_res": float(H_res),
                "iters": it + 1}
    return float(elbo)
