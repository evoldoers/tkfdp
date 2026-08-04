"""Certified variational ELBO for the dynfield moment-matching family.

Evaluates the ELBO *functional* L(q) at the M-projected node marginals -- a
genuine lower bound on log P(data), distinct from the moment-matching forward's
log Z_MM (an EP-style estimate that can fall on either side of the truth and is
NOT a bound; see appendix sec:variational-suppl, eq:mm-not-a-bound).

The variational q factorises over nodes (mean-field over the tree); at each node
it is the appendix family eq:varform

  q_v(theta,x) = q_v(theta) * [ (1-lam_v(theta)) prod_n p_n^v(x_n|theta)
                                + lam_v(theta) prod_n pi^{(c_n,theta)}(x_n) ].

The node parameters {q_v(theta), p_n^v(.|theta), lam_v(theta)} are the exact
M-projection of the forward x backward CLVs (variational.NodeState); they are
read out here from F_v (upward) and D_v (downward), reusing the same
<pi,A>=1 invariant the forward maintains.

ELBO terms (all from appendix sec:variational-suppl):

  L(q) = E_{q_root}[log rho(theta) + sum_n log pi(c_n,theta,x_n)]     (root prior)
       + sum_{edges u->v} E_{q_u,q_v}[log K(x_v,theta_v|x_u,theta_u)] (branches)
       + sum_v H(q_v)                                                (entropy)

Two terms need a bound to stay tractable AND remain a valid lower bound:

  * Edge term. K = sum_{Delta in {0,1}} P_field(theta_v,Delta|theta_u)
    prod_n P_sub^{(Delta)}(x_{v,n}|x_{u,n}); the log-sum over Delta recouples
    the sites. We apply Jensen with the *field-only* responsibility
    r(Delta|theta_u,theta_v) = P_field(theta_v,Delta|theta_u)/P_theta(theta_u->theta_v).
    This choice makes the field factor exact (it collapses to log P_theta) and
    leaves Jensen slack only in the substitution factors; it is a valid lower
    bound for any r that does not depend on (x_u,x_v).

  * Mixture entropy. H(T(.|theta)) >= (1-lam) sum_n H(p_n(.|theta))
    + lam sum_n H(pi^{(c_n,theta)}); lower-bounding H only lowers L.

Because both approximations are one-sided lower bounds, L_computed <= L_exact(q)
<= log P(data): the returned value is certified. It is NOT claimed to be tight;
at depth>1 the M-projection's lambda-overestimate and the field-only
responsibility both loosen it. On depth-1 cap-2 cherries the family contains the
exact posterior and the bound is exact up to the (there-tight) Jensen terms.

Leaves: an observed residue gives p_n = onehot(obs) and lam=0 (zero residue
entropy, zero substitution slack; the incoming edge uses the observed residue),
while the leaf field theta stays latent and its q_leaf(theta) entropy is counted.

SCOPE / KNOWN LOOSENESS. This is the mean-field-OVER-NODES bound: q = prod_v q_v
factorises theta ACROSS nodes. The field is exactly the across-tree-correlated
latent, so this family cannot contain the true posterior even on a single branch
or a cap-2 cherry -- its bound is valid but LOOSE, with a strictly positive gap
that grows with depth (test_field_factorization_gap). It is therefore a certified
FLOOR and a diagnostic, NOT the production object. The tight bound (whose maximum
equals log P on single-branch and cap-2 cherries, where the MM forward is exact)
needs a field-CORRELATED q: keep the L-state field exact via Felsenstein
(L^2/branch) and mean-field only the residues given the field -- the
tree-structured family of appendix eq:tree-branch-cond. That is the next step;
this module validates the term formulas and the "evaluate the functional, not
log Z_MM" discipline that carry over to it.
"""
from __future__ import annotations

import numpy as np

from .mm_clv import (
    MMClv, field_transition_row, beta_no_jump, per_class_field_Q, branch_P)
from .backward_sweep import forward_clvs, backward_theta_marginals
from .variational import NodeState, VariationalState
from .tree import Tree

_LOG_FLOOR = 1e-300


def _safe_log(x):
    return np.log(np.maximum(np.asarray(x, np.float64), _LOG_FLOOR))


def _entropy(p):
    p = np.asarray(p, np.float64)
    return float(-(p * _safe_log(p)).sum())


def extract_node_states(tree: Tree, classes: np.ndarray, rho: np.ndarray,
                        pi_field: np.ndarray, S: np.ndarray, rho_chain: float,
                        eta: float = 1.0, forward=None, down=None
                        ) -> VariationalState:
    """Read out the M-projected variational parameters {q_v, p_n^v, lam_v} at
    every node from the forward (F_v) and backward (D_v) CLVs.

    Uses the F*D 4-term split (see backward_sweep._joint_marginal_FxD):
      F*D = rF rD prod(A_F A_D)  [tracking]
          + rF sD prod A_F + sF rD prod A_D + sF sD  [>=1 regenerated]
    per field state theta. The tracking mass fraction gives 1-lam_v(theta); the
    tracking per-site marginal (normalised pi*A_F*A_D) gives p_n^v(.|theta).
    """
    if forward is None:
        forward = forward_clvs(tree, classes, rho, pi_field, S, rho_chain, eta=eta)
    L = int(rho.shape[0]); m = int(classes.shape[0]); A = int(pi_field.shape[2])
    # backward_theta_marginals returns q_v(theta); we also need D_v, so run the
    # sweep here and reuse its downward pass by asking for both.
    q_thetas, down = _forward_backward(tree, classes, rho, pi_field, S,
                                       rho_chain, eta, forward, down)
    states = []
    for v in range(tree.n_nodes):
        F, D = forward[v], down[v]
        if tree.is_leaf(v):
            # Observed residue: q_leaf(x|theta) = delta(obs) exactly, NOT a
            # variational mixture. lam=0, p=onehot(obs) (uniform for gaps) so the
            # residue contributes zero entropy and the incoming edge uses the
            # observed residue. Only the leaf field theta is latent.
            p = np.empty((m, L, A), np.float64)
            for n in range(m):
                x = int(tree.leaf_obs[v, n])
                if x >= 0:
                    p[n, :, :] = 0.0; p[n, :, x] = 1.0
                else:
                    p[n, :, :] = pi_field[int(classes[n]), :, :]
            st = NodeState(q=np.asarray(q_thetas[v], np.float64).copy(),
                           p=p, lam=np.zeros(L))
            st.check(); states.append(st); continue
        p = np.zeros((m, L, A), np.float64)
        lam = np.zeros(L, np.float64)
        for th in range(L):
            rF, sF = float(F.rho1[th]), float(F.s[th])
            rD, sD = float(D.rho1[th]), float(D.s[th])
            dot = np.empty(m)
            for n in range(m):
                pr = pi_field[int(classes[n]), th, :]
                fa = pr * F.A[n, th] * D.A[n, th]
                dot[n] = float(fa.sum())
                s = dot[n] if dot[n] > _LOG_FLOOR else 1.0
                p[n, th, :] = fa / s                      # tracking per-site marginal
            track = rF * rD * float(np.prod(dot))
            mass = track + rF * sD + sF * rD + sF * sD
            lam[th] = 0.0 if mass <= _LOG_FLOOR else (mass - track) / mass
            # guard: if tracking dot was ~0 (e.g. gap), fall back to pi so p is a
            # valid simplex (does not affect the lam-weighted energy materially).
            for n in range(m):
                if not np.isfinite(p[n, th]).all() or p[n, th].sum() <= _LOG_FLOOR:
                    p[n, th, :] = pi_field[int(classes[n]), th, :]
        st = NodeState(q=np.asarray(q_thetas[v], np.float64).copy(),
                       p=p, lam=np.clip(lam, 0.0, 1.0))
        st.check()
        states.append(st)
    return VariationalState(states=states)


def _forward_backward(tree, classes, rho, pi_field, S, rho_chain, eta,
                      forward, down):
    """Run the backward sweep, returning (q_thetas, down CLVs). Mirrors
    backward_theta_marginals but also hands back the downward CLV list."""
    from .mm_clv import mm_edge, mm_combine, mm_rescale
    from .backward_sweep import _joint_marginal_FxD
    Q = per_class_field_Q(pi_field, S)
    L = int(rho.shape[0]); n_nodes = tree.n_nodes; m = tree.m
    A = pi_field.shape[2]
    downl = [None] * n_nodes
    root = tree.root
    downl[root] = MMClv(rho1=np.ones(L), A=np.ones((m, L, A)),
                        s=np.zeros(L), log_scale=0.0)
    for v in tree.pre_order:
        v = int(v)
        if tree.is_leaf(v):
            continue
        parent_down = downl[v]
        for c in tree.children[v]:
            combined = parent_down
            for sib in tree.children[v]:
                if sib == c:
                    continue
                sm = mm_edge(forward[sib], classes, float(tree.branch_length[sib]),
                             rho, pi_field, Q, rho_chain, eta=eta)
                combined = mm_rescale(mm_combine(mm_rescale(sm), combined,
                                                 classes, pi_field))
            D_c = mm_rescale(mm_edge(combined, classes, float(tree.branch_length[c]),
                                     rho, pi_field, Q, rho_chain, eta=eta))
            downl[c] = D_c
    q_thetas = []
    for v in range(n_nodes):
        jm = _joint_marginal_FxD(forward[v], downl[v], classes, pi_field)
        joint = np.asarray(rho, np.float64) * jm
        tot = float(joint.sum())
        q_thetas.append(joint / tot if tot > 0 and np.isfinite(tot)
                        else np.full(L, 1.0 / L))
    return q_thetas, downl


def _mu(st: NodeState, pi_field, classes):
    """Per-site marginal residue distribution mu[n,theta,a] = (1-lam)p_n + lam pi."""
    m, L, A = st.m(), st.L(), st.A()
    mu = np.empty((m, L, A))
    for n in range(m):
        pr = pi_field[int(classes[n]), :, :]                  # (L,A)
        mu[n] = (1.0 - st.lam)[:, None] * st.p[n] + st.lam[:, None] * pr
    return mu


def elbo(tree: Tree, classes: np.ndarray, rho: np.ndarray, pi_field: np.ndarray,
         S: np.ndarray, rho_chain: float, eta: float = 1.0,
         return_terms: bool = False):
    """Certified variational lower bound on the tree log-likelihood.

    Returns a float (the ELBO). With return_terms=True returns a dict with the
    per-term breakdown {'elbo','root','edges','entropy'} for debugging.
    """
    rho = np.asarray(rho, np.float64)
    L = int(rho.shape[0]); m = int(classes.shape[0]); A = int(pi_field.shape[2])
    Q = per_class_field_Q(pi_field, S)
    forward = forward_clvs(tree, classes, rho, pi_field, S, rho_chain, eta=eta)
    vs = extract_node_states(tree, classes, rho, pi_field, S, rho_chain,
                             eta=eta, forward=forward)
    mu = [_mu(vs[v], pi_field, classes) for v in range(tree.n_nodes)]
    logpi = np.stack([_safe_log(pi_field[int(classes[n])]) for n in range(m)])  # (m,L,A)

    # --- root prior:  E_{q_root}[log rho(theta) + sum_n log pi(c_n,theta,x_n)] ---
    r = tree.root
    qr, lamr = vs[r].q, vs[r].lam
    root_term = float((qr * _safe_log(rho)).sum())
    for th in range(L):
        for n in range(m):
            e = ((1.0 - lamr[th]) * float((vs[r].p[n, th] * logpi[n, th]).sum())
                 + lamr[th] * float((pi_field[int(classes[n]), th] * logpi[n, th]).sum()))
            root_term += float(qr[th]) * e

    # --- branches:  sum_{u->v} E_{q_u,q_v}[log K], Jensen on Delta -------------
    edge_term = 0.0
    for v in range(tree.n_nodes):
        if v == tree.root:
            continue
        u = int(tree.parent[v]); tau = float(tree.branch_length[v])
        qu, qv = vs[u].q, vs[v].q
        # field transition P_theta(u->v) and no-jump survival beta(theta_u)
        Pth = np.stack([field_transition_row(rho, tu, tau, rho_chain)
                        for tu in range(L)])                  # (L,L): Pth[tu,tv]
        beta = np.array([beta_no_jump(rho, tu, tau, rho_chain) for tu in range(L)])
        # Delta=1 substitution E-log:  E_{q_v}[log pi(c_n,theta_v,x_v)]  (renewal)
        Elog1 = np.array([[float((mu[v][n, tv] * logpi[n, tv]).sum())
                           for n in range(m)] for tv in range(L)])   # (L,m)
        for tu in range(L):
            for tv in range(L):
                Ptot = float(Pth[tu, tv])
                if Ptot <= _LOG_FLOOR:
                    continue
                w = float(qu[tu] * qv[tv])
                if w <= 0.0:
                    continue
                r0 = (beta[tu] / Ptot) if tu == tv else 0.0
                r0 = min(max(r0, 0.0), 1.0)
                r1 = 1.0 - r0
                sub = 0.0
                if r0 > 0.0:                                   # Delta=0: GTR, tu==tv
                    for n in range(m):
                        lP = _safe_log(branch_P(Q, int(classes[n]), tu, tau, eta))
                        # E_{q_u,q_v}[log P_sub] = sum_{a,b} mu_u[a] lP[a,b] mu_v[b]
                        sub += r0 * float(mu[u][n, tu] @ (lP @ mu[v][n, tv]))
                sub += r1 * float(Elog1[tv].sum())
                edge_term += w * (np.log(Ptot) + sub)

    # --- entropy:  sum_v H(q_v(theta)) + sum_theta q(theta) H(T(.|theta)) ------
    ent = 0.0
    for v in range(tree.n_nodes):
        st = vs[v]
        ent += _entropy(st.q)
        for th in range(L):
            Ht = ((1.0 - st.lam[th]) * sum(_entropy(st.p[n, th]) for n in range(m))
                  + st.lam[th] * sum(_entropy(pi_field[int(classes[n]), th])
                                     for n in range(m)))
            ent += float(st.q[th]) * Ht

    total = root_term + edge_term + ent
    if return_terms:
        return {"elbo": total, "root": root_term, "edges": edge_term,
                "entropy": ent}
    return total
