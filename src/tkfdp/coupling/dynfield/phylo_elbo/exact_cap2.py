"""Exact cap-2 Felsenstein for the dynamic-field cluster likelihood.

Every TKF-DP cluster is size <= 2, so the compound field-augmented state
(theta, x_1, x_2) admits an EXACT Felsenstein that is cheaper than the naive
A^2-state (400-state) coupled chain. Conditional on theta the two columns
evolve independently, and the resample-at-jump rule (appendix eq:arch-Q-field)
makes the branch operator factor:

  m_c(theta') = <pi^{a1(theta')} kron pi^{a2(theta')}, M_c(theta')>     (stationary proj.)
  NJ(theta)   = beta_tau(theta) * P^{a1(theta)}(tau) M_c(theta) P^{a2(theta)}(tau)^T
  JMP(theta)  = sum_theta' J_tau(theta,theta') m_c(theta'),  J = P_theta - beta*I
  M_v(theta,x,y) = NJ(theta,x,y) + JMP(theta)                 (rank-1 jump add)

Combine at an internal node is the exact elementwise product M_L * M_R (no
moment-matching / mean-field gap). The message is (L, A, A) for a pair and
(L, A) for a singleton. Per-branch cost O(L*A^3) (two A x A matmuls per theta),
not O(A^4). This module is the numpy reference; a JAX-batched version drops
into the marginal-scorer (cluster,labeling)x(rate-bin) enumeration.

See appendix-tkfdp.tex Remark "Exact cap-2 Felsenstein".
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


# --------------------------------------------------------- generators / kernels

def gtr_Q(pi: np.ndarray, S: np.ndarray) -> np.ndarray:
    """GTR generator with exchangeability S and stationary pi: Q_xy = S_xy pi_y
    (x!=y), rows sum to 0. (eq:arch-Q-res form.)"""
    A = pi.shape[0]
    Q = S * pi[None, :]
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(A)] = -Q.sum(axis=1)
    return Q


def field_kernels(rho: np.ndarray, rho_chain: float, tau: float):
    """Return (P_theta, beta, J) for one branch length tau.
      P_theta[i,j] = g*delta + (1-g)*rho[j],  g = exp(-rho_chain*tau)
      beta[i]      = exp(-rho_chain*(1-rho[i])*tau)   (no-jump survival)
      J[i,j]       = P_theta[i,j] - beta[i]*delta_ij  (>=1 jump, end in j)
    """
    L = rho.shape[0]
    g = np.exp(-rho_chain * tau)
    P_theta = g * np.eye(L) + (1.0 - g) * rho[None, :]
    beta = np.exp(-rho_chain * (1.0 - rho) * tau)
    J = P_theta - np.diag(beta)
    return P_theta, beta, J


# --------------------------------------------------------- exact branch operator

def branch_pair(M_c: np.ndarray, tau: float, a1: np.ndarray, a2: np.ndarray,
                P_arch: 'dict[int, np.ndarray]', pi_arch: np.ndarray,
                rho: np.ndarray, rho_chain: float) -> np.ndarray:
    """Propagate a pair message M_c (L,A,A) up a branch of length tau.
    a1[theta], a2[theta] = archetype index for column 1/2 at field theta.
    P_arch[k] = expm(Q^k * tau) for archetype k (A,A). pi_arch (K_a, A)."""
    L, A, _ = M_c.shape
    _, beta, J = field_kernels(rho, rho_chain, tau)
    # stationary projection per theta': m_c(theta')
    m_c = np.array([pi_arch[a1[t]] @ M_c[t] @ pi_arch[a2[t]] for t in range(L)])
    # no-jump propagated part
    NJ = np.zeros_like(M_c)
    for t in range(L):
        P1 = P_arch[a1[t]]; P2 = P_arch[a2[t]]
        NJ[t] = beta[t] * (P1 @ M_c[t] @ P2.T)
    # jump scalar per theta
    JMP = J @ m_c                        # (L,)
    return NJ + JMP[:, None, None]


def branch_single(M_c: np.ndarray, tau: float, a1: np.ndarray,
                  P_arch: 'dict[int, np.ndarray]', pi_arch: np.ndarray,
                  rho: np.ndarray, rho_chain: float) -> np.ndarray:
    """Singleton message M_c (L,A) up a branch."""
    L, A = M_c.shape
    _, beta, J = field_kernels(rho, rho_chain, tau)
    m_c = np.array([pi_arch[a1[t]] @ M_c[t] for t in range(L)])
    NJ = np.array([beta[t] * (P_arch[a1[t]] @ M_c[t]) for t in range(L)])
    JMP = J @ m_c
    return NJ + JMP[:, None]


# --------------------------------------------------------- exact reference (cherry)

def _compound_generator_pair(a1: np.ndarray, a2: np.ndarray, pi_arch, S,
                             rho, rho_chain):
    """Full (L*A*A) compound generator on (theta, x, y): field jumps (resample
    both) + independent residue substitution. Reference / ground truth."""
    L, A = rho.shape[0], pi_arch.shape[1]
    N = L * A * A
    def idx(t, x, y):
        return (t * A + x) * A + y
    Q = np.zeros((N, N))
    Qk = {k: gtr_Q(pi_arch[k], S) for k in set(a1.tolist()) | set(a2.tolist())}
    for t in range(L):
        for x in range(A):
            for y in range(A):
                i = idx(t, x, y)
                # residue 1 substitution (field t fixed)
                for xp in range(A):
                    if xp != x:
                        Q[i, idx(t, xp, y)] += Qk[a1[t]][x, xp]
                # residue 2 substitution
                for yp in range(A):
                    if yp != y:
                        Q[i, idx(t, x, yp)] += Qk[a2[t]][y, yp]
                # field jump to t' != t: resample both from new archetypes
                for tp in range(L):
                    if tp != t:
                        rate = rho_chain * rho[tp]
                        for xp in range(A):
                            for yp in range(A):
                                Q[i, idx(tp, xp, yp)] += (
                                    rate * pi_arch[a1[tp]][xp] * pi_arch[a2[tp]][yp])
        # diagonal
    for i in range(N):
        Q[i, i] = -(Q[i].sum() - Q[i, i])
    return Q, idx


def cherry_ll_bruteforce(obs_L, obs_R, tauL, tauR, a1, a2, pi_arch, S,
                         rho, rho_chain) -> float:
    """Exact pair log-lik on a cherry (root LCA + 2 leaves) via expm of the
    full compound generator. Ground truth for validating the branch operator."""
    L, A = rho.shape[0], pi_arch.shape[1]
    Q, idx = _compound_generator_pair(a1, a2, pi_arch, S, rho, rho_chain)
    PL = expm(Q * tauL); PR = expm(Q * tauR)
    # root prior over (theta, x, y): rho[t] pi^{a1(t)}(x) pi^{a2(t)}(y)
    total = 0.0
    for t in range(L):
        for x in range(A):
            for y in range(A):
                pri = rho[t] * pi_arch[a1[t]][x] * pi_arch[a2[t]][y]
                if pri == 0:
                    continue
                i = idx(t, x, y)
                # leaf L observes obs_L=(xL,yL); marginalise child field
                pl = sum(PL[i, idx(tp, obs_L[0], obs_L[1])] for tp in range(L))
                pr = sum(PR[i, idx(tp, obs_R[0], obs_R[1])] for tp in range(L))
                total += pri * pl * pr
    return float(np.log(total))


def _postorder(root, children):
    order, stack = [], [root]
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(children.get(v, ()))
    return order[::-1]


def exact_pair_ll_tree(parent, tau, leaf_pair, root, a1, a2, pi_arch, S,
                       rho, rho_chain) -> float:
    """Exact pair log-lik on a general tree via the factored branch operator.
    parent (n,), tau (n,): rooted tree (parent[root] < 0). leaf_pair (n, 2)
    int: observed (x, y) at leaves, -1 for internal nodes or gaps."""
    from collections import defaultdict
    L, A = rho.shape[0], pi_arch.shape[1]
    n = len(parent)
    children = defaultdict(list)
    for v in range(n):
        if parent[v] >= 0:
            children[parent[v]].append(v)
    ks = set(a1.tolist()) | set(a2.tolist())
    Qk = {k: gtr_Q(pi_arch[k], S) for k in ks}
    P_cache: dict = {}

    def P_at(t):
        key = round(float(t), 12)
        if key not in P_cache:
            P_cache[key] = {k: expm(Qk[k] * t) for k in ks}
        return P_cache[key]

    def leaf_msg(x, y):
        M = np.zeros((L, A, A))
        xr = np.ones(A) if x < 0 else np.eye(A)[x]
        yr = np.ones(A) if y < 0 else np.eye(A)[y]
        M[:] = np.outer(xr, yr)[None]
        return M

    order = _postorder(root, children)
    # rescale to avoid underflow: track log-scale per node.
    M = {}; logscale = {}
    for v in order:
        if not children[v]:
            Mv = leaf_msg(int(leaf_pair[v, 0]), int(leaf_pair[v, 1]))
            lsc = 0.0
        else:
            Mv = np.ones((L, A, A)); lsc = 0.0
            for c in children[v]:
                prop = branch_pair(M[c], float(tau[c]), a1, a2,
                                   P_at(float(tau[c])), pi_arch, rho, rho_chain)
                Mv = Mv * prop
                lsc += logscale[c]
            mx = Mv.max()
            if mx > 0:
                Mv = Mv / mx; lsc += np.log(mx)
        M[v] = Mv; logscale[v] = lsc
    m_root = np.array([pi_arch[a1[t]] @ M[root][t] @ pi_arch[a2[t]]
                       for t in range(L)])
    return float(np.log(rho @ m_root) + logscale[root])


def cherry_ll_exact(obs_L, obs_R, tauL, tauR, a1, a2, pi_arch, S,
                    rho, rho_chain) -> float:
    """Exact pair log-lik on a cherry via the factored branch operator."""
    L, A = rho.shape[0], pi_arch.shape[1]
    ks = set(a1.tolist()) | set(a2.tolist())
    def leaf_msg(obs):
        M = np.zeros((L, A, A))
        M[:, obs[0], obs[1]] = 1.0
        return M
    def P_arch_at(tau):
        return {k: expm(gtr_Q(pi_arch[k], S) * tau) for k in ks}
    ML = branch_pair(leaf_msg(obs_L), tauL, a1, a2, P_arch_at(tauL),
                     pi_arch, rho, rho_chain)
    MR = branch_pair(leaf_msg(obs_R), tauR, a1, a2, P_arch_at(tauR),
                     pi_arch, rho, rho_chain)
    M_root = ML * MR                              # exact elementwise combine
    m_root = np.array([pi_arch[a1[t]] @ M_root[t] @ pi_arch[a2[t]]
                       for t in range(L)])
    return float(np.log(rho @ m_root))
