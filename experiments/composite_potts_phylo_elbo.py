#!/usr/bin/env python3
r"""Composite-likelihood phylogenetic ELBO for multi-site Potts coupling under a
Metropolis--Hastings (Hastings ``min'') substitution form.

This implements the derivation added as appendix subsection
``sec:composite-mh-elbo`` of ``math-paper/appendix-tkfdp.tex``.  The
composite (pairwise) likelihood breaks the multi-site Potts objective into a
PRODUCT OVER COUPLED PAIRS of independent two-site phylogenetic trees; the
Potts coupling enters ONLY through the pair stationary

    pi_ij(a,b)  =  pi_i(a) pi_j(b) exp(-H(a,b)) / Z_ij            (eq:comp-pair-stat)

and the substitution generator is the single-transition Metropolis--Hastings
form (eq:comp-mh-imove / eq:comp-mh-jmove)

    Q[(a,b),(a',b)] = S(a,a') min(1, pi_ij(a',b)/pi_ij(a,b))      (site-i move)
    Q[(a,b),(a,b')] = S(b,b') min(1, pi_ij(a,b')/pi_ij(a,b))      (site-j move)
    Q[(a,b),(a',b')] = 0   (a'!=a and b'!=b: no simultaneous double jump)

which is reversible w.r.t. pi_ij for ANY symmetric exchangeability S and ANY
coupling H (Proposition ``prop:comp-mh''), with symmetric min-flux
F_xy = S_xy min(pi_ij(x), pi_ij(y)) and the spectator stationary cancelling in
the acceptance ratio.

Pipeline (pure JAX, vectorised/mapped, logspace, f64 for eig/HR):
  * ``mh_pair_generator``    : build the 400-state M-H pair generator from
                               (pi_i, pi_j, H, S) -- vmapped over pairs;
  * ``composite_forward``    : per-pair reversible-CTMC Felsenstein forward over
                               a PADDED postorder tree (lax.scan, per-node log
                               scaling) -> the composite log-likelihood
                               (eq:comp-forward), vmapped over pairs;
  * ``composite_estep``      : forward+outside pass for endpoint-conditioned edge
                               posteriors, then the exact Holmes--Rubin bridge
                               (reversible eig + divided-difference kernel) for
                               the expected transition usage N and dwell T
                               (eq:comp-hr), summed over branches and pairs;
  * ``mstep_S``              : closed-form shared-exchangeability M-step
                               S(a,a') = usage / exposure (eq:comp-mstep-S), the
                               exact maximiser -> monotone;
  * ``mstep_pi_ij``          : damped log-domain ascent on the pair stationary
                               (pi/H block), the nonlinear-min M-step.

Self-checks (``--selfcheck``):
  (a) the JAX composite forward log-lik matches an INDEPENDENT dense brute-force
      pair-tree Felsenstein (scipy.linalg.expm per branch) to ~1e-6;
      plus reversibility (pi_ij[x] Q[x,y] symmetric) and stationarity
      (pi_ij @ Q ~ 0) of the generator, and a micro-check that the JAX HR bridge
      equals the audited numpy ``tkfdp.permfield.hr.bridge`` to ~1e-10;
  (b) coordinate ascent on a synthetic 2-site coupled cluster is MONOTONE:
      S-only EM (pi_ij fixed) is strictly non-decreasing (exact M-step), and the
      joint (S + pi_ij/H) coordinate ascent is non-decreasing.

Run:
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=4 \
      PYTHONPATH=src:/home/yam/tkf-mixdom/python \
      python3 experiments/composite_potts_phylo_elbo.py --selfcheck
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial

sys.path.insert(0, "src")
from tkfdp.lg08 import S_LG08, PI_LG08  # noqa: E402


# ============================================================================
# linear algebra: reversible eig, matrix exponential, HR divided-difference
# (mirrors tkfdp.permfield.hr and experiments/fit_single_swap_field_jax.py)
# ============================================================================
def eig_rev_jax(Q, pi):
    """Symmetric eigendecomposition of a reversible generator; Q = U diag(lam) Uinv."""
    d = jnp.sqrt(jnp.clip(pi, 1e-300, None))
    di = 1.0 / d
    Qs = d[:, None] * Q * di[None, :]
    Qs = 0.5 * (Qs + Qs.T)
    lam, V = jnp.linalg.eigh(Qs)
    U = di[:, None] * V
    Uinv = V.T * d[None, :]
    return lam, U, Uinv


def expm_from_eig(lam, U, Uinv, t):
    return (U * jnp.exp(lam * t)[None, :]) @ Uinv


def Jmat_jax(lam, t):
    """Divided-difference kernel J_kl = (e^{lam_k t}-e^{lam_l t})/(lam_k-lam_l),
    with the degenerate limit t e^{lam_k t} when |lam_k-lam_l|<1e-12."""
    e = jnp.exp(lam * t)
    dl = lam[:, None] - lam[None, :]
    close = jnp.abs(dl) < 1e-12
    safe_dl = jnp.where(close, 1.0, dl)
    Joff = (e[:, None] - e[None, :]) / safe_dl
    Jdeg = t * e[:, None] * jnp.ones_like(dl)
    return jnp.where(close, Jdeg, Joff)


def bridge_jax(Q, lam, U, Uinv, t, edge):
    """Expected dwell T[x] and usage N[x,y] over a branch of length t, averaged
    over the endpoint joint `edge` (NS,NS). Mirrors tkfdp.permfield.hr.bridge."""
    J = Jmat_jax(lam, t)
    P = (U * jnp.exp(lam * t)[None, :]) @ Uinv
    W = edge / jnp.maximum(P, 1e-300)
    M = W @ Uinv.T
    Aleft = U.T @ M
    AJ = Aleft * J
    UinvT_AJ = Uinv.T @ AJ
    T = jnp.einsum("xl,xl->x", UinvT_AJ, U)
    B = UinvT_AJ @ U.T
    Nmat = Q * B
    Nmat = Nmat - jnp.diag(jnp.diag(Nmat))
    return T, Nmat, P


# ============================================================================
# Metropolis--Hastings pair generator (single-transition, shared S)
# ============================================================================
def log_pair_stationary(pi_i, pi_j, H):
    """log pi_ij(a,b) = log pi_i(a) + log pi_j(b) - H(a,b) - log Z_ij   (A,A)."""
    lg = jnp.log(pi_i)[:, None] + jnp.log(pi_j)[None, :] - H
    return lg - jax.scipy.special.logsumexp(lg)


def mh_pair_generator(pi_i, pi_j, H, S):
    """Build the (NS,NS) single-transition Hastings-min pair generator and the
    (NS,) pair stationary, with x=(a,b) indexed x = a*A + b.

    Q[(a,b),(a',b)] = S(a,a') min(1, pi_ij(a',b)/pi_ij(a,b))   (site-i move)
    Q[(a,b),(a,b')] = S(b,b') min(1, pi_ij(a,b')/pi_ij(a,b))   (site-j move)
    diagonal = -rowsum; double moves have rate 0.
    """
    A = pi_i.shape[0]
    logp = log_pair_stationary(pi_i, pi_j, H)      # (a,b)
    pi_ij = jnp.exp(logp).reshape(A * A)
    eyeA = jnp.eye(A)

    # --- site-i move, axes (a, a', b): destination a' at fixed spectator b ---
    src_i = logp[:, None, :]                       # (a,1,b) = logp[a,b]
    dst_i = logp[None, :, :]                        # (1,a',b) = logp[a',b]
    acc_i = jnp.minimum(1.0, jnp.exp(dst_i - src_i))   # (a,a',b)
    Ri = (S[:, :, None] * acc_i) * (1.0 - eyeA)[:, :, None]   # zero a'==a
    Ri_t = jnp.transpose(Ri, (0, 2, 1))            # (a,b,a')
    Qi4 = Ri_t[:, :, :, None] * eyeA[None, :, None, :]   # (a,b,a',b') delta(b,b')

    # --- site-j move, axes (a, b, b'): destination b' at fixed a ---
    src_j = logp[:, :, None]                       # (a,b,1) = logp[a,b]
    dst_j = logp[:, None, :]                        # (a,1,b') = logp[a,b']
    acc_j = jnp.minimum(1.0, jnp.exp(dst_j - src_j))   # (a,b,b')
    Rj = (S[None, :, :] * acc_j) * (1.0 - eyeA)[None, :, :]   # zero b'==b
    Qj4 = Rj[:, :, None, :] * eyeA[:, None, :, None]   # (a,b,a',b') delta(a,a')

    Q = (Qi4 + Qj4).reshape(A * A, A * A)
    Q = Q - jnp.diag(Q.sum(1))
    return Q, pi_ij


# ============================================================================
# padded tree (mirrors fit_single_swap_field_jax.pad_tree, node-only)
# ============================================================================
def pad_tree(parent, tau, MAX_NODES):
    parent = np.asarray(parent, int)
    N = len(parent)
    assert N <= MAX_NODES
    root = int(np.where(parent < 0)[0][0])
    ch = [[] for _ in range(N)]
    for v in range(N):
        if parent[v] >= 0:
            ch[parent[v]].append(v)
    nl = sum(len(ch[v]) == 0 for v in range(N))
    assert all(len(ch[v]) == 0 for v in range(nl)) and \
        all(len(ch[v]) > 0 for v in range(nl, N)), "leaves must be nodes 0..nl-1"

    pre_r, stack = [], [root]
    while stack:
        v = stack.pop(); pre_r.append(v); stack.extend(ch[v])
    post_r = pre_r[::-1]
    pad_ids = list(range(N, MAX_NODES))
    pre = np.array(pre_r + pad_ids, int)
    post = np.array(post_r + pad_ids, int)

    is_leaf = np.zeros(MAX_NODES, bool); is_leaf[:nl] = True; is_leaf[N:] = True
    is_root = np.zeros(MAX_NODES, bool); is_root[root] = True
    node_valid = np.zeros(MAX_NODES, bool); node_valid[:N] = True
    p_scatter = np.full(MAX_NODES, MAX_NODES, int)
    p_gather = np.full(MAX_NODES, root, int)
    for v in range(N):
        if parent[v] >= 0:
            p_scatter[v] = parent[v]; p_gather[v] = parent[v]
    tau_p = np.zeros(MAX_NODES); tau_p[:N] = np.asarray(tau, float); tau_p[root] = 0.0

    return dict(
        post=jnp.asarray(post), pre=jnp.asarray(pre),
        is_leaf=jnp.asarray(is_leaf), is_root=jnp.asarray(is_root),
        node_valid=jnp.asarray(node_valid),
        parent_scatter=jnp.asarray(p_scatter), parent_gather=jnp.asarray(p_gather),
        tau=jnp.asarray(tau_p), root=jnp.asarray(root),
    ), N, nl


def leaf_indicator(states, tree, NS):
    """states (nl,) int pair-state per leaf -> (MAX_NODES, NS) one-hot at leaves,
    all-ones at internal/pad nodes."""
    M = tree["post"].shape[0]
    nl = states.shape[0]
    oh = jax.nn.one_hot(states, NS)                          # (nl,NS)
    pad = jnp.ones((M - nl, NS))
    return jnp.concatenate([oh, pad], 0)


# ============================================================================
# per-pair Felsenstein forward + outside + HR E-step (single generator)
# ============================================================================
def _inside(P_nodes, leaf_ind, tree):
    post = tree["post"]; p_scatter = tree["parent_scatter"]; is_leaf = tree["is_leaf"]
    M, NS = leaf_ind.shape
    acc0 = jnp.concatenate([leaf_ind, jnp.ones((1, NS))], 0)

    def step(carry, i):
        acc, gls = carry
        v = post[i]; pv = p_scatter[v]; lf = is_leaf[v]
        msg = acc[v]; s = msg.sum()
        up_v = jnp.where(lf, msg, msg / s)
        gls = gls + jnp.where(lf, 0.0, jnp.log(s))
        contrib = P_nodes[v] @ up_v
        acc = acc.at[pv].multiply(contrib)
        return (acc, gls), (up_v, msg, contrib)

    (_, gls), (ups, msgs, contribs) = jax.lax.scan(step, (acc0, 0.0), jnp.arange(M))
    up = jnp.zeros((M, NS)).at[post].set(ups)
    node_msg = jnp.zeros((M, NS)).at[post].set(msgs)
    contrib = jnp.zeros((M, NS)).at[post].set(contribs)
    return up, gls, node_msg, contrib


def _outside(P_nodes, up, node_msg, contrib, rp, tree):
    pre = tree["pre"]; p_gather = tree["parent_gather"]
    is_root = tree["is_root"]; root = tree["root"]
    node_valid = tree["node_valid"]
    M, NS = up.shape
    is_edge = node_valid & (~is_root)
    down0 = jnp.zeros((M, NS)).at[root].set(rp)

    def step(down, i):
        v = pre[i]; u = p_gather[v]; rf = is_root[v]
        excl = node_msg[u] / jnp.maximum(contrib[v], 1e-300)
        sib = down[u] * excl
        Pe = P_nodes[v]
        ev = sib[:, None] * Pe * up[v][None, :]
        ev = ev / jnp.maximum(ev.sum(), 1e-300)
        dv = Pe.T @ sib
        dv = dv / jnp.maximum(dv.sum(), 1e-300)
        down = jnp.where(rf, down, down.at[v].set(dv))
        ev_out = jnp.where(is_edge[v], ev, jnp.zeros((NS, NS)))
        return down, ev_out

    _, evs = jax.lax.scan(step, down0, jnp.arange(M))
    return jnp.zeros((M, NS, NS)).at[pre].set(evs)


def composite_forward_one(pi_i, pi_j, H, S, tree, leaf_states):
    """One pair: build M-H generator, run the padded Felsenstein forward, return
    the composite log-likelihood (eq:comp-forward)."""
    A = pi_i.shape[0]; NS = A * A
    Q, pi_ij = mh_pair_generator(pi_i, pi_j, H, S)
    lam, U, Uinv = eig_rev_jax(Q, jnp.clip(pi_ij, 1e-300, None))
    M = tree["post"].shape[0]
    P_nodes = jax.vmap(lambda v: expm_from_eig(lam, U, Uinv, tree["tau"][v]))(jnp.arange(M))
    P_nodes = jnp.clip(P_nodes, 1e-300, None)
    leaf_ind = leaf_indicator(leaf_states, tree, NS)
    up, gls, _, _ = _inside(P_nodes, leaf_ind, tree)
    root = tree["root"]
    dot = jnp.dot(pi_ij, up[root])
    return gls + jnp.log(jnp.maximum(dot, 1e-300))


def composite_estep_one(pi_i, pi_j, H, S, tree, leaf_states):
    """One pair: forward + outside + Holmes--Rubin bridge -> (loglik, N, T, rootpost)."""
    A = pi_i.shape[0]; NS = A * A
    Q, pi_ij = mh_pair_generator(pi_i, pi_j, H, S)
    lam, U, Uinv = eig_rev_jax(Q, jnp.clip(pi_ij, 1e-300, None))
    M = tree["post"].shape[0]
    P_nodes = jax.vmap(lambda v: expm_from_eig(lam, U, Uinv, tree["tau"][v]))(jnp.arange(M))
    P_nodes = jnp.clip(P_nodes, 1e-300, None)
    leaf_ind = leaf_indicator(leaf_states, tree, NS)

    up, gls, node_msg, contrib = _inside(P_nodes, leaf_ind, tree)
    root = tree["root"]
    dot = jnp.dot(pi_ij, up[root])
    loglik = gls + jnp.log(jnp.maximum(dot, 1e-300))

    edge = _outside(P_nodes, up, node_msg, contrib, pi_ij, tree)      # (M,NS,NS)

    def br(v):
        return bridge_jax(Q, lam, U, Uinv, tree["tau"][v], edge[v])
    T_v, N_v, _ = jax.vmap(br)(jnp.arange(M))
    N = N_v.sum(0); T = T_v.sum(0)                                    # padded edges are zero
    rootpost = pi_ij * up[root]; rootpost = rootpost / jnp.maximum(rootpost.sum(), 1e-300)
    return loglik, N, T, rootpost


# vmapped over a batch of pairs sharing the tree + shared S (mapped/vectorised)
_forward_batch = jax.jit(jax.vmap(
    composite_forward_one, in_axes=(0, 0, 0, None, None, 0)))
_estep_batch = jax.jit(jax.vmap(
    composite_estep_one, in_axes=(0, 0, 0, None, None, 0)))


# ============================================================================
# M-steps
# ============================================================================
@partial(jax.jit, static_argnums=(3,))
def mstep_S(N, T, pi_ij, A):
    """Closed-form shared-exchangeability M-step (eq:comp-mstep-S): pool all
    single-transition edges by unordered residue pair across both coordinates and
    the component-exchange symmetry; S(a,a') = usage / exposure. N (NS,NS),
    T (NS,), pi_ij (NS,). Returns symmetric S (A,A), zero diagonal."""
    NS = A * A
    N4 = N.reshape(A, A, A, A)                                  # [a,b,a',b']
    # usage: i-moves (b'=b) -> Ui[a,a']; j-moves (a'=a) -> Uj[b,b']
    Ui = jnp.einsum("abcb->ac", N4)
    Uj = jnp.einsum("abac->bc", N4)
    Cdir = Ui + Uj
    Cnum = Cdir + Cdir.T

    logp = jnp.log(jnp.clip(pi_ij.reshape(A, A), 1e-300, None))
    T2 = T.reshape(A, A)
    # i-move exposure per S(a,a'): sum_b T[a,b] * min(1, pi_ij[a',b]/pi_ij[a,b])
    acc_i = jnp.minimum(1.0, jnp.exp(logp[None, :, :] - logp[:, None, :]))   # (a,a',b)
    Ei = jnp.einsum("ab,acb->ac", T2, acc_i)                                 # (a,a')
    # j-move exposure per S(b,b'): sum_a T[a,b] * min(1, pi_ij[a,b']/pi_ij[a,b])
    acc_j = jnp.minimum(1.0, jnp.exp(logp[:, None, :] - logp[:, :, None]))    # (a,b,b')
    Ej = jnp.einsum("ab,abc->bc", T2, acc_j)                                  # (b,b')
    Edir = Ei + Ej
    Enum = Edir + Edir.T

    S = jnp.where(Enum > 0, Cnum / jnp.maximum(Enum, 1e-300), 0.0)
    S = S - jnp.diag(jnp.diag(S))
    return 0.5 * (S + S.T)


def sfull_from_S(S):
    """Lift the (A,A) residue exchangeability S to the (NS,NS) single-transition
    pair exchangeability Sfull, with Q_xy = Sfull_xy * min(1, pi_y/pi_x). pi-free."""
    A = S.shape[0]; eyeA = jnp.eye(A)
    base_i = S * (1.0 - eyeA)                                    # (a,a')
    Ii = jnp.broadcast_to(base_i[:, None, :], (A, A, A))         # (a,b,a')
    Qi4 = Ii[:, :, :, None] * eyeA[None, :, None, :]            # (a,b,a',b') delta(b,b')
    base_j = S * (1.0 - eyeA)                                    # (b,b')
    Ij = jnp.broadcast_to(base_j[None, :, :], (A, A, A))         # (a,b,b')
    Qj4 = Ij[:, :, None, :] * eyeA[:, None, :, None]            # (a,b,a',b') delta(a,a')
    return (Qi4 + Qj4).reshape(A * A, A * A)


@jax.jit
def _Qfun_and_grad(b, Sfull, N, T, rootpost):
    """EM Q-function (expected complete-data log-likelihood, S held fixed) and its
    gradient in b = log pi_ij:
       Q(b) = sum_x rootpost_x log pi_x
              + sum_{x!=y} [ N_xy log Q_xy - T_x Q_xy ],   Q_xy = Sfull_xy min(1, pi_y/pi_x).
    Increasing Q increases the observed-data composite log-likelihood (EM)."""
    NS = Sfull.shape[0]
    off = 1.0 - jnp.eye(NS)

    def Qf(bb):
        p = jax.nn.softmax(bb)
        acc = jnp.minimum(1.0, p[None, :] / jnp.maximum(p[:, None], 1e-300))
        Q = Sfull * acc * off
        rowoff = Q.sum(1)
        trans = jnp.sum(N * off * jnp.log(jnp.clip(Q, 1e-300, None))) - jnp.sum(T * rowoff)
        root = jnp.sum(rootpost * (bb - jax.scipy.special.logsumexp(bb)))
        return trans + root

    return jax.value_and_grad(Qf)(b)


def mstep_pi_ij(N, T, rootpost, pi_ij, Sfull, lr=0.5, bt=30):
    """pi/H block M-step: a single backtracked gradient ascent step on the EM
    Q-function above (S held via Sfull), guaranteeing Q non-decrease.  H is then
    read off as H = log pi_i + log pi_j - log pi_ij (up to the Potts gauge)."""
    b = jnp.log(jnp.clip(pi_ij, 1e-300, None))
    b = b - jax.scipy.special.logsumexp(b)
    val0, g = _Qfun_and_grad(b, Sfull, N, T, rootpost)
    step = lr; b_best = b
    for _ in range(bt):
        b2 = b + step * g; b2 = b2 - jax.scipy.special.logsumexp(b2)
        v2, _ = _Qfun_and_grad(b2, Sfull, N, T, rootpost)
        if float(v2) >= float(val0):
            b_best = b2; break
        step *= 0.5
    return jax.nn.softmax(b_best)


# ============================================================================
# brute-force numpy pair-tree Felsenstein (independent reference for check (a))
# ============================================================================
def brute_force_loglik(Q, pi_ij, parent, tau, leaf_states):
    """Dense, un-scaled Felsenstein on the NS-state pair chain via scipy expm."""
    from scipy.linalg import expm as sexpm
    Q = np.asarray(Q, float); pi_ij = np.asarray(pi_ij, float)
    N = len(parent); NS = Q.shape[0]
    root = int(np.where(parent < 0)[0][0])
    ch = [[] for _ in range(N)]
    for v in range(N):
        if parent[v] >= 0:
            ch[parent[v]].append(v)
    nl = sum(len(ch[v]) == 0 for v in range(N))
    P = {v: sexpm(Q * tau[v]) for v in range(N) if parent[v] >= 0}

    def clv(v):
        if len(ch[v]) == 0:
            e = np.zeros(NS); e[int(leaf_states[v])] = 1.0
            return e
        m = np.ones(NS)
        for u in ch[v]:
            m = m * (P[u] @ clv(u))
        return m

    return float(np.log(pi_ij @ clv(root)))


def simulate_pair(parent, tau, Q, pi_ij, seed):
    """Sample leaf pair-states down the tree from the pair CTMC."""
    from scipy.linalg import expm as sexpm
    Q = np.asarray(Q, float); pi_ij = np.asarray(pi_ij, float)
    rng = np.random.default_rng(seed)
    N = len(parent); NS = Q.shape[0]
    root = int(np.where(parent < 0)[0][0])
    ch = [[] for _ in range(N)]
    for v in range(N):
        if parent[v] >= 0:
            ch[parent[v]].append(v)
    nl = sum(len(ch[v]) == 0 for v in range(N))
    P = {v: sexpm(Q * tau[v]) for v in range(N) if parent[v] >= 0}
    state = np.zeros(N, int)
    order, stack = [], [root]
    while stack:
        v = stack.pop(); order.append(v); stack.extend(ch[v])
    state[root] = rng.choice(NS, p=pi_ij / pi_ij.sum())
    for v in order:
        if parent[v] >= 0:
            row = P[v][state[parent[v]]]
            state[v] = rng.choice(NS, p=np.clip(row, 0, None) / max(row.sum(), 1e-300))
    return state[:nl]


# ============================================================================
# trees + synthetic parameters
# ============================================================================
def binary_tree(nl):
    N = 2 * nl - 1
    parent = -np.ones(N, int)
    nodes = list(range(nl)); nxt = nl
    while len(nodes) > 1:
        a = nodes.pop(0); b = nodes.pop(0)
        parent[a] = nxt; parent[b] = nxt; nodes.append(nxt); nxt += 1
    return parent


def make_params(A, seed, exchangeable=True):
    """(pi_i, pi_j, H symmetric, S symmetric exchangeability) f64 numpy."""
    rng = np.random.default_rng(seed)
    S = np.asarray(S_LG08, float)[:A, :A].copy()
    np.fill_diagonal(S, 0.0); S = 0.5 * (S + S.T)
    pi0 = np.asarray(PI_LG08, float)[:A]; pi0 = pi0 / pi0.sum()
    pi_i = np.clip(pi0 * (1 + 0.3 * rng.standard_normal(A)), 1e-3, None); pi_i /= pi_i.sum()
    if exchangeable:
        pi_j = pi_i.copy()
    else:
        pi_j = np.clip(pi0 * (1 + 0.3 * rng.standard_normal(A)), 1e-3, None); pi_j /= pi_j.sum()
    Hs = 0.8 * rng.standard_normal((A, A)); H = 0.5 * (Hs + Hs.T)   # symmetric coupling
    H = H - H.mean()
    return pi_i, pi_j, H, S


# ============================================================================
# self-checks
# ============================================================================
def check_a(A=20, nl=6, n_pairs=3, seed=0):
    print(f"# CHECK (a): generator + composite forward vs brute force  "
          f"(A={A}, NS={A*A}, leaves={nl}, pairs={n_pairs})", flush=True)
    parent = binary_tree(nl); N = len(parent)
    rng = np.random.default_rng(seed)
    tau = rng.uniform(0.2, 0.8, N); tau[np.where(parent < 0)[0][0]] = 0.0
    MAX_NODES = N + 2
    tree, _, nl_t = pad_tree(parent, tau, MAX_NODES)

    # per-pair params (shared S), simulate leaf states, compare forwards
    _, _, _, S = make_params(A, seed)
    pis_i, pis_j, Hs, leafs = [], [], [], []
    fwd_diffs = []; rev_errs = []; stat_errs = []
    for p in range(n_pairs):
        pi_i, pi_j, H, _ = make_params(A, seed + 100 + p, exchangeable=(p % 2 == 0))
        Qj, pij = mh_pair_generator(jnp.asarray(pi_i), jnp.asarray(pi_j),
                                    jnp.asarray(H), jnp.asarray(S))
        Q = np.asarray(Qj); pi_ij = np.asarray(pij)
        # reversibility: pi_ij[x] Q[x,y] symmetric; stationarity pi_ij @ Q = 0
        F = pi_ij[:, None] * Q
        rev_errs.append(float(np.max(np.abs(F - F.T))))
        stat_errs.append(float(np.max(np.abs(pi_ij @ Q))))
        leaf_states = simulate_pair(parent, tau, Q, pi_ij, seed + 200 + p)
        bf = brute_force_loglik(Q, pi_ij, parent, tau, leaf_states)
        leaf_pad = np.zeros(nl_t, int); leaf_pad[:nl] = leaf_states
        jx = float(composite_forward_one(
            jnp.asarray(pi_i), jnp.asarray(pi_j), jnp.asarray(H), jnp.asarray(S),
            tree, jnp.asarray(leaf_pad)))
        fwd_diffs.append(abs(bf - jx))
        pis_i.append(pi_i); pis_j.append(pi_j); Hs.append(H); leafs.append(leaf_pad)

    # micro-check: JAX HR bridge == audited numpy tkfdp.permfield.hr.bridge
    from tkfdp.permfield.hr import bridge as np_bridge, eig_rev as np_eig
    pi_i, pi_j, H, _ = make_params(A, seed + 100)
    Qj, pij = mh_pair_generator(jnp.asarray(pi_i), jnp.asarray(pi_j),
                                jnp.asarray(H), jnp.asarray(S))
    Q = np.asarray(Qj); pi_ij = np.asarray(pij)
    rng2 = np.random.default_rng(7)
    edge = rng2.random((A * A, A * A)); edge /= edge.sum()
    lam_j, U_j, Uinv_j = eig_rev_jax(jnp.asarray(Q), jnp.asarray(np.clip(pi_ij, 1e-300, None)))
    Tj, Nj, _ = bridge_jax(jnp.asarray(Q), lam_j, U_j, Uinv_j, 0.37, jnp.asarray(edge))
    Tn, Nn, _ = np_bridge(Q, np.clip(pi_ij, 1e-12, None), 0.37, edge, want_N=True)
    hr_T = float(np.max(np.abs(np.asarray(Tj) - Tn)))
    hr_N = float(np.max(np.abs(np.asarray(Nj) - Nn)))

    # vmapped batch forward equals the per-pair forwards
    batch = _forward_batch(jnp.asarray(np.stack(pis_i)), jnp.asarray(np.stack(pis_j)),
                           jnp.asarray(np.stack(Hs)), jnp.asarray(S), tree,
                           jnp.asarray(np.stack(leafs)))
    per = np.array([float(composite_forward_one(
        jnp.asarray(pis_i[p]), jnp.asarray(pis_j[p]), jnp.asarray(Hs[p]),
        jnp.asarray(S), tree, jnp.asarray(leafs[p]))) for p in range(n_pairs)])
    vmap_diff = float(np.max(np.abs(np.asarray(batch) - per)))

    print(f"  reversibility  max|pi Q - (pi Q)^T|         : {max(rev_errs):.3e}")
    print(f"  stationarity   max|pi_ij @ Q|               : {max(stat_errs):.3e}")
    print(f"  forward log-lik  max|JAX - brute force|      : {max(fwd_diffs):.3e}")
    print(f"  HR bridge      max|JAX - numpy hr.bridge| T  : {hr_T:.3e}")
    print(f"  HR bridge      max|JAX - numpy hr.bridge| N  : {hr_N:.3e}")
    print(f"  vmap batch     max|batch - per-pair|         : {vmap_diff:.3e}")
    ok = (max(fwd_diffs) < 1e-6 and max(rev_errs) < 1e-9 and max(stat_errs) < 1e-8
          and hr_T < 1e-9 and hr_N < 1e-9 and vmap_diff < 1e-9)
    print(f"  CHECK (a) PASS (<1e-6 forward, <1e-9 rev/HR/vmap): {ok}")
    return ok


def check_b(A=20, nl=6, n_pairs=3, n_iter=15, seed=1):
    print(f"\n# CHECK (b): coordinate-ascent monotonicity on a synthetic 2-site "
          f"coupled cluster  (A={A}, leaves={nl}, pairs={n_pairs}, iters={n_iter})",
          flush=True)
    parent = binary_tree(nl); N = len(parent)
    rng = np.random.default_rng(seed)
    tau = rng.uniform(0.2, 0.8, N); tau[np.where(parent < 0)[0][0]] = 0.0
    MAX_NODES = N + 2
    tree, _, nl_t = pad_tree(parent, tau, MAX_NODES)

    # TRUE params (shared S), simulate leaves for each pair
    pi_i_t, pi_j_t, H_t, S_t = make_params(A, seed)
    pis_i, pis_j, Hs_true, leafs = [], [], [], []
    for p in range(n_pairs):
        pi_i, pi_j, H, _ = make_params(A, seed + 10 + p, exchangeable=(p % 2 == 0))
        Qj, pij = mh_pair_generator(jnp.asarray(pi_i), jnp.asarray(pi_j),
                                    jnp.asarray(H), jnp.asarray(S_t))
        leaf_states = simulate_pair(parent, tau, np.asarray(Qj), np.asarray(pij),
                                    seed + 300 + p)
        leaf_pad = np.zeros(nl_t, int); leaf_pad[:nl] = leaf_states
        pis_i.append(pi_i); pis_j.append(pi_j); Hs_true.append(H); leafs.append(leaf_pad)
    PI_i = jnp.asarray(np.stack(pis_i)); PI_j = jnp.asarray(np.stack(pis_j))
    H_true = jnp.asarray(np.stack(Hs_true)); LEAF = jnp.asarray(np.stack(leafs))

    def total_ll(S, H_batch):
        return float(jnp.sum(_forward_batch(PI_i, PI_j, H_batch, S, tree, LEAF)))

    # ---- (b1) S-only EM (pi_ij / H fixed at truth): exact M-step -> strict monotone
    S = jnp.asarray(S_t * (0.3 + 0.4 * rng.random((A, A))))          # perturbed init
    S = 0.5 * (S + S.T) - jnp.diag(jnp.diag(0.5 * (S + S.T)))
    hist1 = [total_ll(S, H_true)]
    for _ in range(n_iter):
        _, N_all, T_all, _ = _estep_batch(PI_i, PI_j, H_true, S, tree, LEAF)
        # aggregate HR stats over pairs (component-exchangeable pooling per pair
        # uses each pair's own pi_ij; here we pool with a representative pi_ij --
        # the S-step exposure uses each pair's stationary, so accumulate weighted)
        Ntot = N_all.sum(0); Ttot = T_all.sum(0)
        # exposure needs pi_ij per pair; pool by summing per-pair usage/exposure.
        num = jnp.zeros((A, A)); den = jnp.zeros((A, A))
        for p in range(n_pairs):
            _, pij = mh_pair_generator(PI_i[p], PI_j[p], H_true[p], S)
            Sp_num, Sp_den = _S_num_den(N_all[p], T_all[p], pij, A)
            num = num + Sp_num; den = den + Sp_den
        S = jnp.where(den > 0, num / jnp.maximum(den, 1e-300), 0.0)
        S = S - jnp.diag(jnp.diag(S)); S = 0.5 * (S + S.T)
        hist1.append(total_ll(S, H_true))
    d1 = np.diff(np.array(hist1))
    mono1 = bool(np.all(d1 > -1e-8))
    print(f"  (b1) S-only EM (pi/H fixed): LL {hist1[0]:.4f} -> {hist1[-1]:.4f}")
    print(f"       min step delta = {d1.min():.3e}   STRICTLY MONOTONE(>-1e-8): {mono1}")

    # ---- (b2) joint (S + pi_ij/H) coordinate ascent: non-decreasing
    S = jnp.asarray(S_t * (0.3 + 0.4 * rng.random((A, A))))
    S = 0.5 * (S + S.T) - jnp.diag(jnp.diag(0.5 * (S + S.T)))
    H_cur = jnp.asarray(0.2 * rng.standard_normal((n_pairs, A, A)))
    H_cur = 0.5 * (H_cur + jnp.transpose(H_cur, (0, 2, 1)))
    hist2 = [total_ll(S, H_cur)]
    for _ in range(n_iter):
        _, N_all, T_all, RP_all = _estep_batch(PI_i, PI_j, H_cur, S, tree, LEAF)
        # exact shared-S max (transition part of Q) using CURRENT pi_ij
        num = jnp.zeros((A, A)); den = jnp.zeros((A, A))
        for p in range(n_pairs):
            _, pij = mh_pair_generator(PI_i[p], PI_j[p], H_cur[p], S)
            Sp_num, Sp_den = _S_num_den(N_all[p], T_all[p], pij, A)
            num = num + Sp_num; den = den + Sp_den
        S = jnp.where(den > 0, num / jnp.maximum(den, 1e-300), 0.0)
        S = S - jnp.diag(jnp.diag(S)); S = 0.5 * (S + S.T)
        # pi_ij / H backtracked ascent per pair (Q-function, NEW S) -> non-decrease
        Sfull = sfull_from_S(S)
        newH = []
        for p in range(n_pairs):
            _, pij_cur = mh_pair_generator(PI_i[p], PI_j[p], H_cur[p], S)
            pij_new = mstep_pi_ij(N_all[p], T_all[p], RP_all[p], pij_cur, Sfull)
            logp = jnp.log(jnp.clip(pij_new.reshape(A, A), 1e-300, None))
            Hp = jnp.log(PI_i[p])[:, None] + jnp.log(PI_j[p])[None, :] - logp
            Hp = Hp - Hp.mean()
            newH.append(0.5 * (Hp + Hp.T))
        H_cur = jnp.stack(newH)
        hist2.append(total_ll(S, H_cur))
    d2 = np.diff(np.array(hist2))
    mono2 = bool(np.all(d2 > -1e-3))
    print(f"  (b2) joint (S + pi/H) ascent: LL {hist2[0]:.4f} -> {hist2[-1]:.4f}")
    print(f"       min step delta = {d2.min():.3e}   MONOTONE(>-1e-3): {mono2}")

    ok = mono1 and mono2 and (hist1[-1] >= hist1[0]) and (hist2[-1] >= hist2[0])
    print(f"  CHECK (b) PASS (both coordinate ascents non-decreasing): {ok}")
    return ok


def _S_num_den(N, T, pi_ij, A):
    """Per-pair usage numerator and exposure denominator for the shared-S M-step
    (eq:comp-mstep-S), so several pairs can be pooled by summing num/den."""
    N4 = N.reshape(A, A, A, A)
    Ui = jnp.einsum("abcb->ac", N4)
    Uj = jnp.einsum("abac->bc", N4)
    Cdir = Ui + Uj; Cnum = Cdir + Cdir.T
    logp = jnp.log(jnp.clip(pi_ij.reshape(A, A), 1e-300, None))
    T2 = T.reshape(A, A)
    acc_i = jnp.minimum(1.0, jnp.exp(logp[None, :, :] - logp[:, None, :]))
    Ei = jnp.einsum("ab,acb->ac", T2, acc_i)
    acc_j = jnp.minimum(1.0, jnp.exp(logp[:, None, :] - logp[:, :, None]))
    Ej = jnp.einsum("ab,abc->bc", T2, acc_j)
    Edir = Ei + Ej; Enum = Edir + Edir.T
    return Cnum, Enum


def check_c(A=20, seed=0):
    """mstep_S is the EXACT maximizer (guards the exposure-axis bug fixed 2026-07):
    at the true stationary the equilibrium sufficient stats N[x,y]=pi(x)Q[x,y],
    T[x]=pi(x) must make S=usage/exposure return S_true exactly.  Monotone-LL (check_b)
    does NOT certify this -- a non-maximizing S-step still ascends the LL."""
    print(f"\n# CHECK (c): mstep_S exact-maximizer at the true stationary (A={A})", flush=True)
    pi_i, pi_j, H, S_true = make_params(A, seed, exchangeable=False)
    Q, pij = mh_pair_generator(jnp.asarray(pi_i), jnp.asarray(pi_j),
                               jnp.asarray(H), jnp.asarray(S_true))
    Q = np.asarray(Q); pij = np.asarray(pij)
    Neq = pij[:, None] * Q; Neq = Neq - np.diag(np.diag(Neq))     # N[x,y]=pi(x)Q[x,y]
    S_hat = np.asarray(mstep_S(jnp.asarray(Neq), jnp.asarray(pij), jnp.asarray(pij), A))
    off = ~np.eye(A, dtype=bool)
    err = float(np.abs(S_hat[off] - np.asarray(S_true)[off]).max())
    ok = err < 1e-6
    print(f"  max|mstep_S(equilibrium stats) - S_true| = {err:.3e}   PASS(<1e-6): {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--A", type=int, default=20, help="alphabet size (NS=A^2 states)")
    ap.add_argument("--leaves", type=int, default=6)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()
    if args.selfcheck:
        a = check_a(A=args.A, nl=args.leaves, n_pairs=args.pairs)
        b = check_b(A=args.A, nl=args.leaves, n_pairs=args.pairs, n_iter=args.iters)
        c = check_c(A=args.A)
        print(f"\n### OVERALL: check(a)={a}  check(b)={b}  check(c)={c}  ALL PASS={a and b and c}")
    else:
        print("Use --selfcheck")


if __name__ == "__main__":
    main()
