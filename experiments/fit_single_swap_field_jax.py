#!/usr/bin/env python3
"""Pure-JAX trainer for the SINGLE-SWAP FIELD (Cayley<=1 truncated permutation
field: identity + the C(C-1)/2 single transpositions, 1+C(C-1)/2 states).

This mirrors the AUDITED numpy reference ``experiments/fit_single_swap_field.py``
(which reuses ``experiments/permfield_elbo.py``) coordinate-ascent EM, but with a
pure-JAX (jax.numpy / lax.scan / vmap) E-step and closed-form M-step.

MODEL (matches the UPDATED numpy reference, commit "free per-state stationary"):
the field is a STAR-F81 CTMC with a FREE per-state stationary ``pi_field`` (one
probability per of the 1+C(C-1)/2 states) and an overall F81 rate ``rho_field``:

    Q[0,k] = rho_field * pi_field[k]      (identity -> transposition tau_k)
    Q[k,0] = rho_field * pi_field[0]      (tau_k -> identity)

reversible w.r.t. ``pi_field`` for any ``pi_field, rho_field`` (no per-pair swap
rates ``w`` -- the swap preferences now live in the free stationary).  The field
M-step is the MAP stationary under an identity-favouring Dirichlet prior
(``field_prior``): ``pi_field ~ (rootc + incoming-usage) + pseudo`` plus the F81
rate ``rho_field = Uf.sum() / sum_i Wf[i](1-pi_field[i])`` -- exactly
``fit_single_swap_field.solve_field_stationary``.

COMPILE-ONCE ACROSS FAMILIES (the refactor): every family tree is padded to a
fixed ``MAX_NODES`` (and common ``MAX_CLUSTERS`` / ``Lmax``) with validity MASKS,
and the tree structure (post/pre orders, parent scatter/gather, tau, masks, root)
is passed as ARGUMENTS to a single top-level ``jax.jit``'d ``full_estep`` -- NOT
captured in a per-family factory closure.  ``MAX_NODES`` is then the only node
dim and is identical across families, so multi-family corpus fitting compiles
ONCE (no per-tree recompile).  Masked padding is a mathematical no-op: pad nodes
scatter to a dummy row and contribute identity ops / zero flux; pad clusters are
zeroed by ``cluster_mask``.

  * the residue effective operator Peff[(c,v)] = expm(gtr_Q(pibar_{c,parent(v)})*tau_v)
    is recomputed per branch (field-posterior dependent), reversible eig vmapped
    over (class,node);
  * exact Holmes-Rubin bridge stats (reversible eig + divided-difference kernel);
  * f64 throughout for validation (tau-binning OFF, no f32).

``--validate`` builds tiny deterministic cases (C=3 and C=4) and compares JAX vs
the numpy reference to 1e-6 on every E-step + M-step quantity, then confirms the
padded jit compiles ONCE across two different-sized families.  Run with:

  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=4 \
      PYTHONPATH=src:/home/yam/tkf-mixdom/python \
      python3 experiments/fit_single_swap_field_jax.py --validate
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from tkfdp.lg08 import S_LG08, PI_LG08  # noqa: E402

A = 20
S_J = jnp.asarray(np.asarray(S_LG08, float))
PI0 = np.asarray(PI_LG08, float)


# ============================================================================
# linear algebra: reversible eig, matrix exponential, HR divided-difference
# ============================================================================
def gtr_Q_jax(pi):
    """F81/GTR generator Q[x,y] = S[x,y]*pi[y] off-diag; diag = -rowsum."""
    Q = S_J * pi[None, :]                       # S diagonal is 0 -> Q diagonal 0
    rs = Q.sum(1)
    return Q - jnp.diag(rs)


def eig_rev_jax(Q, pi):
    """Symmetric eigendecomposition of a reversible generator (mirrors
    tkfdp.permfield.hr.eig_rev). Returns (lam, U, Uinv), Q = U diag(lam) Uinv."""
    d = jnp.sqrt(jnp.clip(pi, 1e-300, None))
    di = 1.0 / d
    Qs = d[:, None] * Q * di[None, :]
    Qs = 0.5 * (Qs + Qs.T)
    lam, V = jnp.linalg.eigh(Qs)
    U = di[:, None] * V
    Uinv = V.T * d[None, :]
    return lam, U, Uinv


def expm_from_eig(lam, U, Uinv, t):
    """P = expm(Q t) from a reversible eig of Q."""
    return (U * jnp.exp(lam * t)[None, :]) @ Uinv


def Jmat_jax(lam, t):
    """Divided-difference kernel J_kl = (e^{lam_k t}-e^{lam_l t})/(lam_k-lam_l),
    with the degenerate limit J -> t e^{lam_k t} when |lam_k-lam_l|<1e-12 (mirrors
    tkfdp.permfield.hr._Jmat, including its off-diagonal degenerate handling)."""
    e = jnp.exp(lam * t)
    dl = lam[:, None] - lam[None, :]
    close = jnp.abs(dl) < 1e-12
    safe_dl = jnp.where(close, 1.0, dl)
    Joff = (e[:, None] - e[None, :]) / safe_dl
    Jdeg = t * e[:, None] * jnp.ones_like(dl)   # value t*e_k in row k
    return jnp.where(close, Jdeg, Joff)


def bridge_jax(Q, lam, U, Uinv, t, edge):
    """Expected dwell T[x] and usage N[x,y] over a branch of length t, averaged
    over the endpoint joint `edge` (n,n). Mirrors tkfdp.permfield.hr.bridge
    exactly (eigenmode contraction). Returns (T, N, P)."""
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
# field CTMC (truncated single-swap, NEW free-stationary star-F81): pure-JAX
# ============================================================================
def trunc_field_jax(C, pi_field, rho_field):
    """STAR-F81 field with a FREE per-state stationary pi_field (matches the numpy
    reference fit_single_swap_field.trunc_field). Jump-to-stationary star form
    Q[0,k]=rho_field*pi_field[k], Q[k,0]=rho_field*pi_field[0]; reversible w.r.t.
    pi_field for any pi_field, rho_field. NOT rate-normalised (rho_field IS the
    field rate). Returns (Q (nS,nS), pi (nS,))."""
    K = C * (C - 1) // 2
    n = 1 + K
    pi = jnp.clip(pi_field, 1e-12, None)
    pi = pi / pi.sum()
    Q = jnp.zeros((n, n))
    Q = Q.at[0, 1:].set(rho_field * pi[1:])      # id -> tau_k
    Q = Q.at[1:, 0].set(rho_field * pi[0])       # tau_k -> id
    Q = Q - jnp.diag(Q.sum(1))                   # diag = -rowsum
    return Q, pi


# ============================================================================
# tree structure: PADDED to MAX_NODES with validity masks (pure numpy host build)
# ============================================================================
def pad_tree(parent, tau, MAX_NODES):
    """Pad a family tree to a fixed MAX_NODES, returning a dict of jnp arrays
    (leading dim MAX_NODES, plus a scalar root index) suitable as a jit ARGUMENT.

    Layout: real leaves 0..nl-1, real internal nl..N-1, pad nodes N..MAX_NODES-1.
    Pad nodes scatter their (identity) contribution into a dummy accumulator row
    (index MAX_NODES) and are marked as leaves so they add nothing to the log-lik;
    their branch length is 0 (Peff = I).  Masks (node_valid / is_root / is_leaf)
    make every padded contribution a mathematical no-op.
    """
    parent = np.asarray(parent, int)
    N = len(parent)
    assert N <= MAX_NODES, f"N={N} > MAX_NODES={MAX_NODES}"
    root = int(np.where(parent < 0)[0][0])
    ch = [[] for _ in range(N)]
    for v in range(N):
        if parent[v] >= 0:
            ch[parent[v]].append(v)
    nl = sum(len(ch[v]) == 0 for v in range(N))
    assert np.all([len(ch[v]) == 0 for v in range(nl)]) and \
        np.all([len(ch[v]) > 0 for v in range(nl, N)]), \
        "reference requires leaves to be nodes 0..nl-1"

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
    p_scatter = np.full(MAX_NODES, MAX_NODES, int)   # root + pad -> dummy row
    p_gather = np.full(MAX_NODES, root, int)         # root + pad -> safe (masked)
    for v in range(N):
        if parent[v] >= 0:
            p_scatter[v] = parent[v]
            p_gather[v] = parent[v]
    tau_p = np.zeros(MAX_NODES); tau_p[:N] = np.asarray(tau, float); tau_p[root] = 0.0

    return dict(
        post=jnp.asarray(post), pre=jnp.asarray(pre),
        is_leaf=jnp.asarray(is_leaf), is_root=jnp.asarray(is_root),
        node_valid=jnp.asarray(node_valid),
        parent_scatter=jnp.asarray(p_scatter), parent_gather=jnp.asarray(p_gather),
        tau=jnp.asarray(tau_p), root=jnp.asarray(root),
    ), N, nl


def build_arch_onehot(arch, C):
    """(nS,C,C) one-hot arch[i,c] -> arch class-archetype indicator, as a jnp
    array to be passed as a (family-independent, C-only) jit argument."""
    nS = arch.shape[0]
    oh = np.zeros((nS, C, C))
    for i in range(nS):
        for c in range(C):
            oh[i, c, arch[i, c]] = 1.0
    return jnp.asarray(oh)


# ============================================================================
# TOP-LEVEL jitted E-step: tree passed as ARGUMENT (one compile across families)
# ============================================================================
_TRACE_COUNT = 0   # incremented once per XLA trace of full_estep (single-compile probe)


def full_estep(tree, arch_onehot, b_fields, leaf_obs, colmask, cmask,
               pis, rho, pi_field, rho_field):
    """One padded, masked E-step over a padded batch of clusters.

    tree        : dict of jnp arrays (MAX_NODES) + scalar root -- the ARGUMENT that
                  makes this compile once across families.
    arch_onehot : (nS,C,C) class-archetype indicator (C-only, family-independent).
    b_fields    : (nCl, MAX_NODES, nS) per-cluster field posteriors.
    leaf_obs    : (nCl, MAX_NODES, Lmax) per-node residue (gap=20 for internal/pad).
    colmask     : (nCl, Lmax) valid-column mask.
    cmask       : (nCl,) valid-cluster mask (pad clusters -> 0).
    pis         : (C,A) archetype stationaries; rho (C,) class weights.
    pi_field    : (nS,) free field stationary; rho_field scalar F81 field rate.
    """
    global _TRACE_COUNT
    _TRACE_COUNT += 1     # python side effect: fires exactly once per trace

    post = tree["post"]; pre = tree["pre"]
    is_leaf = tree["is_leaf"]; is_root = tree["is_root"]; node_valid = tree["node_valid"]
    p_scatter = tree["parent_scatter"]; p_gather = tree["parent_gather"]
    tau = tree["tau"]; root = tree["root"]

    M = post.shape[0]                       # MAX_NODES (shape-derived, never static)
    nS = arch_onehot.shape[0]
    C = arch_onehot.shape[1]
    is_edge = node_valid & (~is_root)       # real, non-root branch

    # ---- global reversible eigs / branch ops ----
    Qc = jax.vmap(gtr_Q_jax)(pis)                                   # (C,A,A)
    lam_a, U_a, Uinv_a = jax.vmap(eig_rev_jax)(Qc, pis)
    Qf, pif = trunc_field_jax(C, pi_field, rho_field)
    lam_f, U_f, Uinv_f = eig_rev_jax(Qf, jnp.clip(pif, 1e-10, None))
    Pf = jax.vmap(lambda v: jnp.clip(
        expm_from_eig(lam_f, U_f, Uinv_f, tau[v]), 1e-300, None))(jnp.arange(M))

    def logPa_av(a, v):
        P = expm_from_eig(lam_a[a], U_a[a], Uinv_a[a], tau[v])
        return jnp.log(jnp.clip(P, 1e-300, None))
    aa, vv = jnp.meshgrid(jnp.arange(C), jnp.arange(M), indexing="ij")
    logPa = jax.vmap(jax.vmap(logPa_av))(aa, vv)                    # (C,M,A,A)

    # ---- residue inside (Felsenstein) ----
    def leaf_indicator(obs_col):
        """obs_col (M,) int -> (M,A): valid leaf residue indicator, else all-ones."""
        oc = jnp.clip(obs_col, 0, A - 1)
        onehot = jax.nn.one_hot(oc, A)
        return jnp.where((obs_col < A)[:, None], onehot, jnp.ones((M, A)))

    def inside_one(Peff_c, leaf_ind):
        acc0 = jnp.concatenate([leaf_ind, jnp.ones((1, A))], 0)     # (M+1,A)

        def step(carry, i):
            acc, gls = carry
            v = post[i]; pv = p_scatter[v]; lf = is_leaf[v]
            msg = acc[v]; s = msg.sum()
            up_v = jnp.where(lf, msg, msg / s)
            gls = gls + jnp.where(lf, 0.0, jnp.log(s))
            contrib = Peff_c[v] @ up_v
            acc = acc.at[pv].multiply(contrib)
            return (acc, gls), (up_v, msg, contrib)

        (_, gls), (ups, msgs, contribs) = jax.lax.scan(
            step, (acc0, 0.0), jnp.arange(M))
        up = jnp.zeros((M, A)).at[post].set(ups)
        node_msg = jnp.zeros((M, A)).at[post].set(msgs)
        contrib = jnp.zeros((M, A)).at[post].set(contribs)
        return up, gls, node_msg, contrib

    def outside_one(Peff_c, up_c, node_msg_c, contrib_c, rp_c):
        down0 = jnp.zeros((M, A)).at[root].set(rp_c)

        def step(down, i):
            v = pre[i]; u = p_gather[v]; rf = is_root[v]
            excl = node_msg_c[u] / contrib_c[v]
            sib = down[u] * excl
            Pe = Peff_c[v]
            ev = sib[:, None] * Pe * up_c[v][None, :]
            ev = ev / jnp.maximum(ev.sum(), 1e-300)
            dv = Pe.T @ sib
            dv = dv / jnp.maximum(dv.sum(), 1e-300)
            down = jnp.where(rf, down, down.at[v].set(dv))
            ev_out = jnp.where(is_edge[v], ev, jnp.zeros((A, A)))   # zero pad+root
            return down, ev_out

        _, evs = jax.lax.scan(step, down0, jnp.arange(M))
        return jnp.zeros((M, A, A)).at[pre].set(evs)

    def field_bp(phi):
        """Field BP over nS states with LOG node potentials phi (M,nS).
        Returns b (M,nS), xi (M,nS,nS) [0 on pad+root branches], logZf."""
        ephi = jnp.exp(phi - phi.max(1, keepdims=True))
        lsc = phi.max(1)
        acc0 = jnp.concatenate([jnp.ones((M, nS)), jnp.ones((1, nS))], 0)

        def istep(carry, i):
            acc, glz = carry
            v = post[i]; pv = p_scatter[v]
            childprod = acc[v]
            m = ephi[v] * childprod; s = m.sum()
            up_v = m / s
            glz = glz + jnp.where(node_valid[v], jnp.log(s) + lsc[v], 0.0)
            contrib = Pf[v] @ up_v
            acc = acc.at[pv].multiply(contrib)
            return (acc, glz), (up_v, childprod, contrib)

        (_, glz), (ups, childprods, contribs) = jax.lax.scan(
            istep, (acc0, 0.0), jnp.arange(M))
        up = jnp.zeros((M, nS)).at[post].set(ups)
        childprod = jnp.zeros((M, nS)).at[post].set(childprods)
        contrib = jnp.zeros((M, nS)).at[post].set(contribs)

        rootvec = pif * up[root]
        logZf = jnp.log(jnp.maximum(rootvec.sum(), 1e-300)) + glz
        b_root = rootvec / rootvec.sum()

        down0 = jnp.zeros((M, nS)).at[root].set(pif)

        def ostep(down, i):
            v = pre[i]; u = p_gather[v]; rf = is_root[v]
            excl = childprod[u] / contrib[v]
            sib = ephi[u] * down[u] * excl
            sib = sib / jnp.maximum(sib.sum(), 1e-300)
            Pe = Pf[v]
            joint = sib[:, None] * Pe * up[v][None, :]
            joint = joint / jnp.maximum(joint.sum(), 1e-300)
            dv = sib @ Pe
            dv = dv / jnp.maximum(dv.sum(), 1e-300)
            down = jnp.where(rf, down, down.at[v].set(dv))
            joint_out = jnp.where(is_edge[v], joint, jnp.zeros((nS, nS)))
            bv_out = jnp.where(is_edge[v], joint.sum(0), jnp.zeros(nS))
            return down, (joint_out, bv_out)

        _, (joints, bvs) = jax.lax.scan(ostep, down0, jnp.arange(M))
        xi = jnp.zeros((M, nS, nS)).at[pre].set(joints)
        b = jnp.zeros((M, nS)).at[pre].set(bvs)
        b = b.at[root].set(b_root)
        return b, xi, logZf

    # ---- per-cluster E-step ----
    def estep_cluster(b_field, leaf_obs_c, colmask_c):
        Lmax = leaf_obs_c.shape[1]
        beta = jnp.einsum("ui,ica->cua", b_field, arch_onehot)       # (C,M,C)
        pibar = jnp.einsum("cua,ax->cux", beta, pis)                 # (C,M,A)

        def eig_cu(pib):
            return eig_rev_jax(gtr_Q_jax(pib), pib)
        lam_e, U_e, Uinv_e = jax.vmap(jax.vmap(eig_cu))(pibar)       # (C,M,*)

        def peff_cv(c, v):
            u = p_gather[v]
            return expm_from_eig(lam_e[c, u], U_e[c, u], Uinv_e[c, u], tau[v])
        cc, vv2 = jnp.meshgrid(jnp.arange(C), jnp.arange(M), indexing="ij")
        Peff = jax.vmap(jax.vmap(peff_cv))(cc, vv2)                  # (C,M,A,A)

        leaf_ind = jax.vmap(leaf_indicator, in_axes=1)(leaf_obs_c)   # (Lmax,M,A)

        def inside_c(Peff_c):
            return jax.vmap(lambda li: inside_one(Peff_c, li))(leaf_ind)
        up, gls, node_msg, contrib = jax.vmap(inside_c)(Peff)
        # up (C,L,M,A), gls (C,L), node_msg (C,L,M,A), contrib (C,L,M,A)

        rp = pibar[:, root, :]                                       # (C,A)
        root_up = up[:, :, root, :]                                  # (C,L,A)
        dot = jnp.einsum("ca,cla->cl", rp, root_up)                  # (C,L)
        col_ll = (gls + jnp.log(jnp.maximum(dot, 1e-300))).T         # (L,C)

        lr = jnp.log(rho)[None, :] + col_ll                         # (L,C)
        gamma = jax.nn.softmax(lr, axis=1)
        col_lse = jax.scipy.special.logsumexp(lr, axis=1)
        obj = jnp.sum(colmask_c * col_lse)
        gamma_m = gamma * colmask_c[:, None]
        gsum = gamma_m.sum(0)

        def outside_c(Peff_c, up_c, nm_c, ct_c, rp_c):
            return jax.vmap(
                lambda u_, n_, c_: outside_one(Peff_c, u_, n_, c_, rp_c)
            )(up_c, nm_c, ct_c)
        q_edge = jax.vmap(outside_c)(Peff, up, node_msg, contrib, rp)  # (C,L,M,A,A)
        EdgeAcc = jnp.einsum("lc,clvab->cvab", gamma_m, q_edge)        # (C,M,A,A)

        root_post = rp[:, None, :] * root_up
        root_post = root_post / jnp.maximum(root_post.sum(-1, keepdims=True), 1e-300)
        RootAcc = jnp.einsum("lc,cla->ca", gamma_m, root_post)        # (C,A)

        # field node potentials
        EL = jnp.einsum("cvxy,avxy->cva", EdgeAcc, logPa)             # (C,M,C)
        contrib_phi = jnp.einsum("ica,cva->vi", arch_onehot, EL)      # (M,nS)
        phi = jnp.zeros((M + 1, nS)).at[p_scatter].add(contrib_phi)[:M]
        logpi = jnp.log(jnp.clip(pis, 1e-300, None))
        ELr = jnp.einsum("cx,ax->ca", RootAcc, logpi)
        root_prior = jnp.einsum("ica,ca->i", arch_onehot, ELr)       # (nS,)
        phi = phi.at[root].add(root_prior)

        b_new, xi, logZf = field_bp(phi)

        # field HR stats
        def br_field(v):
            return bridge_jax(Qf, lam_f, U_f, Uinv_f, tau[v], xi[v])
        Tf, Nf, _ = jax.vmap(br_field)(jnp.arange(M))
        Wf_c = Tf.sum(0); Uf_c = Nf.sum(0); rootc_c = b_new[root]

        # archetype HR stats
        def br_arch(a, c, v):
            T_, N_, _ = bridge_jax(Qc[a], lam_a[a], U_a[a], Uinv_a[a],
                                   tau[v], EdgeAcc[c, v])
            return T_, N_
        a3, c3, v3 = jnp.meshgrid(jnp.arange(C), jnp.arange(C), jnp.arange(M),
                                  indexing="ij")
        Ta_acv, Na_acv = jax.vmap(jax.vmap(jax.vmap(br_arch)))(a3, c3, v3)
        bw = beta[:, p_gather, :]              # bw[c,v,a]=beta[c,parent(v),a]
        bw = jnp.transpose(bw, (2, 0, 1))      # -> (C_arch,C_class,M)
        bw_m = jnp.where(bw >= 1e-9, bw, 0.0)
        Ta_c = jnp.einsum("acv,acvx->ax", bw_m, Ta_acv)
        Na_c = jnp.einsum("acv,acvxy->axy", bw_m, Na_acv)
        beta_root = beta[:, root, :]
        roota_c = jnp.einsum("cx,ca->ax", RootAcc, beta_root)

        return dict(gamma=gamma, col_ll=col_ll, obj=obj, logZf=logZf,
                    Wf=Wf_c, Uf=Uf_c, rootc=rootc_c,
                    Na=Na_c, Ta=Ta_c, roota=roota_c, gsum=gsum, b_new=b_new)

    res = jax.vmap(estep_cluster)(b_fields, leaf_obs, colmask)

    cm = cmask
    agg = dict(
        obj=((res["obj"] + res["logZf"]) * cm).sum(),
        Wf=(res["Wf"] * cm[:, None]).sum(0),
        Uf=(res["Uf"] * cm[:, None, None]).sum(0),
        rootc=(res["rootc"] * cm[:, None]).sum(0),
        Na=(res["Na"] * cm[:, None, None, None]).sum(0),
        Ta=(res["Ta"] * cm[:, None, None]).sum(0),
        roota=(res["roota"] * cm[:, None, None]).sum(0),
        gsum=(res["gsum"] * cm[:, None]).sum(0),
    )
    return res, agg, (Qf, pif)


full_estep_jit = jax.jit(full_estep)


# ============================================================================
# M-steps (pure JAX)
# ============================================================================
@partial(jax.jit, static_argnums=(4,))
def solve_arch_jax(Na, Ta, roota, pis, steps=40, lr=0.2):
    """Global archetype M-step: softmax gradient ascent (mirrors
    permfield_elbo.solve_arch). Na (C,A,A), Ta (C,A), roota (C,A)."""
    z = jnp.log(jnp.clip(pis, 1e-6, None))

    def body(z, _):
        p = jax.nn.softmax(z, axis=1)
        num = roota + Na.sum(1)
        g_p = num / jnp.maximum(p, 1e-12) - jnp.einsum("cx,xy->cy", Ta, S_J)
        g_z = p * (g_p - (p * g_p).sum(1, keepdims=True))
        z = z + lr * g_z / jnp.maximum(num.sum(1, keepdims=True), 1.0)
        return z, None

    z, _ = jax.lax.scan(body, z, None, length=steps)
    return jax.nn.softmax(z, axis=1)


def solve_field_stationary_jax(rootc, Uf, Wf, pseudo, pi_prev, damp=0.5):
    """Field M-step (star-F81, FREE stationary), mirrors
    fit_single_swap_field.solve_field_stationary: MAP stationary
    pi_field ~ (root occ + incoming usage) + pseudo (identity-favouring Dirichlet),
    damped with pi_prev; F81 rate rho_field = Uf.sum() / sum_i Wf[i](1-pi_field[i]).
    Returns (pi_field, rho_field)."""
    N_in = jnp.clip(Uf, 0.0, None).sum(0)                # incoming usage per state
    occ = jnp.clip(rootc, 0.0, None) + N_in
    pi_new = occ + pseudo; pi_new = pi_new / pi_new.sum()
    pi_new = damp * pi_prev + (1 - damp) * pi_new; pi_new = pi_new / pi_new.sum()
    expo = (jnp.clip(Wf, 0.0, None) * (1.0 - pi_new)).sum()
    rho = jnp.clip(Uf, 0.0, None).sum() / jnp.maximum(expo, 1e-9)
    return pi_new, rho


# ============================================================================
# cluster packing (node-indexed obs, padded to MAX_CLUSTERS x MAX_NODES x Lmax)
# ============================================================================
def pack_clusters(msa, partition, MAX_NODES, nl, Lmax, MAX_CLUSTERS):
    """leaf_obs[ci,j,l] = residue at NODE j for column l of cluster ci (gap=20 for
    non-leaf nodes and pad); colmask valid-column; cmask valid-cluster."""
    leaf_obs = np.full((MAX_CLUSTERS, MAX_NODES, Lmax), 20, np.int64)
    colmask = np.zeros((MAX_CLUSTERS, Lmax))
    cmask = np.zeros(MAX_CLUSTERS)
    for ci, cl in enumerate(partition):
        m = len(cl); cmask[ci] = 1.0
        leaf_obs[ci, :nl, :m] = msa[:nl, cl].astype(np.int64)
        colmask[ci, :m] = 1.0
    return leaf_obs, colmask, cmask


# ============================================================================
# full JAX coordinate-ascent trainer (reuses the validated E-step + M-step)
# ============================================================================
def fit_jax(parent, tau, msa, partition, C, n_iter=25, rho_field0=0.3,
            field_strength=6.0, field_id_frac=0.7, seed=0, MAX_NODES=None):
    """Product-of-trees ELBO with the truncated single-swap field (free stationary),
    pure JAX.  Mirrors fit_single_swap_field.fit's coordinate ascent exactly (same
    init, same M-step order).  Returns dict(pis, pi_field, rho_field, rho, hist)."""
    import fit_single_swap_field as FS
    _, _, arch, pairs = FS.trunc_states(C)
    nS = arch.shape[0]
    N = len(parent); nl = msa.shape[0]; nCl = len(partition)
    if MAX_NODES is None:
        MAX_NODES = N
    tree, _, nl_t = pad_tree(parent, tau, MAX_NODES)
    arch_oh = build_arch_onehot(arch, C)
    Lmax = max(len(p) for p in partition)
    leaf_obs, colmask, cmask = pack_clusters(msa, partition, MAX_NODES, nl_t, Lmax, nCl)
    nmemb = int(sum(len(p) for p in partition))

    rng = np.random.default_rng(seed)
    pis = np.clip(PI0[None, :] * (1 + 0.1 * rng.standard_normal((C, A))), 1e-4, None)
    pis /= pis.sum(1, keepdims=True)
    pseudo = FS.field_prior(C, field_strength, field_id_frac)
    pi_field = pseudo / pseudo.sum()
    rho_field = float(rho_field0); rho = np.ones(C) / C
    b_fields = np.tile(pi_field, (nCl, MAX_NODES, 1))

    pis = jnp.asarray(pis); rho = jnp.asarray(rho)
    pi_field = jnp.asarray(pi_field); rho_field = jnp.asarray(rho_field)
    pseudo_j = jnp.asarray(pseudo)
    b_fields = jnp.asarray(b_fields)
    leaf_obs = jnp.asarray(leaf_obs); colmask = jnp.asarray(colmask)
    cmask = jnp.asarray(cmask)

    hist = []
    for _ in range(n_iter):
        res, agg, _ = full_estep_jit(tree, arch_oh, b_fields, leaf_obs, colmask,
                                     cmask, pis, rho, pi_field, rho_field)
        hist.append(float(agg["obj"]))
        pis = solve_arch_jax(agg["Na"], agg["Ta"], agg["roota"], pis)
        pi_field, rho_field = solve_field_stationary_jax(
            agg["rootc"], agg["Uf"], agg["Wf"], pseudo_j, pi_field)
        rho = jnp.clip(agg["gsum"] / max(nmemb, 1), 1e-6, None); rho = rho / rho.sum()
        b_fields = res["b_new"]
    return dict(pis=np.asarray(pis), pi_field=np.asarray(pi_field),
                rho_field=float(rho_field), rho=np.asarray(rho), hist=hist)


# ============================================================================
# numpy reference: replicate the NEW fit() iteration body exactly
# ============================================================================
def numpy_reference(parent, tau, msa, partition, C, rho_field0=0.3,
                    field_strength=6.0, field_id_frac=0.7, warm_iters=0, seed=0):
    """Run `warm_iters` full EM iterations of the audited numpy reference, then one
    MEASURED E-step + M-step, returning the measured-iteration INPUT state
    (pis/pi_field/rho_field/rho/b_fields) and all measured quantities + M-step
    outputs for parity comparison.  warm_iters>0 diverges per-cluster b_fields and
    drives pi_field off the identity-prior mean (the general case)."""
    import permfield_elbo as PE
    import fit_single_swap_field as FS
    from tkfdp.permfield.hr import eig_rev
    from scipy.linalg import expm as sexpm

    N = len(parent); nl, L = msa.shape
    states, dist, arch, pairs = FS.trunc_states(C)
    nS = len(states)
    pre, post, root, ch = PE.orders(parent)
    branches = [v for v in range(N) if parent[v] >= 0]
    rng = np.random.default_rng(seed)
    pis = np.clip(PI0[None, :] * (1 + 0.1 * rng.standard_normal((C, A))), 1e-4, None)
    pis /= pis.sum(1, keepdims=True)
    pseudo = FS.field_prior(C, field_strength, field_id_frac)
    pi_field = pseudo / pseudo.sum()
    rho_field = float(rho_field0); rho = np.ones(C) / C
    _, pif0 = FS.trunc_field(C, pi_field, rho_field=rho_field)
    b_fields = [np.tile(pif0, (N, 1)) for _ in partition]

    def _em_iter(pis, pi_field, rho_field, rho, b_fields, measure):
        Qc = [PE.gtr_Q(pis[a]) for a in range(C)]
        eig_a = [eig_rev(Qc[a], pis[a]) for a in range(C)]
        Qf, pif = FS.trunc_field(C, pi_field, rho_field=rho_field)
        eig_f = eig_rev(Qf, np.clip(pif, 1e-10, None))
        Pf = {v: np.clip(sexpm(Qf * tau[v]), 1e-300, None) for v in branches}
        Na = [np.zeros((A, A)) for _ in range(C)]; Ta = [np.zeros(A) for _ in range(C)]
        roota = [np.zeros(A) for _ in range(C)]
        Wf = np.zeros(nS); Uf = np.zeros((nS, nS)); rootc = np.zeros(nS)
        gsum = np.zeros(C); nmemb = 0; obj = 0.0
        per_cluster = []
        for ci, cl in enumerate(partition):
            sub = msa[:, cl]
            es = PE.column_estep(parent, tau, sub, C, arch, Qc, pis, rho,
                                 b_fields[ci], pre, post, root, ch, nl)
            phi = PE.field_potentials(parent, C, arch, es, pis, N, nS, ch, root)
            b_fields[ci], xi, logZf = PE.field_bp(parent, Pf, pif, phi,
                                                  pre, post, root, ch)
            obj += es["obj"] + logZf
            PE.accum_field(C, tau, xi, b_fields[ci], Qf, pif, eig_f, root,
                           Wf, Uf, rootc)
            PE.accum_arch(C, arch, es, tau, Qc, eig_a, parent, Na, Ta, roota)
            gsum += es["gamma"].sum(0); nmemb += len(cl)
            if measure:
                per_cluster.append(dict(gamma=es["gamma"].copy(),
                                        col_ll=es["col_ll"].copy(),
                                        obj=float(es["obj"]), logZf=float(logZf)))
        pis_new = PE.solve_arch(C, Na, Ta, roota, pis)
        pi_field_new, rho_field_new = FS.solve_field_stationary(
            rootc, Uf, Wf, pseudo, pi_prev=pi_field)
        rho_new = np.clip(gsum / max(nmemb, 1), 1e-6, None); rho_new /= rho_new.sum()
        meas = dict(obj=obj, gsum=gsum, Wf=Wf, Uf=Uf, rootc=rootc,
                    Na=np.stack(Na), Ta=np.stack(Ta), roota=np.stack(roota),
                    per_cluster=per_cluster, pis_new=pis_new,
                    pi_field_new=pi_field_new, rho_field_new=rho_field_new,
                    rho_new=rho_new)
        return pis_new, pi_field_new, rho_field_new, rho_new, meas

    for _ in range(warm_iters):
        pis, pi_field, rho_field, rho, _ = _em_iter(
            pis, pi_field, rho_field, rho, b_fields, measure=False)

    # snapshot the MEASURED-iteration INPUT (column_estep sees this b_field);
    # _em_iter overwrites b_fields[ci] in place with the field_bp result, so run
    # on a working copy and return the untouched input for JAX parity.
    pis_in, pi_field_in, rho_field_in, rho_in = pis, pi_field, rho_field, rho
    b_input = [b.copy() for b in b_fields]
    b_work = [b.copy() for b in b_fields]
    _, _, _, _, meas = _em_iter(pis, pi_field, rho_field, rho, b_work, measure=True)

    return dict(
        pis=pis_in, pi_field=pi_field_in, rho_field=rho_field_in, rho=rho_in,
        arch=arch, b_fields=b_input, pseudo=pseudo,
        **meas,
    )


# ============================================================================
# validation
# ============================================================================
def _build_case(C=3):
    """Tiny deterministic case: 6-leaf fixed tree + mixed m<=2 clusters, NEW model."""
    import fit_single_swap_field as FS
    parent = np.array([6, 6, 7, 7, 8, 8, 9, 9, 10, 10, -1])
    tau = np.array([0.31, 0.42, 0.27, 0.55, 0.19, 0.63, 0.24, 0.48, 0.37, 0.29, 0.0])
    rng = np.random.default_rng(0)
    _, _, arch, pairs = FS.trunc_states(C)
    pis_t = np.clip(PI0[None, :] * (1 + 0.7 * rng.standard_normal((C, A))), 1e-3, None)
    pis_t /= pis_t.sum(1, keepdims=True)
    pi_field_t = rng.dirichlet(FS.field_prior(C, strength=8.0, id_frac=0.6))
    rho_field_t = 0.5
    rho_t = rng.dirichlet(np.ones(C) * 2.0)
    clusters = [2, 2, 1]
    msa, partition = FS.simulate_trunc(parent, tau, C, pis_t, pi_field_t,
                                       rho_field_t, rho_t, clusters, seed=1)
    return parent, tau, msa, partition


def validate(C=3, tol=1e-6, warm_iters=0, pad_extra=3, pad_clusters=1, label=""):
    import fit_single_swap_field as FS
    parent, tau, msa, partition = _build_case(C)
    nl = msa.shape[0]; N = len(parent); nCl = len(partition)
    print(f"# validation case{label}: C={C}, tree N={N} nl={nl}, "
          f"clusters={[len(p) for p in partition]}, warm_iters={warm_iters}, "
          f"padding -> MAX_NODES={N + pad_extra} MAX_CLUSTERS={nCl + pad_clusters}",
          flush=True)

    ref = numpy_reference(parent, tau, msa, partition, C, warm_iters=warm_iters)
    if warm_iters:
        distinct = not np.allclose(ref["b_fields"][0], ref["b_fields"][1])
        print(f"# warmed state: pi_id={ref['pi_field'][0]:.4f} "
              f"off={1 - ref['pi_field'][0]:.4f} rho_field={ref['rho_field']:.4f} "
              f"rho={np.round(ref['rho'], 4)}  per-cluster b_field distinct={distinct}",
              flush=True)

    _, _, arch, _ = FS.trunc_states(C)
    nS = arch.shape[0]
    MAX_NODES = N + pad_extra
    MAX_CLUSTERS = nCl + pad_clusters
    Lmax = max(len(p) for p in partition)
    tree, _, nl_t = pad_tree(parent, tau, MAX_NODES)
    arch_oh = build_arch_onehot(arch, C)
    leaf_obs, colmask, cmask = pack_clusters(msa, partition, MAX_NODES, nl_t, Lmax,
                                             MAX_CLUSTERS)

    # pack b_fields: real nodes from ref, pad nodes = pi_field; pad clusters = pi_field
    b_fields = np.zeros((MAX_CLUSTERS, MAX_NODES, nS))
    for ci in range(nCl):
        b_fields[ci, :N, :] = ref["b_fields"][ci]
        b_fields[ci, N:, :] = ref["pi_field"]
    for ci in range(nCl, MAX_CLUSTERS):
        b_fields[ci, :, :] = ref["pi_field"]

    res, agg, (Qf_j, pif_j) = full_estep_jit(
        tree, arch_oh, jnp.asarray(b_fields), jnp.asarray(leaf_obs),
        jnp.asarray(colmask), jnp.asarray(cmask),
        jnp.asarray(ref["pis"]), jnp.asarray(ref["rho"]),
        jnp.asarray(ref["pi_field"]), jnp.asarray(ref["rho_field"]))

    # M-step (JAX)
    pis_new_j = solve_arch_jax(agg["Na"], agg["Ta"], agg["roota"],
                               jnp.asarray(ref["pis"]))
    pi_field_new_j, rho_field_new_j = solve_field_stationary_jax(
        agg["rootc"], agg["Uf"], agg["Wf"], jnp.asarray(ref["pseudo"]),
        jnp.asarray(ref["pi_field"]))
    nmemb = sum(len(p) for p in partition)
    rho_new_j = np.clip(np.asarray(agg["gsum"]) / max(nmemb, 1), 1e-6, None)
    rho_new_j /= rho_new_j.sum()

    def md(a, b):
        return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))

    rows = []
    gam_diffs, cll_diffs = [], []
    for ci in range(nCl):
        m = len(partition[ci])
        gj = np.asarray(res["gamma"][ci])[:m]
        cj = np.asarray(res["col_ll"][ci])[:m]
        gam_diffs.append(md(gj, ref["per_cluster"][ci]["gamma"]))
        cll_diffs.append(md(cj, ref["per_cluster"][ci]["col_ll"]))
    rows.append(("E: gamma (per-col, all clusters)", max(gam_diffs)))
    rows.append(("E: col_ll (per-col, all clusters)", max(cll_diffs)))

    obj_c_j = np.asarray(res["obj"])[:nCl]; logZf_j = np.asarray(res["logZf"])[:nCl]
    obj_c_ref = np.array([pc["obj"] for pc in ref["per_cluster"]])
    logZf_ref = np.array([pc["logZf"] for pc in ref["per_cluster"]])
    rows.append(("E: es[obj] (per-cluster)", md(obj_c_j, obj_c_ref)))
    rows.append(("E: logZf (per-cluster)", md(logZf_j, logZf_ref)))
    rows.append(("E: total obj (sum es_obj+logZf)", md(agg["obj"], ref["obj"])))

    rows.append(("E: Wf (field dwell)", md(agg["Wf"], ref["Wf"])))
    rows.append(("E: Uf (field usage)", md(agg["Uf"], ref["Uf"])))
    rows.append(("E: rootc (field root occ)", md(agg["rootc"], ref["rootc"])))

    rows.append(("E: Na (arch usage)", md(agg["Na"], ref["Na"])))
    rows.append(("E: Ta (arch dwell)", md(agg["Ta"], ref["Ta"])))
    rows.append(("E: roota (arch root incidence)", md(agg["roota"], ref["roota"])))

    rows.append(("E: gsum (class mass)", md(agg["gsum"], ref["gsum"])))

    rows.append(("M: pi_field (solve_field_stationary)",
                 md(pi_field_new_j, ref["pi_field_new"])))
    rows.append(("M: rho_field (F81 field rate)",
                 md(rho_field_new_j, ref["rho_field_new"])))
    rows.append(("M: pis (archetype grad step)", md(pis_new_j, ref["pis_new"])))
    rows.append(("M: rho (class weights)", md(rho_new_j, ref["rho_new"])))

    # eig invariants (basis-invariant P + sorted eigenvalues only)
    from scipy.linalg import expm as sexpm
    Qf_ref = FS.trunc_field(C, ref["pi_field"], rho_field=ref["rho_field"])[0]
    Pf_ref = sexpm(Qf_ref * float(tau[0]))
    lam_f, U_f, Uinv_f = eig_rev_jax(Qf_j, jnp.clip(pif_j, 1e-10, None))
    Pf_jax = np.asarray(expm_from_eig(lam_f, U_f, Uinv_f, float(tau[0])))
    rows.append(("eig: Pf=expm(Qf*tau0) [P invariant]", md(Pf_jax, Pf_ref)))
    lam_ref = np.sort(np.linalg.eigvals(Qf_ref).real)
    lam_jax = np.sort(np.asarray(lam_f))
    rows.append(("eig: sorted eigenvalues of Qf", md(lam_jax, lam_ref)))

    print("\n" + "=" * 66)
    print(f"{'quantity':44s} {'max-abs-diff':>14s}")
    print("-" * 66)
    worst_estep = 0.0; allpass = True
    for name, d in rows:
        flag = "" if d <= tol else "  <-- FAIL"
        if flag:
            allpass = False
        if name.startswith("E:"):
            worst_estep = max(worst_estep, d)
        print(f"{name:44s} {d:14.3e}{flag}")
    print("=" * 66)
    print(f"E-step worst max-abs-diff: {worst_estep:.3e}   (gate tol = {tol:.0e})")
    print(f"ALL E-step quantities <= {tol:.0e}: "
          f"{all(d <= tol for n, d in rows if n.startswith('E:'))}")
    print(f"ALL quantities (incl M-step) <= {tol:.0e}: {allpass}")
    return allpass, rows


# ============================================================================
# single-compile-across-families confirmation
# ============================================================================
def _binary_tree(nl):
    """Balanced-ish bifurcating tree: leaves 0..nl-1, internal nl..2nl-2, root last."""
    N = 2 * nl - 1
    parent = -np.ones(N, int)
    nodes = list(range(nl)); nxt = nl
    while len(nodes) > 1:
        a = nodes.pop(0); b = nodes.pop(0)
        parent[a] = nxt; parent[b] = nxt; nodes.append(nxt); nxt += 1
    return parent


def _make_family(C, nl, seed, n_clusters=4):
    import fit_single_swap_field as FS
    rng = np.random.default_rng(seed)
    parent = _binary_tree(nl); N = len(parent)
    tau = rng.uniform(0.15, 0.6, N)
    root = int(np.where(parent < 0)[0][0]); tau[root] = 0.0
    _, _, arch, pairs = FS.trunc_states(C)
    pis_t = np.clip(PI0[None, :] * (1 + 0.5 * rng.standard_normal((C, A))), 1e-3, None)
    pis_t /= pis_t.sum(1, keepdims=True)
    pi_field_t = rng.dirichlet(FS.field_prior(C, 8.0, 0.6))
    rho_t = rng.dirichlet(np.ones(C) * 2.0)
    clusters = ([2, 1] * n_clusters)[:n_clusters]
    msa, partition = FS.simulate_trunc(parent, tau, C, pis_t, pi_field_t, 0.5,
                                       rho_t, clusters, seed=seed + 1)
    return parent, tau, msa, partition


def _pack_family(parent, tau, msa, partition, C, nS, MAX_NODES, MAX_CLUSTERS,
                 Lmax, seed=0):
    import fit_single_swap_field as FS
    nl = msa.shape[0]
    _, _, arch, _ = FS.trunc_states(C)
    tree, _, nl_t = pad_tree(parent, tau, MAX_NODES)
    arch_oh = build_arch_onehot(arch, C)
    leaf_obs, colmask, cmask = pack_clusters(msa, partition, MAX_NODES, nl_t, Lmax,
                                             MAX_CLUSTERS)
    rng = np.random.default_rng(seed)
    pis = np.clip(PI0[None, :] * (1 + 0.1 * rng.standard_normal((C, A))), 1e-4, None)
    pis /= pis.sum(1, keepdims=True)
    pseudo = FS.field_prior(C); pi_field = pseudo / pseudo.sum()
    rho = np.ones(C) / C; rho_field = 0.3
    b_fields = np.tile(pi_field, (MAX_CLUSTERS, MAX_NODES, 1))
    return (tree, arch_oh, jnp.asarray(b_fields), jnp.asarray(leaf_obs),
            jnp.asarray(colmask), jnp.asarray(cmask), jnp.asarray(pis),
            jnp.asarray(rho), jnp.asarray(pi_field), jnp.asarray(rho_field))


def confirm_single_compile(C=3):
    global _TRACE_COUNT
    import fit_single_swap_field as FS
    _, _, arch, _ = FS.trunc_states(C); nS = arch.shape[0]

    pA, tA, msaA, partA = _make_family(C, nl=6, seed=11, n_clusters=4)
    pB, tB, msaB, partB = _make_family(C, nl=11, seed=22, n_clusters=6)
    NA, NB = len(pA), len(pB)
    MAX_NODES = max(NA, NB) + 2
    MAX_CLUSTERS = max(len(partA), len(partB))
    Lmax = 2

    print(f"# single-compile test (C={C}): "
          f"family A = {msaA.shape[0]} leaves / N={NA} / {len(partA)} clusters;  "
          f"family B = {msaB.shape[0]} leaves / N={NB} / {len(partB)} clusters")
    print(f"#   both padded to MAX_NODES={MAX_NODES}, MAX_CLUSTERS={MAX_CLUSTERS}, "
          f"Lmax={Lmax}  (true N differ: {NA} vs {NB})")

    packA = _pack_family(pA, tA, msaA, partA, C, nS, MAX_NODES, MAX_CLUSTERS, Lmax,
                         seed=1)
    packB = _pack_family(pB, tB, msaB, partB, C, nS, MAX_NODES, MAX_CLUSTERS, Lmax,
                         seed=2)

    shapesA = tuple(np.asarray(x).shape for x in packA[2:])
    shapesB = tuple(np.asarray(x).shape for x in packB[2:])
    tree_shapesA = {k: tuple(v.shape) for k, v in packA[0].items()}
    tree_shapesB = {k: tuple(v.shape) for k, v in packB[0].items()}
    print(f"#   arg shapes identical A==B: {shapesA == shapesB and tree_shapesA == tree_shapesB}")

    _TRACE_COUNT = 0
    t0 = time.perf_counter()
    rA = full_estep_jit(*packA)
    jax.block_until_ready(rA)
    t1 = time.perf_counter()
    trace_after_A = _TRACE_COUNT
    rB = full_estep_jit(*packB)
    jax.block_until_ready(rB)
    t2 = time.perf_counter()
    trace_after_B = _TRACE_COUNT

    print(f"#   trace count after 1st call (family A): {trace_after_A}")
    print(f"#   trace count after 2nd call (family B, DIFFERENT tree): {trace_after_B}")
    print(f"#   1st call wall time (INCLUDES XLA compile): {t1 - t0:.3f}s")
    print(f"#   2nd call wall time (different tree, NO compile): {t2 - t1:.4f}s")
    speedup = (t1 - t0) / max(t2 - t1, 1e-9)
    single = (trace_after_B == 1)
    print(f"#   compile-amortised speedup 1st/2nd: {speedup:.1f}x")
    print(f"#   SINGLE COMPILE ACROSS FAMILIES: {single}  "
          f"(one trace served both different-sized trees)")
    return single


def compare_fit(C=3, n_iter=12, tol=1e-6):
    """End-to-end: run the audited numpy fit AND fit_jax on the same tiny
    corpus/init; compare objective trajectory + final params."""
    import fit_single_swap_field as FS
    parent, tau, msa, partition = _build_case(C)
    print(f"# end-to-end fit comparison: C={C}, n_iter={n_iter}, "
          f"clusters={[len(p) for p in partition]}", flush=True)
    ref = FS.fit(parent, tau, msa, C, partition=partition, n_iter=n_iter,
                 rho_field0=0.3, verbose=False)
    jx = fit_jax(parent, tau, msa, partition, C, n_iter=n_iter, rho_field0=0.3, seed=0)
    hist_d = float(np.max(np.abs(np.array(ref["hist"]) - np.array(jx["hist"]))))
    pis_d = float(np.max(np.abs(ref["pis"] - jx["pis"])))
    pif_d = float(np.max(np.abs(ref["pi_field"] - jx["pi_field"])))
    rhof_d = float(abs(ref["rho_field"] - jx["rho_field"]))
    rho_d = float(np.max(np.abs(ref["rho"] - jx["rho"])))
    mono = bool(np.all(np.diff(jx["hist"][3:]) > -1e-6))
    print(f"  obj-trajectory max-abs-diff : {hist_d:.3e}")
    print(f"  final pis       max-abs-diff: {pis_d:.3e}")
    print(f"  final pi_field  max-abs-diff: {pif_d:.3e}")
    print(f"  final rho_field abs-diff    : {rhof_d:.3e}")
    print(f"  final rho       max-abs-diff: {rho_d:.3e}")
    print(f"  JAX obj monotone (it>=3)    : {mono}")
    print(f"  numpy final obj={ref['hist'][-1]:.6f}  jax final obj={jx['hist'][-1]:.6f}")
    ok = max(hist_d, pis_d, pif_d, rhof_d, rho_d) <= tol
    print(f"  END-TO-END trajectory match <= {tol:.0e}: {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="staged E/M parity gate (C=3,4; warm 0,5) + single-compile")
    ap.add_argument("--compare-fit", action="store_true",
                    help="end-to-end numpy-vs-jax fit trajectory comparison")
    ap.add_argument("--confirm-compile", action="store_true",
                    help="only the single-compile-across-families confirmation")
    ap.add_argument("--C", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--warm-iters", type=int, default=None)
    args = ap.parse_args()

    if args.validate:
        results = {}
        for C in (3, 4):
            ok0, _ = validate(C=C, tol=args.tol, warm_iters=0,
                              label=f" [C={C} S1 prior-init pi_field]")
            print()
            ok1, _ = validate(C=C, tol=args.tol, warm_iters=5,
                              label=f" [C={C} S2 warmed, diverged b_field + free pi_field]")
            print()
            results[C] = (ok0, ok1)
        sc3 = confirm_single_compile(C=3)
        print()
        sc4 = confirm_single_compile(C=4)
        print("\n### OVERALL")
        for C in (3, 4):
            print(f"  C={C}: S1 prior-init={results[C][0]}  S2 warmed={results[C][1]}")
        print(f"  single-compile C=3={sc3}  C=4={sc4}")
    elif args.confirm_compile:
        confirm_single_compile(C=args.C)
    elif args.compare_fit:
        compare_fit(C=args.C, tol=args.tol)
    elif args.warm_iters is not None:
        validate(C=args.C, tol=args.tol, warm_iters=args.warm_iters)
    else:
        print("Use --validate (E/M parity gate + single-compile), --compare-fit, "
              "or --confirm-compile.")


if __name__ == "__main__":
    main()
