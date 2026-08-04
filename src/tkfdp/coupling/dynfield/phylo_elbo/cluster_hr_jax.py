"""JAX-batched exact factored cap-2 cluster Holmes-Rubin.

Drop-in, numerically-exact JAX port of the numpy `cluster_hr_exact.pair_tree_hr`
per-cluster / per-archetype accumulation used by the supervised archetype
M-step (`supervised_trainer.exact_hr_per_archetype`).  It reproduces the same
compound up-down message pass over the (L,A,A) exact_cap2 messages, extracting
the per-branch per-archetype residue HR in factored O(L*A^3) form:

  * Delta=0 (no field jump): 20-state GTR HR with the partner residue folded
    into an A x A weight matrix (the generalised single_branch_hr `_genhr`).
  * Delta>=1 (>=1 field jump): 40-state (theta,x) endpoint HR with the partner
    marginalised (row-sum of the outside) and pi-projected (pi-weighted inside),
    minus the Delta=0 GTR part when theta_u == theta_v.

The design DELIBERATELY batches over the config axis -- the (class-pair a,b)
x (rate-bin g) mixture the supervised M-step sums over -- because that is the
dominant cost (~64-576 exact-HR evaluations per contact pair).  All configs of a
family share the SAME tree topology + leaf residues; only the per-field
archetype maps (a1, a2) and the effective field rate rho_chain*rate_g differ.
We vmap the single-tree HR over those configs and reuse the corpus-wide
per-archetype GTR eigendecomposition (K_a of them) across every config.

Correctness bar: matches `cluster_hr_exact.pair_tree_hr` to eigendecomposition
round-off (~1e-9 relative), validated by
`analysis/scripts/validate_cluster_hr_exact.py` (numpy-vs-JAX section) on every
regime INCLUDING the real 254-node PF00013 tree at rho_chain>0 with field jumps
active.  No approximation the numpy version does not have.

f64 is mandatory (the HR needs it); this module force-enables jax_enable_x64.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .exact_cap2 import gtr_Q as _gtr_Q_np
from .exact_cap2_jax import _branch_pair
from .tree import build_tree
from .tree_padded import build_padded_tree, compute_node_levels

A_DIM = 20


def _leaf_msg_pair(obs, L, A):
    """(L,A,A) compound leaf message; a residue is a gap (uninformative -> all
    ones) when its observed letter is < 0 OR >= A, matching numpy
    `cluster_hr_exact._pair_leaf_msg`."""
    x, y = obs[0], obs[1]
    gx = (x < 0) | (x >= A)
    gy = (y < 0) | (y >= A)
    ex = jnp.where(gx, jnp.ones(A), jax.nn.one_hot(jnp.clip(x, 0, A - 1), A))
    ey = jnp.where(gy, jnp.ones(A), jax.nn.one_hot(jnp.clip(y, 0, A - 1), A))
    M = jnp.outer(ex, ey)
    return jnp.broadcast_to(M[None], (L, A, A))


# ===========================================================================
#  Reversible-chain HR primitives (JAX; mirror check_cluster_hr / cluster_hr_exact)
# ===========================================================================
def reversible_eigh_jax(Q, p):
    sqrt = jnp.sqrt(p)
    inv = 1.0 / jnp.clip(sqrt, 1e-300, None)
    Qs = sqrt[:, None] * Q * inv[None, :]
    Qs = 0.5 * (Qs + Qs.T)
    lam, V = jnp.linalg.eigh(Qs)
    return lam, V, sqrt, inv


def _I_kl_jax(lam, t):
    e = jnp.exp(lam * t)
    d = lam[:, None] - lam[None, :]
    safe = jnp.where(jnp.abs(d) < 1e-12, 1.0, d)
    off = (e[:, None] - e[None, :]) / safe
    return jnp.where(jnp.abs(d) < 1e-12, t * e[:, None], off)


def _Pt_jax(lam, V, sqrt, inv, t):
    e = jnp.exp(lam * t)
    return inv[:, None] * (V * e[None, :]) @ V.T * sqrt[None, :]


def _Pt_theta(lam, V, sqrt, inv, t):
    """Per-field-state transition matrices: lam (L,A), V (L,A,A) ... -> (L,A,A)."""
    return jax.vmap(lambda lm, Vv, sq, iv: _Pt_jax(lm, Vv, sq, iv, t))(
        lam, V, sqrt, inv)


def _genhr(W, t, lam, V, sqrt, inv, scaled):
    """Generalised single-branch HR with a WEIGHT MATRIX W[i,j] (A,A).
    Returns (EN (A,A) off-diagonal, dwell (A,)).  All args for ONE (A-dim) chain."""
    Ikl = _I_kl_jax(lam, t)
    Wtil = (inv[:, None] * W) * sqrt[None, :]
    M = (V.T @ Wtil @ V) * Ikl
    VMV = V @ M @ V.T
    EN = scaled * VMV
    EN = EN - jnp.diag(jnp.diagonal(EN))
    dwell = jnp.diagonal(VMV)
    return EN, dwell


def _field_kernels_jax(rho, rho_chain, tau):
    L = rho.shape[0]
    g = jnp.exp(-rho_chain * tau)
    P_theta = g * jnp.eye(L) + (1.0 - g) * rho[None, :]
    beta = jnp.exp(-rho_chain * (1.0 - rho) * tau)
    J = P_theta - jnp.diag(beta)
    return beta, J


# ===========================================================================
#  Shared corpus-wide per-archetype GTR eigendecomposition (K_a of them)
# ===========================================================================
def build_shared_arch_eig(pi_arch, S):
    """Per-archetype GTR generator + reversible eigh, computed ONCE and reused
    across every config.  Returns a dict of stacked (K_a, ...) jnp arrays."""
    pi_arch = np.asarray(pi_arch, np.float64)
    S = np.asarray(S, np.float64)
    Ka, A = pi_arch.shape
    gtr_Q = np.stack([_gtr_Q_np(pi_arch[k], S) for k in range(Ka)])   # (Ka,A,A)
    gtr_Q_j = jnp.asarray(gtr_Q)

    def _eig(Q, p):
        return reversible_eigh_jax(Q, p)
    lam, V, sqrt, inv = jax.vmap(_eig)(gtr_Q_j, jnp.asarray(pi_arch))
    scaled = gtr_Q_j * (sqrt[:, :, None] * inv[:, None, :])
    # off-diagonal GTR generator (diagonal zeroed) -- for building the 40-state
    d = jnp.diagonal(gtr_Q_j, axis1=1, axis2=2)                       # (Ka,A)
    offdiag = gtr_Q_j - jnp.eye(A)[None] * d[:, None, :]
    return {
        'pi_arch': jnp.asarray(pi_arch), 'rho': None,
        'gtr_Q': gtr_Q_j, 'gtr_offdiag': offdiag,
        'lam': lam, 'V': V, 'sqrt': sqrt, 'inv': inv, 'scaled': scaled,
        'Ka': Ka, 'A': A,
    }


def _build_q40(Q_an, offdiag_an, pi_an, rho, rho_chain):
    """Compound SINGLETON generator on (theta,x): field jump theta->theta'
    resamples x ~ pi^{a_n(theta')}; residue GTR within field.  L*A-state.
    Q_an/offdiag_an (L,A,A) gathered per-theta; pi_an (L,A)."""
    L, A, _ = Q_an.shape
    rows = []
    for t in range(L):
        cols = []
        for tp in range(L):
            if t == tp:
                cols.append(offdiag_an[t])
            else:
                cols.append(rho_chain * rho[tp]
                            * jnp.broadcast_to(pi_an[tp][None, :], (A, A)))
        rows.append(jnp.concatenate(cols, axis=1))                   # (A, L*A)
    Q = jnp.concatenate(rows, axis=0)                                # (LA, LA)
    d = -Q.sum(axis=1)
    Q = Q + jnp.diag(d)
    p = (rho[:, None] * pi_an).reshape(-1)
    return Q, p


def _single_branch_EN(out_vec, in_vec, t, lam, V, sqrt, inv, scaled):
    """single_branch_hr EN (off-diagonal) for a separable out (x) in weight."""
    Ikl = _I_kl_jax(lam, t)
    g = (out_vec * inv) @ V
    h = (in_vec * sqrt) @ V
    M = g[:, None] * Ikl * h[None, :]
    VMV = V @ M @ V.T
    EN = scaled * VMV
    EN = EN - jnp.diag(jnp.diagonal(EN))
    return EN


# ===========================================================================
#  Per-branch factored HR core (Delta=0 + Delta>=1 + optional jump mass)
# ===========================================================================
def _delta1_residue(O_u, I_v, tau, a_n, other_pi,
                    e40, gtr_lam, gtr_V, gtr_sqrt, gtr_inv, gtr_scaled,
                    beta, Ka, rn):
    """Delta>=1 (>=1 field-jump) per-residue HR via the 40-state (theta,x) chain.
    O_u, I_v (L,A,A) branch outside(parent)/inside(child); a_n (L,) archetype map
    for THIS residue; other_pi (L,A) = partner archetype stationary per field.
    e40 = (lam,V,sqrt,inv,scaled) of the 40-state chain.  gtr_* gathered by a_n.
    Returns (Nk40 (Ka,A,A), Tk40 (Ka,A))."""
    lam40, V40, sqrt40, inv40, scaled40 = e40
    L = a_n.shape[0]
    A = other_pi.shape[1]
    Ikl40 = _I_kl_jax(lam40, tau)
    onehot = jax.nn.one_hot(a_n, Ka)                                 # (L,Ka)
    Nk40 = jnp.zeros((Ka, A, A))
    Tk40 = jnp.zeros((Ka, A))
    for tu in range(L):
        outv = O_u[tu].sum(axis=1) if rn == 1 else O_u[tu].sum(axis=0)  # (A,)
        bu = slice(tu * A, tu * A + A)
        g = (outv * inv40[bu]) @ V40[bu]                            # (LA,)
        for tv in range(L):
            inv_end = (I_v[tv] @ other_pi[tv]) if rn == 1 else (other_pi[tv] @ I_v[tv])
            bv = slice(tv * A, tv * A + A)
            h = (inv_end * sqrt40[bv]) @ V40[bv]                    # (LA,)
            M = g[:, None] * Ikl40 * h[None, :]
            VMV = V40 @ M @ V40.T
            EN40 = scaled40 * VMV
            EN40 = EN40 - jnp.diag(jnp.diagonal(EN40))
            dw40 = jnp.diagonal(VMV)
            for t in range(L):
                blk = EN40[t * A:t * A + A, t * A:t * A + A]
                Nk40 = Nk40 + onehot[t][:, None, None] * blk[None]
                Tk40 = Tk40 + onehot[t][:, None] * dw40[t * A:t * A + A][None]
            if tu == tv:
                W0 = jnp.outer(outv, inv_end)
                EN0, dw0 = _genhr(W0, tau, gtr_lam[tu], gtr_V[tu], gtr_sqrt[tu],
                                  gtr_inv[tu], gtr_scaled[tu])
                Nk40 = Nk40 - beta[tu] * onehot[tu][:, None, None] * EN0[None]
                Tk40 = Tk40 - beta[tu] * onehot[tu][:, None] * dw0[None]
    return Nk40, Tk40


def _branch_hr_core(O_u, I_v, tau, P1, P2, beta, J,
                    a1, a2, e1, e2, pi1, pi2, rho, rho_chain,
                    e40_1, e40_2, eigf, Ka, want_jumps, delta1_enabled=True):
    """Unnormalised per-archetype residue HR for ONE branch.
    e1/e2 = per-residue gathered gtr eig (lam,V,sqrt,inv,scaled) (each L-leading).
    Returns (Nk_b (Ka,A,A), Tk_b (Ka,A), jmass scalar)."""
    A = pi1.shape[1]
    lam1, V1, sqrt1, inv1, scaled1 = e1
    lam2, V2, sqrt2, inv2, scaled2 = e2

    # ---- Delta = 0 : frozen field, residues independent -------------------
    W1 = beta[:, None, None] * jnp.matmul(jnp.matmul(O_u, P2),
                                          jnp.transpose(I_v, (0, 2, 1)))
    W2 = beta[:, None, None] * jnp.matmul(jnp.matmul(jnp.transpose(O_u, (0, 2, 1)),
                                                     P1), I_v)
    EN1, dw1 = jax.vmap(_genhr, in_axes=(0, None, 0, 0, 0, 0, 0))(
        W1, tau, lam1, V1, sqrt1, inv1, scaled1)
    EN2, dw2 = jax.vmap(_genhr, in_axes=(0, None, 0, 0, 0, 0, 0))(
        W2, tau, lam2, V2, sqrt2, inv2, scaled2)
    oh1 = jax.nn.one_hot(a1, Ka)                                     # (L,Ka)
    oh2 = jax.nn.one_hot(a2, Ka)
    Nk_b = (jnp.einsum('lk,lab->kab', oh1, EN1)
            + jnp.einsum('lk,lab->kab', oh2, EN2))
    Tk_b = (jnp.einsum('lk,la->ka', oh1, dw1)
            + jnp.einsum('lk,la->ka', oh2, dw2))

    # ---- Delta >= 1 : reset/renewal decouples the residues -----------------
    if not delta1_enabled:
        return Nk_b, Tk_b, 0.0
    delta_scale = jnp.where(rho_chain > 0, 1.0, 0.0)
    Nk1, Tk1 = _delta1_residue(O_u, I_v, tau, a1, pi2, e40_1,
                               lam1, V1, sqrt1, inv1, scaled1, beta, Ka, 1)
    Nk2, Tk2 = _delta1_residue(O_u, I_v, tau, a2, pi1, e40_2,
                               lam2, V2, sqrt2, inv2, scaled2, beta, Ka, 2)
    Nk_b = Nk_b + delta_scale * (Nk1 + Nk2)
    Tk_b = Tk_b + delta_scale * (Tk1 + Tk2)

    # ---- field-jump mass (residue-independent, config-weighted) -----------
    jmass = 0.0
    if want_jumps:
        L = rho.shape[0]
        lamf, Vf, sqrtf, invf, scaledf = eigf
        m_c = jnp.einsum('la,lab,lb->l', pi1, I_v, pi2)             # (L,)
        Ou_mass = O_u.sum(axis=(1, 2))                             # (L,)
        acc = 0.0
        eyeL = jnp.eye(L)
        for tu in range(L):
            for tv in range(L):
                ENf = _single_branch_EN(eyeL[tu], eyeL[tv], tau,
                                        lamf, Vf, sqrtf, invf, scaledf)
                ejmp_num = ENf.sum() - jnp.trace(ENf)
                Jw = J[tu, tv]
                ejumps = jnp.where(Jw > 1e-300, ejmp_num / jnp.where(Jw > 1e-300, Jw, 1.0), 0.0)
                acc = acc + Ou_mass[tu] * Jw * m_c[tv] * ejumps
        jmass = delta_scale * acc
    return Nk_b, Tk_b, jmass


# ===========================================================================
#  Single-tree factored pair HR (compound up-down) -- ONE config
# ===========================================================================
def _make_hr_one(D, Ka, A, L, want_jumps, delta1_enabled=True):
    """Build a per-config single-tree HR function.  BOTH the tree structure AND
    the corpus-wide eig arrays are RUNTIME arguments, so ONE compiled executable
    is reused across (a) every M-step (pi_archetype changes) and (b) every tree
    whose padded shape falls in the same geomspaced size bin (only the depth `D`
    and the per-level slot counts -- fixed within a bin -- are static).  This
    turns O(#families) compiles into O(#size-bins) ~ O(log max_size).

    Both the inside (bottom-up) and outside (top-down) message passes are ROLLED
    into lax.scan over the (uniform-N_max) depth axis, so the compiled graph is
    O(1) in depth instead of O(D) -- turning ~30 min cold compiles into seconds.

    hr_one(leaf_obs (N_max,2), a1 (L,), a2 (L,), rho_chain_eff,
           child_pos (D,N_max,2), child_branch (D,N_max,2), is_identity (D,N_max),
           slot_mask (D,N_max), root_slot,
           pi_arch, rho, gtr_Q, gtr_offdiag, glam, gV, gsqrt, ginv, gscaled)
      -> (Nk (Ka,A,A), Tk (Ka,A), jumps).  Padded slots (slot_mask==0) and phantom
    identity edges (tau==0) contribute the IDENTITY to the HR (zero
    dwell/counts/jumps) and are masked out of the message scatter, so bin padding
    cannot bias the result.

    `delta1_enabled=False` STATICALLY skips the 40-state Delta>=1 + jump paths
    (valid only when every config has rho_chain_eff == 0, e.g. the invariant
    field-rate bin); it makes those configs as cheap as the numpy early-return."""

    def hr_one(leaf_obs, a1, a2, rho_chain,
               child_pos, child_branch, is_identity, slot_mask, root_slot,
               pi_arch, rho, gtr_Q, gtr_offdiag, glam, gV, gsqrt, ginv, gscaled):
        N_max = leaf_obs.shape[0]
        # per-config gathered gtr eig + partner stationary
        pi1 = pi_arch[a1]; pi2 = pi_arch[a2]                        # (L,A)
        e1 = (glam[a1], gV[a1], gsqrt[a1], ginv[a1], gscaled[a1])
        e2 = (glam[a2], gV[a2], gsqrt[a2], ginv[a2], gscaled[a2])
        Q1 = gtr_Q[a1]; Q2 = gtr_Q[a2]
        off1 = gtr_offdiag[a1]; off2 = gtr_offdiag[a2]

        # 40-state (theta,x) chains (per config, per residue)
        if delta1_enabled:
            Q40_1, p40_1 = _build_q40(Q1, off1, pi1, rho, rho_chain)
            Q40_2, p40_2 = _build_q40(Q2, off2, pi2, rho, rho_chain)
            l1, Vv1, s1, i1 = reversible_eigh_jax(Q40_1, p40_1)
            l2, Vv2, s2, i2 = reversible_eigh_jax(Q40_2, p40_2)
            e40_1 = (l1, Vv1, s1, i1, Q40_1 * (s1[:, None] * i1[None, :]))
            e40_2 = (l2, Vv2, s2, i2, Q40_2 * (s2[:, None] * i2[None, :]))
        else:
            e40_1 = e40_2 = None

        # 2-state field chain (for jump count)
        if want_jumps and delta1_enabled:
            Qf = rho_chain * jnp.broadcast_to(rho[None, :], (L, L))
            Qf = Qf - jnp.diag(jnp.diagonal(Qf))
            Qf = Qf - jnp.diag(Qf.sum(axis=1))
            lf, Vf, sf, iff = reversible_eigh_jax(Qf, rho)
            eigf = (lf, Vf, sf, iff, Qf * (sf[:, None] * iff[None, :]))
        else:
            eigf = None

        # p_comp (root prior / reversibility reference) (L,A,A)
        p_comp = rho[:, None, None] * (pi1[:, :, None] * pi2[:, None, :])

        # ---- inside pass (bottom-up) via lax.scan over levels ----------------
        leaf_msgs = jax.vmap(lambda o: _leaf_msg_pair(o, L, A))(leaf_obs)  # (N_max,L,A,A)

        def _inside_level(prev, x):
            cp, cb, idn = x                                  # (N_max,2),(N_max,2),(N_max,)

            def _combine(cp_i, cb_i, id_i):
                left = prev[cp_i[0]]; right = prev[cp_i[1]]
                tauL = cb_i[0]; tauR = cb_i[1]
                P1L = _Pt_theta(*e1[:4], tauL); P2L = _Pt_theta(*e2[:4], tauL)
                P1R = _Pt_theta(*e1[:4], tauR); P2R = _Pt_theta(*e2[:4], tauR)
                bL, JL = _field_kernels_jax(rho, rho_chain, tauL)
                bR, JR = _field_kernels_jax(rho, rho_chain, tauR)
                mL = _branch_pair(left, P1L, P2L, pi1, pi2, bL, JL)
                mR = _branch_pair(right, P1R, P2R, pi1, pi2, bR, JR)
                out = jnp.where(id_i > 0.5, left, mL * mR)
                s = jnp.maximum(out.sum(), 1e-300)
                return out / s
            new = jax.vmap(_combine)(cp, cb, idn)            # (N_max,L,A,A)
            return new, new
        _, inside_up = jax.lax.scan(
            _inside_level, leaf_msgs, (child_pos, child_branch, is_identity))
        # inside at level l: inside_by_level[l]; [leaf, then levels 1..D]
        inside_by_level = jnp.concatenate([leaf_msgs[None], inside_up], axis=0)  # (D+1,N_max,L,A,A)

        # ---- outside pass (top-down) + per-branch HR via reverse lax.scan ----
        O_init = jnp.zeros((N_max, L, A, A)).at[root_slot].set(p_comp)

        def _outside_level(O_cur, x):
            cp, cb, idn, sm, inside_child = x

            def _proc(O_s, cp_i, cb_i, id_i, sm_i):
                cL = cp_i[0]; cR = cp_i[1]
                tauL = cb_i[0]; tauR = cb_i[1]
                inL = inside_child[cL]; inR = inside_child[cR]
                P1L = _Pt_theta(*e1[:4], tauL); P2L = _Pt_theta(*e2[:4], tauL)
                P1R = _Pt_theta(*e1[:4], tauR); P2R = _Pt_theta(*e2[:4], tauR)
                bL, JL = _field_kernels_jax(rho, rho_chain, tauL)
                bR, JR = _field_kernels_jax(rho, rho_chain, tauR)
                mvL = _branch_pair(inL, P1L, P2L, pi1, pi2, bL, JL)
                mvR = _branch_pair(inR, P1R, P2R, pi1, pi2, bR, JR)
                combinedL = O_s * mvR
                combinedR = O_s * mvL

                hr_mask = sm_i * (1.0 - id_i)
                NkL, TkL, jmL = _branch_hr_core(
                    combinedL, inL, tauL, P1L, P2L, bL, JL, a1, a2, e1, e2,
                    pi1, pi2, rho, rho_chain, e40_1, e40_2, eigf, Ka,
                    want_jumps and delta1_enabled, delta1_enabled)
                ZL = (combinedL * mvL).sum()
                NkR, TkR, jmR = _branch_hr_core(
                    combinedR, inR, tauR, P1R, P2R, bR, JR, a1, a2, e1, e2,
                    pi1, pi2, rho, rho_chain, e40_1, e40_2, eigf, Ka,
                    want_jumps and delta1_enabled, delta1_enabled)
                ZR = (combinedR * mvR).sum()
                safeZL = jnp.where(ZL > 1e-300, ZL, 1.0)
                safeZR = jnp.where(ZR > 1e-300, ZR, 1.0)
                Nk_add = hr_mask * (jnp.where(ZL > 1e-300, NkL / safeZL, 0.0)
                                    + jnp.where(ZR > 1e-300, NkR / safeZR, 0.0))
                Tk_add = hr_mask * (jnp.where(ZL > 1e-300, TkL / safeZL, 0.0)
                                    + jnp.where(ZR > 1e-300, TkR / safeZR, 0.0))
                j_add = hr_mask * (jnp.where(ZL > 1e-300, jmL / safeZL, 0.0)
                                   + jnp.where(ZR > 1e-300, jmR / safeZR, 0.0))

                ceffL = jnp.where(p_comp > 1e-300, combinedL / jnp.where(p_comp > 1e-300, p_comp, 1.0), 0.0)
                ceffR = jnp.where(p_comp > 1e-300, combinedR / jnp.where(p_comp > 1e-300, p_comp, 1.0), 0.0)
                oL = p_comp * _branch_pair(ceffL, P1L, P2L, pi1, pi2, bL, JL)
                oR = p_comp * _branch_pair(ceffR, P1R, P2R, pi1, pi2, bR, JR)
                contribL = sm_i * jnp.where(id_i > 0.5, O_s, oL)
                contribR = sm_i * jnp.where(id_i > 0.5, jnp.zeros_like(oR), oR)
                return Nk_add, Tk_add, j_add, contribL, contribR, cL, cR

            Nk_add, Tk_add, j_add, contribL, contribR, cLs, cRs = jax.vmap(_proc)(
                O_cur, cp, cb, idn, sm)
            O_next = jnp.zeros((N_max, L, A, A))
            O_next = O_next.at[cLs].add(contribL).at[cRs].add(contribR)
            return O_next, (Nk_add.sum(0), Tk_add.sum(0), j_add.sum(0))

        _, (Nk_lv, Tk_lv, j_lv) = jax.lax.scan(
            _outside_level, O_init,
            (child_pos, child_branch, is_identity, slot_mask, inside_by_level[:D]),
            reverse=True)
        Nk = Nk_lv.sum(0); Tk = Tk_lv.sum(0); jumps = j_lv.sum()
        return Nk, Tk, jumps

    return hr_one


# ===========================================================================
#  Single-tree SINGLETON (m=1) HR -- 40-state (theta,x) reversible tree HR
# ===========================================================================
def _single_branch_hr40(out_vec, in_vec, t, lam, V, sqrt, inv, scaled):
    """Standard reversible single-branch HR on the 40-state chain.  Returns
    (EN (LA,LA) off-diagonal, dwell (LA,))."""
    Ikl = _I_kl_jax(lam, t)
    g = (out_vec * inv) @ V
    h = (in_vec * sqrt) @ V
    M = g[:, None] * Ikl * h[None, :]
    VMV = V @ M @ V.T
    EN = scaled * VMV
    EN = EN - jnp.diag(jnp.diagonal(EN))
    dwell = jnp.diagonal(VMV)
    return EN, dwell


def _make_single_hr_one(D, Ka, A, L, want_jumps):
    """Per-config singleton (m=1) HR.  Tree structure AND eig arrays are RUNTIME
    args; the depth up-down is ROLLED into lax.scan (O(1) compile).  Maps
    (leaf_obs (N_max,) int, a_n (L,) int, rho_chain, child_pos (D,N_max,2),
    child_branch (D,N_max,2), is_identity (D,N_max), slot_mask (D,N_max),
    root_slot, pi_arch, rho, gtr_Q, gtr_offdiag) -> (Nk (Ka,A,A), Tk (Ka,A),
    jumps) via the exact 40-state (theta,x) reversible tree HR."""

    def hr_one(leaf_obs, a_n, rho_chain,
               child_pos, child_branch, is_identity, slot_mask, root_slot,
               pi_arch, rho, gtr_Q, gtr_offdiag):
        N_max = leaf_obs.shape[0]
        pi_an = pi_arch[a_n]
        Q40, p40 = _build_q40(gtr_Q[a_n], gtr_offdiag[a_n], pi_an, rho, rho_chain)
        lam, V, sqrt, inv = reversible_eigh_jax(Q40, p40)
        scaled = Q40 * (sqrt[:, None] * inv[None, :])
        onehot = jax.nn.one_hot(a_n, Ka)                             # (L,Ka)

        def _leaf(x):
            gx = (x < 0) | (x >= A)
            e = jnp.where(gx, jnp.ones(A), jax.nn.one_hot(jnp.clip(x, 0, A - 1), A))
            return jnp.broadcast_to(e[None], (L, A)).reshape(-1)      # (LA,)
        leaf_msgs = jax.vmap(_leaf)(leaf_obs)                         # (N_max,LA)

        def _inside_level(prev, x):
            cp, cb, idn = x

            def _combine(cp_i, cb_i, id_i):
                left = prev[cp_i[0]]; right = prev[cp_i[1]]
                PL = _Pt_jax(lam, V, sqrt, inv, cb_i[0])
                PR = _Pt_jax(lam, V, sqrt, inv, cb_i[1])
                out = jnp.where(id_i > 0.5, left, (PL @ left) * (PR @ right))
                s = jnp.maximum(out.sum(), 1e-300)
                return out / s
            new = jax.vmap(_combine)(cp, cb, idn)
            return new, new
        _, inside_up = jax.lax.scan(
            _inside_level, leaf_msgs, (child_pos, child_branch, is_identity))
        inside_by_level = jnp.concatenate([leaf_msgs[None], inside_up], axis=0)

        O_init = jnp.zeros((N_max, L * A)).at[root_slot].set(p40)

        def _outside_level(O_cur, x):
            cp, cb, idn, sm, inside_child = x

            def _proc(O_s, cp_i, cb_i, id_i, sm_i):
                cL = cp_i[0]; cR = cp_i[1]
                tauL = cb_i[0]; tauR = cb_i[1]
                inL = inside_child[cL]; inR = inside_child[cR]
                PL = _Pt_jax(lam, V, sqrt, inv, tauL)
                PR = _Pt_jax(lam, V, sqrt, inv, tauR)
                mvL = PL @ inL; mvR = PR @ inR
                combinedL = O_s * mvR; combinedR = O_s * mvL
                hr_mask = sm_i * (1.0 - id_i)
                ENL, dwL = _single_branch_hr40(combinedL, inL, tauL, lam, V, sqrt, inv, scaled)
                ZL = combinedL @ mvL
                ENR, dwR = _single_branch_hr40(combinedR, inR, tauR, lam, V, sqrt, inv, scaled)
                ZR = combinedR @ mvR

                def _agg(EN, dw, Z):
                    safe = jnp.where(Z > 1e-300, Z, 1.0)
                    Nk_l = jnp.zeros((Ka, A, A)); Tk_l = jnp.zeros((Ka, A))
                    jmp = EN.sum()
                    for t in range(L):
                        blk = EN[t * A:t * A + A, t * A:t * A + A]
                        Nk_l = Nk_l + onehot[t][:, None, None] * blk[None]
                        Tk_l = Tk_l + onehot[t][:, None] * dw[t * A:t * A + A][None]
                        jmp = jmp - blk.sum()
                    good = (Z > 1e-300)
                    return (jnp.where(good, Nk_l / safe, 0.0),
                            jnp.where(good, Tk_l / safe, 0.0),
                            jnp.where(good, jmp / safe, 0.0))
                NkL, TkL, jL = _agg(ENL, dwL, ZL)
                NkR, TkR, jR = _agg(ENR, dwR, ZR)
                Nk_add = hr_mask * (NkL + NkR)
                Tk_add = hr_mask * (TkL + TkR)
                j_add = hr_mask * (jL + jR)
                oL = combinedL @ PL; oR = combinedR @ PR
                contribL = sm_i * jnp.where(id_i > 0.5, O_s, oL)
                contribR = sm_i * jnp.where(id_i > 0.5, jnp.zeros_like(oR), oR)
                return Nk_add, Tk_add, j_add, contribL, contribR, cL, cR

            Nk_add, Tk_add, j_add, contribL, contribR, cLs, cRs = jax.vmap(_proc)(
                O_cur, cp, cb, idn, sm)
            O_next = jnp.zeros((N_max, L * A))
            O_next = O_next.at[cLs].add(contribL).at[cRs].add(contribR)
            return O_next, (Nk_add.sum(0), Tk_add.sum(0), j_add.sum(0))

        _, (Nk_lv, Tk_lv, j_lv) = jax.lax.scan(
            _outside_level, O_init,
            (child_pos, child_branch, is_identity, slot_mask, inside_by_level[:D]),
            reverse=True)
        return Nk_lv.sum(0), Tk_lv.sum(0), j_lv.sum()

    return hr_one


def _get_single_hr_vmap(bucket, D, Ka, A, L, want_jumps):
    key = ('single', bucket, D, Ka, A, L, want_jumps)
    if key not in _HR_FN_CACHE:
        hr_one = _make_single_hr_one(D, Ka, A, L, want_jumps)
        in_axes = (0, 0, 0, 0, 0, 0, 0, 0) + (None,) * 4
        _HR_FN_CACHE[key] = jax.jit(jax.vmap(hr_one, in_axes=in_axes))
    return _HR_FN_CACHE[key]


def single_tree_hr_bucketed(ptrees, leaf_obs_list, a_n_arr, rc_arr, shared,
                            want_jumps=False, b_chunk=128):
    """Exact singleton (m=1) HR over a heterogeneous batch of (column, family).
    Bin-bucketed: one compile per size bin.  ptrees: list of build_ptree dicts;
    leaf_obs_list: list of (N_bucket,) arrays; a_n_arr (n,L); rc_arr (n,).
    Returns (Nk (n,Ka,A,A), Tk (n,Ka,A), jumps (n,))."""
    n = len(ptrees)
    Ka = int(shared['Ka']); A = int(shared['A'])
    L = int(np.asarray(shared['rho_np']).shape[0])
    sargs = (shared['pi_arch'], jnp.asarray(shared['rho_np']),
             shared['gtr_Q'], shared['gtr_offdiag'])
    Nk = np.zeros((n, Ka, A, A)); Tk = np.zeros((n, Ka, A)); J = np.zeros(n)
    rc_np = np.asarray(rc_arr, np.float64)
    groups = defaultdict(list)
    for i, pt in enumerate(ptrees):
        groups[pt['bucket']].append(i)
    for bucket, idxs in groups.items():
        D = bucket[1]
        N_max = _bucket_n_max([ptrees[i]['pt'] for i in idxs], D)
        vfn = _get_single_hr_vmap(bucket, D, Ka, A, L, want_jumps)
        for c0 in range(0, len(idxs), b_chunk):
            sub = idxs[c0:c0 + b_chunk]
            npad = b_chunk - len(sub)
            sub_p = sub + [sub[0]] * npad
            padded = [ptrees[i]['pt'] for i in sub_p]
            cp, cb, idn, sm, root = _struct_tensors(padded, D, N_max)
            lo = jnp.asarray(_pad_leaf_obs([leaf_obs_list[i] for i in sub_p], N_max, 1)[:, :, 0])
            an = jnp.asarray(np.stack([np.asarray(a_n_arr[i]) for i in sub_p]).astype(np.int32))
            rc = jnp.asarray(rc_np[sub_p])
            nk, tk, jj = vfn(lo, an, rc, cp, cb, idn, sm, root, *sargs)
            # ONE device->host transfer per chunk (not per config)
            nk_h = np.asarray(nk); tk_h = np.asarray(tk); jj_h = np.asarray(jj)
            Nk[sub] = nk_h[:len(sub)]; Tk[sub] = tk_h[:len(sub)]; J[sub] = jj_h[:len(sub)]
    return Nk, Tk, J


def single_tree_hr_bucketed_flat(fptrees, leaf_obs_list, a_n_arr, rc_arr, shared,
                                 want_jumps=False, b_chunk=128):
    """Dense postorder-flat singleton HR over a heterogeneous batch.  Drop-in for
    `single_tree_hr_bucketed` but consumes `build_flat_ptree` dicts and scans the
    2n-1 node axis (no D x N_max padding).  Buckets by node-count bin (one compile
    per bin).Returns (Nk (n,Ka,A,A), Tk (n,Ka,A), jumps (n,))."""
    n = len(fptrees)
    Ka = int(shared['Ka']); A = int(shared['A'])
    L = int(np.asarray(shared['rho_np']).shape[0])
    sargs = (shared['pi_arch'], jnp.asarray(shared['rho_np']),
             shared['gtr_Q'], shared['gtr_offdiag'])
    Nk = np.zeros((n, Ka, A, A)); Tk = np.zeros((n, Ka, A)); J = np.zeros(n)
    rc_np = np.asarray(rc_arr, np.float64)
    groups = defaultdict(list)
    for i, ft in enumerate(fptrees):
        groups[ft['bucket']].append(i)
    for bucket, idxs in groups.items():
        M_bucket = int(bucket)
        vfn = _get_single_hr_vmap_flat(M_bucket, Ka, A, L, want_jumps)
        for c0 in range(0, len(idxs), b_chunk):
            sub = idxs[c0:c0 + b_chunk]
            npad = b_chunk - len(sub)
            sub_p = sub + [sub[0]] * npad
            cp, cb, isl, msk, root = _flat_struct_tensors(
                [fptrees[i] for i in sub_p], M_bucket)
            lo = jnp.asarray(_flat_pad_leaf_obs(
                [leaf_obs_list[i] for i in sub_p], M_bucket, 1)[:, :, 0])
            an = jnp.asarray(np.stack([np.asarray(a_n_arr[i]) for i in sub_p]).astype(np.int32))
            rc = jnp.asarray(rc_np[sub_p])
            nk, tk, jj = vfn(lo, an, rc, cp, cb, isl, msk, root, *sargs)
            nk_h = np.asarray(nk); tk_h = np.asarray(tk); jj_h = np.asarray(jj)
            Nk[sub] = nk_h[:len(sub)]; Tk[sub] = tk_h[:len(sub)]; J[sub] = jj_h[:len(sub)]
    return Nk, Tk, J


def single_tree_hr_jax(parent, tau, xcol, a_n, pi_arch, S, rho, rho_chain,
                       A=A_DIM, shared=None, want_jumps=False, pad=True):
    """Single-config wrapper matching cluster_hr_exact.single_tree_hr's signature."""
    rho = np.asarray(rho, np.float64)
    if shared is None:
        shared = make_shared(pi_arch, S, rho)
    else:
        shared = dict(shared); shared['rho_np'] = rho
    ptree = build_ptree(parent, tau, pad=pad)
    lo = pt_leaf_obs_single(ptree, xcol)
    Nk, Tk, J = single_tree_hr_bucketed(
        [ptree], [lo], [np.asarray(a_n)], np.asarray([rho_chain], np.float64),
        shared, want_jumps=want_jumps, b_chunk=1)
    return Nk[0], Tk[0], float(J[0])


# ===========================================================================
#  Tree structure extraction (binarise -> padded level arrays)
# ===========================================================================
def _binarize(parent, tau):
    """Convert an arbitrary-arity rooted tree (parent[root] < 0; leaves and root
    may live at any node index) to a strictly binary one by inserting 0-length
    internal edges (HR-neutral), relabelling so leaves are ids 0..n_leaves-1 and
    root is the last node.  Returns (parent2, tau2, orig_leaf_ids) where
    orig_leaf_ids[new_leaf_i] = the original node id of that leaf (so callers can
    remap per-node residue columns)."""
    parent = [int(p) for p in parent]
    tau = [float(t) for t in tau]
    n = len(parent)
    root = next(v for v in range(n) if parent[v] < 0)
    children = {v: [] for v in range(n)}
    for v in range(n):
        if v != root:
            children[parent[v]].append(v)
    orig_leaves = [v for v in range(n) if not children[v]]
    n_leaves = len(orig_leaves)

    new_parent = {}
    new_tau = {}
    next_id = [n]
    for u in range(n):
        ch = children[u]
        if not ch:
            continue
        cts = [(c, tau[c]) for c in ch]
        if len(cts) <= 2:
            for c, tc in cts:
                new_parent[c] = u; new_tau[c] = tc
        else:
            cur_parent = u
            for i in range(len(cts) - 2):
                c, tc = cts[i]
                x = next_id[0]; next_id[0] += 1
                new_parent[c] = cur_parent; new_tau[c] = tc
                new_parent[x] = cur_parent; new_tau[x] = 0.0
                cur_parent = x
            for c, tc in cts[-2:]:
                new_parent[c] = cur_parent; new_tau[c] = tc

    all_nodes = set(range(n)) | set(new_parent.keys())
    new_children = {v: [] for v in all_nodes}
    for c, p in new_parent.items():
        new_children[p].append(c)
    internals = [v for v in all_nodes if new_children[v]]
    internals_sorted = [v for v in internals if v != root] + [root]
    remap = {}
    for i, v in enumerate(orig_leaves):
        remap[v] = i
    for i, v in enumerate(internals_sorted):
        remap[v] = n_leaves + i
    N2 = len(all_nodes)
    P2 = [-1] * N2; T2 = [0.0] * N2
    for v in all_nodes:
        nv = remap[v]
        if v == root:
            P2[nv] = -1; T2[nv] = 0.0
        else:
            P2[nv] = remap[new_parent[v]]; T2[nv] = new_tau[v]
    return np.asarray(P2, np.int64), np.asarray(T2, np.float64), orig_leaves


def build_ptree(parent, tau, pad=True):
    """Binarise (parent, tau) and build a padded-level tree, bucketed to a
    GEOMSPACED (sqrt2) size bin on (#leaves, depth) when `pad=True`.  Trees whose
    (n_leaves, depth) fall in the same bin get the SAME padded shape -> XLA
    compiles ONE executable per bin, reused across all such families.

    `pad=False` uses minimal padding (N_bucket=n_leaves, D_bucket=depth) -- the
    per-tree UNPADDED reference used by the padded-vs-unpadded fidelity check.

    Returns a dict: pt (PaddedTree), orig_leaves, n_leaves, N_bucket, D_bucket,
    bucket=(N_bucket, D_bucket)."""
    from .tree_padded import sqrt2_bucket
    parent = np.asarray(parent, np.int64); tau = np.asarray(tau, np.float64)
    P2, T2, orig_leaves = _binarize(parent, tau)
    n_leaves = len(orig_leaves)
    dummy = np.full((n_leaves, 2), -1, np.int32)
    tree = build_tree(P2, T2, dummy)
    depth = int(compute_node_levels(tree)[tree.root])
    if pad:
        N_bucket = sqrt2_bucket(n_leaves); D_bucket = max(1, sqrt2_bucket(max(1, depth)))
    else:
        N_bucket = n_leaves; D_bucket = max(1, depth)
    pt = build_padded_tree(tree, N_bucket=N_bucket, D_bucket=D_bucket, m_bucket=2)
    return {'pt': pt, 'orig_leaves': np.asarray(orig_leaves, np.int64),
            'n_leaves': n_leaves, 'N_bucket': int(pt.N_bucket),
            'D_bucket': int(pt.D_bucket), 'bucket': (int(pt.N_bucket), int(pt.D_bucket))}


# ===========================================================================
#  FLAT postorder layout (no phantom lifts) -- dense O(2n-1) scan
# ===========================================================================
#  The level-scan layout above pays a D x N_max padded rectangle AND inflates
#  the node count with phantom identity "waiters" (a node whose child is >1
#  level below is lifted through every intermediate level).  A postorder FLAT
#  scan needs neither: a node gathers its children from ANY earlier flat
#  position, so M_flat = exactly 2n-1 real nodes (no phantoms), and the scan
#  is over that dense axis instead of D x N_max.  Measured ~11.5x fewer
#  slot-ops on the 128-leaf trees.  The per-branch HR primitives are IDENTICAL
#  to the level-scan kernel; only the traversal changes.

def build_flat_ptree(parent, tau, pad=True):
    """Binarise (parent, tau) and lay the 2n-1 nodes out in POSTORDER (children
    before parents), bucketing the node count to a sqrt2 bin so XLA compiles one
    executable per size bin.  Returns a dict with the flat structural arrays
    (all length M_actual = n_nodes, indexed by postorder position):
      child_pos (M,2) int   postorder positions of each node's two children
                            (0 for leaves/pad; masked out)
      child_branch (M,2)    branch lengths child->node (0 for leaves/pad)
      is_leaf (M,)          1.0 at leaf positions
      node_mask (M,)        1.0 real node, 0.0 pad
      leaf_src (M,)         orig-tree node id at leaf positions (-1 elsewhere),
                            so callers gather residue columns per config
      root_pos              postorder position of the root (= M_actual-1)
      M_actual, bucket=M_bucket (sqrt2-rounded, the compiled scan length)."""
    from .tree_padded import sqrt2_bucket
    parent = np.asarray(parent, np.int64); tau = np.asarray(tau, np.float64)
    P2, T2, orig_leaves = _binarize(parent, tau)
    n_leaves = len(orig_leaves)
    dummy = np.full((n_leaves, 2), -1, np.int32)
    tree = build_tree(P2, T2, dummy)
    n_nodes = tree.n_nodes
    po = np.asarray(tree.post_order, np.int64)                    # position -> node id
    pos_of = np.empty(n_nodes, np.int64); pos_of[po] = np.arange(n_nodes)
    M = n_nodes
    child_pos = np.zeros((M, 2), np.int32)
    child_branch = np.zeros((M, 2), np.float64)
    is_leaf = np.zeros(M, np.float64)
    leaf_src = np.full(M, -1, np.int64)
    for k in range(M):
        v = int(po[k])
        if tree.is_leaf(v):
            is_leaf[k] = 1.0
            leaf_src[k] = int(orig_leaves[v])                    # v in 0..n_leaves-1
        else:
            ch = tree.children[v]
            assert len(ch) == 2, f"non-binary node {v}: {ch}"
            for j, c in enumerate(ch):
                child_pos[k, j] = int(pos_of[int(c)])
                child_branch[k, j] = float(T2[int(c)])
    M_bucket = sqrt2_bucket(M) if pad else M
    return {'child_pos': child_pos, 'child_branch': child_branch,
            'is_leaf': is_leaf, 'leaf_src': leaf_src,
            'root_pos': int(pos_of[tree.root]), 'M_actual': int(M),
            'n_leaves': int(n_leaves), 'bucket': int(M_bucket)}


def flat_leaf_obs_single(fptree, xcol):
    """(M_actual,) int32 residue per postorder position for a single column;
    -1 (uninformative) at internal/pad positions.  `xcol` is indexed by ORIGINAL
    tree node id (length n_nodes of the pre-binarised tree)."""
    ls = fptree['leaf_src']; xcol = np.asarray(xcol)
    out = np.full(fptree['M_actual'], -1, np.int32)
    leaf = ls >= 0
    out[leaf] = xcol[ls[leaf]]
    return out


def flat_leaf_obs_pair(fptree, xcol, ycol):
    """(M_actual, 2) int32 residue pair per postorder position; -1 at
    internal/pad positions."""
    ls = fptree['leaf_src']; xcol = np.asarray(xcol); ycol = np.asarray(ycol)
    out = np.full((fptree['M_actual'], 2), -1, np.int32)
    leaf = ls >= 0
    out[leaf, 0] = xcol[ls[leaf]]; out[leaf, 1] = ycol[ls[leaf]]
    return out


def _flat_struct_tensors(fptrees, M_bucket):
    """Stack a batch of flat ptrees (same bucket) into (B, M_bucket, ...) uniform
    arrays, pad positions [M_actual:M_bucket) masked out (node_mask=0)."""
    B = len(fptrees)
    cp = np.zeros((B, M_bucket, 2), np.int32)
    cb = np.zeros((B, M_bucket, 2), np.float64)
    isl = np.zeros((B, M_bucket), np.float64)
    msk = np.zeros((B, M_bucket), np.float64)
    root = np.zeros(B, np.int32)
    for i, ft in enumerate(fptrees):
        M = ft['M_actual']
        cp[i, :M] = ft['child_pos']; cb[i, :M] = ft['child_branch']
        isl[i, :M] = ft['is_leaf']; msk[i, :M] = 1.0
        root[i] = ft['root_pos']
    return (jnp.asarray(cp), jnp.asarray(cb), jnp.asarray(isl),
            jnp.asarray(msk), jnp.asarray(root))


def _flat_pad_leaf_obs(leaf_obs_list, M_bucket, m):
    """Stack (n, M_bucket, m) postorder leaf observations, padded with -1."""
    out = np.full((len(leaf_obs_list), M_bucket, m), -1, np.int32)
    for i, lo in enumerate(leaf_obs_list):
        lo = np.asarray(lo)
        if lo.ndim == 1:
            out[i, :lo.shape[0], 0] = lo
        else:
            out[i, :lo.shape[0], :] = lo
    return out


def _make_single_hr_one_flat(M_bucket, Ka, A, L, want_jumps):
    """Per-config singleton (m=1) HR over a POSTORDER-FLAT node layout.  Same
    exact 40-state (theta,x) reversible tree HR as `_make_single_hr_one`, but the
    inside/outside passes scan over the dense 2n-1 node axis (length M_bucket)
    instead of a D x N_max padded rectangle.  Children are gathered by their
    postorder position (< the node's own position), so no phantom identity nodes
    are needed.  Maps (leaf_obs (M,) int, a_n (L,) int, rho_chain, child_pos
    (M,2) int, child_branch (M,2), is_leaf (M,), node_mask (M,), root_pos,
    pi_arch, rho, gtr_Q, gtr_offdiag) -> (Nk (Ka,A,A), Tk (Ka,A), jumps)."""

    def hr_one(leaf_obs, a_n, rho_chain,
               child_pos, child_branch, is_leaf, node_mask, root_pos,
               pi_arch, rho, gtr_Q, gtr_offdiag):
        pi_an = pi_arch[a_n]
        Q40, p40 = _build_q40(gtr_Q[a_n], gtr_offdiag[a_n], pi_an, rho, rho_chain)
        lam, V, sqrt, inv = reversible_eigh_jax(Q40, p40)
        scaled = Q40 * (sqrt[:, None] * inv[None, :])
        onehot = jax.nn.one_hot(a_n, Ka)                             # (L,Ka)

        def _leaf(x):
            gx = (x < 0) | (x >= A)
            e = jnp.where(gx, jnp.ones(A), jax.nn.one_hot(jnp.clip(x, 0, A - 1), A))
            return jnp.broadcast_to(e[None], (L, A)).reshape(-1)      # (LA,)
        leaf_msgs = jax.vmap(_leaf)(leaf_obs)                         # (M,LA)

        # ---- inside pass (postorder) : one node per scan step ----------------
        pos = jnp.arange(M_bucket)

        def _inside_step(inside, x):
            m, c0, c1, b0, b1, isl = x
            P0 = _Pt_jax(lam, V, sqrt, inv, b0)
            P1 = _Pt_jax(lam, V, sqrt, inv, b1)
            comb = (P0 @ inside[c0]) * (P1 @ inside[c1])
            s = jnp.maximum(comb.sum(), 1e-300)
            comb = comb / s
            new = jnp.where(isl > 0.5, inside[m], comb)              # leaves keep init
            return inside.at[m].set(new), None
        inside0 = leaf_msgs                                          # (M,LA) leaf init
        inside, _ = jax.lax.scan(
            _inside_step, inside0,
            (pos, child_pos[:, 0], child_pos[:, 1],
             child_branch[:, 0], child_branch[:, 1], is_leaf))

        # ---- outside pass (reverse postorder) + per-branch HR ----------------
        O_init = jnp.zeros((M_bucket, L * A)).at[root_pos].set(p40)

        def _outside_step(carry, x):
            O, Nk_acc, Tk_acc, j_acc = carry
            m, c0, c1, b0, b1, isl, msk = x
            O_s = O[m]
            inL = inside[c0]; inR = inside[c1]
            PL = _Pt_jax(lam, V, sqrt, inv, b0)
            PR = _Pt_jax(lam, V, sqrt, inv, b1)
            mvL = PL @ inL; mvR = PR @ inR
            combinedL = O_s * mvR; combinedR = O_s * mvL
            # this node contributes HR only if it is a REAL INTERNAL node
            hr_mask = msk * (1.0 - isl)
            ENL, dwL = _single_branch_hr40(combinedL, inL, b0, lam, V, sqrt, inv, scaled)
            ZL = combinedL @ mvL
            ENR, dwR = _single_branch_hr40(combinedR, inR, b1, lam, V, sqrt, inv, scaled)
            ZR = combinedR @ mvR

            def _agg(EN, dw, Z):
                safe = jnp.where(Z > 1e-300, Z, 1.0)
                Nk_l = jnp.zeros((Ka, A, A)); Tk_l = jnp.zeros((Ka, A))
                jmp = EN.sum()
                for t in range(L):
                    blk = EN[t * A:t * A + A, t * A:t * A + A]
                    Nk_l = Nk_l + onehot[t][:, None, None] * blk[None]
                    Tk_l = Tk_l + onehot[t][:, None] * dw[t * A:t * A + A][None]
                    jmp = jmp - blk.sum()
                good = (Z > 1e-300)
                return (jnp.where(good, Nk_l / safe, 0.0),
                        jnp.where(good, Tk_l / safe, 0.0),
                        jnp.where(good, jmp / safe, 0.0))
            NkL, TkL, jL = _agg(ENL, dwL, ZL)
            NkR, TkR, jR = _agg(ENR, dwR, ZR)
            Nk_add = hr_mask * (NkL + NkR)
            Tk_add = hr_mask * (TkL + TkR)
            j_add = hr_mask * (jL + jR)
            # propagate outside to children (only for real internal nodes)
            oL = combinedL @ PL; oR = combinedR @ PR
            wr = (hr_mask > 0.5)
            O = O.at[c0].set(jnp.where(wr, oL, O[c0]))
            O = O.at[c1].set(jnp.where(wr, oR, O[c1]))
            # accumulate in the carry -- avoids stacking (M,Ka,A,A) per node
            return (O, Nk_acc + Nk_add, Tk_acc + Tk_add, j_acc + j_add), None

        (_, Nk_tot, Tk_tot, j_tot), _ = jax.lax.scan(
            _outside_step,
            (O_init, jnp.zeros((Ka, A, A)), jnp.zeros((Ka, A)), 0.0),
            (pos, child_pos[:, 0], child_pos[:, 1],
             child_branch[:, 0], child_branch[:, 1], is_leaf, node_mask),
            reverse=True)
        return Nk_tot, Tk_tot, j_tot

    return hr_one


def _get_single_hr_vmap_flat(bucket, Ka, A, L, want_jumps):
    key = ('single_flat', bucket, Ka, A, L, want_jumps)
    if key not in _HR_FN_CACHE:
        hr_one = _make_single_hr_one_flat(bucket, Ka, A, L, want_jumps)
        in_axes = (0, 0, 0, 0, 0, 0, 0, 0) + (None,) * 4
        _HR_FN_CACHE[key] = jax.jit(jax.vmap(hr_one, in_axes=in_axes))
    return _HR_FN_CACHE[key]


def _make_hr_one_flat(M_bucket, Ka, A, L, want_jumps, delta1_enabled=True):
    """Per-config factored PAIR HR over a POSTORDER-FLAT node layout.  Identical
    exact compound up-down math as `_make_hr_one` (the Delta=0 + Delta>=1 + jump
    per-branch core), but the inside/outside passes scan the dense 2n-1 node axis
    instead of a D x N_max padded rectangle.  Children gathered by postorder
    position; no phantom identity nodes.  Signature mirrors the level-scan pair fn
    with flat structural tensors:
      hr_one(leaf_obs (M,2), a1 (L,), a2 (L,), rho_chain, child_pos (M,2),
             child_branch (M,2), is_leaf (M,), node_mask (M,), root_pos,
             pi_arch, rho, gtr_Q, gtr_offdiag, glam, gV, gsqrt, ginv, gscaled)
      -> (Nk (Ka,A,A), Tk (Ka,A), jumps)."""

    def hr_one(leaf_obs, a1, a2, rho_chain,
               child_pos, child_branch, is_leaf, node_mask, root_pos,
               pi_arch, rho, gtr_Q, gtr_offdiag, glam, gV, gsqrt, ginv, gscaled):
        pi1 = pi_arch[a1]; pi2 = pi_arch[a2]                        # (L,A)
        e1 = (glam[a1], gV[a1], gsqrt[a1], ginv[a1], gscaled[a1])
        e2 = (glam[a2], gV[a2], gsqrt[a2], ginv[a2], gscaled[a2])
        Q1 = gtr_Q[a1]; Q2 = gtr_Q[a2]
        off1 = gtr_offdiag[a1]; off2 = gtr_offdiag[a2]

        if delta1_enabled:
            Q40_1, p40_1 = _build_q40(Q1, off1, pi1, rho, rho_chain)
            Q40_2, p40_2 = _build_q40(Q2, off2, pi2, rho, rho_chain)
            l1, Vv1, s1, i1 = reversible_eigh_jax(Q40_1, p40_1)
            l2, Vv2, s2, i2 = reversible_eigh_jax(Q40_2, p40_2)
            e40_1 = (l1, Vv1, s1, i1, Q40_1 * (s1[:, None] * i1[None, :]))
            e40_2 = (l2, Vv2, s2, i2, Q40_2 * (s2[:, None] * i2[None, :]))
        else:
            e40_1 = e40_2 = None

        if want_jumps and delta1_enabled:
            Qf = rho_chain * jnp.broadcast_to(rho[None, :], (L, L))
            Qf = Qf - jnp.diag(jnp.diagonal(Qf))
            Qf = Qf - jnp.diag(Qf.sum(axis=1))
            lf, Vf, sf, iff = reversible_eigh_jax(Qf, rho)
            eigf = (lf, Vf, sf, iff, Qf * (sf[:, None] * iff[None, :]))
        else:
            eigf = None

        p_comp = rho[:, None, None] * (pi1[:, :, None] * pi2[:, None, :])  # (L,A,A)
        pos = jnp.arange(M_bucket)

        # ---- inside pass (postorder), compound (L,A,A) messages ----
        leaf_msgs = jax.vmap(lambda o: _leaf_msg_pair(o, L, A))(leaf_obs)  # (M,L,A,A)

        def _inside_step(inside, x):
            m, c0, c1, b0, b1, isl = x
            P1L = _Pt_theta(*e1[:4], b0); P2L = _Pt_theta(*e2[:4], b0)
            P1R = _Pt_theta(*e1[:4], b1); P2R = _Pt_theta(*e2[:4], b1)
            bL, JL = _field_kernels_jax(rho, rho_chain, b0)
            bR, JR = _field_kernels_jax(rho, rho_chain, b1)
            mL = _branch_pair(inside[c0], P1L, P2L, pi1, pi2, bL, JL)
            mR = _branch_pair(inside[c1], P1R, P2R, pi1, pi2, bR, JR)
            comb = mL * mR
            s = jnp.maximum(comb.sum(), 1e-300)
            comb = comb / s
            new = jnp.where(isl > 0.5, inside[m], comb)
            return inside.at[m].set(new), None
        inside, _ = jax.lax.scan(
            _inside_step, leaf_msgs,
            (pos, child_pos[:, 0], child_pos[:, 1],
             child_branch[:, 0], child_branch[:, 1], is_leaf))

        # ---- outside pass (reverse postorder) + per-branch HR ----
        O_init = jnp.zeros((M_bucket, L, A, A)).at[root_pos].set(p_comp)

        def _outside_step(carry, x):
            O, Nk_acc, Tk_acc, j_acc = carry
            m, c0, c1, b0, b1, isl, msk = x
            O_s = O[m]
            inL = inside[c0]; inR = inside[c1]
            P1L = _Pt_theta(*e1[:4], b0); P2L = _Pt_theta(*e2[:4], b0)
            P1R = _Pt_theta(*e1[:4], b1); P2R = _Pt_theta(*e2[:4], b1)
            bL, JL = _field_kernels_jax(rho, rho_chain, b0)
            bR, JR = _field_kernels_jax(rho, rho_chain, b1)
            mvL = _branch_pair(inL, P1L, P2L, pi1, pi2, bL, JL)
            mvR = _branch_pair(inR, P1R, P2R, pi1, pi2, bR, JR)
            combinedL = O_s * mvR
            combinedR = O_s * mvL
            hr_mask = msk * (1.0 - isl)
            NkL, TkL, jmL = _branch_hr_core(
                combinedL, inL, b0, P1L, P2L, bL, JL, a1, a2, e1, e2,
                pi1, pi2, rho, rho_chain, e40_1, e40_2, eigf, Ka,
                want_jumps and delta1_enabled, delta1_enabled)
            ZL = (combinedL * mvL).sum()
            NkR, TkR, jmR = _branch_hr_core(
                combinedR, inR, b1, P1R, P2R, bR, JR, a1, a2, e1, e2,
                pi1, pi2, rho, rho_chain, e40_1, e40_2, eigf, Ka,
                want_jumps and delta1_enabled, delta1_enabled)
            ZR = (combinedR * mvR).sum()
            safeZL = jnp.where(ZL > 1e-300, ZL, 1.0)
            safeZR = jnp.where(ZR > 1e-300, ZR, 1.0)
            Nk_add = hr_mask * (jnp.where(ZL > 1e-300, NkL / safeZL, 0.0)
                                + jnp.where(ZR > 1e-300, NkR / safeZR, 0.0))
            Tk_add = hr_mask * (jnp.where(ZL > 1e-300, TkL / safeZL, 0.0)
                                + jnp.where(ZR > 1e-300, TkR / safeZR, 0.0))
            j_add = hr_mask * (jnp.where(ZL > 1e-300, jmL / safeZL, 0.0)
                               + jnp.where(ZR > 1e-300, jmR / safeZR, 0.0))
            # propagate outside to children (real internal nodes only)
            ceffL = jnp.where(p_comp > 1e-300, combinedL / jnp.where(p_comp > 1e-300, p_comp, 1.0), 0.0)
            ceffR = jnp.where(p_comp > 1e-300, combinedR / jnp.where(p_comp > 1e-300, p_comp, 1.0), 0.0)
            oL = p_comp * _branch_pair(ceffL, P1L, P2L, pi1, pi2, bL, JL)
            oR = p_comp * _branch_pair(ceffR, P1R, P2R, pi1, pi2, bR, JR)
            wr = (hr_mask > 0.5)
            O = O.at[c0].set(jnp.where(wr, oL, O[c0]))
            O = O.at[c1].set(jnp.where(wr, oR, O[c1]))
            return (O, Nk_acc + Nk_add, Tk_acc + Tk_add, j_acc + j_add), None

        (_, Nk_tot, Tk_tot, j_tot), _ = jax.lax.scan(
            _outside_step,
            (O_init, jnp.zeros((Ka, A, A)), jnp.zeros((Ka, A)), 0.0),
            (pos, child_pos[:, 0], child_pos[:, 1],
             child_branch[:, 0], child_branch[:, 1], is_leaf, node_mask),
            reverse=True)
        return Nk_tot, Tk_tot, j_tot

    return hr_one


def _get_hr_vmap_flat(bucket, Ka, A, L, want_jumps, delta1_enabled):
    key = ('pair_flat', bucket, Ka, A, L, want_jumps, delta1_enabled)
    if key not in _HR_FN_CACHE:
        hr_one = _make_hr_one_flat(bucket, Ka, A, L, want_jumps, delta1_enabled)
        in_axes = (0, 0, 0, 0, 0, 0, 0, 0, 0) + (None,) * 9
        _HR_FN_CACHE[key] = jax.jit(jax.vmap(hr_one, in_axes=in_axes))
    return _HR_FN_CACHE[key]


def pair_tree_hr_bucketed_flat(fptrees, leaf_obs_list, a1_arr, a2_arr, rc_arr,
                               shared, want_jumps=False, delta1_enabled=None,
                               b_chunk=128):
    """Dense postorder-flat pair HR over a heterogeneous batch.  Drop-in for
    `pair_tree_hr_bucketed` but consumes `build_flat_ptree` dicts and scans the
    2n-1 node axis (no D x N_max padding).  Buckets by node-count bin.Returns (Nk (n,Ka,A,A), Tk (n,Ka,A), jumps (n,))."""
    n = len(fptrees)
    Ka = int(shared['Ka']); A = int(shared['A'])
    L = int(np.asarray(shared['rho_np']).shape[0])
    rc_np = np.asarray(rc_arr, np.float64)
    if delta1_enabled is None:
        delta1_enabled = bool(np.any(rc_np > 0))
    sargs = _shared_args(shared)
    Nk = np.zeros((n, Ka, A, A)); Tk = np.zeros((n, Ka, A)); J = np.zeros(n)
    groups = defaultdict(list)
    for i, ft in enumerate(fptrees):
        groups[ft['bucket']].append(i)
    for bucket, idxs in groups.items():
        M_bucket = int(bucket)
        vfn = _get_hr_vmap_flat(M_bucket, Ka, A, L, want_jumps, delta1_enabled)
        for c0 in range(0, len(idxs), b_chunk):
            sub = idxs[c0:c0 + b_chunk]
            npad = b_chunk - len(sub)
            sub_p = sub + [sub[0]] * npad
            cp, cb, isl, msk, root = _flat_struct_tensors(
                [fptrees[i] for i in sub_p], M_bucket)
            lo = jnp.asarray(_flat_pad_leaf_obs(
                [leaf_obs_list[i] for i in sub_p], M_bucket, 2))
            a1 = jnp.asarray(np.stack([np.asarray(a1_arr[i]) for i in sub_p]).astype(np.int32))
            a2 = jnp.asarray(np.stack([np.asarray(a2_arr[i]) for i in sub_p]).astype(np.int32))
            rc = jnp.asarray(rc_np[sub_p])
            nk, tk, jj = vfn(lo, a1, a2, rc, cp, cb, isl, msk, root, *sargs)
            nk_h = np.asarray(nk); tk_h = np.asarray(tk); jj_h = np.asarray(jj)
            Nk[sub] = nk_h[:len(sub)]; Tk[sub] = tk_h[:len(sub)]; J[sub] = jj_h[:len(sub)]
    return Nk, Tk, J


def pt_leaf_obs_pair(ptree, xcol, ycol):
    """(N_bucket, 2) int32 leaf observations for a residue-column pair; leaves in
    slots 0..n_leaves-1 (padding leaves stay -1 = uninformative)."""
    ol = ptree['orig_leaves']; N = ptree['N_bucket']
    out = np.full((N, 2), -1, np.int32)
    out[:len(ol), 0] = np.asarray(xcol)[ol]
    out[:len(ol), 1] = np.asarray(ycol)[ol]
    return out


def pt_leaf_obs_single(ptree, xcol):
    """(N_bucket,) int32 leaf observations for a single column."""
    ol = ptree['orig_leaves']; N = ptree['N_bucket']
    out = np.full((N,), -1, np.int32)
    out[:len(ol)] = np.asarray(xcol)[ol]
    return out


def _bucket_n_max(padded_list, D_bucket):
    """Uniform slot count N_max for a bucket: max over its trees & levels of the
    slot count.  Depends only on the bucket's tree SET (stable across M-steps), so
    the compiled shape is reused.  sqrt2-rounded for a bounded shape set."""
    from .tree_padded import sqrt2_bucket
    m = 1
    for pt in padded_list:
        m = max(m, int(pt.N_bucket))
        for l in range(D_bucket):
            m = max(m, int(pt.n_slots(l + 1)))
    return int(sqrt2_bucket(m))


def _struct_tensors(padded_list, D_bucket, N_max):
    """Stack structural tensors for a batch of PaddedTrees sharing a bucket into a
    UNIFORM (B, D, N_max, ...) layout -- every level padded to the bucket-constant
    slot count N_max so the depth axis can be driven by lax.scan (fixed per-step
    shape) and the executable is reused across chunks/iters.  Padded slots
    (slot_mask==0) and phantom identity edges are marked so the HR treats them as
    identity (zero dwell/counts/jumps).  Returns (child_pos (B,D,N_max,2),
    child_branch (B,D,N_max,2), is_identity (B,D,N_max), slot_mask (B,D,N_max),
    root_slot (B,))."""
    B = len(padded_list)
    cp = np.zeros((B, D_bucket, N_max, 2), np.int32)
    cb = np.zeros((B, D_bucket, N_max, 2), np.float64)
    idn = np.zeros((B, D_bucket, N_max), np.float64)
    sm = np.zeros((B, D_bucket, N_max), np.float64)
    for i, pt in enumerate(padded_list):
        for l in range(D_bucket):
            na = pt.n_slots(l + 1)
            cp[i, l, :na] = pt.child_pos[l]; cb[i, l, :na] = pt.child_branch[l]
            smi = pt.slot_mask[l]
            idm = ((pt.child_pos[l][:, 0] == pt.child_pos[l][:, 1])
                   & (pt.child_branch[l][:, 0] == 0.0)
                   & (pt.child_branch[l][:, 1] == 0.0) & (smi > 0.5))
            idn[i, l, :na] = idm.astype(np.float64); sm[i, l, :na] = smi
    root = np.array([pt.root_slot for pt in padded_list], np.int32)
    return (jnp.asarray(cp), jnp.asarray(cb), jnp.asarray(idn),
            jnp.asarray(sm), jnp.asarray(root))


def _pad_leaf_obs(leaf_obs_list, N_max, m):
    """Stack (n, N_bucket, m) leaf observations into (n, N_max, m) padded with -1
    (padding leaves are uninformative)."""
    out = np.full((len(leaf_obs_list), N_max, m), -1, np.int32)
    for i, lo in enumerate(leaf_obs_list):
        lo = np.asarray(lo)
        if lo.ndim == 1:
            out[i, :lo.shape[0], 0] = lo
        else:
            out[i, :lo.shape[0], :] = lo
    return out


# ===========================================================================
#  Public: bin-bucketed batched pair HR (O(#size-bins) compiles)
# ===========================================================================
_HR_FN_CACHE: dict = {}


def _shared_args(shared):
    """The runtime eig arrays passed to the jitted HR fn (order matches the
    hr_one signature tail)."""
    return (shared['pi_arch'], jnp.asarray(shared['rho_np']), shared['gtr_Q'],
            shared['gtr_offdiag'], shared['lam'], shared['V'], shared['sqrt'],
            shared['inv'], shared['scaled'])


def _get_hr_vmap(bucket, D, Ka, A, L, want_jumps, delta1_enabled):
    key = ('pair', bucket, D, Ka, A, L, want_jumps, delta1_enabled)
    if key not in _HR_FN_CACHE:
        hr_one = _make_hr_one(D, Ka, A, L, want_jumps, delta1_enabled)
        # structure tensors are single stacked arrays (batch axis 0); scan drives
        # the depth axis inside -> in_axes 0 for all per-config args, None shared.
        in_axes = (0, 0, 0, 0, 0, 0, 0, 0, 0) + (None,) * 9
        _HR_FN_CACHE[key] = jax.jit(jax.vmap(hr_one, in_axes=in_axes))
    return _HR_FN_CACHE[key]


def pair_tree_hr_bucketed(ptrees, leaf_obs_list, a1_arr, a2_arr, rc_arr, shared,
                          want_jumps=False, delta1_enabled=None, b_chunk=64):
    """Exact factored pair HR over a heterogeneous batch of (config, family-tree).

    ptrees: list (len n) of build_ptree dicts (one per config; families repeat).
    leaf_obs_list: list of (N_bucket,2) arrays. a1_arr/a2_arr (n,L), rc_arr (n,).
    Groups configs by size BIN -> one compiled executable per bin (reused across
    families in the bin and across M-steps).  Each per-bin chunk is PADDED to a
    fixed size `b_chunk` (extra configs masked out) so the batch shape -- hence
    the compiled executable -- is identical across iterations even as the number
    of surviving configs and rho_chain change.  Returns (Nk (n,Ka,A,A),
    Tk (n,Ka,A), jumps (n,)) in input order.  A single `delta1_enabled` governs
    the whole batch (split rc==0 vs rc>0 upstream to skip Delta>=1 cheaply)."""
    n = len(ptrees)
    Ka = int(shared['Ka']); A = int(shared['A'])
    L = int(np.asarray(shared['rho_np']).shape[0])
    rc_np = np.asarray(rc_arr, np.float64)
    if delta1_enabled is None:
        delta1_enabled = bool(np.any(rc_np > 0))
    sargs = _shared_args(shared)
    Nk = np.zeros((n, Ka, A, A)); Tk = np.zeros((n, Ka, A)); J = np.zeros(n)
    groups = defaultdict(list)
    for i, pt in enumerate(ptrees):
        groups[pt['bucket']].append(i)
    for bucket, idxs in groups.items():
        D = bucket[1]
        N_max = _bucket_n_max([ptrees[i]['pt'] for i in idxs], D)
        vfn = _get_hr_vmap(bucket, D, Ka, A, L, want_jumps, delta1_enabled)
        for c0 in range(0, len(idxs), b_chunk):
            sub = idxs[c0:c0 + b_chunk]
            npad = b_chunk - len(sub)                       # pad chunk to fixed size
            sub_p = sub + [sub[0]] * npad                   # replicate a real config
            padded = [ptrees[i]['pt'] for i in sub_p]
            cp, cb, idn, sm, root = _struct_tensors(padded, D, N_max)
            lo = jnp.asarray(_pad_leaf_obs([leaf_obs_list[i] for i in sub_p], N_max, 2))
            a1 = jnp.asarray(np.stack([np.asarray(a1_arr[i]) for i in sub_p]).astype(np.int32))
            a2 = jnp.asarray(np.stack([np.asarray(a2_arr[i]) for i in sub_p]).astype(np.int32))
            rc = jnp.asarray(rc_np[sub_p])
            nk, tk, jj = vfn(lo, a1, a2, rc, cp, cb, idn, sm, root, *sargs)
            # ONE device->host transfer per chunk (not per config)
            nk_h = np.asarray(nk); tk_h = np.asarray(tk); jj_h = np.asarray(jj)
            Nk[sub] = nk_h[:len(sub)]; Tk[sub] = tk_h[:len(sub)]; J[sub] = jj_h[:len(sub)]
    return Nk, Tk, J


def pair_tree_hr_jax(parent, tau, xcol, ycol, a1, a2, pi_arch, S, rho,
                     rho_chain, A=A_DIM, shared=None, want_jumps=False, pad=True):
    """Single-config convenience wrapper matching cluster_hr_exact.pair_tree_hr's
    signature (returns numpy Nk (Ka,A,A), Tk (Ka,A), jumps).  `pad` toggles the
    sqrt2 bin padding (True) vs minimal padding (False) -- both must agree."""
    rho = np.asarray(rho, np.float64)
    if shared is None:
        shared = make_shared(pi_arch, S, rho)
    else:
        shared = dict(shared); shared['rho_np'] = rho
    ptree = build_ptree(parent, tau, pad=pad)
    lo = pt_leaf_obs_pair(ptree, xcol, ycol)
    Nk, Tk, J = pair_tree_hr_bucketed(
        [ptree], [lo], [np.asarray(a1)], [np.asarray(a2)],
        np.asarray([rho_chain], np.float64), shared, want_jumps=want_jumps, b_chunk=1)
    return Nk[0], Tk[0], float(J[0])


def make_shared(pi_arch, S, rho):
    """Build the corpus-wide shared eig cache (used across all configs of an
    M-step).  Attach rho so tree fns can read it."""
    shared = build_shared_arch_eig(pi_arch, S)
    shared['rho_np'] = np.asarray(rho, np.float64)
    return shared
