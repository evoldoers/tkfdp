#!/usr/bin/env python3
r"""JAX port of fit_coupling_mixture_rateI.py -- mixture of K Metropolis-sqrt coupling
components + an orthogonal per-cluster Gamma+Invariant rate class, fit by HR-EM on
tau-binned pairwise cherry counts with a family held-out split.

This is a *numerics port* of the numpy reference experiments/fit_coupling_mixture_rateI.py
(which is the ground truth). Same model, same ECM schedule, same CLI/outputs; the heavy
per-component linear algebra (the reversible eigendecomposition, the expm / divided-
difference Holmes-Rubin bridge kernel, the class x rate x tau grids, the ragged
segment_sum count aggregation) is vectorised/batched/jit'd in JAX at fp64.

Model (identical to the reference):
  * 400 ordered pair-states x=(i,j), index i*20+j.
  * Q_c = Metropolis-sqrt(S, pi_c): a single-transition reversible pair generator,
    Q_{(i,j)->(k,j)} = S_ik sqrt(pi_kj/pi_ij), Q_{(i,j)->(i,l)} = S_jl sqrt(pi_il/pi_ij);
    ONE shared free single-site exchangeability S (warm-init LG08), per-class symmetric
    stationary pi_c (the coupling).
  * Per-cluster rate class from a Gamma(alpha)+Invariant grid (Yang), independent of the
    pairing class. Invariant bin (rate 0, weight p_inv) supports only fully-conserved
    clusters (logP = 0 on the diagonal, -inf off it).

EM (ECM, monotone in the observed-data marginal LL):
  E: r_{gcr} propto w_c rho_r exp[ sum_e n_g^e logP_{c,r}(tb_e) ]  (P at rate_r*tau_center).
  M: w_c = sum_{g,r} r_{gcr}; rho_r = sum_{g,c} r_{gcr}; p_inv = rho_invariant.
     pi_c,S : HR / bridge M-step of the Metropolis-sqrt chain on the responsibility-
              weighted counts placed at the RATE-SCALED branch length rate_r*tau (aggregate
              the HR usage/dwell over the Gamma rates; the invariant bin has 0 dwell).
     alpha  : local 1-D search over the Gamma shape (current alpha always a candidate).

Corpus: data/per_contact_trrosetta/counts.npz (flat pf,pt,tb,cnt + cptr cluster pointers).

Verification: `--verify` runs gate-1 (unit match vs the numpy reference on a small subset,
fp64, one E-step + M-step) and reports the max abs discrepancy. Run it under
CUDA_VISIBLE_DEVICES="" for an exact CPU-vs-CPU comparison.

GPU: pass CUDA_VISIBLE_DEVICES=1 to use GPU 1 (GPU 0 belongs to another user). fp64 is
forced (JAX_ENABLE_X64=1) to match the reference precision."""
from __future__ import annotations

import argparse
import json
import os
import time

# fp64 to match the numpy reference. Do NOT force a platform here: the caller picks the
# device via CUDA_VISIBLE_DEVICES (=1 for GPU 1, ="" for CPU). Must be set before jax import.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import sys
from functools import partial

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tkfdp.lg08 import S_LG08
from tkfdp.coupling.dynfield.phylo_elbo.rate_hetero import discrete_gamma_rates

NA = 20
NS = NA * NA                                                   # 400 pair states

S_LG = np.asarray(S_LG08, float)[:NA, :NA].copy()
S_LG = 0.5 * (S_LG + S_LG.T)
np.fill_diagonal(S_LG, 0.0)
_IU = np.triu_indices(NA, 1)                                   # 190 off-diagonal entries
_SLG_OFF = S_LG[_IU]

_EYE_NA = jnp.eye(NA)
_EYE_NS = jnp.eye(NS)


# =====================================================================================
# Core linear algebra (JAX, fp64) -- exact ports of fit_pair_models / permfield.hr
# =====================================================================================
def met_Q_sqrt(S, pi):
    """Metropolis-sqrt pair generator Q (NS,NS) from single-site exchangeability S (NA,NA)
    and pair stationary pi (NS,). Port of FP._met_Q(S, piM, "sqrt")."""
    piM = jnp.maximum(pi.reshape(NA, NA), 1e-300)
    sq = jnp.sqrt(piM)
    invsq = 1.0 / sq
    # comp1: (i,j)->(k,j): S[i,k]*sqrt(piM[k,j]/piM[i,j]); place at l==j
    A1 = S[:, None, :] * invsq[:, :, None] * jnp.transpose(sq)[None, :, :]   # [i,j,k]
    M1 = jnp.einsum("ijk,jl->ijkl", A1, _EYE_NA)
    # comp2: (i,j)->(i,l): S[j,l]*sqrt(piM[i,l]/piM[i,j]); place at k==i
    A2 = S[None, :, :] * sq[:, None, :] * invsq[:, :, None]                  # [i,j,l]
    M2 = jnp.einsum("ijl,ik->ijkl", A2, _EYE_NA)
    Q = (M1 + M2).reshape(NS, NS)
    Q = Q * (1.0 - _EYE_NS)                                    # zero diagonal
    Q = Q - jnp.diag(Q.sum(1))                                # diag = -rowsum
    return Q


def eig_rev(Q, pi):
    """Symmetric eigendecomposition of a reversible generator. Q = U diag(lam) Uinv.
    Port of permfield.hr.eig_rev (minus the LAPACK retry path)."""
    d = jnp.sqrt(jnp.clip(pi, 1e-300, None))
    di = 1.0 / d
    Qs = d[:, None] * Q * di[None, :]
    Qs = 0.5 * (Qs + Qs.T)
    lam, V = jnp.linalg.eigh(Qs)
    U = di[:, None] * V
    Uinv = V.T * d[None, :]
    return lam, U, Uinv


def _Jmat(lam, t):
    """Divided-difference kernel J_kl = int_0^t e^{lam_k s} e^{lam_l (t-s)} ds. Port of
    permfield.hr._Jmat with the same degeneracy handling (|dl|<1e-12 -> t e^{lam_k t})."""
    e = jnp.exp(lam * t)
    dl = lam[:, None] - lam[None, :]
    close = jnp.abs(dl) < 1e-12
    dl_safe = jnp.where(close, 1.0, dl)
    J_off = (e[:, None] - e[None, :]) / dl_safe
    J_deg = t * e[:, None] * jnp.ones_like(e[None, :])         # t e_k for degenerate rows
    return jnp.where(close, J_deg, J_off)


def bridge_core(lam, U, Uinv, Q, t, edge):
    """Expected dwell Tdw[x] and usage N[x,y] over a branch of length t, averaged over the
    endpoint joint `edge` (NS,NS). Eigenmode contraction; port of permfield.hr.bridge."""
    J = _Jmat(lam, t)
    P = (U * jnp.exp(lam * t)[None, :]) @ Uinv
    W = edge / jnp.maximum(P, 1e-300)
    M = W @ Uinv.T
    Aleft = U.T @ M
    AJ = Aleft * J
    UinvT_AJ = Uinv.T @ AJ
    Tdw = jnp.einsum("xl,xl->x", UinvT_AJ, U)
    B = UinvT_AJ @ U.T
    N = (Q * B) * (1.0 - _EYE_NS)                             # usage, zero diagonal
    return Tdw, N


@partial(jax.jit, static_argnums=())
def bridge_sum_c(lam, U, Uinv, Q, tau, rate_gamma, Ncr_c):
    """Sum the HR bridge over all (gamma rate x tau bin) at rate-scaled branch length
    rate_gamma[r]*tau[t], on the (c-)responsibility-weighted count tensor Ncr_c
    (NS,NS,nT,Rg). Returns (N_c (NS,NS), T_c (NS,)). Batched (vmap) over Rg*nT bridges."""
    Rg = rate_gamma.shape[0]
    nT = tau.shape[0]
    edges = jnp.transpose(Ncr_c, (3, 2, 0, 1)).reshape(Rg * nT, NS, NS)   # [r,t,x,y]
    bs = (rate_gamma[:, None] * tau[None, :]).reshape(Rg * nT)
    Tds, Ns = jax.vmap(lambda b, e: bridge_core(lam, U, Uinv, Q, b, e))(bs, edges)
    return Ns.sum(0), Tds.sum(0)


def mstep_pi_metropolis(S, pi, N, T, steps=40, lr=0.5):
    """Symmetric-pi M-step for the Metropolis-sqrt family: damped gradient ascent on
    log pi_ij of the complete-data LL. Port of FP.mstep_pi_metropolis (kernel="sqrt")."""
    b0 = jnp.log(jnp.maximum(pi, 1e-12)).reshape(NA, NA)
    b0 = 0.5 * (b0 + b0.T)
    scale = jnp.maximum(T.sum(), 1.0)

    def step(b, _):
        pm = jnp.exp(b - b.max())
        pm = pm / pm.sum()
        piv = pm.reshape(NS)
        Q = met_Q_sqrt(S, piv)
        R = N - T[:, None] * Q
        ds = -0.5 / jnp.maximum(piv[:, None], 1e-300)
        dd = 0.5 / jnp.maximum(piv[None, :], 1e-300)
        grad = piv * ((R * ds).sum(1) + (R * dd).sum(0))
        gM = grad.reshape(NA, NA)
        gM = gM + gM.T
        b = b + lr * gM / scale
        b = 0.5 * (b + b.T)
        return b, None

    b, _ = jax.lax.scan(step, b0, None, length=steps)
    pm = jnp.exp(b - b.max())
    return (pm / pm.sum()).reshape(NS)


def met_S_suffstats(N, T, pi):
    """Pooled Metropolis-sqrt S-step sufficient statistics: usage numerator Cnum (pi-free)
    and exposure denominator Hden (pi-dependent). Port of FS.met_S_suffstats(kernel="sqrt")."""
    piM = jnp.maximum(pi.reshape(NA, NA), 1e-300)
    T2 = T.reshape(NA, NA)
    N4 = N.reshape(NA, NA, NA, NA)
    Cdir = jnp.einsum("ijkj->ik", N4) + jnp.einsum("ijil->jl", N4)
    Cnum = Cdir + Cdir.T
    f1 = jnp.sqrt(piM[None, :, :] / piM[:, None, :])          # [a,b,j] sqrt(piM[b,j]/piM[a,j])
    H1 = (T2[:, None, :] * f1).sum(2)
    f2 = jnp.sqrt(piM[:, None, :] / piM[:, :, None])          # [i,a,b] sqrt(piM[i,b]/piM[i,a])
    H2 = (T2[:, :, None] * f2).sum(0)
    Hden = H1 + H2
    Hden = Hden + Hden.T
    return Cnum, Hden


# batched builders over the K components
met_Q_batch = jax.jit(jax.vmap(met_Q_sqrt, in_axes=(None, 0)))
eig_rev_batch = jax.jit(jax.vmap(eig_rev, in_axes=(0, 0)))


@jax.jit
def _grid_c(lam, U, Uinv, tau, rate_gamma):
    """(Rg, nT*NS*NS) log expm(Q_c * rate_gamma[r] * tau[t]) via the reversible eig.
    Layout [r, (t,pf,pt)] so flat index (t*NS+pf)*NS+pt gathers grid[r, tb, pf, pt]."""
    b = rate_gamma[:, None] * tau[None, :]                    # (Rg, nT)
    expo = jnp.exp(lam[None, None, :] * b[:, :, None])        # (Rg, nT, NS)
    P = jnp.einsum("xk,rtk,ky->rtxy", U, expo, Uinv)          # (Rg, nT, NS, NS)
    logP = jnp.log(jnp.clip(P, 1e-300, None))
    return logP.reshape(rate_gamma.shape[0], -1)


# =====================================================================================
# Ragged corpus aggregation (chunked segment_sum, fp64)
# =====================================================================================
@partial(jax.jit, static_argnums=(4,))
def _score_chunk(grid_flat, flat_e, cnt_e, seg_e, G):
    """Per-cluster gamma-bin LL contribution of one nnz chunk: segment_sum over clusters of
    cnt_e * grid_flat[:, flat_e]. Returns (G, Rg)."""
    g = grid_flat[:, flat_e]                                  # (Rg, chunk)
    contrib = (g * cnt_e[None, :]).T                          # (chunk, Rg)
    return jax.ops.segment_sum(contrib, seg_e, num_segments=G)


@partial(jax.jit, static_argnums=(4,))
def _count_chunk(Rc, flatidx_e, cnt_e, seg_e, nbins):
    """Responsibility-weighted count aggregation for one component c, one nnz chunk:
    segment_sum over the (pf,pt,tb) flat index of cnt_e * Rc[seg_e] (Rc=(G,Rg) the gamma-bin
    responsibilities of component c). Returns (nbins, Rg), nbins=NS*NS*nT."""
    w = cnt_e[:, None] * Rc[seg_e]                            # (chunk, Rg)
    return jax.ops.segment_sum(w, flatidx_e, num_segments=nbins)


@partial(jax.jit, static_argnums=())
def _alpha_dot_chunk(grid_flat, flat_e, wnnz_e):
    """sum_e wnnz_e[:, e] . grid_flat[:, flat_e]  contracted over the gamma bins and e.
    wnnz_e is (Rg, chunk), grid_flat (Rg, nT*NS*NS). Returns scalar."""
    g = grid_flat[:, flat_e]                                  # (Rg, chunk)
    return jnp.sum(g * wnnz_e)


def _chunk_bounds(n, chunk):
    return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]


# =====================================================================================
# Corpus load + family split (verbatim numpy port of FS.load_split_core)
# =====================================================================================
def load_split_core(corpus, rng, val_frac, min_counts):
    z = np.load(corpus, allow_pickle=True)
    PF = z["pf"].astype(np.int64)
    PT = z["pt"].astype(np.int64)
    TB = z["tb"].astype(np.int64)
    CNT = z["cnt"].astype(np.float64)
    cptr = z["cptr"].astype(np.int64)
    meta = z["meta"]
    tau = z["tau_centers"].astype(float)
    T = len(tau)
    G = len(cptr) - 1
    SEG = np.repeat(np.arange(G), np.diff(cptr))
    tot_cnt = np.add.reduceat(CNT, cptr[:-1])
    keep = tot_cnt >= min_counts
    fam = meta[:, 0]
    ufam = np.unique(fam)
    rng.shuffle(ufam)
    valfam = set(ufam[:int(len(ufam) * val_frac)].tolist())
    is_val = np.isin(fam, list(valfam))
    tr_g = np.where(keep & ~is_val)[0]
    va_g = np.where(keep & is_val)[0]
    trm = (keep & ~is_val)[SEG]
    vam = (keep & is_val)[SEG]
    trPF, trPT, trTB, trCNT, trSEG = PF[trm], PT[trm], TB[trm], CNT[trm], SEG[trm]
    vaPF, vaPT, vaTB, vaCNT, vaSEG = PF[vam], PT[vam], TB[vam], CNT[vam], SEG[vam]
    tr_tot = tot_cnt[tr_g].sum()
    va_tot = tot_cnt[va_g].sum()
    return dict(PF=PF, PT=PT, SEG=SEG, tau=tau, T=T, G=G, tot_cnt=tot_cnt, keep=keep,
                trPF=trPF, trPT=trPT, trTB=trTB, trCNT=trCNT, trSEG=trSEG, tr_g=tr_g, tr_tot=tr_tot,
                vaPF=vaPF, vaPT=vaPT, vaTB=vaTB, vaCNT=vaCNT, vaSEG=vaSEG, va_g=va_g, va_tot=va_tot)


# =====================================================================================
# small diagnostics (numpy; identical to FS.mi / FS.s_stats)
# =====================================================================================
def mi(pi):
    P = np.asarray(pi, float).reshape(NA, NA)
    P = P / P.sum()
    r = P.sum(1)
    c = P.sum(0)
    return float(np.sum(P * np.log(np.maximum(P, 1e-300) / np.maximum(np.outer(r, c), 1e-300))))


def s_stats(S):
    off = np.asarray(S)[_IU]
    rel = float(np.linalg.norm(off - _SLG_OFF) / max(np.linalg.norm(_SLG_OFF), 1e-300))
    if off.std() < 1e-30 or _SLG_OFF.std() < 1e-30:
        corr = 1.0
    else:
        corr = float(np.corrcoef(off, _SLG_OFF)[0, 1])
    return rel, corr


# =====================================================================================
# EM engine (JAX kernels, host-driven loop) mirroring the reference schedule
# =====================================================================================
class Engine:
    """Holds the on-device corpus (train/val nnz) and runs the batched E/M kernels."""

    def __init__(self, tau, chunk):
        self.tau = jnp.asarray(tau)
        self.nT = len(tau)
        self.nbins = NS * NS * self.nT
        self.chunk = chunk

    def put_split(self, PF, PT, TB, CNT, SEG):
        flat_e = ((TB.astype(np.int64) * NS + PF.astype(np.int64)) * NS
                  + PT.astype(np.int64)).astype(np.int32)          # grid gather [t,pf,pt]
        flatidx = ((PF.astype(np.int64) * NS + PT.astype(np.int64)) * self.nT
                   + TB.astype(np.int64)).astype(np.int32)         # count scatter [pf,pt,tb]
        return dict(
            flat_e=jax.device_put(flat_e),
            flatidx=jax.device_put(flatidx),
            cnt=jax.device_put(CNT.astype(np.float64)),
            seg=jax.device_put(SEG.astype(np.int32)),
            n=len(PF))

    # ---- E-step: per-cluster gamma-bin LL (G, K, Rg) via chunked segment_sum ----
    def cluster_ll_gamma(self, eigs_lam, eigs_U, eigs_Uinv, rate_gamma, split, G):
        K = eigs_lam.shape[0]
        out = np.zeros((G, K, rate_gamma.shape[0]))
        bounds = _chunk_bounds(split["n"], self.chunk)
        rg = jnp.asarray(rate_gamma)
        for c in range(K):
            grid = _grid_c(eigs_lam[c], eigs_U[c], eigs_Uinv[c], self.tau, rg)  # (Rg, nT*NS*NS)
            acc = jnp.zeros((G, rate_gamma.shape[0]))
            for (a, b) in bounds:
                acc = acc + _score_chunk(grid, split["flat_e"][a:b], split["cnt"][a:b],
                                         split["seg"][a:b], G)
            out[:, c, :] = np.asarray(acc)
            del grid
        return out

    # ---- M-step count tensor for component c: (NS,NS,nT,Rg) ----
    def counts_c(self, Rc_dev, split):
        bounds = _chunk_bounds(split["n"], self.chunk)
        acc = jnp.zeros((self.nbins, Rc_dev.shape[1]))
        for (a, b) in bounds:
            acc = acc + _count_chunk(Rc_dev, split["flatidx"][a:b], split["cnt"][a:b],
                                     split["seg"][a:b], self.nbins)
        return acc.reshape(NS, NS, self.nT, Rc_dev.shape[1])

    # ---- alpha CM-step objective Q(a) at candidate gamma rates ----
    def alpha_Q(self, eigs_lam, eigs_U, eigs_Uinv, rate_gamma_cand, wnnz_dev, split):
        """sum_{c,gi,e} wnnz[c][gi,e] * logP_{c, rate_gamma_cand[gi]}(tb_e)[pf_e,pt_e]."""
        K = eigs_lam.shape[0]
        rg = jnp.asarray(rate_gamma_cand)
        bounds = _chunk_bounds(split["n"], self.chunk)
        tot = 0.0
        for c in range(K):
            grid = _grid_c(eigs_lam[c], eigs_U[c], eigs_Uinv[c], self.tau, rg)
            for (a, b) in bounds:
                tot = tot + float(_alpha_dot_chunk(grid, split["flat_e"][a:b], wnnz_dev[c][:, a:b]))
            del grid
        return tot


def build_scores_np(cluster_llg, inv_ll, logw, logrho, is_inv, gamma_pos):
    """Assemble sc[g,c,r] (numpy) from the gamma-bin LL (G,K,Rg), the invariant column
    inv_ll (G,), and log weights. gamma_pos maps rate index r->gamma column (or -1)."""
    G, K, _ = cluster_llg.shape
    R = len(is_inv)
    sc = np.full((G, K, R), -np.inf)
    for r in range(R):
        if is_inv[r]:
            sc[:, :, r] = logw[None, :] + logrho[r] + inv_ll[:, None]
        else:
            gi = gamma_pos[r]
            sc[:, :, r] = logw[None, :] + logrho[r] + cluster_llg[:, :, gi]
    return sc


def responsibilities_np(sc):
    flat = sc.reshape(sc.shape[0], -1)
    m = flat.max(1, keepdims=True)
    e = np.exp(flat - m)
    Z = e.sum(1, keepdims=True)
    return (e / np.maximum(Z, 1e-300)).reshape(sc.shape)


def marginal_percount_ll_np(sc, tot):
    flat = sc.reshape(sc.shape[0], -1)
    m = flat.max(1)
    return float((m + np.log(np.exp(flat - m[:, None]).sum(1))).sum() / tot)


def class_rate_mi(R_sel, w, rho):
    J = R_sel.sum(0)
    J = J / max(J.sum(), 1e-300)
    pc = J.sum(1)
    pr = J.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mij = J * np.log(J / np.maximum(np.outer(pc, pr), 1e-300))
    mival = max(0.0, float(np.nansum(np.where(J > 0, mij, 0.0))))
    Hc = -float(np.sum(pc * np.log(np.maximum(pc, 1e-300))))
    Hr = -float(np.sum(pr * np.log(np.maximum(pr, 1e-300))))
    nmi = 0.0 if min(Hc, Hr) < 1e-9 else mival / min(Hc, Hr)
    return mival, nmi, Hc, Hr


# =====================================================================================
# main fit
# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/per_contact_trrosetta/counts.npz")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--gamma-cats", type=int, default=4)
    ap.add_argument("--pinv", type=float, default=0.2)
    ap.add_argument("--single-rate", action="store_true")
    ap.add_argument("--alpha-init", type=float, default=1.0)
    ap.add_argument("--em-iters", type=int, default=40)
    ap.add_argument("--alpha-every", type=int, default=1)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-counts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-S", dest="free_S", action="store_false")
    ap.set_defaults(free_S=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--chunk", type=int, default=4_000_000,
                    help="nnz chunk size for segment_sum aggregation (memory knob)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="run gate-1 unit-match against the numpy reference and exit")
    a = ap.parse_args()

    if a.verify:
        return verify(a)

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    mode = "single-rate" if a.single_rate else f"gammaI(g{a.gamma_cats})"
    ALPHA_CANDS = (0.6, 0.78, 1.0, 1.28, 1.66)
    print(f"# JAX device: {jax.devices()[0]}  x64={jax.config.jax_enable_x64}", flush=True)

    sp = load_split_core(a.corpus, rng, a.val_frac, a.min_counts)
    PF, PT, SEG, tau, T, G = sp["PF"], sp["PT"], sp["SEG"], sp["tau"], sp["T"], sp["G"]
    tr_g, va_g = sp["tr_g"], sp["va_g"]
    tr_tot, va_tot = sp["tr_tot"], sp["va_tot"]
    conserved = np.bincount(SEG, weights=(PF != PT).astype(float), minlength=G) == 0
    K = a.K
    print(f"# [{mode}] {G} clusters ({keep_sum(sp)} kept, {int(conserved[sp['keep']].sum())} "
          f"fully-conserved); train {len(tr_g)} / val {len(va_g)}; "
          f"{int(tr_tot):,}/{int(va_tot):,} transitions; K={K}", flush=True)

    # rate grid (identical construction to the reference)
    if a.single_rate:
        R_gamma = 1
        rate_vals = np.array([1.0])
        is_inv = np.array([False])
        rho = np.array([1.0])
        alpha = float("nan")
        p_inv = 0.0
    else:
        R_gamma = a.gamma_cats
        alpha = a.alpha_init
        g = discrete_gamma_rates(R_gamma, alpha)
        rate_vals = np.concatenate([[0.0], g])
        is_inv = np.array([True] + [False] * R_gamma)
        p_inv = a.pinv
        rho = np.concatenate([[p_inv], np.full(R_gamma, (1 - p_inv) / R_gamma)])
    R = len(rate_vals)
    gamma_pos = np.full(R, -1, int)
    gp = 0
    for r in range(R):
        if not is_inv[r]:
            gamma_pos[r] = gp
            gp += 1
    rate_gamma = rate_vals[~is_inv]                           # (R_gamma,) the gamma multipliers

    # init pi_c / S / w (identical scheme to the reference; same RNG stream)
    occ = (np.bincount(sp["trPF"], weights=sp["trCNT"], minlength=NS)
           + np.bincount(sp["trPT"], weights=sp["trCNT"], minlength=NS))
    glob = occ / occ.sum()
    pis = []
    for c in range(K):
        nz = rng.normal(0, 0.6, (NA, NA))
        nz = 0.5 * (nz + nz.T)
        p = glob.reshape(NA, NA) * np.exp(nz)
        p = (p / p.sum()).reshape(NS)
        pis.append(p)
    S = S_LG.copy()
    w = np.full(K, 1.0 / K)

    eng = Engine(tau, a.chunk)
    tr = eng.put_split(sp["trPF"], sp["trPT"], sp["trTB"], sp["trCNT"], sp["trSEG"])
    va = eng.put_split(sp["vaPF"], sp["vaPT"], sp["vaTB"], sp["vaCNT"], sp["vaSEG"])

    pis_dev = jnp.asarray(np.array(pis))
    S_dev = jnp.asarray(S)
    Qs = met_Q_batch(S_dev, pis_dev)
    lam, U, Uinv = eig_rev_batch(Qs, jnp.clip(pis_dev, 1e-12, None))

    prev = None
    trll = float("nan")
    monotone = True
    it = 0
    for it in range(a.em_iters):
        logw = np.log(w + 1e-300)
        logrho = np.log(rho + 1e-300)
        # E-step on train
        cll = eng.cluster_ll_gamma(lam, U, Uinv, rate_gamma, tr, G)     # (G,K,Rg)
        inv_ll = np.where(conserved, 0.0, -np.inf)
        sc_full = build_scores_np(cll, inv_ll, logw, logrho, is_inv, gamma_pos)
        sc = sc_full[tr_g]
        trll = marginal_percount_ll_np(sc, tr_tot)
        if prev is not None and trll < prev - 1e-9:
            monotone = False
        Rr = responsibilities_np(sc)                                    # (n_tr,K,R)
        R_full = np.zeros((G, K, R))
        R_full[tr_g] = Rr
        # M-step weights
        w = Rr.sum((0, 2)) + 1e-9
        w /= w.sum()
        rho = Rr.sum((0, 1)) + 1e-12
        rho /= rho.sum()
        if not a.single_rate:
            p_inv = float(rho[0])

        # M-step pi_c, S (rate-scaled HR); Ncr fixed across inner rounds
        Rgam_full = R_full[:, :, ~is_inv]                               # (G,K,Rg)
        Ncr = []
        for c in range(K):
            Rc_dev = jnp.asarray(Rgam_full[:, c, :])
            Ncr.append(eng.counts_c(Rc_dev, tr))                        # (NS,NS,nT,Rg) device
        S_dev = jnp.asarray(S)
        pis_dev = jnp.asarray(np.array(pis))
        rg_dev = jnp.asarray(rate_gamma)
        for _inner in range(a.inner):
            Cnum_tot = jnp.zeros((NA, NA))
            Hden_tot = jnp.zeros((NA, NA))
            new_pis = []
            for c in range(K):
                Q_c = met_Q_sqrt(S_dev, pis_dev[c])
                lam_c, U_c, Uinv_c = eig_rev(Q_c, jnp.clip(pis_dev[c], 1e-12, None))
                N_c, T_c = bridge_sum_c(lam_c, U_c, Uinv_c, Q_c, eng.tau, rg_dev, Ncr[c])
                pi_c = mstep_pi_metropolis(S_dev, pis_dev[c], N_c, T_c)
                Cnum_c, Hden_c = met_S_suffstats(N_c, T_c, pi_c)
                Cnum_tot = Cnum_tot + Cnum_c
                Hden_tot = Hden_tot + Hden_c
                new_pis.append(pi_c)
            pis_dev = jnp.stack(new_pis)
            if a.free_S:
                S_dev = jnp.where(Hden_tot > 0, Cnum_tot / jnp.maximum(Hden_tot, 1e-300), 0.0)
                S_dev = 0.5 * (S_dev + S_dev.T)
                S_dev = S_dev * (1.0 - _EYE_NA)
        S = np.asarray(S_dev)
        pis = [np.asarray(pis_dev[c]) for c in range(K)]
        Qs = met_Q_batch(S_dev, pis_dev)
        lam, U, Uinv = eig_rev_batch(Qs, jnp.clip(pis_dev, 1e-12, None))

        # M-step alpha (Gamma shape) -- rate values only, ECM CM-step
        if (not a.single_rate and R_gamma >= 2 and it % a.alpha_every == 0):
            wnnz_dev = []
            for c in range(K):
                wc = (sp["trCNT"][None, :] * Rgam_full[sp["trSEG"], c, :].T)   # (Rg, n_tr)
                wnnz_dev.append(jax.device_put(wc))
            cands = np.unique(np.concatenate([[alpha], alpha * np.asarray(ALPHA_CANDS)]))
            cands = np.clip(cands, 0.02, 50.0)
            vals = []
            for ca in cands:
                rgc = discrete_gamma_rates(R_gamma, ca)
                vals.append(eng.alpha_Q(lam, U, Uinv, rgc, wnnz_dev, tr))
            best = float(cands[int(np.argmax(vals))])
            alpha = best
            g = discrete_gamma_rates(R_gamma, alpha)
            rate_vals = np.concatenate([[0.0], g])
            rate_gamma = rate_vals[~is_inv]
            del wnnz_dev

        srel, scorr = s_stats(S)
        extra = "" if a.single_rate else f" alpha={alpha:.3f} pinv={p_inv:.3f}"
        print(f"  it {it:2d}: train_mix={trll:.4f}  w={np.round(np.sort(w)[::-1], 3)}"
              f"  MI(pi)={sorted([round(mi(p), 3) for p in pis], reverse=True)}"
              f"  Srel={srel:.3f}{extra}  [{time.time()-t0:.0f}s]", flush=True)
        if prev is not None and abs(trll - prev) < a.tol:
            print("  converged", flush=True)
            break
        prev = trll

    # ---- final held-out scoring ----
    logw = np.log(w + 1e-300)
    logrho = np.log(rho + 1e-300)
    cll_va = eng.cluster_ll_gamma(lam, U, Uinv, rate_gamma, va, G)
    inv_ll = np.where(conserved, 0.0, -np.inf)
    sc_va = build_scores_np(cll_va, inv_ll, logw, logrho, is_inv, gamma_pos)[va_g]
    vall = marginal_percount_ll_np(sc_va, va_tot)
    cll_tr = eng.cluster_ll_gamma(lam, U, Uinv, rate_gamma, tr, G)
    sc_tr = build_scores_np(cll_tr, inv_ll, logw, logrho, is_inv, gamma_pos)[tr_g]
    R_sel = responsibilities_np(sc_tr)
    cr_mi, cr_nmi, Hc, Hr = class_rate_mi(R_sel, w, rho)
    srel, scorr = s_stats(S)
    order = np.argsort(w)[::-1]
    if a.single_rate:
        eff_mult = [1.0]
        eff_w = [1.0]
        cond_mean = 1.0
        rate_cv = 0.0
    else:
        gamma_idx = [r for r in range(R) if not is_inv[r]]
        gw = rho[gamma_idx]
        gv = np.asarray([rate_vals[r] for r in gamma_idx])
        gwn = gw / max(gw.sum(), 1e-300)
        cond_mean = float((gwn * gv).sum())
        eff = gv / max(cond_mean, 1e-300)
        eff_mult = [float(x) for x in eff]
        eff_w = [float(x) for x in gwn]
        rate_cv = float(np.sqrt((gwn * (eff - 1.0) ** 2).sum()))
    res = dict(
        mode=mode, K=K, single_rate=bool(a.single_rate), gamma_cats=int(R_gamma),
        val_per_count_ll=float(vall), train_per_count_ll=float(trll),
        monotone=bool(monotone), n_em_iters=int(it + 1),
        weights=[float(w[c]) for c in order],
        mi_pi=[float(mi(pis[c])) for c in order],
        mi_weighted=float(sum(w[c] * mi(pis[c]) for c in range(K))),
        rate_weights=[float(x) for x in rho], rate_vals=[float(x) for x in rate_vals],
        eff_rate_mult=eff_mult, eff_rate_weight=eff_w, rate_cv=rate_cv,
        alpha=(None if a.single_rate else float(alpha)),
        p_inv=(None if a.single_rate else float(p_inv)),
        mean_rate=float(sum(rho[r] * rate_vals[r] for r in range(R))),
        class_rate_mi_nats=cr_mi, class_rate_nmi=cr_nmi, H_class=Hc, H_rate=Hr,
        shared_s_rel_fro=srel, shared_s_offdiag_corr=scorr,
        n_train_clusters=int(len(tr_g)), n_val_clusters=int(len(va_g)),
        n_conserved_kept=int(conserved[sp["keep"]].sum()), seed=a.seed,
        backend="jax", device=str(jax.devices()[0]))
    print(f"# [{mode}] K={K}: VAL/count={vall:.4f} train={trll:.4f} monotone={monotone} "
          f"iters={it+1}", flush=True)
    if not a.single_rate:
        print(f"#   alpha={alpha:.3f} p_inv={p_inv:.3f} rate_CV={rate_cv:.3f} "
              f"rate_w={[round(x,3) for x in rho]}", flush=True)
    print(f"#   w-mean MI(pi_c)={res['mi_weighted']:.4f}  sharedSrel={srel:.4f} "
          f"sharedScorr={scorr:.4f}", flush=True)
    print(f"#   MI(pi_c) per comp = {[round(x,3) for x in res['mi_pi']]}", flush=True)
    print(f"#   weights  per comp = {[round(x,3) for x in res['weights']]}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        np.savez(a.out.replace(".json", ".npz"),
                 pis=np.array(pis), S=np.asarray(S), weights=np.asarray(w),
                 mi_pi=np.array([mi(pis[c]) for c in range(K)]),
                 tau=np.asarray(tau), K=int(K),
                 val_per_count_ll=float(vall), train_per_count_ll=float(trll))
        print(f"# wrote {a.out} + .npz", flush=True)
    return res


def keep_sum(sp):
    return int(sp["keep"].sum())


# =====================================================================================
# gate-1 verification: unit-match against the numpy reference on a small subset
# =====================================================================================
def verify(a):
    """Compare the JAX E-step (per-cluster scores, responsibilities) and M-step (pi_c, S)
    against the numpy reference (fit_coupling_mixture_rateI) on the first Nsub clusters,
    K=2, R_gamma=1 (R=2), one EM iteration, fp64. Reports max abs discrepancies."""
    import fit_coupling_mixture_rateI as REF
    import fit_pair_models as FP

    Nsub = 400
    K = 2
    R_gamma = 1
    inner = 2
    alpha = 1.0
    print(f"# [verify] device={jax.devices()[0]} Nsub={Nsub} K={K} R_gamma={R_gamma} "
          f"inner={inner}", flush=True)

    z = np.load(a.corpus, allow_pickle=True)
    PF = z["pf"].astype(np.int64)
    PT = z["pt"].astype(np.int64)
    TB = z["tb"].astype(np.int64)
    CNT = z["cnt"].astype(np.float64)
    cptr = z["cptr"].astype(np.int64)
    tau = z["tau_centers"].astype(float)
    T = len(tau)
    lo, hi = int(cptr[0]), int(cptr[Nsub])
    sPF, sPT, sTB, sCNT = PF[lo:hi], PT[lo:hi], TB[lo:hi], CNT[lo:hi]
    SEG = np.repeat(np.arange(Nsub), np.diff(cptr[:Nsub + 1]))
    G = Nsub
    gsel = np.arange(Nsub)
    conserved = np.bincount(SEG, weights=(sPF != sPT).astype(float), minlength=G) == 0

    # deterministic init (shared by both paths)
    rng = np.random.default_rng(123)
    occ = np.bincount(sPF, weights=sCNT, minlength=NS) + np.bincount(sPT, weights=sCNT, minlength=NS)
    glob = occ / occ.sum()
    pis = []
    for c in range(K):
        nz = rng.normal(0, 0.6, (NA, NA))
        nz = 0.5 * (nz + nz.T)
        p = glob.reshape(NA, NA) * np.exp(nz)
        p = (p / p.sum()).reshape(NS)
        pis.append(p)
    S = S_LG.copy()
    w = np.full(K, 1.0 / K)

    g = discrete_gamma_rates(R_gamma, alpha)
    rate_vals = np.concatenate([[0.0], g])
    is_inv = np.array([True] + [False] * R_gamma)
    R = len(rate_vals)
    gamma_idx = [r for r in range(R) if not is_inv[r]]
    p_inv = 0.2
    rho = np.concatenate([[p_inv], np.full(R_gamma, (1 - p_inv) / R_gamma)])
    logw = np.log(w + 1e-300)
    logrho = np.log(rho + 1e-300)

    # ---------- reference E-step ----------
    Qs_ref = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(K)]
    eigs_ref = [FP.eig_rev(Qs_ref[c], np.clip(pis[c], 1e-12, None)) for c in range(K)]
    sc_ref = REF.build_scores(eigs_ref, pis, rate_vals, is_inv, logw, logrho, tau,
                              sPF, sPT, sTB, sCNT, SEG, G, gsel, conserved)
    Rr_ref = REF.responsibilities(sc_ref)

    # ---------- JAX E-step ----------
    gamma_pos = np.full(R, -1, int)
    gp = 0
    for r in range(R):
        if not is_inv[r]:
            gamma_pos[r] = gp
            gp += 1
    rate_gamma = rate_vals[~is_inv]
    eng = Engine(tau, a.chunk)
    split = eng.put_split(sPF, sPT, sTB, sCNT, SEG)
    pis_dev = jnp.asarray(np.array(pis))
    S_dev = jnp.asarray(S)
    Qs_dev = met_Q_batch(S_dev, pis_dev)
    lam, U, Uinv = eig_rev_batch(Qs_dev, jnp.clip(pis_dev, 1e-12, None))
    cll = eng.cluster_ll_gamma(lam, U, Uinv, rate_gamma, split, G)
    inv_ll = np.where(conserved, 0.0, -np.inf)
    sc_jax = build_scores_np(cll, inv_ll, logw, logrho, is_inv, gamma_pos)
    Rr_jax = responsibilities_np(sc_jax)

    # generator sanity
    dQ = float(np.max(np.abs(np.asarray(Qs_dev) - np.array(Qs_ref))))
    # E-step discrepancies (compare finite entries)
    fin = np.isfinite(sc_ref) & np.isfinite(sc_jax)
    d_sc = float(np.max(np.abs(sc_ref[fin] - sc_jax[fin])))
    d_Rr = float(np.max(np.abs(Rr_ref - Rr_jax)))
    same_inf = bool(np.all(np.isinf(sc_ref) == np.isinf(sc_jax)))

    # ---------- reference M-step (weights + pi/S) ----------
    Rr_ref_full = np.zeros((G, K, R))
    Rr_ref_full[gsel] = Rr_ref
    flat = (sPF * NS + sPT) * T + sTB
    S_ref2, pis_ref2, Qs_ref2, eigs_ref2 = REF.mstep_shared(
        Rr_ref_full, rate_vals, gamma_idx, S, pis, tau, inner, True,
        SEG, sCNT, flat, T)

    # ---------- JAX M-step ----------
    Rgam_full = Rr_ref_full[:, :, ~is_inv]                    # use SAME responsibilities
    Ncr = []
    for c in range(K):
        Ncr.append(eng.counts_c(jnp.asarray(Rgam_full[:, c, :]), split))
    S_dev = jnp.asarray(S)
    pis_dev = jnp.asarray(np.array(pis))
    rg_dev = jnp.asarray(rate_gamma)
    for _inner in range(inner):
        Cnum_tot = jnp.zeros((NA, NA))
        Hden_tot = jnp.zeros((NA, NA))
        new_pis = []
        for c in range(K):
            Q_c = met_Q_sqrt(S_dev, pis_dev[c])
            lam_c, U_c, Uinv_c = eig_rev(Q_c, jnp.clip(pis_dev[c], 1e-12, None))
            N_c, T_c = bridge_sum_c(lam_c, U_c, Uinv_c, Q_c, eng.tau, rg_dev, Ncr[c])
            pi_c = mstep_pi_metropolis(S_dev, pis_dev[c], N_c, T_c)
            Cnum_c, Hden_c = met_S_suffstats(N_c, T_c, pi_c)
            Cnum_tot = Cnum_tot + Cnum_c
            Hden_tot = Hden_tot + Hden_c
            new_pis.append(pi_c)
        pis_dev = jnp.stack(new_pis)
        S_dev = jnp.where(Hden_tot > 0, Cnum_tot / jnp.maximum(Hden_tot, 1e-300), 0.0)
        S_dev = 0.5 * (S_dev + S_dev.T)
        S_dev = S_dev * (1.0 - _EYE_NA)
    pis_jax2 = [np.asarray(pis_dev[c]) for c in range(K)]
    S_jax2 = np.asarray(S_dev)

    d_S = float(np.max(np.abs(S_ref2 - S_jax2)))
    d_pi = float(np.max([np.max(np.abs(pis_ref2[c] - pis_jax2[c])) for c in range(K)]))

    # also cross-check the M-step count tensor against a numpy bincount
    c0 = 0
    r0 = gamma_idx[0]
    wnnz = sCNT * Rr_ref_full[SEG, c0, r0]
    Ncr_np = np.bincount(flat, weights=wnnz, minlength=NS * NS * T).reshape(NS, NS, T)
    Ncr_jax0 = np.asarray(Ncr[c0])[:, :, :, 0]
    d_count = float(np.max(np.abs(Ncr_np - Ncr_jax0)))

    print("# ===== gate-1 unit-match (JAX vs numpy reference, fp64) =====", flush=True)
    print(f"#   max|Q_c(jax) - Q_c(ref)|            = {dQ:.3e}", flush=True)
    print(f"#   max|count tensor(jax) - bincount|   = {d_count:.3e}", flush=True)
    print(f"#   E-step: -inf pattern identical      = {same_inf}", flush=True)
    print(f"#   E-step: max|sc(jax) - sc(ref)|      = {d_sc:.3e}", flush=True)
    print(f"#   E-step: max|resp(jax) - resp(ref)|  = {d_Rr:.3e}", flush=True)
    print(f"#   M-step: max|pi_c(jax) - pi_c(ref)|  = {d_pi:.3e}", flush=True)
    print(f"#   M-step: max|S(jax) - S(ref)|        = {d_S:.3e}", flush=True)
    worst = max(dQ, d_count, d_sc, d_Rr, d_pi, d_S)
    ok = same_inf and worst < 1e-5
    print(f"#   WORST discrepancy = {worst:.3e}   GATE-1 {'PASS' if ok else 'FAIL'} "
          f"(threshold 1e-5)", flush=True)
    return dict(dQ=dQ, d_count=d_count, d_sc=d_sc, d_Rr=d_Rr, d_pi=d_pi, d_S=d_S,
                same_inf=same_inf, worst=worst, pass_=ok)


if __name__ == "__main__":
    main()
