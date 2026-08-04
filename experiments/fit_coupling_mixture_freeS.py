#!/usr/bin/env python3
r"""Mixture of coupling components over size-2 clusters -- with ONE SHARED, free
exchangeability matrix.

Extends fit_coupling_mixture.py: instead of pinning the pair-chain exchangeability at
S=LG08, the mixture fits a SINGLE shared 20x20 symmetric S (warm-started at LG08),
common to all K components, while each component keeps its own stationary pi_c (the
coupling).  Picture: one universal exchange process + K coupling archetypes.  A single
shared S (190 params total) is well-determined by the pooled counts; a separate S per
component is not (and empirically the per-component S's come out ~identical anyway).

Per component the endpoint-conditioned HR / bridge E-step gives expected transition
usage N_c and dwell T_c on the responsibility-weighted count tensor, under
Q_c = metropolis_sqrt(S, pi_c).  M-step:
  (a) pi-step  : pi_c = mstep_pi_metropolis(S, pi_c, N_c, T_c, "sqrt")   (at current S)
  (b) S-step   : POOL the Metropolis S sufficient statistics over all components --
                 usage numerator Cnum_c(N_c) and pi_c-dependent exposure denominator
                 Hden_c(T_c, pi_c) -- then S = (sum_c Cnum_c) / (sum_c Hden_c).
  (c) weights  : w_c = mean responsibility.
With --fixed-S the pooled S-step is skipped and S stays LG08 (reproducing
fit_coupling_mixture.py) so the two modes share EXACTLY the same corpus, family split,
init and EM schedule for an apples-to-apples comparison.

Mixture EM over clusters g with sparse tau-binned counts n_g(pf,pt,tb):
  E: r_{gc} = softmax_c[ log w_c + sum_e n_g^e logP_c(tb^e)[pf^e,pt^e] ]
  M: N_c = sum_g r_{gc} n_g ; refit pi_c and the shared S ; w_c = mean_g r_{gc}
Held-out clusters scored as logsumexp_c[log w_c + LL_c(g)] (marginal mixture).

Corpus: data/per_contact_trrosetta/counts.npz (build_per_contact_corpus.py)."""
from __future__ import annotations
import argparse, json, time
from collections import namedtuple
import numpy as np
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import fit_pair_models as FP
from fit_pair_models import NA, NS
from composite_potts_phylo_elbo import S_LG08

S_LG = np.asarray(S_LG08, float)[:NA, :NA].copy()
S_LG = 0.5 * (S_LG + S_LG.T); np.fill_diagonal(S_LG, 0.0)
_IU = np.triu_indices(NA, 1)                                 # 190 off-diagonal entries
_SLG_OFF = S_LG[_IU]


def mi(pi):
    P = np.asarray(pi, float).reshape(NA, NA); P = P / P.sum()
    r = P.sum(1); c = P.sum(0)
    return float(np.sum(P * np.log(np.maximum(P, 1e-300) / np.maximum(np.outer(r, c), 1e-300))))


def s_stats(S):
    """Divergence of the fitted shared S from LG08 (off-diagonal only): relative
    Frobenius ||S-S_LG||/||S_LG|| and Pearson correlation of the 190 off-diagonal
    entries (shape agreement; scale-invariant)."""
    off = S[_IU]
    rel = float(np.linalg.norm(off - _SLG_OFF) / max(np.linalg.norm(_SLG_OFF), 1e-300))
    if off.std() < 1e-30 or _SLG_OFF.std() < 1e-30:
        corr = 1.0
    else:
        corr = float(np.corrcoef(off, _SLG_OFF)[0, 1])
    return rel, corr


def met_S_suffstats(N, T, pi, kernel):
    """The Metropolis S-step sufficient statistics (from mstep_metropolis), returned
    UNreduced so they can be pooled across mixture components: usage numerator
    Cnum (symmetric, pi-free) and exposure denominator Hden (symmetric, depends on pi).
    Pooled shared S = (sum_c Cnum_c) / (sum_c Hden_c)."""
    piM = np.maximum(pi.reshape(NA, NA), 1e-300); T2 = T.reshape(NA, NA)
    N4 = N.reshape(NA, NA, NA, NA)
    Cdir = np.einsum("ijkj->ik", N4) + np.einsum("ijil->jl", N4)
    Cnum = Cdir + Cdir.T
    f1 = FP._met_f(piM[:, None, :], piM[None, :, :], kernel)   # (a,b,j) comp1 exposure
    H1 = (T2[:, None, :] * f1).sum(2)
    f2 = FP._met_f(piM[:, :, None], piM[:, None, :], kernel)   # (i,a,b) comp2 exposure
    H2 = (T2[:, :, None] * f2).sum(0)
    Hden = (H1 + H2); Hden = Hden + Hden.T
    return Cnum, Hden


def logP_grid(Q, pi, tau):
    d = np.sqrt(np.maximum(pi, 1e-300)); dinv = 1.0 / d
    Qs = d[:, None] * Q * dinv[None, :]; Qs = 0.5 * (Qs + Qs.T)
    lam, V = np.linalg.eigh(Qs)
    out = np.empty((len(tau), NS, NS))
    for t, tt in enumerate(tau):
        M = (V * np.exp(lam * tt)[None, :]) @ V.T
        out[t] = np.log(np.clip(dinv[:, None] * M * d[None, :], 1e-300, None))
    return out


def cluster_ll(logP, PF, PT, TB, CNT, SEG, G, logw):
    return logw + np.bincount(SEG, weights=CNT * logP[TB, PF, PT], minlength=G)


def mix_pc(grids, w, PF, PT, TB, CNT, SEG, G, gsel, tot):
    lls = np.array([cluster_ll(grids[c], PF, PT, TB, CNT, SEG, G, np.log(w[c] + 1e-300))
                    for c in range(len(w))]).T[gsel]
    m = lls.max(1)
    return (m + np.log(np.exp(lls - m[:, None]).sum(1))).sum() / tot


def mstep_shared(Ncounts, S, pis, tau, inner, free_S):
    """Joint M-step at fixed responsibilities (fixed Ncounts).  For `inner` rounds:
    per component run the HR bridge E-step under Q_c=met_sqrt(S,pi_c), update pi_c at
    the current shared S, accumulate that component's pooled S-suff-stats; then (if
    free_S) refit the single shared S from the pooled numerator/denominator.
    Returns (S, pis, Qs)."""
    K = len(pis); pis = [p.copy() for p in pis]
    for _ in range(inner):
        Cnum_tot = np.zeros((NA, NA)); Hden_tot = np.zeros((NA, NA))
        for c in range(K):
            Q_c = FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt")
            N_c, T_c = FP.estep(Q_c, pis[c], tau, Ncounts[c])
            pis[c] = FP.mstep_pi_metropolis(S, pis[c], N_c, T_c, "sqrt")   # pi at current S
            Cnum_c, Hden_c = met_S_suffstats(N_c, T_c, pis[c], "sqrt")
            Cnum_tot += Cnum_c; Hden_tot += Hden_c
        if free_S:                                            # pooled shared-S update
            S = np.where(Hden_tot > 0, Cnum_tot / np.maximum(Hden_tot, 1e-300), 0.0)
            S = 0.5 * (S + S.T); np.fill_diagonal(S, 0.0)
    Qs = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(K)]
    return S, pis, Qs


# Shared corpus load + held-out family split, used by this fitter and imported by
# fit_coupling_mixture_rateI (as FS.load_split_core).  The caller passes its own rng
# so the family shuffle consumes from the same stream used for downstream init,
# preserving each fitter's exact RNG sequence.  Per-fitter extras (flat index,
# conserved-cluster flag, global occupancy) are derived from these fields in the caller.
Split = namedtuple("Split",
                   "PF PT SEG tau T G tot_cnt keep "
                   "trPF trPT trTB trCNT trSEG tr_g tr_tot "
                   "vaPF vaPT vaTB vaCNT vaSEG va_g va_tot")


def load_split_core(corpus, rng, val_frac, min_counts):
    """Load the size-2 contact count corpus and split clusters by family into
    train/val.  `rng` (a numpy Generator) is used in place for the family shuffle."""
    z = np.load(corpus, allow_pickle=True)
    PF = z["pf"].astype(np.int64); PT = z["pt"].astype(np.int64)
    TB = z["tb"].astype(np.int64); CNT = z["cnt"].astype(np.float64)
    cptr = z["cptr"].astype(np.int64); meta = z["meta"]; tau = z["tau_centers"].astype(float)
    T = len(tau); G = len(cptr) - 1
    SEG = np.repeat(np.arange(G), np.diff(cptr))
    tot_cnt = np.add.reduceat(CNT, cptr[:-1])
    keep = tot_cnt >= min_counts
    fam = meta[:, 0]; ufam = np.unique(fam); rng.shuffle(ufam)
    valfam = set(ufam[:int(len(ufam) * val_frac)].tolist())
    is_val = np.isin(fam, list(valfam))
    tr_g = np.where(keep & ~is_val)[0]; va_g = np.where(keep & is_val)[0]
    trm = (keep & ~is_val)[SEG]; vam = (keep & is_val)[SEG]
    trPF, trPT, trTB, trCNT, trSEG = PF[trm], PT[trm], TB[trm], CNT[trm], SEG[trm]
    vaPF, vaPT, vaTB, vaCNT, vaSEG = PF[vam], PT[vam], TB[vam], CNT[vam], SEG[vam]
    tr_tot = tot_cnt[tr_g].sum(); va_tot = tot_cnt[va_g].sum()
    return Split(PF, PT, SEG, tau, T, G, tot_cnt, keep,
                 trPF, trPT, trTB, trCNT, trSEG, tr_g, tr_tot,
                 vaPF, vaPT, vaTB, vaCNT, vaSEG, va_g, va_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/per_contact_trrosetta/counts.npz")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--em-iters", type=int, default=60)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-counts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-S", dest="free_S", action="store_false",
                    help="hold the shared S=LG08 (reproduces fit_coupling_mixture.py)")
    ap.set_defaults(free_S=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); t0 = time.time()
    mode = "shared-free-S" if a.free_S else "fixed-S"

    (PF, PT, SEG, tau, T, G, tot_cnt, keep,
     trPF, trPT, trTB, trCNT, trSEG, tr_g, tr_tot,
     vaPF, vaPT, vaTB, vaCNT, vaSEG, va_g, va_tot) = load_split_core(
        a.corpus, rng, a.val_frac, a.min_counts)
    K = a.K
    print(f"# [{mode}] {G} clusters ({keep.sum()} kept); train {len(tr_g)} / val {len(va_g)} "
          f"clusters (family split); {int(tr_tot):,}/{int(va_tot):,} transitions; K={K}", flush=True)

    occ = np.bincount(trPF, weights=trCNT, minlength=NS) + np.bincount(trPT, weights=trCNT, minlength=NS)
    glob = occ / occ.sum()
    flat = (trPF * NS + trPT) * T + trTB                       # nnz -> (400,400,T) flat index

    # init: global stationary perturbed by symmetric log-noise -> diverse components
    pis = []
    for c in range(K):
        nz = rng.normal(0, 0.6, (NA, NA)); nz = 0.5 * (nz + nz.T)
        p = glob.reshape(NA, NA) * np.exp(nz); p = (p / p.sum()).reshape(NS)
        pis.append(p)
    S = S_LG.copy()                                            # single shared S, warm-init LG08
    w = np.full(K, 1.0 / K)
    Qs = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(K)]

    prev = None; trll = float("nan")
    for it in range(a.em_iters):
        grids = [logP_grid(Qs[c], pis[c], tau) for c in range(K)]
        scores = np.array([cluster_ll(grids[c], trPF, trPT, trTB, trCNT, trSEG, G,
                                      np.log(w[c] + 1e-300)) for c in range(K)]).T
        # Score the CURRENT model (pis, w) before the M-step, so trll is the LL of
        # the model carried into this iteration and the convergence test is on a
        # consistent quantity (matches fit_coupling_mixture_rateI.py).
        trll = mix_pc(grids, w, trPF, trPT, trTB, trCNT, trSEG, G, tr_g, tr_tot)
        sc = scores[tr_g]; sc = sc - sc.max(1, keepdims=True)
        R = np.exp(sc); R /= R.sum(1, keepdims=True)
        w = R.mean(0) + 1e-8; w /= w.sum()
        rc_full = np.zeros((G, K)); rc_full[tr_g] = R
        Ncounts = []
        for c in range(K):
            wnnz = trCNT * rc_full[trSEG, c]
            Ncounts.append(np.bincount(flat, weights=wnnz,
                                       minlength=NS * NS * T).reshape(NS, NS, T))
        S, pis, Qs = mstep_shared(Ncounts, S, pis, tau, a.inner, a.free_S)
        srel, scorr = s_stats(S)
        print(f"  it {it:2d}: train_mix={trll:.4f}  w={np.round(np.sort(w)[::-1],3)}  "
              f"MI(pi_c)={sorted([round(mi(p),3) for p in pis], reverse=True)}  "
              f"Srel={srel:.3f} Scorr={scorr:.3f}  [{time.time()-t0:.0f}s]", flush=True)
        if prev is not None and abs(trll - prev) < 1e-5:
            print("  converged", flush=True); break
        prev = trll

    grids = [logP_grid(Qs[c], pis[c], tau) for c in range(K)]
    vall = mix_pc(grids, w, vaPF, vaPT, vaTB, vaCNT, vaSEG, G, va_g, va_tot)
    srel, scorr = s_stats(S)
    order = np.argsort(w)[::-1]
    res = dict(mode=mode, K=K, val_per_count_ll=float(vall), train_per_count_ll=float(trll),
               weights=[float(w[c]) for c in order],
               mi_pi=[float(mi(pis[c])) for c in order],
               mi_weighted=float(sum(w[c] * mi(pis[c]) for c in range(K))),
               shared_s_rel_fro=srel, shared_s_offdiag_corr=scorr,
               n_train_clusters=int(len(tr_g)), n_val_clusters=int(len(va_g)))
    print(f"# [{mode}] K={K}: VAL/count={vall:.4f} train={trll:.4f} "
          f"w-mean MI(pi_c)={res['mi_weighted']:.4f} sharedSrel={srel:.4f} sharedScorr={scorr:.4f}",
          flush=True)
    print(f"#   MI(pi_c) per comp = {[round(x,3) for x in res['mi_pi']]}", flush=True)
    print(f"#   weights  per comp = {[round(x,3) for x in res['weights']]}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
    return res


if __name__ == "__main__":
    main()
