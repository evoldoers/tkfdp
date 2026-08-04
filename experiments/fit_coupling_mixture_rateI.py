#!/usr/bin/env python3
r"""Mixture of coupling components + Gamma+I rate heterogeneity over size-2 clusters.

Extends fit_coupling_mixture_freeS.py.  That script fits ONE shared free single-site
exchangeability S (warm-init LG08) and K pairing classes, each a symmetric joint
stationary pi_c (the coupling), with soft per-cluster responsibilities and a family
held-out split.  This script adds a SECOND, ORTHOGONAL latent: a per-cluster rate class
drawn from a discrete Gamma+Invariant grid (Yang 1994/1996), independent of the pairing
class.  Everything stays exchangeable (symmetric S, symmetric pi_c).

Model.  Per size-2 cluster g with tau-binned transition counts n_g(pf,pt,tb),
the composite likelihood marginalises over BOTH a pairing class c in {1..K}
(weight w_c) AND a rate class r (weight rho_r), which are independent:

    L(g) = sum_c sum_r  w_c rho_r  prod_e  P_c(rate_r * tau_e)[pf_e -> pt_e]^{n_g^e}

  * P_c(t) = expm(Q_c t),  Q_c = metropolis_sqrt(S, pi_c) : exchangeable pair
    generator for class c (shared free S; symmetric pi_c).
  * rate_r : discrete Gamma+I grid.  (R-1) Gamma categories with rate
    multipliers g_1..g_{R-1} of mean 1 and shape alpha (Yang's mean
    discretisation, rate_hetero.discrete_gamma_rates), total weight 1-p_inv;
    PLUS one INVARIANT category (rate 0, weight p_inv) that gives probability 1 to
    no-change transitions (pf==pt) and 0 to any substitution -- so it supports only
    fully-conserved clusters.  The rate is PER-CLUSTER: the size-2 pair shares one
    latent rate across its cherries, and it scales the branch length rate_r*tau_e.

EM (ECM, monotone in the observed-data marginal LL):
  E : r_{gcr} propto w_c rho_r exp[ sum_e n_g^e log P_{c,r}(tb_e) ],  P_{c,r} at
      branch rate_r*tau_center[tb].  The invariant bin's log P is 0 on the diagonal
      and -inf off it, so it only ever competes for clusters with NO substitutions.
  M : w_c   = sum_{g,r} r_{gcr}                         (exact)
      rho_r = sum_{g,c} r_{gcr} ;  p_inv = rho_invariant (exact)
      pi_c, S : HR / bridge M-step of the Metropolis-sqrt chain on the
                responsibility-weighted counts placed at the RATE-SCALED branch
                length rate_r*tau (aggregate the HR usage/dwell N_c,T_c over the
                Gamma rates; the invariant bin contributes 0 dwell so it drops out).
                pi_c via mstep_pi_metropolis; shared S pooled over all (c, Gamma-r).
      alpha : 1-D search (local, current alpha always a candidate => non-decrease)
              maximising the rate-marginal expected complete-data LL.

Held-out clusters (unseen families) scored as logsumexp_{c,r}[log w_c + log rho_r +
LL_{c,r}(g)] / total val counts.  --single-rate collapses the rate grid to a single
bin at rate 1 (no invariant, no alpha), reproducing fit_coupling_mixture_freeS.py on
the IDENTICAL corpus / split / init for an apples-to-apples baseline.

Corpus: data/per_contact_trrosetta/counts.npz (build_per_contact_corpus.py)."""
from __future__ import annotations
import argparse, json, time, os
# This script is pure numpy/scipy, but it imports S_LG08 from tkfdp.lg08, which pulls in
# JAX.  Force JAX to CPU so that running many of these in parallel does not race for GPU
# memory (concurrent GPU inits OOM and hang one worker in a cuDNN retry loop).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import fit_pair_models as FP
from fit_pair_models import NA, NS
import fit_coupling_mixture_freeS as FS   # shared diagnostics + shared-S suff. stats
from tkfdp.lg08 import S_LG08
from tkfdp.coupling.dynfield.phylo_elbo.rate_hetero import discrete_gamma_rates

S_LG = np.asarray(S_LG08, float)[:NA, :NA].copy()
S_LG = 0.5 * (S_LG + S_LG.T); np.fill_diagonal(S_LG, 0.0)
_IU = np.triu_indices(NA, 1)                                 # 190 off-diagonal entries
_SLG_OFF = S_LG[_IU]


# ---------- small diagnostics ----------
# FS.mi(), FS.s_stats(), FS.met_S_suffstats() are shared with fit_coupling_mixture_freeS
# (imported as FS above) so the shared-S sufficient-statistics math and the MI
# diagnostic stay in one place; call them as FS.mi / FS.s_stats / FS.met_S_suffstats.


# ---------- log P grids and cluster scoring ----------
def logP_grid(eig, tau, rate):
    """(T,NS,NS) log expm(Q * rate * tau[t]) from a cached reversible eig
    (lam,U,Uinv)=FP.eig_rev(Q,pi).  rate scales the branch length (Gamma multiplier)."""
    lam, U, Uinv = eig
    out = np.empty((len(tau), NS, NS))
    for t, tt in enumerate(tau):
        P = (U * np.exp(lam * rate * tt)[None, :]) @ Uinv
        out[t] = np.log(np.clip(P, 1e-300, None))
    return out


def cluster_gamma_ll(grid, PF, PT, TB, CNT, SEG, G):
    """Per-cluster sum_e n^e logP for one Gamma rate bin: (G,) via bincount."""
    return np.bincount(SEG, weights=CNT * grid[TB, PF, PT], minlength=G)


def build_scores(eigs, pis, rate_vals, is_inv, logw, logrho, tau,
                 PF, PT, TB, CNT, SEG, G, gsel, conserved):
    """Score tensor sc[i,c,r] = log w_c + log rho_r + LL_{c,r}(g) for clusters gsel.
    Invariant bins (is_inv) are class-independent: 0 if the cluster is fully
    conserved else -inf.  eigs[c] = FP.eig_rev(Q_c, pi_c)."""
    K = len(pis); R = len(rate_vals); ncl = len(gsel)
    sc = np.full((ncl, K, R), -np.inf)
    inv_ll = np.where(conserved[gsel], 0.0, -np.inf)         # (ncl,)
    for c in range(K):
        for r in range(R):
            if is_inv[r]:
                sc[:, c, r] = logw[c] + logrho[r] + inv_ll
            else:
                grid = logP_grid(eigs[c], tau, rate_vals[r])
                ll = cluster_gamma_ll(grid, PF, PT, TB, CNT, SEG, G)[gsel]
                sc[:, c, r] = logw[c] + logrho[r] + ll
    return sc


def responsibilities(sc):
    """Softmax over the flattened (c,r) axis; -inf-safe."""
    flat = sc.reshape(sc.shape[0], -1)
    m = flat.max(1, keepdims=True)
    e = np.exp(flat - m); Z = e.sum(1, keepdims=True)
    return (e / np.maximum(Z, 1e-300)).reshape(sc.shape)


def marginal_percount_ll(sc, tot):
    """Observed-data marginal per-count LL: (1/tot) sum_g logsumexp_{c,r} sc[g,c,r]."""
    flat = sc.reshape(sc.shape[0], -1)
    m = flat.max(1)
    return float((m + np.log(np.exp(flat - m[:, None]).sum(1))).sum() / tot)


# ---------- HR / bridge M-step at rate-scaled branch lengths ----------
def estep_scaled(Q, pi, tau, rate, npair, eig):
    """HR usage N and dwell T summed over tau bins at branch length rate*tau[t]."""
    N = np.zeros((NS, NS)); T = np.zeros(NS)
    for t in range(len(tau)):
        edge = npair[:, :, t]
        if edge.sum() == 0:
            continue
        Tt, Nt, _ = FP.bridge(Q, pi, rate * float(tau[t]), edge, want_N=True, eig=eig)
        N += Nt; T += Tt
    return N, T


def mstep_shared(R_full, rate_vals, gamma_idx, S, pis, tau, inner, free_S,
                 trSEG, trCNT, flat, T):
    """Joint (pi_c, S) M-step at fixed responsibilities R_full (G,K,R).  For `inner`
    ECM rounds: per class c, aggregate HR usage/dwell over the Gamma rate bins on the
    responsibility-weighted counts placed at rate_r*tau; update pi_c at the current S;
    accumulate pooled Metropolis S suff-stats; then (free_S) refit the shared S.
    The invariant bin contributes zero dwell (rate 0) so it is excluded here."""
    K = len(pis); pis = [p.copy() for p in pis]
    for _ in range(inner):
        Cnum_tot = np.zeros((NA, NA)); Hden_tot = np.zeros((NA, NA))
        for c in range(K):
            Q_c = FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt")
            eig_c = FP.eig_rev(Q_c, np.clip(pis[c], 1e-12, None))
            N_c = np.zeros((NS, NS)); T_c = np.zeros(NS)
            for r in gamma_idx:
                wnnz = trCNT * R_full[trSEG, c, r]
                if wnnz.sum() == 0:
                    continue
                Ncr = np.bincount(flat, weights=wnnz,
                                  minlength=NS * NS * T).reshape(NS, NS, T)
                Nn, Tt = estep_scaled(Q_c, pis[c], tau, rate_vals[r], Ncr, eig_c)
                N_c += Nn; T_c += Tt
            pis[c] = FP.mstep_pi_metropolis(S, pis[c], N_c, T_c, "sqrt")
            Cnum_c, Hden_c = FS.met_S_suffstats(N_c, T_c, pis[c], "sqrt")
            Cnum_tot += Cnum_c; Hden_tot += Hden_c
        if free_S:
            S = np.where(Hden_tot > 0, Cnum_tot / np.maximum(Hden_tot, 1e-300), 0.0)
            S = 0.5 * (S + S.T); np.fill_diagonal(S, 0.0)
    Qs = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(K)]
    eigs = [FP.eig_rev(Qs[c], np.clip(pis[c], 1e-12, None)) for c in range(K)]
    return S, pis, Qs, eigs


# ---------- alpha (Gamma shape) 1-D CM-step ----------
def refit_alpha(alpha, R_gamma, eigs, R_full, gamma_idx, tau,
                trPF, trPT, trTB, trCNT, trSEG, G, tr_g, cands_factor):
    """Local 1-D search over the Gamma shape alpha maximising the rate-marginal
    expected complete-data LL  Q(alpha) = sum_{g,c,r>0} R_{gcr} LL_{c,rate_r(alpha)}(g),
    responsibilities and (pi_c,S) held fixed.  The current alpha is always a candidate,
    so Q cannot decrease (ECM monotone).  Returns (alpha, rate_vals_gamma)."""
    if R_gamma < 2:
        return alpha, discrete_gamma_rates(max(R_gamma, 1), alpha)
    # per (c, gamma-bin) responsibility-weighted nnz weights (fixed across candidates)
    K = len(eigs)
    wnnz = {}
    for c in range(K):
        for gi, r in enumerate(gamma_idx):
            wnnz[(c, gi)] = (trCNT * R_full[trSEG, c, r])

    def Q_of(a):
        rates = discrete_gamma_rates(R_gamma, a)             # (R_gamma,) mean 1
        tot = 0.0
        for c in range(K):
            for gi in range(R_gamma):
                grid = logP_grid(eigs[c], tau, rates[gi])
                g_nnz = grid[trTB, trPF, trPT]
                tot += float(np.dot(wnnz[(c, gi)], g_nnz))
        return tot
    cands = np.unique(np.concatenate([[alpha], alpha * np.asarray(cands_factor)]))
    cands = np.clip(cands, 0.02, 50.0)
    vals = np.array([Q_of(a) for a in cands])
    best = cands[int(np.argmax(vals))]
    return float(best), discrete_gamma_rates(R_gamma, best)


def class_rate_mi(R_full, tr_g, w, rho):
    """Mutual information (nats) between the pairing class and the rate class over the
    training clusters, from the summed joint responsibility.  ~0 => independent."""
    Rt = R_full[tr_g]                                        # (n,K,R)
    J = Rt.sum(0); J = J / max(J.sum(), 1e-300)              # joint P(c,r)
    pc = J.sum(1); pr = J.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mij = J * np.log(J / np.maximum(np.outer(pc, pr), 1e-300))
    mival = max(0.0, float(np.nansum(np.where(J > 0, mij, 0.0))))
    Hc = -float(np.sum(pc * np.log(np.maximum(pc, 1e-300))))
    Hr = -float(np.sum(pr * np.log(np.maximum(pr, 1e-300))))
    # with a single class (or single rate) MI is identically 0 and NMI is undefined
    nmi = 0.0 if min(Hc, Hr) < 1e-9 else mival / min(Hc, Hr)
    return mival, nmi, Hc, Hr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/per_contact_trrosetta/counts.npz")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--gamma-cats", type=int, default=4,
                    help="number of discrete-Gamma rate categories (R-1)")
    ap.add_argument("--pinv", type=float, default=0.2, help="initial invariant weight")
    ap.add_argument("--single-rate", action="store_true",
                    help="collapse the rate grid to one bin at rate 1 (no I, no alpha) "
                         "-- reproduces fit_coupling_mixture_freeS.py")
    ap.add_argument("--alpha-init", type=float, default=1.0)
    ap.add_argument("--em-iters", type=int, default=40)
    ap.add_argument("--alpha-every", type=int, default=1,
                    help="refit the Gamma shape alpha every N EM iters (skipping keeps it "
                         "fixed => still ECM-monotone; alpha stabilises early so N>1 is a "
                         "cheap speedup for large K)")
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-counts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-S", dest="free_S", action="store_false",
                    help="hold shared S=LG08")
    ap.set_defaults(free_S=True)
    ap.add_argument("--tol", type=float, default=1e-6, help="train-LL convergence tol")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); t0 = time.time()
    mode = "single-rate" if a.single_rate else f"gammaI(g{a.gamma_cats})"
    ALPHA_CANDS = (0.6, 0.78, 1.0, 1.28, 1.66)               # local multiplicative search

    # ---- corpus + family split (identical to fit_coupling_mixture_freeS) ----
    (PF, PT, SEG, tau, T, G, tot_cnt, keep,
     trPF, trPT, trTB, trCNT, trSEG, tr_g, tr_tot,
     vaPF, vaPT, vaTB, vaCNT, vaSEG, va_g, va_tot) = FS.load_split_core(
        a.corpus, rng, a.val_frac, a.min_counts)
    # per-cluster fully-conserved flag (all transitions on the diagonal) -- eligible
    # for the invariant bin.  Computed on the FULL corpus (a cluster is wholly in one split).
    conserved = np.bincount(SEG, weights=(PF != PT).astype(float), minlength=G) == 0
    flat = (trPF * NS + trPT) * T + trTB                      # nnz -> (400,400,T) flat index
    K = a.K
    print(f"# [{mode}] {G} clusters ({keep.sum()} kept, {int(conserved[keep].sum())} "
          f"fully-conserved); train {len(tr_g)} / val {len(va_g)}; "
          f"{int(tr_tot):,}/{int(va_tot):,} transitions; K={K}", flush=True)

    # ---- rate grid ----
    if a.single_rate:
        R_gamma = 1
        rate_vals = np.array([1.0]); is_inv = np.array([False])
        rho = np.array([1.0]); alpha = float("nan"); p_inv = 0.0
    else:
        R_gamma = a.gamma_cats
        alpha = a.alpha_init
        g = discrete_gamma_rates(R_gamma, alpha)             # mean-1 Gamma multipliers
        rate_vals = np.concatenate([[0.0], g])               # bin 0 = invariant
        is_inv = np.array([True] + [False] * R_gamma)
        p_inv = a.pinv
        rho = np.concatenate([[p_inv], np.full(R_gamma, (1 - p_inv) / R_gamma)])
    R = len(rate_vals)
    gamma_idx = [r for r in range(R) if not is_inv[r]]

    # ---- init pi_c / S / w (identical scheme to fit_coupling_mixture_freeS) ----
    occ = np.bincount(trPF, weights=trCNT, minlength=NS) + np.bincount(trPT, weights=trCNT, minlength=NS)
    glob = occ / occ.sum()
    pis = []
    for c in range(K):
        nz = rng.normal(0, 0.6, (NA, NA)); nz = 0.5 * (nz + nz.T)
        p = glob.reshape(NA, NA) * np.exp(nz); p = (p / p.sum()).reshape(NS)
        pis.append(p)
    S = S_LG.copy()
    w = np.full(K, 1.0 / K)
    Qs = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(K)]
    eigs = [FP.eig_rev(Qs[c], np.clip(pis[c], 1e-12, None)) for c in range(K)]

    prev = None; trll = float("nan"); monotone = True
    for it in range(a.em_iters):
        logw = np.log(w + 1e-300); logrho = np.log(rho + 1e-300)
        # E-step: train responsibilities + train marginal LL
        sc = build_scores(eigs, pis, rate_vals, is_inv, logw, logrho, tau,
                          trPF, trPT, trTB, trCNT, trSEG, G, tr_g, conserved)
        trll = marginal_percount_ll(sc, tr_tot)
        if prev is not None and trll < prev - 1e-9:
            monotone = False
        Rr = responsibilities(sc)                            # (n_tr,K,R)
        R_full = np.zeros((G, K, R)); R_full[tr_g] = Rr
        # M-step weights (exact maximisers)
        w = Rr.sum((0, 2)) + 1e-9; w /= w.sum()
        rho = Rr.sum((0, 1)) + 1e-12; rho /= rho.sum()
        if not a.single_rate:
            p_inv = float(rho[0])
        # M-step pi_c, S (rate-scaled HR)
        S, pis, Qs, eigs = mstep_shared(R_full, rate_vals, gamma_idx, S, pis, tau,
                                        a.inner, a.free_S, trSEG, trCNT, flat, T)
        # M-step alpha (Gamma shape) -- rate values only, ECM CM-step.  The Gamma grid
        # stays mean-1 (Yang) throughout training so the current alpha is a genuine
        # candidate => Q is non-decreasing.  The joint (rate-scale, S-scale) gauge freedom
        # (Q propto S, branch = rate*tau) is left to float here -- it does NOT affect the
        # likelihood or monotonicity -- and is fixed once at the end for interpretable
        # reporting.  (An in-loop gauge rescale desyncs this alpha candidate and breaks
        # monotonicity, so it is deliberately avoided.)
        if (not a.single_rate and R_gamma >= 2 and it % a.alpha_every == 0):
            alpha, g = refit_alpha(alpha, R_gamma, eigs, R_full, gamma_idx, tau,
                                   trPF, trPT, trTB, trCNT, trSEG, G, tr_g, ALPHA_CANDS)
            rate_vals = np.concatenate([[0.0], g])
        srel, scorr = FS.s_stats(S)
        extra = "" if a.single_rate else f" alpha={alpha:.3f} pinv={p_inv:.3f}"
        print(f"  it {it:2d}: train_mix={trll:.4f}  w={np.round(np.sort(w)[::-1], 3)}"
              f"  MI(pi)={sorted([round(FS.mi(p), 3) for p in pis], reverse=True)}"
              f"  Srel={srel:.3f}{extra}  [{time.time()-t0:.0f}s]", flush=True)
        if prev is not None and abs(trll - prev) < a.tol:
            print("  converged", flush=True); break
        prev = trll

    # ---- final held-out scoring ----
    logw = np.log(w + 1e-300); logrho = np.log(rho + 1e-300)
    sc_va = build_scores(eigs, pis, rate_vals, is_inv, logw, logrho, tau,
                        vaPF, vaPT, vaTB, vaCNT, vaSEG, G, va_g, conserved)
    vall = marginal_percount_ll(sc_va, va_tot)
    # class vs rate independence on train clusters
    sc_tr = build_scores(eigs, pis, rate_vals, is_inv, logw, logrho, tau,
                        trPF, trPT, trTB, trCNT, trSEG, G, tr_g, conserved)
    R_full = np.zeros((G, K, R)); R_full[tr_g] = responsibilities(sc_tr)
    cr_mi, cr_nmi, Hc, Hr = class_rate_mi(R_full, tr_g, w, rho)
    srel, scorr = FS.s_stats(S)
    order = np.argsort(w)[::-1]
    # end-of-loop gauge fix (reporting only): the Gamma grid is mean-1 Yang but rho is
    # free, so normalise the effective variable-rate distribution to conditional mean 1.
    # eff_mult = spread of rates actually used; rate_cv = coefficient of variation (the
    # "range of rates").  Pure reporting transform -- does not touch the fitted model/LL.
    if a.single_rate:
        eff_mult = [1.0]; eff_w = [1.0]; cond_mean = 1.0; rate_cv = 0.0
    else:
        gw = rho[gamma_idx]; gv = np.asarray([rate_vals[r] for r in gamma_idx])
        gwn = gw / max(gw.sum(), 1e-300)
        cond_mean = float((gwn * gv).sum())
        eff = gv / max(cond_mean, 1e-300)
        eff_mult = [float(x) for x in eff]; eff_w = [float(x) for x in gwn]
        rate_cv = float(np.sqrt((gwn * (eff - 1.0) ** 2).sum()))
    res = dict(
        mode=mode, K=K, single_rate=bool(a.single_rate), gamma_cats=int(R_gamma),
        val_per_count_ll=float(vall), train_per_count_ll=float(trll),
        monotone=bool(monotone), n_em_iters=int(it + 1),
        weights=[float(w[c]) for c in order],
        mi_pi=[float(FS.mi(pis[c])) for c in order],
        mi_weighted=float(sum(w[c] * FS.mi(pis[c]) for c in range(K))),
        rate_weights=[float(x) for x in rho], rate_vals=[float(x) for x in rate_vals],
        eff_rate_mult=eff_mult, eff_rate_weight=eff_w, rate_cv=rate_cv,
        alpha=(None if a.single_rate else float(alpha)),
        p_inv=(None if a.single_rate else float(p_inv)),
        mean_rate=float(sum(rho[r] * rate_vals[r] for r in range(R))),
        class_rate_mi_nats=cr_mi, class_rate_nmi=cr_nmi, H_class=Hc, H_rate=Hr,
        shared_s_rel_fro=srel, shared_s_offdiag_corr=scorr,
        n_train_clusters=int(len(tr_g)), n_val_clusters=int(len(va_g)),
        n_conserved_kept=int(conserved[keep].sum()), seed=a.seed)
    print(f"# [{mode}] K={K}: VAL/count={vall:.4f} train={trll:.4f} monotone={monotone} "
          f"iters={it+1}", flush=True)
    if not a.single_rate:
        print(f"#   alpha={alpha:.3f} p_inv={p_inv:.3f} rate_CV={rate_cv:.3f} "
              f"rate_w={[round(x,3) for x in rho]}", flush=True)
        print(f"#   eff_rate_mult={[round(x,3) for x in eff_mult]} "
              f"(cond_mean gauge={cond_mean:.3g})", flush=True)
        print(f"#   class<->rate MI={cr_mi:.4f} nats  NMI={cr_nmi:.4f} "
              f"(H_class={Hc:.3f} H_rate={Hr:.3f})", flush=True)
    print(f"#   w-mean MI(pi_c)={res['mi_weighted']:.4f}  sharedSrel={srel:.4f} "
          f"sharedScorr={scorr:.4f}", flush=True)
    print(f"#   MI(pi_c) per comp = {[round(x,3) for x in res['mi_pi']]}", flush=True)
    print(f"#   weights  per comp = {[round(x,3) for x in res['weights']]}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        # also save the fitted components in components_K{K}.npz format (pis, S,
        # weights, mi_pi, tau) so plot_mixture_components / report_mixture_mi /
        # interpret_mixture_coupling can regenerate the figure, MI_dyn and tab:axes
        # from THIS (jointly-fit Gamma+I) mixture.
        np.savez(a.out.replace(".json", ".npz"),
                 pis=np.array(pis), S=np.asarray(S), weights=np.asarray(w),
                 mi_pi=np.array([FS.mi(pis[c]) for c in range(K)]),
                 tau=np.asarray(tau), K=int(K),
                 val_per_count_ll=float(vall), train_per_count_ll=float(trll))
        print(f"# wrote {a.out} + .npz", flush=True)
    return res


if __name__ == "__main__":
    main()
