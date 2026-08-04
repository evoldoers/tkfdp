#!/usr/bin/env python3
r"""Asymmetric (transposed swap-pair) coupling mixture WITH Gamma+I rate heterogeneity.

Combines the two validated fitters:
  * swap-pair machinery  -- fit_coupling_mixture_asym.py  (each free component c enters
    twice, {pi_c, pi_c^swap}, tied, equal weight W_c/2; asymmetric HR pi M-step, no
    symmetrisation; the '-' component scores pi_c on the site-swapped counts).
  * Gamma+I rate grid    -- fit_coupling_mixture_rateI.py (per-cluster discrete-Gamma
    rate class + invariant bin, independent of the pairing class; rate-scaled HR M-step;
    1-D alpha CM-step).

Produces a row matched to the symmetric Gamma+I mixture of fit_coupling_mixture_rateI.py:
P swap-pairs (2P components) is matched to a 2P-component symmetric mixture -- so twoP=4
sits next to the symmetric K=4 mixture, twoP=8 next to K=8.  Same corpus / family split /
gamma-cats / seed, so the asym-vs-symmetric comparison is apples-to-apples.

Verification reductions:
  * --single-rate      collapses the rate grid to one bin at rate 1 (no I, no alpha) and
    must reproduce fit_coupling_mixture_asym.py at 2P (the swap-pair machinery).
  * --force-symmetric  re-symmetrises pi_c every M-step, collapsing each swap-pair to ONE
    symmetric component, so twoP=2P must reproduce fit_coupling_mixture_rateI.py at K=P
    (the Gamma+I machinery).
"""
from __future__ import annotations
import os
# Hard-set (not setdefault): the parent env may carry JAX_PLATFORMS="" (present but
# empty), which setdefault will NOT override -- jax would then try the CUDA backend and
# crash with CUDA_ERROR_NO_DEVICE. This fitter is pure numpy/CPU, so force CPU.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, json, time, sys
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import fit_pair_models as FP
from fit_pair_models import NA, NS
import fit_coupling_mixture_freeS as FS
import fit_coupling_mixture_rateI as RI      # logP_grid, cluster_gamma_ll, estep_scaled
import fit_coupling_mixture_asym as AS        # SWAP_IDX, swap_pi, mstep_pi_metropolis_asym, sym_kl, asym_tv
from tkfdp.coupling.dynfield.phylo_elbo.rate_hetero import discrete_gamma_rates

SWAP_IDX = AS.SWAP_IDX
S_LG = RI.S_LG


# ---------- 2P-component swap-pair x Gamma+I score tensor ----------
def build_scores(eigs, pis, rate_vals, is_inv, logwpair, logrho, tau,
                 PF, PT, PFs, PTs, TB, CNT, SEG, G, gsel, conserved):
    """sc[i, comp, r] for clusters gsel.  comp = 2c ('+', pi_c on counts) or 2c+1 ('-',
    pi_c on site-swapped counts); each carries the tied half-weight log(W_c/2).  The
    invariant bin (rate 0) is class/orientation-independent: 0 if conserved else -inf."""
    P = len(pis); R = len(rate_vals); ncl = len(gsel)
    sc = np.full((ncl, 2 * P, R), -np.inf)
    inv_ll = np.where(conserved[gsel], 0.0, -np.inf)
    for c in range(P):
        lh = logwpair[c] - np.log(2.0)
        for r in range(R):
            if is_inv[r]:
                sc[:, 2 * c, r] = lh + logrho[r] + inv_ll
                sc[:, 2 * c + 1, r] = lh + logrho[r] + inv_ll
            else:
                grid = RI.logP_grid(eigs[c], tau, rate_vals[r])
                llp = RI.cluster_gamma_ll(grid, PF, PT, TB, CNT, SEG, G)[gsel]
                llm = RI.cluster_gamma_ll(grid, PFs, PTs, TB, CNT, SEG, G)[gsel]
                sc[:, 2 * c, r] = lh + logrho[r] + llp
                sc[:, 2 * c + 1, r] = lh + logrho[r] + llm
    return sc


def responsibilities(sc):
    flat = sc.reshape(sc.shape[0], -1)
    m = flat.max(1, keepdims=True)
    e = np.exp(flat - m); Z = e.sum(1, keepdims=True)
    return (e / np.maximum(Z, 1e-300)).reshape(sc.shape)


def marginal_percount_ll(sc, tot):
    flat = sc.reshape(sc.shape[0], -1)
    m = flat.max(1)
    return float((m + np.log(np.exp(flat - m[:, None]).sum(1))).sum() / tot)


# ---------- rate-scaled + swap-folded HR M-step for (pi_c, S) ----------
def mstep_shared(R_full, rate_vals, gamma_idx, S, pis, tau, inner, free_S,
                 trSEG, trCNT, flat_p, flat_m, T, force_symmetric):
    """For each swap-pair c aggregate HR usage/dwell over the Gamma rate bins AND over
    both orientations -- '+' responsibility on the counts, '-' responsibility on the
    site-swapped counts, both folded into pi_c's frame -- then the asymmetric pi-step."""
    P = len(pis); pis = [p.copy() for p in pis]
    for _ in range(inner):
        Cnum_tot = np.zeros((NA, NA)); Hden_tot = np.zeros((NA, NA))
        for c in range(P):
            Q_c = FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt")
            eig_c = FP.eig_rev(Q_c, np.clip(pis[c], 1e-12, None))
            N_c = np.zeros((NS, NS)); T_c = np.zeros(NS)
            for r in gamma_idx:
                wp = trCNT * R_full[trSEG, 2 * c, r]
                wm = trCNT * R_full[trSEG, 2 * c + 1, r]
                if wp.sum() == 0 and wm.sum() == 0:
                    continue
                Ncr = (np.bincount(flat_p, weights=wp, minlength=NS * NS * T)
                       + np.bincount(flat_m, weights=wm, minlength=NS * NS * T)
                       ).reshape(NS, NS, T)
                Nn, Tt = RI.estep_scaled(Q_c, pis[c], tau, rate_vals[r], Ncr, eig_c)
                N_c += Nn; T_c += Tt
            pis[c] = AS.mstep_pi_metropolis_asym(S, pis[c], N_c, T_c)
            if force_symmetric:
                pis[c] = 0.5 * (pis[c] + AS.swap_pi(pis[c]))
            Cnum_c, Hden_c = FS.met_S_suffstats(N_c, T_c, pis[c], "sqrt")
            Cnum_tot += Cnum_c; Hden_tot += Hden_c
        if free_S:
            S = np.where(Hden_tot > 0, Cnum_tot / np.maximum(Hden_tot, 1e-300), 0.0)
            S = 0.5 * (S + S.T); np.fill_diagonal(S, 0.0)
    Qs = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(P)]
    eigs = [FP.eig_rev(Qs[c], np.clip(pis[c], 1e-12, None)) for c in range(P)]
    return S, pis, Qs, eigs


def refit_alpha(alpha, R_gamma, eigs, R_full, gamma_idx, tau,
                trPF, trPT, trPFs, trPTs, trTB, trCNT, trSEG, cands):
    """1-D local search over the Gamma shape maximising the rate-marginal expected
    complete-data LL, summed over both orientations of every swap-pair.  Current alpha is
    always a candidate => ECM non-decreasing."""
    if R_gamma < 2:
        return alpha, discrete_gamma_rates(max(R_gamma, 1), alpha)
    P = len(eigs)
    wp = {}; wm = {}
    for c in range(P):
        for gi, r in enumerate(gamma_idx):
            wp[(c, gi)] = trCNT * R_full[trSEG, 2 * c, r]
            wm[(c, gi)] = trCNT * R_full[trSEG, 2 * c + 1, r]

    def Q_of(a):
        rates = discrete_gamma_rates(R_gamma, a); tot = 0.0
        for c in range(P):
            for gi in range(R_gamma):
                grid = RI.logP_grid(eigs[c], tau, rates[gi])
                tot += float(np.dot(wp[(c, gi)], grid[trTB, trPF, trPT]))
                tot += float(np.dot(wm[(c, gi)], grid[trTB, trPFs, trPTs]))
        return tot
    c_ = np.clip(np.unique(np.concatenate([[alpha], alpha * np.asarray(cands)])), 0.02, 50.0)
    vals = np.array([Q_of(a) for a in c_])
    best = float(c_[int(np.argmax(vals))])
    return best, discrete_gamma_rates(R_gamma, best)


def fit(split, twoP, gamma_cats, pinv, single_rate, alpha_init, em_iters,
        alpha_every, inner, free_S, seed, force_symmetric, verbose=True):
    (trPF, trPT, trPFs, trPTs, trTB, trCNT, trSEG, tr_g, tr_tot,
     vaPF, vaPT, vaPFs, vaPTs, vaTB, vaCNT, vaSEG, va_g, va_tot, G, tau, glob, conserved) = split
    P = twoP // 2
    rng = np.random.default_rng(seed); t0 = time.time()
    Tn = len(tau)
    flat_p = (trPF * NS + trPT) * Tn + trTB
    flat_m = (trPFs * NS + trPTs) * Tn + trTB
    ALPHA_CANDS = (0.6, 0.78, 1.0, 1.28, 1.66)

    # rate grid
    if single_rate:
        R_gamma = 1; rate_vals = np.array([1.0]); is_inv = np.array([False])
        rho = np.array([1.0]); alpha = float("nan"); p_inv = 0.0
    else:
        R_gamma = gamma_cats; alpha = alpha_init
        g = discrete_gamma_rates(R_gamma, alpha)
        rate_vals = np.concatenate([[0.0], g]); is_inv = np.array([True] + [False] * R_gamma)
        p_inv = pinv; rho = np.concatenate([[p_inv], np.full(R_gamma, (1 - p_inv) / R_gamma)])
    R = len(rate_vals); gamma_idx = [r for r in range(R) if not is_inv[r]]

    # init: asymmetric pi_c (symmetric if force_symmetric, for the degenerate reduction)
    pis = []
    for c in range(P):
        nz = rng.normal(0, 0.6, (NA, NA))
        if force_symmetric:
            nz = 0.5 * (nz + nz.T)
        p = glob.reshape(NA, NA) * np.exp(nz); p = (p / p.sum()).reshape(NS)
        pis.append(p)
    S = S_LG.copy(); Wp = np.full(P, 1.0 / P)
    Qs = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(P)]
    eigs = [FP.eig_rev(Qs[c], np.clip(pis[c], 1e-12, None)) for c in range(P)]

    prev = None; trll = float("nan"); monotone = True; it = 0
    for it in range(em_iters):
        logwpair = np.log(Wp + 1e-300); logrho = np.log(rho + 1e-300)
        sc = build_scores(eigs, pis, rate_vals, is_inv, logwpair, logrho, tau,
                          trPF, trPT, trPFs, trPTs, trTB, trCNT, trSEG, G, tr_g, conserved)
        trll = marginal_percount_ll(sc, tr_tot)
        if prev is not None and trll < prev - 1e-9:
            monotone = False
        Rr = responsibilities(sc)                              # (n_tr, 2P, R)
        R_full = np.zeros((G, 2 * P, R)); R_full[tr_g] = Rr
        pair_resp = Rr.reshape(Rr.shape[0], P, 2, R).sum((2, 3))  # (n_tr, P)
        Wp = pair_resp.mean(0) + 1e-9; Wp /= Wp.sum()
        rho = Rr.sum((0, 1)) + 1e-12; rho /= rho.sum()
        if not single_rate:
            p_inv = float(rho[0])
        S, pis, Qs, eigs = mstep_shared(R_full, rate_vals, gamma_idx, S, pis, tau, inner,
                                        free_S, trSEG, trCNT, flat_p, flat_m, Tn, force_symmetric)
        if (not single_rate and R_gamma >= 2 and it % alpha_every == 0):
            alpha, g = refit_alpha(alpha, R_gamma, eigs, R_full, gamma_idx, tau,
                                   trPF, trPT, trPFs, trPTs, trTB, trCNT, trSEG, ALPHA_CANDS)
            rate_vals = np.concatenate([[0.0], g])
        if verbose:
            srel, _ = FS.s_stats(S)
            extra = "" if single_rate else f" alpha={alpha:.3f} pinv={p_inv:.3f}"
            print(f"  it {it:2d}: train_mix={trll:.4f}  W={np.round(np.sort(Wp)[::-1],3)}"
                  f"  MI={sorted([round(FS.mi(p),3) for p in pis],reverse=True)}"
                  f"  asymTV={sorted([round(AS.asym_tv(p),3) for p in pis],reverse=True)}"
                  f"  Srel={srel:.3f}{extra}  [{time.time()-t0:.0f}s]", flush=True)
        if prev is not None and abs(trll - prev) < 1e-6:
            if verbose:
                print("  converged", flush=True)
            break
        prev = trll

    # held-out
    logwpair = np.log(Wp + 1e-300); logrho = np.log(rho + 1e-300)
    sc_va = build_scores(eigs, pis, rate_vals, is_inv, logwpair, logrho, tau,
                         vaPF, vaPT, vaPFs, vaPTs, vaTB, vaCNT, vaSEG, G, va_g, conserved)
    vall = marginal_percount_ll(sc_va, va_tot)
    # cost of forcing symmetry (held out): pi_c -> (pi_c + pi_c^swap)/2
    pis_sym = [0.5 * (p + AS.swap_pi(p)) for p in pis]
    eigs_sym = [FP.eig_rev(FP._met_Q(S, pis_sym[c].reshape(NA, NA), "sqrt"),
                           np.clip(pis_sym[c], 1e-12, None)) for c in range(P)]
    sc_va_sym = build_scores(eigs_sym, pis_sym, rate_vals, is_inv, logwpair, logrho, tau,
                             vaPF, vaPT, vaPFs, vaPTs, vaTB, vaCNT, vaSEG, G, va_g, conserved)
    vall_symforced = marginal_percount_ll(sc_va_sym, va_tot)

    order = np.argsort(Wp)[::-1]
    res = dict(
        mode=("single-rate" if single_rate else f"gammaI(g{R_gamma})"),
        side="asymmetric_swap_pair", twoP=int(twoP), n_pairs=int(P),
        n_components=int(2 * P), gamma_cats=int(R_gamma),
        val_per_count_ll=float(vall), train_per_count_ll=float(trll),
        val_per_count_ll_symforced=float(vall_symforced),
        force_symmetry_cost_val=float(vall - vall_symforced),
        n_pi_params=int(P * (NS - 1)),                      # asym: 399 per free component
        monotone=bool(monotone), n_em_iters=int(it + 1),
        pair_weights=[float(Wp[c]) for c in order],
        mi_pi=[float(FS.mi(pis[c])) for c in order],
        mi_weighted=float(sum(Wp[c] * FS.mi(pis[c]) for c in range(P))),
        weighted_asym_tv=float(sum(Wp[c] * AS.asym_tv(pis[c]) for c in range(P))),
        rate_weights=[float(x) for x in rho], rate_vals=[float(x) for x in rate_vals],
        alpha=(None if single_rate else float(alpha)),
        p_inv=(None if single_rate else float(p_inv)),
        shared_s_rel_fro=float(FS.s_stats(S)[0]),
        shared_s_offdiag_corr=float(FS.s_stats(S)[1]),
        n_train_clusters=int(len(tr_g)), n_val_clusters=int(len(va_g)),
        force_symmetric=bool(force_symmetric), seed=int(seed))
    return res, dict(S=S, pis=np.array(pis), Wp=Wp)


def load_split(corpus, val_frac, min_counts, seed):
    """AS.load_split (adds swapped indices) + the Gamma+I `conserved` flag per cluster."""
    z, sp = AS.load_split(corpus, val_frac, min_counts, seed)
    (trPF, trPT, trPFs, trPTs, trTB, trCNT, trSEG, tr_g, tr_tot,
     vaPF, vaPT, vaPFs, vaPTs, vaTB, vaCNT, vaSEG, va_g, va_tot, G, tau, glob) = sp
    sub = (np.bincount(trSEG, weights=(trPF != trPT).astype(float), minlength=G)
           + np.bincount(vaSEG, weights=(vaPF != vaPT).astype(float), minlength=G))
    conserved = sub == 0
    return z, (trPF, trPT, trPFs, trPTs, trTB, trCNT, trSEG, tr_g, tr_tot,
               vaPF, vaPT, vaPFs, vaPTs, vaTB, vaCNT, vaSEG, va_g, va_tot, G, tau, glob, conserved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/per_contact_trrosetta/counts.npz")
    ap.add_argument("--twoP", type=int, default=8, help="total components (2 x swap-pairs); "
                    "matched to symmetric K=twoP")
    ap.add_argument("--gamma-cats", type=int, default=4)
    ap.add_argument("--pinv", type=float, default=0.2)
    ap.add_argument("--single-rate", action="store_true")
    ap.add_argument("--force-symmetric", action="store_true")
    ap.add_argument("--alpha-init", type=float, default=1.0)
    ap.add_argument("--em-iters", type=int, default=100)
    ap.add_argument("--alpha-every", type=int, default=1)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-counts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-S", dest="free_S", action="store_false"); ap.set_defaults(free_S=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z, split = load_split(a.corpus, a.val_frac, a.min_counts, a.seed)
    G, tr_g, va_g = split[18], split[7], split[16]
    mode = "single-rate" if a.single_rate else f"gammaI(g{a.gamma_cats})"
    print(f"# [asym {mode}] corpus {a.corpus}: {G} clusters; train {len(tr_g)}/val {len(va_g)}; "
          f"twoP={a.twoP} ({a.twoP//2} swap-pairs){' [FORCE-SYM]' if a.force_symmetric else ''}",
          flush=True)
    res, params = fit(split, a.twoP, a.gamma_cats, a.pinv, a.single_rate, a.alpha_init,
                      a.em_iters, a.alpha_every, a.inner, a.free_S, a.seed, a.force_symmetric)
    print(f"# [asym {mode}] twoP={a.twoP}: VAL/count={res['val_per_count_ll']:.4f} "
          f"train={res['train_per_count_ll']:.4f} monotone={res['monotone']} "
          f"iters={res['n_em_iters']} MIw={res['mi_weighted']:.4f} "
          f"asymTVw={res['weighted_asym_tv']:.3f} force_sym_cost={res['force_symmetry_cost_val']:.4f}",
          flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        np.savez(a.out.replace(".json", ".npz"), **params)
        print(f"# wrote {a.out}", flush=True)
    return res


if __name__ == "__main__":
    main()
