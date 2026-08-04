#!/usr/bin/env python3
r"""Paper-2 single-component baselines with Gamma+I, on the per-contact corpus.

Fits the tab:pairfit baseline rows -- Exchangeable (=synchronized, all Klein-4
orbit fluxes), Coupled (single-transition orbit fluxes), and the Metropolis
family (shared free exchangeability S with kernel f in {sqrt, barker, hastings,
gtr}) -- each with Gamma+I rate heterogeneity, on the SAME per-contact corpus /
family split / rate grid as the Metropolis-mixture rows (fit_coupling_mixture_rateI).

Everything is a K=1 special case of the mixture EM: one pairing "class", one
Gamma+I rate latent per size-2 cluster.  The ONLY difference from
fit_coupling_mixture_rateI is the per-component M-step, which here comes from
fit_pair_models (FP): the flux M-step (synchronized/coupled) or the Metropolis
S-/pi-step (the four kernels), applied to HR usage/dwell aggregated over the
Gamma rate bins at rate-scaled branch lengths -- identical aggregation to the
mixture's mstep_shared.

Persists the fitted rate matrix Q (400x400), symmetric flux F, stationary pi, and
the Gamma+I params (alpha, p_inv, rate grid) so the release reproduces each row.

Reuses the mixture's split + Gamma+I scaffolding via `import
fit_coupling_mixture_rateI as RI` (logP_grid, build_scores, responsibilities,
marginal_percount_ll, estep_scaled, refit_alpha)."""
from __future__ import annotations
import argparse, json, time, os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import fit_pair_models as FP
from fit_pair_models import NA, NS
import fit_coupling_mixture_freeS as FS
import fit_coupling_mixture_rateI as RI
from tkfdp.lg08 import S_LG08
from tkfdp.coupling.dynfield.phylo_elbo.rate_hetero import discrete_gamma_rates

S_LG = np.asarray(S_LG08, float)[:NA, :NA].copy()
S_LG = 0.5 * (S_LG + S_LG.T); np.fill_diagonal(S_LG, 0.0)

ALPHA_CANDS = (0.6, 0.78, 1.0, 1.28, 1.66)


def current_Q(model, is_met, kernel, S, F, pi):
    if is_met:
        return FP._met_Q(S, pi.reshape(NA, NA), kernel)
    return FP.Q_from_flux(F, pi)


def model_mstep(model, is_met, kernel, N, T, pi, S, orbit_id, n_orbits, mask):
    """One M-step for `model` given HR stats (N,T) aggregated over the Gamma bins.
    Mirrors fit_pair_models.fit_model's per-iteration S-/flux- and pi-steps
    exactly.  Returns (Q, F, pi, S)."""
    if is_met:
        F = FP.mstep_metropolis(N, T, pi, kernel)          # S-step (pi fixed) -> F=S*sym
        S = FP._met_S_from_F(F, pi, kernel)
        if kernel != "gtr":                                # nonlinear exposure: gradient pi-step
            pi = FP.mstep_pi_metropolis(S, pi, N, T, kernel)
            Q = FP._met_Q(S, pi.reshape(NA, NA), kernel)
        else:                                              # gtr: closed-form pi-step
            pi, _s = FP.mstep_pi(F, pi, N, T)
            Q = FP._met_Q(S, pi.reshape(NA, NA), kernel)
    else:                                                  # synchronized / coupled flux
        F = FP.mstep_flux(N, T, pi, orbit_id, n_orbits, mask)
        pi, s = FP.mstep_pi(F, pi, N, T)
        Q = s * pi[None, :]; np.fill_diagonal(Q, 0.0); Q[np.diag_indices(NS)] = -Q.sum(1)
    return Q, F, pi, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/per_contact_trrosetta/counts.npz")
    ap.add_argument("--model", required=True,
                    help="synchronized | coupled | metropolis_{sqrt,barker,hastings,gtr}")
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
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model = a.model
    is_met = model.startswith("metropolis_")
    kernel = model.split("_", 1)[1] if is_met else None
    if not is_met and model not in ("synchronized", "coupled"):
        raise SystemExit(f"unknown model {model}")
    rng = np.random.default_rng(a.seed); t0 = time.time()
    mode = "single-rate" if a.single_rate else f"gammaI(g{a.gamma_cats})"

    # ---- corpus + split (IDENTICAL to fit_coupling_mixture_rateI) ----
    (PF, PT, SEG, tau, T, G, tot_cnt, keep,
     trPF, trPT, trTB, trCNT, trSEG, tr_g, tr_tot,
     vaPF, vaPT, vaTB, vaCNT, vaSEG, va_g, va_tot) = FS.load_split_core(
        a.corpus, rng, a.val_frac, a.min_counts)
    conserved = np.bincount(SEG, weights=(PF != PT).astype(float), minlength=G) == 0
    flat = (trPF * NS + trPT) * T + trTB
    print(f"# [{model} | {mode}] {G} clusters ({keep.sum()} kept); "
          f"train {len(tr_g)} / val {len(va_g)}; {int(tr_tot):,}/{int(va_tot):,} "
          f"transitions", flush=True)

    # ---- rate grid (identical to RI) ----
    if a.single_rate:
        R_gamma = 1; rate_vals = np.array([1.0]); is_inv = np.array([False])
        rho = np.array([1.0]); alpha = float("nan"); p_inv = 0.0
    else:
        R_gamma = a.gamma_cats; alpha = a.alpha_init
        g = discrete_gamma_rates(R_gamma, alpha)
        rate_vals = np.concatenate([[0.0], g]); is_inv = np.array([True] + [False] * R_gamma)
        p_inv = a.pinv
        rho = np.concatenate([[p_inv], np.full(R_gamma, (1 - p_inv) / R_gamma)])
    R = len(rate_vals); gamma_idx = [r for r in range(R) if not is_inv[r]]

    # ---- init single component (empirical pi; flux/S init like FP.fit_model) ----
    occ = (np.bincount(trPF, weights=trCNT, minlength=NS)
           + np.bincount(trPT, weights=trCNT, minlength=NS))
    pi = (occ / occ.sum())
    pim = pi.reshape(NA, NA); pim = 0.5 * (pim + pim.T); pi = (pim / pim.sum()).reshape(NS)
    orbit_id, n_orbits, is_single = FP.build_orbits()
    off = ~np.eye(NS, dtype=bool)
    mask = is_single if (model == "coupled" or is_met) else off
    S = S_LG.copy()
    if is_met:
        F = None
        Q = FP._met_Q(S, pi.reshape(NA, NA), kernel)
    else:
        F = np.zeros((NS, NS))
        F[off & mask] = 1e-2 * np.sqrt(pi[:, None] * pi[None, :])[off & mask]
        Q = FP.Q_from_flux(F, pi)
    w = np.array([1.0]); logw = np.array([0.0])

    prev = None; trll = float("nan"); monotone = True
    for it in range(a.em_iters):
        logrho = np.log(rho + 1e-300)
        eig = FP.eig_rev(Q, np.clip(pi, 1e-12, None))
        # E-step: rate responsibilities + train marginal LL (K=1 -> single class)
        sc = RI.build_scores([eig], [pi], rate_vals, is_inv, logw, logrho, tau,
                             trPF, trPT, trTB, trCNT, trSEG, G, tr_g, conserved)
        trll = RI.marginal_percount_ll(sc, tr_tot)
        if prev is not None and trll < prev - 1e-9:
            monotone = False
        Rr = RI.responsibilities(sc)                        # (n_tr,1,R)
        R_full = np.zeros((G, 1, R)); R_full[tr_g] = Rr
        rho = Rr.sum((0, 1)) + 1e-12; rho /= rho.sum()
        if not a.single_rate:
            p_inv = float(rho[0])
        # M-step: aggregate HR N,T over Gamma bins at rate-scaled branch lengths,
        # then the model-specific M-step (`inner` ECM rounds).
        for _ in range(a.inner):
            eig = FP.eig_rev(Q, np.clip(pi, 1e-12, None))
            N = np.zeros((NS, NS)); Tt = np.zeros(NS)
            for r in gamma_idx:
                wnnz = trCNT * R_full[trSEG, 0, r]
                if wnnz.sum() == 0:
                    continue
                Ncr = np.bincount(flat, weights=wnnz,
                                  minlength=NS * NS * T).reshape(NS, NS, T)
                Nn, Ttt = RI.estep_scaled(Q, pi, tau, rate_vals[r], Ncr, eig)
                N += Nn; Tt += Ttt
            Q, F, pi, S = model_mstep(model, is_met, kernel, N, Tt, pi, S,
                                      orbit_id, n_orbits, mask)
        # M-step alpha
        if (not a.single_rate and R_gamma >= 2 and it % a.alpha_every == 0):
            eig = FP.eig_rev(Q, np.clip(pi, 1e-12, None))
            alpha, g = RI.refit_alpha(alpha, R_gamma, [eig], R_full, gamma_idx, tau,
                                      trPF, trPT, trTB, trCNT, trSEG, G, tr_g, ALPHA_CANDS)
            rate_vals = np.concatenate([[0.0], g])
        extra = "" if a.single_rate else f" alpha={alpha:.3f} pinv={p_inv:.3f}"
        print(f"  it {it:2d}: train={trll:.4f} MI(pi)={FS.mi(pi):.4f}{extra} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        if prev is not None and abs(trll - prev) < a.tol:
            print("  converged", flush=True); break
        prev = trll

    # ---- final held-out scoring ----
    eig = FP.eig_rev(Q, np.clip(pi, 1e-12, None))
    logrho = np.log(rho + 1e-300)
    sc_va = RI.build_scores([eig], [pi], rate_vals, is_inv, logw, logrho, tau,
                            vaPF, vaPT, vaTB, vaCNT, vaSEG, G, va_g, conserved)
    vall = RI.marginal_percount_ll(sc_va, va_tot)
    if is_met:
        # symmetric flux F = S * sym(pi_x,pi_y) implied by the fitted (S, pi)
        F_out = FP._met_build_from_S(S, pi.reshape(NA, NA), kernel, FP._met_sym)
    else:
        F_out = F
    res = dict(
        model=model, mode=mode, single_rate=bool(a.single_rate),
        gamma_cats=int(R_gamma), val_per_count_ll=float(vall),
        train_per_count_ll=float(trll), monotone=bool(monotone), n_em_iters=int(it + 1),
        mi_pi=float(FS.mi(pi)),
        alpha=(None if a.single_rate else float(alpha)),
        p_inv=(None if a.single_rate else float(p_inv)),
        rate_weights=[float(x) for x in rho], rate_vals=[float(x) for x in rate_vals],
        mean_rate=float(sum(rho[r] * rate_vals[r] for r in range(R))),
        n_params_flux=int(FP.n_params(model, n_orbits, int(is_single.sum() // 2))
                          if hasattr(FP, "n_params") else -1),
        n_train_clusters=int(len(tr_g)), n_val_clusters=int(len(va_g)), seed=a.seed)
    print(f"# [{model} | {mode}] VAL/count={vall:.4f} train={trll:.4f} "
          f"monotone={monotone} iters={it+1}  MI(pi)={FS.mi(pi):.4f}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        np.savez(a.out.replace(".json", ".npz"),
                 model=np.array(model), Q=np.asarray(Q), F=np.asarray(F_out),
                 pi=np.asarray(pi), S=np.asarray(S) if is_met else np.zeros((NA, NA)),
                 alphabet=np.array(list("ACDEFGHIKLMNPQRSTVWY")),
                 rate_vals=np.asarray(rate_vals), rate_weights=np.asarray(rho),
                 alpha=np.array(np.nan if a.single_rate else alpha),
                 p_inv=np.array(0.0 if a.single_rate else p_inv), tau=np.asarray(tau),
                 val_per_count_ll=float(vall), train_per_count_ll=float(trll))
        print(f"# wrote {a.out} + .npz", flush=True)
    return res


if __name__ == "__main__":
    main()
