#!/usr/bin/env python3
r"""Mixture of coevolutionary Potts pair-CTMCs with a NULL-SPIKED, L1-SPARSE coupling prior
(+ Gamma+I rate heterogeneity), fit to the per-contact substitution-count corpus.

Each class k is a reversible pair chain with stationary pi_k(a,b) ~ exp(h_a+h_b+J_ab) and the
shared Metropolis-sqrt generator (exchangeability S).  The coupling J^{(k)} carries a
spike-and-slab prior: a class-level null indicator gamma_k (spike at J=0 = product-of-marginals)
plus a Laplace/L1 slab.  The M-step fits (h_k, J_k) by L1-proximal (soft-threshold) gradient
ascent on the complete-data log-likelihood, then a Bayes-factor/BIC decision turns the coupling
on only where it clears the null.  Derivation: psb-paper/spiked_potts_prior.tex.

Reuses the E-step / Gamma+I / mixture-weight machinery of fit_coupling_mixture_rateI (RI.*) and
the split of fit_coupling_mixture_freeS (FS.*).  Warm-starts S and per-class composition from a
fitted Metropolis mixture (results/mixture_component_char/components_K{K}.npz); J starts at 0
(null) so the run reports which classes the data turn on.  Run with PYTHONPATH=src.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.special as jss
import fit_pair_models as FP                                     # noqa: E402
from fit_pair_models import NA, NS                               # noqa: E402
import fit_coupling_mixture_freeS as FS                          # noqa: E402
import fit_coupling_mixture_rateI as RI                          # noqa: E402
from tkfdp.coupling.dynfield.phylo_elbo.rate_hetero import discrete_gamma_rates  # noqa: E402


# ---------------------------------------------------------------- differentiable Potts core
@jax.jit
def _potts_Q(S, h, J):
    logpi = h[:, None] + h[None, :] + J
    logpi = logpi - jss.logsumexp(logpi)
    sq = jnp.exp(0.5 * logpi)                                   # sqrt pi(a,c)
    T1 = S[:, :, None] * (sq[None, :, :] / sq[:, None, :])      # S[a,b]*sq[b,c]/sq[a,c]
    T2 = S[None, :, :] * (sq[:, None, :] / sq[:, :, None])      # S[c,d]*sq[a,d]/sq[a,c]
    I = jnp.eye(NA)
    Q4 = jnp.einsum('abc,cd->acbd', T1, I) + jnp.einsum('acd,ab->acbd', T2, I)
    Qf = Q4.reshape(NS, NS)
    Qf = Qf - jnp.diag(Qf.sum(1))
    return Qf, jnp.exp(logpi).reshape(NS)


def _cdll(h, J, S, N, T):
    """Per-count expected complete-data LL: (sum_{x!=y} N log Q + sum_x T*Q_xx) / n."""
    Qf, _ = _potts_Q(S, h, J)
    off = jnp.where(N > 0.0, N * jnp.log(jnp.clip(Qf, 1e-300, None)), 0.0)
    n = jnp.maximum(N.sum(), 1.0)
    return (jnp.sum(off) + jnp.sum(T * jnp.diag(Qf))) / n


_cdll_vg = jax.jit(jax.value_and_grad(_cdll, argnums=(0, 1)))


def _fit_class(h0, J0, S, N, T, lam, coupled, inner=60, eta=0.5):
    """L1-proximal gradient ascent of _cdll - lam*|J|_1 over (h, J).  coupled=False holds J=0
    (null / product-of-marginals fit).  Returns (h, J, per-count ll, k_active)."""
    Sj = jnp.asarray(S); Nj = jnp.asarray(N); Tj = jnp.asarray(T)
    h = np.asarray(h0, float).copy()
    J = np.zeros((NA, NA)) if not coupled else np.asarray(J0, float).copy()
    prev = -np.inf
    for _ in range(inner):
        ll, (gh, gJ) = _cdll_vg(jnp.asarray(h), jnp.asarray(J), Sj, Nj, Tj)
        h = h + eta * np.asarray(gh)
        if coupled:
            J = J + eta * np.asarray(gJ)
            J = np.sign(J) * np.maximum(np.abs(J) - eta * lam, 0.0)   # soft-threshold (L1 prox)
            J = 0.5 * (J + J.T); np.fill_diagonal(J, 0.0)
        if abs(float(ll) - prev) < 1e-9:
            break
        prev = float(ll)
    ll = float(_cdll(jnp.asarray(h), jnp.asarray(J), Sj, Nj, Tj))
    k = int((np.abs(np.triu(J, 1)) > 1e-9).sum())
    return h, J, ll, k


def mstep_spiked(R_full, rate_vals, gamma_idx, hs, Js, gammas, S, tau, trPF, trPT, trSEG,
                 trCNT, flat, T, lam, rho, inner):
    """Per-class spiked-Potts M-step at fixed responsibilities.  Aggregate HR usage/dwell over
    the Gamma rate bins, fit the coupled (h,J) by L1-proximal ascent and the null (J=0) fit, and
    choose gamma_k by a Laplace/BIC Bayes-factor test."""
    K = len(hs); Sj = np.asarray(S, float)
    new_h, new_J, new_g, Qs, pis, eigs, info = [], [], [], [], [], [], []
    for c in range(K):
        logpi = hs[c][:, None] + hs[c][None, :] + Js[c]; logpi -= logpi.max()
        pic = np.exp(logpi); pic /= pic.sum()
        Q_c = FP._met_Q(Sj, pic, "sqrt"); eig_c = FP.eig_rev(Q_c, np.clip(pic.reshape(NS), 1e-12, None))
        N_c = np.zeros((NS, NS)); T_c = np.zeros(NS)
        for r in gamma_idx:
            wnnz = trCNT * R_full[trSEG, c, r]
            if wnnz.sum() <= 0:
                continue
            Ncr = np.bincount(flat, weights=wnnz, minlength=NS * NS * T).reshape(NS, NS, T)
            Nn, Tt = RI.estep_scaled(Q_c, pic.reshape(NS), tau, rate_vals[r], Ncr, eig_c)
            N_c += Nn; T_c += Tt
        n_c = max(N_c.sum(), 1.0)
        h1, J1, ll1, k1 = _fit_class(hs[c], Js[c], Sj, N_c, T_c, lam, coupled=True, inner=inner)
        h0, _, ll0, _ = _fit_class(hs[c], None, Sj, N_c, T_c, lam, coupled=False, inner=inner)
        gain = (ll1 - lam * np.abs(np.triu(J1, 1)).sum()) - ll0    # per-count penalised gain
        thresh = (0.5 * k1 * np.log(max(n_c, 2.0)) + np.log((1 - rho) / rho)) / n_c   # BIC + spike
        gam = 1 if gain > thresh else 0
        if gam:
            h, J, k = h1, J1, k1
        else:
            h, J, k = h0, np.zeros((NA, NA)), 0
        logpi = h[:, None] + h[None, :] + J; logpi -= logpi.max()
        pic = np.exp(logpi); pic /= pic.sum(); pic = pic.reshape(NS)
        Qc = FP._met_Q(Sj, pic.reshape(NA, NA), "sqrt")
        ec = FP.eig_rev(Qc, np.clip(pic, 1e-12, None))
        new_h.append(h); new_J.append(J); new_g.append(gam)
        Qs.append(Qc); pis.append(pic); eigs.append(ec)
        info.append((gam, k, gain, thresh, n_c))
    return new_h, new_J, new_g, Qs, pis, eigs, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="data/per_contact_trrosetta/counts.npz")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--init-mixture", default=None)
    ap.add_argument("--lam", type=float, default=0.02, help="L1 rate on J (per-count)")
    ap.add_argument("--rho", type=float, default=0.5, help="prior prob a class couples")
    ap.add_argument("--gamma-cats", type=int, default=4)
    ap.add_argument("--pinv", type=float, default=0.2)
    ap.add_argument("--em-iters", type=int, default=15)
    ap.add_argument("--inner", type=int, default=60)
    ap.add_argument("--alpha-init", type=float, default=1.0)
    ap.add_argument("--alpha-every", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-counts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); t0 = time.time(); K = a.K
    init_path = a.init_mixture or f"results/mixture_component_char/components_K{K}.npz"

    (PF, PT, SEG, tau, T, G, tot_cnt, keep, trPF, trPT, trTB, trCNT, trSEG, tr_g, tr_tot,
     vaPF, vaPT, vaTB, vaCNT, vaSEG, va_g, va_tot) = FS.load_split_core(
        a.corpus, rng, a.val_frac, a.min_counts)
    conserved = np.bincount(SEG, weights=(PF != PT).astype(float), minlength=G) == 0
    flat = (trPF * NS + trPT) * T + trTB
    print(f"# [spiked-Potts K={K}] lam={a.lam} rho={a.rho}; {G} clusters, train {len(tr_g)}/val "
          f"{len(va_g)}; {int(tr_tot):,}/{int(va_tot):,} transitions", flush=True)

    # Gamma+I grid
    R_gamma = a.gamma_cats; alpha = a.alpha_init
    g = discrete_gamma_rates(R_gamma, alpha)
    rate_vals = np.concatenate([[0.0], g]); is_inv = np.array([True] + [False] * R_gamma)
    p_inv = a.pinv; rho_rate = np.concatenate([[p_inv], np.full(R_gamma, (1 - p_inv) / R_gamma)])
    R = len(rate_vals); gamma_idx = [r for r in range(R) if not is_inv[r]]

    # init: S + per-class composition from the Metropolis mixture; J = 0 (null start)
    zc = np.load(init_path, allow_pickle=True); assert int(zc["K"]) == K
    S = np.asarray(zc["S"], float); seed_pis = np.asarray(zc["pis"], float)
    w = np.asarray(zc["weights"], float).copy()
    hs, Js, gammas = [], [], []
    for c in range(K):
        th = seed_pis[c].reshape(NA, NA).sum(1); th = np.maximum(th, 1e-8); th /= th.sum()
        hs.append(np.log(th)); Js.append(np.zeros((NA, NA))); gammas.append(0)
    print(f"# warm-started S + composition from {init_path}; J=0 (all classes start null)", flush=True)
    Qs = [FP._met_Q(S, np.exp(hs[c][:, None] + hs[c][None, :]).reshape(NA, NA) /
                    np.exp(hs[c][:, None] + hs[c][None, :]).sum(), "sqrt") for c in range(K)]
    pis = [np.exp(hs[c][:, None] + hs[c][None, :]).reshape(NS) /
           np.exp(hs[c][:, None] + hs[c][None, :]).sum() for c in range(K)]
    eigs = [FP.eig_rev(Qs[c], np.clip(pis[c], 1e-12, None)) for c in range(K)]

    prev = None; trll = float("nan")
    for it in range(a.em_iters):
        logw = np.log(w + 1e-300); logrho = np.log(rho_rate + 1e-300)
        sc = RI.build_scores(eigs, pis, rate_vals, is_inv, logw, logrho, tau,
                             trPF, trPT, trTB, trCNT, trSEG, G, tr_g, conserved)
        trll = RI.marginal_percount_ll(sc, tr_tot)
        Rr = RI.responsibilities(sc); R_full = np.zeros((G, K, R)); R_full[tr_g] = Rr
        w = Rr.sum((0, 2)) + 1e-9; w /= w.sum()
        rho_rate = Rr.sum((0, 1)) + 1e-12; rho_rate /= rho_rate.sum(); p_inv = float(rho_rate[0])
        hs, Js, gammas, Qs, pis, eigs, info = mstep_spiked(
            R_full, rate_vals, gamma_idx, hs, Js, gammas, S, tau, trPF, trPT, trSEG,
            trCNT, flat, T, a.lam, a.rho, a.inner)
        if R_gamma >= 2 and it % a.alpha_every == 0:
            alpha, g = RI.refit_alpha(alpha, R_gamma, eigs, R_full, gamma_idx, tau,
                                      trPF, trPT, trTB, trCNT, trSEG, G, tr_g,
                                      (0.6, 0.78, 1.0, 1.28, 1.66))
            rate_vals = np.concatenate([[0.0], g])
        n_on = sum(gammas)
        print(f"  it {it:2d}: train_mix={trll:.4f}  coupled_classes={n_on}/{K}  "
              f"w={np.round(np.sort(w)[::-1], 3)} alpha={alpha:.2f} [{time.time()-t0:.0f}s]", flush=True)
        if prev is not None and abs(trll - prev) < 1e-5:
            print("  converged", flush=True); break
        prev = trll

    # held-out + report
    logw = np.log(w + 1e-300); logrho = np.log(rho_rate + 1e-300)
    sc_va = RI.build_scores(eigs, pis, rate_vals, is_inv, logw, logrho, tau,
                            vaPF, vaPT, vaTB, vaCNT, vaSEG, G, va_g, conserved)
    vall = RI.marginal_percount_ll(sc_va, va_tot)
    AA = "ACDEFGHIKLMNPQRSTVWY"
    order = np.argsort(w)[::-1]
    print(f"\n# [spiked-Potts K={K} lam={a.lam}] VAL/count={vall:.4f} train={trll:.4f} "
          f"coupled_classes={sum(gammas)}/{K}", flush=True)
    print(f"#  {'w':>6} {'gamma':>5} {'k_pairs':>7} {'gain/ct':>9}  top coupled pairs", flush=True)
    for c in order:
        gam, k, gain, thr, n_c = info[c]
        Jc = Js[c]; iu = np.triu_indices(NA, 1); mag = np.abs(Jc[iu])
        top = np.argsort(mag)[::-1][:5]
        pairs = " ".join(f"{AA[iu[0][t]]}{AA[iu[1][t]]}{'+' if Jc[iu[0][t],iu[1][t]]>0 else '-'}"
                         for t in top if mag[t] > 1e-9)
        print(f"#   {w[c]:>6.3f} {gam:>5} {k:>7} {gain:>9.5f}  {pairs}", flush=True)
    if a.out:
        json.dump(dict(K=K, lam=a.lam, rho=a.rho, val_per_count_ll=float(vall),
                       train_per_count_ll=float(trll), coupled_classes=int(sum(gammas)),
                       weights=[float(w[c]) for c in order],
                       gammas=[int(gammas[c]) for c in order],
                       k_pairs=[int(info[c][1]) for c in order],
                       gain=[float(info[c][2]) for c in order]), open(a.out, "w"), indent=2)
        print(f"# wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
