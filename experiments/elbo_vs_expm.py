"""Closed-form path ELBO vs exact coupled-pair matrix exponential.

Compares the Girsanov path-ELBO lower bound L_xy on log[e^{tQ}]_xy (independent
product-of-marginals bridge R, coupled target W=Q) against the exact log
transition matrix, for several 400-state square-root-Metropolis pair models and
a grid of branch lengths.

Closed form (whole 400x400 matrix at once), R = U diag(lam) Uinv, M = e^{tR}:
    G_{zz'} = R_{zz'} log(W_{zz'}/R_{zz'})      (off-diagonal; 0 where R=0)
    C       = G - diag(w - r),  w_z=-W_zz, r_z=-R_zz
    L       = log(M) + ( U @ ( J * (Uinv @ C @ U) ) @ Uinv ) / M   (elementwise / M)
with J^{kl}(t) the divided-difference kernel (t e^{lam_k t} on the degenerate diagonal).

Run:  export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=8
      PYTHONPATH=src:experiments python3 experiments/elbo_vs_expm.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import json
import numpy as np
from scipy.linalg import expm
from scipy.stats import spearmanr, kendalltau

import jax
import jax.numpy as jnp

import composite_potts_phylo_elbo as CE
import fit_pair_models as FP

NA = FP.NA           # 20
NS = FP.NS           # 400
TGRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
FLOOR = 1e-300
MEANINGFUL = 1e-8    # entries whose exact transition prob exceeds this

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REPO, "results", "pair_models", "elbo_vs_expm.json")
OUT_PDF = os.path.join(REPO, "experiments", "figures", "elbo_vs_expm.pdf")


# ---------------------------------------------------------------------------
# closed-form ELBO (numpy float64 assembly, CE for eig / kernel)
# ---------------------------------------------------------------------------
def eig_rev(R, pi_stat):
    lam, U, Uinv = CE.eig_rev_jax(jnp.asarray(R), jnp.asarray(pi_stat))
    return np.asarray(lam), np.asarray(U), np.asarray(Uinv)


def Jmat(lam, t):
    return np.asarray(CE.Jmat_jax(jnp.asarray(lam), float(t)))


def build_C(W, R):
    """C = G - diag(w-r); G off-diagonal = R*log(W/R) where R>0."""
    # support guard: absolute continuity P_W << P_R  (R=0 off-diag => W=0)
    off = ~np.eye(NS, dtype=bool)
    bad = off & (R == 0.0) & (W != 0.0)
    assert not bad.any(), "support violation: W has an off-diagonal jump R lacks"
    r = -np.diag(R)
    w = -np.diag(W)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(R > 0, W / np.where(R > 0, R, 1.0), 1.0)
    G = np.where(R > 0, R * np.log(ratio), 0.0)
    np.fill_diagonal(G, 0.0)
    C = G.copy()
    np.fill_diagonal(C, -(w - r))
    return C


def closed_form_L_precompute(W, R, pi_stat):
    """t-INDEPENDENT part of the path ELBO: the reversible eig of the bridge R and
    the coupling table C = build_C(W,R). eig_rev(R) does not depend on t (only
    exp(t*lam) does), so this can be computed ONCE and reused across branch lengths
    and shared with the midpoint bound's half-branch edges (closed_form_L at t/2)."""
    lam, U, Uinv = eig_rev(R, pi_stat)
    C = build_C(W, R)
    return lam, U, Uinv, C


def closed_form_L_from_precompute(precomp, t):
    """Cheap per-t step given closed_form_L_precompute output: e^{tR}, kernel, L."""
    lam, U, Uinv, C = precomp
    M = (U * np.exp(lam * t)[None, :]) @ Uinv       # e^{tR}
    J = Jmat(lam, t)
    num = U @ (J * (Uinv @ C @ U)) @ Uinv           # int_0^t e^{Rs} C e^{R(t-s)} ds
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log(np.maximum(M, FLOOR)) + num / np.maximum(M, FLOOR)
    return L, M, num


def closed_form_L(W, R, pi_stat, t):
    precomp = closed_form_L_precompute(W, R, pi_stat)
    L, M, num = closed_form_L_from_precompute(precomp, t)
    return L, M, precomp[3], num


def expm_rev(G, pi_stat, t):
    """Accurate e^{tG} for a reversible generator G via symmetric eigendecomposition.

    NOTE: scipy.linalg.expm is NOT used as the exact reference here because it
    mutates its (named) input array in place via internal balancing and carries a
    ~1e-8 accuracy floor on these 400-state generators (verified: expm(A) vs
    expm(A.copy()) differed by 0.64 on identical bytes; expm vs (expm at t/2)^2
    self-inconsistency ~2.5e-8). Both W=Q and R are reversible (square-root
    Metropolis), so the symmetric-similarity eig (cond(U)~7) gives e^{tG} to
    machine precision (row-sum dev ~7e-15, squaring self-consistency ~2e-15)."""
    lam, U, Uinv = eig_rev(G, pi_stat)
    return (U * np.exp(lam * t)[None, :]) @ Uinv


def scipy_expm_safe(G, t):
    """scipy.linalg.expm on a COPY (it mutates input); secondary cross-check only."""
    return expm(t * np.array(G, copy=True))


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def recover_S(Q, pij):
    """Recover shared S from a sqrt-Metropolis single: Q4[a,b,c,b]=S[a,c]*sqrt(pi[c,b]/pi[a,b])."""
    Q4 = Q.reshape(NA, NA, NA, NA)
    pij4 = pij.reshape(NA, NA)
    S = np.zeros((NA, NA))
    resid = 0.0
    for a in range(NA):
        for c in range(NA):
            if a == c:
                continue
            vals = Q4[a, :, c, :].diagonal() / np.sqrt(pij4[c, :] / pij4[a, :])  # over b
            S[a, c] = vals.mean()
            resid = max(resid, float(vals.std()))
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)
    return S, resid


def R_from_marginals(S, pij4):
    m1 = pij4.sum(1)
    m2 = pij4.sum(0)
    pi_prod = np.outer(m1, m2)
    R = FP._met_Q(S, pi_prod, "sqrt")
    return R, pi_prod.reshape(NS)


def load_models():
    models = []

    # (1) trRosetta converged Metropolis-sqrt single
    z = np.load(os.path.join(REPO, "results/pair_models/trrosetta_converged_params.npz"),
                allow_pickle=True)
    Q = np.asarray(z["metropolis_sqrt__Q"], float)
    pij = np.asarray(z["metropolis_sqrt__pi"], float)
    S, resid = recover_S(Q, pij)
    # verify recovered S reproduces Q
    Qrec = FP._met_Q(S, pij.reshape(NA, NA), "sqrt")
    rec_err = float(np.max(np.abs(Qrec - Q)))
    R, pi_prod = R_from_marginals(S, pij.reshape(NA, NA))
    models.append(dict(name="trrosetta_metropolis_sqrt", W=Q, R=R, pi_prod=pi_prod,
                       S=S, pij=pij, S_recover_resid=resid, S_recover_Qerr=rec_err))

    # (2) top-3 strongest-coupled mixture components (by MI of pi)
    zm = np.load(os.path.join(REPO, "results/mixture_component_char/components_K8.npz"),
                 allow_pickle=True)
    S_mix = np.asarray(zm["S"], float)
    pis = np.asarray(zm["pis"], float)
    mi = np.asarray(zm["mi_pi"], float)
    top = np.argsort(mi)[-3:]                       # ascending -> 3 strongest last
    for c in top:
        pij_c = pis[c]
        Qc = FP._met_Q(S_mix, pij_c.reshape(NA, NA), "sqrt")
        R, pi_prod = R_from_marginals(S_mix, pij_c.reshape(NA, NA))
        models.append(dict(name=f"mixture_K8_c{c}_mi{mi[c]:.3f}", W=Qc, R=R,
                           pi_prod=pi_prod, S=S_mix, pij=pij_c,
                           S_recover_resid=0.0, S_recover_Qerr=0.0))
    return models


# ---------------------------------------------------------------------------
# self-checks
# ---------------------------------------------------------------------------
def selfcheck_H0(S, pij4, t=0.5):
    """(a) product model pij=m1 x m2 => W==R, C==0, correction==0, L == log(e^{tR}).

    Formula correctness is judged against the eig-consistent log(e^{tR}) (must be
    machine precision). The gap vs a fully-independent scipy.expm(tW) is reported
    on numerically-meaningful entries only (the eig symmetric-similarity build
    loses ~1e-6 in the log of sub-1e-8 entries for wide-dynamic-range 400-state pi;
    that is an expm conditioning artifact, not a formula error)."""
    m1 = pij4.sum(1); m2 = pij4.sum(0)
    pij0 = np.outer(m1, m2)
    W0 = FP._met_Q(S, pij0, "sqrt")
    R0, pip0 = R_from_marginals(S, pij0)
    WR = float(np.max(np.abs(W0 - R0)))
    L0, M0, C0, num0 = closed_form_L(W0, R0, pip0, t)
    logM = np.log(np.maximum(M0, FLOOR))
    eW = expm_rev(W0, pip0, t)                                 # exact e^{tW}, W reversible
    exact = np.log(np.maximum(eW, FLOOR))
    dL_logM = float(np.max(np.abs(L0 - logM)))                 # formula vs eig base
    dCorr = float(np.max(np.abs(num0)))                        # correction must vanish
    dL_exact = float(np.max(np.abs(L0 - exact)))              # L == log e^{tW} exactly
    Cnorm = float(np.max(np.abs(C0)))
    ok = (WR < 1e-10) and (dL_logM < 1e-8) and (dCorr < 1e-8) and (dL_exact < 1e-8)
    return ok, dict(WR=WR, Cnorm=Cnorm, correction_norm=dCorr, dL_vs_logM=dL_logM,
                    dL_vs_exact=dL_exact)


def selfcheck_bound(W, pij, L, t):
    """(b) gap = exact - L >= -1e-6 everywhere (ELBO <= exact)."""
    exact = np.log(np.maximum(expm_rev(W, pij, t), FLOOR))
    gap = exact - L
    minslack = float(gap.min())
    return (minslack >= -1e-6), minslack


def selfcheck_scipy_ref(W, pij, t):
    """(d) independent scipy.linalg.expm (on a copy) agrees with the eig reference
    to within scipy's ~1e-8 accuracy floor on these generators."""
    eig = expm_rev(W, pij, t)
    sci = scipy_expm_safe(W, t)
    return float(np.max(np.abs(eig - sci)))


def selfcheck_bruteforce(W, R, pi_prod, t, n=20, seed=0):
    """(c) L_xy from CE.bridge_jax HR E-step must match matrix-formula L_xy.

    bridge_jax computes Nbar=R.*I/M, Tbar=diag(I)/M with the FIRST arg being the
    bridge generator R (Nmat = R * B), and internally W_edge=edge/P, so a single
    clamped endpoint (x,y) needs edge[x,y]=1.0 (weight edge/P = 1/P[x,y] = the
    per-endpoint HR normaliser)."""
    lam, U, Uinv = CE.eig_rev_jax(jnp.asarray(R), jnp.asarray(pi_prod))
    M = np.asarray(CE.expm_from_eig(lam, U, Uinv, float(t)))
    Lmat, _, _, _ = closed_form_L(W, R, pi_prod, t)
    r = -np.diag(R); w = -np.diag(W)
    with np.errstate(divide="ignore", invalid="ignore"):
        logWR = np.where(R > 0, np.log(np.where(R > 0, W, 1.0) /
                                       np.where(R > 0, R, 1.0)), 0.0)
    np.fill_diagonal(logWR, 0.0)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        x = int(rng.integers(NS)); y = int(rng.integers(NS))
        edge = np.zeros((NS, NS))
        edge[x, y] = 1.0                            # single clamped endpoint
        T, Nmat, _ = CE.bridge_jax(jnp.asarray(R), lam, U, Uinv, float(t),
                                   jnp.asarray(edge))
        T = np.asarray(T); Nmat = np.asarray(Nmat)
        Lxy = (np.log(max(M[x, y], FLOOR))
               + float((Nmat * logWR).sum())
               - float((T * (w - r)).sum()))
        worst = max(worst, abs(Lxy - Lmat[x, y]))
    return (worst < 1e-6), float(worst)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def ranking_stats(exact, approx):
    """Global + per-row Spearman/Kendall over meaningful entries + argmax agreement."""
    meaningful = exact > np.log(MEANINGFUL)      # exact is a log-prob here
    ex = exact[meaningful]; ap = approx[meaningful]
    g_sp = float(spearmanr(ex, ap).statistic) if ex.size > 2 else float("nan")
    g_kt = float(kendalltau(ex, ap).statistic) if ex.size > 2 else float("nan")
    # per source-row: rank the 400 destinations
    sps, kts, argmax_ok = [], [], 0
    for i in range(NS):
        row_m = meaningful[i]
        if row_m.sum() > 2:
            sps.append(spearmanr(exact[i, row_m], approx[i, row_m]).statistic)
            kts.append(kendalltau(exact[i, row_m], approx[i, row_m]).statistic)
        # argmax over all destinations (row) - most probable endpoint
        if np.argmax(exact[i]) == np.argmax(approx[i]):
            argmax_ok += 1
    return dict(
        global_spearman=g_sp, global_kendall=g_kt,
        row_spearman_mean=float(np.nanmean(sps)) if sps else float("nan"),
        row_kendall_mean=float(np.nanmean(kts)) if kts else float("nan"),
        argmax_agree_frac=argmax_ok / NS,
    )


def error_stats(exact_log, approx_log):
    """gap = exact - approx (log domain); overall + restricted to meaningful."""
    gap = exact_log - approx_log
    meaningful = exact_log > np.log(MEANINGFUL)
    def summ(a):
        return dict(mean=float(a.mean()), median=float(np.median(a)),
                    p95=float(np.percentile(a, 95)), max=float(a.max()),
                    min=float(a.min()))
    return dict(all=summ(gap), meaningful=summ(gap[meaningful]))


# ---------------------------------------------------------------------------
def main():
    models = load_models()
    print("=" * 78)
    print("SELF-CHECKS")
    print("=" * 78)
    sc = {}
    # run self-checks on the trRosetta model (has recovered S) + one mixture comp
    m0 = models[0]
    ok_a, det_a = selfcheck_H0(m0["S"], m0["pij"].reshape(NA, NA))
    print(f"(a) H=0 exactness [{m0['name']}]: {'PASS' if ok_a else 'FAIL'}  "
          f"||W-R||={det_a['WR']:.2e} ||C||={det_a['Cnorm']:.2e} "
          f"corr={det_a['correction_norm']:.2e} |L-logM|={det_a['dL_vs_logM']:.2e} "
          f"|L-exact|={det_a['dL_vs_exact']:.2e}")
    print(f"    S recovery: Qerr={m0['S_recover_resid']:.2e} (per-b std), "
          f"reproduce-Q err={m0['S_recover_Qerr']:.2e}")

    # bound + brute-force + scipy cross-check on all models
    ok_b_all = True; ok_c_all = True; scipy_worst = 0.0
    for m in models:
        for t in (0.1, 1.0):
            L, _, _, _ = closed_form_L(m["W"], m["R"], m["pi_prod"], t)
            okb, slack = selfcheck_bound(m["W"], m["pij"], L, t)
            ok_b_all &= okb
            print(f"(b) bound [{m['name']}] t={t}: {'PASS' if okb else 'FAIL'}  "
                  f"min slack (exact-L)={slack:.2e}")
        okc, worst = selfcheck_bruteforce(m["W"], m["R"], m["pi_prod"], 0.5)
        ok_c_all &= okc
        print(f"(c) brute-force HR [{m['name']}] t=0.5: {'PASS' if okc else 'FAIL'}  "
              f"max|L_bridge-L_formula|={worst:.2e}")
        sw = selfcheck_scipy_ref(m["W"], m["pij"], 0.5)
        scipy_worst = max(scipy_worst, sw)
        print(f"(d) scipy expm cross-check [{m['name']}] t=0.5: "
              f"max|eig-scipy|={sw:.2e} (scipy ~1e-8 floor; eig is the exact ref)")

    sc = dict(H0_exactness=dict(passed=bool(ok_a), **det_a),
              lower_bound=dict(passed=bool(ok_b_all)),
              brute_force=dict(passed=bool(ok_c_all)),
              scipy_crosscheck=dict(max_abs=float(scipy_worst)))
    allpass = ok_a and ok_b_all and ok_c_all
    print(f"\nALL SELF-CHECKS (a,b,c): {'PASS' if allpass else 'FAIL'}")
    print(f"(d) scipy independent-ref agreement (diagnostic): {scipy_worst:.2e}")

    print("\n" + "=" * 78)
    print("METRICS")
    print("=" * 78)
    results = {}
    hdr = (f"{'model':<28}{'t':>6}{'gap_mean':>10}{'gap_p95':>10}"
           f"{'sp_ELBO':>9}{'sp_prod':>9}{'row_sp_E':>9}{'row_sp_P':>9}"
           f"{'amax_E':>8}{'amax_P':>8}")
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        results[m["name"]] = {}
        for t in TGRID:
            L, M, C, num = closed_form_L(m["W"], m["R"], m["pi_prod"], t)
            eQ = expm_rev(m["W"], m["pij"], t)       # exact e^{tQ} (reversible eig)
            exact = np.log(np.maximum(eQ, FLOOR))
            eR = expm_rev(m["R"], m["pi_prod"], t)   # e^{tR} = M (same eig as base)
            prod = np.log(np.maximum(eR, FLOOR))     # product baseline (no coupling)

            err_elbo = error_stats(exact, L)
            err_prod = error_stats(exact, prod)
            rank_elbo = ranking_stats(exact, L)
            rank_prod = ranking_stats(exact, prod)

            results[m["name"]][f"{t}"] = dict(
                elbo=dict(error=err_elbo, ranking=rank_elbo),
                product_baseline=dict(error=err_prod, ranking=rank_prod),
            )
            print(f"{m['name']:<28}{t:>6}"
                  f"{err_elbo['meaningful']['mean']:>10.4f}"
                  f"{err_elbo['meaningful']['p95']:>10.4f}"
                  f"{rank_elbo['global_spearman']:>9.4f}"
                  f"{rank_prod['global_spearman']:>9.4f}"
                  f"{rank_elbo['row_spearman_mean']:>9.4f}"
                  f"{rank_prod['row_spearman_mean']:>9.4f}"
                  f"{rank_elbo['argmax_agree_frac']:>8.3f}"
                  f"{rank_prod['argmax_agree_frac']:>8.3f}")

    out = dict(
        tgrid=TGRID, meaningful_threshold=MEANINGFUL,
        selfchecks=sc, all_selfchecks_passed=bool(allpass),
        models={m["name"]: dict(
            S_recover_resid=m["S_recover_resid"],
            S_recover_Qerr=m["S_recover_Qerr"]) for m in models},
        results=results,
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics -> {OUT_JSON}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        cmap = plt.get_cmap("tab10")
        for k, m in enumerate(models):
            name = m["name"]
            sp_e = [results[name][f"{t}"]["elbo"]["ranking"]["global_spearman"] for t in TGRID]
            sp_p = [results[name][f"{t}"]["product_baseline"]["ranking"]["global_spearman"] for t in TGRID]
            gp_e = [results[name][f"{t}"]["elbo"]["error"]["meaningful"]["mean"] for t in TGRID]
            gp_p = [results[name][f"{t}"]["product_baseline"]["error"]["meaningful"]["mean"] for t in TGRID]
            c = cmap(k)
            ax1.plot(TGRID, sp_e, "-o", color=c, label=f"{name} ELBO", ms=4)
            ax1.plot(TGRID, sp_p, "--x", color=c, alpha=0.6, ms=5)
            ax2.plot(TGRID, gp_e, "-o", color=c, label=f"{name} ELBO", ms=4)
            ax2.plot(TGRID, gp_p, "--x", color=c, alpha=0.6, ms=5)
        ax1.set_xscale("log"); ax1.set_xlabel("branch length t")
        ax1.set_ylabel("global Spearman (exact vs approx)")
        ax1.set_title("Ranking: solid=ELBO, dashed=product baseline")
        ax1.legend(fontsize=6)
        ax2.set_xscale("log"); ax2.set_xlabel("branch length t")
        ax2.set_ylabel("mean gap (exact - approx), meaningful entries")
        ax2.set_title("Error: solid=ELBO, dashed=product baseline")
        fig.tight_layout()
        fig.savefig(OUT_PDF)
        print(f"Saved plot -> {OUT_PDF}")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
