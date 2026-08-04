"""Cohn CTBN mean-field  vs  our closed-form path ELBO  vs  exact matrix exponential,
on the 400-state two-amino-acid (2-site Potts) case.

All three methods target ONE shared generator: a Cohn/Glauber 2-site Potts CTBN
(evolsnake `ctbn.q_joint`, N=20, K=2 -> 400 states) whose stationary is set exactly
to a real fitted pair distribution pij and whose exchangeabilities are the fitted S.
So it is the SAME stationary and S as our square-root-Metropolis pair model -- only
the acceptance kernel differs (Glauber vs Metropolis). This lets Cohn's mean-field
run natively (its q_bar/q_tilde assume the Glauber exp(h+2J) form) while our generic
closed-form path ELBO bounds the same exact log-transition.

  exact   :  log[e^{tW}]_xy               (reversible eig; machine precision)
  ours    :  closed-form Girsanov path ELBO L_xy against the independent product
             bridge R (Cohn with J=0); whole 400x400 at once, no ODE
  cohn    :  ctbn.ctbn_variational_log_cond (round-robin mean-field, diffrax ODEs);
             one endpoint pair per solve

Endpoint pairs are drawn per branch length in three regimes -- single-site
substitution, double substitution (both sites change; maximally stresses the
product bridge), and typical (y ~ e^{tW}[x]) -- since the interesting coupling
signal lives off the diagonal.

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
     PYTHONPATH=src:experiments:src/tkfdp/cohn_ctbn python3 experiments/cohn_vs_elbo_pair.py
"""
import os, sys, time, json
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")          # never touch the GPU
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments"))
sys.path.insert(0, os.path.join(REPO, "src/tkfdp/cohn_ctbn"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import fit_pair_models as FP
import elbo_vs_expm as EV                 # closed_form_L, expm_rev, FLOOR
import ctbn                               # evolsnake Cohn implementation

NA, NS = FP.NA, FP.NS
FLOOR = EV.FLOOR
TGRID = list(EV.TGRID)   # match elbo_vs_expm x-grid
N_PER_REGIME = 4                          # endpoint pairs per (model, t, regime)
COHN_MIN_INC = 1e-4
COHN_MAX_UPDATES = 128                     # outer mean-field sweeps (power of 2; converges early)
OUT_JSON = os.path.join(REPO, "results", "pair_models", "cohn_vs_elbo.json")
OUT_PDF = os.path.join(REPO, "experiments", "figures", "cohn_vs_elbo.pdf")

C2 = jnp.array([[0., 1.], [1., 0.]])
SEQ_MASK, NBR_IDX, NBR_MASK, *_ = ctbn.get_Markov_blankets(C2)


def sidx(a, b):
    """canonical little-endian pair index used by q_joint / expm (idx = a + b*NA)."""
    return a + b * NA


def build_glauber(S, pij):
    """Cohn/Glauber 2-site generator W with stationary == pij, plus the independent
    product bridge R (same S, J=0). Returns W, R, pi_prod, and Cohn params for W."""
    pij = pij / pij.sum()
    m1 = pij.sum(1)
    h = np.log(np.maximum(m1, 1e-300))
    J = 0.5 * (np.log(np.maximum(pij, 1e-300)) - h[:, None] - h[None, :])
    Sc = S.copy(); np.fill_diagonal(Sc, 0.0)
    params = {"S": jnp.asarray(Sc), "J": jnp.asarray(J), "h": jnp.asarray(h)}
    params0 = {"S": jnp.asarray(Sc), "J": jnp.zeros((NA, NA)), "h": jnp.asarray(h)}
    W = np.asarray(ctbn.q_joint(NBR_IDX, NBR_MASK, ctbn.normalise_ctbn_params(params)))
    R = np.asarray(ctbn.q_joint(NBR_IDX, NBR_MASK, ctbn.normalise_ctbn_params(params0)))
    pi_prod = np.outer(m1, m1).reshape(NS)
    return W, R, pi_prod, params, pij


def selfcheck_model(W, R, pij, pi_prod):
    piv = pij.reshape(NS)
    stat_err = float(np.max(np.abs(np.asarray(ctbn.exact_eqm(jnp.asarray(W))) - piv)))
    revW = float(np.max(np.abs(piv[:, None] * W - (piv[:, None] * W).T)))
    revR = float(np.max(np.abs(pi_prod[:, None] * R - (pi_prod[:, None] * R).T)))
    off = ~np.eye(NS, dtype=bool)
    support_ok = bool(not (off & (R == 0.0) & (W != 0.0)).any())
    return dict(stationary_err=stat_err, revW=revW, revR=revR, support_ok=support_ok)


def sample_pairs(eW, pij, t, rng):
    """Return list of (x, y, regime) canonical indices for one (model, t).

    Destination weights use M[c,d] = P(dest site0=c, site1=d) = eW[x, c + d*NA],
    obtained by column-major (order='F') reshape so the axes are (site0, site1)
    consistent with sidx -- a C-order reshape would transpose the two sites.
    """
    piv = pij.reshape(NS)
    pairs = []
    # single-site substitution: change site 0 (a->c), keep site 1 = b
    for _ in range(N_PER_REGIME):
        a, b = int(rng.integers(NA)), int(rng.integers(NA))
        x = sidx(a, b)
        M = eW[x].reshape(NA, NA, order="F")     # M[c,d]
        w = M[:, b].copy(); w[a] = 0.0           # d=b fixed, c!=a
        if w.sum() <= 0:
            continue
        c = int(rng.choice(NA, p=w / w.sum()))
        pairs.append((x, sidx(c, b), "single"))
    # double substitution: both sites change (a->c, b->d), c!=a and d!=b
    for _ in range(N_PER_REGIME):
        a, b = int(rng.integers(NA)), int(rng.integers(NA))
        x = sidx(a, b)
        M = eW[x].reshape(NA, NA, order="F").copy()
        M[a, :] = 0.0; M[:, b] = 0.0             # c!=a and d!=b
        if M.sum() <= 0:
            continue
        flat = M.reshape(-1) / M.sum()           # C-order: k = c*NA + d
        k = int(rng.choice(NA * NA, p=flat)); c, d = k // NA, k % NA
        pairs.append((x, sidx(c, d), "double"))
    # typical: x ~ pij, y ~ e^{tW}[x]
    for _ in range(N_PER_REGIME):
        x = int(rng.choice(NS, p=piv))
        row = np.maximum(eW[x], 0.0)
        y = int(rng.choice(NS, p=row / row.sum()))
        pairs.append((x, y, "typical"))
    return pairs


def run_model(name, S, pij_raw, mi, results):
    W, R, pi_prod, params, pij = build_glauber(S, pij_raw)
    chk = selfcheck_model(W, R, pij, pi_prod)
    print(f"\n### {name}  MI(pi)={mi:.3f} ###")
    print(f"  self-check: stat_err={chk['stationary_err']:.1e} revW={chk['revW']:.1e} "
          f"revR={chk['revR']:.1e} support_ok={chk['support_ok']}")
    results[name] = dict(mi=float(mi), selfcheck=chk, t={})
    rng = np.random.default_rng(0)
    for t in TGRID:
        eW = EV.expm_rev(W, pij.reshape(NS), t)
        exact = np.log(np.maximum(eW, FLOOR))
        t0 = time.time()
        L, _, _, _ = EV.closed_form_L(W, R, pi_prod, t)          # whole 400x400
        t_ours_full = time.time() - t0
        # our bound validity across the whole matrix (machine-precision strict bound)
        ours_min_slack = float((exact - L).min())
        pairs = sample_pairs(eW, pij, t, rng)
        rows = []
        cohn_times = []
        n_failed = 0
        for (x, y, regime) in pairs:
            a, b = x % NA, x // NA
            c, d = y % NA, y // NA
            tc = time.time()
            try:
                le, _ = ctbn.ctbn_variational_log_cond(
                    jax.random.PRNGKey(0), jnp.array([a, b]), jnp.array([c, d]),
                    SEQ_MASK, NBR_IDX, NBR_MASK, params, float(t),
                    min_inc=COHN_MIN_INC, max_updates=COHN_MAX_UPDATES)
                le = float(le)
                cohn_times.append(time.time() - tc)
            except Exception as e:               # diffrax non-convergence -> record, don't crash
                le = float("nan"); n_failed += 1
                if n_failed <= 2:
                    print(f"    [cohn non-converge {regime} ({a},{b})->({c},{d}) t={t}: "
                          f"{type(e).__name__}]")
            rows.append(dict(x=x, y=y, regime=regime,
                             exact=float(exact[x, y]), ours=float(L[x, y]), cohn=le))
        ex = np.array([r["exact"] for r in rows])
        ou = np.array([r["ours"] for r in rows])
        co = np.array([r["cohn"] for r in rows])
        ok = ~np.isnan(co)                        # Cohn pairs that converged
        gap_ours = ex - ou
        gap_cohn = ex[ok] - co[ok]
        sp_ours = float(spearmanr(ex, ou).statistic)
        sp_cohn = float(spearmanr(ex[ok], co[ok]).statistic) if ok.sum() > 2 else float("nan")
        t_cohn_med = float(np.median(cohn_times)) if cohn_times else float("nan")
        g_cohn = (dict(mean=float(gap_cohn.mean()), max=float(gap_cohn.max()),
                       min=float(gap_cohn.min())) if gap_cohn.size
                  else dict(mean=float("nan"), max=float("nan"), min=float("nan")))
        results[name]["t"][f"{t}"] = dict(
            t_ours_fullmatrix_s=t_ours_full, t_cohn_perpair_median_s=t_cohn_med,
            ours_fullmatrix_min_slack=ours_min_slack,
            gap_ours=dict(mean=float(gap_ours.mean()), max=float(gap_ours.max()),
                          min=float(gap_ours.min())),
            gap_cohn=g_cohn,
            spearman_ours=sp_ours, spearman_cohn=sp_cohn,
            n_pairs=len(rows), n_cohn_failed=int(n_failed), rows=rows,
        )
        print(f"  t={t:<4}  gap_ours(mean/max)={gap_ours.mean():.4f}/{gap_ours.max():.4f}"
              f"  gap_cohn(mean/max)={g_cohn['mean']:.4f}/{g_cohn['max']:.4f}"
              f"  sp_ours={sp_ours:.3f} sp_cohn={sp_cohn:.3f}  fail={n_failed}/{len(rows)}"
              f"  | ours_full={t_ours_full*1e3:.0f}ms cohn/pair={t_cohn_med:.1f}s"
              f"  ours_matrix_slack={ours_min_slack:.1e}", flush=True)
        jax.clear_caches()   # free compiled diffrax executables before the next t recompiles


def main():
    results = {}
    # 3 real mixture components spanning a coupling gradient + a strong (J x2) variant
    zm = np.load(os.path.join(REPO, "results/mixture_component_char/components_K8.npz"),
                 allow_pickle=True)
    S_mix = np.asarray(zm["S"], float)
    pis = np.asarray(zm["pis"], float)
    mi = np.asarray(zm["mi_pi"], float)
    order = np.argsort(mi)                               # ascending
    picks = [order[len(order)//2], order[-1]]            # mid, strongest (+STRONGx2 below)
    for c in picks:
        run_model(f"mixture_K8_c{c}_mi{mi[c]:.3f}", S_mix, pis[c].reshape(NA, NA),
                  mi[c], results)
    # strong-coupling stress: sharpen the strongest component's coupling (J x2 on log-pij)
    cS = int(order[-1])
    pij_s = pis[cS].reshape(NA, NA); pij_s = pij_s / pij_s.sum()
    m1 = pij_s.sum(1); m2 = pij_s.sum(0)
    logMI = np.log(np.maximum(pij_s, 1e-300)) - np.log(m1)[:, None] - np.log(m2)[None, :]
    pij_strong = np.outer(m1, m2) * np.exp(2.0 * logMI); pij_strong /= pij_strong.sum()
    mi_strong = float((pij_strong * (np.log(np.maximum(pij_strong, 1e-300))
                       - np.log(pij_strong.sum(1))[:, None]
                       - np.log(pij_strong.sum(0))[None, :])).sum())
    run_model(f"mixture_K8_c{cS}_STRONGx2_mi{mi_strong:.3f}", S_mix, pij_strong,
              mi_strong, results)

    out = dict(tgrid=TGRID, n_per_regime=N_PER_REGIME, cohn_min_inc=COHN_MIN_INC,
               na=NA, ns=NS, results=results)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
