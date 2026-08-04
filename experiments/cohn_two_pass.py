"""Two-pass Cohn evaluation for smooth curves + %-non-converged.

Pass 1 (fast, batched, GPU): no-psi Cohn ELBO (experiments/cohn_batch.py) over N1
endpoint pairs per (model, branch length) -> a smooth curve; plus exact, path ELBO
and midpoint ELBO for the same pairs (all whole-matrix / batched). The no-psi Cohn
is a valid, documented Cohn variant that never fails numerically.

Pass 2 (careful, diffrax, subset): the FULL-psi Cohn (src/tkfdp/cohn_ctbn/ctbn.py)
on an N2-pair subset, with a max_steps RETRY (the 'second pass' for stiff pairs) --
records what fraction fail to converge even after the retry, and the full-psi
values where it does converge (to overlay vs no-psi).

Run: CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
     PYTHONPATH=src:experiments:src/tkfdp/cohn_ctbn python3 experiments/cohn_two_pass.py [--smoke]
"""
import os, sys, json, time
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import jax, jax.numpy as jnp
import elbo_vs_expm as EV, fit_pair_models as FP, cohn_vs_elbo_pair as CV, ctbn
import plot_bounds_vs_cohn as P
import cohn_batch as CB
from midpoint_elbo import midpoint_elbo_path_edges

NA, NS = FP.NA, FP.NS
TGRID = EV.TGRID
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "pair_models", "cohn_two_pass.json")
NG, NITER = 192, 12                       # batch grid / mean-field sweeps
MAXSTEPS_1, MAXSTEPS_2 = 1 << 14, 1 << 16  # diffrax pass-2 cap, then retry cap


def sample_pairs(eW, pij, rng, n):
    """x ~ pij, y ~ e^{tW}[x]; return (x, y) canonical indices (n,)."""
    piv = np.maximum(pij.reshape(NS), 0); piv /= piv.sum()
    x = rng.choice(NS, size=n, p=piv)
    y = np.empty(n, dtype=int)
    for s in np.unique(x):
        m = x == s
        row = np.maximum(eW[s], 0); row = row / row.sum()
        y[m] = rng.choice(NS, size=int(m.sum()), p=row)
    return x, y


def diffrax_fullpsi(params, x, y, t):
    """Full-psi diffrax Cohn for one pair at a fixed generous solver budget.
    Returns (value or nan, status), status in {ok, fail}.

    No max_steps retry: the smoke pilot showed a 4x step budget (65536 vs 16384)
    rescues ~0% of failures -- they are the proven inherent endpoint singularity
    of the Cohn F-integral (see analysis/cohn_ctbn_vs_closedform_elbo.md), not an
    under-resolution artifact -- so a retry only multiplies the cost of every
    failing pair for zero gain. A 'fail' here means: did not converge within a
    16384-step diffrax budget."""
    sm, ni, nm = CV.SEQ_MASK, CV.NBR_IDX, CV.NBR_MASK
    a, b, c, d = x % NA, x // NA, y % NA, y // NA
    ctbn.CTBN_MAX_STEPS = MAXSTEPS_1
    try:
        le, _ = ctbn.ctbn_variational_log_cond(
            jax.random.PRNGKey(0), jnp.array([a, b]), jnp.array([c, d]),
            sm, ni, nm, params, float(t), min_inc=1e-5, max_updates=128,
            rtol=1e-4, atol=1e-7)
        return float(le), "ok"
    except Exception:
        return float("nan"), "fail"


def save(results, tg):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(tgrid=list(tg), results=results), f, indent=2)
    os.replace(tmp, OUT)  # atomic


def run_model(name, S, pij_raw, mi, n1, n2, tg, results):
    W, R, pip, params, pij = None, None, None, None, None
    W, R, pip, params, pij = CV.build_glauber(S, pij_raw)
    pijv = pij.reshape(NS)
    Sj, hj, Jj = jnp.asarray(params["S"]), jnp.asarray(params["h"]), jnp.asarray(params["J"])
    precomp = EV.closed_form_L_precompute(W, R, pip)
    rng = np.random.default_rng(0)
    results.setdefault(name, dict(mi=float(mi), t={}))
    print(f"\n### {name}  MI={mi:.3f} ###", flush=True)
    for t in tg:
        if f"{t}" in results[name]["t"]:
            print(f"  t={t:<5} (cached, skip)", flush=True)
            # keep rng stream aligned with a fresh run so cached+new cells are consistent
            sample_pairs(EV.expm_rev(W, pijv, t), pij, rng, n1)
            continue
        eW = EV.expm_rev(W, pijv, t); exact = np.log(np.maximum(eW, EV.FLOOR))
        Lp = EV.closed_form_L_from_precompute(precomp, t)[0]
        Lm = midpoint_elbo_path_edges(W, R, pip, t, precomp=precomp)
        x, y = sample_pairs(eW, pij, rng, n1)
        ex = exact[x, y]
        gap_path = ex - Lp[x, y]
        gap_mid = ex - Lm[x, y]
        # Pass 1: batched no-psi Cohn
        xs = jnp.asarray(np.stack([x % NA, x // NA], 1))
        ys = jnp.asarray(np.stack([y % NA, y // NA], 1))
        Fnp, _ = CB.cohn_batch_elbo(Sj, hj, Jj, xs, ys, float(t), 0.5, NG, NITER, False)
        Fnp = np.asarray(Fnp); gap_cohn_nopsi = ex - Fnp
        # Pass 2: diffrax full-psi on subset
        idx = rng.choice(n1, size=min(n2, n1), replace=False)
        fp_gap, fp_ex, statuses = [], [], []
        t2 = time.time()
        for j in idx:
            val, st = diffrax_fullpsi(params, int(x[j]), int(y[j]), t)
            statuses.append(st)
            if not np.isnan(val):
                fp_ex.append(float(exact[x[j], y[j]]))
                fp_gap.append(float(exact[x[j], y[j]] - val))
        nfail = sum(1 for s in statuses if s == "fail")
        nretry = sum(1 for s in statuses if s == "ok_retry")
        results[name]["t"][f"{t}"] = dict(
            n1=int(n1), n2=int(len(idx)),
            gap_path_mean=float(gap_path.mean()),
            gap_mid_mean=float(gap_mid.mean()),
            gap_cohn_nopsi_mean=float(gap_cohn_nopsi.mean()),
            gap_cohn_nopsi_min=float(gap_cohn_nopsi.min()),
            gap_cohn_fullpsi_mean=(float(np.mean(fp_gap)) if fp_gap else float("nan")),
            frac_fail=nfail / len(idx), frac_retry=nretry / len(idx),
            # per-pair arrays (single source for BOTH the gap and the ranking panels,
            # so every panel shares the same t-grid and the same sampled pairs).
            ex=ex.round(6).tolist(),
            gap_path=gap_path.round(6).tolist(),
            gap_mid=gap_mid.round(6).tolist(),
            gap_cohn_nopsi=gap_cohn_nopsi.round(6).tolist(),
            fullpsi_ex=fp_ex,               # exact log-prob on converged full-psi pairs
            gap_cohn_fullpsi=fp_gap,        # matching gaps
        )
        print(f"  t={t:<5} path={gap_path.mean():.4f} mid={gap_mid.mean():.4f} "
              f"cohn_nopsi={gap_cohn_nopsi.mean():.4f} cohn_fullpsi={(np.mean(fp_gap) if fp_gap else float('nan')):.4f} "
              f"| fail={nfail}/{len(idx)} retry={nretry} ({time.time()-t2:.0f}s)", flush=True)
        save(results, tg)  # incremental: survive interruption of a multi-hour run


def main():
    smoke = "--smoke" in sys.argv
    n1, n2 = (100, 4) if smoke else (1000, 64)
    tg = TGRID[:2] if smoke else TGRID
    zm = np.load(os.path.join(REPO, "results/mixture_component_char/components_K8.npz"),
                 allow_pickle=True)
    S_mix = np.asarray(zm["S"], float); pis = np.asarray(zm["pis"], float)
    mi = np.asarray(zm["mi_pi"], float); order = np.argsort(mi)
    picks = [(order[len(order) // 2], pis[order[len(order) // 2]].reshape(NA, NA), mi[order[len(order) // 2]]),
             (order[-1], pis[order[-1]].reshape(NA, NA), mi[order[-1]])]
    results = {}
    if os.path.exists(OUT):
        try:
            results = json.load(open(OUT)).get("results", {})
            done = sum(len(v.get("t", {})) for v in results.values())
            print(f"Resuming from {OUT} ({done} cells already done)", flush=True)
        except Exception:
            results = {}
    for c, pij_c, mval in picks:
        run_model(f"mixture_K8_c{c}_mi{mval:.3f}", S_mix, pij_c, mval, n1, n2, tg, results)
        save(results, tg)
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
