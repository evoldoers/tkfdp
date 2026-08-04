"""Pass 1 of the two-pass Cohn evaluation: the DENSE, whole-matrix / batched part.

For each (mixture model, branch length t) it computes, on a shared set of n1
importance-sampled endpoint pairs (x ~ pij, y ~ e^{tW}[x]):
  - exact            log[e^{tW}]_xy      (whole 400x400 matrix, one shot)
  - closed-form path ELBO gap            (whole matrix)
  - midpoint-node ELBO gap               (whole matrix)
  - Cohn mean-field (no back-reaction psi) gap  (batched, chunked to fit GPU)

It writes the per-pair arrays AND the sampled joint-state indices x, y, so the
parallel Pass 2 (cohn_pass2_parallel.py) can run the expensive full-psi diffrax
Cohn on the very same pairs.

This stage is cheap (~seconds/cell of real compute) and secures the smooth
curves fast, before the multi-hour full-psi pass.

Run: CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
     PYTHONPATH=src:experiments:src/tkfdp/cohn_ctbn python3 experiments/cohn_pass1.py [--smoke]
"""
import os, sys, json
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import jax, jax.numpy as jnp
import elbo_vs_expm as EV, fit_pair_models as FP, cohn_vs_elbo_pair as CV
import cohn_batch as CB
from midpoint_elbo import midpoint_elbo_path_edges

NA, NS = FP.NA, FP.NS
TGRID = EV.TGRID
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "pair_models", "cohn_pass1.json")
NG, NITER = 192, 12          # batch grid / mean-field sweeps (validated)
CHUNK = 1000                 # no-psi batch chunk (B=1000 ~ 37 GB on the 49 GB GPU)


def save(results, tg):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(tgrid=list(tg), results=results, n_grid=NG, n_iter=NITER), f)
    os.replace(tmp, OUT)


def sample_pairs(eW, pij, rng, n):
    """x ~ pij, y ~ e^{tW}[x]; return joint-state indices (x, y), each (n,)."""
    piv = np.maximum(pij.reshape(NS), 0); piv /= piv.sum()
    x = rng.choice(NS, size=n, p=piv)
    y = np.empty(n, dtype=int)
    for s in np.unique(x):
        m = x == s
        row = np.maximum(eW[s], 0); row = row / row.sum()
        y[m] = rng.choice(NS, size=int(m.sum()), p=row)
    return x, y


def cohn_nopsi_chunked(Sj, hj, Jj, x, y, t):
    """Batched no-psi Cohn F over all pairs, chunked to fit GPU memory."""
    F = np.empty(len(x))
    for lo in range(0, len(x), CHUNK):
        hi = min(lo + CHUNK, len(x))
        xs = jnp.asarray(np.stack([x[lo:hi] % NA, x[lo:hi] // NA], 1))
        ys = jnp.asarray(np.stack([y[lo:hi] % NA, y[lo:hi] // NA], 1))
        Fc, _ = CB.cohn_batch_elbo(Sj, hj, Jj, xs, ys, float(t), 0.5, NG, NITER, False)
        F[lo:hi] = np.asarray(Fc)
    return F


def run_model(name, S, pij_raw, mi, n1, tg, results):
    W, R, pip, params, pij = CV.build_glauber(S, pij_raw)
    pijv = pij.reshape(NS)
    Sj, hj, Jj = jnp.asarray(params["S"]), jnp.asarray(params["h"]), jnp.asarray(params["J"])
    precomp = EV.closed_form_L_precompute(W, R, pip)
    rng = np.random.default_rng(0)
    results.setdefault(name, dict(mi=float(mi), t={}))
    print(f"\n### {name}  MI={mi:.3f} ###", flush=True)
    import time
    for t in tg:
        if f"{t}" in results[name]["t"]:
            print(f"  t={t:<5} (cached, skip)", flush=True)
            sample_pairs(EV.expm_rev(W, pijv, t), pij, rng, n1)  # keep rng aligned
            continue
        t0 = time.time()
        eW = EV.expm_rev(W, pijv, t); exact = np.log(np.maximum(eW, EV.FLOOR))
        Lp = EV.closed_form_L_from_precompute(precomp, t)[0]
        Lm = midpoint_elbo_path_edges(W, R, pip, t, precomp=precomp)
        x, y = sample_pairs(eW, pij, rng, n1)
        ex = exact[x, y]
        gap_path = ex - Lp[x, y]
        gap_mid = ex - Lm[x, y]
        Fnp = cohn_nopsi_chunked(Sj, hj, Jj, x, y, t)
        gap_cohn_nopsi = ex - Fnp
        results[name]["t"][f"{t}"] = dict(
            n1=int(n1),
            x=x.astype(int).tolist(), y=y.astype(int).tolist(),
            ex=ex.round(6).tolist(),
            gap_path=gap_path.round(6).tolist(),
            gap_mid=gap_mid.round(6).tolist(),
            gap_cohn_nopsi=gap_cohn_nopsi.round(6).tolist(),
            gap_path_mean=float(gap_path.mean()),
            gap_mid_mean=float(gap_mid.mean()),
            gap_cohn_nopsi_mean=float(gap_cohn_nopsi.mean()),
            gap_cohn_nopsi_min=float(gap_cohn_nopsi.min()),
        )
        print(f"  t={t:<5} path={gap_path.mean():.4f} mid={gap_mid.mean():.4f} "
              f"cohn_nopsi={gap_cohn_nopsi.mean():.4f}  (n1={n1}, {time.time()-t0:.0f}s)", flush=True)
        save(results, tg)


def main():
    smoke = "--smoke" in sys.argv
    n1 = 200 if smoke else 4000
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
            print(f"Resuming Pass-1 from {OUT}", flush=True)
        except Exception:
            results = {}
    for c, pij_c, mval in picks:
        run_model(f"mixture_K8_c{c}_mi{mval:.3f}", S_mix, pij_c, mval, n1, tg, results)
        save(results, tg)
    print(f"\nPass-1 saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
