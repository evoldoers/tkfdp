"""Re-evaluate Cohn (2010) mean-field at a TIGHT ODE tolerance (rtol=1e-8) so it is
a valid bound, for a fair tightness comparison against our closed-form path ELBO
and the midpoint-node ELBO. Reuses the sampled endpoint pairs from
results/pair_models/cohn_vs_elbo.json (default-tolerance run) on the same
Cohn/Glauber generators; path and midpoint are recomputed at those pairs.

Records, per (model, t, pair): exact, path, midpoint, cohn_loose (rtol=1e-3, from
the original json), cohn_tight (rtol=1e-8, here; NaN if the diffrax solve fails to
converge). Saves results/pair_models/cohn_tight_tol.json.

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
     PYTHONPATH=src:experiments:src/tkfdp/cohn_ctbn python3 experiments/cohn_tight_tol.py
"""
import os, json
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import jax
import jax.numpy as jnp

import elbo_vs_expm as EV
import fit_pair_models as FP
import cohn_vs_elbo_pair as CV
import ctbn
import plot_bounds_vs_cohn as P
from midpoint_elbo import midpoint_elbo_path_edges

NA, NS = FP.NA, FP.NS
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_JSON = os.path.join(REPO, "results", "pair_models", "cohn_vs_elbo.json")
OUT_JSON = os.path.join(REPO, "results", "pair_models", "cohn_tight_tol.json")
RTOL, ATOL, MAXUP = 1e-8, 1e-11, 128


def main():
    d = json.load(open(IN_JSON))
    tg = [float(t) for t in d["tgrid"]]
    zm = np.load(os.path.join(REPO, "results/mixture_component_char/components_K8.npz"),
                 allow_pickle=True)
    S_mix = np.asarray(zm["S"], float); pis = np.asarray(zm["pis"], float)
    sm, ni, nm = CV.SEQ_MASK, CV.NBR_IDX, CV.NBR_MASK
    out = {"tgrid": tg, "rtol": RTOL, "atol": ATOL, "results": {}}

    for name in d["results"]:
        W, R, pip, pijv = P.build_model(name, pis, S_mix)
        c = int(name.split("_c")[1].split("_")[0])
        # Cohn params for this Glauber generator (STRONGx2 uses the sharpened pij)
        if "STRONGx2" in name:
            pj = pis[c].reshape(NA, NA); pj = pj / pj.sum()
            m1 = pj.sum(1); m2 = pj.sum(0)
            lm = np.log(np.maximum(pj, 1e-300)) - np.log(m1)[:, None] - np.log(m2)[None, :]
            pj = np.outer(m1, m2) * np.exp(2.0 * lm)
        else:
            pj = pis[c].reshape(NA, NA)
        _, _, _, params, _ = CV.build_glauber(S_mix, pj)
        out["results"][name] = {"mi": d["results"][name]["mi"], "t": {}}
        for t in tg:
            exact = np.log(np.maximum(EV.expm_rev(W, pijv, t), EV.FLOOR))
            Lp, _, _, _ = EV.closed_form_L(W, R, pip, t)
            Lm = midpoint_elbo_path_edges(W, R, pip, t)
            recs = []
            for r in d["results"][name]["t"][f"{t}"]["rows"]:
                x, y = r["x"], r["y"]; a, b = x % NA, x // NA; c2, e = y % NA, y // NA
                try:
                    le, _ = ctbn.ctbn_variational_log_cond(
                        jax.random.PRNGKey(0), jnp.array([a, b]), jnp.array([c2, e]),
                        sm, ni, nm, params, float(t), min_inc=1e-6, max_updates=MAXUP,
                        rtol=RTOL, atol=ATOL)
                    ct = float(le)
                except Exception:
                    ct = float("nan")
                recs.append(dict(x=x, y=y, regime=r["regime"],
                                 exact=float(exact[x, y]),
                                 path=float(Lp[x, y]), midpoint=float(Lm[x, y]),
                                 cohn_loose=r["cohn"], cohn_tight=ct))
            out["results"][name]["t"][f"{t}"] = recs
            nf = sum(1 for rr in recs if np.isnan(rr["cohn_tight"]))
            okc = [rr["exact"] - rr["cohn_tight"] for rr in recs if not np.isnan(rr["cohn_tight"])]
            gp = np.mean([rr["exact"] - rr["path"] for rr in recs])
            gm = np.mean([rr["exact"] - rr["midpoint"] for rr in recs])
            gc = np.mean(okc) if okc else float("nan")
            print(f"{name:30s} t={t:<4} path={gp:.4f} midpt={gm:.4f} "
                  f"cohn_tight={gc:+.4f} fail={nf}/{len(recs)}", flush=True)
            jax.clear_caches()   # free compiled diffrax executables before next t recompiles

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
