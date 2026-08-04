"""Benchmark the midpoint-node variational ELBO against our closed-form path ELBO
(and the product baseline) vs exact matrix exponential, the same way as
`elbo_vs_expm.py`: whole 400x400 transition matrix, square-root-Metropolis pair
models, a grid of branch lengths.

  exact    : log[e^{tW}]_xy           (reversible eig, machine precision)
  path     : closed-form Girsanov path ELBO against the product bridge R (ours)
  midpoint : mean-field q(z_i)q(z_j) on the branch midpoint, exact coupled halves
  product  : log[e^{tR}]_xy           (independent baseline, no coupling)

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
     PYTHONPATH=src:experiments python3 experiments/midpoint_vs_path_elbo.py
"""
import os, json, time
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import elbo_vs_expm as EV
import fit_pair_models as FP
from midpoint_elbo import midpoint_elbo_matrix, midpoint_elbo_path_edges

NA, NS = FP.NA, FP.NS
FLOOR = EV.FLOOR
TGRID = EV.TGRID
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REPO, "results", "pair_models", "midpoint_vs_path_elbo.json")


def main():
    models = EV.load_models()
    results = {}
    # gap_path   : full-branch closed-form path ELBO (product eig only)
    # gap_mpath  : midpoint node with PATH-ELBO half-edges  (product eig only) -- FAIR test
    # gap_mexact : midpoint node with EXACT coupled half-edges (needs coupled expm) -- diagnostic
    hdr = (f"{'model':<24}{'t':>6}{'gap_path':>9}{'gap_mpath':>10}{'gap_mexact':>11}"
           f"{'gap_prod':>9}{'sp_path':>8}{'sp_mpath':>9}{'slack_mp':>10}{'ms_path':>8}{'ms_mp':>7}")
    print(hdr); print("-" * len(hdr))
    # warm up the JIT so reported timings are steady-state (not first-call compile)
    m0 = models[0]
    midpoint_elbo_path_edges(m0["W"], m0["R"], m0["pi_prod"], 1.0)
    midpoint_elbo_matrix(m0["W"], m0["pij"].reshape(NS), 1.0)
    for m in models:
        name = m["name"]; W = m["W"]; R = m["R"]; pij = m["pij"].reshape(NS); pip = m["pi_prod"]
        results[name] = {}
        for t in TGRID:
            exact = np.log(np.maximum(EV.expm_rev(W, pij, t), FLOOR))
            t0 = time.time(); Lp, _, _, _ = EV.closed_form_L(W, R, pip, t); t_path = time.time() - t0
            t0 = time.time(); Lmp = midpoint_elbo_path_edges(W, R, pip, t); t_mpath = time.time() - t0
            t0 = time.time(); Lme = midpoint_elbo_matrix(W, pij, t); t_mexact = time.time() - t0
            prod = np.log(np.maximum(EV.expm_rev(R, pip, t), FLOOR))

            err_p = EV.error_stats(exact, Lp); err_mp = EV.error_stats(exact, Lmp)
            err_me = EV.error_stats(exact, Lme); err_d = EV.error_stats(exact, prod)
            rk_p = EV.ranking_stats(exact, Lp); rk_mp = EV.ranking_stats(exact, Lmp)
            slack_p = float((exact - Lp).min()); slack_mp = float((exact - Lmp).min())
            slack_me = float((exact - Lme).min())

            results[name][f"{t}"] = dict(
                gap_path=err_p, gap_mpath=err_mp, gap_mexact=err_me, gap_prod=err_d,
                ranking_path=rk_p, ranking_mpath=rk_mp,
                min_slack_path=slack_p, min_slack_mpath=slack_mp, min_slack_mexact=slack_me,
                t_path_s=t_path, t_mpath_s=t_mpath, t_mexact_s=t_mexact)
            print(f"{name:<24}{t:>6}"
                  f"{err_p['meaningful']['mean']:>9.4f}{err_mp['meaningful']['mean']:>10.4f}"
                  f"{err_me['meaningful']['mean']:>11.4f}{err_d['meaningful']['mean']:>9.4f}"
                  f"{rk_p['global_spearman']:>8.4f}{rk_mp['global_spearman']:>9.4f}"
                  f"{slack_mp:>10.1e}{t_path*1e3:>8.0f}{t_mpath*1e3:>7.0f}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(dict(tgrid=TGRID, results=results), f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
