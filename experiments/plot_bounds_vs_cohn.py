#!/usr/bin/env python3
"""Unified accuracy plot on the Cohn/Glauber generator: exact vs three variational
lower bounds -- our closed-form path ELBO, the new midpoint-node ELBO, and Cohn
(2010) mean-field. One panel per coupling strength (no clutter). Cohn's per-pair
values are read from results/pair_models/cohn_vs_elbo.json (expensive ODE solves,
already run); path and midpoint are evaluated fresh at the SAME sampled endpoint
pairs so all three are directly comparable.

Run: CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
     PYTHONPATH=src:experiments python3 experiments/plot_bounds_vs_cohn.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import elbo_vs_expm as EV
import fit_pair_models as FP
import cohn_vs_elbo_pair as CV                 # build_glauber
from midpoint_elbo import midpoint_elbo_path_edges

NA, NS = FP.NA, FP.NS
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COHN_JSON = os.path.join(REPO, "results", "pair_models", "cohn_vs_elbo.json")
OUT = os.path.join(REPO, "experiments", "figures", "bounds_vs_cohn.pdf")

# which models to show (drop the weakest mixture component if too cluttered)
DROP = set()   # e.g. {"mixture_K8_c4_mi0.164"} to remove the weakest


def build_model(name, pis, S_mix):
    """Reconstruct the Glauber W,R,pi_prod,pij for a model name from cohn_vs_elbo."""
    c = int(name.split("_c")[1].split("_")[0])
    pij = pis[c].reshape(NA, NA)
    if "STRONGx2" in name:                      # sharpen coupling: 2x pointwise MI
        pij = pij / pij.sum(); m1 = pij.sum(1); m2 = pij.sum(0)
        logMI = np.log(np.maximum(pij, 1e-300)) - np.log(m1)[:, None] - np.log(m2)[None, :]
        pij = np.outer(m1, m2) * np.exp(2.0 * logMI)
    W, R, pip, _params, pij = CV.build_glauber(S_mix, pij)
    return W, R, pip, pij.reshape(NS)


def main():
    d = json.load(open(COHN_JSON))
    tg = [float(t) for t in d["tgrid"]]
    zm = np.load(os.path.join(REPO, "results/mixture_component_char/components_K8.npz"),
                 allow_pickle=True)
    S_mix = np.asarray(zm["S"], float); pis = np.asarray(zm["pis"], float)
    models = [m for m in d["results"] if m not in DROP]

    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.0), squeeze=False)
    for ax, name in zip(axes[0], models):
        W, R, pip, pijv = build_model(name, pis, S_mix)
        gp_m, gmid_m, gc_m = [], [], []
        for t in tg:
            exact = np.log(np.maximum(EV.expm_rev(W, pijv, t), EV.FLOOR))
            Lp, _, _, _ = EV.closed_form_L(W, R, pip, t)
            Lm = midpoint_elbo_path_edges(W, R, pip, t)
            rows = d["results"][name]["t"][f"{t}"]["rows"]
            gp = [exact[r["x"], r["y"]] - Lp[r["x"], r["y"]] for r in rows]
            gm = [exact[r["x"], r["y"]] - Lm[r["x"], r["y"]] for r in rows]
            gc = [exact[r["x"], r["y"]] - r["cohn"] for r in rows if not np.isnan(r["cohn"])]
            gp_m.append(np.mean(gp)); gmid_m.append(np.mean(gm)); gc_m.append(np.mean(gc))
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        ax.plot(tg, gp_m, "--o", color="#1f77b4", ms=5, lw=1.6, label="path ELBO (ours)")
        ax.plot(tg, gmid_m, "-o", color="#2ca02c", ms=5, lw=2.0, label="+ midpoint node (ours)")
        ax.plot(tg, gc_m, ":s", color="#d62728", ms=5, lw=1.6, label="Cohn 2010 mean-field")
        ax.set_xscale("log"); ax.set_xlabel(r"branch length $t$")
        ax.set_title(f"MI(pi) = {d['results'][name]['mi']:.2f}")
        ax.set_ylabel("mean bound gap  (exact $-$ bound, nats)")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        # annotate the valid-bound region
        ax.text(0.98, 0.02, "gap<0 = bound violated", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, color="0.4")
    fig.suptitle("Endpoint-conditioned bounds on the Cohn/Glauber 400-state pair "
                 "(lower gap = tighter; <0 = invalid)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
