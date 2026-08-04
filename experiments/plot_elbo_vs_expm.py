#!/usr/bin/env python3
r"""Plot the closed-form path-ELBO vs exact matrix-exponential comparison
(reads results/pair_models/elbo_vs_expm.json, written by experiments/elbo_vs_expm.py)
into a paper figure with paper-consistent labels.

Run: python3 experiments/plot_elbo_vs_expm.py \
        --json results/pair_models/elbo_vs_expm.json \
        --out psb-paper/figures/elbo_vs_expm.pdf
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# model key -> label consistent with the paper's notation
LABELS = {
    "trrosetta_metropolis_sqrt": r"Metropolis ($\sqrt{\cdot}$)",
    "mixture_K8_c3_mi0.174": r"mixture class (MI 0.17)",
    "mixture_K8_c6_mi0.181": r"mixture class (MI 0.18)",
    "mixture_K8_c7_mi0.290": r"mixture class (MI 0.29)",
}
COLORS = {
    "trrosetta_metropolis_sqrt": "#1f77b4",
    "mixture_K8_c3_mi0.174": "#ff7f0e",
    "mixture_K8_c6_mi0.181": "#2ca02c",
    "mixture_K8_c7_mi0.290": "#d62728",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/pair_models/elbo_vs_expm.json")
    ap.add_argument("--out", default="psb-paper/figures/elbo_vs_expm.pdf")
    a = ap.parse_args()
    d = json.load(open(a.json))
    tgrid = np.array(d["tgrid"], float)
    models = [m for m in LABELS if m in d["results"]]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for m in models:
        c = COLORS[m]
        tk = {float(k): k for k in d["results"][m]}          # match JSON key format
        r = [d["results"][m][tk[t]] for t in tgrid]
        sp_e = [x["elbo"]["ranking"]["global_spearman"] for x in r]
        sp_p = [x["product_baseline"]["ranking"]["global_spearman"] for x in r]
        gp_e = [x["elbo"]["error"]["meaningful"]["mean"] for x in r]
        gp_p = [x["product_baseline"]["error"]["meaningful"]["mean"] for x in r]
        axL.plot(tgrid, sp_e, "-o", color=c, ms=4, lw=1.6, label=LABELS[m])
        axL.plot(tgrid, sp_p, "--x", color=c, ms=4, lw=1.1, alpha=0.8)
        axR.plot(tgrid, gp_e, "-o", color=c, ms=4, lw=1.6)
        axR.plot(tgrid, gp_p, "--x", color=c, ms=4, lw=1.1, alpha=0.8)

    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_xlabel(r"branch length $t$")
        ax.axhline(0 if ax is axR else 1.0, color="0.7", lw=0.6, ls=":")
    axL.set_ylabel(r"rank correlation with exact $e^{Qt}$")
    axL.set_title("Ranking of transition probabilities")
    axR.set_ylabel(r"mean $\log$-prob gap, exact $-$ approx (nats)")
    axR.set_title("Approximation gap (ELBO is a lower bound)")

    # model legend (colours) + style legend (solid ELBO / dashed product baseline)
    axL.legend(loc="lower left", fontsize=7, frameon=False)
    style = [Line2D([0], [0], color="0.3", ls="-", marker="o", ms=4, label="closed-form ELBO"),
             Line2D([0], [0], color="0.3", ls="--", marker="x", ms=4,
                    label="product baseline (no coupling)")]
    axR.legend(handles=style, loc="lower left", fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
