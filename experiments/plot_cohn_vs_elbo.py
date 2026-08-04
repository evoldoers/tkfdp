#!/usr/bin/env python3
"""Plot the Cohn CTBN mean-field vs closed-form path ELBO vs exact comparison
(reads results/pair_models/cohn_vs_elbo.json, written by cohn_vs_elbo_pair.py).

Left : mean bound gap (exact - approx) vs branch length -- closed-form ELBO (solid)
       vs Cohn mean-field (dashed), one colour per coupling strength.
Right: per-endpoint-pair exact log-prob vs approximation, all models/branches
       pooled; points on/below the y=x line are valid lower bounds. A timing note
       contrasts the whole-matrix closed form with Cohn's per-pair ODE cost.

Run: python3 experiments/plot_cohn_vs_elbo.py
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/pair_models/cohn_vs_elbo.json")
    ap.add_argument("--out", default="experiments/figures/cohn_vs_elbo.pdf")
    a = ap.parse_args()
    d = json.load(open(a.json))
    tgrid = [float(t) for t in d["tgrid"]]
    models = list(d["results"])
    cmap = plt.get_cmap("viridis")
    colors = {m: cmap(i / max(1, len(models) - 1)) for i, m in enumerate(models)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # ---- left: mean gap vs t ----
    ours_full_ms, cohn_pp_s = [], []
    for m in models:
        R = d["results"][m]
        mi = R["mi"]
        g_o = [R["t"][f"{t}"]["gap_ours"]["mean"] for t in tgrid]
        g_c = [R["t"][f"{t}"]["gap_cohn"]["mean"] for t in tgrid]
        axL.plot(tgrid, g_o, "-o", color=colors[m], ms=4, lw=1.8,
                 label=f"MI={mi:.2f}")
        axL.plot(tgrid, g_c, "--x", color=colors[m], ms=5, lw=1.2, alpha=0.85)
        for t in tgrid:
            ours_full_ms.append(R["t"][f"{t}"]["t_ours_fullmatrix_s"] * 1e3)
            cohn_pp_s.append(R["t"][f"{t}"]["t_cohn_perpair_median_s"])
    axL.axhline(0, color="0.7", lw=0.6, ls=":")
    axL.set_xscale("log"); axL.set_xlabel("branch length $t$")
    axL.set_ylabel("mean bound gap  (exact $-$ approx, nats)")
    axL.set_title("Tightness vs branch length")
    leg1 = axL.legend(title="coupling", fontsize=7, loc="upper left", frameon=False)
    axL.add_artist(leg1)
    style = [Line2D([0], [0], color="0.3", ls="-", marker="o", ms=4, label="closed-form ELBO"),
             Line2D([0], [0], color="0.3", ls="--", marker="x", ms=5, label="Cohn mean-field")]
    axL.legend(handles=style, fontsize=7, loc="lower right", frameon=False)

    # ---- right: per-pair exact vs approx scatter ----
    ex_all, ou_all, co_all = [], [], []
    for m in models:
        for t in tgrid:
            for r in d["results"][m]["t"][f"{t}"]["rows"]:
                ex_all.append(r["exact"]); ou_all.append(r["ours"]); co_all.append(r["cohn"])
    ex_all = np.array(ex_all); ou_all = np.array(ou_all); co_all = np.array(co_all)
    lo = float(min(ex_all.min(), np.nanmin(ou_all), np.nanmin(co_all)))
    axR.plot([lo, 0.02], [lo, 0.02], color="0.6", lw=0.8, ls=":", label="y = x (exact)")
    axR.scatter(ex_all, ou_all, s=14, color="#1f77b4", alpha=0.7, label="closed-form ELBO")
    axR.scatter(ex_all, co_all, s=14, color="#d62728", marker="x", alpha=0.7, label="Cohn mean-field")
    axR.set_xlabel("exact  $\\log[e^{tW}]_{xy}$")
    axR.set_ylabel("approximation (lower bound)")
    axR.set_title("Per-pair bound (below y=x $\\Rightarrow$ valid)")
    axR.legend(fontsize=7, loc="upper left", frameon=False)
    # timing annotation
    om = np.median(ours_full_ms); cp = np.nanmedian(cohn_pp_s)
    speed = (cp * 1e3) / (om / (d["ns"] ** 2))    # cohn-per-pair / ours-amortized-per-pair
    axR.text(0.98, 0.02,
             f"closed form: {om:.0f} ms / whole {d['ns']}×{d['ns']} matrix\n"
             f"Cohn: {cp:.1f} s / single pair (ODE)\n"
             f"≈ {speed:,.0f}× faster per pair",
             transform=axR.transAxes, ha="right", va="bottom", fontsize=7,
             bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
