#!/usr/bin/env python3
"""Plot midpoint-node ELBO vs closed-form path ELBO (reads
results/pair_models/midpoint_vs_path_elbo.json).

Left : mean bound gap (exact - approx, meaningful entries) vs branch length --
       path ELBO (dashed) vs midpoint ELBO (solid), one colour per coupling.
Right: whole-400x400 wall-clock per method vs branch length (both strict bounds).
"""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/pair_models/midpoint_vs_path_elbo.json")
    ap.add_argument("--out", default="experiments/figures/midpoint_vs_path_elbo.pdf")
    a = ap.parse_args()
    d = json.load(open(a.json))
    tg = [float(t) for t in d["tgrid"]]
    models = list(d["results"])
    cmap = plt.get_cmap("viridis")
    col = {m: cmap(i / max(1, len(models) - 1)) for i, m in enumerate(models)}

    def label(m):
        if "trrosetta" in m:
            return "trRosetta"
        return m.split("_mi")[-1].join(["MI=", ""]) if "_mi" in m else m

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for m in models:
        R = d["results"][m]
        gp = np.array([R[f"{t}"]["gap_path"]["meaningful"]["mean"] for t in tg])
        gmp = np.array([R[f"{t}"]["gap_mpath"]["meaningful"]["mean"] for t in tg])
        axL.plot(tg, gmp, "-o", color=col[m], ms=4, lw=1.8, label=label(m))    # midpoint (path-edges)
        axL.plot(tg, gp, "--x", color=col[m], ms=5, lw=1.2, alpha=0.85)         # path ELBO
        with np.errstate(divide="ignore", invalid="ignore"):
            axR.plot(tg, gp / gmp, "-o", color=col[m], ms=4, lw=1.8, label=label(m))

    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel(r"branch length $t$"); axL.set_ylabel("mean bound gap (exact $-$ approx, nats)")
    axL.set_title("Tightness (all strict lower bounds)")
    leg1 = axL.legend(title="coupling", fontsize=7, loc="upper left", frameon=False)
    axL.add_artist(leg1)
    style = [Line2D([0], [0], color="0.3", ls="-", marker="o", ms=4, label="midpoint node (ours)"),
             Line2D([0], [0], color="0.3", ls="--", marker="x", ms=5, label="path ELBO (ours)")]
    axL.legend(handles=style, fontsize=7, loc="lower right", frameon=False)

    axR.axhline(1.0, color="0.7", lw=0.6, ls=":")
    axR.set_xscale("log")
    axR.set_xlabel(r"branch length $t$")
    axR.set_ylabel("gap ratio  path / midpoint(path-edges)")
    axR.set_title("Midpoint-node tightening (>1 = tighter than path ELBO)")
    axR.legend(fontsize=7, loc="upper right", frameon=False)

    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
