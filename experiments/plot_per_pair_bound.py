#!/usr/bin/env python3
r"""Per-pair bound plot -- an updated version of the right ("Per-pair bound")
panel of experiments/figures/cohn_vs_elbo.pdf, now built from the two-pass data
(results/pair_models/cohn_two_pass.json): the closed-form path ELBO, the
midpoint-node ELBO, and full-psi Cohn, pooled over both mixture classes and all
branch lengths.

Left  : exact log-transition vs approximation, with the y=x line. A point on or
        below the line is a valid lower bound; a point ABOVE it is a bound
        violation. Each method's legend reports its violation fraction.
Right : residual (approx - exact) vs exact, with the y=0 line -- a magnifier for
        the violations, which are small in absolute terms. Our closed forms stay
        at or below 0 (valid); full-psi Cohn pokes above 0 at long branches (the
        singular overshoot).

Run: python3 experiments/plot_per_pair_bound.py \
        --json results/pair_models/cohn_two_pass.json \
        --out  psb-paper/figures/per_pair_bound.pdf
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (exact-array key, gap-array key, label, colour)
METHODS = [
    ("ex",         "gap_path",         "closed-form path ELBO", "#1f77b4"),
    ("ex",         "gap_mid",          "midpoint-node ELBO",    "#2ca02c"),
    ("fullpsi_ex", "gap_cohn_fullpsi", r"Cohn (full $\psi$)",   "#9467bd"),
]
VIOL_TOL = 1e-6   # ignore sub-nat numerical noise when counting violations


def collect(results, exk, gapk):
    ex, gap = [], []
    for m in results:
        for _, c in results[m]["t"].items():
            e, g = c.get(exk), c.get(gapk)
            if e and g:
                ex.extend(e); gap.extend(g)
    ex = np.asarray(ex, float); gap = np.asarray(gap, float)
    return ex, ex - gap   # (exact, approx); approx = exact - gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/pair_models/cohn_two_pass.json")
    ap.add_argument("--out", default="psb-paper/figures/per_pair_bound.pdf")
    a = ap.parse_args()
    results = json.load(open(a.json))["results"]
    rng = np.random.default_rng(0)

    fig, (axS, axR) = plt.subplots(1, 2, figsize=(10.6, 4.5))
    allex, allres = [], []
    for exk, gapk, label, col in METHODS:
        ex, apx = collect(results, exk, gapk)
        if ex.size == 0:
            continue
        resid = apx - ex                       # > 0  <=>  bound violation
        vfrac = float(np.mean(resid > VIOL_TOL))
        allex.append(ex); allres.append(resid)
        sel = (rng.choice(ex.size, size=4000, replace=False)
               if ex.size > 4000 else np.arange(ex.size))
        lbl = f"{label}  ({100*vfrac:.1f}% violate, n={ex.size:,})"
        axS.scatter(ex[sel], apx[sel], s=6, color=col, alpha=0.30, edgecolors="none", label=lbl)
        axR.scatter(ex[sel], resid[sel], s=6, color=col, alpha=0.30, edgecolors="none")
    allex = np.concatenate(allex); allres = np.concatenate(allres)

    lo, hi = float(allex.min()), float(allex.max()); pad = 0.03 * (hi - lo)
    axS.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.35", lw=1.0, ls="--", zorder=0)
    axS.set_xlim(lo - pad, hi + pad); axS.set_ylim(lo - pad, hi + pad)
    axS.set_xlabel(r"exact $\log[e^{tW}]_{xy}$"); axS.set_ylabel("approximation")
    axS.set_title(r"Per-pair bound: on/below $y{=}x$ $\Rightarrow$ valid")
    axS.legend(fontsize=7, loc="upper left", frameon=False)

    # residual magnifier: scale the y-window to the VIOLATION magnitude (not the
    # much larger valid-side gaps, which would squash the violations against y=0).
    pos = allres[allres > VIOL_TOL]
    vscale = float(np.nanpercentile(pos, 98)) if pos.size else 0.004
    axR.axhline(0, color="0.35", lw=1.0, ls="--", zorder=0)
    axR.set_ylim(-2.0 * vscale, 1.4 * vscale)   # valid gaps below this window are clipped
    axR.set_xlim(lo - pad, hi + pad)
    axR.set_xlabel(r"exact $\log[e^{tW}]_{xy}$")
    axR.set_ylabel(r"residual  approx $-$ exact  (nats)")
    axR.set_title(r"Violation magnifier: residual $>0$ $\Rightarrow$ violated")
    axR.text(0.02, 0.97, "above the line = bound violation", transform=axR.transAxes,
             fontsize=7, va="top", color="0.4")

    fig.suptitle("Per-pair bound and violations (both mixture classes, all branch lengths pooled)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
