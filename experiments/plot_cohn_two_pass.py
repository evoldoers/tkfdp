#!/usr/bin/env python3
r"""Plot the two-pass Cohn-vs-closed-form comparison from a SINGLE data source
(results/pair_models/cohn_two_pass.json, written by experiments/cohn_two_pass.py).

Because every curve is read from that one file, every panel shares the same
branch-length grid and the same sampled endpoint pairs -- there is no cross-panel
x-axis mismatch, and there are no 'coupled expm' points.

Layout: rows = mixture coupling classes, columns =
  (1) "Ranking of transition probabilities" -- Spearman rank correlation of each
      approximation's per-pair log-prob against the exact log[e^{tW}]_xy, vs t.
  (2) "Approximation gap" -- mean per-pair bound gap (exact - approx, nats) vs t;
      all methods are lower bounds so the gap is >= 0.  A grey step on a twin axis
      shows the fraction of full-psi Cohn pairs that FAIL to converge within the
      diffrax step budget (rising with t -- the fragility that motivates the
      closed form).

Methods: closed-form path ELBO, midpoint-node ELBO, Cohn mean-field (no back-
reaction psi; dense, always valid), and full-psi Cohn (true Cohn 2010; only the
converged pairs contribute, hence the companion fail curve).

Run: python3 experiments/plot_cohn_two_pass.py \
        --json results/pair_models/cohn_two_pass.json \
        --out  psb-paper/figures/cohn_two_pass.pdf
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

# method key in the JSON cell -> (exact-array key, gap-array key, label, colour)
METHODS = [
    ("path",         "ex",         "gap_path",         "closed-form path ELBO", "#1f77b4"),
    ("mid",          "ex",         "gap_mid",          "midpoint-node ELBO",    "#2ca02c"),
    ("cohn_nopsi",   "ex",         "gap_cohn_nopsi",   r"Cohn (no $\psi$)",     "#d62728"),
    ("cohn_fullpsi", "fullpsi_ex", "gap_cohn_fullpsi", r"Cohn (full $\psi$)",   "#9467bd"),
]


def _spearman_ci(ex, gap, rng, B=300, cap=2000):
    """Spearman(exact, approx) point estimate + 95%% bootstrap CI half-widths.
    approx = exact - gap. Bootstrap size is capped (the dense n=4000 CI is already ~0)."""
    ex = np.asarray(ex, float); approx = ex - np.asarray(gap, float)
    n = ex.size
    if n < 5 or np.ptp(ex) == 0 or np.ptp(approx) == 0:
        return np.nan, 0.0, 0.0
    r = float(spearmanr(ex, approx).correlation)
    m = min(n, cap)
    idx = rng.integers(0, n, size=(B, m))
    boots = np.array([spearmanr(ex[j], approx[j]).correlation for j in idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return r, max(0.0, r - lo), max(0.0, hi - r)


def _draw_rank(ax, cells, tg, m, mi, rng):
    for key, exk, gapk, label, col in METHODS:
        ts, rmid, rlo, rhi = [], [], [], []
        for t in tg:
            cell = cells[f"{t}"]; ex, gap = cell.get(exk), cell.get(gapk)
            if not ex or not gap:
                continue
            r, dlo, dhi = _spearman_ci(ex, gap, rng)
            ts.append(t); rmid.append(r); rlo.append(dlo); rhi.append(dhi)
        if ts:
            ax.errorbar(ts, rmid, yerr=[rlo, rhi], fmt="-o", color=col, ms=4, lw=1.6,
                        elinewidth=1.0, capsize=2, label=label)
    ax.set_xscale("log"); ax.set_xlabel(r"branch length $t$")
    ax.axhline(1.0, color="0.8", lw=0.6, ls=":")
    ax.set_ylabel("Spearman rank corr. with exact")
    ax.set_title(f"Ranking of transition probabilities\n{m}  (MI={mi:.2f})", fontsize=9)


def _draw_gap(ax, cells, tg, m, mi, legend=False):
    axFail = ax.twinx()
    for key, exk, gapk, label, col in METHODS:
        ts, gmean, gse = [], [], []
        for t in tg:
            cell = cells[f"{t}"]; ex, gap = cell.get(exk), cell.get(gapk)
            if not ex or not gap:      # e.g. full-psi with zero converged pairs
                continue
            g = np.asarray(gap, float)
            ts.append(t); gmean.append(float(g.mean()))
            gse.append(float(g.std(ddof=1) / np.sqrt(g.size)) if g.size > 1 else 0.0)
        if ts:
            ax.errorbar(ts, gmean, yerr=gse, fmt="-o", color=col, ms=4, lw=1.6,
                        elinewidth=1.0, capsize=2, label=label)
    # full-psi non-convergence fraction (twin axis), with +/-1 SE binomial band
    tf = [t for t in tg if cells[f"{t}"].get("n2_done")]
    ff = [cells[f"{t}"]["frac_fail"] for t in tf]; nn = [cells[f"{t}"]["n2_done"] for t in tf]
    if tf:
        fse = [float(np.sqrt(max(p, 1e-9) * (1 - p) / n)) for p, n in zip(ff, nn)]
        axFail.plot(tf, ff, drawstyle="steps-mid", color="0.55", lw=1.0, ls="--")
        axFail.fill_between(tf, [max(0.0, p - e) for p, e in zip(ff, fse)],
                            [min(1.0, p + e) for p, e in zip(ff, fse)],
                            step="mid", color="0.55", alpha=0.20)
    axFail.set_ylim(0, 1.02)
    axFail.set_ylabel(r"frac. full-$\psi$ Cohn non-converged", color="0.4", fontsize=8)
    axFail.tick_params(axis="y", labelcolor="0.4", labelsize=7)
    ax.set_xscale("log"); ax.set_xlabel(r"branch length $t$")
    ax.axhline(0.0, color="0.8", lw=0.6, ls=":")
    ax.set_ylabel("mean bound gap (exact $-$ approx, nats)")
    ax.set_title(f"Approximation gap (lower bound $\\Rightarrow$ gap $\\geq 0$)\n{m}  (MI={mi:.2f})", fontsize=9)
    if legend:
        ax.legend(fontsize=7, loc="upper left", frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/pair_models/cohn_two_pass.json")
    ap.add_argument("--out", default="psb-paper/figures/cohn_two_pass.pdf")
    ap.add_argument("--panels", choices=["both", "gap", "rank"], default="both",
                    help="both = 2x2 (rank|gap per model); gap/rank = that column only, 1xN")
    a = ap.parse_args()
    results = json.load(open(a.json))["results"]
    models = list(results)
    n = len(models)
    rng = np.random.default_rng(0)

    if a.panels == "both":
        fig, axes = plt.subplots(n, 2, figsize=(9.6, 3.4 * n), squeeze=False)
        for i, m in enumerate(models):
            cells, mi = results[m]["t"], results[m]["mi"]
            tg = sorted((float(k) for k in cells), key=float)
            _draw_rank(axes[i][0], cells, tg, m, mi, rng)
            _draw_gap(axes[i][1], cells, tg, m, mi)
        axes[0][0].legend(fontsize=7, loc="lower left", frameon=False)
        note = (r"Error bars: bound gap $\pm1$ SE of the mean; Spearman $95\%$ bootstrap CI; "
                r"grey band $\pm1$ SE (binomial) on non-convergence.")
    else:
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.3), squeeze=False)
        for i, m in enumerate(models):
            cells, mi = results[m]["t"], results[m]["mi"]
            tg = sorted((float(k) for k in cells), key=float)
            if a.panels == "gap":
                _draw_gap(axes[0][i], cells, tg, m, mi, legend=(i == 0))
                note = (r"Error bars: bound gap $\pm1$ SE of the mean; "
                        r"grey band $\pm1$ SE (binomial) on the full-$\psi$ non-convergence fraction.")
            else:
                _draw_rank(axes[0][i], cells, tg, m, mi, rng)
                if i == 0:
                    axes[0][i].legend(fontsize=7, loc="lower left", frameon=False)
                note = r"Error bars: Spearman $95\%$ bootstrap CI."

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.text(0.5, 0.008, note, ha="center", va="bottom", fontsize=7, color="0.4")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
