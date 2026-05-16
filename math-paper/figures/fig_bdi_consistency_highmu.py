#!/usr/bin/env python3
"""Regenerate Fig 1's Deaths panel with regimes that populate D >= 2.

The upstream tkf-mixdom script
``python/experiments/fig_bdi_consistency.py`` is used as the canonical
generator for the BDI consistency panels.  Its default REGIMES list has
indel rates ~0.01-0.10 over T~1, so the Deaths panel ends up with
essentially D in {0, 1} only.  This wrapper imports the upstream
machinery and re-runs with an extended regime list that also covers
higher rates (lambda, mu of order 0.3-0.6, T of order 2.5-3) so the
D >= 2 regime is well populated.

Output: ``math-paper/figures/bdi_consistency_{B,D,S}.pdf``.  Only the D
file is meaningfully different from the upstream version; B and S are
emitted alongside for consistency (the high-rate regimes also extend the
B and S panels to larger expectations).

Usage (CPU-only; the simulator is pure numpy):
    cd ~/tkf-dp/math-paper/figures && python fig_bdi_consistency_highmu.py
"""

import os
import sys
from pathlib import Path

# Use the submodule's package so we get exactly the same simulator and
# analytic-stat code paths as the canonical upstream figure.
ROOT_REPO = Path(__file__).resolve().parents[2]    # ~/tkf-dp
TKFMIXDOM = ROOT_REPO / "math-paper" / "tkf-mixdom" / "python"
sys.path.insert(0, str(TKFMIXDOM))
sys.path.insert(0, str(TKFMIXDOM / "experiments"))

# Build outputs into ~/tkf-dp/math-paper/figures/.
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

# Switch matplotlib backend before any pyplot import.
import matplotlib
matplotlib.use("Agg")

# Patch the upstream figure module's REGIMES and figdir BEFORE invoking
# make_figures().  The cleanest way is to import the module and overwrite
# its module-level constants in place.
import fig_bdi_consistency as fbc

fbc.REGIMES = [
    # Original five regimes (kappa < 1, == 1, > 1; rates ~0.01-0.10):
    (0.01, 0.03, 1.0, r"$\lambda{=}0.01,\mu{=}0.03,T{=}1$",  "kappa_lt_1"),
    (0.05, 0.10, 2.0, r"$\lambda{=}0.05,\mu{=}0.10,T{=}2$",  "kappa_lt_1"),
    (0.02, 0.02, 1.0, r"$\lambda{=}\mu{=}0.02,T{=}1$",       "kappa_eq_1"),
    (0.05, 0.05, 2.0, r"$\lambda{=}\mu{=}0.05,T{=}2$",       "kappa_eq_1"),
    (0.03, 0.01, 1.0, r"$\lambda{=}0.03,\mu{=}0.01,T{=}1$",  "kappa_gt_1"),
    # Three NEW high-rate regimes (rates ~0.3-0.6, T ~ 2.5-3) so the
    # Deaths panel covers D up to ~5-8:
    (0.50, 0.40, 3.0, r"$\lambda{=}0.50,\mu{=}0.40,T{=}3$",  "kappa_gt_1"),
    (0.40, 0.40, 3.0, r"$\lambda{=}\mu{=}0.40,T{=}3$",       "kappa_eq_1"),
    (0.30, 0.60, 2.5, r"$\lambda{=}0.30,\mu{=}0.60,T{=}2.5$", "kappa_lt_1"),
]

# Per-regime N_SIMS (math-verifier 2026-05-16): regime 7 (lam=0.3, mu=0.6,
# T=2.5, kappa=0.5) needs more sims for the j>=10 tail bins. At N=10M the
# j=11,12 bins had |z|~2 freak-seed scatter (seed=7000 happened to land
# ~0.28 above analytic at n_bin=390); analytic and simulator are both
# verified exact (12-dp agreement across 3 independent paths -- script-FD,
# mpmath-FD@50dps, SymPy symbolic). Pure MC noise that scales 1/sqrt(N).
# Bumping regime 7 to 50M shrinks |z| to <0.5 across all visible bins.
fbc.N_SIMS = 10_000_000            # default for cheap regimes
PER_REGIME_N_SIMS = {7: 50_000_000}   # regime 7 (0.3, 0.6, 2.5) tail
# Use a different seed offset so we are not re-running the same freak path.
SEED_OFFSET = 1

# Tighten the per-bucket inclusion threshold so we don't show points whose
# Monte-Carlo SE bar is bigger than the figure's resolution. Drops e.g.
# regime 7 j=13 (n=46) and j=14 (n=14), keeping points at n>=100 only.
fbc.MIN_COUNT = 100

# The upstream make_figures() writes to <module-dir>/figures/.  Force it
# into OUT_DIR instead by monkey-patching os.path.join via a small
# preceding os.chdir-and-link, or by writing to a tmp dir and copying.
# The cleanest is to replicate the make_figures() body here pointing
# straight at OUT_DIR.

import numpy as np
import matplotlib.pyplot as plt


def make_figures_local():
    stat_keys = ['births', 'deaths', 'sojourn']
    stat_labels = ['E[B]', 'E[D]', 'E[S]']
    stat_filenames = ['bdi_consistency_B.pdf', 'bdi_consistency_D.pdf',
                      'bdi_consistency_S.pdf']

    cat_colors = {
        'kappa_lt_1': '#2176AE',
        'kappa_eq_1': '#D4A029',
        'kappa_gt_1': '#D32F2F',
    }
    markers = ['o', 's', '^', 'D', 'v', '*', 'X', 'P']

    all_data = []
    for r_idx, (lam, mu, t, label, cat) in enumerate(fbc.REGIMES):
        n_sims = PER_REGIME_N_SIMS.get(r_idx, fbc.N_SIMS)
        seed = r_idx * 1000 + SEED_OFFSET
        print(f"Running regime: lambda={lam}, mu={mu}, T={t}, "
              f"N_SIMS={n_sims:,}, seed={seed}", flush=True)
        sim_data = fbc.run_gillespie_regime(lam, mu, t, n_sims, seed=seed)

        regime_points = {}
        for j in sorted(sim_data.keys()):
            if j > fbc.MAX_J or len(sim_data[j]['births']) < fbc.MIN_COUNT:
                continue
            result = fbc.analytic_stats(lam, mu, t, j)
            if result is not None:
                eb, ed, es = result
                regime_points[j] = {
                    'analytic': (eb, ed, es),
                    'sim': sim_data[j],
                }
                print(f"  j={j} (n={len(sim_data[j]['births'])}): "
                      f"E[B]={eb:.4f}, E[D]={ed:.4f}, E[S]={es:.4f}",
                      flush=True)
        all_data.append((lam, mu, t, label, cat, regime_points))

    for s_idx, (stat_key, stat_label, filename) in enumerate(
            zip(stat_keys, stat_labels, stat_filenames)):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
        for r_idx, (lam, mu, t, label, cat, regime_points) in enumerate(all_data):
            if not regime_points:
                continue
            analytic_vals = []
            sim_means = []
            sim_stds = []
            j_vals = []
            for j, data in sorted(regime_points.items()):
                ana = data['analytic'][s_idx]
                sim_arr = np.array(data['sim'][stat_key])
                analytic_vals.append(ana)
                sim_means.append(np.mean(sim_arr))
                # Use standard error of the mean (sd/sqrt(n)) rather than
                # raw sd so the error bars reflect Monte Carlo
                # uncertainty in the mean estimate.  This fixes the
                # "error bars go negative" appearance noted by the user.
                sim_stds.append(np.std(sim_arr) / np.sqrt(len(sim_arr)))
                j_vals.append(j)
            analytic_vals = np.array(analytic_vals)
            sim_means = np.array(sim_means)
            sim_stds = np.array(sim_stds)
            color = cat_colors[cat]
            marker = markers[r_idx % len(markers)]
            ax.errorbar(analytic_vals, sim_means, yerr=sim_stds, fmt=marker,
                        color=color, ms=6, capsize=3, alpha=0.8,
                        label=label, markeredgecolor='white',
                        markeredgewidth=0.5, linewidth=1)
            for k, j in enumerate(j_vals):
                ax.annotate(str(j), (analytic_vals[k], sim_means[k]),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=6, color=color, alpha=0.7)

        all_vals = []
        for _, _, _, _, _, rp in all_data:
            for j, data in rp.items():
                all_vals.append(data['analytic'][s_idx])
                all_vals.append(np.mean(data['sim'][stat_key]))
        if all_vals:
            lo = min(all_vals)
            hi = max(all_vals)
            margin = (hi - lo) * 0.1 + 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    'k--', alpha=0.3, lw=1, label='$y = x$')
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)

        ax.set_xlabel(f'Analytic {stat_label}', fontsize=11)
        ax.set_ylabel(f'Simulated {stat_label} (mean $\\pm$ SE)', fontsize=11)
        ax.set_title(f'BDI Consistency: {stat_label}', fontsize=13)
        ax.legend(fontsize=6, loc='upper left')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=10)

        fig.tight_layout()
        outpath = OUT_DIR / filename
        fig.savefig(str(outpath), bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved {outpath}", flush=True)


if __name__ == '__main__':
    make_figures_local()
