#!/usr/bin/env python3
"""Compare the fitted joint pair stationaries pi(x,y) across the pair-substitution
models, from a saved fit_pair_models params file (no re-fit). Emits a numerical
report (pairwise KL / TV, stationary coupling, top log-odds elements) and two
figures: a grid of 20x20 log-colour pi heatmaps, and log-odds difference maps vs
a reference model.

Usage:
  python analysis/scripts/pair_model_pi_analysis.py \
      results/pair_models/trrosetta_rate_converged_params.npz --outdir analysis/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

AA = list("ACDEFGHIKLMNPQRSTVWY")
NICE = {"synchronized": "Synchronized", "coupled": "Coupled",
        "metropolis_barker": "Metropolis (Barker)", "metropolis_sqrt": "Metropolis (sqrt)",
        "metropolis_hastings": "Metropolis (Hastings)", "metropolis_gtr": "Metropolis (GTR)",
        "lumpable": "Lumpable"}


def KL(p, q):
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum())


def _label(ax):
    ax.set_xticks(range(20)); ax.set_yticks(range(20))
    ax.set_xticklabels(AA, fontsize=6); ax.set_yticklabels(AA, fontsize=6)
    ax.tick_params(length=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("params", help="a fit_pair_models <out>_params.npz")
    ap.add_argument("--outdir", default="analysis/figures")
    ap.add_argument("--ref", default="synchronized", help="reference for log-odds diff")
    ap.add_argument("--tag", default=None, help="filename tag (default: params stem)")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.params).stem.replace("_params", "")

    z = np.load(args.params)
    models = [k[:-4] for k in z.files if k.endswith("__pi")]
    order = [m for m in NICE if m in models] + [m for m in models if m not in NICE]
    P = {m: z[f"{m}__pi"].reshape(20, 20) for m in order}

    print(f"# {tag}: models = {order}")
    print("\n=== pairwise KL (milli-bits) [row||col] ===")
    print(" " * 16 + " ".join(f"{m[:8]:>9s}" for m in order))
    for a in order:
        print(f"{a:16s} " + " ".join(
            f"{KL(P[a].ravel(), P[b].ravel())/np.log(2)*1000:9.2f}" for b in order))
    print("\n=== stationary coupling KL(pi || marg⊗marg) milli-bits ===")
    for m in order:
        mg = 0.5 * (P[m].sum(1) + P[m].sum(0)); ind = np.outer(mg, mg); ind /= ind.sum()
        print(f"{m:16s} {KL(P[m].ravel(), ind.ravel())/np.log(2)*1000:8.3f}")
    ref = args.ref
    if ref in P:
        for m in order:
            if m == ref:
                continue
            lor = np.log2(np.clip(P[m], 1e-12, None) / np.clip(P[ref], 1e-12, None))
            idx = np.dstack(np.unravel_index(np.argsort(-np.abs(lor), axis=None), (20, 20)))[0]
            seen, tops = set(), []
            for (i, j) in idx:
                k = tuple(sorted((i, j)))
                if k in seen:
                    continue
                seen.add(k); tops.append(f"{AA[i]}{AA[j]}:{lor[i,j]:+.2f}")
                if len(tops) >= 6:
                    break
            print(f"# top log2-odds {m} vs {ref}: " + "  ".join(tops))

    # Figure 1: pi heatmaps
    vmin = min(P[m][P[m] > 0].min() for m in order); vmax = max(P[m].max() for m in order)
    nc = 3; nr = int(np.ceil(len(order) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 5 * nr), squeeze=False)
    for ax, m in zip(axes.ravel(), order):
        im = ax.imshow(P[m], norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
        ax.set_title(NICE.get(m, m), fontsize=11); _label(ax)
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    fig.suptitle(f"Fitted joint pair stationary π(x,y) — {tag} (log colour)", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.6, label="π(x,y)")
    f1 = out / f"pi_heatmaps_{tag}.png"; fig.savefig(f1, dpi=130, bbox_inches="tight")

    # Figure 2: log-odds diff vs ref
    comps = [m for m in order if m != ref]
    fig, axes = plt.subplots(1, len(comps), figsize=(5.2 * len(comps), 5.2), squeeze=False)
    for ax, m in zip(axes[0], comps):
        lor = np.log2(np.clip(P[m], 1e-12, None) / np.clip(P[ref], 1e-12, None))
        mm = max(np.abs(lor).max(), 1e-6)
        im = ax.imshow(lor, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-mm, vmax=mm))
        ax.set_title(f"log2 π[{NICE.get(m,m)}]/π[{NICE.get(ref,ref)}]\nmax|Δ|={mm:.2f} bits", fontsize=9)
        _label(ax); fig.colorbar(im, ax=ax, shrink=0.8)
    f2 = out / f"pi_logodds_vs_{ref}_{tag}.png"; fig.savefig(f2, dpi=130, bbox_inches="tight")
    print(f"\n# wrote {f1}\n# wrote {f2}")


if __name__ == "__main__":
    main()
