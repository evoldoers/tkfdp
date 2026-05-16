#!/usr/bin/env python3
"""Stacked-area-streamgraph rendering of the same bridge posteriors.

Same data as ``plot_aa_evolution.py`` / ``_k4.py``, different visual:
each state contributes a coloured horizontal band whose vertical extent
equals its probability mass at each time step; bands stack vertically to
sum to 1. Vector graphics (PDF) by default; scales freely without
pixelation.

Single-AA panel: 20 stacked bands (one per amino acid, hue per LG order).
Joint panel:    400 stacked bands (one per (i, j) pair, hue per pair).

To keep the joint plot readable, by default we BIN the 400 bands into 20
groups of 20 (one per anchor amino acid in the first coordinate, i),
showing each as a stacked block with the dominant-pair hue. The optional
``--no-bin`` flag renders all 400 bands.

Recipe:

    cd ~/tkf-dp
    python analysis/scripts/plot_aa_evolution_stacked.py --batch
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_aa_evolution import (
    LG_ORDER, hue_single, hue_pair, aa_to_lg_idx, load_lg_in_lg_order,
    bridge_marginal,
)
from plot_aa_evolution_k4 import (
    load_k4, per_class_rate_matrix, build_k4_coevol_rate_matrix,
    bridge_joint_k4, CANONICAL_EXAMPLES, OUT_DIR,
)


def stacked_marginal_axes(ax, q: np.ndarray, ts: np.ndarray,
                            title: str, ylabel_states: bool = True):
    """Stacked-area plot for a (20, n_t) marginal density."""
    n, _ = q.shape
    # Each row is a band whose height at each t is q[k, t].
    colors = [mcolors.hsv_to_rgb([hue_single(k), 0.7, 0.85]) for k in range(n)]
    cumsum = np.zeros_like(q[0])
    for k in range(n):
        lower = cumsum
        upper = cumsum + q[k]
        ax.fill_between(ts, lower, upper, color=colors[k],
                         edgecolor="none", linewidth=0.0)
        cumsum = upper
    ax.set_xlim(ts[0], ts[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("cumulative probability" if ylabel_states else "")
    ax.set_title(title, fontsize=10)
    # Letter labels at right margin (scalar y, not vector).
    cs = 0.0
    for k in range(n):
        mid = float(cs + q[k, -1] / 2.0)
        if q[k, -1] > 0.012:
            ax.text(ts[-1] * 1.005, mid, LG_ORDER[k],
                     fontsize=7, va="center", ha="left",
                     color=mcolors.hsv_to_rgb([hue_single(k), 0.9, 0.5]))
        cs += float(q[k, -1])


def stacked_joint_axes(ax, qjoint: np.ndarray, ts: np.ndarray,
                         title: str, bin_by_i: bool = True):
    """Stacked-area plot for a (400, n_t) joint density.

    If ``bin_by_i`` (default), bands are grouped by first coordinate i:
    each band is the SUM over j of P(x=(i, j); t) -- so the band heights
    sum to the marginal of the first coordinate. We colour each band with
    the dominant-j-cell's pair-hue.

    With ``bin_by_i=False`` all 400 bands are drawn individually.
    """
    n = 20
    n_t = qjoint.shape[1]
    cumsum = np.zeros(n_t)
    if bin_by_i:
        # Group rows by i (i.e., sum over j for each i).
        groups = qjoint.reshape(n, n, n_t).sum(axis=1)   # (n, n_t)
        # Hue: each i is a band of fixed colour (use the diagonal (i,i)
        # pair-hue as the representative).
        colors = [mcolors.hsv_to_rgb([hue_pair(i, i), 0.7, 0.85])
                   for i in range(n)]
        for i in range(n):
            ax.fill_between(ts, cumsum, cumsum + groups[i],
                             color=colors[i], edgecolor="none", linewidth=0.0)
            cumsum = cumsum + groups[i]
        # Right-margin labels (scalar accumulator)
        cs = 0.0
        for i in range(n):
            mid = float(cs + groups[i, -1] / 2.0)
            if groups[i, -1] > 0.012:
                ax.text(ts[-1] * 1.005, mid, LG_ORDER[i],
                         fontsize=7, va="center", ha="left",
                         color=mcolors.hsv_to_rgb([hue_pair(i, i), 0.9, 0.5]))
            cs += float(groups[i, -1])
    else:
        # All 400 bands; colour by pair-hue.
        for k in range(qjoint.shape[0]):
            i, j = k // n, k % n
            hue = hue_pair(i, j)
            ax.fill_between(ts, cumsum, cumsum + qjoint[k],
                             color=mcolors.hsv_to_rgb([hue, 0.7, 0.85]),
                             edgecolor="none", linewidth=0.0)
            cumsum = cumsum + qjoint[k]
    ax.set_xlim(ts[0], ts[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("time $t$")
    ax.set_title(title, fontsize=10)


def stacked_joint_diff_axes(ax, qjoint_k4: np.ndarray,
                              qjoint_ind: np.ndarray,
                              ts: np.ndarray, n: int,
                              title: str):
    """Stacked-area of K=4 joint binned by first coord i, with each
    band split into bottom min(K4, prod) in pale hue and top
    max(K4 - prod, 0)+ in bright hue. The bright cap directly
    visualises the coevolution excess that the independent-sites
    approximation misses."""
    n_t = qjoint_k4.shape[1]
    band_k4 = qjoint_k4.reshape(n, n, n_t).sum(axis=1)
    band_prod = qjoint_ind.reshape(n, n, n_t).sum(axis=1)
    agreement = np.minimum(band_k4, band_prod)
    excess = np.maximum(band_k4 - band_prod, 0.0)
    pale_hues = [mcolors.hsv_to_rgb([hue_pair(i, i), 0.25, 0.97]) for i in range(n)]
    bright_hues = [mcolors.hsv_to_rgb([hue_pair(i, i), 0.95, 0.80]) for i in range(n)]
    cumsum = np.zeros(n_t)
    for i in range(n):
        ax.fill_between(ts, cumsum, cumsum + agreement[i],
                         color=pale_hues[i], edgecolor="none", linewidth=0.0)
        c_after = cumsum + agreement[i]
        ax.fill_between(ts, c_after, c_after + excess[i],
                         color=bright_hues[i], edgecolor="none", linewidth=0.0)
        cumsum = c_after + excess[i]
    ax.set_xlim(ts[0], ts[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("time $t$")
    ax.set_title(title, fontsize=10)
    cs = 0.0
    for i in range(n):
        mid = float(cs + band_k4[i, -1] / 2.0)
        if band_k4[i, -1] > 0.012:
            ax.text(ts[-1] * 1.005, mid, LG_ORDER[i],
                     fontsize=7, va="center", ha="left",
                     color=mcolors.hsv_to_rgb([hue_pair(i, i), 0.95, 0.45]))
        cs += float(band_k4[i, -1])


def kl_per_t_axes(ax, kl_per_t: np.ndarray,
                    kl_by_i: np.ndarray,
                    ts: np.ndarray, n: int,
                    title: str):
    """Per-time KL[K=4 || independent] as a stacked-area chart over
    all 400 AA-pair cells.

    Each (i, j) cell contributes a band whose RELATIVE height equals
    |kl_term_{ij}(t)| = |q_{K=4}(i,j;t) log(q_{K=4}(i,j;t)/q_indep(i,j;t))|;
    band heights are then renormalized at each t so the total stacked
    height equals the signed per-time KL (= sum of signed kl_terms).
    Bands are coloured by the AA-pair hue (hue_pair(i, j)). The signed
    KL curve is overlaid in black for reference."""
    kl_cell = np.where(np.isfinite(kl_by_i), kl_by_i, 0.0)
    abs_kl = np.abs(kl_cell)
    total_abs = abs_kl.sum(axis=0)
    safe_total = np.where(total_abs > 1e-12, total_abs, 1.0)
    scale = np.where(total_abs > 1e-12, kl_per_t / safe_total, 0.0)
    band_h = abs_kl * scale[None, :]
    cumsum = np.zeros_like(ts)
    n_cells = kl_cell.shape[0]
    for k in range(n_cells):
        if band_h[k].max() < 1e-12:
            continue
        i, j = k // n, k % n
        hue = hue_pair(i, j)
        ax.fill_between(ts, cumsum, cumsum + band_h[k],
                         color=mcolors.hsv_to_rgb([hue, 0.75, 0.85]),
                         edgecolor="none", linewidth=0.0)
        cumsum = cumsum + band_h[k]
    ax.plot(ts, kl_per_t, 'k-', linewidth=1.6, zorder=10, alpha=0.85,
             label=r"total $\mathrm{KL}[\mathrm{K{=}4}\,\|\,\mathrm{indep}]$")
    ax.axhline(0, color='black', linewidth=0.5, zorder=5)
    ax.set_xlim(ts[0], ts[-1])
    ax.set_xlabel("time $t$")
    ax.set_ylabel("nats / pair (per-AA-pair stacked)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper right', fontsize=8, frameon=False)


def make_paper_figure(label: str = "PG_to_DY",
                       from_pair: str = "PG", to_pair: str = "DY",
                       atom_h: int = 0, class_pair: Tuple[int, int] = (0, 2),
                       T: float = 2.0, n_t: int = 200,
                       out_dir: Path = OUT_DIR,
                       k4: dict | None = None,
                       story: str = "Coevolution under K=4 atom") -> Path:
    """Six-row paper figure for one bridge example.

    Rows: marginal site 1, marginal site 2, joint product (binned),
    joint K=4 (binned), K=4 with bright excess-over-product cap,
    per-time KL[K=4 || independent] decomposed by first coord.
    """
    if k4 is None:
        k4 = load_k4()
    Q_lg, pi_lg = load_lg_in_lg_order()
    n = Q_lg.shape[0]
    ts = np.linspace(0.0, T, n_t)

    a0 = aa_to_lg_idx(from_pair[0]); aT = aa_to_lg_idx(to_pair[0])
    b0 = aa_to_lg_idx(from_pair[1]); bT = aa_to_lg_idx(to_pair[1])

    c, cp = class_pair
    pi_class = k4["pi_class"]
    H = k4["atoms"][atom_h]

    Q_a = per_class_rate_matrix(Q_lg, pi_lg, pi_class[c])
    Q_b = per_class_rate_matrix(Q_lg, pi_lg, pi_class[cp])
    qa = bridge_marginal(Q_a, a0, aT, ts, T)
    qb = bridge_marginal(Q_b, b0, bT, ts, T)
    qjoint_ind = (qa[:, None, :] * qb[None, :, :]).reshape(n * n, n_t)
    Q_co = build_k4_coevol_rate_matrix(Q_a, Q_b, pi_class[c], pi_class[cp], H)
    qjoint_k4 = bridge_joint_k4(Q_co, n, a0, aT, b0, bT, ts, T)

    eps = 1e-30
    kl_cell = qjoint_k4 * np.log((qjoint_k4 + eps) / (qjoint_ind + eps))
    kl_cell = np.where(np.isfinite(kl_cell), kl_cell, 0.0)
    kl_per_t = kl_cell.sum(axis=0)

    fig, axes = plt.subplots(5, 1, figsize=(8.5, 16))
    fig.suptitle(
        f"({from_pair[0]},{from_pair[1]}) "
        rf"$\to$ ({to_pair[0]},{to_pair[1]}) "
        f"under K=4 atom {atom_h} (class-pair {class_pair}) — {story}",
        fontsize=11)
    stacked_marginal_axes(axes[0], qa, ts,
                            f"Marginal site 1: {from_pair[0]} $\\to$ {to_pair[0]} "
                            f"(per-class {c})")
    stacked_marginal_axes(axes[1], qb, ts,
                            f"Marginal site 2: {from_pair[1]} $\\to$ {to_pair[1]} "
                            f"(per-class {cp})")
    stacked_joint_axes(axes[2], qjoint_ind, ts,
                        "Joint product of marginals (independent-sites approx.)",
                        bin_by_i=True)
    stacked_joint_axes(axes[3], qjoint_k4, ts,
                        f"Joint under K=4 coevolution (atom {atom_h})",
                        bin_by_i=True)
    kl_per_t_axes(axes[4], kl_per_t, kl_cell, ts, n,
                    "Per-time KL[K=4 || independent] "
                    "(coevolution signal absent from product approx.)")
    plt.tight_layout(rect=[0, 0, 0.97, 0.97])
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"aa_evolution_{label}_paper.pdf"
    png = out_dir / f"aa_evolution_{label}_paper.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}")
    return pdf


def make_stacked_figure(label: str, from_pair: str, to_pair: str,
                         atom_h: int, class_pair: Tuple[int, int],
                         T: float = 2.0, n_t: int = 200,
                         bin_joint: bool = True,
                         out_dir: Path = OUT_DIR,
                         k4: dict | None = None,
                         story: str = "") -> Path:
    """Render one 4-panel STACKED-AREA figure."""
    if k4 is None:
        k4 = load_k4()
    Q_lg, pi_lg = load_lg_in_lg_order()
    ts = np.linspace(0.0, T, n_t)

    a0 = aa_to_lg_idx(from_pair[0]); aT = aa_to_lg_idx(to_pair[0])
    b0 = aa_to_lg_idx(from_pair[1]); bT = aa_to_lg_idx(to_pair[1])

    c, cp = class_pair
    pi_class = k4["pi_class"]
    H = k4["atoms"][atom_h]

    Q_a = per_class_rate_matrix(Q_lg, pi_lg, pi_class[c])
    Q_b = per_class_rate_matrix(Q_lg, pi_lg, pi_class[cp])

    qa = bridge_marginal(Q_a, a0, aT, ts, T)
    qb = bridge_marginal(Q_b, b0, bT, ts, T)
    n = qa.shape[0]
    qjoint_ind = np.empty((n * n, ts.size))
    for s in range(ts.size):
        qjoint_ind[:, s] = np.outer(qa[:, s], qb[:, s]).reshape(-1)
    Q_co = build_k4_coevol_rate_matrix(Q_a, Q_b, pi_class[c], pi_class[cp], H)
    qjoint_k4 = bridge_joint_k4(Q_co, n, a0, aT, b0, bT, ts, T)

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 13))
    fig.suptitle(
        f"({from_pair[0]},{from_pair[1]}) "
        rf"$\to$ ({to_pair[0]},{to_pair[1]}) "
        f"under K=4 atom {atom_h} (class-pair {class_pair}) — {story}\n"
        f"[stacked-area rendering]",
        fontsize=11)
    stacked_marginal_axes(axes[0], qa, ts,
                            f"Marginal {from_pair[0]} $\\to$ {to_pair[0]}")
    stacked_marginal_axes(axes[1], qb, ts,
                            f"Marginal {from_pair[1]} $\\to$ {to_pair[1]}")
    stacked_joint_axes(axes[2], qjoint_ind, ts,
                        "Joint product of marginals "
                        f"(binned by 1st coord)" if bin_joint
                        else "Joint product of marginals (all 400)",
                        bin_by_i=bin_joint)
    stacked_joint_axes(axes[3], qjoint_k4, ts,
                        f"Joint under K=4 (atom {atom_h})"
                        + (" — binned by 1st coord" if bin_joint else ""),
                        bin_by_i=bin_joint)
    plt.tight_layout(rect=[0, 0, 0.97, 0.97])
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"aa_evolution_{label}_stacked.pdf"
    png = out_dir / f"aa_evolution_{label}_stacked.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}")
    return pdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", action="store_true",
                   help="Render all 5 canonical examples as stacked-area.")
    p.add_argument("--paper", action="store_true",
                   help="Render the 6-row PG->DY paper figure.")
    p.add_argument("--no-bin", action="store_true",
                   help="Disable binning of the joint axis (draws all 400 bands).")
    p.add_argument("--label", default=None)
    p.add_argument("--from-pair", default=None, dest="from_pair")
    p.add_argument("--to-pair", default=None, dest="to_pair")
    p.add_argument("--atom", type=int, default=None)
    p.add_argument("--class-pair", default=None, dest="class_pair")
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--n-t", type=int, default=200)
    args = p.parse_args()

    k4 = load_k4()
    bin_joint = not args.no_bin

    if args.paper:
        # Allow override of label/from/to/atom/class-pair while keeping
        # PG->DY defaults.
        label = args.label or "PG_to_DY"
        fp = args.from_pair or "PG"
        tp = args.to_pair or "DY"
        atom_h = args.atom if args.atom is not None else 0
        if args.class_pair:
            c, cp = (int(x) for x in args.class_pair.split(","))
        else:
            c, cp = (0, 2)
        make_paper_figure(label=label, from_pair=fp, to_pair=tp,
                          atom_h=atom_h, class_pair=(c, cp),
                          T=args.T, n_t=args.n_t, k4=k4)
        return 0

    if args.batch:
        for label, fp, tp, atom_h, cls, T, story in CANONICAL_EXAMPLES:
            make_stacked_figure(label, fp, tp, atom_h, cls,
                                 T=T, n_t=args.n_t,
                                 bin_joint=bin_joint,
                                 k4=k4, story=story)
        return 0

    if not (args.label and args.from_pair and args.to_pair
            and args.atom is not None and args.class_pair):
        p.error("Either --batch, or all of --label/--from-pair/--to-pair/"
                "--atom/--class-pair must be given")
    c, cp = (int(x) for x in args.class_pair.split(","))
    make_stacked_figure(args.label, args.from_pair, args.to_pair,
                         args.atom, (c, cp),
                         T=args.T, n_t=args.n_t,
                         bin_joint=bin_joint,
                         k4=k4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
