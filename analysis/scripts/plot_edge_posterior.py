#!/usr/bin/env python3
"""Plot edge marginal posterior vs PDB contacts.

Reads a sweep_infinite_phmm_balibase.py JSON containing the new
mcmc_diag fields ``edge_pos_x_counts``, ``edge_pos_y_counts``,
``edge_cell_counts`` (cold-rung of replica-exchange path or the
single-chain path).

Produces a 3-panel figure for one pair:

  (left)   edge_pos posterior on X-sequence (1D bar), cysteine
           positions highlighted.
  (middle) edge_pos posterior on Y-sequence (1D bar), cys highlighted.
  (right)  per-cell edge posterior heatmap (Lx x Ly) overlaid with
           C-C cells (cysteine-cysteine BAliBASE residue pairs)
           and, if available, PDB-derived C-C contacts under the
           threshold.

Usage:
    python analysis/scripts/plot_edge_posterior.py \
        /tmp/test_edge_posterior_BB12032.json \
        --pair 0 \
        --out math-paper/figures/edge_posterior_BB12032_p0.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _cys_positions(seq: str) -> list[int]:
    return [i + 1 for i, a in enumerate(seq) if a == "C"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="sweep_infinite_phmm_balibase JSON")
    ap.add_argument("--family-idx", type=int, default=0)
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq-i", type=str, default=None,
                    help="X sequence (raw, no gaps). If absent, "
                         "looked up from the bali3pdbm 'in/' file.")
    ap.add_argument("--seq-j", type=str, default=None,
                    help="Y sequence (raw).")
    ap.add_argument("--balibase-root", type=str,
                    default="/home/yam/bio-datasets/data/balibase/bali3pdbm")
    args = ap.parse_args()

    j = json.loads(Path(args.json).read_text())
    fam = j["per_family"][args.family_idx]
    pair = fam["per_pair"][args.pair]
    diag = pair["mcmc_diag"]
    pc0 = diag["per_chain"][0]
    nrec = pc0["n_recorded_for_edges"]
    Lx = pair["len_i"]
    Ly = pair["len_j"]

    # Build sequences (auto-load if needed).
    if args.seq_i and args.seq_j:
        seq_x = args.seq_i
        seq_y = args.seq_j
    else:
        # Parse bali3pdbm in/<family> FASTA-ish.
        fam_name = fam["family"]
        fasta = Path(args.balibase_root) / "in" / fam_name
        lines = fasta.read_text().splitlines()
        entries = {}
        cur_name, cur_seq = None, []
        for ln in lines:
            if ln.startswith(">"):
                if cur_name is not None:
                    entries[cur_name] = "".join(cur_seq).replace("-", "")
                cur_name = ln[1:].strip()
                cur_seq = []
            else:
                cur_seq.append(ln.strip())
        if cur_name is not None:
            entries[cur_name] = "".join(cur_seq).replace("-", "")
        seq_x = entries[pair["name_i"]]
        seq_y = entries[pair["name_j"]]
    assert len(seq_x) == Lx, f"len(seq_x)={len(seq_x)} != Lx={Lx}"
    assert len(seq_y) == Ly, f"len(seq_y)={len(seq_y)} != Ly={Ly}"

    cys_x = _cys_positions(seq_x)
    cys_y = _cys_positions(seq_y)

    # Per-position marginal: divide by 2*nrec because each edge
    # contributes TWO endpoint counts to X (and TWO to Y).
    px = np.array(pc0["edge_pos_x_counts"][1:]) / (2 * nrec)
    py = np.array(pc0["edge_pos_y_counts"][1:]) / (2 * nrec)
    # Cell marginal: each cell appears either 0, 1, or 2 times per
    # sweep (an edge has two endpoints; only one can be at a given
    # cell). Divide by nrec.
    cell_mat = np.zeros((Lx, Ly), dtype=np.float64)
    for (i, j_idx, c) in pc0["edge_cell_counts"]:
        if 1 <= i <= Lx and 1 <= j_idx <= Ly:
            cell_mat[i - 1, j_idx - 1] = c / nrec

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.8])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    name_i = pair.get("name_i", "X")
    name_j = pair.get("name_j", "Y")
    fig.suptitle(
        f"Edge marginal posterior — {fam['family']} pair {args.pair}: "
        f"{name_i} (Lx={Lx}) vs {name_j} (Ly={Ly}), "
        f"n_recorded={nrec}",
        fontsize=11)

    # Panel 0: X positions
    ax0.bar(np.arange(1, Lx + 1), px, color="#4477AA",
            edgecolor="none", width=0.9)
    for ci in cys_x:
        ax0.axvline(ci, color="#EE6677", lw=1.2, alpha=0.55,
                    zorder=0)
    ax0.set_xlabel(f"position on {name_i}")
    ax0.set_ylabel("P(position is edge endpoint | data)")
    ax0.set_title("X (red = Cys)")
    ax0.set_xlim(0.5, Lx + 0.5)

    # Panel 1: Y positions
    ax1.bar(np.arange(1, Ly + 1), py, color="#228833",
            edgecolor="none", width=0.9)
    for cj in cys_y:
        ax1.axvline(cj, color="#EE6677", lw=1.2, alpha=0.55,
                    zorder=0)
    ax1.set_xlabel(f"position on {name_j}")
    ax1.set_ylabel("")
    ax1.set_title("Y (red = Cys)")
    ax1.set_xlim(0.5, Ly + 0.5)

    # Panel 2: cell heatmap + cysteine grid + C-C overlay
    im = ax2.imshow(cell_mat, origin="lower", aspect="auto",
                    extent=(0.5, Ly + 0.5, 0.5, Lx + 0.5),
                    cmap="magma_r",
                    vmin=0.0, vmax=max(0.05, cell_mat.max()))
    # Cysteine lines
    for ci in cys_x:
        ax2.axhline(ci, color="#EE6677", lw=0.4, alpha=0.30, zorder=1)
    for cj in cys_y:
        ax2.axvline(cj, color="#EE6677", lw=0.4, alpha=0.30, zorder=1)
    # Mark every C-C lattice intersection.
    for ci in cys_x:
        for cj in cys_y:
            ax2.plot([cj], [ci], "o", markerfacecolor="none",
                     markeredgecolor="#EE6677", markersize=8,
                     markeredgewidth=1.2)
    ax2.set_xlabel(f"position on {name_j}")
    ax2.set_ylabel(f"position on {name_i}")
    ax2.set_title("per-cell edge posterior  +  C-C grid (red circles)")
    plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02,
                 label="P(cell is edge endpoint | data)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=140,
                bbox_inches="tight")
    print(f"Wrote {args.out} + {args.out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
