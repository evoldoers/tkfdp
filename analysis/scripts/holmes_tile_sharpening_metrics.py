#!/usr/bin/env python3
"""Compute sharpening metrics for one Holmes-tile cache JSON.

Per-row mean Shannon entropy of a posterior row Q[i, :]:

  H_i  = - sum_j ( q_ij / s_i ) * log( q_ij / s_i )    with s_i = sum_j q_ij.

Reported in nats. If a row sums to zero (all entries = 0), it is excluded.

C-C enrichment: average of P_joint over (Cys_i, Cys_j) pairs vs over all
unordered off-diagonal pairs (i < j).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _row_entropies(M: np.ndarray) -> np.ndarray:
    """Per-row Shannon entropy in nats. M is non-negative."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {M.shape}")
    s = M.sum(axis=1, keepdims=True)
    ok = (s.squeeze(-1) > 0)
    P = np.zeros_like(M)
    np.divide(M, s, out=P, where=(s > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_P = np.where(P > 0, np.log(P), 0.0)
    H = -(P * log_P).sum(axis=1)
    H[~ok] = np.nan
    return H


def _cc_enrichment(P: np.ndarray, cys: list[int], L: int):
    """C-C-cell mean over the joint upper-triangle (i<j) divided by the
    mean over ALL off-diagonal (i<j) pairs in 1..L.
    Returns (cc_mean, bg_mean, ratio).
    P is the 1-indexed (L+1, L+1) matrix produced by plot_holmes_tile.
    """
    cc_vals = []
    for a in range(len(cys)):
        for b in range(a + 1, len(cys)):
            i, j = cys[a], cys[b]
            if 1 <= i <= L and 1 <= j <= L:
                cc_vals.append(P[i, j])
    bg_vals = []
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            bg_vals.append(P[i, j])
    cc_mean = float(np.mean(cc_vals)) if cc_vals else 0.0
    bg_mean = float(np.mean(bg_vals)) if bg_vals else 0.0
    ratio = cc_mean / bg_mean if bg_mean > 0 else float("inf")
    return cc_mean, bg_mean, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True,
                    help="Holmes-tile cache JSON")
    ap.add_argument("--label", default=None,
                    help="Label prepended to printed lines")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Optional path to write metrics dict as JSON")
    args = ap.parse_args()

    d = json.loads(args.cache.read_text())
    label = args.label or d.get("family", args.cache.stem)
    Lx = int(d["Lx"]); Ly = int(d["Ly"])
    cys_x = [int(c) for c in d["cys_x"]]
    cys_y = [int(c) for c in d["cys_y"]]
    tau = float(d.get("tau", 0.0))

    Q_baseline = np.asarray(d["Q_baseline"])
    Q_prime = np.asarray(d["Q_prime"])
    Pxx_single = np.asarray(d["Pxx_single"])
    Pxx_joint = np.asarray(d["Pxx_joint"])
    Pyy_single = np.asarray(d["Pyy_single"])
    Pyy_joint = np.asarray(d["Pyy_joint"])

    # Sharpening: per-row entropy mean (averaged over rows that have any mass).
    Hbase = float(np.nanmean(_row_entropies(Q_baseline)))
    Hprime = float(np.nanmean(_row_entropies(Q_prime)))
    Hxx_s = float(np.nanmean(_row_entropies(Pxx_single[1:Lx + 1, 1:Lx + 1])))
    Hxx_j = float(np.nanmean(_row_entropies(Pxx_joint[1:Lx + 1, 1:Lx + 1])))
    Hyy_s = float(np.nanmean(_row_entropies(Pyy_single[1:Ly + 1, 1:Ly + 1])))
    Hyy_j = float(np.nanmean(_row_entropies(Pyy_joint[1:Ly + 1, 1:Ly + 1])))

    # C-C enrichment in panels (d) and (e).
    cc_xx, bg_xx, r_xx = _cc_enrichment(Pxx_joint, cys_x, Lx)
    cc_yy, bg_yy, r_yy = _cc_enrichment(Pyy_joint, cys_y, Ly)
    # Same for the SINGLE panels (a)/(b) baseline for sanity.
    cc_xx_s, bg_xx_s, r_xx_s = _cc_enrichment(Pxx_single, cys_x, Lx)
    cc_yy_s, bg_yy_s, r_yy_s = _cc_enrichment(Pyy_single, cys_y, Ly)

    print(f"=== {label} ===")
    print(f"  Lx={Lx} Ly={Ly}  tau={tau:.3f}  cys_x={cys_x} cys_y={cys_y}")
    print(f"  ALIGN per-row entropy (nats):  baseline={Hbase:.4f}  "
          f"joint={Hprime:.4f}  d_H={Hprime - Hbase:+.4f} "
          f"({(Hprime - Hbase) / max(Hbase, 1e-9) * 100:+.2f}%)")
    print(f"  X-X   per-row entropy (nats):  single  ={Hxx_s:.4f}  "
          f"joint={Hxx_j:.4f}  d_H={Hxx_j - Hxx_s:+.4f} "
          f"({(Hxx_j - Hxx_s) / max(Hxx_s, 1e-9) * 100:+.2f}%)")
    print(f"  Y-Y   per-row entropy (nats):  single  ={Hyy_s:.4f}  "
          f"joint={Hyy_j:.4f}  d_H={Hyy_j - Hyy_s:+.4f} "
          f"({(Hyy_j - Hyy_s) / max(Hyy_s, 1e-9) * 100:+.2f}%)")
    print(f"  X-X JOINT  CC={cc_xx:.5f}  bg={bg_xx:.5f}  ratio={r_xx:.3f}x")
    print(f"  X-X SINGLE CC={cc_xx_s:.5f}  bg={bg_xx_s:.5f}  ratio={r_xx_s:.3f}x")
    print(f"  Y-Y JOINT  CC={cc_yy:.5f}  bg={bg_yy:.5f}  ratio={r_yy:.3f}x")
    print(f"  Y-Y SINGLE CC={cc_yy_s:.5f}  bg={bg_yy_s:.5f}  ratio={r_yy_s:.3f}x")

    metrics = {
        "label": label, "Lx": Lx, "Ly": Ly, "tau": tau,
        "cys_x": cys_x, "cys_y": cys_y,
        "H_align_baseline": Hbase, "H_align_joint": Hprime,
        "H_xx_single": Hxx_s, "H_xx_joint": Hxx_j,
        "H_yy_single": Hyy_s, "H_yy_joint": Hyy_j,
        "cc_xx_joint": cc_xx, "bg_xx_joint": bg_xx, "ratio_xx_joint": r_xx,
        "cc_xx_single": cc_xx_s, "bg_xx_single": bg_xx_s,
        "ratio_xx_single": r_xx_s,
        "cc_yy_joint": cc_yy, "bg_yy_joint": bg_yy, "ratio_yy_joint": r_yy,
        "cc_yy_single": cc_yy_s, "bg_yy_single": bg_yy_s,
        "ratio_yy_single": r_yy_s,
    }
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2))
        print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    main()
