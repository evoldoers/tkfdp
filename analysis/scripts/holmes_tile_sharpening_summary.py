#!/usr/bin/env python3
"""Run holmes_tile_sharpening_metrics on all four caches and print a
concise side-by-side table.

Usage:
    python analysis/scripts/holmes_tile_sharpening_summary.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "math-paper" / "figures"


def _row_entropies(M):
    M = np.asarray(M, dtype=np.float64)
    s = M.sum(axis=1, keepdims=True)
    ok = (s.squeeze(-1) > 0)
    P = np.zeros_like(M)
    np.divide(M, s, out=P, where=(s > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_P = np.where(P > 0, np.log(P), 0.0)
    H = -(P * log_P).sum(axis=1)
    H[~ok] = np.nan
    return H


def _cc_enrichment(P, cys, L):
    cc = []
    for a in range(len(cys)):
        for b in range(a + 1, len(cys)):
            if 1 <= cys[a] <= L and 1 <= cys[b] <= L:
                cc.append(P[cys[a], cys[b]])
    bg = []
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            bg.append(P[i, j])
    cm = float(np.mean(cc)) if cc else 0.0
    bm = float(np.mean(bg)) if bg else 0.0
    return cm, bm, (cm / bm if bm > 0 else float("inf"))


def metrics_for_cache(path: Path) -> dict:
    d = json.loads(path.read_text())
    Lx = int(d["Lx"]); Ly = int(d["Ly"])
    cys_x = [int(c) for c in d["cys_x"]]
    cys_y = [int(c) for c in d["cys_y"]]
    Qb = np.asarray(d["Q_baseline"]); Qp = np.asarray(d["Q_prime"])
    Pxs = np.asarray(d["Pxx_single"]); Pxj = np.asarray(d["Pxx_joint"])
    Pys = np.asarray(d["Pyy_single"]); Pyj = np.asarray(d["Pyy_joint"])
    Hbase = float(np.nanmean(_row_entropies(Qb)))
    Hprime = float(np.nanmean(_row_entropies(Qp)))
    Hxx_s = float(np.nanmean(_row_entropies(Pxs[1:Lx + 1, 1:Lx + 1])))
    Hxx_j = float(np.nanmean(_row_entropies(Pxj[1:Lx + 1, 1:Lx + 1])))
    Hyy_s = float(np.nanmean(_row_entropies(Pys[1:Ly + 1, 1:Ly + 1])))
    Hyy_j = float(np.nanmean(_row_entropies(Pyj[1:Ly + 1, 1:Ly + 1])))
    cc_xx, bg_xx, r_xx = _cc_enrichment(Pxj, cys_x, Lx)
    cc_yy, bg_yy, r_yy = _cc_enrichment(Pyj, cys_y, Ly)
    return {
        "label": d.get("family", path.stem),
        "Lx": Lx, "Ly": Ly,
        "tau": float(d.get("tau", 0.0)),
        "H_align_baseline": Hbase, "H_align_joint": Hprime,
        "H_xx_single": Hxx_s, "H_xx_joint": Hxx_j,
        "H_yy_single": Hyy_s, "H_yy_joint": Hyy_j,
        "cc_xx_joint": cc_xx, "bg_xx_joint": bg_xx, "ratio_xx_joint": r_xx,
        "cc_yy_joint": cc_yy, "bg_yy_joint": bg_yy, "ratio_yy_joint": r_yy,
    }


def main():
    tags = ["close_az30", "close_az100", "distant_az30", "distant_az100"]
    rows = []
    for tag in tags:
        path = FIG / f"holmes_tile_PF00014_{tag}_cache.json"
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        m = metrics_for_cache(path)
        m["tag"] = tag
        rows.append(m)
        # Print one block per cache.
        print(f"=== {tag}  (tau={m['tau']:.3f}) ===")
        print(f"  ALIGN H:   base={m['H_align_baseline']:.4f}  "
              f"joint={m['H_align_joint']:.4f}  "
              f"dH={m['H_align_joint'] - m['H_align_baseline']:+.4f}  "
              f"(joint/base={m['H_align_joint']/max(m['H_align_baseline'],1e-9):.3f}x)")
        print(f"  X-X H:     single={m['H_xx_single']:.4f}  "
              f"joint={m['H_xx_joint']:.4f}  "
              f"dH={m['H_xx_joint'] - m['H_xx_single']:+.4f}")
        print(f"  Y-Y H:     single={m['H_yy_single']:.4f}  "
              f"joint={m['H_yy_joint']:.4f}  "
              f"dH={m['H_yy_joint'] - m['H_yy_single']:+.4f}")
        print(f"  CC ratio (d) X-X: {m['ratio_xx_joint']:.3f}x   "
              f"(cc={m['cc_xx_joint']:.5f} bg={m['bg_xx_joint']:.5f})")
        print(f"  CC ratio (e) Y-Y: {m['ratio_yy_joint']:.3f}x   "
              f"(cc={m['cc_yy_joint']:.5f} bg={m['bg_yy_joint']:.5f})")
        print()

    # Compact table.
    print()
    print(f"{'tag':<14}  {'tau':>5}  {'Hb_al':>6}  {'Hj_al':>6}  "
          f"{'dHal%':>6}  {'dHxx':>7}  {'dHyy':>7}  {'r_xx':>5}  {'r_yy':>5}")
    for m in rows:
        dHal_pct = (m['H_align_joint'] - m['H_align_baseline']) / max(m['H_align_baseline'], 1e-9) * 100
        dHxx = m['H_xx_joint'] - m['H_xx_single']
        dHyy = m['H_yy_joint'] - m['H_yy_single']
        print(f"{m['tag']:<14}  {m['tau']:>5.2f}  "
              f"{m['H_align_baseline']:>6.4f}  {m['H_align_joint']:>6.4f}  "
              f"{dHal_pct:>+6.1f}  {dHxx:>+7.4f}  {dHyy:>+7.4f}  "
              f"{m['ratio_xx_joint']:>5.2f}  {m['ratio_yy_joint']:>5.2f}")


if __name__ == "__main__":
    main()
