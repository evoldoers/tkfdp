#!/usr/bin/env python3
"""Print side-by-side PDB-contact sharpening metrics for the
holmes_tile_pdbedge_pair_az{100,30} runs.

Calls holmes_tile_pdb_contact_metrics for each cache and prints a
compact side-by-side table.

Usage:
    /home/yam/tkf-mixdom/python/.venv/bin/python \\
        analysis/scripts/holmes_tile_pdb_contact_summary.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "scripts"))
from holmes_tile_pdb_contact_metrics import (
    _row_entropies, _mean_at_pairs, _mean_off_contact_pairs,
)


def metrics_for_cache(cache_path: Path, pair_info: dict) -> dict:
    cache = json.loads(cache_path.read_text())
    Lx = int(cache["Lx"]); Ly = int(cache["Ly"])
    tau = float(cache.get("tau", 0.0))
    cache_family = cache.get("family")
    cache_pair = tuple(cache.get("pair", [-1, -1]))

    record = None
    candidates = []
    if pair_info.get("winner"):
        candidates.append(pair_info["winner"])
    candidates += pair_info.get("top", [])
    for r in candidates:
        if r.get("family") == cache_family and tuple(r.get("pair", [])) == cache_pair:
            record = r
            break
    if record is None:
        record = pair_info.get("winner")
    if record is None:
        raise RuntimeError("No pair record")
    qualifying = record["qualifying"]
    pairs_x = [(int(q["u_i_a"]), int(q["u_i_b"])) for q in qualifying]
    pairs_y = [(int(q["u_j_a"]), int(q["u_j_b"])) for q in qualifying]
    pairs_x_set = {(min(a, b), max(a, b)) for a, b in pairs_x}
    pairs_y_set = {(min(a, b), max(a, b)) for a, b in pairs_y}

    Q_baseline = np.asarray(cache["Q_baseline"])
    Q_prime = np.asarray(cache["Q_prime"])
    Pxx_single = np.asarray(cache["Pxx_single"])
    Pxx_joint = np.asarray(cache["Pxx_joint"])
    Pyy_single = np.asarray(cache["Pyy_single"])
    Pyy_joint = np.asarray(cache["Pyy_joint"])

    Hbase = float(np.nanmean(_row_entropies(Q_baseline)))
    Hprime = float(np.nanmean(_row_entropies(Q_prime)))
    Hxx_s = float(np.nanmean(_row_entropies(Pxx_single[1:Lx + 1, 1:Lx + 1])))
    Hxx_j = float(np.nanmean(_row_entropies(Pxx_joint[1:Lx + 1, 1:Lx + 1])))
    Hyy_s = float(np.nanmean(_row_entropies(Pyy_single[1:Ly + 1, 1:Ly + 1])))
    Hyy_j = float(np.nanmean(_row_entropies(Pyy_joint[1:Ly + 1, 1:Ly + 1])))

    M_xx_j = _mean_at_pairs(Pxx_joint, pairs_x, Lx)
    M_xx_s = _mean_at_pairs(Pxx_single, pairs_x, Lx)
    M_yy_j = _mean_at_pairs(Pyy_joint, pairs_y, Ly)
    M_yy_s = _mean_at_pairs(Pyy_single, pairs_y, Ly)
    B_xx_j = _mean_off_contact_pairs(Pxx_joint, pairs_x_set, Lx)
    B_xx_s = _mean_off_contact_pairs(Pxx_single, pairs_x_set, Lx)
    B_yy_j = _mean_off_contact_pairs(Pyy_joint, pairs_y_set, Ly)
    B_yy_s = _mean_off_contact_pairs(Pyy_single, pairs_y_set, Ly)

    def sd(a, b):
        if b is None or b == 0 or np.isnan(b):
            return float("nan")
        return a / b

    return {
        "family": cache_family, "pair": list(cache_pair),
        "Lx": Lx, "Ly": Ly, "tau": tau,
        "n_qualifying": len(qualifying),
        "Hbase": Hbase, "Hprime": Hprime,
        "Hxx_s": Hxx_s, "Hxx_j": Hxx_j,
        "Hyy_s": Hyy_s, "Hyy_j": Hyy_j,
        "M_xx_j": M_xx_j, "M_xx_s": M_xx_s,
        "B_xx_j": B_xx_j, "B_xx_s": B_xx_s,
        "M_yy_j": M_yy_j, "M_yy_s": M_yy_s,
        "B_yy_j": B_yy_j, "B_yy_s": B_yy_s,
        "ratio_xx_jvs": sd(M_xx_j, M_xx_s),
        "ratio_yy_jvs": sd(M_yy_j, M_yy_s),
        "ratio_xx_jb": sd(M_xx_j, B_xx_j),
        "ratio_yy_jb": sd(M_yy_j, B_yy_j),
        "ratio_xx_sb": sd(M_xx_s, B_xx_s),
        "ratio_yy_sb": sd(M_yy_s, B_yy_s),
        "joint_nrec": cache.get("joint_nrec"),
        "single_x_nrec": cache.get("single_x_nrec"),
        "single_y_nrec": cache.get("single_y_nrec"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", type=Path,
                    default=[
                        REPO / "math-paper" / "figures" /
                        "holmes_tile_pdbedge_pair_az100_cache.json",
                        REPO / "math-paper" / "figures" /
                        "holmes_tile_pdbedge_pair_az30_cache.json",
                    ])
    ap.add_argument("--labels", nargs="+", default=["az100", "az30"])
    ap.add_argument("--pair-info", type=Path,
                    default=REPO / "math-paper" / "figures" /
                    "holmes_tile_pdbedge_pair_selection.json")
    ap.add_argument("--out-json", type=Path,
                    default=REPO / "math-paper" / "figures" /
                    "holmes_tile_pdbedge_summary.json")
    args = ap.parse_args()

    pair_info = json.loads(args.pair_info.read_text())
    rows = []
    for label, cache in zip(args.labels, args.caches):
        if not cache.exists():
            print(f"MISSING: {cache}")
            continue
        m = metrics_for_cache(cache, pair_info)
        m["label"] = label
        m["cache_path"] = str(cache)
        rows.append(m)

    if not rows:
        print("no caches found")
        return

    # Print details first.
    for m in rows:
        print(f"=== {m['label']}  ({m['family']} pair=({m['pair'][0]},"
              f"{m['pair'][1]}), tau={m['tau']:.3f}, "
              f"L={m['Lx']}x{m['Ly']}) ===")
        print(f"  joint_nrec={m['joint_nrec']}, "
              f"single_x_nrec={m['single_x_nrec']}, "
              f"single_y_nrec={m['single_y_nrec']}, "
              f"qualifying={m['n_qualifying']}")
        print(f"  ALIGN per-row entropy: base={m['Hbase']:.4f} "
              f"joint={m['Hprime']:.4f} "
              f"dH={m['Hprime']-m['Hbase']:+.4f} "
              f"({(m['Hprime']-m['Hbase'])/max(m['Hbase'],1e-9)*100:+.2f}%)")
        print(f"  X-X   per-row entropy: single={m['Hxx_s']:.4f} "
              f"joint={m['Hxx_j']:.4f} dH={m['Hxx_j']-m['Hxx_s']:+.4f} "
              f"({(m['Hxx_j']-m['Hxx_s'])/max(m['Hxx_s'],1e-9)*100:+.2f}%)")
        print(f"  Y-Y   per-row entropy: single={m['Hyy_s']:.4f} "
              f"joint={m['Hyy_j']:.4f} dH={m['Hyy_j']-m['Hyy_s']:+.4f} "
              f"({(m['Hyy_j']-m['Hyy_s'])/max(m['Hyy_s'],1e-9)*100:+.2f}%)")
        print(f"  PDB-contact X-X (n={m['n_qualifying']}): "
              f"M_joint={m['M_xx_j']:.5f}, M_single={m['M_xx_s']:.5f}, "
              f"B_joint={m['B_xx_j']:.5f}")
        print(f"             joint/single={m['ratio_xx_jvs']:.3f}x  "
              f"joint_sig/bg={m['ratio_xx_jb']:.3f}x  "
              f"single_sig/bg={m['ratio_xx_sb']:.3f}x")
        print(f"  PDB-contact Y-Y (n={m['n_qualifying']}): "
              f"M_joint={m['M_yy_j']:.5f}, M_single={m['M_yy_s']:.5f}, "
              f"B_joint={m['B_yy_j']:.5f}")
        print(f"             joint/single={m['ratio_yy_jvs']:.3f}x  "
              f"joint_sig/bg={m['ratio_yy_jb']:.3f}x  "
              f"single_sig/bg={m['ratio_yy_sb']:.3f}x")
        print()

    # Compact side-by-side table.
    print("=" * 90)
    print(f"{'label':<8} {'tau':>5} {'Hbase':>6} {'Hprime':>6} {'dH%':>6} "
          f"{'dHxx':>7} {'dHyy':>7} {'r_xx_jvs':>9} {'r_xx_jb':>8} "
          f"{'r_yy_jvs':>9} {'r_yy_jb':>8}")
    print("-" * 90)
    for m in rows:
        dH_pct = (m['Hprime']-m['Hbase'])/max(m['Hbase'],1e-9)*100
        dHxx = m['Hxx_j']-m['Hxx_s']
        dHyy = m['Hyy_j']-m['Hyy_s']
        print(f"{m['label']:<8} {m['tau']:>5.2f} {m['Hbase']:>6.4f} "
              f"{m['Hprime']:>6.4f} {dH_pct:>+6.1f} "
              f"{dHxx:>+7.4f} {dHyy:>+7.4f} "
              f"{m['ratio_xx_jvs']:>9.3f} {m['ratio_xx_jb']:>8.3f} "
              f"{m['ratio_yy_jvs']:>9.3f} {m['ratio_yy_jb']:>8.3f}")
    print("=" * 90)
    print("Legend:  Hbase = TKF92 alignment-row entropy (nats)")
    print("         Hprime = joint InfPHMM alignment-row entropy (nats)")
    print("         dH%   = relative reduction (Hprime-Hbase)/Hbase")
    print("         dHxx/dHyy = joint vs single edge-row entropy diff (nats)")
    print("         r_xx_jvs = mean P_joint / mean P_single at PDB-contact pairs (X)")
    print("         r_xx_jb  = signal-to-background = mean P_joint at PDB / off-contact bg (X)")
    print("         Similar for Y axis.")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({"rows": rows}, indent=2))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
