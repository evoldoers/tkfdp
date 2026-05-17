#!/usr/bin/env python3
"""Compute PDB-contact-pair sharpening metrics for a Holmes-tile cache.

Reads the cache JSON produced by plot_holmes_tile.py and reports, for
the SAME family pair, the joint vs single edge-pair posteriors at
qualifying PDB contact column-pairs (those identified by
select_pdb_edge_cov_pair.py).

Inputs:
  --cache : cache JSON from plot_holmes_tile.py.
  --pair-info JSON : a record from select_pdb_edge_cov_pair.py
      containing the 'winner' or 'top' list. Pulls qualifying contact
      pairs for the matching (family, pair) tuple.

Reports per-axis (X-X and Y-Y):
  - mean P_joint at qualifying contact column-pairs (call M_joint)
  - mean P_single at the same column-pairs (M_single)
  - mean P_joint at all OTHER off-diagonal pairs (B_joint, background)
  - mean P_single at all OTHER off-diagonal pairs (B_single)
  - ratio  M_joint / M_single  (joint over single, contact)
  - ratio  M_joint / B_joint   (signal-to-background in joint panel)
  - ratio  M_single / B_single (signal-to-background in single panel)

Plus the same per-row entropy / C-C enrichment summary that
holmes_tile_sharpening_metrics.py already prints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _row_entropies(M: np.ndarray) -> np.ndarray:
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


def _mean_at_pairs(P_1based: np.ndarray, pairs_0based: list[tuple[int, int]],
                    L: int) -> float:
    """Mean of P at the given pairs (input is 0-based, P is 1-based with
    shape (L+1, L+1))."""
    vals = []
    for i, j in pairs_0based:
        # Convert 0-based -> 1-based
        if i + 1 < 1 or i + 1 > L or j + 1 < 1 or j + 1 > L:
            continue
        if i + 1 == j + 1:
            continue
        # Symmetrize: take max(i, j) order so upper triangle is used.
        a, b = min(i + 1, j + 1), max(i + 1, j + 1)
        vals.append(P_1based[a, b])
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _mean_off_contact_pairs(P_1based: np.ndarray,
                             contact_pairs_0based: set[tuple[int, int]],
                             L: int) -> float:
    """Mean of P over all off-diagonal pairs (1..L, 1..L, i < j) that are
    NOT in the contact-pair set (treated as 0-based)."""
    contact_norm = set()
    for i, j in contact_pairs_0based:
        a, b = min(i, j), max(i, j)
        contact_norm.add((a, b))
    vals = []
    for i in range(L):
        for j in range(i + 1, L):
            if (i, j) in contact_norm:
                continue
            vals.append(P_1based[i + 1, j + 1])
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True,
                    help="Holmes-tile cache JSON (output of plot_holmes_tile.py)")
    ap.add_argument("--pair-info", type=Path, required=True,
                    help="JSON output of select_pdb_edge_cov_pair.py "
                    "(uses winner / first matching top entry)")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    cache = json.loads(args.cache.read_text())
    info = json.loads(args.pair_info.read_text())
    label = args.label or cache.get("family", args.cache.stem)
    Lx = int(cache["Lx"]); Ly = int(cache["Ly"])
    tau = float(cache.get("tau", 0.0))
    cys_x = [int(c) for c in cache.get("cys_x", [])]
    cys_y = [int(c) for c in cache.get("cys_y", [])]

    # Match family + pair from info.
    cache_family = cache.get("family")
    cache_pair = tuple(cache.get("pair", [-1, -1]))
    record = None
    candidates = []
    if "winner" in info and info["winner"]:
        candidates.append(info["winner"])
    candidates += info.get("top", [])
    for r in candidates:
        if r.get("family") == cache_family and tuple(r.get("pair", [])) == cache_pair:
            record = r
            break
    if record is None:
        # Fallback: pick winner anyway and warn.
        record = info.get("winner")
        print(f"WARN: no exact match for ({cache_family}, {cache_pair}) in pair-info; "
              f"using winner: {record['family'] if record else 'NONE'}")
    if record is None:
        raise RuntimeError("No pair record found in pair-info JSON.")

    qualifying = record["qualifying"]
    print(f"=== {label} ===")
    print(f"  cache: {args.cache}")
    print(f"  pair: {cache_family} ({cache_pair[0]},{cache_pair[1]})  "
          f"Lx={Lx} Ly={Ly}  tau={tau:.3f}")
    print(f"  qualifying contact pairs: {len(qualifying)}")
    for q in qualifying:
        print(f"    X[{q['u_i_a']:>3},{q['u_i_b']:>3}]={q['aa_x']} "
              f"Y[{q['u_j_a']:>3},{q['u_j_b']:>3}]={q['aa_y']} "
              f"lo_x={q['lo_x']:+.2f} lo_y={q['lo_y']:+.2f}")

    # Extract X and Y contact-pair index sets from qualifying.
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

    # Per-row alignment entropy.
    Hbase = float(np.nanmean(_row_entropies(Q_baseline)))
    Hprime = float(np.nanmean(_row_entropies(Q_prime)))
    # Per-row edge entropy.
    Hxx_s = float(np.nanmean(_row_entropies(Pxx_single[1:Lx + 1, 1:Lx + 1])))
    Hxx_j = float(np.nanmean(_row_entropies(Pxx_joint[1:Lx + 1, 1:Lx + 1])))
    Hyy_s = float(np.nanmean(_row_entropies(Pyy_single[1:Ly + 1, 1:Ly + 1])))
    Hyy_j = float(np.nanmean(_row_entropies(Pyy_joint[1:Ly + 1, 1:Ly + 1])))

    # PDB-contact signal vs background, per axis (joint and single).
    M_xx_j = _mean_at_pairs(Pxx_joint, pairs_x, Lx)
    M_xx_s = _mean_at_pairs(Pxx_single, pairs_x, Lx)
    M_yy_j = _mean_at_pairs(Pyy_joint, pairs_y, Ly)
    M_yy_s = _mean_at_pairs(Pyy_single, pairs_y, Ly)
    # Background = off-pair-set off-diagonal mean.
    B_xx_j = _mean_off_contact_pairs(Pxx_joint, pairs_x_set, Lx)
    B_xx_s = _mean_off_contact_pairs(Pxx_single, pairs_x_set, Lx)
    B_yy_j = _mean_off_contact_pairs(Pyy_joint, pairs_y_set, Ly)
    B_yy_s = _mean_off_contact_pairs(Pyy_single, pairs_y_set, Ly)

    def _safe_div(a, b):
        if b is None or b == 0 or np.isnan(b):
            return float("nan")
        return a / b

    r_xx_jvs = _safe_div(M_xx_j, M_xx_s)
    r_yy_jvs = _safe_div(M_yy_j, M_yy_s)
    r_xx_jb = _safe_div(M_xx_j, B_xx_j)
    r_yy_jb = _safe_div(M_yy_j, B_yy_j)
    r_xx_sb = _safe_div(M_xx_s, B_xx_s)
    r_yy_sb = _safe_div(M_yy_s, B_yy_s)

    print()
    print(f"  ALIGN per-row entropy (nats): baseline={Hbase:.4f} "
          f"joint={Hprime:.4f}  dH={Hprime - Hbase:+.4f} "
          f"({(Hprime - Hbase) / max(Hbase, 1e-9) * 100:+.2f}%)")
    print(f"  X-X   per-row entropy (nats): single={Hxx_s:.4f} "
          f"joint={Hxx_j:.4f}  dH={Hxx_j - Hxx_s:+.4f} "
          f"({(Hxx_j - Hxx_s) / max(Hxx_s, 1e-9) * 100:+.2f}%)")
    print(f"  Y-Y   per-row entropy (nats): single={Hyy_s:.4f} "
          f"joint={Hyy_j:.4f}  dH={Hyy_j - Hyy_s:+.4f} "
          f"({(Hyy_j - Hyy_s) / max(Hyy_s, 1e-9) * 100:+.2f}%)")
    print()
    print(f"  PDB-contact column-pair (X axis, {len(pairs_x)} pairs):")
    print(f"    M_joint  = {M_xx_j:.5f}   M_single = {M_xx_s:.5f}   "
          f"joint/single = {r_xx_jvs:.3f}x")
    print(f"    B_joint  = {B_xx_j:.5f}   B_single = {B_xx_s:.5f}   "
          f"(background, off-contact)")
    print(f"    Signal-to-background:  joint = {r_xx_jb:.3f}x   "
          f"single = {r_xx_sb:.3f}x")
    print(f"  PDB-contact column-pair (Y axis, {len(pairs_y)} pairs):")
    print(f"    M_joint  = {M_yy_j:.5f}   M_single = {M_yy_s:.5f}   "
          f"joint/single = {r_yy_jvs:.3f}x")
    print(f"    B_joint  = {B_yy_j:.5f}   B_single = {B_yy_s:.5f}   "
          f"(background, off-contact)")
    print(f"    Signal-to-background:  joint = {r_yy_jb:.3f}x   "
          f"single = {r_yy_sb:.3f}x")

    metrics = {
        "label": label, "Lx": Lx, "Ly": Ly, "tau": tau,
        "family": cache_family, "pair": list(cache_pair),
        "qualifying_n": len(qualifying),
        "H_align_baseline": Hbase, "H_align_joint": Hprime,
        "H_align_d": Hprime - Hbase,
        "H_xx_single": Hxx_s, "H_xx_joint": Hxx_j,
        "H_xx_d": Hxx_j - Hxx_s,
        "H_yy_single": Hyy_s, "H_yy_joint": Hyy_j,
        "H_yy_d": Hyy_j - Hyy_s,
        "M_xx_joint": M_xx_j, "M_xx_single": M_xx_s,
        "B_xx_joint": B_xx_j, "B_xx_single": B_xx_s,
        "ratio_xx_joint_vs_single": r_xx_jvs,
        "ratio_xx_joint_signal_vs_bg": r_xx_jb,
        "ratio_xx_single_signal_vs_bg": r_xx_sb,
        "M_yy_joint": M_yy_j, "M_yy_single": M_yy_s,
        "B_yy_joint": B_yy_j, "B_yy_single": B_yy_s,
        "ratio_yy_joint_vs_single": r_yy_jvs,
        "ratio_yy_joint_signal_vs_bg": r_yy_jb,
        "ratio_yy_single_signal_vs_bg": r_yy_sb,
    }
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
