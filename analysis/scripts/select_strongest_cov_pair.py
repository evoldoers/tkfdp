#!/usr/bin/env python3
"""Identify the sequence-pair within a Pfam family that contributes the
most to high cells of the composite-likelihood edge-pair triangle
(specifically: the cell-pairs where both columns are conserved Cys).

Reads the per-pair MSA-column edge counts from the cache produced by
``analysis/scripts/plot_holmes_msa_triangle.py`` and prints a ranked
table of pairs by their per-pair contribution to C-C cells in the
top-K table.

This script is what drives Item 6 of the Holmes-tile Figure A
redesign: we want to see whether the WORST-OFFENDER pair (highest
C-C-cell contribution) shows obvious C-C signal in the joint sampler
(d)/(e) regions of the tile when run at a higher sweep count.

Usage
-----

    python analysis/scripts/select_strongest_cov_pair.py \\
        --cache-json math-paper/figures/holmes_msa_triangle_PF00014_cache.json \\
        --pfam-sto ~/bio-datasets/data/pfam/random100/PF00014.sto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_stockholm(path: Path):
    names: list[str] = []
    seqs: dict[str, list[str]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name, frag = parts[0], parts[1].strip()
            if name not in seqs:
                seqs[name] = []
                names.append(name)
            seqs[name].append(frag)
    return names, ["".join(seqs[n]) for n in names]


def _hamming_id_ungapped(raw_a: str, raw_b: str, aligned_a: str, aligned_b: str):
    """Sequence identity on shared non-gap columns (% identical / total compared).

    Computed from aligned strings: columns where BOTH sequences have a
    letter are compared; identical letters count as match.
    """
    matches = 0
    n_compared = 0
    for ca, cb in zip(aligned_a, aligned_b):
        if ca.isalpha() and cb.isalpha():
            n_compared += 1
            if ca.upper() == cb.upper():
                matches += 1
    return matches, n_compared, (matches / n_compared if n_compared else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-json", type=Path, required=True)
    ap.add_argument("--pfam-sto", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=15,
                    help="Use top-K cells from the aggregated triangle (default 15)")
    args = ap.parse_args()

    cache = json.loads(args.cache_json.read_text())
    L = cache["L_msa"]
    per_pair_counts = cache.get("per_pair_col_pair_counts")
    if per_pair_counts is None:
        raise RuntimeError(
            "Cache lacks per_pair_col_pair_counts; regenerate by deleting "
            "the cache and re-running plot_holmes_msa_triangle.py")
    pair_list = cache["pair_list"]
    per_pair_summary = cache["per_pair"]
    col_pair_post = np.asarray(cache["col_pair_post"])

    # Parse stockholm to find Cys + gappy columns.
    names_all, alignments = parse_stockholm(args.pfam_sto)
    aa_grid = np.array([[c.upper() for c in a] for a in alignments])
    c_frac = (aa_grid == "C").mean(axis=0)
    c_cols = [i + 1 for i, f in enumerate(c_frac) if f >= 0.5]
    is_letter = np.vectorize(lambda ch: ch.isalpha())(aa_grid)
    gap_frac = 1.0 - is_letter.mean(axis=0)
    gappy_cols = [i + 1 for i, f in enumerate(gap_frac) if f > 0.5]
    cset = set(c_cols)
    gset = set(gappy_cols)

    print(f"Cys cols ({len(c_cols)}): {c_cols}")
    print(f"Gappy cols ({len(gappy_cols)})")
    print()
    print(f"Triangle aggregated max: P_max={col_pair_post.max():.5f}")

    # Aggregate top-K cells (gappy-filtered).
    trip = []
    for i in range(1, L + 1):
        if i in gset:
            continue
        for j in range(i + 1, L + 1):
            if j in gset:
                continue
            v = float(col_pair_post[i, j])
            if v > 0:
                trip.append((v, i, j))
    trip.sort(reverse=True)
    top_cells = trip[: args.top_k]

    # For each pair, compute per-pair contribution to:
    #   (a) all C-C cells (whether they appear in top_cells or not)
    #   (b) the C-C cells WITHIN top_cells
    #   (c) the top_cells overall (any category)
    n_pairs = len(per_pair_counts)
    print(f"Per-pair contribution analysis ({n_pairs} pairs):")
    rows = []
    for p_idx, ppc in enumerate(per_pair_counts):
        # ppc is list of [i, j, count]
        d = {(int(c1), int(c2)): int(cnt) for c1, c2, cnt in ppc}
        # Sum contribution into all C-C cells.
        cc_mass = 0
        for c1 in c_cols:
            for c2 in c_cols:
                if c1 < c2:
                    cc_mass += d.get((c1, c2), 0)
        # Contribution into top_cells (any category).
        top_mass = 0
        cc_topcell_mass = 0
        mixed_topcell_mass = 0
        none_topcell_mass = 0
        for p, i, j in top_cells:
            cnt = d.get((i, j), 0)
            top_mass += cnt
            if i in cset and j in cset:
                cc_topcell_mass += cnt
            elif i in cset or j in cset:
                mixed_topcell_mass += cnt
            else:
                none_topcell_mass += cnt
        # Also: pair's total recorded
        nrec = per_pair_summary[p_idx]["n_recorded"]
        names_p = per_pair_summary[p_idx]["names"]
        tau = per_pair_summary[p_idx]["tau"]
        # Sequence identity over the aligned columns of this pair.
        s_a_idx = pair_list[p_idx][0]
        s_b_idx = pair_list[p_idx][1]
        a_a, a_b = alignments[s_a_idx], alignments[s_b_idx]
        raw_a = "".join(c for c in a_a if c.isalpha())
        raw_b = "".join(c for c in a_b if c.isalpha())
        m, nc, fid = _hamming_id_ungapped(raw_a, raw_b, a_a, a_b)
        row = {
            "pair_idx": p_idx,
            "pair_list_pos": pair_list[p_idx],
            "names": names_p,
            "tau": tau,
            "Lx": per_pair_summary[p_idx]["Lx"],
            "Ly": per_pair_summary[p_idx]["Ly"],
            "n_recorded": nrec,
            "cc_mass": cc_mass,
            "cc_topcell_mass": cc_topcell_mass,
            "mixed_topcell_mass": mixed_topcell_mass,
            "none_topcell_mass": none_topcell_mass,
            "top_mass": top_mass,
            "id_matches": m,
            "id_compared": nc,
            "id_frac": fid,
        }
        rows.append(row)

    # Print per-pair table sorted by C-C mass.
    print()
    print("--- Sorted by cc_mass (total per-pair count at ALL C-C cells) ---")
    print(f"{'pair':>4}  {'tau':>6}  {'Lx':>3}  {'Ly':>3}  {'nrec':>5}  "
          f"{'cc_mass':>7}  {'cc_top':>6}  {'top':>5}  {'idfrac':>6}  names")
    for r in sorted(rows, key=lambda x: -x["cc_mass"]):
        nm = " / ".join(r["names"])[:60]
        print(f"  {r['pair_idx']:>2}  {r['tau']:>6.3f}  {r['Lx']:>3}  "
              f"{r['Ly']:>3}  {r['n_recorded']:>5}  {r['cc_mass']:>7}  "
              f"{r['cc_topcell_mass']:>6}  {r['top_mass']:>5}  "
              f"{r['id_frac']:>6.3f}  {nm}")

    print()
    print("--- Sorted by top_mass (total per-pair count at TOP-K cells) ---")
    for r in sorted(rows, key=lambda x: -x["top_mass"]):
        nm = " / ".join(r["names"])[:60]
        print(f"  {r['pair_idx']:>2}  {r['tau']:>6.3f}  {r['Lx']:>3}  "
              f"{r['Ly']:>3}  {r['n_recorded']:>5}  {r['cc_mass']:>7}  "
              f"{r['cc_topcell_mass']:>6}  {r['top_mass']:>5}  "
              f"{r['id_frac']:>6.3f}  {nm}")

    # Pick winner: highest C-C cell sum from cells WHERE both i and j are
    # in c_cols (whether or not those cells are top-K, since cc_topcell_mass
    # may be 0 for all pairs if no top-K cell is C-C).
    best = max(rows, key=lambda x: x["cc_mass"])
    print()
    print(f"WINNER (highest cc_mass): pair_idx={best['pair_idx']} "
          f"({best['names'][0]} vs {best['names'][1]})")
    print(f"  tau={best['tau']:.3f}, Lx={best['Lx']}, Ly={best['Ly']}, "
          f"id={best['id_frac']*100:.1f}% over {best['id_compared']} cols")
    print(f"  cc_mass={best['cc_mass']}, top_mass={best['top_mass']}, "
          f"cc_topcell_mass={best['cc_topcell_mass']}")

    # JSON dump for downstream consumption.
    print()
    print("JSON_RESULT_BEGIN")
    print(json.dumps({
        "winner": best,
        "all_pairs": rows,
        "top_cells": [{"P": p, "i": i, "j": j,
                       "cat": ("C-C" if i in cset and j in cset else
                               ("mixed" if i in cset or j in cset else "none"))}
                      for p, i, j in top_cells],
        "c_cols": c_cols,
    }))
    print("JSON_RESULT_END")


if __name__ == "__main__":
    main()
