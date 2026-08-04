#!/usr/bin/env python3
"""CherryML-style COUPLED + marginal substitution counts over structure-labeled
Pfam-seed cherries (train split).

Outputs two count tensors:
  n_pair   (A, A, A, A, T) int64 -- structural-contact column-pair counts:
             n_pair[i,j,k,l,t] = # cherries in which the two contacting columns
             (colA, colB) read residues (i,j) in leaf a and (k,l) in leaf b,
             with cherry divergence tau falling in time bin t.
  n_single (A, A, T)       int64 -- non-contact (singleton) column counts:
             n_single[i,k,t] = # cherries in which a non-contact column reads i
             in leaf a and k in leaf b, tau in bin t.

Conventions (match ~/tkf-mixdom/python/build_tkf92_cherry_counts.py exactly):
  * Alphabet ACDEFGHIKLMNPQRSTVWY (A=20); gap/unknown = index 20 -> skipped.
    A column position contributes only when BOTH cherry leaves are amino acids
    (for pairs: all four of (colA,colB) x (a,b) must be amino acids).
  * Cherries: iterative smallest-combined-branch-length pruning of the FULL
    Pfam seed tree (extract_cherries); tau = bl_a_to_parent + bl_b_to_parent.
    The full seed alignment is used -- NO row thinning (the cherry-count step is
    time-independent, so thinning is unnecessary; see the task note).
  * Time bins: geomspace(TAU_MIN=0.001, TAU_MAX=10.0, T+1) edges, geometric-mean
    centers, bin = clip(searchsorted(edges, tau) - 1, 0, T-1). T=32 default.
  * DIRECTIONAL, NOT symmetrized: counts are stored leaf-a -> leaf-b, and the
    contact pair keeps the partition's (colA<colB) column order. Leaf a/b and
    the two contact columns are exchangeable, so the DOWNSTREAM model applies
    whatever symmetrization it wants (as the tkf-mixdom fitters symmetrize S,
    not the counts). No symmetrization is baked in here.

Contacts come from the SIFTS PDB partition (data/pdb_partition_clv_top1000_sifts),
a greedy max-matching so every column is in <= 1 contact pair; a column is a
"singleton" iff it is in no pair. Both tensors are gathered from the same
structure-labeled family corpus. Column indices are shared across the full seed,
the partition, and the (thinned) CLV corpus (thinning drops rows, not columns),
verified per family by an L check.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, "src")
from types import SimpleNamespace                               # noqa: E402
from tkfdp.bio import has_family, load_family, load_split       # noqa: E402
from tkfdp.lg08 import ALPHA_ORDER                               # noqa: E402
from tkfdp.pfam_data import family_cherries                     # noqa: E402

A = 20
TAU_MIN, TAU_MAX = 0.001, 10.0
DEFAULT_PDIR = Path("data/pdb_partition_clv_top1000_sifts")
CHERRY_CACHE = Path("data/cherry_cache")


def cached_family_cherries(fam):
    """family_cherries(load_family(fam)) with a per-family disk cache. The seed
    alignment + tree are static, so the extracted cherries (tau, aa_a, aa_b) are
    too -- cache them so repeat collector runs (e.g. different --kinds / --n-tau-
    bins over the ~16.7k AF corpus) skip the parse+tree+extract work."""
    cp = CHERRY_CACHE / f"{fam}.npz"
    if cp.exists():
        try:
            d = np.load(cp, allow_pickle=False)
            return SimpleNamespace(n_cherries=int(d["n_cherries"]), L=int(d["L"]),
                                   tau=d["tau"], aa_a=d["aa_a"], aa_b=d["aa_b"])
        except Exception:
            pass                                    # corrupt cache -> recompute
    fc = family_cherries(load_family(fam))          # may raise -> caller handles
    CHERRY_CACHE.mkdir(parents=True, exist_ok=True)
    # atomic write: np.savez APPENDS .npz to a name that lacks it, so target
    # `<fam>.tmp` lands on disk as `<fam>.tmp.npz`; rename that to `<fam>.npz`.
    tmp_base = CHERRY_CACHE / f"{fam}.tmp"           # np.savez -> <fam>.tmp.npz
    np.savez(tmp_base, n_cherries=fc.n_cherries, L=fc.L, tau=fc.tau,
             aa_a=fc.aa_a, aa_b=fc.aa_b)
    (CHERRY_CACHE / f"{fam}.tmp.npz").replace(cp)
    return fc


def geom_bin_edges(n_bins: int, tau_min: float = TAU_MIN, tau_max: float = TAU_MAX):
    """Geometric bin edges + geometric-mean centers (matches tkf-mixdom)."""
    edges = np.geomspace(tau_min, tau_max, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def discretize_tau(tau: float, edges: np.ndarray) -> int:
    """Continuous tau -> bin index, clamped to [0, n_bins-1] (matches tkf-mixdom
    canonical builder: searchsorted side='left', minus 1, clamped)."""
    idx = int(np.searchsorted(edges, tau)) - 1
    return max(0, min(idx, len(edges) - 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n-tau-bins", type=int, default=32)
    ap.add_argument("--kinds", default="all",
                    help="comma-separated contact kinds to treat as contacts "
                         "(saltbridge,cation_pi,disulfide,volume,nn), or 'all'")
    ap.add_argument("--partition-dir", default=str(DEFAULT_PDIR),
                    help="dir of per-family contact npz (pairs/kind/L). Swap for a "
                         "full-contact-map or AlphaFold partition to build a "
                         "different corpus.")
    ap.add_argument("--out", default="data/cherry_counts_struct_train")
    ap.add_argument("--max-fam", type=int, default=None,
                    help="cap #families (debug)")
    args = ap.parse_args()

    PDIR = Path(args.partition_dir)
    edges, centers = geom_bin_edges(args.n_tau_bins)
    T = args.n_tau_bins
    kinds = None if args.kinds == "all" else set(args.kinds.split(","))

    train = set(load_split()[args.split])
    part_fams = [Path(f).stem for f in sorted(glob.glob(str(PDIR / "*.npz")))
                 if "confirmed" not in f]
    fams = [f for f in part_fams if f in train and has_family(f)]
    if args.max_fam:
        fams = fams[:args.max_fam]
    print(f"# {len(fams)} structure-labeled {args.split} families; "
          f"kinds={'all' if kinds is None else sorted(kinds)}; T={T} "
          f"(tau in [{TAU_MIN}, {TAU_MAX}])", flush=True)

    n_pair = np.zeros((A, A, A, A, T), dtype=np.int64)
    n_single = np.zeros((A, A, T), dtype=np.int64)

    kind_pairs = Counter()          # #contact pairs actually used, by kind
    stats = dict(n_families=0, n_cherries=0, pair_obs=0, single_obs=0,
                 n_contact_pairs=0, n_contact_cols=0, n_singleton_cols=0,
                 skipped=[])
    t0 = time.time()
    for fi, fam in enumerate(fams):
        try:
            fc = cached_family_cherries(fam)
        except Exception as e:                                   # noqa: BLE001
            stats["skipped"].append([fam, f"load_family: {e}"]); continue
        if fc.n_cherries == 0:
            stats["skipped"].append([fam, "no cherries"]); continue
        L = fc.L

        z = np.load(PDIR / f"{fam}.npz", allow_pickle=False)
        if int(z["L"]) != L:
            stats["skipped"].append(
                [fam, f"L mismatch part={int(z['L'])} seed={L}"]); continue
        pairs = z["pairs"].astype(np.int64)
        kind_arr = z["kind"].astype(str)
        sel = (np.ones(len(pairs), bool) if kinds is None
               else np.array([k in kinds for k in kind_arr], bool))
        pairs = pairs[sel]; kind_sel = kind_arr[sel]
        inrange = (pairs[:, 0] < L) & (pairs[:, 1] < L) & (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)
        pairs = pairs[inrange]; kind_sel = kind_sel[inrange]
        for k in kind_sel:
            kind_pairs[k] += 1

        contact_cols = set(int(c) for c in pairs.reshape(-1))
        singleton_cols = np.array([c for c in range(L) if c not in contact_cols],
                                  dtype=np.int64)
        stats["n_contact_pairs"] += len(pairs)
        stats["n_contact_cols"] += len(contact_cols)
        stats["n_singleton_cols"] += len(singleton_cols)

        pa0 = pairs[:, 0]; pa1 = pairs[:, 1]         # (P,) contact column indices
        for ci in range(fc.n_cherries):
            tau = float(fc.tau[ci])
            if tau <= 0.0 or not np.isfinite(tau):
                continue
            tb = discretize_tau(tau, edges)
            sa = fc.aa_a[ci]; sb = fc.aa_b[ci]       # (L,) int8, gap/unknown = 20
            stats["n_cherries"] += 1

            # --- contact pairs: (i,j) in leaf a, (k,l) in leaf b ---
            if pa0.size:
                ia = sa[pa0]; ja = sa[pa1]; ka = sb[pa0]; la = sb[pa1]
                m = (ia < A) & (ja < A) & (ka < A) & (la < A)
                if m.any():
                    np.add.at(n_pair[..., tb], (ia[m], ja[m], ka[m], la[m]), 1)
                    stats["pair_obs"] += int(m.sum())

            # --- singleton (non-contact) columns: i in leaf a, k in leaf b ---
            if singleton_cols.size:
                ic = sa[singleton_cols]; kc = sb[singleton_cols]
                m = (ic < A) & (kc < A)
                if m.any():
                    np.add.at(n_single[..., tb], (ic[m], kc[m]), 1)
                    stats["single_obs"] += int(m.sum())

        stats["n_families"] += 1
        if (fi + 1) % 50 == 0:
            print(f"[{fi + 1}/{len(fams)}] {fam}: fams={stats['n_families']} "
                  f"cher={stats['n_cherries']} pairobs={stats['pair_obs']} "
                  f"singobs={stats['single_obs']} t={time.time() - t0:.0f}s",
                  flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "counts.npz",
        n_pair=n_pair, n_single=n_single,
        tau_edges=edges.astype(np.float64), tau_centers=centers.astype(np.float64),
        alphabet=np.array(ALPHA_ORDER), n_tau_bins=np.int64(T),
        tau_min=np.float64(TAU_MIN), tau_max=np.float64(TAU_MAX))

    stats["kinds"] = "all" if kinds is None else sorted(kinds)
    stats["kind_pair_counts"] = dict(kind_pairs)
    stats["total_pair_counts"] = int(n_pair.sum())
    stats["total_single_counts"] = int(n_single.sum())
    stats["alphabet"] = ALPHA_ORDER
    stats["directional_not_symmetrized"] = True
    stats["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(stats, open(out / "meta.json", "w"), indent=2)

    print(f"\n# DONE in {stats['elapsed_s']:.0f}s")
    print(f"#   families used   : {stats['n_families']}/{len(fams)} "
          f"(skipped {len(stats['skipped'])})")
    print(f"#   cherries        : {stats['n_cherries']}")
    print(f"#   contact pairs   : {stats['n_contact_pairs']}  "
          f"by kind: {dict(kind_pairs)}")
    print(f"#   pair-obs counts : {stats['total_pair_counts']:,} "
          f"(n_pair sum)")
    print(f"#   single-obs cnts : {stats['total_single_counts']:,} "
          f"(n_single sum)")
    print(f"#   wrote {out / 'counts.npz'}  (+ meta.json)")
    print(f"#   n_pair {n_pair.shape} {n_pair.dtype} "
          f"({n_pair.nbytes / 1e6:.0f} MB)  n_single {n_single.shape}")


if __name__ == "__main__":
    main()
