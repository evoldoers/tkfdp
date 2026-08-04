#!/usr/bin/env python3
"""FULL contact-map partition (all Cbeta<8A, |i-j|>=7 pairs) per family, for the
de-sparsified cherry-counts corpus (corpus 1 / 3).

Unlike build_pdb_partition.py (which greedily MAX-MATCHES contacting sites, so
every column is in <=1 pair, ~15 pairs/family), this emits EVERY contacting
column pair -- the full contact map. A column participates in many pairs, giving
O(L) contacts/family instead of O(1), which is the cheapest lever to de-sparsify
the coupled (i,j)->(k,l) cherry-counts tensor.

Contact definition matches CherryML (Prillo/Song 2023) exactly:
  Cbeta-Cbeta distance < 8.0 A  AND  primary-sequence separation |i-j| >= 7.
Gly (no CB) falls back to CA. Output schema matches build_pdb_partition (pairs,
dist, kind, pdb_id, chain_id, family, L) so collect_cherry_counts_struct.py
consumes it unchanged via --partition-dir; all pairs get kind "contact"
(the full map is not chemistry-typed).

Reuses the PDB resolution + MSA->residue mapping from build_pdb_partition
(SIFTS candidates, best-column-coverage chain pick); only the contact SELECTION
differs (all-pairs-within-cutoff instead of reciprocal-NN + greedy match)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from build_pdb_partition import (                                # noqa: E402
    _map_pdb_chain, _order_candidates, load_sifts,
    _scop_id_for_family, PFAM_SEED_DIR, parse_stockholm)

CB_MAX_DIST = 8.0
MIN_SEP = 7            # |i - j| >= 7  (CherryML)


def full_contact_for_family(family, expected_L, cb_max_dist, min_sep,
                            candidates=None, max_pdb_try=6):
    """(pairs, dists, kinds, meta) full Cbeta<cb_max_dist, |i-j|>=min_sep contact
    map, or None. Mapping identical to partition_for_family; only the contact
    selection differs."""
    sto = PFAM_SEED_DIR / f"{family}.sto"
    if not sto.exists():
        return None
    msa_seqs = parse_stockholm(sto)
    if not msa_seqs:
        return None
    L = len(next(iter(msa_seqs.values())))
    if expected_L is not None and L != expected_L:
        return {"__mismatch__": (L, expected_L)}

    if candidates is None:
        pdb_id = _scop_id_for_family(family)
        if pdb_id is None:
            return None
        cand_list = [(pdb_id, None, 1.0)]
    else:
        cand_list = _order_candidates(candidates, max_pdb_try)
        if not cand_list:
            return None

    best = best_pdb = None
    for pdb_id, chain_hint, _cov in cand_list:
        m = _map_pdb_chain(pdb_id, chain_hint, msa_seqs, L)
        if m is None:
            continue
        col_to_res, residues, chain_id = m
        if best is None or len(col_to_res) > len(best[0]):
            best = (col_to_res, residues, chain_id)
            best_pdb = pdb_id
    if best is None:
        return None
    col_to_res, residues, chain_id = best

    cols = sorted(c for c in col_to_res if c < L)
    cb = np.stack([residues[col_to_res[c]]["cb"] for c in cols])   # (M,3)
    cols = np.asarray(cols)
    # all pairs within cutoff and nonlocal
    diff = cb[:, None, :] - cb[None, :, :]
    D = np.sqrt((diff * diff).sum(-1))                            # (M,M)
    sep = np.abs(cols[:, None] - cols[None, :])
    iu, ju = np.triu_indices(len(cols), k=1)
    m = (D[iu, ju] < cb_max_dist) & (sep[iu, ju] >= min_sep)
    ii = cols[iu[m]]; jj = cols[ju[m]]; dd = D[iu, ju][m]
    if ii.size == 0:
        return None
    pairs = np.stack([ii, jj], axis=1).astype(np.int32)
    dists = dd.astype(np.float32)
    kinds = np.array(["contact"] * len(pairs), dtype="<U10")
    meta = {"pdb_id": best_pdb, "chain_id": str(chain_id), "L": L,
            "n_pairs": int(len(pairs)), "n_mapped_cols": int(len(col_to_res))}
    return pairs, dists, kinds, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clv-dir", default="data/pfam_processed_clv_top1000_thin128")
    ap.add_argument("--out-dir", default="data/pdb_fullcontact_clv_top1000_sifts")
    ap.add_argument("--cb-max-dist", type=float, default=CB_MAX_DIST)
    ap.add_argument("--min-sep", type=int, default=MIN_SEP,
                    help="keep pairs with |i-j| >= this (CherryML: 7)")
    ap.add_argument("--sifts", default="data/sifts/pdb_chain_pfam.tsv.gz")
    ap.add_argument("--max-pdb-try", type=int, default=6)
    ap.add_argument("--max-families", type=int, default=0)
    args = ap.parse_args()

    clv_dir, out_dir = Path(args.clv_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = json.loads((clv_dir / "index.json").read_text())
    fam_L = {}
    for fam in index["families"]:
        p = clv_dir / f"{fam}.npz"
        if p.exists():
            fam_L[fam] = int(np.load(p)["L"])
    families = list(fam_L)
    if args.max_families:
        families = families[:args.max_families]

    fam2pdb = load_sifts(Path(args.sifts), set(families)) if args.sifts else {}
    print(f"# SIFTS: {len(fam2pdb)}/{len(families)} families have PDB candidates")
    print(f"# full contact map: Cb<{args.cb_max_dist}A, |i-j|>={args.min_sep}")

    summary = {}
    n_ok = n_no = n_mis = 0
    tot_pairs = 0
    for fi, fam in enumerate(families):
        cands = fam2pdb.get(fam)
        res = full_contact_for_family(fam, fam_L[fam], args.cb_max_dist,
                                      args.min_sep, candidates=cands,
                                      max_pdb_try=args.max_pdb_try)
        if res is None:
            n_no += 1
        elif isinstance(res, dict) and "__mismatch__" in res:
            n_mis += 1
        else:
            pairs, dists, kinds, meta = res
            np.savez(out_dir / f"{fam}.npz",
                     pairs=pairs, dist=dists, kind=kinds,
                     pdb_id=meta["pdb_id"], chain_id=meta["chain_id"],
                     family=fam, L=meta["L"])
            summary[fam] = meta
            n_ok += 1
            tot_pairs += meta["n_pairs"]
        if (fi + 1) % 100 == 0:
            print(f"  [{fi+1}/{len(families)}] ok={n_ok} nopdb={n_no} "
                  f"mism={n_mis} pairs={tot_pairs}", flush=True)

    (out_dir / "index.json").write_text(json.dumps({
        "clv_dir": str(clv_dir), "cb_max_dist": args.cb_max_dist,
        "min_sep": args.min_sep, "contact_map": "full",
        "n_families": n_ok, "n_no_pdb": n_no, "n_mismatch": n_mis,
        "total_pairs": tot_pairs,
        "mean_pairs_per_family": round(tot_pairs / max(1, n_ok), 1),
        "families": summary}, indent=2))
    print(f"\n# done: {n_ok} families, {tot_pairs} pairs "
          f"({tot_pairs / max(1, n_ok):.1f}/family)  "
          f"nopdb={n_no} mism={n_mis}\n# wrote {out_dir}")


if __name__ == "__main__":
    main()
