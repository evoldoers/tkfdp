#!/usr/bin/env python3
"""AlphaFold-DB selective-matched contact partitions, to expand the coupled
cherry-counts corpus from ~470 experimentally-solved families to ~all ~19.7k
Pfam train-split families (Corpus B).

Rationale: effective sample size is driven by the number of ~independent
FAMILIES, not by within-family contact density -- so we keep the thin, selective
matched contact set (reciprocal-nearest-neighbour + specific chemistry, each
column in <=1 pair) and instead maximise family breadth via AlphaFold structures.

Per family:
  1. Resolve a UniProt accession from the Pfam seed sequence names
     (ACC/range or ACC_SPECIES/range forms; take the first that fetches).
  2. Fetch AF-{ACC}-F1-model_v4.pdb (coords + per-residue pLDDT in B-factor) and
     the PAE json; cache under data/af_cache/.
  3. Align the AF chain to the best-matching seed MSA row -> column->residue map
     (same BLOSUM global-align logic as the experimental-PDB builder).
  4. Selective MATCHED contacts: reciprocal-NN Cbeta pairs (Cb<8A, |i-j|>=7,
     CherryML definition) + salt-bridge / cation-pi / disulfide chemistry, then a
     greedy maximum matching (each column in <=1 pair).
  5. QUALITY filter: keep a pair only if BOTH residues have pLDDT>=plddt_min AND
     the pairwise PAE (max of both directions) <= pae_max.

Output schema matches build_pdb_partition (pairs/dist/kind/pdb_id[=ACC]/
chain_id/family/L) so collect_cherry_counts_struct.py --partition-dir consumes
it unchanged. Resumable: families whose npz already exists are skipped."""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from build_pdb_partition import (                                # noqa: E402
    CHARGED_ATOMS, RING_ATOMS, sidechain_pairs, reciprocal_nn_pairs,
    greedy_match, _best_msa_row, build_msa_to_residue_map)
from tkfdp.pdb_contacts import THREE_TO_ONE                      # noqa: E402
from tkfdp.bio import PFAM_SEED_DIR, has_family, load_split, parse_stockholm  # noqa: E402

AF_FILES = "https://alphafold.ebi.ac.uk/files"
AF_VER = "v6"          # current AFDB release (was v4; whole-DB version bump)
AF_CACHE = Path("data/af_cache")
AF_CACHE.mkdir(parents=True, exist_ok=True)

# UniProt accession: 6- or 10-char canonical forms.
ACC_RE = re.compile(r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$'
                    r'|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$')


def accession_candidates(seed_names, max_cand=6):
    """Extract candidate UniProt accessions from Pfam seed sequence names.
    ACC/range -> ACC ; ACC_SPECIES/range -> ACC (TrEMBL entry-name form).
    Skips SwissProt gene-mnemonic names (MNEMONIC_SPECIES) that aren't
    accessions. Preserves order, dedups."""
    out, seen = [], set()
    for name in seed_names:
        tok = name.split('/')[0]
        for cand in (tok, tok.split('_')[0]):
            if ACC_RE.match(cand) and cand not in seen:
                seen.add(cand); out.append(cand)
                break
        if len(out) >= max_cand:
            break
    return out


def _fetch(url, dest, throttle=0.1):
    """Cache-through GET with negative caching. Returns dest path, or None on
    404/error. A prior 404 is remembered (`.404` marker) so re-runs never re-hit
    a missing file -- applies to ALL fetches (model AND pae), not just models."""
    if dest.exists():
        return dest
    if dest.with_suffix(dest.suffix + ".404").exists():
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tkfdp/af"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        dest.write_bytes(data)
        time.sleep(throttle)
        return dest
    except urllib.error.HTTPError as e:
        if e.code == 404:
            dest.with_suffix(dest.suffix + ".404").write_text("")   # negative cache
        return None
    except Exception:
        return None


def fetch_af_model(acc, throttle=0.1):
    if (AF_CACHE / f"{acc}.pdb.404").exists():
        return None
    return _fetch(f"{AF_FILES}/AF-{acc}-F1-model_{AF_VER}.pdb",
                  AF_CACHE / f"{acc}.pdb", throttle)


def fetch_af_pae(acc, throttle=0.1):
    return _fetch(f"{AF_FILES}/AF-{acc}-F1-predicted_aligned_error_{AF_VER}.json",
                  AF_CACHE / f"{acc}_pae.json", throttle)


def read_pae(path):
    """(N,N) PAE matrix from an AF v4 PAE json (list-wrapped dict with
    'predicted_aligned_error'), or None."""
    try:
        d = json.loads(Path(path).read_text())
        if isinstance(d, list):
            d = d[0]
        if "predicted_aligned_error" in d:
            return np.asarray(d["predicted_aligned_error"], dtype=np.float32)
        if "distance" in d and "pae" in d:                       # legacy flat form
            n = int(np.sqrt(len(d["pae"])))
            return np.asarray(d["pae"], dtype=np.float32).reshape(n, n)
    except Exception:
        return None
    return None


def extract_af_chain(pdb_path):
    """AF model (single chain) -> residues list [{aa, ca, cb, charged, ring, sg,
    plddt}], in model order (index == PAE index). Mirrors
    build_pdb_partition.extract_chains_rich but adds per-residue pLDDT (the AF
    B-factor column)."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(Path(pdb_path).stem, str(pdb_path))
    for model in struct:
        for chain in model:
            residues = []
            for res in chain:
                hetflag, _, _ = res.id
                if hetflag != " " or res.resname not in THREE_TO_ONE:
                    continue
                if "CA" not in res:
                    continue
                aa = THREE_TO_ONE[res.resname]
                charged = [np.asarray(res[a].coord, np.float64)
                           for a in CHARGED_ATOMS.get(res.resname, ()) if a in res]
                ring_pts = [np.asarray(res[a].coord, np.float64)
                            for a in RING_ATOMS.get(res.resname, ()) if a in res]
                residues.append({
                    "aa": aa,
                    "ca": np.asarray(res["CA"].coord, np.float64),
                    "cb": np.asarray(res["CB"].coord, np.float64) if "CB" in res
                          else np.asarray(res["CA"].coord, np.float64),
                    "charged": (np.stack(charged) if charged
                                else np.zeros((0, 3), np.float64)),
                    "ring": (np.mean(ring_pts, axis=0) if ring_pts else None),
                    "sg": (np.asarray(res["SG"].coord, np.float64)
                           if aa == "C" and "SG" in res else None),
                    "plddt": float(res["CA"].bfactor),
                })
            if residues:
                return residues
    return None


def af_partition_for_family(family, expected_L, args):
    """(pairs, dists, kinds, meta) or None. Selective matched contacts on the AF
    structure, pLDDT+PAE filtered."""
    sto = PFAM_SEED_DIR / f"{family}.sto"
    if not sto.exists():
        return None
    msa_seqs = parse_stockholm(sto)
    if not msa_seqs:
        return None
    L = len(next(iter(msa_seqs.values())))
    if expected_L is not None and L != expected_L:
        return {"__mismatch__": (L, expected_L)}

    cands = accession_candidates(list(msa_seqs), max_cand=args.max_cand)
    best = None
    for acc in cands:
        mp = fetch_af_model(acc, args.throttle)
        if mp is None:
            continue
        try:
            residues = extract_af_chain(mp)
        except Exception:
            residues = None
        if not residues:
            continue
        chain_seq = "".join(r["aa"] for r in residues)
        best_name, best_ungap = _best_msa_row(msa_seqs, chain_seq)
        if best_name is None:
            continue
        col_to_res = build_msa_to_residue_map(msa_seqs[best_name], best_ungap, chain_seq)
        n_map = sum(1 for c in col_to_res if c < L)
        if n_map < max(15, args.min_cov * L):
            continue
        if best is None or n_map > best[0]:
            best = (n_map, acc, residues, col_to_res)
        break        # take first sufficient-coverage hit (fetch-frugal)
    if best is None:
        return None
    n_map, acc, residues, col_to_res = best

    # selective matched contacts on Cbeta (CherryML def) + chemistry
    col_to_cb = {c: residues[t]["cb"] for c, t in col_to_res.items() if c < L}
    hits = sidechain_pairs(col_to_res, residues, args.sb_max_dist,
                           args.cp_max_dist, args.ss_max_dist, args.min_sep)
    nn = reciprocal_nn_pairs(col_to_cb, args.cb_max_dist, args.min_sep)
    pairs, dists, kinds = greedy_match(hits, nn, col_to_res, residues)

    pae = read_pae(fetch_af_pae(acc, args.throttle)) if pairs else None
    keep_pairs, keep_d, keep_k = [], [], []
    for (i, j), d, k in zip(pairs, dists, kinds):
        if max(i, j) >= L:
            continue
        ti, tj = col_to_res.get(i), col_to_res.get(j)
        if ti is None or tj is None:
            continue
        if residues[ti]["plddt"] < args.plddt_min or residues[tj]["plddt"] < args.plddt_min:
            continue
        if pae is not None and ti < pae.shape[0] and tj < pae.shape[0]:
            if max(float(pae[ti, tj]), float(pae[tj, ti])) > args.pae_max:
                continue
        elif args.pae_max < 30.0:            # PAE required but unavailable -> drop
            continue
        keep_pairs.append((i, j)); keep_d.append(d); keep_k.append(k)
    if not keep_pairs:
        return None
    meta = {"pdb_id": acc, "chain_id": "A", "L": L, "n_pairs": len(keep_pairs),
            "n_mapped_cols": n_map,
            "kinds": {k: keep_k.count(k) for k in set(keep_k)}}
    return keep_pairs, keep_d, keep_k, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clv-dir", default="data/pfam_processed_clv_top1000_thin128")
    ap.add_argument("--out-dir", default="data/pdb_af_partition_train")
    ap.add_argument("--split", default="train")
    ap.add_argument("--families-from", default="split",
                    help="'split' = all seed+tree families in the split; or a "
                         "path to a newline family list; or 'clv' = the CLV index")
    ap.add_argument("--plddt-min", type=float, default=70.0)
    ap.add_argument("--pae-max", type=float, default=5.0,
                    help="max pairwise PAE (A) for a kept contact; >=30 disables")
    ap.add_argument("--cb-max-dist", type=float, default=8.0)
    ap.add_argument("--min-sep", type=int, default=6, help="keep |i-j| > this (CherryML sep>=7)")
    ap.add_argument("--sb-max-dist", type=float, default=4.0)
    ap.add_argument("--cp-max-dist", type=float, default=6.0)
    ap.add_argument("--ss-max-dist", type=float, default=2.5)
    ap.add_argument("--min-cov", type=float, default=0.4, help="min mapped-col fraction of L")
    ap.add_argument("--max-cand", type=int, default=6, help="accession candidates to try/family")
    ap.add_argument("--throttle", type=float, default=0.1, help="sleep between AF fetches (s)")
    ap.add_argument("--max-families", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # family set + expected L (from the seed alignment)
    if args.families_from == "split":
        fams = [f for f in load_split()[args.split] if has_family(f)]
    elif args.families_from == "clv":
        idx = json.loads((Path(args.clv_dir) / "index.json").read_text())
        fams = [f for f in idx["families"] if has_family(f)]
    else:
        fams = [ln.strip() for ln in Path(args.families_from).read_text().splitlines()
                if ln.strip() and has_family(ln.strip())]
    if args.max_families:
        fams = fams[:args.max_families]
    print(f"# AF partitions: {len(fams)} {args.split} families; "
          f"pLDDT>={args.plddt_min}, PAE<={args.pae_max}, Cb<{args.cb_max_dist}A, "
          f"|i-j|>{args.min_sep}", flush=True)

    n_ok = n_noacc = n_noaf = n_nopair = n_skip = 0
    tot_pairs = 0; kind_tot = {}
    t0 = time.time()
    for fi, fam in enumerate(fams):
        outp = out_dir / f"{fam}.npz"
        if outp.exists():
            n_skip += 1; continue
        # expected L from seed
        try:
            L0 = len(next(iter(parse_stockholm(PFAM_SEED_DIR / f"{fam}.sto").values())))
        except Exception:
            n_noacc += 1; continue
        res = af_partition_for_family(fam, L0, args)
        if res is None:
            n_noaf += 1
        elif isinstance(res, dict):
            n_nopair += 1
        else:
            pairs, dists, kinds, meta = res
            np.savez(outp, pairs=np.asarray(pairs, np.int32).reshape(-1, 2),
                     dist=np.asarray(dists, np.float32),
                     kind=np.asarray(kinds, "<U10"),
                     pdb_id=meta["pdb_id"], chain_id="A", family=fam, L=meta["L"])
            n_ok += 1; tot_pairs += meta["n_pairs"]
            for k, v in meta["kinds"].items():
                kind_tot[k] = kind_tot.get(k, 0) + v
        if (fi + 1) % 50 == 0:
            print(f"  [{fi+1}/{len(fams)}] ok={n_ok} noaf={n_noaf} nopair={n_nopair} "
                  f"skip={n_skip} pairs={tot_pairs} ({tot_pairs/max(1,n_ok):.1f}/fam) "
                  f"t={time.time()-t0:.0f}s", flush=True)

    (out_dir / "index.json").write_text(json.dumps({
        "source": "alphafold_v4", "split": args.split,
        "plddt_min": args.plddt_min, "pae_max": args.pae_max,
        "cb_max_dist": args.cb_max_dist, "min_sep": args.min_sep,
        "contact_map": "selective_matched",
        "n_ok": n_ok, "n_no_af": n_noaf, "n_no_pair": n_nopair, "n_skipped": n_skip,
        "total_pairs": tot_pairs, "kind_totals": kind_tot}, indent=2))
    print(f"\n# done: {n_ok} families with AF partition, {tot_pairs} pairs "
          f"({tot_pairs/max(1,n_ok):.1f}/fam)\n"
          f"#   no-AF/nomap={n_noaf} no-pair={n_nopair} skipped(existing)={n_skip}\n"
          f"#   kinds={kind_tot}\n#   wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
