#!/usr/bin/env python3
"""Full-alignment version of corpus B: AF-structure selective-matched contacts
(family BREADTH) + DEEP Pfam-full-alignment pairwise-distance cherries (per-family
DEPTH). Combines corpus B's ~15.8k AF families with corpus #4's alignment depth --
the lever the seed-based corpus B was missing (20.6 cherries/fam -> deep).

Per family (needs data/pfam_full_sub/<fam>.npz from pass 1 + the AF partition +
cached AF structure):
  * MSA = the deep full alignment (match columns, our alphabet), pass-1 output.
  * Contacts: map the cached AF structure onto the match columns (align to the
    best full-alignment row), selective matched (reciprocal-NN Cbeta<8A + chemistry,
    greedy match), pLDDT>=70 + PAE<=pae_max filtered -- same definition as corpus B.
  * Cherries: Kimura-distance greedy-NN over the deep MSA (as in corpus #4).
  * Count coupled (i,j)->(k,l) + marginal singletons, tau-binned.

Sharded (--nshards/--shard -> part_i.npz) + --merge, like build_cherry_counts_trrosetta."""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from build_cherry_counts_trrosetta import (                     # noqa: E402
    A, OUR_ALPHA, geom_bin_edges, discretize, kimura_dist_matrix,
    greedy_nn_cherries, lg_ml_tau_bins, lg_ml_t_ml, _ml_rate_scores)
from build_af_partition import extract_af_chain, read_pae, AF_CACHE  # noqa: E402
from build_pdb_partition import (                               # noqa: E402
    _best_msa_row, build_msa_to_residue_map, sidechain_pairs,
    reciprocal_nn_pairs, greedy_match)

AF_DIR = Path("data/pdb_af_partition_train")
SUB_DIR = Path("data/pfam_full_sub")


def _rows_to_seqs(msa, n_cand=20):
    """{row_idx: gapped match-col letter string} for the n_cand deepest (most
    non-gap) rows -- candidates to align the structure against."""
    ng = (msa < A).sum(1)
    cand = np.argsort(-ng)[:n_cand]
    d = {}
    for r in cand:
        d[str(int(r))] = "".join(OUR_ALPHA[x] if x < A else "-" for x in msa[r])
    return d


def process_family(fam, edges, T, cb_max, min_sep, plddt_min, pae_max, min_shared,
                   rate_het=False):
    subp = SUB_DIR / f"{fam}.npz"
    afp = AF_DIR / f"{fam}.npz"
    if not subp.exists() or not afp.exists():
        return None
    msa = np.load(subp)["msa"]                         # (N, L_match) int8, our alphabet
    if msa.ndim != 2 or msa.shape[0] < 2:
        return None
    L = msa.shape[1]
    acc = str(np.load(afp, allow_pickle=True)["pdb_id"])
    pdb = AF_CACHE / f"{acc}.pdb"
    if not pdb.exists():
        return None
    try:
        residues = extract_af_chain(pdb)
    except Exception:
        return None
    if not residues:
        return None
    chain_seq = "".join(r["aa"] for r in residues)
    cand = _rows_to_seqs(msa)
    best_name, best_ungap = _best_msa_row(cand, chain_seq)
    if best_name is None:
        return None
    col_to_res = build_msa_to_residue_map(cand[best_name], best_ungap, chain_seq)
    col_to_res = {c: t for c, t in col_to_res.items() if c < L}
    if len(col_to_res) < max(15, 0.25 * L):
        return None

    col_to_cb = {c: residues[t]["cb"] for c, t in col_to_res.items()}
    hits = sidechain_pairs(col_to_res, residues, 4.0, 6.0, 2.5, min_sep)
    nn = reciprocal_nn_pairs(col_to_cb, cb_max, min_sep)
    pairs, _, _ = greedy_match(hits, nn, col_to_res, residues)

    pae = None
    paep = AF_CACHE / f"{acc}_pae.json"
    if pae_max < 30.0 and paep.exists():
        pae = read_pae(paep)
    kp = []
    for (i, j) in pairs:
        if max(i, j) >= L:
            continue
        ti, tj = col_to_res.get(i), col_to_res.get(j)
        if ti is None or tj is None:
            continue
        if residues[ti]["plddt"] < plddt_min or residues[tj]["plddt"] < plddt_min:
            continue
        if pae is not None and ti < pae.shape[0] and tj < pae.shape[0]:
            if max(float(pae[ti, tj]), float(pae[tj, ti])) > pae_max:
                continue
        elif pae_max < 30.0:
            continue
        kp.append((i, j))
    if not kp:
        return None

    d_mat = kimura_dist_matrix(msa.astype(np.int64), min_shared)
    cherries = greedy_nn_cherries(d_mat)
    if not cherries:
        return None
    pa = np.array([p[0] for p in kp], np.int64); pb = np.array([p[1] for p in kp], np.int64)
    cc = set(int(x) for x in pa) | set(int(x) for x in pb)
    scol = np.array([c for c in range(L) if c not in cc], np.int64)

    NP = np.zeros((A, A, A, A, T), np.int64); NS = np.zeros((A, A, T), np.int64)
    pobs = sobs = 0
    msa64 = msa.astype(np.int64)
    t_ml = lg_ml_t_ml(msa64, cherries)                                # continuous LG-ML tau
    tau_bins = np.clip(np.searchsorted(edges, t_ml) - 1, 0, T - 1)
    r_col = r_pair = None
    if rate_het:                                                     # per-site + joint 2-site ML rates
        ea = np.array([c[0] for c in cherries]); eb = np.array([c[1] for c in cherries])
        score, grid = _ml_rate_scores(msa64, ea, eb, t_ml)
        r_col = grid[score.argmax(0)].astype(float)
        gm = np.exp(np.mean(np.log(np.clip(r_col, 1e-6, None))))
        r_col = np.clip(r_col / max(gm, 1e-9), 0.05, 20.0)
        if pa.size:
            sp = score[:, pa] + score[:, pb]
            r_pair = np.clip(grid[sp.argmax(0)] / max(gm, 1e-9), 0.05, 20.0)
    for ci_, (ia, ib, _tau) in enumerate(cherries):
        sa = msa[ia]; sb = msa[ib]
        i0 = sa[pa]; j0 = sa[pb]; k0 = sb[pa]; l0 = sb[pb]
        mm = (i0 < A) & (j0 < A) & (k0 < A) & (l0 < A)
        if mm.any():
            if rate_het:
                bins = np.clip(np.searchsorted(edges, r_pair * t_ml[ci_]) - 1, 0, T - 1)[mm]
            else:
                bins = np.full(int(mm.sum()), int(tau_bins[ci_]))
            np.add.at(NP, (i0[mm], j0[mm], k0[mm], l0[mm], bins), 1); pobs += int(mm.sum())
        if scol.size:
            ic = sa[scol]; kc = sb[scol]; mm = (ic < A) & (kc < A)
            if mm.any():
                if rate_het:
                    bins = np.clip(np.searchsorted(edges, r_col[scol] * t_ml[ci_]) - 1, 0, T - 1)[mm]
                else:
                    bins = np.full(int(mm.sum()), int(tau_bins[ci_]))
                np.add.at(NS, (ic[mm], kc[mm], bins), 1); sobs += int(mm.sum())
    return NP, NS, dict(n_cherries=len(cherries), n_contacts=len(kp), pair_obs=pobs, single_obs=sobs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/cherry_counts_af_full")
    ap.add_argument("--n-tau-bins", type=int, default=32)
    ap.add_argument("--cb-max", type=float, default=8.0)
    ap.add_argument("--min-sep", type=int, default=6)
    ap.add_argument("--plddt-min", type=float, default=70.0)
    ap.add_argument("--pae-max", type=float, default=5.0)
    ap.add_argument("--min-shared", type=int, default=20)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--max-fam", type=int, default=0)
    ap.add_argument("--rate-het", action="store_true",
                    help="CherryML per-site + joint 2-site-ML rate multiplier: bin "
                         "each transition at rate*tau_ml")
    ap.add_argument("--sub-dir", default=None,
                    help="Pfam-full sub-alignment dir (override; for val/test builds)")
    ap.add_argument("--af-dir", default=None,
                    help="AF partition dir (override; for val/test builds)")
    args = ap.parse_args()
    global SUB_DIR, AF_DIR
    if args.sub_dir:
        SUB_DIR = Path(args.sub_dir)
    if args.af_dir:
        AF_DIR = Path(args.af_dir)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    edges, centers = geom_bin_edges(args.n_tau_bins); T = args.n_tau_bins

    if args.merge:
        parts = sorted(glob.glob(str(out / "part_*.npz")))
        NP = np.zeros((A, A, A, A, T), np.int64); NS = np.zeros((A, A, T), np.int64)
        meta = dict(n_families=0, n_cherries=0, n_contacts=0, pair_obs=0, single_obs=0)
        for p in parts:
            z = np.load(p); NP += z["n_pair"]; NS += z["n_single"]
            for k in list(meta):
                meta[k] += int(z[k]) if k in z.files else 0
        np.savez_compressed(out / "counts.npz", n_pair=NP, n_single=NS,
                            tau_edges=edges, tau_centers=centers,
                            alphabet=np.array(OUR_ALPHA), n_tau_bins=np.int64(T))
        meta.update(alphabet=OUR_ALPHA, total_pair_counts=int(NP.sum()),
                    total_single_counts=int(NS.sum()),
                    contacts="AF selective-matched (pLDDT>=70,PAE<=5)",
                    cherries="Pfam-full deep pairwise-Kimura greedy-NN", n_parts=len(parts))
        json.dump(meta, open(out / "meta.json", "w"), indent=2)
        print(f"# MERGED {len(parts)} parts: {meta['n_families']} fams, "
              f"{meta['n_cherries']} cherries, pair-obs {int(NP.sum()):,} -> {out/'counts.npz'}")
        return

    fams = sorted({Path(p).stem for p in glob.glob(str(SUB_DIR / "*.npz"))})
    fams = fams[args.shard::args.nshards]
    if args.max_fam:
        fams = fams[:args.max_fam]
    NP = np.zeros((A, A, A, A, T), np.int64); NS = np.zeros((A, A, T), np.int64)
    agg = dict(n_families=0, n_cherries=0, n_contacts=0, pair_obs=0, single_obs=0)
    t0 = time.time()
    for k, fam in enumerate(fams):
        r = process_family(fam, edges, T, args.cb_max, args.min_sep,
                           args.plddt_min, args.pae_max, args.min_shared,
                           rate_het=args.rate_het)
        if r is None:
            continue
        np_, ns_, st = r
        NP += np_; NS += ns_
        for kk in ("n_cherries", "n_contacts", "pair_obs", "single_obs"):
            agg[kk] += st[kk]
        agg["n_families"] += 1
        if (k + 1) % 300 == 0:
            print(f"  [shard{args.shard} {k+1}/{len(fams)}] fams={agg['n_families']} "
                  f"cher={agg['n_cherries']} pairobs={agg['pair_obs']} t={time.time()-t0:.0f}s",
                  flush=True)
    np.savez(out / f"part_{args.shard}.npz", n_pair=NP, n_single=NS,
             **{k: np.int64(v) for k, v in agg.items()})
    print(f"# shard {args.shard} done: {agg['n_families']} fams, {agg['n_cherries']} cherries, "
          f"pair-obs {int(NP.sum()):,}", flush=True)


if __name__ == "__main__":
    main()
