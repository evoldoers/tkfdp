#!/usr/bin/env python3
"""Per-CLUSTER (size-2 contact pair) tau-binned transition-count corpus, for the
mixture-of-couplings model. Unlike build_cherry_counts_trrosetta (which POOLS all
contacts of all families into one 20^4 tensor -- destroying per-cluster identity and
washing coupling out ~33x), this keeps EACH contact pair's own transition counts so a
mixture-EM can soft-assign clusters (cluster responsibilities) to K coupling components.

Per family (same cherries/contacts as build_cherry_counts_trrosetta):
  * cherries: Kimura greedy-NN disjoint pairs -> (seqA,seqB); LG-ML tau per cherry.
  * contacts: Cb<8A, |i-j|>=7, greedy maximal matching (each column in <=1 pair).
  * For EACH contact (colA,colB): sparse counts of (i,j)->(k,l) at tau-bin t across
    that family's cherries. Stored CSR: cptr[nc+1], and pf/pt/tb/cnt over all clusters.

Output <out>/part_<shard>.npz: cptr(int64), pf/pt(uint16 in 0..399), tb(uint8),
cnt(int32), meta(int32 [fam_idx,colA,colB,n_cherries] per cluster), fam_ids(str list).
Sharded --nshards/--shard; resumable (skip existing parts); --merge concatenates."""
from __future__ import annotations
import argparse, glob, json, time
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, "experiments")
from build_cherry_counts_trrosetta import (
    A, REMAP, geom_bin_edges, kimura_dist_matrix, greedy_nn_cherries,
    maximal_contact_matching, lg_ml_t_ml,
)


def process_family_pc(npz_path, edges, T, max_seqs, cb_max, min_sep, min_shared, seed):
    """-> (pf, pt, tb, cnt, cluster_meta (nclust,3)=[colA,colB,n_used], cptr) or None."""
    try:
        d = np.load(npz_path, allow_pickle=True)
        msa = np.asarray(d["msa"]); dist6d = np.asarray(d["dist6d"], np.float64)
    except Exception:
        return None
    if msa.ndim != 2 or msa.shape[0] < 4:
        return None
    L = msa.shape[1]
    if dist6d.shape != (L, L):
        return None
    msa = REMAP[msa.astype(np.int64)]
    N = msa.shape[0]
    if N > max_seqs:
        rng = np.random.default_rng(seed)
        pick = np.concatenate([[0], 1 + rng.permutation(N - 1)[:max_seqs - 1]])
        msa = msa[pick]
    d_mat = kimura_dist_matrix(msa, min_shared)
    cherries = greedy_nn_cherries(d_mat)
    if not cherries:
        return None
    pairs, _ = maximal_contact_matching(dist6d, cb_max, min_sep)
    if not pairs:
        return None
    ia = np.array([c[0] for c in cherries]); ib = np.array([c[1] for c in cherries])
    t_ml = lg_ml_t_ml(msa, cherries)
    tb_cher = np.clip(np.searchsorted(edges, t_ml) - 1, 0, T - 1).astype(np.int64)  # (ncher,)
    Sa = msa[ia]; Sb = msa[ib]                                     # (ncher, L)
    pf_all, pt_all, tb_all, cnt_all, meta, cptr = [], [], [], [], [], [0]
    for (ca, cb) in pairs:
        i0 = Sa[:, ca]; j0 = Sa[:, cb]; k0 = Sb[:, ca]; l0 = Sb[:, cb]
        ok = (i0 < A) & (j0 < A) & (k0 < A) & (l0 < A)
        if ok.sum() < 1:
            continue
        pf = (i0[ok] * A + j0[ok]).astype(np.int64)
        pt = (k0[ok] * A + l0[ok]).astype(np.int64)
        tb = tb_cher[ok]
        key = (pf * 400 + pt) * T + tb
        u, c = np.unique(key, return_counts=True)
        kpf = u // (400 * T); rem = u % (400 * T); kpt = rem // T; ktb = rem % T
        pf_all.append(kpf.astype(np.uint16)); pt_all.append(kpt.astype(np.uint16))
        tb_all.append(ktb.astype(np.uint8)); cnt_all.append(c.astype(np.int32))
        meta.append((ca, cb, int(ok.sum())))
        cptr.append(cptr[-1] + len(u))
    if not meta:
        return None
    return (np.concatenate(pf_all), np.concatenate(pt_all), np.concatenate(tb_all),
            np.concatenate(cnt_all), np.array(meta, np.int32), np.array(cptr, np.int64))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tr-dir", default="data/cherryml_trrosetta/training_set")
    ap.add_argument("--out", default="data/per_contact_trrosetta")
    ap.add_argument("--n-tau-bins", type=int, default=32)
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--cb-max", type=float, default=8.0)
    ap.add_argument("--min-sep", type=int, default=7)
    ap.add_argument("--min-shared", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-fam", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    edges, centers = geom_bin_edges(args.n_tau_bins); T = args.n_tau_bins

    if args.merge:
        parts = sorted(glob.glob(str(out / "part_*.npz")),
                       key=lambda p: int(p.split("part_")[1].split(".")[0]))
        PF, PT, TB, CN, META, FAM = [], [], [], [], [], []
        base = 0
        cptr = [0]
        for p in parts:
            z = np.load(p, allow_pickle=True)
            PF.append(z["pf"]); PT.append(z["pt"]); TB.append(z["tb"]); CN.append(z["cnt"])
            fam_ids = list(z["fam_ids"])
            m = z["meta"].copy(); m[:, 0] += len(FAM)                # offset fam idx
            META.append(m); FAM.extend(fam_ids)
            cp = z["cptr"]
            cptr.extend((cp[1:] + base).tolist()); base += int(cp[-1])
        np.savez_compressed(
            out / "counts.npz", pf=np.concatenate(PF), pt=np.concatenate(PT),
            tb=np.concatenate(TB), cnt=np.concatenate(CN), meta=np.concatenate(META),
            cptr=np.array(cptr, np.int64), fam_ids=np.array(FAM),
            tau_centers=centers, n_tau_bins=np.int64(T), alphabet="ACDEFGHIKLMNPQRSTVWY")
        nclust = len(np.concatenate(META))
        print(f"# MERGED {len(parts)} parts: {nclust:,} clusters, "
              f"{int(np.concatenate(CN).sum()):,} transitions -> {out/'counts.npz'}")
        return

    ids = [ln.strip() for ln in (Path(args.tr_dir) / "list15051.txt").read_text().splitlines()
           if ln.strip()]
    ids = ids[args.shard::args.nshards]
    if args.max_fam:
        ids = ids[:args.max_fam]
    partp = out / f"part_{args.shard}.npz"
    PF, PT, TB, CN, META, FAM = [], [], [], [], [], []
    cptr = [0]; base = 0; t0 = time.time(); nfam = 0
    for k, fid in enumerate(ids):
        r = process_family_pc(Path(args.tr_dir) / "npz" / f"{fid}.npz", edges, T,
                              args.max_seqs, args.cb_max, args.min_sep, args.min_shared, args.seed)
        if r is None:
            continue
        pf, pt, tb, cnt, meta, cp = r
        meta = meta.copy()
        meta = np.column_stack([np.full(len(meta), nfam, np.int32), meta])  # [famidx,colA,colB,nused]
        PF.append(pf); PT.append(pt); TB.append(tb); CN.append(cnt); META.append(meta)
        FAM.append(fid)
        cptr.extend((cp[1:] + base).tolist()); base += int(cp[-1]); nfam += 1
        if (k + 1) % 200 == 0:
            nclust = sum(len(m) for m in META)
            print(f"  [shard{args.shard} {k+1}/{len(ids)}] fams={nfam} clusters={nclust} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    if not META:
        print(f"# shard {args.shard}: no clusters"); return
    np.savez(partp, pf=np.concatenate(PF), pt=np.concatenate(PT), tb=np.concatenate(TB),
             cnt=np.concatenate(CN), meta=np.concatenate(META),
             cptr=np.array(cptr, np.int64), fam_ids=np.array(FAM))
    nclust = len(np.concatenate(META))
    print(f"# shard {args.shard} done: {nfam} fams, {nclust:,} clusters, "
          f"{int(np.concatenate(CN).sum()):,} transitions -> {partp}", flush=True)


if __name__ == "__main__":
    main()
