#!/usr/bin/env python3
"""Coupling-responsibility reweighting (paired-vs-null mixture) on trRosetta.

For each contact pair we score its cherry observations under two models:
  * PAIRED  -- the fitted Lumpable pair chain Q_lump (data/lumpable_trrosetta.npz).
  * NULL    -- the EXACT independent-sites reduction A (+) A built from Lumpable's
               marginal generator A. Lumpability is *defined* by the marginals
               being Markov, so A is well-defined and the null has the SAME
               per-site marginals as the paired model -- Lumpable with the joint
               coupling stripped out. No other model gives that exact reduction.

The per-pair log-likelihood ratio LLR_ab = log P_paired - log P_null (summed over
the pair's cherry observations) drives a two-component mixture. Its posterior
responsibility

    r_ab = sigmoid( logit(w) + LLR_ab ),   w = P(paired) prior (EM-fit on TRAIN),

reweights that pair's counts: r_ab of its coupled counts go to the coupled
tensor, (1-r_ab) of each column's marginal goes to the singleton tensor (those
observations look independent, so they belong with the marginals; total
observation count is conserved). w is EM-fit (w <- mean r_ab) on the TRAIN
families only (no val leak) -- and is itself the estimated fraction of contacts
that genuinely coevolve.

Output: a reweighted sharded corpus (same 5-way family round-robin split as the
base corpus; part 4 = val) with FLOAT count tensors, for a 'nuanced'
coupled-model training run. Two passes over the families (pass 1 collects TRAIN
LLRs to EM-fit w; pass 2 reweights and writes shards); family processing is
deterministic so the passes agree.

The fancier next pass (not this one): maintain a separate (sparse) counts tensor
per cluster, re-estimate responsibilities each EM round, and admit MULTIPLE
paired components (different kinds of coupling), not just paired-vs-null.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from build_cherry_counts_trrosetta import (                       # noqa: E402
    A, REMAP, OUR_ALPHA, geom_bin_edges, kimura_dist_matrix,
    greedy_nn_cherries, maximal_contact_matching, lg_ml_t_ml)


def build_logP_grids(Ql, Am, centers):
    """log P over the T tau-bin centres: paired (T,400,400) and marginal (T,20,20)."""
    T = len(centers)
    logPl = np.empty((T, A * A, A * A))
    logPa = np.empty((T, A, A))
    for t, c in enumerate(centers):
        logPl[t] = np.log(np.clip(expm(Ql * c), 1e-300, None))
        logPa[t] = np.log(np.clip(expm(Am * c), 1e-300, None))
    return logPl, logPa


def family_perpair(npz_path, edges, T, max_seqs, cb_max, min_sep, min_shared, seed):
    """-> (n_single_base (A,A,T) float, [ (i0,j0,k0,l0,tb) per contact pair ]) or None.
    Deterministic (seeded subsample); mirrors build_cherry_counts_trrosetta."""
    try:
        d = np.load(npz_path, allow_pickle=True)
        msa = np.asarray(d["msa"]); dist6d = np.asarray(d["dist6d"], np.float64)
    except Exception:
        return None
    if msa.ndim != 2 or msa.shape[0] < 2:
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
    dm = kimura_dist_matrix(msa, min_shared)
    cherries = greedy_nn_cherries(dm)
    if not cherries:
        return None
    pairs, _ = maximal_contact_matching(dist6d, cb_max, min_sep)
    contact = set()
    for a, b in pairs:
        contact.add(a); contact.add(b)
    singleton = np.array([c for c in range(L) if c not in contact], np.int64)

    ia = np.array([c[0] for c in cherries]); ib = np.array([c[1] for c in cherries])
    SA = msa[ia]; SB = msa[ib]                                   # (nc, L)
    t_ml = lg_ml_t_ml(msa, cherries)
    tb_ch = np.clip(np.searchsorted(edges, t_ml) - 1, 0, T - 1)  # per cherry (nc,)

    ns_base = np.zeros((A, A, T))
    if singleton.size:
        ic = SA[:, singleton]; kc = SB[:, singleton]
        tbb = np.broadcast_to(tb_ch[:, None], ic.shape)
        mm = (ic < A) & (kc < A)
        np.add.at(ns_base, (ic[mm], kc[mm], tbb[mm]), 1.0)

    plist = []
    for (a, b) in pairs:
        i0 = SA[:, a]; j0 = SA[:, b]; k0 = SB[:, a]; l0 = SB[:, b]
        mm = (i0 < A) & (j0 < A) & (k0 < A) & (l0 < A)
        if mm.any():
            plist.append((i0[mm], j0[mm], k0[mm], l0[mm], tb_ch[mm]))
    return ns_base, plist


def pair_llr(obs, logPl, logPa):
    """Sum_obs [ log P_paired - log P_null ] for one contact pair."""
    i0, j0, k0, l0, tb = obs
    lp = logPl[tb, i0 * A + j0, k0 * A + l0]
    ln = logPa[tb, i0, k0] + logPa[tb, j0, l0]
    return float((lp - ln).sum())


def fit_w(llrs, iters=500, tol=1e-10):
    """EM for the 2-component mixing weight: w <- mean sigmoid(logit(w)+LLR)."""
    llrs = np.asarray(llrs, float)
    w = 0.5
    for _ in range(iters):
        r = 1.0 / (1.0 + np.exp(-(np.log(w / (1.0 - w)) + llrs)))
        wn = float(r.mean())
        wn = min(max(wn, 1e-6), 1 - 1e-6)
        if abs(wn - w) < tol:
            w = wn; break
        w = wn
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tr-dir", default="data/cherryml_trrosetta/training_set")
    ap.add_argument("--lumpable", default="data/lumpable_trrosetta.npz")
    ap.add_argument("--out", default="data/cherry_counts_trrosetta_resp")
    ap.add_argument("--n-tau-bins", type=int, default=32)
    ap.add_argument("--max-seqs", type=int, default=1024)
    ap.add_argument("--cb-max", type=float, default=8.0)
    ap.add_argument("--min-sep", type=int, default=7)
    ap.add_argument("--min-shared", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=5)
    ap.add_argument("--val-shard", type=int, default=4)
    ap.add_argument("--max-fam", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    edges, centers = geom_bin_edges(args.n_tau_bins); T = args.n_tau_bins

    lz = np.load(args.lumpable)
    Ql = np.asarray(lz["Q"], float); Am = np.asarray(lz["A"], float)
    logPl, logPa = build_logP_grids(Ql, Am, centers)
    print(f"# loaded Lumpable Q_lump{Ql.shape} + marginal A{Am.shape}; "
          f"logP grids built (T={T})", flush=True)

    ids = [ln.strip() for ln in
           (Path(args.tr_dir) / "list15051.txt").read_text().splitlines() if ln.strip()]
    if args.max_fam:
        ids = ids[:args.max_fam]

    def fampath(fid):
        return Path(args.tr_dir) / "npz" / f"{fid}.npz"

    # ---- pass 1: TRAIN LLRs -> EM-fit w ----
    t0 = time.time()
    train_llrs = []; npairs_seen = 0
    for g, fid in enumerate(ids):
        if g % args.nshards == args.val_shard:
            continue                                            # w on train only
        r = family_perpair(fampath(fid), edges, T, args.max_seqs, args.cb_max,
                            args.min_sep, args.min_shared, args.seed)
        if r is None:
            continue
        _, plist = r
        for obs in plist:
            train_llrs.append(pair_llr(obs, logPl, logPa)); npairs_seen += 1
        if (g + 1) % 1000 == 0:
            print(f"  [pass1 {g+1}/{len(ids)}] train-pairs={npairs_seen} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    w = fit_w(train_llrs)
    llr = np.asarray(train_llrs)
    resp = 1.0 / (1.0 + np.exp(-(np.log(w / (1 - w)) + llr)))
    print(f"# PASS1 done: {npairs_seen} train contact-pairs; EM w={w:.4f}  "
          f"(mean r={resp.mean():.4f})", flush=True)
    print(f"#   LLR/pair: median={np.median(llr):.1f} "
          f"[q10={np.quantile(llr,.1):.1f}, q90={np.quantile(llr,.9):.1f}]; "
          f"frac r>0.9={np.mean(resp>0.9):.3f}, r<0.1={np.mean(resp<0.1):.3f}", flush=True)

    # ---- pass 2: reweight every family -> sharded float corpus ----
    logit_w = np.log(w / (1 - w))
    NP = [np.zeros((A, A, A, A, T)) for _ in range(args.nshards)]
    NS = [np.zeros((A, A, T)) for _ in range(args.nshards)]
    agg = [dict(n_families=0, n_contacts=0, pair_mass=0.0, single_mass=0.0)
           for _ in range(args.nshards)]
    t0 = time.time()
    for g, fid in enumerate(ids):
        s = g % args.nshards
        r = family_perpair(fampath(fid), edges, T, args.max_seqs, args.cb_max,
                            args.min_sep, args.min_shared, args.seed)
        if r is None:
            continue
        ns_base, plist = r
        NS[s] += ns_base
        agg[s]["n_families"] += 1
        agg[s]["single_mass"] += float(ns_base.sum())
        for obs in plist:
            rab = 1.0 / (1.0 + np.exp(-(logit_w + pair_llr(obs, logPl, logPa))))
            i0, j0, k0, l0, tb = obs
            np.add.at(NP[s], (i0, j0, k0, l0, tb), rab)
            np.add.at(NS[s], (i0, k0, tb), 1.0 - rab)
            np.add.at(NS[s], (j0, l0, tb), 1.0 - rab)
            agg[s]["n_contacts"] += 1
            agg[s]["pair_mass"] += float(rab * len(tb))
            agg[s]["single_mass"] += float((1.0 - rab) * 2 * len(tb))
        if (g + 1) % 1000 == 0:
            print(f"  [pass2 {g+1}/{len(ids)}] t={time.time()-t0:.0f}s", flush=True)

    for s in range(args.nshards):
        np.savez(out / f"part_{s}.npz", n_pair=NP[s], n_single=NS[s],
                 **{k: np.float64(v) for k, v in agg[s].items()})
    # counts.npz supplies tau_centers + alphabet for fit_pair_models.load_parts
    np.savez_compressed(out / "counts.npz",
                        n_pair=sum(NP), n_single=sum(NS),
                        tau_edges=edges, tau_centers=centers,
                        alphabet=np.array(OUR_ALPHA), n_tau_bins=np.int64(T))
    meta = dict(w=w, nshards=args.nshards, val_shard=args.val_shard,
                lumpable=str(args.lumpable),
                total_pair_mass=sum(a["pair_mass"] for a in agg),
                total_single_mass=sum(a["single_mass"] for a in agg),
                n_contacts=sum(a["n_contacts"] for a in agg))
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"# PASS2 done: w={w:.4f}, coupled-mass={meta['total_pair_mass']:,.0f}, "
          f"single-mass={meta['total_single_mass']:,.0f}, "
          f"contacts={meta['n_contacts']:,} -> {out}", flush=True)


if __name__ == "__main__":
    main()
