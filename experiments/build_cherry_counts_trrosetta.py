#!/usr/bin/env python3
"""Corpus #4: coupled + marginal cherry counts on the CherryML/trRosetta
15,051-family set, via LIGHTWEIGHT pairwise-distance cherries -- NO FastTree (the
compute pole; see feedback_dont_kill_machine). An external cross-check for the
AF-expanded corpus B at CherryML's exact contact definition.

Per family (trRosetta npz: msa uint8 alphabet 'ARNDCQEGHILKMFPSTWYV-', dist6d
raw Cbeta distance matrix):
  * MSA remapped to our alphabet ACDEFGHIKLMNPQRSTVWY (gap=20), subsampled to
    <= --max-seqs (reference row 0 always kept), like CherryML's 1024.
  * Pairwise-distance cherries: Kimura protein distance on shared non-gap match
    columns; greedy nearest-neighbour disjoint pairing -> (seqA, seqB, tau).
    (The lightweight stand-in for CherryML's FastTree-then-cherry extraction.)
  * Contacts: Cbeta<8A (dist6d), |i-j|>=7, greedy MAXIMAL matching (each column
    in <=1 pair) -- CherryML's exact definition.
  * Count coupled (i,j)->(k,l) per contact + marginal singletons per non-contact
    column, tau-binned on the shared geomspace(0.001,10,33) grid.

Sharded: --nshards N --shard i processes list[i::N] into <out>/part_i.npz;
--merge sums the parts into <out>/counts.npz. Resumable (skip existing parts)."""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

A = 20
TAU_MIN, TAU_MAX = 0.001, 10.0
TR_ALPHA = "ARNDCQEGHILKMFPSTWYV"          # trRosetta msa order (0..19)
OUR_ALPHA = "ACDEFGHIKLMNPQRSTVWY"         # our tensor order
# remap[tr_index] -> our_index ; 20 (gap/other) -> 20
REMAP = np.full(256, 20, np.int64)
for _t, _ch in enumerate(TR_ALPHA):
    REMAP[_t] = OUR_ALPHA.index(_ch)
REMAP[20] = 20


def geom_bin_edges(n_bins, tau_min=TAU_MIN, tau_max=TAU_MAX):
    edges = np.geomspace(tau_min, tau_max, n_bins + 1)
    return edges, np.sqrt(edges[:-1] * edges[1:])


def discretize(tau, edges):
    return np.clip(np.searchsorted(edges, tau) - 1, 0, len(edges) - 2)


def kimura_dist_matrix(msa, min_shared=20):
    """(M,M) Kimura protein distance among rows of `msa` (M,L int, gap/other=20)
    on shared non-gap columns. Pairs with < min_shared shared columns -> inf."""
    M, L = msa.shape
    oh = np.zeros((M, L, A), np.float32)
    ng = (msa < A)
    rows, cols = np.nonzero(ng)
    oh[rows, cols, msa[rows, cols]] = 1.0
    ohf = oh.reshape(M, L * A)
    same = ohf @ ohf.T                                   # (M,M) # matched aa
    ngf = ng.astype(np.float32)
    shared = ngf @ ngf.T                                 # (M,M) # both non-gap
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 1.0 - same / np.maximum(shared, 1.0)
        inside = 1.0 - p - 0.2 * p * p
        d = -np.log(np.clip(inside, 1e-9, None))         # Kimura protein distance
    d = np.clip(d, 0.0, TAU_MAX)
    d[shared < min_shared] = np.inf
    np.fill_diagonal(d, np.inf)
    return d


def greedy_nn_cherries(d):
    """Disjoint nearest-neighbour cherries from distance matrix `d` (M,M).
    Returns list of (i, j, tau). Each seq's NN is a candidate; accept in
    increasing-distance order if both endpoints are still free (O(M^2))."""
    M = d.shape[0]
    nn = d.argmin(1)
    nnd = d[np.arange(M), nn]
    order = np.argsort(nnd)
    used = np.zeros(M, bool)
    cherries = []
    for i in order:
        j = nn[i]
        if nnd[i] == np.inf:
            break
        if not used[i] and not used[j]:
            cherries.append((int(i), int(j), float(nnd[i])))
            used[i] = used[j] = True
    return cherries


from tkfdp.lg08 import Q_LG08 as _Q_LG, PI_LG08 as _PI_LG     # noqa: E402


def _lg_logP_grid(t_grid):
    """log P(t)[x,y] over a t grid under the reversible LG rate matrix (mean rate
    1 -> t in substitutions/site). Returns (nt, A*A)."""
    Q = np.asarray(_Q_LG, float); pi = np.asarray(_PI_LG, float)
    d = np.sqrt(pi); dinv = 1.0 / d
    Qs = (d[:, None] * Q) * dinv[None, :]
    Qs = 0.5 * (Qs + Qs.T)
    lam, V = np.linalg.eigh(Qs)
    E = np.exp(lam[None, :] * t_grid[:, None])               # (nt, A)
    VEVt = np.einsum("xk,tk,yk->txy", V, E, V)               # (nt, A, A)
    P = dinv[None, :, None] * VEVt * d[None, None, :]
    return np.log(np.clip(P, 1e-300, None)).reshape(len(t_grid), A * A)


_T_GRID = np.geomspace(TAU_MIN, TAU_MAX, 256)
_LOGPG = _lg_logP_grid(_T_GRID)                              # (256, A*A)


def _lg_pchange_grid(t_grid):
    """p_change(t) = 1 - sum_x pi_x P_xx(t) under LG (mean rate 1). Used to
    moment-estimate per-column/per-cluster relative rate multipliers."""
    Q = np.asarray(_Q_LG, float); pi = np.asarray(_PI_LG, float)
    d = np.sqrt(pi); dinv = 1.0 / d
    Qs = (d[:, None] * Q) * dinv[None, :]; Qs = 0.5 * (Qs + Qs.T)
    lam, V = np.linalg.eigh(Qs)
    wk = (pi[:, None] * V * V).sum(0)                        # sum_x pi_x V_xk^2
    stay = np.exp(lam[None, :] * t_grid[:, None]) @ wk       # sum_x pi_x P_xx(t)
    return 1.0 - stay


_PCHANGE = _lg_pchange_grid(_T_GRID)                         # (256,)


def _pchange(t):
    return np.interp(np.log(np.clip(t, TAU_MIN, TAU_MAX)), np.log(_T_GRID), _PCHANGE)


_RATE_GRID = np.geomspace(0.05, 20.0, 49)                    # relative rate multipliers


def _ml_rate_scores(msa, ea, eb, t_ml):
    """Per-column single-site log-likelihood at each relative rate:
    score[r, col] = sum_cherries log P_LG(rate * tau_ml)[x_col, y_col]. A per-site
    ML rate is argmax over col; a true 2-site JOINT-ML rate for a pair is argmax
    over score[:, colA] + score[:, colB] (product of the two sites' likelihoods
    under one shared rate). Returns (score (nr, L), _RATE_GRID)."""
    Sa = msa[ea].astype(np.int64); Sb = msa[eb].astype(np.int64)  # (nc, L)
    valid = (Sa < A) & (Sb < A)
    gidx = np.where(valid, Sa * A + Sb, 0)                        # (nc, L)
    L = msa.shape[1]; score = np.zeros((len(_RATE_GRID), L))
    for ri, r in enumerate(_RATE_GRID):
        tb = np.clip(np.searchsorted(_T_GRID, r * t_ml), 0, len(_T_GRID) - 1)
        lp = _LOGPG[tb[:, None], gidx]                            # (nc, L)
        score[ri] = (lp * valid).sum(0)
    return score, _RATE_GRID


def lg_ml_t_ml(msa, cherries):
    """Continuous LG maximum-likelihood pairwise-distance tau per cherry."""
    nc = len(cherries)
    ia = np.array([c[0] for c in cherries]); ib = np.array([c[1] for c in cherries])
    Aa = msa[ia].astype(np.int64); Bb = msa[ib].astype(np.int64)
    L = msa.shape[1]; sh = (Aa < A) & (Bb < A)
    ci = np.repeat(np.arange(nc), L).reshape(nc, L)
    gidx = (ci * (A * A) + Aa * A + Bb)[sh]
    Ncnt = np.bincount(gidx, minlength=nc * A * A).reshape(nc, A * A).astype(np.float64)
    return _T_GRID[(Ncnt @ _LOGPG.T).argmax(1)]


def lg_ml_tau_bins(msa, cherries, edges):
    """LG maximum-likelihood pairwise-distance tau BIN per cherry. Kimura is used
    only for PAIRING (greedy_nn_cherries); the tau that gets binned here is the
    LG-ML branch length -- in LG substitutions/site, the same axis as the
    FastTree-based corpora. Vectorised: one bincount builds all per-cherry 20x20
    substitution-count matrices, one matmul scores them against the logP grid."""
    nc = len(cherries)
    ia = np.array([c[0] for c in cherries]); ib = np.array([c[1] for c in cherries])
    Aa = msa[ia].astype(np.int64); Bb = msa[ib].astype(np.int64)
    L = msa.shape[1]
    sh = (Aa < A) & (Bb < A)
    ci = np.repeat(np.arange(nc), L).reshape(nc, L)
    gidx = (ci * (A * A) + Aa * A + Bb)[sh]
    Ncnt = np.bincount(gidx, minlength=nc * A * A).reshape(nc, A * A).astype(np.float64)
    logL = Ncnt @ _LOGPG.T                                   # (nc, nt)
    t_ml = _T_GRID[logL.argmax(1)]
    return np.clip(np.searchsorted(edges, t_ml) - 1, 0, len(edges) - 2)


def maximal_contact_matching(dist6d, cb_max, min_sep):
    """Greedy maximal matching of Cbeta contacts: pairs (i,j) with 0<dist<cb_max
    and |i-j|>=min_sep, taken closest-first, each column used once."""
    L = dist6d.shape[0]
    iu, ju = np.triu_indices(L, k=1)
    d = dist6d[iu, ju]
    sep = ju - iu
    m = (d > 0) & (d < cb_max) & (sep >= min_sep)
    ii, jj, dd = iu[m], ju[m], d[m]
    order = np.argsort(dd)
    used = np.zeros(L, bool)
    pairs, dists = [], []
    for k in order:
        a, b = int(ii[k]), int(jj[k])
        if not used[a] and not used[b]:
            pairs.append((a, b)); dists.append(float(dd[k]))
            used[a] = used[b] = True
    return pairs, dists


def process_family(npz_path, edges, T, max_seqs, cb_max, min_sep, min_shared, seed,
                   rate_het=False):
    """-> (n_pair (A,A,A,A,T), n_single (A,A,T), stats) or None. If rate_het, a
    CherryML-style relative rate multiplier is moment-estimated per column (and
    jointly per contact cluster) and used to scale the cherry divergence when
    choosing each transition's tau bin (bin at rate*tau_ml)."""
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
    msa = REMAP[msa.astype(np.int64)]                    # -> our alphabet, gap=20

    N = msa.shape[0]
    if N > max_seqs:                                     # subsample, keep ref row 0
        rng = np.random.default_rng(seed)
        pick = np.concatenate([[0], 1 + rng.permutation(N - 1)[:max_seqs - 1]])
        msa = msa[pick]
    d_mat = kimura_dist_matrix(msa, min_shared)
    cherries = greedy_nn_cherries(d_mat)
    if not cherries:
        return None
    pairs, _ = maximal_contact_matching(dist6d, cb_max, min_sep)
    contact_cols = set()
    for a, b in pairs:
        contact_cols.add(a); contact_cols.add(b)
    singleton_cols = np.array([c for c in range(L) if c not in contact_cols], np.int64)
    pa = np.array([p[0] for p in pairs], np.int64) if pairs else np.zeros(0, np.int64)
    pb = np.array([p[1] for p in pairs], np.int64) if pairs else np.zeros(0, np.int64)

    n_pair = np.zeros((A, A, A, A, T), np.int64)
    n_single = np.zeros((A, A, T), np.int64)
    n_pobs = n_sobs = 0
    t_ml = lg_ml_t_ml(msa, cherries)                         # continuous LG-ML tau
    tau_bins = np.clip(np.searchsorted(edges, t_ml) - 1, 0, T - 1)

    r_col = r_pair = None
    if rate_het:
        # ML relative rate per column (per-site) and a TRUE 2-site joint-ML rate
        # per contact cluster; recentre to geometric mean 1 (preserve tau calibration).
        ea = np.array([c[0] for c in cherries]); eb = np.array([c[1] for c in cherries])
        score, grid = _ml_rate_scores(msa, ea, eb, t_ml)     # (nr, L)
        r_col = grid[score.argmax(0)].astype(float)          # per-site ML (L,)
        gm = np.exp(np.mean(np.log(np.clip(r_col, 1e-6, None))))
        r_col = np.clip(r_col / max(gm, 1e-9), 0.05, 20.0)
        if pa.size:                                          # joint 2-site ML per cluster
            sp = score[:, pa] + score[:, pb]                 # summed two-site log-lik
            r_pair = np.clip(grid[sp.argmax(0)] / max(gm, 1e-9), 0.05, 20.0)

    for ci_, (ia, ib, _tau) in enumerate(cherries):
        sa = msa[ia]; sb = msa[ib]
        if pa.size:
            i0 = sa[pa]; j0 = sa[pb]; k0 = sb[pa]; l0 = sb[pb]
            mm = (i0 < A) & (j0 < A) & (k0 < A) & (l0 < A)
            if mm.any():
                if rate_het:                                 # bin at rate*tau per cluster
                    tbp = np.clip(np.searchsorted(edges, r_pair * t_ml[ci_]) - 1, 0, T - 1)
                    bins = tbp[mm]
                else:
                    bins = np.full(int(mm.sum()), int(tau_bins[ci_]))
                np.add.at(n_pair, (i0[mm], j0[mm], k0[mm], l0[mm], bins), 1)
                n_pobs += int(mm.sum())
        if singleton_cols.size:
            ic = sa[singleton_cols]; kc = sb[singleton_cols]
            mm = (ic < A) & (kc < A)
            if mm.any():
                if rate_het:                                 # bin at rate*tau per site
                    tbs = np.clip(np.searchsorted(edges, r_col[singleton_cols] * t_ml[ci_]) - 1,
                                  0, T - 1)
                    bins = tbs[mm]
                else:
                    bins = np.full(int(mm.sum()), int(tau_bins[ci_]))
                np.add.at(n_single, (ic[mm], kc[mm], bins), 1)
                n_sobs += int(mm.sum())
    return n_pair, n_single, dict(n_cherries=len(cherries), n_contacts=len(pairs),
                                  L=L, pair_obs=n_pobs, single_obs=n_sobs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tr-dir", default="data/cherryml_trrosetta/training_set")
    ap.add_argument("--out", default="data/cherry_counts_trrosetta")
    ap.add_argument("--n-tau-bins", type=int, default=32)
    ap.add_argument("--max-seqs", type=int, default=1024)
    ap.add_argument("--cb-max", type=float, default=8.0)
    ap.add_argument("--min-sep", type=int, default=7)
    ap.add_argument("--min-shared", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="sum part_*.npz -> counts.npz")
    ap.add_argument("--max-fam", type=int, default=0)
    ap.add_argument("--rate-het", action="store_true",
                    help="CherryML-style per-column/per-cluster rate multiplier: bin "
                         "each transition at rate*tau_ml instead of tau_ml")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    edges, centers = geom_bin_edges(args.n_tau_bins)
    T = args.n_tau_bins

    if args.merge:
        parts = sorted(glob.glob(str(out / "part_*.npz")))
        NP = np.zeros((A, A, A, A, T), np.int64); NS = np.zeros((A, A, T), np.int64)
        meta = dict(n_families=0, n_cherries=0, n_contacts=0, pair_obs=0, single_obs=0)
        for p in parts:
            z = np.load(p)
            NP += z["n_pair"]; NS += z["n_single"]
            for k in list(meta):
                meta[k] += int(z[k]) if k in z.files else 0
        np.savez_compressed(out / "counts.npz", n_pair=NP, n_single=NS,
                            tau_edges=edges, tau_centers=centers,
                            alphabet=np.array(OUR_ALPHA), n_tau_bins=np.int64(T))
        meta.update(alphabet=OUR_ALPHA, total_pair_counts=int(NP.sum()),
                    total_single_counts=int(NS.sum()),
                    contact_def="Cb<8A sep>=7 maximal-matching",
                    cherries="pairwise-Kimura greedy-NN (no FastTree)",
                    n_parts=len(parts))
        json.dump(meta, open(out / "meta.json", "w"), indent=2)
        print(f"# MERGED {len(parts)} parts: {meta['n_families']} families, "
              f"{meta['n_cherries']} cherries, pair-obs {int(NP.sum()):,}, "
              f"single-obs {int(NS.sum()):,} -> {out/'counts.npz'}")
        return

    ids = [ln.strip() for ln in (Path(args.tr_dir) / "list15051.txt").read_text().splitlines()
           if ln.strip()]
    ids = ids[args.shard::args.nshards]
    if args.max_fam:
        ids = ids[:args.max_fam]
    partp = out / f"part_{args.shard}.npz"
    NP = np.zeros((A, A, A, A, T), np.int64); NS = np.zeros((A, A, T), np.int64)
    agg = dict(n_families=0, n_cherries=0, n_contacts=0, pair_obs=0, single_obs=0)
    t0 = time.time()
    for k, fid in enumerate(ids):
        r = process_family(Path(args.tr_dir) / "npz" / f"{fid}.npz", edges, T,
                           args.max_seqs, args.cb_max, args.min_sep,
                           args.min_shared, args.seed, rate_het=args.rate_het)
        if r is None:
            continue
        np_, ns_, st = r
        NP += np_; NS += ns_
        agg["n_families"] += 1; agg["n_cherries"] += st["n_cherries"]
        agg["n_contacts"] += st["n_contacts"]; agg["pair_obs"] += st["pair_obs"]
        agg["single_obs"] += st["single_obs"]
        if (k + 1) % 200 == 0:
            print(f"  [shard{args.shard} {k+1}/{len(ids)}] fams={agg['n_families']} "
                  f"cher={agg['n_cherries']} pairobs={agg['pair_obs']} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    np.savez(partp, n_pair=NP, n_single=NS, **{k: np.int64(v) for k, v in agg.items()})
    print(f"# shard {args.shard} done: {agg['n_families']} fams, "
          f"{agg['n_cherries']} cherries, pair-obs {int(NP.sum()):,} -> {partp}", flush=True)


if __name__ == "__main__":
    main()
