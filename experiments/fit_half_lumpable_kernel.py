#!/usr/bin/env python3
r"""Convex kernel-basis fit of the HALF-LUMPABLE (one-sided) reversible pair CTMC.

Half-lumpable is the most permissive reversible 400-state chain strongly lumpable
to component 1 ALONE: cpt1's marginal is an exact Markov/GTR process, nothing is
imposed on cpt2, and the flux is tied only by TIME REVERSAL (F_{ab,cd}=F_{cd,ab}),
NOT by the component swap.  Mechanically it is the two-sided kernel fit of
fit_lumpable_kernel.py with two changes:
  * orbits are time-reversal-only (n^2(n^2-1)/2 of them, ~79,800 for n=20) rather
    than Klein-4 (component-swap x time-reversal), so the flux carries strictly
    more free directions;
  * only the cpt1 row-marginal constraint sum_l F_{ij,kl} = (pi_ij/rho_i) g_ik is
    imposed (the same rows), so there are FEWER effective constraints.
Both changes ENLARGE the coupling kernel.  Counts are component-swap-symmetrised
(the field label is arbitrary), pi stays symmetric, but the dynamics are
non-exchangeable.  See analysis/lumpable_kernel_basis.md and
experiments/fit_half_lumpable.py (the ALM reference this convex fit replaces).

Everything downstream -- the rational-dual convex M-step, the EM wrapper, the LP,
the boundary report -- is reused unchanged from fit_lumpable_kernel; only the
orbit tying and the setup differ.  Run with PYTHONPATH=src.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
import fit_pair_models as fp                       # noqa: E402
import fit_lumpable_kernel as K                    # noqa: E402
from fit_lumpable_kernel import NA, NS             # noqa: E402


# =====================================================================
# General-n time-reversal orbits (one-sided flux tying).
# =====================================================================
def build_orbits_timerev_n(n):
    """orbit_id over directed off-diagonal transitions tying ONLY the time-reversal
    image (i,j;k,l)~(k,l;i,j) -- not the component swap.  n^2(n^2-1)/2 orbits."""
    NSn = n * n
    ii, jj = np.divmod(np.arange(NSn), n)
    oid = np.full((NSn, NSn), -1, np.int64)
    canon = {}
    nxt = 0
    for x in range(NSn):
        i, j = int(ii[x]), int(jj[x])
        for y in range(NSn):
            if x == y:
                continue
            k, l = int(ii[y]), int(jj[y])
            key = min((i, j, k, l), (k, l, i, j))
            o = canon.get(key)
            if o is None:
                o = nxt; canon[key] = o; nxt += 1
            oid[x, y] = o
    ch1 = ii[:, None] != ii[None, :]
    ch2 = jj[:, None] != jj[None, :]
    is_single = ch1 ^ ch2
    return oid, nxt, is_single


def build_B_half_sparse(n, oid, n_orbits):
    """cpt1 lumpability incidence: (B phi)_{ijk}=sum_l phi[oid(ij,kl)], i!=k."""
    from scipy.sparse import csr_matrix
    rows_ijk = []
    ridx, cidx, data = [], [], []
    r = 0
    for i in range(n):
        for j in range(n):
            x = i * n + j
            for k in range(n):
                if i == k:
                    continue
                rows_ijk.append((i, j, k))
                cnt = {}
                for l in range(n):
                    o = int(oid[x, k * n + l]); cnt[o] = cnt.get(o, 0) + 1
                for o, c in cnt.items():
                    ridx.append(r); cidx.append(o); data.append(float(c))
                r += 1
    B = csr_matrix((data, (ridx, cidx)), shape=(r, n_orbits))
    return B, np.array(rows_ijk, dtype=np.int64)


def dims_half(n, numeric=True):
    """Honest one-sided kernel dimension.  N_phi = n^2(n^2-1)/2 time-reversal
    orbits; kernel = null(B) = N_phi - rank(B).  rank(B) computed exactly."""
    NSn = n * n
    N_phi = NSn * (NSn - 1) // 2
    N_g = n * (n - 1) // 2
    out = dict(n=n, N_phi=N_phi, N_g=N_g)
    if numeric:
        oid, n_orbits, _ = build_orbits_timerev_n(n)
        assert n_orbits == N_phi, (n_orbits, N_phi)
        B, rows = build_B_half_sparse(n, oid, n_orbits)
        if n <= 6:
            rank = int(np.linalg.matrix_rank(B.toarray(), tol=1e-9))
        else:
            BBt = (B @ B.T).toarray()
            ev = np.linalg.eigvalsh(BBt)
            tol = max(BBt.shape) * np.finfo(float).eps * ev.max()
            rank = int((ev > tol).sum())
        out.update(rank_B=rank, kernel=N_phi - rank, n_rows=int(B.shape[0]))
    return out


def exact_rank_half(n):
    """Exact rational rank of the one-sided B via sympy (small n)."""
    import sympy as sp
    oid, n_orbits, _ = build_orbits_timerev_n(n)
    B, _ = build_B_half_sparse(n, oid, n_orbits)
    return int(sp.Matrix(B.toarray().astype(int).tolist()).rank())


# =====================================================================
# Pair-level (NA=20) half-lumpable setup, mirroring K.pair_setup but on
# time-reversal orbits.  Reuses K's convex M-step / EM / LP / boundary.
# =====================================================================
def build_orbits_timerev():
    return build_orbits_timerev_n(NA)


def build_lump_rows_tr(orbit_id):
    """cpt1 lumpability rows on the given (time-reversal) orbit ids; same layout as
    fp.build_lump_rows (ri,rj,rk, ro:(n_rows,NA))."""
    ri, rj, rk, ro = [], [], [], []
    for i in range(NA):
        for j in range(NA):
            x = i * NA + j
            for k in range(NA):
                if i == k:
                    continue
                ri.append(i); rj.append(j); rk.append(k)
                ro.append([int(orbit_id[x, k * NA + l]) for l in range(NA)])
    return np.array(ri), np.array(rj), np.array(rk), np.array(ro)


def pair_setup_half(A, rho):
    """Half-lumpable static setup: time-reversal orbits + cpt1 marginal fixed."""
    from scipy.sparse import csr_matrix
    orbit_id, n_orbits, is_single = build_orbits_timerev()
    pi_prod = np.outer(rho, rho).reshape(NS)
    off = ~np.eye(NS, dtype=bool)
    F0 = np.zeros((NS, NS))
    for j in range(NA):
        idx = np.arange(NA) * NA + j
        F0[np.ix_(idx, idx)] = pi_prod[idx][:, None] * A
    for i in range(NA):
        idx = i * NA + np.arange(NA)
        F0[np.ix_(idx, idx)] += pi_prod[idx][:, None] * A
    np.fill_diagonal(F0, 0.0)
    phi0 = np.full(n_orbits, np.nan)
    for o, v in zip(orbit_id[off], F0[off]):
        if np.isnan(phi0[o]):
            phi0[o] = v
    phi0 = np.nan_to_num(phi0, nan=0.0)
    rows = build_lump_rows_tr(orbit_id)
    ri, rj, rk, ro = rows
    b_marg = phi0[ro].sum(1)
    n_rows = ro.shape[0]
    rr = np.repeat(np.arange(n_rows), NA); cc = ro.ravel()
    B_sparse = csr_matrix((np.ones(rr.size), (rr, cc)),
                          shape=(n_rows, n_orbits)).tocsr()
    rowdat = [(u, m.astype(float)) for u, m in
              (np.unique(ro[a], return_counts=True) for a in range(n_rows))]
    return dict(orbit_id=orbit_id, n_orbits=n_orbits, is_single=is_single,
                pi=pi_prod, F0=F0, phi0=phi0, rows=rows, b_marg=b_marg,
                off=off, B_sparse=B_sparse, B_sparseT=B_sparse.T.tocsr(),
                rowdat=rowdat)


def swap_map():
    """Component-swap index map on the 400 pair states: (i,j) -> (j,i)."""
    return np.arange(NS).reshape(NA, NA).T.reshape(NS)


# =====================================================================
# Drivers
# =====================================================================
def run_dims_half():
    print("# ---- one-sided (half-lumpable) kernel dimension ----", flush=True)
    print(f"# {'n':>3} {'N_phi':>8} {'rank(B)':>8} {'kernel':>8}  "
          f"(two-sided kernel for ref)", flush=True)
    out = {}
    for n in [2, 3, 4, 5, 6, 20]:
        r = dims_half(n, numeric=True)
        two = K.dims(n)["kernel"]
        print(f"# {n:>3} {r['N_phi']:>8} {r['rank_B']:>8} {r['kernel']:>8}  "
              f"(two-sided {two})", flush=True)
        out[n] = r
    return out


def run_corpus_half(corpus, n_em=20, do_lp=True):
    print(f"\n# ============= HALF-LUMPABLE  {corpus} =============", flush=True)
    parts = fp.n_parts(corpus)
    train_ids = [i for i in range(parts) if i != parts - 1]
    val_id = parts - 1
    npair, tau, _ = fp.load_parts(corpus, train_ids)
    vpair, _, _ = fp.load_parts(corpus, [val_id])
    sw = swap_map()
    npair = npair + npair[sw][:, sw]          # component-swap symmetrise
    vpair = vpair + vpair[sw][:, sw]
    print(f"# train pairs={npair.sum():.3e} val pairs={vpair.sum():.3e} "
          f"(swap-symmetrised)", flush=True)
    A, rho, g = K.fit_marginal_gtr(K.marginal_counts_from_pair(npair), tau,
                                   iters=100)
    t0 = time.time()
    setup = pair_setup_half(A, rho)
    ri, rj, rk, ro = setup["rows"]
    res_lift = np.linalg.norm(setup["phi0"][ro].sum(1) - setup["b_marg"])
    print(f"# setup ({time.time()-t0:.0f}s); n_orbits={setup['n_orbits']} "
          f"lift ||B phi0 - b_marg||={res_lift:.1e}", flush=True)
    Q0 = fp.Q_from_flux(setup["F0"], setup["pi"])
    val0 = fp.loglik(Q0, setup["pi"], tau, vpair) / vpair.sum()
    print(f"# A(+)A independent baseline val per-count = {val0:.4f}", flush=True)
    lp = None
    if do_lp:
        t1 = time.time(); lp = K.lp_feasibility(setup)
        print(f"# LP max-slack t = {lp['t']:.3e} (success={lp['success']}) "
              f"[{time.time()-t1:.0f}s]", flush=True)
    t2 = time.time()
    res = K.fit_convex_em(npair, tau, setup, n_em=n_em)
    train_per = res["ll"] / npair.sum()
    val_per = fp.loglik(res["Q"], res["pi"], tau, vpair) / vpair.sum()
    monotone = bool(np.all(np.diff(res["hist"]) > -1.0))
    wall = time.time() - t2
    print(f"# convex-EM done [{wall:.0f}s]  train per={train_per:.4f}  "
          f"VAL per={val_per:.4f}  monotone={monotone}  Mrelres={res['m_relres']:.2e}",
          flush=True)
    bnd = K.boundary_report(res, setup)
    print(f"# boundary: frac off-diag rates ~0 = {bnd['frac_rates_zero']:.4f}, "
          f"min pos rate = {bnd['min_pos_rate']:.3e}; "
          f"double-orbits zero {bnd['double']['n_zero']}/{bnd['double']['n']} "
          f"({bnd['double']['frac_zero']:.3f}), single-orbits zero "
          f"{bnd['single']['n_zero']}/{bnd['single']['n']} "
          f"({bnd['single']['frac_zero']:.3f})", flush=True)
    coup = float(np.linalg.norm(res["phi"] - setup["phi0"])
                 / max(np.linalg.norm(setup["phi0"]), 1e-30))
    tag = corpus.rstrip("/").split("/")[-1]
    import os
    os.makedirs("results/pair_models", exist_ok=True)
    pp = f"results/pair_models/half_lumpable_kernel_{tag}.npz"
    np.savez(pp, Q=res["Q"], pi=res["pi"], phi=res["phi"], A=A, rho=rho,
             val_per=val_per, train_per=train_per)
    print(f"# saved {pp}; coupling ||phi-phi0||/||phi0|| = {coup:.4f} "
          f"wall={wall:.0f}s", flush=True)
    return dict(corpus=corpus, val0_indep=float(val0),
                lp={k: lp[k] for k in ("t", "success")} if lp else None,
                train_per=float(train_per), val_per=float(val_per),
                monotone=monotone, boundary=bnd, coupling_rel=coup,
                m_relres=float(res["m_relres"]), wall_s=float(wall))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", action="store_true")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-lp", action="store_true")
    ap.add_argument("--n-em", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = {}
    if args.dims or args.all:
        result["dims_half"] = run_dims_half()
    corpora = []
    if args.all:
        corpora = ["data/cherry_counts_trrosetta", "data/cherry_counts_af_full"]
    elif args.corpus:
        corpora = [args.corpus]
    for c in corpora:
        result[c] = run_corpus_half(c, n_em=args.n_em, do_lp=not args.no_lp)
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2, default=float)
        print(f"\n# wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
