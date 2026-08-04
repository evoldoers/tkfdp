"""Evaluation harness for the product-of-trees structured mean-field ELBO
(elbo_product_trees) against the exact field-augmented peel and the previous
per-site bound (elbo_persite), on REAL 128-leaf Pfam trees.

Produces the headline numbers for analysis/product_of_trees_elbo_eval.md:
  PART A  absolute exact-ELBO gap for m=1 and m=2 on the PF02457 128-leaf tree,
          head-to-head PoT vs elbo_persite.
  PART B  the DECISIVE metric: pairing-difference error Delta_ELBO - Delta_exact
          across the confirmed-flip pairs, PoT vs elbo_persite.
  PART C  m=1..4 nested tightness + runtime + convergence + warm-start, under a
          tractable REDUCED 6-letter model on the real PF02457 tree topology.

Model matches the live runs: rho=[0.6,0.4], rho_chain=0.15, LG08 (S,pi),
C20 archetypes, enum400 class->arch mapping c -> (c//K_a, c%K_a). CPU/numpy only.

Usage:  JAX_PLATFORMS=cpu PYTHONPATH=src python3 analysis/scripts/eval_product_of_trees.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/tkf-mixdom/python"))
sys.path.insert(0, "src")

from tkfdp.lg08 import get_lg08                                    # noqa: E402
from tkfdp.coupling.dynfield.phylo_elbo.tree import build_tree     # noqa: E402
from tkfdp.coupling.dynfield.phylo_elbo.elbo_product_trees import (  # noqa: E402
    elbo_product_trees)
from tkfdp.coupling.dynfield.phylo_elbo.elbo_persite import elbo_persite  # noqa: E402
from tkfdp.coupling.dynfield.phylo_elbo.exact_peel import (        # noqa: E402
    exact_ll_tree_general)

CLVDIR = "data/pfam_processed_clv_top1000_thin128"
FLIPS = "data/pdb_partition_clv_top1000_sifts/confirmed_flips.json"
CHKPT = "results/enum400dm_discover_train470/_chkpt.npz"
RHO = np.array([0.6, 0.4])
RHO_CHAIN = 0.15
GAP = 20                     # leaf_msa gap code


# --------------------------------------------------------------------------- IO
def load_model():
    from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c20
    S, pi = get_lg08()
    S = np.asarray(S, np.float64)
    pa = np.asarray(le_gascuel_c20()[0], np.float64)
    pa = pa / pa.sum(1, keepdims=True)                            # (K_a=20, A=20)
    d = np.load(CHKPT, allow_pickle=True)
    K_a = int(d["K_a"]); K_c = int(d["K_c"]); L = int(d["L_field"])
    aa = np.stack([[c // K_a, c % K_a] for c in range(K_c)])       # enum400
    pi_field = pa[aa]                                              # (K_c, L, A)
    fids = list(map(str, d["family_ids"]))
    idx = {f: i for i, f in enumerate(fids)}
    classes_of = {f: d[f"fam_{idx[f]}_classes"] for f in fids}
    return S, pi_field, classes_of


def tree_for_cols(fam, cols):
    """Build a Tree over the given columns of a family's thinned CLV tree.
    Leaf residue = leaf_msa value (GAP -> -1)."""
    z = np.load(f"{CLVDIR}/{fam}.npz")
    parent = z["parent"].astype(np.int32)
    tau = z["tau"].astype(np.float64)
    msa = z["leaf_msa"]                                            # (n_leaves, L)
    nl = msa.shape[0]
    leaf_obs = np.full((nl, len(cols)), -1, np.int32)
    for k, c in enumerate(cols):
        col = msa[:, c].astype(np.int32)
        col = np.where(col == GAP, -1, col)
        leaf_obs[:, k] = col
    return build_tree(parent, tau, leaf_obs)


def n_obs_leaves(fam, col):
    z = np.load(f"{CLVDIR}/{fam}.npz")
    return int((z["leaf_msa"][:, col] != GAP).sum())


# ------------------------------------------------------------ single-model calls
def all_three(tree, classes, pi_field, S):
    cls = np.asarray(classes, np.int32)
    ex = exact_ll_tree_general(tree, cls, RHO, pi_field, S, RHO_CHAIN)
    pt = elbo_product_trees(tree, cls, RHO, pi_field, S, RHO_CHAIN)
    ps = elbo_persite(tree, cls, RHO, pi_field, S, RHO_CHAIN)
    return ex, pt, ps


# ============================================================================ A
def part_A(S, pi_field, classes_of, out):
    fam = "PF02457"
    cls = classes_of[fam]
    print("\n=== PART A: absolute gap on the PF02457 128-leaf tree ===")
    print(f"  tree: {np.load(f'{CLVDIR}/{fam}.npz')['leaf_msa'].shape[0]} leaves")
    # m=1 singletons: the two flip columns + a few informative others
    cand = [19, 140, 30, 60, 90, 120]
    m1 = []
    print("  m=1 singletons (col: exact  PoT  persite  gap_PoT gap_persite):")
    for c in cand:
        if n_obs_leaves(fam, c) < 10:
            continue
        tr = tree_for_cols(fam, [c])
        ex, pt, ps = all_three(tr, [cls[c]], pi_field, S)
        m1.append(dict(col=int(c), exact=ex, pot=pt, persite=ps,
                       gap_pot=ex - pt, gap_persite=ex - ps))
        print(f"    {c:4d}: {ex:9.3f} {pt:9.3f} {ps:9.3f}   "
              f"{ex-pt:7.3f} {ex-ps:7.3f}")
    # m=2 pairs: the confirmed flip pair + a few arbitrary pairs
    pairs = [(19, 140), (30, 60), (60, 90), (90, 120)]
    m2 = []
    print("  m=2 pairs   (i,j: exact  PoT  persite  gap_PoT gap_persite):")
    for (i, j) in pairs:
        if n_obs_leaves(fam, i) < 10 or n_obs_leaves(fam, j) < 10:
            continue
        tr = tree_for_cols(fam, [i, j])
        ex, pt, ps = all_three(tr, [cls[i], cls[j]], pi_field, S)
        m2.append(dict(i=int(i), j=int(j), exact=ex, pot=pt, persite=ps,
                       gap_pot=ex - pt, gap_persite=ex - ps))
        print(f"   ({i:3d},{j:3d}): {ex:9.3f} {pt:9.3f} {ps:9.3f}   "
              f"{ex-pt:7.3f} {ex-ps:7.3f}")
    out["A"] = dict(m1=m1, m2=m2)
    g1p = np.array([r["gap_pot"] for r in m1]); g1s = np.array([r["gap_persite"] for r in m1])
    g2p = np.array([r["gap_pot"] for r in m2]); g2s = np.array([r["gap_persite"] for r in m2])
    print(f"  MEAN abs gap  m=1: PoT {g1p.mean():.2f}  persite {g1s.mean():.2f}")
    print(f"  MEAN abs gap  m=2: PoT {g2p.mean():.2f}  persite {g2s.mean():.2f}")


# ============================================================================ B
def part_B(S, pi_field, classes_of, out):
    flips = json.load(open(FLIPS))
    print("\n=== PART B: pairing-difference error  Delta_ELBO - Delta_exact ===")
    print("  Delta = LL(pair, shared field) - LL(i alone) - LL(j alone)")
    rows = []
    for r in flips:
        fam, i, j = r["family"], int(r["i"]), int(r["j"])
        if not os.path.exists(f"{CLVDIR}/{fam}.npz") or fam not in classes_of:
            continue
        cls = classes_of[fam]
        if i >= len(cls) or j >= len(cls):
            continue
        if n_obs_leaves(fam, i) < 10 or n_obs_leaves(fam, j) < 10:
            continue
        try:
            tri = tree_for_cols(fam, [i]); trj = tree_for_cols(fam, [j])
            trij = tree_for_cols(fam, [i, j])
            exi, pti, psi = all_three(tri, [cls[i]], pi_field, S)
            exj, ptj, psj = all_three(trj, [cls[j]], pi_field, S)
            exij, ptij, psij = all_three(trij, [cls[i], cls[j]], pi_field, S)
        except Exception as e:                                    # noqa: BLE001
            print(f"    skip {fam} ({i},{j}): {e}")
            continue
        dex = exij - exi - exj
        dpt = ptij - pti - ptj
        dps = psij - psi - psj
        rows.append(dict(family=fam, i=i, j=j, d_exact=dex,
                         err_pot=dpt - dex, err_persite=dps - dex))
    ep = np.array([r["err_pot"] for r in rows])
    es = np.array([r["err_persite"] for r in rows])
    out["B"] = dict(n=len(rows), rows=rows,
                    pot=dict(mean=float(ep.mean()), std=float(ep.std()),
                             absmax=float(np.abs(ep).max()),
                             absmean=float(np.abs(ep).mean())),
                    persite=dict(mean=float(es.mean()), std=float(es.std()),
                                 absmax=float(np.abs(es).max()),
                                 absmean=float(np.abs(es).mean())))
    print(f"  n pairs = {len(rows)}")
    print(f"  PoT     : mean {ep.mean():+.3f}  std {ep.std():.3f}  "
          f"|mean| {np.abs(ep).mean():.3f}  |max| {np.abs(ep).max():.3f}")
    print(f"  persite : mean {es.mean():+.3f}  std {es.std():.3f}  "
          f"|mean| {np.abs(es).mean():.3f}  |max| {np.abs(es).max():.3f}")
    print("  (baseline for persite reported in the brief: mean -3.3 / std 10.6 / |max| 17.6)")


# ============================================================================ C
_ALPHA = "ACDEFGHIKLMNPQRSTVWY"
# 20 -> 6 physicochemical groups (tractability reduction for the exact m=3,4 peel)
_GROUPS = {"A": 0, "I": 0, "L": 0, "M": 0, "V": 0, "F": 0,
           "W": 1, "Y": 1, "C": 1,
           "K": 2, "R": 2, "H": 2,
           "D": 3, "E": 3,
           "N": 4, "Q": 4, "S": 4, "T": 4,
           "G": 5, "P": 5}
_G20 = np.array([_GROUPS[a] for a in _ALPHA])                     # (20,)


def _reduced_model(pi_field_full):
    """Collapse the 20-letter LG/C20 model to A=6 groups: reduced archetype =
    group-summed C20 profile (renormalised); reduced S = flat (F81). Returns
    (S6, pi_field6 (K_c,L,6), G20-map)."""
    K_c, L, _ = pi_field_full.shape
    pf6 = np.zeros((K_c, L, 6))
    for g in range(6):
        pf6[:, :, g] = pi_field_full[:, :, _G20 == g].sum(2)
    pf6 = pf6 / pf6.sum(2, keepdims=True)
    S6 = np.ones((6, 6)); np.fill_diagonal(S6, 0.0)
    return S6, pf6


def part_C(S, pi_field, classes_of, out):
    fam = "PF02457"
    cls = classes_of[fam]
    S6, pf6 = _reduced_model(pi_field)
    print("\n=== PART C: m=1..4 nested, REDUCED 6-letter model, real PF02457 tree ===")
    # four informative columns (>=10 observed leaves), field-dependent classes
    cand = [c for c in [19, 140, 30, 60, 90, 120, 45, 75]
            if n_obs_leaves(fam, c) >= 10][:4]
    print(f"  columns: {cand}")
    z = np.load(f"{CLVDIR}/{fam}.npz")
    parent = z["parent"].astype(np.int32); tau = z["tau"].astype(np.float64)
    msa = z["leaf_msa"]; nl = msa.shape[0]
    series = []
    for m in (1, 2, 3, 4):
        cols = cand[:m]
        leaf_obs = np.full((nl, m), -1, np.int32)
        for k, c in enumerate(cols):
            col = msa[:, c].astype(np.int32)
            col = np.where(col == GAP, -1, _G20[np.clip(col, 0, 19)])
            leaf_obs[:, k] = col
        tr = build_tree(parent, tau, leaf_obs)
        clsm = np.array([cls[c] for c in cols], np.int32)
        t0 = time.time(); ex = exact_ll_tree_general(tr, clsm, RHO, pf6, S6, RHO_CHAIN); t_ex = time.time() - t0
        t0 = time.time()
        d = elbo_product_trees(tr, clsm, RHO, pf6, S6, RHO_CHAIN, return_terms=True)
        t_pt = time.time() - t0
        t0 = time.time(); ps = elbo_persite(tr, clsm, RHO, pf6, S6, RHO_CHAIN); t_ps = time.time() - t0
        rec = dict(m=m, exact=ex, pot=d["elbo"], persite=ps,
                   gap_pot=ex - d["elbo"], gap_persite=ex - ps,
                   warm=d["warm_elbo"], iters=d["iters"],
                   t_exact=t_ex, t_pot=t_pt, t_persite=t_ps,
                   trace=d["elbo_trace"])
        series.append(rec)
        print(f"  m={m}: exact {ex:9.3f}  PoT {d['elbo']:9.3f} (gap {ex-d['elbo']:6.3f})  "
              f"persite {ps:9.3f} (gap {ex-ps:6.3f})  warm {d['warm_elbo']:9.3f}  "
              f"iters {d['iters']}  t[ex/pt/ps]={t_ex:.2f}/{t_pt:.2f}/{t_ps:.2f}s")
    out["C"] = series
    # convergence + warm-start quality on the m=4 case
    r4 = series[-1]
    print(f"  warm-start recovers {100*(r4['warm']-r4['exact'])/(r4['pot']-r4['exact']) if r4['pot']!=r4['exact'] else float('nan'):.0f}% "
          f"of converged (warm gap {r4['exact']-r4['warm']:.3f} -> conv gap {r4['gap_pot']:.3f})")


def main():
    S, pi_field, classes_of = load_model()
    out = {}
    part_A(S, pi_field, classes_of, out)
    part_B(S, pi_field, classes_of, out)
    part_C(S, pi_field, classes_of, out)
    os.makedirs("analysis/results", exist_ok=True)
    # strip long traces for the json summary
    def clean(o):
        return o
    with open("analysis/results/product_of_trees_eval.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote analysis/results/product_of_trees_eval.json")


if __name__ == "__main__":
    main()
