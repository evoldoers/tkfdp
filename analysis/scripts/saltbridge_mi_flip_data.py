"""Ground-truth (model-free) check: do the PDB salt-bridge column pairs
actually carry compensatory acid-base covariation in the data?

For every size-2 cluster (labelled by kind in the PDB partition) we take the
two columns' residues across all leaves of the (thinned) corpus and measure:

  MI        : mutual information (bits) between the two columns, 20-letter.
  MI_null   : MI expected from finite-sample noise (mean over label-shuffles).
  MI_z      : (MI - MI_null_mean) / MI_null_std  -- coevolution significance.
  charge cells (frequency-weighted over leaves, both non-gap):
     AB = P(acidic_i, basic_j)     BA = P(basic_i, acidic_j)
     AA = P(acidic_i, acidic_j)    BB = P(basic_i, basic_j)
  compensatory : opposite-charge mass (AB+BA) exceeds concordant (AA+BB) AND
                 the pair actually coevolves (MI_z >= 3).
  flip         : BOTH orientations present with real mass -- min(AB, BA) >= 0.03
                 (i.e. the salt bridge SWAPS which side is acid vs base), AND
                 MI_z >= 3.  This is the acid-base FIELD FLIP the model targets.

Aggregated by kind, plus the top flip examples. Salt bridges should show
elevated MI and a real (if minority) flip subset; a conserved-orientation salt
bridge has AB>>BA (or vice versa) and is NOT a flip.

Usage:
  PYTHONPATH=src python analysis/scripts/saltbridge_mi_flip_data.py \
      --clv-dir data/pfam_processed_clv_top1000_thin128 \
      --partition-dir data/pdb_partition_clv_top1000
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tkfdp.lg08 import ALPHA_ORDER  # noqa: E402

ACID = {ALPHA_ORDER.index(a) for a in "DE"}
BASE = {ALPHA_ORDER.index(a) for a in "KRH"}


def _mi_bits(a, b):
    """MI in bits between two integer label vectors (already both-observed)."""
    n = len(a)
    if n == 0:
        return 0.0
    ua = np.unique(a); ub = np.unique(b)
    if len(ua) < 2 or len(ub) < 2:
        return 0.0
    joint = np.zeros((len(ua), len(ub)))
    ia = {v: k for k, v in enumerate(ua)}
    ib = {v: k for k, v in enumerate(ub)}
    for x, y in zip(a, b):
        joint[ia[x], ib[y]] += 1
    joint /= n
    pa = joint.sum(1, keepdims=True); pb = joint.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        m = joint * (np.log2(joint) - np.log2(pa) - np.log2(pb))
    return float(np.nansum(m))


def _mi_null(a, b, rng, n_shuffle=30):
    vals = [_mi_bits(a, rng.permutation(b)) for _ in range(n_shuffle)]
    return float(np.mean(vals)), float(np.std(vals) + 1e-9)


def charge_cells(a, b):
    """Frequency-weighted acid/base joint cells over both-observed leaves."""
    n = len(a)
    if n == 0:
        return dict(AB=0, BA=0, AA=0, BB=0)
    ai = np.array([1 if x in ACID else (2 if x in BASE else 0) for x in a])
    bi = np.array([1 if x in ACID else (2 if x in BASE else 0) for x in b])
    AB = np.mean((ai == 1) & (bi == 2))
    BA = np.mean((ai == 2) & (bi == 1))
    AA = np.mean((ai == 1) & (bi == 1))
    BB = np.mean((ai == 2) & (bi == 2))
    return dict(AB=float(AB), BA=float(BA), AA=float(AA), BB=float(BB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clv-dir", default="data/pfam_processed_clv_top1000_thin128")
    ap.add_argument("--partition-dir", default="data/pdb_partition_clv_top1000")
    ap.add_argument("--mi-z", type=float, default=3.0,
                    help="MI z-score threshold for 'coevolves'.")
    ap.add_argument("--flip-min", type=float, default=0.03,
                    help="min mass on BOTH orientations to count a flip.")
    ap.add_argument("--out-flips", default="",
                    help="write the data-confirmed salt-bridge FLIP pairs "
                         "(family,i,j) to this json for the model eval.")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    clv_dir = Path(args.clv_dir)
    pdir = Path(args.partition_dir)
    import json
    fams = json.loads((clv_dir / "index.json").read_text())["families"]

    rows = {}    # kind -> list of dict
    examples = []
    for fam in fams:
        pf = pdir / f"{fam}.npz"
        cf = clv_dir / f"{fam}.npz"
        if not pf.exists() or not cf.exists():
            continue
        pz = np.load(pf, allow_pickle=True)
        if int(pz["L"]) != int(np.load(cf)["L"]):
            continue
        lm = np.load(cf)["leaf_msa"]        # (n_leaves, L), gap=20
        pairs = np.asarray(pz["pairs"], np.int32).reshape(-1, 2)
        kinds = np.asarray(pz["kind"]).tolist() if "kind" in pz.files \
            else ["nn"] * len(pairs)
        for (i, j), k in zip(pairs, kinds):
            i, j = int(i), int(j)
            col_i, col_j = lm[:, i], lm[:, j]
            both = (col_i < 20) & (col_j < 20)
            a, b = col_i[both], col_j[both]
            n = len(a)
            if n < 15:
                continue
            mi = _mi_bits(a, b)
            m0, s0 = _mi_null(a, b, rng)
            miz = (mi - m0) / s0
            cc = charge_cells(a, b)
            coevolves = miz >= args.mi_z
            comp = coevolves and (cc["AB"] + cc["BA"]) > (cc["AA"] + cc["BB"])
            flip = coevolves and min(cc["AB"], cc["BA"]) >= args.flip_min
            rec = dict(fam=fam, i=i, j=j, n=n, mi=mi, miz=miz,
                       coevolves=coevolves, comp=comp, flip=flip, **cc)
            rows.setdefault(k, []).append(rec)
            if k == "saltbridge" and flip:
                examples.append(rec)

    print(f"# corpus={clv_dir.name}  MI_z>= {args.mi_z}  flip_min={args.flip_min}")
    print(f"\n# {'kind':10s} {'n':>5s} {'medMI':>6s} {'medMIz':>7s} "
          f"{'coevol%':>8s} {'compens%':>9s} {'FLIP%':>6s} {'nFlip':>6s}")
    for k in ["saltbridge", "cation_pi", "disulfide", "volume", "nn"]:
        R = rows.get(k)
        if not R:
            continue
        n = len(R)
        medmi = np.median([r["mi"] for r in R])
        medz = np.median([r["miz"] for r in R])
        cev = np.mean([r["coevolves"] for r in R])
        cmp = np.mean([r["comp"] for r in R])
        flp = np.mean([r["flip"] for r in R])
        nflp = sum(r["flip"] for r in R)
        print(f"# {k:10s} {n:5d} {medmi:6.2f} {medz:7.1f} {cev:8.2f} "
              f"{cmp:9.2f} {flp:6.2f} {nflp:6d}")

    examples.sort(key=lambda r: -min(r["AB"], r["BA"]))
    print(f"\n# top salt-bridge FLIP columns (both acid->base and base->acid "
          f"present, coevolving):")
    print(f"#  {'family':8s} {'i':>4s} {'j':>4s} {'n':>4s} {'MI':>5s} "
          f"{'MIz':>5s}  {'AB':>5s} {'BA':>5s} {'AA':>5s} {'BB':>5s}")
    for r in examples[:15]:
        print(f"#  {r['fam']:8s} {r['i']:4d} {r['j']:4d} {r['n']:4d} "
              f"{r['mi']:5.2f} {r['miz']:5.1f}  {r['AB']:5.2f} {r['BA']:5.2f} "
              f"{r['AA']:5.2f} {r['BB']:5.2f}")

    sb = rows.get("saltbridge", [])
    nfl = sum(r["flip"] for r in sb)
    ncomp = sum(r["comp"] for r in sb)
    ncev = sum(r["coevolves"] for r in sb)
    print(f"\n# SUMMARY salt bridges: {len(sb)} pairs  ->  "
          f"{ncev} coevolve (MI_z>={args.mi_z}), "
          f"{ncomp} charge-compensatory, {nfl} genuine acid-base FLIPS")

    if args.out_flips:
        flips = [dict(family=r["fam"], i=r["i"], j=r["j"], miz=round(r["miz"], 2),
                      AB=round(r["AB"], 3), BA=round(r["BA"], 3))
                 for r in sb if r["flip"]]
        Path(args.out_flips).write_text(json.dumps(flips, indent=2))
        print(f"# wrote {len(flips)} confirmed flip pairs -> {args.out_flips}")


if __name__ == "__main__":
    main()
