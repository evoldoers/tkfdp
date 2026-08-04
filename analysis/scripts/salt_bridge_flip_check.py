"""Does a supervised-dynfield checkpoint assign an acid-base FIELD FLIP to the
PDB salt-bridge clusters?

For each frozen size-2 cluster we know its kind (saltbridge / cation_pi /
disulfide / volume / nn) from the PDB partition. The checkpoint gives each
column a site class c; class c maps (via arch_assignment[c, theta], theta in
0..L_field-1) to LG-C10 archetypes, each with a net charge. A salt-bridge
FLIP means the two coupled columns carry OPPOSITE charge that SWAPS across
field atoms: at some theta the pair is (acidic, basic) and at another theta it
is (basic, acidic) -- the D-K <-> K-D compensatory flip.

We report, per contact kind:
  * frac_diff_class   : clusters whose two columns took different classes
                        (a prerequisite for any flip)
  * frac_charged      : clusters where BOTH columns use a charged archetype at
                        some theta (|q|>=0.5)
  * frac_complementary: clusters whose per-theta charges are anti-correlated
                        (corr(q1,q2) < 0) -- complementary but maybe static
  * frac_hard_flip    : clusters with an explicit (acid,base)->(base,acid) swap
                        across two field atoms
Salt-bridge / cation-pi should beat the nn / volume null if the model learned
the coevolutionary flip.

Usage:
  PYTHONPATH=src python analysis/scripts/salt_bridge_flip_check.py \
      --checkpoint results/dynfield_supervised_pdb_K16_train_thin128_2026-07-17/checkpoints/latest.npz \
      --partition-dir data/pdb_partition_clv_top1000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path.home() / "tkf-mixdom" / "python"))
from tkfdp.lg08 import ALPHA_ORDER  # noqa: E402

CHARGE = np.zeros(20)
CHARGE[ALPHA_ORDER.index('D')] = -1.0
CHARGE[ALPHA_ORDER.index('E')] = -1.0
CHARGE[ALPHA_ORDER.index('K')] = +1.0
CHARGE[ALPHA_ORDER.index('R')] = +1.0
CHARGE[ALPHA_ORDER.index('H')] = +0.1


def lg_charges(K_a: int):
    """Net charge of each LG archetype (C10 if K_a==10, C20 if K_a==20)."""
    from tkfmixdom.jax.core.site_class_profiles import (
        le_gascuel_c10, le_gascuel_c20)
    fn = le_gascuel_c20 if K_a == 20 else le_gascuel_c10
    profiles, _, names = fn()
    profiles = np.asarray(profiles)
    q = profiles @ CHARGE
    return q, names


def per_theta_charge(arch_row, arch_q):
    return np.array([arch_q[a] for a in arch_row])


def classify_cluster(q1, q2, charged_thr=0.15):
    """Return dict of boolean signals for a (q1, q2) per-theta charge pair."""
    charged = (np.abs(q1).max() >= charged_thr) and (np.abs(q2).max() >= charged_thr)
    # anti-correlation of charge across field atoms
    if q1.std() > 1e-9 and q2.std() > 1e-9:
        corr = float(np.corrcoef(q1, q2)[0, 1])
    else:
        corr = 0.0
    complementary = corr < -0.1
    # hard flip: exists theta with (acidic,basic) and theta' with (basic,acidic)
    hard = False
    L = len(q1)
    for t in range(L):
        for u in range(L):
            if t == u:
                continue
            if q1[t] <= -charged_thr and q2[t] >= charged_thr \
               and q1[u] >= charged_thr and q2[u] <= -charged_thr:
                hard = True
    return {"charged": charged, "corr": corr,
            "complementary": complementary, "hard_flip": hard}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--partition-dir", default="data/pdb_partition_clv_top1000")
    ap.add_argument("--flip-pairs", default="",
                    help="json of data-confirmed flip pairs (from "
                         "saltbridge_mi_flip_data.py --out-flips); adds a "
                         "'saltbridge_FLIP' row restricted to those, the sharp "
                         "test of whether the model learned the real flips.")
    args = ap.parse_args()

    import json as _json
    flip_set = set()
    if args.flip_pairs and Path(args.flip_pairs).exists():
        for r in _json.loads(Path(args.flip_pairs).read_text()):
            flip_set.add((r["family"], int(r["i"]), int(r["j"])))

    d = np.load(args.checkpoint, allow_pickle=False)
    arch = d["arch_assignment"]              # (K_c, L_field)
    K_c, L_field = arch.shape
    fam_ids = [str(x) for x in d["family_ids"]]
    step = int(d["step"])
    K_a = int(d["K_a"])

    arch_q, names = lg_charges(K_a)
    print(f"# LG-C{K_a} archetype net charges:")
    for k, (q, nm) in enumerate(zip(arch_q, names)):
        tag = "acidic" if q < -0.3 else ("basic" if q > 0.3 else "neutral")
        print(f"#   arch {k}: q={q:+.2f}  {tag:7s}  {nm}")

    print(f"\n# checkpoint step={step}  K_c={K_c}  L_field={L_field}  "
          f"K_a={K_a}  {len(fam_ids)} families")
    pdir = Path(args.partition_dir)

    agg = {}   # kind -> list of signal dicts
    for fi, fam in enumerate(fam_ids):
        classes = d[f"fam_{fi}_classes"]
        pf = pdir / f"{fam}.npz"
        if not pf.exists():
            continue
        pz = np.load(pf, allow_pickle=True)
        pairs = np.asarray(pz["pairs"], np.int32).reshape(-1, 2)
        kinds = np.asarray(pz["kind"]).tolist() if "kind" in pz.files \
            else ["nn"] * len(pairs)
        for (i, j), k in zip(pairs, kinds):
            i, j = int(i), int(j)
            if i >= len(classes) or j >= len(classes):
                continue
            c1, c2 = int(classes[i]), int(classes[j])
            q1 = per_theta_charge(arch[c1], arch_q)
            q2 = per_theta_charge(arch[c2], arch_q)
            sig = classify_cluster(q1, q2)
            sig["diff_class"] = (c1 != c2)
            agg.setdefault(k, []).append(sig)
            if k == "saltbridge" and (fam, i, j) in flip_set:
                agg.setdefault("saltbridge_FLIP", []).append(sig)

    # Per-column: fraction of columns of each kind routed to a charged
    # archetype (acidic {3,5} or basic {8}) at some theta.
    charged_arch = set(np.where(np.abs(arch_q) >= 0.15)[0].tolist())
    print(f"\n# charged archetypes (|q|>=0.15): {sorted(charged_arch)}")
    col_charged = {}
    for fi, fam in enumerate(fam_ids):
        classes = d[f"fam_{fi}_classes"]
        pf = pdir / f"{fam}.npz"
        if not pf.exists():
            continue
        pz = np.load(pf, allow_pickle=True)
        pairs = np.asarray(pz["pairs"], np.int32).reshape(-1, 2)
        kinds = np.asarray(pz["kind"]).tolist() if "kind" in pz.files \
            else ["nn"] * len(pairs)
        for (i, j), k in zip(pairs, kinds):
            ks = [k]
            if k == "saltbridge" and (fam, int(i), int(j)) in flip_set:
                ks.append("saltbridge_FLIP")
            for kk in ks:
                for col in (int(i), int(j)):
                    if col < len(classes):
                        c = int(classes[col])
                        used = set(arch[c].tolist())
                        col_charged.setdefault(kk, []).append(
                            len(used & charged_arch) > 0)

    print(f"\n# {'kind':14s} {'n':>5s} {'diff_class':>10s} {'charged':>8s} "
          f"{'complem.':>9s} {'hard_flip':>9s}  {'mean_corr':>9s}  {'col_charged':>11s}")
    order = ["saltbridge", "saltbridge_FLIP", "cation_pi", "disulfide",
             "volume", "nn"]
    for k in order + [x for x in agg if x not in order]:
        if k not in agg:
            continue
        S = agg[k]
        n = len(S)
        fdc = np.mean([s["diff_class"] for s in S])
        fch = np.mean([s["charged"] for s in S])
        fco = np.mean([s["complementary"] for s in S])
        fhf = np.mean([s["hard_flip"] for s in S])
        mc = np.mean([s["corr"] for s in S])
        cc = np.mean(col_charged.get(k, [0]))
        print(f"# {k:14s} {n:5d} {fdc:10.2f} {fch:8.2f} {fco:9.2f} "
              f"{fhf:9.2f}  {mc:+9.3f}  {cc:11.2f}")

    # Headline: enrichment of hard-flip in salt bridges vs nn+volume null.
    sb = agg.get("saltbridge", [])
    null = agg.get("nn", []) + agg.get("volume", [])
    if sb and null:
        fhf_sb = np.mean([s["hard_flip"] for s in sb])
        fhf_nl = np.mean([s["hard_flip"] for s in null])
        fco_sb = np.mean([s["complementary"] for s in sb])
        fco_nl = np.mean([s["complementary"] for s in null])
        print(f"\n# HARD acid-base flip: saltbridge {fhf_sb:.3f} vs "
              f"nn/volume null {fhf_nl:.3f}  "
              f"(enrichment {fhf_sb/(fhf_nl+1e-9):.1f}x)")
        print(f"# charge complementarity: saltbridge {fco_sb:.3f} vs "
              f"null {fco_nl:.3f}  (enrichment {fco_sb/(fco_nl+1e-9):.1f}x)")


if __name__ == "__main__":
    main()
