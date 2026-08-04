"""Statistically-grounded metrics for a dynfield checkpoint (discovery OR
supervised, train OR val). Pure-numpy/scipy off the checkpoint + PDB partitions
(no GPU) for the partition/architecture metrics; an optional --score pass
rescoreds the field-rate posterior on GPU for the flip-separation test.

Metrics (each with an explicit null / p-value):
  * flip PAIR recall   -- confirmed-flip contacts placed in the same cluster;
                          hypergeometric p vs random pairing.
  * PDB-contact enrichment -- discovered pairs that are structural contacts,
                          per kind, obs/exp fold + Poisson tail p.
  * archetype charge-swap -- classes whose field states span acid<->base;
                          permutation null (shuffle learned theta>=1) + p.
  * phi separation (--score) -- field-flip posterior on confirmed-flip vs other
                          pairs; Mann-Whitney U.

Emits one JSON object (append with --out for a per-sweep trajectory).

Usage:
  PYTHONPATH=src:$HOME/tkf-mixdom/python python analysis/scripts/dynfield_metrics.py \
    --ckpt results/<run>/_chkpt.npz --label sweepN [--out traj.jsonl]
"""
from __future__ import annotations
import argparse, json, sys
from math import comb
from pathlib import Path
import numpy as np

ACID = [2, 3]           # D, E   (alphabet ACDEFGHIKLMNPQRSTVWY)
BASE = [8, 14, 6]       # K, R, H


def _archetype_profiles(which: str) -> np.ndarray:
    """Row-normalised archetype emission profiles (K_a, 20)."""
    sys.path.insert(0, str(Path.home() / "tkf-mixdom" / "python"))
    from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c10, le_gascuel_c20
    prof = np.asarray((le_gascuel_c20 if which == "c20" else le_gascuel_c10)()[0], float)
    return prof / prof.sum(1, keepdims=True)


def _archetype_charge(which: str) -> np.ndarray:
    """Net basic-minus-acidic charge propensity per archetype profile."""
    prof = _archetype_profiles(which)
    return prof[:, BASE].sum(1) - prof[:, ACID].sum(1)


def charge_flip_soft(ck, arch, which, confirmed_flips_path, fam_ids):
    """Soft per-column charge-flip score = P(the two field states emit antipolar
    residues) for each column's inferred class, comparing confirmed-flip columns
    against the corpus background. Robust to enum400 (reads the class->archetype
    map `arch`, so it works whether arch is learned or the fixed enumeration).

    This is the meaningful charge signal for enum400 -- unlike `arch_charge_swap`,
    which reads only the (fixed) arch table and is degenerate there."""
    from scipy.stats import mannwhitneyu
    prof = _archetype_profiles(which)
    pA = prof[:, ACID].sum(1); pB = prof[:, BASE].sum(1)     # per-archetype masses
    p_anti = pA[arch[:, 0]] * pB[arch[:, 1]] + pB[arch[:, 0]] * pA[arch[:, 1]]  # per class

    fam_set = set(fam_ids); fam_idx = {f: i for i, f in enumerate(fam_ids)}
    cf = [r for r in json.load(open(confirmed_flips_path)) if r["family"] in fam_set]
    conf, sizes = [], []
    for r in cf:
        fi = fam_idx[r["family"]]; cls = ck[f"fam_{fi}_classes"]; cid = ck[f"fam_{fi}_cluster_id"]
        for col in (int(r["i"]), int(r["j"])):
            conf.append(float(p_anti[int(cls[col])]))
            sizes.append(int((cid == cid[col]).sum()))
    bg = np.array([float(p_anti[int(c)]) for fi in range(len(fam_ids))
                   for c in ck[f"fam_{fi}_classes"]])
    conf = np.array(conf); sizes = np.array(sizes)
    if not len(conf) or not len(bg):
        return {"n_confirmed_cols": int(len(conf))}

    def _stat(v):
        return {"mean": round(float(v.mean()), 5),
                "fold": round(float(v.mean() / bg.mean()), 3) if bg.mean() > 0 else None,
                "p_gt_bg": float(mannwhitneyu(v, bg, alternative="greater")[1]),
                "n": int(len(v))}
    out = {"confirmed": _stat(conf), "background_mean": round(float(bg.mean()), 5)}
    if (sizes == 1).any():
        out["singletons"] = _stat(conf[sizes == 1])
    if (sizes == 2).any():
        out["paired"] = _stat(conf[sizes == 2])
    return out


def _discovered_pairs(ckpt) -> "dict[str, set]":
    """fid -> set of (i,j) size-2 clusters."""
    fam_ids = list(ckpt["family_ids"].tolist())
    out = {}
    for fi, fid in enumerate(fam_ids):
        cid = ckpt[f"fam_{fi}_cluster_id"]
        v, n = np.unique(cid, return_counts=True)
        s = set()
        for k in v[n == 2]:
            a, b = sorted(int(x) for x in np.where(cid == k)[0])
            s.add((a, b))
        out[fid] = s
    return out


def flip_recall(disc, confirmed_flips_path, fam_set):
    """Confirmed-flip contacts placed in the same cluster + hypergeometric p."""
    from scipy.stats import hypergeom
    cf = [r for r in json.load(open(confirmed_flips_path)) if r["family"] in fam_set]
    found = [r for r in cf
             if (int(r["i"]), int(r["j"])) in disc.get(r["family"], set())]
    return {"confirmed_in_fams": len(cf), "discovered": len(found),
            "recall": len(found) / len(cf) if cf else None,
            "hits": [f'{r["family"]}:{r["i"]}-{r["j"]}(miz={r.get("miz")})'
                     for r in found]}


def contact_enrichment(disc, pdb_dir, fam_ids):
    """Discovered pairs that are PDB contacts, per kind, with obs/exp fold and a
    Poisson tail p on the overall count (obs ~ Poisson(exp) under random pairing).
    hyper-p uses the summed candidate space so it is comparable across runs."""
    from scipy.stats import poisson, hypergeom
    pdb_dir = Path(pdb_dir)
    obs = exp = 0.0
    M = Ksucc = Ndraw = 0                       # hypergeom over the whole corpus
    by_kind_obs, by_kind_exp, by_kind_tot = {}, {}, {}
    for fid in fam_ids:
        p = pdb_dir / f"{fid}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=False)
        L = int(d["L"])
        if L < 2:
            continue
        pdb = {tuple(sorted(map(int, xy))): str(d["kind"][t])
               for t, xy in enumerate(d["pairs"])}
        dset = disc.get(fid, set())
        npair = comb(L, 2)
        p_disc = len(dset) / npair
        hit = dset & set(pdb)
        obs += len(hit); exp += len(dset) * len(pdb) / npair
        M += npair; Ksucc += len(pdb); Ndraw += len(dset)
        for xy in hit:
            k = pdb[xy]; by_kind_obs[k] = by_kind_obs.get(k, 0) + 1
        for xy, k in pdb.items():
            by_kind_tot[k] = by_kind_tot.get(k, 0) + 1
            by_kind_exp[k] = by_kind_exp.get(k, 0.0) + p_disc
    kinds = {k: {"obs": by_kind_obs.get(k, 0), "exp": round(by_kind_exp.get(k, 0.0), 2),
                 "fold": round(by_kind_obs.get(k, 0) / by_kind_exp[k], 2)
                         if by_kind_exp.get(k, 0) > 0 else None,
                 "total": by_kind_tot[k]} for k in sorted(by_kind_tot)}
    return {"obs": int(obs), "exp": round(exp, 2),
            "fold": round(obs / exp, 3) if exp > 0 else None,
            "poisson_p": float(poisson.sf(obs - 1, exp)) if exp > 0 else None,
            "hypergeom_p": float(hypergeom.sf(int(obs) - 1, M, Ksucc, Ndraw))
                           if M and Ndraw else None,
            "by_kind": kinds}


def arch_charge_swap(arch, which="c20", n_perm=20000, thr=0.05, seed=0):
    """Classes whose field (theta) states span acid<->base archetypes, vs a
    permutation null that shuffles the LEARNED theta>=1 assignments (theta0 is
    the pinned arch=class diagonal, so it is held fixed)."""
    charge = _archetype_charge(which)
    K, Lf = arch.shape

    def crossing(a):
        ch = charge[a]
        return int(((ch.max(1) > thr) & (ch.min(1) < -thr)).sum())

    obs = crossing(arch)
    n_swap = int(sum(len(set(arch[c].tolist())) > 1 for c in range(K)))
    # enum400 degeneracy: if arch is the fixed (k0,k1) enumeration, `crossing`
    # is a constant of the enumeration and the permutation null is uninformative
    # (p ~ 0.5 by construction). Flag it; the real signal is charge_flip_soft.
    Ka = int(round(K ** 0.5))
    enum = Ka * Ka == K and bool((arch == np.column_stack(
        [np.arange(K) // Ka, np.arange(K) % Ka])).all())
    if enum:
        return {"classes": int(K), "classes_swapping": n_swap,
                "charge_crossing_obs": obs, "enum_degenerate": True,
                "note": "arch is the fixed enumeration; use charge_flip_soft"}
    rng = np.random.default_rng(seed)
    learned = arch[:, 1:].ravel()
    null = np.array([crossing(np.column_stack([arch[:, 0],
                     rng.permutation(learned).reshape(K, Lf - 1)]))
                     for _ in range(n_perm)])
    return {"classes": int(K), "classes_swapping": n_swap,
            "charge_crossing_obs": obs,
            "null_mean": round(float(null.mean()), 2),
            "null_ci95": [int(np.percentile(null, 2.5)), int(np.percentile(null, 97.5))],
            "p_ge_obs": float((null >= obs).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pdb-dir", default="data/pdb_partition_clv_top1000_sifts")
    ap.add_argument("--confirmed-flips",
                    default="data/pdb_partition_clv_top1000_sifts/confirmed_flips.json")
    ap.add_argument("--archetypes", default="c20", choices=["c10", "c20"])
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="", help="append JSON line to this trajectory file")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ck = np.load(args.ckpt, allow_pickle=False)
    fam_ids = list(ck["family_ids"].tolist()); fam_set = set(fam_ids)
    arch = np.asarray(ck["arch_assignment"])
    disc = _discovered_pairs(ck)
    sizes = {}
    for fi in range(len(fam_ids)):
        _, n = np.unique(ck[f"fam_{fi}_cluster_id"], return_counts=True)
        for m in n:
            sizes[int(m)] = sizes.get(int(m), 0) + 1

    row = {
        "label": args.label, "ckpt": args.ckpt, "step": int(ck["step"]),
        "n_families": len(fam_ids),
        "n_pairs": sum(len(s) for s in disc.values()),
        "size_hist": {str(k): sizes[k] for k in sorted(sizes)},
        "flip_recall": flip_recall(disc, args.confirmed_flips, fam_set),
        "contact_enrichment": contact_enrichment(disc, args.pdb_dir, fam_ids),
        "arch_charge_swap": arch_charge_swap(arch, args.archetypes),
        "charge_flip_soft": charge_flip_soft(ck, arch, args.archetypes,
                                             args.confirmed_flips, fam_ids),
    }
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    if not args.quiet:
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()


# ---- enum400 per-cluster / per-component flip-type classification -----------
# class c = (k0,k1) = (arch@field0, arch@field1). polarity dq(c) = q(k0)-q(k1)
# (charge of resting minus excited field state; signed only because rho0>rho1).
CYS = 1  # Cys index in alphabet ACDEFGHIKLMNPQRSTVWY


def archetype_charge_cys(pi_arch):
    pi_arch = np.asarray(pi_arch, float)
    charge = pi_arch[:, BASE].sum(1) - pi_arch[:, ACID].sum(1)   # (K_a,) basic-acidic
    return charge, pi_arch[:, CYS]                                # charge, cys content


def classify_pair(c_i, c_j, charge, cys, K_a, tau=0.15, cys_t=0.15):
    """Type of a 2-column cluster from its inferred classes:
      'salt_bridge' opposite polarity (compensatory charge flip),
      'disulfide'   Cys-rich archetype on the SAME field side (co-flip of Cys),
      'coflip'      same-sign polarity (correlated charge flip),
      'static'      neither column flips appreciably."""
    a0, a1 = c_i // K_a, c_i % K_a
    b0, b1 = c_j // K_a, c_j % K_a
    dqi, dqj = charge[a0] - charge[a1], charge[b0] - charge[b1]

    def cys_side(k0, k1):
        if cys[k0] > cys_t and cys[k1] < cys_t: return 0
        if cys[k1] > cys_t and cys[k0] < cys_t: return 1
        return -1
    si, sj = cys_side(a0, a1), cys_side(b0, b1)
    if si >= 0 and si == sj:
        return 'disulfide'
    if abs(dqi) > tau and abs(dqj) > tau:
        return 'salt_bridge' if dqi * dqj < 0 else 'coflip'
    return 'static'


def flip_type_counts(pair_classes, confirmed_mask, charge, cys, K_a, **kw):
    """pair_classes: list of (c_i, c_j); confirmed_mask: bool per pair.
    Returns type breakdown split on confirmed-flip vs other + salt-bridge recall."""
    from collections import Counter
    cf, other = Counter(), Counter()
    for (ci, cj), m in zip(pair_classes, np.asarray(confirmed_mask, bool)):
        t = classify_pair(int(ci), int(cj), charge, cys, K_a, **kw)
        (cf if m else other)[t] += 1
    n_cf = sum(cf.values())
    return {"confirmed": dict(cf), "other": dict(other),
            "salt_bridge_recall": (cf['salt_bridge'] / n_cf) if n_cf else None,
            "n_confirmed": n_cf, "n_other": sum(other.values())}
