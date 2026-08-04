"""Decompose the pairing likelihood to locate the flip mis-ranking defect.

For a cached pair (i,j) with top-N^2 class grid LL_pair[a,b] over top_i x top_j and
per-column singleton vectors sing_ll_i/j, define the pointwise COUPLING GAIN

    g[a,b] = LL_pair[a,b] - sing_ll_i[c_i] - sing_ll_j[c_j]        (c = class index)

= how much the SHARED field makes class-pair (c_i,c_j) more likely than two
INDEPENDENT singletons (the coevolution reward, in nats). classify_pair labels each
(a,b) as salt_bridge (antipolar / compensatory), coflip, static, disulfide.

The question: for a true flip, does the antipolar class-pair have
  (A) LOW g  -> the shared-field generator fails to reward anti-phase (generator fix), or
  (B) HIGH g but LOW posterior weight q=softmax(LL_pair) -> the top-N^2 marginalization
      is swamped by non-flip class-pairs (marginalization-weight fix)?

Groups: (2) all confirmed flips, (3) salt bridges that do NOT flip, (4) contacts
that are NOT salt bridges. Plus (1) one strong single flip pair, verbose.
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "experiments"); sys.path.insert(0, "analysis/scripts")
from tkfdp.coupling.dynfield.phylo_elbo.pairing_cache import PairingCache
from precompute_pairing import _archetypes, perm_null_z
from dynfield_metrics import archetype_charge_cys, classify_pair

PDIR = Path("data/pdb_partition_clv_top1000_sifts")
CLV = "data/pfam_processed_clv_top1000_thin128"
pc = PairingCache("data/pairing_cache")
PA = _archetypes(); CHG, CYS = archetype_charge_cys(PA); K_A = 20
TYPES = ["salt_bridge", "coflip", "static", "disulfide"]


def _softmax(v):
    v = np.asarray(v, float) - np.max(v); e = np.exp(v); return e / e.sum()


def decompose(fam, i, j):
    r = pc.get(fam, i, j)
    si = pc.sing_ll(fam, i); sj = pc.sing_ll(fam, j)
    if r is None or si is None or sj is None:
        return None
    ti = r["top_i"].astype(int); tj = r["top_j"].astype(int)
    LL = np.asarray(r["LL_pair"], float)                       # (N,N)
    N = LL.shape[0]
    g = np.zeros((N, N)); typ = np.empty((N, N), object)
    for a in range(N):
        for b in range(N):
            ci, cj = int(ti[a]), int(tj[b])
            g[a, b] = LL[a, b] - si[ci] - sj[cj]               # coupling gain
            typ[a, b] = classify_pair(ci, cj, CHG, CYS, K_A)
    q = _softmax(LL.ravel()).reshape(N, N)                     # class-pair posterior
    out = {"LR": pc.logodds(fam, i, j, 1.0)}                   # pure likelihood ratio (no Ewens)
    for t in TYPES:
        m = np.array([[typ[a, b] == t for b in range(N)] for a in range(N)])
        out[t] = dict(n=int(m.sum()), post_wt=float(q[m].sum()),
                      g_mean=float(g[m].mean()) if m.any() else np.nan,
                      g_max=float(g[m].max()) if m.any() else np.nan)
    # MAP class-pair
    am = np.unravel_index(np.argmax(LL), LL.shape)
    out["map_type"] = typ[am]; out["g_overall_qmean"] = float((q * g).sum())
    return out


def agg(pairs, label):
    recs = [decompose(f, i, j) for (f, i, j) in pairs]
    recs = [r for r in recs if r is not None]
    if not recs:
        print(f"# {label}: no cached pairs"); return
    print(f"\n# ===== {label}  (n={len(recs)} cached) =====")
    print(f"#   mean pairing LR (pair - singletons, no Ewens): {np.mean([r['LR'] for r in recs]):+.2f}")
    print(f"#   MAP class-pair type distribution: " +
          ", ".join(f"{t}={sum(r['map_type']==t for r in recs)}" for t in TYPES))
    print(f"#   q-weighted mean coupling gain (all types): {np.mean([r['g_overall_qmean'] for r in recs]):+.2f} nats")
    print(f"#   {'type':12s} {'post_wt':>8s} {'g_mean':>8s} {'g_max':>8s}   (post_wt=marginalization weight; g=coupling reward)")
    for t in TYPES:
        pw = np.mean([r[t]["post_wt"] for r in recs])
        gm = np.nanmean([r[t]["g_mean"] for r in recs])
        gx = np.nanmean([r[t]["g_max"] for r in recs])
        print(f"#   {t:12s} {pw:8.3f} {gm:+8.2f} {gx:+8.2f}")


def load_groups():
    cf = json.load(open(PDIR / "confirmed_flips.json"))
    flips = {(r["family"], min(int(r["i"]), int(r["j"])), max(int(r["i"]), int(r["j"]))) for r in cf}
    sb, other = set(), set()
    for p in glob.glob(str(PDIR / "*.npz")):
        if "confirmed" in p:
            continue
        fam = Path(p).stem; d = np.load(p)
        if "pairs" not in d.files:
            continue
        for (a, b), k in zip(d["pairs"], d["kind"]):
            key = (fam, int(min(a, b)), int(max(a, b)))
            if str(k) == "saltbridge":
                sb.add(key)
            else:
                other.add(key)
    sb_noflip = sb - flips
    return flips, sb_noflip, other


def main():
    flips, sb_noflip, other = load_groups()
    # (1) pick one strong single flip pair: cached, strongest antipolar coupling
    best = None
    for (f, i, j) in list(flips):
        d = decompose(f, i, j)
        if d is None or d["salt_bridge"]["n"] == 0:
            continue
        score = d["salt_bridge"]["g_max"]          # strongest antipolar coupling gain
        if best is None or score > best[0]:
            best = (score, f, i, j, d)
    if best:
        _, f, i, j, d = best
        # one miz for the chosen pair only
        try:
            msa = np.load(f"{CLV}/{f}.npz", allow_pickle=True)["leaf_msa"]
            z = float(perm_null_z(msa, K=60)[i, j])
        except Exception:
            z = float("nan")
        print(f"# ===== (1) SINGLE strong flip pair: {f} ({i},{j})  miz={z:.1f}  LR={d['LR']:+.2f}  MAP={d['map_type']} =====")
        for t in TYPES:
            e = d[t]
            print(f"#   {t:12s}: n={e['n']:2d}  post_wt={e['post_wt']:.3f}  g_mean={e['g_mean']:+.2f}  g_max={e['g_max']:+.2f}")

    agg(list(flips), "(2) TRUE SALT BRIDGES THAT FLIP (confirmed flips)")
    agg(list(sb_noflip), "(3) TRUE SALT BRIDGES THAT DO NOT FLIP")
    agg(list(other), "(4) CONTACTS THAT ARE NOT SALT BRIDGES")


if __name__ == "__main__":
    main()
