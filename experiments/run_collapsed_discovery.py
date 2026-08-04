"""Corpus-scale collapsed discovery from the pairing cache.

The site class is Rao-Blackwellized (marginalized over the top-N grid in the
cache); we sample ONLY the size-{1,2} partition via the collapsed CRP z-move
(collapsed_discovery.z_sweep), reading class-marginalized pairing log-odds from
the PairingCache. No cn-move, no site-class labels.

Because the cache stores LL_pair UNSUMMED, the DM class prior enters purely at
lookup time: --dm none => flat class prior; --dm <chkpt.npz> => load the run's
dm_alpha/dm_pi mixture and reweight the cached evidence with it.

Evaluation (honest about the split): the cached families are the TRAIN split, so
discovered-vs-PDB here is IN-SAMPLE -- a sanity check that the collapsed sampler
recovers known contacts, NOT a generalization number. The key readout is whether
the model's keep/reject enriches for real PDB contacts BEYOND the miz shortlist
that fed the cache (hypergeometric). A held-out number needs a test-split cache
(precompute_pairing on --split test), evaluated the same way.

  python experiments/run_collapsed_discovery.py --sweeps 40 --burnin 10 \
      --alpha-z 100 --dm none --split train --kinds saltbridge --out results/collapsed_train470
"""
from __future__ import annotations
import argparse, glob, json, time
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, "src")
from tkfdp.coupling.dynfield.phylo_elbo.collapsed_discovery import CollapsedState, z_sweep
from tkfdp.bio import load_split

CACHE = "data/pairing_cache"
PDIR = Path("data/pdb_partition_clv_top1000_sifts")


class _DM:
    """Minimal DM-mixture view (H, pi, alpha) matching the interface
    collapsed_discovery / pairing_cache use for reweighting."""
    def __init__(self, alpha, pi):
        self.alpha = np.asarray(alpha, float)          # (H, K_c)
        self.pi = np.asarray(pi, float)
        self.H = int(self.alpha.shape[0])


def load_dm(spec):
    if spec in (None, "none", "flat"):
        return None
    d = np.load(spec)
    dm = _DM(d["dm_alpha"], d["dm_pi"])
    if float(np.std(dm.alpha)) == 0.0:
        print(f"# WARNING: {spec} DM is uniform (untrained) -> equivalent to flat", flush=True)
    return dm


def load_pdb(split, kinds):
    """{fam: {'contacts':set, 'flips':set}} for split families with a PDB partition."""
    keep = set(load_split()[split]) if split != "all" else None
    out = {}
    for p in sorted(glob.glob(str(PDIR / "*.npz"))):
        if "confirmed" in p:
            continue
        fam = Path(p).stem
        if keep is not None and fam not in keep:
            continue
        d = np.load(p)
        if "pairs" not in d.files:
            continue
        sel = {(min(int(a), int(b)), max(int(a), int(b)))
               for (a, b), k in zip(d["pairs"], d["kind"]) if str(k) in kinds}
        allc = {(min(int(a), int(b)), max(int(a), int(b))) for (a, b) in d["pairs"]}
        out[fam] = {"contacts": sel, "all_contacts": allc, "flips": set()}
    for r in json.load(open(PDIR / "confirmed_flips.json")):
        f = r["family"]
        if keep is not None and f not in keep:
            continue
        out.setdefault(f, {"contacts": set(), "all_contacts": set(), "flips": set()})
        out[f]["flips"].add((min(int(r["i"]), int(r["j"])), max(int(r["i"]), int(r["j"]))))
    return out


def hypergeom_sf(k, N, K, n):
    """P[X >= k] for X ~ Hypergeom(N pop, K successes, n draws]."""
    from scipy.stats import hypergeom
    return float(hypergeom.sf(k - 1, N, K, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=40)
    ap.add_argument("--burnin", type=int, default=10)
    ap.add_argument("--alpha-z", type=float, default=100.0)
    ap.add_argument("--dm", default="none", help="'none' (flat) or path to a _chkpt.npz with dm_alpha/dm_pi")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--cache-dir", default=CACHE, help="pairing cache dir (default data/pairing_cache)")
    ap.add_argument("--flip-families", action="store_true",
                    help="restrict to the confirmed-flip families (for the field-rate flip test)")
    ap.add_argument("--kinds", default="saltbridge")
    ap.add_argument("--freq-thresh", type=float, default=0.5, help="marginal pairing freq to call a pair discovered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kinds = set(args.kinds.split(","))
    # families = cached AND in split
    cache_dir = args.cache_dir
    cached = {Path(p).name.replace(".pairev.npz", "") for p in glob.glob(f"{cache_dir}/*.pairev.npz")}
    fams = sorted(cached & set(load_split()[args.split])) if args.split != "all" else sorted(cached)
    if args.flip_families:
        cf = json.load(open(PDIR / "confirmed_flips.json"))
        flip_fams = {r["family"] for r in cf}
        fams = sorted(set(fams) & flip_fams)
    print(f"# collapsed discovery: {len(fams)} cached families in split={args.split}  "
          f"alpha_z={args.alpha_z}  dm={args.dm}", flush=True)

    dm = load_dm(args.dm)
    st = CollapsedState(cache_dir, fams, alpha_z=args.alpha_z)
    # shortlist size = number of candidate (cached) pairs
    shortlist = {}
    for fam in st.fams:
        rec = st.pc._family(fam)
        if rec is None:
            continue
        _, idx, _ = rec
        shortlist[fam] = {(int(min(a, b)), int(max(a, b))) for (a, b) in idx.keys()}
    n_short = sum(len(v) for v in shortlist.values())

    rng = np.random.default_rng(args.seed)
    freq: dict = {}
    n_post = 0
    t0 = time.time()
    for s in range(args.sweeps):
        nmoves = z_sweep(st, rng, dm)
        if s >= args.burnin:
            n_post += 1
            for fam, ps in st.pairs().items():
                for ij in ps:
                    freq[(fam, ij)] = freq.get((fam, ij), 0) + 1
        if s % 5 == 0 or s == args.sweeps - 1:
            npair = sum(len(v) for v in st.pairs().values())
            print(f"#  sweep {s:3d}: {nmoves:5d} pair-moves, {npair} pairs held  [{time.time()-t0:.0f}s]", flush=True)

    # discovered = marginal pairing freq >= thresh over post-burnin sweeps
    thr = args.freq_thresh * max(n_post, 1)
    disc = {}
    for (fam, ij), c in freq.items():
        if c >= thr:
            disc.setdefault(fam, set()).add(ij)
    n_disc = sum(len(v) for v in disc.values())

    # ---- eval vs PDB (in-sample if split==train) ----
    pdb = load_pdb(args.split, kinds)
    def overlap(pairset_by_fam, key):
        hit = 0; tot = 0
        for fam, ps in pairset_by_fam.items():
            g = pdb.get(fam, {}).get(key, set())
            hit += len(ps & g); tot += len(ps)
        return hit, tot
    n_pdb_kind = sum(len(pdb[f]["contacts"]) for f in pdb)
    n_pdb_all = sum(len(pdb[f]["all_contacts"]) for f in pdb)
    n_flip = sum(len(pdb[f]["flips"]) for f in pdb)

    sl_hit_k, _ = overlap(shortlist, "contacts")
    di_hit_k, _ = overlap(disc, "contacts")
    sl_hit_a, _ = overlap(shortlist, "all_contacts")
    di_hit_a, _ = overlap(disc, "all_contacts")
    sl_flip, _ = overlap(shortlist, "flips")
    di_flip, _ = overlap(disc, "flips")

    def prec(hit, tot): return hit / tot if tot else 0.0
    # enrichment: is the discovered set's contact hit-rate above the shortlist's?
    p_enrich = hypergeom_sf(di_hit_a, n_short, sl_hit_a, n_disc) if n_disc and sl_hit_a else 1.0

    print("\n# ===== RESULTS (split=%s%s) =====" % (
        args.split, "  [IN-SAMPLE: cache is train]" if args.split == "train" else "  [held-out]"))
    print(f"#  shortlist (miz-passed candidate pairs): {n_short}")
    print(f"#  discovered (collapsed z-move, freq>= {args.freq_thresh}): {n_disc}"
          f"  ({100*n_disc/max(n_short,1):.1f}% of shortlist kept)")
    print(f"#  PDB ground truth: {n_pdb_kind} {args.kinds} contacts, {n_pdb_all} all-kind contacts, {n_flip} confirmed flips")
    print(f"#  --- {args.kinds}-contact precision ---")
    print(f"#     shortlist:  {sl_hit_k}/{n_short} = {prec(sl_hit_k,n_short):.3f}")
    print(f"#     discovered: {di_hit_k}/{n_disc} = {prec(di_hit_k,n_disc):.3f}")
    print(f"#  --- all-PDB-contact precision (enrichment target) ---")
    print(f"#     shortlist:  {sl_hit_a}/{n_short} = {prec(sl_hit_a,n_short):.3f}")
    print(f"#     discovered: {di_hit_a}/{n_disc} = {prec(di_hit_a,n_disc):.3f}   (hypergeom P[>=] = {p_enrich:.2g})")
    print(f"#  --- confirmed-flip recall ---")
    print(f"#     shortlist:  {sl_flip}/{n_flip} = {prec(sl_flip,n_flip):.3f}")
    print(f"#     discovered: {di_flip}/{n_flip} = {prec(di_flip,n_flip):.3f}")

    if args.out:
        outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
        summary = dict(split=args.split, kinds=args.kinds, alpha_z=args.alpha_z, dm=args.dm,
                       sweeps=args.sweeps, burnin=args.burnin, freq_thresh=args.freq_thresh,
                       n_families=len(fams), n_shortlist=n_short, n_discovered=n_disc,
                       n_pdb_kind=n_pdb_kind, n_pdb_all=n_pdb_all, n_flip=n_flip,
                       shortlist_kind_prec=prec(sl_hit_k, n_short), disc_kind_prec=prec(di_hit_k, n_disc),
                       shortlist_all_prec=prec(sl_hit_a, n_short), disc_all_prec=prec(di_hit_a, n_disc),
                       enrich_p=p_enrich, shortlist_flip_recall=prec(sl_flip, n_flip),
                       disc_flip_recall=prec(di_flip, n_flip))
        json.dump(summary, open(outd / "summary.json", "w"), indent=2)
        disc_ser = {f: sorted(list(map(list, ps))) for f, ps in disc.items()}
        json.dump(disc_ser, open(outd / "discovered_pairs.json", "w"))
        print(f"\n# wrote {outd}/summary.json + discovered_pairs.json", flush=True)


if __name__ == "__main__":
    main()
