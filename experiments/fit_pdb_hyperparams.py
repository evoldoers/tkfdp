"""Ultra-efficient supervised hyperparameter fit on the PDB contact partition.

The partition is FIXED (known contacts), so there is nothing to sample: score the
contact pairs' evidence ONCE (per field-rate bin, over the top-N^2 class grid) and
fit the field-rate bin weights and the DM pseudocounts in closed form.

  * field-rate weights  -- mixture EM over the +Gamma+I rate bins:
        E: r_g(pair) prop w_g * exp(LL_g),  LL_g = class-marginal at bin g
        M: w_g = mean_pair r_g
    The invariant (rate-0) weight it converges to IS the flip-prevalence readout:
    a low fitted invariant weight => real contacts flip; a high one => the model
    reads them as static (the charge-under-read, measured at the source).
  * DM pseudocounts alpha -- Minka fixed point on the expected class counts, where
    each pair's class posterior is q(c_i,c_j) prop alpha_{c_i} alpha_{c_j} exp(LL).

Coordinate-ascent between the two (the class prior couples them). Uses the top-N
class approximation, as the cache does.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
from scipy.special import digamma

import sys
sys.path.insert(0, "experiments")
import precompute_pairing as PP
from tkfdp.coupling.dynfield.phylo_elbo import field_rate_discovery as fd

PDIR = Path("data/pdb_partition_clv_top1000_sifts")
CLV = Path("data/pfam_processed_clv_top1000_thin128")


def _lse(v, axis=None):
    v = np.asarray(v, float); m = v.max(axis=axis, keepdims=True)
    return np.squeeze(np.log(np.sum(np.exp(v - m), axis=axis, keepdims=True)) + m, axis=axis)


def load_pairs(kinds, max_pairs=None, seed=0, split="train"):
    """{fam: [(i,j)]} for the requested contact kinds + confirmed flips."""
    from tkfdp.bio import load_split
    keep = set(load_split()[split])
    out = {}
    for p in sorted(glob.glob(str(PDIR / "*.npz"))):
        if "confirmed" in p:
            continue
        fam = Path(p).stem
        if fam not in keep or not (CLV / f"{fam}.npz").exists():
            continue
        d = np.load(p, allow_pickle=False)
        if "pairs" not in d.files:
            continue
        sel = [(int(a), int(b)) for (a, b), k in zip(d["pairs"], d["kind"])
               if str(k) in kinds]
        if sel:
            out[fam] = sel
    cf = json.load(open(PDIR / "confirmed_flips.json"))
    for r in cf:
        if r["family"] in keep and (CLV / f"{r['family']}.npz").exists():
            out.setdefault(r["family"], [])
            ij = (int(r["i"]), int(r["j"]))
            if ij not in out[r["family"]]:
                out[r["family"]].append(ij)
    if max_pairs:
        allp = [(f, ij) for f, ps in out.items() for ij in ps]
        rng = np.random.default_rng(seed)
        allp = [allp[i] for i in rng.permutation(len(allp))[:max_pairs]]
        out = {}
        for f, ij in allp:
            out.setdefault(f, []).append(ij)
    return out


def score_perbin(ds, byfam, topN=8):
    """Per-pair (topN,topN,G) pair grid + (topN,) class indices, per-bin (NOT
    rate-marginalized). Uses the mm forward (_score_specs_rates -> (n,G))."""
    st = ds.state; Kc = st.K_c; G = len(ds.rates)
    mb1 = st.m_bucket_for(1); mb2 = st.m_bucket_for(2)
    fam_ids = [f.family_id for f in st.families]
    recs = []
    for fam, pairs in byfam.items():
        fi = fam_ids.index(fam)
        cols = sorted({c for ij in pairs for c in ij})
        s_specs, s_slot = [], {}
        for col in cols:
            for c in range(Kc):
                s_slot[(col, c)] = len(s_specs)
                cl = np.zeros(mb1, np.int32); cl[0] = c
                s_specs.append((fi, np.array([col], np.int32), cl, mb1))
        ll_s = fd._score_specs_rates(ds, s_specs)                    # (n,G)
        sing = {col: np.stack([ll_s[s_slot[(col, c)]] for c in range(Kc)]) for col in cols}  # (K_c,G)
        # rate-marg singleton to pick top-N per column
        logw = ds.logw
        sm = {col: _lse(sing[col] + logw[None, :], axis=1) for col in cols}   # (K_c,)
        tops = {col: np.argsort(-sm[col])[:topN] for col in cols}
        p_specs, p_slot = [], {}
        for (i, j) in pairs:
            for a, ci in enumerate(tops[i]):
                for b, cj in enumerate(tops[j]):
                    p_slot[(i, j, a, b)] = len(p_specs)
                    cl = np.zeros(mb2, np.int32); cl[0] = ci; cl[1] = cj
                    p_specs.append((fi, np.array([i, j], np.int32), cl, mb2))
        ll_p = fd._score_specs_rates(ds, p_specs)                    # (n,G)
        for (i, j) in pairs:
            grid = np.array([[ll_p[p_slot[(i, j, a, b)]] for b in range(topN)]
                             for a in range(topN)])                  # (topN,topN,G)
            recs.append(dict(fam=fam, i=i, j=j, grid=grid,
                             top_i=tops[i], top_j=tops[j]))
    return recs, G, Kc


def fit(recs, G, Kc, logw0, n_iter=60, alpha0=1.0):
    w = np.exp(logw0); w /= w.sum()
    alpha = np.full(Kc, float(alpha0))
    for it in range(n_iter):
        la = np.log(alpha)
        # per-pair grid class log-prior (single-component DM), up to a constant
        # ---- field-rate EM (E/M) with current class prior ----
        num = np.zeros(G)
        for r in recs:
            lp = la[r["top_i"]][:, None] + la[r["top_j"]][None, :]    # (N,N)
            LL_g = _lse((r["grid"] + lp[:, :, None]).reshape(-1, G), axis=0)  # (G,)
            rg = np.log(w + 1e-300) + LL_g; rg -= rg.max()
            rg = np.exp(rg); rg /= rg.sum()
            num += rg
        w = num / num.sum()
        # ---- DM Minka from class posteriors (rate-marginalized w new w) ----
        counts = np.zeros(Kc); ncl = 0
        for r in recs:
            lp = la[r["top_i"]][:, None] + la[r["top_j"]][None, :]
            LLm = _lse(r["grid"] + np.log(w + 1e-300)[None, None, :], axis=2)  # (N,N)
            q = LLm + lp; q -= q.max(); q = np.exp(q); q /= q.sum()
            counts[r["top_i"]] += q.sum(1)                           # c_i marginal
            counts[r["top_j"]] += q.sum(0)                           # c_j marginal
            ncl += 1
        # Minka fixed-point (symmetric prior gets asymmetric via counts)
        A = alpha.sum(); m = 2.0                                     # 2 classes/cluster
        num_a = counts / np.maximum(ncl, 1) * (digamma(A + m) - digamma(A))  # crude 1-step
        alpha = np.maximum(num_a, 1e-3)
    return w, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="saltbridge")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--max-pairs", type=int, default=300)
    ap.add_argument("--topN", type=int, default=8)
    args = ap.parse_args()
    byfam = load_pairs(set(args.kinds.split(",")), max_pairs=args.max_pairs, split=args.split)
    npairs = sum(len(v) for v in byfam.values())
    print(f"# supervised set: {npairs} pairs over {len(byfam)} families "
          f"(kinds={args.kinds} + confirmed flips)", flush=True)
    ds, rates, weights = PP.build_enum400_ds(list(byfam.keys()))
    print(f"# rates={np.round(rates,3)}  prior weights={np.round(weights,3)}", flush=True)
    import time; t0 = time.time()
    recs, G, Kc = score_perbin(ds, byfam, topN=args.topN)
    print(f"# scored {len(recs)} pairs per-bin in {time.time()-t0:.0f}s", flush=True)
    w, alpha = fit(recs, G, Kc, ds.logw)
    print(f"\n# FITTED field-rate weights: {np.round(w,3)}  (rates {np.round(rates,3)})")
    print(f"#   invariant(rate-0) weight: prior {weights[0]:.3f} -> fitted {w[0]:.3f}")
    print(f"#   flip-prevalence (1 - invariant): prior {1-weights[0]:.3f} -> fitted {1-w[0]:.3f}")
    top = np.argsort(-alpha)[:8]
    Ka = 20
    print(f"# top DM-alpha classes: " +
          ", ".join(f"{c}=({c//Ka},{c%Ka}):{alpha[c]:.2f}" for c in top))


if __name__ == "__main__":
    main()
