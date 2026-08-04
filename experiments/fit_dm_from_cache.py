"""Fit a DM (Dirichlet-multinomial) mixture prior from the PDB-labeled contact
pairs' class posteriors, read straight off the pairing cache.

This is the cheap fit the old enum400dm run never reached: the site class is
already marginalized in the cache (top-N^2 grid of rate-marginalized LL_pair per
pair), so there is NO per-column class Gibbs -- we fit {alpha_h, pi_h} by a
soft EM over (component h, class-pair) directly from the cached evidence.

Field rate is UNTOUCHED (we only read LL_pair, which is already rate-marginalized
at the cache's field-rate weights), so the cache stays valid -- the fitted DM is
consumed purely as a query-time class-prior reweight in collapsed_discovery.

Model per PDB cluster p=(i,j):  h ~ pi ; {c_i,c_j} ~ DM(alpha_h) ; data ~ exp(LL_pair).
Grid prior matches pairing_cache.dm_logprior_grid: log a_h[c_i]+log a_h[c_j]-2 log A_h.

  E: r_p(h) prop pi_h * sum_{a,b} exp(gp_h[a,b] + LL_pair[a,b])
     q_p^h(a,b) = softmax_{a,b}(gp_h + LL_pair)   (within-component class posterior)
  M: pi_h = mean_p r_p(h)
     alpha_h: responsibility-weighted Minka on the expected count vectors n_p^h.

  python experiments/fit_dm_from_cache.py --kinds saltbridge --split train \
      --H 10 --n-iter 40 --out results/dm_fit_pdb_train
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.special import digamma

import sys
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
from tkfdp.coupling.dynfield.phylo_elbo.pairing_cache import PairingCache
from fit_pdb_hyperparams import load_pairs

CACHE = "data/pairing_cache"
K_A = 20                                                # archetypes -> class = (c//20, c%20)


def _lse_ax(v, axis):
    v = np.asarray(v, float); m = v.max(axis=axis, keepdims=True)
    return np.squeeze(np.log(np.exp(v - m).sum(axis=axis, keepdims=True)) + m, axis=axis)


def load_clusters(byfam):
    """Clusters of the PDB-labeled partition, read from the cache:
      pairs     -- [(top_i, top_j, LL_pair)]  size-2, the PDB contacts
      singles   -- [sing_ll (K_c,)]           size-1, every OTHER shortlist column
    Under the PDB partition a contact column is paired; all other shortlist
    columns (the ones discovery actually deliberates over, and the only ones whose
    singleton score enters a log-odds) are singletons. Both cluster types share
    the SAME mixture components -- that shared fit is what makes the pair-vs-
    singleton differential (the correlation the flip signal lives in) honest."""
    pc = PairingCache(CACHE)
    pairs, singles, n_miss = [], [], 0
    for fam, plist in byfam.items():
        contact_cols = set()
        for (i, j) in plist:
            r = pc.get(fam, i, j)
            if r is None:
                n_miss += 1; continue
            pairs.append((r["top_i"].astype(int), r["top_j"].astype(int),
                          np.asarray(r["LL_pair"], float)))
            contact_cols.add(int(i)); contact_cols.add(int(j))
        rec = pc._family(fam)
        if rec is None:
            continue
        _, _, scol = rec
        for col in scol:                                 # shortlist columns
            if col in contact_cols:
                continue                                 # already in a pair cluster
            s = pc.sing_ll(fam, col)
            if s is not None:
                singles.append(np.asarray(s, float))
    return pairs, singles, n_miss


def fit_dm(pairs, singles, K_c=400, H=10, alpha0=1.0, n_iter=40, floor=1e-3, seed=0):
    """Joint DM-mixture EM over BOTH size-2 (double-draw) and size-1 (single-draw)
    clusters, sharing components -- the DM analog of the pairs+singletons HR EM.
    A cluster's draw count m (2 for a pair, 1 for a singleton) sets the DM
    normalizer (-m log A) and the Minka draw term digamma(m+A)-digamma(A)."""
    n_clu = len(pairs) + len(singles)
    # break label symmetry (symmetric init keeps all components identical forever)
    rng = np.random.default_rng(seed)
    alpha = float(alpha0) * np.exp(0.6 * rng.standard_normal((H, K_c)))
    pi = np.full(H, 1.0 / H)
    hist = []
    for it in range(n_iter):
        la = np.log(alpha)                               # (H,K_c)
        A = alpha.sum(1); lA = np.log(A)                 # (H,)
        dig_a = digamma(alpha); dig_A = digamma(A)
        num_acc = np.zeros((H, K_c)); den_acc = np.zeros(H)
        R_sum = np.zeros(H); total_ll = 0.0; lpi = np.log(pi + 1e-300)

        def _accum(lm_h, ncount):                        # lm_h:(H,), ncount:(H,K_c) expected counts
            nonlocal num_acc, den_acc, R_sum, total_ll
            lp = lpi + lm_h; mx = lp.max()
            r = np.exp(lp - mx); Z = r.sum(); r = r / Z
            total_ll += mx + np.log(Z)
            R_sum += r
            m = ncount.sum(1)                            # (H,) draws per cluster (=1 or 2)
            num_acc += r[:, None] * (digamma(ncount + alpha) - dig_a)
            den_acc += r * (digamma(m + A) - dig_A)

        # ---- size-2 clusters (double draw) ----
        for (ti, tj, LL) in pairs:
            gp = la[:, ti][:, :, None] + la[:, tj][:, None, :] - 2.0 * lA[:, None, None]
            joint = LL[None] + gp                        # (H,N,N)
            lm = _lse_ax(joint.reshape(H, -1), axis=1)   # (H,)
            q = np.exp(joint - lm[:, None, None])        # (H,N,N) normalized per comp
            nc = np.zeros((H, K_c))
            np.add.at(nc.T, ti, q.sum(2).T)              # c_i marginal -> classes top_i
            np.add.at(nc.T, tj, q.sum(1).T)              # c_j marginal -> classes top_j
            _accum(lm, nc)
        # ---- size-1 clusters (single draw) ----
        for s in singles:
            sp = la - lA[:, None]                         # (H,K_c) single-class prior
            joint = s[None] + sp                          # (H,K_c)
            lm = _lse_ax(joint, axis=1)                   # (H,)
            q = np.exp(joint - lm[:, None])               # (H,K_c) normalized
            _accum(lm, q)                                 # expected 1-hot counts = q

        pi = R_sum / max(n_clu, 1); pi = pi / pi.sum()
        for h in range(H):
            if den_acc[h] > 1e-9:
                alpha[h] = np.maximum(alpha[h] * num_acc[h] / den_acc[h], floor)
        hist.append(float(total_ll))
    return alpha, pi, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="saltbridge")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--out", default="results/dm_fit_pdb_train")
    args = ap.parse_args()

    byfam = load_pairs(set(args.kinds.split(",")), max_pairs=args.max_pairs, split=args.split)
    pairs, singles, n_miss = load_clusters(byfam)
    print(f"# DM fit (log-LIKELIHOOD of the PDB-labeled partition): "
          f"{len(pairs)} couples (double-draw) + {len(singles)} singletons (single-draw)  "
          f"({n_miss} PDB pairs not in shortlist, dropped) over {len(byfam)} families  "
          f"split={args.split} kinds={args.kinds}", flush=True)
    if not pairs and not singles:
        print("# no cached clusters -- nothing to fit"); return

    alpha, pi, hist = fit_dm(pairs, singles, H=args.H, n_iter=args.n_iter, alpha0=args.alpha0)
    print(f"# EM partition log-likelihood (should be non-decreasing): "
          f"{np.round(hist[0],1)} -> {np.round(hist[-1],1)}  "
          f"(monotone={all(hist[k+1]>=hist[k]-1e-6 for k in range(len(hist)-1))})", flush=True)
    print(f"# alpha moved off uniform: std={alpha.std():.3f} (0 => untrained); "
          f"pi (sorted top5)={np.round(np.sort(pi)[::-1][:5],3)}", flush=True)
    # top classes per used component
    order = np.argsort(-pi)
    for h in order[:min(args.H, 8)]:
        if pi[h] < 1e-3:
            continue
        top = np.argsort(-alpha[h])[:4]
        desc = ", ".join(f"({c//K_A},{c%K_A}):{alpha[h,c]:.2f}" for c in top)
        anti = int(np.sum([(c // K_A) != (c % K_A) for c in top]))  # off-diagonal (heteroarchetype) classes
        print(f"#   comp{h} pi={pi[h]:.3f} A={alpha[h].sum():.1f}  top: {desc}  ({anti}/4 heteroarch)")

    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    np.savez(outd / "dm.npz", dm_alpha=alpha, dm_pi=pi, dm_H=args.H)
    json.dump(dict(kinds=args.kinds, split=args.split, H=args.H, n_iter=args.n_iter,
                   n_couples=len(pairs), n_singletons=len(singles), n_pairs_dropped=n_miss,
                   partition_loglik=hist, alpha_std=float(alpha.std())),
              open(outd / "fit_report.json", "w"), indent=2)
    print(f"\n# wrote {outd}/dm.npz (dm_alpha {alpha.shape}, dm_pi {pi.shape}) + fit_report.json", flush=True)


if __name__ == "__main__":
    main()
