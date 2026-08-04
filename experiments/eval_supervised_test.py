"""Held-out PDB-contact-prediction eval for the supervised enum400 fit.

Fit the model on --split train (train_supervised_enum400.py -> _params.npz), then
run THIS on --split test (or val): rebuild the model on the held-out, clan-disjoint
families with the LEARNED pi_archetype + field-rate, and score the true PDB
contacts / confirmed flips against a background of random column pairs. No family
in test shares a clan with train, so this is genuine generalization, not recall of
the fit set.

Metrics per params set (baseline LG-C20 + flip-averse prior, vs the trained
params):
  * contact-vs-background pairing log-odds separation (AUC / Mann-Whitney),
  * confirmed-flip pairing log-odds + flip-phi,
so we can see whether training away the flip-averse field-rate prior makes
held-out contacts/flips read as favorable.

  python experiments/eval_supervised_test.py --split test \
      --params results/<run>/_params.npz --kinds saltbridge --n-bg 20
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, "experiments")
import precompute_pairing as PP
from tkfdp.coupling.dynfield.phylo_elbo import supervised_trainer as ST
from tkfdp.coupling.dynfield.phylo_elbo import field_rate_discovery as fd

PDIR = Path("data/pdb_partition_clv_top1000_sifts")
CLV = Path("data/pfam_processed_clv_top1000_thin128")


def _lse(v, axis=None):
    v = np.asarray(v, float); m = v.max(axis=axis, keepdims=True)
    return np.squeeze(np.log(np.sum(np.exp(v - m), axis=axis, keepdims=True)) + m, axis=axis)


def load_split_pairs(kinds, split):
    """{fam: {'contacts':[(i,j)], 'flips':set((i,j))}} on the given split."""
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
        sel = [(int(a), int(b)) for (a, b), k in zip(d["pairs"], d["kind"]) if str(k) in kinds]
        if sel:
            out.setdefault(fam, {"contacts": [], "flips": set()})["contacts"] = sel
    for r in json.load(open(PDIR / "confirmed_flips.json")):
        f = r["family"]
        if f in keep and (CLV / f"{f}.npz").exists():
            out.setdefault(f, {"contacts": [], "flips": set()})
            out[f]["flips"].add((min(int(r["i"]), int(r["j"])), max(int(r["i"]), int(r["j"]))))
    return out


def apply_params(ds, params_path, rates, w_prior):
    """Load _params.npz and install pi_archetype + field-rate; return field weights.
    None -> baseline (leave ds at LG-C20 + prior)."""
    if params_path is None:
        return np.asarray(w_prior, float)
    d = np.load(params_path)
    st = ds.state
    st.pi_archetype = np.asarray(d["pi_archetype"], float)
    if "rho_chain" in d.files:
        st.rho_chain = float(d["rho_chain"])
    st.refresh_pi_field()
    ST.rebuild_rate_kernels(ds)
    ds.P_sub_aug = fd.build_P_sub_aug(st._pi_field, st.S, st.bin_centers, ds.rates_sub)
    import jax.numpy as jnp
    ds.P_sub_aug_j = jnp.asarray(ds.P_sub_aug)
    ds.pi_field_aug = fd.build_pi_field_aug(st._pi_field, ds.G_s)
    for key in ("field_rate_weights", "w"):
        if key in d.files:
            return np.asarray(d[key], float)
    return np.asarray(w_prior, float)


def pair_logodds(ds, byfam_pairs, w, alpha_z=100.0, topN=8):
    """Class-marginalized pairing log-odds for each pair, under current ds params
    and field-rate weights w. Uses the trainer's fast per-bin scorer + a static
    singleton marginal. Returns {(fam,i,j): logodds}."""
    st = ds.state; logw = np.log(w + 1e-300)
    recs, G, Kc, tops = ST.score_perbin_fast(ds, byfam_pairs, topN=topN)
    # singleton per-bin marginals for the columns via the full top-N singleton scan
    fam_ids = [f.family_id for f in st.families]
    mb1 = st.m_bucket_for(1)
    s_specs, s_slot = [], {}
    for (fam, col), tc in tops.items():
        fi = fam_ids.index(fam)
        for c in tc:
            s_slot[(fam, col, int(c))] = len(s_specs)
            cl = np.zeros(mb1, np.int32); cl[0] = int(c)
            s_specs.append((fi, np.array([col], np.int32), cl, mb1))
    ll_s = fd._score_specs_rates(ds, s_specs)                      # (n,G)
    sing_marg = {}
    for (fam, col), tc in tops.items():
        rows = np.stack([ll_s[s_slot[(fam, col, int(c))]] for c in tc])  # (topN,G)
        sing_marg[(fam, col)] = _lse(rows + logw[None, :])               # scalar
    out = {}
    for r in recs:
        pm = _lse(r["grid"] + logw[None, None, :])                # pair class+rate marg
        out[(r["fam"], r["i"], r["j"])] = pm - sing_marg[(r["fam"], r["i"])] \
            - sing_marg[(r["fam"], r["j"])] - np.log(alpha_z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--params", default=None, help="_params.npz from a train fit; omit for LG-C20 baseline")
    ap.add_argument("--kinds", default="saltbridge")
    ap.add_argument("--n-bg", type=int, default=15, help="background random pairs per family")
    ap.add_argument("--topN", type=int, default=8)
    ap.add_argument("--max-families", type=int, default=None)
    args = ap.parse_args()

    data = load_split_pairs(set(args.kinds.split(",")), args.split)
    fams = list(data.keys())[:args.max_families] if args.max_families else list(data.keys())
    print(f"# {args.split}: {len(fams)} families  "
          f"({sum(len(data[f]['contacts']) for f in fams)} contacts, "
          f"{sum(len(data[f]['flips']) for f in fams)} flips)", flush=True)
    ds, rates, w_prior = PP.build_enum400_ds(fams)
    w = apply_params(ds, args.params, rates, w_prior)
    print(f"# params: {'trained '+args.params if args.params else 'BASELINE LG-C20 + prior'}  "
          f"w_inv={w[0]:.3f}  rho_chain={ds.state.rho_chain:.3f}", flush=True)

    rng = np.random.default_rng(0)
    byfam = {}; label = {}                          # pair -> 'contact'/'flip'/'bg'
    for f in fams:
        L = ds.state.families[fam_i(ds, f)].L
        contacts = data[f]["contacts"]; flips = data[f]["flips"]
        pset = set(contacts) | {(min(a, b), max(a, b)) for (a, b) in flips}
        bg = []
        for _ in range(args.n_bg * 4):
            i, j = sorted(rng.integers(0, L, 2))
            if i != j and (i, j) not in pset:
                bg.append((i, j))
            if len(bg) >= args.n_bg:
                break
        allp = list(pset) + bg
        if allp:
            byfam[f] = allp
            for (a, b) in allp:
                key = (min(a, b), max(a, b))
                label[(f, min(a, b), max(a, b))] = (
                    "flip" if key in flips else "contact" if key in [(min(x, y), max(x, y)) for (x, y) in contacts] else "bg")

    lo = pair_logodds(ds, byfam, w, topN=args.topN)
    import collections
    by = collections.defaultdict(list)
    for k, v in lo.items():
        by[label[k]].append(v)
    from scipy.stats import mannwhitneyu
    for grp in ("flip", "contact", "bg"):
        a = np.array(by.get(grp, []))
        if len(a):
            print(f"#   {grp:8s} n={len(a):4d}  mean_logodds={a.mean():+.2f}  frac>0={np.mean(a>0):.2f}")
    if by.get("contact") and by.get("bg"):
        U, p = mannwhitneyu(by["contact"], by["bg"], alternative="greater")
        auc = U / (len(by["contact"]) * len(by["bg"]))
        print(f"#   contact-vs-bg AUC={auc:.3f}  (MWU p={p:.2g})")
    if by.get("flip") and by.get("bg"):
        U, p = mannwhitneyu(by["flip"], by["bg"], alternative="greater")
        print(f"#   flip-vs-bg    AUC={U/(len(by['flip'])*len(by['bg'])):.3f}  (p={p:.2g})")


def fam_i(ds, fam):
    return [f.family_id for f in ds.state.families].index(fam)


if __name__ == "__main__":
    main()
