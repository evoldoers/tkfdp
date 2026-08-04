"""Discriminative (classifier-loss) DM training from the pairing cache.

Instead of the generative MLE (fit_dm_from_cache.py), train the DM mixture to
SEPARATE true PDB pairs from spurious ones. Each candidate (i,j) gets a pairing
logit read off the cache,

    logit(i,j) = [ logP_pair(i,j) - logP_sing(i) - logP_sing(j) ] + b,

where each term marginalizes the DM mixture over classes (the pair term couples
c_i,c_j through the shared component -- the ONLY channel through which the DM
moves the logit, since a product-form DM cancels). p_pair = sigmoid(logit).

  positives  = PDB contact pairs  (target p_pair -> 1)
  negatives  = equal # of shortlisted pairs whose BOTH columns are true
               singletons (non-contact) -- the hard spurious pairs discovery
               must reject  (target p_pair -> 0)
  loss = -sum_pos log sigmoid(logit) - sum_neg log sigmoid(-logit)   (balanced)

Minimized over alpha (H,K_c), pi, bias b by Adam autodiff. Field rate untouched
(only cached LL_pair / sing_ll are read), so the cache stays valid. Saves a
dm.npz consumable by run_collapsed_discovery / dm_charge_flip_test.

  python experiments/train_dm_classifier.py --H 10 --steps 800 --out results/dm_clf_pdb_train
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")            # CPU; don't touch the GPUs
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import jax, jax.numpy as jnp
from jax.scipy.special import logsumexp
jax.config.update("jax_enable_x64", True)

from tkfdp.coupling.dynfield.phylo_elbo.pairing_cache import PairingCache
from fit_pdb_hyperparams import load_pairs
from tkfdp.bio import load_split

CACHE = "data/pairing_cache"


def gather_examples(kinds, split, seed=0, neg_ratio=1):
    """Positives (PDB pairs) + negatives (shortlisted pairs, both cols non-contact),
    each as (LL_pair, top_i, top_j, sing_i, sing_j). #neg = neg_ratio * #pos."""
    pc = PairingCache(CACHE)
    byfam = load_pairs(kinds, split=split)
    keep = set(load_split()[split])
    pos, neg = [], []
    contact_cols = {}
    for fam, plist in byfam.items():
        cc = set()
        for (i, j) in plist:
            cc.add(int(i)); cc.add(int(j))
        contact_cols[fam] = cc
    contact_pairs = {(f, min(i, j), max(i, j)) for f, ps in byfam.items() for (i, j) in ps}

    def rec(fam, i, j):
        r = pc.get(fam, i, j)
        si = pc.sing_ll(fam, i); sj = pc.sing_ll(fam, j)
        if r is None or si is None or sj is None:
            return None
        return (np.asarray(r["LL_pair"], float), r["top_i"].astype(int),
                r["top_j"].astype(int), np.asarray(si, float), np.asarray(sj, float))

    for fam, plist in byfam.items():
        for (i, j) in plist:
            e = rec(fam, i, j)
            if e is not None:
                pos.append(e)
    # negatives: shortlisted pairs, both columns true singletons (not contact cols)
    for fam in byfam:
        f = pc._family(fam)
        if f is None:
            continue
        _, idx, _ = f
        cc = contact_cols.get(fam, set())
        for (a, b) in idx:
            key = (fam, min(a, b), max(a, b))
            if a in cc or b in cc or key in contact_pairs:
                continue
            e = rec(fam, a, b)
            if e is not None:
                neg.append((fam, a, b, e))
    rng = np.random.default_rng(seed)
    rng.shuffle(neg)
    n_keep = min(len(neg), neg_ratio * len(pos))
    neg = [e for (_, _, _, e) in neg[:n_keep]]                # neg_ratio * #pos
    return pos, neg


def stack(examples, N, K_c):
    LL = np.zeros((len(examples), N, N)); TI = np.zeros((len(examples), N), int)
    TJ = np.zeros((len(examples), N), int); SI = np.zeros((len(examples), K_c))
    SJ = np.zeros((len(examples), K_c))
    for e, (ll, ti, tj, si, sj) in enumerate(examples):
        LL[e] = ll; TI[e] = ti; TJ[e] = tj; SI[e] = si; SJ[e] = sj
    return (jnp.asarray(LL), jnp.asarray(TI), jnp.asarray(TJ),
            jnp.asarray(SI), jnp.asarray(SJ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="saltbridge")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--neg-ratio", type=int, default=1, help="#negatives = neg_ratio * #positives")
    ap.add_argument("--l2", type=float, default=1.0, help="L2 on log_alpha (toward alpha=1); 0 overfits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/dm_clf_pdb_train")
    args = ap.parse_args()

    pos, neg = gather_examples(set(args.kinds.split(",")), args.split, args.seed, args.neg_ratio)
    N = pos[0][0].shape[0]; K_c = pos[0][3].shape[0]
    print(f"# classifier DM: {len(pos)} positive (PDB) + {len(neg)} negative (spurious singleton) "
          f"pairs  topN={N} K_c={K_c} H={args.H}  split={args.split}", flush=True)
    Lp, Tip, Tjp, Sip, Sjp = stack(pos, N, K_c)
    Ln, Tin, Tjn, Sin, Sjn = stack(neg, N, K_c)

    rng = np.random.default_rng(args.seed)
    log_alpha0 = jnp.asarray(0.6 * rng.standard_normal((args.H, K_c)))   # alpha0=1, symmetry-broken
    params = dict(log_alpha=log_alpha0, raw_pi=jnp.zeros(args.H), b=jnp.asarray(0.0))

    def logits(params, LL, TI, TJ, SI, SJ):
        alpha = jnp.exp(params["log_alpha"])                 # (H,K_c)
        pi = jax.nn.softmax(params["raw_pi"])                # (H,)
        A = alpha.sum(1); denom = A * (A + 1.0)              # (H,)
        logsp = jnp.log((pi[:, None] * alpha / A[:, None]).sum(0) + 1e-300)   # (K_c,)

        def one(ll, ti, tj, si, sj):
            ai = alpha[:, ti]; aj = alpha[:, tj]             # (H,N)
            grid = jnp.einsum('h,ha,hb->ab', pi / denom, ai, aj)  # (N,N)
            lPp = logsumexp(ll + jnp.log(grid + 1e-300))
            lPi = logsumexp(si + logsp); lPj = logsumexp(sj + logsp)
            return lPp - lPi - lPj + params["b"]
        return jax.vmap(one)(LL, TI, TJ, SI, SJ)

    n_tot = len(pos) + len(neg)

    def loss(params):
        lp = logits(params, Lp, Tip, Tjp, Sip, Sjp)
        ln = logits(params, Ln, Tin, Tjn, Sin, Sjn)
        # product of ALL posteriors -> SUM of nll over every example (so the
        # neg_ratio imbalance is preserved, not rebalanced); /n_tot only scales lr.
        # -log sigmoid(x) = softplus(-x) ; -log sigmoid(-x) = softplus(x)
        data = (jnp.sum(jax.nn.softplus(-lp)) + jnp.sum(jax.nn.softplus(ln))) / n_tot
        # L2 on log_alpha (toward alpha=1) -- without it the classifier drives alpha
        # to degenerate extremes (near-deterministic class selection) and overfits.
        reg = args.l2 * jnp.mean(params["log_alpha"] ** 2)
        return data + reg

    # Adam
    lr = args.lr; b1, b2, eps = 0.9, 0.999, 1e-8
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}
    val_and_grad = jax.jit(jax.value_and_grad(loss))

    def auc(params):
        lp = np.asarray(logits(params, Lp, Tip, Tjp, Sip, Sjp))
        ln = np.asarray(logits(params, Ln, Tin, Tjn, Sin, Sjn))
        # Mann-Whitney AUC = P(pos > neg)
        from scipy.stats import mannwhitneyu
        U, _ = mannwhitneyu(lp, ln, alternative="greater")
        return U / (len(lp) * len(ln)), float(lp.mean()), float(ln.mean())

    a0, mp0, mn0 = auc(params)
    print(f"# init  loss={float(loss(params)):.4f}  AUC={a0:.3f}  mean_logit pos={mp0:+.2f} neg={mn0:+.2f}", flush=True)
    for t in range(1, args.steps + 1):
        L, g = val_and_grad(params)
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)
            params[k] = params[k] - lr * mh / (jnp.sqrt(vh) + eps)
        if t % 100 == 0 or t == args.steps:
            a, mp, mn = auc(params)
            print(f"#  step {t:4d}  loss={float(L):.4f}  AUC={a:.3f}  mean_logit pos={mp:+.2f} neg={mn:+.2f}", flush=True)

    alpha = np.asarray(jnp.exp(params["log_alpha"]))
    pi = np.asarray(jax.nn.softmax(params["raw_pi"]))
    b = float(params["b"])
    a_fin, _, _ = auc(params)
    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    np.savez(outd / "dm.npz", dm_alpha=alpha, dm_pi=pi, dm_H=args.H, bias=b)
    json.dump(dict(kinds=args.kinds, split=args.split, H=args.H, steps=args.steps,
                   n_pos=len(pos), n_neg=len(neg), final_auc=float(a_fin), bias=b,
                   alpha_std=float(alpha.std())), open(outd / "clf_report.json", "w"), indent=2)
    print(f"\n# final AUC={a_fin:.3f}  bias={b:+.2f}  alpha_std={alpha.std():.2f}  pi_top={np.round(np.sort(pi)[::-1][:5],3)}")
    print(f"# wrote {outd}/dm.npz + clf_report.json", flush=True)


if __name__ == "__main__":
    main()
