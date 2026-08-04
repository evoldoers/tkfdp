#!/usr/bin/env python3
"""Fit the RENEWAL family of 400-state pair-substitution models and score them
held-out on our coupled cherry-count corpora (same Sum_t n[:,:,t].log P(t) metric
as experiments/fit_pair_models.py, so the numbers drop into the model table).

cpt1 = an autonomous free GTR MASTER (exchangeabilities S1, stationary pi1); its
marginal is a GTR by construction, so the chain stays lumpable-to-cpt1. cpt2 is
the subservient partner: it instantly resamples from pi(x2|x1) whenever cpt1 jumps
(the RENEWAL dual-transition, rate S1[x1,y1]*pi_joint(y1,x2')), and between jumps
it evolves by its own single-component transitions. The three variants differ ONLY
in how free those cpt2 single transitions are (all reversible w.r.t. the asymmetric
pi_joint = pi1(x1) pi(x2|x1)):

  variant               cpt2 single transitions                       new  total
  --------------------  --------------------------------------------  ---  -----
  dependent             FREE reversible flux per field x1 (E[x1])     3800  4389
  renewal_gtr         one GTR, own S2 != S1 (stationary pi(.|x1))    190   779
  renewal_gtr_same    GTR reusing cpt1's S1                            0   589

Nesting dependent > renewal_gtr > renewal_gtr_same. cpt2's free flux is the
lever to gain on Lumpable; renewal_gtr_same is the over-constrained baseline we
keep to show that over-constraining has consequences.

Fit: ML over the params by Adam (early-stop) on the training shards; report
train/val per-count transition LL. cpt1 = lower-index contact column; counts are
component-swap-symmetrized by default (the field label is arbitrary). The proper
fast fit is EM (HR E-step + structured M-step) -- worth writing if refit often.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from fit_pair_models import load_parts, n_parts, empirical_pi         # noqa: E402
from tkfdp.lg08 import Q_LG08 as _QLG, PI_LG08 as _PILG               # noqa: E402

NA, NS = 20, 400
_IU = np.triu_indices(NA, 1)                                          # 190 upper-tri
_AR = jnp.arange(NA)
VARIANTS = ("renewal", "renewal_gtr", "renewal_gtr_same")
NPARAMS = {"renewal": 190 + 3800 + 399, "renewal_gtr": 190 + 190 + 399,
           "renewal_gtr_same": 190 + 399}


def _sym(vec):
    S = jnp.zeros((NA, NA)).at[_IU].set(vec)
    return S + S.T


def build_Q(variant, params):
    """(variant, params dict) -> Q (400,400), reversible w.r.t. pi_joint."""
    S1 = _sym(jnp.exp(params["logS1"]))
    pj = jax.nn.softmax(params["logpi"]).reshape(NA, NA)
    pi1 = pj.sum(1)
    cond = pj / pi1[:, None]                                          # pi(x2|x1)
    # dual (renewal): Qf[x1,x2,y1,y2] = S1[x1,y1]*pj[y1,y2]; broadcast over x2; y1!=x1
    A = jnp.einsum("xy,yz->xyz", S1, pj)                              # (x1,y1,y2)
    Qf = jnp.broadcast_to(A[:, None, :, :], (NA, NA, NA, NA))
    Qf = Qf * (1.0 - jnp.eye(NA)[:, None, :, None])
    # cpt2 alone: B[x1,x2,y2] = Ecpt2[x1][x2,y2]*cond[x1,y2]; y2!=x2; at (x1,x2,x1,y2)
    if variant == "renewal":
        E = jax.vmap(_sym)(jnp.exp(params["logE"]))                  # (20,20,20) per-field
        B = jnp.einsum("aby,ay->aby", E, cond)
    else:
        S2 = S1 if variant == "renewal_gtr_same" else _sym(jnp.exp(params["logS2"]))
        B = jnp.einsum("by,ay->aby", S2, cond)
    B = B * (1.0 - jnp.eye(NA)[None, :, :])
    Qc = jnp.zeros((NA, NA, NA, NA)).at[
        _AR[:, None, None], _AR[None, :, None], _AR[:, None, None], _AR[None, None, :]].set(B)
    Q = (Qf + Qc).reshape(NS, NS)
    return Q - jnp.diag(Q.sum(1))


def logP_all(Q, pjoint_flat, tau):
    d = jnp.sqrt(jnp.clip(pjoint_flat, 1e-30, None))
    Qs = d[:, None] * Q * (1.0 / d)[None, :]
    Qs = 0.5 * (Qs + Qs.T)
    lam, V = jnp.linalg.eigh(Qs)
    E = jnp.exp(lam[None, :] * tau[:, None])
    VEV = jnp.einsum("ak,tk,bk->tab", V, E, V)
    P = (1.0 / d)[None, :, None] * VEV * d[None, None, :]
    return jnp.log(jnp.clip(P, 1e-300, None))


def make_neg_ll(variant):
    def neg_ll(params, n_by_t, tau):
        pj = jax.nn.softmax(params["logpi"]).reshape(NS)
        Q = build_Q(variant, params)
        return -jnp.sum(n_by_t * logP_all(Q, pj, tau))
    return neg_ll


def init_params(variant, pj0):
    S0 = np.asarray(_QLG, float) / np.asarray(_PILG, float)[None, :]
    S0 = 0.5 * (S0 + S0.T)
    logS0 = np.log(np.clip(S0[_IU], 1e-6, None))
    p = {"logS1": jnp.asarray(logS0),
         "logpi": jnp.asarray(np.log(np.clip(pj0.reshape(NS), 1e-9, None)))}
    if variant == "renewal_gtr":
        p["logS2"] = jnp.asarray(logS0)
    elif variant == "renewal":
        p["logE"] = jnp.asarray(np.tile(logS0, (NA, 1)))             # (20,190) per-field
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--corpora", default="trrosetta,trrosetta_rate,af_full")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="results/pair_models/renewal_eval.json")
    ap.add_argument("--no-symmetrize", action="store_true")
    args = ap.parse_args()
    sw = np.arange(NS).reshape(NA, NA).T.reshape(NS)
    import optax

    out = {}
    for variant in [v.strip() for v in args.variants.split(",")]:
        out[variant] = {}
        for name in [c.strip() for c in args.corpora.split(",")]:
            path = f"data/cherry_counts_{name}"
            npar = n_parts(path)
            if npar < 2:
                continue
            val = npar - 1
            npair, tau, _ = load_parts(path, [i for i in range(npar) if i != val])
            vpair, _, _ = load_parts(path, [val])
            if not args.no_symmetrize:
                npair = npair + npair[sw][:, sw]
                vpair = vpair + vpair[sw][:, sw]
            pj0 = empirical_pi(npair).reshape(NA, NA)
            n_by_t = jnp.asarray(np.moveaxis(npair, 2, 0))
            v_by_t = jnp.asarray(np.moveaxis(vpair, 2, 0))
            tau_j = jnp.asarray(tau)
            params = init_params(variant, pj0)
            neg_ll = make_neg_ll(variant)
            sched = optax.cosine_decay_schedule(args.lr, args.steps, alpha=0.02)
            opt = optax.adam(sched); st = opt.init(params)
            vg = jax.jit(jax.value_and_grad(neg_ll))
            tot = float(npair.sum()); prev = None
            for step in range(args.steps):
                loss, g = vg(params, n_by_t, tau_j)
                upd, st = opt.update(g, st); params = optax.apply_updates(params, upd)
                if step % 100 == 0 or step == args.steps - 1:
                    pc = -float(loss) / tot
                    print(f"  [{variant}/{name}] step {step:4d} train={pc:.5f}", flush=True)
                    if prev is not None and abs(pc - prev) < 1e-7:
                        print(f"  [{variant}/{name}] converged at {step}", flush=True); break
                    prev = pc
            tr = -float(neg_ll(params, n_by_t, tau_j)) / tot
            va = -float(neg_ll(params, v_by_t, tau_j)) / float(vpair.sum())
            pj = np.asarray(jax.nn.softmax(params["logpi"])).reshape(NA, NA)
            asym = float(np.abs(pj - pj.T).sum() / pj.sum())
            out[variant][name] = dict(train=tr, val=va, n_params=NPARAMS[variant],
                                      pi_asymmetry=asym)
            print(f"# {variant:20s} {name:15s} train={tr:.4f} val={va:.4f} "
                  f"({NPARAMS[variant]} params; pi asym={asym:.2e})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()
