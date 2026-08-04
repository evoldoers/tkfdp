#!/usr/bin/env python3
"""permfield mood-light model: CORPUS-SCALE (pooled-across-families) trainer.

Extends the VALIDATED single-family shared-field product-of-trees ELBO in
`experiments/permfield_elbo.py` to a whole corpus of Pfam families. The single-
family `permfield_elbo.fit()` fits ONE family; each family independently gets its
own {pi^a, alpha, s, w, rho}. Here the GLOBAL parameters
  {pi^a (archetype equilibria), alpha (Ewens concentration, FIXED prior by
   default), s, w (field exchangeabilities), rho (site-class frequencies)}
are POOLED across MANY families, while each family -- and each cluster within a
family -- keeps its OWN field posterior b_field.

  one set of global params  x  per-family / per-cluster field posteriors

A corpus sweep (structured mean-field coordinate ascent, same primitives as
permfield_elbo, none of which are modified here):
  1. From the current globals build the archetype generators Qc, eig_a and the
     field generator Qf, pif, eig_f ONCE.
  2. Zero the GLOBAL accumulators Na,Ta,roota (archetypes), Wf,Uf,rootc (field),
     gsum,nmemb (rho), obj. For each family: recompute its own Pf = {v:
     expm(Qf*tau_v)} (family-specific tau); for each cluster in the family's
     partition run column_estep -> field_potentials -> field_bp (updating that
     family/cluster's b_field); accumulate accum_arch + accum_field; add
     gamma.sum(0) to gsum, len(cluster) to nmemb, es["obj"]+logZf to obj.
  3. One GLOBAL M-step: pis=solve_arch(...), (alpha,s,w)=solve_field(...),
     rho = gsum normalised.
The corpus objective (summed ELBO over the whole corpus) is appended to a
history and MUST be monotone non-decreasing across sweeps -- the correctness
check.

This model is meant to be run WITH structure supervision. With no partition_fn it
falls back to SINGLETONS (each column its own field) and warns loudly -- the
composite baseline, which cannot discover coupling. A partition-provider hook
(`partition_fn(fam) -> list[np.ndarray]`) supplies the real contact-pair m=2
clusters (from experiments/build_pdb_partition.py); it
contact-pair clusters be plugged in later.

Reuses (imported, UNMODIFIED, from experiments.permfield_elbo):
  perm_states, ewens_p, gtr_Q, orders, field_bp, column_estep, field_potentials,
  accum_arch, solve_arch, accum_field, solve_field
and from tkfdp.permfield / .hr: build_field, eig_rev, bridge.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # never grab a shared GPU by default
import numpy as np
from scipy.linalg import expm as sexpm

sys.path.insert(0, "src")
sys.path.insert(0, os.getcwd())                             # make `experiments` importable
from tkfdp.lg08 import PI_LG08                               # noqa: E402
from tkfdp.permfield import build_field                     # noqa: E402
from tkfdp.permfield.hr import eig_rev                       # noqa: E402
from experiments.permfield_elbo import (                    # noqa: E402
    perm_states, ewens_p, gtr_Q, orders, field_bp, column_estep,
    field_potentials, accum_arch, solve_arch, accum_field, solve_field,
)
from experiments.permfield_fit import _prune_leaves, _leaf_msa  # noqa: E402

A = 20
PI0 = np.asarray(PI_LG08, float)


# ---------- corpus loading ----------
def load_families(clv_dir, n_families, max_leaves, verbose=True):
    """Load the first `n_families` families (sorted filenames) from the CLV dir,
    prune each tree to `max_leaves`, precompute traversal orders. Returns a list
    of family dicts. Families that fail to load or are degenerate are skipped."""
    paths = sorted(glob.glob(os.path.join(clv_dir, "*.npz")))
    fams = []
    for p in paths:
        if len(fams) >= n_families:
            break
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            d = np.load(p, allow_pickle=True)
            parent, tau, keep = _prune_leaves(d["parent"].astype(int),
                                              d["tau"].astype(float), max_leaves)
            msa = _leaf_msa(d, keep)
        except Exception as e:                               # skip unreadable family
            if verbose:
                print(f"  [skip {name}: {e}]", flush=True)
            continue
        N = len(parent); nl, L = msa.shape
        if nl < 2 or L < 1:                                  # degenerate; skip
            if verbose:
                print(f"  [skip {name}: nl={nl} L={L}]", flush=True)
            continue
        pre, post, root, ch = orders(parent)
        branches = [v for v in range(N) if parent[v] >= 0]
        fams.append(dict(name=name, parent=parent, tau=tau, msa=msa, N=N, nl=nl,
                         L=L, pre=pre, post=post, root=root, ch=ch,
                         branches=branches))
    return fams


def singleton_partition(fam):
    """SINGLETONS: each column its own field. The unsupervised fallback -- the
    model is meant to be run WITH a real contact-pair partition; singletons cannot
    discover coupling."""
    return [np.array([j]) for j in range(fam["L"])]


# ---------- corpus trainer ----------
def fit_corpus(fams, C, sweeps=15, alpha0=20.0, learn_alpha=False, seed=0,
               partition_fn=None, verbose=True, init_mode="perturb",
               init_noise=0.1, init_kappa=5.0, alpha_final=None):
    """Pooled-across-families shared-field ELBO. Global params {pi^a, alpha, s, w,
    rho}; per-family/per-cluster field posteriors b_fields[fam][cluster].
    partition_fn=None -> SINGLETONS with a loud warning (run WITH structure
    supervision: pass a partition_fn giving contact-pair m=2 clusters)."""
    if partition_fn is None:
        bar = "!" * 74
        print(f"\n{bar}\n"
              "!! permfield_corpus: NO structure supplied (partition_fn=None).\n"
              "!! This model is DESIGNED to run with STRUCTURE SUPERVISION --\n"
              "!! contact-pair m=2 clusters (experiments/build_pdb_partition.py).\n"
              "!! Falling back to SINGLETONS per family (composite baseline);\n"
              "!! this CANNOT discover coupling and rho will stay ~uniform.\n"
              f"{bar}\n", file=sys.stderr, flush=True)
        partition_fn = singleton_partition
    rng = np.random.default_rng(seed)
    states, dist, arch, pairs = perm_states(C)
    nS = len(states)

    # --- global params init (mirrors permfield_elbo.fit) ---
    # The archetypes sit at a near-symmetric fixed point; the init must break the
    # relabelling symmetry. 'perturb' = multiplicative-lognormal noise on LG08 pi
    # (init_noise); 'dirichlet' = independent Dirichlet(init_kappa*pi_LG08) draws
    # per archetype (much stronger separation -- smaller kappa = more distinct).
    if init_mode == "dirichlet":
        pis = rng.dirichlet(init_kappa * PI0 + 1e-3, size=C)
        pis = np.clip(pis, 1e-4, None); pis /= pis.sum(1, keepdims=True)
    else:
        pis = np.clip(PI0[None, :] * (1 + init_noise * rng.standard_normal((C, A))), 1e-4, None)
        pis /= pis.sum(1, keepdims=True)
    alpha = float(alpha0)
    s = np.ones(max(C - 1, 1)); w = np.ones(len(pairs))
    rho = np.ones(C) / C

    # --- per-family partitions + per-cluster field posteriors (init to Ewens prior) ---
    _, _, pif0, _ = build_field(C, ewens_p(C, alpha), s, w, normalize_rate=False)
    partitions = [ [np.asarray(cl, int) for cl in partition_fn(fam)] for fam in fams ]
    b_fields = [ [np.tile(pif0, (fam["N"], 1)) for _ in partitions[fi]]
                 for fi, fam in enumerate(fams) ]

    # ANNEAL the field-freezing prior: hold alpha high (field frozen at identity) so
    # classes differentiate into real clusters first, then geometrically relax alpha to
    # alpha_final so the field activates around already-good classes (avoids the random-
    # init failure where the field rate rises to permute garbage archetypes).
    alpha_sched = (np.geomspace(alpha0, alpha_final, sweeps)
                   if alpha_final is not None else None)

    hist = []
    for it in range(sweeps):
        if alpha_sched is not None:
            alpha = float(alpha_sched[it])           # scheduled field-prior relaxation
        # --- globals -> generators (built ONCE per sweep) ---
        Qc = [gtr_Q(pis[a]) for a in range(C)]
        eig_a = [eig_rev(Qc[a], pis[a]) for a in range(C)]
        _, Qf, pif, _ = build_field(C, ewens_p(C, alpha), s, w, normalize_rate=False)
        eig_f = eig_rev(Qf, np.clip(pif, 1e-10, None))

        # --- zero global accumulators ---
        Na = [np.zeros((A, A)) for _ in range(C)]; Ta = [np.zeros(A) for _ in range(C)]
        roota = [np.zeros(A) for _ in range(C)]
        Wf = np.zeros(nS); Uf = np.zeros((nS, nS)); rootc = np.zeros(nS)
        gsum = np.zeros(C); nmemb = 0; obj = 0.0

        # --- corpus E-step + accumulate ---
        for fi, fam in enumerate(fams):
            parent, tau = fam["parent"], fam["tau"]
            pre, post, root, ch = fam["pre"], fam["post"], fam["root"], fam["ch"]
            N, nl = fam["N"], fam["nl"]
            Pf = {v: np.clip(sexpm(Qf * tau[v]), 1e-300, None) for v in fam["branches"]}
            for ci, cl in enumerate(partitions[fi]):
                sub = fam["msa"][:, cl]
                es = column_estep(parent, tau, sub, C, arch, Qc, pis, rho,
                                  b_fields[fi][ci], pre, post, root, ch, nl)
                phi = field_potentials(parent, C, arch, es, pis, N, nS, ch, root)
                b_fields[fi][ci], xi, logZf = field_bp(parent, Pf, pif, phi,
                                                       pre, post, root, ch)
                obj += es["obj"] + logZf
                accum_arch(C, arch, es, tau, Qc, eig_a, parent, Na, Ta, roota)
                accum_field(C, tau, xi, b_fields[fi][ci], Qf, pif, eig_f, root,
                            Wf, Uf, rootc)
                gsum += es["gamma"].sum(0); nmemb += len(cl)
        hist.append(obj)

        # --- global M-step ---
        pis = solve_arch(C, Na, Ta, roota, pis)
        alpha, s, w = solve_field(C, Wf, Uf, rootc, alpha, s, w,
                                  learn_alpha=learn_alpha)
        rho = np.clip(gsum / max(nmemb, 1), 1e-6, None); rho /= rho.sum()

        if verbose:
            dobj = obj - hist[-2] if len(hist) > 1 else float("nan")
            print(f"  sweep {it:3d}  obj={obj:14.2f}  dobj={dobj:+11.2f}  "
                  f"rho={np.round(rho,3)}  alpha={alpha:6.3f}  "
                  f"pfield={np.round(pif,3)}", flush=True)
    return dict(pis=pis, alpha=alpha, s=s, w=w, rho=rho, pif=pif,
                b_fields=b_fields, hist=hist, partitions=partitions, fams=fams)


def _entropy(p):
    p = np.clip(p, 1e-300, None)
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--n-families", type=int, default=20)
    ap.add_argument("--max-leaves", type=int, default=24)
    ap.add_argument("--sweeps", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=20.0,
                    help="Ewens field concentration (prior, fixed unless --learn-alpha)")
    ap.add_argument("--learn-alpha", action="store_true",
                    help="fit the Ewens concentration too (weakly identified)")
    ap.add_argument("--clv-dir", default="data/pfam_processed_clv_top1000_thin128")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    C = args.C
    print(f"# permfield CORPUS fit  C={C}  n_families={args.n_families}  "
          f"max_leaves={args.max_leaves}  sweeps={args.sweeps}  "
          f"alpha0={args.alpha}  learn_alpha={args.learn_alpha}", flush=True)
    fams = load_families(args.clv_dir, args.n_families, args.max_leaves)
    ncols = sum(f["L"] for f in fams); nleaf = sum(f["nl"] for f in fams)
    print(f"# loaded {len(fams)} families: {ncols} columns, {nleaf} leaves total "
          f"({[f['name'] for f in fams][:8]}{'...' if len(fams) > 8 else ''})",
          flush=True)

    t0 = time.time()
    res = fit_corpus(fams, C, sweeps=args.sweeps, alpha0=args.alpha,
                     learn_alpha=args.learn_alpha, seed=args.seed)
    dt = time.time() - t0

    hist = np.asarray(res["hist"])
    diffs = np.diff(hist)
    # tolerance: tiny numerical dips allowed; report worst decrease
    mono = bool(np.all(diffs >= -1e-4))
    worst = float(diffs.min()) if len(diffs) else float("nan")
    print(f"\n# corpus fit {dt:.1f}s over {args.sweeps} sweeps")
    print(f"# objective monotone non-decreasing = {mono}  "
          f"(worst sweep-to-sweep delta = {worst:+.4f})")
    print(f"# final corpus obj = {hist[-1]:.2f}")
    print(f"# pooled rho    = {np.round(res['rho'], 4)}")
    print(f"# pooled alpha  = {res['alpha']:.4f}  (Ewens field concentration)")
    print(f"# pooled pfield = {np.round(res['pif'], 4)}")
    print(f"# field s = {np.round(res['s'], 4)}  w = {np.round(res['w'], 4)}")
    ent = [ _entropy(res["pis"][a]) for a in range(C) ]
    print(f"# per-archetype pi entropy (nats) = {np.round(ent, 4)}  "
          f"(uniform-20 = {np.log(20):.4f})")


if __name__ == "__main__":
    main()
