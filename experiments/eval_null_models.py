#!/usr/bin/env python3
r"""Independent-sites NULL baselines for the paper-2b pair-model table, scored on the same
held-out family shard and per-observation normalisation as the coupled models
(fit_pair_models.loglik): mean per-pair transition log-lik  sum n log P_pair / sum n, with
P_pair((a,b)->(a',b')) = P1(a->a') P1(b->b') for a single-site process P1.  Two nulls:
  * LG08 (x) LG08 : both sites evolve under a fixed LG08 GTR (one fitted global rate scale).
  * Lumpable-marginal (x) : both sites under the SINGLE-SITE MARGINAL of the fitted Lumpable
    pair model (its lumped 20-state chain) -- the coupling-isolating null of the responsibility
    mixture.  P1 = the pi(j|i)-averaged marginal of exp(Q_Lump t) (exact for a lumpable chain).
"""
import sys, glob, argparse
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "experiments")
from fit_pair_models import load_parts                          # noqa: E402
from composite_potts_phylo_elbo import S_LG08, PI_LG08          # noqa: E402

A = 20


def gtr_gen(S, pi):
    Q = np.asarray(S, float)[:A, :A] * np.asarray(pi, float)[None, :]
    np.fill_diagonal(Q, 0.0); Q[np.diag_indices(A)] = -Q.sum(1)
    rate = -(np.asarray(pi) * np.diag(Q)).sum()
    return Q / max(rate, 1e-30)


def val_shard(corpus):
    nparts = len(glob.glob(f"{corpus}/part_*.npz")); val = nparts - 1
    vpair, tau, _ = load_parts(corpus, [val])
    return vpair, tau, val


def null_score(P1_list, vpair, tau):
    """per-observation log-lik of the independent-sites null with per-bin marginal
    transition matrices P1_list[t]; identical normalisation to fit_pair_models.loglik."""
    ll = 0.0; tot = 0.0
    for t in range(len(tau)):
        n = vpair[:, :, t].reshape(A, A, A, A)                  # n[a,b,a',b']
        M1 = n.sum((1, 3)); M2 = n.sum((0, 2))                  # marginal transition counts
        logP = np.log(np.clip(P1_list[t], 1e-300, None))
        ll += (M1 * logP).sum() + (M2 * logP).sum(); tot += n.sum()
    return ll / tot


def lg08_null(vpair, tau, fit_scale=True):
    Q1 = gtr_gen(S_LG08, PI_LG08)
    def sc(c):
        return null_score([expm(c * Q1 * tau[t]) for t in range(len(tau))], vpair, tau)
    if not fit_scale:
        return sc(1.0), 1.0
    best = max(((sc(c), c) for c in np.geomspace(0.25, 4.0, 20)), key=lambda z: z[0])
    c0 = best[1]
    best = max([best] + [(sc(c), c) for c in np.linspace(c0 * 0.7, c0 * 1.4, 15)],
               key=lambda z: z[0])
    return best


def lumpable_marginal_null(params_path, vpair, tau):
    d = np.load(params_path)
    Q = np.asarray(d["lumpable__Q"], float); pi = np.asarray(d["lumpable__pi"], float)
    pir = pi.reshape(A, A); picond = pir / pir.sum(1, keepdims=True)      # pi(j|i)
    P1 = []
    for t in range(len(tau)):
        PL = expm(Q * tau[t]).reshape(A, A, A, A)                # P_Lump[(i,j),(k,l)]
        P1.append(np.einsum("ij,ijkl->ik", picond, PL))          # marginal transition, rows sum 1
    return null_score(P1, vpair, tau)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", default=[
        "data/cherry_counts_trrosetta:results/pair_models/lumpable_trrosetta_params.npz",
        "data/cherry_counts_af_full:results/pair_models/lumpable_af_full_params.npz",
    ], help="corpus:lumpable_params pairs")
    args = ap.parse_args()
    for job in args.jobs:
        corpus, lump = job.split(":")
        vpair, tau, val = val_shard(corpus)
        s_lg, c = lg08_null(vpair, tau)
        s_lm = lumpable_marginal_null(lump, vpair, tau)
        print(f"{corpus}  val=part_{val}  npairs={vpair.sum():.3e}", flush=True)
        print(f"    LG08(x)LG08 null        = {s_lg:.4f}   (global rate scale c={c:.3f})")
        print(f"    Lumpable-marginal null  = {s_lm:.4f}", flush=True)
