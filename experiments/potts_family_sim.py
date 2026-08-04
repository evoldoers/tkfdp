#!/usr/bin/env python3
r"""Simulate an MSA from an ALL-INTERACTING Potts model down a real family's phylogeny,
via a sequence-space continuous-time Markov chain (single-residue Metropolis moves whose
acceptance uses the FULL Potts neighbourhood).  This is the generative process the
composite pairwise phylo-ELBO approximates; the whole point is that all residues interact,
so a pair's marginal correlation mixes DIRECT coupling with INDIRECT network paths.

Potts model: fields h (L,A), couplings J (L,L,A,A) with J[i,j]=J[j,i].T, J[i,i]=0.
  energy(x) = sum_i h[i,x_i] + sum_{i<j} J[i,j,x_i,x_j],   P(x) ∝ exp(energy).
Generator (shared LG08 exchangeability S, Metropolis 'min' acceptance, lifted to the full
sequence): rate(i: x_i=a -> a') = S[a,a'] * min(1, exp(dE)),  a'!=a,
  dE = F[i,a'] - F[i,a],   F[i,b] = h[i,b] + sum_{j!=i} J[i,j,b,x_j]   (local field at i).
Reversible w.r.t. P.  A single flat overall rate_scale sets the substitution timescale so
branch lengths tau (LG08 expected-substitutions) are ~comparable.

Simulation: root drawn from the Potts stationary by Gibbs; each branch evolved by exact
Gillespie for its length tau.
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_potts_phylo_elbo import S_LG08

A = 20
THIN_DIR = "data/pfam_processed_clv_top1000_thin128"
PART_DIR = "data/pdb_partition_clv_top1000_sifts"


# ---------------------------------------------------------------------------
# family loader
# ---------------------------------------------------------------------------
def load_family(fam):
    d = np.load(f"{THIN_DIR}/{fam}.npz", allow_pickle=True)
    parent = np.asarray(d["parent"], int); tau = np.asarray(d["tau"], float)
    msa = np.asarray(d["leaf_msa"], np.int64)            # (n_leaves, L)
    L = int(d["L"]); nl = int(d["n_leaves"])
    p = np.load(f"{PART_DIR}/{fam}.npz", allow_pickle=True)
    pairs = np.asarray(p["pairs"], int)                  # (npair, 2) contact columns
    kind = np.asarray(p["kind"])
    root = int(np.where(parent < 0)[0][0])
    children = [[] for _ in range(len(parent))]
    for v in range(len(parent)):
        if parent[v] >= 0:
            children[parent[v]].append(v)
    leaves = [v for v in range(len(parent)) if not children[v]]
    contacts = {frozenset((int(i), int(j))) for i, j in pairs}
    return dict(fam=fam, parent=parent, tau=tau, root=root, children=children,
                leaves=leaves, msa=msa, L=L, nl=nl, contacts=contacts,
                pairs=pairs, kind=kind)


# ---------------------------------------------------------------------------
# Potts local fields / rates
# ---------------------------------------------------------------------------
def local_fields(seq, J, h):
    """F[i,b] = h[i,b] + sum_{j!=i} J[i,j,b,seq_j].   (L,A)."""
    L = len(seq)
    idx = np.broadcast_to(seq[None, :, None, None], (L, L, A, 1))
    Jg = np.take_along_axis(J, idx, axis=3)[..., 0]      # (L,L,A) : J[i,j,b,seq_j]
    return h + Jg.sum(1)                                 # (L,A)  (J[i,i]=0)

def move_rates(seq, F, S):
    """rate[i,a'] = S[seq_i,a'] * min(1, exp(F[i,a']-F[i,seq_i])), 0 on the diagonal."""
    L = len(seq)
    Fcur = F[np.arange(L), seq]                          # (L,)
    acc = np.minimum(1.0, np.exp(np.clip(F - Fcur[:, None], -50, 50)))
    R = S[seq] * acc                                     # (L,A)
    R[np.arange(L), seq] = 0.0
    return R


def gibbs_potts(J, h, n_sweeps, seed, x0=None):
    """Sample one sequence from the Potts stationary by Gibbs (site conditionals)."""
    L = h.shape[0]; rng = np.random.default_rng(seed)
    x = rng.integers(0, A, L) if x0 is None else x0.copy()
    for _ in range(n_sweeps):
        for i in rng.permutation(L):
            # conditional ∝ exp(F[i,b]) with F for THIS site given current others
            idx = np.broadcast_to(x[None, :, None], (1, L, 1))  # tiny gather for site i
            f = h[i] + J[i, :, :, :][np.arange(L), :, x].sum(0)  # sum_j J[i,j,:,x_j]
            f = f - f.max(); p = np.exp(f); p /= p.sum()
            x[i] = rng.choice(A, p=p)
    return x


def estimate_rate_scale(J, h, S, n=6, sweeps=60, seed=0):
    """Flat scale so the equilibrium expected PER-SITE substitution rate ~ 1, matching the
    tree's tau (LG08 per-site expected substitutions).  total equilibrium rate -> L."""
    L = h.shape[0]; tot = []
    for s in range(n):
        x = gibbs_potts(J, h, sweeps, seed + s)
        tot.append(move_rates(x, local_fields(x, J, h), S).sum())
    return L / max(float(np.mean(tot)), 1e-9)


def evolve_branch(seq, tau, J, h, S, rate_scale, rng, max_events=100000):
    seq = seq.copy(); t = 0.0; L = len(seq); n = 0
    while n < max_events:
        F = local_fields(seq, J, h)
        R = move_rates(seq, F, S) * rate_scale
        total = R.sum()
        if total <= 0:
            break
        dt = rng.exponential(1.0 / total)
        if t + dt > tau:
            break
        t += dt; n += 1
        flat = R.ravel(); k = rng.choice(L * A, p=flat / total)
        i, a = divmod(k, A); seq[i] = a
    return seq


def simulate_family(fam_d, J, h, S, seed, root_sweeps=200, rate_scale=None):
    """Evolve one root->tree; return leaf sequences (n_leaves, L) in fam_d['leaves'] order."""
    parent = fam_d["parent"]; tau = fam_d["tau"]; children = fam_d["children"]
    root = fam_d["root"]; L = h.shape[0]
    if rate_scale is None:
        rate_scale = estimate_rate_scale(J, h, S, seed=seed)
    rng = np.random.default_rng(seed)
    state = [None] * len(parent)
    state[root] = gibbs_potts(J, h, root_sweeps, seed + 12345)
    stack = [root]
    while stack:
        v = stack.pop()
        for c in children[v]:
            state[c] = evolve_branch(state[v], float(tau[c]), J, h, S, rate_scale, rng)
            stack.append(c)
    leaf_seqs = np.stack([state[v] for v in fam_d["leaves"]])
    return leaf_seqs, rate_scale


# ---------------------------------------------------------------------------
# validation: CTMC equilibrium must match the Gibbs stationary
# ---------------------------------------------------------------------------
def _check(seed=0):
    rng = np.random.default_rng(seed)
    L = 12
    S = np.asarray(S_LG08, float)[:A, :A].copy(); np.fill_diagonal(S, 0); S = 0.5 * (S + S.T)
    h = 0.5 * rng.standard_normal((L, A))
    J = np.zeros((L, L, A, A))
    for _ in range(L):                                   # sparse random contacts
        i, j = rng.choice(L, 2, replace=False)
        B = 0.6 * rng.standard_normal((A, A)); B = 0.5 * (B + B.T)
        J[i, j] = B; J[j, i] = B.T
    # (1) fields-only Gibbs marginal matches softmax(h)
    Jz = np.zeros_like(J)
    xs = np.stack([gibbs_potts(Jz, h, 40, seed + k) for k in range(400)])
    emp = np.stack([np.bincount(xs[:, i], minlength=A) / len(xs) for i in range(L)])
    tru = np.exp(h - h.max(1, keepdims=True)); tru /= tru.sum(1, keepdims=True)
    print(f"# fields-only Gibbs marginal L1 vs softmax(h): {np.abs(emp-tru).sum(1).mean():.3f}")
    # (2) CTMC on a long branch converges to the Gibbs (coupled) stationary.
    #     Gate against the Gibbs-vs-Gibbs sampling-noise FLOOR (two finite estimates
    #     of a 20-cell / 400-cell distribution differ by noise even when both are exact).
    M = 1500
    rate_scale = estimate_rate_scale(J, h, S, seed=seed)
    def sitemarg(X): return np.stack([np.bincount(X[:, i], minlength=A) / len(X) for i in range(L)])
    gibA = np.stack([gibbs_potts(J, h, 60, seed + 100 + k) for k in range(M)])
    gibB = np.stack([gibbs_potts(J, h, 60, seed + 20000 + k) for k in range(M)])
    x0 = gibbs_potts(J, h, 5, seed)
    ctmc = np.stack([evolve_branch(x0, 25.0, J, h, S, rate_scale,
                                   np.random.default_rng(seed + 200 + k)) for k in range(M)])
    gA, gB, cM = sitemarg(gibA), sitemarg(gibB), sitemarg(ctmc)
    floor = np.abs(gB - gA).sum(1).mean()               # Gibbs-vs-Gibbs noise floor
    d1 = np.abs(cM - gA).sum(1).mean()                   # CTMC-vs-Gibbs
    ij = np.argwhere(np.linalg.norm(J, axis=(2, 3)) > 0); i, j = ij[0]
    def pm(X): return np.bincount(X[:, i] * A + X[:, j], minlength=A * A) / len(X)
    pfloor = np.abs(pm(gibB) - pm(gibA)).sum(); pc = np.abs(pm(ctmc) - pm(gibA)).sum()
    print(f"# single-site L1  CTMC-vs-Gibbs={d1:.3f}   Gibbs-vs-Gibbs floor={floor:.3f}")
    print(f"# contact pair-marginal L1  CTMC-vs-Gibbs={pc:.3f}   floor={pfloor:.3f}")
    print(f"# PASS (CTMC within 1.5x the sampling-noise floor): "
          f"{d1 < 1.5 * floor and pc < 1.5 * pfloor}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fam", default="PF00333")
    args = ap.parse_args()
    if args.check:
        _check()
    else:
        d = load_family(args.fam)
        print(f"# {args.fam}: N={d['nl']} L={d['L']} nodes={len(d['parent'])} "
              f"contacts={len(d['contacts'])}")
