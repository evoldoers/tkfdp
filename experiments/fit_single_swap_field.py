#!/usr/bin/env python3
"""Fit the SINGLE-SWAP field = Cayley<=1 truncated permutation field (identity +
the C(C-1)/2 single transpositions, 1+C(C-1)/2 states instead of C!). Reuses the
validated product-of-trees ELBO + exact-HR machinery of experiments/permfield_elbo
(column inside-outside, field BP, HR bridge accumulators, archetype M-step); the
ONLY differences are the truncated field state space, its GTR generator
(field_ctmc.build_truncated_field -- a reversible star id<->tau), and the field
M-step which fits the archetype-swap rates w_ab and the off-identity weight r=p1/p0
(instead of the full Ewens alpha / s_d over all Cayley distances).

Fits, given cluster-annotated MSAs: (1) swap rates w_ab + r=p1/p0; (2) class
mixture weights rho; (3) archetype stationaries pi^c. --synthetic simulates a
clustered corpus from a truncated field and checks the objective climbs +
parameter recovery, exactly as permfield_elbo does for the full field.

This is the validated numpy reference for the model; the pure-JAX
padded/vmap/logspace performance version (docs/single_swap_field_implementation.md)
is validated against it.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from scipy.linalg import expm as sexpm

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
import permfield_elbo as PE                                          # noqa: E402
from tkfdp.permfield.field_ctmc import build_truncated_field         # noqa: E402
from tkfdp.lg08 import PI_LG08                                       # noqa: E402

A = 20
PI0 = np.asarray(PI_LG08, float)


def trunc_states(C):
    """(states, dist, arch, pairs) for the 1+C(C-1)/2 Cayley<=1 field states."""
    ident = tuple(range(C))
    pairs = [(a, b) for a in range(C) for b in range(a + 1, C)]
    states = [ident] + [tuple(b if x == a else a if x == b else x for x in ident)
                        for (a, b) in pairs]
    dist = np.array([0] + [1] * len(pairs))
    arch = np.array([list(t) for t in states])
    return states, dist, arch, pairs


def trunc_field(C, pi_field, rho_field=1.0):
    """STAR-F81 field with a FREE per-state stationary pi_field (no Ewens/2-level
    constraint -- each of the 1+C(C-1)/2 states has its own probability). Only id<->tau
    edges (tau<->tau is Cayley 2). Jump-to-stationary form Q[i,j] = rho_field*pi_field[j]
    on the star, reversible w.r.t. pi_field for any pi_field, rho_field. rho_field is the
    overall field rate (the swap preferences now live in the free stationary, not in
    per-pair rates w). Returns (Q (nS,nS), pi (nS,))."""
    K = C * (C - 1) // 2; n = 1 + K
    pi = np.clip(np.asarray(pi_field, float), 1e-12, None); pi = pi / pi.sum()
    assert pi.shape == (n,), f"pi_field {pi.shape} != {(n,)}"
    Q = np.zeros((n, n))
    Q[0, 1:] = rho_field * pi[1:]                            # id -> tau_k  (rate ~ pi[tau_k])
    Q[1:, 0] = rho_field * pi[0]                             # tau_k -> id
    np.fill_diagonal(Q, 0.0); Q[np.diag_indices(n)] = -Q.sum(1)
    return Q, pi


def field_prior(C, strength=6.0, id_frac=0.7):
    """Dirichlet pseudocounts over the 1+C(C-1)/2 field states, moderately favouring
    the identity (state 0): mass strength*id_frac on identity, strength*(1-id_frac)
    spread over the transpositions. prior-mean pi_field(identity) = id_frac."""
    K = C * (C - 1) // 2
    ps = np.full(1 + K, strength * (1.0 - id_frac) / max(K, 1))
    ps[0] = strength * id_frac
    return ps


def solve_field_stationary(rootc, Uf, Wf, pseudo, damp=0.5, pi_prev=None):
    """Field M-step (star-F81, reversible, FREE stationary): the MAP stationary under a
    Dirichlet prior is pi_field ∝ (root occupancy + incoming usage) + pseudocounts
    (pseudo = field_prior(), moderately favouring identity). The F81 rate is
    rho_field = total transitions / sum_i Wf[i]*(1-pi_field[i]). Returns (pi_field,
    rho_field). No swap rates w: swap preferences are now IN the free stationary,
    regularised by the identity prior (guards the weak-identifiability over-spread)."""
    N_in = np.clip(Uf, 0.0, None).sum(0)                     # incoming usage per state
    occ = np.clip(rootc, 0.0, None) + N_in
    pi_new = occ + pseudo; pi_new = pi_new / pi_new.sum()
    if pi_prev is not None:
        pi_new = damp * pi_prev + (1 - damp) * pi_new; pi_new = pi_new / pi_new.sum()
    expo = float((np.clip(Wf, 0.0, None) * (1.0 - pi_new)).sum())
    rho = float(np.clip(Uf, 0.0, None).sum() / max(expo, 1e-9))
    return pi_new, rho


def simulate_trunc(parent, tau, C, pis, pi_field, rho_field, rho, clusters, seed=1):
    """Clustered corpus from a single-swap field with FREE stationary pi_field."""
    rng = np.random.default_rng(seed)
    states, dist, arch, pairs = trunc_states(C)
    nS = len(states)
    Qf, pif = trunc_field(C, pi_field, rho_field=rho_field)
    Pf = {v: sexpm(Qf * tau[v]) for v in range(len(parent)) if parent[v] >= 0}
    Qc = [PE.gtr_Q(pis[c]) for c in range(C)]
    Pc = {(a, v): sexpm(Qc[a] * tau[v]) for a in range(C)
          for v in range(len(parent)) if parent[v] >= 0}
    pre, post, root, ch = PE.orders(parent)
    nl = sum(len(ch[v]) == 0 for v in range(len(parent)))
    total = int(sum(clusters))
    msa = np.full((nl, total), 20, np.int8)
    partition, col = [], 0
    for m in clusters:
        th = np.zeros(len(parent), int); th[root] = rng.choice(nS, p=pif)
        for v in pre[1:]:
            th[v] = rng.choice(nS, p=Pf[v][th[parent[v]]])
        cols_here = []
        for _ in range(m):
            c = rng.choice(C, p=rho); x = np.zeros(len(parent), int)
            x[root] = rng.choice(A, p=pis[arch[th[root], c]])
            for v in pre[1:]:
                a = arch[th[parent[v]], c]
                x[v] = rng.choice(A, p=Pc[(a, v)][x[parent[v]]])
            msa[:, col] = x[:nl]; cols_here.append(col); col += 1
        partition.append(np.array(cols_here))
    return msa, partition


def fit(parent, tau, msa, C, partition=None, n_iter=25, seed=0,
        freeze_field=False, pis_init=None, rho_field0=0.3,
        field_strength=6.0, field_id_frac=0.7, verbose=True):
    """Product-of-trees ELBO with the single-swap field (FREE per-state stationary,
    star-F81). Coordinate ascent as permfield_elbo.fit; the field factor is the free
    stationary pi_field (Dirichlet prior favouring identity, strength/id_frac) + the
    F81 rate rho_field, via solve_field_stationary. partition=None -> singletons + loud
    warning."""
    from tkfdp.permfield.hr import eig_rev
    rng = np.random.default_rng(seed)
    N = len(parent); nl, L = msa.shape
    states, dist, arch, pairs = trunc_states(C)
    nS = len(states)
    pre, post, root, ch = PE.orders(parent)
    branches = [v for v in range(N) if parent[v] >= 0]
    if partition is None:
        PE.warn_no_structure(L)
        partition = [np.array([j]) for j in range(L)]
    partition = [np.asarray(cl, int) for cl in partition]
    if pis_init is not None:                                 # warm-seed (e.g. LG-C10)
        pis = np.clip(np.asarray(pis_init, float), 1e-9, None); pis /= pis.sum(1, keepdims=True)
        assert pis.shape == (C, A), f"pis_init {pis.shape} != {(C, A)}"
    else:
        pis = np.clip(PI0[None, :] * (1 + 0.1 * rng.standard_normal((C, A))), 1e-4, None)
        pis /= pis.sum(1, keepdims=True)
    pseudo = field_prior(C, field_strength, field_id_frac)   # identity-favouring Dirichlet
    pi_field = pseudo / pseudo.sum()                         # init stationary = prior mean
    rho_field = float(rho_field0); rho = np.ones(C) / C
    _, pif0 = trunc_field(C, pi_field, rho_field=rho_field)
    b_fields = [np.tile(pif0, (N, 1)) for _ in partition]
    hist = []; gtight = []                                    # ELBO + class-posterior tightness
    for it in range(n_iter):
        Qc = [PE.gtr_Q(pis[a]) for a in range(C)]
        eig_a = [eig_rev(Qc[a], pis[a]) for a in range(C)]
        Qf, pif = trunc_field(C, pi_field, rho_field=rho_field)
        eig_f = eig_rev(Qf, np.clip(pif, 1e-10, None))
        Pf = {v: np.clip(sexpm(Qf * tau[v]), 1e-300, None) for v in branches}
        Na = [np.zeros((A, A)) for _ in range(C)]; Ta = [np.zeros(A) for _ in range(C)]
        roota = [np.zeros(A) for _ in range(C)]
        Wf = np.zeros(nS); Uf = np.zeros((nS, nS)); rootc = np.zeros(nS)
        gsum = np.zeros(C); nmemb = 0; obj = 0.0
        gent = 0.0; gmx = 0.0; gncol = 0                     # class-posterior tightness accum
        for ci, cl in enumerate(partition):
            sub = msa[:, cl]
            es = PE.column_estep(parent, tau, sub, C, arch, Qc, pis, rho,
                                 b_fields[ci], pre, post, root, ch, nl)
            if freeze_field:
                obj += es["obj"]
            else:
                phi = PE.field_potentials(parent, C, arch, es, pis, N, nS, ch, root)
                b_fields[ci], xi, logZf = PE.field_bp(parent, Pf, pif, phi, pre, post,
                                                      root, ch)
                obj += es["obj"] + logZf
                PE.accum_field(C, tau, xi, b_fields[ci], Qf, pif, eig_f, root,
                               Wf, Uf, rootc)
            PE.accum_arch(C, arch, es, tau, Qc, eig_a, parent, Na, Ta, roota)
            g = es["gamma"]; gsum += g.sum(0); nmemb += len(cl)
            ent = -(g * np.log(np.clip(g, 1e-12, None))).sum(1) / np.log(max(C, 2))
            gent += float(ent.sum()); gmx += float(g.max(1).sum()); gncol += g.shape[0]
        hist.append(obj)
        mean_ent = gent / max(gncol, 1); mean_max = gmx / max(gncol, 1)   # class-post tightness
        gtight.append((mean_ent, mean_max))
        pis = PE.solve_arch(C, Na, Ta, roota, pis)
        if not freeze_field:
            pi_field, rho_field = solve_field_stationary(rootc, Uf, Wf, pseudo,
                                                         pi_prev=pi_field)
        rho = np.clip(gsum / max(nmemb, 1), 1e-6, None); rho /= rho.sum()
        if verbose and (it < 3 or it % 5 == 0 or it == n_iter - 1):
            dobj = obj - hist[-2] if len(hist) > 1 else float("nan")
            off = 1.0 - pi_field[0]                          # off-identity stationary mass
            print(f"  it {it:3d}  obj={obj:12.2f} (d={dobj:+8.2f})  "
                  f"gamma[Hnorm={mean_ent:.3f} maxprob={mean_max:.3f}]  "
                  f"pi_id={pi_field[0]:.3f} off={off:.3f} rho_field={rho_field:.3f}", flush=True)
    return dict(pis=pis, pi_field=pi_field, rho_field=rho_field, rho=rho,
                b_fields=b_fields, hist=hist, gtight=gtight, partition=partition,
                states=states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=3)
    ap.add_argument("--family", default="PF00013")
    ap.add_argument("--max-leaves", type=int, default=24)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--r0", type=float, default=0.3)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--n-clusters", type=int, default=120)
    ap.add_argument("--pis-init", choices=("random", "c10"), default="random",
                    help="c10 = warm-seed archetypes at the well-separated LG-C10 profiles")
    ap.add_argument("--rho-field0", type=float, default=0.3, help="initial F81 field rate")
    ap.add_argument("--field-strength", type=float, default=6.0,
                    help="Dirichlet prior strength on the free field stationary")
    ap.add_argument("--field-id-frac", type=float, default=0.7,
                    help="prior-mean stationary mass on the identity state (encourage identity)")
    args = ap.parse_args()
    d = np.load(f"data/pfam_processed_clv_top1000_thin128/{args.family}.npz",
                allow_pickle=True)
    parent, tau, keep = PE._prune_leaves(d["parent"].astype(int),
                                         d["tau"].astype(float), args.max_leaves)
    C = args.C
    fkw = dict(rho_field0=args.rho_field0, field_strength=args.field_strength,
               field_id_frac=args.field_id_frac)
    if args.synthetic:
        rng = np.random.default_rng(0)
        _, _, arch, pairs = trunc_states(C)
        pis_t = np.clip(PI0[None, :] * (1 + 0.7 * rng.standard_normal((C, A))), 1e-3, None)
        pis_t /= pis_t.sum(1, keepdims=True)
        pi_field_t = rng.dirichlet(field_prior(C, strength=8.0, id_frac=0.6))  # free stationary
        rho_field_t = 0.4; rho_t = rng.dirichlet(np.ones(C) * 2.0)
        clusters = [args.m] * args.n_clusters
        msa, partition = simulate_trunc(parent, tau, C, pis_t, pi_field_t, rho_field_t,
                                        rho_t, clusters, seed=1)
        print(f"# synthetic single-swap C={C}, {args.n_clusters} clusters m={args.m} "
              f"({msa.shape[1]} cols, {msa.shape[0]} leaves)  true pi_id={pi_field_t[0]:.3f} "
              f"rho_field={rho_field_t}", flush=True)
        t0 = time.time()
        res = fit(parent, tau, msa, C, partition=partition, n_iter=args.iters, **fkw)
        mono = bool(np.all(np.diff(res["hist"][3:]) > -1.0))
        cost = np.abs(res["pis"][:, None, :] - pis_t[None, :, :]).sum(-1)
        from scipy.optimize import linear_sum_assignment
        rr, cix = linear_sum_assignment(cost)
        perr = float(np.mean([cost[rr[i], cix[i]] for i in range(C)]))
        # gauge-align the field stationary via the archetype (transposition) relabel
        perm = np.empty(C, int); perm[rr] = cix
        pidx = {tuple(sorted(p)): k for k, p in enumerate(pairs)}
        pf = res["pi_field"]; pf_al = np.empty_like(pf); pf_al[0] = pf[0]
        for k, (a, b) in enumerate(pairs):
            pf_al[1 + pidx[tuple(sorted((int(perm[a]), int(perm[b]))))]] = pf[1 + k]
        pcorr = (float(np.corrcoef(pf_al[1:], pi_field_t[1:])[0, 1]) if len(pairs) > 1 else 1.0)
        print(f"# fit {time.time()-t0:.1f}s  monotone(obj)={mono}")
        print(f"# rho L1(gauge)={np.abs(res['rho']-rho_t[cix]).sum():.3f}   "
              f"archetype-profile L1={perr:.3f}")
        print(f"# field stationary: fit pi_id={pf[0]:.3f} vs true {pi_field_t[0]:.3f};  "
              f"off-identity pattern corr={pcorr:.3f}  L1(gauge)={np.abs(pf_al-pi_field_t).sum():.3f}")
    else:
        from experiments.permfield_fit import _leaf_msa
        msa = _leaf_msa(d, keep)
        pis0 = None
        if args.pis_init == "c10":
            sys.path.insert(0, "/home/yam/tkf-mixdom/python")
            from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c10
            prof, _, _ = le_gascuel_c10()
            pis0 = np.asarray(prof, float); pis0 /= pis0.sum(1, keepdims=True)
            assert C == pis0.shape[0], f"--pis-init c10 needs --C {pis0.shape[0]}"
        print(f"# single-swap C={C} on {args.family}: {msa.shape[0]} leaves x {msa.shape[1]} "
              f"cols (pis_init={args.pis_init} rho_field0={args.rho_field0} "
              f"id_frac={args.field_id_frac})", flush=True)
        kw = dict(n_iter=args.iters, pis_init=pis0, **fkw)
        print("# --- STATIC baseline (freeze_field: profile mixture, no field dynamics) ---")
        res_s = fit(parent, tau, msa, C, freeze_field=True, **kw)
        print("# --- DYNAMIC single-swap field ---")
        res_d = fit(parent, tau, msa, C, **kw)

        def _conv(h):
            return (bool(np.all(np.diff(h[3:]) > -1.0)),
                    (h[-1] - h[-5] if len(h) >= 5 else float("nan")))
        ms, dS = _conv(res_s["hist"]); md, dD = _conv(res_d["hist"])
        eS, mxS = res_s["gtight"][-1]; eD, mxD = res_d["gtight"][-1]
        print("\n# === ELBO CONVERGENCE ===")
        print(f"#   static : final={res_s['hist'][-1]:.2f}  monotone={ms}  last5-dobj={dS:+.3f}")
        print(f"#   dynamic: final={res_d['hist'][-1]:.2f}  monotone={md}  last5-dobj={dD:+.3f}")
        print("# === CLASS-POSTERIOR TIGHTNESS  (static baseline vs dynamic field) ===")
        print(f"#   static : Hnorm={eS:.3f}  maxprob={mxS:.3f}")
        print(f"#   dynamic: Hnorm={eD:.3f}  maxprob={mxD:.3f}")
        dH = eD - eS
        verdict = ("PATHOLOGICAL diffusion" if dH > 0.05 else
                   "comparable (healthy)" if dH >= -0.05 else "dynamic sharper")
        print(f"#   -> dynamic vs static dHnorm={dH:+.3f}: {verdict}  "
              f"(some diffuse columns are EXPECTED in both)")
        if pis0 is not None:
            print(f"#   archetype drift from C10 (L1/arch): {np.round(np.abs(res_d['pis']-pis0).sum(1),3)}")
        pf = res_d["pi_field"]; top = np.argsort(-pf[1:])[:6]
        print(f"#   field: pi_id={pf[0]:.3f}  off-identity={1-pf[0]:.3f}  rho_field={res_d['rho_field']:.3f}")
        print(f"#   top off-identity states (idx,pi): "
              f"{[(int(t+1), round(float(pf[t+1]),4)) for t in top]}")


if __name__ == "__main__":
    main()
