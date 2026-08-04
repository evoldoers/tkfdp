#!/usr/bin/env python3
"""MULTI-FAMILY EM driver for the SINGLE-SWAP FIELD with GLOBAL shared params and
per-family-per-cluster LOCAL field posteriors.

Reuses the COMPILE-ONCE JAX E-step of ``experiments/fit_single_swap_field_jax``
(``full_estep_jit``, tree passed as a pytree ARGUMENT so ONE XLA trace serves
every family) and its closed-form M-steps.  The single-family numpy reference
``experiments/fit_single_swap_field`` (which reuses ``permfield_elbo``) is the
ground truth; the JAX E-step is validated against it to ~1e-13.

MODEL (same as the single-family fit):
  * GLOBAL (shared across all families): archetype stationaries ``pis`` (C x 20,
    warm-seeded at LG-C10), the free field stationary ``pi_field`` (1+C(C-1)/2
    states), the class weights ``rho`` (C), and the F81 field rate ``rho_field``.
  * LOCAL: each family's per-cluster field posterior ``b_field`` over the
    1+C(C-1)/2 truncated single-swap states, on that family's tree.

CORPUS: every family that has BOTH a size-2 contact-pair partition
(``data/pdb_partition_clv_top1000_sifts/<fam>.npz``) AND a thinned tree
(``data/pfam_processed_clv_top1000_thin128/<fam>.npz``).  Each contact PAIR is
one m=2 cluster (NOT singletons).  Leaves pruned exactly as the reference
(``permfield_elbo._prune_leaves`` -> ``permfield_fit._leaf_msa``); every tree
padded to a corpus-wide common ``MAX_NODES`` and clusters processed in fixed-size
``CHUNK`` batches so the compiled E-step is traced ONCE.

EM: each iteration runs the compiled E-step over every family/chunk, ACCUMULATES
the pooled sufficient statistics (arch Na/Ta/roota; field Wf/Uf/rootc; class
gsum) summed across all families and clusters, then ONE global M-step
(``solve_arch_jax`` for pis, ``solve_field_stationary_jax`` for pi_field+rho_field
under an identity-favouring Dirichlet prior scaled to the corpus, pooled rho).

Usage:
  # validate the pooling is an exact sum (3 families, f64):
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=6 \
    PYTHONPATH=src:experiments:/home/yam/tkf-mixdom/python \
    python3 experiments/fit_single_swap_field_corpus.py --validate

  # corpus fit (GPU):
  JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 \
    PYTHONPATH=src:experiments:/home/yam/tkf-mixdom/python \
    python3 experiments/fit_single_swap_field_corpus.py --fit --C 10 --iters 20 \
      --out results/single_swap_corpus_C10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                                  # repo root (experiments pkg)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, "/home/yam/tkf-mixdom/python")

# fit_single_swap_field_jax enables x64 at import; keep that for validation.
import jax
import jax.numpy as jnp
import fit_single_swap_field as FS                         # noqa: E402
import fit_single_swap_field_jax as FJ                     # noqa: E402
import permfield_elbo as PE                                # noqa: E402
from experiments.permfield_fit import _leaf_msa            # noqa: E402

A = 20
AA = "ACDEFGHIKLMNPQRSTVWY"                                # LG08 alphabetical order
PART_DIR = "data/pdb_partition_clv_top1000_sifts"
THIN_DIR = "data/pfam_processed_clv_top1000_thin128"


# ============================================================================
# corpus loading (prune leaves as reference; contact pairs = m=2 clusters)
# ============================================================================
def list_corpus_families():
    part = {os.path.basename(f)[:-4] for f in glob.glob(f"{PART_DIR}/*.npz")}
    thin = {os.path.basename(f)[:-4] for f in glob.glob(f"{THIN_DIR}/*.npz")}
    return sorted(part & thin)


def load_family(fam, max_leaves):
    """Prune leaves exactly as the reference, return per-family pack or None."""
    d = np.load(f"{THIN_DIR}/{fam}.npz", allow_pickle=True)
    parent, tau, keep = PE._prune_leaves(d["parent"].astype(int),
                                         d["tau"].astype(float), max_leaves)
    msa = _leaf_msa(d, keep)                               # (nl, L) int8
    p = np.load(f"{PART_DIR}/{fam}.npz", allow_pickle=True)
    pairs = np.asarray(p["pairs"], int)                    # (P, 2) column indices
    kind = np.asarray(p["kind"])
    if pairs.shape[0] == 0:
        return None
    L = msa.shape[1]
    ok = (pairs[:, 0] < L) & (pairs[:, 1] < L)             # guard column bounds
    pairs, kind = pairs[ok], kind[ok]
    if pairs.shape[0] == 0:
        return None
    N = len(parent)
    # leaves are nodes with no children (reference guarantees ids 0..nl-1)
    has_child = np.zeros(N, bool)
    for v in range(N):
        if parent[v] >= 0:
            has_child[parent[v]] = True
    nl = int((~has_child).sum())
    return dict(fam=fam, parent=parent, tau=tau, msa=msa, pairs=pairs, kind=kind,
                N=N, nl=nl, nCl=pairs.shape[0], L=L)


# ============================================================================
# chunk packing (per family): CHUNK clusters -> padded leaf_obs / colmask / b
# ============================================================================
def pack_chunk(fp, b_store, cids, MAX_NODES, nS, Lmax, CHUNK, pi_field,
               freeze_onehot=None):
    leaf_obs = np.full((CHUNK, MAX_NODES, Lmax), 20, np.int64)
    colmask = np.zeros((CHUNK, Lmax))
    cmask = np.zeros(CHUNK)
    b_chunk = np.tile(pi_field, (CHUNK, MAX_NODES, 1))
    nl = fp["nl"]; msa = fp["msa"]; pairs = fp["pairs"]; N = fp["N"]
    for slot, ci in enumerate(cids):
        cmask[slot] = 1.0
        cols = pairs[ci]
        for l, col in enumerate(cols):
            leaf_obs[slot, :nl, l] = msa[:nl, col].astype(np.int64)
            colmask[slot, l] = 1.0
        if freeze_onehot is not None:
            b_chunk[slot, :, :] = freeze_onehot
        else:
            b_chunk[slot, :N, :] = b_store[ci]
    return (jnp.asarray(leaf_obs), jnp.asarray(colmask), jnp.asarray(cmask),
            jnp.asarray(b_chunk))


# ============================================================================
# pooled sufficient-statistic accumulator (numpy, host side)
# ============================================================================
def fresh_pool(C, nS):
    return dict(Na=np.zeros((C, A, A)), Ta=np.zeros((C, A)), roota=np.zeros((C, A)),
                Wf=np.zeros(nS), Uf=np.zeros((nS, nS)), rootc=np.zeros(nS),
                gsum=np.zeros(C), obj=0.0)


def add_agg(pool, agg):
    pool["Na"] += np.asarray(agg["Na"]); pool["Ta"] += np.asarray(agg["Ta"])
    pool["roota"] += np.asarray(agg["roota"]); pool["Wf"] += np.asarray(agg["Wf"])
    pool["Uf"] += np.asarray(agg["Uf"]); pool["rootc"] += np.asarray(agg["rootc"])
    pool["gsum"] += np.asarray(agg["gsum"]); pool["obj"] += float(agg["obj"])


# ============================================================================
# one E-step pass over the whole corpus (pooled stats + tightness + writeback)
# ============================================================================
def corpus_estep(families, packs, b_fields, arch_oh, pis_j, rho_j, pif_j, rhof_j,
                 MAX_NODES, nS, CHUNK, C, pi_field_np, freeze=False,
                 freeze_onehot=None, writeback=True):
    pool = fresh_pool(C, nS)
    Hsum = 0.0; MXsum = 0.0; ncol = 0; nmemb = 0
    Lmax = 2
    for fam in families:
        fp = packs[fam]; tree = fp["tree"]; nCl = fp["nCl"]
        b_store = None if freeze else b_fields[fam]
        for start in range(0, nCl, CHUNK):
            cids = list(range(start, min(start + CHUNK, nCl)))
            leaf_obs, colmask, cmask, b_chunk = pack_chunk(
                fp, b_store, cids, MAX_NODES, nS, Lmax, CHUNK, pi_field_np,
                freeze_onehot=freeze_onehot if freeze else None)
            res, agg, _ = FJ.full_estep_jit(tree, arch_oh, b_chunk, leaf_obs,
                                            colmask, cmask, pis_j, rho_j,
                                            pif_j, rhof_j)
            add_agg(pool, agg)
            gamma = np.asarray(res["gamma"])               # (CHUNK, Lmax, C)
            bnew = np.asarray(res["b_new"]) if (writeback and not freeze) else None
            for slot, ci in enumerate(cids):
                g = gamma[slot, :Lmax, :]                  # (2, C)
                H = -(g * np.log(np.clip(g, 1e-12, None))).sum(1) / np.log(max(C, 2))
                Hsum += float(H.sum()); MXsum += float(g.max(1).sum()); ncol += Lmax
                nmemb += Lmax
                if bnew is not None:
                    b_store[ci] = bnew[slot, :fp["N"]]
    tight = (Hsum / max(ncol, 1), MXsum / max(ncol, 1))
    return pool, tight, nmemb


# ============================================================================
# corpus EM fit
# ============================================================================
def run_fit(families, packs, C, MAX_NODES, n_iter, rho_field0, field_strength,
            field_id_frac, freeze=False, verbose=True, seed=0):
    _, _, arch, pairs = FS.trunc_states(C)
    nS = arch.shape[0]
    arch_oh = FJ.build_arch_onehot(arch, C)

    # warm-seed archetypes at LG-C10
    from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c10
    prof, _, _ = le_gascuel_c10()
    pis = np.asarray(prof, float); pis = np.clip(pis, 1e-9, None)
    pis /= pis.sum(1, keepdims=True)
    assert pis.shape == (C, A), f"C10 gives {pis.shape}, need C={C}"

    pseudo = FS.field_prior(C, field_strength, field_id_frac)
    pi_field = pseudo / pseudo.sum()
    rho_field = float(rho_field0)
    rho = np.ones(C) / C

    # LOCAL per-family per-cluster field posteriors (real nodes only)
    onehot = np.zeros(nS); onehot[0] = 1.0
    freeze_onehot = np.tile(onehot, (MAX_NODES, 1)) if freeze else None
    b_fields = None
    if not freeze:
        b_fields = {fam: np.tile(pi_field, (packs[fam]["nCl"], packs[fam]["N"], 1))
                    for fam in families}

    pseudo_j = jnp.asarray(pseudo)
    hist = []; tights = []; drift_hist = []
    t_iter = []
    for it in range(n_iter):
        pis_j = jnp.asarray(pis); rho_j = jnp.asarray(rho)
        pif_j = jnp.asarray(pi_field); rhof_j = jnp.asarray(rho_field)
        t0 = time.time()
        pool, tight, nmemb = corpus_estep(
            families, packs, b_fields, arch_oh, pis_j, rho_j, pif_j, rhof_j,
            MAX_NODES, nS, CHUNK, C, pi_field,
            freeze=freeze, freeze_onehot=freeze_onehot)
        hist.append(pool["obj"]); tights.append(tight)
        # M-steps
        pis = np.asarray(FJ.solve_arch_jax(
            jnp.asarray(pool["Na"]), jnp.asarray(pool["Ta"]),
            jnp.asarray(pool["roota"]), jnp.asarray(pis)))
        if not freeze:
            pif_new, rhof_new = FJ.solve_field_stationary_jax(
                jnp.asarray(pool["rootc"]), jnp.asarray(pool["Uf"]),
                jnp.asarray(pool["Wf"]), pseudo_j, jnp.asarray(pi_field))
            pi_field = np.asarray(pif_new); rho_field = float(rhof_new)
        rho = np.clip(pool["gsum"] / max(nmemb, 1), 1e-6, None); rho /= rho.sum()
        dt = time.time() - t0; t_iter.append(dt)
        if verbose:
            dobj = hist[-1] - hist[-2] if len(hist) > 1 else float("nan")
            tag = "STATIC" if freeze else "DYNAMIC"
            extra = ("" if freeze else
                     f"  pi_id={pi_field[0]:.3f} off={1 - pi_field[0]:.3f} "
                     f"rho_field={rho_field:.3f}")
            print(f"  [{tag}] it {it:3d}  obj={hist[-1]:14.2f} (d={dobj:+10.2f})  "
                  f"Hnorm={tight[0]:.3f} maxp={tight[1]:.3f}{extra}  ({dt:.1f}s)",
                  flush=True)
    return dict(pis=pis, pi_field=pi_field, rho_field=rho_field, rho=rho,
                hist=hist, tights=tights, b_fields=b_fields, pairs=pairs,
                arch=arch, C=C, nS=nS, t_iter=t_iter)


CHUNK = 8   # cluster batch size (fixed -> single XLA trace); overridable in main


# ============================================================================
# preparation: build padded trees + choose MAX_NODES
# ============================================================================
def prepare(families, max_leaves, pad_extra=2, verbose=True):
    packs = {}; drop = 0
    for fam in families:
        fp = load_family(fam, max_leaves)
        if fp is None:
            drop += 1; continue
        packs[fam] = fp
    fams = [f for f in families if f in packs]
    MAX_NODES = max(packs[f]["N"] for f in fams) + pad_extra
    tot_cl = sum(packs[f]["nCl"] for f in fams)
    if verbose:
        print(f"# corpus: {len(fams)} families ({drop} dropped: no valid pairs), "
              f"{tot_cl} contact-pair clusters, MAX_NODES={MAX_NODES} "
              f"(max_leaves={max_leaves})", flush=True)
    for f in fams:
        tree, _, _ = FJ.pad_tree(packs[f]["parent"], packs[f]["tau"], MAX_NODES)
        packs[f]["tree"] = tree
    return fams, packs, MAX_NODES


# ============================================================================
# VALIDATION: pooled one-iteration stats == sum of single-family stats
# ============================================================================
def validate(n_families=3, max_leaves=32, C=10):
    print(f"# ===== POOLING VALIDATION (C={C}, {n_families} families, "
          f"max_leaves={max_leaves}, f64) =====", flush=True)
    all_fams = list_corpus_families()
    fams, packs, MAX_NODES = prepare(all_fams[:n_families + 4], max_leaves)
    fams = fams[:n_families]
    _, _, arch, pairs = FS.trunc_states(C)
    nS = arch.shape[0]
    arch_oh = FJ.build_arch_onehot(arch, C)

    # fixed GLOBAL params identical for both paths
    from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c10
    prof, _, _ = le_gascuel_c10()
    pis = np.clip(np.asarray(prof, float), 1e-9, None); pis /= pis.sum(1, keepdims=True)
    pseudo = FS.field_prior(C, 6.0, 0.8); pi_field = pseudo / pseudo.sum()
    rho_field = 0.15; rho = np.ones(C) / C
    pis_j = jnp.asarray(pis); rho_j = jnp.asarray(rho)
    pif_j = jnp.asarray(pi_field); rhof_j = jnp.asarray(rho_field)

    # per-family b_fields init at prior mean (real nodes); identical both paths
    b_fields = {f: np.tile(pi_field, (packs[f]["nCl"], packs[f]["N"], 1))
                for f in fams}

    # --- PATH A: single-family stats (one full_estep per family), then sum ---
    Lmax = 2
    singles = fresh_pool(C, nS)
    for f in fams:
        fp = packs[f]; nCl = fp["nCl"]
        # ONE chunk holding all clusters (CHUNK_f = nCl)
        leaf_obs, colmask, cmask, b_chunk = pack_chunk(
            fp, b_fields[f], list(range(nCl)), MAX_NODES, nS, Lmax, nCl, pi_field)
        _, agg, _ = FJ.full_estep_jit(fp["tree"], arch_oh, b_chunk, leaf_obs,
                                      colmask, cmask, pis_j, rho_j, pif_j, rhof_j)
        add_agg(singles, agg)

    # --- PATH B: pooled corpus accumulation (chunked at CHUNK) ---
    pool, _, _ = corpus_estep(fams, packs, b_fields, arch_oh, pis_j, rho_j,
                              pif_j, rhof_j, MAX_NODES, nS, CHUNK, C, pi_field,
                              writeback=False)

    keys = ["Na", "Ta", "roota", "Wf", "Uf", "rootc", "gsum"]
    print(f"{'stat':10s} {'pooled - sum(single)  max-abs-diff':>36s}")
    worst = 0.0
    for k in keys:
        d = float(np.max(np.abs(pool[k] - singles[k])))
        worst = max(worst, d)
        print(f"{k:10s} {d:36.3e}")
    d_obj = abs(pool["obj"] - singles["obj"]); worst = max(worst, d_obj)
    print(f"{'obj':10s} {d_obj:36.3e}")
    print(f"\n# POOLING max-abs-diff over ALL stats: {worst:.3e}   "
          f"(gate 1e-8: {'PASS' if worst <= 1e-8 else 'FAIL'})", flush=True)
    return worst


# ============================================================================
# FLIP ANALYSIS
# ============================================================================
def _chem_class(top_aa):
    """Rough chemistry tag from the archetype's dominant amino acids."""
    s = set(top_aa[:3])
    acid = s & set("DE"); base = s & set("KRH"); arom = s & set("FYW")
    if acid:
        return "acidic"
    if base:
        return "basic"
    if arom:
        return "aromatic"
    if s & set("ILVMAC"):
        return "hydrophobic"
    if s & set("STNQ"):
        return "polar"
    if "G" in s or "P" in s:
        return "special"
    return "mixed"


def _complementary(ca, cb, aa_a, aa_b):
    sa, sb = set(aa_a[:3]), set(aa_b[:3])
    if (ca, cb) in (("acidic", "basic"), ("basic", "acidic")):
        return "COMPLEMENTARY (salt-bridge acid<->base)"
    if ca == "aromatic" and cb == "aromatic":
        return "COMPLEMENTARY (aromatic<->aromatic stack)"
    if ca == "hydrophobic" and cb == "hydrophobic":
        return "plausible (hydrophobic volume swap)"
    if {ca, cb} == {"basic", "aromatic"}:
        return "COMPLEMENTARY (cation-pi)"
    if ca == cb:
        return f"same-class ({ca}) co-conserved"
    return "mixed / not obviously compensatory"


def flip_analysis(res, top_k=10):
    C = res["C"]; pis = res["pis"]; pi_field = res["pi_field"]; pairs = res["pairs"]
    order = np.argsort(-pi_field[1:])                      # off-identity states
    top_aa = [[AA[i] for i in np.argsort(-pis[c])[:4]] for c in range(C)]
    rows = []
    for rank, k in enumerate(order[:top_k]):
        a, b = pairs[k]
        ca, cb = _chem_class(top_aa[a]), _chem_class(top_aa[b])
        verdict = _complementary(ca, cb, top_aa[a], top_aa[b])
        rows.append(dict(rank=rank + 1, state=int(k + 1), weight=float(pi_field[k + 1]),
                         a=int(a), b=int(b), aa_a="".join(top_aa[a][:3]),
                         aa_b="".join(top_aa[b][:3]), ca=ca, cb=cb, verdict=verdict))
    return rows, top_aa


def confirmed_flip_overlap(res, fams, packs, top_states):
    """For confirmed-flip clusters present in the fitted corpus, report their
    fitted field posterior off-identity mass and top swap state, and whether that
    swap is among the globally top-weighted states."""
    cf = json.load(open(f"{PART_DIR}/confirmed_flips.json"))
    pairs = res["pairs"]
    b_fields = res["b_fields"]
    top_set = set(int(s) for s in top_states)
    out = []
    fitted_fams = set(fams)
    for rec in cf:
        fam = rec["family"]
        if fam not in fitted_fams or b_fields is None:
            continue
        fp = packs[fam]
        ci, cj = rec["i"], rec["j"]
        # find the contact-pair cluster matching (i,j)
        match = None
        for idx in range(fp["nCl"]):
            p = fp["pairs"][idx]
            if {int(p[0]), int(p[1])} == {ci, cj}:
                match = idx; break
        if match is None:
            continue
        b = b_fields[fam][match]                          # (N, nS) node posteriors
        node_mean = b.mean(0)                             # mean over nodes
        off_mass = float(1.0 - node_mean[0])
        top_state = int(np.argmax(node_mean[1:]) + 1)
        a, bk = pairs[top_state - 1]
        out.append(dict(family=fam, i=ci, j=cj, off_mass=off_mass,
                        top_state=top_state, arch_pair=(int(a), int(bk)),
                        in_global_top=top_state in top_set, miz=rec.get("miz")))
    return out, len(cf)


def cluster_field_summary(res, fams, packs):
    """Mean off-identity field mass over ALL fitted clusters, by kind."""
    b_fields = res["b_fields"]
    if b_fields is None:
        return {}
    from collections import defaultdict
    acc = defaultdict(lambda: [0.0, 0])
    for fam in fams:
        fp = packs[fam]
        for idx in range(fp["nCl"]):
            b = b_fields[fam][idx]
            off = float(1.0 - b.mean(0)[0])
            kd = str(fp["kind"][idx])
            acc[kd][0] += off; acc[kd][1] += 1
            acc["ALL"][0] += off; acc["ALL"][1] += 1
    return {k: (v[0] / max(v[1], 1), v[1]) for k, v in acc.items()}


# ============================================================================
# main
# ============================================================================
def main():
    global CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--C", type=int, default=10)
    ap.add_argument("--max-leaves", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--static-iters", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--rho-field0", type=float, default=0.15)
    ap.add_argument("--field-strength-per-family", type=float, default=3.0)
    ap.add_argument("--field-id-frac", type=float, default=0.8)
    ap.add_argument("--n-families", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="results/single_swap_corpus_C10")
    ap.add_argument("--val-families", type=int, default=3)
    ap.add_argument("--val-max-leaves", type=int, default=32)
    args = ap.parse_args()
    CHUNK = args.chunk

    if args.validate:
        validate(n_families=args.val_families, max_leaves=args.val_max_leaves,
                 C=args.C)
        return

    if not args.fit:
        print("Use --validate or --fit"); return

    all_fams = list_corpus_families()
    if args.n_families:
        all_fams = all_fams[:args.n_families]
    fams, packs, MAX_NODES = prepare(all_fams, args.max_leaves)
    field_strength = args.field_strength_per_family * len(fams)
    print(f"# field prior: strength={field_strength:.1f} "
          f"({args.field_strength_per_family}/family x {len(fams)} fams), "
          f"id_frac={args.field_id_frac}, rho_field0={args.rho_field0}, "
          f"CHUNK={CHUNK}", flush=True)

    print("\n# === DYNAMIC single-swap field fit ===", flush=True)
    t0 = time.time()
    res = run_fit(fams, packs, args.C, MAX_NODES, args.iters, args.rho_field0,
                  field_strength, args.field_id_frac, freeze=False)
    print(f"# dynamic fit wall: {time.time() - t0:.1f}s", flush=True)

    print("\n# === STATIC freeze-field baseline (C-class profile mixture) ===",
          flush=True)
    res_s = run_fit(fams, packs, args.C, MAX_NODES, args.static_iters,
                    args.rho_field0, field_strength, args.field_id_frac,
                    freeze=True)

    # -------- reports --------
    from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c10
    prof, _, _ = le_gascuel_c10()
    c10 = np.clip(np.asarray(prof, float), 1e-9, None); c10 /= c10.sum(1, keepdims=True)

    h = res["hist"]
    mono = bool(np.all(np.diff(h[3:]) > -abs(h[-1]) * 1e-6 - 1.0))
    last5 = h[-1] - h[-5] if len(h) >= 5 else float("nan")
    eD, mxD = res["tights"][-1]; eS, mxS = res_s["tights"][-1]
    dH = eD - eS
    verdict = ("PATHOLOGICAL diffusion" if dH > 0.05 else
               "comparable (healthy)" if dH >= -0.05 else "dynamic SHARPER")
    drift = np.abs(res["pis"] - c10).sum(1)

    print("\n# ================= RESULTS =================")
    print(f"# ELBO: final={h[-1]:.2f}  monotone(it>=3)={mono}  last5-dobj={last5:+.3f}")
    print(f"# ELBO trajectory (every ~4 it): "
          f"{[round(float(x), 1) for x in h[::max(1, len(h) // 6)]]}")
    print(f"# gamma tightness  DYNAMIC: Hnorm={eD:.3f} maxprob={mxD:.3f}")
    print(f"# gamma tightness  STATIC : Hnorm={eS:.3f} maxprob={mxS:.3f}")
    print(f"# dHnorm(dynamic-static) = {dH:+.3f}  -> {verdict}")
    print(f"# archetype drift from LG-C10 (L1/arch): {np.round(drift, 3)}")
    print(f"# field: pi_id={res['pi_field'][0]:.3f} off={1 - res['pi_field'][0]:.3f} "
          f"rho_field={res['rho_field']:.3f}")

    rows, top_aa = flip_analysis(res, top_k=12)
    print("\n# ============ TOP FAVORED SWAPS (field off-identity stationary) ============")
    print(f"# {'rk':>2} {'state':>5} {'weight':>8}  {'pair':>7}  "
          f"{'AAs(a)':>7} {'AAs(b)':>7}  chemistry -> verdict")
    for r in rows:
        print(f"# {r['rank']:>2} {r['state']:>5} {r['weight']:>8.4f}  "
              f"({r['a']:>2},{r['b']:>2})  {r['aa_a']:>7} {r['aa_b']:>7}  "
              f"{r['ca']}/{r['cb']} -> {r['verdict']}")

    top_states = [r["state"] for r in rows]
    cf_over, n_cf = confirmed_flip_overlap(res, fams, packs, top_states)
    print(f"\n# ============ CONFIRMED-FLIP OVERLAP ({len(cf_over)} of {n_cf} "
          f"confirmed flips fall on fitted contact-pair clusters) ============")
    if cf_over:
        hi = [c for c in cf_over if c["off_mass"] > 0.3]
        print(f"# confirmed-flip clusters with fitted off-identity mass > 0.3: "
              f"{len(hi)}/{len(cf_over)}")
        print(f"# {'family':>10} {'i':>4} {'j':>4} {'offmass':>8} {'topstate':>8} "
              f"{'archpair':>9} {'inTop':>6} {'miz':>5}")
        for c in sorted(cf_over, key=lambda x: -x["off_mass"])[:20]:
            print(f"# {c['family']:>10} {c['i']:>4} {c['j']:>4} {c['off_mass']:>8.3f} "
                  f"{c['top_state']:>8} {str(c['arch_pair']):>9} "
                  f"{str(c['in_global_top']):>6} {str(c['miz']):>5}")
    ksum = cluster_field_summary(res, fams, packs)
    print("\n# mean fitted off-identity field mass by contact KIND (all clusters):")
    for k in ["ALL", "saltbridge", "cation_pi", "disulfide", "volume", "nn"]:
        if k in ksum:
            print(f"#   {k:>12}: mean_off={ksum[k][0]:.3f}  (n={ksum[k][1]})")

    # -------- persist --------
    os.makedirs(args.out, exist_ok=True)
    np.savez(f"{args.out}/fit.npz", pis=res["pis"], pi_field=res["pi_field"],
             rho_field=res["rho_field"], rho=res["rho"], hist=np.asarray(res["hist"]),
             tights=np.asarray(res["tights"]), pairs=res["pairs"],
             drift=drift, c10=c10)
    json.dump(dict(flip_rows=rows, cf_overlap=cf_over, kind_summary=ksum,
                   dHnorm=dH, tight_dyn=[eD, mxD], tight_static=[eS, mxS],
                   elbo_final=h[-1], monotone=mono, last5=last5,
                   families=fams, n_clusters=int(sum(packs[f]["nCl"] for f in fams))),
              open(f"{args.out}/report.json", "w"), indent=1, default=float)
    print(f"\n# saved -> {args.out}/fit.npz + report.json", flush=True)


if __name__ == "__main__":
    main()
