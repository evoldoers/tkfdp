"""Validate the scalable factored pair HR (cluster_hr_exact.pair_tree_hr) against
the dense L*A*A tree HR, in every regime + on a real family tree.

The dense path (cluster_hr_exact.pair_tree_hr_dense) is the same reversible
800-state tree HR validated in analysis/scripts/check_cluster_hr.py (which cross-
checks it against expm and an endpoint-conditioned uniformisation Monte-Carlo).
Here we confirm the COMPOUND up-down factored HR reproduces it -- the scalable
form the trainer needs (check_cluster_hr's factored path enumerates L^n field
configs and does not scale to real trees).

Pure CPU / numpy.  Run:  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python3 \
    analysis/scripts/validate_cluster_hr_exact.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tkfdp.lg08 import get_lg08
from tkfdp.coupling.dynfield.phylo_elbo import cluster_hr_exact as X

A = 20


def make_archetypes():
    S, pi_bg = get_lg08()
    try:
        sys.path.insert(0, os.path.expanduser("~/tkf-mixdom/python"))
        from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c10
        prof, _, _ = le_gascuel_c10()
        prof = np.asarray(prof)
        ACID = [2, 3]; BASE = [8, 14, 6]
        net = prof[:, BASE].sum(1) - prof[:, ACID].sum(1)
        base_rich = int(np.argmax(net)); acid_rich = int(np.argmin(net))
        pi_arch = np.stack([pi_bg, prof[acid_rich], prof[base_rich], prof[0]])
    except Exception as e:
        print(f"[warn] C10 unavailable ({e}); synthetic archetypes")
        acid = pi_bg.copy(); acid[[2, 3]] *= 4.0; acid /= acid.sum()
        base = pi_bg.copy(); base[[8, 14, 6]] *= 4.0; base /= base.sum()
        pi_arch = np.stack([pi_bg, acid, base, pi_bg])
    pi_arch = pi_arch / pi_arch.sum(1, keepdims=True)
    return S, pi_arch


def _reldiff(a, b, floor=1e-9):
    a = np.asarray(a, float); b = np.asarray(b, float)
    denom = np.maximum(np.abs(a) + np.abs(b), floor)
    return float(np.max(np.abs(a - b) / denom))


def make_trees():
    cherry_parent = np.array([-1, 0, 0])
    cherry_tau = np.array([0.0, 0.35, 0.8])
    cherry_leaves = {1: (2, 8), 2: (3, 14)}
    tri_parent = np.array([-1, 0, 0, 1, 1])
    tri_tau = np.array([0.0, 0.4, 0.9, 0.3, 0.6])
    tri_leaves = {2: (8, 3), 3: (2, 14), 4: (3, 8)}
    # a 4-leaf tree with a gap on one leaf's second residue
    q_parent = np.array([-1, 0, 0, 1, 1, 2, 2])
    q_tau = np.array([0.0, 0.5, 0.3, 0.4, 0.7, 0.2, 0.6])
    q_leaves = {3: (2, 8), 4: (3, 20), 5: (8, 3), 6: (14, 14)}   # 20 = gap
    return {
        "cherry": (cherry_parent, cherry_tau, cherry_leaves),
        "3-leaf": (tri_parent, tri_tau, tri_leaves),
        "4-leaf+gap": (q_parent, q_tau, q_leaves),
    }


def _cols_from_leaves(parent, leaves):
    n = len(parent)
    xcol = np.full(n, -1); ycol = np.full(n, -1)
    for v, (x, y) in leaves.items():
        xcol[v] = x; ycol[v] = y
    return xcol, ycol


def run_regime(name, a1, a2, pi_arch, S, rho, rho_chain, trees):
    print(f"\n=== {name}  rho_chain={rho_chain} a1={list(a1)} a2={list(a2)} ===")
    worstN = worstT = worstJ = 0.0
    for tname, (parent, tau, leaves) in trees.items():
        xcol, ycol = _cols_from_leaves(parent, leaves)
        Nk_d, Tk_d, J_d, logZ = X.pair_tree_hr_dense(parent, tau, xcol, ycol,
                                                     a1, a2, pi_arch, S, rho, rho_chain)
        Nk_f, Tk_f, J_f = X.pair_tree_hr(parent, tau, xcol, ycol,
                                         a1, a2, pi_arch, S, rho, rho_chain)
        dN = _reldiff(Nk_d, Nk_f); dT = _reldiff(Tk_d, Tk_f); dJ = abs(J_d - J_f)
        worstN = max(worstN, dN); worstT = max(worstT, dT); worstJ = max(worstJ, dJ)
        print(f"  [{tname:11s}] reldiff N^k={dN:.2e}  T^k={dT:.2e}  "
              f"|dJumps|={dJ:.2e}  (jumps_dense={J_d:.4f})")
    return worstN, worstT, worstJ


def check_singleton(pi_arch, S, rho):
    """Cross-check single_tree_hr (40-state) against the pair path with the
    partner residue set to all-gap and a2==a1 (a spectator that contributes
    nothing) -- both must give residue-1's per-arch HR identically."""
    print("\n=== singleton 40-state vs dense (partner all-gap) ===")
    parent = np.array([-1, 0, 0, 1, 1]); tau = np.array([0.0, 0.4, 0.9, 0.3, 0.6])
    xleaf = {2: 8, 3: 2, 4: 3}
    a_n = np.array([1, 2])
    n = len(parent)
    xcol = np.full(n, -1)
    for v, x in xleaf.items():
        xcol[v] = x
    Nk_s, Tk_s, J_s = X.single_tree_hr(parent, tau, xcol, a_n, pi_arch, S, rho, 0.7)
    # dense reference on the 40-state chain
    Q40, p40 = X.build_single_gen(a_n, pi_arch, S, np.asarray(rho, float), 0.7)
    kids, _ = X._children(parent)
    lm = {}
    for v in range(n):
        if not kids[v]:
            e = np.eye(A)[xcol[v]] if xcol[v] >= 0 else np.ones(A)
            msg = np.zeros(2 * A)
            for th in range(2):
                msg[th * A:th * A + A] = e
            lm[v] = msg
    EN, dw, _ = X.tree_hr(parent, tau, lm, Q40, p40)
    Nk_d, Tk_d, J_d = X._agg40(EN, dw, a_n, pi_arch.shape[0], 2)
    print(f"  reldiff N^k={_reldiff(Nk_d, Nk_s):.2e}  T^k={_reldiff(Tk_d, Tk_s):.2e}"
          f"  |dJ|={abs(J_d - J_s):.2e}")


def check_real_family(pi_arch, S, rho):
    """Factored vs dense on a REAL family tree (a small one, so dense is
    affordable).  Uses the CLV bundle tree + real leaf residues for two columns."""
    print("\n=== real family tree: factored vs dense 800-state ===")
    import glob
    from tkfdp.pfam_data import load_clv_family
    cands = sorted(glob.glob("data/pfam_processed_clv_top1000_thin128/*.npz"))
    cands = [c for c in cands if "index" not in c]
    picked = None
    for c in cands:
        fc = load_clv_family(c)
        if fc.n_nodes <= 40 and fc.L >= 4:          # keep dense HR affordable
            picked = fc; break
    if picked is None:
        # fall back to the smallest available, capped
        fc = min((load_clv_family(c) for c in cands[:20]), key=lambda f: f.n_nodes)
        picked = fc
    fc = picked
    n = fc.n_nodes
    parent = np.asarray(fc.parent).astype(int).copy()
    parent[fc.root_id] = -1
    tau = np.asarray(fc.tau, float)
    # residue columns per node: leaves 0..n_leaves-1 observe leaf_msa; else -1
    i, j = 0, min(3, fc.L - 1)
    xcol = np.full(n, -1); ycol = np.full(n, -1)
    xcol[:fc.n_leaves] = fc.leaf_msa[:, i]
    ycol[:fc.n_leaves] = fc.leaf_msa[:, j]
    print(f"  family {fc.family}  n_nodes={n} n_leaves={fc.n_leaves} L={fc.L}  cols=({i},{j})")
    a1 = np.array([1, 2]); a2 = np.array([2, 1])
    for rho_chain in (0.0, 0.8):
        Nk_d, Tk_d, J_d, logZ = X.pair_tree_hr_dense(parent, tau, xcol, ycol,
                                                     a1, a2, pi_arch, S, rho, rho_chain)
        Nk_f, Tk_f, J_f = X.pair_tree_hr(parent, tau, xcol, ycol,
                                         a1, a2, pi_arch, S, rho, rho_chain)
        print(f"  rho_chain={rho_chain}: reldiff N^k={_reldiff(Nk_d, Nk_f):.2e}  "
              f"T^k={_reldiff(Tk_d, Tk_f):.2e}  |dJumps|={abs(J_d - J_f):.2e} "
              f"(jumps={J_d:.3f})")


def _scale_reld(a, b):
    """Relative diff floored to the ARRAY scale (not a fixed 1e-9), so tiny
    entries whose absolute error is machine-eps don't inflate the metric."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    floor = max(1e-12, 1e-9 * float(np.abs(a).max()))
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), floor)))


def check_numpy_vs_jax(pi_arch, S, rho):
    """Confirm the JAX factored pair HR (cluster_hr_jax.pair_tree_hr_jax) matches
    the numpy factored pair HR (cluster_hr_exact.pair_tree_hr) to ~1e-8 on the
    trainer-consumed quantities (dwell_total = T^k, real_counts = N^k summed over
    source), on every small-tree regime and the singleton (m=1) path."""
    from tkfdp.coupling.dynfield.phylo_elbo import cluster_hr_jax as XJ
    print("\n=== numpy vs JAX: factored pair HR (dwell_total / real_counts) ===")
    shared = XJ.make_shared(pi_arch, S, rho)
    trees = make_trees()
    worst_d = worst_r = worst_j = 0.0
    regimes = [("rc0 a!=b", np.array([1, 2]), np.array([2, 1]), 0.0),
               ("rc0 a==b", np.array([0, 1]), np.array([0, 1]), 0.0),
               ("rc0.4", np.array([1, 2]), np.array([2, 1]), 0.4),
               ("rc1.1", np.array([1, 2]), np.array([2, 1]), 1.1),
               ("rc0.7 a==b", np.array([0, 1]), np.array([0, 1]), 0.7)]
    for name, a1, a2, rc in regimes:
        wd = wr = wj = 0.0
        for tname, (parent, tau, leaves) in trees.items():
            xcol, ycol = _cols_from_leaves(parent, leaves)
            Nn, Tn, Jn = X.pair_tree_hr(parent, tau, xcol, ycol, a1, a2, pi_arch,
                                        S, rho, rc, want_jumps=True)
            Nj, Tj, Jj = XJ.pair_tree_hr_jax(parent, tau, xcol, ycol, a1, a2,
                                             pi_arch, S, rho, rc, shared=shared,
                                             want_jumps=True)
            wd = max(wd, _scale_reld(Tn, Tj))
            wr = max(wr, _scale_reld(Nn.sum(1), np.asarray(Nj).sum(1)))
            wj = max(wj, abs(Jn - Jj))
        print(f"  [{name:11s}] dwell reldiff={wd:.2e}  real_counts reldiff={wr:.2e}"
              f"  |dJumps|={wj:.2e}")
        worst_d = max(worst_d, wd); worst_r = max(worst_r, wr); worst_j = max(worst_j, wj)
    print(f"  WORST dwell={worst_d:.2e}  real_counts={worst_r:.2e}  |dJ|={worst_j:.2e}")
    return worst_d, worst_r, worst_j


def check_numpy_vs_jax_real(pi_arch, S, rho, max_nodes=999999):
    """numpy vs JAX on the REAL 254-node PF00013 family tree, rho_chain=0 and
    rho_chain>0 (field jumps active).  Dense 800-state HR is infeasible at 254
    nodes, so this is factored-numpy vs factored-JAX -- the integration bar."""
    from tkfdp.coupling.dynfield.phylo_elbo import cluster_hr_jax as XJ
    from tkfdp.pfam_data import load_clv_family
    print("\n=== numpy vs JAX: REAL PF00013 254-node tree (factored vs factored) ===")
    path = "data/pfam_processed_clv_top1000_thin128/PF00013.npz"
    if not os.path.exists(path):
        print("  [skip] PF00013 not found"); return 0.0, 0.0, 0.0
    fc = load_clv_family(path)
    n = fc.n_nodes
    if n > max_nodes:
        print(f"  [skip] {n} > max_nodes {max_nodes}"); return 0.0, 0.0, 0.0
    parent = np.asarray(fc.parent).astype(int).copy(); parent[int(fc.root_id)] = -1
    tau = np.asarray(fc.tau, float)
    i, j = 0, min(3, fc.L - 1)
    xcol = np.full(n, -1); ycol = np.full(n, -1)
    xcol[:fc.n_leaves] = fc.leaf_msa[:, i]; ycol[:fc.n_leaves] = fc.leaf_msa[:, j]
    a1 = np.array([1, 2]); a2 = np.array([2, 1])
    shared = XJ.make_shared(pi_arch, S, rho)
    wd = wr = wj = 0.0
    for rc in (0.0, 0.8):
        Nn, Tn, Jn = X.pair_tree_hr(parent, tau, xcol, ycol, a1, a2, pi_arch, S,
                                    rho, rc, want_jumps=True)
        Nj, Tj, Jj = XJ.pair_tree_hr_jax(parent, tau, xcol, ycol, a1, a2, pi_arch,
                                         S, rho, rc, shared=shared, want_jumps=True)
        d = _scale_reld(Tn, Tj); r = _scale_reld(Nn.sum(1), np.asarray(Nj).sum(1))
        jd = abs(Jn - Jj)
        print(f"  rho_chain={rc}: dwell reldiff={d:.2e}  real_counts reldiff={r:.2e}"
              f"  jumps np={Jn:.4f} jax={Jj:.4f} |dJ|={jd:.2e}")
        wd = max(wd, d); wr = max(wr, r); wj = max(wj, jd)
    return wd, wr, wj


def check_padding_fidelity(pi_arch, S, rho):
    """Bin-padding must be HR-neutral: the HR from a geomspaced-bin-PADDED batch
    (where trees of different sizes share a padded shape) must equal the per-tree
    UNPADDED HR to ~1e-8.  Padded nodes/branches must contribute the identity
    (zero dwell/counts/jumps).  Checked on (a) the real 254-node PF00013 tree
    (pad=True vs pad=False) and (b) a mixed-size batch that forces real padding."""
    from tkfdp.coupling.dynfield.phylo_elbo import cluster_hr_jax as XJ
    print("\n=== bin-padding fidelity: padded batch == per-tree unpadded ===")
    shared = XJ.make_shared(pi_arch, S, rho)
    worst = 0.0

    # (a) real 254-node PF00013: pad=True (sqrt2 bin) vs pad=False (minimal)
    import os
    path = "data/pfam_processed_clv_top1000_thin128/PF00013.npz"
    if os.path.exists(path):
        from tkfdp.pfam_data import load_clv_family
        fc = load_clv_family(path); n = fc.n_nodes
        parent = np.asarray(fc.parent).astype(int).copy(); parent[int(fc.root_id)] = -1
        tau = np.asarray(fc.tau, float)
        xcol = np.full(n, -1); ycol = np.full(n, -1)
        xcol[:fc.n_leaves] = fc.leaf_msa[:, 0]; ycol[:fc.n_leaves] = fc.leaf_msa[:, min(3, fc.L - 1)]
        a1 = np.array([1, 2]); a2 = np.array([2, 1])
        for rc in (0.0, 0.8):
            Np, Tp, Jp = XJ.pair_tree_hr_jax(parent, tau, xcol, ycol, a1, a2, pi_arch,
                                             S, rho, rc, shared=shared, want_jumps=True, pad=True)
            Nu, Tu, Ju = XJ.pair_tree_hr_jax(parent, tau, xcol, ycol, a1, a2, pi_arch,
                                             S, rho, rc, shared=shared, want_jumps=True, pad=False)
            d = _scale_reld(Tu, Tp); r = _scale_reld(Nu.sum(1), np.asarray(Np).sum(1))
            worst = max(worst, d, r)
            print(f"  PF00013 rc={rc}: dwell(padded-vs-unpadded)={d:.2e}  real={r:.2e}  |dJ|={abs(Jp-Ju):.2e}")

    # (b) mixed-size batch (cherry/3-leaf/4-leaf) forcing real per-level padding
    trees = make_trees()
    trees["cherry2"] = (np.array([-1, 0, 0, 1, 1, 2, 2]),
                        np.array([0.0, 0.5, 0.3, 0.4, 0.7, 0.2, 0.6]),
                        {3: (2, 8), 4: (3, 14), 5: (8, 3), 6: (14, 2)})
    ptrees = []; los = []; a1s = []; a2s = []; rcs = []; refs = []
    a1 = np.array([1, 2]); a2 = np.array([2, 1])
    for name, (parent, tau, leaves) in trees.items():
        xcol, ycol = _cols_from_leaves(parent, leaves)
        for rc in (0.7, 0.0):
            pt = XJ.build_ptree(parent, tau, pad=True)
            ptrees.append(pt); los.append(XJ.pt_leaf_obs_pair(pt, xcol, ycol))
            a1s.append(a1); a2s.append(a2); rcs.append(rc)
            refs.append(X.pair_tree_hr(parent, tau, xcol, ycol, a1, a2, pi_arch, S,
                                       rho, rc, want_jumps=True))
    buckets = sorted(set(pt["bucket"] for pt in ptrees))
    Nk, Tk, J = XJ.pair_tree_hr_bucketed(ptrees, los, a1s, a2s, np.asarray(rcs),
                                         shared, want_jumps=True)
    wd = wr = 0.0
    for i, (Nn, Tn, Jn) in enumerate(refs):
        wd = max(wd, _scale_reld(Tn, Tk[i])); wr = max(wr, _scale_reld(Nn.sum(1), np.asarray(Nk[i]).sum(1)))
    worst = max(worst, wd, wr)
    print(f"  mixed-size batch ({len(buckets)} bins {buckets}): dwell={wd:.2e}  real={wr:.2e}")
    print(f"  WORST padding fidelity={worst:.2e}")
    return worst


def check_numpy_vs_jax_singleton(pi_arch, S, rho):
    """Singleton (m=1) HR: JAX single_tree_hr_jax vs numpy single_tree_hr (which
    is itself validated == the dense 40-state tree HR by `check_singleton`).
    Also cross-checks the JAX m=1 directly against the dense 40-state HR."""
    from tkfdp.coupling.dynfield.phylo_elbo import cluster_hr_jax as XJ
    print("\n=== numpy vs JAX + dense-40: SINGLETON (m=1) HR ===")
    shared = XJ.make_shared(pi_arch, S, rho)
    parent = np.array([-1, 0, 0, 1, 1]); tau = np.array([0.0, 0.4, 0.9, 0.3, 0.6])
    xleaf = {2: 8, 3: 2, 4: 3}
    n = len(parent); xcol = np.full(n, -1)
    for v, x in xleaf.items():
        xcol[v] = x
    worst = 0.0
    for a_n in (np.array([1, 2]), np.array([0, 1])):
        for rc in (0.0, 0.7):
            Nn, Tn, Jn = X.single_tree_hr(parent, tau, xcol, a_n, pi_arch, S, rho, rc)
            Nj, Tj, Jj = XJ.single_tree_hr_jax(parent, tau, xcol, a_n, pi_arch, S,
                                               rho, rc, shared=shared, want_jumps=True)
            # dense 40-state reference
            Q40, p40 = X.build_single_gen(a_n, pi_arch, S, np.asarray(rho, float), rc)
            kids, _ = X._children(parent)
            lm = {}
            for v in range(n):
                if not kids[v]:
                    e = np.eye(A)[xcol[v]] if xcol[v] >= 0 else np.ones(A)
                    msg = np.zeros(2 * A)
                    for th in range(2):
                        msg[th * A:th * A + A] = e
                    lm[v] = msg
            EN, dw, _ = X.tree_hr(parent, tau, lm, Q40, p40)
            Nd, Td, Jd = X._agg40(EN, dw, a_n, pi_arch.shape[0], 2)
            dr = _scale_reld(Td, Tj); dn = _scale_reld(Nd.sum(1), np.asarray(Nj).sum(1))
            dj = abs(Jd - Jj)
            worst = max(worst, dr, dn)
            print(f"  a_n={a_n.tolist()} rc={rc}: dwell(jax-vs-dense40)={dr:.2e}  "
                  f"real_counts={dn:.2e}  |dJ|={dj:.2e}")
    print(f"  WORST (jax m=1 vs dense-40)={worst:.2e}")
    return worst


def main():
    S, pi_arch = make_archetypes()
    rho = np.array([0.55, 0.45])
    trees = make_trees()
    W = []
    W.append(run_regime("rho_chain=0 (no jumps), a1!=a2", np.array([1, 2]),
                         np.array([2, 1]), pi_arch, S, rho, 0.0, trees))
    W.append(run_regime("rho_chain=0, a1==a2", np.array([0, 1]),
                         np.array([0, 1]), pi_arch, S, rho, 0.0, trees))
    W.append(run_regime("rho_chain=0.4 (jumps), a1!=a2", np.array([1, 2]),
                         np.array([2, 1]), pi_arch, S, rho, 0.4, trees))
    W.append(run_regime("rho_chain=1.1 (jumps), a1!=a2", np.array([1, 2]),
                         np.array([2, 1]), pi_arch, S, rho, 1.1, trees))
    W.append(run_regime("rho_chain=0.7 (jumps), a1==a2", np.array([0, 1]),
                         np.array([0, 1]), pi_arch, S, rho, 0.7, trees))
    check_singleton(pi_arch, S, rho)
    check_real_family(pi_arch, S, rho)
    wN = max(w[0] for w in W); wT = max(w[1] for w in W); wJ = max(w[2] for w in W)
    print(f"\nWORST over all small-tree regimes: N^k={wN:.2e}  T^k={wT:.2e}  |dJ|={wJ:.2e}")
    ok = wN < 1e-8 and wT < 1e-8 and wJ < 1e-8
    print("RESULT (dense vs factored):", "PASS (< 1e-8)" if ok else "FAIL")

    # numpy-vs-JAX (the fast batched port must reproduce the numpy factored HR)
    jd, jr, jj = check_numpy_vs_jax(pi_arch, S, rho)
    js = check_numpy_vs_jax_singleton(pi_arch, S, rho)
    pf = check_padding_fidelity(pi_arch, S, rho)
    big = os.environ.get("HR_JAX_REAL", "1") != "0"
    if big:
        rd, rr, rj = check_numpy_vs_jax_real(pi_arch, S, rho)
    else:
        print("\n[skip real 254-node numpy-vs-jax: HR_JAX_REAL=0]")
        rd = rr = rj = 0.0
    okj = max(jd, jr, rd, rr, js, pf) < 1e-8 and max(jj, rj) < 1e-6
    print("RESULT (numpy vs JAX + padding):",
          "PASS (dwell/real_counts/padding < 1e-8)" if okj else "FAIL")
    return 0 if (ok and okj) else 1


if __name__ == "__main__":
    sys.exit(main())
