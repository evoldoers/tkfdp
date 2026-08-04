#!/usr/bin/env python3
r"""Does the Composite Potts phylo-ELBO recover the Potts energies from data
simulated under a Metropolis--Potts model on a phylogeny?  And how much tree
(depth vs node/leaf count vs number of coupled pairs) does it take?

DESIGN (all reuses the committed, self-checked fitter
``experiments/composite_potts_phylo_elbo.py``):

  ground truth  (pi_i, pi_j, H_true, S_true)  [make_params, fixed seed]
      -> M-H pair generator Q_true, pair stationary pi_ij_true
      -> simulate n_pairs INDEPENDENT residue-pairs down a fixed tree
         (root ~ pi_ij_true, evolve by exp(Q_true tau)); observe LEAVES only.
  fit           freeze pi_i, pi_j at TRUTH; init H = 0 (INDEPENDENT sites, so the
                coupling must be discovered from data, no truth leakage); init S
                at truth (S is not the object of interest).  Pool the E-step
                sufficient stats across the iid pairs and run ONE shared-parameter
                EM (exact S-step + backtracked pi_ij/H ascent) to convergence.
                Read H_fit = log pi_i + log pi_j - log pi_ij_fit.

RECOVERY METRIC -- the GAUGE-INVARIANT Potts interaction.  H is identified only
up to the Potts gauge H(a,b) -> H(a,b)+f(a)+g(b) (absorbable into pi_i, pi_j), so
we compare the DOUBLE-CENTERED coupling
    interaction(H) = H - rowmean - colmean + grandmean         [(A-1)^2 = 361 dof]
Note interaction(H) = -interaction(log pi_ij) exactly (the marginal logs vanish
under double-centering), so this is what any (S, pi_ij) fit can hope to recover.
Reported: Pearson corr, equilibrium-weighted corr (weights pi_ij_true; "where the
coupling matters for typical data"), relative RMSE, and corr on the common-residue
submatrix (top-12 x top-12 by marginal; the rare-residue cells are the hard tail).

CONTROL / CEILING -- the model-free "iid equilibrium samples" estimator: pool ALL
leaf pair-states into a 400-cell histogram (Laplace-smoothed), double-center its
log.  If the tree is deep (leaves ~ independent equilibrium draws) the ELBO fit and
this empirical estimator should agree; the ELBO's job is to add back the tree
correlation + transition information the naive histogram ignores.

Usage:  python3 experiments/composite_recovery_study.py --all   --out results/composite_recovery
        python3 experiments/composite_recovery_study.py --smoke
"""
import os, sys, json, time, argparse
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import jax, jax.numpy as jnp
from scipy.linalg import expm as sexpm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_potts_phylo_elbo import (
    make_params, mh_pair_generator, binary_tree, pad_tree,
    _estep_batch, _forward_batch, mstep_S, mstep_pi_ij, sfull_from_S,
)

# ---------------------------------------------------------------------------
# trees (fixed equal branch length b so a single propagator serves every edge)
# ---------------------------------------------------------------------------
def star_tree(nl, b):
    """nl leaves all attached to a single root; every edge length b."""
    parent = -np.ones(nl + 1, int)
    parent[:nl] = nl                      # leaves 0..nl-1 -> root=nl
    tau = np.full(nl + 1, float(b)); tau[nl] = 0.0
    return parent, tau

def balanced_tree(nl, b):
    """balanced binary tree (nl a power of 2), every edge length b."""
    parent = binary_tree(nl)
    tau = np.full(len(parent), float(b)); tau[np.where(parent < 0)[0][0]] = 0.0
    return parent, tau

def root_to_leaf_subs(parent, tau, Q, pi_ij):
    """expected substitutions along a root->leaf path = (avg jump rate) * path len."""
    avg_rate = float(-np.sum(np.asarray(pi_ij) * np.diag(np.asarray(Q))))
    root = int(np.where(parent < 0)[0][0])
    # depth in edges from root to a leaf (all leaves equidistant for our trees)
    ch = {}
    for v in range(len(parent)):
        ch.setdefault(parent[v], []).append(v)
    depth_edges, v = 0, root
    while v in ch and ch[v]:
        c = [x for x in ch[v] if x != root]
        if not c:
            break
        v = c[0]; depth_edges += 1
    return avg_rate * float(tau[0]) * depth_edges, depth_edges

# ---------------------------------------------------------------------------
# fast vectorised simulator (one propagator, all pairs at once)
# ---------------------------------------------------------------------------
def sim_leaves(parent, b, Q, pi_ij, n_pairs, seed):
    Q = np.asarray(Q, float); pi_ij = np.asarray(pi_ij, float)
    NS = Q.shape[0]; N = len(parent)
    root = int(np.where(parent < 0)[0][0])
    P = sexpm(Q * b)                                     # single edge propagator
    cP = np.cumsum(np.clip(P, 0, None), axis=1); cP /= cP[:, -1:]
    ch = [[] for _ in range(N)]
    for v in range(N):
        if parent[v] >= 0:
            ch[parent[v]].append(v)
    order, stack = [], [root]
    while stack:
        v = stack.pop(); order.append(v); stack.extend(ch[v])
    rng = np.random.default_rng(seed)
    states = np.zeros((N, n_pairs), int)
    p0 = pi_ij / pi_ij.sum()
    states[root] = rng.choice(NS, size=n_pairs, p=p0)
    for v in order:
        if parent[v] < 0:
            continue
        rows = cP[states[parent[v]]]                     # (n_pairs, NS)
        u = rng.random(n_pairs)
        states[v] = np.clip((rows < u[:, None]).sum(1), 0, NS - 1)
    leaves = np.array([v for v in range(N) if len(ch[v]) == 0])
    return states[leaves].T                              # (n_pairs, n_leaves)

# ---------------------------------------------------------------------------
# recovery metric (gauge-invariant double-centred interaction)
# ---------------------------------------------------------------------------
def double_center(M):
    M = np.asarray(M, float)
    return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()

def _pearson(x, y):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def _wcorr(x, y, w):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel(); w = np.asarray(w).ravel()
    w = w / w.sum()
    mx = (w * x).sum(); my = (w * y).sum()
    cxy = (w * (x - mx) * (y - my)).sum()
    vx = (w * (x - mx) ** 2).sum(); vy = (w * (y - my) ** 2).sum()
    if vx < 1e-18 or vy < 1e-18:
        return float("nan")
    return float(cxy / np.sqrt(vx * vy))

def metrics(H_true, H_fit, pi_i, pi_j, pi_ij_true, A):
    it = double_center(H_true); ie = double_center(H_fit)
    w = np.asarray(pi_ij_true, float).reshape(A, A)
    ti = np.argsort(pi_i)[-12:]; tj = np.argsort(pi_j)[-12:]
    sub = np.ix_(ti, tj)
    return dict(
        corr=_pearson(it, ie),
        wcorr=_wcorr(it, ie, w),
        rel_rmse=float(np.sqrt(np.mean((it - ie) ** 2)) / (it.std() + 1e-12)),
        sub_corr=_pearson(it[sub], ie[sub]),
    )

def empirical_pij(LEAF, A):
    """Laplace-smoothed empirical leaf-pair histogram as a normalised NS-vector."""
    NS = A * A
    counts = np.bincount(np.asarray(LEAF).ravel(), minlength=NS).astype(float) + 0.5
    return counts / counts.sum()

def empirical_interaction(LEAF, A):
    """Model-free ceiling: double-centred log of the Laplace-smoothed leaf histogram."""
    p = empirical_pij(LEAF, A).reshape(A, A)
    return -double_center(np.log(p))    # interaction(log pi) = -interaction(H)

def int_metrics(it_true, it_est, pi_i, pi_j):
    """Pearson corr on all 400 cells and on the common top-12x12 submatrix."""
    ti = np.argsort(pi_i)[-12:]; tj = np.argsort(pi_j)[-12:]; sub = np.ix_(ti, tj)
    return _pearson(it_true, it_est), _pearson(it_true[sub], it_est[sub])

def hist_recovery(parent, b, gt, n_pairs, seed):
    """Fast, EM-free: simulate leaves, score the empirical-histogram interaction estimate
    (= the equilibrium sufficient statistic; the interaction MLE tracks this)."""
    pi_i, pi_j, H_true, S_true = gt; A = len(pi_i)
    Q_t, pij_t = mh_pair_generator(jnp.asarray(pi_i), jnp.asarray(pi_j),
                                   jnp.asarray(H_true), jnp.asarray(S_true))
    Q_t = np.asarray(Q_t); pij_t = np.asarray(pij_t)
    LEAF = sim_leaves(parent, b, Q_t, pij_t, n_pairs, seed)
    nl = LEAF.shape[1]
    subs, _ = root_to_leaf_subs(parent, tau_of(parent, b), Q_t, pij_t)
    c, cs = int_metrics(double_center(H_true), empirical_interaction(LEAF, A), pi_i, pi_j)
    return dict(N=int(n_pairs * nl), n_pairs=int(n_pairs), n_leaves=int(nl),
                branch_b=float(b), subs_root_to_leaf=float(subs),
                corr=round(c, 4), sub_corr=round(cs, 4))

def tau_of(parent, b):
    tau = np.full(len(parent), float(b)); tau[np.where(parent < 0)[0][0]] = 0.0
    return tau

# ---------------------------------------------------------------------------
# shared-parameter EM fit (freeze pi_i, pi_j at truth; discover H from H=0)
# ---------------------------------------------------------------------------
def fit_shared(pi_i, pi_j, S_true, LEAF_pad, tree, n_iter, k_inner=25, tol=2e-3,
               pij_init=None, freeze_S=False, H_true=None):
    """Shared-parameter EM to the CONVERGED composite MLE.  pi_i, pi_j frozen at truth;
    S init at truth (S is not the object of interest).  pi_ij initialised at pij_init --
    the method-of-moments empirical leaf histogram (DATA, not truth -> no leakage of the
    coupling), which lands EM in the right basin so it converges in a few outer iters
    rather than crawling from independence.  Each EM iter: E-step (expensive), then run
    the pi_ij/H M-step to completion (cheap inner loop) so the LL is not plateau-stopped
    before parameters converge.  Early-stop on the gauge-invariant interaction change."""
    A = len(pi_i); n_pairs = LEAF_pad.shape[0]
    PI_i = jnp.tile(jnp.asarray(pi_i), (n_pairs, 1))
    PI_j = jnp.tile(jnp.asarray(pi_j), (n_pairs, 1))
    LEAF = jnp.asarray(LEAF_pad)
    S = jnp.asarray(S_true)
    lpi_i = jnp.log(jnp.asarray(pi_i)); lpi_j = jnp.log(jnp.asarray(pi_j))
    if pij_init is None:
        H = jnp.zeros((A, A))                                  # independent-sites init
    else:
        logp0 = jnp.log(jnp.clip(jnp.asarray(pij_init).reshape(A, A), 1e-300, None))
        H = lpi_i[:, None] + lpi_j[None, :] - logp0            # MoM init from histogram
        H = H - H.mean(); H = 0.5 * (H + H.T)
    lls = []; prev_int = double_center(np.asarray(H)); ctraj = []
    it_true = None if H_true is None else double_center(np.asarray(H_true))
    for it in range(n_iter):
        Hb = jnp.broadcast_to(H, (n_pairs, A, A))
        ll, N_all, T_all, RP_all = _estep_batch(PI_i, PI_j, Hb, S, tree, LEAF)
        Ntot = N_all.sum(0); Ttot = T_all.sum(0); RPtot = RP_all.sum(0)
        _, pij = mh_pair_generator(jnp.asarray(pi_i), jnp.asarray(pi_j), H, S)
        if not freeze_S:
            S = mstep_S(Ntot, Ttot, pij, A)                      # exact S-step
        Sfull = sfull_from_S(S)
        for _ in range(k_inner):                                 # run pi_ij M-step to completion
            pij = mstep_pi_ij(Ntot, Ttot, RPtot, pij, Sfull)
        logp = jnp.log(jnp.clip(pij.reshape(A, A), 1e-300, None))
        H = lpi_i[:, None] + lpi_j[None, :] - logp
        H = H - H.mean(); H = 0.5 * (H + H.T)
        lls.append(float(ll.sum()))
        cur_int = double_center(np.asarray(H))
        if it_true is not None:
            ctraj.append(round(_pearson(it_true, cur_int), 3))
        dchg = np.sqrt(np.mean((cur_int - prev_int) ** 2)) / (cur_int.std() + 1e-12)
        prev_int = cur_int
        if it > 10 and dchg < tol:
            break
    if H_true is not None:
        print(f"    corr traj: {ctraj}", flush=True)
    return np.asarray(H), lls

# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------
def run_config(name, parent, tau, n_pairs, n_iter, gt, sim_seed, log=True, freeze_S=True):
    pi_i, pi_j, H_true, S_true = gt
    A = len(pi_i)
    Q_t, pij_t = mh_pair_generator(jnp.asarray(pi_i), jnp.asarray(pi_j),
                                   jnp.asarray(H_true), jnp.asarray(S_true))
    Q_t = np.asarray(Q_t); pij_t = np.asarray(pij_t)
    b = float(tau[0])
    LEAF = sim_leaves(parent, b, Q_t, pij_t, n_pairs, sim_seed)      # (n_pairs, nl)
    nl = LEAF.shape[1]
    MAX_NODES = len(parent) + 2
    tree, _, nl_t = pad_tree(parent, tau, MAX_NODES)
    LEAF_pad = np.zeros((n_pairs, nl_t), int); LEAF_pad[:, :nl] = LEAF
    subs, depth = root_to_leaf_subs(parent, tau, Q_t, pij_t)
    # empirical-histogram ceiling + method-of-moments EM init (data, not truth)
    pij_emp = empirical_pij(LEAF, A)
    ie_emp = empirical_interaction(LEAF, A)
    m_emp = _pearson(double_center(H_true), ie_emp)
    t0 = time.time()
    H_fit, lls = fit_shared(pi_i, pi_j, S_true, LEAF_pad, tree, n_iter, pij_init=pij_emp,
                            freeze_S=freeze_S)
    m = metrics(H_true, H_fit, pi_i, pi_j, pij_t, A)
    rec = dict(name=name, n_pairs=int(n_pairs), n_leaves=int(nl),
               n_samples=int(n_pairs * nl), branch_b=b,
               subs_root_to_leaf=float(subs), depth_edges=int(depth),
               n_iter_run=len(lls), fit_s=round(time.time() - t0, 1),
               ll_final=lls[-1], **{k: round(v, 4) for k, v in m.items()},
               corr_emp=round(m_emp, 4))
    if log:
        print(f"  {name:26s} np={n_pairs:4d} nl={nl:4d} N={n_pairs*nl:6d} "
              f"b={b:4.2f} subs={subs:5.2f} | corr={m['corr']:.3f} "
              f"wcorr={m['wcorr']:.3f} sub={m['sub_corr']:.3f} "
              f"emp={m_emp:.3f} relRMSE={m['rel_rmse']:.3f} "
              f"({rec['fit_s']}s,{len(lls)}it)", flush=True)
    return rec

# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--A", type=int, default=20)
    ap.add_argument("--gt-seed", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--out", default="results/composite_recovery")
    args = ap.parse_args()

    A = args.A
    gt = make_params(A, args.gt_seed, exchangeable=False)   # asymmetric marginals
    pi_i, pi_j, H_true, S_true = gt
    print(f"# ground truth: A={A} seed={args.gt_seed}  "
          f"std(interaction H)={double_center(H_true).std():.3f}  "
          f"min pi_i*pi_j={float(np.outer(pi_i,pi_j).min()):.2e}", flush=True)

    if args.diag:
        # deep, well-mixed star: is the EM degradation the S-confound or a bug?
        p, t = star_tree(32, 2.0)
        Q_t, pij_t = mh_pair_generator(jnp.asarray(pi_i), jnp.asarray(pi_j),
                                       jnp.asarray(H_true), jnp.asarray(S_true))
        LEAF = sim_leaves(p, 2.0, np.asarray(Q_t), np.asarray(pij_t), 32, 11)
        MAX = len(p) + 2; tree, _, nlt = pad_tree(p, t, MAX)
        LP = np.zeros((32, nlt), int); LP[:, :LEAF.shape[1]] = LEAF
        pij_emp = empirical_pij(LEAF, A)
        emp = _pearson(double_center(H_true), empirical_interaction(LEAF, A))
        print(f"# DIAG deep star nl=32 np=32 N=1024 b=2.0  emp(histogram)={emp:.3f}")
        print("  [S FROZEN at truth, fit pi_ij only]")
        Hf, _ = fit_shared(pi_i, pi_j, S_true, LP, tree, 25, pij_init=pij_emp,
                           freeze_S=True, H_true=H_true)
        print(f"    -> corr={metrics(H_true,Hf,pi_i,pi_j,pij_t,A)['corr']:.3f}")
        print("  [S FITTED jointly]")
        Hf, _ = fit_shared(pi_i, pi_j, S_true, LP, tree, 25, pij_init=pij_emp,
                           freeze_S=False, H_true=H_true)
        print(f"    -> corr={metrics(H_true,Hf,pi_i,pi_j,pij_t,A)['corr']:.3f}")
        return

    if args.smoke:
        # (1) METRIC + SIMULATOR validation: huge iid-equilibrium sample -> corr ~ 1.
        p, t = star_tree(128, 5.0)
        Q_t, pij_t = mh_pair_generator(jnp.asarray(pi_i), jnp.asarray(pi_j),
                                       jnp.asarray(H_true), jnp.asarray(S_true))
        for npair in [64, 512]:
            LEAF = sim_leaves(p, 5.0, np.asarray(Q_t), np.asarray(pij_t), npair, 7)
            c = _pearson(double_center(H_true), empirical_interaction(LEAF, A))
            print(f"  [metric-check] N={npair*128:6d} iid-eq samples -> emp corr={c:.3f}")
        # (2) ELBO should now MATCH the histogram on near-iid leaves (deep ceiling)...
        p, t = star_tree(48, 5.0)
        run_config("smoke-ceiling", p, t, n_pairs=48, n_iter=25, gt=gt, sim_seed=2)
        # ...and hold up on a SHALLOW/correlated tree (leaves share ancestry).
        p, t = star_tree(32, 0.3)
        run_config("smoke-shallow", p, t, n_pairs=32, n_iter=25, gt=gt, sim_seed=3)
        return

    ni = args.n_iter
    NL = 32                                            # leaves per tree for the N-sweep

    # === (1) HISTOGRAM recovery vs N_eff (independent equilibrium samples) ===
    # The equilibrium sufficient statistic; the interaction MLE tracks it.  Fast, no EM.
    # Well-mixed star (b=3 => leaves ~ iid pi_ij), scale N by n_pairs; 4 seeds each.
    print("\n# === (1) histogram recovery vs N_eff (well-mixed, 4 seeds) ===")
    hist_N = []
    Ngrid = [(8, NL), (16, NL), (32, NL), (64, NL), (128, NL), (256, NL), (512, NL), (1024, NL)]
    for npair, nl in Ngrid:
        p = star_tree(nl, 3.0)[0]
        rr = [hist_recovery(p, 3.0, gt, npair, seed=500 + npair + 1000 * s) for s in range(4)]
        agg = dict(N=rr[0]["N"], n_pairs=npair, n_leaves=nl,
                   corr=float(np.mean([r["corr"] for r in rr])),
                   corr_sd=float(np.std([r["corr"] for r in rr])),
                   sub_corr=float(np.mean([r["sub_corr"] for r in rr])),
                   sub_corr_sd=float(np.std([r["sub_corr"] for r in rr])))
        hist_N.append(agg)
        print(f"  N={agg['N']:6d}  full corr={agg['corr']:.3f}+-{agg['corr_sd']:.3f}  "
              f"common-12x12={agg['sub_corr']:.3f}+-{agg['sub_corr_sd']:.3f}", flush=True)

    # shape-invariance at fixed N=2048: same N, different (leaves, pairs) -> same corr
    print("# -- shape invariance at N=2048 (well-mixed) --")
    hist_shape = []
    for nl, npair in [(16, 128), (32, 64), (64, 32), (128, 16), (256, 8)]:
        p = star_tree(nl, 3.0)[0]
        r = hist_recovery(p, 3.0, gt, npair, seed=777)
        hist_shape.append(r)
        print(f"  nl={nl:4d} np={npair:4d} N={r['N']}  full={r['corr']:.3f} common={r['sub_corr']:.3f}", flush=True)

    # === (2) DEPTH / mixing sweep: fixed N=2048, vary branch length ===
    print("\n# === (2) recovery vs mixing depth (fixed N=2048, star nl=64) ===")
    hist_depth = []
    for b in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]:
        p = star_tree(64, b)[0]
        rr = [hist_recovery(p, b, gt, 32, seed=900 + int(100 * b) + 7 * s) for s in range(3)]
        agg = dict(branch_b=b, subs_root_to_leaf=rr[0]["subs_root_to_leaf"], N=rr[0]["N"],
                   corr=float(np.mean([r["corr"] for r in rr])),
                   sub_corr=float(np.mean([r["sub_corr"] for r in rr])))
        hist_depth.append(agg)
        print(f"  b={b:4.2f} subs={agg['subs_root_to_leaf']:6.2f}  full={agg['corr']:.3f} "
              f"common={agg['sub_corr']:.3f}", flush=True)

    # === (3) COMPOSITE ELBO (S fixed) confirmation points -- does the fitted model
    #         match/beat the histogram ceiling? plus one shallow (de-correlation) case ===
    print("\n# === (3) composite-ELBO (S fixed at truth) vs histogram ===")
    elbo = []
    for npair in [8, 32, 128]:                                   # N = 256, 1024, 4096
        p, t = star_tree(NL, 3.0)
        elbo.append(run_config(f"elbo/N{npair*NL}", p, t, npair, ni, gt,
                               sim_seed=1500 + npair, freeze_S=True))
    # shallow/correlated: histogram is fooled by shared ancestry; ELBO de-correlates
    p, t = star_tree(64, 0.15)
    elbo.append(run_config("elbo/shallow", p, t, 32, ni, gt, sim_seed=1600, freeze_S=True))

    # === (4) S-nuisance caveat: co-fitting the shared exchangeability degrades recovery ===
    print("\n# === (4) S-nuisance caveat (fit pi_ij; S frozen vs S co-fit) ===")
    p, t = star_tree(NL, 3.0)
    s_frozen = run_config("caveat/S_frozen", p, t, 64, ni, gt, sim_seed=1700, freeze_S=True)
    s_cofit = run_config("caveat/S_cofit", p, t, 64, ni, gt, sim_seed=1700, freeze_S=False)

    data = dict(A=A, gt_seed=args.gt_seed,
                interaction_std=float(double_center(H_true).std()),
                min_pi_prod=float(np.outer(pi_i, pi_j).min()),
                hist_N=hist_N, hist_shape=hist_shape, hist_depth=hist_depth,
                elbo=elbo, caveat=dict(S_frozen=s_frozen, S_cofit=s_cofit))
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/recovery.json", "w") as f:
        json.dump(data, f, indent=1, default=float)
    try:
        make_figure(data, f"{args.out}/recovery.png")
        print(f"# figure -> {args.out}/recovery.png", flush=True)
    except Exception as e:
        print(f"# (figure skipped: {e})", flush=True)
    print(f"\n# saved -> {args.out}/recovery.json", flush=True)


def make_figure(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    hn = data["hist_N"]
    N = [r["N"] for r in hn]
    ax1.errorbar(N, [r["corr"] for r in hn], yerr=[r["corr_sd"] for r in hn],
                 marker="o", label="full 20x20 (400 cells)", capsize=3)
    ax1.errorbar(N, [r["sub_corr"] for r in hn], yerr=[r["sub_corr_sd"] for r in hn],
                 marker="s", label="common 12x12 block", capsize=3)
    # ELBO (S fixed) points
    ep = [(r["n_samples"], r["corr"], r["sub_corr"]) for r in data["elbo"] if "N" in r["name"]]
    if ep:
        ax1.scatter([e[0] for e in ep], [e[1] for e in ep], marker="*", s=140,
                    color="C0", edgecolor="k", zorder=5, label="composite ELBO (S fixed), full")
    ax1.axhline(0.9, ls="--", color="gray", lw=1)
    ax1.set_xscale("log"); ax1.set_xlabel("N_eff  (independent equilibrium residue-pairs)")
    ax1.set_ylabel("recovery of double-centred H  (Pearson corr)")
    ax1.set_title("Potts-energy recovery vs sample count"); ax1.set_ylim(0, 1.02)
    ax1.legend(fontsize=8, loc="lower right"); ax1.grid(alpha=0.3)

    hd = data["hist_depth"]
    ax2.plot([r["subs_root_to_leaf"] for r in hd], [r["corr"] for r in hd], "o-",
             label="full 20x20")
    ax2.plot([r["subs_root_to_leaf"] for r in hd], [r["sub_corr"] for r in hd], "s-",
             label="common 12x12")
    ax2.set_xscale("log"); ax2.set_xlabel("expected substitutions along root->leaf path")
    ax2.set_ylabel("recovery (Pearson corr)")
    ax2.set_title("vs tree depth  (fixed N=2048)"); ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=8, loc="lower right"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)

if __name__ == "__main__":
    main()
