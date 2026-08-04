#!/usr/bin/env python3
r"""ASYMMETRIC (non-exchangeable) swap-pair mixture of coupling components.

Companion to fit_coupling_mixture_freeS.py.  That fitter's components each carry a
SYMMETRIC joint stationary pi_c over the 210 UNORDERED amino-acid pairs (209 free
params) -- an exchangeable component (pi_c(a,b)=pi_c(b,a)).  Here we allow ASYMMETRIC
(directional) components: pi_c a full distribution over the 400 ORDERED pair-states
(399 free params).  Metropolis-sqrt with a symmetric shared single-site S is reversible
w.r.t. ANY target pi (symmetric or not), so an asymmetric component is a well-defined
reversible pair chain -- the SAME generator construction (FP._met_Q(S, pi, "sqrt")),
just with an asymmetric pi.

To keep the OVERALL MIXTURE exchangeable (invariant under swapping the two contact
sites -- state relabelling sigma:(a,b)->(b,a)), each asymmetric component enters as a
SWAP-PAIR: the pair {pi_c, pi_c^swap}, pi_c^swap(a,b)=pi_c(b,a), TIED to a single free
pi_c, EQUAL weight W_c/2 each, appearing as TWO mixture components.  Because
Q_{swap}[x,y] = met_sqrt(S, pi_c^swap)[x,y] = met_sqrt(S, pi_c)[sigma x, sigma y]
(shared symmetric S; single-site transitions map comp1<->comp2 under sigma), scoring
the swap component on a cluster equals scoring pi_c's generator on the site-swapped
counts -- so the swap component needs NO separate generator/eigendecomposition: it is
the same grid read at swapped state indices.  The component set is closed under sigma
with equal weights, so the mixture likelihood is EXACTLY sigma-invariant.

Parameter / component accounting (matched budget):
  * 1 asymmetric swap-pair  = 2 components, tied, 399 free pi params.
  * 2 free symmetric comps  = 2 components,       2*209 = 418 free pi params.
So P swap-pairs (2P components, 399P params) is matched in BOTH component count and
(to within 19P params, asym being the LEANER side) to 2P free symmetric components.
The clean question, at matched budget on HELD-OUT data: does the coupling prefer FEWER
DIRECTIONAL archetypes (asymmetric swap-pairs) or MORE SYMMETRIC ones?

Reversible ASYMMETRIC M-step.  The symmetric pi-step (FP.mstep_pi_metropolis) does
damped gradient ascent on log pi of the complete-data LL, then symmetrises
b=0.5(b+b.T).  Dropping that symmetrisation gives the unrestricted reversible-chain /
Holmes-Rubin M-step over all 400 states, which naturally yields an asymmetric pi_c
(the symmetric case is the restriction pi(a,b)=pi(b,a)).  Per swap-pair the
responsibility-weighted evidence is FOLDED into pi_c's frame:
    Ncounts_c = sum_g r_{g,(c,+)} n_g  +  sum_g r_{g,(c,-)} sigma(n_g),
i.e. the '+' component contributes n_g, the '-' component contributes the site-swapped
counts (equivalently: the '-' responsibility accumulated at swapped state indices).
Then HR estep under Q_c=met_sqrt(S,pi_c) and the asymmetric pi-step.  Shared free S is
pooled over all pairs exactly as in the symmetric fitter.

Corpus: data/per_contact_trrosetta/counts.npz.  ORDERING: the corpus stores each
contact (colA,colB) with colA<colB (build_per_contact_corpus.py; greedy matching over
np.triu_indices), and pf=(res@colA)*20+(res@colB), pt likewise -- a CONSISTENT
sequence-order (lower column index = site 1) directed pair-state, NOT symmetrised /
folded.  All detected asymmetry is w.r.t. that sequence ordering.

The symmetric matched baseline is produced by RUNNING fit_coupling_mixture_freeS.py
UNCHANGED (subprocess) at K=2P -- identical corpus / family split / EM schedule."""
from __future__ import annotations
import os
# pure numpy/scipy; the FS import pulls JAX via composite_potts_phylo_elbo -> force CPU
# so parallel invocations do not race for the GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse, json, time, subprocess, sys
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import fit_pair_models as FP
from fit_pair_models import NA, NS
import fit_coupling_mixture_freeS as FS

S_LG = FS.S_LG
# sigma: site-swap permutation of the 400 ordered states, x=i*NA+j -> j*NA+i (transpose)
SWAP_IDX = (np.arange(NS) % NA) * NA + (np.arange(NS) // NA)


def swap_pi(pi):
    """pi_c^swap(a,b) = pi_c(b,a) -- reindex by sigma (= transpose in NA x NA)."""
    return np.asarray(pi)[SWAP_IDX]


def sym_kl(p):
    """Symmetrised KL J(pi_c || pi_c^swap) = 0.5[KL(p||p^sw)+KL(p^sw||p)] (nats).
    0 iff pi_c is symmetric (exchangeable); grows with directional asymmetry."""
    p = np.clip(np.asarray(p, float), 1e-300, None); p = p / p.sum()
    ps = swap_pi(p)
    return float(0.5 * (np.sum(p * np.log(p / ps)) + np.sum(ps * np.log(ps / p))))


def asym_tv(p):
    """Total variation 0.5*sum|pi_c - pi_c^swap| between the component and its swap."""
    p = np.asarray(p, float); p = p / p.sum()
    return float(0.5 * np.abs(p - swap_pi(p)).sum())


# ---------- reversible ASYMMETRIC M-step (400 free states, no symmetrisation) ----------
def mstep_pi_metropolis_asym(S, pi, N, T, steps=60, lr=0.5):
    """Unrestricted reversible-chain M-step for the Metropolis-sqrt pair chain: damped
    gradient ascent on log pi over ALL 400 ORDERED states of the complete-data LL
    (pi enters nonlinearly through f=sqrt(pi_y/pi_x)).  IDENTICAL to
    FP.mstep_pi_metropolis except the pi(a,b)=pi(b,a) symmetrisation (b=0.5(b+b.T),
    gM+gM.T) is dropped -- so the stationary is free to be asymmetric.  On sigma-
    symmetric (N,T,pi) the gradient is sigma-symmetric, so the symmetric manifold is
    invariant (the degenerate reduction)."""
    b = np.log(np.maximum(np.asarray(pi, float), 1e-12))
    scale = max(float(T.sum()), 1.0)
    for _ in range(steps):
        pm = np.exp(b - b.max()); pm = pm / pm.sum()
        Q = FP._met_Q(S, pm.reshape(NA, NA), "sqrt")
        R = N - T[:, None] * Q                              # residual N_xy - T_x Q_xy
        ds, dd = FP._met_dlogf(pm[:, None], pm[None, :], "sqrt")
        grad = pm * ((R * ds).sum(1) + (R * dd).sum(0))     # d ll / d log pi_z (400,)
        b = b + lr * grad / scale
    pm = np.exp(b - b.max()); return pm / pm.sum()


def mstep_shared_asym(Ncounts, S, pis, tau, inner, free_S):
    """Joint (pi_c, shared S) M-step at fixed responsibility-weighted, swap-folded
    counts Ncounts (list of P tensors, already in each pi_c's '+' frame).  Mirrors
    FS.mstep_shared but with the asymmetric pi-step.  For `inner` rounds: per pair run
    the HR estep under Q_c=met_sqrt(S,pi_c), update pi_c (asymmetric), accumulate the
    pooled Metropolis S suff-stats; then (free_S) refit the single shared symmetric S."""
    P = len(pis); pis = [p.copy() for p in pis]
    for _ in range(inner):
        Cnum_tot = np.zeros((NA, NA)); Hden_tot = np.zeros((NA, NA))
        for c in range(P):
            Q_c = FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt")
            N_c, T_c = FP.estep(Q_c, pis[c], tau, Ncounts[c])
            pis[c] = mstep_pi_metropolis_asym(S, pis[c], N_c, T_c)
            Cnum_c, Hden_c = FS.met_S_suffstats(N_c, T_c, pis[c], "sqrt")
            Cnum_tot += Cnum_c; Hden_tot += Hden_c
        if free_S:
            S = np.where(Hden_tot > 0, Cnum_tot / np.maximum(Hden_tot, 1e-300), 0.0)
            S = 0.5 * (S + S.T); np.fill_diagonal(S, 0.0)
    Qs = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(P)]
    return S, pis, Qs


# ---------- 2P-component swap-pair mixture scoring ----------
def pair_scores(grid_c, logw_half, PF, PT, PFs, PTs, TB, CNT, SEG, G):
    """(G,2) score columns for swap-pair c: '+' scores pi_c's grid at (PF,PT),
    '-' scores the SAME grid at swapped indices (PFs,PTs) = pi_c^swap.  Both carry the
    tied equal log-prior logw_half = log(W_c/2)."""
    llp = FS.cluster_ll(grid_c, PF, PT, TB, CNT, SEG, G, logw_half)
    llm = FS.cluster_ll(grid_c, PFs, PTs, TB, CNT, SEG, G, logw_half)
    return llp, llm


def mixture_marginal(scores2P, gsel, tot):
    """Observed-data per-count marginal LL over the 2P components (logsumexp)."""
    lls = scores2P[gsel]
    m = lls.max(1)
    return float((m + np.log(np.exp(lls - m[:, None]).sum(1))).sum() / tot)


def build_2P_scores(Qs, pis, Wp, tau, PF, PT, PFs, PTs, TB, CNT, SEG, G):
    """(G, 2P) log w_comp + LL_comp for every swap-pair component; grids built once."""
    P = len(pis)
    cols = np.empty((G, 2 * P))
    for c in range(P):
        grid = FS.logP_grid(Qs[c], pis[c], tau)
        lh = np.log(Wp[c] / 2.0 + 1e-300)
        llp, llm = pair_scores(grid, lh, PF, PT, PFs, PTs, TB, CNT, SEG, G)
        cols[:, 2 * c] = llp; cols[:, 2 * c + 1] = llm
    return cols


# ================================ asymmetric fit ================================
def fit_asym(z, split, twoP, em_iters=60, inner=2, free_S=True, seed=0,
             symmetric_init=False, force_symmetric=False, verbose=True):
    """Fit P=twoP//2 asymmetric swap-pairs (2P components).  `symmetric_init` starts
    from symmetric pi_c (degenerate); `force_symmetric` re-symmetrises pi_c every
    M-step (projects onto the exchangeable manifold) -- together they make the run
    track the symmetric mixture, for the degenerate-reduction end-to-end check."""
    (trPF, trPT, trPFs, trPTs, trTB, trCNT, trSEG, tr_g, tr_tot,
     vaPF, vaPT, vaPFs, vaPTs, vaTB, vaCNT, vaSEG, va_g, va_tot, G, tau, glob) = split
    P = twoP // 2
    rng = np.random.default_rng(seed); t0 = time.time()
    flat_p = (trPF * NS + trPT) * len(tau) + trTB           # '+' folded-count flat idx
    flat_m = (trPFs * NS + trPTs) * len(tau) + trTB         # '-' folded-count flat idx
    Tn = len(tau)

    # init: global stationary x log-noise (asymmetric unless symmetric_init)
    pis = []
    for c in range(P):
        nz = rng.normal(0, 0.6, (NA, NA))
        if symmetric_init:
            nz = 0.5 * (nz + nz.T)
        p = glob.reshape(NA, NA) * np.exp(nz); p = (p / p.sum()).reshape(NS)
        pis.append(p)
    S = S_LG.copy()
    Wp = np.full(P, 1.0 / P)
    Qs = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(P)]

    prev = None; trll = float("nan"); monotone = True; it = 0
    for it in range(em_iters):
        cols = build_2P_scores(Qs, pis, Wp, tau, trPF, trPT, trPFs, trPTs,
                               trTB, trCNT, trSEG, G)
        trll = mixture_marginal(cols, tr_g, tr_tot)
        if prev is not None and trll < prev - 1e-9:
            monotone = False
        sc = cols[tr_g]; sc = sc - sc.max(1, keepdims=True)
        R = np.exp(sc); R /= R.sum(1, keepdims=True)         # (n_tr, 2P)
        # tied equal-weight swap-pairs: pair weight = both signs; each comp gets W_c/2
        pair_resp = R.reshape(R.shape[0], P, 2).sum(2)       # (n_tr, P)
        Wp = pair_resp.mean(0) + 1e-9; Wp /= Wp.sum()
        R_full = np.zeros((G, 2 * P)); R_full[tr_g] = R
        # swap-folded responsibility-weighted counts per pair (in pi_c's '+' frame)
        Ncounts = []
        for c in range(P):
            wp = trCNT * R_full[trSEG, 2 * c]
            wm = trCNT * R_full[trSEG, 2 * c + 1]
            Nc = (np.bincount(flat_p, weights=wp, minlength=NS * NS * Tn)
                  + np.bincount(flat_m, weights=wm, minlength=NS * NS * Tn)
                  ).reshape(NS, NS, Tn)
            Ncounts.append(Nc)
        S, pis, Qs = mstep_shared_asym(Ncounts, S, pis, tau, inner, free_S)
        if force_symmetric:                                  # project onto exchangeable
            pis = [0.5 * (p + swap_pi(p)) for p in pis]
            Qs = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(P)]
        srel, scorr = FS.s_stats(S)
        tv = [round(asym_tv(p), 3) for p in pis]
        if verbose:
            print(f"  it {it:2d}: train_mix={trll:.4f}  W={np.round(np.sort(Wp)[::-1],3)}"
                  f"  MI={sorted([round(FS.mi(p),3) for p in pis],reverse=True)}"
                  f"  asymTV={sorted(tv,reverse=True)}  Srel={srel:.3f}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
        if prev is not None and abs(trll - prev) < 1e-5:
            if verbose:
                print("  converged", flush=True)
            break
        prev = trll

    # ---- held-out marginal + asymmetry diagnostics ----
    cols_va = build_2P_scores(Qs, pis, Wp, tau, vaPF, vaPT, vaPFs, vaPTs,
                              vaTB, vaCNT, vaSEG, G)
    vall = mixture_marginal(cols_va, va_g, va_tot)

    # cost of forcing symmetry: re-score held-out with each pi_c -> (pi_c+pi_c^swap)/2
    pis_sym = [0.5 * (p + swap_pi(p)) for p in pis]
    Qs_sym = [FP._met_Q(S, pis_sym[c].reshape(NA, NA), "sqrt") for c in range(P)]
    cols_va_sym = build_2P_scores(Qs_sym, pis_sym, Wp, tau, vaPF, vaPT, vaPFs, vaPTs,
                                  vaTB, vaCNT, vaSEG, G)
    vall_symforced = mixture_marginal(cols_va_sym, va_g, va_tot)
    cols_tr_sym = build_2P_scores(Qs_sym, pis_sym, Wp, tau, trPF, trPT, trPFs, trPTs,
                                  trTB, trCNT, trSEG, G)
    trll_symforced = mixture_marginal(cols_tr_sym, tr_g, tr_tot)

    order = np.argsort(Wp)[::-1]
    res = dict(
        side="asymmetric_swap_pair", twoP=int(twoP), n_pairs=int(P),
        n_components=int(2 * P), n_pi_params=int(P * (NS - 1)),
        val_per_count_ll=float(vall), train_per_count_ll=float(trll),
        val_per_count_ll_symforced=float(vall_symforced),
        train_per_count_ll_symforced=float(trll_symforced),
        force_symmetry_cost_val=float(vall - vall_symforced),
        force_symmetry_cost_train=float(trll - trll_symforced),
        pair_weights=[float(Wp[c]) for c in order],
        sym_kl_per_pair=[float(sym_kl(pis[c])) for c in order],
        asym_tv_per_pair=[float(asym_tv(pis[c])) for c in order],
        mi_pi=[float(FS.mi(pis[c])) for c in order],
        weighted_sym_kl=float(sum(Wp[c] * sym_kl(pis[c]) for c in range(P))),
        weighted_asym_tv=float(sum(Wp[c] * asym_tv(pis[c]) for c in range(P))),
        monotone=bool(monotone), n_em_iters=int(it + 1),
        shared_s_rel_fro=float(FS.s_stats(S)[0]),
        shared_s_offdiag_corr=float(FS.s_stats(S)[1]),
        n_train_clusters=int(len(tr_g)), n_val_clusters=int(len(va_g)),
        symmetric_init=bool(symmetric_init), force_symmetric=bool(force_symmetric),
        seed=int(seed))
    return res, dict(S=S, pis=np.array(pis), Wp=Wp), (Qs, pis, Wp)


# ---------- corpus load + family split (identical to fit_coupling_mixture_freeS) ----------
def load_split(corpus, val_frac=0.2, min_counts=10, seed=0):
    z = np.load(corpus, allow_pickle=True)
    PF = z["pf"].astype(np.int64); PT = z["pt"].astype(np.int64)
    TB = z["tb"].astype(np.int64); CNT = z["cnt"].astype(np.float64)
    cptr = z["cptr"].astype(np.int64); meta = z["meta"]; tau = z["tau_centers"].astype(float)
    G = len(cptr) - 1
    SEG = np.repeat(np.arange(G), np.diff(cptr))
    tot_cnt = np.add.reduceat(CNT, cptr[:-1])
    keep = tot_cnt >= min_counts
    rng = np.random.default_rng(seed)
    fam = meta[:, 0]; ufam = np.unique(fam); rng.shuffle(ufam)
    valfam = set(ufam[:int(len(ufam) * val_frac)].tolist())
    is_val = np.isin(fam, list(valfam))
    tr_g = np.where(keep & ~is_val)[0]; va_g = np.where(keep & is_val)[0]
    trm = (keep & ~is_val)[SEG]; vam = (keep & is_val)[SEG]
    trPF, trPT, trTB, trCNT, trSEG = PF[trm], PT[trm], TB[trm], CNT[trm], SEG[trm]
    vaPF, vaPT, vaTB, vaCNT, vaSEG = PF[vam], PT[vam], TB[vam], CNT[vam], SEG[vam]
    trPFs, trPTs = SWAP_IDX[trPF], SWAP_IDX[trPT]
    vaPFs, vaPTs = SWAP_IDX[vaPF], SWAP_IDX[vaPT]
    tr_tot = tot_cnt[tr_g].sum(); va_tot = tot_cnt[va_g].sum()
    occ = (np.bincount(trPF, weights=trCNT, minlength=NS)
           + np.bincount(trPT, weights=trCNT, minlength=NS))
    glob = occ / occ.sum()
    split = (trPF, trPT, trPFs, trPTs, trTB, trCNT, trSEG, tr_g, tr_tot,
             vaPF, vaPT, vaPFs, vaPTs, vaTB, vaCNT, vaSEG, va_g, va_tot, G, tau, glob)
    return z, split


# ---------- symmetric matched baseline: run fit_coupling_mixture_freeS UNCHANGED ----------
def run_symmetric_baseline(K, seed, outdir):
    out = os.path.join(outdir, f"sym_K{K}.json")
    cmd = [sys.executable, "experiments/fit_coupling_mixture_freeS.py",
           "--K", str(K), "--seed", str(seed), "--em-iters", "60", "--inner", "2",
           "--out", out]
    env = dict(os.environ, OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6")
    print(f"# [sym baseline] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    return json.load(open(out))


# ================================ degenerate check ================================
def degenerate_check(split, seed=0):
    """EXACT model-level reduction: with SYMMETRIC pi_c the swap-pair mixture is the
    SAME probability model as the symmetric mixture (identical per-cluster held-out LL).
    Also: (a) the asymmetric M-step maps sigma-symmetric (N,T,pi) to a symmetric pi
    (the symmetric manifold is invariant); (b) the fitted mixture is exactly
    sigma-swap-invariant.  Returns a dict of the max abs discrepancies."""
    (trPF, trPT, trPFs, trPTs, trTB, trCNT, trSEG, tr_g, tr_tot,
     vaPF, vaPT, vaPFs, vaPTs, vaTB, vaCNT, vaSEG, va_g, va_tot, G, tau, glob) = split
    rng = np.random.default_rng(seed); P = 3
    # random SYMMETRIC pi_c + random symmetric S + random weights
    S = S_LG * np.exp(0.2 * rng.normal(size=S_LG.shape)); S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)
    pis, w = [], rng.random(P) + 0.1; w /= w.sum()
    gsym = 0.5 * (glob.reshape(NA, NA) + glob.reshape(NA, NA).T)   # symmetric base
    for c in range(P):
        nz = rng.normal(0, 0.5, (NA, NA)); nz = 0.5 * (nz + nz.T)
        pm = gsym * np.exp(nz); pm = 0.5 * (pm + pm.T)             # force EXACT symmetry
        pis.append((pm / pm.sum()).reshape(NS))
    Qs = [FP._met_Q(S, pis[c].reshape(NA, NA), "sqrt") for c in range(P)]

    # (1) symmetric mixture held-out per-cluster LL (FS machinery, K=P)
    grids = [FS.logP_grid(Qs[c], pis[c], tau) for c in range(P)]
    sym_cluster = np.array([FS.cluster_ll(grids[c], vaPF, vaPT, vaTB, vaCNT, vaSEG, G,
                                          np.log(w[c] + 1e-300)) for c in range(P)]).T[va_g]
    m = sym_cluster.max(1); sym_ll = m + np.log(np.exp(sym_cluster - m[:, None]).sum(1))
    # (2) swap-pair mixture held-out per-cluster LL: pair weight W_c=w_c, each comp W_c/2
    cols = build_2P_scores(Qs, pis, w, tau, vaPF, vaPT, vaPFs, vaPTs,
                           vaTB, vaCNT, vaSEG, G)[va_g]
    m2 = cols.max(1); swp_ll = m2 + np.log(np.exp(cols - m2[:, None]).sum(1))
    d_model = float(np.abs(sym_ll - swp_ll).max())

    # (3) asym M-step preserves symmetry: sigma-symmetric (N,T) + symmetric pi -> symmetric pi
    N0 = np.abs(rng.normal(size=(NS, NS))); N0 = 0.5 * (N0 + N0[SWAP_IDX][:, SWAP_IDX])
    np.fill_diagonal(N0, 0.0)
    T0 = np.abs(rng.normal(size=NS)); T0 = 0.5 * (T0 + T0[SWAP_IDX])
    p_out = mstep_pi_metropolis_asym(S, pis[0], N0, T0, steps=80)
    d_mstep = float(np.abs(p_out - swap_pi(p_out)).max())

    # (4) fitted-model exact swap invariance: L_mix(g) == L_mix(sigma g)
    d_swapinv = float(np.abs(swp_ll - swp_ll).max())  # placeholder; real check below
    cols_swapped = build_2P_scores(Qs, pis, w, tau, vaPFs, vaPTs, vaPF, vaPT,
                                   vaTB, vaCNT, vaSEG, G)[va_g]
    ms = cols_swapped.max(1); swp_ll_sig = ms + np.log(np.exp(cols_swapped - ms[:, None]).sum(1))
    d_swapinv = float(np.abs(swp_ll - swp_ll_sig).max())
    return dict(model_reduction_maxabs=d_model, mstep_symmetry_maxabs=d_mstep,
                swap_invariance_maxabs=d_swapinv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/per_contact_trrosetta/counts.npz")
    ap.add_argument("--sweep", default="2,4,8", help="comma list of matched sizes 2P")
    ap.add_argument("--em-iters", type=int, default=60)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-counts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results/mixture_asym")
    ap.add_argument("--no-sym", action="store_true", help="skip the symmetric baseline")
    ap.add_argument("--degenerate-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    z, split = load_split(a.corpus, a.val_frac, a.min_counts, a.seed)
    G = split[18]; tr_g = split[7]; va_g = split[16]
    print(f"# corpus {a.corpus}: {G} clusters; train {len(tr_g)} / val {len(va_g)} "
          f"(family split, seed {a.seed}); ordering = sequence order (colA<colB), "
          f"NOT folded", flush=True)

    # ---- degenerate reduction check ----
    print("# ==== degenerate check (symmetric pi_c => reduces to symmetric mixture) ====",
          flush=True)
    dchk = degenerate_check(split, a.seed)
    print(f"#   model-level reduction  max|LL_swap - LL_sym|   = {dchk['model_reduction_maxabs']:.2e}",
          flush=True)
    print(f"#   asym M-step symmetry   max|pi - pi^swap|       = {dchk['mstep_symmetry_maxabs']:.2e}",
          flush=True)
    print(f"#   fitted swap-invariance max|LL(g) - LL(sigma g)|= {dchk['swap_invariance_maxabs']:.2e}",
          flush=True)
    assert dchk["model_reduction_maxabs"] < 1e-8, "degenerate reduction FAILED"
    assert dchk["swap_invariance_maxabs"] < 1e-8, "swap invariance FAILED"
    assert dchk["mstep_symmetry_maxabs"] < 1e-6, "asym M-step breaks symmetric manifold"
    print("#   degenerate check PASSED", flush=True)
    json.dump(dchk, open(os.path.join(a.outdir, "degenerate_check.json"), "w"), indent=2)
    if a.degenerate_only:
        return

    sizes = [int(x) for x in a.sweep.split(",")]
    summary = dict(corpus=a.corpus, seed=a.seed, ordering="sequence(colA<colB), not folded",
                   degenerate_check=dchk, rows=[])
    for twoP in sizes:
        print(f"\n# ============ matched size 2P={twoP} "
              f"({twoP//2} asym swap-pairs  vs  {twoP} symmetric comps) ============",
              flush=True)
        t0 = time.time()
        res, params, _ = fit_asym(z, split, twoP, a.em_iters, a.inner, True, a.seed)
        np.savez(os.path.join(a.outdir, f"asym_2P{twoP}.npz"), **params)
        json.dump(res, open(os.path.join(a.outdir, f"asym_2P{twoP}.json"), "w"), indent=2)
        sym = None
        if not a.no_sym:
            sym = run_symmetric_baseline(twoP, a.seed, a.outdir)
        sym_val = None if sym is None else float(sym["val_per_count_ll"])
        delta = None if sym_val is None else float(res["val_per_count_ll"] - sym_val)
        row = dict(twoP=twoP, n_pairs=twoP // 2,
                   asym_val=res["val_per_count_ll"], sym_val=sym_val,
                   delta_asym_minus_sym=delta,
                   asym_train=res["train_per_count_ll"],
                   sym_train=(None if sym is None else float(sym["train_per_count_ll"])),
                   force_symmetry_cost_val=res["force_symmetry_cost_val"],
                   weighted_sym_kl=res["weighted_sym_kl"],
                   weighted_asym_tv=res["weighted_asym_tv"],
                   sym_kl_per_pair=res["sym_kl_per_pair"],
                   asym_tv_per_pair=res["asym_tv_per_pair"],
                   asym_params=res["n_pi_params"], sym_params=twoP * (210 - 1),
                   asym_monotone=res["monotone"])
        summary["rows"].append(row)
        w = ("ASYM" if (delta is not None and delta > 0) else
             "SYM" if delta is not None else "n/a")
        print(f"# >> 2P={twoP}: asym_val={res['val_per_count_ll']:.4f}  "
              f"sym_val={sym_val}  delta={delta}  winner={w}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(summary, open(os.path.join(a.outdir, "summary.json"), "w"), indent=2)

    print("\n# ==== MATCHED-BUDGET SUMMARY (held-out per-count LL) ====", flush=True)
    print(f"# {'2P':>3} {'asym_val':>10} {'sym_val':>10} {'delta':>9} "
          f"{'wSymKL':>8} {'wTV':>7} {'winner':>7}", flush=True)
    for r in summary["rows"]:
        w = ("ASYM" if (r["delta_asym_minus_sym"] or 0) > 0 else "SYM")
        sv = "n/a" if r["sym_val"] is None else f"{r['sym_val']:.4f}"
        dl = "n/a" if r["delta_asym_minus_sym"] is None else f"{r['delta_asym_minus_sym']:+.4f}"
        print(f"# {r['twoP']:>3} {r['asym_val']:>10.4f} {sv:>10} {dl:>9} "
              f"{r['weighted_sym_kl']:>8.4f} {r['weighted_asym_tv']:>7.3f} {w:>7}", flush=True)
    json.dump(summary, open(os.path.join(a.outdir, "summary.json"), "w"), indent=2)
    print(f"# wrote {os.path.join(a.outdir,'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
