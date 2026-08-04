#!/usr/bin/env python3
r"""How many coupling components (pairing archetypes) does the coevolution data
support?  A STATIONARY-ONLY probe: the number of components is a stationary
question (rate-invariant), so it needs only the tree-weighted per-contact
amino-acid-PAIR count tables -- NOT the CTMC.

We fit a Dirichlet-process mixture of Dirichlet-multinomials (and, as a robust
cross-check, the finite Holmes-Harris-Quince Dirichlet-multinomial mixture) to
the tree-weighted symmetric 210-unordered-pair count vectors of each structural
contact, and read off the posterior / evidence-optimal number of components.

Everything is exchangeable: counts are folded to the 210 unordered amino-acid
pairs, so component means are symmetric joints (a coupling on top of the marginal
chemistry).  numpy / scipy only -- no JAX.

Subcommands
-----------
  build : per-contact tree-weighted (theta=0.8 identity reweighting) symmetric
          210-pair count tables over the CherryML/trRosetta families (contact
          def Cb<8A, |i-j|>=7, greedy maximal matching -- reused from
          build_cherry_counts_trrosetta).  Cached sparse (weighted+unweighted
          share the same nonzero support).
  fit   : the finite DMM sweep (K=1..Kmax; held-out predictive, BIC, AIC,
          HHQ-Laplace evidence), the DP-DMM collapsed Gibbs (occupied-K vs
          alpha, + Gamma-hyperprior), archetype interpretability (MI + biophysics),
          on both weighted and unweighted tables.  Writes a JSON report.

Data is read from the main-repo absolute path by default (it is gitignored and
does not live in the worktree)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln, psi  # psi = digamma
from scipy.special import polygamma

# ---------------------------------------------------------------------------
# Alphabet / contact machinery (reused from build_cherry_counts_trrosetta)
# ---------------------------------------------------------------------------
sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
from build_cherry_counts_trrosetta import A, REMAP, maximal_contact_matching  # noqa: E402

OUR_ALPHA = "ACDEFGHIKLMNPQRSTVWY"
S = 20
NPAIR = S * (S + 1) // 2  # 210 unordered amino-acid pairs (incl. diagonal)

# unordered-pair index maps
PAIR_IDX = np.zeros((S, S), np.int64)
INV_A = np.zeros(NPAIR, np.int64)
INV_B = np.zeros(NPAIR, np.int64)
_k = 0
for _a in range(S):
    for _b in range(_a, S):
        PAIR_IDX[_a, _b] = _k
        PAIR_IDX[_b, _a] = _k
        INV_A[_k] = _a
        INV_B[_k] = _b
        _k += 1
assert _k == NPAIR

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(_REPO_ROOT, "data", "cherryml_trrosetta", "training_set")


# ---------------------------------------------------------------------------
# Tree-weighting proxy: theta-identity sequence reweighting (pure numpy)
# ---------------------------------------------------------------------------
def seq_weights(msa, theta=0.8, chunk=256):
    """w_s = 1 / #{sequences (incl. self) with >= theta fractional identity to s};
    M_eff = sum_s w_s.  Identity = fraction of the L columns on which two rows
    agree (gap==gap counts as agreement, standard DCA convention)."""
    N, L = msa.shape
    counts = np.zeros(N, np.float64)
    for s0 in range(0, N, chunk):
        e = min(s0 + chunk, N)
        fid = (msa[s0:e, None, :] == msa[None, :, :]).mean(2)  # (e-s0, N)
        counts[s0:e] = (fid >= theta).sum(1)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    return w, float(w.sum())


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------
def build(args):
    ids = [l.strip() for l in (Path(args.tr_dir) / "list15051.txt").read_text().splitlines()
           if l.strip()]
    if args.max_fam:
        ids = ids[: args.max_fam]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    idx_rows, wval_rows, uval_rows, ptr = [], [], [], [0]
    meta = []          # [fam_idx, colA, colB]
    fam_ids = []
    meff_fam = []
    t0 = time.time()
    nfam = 0
    for fi, fid in enumerate(ids):
        p = Path(args.tr_dir) / "npz" / f"{fid}.npz"
        try:
            d = np.load(p, allow_pickle=True)
            msa = np.asarray(d["msa"])
            dist6d = np.asarray(d["dist6d"], np.float64)
        except Exception:
            continue
        if msa.ndim != 2 or msa.shape[0] < 4:
            continue
        L = msa.shape[1]
        if dist6d.shape != (L, L):
            continue
        msa = REMAP[msa.astype(np.int64)]              # -> our alphabet, gap=20
        N = msa.shape[0]
        if N > args.max_seqs:                          # subsample, keep ref row 0
            rng = np.random.default_rng(args.seed + fi)
            pick = np.concatenate([[0], 1 + rng.permutation(N - 1)[: args.max_seqs - 1]])
            msa = msa[pick]
        w, meff = seq_weights(msa, theta=args.theta)
        pairs, _ = maximal_contact_matching(dist6d, args.cb_max, args.min_sep)
        if not pairs:
            continue
        fam_kept = 0
        for (ca, cb) in pairs:
            x = msa[:, ca]
            y = msa[:, cb]
            ok = (x < A) & (y < A)
            if ok.sum() < 2:
                continue
            xo = x[ok]
            yo = y[ok]
            lo = np.minimum(xo, yo)
            hi = np.maximum(xo, yo)
            pidx = PAIR_IDX[lo, hi]
            wc = np.bincount(pidx, weights=w[ok], minlength=NPAIR)
            uc = np.bincount(pidx, minlength=NPAIR).astype(np.float64)
            neff = wc.sum()
            if neff < args.min_neff:
                continue
            nz = np.nonzero(uc)[0]
            idx_rows.append(nz.astype(np.int16))
            wval_rows.append(wc[nz].astype(np.float32))
            uval_rows.append(uc[nz].astype(np.float32))
            ptr.append(ptr[-1] + len(nz))
            meta.append((nfam, int(ca), int(cb)))
            fam_kept += 1
        if fam_kept:
            fam_ids.append(fid)
            meff_fam.append(meff)
            nfam += 1
        if (fi + 1) % 200 == 0:
            print(f"  [{fi+1}/{len(ids)}] fams={nfam} contacts={len(meta)} "
                  f"t={time.time()-t0:.0f}s", flush=True)

    if not meta:
        print("# no contacts kept"); return
    idx = np.concatenate(idx_rows)
    wval = np.concatenate(wval_rows)
    uval = np.concatenate(uval_rows)
    ptr = np.array(ptr, np.int64)
    meta = np.array(meta, np.int32)
    np.savez_compressed(
        out, ptr=ptr, idx=idx, wval=wval, uval=uval, meta=meta,
        fam_ids=np.array(fam_ids), meff_fam=np.array(meff_fam, np.float32),
        npair=np.int64(NPAIR), theta=np.float64(args.theta),
        min_neff=np.float64(args.min_neff), max_seqs=np.int64(args.max_seqs))
    Neff = np.add.reduceat(wval, ptr[:-1])
    Nraw = np.add.reduceat(uval, ptr[:-1])
    print(f"# BUILT {len(meta):,} contacts over {nfam} families -> {out}")
    print(f"#   effective-count (weighted N_eff) quantiles: "
          f"min={Neff.min():.0f} 10%={np.percentile(Neff,10):.0f} "
          f"med={np.median(Neff):.0f} 90%={np.percentile(Neff,90):.0f} "
          f"max={Neff.max():.0f} mean={Neff.mean():.0f}")
    print(f"#   raw-count (unweighted) median={np.median(Nraw):.0f}; "
          f"redundancy factor med(N_raw/N_eff)={np.median(Nraw/Neff):.2f}")
    print(f"#   family M_eff: med={np.median(meff_fam):.1f} "
          f"(raw seqs subsampled to <= {args.max_seqs})")


# ---------------------------------------------------------------------------
# Sparse Dirichlet-multinomial (Polya) machinery
# ---------------------------------------------------------------------------
class Corpus:
    """CSR sparse count corpus over NPAIR categories."""
    def __init__(self, ptr, idx, val):
        self.ptr = ptr
        self.idx = idx.astype(np.int64)
        self.val = val.astype(np.float64)
        self.G = len(ptr) - 1
        self.N = np.add.reduceat(self.val, ptr[:-1]) if len(self.val) else np.zeros(self.G)
        # row id per nonzero
        self.rowid = np.repeat(np.arange(self.G), np.diff(ptr))

    def subset(self, gsel):
        """Return a new Corpus with only rows in boolean/index gsel (vectorised)."""
        gsel = np.atleast_1d(gsel)
        if gsel.dtype == bool:
            gsel = np.nonzero(gsel)[0]
        if len(gsel) == 0:
            return Corpus(np.zeros(1, np.int64), np.zeros(0, np.int64), np.zeros(0))
        lens = (self.ptr[gsel + 1] - self.ptr[gsel]).astype(np.int64)
        new_ptr = np.concatenate([[0], np.cumsum(lens)])
        out_n = int(new_ptr[-1])
        run_id = np.repeat(np.arange(len(gsel)), lens)
        within = np.arange(out_n) - new_ptr[run_id]
        src = self.ptr[gsel][run_id] + within
        return Corpus(new_ptr, self.idx[src], self.val[src])


def polya_loglik_all(corpus, beta):
    """Per-row Polya (Dirichlet-multinomial) log-lik under each of K components.
    beta: (K, NPAIR) > 0.  Returns (G, K).  Drops the (component-independent)
    multinomial coefficient log(N!/prod x!)."""
    K = beta.shape[0]
    B = beta.sum(1)                                   # (K,)
    lgB = gammaln(B)                                  # (K,)
    ll = (lgB[None, :] - gammaln(corpus.N[:, None] + B[None, :]))  # (G,K)
    lg_beta = gammaln(beta)                           # (K, NPAIR)
    # sum over nonzero entries: gammaln(x_gi + beta_ki) - gammaln(beta_ki)
    # vectorised over nonzeros and components
    val = corpus.val                                  # (nnz,)
    idx = corpus.idx                                  # (nnz,)
    rowid = corpus.rowid
    # (nnz, K): gammaln(val + beta[:, idx].T) - lg_beta[:, idx].T
    beta_nz = beta[:, idx].T                          # (nnz, K)
    term = gammaln(val[:, None] + beta_nz) - lg_beta[:, idx].T  # (nnz, K)
    # scatter-add into rows
    for k in range(K):
        ll[:, k] += np.bincount(rowid, weights=term[:, k], minlength=corpus.G)
    return ll


def polya_mixture_ll(corpus, beta, logpi):
    """sum_g log sum_k pi_k DM(n_g | beta_k)."""
    ll = polya_loglik_all(corpus, beta) + logpi[None, :]
    m = ll.max(1)
    return float((m + np.log(np.exp(ll - m[:, None]).sum(1))).sum()), ll


def minka_update(corpus, resp, beta, prior_a=1.0, prior_b=0.0, iters=1):
    """Minka fixed-point M-step for a mixture of Polyas (weighted by responsibilities
    resp (G,K)).  Optional weak Gamma(prior_a, rate=prior_b) prior on each beta_ki
    (MAP): adds (prior_a-1)/beta - prior_b to the score => multiplicative factor.
    Returns updated beta (K, NPAIR)."""
    K = beta.shape[0]
    beta = beta.copy()
    val = corpus.val
    idx = corpus.idx
    rowid = corpus.rowid
    for _ in range(iters):
        B = beta.sum(1)                               # (K,)
        # denom_k = sum_g resp_gk (psi(N_g + B_k) - psi(B_k))
        denom = (resp * (psi(corpus.N[:, None] + B[None, :]) - psi(B)[None, :])).sum(0)  # (K,)
        beta_nz = beta[:, idx].T                       # (nnz, K)
        # numer contribution per nonzero: resp[row,k] * (psi(val+beta_ki) - psi(beta_ki))
        rr = resp[rowid]                               # (nnz, K)
        contrib = rr * (psi(val[:, None] + beta_nz) - psi(beta_nz))  # (nnz, K)
        numer = np.zeros((K, NPAIR))
        for k in range(K):
            numer[k] = np.bincount(idx, weights=contrib[:, k], minlength=NPAIR)
        # MAP with Gamma(prior_a, rate=prior_b): factor multiplies numerator by
        # ((numer + (prior_a-1)) / (denom + prior_b)); keep it simple & stable.
        num = numer + (prior_a - 1.0)
        den = denom[:, None] + prior_b
        beta = beta * np.clip(num, 1e-12, None) / np.clip(den, 1e-12, None)
        beta = np.clip(beta, 1e-6, 1e6)
    return beta


def mstep_mean(corpus, resp):
    """Multinomial mean MLE per component (responsibility-pooled composition),
    rows summing to 1.  Used when the concentration is HELD FIXED (regime study)."""
    K = resp.shape[1]
    rr = resp[corpus.rowid]                              # (nnz, K)
    M = np.zeros((K, NPAIR))
    for k in range(K):
        M[k] = np.bincount(corpus.idx, weights=corpus.val * rr[:, k], minlength=NPAIR)
    M += 1e-9
    M /= M.sum(1, keepdims=True)
    return M


def fit_dmm(corpus, K, seed=0, max_iters=200, tol=1e-4, inner=1, verbose=False,
            init_beta=None, prior_a=1.0, prior_b=0.0, fixed_B=None):
    """EM for a K-component Dirichlet-multinomial mixture on `corpus`.
    If fixed_B is None the per-component Dirichlet concentration is FIT FREELY
    (Minka's fixed point, which jointly fits mean direction and concentration =
    overdispersion).  If fixed_B is a number, the concentration is HELD at that
    value and only the mean direction is fit -- large fixed_B => near-multinomial
    (under-dispersed) components; small fixed_B => forced high overdispersion.
    Returns (beta (K,NPAIR), pi (K,), ll_trace, final_ll)."""
    rng = np.random.default_rng(seed)
    G = corpus.G
    # global mean direction
    glob = np.zeros(NPAIR)
    np.add.at(glob, corpus.idx, corpus.val)
    glob = glob / glob.sum()
    conc0 = 50.0 if fixed_B is None else float(fixed_B)
    if init_beta is None:
        beta = np.empty((K, NPAIR))
        for k in range(K):
            jit = rng.gamma(shape=5.0, size=NPAIR)     # dirichlet-ish jitter
            m = glob * jit
            m = m / m.sum()
            beta[k] = np.clip(conc0 * m, 1e-4, None)
    else:
        beta = init_beta.copy()
    pi = np.full(K, 1.0 / K)
    logpi = np.log(pi)
    prev = None
    trace = []
    for it in range(max_iters):
        ll = polya_loglik_all(corpus, beta) + logpi[None, :]   # (G,K)
        m = ll.max(1)
        tot = float((m + np.log(np.exp(ll - m[:, None]).sum(1))).sum())
        resp = np.exp(ll - m[:, None])
        resp /= resp.sum(1, keepdims=True)
        pi = resp.mean(0) + 1e-12
        pi /= pi.sum()
        logpi = np.log(pi)
        if fixed_B is None:
            beta = minka_update(corpus, resp, beta, prior_a, prior_b, iters=inner)
        else:
            beta = fixed_B * mstep_mean(corpus, resp)      # concentration held fixed
        trace.append(tot)
        if verbose and (it % 10 == 0 or it == max_iters - 1):
            print(f"    K={K} it={it:3d} ll={tot:.1f} "
                  f"pi_sorted={np.round(np.sort(pi)[::-1][:6],3)}", flush=True)
        if prev is not None and abs(tot - prev) < tol * max(1.0, abs(prev)):
            break
        prev = tot
    return beta, pi, trace, tot


# ---------------------------------------------------------------------------
# Model evidence: BIC, AIC, HHQ-Laplace
# ---------------------------------------------------------------------------
def laplace_evidence(corpus, beta, pi, prior_b=0.0):
    """HHQ-style Laplace approximation to the negative log-evidence.

    -log P(D|K) ~ NLL(theta*) + 1/2 log det H  - (d/2) log(2 pi)
    Hessian of NLL is block-diagonal across components (each block is
    diagonal-plus-rank-1 for the Polya) plus the (K-1) weight block (BIC term).
    Returns log-evidence (higher = better)."""
    K = beta.shape[0]
    ll = polya_loglik_all(corpus, beta) + np.log(pi)[None, :]
    m = ll.max(1)
    nll = -float((m + np.log(np.exp(ll - m[:, None]).sum(1))).sum())
    resp = np.exp(ll - m[:, None]); resp /= resp.sum(1, keepdims=True)   # (G,K)

    B = beta.sum(1)
    logdet = 0.0
    d = 0
    val = corpus.val; idx = corpus.idx; rowid = corpus.rowid
    beta_nz = beta[:, idx].T
    rr = resp[rowid]
    # trigamma terms
    # diagonal of -LL Hessian block k, entry i:
    #   Dk_i = sum_g r_gk [ psi'(beta_ki) - psi'(x_gi + beta_ki) ]   (>=0)
    # rank-1 coefficient (negative):
    #   c_k = sum_g r_gk [ psi'(N_g + B_k) - psi'(B_k) ]             (<=0)
    tg_beta = polygamma(1, beta_nz)                                   # (nnz,K)
    tg_xbeta = polygamma(1, val[:, None] + beta_nz)                  # (nnz,K)
    dcontrib = rr * (tg_beta - tg_xbeta)                             # (nnz,K)
    c_k = (resp * (polygamma(1, corpus.N[:, None] + B[None, :])
                   - polygamma(1, B)[None, :])).sum(0)               # (K,)
    for k in range(K):
        Dk = np.bincount(idx, weights=dcontrib[:, k], minlength=NPAIR)  # (NPAIR,)
        Dk = np.clip(Dk, 1e-10, None)
        # det(D + c 11^T) = prod(D) * (1 + c * sum(1/D))
        ld = np.log(Dk).sum() + np.log(max(1.0 + c_k[k] * (1.0 / Dk).sum(), 1e-12))
        logdet += ld
        d += NPAIR
    # weight block: approximate by BIC-style (K-1)/2 log G
    d += (K - 1)
    G = corpus.G
    logdet += (K - 1) * np.log(max(G, 2))
    logev = -nll - 0.5 * logdet + 0.5 * d * np.log(2 * np.pi)
    return logev, nll


def bic_aic(nll, K, G, params_per_comp=NPAIR):
    """params_per_comp = NPAIR when the concentration is free (mean direction
    NPAIR-1 + concentration 1); NPAIR-1 when the concentration is held fixed."""
    p = K * params_per_comp + (K - 1)
    bic = 2 * nll + p * np.log(G)
    aic = 2 * nll + 2 * p
    return bic, aic, p


def variance_decomp(corpus, resp, beta, cap=30000, seed=0):
    """ANOVA-style split of the per-contact composition spread (over the 210-pair
    simplex) into BETWEEN-archetype (component-mean spread) and WITHIN-archetype
    (residual scatter around the assigned mean), plus a multinomial sampling-noise
    floor so the WITHIN part can be corrected to the *true* Dirichlet overdispersion.

    Returns fractions: frac_between_raw (of observed spread) and
    frac_between_signal (of noise-corrected spread) -- the honest 'how much of the
    real composition structure is captured by K archetype means'."""
    rng = np.random.default_rng(seed)
    G = corpus.G
    if G > cap:
        sub = np.sort(rng.permutation(G)[:cap])
        c = corpus.subset(sub); r = resp[sub]
    else:
        c = corpus; r = resp
    K = beta.shape[0]
    m = beta / beta.sum(1, keepdims=True)               # (K,NPAIR) means
    pi = r.mean(0); pi = pi / pi.sum()
    mbar = (pi[:, None] * m).sum(0)                      # global mean composition
    mm = (m ** 2).sum(1)                                 # ||m_k||^2  (K,)
    # per-contact f_g (sparse): ||f_g||^2, sampling noise, and f_g.m_k
    Ng = c.N
    fval = c.val / Ng[c.rowid]                            # (nnz,) composition entries
    fsq = np.bincount(c.rowid, weights=fval ** 2, minlength=c.G)  # ||f_g||^2
    v_samp_g = np.bincount(c.rowid, weights=fval * (1 - fval), minlength=c.G) / Ng
    # term2 = sum_g sum_k r_gk (f_g . m_k)
    rr = r[c.rowid]                                       # (nnz,K)
    fm = fval[:, None] * m[:, c.idx].T                   # (nnz,K): f_gi * m_ki
    dot_gk = np.zeros((c.G, K))
    for k in range(K):
        dot_gk[:, k] = np.bincount(c.rowid, weights=fm[:, k], minlength=c.G)
    SS_within = float((fsq[:, None] * r).sum() - 2 * (r * dot_gk).sum()
                      + (r * mm[None, :]).sum()) / c.G
    SS_between = float((pi * ((m - mbar[None, :]) ** 2).sum(1)).sum())
    V_samp = float(v_samp_g.mean())
    SS_total = SS_within + SS_between
    SS_within_true = max(SS_within - V_samp, 1e-12)
    return dict(SS_total=SS_total, SS_between=SS_between, SS_within=SS_within,
                V_sampling=V_samp,
                frac_between_raw=SS_between / max(SS_total, 1e-12),
                frac_between_signal=SS_between / (SS_between + SS_within_true))


# ---------------------------------------------------------------------------
# DP mixture of Polyas: collapsed Gibbs (fixed Dirichlet base measure)
# ---------------------------------------------------------------------------
def dp_gibbs(corpus, alpha, glob, base_conc=50.0, n_sweeps=60, burn=20, seed=0,
             map_steps=2, sample_alpha=False, alpha_gamma=(2.0, 1.0),
             init_z=None, init_betas=None, refit_fixed_B=None, verbose=False):
    """Collapsed Gibbs for a DP mixture of Dirichlet-multinomials with a FIXED
    Dirichlet base measure G0 = Dir(base_conc * glob).  New components are born at
    the base mean (prior-predictive DM(n | base_conc*glob)) and adapt via a MAP
    (Minka) refit of every occupied component's beta at each sweep end.

    Betas are held fixed within a sweep (stochastic-EM style), so the per-row
    per-component likelihood is one vectorised polya_loglik_all call; only the CRP
    count-weights change inside the assignment loop.  Returns occupied-K stats."""
    rng = np.random.default_rng(seed)
    G = corpus.G
    beta0 = np.clip(base_conc * glob, 1e-5, None)          # fixed base-measure mean
    L0 = polya_loglik_all(corpus, beta0[None, :])[:, 0]    # (G,) base predictive
    a_g, b_g = alpha_gamma

    if init_z is not None and init_betas is not None:
        z = init_z.copy()
        beta_list = [b.copy() for b in init_betas]
    else:
        z = np.zeros(G, np.int64)
        beta_list = [beta0.copy()]
    # compact
    uniq, z = np.unique(z, return_inverse=True)
    beta_list = [beta_list[u] for u in uniq]

    occ_trace, alpha_trace = [], []
    NEG = -1e30
    us = None
    for sweep in range(n_sweeps):
        nc0 = len(beta_list)
        beta_mat = np.array(beta_list)                     # (nc0, NPAIR)
        L = polya_loglik_all(corpus, beta_mat)             # (G, nc0) fixed this sweep
        # counts: [0..nc0-1] original comps (use L columns); [nc0..] newborns (use L0)
        counts = np.bincount(z, minlength=nc0).astype(np.float64)
        newborn = []                                       # counts for comps born this sweep
        log_alpha = np.log(alpha)
        us = rng.random(G)                                 # one uniform per contact
        for g in range(G):
            k_old = z[g]
            counts[k_old] -= 1
            n_nb = len(newborn)
            # vectorised candidate log-probs
            lp = np.empty(nc0 + n_nb + 1)
            with np.errstate(divide="ignore"):
                lp[:nc0] = np.where(counts > 0, np.log(counts), NEG) + L[g]
                if n_nb:
                    nb = np.asarray(newborn)
                    lp[nc0:nc0 + n_nb] = np.where(nb > 0, np.log(nb), NEG) + L0[g]
            lp[-1] = log_alpha + L0[g]
            lp -= lp.max()
            p = np.exp(lp); c = np.cumsum(p)
            ch = int(np.searchsorted(c, us[g] * c[-1]))
            if ch < nc0:
                z[g] = ch; counts[ch] += 1
            elif ch < nc0 + n_nb:
                z[g] = ch; newborn[ch - nc0] += 1          # join an existing newborn
            else:
                z[g] = nc0 + n_nb; newborn.append(1.0)     # birth at base mean
        # assemble full counts and betas (originals + newborns), drop empties
        all_counts = np.concatenate([counts, np.asarray(newborn)]) if newborn else counts
        all_betas = beta_list + [beta0.copy() for _ in newborn]
        alive = np.nonzero(all_counts > 0)[0]
        remap = -np.ones(len(all_counts), np.int64); remap[alive] = np.arange(len(alive))
        z = remap[z]
        beta_list = [all_betas[k] for k in alive]
        nc = len(beta_list)
        # MAP refit of every occupied component.  Free concentration (Minka) unless
        # refit_fixed_B is set (near-multinomial regime for the DP inflation check).
        resp = np.zeros((G, nc)); resp[np.arange(G), z] = 1.0
        if refit_fixed_B is None:
            beta_mat = minka_update(corpus, resp, np.array(beta_list), iters=map_steps)
        else:
            beta_mat = refit_fixed_B * mstep_mean(corpus, resp)
        beta_list = [np.clip(beta_mat[k], 1e-5, 1e6) for k in range(nc)]
        if sample_alpha:                                   # Escobar-West update
            kcur = nc
            eta = rng.beta(alpha + 1, G)
            odds = (a_g + kcur - 1) / (G * (b_g - np.log(eta)))
            pi_eta = odds / (1 + odds)
            shape = a_g + kcur - (0 if rng.random() < pi_eta else 1)
            alpha = float(np.clip(rng.gamma(shape, 1.0 / (b_g - np.log(eta))), 1e-3, 1e4))
        occ_trace.append(nc)
        alpha_trace.append(alpha)
        if verbose and (sweep % 10 == 0 or sweep == n_sweeps - 1):
            print(f"    [alpha={alpha:.2f}] sweep {sweep:2d} occ_K={nc}", flush=True)
    post = occ_trace[burn:]
    return dict(occ_K_trace=occ_trace, occ_K_mean=float(np.mean(post)),
                occ_K_sd=float(np.std(post)), alpha_final=float(alpha),
                alpha_mean=float(np.mean(alpha_trace[burn:])), final_z=z,
                final_betas=beta_list)


# ---------------------------------------------------------------------------
# Interpretability: component mean -> 20x20 joint -> MI + biophysics
# ---------------------------------------------------------------------------
# amino-acid property groups (indices into OUR_ALPHA)
POS = [OUR_ALPHA.index(c) for c in "KR"]         # positive
NEG = [OUR_ALPHA.index(c) for c in "DE"]         # negative
HISK = OUR_ALPHA.index("H")
AROM = [OUR_ALPHA.index(c) for c in "FWY"]       # aromatic
CYS = OUR_ALPHA.index("C")
HYDRO = [OUR_ALPHA.index(c) for c in "AILMFVW"]  # hydrophobic core
TINY = [OUR_ALPHA.index(c) for c in "AGS"]
# Kyte-Doolittle-ish volume proxy (A^3), for size matching
VOL = {c: v for c, v in zip("ACDEFGHIKLMNPQRSTVWY",
       [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,
        162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6])}
VOLV = np.array([VOL[c] for c in OUR_ALPHA])


def pair_mean_to_joint(q):
    """210-vector q (sums to 1) -> symmetric 20x20 joint P."""
    P = np.zeros((S, S))
    a = INV_A; b = INV_B
    diag = a == b
    P[a[diag], b[diag]] = q[diag]
    off = ~diag
    P[a[off], b[off]] = q[off] / 2
    P[b[off], a[off]] = q[off] / 2
    return P


def mi_of_joint(P):
    P = P / P.sum()
    r = P.sum(1); c = P.sum(0)
    return float(np.sum(P * np.log(np.maximum(P, 1e-300)
                                   / np.maximum(np.outer(r, c), 1e-300))))


def biophysics(q):
    """Describe a component's mean pair distribution q (210,)."""
    P = pair_mean_to_joint(q)
    P = P / P.sum()
    marg = P.sum(1)
    # salt-bridge (opposite charge) mass minus like-charge mass
    def mass(setA, setB):
        return float(P[np.ix_(setA, setB)].sum() + P[np.ix_(setB, setA)].sum())
    salt = mass(POS, NEG)
    likepos = float(P[np.ix_(POS, POS)].sum())
    likeneg = float(P[np.ix_(NEG, NEG)].sum())
    arom_arom = float(P[np.ix_(AROM, AROM)].sum())
    disulf = float(P[CYS, CYS])
    hydro_hydro = float(P[np.ix_(HYDRO, HYDRO)].sum())
    # size matching: corr of residue volume across the pair (exchangeable)
    # E[vol_a vol_b] - E[vol_a]E[vol_b] over the joint, normalised
    va = VOLV[:, None]; vb = VOLV[None, :]
    Ev = float((P * (va * vb)).sum())
    Em = float((marg * VOLV).sum())
    Ev2 = float((marg * VOLV ** 2).sum())
    var = max(Ev2 - Em ** 2, 1e-9)
    size_corr = (Ev - Em ** 2) / var
    # top pairs
    order = np.argsort(q)[::-1][:6]
    top = [(OUR_ALPHA[INV_A[k]] + OUR_ALPHA[INV_B[k]], round(float(q[k]), 3))
           for k in order]
    return dict(mi=round(mi_of_joint(P), 4), salt_bridge=round(salt, 4),
                like_pos=round(likepos, 4), like_neg=round(likeneg, 4),
                aromatic=round(arom_arom, 4), disulfide=round(disulf, 4),
                hydrophobic=round(hydro_hydro, 4), size_corr=round(size_corr, 3),
                top_pairs=top)


def label_component(bp):
    """Heuristic one-word archetype label from biophysics dict."""
    tags = []
    if bp["disulfide"] > 0.02:
        tags.append("disulfide(C-C)")
    if bp["salt_bridge"] > 0.06 and bp["salt_bridge"] > 1.5 * (bp["like_pos"] + bp["like_neg"]):
        tags.append("salt-bridge(+/-)")
    if bp["aromatic"] > 0.05:
        tags.append("aromatic")
    if bp["hydrophobic"] > 0.35:
        tags.append("hydrophobic-core")
    if bp["size_corr"] > 0.15:
        tags.append("size-matched")
    if bp["mi"] < 0.03:
        tags.append("background/chemistry")
    return ",".join(tags) if tags else "mixed"


def per_contact_mi(corpus, cap=20000, seed=0):
    """Per-contact empirical MI, grounding the ~0.16 per-contact figure.

    The plugin MI of a single contact's 20x20 joint is badly finite-sample-biased
    (a few hundred counts over ~400 cells), so we also report a Miller-Madow
    correction and a marginal-independent NULL floor (shuffle-free analytic:
    resample counts from the outer product of the contact's own marginals at the
    same N).  The bias-corrected coupling is roughly plugin - null_floor."""
    rng = np.random.default_rng(seed)
    G = corpus.G
    sel = rng.permutation(G)[:min(cap, G)]
    plug, mm, null = [], [], []
    for g in sel:
        s0, e0 = corpus.ptr[g], corpus.ptr[g + 1]
        P = pair_mean_to_joint_from_sparse(corpus.idx[s0:e0], corpus.val[s0:e0])
        N = P.sum()
        Pn = P / N
        mi = mi_of_joint(Pn)
        plug.append(mi)
        # Miller-Madow: plugin MI is biased UP; MI_MM = MI - (m_xy - m_x - m_y + 1)/(2N)
        r = Pn.sum(1); c = Pn.sum(0)
        m_xy = int((P > 0).sum()); m_x = int((r > 0).sum()); m_y = int((c > 0).sum())
        mm.append(mi - (m_xy - m_x - m_y + 1) / (2.0 * max(N, 1)))
        # null floor: draw N counts from independent marginals, plugin MI
        rr = np.maximum(r, 0); rr /= rr.sum(); cc = np.maximum(c, 0); cc /= cc.sum()
        a = rng.choice(S, size=int(round(N)), p=rr)
        b = rng.choice(S, size=int(round(N)), p=cc)
        Pnull = np.zeros((S, S)); np.add.at(Pnull, (a, b), 1.0)
        null.append(mi_of_joint(Pnull))
    plug = np.array(plug); mm = np.array(mm); null = np.array(null)
    return dict(plugin_mean=float(plug.mean()), plugin_median=float(np.median(plug)),
                miller_madow_mean=float(mm.mean()),
                null_floor_mean=float(null.mean()),
                bias_corrected_mean=float((plug - null).mean()))


def pair_mean_to_joint_from_sparse(idx, val):
    """Sparse 210-count -> 20x20 symmetric count joint (sums to val.sum() = N_eff)."""
    q = np.zeros(NPAIR); q[idx] = val
    return pair_mean_to_joint(q)


# ---------------------------------------------------------------------------
# FIT (driver)
# ---------------------------------------------------------------------------
def fit(args):
    z = np.load(args.corpus, allow_pickle=True)
    ptr = z["ptr"]; idx = z["idx"]; meta = z["meta"]
    fam = meta[:, 0]
    results = {"corpus": args.corpus, "n_contacts": int(len(ptr) - 1),
               "n_families": int(fam.max() + 1), "K_list": list(args.K_list)}
    rng = np.random.default_rng(args.seed)

    # optional contact subsample (shared across weighted/unweighted for comparability)
    G_all = len(ptr) - 1
    if args.max_contacts and args.max_contacts < G_all:
        sel = np.sort(np.random.default_rng(args.seed).permutation(G_all)[:args.max_contacts])
        results["n_contacts_used"] = int(len(sel))
    else:
        sel = np.arange(G_all)
        results["n_contacts_used"] = int(G_all)
    fam = fam[sel]

    for wt in ("weighted", "unweighted"):
        val = z["wval"] if wt == "weighted" else z["uval"]
        full = Corpus(ptr, idx, val).subset(sel)
        glob = np.zeros(NPAIR); np.add.at(glob, full.idx, full.val); glob /= glob.sum()
        print(f"\n===== {wt.upper()} : {full.G:,} contacts, "
              f"mean N={full.N.mean():.0f} =====", flush=True)

        # family split for held-out predictive (train fit; val = held-out families)
        ufam = np.unique(fam); rng2 = np.random.default_rng(args.seed)
        rng2.shuffle(ufam)
        valfam = set(ufam[: int(len(ufam) * args.val_frac)].tolist())
        is_val = np.isin(fam, list(valfam))
        tr = full.subset(~is_val)
        va = full.subset(is_val)

        # per-contact MI grounding (weighted only, once)
        if wt == "weighted":
            mi_info = per_contact_mi(full)
            results["per_contact_MI"] = {k: round(v, 4) for k, v in mi_info.items()}
            print(f"  per-contact MI: plugin={mi_info['plugin_mean']:.3f} "
                  f"null_floor={mi_info['null_floor_mean']:.3f} "
                  f"bias_corrected={mi_info['bias_corrected_mean']:.3f} "
                  f"miller_madow={mi_info['miller_madow_mean']:.3f}")

        # ---- finite DMM sweep (ONE fit per K on train; evidence on train) ----
        sweep = []
        best_beta = {}
        for K in args.K_list:
            beta, pi, _, _ = fit_dmm(tr, K, seed=args.seed, max_iters=args.em_iters,
                                     inner=args.inner, prior_a=args.prior_a,
                                     prior_b=args.prior_b)
            tr_ll, _ = polya_mixture_ll(tr, beta, np.log(pi))
            va_ll, _ = polya_mixture_ll(va, beta, np.log(pi))
            bic, aic, p = bic_aic(-tr_ll, K, tr.G)
            logev, _ = laplace_evidence(tr, beta, pi)
            miw = float(sum(pi[k] * mi_of_joint(pair_mean_to_joint(beta[k] / beta[k].sum()))
                            for k in range(K)))
            rec = dict(K=K, heldout_ll=va_ll, heldout_ll_per_contact=va_ll / va.G,
                       train_ll=tr_ll, BIC=bic, AIC=aic, laplace_logev=logev,
                       n_params=p, weighted_mean_MI=round(miw, 4),
                       n_effective_comps=round(float(1.0 / (pi ** 2).sum()), 2))
            sweep.append(rec)
            best_beta[K] = (beta, pi)
            print(f"  K={K:2d}  heldout/contact={va_ll/va.G:9.2f}  BIC={bic:14.0f}  "
                  f"Laplace_logev={logev:14.0f}  wMI={miw:.4f}  "
                  f"K_eff={1.0/(pi**2).sum():.1f}", flush=True)

        K_ho = int(sweep[int(np.argmax([r["heldout_ll"] for r in sweep]))]["K"])
        K_bic = int(sweep[int(np.argmin([r["BIC"] for r in sweep]))]["K"])
        K_lap = int(sweep[int(np.argmax([r["laplace_logev"] for r in sweep]))]["K"])
        results[wt] = dict(dmm_sweep=sweep, K_heldout=K_ho, K_BIC=K_bic, K_laplace=K_lap)
        print(f"  -> K* heldout={K_ho}  K* BIC={K_bic}  K* Laplace={K_lap}")

        # ---- archetype interpretability at held-out-optimal K ----
        Kstar = K_ho
        betaF, piF = best_beta[Kstar]
        order = np.argsort(piF)[::-1]
        comps = []
        for rank, k in enumerate(order):
            q = betaF[k] / betaF[k].sum()
            bp = biophysics(q)
            comps.append(dict(rank=rank, weight=round(float(piF[k]), 4),
                              conc=round(float(betaF[k].sum()), 1),
                              label=label_component(bp), **bp))
        results[wt]["archetypes_at_Kstar"] = dict(K=Kstar, components=comps)
        print(f"  archetypes at K*={Kstar} (held-out):")
        for c in comps:
            print(f"    [{c['rank']}] w={c['weight']:.3f} MI={c['mi']:.3f} "
                  f"salt={c['salt_bridge']:.3f} arom={c['aromatic']:.3f} "
                  f"disulf={c['disulfide']:.3f} hydro={c['hydrophobic']:.3f} "
                  f"size={c['size_corr']:.2f} :: {c['label']} :: top={c['top_pairs'][:4]}")

        # ---- DP occupied-K vs alpha (seeded from a moderate-K DMM partition) ----
        if args.dp:
            dp_G = min(args.dp_contacts, full.G)
            dp_sel = rng.permutation(full.G)[:dp_G]
            dp_corp = full.subset(dp_sel)
            dp_glob = np.zeros(NPAIR); np.add.at(dp_glob, dp_corp.idx, dp_corp.val)
            dp_glob /= dp_glob.sum()
            # seed partition from a K=seed_K DMM fit on the DP subset
            sb, spi = fit_dmm(dp_corp, args.dp_seed_K, seed=args.seed,
                              max_iters=args.em_iters, inner=args.inner)[:2]
            seed_ll = polya_loglik_all(dp_corp, sb) + np.log(spi)[None, :]
            seed_z = seed_ll.argmax(1)
            seed_betas = [sb[k] for k in range(args.dp_seed_K)]
            dp_rows = []
            for al in args.alpha_grid:
                r = dp_gibbs(dp_corp, al, dp_glob, base_conc=args.base_conc,
                             n_sweeps=args.dp_sweeps, burn=args.dp_burn, seed=args.seed,
                             init_z=seed_z, init_betas=seed_betas)
                dp_rows.append(dict(alpha=al, occ_K_mean=r["occ_K_mean"],
                                    occ_K_sd=r["occ_K_sd"]))
                print(f"  DP alpha={al:6.2f} -> occ_K={r['occ_K_mean']:.1f} "
                      f"+/- {r['occ_K_sd']:.1f}  (on {dp_G} contacts)", flush=True)
            rh = dp_gibbs(dp_corp, 1.0, dp_glob, base_conc=args.base_conc,
                          n_sweeps=args.dp_sweeps, burn=args.dp_burn, seed=args.seed,
                          sample_alpha=True, alpha_gamma=(2.0, 1.0),
                          init_z=seed_z, init_betas=seed_betas)
            results[wt]["dp"] = dict(n_dp_contacts=int(dp_G), alpha_grid=dp_rows,
                                     gamma_hyperprior=dict(occ_K_mean=rh["occ_K_mean"],
                                                           alpha_mean=rh["alpha_mean"]))
            print(f"  DP Gamma-hyperprior: occ_K={rh['occ_K_mean']:.1f} "
                  f"(inferred alpha_mean={rh['alpha_mean']:.2f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n# wrote {args.out}")


def overdisp(args):
    """The overdispersion<->K confound study: sweep K under several per-component
    concentration REGIMES (near-multinomial / intermediate / free / high-overdispersion),
    report K* (held-out & BIC-with-correct-param-count), the fitted concentrations,
    and the between-vs-within-archetype variance decomposition; on weighted AND
    unweighted counts; plus a DP occupied-K free-vs-near-multinomial inflation check."""
    z = np.load(args.corpus, allow_pickle=True)
    ptr = z["ptr"]; idx = z["idx"]; meta = z["meta"]; fam = meta[:, 0]
    G_all = len(ptr) - 1
    sel = np.sort(np.random.default_rng(args.seed).permutation(G_all)[:args.max_contacts]) \
        if args.max_contacts and args.max_contacts < G_all else np.arange(G_all)
    fam = fam[sel]
    regimes = [("near_multinomial", 1e4), ("intermediate", 300.0),
               ("free", None), ("high_overdispersion", 8.0)]
    results = {"corpus": args.corpus, "n_contacts_used": int(len(sel)),
               "K_list": list(args.K_list), "regimes": [r[0] for r in regimes]}
    for wt in ("weighted", "unweighted"):
        val = z["wval"] if wt == "weighted" else z["uval"]
        full = Corpus(ptr, idx, val).subset(sel)
        ufam = np.unique(fam); rng2 = np.random.default_rng(args.seed); rng2.shuffle(ufam)
        valfam = set(ufam[: int(len(ufam) * args.val_frac)].tolist())
        is_val = np.isin(fam, list(valfam))
        tr = full.subset(~is_val); va = full.subset(is_val)
        print(f"\n########## {wt.upper()} : {full.G:,} contacts (mean N={full.N.mean():.0f}) "
              f"##########", flush=True)
        results[wt] = {}
        for rname, B in regimes:
            ppc = NPAIR if B is None else NPAIR - 1
            rows = []
            for K in args.K_list:
                beta, pi, _, _ = fit_dmm(tr, K, seed=args.seed, max_iters=args.em_iters,
                                         inner=args.inner, fixed_B=B)
                va_ll, _ = polya_mixture_ll(va, beta, np.log(pi))
                tr_ll, _ = polya_mixture_ll(tr, beta, np.log(pi))
                bic, aic, p = bic_aic(-tr_ll, K, tr.G, params_per_comp=ppc)
                resp = polya_loglik_all(tr, beta) + np.log(pi)[None, :]
                resp = np.exp(resp - resp.max(1, keepdims=True))
                resp /= resp.sum(1, keepdims=True)
                dec = variance_decomp(tr, resp, beta)
                Bk = beta.sum(1)
                miw = float(sum(pi[k] * mi_of_joint(pair_mean_to_joint(beta[k] / beta[k].sum()))
                                for k in range(K)))
                rows.append(dict(K=K, heldout_ll_per_contact=va_ll / va.G, BIC=bic,
                                 fitted_B_med=float(np.median(Bk)),
                                 fitted_B_min=float(Bk.min()), fitted_B_max=float(Bk.max()),
                                 frac_between_raw=round(dec["frac_between_raw"], 4),
                                 frac_between_signal=round(dec["frac_between_signal"], 4),
                                 weighted_mean_MI=round(miw, 4)))
            K_ho = rows[int(np.argmax([r["heldout_ll_per_contact"] for r in rows]))]["K"]
            K_bic = rows[int(np.argmin([r["BIC"] for r in rows]))]["K"]
            results[wt][rname] = dict(fixed_B=B, K_heldout=int(K_ho), K_BIC=int(K_bic),
                                      sweep=rows)
            print(f"  [{rname:20s} B={str(B):>6}]  K*_heldout={K_ho:>3}  K*_BIC={K_bic:>3}  "
                  f"fittedB(med@Kmax)={rows[-1]['fitted_B_med']:.0f}  "
                  f"frac_between_signal(@Kmax)={rows[-1]['frac_between_signal']:.3f}  "
                  f"wMI={rows[-1]['weighted_mean_MI']:.3f}", flush=True)
            for r in rows:
                print(f"      K={r['K']:>3} ho/contact={r['heldout_ll_per_contact']:9.2f} "
                      f"BIC={r['BIC']:14.0f} Bmed={r['fitted_B_med']:8.0f} "
                      f"frac_btwn(sig)={r['frac_between_signal']:.3f} "
                      f"raw={r['frac_between_raw']:.3f} wMI={r['weighted_mean_MI']:.3f}", flush=True)

        # DP inflation: free vs near-multinomial refit
        if args.dp:
            dp_G = min(args.dp_contacts, full.G)
            dp_sel = np.sort(np.random.default_rng(args.seed + 1).permutation(full.G)[:dp_G])
            dp_c = full.subset(dp_sel)
            dp_glob = np.zeros(NPAIR); np.add.at(dp_glob, dp_c.idx, dp_c.val); dp_glob /= dp_glob.sum()
            dp_res = {}
            for al in args.dp_alpha:
                rf = dp_gibbs(dp_c, al, dp_glob, base_conc=50.0, n_sweeps=args.dp_sweeps,
                              burn=args.dp_burn, seed=args.seed)
                rm = dp_gibbs(dp_c, al, dp_glob, base_conc=1e4, n_sweeps=args.dp_sweeps,
                              burn=args.dp_burn, seed=args.seed, refit_fixed_B=1e4)
                dp_res[str(al)] = dict(free=rf["occ_K_mean"], near_multinomial=rm["occ_K_mean"])
                print(f"  DP alpha={al}: occ_K free(overdisp-fit)={rf['occ_K_mean']:.1f}  "
                      f"near-multinomial={rm['occ_K_mean']:.1f}  (on {dp_G} contacts)", flush=True)
            results[wt]["dp_inflation"] = dict(n_dp_contacts=int(dp_G), by_alpha=dp_res)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n# wrote {args.out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--tr-dir", default=DEFAULT_DATA)
    b.add_argument("--out", default="data/dpmm_column_pairs/counts.npz")
    b.add_argument("--max-fam", type=int, default=3000)
    b.add_argument("--max-seqs", type=int, default=512)
    b.add_argument("--theta", type=float, default=0.8)
    b.add_argument("--cb-max", type=float, default=8.0)
    b.add_argument("--min-sep", type=int, default=7)
    b.add_argument("--min-neff", type=float, default=25.0)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=build)

    f = sub.add_parser("fit")
    f.add_argument("--corpus", default="data/dpmm_column_pairs/counts.npz")
    f.add_argument("--out", default="results/dpmm_column_pairs/report.json")
    f.add_argument("--K-list", type=int, nargs="+",
                   default=[1, 2, 4, 6, 8, 12, 16, 20, 25, 30, 40, 50])
    f.add_argument("--em-iters", type=int, default=45)
    f.add_argument("--max-contacts", type=int, default=80000,
                   help="subsample this many contacts for the fit (0 = all); K* grows "
                        "only ~log(N) so a subsample answers the count question fast")
    f.add_argument("--inner", type=int, default=1)
    f.add_argument("--val-frac", type=float, default=0.2)
    f.add_argument("--prior-a", type=float, default=1.0)
    f.add_argument("--prior-b", type=float, default=0.0)
    f.add_argument("--dp", action="store_true", default=True)
    f.add_argument("--no-dp", dest="dp", action="store_false")
    f.add_argument("--dp-contacts", type=int, default=8000)
    f.add_argument("--dp-sweeps", type=int, default=40)
    f.add_argument("--dp-burn", type=int, default=18)
    f.add_argument("--dp-seed-K", type=int, default=15)
    f.add_argument("--alpha-grid", type=float, nargs="+",
                   default=[0.3, 1.0, 3.0, 10.0, 30.0])
    f.add_argument("--base-conc", type=float, default=50.0)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=fit)

    o = sub.add_parser("overdisp", help="overdispersion<->K regime study + variance split")
    o.add_argument("--corpus", default="data/dpmm_column_pairs/counts.npz")
    o.add_argument("--out", default="results/dpmm_column_pairs/overdisp.json")
    o.add_argument("--K-list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    o.add_argument("--em-iters", type=int, default=40)
    o.add_argument("--inner", type=int, default=1)
    o.add_argument("--max-contacts", type=int, default=30000)
    o.add_argument("--val-frac", type=float, default=0.2)
    o.add_argument("--dp", action="store_true", default=True)
    o.add_argument("--no-dp", dest="dp", action="store_false")
    o.add_argument("--dp-contacts", type=int, default=8000)
    o.add_argument("--dp-sweeps", type=int, default=40)
    o.add_argument("--dp-burn", type=int, default=18)
    o.add_argument("--dp-alpha", type=float, nargs="+", default=[1.0, 10.0])
    o.add_argument("--seed", type=int, default=0)
    o.set_defaults(func=overdisp)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
