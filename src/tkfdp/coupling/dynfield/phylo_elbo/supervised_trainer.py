"""Supervised enum400 dynfield training on the FIXED PDB contact partition.

Learns pi_archetype (K_a archetype profiles), the +Gamma+I field-rate bin
weights `w`, and `rho_chain`; holds fixed the DM class-concentration `alpha`,
the partition, `S`, `rho`, and the enum400 arch enumeration. The site class is
MARGINALIZED (collapsed), not sampled.

E-step: per contact cluster, per field-rate bin g, score the class grid
  (pair: top-N^2, singleton: all K_c) -> per-bin class-marginal M_C(g),
  field-rate responsibility r_C(g) prop w_g exp(M_C(g)).
M-step:
  (a) w_g <- mean_C r_C(g)                             (mixture EM, monotone)
  (b) rho_chain <- GEM line search on F_contacts       (accept-if-better)
  (c) pi_archetype <- collapsed per-archetype Holmes-Rubin Dirichlet update,
      responsibilities = per-column frozen-field archetype posterior; guarded.

Derivation: analysis/supervised_enum400_training.md. Model spec:
math-paper/appendix-tkfdp.tex ("Rate heterogeneity (+Gamma+I)").

Uses the mm forward by default (field_rate_discovery._score_specs_rates);
`scorer='exact'` is left as a hook for the caller (not wired into the fast
per-bin path here).
"""
from __future__ import annotations

import numpy as np

from . import field_rate_discovery as fd
from .marginal_scorer import _logsumexp


# --------------------------------------------------------------- E-step scoring

def _lse(v, axis=None):
    v = np.asarray(v, float)
    m = v.max(axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        out = np.log(np.sum(np.exp(v - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else float(out)


def score_pairs_perbin(ds, byfam, topN=8):
    """Per-pair (topN, topN, G) class-grid LL + (topN,) top class indices per
    column, per field-rate bin (NOT rate-marginalized). Mirrors
    fit_pdb_hyperparams.score_perbin but keeps the DiscoveryState scorer.

    Returns (recs, G, Kc) where each rec = dict(fam, i, j, grid (N,N,G),
    top_i, top_j)."""
    st = ds.state
    Kc = st.K_c
    G = len(ds.rates)
    mb1 = st.m_bucket_for(1)
    mb2 = st.m_bucket_for(2)
    fam_ids = [f.family_id for f in st.families]
    logw0 = ds.logw            # prior weights only used to pick top-N columns
    recs = []
    for fam, pairs in byfam.items():
        fi = fam_ids.index(fam)
        cols = sorted({c for ij in pairs for c in ij})
        s_specs, s_slot = [], {}
        for col in cols:
            for c in range(Kc):
                s_slot[(col, c)] = len(s_specs)
                cl = np.zeros(mb1, np.int32)
                cl[0] = c
                s_specs.append((fi, np.array([col], np.int32), cl, mb1))
        ll_s = fd._score_specs_rates(ds, s_specs)                     # (n, G)
        sing = {col: np.stack([ll_s[s_slot[(col, c)]] for c in range(Kc)])
                for col in cols}                                      # (K_c, G)
        sm = {col: _lse(sing[col] + logw0[None, :], axis=1) for col in cols}
        tops = {col: np.argsort(-sm[col])[:topN] for col in cols}
        p_specs, p_slot = [], {}
        for (i, j) in pairs:
            for a, ci in enumerate(tops[i]):
                for b, cj in enumerate(tops[j]):
                    p_slot[(i, j, a, b)] = len(p_specs)
                    cl = np.zeros(mb2, np.int32)
                    cl[0] = ci
                    cl[1] = cj
                    p_specs.append((fi, np.array([i, j], np.int32), cl, mb2))
        ll_p = fd._score_specs_rates(ds, p_specs)                     # (n, G)
        for (i, j) in pairs:
            grid = np.array([[ll_p[p_slot[(i, j, a, b)]] for b in range(topN)]
                             for a in range(topN)])                   # (N, N, G)
            recs.append(dict(fam=fam, i=i, j=j, grid=grid,
                             top_i=tops[i], top_j=tops[j]))
    return recs, G, Kc


def _static_topN_cols(ds, cols_by_fam, topN):
    """Top-N classes per column by the rho_chain-INDEPENDENT rate-0 static
    (K_a diagonal single-archetype log-liks), for arbitrary columns.
    `cols_by_fam`: {fam: iterable of cols}. Returns {(fam,col): top (topN,)}.
    ~K_a/K_c cheaper than the full K_c scan; shared by the pair and singleton
    E-step paths and reused across the rho_chain line search (field-frozen)."""
    from .corpus_state import _score_cluster_batch
    st = ds.state; K_a = st.pi_archetype.shape[0]; mb1 = st.m_bucket_for(1)
    fam_ids = [f.family_id for f in st.families]
    lr = np.log(np.asarray(st.rho, float))
    a0 = np.arange(K_a * K_a) // K_a; a1 = np.arange(K_a * K_a) % K_a
    order, specs = [], []
    for fam, cols in cols_by_fam.items():
        fi = fam_ids.index(fam)
        cols = sorted({int(c) for c in cols})
        if not cols:
            continue
        rb = fd._rate_bins(ds, fi, np.asarray(cols, np.int32))
        for pos, col in enumerate(cols):
            order.append((fam, int(col)))
            for k in range(K_a):
                cls = np.zeros(mb1, np.int32)
                cls[:1] = fd._aug(np.array([k * K_a + k], np.int32),
                                  np.array([rb[pos]]), ds.G_s)
                specs.append((fi, np.array([col], np.int32), cls, mb1))
    if not specs:
        return {}
    Ss = _score_cluster_batch(st, specs, tables=fd._tables_g(ds, 0)).reshape(len(order), K_a)
    tops = {}
    for i, key in enumerate(order):
        cll0 = np.logaddexp(lr[0] + Ss[i, a0], lr[1] + Ss[i, a1])   # rate-0 class LL
        tops[key] = np.argsort(-cll0)[:topN]
    return tops


def _static_topN(ds, byfam, topN):
    """Top-N classes per column of the contact pairs (rate-0 static). Thin
    wrapper over `_static_topN_cols` for the pair path."""
    cols_by_fam = {fam: {c for ij in pairs for c in ij} for fam, pairs in byfam.items()}
    return _static_topN_cols(ds, cols_by_fam, topN)


def score_perbin_fast(ds, byfam, topN=8, tops=None):
    """Fast score_pairs_perbin: rank classes by the rate-0 static (no 400-class
    scan), then full-score ONLY the top-N^2 pair grids per bin. `tops` (from
    _static_topN) is reused across the rho_chain line search. Returns (recs,G,Kc,tops)."""
    st = ds.state; Kc = st.K_c; G = len(ds.rates); mb2 = st.m_bucket_for(2)
    fam_ids = [f.family_id for f in st.families]
    if tops is None:
        tops = _static_topN(ds, byfam, topN)
    p_specs, p_slot = [], {}
    for fam, pairs in byfam.items():
        fi = fam_ids.index(fam)
        for (i, j) in pairs:
            for a, ci in enumerate(tops[(fam, i)]):
                for b, cj in enumerate(tops[(fam, j)]):
                    p_slot[(fam, i, j, a, b)] = len(p_specs)
                    cl = np.zeros(mb2, np.int32); cl[0] = ci; cl[1] = cj
                    p_specs.append((fi, np.array([i, j], np.int32), cl, mb2))
    ll_p = fd._score_specs_rates(ds, p_specs)                          # (n,G) all pairs
    recs = []
    for fam, pairs in byfam.items():
        for (i, j) in pairs:
            grid = np.array([[ll_p[p_slot[(fam, i, j, a, b)]] for b in range(topN)]
                             for a in range(topN)])
            recs.append(dict(fam=fam, i=i, j=j, grid=grid,
                             top_i=tops[(fam, i)], top_j=tops[(fam, j)]))
    return recs, G, Kc, tops


def singleton_cols(ds, byfam, frac=1.0, per_fam_cap=None, seed=0):
    """{fam: [cols]} of every column NOT in a contact pair, over the supervised
    families.  `frac`/`per_fam_cap` optionally subsample (default: ALL columns).
    Returns (cols_by_fam, n_total, n_covered) so callers can LOG coverage."""
    st = ds.state
    fam_ids = [f.family_id for f in st.families]
    rng = np.random.default_rng(seed)
    out = {}
    n_total = 0
    n_cov = 0
    for fam, pairs in byfam.items():
        fi = fam_ids.index(fam)
        L = st.families[fi].L
        contact = {c for ij in pairs for c in ij}
        cols = [c for c in range(L) if c not in contact]
        n_total += len(cols)
        if frac < 1.0 and cols:
            k = max(1, int(round(frac * len(cols))))
            cols = sorted(rng.choice(cols, size=min(k, len(cols)), replace=False).tolist())
        if per_fam_cap is not None and len(cols) > per_fam_cap:
            cols = sorted(rng.choice(cols, size=per_fam_cap, replace=False).tolist())
        n_cov += len(cols)
        if cols:
            out[fam] = cols
    return out, n_total, n_cov


def score_singletons_perbin(ds, cols_by_fam, topN=8, tops=None):
    """Per-column (topN, G) class-grid LL + (topN,) top class indices, per
    field-rate bin, for singleton (m=1) columns.

    Uses the SAME rate-0 static top-N pruning as the pair path (`_static_topN_cols`)
    -- scoring only the top-N classes per column instead of the full K_c=400 scan
    that used to dominate the E-step (K_c/topN ~ 50x fewer forward evals).  `tops`
    (from a shared `_static_topN_cols` call) is reused across the rho_chain line
    search.  Returns recs dict(fam, col, grid (topN,G), top (topN,), kind='singleton')."""
    st = ds.state
    mb1 = st.m_bucket_for(1)
    fam_ids = [f.family_id for f in st.families]
    if tops is None:
        tops = _static_topN_cols(ds, cols_by_fam, topN)
    s_specs, s_slot = [], {}
    for fam, cols in cols_by_fam.items():
        fi = fam_ids.index(fam)
        for col in cols:
            col = int(col)
            for a, c in enumerate(tops[(fam, col)]):
                s_slot[(fam, col, a)] = len(s_specs)
                cl = np.zeros(mb1, np.int32); cl[0] = int(c)
                s_specs.append((fi, np.array([col], np.int32), cl, mb1))
    ll_s = fd._score_specs_rates(ds, s_specs)                          # (n, G)
    recs = []
    for fam, cols in cols_by_fam.items():
        for col in cols:
            col = int(col)
            grid = np.stack([ll_s[s_slot[(fam, col, a)]] for a in range(topN)])  # (topN,G)
            recs.append(dict(fam=fam, col=col, grid=grid,
                             top=np.asarray(tops[(fam, col)]), kind="singleton"))
    return recs


def _class_logprior_grid(rec, alpha_log):
    """Class log-prior over a cluster's top-N grid.

    `alpha_log` may be: None (flat, cancels); a per-class array (K_c,) -> SEPARABLE
    prior la[c_i]+la[c_j]; or a DM-mixture object (has `.alpha (H,K_c)`, `.pi (H)`)
    -> the NON-SEPARABLE mixture predictive log P_DM, which is the whole point of the
    DM (it couples c_i,c_j through the shared component so it does NOT cancel against
    singletons). Pairs -> (N,N); singletons -> (N,)."""
    if alpha_log is None:
        return 0.0
    if hasattr(alpha_log, "alpha"):                       # DM mixture prior
        return _dm_logprior_grid(rec, alpha_log)
    la = np.asarray(alpha_log, float)
    if rec.get("kind") == "singleton":
        return la[rec["top"]]
    return la[rec["top_i"]][:, None] + la[rec["top_j"]][None, :]


def _dm_logprior_grid(rec, dm):
    """log P_DM over a cluster's top-N grid under DM mixture `dm` (alpha (H,K_c),
    pi (H)). Singleton (single draw): logsumexp_h[log pi_h + log a_h[c] - log A_h].
    Pair (double draw): logsumexp_h[log pi_h + log a_h[c_i]+log a_h[c_j] - 2 log A_h]
    -- the mixture couples c_i,c_j (non-separable)."""
    from scipy.special import logsumexp
    a = np.asarray(dm.alpha, float)
    la = np.log(a); lA = np.log(a.sum(1)); lpi = np.log(np.asarray(dm.pi, float) + 1e-300)
    if rec.get("kind") == "singleton":
        c = rec["top"].astype(int)                        # (N,)
        comp = lpi[:, None] + la[:, c] - lA[:, None]      # (H,N)
        return logsumexp(comp, axis=0)                    # (N,)
    ci = rec["top_i"].astype(int); cj = rec["top_j"].astype(int)
    comp = (lpi[:, None, None] + la[:, ci][:, :, None]
            + la[:, cj][:, None, :] - 2.0 * lA[:, None, None])   # (H,Ni,Nj)
    return logsumexp(comp, axis=0)                        # (Ni,Nj)


def make_dm(K_c, H=10, alpha_pi=5.0, alpha0=1.0, seed=0):
    """Fresh FREE DM mixture prior (dm_prior.DMPrior), symmetry-broken so the H
    components can differentiate (a symmetric init stays a single component)."""
    from .dm_prior import DMPrior
    dm = DMPrior(K_c, H=H, alpha_pi=alpha_pi, alpha0=alpha0)
    rng = np.random.default_rng(seed)
    dm.alpha = float(alpha0) * np.exp(0.6 * rng.standard_normal((H, K_c)))
    dm.pi = np.full(H, 1.0 / H)
    return dm


def make_swap_dm(K_c, K_a, conc=10.0, base=0.2, include_static=True):
    """STRUCTURED DM prior: one component per archetype-pair {A,B} (A<B), each with
    alpha concentrated on the two SWAP classes (A,B)=A*K_a+B and (B,A)=B*K_a+A, so a
    cluster drawn from it is a compensatory archetype swap A<->B. NO charge knowledge
    is baked in -- WHICH swaps coevolve (charge, Cys, ...) is discovered via which
    components the learned pi upweights. Optional static component favors the diagonal
    (A,A) classes. Intended with freeze_alpha=True (learn only pi = which swaps matter).

    Does NOT capture non-swap compensation (e.g. disulfide on/off release) -- that is
    a covarion, not a swap, and out of scope for this prior."""
    from .dm_prior import DMPrior
    pairs = [(A, B) for A in range(K_a) for B in range(A + 1, K_a)]   # C(K_a,2)
    H = len(pairs) + (1 if include_static else 0)
    dm = DMPrior(K_c, H=H, alpha_pi=5.0, alpha0=base)
    alpha = np.full((H, K_c), float(base))
    for h, (A, B) in enumerate(pairs):
        alpha[h, A * K_a + B] = conc
        alpha[h, B * K_a + A] = conc
    if include_static:
        for A in range(K_a):
            alpha[-1, A * K_a + A] = conc
    dm.alpha = alpha
    dm.pi = np.full(H, 1.0 / H)
    dm.swap_pairs = pairs
    dm.comp_labels = [f"swap({A},{B})" for (A, B) in pairs] + \
        (["static"] if include_static else [])
    return dm


def mstep_dm(recs, dm, logw, n_inner=3, floor=1e-3, freeze_alpha=False):
    """Soft-EM update of the DM mixture from the E-step class posteriors over pairs
    AND singletons -- the DM analog of the pairs+singletons HR step, run INSIDE the
    trainer loop (vs offline fit_dm_from_cache). Each cluster's per-bin grid is
    rate-marginalized with logw, then the mixture is refit by responsibility-weighted
    Minka. `freeze_alpha` (for the seeded swap DM) updates ONLY pi -- learning which
    seeded swap components the data favors, keeping the swap structure fixed. Returns
    dm (mutated)."""
    from scipy.special import digamma, logsumexp
    K_c = dm.alpha.shape[1]; H = dm.H
    clu = []                                              # (kind, class_idx, LL_grid)
    for r in recs:
        g = np.asarray(r["grid"], float)
        if r.get("kind") == "singleton":
            clu.append(("s", r["top"].astype(int), logsumexp(g + logw[None, :], axis=1)))
        else:
            clu.append(("p", (r["top_i"].astype(int), r["top_j"].astype(int)),
                        logsumexp(g + logw[None, None, :], axis=2)))
    n = max(len(clu), 1)
    for _ in range(n_inner):
        alpha = dm.alpha; pi = dm.pi
        la = np.log(alpha); A = alpha.sum(1); lA = np.log(A)
        dig_a = digamma(alpha); dig_A = digamma(A); lpi = np.log(pi + 1e-300)
        num = np.zeros((H, K_c)); den = np.zeros(H); Rsum = np.zeros(H)
        for kind, cidx, LL in clu:
            if kind == "s":
                c = cidx
                gp = la[:, c] - lA[:, None]                       # (H,N)
                joint = LL[None, :] + gp                          # (H,N)
                lm = logsumexp(joint, axis=1)                     # (H,)
                q = np.exp(joint - lm[:, None])                   # (H,N)
                nc = np.zeros((H, K_c)); np.add.at(nc.T, c, q.T)
            else:
                ci, cj = cidx
                gp = la[:, ci][:, :, None] + la[:, cj][:, None, :] - 2.0 * lA[:, None, None]
                joint = LL[None] + gp                             # (H,Ni,Nj)
                lm = logsumexp(joint.reshape(H, -1), axis=1)      # (H,)
                q = np.exp(joint - lm[:, None, None])             # (H,Ni,Nj)
                nc = np.zeros((H, K_c))
                np.add.at(nc.T, ci, q.sum(2).T); np.add.at(nc.T, cj, q.sum(1).T)
            lp = lpi + lm; mx = lp.max(); r = np.exp(lp - mx); r /= r.sum()
            Rsum += r
            m = nc.sum(1)
            num += r[:, None] * (digamma(nc + alpha) - dig_a)
            den += r * (digamma(m + A) - dig_A)
        dm.pi = Rsum / n; dm.pi = dm.pi / dm.pi.sum()
        if not freeze_alpha:
            for h in range(H):
                if den[h] > 1e-9:
                    dm.alpha[h] = np.maximum(alpha[h] * num[h] / den[h], floor)
    return dm


def cluster_marginals_perbin(recs, alpha_log=None):
    """Per-cluster per-bin class-marginal M_C(g), shape (n_clusters, G)."""
    G = recs[0]["grid"].shape[-1]
    M = np.zeros((len(recs), G))
    for i, r in enumerate(recs):
        lp = _class_logprior_grid(r, alpha_log)
        M[i] = _lse((r["grid"] + np.asarray(lp)[..., None]).reshape(-1, G), axis=0)
    return M


def field_rate_responsibilities(M, w):
    """r_C(g) prop w_g exp(M_C(g)); returns (n, G) responsibilities + (n,)
    per-cluster marginal log-lik lse_g(log w_g + M_C(g))."""
    logw = np.log(np.asarray(w, float) + 1e-300)
    lp = M + logw[None, :]
    cll = np.array([_logsumexp(row) for row in lp])                   # (n,)
    r = np.exp(lp - cll[:, None])
    return r, cll


def flip_prevalence(M, w):
    """Mean over clusters of phi(C) = 1 - r_C(0) (mass off the invariant bin)."""
    r, _ = field_rate_responsibilities(M, w)
    return float((1.0 - r[:, 0]).mean()), (1.0 - r[:, 0])


def corpus_F(M, w):
    """F = sum_C lse_g(log w_g + M_C(g)) -- the marginalized objective."""
    _, cll = field_rate_responsibilities(M, w)
    return float(cll.sum())


# ----------------------------------------------------------------- rho_chain

def rebuild_rate_kernels(ds):
    """Rebuild ds.rate_kernels for the current state.rho_chain (field beta/W;
    P_sub is rho_chain-independent). Cheap: G field_kernel_tables calls."""
    import jax.numpy as jnp
    from .tau_binning import field_kernel_tables
    st = ds.state
    ds.rate_kernels = []
    for r in ds.rates:
        beta, W = field_kernel_tables(st.bin_centers, st.rho,
                                      st.rho_chain * float(r))
        ds.rate_kernels.append((jnp.asarray(np.asarray(beta)),
                                jnp.asarray(np.asarray(W))))


def mstep_rho_chain(ds, byfam, w, topN, alpha_log=None,
                    grid=(0.5, 0.7, 1.0, 1.4, 2.0), sing_cols=None):
    """GEM line search: re-score clusters at rho_chain * m for m in grid, keep
    the argmax of F. Returns (best_rho_chain, best_F, recs_p_at_best, recs_s_at_best)
    -- the pair and singleton recs at the winning rho, so the caller can reuse them
    instead of re-scoring (they are exactly what a re-score at best rho would give).

    When `sing_cols` ({fam:[cols]}) is given, singleton (m=1) clusters are scored
    alongside the contact pairs and F is over pairs AND singletons (EM-consistent
    with the HR)."""
    st = ds.state
    rc0 = float(st.rho_chain)
    best = (rc0, -np.inf, None, None)
    tops = _static_topN(ds, byfam, topN)          # rho_chain-independent; pick once
    tops_s = _static_topN_cols(ds, sing_cols, topN) if sing_cols else None  # ditto, once
    for m in grid:
        st.rho_chain = rc0 * float(m)
        rebuild_rate_kernels(ds)
        recs_p, _, _, _ = score_perbin_fast(ds, byfam, topN=topN, tops=tops)
        recs_s = score_singletons_perbin(ds, sing_cols, topN=topN, tops=tops_s) if sing_cols else []
        M = cluster_marginals_perbin(recs_p + recs_s, alpha_log)
        F = corpus_F(M, w)
        if F > best[1]:
            best = (st.rho_chain, F, recs_p, recs_s)
    st.rho_chain = best[0]
    rebuild_rate_kernels(ds)
    return best


# ----------------------------------------------- pi_archetype (HR Dirichlet)

def _all_columns(ds, byfam):
    """{fi: (n_col,) int cols} of every column in each family in byfam (the
    supervised partition covers all columns: contacts as pairs, rest as
    singletons -> pi_archetype accumulates over all columns)."""
    fam_ids = [f.family_id for f in ds.state.families]
    out = {}
    for fam in byfam:
        fi = fam_ids.index(fam)
        out[fi] = np.arange(ds.state.families[fi].L, dtype=np.int32)
    return out


def column_arch_responsibilities(ds, cols_by_fi, arch_prior_log=None,
                                  b_chunk=4096):
    """Per-column frozen-field archetype posterior rho_s(k), shape (n_cols, K_a),
    via ONE batched invariant-bin (rate-0) single-archetype forward per column.

    rho_s(k) = softmax_k( Ss[k] + arch_prior_log[k] ), Ss[k] = static single-col
    log-lik of the column under archetype k (enum400 diagonal class (k,k)).
    Returns (order list of (fi, col), rho (n_cols, K_a))."""
    st = ds.state
    K_a = st.pi_archetype.shape[0]
    mb1 = st.m_bucket_for(1)
    order = []
    specs = []
    for fi, cols in cols_by_fi.items():
        rb = fd._rate_bins(ds, fi, np.asarray(cols, np.int32))
        for pos, col in enumerate(np.asarray(cols).tolist()):
            order.append((fi, int(col)))
            for k in range(K_a):
                cls = np.zeros(mb1, np.int32)
                cls[:1] = fd._aug(np.array([k * K_a + k], np.int32),
                                  np.array([rb[pos]]), ds.G_s)
                specs.append((fi, np.array([col], np.int32), cls, mb1))
    from .corpus_state import _score_cluster_batch
    ll = _score_cluster_batch(st, specs, tables=fd._tables_g(ds, 0))   # (n*K_a,)
    Ss = ll.reshape(len(order), K_a)
    if arch_prior_log is not None:
        Ss = Ss + np.asarray(arch_prior_log)[None, :]
    Ss = Ss - Ss.max(1, keepdims=True)
    rho = np.exp(Ss)
    rho /= rho.sum(1, keepdims=True)
    return order, rho


def _f81_Q(pi, S):
    Q = (S - np.diag(np.diag(S))) * np.asarray(pi)[None, :]
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def accumulate_hr_per_archetype(ds, order, rho, rng, S, resp_thresh=1e-3,
                                n_history=1, chunk=16384):
    """Responsibility-weighted Holmes-Rubin dwell + destination counts per
    archetype, over LG08-sampled branch histories.

    Returns (dwell_total (K_a, A), real_counts (K_a, A)).
    order/rho from column_arch_responsibilities."""
    import jax.numpy as jnp
    from tkfdp.eta_site import hr_batch_jit
    from tkfdp.pfam_data import load_clv_family
    from .corpus_state import _cluster_columns_by_id  # noqa: F401 (parity)

    st = ds.state
    K_a = st.pi_archetype.shape[0]
    A = st.pi_archetype.shape[1]
    # index columns by family
    cols_by_fi = {}
    for idx, (fi, col) in enumerate(order):
        cols_by_fi.setdefault(fi, []).append((idx, col))

    # gather per-(archetype) endpoint arrays weighted by rho
    a_by_k = [[] for _ in range(K_a)]
    b_by_k = [[] for _ in range(K_a)]
    t_by_k = [[] for _ in range(K_a)]
    w_by_k = [[] for _ in range(K_a)]

    fam_ids = [f.family_id for f in st.families]
    for fi, entries in cols_by_fi.items():
        # load the raw CLV bundle (histories) for this family
        path = _clv_path_for(fam_ids[fi])
        if path is None:
            continue
        fc = load_clv_family(path)
        for _h in range(n_history):
            X = fc.sample_history(rng)                       # (n_nodes, L)
            fch = fc.extract_branch_cherries(X)              # per-branch endpoints
            aa_a = np.minimum(fch.aa_a.astype(np.int64), 19)  # (n_br, L)
            aa_b = np.minimum(fch.aa_b.astype(np.int64), 19)
            tau = np.asarray(fch.tau, np.float64)             # (n_br,)
            valid = (fch.aa_a < 20) & (fch.aa_b < 20)         # (n_br, L)
            for idx, col in entries:
                # col here is the CLV/raw column index == dynfield column index
                if col >= aa_a.shape[1]:
                    continue
                vb = valid[:, col]
                if not vb.any():
                    continue
                rvec = rho[idx]                              # (K_a,)
                a_c = aa_a[vb, col]
                b_c = aa_b[vb, col]
                t_c = tau[vb]
                for k in range(K_a):
                    wk = float(rvec[k])
                    if wk < resp_thresh:
                        continue
                    a_by_k[k].append(a_c)
                    b_by_k[k].append(b_c)
                    t_by_k[k].append(t_c)
                    w_by_k[k].append(np.full(a_c.shape[0], wk))

    dwell_total = np.zeros((K_a, A))
    real_counts = np.zeros((K_a, A))
    for k in range(K_a):
        if not a_by_k[k]:
            continue
        a_full = np.concatenate(a_by_k[k])
        b_full = np.concatenate(b_by_k[k])
        t_full = np.concatenate(t_by_k[k]).astype(np.float64)
        w_full = np.concatenate(w_by_k[k])
        Q = _f81_Q(st.pi_archetype[k], S)
        ndQ = -np.diag(Q)
        Q_j = jnp.asarray(Q)
        pi_j = jnp.asarray(st.pi_archetype[k])
        ndQ_j = jnp.asarray(ndQ)
        B = a_full.shape[0]
        for s0 in range(0, B, chunk):
            s1 = min(s0 + chunk, B)
            ax = a_full[s0:s1]
            bx = b_full[s0:s1]
            tx = t_full[s0:s1]
            wx = w_full[s0:s1]
            n_real = ax.shape[0]
            if n_real < chunk:
                pad = chunk - n_real
                ax = np.concatenate([ax, np.zeros(pad, ax.dtype)])
                bx = np.concatenate([bx, np.zeros(pad, bx.dtype)])
                tx = np.concatenate([tx, np.full(pad, 0.1)])
            _, _, dwell_c = hr_batch_jit(Q_j, pi_j, ndQ_j,
                                         jnp.asarray(ax), jnp.asarray(bx),
                                         jnp.asarray(tx))
            dwell = np.asarray(dwell_c)[:n_real]             # (n_real, A)
            dwell_total[k] += (dwell * wx[:, None]).sum(0)
            # destination counts weighted by rho
            np.add.at(real_counts[k], bx[:n_real], wx)
    return dwell_total, real_counts


def _clv_tree(fam):
    """(parent, tau, leaf_msa, n_leaves, n_nodes, L) for a family's CLV bundle,
    cached. parent[root] forced to -1. Returns None if no CLV file."""
    global _CLV_TREE_CACHE
    try:
        _CLV_TREE_CACHE
    except NameError:
        _CLV_TREE_CACHE = {}
    if fam in _CLV_TREE_CACHE:
        return _CLV_TREE_CACHE[fam]
    from tkfdp.pfam_data import load_clv_family
    path = _clv_path_for(fam)
    if path is None:
        _CLV_TREE_CACHE[fam] = None
        return None
    fc = load_clv_family(path)
    parent = np.asarray(fc.parent).astype(np.int64).copy()
    parent[int(fc.root_id)] = -1
    out = (parent, np.asarray(fc.tau, np.float64),
           np.asarray(fc.leaf_msa), int(fc.n_leaves), int(fc.n_nodes), int(fc.L))
    _CLV_TREE_CACHE[fam] = out
    return out


def _gpu_visible():
    """True if JAX sees a GPU (used to pick the exact-HR backend)."""
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def exact_hr_per_archetype(ds, recs, w, S, alpha_log=None, q_thresh=1e-3,
                           hr_backend="auto", b_chunk=16):
    """EXACT factored per-archetype Holmes-Rubin over the contact-pair clusters,
    replacing the Monte-Carlo / rate-0 approximation of
    `accumulate_hr_per_archetype`.

    `hr_backend`: 'jax' (batched over configs; fast), 'numpy' (reference), or
    'auto' (jax when a GPU is visible, else numpy).  Both are numerically exact;
    the JAX path is validated against the numpy path to ~1e-8.

    For each pair cluster (E-step `rec` with grid (N,N,G) over top-N x top-N
    classes x G field-rate bins), form the TRUE joint posterior
        q_C(a,b,g) prop w_g exp(grid[a,b,g] + class_logprior[a,b])
    -- exactly the field-model posterior that defines the objective F (via
    `cluster_marginals_perbin` + `field_rate_responsibilities`).  For each
    config above `q_thresh`, run the exact factored cap-2 pair HR
    (`cluster_hr_exact.pair_tree_hr`, validated to match the dense 800-state tree
    HR) on the family's real CLV tree at rho_chain * rate_g, mapping each class to
    its per-field archetypes via arch_assignment, and accumulate

        dwell_total[k] += q * T^k          real_counts[k] += q * sum_x N^k[x, .].

    Returns (dwell_total (K_a, A), real_counts (K_a, A)) -- same signature as
    `accumulate_hr_per_archetype`, but EXACT (no sampled history, no rate-0
    responsibilities), so the pi_archetype M-step is a genuine EM step on F.
    """
    if hr_backend == "auto":
        hr_backend = "jax" if _gpu_visible() else "numpy"
    if hr_backend == "jax":
        return exact_hr_per_archetype_jax(ds, recs, w, S, alpha_log, q_thresh,
                                          b_chunk=b_chunk)
    from . import cluster_hr_exact as X
    st = ds.state
    K_a = st.pi_archetype.shape[0]
    A = st.pi_archetype.shape[1]
    aa = st.arch_assignment                              # (K_c, L) class -> arch@theta
    pi_arch = st.pi_archetype
    rho = np.asarray(st.rho, float)
    rho_chain = float(st.rho_chain)
    rates = np.asarray(ds.rates, float)
    logw = np.log(np.asarray(w, float) + 1e-300)

    shared = X.SharedArchEig(pi_arch, S, rho, A)         # per-M-step eig cache
    dwell_total = np.zeros((K_a, A))
    real_counts = np.zeros((K_a, A))

    for rec in recs:
        fam = rec["fam"]
        tree = _clv_tree(fam)
        if tree is None:
            continue
        parent, tau, leaf_msa, n_leaves, n_nodes, Lcol = tree
        if rec.get("kind") == "singleton":
            col = int(rec["col"])
            if col >= Lcol:
                continue
            grid = np.asarray(rec["grid"], float)       # (topN, G)
            top = np.asarray(rec["top"])
            lp = _class_logprior_grid(rec, alpha_log)
            logq = grid + np.asarray(lp)[..., None] + logw[None, :]
            logq = logq - logq.max()
            q = np.exp(logq); q /= q.sum()
            xcol = np.full(n_nodes, -1, np.int64)
            xcol[:n_leaves] = leaf_msa[:, col]
            Nt, G = q.shape
            for a in range(Nt):
                for g in range(G):
                    w_cfg = float(q[a, g])
                    if w_cfg < q_thresh:
                        continue
                    a_n = aa[int(top[a])]
                    rc = rho_chain * float(rates[g])
                    Nk, Tk, _ = X.single_tree_hr(parent, tau, xcol, a_n,
                                                 pi_arch, S, rho, rc, A=A)
                    dwell_total += w_cfg * Tk
                    real_counts += w_cfg * Nk.sum(axis=1)
            continue
        i = int(rec["i"]); j = int(rec["j"])
        grid = np.asarray(rec["grid"], float)           # (N, N, G)
        top_i = np.asarray(rec["top_i"]); top_j = np.asarray(rec["top_j"])
        if i >= Lcol or j >= Lcol:
            continue
        # joint posterior q(a,b,g)
        lp = _class_logprior_grid(rec, alpha_log)
        logq = grid + np.asarray(lp)[..., None] + logw[None, None, :]
        logq = logq - logq.max()
        q = np.exp(logq); q /= q.sum()
        # residue columns on the tree (leaves observe leaf_msa; else uninformative)
        xcol = np.full(n_nodes, -1, np.int64); ycol = np.full(n_nodes, -1, np.int64)
        xcol[:n_leaves] = leaf_msa[:, i]
        ycol[:n_leaves] = leaf_msa[:, j]
        N_i, N_j, G = q.shape
        for a in range(N_i):
            for b in range(N_j):
                for g in range(G):
                    w_cfg = float(q[a, b, g])
                    if w_cfg < q_thresh:
                        continue
                    a1 = aa[int(top_i[a])]              # (L,) arch per field, col i
                    a2 = aa[int(top_j[b])]              # (L,) arch per field, col j
                    rc = rho_chain * float(rates[g])
                    Nk, Tk, _ = X.pair_tree_hr(parent, tau, xcol, ycol, a1, a2,
                                               pi_arch, S, rho, rc, A=A,
                                               shared=shared, want_jumps=False)
                    dwell_total += w_cfg * Tk
                    real_counts += w_cfg * Nk.sum(axis=1)   # sum over source -> dest
    return dwell_total, real_counts


def exact_hr_per_archetype_jax(ds, recs, w, S, alpha_log=None, q_thresh=1e-3,
                               b_chunk=16):
    """JAX-batched drop-in for `exact_hr_per_archetype`.

    Numerically exact (matches the numpy factored cap-2 HR to ~1e-8; see
    `analysis/scripts/validate_cluster_hr_exact.py`), but batches the exact HR
    over the (class-pair a,b) x (rate-bin g) CONFIG axis -- the dominant cost --
    and reuses the corpus-wide per-archetype eigendecomposition across every
    config.  Configs of a family are further batched into one vmapped call
    (they share the family tree topology + leaf residues).

    Returns (dwell_total (K_a, A), real_counts (K_a, A)) -- same as the numpy
    path.  Skips families whose CLV tree is unavailable.

    Cross-family batching: configs from EVERY family are pooled and dispatched by
    geomspaced size bin, so XLA compiles O(#size-bins) executables (reused across
    families and M-steps), NOT O(#families).  Configs are split into no-jump
    (rho_chain_eff==0, invariant bin) and jump sub-batches so the former skip the
    40-state Delta>=1 path entirely."""
    from . import cluster_hr_jax as XJ
    st = ds.state
    K_a = st.pi_archetype.shape[0]
    A = st.pi_archetype.shape[1]
    aa = st.arch_assignment
    pi_arch = st.pi_archetype
    rho = np.asarray(st.rho, float)
    rho_chain = float(st.rho_chain)
    rates = np.asarray(ds.rates, float)
    logw = np.log(np.asarray(w, float) + 1e-300)

    shared = XJ.make_shared(pi_arch, S, rho)
    dwell_total = np.zeros((K_a, A))
    real_counts = np.zeros((K_a, A))

    byfam = {}
    for rec in recs:
        byfam.setdefault(rec["fam"], []).append(rec)

    # per-family DENSE postorder-flat tree (node-count bucketed), cached across
    # families in this call.  Both singleton and pair HR consume this layout via
    # the exact flat kernels (no D x N_max padding).
    _fpt = {}

    def fptree_for(fam, tree):
        if fam not in _fpt:
            parent, tau = tree[0], tree[1]
            _fpt[fam] = XJ.build_flat_ptree(parent, tau, pad=True)
        return _fpt[fam]

    # ---- pool configs across ALL families ----
    p_pt, p_lo, p_a1, p_a2, p_rc, p_q = [], [], [], [], [], []
    s_pt, s_lo, s_an, s_rc, s_q = [], [], [], [], []
    for fam, frecs in byfam.items():
        tree = _clv_tree(fam)
        if tree is None:
            continue
        parent, tau, leaf_msa, n_leaves, n_nodes, Lcol = tree
        fptree = fptree_for(fam, tree)
        for rec in frecs:
            if rec.get("kind") == "singleton":
                col = int(rec["col"])
                if col >= Lcol:
                    continue
                grid = np.asarray(rec["grid"], float)          # (topN, G)
                top = np.asarray(rec["top"])
                lp = _class_logprior_grid(rec, alpha_log)
                logq = grid + np.asarray(lp)[..., None] + logw[None, :]
                logq = logq - logq.max(); q = np.exp(logq); q /= q.sum()
                xcol = np.full(n_nodes, -1, np.int64); xcol[:n_leaves] = leaf_msa[:, col]
                lo = XJ.flat_leaf_obs_single(fptree, xcol)
                Nt, G = q.shape
                for a in range(Nt):
                    for g in range(G):
                        wcfg = float(q[a, g])
                        if wcfg < q_thresh:
                            continue
                        s_pt.append(fptree); s_lo.append(lo)
                        s_an.append(aa[int(top[a])])
                        s_rc.append(rho_chain * float(rates[g])); s_q.append(wcfg)
            else:
                i = int(rec["i"]); j = int(rec["j"])
                if i >= Lcol or j >= Lcol:
                    continue
                grid = np.asarray(rec["grid"], float)          # (N,N,G)
                top_i = np.asarray(rec["top_i"]); top_j = np.asarray(rec["top_j"])
                lp = _class_logprior_grid(rec, alpha_log)
                logq = grid + np.asarray(lp)[..., None] + logw[None, None, :]
                logq = logq - logq.max(); q = np.exp(logq); q /= q.sum()
                xcol = np.full(n_nodes, -1, np.int64); ycol = np.full(n_nodes, -1, np.int64)
                xcol[:n_leaves] = leaf_msa[:, i]; ycol[:n_leaves] = leaf_msa[:, j]
                lo = XJ.flat_leaf_obs_pair(fptree, xcol, ycol)
                Ni, Nj, G = q.shape
                for a in range(Ni):
                    for b in range(Nj):
                        for g in range(G):
                            wcfg = float(q[a, b, g])
                            if wcfg < q_thresh:
                                continue
                            p_pt.append(fptree); p_lo.append(lo)
                            p_a1.append(aa[int(top_i[a])]); p_a2.append(aa[int(top_j[b])])
                            p_rc.append(rho_chain * float(rates[g])); p_q.append(wcfg)

    # ---- pair HR (split no-jump / jump), bin-bucketed across families ----
    if p_pt:
        p_rc = np.asarray(p_rc, np.float64); p_q = np.asarray(p_q, np.float64)
        for mask, d1 in ((p_rc <= 0, False), (p_rc > 0, True)):
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                continue
            Nk, Tk, _ = XJ.pair_tree_hr_bucketed_flat(
                [p_pt[i] for i in idx], [p_lo[i] for i in idx],
                [p_a1[i] for i in idx], [p_a2[i] for i in idx], p_rc[idx],
                shared, want_jumps=False, delta1_enabled=d1,
                b_chunk=max(b_chunk, 256))
            qm = p_q[idx]
            dwell_total += np.einsum('b,bka->ka', qm, Tk)
            real_counts += np.einsum('b,bka->ka', qm, Nk.sum(axis=2))

    # ---- singleton HR, dense postorder-flat, bin-bucketed across families ----
    if s_pt:
        s_rc = np.asarray(s_rc, np.float64); s_q = np.asarray(s_q, np.float64)
        Nks, Tks, _ = XJ.single_tree_hr_bucketed_flat(
            s_pt, s_lo, s_an, s_rc, shared, want_jumps=False,
            b_chunk=max(b_chunk, 512))
        dwell_total += np.einsum('b,bka->ka', s_q, Tks)
        real_counts += np.einsum('b,bka->ka', s_q, Nks.sum(axis=2))
    return dwell_total, real_counts


def mstep_pi_archetype(ds, dwell_total, real_counts, S, kappa_pi, pi_bar, sparse=False):
    """Dirichlet secret-destination update, K_c -> K_a. Returns a candidate
    pi_archetype (K_a, A); caller guards on F. `sparse`=True uses the clamped MAP
    mode (for a sparsity-inducing prior_alpha<1) instead of the posterior mean."""
    from tkfdp import svi
    K_a = ds.state.pi_archetype.shape[0]
    return svi.update_pi_class(K_a, ds.state.pi_archetype.copy(),
                               dwell_total, real_counts, S, kappa_pi, pi_bar,
                               sparse=sparse)


def apply_pi_archetype(ds, pi_new, rates, weights):
    """Set pi_archetype and rebuild all derived tables (state + discovery)."""
    st = ds.state
    st.pi_archetype = np.ascontiguousarray(pi_new, np.float64)
    st._p_sub_by_arch = None                                  # invalidate cache
    st.refresh_pi_field()
    new_ds = fd.build_discovery_state(st, rates, weights)
    new_ds.enum400 = True
    return new_ds


# ------------------------------------------------------------------- helpers

_CLV_DIR = None


def set_clv_dir(path):
    global _CLV_DIR
    from pathlib import Path
    _CLV_DIR = Path(path)


def _clv_path_for(fam):
    from pathlib import Path
    if _CLV_DIR is None:
        return None
    p = _CLV_DIR / f"{fam}.npz"
    return str(p) if p.exists() else None


def archetype_charge(pi_arch, acid=(2, 3), base=(8, 14, 6)):
    """Net basic-minus-acidic charge per archetype (D,E acidic; K,R,H basic in
    the ACDEFGHIKLMNPQRSTVWY alphabet -- matches analysis/scripts/dynfield_metrics).
    Sanity: acidic archetypes stay < 0, basic archetypes stay > 0."""
    pi_arch = np.asarray(pi_arch)
    return pi_arch[:, list(base)].sum(1) - pi_arch[:, list(acid)].sum(1)
