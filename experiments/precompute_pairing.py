"""Stage A of the precompute pairing pipeline (analysis/precompute_pairing_pipeline.md):
the class-free, bias-corrected MI prefilter over candidate column pairs.

Raw plug-in MI is dominated by finite-sample bias (conserved / many-state columns
get spurious MI), so it is a poor prefilter (the real flip sits at only top-16%).
Two corrections were tried:
  * APC (average product correction) -- FAILS at n~128: its top pairs are pure
    noise (perm-z ~ 0) and the real flip stays at top-18%. APC is a large-MSA (DCA)
    trick; it does not correct the per-pair sampling bias at this depth. Kept below
    (`apc`) only as a documented dead-end.
  * perm -- permutation z-score `miz` (shuffle each column's leaves, preserving its
    marginal + n so the same bias). WORKS: the real flip jumps to top-2%. Vectorized
    over columns (`perm_null_z`) so one null draw scores every pair at once
    (~19 s/family, K=100). This is the prefilter.

Operating point (z* sweep over confirmed-flip families, `--sweep`): z*=3.0 retains
92% of confirmed flips at 9% shortlist; z*=2.5 retains 96% at 11%. ~10x candidate
reduction. NOTE the recall is <100% -- some confirmed flips have genuinely low
covariation (perm-z down to ~1.3); log the recall at the chosen z* (no silent caps).

The prefilter is computed from raw leaf residues only -- no model, no classes, no
arch map -- so it never needs recomputing.

CLI:
  # z* recall sweep on families that carry confirmed flips (pick the operating pt)
  python experiments/precompute_pairing.py --sweep
  # write shortlists for a split
  python experiments/precompute_pairing.py --shortlist --z-star 3.0 --out-dir data/pairing_cache
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

GAP = 20
CLV_DIR = Path("data/pfam_processed_clv_top1000_thin128")
CONFIRMED = Path("data/pdb_partition_clv_top1000_sifts/confirmed_flips.json")


def _onehot(msa):
    """(n,L) int8 msa -> (oh (n,L,20) float, notgap (n,L) bool). Gaps zeroed."""
    n, L = msa.shape
    oh = np.zeros((n, L, 20), np.float32)
    ng = msa != GAP
    r, c = np.where(ng)
    oh[r, c, msa[ng]] = 1.0
    return oh, ng


def all_pairs_mi(msa):
    """Vectorized plug-in MI (nats) for every column pair.
    Returns MI (L,L), n_ij (L,L) pair sample sizes, kstates (L,) observed-states."""
    n, L = msa.shape
    oh, ng = _onehot(msa)                                   # (n,L,20),(n,L)
    ngf = ng.astype(np.float32)
    n_ij = ngf.T @ ngf                                      # (L,L) both-non-gap counts
    J = np.einsum('nia,njb->ijab', oh, oh, optimize=True)   # (L,L,20,20) joint counts
    with np.errstate(divide='ignore', invalid='ignore'):
        nz = np.maximum(n_ij, 1.0)[:, :, None, None]
        P = J / nz                                          # joint prob per pair
        Pi = P.sum(3, keepdims=True)                        # marg over j-residue (L,L,20,1)
        Pj = P.sum(2, keepdims=True)                        # marg over i-residue (L,L,1,20)
        ratio = P / (Pi * Pj)
        term = np.where(P > 0, P * np.log(np.where(ratio > 0, ratio, 1.0)), 0.0)
        MI = term.sum((2, 3))
    MI[n_ij < 5] = 0.0
    np.fill_diagonal(MI, 0.0)
    kstates = np.array([len(np.unique(msa[ng[:, c], c])) for c in range(L)])
    return MI, n_ij, kstates


def apc(MI):
    """Average-product-corrected MI: MI - (mean_i * mean_j)/mean_all. Off-diagonal
    background subtracted; a cheap analytic bias/conservation correction."""
    L = MI.shape[0]
    off = ~np.eye(L, dtype=bool)
    col_mean = np.array([MI[i, off[i]].mean() for i in range(L)])
    tot = MI[off].mean()
    A = MI - np.outer(col_mean, col_mean) / max(tot, 1e-12)
    np.fill_diagonal(A, 0.0)
    return A


def _mi_single(a, b):
    """plug-in MI (nats) for two 1-D residue arrays (already non-gap-aligned)."""
    n = len(a)
    if n < 5:
        return 0.0
    P = np.zeros((20, 20))
    np.add.at(P, (a, b), 1.0)
    P /= n
    pi = P.sum(1); pj = P.sum(0)
    nz = P > 0
    return float((P[nz] * np.log(P[nz] / (pi[:, None] * pj[None, :])[nz])).sum())


def miz_perm(msa, i, j, K=200, seed=0):
    """Permutation MI z-score for one pair (the validated `miz`)."""
    a, b = msa[:, i], msa[:, j]
    ok = (a != GAP) & (b != GAP)
    a, b = a[ok], b[ok]
    obs = _mi_single(a, b)
    rng = np.random.default_rng(seed)
    null = np.array([_mi_single(a, rng.permutation(b)) for _ in range(K)])
    return obs, (obs - null.mean()) / (null.std() + 1e-9)


def perm_null_z(msa, K=100, seed=0):
    """Vectorized permutation z-score for ALL pairs. Each null draw shuffles every
    column independently (preserving each column's marginal + gap count), then
    reuses all_pairs_mi -- one draw scores the whole matrix at once. Returns the
    (L,L) z matrix. APC fails here (large-MSA trick); this is the gold-standard
    `miz` for every pair, made affordable by batching the null over columns."""
    n, L = msa.shape
    obs, _, _ = all_pairs_mi(msa)
    rng = np.random.default_rng(seed)
    mean = np.zeros((L, L)); M2 = np.zeros((L, L))
    for k in range(1, K + 1):
        sh = np.empty_like(msa)
        for c in range(L):                                 # independent column shuffle
            sh[:, c] = msa[rng.permutation(n), c]
        mi_k, _, _ = all_pairs_mi(sh)
        d = mi_k - mean; mean += d / k; M2 += d * (mi_k - mean)
    std = np.sqrt(M2 / max(K - 1, 1))
    z = (obs - mean) / (std + 1e-9)
    np.fill_diagonal(z, 0.0)
    return z


def _run_families():
    idx = json.loads((CLV_DIR / "index.json").read_text())["families"]
    cf = json.load(open(CONFIRMED))
    fams = sorted({r["family"] for r in cf} & set(idx))
    return fams, cf


def sweep(z_stars, K=100, n_families=None, seed=0):
    """z* operating-point sweep: confirmed-flip recall vs shortlist fraction."""
    fams, cf = _run_families()
    if n_families:
        fams = fams[:n_families]
    by_fam = {}
    for r in cf:
        by_fam.setdefault(r["family"], []).append((int(r["i"]), int(r["j"])))
    flip_z, all_frac = [], []                     # per confirmed flip; per family
    n_flip = n_pairs = 0
    for fi, fam in enumerate(fams):
        p = CLV_DIR / f"{fam}.npz"
        if not p.exists():
            continue
        msa = np.load(p)["leaf_msa"]; L = msa.shape[1]
        Z = perm_null_z(msa, K=K, seed=seed)
        off = ~np.eye(L, dtype=bool)
        zvals = Z[off]                            # 2x each pair; fine for fractions
        for (i, j) in by_fam.get(fam, []):
            if i < L and j < L:
                flip_z.append(float(Z[i, j])); n_flip += 1
        all_frac.append((fam, zvals))
        n_pairs += L * (L - 1) // 2
        print(f"  [{fi+1}/{len(fams)}] {fam}: L={L}, {len(by_fam.get(fam,[]))} flips", flush=True)
    flip_z = np.array(flip_z)
    print(f"\n# {n_flip} confirmed flips over {len(all_frac)} families ({n_pairs} pairs); K={K}")
    print(f"# confirmed-flip perm-z: median={np.median(flip_z):.2f} "
          f"p10={np.percentile(flip_z,10):.2f} min={flip_z.min():.2f}")
    print(f"\n{'z*':>5} {'flip_recall':>12} {'shortlist_frac':>15} {'~pairs_kept/fam':>16}")
    for zs in z_stars:
        recall = float((flip_z > zs).mean())
        fracs = [float((zv > zs).mean()) for _, zv in all_frac]
        mean_frac = float(np.mean(fracs))
        avg_pairs = np.mean([len(zv) // 2 for _, zv in all_frac])
        print(f"{zs:5.1f} {recall:12.3f} {mean_frac:15.4f} {mean_frac*avg_pairs:16.0f}")


# --------------------------------------------------------------------------
# Stage B: exact pair-evidence cache (the 800-state peel over top-N^2 classes).
# Ports the validated exact peel (shared expm(Q_arch,tau) cache) with GENERAL
# arch indexing (class -> arch_assignment[class]). Numpy reference; the hot loop
# (pairs x N^2 x rates of the pair peel) is the piece to move to batched JAX/GPU.
# --------------------------------------------------------------------------
import sys as _sys                                                # noqa: E402
from collections import defaultdict as _dd                        # noqa: E402


def _archetypes():
    _sys.path.insert(0, str(Path.home() / "tkf-mixdom" / "python"))
    from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c20
    P = np.asarray(le_gascuel_c20()[0], float)
    return P / P.sum(1, keepdims=True)


def _lse(v):
    v = np.asarray(v, float); m = v.max()
    return float(np.log(np.sum(np.exp(v - m))) + m) if np.isfinite(m) else -np.inf


class _Peel:
    """Per-family tree + leaf obs + shared expm(Q_arch,tau) cache for exact
    singleton / pair field-marginalized peels (dynfield model)."""
    def __init__(self, npz, cols, PI, S):
        from tkfdp.coupling.dynfield.phylo_elbo.exact_cap2 import _postorder
        from scipy.linalg import expm
        from tkfdp.coupling.dynfield.phylo_elbo.exact_cap2 import gtr_Q
        self.PI, self.S, self.A = PI, S, PI.shape[1]
        self._expm, self._gtr_Q = expm, gtr_Q
        d = np.load(npz); self.parent = d["parent"]; self.tau = d["tau"]; clv = d["clv"]
        n = len(self.parent); self.n = n
        ch = _dd(list)
        for v in range(n):
            if self.parent[v] >= 0:
                ch[int(self.parent[v])].append(v)
        self.children = ch
        self.root = int(np.where(self.parent < 0)[0][0])
        self.order = _postorder(self.root, ch)
        has = np.zeros(n, bool)
        for v in range(n):
            if self.parent[v] >= 0:
                has[int(self.parent[v])] = True
        self.leaf = ~has
        self.obs = {}
        for c in cols:
            o = np.full(n, -1, int)
            for v in range(n):
                if self.leaf[v]:
                    cv = clv[v, c]
                    o[v] = int(np.argmax(cv)) if cv.max() > 0.99 and cv.sum() < 1.5 else -1
            self.obs[int(c)] = o
        self.Pk = {}

    def P(self, t, k):
        key = (round(float(t), 12), int(k))
        if key not in self.Pk:
            self.Pk[key] = self._expm(self._gtr_Q(self.PI[k], self.S) * t)
        return self.Pk[key]

    def singleton_ll(self, col, a, rho, rho_chain):
        from tkfdp.coupling.dynfield.phylo_elbo.exact_cap2 import field_kernels
        L = rho.shape[0]; A = self.A; obs = self.obs[int(col)]; M = {}; ls = {}
        for v in self.order:
            if self.leaf[v]:
                x = int(obs[v]); r = np.ones(A) if x < 0 else np.eye(A)[x]
                M[v] = np.repeat(r[None], L, 0); ls[v] = 0.0
            else:
                Mv = np.ones((L, A)); lsc = 0.0
                for c in self.children[v]:
                    t = float(self.tau[c]); _, beta, J = field_kernels(rho, rho_chain, t)
                    Mc = M[c]; m_c = np.array([self.PI[a[k]] @ Mc[k] for k in range(L)])
                    NJ = np.array([beta[k] * (self.P(t, a[k]) @ Mc[k]) for k in range(L)])
                    Mv = Mv * (NJ + (J @ m_c)[:, None]); lsc += ls[c]
                mx = Mv.max()
                if mx > 0:
                    Mv /= mx; lsc += np.log(mx)
                M[v] = Mv; ls[v] = lsc
        m_root = np.array([self.PI[a[k]] @ M[self.root][k] for k in range(L)])
        return float(np.log(rho @ m_root) + ls[self.root])

    def pair_ll(self, ci, cj, a_i, a_j, rho, rho_chain):
        from tkfdp.coupling.dynfield.phylo_elbo.exact_cap2 import field_kernels
        L = rho.shape[0]; A = self.A; os_, ot = self.obs[int(ci)], self.obs[int(cj)]
        M = {}; ls = {}
        for v in self.order:
            if self.leaf[v]:
                x, y = int(os_[v]), int(ot[v])
                xr = np.ones(A) if x < 0 else np.eye(A)[x]
                yr = np.ones(A) if y < 0 else np.eye(A)[y]
                M[v] = np.repeat(np.outer(xr, yr)[None], L, 0); ls[v] = 0.0
            else:
                Mv = np.ones((L, A, A)); lsc = 0.0
                for c in self.children[v]:
                    t = float(self.tau[c]); _, beta, J = field_kernels(rho, rho_chain, t)
                    Mc = M[c]
                    m_c = np.array([self.PI[a_i[k]] @ Mc[k] @ self.PI[a_j[k]] for k in range(L)])
                    NJ = np.array([beta[k] * (self.P(t, a_i[k]) @ Mc[k] @ self.P(t, a_j[k]).T)
                                   for k in range(L)])
                    Mv = Mv * (NJ + (J @ m_c)[:, None, None]); lsc += ls[c]
                mx = Mv.max()
                if mx > 0:
                    Mv /= mx; lsc += np.log(mx)
                M[v] = Mv; ls[v] = lsc
        m_root = np.array([self.PI[a_i[k]] @ M[self.root][k] @ self.PI[a_j[k]] for k in range(L)])
        return float(np.log(rho @ m_root) + ls[self.root])


def pair_evidence_family(npz, pairs, arch, rho, rho_chain, rates, weights, PI, S,
                         topN=12):
    """Exact pair-evidence for a family's shortlisted pairs. For each pair returns
    the UNSUMMED top-N x top-N class-combo LL_pair grid (rate-marginalized) plus
    the full-K_c singleton marginals -- everything the DM-reweighted z-move lookup
    needs. arch: (K_c, L_field) class->archetype map."""
    logw = np.log(np.asarray(weights, float) / np.sum(weights))
    K_c = arch.shape[0]
    cols = sorted({int(c) for p in pairs for c in p})
    peel = _Peel(npz, cols, PI, S)

    def rmarg(fn):
        return _lse(np.array([fn(rho_chain * float(r)) for r in rates]) + logw)

    # singleton LL over ALL K_c classes per column (rate-marginalized)
    sing = {c: np.array([rmarg(lambda rc, cl=cl: peel.singleton_ll(c, arch[cl], rho, rc))
                         for cl in range(K_c)]) for c in cols}
    recs = []
    for (i, j) in pairs:
        Li, Lj = sing[i], sing[j]
        ti = np.argsort(-Li)[:topN]; tj = np.argsort(-Lj)[:topN]
        LLp = np.array([[rmarg(lambda rc, ci=ti[a_], cj=tj[b_]:
                               peel.pair_ll(i, j, arch[ci], arch[cj], rho, rc))
                         for b_ in range(len(tj))] for a_ in range(len(ti))])
        recs.append(dict(i=i, j=j, top_i=ti, top_j=tj, LL_pair=LLp,
                         logZ_s_i=_lse(Li), logZ_s_j=_lse(Lj)))
    return recs


# --------------------------------------------------------------------------
# Stage B, JAX/GPU: drive the validated batched exact peel (exact_cap2_jax via
# elbo_audit._score_specs_exact) over the shortlist x top-N^2 class grid. Reuses
# the tau-binned P_sub tables, so it carries the same (small) binning as the live
# sampler -- consistent with training, ~= the unbinned numpy reference.
# --------------------------------------------------------------------------
def build_enum400_ds(fams, rho_chain=0.15, field_rho0=0.6, field_rate_bins=3,
                     field_alpha=0.5, p_inv=0.5, seed=0, weights_override=None):
    """Build a discovery state (enum400 config) for the given families -- the
    trees + tau-binned P_sub/pi_field tables the JAX peel needs.

    `weights_override` (len field_rate_bins+1): use these EXACT field-rate mixture
    weights instead of the gamma_plus_inv_rates(field_alpha, p_inv) prior -- e.g.
    a field-rate harvested by the supervised trainer (train_supervised_enum400's
    _params.npz `field_rate_weights`). The rates grid is unchanged; only the
    mixture prior (which the cache's LL_pair is rate-marginalized over) changes."""
    from tkfdp.lg08 import get_lg08, PI_LG08  # noqa: F401
    from tkfdp.coupling.dynfield.phylo_elbo.corpus_state import build_corpus_state
    from tkfdp.coupling.dynfield.phylo_elbo import field_rate_discovery as fd
    from tkfdp.coupling.dynfield.phylo_elbo.rate_hetero import gamma_plus_inv_rates
    S, _ = get_lg08(); pa = _archetypes(); Ka = 20; Kc = Ka * Ka
    paths = [str(CLV_DIR / f"{f}.npz") for f in fams]
    rng = np.random.default_rng(seed)
    st = build_corpus_state(paths, K_c=Kc, K_a=Ka, L_field=2, pi_archetype=pa,
                            S=np.asarray(S), rho_chain=rho_chain, rng=rng, n_tau_bins=32)
    st.arch_assignment = np.stack([np.arange(Kc) // Ka, np.arange(Kc) % Ka], 1).astype(np.int32)
    st.rho = np.array([field_rho0, 1.0 - field_rho0], np.float64)
    st.refresh_pi_field()
    rates, weights = gamma_plus_inv_rates(field_rate_bins, field_alpha, p_inv)
    if weights_override is not None:
        weights = np.asarray(weights_override, float); weights = weights / weights.sum()
    ds = fd.build_discovery_state(st, rates, weights); ds.enum400 = True
    return ds, rates, weights


def _score_rate_marg(st, specs, P_sub, pi_field, rho, rho_chain, rates, logw,
                     b_chunk=512):
    """Rate-MARGINALIZED exact log-lik per spec, folding the field-rate bins into
    the batch via an outer vmap over (beta,J) -- ONE compiled forward per
    (bucket,chunk) covers all rates, vs the Python 4x rate loop in
    _score_specs_exact. Returns (n_specs,) numpy. GPU-ready."""
    import jax, jax.numpy as jnp
    from collections import defaultdict
    from tkfdp.coupling.dynfield.phylo_elbo.corpus_state import _materialize_padded_cluster
    from tkfdp.coupling.dynfield.phylo_elbo.tree_batch import (
        _bucket_shape_batch, _stack_padded_batch_binned, bucket_key_from_padded)
    from tkfdp.coupling.dynfield.phylo_elbo.exact_cap2_jax import (
        field_beta_J_bins, exact_pair_tree_ll_batch, exact_single_tree_ll_batch)
    jax.config.update("jax_enable_x64", True)
    real, bins = [], []
    for (fi, cols, cls, mb) in specs:
        real.append((_materialize_padded_cluster(st.families[fi], cols, mb), cls))
        bins.append(st.bin_idx_by_family[fi])
    beta_all = jnp.stack([field_beta_J_bins(st.bin_centers, rho, float(rho_chain) * float(r))[0]
                          for r in rates])
    J_all = jnp.stack([field_beta_J_bins(st.bin_centers, rho, float(rho_chain) * float(r))[1]
                       for r in rates])
    P_sub_j = jnp.asarray(P_sub); pi_j = jnp.asarray(pi_field)
    rho_j = jnp.asarray(np.asarray(rho, np.float64)); logw_j = jnp.asarray(logw)
    out = np.zeros(len(specs), np.float64)
    by_bucket = defaultdict(list)
    for i, (pt, cls) in enumerate(real):
        by_bucket[bucket_key_from_padded(pt, cls)].append(i)
    for bucket, idxs in by_bucket.items():
        mb = bucket[0]
        fn = exact_single_tree_ll_batch if mb == 1 else exact_pair_tree_ll_batch
        for c0 in range(0, len(idxs), b_chunk):
            sub = idxs[c0:c0 + b_chunk]; padded = [real[i] for i in sub]
            shape = _bucket_shape_batch(padded, bucket)
            cbin = _stack_padded_batch_binned(padded, bucket, [bins[i] for i in sub])
            classes_b = jnp.asarray(np.stack([np.asarray(cl, np.int32) for _, cl in padded]))
            leaf_obs = shape['leaf_obs'][:, :, :mb]

            def per_rate(beta_bins, J_bins):
                return fn(leaf_obs, shape['leaf_mask'], shape['child_pos_by_level'],
                          cbin, shape['is_identity_by_level'], shape['root_slot'],
                          classes_b, pi_j, P_sub_j, beta_bins, J_bins, rho_j)   # (n_sub,)
            ll_gr = jax.vmap(per_rate)(beta_all, J_all)                          # (G,n_sub)
            marg = jax.scipy.special.logsumexp(ll_gr + logw_j[:, None], axis=0)
            out[np.asarray(sub)] = np.asarray(marg)
    return out


def pair_evidence_jax(ds, fi, pairs, topN=12, b_chunk=512, scorer="mm"):
    """Pair-evidence via the batched JAX forward. scorer='mm' (default) uses the
    moment-matching forward -- A^2, ~21x faster than the A^3 exact peel, tight to
    +0.089 nat vs true-exact (which is SMALLER than the exact peel's own ~0.2-nat
    tau-binning error) and is the sampler's own scorer, so caching it is pure
    amortization. scorer='exact' uses the 800-state peel (validation / high
    fidelity, but ~90h corpus-wide). Singletons (all K_c) and the pair grids are
    each ONE rate-marginalized batched call."""
    st = ds.state; Kc = st.K_c; rates = ds.rates; logw = ds.logw
    mb1 = st.m_bucket_for(1); mb2 = st.m_bucket_for(2)

    if scorer == "mm":
        from tkfdp.coupling.dynfield.phylo_elbo import field_rate_discovery as fd

        def score(specs):
            ll = fd._score_specs_rates(ds, specs)                       # (n,G)
            return np.array([_lse(np.asarray(row) + logw) for row in ll])
    else:
        def score(specs):
            return _score_rate_marg(st, specs, ds.P_sub_aug, ds.pi_field_aug,
                                    st.rho, st.rho_chain, rates, logw, b_chunk)

    cols = sorted({int(c) for p in pairs for c in p})
    s_specs, s_slot = [], {}
    for col in cols:
        for c in range(Kc):
            s_slot[(col, c)] = len(s_specs)
            cl = np.zeros(mb1, np.int32); cl[0] = c
            s_specs.append((fi, np.array([col], np.int32), cl, mb1))
    ll_s = score(s_specs)                                                # (n,) rate-marg
    sing = {col: ll_s[[s_slot[(col, c)] for c in range(Kc)]] for col in cols}
    tops = {p: (np.argsort(-sing[p[0]])[:topN], np.argsort(-sing[p[1]])[:topN]) for p in pairs}
    p_specs, p_slot = [], {}
    for (i, j) in pairs:
        ti, tj = tops[(i, j)]
        for a, ci in enumerate(ti):
            for b, cj in enumerate(tj):
                p_slot[(i, j, a, b)] = len(p_specs)
                cl = np.zeros(mb2, np.int32); cl[0] = ci; cl[1] = cj
                p_specs.append((fi, np.array([i, j], np.int32), cl, mb2))
    ll_p = score(p_specs)
    recs = []
    for (i, j) in pairs:
        ti, tj = tops[(i, j)]
        LLp = np.array([[ll_p[p_slot[(i, j, a, b)]] for b in range(len(tj))]
                        for a in range(len(ti))])
        recs.append(dict(i=i, j=j, top_i=ti, top_j=tj, LL_pair=LLp,
                         logZ_s_i=_lse(sing[i]), logZ_s_j=_lse(sing[j])))
    return recs, sing


def score_singletons_jax(ds, fi, cols, b_chunk=512, scorer="mm"):
    """Per-column singleton LL over ALL K_c classes (rate-marginalized) for the
    given columns -- the cheap piece to backfill onto an existing pair cache."""
    st = ds.state; Kc = st.K_c; rates = ds.rates; logw = ds.logw
    mb1 = st.m_bucket_for(1)
    if scorer == "mm":
        from tkfdp.coupling.dynfield.phylo_elbo import field_rate_discovery as fd

        def score(specs):
            ll = fd._score_specs_rates(ds, specs)
            return np.array([_lse(np.asarray(row) + logw) for row in ll])
    else:
        def score(specs):
            return _score_rate_marg(st, specs, ds.P_sub_aug, ds.pi_field_aug,
                                    st.rho, st.rho_chain, rates, logw, b_chunk)
    cols = sorted({int(c) for c in cols})
    specs, slot = [], {}
    for col in cols:
        for c in range(Kc):
            slot[(col, c)] = len(specs)
            cl = np.zeros(mb1, np.int32); cl[0] = c
            specs.append((fi, np.array([col], np.int32), cl, mb1))
    ll_s = score(specs)
    return {col: ll_s[[slot[(col, c)] for c in range(Kc)]] for col in cols}


def write_pair_cache(path, recs, sing, topN):
    """Persist the pair records + the FULL per-column singleton LL vectors (all
    K_c, so the DM-reweighted singleton marginal is exact for any DM)."""
    sing_cols = np.array(sorted(sing.keys()), np.int32)
    np.savez_compressed(
        path,
        ij=np.array([[r["i"], r["j"]] for r in recs], np.int32),
        top_i=np.stack([r["top_i"] for r in recs]).astype(np.int32),
        top_j=np.stack([r["top_j"] for r in recs]).astype(np.int32),
        LL_pair=np.stack([r["LL_pair"] for r in recs]).astype(np.float64),
        logZ_s_i=np.array([r["logZ_s_i"] for r in recs]),      # kept (== lse(sing_ll))
        logZ_s_j=np.array([r["logZ_s_j"] for r in recs]),
        sing_cols=sing_cols,
        sing_ll=np.stack([sing[int(c)] for c in sing_cols]).astype(np.float64),
        topN=np.int32(topN))


def cache_logodds(rec_or_npz, idx, alpha_z, dm_logp=None):
    """Class-marginalized pairing log-odds from a cached record (the z-move
    lookup). dm_logp: optional (N,N) DM log-prior over the class grid to reweight;
    None = flat class prior."""
    if isinstance(rec_or_npz, dict):
        LL, zi, zj = rec_or_npz["LL_pair"], rec_or_npz["logZ_s_i"], rec_or_npz["logZ_s_j"]
    else:
        LL = rec_or_npz["LL_pair"][idx]; zi = rec_or_npz["logZ_s_i"][idx]
        zj = rec_or_npz["logZ_s_j"][idx]
    grid = LL if dm_logp is None else LL + dm_logp
    return _lse(np.asarray(grid).ravel()) - float(zi) - float(zj) - np.log(alpha_z)


def _train_families(split="train"):
    from tkfdp.bio import load_split
    fams_all = json.loads((CLV_DIR / "index.json").read_text())["families"]
    sp = load_split()
    keep = set()
    for s in split.split(","):                        # one or more of train/val/test
        keep |= set(sp[s])
    pdir = Path("data/pdb_partition_clv_top1000_sifts")
    return [f for f in fams_all if f in keep and (pdir / f"{f}.npz").exists()]


def run_precompute(out_dir, z_star=2.5, topN=12, K=100, b_chunk=512,
                   n_families=None, resume=True, log_stream=None, scorer="mm",
                   max_pairs=2000, shard=0, nshards=1, split="train",
                   field_rate_params=None, fam_list=None):
    """RESUMABLE corpus precompute: per-family Stage A (miz shortlist) + Stage B
    (exact pair-evidence cache). Writes data/pairing_cache/<fam>.{miz,pairev}.npz
    atomically (tmp -> os.replace); on restart, families whose .pairev.npz already
    exists are skipped, so a crash/bug never restarts from scratch.

    field_rate_params: path to a train_supervised_enum400 _params.npz -- rebuild
    LL_pair at that HARVESTED field-rate (field_rate_weights + rho_chain) instead
    of the flip-averse prior. fam_list: restrict to these families (e.g. the
    confirmed-flip set) instead of the whole split."""
    import os, time
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    wk, rc = None, 0.15
    if field_rate_params:
        _p = np.load(field_rate_params)
        wk = np.asarray(_p["field_rate_weights"], float)
        rc = float(_p["rho_chain"])
        print(f"# field-rate override from {field_rate_params}: weights={np.round(wk,4)} rho_chain={rc}", flush=True)
    if fam_list is not None:
        elig = set(_train_families("train,val,test"))
        fams = [f for f in fam_list if f in elig]
    else:
        fams = _train_families(split)
    if n_families:
        fams = fams[:n_families]

    def emit(s):
        print(s, flush=True)
        if log_stream:
            log_stream.write(s + "\n"); log_stream.flush()

    todo = [f for f in fams if not (resume and (out / f"{f}.pairev.npz").exists())]
    if nshards > 1:                                   # split remaining across GPUs
        todo = todo[shard::nshards]
    emit(f"# precompute: {len(fams)} families, {len(todo)} to do "
         f"(resume skips, shard {shard}/{nshards}); z*={z_star} topN={topN} K={K}")
    if not todo:
        emit("# all families already cached; nothing to do"); return
    t0 = time.time()
    ds, rates, weights = build_enum400_ds(todo, rho_chain=rc, weights_override=wk)
    fam_ids = [f.family_id for f in ds.state.families]
    emit(f"# state built for {len(todo)} families in {time.time() - t0:.0f}s")
    for k, fam in enumerate(todo):
        if resume and (out / f"{fam}.pairev.npz").exists():
            continue                                     # double-check (parallel safety)
        fi = fam_ids.index(fam)
        ts = time.time()
        msa = np.load(CLV_DIR / f"{fam}.npz")["leaf_msa"]; L = int(msa.shape[1])
        Z = perm_null_z(msa, K=K)
        cand = [(int(i), int(j)) for i in range(L) for j in range(i + 1, L)
                if Z[i, j] > z_star]
        n_cand = len(cand); capped = False
        if max_pairs and n_cand > max_pairs:                 # keep top-max_pairs by miz
            cand.sort(key=lambda ij: -Z[ij[0], ij[1]])
            cand = cand[:max_pairs]; capped = True
        pairs = cand
        t_mi = time.time() - ts
        if pairs:
            recs, sing = pair_evidence_jax(ds, fi, pairs, topN=topN,
                                           b_chunk=b_chunk, scorer=scorer)
        else:
            recs, sing = [], {}
        tmp = out / f"{fam}.pairev.tmp.npz"
        if recs:
            write_pair_cache(tmp, recs, sing, topN)
        else:
            np.savez_compressed(tmp, ij=np.zeros((0, 2), np.int32),
                                sing_cols=np.zeros(0, np.int32), topN=np.int32(topN))
        os.replace(tmp, out / f"{fam}.pairev.npz")
        np.savez_compressed(out / f"{fam}.miz.npz",
                            ij=np.array(pairs, np.int32).reshape(-1, 2),
                            z_star=np.float64(z_star))
        cap_note = f" CAPPED from {n_cand}" if capped else ""
        emit(f"# [{k + 1}/{len(todo)}] {fam}: L={L} shortlist={len(pairs)}{cap_note} "
             f"(miz {t_mi:.0f}s, peel {time.time() - ts - t_mi:.0f}s) "
             f"[elapsed {time.time() - t0:.0f}s]")
    emit(f"# precompute done: {len(todo)} families in {time.time() - t0:.0f}s")


def backfill_singletons(out_dir, scorer="mm", n=None, log_stream=None):
    """Add the per-column singleton LL vectors (sing_cols, sing_ll) to caches
    written before the enriched format. Only the cheap singleton scan is redone;
    the (expensive) pair grid is preserved from the existing file. Idempotent:
    families already carrying sing_ll are skipped."""
    import os, time
    out = Path(out_dir)
    todo = []
    for p in sorted(out.glob("*.pairev.npz")):
        d = np.load(p)
        if "sing_cols" in d.files and d["sing_cols"].size:
            continue
        if d["ij"].reshape(-1, 2).shape[0] == 0:            # empty shortlist: mark done
            np.savez_compressed(p, **{k: d[k] for k in d.files},
                                sing_cols=np.zeros(0, np.int32))
            continue
        todo.append(p.name.split(".pairev")[0])
    if n:
        todo = todo[:n]

    def emit(s):
        print(s, flush=True)
        if log_stream:
            log_stream.write(s + "\n"); log_stream.flush()
    emit(f"# backfill: {len(todo)} old-format families to enrich")
    if not todo:
        return
    ds, _, _ = build_enum400_ds(todo)
    fam_ids = [f.family_id for f in ds.state.families]
    for k, fam in enumerate(todo):
        t = time.time()
        p = out / f"{fam}.pairev.npz"
        d = dict(np.load(p))
        cols = np.unique(d["ij"].reshape(-1, 2)).tolist()
        sing = score_singletons_jax(ds, fam_ids.index(fam), cols, scorer=scorer)
        sc = np.array(sorted(sing.keys()), np.int32)
        d["sing_cols"] = sc
        d["sing_ll"] = np.stack([sing[int(c)] for c in sc]).astype(np.float64)
        tmp = out / f"{fam}.pairev.tmp.npz"
        np.savez_compressed(tmp, **d); os.replace(tmp, p)
        emit(f"# [{k + 1}/{len(todo)}] {fam}: +sing_ll ({len(sc)} cols, {time.time() - t:.0f}s)")
    emit("# backfill done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--out-dir", default="data/pairing_cache")
    ap.add_argument("--z-star", type=float, default=2.5)
    ap.add_argument("--topN", type=int, default=12)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--b-chunk", type=int, default=512)
    ap.add_argument("--n-families", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--scorer", choices=["mm", "exact"], default="mm")
    ap.add_argument("--max-pairs", type=int, default=2000,
                    help="cap shortlist to top-N pairs by miz (0=uncapped)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process does families[i::n]")
    ap.add_argument("--split", default="train", help="one or more of train/val/test (comma-sep)")
    ap.add_argument("--field-rate-params", default=None,
                    help="_params.npz to rebuild LL_pair at a harvested field-rate")
    ap.add_argument("--family-list", default=None,
                    help="restrict to these families: comma-sep, or a path to a file (one/line)")
    ap.add_argument("--flip-families", action="store_true",
                    help="restrict to the confirmed-flip families (shortcut)")
    args = ap.parse_args()
    _sh, _ns = (int(x) for x in args.shard.split("/"))
    fam_list = None
    if args.flip_families:
        import json as _json
        cf = _json.load(open("data/pdb_partition_clv_top1000_sifts/confirmed_flips.json"))
        fam_list = sorted({r["family"] for r in cf})
    elif args.family_list:
        if Path(args.family_list).exists():
            fam_list = [l.strip() for l in open(args.family_list) if l.strip()]
        else:
            fam_list = [f.strip() for f in args.family_list.split(",") if f.strip()]
    if args.sweep:
        sweep([2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0], K=args.K, n_families=args.n_families)
    elif args.precompute:
        run_precompute(args.out_dir, z_star=args.z_star, topN=args.topN, K=args.K,
                       b_chunk=args.b_chunk, n_families=args.n_families,
                       resume=not args.no_resume, scorer=args.scorer,
                       max_pairs=args.max_pairs, shard=_sh, nshards=_ns, split=args.split,
                       field_rate_params=args.field_rate_params, fam_list=fam_list)
    elif args.backfill:
        backfill_singletons(args.out_dir, scorer=args.scorer, n=args.n_families)


if __name__ == "__main__":
    main()
