"""Scalable +Gamma+I dynfield: SAMPLE site classes (Gibbs), MARGINALISE the
per-cluster field-rate (G+1 Gamma+I bins, incl. the invariant static/no-flip
bin). Cost is ~G x the base forward per cluster (not K_c^2 as in exact class
marginalisation), so it runs at full corpus scale on the fast MF forward.

Flip statistic: phi(C) = P(r_C != 0 | C), the posterior on the non-invariant
field-rate bins. Learned target = arch_assignment (frozen LG-Cxx archetypes),
sampled by MH over the field-rate-marginal evidence. Partition frozen.

See appendix-tkfdp.tex "Rate heterogeneity (+Gamma+I)".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corpus_state import _score_cluster_batch, _cluster_classes_padded
from .marginal_scorer import _logsumexp
from .tau_binning import gtr_P_bins, field_kernel_tables


def build_pi_field_aug(pi_field, G_s):
    """pi_field_aug[c*G_s+r] = pi_field[c] (stationary is rate-independent)."""
    K_c, L, A = pi_field.shape
    return np.repeat(pi_field, G_s, axis=0)                  # (K_c*G_s, L, A)


def build_P_sub_aug(pi_field, S, bin_centers, rates_sub, out=None,
                    classes=None):
    """P_sub_aug[b, c*G_s+r, theta] = expm(Q^{c,theta} * rates_sub[r] * tau_b).
    Per-site substitution rate baked into the (base_class x rate_bin) axis.
    If `classes` given, (re)build only those base classes' slices in `out`."""
    K_c, L, A = pi_field.shape
    G_s = len(rates_sub)
    n_bins = len(bin_centers)
    if out is None:
        out = np.zeros((n_bins, K_c * G_s, L, A, A), dtype=np.float64)
    cs = range(K_c) if classes is None else classes
    for c in cs:
        for r, m in enumerate(rates_sub):
            for th in range(L):
                out[:, c * G_s + r, th] = gtr_P_bins(
                    pi_field[c, th], S, np.asarray(bin_centers) * float(m))
    return out


def aug_field_rate_tables(fr, r_field):
    """Kernel tables for one field-rate bin: field beta/W at rho_chain*r_field
    (rate=0 -> static), augmented P_sub / pi_field carrying the per-site rate."""
    import jax.numpy as jnp
    beta, W = field_kernel_tables(fr.state.bin_centers, fr.state.rho,
                                  fr.state.rho_chain * float(r_field))
    return {'beta': jnp.asarray(np.asarray(beta)),
            'W': jnp.asarray(np.asarray(W)),
            'P_sub': jnp.asarray(fr.P_sub_aug),
            'pi_field': fr.pi_field_aug,
            'bin_centers': fr.state.bin_centers}


@dataclass
class FieldRateState:
    state: object                       # CorpusState, partition frozen
    clusters: list                      # [(fi, cols)]
    rates: np.ndarray                   # (G_field,) rates[0]=0 invariant (FIELD)
    weights: np.ndarray                 # (G_field,) prior; weights[0]=p_inv
    site_rate_bin: dict = field(default_factory=dict)    # family_id -> (L,) int
    rates_sub: np.ndarray = None        # (G_s,) per-SITE substitution multipliers
    S: np.ndarray = None                # exchangeability (for P_sub_aug rebuild)
    G_s: int = 1
    enum400: bool = False               # class==(arch@0,arch@1); invariant fast path
    pi_field_aug: np.ndarray = None     # (K_c*G_s, L, A)
    P_sub_aug: np.ndarray = None        # (n_bins, K_c*G_s, L, A, A)
    col_to_cluster: dict = field(default_factory=dict)   # (fi,col)->cluster idx
    ll_by_rate: np.ndarray = None       # (n_clusters, G_field) under cur classes

    @property
    def logw(self):
        return np.log(np.asarray(self.weights, float))


def build_field_rate_state(state, clusters, rates, weights,
                           site_rate_bin=None, rates_sub=None) -> FieldRateState:
    """site_rate_bin: {family_id: (L,) int rate-bin per column}; rates_sub:
    (G_s,) per-site substitution multipliers. If omitted, per-site rate is off
    (G_s=1, m=1)."""
    fr = FieldRateState(state=state, clusters=list(clusters),
                        rates=np.asarray(rates, float),
                        weights=np.asarray(weights, float),
                        site_rate_bin=site_rate_bin or {},
                        rates_sub=(np.asarray(rates_sub, float)
                                   if rates_sub is not None else np.array([1.0])),
                        S=state.S if hasattr(state, 'S') else None)
    fr.G_s = len(fr.rates_sub)
    for ci, (fi, cols) in enumerate(fr.clusters):
        for c in np.asarray(cols).tolist():
            fr.col_to_cluster[(fi, int(c))] = ci
    fr.pi_field_aug = build_pi_field_aug(state._pi_field, fr.G_s)
    fr.P_sub_aug = build_P_sub_aug(state._pi_field, fr.S, state.bin_centers,
                                   fr.rates_sub)
    return fr


def _rate_bins(fr, fi, cols):
    """(m,) per-site substitution-rate-bin index for a cluster's columns."""
    fam_id = fr.state.families[fi].family_id
    rb = fr.site_rate_bin.get(fam_id)
    if rb is None:
        return np.zeros(len(cols), np.int32)
    return np.asarray(rb)[np.asarray(cols)]


def _aug(base_cls, rate_bins, G_s):
    return (np.asarray(base_cls, np.int32) * G_s
            + np.asarray(rate_bins, np.int32)).astype(np.int32)


def _spec(fr, ci, base_override=None, pos=None):
    """Padded AUGMENTED-class spec for cluster ci. base_override/pos lets cn
    Gibbs vary one column's base class."""
    fi, cols = fr.clusters[ci]
    fam = fr.state.families[fi]
    cols = np.asarray(cols, np.int32)
    m = int(cols.shape[0]); mb = fr.state.m_bucket_for(m)
    base = fam.classes[cols].copy()
    if base_override is not None:
        base[pos] = base_override
    aug = _aug(base, _rate_bins(fr, fi, cols), fr.G_s)
    cls = np.zeros(mb, np.int32); cls[:m] = aug
    return (fi, cols, cls, mb)


def _tables_g(fr, g):
    return aug_field_rate_tables(fr, fr.rates[g])


def _score_specs_rates(fr, specs) -> np.ndarray:
    """(len(specs), G_field) LL under each field-rate bin, augmented tables."""
    out = np.zeros((len(specs), len(fr.rates)), float)
    for g in range(len(fr.rates)):
        out[:, g] = _score_cluster_batch(fr.state, specs, tables=_tables_g(fr, g))
    return out


def init_ll(fr: FieldRateState) -> None:
    fr.ll_by_rate = _score_specs_rates(fr, [_spec(fr, ci)
                                            for ci in range(len(fr.clusters))])


def marginal_and_flip(fr: FieldRateState):
    logw = fr.logw
    terms = fr.ll_by_rate + logw[None, :]                 # (n, G)
    cluster_ll = np.array([_logsumexp(t) for t in terms])
    if len(fr.rates) > 1:
        num = np.array([_logsumexp(t[1:]) for t in terms])
        flip = np.exp(num - cluster_ll)
    else:
        flip = np.zeros(len(cluster_ll))
    return cluster_ll, flip


def corpus_ll(fr): return float(marginal_and_flip(fr)[0].sum())


def cn_gibbs_move(fr: FieldRateState, fi: int, s: int, rng,
                  alpha_c_log=None) -> int:
    """K_c-way Gibbs on classes[fi][s] using the field-rate-marginal cluster
    evidence. Rescores only the one cluster containing site s."""
    state = fr.state; K_c = state.K_c
    fam = state.families[fi]
    ci = fr.col_to_cluster[(fi, s)]
    fi_c, cols = fr.clusters[ci]
    cols = np.asarray(cols, np.int32); m = int(cols.shape[0])
    mb = state.m_bucket_for(m)
    pos = int(np.where(cols == s)[0][0])
    c_curr = int(fam.classes[s])
    # candidate specs: vary BASE class at position `pos` (augmented internally).
    specs = [_spec(fr, ci, base_override=c, pos=pos) for c in range(K_c)]
    if fr.enum400:
        # invariant (r=0) bin via per-column static forwards + rho-combine
        # (K_a per column, not K_a^2 field forwards); active bins as usual.
        ll = np.empty((K_c, len(fr.rates)))
        ll[:, 0] = cn_invariant_column_enum400(fr, ci, pos)
        for g in range(1, len(fr.rates)):
            ll[:, g] = _score_cluster_batch(fr.state, specs, tables=_tables_g(fr, g))
    else:
        ll = _score_specs_rates(fr, specs)                # (K_c, G_field)
    marg = np.array([_logsumexp(ll[c] + fr.logw) for c in range(K_c)])
    logp = marg + (alpha_c_log if alpha_c_log is not None else 0.0)
    logp -= logp.max(); p = np.exp(logp); p /= p.sum()
    c_new = int(rng.choice(K_c, p=p))
    fam.classes[s] = c_new
    fr.ll_by_rate[ci] = ll[c_new]
    return c_new


def arch_mh_move(fr: FieldRateState, c: int, theta: int, rng,
                 log_prior=None) -> bool:
    """MH on arch_assignment[c, theta] over the field-rate-marginal corpus
    evidence. Rescores only clusters that currently have a class-c column."""
    state = fr.state; K_a = state.K_a; aa = state.arch_assignment
    k_curr = int(aa[c, theta])
    # affected clusters: any cluster with a column of class c
    aff = [ci for ci, (fi, cols) in enumerate(fr.clusters)
           if np.any(state.families[fi].classes[np.asarray(cols)] == c)]
    ll_curr = corpus_ll(fr)
    k_prop = int(rng.integers(K_a - 1))
    if k_prop >= k_curr:
        k_prop += 1
    if not aff:
        aa[c, theta] = k_prop; state.apply_arch_slices([(c, theta)]); return True
    saved = fr.ll_by_rate[aff].copy()
    aa[c, theta] = k_prop; state.apply_arch_slices([(c, theta)])
    _refresh_aug_class(fr, c)
    fr.ll_by_rate[aff] = _score_specs_rates(fr, [_spec(fr, ci) for ci in aff])
    ll_prop = corpus_ll(fr)
    lr = ll_prop - ll_curr
    if log_prior is not None:
        lr += float(log_prior[k_prop] - log_prior[k_curr])
    if np.log(rng.uniform() + 1e-300) < lr:
        return True
    aa[c, theta] = k_curr; state.apply_arch_slices([(c, theta)])
    _refresh_aug_class(fr, c)
    fr.ll_by_rate[aff] = saved
    return False


def _refresh_aug_class(fr, c):
    """Rebuild the augmented pi_field / P_sub slices for base class c after its
    arch_assignment changed (state._pi_field[c] updated by apply_arch_slices)."""
    G_s = fr.G_s
    fr.pi_field_aug[c * G_s:(c + 1) * G_s] = fr.state._pi_field[c]
    build_P_sub_aug(fr.state._pi_field, fr.S, fr.state.bin_centers,
                    fr.rates_sub, out=fr.P_sub_aug, classes=[c])


def run_training(fr: FieldRateState, n_sweeps: int, rng, fix_theta0=True,
                 do_cn=True, do_arch=True, confirmed_flip_mask=None,
                 log_stream=None, out_dir=None, start_sweep=0, dm=None,
                 ckpt_every_sec=0):
    """out_dir/start_sweep: atomic sweep-level checkpoint/resume of the canonical
    state (classes + arch_assignment + step + rng). Clusters are fixed here
    (supervised partition), so only classes/arch evolve; on resume rebuild fr via
    build_field_rate_state after corpus_state.load_checkpoint, then init_ll."""
    import time
    from . import corpus_state as cs
    K_c, L = fr.state.arch_assignment.shape
    thetas = list(range(1, L)) if fix_theta0 else list(range(L))
    arch_entries = [(c, t) for c in range(K_c) for t in thetas]
    # per-family site list (only sites in a cluster)
    sites = [(fi, int(s)) for (fi, cols) in fr.clusters for s in np.asarray(cols)]
    t0 = time.time()

    def _log(sweep):
        cll, flip = marginal_and_flip(fr)
        msg = (f"# sweep={sweep:3d} corpus_ll={cll.sum():+.1f} "
               f"mean_flip={flip.mean():.3f} p_inv={1-flip.mean():.3f} "
               f"t={time.time()-t0:.0f}s")
        if confirmed_flip_mask is not None:
            cf = np.asarray(confirmed_flip_mask, bool)
            if cf.any():
                msg += (f"  flip[conf]={flip[cf].mean():.3f} "
                        f"flip[other]={flip[~cf].mean():.3f}")
        print(msg, flush=True)
        if log_stream: log_stream.write(msg + "\n"); log_stream.flush()

    _log(start_sweep)
    for sweep in range(start_sweep + 1, n_sweeps + 1):
        if dm is not None:                              # DM prior: Gibbs h_C | classes
            for ci, (fi, cols) in enumerate(fr.clusters):
                dm.sample_h(fr.state, fi, cols, ci, rng)
        last_ckpt = time.time(); n_done = 0
        if do_cn:
            perm = rng.permutation(len(sites))
            for idx in perm:
                fi, s = sites[idx]; acl = None
                if dm is not None:
                    ci = fr.col_to_cluster[(fi, s)]
                    acl = dm.cn_logprior(fr.state, fi, fr.clusters[ci][1], s, ci)
                cn_gibbs_move(fr, fi, s, rng, alpha_c_log=acl)
                n_done += 1
                # intra-sweep safety net: sweeps on the full corpus run hours, so
                # checkpoint at step=sweep-1 (replay this sweep on resume, from the
                # partition it has already improved). Cheap vs a full sweep.
                if (out_dir is not None and ckpt_every_sec
                        and time.time() - last_ckpt >= ckpt_every_sec):
                    cs.save_checkpoint_atomic(
                        fr.state, out_dir, sweep - 1, rng, log_stream=log_stream,
                        t0=t0, label=f"sweep{sweep}~mid {n_done}/{len(sites)} cn",
                        dm=dm, dm_hcol=_dm_hcol(fr, dm))
                    last_ckpt = time.time()
        if do_arch:
            for idx in rng.permutation(len(arch_entries)):
                c, t = arch_entries[idx]; arch_mh_move(fr, c, t, rng)
        if dm is not None:                              # update hyperparams
            dm.update_alpha([(ci, fi, cols, fr.state)
                             for ci, (fi, cols) in enumerate(fr.clusters)])
            dm.update_pi(range(len(fr.clusters)), rng)
        if out_dir is not None:
            cs.save_checkpoint_atomic(fr.state, out_dir, sweep, rng,
                                      log_stream=log_stream, t0=t0,
                                      dm=dm, dm_hcol=_dm_hcol(fr, dm))
        _log(sweep)
    return fr


def _dm_hcol(fr, dm):
    """{fi: (n_col,) int} of each column's DM component, from dm.h keyed by the
    cluster index (supervised keying). None if no DM."""
    if dm is None:
        return None
    out = {fi: np.zeros(len(fam.classes), np.int32)
           for fi, fam in enumerate(fr.state.families)}
    for ci, (fi, cols) in enumerate(fr.clusters):
        out[fi][np.asarray(cols)] = dm.h.get(ci, 0)
    return out


def restore_dm(fr, dm, ckpt_path):
    """Restore dm (alpha/pi/H/alpha_pi + per-cluster component h) from a
    checkpoint after fr.clusters is built. Rebuilds h with the supervised
    cluster-index key. No-op if no DM or the checkpoint predates DM persistence."""
    from . import corpus_state as cs
    if dm is None:
        return
    hcol = cs.load_dm(dm, ckpt_path, fr.state)
    if hcol is None:
        return
    dm.h = {}
    for ci, (fi, cols) in enumerate(fr.clusters):
        if fi in hcol:
            dm.h[ci] = int(hcol[fi][int(np.asarray(cols)[0])])


# ---- enum400 invariant-bin fast path (exact) --------------------------------
# In the invariant (r=0) bin the field is frozen, so columns are conditionally
# independent: a cluster's r=0 LL is a rho-weighted 2-term combine of per-column
# STATIC single-archetype forwards. This collapses the r=0 bin's K_c=K_a^2
# candidate forwards to K_a per column + a cheap combine (user's optimisation).

def _col_invariant_static(fr, fi, col):
    """(K_a,) invariant single-column log-lik of `col` frozen at archetype k
    (enum400 diagonal class (k,k), m=1 spec, rate bin 0)."""
    K_a = fr.state.pi_archetype.shape[0]
    mb = fr.state.m_bucket_for(1)
    colv = np.array([int(col)], np.int32)
    rb = _rate_bins(fr, fi, colv)
    specs = []
    for k in range(K_a):
        cls = np.zeros(mb, np.int32)
        cls[:1] = _aug(np.array([k * K_a + k], np.int32), rb, fr.G_s)
        specs.append((fi, colv, cls, mb))
    return _score_cluster_batch(fr.state, specs, tables=_tables_g(fr, 0))  # (K_a,)


def cn_invariant_column_enum400(fr, ci, pos):
    """(K_c,) invariant-bin (r=0) log-lik for all K_a^2 candidate classes at
    position `pos` of cluster ci, via per-column statics + rho-combine. Exact."""
    K_a = fr.state.pi_archetype.shape[0]
    fi, cols = fr.clusters[ci]
    cols = np.asarray(cols, np.int32); m = int(cols.shape[0])
    rho = np.asarray(fr.state.rho, float)
    lr0, lr1 = np.log(rho[0]), np.log(rho[1])
    a0 = np.arange(K_a * K_a) // K_a
    a1 = np.arange(K_a * K_a) % K_a
    Ss = _col_invariant_static(fr, fi, int(cols[pos]))          # (K_a,) moving col
    # any m: frozen field => columns independent, so sum the FIXED columns'
    # static log-lik at their class's arch@0 / arch@1 (empty for m=1).
    lP0 = lP1 = 0.0
    for jp in range(m):
        if jp == pos:
            continue
        jcls = int(fr.state.families[fi].classes[int(cols[jp])])
        Sj = _col_invariant_static(fr, fi, int(cols[jp]))
        lP0 += Sj[jcls // K_a]; lP1 += Sj[jcls % K_a]
    return np.logaddexp(lr0 + Ss[a0] + lP0, lr1 + Ss[a1] + lP1)                                 # (K_c,)
