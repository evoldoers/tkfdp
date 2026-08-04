"""Unsupervised +Gamma+I dynfield: DISCOVER the cap-2 cluster partition (CRP
z-move, alpha_z fixed) instead of using PDB-supervised labels, while sampling
site classes, marginalising the per-cluster field-rate (Gamma+I, static bin),
and holding per-site substitution rates fixed (augmented base_class x rate_bin,
via field_rate_trainer helpers).

Cluster identity is state.families[fi].cluster_id (starts all-singletons). The
z-move is a Neal-3 CRP over cluster_id[s] scored under the field-rate-marginal;
with max_cluster_size=2 the candidates are: stay singleton / split a pair /
pair s with another singleton. ll_by_rate is cached per cluster identity so the
partition can change dynamically.

See appendix-tkfdp.tex "Rate heterogeneity (+Gamma+I)". Reuses the augmented
per-site-rate scoring of field_rate_trainer (no forward change).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corpus_state import (_score_cluster_batch, _cluster_columns_by_id,
                          _cluster_key)
from .marginal_scorer import _logsumexp
from .field_rate_trainer import (build_pi_field_aug, build_P_sub_aug,
                                _aug, _rate_bins, _refresh_aug_class)


@dataclass
class DiscoveryState:
    state: object
    rates: np.ndarray                   # field-rate multipliers (rates[0]=0)
    weights: np.ndarray
    site_rate_bin: dict = field(default_factory=dict)
    rates_sub: np.ndarray = None
    S: np.ndarray = None
    G_s: int = 1
    enum400: bool = False               # class==(arch@0,arch@1); invariant fast path
    dm: object = None                   # mixture-of-DM class-composition prior
    pi_field_aug: np.ndarray = None
    P_sub_aug: np.ndarray = None        # (n_bins, K_c*G_s, ...) numpy (mutated)
    P_sub_aug_j: object = None          # cached DEVICE copy; refreshed on arch
    rate_kernels: list = field(default_factory=list)  # per-g (beta_j, W_j), const
    ll_by_rate: dict = field(default_factory=dict)   # cluster_key -> (G,) LL

    @property
    def logw(self):
        return np.log(np.asarray(self.weights, float))


def build_discovery_state(state, rates, weights, site_rate_bin=None,
                          rates_sub=None) -> DiscoveryState:
    ds = DiscoveryState(state=state, rates=np.asarray(rates, float),
                        weights=np.asarray(weights, float),
                        site_rate_bin=site_rate_bin or {},
                        rates_sub=(np.asarray(rates_sub, float)
                                   if rates_sub is not None else np.array([1.0])),
                        S=state.S)
    ds.G_s = len(ds.rates_sub)
    ds.pi_field_aug = build_pi_field_aug(state._pi_field, ds.G_s)
    ds.P_sub_aug = build_P_sub_aug(state._pi_field, ds.S, state.bin_centers,
                                   ds.rates_sub)
    import jax.numpy as jnp
    from .tau_binning import field_kernel_tables
    ds.P_sub_aug_j = jnp.asarray(ds.P_sub_aug)     # device copy, cached
    # Field kernels beta/W depend only on (bin_centers, rho, rho_chain*r_g) --
    # all constant across the run (no rho/rho_chain M-step in discovery). Build
    # the G device tables ONCE here; _tables_g just reuses them (previously
    # field_kernel_tables + a host->device transfer ran on every scoring call,
    # ~4x per move, hundreds of thousands of times per sweep).
    ds.rate_kernels = []
    for r in ds.rates:
        beta, W = field_kernel_tables(state.bin_centers, state.rho,
                                      state.rho_chain * float(r))
        ds.rate_kernels.append((jnp.asarray(np.asarray(beta)),
                                jnp.asarray(np.asarray(W))))
    return ds


def _tables_g(ds, g):
    """Tables for field-rate bin g: cached constant beta/W (built once in
    build_discovery_state) + the CURRENT device P_sub reference (rebound by arch
    moves, so read it live -- do not cache it inside the dict)."""
    beta, W = ds.rate_kernels[g]
    return {'beta': beta, 'W': W, 'P_sub': ds.P_sub_aug_j,
            'pi_field': ds.pi_field_aug, 'bin_centers': ds.state.bin_centers}


def _spec_for(ds, fi, cols):
    fam = ds.state.families[fi]
    cols = np.asarray(cols, np.int32); m = int(cols.shape[0])
    mb = ds.state.m_bucket_for(m)
    base = fam.classes[cols]
    aug = _aug(base, _rate_bins(ds, fi, cols), ds.G_s)
    cls = np.zeros(mb, np.int32); cls[:m] = aug
    return (fi, cols, cls, mb)


def _score_specs_rates(ds, specs) -> np.ndarray:
    out = np.zeros((len(specs), len(ds.rates)), float)
    for g in range(len(ds.rates)):
        out[:, g] = _score_cluster_batch(
            ds.state, specs, tables=_tables_g(ds, g))
    return out


def _marg(ll_g, logw):
    return _logsumexp(np.asarray(ll_g) + logw)


def init_ll(ds: DiscoveryState) -> None:
    """Score every current cluster (all singletons at start) under G rates."""
    ds.ll_by_rate = {}
    specs, keys = [], []
    for fi, fam in enumerate(ds.state.families):
        for cid, cols in _cluster_columns_by_id(fam.cluster_id).items():
            specs.append(_spec_for(ds, fi, cols))
            keys.append(_cluster_key(fi, fam.cluster_id, cid))
    ll = _score_specs_rates(ds, specs)
    for k, row in zip(keys, ll):
        ds.ll_by_rate[k] = row


def corpus_ll(ds: DiscoveryState) -> float:
    logw = ds.logw
    return float(sum(_marg(v, logw) for v in ds.ll_by_rate.values()))


def z_move(ds: DiscoveryState, fi: int, s: int, rng) -> None:
    """Neal-3 CRP on cluster_id[fi][s] under the field-rate-marginal evidence,
    cap-2 (max_cluster_size). alpha_z fixed (state.alpha_z)."""
    state = ds.state; fam = state.families[fi]; logw = ds.logw
    cap = state.max_cluster_size
    cid_curr = int(fam.cluster_id[s])
    clusters = _cluster_columns_by_id(fam.cluster_id)
    curr = clusters[cid_curr]; n_curr = int(curr.shape[0])
    curr_minus = np.asarray([c for c in curr.tolist() if c != s], np.int32)

    # score any needed new configs in one batch
    specs, slot = [], {}
    def add(label, cols):
        slot[label] = len(specs); specs.append(_spec_for(ds, fi, cols))
    key_curr = _cluster_key(fi, fam.cluster_id, cid_curr)
    if key_curr not in ds.ll_by_rate:
        add('curr', curr)
    if n_curr > 1:
        add('curr_minus', curr_minus)
        add('singleton', np.asarray([s], np.int32))
    others = []
    for c_o, mem in clusters.items():
        if c_o == cid_curr or int(mem.shape[0]) + 1 > cap:
            continue
        key_o = _cluster_key(fi, fam.cluster_id, c_o)
        if key_o not in ds.ll_by_rate:
            add(('other', c_o), mem)
        add(('other_plus', c_o), np.sort(np.append(mem, s)))
        others.append(c_o)
    ll = _score_specs_rates(ds, specs) if specs else np.zeros((0, len(ds.rates)))
    for label, i in slot.items():
        if label == 'curr':
            ds.ll_by_rate[key_curr] = ll[i]
        elif isinstance(label, tuple) and label[0] == 'other':
            ds.ll_by_rate[_cluster_key(fi, fam.cluster_id, label[1])] = ll[i]

    m_curr = _marg(ds.ll_by_rate[key_curr], logw)
    cand_ids, cand_logp = [cid_curr], [0.0]
    if n_curr > 1:
        m_minus = _marg(ll[slot['curr_minus']], logw)
        m_single = _marg(ll[slot['singleton']], logw)
        cand_ids.append(-1)                       # split off a fresh singleton
        cand_logp.append((m_minus + m_single - m_curr)
                         + np.log(state.alpha_z) - np.log(max(n_curr - 1, 1)))
    for c_o in others:
        key_o = _cluster_key(fi, fam.cluster_id, c_o)
        m_o = _marg(ds.ll_by_rate[key_o], logw)
        m_op = _marg(ll[slot[('other_plus', c_o)]], logw)
        m_left = 0.0 if n_curr == 1 else _marg(ll[slot['curr_minus']], logw)
        base = m_left - m_curr        # cost of removing s from current cluster
        cand_ids.append(c_o)
        cand_logp.append(base + m_op - m_o
                         - np.log(state.alpha_z) + np.log(int(clusters[c_o].shape[0])))
    lp = np.array(cand_logp); lp -= lp.max(); p = np.exp(lp); p /= p.sum()
    choice = cand_ids[int(rng.choice(len(cand_ids), p=p))]
    if choice == cid_curr:
        return
    # apply the move + refresh the ll cache for affected clusters
    if choice == -1:
        new_id = int(fam.cluster_id.max()) + 1
        fam.cluster_id[s] = new_id
        ds.ll_by_rate[_cluster_key(fi, fam.cluster_id, new_id)] = ll[slot['singleton']]
        if n_curr > 1:
            ds.ll_by_rate[_cluster_key(fi, fam.cluster_id, cid_curr)] = ll[slot['curr_minus']]
        ds.ll_by_rate.pop(key_curr, None)
    else:
        # join cluster `choice`. Do the move FIRST so the remnant/merged keys
        # are computed from the NEW column sets; pop the two stale old keys.
        key_o_old = _cluster_key(fi, fam.cluster_id, choice)
        fam.cluster_id[s] = choice
        ds.ll_by_rate.pop(key_curr, None)          # s's old cluster (with s)
        ds.ll_by_rate.pop(key_o_old, None)         # choice's old cluster
        if n_curr > 1:
            ds.ll_by_rate[_cluster_key(fi, fam.cluster_id, cid_curr)] = ll[slot['curr_minus']]
        ds.ll_by_rate[_cluster_key(fi, fam.cluster_id, choice)] = ll[slot[('other_plus', choice)]]


def _col_invariant_static(ds, fi, col):
    """(K_a,) invariant single-column log-lik of `col` frozen at archetype k
    (enum400 diagonal class (k,k), m=1 spec, rate bin 0)."""
    K_a = ds.state.pi_archetype.shape[0]; mb = ds.state.m_bucket_for(1)
    colv = np.array([int(col)], np.int32); rb = _rate_bins(ds, fi, colv)
    specs = []
    for k in range(K_a):
        cls = np.zeros(mb, np.int32)
        cls[:1] = _aug(np.array([k * K_a + k], np.int32), rb, ds.G_s)
        specs.append((fi, colv, cls, mb))
    return _score_cluster_batch(ds.state, specs, tables=_tables_g(ds, 0))  # (K_a,)


def cn_invariant_column(ds, fi, cols, pos):
    """(K_c,) invariant-bin log-lik for all K_a^2 candidate classes at `pos` via
    per-column statics + rho-combine (columns independent when field frozen)."""
    K_a = ds.state.pi_archetype.shape[0]
    cols = np.asarray(cols, np.int32); m = len(cols)
    rho = np.asarray(ds.state.rho, float); lr0, lr1 = np.log(rho[0]), np.log(rho[1])
    a0 = np.arange(K_a * K_a) // K_a; a1 = np.arange(K_a * K_a) % K_a
    Ss = _col_invariant_static(ds, fi, int(cols[pos]))          # (K_a,) moving col
    # any m: frozen field => columns independent; sum FIXED columns' static
    # log-lik at their arch@0 / arch@1 (empty for m=1).
    lP0 = lP1 = 0.0
    for jp in range(m):
        if jp == pos:
            continue
        jcls = int(ds.state.families[fi].classes[int(cols[jp])])
        Sj = _col_invariant_static(ds, fi, int(cols[jp]))
        lP0 += Sj[jcls // K_a]; lP1 += Sj[jcls % K_a]
    return np.logaddexp(lr0 + Ss[a0] + lP0, lr1 + Ss[a1] + lP1)


def cn_move(ds: DiscoveryState, fi: int, s: int, rng) -> None:
    state = ds.state; fam = state.families[fi]; K_c = state.K_c; logw = ds.logw
    clusters = _cluster_columns_by_id(fam.cluster_id)
    cid = int(fam.cluster_id[s]); cols = clusters[cid]
    pos = int(np.where(cols == s)[0][0])
    specs = []
    for c in range(K_c):
        base = fam.classes[cols].copy(); base[pos] = c
        aug = _aug(base, _rate_bins(ds, fi, cols), ds.G_s)
        mb = state.m_bucket_for(len(cols)); cls = np.zeros(mb, np.int32); cls[:len(cols)] = aug
        specs.append((fi, cols, cls, mb))
    if ds.enum400:                                  # invariant bin via fast combine
        ll = np.empty((K_c, len(ds.rates)))
        ll[:, 0] = cn_invariant_column(ds, fi, cols, pos)
        for g in range(1, len(ds.rates)):
            ll[:, g] = _score_cluster_batch(ds.state, specs, tables=_tables_g(ds, g))
    else:
        ll = _score_specs_rates(ds, specs)
    marg = np.array([_marg(ll[c], logw) for c in range(K_c)])
    if ds.dm is not None:                           # mixture-of-DM class prior
        key = frozenset(int(x) for x in cols.tolist())
        marg = marg + ds.dm.cn_logprior(ds.state, fi, cols, s, key)
    marg -= marg.max(); p = np.exp(marg); p /= p.sum()
    c_new = int(rng.choice(K_c, p=p)); fam.classes[s] = c_new
    ds.ll_by_rate[_cluster_key(fi, fam.cluster_id, cid)] = ll[c_new]


def _refresh_device_class(ds, c):
    """Update only class c's slice of the cached device P_sub (~32MB/K_c), not
    the whole tensor, after its numpy slice was rebuilt by _refresh_aug_class."""
    import jax.numpy as jnp
    sl = slice(c * ds.G_s, (c + 1) * ds.G_s)
    ds.P_sub_aug_j = ds.P_sub_aug_j.at[:, sl].set(
        jnp.asarray(ds.P_sub_aug[:, sl]))


def arch_mh(ds: DiscoveryState, c: int, theta: int, rng) -> bool:
    state = ds.state; K_a = state.K_a; aa = state.arch_assignment; logw = ds.logw
    k_curr = int(aa[c, theta])
    # affected clusters: any with a class-c column. The MH ratio depends ONLY on
    # these -- every other cluster's LL is identical before/after and cancels --
    # so we sum marg() over `aff` (a few thousand) instead of corpus_ll() over
    # all ~55k twice. Also avoids subtracting two ~9M sums (catastrophic
    # cancellation); the small-delta form is strictly more accurate.
    aff = []
    for fi, fam in enumerate(state.families):
        for cid, cols in _cluster_columns_by_id(fam.cluster_id).items():
            if np.any(fam.classes[cols] == c):
                aff.append((fi, cols, _cluster_key(fi, fam.cluster_id, cid)))
    saved = {key: ds.ll_by_rate[key] for _, _, key in aff}
    old_aff = sum(_marg(saved[key], logw) for _, _, key in aff)
    k_prop = int(rng.integers(K_a - 1)); k_prop += (k_prop >= k_curr)
    aa[c, theta] = k_prop; state.apply_arch_slices([(c, theta)]); _refresh_aug_class(ds, c); _refresh_device_class(ds, c)
    new = _score_specs_rates(ds, [_spec_for(ds, fi, cols) for fi, cols, _ in aff])
    for (_, _, key), row in zip(aff, new):
        ds.ll_by_rate[key] = row
    new_aff = sum(_marg(ds.ll_by_rate[key], logw) for _, _, key in aff)
    if np.log(rng.uniform() + 1e-300) < new_aff - old_aff:
        return True
    aa[c, theta] = k_curr; state.apply_arch_slices([(c, theta)]); _refresh_aug_class(ds, c); _refresh_device_class(ds, c)
    ds.ll_by_rate.update(saved)
    return False


def _flip_stats(ds: DiscoveryState, confirmed_flip):
    """(n_pairs, flip_conf list, flip_other list) over current discovered pairs.
    Module-level: no closure over run state."""
    logw = ds.logw
    fc, fo = [], []
    for fi, fam in enumerate(ds.state.families):
        fid = fam.family_id
        for cid, cols in _cluster_columns_by_id(fam.cluster_id).items():
            if len(cols) != 2:
                continue
            v = ds.ll_by_rate.get(_cluster_key(fi, fam.cluster_id, cid))
            if v is None:
                continue
            tot = _marg(v, logw)
            phi = float(np.exp(_logsumexp(np.asarray(v[1:]) + logw[1:]) - tot))
            i, j = sorted(int(x) for x in cols)
            (fc if (fid, i, j) in confirmed_flip else fo).append(phi)
    return len(fc) + len(fo), fc, fo


def _log_line(ds: DiscoveryState, tag, t0, confirmed_flip, extra=""):
    import time
    npair = sum(1 for fam in ds.state.families
                for _, c in _cluster_columns_by_id(fam.cluster_id).items()
                if len(c) == 2)
    msg = (f"# {tag} corpus_ll={corpus_ll(ds):+.1f} n_pairs={npair} "
           f"t={time.time()-t0:.0f}s{extra}")
    if confirmed_flip is not None:
        _, fc, fo = _flip_stats(ds, confirmed_flip)
        if fc:
            msg += f" flip[conf]={np.mean(fc):.3f}"
        if fo:
            msg += f" flip[other]={np.mean(fo):.3f}"
    return msg


def _apply_move(ds, kind, item, rng):
    """Dispatch one move on externalized state ds. No closures/hidden state."""
    if kind == 0:
        z_move(ds, item[0], item[1], rng)
    elif kind == 1:
        cn_move(ds, item[0], item[1], rng)
    else:
        arch_mh(ds, item[0], item[1], rng)


def _dm_hcol(ds):
    """{fi: (n_col,) int} of each column's DM component, from ds.dm.h keyed by
    frozenset(cluster columns). None if no DM. Persisted by save_checkpoint."""
    if ds.dm is None:
        return None
    out = {}
    for fi, fam in enumerate(ds.state.families):
        arr = np.zeros(len(fam.cluster_id), np.int32)
        for cid, cols in _cluster_columns_by_id(fam.cluster_id).items():
            key = frozenset(int(x) for x in cols.tolist())
            arr[np.asarray(cols)] = ds.dm.h.get(key, 0)
        out[fi] = arr
    return out


def restore_dm(ds, ckpt_path):
    """Restore ds.dm (alpha/pi/H/alpha_pi + per-cluster component h) from a
    checkpoint after the partition has been loaded. No-op if no DM or the
    checkpoint predates DM persistence. Rebuilds h with the discovery frozenset
    key so cn_logprior/sample_h hit."""
    from . import corpus_state as cs
    if ds.dm is None:
        return
    hcol = cs.load_dm(ds.dm, ckpt_path, ds.state)
    if hcol is None:
        return
    ds.dm.h = {}
    for fi, fam in enumerate(ds.state.families):
        if fi not in hcol:
            continue
        for cid, cols in _cluster_columns_by_id(fam.cluster_id).items():
            key = frozenset(int(x) for x in cols.tolist())
            ds.dm.h[key] = int(hcol[fi][int(np.asarray(cols)[0])])


def run_interleaved(ds: DiscoveryState, n_sweeps, rng, fix_theta0=True,
                    arch_passes=1, confirmed_flip=None, log_stream=None,
                    out_dir=None, start_sweep=0, ckpt_every_sec=900):
    """Truly-random interleaving of z / cn / arch moves. Each sweep builds three
    lists -- z over columns, cn over columns, cθ over (class,field) (arch_passes
    copies) -- then repeatedly draws a list with probability proportional to its
    remaining length and pops the next item. No closures / hidden state (moves
    act on the externalized ds cache); the move batch shapes are unchanged so no
    extra JIT compiles vs the blocked schedule.

    Checkpoint/resume: if out_dir is given, an atomic rolling checkpoint of the
    canonical state (partition/classes/arch + step + rng) is written after every
    sweep AND every ckpt_every_sec within a sweep (0=off). Pass start_sweep>0
    (from corpus_state.load_checkpoint) to continue. Sweep-boundary resume is
    bit-exact (validated); a mid-sweep checkpoint records step=sweep-1, so resume
    replays the interrupted sweep from the partition/classes/arch it had already
    improved -- no learned state is lost, at most ckpt_every_sec of wall-clock."""
    import time
    from . import corpus_state as cs
    K_c, L = ds.state.arch_assignment.shape
    thetas = list(range(1, L)) if fix_theta0 else list(range(L))
    cols = [(fi, int(s)) for fi, fam in enumerate(ds.state.families)
            for s in range(fam.L)]
    arch_entries = [(c, t) for c in range(K_c) for t in thetas]
    t0 = time.time()

    if out_dir is not None:
        every = ("off" if not ckpt_every_sec
                 else f"every sweep + every {ckpt_every_sec}s within a sweep")
        b = f"# checkpoint: rolling, {every} -> {out_dir}/_chkpt.npz"
        print(b, flush=True)
        if log_stream:
            log_stream.write(b + "\n"); log_stream.flush()
    tag0 = f"sweep={start_sweep:3d}" if start_sweep else "sweep=  0"
    m = _log_line(ds, tag0, t0, confirmed_flip,
                  extra="  (resumed)" if start_sweep else "")
    print(m, flush=True)
    if log_stream:
        log_stream.write(m + "\n"); log_stream.flush()
    def _cur_clusters():
        return [(frozenset(int(x) for x in c.tolist()), fi, c)
                for fi, fam in enumerate(ds.state.families)
                for _, c in _cluster_columns_by_id(fam.cluster_id).items()]

    for sweep in range(start_sweep + 1, n_sweeps + 1):
        if ds.dm is not None:                       # Gibbs h_C | current partition+classes
            for key, fi, c in _cur_clusters():
                ds.dm.sample_h(ds.state, fi, c, key, rng)
        # (kind, item) lists; pop from the end == random draw (pre-shuffled).
        z_list = [cols[i] for i in rng.permutation(len(cols))]
        cn_list = [cols[i] for i in rng.permutation(len(cols))]
        arch_list = []
        for _ in range(arch_passes):
            arch_list += [arch_entries[i] for i in rng.permutation(len(arch_entries))]
        lists = [z_list, cn_list, arch_list]
        n_moves = [0, 0, 0]
        last_ckpt = time.time()
        while lists[0] or lists[1] or lists[2]:
            lens = (len(lists[0]), len(lists[1]), len(lists[2]))
            r = int(rng.integers(lens[0] + lens[1] + lens[2]))
            kind = 0 if r < lens[0] else (1 if r < lens[0] + lens[1] else 2)
            _apply_move(ds, kind, lists[kind].pop(), rng)
            n_moves[kind] += 1
            # intra-sweep safety net: replay this sweep on resume, but from the
            # partition it has already improved (step=sweep-1). Cheap vs a sweep.
            if (out_dir is not None and ckpt_every_sec
                    and time.time() - last_ckpt >= ckpt_every_sec):
                done = sum(n_moves)
                tot = lens[0] + lens[1] + lens[2] + done
                cs.save_checkpoint_atomic(
                    ds.state, out_dir, sweep - 1, rng, log_stream=log_stream,
                    t0=t0, label=f"sweep{sweep}~mid {done}/{tot} moves",
                    dm=ds.dm, dm_hcol=_dm_hcol(ds))
                last_ckpt = time.time()
        if ds.dm is not None:                       # update DM hyperparameters
            cur = _cur_clusters()
            ds.dm.update_alpha([(k, fi, c, ds.state) for k, fi, c in cur])
            ds.dm.update_pi([k for k, _, _ in cur], rng)
        if out_dir is not None:
            cs.save_checkpoint_atomic(ds.state, out_dir, sweep, rng,
                                      log_stream=log_stream, t0=t0,
                                      dm=ds.dm, dm_hcol=_dm_hcol(ds))
        m = _log_line(ds, f"sweep={sweep:3d}", t0, confirmed_flip,
                      extra=f"  moves z/cn/arch={n_moves[0]}/{n_moves[1]}/{n_moves[2]}")
        print(m, flush=True)
        if log_stream:
            log_stream.write(m + "\n"); log_stream.flush()
    return ds
