"""Stage C: MCMC for the archetype-orbit permutation field (supervised /
fixed-cluster first). State = (site-class/base-archetype per column, orbit
partition of the K archetypes). Field trajectory + rate are marginalised in the
per-cluster likelihood (orbit_scorer_jax forwards, validated == numpy).

Moves:
  * cn      : Gibbs on a column's base archetype (== class), over K.
  * orbit   : split/merge on the archetype orbit partition (the novel move),
              proposed on archetype pairs that actually co-occur in a cluster
              (candidate flips); MH with a DP(alpha_orbit) prior.

Scoring here is per-tree (correct, validated); batching is a later optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corpus_state import _materialize_padded_cluster, _cluster_columns_by_id
from .marginal_scorer import _logsumexp
from . import orbit_scorer_jax as oj


@dataclass
class OrbitState:
    state: object
    clusters: list                       # [(fi, cols np.int32)]
    orbit_id: np.ndarray                 # (K,) archetype -> orbit label
    rates: np.ndarray                    # field-rate grid (rates[0]=0 invariant)
    weights: np.ndarray
    alpha_orbit: float = 50.0
    tables: dict = None
    ll: np.ndarray = None                # (n_clusters, G)

    @property
    def logw(self):
        return np.log(np.asarray(self.weights, float))


def build_orbit_state(state, clusters, rates, weights, alpha_orbit=50.0):
    K = state.pi_archetype.shape[0]
    os = OrbitState(state=state, clusters=[(int(fi), np.asarray(c, np.int32))
                                           for fi, c in clusters],
                    orbit_id=np.arange(K, dtype=np.int64),
                    rates=np.asarray(rates, float),
                    weights=np.asarray(weights, float), alpha_orbit=alpha_orbit)
    # Per-archetype transition bins -- orbit-INDEPENDENT, so never rebuilt on an
    # orbit move (the move only changes orbit_id, which re-enumerates Theta_C).
    os.tables = oj.build_arch_P(state.pi_archetype, state.S, state.bin_centers)
    return os


def score_cluster(os: OrbitState, ci: int) -> np.ndarray:
    """(G,) log-lik over field-rate bins for cluster ci. INDEPENDENT per-orbit
    fields: group the columns by orbit and sum one forward per group (columns
    sharing an orbit stay coupled; different orbits factorise)."""
    from collections import defaultdict
    fi, cols = os.clusters[ci]
    fam = os.state.families[fi]
    classes = [int(fam.classes[c]) for c in cols.tolist()]
    groups = defaultdict(list)
    for pos, c in enumerate(classes):
        groups[int(os.orbit_id[c])].append(pos)
    out = np.zeros(len(os.rates))
    for _, positions in groups.items():
        gcols = np.asarray([int(cols[p]) for p in positions], np.int32)
        gcls = [classes[p] for p in positions]
        pt = _materialize_padded_cluster(fam, gcols, os.state.m_bucket_for(len(gcols)))
        for g, r in enumerate(os.rates):
            out[g] += oj.score_cluster(pt, gcls, os.orbit_id, os.tables,
                                       os.state.rho_chain, float(r))
    return out


def _cluster_units(os: OrbitState, ci: int):
    """List of (pt, a_cols) units for cluster ci -- one per orbit-group."""
    fi, cols = os.clusters[ci]
    fam = os.state.families[fi]
    classes = [int(fam.classes[c]) for c in cols.tolist()]
    from collections import defaultdict
    groups = defaultdict(list)
    for pos, c in enumerate(classes):
        groups[int(os.orbit_id[c])].append(pos)
    units = []
    for _, positions in groups.items():
        gcols = np.asarray([int(cols[p]) for p in positions], np.int32)
        gcls = [classes[p] for p in positions]
        pt = _materialize_padded_cluster(fam, gcols, os.state.m_bucket_for(len(gcols)))
        _, a_cols = oj.joint_field(gcls, os.orbit_id)
        units.append((pt, a_cols))
    return units


def score_clusters_batched(os: OrbitState, cis) -> np.ndarray:
    """(len(cis), G) -- all orbit-group units of all clusters scored in one
    batched call per rate bin (the per-tree dispatch bottleneck removed)."""
    cis = list(cis)
    all_units, owner = [], []
    for k, ci in enumerate(cis):
        for u in _cluster_units(os, ci):
            all_units.append(u); owner.append(k)
    out = np.zeros((len(cis), len(os.rates)))
    for g, r in enumerate(os.rates):
        ll = oj.score_units_batched(all_units, os.tables, os.state.rho_chain, float(r))
        for ui, k in enumerate(owner):
            out[k, g] += ll[ui]
    return out


def init_ll(os: OrbitState):
    os.ll = score_clusters_batched(os, range(len(os.clusters)))


def _marg(row, logw):
    return _logsumexp(np.asarray(row) + logw)


def corpus_ll(os: OrbitState) -> float:
    lw = os.logw
    return float(sum(_marg(r, lw) for r in os.ll))


# ------------------------------------------------------------------ cn move

def cn_move(os: OrbitState, ci: int, pos: int, rng):
    """Gibbs the base archetype (== class) of one column over K archetypes, with
    all K candidates' units scored in one batched call per rate."""
    fi, cols = os.clusters[ci]
    fam = os.state.families[fi]; col = int(cols[pos]); K = os.state.pi_archetype.shape[0]
    lw = os.logw; old = int(fam.classes[col])
    all_units, owner = [], []
    for k in range(K):
        fam.classes[col] = k
        for u in _cluster_units(os, ci):
            all_units.append(u); owner.append(k)
    fam.classes[col] = old
    rows = np.zeros((K, len(os.rates)))
    for g, r in enumerate(os.rates):
        ll = oj.score_units_batched(all_units, os.tables, os.state.rho_chain, float(r))
        for ui, k in enumerate(owner):
            rows[k, g] += ll[ui]
    margs = np.array([_marg(rows[k], lw) for k in range(K)])
    p = np.exp(margs - margs.max()); p /= p.sum()
    ch = int(rng.choice(K, p=p)); fam.classes[col] = ch; os.ll[ci] = rows[ch]


# --------------------------------------------------------------- orbit moves

def _clusters_touching(os, archs):
    """cluster indices with a column whose class is in the set `archs`."""
    archs = set(int(a) for a in archs)
    out = []
    for ci, (fi, cols) in enumerate(os.clusters):
        cl = os.state.families[fi].classes
        if any(int(cl[c]) in archs for c in cols.tolist()):
            out.append(ci)
    return out


def _rescore(os, cis):
    cis = list(cis)
    rows = score_clusters_batched(os, cis)
    return {ci: rows[k] for k, ci in enumerate(cis)}


def orbit_move(os: OrbitState, a: int, b: int, rng) -> str:
    """Propose toggling the orbit relation of archetypes a,b (both must be in
    singleton orbits to MERGE, or together in a pair-orbit to SPLIT). MH with a
    DP(alpha_orbit) prior. s_max=2. Returns 'merge'/'split'/'reject'/'skip'."""
    oid = os.orbit_id; lw = os.logw
    same = oid[a] == oid[b]
    sz_a = int((oid == oid[a]).sum()); sz_b = int((oid == oid[b]).sum())
    if not same and (sz_a > 1 or sz_b > 1):
        return 'skip'                              # would exceed s_max=2
    aff = _clusters_touching(os, [a, b])
    old_ll = {ci: os.ll[ci].copy() for ci in aff}
    old_marg = sum(_marg(os.ll[ci], lw) for ci in aff)
    old_oid = oid.copy()
    if same:                                        # SPLIT a,b apart
        oid[b] = int(oid.max()) + 1; dprior = +np.log(os.alpha_orbit)   # +1 orbit
        kind = 'split'
    else:                                           # MERGE a,b together
        oid[oid == oid[b]] = oid[a]; dprior = -np.log(os.alpha_orbit)   # -1 orbit
        kind = 'merge'
    # tables are orbit-independent; the move only changes orbit_id (re-enumerates
    # Theta_C inside score_cluster), so no table rebuild.
    new = _rescore(os, aff)
    new_marg = sum(_marg(new[ci], lw) for ci in aff)
    if np.log(rng.uniform() + 1e-300) < (new_marg - old_marg) + dprior:
        for ci, row in new.items():
            os.ll[ci] = row
        return kind
    os.orbit_id = old_oid                            # revert
    for ci in aff:
        os.ll[ci] = old_ll[ci]
    return 'reject'


def cooccurring_pairs(os):
    """(a,b) archetype pairs that co-occur in a size-2 cluster (candidate flips)."""
    s = set()
    for fi, cols in os.clusters:
        if len(cols) == 2:
            cl = os.state.families[fi].classes
            a, b = int(cl[cols[0]]), int(cl[cols[1]])
            if a != b:
                s.add((min(a, b), max(a, b)))
    return sorted(s)


def flip_posterior(os: OrbitState) -> np.ndarray:
    """phi(C) per cluster = P(active field-rate | C) for SAME-ORBIT pair
    clusters (structural coupling), else 0."""
    lw = os.logw; out = np.zeros(len(os.clusters))
    for ci, (fi, cols) in enumerate(os.clusters):
        if len(cols) != 2:
            continue
        cl = os.state.families[fi].classes
        a, b = int(cl[cols[0]]), int(cl[cols[1]])
        if a != b and os.orbit_id[a] == os.orbit_id[b]:
            tot = _marg(os.ll[ci], lw)
            out[ci] = float(np.exp(_logsumexp(os.ll[ci][1:] + lw[1:]) - tot))
    return out


def run_training(os, n_sweeps, rng, confirmed_mask=None, do_cn=True, log=print):
    import time
    cols = [(ci, p) for ci, (_, c) in enumerate(os.clusters) for p in range(len(c))]
    t0 = time.time()

    def _log(sw, extra=""):
        phi = flip_posterior(os)
        n_pair_orbits = int(os.orbit_id.size - len(np.unique(os.orbit_id)))  # K - #labels
        coupled = int(sum(1 for fi, c in os.clusters
                          if len(c) == 2 and
                          os.orbit_id[os.state.families[fi].classes[c[0]]] ==
                          os.orbit_id[os.state.families[fi].classes[c[1]]]))
        m = (f"# orbit sweep={sw:3d} corpus_ll={corpus_ll(os):+.1f} "
             f"n_pair_orbits={n_pair_orbits} coupled_pairs={coupled} "
             f"t={time.time()-t0:.0f}s{extra}")
        if confirmed_mask is not None:
            cf = np.asarray(confirmed_mask, bool)
            phi_pair = phi[[ci for ci, (_, c) in enumerate(os.clusters) if len(c) == 2]]
            if cf.any() and (~cf).any():
                m += (f" phi[conf]={phi_pair[cf].mean():.3f} "
                      f"phi[other]={phi_pair[~cf].mean():.3f}")
        log(m)

    _log(0)
    for sw in range(1, n_sweeps + 1):
        if do_cn:
            for ci, p in [cols[i] for i in rng.permutation(len(cols))]:
                cn_move(os, ci, p, rng)
        n = {'merge': 0, 'split': 0, 'reject': 0, 'skip': 0}
        for (a, b) in cooccurring_pairs(os):
            n[orbit_move(os, a, b, rng)] += 1
        _log(sw, extra=f"  moves {n}")
    return os
