"""Training loop for the class-marginalised, field-rate-marginalised (+Gamma+I)
dynfield model. Archetypes frozen (LG-Cxx); the learned target is
arch_assignment, sampled by Gibbs over the class+rate-marginal evidence.

State:
  * a flat spec list of every (cluster, class-labeling), with spans per
    cluster and the set of classes each spec's labeling touches;
  * a cached ll_table (n_specs, G) of the field-HMM forward LL under each
    field-rate bin;
  * shared-global field-rate bins (Gamma+I), rho_chain, p_inv.

An arch move on (c*, theta*) recomputes only the specs whose labeling
contains class c* (1 of K_c per singleton, 2K_c-1 of K_c^2 per doublet),
batched under each rate bin, mirroring corpus_state.atomic_arch_gibbs but
over the marginalised evidence. See appendix-tkfdp.tex "Rate heterogeneity".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corpus_state import _score_cluster_batch, _cluster_classes_padded
from .marginal_scorer import enumerate_labelings, rate_bin_tables, _logsumexp


@dataclass
class MarginalState:
    state: object                       # CorpusState (frozen archetypes)
    clusters: list                      # [(fi, cols)]
    rates: np.ndarray                   # (G,) field-rate mult, rates[0]=0
    weights: np.ndarray                 # (G,) field-rate prior
    specs: list = field(default_factory=list)     # [(fi, cols, cls_padded, mb)]
    spans: list = field(default_factory=list)     # [(start, stop, m)] per cluster
    spec_labeling: list = field(default_factory=list)  # per spec: tuple of classes
    class_to_specs: dict = field(default_factory=dict) # class -> [spec idx]
    ll_table: np.ndarray = None         # (n_specs, G)

    @property
    def logw(self):
        return np.log(np.asarray(self.weights, dtype=np.float64))


def build_marginal_state(state, clusters, rates, weights) -> MarginalState:
    K_c = state.K_c
    ms = MarginalState(state=state, clusters=list(clusters),
                       rates=np.asarray(rates, float),
                       weights=np.asarray(weights, float))
    for (fi, cols) in ms.clusters:
        fam = state.families[fi]
        cols = np.asarray(cols, dtype=np.int32)
        m = int(cols.shape[0])
        mb = state.m_bucket_for(m)
        start = len(ms.specs)
        for lab in enumerate_labelings(m, K_c):
            cls_padded = _cluster_classes_padded(fam, cols, mb).copy()
            cls_padded[:m] = np.asarray(lab, dtype=np.int32)
            si = len(ms.specs)
            ms.specs.append((fi, cols, cls_padded, mb))
            ms.spec_labeling.append(tuple(lab))
            for c in set(lab):
                ms.class_to_specs.setdefault(c, []).append(si)
        ms.spans.append((start, len(ms.specs), m))
    return ms


def _score_specs_all_bins(ms: MarginalState, spec_idx) -> np.ndarray:
    """(len(spec_idx), G) LL for the given specs under every field-rate bin."""
    specs = [ms.specs[i] for i in spec_idx]
    out = np.zeros((len(spec_idx), len(ms.rates)), dtype=np.float64)
    for g in range(len(ms.rates)):
        tables = rate_bin_tables(ms.state, ms.rates[g])
        out[:, g] = _score_cluster_batch(ms.state, specs, tables=tables)
    return out


def init_ll_table(ms: MarginalState) -> None:
    ms.ll_table = _score_specs_all_bins(ms, list(range(len(ms.specs))))


def marginalize(ms: MarginalState):
    """Per-cluster class+rate marginal LL and flip posterior phi(C)."""
    logw = ms.logw
    n = len(ms.clusters)
    cluster_ll = np.zeros(n)
    flip_post = np.zeros(n)
    for ci, (start, stop, m) in enumerate(ms.spans):
        n_lab = stop - start
        clp = -np.log(n_lab)                       # uniform class prior
        terms = ms.ll_table[start:stop] + logw[None, :] + clp   # (n_lab, G)
        cluster_ll[ci] = _logsumexp(terms.ravel())
        num = _logsumexp(terms[:, 1:].ravel()) if len(ms.rates) > 1 else -np.inf
        flip_post[ci] = float(np.exp(num - cluster_ll[ci]))
    return cluster_ll, flip_post


def corpus_ll(ms: MarginalState) -> float:
    return float(marginalize(ms)[0].sum())


def arch_gibbs_move(ms: MarginalState, c: int, theta: int,
                    rho_arch: np.ndarray, rng) -> int:
    """K_a-way Gibbs on arch_assignment[c, theta] over the marginal evidence.
    Recomputes only the class-c specs; updates ms.ll_table and state on the
    sampled archetype. Returns the new archetype index."""
    state = ms.state
    K_a = state.K_a
    aa = state.arch_assignment
    k_curr = int(aa[c, theta])
    aff = ms.class_to_specs.get(c, [])
    if not aff:
        p = rho_arch / rho_arch.sum()
        k_new = int(rng.choice(K_a, p=p))
        if k_new != k_curr:
            aa[c, theta] = k_new
            state.apply_arch_slices([(c, theta)])
        return k_new

    # Baseline: marginal corpus LL contribution of the affected clusters is
    # entangled through logsumexp, so we score the FULL corpus LL per
    # candidate (affected specs vary; unaffected specs' ll cached).
    base_ll = ms.ll_table.copy()
    cand_ll_tables = {}       # k -> (len(aff), G) new ll for affected specs
    cand_corpus = np.full(K_a, -np.inf)
    for k in range(K_a):
        if k != k_curr:
            aa[c, theta] = k
            state.apply_arch_slices([(c, theta)])
            new_aff = _score_specs_all_bins(ms, aff)
            cand_ll_tables[k] = new_aff
            ms.ll_table[aff, :] = new_aff
        else:
            cand_ll_tables[k] = base_ll[aff, :]
            ms.ll_table[aff, :] = base_ll[aff, :]
        cand_corpus[k] = corpus_ll(ms)
        ms.ll_table = base_ll.copy()          # restore for next candidate
    # restore arch to current
    aa[c, theta] = k_curr
    state.apply_arch_slices([(c, theta)])

    log_p = cand_corpus + np.log(np.maximum(rho_arch, 1e-300))
    log_p -= log_p.max()
    p = np.exp(log_p); p /= p.sum()
    k_new = int(rng.choice(K_a, p=p))
    # commit
    ms.ll_table[aff, :] = cand_ll_tables[k_new]
    if k_new != k_curr:
        aa[c, theta] = k_new
        state.apply_arch_slices([(c, theta)])
    return k_new


def arch_mh_move(ms: MarginalState, c: int, theta: int, rng,
                 log_prior: 'np.ndarray | None' = None) -> bool:
    """Single-candidate MH on arch_assignment[c, theta] over the marginal
    evidence: propose k' != k_curr uniformly (symmetric), accept via the
    marginal corpus-LL ratio (+ optional archetype log-prior). Recomputes
    only the class-c specs. ~K_a x cheaper than the full Gibbs. Returns
    True on accept."""
    state = ms.state
    K_a = state.K_a
    aa = state.arch_assignment
    k_curr = int(aa[c, theta])
    aff = ms.class_to_specs.get(c, [])
    ll_curr = corpus_ll(ms)
    k_prop = int(rng.integers(K_a - 1))
    if k_prop >= k_curr:
        k_prop += 1                         # uniform over k != k_curr
    if not aff:
        aa[c, theta] = k_prop               # no data: prior-only
        state.apply_arch_slices([(c, theta)])
        return True
    saved = ms.ll_table[aff, :].copy()
    aa[c, theta] = k_prop
    state.apply_arch_slices([(c, theta)])
    ms.ll_table[aff, :] = _score_specs_all_bins(ms, aff)
    ll_prop = corpus_ll(ms)
    log_ratio = ll_prop - ll_curr
    if log_prior is not None:
        log_ratio += float(log_prior[k_prop] - log_prior[k_curr])
    if np.log(rng.uniform() + 1e-300) < log_ratio:
        return True                         # keep proposal
    # reject: restore
    aa[c, theta] = k_curr
    state.apply_arch_slices([(c, theta)])
    ms.ll_table[aff, :] = saved
    return False


def run_marginal_training(ms: MarginalState, n_sweeps: int, rng,
                          fix_theta0: bool = True, pinv_every: int = 2,
                          confirmed_flip_mask=None, log_stream=None,
                          out_dir=None, start_sweep=0):
    """MH sweeps over arch_assignment[c, theta] (theta>=1 if fix_theta0) with
    periodic p_inv M-steps. Logs corpus LL, mean flip posterior, and the
    flip posterior split on confirmed-flip vs other clusters.
    out_dir/start_sweep: atomic sweep-level checkpoint/resume (only arch evolves
    here; clusters/classes are fixed)."""
    import time
    from . import corpus_state as cs
    K_c, L = ms.state.arch_assignment.shape
    thetas = list(range(1, L)) if fix_theta0 else list(range(L))
    entries = [(c, t) for c in range(K_c) for t in thetas]
    t0 = time.time()

    def _log(sweep):
        cll, flip = marginalize(ms)
        msg = (f"# sweep={sweep:3d}  corpus_ll={cll.sum():+.2f}  "
               f"mean_flip={flip.mean():.3f}  p_inv={1-flip.mean():.3f}  "
               f"elapsed={time.time()-t0:.1f}s")
        if confirmed_flip_mask is not None:
            cf = np.asarray(confirmed_flip_mask, bool)
            if cf.any():
                msg += (f"  flip[confirmed]={flip[cf].mean():.3f}  "
                        f"flip[other]={flip[~cf].mean():.3f}")
        print(msg, flush=True)
        if log_stream:
            log_stream.write(msg + "\n"); log_stream.flush()

    _log(start_sweep)
    for sweep in range(start_sweep + 1, n_sweeps + 1):
        order = rng.permutation(len(entries))
        n_acc = 0
        for idx in order:
            c, theta = entries[idx]
            n_acc += arch_mh_move(ms, c, theta, rng)
        if sweep % pinv_every == 0:
            pass  # p_inv is a reported diagnostic; prior stays symmetric here
        if out_dir is not None:
            cs.save_checkpoint_atomic(ms.state, out_dir, sweep, rng,
                                      log_stream=log_stream, t0=t0)
        _log(sweep)
    return ms


def pinv_mstep(ms: MarginalState, a_p=1.0, b_p=1.0) -> float:
    """Beta(a_p,b_p)-conjugate update of p_inv from flip posteriors:
    p_inv <- (a_p-1 + sum_C (1-phi(C))) / (a_p+b_p-2 + |C|)."""
    _, flip = marginalize(ms)
    q_inv = 1.0 - flip
    p_inv = (a_p - 1.0 + q_inv.sum()) / (a_p + b_p - 2.0 + len(flip))
    return float(np.clip(p_inv, 1e-4, 1 - 1e-4))
