"""Checkpoint ELBO audit: does the moment-matching forward (what training
optimizes, `corpus_ll`) track the true tree likelihood?

For each cluster we score the SAME (PaddedTree, augmented-classes) two ways --
the moment-matching forward `_score_cluster_batch` and a certified reference --
then marginalise over field-rate bins and compare. For cap-2 clusters the
reference is the EXACT cap-2 forward (`bucketed_exact_ll`), so the gap is the
exact moment-projection error (mm - exact). m>2 clusters are skipped by the exact
reference (a per-site-ELBO reference is the follow-on).

Run via `train_marginal_dynfield.py --audit-elbo --resume <ckpt>` (reuses the
full build path); reports total corpus_ll both ways plus the per-cluster gap
distribution, then exits. Cheap (checkpoint cadence), no training-loop changes.
"""
from __future__ import annotations

import numpy as np

from .corpus_state import _materialize_padded_cluster, _cluster_columns_by_id
from .field_bp import _logsumexp


def _real_and_bins(state, specs):
    real, bins = [], []
    for (fi, cols, cls_padded, mb) in specs:
        pt = _materialize_padded_cluster(state.families[fi], cols, mb)
        real.append((pt, cls_padded))
        bins.append(state.bin_idx_by_family[fi])
    return real, bins


def _score_specs_exact(state, specs, P_sub_aug, pi_field_aug, rho, rho_chain, rates):
    """(len(specs), G) EXACT cap-2 log-lik per field-rate bin. Requires every
    cluster to be m<=2 (raises otherwise)."""
    from .exact_cap2_jax import bucketed_exact_ll
    for (fi, cols, cls_padded, mb) in specs:
        if int(mb) > 2:
            raise ValueError(f"exact reference is cap-2 only; got m_bucket={mb}")
    real, bins = _real_and_bins(state, specs)
    tables = {'P_sub': np.asarray(P_sub_aug), 'pi_field': np.asarray(pi_field_aug),
              'bin_centers': np.asarray(state.bin_centers)}
    out = np.zeros((len(specs), len(rates)), np.float64)
    for g, r in enumerate(rates):
        out[:, g] = np.asarray(
            bucketed_exact_ll(real, bins, rho, tables, float(rho_chain) * float(r)))
    return out


def run_audit(obj, kind, sample=None, rng=None):
    """Audit a supervised (kind='fr') or discovery (kind='ds') state.

    obj: FieldRateState or DiscoveryState (partition/classes already loaded).
    sample: if set, audit a random subset of this many clusters (else all).
    Returns a dict; also prints a summary.
    """
    state = obj.state
    if kind == 'fr':
        from . import field_rate_trainer as T
        clusters = list(obj.clusters)
        specs = [T._spec(obj, ci) for ci in range(len(clusters))]
        mms = T._score_specs_rates
    else:
        from . import field_rate_discovery as T
        clusters = []
        for fi, fam in enumerate(state.families):
            for _, cols in _cluster_columns_by_id(fam.cluster_id).items():
                clusters.append((fi, cols))
        specs = [T._spec_for(obj, fi, cols) for fi, cols in clusters]
        mms = T._score_specs_rates
    m_of = np.array([int(len(np.asarray(cols))) for _, cols in
                     ([(f, c) for f, c in clusters])], np.int32) if kind == 'fr' \
        else np.array([len(np.asarray(c)) for _, c in clusters], np.int32)

    idx = np.arange(len(specs))
    if sample and sample < len(specs):
        rng = rng or np.random.default_rng(0)
        idx = np.sort(rng.choice(len(specs), size=sample, replace=False))
    specs = [specs[i] for i in idx]; m_of = m_of[idx]

    rates = np.asarray(obj.rates, float); logw = obj.logw
    ll_mm = mms(obj, specs)                                    # (n, G)
    ll_ex = _score_specs_exact(state, specs, obj.P_sub_aug, obj.pi_field_aug,
                               state.rho, state.rho_chain, rates)

    def marg(row):
        return _logsumexp(np.asarray(row) + logw)
    mm = np.array([marg(ll_mm[c]) for c in range(len(specs))])
    ex = np.array([marg(ll_ex[c]) for c in range(len(specs))])
    gap = mm - ex                                             # mm - exact
    pair = m_of == 2
    out = {
        "n_clusters": int(len(specs)),
        "n_pairs": int(pair.sum()),
        "corpus_ll_mm": float(mm.sum()),
        "corpus_ll_exact": float(ex.sum()),
        "total_gap": float(gap.sum()),
        "gap_mean": float(gap.mean()), "gap_absmax": float(np.abs(gap).max()),
        "gap_q": [float(np.quantile(gap, q)) for q in (0.5, 0.9, 0.99)],
        "pair_gap_mean": float(gap[pair].mean()) if pair.any() else None,
        "pair_gap_absmax": float(np.abs(gap[pair]).max()) if pair.any() else None,
    }
    print("# ELBO AUDIT (moment-matching forward vs EXACT cap-2):", flush=True)
    print(f"#   clusters={out['n_clusters']} (pairs={out['n_pairs']})", flush=True)
    print(f"#   corpus_ll  mm={out['corpus_ll_mm']:+.2f}  exact={out['corpus_ll_exact']:+.2f}"
          f"  total_gap(mm-exact)={out['total_gap']:+.2f}", flush=True)
    print(f"#   per-cluster gap: mean={out['gap_mean']:+.4f} |max|={out['gap_absmax']:.4f} "
          f"median/p90/p99={out['gap_q'][0]:+.4f}/{out['gap_q'][1]:+.4f}/{out['gap_q'][2]:+.4f}",
          flush=True)
    if out["pair_gap_mean"] is not None:
        print(f"#   pairs only: gap mean={out['pair_gap_mean']:+.4f} "
              f"|max|={out['pair_gap_absmax']:.4f}", flush=True)
    return out
