"""Class-marginalised, field-rate-marginalised (+Gamma+I) cluster scorer.

Per appendix-tkfdp.tex "Rate heterogeneity (+Gamma+I)". For each frozen
cap-2 cluster we sum the field-HMM forward over:
  * all class labelings  -- K_c for a singleton, K_c^2 for a pair
    (a class flips among archetypes as the shared field evolves, so we
    cannot partition by archetype -- Prop. hier-fels);
  * the shared-global field-rate bins r_g (Gamma+I): rho_chain -> rho_chain*r_g,
    with r_0 = 0 the invariant (static / no-flip) bin.

The evidence unit is the per-(cluster, class-labeling) forward LL, scored
by the existing padded/chunked/JIT'd `_score_cluster_batch`; we only add
the labeling enumeration, the G+1 field-rate table variants (rebuild
beta/W at rho_chain*r_g, reuse the rho_chain-independent P_sub), and the
logsumexp aggregation. The per-site substitution rate m_n is baked into
P_sub at init (rate_hetero.init_site_rates) and so is already inside the
cached forward.

Returns, per cluster: the class+rate marginal log-evidence and the flip
posterior phi(C) = P(r_C != 0 | C).
"""
from __future__ import annotations

import numpy as np

from .tau_binning import field_kernel_tables


def enumerate_labelings(m: int, K_c: int) -> 'list[tuple[int, ...]]':
    """All class labelings of a size-m cluster. m=1 -> K_c; m=2 -> K_c^2."""
    if m == 1:
        return [(c,) for c in range(K_c)]
    if m == 2:
        return [(c1, c2) for c1 in range(K_c) for c2 in range(K_c)]
    raise ValueError(f"marginal scorer is cap-2 only; got m={m}")


def rate_bin_tables(state, r: float) -> dict:
    """Kernel tables with the field chain at rho_chain*r. Rebuilds only the
    field kernels beta/W (O(n_bins * L^2)); reuses the rho_chain-independent
    substitution P_sub from state.kernel_tables. r=0 -> beta=1, W=0 (static)."""
    import jax.numpy as jnp
    base = state.kernel_tables                       # has P_sub (+ beta/W)
    beta, W = field_kernel_tables(state.bin_centers, state.rho,
                                  state.rho_chain * float(r))
    out = dict(base)
    out['beta'] = jnp.asarray(np.asarray(beta))
    out['W'] = jnp.asarray(np.asarray(W))
    return out


def _logsumexp(a: np.ndarray, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else float(out)


def score_clusters_marginalized(
    state, clusters, rates, weights,
    class_log_prior=None,
):
    """Class- and field-rate-marginalised evidence + flip posterior.

    Args:
      state: CorpusState (provides families, K_c, kernel tables, scorer).
      clusters: list of (fi, cols) -- the frozen partition clusters
        (cols: np.ndarray of the cap-<=2 column indices).
      rates:   (G+1,) field-rate multipliers, rates[0] = 0 (invariant).
      weights: (G+1,) field-rate prior; weights[0] = p_inv.
      class_log_prior: optional callable labeling -> log prior; default
        uniform over labelings (symmetric class prior).

    Returns dict:
      cluster_ll:  (n_clusters,) marginal log-evidence per cluster.
      flip_post:   (n_clusters,) phi(C) = P(r_C != 0 | C).
      corpus_ll:   float, sum of cluster_ll.
    """
    from .corpus_state import (_score_cluster_batch, _cluster_classes_padded,
                               _materialize_padded_cluster)  # noqa: F401
    K_c = state.K_c
    G = len(rates)
    logw = np.log(np.asarray(weights, dtype=np.float64))

    # Build the flat spec list: one per (cluster, labeling). Track spans.
    specs = []
    spans = []          # (start, stop, m) into specs, per cluster
    lab_by_cluster = []
    for (fi, cols) in clusters:
        fam = state.families[fi]
        cols = np.asarray(cols, dtype=np.int32)
        m = int(cols.shape[0])
        mb = state.m_bucket_for(m)
        labs = enumerate_labelings(m, K_c)
        start = len(specs)
        for lab in labs:
            cls = np.zeros(fam.L, dtype=np.int32)
            cls[cols] = np.asarray(lab, dtype=np.int32)
            cls_padded = _cluster_classes_padded(fam, cols, mb)
            # override with this labeling's classes (in column order)
            cls_padded = cls_padded.copy()
            cls_padded[:m] = np.asarray(lab, dtype=np.int32)
            specs.append((fi, cols, cls_padded, mb))
        spans.append((start, len(specs), m))
        lab_by_cluster.append(labs)

    # Score every (cluster, labeling) spec under each field-rate bin.
    # ll_by_bin[g] : (len(specs),)
    ll_by_bin = np.zeros((G, len(specs)), dtype=np.float64)
    for g in range(G):
        tables = rate_bin_tables(state, rates[g])
        ll_by_bin[g] = _score_cluster_batch(state, specs, tables=tables)

    # Marginalise per cluster.
    n = len(clusters)
    cluster_ll = np.zeros(n, dtype=np.float64)
    flip_post = np.zeros(n, dtype=np.float64)
    for ci, (start, stop, m) in enumerate(spans):
        n_lab = stop - start
        # class log-prior over labelings (default uniform).
        if class_log_prior is None:
            clp = -np.log(n_lab) * np.ones(n_lab)
        else:
            clp = np.array([class_log_prior(l) for l in lab_by_cluster[ci]])
        # terms[g, lab] = logw[g] + clp[lab] + ll
        terms = ll_by_bin[:, start:stop] + logw[:, None] + clp[None, :]
        cluster_ll[ci] = _logsumexp(terms.ravel())
        # flip posterior = mass on g != 0
        num = _logsumexp(terms[1:].ravel()) if G > 1 else -np.inf
        flip_post[ci] = float(np.exp(num - cluster_ll[ci]))
    return dict(cluster_ll=cluster_ll, flip_post=flip_post,
               corpus_ll=float(cluster_ll.sum()))
