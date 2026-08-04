"""Routine diagnostics for a trained tied-θ + Γ+I + mini-batch SVI run.

Computes and prints:

1. Model summary — per-archetype pi + top residues, arch_assignment, rho,
   rho_arch, pi_class marginals, Γ+I bin distribution.

2. Cluster residue-correlation ratio — mean(intra-cluster leaf-CLV
   correlation) / mean(inter-cluster). Higher than 1 indicates same-
   cluster columns actually co-vary in their leaf residue distributions.

3. Cluster class composition — per-cluster entropy over site classes.
   Compare to uniform baseline log(K_c). Low entropy = single-class
   clusters (correlated-mode coevolution). High entropy = mixed classes
   (θ-flip-driven synchrony across classes).

4. Class co-occurrence + log odds ratio — for each (c_1, c_2), count
   clusters where both appear together vs one-only vs neither. Log OR
   ranks pairs strongly attracted vs repelled.

5. Held-out LL — score leaf residues of held-out families under
   trained pi_class[c_s] vs LG08 stationary. Δ per obs in nats.

Args: `python svi_diagnostics.py <chkpt-dir> [--holdout-first N]
    [--corr-max-families M]`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from tkfdp.lg08 import PI_LG08
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from tkfdp.lg08 import PI_LG08

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
A = 20


def clusters_from_cluster_id(cluster_id):
    d = {}
    for s, cid in enumerate(cluster_id):
        cid = int(cid)
        if cid < 0:
            continue
        d.setdefault(cid, []).append(int(s))
    return d


def load_chkpt(chkpt_dir):
    arrs = np.load(chkpt_dir / 'state.npz')
    with open(chkpt_dir / 'meta.json') as f:
        meta = json.load(f)
    return arrs, meta


def print_model_summary(arrs, meta):
    pi_arch = arrs['pi_archetype']
    arch_assign = arrs['arch_assignment']
    K_a = pi_arch.shape[0]
    K_c = arch_assign.shape[0]
    L_theta = arch_assign.shape[1]
    usage = np.bincount(arch_assign.ravel(), minlength=K_a)

    print(f"=== iter {meta['iter']}  n_families={len(meta['family_names'])}  "
          f"K_c={K_c}  K_a={K_a}  L_θ={L_theta} ===")

    print(f"\n8 archetypes (pi_arch):")
    for k in range(K_a):
        p = pi_arch[k]
        ent = -np.sum(p * np.log(np.maximum(p, 1e-30)))
        order = np.argsort(p)[::-1]
        top = ' '.join(f'{ALPHABET[a]}:{p[a]:.2f}' for a in order[:5])
        print(f"  arch {k}: used={usage[k]:>2d}  ent={ent:.2f}  {top}")

    print(f"\narch_assignment ({K_c} × {L_theta}):")
    for c in range(K_c):
        print(f"  class {c}: {arch_assign[c].tolist()}")

    print(f"\nrho:      {arrs['rho'].tolist()}")
    print(f"rho_arch: {np.round(arrs['rho_arch'], 3).tolist()}")

    pi_class = arrs['pi_class']
    print(f"\npi_class marginal entropies (nats):")
    for c in range(K_c):
        p = pi_class[c]
        ent = -np.sum(p * np.log(np.maximum(p, 1e-30)))
        order = np.argsort(p)[::-1]
        top3 = ' '.join(f'{ALPHABET[a]}:{p[a]:.2f}' for a in order[:3])
        print(f"  class {c}: {ent:.2f}  {top3}")

    # Γ+I bin distribution
    srb_counts = np.zeros(5)
    n_fam = len(meta['family_names'])
    for fi in range(n_fam):
        key = f'site_rate_bin_{fi}'
        if key in arrs.files:
            b = arrs[key]
            for x in range(5):
                srb_counts[x] += int((b == x).sum())
    if srb_counts.sum() > 0:
        print(f"\nΓ+I bins across {int(srb_counts.sum())} columns:")
        labels = ['invariant', 'slow', 'below-avg', 'avg', 'fast']
        for x, (n, lab) in enumerate(zip(srb_counts, labels)):
            print(f"  bin {x} {lab:>10}: {int(n):>7d} = {n/srb_counts.sum()*100:.1f}%")


def _per_column_freq(clv_leaf):
    """Given (n_leaves, L, A) CLV, return (L, A) normalized average
    leaf residue distribution. Skip gap/ambiguous rows (row_sum ≠ 1 or
    row is uniform)."""
    row_sums = clv_leaf.sum(axis=-1, keepdims=True)
    valid = row_sums > 0.99
    norm = np.where(valid, clv_leaf / np.maximum(row_sums, 1e-30), 0.0)
    weights = valid[..., 0].astype(np.float64)
    col_freq = norm.sum(axis=0) / np.maximum(weights.sum(axis=0, keepdims=True).T
                                                  , 1e-30)
    # Above returns shape mismatch fix:
    col_freq = norm.sum(axis=0)                                  # (L, A)
    col_weights = weights.sum(axis=0)                            # (L,)
    col_freq = col_freq / np.maximum(col_weights[:, None], 1e-30)
    # Renormalize each row to unit sum
    row_sums2 = col_freq.sum(axis=1, keepdims=True)
    col_freq = col_freq / np.maximum(row_sums2, 1e-30)
    return col_freq


def cluster_correlation_ratio(chkpt_dir, arrs, meta, corpus_dir,
                                    max_families=50):
    """Ratio of mean(intra-cluster) to mean(inter-cluster) column-column
    residue-frequency correlation, averaged across families.

    Correlation is centred cosine on the per-column leaf-residue
    frequency vector. Same-cluster columns should be more similar than
    random pairs if the model captures shared evolution."""
    n_fam = min(len(meta['family_names']), max_families)
    ratios = []
    intra_list = []
    inter_list = []
    for fi in range(n_fam):
        fam = meta['family_names'][fi]
        fpath = Path(corpus_dir) / f'{fam}.npz'
        if not fpath.exists():
            continue
        d = np.load(fpath, allow_pickle=True)
        clv_leaf = d['clv'][:int(d['n_leaves'])]
        col_freq = _per_column_freq(clv_leaf)                     # (L, A)
        L = col_freq.shape[0]
        # Centre and cosine
        centred = col_freq - col_freq.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(centred, axis=1)
        centred = centred / np.maximum(norm[:, None], 1e-30)
        C = centred @ centred.T                                   # (L, L) cos

        # Cluster IDs
        key = f'cluster_id_{fi}'
        if key not in arrs.files:
            partner = arrs[f'partner_{fi}']
            cid = np.arange(L, dtype=np.int32)
            for s in range(L):
                t = int(partner[s])
                if t >= 0 and t > s:
                    cid[t] = cid[s]
            cluster_id = cid
        else:
            cluster_id = arrs[key]

        # Iterate cluster pairs (i<j)
        same = cluster_id[:, None] == cluster_id[None, :]
        mask_upper = np.triu(np.ones((L, L), dtype=bool), k=1)
        intra_mask = same & mask_upper
        inter_mask = (~same) & mask_upper
        if intra_mask.sum() < 2 or inter_mask.sum() < 2:
            continue
        intra_mean = float(C[intra_mask].mean())
        inter_mean = float(C[inter_mask].mean())
        intra_list.append(intra_mean)
        inter_list.append(inter_mean)
        ratios.append(intra_mean - inter_mean)

    print(f"\n=== Cluster residue-correlation (over {len(ratios)} families) ===")
    if not ratios:
        print("  no families processed")
        return
    intra_mean = float(np.mean(intra_list))
    inter_mean = float(np.mean(inter_list))
    diff = float(np.mean(ratios))
    print(f"  intra-cluster cos: {intra_mean:+.4f}")
    print(f"  inter-cluster cos: {inter_mean:+.4f}")
    print(f"  Δ (intra − inter): {diff:+.4f}")
    print(f"  Sign of Δ > 0 = same-cluster columns are MORE similar than random pairs.")


def cluster_class_composition(arrs, meta):
    """Per-cluster distribution over site classes; report mean entropy
    (weighted by cluster size) vs uniform baseline log(K_c)."""
    K_c = int(meta['K_c'])
    weighted_ent_sum = 0.0
    total_weight = 0.0
    entropies = []
    cluster_sizes = []
    for fi in range(len(meta['family_names'])):
        cls_key = f'cls_{fi}'
        cid_key = f'cluster_id_{fi}'
        if cls_key not in arrs.files:
            continue
        cls = arrs[cls_key]
        if cid_key in arrs.files:
            cluster_id = arrs[cid_key]
        else:
            partner = arrs[f'partner_{fi}']
            L = int(cls.shape[0])
            cluster_id = np.arange(L, dtype=np.int32)
            for s in range(L):
                t = int(partner[s])
                if t >= 0 and t > s:
                    cluster_id[t] = cluster_id[s]
        clusters = clusters_from_cluster_id(cluster_id)
        for cid, members in clusters.items():
            if len(members) < 2:
                continue
            classes_here = cls[members]
            hist = np.bincount(classes_here, minlength=K_c).astype(np.float64)
            p = hist / hist.sum()
            ent = -np.sum(p * np.log(np.maximum(p, 1e-30)))
            entropies.append(ent)
            cluster_sizes.append(len(members))
            weighted_ent_sum += ent * len(members)
            total_weight += len(members)

    print(f"\n=== Within-cluster class-composition entropy ===")
    if not entropies:
        print("  no multi-column clusters")
        return
    ents = np.array(entropies)
    sizes = np.array(cluster_sizes)
    baseline = np.log(K_c)
    print(f"  n multi-column clusters: {len(ents)}")
    print(f"  size-weighted mean entropy: {weighted_ent_sum/total_weight:.3f} nats")
    print(f"  uniform-K_c baseline:       {baseline:.3f} nats")
    print(f"  fraction with H = 0 (single-class): "
          f"{(ents == 0).mean()*100:.1f}%")
    print(f"  fraction with H > 0.5 · log(K_c): "
          f"{(ents > 0.5 * baseline).mean()*100:.1f}%")


def class_cooccurrence_and_odds(arrs, meta, n_permutations=100):
    """For each pair (c1, c2), compute LOR of joint presence in
    multi-column clusters, size-adjusted via a permutation null that
    shuffles site-class labels within each family while keeping
    cluster IDs fixed.

    Under the raw LOR, cluster-size heterogeneity confounds the null:
    a large cluster is more likely to contain both any pair of classes
    just by having many columns, giving positive raw LOR for every pair
    even under random assignment. The permutation-adjusted signal
    (observed LOR − mean shuffled LOR) removes that structural bias.
    """
    K_c = int(meta['K_c'])
    fam_data = []      # list of (cls_array, cluster_id_array) per family
    for fi in range(len(meta['family_names'])):
        cls_key = f'cls_{fi}'
        cid_key = f'cluster_id_{fi}'
        if cls_key not in arrs.files:
            continue
        cls = arrs[cls_key].copy()
        if cid_key in arrs.files:
            cluster_id = arrs[cid_key]
        else:
            partner = arrs[f'partner_{fi}']
            L = int(cls.shape[0])
            cluster_id = np.arange(L, dtype=np.int32)
            for s in range(L):
                t = int(partner[s])
                if t >= 0 and t > s:
                    cluster_id[t] = cluster_id[s]
        fam_data.append((cls, np.asarray(cluster_id)))

    if not fam_data:
        print("\n=== Class co-occurrence: no data ===")
        return

    def build_presence_matrix(fam_data):
        rows_local = []
        for cls, cluster_id in fam_data:
            clusters = clusters_from_cluster_id(cluster_id)
            for cid, members in clusters.items():
                if len(members) < 2:
                    continue
                v = np.zeros(K_c, dtype=bool)
                for m in members:
                    v[int(cls[m])] = True
                rows_local.append(v)
        return np.array(rows_local, dtype=bool)

    def pair_lors(P):
        lors = np.zeros((K_c, K_c))
        for c1 in range(K_c):
            for c2 in range(K_c):
                if c1 == c2:
                    continue
                a = int(np.sum(P[:, c1] & P[:, c2]))
                b = int(np.sum(P[:, c1] & ~P[:, c2]))
                c_only = int(np.sum(~P[:, c1] & P[:, c2]))
                d = int(np.sum(~P[:, c1] & ~P[:, c2]))
                lors[c1, c2] = np.log(
                    ((a + 0.5) * (d + 0.5))
                    / ((b + 0.5) * (c_only + 0.5)))
        return lors

    # Observed
    P_obs = build_presence_matrix(fam_data)
    N = P_obs.shape[0]
    print(f"\n=== Class co-occurrence within {N} multi-column clusters ===")
    p_c = P_obs.mean(axis=0)
    print(f"  Marginal presence per class:")
    for c in range(K_c):
        print(f"    class {c}: appears in {p_c[c]*100:.1f}% of clusters")
    lor_obs = pair_lors(P_obs)

    # Permutation null: shuffle site-class labels within each family.
    print(f"\n  Running {n_permutations} within-family class-label "
          f"permutations for size-adjusted null...", flush=True)
    rng = np.random.default_rng(0)
    lor_null_samples = []
    for _ in range(n_permutations):
        shuffled = []
        for cls, cluster_id in fam_data:
            perm = rng.permutation(len(cls))
            shuffled.append((cls[perm], cluster_id))
        P_shuffled = build_presence_matrix(shuffled)
        lor_null_samples.append(pair_lors(P_shuffled))
    lor_null = np.mean(lor_null_samples, axis=0)
    lor_null_sd = np.std(lor_null_samples, axis=0)
    lor_adj = lor_obs - lor_null
    z = lor_adj / np.maximum(lor_null_sd, 1e-6)

    rows = []
    for c1 in range(K_c):
        for c2 in range(c1 + 1, K_c):
            rows.append((c1, c2, lor_obs[c1, c2], lor_null[c1, c2],
                            lor_adj[c1, c2], z[c1, c2]))
    rows.sort(key=lambda r: -r[4])
    print(f"\n  Top-5 pairs by SIZE-ADJUSTED LOR (obs − null):")
    print(f"    (c1, c2)   obs   null   adj    Z-score")
    for c1, c2, obs, nul, adj, zv in rows[:5]:
        print(f"    ({c1}, {c2})   {obs:+.2f}  {nul:+.2f}  {adj:+.2f}   "
              f"{zv:+.2f}")
    print(f"\n  Bottom-5:")
    for c1, c2, obs, nul, adj, zv in rows[-5:]:
        print(f"    ({c1}, {c2})   {obs:+.2f}  {nul:+.2f}  {adj:+.2f}   "
              f"{zv:+.2f}")
    print(f"\n  Note: raw LORs positive-biased by cluster-size structure. "
          f"Adjusted values compare against permutation null that "
          f"preserves per-family class marginals + cluster geometry, so "
          f"positive adj_LOR = genuine attraction beyond size effect.")


def held_out_ll(arrs, meta, corpus_dir, held_out_families):
    """Compute mean per-observation log-likelihood of leaf residues under
    trained pi_class[c_s] vs fixed LG08 stationary, on families NOT in
    the training set."""
    pi_class = arrs['pi_class']
    K_c = pi_class.shape[0]
    pi_lg = np.asarray(PI_LG08)
    total_LL_trained = 0.0
    total_LL_lg = 0.0
    total_n = 0
    fams_done = 0
    for fam in held_out_families:
        fpath = Path(corpus_dir) / f'{fam}.npz'
        if not fpath.exists():
            continue
        d = np.load(fpath, allow_pickle=True)
        clv_leaf = d['clv'][:int(d['n_leaves'])]
        # Uniform class prior (we haven't run inference for these fams)
        col_freq = _per_column_freq(clv_leaf)
        # Score each column with the best trained class assignment
        # (equivalent to a marginal LL over uniform prior on c)
        for s in range(col_freq.shape[0]):
            obs = col_freq[s]
            if obs.sum() < 0.9:
                continue
            # log-sum-exp over K_c to marginalise class
            log_pcs = np.log(np.maximum(pi_class, 1e-30)) @ obs
            log_p_trained = float(_logsumexp(log_pcs) - np.log(K_c))
            log_p_lg = float(np.sum(obs * np.log(pi_lg)))
            total_LL_trained += log_p_trained
            total_LL_lg += log_p_lg
            total_n += 1
        fams_done += 1

    if total_n == 0:
        print("\n=== Held-out LL: no held-out data ===")
        return
    print(f"\n=== Held-out LL ({fams_done} fams, {total_n} obs) ===")
    print(f"  Trained (marginal over K_c): {total_LL_trained/total_n:+.4f} nats/obs")
    print(f"  LG08 null:                   {total_LL_lg/total_n:+.4f} nats/obs")
    delta = (total_LL_trained - total_LL_lg) / total_n
    print(f"  Δ per obs: {delta:+.4f} nats  ({delta/np.log(2):+.4f} bits)")
    print(f"  → Δ > 0 means trained > LG08 (model captures information).")


def _logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('chkpt_dir', type=Path)
    p.add_argument('--corpus-dir', type=Path,
                   default=Path('/home/yam/tkf-dp/data/pfam_processed_clv_top1000'))
    p.add_argument('--corr-max-families', type=int, default=50,
                   help='Cap the correlation-ratio scan (per-family '
                         'O(L^2 A) — expensive for large L).')
    p.add_argument('--holdout-first', type=int, default=20,
                   help='Score first N held-out (non-training) families '
                         'against LG08 null.')
    args = p.parse_args()

    arrs, meta = load_chkpt(args.chkpt_dir)

    print_model_summary(arrs, meta)
    cluster_correlation_ratio(arrs=arrs, meta=meta, corpus_dir=args.corpus_dir,
                                    max_families=args.corr_max_families,
                                    chkpt_dir=args.chkpt_dir)
    cluster_class_composition(arrs, meta)
    class_cooccurrence_and_odds(arrs, meta)

    # Choose held-out families: those in corpus but not in training
    training_set = set(meta['family_names'])
    all_files = sorted(Path(args.corpus_dir).glob('*.npz'))
    held_out = [f.stem for f in all_files if f.stem not in training_set]
    held_out_ll(arrs, meta, args.corpus_dir,
                    held_out[:args.holdout_first])


if __name__ == '__main__':
    main()
