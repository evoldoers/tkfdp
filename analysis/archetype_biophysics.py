"""Steps 1 and 2 of the archetype-checkpoint analysis (per Ian, 2026-07-01):

  1. Archetype biophysics: per-archetype hydrophobicity / charge / aromaticity /
     volume, so we can label each k with its dominant property.

  2. Coordinated field-flip detection: for each field-flip pair (theta, theta')
     find pairs of site classes whose archetype assignment changes in a
     *compensatory* way -- e.g. (positive, negative) -> (negative, positive)
     across a field flip. Salt-bridge flips (K-D vs D-K) are the motivating
     example; more generally, we score each column pair by minus the product
     of their per-property deltas (opposite-sign deltas score positive).

Alphabet order matches src/tkfdp/lg08.py: ACDEFGHIKLMNPQRSTVWY.

Usage:
    python analysis/archetype_biophysics.py <checkpoint_dir>

<checkpoint_dir> should contain state.npz + meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ALPHA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy indices, indexed by alphabetical AA order.
KYTE_DOOLITTLE = np.array([
    1.8,   # A
    2.5,   # C
    -3.5,  # D
    -3.5,  # E
    2.8,   # F
    -0.4,  # G
    -3.2,  # H
    4.5,   # I
    -3.9,  # K
    3.8,   # L
    1.9,   # M
    -3.5,  # N
    -1.6,  # P
    -3.5,  # Q
    -4.5,  # R
    -0.8,  # S
    -0.7,  # T
    4.2,   # V
    -0.9,  # W
    -1.3,  # Y
], dtype=np.float64)

# Formal charge at pH ~7 (H taken as neutral / mildly positive).
CHARGE = np.zeros(20, dtype=np.float64)
CHARGE[ALPHA_ORDER.index('D')] = -1.0
CHARGE[ALPHA_ORDER.index('E')] = -1.0
CHARGE[ALPHA_ORDER.index('K')] = +1.0
CHARGE[ALPHA_ORDER.index('R')] = +1.0
CHARGE[ALPHA_ORDER.index('H')] = +0.1  # partial at physiological pH

# Aromatic indicator (F, W, Y).
AROMATIC = np.zeros(20, dtype=np.float64)
for aa in "FWY":
    AROMATIC[ALPHA_ORDER.index(aa)] = 1.0

# Zamyatnin residue volumes (A^3), indexed alphabetically.
VOLUME = np.array([
    88.6,   # A
    108.5,  # C
    111.1,  # D
    138.4,  # E
    189.9,  # F
    60.1,   # G
    153.2,  # H
    166.7,  # I
    168.6,  # K
    166.7,  # L
    162.9,  # M
    114.1,  # N
    112.7,  # P
    143.8,  # Q
    173.4,  # R
    89.0,   # S
    116.1,  # T
    140.0,  # V
    227.8,  # W
    193.6,  # Y
], dtype=np.float64)

PROPERTIES = {
    'hydropathy': KYTE_DOOLITTLE,
    'charge':     CHARGE,
    'aromatic':   AROMATIC,
    'volume':     VOLUME,
}


def load_state(chkpt_dir: Path):
    """Load pi_archetype, arch_assignment, rho, rho_arch, rho_chain and
    per-MSA cluster_id + cls arrays from an SVI checkpoint directory."""
    st = np.load(chkpt_dir / "state.npz", allow_pickle=False)
    meta = json.loads((chkpt_dir / "meta.json").read_text())
    pi_arch = st['pi_archetype'] if 'pi_archetype' in st.files else None
    arch = st['arch_assignment'] if 'arch_assignment' in st.files else None
    rho = st['rho'] if 'rho' in st.files else None
    rho_arch = st['rho_arch'] if 'rho_arch' in st.files else None
    rho_chain = (float(st['rho_chain']) if 'rho_chain' in st.files
                    else float(meta.get('rho_chain', np.nan)))
    if pi_arch is None or arch is None:
        raise SystemExit(
            f"checkpoint at {chkpt_dir} does not contain pi_archetype / "
            f"arch_assignment; keys present: {sorted(st.files)}")
    # cluster_id_<msa_idx> and cls_<msa_idx> for each MSA in the corpus.
    cluster_id_by_msa: 'list[np.ndarray]' = []
    cls_by_msa: 'list[np.ndarray]' = []
    idx = 0
    while f'cluster_id_{idx}' in st.files:
        cluster_id_by_msa.append(st[f'cluster_id_{idx}'])
        cls_key = f'cls_{idx}'
        cls_by_msa.append(st[cls_key] if cls_key in st.files else None)
        idx += 1
    return dict(pi_arch=pi_arch, arch=arch, rho=rho, rho_arch=rho_arch,
                  rho_chain=rho_chain, meta=meta,
                  cluster_id_by_msa=cluster_id_by_msa,
                  cls_by_msa=cls_by_msa)


def step1_archetype_biophysics(pi_arch: np.ndarray) -> np.ndarray:
    """Compute per-archetype expected biophysical properties. Returns
    (K_a, n_props) matrix `scores`."""
    K_a, A = pi_arch.shape
    assert A == 20, f"expected 20 amino acids, got {A}"
    scores = np.zeros((K_a, len(PROPERTIES)), dtype=np.float64)
    for j, (name, vec) in enumerate(PROPERTIES.items()):
        scores[:, j] = pi_arch @ vec
    return scores


def print_step1(scores: np.ndarray, pi_arch: np.ndarray, rho_arch=None):
    K_a, _ = scores.shape
    prop_names = list(PROPERTIES.keys())
    print("\n===== Step 1: archetype biophysics =====")
    print(f"{'k':>3}  ", end="")
    for name in prop_names:
        print(f"{name:>10}  ", end="")
    print("weight  |  top-4 residues")
    for k in range(K_a):
        row = f"{k:>3}  "
        for j in range(len(prop_names)):
            row += f"{scores[k, j]:>+10.3f}  "
        w = float(rho_arch[k]) if rho_arch is not None else float('nan')
        row += f"{w:>5.3f}  |  "
        top = np.argsort(pi_arch[k])[::-1][:4]
        row += "  ".join(f"{ALPHA_ORDER[i]}={pi_arch[k, i]:.2f}"
                            for i in top)
        print(row)


def _label_archetype(k: int, scores_row: np.ndarray) -> str:
    """One-word label based on which property is most extreme (z-score wise
    relative to the average across archetypes -- caller passes normalised
    scores)."""
    props = list(PROPERTIES.keys())
    j = int(np.argmax(np.abs(scores_row)))
    sign = "+" if scores_row[j] > 0 else "-"
    return f"{sign}{props[j]}"


def _enumerate_flip_scores(arch: np.ndarray, scores_std: np.ndarray,
                             rho=None):
    """Return a list of (total_score, dominant_prop_idx) tuples for every
    (theta, theta', c1, c2) 4-tuple with both classes mediator-active
    along the flip. Used by both step2 (ranking) and step4 (permutation
    null)."""
    K_c, L = arch.shape
    K_a, n_props = scores_std.shape
    out = []
    for theta in range(L):
        for theta_p in range(L):
            if theta_p == theta:
                continue
            fw = 1.0 if rho is None else float(rho[theta] * rho[theta_p])
            for c in range(K_c):
                k_c_a = int(arch[c, theta]); k_c_b = int(arch[c, theta_p])
                if k_c_a == k_c_b:
                    continue
                for cp in range(c + 1, K_c):
                    k_cp_a = int(arch[cp, theta]); k_cp_b = int(arch[cp, theta_p])
                    if k_cp_a == k_cp_b:
                        continue
                    dc = scores_std[k_c_b] - scores_std[k_c_a]
                    dcp = scores_std[k_cp_b] - scores_std[k_cp_a]
                    per_prop = -dc * dcp
                    total = float(per_prop.sum()) * fw
                    if total <= 0:
                        continue
                    j_max = int(np.argmax(np.abs(per_prop)))
                    out.append((total, j_max))
    return out


def step2_coordinated_flips(arch: np.ndarray, scores_raw: np.ndarray,
                              rho=None, top_n: int = 20):
    """For every field-flip pair (theta, theta') and every column-class pair
    (c, c') detect compensatory flips.

    Compensation score for property p at (c, c', theta, theta'):
      delta_c   = scores_raw[arch[c,  theta'],  p] - scores_raw[arch[c,  theta], p]
      delta_c'  = scores_raw[arch[c', theta'], p] - scores_raw[arch[c', theta], p]
      score     = -delta_c * delta_c'
                  (positive when the two deltas point in opposite directions
                  along property p; magnitude reflects how big the swings are)

    Aggregated across properties by summing, weighted by the field-flip
    probability rho[theta']*rho[theta] to prefer flips between well-populated
    field states.

    Prints the top_n coordinated flips grouped by (theta -> theta').
    """
    K_c, L = arch.shape
    K_a, n_props = scores_raw.shape
    prop_names = list(PROPERTIES.keys())

    # Standardise property scores so aromatic / charge / hydropathy /
    # volume enter the compensation product on comparable scales.
    scores_std = scores_raw / (scores_raw.std(axis=0, keepdims=True) + 1e-9)

    results = []
    for theta in range(L):
        for theta_p in range(L):
            if theta_p == theta:
                continue
            field_w = 1.0 if rho is None else float(rho[theta] * rho[theta_p])
            for c in range(K_c):
                k_c_a = int(arch[c, theta]);  k_c_b = int(arch[c, theta_p])
                if k_c_a == k_c_b:
                    continue  # class c is a specializer along this flip
                for cp in range(c + 1, K_c):
                    k_cp_a = int(arch[cp, theta]); k_cp_b = int(arch[cp, theta_p])
                    if k_cp_a == k_cp_b:
                        continue
                    delta_c = scores_std[k_c_b] - scores_std[k_c_a]
                    delta_cp = scores_std[k_cp_b] - scores_std[k_cp_a]
                    per_prop = -delta_c * delta_cp
                    total = float(per_prop.sum()) * field_w
                    if total <= 0:
                        continue
                    # Identify the dominant compensating property.
                    j_max = int(np.argmax(np.abs(per_prop)))
                    results.append({
                        'theta': theta, 'theta_p': theta_p,
                        'c1': c, 'c2': cp,
                        'k_c1_before': k_c_a, 'k_c1_after': k_c_b,
                        'k_c2_before': k_cp_a, 'k_c2_after': k_cp_b,
                        'total_score': total,
                        'dominant_prop': prop_names[j_max],
                        'dominant_score': float(per_prop[j_max]),
                        'delta_c1_prop': float(delta_c[j_max]),
                        'delta_c2_prop': float(delta_cp[j_max]),
                    })
    results.sort(key=lambda r: r['total_score'], reverse=True)

    print(f"\n===== Step 2: top {top_n} coordinated / compensatory flips ====")
    print(f"{'theta':>5} {'theta_p':>7}  {'c1':>3} {'c2':>3}  "
            f"{'k(c1,theta) -> k(c1,theta_p)':>28}  "
            f"{'k(c2,theta) -> k(c2,theta_p)':>28}  "
            f"{'total':>7}  {'prop':>10}  "
            f"{'delta_c1':>9}  {'delta_c2':>9}")
    for r in results[:top_n]:
        c1_flip = f"{r['k_c1_before']} -> {r['k_c1_after']}"
        c2_flip = f"{r['k_c2_before']} -> {r['k_c2_after']}"
        print(f"{r['theta']:>5} {r['theta_p']:>7}  "
                f"{r['c1']:>3} {r['c2']:>3}  "
                f"{c1_flip:>28}  {c2_flip:>28}  "
                f"{r['total_score']:>+7.3f}  {r['dominant_prop']:>10}  "
                f"{r['delta_c1_prop']:>+9.3f}  {r['delta_c2_prop']:>+9.3f}")
    return results


def step4_permutation_test(arch: np.ndarray, scores_raw: np.ndarray,
                             rho=None, rho_arch=None,
                             n_perms: int = 1000, top_n: int = 10,
                             mode: str = "row", rng=None):
    """Null test on arch_assignment.

    mode='row': for each class c, shuffle the L archetype values across
      field states. Preserves the multiset of archetypes each class uses
      (specialisers stay specialisers, mediators keep the same 'pool' of
      alternating archetypes). Tests whether the specific *alignment* of
      archetypes across θ values is informative given the same usage
      multiset per class. CONSERVATIVE -- easily swamped when the
      compensation potential is baked into the multiset.

    mode='iid': for each (c, θ), draw arch[c, θ] independently from
      Categorical(rho_arch). Preserves only the marginal archetype-usage
      distribution rho_arch; destroys all class × θ structure. Tests
      whether the class-θ coordination pattern is stronger than what you'd
      see given the same rho_arch marginal alone. STRONGER null.

    For each shuffle, take the top-1 compensation score across all
    (c1, c2, θ, θ') 4-tuples. Observed top-N compared to this
    distribution.
    """
    K_c, L = arch.shape
    K_a = int(arch.max() + 1) if rho_arch is None else int(rho_arch.shape[0])
    scores_std = scores_raw / (scores_raw.std(axis=0, keepdims=True) + 1e-9)

    observed = sorted(_enumerate_flip_scores(arch, scores_std, rho=rho),
                        reverse=True)[:top_n]
    observed_scores = [t[0] for t in observed]

    if rng is None:
        rng = np.random.default_rng(0)
    null_top = np.zeros(n_perms, dtype=np.float64)
    if mode == "row":
        for i in range(n_perms):
            arch_perm = arch.copy()
            for c in range(K_c):
                rng.shuffle(arch_perm[c])
            hits = _enumerate_flip_scores(arch_perm, scores_std, rho=rho)
            null_top[i] = max([t[0] for t in hits], default=0.0)
    elif mode == "iid":
        if rho_arch is None:
            raise ValueError("mode='iid' requires rho_arch")
        p = np.asarray(rho_arch, dtype=np.float64)
        p = p / p.sum()
        for i in range(n_perms):
            arch_perm = rng.choice(K_a, size=arch.shape, p=p)
            hits = _enumerate_flip_scores(arch_perm, scores_std, rho=rho)
            null_top[i] = max([t[0] for t in hits], default=0.0)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    p_values = [float(np.mean(null_top >= obs)) for obs in observed_scores]
    n_tests = K_c * (K_c - 1) // 2 * L * (L - 1) // 2 * len(PROPERTIES)
    p_bonf = 0.05 / max(n_tests, 1)

    print(f"\n===== Step 4: permutation null test on arch_assignment "
            f"(mode={mode}, n_perms={n_perms}) =====")
    print(f"n_tests family = {n_tests}, "
            f"Bonferroni threshold p < {p_bonf:.2e}")
    print(f"null top-1 score distribution: "
            f"mean={null_top.mean():.3f}  "
            f"p50={np.median(null_top):.3f}  "
            f"p95={np.quantile(null_top, 0.95):.3f}  "
            f"p99={np.quantile(null_top, 0.99):.3f}  "
            f"max={null_top.max():.3f}")
    print(f"observed top-{top_n} vs null:")
    print(f"{'rank':>4}  {'observed':>10}  {'null p':>10}  significant")
    for i, (obs, p) in enumerate(zip(observed_scores, p_values), 1):
        sig = "***" if p < p_bonf else ("**" if p < 0.01 else
                                            "*" if p < 0.05 else "")
        print(f"{i:>4}  {obs:>+10.3f}  {p:>10.4f}  {sig}")
    return observed_scores, null_top, p_values


def step5_replicate_across_checkpoints(chkpt_a_dir: Path, chkpt_b_dir: Path,
                                            top_n: int = 20):
    """Compare the biophysical *signature* of top compensation flips
    between two checkpoints, agnostic to class labelling.

    Each flip is characterised by (dominant_property, sign_of_delta_c1,
    sign_of_delta_c2). A signature is `charge:+-` etc. -- the class
    identities don't need to align across runs.

    Prints:
      - top-N flip signatures per checkpoint
      - overlap of the top-N signatures
    """
    print("\n===== Step 5: cross-checkpoint signature replication =====")
    def _sig(hits, top_n, scores_std):
        """Return top-N (signature, score) list from a scored arch."""
        return None
    def _load_and_signatures(chkpt_dir, top_n):
        state = load_state(chkpt_dir)
        arch = np.asarray(state['arch'], dtype=np.int64)
        rho = None if state['rho'] is None else np.asarray(state['rho'])
        pi_arch = np.asarray(state['pi_arch'])
        scores = step1_archetype_biophysics(pi_arch)
        scores_std = scores / (scores.std(axis=0, keepdims=True) + 1e-9)
        prop_names = list(PROPERTIES.keys())
        # Get the full hit list and take top_n by score.
        K_c, L = arch.shape
        hits = []
        for theta in range(L):
            for theta_p in range(L):
                if theta_p == theta:
                    continue
                fw = 1.0 if rho is None else float(rho[theta] * rho[theta_p])
                for c in range(K_c):
                    if int(arch[c, theta]) == int(arch[c, theta_p]):
                        continue
                    for cp in range(c + 1, K_c):
                        if int(arch[cp, theta]) == int(arch[cp, theta_p]):
                            continue
                        dc = scores_std[int(arch[c, theta_p])] - scores_std[int(arch[c, theta])]
                        dcp = scores_std[int(arch[cp, theta_p])] - scores_std[int(arch[cp, theta])]
                        per_prop = -dc * dcp
                        total = float(per_prop.sum()) * fw
                        if total <= 0:
                            continue
                        j = int(np.argmax(np.abs(per_prop)))
                        sig = (prop_names[j],
                                 '+' if dc[j] > 0 else '-',
                                 '+' if dcp[j] > 0 else '-')
                        hits.append((total, sig))
        hits.sort(key=lambda x: x[0], reverse=True)
        return hits[:top_n]
    hits_a = _load_and_signatures(chkpt_a_dir, top_n)
    hits_b = _load_and_signatures(chkpt_b_dir, top_n)
    sigs_a = [h[1] for h in hits_a]
    sigs_b = [h[1] for h in hits_b]
    from collections import Counter
    ca, cb = Counter(sigs_a), Counter(sigs_b)
    common = set(ca.keys()) & set(cb.keys())
    print(f"top-{top_n} signatures at {chkpt_a_dir}:")
    for score, sig in hits_a:
        print(f"  {score:>+7.3f}  {sig[0]:>10} sign(dc1)={sig[1]} sign(dc2)={sig[2]}")
    print(f"top-{top_n} signatures at {chkpt_b_dir}:")
    for score, sig in hits_b:
        print(f"  {score:>+7.3f}  {sig[0]:>10} sign(dc1)={sig[1]} sign(dc2)={sig[2]}")
    print(f"\noverlap in top-{top_n} signatures: "
            f"{len(common)} / {len(set(sigs_a))} unique  "
            f"({100 * len(common) / max(len(set(sigs_a)), 1):.0f}%)")
    for sig in common:
        print(f"  {sig[0]:>10} {sig[1]}{sig[2]}  "
                f"appears {ca[sig]}× in A, {cb[sig]}× in B")


def step3_cluster_size_distribution(cluster_id_by_msa: 'list[np.ndarray]',
                                       cls_by_msa: 'list[np.ndarray]' = None,
                                       big_threshold: int = 10,
                                       null_seed: int = 42):
    """Report cluster-size distribution across all MSAs, plus diagnostics
    on the tail of big clusters. A long tail signals pathological
    over-clustering (e.g. one CRP mode eating everything, or the model
    finding a cheap emission-explanation shortcut).

    Prints:
      A. Quantiles + histogram + top-N (descriptive).
      B. Big-cluster attribution (are the size >= big_threshold clusters
         real coevolution or CRP-bribed / emission-shortcut artefacts?).
      C. Null baseline: how many size >= big_threshold clusters would
         appear under a random CRP-matched partition (same n_columns,
         same n_clusters per MSA)?
      D. Actionable recommendation based on the above.
    """
    sizes_all: 'list[int]' = []
    per_msa: 'list[np.ndarray]' = []
    for cid in cluster_id_by_msa:
        cid = np.asarray(cid)
        if cid.size == 0:
            per_msa.append(np.zeros(0, dtype=np.int64))
            continue
        _uniq, cnts = np.unique(cid[cid >= 0], return_counts=True)
        per_msa.append(cnts.astype(np.int64))
        sizes_all.extend(int(c) for c in cnts)
    if not sizes_all:
        print("\n===== Step 3: cluster-size distribution =====")
        print("no clusters found in state.npz")
        return
    sizes = np.asarray(sizes_all, dtype=np.int64)

    print("\n===== Step 3A: cluster-size distribution (descriptive) =====")
    print(f"total clusters: {len(sizes)}, total columns: {int(sizes.sum())}")
    print(f"size stats: mean={sizes.mean():.2f}  sd={sizes.std():.2f}  "
            f"median={int(np.median(sizes))}  "
            f"p75={int(np.quantile(sizes, 0.75))}  "
            f"p90={int(np.quantile(sizes, 0.9))}  "
            f"p99={int(np.quantile(sizes, 0.99))}  max={int(sizes.max())}")
    for cutoff in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        frac_cols = float(sizes[sizes >= cutoff].sum()) / max(1, int(sizes.sum()))
        n_such = int((sizes >= cutoff).sum())
        print(f"  clusters of size >= {cutoff:>3}: {n_such:>5} "
                f"({frac_cols*100:.1f}% of columns)")
    print("\n  size bin       count   pct")
    bins = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257]
    labels = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64",
                "65-128", "129-256", "257+"]
    total = len(sizes)
    for i, lab in enumerate(labels):
        lo = bins[i]
        hi = bins[i + 1] if i + 1 < len(bins) else int(sizes.max()) + 1
        n = int(((sizes >= lo) & (sizes < hi)).sum())
        pct = 100.0 * n / max(1, total)
        bar = "#" * int(pct)
        print(f"  {lab:>10}   {n:>6}   {pct:5.1f}  {bar}")
    top = np.sort(sizes)[::-1][:20]
    print(f"\n  top-20 sizes: {top.tolist()}")

    # ----- 3B: attribution of big clusters -----
    print(f"\n===== Step 3B: big-cluster attribution (size >= {big_threshold}) =====")
    print("Question: are these size >= threshold clusters (which are too big to")
    print("be structural contacts) picking up real coordinated evolution, or")
    print("are they CRP-bribed / emission-shortcut artefacts?")
    print("")
    print("Diagnostics per big cluster:")
    print("  class_entropy_bits: Shannon entropy over site-class labels of")
    print("    the cluster's columns. LOW (near 0) means all columns share")
    print("    one site class -> the merge is emission-shortcut (columns look")
    print("    interchangeable to the model, joining them is nearly free).")
    print("    HIGH (near log2(K_c)) means columns span many classes -> the")
    print("    merge is claiming real coordination across heterogeneous sites.")
    print("  n_columns: cluster size (columns in this cluster).")
    print("  msa_idx: which MSA the cluster is in.")
    print("")
    if cls_by_msa is None or all(c is None for c in cls_by_msa):
        print("(cls arrays not in checkpoint; skipping class-entropy diagnostic)")
    else:
        _report_big_clusters(cluster_id_by_msa, cls_by_msa, big_threshold)

    # ----- 3C: null baseline -----
    print(f"\n===== Step 3C: null baseline (adversarial) =====")
    print("Question: how many size >= threshold clusters would a RANDOM")
    print("partition (same n_columns and n_clusters per MSA, uniform random")
    print("assignment) produce? If comparable, the observed big-cluster mass")
    print("is a CRP prior artefact, not a signal.")
    n_obs = int((sizes >= big_threshold).sum())
    n_null_trials = 20
    rng = np.random.default_rng(null_seed)
    n_null = []
    for _ in range(n_null_trials):
        cnt = 0
        for cnts in per_msa:
            if cnts.size == 0: continue
            n_clusters = cnts.size
            n_cols = int(cnts.sum())
            # Uniform random assignment of columns to clusters (same
            # #columns, same #clusters). Note: some clusters can be empty.
            asg = rng.integers(0, n_clusters, size=n_cols)
            _u, cc = np.unique(asg, return_counts=True)
            cnt += int((cc >= big_threshold).sum())
        n_null.append(cnt)
    n_null = np.asarray(n_null)
    print(f"  observed size >= {big_threshold}: {n_obs}")
    print(f"  null (n={n_null_trials}) mean {n_null.mean():.1f}, "
            f"sd {n_null.std():.1f}, max {n_null.max()}")
    excess = n_obs - n_null.mean()
    z = excess / max(n_null.std(), 1e-9)
    print(f"  excess: {excess:+.1f}  (z ≈ {z:+.1f})")
    if abs(z) < 2.0:
        print("  VERDICT: big-cluster mass indistinguishable from random")
        print("  partition -> most likely CRP prior artefact, not signal.")
    elif z > 5.0:
        print("  VERDICT: big-cluster mass strongly exceeds null -> the")
        print("  model IS extracting coordinated structure beyond what a")
        print("  random partition of the same size would produce.")
    else:
        print("  VERDICT: mild excess over null. Suggestive but not clean.")

    # ----- 3D: recommendation -----
    print(f"\n===== Step 3D: recommendation =====")
    if abs(z) < 2.0 and n_obs > 10:
        print("Big-cluster mass looks like a CRP prior artefact.")
        print("Preferred remedy: tighten the prior on alpha_z, keeping its")
        print("mode fixed. Do NOT shift the mode. Under Gamma(a, b) the mean")
        print("is a/b and sd is sqrt(a)/b; to reduce sd at fixed mean, scale")
        print("both a and b up together. Current default a=100, b=1 gives")
        print("mean=100 with sd=10. Try Gamma(a=1000, b=10): same mean 100,")
        print("sd ~3.2 -- three-fold sharper. If the tail shrinks without")
        print("hurting LL, the previous mass was CRP wander, not signal.")
        print("Only shift the mode (change the ratio a/b) if you have direct")
        print("independent evidence that the data-implied cluster count is")
        print("wrong for the corpus size, which is unusual.")
    elif z > 5.0 and cls_by_msa is not None:
        # Distinguish real coevolution from emission shortcut using class entropy
        print("Big-cluster mass exceeds null. Now check class_entropy in 3B:")
        print("  - Low mean class_entropy -> emission shortcut (all columns")
        print("    in a big cluster share one class; the model isn't gaining")
        print("    coordination, just packaging homogeneous columns).")
        print("  - High mean class_entropy -> plausible real coordination")
        print("    across biophysically heterogeneous sites.")
    else:
        print("No pathological big-cluster tail detected under the null.")
        print("No adjustment recommended on cluster-size grounds alone.")


def _report_big_clusters(cluster_id_by_msa, cls_by_msa, threshold, top_show=20):
    """Print per-big-cluster class entropy + size, sorted by size desc."""
    rows = []
    for m, (cid, cls) in enumerate(zip(cluster_id_by_msa, cls_by_msa)):
        if cid is None or cls is None or cid.size == 0: continue
        cid = np.asarray(cid); cls = np.asarray(cls)
        # Skip padding cluster_id = -1 if any
        valid = cid >= 0
        uniq, inv = np.unique(cid[valid], return_inverse=True)
        for u_idx, u in enumerate(uniq):
            members = np.where(cid == u)[0]
            if members.size < threshold: continue
            mem_cls = cls[members]
            uc, ccnt = np.unique(mem_cls, return_counts=True)
            p = ccnt / ccnt.sum()
            ent = -float((p * np.log2(p + 1e-30)).sum())
            rows.append((int(members.size), ent, m,
                          {int(u): int(c) for u, c in zip(uc, ccnt)}))
    rows.sort(key=lambda r: -r[0])
    print(f"  found {len(rows)} clusters of size >= {threshold}")
    if not rows:
        return
    ents = np.asarray([r[1] for r in rows])
    print(f"  class_entropy_bits summary: "
            f"mean={ents.mean():.2f}  sd={ents.std():.2f}  "
            f"p50={float(np.median(ents)):.2f}  "
            f"p90={float(np.quantile(ents, 0.9)):.2f}  "
            f"max={ents.max():.2f}")
    print(f"\n  top {min(top_show, len(rows))} by size (largest first):")
    print(f"  {'n_cols':>7} {'entropy_bits':>12} {'msa':>4}  class_hist")
    for (sz, ent, m, hist) in rows[:top_show]:
        hist_str = " ".join(f"c{c}:{n}" for c, n in sorted(hist.items()))
        print(f"  {sz:>7} {ent:>12.2f} {m:>4}  {hist_str}")


def step6_composite_likelihood_artifact(
        pi_arch: np.ndarray,
        arch_assignment: np.ndarray,
        rho: np.ndarray,
        rho_chain: float,
        cluster_id_by_msa: 'list[np.ndarray]',
        cls_by_msa: 'list[np.ndarray]',
        per_family_data: 'list[dict]',
        min_cluster_size: int = 2,
        big_threshold: int = 10):
    """Test whether clusters are exploiting composite-likelihood freedom.

    Motivation: under composite likelihood (independent pairwise cherries),
    a cluster does not need any within-cherry field jumps to fit the data.
    Each cherry q can independently pick its own theta_X value at the
    branch root, and the model marginalises via rho[theta] * pi_arch. This
    means a cluster of columns whose observations shift together across
    cherries (but stay consistent within each cherry) can be modelled by
    a static-per-cherry field state that varies across cherries -- with
    rho_chain -> 0 and no within-cherry dynamics.

    The signature of this artifact is:
      cross_cherry_theta_entropy HIGH (multiple theta_X used across cherries)
      mean_within_cherry_jump_prob LOW (rho_chain * t << 1 for all cherries)

    Whereas real within-cherry co-variation would require:
      cross_cherry_theta_entropy HIGH
      mean_within_cherry_jump_prob COMPARABLE TO 1
      (or at least non-trivial)

    Per cluster (size >= min_cluster_size), per cherry q that observes
    at least one cluster column (both-sequences aligned): compute the
    posterior over theta_X under the rho_chain=0 approximation
    (theta_Y = theta_X, so X and Y observations both condition on the
    same field state):

      log p(theta_X | X_q, Y_q, cluster) proportional_to
        log rho[theta_X]
        + sum_{n in obs cluster positions in cherry q}
              [log pi_arch[arch[cls[n], theta_X], X_n_q]
                + log pi_arch[arch[cls[n], theta_X], Y_n_q]]

    Take argmax over theta_X per cherry, then Shannon entropy across
    cherries. Within-cherry jump prob = 1 - exp(-rho_chain * t_q).
    """
    K_c, L_max = arch_assignment.shape
    K_a, A = pi_arch.shape
    if rho is None:
        print("\n===== Step 6: composite-likelihood artifact test =====")
        print("(no rho vector in checkpoint; cannot compute theta_X posterior)")
        return
    rho = np.asarray(rho, dtype=np.float64)

    # Precompute log_pi_by_ct[cls, theta, aa]:
    log_pi = np.log(np.maximum(pi_arch, 1e-300))              # (K_a, A)
    log_pi_by_ct = log_pi[arch_assignment]                     # (K_c, L, A)
    log_rho = np.log(np.maximum(rho, 1e-300))                  # (L,)

    per_cluster: 'list[dict]' = []
    for msa_idx, (cid, cls) in enumerate(zip(cluster_id_by_msa, cls_by_msa)):
        if cid is None or cls is None or msa_idx >= len(per_family_data):
            continue
        fd = per_family_data[msa_idx]
        aa_a = np.asarray(fd['aa_a'])                          # (Q, L_fam)
        aa_b = np.asarray(fd['aa_b'])                          # (Q, L_fam)
        both_aa = np.asarray(fd['both_aa'])                    # (Q, L_fam) bool
        tau = np.asarray(fd['tau'], dtype=np.float64)          # (Q,)
        cid = np.asarray(cid); cls = np.asarray(cls)
        uniq = np.unique(cid[cid >= 0])
        for cluster_id in uniq:
            members = np.where(cid == int(cluster_id))[0]      # (m,) column idx
            if members.size < int(min_cluster_size):
                continue

            # For each cherry q, compute posterior theta and jump prob.
            theta_argmax: 'list[int]' = []
            jump_probs: 'list[float]' = []
            for q in range(aa_a.shape[0]):
                obs_mask = both_aa[q, members]                  # (m,)
                if not np.any(obs_mask):
                    continue
                obs_pos = members[obs_mask]                     # column indices
                obs_cls = cls[obs_pos]                          # (n_obs,)
                X_q = aa_a[q, obs_pos].astype(np.int64)         # (n_obs,)
                Y_q = aa_b[q, obs_pos].astype(np.int64)

                # Vectorised: log_pi_by_ct[obs_cls, :, X_q] has shape
                # (n_obs, L) via fancy indexing on axes 0 and 2.
                lp_X = log_pi_by_ct[obs_cls, :, X_q].sum(axis=0)  # (L,)
                lp_Y = log_pi_by_ct[obs_cls, :, Y_q].sum(axis=0)  # (L,)
                log_p_theta = log_rho + lp_X + lp_Y
                theta_argmax.append(int(np.argmax(log_p_theta)))
                jump_probs.append(1.0 - float(np.exp(-rho_chain * tau[q])))

            if not theta_argmax:
                continue
            theta_arr = np.asarray(theta_argmax)
            _u, cnt = np.unique(theta_arr, return_counts=True)
            p = cnt / cnt.sum()
            cross_ent = -float((p * np.log2(p + 1e-30)).sum())
            per_cluster.append({
                'msa_idx': msa_idx,
                'cluster_id': int(cluster_id),
                'size': int(members.size),
                'n_cherries': int(len(theta_argmax)),
                'cross_theta_entropy_bits': cross_ent,
                'mean_within_cherry_jump_prob': float(np.mean(jump_probs)),
                'n_theta_values_used': int(_u.size),
            })

    print("\n===== Step 6: composite-likelihood artifact test =====")
    print(f"rho_chain = {rho_chain:.4f}")
    print(f"analysed {len(per_cluster)} clusters of size >= {min_cluster_size}")
    if not per_cluster:
        print("no clusters to analyse")
        return

    sizes = np.asarray([r['size'] for r in per_cluster])
    cross_ent_arr = np.asarray([r['cross_theta_entropy_bits'] for r in per_cluster])
    jump_arr = np.asarray([r['mean_within_cherry_jump_prob'] for r in per_cluster])
    n_theta_arr = np.asarray([r['n_theta_values_used'] for r in per_cluster])

    print("\n----- global summary -----")
    print(f"  cross_cherry_theta_entropy (bits): "
            f"mean={cross_ent_arr.mean():.2f}  sd={cross_ent_arr.std():.2f}  "
            f"p50={float(np.median(cross_ent_arr)):.2f}  "
            f"p90={float(np.quantile(cross_ent_arr, 0.9)):.2f}")
    print(f"  mean_within_cherry_jump_prob:      "
            f"mean={jump_arr.mean():.4f}  sd={jump_arr.std():.4f}  "
            f"p50={float(np.median(jump_arr)):.4f}  "
            f"p90={float(np.quantile(jump_arr, 0.9)):.4f}")
    print(f"  n_theta_values_used per cluster:   "
            f"mean={n_theta_arr.mean():.2f}  "
            f"1={int((n_theta_arr==1).sum())}  "
            f"2={int((n_theta_arr==2).sum())}  "
            f"3={int((n_theta_arr==3).sum())}  "
            f">=4={int((n_theta_arr>=4).sum())}")

    # Joint cross-tab: cross-entropy bin x jump-prob bin.
    print("\n----- joint distribution (cluster counts) -----")
    print("  rows: cross_theta_entropy bins (bits)")
    print("  cols: mean_within_cherry_jump_prob bins")
    print("  ARTIFACT SIGNATURE = high-entropy + low-jump quadrant (bottom-right of top-left cell block)")
    print("")
    ent_bins = [0.0, 0.5, 1.0, 1.5, np.inf]
    ent_labels = ["[0, 0.5)", "[0.5, 1.0)", "[1.0, 1.5)", "[1.5+)"]
    jump_bins = [0.0, 0.01, 0.05, 0.20, 1.0 + 1e-9]
    jump_labels = ["<1%", "[1%, 5%)", "[5%, 20%)", "[20%+)"]
    print("  cross_ent \\ jump  |  " + "  ".join(f"{jl:>10}" for jl in jump_labels))
    print("  " + "-" * (18 + 12 * len(jump_labels)))
    for i, el in enumerate(ent_labels):
        e_lo, e_hi = ent_bins[i], ent_bins[i + 1]
        row = []
        for j in range(len(jump_labels)):
            j_lo, j_hi = jump_bins[j], jump_bins[j + 1]
            mask = ((cross_ent_arr >= e_lo) & (cross_ent_arr < e_hi)
                        & (jump_arr >= j_lo) & (jump_arr < j_hi))
            row.append(int(mask.sum()))
        print(f"  {el:>17} |  " + "  ".join(f"{c:>10}" for c in row))

    # Big-cluster split: are the size >= big_threshold clusters
    # concentrated in the artifact quadrant?
    print(f"\n----- clusters of size >= {big_threshold} -----")
    big = np.asarray(sizes) >= int(big_threshold)
    n_big = int(big.sum())
    if n_big > 0:
        artifact_mask = (cross_ent_arr >= 1.0) & (jump_arr < 0.05)
        n_big_artifact = int((big & artifact_mask).sum())
        print(f"  count: {n_big}  "
                f"in artifact quadrant (cross_ent>=1 bit AND jump<5%): "
                f"{n_big_artifact} ({100 * n_big_artifact / max(1, n_big):.1f}%)")
        print(f"  big-cluster mean cross_entropy: "
                f"{float(cross_ent_arr[big].mean()):.2f} bits")
        print(f"  big-cluster mean jump_prob:     "
                f"{float(jump_arr[big].mean()):.4f}")

    # ----- Tau-stratified test: refutes the "short-branch bias" alternative -----
    # Alternative hypothesis (Ian, 2026-07-02): maybe within_cherry_jump_prob
    # is low simply because corpus cherry branch lengths (tau) are all short,
    # and the bigger evolutionary shifts happen between cherries. If true,
    # short-tau clusters should have low jump_prob AND low cross-cherry entropy
    # (data uninformative), while long-tau clusters should show real signal.
    # If the artifact signature persists uniformly across tau bins, the
    # short-branch explanation is refuted.
    all_tau = np.concatenate([np.asarray(fd['tau']) for fd in per_family_data
                                    if fd is not None])
    print(f"\n----- corpus branch-length distribution -----")
    print(f"  cherries total: {all_tau.size}")
    print(f"  tau min={all_tau.min():.3f}  mean={all_tau.mean():.3f}  "
            f"sd={all_tau.std():.3f}  "
            f"p50={float(np.median(all_tau)):.3f}  "
            f"p90={float(np.quantile(all_tau, 0.9)):.3f}  "
            f"p99={float(np.quantile(all_tau, 0.99)):.3f}  "
            f"max={all_tau.max():.3f}")
    print(f"  at inferred rho_chain={rho_chain:.4f}:")
    print(f"    E[jumps] on mean-tau cherry: {rho_chain * all_tau.mean():.4f}")
    print(f"    E[jumps] on p99-tau cherry:  "
            f"{rho_chain * float(np.quantile(all_tau, 0.99)):.4f}")
    print(f"    E[jumps] on max-tau cherry:  "
            f"{rho_chain * all_tau.max():.4f}")
    r_visible = 0.5  # a moderate biological rho_chain value
    frac_visible = float(np.mean(1 - np.exp(-r_visible * all_tau) > 0.1))
    print(f"  IF rho_chain were {r_visible} (moderate biological value):")
    print(f"    fraction of cherries with jump_prob > 10%: {frac_visible*100:.1f}%")
    if frac_visible < 0.3:
        print(f"    → corpus IS short-branch biased; artifact test is INCONCLUSIVE")
        print(f"      about rho_chain because tau distribution can't distinguish")
        print(f"      rho_chain=0 from small rho_chain.")
    else:
        print(f"    → corpus has ENOUGH long-branch cherries that within-cherry")
        print(f"      jumps would be visible at any moderate rho_chain. The")
        print(f"      inferred rho_chain -> 0 is NOT a short-branch artifact.")

    # Recompute per-cluster mean tau at cluster (weighted by cherries with obs).
    per_cluster_mean_tau = np.zeros(len(per_cluster))
    ci = 0
    for msa_idx, (cid, cls) in enumerate(zip(cluster_id_by_msa, cls_by_msa)):
        if cid is None or cls is None or msa_idx >= len(per_family_data): continue
        fd = per_family_data[msa_idx]
        if fd is None: continue
        both_aa = np.asarray(fd['both_aa'])
        tau_f = np.asarray(fd['tau'])
        cid = np.asarray(cid)
        uniq = np.unique(cid[cid >= 0])
        for cluster_id in uniq:
            members = np.where(cid == int(cluster_id))[0]
            if members.size < min_cluster_size: continue
            obs_mask_per_cherry = np.any(both_aa[:, members], axis=1)
            if not obs_mask_per_cherry.any(): continue
            per_cluster_mean_tau[ci] = float(tau_f[obs_mask_per_cherry].mean())
            ci += 1
    per_cluster_mean_tau = per_cluster_mean_tau[:ci]

    print(f"\n----- artifact signature stratified by mean cluster tau -----")
    tau_bins = [0.0, 0.3, 0.7, 1.5, np.inf]
    tau_labels = ["tau < 0.3", "0.3-0.7", "0.7-1.5", "1.5+"]
    print(f"  bin              n  cross_ent  jump_prob  artifact%")
    print(f"  " + "-" * 55)
    for i, tl in enumerate(tau_labels):
        t_lo, t_hi = tau_bins[i], tau_bins[i + 1]
        m = (per_cluster_mean_tau >= t_lo) & (per_cluster_mean_tau < t_hi)
        n = int(m.sum())
        if n == 0:
            print(f"  {tl:>15}  {n:>4}   (empty)"); continue
        mean_ent = float(cross_ent_arr[m].mean())
        mean_jp = float(jump_arr[m].mean())
        frac_art = float(((cross_ent_arr[m] >= 1.0) & (jump_arr[m] < 0.05)).mean())
        print(f"  {tl:>15}  {n:>4}     {mean_ent:.2f}     {mean_jp:.4f}     "
                f"{100*frac_art:.1f}%")
    print(f"  If artifact fraction is high AND uniform across tau bins,")
    print(f"  short-branch bias is REFUTED as the explanation. If artifact")
    print(f"  fraction decreases with increasing tau, short-branch bias is")
    print(f"  the more likely explanation.")

    print("\n----- interpretation -----")
    frac_low_jump = float((jump_arr < 0.05).mean())
    frac_high_ent = float((cross_ent_arr >= 1.0).mean())
    frac_artifact = float(((cross_ent_arr >= 1.0) & (jump_arr < 0.05)).mean())
    print(f"  fraction of clusters with mean_jump_prob < 5%:  {100*frac_low_jump:.1f}%")
    print(f"  fraction with cross_theta_entropy >= 1 bit:     {100*frac_high_ent:.1f}%")
    print(f"  fraction with BOTH (composite-lik artifact):    {100*frac_artifact:.1f}%")
    if frac_artifact > 0.30 and frac_low_jump > 0.80:
        print("  VERDICT: Strong signature of composite-likelihood exploitation.")
        print("  Many clusters have high cross-cherry theta_X entropy but ~zero")
        print("  within-cherry jump probability. Clustering is being driven by")
        print("  cross-cherry aa distribution shifts that pair-wise composite")
        print("  likelihood can 'explain' via per-cherry theta_X selection,")
        print("  NOT by any actual within-cherry field dynamics. The rho_chain")
        print("  crash to ~0 is consistent with this: the data has no evidence")
        print("  for within-cherry field jumps, so the model rationally sets")
        print("  the field-flip rate to zero.")
    elif frac_artifact > 0.10:
        print("  VERDICT: Suggestive of composite-likelihood exploitation but")
        print("  not clean. Some clusters have the signature; others use only")
        print("  one theta value across cherries (i.e. cross_ent=0), consistent")
        print("  with 'real' single-arch clusters.")
    else:
        print("  VERDICT: No strong signature of composite-likelihood exploitation.")
        print("  Cross-cherry entropy is low; clusters mostly use one theta_X.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint_dir", type=Path,
                    help="Path to a checkpoint directory with state.npz + "
                          "meta.json (e.g. results/.../_best_chkpt)")
    p.add_argument("--top-n", type=int, default=20,
                    help="Number of top compensatory flips to print.")
    p.add_argument("--n-perms", type=int, default=1000,
                    help="Number of permutation-null shuffles for Step 4.")
    p.add_argument("--null-mode", type=str, default="row",
                    choices=("row", "iid", "both"),
                    help="Null model for Step 4. 'row' preserves each "
                          "class's multiset of archetypes; 'iid' draws "
                          "each (c, θ) from Categorical(rho_arch); 'both' "
                          "runs both.")
    p.add_argument("--compare-to", type=Path, default=None,
                    help="Second checkpoint directory to compare against "
                          "(cross-checkpoint biophysical signature test).")
    p.add_argument("--corpus-dir", type=Path, default=None,
                    help="Processed corpus directory (e.g. "
                          "data/pfam_processed_top1000). Required for "
                          "Step 6 (composite-likelihood artifact test) "
                          "which needs the cherry observations to compute "
                          "per-cherry posterior theta_X.")
    args = p.parse_args()

    state = load_state(args.checkpoint_dir)
    pi_arch = np.asarray(state['pi_arch'])
    arch = np.asarray(state['arch'], dtype=np.int64)
    rho = None if state['rho'] is None else np.asarray(state['rho'])
    rho_arch = None if state['rho_arch'] is None else np.asarray(state['rho_arch'])

    K_c, L = arch.shape
    K_a, A = pi_arch.shape
    print(f"Loaded checkpoint: K_c={K_c}, L={L}, K_a={K_a}, A={A}")
    if rho is not None:
        print(f"rho = {rho}")
    if rho_arch is not None:
        print(f"rho_arch = {rho_arch}")

    scores = step1_archetype_biophysics(pi_arch)
    print_step1(scores, pi_arch, rho_arch=rho_arch)

    # Show arch_assignment matrix for quick inspection.
    print("\n===== arch_assignment[c, theta] =====")
    print("theta:  " + "  ".join(f"{t:>2}" for t in range(L)))
    for c in range(K_c):
        row = f"c={c}:   " + "  ".join(f"{int(arch[c, t]):>2}" for t in range(L))
        row += "    " + ("specializer" if len(set(arch[c].tolist())) == 1
                            else "mediator")
        print(row)

    step2_coordinated_flips(arch, scores, rho=rho, top_n=args.top_n)

    modes = ("row", "iid") if args.null_mode == "both" else (args.null_mode,)
    for mode in modes:
        step4_permutation_test(arch, scores, rho=rho, rho_arch=rho_arch,
                                  n_perms=args.n_perms,
                                  top_n=min(args.top_n, 10), mode=mode)

    if args.compare_to is not None:
        step5_replicate_across_checkpoints(
            args.checkpoint_dir, args.compare_to, top_n=args.top_n)

    step3_cluster_size_distribution(state['cluster_id_by_msa'],
                                        state['cls_by_msa'])

    if args.corpus_dir is not None:
        fam_names = state['meta'].get('family_names')
        if fam_names is None:
            print("\n===== Step 6 skipped =====")
            print("(no family_names in checkpoint meta — corpus alignment "
                  "unknown)")
        else:
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                                            / "src"))
                from tkfdp.pfam_data_fast import families_from_processed
            except ImportError as e:
                print(f"\n===== Step 6 skipped (import failed) =====\n{e}")
            else:
                fam_cs = families_from_processed(
                    args.corpus_dir, n_families=None, min_cherries=2)
                by_name = {fc.family: fc for fc in fam_cs}
                per_family_data = []
                for name in fam_names:
                    if name not in by_name:
                        per_family_data.append(None)
                        continue
                    fc = by_name[name]
                    per_family_data.append(dict(
                        family=fc.family, L=fc.L, n_cherries=fc.n_cherries,
                        tau=np.asarray(fc.tau, dtype=np.float64),
                        aa_a=np.asarray(fc.aa_a),
                        aa_b=np.asarray(fc.aa_b),
                        both_aa=fc.both_aa_mask()))
                step6_composite_likelihood_artifact(
                    pi_arch, arch, rho, float(state['rho_chain']),
                    state['cluster_id_by_msa'], state['cls_by_msa'],
                    per_family_data)


if __name__ == "__main__":
    main()
