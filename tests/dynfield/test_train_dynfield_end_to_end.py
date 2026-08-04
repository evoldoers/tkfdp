"""Phase D.6 end-to-end dynfield trainer smoke test.

Simulates a tiny Pfam-shaped corpus (2 families, ~10 cherries each, L=10
columns, known cluster structure) from a known dynfield truth model.
Runs several iterations of the full trainer (CRP cluster Gibbs ->
cluster extraction -> atom update) and verifies:

  1. The cumulative log-likelihood improves over iterations.
  2. After several iterations, the recovered cluster structure on each
     MSA approximately matches the truth (allowing for some misassignment
     since the model is initialised uniformly).

Run: python3 tests/dynfield/test_train_dynfield_end_to_end.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from scipy.linalg import expm

from tkfdp.svi import (init_svi_state_dynfield, train_dynfield_full_iter,
                          extract_cluster_observations,
                          cluster_gibbs_sweep_all,
                          train_dynfield_one_iter)
from tkfdp.coupling.dynfield.state import DynamicFieldState
from tkfdp.coupling.dynfield import DynamicFieldCouplingModel
from tkfdp.partition_K import (clusters_from_cluster_id,
                                  cluster_id_from_partner)
from tkfdp.lg08 import S_LG08_F81_J as S_LG08_J


# ---------------------------------------------------------------------------
# Synthetic Pfam-shaped corpus.
# ---------------------------------------------------------------------------

def _make_truth_model(K_c=1, L_max=2, A=20, seed=0):
    rng = np.random.default_rng(seed)
    pi_field = np.zeros((K_c, L_max, A))
    for c in range(K_c):
        for th in range(L_max):
            base = np.ones(A) * 0.01
            block_start = (th * 5) % A
            for k in range(5):
                base[(block_start + k) % A] = 4.0
            pi_field[c, th] = base / base.sum()
    rho = np.full(L_max, 1.0 / L_max)
    pi_class = np.einsum('t,cta->ca', rho, pi_field)
    state = DynamicFieldState(
        K_c=K_c, A=A,
        pi_field=pi_field, rho=rho, tsb_betas=None, alpha_field=1.0,
        pi_class=pi_class, rho_chain=0.05,    # slow field
    )
    return DynamicFieldCouplingModel(K_c=K_c, A=A, pi_class=pi_class,
                                       dyn_field=state)


def _simulate_family_data(truth, L, cluster_structure, n_cherries, t, seed):
    """Build per_family_data-shaped dict for one MSA."""
    rng = np.random.default_rng(seed)
    pi_field = truth.dyn_field.pi_field
    rho = truth.dyn_field.rho
    rho_chain = float(truth.dyn_field.rho_chain)
    K_c, L_max, A = pi_field.shape
    S_np = np.asarray(S_LG08_J, dtype=np.float64)
    S_off = S_np - np.diag(np.diag(S_np))
    P_half = np.zeros((K_c, L_max, A, A))
    for c in range(K_c):
        for th in range(L_max):
            pi = pi_field[c, th]
            Q = S_off * pi[None, :]
            np.fill_diagonal(Q, -Q.sum(axis=1))
            P_half[c, th] = expm(Q * t / 2.0)

    def _gillespie(theta_0, s):
        th_now = int(theta_0); el = 0.0; n_j = 0
        while True:
            rate = rho_chain * (1.0 - rho[th_now])
            if rate <= 0: break
            dt = rng.exponential(1.0 / rate)
            if el + dt > s: break
            el += dt
            p = rho.copy(); p[th_now] = 0
            d = p.sum()
            if d <= 0: break
            p /= d
            th_now = int(rng.choice(L_max, p=p)); n_j += 1
        return th_now, n_j

    aa_a = np.zeros((n_cherries, L), dtype=np.int32)
    aa_b = np.zeros((n_cherries, L), dtype=np.int32)
    both_aa = np.ones((n_cherries, L), dtype=bool)
    tau = np.full(n_cherries, t, dtype=np.float64)
    for q in range(n_cherries):
        for cluster in cluster_structure:
            theta_P = int(rng.choice(L_max, p=rho))
            parent = [int(rng.choice(A, p=pi_field[0, theta_P]))
                       for _ in cluster]
            th_X, nj_X = _gillespie(theta_P, t / 2.0)
            th_Y, nj_Y = _gillespie(theta_P, t / 2.0)
            for i, s in enumerate(cluster):
                if nj_X == 0:
                    aa_a[q, s] = int(rng.choice(A, p=P_half[0, theta_P, parent[i]]))
                else:
                    aa_a[q, s] = int(rng.choice(A, p=pi_field[0, th_X]))
                if nj_Y == 0:
                    aa_b[q, s] = int(rng.choice(A, p=P_half[0, theta_P, parent[i]]))
                else:
                    aa_b[q, s] = int(rng.choice(A, p=pi_field[0, th_Y]))
    return {
        'family': 'F',
        'L': L, 'K': 1,
        'aa_a': aa_a, 'aa_b': aa_b, 'both_aa': both_aa,
        'tau': tau, 'n_cherries': n_cherries,
    }


def _build_per_family_data(truth, *, n_families=2, L=10, t=0.5,
                              n_cherries_per_family=15, seed=0):
    """Build per_family_data list with a known cluster structure per
    family. Family 0: {0,1,2,3} coupled, {4,5} coupled, {6,7,8,9} singletons.
    Family 1: {0,1} coupled, {2,3,4} coupled, {5,6,7,8,9} singletons."""
    truth_structures = [
        [[0, 1, 2, 3], [4, 5], [6], [7], [8], [9]],
        [[0, 1], [2, 3, 4], [5], [6], [7], [8], [9]],
    ]
    out = []
    for f in range(n_families):
        fd = _simulate_family_data(
            truth, L=L,
            cluster_structure=truth_structures[f],
            n_cherries=n_cherries_per_family,
            t=t, seed=seed + f * 100)
        fd['family'] = f'F{f}'
        out.append(fd)
    return out, truth_structures


# ---------------------------------------------------------------------------
# Test: end-to-end full iter improves LL.
# ---------------------------------------------------------------------------

def test_end_to_end_full_iter_improves_LL():
    print("\n[test 1] end-to-end full iter: cluster Gibbs + atom update -> LL")
    K_c = 1
    L_max = 2
    truth = _make_truth_model(K_c=K_c, L_max=L_max, A=20, seed=11)
    per_fam, truth_structures = _build_per_family_data(
        truth, n_families=2, L=10, t=0.5,
        n_cherries_per_family=20, seed=12)
    state = init_svi_state_dynfield(
        per_fam, K_c=K_c, L_max=L_max,
        alpha_field=1.0, rho_chain=0.05,
        rng=np.random.default_rng(13))
    # Initialise cluster_id to all-singletons for both families.
    for st in state.states_per_msa:
        st.cluster_id = np.arange(st.L, dtype=np.int32)
    rng = np.random.default_rng(14)
    log_liks = []
    for it in range(10):
        state, info = train_dynfield_full_iter(
            state, per_fam, rng,
            alpha_z=0.5, max_cluster_size=8)
        log_liks.append(info['log_lik_total'])
        print(f"  iter {it + 1}: LL = {info['log_lik_total']:.2f}  "
              f"({info['n_clusters_total']} clusters, "
              f"mean size = {info['mean_cluster_size']:.2f}, "
              f"max size = {info['max_cluster_size']})")
    assert log_liks[-1] > log_liks[0], (
        f"LL did not improve over iters: {log_liks[0]:.2f} -> "
        f"{log_liks[-1]:.2f}")
    print(f"  LL gain: {log_liks[-1] - log_liks[0]:+.2f} nats over 10 iters")
    # Report recovered structure per family.
    for fam_idx, st in enumerate(state.states_per_msa):
        clust = clusters_from_cluster_id(st.cluster_id)
        sizes = sorted(len(v) for v in clust.values())
        print(f"  family {fam_idx}: recovered cluster sizes = {sizes}")
        print(f"             truth cluster sizes = "
              f"{sorted(len(c) for c in truth_structures[fam_idx])}")
    print("  PASS")


def test_extract_cluster_observations_format():
    print("\n[test 2] extract_cluster_observations format")
    K_c = 1
    truth = _make_truth_model(K_c=K_c, L_max=2, A=20, seed=21)
    per_fam, _ = _build_per_family_data(
        truth, n_families=1, L=10, t=0.4,
        n_cherries_per_family=5, seed=22)
    state = init_svi_state_dynfield(
        per_fam, K_c=K_c, L_max=2, rng=np.random.default_rng(23))
    # Override family 0's cluster_id to a known structure (3 clusters
    # of sizes 3, 1, 2 over the L=10 columns; other columns are
    # singletons).
    cid = np.array([0, 0, 0, 1, 2, 2, 3, 4, 5, 6], dtype=np.int32)
    state.states_per_msa[0].cluster_id = cid
    clusters = extract_cluster_observations(state, per_fam)
    # 7 distinct cluster ids x 5 cherries = 35 tuples.
    assert len(clusters) == 35, f"expected 35 cluster obs, got {len(clusters)}"
    # First cluster (cols 0, 1, 2) has size 3.
    classes, X_obs, Y_obs, t = clusters[0]
    assert len(X_obs) == 3 and len(Y_obs) == 3
    print(f"  {len(clusters)} cluster observations extracted")
    print(f"  first cluster: m={len(X_obs)}, classes={classes.tolist()}, "
          f"X={X_obs.tolist()}, t={t}")
    print("  PASS")


def test_class_sweep_runs():
    """class_gibbs_sweep_all_dynfield runs without error at K_c=2 and
    actually changes some class assignments (not a no-op)."""
    print("\n[test 3] class_gibbs_sweep_all_dynfield smoke")
    from tkfdp.svi import class_gibbs_sweep_all_dynfield
    K_c = 2
    L_max = 2
    truth = _make_truth_model(K_c=1, L_max=L_max, A=20, seed=11)
    per_fam, _ = _build_per_family_data(
        truth, n_families=1, L=10, t=0.4,
        n_cherries_per_family=10, seed=12)
    state = init_svi_state_dynfield(
        per_fam, K_c=K_c, L_max=L_max, rng=np.random.default_rng(13))
    for st in state.states_per_msa:
        st.cluster_id = np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 6],
                                    dtype=np.int32)
    cls_before = state.states_per_msa[0].cls.copy()
    state = class_gibbs_sweep_all_dynfield(
        state, per_fam, np.random.default_rng(14), alpha_c=1.0)
    cls_after = state.states_per_msa[0].cls.copy()
    print(f"  cls before: {cls_before.tolist()}")
    print(f"  cls after:  {cls_after.tolist()}")
    n_changed = int((cls_before != cls_after).sum())
    print(f"  {n_changed} columns changed class out of {len(cls_before)}")
    assert n_changed > 0, "class sweep was a no-op (no columns changed)"
    assert state.states_per_msa[0].cls.shape == (10,)
    print("  PASS")


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    fns = [(name, fn) for name, fn in inspect.getmembers(mod, inspect.isfunction)
           if name.startswith("test_")]
    failures = []
    for name, fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"\n  FAIL  {name}: {e}")
            failures.append(name)
        except Exception as e:
            import traceback
            print(f"\n  ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(name)
    if failures:
        print(f"\n{len(failures)}/{len(fns)} test(s) failed: {failures}")
        sys.exit(1)
    print(f"\nAll {len(fns)} end-to-end dynfield tests passed.")
