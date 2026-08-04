"""Smoke + correctness tests for gibbs_sweep_cluster (Phase D.5 step 2).

What we verify:

  1. cluster_id <-> partner round-trip via the helper functions.
  2. canonical_cluster_ids re-numbers contiguously from 0.
  3. On a synthetic corpus simulated under a known cluster structure
     and a dynfield model, the CRP sweep concentrates on partitions
     that group the genuinely-coupled columns together.

Run: python3 tests/dynfield/test_cluster_gibbs.py
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

from tkfdp.partition_K import (FamilyKState, gibbs_sweep_cluster,
                                  cluster_id_from_partner, partner_from_cluster_id,
                                  canonical_cluster_ids, clusters_from_cluster_id)
from tkfdp.coupling.dynfield.state import DynamicFieldState
from tkfdp.coupling.dynfield import DynamicFieldCouplingModel
from tkfdp.coupling.dynfield import emission as _em
from tkfdp.lg08 import S_LG08_F81_J as S_LG08_J


# ---------------------------------------------------------------------------
# Test 1: cluster_id <-> partner helpers.
# ---------------------------------------------------------------------------

def test_cluster_id_partner_roundtrip():
    print("\n[test 1] cluster_id <-> partner round-trip")
    partner = np.array([1, 0, -1, 4, 3, -1, 7, 6], dtype=np.int32)
    cid = cluster_id_from_partner(partner)
    print(f"  partner    = {partner.tolist()}")
    print(f"  cluster_id = {cid.tolist()}")
    # Round-trip back.
    partner_back = partner_from_cluster_id(cid)
    assert np.array_equal(partner_back, partner), (
        f"round-trip failed: {partner_back} != {partner}")
    # Number of distinct cluster ids should equal n_pairs + n_singletons.
    n_pairs = int((partner >= 0).sum() // 2)
    n_singletons = int((partner == -1).sum())
    assert len(set(cid.tolist())) == n_pairs + n_singletons
    print("  PASS")


def test_canonical_cluster_ids():
    print("\n[test 2] canonical_cluster_ids renumbers contiguously")
    cid = np.array([3, 0, 3, 7, 0, 7, 7], dtype=np.int32)
    can = canonical_cluster_ids(cid)
    print(f"  in  = {cid.tolist()}")
    print(f"  out = {can.tolist()}")
    # First-appearance order: 3 -> 0, 0 -> 1, 7 -> 2.
    expected = np.array([0, 1, 0, 2, 1, 2, 2])
    assert np.array_equal(can, expected)
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: CRP sweep concentrates on known cluster structure.
# ---------------------------------------------------------------------------

def _make_truth_model(K_c=1, L_max=3, A=20, seed=0):
    """Single-class model so cls assignments don't dominate; the dynfield
    structure carries all the signal."""
    rng = np.random.default_rng(seed)
    # Make the per-(c, theta) stationaries very distinct (different
    # peaky distributions per theta) so a coupled cluster shows clear
    # field-shared structure.
    pi_field = np.zeros((K_c, L_max, A))
    for c in range(K_c):
        for th in range(L_max):
            base = np.ones(A) * 0.01
            # Make theta-th block of residues peak.
            block_start = (th * 5) % A
            for k in range(5):
                base[(block_start + k) % A] = 4.0
            pi_field[c, th] = base / base.sum()
    rho = np.full(L_max, 1.0 / L_max)
    pi_class = np.einsum('t,cta->ca', rho, pi_field)
    state = DynamicFieldState(
        K_c=K_c, A=A,
        pi_field=pi_field, rho=rho, tsb_betas=None, alpha_field=1.0,
        pi_class=pi_class, rho_chain=0.05,    # slow field; strong coupling
    )
    return DynamicFieldCouplingModel(K_c=K_c, A=A, pi_class=pi_class,
                                       dyn_field=state)


def _simulate_cherries(truth, *, L, cluster_structure, n_cherries, t, seed):
    """Simulate a per-family corpus of cherries with a known cluster
    structure.

    cluster_structure: list of lists of column indices; columns in the
    same list are coupled (share theta_P trajectory). All columns are
    of the single class 0.
    """
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
            # Parent residues per site (independent given theta_P).
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
    return aa_a, aa_b, both_aa, tau


def _make_cluster_loglik_fn(model, aa_a, aa_b, both_aa, tau, cls):
    """Returns a function that scores a subset of columns under the
    dynfield model summed across cherries (where all sites are
    observed)."""
    Q_cf = _em.per_class_field_Q(model.dyn_field.pi_field)
    # Precompute Sigma per unique tau.
    unique_t = np.unique(tau)
    Sigma_per_t = {}
    for t_val in unique_t:
        P_half = _em.per_class_field_P_half(Q_cf, t=float(t_val))
        Sigma_per_t[float(t_val)] = _em.per_class_field_cherry_sigma(
            P_half, model.dyn_field.pi_field)

    def fn(columns):
        cols = np.asarray(columns, dtype=np.int64)
        if len(cols) == 0:
            return 0.0
        classes = cls[cols]
        ll = 0.0
        for q in range(aa_a.shape[0]):
            if not both_aa[q, cols].all():
                continue
            X_obs = aa_a[q, cols]
            Y_obs = aa_b[q, cols]
            t_q = float(tau[q])
            P_per_theta_case, _ = _em.cluster_emission_per_theta(
                t=t_q, rho=model.dyn_field.rho,
                pi_field=model.dyn_field.pi_field,
                classes=classes, X_obs=X_obs, Y_obs=Y_obs,
                rho_chain=float(model.dyn_field.rho_chain),
                precomputed_Sigma=Sigma_per_t[t_q])
            total = float(P_per_theta_case.sum())
            ll += np.log(max(total, 1e-300))
        return ll
    return fn


def test_cluster_sweep_recovers_known_structure():
    print("\n[test 3] CRP sweep concentrates mass on known cluster "
          "structure")
    truth = _make_truth_model(K_c=1, L_max=3, A=20, seed=43)
    L = 8
    # Truth: columns (0, 1, 2) coupled; (3, 4) coupled; (5, 6, 7) singletons.
    cluster_truth = [[0, 1, 2], [3, 4], [5], [6], [7]]
    n_cherries = 60
    aa_a, aa_b, both_aa, tau = _simulate_cherries(
        truth, L=L, cluster_structure=cluster_truth,
        n_cherries=n_cherries, t=0.5, seed=44)
    cls = np.zeros(L, dtype=np.int32)
    # Init from random partition (independent columns).
    state = FamilyKState(
        family='test', L=L, K=1,
        partner=-np.ones(L, dtype=np.int32),
        cls=cls,
    )
    state.cluster_id = np.arange(L, dtype=np.int32)   # all singletons
    fn = _make_cluster_loglik_fn(truth, aa_a, aa_b, both_aa, tau, cls)
    # Run several sweeps.
    rng = np.random.default_rng(45)
    for it in range(8):
        state = gibbs_sweep_cluster(
            state, fn, rng, alpha_z=0.5, max_cluster_size=8)
        clust = clusters_from_cluster_id(state.cluster_id)
        sizes = sorted(len(v) for v in clust.values())
        print(f"  iter {it + 1}: cluster sizes = {sizes}, "
              f"n_clusters = {len(clust)}")
    final_clusters = clusters_from_cluster_id(state.cluster_id)
    # Check that {0, 1, 2} are in the same cluster, and {3, 4} are in
    # the same cluster.
    cid_of = lambda s: int(state.cluster_id[s])
    print(f"  final cluster ids: {state.cluster_id.tolist()}")
    assert cid_of(0) == cid_of(1) == cid_of(2), (
        f"columns 0, 1, 2 not in the same cluster: "
        f"cids {cid_of(0), cid_of(1), cid_of(2)}")
    assert cid_of(3) == cid_of(4), (
        f"columns 3, 4 not in the same cluster: cids {cid_of(3), cid_of(4)}")
    # Singletons 5, 6, 7 should not be in any of the coupled clusters.
    assert cid_of(5) != cid_of(0) and cid_of(5) != cid_of(3), (
        "column 5 inadvertently joined a coupled cluster")
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
    print(f"\nAll {len(fns)} cluster Gibbs tests passed.")
