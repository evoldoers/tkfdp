"""Phase D.5: variable-size cluster emission and soft-EM attribution.

Verifies:

  1. At m=2 the new attribute_cluster_soft agrees with the existing
     attribute_cherry_doublet_soft to machine precision.
  2. At m=1 the new attribute_cluster_soft agrees with the existing
     attribute_cherry_singlet_soft.
  3. At m=3 cluster training improves cumulative log LL on synthetic
     data (smoke run; no truth comparison since simulating size-3
     dynfield clusters from a known model is itself a fixture we'd
     have to write).
  4. cluster_emission_per_theta marginalised over (theta_P, case)
     equals the cluster joint probability that a size-m direct evaluator
     would compute -- i.e., the per-theta_P decomposition is consistent.

Run: python3 tests/dynfield/test_cluster_variable_size.py
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

from tkfdp.coupling import get as get_variant
from tkfdp.coupling.dynfield.state import DynamicFieldState
from tkfdp.coupling.dynfield import emission as _em
from tkfdp.coupling.dynfield import updates as _up
from tkfdp.lg08 import S_LG08_F81_J as S_LG08_J


def make_model(K_c=2, L_max=3, A=20, seed=0):
    rng = np.random.default_rng(seed)
    pi_field = rng.dirichlet(np.ones(A) * 0.5, size=(K_c, L_max))
    rho = rng.dirichlet(np.ones(L_max) * 1.5)
    pi_class = np.einsum('t,cta->ca', rho, pi_field)
    state = DynamicFieldState(
        K_c=K_c, A=A,
        pi_field=pi_field, rho=rho, tsb_betas=None, alpha_field=1.0,
        pi_class=pi_class, rho_chain=1.0,
    )
    Model = get_variant('dynamic_field')
    return Model(K_c=K_c, A=A, pi_class=pi_class, dyn_field=state)


# ---------------------------------------------------------------------------
# Test 1: m=2 generic == size-2 specialisation.
# ---------------------------------------------------------------------------

def test_m2_matches_size2_specialisation():
    print("\n[test 1] attribute_cluster_soft (m=2) == "
          "attribute_cherry_doublet_soft")
    model = make_model(K_c=2, L_max=3, seed=11)
    caches = _up.precompute_per_cluster_caches(
        t=0.4, rho=model.dyn_field.rho, pi_field=model.dyn_field.pi_field,
        rho_chain=float(model.dyn_field.rho_chain))
    c_i, c_j = 0, 1
    Xi, Xj, Yi, Yj = 3, 7, 11, 15
    N2, r2, ll2 = _up.attribute_cherry_doublet_soft(
        caches, c_i, c_j, Xi, Xj, Yi, Yj)
    classes = np.array([c_i, c_j])
    X_obs = np.array([Xi, Xj])
    Y_obs = np.array([Yi, Yj])
    Nm, rm, llm = _up.attribute_cluster_soft(caches, classes, X_obs, Y_obs)
    err_N = float(np.max(np.abs(Nm - N2)))
    err_r = float(np.max(np.abs(rm - r2)))
    err_ll = abs(llm - ll2)
    print(f"  max |N_diff| = {err_N:.3e}")
    print(f"  max |r_diff| = {err_r:.3e}")
    print(f"  log_lik diff = {err_ll:.3e}")
    assert err_N < 1e-12 and err_r < 1e-12 and err_ll < 1e-10
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2: m=1 generic == singlet specialisation.
# ---------------------------------------------------------------------------

def test_m1_matches_singlet_specialisation():
    print("\n[test 2] attribute_cluster_soft (m=1) == "
          "attribute_cherry_singlet_soft")
    model = make_model(K_c=3, L_max=2, seed=12)
    caches = _up.precompute_per_cluster_caches(
        t=0.6, rho=model.dyn_field.rho, pi_field=model.dyn_field.pi_field,
        rho_chain=float(model.dyn_field.rho_chain))
    c = 1
    Xi, Yi = 4, 9
    N1, r1, ll1 = _up.attribute_cherry_singlet_soft(caches, c, Xi, Yi)
    classes = np.array([c])
    X_obs = np.array([Xi])
    Y_obs = np.array([Yi])
    Nm, rm, llm = _up.attribute_cluster_soft(caches, classes, X_obs, Y_obs)
    err_N = float(np.max(np.abs(Nm - N1)))
    err_r = float(np.max(np.abs(rm - r1)))
    err_ll = abs(llm - ll1)
    print(f"  max |N_diff| = {err_N:.3e}")
    print(f"  max |r_diff| = {err_r:.3e}")
    print(f"  log_lik diff = {err_ll:.3e}")
    assert err_N < 1e-12 and err_r < 1e-12 and err_ll < 1e-10
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: m=3 cluster training improves LL on synthetic data.
# ---------------------------------------------------------------------------

def _simulate_cluster_corpus_m3(truth, *, t, n_clusters, seed):
    """Simulate clusters of size 3 from the truth model via Gillespie
    sampling on each cherry half-edge."""
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
        theta_now = int(theta_0); el = 0.0; n_j = 0
        while True:
            rate = rho_chain * (1.0 - rho[theta_now])
            if rate <= 0: break
            dt = rng.exponential(1.0 / rate)
            if el + dt > s: break
            el += dt
            p = rho.copy(); p[theta_now] = 0
            d = p.sum()
            if d <= 0: break
            p /= d
            theta_now = int(rng.choice(L_max, p=p)); n_j += 1
        return theta_now, n_j

    clusters = []
    for _ in range(n_clusters):
        cs = rng.choice(K_c, size=3)
        th_P = int(rng.choice(L_max, p=rho))
        # Per-site parent residues correlated only via theta_P.
        ps = [int(rng.choice(A, p=pi_field[int(cs[i]), th_P])) for i in range(3)]
        # X edge.
        th_X, nj_X = _gillespie(th_P, t / 2.0)
        if nj_X == 0:
            X = [int(rng.choice(A, p=P_half[int(cs[i]), th_P, ps[i]]))
                  for i in range(3)]
        else:
            X = [int(rng.choice(A, p=pi_field[int(cs[i]), th_X])) for i in range(3)]
        # Y edge.
        th_Y, nj_Y = _gillespie(th_P, t / 2.0)
        if nj_Y == 0:
            Y = [int(rng.choice(A, p=P_half[int(cs[i]), th_P, ps[i]]))
                  for i in range(3)]
        else:
            Y = [int(rng.choice(A, p=pi_field[int(cs[i]), th_Y])) for i in range(3)]
        clusters.append((np.array(cs, dtype=np.int64),
                          np.array(X, dtype=np.int64),
                          np.array(Y, dtype=np.int64)))
    return clusters


# test_m3_cluster_training_improves removed: exercised the flat
# pi_field Dirichlet M-step (update_pi_field_dirichlet) that no longer
# exists after the archetype-only refactor. Archetype-path training is
# covered by the end-to-end smoke tests in experiments/train_dynfield.py.


# ---------------------------------------------------------------------------
# Test 4: cluster_emission_per_theta sum equals direct cluster joint eval.
# ---------------------------------------------------------------------------

def _direct_cluster_joint(t, rho, pi_field, classes, X_obs, Y_obs,
                            S_arr, rho_chain=1.0):
    """Brute-force compute P(X_obs, Y_obs | classes, t) by summing the
    4-case formula at each theta_P. Used as a reference."""
    K_c, L_max, A = pi_field.shape
    m = len(classes)
    s = t / 2.0
    alpha = float(np.exp(-rho_chain * s))
    beta = np.exp(-rho_chain * (1.0 - rho) * s)
    denom = 1.0 - beta
    safe = denom > 1e-15
    w_J = np.where(safe, (1.0 - alpha) / np.where(safe, denom, 1.0), 1.0)
    w_pi = np.where(safe, (alpha - beta) / np.where(safe, denom, 1.0), 0.0)
    S_off = S_arr - np.diag(np.diag(S_arr))
    P_half = np.zeros((K_c, L_max, A, A))
    for c in range(K_c):
        for th in range(L_max):
            Q = S_off * pi_field[c, th][None, :]
            np.fill_diagonal(Q, -Q.sum(axis=1))
            P_half[c, th] = expm(Q * s)
    Sigma = np.einsum('ctp,ctpa,ctpb->ctab', pi_field, P_half, P_half)
    total = 0.0
    for th_P in range(L_max):
        b = beta[th_P]; one_m_b = 1.0 - b
        sigma_prod = np.prod([Sigma[classes[i], th_P, X_obs[i], Y_obs[i]]
                                for i in range(m)])
        pi_prod_X_theta_P = np.prod([pi_field[classes[i], th_P, X_obs[i]]
                                       for i in range(m)])
        pi_prod_Y_theta_P = np.prod([pi_field[classes[i], th_P, Y_obs[i]]
                                       for i in range(m)])
        # J^(classes)(X) = sum_theta rho * prod_i pi[c, theta, X_i]
        J_X = sum(rho[th] * np.prod(
            [pi_field[classes[i], th, X_obs[i]] for i in range(m)])
            for th in range(L_max))
        J_Y = sum(rho[th] * np.prod(
            [pi_field[classes[i], th, Y_obs[i]] for i in range(m)])
            for th in range(L_max))
        J_prime_X = w_J[th_P] * J_X + w_pi[th_P] * pi_prod_X_theta_P
        J_prime_Y = w_J[th_P] * J_Y + w_pi[th_P] * pi_prod_Y_theta_P
        term = (b * b * sigma_prod
                + b * one_m_b * pi_prod_X_theta_P * J_prime_Y
                + one_m_b * b * J_prime_X * pi_prod_Y_theta_P
                + one_m_b ** 2 * J_prime_X * J_prime_Y)
        total += rho[th_P] * term
    return total


def test_cluster_emission_per_theta_marginalises_to_joint():
    print("\n[test 4] cluster_emission_per_theta sum matches direct eval (m=3)")
    model = make_model(K_c=3, L_max=4, seed=44)
    classes = np.array([0, 1, 2])
    X_obs = np.array([3, 5, 7])
    Y_obs = np.array([11, 13, 17])
    t = 0.5
    S_arr = np.asarray(S_LG08_J, dtype=np.float64)
    P_per_theta_case, info = _em.cluster_emission_per_theta(
        t=t, rho=model.dyn_field.rho,
        pi_field=model.dyn_field.pi_field,
        classes=classes, X_obs=X_obs, Y_obs=Y_obs,
        rho_chain=float(model.dyn_field.rho_chain))
    sum_method = float(P_per_theta_case.sum())
    direct = _direct_cluster_joint(
        t=t, rho=model.dyn_field.rho,
        pi_field=model.dyn_field.pi_field,
        classes=classes, X_obs=X_obs, Y_obs=Y_obs,
        S_arr=S_arr, rho_chain=float(model.dyn_field.rho_chain))
    err = abs(sum_method - direct)
    print(f"  sum(P_per_theta_case) = {sum_method:.6e}")
    print(f"  direct eval            = {direct:.6e}")
    print(f"  abs diff               = {err:.3e}")
    assert err / max(direct, 1e-300) < 1e-12, (
        f"per-theta sum != direct eval: rel err {err / direct:.3e}")
    print("  PASS")


# ---------------------------------------------------------------------------
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
    print(f"\nAll {len(fns)} variable-size cluster tests passed.")
