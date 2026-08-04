"""Phase D.4 smoke test: end-to-end dynfield SVI atom-update step.

Verifies:

  1. `init_svi_state_dynfield` produces an SVIState with the expected
     dyn_field shape contract.
  2. `state.coupling` dispatches to DynamicFieldCouplingModel when
     coupling_variant='dynamic_field'.
  3. `train_dynfield_one_iter` runs end-to-end on a small synthetic
     corpus of coupled + uncoupled cherries, and the cumulative log
     likelihood improves over a few iterations.

The Potts side is exercised by the existing test_a1_corrections /
test_a1_rung suites; we don't re-test it here.

Run: python3 tests/dynfield/test_svi_dynfield.py
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

from tkfdp.svi import (SVIState, init_svi_state_dynfield,
                          train_dynfield_one_iter)
from tkfdp.coupling.dynfield import DynamicFieldCouplingModel
from tkfdp.coupling.dynfield.state import DynamicFieldState
from tkfdp.lg08 import S_LG08_F81_J as S_LG08_J


# ---------------------------------------------------------------------------
# Fixture: build a tiny synthetic Pfam-shaped corpus.
# ---------------------------------------------------------------------------

def _build_per_family_data(n_families=2, L=8, n_cherries_per_family=20,
                            K_c=2, A=20, seed=0):
    """Build a per-family data list with the minimum fields
    init_svi_state_dynfield expects: family, L, K (n_pairs)."""
    rng = np.random.default_rng(seed)
    per_family = []
    for f in range(n_families):
        per_family.append({
            'family': f'F{f}',
            'L': L,
            'K': max(1, L // 4),      # n_pairs hint for init_random_K
        })
    return per_family


def _make_truth_model(K_c=2, L_max=2, A=20, seed=0):
    rng = np.random.default_rng(seed)
    pi_field = rng.dirichlet(np.ones(A) * 0.5, size=(K_c, L_max))
    rho = rng.dirichlet(np.ones(L_max) * 1.5)
    pi_class = np.einsum('t,cta->ca', rho, pi_field)
    state = DynamicFieldState(
        K_c=K_c, A=A,
        pi_field=pi_field, rho=rho, tsb_betas=None, alpha_field=1.0,
        pi_class=pi_class, rho_chain=1.0,
    )
    return DynamicFieldCouplingModel(K_c=K_c, A=A, pi_class=pi_class,
                                       dyn_field=state)


def _simulate_cherries(truth_model, *, n_coupled, n_uncoupled,
                         t: float, seed: int):
    """Generate coupled + uncoupled cherries from the truth model."""
    rng = np.random.default_rng(seed)
    K_c, L_max, A = truth_model.dyn_field.pi_field.shape
    pi_c = np.full(K_c, 1.0 / K_c)
    pi_field = truth_model.dyn_field.pi_field
    rho = truth_model.dyn_field.rho
    rho_chain = float(truth_model.dyn_field.rho_chain)
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
        theta_now = int(theta_0)
        elapsed = 0.0; n_jumps = 0
        while True:
            rate = rho_chain * (1.0 - rho[theta_now])
            if rate <= 0: break
            dt = rng.exponential(1.0 / rate)
            if elapsed + dt > s: break
            elapsed += dt
            p_targets = rho.copy(); p_targets[theta_now] = 0.0
            denom = p_targets.sum()
            if denom <= 0: break
            p_targets /= denom
            theta_now = int(rng.choice(L_max, p=p_targets))
            n_jumps += 1
        return theta_now, n_jumps

    clusters = []
    # Coupled (size 2) clusters.
    for _ in range(n_coupled):
        c_i = int(rng.choice(K_c, p=pi_c))
        c_j = int(rng.choice(K_c, p=pi_c))
        th_P = int(rng.choice(L_max, p=rho))
        p_i = int(rng.choice(A, p=pi_field[c_i, th_P]))
        p_j = int(rng.choice(A, p=pi_field[c_j, th_P]))
        th_X, nj_X = _gillespie(th_P, t / 2.0)
        if nj_X == 0:
            Xi = int(rng.choice(A, p=P_half[c_i, th_P, p_i]))
            Xj = int(rng.choice(A, p=P_half[c_j, th_P, p_j]))
        else:
            Xi = int(rng.choice(A, p=pi_field[c_i, th_X]))
            Xj = int(rng.choice(A, p=pi_field[c_j, th_X]))
        th_Y, nj_Y = _gillespie(th_P, t / 2.0)
        if nj_Y == 0:
            Yi = int(rng.choice(A, p=P_half[c_i, th_P, p_i]))
            Yj = int(rng.choice(A, p=P_half[c_j, th_P, p_j]))
        else:
            Yi = int(rng.choice(A, p=pi_field[c_i, th_Y]))
            Yj = int(rng.choice(A, p=pi_field[c_j, th_Y]))
        clusters.append((np.array([c_i, c_j]),
                          np.array([Xi, Xj]),
                          np.array([Yi, Yj]), t))
    # Singleton (size 1) clusters.
    for _ in range(n_uncoupled):
        c = int(rng.choice(K_c, p=pi_c))
        th_P = int(rng.choice(L_max, p=rho))
        p = int(rng.choice(A, p=pi_field[c, th_P]))
        th_X, nj_X = _gillespie(th_P, t / 2.0)
        if nj_X == 0:
            Xi = int(rng.choice(A, p=P_half[c, th_P, p]))
        else:
            Xi = int(rng.choice(A, p=pi_field[c, th_X]))
        th_Y, nj_Y = _gillespie(th_P, t / 2.0)
        if nj_Y == 0:
            Yi = int(rng.choice(A, p=P_half[c, th_P, p]))
        else:
            Yi = int(rng.choice(A, p=pi_field[c, th_Y]))
        clusters.append((np.array([c]), np.array([Xi]),
                          np.array([Yi]), t))
    return clusters


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

def test_init_svi_state_dynfield_shape_contract():
    print("\n[test 1] init_svi_state_dynfield shape contract")
    per_fam = _build_per_family_data(n_families=2, L=6)
    state = init_svi_state_dynfield(
        per_fam, K_c=3, A=20, L_max=4, alpha_field=1.5, rho_chain=1.0)
    assert state.K_c == 3
    assert state.A == 20
    assert state.coupling_variant == 'dynamic_field'
    assert state.potts_dp is None
    assert state.dyn_field is not None
    assert state.dyn_field.pi_field.shape == (3, 4, 20)
    assert state.dyn_field.rho.shape == (4,)
    assert abs(state.dyn_field.rho.sum() - 1.0) < 1e-12
    print(f"  K_c={state.K_c}, A={state.A}, L_max={state.dyn_field.L_max}, "
          f"rho_chain={state.dyn_field.rho_chain}")
    print(f"  variant tag: {state.coupling_variant!r}")
    print(f"  pi_field shape: {state.dyn_field.pi_field.shape}")
    print("  PASS")


def test_state_coupling_dispatches_dynfield():
    print("\n[test 2] state.coupling dispatches to DynamicFieldCouplingModel")
    per_fam = _build_per_family_data(n_families=1, L=4)
    state = init_svi_state_dynfield(per_fam, K_c=2, L_max=2)
    coupling = state.coupling
    assert isinstance(coupling, DynamicFieldCouplingModel)
    assert coupling.variant == 'dynamic_field'
    # Try a build_doublet_emission call to verify dispatch wiring.
    P_doublet = coupling.build_doublet_emission(
        t=0.4, pi_c=np.array([0.5, 0.5]), pair_background='per_class')
    assert P_doublet.shape == (20, 20, 20, 20)
    assert (P_doublet >= -1e-15).all()
    print(f"  coupling.variant = {coupling.variant!r}")
    print(f"  P_doublet sum (should be close to 1): {float(P_doublet.sum()):.4f}")
    print("  PASS")


def test_train_dynfield_one_iter_improves_LL():
    print("\n[test 3] train_dynfield_one_iter improves cumulative log LL")
    K_c, L_max = 2, 2
    truth = _make_truth_model(K_c=K_c, L_max=L_max, A=20, seed=300)
    clusters = _simulate_cherries(
        truth, n_coupled=500, n_uncoupled=500, t=0.4, seed=301)
    # Init student.
    per_fam = _build_per_family_data(n_families=1, L=4)
    state = init_svi_state_dynfield(
        per_fam, K_c=K_c, L_max=L_max, rho_chain=1.0)
    # Train.
    log_liks = []
    for it in range(6):
        state, info = train_dynfield_one_iter(state, clusters)
        log_liks.append(info['log_lik_total'])
        print(f"  iter {it + 1}: LL_total = {info['log_lik_total']:.2f}  "
              f"(n_clusters={info['n_clusters']})")
    assert log_liks[-1] > log_liks[0], (
        f"training did not improve LL: {log_liks[0]:.2f} -> {log_liks[-1]:.2f}")
    print(f"  LL gain: {log_liks[-1] - log_liks[0]:+.2f} nats over 6 iters")
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
    print(f"\nAll {len(fns)} SVI dynfield smoke tests passed.")
