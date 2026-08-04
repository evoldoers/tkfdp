"""Smoke test for DynamicFieldCouplingModel (Phase C).

What we verify:

  1. Instantiation through the variant registry resolves to the new class.
  2. `build_doublet_emission(...)` matches an explicit Phase-A 4-case
     construction to machine precision, at multiple branch lengths.
  3. The K_c-marginalised doublet marginalises (over the right-column
     observations) to the singlet -- the dynamic-field analogue of the
     "no Sinkhorn needed" claim verified at the stationary level in
     test_math_precompute.py::test_field_marginal_joint_is_marginal_consistent,
     now extended to the full t-dependent cherry joint.
  4. `build_M_tensor(...)` equals `P_doublet / (P_singlet ⊗ P_singlet)`.
  5. `build_M_tensor_typed(...)` returns the six edge-type tensors with
     the correct shapes.
  6. `to_npz` / `from_npz` round-trips the persistent state.

We do NOT test "L_max=1 reduces to per-class GTR" because that
reduction does NOT hold: at L_max=1 the F81-on-DP field chain still has
rate-1 self-jumps which, under instant re-equilibration, refresh the
residue from `pi_field[c, 0]` at exponentially distributed times. So
dynfield at L_max=1 is a covarion-like model, not plain per-class GTR.
The design-doc G.4 "Potts-limit consistency" test needs revisiting; this
file documents the mismatch.

Run: python3 tests/dynfield/test_coupling_model.py
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
from tkfdp.lg08 import S_LG08_F81_J as S_LG08_J


# ---------------------------------------------------------------------------
# Fixture builder.
# ---------------------------------------------------------------------------

def make_fixture(K_c: int = 2, L_max: int = 3, A: int = 20, seed: int = 0
                 ):
    """Build a small DynamicFieldCouplingModel for testing."""
    rng = np.random.default_rng(seed)
    pi_field = rng.dirichlet(np.ones(A) * 0.7, size=(K_c, L_max))   # (K, L, A)
    # Stick-breaking rho.
    betas = rng.beta(1.0, 1.0, size=L_max - 1)
    remaining = 1.0
    rho = np.empty(L_max)
    for i in range(L_max - 1):
        rho[i] = remaining * betas[i]
        remaining *= 1.0 - betas[i]
    rho[L_max - 1] = remaining
    # Field-marginal pi as pi_class.
    pi_class = np.einsum('t,cta->ca', rho, pi_field)
    state = DynamicFieldState(
        K_c=K_c, A=A,
        pi_field=pi_field, rho=rho, tsb_betas=betas, alpha_field=1.0,
        pi_class=pi_class,
    )
    Model = get_variant('dynamic_field')
    return Model(K_c=K_c, A=A, pi_class=pi_class, dyn_field=state), pi_field, rho


# ---------------------------------------------------------------------------
# Test 1: registry + instantiation.
# ---------------------------------------------------------------------------

def test_registry_resolves_dynamic_field():
    print("\n[test 1] Registry resolves 'dynamic_field' variant")
    Model = get_variant('dynamic_field')
    assert Model.variant == 'dynamic_field', (
        f"got variant tag {Model.variant!r}")
    print(f"  variant tag = {Model.variant!r}")
    model, _, _ = make_fixture()
    assert model.variant == 'dynamic_field'
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2: explicit 4-case construction agreement.
# ---------------------------------------------------------------------------

def _explicit_4case_doublet(t, pi_c, pi_field, rho, S_arr, rho_chain=1.0):
    """Independent Interp 2 (no-self-jump CTMC) reconstruction of the
    4-case cluster joint. Returns (A, A, A, A) in the block_likelihoods
    layout (a, b, c, d) = (Xi, Yi, Xj, Yj)."""
    K_c, L_max, A = pi_field.shape
    s = t / 2.0
    alpha = float(np.exp(-rho_chain * s))
    beta = np.exp(-rho_chain * (1.0 - rho) * s)                 # (L_max,)
    # Per-(c, theta) generator and half-edge transition.
    S_off = S_arr - np.diag(np.diag(S_arr))
    Q_cf = np.zeros((K_c, L_max, A, A))
    for c in range(K_c):
        for th in range(L_max):
            pi = pi_field[c, th]
            Q = S_off * pi[None, :]
            np.fill_diagonal(Q, -Q.sum(axis=1))
            Q_cf[c, th] = Q
    P_half = np.zeros_like(Q_cf)
    for c in range(K_c):
        for th in range(L_max):
            P_half[c, th] = expm(Q_cf[c, th] * s)
    # Per-(c, theta) cherry joint Sigma.
    Sigma = np.einsum('ctp,ctpa,ctpb->ctab', pi_field, P_half, P_half)
    # Field-marginal joint J^(c1, c2).
    J = np.einsum('t,uta,vtb->uvab', rho, pi_field, pi_field)        # (K, K, A, A)
    # Interp 2: J' per theta_P = w_J J + w_pi pi(theta_P) outer pi(theta_P).
    denom = 1.0 - beta
    safe = denom > 1e-15
    w_J = np.where(safe, (1.0 - alpha) / np.where(safe, denom, 1.0), 1.0)
    w_pi = np.where(safe, (alpha - beta) / np.where(safe, denom, 1.0), 0.0)
    J_prime = (np.einsum('t,uvab->uvtab', w_J, J)
                + np.einsum('t,uta,vtb->uvtab', w_pi, pi_field, pi_field))
    # Internal layout (c1, c2, Xi, Xj, Yi, Yj).
    nn = rho * beta * beta
    T1 = np.einsum('t,utxa,vtyb->uvxyab', nn, Sigma, Sigma)
    ny = rho * beta * (1.0 - beta)
    T2 = np.einsum('t,utx,vty,uvtab->uvxyab', ny, pi_field, pi_field, J_prime)
    yn = rho * (1.0 - beta) * beta
    T3 = np.einsum('t,uvtxy,uta,vtb->uvxyab', yn, J_prime, pi_field, pi_field)
    yy = rho * (1.0 - beta) ** 2
    T4 = np.einsum('t,uvtxy,uvtab->uvxyab', yy, J_prime, J_prime)
    P_internal = T1 + T2 + T3 + T4
    # K_c-marginalise + transpose to (a, b, c, d) = (Xi, Yi, Xj, Yj).
    weights = pi_c[:, None] * pi_c[None, :]
    marg = np.einsum('uv,uvxyab->xyab', weights, P_internal)
    return np.transpose(marg, (0, 2, 1, 3))


def test_doublet_matches_explicit_4case():
    print("\n[test 2] build_doublet_emission == explicit 4-case construction")
    model, pi_field, rho = make_fixture(K_c=2, L_max=3, A=20, seed=12)
    pi_c = np.array([0.4, 0.6])
    S_arr = np.asarray(S_LG08_J, dtype=np.float64)
    max_err = 0.0
    for t in [0.2, 0.7, 2.0]:
        P_doublet = model.build_doublet_emission(
            t, pi_c=pi_c, pair_background='per_class')
        P_ref = _explicit_4case_doublet(t, pi_c, pi_field, rho, S_arr)
        err = float(np.max(np.abs(P_doublet - P_ref)))
        print(f"  t={t}: max |adapter - explicit| = {err:.3e}")
        max_err = max(max_err, err)
        assert P_doublet.shape == (20, 20, 20, 20)
        assert (P_doublet >= -1e-15).all(), "doublet has negative entries"
    assert max_err < 1e-10, (
        f"adapter and explicit 4-case disagree: max err {max_err:.3e}")
    print(f"  max err over t in [0.2, 0.7, 2.0] = {max_err:.3e}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: marginal consistency at arbitrary t (no Sinkhorn needed).
# ---------------------------------------------------------------------------

def test_doublet_marginalises_to_singlet():
    """Sum of P_doublet over the right column (c, d) should equal
    P_singlet on the left column (a, b). This is the t-dependent
    extension of the stationary marginal-consistency claim verified by
    test_math_precompute.py::test_field_marginal_joint_is_marginal_consistent."""
    print("\n[test 3] sum_{c,d} P_doublet[a,b,c,d] == P_singlet[a, b]")
    model, _, _ = make_fixture(K_c=2, L_max=3, A=20, seed=13)
    pi_c = np.array([0.3, 0.7])
    max_err_right = 0.0
    max_err_left = 0.0
    for t in [0.2, 0.7, 2.0]:
        P_doublet = model.build_doublet_emission(
            t, pi_c=pi_c, pair_background='per_class')
        P_singlet, _, _ = model.build_singlet_emission(t, pi_c=pi_c)
        # marginal over the right column (axes 2, 3).
        marg_right = P_doublet.sum(axis=(2, 3))
        # marginal over the left column (axes 0, 1).
        marg_left = P_doublet.sum(axis=(0, 1))
        err_r = float(np.max(np.abs(marg_right - P_singlet)))
        err_l = float(np.max(np.abs(marg_left - P_singlet)))
        max_err_right = max(max_err_right, err_r)
        max_err_left = max(max_err_left, err_l)
        print(f"  t={t}: right-marg err={err_r:.3e}, "
              f"left-marg err={err_l:.3e}")
        assert err_r < 1e-12, f"right marginal mismatch at t={t}: {err_r:.3e}"
        assert err_l < 1e-12, f"left marginal mismatch at t={t}: {err_l:.3e}"
    print(f"  max err = {max(max_err_right, max_err_left):.3e}")
    print("  ==> no Sinkhorn correction needed. PASS")


# ---------------------------------------------------------------------------
# Test 4: M-tensor = P_doublet / (P_singlet ⊗ P_singlet).
# ---------------------------------------------------------------------------

def test_M_tensor_definition():
    print("\n[test 4] build_M_tensor = P_doublet / (P_singlet ⊗ P_singlet)")
    model, _, _ = make_fixture(K_c=2, L_max=3, A=20, seed=14)
    pi_c = np.array([0.4, 0.6])
    for t in [0.3, 1.0]:
        P_singlet, _, _ = model.build_singlet_emission(t, pi_c=pi_c)
        P_doublet = model.build_doublet_emission(
            t, pi_c=pi_c, pair_background='per_class')
        M = model.build_M_tensor(t, pi_c=pi_c, pair_background='per_class')
        denom = P_singlet[:, :, None, None] * P_singlet[None, None, :, :]
        M_ref = P_doublet / np.clip(denom, 1e-300, None)
        err = float(np.max(np.abs(M - M_ref)))
        print(f"  t={t}: max |M - P_doublet/denom| = {err:.3e}")
        assert err < 1e-12
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 5: M-tensor typed shape contract.
# ---------------------------------------------------------------------------

def test_M_tensor_typed_shapes():
    print("\n[test 5] build_M_tensor_typed shape contract")
    model, _, _ = make_fixture(K_c=2, L_max=3, A=20, seed=15)
    pi_c = np.array([0.4, 0.6])
    typed = model.build_M_tensor_typed(0.5, pi_c=pi_c,
                                        pair_background='per_class')
    expected_shapes = {
        'MM': (20, 20, 20, 20),
        'MI': (20, 20, 20),
        'MD': (20, 20, 20),
        'II': (20, 20),
        'DD': (20, 20),
        'ID': (20, 20),
    }
    for k, exp in expected_shapes.items():
        assert k in typed, f"typed dict missing key {k!r}"
        assert typed[k].shape == exp, (
            f"typed[{k!r}] shape {typed[k].shape} != {exp}")
        print(f"  {k}: shape {typed[k].shape}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 6: to_npz / from_npz round-trip.
# ---------------------------------------------------------------------------

def test_npz_roundtrip():
    print("\n[test 6] to_npz / from_npz round-trip")
    model, _, _ = make_fixture(K_c=3, L_max=4, A=20, seed=16)
    arrs = model.to_npz()
    expected_keys = {'pi_class', 'pi_field', 'rho', 'tsb_betas'}
    assert set(arrs.keys()) >= {'pi_class', 'pi_field', 'rho'}, (
        f"missing keys in to_npz output; got {sorted(arrs.keys())}")
    meta = {'K_c': 3, 'alpha_field': 1.0}
    Model = get_variant('dynamic_field')
    rebuilt = Model.from_npz(arrs, meta)
    err_pi = float(np.max(np.abs(rebuilt.pi_class - model.pi_class)))
    err_pf = float(np.max(np.abs(rebuilt.dyn_field.pi_field
                                  - model.dyn_field.pi_field)))
    err_rho = float(np.max(np.abs(rebuilt.dyn_field.rho - model.dyn_field.rho)))
    print(f"  pi_class err={err_pi:.3e}, pi_field err={err_pf:.3e}, "
          f"rho err={err_rho:.3e}")
    assert max(err_pi, err_pf, err_rho) < 1e-15
    print("  PASS")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import inspect, sys
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
    print(f"\nAll {len(fns)} coupling-model smoke tests passed.")
