"""Tests for the rho_chain (F81-on-DP rate multiplier) parameter.

Verifies the Potts-limit consistency claim (revised G.4 from
docs/dynfield_design.md): as rho_chain -> 0, the dynamic-field cap-2
cluster joint emission reduces to the per-class GTR cherry joint
product (independent columns, no field coupling).

Also verifies that the f81_dp_transition decay rate scales linearly
with rho_chain (sanity check for the rate parameterisation).

Run: python3 tests/dynfield/test_rho_chain.py
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
from tkfdp.coupling.dynfield import dp_field as _dp
from tkfdp.lg08 import S_LG08_F81_J as S_LG08_J


def make_model(pi_field, rho, *, rho_chain=1.0, K_c=None, A=20):
    K_c = K_c if K_c is not None else pi_field.shape[0]
    pi_class = np.einsum('t,cta->ca', rho, pi_field)
    state = DynamicFieldState(
        K_c=K_c, A=A,
        pi_field=pi_field, rho=rho, tsb_betas=None, alpha_field=1.0,
        pi_class=pi_class, rho_chain=rho_chain,
    )
    Model = get_variant('dynamic_field')
    return Model(K_c=K_c, A=A, pi_class=pi_class, dyn_field=state)


# ---------------------------------------------------------------------------
# Test 1: F81-on-DP decay rate scales linearly with rho_chain.
# ---------------------------------------------------------------------------

def test_f81_dp_decay_scales_with_rho_chain():
    """f81_dp_transition(rho, t, rho_chain) should equal
    f81_dp_transition(rho, rho_chain * t, 1.0) to machine precision -- the
    rate enters multiplicatively in the decay exp(-rho_chain * t)."""
    print("\n[test 1] f81_dp_transition decay scales linearly with rho_chain")
    rho = np.array([0.5, 0.3, 0.2])
    t = 0.7
    P_unit = _dp.f81_dp_transition(rho, t * 0.4, rho_chain=1.0)
    P_scaled = _dp.f81_dp_transition(rho, t, rho_chain=0.4)
    err = float(np.max(np.abs(P_unit - P_scaled)))
    print(f"  max |P(t*r, rho_chain=1) - P(t, rho_chain=r)| = {err:.3e}")
    assert err < 1e-15
    # Also verify against expm of the rate-scaled generator.
    Q = _dp.f81_dp_generator(rho, rho_chain=0.4)
    P_expm = expm(Q * t)
    err2 = float(np.max(np.abs(P_scaled - P_expm)))
    print(f"  max |P_closed - expm(rho_chain * Q * t)|       = {err2:.3e}")
    assert err2 < 1e-12
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2: rho_chain=0 -> P_doublet = per-class GTR cherry joint product.
# ---------------------------------------------------------------------------

def test_potts_limit_rho_chain_zero():
    """At rho_chain=0 the field NEVER jumps (p_nj = 1; nn_weight = 1; all
    other case weights = 0). The dynfield doublet should reduce to the
    product of per-(c, theta_P) GTR cherry joints, summed over theta_P
    with rho. With L_max=1 (single field atom) this is exactly the
    per-class GTR cherry joint product.
    """
    print("\n[test 2] rho_chain=0 reduces to per-class GTR cherry joint product")
    K_c, L_max, A = 2, 1, 20
    rng = np.random.default_rng(33)
    pi_field = rng.dirichlet(np.ones(A) * 0.5, size=(K_c, L_max))
    rho = np.array([1.0])
    pi_c = np.array([0.4, 0.6])
    # rho_chain = 0 -> no field jumps.
    model = make_model(pi_field, rho, rho_chain=0.0)
    t = 0.5
    P_doublet = model.build_doublet_emission(
        t, pi_c=pi_c, pair_background='per_class')           # (A, A, A, A)
    # Reference: explicit per-class GTR cherry joint product.
    S_np = np.asarray(S_LG08_J, dtype=np.float64)
    S_off = S_np - np.diag(np.diag(S_np))
    Sigma_per_class = np.zeros((K_c, A, A))
    for c in range(K_c):
        pi = pi_field[c, 0]
        Q_c = S_off * pi[None, :]
        np.fill_diagonal(Q_c, -Q_c.sum(axis=1))
        P_half = expm(Q_c * t / 2.0)
        Sigma_per_class[c] = np.einsum('p,pa,pb->ab', pi, P_half, P_half)
    # Per-class doublet = product of singletons (uncoupled GTR per column).
    P_ref = np.zeros((A, A, A, A))
    for c1 in range(K_c):
        for c2 in range(K_c):
            # Layout (Xi, Yi, Xj, Yj):
            P_ref += pi_c[c1] * pi_c[c2] * np.einsum(
                'xa,yb->xayb', Sigma_per_class[c1], Sigma_per_class[c2])
    err = float(np.max(np.abs(P_doublet - P_ref)))
    print(f"  max |dynfield(rho_chain=0) - per-class GTR product| = {err:.3e}")
    assert err < 1e-12, (
        f"Potts-limit residual {err:.3e} > 1e-12")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: rho_chain -> infty -> P_doublet -> field-marginal stationary product.
# ---------------------------------------------------------------------------

def test_high_rho_chain_limit():
    """At rho_chain >> 1 the field jumps fast; per-edge no-jump probability
    p_nj = exp(-rho_chain * t/2) -> 0. All (nn, *) and (*, nn) terms
    vanish; only (yj, yj) survives with weight (1 - p_nj)^2 -> 1. So
    P_doublet -> J^(c1, c2)(X) * J^(c1, c2)(Y) class-marginalised.
    """
    print("\n[test 3] rho_chain -> infty reduces to field-marginal "
          "stationary product")
    K_c, L_max, A = 2, 3, 20
    rng = np.random.default_rng(34)
    pi_field = rng.dirichlet(np.ones(A) * 0.5, size=(K_c, L_max))
    rho = rng.dirichlet(np.ones(L_max))
    pi_c = np.array([0.4, 0.6])
    model_inf = make_model(pi_field, rho, rho_chain=200.0)
    t = 0.5
    P_doublet = model_inf.build_doublet_emission(
        t, pi_c=pi_c, pair_background='per_class')
    # Reference: J^(c1, c2)(X) * J^(c1, c2)(Y) marginalised.
    J = np.einsum('t,uta,vtb->uvab', rho, pi_field, pi_field)        # (K, K, A, A)
    weights = pi_c[:, None] * pi_c[None, :]
    # Internal layout (Xi, Xj, Yi, Yj):
    P_ref_internal = np.einsum('uv,uvxy,uvab->xyab', weights, J, J)
    # Transpose to (Xi, Yi, Xj, Yj):
    P_ref = np.transpose(P_ref_internal, (0, 2, 1, 3))
    err = float(np.max(np.abs(P_doublet - P_ref)))
    print(f"  rho_chain=200, t=0.5: max |dynfield - J⊗J marg| = {err:.3e}")
    # p_nj = exp(-100) ~ 0; residual should be at machine precision.
    assert err < 1e-12
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 4: rho_chain=1.0 default reproduces Phase A 4-case formula.
# ---------------------------------------------------------------------------

def test_rho_chain_default_matches_explicit_interp2():
    """rho_chain=1.0 default reproduces the Interp 2 (no-self-jump CTMC)
    4-case formula with the J' convex-combination decomposition.
    """
    print("\n[test 4] rho_chain=1.0 default matches explicit Interp 2 4-case")
    K_c, L_max, A = 2, 3, 20
    rng = np.random.default_rng(35)
    pi_field = rng.dirichlet(np.ones(A) * 0.5, size=(K_c, L_max))
    rho = rng.dirichlet(np.ones(L_max))
    pi_c = np.array([0.5, 0.5])
    model = make_model(pi_field, rho)   # rho_chain defaults to 1.0
    t = 0.4
    rho_chain = 1.0
    P_doublet = model.build_doublet_emission(
        t, pi_c=pi_c, pair_background='per_class')
    # Explicit Interp 2 4-case.
    S_np = np.asarray(S_LG08_J, dtype=np.float64)
    S_off = S_np - np.diag(np.diag(S_np))
    Q_cf = np.zeros((K_c, L_max, A, A))
    for c in range(K_c):
        for th in range(L_max):
            Q = S_off * pi_field[c, th][None, :]
            np.fill_diagonal(Q, -Q.sum(axis=1))
            Q_cf[c, th] = Q
    P_half = np.zeros_like(Q_cf)
    for c in range(K_c):
        for th in range(L_max):
            P_half[c, th] = expm(Q_cf[c, th] * t / 2.0)
    Sigma = np.einsum('ctp,ctpa,ctpb->ctab', pi_field, P_half, P_half)
    J = np.einsum('t,uta,vtb->uvab', rho, pi_field, pi_field)
    s = t / 2.0
    alpha = float(np.exp(-rho_chain * s))
    beta = np.exp(-rho_chain * (1.0 - rho) * s)
    denom = 1.0 - beta
    safe = denom > 1e-15
    w_J = np.where(safe, (1.0 - alpha) / np.where(safe, denom, 1.0), 1.0)
    w_pi = np.where(safe, (alpha - beta) / np.where(safe, denom, 1.0), 0.0)
    J_prime = (np.einsum('t,uvab->uvtab', w_J, J)
                + np.einsum('t,uta,vtb->uvtab', w_pi, pi_field, pi_field))
    nn = rho * beta * beta
    T1 = np.einsum('t,utxa,vtyb->uvxyab', nn, Sigma, Sigma)
    ny = rho * beta * (1.0 - beta)
    T2 = np.einsum('t,utx,vty,uvtab->uvxyab', ny, pi_field, pi_field, J_prime)
    yn = rho * (1.0 - beta) * beta
    T3 = np.einsum('t,uvtxy,uta,vtb->uvxyab', yn, J_prime, pi_field, pi_field)
    yy = rho * (1.0 - beta) ** 2
    T4 = np.einsum('t,uvtxy,uvtab->uvxyab', yy, J_prime, J_prime)
    P_internal = T1 + T2 + T3 + T4
    weights = pi_c[:, None] * pi_c[None, :]
    marg = np.einsum('uv,uvxyab->xyab', weights, P_internal)
    P_ref = np.transpose(marg, (0, 2, 1, 3))
    err = float(np.max(np.abs(P_doublet - P_ref)))
    print(f"  max |default rho_chain=1.0 vs explicit Interp 2| = {err:.3e}")
    assert err < 1e-12
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
    print(f"\nAll {len(fns)} rho_chain tests passed.")
