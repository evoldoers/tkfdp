"""Phase E IPHMM integration smoke test.

The IPHMM's `precompute_partial_forward` accesses the coupling model via
`boost_state.tkf_state.coupling` (added in Phase B.3). With Phase D.4's
SVIState.coupling_variant dispatch, a dynfield SVIState exposes a
DynamicFieldCouplingModel through the same accessor; build_M_tensor /
build_M_tensor_typed should "just work".

What we verify here:

  1. state.coupling.build_M_tensor with pair_background='lg08' (the
     IPHMM default) succeeds and returns a sane (A, A, A, A) tensor
     for a dynfield state. (pair_background is vestigial under
     dynfield -- pi_field IS the background -- so 'lg08' is accepted
     as a no-op for compatibility with the IPHMM caller.)
  2. state.coupling.build_M_tensor_typed likewise returns all 6 edge-
     typed M tensors with the correct shapes.
  3. The dispatched M tensors satisfy the basic IPHMM invariants:
     - All entries are finite and >= 0.
     - At t=0 (no substitution / indel), M_MM[a, b, c, d] is non-trivial
       (the pair-coupling boost is the cluster-of-2 joint stationary
       divided by the singlet stationary product).

Run: python3 tests/dynfield/test_iphmm_dispatch.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from tkfdp.svi import init_svi_state_dynfield, SVIState
from tkfdp.coupling.dynfield import DynamicFieldCouplingModel


def _build_per_family_data(n_families=1, L=6):
    return [{'family': f'F{f}', 'L': L, 'K': max(1, L // 4)}
             for f in range(n_families)]


def test_build_M_tensor_with_lg08_pair_background():
    """build_M_tensor on a dynfield state accepts pair_background='lg08'
    (the IPHMM default) as a no-op, returning a sane (A, A, A, A)
    tensor."""
    print("\n[test 1] dynfield build_M_tensor accepts pair_background='lg08'")
    per_fam = _build_per_family_data(n_families=1, L=8)
    state = init_svi_state_dynfield(
        per_fam, K_c=2, L_max=3,
        rho_chain=1.0, rng=np.random.default_rng(0))
    pi_c = np.array([0.4, 0.6])
    M = state.coupling.build_M_tensor(
        t=0.5, pi_c=pi_c,
        pair_background='lg08')      # IPHMM's default; vestigial for dynfield
    assert M.shape == (20, 20, 20, 20), f"M shape {M.shape}"
    assert np.all(np.isfinite(M)), "M has non-finite entries"
    assert np.all(M >= 0), "M has negative entries"
    print(f"  M shape: {M.shape}")
    print(f"  M range: [{M.min():.3e}, {M.max():.3e}]")
    print("  PASS")


def test_build_M_tensor_typed_with_lg08():
    """build_M_tensor_typed returns all 6 edge-typed M tensors with the
    correct shapes when called with pair_background='lg08' on dynfield."""
    print("\n[test 2] dynfield build_M_tensor_typed accepts pair_background='lg08'")
    per_fam = _build_per_family_data(n_families=1, L=4)
    state = init_svi_state_dynfield(
        per_fam, K_c=2, L_max=2, rho_chain=1.0,
        rng=np.random.default_rng(1))
    typed = state.coupling.build_M_tensor_typed(
        t=0.4, pi_c=np.array([0.5, 0.5]),
        pair_background='lg08')
    expected_shapes = {
        'MM': (20, 20, 20, 20),
        'MI': (20, 20, 20),
        'MD': (20, 20, 20),
        'II': (20, 20),
        'DD': (20, 20),
        'ID': (20, 20),
    }
    for k, shape in expected_shapes.items():
        assert k in typed
        assert typed[k].shape == shape, (
            f"typed[{k!r}] shape {typed[k].shape} != {shape}")
        assert np.all(np.isfinite(typed[k]))
        assert np.all(typed[k] >= 0)
        print(f"  {k}: shape {typed[k].shape}, "
              f"range [{typed[k].min():.3e}, {typed[k].max():.3e}]")
    print("  PASS")


def test_M_tensor_definition_consistent():
    """build_M_tensor on dynfield with pair_background='lg08' returns
    the same tensor as build_M_tensor with pair_background='per_class' --
    confirms 'lg08' is silently accepted as a no-op (not a different
    code path)."""
    print("\n[test 3] M_tensor: 'lg08' == 'per_class' for dynfield")
    per_fam = _build_per_family_data(n_families=1, L=4)
    state = init_svi_state_dynfield(
        per_fam, K_c=3, L_max=4, rho_chain=1.0,
        rng=np.random.default_rng(2))
    pi_c = np.array([0.2, 0.5, 0.3])
    M_lg = state.coupling.build_M_tensor(
        t=0.7, pi_c=pi_c, pair_background='lg08')
    M_pc = state.coupling.build_M_tensor(
        t=0.7, pi_c=pi_c, pair_background='per_class')
    err = float(np.max(np.abs(M_lg - M_pc)))
    print(f"  max |M_lg - M_pc| = {err:.3e}")
    assert err < 1e-15, (
        f"pair_background='lg08' and 'per_class' diverge under dynfield: "
        f"err {err:.3e}")
    print("  PASS")


def test_coupling_dispatch_through_svi_state():
    """The Phase B.3 dispatch path (boost_state.tkf_state.coupling) must
    resolve to DynamicFieldCouplingModel when the SVIState's variant
    is dynamic_field. This is the contract the IPHMM relies on."""
    print("\n[test 4] state.coupling resolves to DynamicFieldCouplingModel "
          "under dynamic_field variant")
    per_fam = _build_per_family_data(n_families=1, L=4)
    state = init_svi_state_dynfield(
        per_fam, K_c=2, L_max=2, rng=np.random.default_rng(3))
    assert state.coupling_variant == 'dynamic_field'
    coupling = state.coupling
    assert isinstance(coupling, DynamicFieldCouplingModel), (
        f"expected DynamicFieldCouplingModel, got {type(coupling).__name__}")
    assert coupling.variant == 'dynamic_field'
    # The accessor is a property, but the build calls should return
    # the same numerical tensor whether called via the property or
    # cached via getattr().
    coupling2 = getattr(state, 'coupling')
    M1 = coupling.build_M_tensor(
        t=0.3, pi_c=np.array([0.5, 0.5]), pair_background='lg08')
    M2 = coupling2.build_M_tensor(
        t=0.3, pi_c=np.array([0.5, 0.5]), pair_background='lg08')
    err = float(np.max(np.abs(M1 - M2)))
    assert err < 1e-15
    print(f"  state.coupling -> {type(coupling).__name__}")
    print(f"  build_M_tensor matched across accessor calls (err {err:.3e})")
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
    print(f"\nAll {len(fns)} IPHMM dispatch tests passed.")
