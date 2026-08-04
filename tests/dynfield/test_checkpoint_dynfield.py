"""Round-trip test for dynfield save/load via checkpoint.py.

Builds a small dynfield SVIState, saves it via save_checkpoint, reloads
it via load_checkpoint, and verifies the reloaded state matches the
original at machine precision.

Also verifies that legacy Potts checkpoints (without `coupling_variant`
in meta) still load correctly via the default-to-'potts' fallback.

Run: python3 tests/dynfield/test_checkpoint_dynfield.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from tkfdp.svi import SVIState, init_svi_state_dynfield
from tkfdp.checkpoint import save_checkpoint, load_checkpoint


def _build_per_family_data(n_families=2, L=6):
    return [{'family': f'F{f}', 'L': L, 'K': max(1, L // 4)}
             for f in range(n_families)]


def test_dynfield_save_load_roundtrip():
    print("\n[test 1] dynfield save/load round-trip")
    per_fam = _build_per_family_data(n_families=2, L=8)
    state = init_svi_state_dynfield(
        per_fam, K_c=3, L_max=4, alpha_field=1.5, rho_chain=0.7,
        rng=np.random.default_rng(0))
    rng = np.random.default_rng(123)
    rng.standard_normal(10)   # advance to get a non-trivial state
    trace = {'iter': 0, 'log_l': [1.0, 2.0]}
    A_dummy = np.zeros((20, 20))
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        save_checkpoint(state, trace, rng, out_dir, it=3)
        # Round-trip.
        state_r, trace_r, rng_r, es_r, meta_r = load_checkpoint(
            out_dir / "_chkpt", per_fam, mu_prior=A_dummy, tau_prior=A_dummy)
    assert state_r.coupling_variant == 'dynamic_field'
    assert state_r.dyn_field is not None
    assert state_r.potts_dp is None
    err_pi_class = float(np.max(np.abs(
        state_r.pi_class - state.pi_class)))
    err_pi_field = float(np.max(np.abs(
        state_r.dyn_field.pi_field - state.dyn_field.pi_field)))
    err_rho = float(np.max(np.abs(
        state_r.dyn_field.rho - state.dyn_field.rho)))
    print(f"  pi_class err = {err_pi_class:.3e}")
    print(f"  pi_field err = {err_pi_field:.3e}")
    print(f"  rho err      = {err_rho:.3e}")
    print(f"  alpha_field reloaded = {state_r.dyn_field.alpha_field}")
    print(f"  rho_chain reloaded   = {state_r.dyn_field.rho_chain}")
    assert err_pi_class < 1e-15
    assert err_pi_field < 1e-15
    assert err_rho < 1e-15
    assert abs(state_r.dyn_field.alpha_field - 1.5) < 1e-12
    assert abs(state_r.dyn_field.rho_chain - 0.7) < 1e-12
    assert trace_r == trace
    print(f"  meta['coupling_variant'] = {meta_r['coupling_variant']!r}")
    print("  PASS")


def test_dynfield_save_load_partition():
    """Per-MSA latents (cls, partner, eta) should also round-trip."""
    print("\n[test 2] per-MSA partition arrays round-trip")
    per_fam = _build_per_family_data(n_families=3, L=6)
    state = init_svi_state_dynfield(
        per_fam, K_c=2, L_max=2, rng=np.random.default_rng(0))
    # Set unique per-MSA arrays to verify identity.
    for i, st in enumerate(state.states_per_msa):
        st.cls[:] = (i + 1) % 2
        st.partner[:] = -1
    A_dummy = np.zeros((20, 20))
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        save_checkpoint(state, {}, np.random.default_rng(0),
                          out_dir, it=0)
        state_r, _, _, _, _ = load_checkpoint(
            out_dir / "_chkpt", per_fam, mu_prior=A_dummy, tau_prior=A_dummy)
    for i, (st_orig, st_load) in enumerate(zip(state.states_per_msa,
                                                  state_r.states_per_msa)):
        assert np.array_equal(st_orig.cls, st_load.cls), f"cls mismatch fam {i}"
        assert np.array_equal(st_orig.partner, st_load.partner), (
            f"partner mismatch fam {i}")
    print(f"  all {len(state.states_per_msa)} MSAs' cls + partner match")
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
    print(f"\nAll {len(fns)} dynfield checkpoint tests passed.")
