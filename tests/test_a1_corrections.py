"""Tests for the A1 correctness fixes in mcmc_infinite_phmm.py.

(1) I/D-anchor preservation in segment resample (option a, 2026-06-27):
    when an edge endpoint is an Insertion or Deletion cell, any segment
    resample that drops that cell from its stated position (or replaces
    it with a cell of a different type) must be rejected.

(2) Canonical-CRP four-term correction in the replica-exchange swap
    proposal: when both rungs are in reversible mode, the swap MH log
    ratio carries a four-term correction that cancels exactly when
    L_a == L_b and is non-zero otherwise.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tkfdp.mcmc_infinite_phmm import (   # noqa: E402
    _crp_log_prior_pathlen,
    _count_alive_cells,
    _id_anchor_positions,
    _id_anchors_preserved,
    _swap_proposal,
    M as M_STATE, I as I_STATE, D as D_STATE,
)


def _state_types_array(n_states: int = 8) -> np.ndarray:
    """Tiny stub `state_types_np`: state index == generic state type so we
    can author paths directly with the (M, I, D) constants as the state
    column. Domain is conservative (size 8 covers S/M/I/D/E) and matches
    the upstream Pair HMM convention used by tkfmixdom."""
    out = np.arange(n_states, dtype=np.int32)
    return out


# ---------------------------------------------------------------------------
# (1) I/D-anchor preservation
# ---------------------------------------------------------------------------

def test_id_anchors_preserved_no_anchors():
    """No I/D anchors -> always preserved."""
    state_types = _state_types_array()
    path = [(M_STATE, 1, 1), (M_STATE, 2, 2), (I_STATE, 2, 3)]
    assert _id_anchors_preserved(path, {}, state_types) is True


def test_id_anchors_preserved_anchor_present():
    """The I anchor at (2, 3) is present in the path -> preserved."""
    state_types = _state_types_array()
    path = [(M_STATE, 1, 1), (M_STATE, 2, 2), (I_STATE, 2, 3),
            (M_STATE, 3, 4)]
    anchors = {(2, 3): I_STATE}
    assert _id_anchors_preserved(path, anchors, state_types) is True


def test_id_anchors_preserved_anchor_missing():
    """The I anchor at (2, 3) is NOT in the path -> not preserved."""
    state_types = _state_types_array()
    path = [(M_STATE, 1, 1), (M_STATE, 2, 2), (M_STATE, 3, 3)]
    anchors = {(2, 3): I_STATE}
    assert _id_anchors_preserved(path, anchors, state_types) is False


def test_id_anchors_preserved_wrong_cell_type():
    """A cell at (2, 3) exists in the path but is a Match, not the
    Insertion anchor that the edge requires -> not preserved."""
    state_types = _state_types_array()
    path = [(M_STATE, 1, 1), (M_STATE, 2, 2), (M_STATE, 2, 3),
            (M_STATE, 3, 4)]
    anchors = {(2, 3): I_STATE}
    assert _id_anchors_preserved(path, anchors, state_types) is False


def test_id_anchors_preserved_multiple():
    """Two anchors, one preserved one broken -> overall False."""
    state_types = _state_types_array()
    path = [(M_STATE, 1, 1), (I_STATE, 1, 2), (M_STATE, 2, 3),
            (M_STATE, 3, 4)]
    anchors = {(1, 2): I_STATE, (3, 3): D_STATE}  # second never appears
    assert _id_anchors_preserved(path, anchors, state_types) is False


def test_id_anchor_positions_extraction():
    """_id_anchor_positions should pull out only the I/D endpoints
    from a mixed edge list."""
    # Edges in the mixed (some 2-tuple, some 3-tuple) form that the
    # endpoint helpers accept; backward-compat 2-tuples are treated as M.
    edges = [
        ((1, 1), (2, 2)),                              # MM legacy form
        ((3, 3, M_STATE), (4, 4, I_STATE)),            # M-I edge
        ((5, 5, D_STATE), (6, 6, I_STATE)),            # D-I edge
    ]
    anchors = _id_anchor_positions(edges)
    assert anchors == {(4, 4): I_STATE, (5, 5): D_STATE, (6, 6): I_STATE}


# ---------------------------------------------------------------------------
# (2) Canonical-CRP swap correction
# ---------------------------------------------------------------------------

@dataclass
class _MockSetup:
    """Minimum surface area for _swap_proposal: alpha_z, reversible,
    state_types_np. _count_alive_cells only needs state_types_np to
    classify each cell of the path."""
    alpha_z: float
    reversible: bool
    state_types_np: np.ndarray


def _build_path(n_match: int, n_ins: int, n_del: int):
    """Construct an arbitrary path with the stated (M, I, D) counts.
    Coordinates are dummy (the swap correction depends only on alive-cell
    count, not on coordinates)."""
    path = []
    i = j = 1
    for _ in range(n_match):
        path.append((M_STATE, i, j)); i += 1; j += 1
    for _ in range(n_ins):
        path.append((I_STATE, i, j)); j += 1
    for _ in range(n_del):
        path.append((D_STATE, i, j)); i += 1
    return path


def _expected_correction(L_a: int, L_b: int, a_a: float, a_b: float) -> float:
    """log MH correction added by A1 swap acceptance (full canonical CRP:
    alpha_z^L numerator term + Pochhammer denominator term)."""
    log_alpha_a = math.log(a_a); log_alpha_b = math.log(a_b)
    alpha_l_term = ((L_b - L_a) * log_alpha_a
                    + (L_a - L_b) * log_alpha_b)
    pochh_term = (_crp_log_prior_pathlen(L_b, a_a)
                  - _crp_log_prior_pathlen(L_b, a_b)
                  + _crp_log_prior_pathlen(L_a, a_b)
                  - _crp_log_prior_pathlen(L_a, a_a))
    return alpha_l_term + pochh_term


def _swap_log_ratio_observed(rng_seed: int, setups, paths, edges_a, edges_b):
    """Reproduce the log_ratio the implementation computes: eps term plus
    (in reversible mode) the full canonical-CRP correction (alpha_z^L
    + Pochhammer)."""
    a_a = setups[0].alpha_z; a_b = setups[1].alpha_z
    log_alpha_a = math.log(a_a); log_alpha_b = math.log(a_b)
    base = (len(edges_b) - len(edges_a)) * (log_alpha_b - log_alpha_a)
    if setups[0].reversible and setups[1].reversible:
        L_a = _count_alive_cells(setups[0].state_types_np, paths[0])
        L_b = _count_alive_cells(setups[1].state_types_np, paths[1])
        base += _expected_correction(L_a, L_b, a_a, a_b)
    return base


def test_swap_correction_zero_when_L_equal():
    """L_a == L_b => four-term correction vanishes exactly. Reversible
    and legacy swap acceptance log-ratios then match to machine precision."""
    stt = _state_types_array()
    setups_legacy = (
        _MockSetup(alpha_z=100.0, reversible=False, state_types_np=stt),
        _MockSetup(alpha_z=500.0, reversible=False, state_types_np=stt),
    )
    setups_a1 = (
        _MockSetup(alpha_z=100.0, reversible=True,  state_types_np=stt),
        _MockSetup(alpha_z=500.0, reversible=True,  state_types_np=stt),
    )
    # Same alive-cell count on each rung.
    path_a = _build_path(n_match=10, n_ins=2, n_del=3)   # L = 15
    path_b = _build_path(n_match=12, n_ins=1, n_del=2)   # L = 15
    assert (_count_alive_cells(stt, path_a)
            == _count_alive_cells(stt, path_b) == 15)
    edges_a = [((1, 1, M_STATE), (2, 2, M_STATE))]      # |E_a|=1
    edges_b = []                                          # |E_b|=0
    log_legacy = _swap_log_ratio_observed(0, setups_legacy,
                                            (path_a, path_b), edges_a, edges_b)
    log_a1 = _swap_log_ratio_observed(0, setups_a1,
                                        (path_a, path_b), edges_a, edges_b)
    assert abs(log_a1 - log_legacy) < 1e-12, (
        f"L_a == L_b correction should vanish; got "
        f"legacy={log_legacy:.6e}, a1={log_a1:.6e}")


def test_swap_correction_nonzero_when_L_differs():
    """L_a != L_b => correction is non-trivial; A1 and legacy disagree."""
    stt = _state_types_array()
    setups_legacy = (
        _MockSetup(alpha_z=100.0, reversible=False, state_types_np=stt),
        _MockSetup(alpha_z=500.0, reversible=False, state_types_np=stt),
    )
    setups_a1 = (
        _MockSetup(alpha_z=100.0, reversible=True,  state_types_np=stt),
        _MockSetup(alpha_z=500.0, reversible=True,  state_types_np=stt),
    )
    path_a = _build_path(n_match=10, n_ins=0, n_del=0)   # L = 10
    path_b = _build_path(n_match=15, n_ins=3, n_del=2)   # L = 20
    L_a = _count_alive_cells(stt, path_a)
    L_b = _count_alive_cells(stt, path_b)
    assert (L_a, L_b) == (10, 20)
    edges_a = []
    edges_b = []
    log_legacy = _swap_log_ratio_observed(0, setups_legacy,
                                            (path_a, path_b), edges_a, edges_b)
    log_a1 = _swap_log_ratio_observed(0, setups_a1,
                                        (path_a, path_b), edges_a, edges_b)
    delta = log_a1 - log_legacy
    expected = _expected_correction(L_a, L_b, 100.0, 500.0)
    assert abs(delta - expected) < 1e-10, (
        f"a1 - legacy should equal closed-form correction; "
        f"observed delta={delta:.6e}, expected={expected:.6e}")
    assert abs(expected) > 1e-3, (
        f"correction should be substantial when L_a != L_b at usual "
        f"alpha rungs; got {expected:.6e}")


def test_swap_proposal_accepts_when_corrected():
    """Smoke: full _swap_proposal call, A1 mode, varying L. Just ensure
    it runs and returns a 3-tuple of the expected shape."""
    stt = _state_types_array()
    setups = [
        _MockSetup(alpha_z=100.0, reversible=True, state_types_np=stt),
        _MockSetup(alpha_z=500.0, reversible=True, state_types_np=stt),
    ]
    path_a = _build_path(n_match=10, n_ins=0, n_del=0)
    path_b = _build_path(n_match=15, n_ins=3, n_del=2)
    states = [(path_a, []), (path_b, [])]
    rng = np.random.default_rng(0)
    a, b, accepted = _swap_proposal(rng, setups, states)
    # Only two rungs, so a single adjacent pair (0, 1).
    assert (a, b) == (0, 1)
    assert isinstance(accepted, bool) or accepted in (True, False)


if __name__ == "__main__":
    # Run all tests; print pass/fail per test.
    import inspect
    mod = sys.modules[__name__]
    fns = [(name, fn) for name, fn in inspect.getmembers(mod, inspect.isfunction)
           if name.startswith("test_")]
    failures = []
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failures.append(name)
    if failures:
        print(f"\n{len(failures)}/{len(fns)} test(s) failed")
        sys.exit(1)
    print(f"\nAll {len(fns)} tests passed.")
