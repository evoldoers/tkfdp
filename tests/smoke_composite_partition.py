"""Verification of src/tkfdp/composite_partition.py.

V.1 — Trivial M=1 closed form. With all M_obs(e) = 1, Z_E reduces to the
      matching polynomial of the complete graph, which has the closed
      form sum_{k=0..floor(N_M/2)} C(N_M, 2k) (2k-1)!! eps^k.
      Check: AIS estimate matches closed form to <5% relative error.
V.2 — Brute-force enumeration at small N_M (<= 14). Compare AIS to
      explicit enumeration of all matchings using non-trivial M_obs.
V.3 — alpha_z -> infinity. Check that AIS estimate -> 0 (log Z_E -> 0)
      as alpha_z gets very large (single-edge contribution suppressed
      by 1/alpha_z).
V.4 — MSA roundtrip: build a synthetic MSA from a known path, verify
      msa_pair_to_path inverts _msa_from_col_assignments correctly.

Run:
    python tests/smoke_composite_partition.py
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
TKFMIXDOM_ROOT = Path.home() / "tkf-mixdom" / "python"
sys.path.insert(0, str(TKFMIXDOM_ROOT))
sys.path.insert(0, str(HERE))

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.params import M as M_STATE

from tkfdp.lg08 import PI_LG08
from tkfdp.potts_dp import PottsDPState, canonical_pair_idx_table
from tkfdp.coupled_annealing import build_boost_state
from tkfdp.mcmc_infinite_phmm import (
    precompute_partial_forward, _initial_alignment, _match_cells_of,
    _log_M_obs,
)
from tkfdp.composite_partition import (
    estimate_log_Z_E,
    log_Z_E_closed_form_M1,
    log_Z_E_brute_force,
    msa_pair_to_path,
    composite_loglik_cherry,
)


# ----------------------------------------------------------------------------
# Test scaffolding (stripped from smoke_mcmc_infinite_phmm.py).
# ----------------------------------------------------------------------------

@dataclass
class FakeSVIState:
    K_c: int
    A: int
    pi_class: np.ndarray
    potts_dp: object


def make_state(K_c=1, with_h=False, H_scale=0.0, seed=0):
    A_local = 20
    rng = np.random.default_rng(seed)
    pi_class = np.tile(np.asarray(PI_LG08), (K_c, 1))
    n_pairs = K_c * (K_c + 1) // 2
    atoms = rng.standard_normal((n_pairs, A_local, A_local)) * H_scale
    atoms = 0.5 * (atoms + np.transpose(atoms, (0, 2, 1)))
    cp_idx, _ = canonical_pair_idx_table(K_c)
    assignments = np.asarray(cp_idx, dtype=np.int64)
    counts = np.ones(n_pairs, dtype=np.int64)
    h_pairs = np.zeros((n_pairs, 2, A_local)) if with_h else None
    pdp = PottsDPState(
        K_c=K_c, A=A_local, atoms=atoms, assignments=assignments,
        counts=counts, alpha_H=1.0, h_pairs=h_pairs,
    )
    return FakeSVIState(K_c=K_c, A=A_local,
                        pi_class=pi_class, potts_dp=pdp)


def make_test_pair(L=12, seed=0, dy=2):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 20, L).astype(np.int32)
    y = rng.integers(0, 20, L + dy).astype(np.int32)
    return x, y


def make_boost_state_for_pair(state, x_seq, y_seq, t):
    pair_post = {(0, 1): np.zeros((x_seq.shape[0], y_seq.shape[0]))}
    pair_taus = {(0, 1): float(t)}
    seqs_int = [x_seq, y_seq]
    bs = build_boost_state(pair_post, pair_taus, seqs_int, state)
    return bs[(0, 1)]


def make_setup(L=8, t=0.4, alpha_z=10.0, H_scale=0.0, seed=0):
    Q_lg, pi_lg = rate_matrix_lg()
    x, y = make_test_pair(L=L, seed=seed)
    state = make_state(K_c=1, with_h=False, H_scale=H_scale, seed=seed)
    bs = make_boost_state_for_pair(state, x, y, t)
    setup = precompute_partial_forward(
        x, y, t, 0.02, 0.05, 0.5, Q_lg, pi_lg, bs, alpha_z=alpha_z)
    return setup, x, y


def init_path_and_match_cells(setup, seed=0):
    rng_key = jax.random.PRNGKey(seed)
    A_obs = _initial_alignment(rng_key, setup, init_mode="viterbi")
    cells = _match_cells_of(A_obs)
    return A_obs, cells


# ----------------------------------------------------------------------------
# V.1 — closed form check (M = 1).
# ----------------------------------------------------------------------------

def test_V1_closed_form_M1():
    print("\n=== V.1: M=1 closed form vs AIS ===")
    # Force M_obs = 1 (log = 0) explicitly, regardless of how the boost-state
    # builder normalises the per-AA ratios at H=0. This isolates the AIS
    # algorithm from any boost-state convention details.
    setup, x, y = make_setup(L=10, t=0.3, alpha_z=20.0, H_scale=0.0, seed=42)
    setup.M_obs = np.zeros_like(setup.M_obs)

    A_obs, cells = init_path_and_match_cells(setup)
    N_M = len(cells)
    print(f"  N_match = {N_M}, alpha_z = {setup.alpha_z}")
    # Closed form.
    log_Z_cf = log_Z_E_closed_form_M1(N_M, setup.alpha_z)
    print(f"  closed-form log Z_E = {log_Z_cf:.6f}")
    # AIS estimate (need substantial compute at N_M=10 with M=1).
    log_Z_ais, diag = estimate_log_Z_E(
        setup, A_obs, alpha_z=setup.alpha_z,
        n_ais_steps=80, n_inner_sweeps=200, n_chains=64, seed=7)
    per_chain = np.asarray(diag.log_Z_per_chain)
    print(f"  AIS log Z_E       = {log_Z_ais:.6f}  "
          f"(per-chain stdev = {per_chain.std():.4f}, "
          f"runtime = {diag.runtime_seconds:.1f}s)")
    print(f"  per-chain log Z_E: {per_chain}")
    rel_err = abs(np.exp(log_Z_ais - log_Z_cf) - 1.0)
    print(f"  rel err on Z_E = {rel_err:.4f}")
    assert rel_err < 0.05, f"rel err {rel_err} > 0.05"
    print("  PASS")


# ----------------------------------------------------------------------------
# V.2 — brute-force enumeration check.
# ----------------------------------------------------------------------------

def test_V2_brute_force():
    print("\n=== V.2: brute-force enumeration vs AIS ===")
    # Use a small alignment so N_M is small enough to enumerate all
    # matchings. With nontrivial H, M_obs has nontrivial structure.
    # Choose alpha_z=20 (mild) so equilibrium |E| is small and AIS
    # converges quickly.
    setup, x, y = make_setup(L=8, t=0.5, alpha_z=20.0, H_scale=0.5, seed=23)
    A_obs, cells = init_path_and_match_cells(setup)
    N_M = len(cells)
    print(f"  N_match = {N_M}, alpha_z = {setup.alpha_z}")
    if N_M > 14:
        print(f"  N_M = {N_M} > 14; skipping brute force (too expensive).")
        return
    # Brute-force.
    log_M_lookup = lambda a, b: _log_M_obs(setup, a, b)
    t0 = time.time()
    log_Z_bf = log_Z_E_brute_force(cells, log_M_lookup, setup.alpha_z)
    bf_time = time.time() - t0
    print(f"  brute-force log Z_E = {log_Z_bf:.6f}  ({bf_time:.2f}s)")
    # AIS estimate.
    log_Z_ais, diag = estimate_log_Z_E(
        setup, A_obs, alpha_z=setup.alpha_z,
        n_ais_steps=40, n_inner_sweeps=100, n_chains=32, seed=11)
    print(f"  AIS log Z_E         = {log_Z_ais:.6f}  "
          f"(per-chain stdev = {np.std(diag.log_Z_per_chain):.4f}, "
          f"runtime = {diag.runtime_seconds:.1f}s)")
    rel_err = abs(np.exp(log_Z_ais - log_Z_bf) - 1.0)
    print(f"  rel err on Z_E = {rel_err:.4f}")
    assert rel_err < 0.10, f"rel err {rel_err} > 0.10"
    print("  PASS")


def test_V2b_brute_force_strong_coupling():
    print("\n=== V.2b: brute force with strong Potts coupling ===")
    setup, x, y = make_setup(L=7, t=0.4, alpha_z=10.0, H_scale=2.0, seed=5)
    A_obs, cells = init_path_and_match_cells(setup)
    N_M = len(cells)
    print(f"  N_match = {N_M}, alpha_z = {setup.alpha_z}, H_scale=2.0")
    if N_M > 12:
        print(f"  N_M = {N_M} > 12; skipping (too expensive).")
        return
    log_M_lookup = lambda a, b: _log_M_obs(setup, a, b)
    log_Z_bf = log_Z_E_brute_force(cells, log_M_lookup, setup.alpha_z)
    log_Z_ais, diag = estimate_log_Z_E(
        setup, A_obs, alpha_z=setup.alpha_z,
        n_ais_steps=40, n_inner_sweeps=200, n_chains=32, seed=42)
    print(f"  brute = {log_Z_bf:.4f}   AIS = {log_Z_ais:.4f}   "
          f"diff = {log_Z_ais - log_Z_bf:+.4f}")
    rel_err = abs(np.exp(log_Z_ais - log_Z_bf) - 1.0)
    print(f"  rel err = {rel_err:.4f}")
    assert rel_err < 0.10, f"strong-coupling rel err {rel_err} > 0.10"
    print("  PASS")


# ----------------------------------------------------------------------------
# V.3 — alpha_z -> infinity gives log Z_E -> 0.
# ----------------------------------------------------------------------------

def test_V3_alpha_z_inf_limit():
    print("\n=== V.3: alpha_z -> infinity gives log Z_E -> 0 ===")
    setup, x, y = make_setup(L=10, t=0.4, alpha_z=1e10, H_scale=0.5, seed=11)
    A_obs, cells = init_path_and_match_cells(setup)
    N_M = len(cells)
    print(f"  N_match = {N_M}, alpha_z = 1e10")
    log_Z_ais, diag = estimate_log_Z_E(
        setup, A_obs, alpha_z=setup.alpha_z,
        alpha_z_init=1e12,  # very mild ratio: 1e12 -> 1e10 = 100x
        n_ais_steps=10, n_inner_sweeps=50, n_chains=4, seed=3)
    # First-order term: N_M choose 2 * (1e-10) * mean(M_obs).
    # Tiny: log Z_E should be ~0.
    print(f"  AIS log Z_E = {log_Z_ais:.3e}")
    assert abs(log_Z_ais) < 1e-3, f"|log Z_E| = {abs(log_Z_ais)} > 1e-3"
    print("  PASS")


# ----------------------------------------------------------------------------
# V.4 — MSA -> path roundtrip.
# ----------------------------------------------------------------------------

def test_V4_msa_roundtrip():
    print("\n=== V.4: MSA <-> path roundtrip ===")
    # Build a known path, encode as MSA rows, decode back, compare.
    # Path: M(1,1), I(1,2), D(2,2), M(3,3), M(4,4)
    M = int(M_STATE)
    I_st = 2; D_st = 3
    path_truth = [(M, 1, 1), (I_st, 1, 2), (D_st, 2, 2),
                  (M, 3, 3), (M, 4, 4)]
    # Sequences (any AAs):
    x_seq = np.array([5, 6, 7, 8], dtype=np.int32)  # Lx=4
    y_seq = np.array([10, 11, 12, 13], dtype=np.int32)  # Ly=4
    # Encode as rows. column 0: M => x[0],y[0]. col 1: I => -1, y[1].
    # col 2: D => x[1], -1. col 3: M => x[2],y[2]. col 4: M => x[3],y[3].
    row_x = np.array([5, -1, 6, 7, 8], dtype=np.int32)
    row_y = np.array([10, 11, -1, 12, 13], dtype=np.int32)
    path_dec = msa_pair_to_path(row_x, row_y)
    print(f"  truth: {path_truth}")
    print(f"  decoded: {path_dec}")
    assert path_dec == path_truth, f"path mismatch"
    print("  PASS")


# ----------------------------------------------------------------------------
# V.5 — composite_loglik_cherry end-to-end.
# ----------------------------------------------------------------------------

def test_V5_composite_loglik_cherry():
    print("\n=== V.5: composite_loglik_cherry end-to-end ===")
    Q_lg, pi_lg = rate_matrix_lg()
    state = make_state(K_c=1, with_h=False, H_scale=0.5, seed=0)
    x, y = make_test_pair(L=8, seed=1)
    t = 0.4
    bs = make_boost_state_for_pair(state, x, y, t)
    # Build a "MSA pair" from a Viterbi alignment of the baseline.
    setup = precompute_partial_forward(
        x, y, t, 0.02, 0.05, 0.5, Q_lg, pi_lg, bs, alpha_z=10.0)
    A_obs, cells = init_path_and_match_cells(setup)
    # Convert path to MSA rows.
    Lx, Ly = setup.Lx, setup.Ly
    L_aln = len(A_obs)
    row_x = np.full(L_aln, -1, dtype=np.int32)
    row_y = np.full(L_aln, -1, dtype=np.int32)
    M_st = int(M_STATE); I_st = 2; D_st = 3
    for k, (st, i, j) in enumerate(A_obs):
        if st == M_st:
            row_x[k] = int(x[i - 1]); row_y[k] = int(y[j - 1])
        elif st == I_st:
            row_y[k] = int(y[j - 1])
        elif st == D_st:
            row_x[k] = int(x[i - 1])
    # Now run composite_loglik_cherry.
    log_pi, log_Z_E, log_p, diag, _ = composite_loglik_cherry(
        msa_row_x=row_x, msa_row_y=row_y,
        x_seq=x, y_seq=y, t=t,
        Q_lg=Q_lg, pi_lg=pi_lg, boost_state=bs,
        alpha_z=10.0,
        n_ais_steps=20, n_inner_sweeps=40, n_chains=4, seed=99,
    )
    print(f"  log_pi_TKF92 = {log_pi:.4f}")
    print(f"  log_Z_E      = {log_Z_E:.4f}")
    print(f"  log_p_A      = {log_p:.4f}")
    assert log_pi < 0
    assert log_Z_E >= 0  # Z_E >= 1 always (empty matching contributes 1).
    print("  PASS")


def main():
    print("Running composite_partition.py verification suite\n")
    t0 = time.time()
    test_V4_msa_roundtrip()
    test_V3_alpha_z_inf_limit()
    test_V1_closed_form_M1()
    test_V2_brute_force()
    test_V2b_brute_force_strong_coupling()
    test_V5_composite_loglik_cherry()
    print(f"\nAll tests passed in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
