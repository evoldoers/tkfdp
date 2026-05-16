"""Smoke tests for src/tkfdp/aug_phmm_antidiag.py.

Cross-validates the antidiagonal-wavefront DP against the row-scan
reference implementation in src/tkfdp/aug_phmm.py. The two should agree
to ~1e-10 because they implement the same recursion in different
traversal orders.

Tests:
  1. Cross-validation against aug_phmm row-scan, sweeping L = 8..32
     and H_scale in {0.0, 0.2}. Max-diff < 1e-10.
  2. Padding-mask correctness (same as smoke_aug_phmm test 3).
  3. Augmented partition function consistency (alpha-end vs beta-start).
  4. Tag conservation at (0, 0).
  5. Match-cell tag-update sparsity at (1, 1).
  6. Performance benchmark: row-scan vs antidiagonal at L = 50, 100, 200
     after JIT warmup.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import os
# Force CPU for tests by default; set TKFDP_TEST_GPU=1 to use the GPU
# (the antidiagonal version is designed for GPU but must work on CPU).
if os.environ.get("TKFDP_TEST_GPU", "0") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Path setup.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
TKFMIXDOM_ROOT = Path.home() / "tkf-mixdom" / "python"
sys.path.insert(0, str(TKFMIXDOM_ROOT))
sys.path.insert(0, str(HERE))

from tkfmixdom.jax.dp.hmm import _find_e_idx
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.params import M as M_STATE, S as S_STATE

from tkfdp.aug_phmm import (
    aug_phmm_corrected_posterior,
    forward_aug_phmm, backward_aug_phmm,
    build_M_tensor_aa_marginal,
    TAG_NO_EDGE, TAG_DONE, N_TAGS, N_AB_TAGS, A,
)
from tkfdp.aug_phmm_antidiag import (
    aug_phmm_antidiag_corrected_posterior,
    forward_aug_phmm_antidiag, backward_aug_phmm_antidiag,
    _aug_forward_antidiag_jit, _aug_backward_antidiag_jit,
)
from tkfdp.coupled_annealing import build_boost_state
from tkfdp.lg08 import PI_LG08
from tkfdp.potts_dp import PottsDPState, canonical_pair_idx_table


# ============================================================================
# Helpers (mirrors smoke_aug_phmm.py)
# ============================================================================

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


def make_test_pair(L=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 20, L).astype(np.int32)
    y = rng.integers(0, 20, L + 2).astype(np.int32)
    return x, y


def make_boost_state_for_pair(state, x_seq, y_seq, t):
    pair_post = {(0, 1): np.zeros((x_seq.shape[0], y_seq.shape[0]))}
    pair_taus = {(0, 1): float(t)}
    seqs_int = [x_seq, y_seq]
    bs = build_boost_state(pair_post, pair_taus, seqs_int, state)
    return bs[(0, 1)]


# ============================================================================
# Test 1: Cross-validation against row-scan
# ============================================================================

def test1_xval_against_rowscan():
    print("\n=== Test 1: cross-validation vs row-scan aug_phmm ===")
    Q_lg, pi_lg = rate_matrix_lg()
    t = 0.4

    max_diff_overall = 0.0
    max_logL_diff_overall = 0.0
    for L in [8, 10, 12, 16, 24, 32]:
        for H_scale in [0.0, 0.2]:
            x, y = make_test_pair(L=L, seed=L * 7 + int(H_scale * 100))
            state = make_state(K_c=1, with_h=False, H_scale=H_scale, seed=L)
            bs = make_boost_state_for_pair(state, x, y, t)

            Q_row, L_row, Q_base_row, log_F0_row = aug_phmm_corrected_posterior(
                x, y, t, 0.02, 0.05, 0.5, Q_lg, pi_lg, bs,
                alpha_z=100.0, q_min=0.0)
            Q_diag, L_diag, Q_base_diag, log_F0_diag = (
                aug_phmm_antidiag_corrected_posterior(
                    x, y, t, 0.02, 0.05, 0.5, Q_lg, pi_lg, bs,
                    alpha_z=100.0, q_min=0.0))

            diff_Q = float(np.max(np.abs(Q_row - Q_diag)))
            diff_baseline = float(np.max(np.abs(Q_base_row - Q_base_diag)))
            diff_logL = abs(np.log(max(L_row, 1e-300))
                            - np.log(max(L_diag, 1e-300)))
            diff_logF0 = abs(log_F0_row - log_F0_diag)
            print(f"  L={L:2d}, H={H_scale:.1f}: "
                  f"max|Q'_row - Q'_diag|={diff_Q:.3e}, "
                  f"baseline diff={diff_baseline:.3e}, "
                  f"log L diff={diff_logL:.3e}, "
                  f"log F0 diff={diff_logF0:.3e}")
            max_diff_overall = max(max_diff_overall, diff_Q)
            max_logL_diff_overall = max(max_logL_diff_overall, diff_logL)
            assert diff_Q < 1e-10, \
                f"Q' mismatch (L={L}, H={H_scale}): {diff_Q:.3e}"
            assert diff_baseline < 1e-12, \
                f"Q baseline mismatch (L={L}, H={H_scale}): {diff_baseline:.3e}"
            assert diff_logL < 1e-10, \
                f"log L_exact mismatch (L={L}, H={H_scale}): {diff_logL:.3e}"
    print(f"  Worst Q' diff overall: {max_diff_overall:.3e}")
    print(f"  Worst log L_exact diff overall: {max_logL_diff_overall:.3e}")
    print("  PASS")


# ============================================================================
# Test 2: Padding-mask correctness
# ============================================================================

def test2_padding_mask():
    print("\n=== Test 2: padding-mask correctness ===")
    Q_lg, pi_lg = rate_matrix_lg()
    t = 0.4
    rng = np.random.default_rng(3)
    L = 10
    x = rng.integers(0, 20, L).astype(np.int32)
    y = rng.integers(0, 20, L + 2).astype(np.int32)
    Lx, Ly = x.shape[0], y.shape[0]
    state = make_state(K_c=1, with_h=False, H_scale=0.2, seed=3)
    bs = make_boost_state_for_pair(state, x, y, t)

    Q_unp, L_unp, _, _ = aug_phmm_antidiag_corrected_posterior(
        x, y, t, 0.02, 0.05, 0.5, Q_lg, pi_lg, bs,
        alpha_z=100.0, q_min=0.0)

    # Pass a padded sequence (extra junk at end) but real_Lx/Ly = original.
    log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
        0.02, 0.05, t, 0.5, Q_lg, pi_lg)
    M_tensor_np = build_M_tensor_aa_marginal(bs)
    log_M_tensor = jnp.log(jnp.clip(jnp.asarray(M_tensor_np), 1e-300, None))
    log_eps_j = jnp.asarray(np.log(1.0 / 100.0))

    Lx_pad_test = 16
    Ly_pad_test = 16
    if Lx >= Lx_pad_test or Ly >= Ly_pad_test:
        print("  SKIP (sizes too small)")
        return
    x_padded = np.zeros(Lx_pad_test, dtype=np.int32)
    x_padded[:Lx] = x
    x_padded[Lx:] = rng.integers(0, 20, Lx_pad_test - Lx)
    y_padded = np.zeros(Ly_pad_test, dtype=np.int32)
    y_padded[:Ly] = y
    y_padded[Ly:] = rng.integers(0, 20, Ly_pad_test - Ly)

    alpha_pad, _, _ = forward_aug_phmm_antidiag(
        log_trans, state_types, sub_matrix, pi_out,
        jnp.asarray(x_padded), jnp.asarray(y_padded),
        jnp.asarray(Lx), jnp.asarray(Ly),
        log_M_tensor, log_eps_j)

    e_idx = _find_e_idx(state_types)
    end_alpha = alpha_pad[Lx, Ly, :, :]
    end_no_edge = end_alpha[:, TAG_NO_EDGE] + log_trans[:, e_idx]
    end_done = end_alpha[:, TAG_DONE] + log_trans[:, e_idx]
    log_L_pad = float(jax.nn.logsumexp(jnp.concatenate([end_no_edge, end_done])))

    diff_logL = abs(log_L_pad - float(np.log(L_unp)))
    print(f"  log L padded vs unpadded: diff={diff_logL:.3e}")
    assert diff_logL < 1e-6, f"log L mismatch: {diff_logL:.3e}"
    print("  PASS")


# ============================================================================
# Test 3: Augmented partition function consistency
# ============================================================================

def test3_partition_consistency():
    print("\n=== Test 3: augmented partition function consistency ===")
    Q_lg, pi_lg = rate_matrix_lg()
    t = 0.4
    L = 10
    x, y = make_test_pair(L=L, seed=4)
    Lx, Ly = x.shape[0], y.shape[0]
    state = make_state(K_c=1, with_h=False, H_scale=0.3, seed=4)
    bs = make_boost_state_for_pair(state, x, y, t)

    log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
        0.02, 0.05, t, 0.5, Q_lg, pi_lg)
    M_tensor_np = build_M_tensor_aa_marginal(bs)
    log_M_tensor = jnp.log(jnp.clip(jnp.asarray(M_tensor_np), 1e-300, None))
    log_eps_j = jnp.asarray(np.log(1.0 / 100.0))

    alpha, _, _ = forward_aug_phmm_antidiag(
        log_trans, state_types, sub_matrix, pi_out,
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(Lx), jnp.asarray(Ly),
        log_M_tensor, log_eps_j)
    beta = backward_aug_phmm_antidiag(
        log_trans, state_types, sub_matrix, pi_out,
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(Lx), jnp.asarray(Ly),
        log_M_tensor, log_eps_j)

    # L_exact via alpha at end
    e_idx = _find_e_idx(state_types)
    end_alpha = alpha[Lx, Ly, :, :]
    end_no_edge = end_alpha[:, TAG_NO_EDGE] + log_trans[:, e_idx]
    end_done = end_alpha[:, TAG_DONE] + log_trans[:, e_idx]
    log_L_alpha = float(jax.nn.logsumexp(jnp.concatenate([end_no_edge, end_done])))

    # L_exact via beta at start: beta[0, 0, S, no_edge]
    log_L_beta = float(beta[0, 0, S_STATE, TAG_NO_EDGE])
    diff = abs(log_L_alpha - log_L_beta)
    print(f"  log L_exact (alpha end):  {log_L_alpha:.6f}")
    print(f"  log L_exact (beta start): {log_L_beta:.6f}")
    print(f"  diff: {diff:.3e}")
    assert diff < 1e-7, f"L_exact mismatch: {diff:.3e}"

    # Also cross-check the row-scan vs antidiag alpha and beta tables
    # are pointwise identical at all in-bounds cells (forward) and across
    # the whole padded grid (backward).
    alpha_row, _, _ = forward_aug_phmm(
        log_trans, state_types, sub_matrix, pi_out,
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(Lx), jnp.asarray(Ly),
        log_M_tensor, log_eps_j)
    beta_row = backward_aug_phmm(
        log_trans, state_types, sub_matrix, pi_out,
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(Lx), jnp.asarray(Ly),
        log_M_tensor, log_eps_j)
    a_in = np.asarray(alpha)[:Lx + 1, :Ly + 1]
    a_row_in = np.asarray(alpha_row)[:Lx + 1, :Ly + 1]
    b_in = np.asarray(beta)[:Lx + 1, :Ly + 1]
    b_row_in = np.asarray(beta_row)[:Lx + 1, :Ly + 1]
    # Compare only finite entries (NEG_INF cells may differ in sentinel
    # values across implementations and don't carry probability mass).
    finite_a = np.isfinite(a_row_in) & (a_row_in > -1e20)
    finite_b = np.isfinite(b_row_in) & (b_row_in > -1e20)
    max_diff_alpha = float(np.max(np.abs(a_in[finite_a] - a_row_in[finite_a])))
    max_diff_beta = float(np.max(np.abs(b_in[finite_b] - b_row_in[finite_b])))
    print(f"  max |alpha_diag - alpha_row| (finite): {max_diff_alpha:.3e}")
    print(f"  max |beta_diag  - beta_row|  (finite): {max_diff_beta:.3e}")
    assert max_diff_alpha < 1e-9, f"alpha mismatch: {max_diff_alpha}"
    assert max_diff_beta < 1e-9, f"beta mismatch: {max_diff_beta}"
    print("  PASS")


# ============================================================================
# Test 4: Tag conservation (initial state)
# ============================================================================

def test4_tag_conservation():
    print("\n=== Test 4: tag conservation at (0, 0) ===")
    Q_lg, pi_lg = rate_matrix_lg()
    t = 0.4
    L = 8
    x, y = make_test_pair(L=L, seed=5)
    Lx, Ly = x.shape[0], y.shape[0]
    state = make_state(K_c=1, with_h=False, H_scale=0.3, seed=5)
    bs = make_boost_state_for_pair(state, x, y, t)

    log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
        0.02, 0.05, t, 0.5, Q_lg, pi_lg)
    M_tensor_np = build_M_tensor_aa_marginal(bs)
    log_M_tensor = jnp.log(jnp.clip(jnp.asarray(M_tensor_np), 1e-300, None))
    log_eps_j = jnp.asarray(np.log(1.0 / 100.0))

    alpha, _, _ = forward_aug_phmm_antidiag(
        log_trans, state_types, sub_matrix, pi_out,
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(Lx), jnp.asarray(Ly),
        log_M_tensor, log_eps_j)
    cell00 = np.asarray(alpha[0, 0, :, :])
    print(f"  alpha[0, 0, S, no_edge] = {cell00[S_STATE, TAG_NO_EDGE]:.3f}")
    assert cell00[S_STATE, TAG_NO_EDGE] == 0.0
    mask_zero = np.ones_like(cell00, dtype=bool)
    mask_zero[S_STATE, TAG_NO_EDGE] = False
    max_other = float(np.max(cell00[mask_zero]))
    print(f"  max alpha[0, 0, *, *] for other (state, tag): {max_other:.3e}")
    assert max_other < -1e10
    print("  PASS")


# ============================================================================
# Test 5: Match-cell tag updates
# ============================================================================

def test5_match_cell_tag_updates():
    print("\n=== Test 5: Match-cell tag-update sparsity at (1, 1) ===")
    Q_lg, pi_lg = rate_matrix_lg()
    t = 0.4
    L = 8
    x, y = make_test_pair(L=L, seed=6)
    Lx, Ly = x.shape[0], y.shape[0]
    state = make_state(K_c=1, with_h=False, H_scale=0.3, seed=6)
    bs = make_boost_state_for_pair(state, x, y, t)

    log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
        0.02, 0.05, t, 0.5, Q_lg, pi_lg)
    M_tensor_np = build_M_tensor_aa_marginal(bs)
    log_M_tensor = jnp.log(jnp.clip(jnp.asarray(M_tensor_np), 1e-300, None))
    log_eps_j = jnp.asarray(np.log(1.0 / 100.0))

    alpha, _, _ = forward_aug_phmm_antidiag(
        log_trans, state_types, sub_matrix, pi_out,
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(Lx), jnp.asarray(Ly),
        log_M_tensor, log_eps_j)
    alpha_np = np.asarray(alpha)

    cell11 = alpha_np[1, 1, M_STATE, :]
    c0 = int(x[0]); d0 = int(y[0])
    cd_tag0 = 1 + c0 * A + d0
    log_trans_np = np.asarray(log_trans)
    pi_np = np.asarray(pi_out)
    sub_np = np.asarray(sub_matrix)
    expected_no_edge = (log_trans_np[S_STATE, M_STATE]
                       + np.log(pi_np[c0]) + np.log(sub_np[c0, d0]))
    diff_no_edge = abs(cell11[TAG_NO_EDGE] - expected_no_edge)
    print(f"  alpha[1,1,M,no_edge] = {cell11[TAG_NO_EDGE]:.4f}, expected "
          f"{expected_no_edge:.4f}, diff={diff_no_edge:.3e}")
    assert diff_no_edge < 1e-10
    log_eps = np.log(1.0 / 100.0)
    expected_cd = expected_no_edge + log_eps
    diff_cd = abs(cell11[cd_tag0] - expected_cd)
    print(f"  alpha[1,1,M,(c0,d0)] = {cell11[cd_tag0]:.4f}, expected "
          f"{expected_cd:.4f}, diff={diff_cd:.3e}")
    assert diff_cd < 1e-10
    assert cell11[TAG_DONE] < -1e10
    other_ab_max = -np.inf
    for tag in range(1, 1 + N_AB_TAGS):
        if tag != cd_tag0:
            other_ab_max = max(other_ab_max, cell11[tag])
    print(f"  max alpha[1,1,M,other (a,b)] = {other_ab_max:.3e}")
    assert other_ab_max < -1e10
    print("  PASS")


# ============================================================================
# Benchmark: row-scan vs antidiagonal at L = 50, 100, 200
# ============================================================================

def benchmark_throughput():
    print("\n=== Benchmark: row-scan vs antidiagonal ===")
    Q_lg, pi_lg = rate_matrix_lg()
    t = 0.4
    state = make_state(K_c=1, with_h=False, H_scale=0.2, seed=42)

    print(f"  device: {jax.devices()}")
    print(f"  L  | row-scan (s)  | antidiag (s)  | speedup")
    print(f"  ---|---------------|---------------|--------")
    for L in [50, 100, 200]:
        x, y = make_test_pair(L=L, seed=L)
        bs = make_boost_state_for_pair(state, x, y, t)

        # Warmup (compile both pipelines).
        Q_row, _, _, _ = aug_phmm_corrected_posterior(
            x, y, t, 0.02, 0.05, 0.5, Q_lg, pi_lg, bs,
            alpha_z=100.0, q_min=0.0)
        jax.block_until_ready(jnp.asarray(Q_row))
        Q_diag, _, _, _ = aug_phmm_antidiag_corrected_posterior(
            x, y, t, 0.02, 0.05, 0.5, Q_lg, pi_lg, bs,
            alpha_z=100.0, q_min=0.0)
        jax.block_until_ready(jnp.asarray(Q_diag))

        n_iters = 5
        t0 = time.time()
        for _ in range(n_iters):
            Q_row, _, _, _ = aug_phmm_corrected_posterior(
                x, y, t, 0.02, 0.05, 0.5, Q_lg, pi_lg, bs,
                alpha_z=100.0, q_min=0.0)
        jax.block_until_ready(jnp.asarray(Q_row))
        t_row = (time.time() - t0) / n_iters

        t0 = time.time()
        for _ in range(n_iters):
            Q_diag, _, _, _ = aug_phmm_antidiag_corrected_posterior(
                x, y, t, 0.02, 0.05, 0.5, Q_lg, pi_lg, bs,
                alpha_z=100.0, q_min=0.0)
        jax.block_until_ready(jnp.asarray(Q_diag))
        t_diag = (time.time() - t0) / n_iters

        diff = float(np.max(np.abs(Q_row - Q_diag)))
        speedup = t_row / max(t_diag, 1e-9)
        print(f"  {L:3d}| {t_row:.4f}        | {t_diag:.4f}        | "
              f"{speedup:.2f}x   (max|Q_row-Q_diag|={diff:.2e})")


# ============================================================================
# Main
# ============================================================================

def main():
    test1_xval_against_rowscan()
    test2_padding_mask()
    test3_partition_consistency()
    test4_tag_conservation()
    test5_match_cell_tag_updates()
    benchmark_throughput()
    print("\nAll antidiag aug-PHMM smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
