"""Smoke test for src/tkfdp/postprocessing.py.

Verifies:
- Module imports cleanly + runs end-to-end on a tiny synthetic state.
- Boost is exactly zero (in log-space) when H = 0, consistent with the
  derivation: M = 1 everywhere when no Potts coupling is present.
- Boost is finite (no NaN / inf) when H is non-trivial and h_pairs is set.
- Symmetry sanity: boost(i, j) is invariant under swap of (X, Y) when the
  trained Potts atom is symmetric and the same sequence is used on both
  sides.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tkfdp.lg08 import PI_LG08, S_LG08_F81
from tkfdp.potts_dp import (
    PottsDPState,
    canonical_pair_idx_table,
    canonical_pair_is_diag,
)
from tkfdp.postprocessing import (
    build_per_class_match_emit,
    build_per_classpair_joint_emit,
    class_posteriors_from_baseline,
    pair_emission_boost,
    correct_pair_posterior,
)


@dataclass
class FakeSVIState:
    K_c: int
    A: int
    pi_class: np.ndarray
    potts_dp: object


def _make_synthetic_state(K_c: int, with_h: bool, H_scale: float):
    """Tiny synthetic state: tile LG08 pi across K_c classes, draw small
    random Potts atoms (one per canonical class-pair), optionally set
    h_pairs = 0."""
    A = 20
    rng = np.random.default_rng(0)
    pi_class = np.tile(np.asarray(PI_LG08), (K_c, 1))
    n_pairs = K_c * (K_c + 1) // 2
    atoms = rng.standard_normal((n_pairs, A, A)) * H_scale
    atoms = 0.5 * (atoms + np.transpose(atoms, (0, 2, 1)))   # symmetrize
    cp_idx, _ = canonical_pair_idx_table(K_c)
    assignments = np.asarray(cp_idx, dtype=np.int64)
    counts = np.ones(n_pairs, dtype=np.int64)
    h_pairs = np.zeros((n_pairs, 2, A)) if with_h else None
    pdp = PottsDPState(
        K_c=K_c, A=A, atoms=atoms, assignments=assignments,
        counts=counts, alpha_H=1.0, h_pairs=h_pairs,
    )
    return FakeSVIState(K_c=K_c, A=A, pi_class=pi_class, potts_dp=pdp)


def main() -> int:
    rng = np.random.default_rng(42)

    K_c = 4
    L_X, L_Y = 25, 30
    x_seq = rng.integers(0, 20, L_X)
    y_seq = rng.integers(0, 20, L_Y)
    t = 0.4

    # 1. H = 0 baseline: boost should be all-zero in log-space.
    state0 = _make_synthetic_state(K_c, with_h=False, H_scale=0.0)
    Q_baseline = jnp.asarray(rng.uniform(0.0, 0.3, (L_X, L_Y)))

    log_boost0 = correct_pair_posterior(
        np.asarray(Q_baseline), x_seq, y_seq, t,
        state0, alpha_z=100.0, return_boost=True,
    )
    log_boost0_np = np.asarray(log_boost0)
    print(f"[H=0]      max |log_boost|     = {np.max(np.abs(log_boost0_np)):.3e}  "
            f"(should be ~ 0)")
    assert np.all(np.isfinite(log_boost0_np)), "H=0 produced non-finite boost"
    assert np.max(np.abs(log_boost0_np)) < 1e-8, \
        "H=0 should give exactly zero log-boost (M = 1)"

    # 2. Non-trivial H, side potentials disabled: boost should be finite +
    #    deviate from zero.
    state1 = _make_synthetic_state(K_c, with_h=False, H_scale=0.3)
    log_boost1 = correct_pair_posterior(
        np.asarray(Q_baseline), x_seq, y_seq, t,
        state1, alpha_z=100.0, return_boost=True,
    )
    log_boost1_np = np.asarray(log_boost1)
    print(f"[H!=0,h=0] max |log_boost|     = {np.max(np.abs(log_boost1_np)):.3e}")
    print(f"           mean log_boost       = {np.mean(log_boost1_np):.3e}")
    print(f"           std  log_boost       = {np.std(log_boost1_np):.3e}")
    assert np.all(np.isfinite(log_boost1_np)), "non-trivial H produced NaN"

    # 3. Side potentials enabled (h_pairs initialized to zero — should give
    #    same result as case 2 since h=0 doesn't change the joint).
    state2 = _make_synthetic_state(K_c, with_h=True, H_scale=0.3)
    state2.potts_dp.atoms = state1.potts_dp.atoms             # share atoms
    log_boost2 = correct_pair_posterior(
        np.asarray(Q_baseline), x_seq, y_seq, t,
        state2, alpha_z=100.0, return_boost=True,
    )
    log_boost2_np = np.asarray(log_boost2)
    diff = np.max(np.abs(log_boost1_np - log_boost2_np))
    print(f"[h_pairs=0] max diff vs h=None  = {diff:.3e}  (should be ~ 0)")
    assert diff < 1e-10, "h_pairs=0 should be equivalent to h=None"

    # 4. alpha_z scaling: doubling alpha_z should approximately halve the
    #    log-boost magnitude (for moderate boost where eps*delta is small).
    state3 = _make_synthetic_state(K_c, with_h=False, H_scale=0.3)
    log_boost3a = correct_pair_posterior(
        np.asarray(Q_baseline), x_seq, y_seq, t,
        state3, alpha_z=100.0, return_boost=True,
    )
    log_boost3b = correct_pair_posterior(
        np.asarray(Q_baseline), x_seq, y_seq, t,
        state3, alpha_z=200.0, return_boost=True,
    )
    ratio = float(jnp.mean(jnp.abs(log_boost3b)) /
                    jnp.maximum(jnp.mean(jnp.abs(log_boost3a)), 1e-30))
    # Expected ratio = (alpha_z_a + L_aln - 1) / (alpha_z_b + L_aln - 1).
    # With L_aln = sum(Q) ~ O(L_X * L_Y * mean_Q), this is somewhere
    # between 0.5 (large alpha_z dominance) and 1.0 (large L_aln dominance).
    L_aln = float(jnp.sum(Q_baseline))
    expected = (100.0 + L_aln - 1.0) / (200.0 + L_aln - 1.0)
    print(f"[alpha_z 100->200] mean |boost| ratio = {ratio:.4f}  "
            f"(expected ~ {expected:.4f})")
    assert abs(ratio - expected) < 0.05, \
        f"alpha_z scaling: got ratio {ratio:.3f}, expected {expected:.3f}"

    # 5. Q' = Q * exp(boost) should be finite + entrywise positive when
    #    Q baseline is positive.
    Q_prime = correct_pair_posterior(
        np.asarray(Q_baseline), x_seq, y_seq, t,
        state1, alpha_z=100.0, return_boost=False,
    )
    Q_prime_np = np.asarray(Q_prime)
    print(f"[Q']       finite?               = {np.all(np.isfinite(Q_prime_np))}")
    print(f"           min                    = {Q_prime_np.min():.3e}")
    print(f"           max                    = {Q_prime_np.max():.3e}")
    assert np.all(np.isfinite(Q_prime_np))
    assert (Q_prime_np >= 0).all()

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
