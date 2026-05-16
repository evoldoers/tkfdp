"""Worked-example tests for the unified P_singlet / P_doublet / M
builders in `tkfdp.block_likelihoods`.

Four canonical regimes documented in
`math-paper/.review/math-verifier-2026-05-14.md` (#111 / task list):

1. ``test_M_reduces_at_t0``: M-tensor at t -> 0 reduces to the
   stationary log-odds matrix
       M[a, a, c, c] = pi_joint(a, c) / (pi_singlet(a) pi_singlet(c)).

2. ``test_disulfide_persistence``: when the H atoms include a strong
   negative (attractive) coupling on the (C, C) cell, the resulting
   M-tensor still shows that signal at moderate branch lengths
   (t = 0.5, 1.0, 2.0) -- the coupling is not washed out by
   single-site substitution dynamics.

3. ``test_multi_cys_covariation``: the M_solo marginal under the same
   strong-CC state shows the C-C boost AND propagates to multi-Cys
   contexts (no destructive cancellation across class pairs).

4. ``test_single_cys_negative_control``: when H = 0 (no Potts coupling
   anywhere), the M-tensor is identically 1 (and log_2 M = 0)
   everywhere -- there is no spurious covariation signal.

All four use small synthetic states so the tests are fast and do not
depend on a downloaded K=4 checkpoint.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

import numpy as np    # noqa: E402


A = 20
CYS = 1                          # 'C' index in 'ACDEFGHIKLMNPQRSTVWY'


def _make_state(atoms_factory, K_c: int = 3, seed: int = 1,
                 pi_class_random: bool = False):
    """Build a tiny PottsDPState-backed state.

    Args:
        atoms_factory: ``(rng, n_atoms, A) -> (n, A, A)`` atom builder.
        K_c: number of latent classes.
        seed: rng seed.
        pi_class_random: if True, draw random per-class pi_class from
            Dirichlet(1). Default False: every class uses pi_class[c] =
            PI_LG08, so the canonical convention is exactly self-
            consistent (pi_singlet = PI_LG08, eliminating
            per-class-vs-pair-background drift). Use True only for
            tests that explicitly want to inject drift.
    """
    from tkfdp.potts_dp import PottsDPState
    from tkfdp.lg08 import PI_LG08_J
    rng = np.random.default_rng(seed)
    if pi_class_random:
        pi_class = rng.dirichlet(np.ones(A), size=K_c).astype(np.float32)
    else:
        pi_class = np.tile(np.asarray(PI_LG08_J, dtype=np.float32),
                            (K_c, 1))
    n_atoms = K_c * (K_c + 1) // 2
    atoms = atoms_factory(rng, n_atoms, A).astype(np.float32)
    # Force symmetric atoms (TKF-DP convention: H_{c c'} symmetric in a, b).
    atoms = 0.5 * (atoms + atoms.transpose(0, 2, 1))
    assignments = np.zeros((K_c, K_c), dtype=np.int64)
    k = 0
    for c1 in range(K_c):
        for c2 in range(c1, K_c):
            assignments[c1, c2] = k
            assignments[c2, c1] = k
            k += 1
    counts = np.ones(n_atoms, dtype=np.int64)

    class _S:
        pass
    s = _S()
    s.K_c = K_c
    s.A = A
    s.pi_class = pi_class
    s.potts_dp = PottsDPState(K_c=K_c, A=A,
        atoms=atoms, assignments=assignments,
        counts=counts, alpha_H=1.0)
    return s


# ---------------------------------------------------------------------------
# Test 1: t -> 0 reduction of M
# ---------------------------------------------------------------------------

def test_M_reduces_at_t0():
    """At t -> 0, M-tensor diagonal cells equal the stationary log-odds
    matrix pi_joint(a, c) / (pi_singlet(a) pi_singlet(c)). This is the
    fundamental t-independent identity for the single-seq sampler's
    M_solo and the joint sampler's M tensor."""
    from tkfdp.block_likelihoods import (
        build_M_tensor, build_doublet_emission, build_singlet_emission)

    state = _make_state(
        lambda rng, n, A: rng.normal(0.0, 0.4, size=(n, A, A)),
        K_c=2, seed=11)
    pi_c = np.array([0.3, 0.7])
    t = 1e-4
    M = build_M_tensor(state, t, pi_c=pi_c, pair_background='lg08')
    P_d = build_doublet_emission(state, t, pi_c=pi_c, pair_background='lg08')
    P_s, _, _ = build_singlet_emission(state, t, pi_c=pi_c)

    # pi_joint from diagonal of doublet, pi_singlet from row sum of singlet.
    pi_joint = np.array([[P_d[a, a, c, c] for c in range(A)] for a in range(A)])
    pi_singlet = P_s.sum(axis=1)
    expected = pi_joint / (pi_singlet[:, None] * pi_singlet[None, :])
    diag_M = np.array([[M[a, a, c, c] for c in range(A)] for a in range(A)])

    err = float(np.max(np.abs(np.log(np.maximum(diag_M, 1e-300))
                                - np.log(np.maximum(expected, 1e-300)))))
    assert err < 1e-3, f"max log-error at t->0 = {err:.2e}"


# ---------------------------------------------------------------------------
# Test 2: disulfide-like persistence
# ---------------------------------------------------------------------------

def _strong_cc_atoms(rng, n_atoms, A, cc_weight=-3.0):
    """Random atoms with a strong attractive coupling at (C, C)."""
    atoms = rng.normal(0.0, 0.2, size=(n_atoms, A, A))
    atoms[:, CYS, CYS] = cc_weight                            # H < 0 -> attract
    return atoms


def test_disulfide_persistence():
    """A strong negative H at (C, C) cell should produce log_2 M[C, C, C, C]
    > 1.0 at t -> 0, AND log_2 M_solo[C, C] > 1.0 at every t in
    {0.5, 1.0, 2.0} -- the coupling signal persists through finite-t
    substitution dynamics."""
    from tkfdp.block_likelihoods import build_M_tensor
    from tkfdp.single_seq_edge_mcmc import _build_M_solo_canonical

    state = _make_state(
        lambda rng, n, A: _strong_cc_atoms(rng, n, A, cc_weight=-3.0),
        K_c=2, seed=23)
    pi_c = np.array([0.4, 0.6])

    # log_2 M[C, C, C, C] at t -> 0
    M0 = build_M_tensor(state, 1e-4, pi_c=pi_c, pair_background='lg08')
    log2_cc_t0 = float(np.log2(max(M0[CYS, CYS, CYS, CYS], 1e-300)))
    assert log2_cc_t0 > 1.0, (
        f"Expected log_2 M[C,C,C,C] > 1.0 at t=0, got {log2_cc_t0:+.3f}")

    # log_2 M_solo[C, C] at t in {0.5, 1.0, 2.0} (t-independent)
    for t in (0.5, 1.0, 2.0):
        M_solo = _build_M_solo_canonical(state, pi_c, t,
                                          pair_background='lg08')
        log2_solo_cc = float(np.log2(max(M_solo[CYS, CYS], 1e-300)))
        assert log2_solo_cc > 1.0, (
            f"Expected log_2 M_solo[C, C] > 1.0 at t={t}, "
            f"got {log2_solo_cc:+.3f}")


# ---------------------------------------------------------------------------
# Test 3: multi-Cys covariation
# ---------------------------------------------------------------------------

def test_multi_cys_covariation():
    """Under the strong-CC state, the M_solo[C, C] log-odds should be
    larger than M_solo[A, R] (a pair with no induced coupling) and
    larger than the geometric mean of all non-CC cells. I.e. the model
    captures the targeted covariation without spurious signal at
    background pairs."""
    from tkfdp.single_seq_edge_mcmc import _build_M_solo_canonical

    state = _make_state(
        lambda rng, n, A: _strong_cc_atoms(rng, n, A, cc_weight=-3.0),
        K_c=2, seed=37)
    pi_c = np.array([0.4, 0.6])
    M_solo = _build_M_solo_canonical(state, pi_c, 1.0,
                                      pair_background='lg08')
    log2_solo = np.log2(np.maximum(M_solo, 1e-300))

    # Target: C-C should be much larger than a random background pair.
    log2_cc = log2_solo[CYS, CYS]
    log2_ar = log2_solo[0, 14]                  # A-R (background)
    assert log2_cc - log2_ar > 1.5, (
        f"Expected log_2 M_solo[C,C] - log_2 M_solo[A,R] > 1.5, "
        f"got {log2_cc - log2_ar:+.3f}")

    # And C-C should exceed the geometric mean (= mean of log_2) of all
    # off-Cys cells.
    mask = np.ones((A, A), dtype=bool)
    mask[CYS, :] = False
    mask[:, CYS] = False
    mean_log2_offCC = float(log2_solo[mask].mean())
    assert log2_cc - mean_log2_offCC > 1.0, (
        f"Expected log_2 M_solo[C,C] - <log_2 M_solo[off-CC]> > 1.0, "
        f"got {log2_cc - mean_log2_offCC:+.3f}")


# ---------------------------------------------------------------------------
# Test 4: single-Cys negative control
# ---------------------------------------------------------------------------

def test_single_cys_negative_control():
    """When H = 0 (Potts coupling identically zero), the M-tensor must
    be 1 everywhere and the M_solo marginal must be 1 everywhere -- no
    spurious covariation signal anywhere in the alphabet.

    This is a critical regression check: any future bug that introduces
    a non-trivial coupling baseline (e.g. forgetting to subtract the
    independent-product denominator, or using the wrong stationary)
    will fail this test on every cell."""
    from tkfdp.block_likelihoods import build_M_tensor
    from tkfdp.single_seq_edge_mcmc import _build_M_solo_canonical

    state = _make_state(
        lambda rng, n, A: np.zeros((n, A, A)),     # H = 0 everywhere
        K_c=2, seed=2)
    pi_c = np.array([0.5, 0.5])

    # M tensor: should be 1 in every cell.
    M = build_M_tensor(state, 1.0, pi_c=pi_c, pair_background='lg08')
    log2_M = np.log2(np.maximum(M, 1e-300))
    assert np.max(np.abs(log2_M)) < 5e-2, (
        f"H=0 should give M=1 everywhere; "
        f"max |log_2 M| = {float(np.max(np.abs(log2_M))):.3e}")

    # M_solo: should be 1 in every cell.
    M_solo = _build_M_solo_canonical(state, pi_c, 1.0,
                                      pair_background='lg08')
    log2_M_solo = np.log2(np.maximum(M_solo, 1e-300))
    assert np.max(np.abs(log2_M_solo)) < 5e-2, (
        f"H=0 should give M_solo=1 everywhere; "
        f"max |log_2 M_solo| = {float(np.max(np.abs(log2_M_solo))):.3e}")


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
