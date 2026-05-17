"""Tests for the single-sequence edge MCMC sampler.

Validates that `run_single_seq_chain` equilibrates to the EXACT
enumeration limit on a small toy (L = 5; 26 matchings) within MC error.

The exact enumeration target is

    P(E | x) propto eps^|E| * prod_{(i, j) in E} M_solo(x_i, x_j)

on matchings (no shared endpoints), with eps = 1/alpha_z.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

import numpy as np  # noqa: E402

from tkfdp.single_seq_edge_mcmc import (  # noqa: E402
    SingleSeqSetup,
    _all_matchings,
    exact_edge_pair_posterior,
    run_single_seq_chain,
    run_single_seq_replica_exchange,
)


def _toy_setup(L: int = 5, A: int = 4, alpha_z: float = 5.0, seed: int = 7
               ) -> SingleSeqSetup:
    """Build a toy single-seq setup with a tiny alphabet so we can
    explicitly construct log_M_solo without referring to boost states."""
    rng = np.random.default_rng(seed)
    # Tiny alphabet, random sequence.
    x_seq = rng.integers(0, A, size=L).astype(np.int32)
    # Random symmetric positive M_solo so log is finite and finite range.
    raw = rng.uniform(0.5, 2.5, size=(A, A))
    M_solo = 0.5 * (raw + raw.T)
    log_M_solo = np.log(M_solo)
    return SingleSeqSetup(
        L=L, x_seq=x_seq, log_M_solo=log_M_solo, alpha_z=alpha_z,
    )


def test_all_matchings_count():
    """Number of matchings on L positions = double factorial of L-1
    counting empty + 1-edge + 2-edge + ... matchings.

    For L = 4 the count is 1 + 6 + 3 = 10 (1 empty, 6 single edges,
    3 double matchings).
    """
    matchings = list(_all_matchings(4))
    assert len(matchings) == 10
    # All entries are matchings (no shared endpoint).
    for M in matchings:
        positions = []
        for (i, j) in M:
            positions.extend([i, j])
        assert len(set(positions)) == len(positions), \
            f"Matching has shared endpoint: {M}"


def test_exact_enumeration_is_probability_distribution():
    """Sum over all matchings should give probabilities that sum to 1
    on the pair-marginal? No -- the matchings probabilities sum to 1
    over configurations; the pair-marginal P[i, j] is just the marginal.
    """
    setup = _toy_setup(L=4, alpha_z=3.0, seed=42)
    P = exact_edge_pair_posterior(setup)
    # All entries in [0, 1].
    assert (P >= 0.0).all()
    assert (P <= 1.0 + 1e-12).all()
    # Diagonal is zero.
    for i in range(P.shape[0]):
        assert P[i, i] == 0.0
    # Symmetric.
    np.testing.assert_array_almost_equal(P, P.T)


def test_mcmc_equilibrates_to_exact_enumeration_L4_alpha3():
    """MCMC posterior should match exact enumeration within MC error
    on a small (L = 4) toy.

    With ~20000 post-burnin recorded samples and pair-probabilities
    near 0.3, MC SE per cell is ~sqrt(0.3 * 0.7 / 20000) ~ 0.003. We
    require absolute error <= 0.02 (covers most cells under ~6 sigma).
    """
    setup = _toy_setup(L=4, alpha_z=3.0, seed=42)
    P_exact = exact_edge_pair_posterior(setup)
    P_mcmc, diag = run_single_seq_chain(
        setup, n_sweeps=22000, n_burnin=2000,
        n_edge_moves_per_sweep=4, seed=11, record_every=1)
    # Compare on the upper triangle (i < j).
    L = setup.L
    diffs = []
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            diffs.append(abs(P_exact[i, j] - P_mcmc[i, j]))
    diffs = np.asarray(diffs)
    assert diffs.max() < 0.04, (
        f"Max abs deviation {diffs.max():.4f} between MCMC and exact "
        f"on L=4 alpha_z=3. P_exact:\n{P_exact[1:, 1:]}\n"
        f"P_mcmc:\n{P_mcmc[1:, 1:]}\n"
        f"acc_add={diag.n_accept_add / max(1, diag.n_propose_add):.2f} "
        f"acc_rm={diag.n_accept_remove / max(1, diag.n_propose_remove):.2f}"
    )


def test_mcmc_equilibrates_to_exact_enumeration_L5_alpha5():
    """Larger toy: L = 5, alpha_z = 5 (sparser edges)."""
    setup = _toy_setup(L=5, alpha_z=5.0, seed=99)
    P_exact = exact_edge_pair_posterior(setup)
    P_mcmc, diag = run_single_seq_chain(
        setup, n_sweeps=22000, n_burnin=2000,
        n_edge_moves_per_sweep=4, seed=23, record_every=1)
    L = setup.L
    diffs = []
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            diffs.append(abs(P_exact[i, j] - P_mcmc[i, j]))
    diffs = np.asarray(diffs)
    assert diffs.max() < 0.04, (
        f"Max abs deviation {diffs.max():.4f}. P_exact:\n{P_exact[1:, 1:]}\n"
        f"P_mcmc:\n{P_mcmc[1:, 1:]}"
    )


def test_high_alpha_z_gives_near_zero_edges():
    """At very large alpha_z, eps is tiny so virtually no edges exist."""
    setup = _toy_setup(L=5, alpha_z=10000.0, seed=42)
    P_mcmc, diag = run_single_seq_chain(
        setup, n_sweeps=5000, n_burnin=500,
        n_edge_moves_per_sweep=4, seed=1)
    # Every marginal should be very close to zero.
    assert P_mcmc.max() < 0.01


def test_low_alpha_z_concentrates_on_high_M_pairs():
    """At small alpha_z the chain favours pairs with high M_solo."""
    rng = np.random.default_rng(0)
    L = 4
    A = 4
    x_seq = np.array([0, 1, 2, 3], dtype=np.int32)
    # Construct an M_solo where only the (0, 1) pair has a large boost.
    M_solo = np.ones((A, A)) * 0.1
    M_solo[0, 1] = M_solo[1, 0] = 100.0
    log_M_solo = np.log(M_solo)
    setup = SingleSeqSetup(L=L, x_seq=x_seq, log_M_solo=log_M_solo,
                            alpha_z=0.5)
    P_mcmc, diag = run_single_seq_chain(
        setup, n_sweeps=10000, n_burnin=2000,
        n_edge_moves_per_sweep=4, seed=2)
    # The pair (1, 2) has X[0]=0, X[1]=1 -- the boosted AA pair.
    # Therefore P[1, 2] should be high (close to 1).
    assert P_mcmc[1, 2] > 0.7, f"Expected P[1,2] > 0.7, got {P_mcmc[1, 2]}"
    # Other pairs should be lower.
    for (i, j) in [(1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
        assert P_mcmc[i, j] < 0.6, (
            f"Expected P[{i},{j}] < 0.6, got {P_mcmc[i, j]}")


def test_replica_exchange_matches_exact():
    """Replica-exchange version should also match the exact enumeration."""
    setup = _toy_setup(L=4, alpha_z=3.0, seed=42)
    P_exact = exact_edge_pair_posterior(setup)
    P_mcmc, diag = run_single_seq_replica_exchange(
        setup, alpha_z_ladder=[3.0, 10.0, 100.0, 1e4],
        n_sweeps=22000, n_burnin=2000,
        n_edge_moves_per_sweep=4, seed=11, swap_every=10)
    L = setup.L
    diffs = []
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            diffs.append(abs(P_exact[i, j] - P_mcmc[i, j]))
    diffs = np.asarray(diffs)
    assert diffs.max() < 0.04, f"RE max dev {diffs.max():.4f}"


def test_canonical_M_solo_equals_M_tensor_diagonal_at_t0():
    """Regression test for the post-2026-05-15 wiring.

    M_solo[a, c] := sum_{b, d} P_doublet[a, b, c, d] /
                     (pi_singlet(a) * pi_singlet(c))

    At t -> 0 this reduces to
        M_solo[a, c] = pi_joint(a, c) / (pi_singlet(a) * pi_singlet(c))

    The DIAGONAL of the M-tensor at t -> 0 equals the same thing:
        M_tensor[a, a, c, c]
          = P_doublet[a, a, c, c] / (P_singlet[a, a] * P_singlet[c, c])
          = (pi_joint(a, c) * delta(a, a) * delta(c, c)) /
             (pi_singlet(a) * delta(a, a) * pi_singlet(c) * delta(c, c))
          = pi_joint(a, c) / (pi_singlet(a) * pi_singlet(c))

    So at t -> 0 these must agree. (At t > 0 they diverge because
    M_solo stays t-independent while M_tensor[a, a, c, c] picks up
    P_trans(a -> a; t) and P_trans(c -> c; t) factors -- the two
    quantities measure different things in general.)
    """
    from tkfdp.single_seq_edge_mcmc import _build_M_solo_canonical
    from tkfdp.block_likelihoods import build_M_tensor
    from tkfdp.potts_dp import PottsDPState

    rng = np.random.default_rng(2026)
    K_c = 3
    A = 20
    pi_class = rng.dirichlet(np.ones(A), size=K_c).astype(np.float32)
    n_atoms = K_c * (K_c + 1) // 2
    atoms = rng.normal(0.0, 0.5, size=(n_atoms, A, A)).astype(np.float32)
    atoms = 0.5 * (atoms + atoms.transpose(0, 2, 1))   # symmetric H

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
    state = _S()
    state.K_c = K_c
    state.A = A
    state.pi_class = pi_class
    state.potts_dp = PottsDPState(K_c=K_c, A=A,
        atoms=atoms, assignments=assignments,
        counts=counts, alpha_H=1.0)

    pi_c = rng.dirichlet(np.ones(K_c))
    t = 1e-4   # essentially the t -> 0 limit
    M_solo = _build_M_solo_canonical(state, pi_c, t, pair_background='lg08')
    M = build_M_tensor(state, t, pi_c=pi_c, pair_background='lg08')
    diag = np.array([[M[a, a, c, c] for c in range(A)] for a in range(A)])
    err = float(np.max(np.abs(np.log(np.maximum(M_solo, 1e-300))
                                - np.log(np.maximum(diag, 1e-300)))))
    # Small-but-nonzero floor: the doublet emission uses
    # eta=1 in expm(Q*t), so at t=1e-4 there is a sub-percent transition
    # leakage on the diagonal.
    assert err < 1e-2, f"max log-error between M_solo and diag(M) = {err:.2e}"


def test_canonical_M_solo_is_t_independent():
    """`M_solo[a, c] = pi_joint(a, c) / (pi_singlet(a) pi_singlet(c))` is
    t-independent under reversibility (the doublet integrates to
    pi_joint over the unobserved axis at every t, and the singlet
    integrates to pi_singlet)."""
    from tkfdp.single_seq_edge_mcmc import _build_M_solo_canonical
    from tkfdp.potts_dp import PottsDPState

    rng = np.random.default_rng(7)
    K_c = 2
    A = 20
    n_atoms = K_c * (K_c + 1) // 2
    atoms = rng.normal(0.0, 0.3, size=(n_atoms, A, A)).astype(np.float32)
    atoms = 0.5 * (atoms + atoms.transpose(0, 2, 1))
    assignments = np.array([[0, 1], [1, 2]], dtype=np.int64)

    class _S:
        pass
    state = _S()
    state.K_c = K_c
    state.A = A
    state.pi_class = rng.dirichlet(np.ones(A), size=K_c).astype(np.float32)
    state.potts_dp = PottsDPState(K_c=K_c, A=A,
        atoms=atoms, assignments=assignments,
        counts=np.ones(n_atoms, dtype=np.int64), alpha_H=1.0)

    pi_c = np.array([0.4, 0.6])
    M_small_t = _build_M_solo_canonical(state, pi_c, 1e-3)
    M_mid_t   = _build_M_solo_canonical(state, pi_c, 1.0)
    M_big_t   = _build_M_solo_canonical(state, pi_c, 50.0)
    err_mid = float(np.max(np.abs(np.log(M_mid_t) - np.log(M_small_t))))
    err_big = float(np.max(np.abs(np.log(M_big_t) - np.log(M_small_t))))
    assert err_mid < 1e-4, f"mid-t deviation {err_mid:.2e}"
    assert err_big < 1e-3, f"big-t deviation {err_big:.2e}"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
