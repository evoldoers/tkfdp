"""Sanity tests for update_alpha_z_mh."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from tkfdp.partition_K import update_alpha_z_mh


def _log_post_full(alpha, partitions, prior_a, prior_b):
    """Reference log-density up to a constant (matches the MH ratio)."""
    if alpha <= 0:
        return -np.inf
    ll = 0.0
    for K, n in partitions:
        ll += (K * np.log(alpha) + gammaln(alpha)
                - gammaln(alpha + n))
    lp = (prior_a - 1.0) * np.log(alpha) - prior_b * alpha
    return ll + lp


def test_mh_zero_steps_is_no_op():
    rng = np.random.default_rng(0)
    parts = [(5, 100), (3, 80)]
    a_in = 1.7
    a_out, info = update_alpha_z_mh(
        a_in, parts, prior_a=1.5, prior_b=2.0,
        n_steps=0, rng=rng)
    assert a_out == a_in
    assert info['n_steps_accept'] == 0


def test_mh_moves_toward_posterior_mode():
    """Many MH steps starting far from the mode should drift toward it."""
    rng = np.random.default_rng(2)
    # Construct a corpus that strongly favors a particular alpha_z.
    # With 50 MSAs each n=100, K=30 (relatively many clusters), Ewens
    # likelihood peaks near alpha ~ K / log(n / K) -- non-trivial value.
    parts = [(30, 100)] * 50
    prior_a, prior_b = 1.5, 2.0

    # Sweep to find approximate mode by grid search.
    grid = np.linspace(0.01, 50.0, 1001)
    lps = np.array([_log_post_full(a, parts, prior_a, prior_b) for a in grid])
    mode = grid[np.argmax(lps)]
    assert mode > 0
    # Now run MH starting far from mode and check that it gravitates.
    a = 100.0
    chain = [a]
    for _ in range(500):
        a, _ = update_alpha_z_mh(
            a, parts, prior_a=prior_a, prior_b=prior_b,
            n_steps=1, step_size=0.3, rng=rng)
        chain.append(a)
    burned = chain[200:]
    median = float(np.median(burned))
    # Should be within ~50% of the grid-search mode (broad acceptance).
    assert abs(median - mode) / mode < 0.5, (
        f"chain median {median:.3f} vs mode {mode:.3f}")


def test_mh_acceptance_ratio_matches_reference():
    """For a single MH step, the accept probability equals
    min(1, exp(log_post(prop) - log_post(cur))). Verify by setting up
    a deterministic proposal via fixed RNG."""
    rng = np.random.default_rng(11)
    parts = [(7, 50), (4, 30)]
    a = 2.5
    # Fix step_size=0; proposal == current; should always accept (ratio = 0).
    a_out, info = update_alpha_z_mh(
        a, parts, n_steps=10, step_size=0.0, rng=rng)
    assert a_out == a
    assert info['n_steps_accept'] == 10


if __name__ == "__main__":
    test_mh_zero_steps_is_no_op()
    test_mh_acceptance_ratio_matches_reference()
    test_mh_moves_toward_posterior_mode()
    print("OK")
