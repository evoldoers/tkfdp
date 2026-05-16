"""Unit tests for analysis/scripts/_mcmc_diagnostics.py.

Tests:
  * ESS via Geyer-IACT is N for an IID chain (within 30% tolerance).
  * ESS is much smaller than N for a highly correlated AR(1) chain.
  * Gelman-Rubin r-hat is ~1.0 for IID chains and >2 for different-mean chains.
  * Acceptance-rate extraction from MCMCDiagnostics-like objects.
  * diags_to_json shape with single-chain / multi-chain / replica-exchange.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'analysis' / 'scripts'))

from _mcmc_diagnostics import (
    ess_iact,
    rhat_gelman_rubin,
    acceptance_rates,
    diags_to_json,
)


def test_ess_iact_iid_close_to_n():
    """IID samples have negligible autocorrelation, so ESS ~ N."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    ess = ess_iact(x)
    # Geyer truncation may produce small floor effects; require ESS in [700, 1300].
    assert 700 <= ess <= 1300, f'IID ESS=1000, got {ess}'


def test_ess_iact_ar1_much_less_than_n():
    """AR(1) with phi=0.95 has integrated tau ~ 1+2*phi/(1-phi) ~ 39, so
    ESS ~ N / tau = 1000 / 39 ~ 25.  Allow factor-2 slack."""
    rng = np.random.default_rng(42)
    n = 1000
    phi = 0.95
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    ess = ess_iact(x)
    assert ess < 100, f'AR(1)(0.95) ESS should be << N, got {ess}'


def test_ess_iact_short_chain_returns_n():
    """N < 4 -> ESS = N (no autocorrelation estimate)."""
    assert ess_iact([1.0, 2.0]) == 2.0
    assert ess_iact([1.0]) == 1.0
    assert ess_iact([]) == 0.0


def test_ess_iact_constant_chain():
    """A constant chain has zero variance; treat as ESS = N (no info)."""
    x = np.full(100, 7.0)
    assert ess_iact(x) == 100.0


def test_rhat_iid_close_to_one():
    """4 IID chains from the same distribution -> r-hat ~ 1.0."""
    rng = np.random.default_rng(42)
    chains = [rng.standard_normal(500) for _ in range(4)]
    r = rhat_gelman_rubin(chains)
    assert r is not None
    assert abs(r - 1.0) < 0.1, f'r-hat for IID should be ~1.0, got {r}'


def test_rhat_different_means_above_two():
    """Chains with different means -> r-hat >> 1."""
    rng = np.random.default_rng(42)
    chains = [rng.standard_normal(500) + i * 5 for i in range(4)]
    r = rhat_gelman_rubin(chains)
    assert r > 2.0, f'r-hat for diff-mean chains should be >2, got {r}'


def test_rhat_less_than_two_chains_returns_none():
    """Need at least 2 chains for r-hat."""
    assert rhat_gelman_rubin([np.random.randn(100)]) is None
    assert rhat_gelman_rubin([]) is None


class _FakeDiag:
    """Minimal stand-in for MCMCDiagnostics."""
    def __init__(self, n_match, log_pi, n_edges,
                 acc_seg=1.0, acc_add=0.1, acc_remove=0.05):
        self.n_match_trace = list(n_match)
        self.log_pi_trace = list(log_pi)
        self.n_edges_trace = list(n_edges)
        self.n_sweeps = 500
        self.n_burnin = 100
        self.n_propose_seg = 500
        self.n_accept_seg = int(500 * acc_seg)
        self.n_propose_add = 4000
        self.n_accept_add = int(4000 * acc_add)
        self.n_propose_remove = 4000
        self.n_accept_remove = int(4000 * acc_remove)
        self.runtime_seconds = 10.0


def test_acceptance_rates_extraction():
    d = _FakeDiag([1] * 10, [2.0] * 10, [0] * 10,
                  acc_seg=1.0, acc_add=0.25, acc_remove=0.1)
    r = acceptance_rates(d)
    assert r['acc_seg'] == 1.0
    assert abs(r['acc_add'] - 0.25) < 1e-6
    assert abs(r['acc_remove'] - 0.1) < 1e-6
    assert r['n_propose_seg'] == 500
    assert r['n_accept_seg'] == 500


def test_diags_to_json_single_chain():
    rng = np.random.default_rng(0)
    d = _FakeDiag(
        n_match=rng.integers(0, 50, 400).tolist(),
        log_pi=rng.standard_normal(400).tolist(),
        n_edges=rng.integers(0, 5, 400).tolist(),
    )
    out = diags_to_json([d], is_replica_exchange=False,
                          q_l1_vs_baseline=0.001)
    assert out['n_chains'] == 1
    assert out['is_replica_exchange'] is False
    assert len(out['per_chain']) == 1
    pc = out['per_chain'][0]
    assert pc['n_recorded'] == 400
    assert pc['ess_n_match'] is not None
    assert pc['ess_log_pi'] is not None
    assert pc['acc_seg'] == 1.0
    # r-hat requires >=2 chains.
    assert out['r_hat_n_match'] is None
    assert out['q_l1_vs_baseline'] == 0.001


def test_diags_to_json_multi_chain_has_rhat():
    rng = np.random.default_rng(0)
    chains = []
    for i in range(4):
        chains.append(_FakeDiag(
            n_match=rng.integers(0, 50, 400).tolist(),
            log_pi=rng.standard_normal(400).tolist(),
            n_edges=rng.integers(0, 5, 400).tolist(),
        ))
    out = diags_to_json(chains, is_replica_exchange=False)
    assert out['n_chains'] == 4
    assert out['r_hat_n_match'] is not None
    assert out['r_hat_log_pi'] is not None
    assert out['between_chain_n_match_sd'] is not None
    assert abs(out['r_hat_n_match'] - 1.0) < 0.3  # should be ~1.0


def test_diags_to_json_replica_exchange_skips_rhat():
    """Replica-exchange rungs sample different targets, so r-hat across
    them is meaningless and should be None."""
    rng = np.random.default_rng(0)
    rungs = []
    for alpha_idx in range(5):
        rungs.append(_FakeDiag(
            n_match=rng.integers(0, 50 - alpha_idx * 8, 400).tolist(),
            log_pi=rng.standard_normal(400).tolist(),
            n_edges=rng.integers(0, 5, 400).tolist(),
        ))
    out = diags_to_json(rungs, is_replica_exchange=True,
                          alpha_z_ladder=[100, 500, 1000, 10000, 1e6],
                          swap_n_propose=[10, 10, 10, 10],
                          swap_n_accept=[3, 2, 1, 0])
    assert out['is_replica_exchange'] is True
    assert out['n_chains'] == 5
    assert out['r_hat_n_match'] is None
    assert out['r_hat_log_pi'] is None
    assert out['alpha_z_ladder'] == [100.0, 500.0, 1000.0, 10000.0, 1e6]
    assert out['swap_acc_rates'] == [0.3, 0.2, 0.1, 0.0]


def test_diags_to_json_downsamples_long_traces():
    """Traces longer than 5000 are downsampled to keep JSON size sane."""
    d = _FakeDiag(
        n_match=list(range(10000)),
        log_pi=[float(i) for i in range(10000)],
        n_edges=list(range(10000)),
    )
    out = diags_to_json([d], is_replica_exchange=False)
    pc = out['per_chain'][0]
    assert len(pc['log_pi_trace']) <= 5000
    assert pc.get('_trace_downsample_stride') is not None
    assert pc['n_recorded'] == 10000  # still records full count
