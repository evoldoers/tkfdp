"""Unit tests for the X-X / Y-Y edge-pair projection accumulators.

The MCMC sampler maintains `edge_pair_x_counts` and `edge_pair_y_counts`
on `MCMCDiagnostics` so that the joint Infinite Pair HMM coupled-pair
distribution can be projected onto each sequence axis (X or Y) for
comparison against single-sequence edge MCMC baselines.

For each edge ((i1, j1), (i2, j2)) recorded in a post-burnin sweep, the
projection contributes +1 to:

    edge_pair_x_counts[(min(i1, i2), max(i1, i2))]
    edge_pair_y_counts[(min(j1, j2), max(j1, j2))]

These tests construct hand-crafted edge sets, simulate what the run-time
accumulators do, and verify the projection matches the spec.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'analysis' / 'scripts'))

import numpy as np  # noqa: E402
from tkfdp.mcmc_infinite_phmm import MCMCDiagnostics  # noqa: E402
from _mcmc_diagnostics import diags_to_json  # noqa: E402


def _accumulate(diag: MCMCDiagnostics, edges, Lx: int, Ly: int) -> None:
    """Mirror the inlined post-burnin block from run_mcmc_chain / cold-rung."""
    diag.n_recorded_for_edges += 1
    for (a, b) in edges:
        ai, aj = a
        bi, bj = b
        if 1 <= ai <= Lx:
            diag.edge_pos_x_counts[ai] += 1
        if 1 <= bi <= Lx:
            diag.edge_pos_x_counts[bi] += 1
        if 1 <= aj <= Ly:
            diag.edge_pos_y_counts[aj] += 1
        if 1 <= bj <= Ly:
            diag.edge_pos_y_counts[bj] += 1
        diag.edge_cell_counts[(ai, aj)] = (
            diag.edge_cell_counts.get((ai, aj), 0) + 1)
        diag.edge_cell_counts[(bi, bj)] = (
            diag.edge_cell_counts.get((bi, bj), 0) + 1)
        if 1 <= ai <= Lx and 1 <= bi <= Lx:
            key_x = (min(ai, bi), max(ai, bi))
            diag.edge_pair_x_counts[key_x] = (
                diag.edge_pair_x_counts.get(key_x, 0) + 1)
        if 1 <= aj <= Ly and 1 <= bj <= Ly:
            key_y = (min(aj, bj), max(aj, bj))
            diag.edge_pair_y_counts[key_y] = (
                diag.edge_pair_y_counts.get(key_y, 0) + 1)


def _empty_diag(Lx: int, Ly: int) -> MCMCDiagnostics:
    d = MCMCDiagnostics()
    d.edge_pos_x_counts = [0] * (Lx + 1)
    d.edge_pos_y_counts = [0] * (Ly + 1)
    return d


def test_single_edge_canonical_order():
    """An edge ((3, 4), (7, 2)) projects to X-pair (3, 7) and Y-pair (2, 4)."""
    d = _empty_diag(Lx=10, Ly=10)
    edges = [((3, 4), (7, 2))]
    _accumulate(d, edges, Lx=10, Ly=10)
    assert d.edge_pair_x_counts == {(3, 7): 1}
    assert d.edge_pair_y_counts == {(2, 4): 1}
    # Endpoint counts: 2 endpoints per edge, one on each axis.
    assert sum(d.edge_pos_x_counts) == 2
    assert sum(d.edge_pos_y_counts) == 2
    assert d.edge_pos_x_counts[3] == 1
    assert d.edge_pos_x_counts[7] == 1
    assert d.edge_pos_y_counts[4] == 1
    assert d.edge_pos_y_counts[2] == 1


def test_swapping_endpoint_order_gives_same_unordered_pair():
    """The pair (a, b) is unordered: ((7, 2), (3, 4)) must give the SAME
    projection as ((3, 4), (7, 2))."""
    d1 = _empty_diag(Lx=10, Ly=10)
    d2 = _empty_diag(Lx=10, Ly=10)
    _accumulate(d1, [((3, 4), (7, 2))], Lx=10, Ly=10)
    _accumulate(d2, [((7, 2), (3, 4))], Lx=10, Ly=10)
    assert d1.edge_pair_x_counts == d2.edge_pair_x_counts
    assert d1.edge_pair_y_counts == d2.edge_pair_y_counts


def test_multiple_sweeps_accumulate():
    """Same edge in multiple sweeps -> count increments."""
    d = _empty_diag(Lx=10, Ly=10)
    edges = [((3, 4), (7, 2))]
    for _ in range(5):
        _accumulate(d, edges, Lx=10, Ly=10)
    assert d.edge_pair_x_counts[(3, 7)] == 5
    assert d.edge_pair_y_counts[(2, 4)] == 5
    assert d.n_recorded_for_edges == 5


def test_multiple_edges_one_sweep():
    """Two edges in a single sweep both contribute."""
    d = _empty_diag(Lx=10, Ly=10)
    edges = [((3, 4), (7, 2)), ((1, 5), (9, 8))]
    _accumulate(d, edges, Lx=10, Ly=10)
    assert d.edge_pair_x_counts == {(3, 7): 1, (1, 9): 1}
    assert d.edge_pair_y_counts == {(2, 4): 1, (5, 8): 1}


def test_distinct_x_pairs_same_y_pair():
    """Two edges with same Y-projection but different X-projection."""
    d = _empty_diag(Lx=10, Ly=10)
    edges = [((3, 4), (7, 2)), ((5, 4), (8, 2))]
    _accumulate(d, edges, Lx=10, Ly=10)
    assert d.edge_pair_x_counts == {(3, 7): 1, (5, 8): 1}
    # Both edges have Y-projection (2, 4); they aggregate.
    assert d.edge_pair_y_counts == {(2, 4): 2}


def test_endpoint_excluded_when_out_of_bounds_skips_pair():
    """If either endpoint's X position is out of range, no X-pair count
    is added. (Defensive: in practice both endpoints are always Match
    cells in 1..Lx x 1..Ly, but the sampler's accumulator guards.)"""
    d = _empty_diag(Lx=10, Ly=10)
    # bi=0 out of range -> no X-pair count, but Y-pair still records.
    edges = [((3, 4), (0, 2))]
    _accumulate(d, edges, Lx=10, Ly=10)
    assert d.edge_pair_x_counts == {}
    assert d.edge_pair_y_counts == {(2, 4): 1}


def test_serialization_round_trip():
    """diags_to_json should include the new fields as list-of-triples."""
    d = _empty_diag(Lx=10, Ly=10)
    _accumulate(d, [((3, 4), (7, 2)), ((1, 5), (9, 8))], Lx=10, Ly=10)
    d.n_match_trace = [2]
    d.log_pi_trace = [-3.14]
    d.n_edges_trace = [2]
    out = diags_to_json([d], is_replica_exchange=False)
    pc = out['per_chain'][0]
    # Convert back to dict for comparison.
    xp = {(int(r[0]), int(r[1])): int(r[2]) for r in pc['edge_pair_x_counts']}
    yp = {(int(r[0]), int(r[1])): int(r[2]) for r in pc['edge_pair_y_counts']}
    assert xp == {(3, 7): 1, (1, 9): 1}
    assert yp == {(2, 4): 1, (5, 8): 1}


def test_dense_projection_matrix_construction():
    """Verify we can build a triangular dense matrix from the sparse dict."""
    d = _empty_diag(Lx=5, Ly=5)
    edges_sweep_a = [((1, 1), (5, 5)), ((2, 3), (4, 4))]
    edges_sweep_b = [((1, 1), (5, 5))]
    edges_sweep_c = []
    _accumulate(d, edges_sweep_a, Lx=5, Ly=5)
    _accumulate(d, edges_sweep_b, Lx=5, Ly=5)
    _accumulate(d, edges_sweep_c, Lx=5, Ly=5)
    # P_xx[i1, i2] = count / n_recorded_for_edges, symmetric upper triangle.
    Lx = 5
    P_xx = np.zeros((Lx + 1, Lx + 1), dtype=np.float64)
    for (i1, i2), c in d.edge_pair_x_counts.items():
        P_xx[i1, i2] = c / d.n_recorded_for_edges
        P_xx[i2, i1] = P_xx[i1, i2]
    # (1, 5) appeared in sweeps a + b => 2/3.
    assert abs(P_xx[1, 5] - 2.0 / 3.0) < 1e-12
    assert abs(P_xx[5, 1] - 2.0 / 3.0) < 1e-12
    # (2, 4) appeared in sweep a only => 1/3.
    assert abs(P_xx[2, 4] - 1.0 / 3.0) < 1e-12
    # Diagonal must be zero (edges connect DISTINCT cells; same X-position
    # would imply a self-loop, which the sampler does not propose).
    for i in range(Lx + 1):
        assert P_xx[i, i] == 0.0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
