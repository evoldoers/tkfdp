"""Unit test for the MSA-column projection logic in
analysis/scripts/plot_holmes_msa_triangle.py.

Given an aligned sequence and a list of sequence-position edge pairs,
the projection
    col_pair_counts[(min(col(p1), col(p2)), max(col(p1), col(p2)))] += 1
must be invariant under endpoint-order swap and skip gaps correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis" / "scripts"))

from plot_holmes_msa_triangle import aligned_to_seq_and_map, _aa_to_int  # noqa: E402


def test_aligned_to_seq_basic():
    """Simple aligned: A-CD- maps to seq ACD at positions 1, 2, 3 and
    col_to_pos = [0, 1, 0, 2, 3, 0]. Alphabet ARNDCQEGHILKMFPSTWYV:
    A=0, C=4, D=3, so seq_int = [0, 4, 3]."""
    A2I = _aa_to_int()
    seq_int, c2p, p2c = aligned_to_seq_and_map("A-CD-", A2I)
    assert seq_int.tolist() == [0, 4, 3]
    # col_to_pos: col 1 -> pos 1, col 2 -> 0 (gap), col 3 -> pos 2,
    # col 4 -> pos 3, col 5 -> 0 (gap). Plus index 0 unused.
    assert c2p.tolist() == [0, 1, 0, 2, 3, 0]
    assert p2c.tolist() == [0, 1, 3, 4]


def test_gap_chars():
    """Both '-' and '.' should map to gap, all other letters to residues."""
    A2I = _aa_to_int()
    seq_int, c2p, p2c = aligned_to_seq_and_map("A.C-D", A2I)
    assert seq_int.shape[0] == 3
    assert c2p[1] == 1
    assert c2p[2] == 0  # gap
    assert c2p[3] == 2  # C is pos 2
    assert c2p[4] == 0  # gap
    assert c2p[5] == 3  # D is pos 3


def test_pos_to_col_inverse():
    """For all residues, p2c[c2p[col]] should map back to col."""
    A2I = _aa_to_int()
    aln = "...ACGT...AC..A"
    seq_int, c2p, p2c = aligned_to_seq_and_map(aln, A2I)
    L = seq_int.shape[0]
    for p in range(1, L + 1):
        col = int(p2c[p])
        assert int(c2p[col]) == p, (
            f"pos {p} -> col {col} -> pos {c2p[col]} (round-trip failed)")


def test_edge_projection_simulation():
    """Simulate a one-edge sample on a small aligned pair and check that
    projection logic produces the expected MSA-column-pair counts."""
    A2I = _aa_to_int()
    # X aligned: "AC.GT" (4 residues at cols 1,2,4,5 in a 5-col MSA).
    s_x, c2p_x, p2c_x = aligned_to_seq_and_map("AC.GT", A2I)
    # Y aligned: "A.CGT" (4 residues at cols 1,3,4,5).
    s_y, c2p_y, p2c_y = aligned_to_seq_and_map("A.CGT", A2I)
    Lx, Ly = s_x.shape[0], s_y.shape[0]
    assert Lx == 4 and Ly == 4
    # An edge ((1, 1), (3, 4)) connects X-pos 1 (col 1) to X-pos 3 (col 4) on
    # the X axis, and Y-pos 1 (col 1) to Y-pos 4 (col 5) on the Y axis.
    # Both edge_pair_x and edge_pair_y projections should produce:
    #   X: (1, 4) and Y: (1, 5).
    # Aggregated MSA-column-pair counts should be:
    #   {(1, 4): 1, (1, 5): 1}.
    edge_pair_x_counts = {(1, 3): 1}
    edge_pair_y_counts = {(1, 4): 1}
    L_msa = 5
    pos_to_col_a = np.zeros(Lx + 1, dtype=np.int32)
    for col_idx in range(c2p_x.shape[0]):
        p = int(c2p_x[col_idx])
        if 1 <= p <= Lx:
            pos_to_col_a[p] = col_idx
    pos_to_col_b = np.zeros(Ly + 1, dtype=np.int32)
    for col_idx in range(c2p_y.shape[0]):
        p = int(c2p_y[col_idx])
        if 1 <= p <= Ly:
            pos_to_col_b[p] = col_idx
    col_pair_counts: dict[tuple[int, int], int] = {}
    for (i1, i2), c in edge_pair_x_counts.items():
        col1, col2 = int(pos_to_col_a[i1]), int(pos_to_col_a[i2])
        if col1 == 0 or col2 == 0 or col1 == col2:
            continue
        key = (min(col1, col2), max(col1, col2))
        col_pair_counts[key] = col_pair_counts.get(key, 0) + c
    for (j1, j2), c in edge_pair_y_counts.items():
        col1, col2 = int(pos_to_col_b[j1]), int(pos_to_col_b[j2])
        if col1 == 0 or col2 == 0 or col1 == col2:
            continue
        key = (min(col1, col2), max(col1, col2))
        col_pair_counts[key] = col_pair_counts.get(key, 0) + c
    # X-pos 1 -> col 1; X-pos 3 -> col 4.  Y-pos 1 -> col 1; Y-pos 4 -> col 5.
    assert col_pair_counts == {(1, 4): 1, (1, 5): 1}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
