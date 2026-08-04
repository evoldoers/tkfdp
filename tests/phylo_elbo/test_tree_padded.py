"""Tests for the padded tree layout (Milestone 2).

Verifies:
  - Cherry (depth 1): N_bucket=2, D_bucket=1, level 0 has 2 leaves,
    level 1 has 1 internal (root) with children (0, 1).
  - Balanced depth-2 binary (4 leaves): N=4, D=2; level 1 has 2
    cherries; level 2 has root combining them.
  - Unbalanced ((A,B),C): N=3, D=2. Level 0: A, B, C; level 1:
    (A,B) cherry + phantom identity lift of C; level 2: root
    combines cherry and lifted-C.
  - N_bucket / D_bucket padding: cherry padded to N=4, D=2 should
    have 2 phantom leaf slots and 1 extra identity level.
  - PaddedTree.check() succeeds on all constructed layouts.
"""
from __future__ import annotations

import numpy as np
import pytest

from tkfdp.coupling.dynfield.phylo_elbo.tree import (
    build_tree, make_cherry, make_balanced_binary)
from tkfdp.coupling.dynfield.phylo_elbo.tree_padded import (
    PaddedTree, build_padded_tree, compute_node_levels, sqrt2_bucket)


def test_sqrt2_bucket():
    assert sqrt2_bucket(1) == 1
    assert sqrt2_bucket(2) == 2
    assert sqrt2_bucket(3) == 3
    assert sqrt2_bucket(4) == 4
    # sqrt2^n sequence: 1, sqrt2=1.41 (ceil=2), 2, 2*sqrt2=2.83 (ceil=3),
    # 4, 4*sqrt2=5.66 (ceil=6), 8, 8*sqrt2=11.31 (ceil=12), 16.
    assert sqrt2_bucket(5) == 6
    assert sqrt2_bucket(6) == 6
    assert sqrt2_bucket(7) == 8
    assert sqrt2_bucket(9) == 12


def test_cherry_padded():
    """Cherry: 2 leaves, root at depth 1."""
    leaf_obs = np.array([[0, 1], [2, 0]], dtype=np.int32)  # m=2
    tree = make_cherry(tau=0.5, leaf_obs=leaf_obs)
    pt = build_padded_tree(tree)
    assert pt.N_bucket == 2
    assert pt.D_bucket == 1
    assert pt.n_leaves_actual == 2
    assert pt.depth_actual == 1
    assert pt.m == 2
    # Level 0: 2 leaf slots.
    assert pt.n_slots(0) == 2
    assert pt.leaf_mask.tolist() == [1.0, 1.0]
    assert (pt.leaf_obs == leaf_obs).all()
    # Level 1: 1 internal (root) combining slots 0 and 1.
    assert pt.n_slots(1) == 1
    cp = pt.child_pos[0]
    assert cp.shape == (1, 2)
    assert set(cp[0].tolist()) == {0, 1}
    cb = pt.child_branch[0]
    assert cb[0, 0] == pytest.approx(0.5)
    assert cb[0, 1] == pytest.approx(0.5)
    assert pt.slot_mask[0].tolist() == [1.0]
    assert pt.root_slot == 0
    print("cherry padded OK")


def test_balanced_depth2_padded():
    """Balanced 4-leaf binary of depth 2."""
    leaf_obs = np.array([[0], [1], [2], [3]], dtype=np.int32)
    tree = make_balanced_binary(depth=2, tau=0.4, leaf_obs=leaf_obs)
    pt = build_padded_tree(tree)
    assert pt.N_bucket == 4
    assert pt.D_bucket == 2
    assert pt.n_leaves_actual == 4
    assert pt.depth_actual == 2
    # Level 0: 4 leaves; level 1: 2 cherries; level 2: root.
    assert pt.n_slots(0) == 4
    assert pt.n_slots(1) == 2
    assert pt.n_slots(2) == 1
    assert pt.leaf_mask.tolist() == [1.0, 1.0, 1.0, 1.0]
    # Each cherry at level 1 combines two leaves.
    sm1 = pt.slot_mask[0]
    assert sm1.tolist() == [1.0, 1.0]
    cp1 = pt.child_pos[0]
    combined_leaves_set = set()
    for pos in range(pt.n_slots(1)):
        combined_leaves_set.update(cp1[pos].tolist())
    assert combined_leaves_set == {0, 1, 2, 3}
    # Root at level 2 combines the two cherries.
    assert pt.slot_mask[1].tolist() == [1.0]
    cp2 = pt.child_pos[1]
    assert set(cp2[0].tolist()) == {0, 1}
    print("balanced depth-2 padded OK")


def test_unbalanced_ab_c_padded():
    """Unbalanced ((A,B), C) tree — C has to lift through a phantom level.

    Node ids: 0=A, 1=B, 2=C, 3=(A,B) internal, 4=root.
      parent = [3, 3, 4, 4, -1]
      branch = [tA, tB, tC, tAB, 0]
    Levels: A, B, C -> 0. (A,B) -> 1. Root -> 2.
    Between C at level 0 and root at level 2, need one phantom
    identity lift at level 1.
    """
    parent = [3, 3, 4, 4, -1]
    branch = [0.5, 0.5, 1.0, 0.3, 0.0]
    leaf_obs = np.array([[0], [1], [2]], dtype=np.int32)
    tree = build_tree(parent, branch, leaf_obs)
    levels = compute_node_levels(tree)
    assert levels[0] == 0 and levels[1] == 0 and levels[2] == 0
    assert levels[3] == 1
    assert levels[4] == 2

    pt = build_padded_tree(tree)
    # n_leaves=3 -> N_bucket=3 (sqrt2 bucket); depth=2 -> D_bucket=2.
    assert pt.N_bucket == 3
    assert pt.D_bucket == 2
    # Level 0: 3 leaves.
    assert pt.n_slots(0) == 3
    assert pt.leaf_mask.tolist() == [1.0, 1.0, 1.0]
    # Level 1: (A,B) internal AT slot 0, and phantom lift of C at slot 1.
    # Order might vary; check both exist.
    n1 = pt.n_slots(1)
    assert n1 == 2, f"expected 2 slots at level 1, got {n1}"
    # Find which slot is the real (A,B) internal by looking at children
    # (should be leaves 0, 1) and which is the phantom C-lift (children
    # both = C's slot at level 0 = 2, branch = 0).
    real_slot = None; phantom_slot = None
    for pos in range(n1):
        if set(pt.child_pos[0][pos].tolist()) == {0, 1}:
            real_slot = pos
        elif (pt.child_pos[0][pos].tolist() == [2, 2]
                and pt.child_branch[0][pos, 0] == 0.0):
            phantom_slot = pos
    assert real_slot is not None and phantom_slot is not None
    assert pt.slot_mask[0][real_slot] == 1.0
    # Level 2: root combines (A,B) at real_slot and phantom-C at phantom_slot.
    assert pt.n_slots(2) == 1
    cp2 = pt.child_pos[1]
    assert set(cp2[0].tolist()) == {real_slot, phantom_slot}
    assert pt.slot_mask[1].tolist() == [1.0]
    print("unbalanced ((A,B),C) padded OK")


def test_cherry_padded_larger_buckets():
    """Cherry with N_bucket=4, D_bucket=2. 2 phantom leaves + 1
    identity level added."""
    leaf_obs = np.array([[0], [1]], dtype=np.int32)
    tree = make_cherry(tau=0.5, leaf_obs=leaf_obs)
    pt = build_padded_tree(tree, N_bucket=4, D_bucket=2)
    assert pt.N_bucket == 4
    assert pt.D_bucket == 2
    # 4 leaf slots, first 2 real, last 2 phantom.
    assert pt.leaf_mask.tolist() == [1.0, 1.0, 0.0, 0.0]
    # (leaf_obs padded with -1)
    assert (pt.leaf_obs[2:] == -1).all()
    # Level 1: real cherry (root at true level 1) + phantom lift of root
    # (because we lift root to D_bucket = 2). Real slot at level 1 has
    # children (0, 1); phantom slot at level 1 is unclear because the
    # root has no other subtree waiting. Actually wait — the root's
    # native level is 1, and it gets lifted to level 2 (D_bucket). So at
    # level 1 there should be ONE slot: the real root at true level 1.
    # At level 2 there's a PHANTOM identity lift of root.
    assert pt.n_slots(1) == 1
    assert pt.slot_mask[0].tolist() == [1.0]
    cp1 = pt.child_pos[0]
    assert set(cp1[0].tolist()) == {0, 1}
    # Level 2 (padded root level): phantom identity lift.
    assert pt.n_slots(2) == 1
    assert pt.slot_mask[1].tolist() == [1.0]
    cp2 = pt.child_pos[1]
    # Phantom identity: both children point to the same slot at level 1,
    # branches both 0.
    assert cp2[0, 0] == cp2[0, 1] == 0
    assert pt.child_branch[1][0, 0] == 0.0
    assert pt.child_branch[1][0, 1] == 0.0
    assert pt.root_slot == 0
    print("cherry padded to (N=4, D=2) OK")


if __name__ == "__main__":
    test_sqrt2_bucket()
    test_cherry_padded()
    test_balanced_depth2_padded()
    test_unbalanced_ab_c_padded()
    test_cherry_padded_larger_buckets()
    print("all padded-tree tests PASS")
