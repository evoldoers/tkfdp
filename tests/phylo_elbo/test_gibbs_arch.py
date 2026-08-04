"""Test the arch_assignment Gibbs sweep (M5).

Setup: synthesize a small corpus where the "true" arch_assignment[c, θ]
is a known non-identity permutation for θ != 0. Draw pi_archetype
as random K_a=3 simplex points that differ noticeably. Simulate leaf
observations that consistently reflect the true arch_assignment
(each leaf's residues drawn from the archetype's stationary at the
class assigned by the true arch).

Then start Gibbs from the identity arch_assignment (arch[c, θ] = c for
all c, θ) and run a few sweeps. Expected: arch_assignment drifts
toward the true mapping (or a permutation-equivalent).
"""
from __future__ import annotations

import numpy as np
import pytest


def _simulate_leaf_obs_from_arch(rng, pi_archetype, arch_assignment,
                                     rho, tree, classes, rho_chain,
                                     n_repeats=1):
    """Simulate leaf observations by drawing directly from the equilibrium
    at (c, theta_root) sampled per cluster. Simplified: for each cluster,
    sample theta_root ~ rho, then per site n draw the same residue at
    every leaf from pi^{arch(c_n, theta_root)}.

    This creates strongly-informative observations that will pin the
    arch_assignment posterior sharply on the true mapping.
    """
    K_a, A = pi_archetype.shape
    K_c, L = arch_assignment.shape
    N_leaves = tree.n_leaves
    m = int(classes.shape[0])

    theta_root = int(rng.choice(L, p=rho))
    leaf_obs = np.zeros((N_leaves, m), dtype=np.int32)
    for n in range(m):
        c = int(classes[n])
        k = int(arch_assignment[c, theta_root])
        # Draw the SAME residue at every leaf from pi_arch[k, :].
        # This makes the site strongly informative (no substitution).
        a = int(rng.choice(A, p=pi_archetype[k]))
        leaf_obs[:, n] = a
    return leaf_obs


def test_gibbs_arch_moves_toward_true():
    from tkfdp.coupling.dynfield.phylo_elbo.tree import (
        make_balanced_binary, make_cherry)
    from tkfdp.coupling.dynfield.phylo_elbo.gibbs_arch import (
        gibbs_arch_assignment_sweep)

    rng = np.random.default_rng(42)
    L = 3
    K_c = 3
    K_a = 3
    A = 4

    # Sharp archetypes: 3 well-separated simplex points.
    pi_archetype = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
    ])
    rho_arch = np.full(K_a, 1.0 / K_a)

    # True arch_assignment: identity at theta=0, cyclic shift at theta=1,
    # 2-shift at theta=2.
    true_arch = np.array([
        [0, 1, 2],       # class 0
        [1, 2, 0],       # class 1
        [2, 0, 1],       # class 2
    ], dtype=np.int32)

    # Uniform rho (equal prob per theta).
    rho = np.full(L, 1.0 / L)
    # Uniform S (F81 substitution).
    S = np.ones((A, A)) - np.eye(A)
    rho_chain = 0.05  # low jump rate: theta strongly correlated across tree

    # Synthesize a corpus: 20 clusters, mostly cherries + a few depth-2.
    clusters = []
    for _ in range(20):
        # Random tree shape.
        shape = rng.choice(['cherry', 'depth2'])
        m = int(rng.integers(1, 4))
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
        if shape == 'cherry':
            leaf_obs = _simulate_leaf_obs_from_arch(
                rng, pi_archetype, true_arch, rho,
                make_cherry(tau=0.1, leaf_obs=np.zeros((2, m), dtype=np.int32)),
                classes, rho_chain)
            tree = make_cherry(tau=0.1, leaf_obs=leaf_obs)
        else:
            leaf_obs = _simulate_leaf_obs_from_arch(
                rng, pi_archetype, true_arch, rho,
                make_balanced_binary(
                    depth=2, tau=0.1,
                    leaf_obs=np.zeros((4, m), dtype=np.int32)),
                classes, rho_chain)
            tree = make_balanced_binary(depth=2, tau=0.1, leaf_obs=leaf_obs)
        clusters.append((tree, classes))

    # Start Gibbs from identity arch_assignment.
    aa = np.tile(np.arange(K_c)[:, None], (1, L)).astype(np.int32)
    # But then theta=0 is identity by design; the sweep is on theta != 0.

    for sweep in range(3):
        aa, info = gibbs_arch_assignment_sweep(
            clusters, pi_archetype, aa, rho_arch, rho, S, rho_chain,
            rng, fix_theta0=True)
        matches = int((aa == true_arch).sum())
        print(f"  sweep {sweep + 1}: aa=\n{aa}\n  matches "
                f"true={matches}/{K_c * L}  changed={info['n_changed']}")

    # After 3 sweeps we should have moved substantially toward the true.
    # Not necessarily exact match (permutation symmetry, finite sample),
    # but the fraction of correct entries at theta != 0 should be > 1/K_a
    # (random chance).
    aa_nonzero_theta = aa[:, 1:]
    true_nonzero_theta = true_arch[:, 1:]
    frac_correct = float((aa_nonzero_theta == true_nonzero_theta).sum()) / (
        K_c * (L - 1))
    print(f"  final: frac_correct at theta != 0 = {frac_correct:.2f} "
            f"(random = {1 / K_a:.2f})")
    assert frac_correct > 1.5 / K_a, \
        f"frac_correct {frac_correct} <= {1.5 / K_a} = 1.5/K_a"


if __name__ == "__main__":
    test_gibbs_arch_moves_toward_true()
    print("M5 test PASS")
