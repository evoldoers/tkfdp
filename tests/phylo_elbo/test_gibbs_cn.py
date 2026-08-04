"""Test the c_n Gibbs sweep (M6).

Synthetic: create a 3-class model with well-separated per-class
stationaries. Simulate leaf observations by drawing each site's
residue from pi_field[true_c_n, theta_root, :] (with theta_root
sampled per cluster). Start c_n at randomly-shuffled values, run a
few Gibbs sweeps, expect classes to converge toward the true labels.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_gibbs_cn_moves_toward_true():
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_cherry
    from tkfdp.coupling.dynfield.phylo_elbo.gibbs_cn import (
        gibbs_cn_sweep)

    rng = np.random.default_rng(7)
    L = 2
    K_c = 3
    A = 4
    n_clusters = 20
    m = 4        # per cluster
    rho_chain = 0.05

    # Very sharp per-class equilibria (deltas at 3 different residues).
    pi_field = np.zeros((K_c, L, A), dtype=np.float64)
    for c in range(K_c):
        for l in range(L):
            base = np.full(A, 0.05)
            base[c] = 1.0 - 0.05 * (A - 1)
            pi_field[c, l] = base

    rho = np.full(L, 1.0 / L)
    S = np.ones((A, A)) - np.eye(A)

    # Synthesize corpus + true classes per cluster.
    clusters = []
    true_classes_list = []
    for _ in range(n_clusters):
        true_classes = rng.integers(0, K_c, size=m).astype(np.int32)
        theta_root = int(rng.choice(L, p=rho))
        # Draw same residue at both leaves from pi_field[true_c_n, theta_root].
        leaf_obs = np.zeros((2, m), dtype=np.int32)
        for n in range(m):
            c = int(true_classes[n])
            a = int(rng.choice(A, p=pi_field[c, theta_root]))
            leaf_obs[:, n] = a
        tree = make_cherry(tau=0.1, leaf_obs=leaf_obs)
        # Start with a RANDOM shuffled class assignment (not true).
        init_classes = rng.integers(0, K_c, size=m).astype(np.int32)
        clusters.append((tree, init_classes))
        true_classes_list.append(true_classes)

    def frac_correct(clusters_now):
        total = 0
        correct = 0
        for i in range(len(clusters_now)):
            classes_now = clusters_now[i][1]
            total += m
            correct += int((classes_now == true_classes_list[i]).sum())
        return correct / total

    frac_before = frac_correct(clusters)
    print(f"  before: frac_correct = {frac_before:.3f}")

    for sweep in range(3):
        new_classes, info = gibbs_cn_sweep(
            clusters, pi_field, rho, S, rho_chain, K_c, rng)
        clusters = [(clusters[i][0], new_classes[i])
                     for i in range(len(clusters))]
        f = frac_correct(clusters)
        print(f"  sweep {sweep + 1}: frac_correct = {f:.3f} "
                f"changed={info['n_changed']}/{info['n_sites_total']}")

    f_final = frac_correct(clusters)
    assert f_final > 0.65, \
        f"frac_correct {f_final} <= 0.65 after 3 sweeps"


if __name__ == "__main__":
    test_gibbs_cn_moves_toward_true()
    print("M6 test PASS")
