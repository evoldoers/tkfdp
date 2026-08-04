"""Compare backward-sweep q_v(theta) against brute-force exact Felsenstein.

Directly enumerates all hidden states (theta at every node, x at every
internal node) on a small tree to compute the exact per-node posterior
q_v(theta_v) = P(theta_v | observations). Compares to
backward_theta_marginals output.

This bypasses the "all q_v = q_root at rho_chain=0" consistency check
and gives a per-node ground truth at ANY rho_chain, including regimes
where the moment-matching family IS exact (depth-1 cherries; any depth
at rho_chain=0 or rho_chain -> infty).

Setup for tractability: L=2 field states, A=3 alphabet, m=1 site,
K_c=1 class. Tree = balanced binary depth 1 or 2 (2 or 4 leaves).

At depth 1 cherry: mm_combine is exact (single sibling); backward should
match exactly.
At depth 2 with rho_chain=0: field never jumps, so posterior should
match exactly.
At depth 2 with rho_chain > 0: mm_combine at root uses a nontrivial
projection; some bias expected, quantify it.
"""
import itertools

import numpy as np
import pytest
from scipy.linalg import expm


def field_kernel(theta_v_from, theta_v_to, rho, rho_chain, tau):
    """Field CTMC transition P(theta_to | theta_from; tau)."""
    et = np.exp(-rho_chain * tau)
    return (et if theta_v_from == theta_v_to else 0.0) + (1 - et) * rho[theta_v_to]


def sub_matrix(pi_field, S, c, theta, tau):
    """Substitution matrix exp(Q(c, theta) * tau)."""
    A = pi_field.shape[2]
    Q = np.zeros((A, A))
    for a in range(A):
        for b in range(A):
            if a != b:
                Q[a, b] = S[a, b] * pi_field[c, theta, b]
        Q[a, a] = -Q[a, :].sum()
    return expm(Q * tau)


def compound_kernel(x_v, theta_v, x_u, theta_u, c, tau, pi_field, S,
                        rho, rho_chain):
    """K(x_u, theta_u | x_v, theta_v; tau) under Interp 2 for one site."""
    beta = np.exp(-rho_chain * (1 - rho[theta_v]) * tau)
    # No-jump term: theta_u = theta_v, substitution under Q(c, theta_v)
    P = sub_matrix(pi_field, S, c, theta_v, tau)
    no_jump = (beta if theta_u == theta_v else 0.0) * P[x_v, x_u]
    # Jump term: W(theta_v, theta_u; tau) * pi(x_u, theta_u)
    K_field = field_kernel(theta_v, theta_u, rho, rho_chain, tau)
    W = K_field - (beta if theta_u == theta_v else 0.0)
    jump = W * pi_field[c, theta_u, x_u]
    return no_jump + jump


def exact_posterior_over_theta(tree, classes, rho, pi_field, S, rho_chain):
    """Brute-force exact q_v(theta) at every node for m=1 site.

    Enumerates all (theta_all, x_internal_all) configurations and sums
    the joint P(obs, hidden) with the prior at root and per-branch
    kernels. Returns per-node q_v(theta) simplex (L,).
    """
    L = int(rho.shape[0])
    A_alph = pi_field.shape[2]
    m = int(classes.shape[0])
    assert m == 1, "exact enum specialised to m=1 for tractability"
    c = int(classes[0])

    n_nodes = tree.n_nodes
    root = tree.root
    is_leaf = np.array([tree.is_leaf(v) for v in range(n_nodes)])
    internal = [v for v in range(n_nodes) if not is_leaf[v]]
    n_internal = len(internal)

    # Joint per-node theta counter
    q_theta = np.zeros((n_nodes, L), dtype=np.float64)
    total_mass = 0.0

    # Iterate over all (theta_v for every v, x_v for every internal v)
    for theta_config in itertools.product(range(L), repeat=n_nodes):
        for x_int_config in itertools.product(range(A_alph),
                                                    repeat=n_internal):
            # Assemble the x-array: leaves have observed, internals from
            # x_int_config.
            x_arr = np.zeros(n_nodes, dtype=np.int64)
            for i_int, v in enumerate(internal):
                x_arr[v] = x_int_config[i_int]
            for v in range(n_nodes):
                if is_leaf[v]:
                    x_arr[v] = int(tree.leaf_obs[v, 0])

            # Compute joint probability
            # Prior at root: rho[theta_root] * pi(x_root, theta_root)
            th_r = int(theta_config[root])
            x_r = int(x_arr[root])
            joint_p = rho[th_r] * pi_field[c, th_r, x_r]

            # Each branch: multiply by K(child | parent; tau)
            for v in range(n_nodes):
                if v == root: continue
                p = int(tree.parent[v])
                tau_v = float(tree.branch_length[v])
                th_p = int(theta_config[p]); x_p = int(x_arr[p])
                th_v = int(theta_config[v]); x_v = int(x_arr[v])
                joint_p *= compound_kernel(x_p, th_p, x_v, th_v, c,
                                                tau_v, pi_field, S, rho,
                                                rho_chain)

            total_mass += joint_p
            for v in range(n_nodes):
                q_theta[v, theta_config[v]] += joint_p

    # Normalise per node
    for v in range(n_nodes):
        s = q_theta[v].sum()
        if s > 0:
            q_theta[v] /= s
    return q_theta


def _random_model(rng):
    A_alph = 3; L = 2; K_c = 1
    pi_field = rng.dirichlet(np.ones(A_alph), size=(K_c, L))
    S_raw = rng.uniform(0.5, 2.0, size=(A_alph, A_alph))
    S = (S_raw + S_raw.T) / 2
    np.fill_diagonal(S, 0.0)
    rho = rng.dirichlet(np.full(L, 2.0))
    return pi_field, S, rho, K_c, L, A_alph


@pytest.mark.parametrize("rho_chain", [0.0, 0.5, 2.0])
def test_cherry_backward_matches_exact(rho_chain):
    from tkfdp.coupling.dynfield.phylo_elbo.backward_sweep import (
        backward_theta_marginals)
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_cherry

    rng = np.random.default_rng(0)
    pi_field, S, rho, K_c, L, A_alph = _random_model(rng)
    classes = np.zeros(1, dtype=np.int32)
    leaf_obs = np.array([[0], [2]], dtype=np.int32)   # 2 leaves, m=1
    tree = make_cherry(tau=0.5, leaf_obs=leaf_obs)

    q_mm = backward_theta_marginals(tree, classes, rho, pi_field, S,
                                          rho_chain=rho_chain)
    q_exact = exact_posterior_over_theta(tree, classes, rho, pi_field,
                                                S, rho_chain=rho_chain)

    print(f"\ncherry (depth 1) rho_chain={rho_chain}:")
    for v in range(tree.n_nodes):
        diff = float(np.abs(q_mm[v] - q_exact[v]).max())
        print(f"  node {v}: mm={q_mm[v].round(4).tolist()}  "
                f"exact={q_exact[v].round(4).tolist()}  |diff|_max={diff:.2e}")
    max_all = max(float(np.abs(q_mm[v] - q_exact[v]).max())
                   for v in range(tree.n_nodes))
    # At depth 1, ELBO is exact -> match to machine precision.
    assert max_all < 1e-8, f"cherry backward max_diff {max_all:.2e}"


@pytest.mark.parametrize("rho_chain", [0.0, 0.5, 2.0])
def test_depth2_backward_matches_exact(rho_chain):
    from tkfdp.coupling.dynfield.phylo_elbo.backward_sweep import (
        backward_theta_marginals)
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_balanced_binary

    rng = np.random.default_rng(1)
    pi_field, S, rho, K_c, L, A_alph = _random_model(rng)
    classes = np.zeros(1, dtype=np.int32)
    leaf_obs = np.array([[0], [1], [2], [0]], dtype=np.int32)
    tree = make_balanced_binary(depth=2, tau=0.4, leaf_obs=leaf_obs)

    q_mm = backward_theta_marginals(tree, classes, rho, pi_field, S,
                                          rho_chain=rho_chain)
    q_exact = exact_posterior_over_theta(tree, classes, rho, pi_field,
                                                S, rho_chain=rho_chain)

    print(f"\ndepth 2 rho_chain={rho_chain}:")
    for v in range(tree.n_nodes):
        diff = float(np.abs(q_mm[v] - q_exact[v]).max())
        print(f"  node {v}: mm={q_mm[v].round(4).tolist()}  "
                f"exact={q_exact[v].round(4).tolist()}  |diff|_max={diff:.2e}")
    max_all = max(float(np.abs(q_mm[v] - q_exact[v]).max())
                   for v in range(tree.n_nodes))
    if rho_chain == 0.0:
        # ELBO exact at rho_chain=0 for any depth.
        assert max_all < 1e-8, (
            f"depth 2 backward at rho_chain=0 max_diff {max_all:.2e}")
    else:
        # Report only; controlled bias expected at intermediate rho_chain.
        print(f"  max diff at rho_chain={rho_chain}: {max_all:.4e}")


if __name__ == "__main__":
    for rc in [0.0, 0.5, 2.0]:
        try:
            test_cherry_backward_matches_exact(rc)
            print(f"cherry rho_chain={rc}: PASS")
        except AssertionError as e:
            print(f"cherry rho_chain={rc}: FAIL — {e}")
        try:
            test_depth2_backward_matches_exact(rc)
            print(f"depth 2 rho_chain={rc}: PASS")
        except AssertionError as e:
            print(f"depth 2 rho_chain={rc}: FAIL — {e}")
