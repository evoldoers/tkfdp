"""Verify MM tree log-lik in the two exact-recovery limits.

At rho_chain -> 0 (no field jumps): the field is drawn once at the
root from rho and never changes. Conditional on theta_root = theta,
the m columns are independent, each evolving under the fixed archetype
pi_field(c_n, theta) via GTR with substitution matrix
P_subst(c_n, theta; t) = exp(Q(c_n, theta) * t) on each edge. The block
log-likelihood is therefore

    log P(observations | tree, model) = log sum_theta rho[theta]
        * prod_n P_col(x_col_n | tree, c_n, theta)

with each per-column Felsenstein computed by the standard leaf-up
pruning. This test compares mm_tree_log_lik at rho_chain = 1e-8 (a
numerical approximation of the limit) against the exact per-column
Felsenstein computation on a balanced-binary depth-2 (4 leaves) tree.
"""
import numpy as np
import pytest


def per_column_felsenstein(tree, class_c: int, theta: int,
                              obs: np.ndarray, pi_field: np.ndarray,
                              Q: np.ndarray) -> float:
    """Standard Felsenstein: log P(column observations | tree, class,
    theta). Root marginalized against pi_field(c, theta).

    obs: (n_leaves,) int32.
    """
    from scipy.linalg import expm
    A = pi_field.shape[2]
    L = pi_field.shape[1]

    def clv(v: int) -> np.ndarray:
        if tree.is_leaf(v):
            v_clv = np.zeros(A)
            x = int(obs[v])
            if x >= 0:
                v_clv[x] = 1.0
            else:
                v_clv[:] = 1.0                            # gap uninformative
            return v_clv
        result = np.ones(A)
        for c in tree.children[v]:
            c_clv = clv(c)
            tau_c = float(tree.branch_length[c])
            P = expm(Q[class_c, theta] * tau_c)
            result *= P @ c_clv
        return result

    root_clv = clv(tree.root)
    return float(np.log(np.dot(pi_field[class_c, theta], root_clv)))


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_mm_matches_per_column_felsenstein_at_rho_chain_zero(depth):
    """At rho_chain -> 0, block log-lik = sum-log-column-marginal-theta."""
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import per_class_field_Q
    from tkfdp.coupling.dynfield.phylo_elbo.tree_log_lik import tree_log_lik_mm
    from tkfdp.coupling.dynfield.phylo_elbo.tree import make_balanced_binary

    rng = np.random.default_rng(0)
    A = 4
    L = 3
    K_c = 2
    m = 2
    n_leaves = 2 ** depth

    pi_field = rng.dirichlet(np.ones(A), size=(K_c, L))
    S_raw = rng.uniform(0.5, 2.0, size=(A, A))
    S = (S_raw + S_raw.T) / 2
    np.fill_diagonal(S, 0.0)
    rho = rng.dirichlet(np.full(L, 2.0))
    Q = per_class_field_Q(pi_field, S)
    classes = rng.integers(0, K_c, size=m).astype(np.int32)
    leaf_obs = rng.integers(0, A, size=(n_leaves, m)).astype(np.int32)
    tau = 0.4

    tree = make_balanced_binary(depth=depth, tau=tau, leaf_obs=leaf_obs)

    rho_chain = 1e-8                                        # -> 0 limit
    mm_ll = tree_log_lik_mm(tree, classes, rho, pi_field, S,
                                rho_chain=rho_chain)

    # Reference: block LL = log sum_theta rho[theta]
    #            * prod_n exp(per_column_felsenstein(tree, c_n, theta))
    per_theta = np.zeros(L)
    for th in range(L):
        log_col_prod = 0.0
        for n in range(m):
            log_col_prod += per_column_felsenstein(
                tree, int(classes[n]), th, leaf_obs[:, n], pi_field, Q)
        per_theta[th] = log_col_prod
    # Add log rho[theta] and log-sum-exp.
    max_log = np.max(per_theta + np.log(rho + 1e-300))
    ref_ll = float(max_log + np.log(
        np.sum(np.exp(per_theta + np.log(rho + 1e-300) - max_log))))

    print(f"depth={depth}  mm={mm_ll:.6f}  ref={ref_ll:.6f}  "
            f"diff={mm_ll - ref_ll:+.2e}")
    assert np.isclose(mm_ll, ref_ll, atol=1e-4, rtol=1e-6), (
        f"MM tree log-lik at rho_chain=1e-8 depth={depth}: "
        f"{mm_ll} != {ref_ll}")


if __name__ == "__main__":
    for d in [1, 2, 3]:
        test_mm_matches_per_column_felsenstein_at_rho_chain_zero(d)
        print(f"  depth={d}: PASS")
