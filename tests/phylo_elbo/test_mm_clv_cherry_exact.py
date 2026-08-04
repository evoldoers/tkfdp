"""Verify MM ELBO recovers exact cherry likelihood.

At depth 1 (cherry), the moment-matching projection is exact. The
tree-ELBO log-likelihood via mm_edge / mm_mass_two on a 2-leaf cherry
must match the direct pair-marginal computation

    P((X, Y) | classes, tau)
      = sum_theta_v rho[theta_v] * prod_n sum_theta_u prod_n [
          K(x_n_obs_a, theta_u | x_n_obs_b, theta_v; tau)
          * K(...) ]

For a single cluster of m columns evaluated on a cherry of two leaves
this reduces to (by pulley-rooting at one leaf):

    P((X_a, X_b) | classes, tau, theta_v)
      = beta(theta_v) * prod_n P^{c_n, theta_v}(x_a_n, x_b_n; tau)
      + sum_{theta_u} W(theta_v, theta_u; tau)
                       * prod_n pi_field(c_n, theta_u)(x_b_n)   (jump case)
    x pi_field-marginal of x_a  (root emission at theta_v)

which is exactly what mm_mass_two integrates against rho.

This test builds a small (m=2, L=3, A=4) toy model, runs the MM
pipeline on a cherry, and compares to the direct enumeration.
"""
import numpy as np
import pytest


@pytest.mark.parametrize("rho_chain", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("tau", [0.1, 0.5, 1.5])
def test_mm_cherry_matches_direct_enumeration(rho_chain, tau):
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        MMClv, leaf_clv, mm_edge, mm_mass_two, per_class_field_Q,
        branch_P, beta_no_jump, field_transition_row)

    rng = np.random.default_rng(0)
    A = 4                              # 4-letter alphabet
    L = 3                              # 3 field states
    K_c = 2                            # 2 site classes
    m = 2                              # 2-column cluster

    # Random simplex per (c, theta).
    pi_field = rng.dirichlet(np.ones(A), size=(K_c, L))
    # Symmetric exchangeability with zero diagonal.
    S_raw = rng.uniform(0.5, 2.0, size=(A, A))
    S = (S_raw + S_raw.T) / 2
    np.fill_diagonal(S, 0.0)
    # Field TSB weights.
    rho = rng.dirichlet(np.full(L, 2.0))

    Q = per_class_field_Q(pi_field, S)

    # Observations: 2 leaves x 2 columns, in [0, A).
    x_a = rng.integers(0, A, size=m).astype(np.int32)
    x_b = rng.integers(0, A, size=m).astype(np.int32)
    classes = rng.integers(0, K_c, size=m).astype(np.int32)

    # ----- Direct enumeration -----
    # Pulley-root at leaf a: theta_v = theta at leaf a. Single branch
    # of length tau from a to b.
    #   P(observed x_a, x_b at leaf b | model)
    #     = sum_{theta_v} rho[theta_v] * pi_field(c_n, theta_v)(x_a_n)
    #                     * [ beta * prod_n P(x_a_n, x_b_n; tau)
    #                         + sum_{theta_u} W(theta_v, theta_u; tau)
    #                                        * prod_n pi_field(c_n, theta_u)(x_b_n) ]
    direct = 0.0
    for th_v in range(L):
        # Leaf-a emission at theta_v: prod_n pi_field(c_n, theta_v)(x_a_n)
        pi_a = 1.0
        for n in range(m):
            pi_a *= pi_field[classes[n], th_v, x_a[n]]
        # Case 0: no jump, theta_v = theta_a = theta_b
        b_prob = beta_no_jump(rho, th_v, tau, rho_chain)
        no_jump_prod = 1.0
        for n in range(m):
            P = branch_P(Q, classes[n], th_v, tau)
            no_jump_prod *= P[x_a[n], x_b[n]]
        no_jump = b_prob * no_jump_prod
        # Case >=1: at least one jump, theta_b = theta_u drawn from
        # the jump distribution.
        jump_total = 0.0
        pf_row = field_transition_row(rho, th_v, tau, rho_chain)
        for th_u in range(L):
            w = pf_row[th_u] - (b_prob if th_u == th_v else 0.0)
            prod_pi = 1.0
            for n in range(m):
                prod_pi *= pi_field[classes[n], th_u, x_b[n]]
            jump_total += w * prod_pi
        direct += rho[th_v] * pi_a * (no_jump + jump_total)
    direct_ll = float(np.log(direct))

    # ----- Pulley-rooted MM: single branch, leaf X at root, leaf Y  -----
    # For the pulley-rooted cherry, we don't use mm_mass_two directly
    # because that assumes 2 children of a root; here it's 1 branch.
    # We use the direct formula above as the ground truth and verify
    # by building the equivalent (2-child) cherry through moment-
    # matched double-branch, which should agree in the limit.
    #
    # Simpler approach: build a 2-child cherry with two branches of
    # length tau/2 each, and observe that the total likelihood
    # matches an equivalent single-branch of length tau. Under a
    # reversible F81-on-DP field and reversible GTR per arch, this
    # equivalence holds because the joint stationary factors and the
    # midpoint theta_v is integrated against rho consistently.
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import mm_rescale
    obs_a = x_a
    obs_b = x_b
    tau_half = tau / 2.0
    clv_a = leaf_clv(obs_a, classes, pi_field)
    clv_b = leaf_clv(obs_b, classes, pi_field)
    msg_a = mm_edge(clv_a, classes, tau_half, rho, pi_field, Q, rho_chain)
    msg_b = mm_edge(clv_b, classes, tau_half, rho, pi_field, Q, rho_chain)
    mass, log_scale = mm_mass_two(msg_a, msg_b, classes, rho, pi_field)
    mm_ll = float(np.log(mass)) + log_scale

    # Expect agreement to double precision (the MM is exact at cherries
    # per evolmoves docs; midpoint rooting is equivalent to single-branch
    # rooting under reversibility, so the two log-liks should match).
    assert np.isfinite(direct_ll) and np.isfinite(mm_ll), (
        f"log-lik must be finite: direct={direct_ll}, mm={mm_ll}")
    # Verified: they should agree tightly under reversibility.
    print(f"direct_ll={direct_ll:.8f}, mm_ll={mm_ll:.8f}, "
            f"diff={mm_ll - direct_ll:+.2e}")
    # Allow up to 1e-8 relative tolerance for double-precision expm.
    assert np.isclose(mm_ll, direct_ll, atol=1e-8, rtol=1e-8), (
        f"MM cherry log-lik {mm_ll} != direct {direct_ll}")


if __name__ == "__main__":
    for rc in [0.05, 0.5, 2.0]:
        for tau in [0.1, 0.5, 1.5]:
            test_mm_cherry_matches_direct_enumeration(rc, tau)
            print(f"  rho_chain={rc}, tau={tau}: PASS")
