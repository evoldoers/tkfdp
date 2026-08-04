"""Depth-2 tests for the MM CLV pipeline.

Layout:
      R (root, hidden)
     / tau_R           \\ tau_R
    A (internal)         B (internal)
    | tau_A               | tau_B
    a1 a2                 b1 b2

Two flavours:
  test_depth2_rho_chain_zero_exact:
    At rho_chain = 0 the field never jumps, so s stays 0 throughout
    the pipeline. mm_combine's four cross-terms reduce to the single
    rank-1 product r_1 r_2 * prod A_1 A_2 which IS exactly in the
    family. mm_edge is also exact under the (r * prod A + s * 1)
    family. Result: mm matches direct enumeration to double precision.

  test_depth2_rho_chain_positive_bounded:
    At rho_chain > 0, mm_combine at internal nodes performs a genuine
    moment-matching projection (three rank-1 factors merge into one),
    so mm cannot equal direct exactly. Bound the discrepancy at 1%
    log-lik as a regression check for the marg_child fix (pre-fix it
    was >6% total-theta-mass at depth 2). This bound is the observed
    mm_combine projection error, NOT an artefact of the mm_edge fix.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest


def _cluster_kernel(pi_field, rho, rho_chain, classes, tau,
                        th_p, th_c, x_p, x_c, Q):
    """One-branch cluster kernel."""
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        beta_no_jump, branch_P, field_transition_row)
    m = int(classes.shape[0])
    b = beta_no_jump(rho, th_p, tau, rho_chain)
    no_jump = 0.0
    if th_p == th_c:
        prod_P = 1.0
        for n in range(m):
            P = branch_P(Q, int(classes[n]), th_p, tau)
            prod_P *= P[x_p[n], x_c[n]]
        no_jump = b * prod_P
    pf_row = field_transition_row(rho, th_p, tau, rho_chain)
    w = pf_row[th_c] - (b if th_c == th_p else 0.0)
    prod_pi = 1.0
    for n in range(m):
        prod_pi *= pi_field[int(classes[n]), th_c, x_c[n]]
    return no_jump + w * prod_pi


def _run_depth2(rho_chain, tau_R, tau_A, tau_B, rng_seed=11):
    """Compute (mm_ll, direct_ll) for a random depth-2 tree."""
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        leaf_clv, mm_combine, mm_edge, mm_mass_two, per_class_field_Q)

    rng = np.random.default_rng(rng_seed)
    L, m, A_alpha, K_c = 2, 2, 3, 1

    pi_field = rng.dirichlet(np.ones(A_alpha), size=(K_c, L))
    S_raw = rng.uniform(0.5, 2.0, size=(A_alpha, A_alpha))
    S = (S_raw + S_raw.T) / 2
    np.fill_diagonal(S, 0.0)
    rho = rng.dirichlet(np.full(L, 2.0))
    classes = np.zeros(m, dtype=np.int32)
    Q = per_class_field_Q(pi_field, S)

    x_a1 = rng.integers(0, A_alpha, size=m).astype(np.int32)
    x_a2 = rng.integers(0, A_alpha, size=m).astype(np.int32)
    x_b1 = rng.integers(0, A_alpha, size=m).astype(np.int32)
    x_b2 = rng.integers(0, A_alpha, size=m).astype(np.int32)

    # ----- Direct enumeration -----
    def leaf_emit(x_parent, th_parent, x_leaf, tau):
        total = 0.0
        for th_leaf in range(L):
            total += _cluster_kernel(
                pi_field, rho, rho_chain, classes, tau,
                th_parent, th_leaf, x_parent, x_leaf, Q)
        return total

    def subtree_emit(x_root, th_root, x1, x2, tau_up, tau_leaf):
        total = 0.0
        for th_int in range(L):
            for x_int in itertools.product(range(A_alpha), repeat=m):
                Kv = _cluster_kernel(
                    pi_field, rho, rho_chain, classes, tau_up,
                    th_root, th_int, x_root, x_int, Q)
                total += Kv * leaf_emit(x_int, th_int, x1, tau_leaf) \
                             * leaf_emit(x_int, th_int, x2, tau_leaf)
        return total

    direct = 0.0
    for th_R in range(L):
        for x_R in itertools.product(range(A_alpha), repeat=m):
            pi_R = float(rho[th_R])
            for n in range(m):
                pi_R *= pi_field[int(classes[n]), th_R, x_R[n]]
            direct += pi_R \
                    * subtree_emit(x_R, th_R, x_a1, x_a2, tau_R, tau_A) \
                    * subtree_emit(x_R, th_R, x_b1, x_b2, tau_R, tau_B)
    direct_ll = float(np.log(direct))

    # ----- MM pipeline -----
    clv_a1 = leaf_clv(x_a1, classes, pi_field)
    clv_a2 = leaf_clv(x_a2, classes, pi_field)
    clv_b1 = leaf_clv(x_b1, classes, pi_field)
    clv_b2 = leaf_clv(x_b2, classes, pi_field)
    msg_a1 = mm_edge(clv_a1, classes, tau_A, rho, pi_field, Q, rho_chain)
    msg_a2 = mm_edge(clv_a2, classes, tau_A, rho, pi_field, Q, rho_chain)
    msg_b1 = mm_edge(clv_b1, classes, tau_B, rho, pi_field, Q, rho_chain)
    msg_b2 = mm_edge(clv_b2, classes, tau_B, rho, pi_field, Q, rho_chain)
    clv_A = mm_combine(msg_a1, msg_a2, classes, pi_field)
    clv_B = mm_combine(msg_b1, msg_b2, classes, pi_field)
    msg_A_up = mm_edge(clv_A, classes, tau_R, rho, pi_field, Q, rho_chain)
    msg_B_up = mm_edge(clv_B, classes, tau_R, rho, pi_field, Q, rho_chain)
    mm_mass, log_scale = mm_mass_two(
        msg_A_up, msg_B_up, classes, rho, pi_field)
    mm_ll = float(np.log(mm_mass)) + log_scale

    return mm_ll, direct_ll


@pytest.mark.parametrize("tau_R,tau_A,tau_B",
                                       [(0.3, 0.2, 0.2), (0.5, 0.4, 0.1)])
def test_depth2_rho_chain_zero_exact(tau_R, tau_A, tau_B):
    """rho_chain = 0 -> W = 0 -> s stays 0 -> mm exact under family form."""
    mm_ll, direct_ll = _run_depth2(0.0, tau_R, tau_A, tau_B)
    print(f"[depth2 rho_c=0 tau_R={tau_R} tau_A={tau_A} tau_B={tau_B}] "
            f"direct={direct_ll:+.10f} mm={mm_ll:+.10f} "
            f"diff={mm_ll - direct_ll:+.2e}")
    assert np.isclose(mm_ll, direct_ll, atol=1e-9, rtol=1e-9), \
        f"rho_chain=0 mm should be exact: MM {mm_ll} vs direct {direct_ll}"


@pytest.mark.parametrize("rho_chain", [0.1, 0.5, 1.5])
@pytest.mark.parametrize("tau_R,tau_A,tau_B",
                                       [(0.3, 0.2, 0.2), (0.5, 0.4, 0.1)])
def test_depth2_rho_chain_positive_bounded(rho_chain, tau_R, tau_A, tau_B):
    """rho_chain > 0: mm has mm_combine projection error, should be <1%
    log-lik (regression check for the marg_child fix — pre-fix was
    ~6% total-theta-mass at depth 2)."""
    mm_ll, direct_ll = _run_depth2(rho_chain, tau_R, tau_A, tau_B)
    diff = mm_ll - direct_ll
    print(f"[depth2 rho_c={rho_chain} tau_R={tau_R} tau_A={tau_A} "
            f"tau_B={tau_B}] direct={direct_ll:+.10f} mm={mm_ll:+.10f} "
            f"diff={diff:+.2e}")
    assert abs(diff) < 1e-2, \
        f"mm depth-2 log-lik off by {diff} (> 1% bound); mm_combine " \
        f"projection error too large or marg_child fix regressed"


if __name__ == "__main__":
    for (tR, tA, tB) in [(0.3, 0.2, 0.2), (0.5, 0.4, 0.1)]:
        test_depth2_rho_chain_zero_exact(tR, tA, tB)
    for rc in [0.1, 0.5, 1.5]:
        for (tR, tA, tB) in [(0.3, 0.2, 0.2), (0.5, 0.4, 0.1)]:
            test_depth2_rho_chain_positive_bounded(rc, tR, tA, tB)
    print("all depth-2 MM tests PASS")
