"""Verify JAX mm_edge/mm_combine/mm_mass_* match the numpy reference.

Runs each JAX primitive on random inputs and compares to the numpy
counterpart in mm_clv.py to double precision.

Also runs the full cherry pipeline (leaf_clv_jax + mm_edge_jax x2 +
mm_mass_two_jax) against the numpy pipeline, expecting agreement to
double-precision (both are exact at cherries).
"""
from __future__ import annotations

import numpy as np
import pytest


def _random_family_clv(rng, L, m, A_alpha, pi_arch, positive_s=True):
    """Random (r, s, A) satisfying <pi_arch, A>_{l, n} = 1."""
    A = rng.uniform(0.5, 1.5, size=(m, L, A_alpha))
    inner = np.einsum('nla,nla->nl', pi_arch, A)  # <pi, A>_{n, l}
    A = A / inner[:, :, None]
    r = rng.uniform(0.3, 1.0, size=L)
    s = rng.uniform(0.3, 1.0, size=L) if positive_s else np.zeros(L)
    return {'r': r, 's': s, 'A': A}


def _make_random_model(rng, L, m, A_alpha, K_c, tau):
    """Random pi_field, S, rho, classes, and derived P_sub / pi_arch."""
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        per_class_field_Q, branch_P)
    pi_field = rng.dirichlet(np.ones(A_alpha), size=(K_c, L))
    S_raw = rng.uniform(0.5, 2.0, size=(A_alpha, A_alpha))
    S = (S_raw + S_raw.T) / 2
    np.fill_diagonal(S, 0.0)
    rho = rng.dirichlet(np.full(L, 2.0))
    if K_c == 1:
        classes = np.zeros(m, dtype=np.int32)
    else:
        classes = rng.integers(0, K_c, size=m).astype(np.int32)
    Q = per_class_field_Q(pi_field, S)
    P_sub = np.zeros((m, L, A_alpha, A_alpha))
    for n in range(m):
        for k in range(L):
            P_sub[n, k] = branch_P(Q, int(classes[n]), k, tau)
    # pi_arch[n, l, a] = pi_field[classes[n], l, a]
    pi_arch = pi_field[classes]  # (m, L, A)
    return {'pi_field': pi_field, 'S': S, 'rho': rho, 'classes': classes,
              'Q': Q, 'P_sub': P_sub, 'pi_arch': pi_arch, 'tau': tau}


def _mm_clv_to_dict(mm):
    return {'r': np.asarray(mm.rho1), 's': np.asarray(mm.s),
              'A': np.asarray(mm.A)}


def test_mm_edge_jax_matches_numpy():
    import jax
    jax.config.update("jax_enable_x64", True)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        MMClv, mm_edge as mm_edge_np)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv_jax import (
        mm_edge_jax)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        beta_no_jump, field_transition_row)
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    L, m, A_alpha, K_c = 3, 3, 4, 2
    tau, rho_chain = 0.4, 0.5
    S = _make_random_model(rng, L, m, A_alpha, K_c, tau)
    F_child = _random_family_clv(rng, L, m, A_alpha, S['pi_arch'])
    F_child_mm = MMClv(rho1=F_child['r'], s=F_child['s'], A=F_child['A'],
                          log_scale=0.0)

    # Numpy path.
    F_np = mm_edge_np(F_child_mm, S['classes'], tau, S['rho'], S['pi_field'],
                        S['Q'], rho_chain)

    # JAX path.
    beta = np.array([beta_no_jump(S['rho'], k, tau, rho_chain)
                       for k in range(L)])
    # W[l_from, l_to] = pf_row[l_to] - beta if l_to == l_from else pf_row[l_to]
    W = np.zeros((L, L))
    for l_from in range(L):
        pf = field_transition_row(S['rho'], l_from, tau, rho_chain)
        for l_to in range(L):
            W[l_from, l_to] = pf[l_to] - (beta[l_from] if l_to == l_from else 0.0)

    F_jax = mm_edge_jax(
        {k: jnp.asarray(v) for k, v in F_child.items()},
        jnp.asarray(S['P_sub']),
        jnp.asarray(beta),
        jnp.asarray(W))
    F_jax_np = {k: np.asarray(v) for k, v in F_jax.items()}

    F_np_dict = _mm_clv_to_dict(F_np)
    assert np.allclose(F_np_dict['r'], F_jax_np['r'], atol=1e-12), \
        (F_np_dict['r'], F_jax_np['r'])
    assert np.allclose(F_np_dict['s'], F_jax_np['s'], atol=1e-12)
    assert np.allclose(F_np_dict['A'], F_jax_np['A'], atol=1e-12)
    print("mm_edge_jax matches numpy OK")


def test_mm_combine_jax_matches_numpy():
    import jax
    jax.config.update("jax_enable_x64", True)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        MMClv, mm_combine as mm_combine_np)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv_jax import (
        mm_combine_jax)
    import jax.numpy as jnp

    rng = np.random.default_rng(1)
    L, m, A_alpha, K_c = 3, 3, 4, 2
    S = _make_random_model(rng, L, m, A_alpha, K_c, 0.3)
    F_X = _random_family_clv(rng, L, m, A_alpha, S['pi_arch'])
    F_Y = _random_family_clv(rng, L, m, A_alpha, S['pi_arch'])
    X_mm = MMClv(rho1=F_X['r'], s=F_X['s'], A=F_X['A'], log_scale=0.0)
    Y_mm = MMClv(rho1=F_Y['r'], s=F_Y['s'], A=F_Y['A'], log_scale=0.0)

    F_np = _mm_clv_to_dict(mm_combine_np(X_mm, Y_mm, S['classes'],
                                                 S['pi_field']))
    F_jax = mm_combine_jax(
        {k: jnp.asarray(v) for k, v in F_X.items()},
        {k: jnp.asarray(v) for k, v in F_Y.items()},
        jnp.asarray(S['pi_arch']))
    F_jax_np = {k: np.asarray(v) for k, v in F_jax.items()}

    assert np.allclose(F_np['r'], F_jax_np['r'], atol=1e-12), \
        (F_np['r'], F_jax_np['r'])
    assert np.allclose(F_np['s'], F_jax_np['s'], atol=1e-12)
    assert np.allclose(F_np['A'], F_jax_np['A'], atol=1e-12)
    print("mm_combine_jax matches numpy OK")


def test_mm_mass_jax_matches_numpy():
    import jax
    jax.config.update("jax_enable_x64", True)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        MMClv, mm_mass_one as mm_mass_one_np,
        mm_mass_two as mm_mass_two_np)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv_jax import (
        mm_mass_one_jax, mm_mass_two_jax)
    import jax.numpy as jnp

    rng = np.random.default_rng(2)
    L, m, A_alpha, K_c = 3, 3, 4, 2
    S = _make_random_model(rng, L, m, A_alpha, K_c, 0.3)
    F_X = _random_family_clv(rng, L, m, A_alpha, S['pi_arch'])
    F_Y = _random_family_clv(rng, L, m, A_alpha, S['pi_arch'])
    X_mm = MMClv(rho1=F_X['r'], s=F_X['s'], A=F_X['A'], log_scale=0.0)
    Y_mm = MMClv(rho1=F_Y['r'], s=F_Y['s'], A=F_Y['A'], log_scale=0.0)

    m1_np, _ = mm_mass_one_np(X_mm, S['classes'], S['rho'], S['pi_field'])
    m1_jax = float(mm_mass_one_jax(
        {k: jnp.asarray(v) for k, v in F_X.items()},
        jnp.asarray(S['rho'])))
    assert np.isclose(m1_np, m1_jax, atol=1e-12), (m1_np, m1_jax)

    m2_np, _ = mm_mass_two_np(X_mm, Y_mm, S['classes'], S['rho'],
                                    S['pi_field'])
    m2_jax = float(mm_mass_two_jax(
        {k: jnp.asarray(v) for k, v in F_X.items()},
        {k: jnp.asarray(v) for k, v in F_Y.items()},
        jnp.asarray(S['rho']),
        jnp.asarray(S['pi_arch'])))
    assert np.isclose(m2_np, m2_jax, atol=1e-12), (m2_np, m2_jax)
    print("mm_mass_{one,two}_jax match numpy OK")


def test_leaf_clv_jax_matches_numpy():
    import jax
    jax.config.update("jax_enable_x64", True)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        leaf_clv as leaf_clv_np)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv_jax import (
        leaf_clv_jax)
    import jax.numpy as jnp

    rng = np.random.default_rng(3)
    L, m, A_alpha, K_c = 3, 4, 5, 2
    S = _make_random_model(rng, L, m, A_alpha, K_c, 0.1)
    obs = rng.integers(0, A_alpha, size=m).astype(np.int32)

    F_np = _mm_clv_to_dict(leaf_clv_np(obs, S['classes'], S['pi_field']))
    F_jax = leaf_clv_jax(jnp.asarray(obs), jnp.asarray(S['pi_arch']))
    F_jax_np = {k: np.asarray(v) for k, v in F_jax.items()}

    assert np.allclose(F_np['r'], F_jax_np['r'], atol=1e-12), \
        (F_np['r'], F_jax_np['r'])
    assert np.allclose(F_np['s'], F_jax_np['s'], atol=1e-12)
    assert np.allclose(F_np['A'], F_jax_np['A'], atol=1e-12)
    print("leaf_clv_jax matches numpy OK")


def test_full_cherry_pipeline_jax_matches_numpy():
    """End-to-end: leaf_clv + mm_edge + mm_mass_two, both paths."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv import (
        MMClv, leaf_clv, mm_edge, mm_mass_two, beta_no_jump,
        field_transition_row)
    from tkfdp.coupling.dynfield.phylo_elbo.mm_clv_jax import (
        leaf_clv_jax, mm_edge_jax, mm_mass_two_jax)
    import jax.numpy as jnp

    rng = np.random.default_rng(4)
    L, m, A_alpha, K_c = 3, 4, 5, 2
    tau, rho_chain = 0.5, 0.4
    S = _make_random_model(rng, L, m, A_alpha, K_c, tau / 2)
    obs_a = rng.integers(0, A_alpha, size=m).astype(np.int32)
    obs_b = rng.integers(0, A_alpha, size=m).astype(np.int32)

    # Numpy.
    clv_a = leaf_clv(obs_a, S['classes'], S['pi_field'])
    clv_b = leaf_clv(obs_b, S['classes'], S['pi_field'])
    msg_a = mm_edge(clv_a, S['classes'], tau / 2, S['rho'], S['pi_field'],
                     S['Q'], rho_chain)
    msg_b = mm_edge(clv_b, S['classes'], tau / 2, S['rho'], S['pi_field'],
                     S['Q'], rho_chain)
    mass_np, _ = mm_mass_two(msg_a, msg_b, S['classes'], S['rho'],
                                    S['pi_field'])

    # JAX.
    beta = np.array([beta_no_jump(S['rho'], k, tau / 2, rho_chain)
                       for k in range(L)])
    W = np.zeros((L, L))
    for l_from in range(L):
        pf = field_transition_row(S['rho'], l_from, tau / 2, rho_chain)
        for l_to in range(L):
            W[l_from, l_to] = pf[l_to] - (beta[l_from] if l_to == l_from else 0.0)

    clv_a_j = leaf_clv_jax(jnp.asarray(obs_a), jnp.asarray(S['pi_arch']))
    clv_b_j = leaf_clv_jax(jnp.asarray(obs_b), jnp.asarray(S['pi_arch']))
    msg_a_j = mm_edge_jax(clv_a_j, jnp.asarray(S['P_sub']),
                              jnp.asarray(beta), jnp.asarray(W))
    msg_b_j = mm_edge_jax(clv_b_j, jnp.asarray(S['P_sub']),
                              jnp.asarray(beta), jnp.asarray(W))
    mass_jax = float(mm_mass_two_jax(msg_a_j, msg_b_j,
                                            jnp.asarray(S['rho']),
                                            jnp.asarray(S['pi_arch'])))

    print(f"[cherry] mass_np={mass_np:.10e} mass_jax={mass_jax:.10e} "
            f"rel_err={abs(mass_np - mass_jax) / abs(mass_np):.2e}")
    assert np.isclose(mass_np, mass_jax, atol=1e-12, rtol=1e-12), \
        (mass_np, mass_jax)


if __name__ == "__main__":
    test_mm_edge_jax_matches_numpy()
    test_mm_combine_jax_matches_numpy()
    test_mm_mass_jax_matches_numpy()
    test_leaf_clv_jax_matches_numpy()
    test_full_cherry_pipeline_jax_matches_numpy()
    print("all JAX-vs-numpy agreement tests PASS")
