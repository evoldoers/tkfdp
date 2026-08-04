"""JAX/JIT primitives for the moment-matching (r * prod A + s * 1) CLV.

Milestone 1 of the phylo-ELBO Gibbs plan
(`docs/phylo_elbo_gibbs_jax_plan.md`).

Companion to `mm_clv.py` (numpy reference implementation). Same
semantics, vmap-friendly signatures for level-by-level tree forward
passes.

Data layout convention:
  Single-tree, per-cluster CLV as a dict with:
    'r':   (L,)          per-theta rank-1 mass
    's':   (L,)          per-theta scalar (constant-in-x) mass
    'A':   (m, L, A)     per-site per-theta tracking factor,
                          normalised so <pi_arch, A(.,l,n)> = 1.

Batched form (vmap axis 0): each field has a leading `batch` dim.

Model tensors passed alongside:
  P_sub:    (m, L, A, A)  substitution transition P^{c_n, theta}(a, b; tau)
  pi_arch:  (m, L, A)     archetype-materialised stationary at (c_n, theta)
  beta:     (L,)          no-jump probability beta(theta, tau)
  W:        (L, L)        jump weight W(theta_p, theta_v; tau)

For batching, all of these can have a leading (batch,) dim; vmap
handles it via in_axes broadcasting rules.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# 1e-300 is chosen so 1 + 1e-300 == 1 (safe divisor without shifting
# small legitimate values). Only used to avoid /0 in per-site A_p
# reconstruction; padded/inactive slots multiply through by mask.
_TINY = 1e-300


def mm_edge_jax(F_child: dict, P_sub: jnp.ndarray,
                  beta: jnp.ndarray, W: jnp.ndarray) -> dict:
    """Forward-propagate a child CLV up one branch (JAX).

    Args:
      F_child: dict with 'r' (L,), 's' (L,), 'A' (m, L, A).
      P_sub: (m, L, A, A) substitution transition matrix per (site, theta).
      beta: (L,) no-jump probability.
      W: (L, L) jump weight matrix.

    Returns:
      F_parent dict with same shapes as F_child.

    Formulas (under family form F = r * prod A + s * 1, invariant
    <pi_arch, A> = 1):
      r_v(theta) = beta(theta) * r_c(theta)
      A_v(a | theta, n) = sum_b P_sub[n, theta, a, b] * A_c(b | theta, n)
      s_v(theta) = beta(theta) * s_c(theta)
                  + sum_theta_u W(theta, theta_u) * (r_c + s_c)(theta_u)
    """
    r_c = F_child['r']
    s_c = F_child['s']
    A_c = F_child['A']

    r_v = beta * r_c
    # A_v[n, l, a] = sum_b P_sub[n, l, a, b] * A_c[n, l, b]
    A_v = jnp.einsum('nlab,nlb->nla', P_sub, A_c)
    marg = r_c + s_c
    s_v = beta * s_c + W @ marg
    return {'r': r_v, 's': s_v, 'A': A_v}


def mm_combine_jax(F_X: dict, F_Y: dict, pi_arch: jnp.ndarray) -> dict:
    """Combine two sibling CLVs at an internal node (JAX).

    Args:
      F_X, F_Y: dicts with 'r' (L,), 's' (L,), 'A' (m, L, A).
      pi_arch: (m, L, A) archetype equilibrium per (site, theta).

    Returns:
      F_parent dict.

    Under family F = r * prod A + s * 1, the exact product decomposes:
      X * Y = r_X r_Y prod (A_X A_Y)     -- rank-1, kernel A_X A_Y
            + r_X s_Y prod A_X            -- rank-1, kernel A_X
            + s_X r_Y prod A_Y            -- rank-1, kernel A_Y
            + s_X s_Y                     -- constant
    Projection: constant -> s_p; three rank-1 terms fold into one A_p
    via per-site moment matching under <pi_arch, .>_n weighting at
    other sites.
    """
    rX = F_X['r']; sX = F_X['s']; AX = F_X['A']
    rY = F_Y['r']; sY = F_Y['s']; AY = F_Y['A']

    # dot_AB[n, l] = <pi_arch, A_X * A_Y>_{n, l} = sum_a pi_arch A_X A_Y
    dot_AB = jnp.einsum('nla,nla,nla->nl', pi_arch, AX, AY)
    # prod_dot[l] = prod_n dot_AB[n, l]
    prod_dot = jnp.prod(dot_AB, axis=0)
    # rest_dot[n, l] = prod_dot / dot_AB[n, l] (safe divisor: pad slot
    # dot_AB will be 1, so rest_dot = prod_dot regardless).
    rest_dot = prod_dot[None, :] / jnp.where(
        dot_AB > 0.0, dot_AB, jnp.ones_like(dot_AB))

    s_p = sX * sY
    r_p = rX * rY * prod_dot + rX * sY + sX * rY

    # m_a[n, l, a] = per-site marginal of X * Y at site n under
    # <pi, .>-weighting at other sites (including the constant-term
    # s_X s_Y contribution).
    m_a = (rX[None, :, None] * rY[None, :, None]
             * rest_dot[:, :, None] * AX * AY
           + rX[None, :, None] * sY[None, :, None] * AX
           + sX[None, :, None] * rY[None, :, None] * AY
           + sX[None, :, None] * sY[None, :, None])
    # A_p = (m_a - s_p) / r_p; guard against r_p = 0 (fallback A_p = AX).
    safe_rp = jnp.where(r_p > 0.0, r_p, jnp.ones_like(r_p))
    A_p = jnp.where(
        (r_p > 0.0)[None, :, None],
        (m_a - s_p[None, :, None]) / safe_rp[None, :, None],
        AX)
    return {'r': r_p, 's': s_p, 'A': A_p}


def mm_mass_two_jax(F_X: dict, F_Y: dict, rho: jnp.ndarray,
                        pi_arch: jnp.ndarray) -> jnp.ndarray:
    """Root mass sum_theta rho[theta] * <pi_all, X * Y>(theta) (JAX).

    Uses the exact 4-cross-term expansion under family F = r*prodA + s*1
    with invariants <pi_arch, A> = 1 and <pi_arch, 1> = 1 per site:
      <pi_all, X * Y>(theta) = r_X r_Y * prod_dot + r_X s_Y + s_X r_Y + s_X s_Y
    """
    rX = F_X['r']; sX = F_X['s']; AX = F_X['A']
    rY = F_Y['r']; sY = F_Y['s']; AY = F_Y['A']
    dot_AB = jnp.einsum('nla,nla,nla->nl', pi_arch, AX, AY)
    prod_dot = jnp.prod(dot_AB, axis=0)
    per_theta = rX * rY * prod_dot + rX * sY + sX * rY + sX * sY
    return jnp.sum(rho * per_theta)


def mm_mass_one_jax(F: dict, rho: jnp.ndarray) -> jnp.ndarray:
    """Root mass sum_theta rho[theta] * <pi_all, F>(theta) at a
    single-child root (JAX). Under invariant <pi_arch, A> = 1,
    <pi_all, F>(theta) = r + s, so mass = sum_theta rho[theta]*(r+s)."""
    return jnp.sum(rho * (F['r'] + F['s']))


def leaf_clv_jax(residues: jnp.ndarray, pi_arch: jnp.ndarray,
                   *, gap_marker: int = -1) -> dict:
    """Construct a leaf CLV from observed residues (JAX).

    Args:
      residues: (m,) int; observed residue at each site (gap_marker
        marks unobserved).
      pi_arch: (m, L, A) archetype equilibrium per (site, theta).
      gap_marker: value in `residues` marking gaps (default -1).

    At each site n and theta:
      Observed (x = residues[n] >= 0):
        r_leaf(theta) *= pi_arch[n, theta, x];
        A_leaf(a | theta, n) = 1/pi_arch[n, theta, x] if a == x else 0.
        (Then <pi_arch, A_leaf>_n = 1.)
      Gapped:
        A_leaf(a | theta, n) = 1  (all a)
        no r_leaf multiplier.
      s_leaf = 0 always.
    """
    m, L, A = pi_arch.shape
    obs = residues  # (m,)
    is_obs = obs != gap_marker  # (m,)

    # For observed sites, gather pi_arch[n, l, x]:
    #   pi_at_x[n, l] = pi_arch[n, l, obs[n]] (or 1 if gapped).
    safe_obs = jnp.where(is_obs, obs, 0)  # avoid negative index for gaps
    pi_at_x = jnp.take_along_axis(
        pi_arch, safe_obs[:, None, None].astype(jnp.int32).repeat(L, axis=1),
        axis=2).squeeze(-1)  # (m, L)
    # For gapped sites, contribute pi_at_x = 1 (log = 0) to r product.
    pi_at_x = jnp.where(is_obs[:, None], pi_at_x, jnp.ones_like(pi_at_x))
    # r_leaf(theta) = prod_n pi_at_x[n, theta].
    r_leaf = jnp.prod(pi_at_x, axis=0)  # (L,)
    s_leaf = jnp.zeros(L)

    # A_leaf[n, l, a] = 1/pi_arch[n, l, obs[n]] * one_hot(obs[n])
    #                for observed n; = 1 (uniform) for gapped n.
    one_hot = jax.nn.one_hot(safe_obs, A)  # (m, A)
    # Broadcast to (m, L, A)
    one_hot_full = jnp.broadcast_to(one_hot[:, None, :], (m, L, A))
    # Divide by pi_arch at obs residue per (n, l) - safe divisor.
    safe_pi = jnp.where(pi_at_x > 0.0, pi_at_x, jnp.ones_like(pi_at_x))
    A_obs = one_hot_full / safe_pi[:, :, None]
    A_gap = jnp.ones_like(A_obs)
    A_leaf = jnp.where(is_obs[:, None, None], A_obs, A_gap)

    return {'r': r_leaf, 's': s_leaf, 'A': A_leaf}
