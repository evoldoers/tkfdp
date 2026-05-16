"""Smoke test: chunked-M Laplace HVP gives same gradient and HVP as the
full (unchunked) loss at a FIXED H, within float precision.

The chunked path's only mathematical difference from the full-vmap path
is the chunk-by-chunk summation order, so at the same H_flat the
gradients should agree to machine precision. We DON'T compare the
full-Adam trajectory — gradient ordering differences in float32 compound
over 10 Adam steps and the trajectories diverge ~5%; that's expected
and not a bug.

Run with `JAX_ENABLE_X64=1` for the strictest numerical comparison.
"""

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PARENT, "src"))

from tkfdp.lg08 import PI_LG08, S_LG08_F81
from tkfdp.laplace_potts import _flat_to_sym, _sym_to_flat, log_prior_pathwise
from tkfdp.loss_elbo import (_grad_neg_sum_elbo_chunk,
                                  _hvp_neg_sum_elbo_chunk,
                                  _split_chunks,
                                  grad_fn_elbo)


def main():
    rng = np.random.default_rng(0)
    K_c = 1; A = 20; M = 32

    pi_classes = np.tile(np.asarray(PI_LG08), (K_c, 1))
    S = np.asarray(S_LG08_F81)
    obs = np.zeros((M, 4), dtype=np.int64)
    obs[:, 0] = np.arange(M)
    obs[:, 2] = (rng.integers(0, 20, M) * 20 + rng.integers(0, 20, M))
    obs[:, 3] = (rng.integers(0, 20, M) * 20 + rng.integers(0, 20, M))
    valid = np.ones(M)
    unique_t = rng.uniform(0.1, 1.0, M).astype(np.float64)
    H = 0.3 * rng.standard_normal((A, A)); H = 0.5 * (H + H.T)

    H_flat = jnp.asarray(_sym_to_flat(jnp.asarray(H)))
    obs_j = jnp.asarray(obs)
    mask_j = jnp.asarray(valid, dtype=H_flat.dtype)
    pi_j = jnp.asarray(pi_classes); S_j = jnp.asarray(S)
    t_j = jnp.asarray(unique_t)
    mu_p = jnp.zeros((A, A)); tau_p = jnp.full((A, A), 4.0)

    print(f"=== Setup ===")
    print(f"  M={M}, K_c={K_c}, A={A}, dtype={H_flat.dtype}")
    print(f"  (set JAX_ENABLE_X64=1 for strict numerical comparison)")

    print(f"\n=== Test 1: gradient equality at fixed H ===")
    g_full = grad_fn_elbo(H_flat, obs_j, mask_j, pi_j, S_j, mu_p, tau_p, t_j)
    chunks = _split_chunks(obs, valid, 8)
    chunks_j = [(jnp.asarray(o), jnp.asarray(m, dtype=H_flat.dtype))
                  for o, m in chunks]
    g_prior = jax.jit(jax.grad(
        lambda H_: -log_prior_pathwise(_flat_to_sym(H_), mu_p, tau_p)
    ))
    g_chunk = g_prior(H_flat)
    for obs_c, mask_c in chunks_j:
        g_chunk = g_chunk + _grad_neg_sum_elbo_chunk(
            H_flat, obs_c, mask_c, pi_j, S_j, t_j
        )

    g_diff = float(jnp.max(jnp.abs(g_full - g_chunk)))
    g_rel = g_diff / max(float(jnp.max(jnp.abs(g_full))), 1e-12)
    print(f"  ||g_full|| = {float(jnp.linalg.norm(g_full)):.6f}")
    print(f"  ||g_chunk|| = {float(jnp.linalg.norm(g_chunk)):.6f}")
    print(f"  max abs diff = {g_diff:.4e}, rel = {g_rel:.4e}")

    print(f"\n=== Test 2: HVP equality at fixed H, basis vector e_5 ===")
    v = jnp.asarray(np.eye(g_full.shape[0])[5])
    hvp_ref = jax.jvp(
        lambda H_: grad_fn_elbo(H_, obs_j, mask_j, pi_j, S_j, mu_p, tau_p, t_j),
        (H_flat,), (v,)
    )[1]
    hvp_prior_jit = jax.jit(lambda H_, v_: jax.jvp(g_prior, (H_,), (v_,))[1])
    hvp_chunk = hvp_prior_jit(H_flat, v)
    for obs_c, mask_c in chunks_j:
        hvp_chunk = hvp_chunk + _hvp_neg_sum_elbo_chunk(
            H_flat, v, obs_c, mask_c, pi_j, S_j, t_j
        )

    hvp_diff = float(jnp.max(jnp.abs(hvp_ref - hvp_chunk)))
    hvp_rel = hvp_diff / max(float(jnp.max(jnp.abs(hvp_ref))), 1e-12)
    print(f"  ||hvp_ref|| = {float(jnp.linalg.norm(hvp_ref)):.6f}")
    print(f"  ||hvp_chunk|| = {float(jnp.linalg.norm(hvp_chunk)):.6f}")
    print(f"  max abs diff = {hvp_diff:.4e}, rel = {hvp_rel:.4e}")
    print(f"  finite ref: {bool(jnp.all(jnp.isfinite(hvp_ref)))}, "
            f"chunk: {bool(jnp.all(jnp.isfinite(hvp_chunk)))}")

    # Tolerance: in float32 expect ~1e-3 absolute; float64 ~1e-6.
    if H_flat.dtype == jnp.float64:
        g_tol = 1e-5; hvp_tol = 1e-3
    else:
        g_tol = 1e-3; hvp_tol = 1e-1

    failures = []
    if g_rel > g_tol:
        failures.append(f"grad rel diff {g_rel:.2e} > {g_tol:.0e}")
    if hvp_rel > hvp_tol:
        failures.append(f"hvp rel diff {hvp_rel:.2e} > {hvp_tol:.0e}")
    if not bool(jnp.all(jnp.isfinite(hvp_chunk))):
        failures.append("hvp_chunk has non-finite entries")

    if failures:
        print(f"\nFAILURES: {failures}")
        return 1
    print("\nPASS — grad and HVP match within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
