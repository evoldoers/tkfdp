"""Smoke test the ELBO loss: at H=0 it should match the exact log P
(since the ELBO is tight at H=0). At nonzero H it should be ≤ exact."""

import time
import numpy as np
import jax
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tkfdp.loss_elbo import _elbo_traceable, loss_fn_elbo_with_tau
from src.tkfdp.generator import build_joint_Q, joint_stationary, symmetrize_eigh
from src.tkfdp.lg08 import PI_LG08, S_LG08_F81
from src.tkfdp.laplace_potts import _sym_to_flat


def exact_log_P(H, x_a_1, x_a_2, x_b_1, x_b_2, t):
    Q = build_joint_Q(H, jnp.asarray(PI_LG08), S=jnp.asarray(S_LG08_F81))
    pi_j = joint_stationary(H, jnp.asarray(PI_LG08))
    Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
    a = x_a_1 * 20 + x_a_2; b = x_b_1 * 20 + x_b_2
    expL = jnp.exp(Lambda * t)
    inv = 1.0 / sqrt_pij[a]; sb = sqrt_pij[b]
    P = inv * jnp.sum(U_sym[a] * expL * U_sym[b]) * sb
    return float(jnp.log(jnp.clip(P, 1e-300, 1.0)))


def exact_log_P_per_site(x_a_1, x_a_2, x_b_1, x_b_2, t):
    """At H=0 only: log P factorizes — compute each site via 20-dim eigh."""
    from src.tkfdp.lg08 import Q_LG08
    Q = jnp.asarray(Q_LG08)
    pi = jnp.asarray(PI_LG08)
    sqrt = jnp.sqrt(pi); inv = 1.0 / sqrt
    Q_sym = sqrt[:, None] * Q * inv[None, :]
    Q_sym = 0.5 * (Q_sym + Q_sym.T)
    Lam, Us = jnp.linalg.eigh(Q_sym)
    expL = jnp.exp(Lam * t)
    P1 = inv[x_a_1] * jnp.sum(Us[x_a_1] * expL * Us[x_b_1]) * sqrt[x_b_1]
    P2 = inv[x_a_2] * jnp.sum(Us[x_a_2] * expL * Us[x_b_2]) * sqrt[x_b_2]
    return float(jnp.log(P1)) + float(jnp.log(P2))


def main():
    rng = np.random.default_rng(0)
    A = 20

    print("=== Test 1: ELBO at H=0 should match exact log P (via per-site reference) ===")
    H = jnp.zeros((A, A))
    failures = 0
    for trial in range(5):
        x_a_1, x_a_2 = rng.integers(0, 20, size=2)
        x_b_1, x_b_2 = rng.integers(0, 20, size=2)
        t = float(rng.uniform(0.1, 1.0))
        elbo = float(_elbo_traceable(H, jnp.asarray(PI_LG08), jnp.asarray(PI_LG08), jnp.asarray(S_LG08_F81),
                                        int(x_a_1), int(x_a_2), int(x_b_1), int(x_b_2), t))
        exact = exact_log_P(H, int(x_a_1), int(x_a_2), int(x_b_1), int(x_b_2), t)
        ref_ps = exact_log_P_per_site(int(x_a_1), int(x_a_2), int(x_b_1), int(x_b_2), t)
        ok = abs(elbo - ref_ps) < 1e-3
        marker = "OK" if ok else "FAIL"
        if not ok: failures += 1
        print(f"  ({x_a_1},{x_a_2})->({x_b_1},{x_b_2}) t={t:.3f}: "
                f"elbo={elbo:.4f} exact_400={exact:.4f} per_site={ref_ps:.4f} "
                f"elbo-per_site={elbo-ref_ps:+.2e}  {marker}")

    print()
    print("=== Test 2: ELBO at nonzero H should be <= exact (Jensen) ===")
    H = 0.3 * (rng.standard_normal((A, A)).astype(np.float32))
    H = 0.5 * (H + H.T); H = jnp.asarray(H)
    for trial in range(5):
        x_a_1, x_a_2 = rng.integers(0, 20, size=2)
        x_b_1, x_b_2 = rng.integers(0, 20, size=2)
        t = float(rng.uniform(0.1, 1.0))
        elbo = float(_elbo_traceable(H, jnp.asarray(PI_LG08), jnp.asarray(PI_LG08), jnp.asarray(S_LG08_F81),
                                        int(x_a_1), int(x_a_2), int(x_b_1), int(x_b_2), t))
        exact = exact_log_P(H, int(x_a_1), int(x_a_2), int(x_b_1), int(x_b_2), t)
        gap = exact - elbo
        ok = gap >= -1e-3
        marker = "OK" if ok else "FAIL (ELBO > exact)"
        if not ok: failures += 1
        print(f"  ({x_a_1},{x_a_2})->({x_b_1},{x_b_2}) t={t:.3f}: elbo={elbo:.4f} exact={exact:.4f} gap={gap:+.4f}  {marker}")

    print()
    print("=== Test 3: Vmap'd batched ELBO loss timing (M=200 cherries) ===")
    M = 200
    x_a_1_arr = jnp.asarray(rng.integers(0, 20, size=M).astype(np.int32))
    x_a_2_arr = jnp.asarray(rng.integers(0, 20, size=M).astype(np.int32))
    x_b_1_arr = jnp.asarray(rng.integers(0, 20, size=M).astype(np.int32))
    x_b_2_arr = jnp.asarray(rng.integers(0, 20, size=M).astype(np.int32))
    tau_per = jnp.asarray(rng.uniform(0.1, 1.0, size=M).astype(np.float32))
    valid = jnp.ones(M, dtype=np.float32)
    H_flat = _sym_to_flat(H)
    mu_prior_mat = jnp.zeros((A, A))
    tau_prior_mat = 0.5 * jnp.ones((A, A))

    # First call: includes JIT compile
    t0 = time.time()
    L = float(loss_fn_elbo_with_tau(H_flat, x_a_1_arr, x_a_2_arr, x_b_1_arr, x_b_2_arr,
                                      tau_per, valid, mu_prior_mat, tau_prior_mat))
    t1 = time.time()
    # Second call (compiled)
    L = float(loss_fn_elbo_with_tau(H_flat, x_a_1_arr, x_a_2_arr, x_b_1_arr, x_b_2_arr,
                                      tau_per, valid, mu_prior_mat, tau_prior_mat))
    t2 = time.time()
    # Third call to confirm
    L = float(loss_fn_elbo_with_tau(H_flat, x_a_1_arr, x_a_2_arr, x_b_1_arr, x_b_2_arr,
                                      tau_per, valid, mu_prior_mat, tau_prior_mat))
    t3 = time.time()
    print(f"  Total ELBO sum (M={M}): {L:.2f}")
    print(f"  Compile + first eval: {(t1-t0)*1000:.0f} ms")
    print(f"  Steady eval:          {(t2-t1)*1000:.0f} ms, then {(t3-t2)*1000:.0f} ms")

    # Grad timing
    grad_fn = jax.jit(jax.grad(loss_fn_elbo_with_tau))
    t0 = time.time()
    g = grad_fn(H_flat, x_a_1_arr, x_a_2_arr, x_b_1_arr, x_b_2_arr,
                  tau_per, valid, mu_prior_mat, tau_prior_mat)
    g.block_until_ready()
    t1 = time.time()
    g = grad_fn(H_flat, x_a_1_arr, x_a_2_arr, x_b_1_arr, x_b_2_arr,
                  tau_per, valid, mu_prior_mat, tau_prior_mat)
    g.block_until_ready()
    t2 = time.time()
    print(f"  Grad compile + first: {(t1-t0)*1000:.0f} ms")
    print(f"  Grad steady:          {(t2-t1)*1000:.0f} ms")

    print()
    print()
    print("=== Test 4: production-signature loss_fn_elbo (K_c=1) ===")
    from src.tkfdp.loss_elbo import loss_fn_elbo
    K_c = 1
    pi_classes = jnp.asarray(PI_LG08)[None, :]                  # (1, A)
    M = 50
    obs = np.zeros((M, 4), dtype=np.int64)
    obs[:, 0] = np.arange(M, dtype=np.int64)                    # t_idx
    obs[:, 1] = 0                                                # cp_ord = 0*1+0 = 0
    obs[:, 2] = (rng.integers(0, 20, size=M) * 20
                  + rng.integers(0, 20, size=M)).astype(np.int64)
    obs[:, 3] = (rng.integers(0, 20, size=M) * 20
                  + rng.integers(0, 20, size=M)).astype(np.int64)
    valid = jnp.ones(M, dtype=np.float32)
    unique_t = jnp.asarray(rng.uniform(0.1, 1.0, size=M).astype(np.float32))
    H_flat = _sym_to_flat(H)
    t0 = time.time()
    L = float(loss_fn_elbo(H_flat, jnp.asarray(obs), valid, pi_classes,
                              jnp.asarray(S_LG08_F81), mu_prior_mat, tau_prior_mat,
                              unique_t))
    t1 = time.time()
    L = float(loss_fn_elbo(H_flat, jnp.asarray(obs), valid, pi_classes,
                              jnp.asarray(S_LG08_F81), mu_prior_mat, tau_prior_mat,
                              unique_t))
    t2 = time.time()
    print(f"  Loss (M={M}): {L:.2f}")
    print(f"  Compile + first eval: {(t1-t0)*1000:.0f} ms")
    print(f"  Steady eval:          {(t2-t1)*1000:.0f} ms")

    print()
    print("=== Summary ===")
    print(f"  Failures: {failures}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
