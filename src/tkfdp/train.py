"""Adam SGD on H, with symmetric/zero-trace projection."""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .composite import composite_log_likelihood, project_to_symmetric_zero_trace


@dataclass
class TrainResult:
    H: np.ndarray
    log_l_history: list[float]
    H_norm_history: list[float]
    elapsed: float


def fit_H(unique_t: np.ndarray,
          obs: np.ndarray,
          n_steps: int = 2000,
          lr: float = 0.01,
          l2: float = 1e-3,
          init_H: np.ndarray | None = None,
          log_every: int = 50,
          minibatch: int | None = None,
          seed: int = 0,
          verbose: bool = True) -> TrainResult:
    """Maximize composite_log_likelihood w.r.t. H via Adam.

    `obs` is (M, 3): (t_idx, start_state, end_state).
    `unique_t` is (K,) cherry distances.
    """
    A = 20
    if init_H is None:
        # Tiny random init breaks the degenerate-eigenvalue NaN at exact H=0
        # (where the joint Q is a tensor sum of two LG08 chains).
        rng_init = np.random.default_rng(seed)
        H_np = rng_init.normal(size=(A, A)) * 0.01
        H_np = 0.5 * (H_np + H_np.T)
        H = jnp.asarray(H_np - np.trace(H_np) / A * np.eye(A))
    else:
        H = jnp.asarray(init_H)

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(H)

    unique_t_j = jnp.asarray(unique_t)
    obs_j_full = jnp.asarray(obs)

    @jax.jit
    def penalized_neg_log_l(H_, t_, o_):
        # Negative composite log-L with L2 penalty on the off-diagonal of H.
        # Keep zero-trace projection separate so the penalty doesn't fight it.
        ll = composite_log_likelihood(H_, t_, o_)
        off = H_ - jnp.diag(jnp.diag(H_))
        return -ll + 0.5 * l2 * jnp.sum(off * off)

    grad_fn = jax.jit(jax.grad(penalized_neg_log_l))

    @jax.jit
    def loss_fn(H_, t_, o_):
        return composite_log_likelihood(H_, t_, o_)

    rng = np.random.default_rng(seed)
    log_l_hist: list[float] = []
    H_norm_hist: list[float] = []

    t0 = time.time()
    for step in range(n_steps):
        if minibatch is None:
            o_b = obs_j_full
        else:
            idx = rng.integers(0, obs.shape[0], size=minibatch)
            o_b = jnp.asarray(obs[idx])

        g = grad_fn(H, unique_t_j, o_b)  # gradient of -log L (we minimize)
        g = (g + g.T) / 2
        g = g - jnp.trace(g) / A * jnp.eye(A)

        updates, opt_state = optimizer.update(g, opt_state)
        H = optax.apply_updates(H, updates)
        H = project_to_symmetric_zero_trace(H)

        if (step + 1) % log_every == 0 or step == 0:
            ll = float(loss_fn(H, unique_t_j, obs_j_full))
            hn = float(jnp.linalg.norm(H))
            log_l_hist.append(ll)
            H_norm_hist.append(hn)
            if verbose:
                print(f"  step {step+1:5d}: log L = {ll:12.2f}   ||H||_F = {hn:.4f}")

    elapsed = time.time() - t0
    return TrainResult(
        H=np.asarray(H),
        log_l_history=log_l_hist,
        H_norm_history=H_norm_hist,
        elapsed=elapsed,
    )
