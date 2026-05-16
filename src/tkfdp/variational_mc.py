"""Monte Carlo verification of the variational ELBO on a 2-site cluster.

Direct sampling of bridge paths under the factored variational law:
- For each site, sample a CTMC bridge from x_a^s to x_b^s under the constant
  rate Q_hat^s. Use rejection sampling (cheap for our 20-state, t~1
  problem).
- For each sampled joint path, compute the Girsanov log-ratio
  log P_unc^joint(path | x_a) - log Q_var_unc^joint(path | x_a) where
  P_unc uses the true joint generator Q^joint(H) and Q_var_unc uses the
  factored Q_var^joint = (Q_hat^1 ⊕ Q_hat^2 in the joint sense).
- Average over samples; add log P_var^prod(x_b | x_a). This is the MC
  ELBO estimate.

If MC ELBO is a strict lower bound on log P_exact (within MC error), the
formula is right; my closed-form is buggy. If MC also violates, the
formula derivation is wrong.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .lg08 import Q_LG08, PI_LG08
from .variational_hr import _build_Q_hat as build_Q_hat
from .variational_hr import fit_constant_rate_then_elbo


def fixed_point(H, x_a, x_b, t, n_iter=30, damping=0.5, tol=1e-9):
    """Compatibility shim for code that previously called variational.fixed_point.
    Returns dict with bar_q_1, bar_q_2 (called bar_p in HR module)."""
    res = fit_constant_rate_then_elbo(H, x_a, x_b, t, n_iter=n_iter,
                                        damping=damping, tol=tol)
    return dict(
        bar_q_1=res["bar_p_1"],
        bar_q_2=res["bar_p_2"],
        log_P_var_1=res["log_P1"],
        log_P_var_2=res["log_P2"],
    )


def gillespie_one_site(Q: np.ndarray, x_start: int, T: float,
                       rng: np.random.Generator) -> tuple[list[float], list[int]]:
    """Forward Gillespie: sample a CTMC trajectory from x_start over [0, T].
    Returns (jump_times, post_jump_states); both lists of length n_jumps.
    """
    A = Q.shape[0]
    times: list[float] = []
    states: list[int] = []
    t = 0.0
    x = x_start
    while True:
        rate_total = float(-Q[x, x])
        if rate_total <= 0:
            break
        dt = rng.exponential(1.0 / rate_total)
        if t + dt >= T:
            break
        t += dt
        # Choose target state weighted by Q[x, :] off-diagonal
        weights = np.maximum(Q[x, :], 0.0).copy()
        weights[x] = 0.0
        weights = weights / weights.sum()
        x_new = int(rng.choice(A, p=weights))
        times.append(t)
        states.append(x_new)
        x = x_new
    return times, states


def sample_bridge_rejection(Q: np.ndarray, x_a: int, x_b: int, T: float,
                            n_keep: int, rng: np.random.Generator,
                            max_attempts_per_keep: int = 5000):
    """Reject-sample bridge paths under Q from x_a to x_b at time T.
    Returns list of (jump_times, post_jump_states), and the acceptance fraction.
    """
    out = []
    attempts = 0
    while len(out) < n_keep:
        attempts += 1
        times, states = gillespie_one_site(Q, x_a, T, rng)
        x_end = x_a if not states else states[-1]
        if x_end == x_b:
            out.append((times, states))
        if attempts > n_keep * max_attempts_per_keep:
            break
    return out, attempts


def path_log_density_joint(joint_path_1: tuple[list[float], list[int]],
                           joint_path_2: tuple[list[float], list[int]],
                           x_a: tuple[int, int],
                           Q_joint_fn,
                           T: float) -> float:
    """log density of the joint path under the joint CTMC with rate matrix
    given by Q_joint_fn(x_1, x_2) -> (20*20)? We implement this by computing
    contributions:
      - For each interval where (x_1, x_2) is constant, contribute Q_diag * dt.
      - For each jump (in either site), contribute log of the joint rate.

    Q_joint_fn(x_1, x_2): returns the row Q^joint((x_1, x_2) -> .) for
    site-1 jumps as a (20,) vector, and similarly for site-2 jumps. Or we
    implement directly using the H-parameterized form.
    """
    # Build a unified event list with (time, kind, new_state) tuples.
    # kind: 'site1' or 'site2'.
    events = []
    for tt, xs in zip(joint_path_1[0], joint_path_1[1]):
        events.append((tt, 'site1', xs))
    for tt, xs in zip(joint_path_2[0], joint_path_2[1]):
        events.append((tt, 'site2', xs))
    events.sort(key=lambda e: e[0])

    # Walk through events, tracking (x_1, x_2) and accumulating log density.
    x_1, x_2 = x_a
    t_prev = 0.0
    log_d = 0.0
    for (t_jump, kind, x_new) in events:
        dt = t_jump - t_prev
        # Accumulate Q_joint_diag(x_1, x_2) * dt (negative)
        log_d += Q_joint_fn(x_1, x_2)['diag'] * dt
        # Apply the jump
        if kind == 'site1':
            log_d += Q_joint_fn(x_1, x_2)['site1'][x_new]['log_rate']
            x_1 = x_new
        else:
            log_d += Q_joint_fn(x_1, x_2)['site2'][x_new]['log_rate']
            x_2 = x_new
        t_prev = t_jump
    # Tail interval to T
    log_d += Q_joint_fn(x_1, x_2)['diag'] * (T - t_prev)
    return log_d


def make_Q_joint_fn_true(H: np.ndarray):
    """Return a function (x_1, x_2) -> dict with site-1 rates, site-2 rates,
    and diagonal value, all under the GTR-Metropolis-correction joint generator.
    """
    A = H.shape[0]
    def fn(x_1, x_2):
        # site-1 rate from x_1 to x_1' (for x_1' != x_1):
        # Q_LG08[x_1, x_1'] * exp(-0.5 (H[x_1', x_2] - H[x_1, x_2]))
        rate_1 = Q_LG08[x_1, :].copy()
        rate_1[x_1] = 0.0
        for xp in range(A):
            if xp == x_1:
                continue
            rate_1[xp] = rate_1[xp] * np.exp(-0.5 * (H[xp, x_2] - H[x_1, x_2]))
        rate_2 = Q_LG08[x_2, :].copy()
        rate_2[x_2] = 0.0
        for xp in range(A):
            if xp == x_2:
                continue
            rate_2[xp] = rate_2[xp] * np.exp(-0.5 * (H[x_1, xp] - H[x_1, x_2]))
        diag = -(rate_1.sum() + rate_2.sum())
        return dict(
            diag=diag,
            site1={xp: dict(rate=rate_1[xp], log_rate=np.log(max(rate_1[xp], 1e-300))) for xp in range(A) if xp != x_1},
            site2={xp: dict(rate=rate_2[xp], log_rate=np.log(max(rate_2[xp], 1e-300))) for xp in range(A) if xp != x_2},
        )
    return fn


def make_Q_joint_fn_var(Q_hat_1: np.ndarray, Q_hat_2: np.ndarray):
    """Return a function for the variational factored generator."""
    A = Q_hat_1.shape[0]
    def fn(x_1, x_2):
        rate_1 = Q_hat_1[x_1, :].copy()
        rate_1[x_1] = 0.0
        rate_2 = Q_hat_2[x_2, :].copy()
        rate_2[x_2] = 0.0
        diag = -(rate_1.sum() + rate_2.sum())
        return dict(
            diag=diag,
            site1={xp: dict(rate=rate_1[xp], log_rate=np.log(max(rate_1[xp], 1e-300))) for xp in range(A) if xp != x_1},
            site2={xp: dict(rate=rate_2[xp], log_rate=np.log(max(rate_2[xp], 1e-300))) for xp in range(A) if xp != x_2},
        )
    return fn


def elbo_monte_carlo(H: np.ndarray, x_a, x_b, t: float, n_keep: int,
                     seed: int = 0):
    """Monte Carlo estimate of the variational ELBO at the geometric-mean
    fixed point.

    Returns dict with mean, se, n_attempts, log_P_var_prod.
    """
    rng = np.random.default_rng(seed)
    fp = fixed_point(H, x_a, x_b, t)
    Q_hat_1 = np.asarray(build_Q_hat(jnp.asarray(H), jnp.asarray(fp["bar_q_2"])))
    Q_hat_2 = np.asarray(build_Q_hat(jnp.asarray(H), jnp.asarray(fp["bar_q_1"])))

    # Sample bridges per site
    paths_1, attempts_1 = sample_bridge_rejection(Q_hat_1, x_a[0], x_b[0], t, n_keep, rng)
    rng2 = np.random.default_rng(seed + 1)
    paths_2, attempts_2 = sample_bridge_rejection(Q_hat_2, x_a[1], x_b[1], t, n_keep, rng2)
    n_pairs = min(len(paths_1), len(paths_2))
    if n_pairs == 0:
        return dict(mc_elbo=float('nan'), se=float('nan'), n_pairs=0,
                    n_attempts_1=attempts_1, n_attempts_2=attempts_2)

    Q_true_fn = make_Q_joint_fn_true(H)
    Q_var_fn = make_Q_joint_fn_var(Q_hat_1, Q_hat_2)

    girsanov_vals = np.zeros(n_pairs)
    for i in range(n_pairs):
        log_p_true = path_log_density_joint(paths_1[i], paths_2[i], x_a, Q_true_fn, t)
        log_p_var = path_log_density_joint(paths_1[i], paths_2[i], x_a, Q_var_fn, t)
        girsanov_vals[i] = log_p_true - log_p_var

    girsanov_mean = float(girsanov_vals.mean())
    girsanov_se = float(girsanov_vals.std(ddof=1) / np.sqrt(n_pairs))
    log_P_var_prod = fp["log_P_var_1"] + fp["log_P_var_2"]
    elbo_mc = log_P_var_prod + girsanov_mean
    return dict(
        mc_elbo=elbo_mc,
        se=girsanov_se,
        log_P_var_prod=log_P_var_prod,
        girsanov_mean=girsanov_mean,
        n_pairs=n_pairs,
        n_attempts_1=attempts_1,
        n_attempts_2=attempts_2,
    )
