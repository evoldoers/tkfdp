"""TKF91 BDI branch sampler.

Two implementations:

1. `gillespie_unconditional`: forward Gillespie simulation tracking
   (N_a, N_g, B, D, S) over [0, T]. Returns the trajectory summary.
2. `sample_branch_history`: rejection-conditioned sampler — repeatedly
   draws unconditioned trajectories until one ends with N_a(T) = j (and,
   optionally, N_g(T) = g). Used as the simulation reference for
   gravestone_evaluation.md Section 3 (B, D, S means vs closed-form
   expectations).

Birth dynamics include the TKF91 "immortal link": total birth rate is
(N_a + 1) * lambda (the +1 is the immortal source). Deaths happen at
total rate N_a * mu and increment N_g.

The recursive midpoint-traceback sampler of gravestone_implementation.md
Section 2 is left as future work — its acceptance test is precisely a
match against the rejection sampler implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BranchHistory:
    n_births: int
    n_deaths: int
    n_alive_end: int
    n_grave_end: int
    integrated_alive: float            # S = integral_0^T N_a(t) dt
    event_times: np.ndarray            # (n_events,) sorted
    event_kinds: np.ndarray            # (n_events,) 0=birth, 1=death


def gillespie_unconditional(i: int, T: float, lam: float, mu: float,
                            rng: np.random.Generator,
                            max_events: int = 10000) -> BranchHistory:
    """Single forward draw of the TKF91 BDI process from (N_a=i, N_g=0)
    over [0, T]. Returns a BranchHistory; statistics are computed exactly
    from the (piecewise constant) trajectory."""
    t = 0.0
    N_a = i
    N_g = 0
    B = 0
    D = 0
    S = 0.0
    times = []
    kinds = []
    n_events = 0

    while True:
        # Total event rate
        rate_b = (N_a + 1) * lam       # births (alive fragments + immortal link)
        rate_d = N_a * mu              # deaths
        rate = rate_b + rate_d
        if rate <= 0:
            # If N_a = 0 and rate_b = lam (immortal link only), this never fires.
            # Else we'd be done. Sanity:
            assert N_a == 0 and lam == 0
            break

        dt = rng.exponential(1.0 / rate)
        if t + dt >= T:
            S += N_a * (T - t)
            t = T
            break

        # Accumulate dwell time before the event
        S += N_a * dt
        t += dt

        # Decide event type
        if rng.random() < rate_b / rate:
            # Birth
            N_a += 1
            B += 1
            kinds.append(0)
        else:
            # Death
            assert N_a > 0
            N_a -= 1
            N_g += 1
            D += 1
            kinds.append(1)
        times.append(t)
        n_events += 1
        if n_events > max_events:
            raise RuntimeError(
                f"Trajectory exceeded {max_events} events (lam={lam}, mu={mu}, T={T}, i={i})"
            )

    return BranchHistory(
        n_births=B,
        n_deaths=D,
        n_alive_end=N_a,
        n_grave_end=N_g,
        integrated_alive=S,
        event_times=np.asarray(times),
        event_kinds=np.asarray(kinds),
    )


def sample_branch_history(i: int, j: int, T: float,
                          lam: float, mu: float,
                          n_keep: int,
                          rng: np.random.Generator,
                          max_attempts_per_keep: int = 500) -> tuple[list[BranchHistory], int]:
    """Rejection sampling: produce `n_keep` BranchHistory objects each
    conditioned on N_a(0) = i and N_a(T) = j. Returns (histories,
    total_attempts) so the caller can monitor acceptance rate.
    """
    out: list[BranchHistory] = []
    attempts = 0
    while len(out) < n_keep:
        attempts += 1
        h = gillespie_unconditional(i, T, lam, mu, rng)
        if h.n_alive_end == j:
            out.append(h)
        if attempts > n_keep * max_attempts_per_keep:
            break
    return out, attempts
