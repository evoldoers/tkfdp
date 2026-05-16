"""Cohn et al. (2010) inhomogeneous mean-field — algorithm right, F
integration converges slowly.

For the strict closed-form bound on a 2-site cluster, use
`variational_hr.py` instead (the Holmes-Rubin pair-of-eigenvalues
formulation). It is deterministic, fast, and verified strict on a
125-cell test grid. The MC reference is `variational_mc.py`.

DIAGNOSIS (debugged 2026-05-08).

The structural Cohn algorithm (psi term, geometric-mean Q_tilde,
forward-backward sweeps with Diffrax) is mathematically correct here:
at H = 0 the `use_psi=True` and `use_psi=False` paths give *identical*
bridge marginals and identical F values, consistent with the theoretical
analysis that at H=0 psi is x-independent and only rescales rho by a
uniform u-dependent factor (which cancels in p = q rho / Z). For nonzero
H, psi correctly couples the rho ODE to the neighbour's bridge.

The remaining issue is *purely numerical*: the F integrand has finite
endpoint values but its derivatives involve `log p` and `log gamma`,
which have log-singular behaviour at u=0 and u=t (where the bridge
marginal collapses to a delta). Trapezoidal-rule integration of such
integrands converges only as O(1/n_grid), not the O(1/n_grid^2) you'd
expect for smooth ones. Empirically:

    H=0 gap (F > log P_exact, should be 0):
      n_grid=21  → +0.39 nat
      n_grid=41  → +0.26
      n_grid=81  → +0.14
      n_grid=321 → +0.04

For nonzero H this discretization error is amplified by the dynamics
that psi induces on rho (more dynamic rho => sharper boundary
features => worse trapezoidal error), so at modest n_grid the
*algorithm-correct* psi version of F looks worse than the
*algorithm-wrong* no-psi version. This is misleading: at n_grid → infty
(and proper boundary handling) `use_psi=True` gives the strict bound.

To make this module produce reliable strict bounds at modest n_grid,
options are:

  (a) replace `jnp.trapezoid` with quadrature that handles log-singular
      integrands (Gauss-Jacobi with appropriate weight, or substitution
      u = sin^2 etc.);
  (b) analytically integrate the boundary regions of the F integrand and
      apply a smooth-rule quadrature in the bulk;
  (c) use the eigendecomposition expansion of the integrand at any
      constant Q_hat — i.e., the closed form already implemented in
      `variational_hr.py`.

For 2-site, (c) is what we ship. (a) or (b) are the path to a robust
diffrax implementation for K > 2 clusters.
"""

Original docstring follows.

This implements Cohn's Algorithm 1 (round-robin per-component update) with
the FULL Euler-Lagrange ODEs from eq (16) of the paper, including the
*psi* neighbour-coupling term in the rho equation. See `~/tkf-dp/refs/cohn2010.tex`
(reverse-transcribed from the JMLR PDF) for the canonical equations.

For the 2-site case (sites 1 and 2 are each other's only neighbour), with
the GTR-Metropolis-correction joint generator:

    Q^s(x → x'; y) = Q_LG08(x, x') * exp(-0.5 * (H[x', y] - H[x, y]))

The Cohn quantities specialise as follows. Let u in [0, t] be time on the
branch, p_s(.; u) be the bridge marginal of site s.

- Inhomogeneous variational rate (Q_hat = q_tilde):
    log Q_hat^1(x → x'; u) = log Q_LG08(x, x')
                           - 0.5 * ((H @ p_2(u))[x'] - (H @ p_2(u))[x])
- Arithmetic-mean true rate over the OTHER site's bridge marginal:
    Q_bar^1(x → x'; u) = Q_LG08(x, x')
                       * sum_{y} p_2(y; u) exp(-0.5 * (H[x', y] - H[x, y]))
- Conditional rates (for the psi sum) - in 2-site case there is no further
  averaging, so q_bar_cond^2(x_2 → y_2 | x_1) = q^2(x_2 → y_2 | x_1):
    q_cond^2(x_2 → y_2 | x_1) = Q_LG08(x_2, y_2) * exp(-0.5 * (H[y_2, x_1] - H[x_2, x_1]))

The rho ODE (Cohn eq 16, second line):

    d rho^1_{x_1} / du = -rho^1_{x_1} * (Q_bar^1_{diag}(x_1; u) + psi^1_{x_1}(u))
                       - sum_{y_1 != x_1} Q_tilde^1_{x_1, y_1}(u) * rho^1_{y_1}

where Q_bar^1_{diag}(x; u) = -sum_{y != x} Q_bar^1(x, y; u), and

    psi^1_{x_1}(u) = sum_{x_2} p_2(x_2; u) * q_cond^2_{diag}(x_2, x_2 | x_1)
                   + sum_{x_2, y_2 != x_2} gamma^2(x_2, y_2; u) * ln q_cond^2(x_2, y_2 | x_1)

with gamma^2 from the algebraic constraint (Cohn eq 17):
    gamma^2(x_2, y_2; u) = p_2(x_2; u) * Q_tilde^2(x_2, y_2; u) * rho^2(y_2; u) / rho^2(x_2; u)

The ELBO is integrated via Cohn's F formula (E + H):
    F = sum_s integral [ sum_x mu^s_x q_bar^s_{x,x} + sum_{x,y!=x} gamma^s_{x,y} ln q_tilde^s_{x,y} ] du
      + sum_s integral [ sum_{x, y != x} gamma^s_{x,y} * (1 + ln mu^s_x - ln gamma^s_{x,y}) ] du
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import diffrax

from .lg08 import Q_LG08

A = 20
EPS = 1e-30


# -- Rate matrices used by the solvers --------------------------------------

def build_Q_tilde(H: jnp.ndarray, p_other_u: jnp.ndarray) -> jnp.ndarray:
    """Geometric-mean (variational) rate Q_tilde at one time slice.

    Q_tilde[x, x'] = Q_LG08[x, x'] * exp(-0.5 * (H_eff[x'] - H_eff[x]))   x != x'
    where H_eff = H @ p_other_u.
    """
    H_eff = H @ p_other_u
    factor = jnp.exp(-0.5 * (H_eff[None, :] - H_eff[:, None]))   # (x, x')
    Q_off = jnp.where(jnp.eye(A, dtype=bool), 0.0, jnp.asarray(Q_LG08) * factor)
    Q = Q_off - jnp.diag(Q_off.sum(axis=-1))
    return Q


def build_Q_bar(H: jnp.ndarray, p_other_u: jnp.ndarray) -> jnp.ndarray:
    """Arithmetic-mean true rate Q_bar at one time slice.

    Q_bar[x, x'] = Q_LG08[x, x'] * sum_y p_other(y) exp(-0.5*(H[x',y] - H[x,y]))   x != x'
    """
    # arith[x, x'] = sum_y p_other(y) * exp(-0.5*(H[x', y] - H[x, y]))
    eH_pos = jnp.exp(0.5 * H)    # (x, y)
    eH_neg = jnp.exp(-0.5 * H)   # (x', y) -- because H is sym this == eH_neg[x', y]
    arith = (eH_pos * p_other_u[None, :]) @ eH_neg.T   # (x, x')
    Q_off = jnp.where(jnp.eye(A, dtype=bool), 0.0, jnp.asarray(Q_LG08) * arith)
    Q = Q_off - jnp.diag(Q_off.sum(axis=-1))
    return Q


def _interp_at(grid_values: jnp.ndarray, t_grid: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Linearly interpolate a (T, ...) tensor at time u in [t_grid[0], t_grid[-1]]."""
    idx = jnp.clip(jnp.searchsorted(t_grid, u) - 1, 0, len(t_grid) - 2)
    u0 = t_grid[idx]; u1 = t_grid[idx + 1]
    alpha = jnp.clip((u - u0) / (u1 - u0 + 1e-12), 0.0, 1.0)
    g0 = grid_values[idx]; g1 = grid_values[idx + 1]
    return (1 - alpha) * g0 + alpha * g1


# -- Forward / backward ODE solvers -----------------------------------------

def solve_forward_q(Q_grid: jnp.ndarray, t_grid: jnp.ndarray, x_a: int) -> jnp.ndarray:
    """Solve dq/du = q Q_hat(u) on the time grid, starting at q(0) = e_{x_a}."""
    q0 = jnp.zeros(A).at[x_a].set(1.0)
    def rhs(u, q, args):
        Q = _interp_at(Q_grid, t_grid, u)
        return q @ Q
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Tsit5(),
        t0=float(t_grid[0]), t1=float(t_grid[-1]), dt0=None,
        y0=q0, saveat=diffrax.SaveAt(ts=t_grid),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
        max_steps=16384,
    )
    return sol.ys


def solve_backward_rho_simple(Q_grid: jnp.ndarray, t_grid: jnp.ndarray, x_b: int) -> jnp.ndarray:
    """Solve drho/du = -Q_hat(u) rho ON the time grid (no psi term).

    Used as a baseline before adding psi.
    """
    rho_T = jnp.zeros(A).at[x_b].set(1.0)
    s_grid = t_grid[-1] - t_grid[::-1] + t_grid[0]
    def rhs(s, rho, args):
        u = t_grid[-1] - s + t_grid[0]
        Q = _interp_at(Q_grid, t_grid, u)
        return Q @ rho
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Tsit5(),
        t0=float(s_grid[0]), t1=float(s_grid[-1]), dt0=None,
        y0=rho_T, saveat=diffrax.SaveAt(ts=s_grid),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
        max_steps=16384,
    )
    return sol.ys[::-1]


def solve_backward_rho_cohn(Q_tilde_self_grid: jnp.ndarray,
                              Q_bar_self_diag_grid: jnp.ndarray,
                              psi_self_grid: jnp.ndarray,
                              t_grid: jnp.ndarray, x_b: int) -> jnp.ndarray:
    """Solve Cohn's full backward equation (eq 16 second line):
        drho/du = -rho * (Q_bar_diag + psi) - sum_{y!=x} Q_tilde[x, y] * rho[y]

    Q_tilde_self_grid: (T, A, A) inhomogeneous Q_tilde for THIS site
    Q_bar_self_diag_grid: (T, A) diagonal of Q_bar for THIS site (negative numbers)
    psi_self_grid: (T, A) the psi vector for this site
    """
    rho_T = jnp.zeros(A).at[x_b].set(1.0)
    s_grid = t_grid[-1] - t_grid[::-1] + t_grid[0]

    def rhs(s, rho, args):
        u = t_grid[-1] - s + t_grid[0]
        Q_tilde = _interp_at(Q_tilde_self_grid, t_grid, u)
        Q_bar_diag = _interp_at(Q_bar_self_diag_grid, t_grid, u)
        psi = _interp_at(psi_self_grid, t_grid, u)
        # off-diagonal of Q_tilde @ rho
        Q_tilde_off = Q_tilde - jnp.diag(jnp.diag(Q_tilde))
        coupling = Q_tilde_off @ rho
        # ds = -du, so drho/ds = -drho/du = +rho * (Q_bar_diag + psi) + Q_tilde_off @ rho
        drho_du = -rho * (Q_bar_diag + psi) - coupling
        return -drho_du

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Tsit5(),
        t0=float(s_grid[0]), t1=float(s_grid[-1]), dt0=None,
        y0=rho_T, saveat=diffrax.SaveAt(ts=s_grid),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
        max_steps=16384,
    )
    return sol.ys[::-1]


# -- Cohn quantities --------------------------------------------------------

def compute_p_bridge(q_grid: jnp.ndarray, rho_grid: jnp.ndarray) -> jnp.ndarray:
    """p(x; u) = q(x; u) * rho(x; u) / Z, normalised per time slice."""
    p = q_grid * rho_grid
    Z = p.sum(axis=-1, keepdims=True)
    return p / jnp.clip(Z, EPS, None)


def compute_gamma(p_self_grid: jnp.ndarray, Q_tilde_self_grid: jnp.ndarray,
                  rho_self_grid: jnp.ndarray) -> jnp.ndarray:
    """gamma_{x, y}(t) = p_x(t) * Q_tilde_{x,y}(t) * rho_y(t) / rho_x(t)
    Returns (T, A, A) tensor."""
    rho_safe = jnp.clip(rho_self_grid, EPS, None)
    # gamma[t, x, y] = p[t, x] * Q_tilde[t, x, y] * rho[t, y] / rho[t, x]
    return (p_self_grid[:, :, None] / rho_safe[:, :, None]) * Q_tilde_self_grid * rho_self_grid[:, None, :]


def compute_psi_2site(H: jnp.ndarray,
                       p_other_grid: jnp.ndarray,
                       gamma_other_grid: jnp.ndarray) -> jnp.ndarray:
    """Compute psi^self at every time u, for the 2-site case where the only
    neighbour of site 'self' is the 'other' site (whose bridge marginal is
    p_other and joint flow is gamma_other).

    psi^self_{x_self}(u) = sum_{x_o} p_other(x_o; u) * q_cond^other_{x_o, x_o | x_self}
                         + sum_{x_o, y_o != x_o} gamma_other(x_o, y_o; u)
                                                * ln q_cond^other(x_o, y_o | x_self)

    For the 2-site case, q_cond^other = q^other (no further averaging) =
    Q_LG08(x_o, y_o) * exp(-0.5 * (H[y_o, x_self] - H[x_o, x_self]))  for y_o != x_o
    """
    Q_LG = jnp.asarray(Q_LG08)
    Q_LG_off = Q_LG - jnp.diag(jnp.diag(Q_LG))                 # (A_o, A_o)

    # Build q_cond[x_self, x_o, y_o] = Q_LG08(x_o, y_o) * exp(-0.5 * (H[y_o, x_self] - H[x_o, x_self]))
    # for y_o != x_o (off-diagonal); diagonal handled separately.
    # H[y_o, x_self] - H[x_o, x_self] indexed by (x_self, x_o, y_o):
    dH = H.T[None, :, None] - H.T[:, None, None]   # (x_self=H_axis_swap, x_o, y_o)?
    # Let me build it cleanly via einsum:
    # dH[x_self, x_o, y_o] = H[y_o, x_self] - H[x_o, x_self]
    dH = (H.T)[None, None, :] - (H.T)[None, :, None]   # produce shape (1, x_o, y_o)?
    # Cleanest: build via broadcasting
    H_y_xs = H[None, :, :]              # (_, y, x_self)  shape (1, A, A)  H[y, x_self]
    # We want dH[x_self, x_o, y_o] = H[y_o, x_self] - H[x_o, x_self]
    # = broadcasting: H[y_o, x_self] is (y_o, x_self); H[x_o, x_self] is (x_o, x_self).
    # We want axes (x_self, x_o, y_o). Permute:
    dH = jnp.transpose(H, (1, 0))[:, None, :] - jnp.transpose(H, (1, 0))[:, :, None]
    # Above: H.T has shape (axis0=second_index_of_H, axis1=first_index_of_H), so
    #   H.T[x_self, y_o] = H[y_o, x_self]
    #   H.T[:, None, :] adds new axis 1 for x_o; shape (x_self, 1, y_o); value H[y_o, x_self]
    #   H.T[:, :, None] adds new axis 2 for y_o; shape (x_self, x_o, 1); value H[x_o, x_self]
    # dH[x_self, x_o, y_o] = H[y_o, x_self] - H[x_o, x_self]   ✓

    q_cond_off = Q_LG_off[None, :, :] * jnp.exp(-0.5 * dH)   # (x_self, x_o, y_o), zero on x_o == y_o
    # diagonal of q_cond^other (negative row sum): q_cond_diag[x_self, x_o] = -sum_{y_o != x_o} q_cond_off[x_self, x_o, y_o]
    q_cond_diag = -q_cond_off.sum(axis=-1)                    # (x_self, x_o)

    # ln q_cond^other(x_o, y_o | x_self) for y_o != x_o.
    log_q_cond_off = jnp.where(
        jnp.eye(A, dtype=bool)[None, :, :],
        0.0,
        jnp.log(jnp.clip(q_cond_off, EPS, None))
    )

    # psi^self(u, x_self) = sum_{x_o} p_other(u, x_o) * q_cond_diag(x_self, x_o)
    #                      + sum_{x_o, y_o != x_o} gamma_other(u, x_o, y_o) * log_q_cond_off(x_self, x_o, y_o)
    term1 = jnp.einsum('ux,sx->us', p_other_grid, q_cond_diag)            # (u, x_self)
    term2 = jnp.einsum('uxy,sxy->us', gamma_other_grid, log_q_cond_off)   # (u, x_self)
    return term1 + term2


# -- ELBO via Cohn's F formula ---------------------------------------------

def compute_F_integrand(p_self_grid, Q_bar_self_grid, gamma_self_grid, Q_tilde_self_grid):
    """Per-site contribution to dF/du at every time slice (Cohn §3.2 + §4):
        dF^s/du = sum_x p_s(x) * Q_bar^s_{x,x}
                + sum_{x, y != x} gamma^s_{x, y} * (ln Q_tilde^s_{x, y} + 1 + ln p_s(x) - ln gamma^s_{x, y})
    Returns (T,) array.
    """
    # First term: sum_x p_s(x) * Q_bar_diag[x]
    Q_bar_diag = jnp.diagonal(Q_bar_self_grid, axis1=-2, axis2=-1)   # (T, A)
    term1 = jnp.sum(p_self_grid * Q_bar_diag, axis=-1)               # (T,)

    # Second term: sum_{x, y != x} gamma * (ln Q_tilde + 1 + ln p_s_x - ln gamma)
    mask = ~jnp.eye(A, dtype=bool)
    log_Q_tilde = jnp.where(mask[None, :, :],
                             jnp.log(jnp.clip(Q_tilde_self_grid, EPS, None)),
                             0.0)
    log_p = jnp.log(jnp.clip(p_self_grid, EPS, None))                # (T, A)
    log_gamma = jnp.where(mask[None, :, :],
                           jnp.log(jnp.clip(gamma_self_grid, EPS, None)),
                           0.0)
    coeff = log_Q_tilde + 1.0 + log_p[:, :, None] - log_gamma        # (T, A, A)
    coeff = jnp.where(mask[None, :, :], coeff, 0.0)
    term2 = jnp.sum(gamma_self_grid * coeff, axis=(-2, -1))          # (T,)
    return term1 + term2


# -- Outer fixed-point iteration --------------------------------------------

def fit_inhomog_fixed_point(H: np.ndarray, x_a: tuple[int, int], x_b: tuple[int, int],
                              t: float, n_grid: int = 41, n_iter: int = 20,
                              damping: float = 0.5, tol: float = 1e-7,
                              use_psi: bool = True,
                              verbose: bool = False):
    """Cohn's Algorithm 1 specialised to a 2-site cluster.

    use_psi: if True, use the FULL Cohn rho ODE (with psi). If False, use the
    naive backward equation -Q_hat * rho (the previous approximation).
    """
    H_j = jnp.asarray(H)
    t_grid = jnp.linspace(0.0, t, n_grid)

    Q_LG_j = jnp.asarray(Q_LG08)
    # Initialise: constant Q_tilde = Q_LG08 (no coupling)
    Q_tilde_1_grid = jnp.tile(Q_LG_j, (n_grid, 1, 1))
    Q_tilde_2_grid = jnp.tile(Q_LG_j, (n_grid, 1, 1))
    Q_bar_1_grid = jnp.tile(Q_LG_j, (n_grid, 1, 1))
    Q_bar_2_grid = jnp.tile(Q_LG_j, (n_grid, 1, 1))
    psi_1_grid = jnp.zeros((n_grid, A))
    psi_2_grid = jnp.zeros((n_grid, A))

    # Initial bridge marginals via simple solve
    q_1 = solve_forward_q(Q_tilde_1_grid, t_grid, x_a[0])
    rho_1 = solve_backward_rho_simple(Q_tilde_1_grid, t_grid, x_b[0])
    q_2 = solve_forward_q(Q_tilde_2_grid, t_grid, x_a[1])
    rho_2 = solve_backward_rho_simple(Q_tilde_2_grid, t_grid, x_b[1])
    p_1 = compute_p_bridge(q_1, rho_1)
    p_2 = compute_p_bridge(q_2, rho_2)

    history = []
    for it in range(n_iter):
        # Update site 1's auxiliary tensors using current p_2, gamma_2
        gamma_2 = compute_gamma(p_2, Q_tilde_2_grid, rho_2)
        gamma_1 = compute_gamma(p_1, Q_tilde_1_grid, rho_1)

        # Inhomogeneous Q_tilde and Q_bar for each site at each time slice
        Q_tilde_1_new = jax.vmap(lambda p: build_Q_tilde(H_j, p))(p_2)
        Q_tilde_2_new = jax.vmap(lambda p: build_Q_tilde(H_j, p))(p_1)
        Q_bar_1_new = jax.vmap(lambda p: build_Q_bar(H_j, p))(p_2)
        Q_bar_2_new = jax.vmap(lambda p: build_Q_bar(H_j, p))(p_1)

        if use_psi:
            psi_1_new = compute_psi_2site(H_j, p_2, gamma_2)
            psi_2_new = compute_psi_2site(H_j, p_1, gamma_1)
        else:
            psi_1_new = jnp.zeros((n_grid, A))
            psi_2_new = jnp.zeros((n_grid, A))

        # Damp
        Q_tilde_1_grid = damping * Q_tilde_1_new + (1 - damping) * Q_tilde_1_grid
        Q_tilde_2_grid = damping * Q_tilde_2_new + (1 - damping) * Q_tilde_2_grid
        Q_bar_1_grid = damping * Q_bar_1_new + (1 - damping) * Q_bar_1_grid
        Q_bar_2_grid = damping * Q_bar_2_new + (1 - damping) * Q_bar_2_grid
        psi_1_grid = damping * psi_1_new + (1 - damping) * psi_1_grid
        psi_2_grid = damping * psi_2_new + (1 - damping) * psi_2_grid

        # Re-solve forward (uses new Q_tilde) and backward (uses new Q_bar_diag, psi, Q_tilde)
        q_1 = solve_forward_q(Q_tilde_1_grid, t_grid, x_a[0])
        q_2 = solve_forward_q(Q_tilde_2_grid, t_grid, x_a[1])

        if use_psi:
            Q_bar_1_diag = jnp.diagonal(Q_bar_1_grid, axis1=-2, axis2=-1)   # (T, A)
            Q_bar_2_diag = jnp.diagonal(Q_bar_2_grid, axis1=-2, axis2=-1)
            rho_1 = solve_backward_rho_cohn(Q_tilde_1_grid, Q_bar_1_diag, psi_1_grid, t_grid, x_b[0])
            rho_2 = solve_backward_rho_cohn(Q_tilde_2_grid, Q_bar_2_diag, psi_2_grid, t_grid, x_b[1])
        else:
            rho_1 = solve_backward_rho_simple(Q_tilde_1_grid, t_grid, x_b[0])
            rho_2 = solve_backward_rho_simple(Q_tilde_2_grid, t_grid, x_b[1])

        p_1_new = compute_p_bridge(q_1, rho_1)
        p_2_new = compute_p_bridge(q_2, rho_2)
        delta = float(max(jnp.abs(p_1_new - p_1).max(), jnp.abs(p_2_new - p_2).max()))
        history.append(delta)
        p_1 = p_1_new; p_2 = p_2_new
        if verbose:
            print(f"  iter {it+1}: max|Δp| = {delta:.2e}")
        if delta < tol:
            break

    # Final ELBO via Cohn's F = E + H formula, integrated over t.
    gamma_1 = compute_gamma(p_1, Q_tilde_1_grid, rho_1)
    gamma_2 = compute_gamma(p_2, Q_tilde_2_grid, rho_2)
    dF_1 = compute_F_integrand(p_1, Q_bar_1_grid, gamma_1, Q_tilde_1_grid)
    dF_2 = compute_F_integrand(p_2, Q_bar_2_grid, gamma_2, Q_tilde_2_grid)
    du = float(t_grid[1] - t_grid[0])
    F = float(jnp.trapezoid(dF_1, dx=du) + jnp.trapezoid(dF_2, dx=du))

    return dict(
        elbo=F,
        n_iter=len(history),
        history=history,
        t_grid=np.asarray(t_grid),
        p_1=np.asarray(p_1), p_2=np.asarray(p_2),
        rho_1=np.asarray(rho_1), rho_2=np.asarray(rho_2),
        q_1=np.asarray(q_1), q_2=np.asarray(q_2),
    )
