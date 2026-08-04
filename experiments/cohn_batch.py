"""Batch-native Cohn (2010) mean-field for the 2-site Glauber-Potts pair.

A NEW, batched version of the Cohn inhomogeneous mean-field -- it does NOT replace
the vendored diffrax implementation (`src/tkfdp/cohn_ctbn/ctbn.py`) or the K=2
diffrax reference (`src/tkfdp/variational_cohn.py`); it reuses their equations on a
FIXED time grid so it can run thousands of endpoint pairs at once on the GPU.

Why it batches where the others don't:
  - FIXED grid + piecewise-constant matrix-exponential propagation of the forward q
    and backward rho ODEs -- no diffrax, no adaptive stepping, no `max_steps` host
    callback (the thing that made jax.vmap of the diffrax version crash / OOM).
    Memory is ~ B*n_grid*A, not B*max_steps.
  - GLAUBER generator q(x->y | nbr z) = S[x,y] exp(h[y] + 2 J[y,z]), matching the
    evolsnake benchmark so it bounds the SAME exact log[e^{tW}]_xy as our path ELBO
    and midpoint bound.

Everything is a batched tensor op (leading B axis) with lax.scan over grid steps.
`cohn_batch_elbo(S,h,J,xs,ys,T,damping,n_grid,n_iter)` -> (F (B,), delta (B,)),
delta = last-sweep max|dp| per pair (converged if small; fixed-grid has no
max_steps failure mode). Validated against the diffrax Cohn in
experiments/cohn_batch_validate.py.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

A = 20
EPS = 1e-30
_OFF = None   # set at call time: (A,A) off-diagonal mask


def _diagm(v):                                    # (...,A) -> (...,A,A) diagonal
    return v[..., :, None] * jnp.eye(A)


def _Q_tilde(S, h, J, p_other):
    """Geometric-mean variational rate S[x,y] exp(h[y]+2(J@p_other)[y]). p_other (M,A)->(M,A,A)."""
    Jp = p_other @ J.T                            # (M,A) = (J @ p_other)[y]
    off = S[None] * jnp.exp(h[None, None, :] + 2.0 * Jp[:, None, :]) * (1.0 - jnp.eye(A))[None]
    return off - _diagm(off.sum(-1))


def _Q_bar(S, h, E2, p_other):
    """Arithmetic-mean true rate S[x,y] exp(h[y]) <exp(2J[y,.])>_{p_other}. (M,A)->(M,A,A)."""
    avg = p_other @ E2.T                          # (M,A) = sum_z p(z) exp(2 J[y,z])
    off = S[None] * (jnp.exp(h)[None, None, :] * avg[:, None, :]) * (1.0 - jnp.eye(A))[None]
    return off - _diagm(off.sum(-1))


def _forward_q(rate_grid, q0, du_grid):
    """dq/du = q rate(u), q(0)=q0, piecewise-constant on a (possibly non-uniform)
    grid. rate_grid (n,B,A,A); du_grid (n-1,) interval widths -> (n,B,A)."""
    M = expm(rate_grid[:-1] * du_grid[:, None, None, None])   # (n-1,B,A,A)
    def body(q, Mk):
        qn = jnp.einsum('bx,bxy->by', q, Mk)
        return qn, qn
    _, tail = jax.lax.scan(body, q0, M)
    return jnp.concatenate([q0[None], tail], axis=0)


def _backward_rho(gen_grid, rhoT, du_grid):
    """drho/du = -gen(u) rho, rho(T)=rhoT, piecewise-constant. gen_grid (n,B,A,A);
    du_grid (n-1,) -> (n,B,A)."""
    M = expm(gen_grid[:-1] * du_grid[:, None, None, None])    # M_k over interval [k,k+1]
    def body(rho, Mk):
        rn = jnp.einsum('bxy,by->bx', Mk, rho)
        return rn, rn
    _, tail = jax.lax.scan(body, rhoT, M[::-1])               # intervals n-2..0
    rhos = jnp.concatenate([rhoT[None], tail], axis=0)         # rho_{n-1}..rho_0
    return rhos[::-1]                                          # -> rho_0..rho_{n-1}


def _psi(S, h, J, p_other_grid, gamma_other_grid):
    """psi^self(u,x_self): Glauber q_cond(x_o->y_o|x_self)=S[x_o,y_o]exp(h[y_o]+2J[y_o,x_self]).
    p_other_grid (n,B,A), gamma_other_grid (n,B,A,A) -> (n,B,A)."""
    qc_off = S[None] * jnp.exp(h[None, None, :] + 2.0 * J.T[:, None, :]) * (1.0 - jnp.eye(A))[None]  # (x_self,x_o,y_o)
    qc_diag = -qc_off.sum(-1)                                                                        # (x_self,x_o)
    log_qc_off = jnp.where((1.0 - jnp.eye(A))[None] > 0, jnp.log(jnp.clip(qc_off, EPS, None)), 0.0)
    term1 = jnp.einsum('ubx,sx->ubs', p_other_grid, qc_diag)
    term2 = jnp.einsum('ubxy,sxy->ubs', gamma_other_grid, log_qc_off)
    return term1 + term2


def _gamma(q_self, Qt_self, rho_self):
    """gamma[x,y] = p[x] Qt[x,y] rho[y]/rho[x], computed from q (NOT p) so the rho[x]
    denominator cancels analytically -- p[x]/rho[x] = q[x]/Z -- and does not blow up
    where the bridge marginal collapses at the endpoints (the singular region that
    breaks the naive p[x]/clip(rho[x]) form). q,rho (n,B,A), Qt (n,B,A,A)->(n,B,A,A)."""
    Z = jnp.clip((q_self * rho_self).sum(-1, keepdims=True), EPS, None)   # (n,B,1)
    return (q_self[..., :, None] * Qt_self * rho_self[..., None, :]) / Z[..., None]


def _p_bridge(q, rho):
    p = q * rho
    return p / jnp.clip(p.sum(-1, keepdims=True), EPS, None)


def _F_integrand(p, Qbar, gamma, Qt):
    """Per-site dF/du on the grid -> (n,B)."""
    Qbar_diag = jnp.diagonal(Qbar, axis1=-2, axis2=-1)                       # (n,B,A)
    t1 = jnp.sum(p * Qbar_diag, -1)
    mask = (1.0 - jnp.eye(A))[None, None]
    lQt = jnp.where(mask > 0, jnp.log(jnp.clip(Qt, EPS, None)), 0.0)
    lp = jnp.log(jnp.clip(p, EPS, None))
    lg = jnp.where(mask > 0, jnp.log(jnp.clip(gamma, EPS, None)), 0.0)
    coeff = jnp.where(mask > 0, lQt + 1.0 + lp[..., :, None] - lg, 0.0)
    t2 = jnp.sum(gamma * coeff, (-2, -1))
    return t1 + t2


def _bgen(Qt, Qb, psi):
    """Backward generator B(u) = diag(Qbar_diag + psi) + Qt_off, so drho/du = -B rho."""
    Qt_off = Qt * (1.0 - jnp.eye(A))[None, None]
    Qb_diag = jnp.diagonal(Qb, axis1=-2, axis2=-1)
    return _diagm(Qb_diag + psi) + Qt_off


@partial(jax.jit, static_argnums=(7, 8, 9, 10))
def cohn_batch_elbo(S, h, J, xs, ys, T, damping=0.5, n_grid=64, n_iter=12, use_psi=True, return_bridge=False):
    """Batched Cohn ELBO for B endpoint pairs on the Glauber generator.
    S,h,J (A,A),(A,),(A,A); xs,ys (B,2) int; T scalar. -> (F (B,), delta (B,))."""
    B = xs.shape[0]
    E2 = jnp.exp(2.0 * J)
    # NON-UNIFORM grid clustered at both endpoints (sin^2 / Chebyshev spacing) to
    # resolve the log-singular F integrand near t=0 and t=T at modest n_grid.
    s = jnp.linspace(0.0, 1.0, n_grid)
    grid = T * (1.0 - jnp.cos(jnp.pi * s)) / 2.0
    du_grid = jnp.diff(grid)                       # (n_grid-1,)
    eye = jnp.eye(A)
    q0_1 = eye[xs[:, 0]]; q0_2 = eye[xs[:, 1]]
    rhoT_1 = eye[ys[:, 0]]; rhoT_2 = eye[ys[:, 1]]

    def build(pg):                                  # (n,B,A) -> Qt,Qb (n,B,A,A)
        flat = pg.reshape(-1, A)
        Qt = _Q_tilde(S, h, J, flat).reshape(n_grid, B, A, A)
        Qb = _Q_bar(S, h, E2, flat).reshape(n_grid, B, A, A)
        return Qt, Qb

    # init: independent single-site process (J=0), constant rate -> damped-rate state
    S_off = S * (1.0 - eye)
    Q0 = S_off * jnp.exp(h)[None, :]; Q0 = Q0 - jnp.diag(Q0.sum(-1))
    Q0g = jnp.broadcast_to(Q0, (n_grid, B, A, A))
    Qt1g = Q0g; Qt2g = Q0g; Qb1g = Q0g; Qb2g = Q0g
    psi1g = jnp.zeros((n_grid, B, A)); psi2g = jnp.zeros((n_grid, B, A))
    q1 = _forward_q(Q0g, q0_1, du_grid); rho1 = _backward_rho(Q0g, rhoT_1, du_grid)
    q2 = _forward_q(Q0g, q0_2, du_grid); rho2 = _backward_rho(Q0g, rhoT_2, du_grid)
    p1 = _p_bridge(q1, rho1); p2 = _p_bridge(q2, rho2)

    zeros = jnp.zeros((n_grid, B, A))

    def _sweep(_i, carry):
        # GAUSS-SEIDEL sweep: update site 1 from current p2, then site 2 from NEW p1.
        # A lax.fori_loop body (NOT a Python loop) so the graph stays compact and the
        # compile/memory cost is independent of n_iter.
        p1, p2, q1, q2, rho1, rho2, Qt1g, Qt2g, Qb1g, Qb2g, psi1g, psi2g, _d = carry
        p1_prev, p2_prev = p1, p2
        g2 = _gamma(q2, Qt2g, rho2)
        Qt1n, Qb1n = build(p2)
        psi1n = _psi(S, h, J, p2, g2) if use_psi else zeros
        Qt1g = damping * Qt1n + (1 - damping) * Qt1g
        Qb1g = damping * Qb1n + (1 - damping) * Qb1g
        psi1g = damping * psi1n + (1 - damping) * psi1g
        q1 = _forward_q(Qt1g, q0_1, du_grid)
        rho1 = _backward_rho(_bgen(Qt1g, Qb1g, psi1g), rhoT_1, du_grid)
        p1 = _p_bridge(q1, rho1)

        g1 = _gamma(q1, Qt1g, rho1)
        Qt2n, Qb2n = build(p1)
        psi2n = _psi(S, h, J, p1, g1) if use_psi else zeros
        Qt2g = damping * Qt2n + (1 - damping) * Qt2g
        Qb2g = damping * Qb2n + (1 - damping) * Qb2g
        psi2g = damping * psi2n + (1 - damping) * psi2g
        q2 = _forward_q(Qt2g, q0_2, du_grid)
        rho2 = _backward_rho(_bgen(Qt2g, Qb2g, psi2g), rhoT_2, du_grid)
        p2 = _p_bridge(q2, rho2)
        _d = jnp.maximum(jnp.abs(p1 - p1_prev).max((0, 2)), jnp.abs(p2 - p2_prev).max((0, 2)))
        return (p1, p2, q1, q2, rho1, rho2, Qt1g, Qt2g, Qb1g, Qb2g, psi1g, psi2g, _d)

    init = (p1, p2, q1, q2, rho1, rho2, Qt1g, Qt2g, Qb1g, Qb2g, psi1g, psi2g, jnp.zeros(B))
    (p1, p2, q1, q2, rho1, rho2, Qt1g, Qt2g, Qb1g, Qb2g,
     psi1g, psi2g, delta) = jax.lax.fori_loop(0, n_iter, _sweep, init)

    g1 = _gamma(q1, Qt1g, rho1); g2 = _gamma(q2, Qt2g, rho2)
    dF1 = _F_integrand(p1, Qb1g, g1, Qt1g); dF2 = _F_integrand(p2, Qb2g, g2, Qt2g)
    F = jnp.trapezoid(dF1, x=grid, axis=0) + jnp.trapezoid(dF2, x=grid, axis=0)
    if return_bridge:
        return F, delta, p1, p2, rho1, rho2, dF1, dF2, grid
    return F, delta
