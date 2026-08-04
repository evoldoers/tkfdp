"""Holmes-Rubin endpoint-conditioned bridge statistics for a reversible CTMC.

For a reversible generator Q (stationary pi) and a branch of length t, given the
joint endpoint distribution edge[a,b] = q(X(0)=a, X(t)=b) (a variational two-slice
marginal), returns the expected dwell time in each state and the expected number
of x->y transitions, summed over the branch and averaged over the endpoints:

    T[x]    = sum_{a,b} edge[a,b] * E[ time in x        | X(0)=a, X(t)=b ]
    N[x,y]  = sum_{a,b} edge[a,b] * E[ #(x->y jumps)    | X(0)=a, X(t)=b ]

Standard HR identities (reversible eigendecomposition Q = U diag(lam) U^{-1}):

    E[time in x | a,b]      = (1/P_ab) sum_{k,l} U_ak Uinv_kx U_xl Uinv_lb J_kl
    E[#x->y jumps | a,b]    = (Q_xy/P_ab) sum_{k,l} U_ak Uinv_kx U_yl Uinv_lb J_kl

with the divided-difference kernel J_kl = integral_0^t e^{lam_k s} e^{lam_l (t-s)} ds
= (e^{lam_k t} - e^{lam_l t})/(lam_k - lam_l), or t e^{lam_k t} when lam_k = lam_l.

The two contractions above are done without materialising the O(A^4) endpoint
tensor: define G[x,b] = sum_{a} edge[a,b] * (contribution), handled by grouping
the a-sum with U. See `bridge` for the eigenmode contraction.
"""
from __future__ import annotations

import numpy as np


def eig_rev(Q, pi):
    """Symmetric eigendecomposition of a reversible generator.
    Returns (lam, U, Uinv) with Q = U diag(lam) Uinv."""
    d = np.sqrt(np.clip(pi, 1e-300, None))
    di = 1.0 / d
    Qs = d[:, None] * Q * di[None, :]
    Qs = 0.5 * (Qs + Qs.T)
    if not np.all(np.isfinite(Qs)):
        raise ValueError(f"eig_rev: non-finite symmetrized generator "
                         f"(pi min={pi.min():.2e}, max|Q|={np.abs(Q).max():.2e})")
    try:
        lam, V = np.linalg.eigh(Qs)
    except np.linalg.LinAlgError:
        # LAPACK dsyevd occasionally fails to converge; retry with a tiny
        # diagonal jitter and the (more robust) divide-and-conquer-free driver
        import scipy.linalg as sla
        jit = 1e-9 * np.trace(np.abs(Qs)) / Qs.shape[0]
        lam, V = sla.eigh(Qs + jit * np.eye(Qs.shape[0]))
    U = di[:, None] * V           # Q = U diag(lam) U^{-1}, U^{-1} = V^T diag(d)
    Uinv = V.T * d[None, :]
    return lam, U, Uinv


def _Jmat(lam, t):
    """Divided-difference kernel J_kl = int_0^t e^{lam_k s} e^{lam_l (t-s)} ds."""
    e = np.exp(lam * t)
    dl = lam[:, None] - lam[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        J = (e[:, None] - e[None, :]) / dl
    diag = t * e
    J[np.abs(dl) < 1e-12] = 0.0
    J = J + np.diag(diag - np.diag(J))
    # fix near-diagonal entries set to 0 above where |dl|<tol but off-diagonal
    close = np.abs(dl) < 1e-12
    if close.any():
        # for k!=l but degenerate eigenvalues, J -> t e^{lam t} as well
        ii, jj = np.where(close)
        J[ii, jj] = t * e[ii]
    return J


def bridge(Q, pi, t, edge, want_N=True, eig=None):
    """Expected dwell T[x] and (optionally) usage N[x,y] over a branch of length
    t, averaged over the endpoint joint `edge` (A,A). Returns (T, N, P) with
    P = expm(Qt). Eigenmode contraction, O(A^3) per branch. Pass `eig`=(lam,U,Uinv)
    from `eig_rev` to reuse a cached decomposition (the reversible eigh is the same
    for every branch of a given generator)."""
    A = Q.shape[0]
    lam, U, Uinv = eig_rev(Q, pi) if eig is None else eig
    J = _Jmat(lam, t)                                   # (A,A) over eigenmodes
    P = (U * np.exp(lam * t)[None, :]) @ Uinv
    Pab = P
    W = edge / np.maximum(Pab, 1e-300)                  # w[a,b] = edge/P_ab
    # Left factor over a-index: La[k,x?]. We need for each x:
    #   E_time[x] = sum_ab W[a,b] sum_kl U_ak Uinv_kx U_xl Uinv_lb J_kl
    # Group: Aleft[k,l] = sum_a U_ak * ( sum_b W[a,b] Uinv_lb )
    #        = sum_a U_ak M[a,l],  M = W @ Uinv.T        (M[a,l]=sum_b W[a,b]Uinv_lb)
    M = W @ Uinv.T                                      # (A,A) index [a,l]
    Aleft = U.T @ M                                     # (A,A) index [k,l] = sum_a U_ak M[a,l]
    AJ = Aleft * J                                      # [k,l]
    # E_time[x] = sum_kl Uinv_kx AJ[k,l] U_xl = sum_x-diag of (Uinv.T @ AJ @ U.T?)
    # Uinv_kx = Uinv[k,x]; U_xl = U[x,l]. So E_time[x] = sum_kl Uinv[k,x] AJ[k,l] U[x,l]
    #   = ( (Uinv.T @ AJ) elementwise* U ).sum over l for each x
    UinvT_AJ = Uinv.T @ AJ                              # [x,l]
    T = np.einsum("xl,xl->x", UinvT_AJ, U)              # E_time per x
    N = None
    if want_N:
        # E_jump[x,y] = Q_xy sum_ab W[a,b] sum_kl U_ak Uinv_kx U_yl Uinv_lb J_kl
        #   = Q_xy sum_kl Uinv_kx AJ[k,l] U_yl
        # Build B[x,y] = sum_kl Uinv[k,x] AJ[k,l] U[y,l] = (Uinv.T @ AJ @ U.T)[x,y]
        B = UinvT_AJ @ U.T                              # [x,y]
        N = Q * B
        np.fill_diagonal(N, 0.0)
    return T, N, P
