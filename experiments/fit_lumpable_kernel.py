#!/usr/bin/env python3
r"""Convex kernel-basis fit of the two-sided Lumpable reversible-exchangeable
pair CTMC (paper2b).

The joint 400-state reversible generator is written as

    Q = Q_0 + sum_alpha theta_alpha K_alpha

where Q_0 is the A(+)A lift of the (fixed) marginal GTR generator, and the
{K_alpha} span the LUMPABILITY KERNEL: reversible + component-exchangeable flux
perturbations that leave every one-component (marginal) observable unchanged.
Concretely, in Klein-orbit flux coordinates phi (F_{xy}=phi_{orbit(xy)}), the
kernel is null(B), where B is the block-sum incidence map

    (B phi)_{ijk} = sum_l F_{ij,kl},   i != k

so B phi = 0 <=> the perturbation redistributes flux among the second-coordinate
targets l WITHIN each marginal edge i->k while leaving the marginal rate fixed.

dim(kernel) = N_phi - rank(B) = N_phi - N_c
            = d_n - n(n-1)/2,
  N_phi = (n^4+n^2-2n)/4          (Burnside: reversible+exchangeable orbits)
  N_c   = n(n-1)(2n-1)/2          (rank of B / independent lumpability constraints)
  d_n   = n(n-1)(n^2-3n+6)/4      (two-sided-lumpable reversible-exchangeable flux dim)
Checks: n=2 -> 1,  n=4 -> 24,  n=20 -> 32680.

For FIXED pi and FIXED marginal, the CTMC complete-data log-likelihood
  ell(Q) = sum_{x!=y} N_xy log Q_xy - sum_x T_x sum_{y!=x} Q_xy
is CONCAVE in the flux, so the M-step is a convex program
  max_phi  sum_o [C_o log phi_o - H_o phi_o]   s.t.  B phi = b_marg,  phi >= 0,
solved to the GLOBAL optimum by a rational (Sinkhorn-style) dual coordinate
ascent.  Wrapped in EM (endpoint-conditioned bridge E-step), the whole fit is
monotone and lands at the global coupling optimum given pi.

Run with  PYTHONPATH=src  from the repo root.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
import fit_pair_models as fp                      # noqa: E402
from tkfdp.permfield.hr import bridge, eig_rev    # noqa: E402


# =====================================================================
# General-n Klein-orbit + lumpability machinery (for dimension checks
# and the small-n recovery experiments).  NA=20 pair machinery is reused
# from fit_pair_models directly.
# =====================================================================
def build_orbits_n(n):
    """Klein-4 (V4) orbit id over directed off-diagonal transitions on [n]^2.

    Returns (orbit_id (NS,NS) int, n_orbits, is_single (NS,NS) bool)."""
    NS = n * n
    ii, jj = np.divmod(np.arange(NS), n)
    orbit_id = np.full((NS, NS), -1, np.int64)
    canon = {}
    nxt = 0
    for x in range(NS):
        i, j = int(ii[x]), int(jj[x])
        for y in range(NS):
            if x == y:
                continue
            k, l = int(ii[y]), int(jj[y])
            imgs = ((i, j, k, l), (k, l, i, j), (j, i, l, k), (l, k, j, i))
            key = min(imgs)
            o = canon.get(key)
            if o is None:
                o = nxt
                canon[key] = o
                nxt += 1
            orbit_id[x, y] = o
    ch1 = ii[:, None] != ii[None, :]
    ch2 = jj[:, None] != jj[None, :]
    is_single = ch1 ^ ch2
    return orbit_id, nxt, is_single


def build_B_sparse(n, orbit_id, n_orbits):
    """Sparse block-sum incidence B: (B phi)_{ijk}=sum_l phi[orbit(ij,kl)], i!=k.

    Rows indexed r = (i*n+k? ) via an explicit list; returns (B_csr, rows) where
    rows is an (n_rows,3) int array of (i,j,k)."""
    from scipy.sparse import csr_matrix
    rows_ijk = []
    data, ridx, cidx = [], [], []
    r = 0
    for i in range(n):
        for j in range(n):
            x = i * n + j
            for k in range(n):
                if i == k:
                    continue
                rows_ijk.append((i, j, k))
                # accumulate multiplicity per orbit over l
                cnt = {}
                for l in range(n):
                    o = int(orbit_id[x, k * n + l])
                    cnt[o] = cnt.get(o, 0) + 1
                for o, c in cnt.items():
                    ridx.append(r)
                    cidx.append(o)
                    data.append(float(c))
                r += 1
    B = csr_matrix((data, (ridx, cidx)), shape=(r, n_orbits))
    return B, np.array(rows_ijk, dtype=np.int64)


def dims(n):
    """Closed-form dimension bookkeeping for component alphabet size n.

    HONEST dimensions are set by the EXACT rank of the block-sum incidence B,
    rank(B) = (n-1)(n^2-n+1) (sympy-verified rational rank), NOT the paper's
    nominal N_c = n(n-1)(2n-1)/2, which over-counts the independent lumpability
    constraints by C(n-1,2) = (n-1)(n-2)/2.  Consequences:
      kernel (marginal-fixing coupling subspace) = null(B) = N_phi - rank(B)
             = (n-1)^2 (n^2-2n+4) / 4          [paper target d_n - N_g under-counts]
      d_n_true (two-sided-lumpable flux dim)     = N_phi + N_g - rank(B)
             = d_n_paper + (n-1)(n-2)/2 ."""
    N_phi = (n ** 4 + n ** 2 - 2 * n) // 4
    N_g = n * (n - 1) // 2
    N_c_paper = n * (n - 1) * (2 * n - 1) // 2
    d_n_paper = n * (n - 1) * (n ** 2 - 3 * n + 6) // 4
    rank_B = (n - 1) * (n ** 2 - n + 1)                 # exact
    kernel = N_phi - rank_B                             # honest marginal-fixing dim
    assert kernel == (n - 1) ** 2 * (n ** 2 - 2 * n + 4) // 4
    d_n_true = N_phi + N_g - rank_B
    return dict(n=n, N_phi=N_phi, N_g=N_g, N_c_paper=N_c_paper,
                d_n_paper=d_n_paper, rank_B=rank_B, kernel=kernel,
                d_n_true=d_n_true, kernel_paper_target=d_n_paper - N_g,
                gap=(n - 1) * (n - 2) // 2)


def verify_dims(n, numeric=True):
    """Verify N_phi (orbit count) and the honest rank(B)/null(B) numerically."""
    d = dims(n)
    orbit_id, n_orbits, is_single = build_orbits_n(n)
    ok_phi = (n_orbits == d["N_phi"])
    out = dict(d, N_phi_built=int(n_orbits), ok_Nphi=bool(ok_phi))
    if numeric:
        B, rows = build_B_sparse(n, orbit_id, n_orbits)
        if n <= 8:
            Bd = B.toarray()
            rank = int(np.linalg.matrix_rank(Bd, tol=1e-9))
            from scipy.linalg import null_space
            null_dim = null_space(Bd, rcond=1e-9).shape[1]
        else:
            BBt = (B @ B.T).toarray()
            ev = np.linalg.eigvalsh(BBt)
            tol = max(BBt.shape) * np.finfo(float).eps * ev.max()
            rank = int((ev > tol).sum())
            null_dim = int(n_orbits - rank)
        out.update(rank_B_num=rank, ok_rank=bool(rank == d["rank_B"]),
                   null_dim=null_dim, ok_kernel=bool(null_dim == d["kernel"]),
                   n_rows=int(B.shape[0]))
    return out


def kernel_basis_dense(n):
    """Explicit orthonormal kernel basis E (N_phi x kernel) via null(B). Small n."""
    from scipy.linalg import null_space
    orbit_id, n_orbits, _ = build_orbits_n(n)
    B, rows = build_B_sparse(n, orbit_id, n_orbits)
    E = null_space(B.toarray(), rcond=1e-9)
    return E, orbit_id, n_orbits, B, rows


# =====================================================================
# Marginal single-site GTR fit (20-state reversible EM) from n_single.
# =====================================================================
def marginal_counts_from_pair(npair):
    """Single-site transition counts obtained by MARGINALISING the pair tensor over
    the spectator coordinate (exchangeable-symmetrised over the two coordinates).
    This is the self-consistent marginal of the pair PROCESS -- using an external
    single-site count set (e.g. all-sites n_single) mismatches the contact-biased
    composition and makes even the A(+)A independent baseline underperform."""
    p5 = npair.reshape(NA, NA, NA, NA, -1)
    M1 = p5.sum(axis=(1, 3))          # coordinate 1: sum over spectator (j,l)
    M2 = p5.sum(axis=(0, 2))          # coordinate 2: sum over spectator (i,k)
    return M1 + M2


def fit_marginal_gtr(nsingle, tau, rho=None, iters=60, tol=1e-9, verbose=False):
    """Fit a reversible GTR (symmetric flux) on n states from single-site cherry
    transition counts nsingle (n,n,T).  pi is HELD at rho (empirical marginal) so
    the result is the product-consistent marginal generator.  Closed-form flux
    M-step phi_ab = C_ab/H_ab.  Returns (A generator, rho, g symmetric flux)."""
    n = nsingle.shape[0]
    occ = 0.5 * (nsingle.sum(1).sum(1) + nsingle.sum(0).sum(1))
    if rho is None:
        rho = occ / occ.sum()
    rho = np.clip(rho, 1e-12, None); rho = rho / rho.sum()
    off = ~np.eye(n, dtype=bool)
    # init: small symmetric flux
    F = 1e-2 * np.sqrt(np.outer(rho, rho)); F[~off] = 0.0
    Q = fp_Q_from_flux(F, rho)
    prev = -np.inf
    for it in range(iters):
        eig = eig_rev(Q, np.clip(rho, 1e-12, None))
        N = np.zeros((n, n)); T = np.zeros(n)
        for t in range(tau.shape[0]):
            edge = nsingle[:, :, t].astype(float)
            if edge.sum() == 0:
                continue
            Tt, Nt, _ = bridge(Q, rho, tau[t], edge, want_N=True, eig=eig)
            N += Nt; T += Tt
        # symmetric-orbit (unordered pair) closed form
        C = N + N.T
        H = (T / rho)[:, None] + (T / rho)[None, :]
        F = np.where((H > 0) & off, C / np.maximum(H, 1e-300), 0.0)
        F[~off] = 0.0
        Q = fp_Q_from_flux(F, rho)
        ll = _single_loglik(Q, rho, tau, nsingle)
        if verbose and (it < 2 or it % 10 == 0):
            print(f"    [marg-gtr] it {it:2d} LL={ll:.1f}", flush=True)
        if it > 3 and ll - prev < tol * abs(nsingle.sum()):
            break
        prev = ll
    A = Q.copy()
    return A, rho, F


def fp_Q_from_flux(F, pi):
    n = F.shape[0]
    Q = F / np.maximum(pi[:, None], 1e-300)
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(n)] = -Q.sum(1)
    return Q


def _single_loglik(Q, pi, tau, nsingle):
    lam, U, Uinv = eig_rev(Q, np.clip(pi, 1e-12, None))
    ll = 0.0
    for t in range(tau.shape[0]):
        edge = nsingle[:, :, t].astype(float)
        if edge.sum() == 0:
            continue
        P = (U * np.exp(lam * tau[t])[None, :]) @ Uinv
        ll += float((edge * np.log(np.clip(P, 1e-300, None))).sum())
    return ll


# =====================================================================
# Pair-level kernel-basis lumpable fit (NA=20), reusing fit_pair_models.
# =====================================================================
NA = fp.NA
NS = fp.NS


def pair_setup(A, rho):
    """Build product pi, orbit machinery, b_marg (fixed marginal RHS), and the
    A(+)A lift orbit fluxes phi0.  Returns a dict of static objects."""
    orbit_id, n_orbits, is_single = fp.build_orbits()
    pi_prod = np.outer(rho, rho).reshape(NS)
    g = A * rho[:, None]                         # symmetric marginal flux g_ik = rho_i A_ik
    g = 0.5 * (g + g.T)
    np.fill_diagonal(g, 0.0)
    # A(+)A lift flux F0 (single-transition only): F0[(i,j),(k,j)] = rho_i rho_j A_ik
    off = ~np.eye(NS, dtype=bool)
    ii, jj = np.divmod(np.arange(NS), NA)
    F0 = np.zeros((NS, NS))
    for j in range(NA):                          # comp-1 change (i,j)->(k,j)
        idx = np.arange(NA) * NA + j
        F0[np.ix_(idx, idx)] = pi_prod[idx][:, None] * A       # rho_i rho_j A_ik
    for i in range(NA):                          # comp-2 change (i,j)->(i,l)
        idx = i * NA + np.arange(NA)
        F0[np.ix_(idx, idx)] += pi_prod[idx][:, None] * A
    np.fill_diagonal(F0, 0.0)
    # orbit fluxes phi0 (any representative per orbit; F0 is orbit-constant by build)
    phi0 = np.zeros(n_orbits)
    oid_off = orbit_id[off]
    phi0_seen = np.full(n_orbits, np.nan)
    fvals = F0[off]
    for o, v in zip(oid_off, fvals):
        if np.isnan(phi0_seen[o]):
            phi0_seen[o] = v
    phi0 = np.nan_to_num(phi0_seen, nan=0.0)
    # b_marg = B phi0 (rows over (i,j,k))
    rows = fp.build_lump_rows(orbit_id)          # (ri,rj,rk,ro) ro:(n_rows,NA)
    ri, rj, rk, ro = rows
    b_marg = phi0[ro].sum(1)                     # (n_rows,)
    # sparse block-sum incidence B (n_rows x n_orbits) with orbit multiplicities
    from scipy.sparse import csr_matrix
    n_rows = ro.shape[0]
    rr = np.repeat(np.arange(n_rows), NA)
    cc = ro.ravel()
    B_sparse = csr_matrix((np.ones(rr.size), (rr, cc)),
                          shape=(n_rows, n_orbits)).tocsr()
    rowdat = [(u, m.astype(float)) for u, m in
              (np.unique(ro[a], return_counts=True) for a in range(n_rows))]
    return dict(orbit_id=orbit_id, n_orbits=n_orbits, is_single=is_single,
                pi=pi_prod, g=g, F0=F0, phi0=phi0, rows=rows, b_marg=b_marg,
                off=off, B_sparse=B_sparse, B_sparseT=B_sparse.T.tocsr(),
                rowdat=rowdat)


def orbit_aggregate(N, T, pi, orbit_id, n_orbits, off):
    """C_o = sum_orbit N_xy ; H_o = sum_orbit T_x/pi_x  (over all off-diagonal)."""
    oid = orbit_id[off]
    C_o = np.bincount(oid, weights=N[off], minlength=n_orbits)
    Hmat = (T[:, None] / np.maximum(pi[:, None], 1e-300) * np.ones((1, NS)))
    H_o = np.bincount(oid, weights=Hmat[off], minlength=n_orbits)
    return C_o, H_o


def mstep_convex(C_o, H_o, setup, lam=None, sweeps=60, kappa=0.0,
                 rtol=1e-9, verbose=False):
    """Global convex M-step: max_phi sum_o[C_o log phi_o - H_o phi_o] s.t.
    B phi = b_marg, phi>=0.  Solved by the RATIONAL DUAL coordinate ascent (the
    Poisson analogue of Sinkhorn/IPF): phi_o(lam)=C_o/(H_o+(B^T lam)_o), sweeping
    the dual coordinate of each block-sum row a to solve the 1-D monotone equation
    sum_o B_ao C_o/(d_o + B_ao delta) = b_a exactly (safeguarded per-row Newton).

    Scale-robust (each row solved exactly), unlike a global Newton on the raw
    ill-conditioned dual.  `kappa` is an optional flux pseudocount (Laplace prior,
    C_o -> C_o+kappa) that keeps the optimum strictly interior; kappa=0 lets the
    coupling wash out to the positivity boundary.  Returns (phi, lam, relres)."""
    ro = setup["rows"][3]                        # (n_rows, NA) orbit ids over l
    b = setup["b_marg"]
    rowdat = setup["rowdat"]
    n_orbits = setup["n_orbits"]
    n_rows = ro.shape[0]
    Ce = C_o + kappa
    if lam is None:
        lam = np.zeros(n_rows)
    d = H_o.copy()
    if np.any(lam != 0):
        BTlam = np.zeros(n_orbits)
        np.add.at(BTlam, ro.ravel(), np.repeat(lam, NA))
        d += BTlam
    bnorm = max(np.linalg.norm(b), 1e-30)
    relres = np.inf
    prev = np.inf
    old_err = np.seterr(divide="ignore", invalid="ignore")  # den->0 at domain edge is safeguarded
    for sw in range(sweeps):
        for a in range(n_rows):
            ou, mu = rowdat[a]
            Cu = Ce[ou]; du = d[ou]; ba = b[a]
            lo = -(du / mu).min() + 1e-30
            delta = 0.0
            if Cu.sum() <= 0:                    # row has no observed flux: unsatisfiable
                continue
            for _ in range(50):                  # per-row safeguarded Newton
                den = du + mu * delta
                f = (mu * Cu / den).sum() - ba
                if abs(f) <= 1e-13 * (ba + 1e-30):
                    break
                fp = -(mu * mu * Cu / den ** 2).sum()
                nd = delta - f / fp if fp != 0 else np.inf
                if (not np.isfinite(nd)) or nd <= lo:  # bisection fallback
                    if f > 0:
                        hi = max(delta, 0.0) + 1.0
                        while (mu * Cu / (du + mu * hi)).sum() - ba > 0:
                            hi *= 2.0
                        alo, ahi = delta, hi
                    else:
                        alo, ahi = lo, delta
                    for _ in range(70):
                        mid = 0.5 * (alo + ahi)
                        if (mu * Cu / (du + mu * mid)).sum() - ba > 0:
                            alo = mid
                        else:
                            ahi = mid
                    nd = 0.5 * (alo + ahi)
                    break
                delta = nd
            d[ou] = du + mu * delta
            lam[a] += delta
        phi = Ce / d
        relres = float(np.linalg.norm(phi[ro].sum(1) - b) / bnorm)
        if verbose and (sw < 3 or sw % 10 == 0):
            print(f"      [convex M] sweep {sw:3d} relres={relres:.3e}", flush=True)
        if relres < 1e-9 or abs(prev - relres) < rtol * relres:
            break
        prev = relres
    np.seterr(**old_err)
    phi = Ce / d
    relres = float(np.linalg.norm(phi[ro].sum(1) - b) / bnorm)
    return phi, lam, relres


def mstep_convex_newton(C_o, H_o, setup, lam=None, iters=60, tol=1e-11,
                        ridge=1e-10, verbose=False):
    """Global convex M-step via the RATIONAL DUAL (Sinkhorn analogue), solved by
    damped Newton.  Fixed marginal => fixed RHS b_marg.

      min_lam  Psi(lam) = b.lam - sum_o C_o log(H_o + (B^T lam)_o),  H+B^Tlam>0
      phi_o(lam) = C_o/(H_o + (B^T lam)_o)   (0 if C_o=0),  grad = b - B phi,
      Hess = B diag(C_o/(H_o+(B^T lam)_o)^2) B^T  (PSD).

    At the optimum B phi = b_marg and phi>=0, i.e. the global maximiser of the
    concave complete-data auxiliary over {B phi = b_marg, phi>=0}."""
    B = setup["B_sparse"]
    b = setup["b_marg"]
    n_rows, n_orbits = B.shape
    if lam is None:
        lam = np.zeros(n_rows)
    active = C_o > 0
    Cact = np.where(active, C_o, 0.0)
    tiny = 1e-300

    def denom(lm):
        return H_o + (B.T @ lm)

    def psi(lm):
        d = denom(lm)
        if np.any(d[active] <= 0):
            return np.inf
        return float(b @ lm - (Cact * np.log(np.maximum(d, tiny))).sum())

    relres = np.inf
    for it in range(iters):
        d = denom(lam)
        phi = np.where(active, Cact / np.maximum(d, tiny), 0.0)
        grad = b - B @ phi
        relres = float(np.linalg.norm(grad) / max(np.linalg.norm(b), 1e-30))
        if relres < tol:
            break
        w = np.where(active, Cact / np.maximum(d, tiny) ** 2, 0.0)
        Bw = B.multiply(w[None, :])
        Hs = (Bw @ B.T).toarray()
        Hs[np.diag_indices(n_rows)] += ridge
        try:
            step = np.linalg.solve(Hs, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Hs, grad, rcond=None)[0]
        # feasibility + Armijo line search on Psi (minimise): lam <- lam - a step
        a = 1.0
        f0 = psi(lam)
        for _ in range(60):
            trial = lam - a * step
            ft = psi(trial)
            if np.isfinite(ft) and ft <= f0 - 1e-4 * a * (grad @ step):
                break
            a *= 0.5
        lam = lam - a * step
        if verbose and (it < 3 or it % 10 == 0):
            print(f"      [newton M] it {it:2d} relres={relres:.2e} a={a:.2e}",
                  flush=True)
    d = denom(lam)
    phi = np.where(active, Cact / np.maximum(d, tiny), 0.0)
    relres = float(np.linalg.norm(b - B @ phi) / max(np.linalg.norm(b), 1e-30))
    return phi, lam, relres


def flux_from_orbit(phi, orbit_id, off):
    F = np.zeros((NS, NS))
    F[off] = phi[orbit_id[off]]
    return F


def interior_init_flux(setup, eps=1e-2):
    """Strictly-positive lumpable-ish init: A(+)A lift plus a small positive flux
    on every double-transition orbit.  Needed because the pure A(+)A lift has
    EXACTLY-ZERO double-transition rates, which the Holmes-Rubin E-step maps to
    zero expected usage -- an absorbing boundary that freezes the coupling at 0.
    The added flux perturbs the marginal only at init; the first M-step restores
    B phi = b_marg exactly."""
    orbit_id = setup["orbit_id"]; n_orbits = setup["n_orbits"]
    off = setup["off"]; is_single = setup["is_single"]; pi = setup["pi"]
    phi0 = setup["phi0"].copy()
    scale = eps * phi0[phi0 > 0].mean()
    oid_double = np.unique(orbit_id[off & (~is_single)])
    phi_init = phi0.copy()
    phi_init[oid_double] = np.maximum(phi_init[oid_double], scale)
    return flux_from_orbit(phi_init, orbit_id, off)


def fit_convex_em(npair, tau, setup, n_em=40, verbose=True, kappa=0.0,
                  m_sweeps=40):
    """EM: bridge E-step + convex M-step (fixed pi, fixed marginal). Monotone."""
    pi = setup["pi"]
    orbit_id = setup["orbit_id"]; n_orbits = setup["n_orbits"]; off = setup["off"]
    # init at a strictly-interior generator (see interior_init_flux)
    F = interior_init_flux(setup)
    Q = fp.Q_from_flux(F, pi)
    lam = None
    hist = []
    relres = np.nan
    for it in range(n_em):
        N, T = fp.estep(Q, pi, tau, npair)
        C_o, H_o = orbit_aggregate(N, T, pi, orbit_id, n_orbits, off)
        phi, lam, relres = mstep_convex(C_o, H_o, setup, lam=lam, kappa=kappa,
                                        sweeps=m_sweeps)
        F = flux_from_orbit(phi, orbit_id, off)
        Q = fp.Q_from_flux(F, pi)
        ll = fp.loglik(Q, pi, tau, npair)
        hist.append(ll)
        if verbose and (it < 3 or it % 5 == 0 or it == n_em - 1):
            print(f"    [convex-EM] it {it:2d} LL={ll:.1f} per={ll/npair.sum():.4f} "
                  f"Mrelres={relres:.2e}", flush=True)
        if it > 4 and abs(hist[-1] - hist[-2]) < 1e-8 * npair.sum():
            break
    return dict(Q=Q, F=F, pi=pi, phi=phi, hist=hist, ll=hist[-1],
                m_relres=relres)


# =====================================================================
# LP feasibility: max slack t s.t. rates >= t, marginal fixed.
# =====================================================================
def lp_feasibility(setup, verbose=False, signed_t=False):
    """Max-slack feasibility LP: does a STRICTLY-POSITIVE lumpable generator with
    this (pi, marginal) exist?  Maximise the uniform rate margin t subject to
    q_xy = phi_{o(xy)}/pi_x >= t for every off-diagonal (x,y) and B phi = b_marg.
    Substituting phi = t*pmax + s (s>=0, pmax_o = max source pi over the orbit)
    folds the rate floor into the bounds and removes n_orbits explicit inequality
    rows -- a much smaller LP that HiGHS solves fast.  Then B s + t*c = b_marg,
    c = B pmax.  t>0 => strictly-positive lumpable generator exists; t=0 => only a
    boundary generator (some rates pinned at 0).

    With `signed_t=True`, t is unbounded below: t<0 means every solution of
    B phi = b_marg has some negative rate, i.e. the positive lumpable cone for this
    (pi, marginal) is EMPTY (no valid generator at all).  With the default t>=0 the
    LP instead reports infeasible (status 2) when the cone is empty."""
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, hstack
    orbit_id = setup["orbit_id"]; n_orbits = setup["n_orbits"]
    off = setup["off"]; pi = setup["pi"]
    B = setup["B_sparse"]
    NSn = pi.shape[0]
    b_marg = setup["b_marg"]
    n_rows = B.shape[0]
    # pmax_o = max over directed off-diag (x,y) in orbit o of pi_x (source mass)
    oid = orbit_id[off]
    src = np.repeat(np.arange(NSn), NSn).reshape(NSn, NSn)[off]
    pmax = np.zeros(n_orbits)
    np.maximum.at(pmax, oid, pi[src])
    c_col = B @ pmax                             # (n_rows,)  B applied to pmax
    # variables x = [s (n_orbits) >= 0, t]; max t  s.t.  B s + t*c_col = b_marg
    obj = np.zeros(n_orbits + 1); obj[-1] = -1.0
    A_eq = hstack([B, csr_matrix(c_col.reshape(-1, 1))]).tocsr()
    tb = (None, None) if signed_t else (0, None)
    bounds = [(0, None)] * n_orbits + [tb]
    res = linprog(obj, A_eq=A_eq, b_eq=b_marg, bounds=bounds,
                  method="highs", options={"time_limit": 600, "presolve": True})
    if res.success:
        s = np.asarray(res.x[:n_orbits]); t = float(res.x[-1])
        phi = t * pmax + s
    else:
        t = float("nan"); phi = None
    return dict(t=t, status=res.status, message=res.message,
                success=res.success, phi=phi)


# =====================================================================
# Boundary analysis at the MLE.
# =====================================================================
def boundary_report(res, setup, thr=1e-10):
    """Fraction of off-diagonal rates at the positivity boundary + min slack."""
    Q = res["Q"]; off = setup["off"]
    q = Q[off]
    scale = q.max()
    tiny = thr * scale
    frac_rates = float((q <= tiny).mean())
    frac_1e6 = float((q <= 1e-6 * scale).mean())
    frac_1e4 = float((q <= 1e-4 * scale).mean())
    min_rate = float(q[q > 0].min()) if (q > 0).any() else 0.0
    # orbit-level
    phi = res["phi"]
    orbit_id = setup["orbit_id"]
    is_single = setup["is_single"]
    off_single = off & is_single
    off_double = off & (~is_single)
    oid_single = np.unique(orbit_id[off_single])
    oid_double = np.unique(orbit_id[off_double])
    phi_scale = phi.max()
    def stats(oids):
        p = phi[oids]
        return dict(n=int(len(oids)),
                    frac_zero=float((p <= thr * phi_scale).mean()),
                    n_zero=int((p <= thr * phi_scale).sum()))
    return dict(frac_rates_zero=frac_rates, frac_rates_below_1e6=frac_1e6,
                frac_rates_below_1e4=frac_1e4, min_pos_rate=min_rate,
                max_rate=float(scale), n_offdiag=int(off.sum()),
                single=stats(oid_single), double=stats(oid_double),
                n_orbits_zero=int((phi <= thr * phi_scale).sum()),
                n_orbits=int(len(phi)))


# =====================================================================
# Small-n recovery experiments.
# =====================================================================
def recover_n2():
    """n=2: 3 total DOF (2 stationary + 1 gauged rate). Brute-force the exact
    global MLE on a synthetic coupled 2x2 count set (single time bin) and confirm
    the convex EM (fixed product pi, fixed marginal) recovers it."""
    n = 2
    orbit_id, n_orbits, is_single = build_orbits_n(n)   # NS=4
    NSn = n * n
    # --- ground-truth coupled generator (reversible, exchangeable) ---
    rng = np.random.default_rng(0)
    rho = np.array([0.6, 0.4])
    pi = np.outer(rho, rho).reshape(NSn)
    # marginal generator A (2-state GTR)
    a = 0.8
    A = np.array([[-a * rho[1] / rho[0] * rho[0] / rho[0], 0.0],
                  [0.0, 0.0]])
    # simpler explicit: symmetric marginal flux g01
    g01 = 0.3
    A = np.array([[0.0, g01 / rho[0]], [g01 / rho[1], 0.0]])
    A[0, 0] = -A[0, 1]; A[1, 1] = -A[1, 0]
    # build a lumpable coupled Q_true with a KNOWN kernel coefficient
    setup = _setup_n(n, A, rho)
    E, *_ = kernel_basis_dense(n)          # (n_orbits, 1) for n=2
    theta_true = np.array([0.15])
    phi_true = setup["phi0"] + E @ theta_true
    if (phi_true < 0).any():
        theta_true = np.array([0.05])
        phi_true = setup["phi0"] + E @ theta_true
    F_true = flux_from_orbit_n(phi_true, orbit_id, n)
    Q_true = fp_Q_from_flux(F_true, pi)
    # --- simulate counts at a few time bins ---
    tau = np.array([0.1, 0.3, 0.7])
    lam, U, Uinv = eig_rev(Q_true, np.clip(pi, 1e-12, None))
    npair = np.zeros((NSn, NSn, tau.shape[0]))
    Ntot = 2_000_000
    for t in range(tau.shape[0]):
        P = (U * np.exp(lam * tau[t])[None, :]) @ Uinv
        joint = pi[:, None] * np.clip(P, 0, None)
        joint = joint / joint.sum()
        npair[:, :, t] = Ntot * joint
    # --- brute-force MLE over theta (1-D) at fixed pi, marginal ---
    def negll(theta):
        phi = setup["phi0"] + E @ np.atleast_1d(theta)
        if (phi < 0).any():
            return 1e18
        F = flux_from_orbit_n(phi, orbit_id, n)
        Q = fp_Q_from_flux(F, pi)
        return -_pair_loglik_n(Q, pi, tau, npair)
    grid = np.linspace(-0.5, 0.5, 20001)
    vals = np.array([negll(g) for g in grid])
    theta_grid = grid[np.argmin(vals)]
    ll_grid = -vals.min()
    # --- convex EM ---
    res = fit_convex_em_n(npair, tau, setup, n, n_em=60, verbose=False)
    phi_em = res["phi"]
    theta_em = float(np.linalg.lstsq(E, phi_em - setup["phi0"], rcond=None)[0][0])
    ll_em = res["ll"]
    return dict(theta_true=float(theta_true[0]), theta_grid=float(theta_grid),
                theta_em=theta_em, ll_grid=float(ll_grid), ll_em=float(ll_em),
                per_grid=float(ll_grid / npair.sum()),
                per_em=float(ll_em / npair.sum()))


def recover_n4():
    """n=4: 24-dim kernel. Generate data from a known lumpable Q with nonzero
    coupling; confirm the convex fit recovers an INTERIOR optimum with the
    coupling present (kernel coeffs recovered, LL matches truth)."""
    n = 4
    orbit_id, n_orbits, is_single = build_orbits_n(n)
    NSn = n * n
    rng = np.random.default_rng(1)
    rho = np.array([0.35, 0.3, 0.2, 0.15])
    pi = np.outer(rho, rho).reshape(NSn)
    # marginal GTR: random symmetric flux
    g = np.abs(rng.normal(size=(n, n))) * 0.2
    g = np.triu(g, 1); g = g + g.T
    A = g / rho[:, None]
    A[np.diag_indices(n)] = 0.0
    A[np.diag_indices(n)] = -A.sum(1)
    setup = _setup_n(n, A, rho)
    E, *_ = kernel_basis_dense(n)          # (n_orbits, kernel dim)
    phi0 = setup["phi0"]
    # KNOWN interior coupled generator: convex combo of the A(+)A lift (boundary,
    # zero double-flux) and a strictly-interior lumpable point from the LP
    # (same marginal).  phi_true = (1-w) phi0 + w phi_int is strictly positive and
    # carries genuine, nonzero coupling that PRESERVES the marginal.
    lp = lp_feasibility(setup)
    phi_int = lp["phi"]
    w = 0.5
    phi_true = (1.0 - w) * phi0 + w * phi_int
    theta_true = np.linalg.lstsq(E, phi_true - phi0, rcond=None)[0]
    F_true = flux_from_orbit_n(phi_true, orbit_id, n)
    Q_true = fp_Q_from_flux(F_true, pi)
    tau = np.array([0.05, 0.15, 0.4, 0.9])
    lam, U, Uinv = eig_rev(Q_true, np.clip(pi, 1e-12, None))
    npair = np.zeros((NSn, NSn, tau.shape[0]))
    Ntot = 5_000_000
    for t in range(tau.shape[0]):
        P = (U * np.exp(lam * tau[t])[None, :]) @ Uinv
        joint = pi[:, None] * np.clip(P, 0, None); joint /= joint.sum()
        npair[:, :, t] = Ntot * joint
    res = fit_convex_em_n(npair, tau, setup, n, n_em=80, verbose=False)
    phi_em = res["phi"]
    theta_em = np.linalg.lstsq(E, phi_em - phi0, rcond=None)[0]
    # coupling magnitude: ||theta|| relative
    coup_true = float(np.linalg.norm(theta_true))
    coup_em = float(np.linalg.norm(theta_em))
    theta_corr = float(np.corrcoef(theta_true, theta_em)[0, 1])
    # LL of truth vs fit
    ll_true = _pair_loglik_n(Q_true, pi, tau, npair)
    # interior? min phi vs scale
    min_phi = float(phi_em[phi_em > 0].min())
    interior = bool((phi_em > 1e-8 * phi_em.max()).all())
    return dict(kernel_dim=E.shape[1], coup_true=coup_true, coup_em=coup_em,
                theta_corr=theta_corr,
                ll_true_per=float(ll_true / npair.sum()),
                ll_em_per=float(res["ll"] / npair.sum()),
                min_phi=min_phi, phi_scale=float(phi_em.max()),
                interior=interior,
                relerr_theta=float(np.linalg.norm(theta_em - theta_true)
                                   / max(coup_true, 1e-12)))


# ---- general-n helpers used by recovery experiments ----
def flux_from_orbit_n(phi, orbit_id, n):
    NSn = n * n
    off = ~np.eye(NSn, dtype=bool)
    F = np.zeros((NSn, NSn))
    F[off] = phi[orbit_id[off]]
    return F


def _pair_loglik_n(Q, pi, tau, npair):
    n2 = Q.shape[0]
    lam, U, Uinv = eig_rev(Q, np.clip(pi, 1e-12, None))
    ll = 0.0
    for t in range(tau.shape[0]):
        edge = npair[:, :, t]
        if edge.sum() == 0:
            continue
        P = (U * np.exp(lam * tau[t])[None, :]) @ Uinv
        ll += float((edge * np.log(np.clip(P, 1e-300, None))).sum())
    return ll


def _build_lump_rows_n(orbit_id, n):
    ri, rj, rk, ro = [], [], [], []
    for i in range(n):
        for j in range(n):
            x = i * n + j
            for k in range(n):
                if i == k:
                    continue
                ri.append(i); rj.append(j); rk.append(k)
                ro.append([int(orbit_id[x, k * n + l]) for l in range(n)])
    return np.array(ri), np.array(rj), np.array(rk), np.array(ro)


def _setup_n(n, A, rho):
    """General-n analogue of pair_setup for the recovery experiments."""
    orbit_id, n_orbits, is_single = build_orbits_n(n)
    NSn = n * n
    pi = np.outer(rho, rho).reshape(NSn)
    off = ~np.eye(NSn, dtype=bool)
    F0 = np.zeros((NSn, NSn))
    for j in range(n):
        idx = np.arange(n) * n + j
        F0[np.ix_(idx, idx)] = pi[idx][:, None] * A
    for i in range(n):
        idx = i * n + np.arange(n)
        F0[np.ix_(idx, idx)] += pi[idx][:, None] * A
    np.fill_diagonal(F0, 0.0)
    phi0_seen = np.full(n_orbits, np.nan)
    for o, v in zip(orbit_id[off], F0[off]):
        if np.isnan(phi0_seen[o]):
            phi0_seen[o] = v
    phi0 = np.nan_to_num(phi0_seen, nan=0.0)
    rows = _build_lump_rows_n(orbit_id, n)
    ri, rj, rk, ro = rows
    b_marg = phi0[ro].sum(1)
    return dict(orbit_id=orbit_id, n_orbits=n_orbits, is_single=is_single,
                pi=pi, F0=F0, phi0=phi0, rows=rows, b_marg=b_marg, off=off, n=n)


def orbit_aggregate_n(N, T, pi, orbit_id, n_orbits, off):
    NSn = pi.shape[0]
    oid = orbit_id[off]
    C_o = np.bincount(oid, weights=N[off], minlength=n_orbits)
    Hmat = (T[:, None] / np.maximum(pi[:, None], 1e-300) * np.ones((1, NSn)))
    H_o = np.bincount(oid, weights=Hmat[off], minlength=n_orbits)
    return C_o, H_o


def mstep_convex_n(C_o, H_o, setup, lam=None, sweeps=2000):
    """General-n version of mstep_convex (n small)."""
    ri, rj, rk, ro = setup["rows"]
    b = setup["b_marg"]; n_orbits = setup["n_orbits"]; n = setup["n"]
    n_rows = ro.shape[0]
    if lam is None:
        lam = np.zeros(n_rows)
    BTlam = np.zeros(n_orbits)
    np.add.at(BTlam, ro.ravel(), np.repeat(lam, n))
    d = H_o + BTlam
    active = C_o > 0
    for sw in range(sweeps):
        maxstep = 0.0
        for a in range(n_rows):
            uo, mult = np.unique(ro[a], return_counts=True)
            act = active[uo]
            if not act.any():
                continue
            ou = uo[act]; mu = mult[act].astype(float)
            Cu = C_o[ou]; du = d[ou]; ba = b[a]

            def f(delta):
                return float((mu * Cu / (du + mu * delta)).sum())
            f0 = f(0.0)
            if abs(f0 - ba) <= 1e-13 * (abs(ba) + 1e-30):
                continue
            lo = -(du / mu).min() + 1e-15
            if f0 > ba:
                hi = 1.0
                while f(hi) > ba and hi < 1e30:
                    hi *= 2.0
                a_lo, a_hi = 0.0, hi
            else:
                a_lo, a_hi = lo, 0.0
            for _ in range(100):
                mid = 0.5 * (a_lo + a_hi)
                if f(mid) > ba:
                    a_lo = mid
                else:
                    a_hi = mid
                if a_hi - a_lo < 1e-15 * (abs(mid) + 1e-12):
                    break
            delta = 0.5 * (a_lo + a_hi)
            lam[a] += delta; d[ou] += mu * delta
            maxstep = max(maxstep, abs(delta))
        if maxstep < 1e-14:
            break
    phi = np.where(active, C_o / np.maximum(d, 1e-300), 0.0)
    return phi, lam


def fit_convex_em_n(npair, tau, setup, n, n_em=60, verbose=False, eps=1e-2):
    pi = setup["pi"]; orbit_id = setup["orbit_id"]
    n_orbits = setup["n_orbits"]; off = setup["off"]; is_single = setup["is_single"]
    # strictly-interior init (see interior_init_flux): avoid the absorbing zero
    # double-transition boundary of the pure A(+)A lift.
    phi0 = setup["phi0"].copy()
    scale = eps * phi0[phi0 > 0].mean()
    oid_double = np.unique(orbit_id[off & (~is_single)])
    phi_init = phi0.copy()
    phi_init[oid_double] = np.maximum(phi_init[oid_double], scale)
    F = flux_from_orbit_n(phi_init, orbit_id, n)
    Q = fp_Q_from_flux(F, pi)
    lam = None; hist = []
    for it in range(n_em):
        eig = eig_rev(Q, np.clip(pi, 1e-12, None))
        NSn = n * n
        N = np.zeros((NSn, NSn)); T = np.zeros(NSn)
        for t in range(tau.shape[0]):
            edge = npair[:, :, t]
            if edge.sum() == 0:
                continue
            Tt, Nt, _ = bridge(Q, pi, tau[t], edge, want_N=True, eig=eig)
            N += Nt; T += Tt
        C_o, H_o = orbit_aggregate_n(N, T, pi, orbit_id, n_orbits, off)
        phi, lam = mstep_convex_n(C_o, H_o, setup, lam=lam)
        F = flux_from_orbit_n(phi, orbit_id, n)
        Q = fp_Q_from_flux(F, pi)
        ll = _pair_loglik_n(Q, pi, tau, npair)
        hist.append(ll)
        if verbose:
            print(f"      [emn] it {it} LL={ll:.2f}")
        if it > 4 and abs(hist[-1] - hist[-2]) < 1e-9 * npair.sum():
            break
    return dict(Q=Q, F=F, pi=pi, phi=phi, hist=hist, ll=hist[-1])


# =====================================================================
# Drivers
# =====================================================================
def run_dims():
    print("# ---- kernel dimension verification (HONEST rank vs paper target) ----",
          flush=True)
    print(f"# {'n':>3} {'N_phi':>8} {'rank(B)':>8} {'null(B)':>8} {'kernel':>8} "
          f"{'paper_tgt':>10} {'gap':>5}  ok", flush=True)
    out = {}
    for n in [2, 3, 4, 5, 6, 20]:
        r = verify_dims(n, numeric=True)
        ok = (r.get("ok_Nphi") and r.get("ok_rank", True)
              and r.get("ok_kernel", True))
        print(f"# {n:>3} {r['N_phi']:>8} {r['rank_B']:>8} "
              f"{r.get('null_dim','-'):>8} {r['kernel']:>8} "
              f"{r['kernel_paper_target']:>10} {r['gap']:>5}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        out[n] = r
    print("# kernel = null(B) = (n-1)^2(n^2-2n+4)/4 (honest); paper target "
          "d_n-N_g undercounts by C(n-1,2).", flush=True)
    return out


def run_corpus(corpus, n_em=40, do_lp=True, kappa=0.0):
    print(f"\n# ================= corpus {corpus} =================", flush=True)
    parts = fp.n_parts(corpus)
    train_ids = [i for i in range(parts) if i != parts - 1]
    val_id = parts - 1
    npair, tau, _ = fp.load_parts(corpus, train_ids)
    vpair, _, _ = fp.load_parts(corpus, [val_id])
    # marginal single-site transition counts: derived FROM the pair tensor so the
    # fixed marginal is self-consistent with the pair process.
    nsingle = marginal_counts_from_pair(npair)
    print(f"# train pairs={npair.sum():.3e} val pairs={vpair.sum():.3e} "
          f"marginal-counts={nsingle.sum():.3e}", flush=True)
    t0 = time.time()
    A, rho, g = fit_marginal_gtr(nsingle, tau, iters=100, verbose=False)
    print(f"# marginal GTR fit ({time.time()-t0:.1f}s); rho range "
          f"[{rho.min():.3e},{rho.max():.3e}]", flush=True)
    setup = pair_setup(A, rho)
    # sanity: A(+)A lift satisfies B phi0 = b_marg exactly
    ri, rj, rk, ro = setup["rows"]
    res_lift = np.linalg.norm(setup["phi0"][ro].sum(1) - setup["b_marg"])
    print(f"# lift check ||B phi0 - b_marg|| = {res_lift:.2e}", flush=True)
    # baseline: A(+)A independent (no coupling) held-out
    Q0 = fp.Q_from_flux(setup["F0"], setup["pi"])
    val0 = fp.loglik(Q0, setup["pi"], tau, vpair) / vpair.sum()
    print(f"# A(+)A independent baseline val per-count = {val0:.4f}", flush=True)
    lp = None
    if do_lp:
        t1 = time.time()
        lp = lp_feasibility(setup)
        print(f"# LP max-slack t = {lp['t']:.3e} (status={lp['status']}, "
              f"{lp['message']}) [{time.time()-t1:.1f}s]", flush=True)
    t2 = time.time()
    res = fit_convex_em(npair, tau, setup, n_em=n_em, kappa=kappa)
    train_per = res["ll"] / npair.sum()
    val_per = fp.loglik(res["Q"], res["pi"], tau, vpair) / vpair.sum()
    monotone = bool(np.all(np.diff(res["hist"]) > -1.0))
    print(f"# convex-EM done [{time.time()-t2:.1f}s]  train per={train_per:.4f}  "
          f"VAL per={val_per:.4f}  monotone={monotone}", flush=True)
    bnd = boundary_report(res, setup)
    print(f"# boundary: frac off-diag rates ~0 = {bnd['frac_rates_zero']:.4f}, "
          f"min pos rate = {bnd['min_pos_rate']:.3e}; "
          f"double-orbits zero {bnd['double']['n_zero']}/{bnd['double']['n']}, "
          f"single-orbits zero {bnd['single']['n_zero']}/{bnd['single']['n']}",
          flush=True)
    # coupling magnitude relative to the independent lift
    coup = float(np.linalg.norm(res["phi"] - setup["phi0"])
                 / max(np.linalg.norm(setup["phi0"]), 1e-30))
    tag = corpus.rstrip("/").split("/")[-1]
    import os
    os.makedirs("results/pair_models", exist_ok=True)
    pp = f"results/pair_models/lumpable_kernel_{tag}.npz"
    np.savez(pp, Q=res["Q"], pi=res["pi"], phi=res["phi"], F=res["F"],
             A=A, rho=rho, val_per=val_per, train_per=train_per,
             m_relres=res["m_relres"])
    print(f"# saved {pp}; coupling ||phi-phi0||/||phi0|| = {coup:.4f}", flush=True)
    return dict(corpus=corpus, val0_indep=float(val0), lp=lp,
                train_per=float(train_per), val_per=float(val_per),
                monotone=monotone, boundary=bnd, coupling_rel=coup,
                m_relres=float(res["m_relres"]),
                hist=[float(h) for h in res["hist"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", action="store_true")
    ap.add_argument("--smalln", action="store_true")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--no-lp", action="store_true")
    ap.add_argument("--n-em", type=int, default=40)
    ap.add_argument("--kappa", type=float, default=0.0,
                    help="flux pseudocount (Laplace prior); 0 lets coupling wash out")
    ap.add_argument("--out", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    result = {}
    if args.dims or args.all:
        result["dims"] = run_dims()
    if args.smalln or args.all:
        print("\n# ---- small-n recovery ----", flush=True)
        r2 = recover_n2()
        print(f"# n=2: theta_true={r2['theta_true']:.4f} grid={r2['theta_grid']:.4f} "
              f"em={r2['theta_em']:.4f} | per grid={r2['per_grid']:.5f} "
              f"em={r2['per_em']:.5f}", flush=True)
        result["recover_n2"] = r2
        r4 = recover_n4()
        print(f"# n=4: kernel_dim={r4['kernel_dim']} coup_true={r4['coup_true']:.3f} "
              f"coup_em={r4['coup_em']:.3f} corr={r4['theta_corr']:.4f} "
              f"relerr={r4['relerr_theta']:.4f} interior={r4['interior']} | "
              f"per true={r4['ll_true_per']:.5f} em={r4['ll_em_per']:.5f}", flush=True)
        result["recover_n4"] = r4
    corpora = []
    if args.all:
        corpora = ["data/cherry_counts_trrosetta", "data/cherry_counts_af_full"]
    elif args.corpus:
        corpora = [args.corpus]
    for c in corpora:
        result[c] = run_corpus(c, n_em=args.n_em, do_lp=not args.no_lp,
                               kappa=args.kappa)
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2, default=float)
        print(f"\n# wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
