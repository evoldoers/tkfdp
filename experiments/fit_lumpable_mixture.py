#!/usr/bin/env python3
r"""Mixture of LUMPABLE coupling components + Gamma+I rate heterogeneity over size-2
clusters, fit AS A MIXTURE with responsibility-weighted counts.

The lumpable analogue of fit_coupling_mixture_rateI.py: SAME corpus, family split, mixture
EM, Gamma+I rate latent and held-out scoring (all its rate machinery is imported and reused
as RI.*), EXCEPT the per-component kernel is a full two-sided EXACTLY-LUMPABLE reversible
400-state pair CTMC (Klein-orbit flux F_c + symmetric stationary pi_c, fit by the constrained
lumpable flux/pi M-steps of fit_pair_models) instead of metropolis_sqrt(S,pi_c).  There is no
shared exchangeability -- each component is a free lumpable chain.

Two independent latents, exactly as in the Metropolis rateI mixture:
  * pairing class c in {1..K}, weight w_c : component Q_c = lumpable chain (flux F_c, pi_c)
  * rate class r : discrete Gamma+Invariant grid (Yang), weight rho_r, PER-CLUSTER, orthogonal
    to c.  The invariant bin (rate 0) supports only fully-conserved clusters.
    L(g) = sum_c sum_r w_c rho_r prod_e P_c(rate_r*tau_e)[pf_e->pt_e]^{n_g^e}.

Each component is warm-started at the HIGHLY-DEGENERATE RENEWAL point F81(pi_c):
Q_xy = pi_c(y), flux F = pi_c(x)pi_c(y), EXACTLY lumpable (fit_pair_models.renewal_init).  The
pi_c seeds come from a fitted Metropolis-sqrt coupling mixture at the SAME K
(--init-mixture components_K{K}.npz), so this refits the same K coupling archetypes under the
richer lumpable kernel WITH Gamma+I, comparable head-to-head against fit_coupling_mixture_rateI.

M-step per EM iter: E-step responsibilities (RI.build_scores over c,r) -> exact w_c, rho_r,
p_inv -> per component a JOINT (pi_c, flux) two-sided-lumpable M-step that INCREASES the
expected complete-data log-likelihood L(phi,pi) = sum_{x!=y} N_xy log(F_xy/pi_x) - sum_x
T_x R_x/pi_x (so EM is monotone).  The lumpability manifold itself depends on pi (block-sums
must be prop. to pi_ij), so pi and the flux are fit together: pi is damped from the old value
toward its ML target pi_ml = T_x R_x/N_{x+} (the DYNAMICS-based coordinate maximiser, which
carries the coupling -- not occupancy), and at each candidate pi the EXACT flux is re-fit by
mstep_schur_newton -- an equality-constrained Newton on max sum_o[C log phi - H phi] s.t.
P_perp(B phi)=0, solving the dual Schur complement S = P_perp B diag(phi^2/C) B^T P_perp
DIRECTLY (fp64 LU on GPU; S formed by scatter-add to dodge the crippled A6000 fp64 matmul).
Because the KKT system is solved exactly, phi stays on the lumpable manifold (relres ~1e-7 =
machine-zero, KL departure ~1e-13 nats).  The t=0 candidate is warm-started from the previous
(feasible) flux so it can only climb above L_old -- the monotone anchor -- while a larger pi
step is taken whenever it increases L.  Then the 1-D Gamma-shape alpha CM-step
(RI.refit_alpha).  Resumable (atomic per-iter checkpoint).  Run with PYTHONPATH=src."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import sys
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)                     # fp64 for the machine-zero solve
import jax.numpy as jnp
import jax.scipy.linalg
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import fit_pair_models as FP                                    # noqa: E402
from fit_pair_models import NA, NS                              # noqa: E402
import fit_coupling_mixture_freeS as FS                         # noqa: E402  (mi, load_split_core)
import fit_coupling_mixture_rateI as RI                         # noqa: E402  (Gamma+I machinery)
import fit_lumpable_kernel as LK                                # noqa: E402  (EXACT convex M-step)
from tkfdp.coupling.dynfield.phylo_elbo.rate_hetero import discrete_gamma_rates  # noqa: E402


def _cg(Hv, b, iters, tol=1e-10):
    x = np.zeros_like(b); r = b.copy(); p = r.copy(); rs = float(r @ r); b2 = max(float(b @ b), 1e-30)
    for _ in range(iters):
        Hp = Hv(p); a = rs / max(float(p @ Hp), 1e-30)
        x += a * p; r -= a * Hp; rs2 = float(r @ r)
        if rs2 < tol * tol * b2:
            break
        p = r + (rs2 / rs) * p; rs = rs2
    return x


_BBT = {}                                                     # cache the B B^T scatter pattern per setup


def _bbt_pattern(st, nr):
    """Precompute the sparsity pattern of B diag(.) B^T as a scatter: for each orbit o (a column
    of B with rows rows_o), it contributes to every (a,a') in rows_o x rows_o.  Returns device
    arrays (flat_ij, orbit) so that M0 = scatter-add(s[orbit] -> flat index a*nr+a').  This forms
    the fp64 Schur matrix WITHOUT the crippled A6000 fp64 dense matmul (~8 s); scatter is ~ms."""
    k = id(st)
    if k not in _BBT:
        Bs = st["B_sparse"] if st["B_sparse"].shape[0] == nr else st["B_sparse"].T
        Bc = Bs.tocsc()
        I, J, O = [], [], []
        for o in range(Bc.shape[1]):
            rows = Bc.indices[Bc.indptr[o]:Bc.indptr[o + 1]]
            if len(rows) == 0:
                continue
            I.append(np.repeat(rows, len(rows))); J.append(np.tile(rows, len(rows)))
            O.append(np.full(len(rows) ** 2, o))
        flat = jnp.asarray(np.concatenate(I) * nr + np.concatenate(J))
        _BBT[k] = (flat, jnp.asarray(np.concatenate(O)))
    return _BBT[k]


@partial(jax.jit, static_argnums=(4,))
def _schur_S(flat_ij, orbit, Ug, s, nr, ridge):
    """Form S = P_perp B diag(s) B^T P_perp + ridge*(tr/n) I (fp64) via scatter-add (no dense fp64
    matmul).  P_perp = I - Ug Ug^T (Ug fp64, orthonormal per-(i,k)-block pi-directions)."""
    M0 = jnp.zeros(nr * nr, dtype=jnp.float64).at[flat_ij].add(s[orbit]).reshape(nr, nr)
    UtM = Ug.T @ M0; MU = M0 @ Ug
    S = M0 - Ug @ UtM - MU @ Ug.T + Ug @ ((Ug.T @ MU) @ Ug.T)
    rs = ridge * (jnp.trace(S) / nr)
    return S + rs * jnp.eye(nr), rs


def mstep_schur_newton(C_o, H_o, pi_c, st, newton=20, kappa=0.0, ridge=1e-10, tol=1e-9,
                       otol=1e-8, phi0=None):
    """EXACT free-marginal two-sided-lumpable flux M-step by equality-constrained Newton with a
    DIRECT solve of the dual Schur complement -- the machine-zero solver (relres ~1e-7).

    Maximise sum_o[C_o log phi_o - H_o phi_o] s.t. A phi = P_perp(B phi) = 0, phi>=0.  Positivity
    is free: the C log phi term is a log-barrier, so on the support (C_o>0) the optimum keeps
    phi>0.  It is thus a pure equality-constrained concave max; the KKT/Newton step solves
        S nu = A H^{-1} g ,   d phi = H^{-1}(g - A^T nu) ,   H = diag(C/phi^2),
    with S = A H^{-1} A^T = P_perp B diag(phi^2/C) B^T P_perp -- only m=n_rows (~7600) square
    (~460 MB dense).  S is FORMED by a scatter-add over the precomputed B B^T pattern (fp64,
    ~ms; dodges the A6000 crippled fp64 dense matmul) and factorised by fp64 LU on GPU each
    Newton step (S has a P_perp null space -> not SPD, so LU with pivoting, not Cholesky).
    Because the KKT system is solved DIRECTLY (not by CG), A d phi = 0 to ~fp64 precision, so
    from the feasible renewal flux every iterate stays lumpable (relres ~1e-7, KL departure
    ~1e-13 nats) while the objective climbs -- unlike the matrix-free CG dual (mstep_convex_proj)
    which drifts/crawls at ~1e-3.  P_perp = I - U U^T, U the orthonormal per-(i,k)-block
    pi-directions.  Warm-startable from a feasible phi0.  Returns (phi, relres)."""
    ri, rj, rk, ro = st["rows"]; n_orbits = st["n_orbits"]; off = st["off"]; oid = st["orbit_id"]
    nr = ro.shape[0]; sup = C_o > 0; Ce = C_o + kappa
    Bs = st["B_sparse"] if st["B_sparse"].shape[0] == nr else st["B_sparse"].T
    pr = pi_c.reshape(NA, NA)[ri, rj]; ikb = ri * NA + rk
    _, inv = np.unique(ikb, return_inverse=True); nb = int(inv.max()) + 1
    nrm = np.sqrt(np.maximum(np.bincount(inv, weights=pr * pr), 1e-300))
    U = sp.csr_matrix((pr / nrm[inv], (np.arange(nr), inv)), shape=(nr, nb))   # orthonormal cols

    def Bf(v):
        return v[ro].sum(1)

    def Ppv(v):                                              # P_perp on a row-vector
        return v - U @ (U.T @ v)

    def PpM(M):                                              # P_perp on both sides of dense M
        UtM = U.T @ M; MU = M @ U
        return M - U @ UtM - MU @ U.T + U @ ((U.T @ MU) @ U.T)

    def relres(phi):
        Bphi = Bf(phi); return float(np.linalg.norm(Ppv(Bphi)) / max(np.linalg.norm(Bphi), 1e-30))

    if phi0 is not None:                                     # warm start (must be feasible at pi_c)
        phi = np.maximum(phi0, 1e-30)
    else:
        p2, Fr = FP.renewal_init(pi_c)                       # feasible start (Bphi = pi_ij rho_k)
        phi = np.zeros(n_orbits); phi[oid[off]] = Fr[off]; phi = np.maximum(phi, 1e-30)
    def obj(p):
        return float((Ce[sup] * np.log(np.maximum(p[sup], 1e-300)) - H_o[sup] * p[sup]).sum())

    flat_ij, orbit = _bbt_pattern(st, nr)                     # B B^T scatter pattern (device)
    Ug = jnp.asarray(U.toarray(), dtype=jnp.float64)          # U (fp64) on device
    rr = relres(phi)
    for _ in range(newton):
        if rr < tol and _ > 0:
            break
        g = np.where(sup, Ce / np.maximum(phi, 1e-300) - H_o, 0.0)
        s = np.where(sup, phi * phi / np.maximum(Ce, 1e-300), 0.0)
        rhs = Ppv(Bf(s * g))
        S64, _rs = _schur_S(flat_ij, orbit, Ug, jnp.asarray(s), nr, ridge)   # exact fp64 Schur (GPU)
        nu = Ppv(np.asarray(jnp.linalg.solve(S64, jnp.asarray(rhs))))        # fp64 solve; drop null(U)
        dphi = s * (g - Bs.T @ Ppv(nu))
        f0 = obj(phi); a = 1.0; ok = False
        for _ in range(50):                                 # backtracking, keep phi>0 & ascend
            pt = phi + a * dphi
            if np.all(pt[sup] > 0):
                ft = obj(pt)
                if ft >= f0 - 1e-12:
                    phi = pt; ok = True; break
            a *= 0.5
        if not ok:
            break
        rr = relres(phi)
        if _ > 0 and ft - f0 < otol * max(abs(f0), 1.0) and rr < 1e-4:   # objective converged
            break
    return phi, relres(phi)


def mstep_convex_proj(C_o, H_o, pi_c, st, newton=60, cg_iters=200, kappa=0.0, tol=1e-8):
    """EXACT free-marginal lumpable flux M-step: maximise sum_o[C_o log phi_o - H_o phi_o]
    s.t. the PROJECTED lumpability constraint P_perp(B phi)=0 (i.e. B phi in range D(pi): the
    block row-sums are proportional across the spectator j to pi_{i.}, marginal A FREE), phi>=0.
    Solved on the convex dual nu (in the perp space) by a matrix-free CG-Newton:
      phi_o(nu) = (C_o+kappa)/(H_o + (B^T P_perp nu)_o),  grad = -P_perp(B phi),
      Hess u = P_perp B diag(phi^2/(C+kappa)) B^T P_perp u  (PSD).
    At the optimum P_perp(B phi)=0 EXACTLY (an exactly two-sided-lumpable chain; lumpresid->0)
    with the marginal A_ik = (B phi)_ijk/pi_ij read off, unlike the fixed-b_marg mstep_convex
    which pins A and leaves a feasibility residual.  Returns (phi, relres)."""
    ri, rj, rk, ro = st["rows"]; n_orbits = st["n_orbits"]
    pr = pi_c.reshape(NA, NA)[ri, rj]; ik = ri * NA + rk; nik = NA * NA
    ror = ro.ravel(); rep = np.repeat(np.arange(len(ri)), NA)  # unused placeholder-safe
    Ce = C_o + kappa

    def Bf(v):
        return v[ro].sum(1)

    def BT(v):
        o = np.zeros(n_orbits); np.add.at(o, ror, np.repeat(v, NA)); return o

    def Pp(v):
        return FP._perp(v, pr, ik, nik)

    def Psi(nn):
        d = H_o + BT(Pp(nn))
        return np.inf if np.any(d <= 0) else -float((Ce * np.log(np.maximum(d, 1e-300))).sum())

    # COLD start each M-step: H_o and pi_c shift substantially between EM iters (the
    # responsibilities change), so a warm-started dual drives the denominator negative and
    # blows phi up -- cold start is the robust choice, with `newton` big enough to converge.
    nu = np.zeros(len(ri)); relres = np.inf
    for _ in range(newton):
        d = np.maximum(H_o + BT(Pp(nu)), 1e-30); phi = Ce / d
        Bphi = Bf(phi); res = Pp(Bphi)
        relres = float(np.linalg.norm(res) / max(np.linalg.norm(Bphi), 1e-30))
        if relres < tol:
            break
        w = phi * phi / np.maximum(Ce, 1e-300)
        step = _cg(lambda u: Pp(Bf(w * BT(Pp(u)))) + 1e-12 * u, res, cg_iters)
        f0 = Psi(nu); a = 1.0                                   # backtracking line search (min Psi)
        for _ in range(40):
            if Psi(nu + a * step) <= f0:
                break
            a *= 0.5
        nu = Pp(nu + a * step)
    d = np.maximum(H_o + BT(Pp(nu)), 1e-30); phi = Ce / d
    relres = float(np.linalg.norm(Pp(Bf(phi))) / max(np.linalg.norm(Bf(phi)), 1e-30))
    return phi, relres


def marginal_gtr_fit(N, T, rho):
    """Closed-form reversible single-site (marginal) generator A from pair HR stats, used to
    pin each lumpable component's product-stationary marginal.  Directed marginal usage U
    (both exchangeable coordinates) and dwell D give the reversible exchangeability
    s_ab = (U_ab+U_ba)/(D_a rho_b + D_b rho_a); returns the rate matrix A_ab = s_ab rho_b."""
    N4 = N.reshape(NA, NA, NA, NA)
    U = N4.sum((1, 3)) + N4.sum((0, 2))                        # coord-1 + coord-2 usage a->b
    np.fill_diagonal(U, 0.0)
    T2 = T.reshape(NA, NA); D = T2.sum(1) + T2.sum(0)          # marginal dwell per residue
    den = D[:, None] * rho[None, :] + rho[:, None] * D[None, :]
    s = np.where(den > 0, (U + U.T) / np.maximum(den, 1e-300), 0.0)
    np.fill_diagonal(s, 0.0)
    return s * rho[None, :]


def component_setup(pi_pair, A, st):
    """Per-component LK setup for a CORRELATED (non-product) joint stationary pi_pair with a
    FITTED single-site marginal generator A (A_ik, reversible on rho=marg(pi_pair)).  The
    two-sided lumpability RHS is (B phi)_{ijk} = pi_ij * A_ik, so b_marg_{ijk} = pi_ij A_ik --
    the marginal is fit from the data (marginal_gtr_fit), NOT frozen at the F81 renewal rate
    rho_k.  For non-product pi and a non-renewal A exact two-sided lumpability can be slightly
    infeasible (the 0/K result); mstep_convex returns the max-likelihood flux closest to the
    constraint (relres reports the residual), which the double-transition coupling absorbs."""
    ri, rj, rk = st["rows"][0], st["rows"][1], st["rows"][2]
    b_marg = pi_pair.reshape(NA, NA)[ri, rj] * A[ri, rk]
    setup = dict(st); setup["pi"] = pi_pair; setup["b_marg"] = b_marg
    return setup


def metropolis_init(pi_seed):
    """Init keeping the FULL (coupled) seed stationary pi_seed -- e.g. a Metropolis mixture
    component's pi_c, MI>0 -- with a feasible F81 flux at that pi_c.  Used (with FREEZEPI) to test
    whether a lumpable chain can reach good likelihood WITH a coupled stationary, or whether ML
    drives pi back to product regardless.  Returns (pi_pair, F)."""
    pim = 0.5 * (pi_seed.reshape(NA, NA) + pi_seed.reshape(NA, NA).T)
    pim = np.maximum(pim, 1e-12); pim = (pim / pim.sum()).reshape(NS)
    _, F = FP.renewal_init(pim)
    return pim, F


def independent_init(pi_seed):
    """Independent-sites init for a lumpable component: product stationary rho(x)rho(y), with
    rho the marginal of pi_seed, and single-site LG08-form dynamics on each coordinate (an exact
    two-sided-lumpable chain -- independent projection is Markov).  Per class this IS the
    independence null, so EM (monotone in the observed likelihood) starting here cannot converge
    below independent LG08 sites; the earlier F81 renewal warm start has a poor marginal and drops
    EM into a worse local basin.  Returns (pi_pair, F) with F the symmetric off-diagonal flux."""
    from tkfdp import lg08 as LG
    S = np.asarray(LG.S_LG08, float)
    rho = pi_seed.reshape(NA, NA).sum(1); rho = rho / rho.sum()
    Q1 = S * rho[None, :]; np.fill_diagonal(Q1, 0.0); Q1[np.diag_indices(NA)] = -Q1.sum(1)
    Qp = np.kron(Q1, np.eye(NA)) + np.kron(np.eye(NA), Q1)          # independent-sites generator
    pip = np.kron(rho, rho)                                         # product stationary
    F = pip[:, None] * Qp; np.fill_diagonal(F, 0.0)                 # F_xy = pi_x Q_xy, symmetric
    return pip, F


def _full_cdll(N_c, T_c, F, pi):
    """Expected complete-data log-likelihood of the reversible pair chain Q=F/pi given the HR
    sufficient statistics (usage N_xy, dwell T_x): sum_{x!=y} N_xy log(F_xy/pi_x) - sum_x T_x
    R_x/pi_x, R_x = sum_{y!=x} F_xy.  This is the SINGLE objective the joint (pi,phi) M-step
    must increase for EM monotonicity -- both the flux and the stationary enter it, and it is
    exactly the flux objective sum_o[C_o log phi_o - H_o(pi) phi_o] minus sum_x N_x+ log pi_x."""
    pic = np.maximum(pi, 1e-300); R = F.sum(1)
    m = N_c > 0
    xrow = np.broadcast_to(np.arange(NS)[:, None], N_c.shape)
    usage = float((N_c[m] * (np.log(np.maximum(F[m], 1e-300)) - np.log(pic[xrow[m]]))).sum())
    expo = float((T_c * R / pic).sum())
    return usage - expo


def mstep_lumpable_components(R_full, rate_vals, gamma_idx, pis, Fs, tau, trPF, trPT,
                             trSEG, trCNT, flat, T, st, fm_newton, guard=0.05):
    """Per-component EXACT lumpable M-step at fixed responsibilities R_full (G,K,R).  Per
    component: (1) aggregate HR usage/dwell over the Gamma rate bins (branch rate_r*tau) on
    the responsibility-weighted counts; (2) the CORRELATED (non-product) joint stationary
    pi_c from the responsibility-weighted occupancy -- this carries the coupling, MI(pi_c)>0;
    (3) the GLOBAL convex coupling M-step LK.mstep_convex (rational Sinkhorn/IPF dual on the
    null(B) lumpable kernel) -- the exact maximiser of sum_o[C_o log phi_o - H_o phi_o] s.t.
    B phi = pi_ij rho_k (two-sided lumpability with the F81 renewal marginal), phi>=0,
    replacing the penalty-based ALM.  Initialised at F81(pi_c) (exactly lumpable for any pi_c)
    and moving toward coupled double-transition dynamics.  Invariant bin excluded (gamma_idx)."""
    K = len(pis)
    orbit_id = st["orbit_id"]; n_orbits = st["n_orbits"]; off = st["off"]
    new_pis, new_Fs, Qs, eigs = [], [], [], []
    rr_max = 0.0
    for c in range(K):
        Q_c = FP.Q_from_flux(Fs[c], pis[c])
        eig_c = FP.eig_rev(Q_c, np.clip(pis[c], 1e-12, None))
        N_c = np.zeros((NS, NS)); T_c = np.zeros(NS)
        for r in gamma_idx:                                    # HR stats over Gamma rate bins
            wnnz = trCNT * R_full[trSEG, c, r]
            if wnnz.sum() <= 0:
                continue
            Ncr = np.bincount(flat, weights=wnnz, minlength=NS * NS * T).reshape(NS, NS, T)
            Nn, Tt = RI.estep_scaled(Q_c, pis[c], tau, rate_vals[r], Ncr, eig_c)
            N_c += Nn; T_c += Tt
        # JOINT (pi, phi) two-sided-lumpable M-step, guarded on the complete-data LL.  The
        # lumpability manifold itself depends on pi (the block-sums must be prop. to pi_ij), so
        # updating pi by a separate formula makes the old flux infeasible and breaks EM ascent.
        # Instead: the ML-target stationary pi_ml = T_x R_x/N_{x+} (the coordinate maximiser of
        # the complete-data LL over pi given the flux, estimated from the DYNAMICS which carry
        # the coupling -- not occupancy), then damp pi from the old value toward pi_ml and RE-FIT
        # the exact flux (mstep_schur_newton, machine-zero lumpable) at each candidate pi, keeping
        # every candidate feasible.  Accept the first t that increases the complete-data LL;
        # t=0 (refit at the old pi) is always >= L_old (the flux step alone improves it), so EM
        # monotonicity is guaranteed while pi still advances when a larger step helps.
        r_x = Fs[c].sum(1); Nout = N_c.sum(1)
        pi_ml = np.where(Nout > 0, T_c * r_x / np.maximum(Nout, 1e-300), pis[c])
        pim = 0.5 * (pi_ml.reshape(NA, NA) + pi_ml.reshape(NA, NA).T)      # component-exchangeable
        pim = np.maximum(pim, 1e-12); pi_ml = (pim / pim.sum()).reshape(NS)
        L_old = _full_cdll(N_c, T_c, Fs[c], pis[c])
        phi_prev = np.zeros(n_orbits); phi_prev[st["orbit_id"][off]] = Fs[c][off]   # warm-start seed
        best = None
        _ts = (0.0,) if os.environ.get("FREEZEPI") else (1.0, 0.0)   # FREEZEPI: hold pi at init
        for t in _ts:                                          # t=0 (warm) is the monotone anchor
            pi_t = (1.0 - t) * pis[c] + t * pi_ml              # convex combo: symmetric, sums to 1
            C_o, H_o = LK.orbit_aggregate(N_c, T_c, pi_t, orbit_id, n_orbits, off)
            kappa = 1e-6 * max(C_o.sum(), 1.0) / n_orbits
            # t=0: warm-start from the previous (feasible) flux so the exact solve can only CLIMB
            # above L_old -- guaranteeing the monotone anchor.  t>0: pi moved, previous flux is
            # infeasible there, so cold-start from the renewal flux at pi_t.  Take the best over
            # t (no early break): pi advances whenever a damped step beats the anchor.
            phi, rr = mstep_schur_newton(C_o, H_o, pi_t, st, newton=fm_newton, kappa=kappa,
                                         phi0=(phi_prev if t == 0.0 else None))
            F_t = LK.flux_from_orbit(phi, orbit_id, off)
            L_t = _full_cdll(N_c, T_c, F_t, pi_t)
            if best is None or L_t > best[3]:
                best = (F_t, pi_t, rr, L_t)
        Fc, pi_c, rr, L_best = best
        Qc = FP.Q_from_flux(Fc, pi_c)
        bad = (not np.all(np.isfinite(Qc))) or np.abs(Qc).max() > 1e6 or rr > guard
        if bad:                                                # non-lumpable/blow-up: keep prior
            print(f"    [guard] component {c}: relres={rr:.1e} max|Q|={np.abs(Qc).max():.1e} "
                  f"> tol; keeping previous (lumpable) generator", flush=True)
            Fc = Fs[c]; pi_c = pis[c]; Qc = FP.Q_from_flux(Fc, pi_c)
        else:
            rr_max = max(rr_max, rr)
        ec = FP.eig_rev(Qc, np.clip(pi_c, 1e-12, None))
        new_pis.append(pi_c); new_Fs.append(Fc); Qs.append(Qc); eigs.append(ec)
    return new_pis, new_Fs, Qs, eigs, rr_max


def save_resume(path, pis, Fs, w, rho, alpha, it, extra):
    tmp = path + ".tmp"
    np.savez(tmp, pis=np.asarray(pis), Fs=np.asarray(Fs), w=np.asarray(w),
             rho=np.asarray(rho), alpha=alpha, it=it, **extra)
    os.replace(tmp + ".npz", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="data/per_contact_trrosetta/counts.npz")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--init-mixture", default=None,
                    help="components_K{K}.npz whose pi_c seed the F81 renewal warm start "
                         "(default results/mixture_component_char/components_K{K}.npz)")
    ap.add_argument("--gamma-cats", type=int, default=4, help="discrete-Gamma rate categories")
    ap.add_argument("--pinv", type=float, default=0.2, help="initial invariant weight")
    ap.add_argument("--single-rate", action="store_true",
                    help="collapse the rate grid to one bin at rate 1 (no I, no alpha)")
    ap.add_argument("--alpha-init", type=float, default=1.0)
    ap.add_argument("--alpha-every", type=int, default=2, help="refit Gamma shape every N iters")
    ap.add_argument("--em-iters", type=int, default=30)
    ap.add_argument("--m-sweeps", type=int, default=60,
                    help="(unused; kept for CLI compat)")
    ap.add_argument("--fm-newton", type=int, default=60,
                    help="free-marginal projected-dual CG-Newton iterations per component M-step")
    ap.add_argument("--alm-outer", type=int, default=12, help="(deprecated; unused, kept for CLI compat)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-counts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--resume", default=None, help="resume npz (atomic per-iter checkpoint)")
    ap.add_argument("--wall-budget", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); t0 = time.time()
    K = a.K
    mode = "single-rate" if a.single_rate else f"gammaI(g{a.gamma_cats})"
    init_path = a.init_mixture or f"results/mixture_component_char/components_K{K}.npz"
    ALPHA_CANDS = (0.6, 0.78, 1.0, 1.28, 1.66)

    (PF, PT, SEG, tau, T, G, tot_cnt, keep,
     trPF, trPT, trTB, trCNT, trSEG, tr_g, tr_tot,
     vaPF, vaPT, vaTB, vaCNT, vaSEG, va_g, va_tot) = FS.load_split_core(
        a.corpus, rng, a.val_frac, a.min_counts)
    conserved = np.bincount(SEG, weights=(PF != PT).astype(float), minlength=G) == 0
    flat = (trPF * NS + trPT) * T + trTB
    print(f"# [lumpable-mix {mode} K={K}] {G} clusters ({keep.sum()} kept, "
          f"{int(conserved[keep].sum())} conserved); train {len(tr_g)}/val {len(va_g)}; "
          f"{int(tr_tot):,}/{int(va_tot):,} transitions", flush=True)

    # static LK machinery (orbits, lumpability rows, B / rowdat) -- built once, marginal-
    # independent; per-component setups reuse it and only recompute b_marg.
    st = LK.pair_setup(np.zeros((NA, NA)), np.full(NA, 1.0 / NA))
    print(f"# orbits {st['n_orbits']}; EXACT free-marginal two-sided-lumpable M-step (Schur-"
          f"Cholesky equality-constrained Newton; lumpresid ~1e-9 = machine-zero, free LG-like "
          f"marginal, NOT renewal), fm-newton={a.fm_newton}", flush=True)

    # ---- Gamma+I rate grid (identical to fit_coupling_mixture_rateI) ----
    if a.single_rate:
        R_gamma = 1; rate_vals = np.array([1.0]); is_inv = np.array([False])
        rho = np.array([1.0]); alpha = float("nan"); p_inv = 0.0
    else:
        R_gamma = a.gamma_cats; alpha = a.alpha_init
        g = discrete_gamma_rates(R_gamma, alpha)
        rate_vals = np.concatenate([[0.0], g]); is_inv = np.array([True] + [False] * R_gamma)
        p_inv = a.pinv
        rho = np.concatenate([[p_inv], np.full(R_gamma, (1 - p_inv) / R_gamma)])
    R = len(rate_vals); gamma_idx = [r for r in range(R) if not is_inv[r]]

    # ---- init: F81 renewal warm start from the Metropolis mixture pi_c (or resume) ----
    start_it = 0
    if a.resume and os.path.exists(a.resume):
        z = np.load(a.resume, allow_pickle=True)
        pis = [np.asarray(p, float) for p in z["pis"]]
        Fs = [np.asarray(f, float) for f in z["Fs"]]
        w = np.asarray(z["w"], float); rho = np.asarray(z["rho"], float)
        alpha = float(z["alpha"]); start_it = int(z["it"]) + 1
        if "rate_vals" in z.files:
            rate_vals = np.asarray(z["rate_vals"], float)
        print(f"# resumed from {a.resume} at iter {start_it}", flush=True)
    else:
        zc = np.load(init_path, allow_pickle=True)
        assert int(zc["K"]) == K, f"init K {int(zc['K'])} != requested K {K}"
        seed_pis = np.asarray(zc["pis"], float); w = np.asarray(zc["weights"], float).copy()
        pis, Fs = [], []
        _pinit = metropolis_init if os.environ.get("PIINIT") == "metropolis" else independent_init
        for c in range(K):
            p, Fc = _pinit(seed_pis[c])                         # product pi + LG08 single-site (default)
            pis.append(p); Fs.append(Fc)
        print(f"# warm-started K={K} lumpable components at INDEPENDENT-sites (product pi + LG08), "
              f"per-class composition from {init_path} -- nests the independence null so EM cannot "
              f"converge below it (unlike the F81 renewal init, which drops into a poor-marginal basin)", flush=True)
    Qs = [FP.Q_from_flux(Fs[c], pis[c]) for c in range(K)]
    eigs = [FP.eig_rev(Qs[c], np.clip(pis[c], 1e-12, None)) for c in range(K)]

    prev = None; trll = float("nan"); monotone = True; it = start_it - 1
    for it in range(start_it, a.em_iters):
        logw = np.log(w + 1e-300); logrho = np.log(rho + 1e-300)
        sc = RI.build_scores(eigs, pis, rate_vals, is_inv, logw, logrho, tau,
                             trPF, trPT, trTB, trCNT, trSEG, G, tr_g, conserved)
        trll = RI.marginal_percount_ll(sc, tr_tot)
        if prev is not None and trll < prev - 1e-9:
            monotone = False
        Rr = RI.responsibilities(sc)
        R_full = np.zeros((G, K, R)); R_full[tr_g] = Rr
        w = Rr.sum((0, 2)) + 1e-9; w /= w.sum()
        rho = Rr.sum((0, 1)) + 1e-12; rho /= rho.sum()
        if not a.single_rate:
            p_inv = float(rho[0])
        pis, Fs, Qs, eigs, rr_max = mstep_lumpable_components(
            R_full, rate_vals, gamma_idx, pis, Fs, tau, trPF, trPT, trSEG, trCNT, flat, T,
            st, a.fm_newton)
        if (not a.single_rate and R_gamma >= 2 and it % a.alpha_every == 0):
            alpha, g = RI.refit_alpha(alpha, R_gamma, eigs, R_full, gamma_idx, tau,
                                      trPF, trPT, trTB, trCNT, trSEG, G, tr_g, ALPHA_CANDS)
            rate_vals = np.concatenate([[0.0], g])
        extra = "" if a.single_rate else f" alpha={alpha:.3f} pinv={p_inv:.3f}"
        print(f"  it {it:2d}: train_mix={trll:.4f}  w={np.round(np.sort(w)[::-1], 3)}  "
              f"MI(pi)={sorted([round(FS.mi(p), 3) for p in pis], reverse=True)}{extra}  "
              f"lumpresid={rr_max:.1e}  [{time.time()-t0:.0f}s]", flush=True)
        if a.resume:
            save_resume(a.resume, pis, Fs, w, rho, alpha, it, dict(K=K, rate_vals=rate_vals))
        if prev is not None and abs(trll - prev) < a.tol:
            print("  converged", flush=True); break
        prev = trll
        if a.wall_budget and time.time() - t0 > a.wall_budget:
            print(f"# wall budget {a.wall_budget}s hit at iter {it}; checkpointed", flush=True)
            break

    # ---- final held-out + diagnostics ----
    logw = np.log(w + 1e-300); logrho = np.log(rho + 1e-300)
    sc_va = RI.build_scores(eigs, pis, rate_vals, is_inv, logw, logrho, tau,
                            vaPF, vaPT, vaTB, vaCNT, vaSEG, G, va_g, conserved)
    vall = RI.marginal_percount_ll(sc_va, va_tot)
    sc_tr = RI.build_scores(eigs, pis, rate_vals, is_inv, logw, logrho, tau,
                            trPF, trPT, trTB, trCNT, trSEG, G, tr_g, conserved)
    R_full = np.zeros((G, K, R)); R_full[tr_g] = RI.responsibilities(sc_tr)
    cr_mi, cr_nmi, Hc, Hr = RI.class_rate_mi(R_full, tr_g, w, rho)
    order = np.argsort(w)[::-1]
    res = dict(model="lumpable_mixture", mode=mode, K=K, single_rate=bool(a.single_rate),
               gamma_cats=int(R_gamma), val_per_count_ll=float(vall),
               train_per_count_ll=float(trll), monotone=bool(monotone), n_em_iters=int(it + 1),
               weights=[float(w[c]) for c in order],
               mi_pi=[float(FS.mi(pis[c])) for c in order],
               mi_weighted=float(sum(w[c] * FS.mi(pis[c]) for c in range(K))),
               rate_weights=[float(x) for x in rho], rate_vals=[float(x) for x in rate_vals],
               alpha=(None if a.single_rate else float(alpha)),
               p_inv=(None if a.single_rate else float(p_inv)),
               mean_rate=float(sum(rho[r] * rate_vals[r] for r in range(R))),
               class_rate_mi_nats=cr_mi, class_rate_nmi=cr_nmi, H_class=Hc, H_rate=Hr,
               n_train_clusters=int(len(tr_g)), n_val_clusters=int(len(va_g)),
               init_mixture=init_path, alm_outer=a.alm_outer, seed=a.seed)
    print(f"# [lumpable-mix {mode}] K={K}: VAL/count={vall:.4f} train={trll:.4f} "
          f"monotone={monotone} iters={it+1}", flush=True)
    if not a.single_rate:
        print(f"#   alpha={alpha:.3f} p_inv={p_inv:.3f} rate_w={[round(x,3) for x in rho]}", flush=True)
        print(f"#   class<->rate MI={cr_mi:.4f} nats  NMI={cr_nmi:.4f}", flush=True)
    print(f"#   w-mean MI(pi_c)={res['mi_weighted']:.4f}", flush=True)
    print(f"#   MI(pi_c) per comp = {[round(x, 3) for x in res['mi_pi']]}", flush=True)
    print(f"#   weights  per comp = {[round(x, 3) for x in res['weights']]}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"# wrote {a.out}", flush=True)
    return res


if __name__ == "__main__":
    main()
