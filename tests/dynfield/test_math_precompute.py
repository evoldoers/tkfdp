"""Numerical verification of the dynamic latent-field math (Phase A of
the dynfield implementation plan; see docs/dynfield_math.md).

Self-contained: no tkfdp imports yet — the implementation does not
exist. These tests pin down the math the implementation must respect:

  1. F81-on-DP detailed balance.
  2. Strong-lumping consistency on the occupied + tail block.
  3. Cap-2 joint cross-check vs brute Kronecker at the no-jump limit.
  4. Cap-2 joint via the L^3 sum vs an explicit jump-trajectory
     enumeration.
  5. Marginal consistency of the field-marginal joint => no Sinkhorn
     correction needed.

Run: python3 tests/dynfield/test_math_precompute.py
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.linalg import expm


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------

def make_field_rho(L: int, alpha: float = 1.0, seed: int = 0) -> np.ndarray:
    """Truncated stick-breaking weights for a DP with concentration alpha,
    truncated at L atoms. Returns (L,) normalised."""
    rng = np.random.default_rng(seed)
    betas = rng.beta(1.0, alpha, size=L - 1)
    rho = np.empty(L, dtype=np.float64)
    remaining = 1.0
    for i in range(L - 1):
        rho[i] = remaining * betas[i]
        remaining *= 1.0 - betas[i]
    rho[L - 1] = remaining
    return rho


def f81_generator(rho: np.ndarray) -> np.ndarray:
    """F81 generator Q[θ→θ'] = ρ_θ' for θ != θ'; diag chosen so rows sum
    to 0. Reversible w.r.t. rho.

    rate normalisation: mean substitution rate at rho is
    -sum_θ rho_θ Q[θ,θ] = sum_θ rho_θ (1 - rho_θ). We do NOT renormalise
    here -- the test fixtures use rho directly, and the math derivations
    pick a consistent convention.
    """
    L = rho.shape[0]
    Q = np.tile(rho[None, :], (L, 1))
    np.fill_diagonal(Q, 0.0)
    Q -= np.diag(Q.sum(axis=1))
    return Q


def f81_transition_closed(rho: np.ndarray, t: float) -> np.ndarray:
    """Closed-form F81 transition matrix on the (L,) state space.

    Q = 1 rho^T - I has eigenvalues {0, -1, -1, ...} (one zero eigenvalue
    with eigenvector 1; the rest are -1 in the rank-(L-1) subspace
    orthogonal to ρ). Hence
        exp(Q t) = (1 ρ^T) + (I - 1 ρ^T) exp(-t)
    or equivalently
        P[θ→θ'; t] = rho_θ' + (δ_{θθ'} - rho_θ') exp(-t)
    -- decay is uniformly rate 1, independent of the source state.
    """
    L = rho.shape[0]
    decay = np.exp(-t)                      # scalar -- uniform across sources
    P = rho[None, :] + (np.eye(L) - rho[None, :]) * decay
    return P


def per_class_field_pi(K_c: int, L: int, A: int = 20, seed: int = 0
                         ) -> np.ndarray:
    """Random per-(class, field) stationary tensor pi^(c, theta).
    Shape (K_c, L, A). Each row is a Dirichlet draw, intentionally
    non-uniform so per-(c, θ) tilting is observable."""
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.ones(A) * 0.7, size=(K_c, L))    # peaky enough
    return pi


def per_class_field_Q(pi_class_field: np.ndarray, S: np.ndarray
                       ) -> np.ndarray:
    """GTR(S, pi) per (c, theta). Shape (K_c, L, A, A)."""
    K_c, L, A = pi_class_field.shape
    Q = np.zeros((K_c, L, A, A))
    for c in range(K_c):
        for th in range(L):
            pi = pi_class_field[c, th]
            S_off = S - np.diag(np.diag(S))
            Qc = S_off * pi[None, :]
            np.fill_diagonal(Qc, -Qc.sum(axis=1))
            Q[c, th] = Qc
    return Q


def lg08_like_S(A: int = 20, seed: int = 0) -> np.ndarray:
    """Symmetric exchangeability matrix with positive off-diagonal,
    structure mimicking LG08 just to get something non-trivial. Not
    actually LG08 -- we want the test self-contained."""
    rng = np.random.default_rng(seed)
    S = rng.gamma(2.0, 1.0, size=(A, A))
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)
    return S


# ---------------------------------------------------------------------------
# Test 1: F81-on-DP detailed balance.
# ---------------------------------------------------------------------------

def test_f81_dp_detailed_balance():
    print("\n[test 1] F81-on-DP detailed balance")
    rho = make_field_rho(L=6, alpha=1.5, seed=11)
    # Verify closed-form P agrees with expm(Q t) first.
    Q = f81_generator(rho)
    for t in [0.1, 0.5, 1.0, 3.0]:
        P_closed = f81_transition_closed(rho, t)
        P_expm = expm(Q * t)
        err = float(np.max(np.abs(P_closed - P_expm)))
        assert err < 1e-10, (
            f"closed-form vs expm at t={t}: max err={err:.3e}")
    # Detailed balance: rho_θ P(θ→θ'; t) == rho_θ' P(θ'→θ; t).
    max_db = 0.0
    for t in [0.1, 0.5, 1.0, 3.0]:
        P = f81_transition_closed(rho, t)
        lhs = rho[:, None] * P                   # (L, L)
        rhs = rho[None, :] * P.T
        max_db = max(max_db, float(np.max(np.abs(lhs - rhs))))
    print(f"  closed/expm agree; detailed-balance residual = {max_db:.3e}")
    assert max_db < 1e-12, f"detailed balance failed: max residual = {max_db}"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2: Lumping consistency on occupied + tail.
# ---------------------------------------------------------------------------

def _lumped_chain(rho: np.ndarray, occupied: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
    """Return (rho_lumped, Q_lumped) on the |O|+1-state lumped space.
    occupied: boolean (L,) mask of occupied atoms; the tail is the rest."""
    L = rho.shape[0]
    O = np.where(occupied)[0]
    tail_mass = float(rho[~occupied].sum())
    rho_lumped = np.concatenate([rho[O], [tail_mass]])
    # In the strong-lumpability limit (tail rows close to identical), the
    # lumped chain is F81 with stationary rho_lumped. Build it via the
    # same F81 form.
    Q_l = f81_generator(rho_lumped)
    return rho_lumped, Q_l


def test_lumping_consistency():
    """Aggregate the full-chain transition into the lumped state space and
    compare against the closed-form lumped F81 chain. The discrepancy
    must be O(max tail-atom mass)."""
    print("\n[test 2] Lumping consistency on occupied + tail")
    rho = make_field_rho(L=10, alpha=0.5, seed=22)
    # Mark the top-K-by-mass atoms as 'occupied'.
    for K in [3, 6]:
        order = np.argsort(-rho)
        occupied = np.zeros_like(rho, dtype=bool)
        occupied[order[:K]] = True
        max_tail = float(rho[~occupied].max()) if not occupied.all() else 0.0
        rho_l, Q_l = _lumped_chain(rho, occupied)
        for t in [0.5, 2.0]:
            # Full chain transition aggregated by the lumping.
            P_full = f81_transition_closed(rho, t)
            # Lumped state index: occupied atoms 0..K-1, tail at index K.
            L = rho.shape[0]
            P_agg = np.zeros((K + 1, K + 1))
            O = np.where(occupied)[0]
            for a, oa in enumerate(O):
                for b, ob in enumerate(O):
                    P_agg[a, b] = P_full[oa, ob]
                P_agg[a, K] = P_full[oa, ~occupied].sum()
            # Strong-lumpability: each tail source SHOULD aggregate the
            # same way. Take a uniform-over-tail average as the lumped
            # representative.
            tail_idx = np.where(~occupied)[0]
            for b, ob in enumerate(O):
                P_agg[K, b] = (rho[tail_idx] * P_full[tail_idx, ob]).sum() \
                                / max(rho[tail_idx].sum(), 1e-300)
            P_agg[K, K] = (rho[tail_idx][:, None] *
                            P_full[tail_idx][:, ~occupied]).sum() \
                              / max(rho[tail_idx].sum(), 1e-300)
            # Closed-form lumped chain.
            P_lumped = f81_transition_closed(rho_l, t)
            diff = float(np.max(np.abs(P_agg - P_lumped)))
            # Expected order of error: O(max_tail). Allow 5x slack.
            ok = diff < 5.0 * max(max_tail, 1e-12)
            print(f"  K={K}  t={t:.1f}  max_tail={max_tail:.3e}  "
                  f"lumped-vs-aggregated diff={diff:.3e}  {'OK' if ok else 'FAIL'}")
            assert ok, (
                f"lumping residual {diff:.3e} > 5x max_tail {max_tail:.3e}; "
                f"K={K}, t={t}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: Cap-2 joint at L=1 vs brute Kronecker.
# ---------------------------------------------------------------------------

def test_kronecker_sum_identity_for_independent_sites():
    """At L=1 the field is degenerate (no jumps possible). Both sites of
    the cluster see the same field theta=0 throughout. Under instant
    re-equilibration the leaf residue distribution is the per-(c, theta)
    stationary at the leaf field; at L=1 that's just pi^(c, 0). This
    test is therefore the L=1 reduction: the joint generator on
    (A, A) factors as Kronecker-sum and its expm factors as Kronecker
    product, which is a textbook identity. Verifying it cross-checks
    the per-(c, theta) generator construction itself.
    """
    print("\n[test 3] Kronecker-sum identity for L=1 (per-site CTMC reduction)")
    A = 4
    L = 1
    K_c = 2
    pi_cf = per_class_field_pi(K_c, L, A=A, seed=33)
    S = lg08_like_S(A=A, seed=33)
    Qcf = per_class_field_Q(pi_cf, S)
    max_err = 0.0
    for c_i in range(K_c):
        for c_j in range(K_c):
            for t in [0.2, 0.8]:
                Q_i = Qcf[c_i, 0]; Q_j = Qcf[c_j, 0]
                Q_pair = (np.kron(Q_i, np.eye(A))
                          + np.kron(np.eye(A), Q_j))
                P_pair_brute = expm(Q_pair * t)
                P_pair_kron = np.kron(expm(Q_i * t), expm(Q_j * t))
                err = float(np.max(np.abs(P_pair_brute - P_pair_kron)))
                max_err = max(max_err, err)
    print(f"  max |expm(Q⊕Q) - expm(Q)⊗expm(Q)| = {max_err:.3e}")
    assert max_err < 1e-10
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 4: Cap-2 joint via L^3 sum vs explicit jump-trajectory integral.
# ---------------------------------------------------------------------------

def _gillespie_field_trajectory(theta_0, rho, rho_chain, t, rng):
    """Gillespie-simulate the F81-on-DP field chain from theta_0 over
    time t. Returns (theta_end, n_jumps)."""
    theta_now = int(theta_0)
    s = 0.0
    n_jumps = 0
    L = rho.shape[0]
    while True:
        rate_out = float(rho_chain) * float(1.0 - rho[theta_now])
        if rate_out <= 0.0:
            break
        dt = rng.exponential(1.0 / rate_out)
        if s + dt > t:
            break
        s += dt
        p_targets = rho.copy()
        p_targets[theta_now] = 0.0
        denom = p_targets.sum()
        if denom <= 0.0:
            break
        p_targets /= denom
        theta_now = int(rng.choice(L, p=p_targets))
        n_jumps += 1
    return theta_now, n_jumps


def test_cap2_4case_closed_form_matches_faithful_trajectory_MC():
    """The 4-case Interp 2 closed form must agree with a Gillespie
    trajectory MC. Under Interp 2 the per-half-edge no-jump probability
    is STATE-DEPENDENT: beta(theta_P) = exp(-rho_chain * (1 - rho[theta_P])
    * t / 2). Under no jump the leaf residue is parent-evolved under
    Q^(c, theta_P); under >= 1 jump it's drawn from pi^(c, theta_end)
    where theta_end is the field at the end of the half-edge.
    """
    print("\n[test 4] Cap-2 Interp 2 4-case closed form vs Gillespie MC")
    A = 3
    L = 3
    K_c = 2
    rho_chain = 1.0
    pi_cf = per_class_field_pi(K_c, L, A=A, seed=44)
    rho = make_field_rho(L=L, alpha=0.8, seed=44)
    S = lg08_like_S(A=A, seed=44)
    Qcf = per_class_field_Q(pi_cf, S)
    t = 0.6
    c_i, c_j = 0, 1

    # Per-(c, θ) half-edge transition matrices.
    P_half = np.zeros((K_c, L, A, A))
    for c in range(K_c):
        for th in range(L):
            P_half[c, th] = expm(Qcf[c, th] * (t / 2.0))

    Sigma = np.zeros((K_c, L, A, A))
    for c in range(K_c):
        for th in range(L):
            pi_p = pi_cf[c, th]
            Sigma[c, th] = np.einsum(
                'p,pa,pb->ab', pi_p, P_half[c, th], P_half[c, th])

    # Interp 2 weights.
    s = t / 2.0
    alpha = float(np.exp(-rho_chain * s))
    beta = np.exp(-rho_chain * (1.0 - rho) * s)             # (L,)
    denom = 1.0 - beta
    safe = denom > 1e-15
    w_J = np.where(safe, (1.0 - alpha) / np.where(safe, denom, 1.0), 1.0)
    w_pi = np.where(safe, (alpha - beta) / np.where(safe, denom, 1.0), 0.0)

    # Field-marginal joint J(c_i, c_j)(a, b).
    J = np.einsum('t,ta,tb->ab', rho, pi_cf[c_i], pi_cf[c_j])           # (A, A)
    # Per-theta_P J': J'[theta_P, a, b] = w_J[theta_P] * J + w_pi[theta_P] * pi_ci(theta_P) (x) pi_cj(theta_P).
    J_prime = (w_J[:, None, None] * J[None, :, :]
                + w_pi[:, None, None]
                  * pi_cf[c_i, :, :, None] * pi_cf[c_j, :, None, :])     # (L, A, A)

    # 4-case closed form, conditional on theta_P, summed over theta_P.
    closed = np.zeros((A, A, A, A))         # (Xi, Xj, Yi, Yj)
    for th_P in range(L):
        b = float(beta[th_P]); one_m_b = 1.0 - b
        nn_nn = np.einsum('xy,uv->xuyv', Sigma[c_i, th_P], Sigma[c_j, th_P])
        nn_yj = np.einsum(
            'x,u,yv->xuyv', pi_cf[c_i, th_P], pi_cf[c_j, th_P], J_prime[th_P])
        yj_nn = np.einsum(
            'xu,y,v->xuyv', J_prime[th_P], pi_cf[c_i, th_P], pi_cf[c_j, th_P])
        yj_yj = np.einsum('xu,yv->xuyv', J_prime[th_P], J_prime[th_P])
        term = (b * b * nn_nn
                + b * one_m_b * nn_yj
                + one_m_b * b * yj_nn
                + one_m_b * one_m_b * yj_yj)
        closed += rho[th_P] * term

    # Faithful Gillespie MC.
    rng = np.random.default_rng(45)
    n_trials = 500000
    counts = np.zeros((A, A, A, A))
    for _ in range(n_trials):
        th_P = int(rng.choice(L, p=rho))
        P_i = int(rng.choice(A, p=pi_cf[c_i, th_P]))
        P_j = int(rng.choice(A, p=pi_cf[c_j, th_P]))
        # X edge Gillespie.
        th_X_end, nj_X = _gillespie_field_trajectory(
            th_P, rho, rho_chain, s, rng)
        if nj_X == 0:
            Xi = int(rng.choice(A, p=P_half[c_i, th_P, P_i]))
            Xj = int(rng.choice(A, p=P_half[c_j, th_P, P_j]))
        else:
            Xi = int(rng.choice(A, p=pi_cf[c_i, th_X_end]))
            Xj = int(rng.choice(A, p=pi_cf[c_j, th_X_end]))
        # Y edge Gillespie.
        th_Y_end, nj_Y = _gillespie_field_trajectory(
            th_P, rho, rho_chain, s, rng)
        if nj_Y == 0:
            Yi = int(rng.choice(A, p=P_half[c_i, th_P, P_i]))
            Yj = int(rng.choice(A, p=P_half[c_j, th_P, P_j]))
        else:
            Yi = int(rng.choice(A, p=pi_cf[c_i, th_Y_end]))
            Yj = int(rng.choice(A, p=pi_cf[c_j, th_Y_end]))
        counts[Xi, Xj, Yi, Yj] += 1
    mc = counts / n_trials
    diff = float(np.max(np.abs(mc - closed)))
    se = 1.0 / np.sqrt(n_trials)
    print(f"  Interp 2 closed vs Gillespie MC ({n_trials} trials): "
          f"max |diff|={diff:.4f}  ~SE_MC={se:.4f}")
    assert diff < 6 * se, (
        f"Interp 2 closed form and Gillespie MC disagree: {diff:.4f} "
        f"vs ~SE {se:.4f}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 5: Field-marginal joint => marginal consistency => no Sinkhorn.
# ---------------------------------------------------------------------------

def test_field_marginal_joint_is_marginal_consistent():
    """sum_c pi_joint(a, c) = pi_lone(a) by construction (de Finetti)."""
    print("\n[test 5] Dynamic-field joint stationary is marginal-consistent")
    A = 5
    L = 4
    K_c = 3
    pi_cf = per_class_field_pi(K_c, L, A=A, seed=55)
    rho = make_field_rho(L=L, alpha=1.5, seed=55)
    # Use empirical class prior pi_c proportional to a random Dirichlet.
    rng = np.random.default_rng(55)
    pi_c = rng.dirichlet(np.ones(K_c) * 1.5)

    # Lone single-site stationary at class c:
    #   pi^(c)(a) = sum_θ rho_θ pi^(c, θ)(a)
    pi_lone_per_c = (rho[None, :, None] * pi_cf).sum(axis=1)   # (K_c, A)

    # Cluster-of-2 joint marginal at (c1, c2):
    #   pi_joint(a, b; c1, c2) = sum_θ rho_θ pi^(c1, θ)(a) pi^(c2, θ)(b)
    # Class-marginalized over (c1, c2) with prior pi_c:
    #   pi_joint(a, b) = sum_{c1, c2} pi_c(c1) pi_c(c2)
    #                       sum_θ rho_θ pi^(c1, θ)(a) pi^(c2, θ)(b)
    joint = np.zeros((A, A))
    for c1 in range(K_c):
        for c2 in range(K_c):
            for th in range(L):
                joint += (pi_c[c1] * pi_c[c2] * rho[th]
                            * np.outer(pi_cf[c1, th], pi_cf[c2, th]))

    # Lone single-site marginal stationary class-marginalized:
    pi_lone = (pi_c[:, None] * pi_lone_per_c).sum(axis=0)   # (A,)
    pi_lone /= pi_lone.sum()
    # Marginal of joint matches lone:
    row_marg = joint.sum(axis=1)
    col_marg = joint.sum(axis=0)
    row_err = float(np.max(np.abs(row_marg - pi_lone)))
    col_err = float(np.max(np.abs(col_marg - pi_lone)))
    print(f"  row-marg err = {row_err:.3e}, col-marg err = {col_err:.3e}")
    assert row_err < 1e-12, f"row marginal != lone stationary: {row_err}"
    assert col_err < 1e-12, f"col marginal != lone stationary: {col_err}"
    print("  ==> no Sinkhorn correction needed under the dynamic-field "
          "variant. PASS")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import inspect, sys
    mod = sys.modules[__name__]
    fns = [(name, fn) for name, fn in inspect.getmembers(mod, inspect.isfunction)
           if name.startswith("test_")]
    failures = []
    for name, fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"\n  FAIL  {name}: {e}")
            failures.append(name)
        except Exception as e:
            import traceback
            print(f"\n  ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(name)
    if failures:
        print(f"\n{len(failures)}/{len(fns)} test(s) failed: {failures}")
        sys.exit(1)
    print(f"\nAll {len(fns)} math precompute tests passed.")
