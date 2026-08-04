"""Moment-matching CLV (conditional likelihood vector) primitives, ported
faithfully from evolmoves `ts/cluster/dynfield-emission.ts` (functions
`mmEdge`, `mmCombine`, `mmMassOne`, `mmMassTwo`, `mmRescale`, `leafClv`).

The CLV is represented as

    CLV(x, theta) = rho1[theta] * prod_n A[n, theta, x_n]
                    + s[theta]

i.e. a rank-1 (tracking) term plus a CONSTANT-in-x scalar (the regenerated
component: when the field jumped on the edge below, the residue was resampled
so the subtree likelihood no longer depends on x). The scalar is s[theta], NOT
s[theta] * prod_n pi_field -- the mass formulas rely on <pi, 1> = 1 for the
scalar term (see mm_mass_one / _joint_marginal_FxD), and treating it as a
prod-pi term reintroduces the >6% mass leak fixed 2026-07-13.

with the normalisation convention that
`< pi_field[c_n, theta, :], A[n, theta, :] > == 1` per site per theta
(preserved through propagation and internal-node combining). This
convention makes the mass formulas clean:

    sum_{x} CLV(x, theta) * pi_field(c_n, theta)(x_n) evaluated against
    the joint stationary rho * prod pi_field  == the ROOT log-mass

which is exactly what mmMassTwo / mmMassOne compute.

Structure of the two-term CLV:
  - `rho1[theta]` (L,)               magnitude of the rank-1 (tracking)
                                     component per field state
  - `A[n, theta, a]` (m, L, A)       per-site per-theta simplex point
                                     normalised as < pi, A > = 1
  - `s[theta]` (L,)                  magnitude of the scalar
                                     (regenerated) component per field
                                     state
  - `log_scale` (float)              Felsenstein rescaling factor
                                     accumulated over the tree

Kernel conventions (Interp 2 F81-on-DP field, GTR-per-arch substitution):
  - Field transition: P_F(theta_v -> theta_u; tau)
                    = exp(-rho_chain*tau) * delta(theta_v == theta_u)
                    + (1 - exp(-rho_chain*tau)) * rho[theta_u]
  - No-jump probability: beta(theta_v, tau)
                    = exp(-rho_chain * (1 - rho[theta_v]) * tau)
  - Jump weight: W(theta_v, theta_u; tau)
                    = P_F(theta_v -> theta_u; tau)
                    - delta(theta_v == theta_u) * beta(theta_v, tau)

At depth-1 cherries and in the rho_chain -> 0, infty limits the
projection is exact; deeper trees pay a controlled bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.linalg import expm


@dataclass
class MMClv:
    """Moment-matching CLV. See module docstring for the representation."""
    rho1: np.ndarray                     # (L,) rank-1 magnitudes
    A: np.ndarray                        # (m, L, A) normalised marginals
    s: np.ndarray                        # (L,) scalar magnitudes
    log_scale: float                     # rescaling factor

    def L(self) -> int:
        return int(self.rho1.shape[0])

    def m(self) -> int:
        return int(self.A.shape[0])

    def A_alphabet(self) -> int:
        return int(self.A.shape[2])

    def check_normalisation(self, pi_field: np.ndarray,
                                classes: np.ndarray,
                                atol: float = 1e-8) -> None:
        """Assert that < pi_field[c_n, theta, :], A[n, theta, :] > = 1
        for all (n, theta). pi_field shape: (K_c, L, A). classes: (m,).
        """
        m = self.m()
        for n in range(m):
            c = int(classes[n])
            for th in range(self.L()):
                dot = float(np.dot(pi_field[c, th], self.A[n, th]))
                if not np.isfinite(dot) or abs(dot - 1.0) > atol:
                    raise AssertionError(
                        f"< pi, A[{n}, {th}] > = {dot} != 1 (site {n}, "
                        f"class {c}, theta {th})")


# ---------------------------------------------------------------------------
# Field transition + no-jump probability + jump weights.
# ---------------------------------------------------------------------------


def field_transition_row(rho: np.ndarray, th_from: int, t: float,
                             rho_chain: float) -> np.ndarray:
    """P_F(theta_v = th_from -> theta_u; t) for all theta_u; shape (L,).

    F81-on-DP: exp(-rho_chain*t) * delta + (1 - exp(-rho_chain*t)) * rho.
    """
    et = float(np.exp(-rho_chain * t))
    out = (1.0 - et) * rho.astype(np.float64).copy()
    out[th_from] += et
    return out


def beta_no_jump(rho: np.ndarray, th_v: int, t: float,
                    rho_chain: float) -> float:
    """State-dependent no-jump probability beta(theta_v, t)."""
    return float(np.exp(-rho_chain * (1.0 - rho[th_v]) * t))


# ---------------------------------------------------------------------------
# Per-class per-field substitution kernel exp(Q * t) helpers.
# ---------------------------------------------------------------------------


def per_class_field_Q(pi_field: np.ndarray, S: np.ndarray) -> np.ndarray:
    """GTR rate matrix Q(c, theta) per (site class, field state).

    pi_field: (K_c, L, A). S: (A, A) exchangeability (symmetric, zero
    diagonal). Returns Q of shape (K_c, L, A, A) with row sums zero.
    """
    K_c, L, A = pi_field.shape
    assert S.shape == (A, A)
    # Off-diagonal: Q[c, th, a, b] = S[a, b] * pi[c, th, b] for a != b
    Q = S[None, None, :, :] * pi_field[:, :, None, :]     # (K_c, L, A, A)
    # Zero the diagonal, then fill with negative row sum.
    for a in range(A):
        Q[:, :, a, a] = 0.0
    row_sum = Q.sum(axis=-1)                               # (K_c, L, A)
    for a in range(A):
        Q[:, :, a, a] = -row_sum[:, :, a]
    return Q


def branch_P(Q: np.ndarray, c: int, th: int, t: float,
               eta: float = 1.0) -> np.ndarray:
    """Per-class per-field transition matrix P = exp(Q(c, th) * t * eta).
    (A, A). Uses scipy.linalg.expm.
    """
    return expm(Q[int(c), int(th)] * float(t) * float(eta))


# ---------------------------------------------------------------------------
# Leaf CLV.
# ---------------------------------------------------------------------------


def leaf_clv(residues: np.ndarray,
                classes: np.ndarray,
                pi_field: np.ndarray) -> MMClv:
    """Construct the CLV at an observed leaf.

    Args:
      residues: (m,) int; observed residue at each cluster column.
        -1 marks gapped (uninformative); at gaps A is uniform-ish and
        the leaf carries no information.
      classes: (m,) int; per-column site-class assignment.
      pi_field: (K_c, L, A) archetype-materialised stationary.

    Returns MMClv with s = 0, rho1[theta] = prod_n pi_field[c_n, theta,
    x_obs_n], A[n, theta, x_obs_n] = 1 / pi_field[c_n, theta, x_obs_n]
    (delta at observed / normalised). Gapped sites carry
    rho1_contribution = 1 and A[n, theta, :] uniform (not the strict
    delta form; details in the discussion below).
    """
    m = int(residues.shape[0])
    K_c, L, A = pi_field.shape
    rho1 = np.ones(L, dtype=np.float64)
    Aa = np.zeros((m, L, A), dtype=np.float64)
    s = np.zeros(L, dtype=np.float64)
    for si in range(m):
        c = int(classes[si])
        x = int(residues[si])
        if x >= 0:
            # Delta-observed site: rho1 *= pi(x_obs); A = delta(x_obs)/pi.
            for th in range(L):
                px = float(pi_field[c, th, x])
                Aa[si, th, x] = 1.0 / max(px, 1e-300)
                rho1[th] *= px
        else:
            # Gapped site: no information from this position.
            # Convention: A[n, theta, :] = 1_A (constant 1 vector)
            # → < pi, A > = < pi, 1 > = 1 (since pi is a simplex).
            # rho1 contribution from this site is 1 (no π-factor).
            Aa[si, :, :] = 1.0
    return MMClv(rho1=rho1, A=Aa, s=s, log_scale=0.0)


# ---------------------------------------------------------------------------
# Per-edge propagation.
# ---------------------------------------------------------------------------


def mm_edge_backward(parent_aggr: MMClv,
                        classes: np.ndarray,
                        t: float,
                        rho: np.ndarray,
                        pi_field: np.ndarray,
                        Q: np.ndarray,
                        rho_chain: float,
                        eta: float = 1.0) -> MMClv:
    """Backward propagation from parent-side aggregate M_v to child-side D_v.

    Direct implementation of the recursion derived in
    par:arch-phylo-elbo-backward (appendix-tkfdp.tex, esp. eq:arch-elbo-sD
    and the transposed-P update for A_D):

      r_D(theta_v) = beta(theta_v, t) * r_M(theta_v)
      A_D[n, theta_v, x] = sum_y P^(c_n, theta_v)(y, x; t) * A_M[n, theta_v, y]
      s_D(theta_v) = beta(theta_v, t) * s_M(theta_v)
                    + sum_{theta_p} W(theta_p, theta_v; t) * marg_M(theta_p)

    where marg_M(theta_p) = r_M + s_M (evolmoves convention; approximate
    for < pi, pi > > 1 per site but exact when the leaf-side layer of
    the tree has s = 0). For this MVP we follow the same convention as
    the forward mm_edge; a more principled version would carry
    the prod ||pi||_2^2 factor and is a scoped follow-on.

    The transposed substitution direction (P(y, x) rather than P(x, y))
    is the mathematical distinction from mm_edge (forward). Under GTR
    reversibility this differs from the forward substitution
    propagation by
        A_D = D_pi^{-1} A_M^{forward-shape} D_pi
    but the numerics are distinct.
    """
    L = int(rho.shape[0])
    A_alph = int(pi_field.shape[2])
    m = int(classes.shape[0])

    rho1 = np.zeros(L, dtype=np.float64)
    s = np.zeros(L, dtype=np.float64)
    Aa = np.zeros((m, L, A_alph), dtype=np.float64)

    # marg_parent(theta_p) = sum_{x_p} M_v(x_p, theta_p): the L1 sum over
    # x_p of the parent-aggregated message per theta_p, WITHOUT the
    # per-site pi weighting.
    #
    # This is DIFFERENT from the forward marg_child, which is the pi-
    # WEIGHTED integral sum_{x_c} prod pi(x_c, theta_c) * F_c(x_c, theta_c)
    # because in the forward the jump-emission K(x_c, theta_c | x_v, theta_v)
    # emits pi(x_c) at the child, which meets the child's CLV in an inner
    # product against pi. In the backward, the jump-emission K(x_v, theta_v
    # | x_p, theta_p) emits pi(x_v) at the child (v), NOT at the parent
    # (p) that we are summing over, so no pi factor at the parent side.
    #
    # Formula: marg_parent(theta_p) = r_M * prod_n <A_M[n, theta_p, .]>_1
    #                               + s_M * 1
    # where <.>_1 is the plain L1 sum (not pi-weighted).
    #
    # At a leaf sibling propagated up (A = delta(x_obs)/pi and s = 0),
    # <A>_1 = 1/pi(x_obs), so r_M * prod <A>_1 = pi(x_obs) * prod(1/pi) = 1.
    # This matches the correct sum_{x_leaf} CLV_leaf(x, theta) = 1 (leaf
    # CLV is a delta over residues).
    marg_parent = np.zeros(L, dtype=np.float64)
    for th_p in range(L):
        prod_A_L1 = 1.0
        for si in range(m):
            prod_A_L1 *= float(np.sum(parent_aggr.A[si, th_p, :]))
        marg_parent[th_p] = (parent_aggr.rho1[th_p] * prod_A_L1
                                    + parent_aggr.s[th_p])

    for th_v in range(L):
        b = beta_no_jump(rho, th_v, t, rho_chain)
        rho1[th_v] = b * parent_aggr.rho1[th_v]
        # Per-site: A_D[x] = P @ A_M[x] (forward substitution direction).
        # Empirically this gives exact ρ_chain=0 posteriors at all
        # depths under the joint F * D * pi integrator. The alternative
        # P^T destroys ρ_chain=0 correctness. Deeper theoretical
        # justification via the reversibility identity remains to be
        # written; see par:arch-phylo-elbo-backward.
        for si in range(m):
            c_n = int(classes[si])
            P = branch_P(Q, c_n, th_v, t, eta=eta)
            Aa[si, th_v, :] = P @ parent_aggr.A[si, th_v, :]
        # Jump contribution to s: W(theta_p, theta_v; t) with theta_p as
        # SOURCE (parent as source of the forward field jump).
        jump = 0.0
        for th_p in range(L):
            pf_from_th_p = field_transition_row(rho, th_p, t, rho_chain)
            w = pf_from_th_p[th_v] - (b if th_p == th_v else 0.0)
            jump += w * marg_parent[th_p]
        s[th_v] = b * parent_aggr.s[th_v] + jump

    return MMClv(rho1=rho1, A=Aa, s=s, log_scale=parent_aggr.log_scale)


def mm_edge(child: MMClv,
              classes: np.ndarray,
              t: float,
              rho: np.ndarray,
              pi_field: np.ndarray,
              Q: np.ndarray,
              rho_chain: float,
              eta: float = 1.0) -> MMClv:
    """Propagate a child CLV to a parent-side edge message through a
    branch of length t. Returns a new MMClv in the (rho1 + s) form.

    Ported from evolmoves ts/cluster/dynfield-emission.ts::mmEdge.

    marg_child(theta_u) = rho1_u + s_u                (each A norm)
    beta(theta_v) = exp(-rho_chain * (1 - rho[theta_v]) * t)
    rho1_v[theta_v] = beta(theta_v) * rho1_u[theta_v]
    A_v[n, theta_v, :] = P^(c_n, theta_v)(:, ·) @ A_u[n, theta_v, :]
    s_v[theta_v] = beta(theta_v) * s_u[theta_v]
                 + sum_{theta_u} W(theta_v, theta_u) * marg_child(theta_u)
    """
    L = int(rho.shape[0])
    A = int(pi_field.shape[2])
    m = int(classes.shape[0])

    rho1 = np.zeros(L, dtype=np.float64)
    s = np.zeros(L, dtype=np.float64)
    Aa = np.zeros((m, L, A), dtype=np.float64)

    # marg_child(theta_u) = <pi_all, F_c(., theta_u)>
    #                     = r_c * prod_n <pi, A_c[n]> + s_c * prod_n <pi, 1>
    # Under the (r * prod A + s * 1) family form with invariants
    # <pi, A> = 1 and <pi, 1> = 1, both products equal 1, so
    # marg_child(theta_u) = r_c + s_c.
    #
    # Previous versions of this code multiplied s_c by prod_n ||pi_n||_2^2
    # -- a bug that assumed the scalar component was prod_n pi(x_n) rather
    # than the constant 1. That gave a substantial mass leak once s > 0
    # (measured >6% total-theta-mass error at depth-2, versus documented
    # "~5e-3 leaf-posterior bias" that was actually a lower bound). Fixed
    # 2026-07-13; see paper par:arch-phylo-elbo-mmedge for the correct
    # closed-form propagation under this family.
    marg_child = child.rho1 + child.s  # (L,)

    for th_v in range(L):
        b = beta_no_jump(rho, th_v, t, rho_chain)
        rho1[th_v] = b * child.rho1[th_v]
        # A propagation per site: (P @ child.A[n, th_v, :])
        for si in range(m):
            P = branch_P(Q, int(classes[si]), th_v, t, eta=eta)
            Aa[si, th_v, :] = P @ child.A[si, th_v, :]
        # Jump contribution to s.
        pf_row = field_transition_row(rho, th_v, t, rho_chain)
        jump = 0.0
        for th_u in range(L):
            w = pf_row[th_u] - (b if th_u == th_v else 0.0)
            jump += w * marg_child[th_u]
        s[th_v] = b * child.s[th_v] + jump

    return MMClv(rho1=rho1, A=Aa, s=s, log_scale=child.log_scale)


# ---------------------------------------------------------------------------
# Internal-node combining via moment-matching projection.
# ---------------------------------------------------------------------------


def mm_combine(X: MMClv, Y: MMClv,
                 classes: np.ndarray,
                 pi_field: np.ndarray) -> MMClv:
    """Combine two edge-messages at an internal (non-root) node.

    Ported from evolmoves ts/cluster/dynfield-emission.ts::mmCombine.

    Exact product (per theta):
      M[theta] = rA rB * prodDot(theta)
               + rA * sB + sA * rB + sA * sB
    where prodDot(theta) = prod_n < pi(c_n, theta), X.A[n, th] * Y.A[n, th] >.

    Projection:
      s_v[theta] = sA * sB                     (scalar x scalar stays scalar)
      rho1_v[theta] = M[theta] - s_v[theta]
      A_v[n, theta, a] = (m_a - s_v) / rho1_v
      where m_a = (per-site local marginal fixing x_n = a and integrating
      over other sites' x_{n'} under < pi, . >):
        m_a = rA rB * restDot(theta, n) * X.A[n, th, a] * Y.A[n, th, a]
            + rA * sB * X.A[n, th, a]
            + sA * rB * Y.A[n, th, a]
            + sA * sB
      with restDot(theta, n) = prod_{n' != n} dotAB(theta, n').

    Preserves the invariant < pi, A_v[n, th, :] > = 1 exactly.
    """
    L = int(pi_field.shape[1])
    A = int(pi_field.shape[2])
    m = int(classes.shape[0])
    assert X.m() == Y.m() == m
    assert X.L() == Y.L() == L

    rho1 = np.zeros(L, dtype=np.float64)
    s = np.zeros(L, dtype=np.float64)
    Aa = np.zeros((m, L, A), dtype=np.float64)

    for th in range(L):
        dot_AB = np.zeros(m, dtype=np.float64)
        prod_dot = 1.0
        for si in range(m):
            pi_row = pi_field[int(classes[si]), th]       # (A,)
            d = float(np.sum(pi_row * X.A[si, th] * Y.A[si, th]))
            dot_AB[si] = d
            prod_dot *= d
        rA = X.rho1[th]; rB = Y.rho1[th]
        sA = X.s[th]; sB = Y.s[th]
        M = rA * rB * prod_dot + rA * sB + sA * rB + sA * sB
        sv = sA * sB
        r1 = M - sv
        s[th] = sv
        rho1[th] = r1
        for si in range(m):
            rest_dot = prod_dot / dot_AB[si] if dot_AB[si] > 0 else 0.0
            m_a = (rA * rB * rest_dot * (X.A[si, th] * Y.A[si, th])
                     + rA * sB * X.A[si, th]
                     + sA * rB * Y.A[si, th]
                     + sA * sB)                             # (A,)
            if r1 > 0:
                Aa[si, th] = (m_a - sv) / r1
            else:
                Aa[si, th] = 0.0

    return MMClv(rho1=rho1, A=Aa, s=s,
                   log_scale=X.log_scale + Y.log_scale)


# ---------------------------------------------------------------------------
# Rescaling.
# ---------------------------------------------------------------------------


def mm_rescale(clv: MMClv) -> MMClv:
    """Factor max(rho1, s) into log_scale; A_n stay normalised."""
    mx = 0.0
    for th in range(clv.L()):
        mx = max(mx, clv.rho1[th], clv.s[th])
    if not (mx > 0.0) or not np.isfinite(mx):
        return clv
    return MMClv(rho1=clv.rho1 / mx, A=clv.A, s=clv.s / mx,
                   log_scale=clv.log_scale + float(np.log(mx)))


# ---------------------------------------------------------------------------
# Root-mass integration (exact; no projection at the root).
# ---------------------------------------------------------------------------


def mm_mass_two(X: MMClv, Y: MMClv,
                  classes: np.ndarray,
                  rho: np.ndarray,
                  pi_field: np.ndarray) -> 'tuple[float, float]':
    """Exact mass of the product of two edge-messages against rho * prod
    pi_field at the root.

    Returns (mass, log_scale).
    """
    L = int(rho.shape[0])
    m = int(classes.shape[0])
    total = 0.0
    for th in range(L):
        prod_dot = 1.0
        for si in range(m):
            pi_row = pi_field[int(classes[si]), th]
            d = float(np.sum(pi_row * X.A[si, th] * Y.A[si, th]))
            prod_dot *= d
        rA = X.rho1[th]; rB = Y.rho1[th]
        sA = X.s[th]; sB = Y.s[th]
        total += float(rho[th]) * (
            rA * rB * prod_dot + rA * sB + sA * rB + sA * sB)
    return total, X.log_scale + Y.log_scale


def mm_mass_one(X: MMClv,
                  classes: np.ndarray,
                  rho: np.ndarray,
                  pi_field: np.ndarray) -> 'tuple[float, float]':
    """Exact mass of a single edge-message against rho * prod pi_field
    (used at a degenerate single-child root).
    """
    L = int(rho.shape[0])
    m = int(classes.shape[0])
    total = 0.0
    for th in range(L):
        prod_ = 1.0
        for si in range(m):
            pi_row = pi_field[int(classes[si]), th]
            d = float(np.sum(pi_row * X.A[si, th]))
            prod_ *= d
        total += float(rho[th]) * (X.rho1[th] * prod_ + X.s[th])
    return total, X.log_scale
