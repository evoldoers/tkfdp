"""Per-(c, theta) generators and 4-case cluster joint emission for the
dynamic-latent-field coupling variant (Interp 2: textbook F81-on-DP CTMC).

See docs/dynfield_math.md for the math. In summary:

The cap-2 cluster joint emission at a cherry of diameter t under the
F81-on-DP CTMC with rate `rho_chain` depends on the parent field
theta_P and on the no-jump status of each of the two cherry half-edges
(X-edge, Y-edge). The half-edge no-jump probability is STATE-DEPENDENT:

  beta(theta_P) = exp(-rho_chain * (1 - rho[theta_P]) * t / 2)

(In contrast, Interp 1 -- the simplified mixture model that the Phase A
draft used -- has `p_nj = exp(-rho_chain * t / 2)` uniformly across
theta_P. Interp 1 is what you get if you count self-jumps as resampling
events; Interp 2 is the textbook F81 CTMC with only real transitions
counted. Both have the same field marginal but different
(residue, field) joint dynamics; Interp 2 keeps residues
parent-correlated longer at frequent fields, which is the physically
sensible regime.)

Under no-jump on a half-edge, the leaf residues evolve from the
unobserved parent residues under Q^(c, theta_P) over t/2. Under >=1
jump on a half-edge, the leaf residues resample from
`pi^(c, theta_end)` where theta_end is the post-jump field, distributed
as

  P(theta_end | >=1 jump, theta_P, t/2) =
       rho[theta_end] (1 - alpha) / (1 - beta(theta_P))
       + delta_{theta_end, theta_P} (alpha - beta(theta_P)) / (1 - beta(theta_P))

with `alpha = exp(-rho_chain * t / 2)`. This conditional distribution
decomposes into a convex combination of two factors:

  weight_J(theta_P)  = (1 - alpha) / (1 - beta(theta_P))
  weight_pi(theta_P) = (alpha - beta(theta_P)) / (1 - beta(theta_P))

(These sum to 1.) The post-jump emission field at the two coupled
columns then has the joint distribution

  J'(c_i, c_j, theta_P)(a, b) =
       weight_J(theta_P)  * J(c_i, c_j)(a, b)
     + weight_pi(theta_P) * pi^(c_i, theta_P)(a) * pi^(c_j, theta_P)(b)

where J(c_i, c_j)(a, b) = sum_theta rho[theta] pi^(c_i, theta)(a)
pi^(c_j, theta)(b) is the field-marginal joint stationary (the same J
as in Interp 1) and pi^(c, theta_P) is the per-(c, theta) stationary.

The cap-2 cherry doublet, given (c_i, c_j, theta_P), is

  P^(c_i, c_j, theta_P)(X_i, X_j, Y_i, Y_j; t) =
       beta(theta_P)^2         * Sigma^(c_i, theta_P)(X_i, Y_i; t)
                                * Sigma^(c_j, theta_P)(X_j, Y_j; t)
     + beta(theta_P)(1-beta(theta_P)) * pi^(c_i, theta_P)(X_i)
                                * pi^(c_j, theta_P)(X_j)
                                * J'(c_i, c_j, theta_P)(Y_i, Y_j)
     + (1-beta(theta_P))*beta(theta_P) * J'(c_i, c_j, theta_P)(X_i, X_j)
                                * pi^(c_i, theta_P)(Y_i)
                                * pi^(c_j, theta_P)(Y_j)
     + (1-beta(theta_P))^2   * J'(c_i, c_j, theta_P)(X_i, X_j)
                                * J'(c_i, c_j, theta_P)(Y_i, Y_j)

summed over theta_P with rho[theta_P], with the per-(c, theta) GTR
cherry joint

  Sigma^(c, theta)(a, b; t) = sum_p pi^(c, theta)(p) P_half^(c, theta)(p, a)
                                                  * P_half^(c, theta)(p, b)

(`P_half = expm(Q^(c, theta) * t / 2)`). Marginalising over (c1, c2)
with the empirical class prior pi_c yields the (A, A, A, A) doublet
emission needed by the TKF-DP composite likelihood and the IPHMM
M-tensor.

Tested by `tests/dynfield/test_math_precompute.py` (Phase A) against a
faithful Gillespie trajectory MC.

Reversibility: the joint (theta, residue) chain is reversible w.r.t.
pi(theta, x) = rho[theta] * pi^(c, theta)(x). See dynfield_math.md
"Reversibility under Interp 2" for the detailed-balance check on the
real field jumps.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import expm as _expm

from ...lg08 import S_LG08_F81_J as S_LG08_J


A = 20


# ---------------------------------------------------------------------------
# Per-(class, field) generators and half-edge transitions.
# ---------------------------------------------------------------------------

def per_class_field_Q(pi_field: np.ndarray, S: Optional[np.ndarray] = None
                       ) -> np.ndarray:
    """GTR(S, pi_field[c, theta]) per (c, theta).

    Returns (K_c, L_max, A, A) generators. Identical construction to
    block_likelihoods._per_class_Q (the per-class generator used by the
    Potts variant), iterated over field atoms; reduces to the per-class
    generator when L_max=1 and pi_field[:, 0, :] = pi_class.
    """
    pi_field = np.asarray(pi_field, dtype=np.float64)
    K_c, L_max, A_ = pi_field.shape
    assert A_ == A, f"pi_field alphabet {A_} != {A}"
    S_arr = np.asarray(S_LG08_J if S is None else S, dtype=np.float64)
    S_off = S_arr - np.diag(np.diag(S_arr))
    out = np.zeros((K_c, L_max, A, A), dtype=np.float64)
    for c in range(K_c):
        for th in range(L_max):
            pi = pi_field[c, th]
            Q = S_off * pi[None, :]
            np.fill_diagonal(Q, -Q.sum(axis=1))
            out[c, th] = Q
    return out


def per_class_field_P_half(Q_cf: np.ndarray, t: float,
                            eta: float = 1.0) -> np.ndarray:
    """(K_c, L_max, A, A) half-edge transition matrices `expm(Q_cf * t/2 * eta)`.

    Used by the (no-jump, no-jump) case of the 4-case cherry doublet
    closed form. K_c * L_max separate 20x20 expm calls -- typical
    workload of ~32 expm's at K_c=4 / L_max=8.
    """
    Q_cf = np.asarray(Q_cf, dtype=np.float64)
    K_c, L_max, A_, A__ = Q_cf.shape
    assert A_ == A and A__ == A
    out = np.zeros_like(Q_cf)
    half = 0.5 * float(t) * float(eta)
    for c in range(K_c):
        for th in range(L_max):
            out[c, th] = _expm(Q_cf[c, th] * half)
    return out


def per_class_field_cherry_sigma(P_half: np.ndarray, pi_field: np.ndarray
                                   ) -> np.ndarray:
    """Per-(c, theta) cherry joint emission Sigma^(c, theta)(a, b; t).

    `Sigma[c, theta, a, b] = sum_p pi_field[c, theta, p] *
    P_half[c, theta, p, a] * P_half[c, theta, p, b]` -- the joint of the
    two cherry leaves at one site under the no-jump-on-both-edges branch.

    Shape (K_c, L_max, A, A).
    """
    P_half = np.asarray(P_half, dtype=np.float64)
    pi_field = np.asarray(pi_field, dtype=np.float64)
    return np.einsum('ctp,ctpa,ctpb->ctab', pi_field, P_half, P_half)


# ---------------------------------------------------------------------------
# Field-marginal joint stationary J^(c1, c2)(a, b).
# ---------------------------------------------------------------------------

def field_marginal_pi(rho: np.ndarray, pi_field: np.ndarray) -> np.ndarray:
    """`pi_marg^(c)(a) = sum_theta rho[theta] * pi_field[c, theta, a]`.

    Shape (K_c, A). The lone-site stationary at class c -- the marginal
    consumed at insert/delete cells, equal to the marginal of any
    field-mixed joint.
    """
    rho = np.asarray(rho, dtype=np.float64)
    pi_field = np.asarray(pi_field, dtype=np.float64)
    return np.einsum('t,cta->ca', rho, pi_field)


def field_marginal_joint(rho: np.ndarray, pi_field: np.ndarray
                          ) -> np.ndarray:
    """`J^(c1, c2)(a, b) = sum_theta rho[theta] *
       pi_field[c1, theta, a] * pi_field[c2, theta, b]`.

    Shape (K_c, K_c, A, A). The field-marginal joint stationary at the
    cluster of two coupled sites of classes (c1, c2); the >=1-jump
    leaves emit from this distribution at high rho_chain (where the
    post-jump field is approximately rho-distributed independent of
    theta_P).
    """
    rho = np.asarray(rho, dtype=np.float64)
    pi_field = np.asarray(pi_field, dtype=np.float64)
    return np.einsum('t,uta,vtb->uvab', rho, pi_field, pi_field)


# ---------------------------------------------------------------------------
# Interp 2 weights: state-dependent no-jump probability and the J'
# decomposition.
# ---------------------------------------------------------------------------

def jump_weights(rho: np.ndarray, t: float, rho_chain: float = 1.0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per-theta_P weights for the Interp 2 cap-2 doublet/singlet.

    Returns (beta, w_J, w_pi, alpha) all per-theta_P (shape (L_max,)
    for beta, w_J, w_pi; alpha is a scalar):

      alpha             = exp(-rho_chain * t / 2)              [scalar]
      beta(theta_P)     = exp(-rho_chain * (1 - rho_theta_P) * t / 2)
      w_J(theta_P)      = (1 - alpha) / (1 - beta(theta_P))
      w_pi(theta_P)     = (alpha - beta(theta_P)) / (1 - beta(theta_P))

    For theta_P with rho_theta_P = 1 (absorbing field state -- the
    "never leaves" boundary case), beta = 1 and the >=1-jump
    contribution is zero, so w_J and w_pi don't matter; we set them to
    (1.0, 0.0) by convention to avoid 0/0.
    """
    rho = np.asarray(rho, dtype=np.float64)
    s = float(t) / 2.0
    r = float(rho_chain)
    alpha = float(np.exp(-r * s))
    beta = np.exp(-r * (1.0 - rho) * s)
    denom = 1.0 - beta
    # Boundary: where denom == 0 (rho_theta = 1), the >=1-jump weight is
    # itself zero so w_J/w_pi never get used; set them to safe defaults.
    safe = denom > 1e-15
    w_J = np.where(safe, (1.0 - alpha) / np.where(safe, denom, 1.0), 1.0)
    w_pi = np.where(safe, (alpha - beta) / np.where(safe, denom, 1.0), 0.0)
    return beta, w_J, w_pi, alpha


def field_marginal_joint_per_theta(rho: np.ndarray, pi_field: np.ndarray,
                                     t: float, rho_chain: float = 1.0
                                     ) -> np.ndarray:
    """Per-theta_P post-jump joint emission `J'(c1, c2, theta_P)(a, b)`.

    `J'(c1, c2, theta_P) = w_J(theta_P) * J(c1, c2)
                          + w_pi(theta_P) * pi_field[c1, theta_P, :]
                                          (x) pi_field[c2, theta_P, :]`

    Shape (K_c, K_c, L_max, A, A). At low rho_chain, w_pi -> 1 and
    J' is concentrated on pi_field[*, theta_P] (post-jump field stays
    near theta_P). At high rho_chain, w_J -> 1 and J' -> J (post-jump
    field is rho-distributed independent of theta_P).
    """
    rho = np.asarray(rho, dtype=np.float64)
    pi_field = np.asarray(pi_field, dtype=np.float64)
    K_c, L_max, A_ = pi_field.shape
    assert A_ == A
    J = field_marginal_joint(rho, pi_field)                          # (K, K, A, A)
    _, w_J, w_pi, _ = jump_weights(rho, t=t, rho_chain=rho_chain)
    # J'[u, v, theta_P, a, b] = w_J[theta_P] * J[u, v, a, b]
    #                          + w_pi[theta_P] * pi[u, theta_P, a] * pi[v, theta_P, b]
    term_J = np.einsum('t,uvab->uvtab', w_J, J)
    term_pi = np.einsum('t,uta,vtb->uvtab', w_pi, pi_field, pi_field)
    return term_J + term_pi


# ---------------------------------------------------------------------------
# Cap-2 cluster joint emission (cherry geometry).
# ---------------------------------------------------------------------------

def cherry_doublet_4case(t: float, rho: np.ndarray, pi_field: np.ndarray,
                          S: Optional[np.ndarray] = None,
                          eta: float = 1.0,
                          rho_chain: float = 1.0,
                          ) -> np.ndarray:
    """Cap-2 cluster joint emission `P_doublet^{c1, c2}(X_i, X_j, Y_i, Y_j; t)`
    for all (c1, c2) pairs at once.

    Returns a (K_c, K_c, A, A, A, A) tensor indexed as
    `[c1, c2, X_i, X_j, Y_i, Y_j]`. The 4-case sum (see module docstring)
    is implemented as four einsum terms plus the parent-field weight.

    `rho_chain` is the F81-on-DP rate multiplier; the per-half-edge
    no-jump probability is `p_nj = exp(-rho_chain * t / 2)`. As
    `rho_chain -> 0`, `p_nj -> 1` and the doublet reduces to the
    per-(c, theta_P) GTR cherry joint Sigma summed over theta_P with
    rho (= per-class GTR limit with the initial-field stationary).

    To match the block_likelihoods convention `[a, b, c, d]` with
    `(a, b)` = left endpoint (X_i, Y_i) and `(c, d)` = right endpoint
    (X_j, Y_j), transpose to `(c1, c2, X_i, Y_i, X_j, Y_j)`. See
    `class_marginal_doublet` for the consumer-facing K_c-marginalised
    output that does the transpose.
    """
    rho = np.asarray(rho, dtype=np.float64)
    pi_field = np.asarray(pi_field, dtype=np.float64)
    K_c, L_max, A_ = pi_field.shape
    assert A_ == A

    Q_cf = per_class_field_Q(pi_field, S=S)                          # (K, L, A, A)
    P_half = per_class_field_P_half(Q_cf, t=t, eta=eta)               # (K, L, A, A)
    Sigma = per_class_field_cherry_sigma(P_half, pi_field)            # (K, L, A, A)
    # Interp 2 weights and per-theta_P J'.
    beta, _, _, _ = jump_weights(rho, t=t, rho_chain=rho_chain)
    J_prime = field_marginal_joint_per_theta(
        rho, pi_field, t=t, rho_chain=rho_chain)                     # (K, K, L, A, A)

    # Working internal layout: (c1, c2, X_i, X_j, Y_i, Y_j) with subscripts
    # (u=c1, v=c2, x=X_i, y=X_j, a=Y_i, b=Y_j). The 4-case sum integrates
    # theta_P with rho[theta_P]; per-theta_P weights are beta(theta_P)
    # for "no jump on this half-edge" and 1-beta(theta_P) for ">=1 jump".
    nn = rho * beta * beta                                          # (L,)
    T1 = np.einsum('t,utxa,vtyb->uvxyab', nn, Sigma, Sigma)
    ny = rho * beta * (1.0 - beta)
    T2 = np.einsum('t,utx,vty,uvtab->uvxyab', ny, pi_field, pi_field, J_prime)
    yn = rho * (1.0 - beta) * beta
    T3 = np.einsum('t,uvtxy,uta,vtb->uvxyab', yn, J_prime, pi_field, pi_field)
    yy = rho * (1.0 - beta) ** 2
    T4 = np.einsum('t,uvtxy,uvtab->uvxyab', yy, J_prime, J_prime)

    # Internal layout: (c1, c2, X_i, X_j, Y_i, Y_j)
    return T1 + T2 + T3 + T4


def class_marginal_doublet(t: float, pi_c: np.ndarray, rho: np.ndarray,
                            pi_field: np.ndarray,
                            S: Optional[np.ndarray] = None,
                            eta: float = 1.0,
                            rho_chain: float = 1.0,
                            ) -> np.ndarray:
    """K_c-marginalised cap-2 cluster joint emission, in the
    block_likelihoods doublet layout.

    Returns (A, A, A, A) indexed as `[a, b, c, d]` with `(a, b)` = left
    endpoint (X_i, Y_i), `(c, d)` = right endpoint (X_j, Y_j).

    Args:
      t: cherry diameter.
      pi_c: (K_c,) empirical class prior.
      rho: (L_max,) DP stick-breaking weights for the field selector.
      pi_field: (K_c, L_max, A) per-(class, field) stationary.
      S: optional exchangeability matrix override (defaults to LG08
        F81-normalised).
      eta: rate multiplier (default 1.0; matches the K=4 release).
      rho_chain: F81-on-DP rate multiplier (default 1.0).
    """
    pi_c = np.asarray(pi_c, dtype=np.float64)
    K_c = pi_c.shape[0]
    P_internal = cherry_doublet_4case(
        t, rho=rho, pi_field=pi_field, S=S, eta=eta,
        rho_chain=rho_chain)                          # (K, K, Xi, Xj, Yi, Yj)
    # Class-marginalise: sum_{c1, c2} pi_c(c1) pi_c(c2) * P_internal
    weights = pi_c[:, None] * pi_c[None, :]                           # (K, K)
    marg = np.einsum('uv,uvxyab->xyab', weights, P_internal)          # (Xi, Xj, Yi, Yj)
    # Transpose to block_likelihoods layout (a, b, c, d) = (Xi, Yi, Xj, Yj):
    #   internal axes (0=Xi, 1=Xj, 2=Yi, 3=Yj) -> (Xi, Yi, Xj, Yj) = (0, 2, 1, 3)
    return np.transpose(marg, (0, 2, 1, 3))


# ---------------------------------------------------------------------------
# Cap-2 cluster joint emission, singlet block (uncoupled column).
# ---------------------------------------------------------------------------

def cherry_singlet_4case(t: float, rho: np.ndarray, pi_field: np.ndarray,
                          S: Optional[np.ndarray] = None,
                          eta: float = 1.0,
                          rho_chain: float = 1.0,
                          ) -> np.ndarray:
    """Cap-2 singleton emission at a single uncoupled column per class.

    For an uncoupled column at class c, the joint of the two leaf
    residues (X_i, Y_i) at cherry diameter t under the dynamic-field
    model is the same 4-case sum specialised to the single-site
    (c1 = c2 = c, but one column only -- no within-cherry residue
    coupling). Equivalently, J collapses to outer(pi_marg, pi_marg) on
    the diagonal class.

    Returns (K_c, A, A) indexed `[c, X_i, Y_i]`.

    `rho_chain` is the F81-on-DP rate multiplier (default 1.0).
    """
    rho = np.asarray(rho, dtype=np.float64)
    pi_field = np.asarray(pi_field, dtype=np.float64)
    K_c, L_max, A_ = pi_field.shape
    assert A_ == A

    Q_cf = per_class_field_Q(pi_field, S=S)
    P_half = per_class_field_P_half(Q_cf, t=t, eta=eta)
    Sigma = per_class_field_cherry_sigma(P_half, pi_field)            # (K, L, A, A)
    pi_marg = field_marginal_pi(rho, pi_field)                        # (K, A)
    beta, w_J, w_pi, _ = jump_weights(rho, t=t, rho_chain=rho_chain)
    # Per-theta_P post-jump singleton emission `pi_marg'(c, theta_P)`:
    #   = w_J[theta_P] * pi_marg[c, :] + w_pi[theta_P] * pi_field[c, theta_P, :]
    # Shape (K_c, L_max, A).
    pi_prime = (np.einsum('t,ca->cta', w_J, pi_marg)
                + np.einsum('t,cta->cta', w_pi, pi_field))            # (K_c, L_max, A)

    # 4-case sum, theta_P-marginalised.
    nn = rho * beta * beta
    T1 = np.einsum('t,ctxa->cxa', nn, Sigma)
    ny = rho * beta * (1.0 - beta)
    # nn-yj: pi[c, theta_P, Xi] * pi_prime[c, theta_P, Yi]
    T2 = np.einsum('t,ctx,cta->cxa', ny, pi_field, pi_prime)
    yn = rho * (1.0 - beta) * beta
    T3 = np.einsum('t,ctx,cta->cxa', yn, pi_prime, pi_field)
    yy = rho * (1.0 - beta) ** 2
    T4 = np.einsum('t,ctx,cta->cxa', yy, pi_prime, pi_prime)

    return T1 + T2 + T3 + T4


def class_marginal_singlet(t: float, pi_c: np.ndarray, rho: np.ndarray,
                            pi_field: np.ndarray,
                            S: Optional[np.ndarray] = None,
                            eta: float = 1.0,
                            rho_chain: float = 1.0,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K_c-marginalised singleton emission in the block_likelihoods layout.

    Returns `(P_singlet, pi_out_eff, sub_matrix_eff)`:
      P_singlet: (A, A) cherry joint emission, K_c-marginalised.
      pi_out_eff: (A,) lone-site stationary = sum_c pi_c[c] * pi_marg^(c).
      sub_matrix_eff: (A, A) = P_singlet[a, b] / pi_out_eff[a].

    The shape contract matches block_likelihoods.build_singlet_emission
    so consumers (make_tkf92_pair_hmm, IPHMM precompute) work unchanged.

    `rho_chain` is the F81-on-DP rate multiplier (default 1.0).
    """
    pi_c = np.asarray(pi_c, dtype=np.float64)
    K_c = pi_c.shape[0]
    per_class = cherry_singlet_4case(
        t, rho=rho, pi_field=pi_field, S=S, eta=eta, rho_chain=rho_chain)
    # (K_c, A, A) -> (A, A) by class-marginalising:
    P_singlet = np.einsum('c,cab->ab', pi_c, per_class)
    pi_marg = field_marginal_pi(rho, pi_field)                       # (K_c, A)
    pi_out_eff = np.einsum('c,ca->a', pi_c, pi_marg)                 # (A,)
    sub_matrix_eff = P_singlet / np.clip(pi_out_eff[:, None], 1e-300, None)
    return P_singlet, pi_out_eff, sub_matrix_eff


# ---------------------------------------------------------------------------
# Variable-size cluster emission (Phase D.5).
# ---------------------------------------------------------------------------

def cluster_emission_per_theta(
        t: float,
        rho: np.ndarray, pi_field: np.ndarray,
        classes: np.ndarray, X_obs: np.ndarray, Y_obs: np.ndarray,
        *, S: Optional[np.ndarray] = None,
        eta: float = 1.0,
        rho_chain: float = 1.0,
        precomputed_Sigma: Optional[np.ndarray] = None,
        ) -> tuple[np.ndarray, dict]:
    """Per-theta_P cap-2 cluster joint emission for a cluster of arbitrary
    size m >= 1 (variable-cluster-size analogue of cherry_doublet_4case).

    The dynfield model's per-theta_P emissions factor over the sites in
    the cluster, so the size-m formula has the same 4-case structure
    as the size-2 case but with per-site products replacing the
    two-site outer products.

    Inputs:
      classes: (m,) int array of per-site class labels c_1, ..., c_m.
      X_obs:   (m,) int array of leaf-X residues.
      Y_obs:   (m,) int array of leaf-Y residues.
      precomputed_Sigma: optional (K_c, L_max, A, A) Sigma tensor to
        avoid re-computing the per-(c, theta) cherry joint (precompute
        once per t).

    Returns:
      P_per_theta_case: (L_max, 4) array of per-(theta_P, case)
        unnormalised likelihoods rho[theta_P] * case_weight * case_value.
        (Already weighted by rho.) Summed over theta_P and case gives
        the cluster emission probability p(X_obs, Y_obs | classes, t).
      info: dict with the per-cluster intermediate quantities used by the
        attribution path (pi_prod_X, pi_prod_Y, J_X, J_Y, J_prime_X,
        J_prime_Y, beta, w_J, w_pi, Sigma_prod).
    """
    rho = np.asarray(rho, dtype=np.float64)
    pi_field = np.asarray(pi_field, dtype=np.float64)
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)
    X_obs = np.asarray(X_obs, dtype=np.int64).reshape(-1)
    Y_obs = np.asarray(Y_obs, dtype=np.int64).reshape(-1)
    K_c, L_max, A_ = pi_field.shape
    assert A_ == A
    m = classes.shape[0]
    assert X_obs.shape == (m,) and Y_obs.shape == (m,)

    # Per-(c, theta) cherry joint Sigma -- shared across clusters, so the
    # caller can precompute once per t and pass in.
    if precomputed_Sigma is None:
        Q_cf = per_class_field_Q(pi_field, S=S)
        P_half = per_class_field_P_half(Q_cf, t=t, eta=eta)
        Sigma = per_class_field_cherry_sigma(P_half, pi_field)
    else:
        Sigma = precomputed_Sigma

    beta, w_J, w_pi, _ = jump_weights(rho, t=t, rho_chain=rho_chain)

    # Per-site, per-theta pi value: shape (m, L_max).
    pi_at_X = pi_field[classes, :, X_obs]                  # (m, L_max)
    pi_at_Y = pi_field[classes, :, Y_obs]
    # Cluster-product over sites, per theta. shape (L_max,).
    pi_prod_X = pi_at_X.prod(axis=0)
    pi_prod_Y = pi_at_Y.prod(axis=0)
    # Per-site, per-theta Sigma value: shape (m, L_max).
    Sigma_at = Sigma[classes, :, X_obs, Y_obs]             # (m, L_max)
    Sigma_prod = Sigma_at.prod(axis=0)                     # (L_max,)
    # J^(classes)(X) = sum_theta rho[theta] * pi_prod_X[theta], scalar.
    J_X = float((rho * pi_prod_X).sum())
    J_Y = float((rho * pi_prod_Y).sum())
    # J'^(classes, theta_P) -- (L_max,) for each side.
    J_prime_X = w_J * J_X + w_pi * pi_prod_X
    J_prime_Y = w_J * J_Y + w_pi * pi_prod_Y

    # 4-case per-theta values (before rho weighting).
    T1 = Sigma_prod                                         # (L_max,)
    T2 = pi_prod_X * J_prime_Y
    T3 = J_prime_X * pi_prod_Y
    T4 = J_prime_X * J_prime_Y

    # Case weights from beta.
    one_m_beta = 1.0 - beta
    L_T1 = beta * beta * T1
    L_T2 = beta * one_m_beta * T2
    L_T3 = one_m_beta * beta * T3
    L_T4 = one_m_beta * one_m_beta * T4
    P_T1 = rho * L_T1
    P_T2 = rho * L_T2
    P_T3 = rho * L_T3
    P_T4 = rho * L_T4
    P_per_theta_case = np.stack([P_T1, P_T2, P_T3, P_T4], axis=1)  # (L_max, 4)

    info = {
        'beta': beta, 'w_J': w_J, 'w_pi': w_pi,
        'pi_prod_X': pi_prod_X, 'pi_prod_Y': pi_prod_Y,
        'J_X': J_X, 'J_Y': J_Y,
        'J_prime_X': J_prime_X, 'J_prime_Y': J_prime_Y,
        'Sigma': Sigma, 'rho': rho, 'pi_field': pi_field,
        'classes': classes, 'X_obs': X_obs, 'Y_obs': Y_obs,
    }
    return P_per_theta_case, info


# ---------------------------------------------------------------------------
# Batched variable-size cluster emission across multiple cherries.
# ---------------------------------------------------------------------------

def precompute_cluster_emission_per_cherry(
        tau: np.ndarray,
        rho: np.ndarray,
        pi_field: np.ndarray,
        *,
        S: Optional[np.ndarray] = None,
        eta: float = 1.0,
        rho_chain: float = 1.0,
        ) -> dict:
    """Build per-cherry tables consumed by `cluster_emission_batched`.

    Computes for each cherry q (diameter `tau[q]`):
      - `Sigma_per_cherry[q, c, theta, a, b]` -- per-(c, theta) cherry joint.
      - `beta_per_cherry[q, theta]`, `w_J_per_cherry[q, theta]`,
        `w_pi_per_cherry[q, theta]` -- Interp 2 weights.

    Returns a dict with `Sigma`, `beta`, `w_J`, `w_pi`, `rho`, `pi_field`,
    `rho_chain` keys. All arrays are np.float64.
    """
    rho = np.asarray(rho, dtype=np.float64)
    pi_field = np.asarray(pi_field, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    K_c, L_max, A_ = pi_field.shape
    assert A_ == A
    n_cherries = tau.shape[0]

    Q_cf = per_class_field_Q(pi_field, S=S)
    Sigma_per_cherry = np.zeros((n_cherries, K_c, L_max, A, A),
                                 dtype=np.float64)
    cache: dict = {}
    for q in range(n_cherries):
        key = float(tau[q])
        cached = cache.get(key)
        if cached is None:
            P_half = per_class_field_P_half(Q_cf, t=key, eta=eta)
            cached = per_class_field_cherry_sigma(P_half, pi_field)
            cache[key] = cached
        Sigma_per_cherry[q] = cached

    s_per_q = 0.5 * tau                                      # (n,)
    r = float(rho_chain)
    one_m_rho = 1.0 - rho                                    # (L,)
    beta = np.exp(-r * one_m_rho[None, :] * s_per_q[:, None])  # (n, L)
    alpha = np.exp(-r * s_per_q)                             # (n,)
    denom = 1.0 - beta                                       # (n, L)
    safe = denom > 1e-15
    safe_denom = np.where(safe, denom, 1.0)
    w_J = np.where(safe, (1.0 - alpha)[:, None] / safe_denom, 1.0)
    w_pi = np.where(safe, (alpha[:, None] - beta) / safe_denom, 0.0)

    return {
        'Sigma': Sigma_per_cherry,
        'beta': beta,
        'w_J': w_J,
        'w_pi': w_pi,
        'rho': rho,
        'pi_field': pi_field,
        'rho_chain': float(rho_chain),
    }


def cluster_emission_batched(
        classes: np.ndarray,
        X_batch: np.ndarray,
        Y_batch: np.ndarray,
        mask_X: np.ndarray,
        mask_Y: np.ndarray,
        per_cherry: dict,
        ) -> np.ndarray:
    """Per-cherry cluster emission probability for a single cluster
    `classes` evaluated jointly across all cherries, **marginalising
    over gapped leaf positions** rather than dropping cherries.

    The stochasticity identities of the per-(c, theta) factors
    (`Sigma.per_class_field_cherry_sigma`) give clean rules for
    marginalising an unobserved AA at site s:
      - If X[q, s] is gapped: pi_at_X[q, s, theta] -> 1.0 (sum_a pi = 1).
      - If Y[q, s] is gapped: pi_at_Y[q, s, theta] -> 1.0.
      - If only one of X, Y observed at s: Sigma_at[q, s, theta] reduces
        to pi^(c_s, theta)(observed leaf) by sum_a Sigma(a, b; t) = pi(b).
      - If both gapped at s: Sigma_at[q, s, theta] -> 1.0
        (sum_{a, b} Sigma = 1).
    Cherries with all sites gapped contribute total = 1.0 (log = 0).

    Inputs:
      classes:    (m,) int array of per-site class labels c_1, ..., c_m.
      X_batch:    (n_cherries, m) int array; values at masked-out
                  positions are ignored but must be valid AA indices
                  (the kernel uses fancy indexing).
      Y_batch:    (n_cherries, m) int.
      mask_X:     (n_cherries, m) bool; True where leaf-X is observed.
      mask_Y:     (n_cherries, m) bool; True where leaf-Y is observed.
      per_cherry: dict from `precompute_cluster_emission_per_cherry`.

    Returns:
      total_per_cherry: (n_cherries,) array of per-cherry total emission
        probabilities (NOT log). No cherry filtering.
    """
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)
    pi_field = per_cherry['pi_field']
    rho = per_cherry['rho']
    K_c, L_max, A_ = pi_field.shape

    mask_X = np.asarray(mask_X, dtype=bool)
    mask_Y = np.asarray(mask_Y, dtype=bool)
    X_batch = np.asarray(X_batch, dtype=np.int64)
    Y_batch = np.asarray(Y_batch, dtype=np.int64)
    n, m = X_batch.shape
    L = L_max

    if m == 0:
        return np.ones((n,), dtype=np.float64)

    Sigma = per_cherry['Sigma']                     # (n, K, L, A, A)
    beta = per_cherry['beta']                       # (n, L)
    w_J = per_cherry['w_J']                         # (n, L)
    w_pi = per_cherry['w_pi']                       # (n, L)

    # Sanitise gather indices so fancy indexing never produces a stale
    # out-of-range read at gapped positions (we'll mask them out anyway).
    X_safe = np.where(mask_X, X_batch, 0)
    Y_safe = np.where(mask_Y, Y_batch, 0)

    classes_b = classes[None, :, None]              # (1, m, 1)
    theta_b = np.arange(L)[None, None, :]           # (1, 1, L)
    X_b = X_safe[:, :, None]                        # (n, m, 1)
    Y_b = Y_safe[:, :, None]                        # (n, m, 1)
    pi_at_X = pi_field[classes_b, theta_b, X_b]     # (n, m, L)
    pi_at_Y = pi_field[classes_b, theta_b, Y_b]     # (n, m, L)
    # Marginalise gapped sites: factor -> 1.0.
    mX = mask_X[:, :, None]
    mY = mask_Y[:, :, None]
    pi_at_X = np.where(mX, pi_at_X, 1.0)
    pi_at_Y = np.where(mY, pi_at_Y, 1.0)

    q_b = np.arange(n)[:, None, None]               # (n, 1, 1)
    Sigma_full = Sigma[q_b, classes_b, theta_b, X_b, Y_b]  # (n, m, L)
    # Sigma_at marginalisation:
    #   (X obs, Y obs): Sigma[c, theta, X, Y]
    #   (X obs, Y gap): sum_y Sigma = pi(X) = pi_at_X
    #   (X gap, Y obs): sum_x Sigma = pi(Y) = pi_at_Y
    #   (X gap, Y gap): sum_{x,y} Sigma = 1.0 = pi_at_X * pi_at_Y
    # In all three not-both-observed cases the right value is
    # pi_at_X * pi_at_Y (since the gapped side's pi factor is 1).
    both_obs = mX & mY
    Sigma_at = np.where(both_obs, Sigma_full, pi_at_X * pi_at_Y)

    pi_prod_X = pi_at_X.prod(axis=1)                # (n, L)
    pi_prod_Y = pi_at_Y.prod(axis=1)
    Sigma_prod = Sigma_at.prod(axis=1)

    J_X = (rho[None, :] * pi_prod_X).sum(axis=1, keepdims=True)  # (n, 1)
    J_Y = (rho[None, :] * pi_prod_Y).sum(axis=1, keepdims=True)
    J_prime_X = w_J * J_X + w_pi * pi_prod_X        # (n, L)
    J_prime_Y = w_J * J_Y + w_pi * pi_prod_Y

    one_m_beta = 1.0 - beta                         # (n, L)
    P_per_theta = rho[None, :] * (
        beta * beta * Sigma_prod
        + beta * one_m_beta * pi_prod_X * J_prime_Y
        + one_m_beta * beta * J_prime_X * pi_prod_Y
        + one_m_beta * one_m_beta * J_prime_X * J_prime_Y
    )                                               # (n, L)
    return P_per_theta.sum(axis=1)                  # (n,)


def cluster_emission_batched_soft(
        classes: np.ndarray,
        X_soft: np.ndarray,
        Y_soft: np.ndarray,
        per_cherry: dict,
        ) -> np.ndarray:
    """Soft-observation variant of `cluster_emission_batched`.

    Whereas the hard version indexes pi_field and Sigma at concrete AA
    values `aa_a, aa_b`, this version marginalises them against
    per-cherry, per-site PSWMs `X_soft, Y_soft` of shape
    `(n_cherries, m, A)`. Both terms are LINEAR in X_soft and Y_soft, so
    the substitution is a direct tensor contraction:

      pi_at_X_soft[q, s, theta] = sum_a X_soft[q, s, a] * pi(c_s, theta, a)
      Sigma_soft [q, s, theta] = sum_{a, b} X_soft[q, s, a] * Y_soft[q, s, b]
                                   * Sigma[q, c_s, theta, a, b]

    When X_soft and Y_soft are delta functions on hard residues, this
    exactly reduces to `cluster_emission_batched`.

    No mask parameter: every leaf-site pair is assumed to have a valid
    PSWM (the peeling in `tkfdp.pswm_peeling` returns pi as fallback for
    fully-gapped columns, so every position is a proper distribution).

    Inputs:
      classes: (m,) int per-site class labels.
      X_soft, Y_soft: (n_cherries, m, A) float, per-site PSWMs.
      per_cherry: dict from `precompute_cluster_emission_per_cherry`.

    Returns:
      total_per_cherry: (n_cherries,) float, per-cherry total emission
        probability (NOT log).
    """
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)
    pi_field = per_cherry['pi_field']                    # (K_c, L, A)
    rho = per_cherry['rho']                              # (L,)
    K_c, L_max, A_ = pi_field.shape

    X_soft = np.asarray(X_soft, dtype=np.float64)
    Y_soft = np.asarray(Y_soft, dtype=np.float64)
    n, m, A_x = X_soft.shape
    assert A_x == A_, f"A mismatch: X_soft has A={A_x}, pi_field has A={A_}"
    assert classes.shape[0] == m, f"classes has {classes.shape[0]}, expected {m}"

    if m == 0:
        return np.ones((n,), dtype=np.float64)

    Sigma = per_cherry['Sigma']                          # (n, K, L, A, A)
    beta = per_cherry['beta']                            # (n, L)
    w_J = per_cherry['w_J']                              # (n, L)
    w_pi = per_cherry['w_pi']                            # (n, L)

    pi_at_classes = pi_field[classes]                    # (m, L, A)
    # pi_at_X[q, s, theta] = sum_a X_soft[q, s, a] * pi_at_classes[s, theta, a]
    pi_at_X = np.einsum('qsa,sta->qst', X_soft, pi_at_classes)  # (n, m, L)
    pi_at_Y = np.einsum('qsa,sta->qst', Y_soft, pi_at_classes)

    # Sigma_at_classes[q, s, theta, a, b] = Sigma[q, classes[s], theta, a, b]
    Sigma_at_classes = Sigma[:, classes, :, :, :]        # (n, m, L, A, A)
    # Sigma_at[q, s, theta] = sum_{a, b} X_soft[q, s, a] * Y_soft[q, s, b]
    #                       * Sigma_at_classes[q, s, theta, a, b]
    Sigma_at = np.einsum('qsa,qsb,qstab->qst',
                             X_soft, Y_soft, Sigma_at_classes)

    pi_prod_X = pi_at_X.prod(axis=1)                     # (n, L)
    pi_prod_Y = pi_at_Y.prod(axis=1)
    Sigma_prod = Sigma_at.prod(axis=1)

    J_X = (rho[None, :] * pi_prod_X).sum(axis=1, keepdims=True)
    J_Y = (rho[None, :] * pi_prod_Y).sum(axis=1, keepdims=True)
    J_prime_X = w_J * J_X + w_pi * pi_prod_X
    J_prime_Y = w_J * J_Y + w_pi * pi_prod_Y

    one_m_beta = 1.0 - beta
    P_per_theta = rho[None, :] * (
        beta * beta * Sigma_prod
        + beta * one_m_beta * pi_prod_X * J_prime_Y
        + one_m_beta * beta * J_prime_X * pi_prod_Y
        + one_m_beta * one_m_beta * J_prime_X * J_prime_Y
    )
    return P_per_theta.sum(axis=1)


# ---------------------------------------------------------------------------
# JAX-batched cluster scorer: B clusters in parallel, device-resident state.
# ---------------------------------------------------------------------------

def _shared_jax_kernel():
    """Module-level @jit'd kernel shared across all `BatchedDynfieldScorer`
    instances. The JAX compile cache is keyed on the shape signature of the
    arguments, so MSAs with the same (n_cherries, L_cols, K_c, L_max,
    m_max, bucket) shape signature share the compiled binary -- one compile
    per shape, not per scorer instance.

    Lazily constructed on first use and cached on this module.
    """
    import jax
    import jax.numpy as jnp
    from jax import jit

    @jit
    def kernel(pi_field_d, rho_d, Sigma_d, beta_d, w_J_d, w_pi_d,
                aa_a_d, aa_b_d, both_aa_d,
                padded_cols, m_mask, classes_padded):
        # Gap-marginalised cluster log-likelihood. The per-(c, theta)
        # stochasticity identities (sum_a pi = 1, sum_a Sigma(a, b) = pi(b),
        # sum_{a, b} Sigma = 1) let us replace gapped-leaf factors with
        # 1.0 (pi_at) or pi(observed) (Sigma_at), without dropping the
        # cherry. Cherries with all sites gapped contribute log 0.
        # `both_aa_d` retained in the signature for API stability but no
        # longer used; X-obs / Y-obs are derived from aa_a_d / aa_b_d
        # directly (gap = 20).
        del both_aa_d
        L_max = pi_field_d.shape[1]
        n_cherries = Sigma_d.shape[0]

        X_bqs = aa_a_d[:, padded_cols]                       # (n, B, m_max)
        X_bqs = jnp.transpose(X_bqs, (1, 0, 2))               # (B, n, m_max)
        Y_bqs = aa_b_d[:, padded_cols]
        Y_bqs = jnp.transpose(Y_bqs, (1, 0, 2))

        # Per-(B, q, s) observation masks (True iff cluster member AND
        # leaf observed at that site).
        m_mask_b = m_mask[:, None, :]                         # (B, 1, m_max)
        X_obs = (X_bqs < 20) & m_mask_b                       # (B, n, m_max)
        Y_obs = (Y_bqs < 20) & m_mask_b
        both_obs = X_obs & Y_obs

        # Sanitise gather indices (any "safe" valid value; the mask kills
        # the factor downstream).
        X_safe = jnp.where(X_obs, X_bqs, 0)
        Y_safe = jnp.where(Y_obs, Y_bqs, 0)
        classes_safe = jnp.where(m_mask, classes_padded, 0)

        classes_b = classes_safe[:, None, :, None]
        theta_b = jnp.arange(L_max)[None, None, None, :]
        X_b = X_safe[:, :, :, None]
        Y_b = Y_safe[:, :, :, None]
        pi_at_X = pi_field_d[classes_b, theta_b, X_b]         # (B, n, m_max, L)
        pi_at_Y = pi_field_d[classes_b, theta_b, Y_b]
        q_b = jnp.arange(n_cherries)[None, :, None, None]
        Sigma_full = Sigma_d[q_b, classes_b, theta_b, X_b, Y_b]

        # Marginalise gapped leaves -> factor 1.0 (= identity for prod).
        mX = X_obs[:, :, :, None].astype(pi_at_X.dtype)
        mY = Y_obs[:, :, :, None].astype(pi_at_Y.dtype)
        pi_at_X = mX * pi_at_X + (1.0 - mX)
        pi_at_Y = mY * pi_at_Y + (1.0 - mY)
        # Sigma_at: full Sigma when both observed, else pi_at_X * pi_at_Y
        # (which evaluates to pi(X) / pi(Y) / 1.0 in the X-only / Y-only /
        # both-gap cases by the stochasticity identities above).
        bo = both_obs[:, :, :, None].astype(pi_at_X.dtype)
        Sigma_at = bo * Sigma_full + (1.0 - bo) * (pi_at_X * pi_at_Y)

        pi_prod_X = pi_at_X.prod(axis=2)
        pi_prod_Y = pi_at_Y.prod(axis=2)
        Sigma_prod = Sigma_at.prod(axis=2)

        J_X = (rho_d[None, None, :] * pi_prod_X).sum(axis=2, keepdims=True)
        J_Y = (rho_d[None, None, :] * pi_prod_Y).sum(axis=2, keepdims=True)
        beta_q = beta_d[None, :, :]
        w_J_q = w_J_d[None, :, :]
        w_pi_q = w_pi_d[None, :, :]
        J_prime_X = w_J_q * J_X + w_pi_q * pi_prod_X
        J_prime_Y = w_J_q * J_Y + w_pi_q * pi_prod_Y

        one_m_beta = 1.0 - beta_q
        P_per_theta = rho_d[None, None, :] * (
            beta_q * beta_q * Sigma_prod
            + beta_q * one_m_beta * pi_prod_X * J_prime_Y
            + one_m_beta * beta_q * J_prime_X * pi_prod_Y
            + one_m_beta * one_m_beta * J_prime_X * J_prime_Y
        )
        total = P_per_theta.sum(axis=2)
        log_total = jnp.log(jnp.maximum(total, 1e-300))
        # All cherries contribute -- gap-only cherries naturally produce
        # log_total = 0 (total = 1.0) by the stochasticity identities.
        return log_total.sum(axis=1)
    return kernel


_SHARED_KERNEL = None


def _get_shared_kernel():
    global _SHARED_KERNEL
    if _SHARED_KERNEL is None:
        # Enable fp64 globally before constructing the kernel.
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass
        _SHARED_KERNEL = _shared_jax_kernel()
    return _SHARED_KERNEL


def _make_geom_buckets(B_max: int, ratio: float = None) -> 'list[int]':
    """Geometric bucket sizes at powers of `ratio` (default sqrt(2)),
    rounded to nearest int and dedup'd, starting at 1 and ending at B_max.

    Used to bucket variable batch sizes B passed to the JAX kernel so that
    the kernel only recompiles for a bounded set of shapes (one per
    bucket). Common small-B values (1, 2, 3) get exact buckets; larger B
    is padded up to <= 41% over the request.
    """
    import math as _math
    if ratio is None:
        ratio = _math.sqrt(2.0)
    buckets: 'list[int]' = []
    seen: set = set()
    val = 1.0
    while True:
        b = max(1, round(val))
        if b > B_max:
            break
        if b not in seen:
            seen.add(b)
            buckets.append(b)
        val *= ratio
    if not buckets or buckets[-1] != B_max:
        buckets.append(B_max)
    return sorted(set(buckets))


class BatchedDynfieldScorer:
    """JAX-JIT'd batched cluster log-likelihood scorer for a single MSA.

    Construction is O(n_cherries * K_c * L_max) numpy work plus one device
    transfer of the per-cherry Sigma / beta tables (~10MB at K_c=L_max=4,
    n_cherries=200). The hot path is `scorer.score_batch(list_of_column_lists,
    cls_array)`, which:
      1. Rounds B up to the nearest sqrt(2)-spaced bucket (exact at B in
         {1, 2, 3}; <= 41% over-pad otherwise).
      2. Calls the cached JIT kernel for that bucket.
      3. Returns (B,) log-likelihoods (un-padded slice).

    The kernel re-JITs once per distinct bucket size; ~20 distinct buckets
    cover the range up to L_cols+3.
    """
    def __init__(self, aa_a: np.ndarray, aa_b: np.ndarray,
                  both_aa: np.ndarray, tau: np.ndarray,
                  rho: np.ndarray, pi_field: np.ndarray,
                  *, rho_chain: float = 1.0, max_cluster_size: int = 16,
                  batch_size: Optional[int] = None,
                  n_cherries_pad: Optional[int] = None,
                  L_cols_pad: Optional[int] = None,
                  S: Optional[np.ndarray] = None, eta: float = 1.0):
        """Construct a per-MSA scorer.

        `n_cherries_pad` and `L_cols_pad` (when set) pad the per-cherry
        and per-column tensors to the requested sizes. Padded cherries are
        filled with the gap marker so they contribute log 0 under the
        gap-marginalisation rule. Padded column positions are likewise
        gap-marked; CRP candidates never reference them (they only use
        cols in [0, L_cols)) but the kernel still computes the same shape.

        This is the key knob for sharing the JAX compile cache ACROSS
        MSAs of different sizes: set both pads to corpus-wide maxima
        once per training run, and all MSAs land on a single compiled
        kernel per B-bucket.
        """
        import jax
        import jax.numpy as jnp
        from jax import jit
        # Require fp64 to match the numpy reference; safe to enable globally
        # (other JAX users in this codebase already assume fp64 via
        # `dtype=jnp.float64` literals).
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass
        self._jax = jax
        self._jnp = jnp

        per = precompute_cluster_emission_per_cherry(
            tau=tau, rho=rho, pi_field=pi_field,
            S=S, eta=eta, rho_chain=rho_chain)
        self.n_cherries = aa_a.shape[0]
        self.L_cols = aa_a.shape[1]
        self.K_c, self.L_max, self.A = pi_field.shape
        self.m_max = int(max_cluster_size)
        n_cherries_pad = int(n_cherries_pad) if n_cherries_pad else self.n_cherries
        L_cols_pad = int(L_cols_pad) if L_cols_pad else self.L_cols
        assert n_cherries_pad >= self.n_cherries
        assert L_cols_pad >= self.L_cols
        self.n_cherries_pad = n_cherries_pad
        self.L_cols_pad = L_cols_pad
        # B_max covers the worst-case CRP candidate count per column:
        # (stay) + (singleton) + 2 * (n_other_clusters), n_other_clusters
        # at most L-1 with all singletons. We use L_cols_pad+3 (the pad,
        # not the original L_cols) so the bucket set matches across MSAs.
        if batch_size is None:
            self.B_max = int(self.L_cols_pad + 3)
        else:
            self.B_max = int(batch_size)
        # Geometric buckets so the kernel only recompiles for a bounded
        # set of B shapes. Common small B (1, 2, 3) get exact buckets.
        self._buckets = _make_geom_buckets(self.B_max)
        # Lazy per-bucket numpy host buffers (reused across calls).
        self._buf_cache: dict = {}

        # Pad per-cherry tensors. Pad cherries get aa = 20 (gap) so they
        # contribute log 0 under gap-marginalisation.
        #
        # The Sigma value at pad rows is arbitrary (masked out by
        # both_obs=False -> Sigma_at = pi_at_X * pi_at_Y = 1 in the kernel).
        # But beta/w_J/w_pi enter the 4-case mixture as
        #   P_per_theta = rho * (beta^2 T1 + beta(1-beta)(T2+T3) + (1-beta)^2 T4)
        # with T_k built from pi_prod_X / pi_prod_Y / J_prime_X / J_prime_Y
        # (all = 1.0 for pad cherries). The mixture collapses to rho * 1
        # iff w_J + w_pi = 1 (since J_prime_(X,Y) = w_J * J_(X,Y) + w_pi *
        # pi_prod_(X,Y) -> w_J + w_pi = 1 when both J and pi_prod are 1).
        # At t=0 (the canonical "no time elapsed" cherry) beta=1, w_J=1,
        # w_pi=0 -- we use that as the pad fill.
        n_real = self.n_cherries
        K, L = self.K_c, self.L_max
        A = self.A
        Sigma_pad = np.zeros((n_cherries_pad, K, L, A, A), dtype=np.float64)
        Sigma_pad[:n_real] = per['Sigma']
        beta_pad = np.ones((n_cherries_pad, L), dtype=np.float64)
        beta_pad[:n_real] = per['beta']
        w_J_pad = np.ones((n_cherries_pad, L), dtype=np.float64)
        w_J_pad[:n_real] = per['w_J']
        w_pi_pad = np.zeros((n_cherries_pad, L), dtype=np.float64)
        w_pi_pad[:n_real] = per['w_pi']
        aa_a_pad = np.full((n_cherries_pad, L_cols_pad), 20, dtype=np.int32)
        aa_a_pad[:n_real, :self.L_cols] = aa_a.astype(np.int32)
        aa_b_pad = np.full((n_cherries_pad, L_cols_pad), 20, dtype=np.int32)
        aa_b_pad[:n_real, :self.L_cols] = aa_b.astype(np.int32)
        both_aa_pad = np.zeros((n_cherries_pad, L_cols_pad), dtype=bool)
        both_aa_pad[:n_real, :self.L_cols] = both_aa.astype(bool)

        self._pi_field_d = jnp.asarray(per['pi_field'])
        self._rho_d = jnp.asarray(per['rho'])
        self._Sigma_d = jnp.asarray(Sigma_pad)
        self._beta_d = jnp.asarray(beta_pad)
        self._w_J_d = jnp.asarray(w_J_pad)
        self._w_pi_d = jnp.asarray(w_pi_pad)
        self._aa_a_d = jnp.asarray(aa_a_pad)
        self._aa_b_d = jnp.asarray(aa_b_pad)
        self._both_aa_d = jnp.asarray(both_aa_pad)

        # Route through the module-level shared @jit kernel so the compile
        # cache survives across MSAs and SVI iters (keyed on shape tuple,
        # not on closure identity).
        self._kernel = _get_shared_kernel()

    def score_batch(self, columns_list: 'list[np.ndarray]',
                     cls: np.ndarray) -> np.ndarray:
        """Score a list of B clusters (variable per-cluster size).

        Args:
          columns_list: list of column-index arrays/lists, one per candidate
            cluster. Each entry has length in [0, m_max].
          cls: (L_cols,) int array of per-column class labels.

        Returns: (B,) np.ndarray of log-likelihoods, one per request.

        Empty clusters (m=0) return 0.0 by convention.
        """
        B = len(columns_list)
        if B == 0:
            return np.zeros((0,), dtype=np.float64)
        if B > self.B_max:
            # Chunk via recursion -- each chunk fits in a bucket.
            out = np.empty(B, dtype=np.float64)
            for start in range(0, B, self.B_max):
                end = min(start + self.B_max, B)
                out[start:end] = self.score_batch(
                    columns_list[start:end], cls)
            return out

        # Round B up to the nearest sqrt(2)-spaced bucket so the JIT
        # kernel sees a bounded set of shapes (one compile per bucket).
        bucket = self._find_bucket(B)
        pad_cols, pad_mmask, pad_cls = self._get_buffers(bucket)
        pad_cols.fill(0)
        pad_mmask.fill(False)
        pad_cls.fill(0)
        for i, cols in enumerate(columns_list):
            cols_arr = np.asarray(cols, dtype=np.int64)
            m = cols_arr.shape[0]
            if m == 0:
                continue
            if m > self.m_max:
                raise ValueError(
                    f"cluster size {m} exceeds m_max={self.m_max}")
            pad_cols[i, :m] = cols_arr
            pad_mmask[i, :m] = True
            pad_cls[i, :m] = cls[cols_arr]

        jnp = self._jnp
        pc = jnp.asarray(pad_cols)
        mm = jnp.asarray(pad_mmask)
        cp = jnp.asarray(pad_cls)
        out = self._kernel(
            self._pi_field_d, self._rho_d, self._Sigma_d,
            self._beta_d, self._w_J_d, self._w_pi_d,
            self._aa_a_d, self._aa_b_d, self._both_aa_d,
            pc, mm, cp)
        return np.asarray(out)[:B]

    def score_batch_with_classes(self,
                                    columns_list: 'list[np.ndarray]',
                                    classes_list: 'list[np.ndarray]',
                                    ) -> np.ndarray:
        """Score B candidates where each has its OWN class vector.

        Unlike `score_batch` which derives per-candidate classes from a
        shared `cls[cols]` gather, this variant takes the class labels
        directly per candidate. Use case: class Gibbs sweep, where the K
        candidates for column s all share the same `columns` but differ
        in one entry of `classes`.

        Args:
          columns_list: list of B column-index arrays (each of variable
            length in [0, m_max]).
          classes_list: list of B class-label arrays; `len(classes_list[i])
            == len(columns_list[i])` for each i.

        Returns: (B,) np.ndarray of log-likelihoods.
        """
        B = len(columns_list)
        if B == 0:
            return np.zeros((0,), dtype=np.float64)
        assert len(classes_list) == B
        if B > self.B_max:
            out = np.empty(B, dtype=np.float64)
            for start in range(0, B, self.B_max):
                end = min(start + self.B_max, B)
                out[start:end] = self.score_batch_with_classes(
                    columns_list[start:end], classes_list[start:end])
            return out

        bucket = self._find_bucket(B)
        pad_cols, pad_mmask, pad_cls = self._get_buffers(bucket)
        pad_cols.fill(0)
        pad_mmask.fill(False)
        pad_cls.fill(0)
        for i, (cols, classes) in enumerate(zip(columns_list, classes_list)):
            cols_arr = np.asarray(cols, dtype=np.int64)
            cls_arr = np.asarray(classes, dtype=np.int64)
            m = cols_arr.shape[0]
            if m == 0:
                continue
            assert cls_arr.shape[0] == m, (
                f"columns_list[{i}] and classes_list[{i}] have different "
                f"sizes: {m} vs {cls_arr.shape[0]}")
            if m > self.m_max:
                raise ValueError(
                    f"cluster size {m} exceeds m_max={self.m_max}")
            pad_cols[i, :m] = cols_arr
            pad_mmask[i, :m] = True
            pad_cls[i, :m] = cls_arr

        jnp = self._jnp
        pc = jnp.asarray(pad_cols)
        mm = jnp.asarray(pad_mmask)
        cp = jnp.asarray(pad_cls)
        out = self._kernel(
            self._pi_field_d, self._rho_d, self._Sigma_d,
            self._beta_d, self._w_J_d, self._w_pi_d,
            self._aa_a_d, self._aa_b_d, self._both_aa_d,
            pc, mm, cp)
        return np.asarray(out)[:B]

    def _find_bucket(self, B: int) -> int:
        """Smallest geom-spaced bucket >= B."""
        import bisect
        i = bisect.bisect_left(self._buckets, B)
        if i >= len(self._buckets):
            return self._buckets[-1]
        return self._buckets[i]

    def _get_buffers(self, bucket: int):
        buf = self._buf_cache.get(bucket)
        if buf is None:
            buf = (np.zeros((bucket, self.m_max), dtype=np.int32),
                    np.zeros((bucket, self.m_max), dtype=bool),
                    np.zeros((bucket, self.m_max), dtype=np.int32))
            self._buf_cache[bucket] = buf
        return buf

    def score(self, columns, cls: np.ndarray) -> float:
        """Score a single cluster (scalar API).

        Routes through the batched kernel with B=1 -- useful for callers
        that haven't been refactored to collect batches. For hot loops,
        use `score_batch`.
        """
        return float(self.score_batch([columns], cls)[0])
