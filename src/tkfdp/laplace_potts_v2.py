"""JIT-hoisted Laplace MAP + diagonal Hessian for Potts atoms.

The earlier `laplace_potts.laplace_component_diag` re-traced + recompiled
the JAX pipeline on every call (once per class-pair × once per outer
iter), since `neg_log_post_fn` was a fresh Python closure each time and
the cherry-observation arrays had varying shapes per class-pair.

This v2 module exposes static-shape JIT'd primitives:

  loss_fn(H_flat, obs_packed, valid_mask, pi_classes, S, mu_prior,
            tau_prior, unique_t)
                   -> scalar neg-log-post
  grad_fn(H_flat, obs_packed, valid_mask, pi_classes, S, mu_prior,
            tau_prior, unique_t)
                   -> (d,) gradient
  hvp_fn(H_flat, v, obs_packed, valid_mask, pi_classes, S, mu_prior,
            tau_prior, unique_t)
                   -> (d,) Hessian-vector product

The caller pads each class-pair's observation array to a common M_max
(across the whole CRP-Gibbs sweep) and supplies a valid_mask so the
gather + sum is shape-static. JAX then JIT-compiles the loss / grad /
HVP exactly once per (M_max, K_c, n_t) shape signature; all subsequent
class-pairs and Adam steps reuse the cached graph.

For find_map_potts: an Adam loop over `grad_fn` gives the MAP. For
hessian_diag_at: 210 sequential HVPs (no vmap to keep memory bounded)
all hit the same cached `hvp_fn`. Across one CRP-Gibbs sweep the
expected speedup over `laplace_component_diag` is order-of-magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .generator import (
    A,
    build_joint_Q_pair,
    joint_stationary_pair,
    log_transition_matrices,
    symmetrize_eigh,
)
from .laplace_potts import (
    _flat_to_sym,
    _sym_to_flat,
    log_prior_pathwise,
)


# --- Core JIT'd primitives -------------------------------------------------

@partial(jax.jit, static_argnames=())
def loss_fn(H_flat, obs_packed, valid_mask, pi_classes, S, mu_prior,
              tau_prior, unique_t, h_a_table=None, h_b_table=None):
    """Static-shape neg-log-post — memory-efficient (no full log_P tensor).

    H_flat:       (d,) symmetric-slice flat parameter (d = A(A+1)/2 = 210).
    obs_packed:   (M_max, 4) int — [t_idx, c1*K_c+c2, start, end].
    valid_mask:   (M_max,) float — 1 where obs is real, 0 where padded.
    pi_classes:   (K_c, A) per-class stationary.
    S:            (A, A) F81 exchangeability.
    mu_prior:     (A, A) per-AA-pair Gaussian mean.
    tau_prior:    (A, A) per-AA-pair Gaussian precision.
    unique_t:     (n_t,) cherry distances (quantized).

    Memory: caches only per-(c1, c2) eigendecomposition of the joint Q
    (Lambda, U_sym, sqrt_pij — small: K_c² × A² × A² = 26 MB at K_c=8).
    Per-obs log P[a, b](t) is computed on-the-fly via two eigenvector-
    row gathers + one dot product — no materialization of the full
    (K_c, K_c, n_t, A², A²) log-P tensor (which would be 87 GB at
    K_c=8 with n_t=53).
    """
    H_mat = _flat_to_sym(H_flat)
    K_c = pi_classes.shape[0]
    if h_a_table is None:
        h_a_table = jnp.zeros((K_c, K_c, A))
    if h_b_table is None:
        h_b_table = jnp.zeros((K_c, K_c, A))

    def per_pair_eigh(pi1, pi2, h_a, h_b):
        Q = build_joint_Q_pair(H_mat, pi1, pi2, S=S, h_a=h_a, h_b=h_b)
        pi_j = joint_stationary_pair(H_mat, pi1, pi2, h_a=h_a, h_b=h_b)
        Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
        return Lambda, U_sym, sqrt_pij

    # 2D vmap over (c1, c2) with h_a_table/h_b_table indexed by both dims.
    Lambdas, U_syms, sqrt_pijs = jax.vmap(jax.vmap(
        per_pair_eigh, in_axes=(None, 0, 0, 0)),
        in_axes=(0, None, 0, 0))(pi_classes, pi_classes,
                                       h_a_table, h_b_table)

    t_idx = obs_packed[:, 0]
    cp_ord = obs_packed[:, 1]
    c1 = cp_ord // K_c
    c2 = cp_ord % K_c
    start = obs_packed[:, 2]; end = obs_packed[:, 3]

    # Per-obs gather (only the rows we need from U_sym).
    L_obs = Lambdas[c1, c2]                                  # (M_max, A²)
    Ua = U_syms[c1, c2, start]                               # (M_max, A²)
    Ub = U_syms[c1, c2, end]                                 # (M_max, A²)
    inv_a = 1.0 / sqrt_pijs[c1, c2, start]                   # (M_max,)
    sb = sqrt_pijs[c1, c2, end]                              # (M_max,)
    tau_obs = unique_t[t_idx]                                # (M_max,)
    expL = jnp.exp(L_obs * tau_obs[:, None])                 # (M_max, A²)
    P_obs = inv_a * jnp.sum(Ua * expL * Ub, axis=1) * sb     # (M_max,)
    log_p_obs = jnp.log(jnp.clip(P_obs, 1e-300, 1.0))

    log_pr = log_prior_pathwise(H_mat, mu_prior, tau_prior)
    return -jnp.sum(log_p_obs * valid_mask) - log_pr


grad_fn = jax.jit(jax.grad(loss_fn))


@partial(jax.jit, static_argnames=())
def loss_fn_with_h(H_flat, h_pairs, obs_packed, valid_mask, pi_classes, S,
                     mu_prior, tau_prior, unique_t, cp_idx, cp_swap,
                     is_diag_pair, h_prior_tau, h_share):
    """loss_fn extended with h_pairs as a co-optimized parameter.

    h_pairs: (K_c(K_c+1)/2, 2, A) per-class-pair side potentials.
    cp_idx, cp_swap: (K_c, K_c) lookup tables (canonical-pair index +
        swap flag for ordered (c1, c2)).
    is_diag_pair: (K_c(K_c+1)/2,) bool, True at self-pair (c, c) indices.
        For self-pairs the two sites are exchangeable, so we tie
        h_pairs[i, 1, :] := h_pairs[i, 0, :] (keeping the joint pair
        distribution symmetric and joint Q reversible). The redundant
        slot-1 entries on diagonals also receive zero prior contribution
        so the prior counts each diagonal h-vector exactly once.
    h_prior_tau: scalar Gaussian-prior precision on h_pairs (centered at 0).
    h_share: scalar in (0, 1] — fraction of the total Gaussian prior
        attributed to THIS atom's loss. With per-atom Adam, callers pass
        1/K_H so summing across all K_H atom calls gives the full prior.
    """
    # Tie h_pairs[diag, 1, :] = h_pairs[diag, 0, :] for self-pairs (c, c).
    slot0 = h_pairs[:, 0, :]
    slot1 = h_pairs[:, 1, :]
    slot1_eff = jnp.where(is_diag_pair[:, None], slot0, slot1)
    h_pairs_eff = jnp.stack([slot0, slot1_eff], axis=1)

    h_a_table = h_pairs_eff[cp_idx, cp_swap]              # (K_c, K_c, A)
    h_b_table = h_pairs_eff[cp_idx, 1 - cp_swap]
    base_loss = loss_fn(
        H_flat, obs_packed, valid_mask, pi_classes, S, mu_prior, tau_prior,
        unique_t, h_a_table=h_a_table, h_b_table=h_b_table,
    )
    # Prior: count slot 0 always, slot 1 only on off-diagonals (slot 1 is
    # redundant on diagonals after tying).
    prior_slot0 = jnp.sum(slot0 ** 2)
    prior_slot1 = jnp.sum(jnp.where(is_diag_pair[:, None], 0.0, slot1 ** 2))
    prior_h = 0.5 * h_prior_tau * h_share * (prior_slot0 + prior_slot1)
    return base_loss + prior_h


grad_fn_with_h = jax.jit(jax.grad(loss_fn_with_h, argnums=(0, 1)))


@partial(jax.jit, static_argnames=())
def hvp_fn(H_flat, v, obs_packed, valid_mask, pi_classes, S, mu_prior,
             tau_prior, unique_t):
    """Hessian-vector product = ∂grad/∂H_flat · v at H_flat."""
    g = lambda H: grad_fn(H, obs_packed, valid_mask, pi_classes, S,
                            mu_prior, tau_prior, unique_t)
    _, hv = jax.jvp(g, (H_flat,), (v,))
    return hv


# --- Padding helper --------------------------------------------------------

def pad_obs(obs: np.ndarray, M_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad a (M, 4) obs array to (M_max, 4) with dummy zeros and return a
    (M_max,) float valid_mask (1 = real, 0 = padded)."""
    M = obs.shape[0]
    if M == 0:
        return np.zeros((M_max, 4), dtype=np.int64), np.zeros(M_max)
    obs_p = np.zeros((M_max, 4), dtype=np.int64)
    obs_p[:M] = obs
    mask = np.zeros(M_max, dtype=np.float64)
    mask[:M] = 1.0
    return obs_p, mask


# --- High-level: MAP + diagonal-Hessian in one call ------------------------

@dataclass
class LaplaceComponentV2:
    H_hat: np.ndarray
    log_lik_at_hat: float
    log_prior_at_hat: float
    log_det_post_prec: float
    d: int


def laplace_component_diag_jit(obs_packed: np.ndarray, valid_mask: np.ndarray,
                                  pi_classes: np.ndarray, S: np.ndarray,
                                  mu_prior: np.ndarray, tau_prior: np.ndarray,
                                  unique_t: np.ndarray,
                                  H_init: np.ndarray,
                                  n_steps: int = 30, lr: float = 0.05
                                  ) -> LaplaceComponentV2:
    """JIT-hoisted Laplace component:
    1. Adam MAP using the cached `grad_fn` (eigh + transition matrices
       recomputed each step since H is changing).
    2. Diagonal Hessian via `jax.linearize` at H_hat — the eigh + expm +
       log_P pipeline is evaluated ONCE at H_hat, then 210 HVPs are
       cheap linear evaluations on the cached linearization.

    Step 2 is the user-suggested key optimization: at the converged H_hat,
    the eigh / expm at the MAP doesn't change across HVPs, so the JVP
    machinery should reuse the linearization rather than re-traversing
    the whole forward pass each time.
    """
    obs_j = jnp.asarray(obs_packed)
    mask_j = jnp.asarray(valid_mask, dtype=jnp.float64)
    pi_j = jnp.asarray(pi_classes)
    S_j = jnp.asarray(S)
    mu_j = jnp.asarray(mu_prior); tau_j = jnp.asarray(tau_prior)
    t_j = jnp.asarray(unique_t)

    # 1. Adam MAP (eigh recomputed each iter since H changes)
    H_flat = jnp.asarray(_sym_to_flat(jnp.asarray(H_init)))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(H_flat)
    for _ in range(n_steps):
        g = grad_fn(H_flat, obs_j, mask_j, pi_j, S_j, mu_j, tau_j, t_j)
        updates, opt_state = optimizer.update(g, opt_state)
        H_flat = optax.apply_updates(H_flat, updates)

    H_hat = np.asarray(_flat_to_sym(H_flat))

    # 2. Diagonal Hessian via linearize-at-H_hat — the eigh + expm pipeline
    # at H_hat is computed ONCE (folded into the linearization), then each
    # HVP is just a linear application of the cached JVP.
    grad_at_obs = lambda H: grad_fn(H, obs_j, mask_j, pi_j, S_j, mu_j, tau_j, t_j)
    _, hvp_fn_at_H_hat = jax.linearize(grad_at_obs, H_flat)
    hvp_jit = jax.jit(hvp_fn_at_H_hat)

    d = H_flat.shape[0]
    diags = np.zeros(d)
    eye_np = np.eye(d)
    for i in range(d):
        hv = hvp_jit(jnp.asarray(eye_np[i]))
        diags[i] = float(hv[i])
    # NaN-safe: float32 prior-dominated directions can produce non-positive
    # or NaN Hessian-diagonal entries. Floor + NaN-replace before log.
    diags = np.where(np.isnan(diags), 1e-6, diags)
    post_prec_diag = np.maximum(diags, 1e-6)
    log_det = float(np.sum(np.log(post_prec_diag)))

    # log lik / log prior at H_hat
    nlp = float(loss_fn(H_flat, obs_j, mask_j, pi_j, S_j, mu_j, tau_j, t_j))
    H_mat_j = jnp.asarray(H_hat)
    log_pr = float(log_prior_pathwise(H_mat_j, mu_j, tau_j))
    log_lik = -nlp - log_pr

    return LaplaceComponentV2(
        H_hat=H_hat, log_lik_at_hat=log_lik, log_prior_at_hat=log_pr,
        log_det_post_prec=log_det, d=d,
    )


def laplace_log_evidence_v2(comp: LaplaceComponentV2) -> float:
    """log p(data) ≈ log L(data | H_hat) + log G_0(H_hat)
                      + (d/2) log(2π) - 0.5 log det(post_prec)."""
    return (comp.log_lik_at_hat + comp.log_prior_at_hat
            + 0.5 * comp.d * np.log(2 * np.pi)
            - 0.5 * comp.log_det_post_prec)


# --- Existing-atom log-pair-likelihood (also JIT'd) ------------------------

@partial(jax.jit, static_argnames=())
def existing_atom_log_lik(H_atom, obs_packed, valid_mask, pi_classes, S,
                            unique_t, h_a_table=None, h_b_table=None):
    """Sum over (cherry, edge) of log P[(a_s, a_t), (b_s, b_t)](τ; H_atom)
    for the supplied padded obs. Used in CRP-Gibbs as the score for an
    existing atom. JIT'd once per shape signature.

    h_a_table, h_b_table: optional (K_c, K_c, A) per-class-pair side
    potentials. None or all-zeros disables side potentials.
    """
    K_c = pi_classes.shape[0]
    if h_a_table is None:
        h_a_table = jnp.zeros((K_c, K_c, A))
    if h_b_table is None:
        h_b_table = jnp.zeros((K_c, K_c, A))

    def per_pair(pi1, pi2, h_a, h_b):
        Q = build_joint_Q_pair(H_atom, pi1, pi2, S=S, h_a=h_a, h_b=h_b)
        pi_j = joint_stationary_pair(H_atom, pi1, pi2, h_a=h_a, h_b=h_b)
        Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
        return log_transition_matrices(unique_t, Lambda, U_sym, sqrt_pij)
    log_P = jax.vmap(jax.vmap(per_pair, in_axes=(None, 0, 0, 0)),
                       in_axes=(0, None, 0, 0))(
        pi_classes, pi_classes, h_a_table, h_b_table)
    t_idx = obs_packed[:, 0]; cp_ord = obs_packed[:, 1]
    c1 = cp_ord // K_c; c2 = cp_ord % K_c
    start = obs_packed[:, 2]; end = obs_packed[:, 3]
    log_p_obs = log_P[c1, c2, t_idx, start, end]
    return jnp.sum(log_p_obs * valid_mask)
