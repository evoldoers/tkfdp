"""ELBO-based loss for the SVI v2 pipeline.

Replaces the exact `loss_fn` (which builds the joint 400×400 Q and
gathers exp(Q*τ)[a, b]) with the strict variational lower bound from
the Holmes-Rubin closed-form construction.

At constant variational rate Q_hat (geometric-mean construction at the
time-averaged 1-site marginal bar_p) the ELBO is closed-form, exact at
H = 0, and always ≤ log P_exact by Jensen.

This module supports class-specific stationaries pi_1, pi_2 (so it
plumbs through the post-2026-05-08 F81 parameterization where each
class carries its own pi^(c) updated via secret-destination Dirichlet
conjugacy). The unconditional baseline rate is therefore
    Q_unc^s(x, x') = S[x, x'] * pi_s(x')    (off-diagonal)
and the geometric-mean variational rate is
    Q_hat^s(x, x') = Q_unc^s(x, x') * exp(-0.5 * (H_eff[x'] - H_eff[x]))
with H_eff = H @ bar_p_other.

The inner (bar_p_1, bar_p_2) fixed point is unrolled as `jax.lax.scan`
with a fixed N_FP_DEFAULT damped iterations so the scheme remains
JIT-traceable and end-to-end differentiable in H.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .laplace_potts import _flat_to_sym, _sym_to_flat, log_prior_pathwise
from .lg08 import PI_LG08, S_LG08_F81


A = 20
EPS = 1e-30
JITTER = 1e-8

N_FP_DEFAULT = 12
DAMPING_DEFAULT = 0.5


# -- Building blocks (JAX-traceable) -----------------------------------------

def _build_Q_unc(pi, S):
    """Unconditional F81 generator at stationary pi.
       Q_unc[x, x'] = S[x, x'] * pi(x') for x != x', diag = -row sums.
    """
    S_off = S - jnp.diag(jnp.diag(S))
    Q_off = S_off * pi[None, :]
    return Q_off - jnp.diag(Q_off.sum(axis=-1))


def _build_Q_hat(H, pi, S, p_other):
    """Geometric-mean variational rate at fixed (constant) p_other,
    base = F81 generator at pi.
    """
    H_eff = H @ p_other
    factor = jnp.exp(-0.5 * (H_eff[None, :] - H_eff[:, None]))
    Q_unc = _build_Q_unc(pi, S)
    Q_off = jnp.where(jnp.eye(A, dtype=bool), 0.0,
                       (Q_unc - jnp.diag(jnp.diag(Q_unc))) * factor)
    return Q_off - jnp.diag(Q_off.sum(axis=-1))


def _eigh_via_symmetrization(Q, pi):
    sqrt = jnp.sqrt(pi)
    inv = 1.0 / jnp.clip(sqrt, EPS, None)
    Q_sym = sqrt[:, None] * Q * inv[None, :]
    Q_sym = 0.5 * (Q_sym + Q_sym.T) + JITTER * jnp.eye(A)
    Lambda, U_sym = jnp.linalg.eigh(Q_sym)
    Lambda = Lambda - JITTER
    U = inv[:, None] * U_sym
    Vinv = U_sym.T * sqrt[None, :]
    return Lambda, U, Vinv


def _hr_pair(alpha, beta, t):
    diff = alpha - beta
    safe_diff = jnp.where(jnp.abs(diff) < 1e-12, 1.0, diff)
    off = (jnp.exp(alpha * t) - jnp.exp(beta * t)) / safe_diff
    on = t * jnp.exp(alpha * t)
    return jnp.where(jnp.abs(diff) < 1e-12, on, off)


def _stationary_from_H(H, pi_base, p_other):
    """pi_hat ∝ pi_base * exp(-H @ p_other)."""
    log_pi = jnp.log(pi_base) - H @ p_other
    return jax.nn.softmax(log_pi)


# -- Core: closed-form ELBO at given Q_hat ----------------------------------

def _elbo_at_Qhat(H, pi_1, pi_2, S,
                    x_a_1, x_a_2, x_b_1, x_b_2, t,
                    Q_hat_1, Q_hat_2, pi_hat_1, pi_hat_2):
    """Closed-form strict ELBO at given (constant) Q_hat^s.

    Q_unc^s = F81 generator at pi_s (the per-class stationary) — used
    in the KL terms as the unconditional baseline.
    """
    Lambda1, U1, V1inv = _eigh_via_symmetrization(Q_hat_1, pi_hat_1)
    Lambda2, U2, V2inv = _eigh_via_symmetrization(Q_hat_2, pi_hat_2)

    expL1_t = jnp.exp(Lambda1 * t)
    expL2_t = jnp.exp(Lambda2 * t)
    P1 = jnp.dot(U1[x_a_1, :] * expL1_t, V1inv[:, x_b_1])
    P2 = jnp.dot(U2[x_a_2, :] * expL2_t, V2inv[:, x_b_2])
    log_P1 = jnp.log(jnp.clip(P1, EPS, None))
    log_P2 = jnp.log(jnp.clip(P2, EPS, None))

    alpha_b = (Lambda1[:, None] + Lambda2[None, :])[:, None, :, None]
    beta_b = (Lambda1[:, None] + Lambda2[None, :])[None, :, None, :]
    M = _hr_pair(alpha_b * jnp.ones((1, A, 1, A)),
                  beta_b * jnp.ones((A, 1, A, 1)), t)

    # Site-1 spectral basis factors (used by S1_J, S1_K, I_gamma_1):
    #   a1[x, k] = U1[xa1, k] * V1inv[k, x]            (k = source eigenvector)
    #   b1[x, l] = U1[x, l] * V1inv[l, xb1]            (l = sink eigenvector)
    # Then S1_J[x, k, l] = a1[x, k] * b1[x, l] (DIAGONAL on x — same x in both factors)
    # while S1_K[x, y, k, l] = a1[x, k] * b1[y, l] (outer product on x, y).
    a1 = U1[x_a_1, :][None, :] * V1inv.T                # (x, k)  with V1inv[k, x] = V1inv.T[x, k]
    b1 = U1 * V1inv[:, x_b_1][None, :]                  # (x, l)
    a2 = U2[x_a_2, :][None, :] * V2inv.T                # (x, k)
    b2 = U2 * V2inv[:, x_b_2][None, :]                  # (x, l)
    S1_J = a1[:, :, None] * b1[:, None, :]              # (x_1, k, l) — diag on x
    S2_J = a2[:, :, None] * b2[:, None, :]              # (x_2, k, l) — diag on x

    T1_J = jnp.einsum('klmn,xmn->xkl', M, S2_J)         # (x_2, k_1, l_1)
    J = jnp.einsum('xkl,ykl->xy', T1_J, S1_J) / (P1 * P2)
    J = J.T  # (x_1, x_2)

    # K[x, y, z] = sum_{k, l} S1_K[x, y, k, l] * T1_J[z, k, l] * (Q_hat_1[x, y] / (P1 P2))
    # With S1_K = a1[x, k] * b1[y, l]:
    #   tmp1[x, z, l] = sum_k a1[x, k] * T1_J[z, k, l]   (A, A, A) — no A^5 intermediate
    #   K[x, y, z]    = sum_l b1[y, l] * tmp1[x, z, l]   (A, A, A)
    tmp1 = jnp.einsum('xk,zkl->xzl', a1, T1_J)
    K = jnp.einsum('yl,xzl->xyz', b1, tmp1) * (Q_hat_1 / (P1 * P2))[:, :, None]

    T2_J = jnp.einsum('klmn,xkl->xmn', M, S1_J)         # (x_1, k_2, l_2)
    tmp2 = jnp.einsum('xk,zkl->xzl', a2, T2_J)
    K2 = jnp.einsum('yl,xzl->xyz', b2, tmp2) * (Q_hat_2 / (P1 * P2))[:, :, None]

    # I_gamma[x, y] = sum_{k, l} a[x, k] * b[y, l] * HR(L_k, L_l, t) * Q_hat[x, y] / P
    HR_1 = _hr_pair(Lambda1[:, None] * jnp.ones((1, A)),
                     Lambda1[None, :] * jnp.ones((A, 1)), t)
    # einsum 'xk,yl,kl->xy': contract first over k, then over l (A^3 max intermediate).
    tmp_g1 = jnp.einsum('xk,kl->xl', a1, HR_1)          # (x, l)
    I_gamma_1 = (Q_hat_1 / P1) * jnp.einsum('xl,yl->xy', tmp_g1, b1)
    HR_2 = _hr_pair(Lambda2[:, None] * jnp.ones((1, A)),
                     Lambda2[None, :] * jnp.ones((A, 1)), t)
    tmp_g2 = jnp.einsum('xk,kl->xl', a2, HR_2)
    I_gamma_2 = (Q_hat_2 / P2) * jnp.einsum('xl,yl->xy', tmp_g2, b2)

    bar_p_1 = J.sum(axis=1) / t
    bar_p_2 = J.sum(axis=0) / t

    mask_off = ~jnp.eye(A, dtype=bool)
    Q_unc_1 = _build_Q_unc(pi_1, S)
    Q_unc_2 = _build_Q_unc(pi_2, S)
    Q_unc_1_off = Q_unc_1 - jnp.diag(jnp.diag(Q_unc_1))
    Q_unc_2_off = Q_unc_2 - jnp.diag(jnp.diag(Q_unc_2))

    # Site 1 KL terms
    dH_1 = H[None, :, :] - H[:, None, :]                              # (x_1, y_1, x_2)
    AM_1 = Q_unc_1_off[:, :, None] * jnp.exp(-0.5 * dH_1)
    KL_diag_1 = jnp.einsum('xz,xyz->', J, AM_1) \
                - t * jnp.einsum('x,xy->', bar_p_1, Q_hat_1 * mask_off)
    log_G_1 = jnp.where(mask_off,
                          jnp.log(jnp.clip(Q_hat_1 / jnp.clip(Q_unc_1_off, EPS, None), EPS, None)),
                          0.0)
    KL_jump_1 = 0.5 * jnp.einsum('xyz,xyz->', K, dH_1) \
                + jnp.einsum('xy,xy->', log_G_1, I_gamma_1 * mask_off)

    # Site 2 KL terms
    H_y2_x1 = jnp.broadcast_to(H[None, :, :], (A, A, A))
    H_x2_x1 = H[:, None, :]
    dH_2 = H_y2_x1 - H_x2_x1
    AM_2 = Q_unc_2_off[:, :, None] * jnp.exp(-0.5 * dH_2)
    KL_diag_2 = jnp.einsum('xz,xyz->', J.T, AM_2) \
                - t * jnp.einsum('y,yx->', bar_p_2, Q_hat_2 * mask_off)
    log_G_2 = jnp.where(mask_off,
                          jnp.log(jnp.clip(Q_hat_2 / jnp.clip(Q_unc_2_off, EPS, None), EPS, None)),
                          0.0)
    KL_jump_2 = 0.5 * jnp.einsum('xyz,xyz->', K2, dH_2) \
                + jnp.einsum('xy,xy->', log_G_2, I_gamma_2 * mask_off)

    KL_total = KL_diag_1 + KL_jump_1 + KL_diag_2 + KL_jump_2
    elbo = log_P1 + log_P2 - KL_total
    return elbo, bar_p_1, bar_p_2


def _bar_p_step(carry, _, H, pi_1, pi_2, S, x_a_1, x_a_2, x_b_1, x_b_2, t,
                  damping):
    """One damped fixed-point iteration on (bar_p_1, bar_p_2).

    Wrapped in jax.checkpoint at the scan-body level: the reverse pass
    recomputes the O(A^5) einsum chain rather than storing it for all
    N_FP iterations. Without this, the backward pass through
    `jax.vmap(_elbo_traceable)` over M ~ 200 cherries materializes
    O(N_FP × M × A^5) ~ 100 GB of intermediates and OOMs the GPU.
    """
    bar_p_1, bar_p_2 = carry
    Q_hat_1 = _build_Q_hat(H, pi_1, S, bar_p_2)
    Q_hat_2 = _build_Q_hat(H, pi_2, S, bar_p_1)
    pi_hat_1 = _stationary_from_H(H, pi_1, bar_p_2)
    pi_hat_2 = _stationary_from_H(H, pi_2, bar_p_1)
    _, new_bp_1, new_bp_2 = _elbo_at_Qhat(
        H, pi_1, pi_2, S,
        x_a_1, x_a_2, x_b_1, x_b_2, t,
        Q_hat_1, Q_hat_2, pi_hat_1, pi_hat_2,
    )
    new_bp_1 = jnp.clip(new_bp_1, 1e-12, None); new_bp_1 = new_bp_1 / new_bp_1.sum()
    new_bp_2 = jnp.clip(new_bp_2, 1e-12, None); new_bp_2 = new_bp_2 / new_bp_2.sum()
    new_bp_1 = damping * new_bp_1 + (1 - damping) * bar_p_1
    new_bp_2 = damping * new_bp_2 + (1 - damping) * bar_p_2
    return (new_bp_1, new_bp_2), None


def _elbo_traceable(H, pi_1, pi_2, S,
                      x_a_1, x_a_2, x_b_1, x_b_2, t,
                      n_fp=N_FP_DEFAULT, damping=DAMPING_DEFAULT):
    """Strict ELBO at the converged (bar_p_1, bar_p_2) fixed point.

    H        — (A, A) Potts atom.
    pi_1, pi_2 — per-site (per-class) stationaries.
    S        — F81 exchangeability.

    The inner scan body is gradient-checkpointed so the reverse pass
    recomputes per-step intermediates instead of storing them across
    all N_FP iterations and across the outer vmap-over-M-cherries.
    """
    init = (pi_1, pi_2)
    step_fn = partial(_bar_p_step,
                        H=H, pi_1=pi_1, pi_2=pi_2, S=S,
                        x_a_1=x_a_1, x_a_2=x_a_2,
                        x_b_1=x_b_1, x_b_2=x_b_2,
                        t=t, damping=damping)
    step_fn_ckpt = jax.checkpoint(step_fn)
    (bar_p_1, bar_p_2), _ = jax.lax.scan(step_fn_ckpt, init, None, length=n_fp)
    Q_hat_1 = _build_Q_hat(H, pi_1, S, bar_p_2)
    Q_hat_2 = _build_Q_hat(H, pi_2, S, bar_p_1)
    pi_hat_1 = _stationary_from_H(H, pi_1, bar_p_2)
    pi_hat_2 = _stationary_from_H(H, pi_2, bar_p_1)
    elbo, _, _ = _elbo_at_Qhat(
        H, pi_1, pi_2, S,
        x_a_1, x_a_2, x_b_1, x_b_2, t,
        Q_hat_1, Q_hat_2, pi_hat_1, pi_hat_2,
    )
    return elbo


# -- Loss matching laplace_potts_v2.loss_fn signature -----------------------

@partial(jax.jit, static_argnames=())
def loss_fn_elbo(H_flat, obs_packed, valid_mask, pi_classes, S, mu_prior,
                   tau_prior, unique_t):
    """Strict ELBO loss matching the laplace_potts_v2.loss_fn signature.

    obs_packed[:, 0] = t_idx into unique_t
    obs_packed[:, 1] = c1 * K_c + c2  (class indices)
    obs_packed[:, 2] = a_s * 20 + a_t (start state)
    obs_packed[:, 3] = b_s * 20 + b_t (end state)

    For each cherry, runs the inner bar_p fixed point, then evaluates
    the strict ELBO at the converged Q_hat.
    """
    H_mat = _flat_to_sym(H_flat)
    K_c = pi_classes.shape[0]

    t_idx = obs_packed[:, 0]
    cp_ord = obs_packed[:, 1]
    c1 = cp_ord // K_c
    c2 = cp_ord % K_c
    start = obs_packed[:, 2]; end = obs_packed[:, 3]
    x_a_1 = start // 20; x_a_2 = start % 20
    x_b_1 = end // 20; x_b_2 = end % 20
    tau_obs = unique_t[t_idx]
    pi_1_arr = pi_classes[c1]                                     # (M, A)
    pi_2_arr = pi_classes[c2]                                     # (M, A)

    elbo_per = jax.vmap(
        _elbo_traceable,
        in_axes=(None, 0, 0, None, 0, 0, 0, 0, 0, None, None),
    )(H_mat, pi_1_arr, pi_2_arr, S,
       x_a_1, x_a_2, x_b_1, x_b_2, tau_obs,
       N_FP_DEFAULT, DAMPING_DEFAULT)

    log_pr = log_prior_pathwise(H_mat, mu_prior, tau_prior)
    return -jnp.sum(elbo_per * valid_mask) - log_pr


grad_fn_elbo = jax.jit(jax.grad(loss_fn_elbo))


# -- Chunked-M ELBO: per-chunk loss / grad / HVP without stored residuals --
#
# loss_fn_elbo + grad_fn_elbo above hold the entire (M, ...) computation in
# one JIT graph. For the diagonal Hessian via jax.linearize, that puts
# O(M × N_FP × A^4) forward residuals into linearize's captured constants
# (~12 GB at M=100, OOM at M=1000). The chunked path below splits M into
# fixed-size pieces and runs a Python loop:
#
#   * `_neg_sum_elbo_chunk` is the ELBO sum over one chunk of cherries
#     (no prior — added once outside the chunk loop).
#   * `_grad_neg_sum_elbo_chunk` is its JIT'd gradient.
#   * `_hvp_neg_sum_elbo_chunk` computes hvp_chunk(H, v) via
#       jax.jvp(grad_chunk_at_v)(H, v)
#     — no linearize, no stored residuals between successive (v) calls.
#
# Each call pays one extra forward pass through the chunk vs the
# linearize-cached path, but peak memory is bounded by chunk_size, which
# keeps the K_c > 1 ELBO Laplace step in budget. JIT cache hits across
# chunks of the same shape, so compile cost is paid once.

@partial(jax.jit, static_argnames=())
def _neg_sum_elbo_chunk(H_flat, obs_chunk, mask_chunk, pi_classes, S, unique_t):
    """Negative sum of ELBO over a chunk of cherries (no prior).

    Same per-cherry computation as `loss_fn_elbo` but no prior term, so
    the sum across chunks plus a single prior gradient gives the full
    posterior gradient.
    """
    H_mat = _flat_to_sym(H_flat)
    K_c = pi_classes.shape[0]
    t_idx = obs_chunk[:, 0]; cp_ord = obs_chunk[:, 1]
    c1 = cp_ord // K_c; c2 = cp_ord % K_c
    start = obs_chunk[:, 2]; end = obs_chunk[:, 3]
    x_a_1 = start // 20; x_a_2 = start % 20
    x_b_1 = end // 20; x_b_2 = end % 20
    tau_obs = unique_t[t_idx]
    pi_1_arr = pi_classes[c1]; pi_2_arr = pi_classes[c2]
    elbo_per = jax.vmap(
        _elbo_traceable,
        in_axes=(None, 0, 0, None, 0, 0, 0, 0, 0, None, None),
    )(H_mat, pi_1_arr, pi_2_arr, S,
       x_a_1, x_a_2, x_b_1, x_b_2, tau_obs,
       N_FP_DEFAULT, DAMPING_DEFAULT)
    return -jnp.sum(elbo_per * mask_chunk)


_grad_neg_sum_elbo_chunk = jax.jit(jax.grad(_neg_sum_elbo_chunk))


@partial(jax.jit, static_argnames=())
def _hvp_neg_sum_elbo_chunk(H_flat, v, obs_chunk, mask_chunk, pi_classes, S,
                              unique_t):
    """Hessian-vector product of the chunk neg-loss at H_flat in direction v.

    Computes via forward-mode JVP applied to the gradient: jax.jvp(grad)(v)
    returns (grad, hvp). No persistent residuals across (v) calls — every
    call re-runs forward+backward through the chunk. Memory peak per call
    is bounded by chunk_size × N_FP × per-step intermediates.
    """
    grad_at = lambda H_: _grad_neg_sum_elbo_chunk(
        H_, obs_chunk, mask_chunk, pi_classes, S, unique_t
    )
    _, hvp = jax.jvp(grad_at, (H_flat,), (v,))
    return hvp


def _split_chunks(obs_packed: np.ndarray, valid_mask: np.ndarray,
                    chunk_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pad-and-split obs into fixed-size chunks. Last chunk gets zero-mask
    padding to a multiple of chunk_size so JIT shape stays static.
    """
    M = obs_packed.shape[0]
    n_full = M // chunk_size
    rem = M - n_full * chunk_size
    chunks = []
    for i in range(n_full):
        s = i * chunk_size; e = s + chunk_size
        chunks.append((obs_packed[s:e], valid_mask[s:e]))
    if rem > 0:
        last_obs = np.zeros((chunk_size, obs_packed.shape[1]),
                              dtype=obs_packed.dtype)
        last_mask = np.zeros(chunk_size, dtype=valid_mask.dtype)
        last_obs[:rem] = obs_packed[n_full * chunk_size:]
        last_mask[:rem] = valid_mask[n_full * chunk_size:]
        chunks.append((last_obs, last_mask))
    return chunks


def laplace_component_diag_jit_elbo_chunked(
        obs_packed: np.ndarray, valid_mask: np.ndarray,
        pi_classes: np.ndarray, S: np.ndarray,
        mu_prior: np.ndarray, tau_prior: np.ndarray,
        unique_t: np.ndarray,
        H_init: np.ndarray,
        n_steps: int = 30, lr: float = 0.05,
        chunk_size: int = 32,
) -> "LaplaceComponentELBO":
    """Memory-bounded version of `laplace_component_diag_jit_elbo`.

    Splits the obs into chunks of `chunk_size` cherries; Adam MAP and the
    diagonal Hessian both use Python loops over chunks that call jit'd
    per-chunk grad / HVP functions. Peak memory per JIT call is bounded
    by `chunk_size × N_FP × A^4`; safe for M up to thousands of cherries.

    Each diagonal-Hessian basis vector costs N_chunks × HVP_per_chunk.
    For d=210 and N_chunks=32 (M=1024, chunk=32) this is ~6700 HVP-chunk
    JIT'd calls per Laplace component — slower than the linearize-cached
    path but bounded in memory.
    """
    chunks = _split_chunks(np.asarray(obs_packed),
                              np.asarray(valid_mask).astype(np.float64),
                              chunk_size)
    chunks_j = [(jnp.asarray(o), jnp.asarray(m)) for o, m in chunks]
    pi_j = jnp.asarray(pi_classes)
    S_j = jnp.asarray(S)
    mu_j = jnp.asarray(mu_prior); tau_j = jnp.asarray(tau_prior)
    t_j = jnp.asarray(unique_t)

    H_flat = jnp.asarray(_sym_to_flat(jnp.asarray(H_init)))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(H_flat)

    grad_log_prior_jit = jax.jit(jax.grad(
        lambda H_: -log_prior_pathwise(_flat_to_sym(H_), mu_j, tau_j)
    ))
    hvp_log_prior_jit = jax.jit(lambda H_, v: jax.jvp(
        grad_log_prior_jit, (H_,), (v,))[1])

    for _ in range(n_steps):
        g = grad_log_prior_jit(H_flat)
        for obs_c, mask_c in chunks_j:
            g = g + _grad_neg_sum_elbo_chunk(H_flat, obs_c, mask_c, pi_j,
                                                 S_j, t_j)
        updates, opt_state = optimizer.update(g, opt_state)
        H_flat = optax.apply_updates(H_flat, updates)

    H_hat = np.asarray(_flat_to_sym(H_flat))

    # Diagonal Hessian via per-chunk JVP — no stored residuals between calls.
    d = H_flat.shape[0]
    diags = np.zeros(d)
    eye_np = np.eye(d)
    for i in range(d):
        v = jnp.asarray(eye_np[i])
        hv = hvp_log_prior_jit(H_flat, v)
        for obs_c, mask_c in chunks_j:
            hv = hv + _hvp_neg_sum_elbo_chunk(
                H_flat, v, obs_c, mask_c, pi_j, S_j, t_j
            )
        diags[i] = float(hv[i])
    # Float32 paths can produce NaN or non-positive Hessian-diagonal entries
    # (most commonly from the prior-dominated weak-curvature directions).
    # Floor + NaN-replace before log so log_det stays finite.
    diags = np.where(np.isnan(diags), 1e-6, diags)
    post_prec_diag = np.maximum(diags, 1e-6)
    log_det = float(np.sum(np.log(post_prec_diag)))

    # log L (sum-ELBO) at H_hat — chunked sum.
    nlp = 0.0
    for obs_c, mask_c in chunks_j:
        nlp += float(_neg_sum_elbo_chunk(H_flat, obs_c, mask_c, pi_j, S_j, t_j))
    log_pr = float(log_prior_pathwise(jnp.asarray(H_hat), mu_j, tau_j))
    log_lik = -nlp                # NB: nlp here excludes the prior

    return LaplaceComponentELBO(
        H_hat=H_hat, log_lik_at_hat=log_lik, log_prior_at_hat=log_pr,
        log_det_post_prec=log_det, d=d,
    )


# -- CRP-Gibbs side: existing-atom log-lik via ELBO (no prior) -------------

@partial(jax.jit, static_argnames=())
def existing_atom_log_lik_elbo(H_atom, obs_packed, valid_mask, pi_classes, S,
                                  unique_t):
    """Sum over (cherry, edge) of strict ELBO[(a_s, a_t), (b_s, b_t)](τ;
    H_atom, pi_classes[c_s], pi_classes[c_t], S) for the supplied padded
    obs. Matches `laplace_potts_v2.existing_atom_log_lik` signature.

    Returns the sum of ELBO over valid cherries; this is a lower bound on
    the exact pair-loglik used by the CRP-Gibbs scoring path.
    """
    K_c = pi_classes.shape[0]
    t_idx = obs_packed[:, 0]; cp_ord = obs_packed[:, 1]
    c1 = cp_ord // K_c; c2 = cp_ord % K_c
    start = obs_packed[:, 2]; end = obs_packed[:, 3]
    x_a_1 = start // 20; x_a_2 = start % 20
    x_b_1 = end // 20; x_b_2 = end % 20
    tau_obs = unique_t[t_idx]
    pi_1_arr = pi_classes[c1]; pi_2_arr = pi_classes[c2]
    elbo_per = jax.vmap(
        _elbo_traceable,
        in_axes=(None, 0, 0, None, 0, 0, 0, 0, 0, None, None),
    )(H_atom, pi_1_arr, pi_2_arr, S,
       x_a_1, x_a_2, x_b_1, x_b_2, tau_obs,
       N_FP_DEFAULT, DAMPING_DEFAULT)
    return jnp.sum(elbo_per * valid_mask)


# -- New-atom Laplace evidence under the ELBO loss --------------------------

from dataclasses import dataclass

import optax


@dataclass
class LaplaceComponentELBO:
    H_hat: np.ndarray
    log_lik_at_hat: float          # this is the ELBO at H_hat (lower bound)
    log_prior_at_hat: float
    log_det_post_prec: float
    d: int


def laplace_component_diag_jit_elbo(obs_packed: np.ndarray,
                                       valid_mask: np.ndarray,
                                       pi_classes: np.ndarray, S: np.ndarray,
                                       mu_prior: np.ndarray,
                                       tau_prior: np.ndarray,
                                       unique_t: np.ndarray,
                                       H_init: np.ndarray,
                                       n_steps: int = 30, lr: float = 0.05
                                       ) -> LaplaceComponentELBO:
    """Same Laplace-MAP + diagonal-Hessian construction as
    `laplace_potts_v2.laplace_component_diag_jit`, but with the ELBO loss
    in place of the exact 400-state log-P.

    Caveat 1 (statistical): substituting ELBO for log L in the Laplace
    evidence gives a BIASED estimate of log p(data) — both the location
    of H_hat and the log L value at H_hat shift by the (data-dependent)
    Jensen gap. For CRP-Gibbs it's still useful as a comparable score
    for new-vs-existing atom evidence as long as both branches use the
    same loss family.

    Caveat 2 (memory): jax.linearize on grad_fn_elbo through the inner
    scan stores per-cherry per-FP-iter forward residuals, so peak
    captured-constant size scales as O(M × N_FP × A^4). At M ~ 100,
    N_FP=12, A=20 this is ~12 GB and triggers the JAX
    captured-constants warning at 11+ GB. With M ~ 1000 (typical at
    K_c=8 with all class-pair atoms aggregated) this OOMs. Two
    mitigations to add when needed: drop N_FP to 4-6 (linear
    reduction), or chunk M and replace linearize with sequential
    grad-of-jvp HVPs that don't store per-cherry residuals.
    """
    obs_j = jnp.asarray(obs_packed)
    mask_j = jnp.asarray(valid_mask, dtype=jnp.float64)
    pi_j = jnp.asarray(pi_classes)
    S_j = jnp.asarray(S)
    mu_j = jnp.asarray(mu_prior); tau_j = jnp.asarray(tau_prior)
    t_j = jnp.asarray(unique_t)

    H_flat = jnp.asarray(_sym_to_flat(jnp.asarray(H_init)))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(H_flat)
    for _ in range(n_steps):
        g = grad_fn_elbo(H_flat, obs_j, mask_j, pi_j, S_j, mu_j, tau_j, t_j)
        updates, opt_state = optimizer.update(g, opt_state)
        H_flat = optax.apply_updates(H_flat, updates)

    H_hat = np.asarray(_flat_to_sym(H_flat))

    # Diagonal Hessian via linearize at H_hat (same trick as v2 exact).
    grad_at_obs = lambda H: grad_fn_elbo(
        H, obs_j, mask_j, pi_j, S_j, mu_j, tau_j, t_j
    )
    _, hvp_fn_at_H_hat = jax.linearize(grad_at_obs, H_flat)
    hvp_jit = jax.jit(hvp_fn_at_H_hat)

    d = H_flat.shape[0]
    diags = np.zeros(d)
    eye_np = np.eye(d)
    for i in range(d):
        hv = hvp_jit(jnp.asarray(eye_np[i]))
        diags[i] = float(hv[i])
    # Float32 paths can produce NaN or non-positive Hessian-diagonal entries
    # (most commonly from the prior-dominated weak-curvature directions).
    # Floor + NaN-replace before log so log_det stays finite.
    diags = np.where(np.isnan(diags), 1e-6, diags)
    post_prec_diag = np.maximum(diags, 1e-6)
    log_det = float(np.sum(np.log(post_prec_diag)))

    nlp = float(loss_fn_elbo(H_flat, obs_j, mask_j, pi_j, S_j, mu_j, tau_j, t_j))
    H_mat_j = jnp.asarray(H_hat)
    log_pr = float(log_prior_pathwise(H_mat_j, mu_j, tau_j))
    log_lik = -nlp - log_pr           # this is sum-ELBO at H_hat

    return LaplaceComponentELBO(
        H_hat=H_hat, log_lik_at_hat=log_lik, log_prior_at_hat=log_pr,
        log_det_post_prec=log_det, d=d,
    )


def laplace_log_evidence_elbo(comp: LaplaceComponentELBO) -> float:
    """log p(data) lower bound via Laplace at the ELBO-MAP."""
    return (comp.log_lik_at_hat + comp.log_prior_at_hat
            + 0.5 * comp.d * np.log(2 * np.pi)
            - 0.5 * comp.log_det_post_prec)


# -- Convenience: keep the old (PI_LG08-fixed, K_c=1) entrypoint for tests --

@partial(jax.jit, static_argnames=('n_fp',))
def loss_fn_elbo_with_tau(H_flat, x_a_1_arr, x_a_2_arr, x_b_1_arr, x_b_2_arr,
                            tau_per_obs, valid_mask, mu_prior, tau_prior,
                            n_fp=N_FP_DEFAULT):
    """K_c=1 single-atom ELBO loss with explicit tau, pi_LG08 fixed.

    Kept for the smoke test in tests/smoke_loss_elbo.py — caller passes
    pre-decoded (x_a_1, x_a_2, x_b_1, x_b_2, tau) arrays. For production
    use loss_fn_elbo (matches the laplace_potts_v2.loss_fn signature
    and supports K_c > 1 with class-specific pi).
    """
    H_mat = _flat_to_sym(H_flat)
    pi_LG = jnp.asarray(PI_LG08)
    S = jnp.asarray(S_LG08_F81)
    elbo_per = jax.vmap(
        _elbo_traceable,
        in_axes=(None, None, None, None, 0, 0, 0, 0, 0, None, None),
    )(H_mat, pi_LG, pi_LG, S,
       x_a_1_arr, x_a_2_arr, x_b_1_arr, x_b_2_arr, tau_per_obs,
       n_fp, DAMPING_DEFAULT)
    log_pr = log_prior_pathwise(H_mat, mu_prior, tau_prior)
    return -jnp.sum(elbo_per * valid_mask) - log_pr
