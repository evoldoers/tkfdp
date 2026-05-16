"""Laplace approximation (with multi-seed mixture) for the Potts atom
integrals in main.tex §7.4.

Each Potts atom H_h is real-valued (A x A symmetric, A = 20). The
path-DCA likelihood
    log p_pathDCA(data | H_h)
is locally Gaussian in H_h to second order, so

    log p(data | H_h) ≈ log p(data | hat H_h) - 0.5 (H_h - hat H_h)^T Λ_h (H_h - hat H_h),

with Λ_h = -∇²_{H_h} log p_pathDCA at the MAP. Combined with the
Gaussian prior G_0^H = ∏ N(H_h(i, j) | μ_kl, τ_kl^{-1}) the posterior
is Gaussian and the log-evidence integral is closed form:

    log ∫ p(data | H') G_0(H') dH'
        ≈ log p(data | hat H_h) + log G_0(hat H_h)
          + (d/2) log(2π) - 0.5 log det(diag(τ_kl) + Λ_h),

with d = A(A+1)/2 the symmetric-slice dimension. Bernstein-von Mises
gives O(N_h^{-1}) error in atom sufficient-statistic count.

Multi-seed mixture (main.tex §7.4 'Multi-seed Laplace mixture' para):
for atoms with multiple plausible coupling patterns (e.g., charge-
complementary vs. hydrophobic) the single-mode Laplace under-estimates
mass elsewhere. Run the optimizer from K_seed seeds (prior mean,
previous iterate, sibling atom, ...), get K_seed components, combine
as weighted Gaussian mixture with weights

    w_k ∝ p_pathDCA(data | hat H_h^(k)) · G_0^H(hat H_h^(k))
          · (2π)^{d/2} |hat Σ_h^(k)|^{1/2}.

The mixture estimate of the integral is sum_k w_k.

This module exposes:
- log_prior_pathwise(H, mu, tau): log G_0^H per AA-pair Gaussian.
- find_map_potts(loss_fn, H_init, n_steps, lr): Newton/L-BFGS-via-JAX
  for MAP H. Uses jaxopt or pure JAX.
- laplace_log_evidence(...): the closed-form expression.
- multi_seed_mixture(seeds, ...): combines K_seed Laplace components.

The path-DCA likelihood is supplied as a callable; this module is
agnostic to the exact form (cluster size 2 or larger).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


# --- Symmetric slice helpers (operate on flat upper-triangle params) --------

A = 20


def _flat_to_sym(flat: jnp.ndarray) -> jnp.ndarray:
    """Reshape flat (A*(A+1)//2,) to symmetric (A, A) matrix.

    Index convention: flat[k] for k = i*A - i*(i-1)//2 + (j-i), i<=j.
    """
    H = jnp.zeros((A, A))
    iu = jnp.triu_indices(A)
    H = H.at[iu].set(flat)
    H = H + H.T - jnp.diag(jnp.diag(H))
    return H


def _sym_to_flat(H: jnp.ndarray) -> jnp.ndarray:
    iu = jnp.triu_indices(A)
    return H[iu]


def log_prior_pathwise(H: jnp.ndarray,
                        mu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
    """log G_0^H(H) = sum_{i <= j} log N(H[i, j] | mu[i, j], tau[i, j]^{-1}).

    `mu`, `tau` are A x A matrices indexed by AA pair (k, l) =
    (min(i, j), max(i, j)); only the i <= j entries are read."""
    iu = jnp.triu_indices(A)
    H_flat = H[iu]; mu_flat = mu[iu]; tau_flat = tau[iu]
    # log N(x | mu, tau^{-1}) = 0.5 (log tau - log(2π)) - 0.5 tau (x - mu)^2
    return jnp.sum(0.5 * (jnp.log(tau_flat) - jnp.log(2 * jnp.pi))
                   - 0.5 * tau_flat * (H_flat - mu_flat) ** 2)


# --- MAP optimization ------------------------------------------------------

@dataclass
class LaplaceComponent:
    H_hat: np.ndarray        # (A, A) symmetric MAP estimate
    log_lik_at_hat: float    # log p(data | H_hat)
    log_prior_at_hat: float  # log G_0(H_hat)
    log_det_post_prec: float # log det(diag(tau) + Hessian)  in flat dim
    d: int                   # = A * (A+1) / 2 = 210 for A=20


def find_map_potts(neg_log_post_fn,
                    H_init: np.ndarray,
                    n_steps: int = 50,
                    lr: float = 0.1) -> tuple[np.ndarray, float]:
    """Find the MAP H by gradient descent on `neg_log_post_fn(H_flat) ->
    scalar`, where H_flat is the (A*(A+1)/2,) symmetric-slice flat
    parameterization. Returns (H_hat (A, A), final_neg_log_post).

    Uses a simple Adam-like optimizer for robustness across seeds; for
    fast convergence near a smooth minimum, swap in scipy.optimize or
    jaxopt L-BFGS.
    """
    import optax
    H_flat = jnp.asarray(_sym_to_flat(jnp.asarray(H_init)))
    optimizer = optax.adam(lr)
    state = optimizer.init(H_flat)
    grad_fn = jax.jit(jax.grad(neg_log_post_fn))
    val_fn = jax.jit(neg_log_post_fn)
    for _ in range(n_steps):
        g = grad_fn(H_flat)
        updates, state = optimizer.update(g, state)
        H_flat = optax.apply_updates(H_flat, updates)
    final = float(val_fn(H_flat))
    H_hat = np.asarray(_flat_to_sym(H_flat))
    return H_hat, final


def hessian_at(neg_log_post_fn, H_hat: np.ndarray) -> np.ndarray:
    """Hessian of neg_log_post (in flat-param space) at H_hat.
    Returns (d, d) matrix where d = A(A+1)/2."""
    H_flat = jnp.asarray(_sym_to_flat(jnp.asarray(H_hat)))
    H_mat = np.asarray(jax.hessian(neg_log_post_fn)(H_flat))
    return H_mat


# --- Laplace log evidence and multi-seed mixture ---------------------------

def laplace_component(neg_log_post_fn, H_init: np.ndarray,
                       mu_prior: np.ndarray, tau_prior: np.ndarray,
                       n_steps: int = 50, lr: float = 0.1
                       ) -> LaplaceComponent:
    """Run the Newton/Adam MAP solver, evaluate log p(data | H_hat),
    log G_0(H_hat), and the posterior precision determinant via
    diag(tau_prior_flat) + Hessian_at_H_hat. Returns a LaplaceComponent."""
    H_hat, neg_lp_at_hat = find_map_potts(neg_log_post_fn, H_init, n_steps, lr)
    iu = np.triu_indices(A)
    tau_flat = tau_prior[iu]

    # Hessian at the MAP — this is Hessian of -log p (data + prior),
    # so it equals Λ + diag(tau) (information of likelihood + prior).
    H_post_prec = hessian_at(neg_log_post_fn, H_hat)
    # log det
    sign, logdet = np.linalg.slogdet(H_post_prec)
    if sign <= 0:
        # Hessian not PD; jitter
        d = H_post_prec.shape[0]
        H_post_prec = H_post_prec + 1e-3 * np.eye(d)
        sign, logdet = np.linalg.slogdet(H_post_prec)

    # neg log post at H_hat = -log p(data | H_hat) - log G_0(H_hat) (no
    # constants), so log p(data) + log G_0 = -neg_lp_at_hat.
    log_lik_at_hat = float(-neg_lp_at_hat) - float(log_prior_pathwise(
        jnp.asarray(H_hat), jnp.asarray(mu_prior), jnp.asarray(tau_prior)))
    log_prior_at_hat = float(log_prior_pathwise(
        jnp.asarray(H_hat), jnp.asarray(mu_prior), jnp.asarray(tau_prior)))

    return LaplaceComponent(
        H_hat=H_hat, log_lik_at_hat=log_lik_at_hat,
        log_prior_at_hat=log_prior_at_hat,
        log_det_post_prec=float(logdet),
        d=A * (A + 1) // 2,
    )


def laplace_log_evidence(comp: LaplaceComponent) -> float:
    """Closed-form log evidence under a single Laplace component:
    log p(data) ≈ log p(data | H_hat) + log G_0(H_hat)
                + (d/2) log(2π) - 0.5 log det(post_prec).
    """
    return (comp.log_lik_at_hat + comp.log_prior_at_hat
            + 0.5 * comp.d * np.log(2 * np.pi)
            - 0.5 * comp.log_det_post_prec)


def hessian_diag_at(neg_log_post_fn, H_hat: np.ndarray) -> np.ndarray:
    """Diagonal of the Hessian at H_hat via cached sequential HVPs.

    Use `jax.linearize` to specialize the gradient at H_hat once, then
    JIT-compile the resulting JVP closure. After the first call the
    trace + compile is reused for every basis vector, so subsequent
    HVPs are only the FLOPs (no recompile).

    Sequential rather than vmap'd to keep memory bounded — vmap stacks
    the intermediates in the gradient (the K_c² × n_t × A² × A² log_P
    tensor), which scales by d=210 and OOMs.

    Cost: d sequential HVPs at JIT-compile-once cost.
    """
    H_flat = jnp.asarray(_sym_to_flat(jnp.asarray(H_hat)))
    d = H_flat.shape[0]
    grad_fn = jax.grad(neg_log_post_fn)
    # linearize at H_hat; hvp(v) = ∂grad/∂x · v = Hessian · v
    _, hvp_fn = jax.linearize(grad_fn, H_flat)
    hvp_jit = jax.jit(hvp_fn)
    eye_np = np.eye(d)
    diags = np.zeros(d)
    for i in range(d):
        hv = hvp_jit(jnp.asarray(eye_np[i]))
        diags[i] = float(hv[i])
    return diags


def laplace_component_diag(neg_log_post_fn, H_init: np.ndarray,
                              mu_prior: np.ndarray, tau_prior: np.ndarray,
                              n_steps: int = 50, lr: float = 0.1
                              ) -> LaplaceComponent:
    """Like laplace_component but uses the diagonal Hessian only. Per
    main.tex §7.4: the Gaussian prior G_0^H is diagonal in the per-AA-
    pair parameterization, so the posterior precision diag(τ_kl) + Λ_h
    is itself diagonal under a diagonal-Hessian approximation, and
    log det is sum of logs.

    Cost: O(d) HVPs vs O(d²) for the full Hessian — recommended for the
    new-atom marginal in the Potts DP CRP-Gibbs (item 6).

    Note: the convention here matches `laplace_component`: `neg_log_post_fn`
    is the negative log of (likelihood × prior), so its Hessian directly
    gives the posterior precision. The diagonal of this Hessian is the
    posterior-precision diagonal, no `+ tau_prior` needed.
    """
    H_hat, neg_lp_at_hat = find_map_potts(neg_log_post_fn, H_init, n_steps, lr)
    H_diag = hessian_diag_at(neg_log_post_fn, H_hat)
    post_prec_diag = np.maximum(H_diag, 1e-6)
    log_det = float(np.sum(np.log(post_prec_diag)))

    log_lik_at_hat = float(-neg_lp_at_hat) - float(log_prior_pathwise(
        jnp.asarray(H_hat), jnp.asarray(mu_prior), jnp.asarray(tau_prior)))
    log_prior_at_hat = float(log_prior_pathwise(
        jnp.asarray(H_hat), jnp.asarray(mu_prior), jnp.asarray(tau_prior)))
    return LaplaceComponent(
        H_hat=H_hat, log_lik_at_hat=log_lik_at_hat,
        log_prior_at_hat=log_prior_at_hat,
        log_det_post_prec=log_det,
        d=A * (A + 1) // 2,
    )


def multi_seed_mixture(neg_log_post_fn, seeds: list[np.ndarray],
                        mu_prior: np.ndarray, tau_prior: np.ndarray,
                        n_steps: int = 50, lr: float = 0.1
                        ) -> tuple[list[LaplaceComponent], np.ndarray]:
    """Run Laplace from `seeds` (list of A x A H_init), return list of
    Laplace components plus the per-seed log evidence. The log mixture
    evidence is logsumexp over the per-seed log evidences (uniform mixing
    weights w_k ∝ exp(log_evidence_k); the (2π)^{d/2} |Σ|^{1/2} term in
    the per-seed weight is identical to the single-seed log evidence,
    which is what we already compute)."""
    components = []
    log_evs = []
    for H_init in seeds:
        c = laplace_component(neg_log_post_fn, H_init, mu_prior, tau_prior,
                                n_steps=n_steps, lr=lr)
        components.append(c)
        log_evs.append(laplace_log_evidence(c))
    return components, np.array(log_evs)


def log_mixture_evidence(log_evs: np.ndarray) -> float:
    """log [sum_k exp(log_ev_k)] — basin-coverage interpretation per main.tex §7.4.

    Each Laplace component covers a distinct basin's contribution to the
    integral. The mixture estimate is the SUM (not the average) of the
    per-basin Gaussian-integral estimates. Earlier code subtracted log K
    here, which would have been right under an importance-sampling-against-
    a-mixture-proposal interpretation, but that's not what the paper
    specifies.

    For deduplicated seeds (so basins aren't double-counted), this is
    the correct estimator of log ∫ p(data | H') G_0^H(H') dH'.
    """
    import jax.scipy.special as jsp
    return float(jsp.logsumexp(jnp.asarray(log_evs)))
