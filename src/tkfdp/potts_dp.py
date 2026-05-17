"""Third Dirichlet process: D_H ~ DP(alpha_H, G_0^H) over Potts atoms.

Per main.tex \S2 (post-2026-05-08 reparam) and the handoff item 6:
each unordered class-pair {c, c'} is assigned to a Potts atom h_{cc'}
drawn from a DP. With alpha_H small the K(K+1)/2 class-pair indices
collapse onto a small number K_H of canonical Potts matrices. This
makes the model identifiable for K_c >> 1 (without it the Potts
parameters scale as K_c^2 * A^2 = ~226k at K_c = 60, A = 20).

The materialized slice for class-pair (c, c') is

    H_{cc'}(i, j) = H_{h_{cc'}}(i, j),

where H_h is symmetric in (i, j) by construction (item 5 / G_0^H).

This module exposes:

  PottsDPState (dataclass): (atoms, assignments, counts, alpha_H)
    - atoms: (K_H_active, A, A) array of Potts matrices
    - assignments: (K_c, K_c) symmetric int array, assignments[c, c'] is
      the index into atoms for the unordered pair {c, c'}
    - counts: (K_H_active,) class-pair counts per atom (== how many
      unordered (c, c') indices are assigned to atom h)
    - alpha_H: scalar concentration

  init_potts_dp(K_c, alpha_H, mu_prior, tau_prior, rng): start with all
    class-pairs assigned to a single new atom drawn from G_0^H.

  gibbs_step_assignment(state, c, cp, log_pair_lik_fn,
                          log_new_atom_marginal_fn): CRP-Gibbs over
    h_{c, c'}; existing-atom branch evaluated by log_pair_lik_fn(H_h),
    new-atom branch via Laplace from laplace_potts.py.

  escobar_west_alpha_H_update: Gamma posterior update on alpha_H given
    K_H_active and N_pairs = K_c * (K_c + 1) / 2.

  jain_neal_split_merge (optional, deferred): split/merge proposals on
    the atom partition.

The path-DCA likelihood is supplied as a callable of (H_atom -> scalar
log-likelihood for the (cherry, edge) observations using that atom).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

from .laplace_potts import (LaplaceComponent, laplace_log_evidence,
                              log_mixture_evidence, multi_seed_mixture)


@dataclass
class PottsDPState:
    K_c: int                              # number of site classes
    A: int                                # alphabet size (20)
    atoms: np.ndarray                     # (K_H_active, A, A) symmetric
    assignments: np.ndarray               # (K_c, K_c) int, symm; -1 if no class-pair instantiated
    counts: np.ndarray                    # (K_H_active,) int
    alpha_H: float = 1.0
    mu_prior: np.ndarray | None = None    # (A, A)
    tau_prior: np.ndarray | None = None   # (A, A)
    # TSB extension: stick weights ρ over the K_H_max truncated atom slots.
    # When non-None, the assignment resample uses TSB-Categorical instead of
    # CRP-Gibbs and atom slots remain allocated even when no class-pair is
    # currently assigned. K_H_active = K_H_max = K_c(K_c+1)/2 always.
    rho: np.ndarray | None = None         # (K_H_max,)
    tsb_betas: np.ndarray | None = None   # (K_H_max - 1,) stick proportions
    # Side-potential extension: per-class-pair Gaussian-prior h vectors that
    # modify the per-site stationary at each pair. Stored at the canonical
    # unordered-pair index (c <= c'); shape (K_c(K_c+1)/2, 2, A) where
    # h_pairs[k, 0] = h_a (vector for the first class of canonical pair) and
    # h_pairs[k, 1] = h_b (second class). For self-pair (c,c) the two slots
    # are tied to the same vector. None disables side potentials.
    h_pairs: np.ndarray | None = None     # (K_c(K_c+1)/2, 2, A) or None


def _class_pair_idx(K_c: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Map for unordered class pairs (c, c') with c <= c'. Returns:
       sym_to_pair_idx: (K_c, K_c) -> int (linearized triangular)
       pairs: list of (c, c') with c <= c'.
    """
    pairs = []
    for c in range(K_c):
        for cp in range(c, K_c):
            pairs.append((c, cp))
    sym_to_pair_idx = -np.ones((K_c, K_c), dtype=np.int64)
    for k, (c, cp) in enumerate(pairs):
        sym_to_pair_idx[c, cp] = k
        sym_to_pair_idx[cp, c] = k
    return sym_to_pair_idx, pairs


def init_potts_dp(K_c: int, alpha_H: float,
                   mu_prior: np.ndarray, tau_prior: np.ndarray,
                   rng: np.random.Generator,
                   strategy: str = 'one_atom') -> PottsDPState:
    """Initialize the Potts DP. `strategy`:
       'one_atom':  every class-pair assigned to a single atom drawn from G_0^H.
       'fresh':     each class-pair gets its own atom (large K_H_active = K_c (K_c+1)/2).
    The prior atoms are drawn iid from the per-AA-pair Gaussian base
    measure G_0^H = N(mu_prior[i, j], tau_prior[i, j]^{-1}) for i <= j,
    reflected to a symmetric (A, A) matrix.
    """
    A = mu_prior.shape[0]
    n_pairs = K_c * (K_c + 1) // 2
    if strategy == 'one_atom':
        atom = _draw_atom_from_g0(mu_prior, tau_prior, rng)
        atoms = atom[None, ...]
        assignments = np.zeros((K_c, K_c), dtype=np.int64)
        counts = np.array([n_pairs], dtype=np.int64)
    elif strategy == 'fresh':
        atoms = np.stack([_draw_atom_from_g0(mu_prior, tau_prior, rng)
                            for _ in range(n_pairs)])
        assignments = np.zeros((K_c, K_c), dtype=np.int64)
        cp_table, pairs = _class_pair_idx(K_c)
        for k, (c, cp) in enumerate(pairs):
            assignments[c, cp] = k
            assignments[cp, c] = k
        counts = np.ones(n_pairs, dtype=np.int64)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return PottsDPState(K_c=K_c, A=A, atoms=atoms, assignments=assignments,
                          counts=counts, alpha_H=alpha_H,
                          mu_prior=mu_prior, tau_prior=tau_prior)


def _draw_atom_from_g0(mu_prior: np.ndarray, tau_prior: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    A = mu_prior.shape[0]
    iu = np.triu_indices(A)
    sigma = 1.0 / np.sqrt(tau_prior[iu])
    sample_flat = rng.normal(loc=mu_prior[iu], scale=sigma)
    H = np.zeros((A, A))
    H[iu] = sample_flat
    H = H + H.T - np.diag(np.diag(H))
    return H


def gibbs_step_assignment(state: PottsDPState, c: int, cp: int,
                           log_pair_lik_fn,
                           new_atom_marginal_fn,
                           rng: np.random.Generator) -> PottsDPState:
    """CRP-Gibbs step: resample h_{c, c'} given the rest of the assignments.

    log_pair_lik_fn: callable(H_atom) -> scalar log-likelihood of the
        observations using class-pair {c, c'} under H_atom.
    new_atom_marginal_fn: callable() -> (log_marginal, H_hat).
        log_marginal is log p(data_{c, c'}) marginalized over a fresh
        atom drawn from G_0^H (typically the Laplace mixture from
        `laplace_potts.py`). H_hat is the corresponding MAP / mode-of-the-
        best-component, used as the new atom's initial value if this
        branch wins. Threading H_hat (rather than drawing a fresh prior
        sample) is essential for the new-atom branch to actually
        exploit the data evidence the marginal scored.

    Predictive (CRP):
      P(h_{c, c'} = h | rest) ∝ counts^{-(c, c')}[h] * L_h        existing
                              ∝ alpha_H * marginal_new           new atom

    where L_h = exp(log_pair_lik_fn(atoms[h])) and marginal_new =
    exp(log_marginal) from new_atom_marginal_fn().
    """
    # Detach (c, c') from its current atom
    h_curr = int(state.assignments[c, cp])
    state.counts[h_curr] -= 1
    new_atoms = state.atoms; new_counts = state.counts
    if state.counts[h_curr] == 0:
        # Drop the empty atom (compact representation)
        keep = np.arange(len(state.counts)) != h_curr
        new_atoms = state.atoms[keep]
        new_counts = state.counts[keep]
        # Re-index assignments
        idx_map = -np.ones(len(state.counts), dtype=np.int64)
        idx_map[keep] = np.arange(keep.sum())
        new_assignments = np.where(state.assignments >= 0, idx_map[state.assignments], -1)
    else:
        new_assignments = state.assignments.copy()
    state = PottsDPState(K_c=state.K_c, A=state.A, atoms=new_atoms,
                            assignments=new_assignments, counts=new_counts,
                            alpha_H=state.alpha_H,
                            mu_prior=state.mu_prior, tau_prior=state.tau_prior)

    # Compute the new-atom marginal once (returns (log_marg, H_hat)).
    log_new_marg, H_hat_new = new_atom_marginal_fn()

    # Score existing atoms + new-atom branch
    log_scores = []
    for h in range(len(state.atoms)):
        lp = float(log_pair_lik_fn(state.atoms[h]))
        log_scores.append(np.log(state.counts[h]) + lp)
    log_scores.append(np.log(state.alpha_H) + float(log_new_marg))
    log_scores = np.asarray(log_scores)

    # Sample from categorical
    log_scores -= log_scores.max()
    probs = np.exp(log_scores); probs /= probs.sum()
    choice = int(rng.choice(len(probs), p=probs))

    if choice < len(state.atoms):
        # Reassign to existing atom
        state.counts[choice] += 1
        state.assignments[c, cp] = choice
        state.assignments[cp, c] = choice
    else:
        # Create new atom from the Laplace MAP H_hat (NOT a fresh prior draw —
        # the marginal score reflects the data evidence concentrated at H_hat).
        state.atoms = np.concatenate([state.atoms, H_hat_new[None, ...]], axis=0)
        state.counts = np.concatenate([state.counts, [1]])
        state.assignments[c, cp] = len(state.atoms) - 1
        state.assignments[cp, c] = len(state.atoms) - 1
    return state


def alpha_H_map_update(K_H_active: int, n_pairs_total: int,
                         prior_a: float = 2.0, prior_b: float = 2.0) -> float:
    """1D Brent MAP on alpha_H given (K_H_active, N_pairs_total) and a
    Gamma(prior_a, prior_b) prior.

    The conditional log-posterior on alpha is
       log p(alpha | K, N) = (prior_a + K - 1) log alpha - prior_b alpha
                              + log Γ(alpha) - log Γ(alpha + N)  + const,
    derived from the CRP marginal. Computed by minimize_scalar over
    log-alpha.

    Note: this is NOT the full Escobar-West auxiliary-variable Gibbs
    sampler (which would draw alpha from the conditional rather than
    return the MAP). It's a fast deterministic update suitable for
    SVI-style alternation. The Gibbs version requires the current
    alpha and is left as a future drop-in.
    """
    from scipy.optimize import minimize_scalar
    def neg_log_post(log_a):
        alpha = float(np.exp(log_a))
        return -((prior_a + K_H_active - 1) * log_a - prior_b * alpha
                  + gammaln(alpha) - gammaln(alpha + n_pairs_total))
    res = minimize_scalar(neg_log_post, bounds=(np.log(1e-3), np.log(1e3)),
                            method='bounded', options=dict(xatol=1e-3))
    return float(np.exp(res.x))


def escobar_west_alpha_H_update(K_H_active: int, n_pairs_total: int,
                                  alpha_curr: float,
                                  prior_a: float = 2.0, prior_b: float = 2.0,
                                  rng: np.random.Generator | None = None) -> float:
    """Escobar--West (1995) auxiliary-variable Gibbs update for alpha_H.

      eta_aux ~ Beta(alpha_curr + 1, N)
      epsilon = (a + K - 1) / (N (b - log eta) + a + K - 1)
      With prob epsilon:  alpha ~ Gamma(a + K, b - log eta)
      Else:               alpha ~ Gamma(a + K - 1, b - log eta).
    """
    if rng is None:
        rng = np.random.default_rng()
    eta_aux = rng.beta(alpha_curr + 1, n_pairs_total)
    log_eta_aux = np.log(max(eta_aux, 1e-300))
    eps = (prior_a + K_H_active - 1) / (
        n_pairs_total * (prior_b - log_eta_aux) + prior_a + K_H_active - 1)
    if rng.random() < eps:
        return float(rng.gamma(prior_a + K_H_active, 1.0 / (prior_b - log_eta_aux)))
    else:
        return float(rng.gamma(prior_a + K_H_active - 1, 1.0 / (prior_b - log_eta_aux)))


# --- Helper for the new-atom marginal via Laplace mixture -------------------

def new_atom_log_marginal_via_laplace(neg_log_post_fn,
                                        seeds: list[np.ndarray],
                                        mu_prior: np.ndarray,
                                        tau_prior: np.ndarray,
                                        n_steps: int = 30, lr: float = 0.05
                                        ) -> tuple[float, np.ndarray]:
    """Wrap multi_seed_mixture for use in the CRP new-atom branch.

    `neg_log_post_fn` is the negative-log-posterior on H (= -log L(data|H) -
    log G_0(H)) for the class-pair (c, c') being resampled, in the flat
    H_flat parameterization expected by laplace_potts.

    Returns (log marginal, best H_hat). The marginal is the SUM of per-
    component Laplace evidences (basin-coverage interpretation per main.tex
    §7.4); H_hat is the MAP from the highest-evidence seed and is used as
    the new atom's initial value when the CRP picks the new-atom branch.
    """
    components, log_evs = multi_seed_mixture(neg_log_post_fn, seeds,
                                                mu_prior, tau_prior,
                                                n_steps=n_steps, lr=lr)
    log_marg = log_mixture_evidence(log_evs)
    best = int(np.argmax(log_evs))
    return log_marg, components[best].H_hat


# --- TSB (truncated stick-breaking) variant ---------------------------------
#
# The CRP-Gibbs above is sticky on the low-K side (alpha_H * marg_new must
# beat n_existing * marg_existing to spawn a 2nd atom; with one existing
# atom that fits all K_c(K_c+1)/2 pairs OK, this is a high bar). The TSB
# variant truncates at K_H_max = K_c(K_c+1)/2, allocates that many atoms
# from the prior at init, and resamples class-pair-to-atom assignments via
# Categorical(rho * lik). Stick weights rho are conjugate-Beta-updated from
# the per-atom counts, mirroring tsb.py for site classes.

def _kh_max(K_c: int) -> int:
    return K_c * (K_c + 1) // 2


def canonical_pair_idx_table(K_c: int) -> tuple[np.ndarray, np.ndarray]:
    """Build lookup tables for the canonical-pair indexing used by h_pairs.

    Returns:
      cp_idx[c1, c2] = index k in {0..K_c(K_c+1)/2 - 1} of the unordered
                        pair {c1, c2}, with the convention min(c1,c2) ≤
                        max(c1,c2) for the canonical (a, b) ordering.
      cp_swap[c1, c2] = 0 if c1 <= c2 (canonical orientation), 1 if c1 > c2
                         (need to swap h_a/h_b at gather).
    """
    n_canonical = _kh_max(K_c)
    cp_idx = np.zeros((K_c, K_c), dtype=np.int64)
    cp_swap = np.zeros((K_c, K_c), dtype=np.int64)
    k = 0
    for c in range(K_c):
        for cp in range(c, K_c):
            cp_idx[c, cp] = k
            cp_idx[cp, c] = k
            # Swap only when ordered pair has c1 > c2 (off-canonical orientation)
            cp_swap[cp, c] = 1
            k += 1
    return cp_idx, cp_swap


def canonical_pair_is_diag(K_c: int) -> np.ndarray:
    """Boolean mask of shape (K_c(K_c+1)/2,), True where the canonical pair
    is a self-pair (c, c). For self-pairs the two sites are exchangeable,
    so the side-potential vectors h_a, h_b must be tied (h_a = h_b) to
    keep the joint pair distribution symmetric and the joint Q reversible.
    """
    cp_idx, _ = canonical_pair_idx_table(K_c)
    n_canonical = _kh_max(K_c)
    is_diag = np.zeros(n_canonical, dtype=bool)
    for c in range(K_c):
        is_diag[int(cp_idx[c, c])] = True
    return is_diag


def symmetrize_h_pairs_diag(h_pairs: np.ndarray, K_c: int) -> np.ndarray:
    """In-place project h_pairs onto the symmetry constraint h_a = h_b on
    self-pairs (c, c). Slot 1 is overwritten with slot 0 for diagonal
    canonical pairs; off-diagonal pairs are unchanged.
    """
    if h_pairs is None:
        return h_pairs
    is_diag = canonical_pair_is_diag(K_c)
    diag_idx = np.flatnonzero(is_diag)
    h_pairs[diag_idx, 1, :] = h_pairs[diag_idx, 0, :]
    return h_pairs


def init_potts_tsb(K_c: int, alpha_H: float,
                     mu_prior: np.ndarray, tau_prior: np.ndarray,
                     rng: np.random.Generator,
                     K_H_max: int | None = None,
                     use_side_potentials: bool = False) -> PottsDPState:
    """Initialize a TSB-Potts state: K_H_max atoms drawn from G_0^H, with
    each unordered (c, c') class-pair assigned to a DIFFERENT atom slot
    (since K_H_max == #class-pairs, the assignment is bijective).
    Uniform stick weights at init.

    Why bijective init rather than all-on-atom-0: the TSB resample uses
    Categorical(rho * likelihood). If only atom 0 was ever trained on
    data while atoms 1..K_H_max-1 stayed at prior init, atom 0's
    per-pair likelihood dominates and the resample collapses everything
    back onto atom 0 -- a chicken-and-egg failure mode where the unused
    atoms can never gain data to differentiate. Starting bijective
    ensures every atom gets at least one class-pair's data through the
    early per-outer atom-MAP, so by the first TSB resample all atoms
    have meaningful per-pair likelihoods.
    """
    A = mu_prior.shape[0]
    K_H_natural = _kh_max(K_c)
    if K_H_max is None or K_H_max >= K_H_natural:
        K_H_max = K_H_natural
        # Bijective: each unordered (c, c') -> own atom (one-to-one).
        atoms = np.stack([
            _draw_atom_from_g0(mu_prior, tau_prior, rng)
            for _ in range(K_H_max)
        ], axis=0)
        counts = np.ones(K_H_max, dtype=np.int64)
        assignments = np.zeros((K_c, K_c), dtype=np.int64)
        cp_table, pairs = _class_pair_idx(K_c)
        for k, (c, cp) in enumerate(pairs):
            assignments[c, cp] = k
            assignments[cp, c] = k
    else:
        # K_H_max < #unordered class-pairs — round-robin init: pair k -> atom (k % K_H_max).
        # Spreads class-pairs roughly evenly across atoms; TSB resampling
        # specializes them.
        atoms = np.stack([
            _draw_atom_from_g0(mu_prior, tau_prior, rng)
            for _ in range(K_H_max)
        ], axis=0)
        counts = np.zeros(K_H_max, dtype=np.int64)
        assignments = np.zeros((K_c, K_c), dtype=np.int64)
        cp_table, pairs = _class_pair_idx(K_c)
        for k, (c, cp) in enumerate(pairs):
            h = k % K_H_max
            assignments[c, cp] = h
            assignments[cp, c] = h
            counts[h] += 1
    rho = np.full(K_H_max, 1.0 / K_H_max)
    tsb_betas = np.full(K_H_max - 1, 1.0 / K_H_max)
    h_pairs = None
    if use_side_potentials:
        n_canonical = _kh_max(K_c)             # K_c(K_c+1)/2
        h_pairs = np.zeros((n_canonical, 2, A), dtype=np.float64)
    return PottsDPState(
        K_c=K_c, A=A, atoms=atoms, assignments=assignments, counts=counts,
        alpha_H=alpha_H, mu_prior=mu_prior, tau_prior=tau_prior,
        rho=rho, tsb_betas=tsb_betas, h_pairs=h_pairs,
    )


def tsb_resample_assignments(state: PottsDPState,
                                log_lik_per_pair_per_atom: np.ndarray,
                                rng: np.random.Generator) -> PottsDPState:
    """Resample each unordered class-pair (c, c')'s atom assignment via
    Categorical(rho * lik). Returns a new PottsDPState with updated
    assignments and counts.

    log_lik_per_pair_per_atom: (K_H_max, K_c, K_c) — log P(data_{c, c'}
    | atoms[h], pi_class[c], pi_class[c']). Symmetric in (c, c'); only
    the upper triangle including diagonal is used.
    """
    K_c = state.K_c
    K_H_max = state.atoms.shape[0]
    new_assignments = state.assignments.copy()
    new_counts = np.zeros(K_H_max, dtype=np.int64)
    log_rho = np.log(np.clip(state.rho, 1e-12, None))
    for c in range(K_c):
        for cp in range(c, K_c):
            log_post = log_rho + log_lik_per_pair_per_atom[:, c, cp]
            log_post -= log_post.max()
            probs = np.exp(log_post); probs /= probs.sum()
            h = int(rng.choice(K_H_max, p=probs))
            new_assignments[c, cp] = h
            new_assignments[cp, c] = h
            new_counts[h] += 1
    return PottsDPState(
        K_c=K_c, A=state.A, atoms=state.atoms,
        assignments=new_assignments, counts=new_counts,
        alpha_H=state.alpha_H,
        mu_prior=state.mu_prior, tau_prior=state.tau_prior,
        rho=state.rho, tsb_betas=state.tsb_betas,
    )


def tsb_update_rho(state: PottsDPState,
                      rng: np.random.Generator,
                      mode: str = "sample") -> PottsDPState:
    """Conjugate Beta posterior on stick proportions given per-atom
    counts. Uses the same scheme as tkfdp.tsb.update_betas_from_counts
    but on the Potts-atom counts vector. mode ∈ {sample, map}.
    """
    from .tsb import stick_to_weights, update_betas_from_counts
    counts = state.counts.astype(np.float64)
    new_betas = update_betas_from_counts(
        counts, state.alpha_H, rng=rng, mode=mode
    )
    new_rho = stick_to_weights(new_betas)
    return PottsDPState(
        K_c=state.K_c, A=state.A, atoms=state.atoms,
        assignments=state.assignments, counts=state.counts,
        alpha_H=state.alpha_H,
        mu_prior=state.mu_prior, tau_prior=state.tau_prior,
        rho=new_rho, tsb_betas=new_betas,
    )
