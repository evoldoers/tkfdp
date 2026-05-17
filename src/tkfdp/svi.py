"""Composed SVI loop for the new (post-2026-05-08) reparameterization.

Per main.tex \S2 + \S7.4 / 7.5, the substitution-side fitting pipeline
composes four conjugate / closed-form pieces:

  - F81 generator (item 1, generator.py): builds Q per class from
    (S, pi_class, eta_pair, H_potts_pair).
  - Per-site eta_s with Gamma posterior (item 2, eta_site.py):
    closed-form (a + N_acc_s, b + T_tilde_s) given Holmes-Rubin
    sufficient stats accumulated from cherries.
  - Per-class pi^(c) with Dirichlet posterior (item 3,
    secret_destination.py): closed-form posterior given expected
    ghost counts under the secret-destination augmentation.
  - Potts DP (items 4 + 6, laplace_potts.py + potts_dp.py):
    H atoms via Laplace MAP, h_{c, c'} assignments via CRP-Gibbs.

This module orchestrates one SVI outer-step over a corpus of MSAs.
The state object `SVIState` carries:

  hyper:
    a_eta, b_eta:           Gamma prior on per-site rate.
    kappa_pi, pi_bar:       Dirichlet base measure on per-class profiles.
    mu_kl, tau_kl:          Gaussian base measure on Potts entries
                              (A x A matrices indexed by AA pair).
    alpha_c:                Site-class DP concentration (TSB).
    alpha_H:                Potts atom DP concentration.
  global:
    K_c:                    Number of site classes.
    pi_class:               (K_c, A) per-class stationary distributions.
    potts_dp:               PottsDPState (atoms, assignments, alpha_H).
    tsb_log_rho:            (K_c,) log site-class weights (truncated SB).
  per-MSA:
    states:                 List of FamilyKState (partition + class labels).
    eta_site:               List of (L,) arrays of per-column eta_s
                              posterior means.

For now this module assumes:
  - Partition is fixed (cluster-1 only — no pair edges) to focus on the
    per-class profile + per-site rate + Potts atom updates. Adding the
    Potts atom update in earnest requires the partition Gibbs, which
    will be a follow-up commit.
  - The simplified pipeline (K_c finite, e.g. 2-4); TSB hyperparameters
    update lazily via an Escobar-West-style scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from scipy.special import gammaln

from .eta_site import (hr_per_cherry, negative_binomial_log_marginal,
                        per_column_sufficient_stats, posterior_eta_mean)
from .generator import build_joint_Q, joint_stationary, symmetrize_eigh
from .lg08 import PI_LG08, S_LG08_F81
from .partition_K import FamilyKState, init_random_K
from .potts_dp import PottsDPState, alpha_H_map_update, init_potts_dp
from .secret_destination import (em_pi_update, expected_ghost_counts,
                                   dirichlet_log_marginal,
                                   dirichlet_posterior_mean)


# --- Holmes--Rubin per-class accumulators ----------------------------------

def hr_per_class_per_msa(aa_a: np.ndarray, aa_b: np.ndarray,
                          tau: np.ndarray, both_aa: np.ndarray,
                          cls: np.ndarray, K_c: int,
                          pi_class: np.ndarray, S: np.ndarray,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate Holmes-Rubin per (class, AA) sufficient stats across the MSA.

    HR is computed at the unit-rate F81 generator Q_class = S * pi_class[c]
    (no eta scaling — eta_s is integrated out via the closed-form
    Negative-Binomial marginal on the per-site rate evidence; tracking
    eta_s as a parameter is unnecessary for the SVI updates per main.tex
    §7.4 / 7.5).

    Inputs:
      aa_a, aa_b: (C, L) cherry endpoint AAs.
      tau: (C,) cherry distances.
      both_aa: (C, L) both-residue mask.
      cls: (L,) class assignments c_s ∈ {0..K_c-1}.
      pi_class: (K_c, A) per-class stationary.
      S: (A, A) F81 exchangeability.

    Outputs (per class c):
      N_acc[c]:       total expected substitutions across columns in class c.
      dwell_total[c]: (A,) summed dwell across all (cherry, col) in class c.
      Qs:             (K_c, A, A) per-class F81 generators (returned for
                       reuse downstream).
    """
    import jax.numpy as jnp
    from .eta_site import hr_batch_jit
    A = pi_class.shape[1]
    L = aa_a.shape[1]
    N_acc = np.zeros(K_c)
    dwell_total = np.zeros((K_c, A))
    # Build per-class F81 Q
    Qs = np.zeros((K_c, A, A))
    for c in range(K_c):
        Q = (S - np.diag(np.diag(S))) * pi_class[c][None, :]
        np.fill_diagonal(Q, -Q.sum(axis=1))
        Qs[c] = Q

    # Vectorized + chunked: gather (cherry, col, class) triples for valid
    # entries, batch by class, then process the per-class batch in fixed-
    # size chunks. JIT compiles ONCE per chunk size, reused across chunks
    # and across classes. Avoids the 2M-element single-vmap that was
    # taking minutes to trace for the per-cherry per-col pairs at 1000
    # families × K_c=4.
    CHUNK = 16384
    aa_a64 = np.minimum(aa_a.astype(np.int64), 19)
    aa_b64 = np.minimum(aa_b.astype(np.int64), 19)
    for c in range(K_c):
        col_mask = (cls == c)
        if not col_mask.any(): continue
        col_idx = np.flatnonzero(col_mask)
        valid_cc = both_aa[:, col_idx]
        if not valid_cc.any(): continue
        ch_idx, sub_col_idx = np.where(valid_cc)
        actual_col = col_idx[sub_col_idx]
        a_full = aa_a64[ch_idx, actual_col]
        b_full = aa_b64[ch_idx, actual_col]
        t_full = tau[ch_idx].astype(np.float64)
        Q_c = Qs[c]; pi_c = pi_class[c]
        neg_diag_Q = -np.diag(Q_c)
        Q_j = jnp.asarray(Q_c); pi_j = jnp.asarray(pi_c)
        ndQ_j = jnp.asarray(neg_diag_Q)
        B = a_full.shape[0]
        N_sum = 0.0; dwell_sum = np.zeros(A)
        for start in range(0, B, CHUNK):
            end = min(start + CHUNK, B)
            # Pad last chunk so JIT shape is static (CHUNK, ).
            sl = slice(start, end)
            ax = a_full[sl]; bx = b_full[sl]; tx = t_full[sl]
            n_real = ax.shape[0]
            if n_real < CHUNK:
                ax = np.concatenate([ax, np.zeros(CHUNK - n_real, dtype=ax.dtype)])
                bx = np.concatenate([bx, np.zeros(CHUNK - n_real, dtype=bx.dtype)])
                tx = np.concatenate([tx, np.ones(CHUNK - n_real, dtype=tx.dtype) * 0.1])
            N_c, _, dwell_c = hr_batch_jit(
                Q_j, pi_j, ndQ_j,
                jnp.asarray(ax), jnp.asarray(bx), jnp.asarray(tx),
            )
            N_arr = np.asarray(N_c)[:n_real]
            dwell_arr = np.asarray(dwell_c)[:n_real]
            N_sum += float(N_arr.sum())
            dwell_sum += dwell_arr.sum(axis=0)
        N_acc[c] = N_sum
        dwell_total[c] = dwell_sum
    return N_acc, dwell_total, Qs


# --- Per-class pi update via secret-destination Dirichlet posterior --------

def update_pi_class(K_c: int, pi_class_curr: np.ndarray,
                     dwell_total: np.ndarray, real_counts: np.ndarray,
                     S: np.ndarray, kappa_pi: float,
                     pi_bar: np.ndarray, eta_one: float = 1.0,
                     n_iters: int = 4) -> np.ndarray:
    """Update each class's pi via the secret-destination Dirichlet posterior.

    For each class c:
      ghost_y = pi_class[c, y] * eta * (T_S - dwell @ S_off[:, y])   (item 3)
      posterior pi ~ Dirichlet(kappa_pi * pi_bar + N_real + ghost)
      mean = posterior / posterior.sum().
    Iterates a few times so the ghost expectation is consistent with the
    updated pi.

    `dwell_total[c]` (A,) is the per-class summed dwell time across all
    columns in class c, eta-corrected.
    `real_counts[c]` (A,) is the per-class summed # of jumps to destination
    state y across all cherry-columns in class c. (Aggregated separately.)
    """
    A = S.shape[0]
    prior_alpha = kappa_pi * pi_bar
    pi_new = pi_class_curr.copy()
    for c in range(K_c):
        if dwell_total[c].sum() < 1e-12:
            continue
        pi_curr = pi_new[c].copy()
        for _ in range(n_iters):
            ghost = expected_ghost_counts(pi_curr, S, dwell_total[c], eta=eta_one)
            pi_post = dirichlet_posterior_mean(prior_alpha, real_counts[c], ghost)
            if np.max(np.abs(pi_post - pi_curr)) < 1e-4:
                pi_curr = pi_post; break
            pi_curr = pi_post
        pi_new[c] = pi_curr
    return pi_new


# --- Per-cherry destination-count accumulator (real_counts per class) ------

def accumulate_real_counts(aa_a: np.ndarray, aa_b: np.ndarray,
                            both_aa: np.ndarray, cls: np.ndarray,
                            K_c: int, A: int = 20) -> np.ndarray:
    """Per-class destination-count vector N^(c)_y = sum over cherries and
    columns in class c of how often destination y was observed.
    Approximation: count cherry endpoint b's (= number of times the chain
    'ended' at y from any starting state), per class.

    This is an EM-style approximation to E[N^(c)_y]; for VBEM the strict
    formula uses HR transition-counts E[N_xy], summed over x and weighted
    appropriately. For the simplified pipeline this naive count is
    adequate.
    """
    counts = np.zeros((K_c, A))
    # Vectorized: cherry × column → flat (class, b) accumulation via
    # bincount. Replaces the O(L × C) Python loop.
    cls_per_pos = np.broadcast_to(cls[None, :], aa_b.shape)   # (C, L)
    b_per_pos = np.minimum(aa_b.astype(np.int64), 19)         # (C, L)
    flat_idx = (cls_per_pos.astype(np.int64) * A
                  + b_per_pos)[both_aa]                          # (n_valid,)
    flat_counts = np.bincount(flat_idx, minlength=K_c * A)
    counts = flat_counts.reshape(K_c, A).astype(np.float64)
    return counts


# --- Per-site eta update (Gamma posterior) ---------------------------------

def per_column_log_marginal_class_specific(aa_a: np.ndarray, aa_b: np.ndarray,
                                                tau: np.ndarray, both_aa: np.ndarray,
                                                cls: np.ndarray, K_c: int,
                                                pi_class: np.ndarray, S: np.ndarray,
                                                a_eta: float, b_eta: float) -> np.ndarray:
    """Per-column closed-form Negative-Binomial log marginal:
        log p(N_acc_s | T̃_s, a_eta, b_eta)
    computed at Q_F81 with the column's class-specific pi (NOT eta-scaled —
    eta_s is what's being marginalized).
    Returns (L,) log marginals."""
    import jax.numpy as jnp
    from .eta_site import hr_batch_jit
    L = aa_a.shape[1]
    out = np.zeros(L)
    aa_a64 = np.minimum(aa_a.astype(np.int64), 19)
    aa_b64 = np.minimum(aa_b.astype(np.int64), 19)
    Qs = []
    for c in range(K_c):
        Q = (S - np.diag(np.diag(S))) * pi_class[c][None, :]
        np.fill_diagonal(Q, -Q.sum(axis=1))
        Qs.append(Q)
    # Vectorized + chunked (same pattern as hr_per_class_per_msa).
    CHUNK = 16384
    for c in range(K_c):
        col_mask = (cls == c)
        if not col_mask.any(): continue
        col_idx = np.flatnonzero(col_mask)
        valid_cc = both_aa[:, col_idx]
        if not valid_cc.any(): continue
        ch_idx, sub_col_idx = np.where(valid_cc)
        actual_col = col_idx[sub_col_idx]
        a_full = aa_a64[ch_idx, actual_col]
        b_full = aa_b64[ch_idx, actual_col]
        t_full = tau[ch_idx].astype(np.float64)
        Q_c = Qs[c]; pi_c = pi_class[c]
        neg_diag_Q = -np.diag(Q_c)
        Q_j = jnp.asarray(Q_c); pi_j = jnp.asarray(pi_c)
        ndQ_j = jnp.asarray(neg_diag_Q)
        B = a_full.shape[0]
        N_arr_full = np.zeros(B); T_arr_full = np.zeros(B)
        for start in range(0, B, CHUNK):
            end = min(start + CHUNK, B)
            sl = slice(start, end)
            ax = a_full[sl]; bx = b_full[sl]; tx = t_full[sl]
            n_real = ax.shape[0]
            if n_real < CHUNK:
                ax = np.concatenate([ax, np.zeros(CHUNK - n_real, dtype=ax.dtype)])
                bx = np.concatenate([bx, np.zeros(CHUNK - n_real, dtype=bx.dtype)])
                tx = np.concatenate([tx, np.ones(CHUNK - n_real, dtype=tx.dtype) * 0.1])
            N_c, T_c, _ = hr_batch_jit(
                Q_j, pi_j, ndQ_j,
                jnp.asarray(ax), jnp.asarray(bx), jnp.asarray(tx),
            )
            N_arr_full[start:end] = np.asarray(N_c)[:n_real]
            T_arr_full[start:end] = np.asarray(T_c)[:n_real]
        # Accumulate per column.
        N_per_col = np.zeros(L); T_per_col = np.zeros(L)
        np.add.at(N_per_col, actual_col, N_arr_full)
        np.add.at(T_per_col, actual_col, T_arr_full)
        for s in col_idx:
            if T_per_col[s] > 0 or N_per_col[s] > 0:
                out[s] = negative_binomial_log_marginal(
                    N_per_col[s], T_per_col[s], a_eta, b_eta
                )
    return out


def update_eta_per_col_diagnostic(aa_a: np.ndarray, aa_b: np.ndarray, tau: np.ndarray,
                                     both_aa: np.ndarray, cls: np.ndarray, K_c: int,
                                     pi_class: np.ndarray, S: np.ndarray,
                                     a_eta: float, b_eta: float) -> np.ndarray:
    """Diagnostic: per-column eta posterior mean (a + N_acc) / (b + T̃).
    Not used in the SVI update path — eta is integrated out via the
    Negative-Binomial marginal in `per_column_log_marginal_class_specific`.
    Useful for visualization or to check rate heterogeneity across columns.
    """
    L = aa_a.shape[1]
    eta_new = np.zeros(L)
    Qs = []
    for c in range(K_c):
        Q = (S - np.diag(np.diag(S))) * pi_class[c][None, :]
        np.fill_diagonal(Q, -Q.sum(axis=1))
        Qs.append(Q)
    for s in range(L):
        c_s = int(cls[s])
        Q = Qs[c_s]; pi_s = pi_class[c_s]
        v = both_aa[:, s]
        if not v.any():
            eta_new[s] = a_eta / b_eta; continue
        N_acc = 0.0; T_tilde = 0.0
        for c_idx in np.flatnonzero(v):
            a = int(aa_a[c_idx, s]); b = int(aa_b[c_idx, s])
            t = float(tau[c_idx])
            N_c, T_c, _ = hr_per_cherry(a, b, t, Q, pi_s)
            N_acc += N_c; T_tilde += T_c
        eta_new[s] = posterior_eta_mean(N_acc, T_tilde, a_eta, b_eta)
    return eta_new


# --- SVI state container ---------------------------------------------------

@dataclass
class SVIState:
    K_c: int
    A: int
    pi_class: np.ndarray            # (K_c, A)
    potts_dp: Optional[PottsDPState]
    states_per_msa: list             # FamilyKState per MSA
    eta_per_msa: list                # (L,) np.ndarray per MSA
    # Hyperparams
    a_eta: float = 2.0
    b_eta: float = 2.0
    kappa_pi: float = 1.0
    alpha_c: float = 1.0
    alpha_H: float = 1.0


def em_warmup_site_classes(state: SVIState, per_family_data: list,
                                kappa_pi: float, pi_bar: np.ndarray,
                                n_iters: int,
                                rng: np.random.Generator,
                                tol: float = 1e-5,
                                n_seeds: int = 1,
                                verbose: bool = False) -> SVIState:
    """Pre-SVI deterministic soft EM warm-up on column → site-class.

    No Potts coupling, no partner moves. Soft posteriors throughout
    (no sampling); after convergence the hard cls are set to the MAP
    (argmax of the final soft posterior).

    Implementation: column counts from all families concatenated into
    one (L_total, A) tensor; E-step is one JIT'd JAX softmax; M-step is
    one JIT'd matmul + normalize. Per-iter cost is dominated by
    L_total × K_c × A flops on GPU — milliseconds.

    Symmetry-break: initial pi_class sampled from Dir(kappa_pi * pi_bar)
    per class. Soft EM from identical pi sits at a saddle point.

    Convergence: stop when max-class L1(delta pi_class) < tol.
    """
    import jax
    import jax.numpy as jnp
    from .tsb import stick_to_weights, update_betas_from_counts
    K_c = state.K_c
    A = state.A

    # Concatenate per-family per-column counts into one big (L_total, A).
    fam_ns = []
    fam_lengths = []
    for fd in per_family_data:
        L = fd['L']
        ba = fd['both_aa']
        aa_a = np.minimum(fd['aa_a'].astype(np.int64), 19)
        aa_b = np.minimum(fd['aa_b'].astype(np.int64), 19)
        ns = np.zeros((L, A), dtype=np.int64)
        for s in range(L):
            v = ba[:, s]
            if not v.any(): continue
            for col_arr in (aa_a, aa_b):
                np.add.at(ns[s], col_arr[v, s], 1)
        fam_ns.append(ns)
        fam_lengths.append(L)
    N_total = jnp.asarray(np.concatenate(fam_ns, axis=0), dtype=jnp.float64)
    boundaries = np.cumsum([0] + fam_lengths)              # for splitting q back

    pi_bar_j = jnp.asarray(pi_bar)
    prior_alpha_j = kappa_pi * pi_bar_j
    init_alpha = kappa_pi * pi_bar

    @jax.jit
    def em_step(N_total, pi, rho, prior_alpha):
        log_pi = jnp.log(jnp.clip(pi, 1e-12, None))
        log_rho = jnp.log(jnp.clip(rho, 1e-12, None))
        log_post = N_total @ log_pi.T + log_rho[None, :]
        q = jax.nn.softmax(log_post, axis=-1)
        soft_class_counts = q.sum(axis=0)
        n_per_class = q.T @ N_total
        post = prior_alpha[None, :] + n_per_class
        new_pi = post / post.sum(axis=-1, keepdims=True)
        delta_l1 = jnp.max(jnp.abs(new_pi - pi).sum(axis=-1))
        # Marginal data log-likelihood under the mixture:
        # log p(data) = Σ_s logsumexp_c [log_rho[c] + ns[s] @ log_pi[c]]
        log_lik = jax.scipy.special.logsumexp(log_post, axis=-1).sum()
        return new_pi, q, soft_class_counts, delta_l1, log_lik

    # Multi-seed loop: run soft EM from N_seeds independent Dirichlet inits,
    # log each fixed point, pick the best by data log-likelihood.
    best = None
    seed_summaries = []
    for seed_idx in range(n_seeds):
        # Independent Dirichlet sample for symmetry-breaking init.
        pi_init = np.stack([rng.dirichlet(init_alpha) for _ in range(K_c)], axis=0)
        pi = jnp.asarray(pi_init)
        rho = jnp.full(K_c, 1.0 / K_c)
        final_q = None
        last_log_lik = None
        for it in range(n_iters):
            new_pi, q, soft_counts, delta_l1, log_lik = em_step(
                N_total, pi, rho, prior_alpha_j
            )
            pi = new_pi
            new_betas = update_betas_from_counts(
                np.asarray(soft_counts), state.alpha_c, rng=rng, mode='map',
            )
            rho = jnp.asarray(stick_to_weights(new_betas))
            final_q = q
            last_log_lik = float(log_lik)
            if float(delta_l1) < tol:
                break
        n_iters_done = it + 1
        entropies = [-float((p * np.log2(np.clip(p, 1e-12, None))).sum())
                        for p in np.asarray(pi)]
        soft_counts_np = np.asarray(soft_counts)
        if verbose:
            print(f"  EM seed {seed_idx+1}/{n_seeds}: "
                    f"converged in {n_iters_done} iters, "
                    f"log_lik={last_log_lik:.1f}, "
                    f"soft_counts={[f'{c:.0f}' for c in soft_counts_np]}, "
                    f"pi entropies={[f'{e:.3f}' for e in entropies]}")
        seed_summaries.append(dict(
            seed_idx=seed_idx, log_lik=last_log_lik, n_iters=n_iters_done,
            pi=np.asarray(pi), rho=np.asarray(rho),
            soft_counts=soft_counts_np.tolist(),
            entropies=entropies,
        ))
        if best is None or last_log_lik > best['log_lik']:
            best = dict(pi=np.asarray(pi), rho=np.asarray(rho),
                          q=np.asarray(final_q), log_lik=last_log_lik,
                          seed_idx=seed_idx)

    if verbose:
        ranking = sorted(seed_summaries, key=lambda s: -s['log_lik'])
        print(f"\n  EM seed ranking by training log-lik:")
        for r in ranking:
            star = " (best)" if r['seed_idx'] == best['seed_idx'] else ""
            print(f"    seed {r['seed_idx']}: log_lik={r['log_lik']:.1f}, "
                    f"entropies={[f'{e:.3f}' for e in r['entropies']]}{star}")
        print(f"  Picking seed {best['seed_idx']} (log_lik={best['log_lik']:.1f})")

    state.pi_class = best['pi']
    cls_full = np.argmax(best['q'], axis=-1).astype(np.int32)
    for fam_idx in range(len(per_family_data)):
        s, e = int(boundaries[fam_idx]), int(boundaries[fam_idx + 1])
        state.states_per_msa[fam_idx].cls = cls_full[s:e]
    return state


def init_svi_state(per_family_data: list, K_c: int, A: int = 20,
                    init_pair_fraction: float = 0.0,
                    K_H_max: int | None = None,
                    use_side_potentials: bool = False,
                    rng: Optional[np.random.Generator] = None) -> SVIState:
    """Initialize SVIState: per-class pi at LG08, per-site eta at 1,
    random class assignments, optional random pair init for partition,
    Potts DP collapsed to a single atom.
    """
    if rng is None: rng = np.random.default_rng(0)
    pi_class = np.tile(np.asarray(PI_LG08), (K_c, 1))
    states = []; etas = []
    for fd in per_family_data:
        L = fd['L']
        n_pairs_init = int(L * init_pair_fraction / 2)
        st = init_random_K(fd['family'], L, K_c, n_pairs=n_pairs_init, rng=rng)
        states.append(st)
        etas.append(np.ones(L))
    mu_prior = np.zeros((A, A)); tau_prior = np.full((A, A), 4.0)
    # TSB: K_H_max = K_c(K_c+1)/2 atoms always allocated (truncated stick-
    # breaking). Symmetric with the site-class TSB; replaces the CRP-Gibbs
    # variant (init_potts_dp) which was sticky on the low-K side.
    from .potts_dp import init_potts_tsb
    potts_dp = init_potts_tsb(K_c=K_c, alpha_H=1.0, mu_prior=mu_prior,
                                tau_prior=tau_prior, rng=rng,
                                K_H_max=K_H_max,
                                use_side_potentials=use_side_potentials)
    return SVIState(K_c=K_c, A=A, pi_class=pi_class, potts_dp=potts_dp,
                    states_per_msa=states, eta_per_msa=etas)


# --- Phase B: pair-aware updates -------------------------------------------

def build_atom_log_P_cache(state: SVIState, unique_t: np.ndarray,
                             S: np.ndarray) -> np.ndarray:
    """Build a (K_H_active, K_c, K_c, n_t, A^2, A^2) cache of log
    transition matrices for every (atom, c_s, c_t, tau) combination.
    Memory: K_H * K_c^2 * n_t * 160000 floats — only viable for small
    K_H * K_c^2 * n_t. For our test corpus (~few atoms, K_c=2-3, ~250
    unique tau), this is ~10 GB at K_H=4, K_c=4. Use float32 if needed.

    For larger problems, build on demand inside the Gibbs sweep.
    """
    from .generator import (build_joint_Q_pair, joint_stationary_pair,
                              symmetrize_eigh, log_transition_matrices)
    K_H = state.potts_dp.atoms.shape[0]
    K_c = state.K_c
    A = state.A; A2 = A * A
    n_t = len(unique_t)
    log_P = np.zeros((K_H, K_c, K_c, n_t, A2, A2), dtype=np.float64)
    import jax.numpy as jnp
    unique_t_j = jnp.asarray(unique_t)
    for h in range(K_H):
        H = jnp.asarray(state.potts_dp.atoms[h])
        for c1 in range(K_c):
            pi1 = jnp.asarray(state.pi_class[c1])
            for c2 in range(K_c):
                pi2 = jnp.asarray(state.pi_class[c2])
                Q = build_joint_Q_pair(H, pi1, pi2, S=jnp.asarray(S))
                pi_j = joint_stationary_pair(H, pi1, pi2)
                Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
                log_P_h_c1_c2 = log_transition_matrices(unique_t_j, Lambda,
                                                          U_sym, sqrt_pij)
                log_P[h, c1, c2] = np.asarray(log_P_h_c1_c2)
    return log_P


def gather_pair_likelihood_for_atom(state: SVIState, atom_idx: int,
                                       per_family_data: list,
                                       unique_t: np.ndarray, inv_t: list,
                                       S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a Potts atom h, gather the (M, 4) observation tensor of all
    (cherry, edge) tuples whose class-pair is currently assigned to h.
    Returns (obs_array, sum_log_P_at_init).

    obs_array: (M, 4) [t_idx, class_pair_idx, start_state, end_state]
    where class_pair_idx is into the per-atom-c1c2 enumeration
    (only c1 <= c2 unique pairs that map to this atom).
    """
    K_c = state.K_c
    rows = []
    for fd, st in zip(per_family_data, state.states_per_msa):
        cls = st.cls
        for s in range(st.L):
            t = int(st.partner[s])
            if t <= s:
                continue
            c_s, c_t = int(cls[s]), int(cls[t])
            atom_for_pair = int(state.potts_dp.assignments[c_s, c_t])
            if atom_for_pair != atom_idx:
                continue
            valid = fd['both_aa'][:, s] & fd['both_aa'][:, t]
            if not valid.any():
                continue
            tau_idx_c = inv_t[fd['family']][valid]
            a_s = fd['aa_a'][valid, s].astype(np.int64); a_t = fd['aa_a'][valid, t].astype(np.int64)
            b_s = fd['aa_b'][valid, s].astype(np.int64); b_t = fd['aa_b'][valid, t].astype(np.int64)
            start = a_s * 20 + a_t; end = b_s * 20 + b_t
            cp = c_s * K_c + c_t  # ordered class-pair index
            cp_arr = np.full(int(valid.sum()), cp, dtype=np.int64)
            rows.append(np.column_stack([tau_idx_c, cp_arr, start, end]))
    if not rows:
        return np.zeros((0, 4), dtype=np.int64), 0.0
    return np.concatenate(rows, axis=0), 0.0


def build_log_P_cache_K_atoms(state: SVIState, unique_t: np.ndarray,
                                  S: np.ndarray) -> np.ndarray:
    """Build a (K_c, K_c, n_t, A^2, A^2) log P cache, indexed by class
    pair (c_s, c_t) and time. The H atom for each class pair is looked
    up via state.potts_dp.assignments[c_s, c_t]. Per-class pi is
    state.pi_class[c].
    """
    from .generator import (build_joint_Q_pair, joint_stationary_pair,
                              symmetrize_eigh, log_transition_matrices)
    import jax.numpy as jnp
    K_c = state.K_c
    A2 = state.A * state.A
    n_t = len(unique_t)
    log_P = np.zeros((K_c, K_c, n_t, A2, A2), dtype=np.float64)
    unique_t_j = jnp.asarray(unique_t)
    S_j = jnp.asarray(S)
    use_h = state.potts_dp.h_pairs is not None
    if use_h:
        from .potts_dp import canonical_pair_idx_table
        cp_idx_np, cp_swap_np = canonical_pair_idx_table(K_c)
    for c1 in range(K_c):
        pi1 = jnp.asarray(state.pi_class[c1])
        for c2 in range(K_c):
            pi2 = jnp.asarray(state.pi_class[c2])
            atom_idx = int(state.potts_dp.assignments[c1, c2])
            H = jnp.asarray(state.potts_dp.atoms[atom_idx])
            if use_h:
                k = int(cp_idx_np[c1, c2])
                swap = int(cp_swap_np[c1, c2])
                h_a = jnp.asarray(state.potts_dp.h_pairs[k, swap])
                h_b = jnp.asarray(state.potts_dp.h_pairs[k, 1 - swap])
                Q = build_joint_Q_pair(H, pi1, pi2, S=S_j, h_a=h_a, h_b=h_b)
                pi_j = joint_stationary_pair(H, pi1, pi2, h_a=h_a, h_b=h_b)
            else:
                Q = build_joint_Q_pair(H, pi1, pi2, S=S_j)
                pi_j = joint_stationary_pair(H, pi1, pi2)
            Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
            log_P_h = log_transition_matrices(unique_t_j, Lambda, U_sym, sqrt_pij)
            log_P[c1, c2] = np.asarray(log_P_h)
    return log_P


def _build_neg_log_post_for_atom(state: SVIState, atom_obs: np.ndarray,
                                    K_c: int, S_j, unique_t_j,
                                    mu_prior_j, tau_prior_j,
                                    pi_classes_j):
    """Build a JIT'd neg-log-post function for use in find_map_potts /
    laplace_component_diag for a particular atom's observation set."""
    from .generator import (build_joint_Q_pair, joint_stationary_pair,
                              symmetrize_eigh, log_transition_matrices)
    from .laplace_potts import _flat_to_sym, log_prior_pathwise
    import jax
    import jax.numpy as jnp
    if atom_obs.shape[0] == 0:
        return None
    t_idx = jnp.asarray(atom_obs[:, 0])
    cp_ord = atom_obs[:, 1]
    c1_arr = jnp.asarray((cp_ord // K_c).astype(np.int64))
    c2_arr = jnp.asarray((cp_ord % K_c).astype(np.int64))
    start = jnp.asarray(atom_obs[:, 2]); end = jnp.asarray(atom_obs[:, 3])

    def neg_log_post(H_flat):
        H_mat = _flat_to_sym(H_flat)
        def per_class_pair_log_P(pi1, pi2):
            Q = build_joint_Q_pair(H_mat, pi1, pi2, S=S_j)
            pi_j = joint_stationary_pair(H_mat, pi1, pi2)
            Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
            return log_transition_matrices(unique_t_j, Lambda, U_sym, sqrt_pij)
        log_P = jax.vmap(jax.vmap(per_class_pair_log_P, in_axes=(None, 0)),
                          in_axes=(0, None))(pi_classes_j, pi_classes_j)
        log_p_obs = log_P[c1_arr, c2_arr, t_idx, start, end]
        log_pr = log_prior_pathwise(H_mat, mu_prior_j, tau_prior_j)
        return -jnp.sum(log_p_obs) - log_pr
    return neg_log_post


def _gather_pair_obs(state: SVIState, c: int, cp: int,
                       per_family_data: list, inv_t: dict,
                       K_c: int) -> np.ndarray:
    """Gather observations for class-pair (c, c') across all MSAs."""
    rows = []
    for fd, st in zip(per_family_data, state.states_per_msa):
        cls = st.cls
        for s in range(st.L):
            t = int(st.partner[s])
            if t <= s: continue
            c_s, c_t = int(cls[s]), int(cls[t])
            if not ((c_s == c and c_t == cp) or (c_s == cp and c_t == c)):
                continue
            valid = fd['both_aa'][:, s] & fd['both_aa'][:, t]
            if not valid.any(): continue
            tau_idx_c = inv_t[fd['family']][valid]
            a_s = fd['aa_a'][valid, s].astype(np.int64)
            a_t = fd['aa_a'][valid, t].astype(np.int64)
            b_s = fd['aa_b'][valid, s].astype(np.int64)
            b_t = fd['aa_b'][valid, t].astype(np.int64)
            start = a_s * 20 + a_t; end = b_s * 20 + b_t
            cp_arr = np.full(int(valid.sum()), c_s * K_c + c_t, dtype=np.int64)
            rows.append(np.column_stack([tau_idx_c, cp_arr, start, end]))
    if not rows:
        return np.zeros((0, 4), dtype=np.int64)
    return np.concatenate(rows, axis=0)


def potts_tsb_sweep(state: SVIState, per_family_data: list,
                       unique_t: np.ndarray, inv_t: dict,
                       S: np.ndarray,
                       rng: np.random.Generator,
                       loss_kind: str = "exact") -> SVIState:
    """One TSB resample of class-pair → atom assignments + stick-weight
    update. Replaces the CRP-Gibbs sweep at K_H_max = K_c(K_c+1)/2.

    Procedure:
    1. For each unordered (c, c') class-pair, gather its observed cherry
       edges from all training MSAs.
    2. For each atom h in {0..K_H_max-1}, evaluate sum-log-likelihood of
       those edges under H_h (using existing_atom_log_lik / its ELBO twin).
    3. Resample (c, c') -> h via Categorical(rho * lik) — uses the TSB
       stick weights as the prior over atom slots.
    4. Conjugate Beta-posterior update of stick proportions from the new
       per-atom counts (`tsb_update_rho`).

    Stays internally consistent for both `loss_kind="exact"` and
    `loss_kind="elbo"` (the two variants of `existing_atom_log_lik`).
    """
    if loss_kind == "exact":
        from .laplace_potts_v2 import existing_atom_log_lik, pad_obs
    elif loss_kind == "elbo":
        from .laplace_potts_v2 import pad_obs
        from .loss_elbo import existing_atom_log_lik_elbo as existing_atom_log_lik
    else:
        raise ValueError(f"unknown loss_kind={loss_kind!r}")

    from .potts_dp import (_class_pair_idx, tsb_resample_assignments,
                                tsb_update_rho)
    K_c = state.K_c
    K_H_max = state.potts_dp.atoms.shape[0]
    cp_table, pairs = _class_pair_idx(K_c)

    # Pre-gather per-(c, c') padded obs.
    pair_obs_dict = {}
    M_max = 0
    for c, cp in pairs:
        po = _gather_pair_obs(state, c, cp, per_family_data, inv_t, K_c)
        pair_obs_dict[(c, cp)] = po
        if po.shape[0] > M_max:
            M_max = po.shape[0]
    if M_max == 0:
        return state

    pi_j = jnp.asarray(state.pi_class)
    S_j = jnp.asarray(S)
    t_j = jnp.asarray(unique_t)

    # When side-potentials are active, build per-(c1, c2) h_a_table /
    # h_b_table from h_pairs once. existing_atom_log_lik takes them as
    # optional args; per-class-pair scoring routes through these.
    h_a_table_j = h_b_table_j = None
    if state.potts_dp.h_pairs is not None:
        from .potts_dp import canonical_pair_idx_table
        cp_idx_np, cp_swap_np = canonical_pair_idx_table(K_c)
        cp_idx_j = jnp.asarray(cp_idx_np)
        cp_swap_j = jnp.asarray(cp_swap_np)
        h_pairs_j = jnp.asarray(state.potts_dp.h_pairs)
        h_a_table_j = h_pairs_j[cp_idx_j, cp_swap_j]
        h_b_table_j = h_pairs_j[cp_idx_j, 1 - cp_swap_j]

    # log_lik[h, c, c'] = sum-log-likelihood of class-pair (c, c')'s obs
    # under atom h, with class-pair pi (pi_class[c], pi_class[c']).
    log_lik = np.zeros((K_H_max, K_c, K_c), dtype=np.float64)
    for c, cp in pairs:
        po = pair_obs_dict[(c, cp)]
        if po.shape[0] == 0:
            continue
        obs_padded, valid = pad_obs(po, M_max)
        obs_j = jnp.asarray(obs_padded); mask_j = jnp.asarray(valid)
        for h in range(K_H_max):
            ll = float(existing_atom_log_lik(
                jnp.asarray(state.potts_dp.atoms[h]),
                obs_j, mask_j, pi_j, S_j, t_j,
                h_a_table=h_a_table_j, h_b_table=h_b_table_j,
            ))
            log_lik[h, c, cp] = ll
            log_lik[h, cp, c] = ll

    state.potts_dp = tsb_resample_assignments(state.potts_dp, log_lik, rng)
    state.potts_dp = tsb_update_rho(state.potts_dp, rng, mode="sample")
    return state


def potts_dp_crp_sweep(state: SVIState, per_family_data: list,
                         unique_t: np.ndarray, inv_t: dict,
                         S: np.ndarray, mu_prior: np.ndarray,
                         tau_prior: np.ndarray,
                         rng: np.random.Generator,
                         n_laplace_steps: int = 20,
                         loss_kind: str = "exact") -> SVIState:
    """One full pass of CRP-Gibbs over all unordered class-pairs.

    `loss_kind` selects the loss family for both branches of the
    new-vs-existing-atom score:
    - "exact": JIT-hoisted exact-log-P primitives from
                `laplace_potts_v2` (existing_atom_log_lik,
                laplace_component_diag_jit).
    - "elbo":  Holmes-Rubin closed-form ELBO at constant Q_hat with
                damped (bar_p_1, bar_p_2) fixed-point inner iteration
                (`loss_elbo.existing_atom_log_lik_elbo` and
                `loss_elbo.laplace_component_diag_jit_elbo`). The CRP
                comparison stays internally consistent — both branches
                use ELBO — but the absolute scores are biased by the
                Jensen gap relative to the exact log-P.

    Padding to M_max keeps the JAX trace cached across class-pairs
    within one sweep.
    """
    if loss_kind == "exact":
        from .laplace_potts_v2 import (existing_atom_log_lik,
                                           laplace_component_diag_jit,
                                           laplace_log_evidence_v2 as
                                               laplace_log_evidence,
                                           pad_obs)
    elif loss_kind == "elbo":
        # Default to the chunked-M variant for ELBO: bounded peak memory
        # vs the linearize-cached variant which OOMs at K_c > 1 / large M.
        from .loss_elbo import (existing_atom_log_lik_elbo as
                                    existing_atom_log_lik,
                                    laplace_component_diag_jit_elbo_chunked as
                                    laplace_component_diag_jit,
                                    laplace_log_evidence_elbo as
                                    laplace_log_evidence)
        from .laplace_potts_v2 import pad_obs
    else:
        raise ValueError(f"unknown loss_kind={loss_kind!r}")
    from .potts_dp import _class_pair_idx, gibbs_step_assignment
    cp_table, pairs = _class_pair_idx(state.K_c)
    K_c = state.K_c

    # Pre-gather all pair observations and find M_max for padding.
    pair_obs_dict = {}
    M_max = 0
    for k, (c, cp) in enumerate(pairs):
        po = _gather_pair_obs(state, c, cp, per_family_data, inv_t, K_c)
        pair_obs_dict[(c, cp)] = po
        if po.shape[0] > M_max:
            M_max = po.shape[0]
    if M_max == 0:
        return state

    # Pad and convert to JAX-friendly arrays once per CRP sweep.
    pi_class_arr = state.pi_class.copy()

    order = rng.permutation(len(pairs))
    for k in order:
        c, cp = pairs[k]
        po = pair_obs_dict[(c, cp)]
        if po.shape[0] == 0:
            continue
        obs_padded, valid_mask = pad_obs(po, M_max)

        def make_log_pair_lik_fn(obs_padded=obs_padded, valid_mask=valid_mask):
            obs_j = jnp.asarray(obs_padded); mask_j = jnp.asarray(valid_mask)
            pi_j = jnp.asarray(pi_class_arr); S_j = jnp.asarray(S)
            t_j = jnp.asarray(unique_t)
            def fn(H_atom):
                return float(existing_atom_log_lik(
                    jnp.asarray(H_atom), obs_j, mask_j, pi_j, S_j, t_j
                ))
            return fn

        log_pair_lik_fn = make_log_pair_lik_fn()

        H_init_for_new = state.potts_dp.atoms[
            int(state.potts_dp.assignments[c, cp])
        ]
        def new_atom_marginal_fn(obs_padded=obs_padded, valid_mask=valid_mask,
                                    H_init=H_init_for_new):
            comp = laplace_component_diag_jit(
                obs_padded, valid_mask, pi_class_arr, S, mu_prior,
                tau_prior, unique_t, H_init,
                n_steps=n_laplace_steps, lr=0.05
            )
            return laplace_log_evidence(comp), comp.H_hat

        state.potts_dp = gibbs_step_assignment(
            state.potts_dp, c, cp, log_pair_lik_fn,
            new_atom_marginal_fn, rng
        )
    return state


def update_potts_atoms_jit(state: SVIState, per_family_data: list,
                              unique_t: np.ndarray, inv_t: dict,
                              S: np.ndarray, mu_prior: np.ndarray,
                              tau_prior: np.ndarray,
                              n_steps: int = 30, lr: float = 0.05,
                              loss_kind: str = "exact",
                              h_prior_tau: float = 4.0) -> SVIState:
    """JIT-hoisted version: per-atom Adam MAP.

    `loss_kind` selects the gradient routine:
    - "exact": full 400-state log P pair loss from `laplace_potts_v2`.
    - "elbo":  Holmes-Rubin closed-form ELBO at constant Q_hat with
                damped (bar_p_1, bar_p_2) fixed-point inner iteration
                (`loss_elbo.grad_fn_elbo`). Strict lower bound on log P;
                gradient is biased only insofar as the bound is loose.

    Pads each atom's observations to a common M_max so JAX caches the
    trace across atoms within one outer iteration.
    """
    from .laplace_potts_v2 import pad_obs
    if loss_kind == "exact":
        from .laplace_potts_v2 import grad_fn
    elif loss_kind == "elbo":
        from .loss_elbo import grad_fn_elbo as grad_fn
    else:
        raise ValueError(f"unknown loss_kind={loss_kind!r}")
    import optax
    K_H = state.potts_dp.atoms.shape[0]
    K_c = state.K_c

    # Gather per-atom observations
    atom_obs_list = []
    for h in range(K_H):
        rows = []
        for fd, st in zip(per_family_data, state.states_per_msa):
            cls = st.cls
            for s in range(st.L):
                t = int(st.partner[s])
                if t <= s: continue
                c_s, c_t = int(cls[s]), int(cls[t])
                if int(state.potts_dp.assignments[c_s, c_t]) != h:
                    continue
                valid = fd['both_aa'][:, s] & fd['both_aa'][:, t]
                if not valid.any(): continue
                tau_idx_c = inv_t[fd['family']][valid]
                a_s = fd['aa_a'][valid, s].astype(np.int64)
                a_t = fd['aa_a'][valid, t].astype(np.int64)
                b_s = fd['aa_b'][valid, s].astype(np.int64)
                b_t = fd['aa_b'][valid, t].astype(np.int64)
                start = a_s * 20 + a_t; end = b_s * 20 + b_t
                cp_arr = np.full(int(valid.sum()), c_s * K_c + c_t, dtype=np.int64)
                rows.append(np.column_stack([tau_idx_c, cp_arr, start, end]))
        if rows:
            atom_obs_list.append(np.concatenate(rows, axis=0))
        else:
            atom_obs_list.append(np.zeros((0, 4), dtype=np.int64))

    M_max = max((po.shape[0] for po in atom_obs_list), default=0)
    if M_max == 0:
        return state

    pi_j = jnp.asarray(state.pi_class)
    S_j = jnp.asarray(S)
    mu_j = jnp.asarray(mu_prior); tau_j = jnp.asarray(tau_prior)
    t_j = jnp.asarray(unique_t)
    optimizer = optax.adam(lr)

    use_h = state.potts_dp.h_pairs is not None
    if use_h:
        # Joint Adam over (H_flat, h_pairs) when side potentials enabled.
        from .potts_dp import (canonical_pair_idx_table,
                                  canonical_pair_is_diag,
                                  symmetrize_h_pairs_diag)
        from .laplace_potts_v2 import grad_fn_with_h
        cp_idx_np, cp_swap_np = canonical_pair_idx_table(K_c)
        cp_idx_j = jnp.asarray(cp_idx_np)
        cp_swap_j = jnp.asarray(cp_swap_np)
        is_diag_pair_j = jnp.asarray(canonical_pair_is_diag(K_c))
        # Defensive: ensure h_pairs respects the self-pair tying invariant
        # before we hand it to the JIT'd Adam loop. (It is also re-enforced
        # at the end of the sweep via symmetrize_h_pairs_diag.)
        symmetrize_h_pairs_diag(state.potts_dp.h_pairs, K_c)
        h_pairs_j = jnp.asarray(state.potts_dp.h_pairs)
        # Each atom's Adam loop sees 1/K_H of the total Gaussian prior on
        # h_pairs (so summed across all K_H atom calls, the full prior is
        # applied once). Approximate but simple; overcounts on h_pairs that
        # aren't owned by this atom (gradient on them is zero so the prior
        # pull is shared by all K_H atoms equally — net effect over a sweep
        # is single-counted prior on each h_pair).
        h_share = 1.0 / max(K_H, 1)

    new_atoms = []
    for h in range(K_H):
        po = atom_obs_list[h]
        if po.shape[0] == 0:
            new_atoms.append(state.potts_dp.atoms[h]); continue
        obs_padded, mask = pad_obs(po, M_max)
        from .laplace_potts import _sym_to_flat, _flat_to_sym
        H_flat = jnp.asarray(_sym_to_flat(jnp.asarray(state.potts_dp.atoms[h])))
        obs_j = jnp.asarray(obs_padded); mask_j = jnp.asarray(mask)
        if use_h:
            params = (H_flat, h_pairs_j)
            opt_state = optimizer.init(params)
            for _ in range(n_steps):
                g_H, g_h_pairs = grad_fn_with_h(
                    params[0], params[1], obs_j, mask_j, pi_j, S_j, mu_j,
                    tau_j, t_j, cp_idx_j, cp_swap_j, is_diag_pair_j,
                    h_prior_tau, h_share,
                )
                updates, opt_state = optimizer.update((g_H, g_h_pairs), opt_state)
                params = optax.apply_updates(params, updates)
            H_flat, h_pairs_j = params
            new_atoms.append(np.asarray(_flat_to_sym(H_flat)))
        else:
            opt_state = optimizer.init(H_flat)
            for _ in range(n_steps):
                g = grad_fn(H_flat, obs_j, mask_j, pi_j, S_j, mu_j, tau_j, t_j)
                updates, opt_state = optimizer.update(g, opt_state)
                H_flat = optax.apply_updates(H_flat, updates)
            new_atoms.append(np.asarray(_flat_to_sym(H_flat)))
    state.potts_dp.atoms = np.stack(new_atoms)
    if use_h:
        # np.asarray on a JAX array returns a read-only view; np.array
        # copies to a writable buffer so symmetrize_h_pairs_diag can
        # mutate it in place.
        state.potts_dp.h_pairs = np.array(h_pairs_j)
        # Re-enforce the self-pair tying invariant on the saved state. Inside
        # loss_fn_with_h slot 1 on diagonals receives no gradient, so the
        # tied condition is preserved by Adam — but defensive projection
        # keeps the saved checkpoint exactly on the constraint surface.
        symmetrize_h_pairs_diag(state.potts_dp.h_pairs, K_c)
    return state


# ----------------------------------------------------------------------------
# Legacy single-atom MAP (kept for the v1 SVI loop in case of fallback)
# ----------------------------------------------------------------------------

def update_potts_atom_laplace_jit(state: SVIState, atom_idx: int,
                                     per_family_data: list,
                                     unique_t: np.ndarray, inv_t: dict,
                                     S: np.ndarray, mu_prior: np.ndarray,
                                     tau_prior: np.ndarray,
                                     n_steps: int = 30, lr: float = 0.05
                                     ) -> np.ndarray:
    """Single-atom Laplace MAP via JAX gradient on H_flat. Builds the
    pair-Q on-demand inside the JIT'd loss to avoid the (K_c, K_c, n_t,
    A^2, A^2) cache. Per-step cost: K_c^2 eigh of A^2 x A^2 matrices,
    fine when K_c is small.

    Returns the updated H atom (A, A) symmetric.
    """
    from .generator import (build_joint_Q_pair, joint_stationary_pair,
                              symmetrize_eigh, log_transition_matrices)
    from .laplace_potts import _flat_to_sym, log_prior_pathwise, _sym_to_flat
    import jax
    import jax.numpy as jnp
    import optax

    # Gather cherry observations for this atom
    obs, _ = gather_pair_likelihood_for_atom(
        state, atom_idx, per_family_data, unique_t, inv_t, S
    )
    if obs.shape[0] == 0:
        return state.potts_dp.atoms[atom_idx]

    K_c = state.K_c
    t_idx = jnp.asarray(obs[:, 0])
    cp_ord = obs[:, 1]
    c1_arr = jnp.asarray((cp_ord // K_c).astype(np.int64))
    c2_arr = jnp.asarray((cp_ord % K_c).astype(np.int64))
    start = jnp.asarray(obs[:, 2]); end = jnp.asarray(obs[:, 3])
    unique_t_j = jnp.asarray(unique_t)
    S_j = jnp.asarray(S)
    mu_j = jnp.asarray(mu_prior); tau_j = jnp.asarray(tau_prior)
    pi_classes_j = jnp.asarray(state.pi_class)
    n_obs = obs.shape[0]

    def neg_log_post(H_flat):
        H_mat = _flat_to_sym(H_flat)
        # Build per-(c1, c2) log_P cache via vmap
        def per_class_pair_log_P(pi1, pi2):
            Q = build_joint_Q_pair(H_mat, pi1, pi2, S=S_j)
            pi_j = joint_stationary_pair(H_mat, pi1, pi2)
            Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
            return log_transition_matrices(unique_t_j, Lambda, U_sym, sqrt_pij)
        # log_P[c1, c2, t, start, end]: shape (K_c, K_c, n_t, A^2, A^2)
        log_P = jax.vmap(jax.vmap(per_class_pair_log_P, in_axes=(None, 0)),
                          in_axes=(0, None))(pi_classes_j, pi_classes_j)
        # Gather observations
        log_p_obs = log_P[c1_arr, c2_arr, t_idx, start, end]
        log_pr = log_prior_pathwise(H_mat, mu_j, tau_j)
        return -jnp.sum(log_p_obs) - log_pr

    # JIT the loss + grad
    grad_fn = jax.jit(jax.grad(neg_log_post))

    H_flat = jnp.asarray(_sym_to_flat(jnp.asarray(state.potts_dp.atoms[atom_idx])))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(H_flat)
    for _ in range(n_steps):
        g = grad_fn(H_flat)
        updates, opt_state = optimizer.update(g, opt_state)
        H_flat = optax.apply_updates(H_flat, updates)
    return np.asarray(_flat_to_sym(H_flat))
