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
from .lg08 import PI_LG08, S_LG08_F81
from .partition_K import FamilyKState, init_random_K
from .potts_dp import PottsDPState, alpha_H_map_update, init_potts_dp
from .secret_destination import (em_pi_update, expected_ghost_counts,
                                   dirichlet_log_marginal,
                                   dirichlet_posterior_mean,
                                   dirichlet_posterior_mode)


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
                     n_iters: int = 4, sparse: bool = False) -> np.ndarray:
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
    combine = dirichlet_posterior_mode if sparse else dirichlet_posterior_mean
    pi_new = pi_class_curr.copy()
    for c in range(K_c):
        if dwell_total[c].sum() < 1e-12:
            continue
        pi_curr = pi_new[c].copy()
        for _ in range(n_iters):
            ghost = expected_ghost_counts(pi_curr, S, dwell_total[c], eta=eta_one)
            pi_post = combine(prior_alpha, real_counts[c], ghost)
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
    # A1 correctness switch (2026-06-27). When True the joint pair
    # stationary used at every coupling site (sampling, Laplace MAP,
    # log_P caches) is Sinkhorn-corrected via
    # generator.joint_stationary_pair_a1; row/col marginals equal
    # (pi^(c1), pi^(c2)) and the augmented generator is reversible
    # across indel events. Released pre-2026-06-27 checkpoints were
    # trained with reversible=False (the A3 free-h formulation); load
    # them via the legacy / --pre-sinkhorn pathway.
    reversible: bool = True

    # ---- Dynamic-field variant state (Phase D.4, 2026-06-28) -------------
    # When `coupling_variant == 'dynamic_field'`, the substitution-side
    # state lives in `dyn_field` instead of `potts_dp`. `coupling`
    # dispatches accordingly.
    dyn_field: Optional[object] = None         # DynamicFieldState | None
    coupling_variant: str = 'potts'            # 'potts' | 'dynamic_field'

    # ---- CouplingModel adapter (Phase B, 2026-06-28) ----------------------
    # `state.coupling` returns a CouplingModel for the variant this state
    # represents. The variant is selected by `coupling_variant`, defaulting
    # to 'potts' for backward compatibility with all pre-D.4 checkpoints
    # and call sites. New code can call
    # `state.coupling.build_M_tensor(t, pi_c=pi_c, ...)` for either
    # variant.
    @property
    def coupling(self):
        if self.coupling_variant == 'dynamic_field':
            from .coupling.dynfield import DynamicFieldCouplingModel
            return DynamicFieldCouplingModel.from_svi_state(self)
        # Default: Potts variant.
        from .coupling.potts import PottsCouplingModel
        return PottsCouplingModel.from_svi_state(self)


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
    # Under PSWM: replace hard AA counts with the expected count under
    # the soft PSWM distributions at each branch endpoint. Equivalent to
    # the hard path when PSWMs are delta functions.
    fam_ns = []
    fam_lengths = []
    for fd in per_family_data:
        L = fd['L']
        ba = fd['both_aa']
        if 'pswm_a' in fd and 'pswm_b' in fd:
            ba_f = ba.astype(np.float64)                    # (C, L)
            pswm_a = np.asarray(fd['pswm_a'], dtype=np.float64)   # (C, L, A)
            pswm_b = np.asarray(fd['pswm_b'], dtype=np.float64)
            ns = ((pswm_a * ba_f[..., None]).sum(axis=0)
                    + (pswm_b * ba_f[..., None]).sum(axis=0))   # (L, A)
        else:
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
    from .potts_dp import init_potts_tsb
    potts_dp = init_potts_tsb(K_c=K_c, alpha_H=1.0, mu_prior=mu_prior,
                                tau_prior=tau_prior, rng=rng,
                                K_H_max=K_H_max)
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
                              build_joint_Q_pair_a1, joint_stationary_pair_a1,
                              symmetrize_eigh, log_transition_matrices)
    _build_Q = build_joint_Q_pair_a1 if state.reversible else build_joint_Q_pair
    _stat = joint_stationary_pair_a1 if state.reversible else joint_stationary_pair
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
                Q = _build_Q(H, pi1, pi2, S=jnp.asarray(S))
                pi_j = _stat(H, pi1, pi2)
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

    Stored as float32 to halve memory (3.3 GB -> 1.6 GB at K_c=16,
    n_t=10). Per-cherry log-likelihood accumulation in _precompute_pair_LL
    upcasts to float64 for stability, so the only effective loss is
    ~7-digit precision on each transition log-probability — well below
    the per-cherry MCMC noise floor.
    """
    from .generator import (build_joint_Q_pair, joint_stationary_pair,
                              build_joint_Q_pair_a1, joint_stationary_pair_a1,
                              symmetrize_eigh, log_transition_matrices)
    _build_Q = build_joint_Q_pair_a1 if state.reversible else build_joint_Q_pair
    _stat = joint_stationary_pair_a1 if state.reversible else joint_stationary_pair
    import jax.numpy as jnp
    K_c = state.K_c
    A2 = state.A * state.A
    n_t = len(unique_t)
    log_P = np.zeros((K_c, K_c, n_t, A2, A2), dtype=np.float32)
    unique_t_j = jnp.asarray(unique_t)
    S_j = jnp.asarray(S)
    for c1 in range(K_c):
        pi1 = jnp.asarray(state.pi_class[c1])
        for c2 in range(K_c):
            pi2 = jnp.asarray(state.pi_class[c2])
            atom_idx = int(state.potts_dp.assignments[c1, c2])
            H = jnp.asarray(state.potts_dp.atoms[atom_idx])
            Q = _build_Q(H, pi1, pi2, S=S_j)
            pi_j = _stat(H, pi1, pi2)
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
                              build_joint_Q_pair_a1, joint_stationary_pair_a1,
                              symmetrize_eigh, log_transition_matrices)
    _build_Q = build_joint_Q_pair_a1 if state.reversible else build_joint_Q_pair
    _stat = joint_stationary_pair_a1 if state.reversible else joint_stationary_pair
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
            Q = _build_Q(H_mat, pi1, pi2, S=S_j)
            pi_j = _stat(H_mat, pi1, pi2)
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

    Short-circuit at K_H_max=1: with only one atom to pick, the
    Categorical(rho * lik) resample over atom slots is degenerate —
    every class-pair must be assigned to atom 0. The whole loop
    (K_c(K_c+1)/2 outer obs gathers × K_H=1 existing_atom_log_lik
    calls, each of which is K_c²-scaling inside via the vmap over
    class-pairs) is wasted work whose only output is unused, and the
    JAX transient allocations along the K_c⁴ inner expansion are
    plausibly what triggered the K=16 OOM. Skip entirely when there's
    nothing to sample.
    """
    K_H_max = state.potts_dp.atoms.shape[0]
    if K_H_max == 1:
        return state

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

    Short-circuit at K_H_max=1: with only one atom and no new-atom
    branch worth proposing (alpha_H * marg_new vs n_existing *
    marg_existing would always pick the existing one without K_c²
    of new evidence), the entire sweep is wasted work. Skip.
    """
    if state.potts_dp.atoms.shape[0] == 1:
        return state

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
                              loss_kind: str = "exact") -> SVIState:
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

    new_atoms = []
    for h in range(K_H):
        po = atom_obs_list[h]
        if po.shape[0] == 0:
            new_atoms.append(state.potts_dp.atoms[h]); continue
        obs_padded, mask = pad_obs(po, M_max)
        from .laplace_potts import _sym_to_flat, _flat_to_sym
        H_flat = jnp.asarray(_sym_to_flat(jnp.asarray(state.potts_dp.atoms[h])))
        obs_j = jnp.asarray(obs_padded); mask_j = jnp.asarray(mask)
        opt_state = optimizer.init(H_flat)
        for _ in range(n_steps):
            g = grad_fn(H_flat, obs_j, mask_j, pi_j, S_j, mu_j, tau_j, t_j)
            updates, opt_state = optimizer.update(g, opt_state)
            H_flat = optax.apply_updates(H_flat, updates)
        new_atoms.append(np.asarray(_flat_to_sym(H_flat)))
    state.potts_dp.atoms = np.stack(new_atoms)
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
                              build_joint_Q_pair_a1, joint_stationary_pair_a1,
                              symmetrize_eigh, log_transition_matrices)
    _build_Q = build_joint_Q_pair_a1 if state.reversible else build_joint_Q_pair
    _stat = joint_stationary_pair_a1 if state.reversible else joint_stationary_pair
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
            Q = _build_Q(H_mat, pi1, pi2, S=S_j)
            pi_j = _stat(H_mat, pi1, pi2)
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


# ===========================================================================
# Phase D.4: dynamic-latent-field variant training
# ===========================================================================
#
# `init_svi_state_dynfield` initializes an SVIState carrying a
# DynamicFieldState in place of PottsDPState. `train_dynfield_one_iter`
# does one outer iter of soft-EM atom updates: per-cherry sufficient
# stats accumulation -> Dirichlet update on pi_field -> TSB Beta MAP on
# rho. The orchestration around it (partition Gibbs, EM warmup,
# per-class pi update) stays in the experiment script (exp2_pfam_v2.py
# or a parallel dynfield trainer); this function is the variant-specific
# atom-update step.
#
# Inputs to `train_dynfield_one_iter` are precomputed cherry
# observations:
#
#   coupled_cherries: list of (c_i, c_j, X_i, X_j, Y_i, Y_j, t) -- one
#     per coupled column pair per cherry.
#   uncoupled_cherries: list of (c, X_i, Y_i, t) -- one per uncoupled
#     column per cherry.
#
# The dispatcher pulls these from FamilyKState + per_family_data; see
# the Potts orchestration in exp2_pfam_v2.py for the analogous data
# preparation steps.
# ===========================================================================

def init_svi_state_dynfield(per_family_data: list, K_c: int, A: int = 20,
                              L_max: int = 8, alpha_field: float = 1.0,
                              rho_chain: float = 1.0,
                              K_a: int = 16, alpha_arch: float = 1.0,
                              alpha_prior: float = 1.0,
                              rng: Optional[np.random.Generator] = None,
                              ) -> SVIState:
    """Initialize an SVIState for the dynamic-latent-field variant.

    Mirrors `init_svi_state` for the Potts variant: per-class pi at
    LG08, per-site eta at 1, random class assignments. Archetype state
    is initialised from a TSB(alpha_arch) prior with K_a archetype
    slots; arch_assignment maps each (c, theta) to a random archetype,
    and pi_field is materialised as pi_archetype[arch_assignment, :].

    `rho` is initialised uniform (1/L_max each); the TSB update will
    sharpen it as clusters get assigned.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    from .lg08 import PI_LG08
    from .coupling.dynfield.state import DynamicFieldState
    from .coupling.dynfield.archetypes import init_archetype_state

    pi_class = np.tile(np.asarray(PI_LG08), (K_c, 1))

    # Archetype init (mandatory).
    pi_arch, arch_assignment, rho_arch, tsb_betas_arch = init_archetype_state(
        K_c=K_c, L_max=L_max, A=A, K_a=K_a, rng=rng,
        alpha_prior=alpha_prior, alpha_arch=alpha_arch)
    pi_field = pi_arch[arch_assignment]

    rho = np.full(L_max, 1.0 / L_max, dtype=np.float64)
    tsb_betas = None
    pi_class_eff = np.einsum('t,cta->ca', rho, pi_field)

    dyn_field = DynamicFieldState(
        K_c=K_c, A=A,
        pi_field=pi_field, rho=rho, tsb_betas=tsb_betas,
        alpha_field=alpha_field,
        pi_class=pi_class_eff,
        rho_chain=rho_chain,
        pi_archetype=pi_arch,
        arch_assignment=arch_assignment,
        rho_arch=rho_arch,
        tsb_betas_arch=tsb_betas_arch,
        alpha_arch=alpha_arch,
    )

    # FamilyKState init (random partition or initialise from data).
    from .partition_K import init_random_K
    states_per_msa = [
        init_random_K(family=fd.get('family', f'fam_{idx}'),
                       L=fd['L'], K=K_c,
                       n_pairs=fd.get('K', fd['L'] // 4),
                       rng=rng)
        for idx, fd in enumerate(per_family_data)]
    eta_per_msa = [np.ones(fd['L'], dtype=np.float64)
                    for fd in per_family_data]

    return SVIState(
        K_c=K_c, A=A,
        pi_class=pi_class_eff, potts_dp=None,
        states_per_msa=states_per_msa,
        eta_per_msa=eta_per_msa,
        dyn_field=dyn_field,
        coupling_variant='dynamic_field',
    )


def train_dynfield_one_iter(state: SVIState,
                              clusters: list,
                              *, alpha_prior: float = 1.0,
                              rng: 'np.random.Generator | None' = None,
                              use_hr_mstep: bool = False,
                              rho_chain_prior_a: float = 1.5,
                              rho_chain_prior_b: float = 5.0,
                              bins_per_cluster: 'list | None' = None,
                              weight_per_cluster: 'list | None' = None,
                              do_split_merge: bool = False,
                              split_merge_dead_threshold: float = 0.01,
                              split_merge_tvd_threshold: float = 0.05,
                              split_merge_max_moves: int = 2,
                              ) -> tuple[SVIState, dict]:
    """One outer iter of dynfield soft-EM atom updates.

    Steps:
      1. Accumulate soft-EM sufficient stats from all cluster observations.
      2. Update pi_field via Dirichlet-conjugate posterior mean.
      3. Update rho via TSB Beta MAP from cluster-theta counts.
      4. Refresh state.pi_class = field-marginal stationary.

    Args:
      state: SVIState with coupling_variant='dynamic_field'.
      clusters: list of (classes, X_obs, Y_obs, t) tuples; each entry
        is a cluster of arbitrary size m >= 1. classes, X_obs, Y_obs are
        1-D int arrays of length m (or python lists / tuples).
      alpha_prior: Dirichlet base measure concentration on pi_field.

    Returns:
      (state_new, info_dict). info_dict has keys 'log_lik_total',
      'n_clusters'.
    """
    if state.coupling_variant != 'dynamic_field':
        raise ValueError(
            f"train_dynfield_one_iter requires coupling_variant="
            f"'dynamic_field', got {state.coupling_variant!r}")
    from .coupling.dynfield import updates as _up
    from .coupling.dynfield import DynamicFieldCouplingModel

    K_c = state.K_c
    L_max = state.dyn_field.L_max
    model = DynamicFieldCouplingModel.from_svi_state(state)

    if clusters:
        c_obs = [(classes, X, Y) for (classes, X, Y, _) in clusters]
        c_t = np.asarray([t for (_, _, _, t) in clusters], dtype=np.float64)
        stats = _up.accumulate_cluster_stats_soft(
            model, c_obs, t_per_cluster=c_t)
    else:
        c_obs = []
        c_t = np.zeros(0, dtype=np.float64)
        stats = {'N': np.zeros((K_c, L_max, state.A)),
                  'r': np.zeros(L_max), 'n_clust': 0, 'log_lik': 0.0}

    # 2. TSB Beta MAP on rho (field-atom stick-breaking weights).
    tsb_new, rho_new = _up.update_rho_tsb(model, stats['r'], mode='map')
    _up.apply_updates_inplace(
        model, tsb_betas_new=tsb_new, rho_new=rho_new)
    state.dyn_field = model.dyn_field
    state.pi_class = model.pi_class

    # 3. Archetype update.
    #
    # Two paths:
    #  (a) use_hr_mstep=False: sample arch_assignment from the
    #      Categorical posterior on N[c, theta, ·]; Dirichlet posterior
    #      mean on pi_archetype.
    #  (b) use_hr_mstep=True (default in the archetype workflow): same
    #      arch_assignment sample, but pi_archetype via GTR fixed-point
    #      on HR sufficient stats (V, U, W) accumulated corpus-wide by
    #      hr_cluster_stats, and rho_chain via closed-form Gamma
    #      posterior. Requires a second pass over clusters to accumulate
    #      HR stats.
    hr_info: dict = {}
    if True:  # archetype path is now the only path
        if rng is None:
            rng = np.random.default_rng()
        if use_hr_mstep:
            from .coupling.dynfield.archetypes import (
                _update_rho_arch_tsb_expected, update_rho_arch_tsb)
            from .coupling.dynfield.hr import (
                update_pi_archetype_gtr, update_rho_chain_gamma)
            from .coupling.dynfield.hr_jax import (
                accumulate_cluster_stats_hr_jax,
                accumulate_cluster_stats_hr_jax_gammaI,
                aggregate_by_arch)
            from .coupling.dynfield.archetypes import (
                soft_arch_posterior, _update_rho_arch_tsb_expected)
            from tkfdp.lg08 import S_LG08
            S = np.asarray(S_LG08, dtype=np.float64)
            # Stochastic-EM E-step (single-sweep, no EMA):
            # 1. HR pass -> per-(c, θ) V, U, W under current arch_assignment
            #    and per-arch pi_arch (approximation; strictly, the
            #    trajectory posterior depends on arch too, but under
            #    coordinate ascent we treat V, U, W as data given the
            #    current model).
            # 2. Gibbs sample A[c, θ] from the HR-based Categorical
            #    conditional using per-(c, θ) sufficient stats and the
            #    current pi_archetype.
            # 3. Aggregate V, U, W per archetype using the fresh A.
            # 4. Newton GTR fixed-point M-step on pi_archetype.
            # 5. Gamma-posterior M-step on rho_chain.
            # 6. TSB update on rho_arch from hard sample counts.
            # Option 3 (pragmatic): use the flat-Dirichlet posterior on
            # N counts (attribute_cluster_soft output) for the arch update
            # -- deliberately different from the HR-based conditional so
            # the E-step for A and M-step for pi_arch are loosely coupled.
            # This leaves pi_arch room to explore during training and
            # empirically converges to a better LL than the theoretically-
            # consistent HR-based conditional (which collapses arch_probs
            # to a delta and locks in early archetype choices).
            arch_probs_flat = soft_arch_posterior(
                stats['N'], state.dyn_field.pi_archetype,
                state.dyn_field.rho_arch)
            arch_new = np.argmax(arch_probs_flat, axis=2).astype(np.int32)
            state.dyn_field.arch_assignment = arch_new
            # HR pass for the pi_arch and rho_chain M-steps.
            # +Γ+I: K_rate_bins > 0 activates the Gamma+I mixture on the
            # per-cluster effective flip rate.
            use_gammaI = (getattr(state.dyn_field, 'K_rate_bins', None)
                            not in (None, 0))
            # Per-site +Γ+I: K_rate_bins_site > 0 activates the per-site
            # substitution-rate mixture (see appendix-tkfdp.tex
            # par:arch-gamma-plus-I-persite).
            use_gammaI_site = (getattr(state.dyn_field, 'K_rate_bins_site',
                                          None) not in (None, 0))
            if use_gammaI_site:
                from .coupling.dynfield.hr_jax import gamma_quantile_means
                K_r_s = int(state.dyn_field.K_rate_bins_site)
                alpha_g_s = float(state.dyn_field.alpha_gamma_site or 1.0)
                site_bin_means = gamma_quantile_means(alpha_g_s, K_r_s)
                # Full bin means array indexed by bin id
                # (0 = invariant, 1..K_r_s = Gamma quantiles).
                site_bin_means_full = np.concatenate(
                    [[0.0], site_bin_means])
            else:
                site_bin_means_full = None
            if use_gammaI:
                hr_stats = accumulate_cluster_stats_hr_jax_gammaI(
                    model, c_obs, t_per_cluster=c_t,
                    bins_per_cluster=bins_per_cluster,
                    bin_means_full_site=site_bin_means_full,
                    weight_per_cluster=weight_per_cluster)
                # V, U, W already weighted over bins per cluster.
                # For rho_chain M-step, hr_stats['T_sum'] is
                #   Σ_q Σ_b q_q(b) · m_b · t_q  (invariant contributes 0)
                # and hr_stats['N_theta_sum'] is
                #   Σ_q Σ_b q_q(b) · N_theta[q, b] .
            else:
                hr_stats = accumulate_cluster_stats_hr_jax(
                    model, c_obs, t_per_cluster=c_t,
                    bins_per_cluster=bins_per_cluster,
                    bin_means_full=site_bin_means_full,
                    weight_per_cluster=weight_per_cluster)
            V_agg, U_agg, W_agg = aggregate_by_arch(
                hr_stats['V'], hr_stats['U'], hr_stats['W'],
                arch_new, state.dyn_field.K_a)
            pi_arch_new = update_pi_archetype_gtr(
                V_agg, U_agg, W_agg, S, alpha_prior=alpha_prior)
            state.dyn_field.pi_archetype = pi_arch_new
            rc_new = update_rho_chain_gamma(
                hr_stats['N_theta_sum'], hr_stats['T_sum'],
                prior_a=rho_chain_prior_a, prior_b=rho_chain_prior_b,
                mode='map')
            state.dyn_field.rho_chain = float(rc_new)
            # p_inv Beta-conjugate M-step (Beta(a_p, b_p) prior with
            # a_p = b_p = 2 as a weak default; updates to the posterior
            # mode given the mean-bin-posterior q_inv_mean over clusters).
            if use_gammaI:
                a_p, b_p = 2.0, 2.0
                q_inv_sum = float(hr_stats['q_inv_mean']
                                     * hr_stats['n_clust'])
                Q = int(hr_stats['n_clust'])
                p_inv_new = (a_p - 1.0 + q_inv_sum) / max(
                    Q + a_p + b_p - 2.0, 1e-9)
                state.dyn_field.p_inv = float(np.clip(p_inv_new, 1e-6, 1 - 1e-6))
            n_k_soft = arch_probs_flat.reshape(
                -1, arch_probs_flat.shape[-1]).sum(axis=0)
            tsb_arch_new, rho_arch_new = _update_rho_arch_tsb_expected(
                n_k_soft, state.dyn_field.K_a,
                alpha_arch=float(state.dyn_field.alpha_arch))
            state.dyn_field.rho_arch = rho_arch_new
            state.dyn_field.tsb_betas_arch = tsb_arch_new
            hr_info = {
                'hr_log_lik': float(hr_stats['log_lik']),
                'hr_N_theta_sum': float(hr_stats['N_theta_sum']),
                'hr_T_sum': float(hr_stats['T_sum']),
                'hr_n_clust': int(hr_stats['n_clust']),
                'hr_rho_chain': float(rc_new),
            }
            if use_gammaI:
                hr_info['gammaI_q_inv_mean'] = float(hr_stats['q_inv_mean'])
                hr_info['gammaI_q_bin_mean'] = np.asarray(
                    hr_stats['q_bin_mean']).tolist()
                hr_info['gammaI_p_inv'] = float(state.dyn_field.p_inv)
        else:
            from .coupling.dynfield.archetypes import archetype_gibbs_step
            (pi_archetype_new, arch_new,
             rho_arch_new, tsb_betas_arch_new) = archetype_gibbs_step(
                stats['N'],
                state.dyn_field.pi_archetype,
                state.dyn_field.arch_assignment,
                state.dyn_field.rho_arch,
                rng,
                alpha_prior=alpha_prior,
                alpha_arch=float(state.dyn_field.alpha_arch),
            )
            state.dyn_field.pi_archetype = pi_archetype_new
            state.dyn_field.arch_assignment = arch_new
            state.dyn_field.rho_arch = rho_arch_new
            state.dyn_field.tsb_betas_arch = tsb_betas_arch_new
        # Overwrite pi_field with the archetype-materialised tensor.
        state.dyn_field.materialise_pi_field()
        # Recompute pi_class from the new pi_field.
        state.dyn_field.pi_class = np.einsum(
            't,cta->ca', state.dyn_field.rho, state.dyn_field.pi_field)
        state.pi_class = state.dyn_field.pi_class

    # After all atom / archetype updates, compute LL again to reveal how much
    # the atom step moved the pure LL (excluding the CRP prior effect).
    if clusters:
        model_after = DynamicFieldCouplingModel.from_svi_state(state)
        c_obs_after = [(cls, X, Y) for (cls, X, Y, _) in clusters]
        c_t_after = np.asarray([t for (_, _, _, t) in clusters],
                                  dtype=np.float64)
        stats_after = _up.accumulate_cluster_stats_soft(
            model_after, c_obs_after, t_per_cluster=c_t_after)
        ll_after_atom = float(stats_after['log_lik'])
    else:
        ll_after_atom = 0.0

    # Optional split-merge move on the archetype block (see appendix
    # par:arch-split-merge). Fires periodically from the outer loop.
    sm_info = {}
    ll_after_sm = ll_after_atom
    if do_split_merge and clusters:
        from .coupling.dynfield.split_merge import split_merge_step
        sm_info = split_merge_step(
            state, stats['N'], rng,
            max_moves_per_call=int(split_merge_max_moves),
            dead_threshold=float(split_merge_dead_threshold),
            merge_tvd_threshold=float(split_merge_tvd_threshold),
            verbose=True)
        if sm_info.get('n_moves_accepted', 0) > 0:
            # Recompute LL under the post-split-merge state so the outer
            # loop can report and best-track against the true end-of-iter
            # LL (the atom-only ll_after_atom may show a large negative
            # dLL_atom due to a transient collapse that split-merge just
            # rescued).
            model_sm = DynamicFieldCouplingModel.from_svi_state(state)
            stats_sm = _up.accumulate_cluster_stats_soft(
                model_sm, c_obs_after, t_per_cluster=c_t_after)
            ll_after_sm = float(stats_sm['log_lik'])

    info = {
        'log_lik_total':      float(stats['log_lik']),  # LL after CRP, before atom
        'log_lik_after_atom': ll_after_atom,             # LL after atom updates
        'log_lik_after_sm':   ll_after_sm,               # LL after split-merge
        'n_clusters':         int(stats['n_clust']),
        **hr_info,
    }
    if sm_info:
        info['split_merge'] = sm_info
    return state, info


# ===========================================================================
# Phase D.6: end-to-end dynfield trainer wiring
# ===========================================================================
#
# `extract_cluster_observations` walks the corpus + per-MSA cluster_id
# arrays and produces the (classes, X_obs, Y_obs, t) tuples that
# `train_dynfield_one_iter` consumes.
#
# `make_cluster_loglik_fn` returns a per-MSA closure that scores any
# subset of columns under the dynfield model summed over cherries; the
# CRP sweep `gibbs_sweep_cluster` consumes this.
#
# `cluster_gibbs_sweep_all` runs the CRP sweep across all MSAs in the
# corpus.
#
# `train_dynfield_full_iter` does one complete outer iter: cluster
# Gibbs per MSA -> cluster extraction -> atom update.
#
# (A class-label sweep would update state.states_per_msa[*].cls per
# column conditional on cluster structure; deferred to a follow-up
# alongside the EM-warmup integration.)
# ===========================================================================

def make_cluster_loglik_fn(coupling_model, aa_a: np.ndarray, aa_b: np.ndarray,
                             both_aa: np.ndarray, tau: np.ndarray,
                             cls: np.ndarray,
                             pswm_a: 'np.ndarray | None' = None,
                             pswm_b: 'np.ndarray | None' = None):
    """Return a function `(columns) -> log p(cluster obs | cluster, classes)`
    that scores any subset of columns at a single MSA under the dynfield
    coupling model, summed over cherries where ALL cluster members are
    observed (both_aa[q, c] = True for every c in the cluster).

    Cached: per-cherry Sigma + Interp 2 weights are precomputed once per
    MSA; the closure batches all cherries in a single vectorised emission
    call per cluster candidate (no Python loop over cherries).

    When `pswm_a` and `pswm_b` are provided (both (n_cherries, L, A)), the
    closure dispatches to `cluster_emission_batched_soft` — each site's
    (X, Y) is a proper distribution instead of a single AA index. Hard
    aa_a/aa_b are ignored in that path (they exist alongside PSWMs only
    for the HR M-step, which is still hard-observation for now).
    """
    from .coupling.dynfield import emission as _em
    per_cherry = _em.precompute_cluster_emission_per_cherry(
        tau=tau,
        rho=coupling_model.dyn_field.rho,
        pi_field=coupling_model.dyn_field.pi_field,
        rho_chain=float(coupling_model.dyn_field.rho_chain),
    )
    soft = pswm_a is not None and pswm_b is not None

    def fn(columns):
        cols = np.asarray(columns, dtype=np.int64)
        if cols.shape[0] == 0:
            return 0.0
        classes = cls[cols]
        if soft:
            X_soft = pswm_a[:, cols, :]                 # (n_cherries, m, A)
            Y_soft = pswm_b[:, cols, :]
            totals = _em.cluster_emission_batched_soft(
                classes=classes,
                X_soft=X_soft, Y_soft=Y_soft,
                per_cherry=per_cherry,
            )
        else:
            X_batch = aa_a[:, cols]
            Y_batch = aa_b[:, cols]
            mask_X = aa_a[:, cols] < 20
            mask_Y = aa_b[:, cols] < 20
            totals = _em.cluster_emission_batched(
                classes=classes,
                X_batch=X_batch, Y_batch=Y_batch,
                mask_X=mask_X, mask_Y=mask_Y,
                per_cherry=per_cherry,
            )
        return float(np.log(np.maximum(totals, 1e-300)).sum())
    return fn


def init_site_rate_bins(state: SVIState, rng: np.random.Generator,
                              per_family_data: 'list | None' = None,
                              inv_entropy_threshold_nats: float = 0.5,
                              ) -> None:
    """Initialise state.states_per_msa[fam].site_rate_bin arrays.

    Columns with sufficiently peaked leaf-CLV posteriors (per-column
    entropy below `inv_entropy_threshold_nats`, computed as the
    average over non-gap leaves) are seeded to the invariant bin (0);
    the remaining columns are drawn uniformly from the Γ bins
    {1, …, K_rate_bins_site}.

    Rationale: the Felsenstein-conditional X sampler (`pswm_peeling`)
    now takes per-column m into account, so an invariant column's X
    is drawn under P = I, guaranteeing X = Y everywhere and making
    bin 0's `NEG_INF-if-X≠Y` semantics self-consistent. But bin 0 is
    unreachable from any state where X was drawn under m > 0 — any
    single X ≠ Y draw kills log_P0[b=0]. Seeding bin 0 where the raw
    data supports it (heavily-peaked leaf CLVs) bootstraps the fixed
    point that the Gibbs step (gibbs_resample_site_bins) then only
    needs to maintain.

    For the Γ-bin fallback, uniform 1..K_rate_bins_site is retained
    from the pre-fix scheme: any m > 0 works with the m-aware sampler
    and the bin Gibbs re-balances the mixture immediately.

    No-op if the per-site extension is disabled
    (`state.dyn_field.K_rate_bins_site` is None or 0).

    Args:
      state: SVI state; site_rate_bin is written in place.
      rng: numpy generator.
      per_family_data: if provided, use each family's leaf-CLV bundle
        (`fd['_clv']`) to compute per-column entropies and seed bin 0
        where they fall below threshold. If None (legacy call), seeds
        every column to a random Γ bin — safe but slow to converge to
        the invariant regime under the m-aware sampler.
      inv_entropy_threshold_nats: leaf-CLV average entropy in nats
        below which a column is seeded as invariant. Default 0.5 nats
        corresponds to a single-residue mass ≥ ~90%. Lower to be
        stricter, higher to be more permissive.
    """
    if state.coupling_variant != 'dynamic_field':
        return
    K_r_s = getattr(state.dyn_field, 'K_rate_bins_site', None)
    if K_r_s in (None, 0):
        return
    A = 20
    for fam_idx, st in enumerate(state.states_per_msa):
        L = int(st.L)
        gamma_bin = rng.integers(1, K_r_s + 1, size=L)
        if per_family_data is not None and fam_idx < len(per_family_data):
            fd = per_family_data[fam_idx]
            fc = fd.get('_clv')
            if fc is not None and hasattr(fc, 'leaf_msa'):
                # Detect strictly-invariant columns: all non-gap leaf
                # residues are literally the same. This is the ONLY
                # condition under which the m-aware Felsenstein sampler
                # can produce X_p = X_c on every branch of the tree
                # (leaves are fixed to observed, and P = I preserves
                # internal-node states); any mixture at the leaves
                # produces X_p != X_c on branches ending at
                # minority-residue leaves, which then hits the m=0
                # bridge kernel's P_XY = 0 → division-by-tiny blowup in
                # the M=0 HR SS accumulator. Using a soft entropy
                # threshold (e.g. < 0.5 nats) breaks this — a 90/10
                # column still has 10% minority leaves.
                leaf_msa = np.asarray(fc.leaf_msa)  # (n_leaves, L) int8
                invariant_mask = np.zeros(L, dtype=bool)
                for s in range(L):
                    col = leaf_msa[:, s]
                    valid = (col >= 0) & (col < A)
                    if not valid.any():
                        continue
                    unique_res = np.unique(col[valid])
                    if unique_res.shape[0] == 1:
                        invariant_mask[s] = True
                gamma_bin[invariant_mask] = 0
        st.site_rate_bin = gamma_bin.astype(np.int32)


def extract_cluster_bins(state: SVIState, per_family_data: list):
    """Return per-cluster site_rate_bin arrays, matched to the ordering
    of extract_cluster_observations. Each entry is a numpy int32 array
    of length equal to the number of both-observed sites in the cluster
    at that cherry, holding bin ids in {0..K_rate_bins_site}.

    Returns None if the per-site +Γ+I extension is disabled.
    """
    if state.coupling_variant != 'dynamic_field':
        return None
    K_r_s = getattr(state.dyn_field, 'K_rate_bins_site', None)
    if K_r_s in (None, 0):
        return None
    from .partition_K import (cluster_id_from_partner,
                                clusters_from_cluster_id)
    out = []
    for fam_idx, fd in enumerate(per_family_data):
        st = state.states_per_msa[fam_idx]
        if st.cluster_id is None:
            st.cluster_id = cluster_id_from_partner(st.partner)
        if st.site_rate_bin is None:
            # Late-init: shouldn't happen if init_site_rate_bins was called.
            continue
        clusters = clusters_from_cluster_id(st.cluster_id)
        both_aa = fd['both_aa']
        for _, members in clusters.items():
            mem = np.asarray(members, dtype=np.int64)
            obs = both_aa[:, mem]
            n_obs_per_cherry = obs.sum(axis=1)
            for q in np.flatnonzero(n_obs_per_cherry > 0):
                idx_obs = np.flatnonzero(obs[q])
                bins_q = st.site_rate_bin[mem[idx_obs]].astype(np.int32)
                out.append(bins_q)
    return out


def gibbs_resample_site_bins(state: SVIState, per_family_data: list,
                                 rng: np.random.Generator,
                                 batch_ids: 'list[int] | None' = None) -> dict:
    """Column-level Rao-Blackwellised per-site bin Gibbs sampling
    (par:arch-gamma-plus-I-persite). Updates
    state.states_per_msa[*].site_rate_bin in place.

    Bins are persisted per (family, column); the correct Gibbs update
    conditions on the aggregate log-likelihood across all clusters
    (all cherries) that visit that column. We build a
    column-to-(cluster, site) index, precompute per-cluster per-site
    per-bin residue log-emissions and per-cluster running Case 0 /
    Case ≥1 log-lik sums, then walk columns in a random order —
    proposing bins from the column-aggregated posterior and updating
    the affected clusters' running sums after each accept.

    Bin 0 (invariant) is included in the posterior. Users who want to
    disable the invariant bin should set `p_inv_site_init=0`; the
    resulting log-prior on bin 0 is effectively -inf (via the
    numerical floor on `prior`), and the Gibbs will never sample it.

    Returns diagnostic dict.
    """
    if state.coupling_variant != 'dynamic_field':
        return {}
    K_r_s = getattr(state.dyn_field, 'K_rate_bins_site', None)
    if K_r_s in (None, 0):
        return {}
    from .partition_K import (cluster_id_from_partner,
                                clusters_from_cluster_id)
    from .coupling.dynfield.hr_jax import (
        gamma_quantile_means, _eigendecomp_np, _logsumexp_np)
    from tkfdp.lg08 import S_LG08

    alpha_g_s = float(state.dyn_field.alpha_gamma_site or 1.0)
    p_inv_s = float(state.dyn_field.p_inv_site or 0.0)
    site_bin_means = gamma_quantile_means(alpha_g_s, K_r_s)
    bin_means_full = np.concatenate([[0.0], site_bin_means])
    prior = np.concatenate([[p_inv_s],
                              np.full(K_r_s, (1.0 - p_inv_s) / K_r_s)])
    log_prior = np.log(np.maximum(prior, 1e-300))
    K_bins = int(bin_means_full.shape[0])
    NEG_INF = -1e18

    dyn = state.dyn_field
    pi_arch = np.asarray(dyn.pi_archetype, dtype=np.float64)
    arch_assignment = np.asarray(dyn.arch_assignment, dtype=np.int32)
    rho = np.asarray(dyn.rho, dtype=np.float64)
    rho_chain = float(dyn.rho_chain)
    L = int(rho.shape[0])
    S = np.asarray(S_LG08, dtype=np.float64)
    xi, U_arch, D_arch = _eigendecomp_np(pi_arch, S)

    # 1. Enumerate (cluster, cherry, sites) → columns, and build the
    #    column-to-(cluster, site) index.
    clusters_data = []   # list of dicts with classes, X, Y, tau, cols (fam-local col ids), fam_idx
    col_to_sites = {}    # (fam_idx, col_id) → list of (cluster_idx, site_idx)
    _batch_set = None if batch_ids is None else set(int(i) for i in batch_ids)
    for fam_idx, fd in enumerate(per_family_data):
        if _batch_set is not None and fam_idx not in _batch_set:
            continue
        st = state.states_per_msa[fam_idx]
        if st.cluster_id is None:
            st.cluster_id = cluster_id_from_partner(st.partner)
        if st.site_rate_bin is None:
            continue
        clusters = clusters_from_cluster_id(st.cluster_id)
        cls = st.cls
        aa_a = fd['aa_a']; aa_b = fd['aa_b']
        both_aa = fd['both_aa']; tau = fd['tau']
        for _, members in clusters.items():
            mem = np.asarray(members, dtype=np.int64)
            classes_mem = cls[mem].astype(np.int64)
            obs = both_aa[:, mem]
            n_obs_per_cherry = obs.sum(axis=1)
            for q in np.flatnonzero(n_obs_per_cherry > 0):
                idx_obs = np.flatnonzero(obs[q])
                classes_q = classes_mem[idx_obs]
                X_q = aa_a[q, mem[idx_obs]].astype(np.int64)
                Y_q = aa_b[q, mem[idx_obs]].astype(np.int64)
                cols_q = mem[idx_obs].astype(np.int64)
                cluster_idx = len(clusters_data)
                clusters_data.append({
                    'fam_idx': fam_idx,
                    'classes': classes_q,
                    'X': X_q, 'Y': Y_q,
                    'tau': float(tau[q]),
                    'cols': cols_q,
                })
                for site_idx, col_id in enumerate(cols_q):
                    col_to_sites.setdefault((fam_idx, int(col_id)), []).append(
                        (cluster_idx, site_idx))

    if not clusters_data:
        return {}

    # 2. Precompute per-cluster: log_P0, log_LJ (per-site per-θ per-bin),
    #    log_w_A_diag, log_w_B (τ-dependent field weights).
    #    Also initialise running log_A_diag, log_B at current bins.
    cluster_precomp = []
    for c in clusters_data:
        classes = c['classes']; X = c['X']; Y = c['Y']; tau = c['tau']
        fam_idx = c['fam_idx']
        N = int(classes.size)
        st = state.states_per_msa[fam_idx]
        current_bins = st.site_rate_bin[c['cols']].astype(np.int32)

        # Field-side weights.
        beta_t = np.exp(-rho_chain * (1.0 - rho) * tau)
        g_t = float(np.exp(-rho_chain * tau))
        P_field = (g_t * np.eye(L)
                     + (1.0 - g_t) * rho[None, :] * np.ones((L, 1)))
        P_no_jump = np.diag(beta_t)
        P_jump = np.maximum(P_field - P_no_jump, 0.0)
        log_w_A_diag = np.log(np.maximum(rho * beta_t, 1e-300))
        log_w_B = np.log(np.maximum(rho[:, None] * P_jump, 1e-300))

        # Per-site per-θ per-bin emissions.
        k_by_theta = arch_assignment[classes]
        log_P0 = np.full((N, L, K_bins), NEG_INF)
        log_LJ = np.full((N, L, K_bins), NEG_INF)
        for n in range(N):
            Xn = int(X[n]); Yn = int(Y[n]); X_eq_Y = (Xn == Yn)
            for th in range(L):
                k = int(k_by_theta[n, th])
                log_pi_kY = float(np.log(max(pi_arch[k, Yn], 1e-300)))
                for b in range(K_bins):
                    m_b = float(bin_means_full[b])
                    if m_b == 0.0:
                        log_P0[n, th, b] = 0.0 if X_eq_Y else NEG_INF
                        log_LJ[n, th, b] = log_pi_kY
                    else:
                        exp_xi_t = np.exp(m_b * xi[k] * tau)
                        P_sym = (U_arch[k] * exp_xi_t[None, :]) @ U_arch[k].T
                        P_XY = (D_arch[k][Yn] / max(D_arch[k][Xn], 1e-300)
                                    * P_sym[Xn, Yn])
                        log_P0[n, th, b] = float(np.log(max(P_XY, 1e-300)))
                        log_LJ[n, th, b] = log_pi_kY

        # Running sums at current bins.
        log_A_diag = log_w_A_diag.copy()
        addend_ty = np.zeros(L)
        for n in range(N):
            log_A_diag += log_P0[n, :, current_bins[n]]
            addend_ty += log_LJ[n, :, current_bins[n]]
        log_B = log_w_B + addend_ty[None, :]

        cluster_precomp.append({
            'log_P0': log_P0, 'log_LJ': log_LJ,
            'log_A_diag': log_A_diag, 'log_B': log_B,
            'current_bins': current_bins.copy(),
        })

    # 3. Sequential column-level Gibbs.
    n_flipped = 0
    columns = list(col_to_sites.items())
    rng.shuffle(columns)  # random order

    for (fam_idx, col_id), sites in columns:
        st = state.states_per_msa[fam_idx]
        current_bin = int(st.site_rate_bin[col_id])

        # Aggregate LL contribution over affected clusters for each candidate bin.
        LL_col = np.zeros(K_bins)
        for cluster_idx, site_idx in sites:
            cp = cluster_precomp[cluster_idx]
            log_A_diag_cur = cp['log_A_diag']
            log_B_cur = cp['log_B']
            log_P0_ns = cp['log_P0'][site_idx]
            log_LJ_ns = cp['log_LJ'][site_idx]

            # LL of current cluster state at current_bin (baseline).
            log_A_new = log_A_diag_cur - log_P0_ns[:, current_bin]
            log_LJ_delta_ty_cur = -log_LJ_ns[:, current_bin]

            # Compute LL_cluster at each candidate b.
            for b in range(K_bins):
                log_A_diag_b = log_A_new + log_P0_ns[:, b]
                log_B_b = log_B_cur + (log_LJ_delta_ty_cur
                                            + log_LJ_ns[:, b])[None, :]
                log_P_pair = log_B_b.copy()
                for tx in range(L):
                    log_P_pair[tx, tx] = np.logaddexp(
                        log_A_diag_b[tx], log_B_b[tx, tx])
                LL_col[b] += _logsumexp_np(log_P_pair.reshape(-1))

        # Posterior.
        log_post = log_prior + LL_col
        log_post -= _logsumexp_np(log_post)
        p = np.exp(log_post)
        s = float(p.sum())
        if s <= 0.0 or not np.isfinite(s):
            # Fallback: uniform over Γ bins.
            p = np.zeros(K_bins)
            p[1:] = 1.0 / max(K_bins - 1, 1)
            s = 1.0
        p = p / s
        new_bin = int(rng.choice(K_bins, p=p))

        if new_bin != current_bin:
            n_flipped += 1
            # Update column bin and all affected clusters' running sums.
            st.site_rate_bin[col_id] = new_bin
            for cluster_idx, site_idx in sites:
                cp = cluster_precomp[cluster_idx]
                log_P0_ns = cp['log_P0'][site_idx]
                log_LJ_ns = cp['log_LJ'][site_idx]
                cp['log_A_diag'] = (cp['log_A_diag']
                                      - log_P0_ns[:, current_bin]
                                      + log_P0_ns[:, new_bin])
                cp['log_B'] = (cp['log_B']
                                 + (log_LJ_ns[:, new_bin]
                                      - log_LJ_ns[:, current_bin])[None, :])
                cp['current_bins'][site_idx] = new_bin

    # Diagnostics.
    n_inv = 0; n_total = 0
    for fam_idx, _ in enumerate(per_family_data):
        st = state.states_per_msa[fam_idx]
        if st.site_rate_bin is not None:
            b = st.site_rate_bin
            n_inv += int((b == 0).sum())
            n_total += int(b.size)
    return {'n_columns_resampled': len(columns),
              'n_columns_flipped': n_flipped,
              'frac_invariant': n_inv / max(n_total, 1)}


def extract_cluster_observations(state: SVIState,
                                    per_family_data: list) -> list:
    """Walk per_family_data + state.states_per_msa to produce the
    (classes, X_obs, Y_obs, t) tuples for `train_dynfield_one_iter`.

    Each (cluster, cherry) pair contributes one tuple containing the
    **both-observed subset** of cluster sites at that cherry: sites where
    both leaves carry an AA. Cherries with no both-observed cluster site
    are skipped. (Partially observed sites -- X-only or Y-only -- carry
    smaller per-(c, theta) weight and are dropped at this layer; the
    cluster Gibbs and class Gibbs paths use the gap-aware kernel directly
    and do not lose them.)
    """
    from .partition_K import (cluster_id_from_partner,
                                clusters_from_cluster_id)
    out = []
    for fam_idx, fd in enumerate(per_family_data):
        st = state.states_per_msa[fam_idx]
        if st.cluster_id is None:
            st.cluster_id = cluster_id_from_partner(st.partner)
        clusters = clusters_from_cluster_id(st.cluster_id)
        cls = st.cls
        aa_a = fd['aa_a']; aa_b = fd['aa_b']
        both_aa = fd['both_aa']; tau = fd['tau']
        for _, members in clusters.items():
            mem = np.asarray(members, dtype=np.int64)
            classes_mem = cls[mem].astype(np.int64)
            # Per-(cherry, site) both-observed mask: True where both leaves
            # are non-gap at this cluster column.
            obs = both_aa[:, mem]                       # (n_cherries, m) bool
            n_obs_per_cherry = obs.sum(axis=1)
            for q in np.flatnonzero(n_obs_per_cherry > 0):
                idx_obs = np.flatnonzero(obs[q])        # sites observed at q
                classes_q = classes_mem[idx_obs]
                X_q = aa_a[q, mem[idx_obs]].astype(np.int64)
                Y_q = aa_b[q, mem[idx_obs]].astype(np.int64)
                out.append((classes_q, X_q, Y_q, float(tau[q])))
    return out


def cluster_gibbs_sweep_all(state: SVIState, per_family_data: list,
                              rng: np.random.Generator,
                              *, alpha_z: float = 1.0,
                              max_cluster_size: int = 16,
                              temperature: float = 1.0,
                              use_jax_batched: bool = True,
                              batch_ids: 'list[int] | None' = None,
                              ) -> SVIState:
    """Run gibbs_sweep_cluster for each MSA in the corpus, using a
    dynfield cluster_loglik_fn closure on the current coupling state.

    When `use_jax_batched=True` (default), constructs a per-MSA JAX
    `BatchedDynfieldScorer` and uses its batched API to score all
    candidates per column in a single device call -- the largest single
    speedup over the per-cherry path.

    Mutates `state.states_per_msa[*].cluster_id` in place.
    """
    if state.coupling_variant != 'dynamic_field':
        raise ValueError(
            f"cluster_gibbs_sweep_all requires coupling_variant="
            f"'dynamic_field', got {state.coupling_variant!r}")
    from .partition_K import gibbs_sweep_cluster
    from .coupling.dynfield import DynamicFieldCouplingModel
    from .coupling.dynfield import emission as _em
    model = DynamicFieldCouplingModel.from_svi_state(state)
    # Corpus-wide pads so every MSA's scorer routes through the same
    # cached JAX kernel shape (one compile per B-bucket, not per MSA).
    n_cherries_pad = max(int(fd['aa_a'].shape[0]) for fd in per_family_data)
    L_cols_pad = max(int(fd['aa_a'].shape[1]) for fd in per_family_data)
    _batch_set = None if batch_ids is None else set(int(i) for i in batch_ids)
    for fam_idx, fd in enumerate(per_family_data):
        if _batch_set is not None and fam_idx not in _batch_set:
            continue
        st = state.states_per_msa[fam_idx]
        # Soft cluster emission is only dispatched when the family
        # explicitly opts in via `use_soft_cluster_emission=True`.
        # Otherwise even PSWM-loaded families route through the fast
        # JAX BatchedDynfieldScorer with the current (MC-resampled)
        # hard aa_a / aa_b — cross-iter averaging gives soft in
        # expectation, and the per-iter numpy einsum-over-A^2 soft
        # path is prohibitively slow at 100k+ cherries.
        use_soft = bool(fd.get('use_soft_cluster_emission', False))
        fn = make_cluster_loglik_fn(
            model, fd['aa_a'], fd['aa_b'], fd['both_aa'], fd['tau'],
            st.cls,
            pswm_a=(fd.get('pswm_a') if use_soft else None),
            pswm_b=(fd.get('pswm_b') if use_soft else None))
        batched_fn = None
        if use_jax_batched and not use_soft:
            try:
                scorer = _em.BatchedDynfieldScorer(
                    aa_a=fd['aa_a'], aa_b=fd['aa_b'],
                    both_aa=fd['both_aa'], tau=fd['tau'],
                    rho=model.dyn_field.rho,
                    pi_field=model.dyn_field.pi_field,
                    rho_chain=float(model.dyn_field.rho_chain),
                    max_cluster_size=max_cluster_size,
                    n_cherries_pad=n_cherries_pad,
                    L_cols_pad=L_cols_pad,
                )
                cls_local = st.cls
                batched_fn = lambda reqs, _s=scorer, _c=cls_local: (
                    _s.score_batch(reqs, _c))
            except Exception as e:
                # Fall back silently to scalar path on JAX errors.
                import warnings as _w
                _w.warn(f"BatchedDynfieldScorer init failed "
                        f"({type(e).__name__}: {e}); falling back to scalar.")
        state.states_per_msa[fam_idx] = gibbs_sweep_cluster(
            st, fn, rng, alpha_z=alpha_z,
            max_cluster_size=max_cluster_size,
            temperature=temperature,
            batched_score_fn=batched_fn)
    return state


def train_dynfield_full_iter(state: SVIState, per_family_data: list,
                                rng: np.random.Generator,
                                *, alpha_z: float = 1.0,
                                max_cluster_size: int = 16,
                                alpha_prior: float = 1.0,
                                rho_chain_mh_steps: int = 0,
                                rho_chain_prior_a: float = 1.5,
                                rho_chain_prior_b: float = 5.0,
                                rho_chain_step_size: float = 0.3,
                                ) -> tuple[SVIState, dict]:
    """One complete outer iter of the dynfield trainer.

    Steps:
      1. CRP cluster Gibbs sweep per MSA (updates cluster_id).
      2. Extract (classes, X_obs, Y_obs, t) tuples across the corpus.
      3. Atom update: soft EM + Dirichlet on pi_field + TSB on rho.
      4. Optional: rho_chain MH update under Gamma(a, b) prior
         (`rho_chain_mh_steps > 0`).

    Returns (state_new, info_dict) where info_dict carries
    log-likelihood, cluster-count summary statistics, and rho_chain MH
    summary if run.
    """
    state = cluster_gibbs_sweep_all(
        state, per_family_data, rng,
        alpha_z=alpha_z, max_cluster_size=max_cluster_size)
    clusters = extract_cluster_observations(state, per_family_data)
    # Per-cluster site_rate_bin arrays for the per-site +Γ+I extension
    # (par:arch-gamma-plus-I-persite). Returns None when disabled.
    bins_per_cluster = extract_cluster_bins(state, per_family_data)
    state, atom_info = train_dynfield_one_iter(
        state, clusters, alpha_prior=alpha_prior,
        bins_per_cluster=bins_per_cluster)
    rho_info = {}
    if rho_chain_mh_steps > 0:
        from .coupling.dynfield import updates as _up
        from .coupling.dynfield import DynamicFieldCouplingModel
        model = DynamicFieldCouplingModel.from_svi_state(state)
        c_obs = [(classes, X, Y) for (classes, X, Y, _) in clusters]
        c_t = np.asarray([t for (_, _, _, t) in clusters], dtype=np.float64)
        new_rc, rho_info = _up.update_rho_chain_mh(
            model, c_obs, c_t,
            prior_a=rho_chain_prior_a, prior_b=rho_chain_prior_b,
            n_steps=rho_chain_mh_steps,
            step_size=rho_chain_step_size,
            rng=rng)
        state.dyn_field.rho_chain = new_rc
    # Cluster-size distribution summary.
    sizes = []
    for st in state.states_per_msa:
        if st.cluster_id is not None:
            from .partition_K import clusters_from_cluster_id
            cmap = clusters_from_cluster_id(st.cluster_id)
            sizes.extend(len(v) for v in cmap.values())
    sizes_arr = np.asarray(sizes) if sizes else np.zeros(1)
    info = {
        'log_lik_total': atom_info['log_lik_total'],
        'n_clusters':    atom_info['n_clusters'],
        'mean_cluster_size': float(sizes_arr.mean()),
        'max_cluster_size':  int(sizes_arr.max()),
        'n_clusters_total':  int(len(sizes)),
        'rho_chain':         float(state.dyn_field.rho_chain),
    }
    if rho_info:
        info['rho_chain_mh_accept'] = int(rho_info['n_steps_accept'])
        info['rho_chain_post_log_lik'] = float(rho_info['final_log_lik'])
    return state, info


# ===========================================================================
# Class-label Gibbs sweep for dynfield (Phase D.6 follow-up)
# ===========================================================================

def class_gibbs_sweep_all_dynfield(state: SVIState, per_family_data: list,
                                      rng: np.random.Generator,
                                      *, alpha_c: float = 1.0,
                                      class_log_weights: 'np.ndarray | None' = None,
                                      use_jax_batched: bool = True,
                                      batch_ids: 'list[int] | None' = None,
                                      ) -> SVIState:
    """Class-label Gibbs sweep for the dynfield variant.

    For each column s in each MSA, resamples `cls[s]` from
    P(cls[s] = k | rest, observations) ∝ P(cls[s] = k) *
    P(cluster_containing_s | cls swapped at s, observations).

    Args:
      alpha_c: symmetric finite-K Dirichlet-Multinomial concentration on
        class assignments. Used as the prior unless `class_log_weights`
        is provided.
      class_log_weights: optional (K_c,) array of log p(c) -- if set,
        overrides the symmetric prior (e.g., to pass TSB stick weights).
      use_jax_batched: if True (default), score all K_c candidate class
        labels per column in a single batched JAX kernel call via
        `BatchedDynfieldScorer.score_batch_with_classes`. This is the
        dominant per-iter cost at K_c >= 8; the numpy fallback loop is
        retained for CPU-only environments and legacy paths.

    Mutates `state.states_per_msa[*].cls` in place. cluster_id is read
    but not modified -- run a cluster sweep separately if you want to
    update both.
    """
    if state.coupling_variant != 'dynamic_field':
        raise ValueError(
            f"class_gibbs_sweep_all_dynfield requires coupling_variant="
            f"'dynamic_field', got {state.coupling_variant!r}")
    from .partition_K import (cluster_id_from_partner,
                                clusters_from_cluster_id)
    from .coupling.dynfield import DynamicFieldCouplingModel
    from .coupling.dynfield import emission as _em

    model = DynamicFieldCouplingModel.from_svi_state(state)
    K_c = state.K_c

    # Corpus-wide pads for shared JAX kernel cache.
    n_cherries_pad = max(int(fd['aa_a'].shape[0]) for fd in per_family_data)
    L_cols_pad = max(int(fd['aa_a'].shape[1]) for fd in per_family_data)

    _batch_set = None if batch_ids is None else set(int(i) for i in batch_ids)
    for fam_idx, fd in enumerate(per_family_data):
        if _batch_set is not None and fam_idx not in _batch_set:
            continue
        st = state.states_per_msa[fam_idx]
        L = st.L
        if st.cluster_id is None:
            st.cluster_id = cluster_id_from_partner(st.partner)
        aa_a = fd['aa_a']; aa_b = fd['aa_b']
        both_aa = fd['both_aa']; tau = fd['tau']
        use_soft = bool(fd.get('use_soft_cluster_emission', False))
        pswm_a = fd.get('pswm_a') if use_soft else None
        pswm_b = fd.get('pswm_b') if use_soft else None
        soft = pswm_a is not None and pswm_b is not None

        # Numpy scorer (per-cherry cache) for the fallback path.
        per_cherry = _em.precompute_cluster_emission_per_cherry(
            tau=tau,
            rho=model.dyn_field.rho,
            pi_field=model.dyn_field.pi_field,
            rho_chain=float(model.dyn_field.rho_chain),
        )
        scorer = None
        if use_jax_batched and not soft:
            try:
                scorer = _em.BatchedDynfieldScorer(
                    aa_a=aa_a, aa_b=aa_b, both_aa=both_aa, tau=tau,
                    rho=model.dyn_field.rho,
                    pi_field=model.dyn_field.pi_field,
                    rho_chain=float(model.dyn_field.rho_chain),
                    max_cluster_size=int(np.max(np.bincount(st.cluster_id))
                                          if st.cluster_id.size > 0 else 1),
                    n_cherries_pad=n_cherries_pad,
                    L_cols_pad=L_cols_pad,
                )
            except Exception as e:
                import warnings as _w
                _w.warn(f"BatchedDynfieldScorer init failed for class sweep "
                        f"({type(e).__name__}: {e}); falling back to numpy.")
                scorer = None

        clusters = clusters_from_cluster_id(st.cluster_id)
        # Index: column -> its cluster members (so we don't rebuild every step).
        col_to_members = {}
        for cid, members in clusters.items():
            mem_arr = np.asarray(members, dtype=np.int64)
            for c in members:
                col_to_members[c] = mem_arr

        def _score_np(columns, classes):
            cols = np.asarray(columns, dtype=np.int64)
            if cols.shape[0] == 0:
                return 0.0
            if soft:
                X_soft = pswm_a[:, cols, :]
                Y_soft = pswm_b[:, cols, :]
                totals = _em.cluster_emission_batched_soft(
                    classes=classes,
                    X_soft=X_soft, Y_soft=Y_soft,
                    per_cherry=per_cherry,
                )
            else:
                X_batch = aa_a[:, cols]
                Y_batch = aa_b[:, cols]
                mask_X = aa_a[:, cols] < 20
                mask_Y = aa_b[:, cols] < 20
                totals = _em.cluster_emission_batched(
                    classes=classes,
                    X_batch=X_batch, Y_batch=Y_batch,
                    mask_X=mask_X, mask_Y=mask_Y,
                    per_cherry=per_cherry,
                )
            return float(np.log(np.maximum(totals, 1e-300)).sum())

        order = rng.permutation(L)
        for s in order:
            s = int(s)
            mem_arr = col_to_members[s]
            # Class prior.
            if class_log_weights is not None:
                log_class_prior = class_log_weights
            else:
                counts = np.bincount(
                    st.cls, minlength=K_c).astype(np.float64)
                counts[st.cls[s]] -= 1.0
                log_class_prior = (np.log(counts + alpha_c / K_c)
                                    - np.log(L - 1 + alpha_c))

            # Build K_c candidates that share `mem_arr` (columns) and vary
            # only at s's position within the cluster.
            idx_in_cluster = int(np.where(mem_arr == s)[0][0])
            base_classes = st.cls[mem_arr].copy().astype(np.int64)
            cls_trials = np.tile(base_classes, (K_c, 1))
            cls_trials[:, idx_in_cluster] = np.arange(K_c)

            if scorer is not None:
                # Batched: one JAX call scores all K_c candidates.
                columns_list = [mem_arr for _ in range(K_c)]
                classes_list = [cls_trials[k] for k in range(K_c)]
                lls = scorer.score_batch_with_classes(
                    columns_list, classes_list)
                log_p = lls + log_class_prior
            else:
                log_p = np.zeros(K_c)
                for k in range(K_c):
                    ll_k = _score_np(mem_arr, cls_trials[k])
                    log_p[k] = ll_k + log_class_prior[k]

            log_p -= log_p.max()
            probs = np.exp(log_p)
            probs /= probs.sum()
            st.cls[s] = int(rng.choice(K_c, p=probs))

    return state
