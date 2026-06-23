"""Like fit_supervised_H.py but with an extra "uncoupled" mixture component.

Mixture structure (per column-pair (i, j)):
  - latent ζ_{ij} = 0 with prior w_un  → both columns evolve independently
                                          under the K=8 singlet mixture
  - latent ζ_{ij} = 1 with prior 1-w_un → coupled via H, then conditional on
                                            (c1, c2) ∈ {1..K_c}² with prior
                                            ρ_{c1} ρ_{c2}

Total 1 + K_c² = 65 components. H is fit by EM over the coupled components
only; the uncoupled component is a fixed singlet K=8 emission table (depends
only on π_class, ρ, S_LG — all fixed).

Updating w_un: in the M-step, w_un_new = mean over pairs of γ_uncoupled_{ij}
(plus optional Beta(a, b) MAP prior).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contacts', required=True)
    ap.add_argument('--k8-state',
                    default='/home/yam/tkf-dp/results/_preserved/K8_KH1_top8000_iter68/state.npz')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-em-iters', type=int, default=25)
    ap.add_argument('--n-mstep-iters', type=int, default=30)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--prior-tau', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--warm-start', action='store_true')
    ap.add_argument('--init-w-un', type=float, default=0.5,
                    help='Initial mixture weight on the uncoupled component.')
    ap.add_argument('--beta-a', type=float, default=1.0,
                    help='Beta prior shape for w_un (default Uniform).')
    ap.add_argument('--beta-b', type=float, default=1.0)
    ap.add_argument('--freeze-w-un', action='store_true',
                    help='Fix w_un at --init-w-un (no M-step update).')
    args = ap.parse_args()

    sys.path.insert(0, '/home/yam/tkf-dp/src')
    import jax
    import jax.numpy as jnp
    from tkfdp.generator import (
        build_joint_Q_pair, joint_stationary_pair,
        symmetrize_eigh, log_transition_matrices,
    )
    from tkfdp.lg08 import S_LG08_F81

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading contacts from {args.contacts}')
    d = np.load(args.contacts, allow_pickle=True)
    anc1, anc2 = np.asarray(d['anc1']), np.asarray(d['anc2'])
    des1, des2 = np.asarray(d['des1']), np.asarray(d['des2'])
    pair_idx = np.asarray(d['pair_idx'])
    tau_bin = np.asarray(d['tau_bin'])
    tau_centers = np.asarray(d['tau_centers'], dtype=np.float64)
    n_pairs = int(d['n_pairs'])
    n_t = int(d['n_time_bins'])
    print(f'  {len(anc1):,} cherries across {n_pairs:,} column-pairs')

    print(f'Loading K=8 state from {args.k8_state}')
    k8 = np.load(args.k8_state, allow_pickle=True)
    pi_class = np.asarray(k8['pi_class'], dtype=np.float64)
    K_c, A = pi_class.shape
    rho = np.zeros(K_c, dtype=np.int64)
    for k in k8.files:
        if re.match(r'cls_\d+$', k):
            rho += np.bincount(k8[k].astype(int), minlength=K_c)
    rho = rho / rho.sum() if rho.sum() else np.ones(K_c) / K_c
    print(f'  K_c={K_c}, A={A}, ρ={rho.round(3).tolist()}')

    if args.warm_start:
        H_init = 0.5 * (k8['potts_atoms'][0] + k8['potts_atoms'][0].T)
        print(f'  warm-start H, ‖H‖_F={np.linalg.norm(H_init):.3f}')
    else:
        rng = np.random.default_rng(args.seed)
        Z = rng.normal(0.0, 0.3, (A, A))
        H_init = 0.5 * (Z + Z.T)
        print(f'  random-init H (seed={args.seed}), ‖H‖_F={np.linalg.norm(H_init):.3f}')

    pi_class_j = jnp.asarray(pi_class)
    rho_j = jnp.asarray(rho)
    S_j = jnp.asarray(S_LG08_F81)
    tau_centers_j = jnp.asarray(tau_centers)

    # Pair-sorted cherry arrays
    sort_idx = np.argsort(pair_idx, kind='stable')
    pair_idx_s = pair_idx[sort_idx]
    anc1 = anc1[sort_idx]; anc2 = anc2[sort_idx]
    des1 = des1[sort_idx]; des2 = des2[sort_idx]
    tau_bin = tau_bin[sort_idx]
    anc1_j = jnp.asarray(anc1.astype(np.int32))
    anc2_j = jnp.asarray(anc2.astype(np.int32))
    des1_j = jnp.asarray(des1.astype(np.int32))
    des2_j = jnp.asarray(des2.astype(np.int32))
    tau_bin_j = jnp.asarray(tau_bin.astype(np.int32))
    pair_idx_j = jnp.asarray(pair_idx_s.astype(np.int32))
    start_400 = anc1_j * A + anc2_j
    end_400   = des1_j * A + des2_j
    A2 = A * A

    # --- Precompute uncoupled singlet emission log_M(t, anc, des) ---
    # M(a, d, t) = sum_c ρ_c · π_c(a) · exp(Q_c · t)[a, d]
    # where Q_c is F81-form with shared S_LG08 and per-class π_c.
    def build_singlet_Pc(c):
        pi_c = pi_class_j[c]
        # F81 single-site rate matrix:
        S_off = S_j - jnp.diag(jnp.diag(S_j))
        Q = S_off * pi_c[None, :]
        Q = Q - jnp.diag(Q.sum(axis=1))
        # eigh-based matrix exponential
        sqrt_pi = jnp.sqrt(pi_c)
        Q_sym = (Q * sqrt_pi[None, :]) / sqrt_pi[:, None]
        Q_sym = 0.5 * (Q_sym + Q_sym.T)
        eigvals, eigvecs = jnp.linalg.eigh(Q_sym)
        # exp(Q t) = diag(sqrt_pi)^-1 · V · diag(exp(λ t)) · V^T · diag(sqrt_pi)
        def Pt(t):
            E = jnp.exp(eigvals * t)
            M_sym = eigvecs @ jnp.diag(E) @ eigvecs.T
            return (M_sym / sqrt_pi[None, :]) * sqrt_pi[:, None]
        return jax.vmap(Pt)(tau_centers_j)  # (n_t, A, A)

    P_per_class = jax.vmap(build_singlet_Pc)(jnp.arange(K_c))  # (K_c, n_t, A, A)
    pi_a_per_class = pi_class_j[:, None, :, None]  # (K_c, 1, A, 1) for broadcasting
    M_joint = (rho_j[:, None, None, None] * pi_a_per_class * P_per_class).sum(axis=0)
    log_M = jnp.log(jnp.clip(M_joint, 1e-30, 1.0))  # (n_t, A, A)

    # Uncoupled log-lik per cherry: log_M[tau_bin, anc1, des1] + log_M[tau_bin, anc2, des2]
    @jax.jit
    def cherry_loglik_un_per_pair():
        lp_c1 = log_M[tau_bin_j, anc1_j, des1_j]   # (N,)
        lp_c2 = log_M[tau_bin_j, anc2_j, des2_j]
        per_cherry = lp_c1 + lp_c2
        return jax.ops.segment_sum(per_cherry, pair_idx_j, num_segments=n_pairs)

    ll_un = cherry_loglik_un_per_pair()  # (n_pairs,) — FIXED (doesn't depend on H)
    print(f'\nUncoupled log-lik per pair: mean={float(ll_un.mean()):.2f}, '
          f'min={float(ll_un.min()):.2f}, max={float(ll_un.max()):.2f}')

    @jax.jit
    def build_logP_coupled(H):
        H_sym = 0.5 * (H + H.T)
        def per_cc(c1, c2):
            pi1, pi2 = pi_class_j[c1], pi_class_j[c2]
            Q = build_joint_Q_pair(H_sym, pi1, pi2, S=S_j)
            pi_j = joint_stationary_pair(H_sym, pi1, pi2)
            Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
            return log_transition_matrices(tau_centers_j, Lambda, U_sym, sqrt_pij)
        return jax.vmap(lambda c1: jax.vmap(lambda c2: per_cc(c1, c2))(jnp.arange(K_c)))(jnp.arange(K_c))

    @jax.jit
    def cherry_loglik_co_per_pair_cc(H):
        log_P = build_logP_coupled(H)
        lp_per_cherry = log_P[:, :, tau_bin_j, start_400, end_400]
        return jax.ops.segment_sum(
            lp_per_cherry.transpose(2, 0, 1), pair_idx_j, num_segments=n_pairs)

    log_rho_pair = jnp.log(rho_j + 1e-30)[:, None] + jnp.log(rho_j + 1e-30)[None, :]

    @jax.jit
    def e_step(H, w_un):
        ll_co = cherry_loglik_co_per_pair_cc(H)  # (n_pairs, K_c, K_c)
        log_un = jnp.log(w_un + 1e-30) + ll_un  # (n_pairs,)
        log_co_per_cc = ll_co + log_rho_pair    # (n_pairs, K_c, K_c)
        log_co_marg = jax.scipy.special.logsumexp(
            log_co_per_cc.reshape(n_pairs, -1), axis=1) + jnp.log(1.0 - w_un + 1e-30)
        m = jnp.maximum(log_un, log_co_marg)
        log_marg = m + jnp.log(jnp.exp(log_un - m) + jnp.exp(log_co_marg - m))
        gamma_un = jnp.exp(log_un - log_marg)  # (n_pairs,)
        gamma_co_total = jnp.exp(log_co_marg - log_marg)  # (n_pairs,)
        log_co_cc_marg = jax.scipy.special.logsumexp(
            log_co_per_cc.reshape(n_pairs, -1), axis=1, keepdims=True)
        gamma_co_cc = jnp.exp(
            (log_co_per_cc.reshape(n_pairs, -1) - log_co_cc_marg)
        ).reshape(n_pairs, K_c, K_c) * gamma_co_total[:, None, None]
        return gamma_un, gamma_co_cc, log_marg

    @jax.jit
    def expected_complete_loglik(H, gamma_co_cc):
        """Only the coupled components contribute to H's M-step."""
        ll = cherry_loglik_co_per_pair_cc(H)  # (n_pairs, K_c, K_c)
        ecll = (gamma_co_cc * ll).sum() / float(n_pairs)
        H_sym = 0.5 * (H + H.T)
        off = H_sym - jnp.diag(jnp.diag(H_sym))
        prior = -0.5 * args.prior_tau * (off ** 2).sum() / float(n_pairs)
        return ecll + prior

    grad_obj = jax.jit(jax.grad(
        lambda H, gamma: -expected_complete_loglik(H, gamma)))

    H = jnp.asarray(H_init, dtype=jnp.float64)
    w_un = jnp.float64(args.init_w_un)
    m_state = jnp.zeros_like(H); v_state = jnp.zeros_like(H)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    print(f'\nStarting EM (n_em={args.n_em_iters}, n_mstep={args.n_mstep_iters}, '
          f'lr={args.lr}, init w_un={float(w_un):.3f}, '
          f'freeze_w_un={args.freeze_w_un})')
    history = []
    for it in range(args.n_em_iters):
        t0 = time.time()
        gamma_un, gamma_co_cc, log_marg = e_step(H, w_un)
        marg_ll = float(log_marg.sum())
        mean_gamma_un = float(gamma_un.mean())

        # M-step on H (Adam, only coupled components)
        for ms in range(args.n_mstep_iters):
            g = grad_obj(H, gamma_co_cc)
            g = 0.5 * (g + g.T)
            g = jnp.clip(g, -5.0, 5.0)
            t = it * args.n_mstep_iters + ms + 1
            m_state = beta1 * m_state + (1 - beta1) * g
            v_state = beta2 * v_state + (1 - beta2) * (g ** 2)
            mhat = m_state / (1 - beta1 ** t)
            vhat = v_state / (1 - beta2 ** t)
            H = H - args.lr * mhat / (jnp.sqrt(vhat) + eps)
            H = 0.5 * (H + H.T)

        # M-step on w_un (closed form: Beta(a, b) MAP)
        if not args.freeze_w_un:
            s_un = float(gamma_un.sum())
            s_co = float(n_pairs - s_un)
            w_un = jnp.float64(
                (s_un + args.beta_a - 1.0) / (n_pairs + args.beta_a + args.beta_b - 2.0))
            w_un = jnp.clip(w_un, 1e-6, 1.0 - 1e-6)

        norm_H = float(jnp.linalg.norm(H))
        ecll = float(expected_complete_loglik(H, gamma_co_cc))
        elapsed = time.time() - t0
        history.append({
            'iter': it, 'marg_loglik': marg_ll, 'ecll': ecll,
            'norm_H': norm_H, 'w_un': float(w_un),
            'mean_gamma_un': mean_gamma_un, 'elapsed_s': elapsed,
        })
        print(f'  EM[{it+1}/{args.n_em_iters}]  marg_LL={marg_ll:.1f}  '
              f'mean γ_un={mean_gamma_un:.3f}  w_un={float(w_un):.3f}  '
              f'‖H‖={norm_H:.3f}  ({elapsed:.1f}s)')

    H_np = np.asarray(H)
    np.savez(out_dir / 'state.npz',
             potts_atoms=H_np[None, :, :].astype(np.float32),
             pi_class=pi_class.astype(np.float32),
             rho=rho.astype(np.float32),
             w_un=np.float32(float(w_un)))
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump({
            'source_ckpt': args.k8_state,
            'contacts_file': args.contacts,
            'K_c': K_c, 'A': A, 'n_pairs': n_pairs,
            'n_cherries': int(len(tau_bin)),
            'n_em_iters': args.n_em_iters,
            'n_mstep_iters': args.n_mstep_iters,
            'lr': args.lr, 'prior_tau': args.prior_tau,
            'seed': args.seed, 'warm_start': args.warm_start,
            'init_w_un': args.init_w_un, 'final_w_un': float(w_un),
            'beta_a': args.beta_a, 'beta_b': args.beta_b,
            'final_marg_loglik': float(history[-1]['marg_loglik']),
            'final_norm_H': float(history[-1]['norm_H']),
            'final_mean_gamma_un': float(history[-1]['mean_gamma_un']),
        }, f, indent=2)
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\nfinal marg_loglik = {history[-1]["marg_loglik"]:.1f}')
    print(f'final ‖H‖_F = {history[-1]["norm_H"]:.3f}')
    print(f'final w_un = {float(w_un):.3f}  (mean γ_un = {history[-1]["mean_gamma_un"]:.3f})')


if __name__ == '__main__':
    main()
