"""Fit a supervised Potts H matrix on PDB-contact column-pair cherry
counts, with the K trained TKF-DP site classes as a fixed mixture
background.

The model: each contact column-pair (i, j) has an unknown class-pair
assignment (c_i, c_j) drawn from independent prior  ρ_{c_i} ρ_{c_j}.
Given (c_i, c_j), the cherry observations (anc, des, tau) for that
pair are emitted from the 400x400 coupled rate matrix Q_{c_i, c_j}(H)
built from a SHARED H plus per-class equilibria π_{c_i}, π_{c_j}.

EM:
  E-step: γ_{ij}(c, c') = posterior over class-pair, given current H.
  M-step: maximize Σ_{ij, c, c'} γ_{ij}(c, c') log P(data_ij | c, c', H)
            with respect to H, via Adam (or LBFGS); shape (A, A) symmetric.

Outputs to <out-dir>/state.npz {potts_atoms, pi_class, rho} so the
downstream tooling (sweep_infinite_phmm_balibase, etc.) can consume it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# float64 is required: eigh-gradient through symmetrize_eigh is unstable
# in float32 at K_c^2 * 400 eigenvalues (degenerate or near-degenerate
# eigenvalues blow up the 1/(λ_i - λ_j) terms in the autodiff rule).
os.environ.setdefault('JAX_ENABLE_X64', '1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contacts', required=True,
                    help='Output .npz from build_supervised_contacts.py')
    ap.add_argument('--k8-state',
                    default='/home/yam/tkf-dp/results/_preserved/K8_KH1_top8000_iter68/state.npz',
                    help='K=8 TKF-DP checkpoint for π_class and ρ.')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-em-iters', type=int, default=30)
    ap.add_argument('--n-mstep-iters', type=int, default=50,
                    help='Adam steps per M-step.')
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--prior-tau', type=float, default=0.1,
                    help='L2 prior on H off-diagonal entries.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--warm-start', action='store_true',
                    help='Initialize H from the K=8 checkpoint atom.')
    args = ap.parse_args()

    sys.path.insert(0, '/home/yam/tkf-dp/src')
    import jax
    import jax.numpy as jnp
    import re
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
    print(f'  {len(anc1):,} cherries across {n_pairs:,} column-pairs, '
          f'n_time_bins={n_t}')

    print(f'Loading K=8 state from {args.k8_state}')
    k8 = np.load(args.k8_state, allow_pickle=True)
    pi_class = np.asarray(k8['pi_class'], dtype=np.float64)  # (K_c, A)
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
    S_j = jnp.asarray(S_LG08_F81)
    tau_centers_j = jnp.asarray(tau_centers)

    # Precompute per-pair cherry tensors:
    # Aggregate "weighted counts" by class-pair * time_bin * (start_400, end_400)
    # is the natural M-step accumulator. We'll do E-step per-pair, then
    # aggregate.

    # First: assemble per-pair index slices (cherry observations per pair)
    sort_idx = np.argsort(pair_idx, kind='stable')
    pair_idx_s = pair_idx[sort_idx]
    anc1 = anc1[sort_idx]; anc2 = anc2[sort_idx]
    des1 = des1[sort_idx]; des2 = des2[sort_idx]
    tau_bin = tau_bin[sort_idx]
    boundaries = np.concatenate([[0], np.where(np.diff(pair_idx_s) > 0)[0] + 1,
                                  [len(pair_idx_s)]])
    print(f'  per-pair boundaries: {len(boundaries) - 1} pairs')

    # Aggregate per-pair cherry counts as (n_pairs, n_t, A2*A2) sparse-ish.
    # Actually use a flat per-cherry tensor and do gather/aggregate.
    anc1_j = jnp.asarray(anc1.astype(np.int32))
    anc2_j = jnp.asarray(anc2.astype(np.int32))
    des1_j = jnp.asarray(des1.astype(np.int32))
    des2_j = jnp.asarray(des2.astype(np.int32))
    tau_bin_j = jnp.asarray(tau_bin.astype(np.int32))
    pair_idx_j = jnp.asarray(pair_idx_s.astype(np.int32))
    start_400 = anc1_j * A + anc2_j   # (Ncherries,)
    end_400   = des1_j * A + des2_j   # (Ncherries,)
    A2 = A * A

    @jax.jit
    def build_logP_all(H):
        """Return log_P[(c, c'), t, start_400, end_400] of shape (K_c, K_c, n_t, A2, A2)."""
        H_sym = 0.5 * (H + H.T)
        def per_cc(c1, c2):
            pi1, pi2 = pi_class_j[c1], pi_class_j[c2]
            Q = build_joint_Q_pair(H_sym, pi1, pi2, S=S_j)
            pi_j = joint_stationary_pair(H_sym, pi1, pi2)
            Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
            return log_transition_matrices(tau_centers_j, Lambda, U_sym, sqrt_pij)
        # vmap over c2 inside c1
        log_P = jax.vmap(lambda c1: jax.vmap(lambda c2: per_cc(c1, c2))(jnp.arange(K_c)))(jnp.arange(K_c))
        return log_P  # (K_c, K_c, n_t, A2, A2)

    @jax.jit
    def cherry_loglik_per_pair_cc(H):
        """Return log_lik[pair, c, c'] = sum_cherries log P(start, end | c, c', t_bin)."""
        log_P = build_logP_all(H)  # (K_c, K_c, n_t, A2, A2)
        # Gather: for each cherry, log_P[:, :, tau_bin, start_400, end_400]
        lp_per_cherry = log_P[:, :, tau_bin_j, start_400, end_400]  # (K_c, K_c, N)
        # Sum over cherries within each pair: use segment_sum on the last axis
        per_pair = jax.ops.segment_sum(
            lp_per_cherry.transpose(2, 0, 1),  # (N, K_c, K_c)
            pair_idx_j, num_segments=n_pairs)   # (n_pairs, K_c, K_c)
        return per_pair

    log_rho = jnp.log(jnp.asarray(rho) + 1e-30)
    log_rho_pair = log_rho[:, None] + log_rho[None, :]  # (K_c, K_c)

    @jax.jit
    def e_step(H):
        """Return γ[pair, c, c'] = softmax over (c, c') of log_rho_pair + log_lik_per_pair_cc."""
        ll = cherry_loglik_per_pair_cc(H)  # (n_pairs, K_c, K_c)
        logits = ll + log_rho_pair
        m = jnp.max(logits.reshape(n_pairs, -1), axis=1, keepdims=True)
        ex = jnp.exp(logits.reshape(n_pairs, -1) - m)
        gamma = (ex / ex.sum(axis=1, keepdims=True)).reshape(n_pairs, K_c, K_c)
        log_marg = (m.squeeze(1) + jnp.log(ex.sum(axis=1)))
        return gamma, log_marg

    @jax.jit
    def expected_complete_loglik(H, gamma):
        """E_{γ}[Σ_ij Σ_{c,c'} γ_ij(c,c') Σ_cherries log P_t(start,end|c,c',H)] − prior.
        Normalized by n_pairs so the gradient magnitude is per-pair-scale.
        Prior also normalized so its weight (prior_tau) is in per-pair units."""
        ll = cherry_loglik_per_pair_cc(H)  # (n_pairs, K_c, K_c)
        ecll = (gamma * ll).sum() / float(n_pairs)
        H_sym = 0.5 * (H + H.T)
        off = H_sym - jnp.diag(jnp.diag(H_sym))
        prior = -0.5 * args.prior_tau * (off ** 2).sum() / float(n_pairs)
        return ecll + prior

    neg_obj = jax.jit(lambda H, gamma: -expected_complete_loglik(H, gamma))
    grad_obj = jax.jit(jax.grad(lambda H, gamma: -expected_complete_loglik(H, gamma)))

    # Adam state — keep float64 for numerical stability through eigh grad
    H = jnp.asarray(H_init, dtype=jnp.float64)
    m_state = jnp.zeros_like(H); v_state = jnp.zeros_like(H)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    print(f'\nStarting EM (n_em_iters={args.n_em_iters}, n_mstep={args.n_mstep_iters}, lr={args.lr})')
    history = []
    for it in range(args.n_em_iters):
        t0 = time.time()
        gamma, log_marg = e_step(H)
        marg_ll = float(log_marg.sum())
        # M-step: Adam on H with gradient clipping
        for ms in range(args.n_mstep_iters):
            g = grad_obj(H, gamma)
            # symmetrize gradient
            g = 0.5 * (g + g.T)
            # gradient clip (per-element)
            g = jnp.clip(g, -5.0, 5.0)
            t = it * args.n_mstep_iters + ms + 1
            m_state = beta1 * m_state + (1 - beta1) * g
            v_state = beta2 * v_state + (1 - beta2) * (g ** 2)
            mhat = m_state / (1 - beta1 ** t)
            vhat = v_state / (1 - beta2 ** t)
            H = H - args.lr * mhat / (jnp.sqrt(vhat) + eps)
            H = 0.5 * (H + H.T)
        ecll = float(expected_complete_loglik(H, gamma))
        norm_H = float(jnp.linalg.norm(H))
        elapsed = time.time() - t0
        history.append({
            'iter': it,
            'marg_loglik': marg_ll,
            'ecll': ecll,
            'norm_H': norm_H,
            'elapsed_s': elapsed,
        })
        print(f'  EM[{it+1}/{args.n_em_iters}]  marg_LL={marg_ll:.1f}  '
              f'ECLL={ecll:.1f}  ‖H‖={norm_H:.3f}  ({elapsed:.1f}s)')

    # Save state
    state_path = out_dir / 'state.npz'
    H_np = np.asarray(H)
    np.savez(state_path,
             potts_atoms=H_np[None, :, :].astype(np.float32),
             pi_class=pi_class.astype(np.float32),
             rho=rho.astype(np.float32),
             ema_history=np.array([h['marg_loglik'] for h in history], dtype=np.float32))
    print(f'wrote {state_path}')

    with open(out_dir / 'meta.json', 'w') as f:
        json.dump({
            'source_ckpt': args.k8_state,
            'contacts_file': args.contacts,
            'K_c': K_c, 'A': A, 'n_pairs': n_pairs, 'n_cherries': int(len(tau_bin)),
            'n_time_bins': n_t,
            'tau_centers': tau_centers.tolist(),
            'rho': rho.tolist(),
            'n_em_iters': args.n_em_iters,
            'n_mstep_iters': args.n_mstep_iters,
            'lr': args.lr,
            'prior_tau': args.prior_tau,
            'seed': args.seed,
            'warm_start': args.warm_start,
            'final_marg_loglik': float(history[-1]['marg_loglik']),
            'final_norm_H': float(history[-1]['norm_H']),
        }, f, indent=2)
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f'final marg_loglik = {history[-1]["marg_loglik"]:.1f}')
    print(f'final ‖H‖_F = {history[-1]["norm_H"]:.3f}')


if __name__ == '__main__':
    main()
