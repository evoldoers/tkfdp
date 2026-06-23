"""Compute val_log_likelihood_class_marginal for a supervised-H checkpoint,
swapping the H into the unsupervised K=8 SVIState shell. Same val
families / MCMC config as the unsupervised K=8 training run, so the
val_LL numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-chkpt',
                    default='/home/yam/tkf-dp/results/exp2_v2_K8_KH1_top8000_2026-05-22/_best_chkpt',
                    help='Unsupervised K=8 checkpoint dir (provides SVIState shell).')
    ap.add_argument('--supervised-state',
                    required=True,
                    help='state.npz containing the supervised H + pi_class.')
    ap.add_argument('--processed-dir',
                    default='/home/yam/tkf-dp/data/pfam_processed_top8000',
                    help='Per-family cherry NPZ dir.')
    ap.add_argument('--val-families',
                    default=('PF00076,PF00003,PF00004,PF00006,PF00009,PF00010,'
                             'PF00029,PF00032,PF00046,PF00052,PF00074,PF00080,'
                             'PF00086,PF00096,PF00100,PF00104,PF00110,PF00112,'
                             'PF00114,PF00115,PF00119,PF00121,PF00141,PF00144,'
                             'PF00146,PF00149,PF00158,PF00162,PF00168,PF00174,'
                             'PF00177,PF00187,PF00193,PF00194,PF00199,PF00205,'
                             'PF00210,PF00219,PF00220,PF00221,PF00227,PF00231,'
                             'PF00233,PF00234,PF00235,PF00237,PF00238,PF00241,'
                             'PF00244,PF00249'),
                    help='Comma-separated val family list (matches training).')
    ap.add_argument('--n-burnin', type=int, default=50)
    ap.add_argument('--n-samples', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    sys.path.insert(0, '/home/yam/tkf-dp/src')
    from tkfdp.val_loglik_v2 import val_log_likelihood_class_marginal
    from tkfdp.checkpoint import load_globals_from_checkpoint
    from tkfdp.pfam_data import families_from_list

    # Load unsupervised base SVIState shell
    base_chkpt = Path(args.base_chkpt)
    pi_class, potts_dp, meta = load_globals_from_checkpoint(
        base_chkpt,
        mu_prior=np.zeros(20 * 21 // 2),
        tau_prior=np.ones(20 * 21 // 2) * 0.5,
    )
    K_c = int(meta['K_c'])
    print(f'Loaded base shell: K_c={K_c}, pi_class shape={pi_class.shape}')

    # Substitute the supervised H
    sup = np.load(args.supervised_state, allow_pickle=True)
    H_sup = np.asarray(sup['potts_atoms'])
    assert H_sup.shape == potts_dp.atoms.shape, (
        f'shape mismatch: base atoms {potts_dp.atoms.shape} vs '
        f'supervised {H_sup.shape}')
    print(f'Substituting supervised H: '
          f'‖H_base‖_F={np.linalg.norm(potts_dp.atoms):.3f} '
          f'→ ‖H_sup‖_F={np.linalg.norm(H_sup):.3f}')
    potts_dp = potts_dp._replace(atoms=H_sup) \
        if hasattr(potts_dp, '_replace') else potts_dp
    # PottsDPState is a regular dataclass; field assignment works:
    try:
        object.__setattr__(potts_dp, 'atoms', H_sup)
    except Exception:
        pass

    # Empirical pi_c — use the one stored with the supervised checkpoint
    # if it's K_c-shaped; fall back to base.
    pi_c_sup = np.asarray(sup['rho']) if 'rho' in sup.files else None
    if pi_c_sup is None or pi_c_sup.shape != (K_c,):
        # Fall back: empirical pi_c from base checkpoint
        import re
        base_arr = np.load(base_chkpt / 'state.npz')
        total = np.zeros(K_c, dtype=np.int64)
        for k in base_arr.files:
            if re.match(r'cls_\d+$', k):
                total += np.bincount(base_arr[k].astype(int), minlength=K_c)
        pi_c_sup = total / total.sum()
    print(f'pi_c for class-marginalization: {pi_c_sup.round(3).tolist()}')

    # Load val families directly by name (the original training corpus
    # may not include all of them in index.json — they're held-out).
    # Match training default of min_cherries=8.
    val_ids = [f.strip() for f in args.val_families.split(',') if f.strip()]
    val_families = families_from_list(val_ids, min_cherries=8)
    print(f'{len(val_families)} val families, '
          f'{sum(fc.n_cherries for fc in val_families)} cherries total')

    # Build minimal SVIState. We need .K_c, .A, .pi_class, .potts_dp,
    # .a_eta, .b_eta. The rest is per-family latents we don't need
    # for the val MCMC (val builds its own partition state).
    a_eta = float(meta.get('a_eta', 2.0))
    b_eta = float(meta.get('b_eta', 2.0))
    A = pi_class.shape[1]

    # Construct a fake SVIState-like namespace
    class _Shell:
        pass
    state = _Shell()
    state.K_c = K_c
    state.A = A
    state.pi_class = pi_class
    state.potts_dp = potts_dp
    state.a_eta = a_eta
    state.b_eta = b_eta
    state.states_per_msa = []  # unused by class-marginal val

    print(f'\nRunning val_log_likelihood_class_marginal '
          f'(n_burnin={args.n_burnin}, n_samples={args.n_samples})...')
    t0 = time.time()
    sum_score_mean, results = val_log_likelihood_class_marginal(
        state, val_families, pi_c=pi_c_sup,
        n_burnin=args.n_burnin, n_samples=args.n_samples,
        seed=args.seed, verbose=args.verbose,
    )
    elapsed = time.time() - t0

    print(f'\nval_LL (sum of per-family score_mean): {sum_score_mean:.2f}')
    print(f'wall time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
