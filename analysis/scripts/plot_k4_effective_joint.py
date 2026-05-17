#!/usr/bin/env python3
"""Effective 20x20 joint stationary distribution at two coupled sites,
marginalised over the K=4 site-class assignments. Replaces the
ten-Potts-atom panel.

Canonical (default) convention:
  pair_background='lg08'   -- the joint generator Q_pair^{c1,c2} uses
                              pi_a = pi_b = PI_LG08_J for every (c1, c2).
                              MATCHES how the released Potts atoms were
                              trained (multiclass.py's
                              composite_log_likelihood_K calls
                              build_joint_Q(H) with pi=PI_LG08_J).
  pi_c_source='empirical'  -- pi_c is the empirical class prior from
                              the SVI training-set column class
                              assignments (cls_* arrays in the
                              checkpoint, ~ [0.20, 0.23, 0.33, 0.24]
                              for the K=4 emwarm release).

Legacy interim convention:
  pair_background='per_class'  -- pi_a = pi_class[c1], pi_b = pi_class[c2].
  pi_c_source='uniform'        -- pi_c[c] = 1/K_c.

The panel builds the M-tensor via build_M_tensor in
src/tkfdp/block_likelihoods.py (math-verifier-clean since 055947c).
At t -> 0, x = y on both endpoints, the diagonal slice
M[a, a, c, c] approximates the stationary log-odds
pi_joint(a, c) / (pi_marg(a) * pi_marg(c)) -- the analogue of the
plmDCA / Marks et al. cherry-pair "interaction" matrix.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make src/ importable. The script lives at analysis/scripts/, src/ at top.
_SRC = (Path(__file__).resolve().parent.parent.parent / 'src').as_posix()
import sys
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tkfdp.block_likelihoods import (build_M_tensor,
                                       empirical_pi_c_from_checkpoint)
from tkfdp.svi import SVIState
from tkfdp.potts_dp import PottsDPState

AA_ORDER = list('ACDEFGHIKLMNPQRSTVWY')
A = 20


def _load_state(ckpt_path: str) -> SVIState:
    """Load a SVIState skeleton sufficient for build_M_tensor: only
    pi_class and potts_dp.{atoms, assignments, h_pairs} are read."""
    d = np.load(ckpt_path, allow_pickle=True)
    pi_class = np.asarray(d['pi_class'], dtype=np.float64)
    K_c, A_ = pi_class.shape
    atoms = np.asarray(d['potts_atoms'], dtype=np.float64)
    assignments = np.asarray(d['potts_assignments'], dtype=np.int64)
    counts = np.asarray(d['potts_counts'], dtype=np.int64) \
        if 'potts_counts' in d.files else np.zeros(atoms.shape[0], dtype=np.int64)
    potts_dp = PottsDPState(K_c=K_c, A=A_, atoms=atoms,
                             assignments=assignments, counts=counts,
                             alpha_H=1.0, mu_prior=None, tau_prior=None,
                             rho=None, tsb_betas=None, h_pairs=None)
    return SVIState(K_c=K_c, A=A_, pi_class=pi_class, potts_dp=potts_dp,
                     states_per_msa=[], eta_per_msa=[])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',
                   default=str(Path('~/tkf-dp/results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz').expanduser()))
    p.add_argument('--out',
                   default=str(Path('~/tkf-dp/math-paper/figures/k4_potts_atoms.pdf').expanduser()),
                   help='Output PDF path (overwrites the per-atom panel).')
    p.add_argument('--pair-background', choices=['lg08', 'per_class'],
                   default='lg08',
                   help='LG08 (canonical, matches training) or per_class (legacy interim).')
    p.add_argument('--pi-c-source', choices=['empirical', 'uniform'],
                   default='empirical',
                   help='Class prior source: empirical from cls_* in checkpoint, '
                        'or uniform 1/K_c (legacy interim).')
    p.add_argument('--t', type=float, default=0.001,
                   help='Branch length t at which to evaluate M (default 1e-3, '
                        'effectively the t -> 0 stationary log-odds).')
    args = p.parse_args()

    state = _load_state(args.ckpt)
    K_c = state.K_c

    if args.pi_c_source == 'empirical':
        pi_c = empirical_pi_c_from_checkpoint(args.ckpt)
    else:
        pi_c = np.full(K_c, 1.0 / K_c, dtype=np.float64)
    print(f'pi_c ({args.pi_c_source}) = {pi_c}', flush=True)
    print(f'pair_background = {args.pair_background}', flush=True)
    print(f'evaluation t = {args.t}', flush=True)

    # M-tensor under the chosen convention. At t -> 0, M[a,a,c,c] is
    # the stationary log-odds for the coupled site pair (a, c).
    M = build_M_tensor(state, t=float(args.t), eta=1.0, pi_c=pi_c,
                        pair_background=args.pair_background, n_rate_bins=1)
    log_odds = np.zeros((A, A), dtype=np.float64)
    diag_M = np.zeros((A, A), dtype=np.float64)
    for a in range(A):
        for c in range(A):
            diag_M[a, c] = M[a, a, c, c]
    log_odds = np.log2(diag_M + 1e-30)

    # The "joint" panel can show pi_joint(a, c) = pi_marg(a) * pi_marg(c) * 2^log_odds.
    # We compute pi_marg from the singlet emission diagonal.
    from tkfdp.block_likelihoods import build_singlet_emission
    P_singlet, pi_out_eff, _ = build_singlet_emission(
        state, t=float(args.t), eta=1.0, pi_c=pi_c, n_rate_bins=1)
    pi_marg = np.zeros(A, dtype=np.float64)
    for a in range(A):
        pi_marg[a] = P_singlet[a, a]    # = pi_marg(a) at t -> 0
    pi_marg /= pi_marg.sum()  # safety-normalise

    indep = pi_marg[:, None] * pi_marg[None, :]
    indep /= indep.sum()
    joint = indep * (2.0 ** log_odds)
    joint /= joint.sum()

    fig, (ax_j, ax_lo) = plt.subplots(1, 2, figsize=(11, 5.2),
                                       constrained_layout=True)

    im_j = ax_j.imshow(joint, cmap='viridis', aspect='equal')
    ax_j.set_title(r'$\pi_{\mathrm{joint}}(a, b)$' +
                    f'\n(effective joint, K={K_c} site classes,\n'
                    f'{args.pair_background} pair-stationary, '
                    f'{args.pi_c_source} $\\pi_c$)', fontsize=10)
    ax_j.set_xticks(range(A)); ax_j.set_yticks(range(A))
    ax_j.set_xticklabels(AA_ORDER, fontsize=7, rotation=90)
    ax_j.set_yticklabels(AA_ORDER, fontsize=7)
    ax_j.set_xlabel('amino acid $b$ (site 2)')
    ax_j.set_ylabel('amino acid $a$ (site 1)')
    cb_j = fig.colorbar(im_j, ax=ax_j, fraction=0.045, pad=0.02)
    cb_j.set_label(r'probability', fontsize=9)

    vmax = float(np.abs(log_odds).max())
    im_lo = ax_lo.imshow(log_odds, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          aspect='equal')
    ax_lo.set_title(r'$\log_2 \pi_{\mathrm{joint}}(a, b) / '
                     r'[\pi_{\mathrm{marg}}(a) \pi_{\mathrm{marg}}(b)]$' +
                     '\n(coupling-induced deviation\n'
                     'from independence; red = attractive)',
                     fontsize=10)
    ax_lo.set_xticks(range(A)); ax_lo.set_yticks(range(A))
    ax_lo.set_xticklabels(AA_ORDER, fontsize=7, rotation=90)
    ax_lo.set_yticklabels(AA_ORDER, fontsize=7)
    ax_lo.set_xlabel('amino acid $b$ (site 2)')
    cb_lo = fig.colorbar(im_lo, ax=ax_lo, fraction=0.045, pad=0.02)
    cb_lo.set_label(r'$\log_2$ odds ratio', fontsize=9)

    fig.suptitle(
        f'Effective joint stationary at a coupled site pair, K={K_c} TKF-DP\n'
        f'(\\texttt{{results/K4-emwarm-top1000-2026-05-09}}, '
        f'{args.pair_background}, {args.pi_c_source}, $t = {args.t}$)',
        fontsize=11, y=1.04)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    print(f'wrote {args.out}', flush=True)

    # Diagnostic: top-15 / bottom-15 coupled pairs by log-odds.
    flat = [(log_odds[i, j], AA_ORDER[i], AA_ORDER[j])
             for i in range(A) for j in range(A)]
    flat.sort(reverse=True)
    print('\nTop 15 attractive pairs (highest log-odds):')
    for lo, a, b in flat[:15]:
        print(f'  {a}-{b}: log2-odds = {lo:+.4f}')
    print('\nBottom 15 repulsive pairs (most negative log-odds):')
    for lo, a, b in flat[-15:]:
        print(f'  {a}-{b}: log2-odds = {lo:+.4f}')


if __name__ == '__main__':
    main()
