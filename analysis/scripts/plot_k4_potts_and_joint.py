#!/usr/bin/env python3
"""K=4 model-anatomy figures.

Renders the underlying structure of the K=4 emwarm model — i.e. the
things the model actually parameterises, before any inference is run
through it:

  Figure A (k4_potts_average.{pdf,png}):
    The class-CRP-weighted average Potts tensor
        H_avg[a, b] = sum_{c1, c2} pi_c[c1] pi_c[c2]
                       * atoms[ assignments[c1, c2] ][a, b]
    rendered as a single A x A heatmap, alongside the class prior
    pi_c bar.

  Figure B (k4_stationary_joint.{pdf,png}):
    Three-panel comparison: the K=4 stationary joint at a coupled
    cell-pair P_coupled(a, b) (canonical convention: LG08 pair
    background, empirical pi_c, t -> 0), the independent product
    P_indep(a, b) = pi(a) pi(b), and the residual P_coupled - P_indep.

The companion log_2 odds heatmap is in plot_k4_coupling_and_doublet.py
(figure k4_coupling_score), and the 400x400 doublet substitution
sanity check is in the same script (figure k4_doublet_substitution).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

from tkfdp.block_likelihoods import (                          # noqa: E402
    build_M_tensor, build_doublet_emission,
    build_singlet_emission,
    empirical_pi_c_from_checkpoint)
from tkfdp.lg08 import PI_LG08_J                                # noqa: E402
from tkfdp.potts_dp import PottsDPState                         # noqa: E402

AA = 'ACDEFGHIKLMNPQRSTVWY'
A = 20


def _load_state(ckpt: str):
    d = np.load(ckpt, allow_pickle=True)

    class _S:
        pass
    s = _S()
    s.K_c = int(d['pi_class'].shape[0])
    s.A = A
    s.pi_class = np.asarray(d['pi_class'], dtype=np.float32)
    s.potts_dp = PottsDPState(
        K_c=s.K_c, A=A,
        atoms=np.asarray(d['potts_atoms'], dtype=np.float32),
        assignments=np.asarray(d['potts_assignments'], dtype=np.int64),
        counts=np.asarray(d['potts_counts'], dtype=np.int64),
        alpha_H=1.0)
    return s


def figure_potts_average(state, pi_c, out_path: Path):
    """Render the class-CRP-weighted average Potts tensor

        H_avg[a, b] = sum_{c1, c2} pi_c[c1] pi_c[c2]
                      * atoms[ assignments[c1, c2] ][a, b]

    as a single A x A heatmap, plus the empirical pi_c bar.
    """
    atoms = np.asarray(state.potts_dp.atoms, dtype=np.float64)    # (K_H, A, A)
    assignments = np.asarray(state.potts_dp.assignments,
                              dtype=np.int64)                       # (K_c, K_c)
    K_H = atoms.shape[0]
    K_c = assignments.shape[0]
    pi_c = np.asarray(pi_c, dtype=np.float64)

    # H_avg[a, b] = sum_{c1, c2} pi_c[c1] pi_c[c2] atoms[a(c1,c2)][a, b]
    H_per_pair = atoms[assignments]                  # (K_c, K_c, A, A)
    pair_w = pi_c[:, None] * pi_c[None, :]           # (K_c, K_c)
    H_avg = (pair_w[:, :, None, None] * H_per_pair).sum(axis=(0, 1))   # (A, A)

    # Also compute pair weight per atom for the caption.
    atom_w = np.zeros(K_H, dtype=np.float64)
    for c1 in range(K_c):
        for c2 in range(K_c):
            atom_w[assignments[c1, c2]] += pair_w[c1, c2]

    fig = plt.figure(figsize=(8, 7.0))
    # Convention: pi_joint ~ exp(-H_avg). So NEGATIVE H means ENRICHMENT
    # (e.g. C-C: H_avg = -1.04 -> pi_joint(C, C) > pi(C)^2). We use
    # 'coolwarm_r' (red = negative = enriched, blue = positive = depleted)
    # so the figure's visual semantics match the joint-distribution panels.
    ax = fig.add_subplot(1, 1, 1)
    vmax = float(np.max(np.abs(H_avg)))
    im = ax.imshow(H_avg, cmap='coolwarm_r', vmin=-vmax, vmax=vmax,
                    interpolation='nearest', origin='upper')
    ax.set_xticks(range(A)); ax.set_yticks(range(A))
    ax.set_xticklabels(list(AA), fontsize=9)
    ax.set_yticklabels(list(AA), fontsize=9)
    ax.set_xlabel('AA at site 2', fontsize=10)
    ax.set_ylabel('AA at site 1', fontsize=10)
    ax.set_title(
        r'K=4 class-CRP-averaged Potts coupling '
        r'$\bar H(a, b) = \sum_{c_1, c_2}\pi_{c_1}\pi_{c_2}\,H_{[c_1, c_2]}(a, b)$'
        + '\n(convention $\\pi_{\\rm joint} \\propto \\exp(-\\bar H)$: '
        + r'red = $\bar H < 0$ = enriched, blue = depleted)',
        fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\bar H[a, b]$ (more negative $\to$ more enriched)',
                    fontsize=10)

    # Annotate top-magnitude cells.
    thresh = vmax * 0.4
    for a in range(A):
        for b in range(A):
            v = H_avg[a, b]
            if abs(v) >= thresh:
                color = 'white' if abs(v) > vmax * 0.6 else 'black'
                ax.text(b, a, f'{v:+.2f}', ha='center', va='center',
                        fontsize=6, color=color)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path} + .png')
    print(f'  H_avg range: [{H_avg.min():+.3f}, {H_avg.max():+.3f}], '
          f'mean abs = {np.abs(H_avg).mean():.3f}')
    # Top-5 positive and negative cells (upper triangle).
    flat = sorted(((H_avg[a, b], AA[a], AA[b]) for a in range(A)
                   for b in range(a, A)), reverse=True)
    print('  Top-5 positive H_avg cells (upper triangle):')
    for v, a, b in flat[:5]:
        print(f'    {a}-{b}: {v:+.3f}')
    print('  Top-5 negative H_avg cells:')
    for v, a, b in flat[-5:][::-1]:
        print(f'    {a}-{b}: {v:+.3f}')
    print(f'  Per-atom CRP weight: '
          f'{[f"{w:.3f}" for w in atom_w]}')


def figure_stationary_joint(state, pi_c, out_path: Path):
    """Render the K=4 stationary joint at a coupled site-pair under
    the canonical convention, alongside the independent product
    PI_LG08(a) * PI_LG08(b) and the residual (joint - indep).

    Uses M = build_M_tensor at t -> 0 to extract
        pi_joint(a, b) = sum_{c1, c2} pi_c(c1) pi_c(c2)
                          * pi_joint^{c1, c2}(a, b)
                       = P_coupled[a, a, b, b] * P_indep(a)P_indep(b)
                                                  / nothing
    Actually we compute it directly via build_doublet_emission at
    t -> 0, where the joint emission collapses to delta(anc, desc)
    times pi_joint -- so pi_joint(a, b) = P_doublet[a, a, b, b] at
    small t (renormalised so rows sum to pi_marg).
    """
    # build_doublet_emission has indices (a, b, c, d) = (X-AA at site 1,
    # Y-AA at site 1, X-AA at site 2, Y-AA at site 2). At t -> 0 the
    # joint P_doublet[a, b, c, d] ~= pi_joint(a, c) * delta(a, b) *
    # delta(c, d). So:
    #   pi_joint(a, c) = P_doublet[a, a, c, c]   (extracted at small t)
    P_doublet = build_doublet_emission(
        state, 1e-4, pi_c=pi_c, pair_background='lg08')
    pi_joint = np.zeros((A, A), dtype=np.float64)
    for a in range(A):
        for c in range(A):
            pi_joint[a, c] = P_doublet[a, a, c, c]
    # Should sum to ~1 but normalise to be safe.
    pi_joint = pi_joint / pi_joint.sum()

    pi_marg = pi_joint.sum(axis=1)
    pi_indep = np.outer(pi_marg, pi_marg)

    residual = pi_joint - pi_indep

    # Make the heatmaps log-scale for the two probability panels so
    # the dynamic range is readable (some entries ~ 0.05, others ~ 1e-4).
    log10_joint = np.log10(np.maximum(pi_joint, 1e-12))
    log10_indep = np.log10(np.maximum(pi_indep, 1e-12))
    vmin = min(log10_joint.min(), log10_indep.min())
    vmax = max(log10_joint.max(), log10_indep.max())

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    im0 = axes[0].imshow(log10_joint, cmap='viridis',
                          vmin=vmin, vmax=vmax,
                          interpolation='nearest', origin='upper')
    axes[0].set_title(r'$\log_{10} P_{\mathrm{coupled}}(a, b)$' + '\n'
                       r'(K=4 stationary joint, canonical conv., $t \to 0$)',
                       fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(log10_indep, cmap='viridis',
                          vmin=vmin, vmax=vmax,
                          interpolation='nearest', origin='upper')
    axes[1].set_title(r'$\log_{10} P_{\mathrm{indep}}(a, b) = \log_{10}[\pi(a)\,\pi(b)]$'
                       + '\n(induced LG08-pair marginal under empirical $\\pi_c$)',
                       fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    rmax = float(np.max(np.abs(residual)))
    im2 = axes[2].imshow(residual, cmap='RdBu_r',
                          vmin=-rmax, vmax=rmax,
                          interpolation='nearest', origin='upper')
    axes[2].set_title(r'$P_{\mathrm{coupled}} - P_{\mathrm{indep}}$' + '\n'
                       '(red = enriched at coupled site, blue = depleted)',
                       fontsize=10)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks(range(A)); ax.set_yticks(range(A))
        ax.set_xticklabels(list(AA), fontsize=8)
        ax.set_yticklabels(list(AA), fontsize=8)
        ax.set_xlabel('AA at site 2', fontsize=9)
        ax.set_ylabel('AA at site 1', fontsize=9)

    # Annotate top-5 positive residual cells on the residual panel.
    flat = sorted(((residual[a, b], a, b) for a in range(A)
                   for b in range(a, A)), reverse=True)
    top_positive = flat[:5]
    top_negative = flat[-5:][::-1]
    for v, a, b in top_positive + top_negative:
        axes[2].text(b, a, f'{v:+.3f}', ha='center', va='center',
                      fontsize=6,
                      color='white' if abs(v) > rmax * 0.5 else 'black')

    fig.suptitle('K=4 stationary joint at a coupled site-pair vs the '
                 'induced independent product',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path} + .png')
    print('  Top-5 enriched cells (P_coupled - P_indep):')
    for v, a, b in top_positive:
        print(f'    {AA[a]}-{AA[b]}: {v:+.4f}')
    print('  Top-5 depleted cells:')
    for v, a, b in top_negative:
        print(f'    {AA[a]}-{AA[b]}: {v:+.4f}')
    print(f'  pi_marg deviation from PI_LG08: '
          f'max |.| = {float(np.max(np.abs(pi_marg - np.asarray(PI_LG08_J)))):.4f}')


def figure_potts_and_mi(state, pi_c, out_path: Path,
                         ts=None):
    """Two-panel figure for main.tex fig:k4-potts-atoms.

    LEFT: class-CRP-weighted average Potts coupling tensor
        H_avg[a, b] = sum_{c1, c2} pi_c[c1] pi_c[c2]
                       * atoms[ assignments[c1, c2] ][a, b]
    (coolwarm_r so red = negative H = enriched).

    RIGHT: the "coupled-trajectory mutual information" curve
        E_t = E_{(X(0), X(t)) ~ P_doublet}
                [ log_2 (P_doublet / (P_singlet x P_singlet)) ]
    where X = (AA at site 1, AA at site 2) is the coupled doublet
    state and (P_singlet x P_singlet) uses the canonical-convention
    singlet (which is NOT quite the doublet's marginal: max|pi_marg -
    pi_singlet| approx 0.02 in this model). Computed on a log-spaced
    grid of t and plotted on a log-x axis.
    """
    atoms = np.asarray(state.potts_dp.atoms, dtype=np.float64)
    assignments = np.asarray(state.potts_dp.assignments, dtype=np.int64)
    K_c = assignments.shape[0]
    pi_c = np.asarray(pi_c, dtype=np.float64)

    # LEFT: H_avg
    H_per_pair = atoms[assignments]
    pair_w = pi_c[:, None] * pi_c[None, :]
    H_avg = (pair_w[:, :, None, None] * H_per_pair).sum(axis=(0, 1))

    # RIGHT: sweep t and compute E_t
    if ts is None:
        ts = np.concatenate([
            np.array([1e-4]),                      # the t -> 0 limit point
            np.logspace(-3, 2, 24),                 # main sweep
        ])
    Es = np.zeros(len(ts), dtype=np.float64)
    for i, t in enumerate(ts):
        P_d = build_doublet_emission(state, float(t),
                                       pi_c=pi_c, pair_background='lg08')
        P_s, _, _ = build_singlet_emission(state, float(t), pi_c=pi_c)
        P_indep = (P_s[:, :, None, None]
                    * P_s[None, None, :, :])
        mask = (P_d > 1e-300) & (P_indep > 1e-300)
        log_ratio = np.where(mask,
            np.log2(P_d / np.where(P_indep > 1e-300, P_indep, 1.0)),
            0.0)
        Es[i] = float((P_d * log_ratio).sum())

    fig = plt.figure(figsize=(15, 6.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.3)

    # LEFT panel
    axL = fig.add_subplot(gs[0, 0])
    vmax = float(np.max(np.abs(H_avg)))
    imL = axL.imshow(H_avg, cmap='coolwarm_r', vmin=-vmax, vmax=vmax,
                      interpolation='nearest', origin='upper')
    axL.set_xticks(range(A)); axL.set_yticks(range(A))
    axL.set_xticklabels(list(AA), fontsize=8)
    axL.set_yticklabels(list(AA), fontsize=8)
    axL.set_xlabel('AA at site 2', fontsize=9)
    axL.set_ylabel('AA at site 1', fontsize=9)
    axL.set_title(
        r'Potts interaction score $\bar H(a, b) = '
        r'\sum_{c_1, c_2} \pi_{c_1} \pi_{c_2}\,H_{[c_1, c_2]}(a, b)$' + '\n'
        r'(convention $\pi_{\rm joint} \propto \exp(-\bar H)$: '
        r'red = enriched, blue = depleted)', fontsize=10)
    cbarL = plt.colorbar(imL, ax=axL, fraction=0.046, pad=0.04)
    cbarL.set_label(r'$\bar H[a, b]$', fontsize=10)
    # Annotate cells with |H_avg| above 0.4 * vmax.
    thresh = vmax * 0.4
    for a in range(A):
        for b in range(A):
            v = H_avg[a, b]
            if abs(v) >= thresh:
                color = 'white' if abs(v) > vmax * 0.6 else 'black'
                axL.text(b, a, f'{v:+.2f}', ha='center', va='center',
                          fontsize=5, color=color)

    # RIGHT panel
    axR = fig.add_subplot(gs[0, 1])
    axR.semilogx(ts, Es, marker='o', color='C0', linewidth=2.0,
                  markersize=5)
    axR.axhline(Es[0], color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
    axR.axhline(Es[-1], color='gray', linewidth=0.7, linestyle=':', alpha=0.5)
    axR.set_xlabel(r'branch length $t$', fontsize=10)
    axR.set_ylabel(r'$E_t\;[\log_2 P_{\mathrm{doublet}} / '
                    r'(P_{\mathrm{singlet}} P_{\mathrm{singlet}})]$',
                    fontsize=10)
    axR.set_title(
        r'Coupled-trajectory mutual information $E_t$ vs $t$' + '\n'
        r'(expectation under $P_{\mathrm{doublet}}$ of the '
        r'log-odds-ratio against the singlet product)',
        fontsize=10)
    axR.grid(True, alpha=0.3)
    # Annotate the t -> 0, peak, and t -> inf values.
    i_peak = int(np.argmax(Es))
    axR.annotate(f'$E_0 \\approx {Es[0]:.3f}$',
                  xy=(ts[0], Es[0]), xytext=(ts[0] * 2, Es[0] - 0.05),
                  fontsize=9,
                  arrowprops={'arrowstyle': '-', 'color': 'gray',
                               'alpha': 0.5})
    axR.annotate(f'peak $\\approx {Es[i_peak]:.3f}$ at $t \\approx {ts[i_peak]:.2g}$',
                  xy=(ts[i_peak], Es[i_peak]),
                  xytext=(ts[i_peak] * 1.5, Es[i_peak] + 0.03),
                  fontsize=9,
                  arrowprops={'arrowstyle': '->', 'color': 'gray',
                               'alpha': 0.5})
    axR.annotate(f'$E_\\infty \\approx {Es[-1]:.3f}$',
                  xy=(ts[-1], Es[-1]),
                  xytext=(ts[-1] * 0.05, Es[-1] - 0.05),
                  fontsize=9,
                  arrowprops={'arrowstyle': '-', 'color': 'gray',
                               'alpha': 0.5})

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path} + .png')
    print(f'  E(t->0) = {Es[0]:.4f}, peak {Es[i_peak]:.4f} at t={ts[i_peak]:.3g}, '
          f'asymp {Es[-1]:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(
        REPO / 'results' / 'K4-emwarm-top1000-2026-05-09'
        / '_best_chkpt' / 'state.npz'))
    ap.add_argument('--out-dir', type=Path,
                    default=REPO / 'math-paper' / 'figures')
    args = ap.parse_args()

    state = _load_state(args.ckpt)
    pi_c = empirical_pi_c_from_checkpoint(args.ckpt)
    print(f'Loaded K=4 state: K_c={state.K_c}, A={state.A}, '
          f'K_H={state.potts_dp.atoms.shape[0]}')
    print(f'Empirical pi_c = {[f"{x:.3f}" for x in pi_c]}')

    figure_potts_average(
        state, pi_c, args.out_dir / 'k4_potts_average.pdf')
    figure_stationary_joint(
        state, pi_c, args.out_dir / 'k4_stationary_joint.pdf')
    # main.tex figure (single-panel Potts heatmap; the MI-vs-t panel
    # was dropped per user direction).
    figure_potts_average(
        state, pi_c, args.out_dir / 'k4_potts_atoms.pdf')


if __name__ == '__main__':
    main()
