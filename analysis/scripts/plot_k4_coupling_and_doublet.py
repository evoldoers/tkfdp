#!/usr/bin/env python3
"""K=4 analytical figures: (1) 20x20 stationary coupling-score heatmap,
(2) 400x400 effective doublet substitution matrix rendered as a heatmap
with overlaid top-K bubbles per row.

Both purely analytical (no MCMC). Uses block_likelihoods directly
under the canonical convention (LG08 pair background, empirical pi_c
from training counts).

Usage:
    python analysis/scripts/plot_k4_coupling_and_doublet.py \
        --ckpt results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz \
        --t 1.0 --out-dir math-paper/figures
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

from tkfdp.block_likelihoods import (                       # noqa: E402
    build_M_tensor, build_doublet_emission,
    empirical_pi_c_from_checkpoint)
from tkfdp.potts_dp import PottsDPState                       # noqa: E402

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


def figure_coupling_score(state, pi_c, out_path: Path):
    """Figure 1: 20x20 heatmap of log_2 M[a, a, b, b] at t=0.

    Diverging colormap centred at 0. Cell (a, b) shows the stationary
    coupling score log_2[P(coupled site emits (a, b)) / (P(a) P(b))],
    i.e. how much MORE / less likely the model thinks (a, b) co-occurs
    at a coupled site than under independent emission.
    """
    M = build_M_tensor(state, 0.001, pi_c=pi_c, pair_background='lg08')
    score = np.zeros((A, A), dtype=np.float64)
    for a in range(A):
        for b in range(A):
            score[a, b] = np.log2(max(M[a, a, b, b], 1e-300))

    fig, ax = plt.subplots(figsize=(7.5, 7))
    vmax = float(np.max(np.abs(score)))
    im = ax.imshow(score, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                    interpolation='nearest', origin='upper')
    ax.set_xticks(range(A)); ax.set_yticks(range(A))
    ax.set_xticklabels(list(AA), fontsize=9)
    ax.set_yticklabels(list(AA), fontsize=9)
    ax.set_xlabel('AA at site 2', fontsize=10)
    ax.set_ylabel('AA at site 1', fontsize=10)
    ax.set_title(
        r'K=4 stationary coupling score: '
        r'$\log_2 [P_{\mathrm{coupled}}(a, b)\ /\ (P(a)\,P(b))]$' + '\n'
        r'canonical convention: LG08 pair background, empirical '
        r'$\pi_c=(0.20, 0.23, 0.33, 0.24)$, $t \to 0$',
        fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\log_2$ odds', fontsize=10)
    # Annotate top-magnitude cells
    thresh = 1.0
    for a in range(A):
        for b in range(A):
            v = score[a, b]
            if abs(v) >= thresh:
                color = ('white' if abs(v) > vmax * 0.6
                          else ('black' if abs(v) > 1.5 else '0.35'))
                ax.text(b, a, f'{v:+.1f}', ha='center', va='center',
                        fontsize=6, color=color)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path} + .png')
    # Sanity: surface top-5 positive and top-5 negative for the caption
    flat = sorted(((score[a, b], AA[a], AA[b]) for a in range(A)
                   for b in range(a, A)), reverse=True)
    print('  Top-5 positive (canonical convention):')
    for v, a, b in flat[:5]:
        print(f'    {a}-{b}: log2={v:+.3f}')
    print('  Top-5 negative:')
    for v, a, b in flat[-5:][::-1]:
        print(f'    {a}-{b}: log2={v:+.3f}')


def figure_doublet_substitution(state, pi_c, out_path: Path, t: float = 1.0,
                                 K_top: int = 3, bubble_threshold: float = 0.02):
    """Figure 2: 400x400 effective doublet substitution matrix.

    Heatmap of log10 P(descendant pair | ancestor pair; t) with cyan
    bubble overlays for the K_top most-probable descendants per row.

    Layout: rows/cols are pair states a*A + c (ancestor AA at site 1 =
    a, ancestor AA at site 2 = c). Pair states are grouped into 20
    super-rows of 20 each: super-row index = AA at site 1, intra-block
    index = AA at site 2.
    """
    # Build joint P_doublet[a, b, c, d] = P(observed (a, b) at left
    # endpoint, (c, d) at right endpoint; t) under the K=4 canonical
    # convention. Reshape to (A^2 ancestor, A^2 descendant):
    #   ancestor pair index = a * A + c (X-AA at site 1, X-AA at site 2)
    #   descendant pair index = b * A + d
    # Then renormalise rows to get P(desc | anc, t).
    P_doublet = build_doublet_emission(
        state, t, pi_c=pi_c, pair_background='lg08')   # (A, A, A, A)
    # Transpose (a, b, c, d) -> (a, c, b, d) and reshape:
    P_anc_desc = P_doublet.transpose(0, 2, 1, 3).reshape(A * A, A * A)
    row_sums = P_anc_desc.sum(axis=1, keepdims=True)
    P_cond = P_anc_desc / np.clip(row_sums, 1e-300, None)

    log_P = np.log10(np.maximum(P_cond, 1e-10))

    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(log_P, cmap='magma_r', vmin=-5, vmax=0,
                    interpolation='nearest', origin='upper')
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r'$\log_{10}\ P(\mathrm{desc.\ pair}\ |\ \mathrm{anc.\ pair};\ t)$',
                   fontsize=10)

    # Overlay top-K bubbles per row.
    n_bubbles = 0
    for row in range(A * A):
        top_cols = np.argsort(-P_cond[row])[:K_top]
        for col in top_cols:
            p = float(P_cond[row, col])
            if p < bubble_threshold:
                continue
            size = 60.0 * p   # max p = 1 -> size 60
            ax.scatter(col, row, s=size, facecolor='none',
                        edgecolor='cyan', linewidth=0.6, alpha=0.75,
                        zorder=3)
            n_bubbles += 1

    # Block separator lines every A=20 cells.
    for i in range(1, A):
        ax.axhline(i * A - 0.5, color='white', linewidth=0.4, alpha=0.4)
        ax.axvline(i * A - 0.5, color='white', linewidth=0.4, alpha=0.4)

    # Tick labels at block centres: AA at site 1.
    aa_centres = [i * A + A / 2 - 0.5 for i in range(A)]
    ax.set_xticks(aa_centres); ax.set_yticks(aa_centres)
    ax.set_xticklabels(list(AA), fontsize=10)
    ax.set_yticklabels(list(AA), fontsize=10)
    ax.set_xlabel('Descendant AA-pair (block = AA at site 1)', fontsize=10)
    ax.set_ylabel('Ancestor AA-pair (block = AA at site 1)', fontsize=10)
    ax.set_title(
        f'K=4 effective doublet substitution matrix at $t={t}$ '
        f'(canonical: LG08 pair bg, empirical $\\pi_c$)\n'
        f'heatmap: $\\log_{{10}} P(\\mathrm{{desc}}|\\mathrm{{anc}}; t)$; '
        f'cyan bubbles: top-{K_top} per row with $P \\geq {bubble_threshold}$ '
        f'(total {n_bubbles} bubbles)',
        fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path} + .png ({n_bubbles} bubbles)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(
        REPO / 'results' / 'K4-emwarm-top1000-2026-05-09'
        / '_best_chkpt' / 'state.npz'))
    ap.add_argument('--t', type=float, default=1.0,
                    help='Branch length for the doublet substitution matrix.')
    ap.add_argument('--out-dir', type=Path,
                    default=REPO / 'math-paper' / 'figures')
    ap.add_argument('--K-top', type=int, default=3,
                    help='Number of top transitions per row to bubble-mark.')
    ap.add_argument('--bubble-threshold', type=float, default=0.02,
                    help='Minimum P(desc | anc) to render a bubble.')
    args = ap.parse_args()

    state = _load_state(args.ckpt)
    pi_c = empirical_pi_c_from_checkpoint(args.ckpt)
    print(f'Loaded K=4 state: K_c={state.K_c}, A={state.A}')
    print(f'Empirical pi_c = {[f"{x:.3f}" for x in pi_c]}')

    figure_coupling_score(
        state, pi_c, args.out_dir / 'k4_coupling_score.pdf')
    figure_doublet_substitution(
        state, pi_c, args.out_dir / 'k4_doublet_substitution.pdf',
        t=args.t, K_top=args.K_top,
        bubble_threshold=args.bubble_threshold)


if __name__ == '__main__':
    main()
