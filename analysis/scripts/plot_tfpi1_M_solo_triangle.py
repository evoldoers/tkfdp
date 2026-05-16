#!/usr/bin/env python3
"""Render the single-sequence M_solo log-odds triangle on TFPI1_RABIT.

For a single sequence x of length L, the K=4 model's per-pair coupling
boost is M_solo[x_i, x_j] for every unordered pair (i, j). This is the
PER-CELL boost the single_seq_edge_mcmc sampler sees BEFORE the prior
penalty and matching constraint kick in. Plotted as an upper-triangle
heatmap with Cys positions highlighted.

No MCMC required.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

import numpy as np                                            # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402

from tkfdp.single_seq_edge_mcmc import _build_M_solo_canonical    # noqa: E402
from tkfdp.block_likelihoods import empirical_pi_c_from_checkpoint  # noqa: E402
from tkfdp.potts_dp import PottsDPState                            # noqa: E402

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA2I = {a: i for i, a in enumerate(AA)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(
        REPO / 'results' / 'K4-emwarm-top1000-2026-05-09'
        / '_best_chkpt' / 'state.npz'))
    ap.add_argument('--seq', default='FCFLEEDPGICRGYITRYFYNNQSKQCERFKYGGCLGNLNNFESLEECKNTCE',
                    help='Sequence (defaults to TFPI1_RABIT/120-172).')
    ap.add_argument('--seq-name', default='TFPI1_RABIT/120-172')
    ap.add_argument('--out', type=Path,
                    default=REPO / 'math-paper' / 'figures'
                    / 'tfpi1_M_solo_triangle.pdf')
    args = ap.parse_args()

    # Load K=4 state.
    d = np.load(args.ckpt, allow_pickle=True)

    class _S:
        pass
    state = _S()
    state.K_c = int(d['pi_class'].shape[0])
    state.A = 20
    state.pi_class = np.asarray(d['pi_class'], dtype=np.float32)
    state.potts_dp = PottsDPState(
        K_c=state.K_c, A=20,
        atoms=np.asarray(d['potts_atoms'], dtype=np.float32),
        assignments=np.asarray(d['potts_assignments'], dtype=np.int64),
        counts=np.asarray(d['potts_counts'], dtype=np.int64),
        alpha_H=1.0)
    pi_c = empirical_pi_c_from_checkpoint(args.ckpt)

    M_solo = _build_M_solo_canonical(state, pi_c, t=1.0,
                                       pair_background='lg08')
    log2_M = np.log2(np.maximum(M_solo, 1e-300))

    x = np.array([AA2I[a] for a in args.seq])
    L = len(x)

    # Build the (L, L) per-position log_2 M_solo[x_i, x_j], upper triangle only.
    tile = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            tile[i, j] = log2_M[x[i], x[j]]
    mask = np.tri(L, k=0, dtype=bool)
    tile_masked = np.ma.masked_where(mask, tile)

    cys_pos = [i + 1 for i, a in enumerate(args.seq) if a == 'C']

    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = float(np.max(np.abs(tile)))
    im = ax.imshow(tile_masked, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                    interpolation='nearest', origin='upper')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\log_2 M_{\rm solo}[x_i, x_j]$ '
                    '(canonical: LG08 pair bg, empirical $\\pi_c$)',
                    fontsize=10)

    # Mark Cys positions on both axes.
    for cp in cys_pos:
        ax.axhline(cp - 1, color='lime', linewidth=0.4, alpha=0.5)
        ax.axvline(cp - 1, color='lime', linewidth=0.4, alpha=0.5)
    # Sequence labels every 5 + at Cys positions.
    ticks = sorted(set(list(range(0, L, 5)) + [c - 1 for c in cys_pos]))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'{args.seq[i]}{i+1}' for i in ticks],
                         fontsize=7, rotation=90)
    ax.set_yticklabels([f'{args.seq[i]}{i+1}' for i in ticks],
                         fontsize=7)

    # Annotate each Cys-Cys cell.
    for i in cys_pos:
        for j in cys_pos:
            if j > i:
                ax.text(j - 1, i - 1, f'{tile[i-1, j-1]:+.1f}',
                          ha='center', va='center', fontsize=7,
                          color='white' if tile[i-1, j-1] > vmax * 0.5 else 'black')

    ax.set_title(
        f'Per-pair coupling boost $\\log_2 M_{{\\rm solo}}[x_i, x_j]$ '
        f'for {args.seq_name} (L={L})\n'
        f'K=4 emwarm canonical convention. '
        f'Cys positions (lime lines): '
        f'{", ".join(f"C{c}" for c in cys_pos)}. '
        f'All 15 Cys-Cys pairs have $\\log_2 M_{{\\rm solo}} = +{log2_M[1, 1]:.2f}$.',
        fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.out, bbox_inches='tight')
    fig.savefig(args.out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {args.out} + .png')
    print(f'  Top non-Cys-Cys cells:')
    nonc = []
    for i in range(L):
        for j in range(i + 1, L):
            if AA[x[i]] == 'C' and AA[x[j]] == 'C':
                continue
            nonc.append((tile[i, j], i + 1, j + 1, AA[x[i]], AA[x[j]]))
    nonc.sort(reverse=True)
    for v, i, j, a, b in nonc[:5]:
        print(f'    ({i:2d}, {j:2d}) {a}-{b}: {v:+.3f}')


if __name__ == '__main__':
    main()
