#!/usr/bin/env python3
"""Plot the 10 Potts atoms from the released K=4 EM-warmup checkpoint.

Loads the (10, 20, 20) potts_atoms tensor from
~/tkf-dp/results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz
and emits a 4x4 grid of heatmaps (diagonal pairs first, then
off-diagonal pairs), saved at ~/tkf-dp/math-paper/figures/k4_potts_atoms.pdf.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AA_ORDER = list('ACDEFGHIKLMNPQRSTVWY')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',
                   default=str(Path('~/tkf-dp/results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz').expanduser()))
    p.add_argument('--out',
                   default=str(Path('~/tkf-dp/math-paper/figures/k4_potts_atoms.pdf').expanduser()))
    args = p.parse_args()

    d = np.load(args.ckpt, allow_pickle=True)
    atoms = np.asarray(d['potts_atoms'])         # (10, A, A)
    assign = np.asarray(d['potts_assignments'])  # (K_c, K_c)
    counts = np.asarray(d['potts_counts'])       # (10,)
    pi_cls = np.asarray(d['pi_class'])           # (K_c, A)
    K_c = pi_cls.shape[0]
    A = pi_cls.shape[1]
    assert atoms.shape == (10, A, A)

    # Map atom index -> the unordered (c, c') pair(s) that use it.
    pair_labels = {a: [] for a in range(10)}
    for c in range(K_c):
        for c2 in range(c, K_c):
            pair_labels[int(assign[c, c2])].append((c, c2))

    fig, axes = plt.subplots(4, 3, figsize=(11, 13), constrained_layout=True)
    vmax = float(np.abs(atoms).max())
    diag_atoms = sorted(range(10),
                         key=lambda a: (min((c == c2)
                                            for c, c2 in pair_labels.get(a, [(0, 1)])
                                            if pair_labels.get(a)),))
    # Order: atoms used by diagonal (c==c') pairs first, then off-diag.
    diag_atoms = [a for a in range(10)
                   if pair_labels[a] and any(c == c2 for c, c2 in pair_labels[a])]
    offdiag_atoms = [a for a in range(10) if a not in diag_atoms]
    order = diag_atoms + offdiag_atoms
    # Pad to 12 panels (4x3) if needed.
    while len(order) < 12:
        order.append(None)

    for k, atom_idx in enumerate(order):
        ax = axes.flat[k]
        if atom_idx is None or atom_idx >= 10:
            ax.axis('off')
            continue
        H = atoms[atom_idx]
        im = ax.imshow(H, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        aspect='equal')
        pairs = pair_labels[atom_idx]
        pair_str = ', '.join(f'({c},{c2})' for c, c2 in pairs)
        usage = counts[atom_idx]
        ax.set_title(f'atom {atom_idx}: class-pairs {pair_str}\nusage={usage}',
                      fontsize=9)
        ax.set_xticks(range(A))
        ax.set_yticks(range(A))
        ax.set_xticklabels(AA_ORDER, fontsize=6, rotation=90)
        ax.set_yticklabels(AA_ORDER, fontsize=6)

    cbar = fig.colorbar(im, ax=axes.flat[:len(order)],
                         orientation='horizontal', fraction=0.04, pad=0.04,
                         shrink=0.4, location='bottom')
    cbar.set_label('Potts coupling H[a, b]  (units of -log P; '
                    'red = attractive, blue = repulsive)',
                    fontsize=9)

    fig.suptitle(
        r'K=4 TKF-DP Potts atoms (\texttt{results/K4-emwarm-top1000-2026-05-09})',
        fontsize=11, y=1.02)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
