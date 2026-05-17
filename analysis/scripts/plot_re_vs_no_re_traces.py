#!/usr/bin/env python3
"""Plot RE vs no-RE convergence + cache-targeting visual.

Reads the two JSONs produced by aligned invocations of
sweep_infinite_phmm_balibase.py on the same pair:
  /tmp/re_demo_A_no_re.json    -- 4 chains, no RE, alpha_z=100
  /tmp/re_demo_B_with_re.json  -- 4 replicates of RE with the
                                  validated ladder

Produces a 3-row x 2-col figure:
  row 1: log_pi traces            (left=no-RE, right=with-RE)
  row 2: n_match traces           (left=no-RE, right=with-RE)
  row 3: mu_cache_size traces     (left=no-RE, right=with-RE)
         + horizontal reference at prepop_n_anchors (the L^{3/2}
         target). The gap between the trace and the reference is
         the chain-driven cache growth -- a visual indicator of how
         well-targeted the prepop set was.

Usage:
    python analysis/scripts/plot_re_vs_no_re_traces.py \
        /tmp/re_demo_A_no_re.json /tmp/re_demo_B_with_re.json \
        --out math-paper/figures/re_vs_no_re_traces.pdf
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def extract_no_re(j):
    """Return list of (label, log_pi, n_match, cache_size_trace) per chain."""
    fam = j['per_family'][0]
    pair = fam['per_pair'][0]
    diag = pair['mcmc_diag']
    out = []
    for k, pc in enumerate(diag.get('per_chain', [])):
        out.append({
            'label': f'chain {k}',
            'log_pi': pc.get('log_pi_trace') or [],
            'n_match': pc.get('n_match_trace') or [],
            'cache': pc.get('mu_cache_size_trace') or [],
        })
    return out, pair, diag


def extract_with_re(j):
    """For RE: per_chain is per-rung of replicate 0. The cold rung
    (idx 0) is what we want. For per-rep traces we use
    re_replicate_traces (only n_match across reps; log_pi /
    cache are bundled rep-0 only)."""
    fam = j['per_family'][0]
    pair = fam['per_pair'][0]
    diag = pair['mcmc_diag']
    cold = diag['per_chain'][0]
    # Per-rep cold-rung n_match if available; else fall back to cold-rung
    # single trace.
    re_replicate_traces = diag.get('rhat_n_match_re_replicates')  # may not be the right key
    return [{
        'label': 'cold rung',
        'log_pi': cold.get('log_pi_trace') or [],
        'n_match': cold.get('n_match_trace') or [],
        'cache': cold.get('mu_cache_size_trace') or [],
    }], pair, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json_no_re', help='/tmp/re_demo_A_no_re.json')
    ap.add_argument('json_with_re', help='/tmp/re_demo_B_with_re.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    da = json.loads(Path(args.json_no_re).read_text())
    db = json.loads(Path(args.json_with_re).read_text())

    chains_a, pair_a, diag_a = extract_no_re(da)
    chains_b, pair_b, diag_b = extract_with_re(db)

    Lx = pair_a['len_i']
    Ly = pair_a['len_j']
    # Estimate prepop_n_anchors: ceil(max(Lx, Ly)^{3/2}).
    Leff = max(Lx, Ly)
    prepop = int(np.ceil(Leff * np.sqrt(Leff)))

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex='col')
    title = (f"RE vs no-RE on {pair_a.get('name_i')}-{pair_a.get('name_j')} "
              f"(L={Lx}x{Ly}, {pair_a['n_cells']} cells; prepop_n_anchors "
              f"~={prepop})")
    fig.suptitle(title, fontsize=11)

    cols_no_re = plt.cm.tab10(np.linspace(0, 1, max(4, len(chains_a))))
    cols_with_re = plt.cm.viridis(np.linspace(0.15, 0.85, max(1, len(chains_b))))

    for (col, chains, title_str) in [
            (0, chains_a, 'no RE (single chain, alpha_z=100)'),
            (1, chains_b, 'RE ladder [100,178,316,562,1000,5000]')]:
        for k, ch in enumerate(chains):
            color = cols_no_re[k] if col == 0 else cols_with_re[k]
            x = np.arange(len(ch['log_pi']))
            if len(ch['log_pi']) > 0:
                axes[0, col].plot(x, ch['log_pi'], color=color,
                                    alpha=0.85, label=ch['label'])
            if len(ch['n_match']) > 0:
                axes[1, col].plot(x, ch['n_match'], color=color,
                                    alpha=0.85, label=ch['label'])
            if len(ch['cache']) > 0:
                axes[2, col].plot(x, ch['cache'], color=color,
                                    alpha=0.85, label=ch['label'])
        axes[0, col].set_title(title_str, fontsize=10)
        axes[0, col].set_ylabel('log $\\pi$' if col == 0 else '')
        axes[1, col].set_ylabel('# Match cells' if col == 0 else '')
        axes[2, col].set_ylabel('|mu_cache|' if col == 0 else '')
        axes[2, col].axhline(prepop, color='red', ls='--', lw=1,
                             label=f'prepop = {prepop}')
        axes[2, col].set_xlabel('post-burnin sweep')
        axes[0, col].legend(fontsize=7, loc='best', frameon=False)
        axes[2, col].legend(fontsize=7, loc='best', frameon=False)

    # Sync y-limits per row for fair visual comparison.
    for row in range(3):
        ymin = min(axes[row, 0].get_ylim()[0], axes[row, 1].get_ylim()[0])
        ymax = max(axes[row, 0].get_ylim()[1], axes[row, 1].get_ylim()[1])
        for col in range(2):
            axes[row, col].set_ylim(ymin, ymax)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    fig.savefig(args.out.with_suffix('.png'), dpi=120, bbox_inches='tight')
    print(f'Wrote {args.out} + {args.out.with_suffix(".png")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
