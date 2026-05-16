#!/usr/bin/env python3
"""Summarise the mcmc_diag sub-dicts from an infinite-PHMM sweep JSON.

Computes corpus-level convergence summaries (ESS distribution, r-hat
distribution, q_l1 distribution, swap acceptance rates, acc rates per
move type) and writes a one-line-per-pair CSV for downstream
plotting.

Usage:
  python3 analysis/scripts/summarize_mcmc_diag.py \\
      math-paper/results/infinite_phmm_balibase_k4_top_rung_validation.json

  python3 analysis/scripts/summarize_mcmc_diag.py \\
      math-paper/results/infinite_phmm_balibase_k4_replicaexchange.json \\
      --out-csv math-paper/results/infinite_phmm_re_diag_per_pair.csv

Always prints the corpus aggregates to stdout.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


def _percentiles(arr, ps=(5, 25, 50, 75, 95)):
    if not arr:
        return {f'p{p}': None for p in ps}
    a = np.asarray(arr, dtype=np.float64)
    return {f'p{p}': float(np.percentile(a, p)) for p in ps}


def _stats(arr):
    if not arr:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'n': 0}
    a = np.asarray(arr, dtype=np.float64)
    return {
        'mean': float(a.mean()),
        'std': float(a.std()),
        'min': float(a.min()),
        'max': float(a.max()),
        'n': int(a.size),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('json_path', type=str,
                   help='Path to sweep JSON.')
    p.add_argument('--out-csv', type=str, default=None,
                   help='If given, write per-pair diag rows to this CSV.')
    args = p.parse_args()

    d = json.loads(Path(args.json_path).read_text())
    if isinstance(d, list):
        per_family = d
        mcmc_config = {}
    else:
        per_family = d.get('per_family', [])
        mcmc_config = d.get('mcmc_config', {})

    ess_n_match = []
    ess_log_pi = []
    r_hat_match = []
    r_hat_log_pi = []
    q_l1 = []
    acc_seg = []; acc_add = []; acc_remove = []
    mean_n_edges = []
    mean_n_match = []
    runtime_s = []
    swap_acc = []

    per_pair_rows = []

    for f in per_family:
        for r in f.get('per_pair', []):
            diag = r.get('mcmc_diag') or {}
            if '_diag_error' in diag:
                continue
            per_chain = diag.get('per_chain', [])
            for c in per_chain:
                if c.get('ess_n_match') is not None:
                    ess_n_match.append(c['ess_n_match'])
                if c.get('ess_log_pi') is not None:
                    ess_log_pi.append(c['ess_log_pi'])
                if c.get('mean_n_edges') is not None:
                    mean_n_edges.append(c['mean_n_edges'])
                if c.get('mean_n_match') is not None:
                    mean_n_match.append(c['mean_n_match'])
                acc_seg.append(c.get('acc_seg') or 0.0)
                acc_add.append(c.get('acc_add') or 0.0)
                acc_remove.append(c.get('acc_remove') or 0.0)
                runtime_s.append(c.get('runtime_seconds') or 0.0)
            if diag.get('r_hat_n_match') is not None:
                r_hat_match.append(diag['r_hat_n_match'])
            if diag.get('r_hat_log_pi') is not None:
                r_hat_log_pi.append(diag['r_hat_log_pi'])
            if diag.get('q_l1_vs_baseline') is not None:
                q_l1.append(diag['q_l1_vs_baseline'])
            if diag.get('swap_acc_rates') is not None:
                for s in diag['swap_acc_rates']:
                    swap_acc.append(s)

            # Per-pair CSV row.
            per_pair_rows.append({
                'family': f.get('family', '?'),
                'name_i': r.get('name_i', '?'),
                'name_j': r.get('name_j', '?'),
                'len_i': r.get('len_i', 0),
                'len_j': r.get('len_j', 0),
                'e_tp': r.get('e_tp'),
                'total_mass': r.get('total_mass'),
                'gold': r.get('gold'),
                'baseline_e_tp': r.get('baseline_e_tp'),
                'baseline_total_mass': r.get('baseline_total_mass'),
                'q_l1_vs_baseline': diag.get('q_l1_vs_baseline'),
                'r_hat_n_match': diag.get('r_hat_n_match'),
                'r_hat_log_pi': diag.get('r_hat_log_pi'),
                'between_chain_n_match_sd': diag.get('between_chain_n_match_sd'),
                'mode': r.get('mcmc_mode'),
                'n_chains': diag.get('n_chains'),
                'mcmc_time_s': r.get('mcmc_time_s'),
                'ess_n_match_chain0': (per_chain[0].get('ess_n_match')
                                         if per_chain else None),
                'ess_log_pi_chain0': (per_chain[0].get('ess_log_pi')
                                        if per_chain else None),
                'mean_n_edges_chain0': (per_chain[0].get('mean_n_edges')
                                          if per_chain else None),
                'acc_seg_chain0': (per_chain[0].get('acc_seg')
                                     if per_chain else None),
                'acc_add_chain0': (per_chain[0].get('acc_add')
                                     if per_chain else None),
                'acc_remove_chain0': (per_chain[0].get('acc_remove')
                                        if per_chain else None),
            })

    print(f'=== MCMC diagnostics summary: {args.json_path}\n')
    print(f'mcmc_config: {mcmc_config}\n')
    print(f'n_families: {len(per_family)}')
    print(f'n_pairs (with mcmc_diag): {len(per_pair_rows)}')
    print()
    print(f'ESS (n_match) per-chain stats:')
    s = _stats(ess_n_match); pc = _percentiles(ess_n_match)
    print(f'  mean={s["mean"]}, std={s["std"]}, min={s["min"]}, max={s["max"]}')
    print(f'  percentiles: p5={pc["p5"]}, p50={pc["p50"]}, p95={pc["p95"]}')
    print()
    print(f'ESS (log_pi) per-chain stats:')
    s = _stats(ess_log_pi); pc = _percentiles(ess_log_pi)
    print(f'  mean={s["mean"]}, std={s["std"]}, min={s["min"]}, max={s["max"]}')
    print(f'  percentiles: p5={pc["p5"]}, p50={pc["p50"]}, p95={pc["p95"]}')
    print()
    if r_hat_match:
        print(f'r-hat (n_match) over pairs:')
        s = _stats(r_hat_match); pc = _percentiles(r_hat_match)
        print(f'  mean={s["mean"]}, std={s["std"]}, min={s["min"]}, max={s["max"]}')
        print(f'  percentiles: p5={pc["p5"]}, p50={pc["p50"]}, p95={pc["p95"]}')
    if r_hat_log_pi:
        print(f'r-hat (log_pi) over pairs:')
        s = _stats(r_hat_log_pi); pc = _percentiles(r_hat_log_pi)
        print(f'  mean={s["mean"]}, std={s["std"]}, min={s["min"]}, max={s["max"]}')
        print(f'  percentiles: p5={pc["p5"]}, p50={pc["p50"]}, p95={pc["p95"]}')
    print()
    if q_l1:
        print(f'q_l1_vs_baseline:')
        s = _stats(q_l1); pc = _percentiles(q_l1)
        print(f'  mean={s["mean"]}, std={s["std"]}, min={s["min"]}, max={s["max"]}')
        print(f'  percentiles: p5={pc["p5"]}, p50={pc["p50"]}, p95={pc["p95"]}')
        print()
    print(f'Move-acceptance rates (across pair-chains):')
    print(f'  acc_seg:    {_stats(acc_seg)}')
    print(f'  acc_add:    {_stats(acc_add)}')
    print(f'  acc_remove: {_stats(acc_remove)}')
    print()
    print(f'mean_n_edges per pair-chain: {_stats(mean_n_edges)}')
    print(f'mean_n_match per pair-chain: {_stats(mean_n_match)}')
    if swap_acc:
        print(f'\nReplica-exchange swap acceptance rates (per rung pair):')
        print(f'  {_stats(swap_acc)}')

    # Corpus F1 vs baseline F1.
    e_tp = sum(r['e_tp'] for r in per_pair_rows if r.get('e_tp') is not None)
    mass = sum(r['total_mass'] for r in per_pair_rows if r.get('total_mass') is not None)
    gold = sum(r['gold'] for r in per_pair_rows if r.get('gold') is not None)
    f1 = 2 * e_tp / (mass + gold) if (mass + gold) > 0 else None

    b_e_tp = sum(r['baseline_e_tp'] for r in per_pair_rows if r.get('baseline_e_tp') is not None)
    b_mass = sum(r['baseline_total_mass'] for r in per_pair_rows if r.get('baseline_total_mass') is not None)
    b_f1 = 2 * b_e_tp / (b_mass + gold) if (b_mass + gold) > 0 else None

    print(f"\nCorpus Q' F1:        {f1:.4f} ({len(per_pair_rows)} pairs)")
    if b_f1 is not None:
        print(f"Corpus baseline F1:  {b_f1:.4f}")
        print(f"|Q' F1 - baseline F1| = {abs(f1 - b_f1):.4f}")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        if per_pair_rows:
            cols = list(per_pair_rows[0].keys())
            with out.open('w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(per_pair_rows)
            print(f'\nWrote per-pair CSV: {out} ({len(per_pair_rows)} rows)')


if __name__ == '__main__':
    main()
