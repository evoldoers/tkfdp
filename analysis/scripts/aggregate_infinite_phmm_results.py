"""Aggregate per-pair sufficient stats from
math-paper/results/infinite_phmm_balibase.json into corpus-level
expected SP and TC scores for Table 1 in main.tex.

This script reads the saved JSON: each per-pair row has e_tp,
total_mass, gold, n_cells from the running posterior mean Q' of the
MCMC sweep.  Pool these by simple summation -- the soft confusion
matrix is additive across pairs and families.

If the sweep also wrote opt_acc_e_tp / opt_acc_total_mass
(Holmes-Durbin optimal-accuracy indicator -- monotone-increasing
matching of max expected TP through Q'), we also produce a
"variant" corpus aggregate.

Usage:
    python aggregate_infinite_phmm_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS_JSON = REPO / "math-paper" / "results" / "infinite_phmm_balibase.json"


def _f1_from_sufficient_stats(e_tp, total_mass, gold):
    if total_mass <= 0 or gold <= 0:
        return float('nan'), float('nan'), float('nan')
    prec = e_tp / total_mass
    recall = e_tp / gold
    f1 = 2 * e_tp / (total_mass + gold)
    return prec, recall, f1


def aggregate(rows):
    """Pool per-pair sufficient stats into corpus-level totals."""
    e_tp = 0.0
    tot_mass = 0.0
    opt_e_tp = 0.0
    opt_tot_mass = 0.0
    have_opt = False
    gold = 0
    n_cells = 0
    n_pairs = 0
    by_family = {}
    for r in rows:
        if 'error' in r:
            continue
        fam = r['family']
        for pair in r.get('per_pair', []):
            e_tp += pair['e_tp']
            tot_mass += pair['total_mass']
            gold += pair['gold']
            n_cells += pair['n_cells']
            n_pairs += 1
            if 'opt_acc_e_tp' in pair:
                opt_e_tp += pair['opt_acc_e_tp']
                opt_tot_mass += pair['opt_acc_total_mass']
                have_opt = True
            by_family.setdefault(fam, []).append(pair)
    prec, recall, f1 = _f1_from_sufficient_stats(e_tp, tot_mass, gold)
    out = {
        'n_pairs': n_pairs,
        'n_families': len(by_family),
        'micro': {
            'e_tp': e_tp,
            'total_mass': tot_mass,
            'gold': gold,
            'n_cells': n_cells,
            'soft_precision': prec,
            'soft_recall': recall,
            'soft_F1': f1,
            'soft_SP': recall,   # SP = sensitivity / recall in BAliBase
        },
        'per_family': {fam: {
            'n_pairs': len(pairs),
            'e_tp_sum': sum(p['e_tp'] for p in pairs),
            'total_mass_sum': sum(p['total_mass'] for p in pairs),
            'gold_sum': sum(p['gold'] for p in pairs),
        } for fam, pairs in by_family.items()},
    }
    if have_opt:
        op, orec, of1 = _f1_from_sufficient_stats(opt_e_tp, opt_tot_mass, gold)
        out['micro_optimal_accuracy'] = {
            'e_tp': opt_e_tp,
            'total_mass': opt_tot_mass,
            'gold': gold,
            'soft_precision': op,
            'soft_recall': orec,
            'soft_F1': of1,
            'soft_SP': orec,
        }
    return out


def main():
    if not RESULTS_JSON.exists():
        print(f"No results JSON at {RESULTS_JSON}; the sweep has not run yet",
              file=sys.stderr)
        return 1
    rows = json.loads(RESULTS_JSON.read_text())
    out = aggregate(rows)
    out_path = RESULTS_JSON.with_suffix('.aggregate.json')
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out['micro'], indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
