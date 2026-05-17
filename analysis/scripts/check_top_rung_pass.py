#!/usr/bin/env python3
"""Decide whether the TASK A step 5 top-rung validation passed.

Pass criterion (per Claude / 2026-05-13 brief):
  Q' corpus soft-F1 within 0.02 of the TKF92 baseline corpus_post F1
  (0.450 on the L<150 BAliBase bali3pdbm 22-family subset).

Reads:
  ~/tkf-dp/math-paper/results/infinite_phmm_balibase_k4_top_rung_validation.json

Exits 0 if passing, 1 if failing.  Prints a short report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VAL_JSON = REPO / 'math-paper' / 'results' / 'infinite_phmm_balibase_k4_top_rung_validation.json'

# TKF92 baseline corpus_post F1 on the L<150 subset
# (from expected_balibase_tkf92_l150.json, corpus_post.micro).
TKF92_BASELINE_F1 = 0.4504
TOLERANCE = 0.02


def main():
    if not VAL_JSON.exists():
        print(f'ERR: {VAL_JSON} not found.', file=sys.stderr)
        sys.exit(1)
    d = json.loads(VAL_JSON.read_text())
    if isinstance(d, list):
        per_family = d
        cfg = {}
    else:
        per_family = d.get('per_family', [])
        cfg = d.get('mcmc_config', {})

    if len(per_family) < 22:
        print(f'WARN: only {len(per_family)}/22 families completed; '
              f'top-rung run may not be finished.')

    e_tp = 0.0
    mass = 0.0
    gold = 0
    n_pairs = 0
    baseline_e_tp = 0.0
    baseline_mass = 0.0

    for f in per_family:
        for r in f.get('per_pair', []):
            e_tp += r['e_tp']
            mass += r['total_mass']
            gold += r['gold']
            n_pairs += 1
            if 'baseline_e_tp' in r:
                baseline_e_tp += r['baseline_e_tp']
            if 'baseline_total_mass' in r:
                baseline_mass += r['baseline_total_mass']

    f1 = 2 * e_tp / (mass + gold) if (mass + gold) > 0 else None
    sp = e_tp / gold if gold > 0 else None
    prec = e_tp / mass if mass > 0 else None
    baseline_f1 = (2 * baseline_e_tp / (baseline_mass + gold)
                     if (baseline_mass + gold) > 0 else None)

    print(f'=== TASK A step 5: top-rung validation result ===')
    print(f'  mcmc_config: {cfg}')
    print(f'  n_families: {len(per_family)}/22')
    print(f'  n_pairs: {n_pairs}/187')
    print(f'  Q\' corpus F1:           {f1:.4f}')
    print(f'  Q\' corpus soft-recall:  {sp:.4f}')
    print(f'  Q\' corpus soft-prec:    {prec:.4f}')
    if baseline_f1 is not None:
        print(f'  Per-pair-baseline F1:   {baseline_f1:.4f}  (Q_baseline from FB)')
        print(f'  |Q\' F1 - baseline F1| =  {abs(f1 - baseline_f1):.4f}')
    print()
    print(f'  Target TKF92 baseline F1 (L<150, lg08, corpus_post): {TKF92_BASELINE_F1:.4f}')
    delta = abs(f1 - TKF92_BASELINE_F1) if f1 is not None else None
    print(f'  |Q\' F1 - target F1|     = {delta:.4f}')
    print(f'  Tolerance:                {TOLERANCE}')
    if delta is not None and delta < TOLERANCE:
        print()
        print(f'  *** PASS *** ({delta:.4f} < {TOLERANCE})')
        sys.exit(0)
    else:
        print()
        print(f'  *** FAIL *** ({delta:.4f} >= {TOLERANCE})')
        sys.exit(1)


if __name__ == '__main__':
    main()
