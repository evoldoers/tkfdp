#!/usr/bin/env python3
"""Cell-by-cell comparison of the top-rung MCMC Q' against the cached
TKF92 FB posteriors.

At alpha_z = 1e6 the bounded-eps prior pins |E|=0, so the sampler
reduces to a pure-Gibbs alignment chain under the TKF92 indel model.
Run with the corpus-fitted indel rates (ins=0.04581, del=0.04680,
ext=0.6835) it should reproduce -- to within MCMC sampling noise --
the per-pair forward-backward posteriors that the BAliBase
tkf92_lg08 method cached at the same rates.

The top-rung JSON does NOT serialise Q'. We have, per pair, the
soft-F1 sufficient stats (e_tp, total_mass, gold, n_cells), which
let us compare aggregate F1 and SP. For a cell-by-cell diff we'd
need to plumb Q' itself out of the sampler -- not currently done.
This script reports the aggregate-level reduction-test:

  TKF92 cached (from expected_balibase_tkf92_l150.json corpus_post):
    F1, SP, e_tp, total_mass, gold

  Top-rung MCMC (from infinite_phmm_balibase_k4_top_rung_validation.json):
    F1, SP, e_tp, total_mass, gold (pooled across all per_pair rows)

  Differences should be below MCMC sampling noise (~ 1 / sqrt(ESS)
  per pair, summed in quadrature across pairs).
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TKF92_JSON = (Path.home() / "tkf-mixdom" / "python" / "experiments"
              / "expected_balibase" / "expected_balibase_tkf92_l150.json")
TOPRUNG_JSON = (REPO / "math-paper" / "results"
                / "infinite_phmm_balibase_k4_top_rung_validation.json")


def f1_of(e_tp, total_mass, gold):
    den = total_mass + gold
    return None if den == 0 else 2.0 * e_tp / den


def sp_of(e_tp, gold):
    return None if gold == 0 else e_tp / gold


def pool_tkf92_post():
    """Pool corpus_post stats from the existing TKF92 L<150 JSON.
    These are the posterior-soft (E_pairs) sufficient stats at the
    fitted indel rates."""
    d = json.loads(TKF92_JSON.read_text())
    pairs = [p for fam in d['per_family']
              for p in (fam.get('per_pair_post') or [])]
    if not pairs:
        return None
    e_tp = sum(p['e_tp'] for p in pairs)
    mass = sum(p['total_mass'] for p in pairs)
    gold = sum(p['gold'] for p in pairs)
    cells = sum(p.get('n_cells', 0) for p in pairs)
    return {'e_tp': e_tp, 'total_mass': mass, 'gold': gold,
            'n_cells': cells, 'n_pairs': len(pairs)}


def pool_toprung():
    """Pool per-pair stats from the top-rung MCMC sweep JSON."""
    d = json.loads(TOPRUNG_JSON.read_text())
    fams = d.get('per_family', d) if isinstance(d, dict) else d
    pairs = []
    for fam in fams:
        if not isinstance(fam, dict):
            continue
        for p in (fam.get('per_pair') or []):
            pairs.append(p)
    if not pairs:
        return None
    e_tp = sum(p['e_tp'] for p in pairs)
    mass = sum(p['total_mass'] for p in pairs)
    gold = sum(p['gold'] for p in pairs)
    cells = sum(p.get('n_cells', 0) for p in pairs)
    return {'e_tp': e_tp, 'total_mass': mass, 'gold': gold,
            'n_cells': cells, 'n_pairs': len(pairs)}


def main():
    tkf = pool_tkf92_post()
    top = pool_toprung()
    if tkf is None or top is None:
        print("missing data; aborting", file=sys.stderr)
        sys.exit(1)
    print(f"\n{'metric':<14} {'TKF92 (cached)':>16} {'top-rung MCMC':>16} "
          f"{'absolute diff':>14} {'rel diff':>10}")
    print('-' * 76)
    for name, get in (
        ('n_pairs', lambda d: d['n_pairs']),
        ('e_tp', lambda d: d['e_tp']),
        ('total_mass', lambda d: d['total_mass']),
        ('gold', lambda d: d['gold']),
        ('n_cells', lambda d: d['n_cells']),
        ('F1', lambda d: f1_of(d['e_tp'], d['total_mass'], d['gold'])),
        ('SP', lambda d: sp_of(d['e_tp'], d['gold'])),
    ):
        a, b = get(tkf), get(top)
        if a is None or b is None:
            print(f"{name:<14} {a!s:>16} {b!s:>16}")
            continue
        diff = b - a
        rel = (diff / a) if a not in (0, 0.0) else float('nan')
        if name in ('F1', 'SP'):
            print(f"{name:<14} {a:>16.4f} {b:>16.4f} {diff:>+14.4f} "
                  f"{rel:>+10.2%}")
        elif isinstance(a, int):
            print(f"{name:<14} {a:>16d} {b:>16d} {diff:>+14d} "
                  f"{rel:>+10.2%}")
        else:
            print(f"{name:<14} {a:>16.2f} {b:>16.2f} {diff:>+14.2f} "
                  f"{rel:>+10.2%}")
    print(f"\n  PASS  if F1 and SP differ by <0.02 (~MCMC noise on n_pairs={top['n_pairs']})")


if __name__ == '__main__':
    main()
