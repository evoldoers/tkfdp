#!/usr/bin/env python3
"""Filter the existing BAliBase TFPN result JSONs down to the L<150
subset (families whose longest sequence is < 150 residues) and report
the corpus stats for direct comparison with the infinite-PHMM-K=4
sampler row (which is restricted to that same subset).

Inputs:
  ~/tkf-mixdom/python/experiments/expected_balibase/expected_balibase_*.json
  ~/tkf-dp/math-paper/results/infinite_phmm_balibase_k4.json (when ready)

Output: a small JSON summary at
  ~/tkf-dp/math-paper/results/balibase_l150_summary.json
plus a stdout table.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


EXPECTED_BALIBASE_DIR = (
    Path('~/tkf-mixdom/python/experiments/expected_balibase').expanduser())
INFINITE_PHMM_K4 = (
    Path('~/tkf-dp/math-paper/results/infinite_phmm_balibase_k4.json')
    .expanduser())
INFINITE_PHMM_K1 = (
    Path('~/tkf-dp/math-paper/results/infinite_phmm_balibase.json')
    .expanduser())
BALI_IN = Path('~/bio-datasets/data/balibase/bali3pdbm/in').expanduser()


def family_max_len(family: str) -> int:
    """Cheap scan of a BAliBase FASTA for the longest ungapped seq."""
    path = BALI_IN / family
    if not path.exists():
        return 0
    n = max_l = cur = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith('>'):
                if cur > max_l:
                    max_l = cur
                cur = 0
                n += 1
            else:
                cur += sum(1 for c in line.strip() if c.isalpha())
        if cur > max_l:
            max_l = cur
    return max_l


def subset_corpus(method_json: dict, max_len: int) -> dict:
    """Pool e_tp / total_mass / gold / n_cells over the per_pair_*
    lists of families with max-seq-len < max_len. Returns {branch: micro}
    for branches actually present in the JSON."""
    keep = [f for f in method_json['per_family']
             if family_max_len(f['family']) < max_len
             and not f.get('_failed_family')]
    out = {}
    for branch in ('post', 'hard', 'opt', 'fsa_sps'):
        pairs = []
        for f in keep:
            pp = f.get(f'per_pair_{branch}')
            if pp is None:
                continue
            pairs.extend(pp)
        if not pairs:
            continue
        e_tp = sum(r['e_tp'] for r in pairs)
        mass = sum(r['total_mass'] for r in pairs)
        gold = sum(r['gold'] for r in pairs)
        cells = sum(r.get('n_cells', 0) for r in pairs)
        out[branch] = {
            'e_tp': float(e_tp),
            'total_mass': float(mass),
            'gold': int(gold),
            'n_cells': int(cells),
            'n_pairs': int(len(pairs)),
        }
    out['n_families'] = int(len(keep))
    return out


def subset_corpus_infinite_phmm(path: Path, max_len: int) -> dict | None:
    """The infinite-PHMM JSON has a different shape: a list of per-family
    dicts each with a 'per_pair' list of pair rows."""
    if not path.exists():
        return None
    fams = json.loads(path.read_text())
    keep = [f for f in fams
             if family_max_len(f['family']) < max_len
             and 'per_pair' in f]
    pairs = []
    for f in keep:
        pairs.extend(f['per_pair'])
    if not pairs:
        return None
    e_tp = sum(r['e_tp'] for r in pairs)
    mass = sum(r['total_mass'] for r in pairs)
    gold = sum(r['gold'] for r in pairs)
    cells = sum(r.get('n_cells', 0) for r in pairs)
    return {'hard': {  # the sampler emits hard 0/1 posteriors
        'e_tp': float(e_tp),
        'total_mass': float(mass),
        'gold': int(gold),
        'n_cells': int(cells),
        'n_pairs': int(len(pairs)),
        'n_families': int(len(keep)),
    }}


def f1_from(m):
    den = m['total_mass'] + m['gold']
    return None if den == 0 else 2 * m['e_tp'] / den


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--max-len', type=int, default=150)
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    summary = {}
    print(f"L<{args.max_len} corpus filter (BAliBase bali3pdbm)\n")
    print(f"{'method':<22} {'fams':>5} {'pairs':>6} "
          f"{'F1_post':>7} {'F1_hard':>7} {'F1_opt':>7} {'F1_sps':>7} "
          f"{'SP_post':>7} {'SP_hard':>7}")
    print('-' * 96)
    for path in sorted(EXPECTED_BALIBASE_DIR.glob('expected_balibase_*.json')):
        d = json.loads(path.read_text())
        name = d['method_name']
        sub = subset_corpus(d, args.max_len)
        summary[name] = sub
        n_fams = sub.pop('n_families', 0)
        n_pairs = next((b['n_pairs'] for b in sub.values()), 0)
        f1 = {b: f1_from(sub[b]) if b in sub else None
               for b in ('post', 'hard', 'opt', 'fsa_sps')}
        sp_p = sub['post']['e_tp'] / sub['post']['gold'] if 'post' in sub else None
        sp_h = sub['hard']['e_tp'] / sub['hard']['gold'] if 'hard' in sub else None
        def fmt(x): return f'{x:.3f}' if x is not None else '   -  '
        print(f"{name:<22} {n_fams:>5} {n_pairs:>6} "
              f"{fmt(f1['post']):>7} {fmt(f1['hard']):>7} "
              f"{fmt(f1['opt']):>7} {fmt(f1['fsa_sps']):>7} "
              f"{fmt(sp_p):>7} {fmt(sp_h):>7}")

    for label, path in (('infinite_phmm_K1_stub', INFINITE_PHMM_K1),
                         ('infinite_phmm_K4', INFINITE_PHMM_K4)):
        sub = subset_corpus_infinite_phmm(path, args.max_len)
        if sub is None:
            print(f"{label:<22} (file missing; skipping)")
            continue
        summary[label] = sub
        n_fams = sub['hard']['n_families']
        n_pairs = sub['hard']['n_pairs']
        f1 = f1_from(sub['hard'])
        sp = sub['hard']['e_tp'] / sub['hard']['gold']
        def fmt(x): return f'{x:.3f}' if x is not None else '   -  '
        print(f"{label:<22} {n_fams:>5} {n_pairs:>6} "
              f"{'   -  ':>7} {fmt(f1):>7} "
              f"{'   -  ':>7} {'   -  ':>7} "
              f"{'   -  ':>7} {fmt(sp):>7}")

    out = Path(args.out) if args.out else (
        Path('~/tkf-dp/math-paper/results/balibase_l150_summary.json')
        .expanduser())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
