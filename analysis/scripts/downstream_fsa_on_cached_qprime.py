#!/usr/bin/env python3
"""Downstream FSA + opt-acc + MSA scoring on cached Q' arrays.

For a method whose per-pair Q' arrays are already on disk (in
``~/.cache/tkf-mixdom-balibase/<method>/<family>.npz``), produces:

  - per-pair opt-acc e_tp / total_mass (via Holmes-Durbin DP on Q')
  - per-pair MSA F1 / SP / TC at gap_factor in {0, 1} (via FSA reconstruction)
  - corpus aggregate of each metric
  - one JSON output file with all of the above

Suitable for filling in the FSA / opt-acc columns of the headline
table for methods whose soft posteriors live in the cache (TKF-DP K=4
RE in particular).

Usage:
    python downstream_fsa_on_cached_qprime.py \\
        --method infinite_phmm_mcmc_K4_coupled_RE \\
        --params-key f1ecfc23e19e84fe \\
        --out math-paper/results/k4_re_downstream_fsa.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

TKFMIXDOM = Path.home() / "tkf-mixdom" / "python"
sys.path.insert(0, str(TKFMIXDOM))
sys.path.insert(0, str(TKFMIXDOM / "experiments"))
sys.path.insert(0, str(TKFMIXDOM / "tkfmixdom" / "util"))

from tkfmixdom.util import balibase_pair_cache as ppcache
from tkfmixdom.util.msa_benchmark import parse_fasta, sp_tc_score
from expected_pair_f1 import expected_pair_f1, ref_to_pair_truth
from expected_pairwise_balibase import (
    _optimal_accuracy_indicator, _build_msa_with_gap_factor)


def parse_ref(path: str | Path) -> dict[str, str]:
    """Parse a BAliBASE .ref/.fasta alignment (gapped sequences). The ref
    files are plain FASTA with gap characters; reuse parse_fasta."""
    return parse_fasta(str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True,
                   help='Cache method name '
                        '(e.g. infinite_phmm_mcmc_K4_coupled_RE)')
    p.add_argument('--params-key', required=True,
                   help='Cache params_key (16-hex) for the cached Q\' arrays')
    p.add_argument('--balibase-dir', type=str,
                   default=str(Path.home() / 'bio-datasets'
                                  / 'BAliBASE3' / 'bali3pdbm'))
    p.add_argument('--out', required=True,
                   help='Output JSON path for the per-family + corpus stats')
    p.add_argument('--families', type=str, default=None,
                   help='Comma-separated subset of family names; default: '
                        'all cached families.')
    p.add_argument('--gap-factor-1', action='store_true', default=True,
                   help='Compute gap_factor=1 FSA reconstruction (default on)')
    p.add_argument('--gap-factor-0', action='store_true', default=True,
                   help='Compute gap_factor=0 FSA reconstruction (default on)')
    p.add_argument('--opt-acc', action='store_true', default=True,
                   help='Compute Holmes-Durbin optimal-accuracy DP (default on)')
    p.add_argument('--fsa-n-seeds', type=int, default=5,
                   help='Number of distinct random seeds to try for the FSA '
                        'sequence-annealing restarts. The reported '
                        'msa_sp/msa_tc are the MEAN over seeds; full '
                        'per-seed values are stored in '
                        '*_per_seed for reproducibility. Default: 5.')
    p.add_argument('--fsa-base-seed', type=int, default=42,
                   help='Base seed; subsequent seeds are base + k*997 for '
                        'k in [0, fsa_n_seeds).')
    p.add_argument('--fsa-anneal-iters', type=int, default=3,
                   help='Inner annealing iterations per FSA call.')
    args = p.parse_args()

    cache_root = Path.home() / '.cache' / 'tkf-mixdom-balibase' / args.method
    if not cache_root.is_dir():
        print(f"ERROR: cache directory not found: {cache_root}",
               file=sys.stderr)
        return 1

    # Discover cached families
    available = sorted(p.stem for p in cache_root.glob('*.npz'))
    if args.families:
        wanted = set(s.strip() for s in args.families.split(',') if s.strip())
        available = [f for f in available if f in wanted]
    print(f"Processing {len(available)} cached families from {cache_root}",
           flush=True)

    in_dir = Path(args.balibase_dir) / 'in'
    ref_dir = Path(args.balibase_dir) / 'ref'

    per_family = []
    corpus = {
        'opt_acc_e_tp': 0.0, 'opt_acc_total_mass': 0.0,
        'gap1_e_tp': 0.0, 'gap1_total_mass': 0.0,
        'gap1_msa_sp_sum': 0.0, 'gap1_msa_tc_sum': 0.0,
        'gap0_e_tp': 0.0, 'gap0_total_mass': 0.0,
        'gap0_msa_sp_sum': 0.0, 'gap0_msa_tc_sum': 0.0,
        'gold_total': 0,
        'msa_score_count': 0,
    }
    t_start = time.time()

    for fi, fam in enumerate(available):
        loaded = ppcache.load(args.method, fam, args.params_key)
        if loaded is None:
            print(f"  [{fi+1}/{len(available)}] {fam}: cache key mismatch; "
                  "skipping", flush=True)
            continue
        pair_posteriors, kind, _failed = loaded

        fa_path = in_dir / fam
        ref_path = ref_dir / fam
        if not fa_path.exists() or not ref_path.exists():
            print(f"  [{fi+1}/{len(available)}] {fam}: input missing; skip",
                   flush=True)
            continue
        fasta = parse_fasta(str(fa_path))
        # parse_fasta returns dict-like; need name->seq.
        if isinstance(fasta, dict):
            names = list(fasta.keys())
            seqs = [fasta[n] for n in names]
        else:
            # Support tuple-list form (name, seq, ...)
            names = [t[0] for t in fasta]
            seqs = [t[1] if isinstance(t[1], str) else None for t in fasta]
        ref_aln = parse_ref(ref_path)
        # int sequences for FSA: encode as 0..19 (with -1 for unknown).
        # _build_msa_with_gap_factor expects a name->seq dict.
        AA = "ACDEFGHIKLMNPQRSTVWY"
        AA_INDEX = {c: i for i, c in enumerate(AA)}
        int_seqs_dict = {n: np.array([AA_INDEX.get(c.upper(), 19) for c in s],
                                       dtype=np.int32)
                           for n, s in zip(names, seqs)}

        fam_rec = {'family': fam, 'per_pair': []}
        t_fam = time.time()
        # Helper: run FSA over n_seeds, return per-seed metrics + cell-pool F1.
        def _msa_to_pair_indicators(msa_dict):
            """Extract per-pair cell-indicators (0/1) from an MSA: a pair
            (i, j) is 'aligned' iff both sequences have an ungapped residue
            in the same MSA column. Returns dict[(i,j)] -> (Lx, Ly) ndarray."""
            order = list(msa_dict.keys())
            seqs_aligned = [msa_dict[n] for n in order]
            # Map each (msa_idx, ungapped_pos) per sequence
            ungapped_at = []
            for s in seqs_aligned:
                m = []
                pos = 0
                for c in s:
                    if c not in ('.', '-'):
                        m.append((pos, len(m) + 0))  # column index implicit
                        pos += 1
                ungapped_at.append(s)
            indicators = {}
            n = len(order)
            for i in range(n):
                for j in range(i + 1, n):
                    # Build (Li, Lj) zero matrix; set 1 where the same MSA
                    # column has an ungapped residue from both.
                    si = seqs_aligned[i]
                    sj = seqs_aligned[j]
                    Li = sum(1 for c in si if c not in '-.')
                    Lj = sum(1 for c in sj if c not in '-.')
                    ind = np.zeros((Li, Lj), dtype=np.float32)
                    pi = pj = 0
                    pos_i_at_col = {}
                    p = 0
                    for col, c in enumerate(si):
                        if c not in '-.':
                            pos_i_at_col[col] = p
                            p += 1
                    p = 0
                    for col, c in enumerate(sj):
                        if c not in '-.':
                            if col in pos_i_at_col:
                                ind[pos_i_at_col[col], p] = 1.0
                            p += 1
                    indicators[(i, j)] = ind
            return indicators

        def _multi_seed_fsa(gap_factor):
            sps, tcs, msa_f1s = [], [], []
            for k in range(args.fsa_n_seeds):
                seed = args.fsa_base_seed + k * 997
                _, msa = _build_msa_with_gap_factor(
                    int_seqs_dict, pair_posteriors,
                    gap_factor=gap_factor,
                    n_anneal=args.fsa_anneal_iters, seed=seed)
                sp, tc = sp_tc_score(msa, ref_aln)
                sps.append(float(sp)); tcs.append(float(tc))
                # MSA cell-indicator pool F1
                indicators = _msa_to_pair_indicators(msa)
                e_tp_pool = total_mass_pool = gold_pool = 0
                for (ii, jj), ind in indicators.items():
                    name_x = names[ii]; name_y = names[jj]
                    if name_x not in ref_aln or name_y not in ref_aln:
                        continue
                    truth = ref_to_pair_truth(
                        ref_aln, name_x, name_y, core_only=True)
                    row = expected_pair_f1(ind, truth)
                    e_tp_pool += row['e_tp']
                    total_mass_pool += row['total_mass']
                    gold_pool += row['gold']
                if (total_mass_pool + gold_pool) > 0:
                    msa_f1s.append(2 * e_tp_pool / (total_mass_pool + gold_pool))
                else:
                    msa_f1s.append(0.0)
            return {
                'sp_per_seed': sps,
                'tc_per_seed': tcs,
                'msa_f1_per_seed': msa_f1s,
                'sp_mean': float(np.mean(sps)),
                'sp_std': float(np.std(sps)),
                'tc_mean': float(np.mean(tcs)),
                'tc_std': float(np.std(tcs)),
                'msa_f1_mean': float(np.mean(msa_f1s)),
                'msa_f1_std': float(np.std(msa_f1s)),
                'sp_max': float(np.max(sps)),
                'tc_max': float(np.max(tcs)),
                'n_seeds': args.fsa_n_seeds,
            }

        if args.gap_factor_1:
            try:
                r1 = _multi_seed_fsa(1.0)
                fam_rec['msa_sp_g1'] = r1['sp_mean']
                fam_rec['msa_tc_g1'] = r1['tc_mean']
                fam_rec['msa_f1_g1'] = r1['msa_f1_mean']
                fam_rec['msa_sp_g1_std'] = r1['sp_std']
                fam_rec['msa_tc_g1_std'] = r1['tc_std']
                fam_rec['msa_f1_g1_std'] = r1['msa_f1_std']
                fam_rec['msa_sp_g1_per_seed'] = r1['sp_per_seed']
                fam_rec['msa_tc_g1_per_seed'] = r1['tc_per_seed']
                fam_rec['msa_f1_g1_per_seed'] = r1['msa_f1_per_seed']
                corpus['gap1_msa_sp_sum'] += r1['sp_mean']
                corpus['gap1_msa_tc_sum'] += r1['tc_mean']
                corpus['gap1_msa_f1_sum'] = (
                    corpus.get('gap1_msa_f1_sum', 0.0) + r1['msa_f1_mean'])
                corpus['msa_score_count'] += 1
            except Exception as e:
                fam_rec['msa_g1_err'] = f"{type(e).__name__}: {e}"
        if args.gap_factor_0:
            try:
                r0 = _multi_seed_fsa(0.0)
                fam_rec['msa_sp_g0'] = r0['sp_mean']
                fam_rec['msa_tc_g0'] = r0['tc_mean']
                fam_rec['msa_f1_g0'] = r0['msa_f1_mean']
                fam_rec['msa_sp_g0_std'] = r0['sp_std']
                fam_rec['msa_tc_g0_std'] = r0['tc_std']
                fam_rec['msa_f1_g0_std'] = r0['msa_f1_std']
                fam_rec['msa_sp_g0_per_seed'] = r0['sp_per_seed']
                fam_rec['msa_tc_g0_per_seed'] = r0['tc_per_seed']
                fam_rec['msa_f1_g0_per_seed'] = r0['msa_f1_per_seed']
                corpus['gap0_msa_sp_sum'] += r0['sp_mean']
                corpus['gap0_msa_tc_sum'] += r0['tc_mean']
                corpus['gap0_msa_f1_sum'] = (
                    corpus.get('gap0_msa_f1_sum', 0.0) + r0['msa_f1_mean'])
            except Exception as e:
                fam_rec['msa_g0_err'] = f"{type(e).__name__}: {e}"
        for (i, j), Q_p in pair_posteriors.items():
            name_x = names[i]; name_y = names[j]
            if name_x not in ref_aln or name_y not in ref_aln:
                continue
            truth = ref_to_pair_truth(ref_aln, name_x, name_y, core_only=True)
            row_pair = {
                'pair': (i, j),
                'name_i': name_x, 'name_j': name_y,
                'len_i': len(seqs[i]), 'len_j': len(seqs[j]),
            }
            if args.opt_acc:
                ind = _optimal_accuracy_indicator(np.asarray(Q_p))
                opt_row = expected_pair_f1(ind, truth)
                row_pair['opt_acc_e_tp'] = float(opt_row['e_tp'])
                row_pair['opt_acc_total_mass'] = float(opt_row['total_mass'])
                row_pair['gold'] = int(opt_row['gold'])
                corpus['opt_acc_e_tp'] += opt_row['e_tp']
                corpus['opt_acc_total_mass'] += opt_row['total_mass']
                corpus['gold_total'] += opt_row['gold']
            fam_rec['per_pair'].append(row_pair)
        fam_rec['time_s'] = time.time() - t_fam
        per_family.append(fam_rec)
        gap1 = fam_rec.get('msa_sp_g1', '-')
        gap0 = fam_rec.get('msa_sp_g0', '-')
        gap1_str = f'{gap1:.3f}' if isinstance(gap1, float) else gap1
        gap0_str = f'{gap0:.3f}' if isinstance(gap0, float) else gap0
        print(f"  [{fi+1}/{len(available)}] {fam}: "
              f"opt_acc_eTP={corpus['opt_acc_e_tp']:.1f}, "
              f"SP[g=1]={gap1_str}, SP[g=0]={gap0_str}, "
              f"t={fam_rec['time_s']:.1f}s", flush=True)

    elapsed = time.time() - t_start
    output = {
        'method': args.method,
        'params_key': args.params_key,
        'n_families': len(per_family),
        'elapsed_seconds': elapsed,
        'per_family': per_family,
        'corpus': corpus,
        'corpus_aggregates': {
            'opt_acc_F1': (
                2 * corpus['opt_acc_e_tp']
                / max(1.0, corpus['opt_acc_total_mass'] + corpus['gold_total'])
            ),
            'gap1_msa_SP_mean': (
                corpus['gap1_msa_sp_sum']
                / max(1, corpus['msa_score_count'])),
            'gap1_msa_TC_mean': (
                corpus['gap1_msa_tc_sum']
                / max(1, corpus['msa_score_count'])),
            'gap0_msa_SP_mean': (
                corpus['gap0_msa_sp_sum']
                / max(1, corpus['msa_score_count'])),
            'gap0_msa_TC_mean': (
                corpus['gap0_msa_tc_sum']
                / max(1, corpus['msa_score_count'])),
        },
    }
    Path(args.out).write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote {args.out} (corpus opt-acc F1 = "
          f"{output['corpus_aggregates']['opt_acc_F1']:.4f}, "
          f"SP[g=1]={output['corpus_aggregates']['gap1_msa_SP_mean']:.4f})",
           flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
