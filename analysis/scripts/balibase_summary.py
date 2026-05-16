#!/usr/bin/env python3
"""Expanded BAliBase corpus-summary table.

Reads the ``expected_balibase_*.json`` and ``*_l150.json`` files
produced by ``tkf-mixdom/python/experiments/expected_pairwise_balibase.py``
plus the infinite-PHMM sampler JSONs in
``~/tkf-dp/math-paper/results/``, and emits per-method rows with
the column structure described in TASK B (Claude / 2026-05-13):

Soft methods (TKF92, MixDom, TKF92-K20, CherryML-C20, infinite-PHMM-K4):
  * E_pairs[F1]                  -- corpus_post.micro
  * E_pairs[e_tp]                -- corpus_post.micro
  * E_pairs[F1(optacc)]          -- corpus_opt.micro (Holmes-Durbin DP)
  * E_pairs[e_tp(optacc)]        -- corpus_opt.micro
  * E_pairs in FSA0 MSA[F1]      -- corpus_fsa_sps.micro
  * E_pairs in FSA0 MSA[SPS]     -- average per-family msa_sp_g0
  * E_pairs in FSA0 MSA[TCS]     -- average per-family msa_tc_g0
  * E_pairs in FSA1 MSA[F1]      -- corpus_hard.micro
  * E_pairs in FSA1 MSA[SPS]     -- average per-family msa_sp_g1
  * E_pairs in FSA1 MSA[TCS]     -- average per-family msa_tc_g1
  * E_pairs,L<150[F1]            -- same as col 1 filtered to L<150
  * E_pairs,L<150[e_tp]
  * E_pairs,L<150[F1(optacc)]
  * E_pairs,L<150[e_tp(optacc)]

Hard methods (MAFFT, MUSCLE):
  Posterior + optacc + FSA0 columns left blank ("--");
  FSA1 columns are the method's own MSA (= corpus_hard).

Inputs:
  ~/tkf-mixdom/python/experiments/expected_balibase/expected_balibase_<METHOD>.json
  ~/tkf-mixdom/python/experiments/expected_balibase/expected_balibase_<METHOD>_l150.json
  ~/tkf-dp/math-paper/results/infinite_phmm_balibase_k4.json
  ~/tkf-dp/math-paper/results/infinite_phmm_balibase_k4_replicaexchange.json
  ~/tkf-dp/math-paper/results/infinite_phmm_balibase_k4_top_rung_validation.json

Outputs:
  ~/tkf-dp/math-paper/results/balibase_summary_full.csv
  ~/tkf-dp/math-paper/results/balibase_summary_full.md
  ~/tkf-dp/math-paper/results/balibase_summary_l150.csv
  ~/tkf-dp/math-paper/results/balibase_summary_l150.md

Also prints the markdown tables to stdout.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Any


REPO = Path(__file__).resolve().parents[2]    # ~/tkf-dp

EXPECTED_BALIBASE_DIR = (
    Path('~/tkf-mixdom/python/experiments/expected_balibase').expanduser())
RESULTS_DIR = REPO / "math-paper" / "results"
BALI_IN = Path('~/bio-datasets/data/balibase/bali3pdbm/in').expanduser()


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------


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


def f1_from_micro(m: Optional[Mapping[str, float]]) -> Optional[float]:
    if not m:
        return None
    den = m.get('total_mass', 0) + m.get('gold', 0)
    if den <= 0:
        return None
    return 2 * m['e_tp'] / den


def pool_per_pair(per_pair_lists: List[List[Mapping[str, Any]]]) -> Optional[Dict[str, float]]:
    """Pool a list of per-pair lists into corpus-level sufficient stats."""
    pairs = []
    for lst in per_pair_lists:
        if lst is None:
            continue
        pairs.extend(lst)
    if not pairs:
        return None
    return {
        'e_tp': float(sum(r['e_tp'] for r in pairs)),
        'total_mass': float(sum(r['total_mass'] for r in pairs)),
        'gold': int(sum(r['gold'] for r in pairs)),
        'n_cells': int(sum(r.get('n_cells', 0) for r in pairs)),
        'n_pairs': int(len(pairs)),
    }


# --------------------------------------------------------------------------
# Soft / hard method JSON readers.
# --------------------------------------------------------------------------


def _filter_per_family(per_family: List[Mapping], max_len: Optional[int]) -> List[Mapping]:
    """If max_len is None, return all not-failed families. Otherwise
    additionally filter to families with longest-ungapped-seq < max_len."""
    out = []
    for f in per_family:
        if f.get('_failed_family'):
            continue
        if max_len is not None and family_max_len(f['family']) >= max_len:
            continue
        out.append(f)
    return out


def extract_balibase_json(path: Path, max_len: Optional[int]) -> Optional[Dict[str, Any]]:
    """Read a tkf-mixdom expected_balibase_*.json (soft or hard) and
    return the per-corpus stats restricted to the optionally-filtered
    family set."""
    if not path.exists():
        return None
    d = json.load(path.open())
    per_family = d.get('per_family', [])
    keep = _filter_per_family(per_family, max_len)
    method_name = d.get('method_name', d.get('config', {}).get('method', '?'))
    method_type = d.get('method_type', '?')   # 'soft' or 'hard'

    # Re-pool per_pair_{post,opt,hard,fsa_sps} on the filtered set.
    pools = {}
    for branch in ('post', 'opt', 'hard', 'fsa_sps'):
        pools[branch] = pool_per_pair(
            [f.get(f'per_pair_{branch}') for f in keep])

    # Average per-family MSA-level SP/TC at gap-factor 0/1.
    def avg_field(field: str) -> Optional[float]:
        vals = [f.get(field) for f in keep
                 if f.get(field) is not None]
        return float(sum(vals)) / len(vals) if vals else None

    return {
        'method_name': method_name,
        'method_type': method_type,
        'n_families': len(keep),
        'n_pairs': pools['hard']['n_pairs'] if pools['hard'] else (
            pools['post']['n_pairs'] if pools['post'] else 0),
        'pools': pools,
        'msa_sp_g0_mean': avg_field('msa_sp_g0'),
        'msa_tc_g0_mean': avg_field('msa_tc_g0'),
        'msa_sp_g1_mean': avg_field('msa_sp_g1'),
        'msa_tc_g1_mean': avg_field('msa_tc_g1'),
    }


# --------------------------------------------------------------------------
# Infinite-PHMM JSON reader (different shape: {'mcmc_config', 'per_family'}
# OR legacy bare list of family dicts each with 'per_pair'; we accept both).
# --------------------------------------------------------------------------


def extract_infinite_phmm_json(path: Path, max_len: Optional[int],
                                method_label: str) -> Optional[Dict[str, Any]]:
    """Read an infinite_phmm_balibase*.json sampler output.  The
    sampler emits hard 0/1 per-pair posteriors and a Q' soft posterior;
    expected_pair_f1's output is the soft suff stats for Q' (so it goes
    in the corpus_post / E_pairs[F1] column).  No FSA branch."""
    if not path.exists():
        return None
    raw = json.load(path.open())
    if isinstance(raw, dict) and 'per_family' in raw:
        per_family = raw['per_family']
        method_name = raw.get('method_label', method_label)
        mcmc_config = raw.get('mcmc_config', {})
    elif isinstance(raw, list):
        per_family = raw
        method_name = method_label
        mcmc_config = {}
    else:
        return None

    keep = []
    for f in per_family:
        if f.get('error') or f.get('skipped'):
            continue
        if max_len is not None and family_max_len(f.get('family', '')) >= max_len:
            continue
        if 'per_pair' in f and f['per_pair']:
            keep.append(f)
    if not keep:
        return None

    # Pool soft posterior Q' suff stats. The 'opt_acc_e_tp'/'opt_acc_total_mass'
    # are pooled into an 'optacc' branch.
    pp_post = []
    pp_opt = []
    pp_baseline = []  # FB-posterior baseline (only when sampler logged it).
    for f in keep:
        for r in f['per_pair']:
            pp_post.append(r)
            if 'opt_acc_e_tp' in r and 'opt_acc_total_mass' in r:
                # n_cells / gold are common across all branches for a pair.
                pp_opt.append({
                    'e_tp': r['opt_acc_e_tp'],
                    'total_mass': r['opt_acc_total_mass'],
                    'gold': r['gold'],
                    'n_cells': r.get('n_cells', 0),
                })
            if 'baseline_e_tp' in r and 'baseline_total_mass' in r:
                pp_baseline.append({
                    'e_tp': r['baseline_e_tp'],
                    'total_mass': r['baseline_total_mass'],
                    'gold': r['gold'],
                    'n_cells': r.get('n_cells', 0),
                })

    pools = {
        'post': pool_per_pair([pp_post]),
        'opt': pool_per_pair([pp_opt]) if pp_opt else None,
        'hard': None,
        'fsa_sps': None,
        'baseline': pool_per_pair([pp_baseline]) if pp_baseline else None,
    }
    return {
        'method_name': method_name,
        'method_type': 'soft',
        'n_families': len(keep),
        'n_pairs': len(pp_post),
        'pools': pools,
        'mcmc_config': mcmc_config,
        'msa_sp_g0_mean': None,
        'msa_tc_g0_mean': None,
        'msa_sp_g1_mean': None,
        'msa_tc_g1_mean': None,
    }


# --------------------------------------------------------------------------
# Column assembly.
# --------------------------------------------------------------------------


# Column order matches TASK B.
COLUMN_HEADERS = [
    # E_pairs (full corpus or subset, controlled by extract_*).
    'F1', 'e_tp',
    'F1(optacc)', 'e_tp(optacc)',
    # E_pairs in FSA0 MSA: F1 from corpus_fsa_sps; SPS, TCS from avg msa_*_g0.
    'F1[FSA0]', 'SPS[FSA0]', 'TCS[FSA0]',
    # E_pairs in FSA1 MSA: F1 from corpus_hard; SPS, TCS from avg msa_*_g1.
    'F1[FSA1]', 'SPS[FSA1]', 'TCS[FSA1]',
    # L<150 short-corpus repeats (post / opt only -- the FSA + MSA-SP
    # columns are already reported via the L150 row).
    'F1(L<150)', 'e_tp(L<150)',
    'F1(optacc;L<150)', 'e_tp(optacc;L<150)',
]


# Wraps None as a dash and floats with 3 decimals.
def _fmt(x: Optional[float], width: int = 6,
         missing: str = '--', fmt: str = '{:.3f}') -> str:
    if x is None:
        return missing.rjust(width)
    return fmt.format(x).rjust(width)


def _fmt_int(x: Optional[float], width: int = 6,
             missing: str = '--', fmt: str = '{:.0f}') -> str:
    if x is None:
        return missing.rjust(width)
    return fmt.format(x).rjust(width)


def assemble_row(soft_row: Dict[str, Any],
                  l150_row: Optional[Dict[str, Any]],
                  is_hard: bool = False) -> List[Optional[float]]:
    """Return a list of length len(COLUMN_HEADERS) -- the soft / hard
    cells for a method.  For hard methods (no posterior / optacc /
    FSA0 branches), those cells are None."""
    p = soft_row['pools']

    row: List[Optional[float]] = []
    if is_hard:
        # F1, e_tp from posterior: not applicable.
        row.append(None); row.append(None)
        # F1(optacc), e_tp(optacc): not applicable.
        row.append(None); row.append(None)
        # FSA0 columns: not applicable (no posterior to feed into FSA).
        row.append(None); row.append(None); row.append(None)
    else:
        # F1, e_tp (E_pairs over corpus_post.micro).
        post = p.get('post')
        row.append(f1_from_micro(post)); row.append(post['e_tp'] if post else None)
        # F1(optacc), e_tp(optacc).
        opt = p.get('opt')
        row.append(f1_from_micro(opt)); row.append(opt['e_tp'] if opt else None)
        # FSA0 (corpus_fsa_sps + msa_*_g0).
        fsa0 = p.get('fsa_sps')
        row.append(f1_from_micro(fsa0))
        row.append(soft_row.get('msa_sp_g0_mean'))
        row.append(soft_row.get('msa_tc_g0_mean'))

    # FSA1 (corpus_hard + msa_*_g1) -- applies to BOTH soft and hard.
    hard = p.get('hard')
    row.append(f1_from_micro(hard))
    row.append(soft_row.get('msa_sp_g1_mean'))
    row.append(soft_row.get('msa_tc_g1_mean'))

    # L<150 columns.
    if l150_row is not None:
        p150 = l150_row['pools']
        if is_hard:
            # For hard methods at L<150, F1 / e_tp = corpus_hard pool.
            hard150 = p150.get('hard')
            row.append(f1_from_micro(hard150))
            row.append(hard150['e_tp'] if hard150 else None)
            # optacc not applicable.
            row.append(None); row.append(None)
        else:
            post150 = p150.get('post')
            row.append(f1_from_micro(post150))
            row.append(post150['e_tp'] if post150 else None)
            opt150 = p150.get('opt')
            row.append(f1_from_micro(opt150))
            row.append(opt150['e_tp'] if opt150 else None)
    else:
        row.extend([None] * 4)

    return row


def render_markdown(rows: List[Tuple[str, List[Optional[float]]]],
                     header: List[str]) -> str:
    """Render a markdown table.  rows is a list of (method_label, cells)."""
    lines = []
    # Header
    lines.append('| Method | ' + ' | '.join(header) + ' |')
    # Separator
    lines.append('|--------|' + '|'.join(['---'] * len(header)) + '|')
    for label, cells in rows:
        cells_fmt = []
        for col_name, c in zip(header, cells):
            if c is None:
                cells_fmt.append('---')
            elif 'e_tp' in col_name:
                cells_fmt.append(f'{c:.1f}')
            else:
                cells_fmt.append(f'{c:.3f}')
        lines.append(f'| {label} | ' + ' | '.join(cells_fmt) + ' |')
    return '\n'.join(lines)


def render_text_table(rows: List[Tuple[str, List[Optional[float]]]],
                        header: List[str]) -> str:
    """Plain-text table for stdout (wider than markdown, with column padding)."""
    lines = []
    label_w = max(len(label) for label, _ in rows) if rows else 8
    label_w = max(label_w, len('Method'))
    col_w = max(8, max(len(c) for c in header) + 1)
    head = f"{'Method'.ljust(label_w)}  " + ' '.join(c.rjust(col_w) for c in header)
    lines.append(head)
    lines.append('-' * len(head))
    for label, cells in rows:
        cells_fmt = []
        for col_name, c in zip(header, cells):
            if c is None:
                cells_fmt.append(_fmt(None, width=col_w))
            elif 'e_tp' in col_name:
                cells_fmt.append(_fmt(c, width=col_w, fmt='{:.1f}'))
            else:
                cells_fmt.append(_fmt(c, width=col_w, fmt='{:.3f}'))
        lines.append(f"{label.ljust(label_w)}  " + ' '.join(cells_fmt))
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------


# Ordered list of methods to include.  Each entry:
#   (display_label, soft_file_basename (None for hard-only/infinite-PHMM),
#    is_hard, infinite_phmm_path_or_None, infinite_phmm_label_or_None)
METHOD_TABLE = [
    # Soft methods.
    ('TKF92',         'tkf92',         False, None, None),
    ('MixDom-d3f1',   'mixdom_d3f1',   False, None, None),
    ('TKF92-K20',     'tkf92_K20',     False, None, None),
    ('CherryML-C20',  'cherryml_C20',  False, None, None),
    # Hard methods.
    ('MAFFT',         'mafft',         True,  None, None),
    ('MUSCLE',        'muscle',        True,  None, None),
    # Infinite PHMM K=4.
    ('inf-PHMM-K4 (single)',
        None, False,
        RESULTS_DIR / 'infinite_phmm_balibase_k4.json',
        'infinite_phmm_mcmc_K1_stub'),
    # Replica-exchange variant (only present when Task A step 6 has run).
    ('inf-PHMM-K4 (RE)',
        None, False,
        RESULTS_DIR / 'infinite_phmm_balibase_k4_replicaexchange.json',
        'infinite_phmm_mcmc_K4_coupled_RE'),
    # Top-rung validation (only present after Task A step 5 has run).
    ('inf-PHMM (top rung)',
        None, False,
        RESULTS_DIR / 'infinite_phmm_balibase_k4_top_rung_validation.json',
        'infinite_phmm_top_rung'),
]


def build_method_rows(max_len: Optional[int] = None,
                       max_len_for_l150: int = 150) -> List[Tuple[str, List[Optional[float]]]]:
    """Build the per-method rows for a single column-set (either full
    corpus = max_len=None or L<150 = max_len=150)."""
    out: List[Tuple[str, List[Optional[float]]]] = []
    for label, basename, is_hard, ipm_path, ipm_label in METHOD_TABLE:
        soft_row = None
        l150_row = None
        if basename is not None:
            soft_path = EXPECTED_BALIBASE_DIR / f'expected_balibase_{basename}.json'
            l150_path = EXPECTED_BALIBASE_DIR / f'expected_balibase_{basename}_l150.json'
            # If the leading-column set is itself the L<150 subset, prefer
            # the _l150.json file (which has the MSA-level SP/TC fields
            # that the legacy full-corpus JSON lacks).
            if (max_len is not None and max_len == max_len_for_l150
                    and l150_path.exists()):
                soft_row = extract_balibase_json(l150_path, max_len=max_len)
            else:
                soft_row = extract_balibase_json(soft_path, max_len=max_len)
                if soft_row is None:
                    # Fall back to the L<150 file restricted to itself (so an
                    # entry still appears even if the full-corpus file is
                    # absent).
                    soft_row = extract_balibase_json(l150_path, max_len=max_len)
            l150_row = extract_balibase_json(l150_path,
                                              max_len=max_len_for_l150)
        elif ipm_path is not None:
            # Infinite-PHMM file: only L<150 rows are meaningful.
            row = extract_infinite_phmm_json(ipm_path, max_len=max_len_for_l150,
                                              method_label=ipm_label)
            if row is None:
                continue
            soft_row = row
            l150_row = row    # same content; the JSON IS the L<150 sweep
        else:
            continue
        if soft_row is None:
            continue
        cells = assemble_row(soft_row, l150_row, is_hard=is_hard)
        # Annotate methods that ran only the L<150 subset.
        suffix = ''
        if basename is None and ipm_path is not None:
            cfg = soft_row.get('mcmc_config', {})
            n_chains = cfg.get('n_chains')
            ladder = cfg.get('alpha_z_ladder')
            top = cfg.get('top_rung_only')
            tags = []
            if ladder is not None:
                tags.append(f'ladder={len(ladder)}r')
            if n_chains is not None and n_chains > 1 and ladder is None:
                tags.append(f'C{n_chains}')
            if top:
                tags.append('topRung')
            if tags:
                suffix = f' [{", ".join(tags)}]'
        out.append((label + suffix, cells))
    return out


def write_csv(rows: List[Tuple[str, List[Optional[float]]]],
                header: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['method'] + header)
        for label, cells in rows:
            w.writerow([label] + [
                ('' if c is None else f'{c:.6f}') for c in cells
            ])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--max-len-for-l150', type=int, default=150,
                   help='Eligibility ceiling for the L<150 columns.')
    p.add_argument('--out-prefix', type=str,
                   default=str(RESULTS_DIR / 'balibase_summary'),
                   help='Output basename (without extension).')
    args = p.parse_args()

    # Full-corpus rows (no L cap on the first set of columns; L<150 cap
    # on the trailing four columns).
    rows_full = build_method_rows(max_len=None,
                                   max_len_for_l150=args.max_len_for_l150)
    # L<150-only rows -- both the leading and trailing columns are
    # restricted to the L<150 subset.  Useful as a standalone table.
    rows_l150 = build_method_rows(max_len=args.max_len_for_l150,
                                    max_len_for_l150=args.max_len_for_l150)

    md_full = render_markdown(rows_full, COLUMN_HEADERS)
    md_l150 = render_markdown(rows_l150, COLUMN_HEADERS)
    txt_full = render_text_table(rows_full, COLUMN_HEADERS)
    txt_l150 = render_text_table(rows_l150, COLUMN_HEADERS)

    print('## Full BAliBase bali3pdbm corpus (E_pairs columns), L<150 subset (last 4 cols)\n')
    print(txt_full)
    print('\n\n## L<150 BAliBase bali3pdbm subset (all columns restricted)\n')
    print(txt_l150)

    out_full_csv = Path(args.out_prefix + '_full.csv')
    out_full_md = Path(args.out_prefix + '_full.md')
    out_l150_csv = Path(args.out_prefix + '_l150.csv')
    out_l150_md = Path(args.out_prefix + '_l150.md')

    write_csv(rows_full, COLUMN_HEADERS, out_full_csv)
    write_csv(rows_l150, COLUMN_HEADERS, out_l150_csv)
    out_full_md.write_text(
        '# BAliBase bali3pdbm corpus summary (E_pairs columns; '
        'last 4 cols restricted to L<150).\n\n' + md_full + '\n')
    out_l150_md.write_text(
        '# BAliBase L<150 subset (max-seq-len < 150 across all columns).\n\n'
        + md_l150 + '\n')
    print(f'\nWrote:\n  {out_full_csv}\n  {out_full_md}\n  {out_l150_csv}\n  {out_l150_md}')


if __name__ == '__main__':
    main()
