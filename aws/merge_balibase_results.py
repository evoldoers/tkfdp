#!/usr/bin/env python3
"""Merge per-pair AWS job outputs back into a single replicaexchange.json
+ a per-family Q'-cache mirror.

After the AWS sweep completes, each pair's result lives at:
  s3://<bucket>/balibase-runs/v10-aws/<FAM>_<I>_<J>.json          (diag)
  s3://<bucket>/balibase-runs/v10-aws/qprime/<FAM>_<I>_<J>.npz    (Q')
  s3://<bucket>/balibase-runs/v10-aws/qprime/<FAM>_<I>_<J>_qpcache.json  (Q' meta)

This script pulls everything under the prefix and:
  1. Merges the diagnostics JSONs into the canonical per_family
     structure expected by the downstream FSA / Table-1 tooling.
  2. Stitches the per-pair Q'-cache .npz files into one .npz per
     family, in the format the local watcher expects in
     math-paper/results/qprime_cache/infinite_phmm_mcmc_K4_coupled_RE/.

Usage:
    python aws/merge_balibase_results.py \
        --prefix balibase-runs/v10-aws \
        --out math-paper/results/infinite_phmm_balibase_k4_replicaexchange.json \
        --qprime-cache-dir math-paper/results/qprime_cache/infinite_phmm_mcmc_K4_coupled_RE
"""
import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

S3_BUCKET = 'tkf-mixdom-gpu-618647024028'
AWS_PROFILE = 'tkf-gpu'


def list_pair_files(prefix):
    """List all JSON files under s3://{bucket}/{prefix}/."""
    out = subprocess.check_output(
        ['aws', 's3', 'ls', f's3://{S3_BUCKET}/{prefix}/',
         '--profile', AWS_PROFILE], text=True)
    files = []
    for line in out.splitlines():
        parts = line.split()
        if not parts or not parts[-1].endswith('.json'):
            continue
        files.append(parts[-1])
    return sorted(files)


def fetch_pair(prefix, name, tmpdir):
    """Download one pair's JSON locally; return parsed dict."""
    local = Path(tmpdir) / name
    subprocess.check_call(
        ['aws', 's3', 'cp', f's3://{S3_BUCKET}/{prefix}/{name}', str(local),
         '--profile', AWS_PROFILE, '--quiet'])
    return json.loads(local.read_text())


def _merge_qprime_npz(per_pair_npzs, fam_out_path):
    """Per-family stitching: each input is one (FAM, I, J).npz with
    post_<i,j>_<X|Y> entries; output is one <FAM>.npz with all
    pairs' Q' arrays concatenated, matching the watcher schema."""
    import numpy as np
    merged = {}
    for npz_path in per_pair_npzs:
        with np.load(npz_path, allow_pickle=True) as z:
            for k in z.files:
                # First write wins for any shared metadata; per-pair
                # post_* keys are pair-specific so collisions are
                # not expected.
                if k not in merged:
                    merged[k] = z[k]
    fam_out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fam_out_path, **merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='balibase-runs/v10-aws')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--qprime-cache-dir', type=Path, default=None,
                    help='If set, also stitch per-pair .npz/.json from '
                         's3://<bucket>/<prefix>/qprime/ into per-family '
                         'files at this directory (matching the local '
                         'watcher mirror schema).')
    args = ap.parse_args()

    names = list_pair_files(args.prefix)
    # Filter out the qprime/ sub-prefix entries.
    diag_names = [n for n in names if not n.startswith('qprime/')]
    print(f'Found {len(diag_names)} diagnostics JSONs under '
          f's3://{S3_BUCKET}/{args.prefix}/')

    if not diag_names:
        print('Nothing to merge.', file=sys.stderr)
        return 2

    by_family = defaultdict(list)  # fam -> list of per_pair dicts
    mcmc_config = None
    method_label = None

    with tempfile.TemporaryDirectory() as tmp:
        for i, name in enumerate(diag_names):
            d = fetch_pair(args.prefix, name, tmp)
            if mcmc_config is None:
                mcmc_config = d.get('mcmc_config')
                method_label = d.get('method_label')
            for fam_obj in d.get('per_family', []):
                fam = fam_obj['family']
                for pair in fam_obj.get('per_pair', []):
                    by_family[fam].append(pair)
            if (i + 1) % 20 == 0:
                print(f'  fetched diag {i+1}/{len(diag_names)}')

    # Reconstruct the per_family structure.
    per_family = []
    for fam in sorted(by_family):
        pairs = sorted(by_family[fam], key=lambda p: tuple(p.get('pair', (0, 0))))
        per_family.append({
            'family': fam,
            'n_seqs': pairs[0].get('n_seqs') if pairs else None,
            'per_pair': pairs,
        })

    merged = {
        'method_label': method_label,
        'mcmc_config': mcmc_config,
        'per_family': per_family,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2))
    print(f'Wrote {args.out}: {sum(len(f["per_pair"]) for f in per_family)} pairs '
          f'across {len(per_family)} families.')

    # Optional Q' cache stitching.
    if args.qprime_cache_dir:
        print()
        print('Stitching Q\\'-cache per family...')
        qprime_prefix = f'{args.prefix}/qprime'
        qprime_names = list_pair_files(qprime_prefix)
        npz_names = [n for n in qprime_names if n.endswith('.npz')]
        print(f'  found {len(npz_names)} per-pair .npz files at '
              f's3://{S3_BUCKET}/{qprime_prefix}/')
        # Group by family: filenames are <FAM>_<I>_<J>.npz
        by_fam_npz = defaultdict(list)
        for n in npz_names:
            fam = n.split('_')[0]  # BB12041 from BB12041_0_1.npz
            by_fam_npz[fam].append(n)
        with tempfile.TemporaryDirectory() as tmp:
            for fam, fam_npzs in sorted(by_fam_npz.items()):
                local_paths = []
                for n in fam_npzs:
                    local = Path(tmp) / n
                    subprocess.check_call(
                        ['aws', 's3', 'cp',
                         f's3://{S3_BUCKET}/{qprime_prefix}/{n}',
                         str(local), '--profile', AWS_PROFILE, '--quiet'])
                    local_paths.append(local)
                out_npz = args.qprime_cache_dir / f'{fam}.npz'
                _merge_qprime_npz(local_paths, out_npz)
                print(f'  {fam}: stitched {len(local_paths)} pairs -> {out_npz}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
