#!/usr/bin/env python3
"""Direct-launch driver for BAliBASE pair sweep — bypasses sky's
managed-jobs controller (which wedged twice under 143 concurrent
launches). Uses regular `sky launch -c <name> --down -y` per pair;
each cluster is independent.

Local-side concurrency is bounded so we don't hammer AWS API.
"""
import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

SKY = shutil.which('sky') or os.path.expanduser(
    '~/tkf-mixdom/python/.venv/bin/sky')

# 22 BAliBASE bali3pdbm families with max_seq_len < 150 (n_seqs as in
# the v2 launcher; runtime --max-len 150 filter discards individual
# seqs above the threshold). C(n, 2) per family, total = 210 pairs.
FAMILIES = [
    ('BB11001', 6),  ('BB11013', 5),  ('BB11021', 4),  ('BB11029', 4),
    ('BB11035', 4),  ('BB12014', 5),  ('BB12021', 5),  ('BB12032', 5),
    ('BB12041', 3),  ('BB20001', 5),  ('BB20008', 4),  ('BB20015', 6),
    ('BB20030', 4),  ('BB20033', 4),  ('BB20038', 4),  ('BB30015', 4),
    ('BB30022', 4),  ('BB30025', 6),  ('BB40018', 4),  ('BB40029', 9),
    ('BB40038', 4),  ('BB40045', 5),
]


def enumerate_pairs():
    from itertools import combinations
    out = []
    for fam, n in FAMILIES:
        for i, j in combinations(range(n), 2):
            out.append((fam, i, j))
    return out


def pairs_already_in_s3(prefix):
    """Return set of (FAMILY, I, J) triples that already have a >100kB
    result JSON on S3 under prefix (so we skip them). Small stubs
    (~600B) from broken bundles are NOT counted — we want to redo them."""
    r = subprocess.run(
        ['aws', 's3', 'ls',
         f's3://tkf-mixdom-gpu-<AWS_ACCOUNT>/{prefix.rstrip("/")}/'],
        env={**os.environ, 'AWS_PROFILE': 'tkf-gpu'},
        capture_output=True, text=True, timeout=60)
    done = set()
    for line in r.stdout.splitlines():
        m = re.search(r'(\d+)\s+(BB\d+)_(\d+)_(\d+)\.json$', line)
        if m:
            size = int(m.group(1))
            if size > 100_000:
                done.add((m.group(2), int(m.group(3)), int(m.group(4))))
    return done


def active_launches(prefix='balibase-direct-'):
    r = subprocess.run(['pgrep', '-af', f'sky launch -c {prefix}'],
                       capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--concurrency', type=int, default=8,
                    help='Max concurrent sky launches')
    ap.add_argument('--gap-seconds', type=float, default=4.0,
                    help='Sleep between launches')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--yaml', default='aws/balibase_one_pair_v2.yaml',
                    help='Worker task YAML (e.g. aws/balibase_one_pair_k8.yaml).')
    ap.add_argument('--s3-prefix', default='balibase-runs/v11-aws-canonical',
                    help='S3 prefix to check for already-done pairs.')
    ap.add_argument('--name-suffix', default='',
                    help='Optional suffix appended to each sky cluster name '
                         '(e.g. "supH") to avoid collisions with stale clusters '
                         'from previous runs.')
    args = ap.parse_args()

    pairs = enumerate_pairs()
    done = pairs_already_in_s3(args.s3_prefix)
    # Also skip pairs whose direct-launch cluster is already in flight
    # (from a previous instance of this driver that we superseded).
    inflight_path = Path('/tmp/inflight_clusters.txt')
    inflight = set()
    if inflight_path.exists():
        for line in inflight_path.read_text().splitlines():
            m = re.match(r'balibase-direct-(\w+)-(\d+)-(\d+)', line.strip())
            if m:
                inflight.add((m.group(1).upper(), int(m.group(2)),
                              int(m.group(3))))
    pending = [p for p in pairs if p not in done and p not in inflight]
    print(f'Total pairs: {len(pairs)}')
    print(f'Already in S3: {len(done)}')
    print(f'Already in flight: {len(inflight)}')
    print(f'To launch: {len(pending)}')
    print(f'Concurrency: {args.concurrency}, gap: {args.gap_seconds}s')

    if args.dry_run:
        for p in pending[:5]:
            print('  WOULD launch', p)
        return 0

    for k, (fam, i, j) in enumerate(pending):
        # Throttle: wait if too many in flight
        while active_launches() >= args.concurrency:
            time.sleep(args.gap_seconds)
        name = f'balibase-direct-{fam.lower()}-{i}-{j}'
        if args.name_suffix:
            name = f'{name}-{args.name_suffix}'
        cmd = [SKY, 'launch', '-c', name,
               '--env', f'FAMILY={fam}',
               '--env', f'PAIR_I={i}',
               '--env', f'PAIR_J={j}',
               '--down',
               '--idle-minutes-to-autostop', '20',
               '-y',
               args.yaml]
        log = f'/tmp/sky_direct_{name}.log'
        # Fire and forget; redirect output to per-pair log
        with open(log, 'w') as f:
            subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f'  [{k+1}/{len(pending)}] launched {fam} ({i}, {j}) '
              f'-> {name}')
        time.sleep(args.gap_seconds)
    print('all submitted')


if __name__ == '__main__':
    main()
