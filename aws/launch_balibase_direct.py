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

# 22 families × ~9 pairs ≈ 210 (some get filtered at runtime by max-len).
FAMILIES = [
    ('BB11001', 4),
    ('BB11013', 5),
    ('BB11021', 3),
    ('BB11029', 3),
    ('BB11035', 4),
    ('BB12014', 6),
    ('BB12021', 4),
    ('BB12032', 5),
    ('BB12041', 3),
    ('BB20001', 3),
    ('BB20008', 8),
    ('BB20015', 4),
    ('BB20030', 4),
    ('BB20033', 3),
    ('BB20038', 3),
    ('BB30015', 5),
    ('BB30022', 3),
    ('BB30025', 6),
    ('BB40018', 3),
    ('BB40029', 9),
    ('BB40038', 4),
    ('BB40045', 3)
]


def enumerate_pairs():
    from itertools import combinations
    out = []
    for fam, n in FAMILIES:
        for i, j in combinations(range(n), 2):
            out.append((fam, i, j))
    return out


def pairs_already_in_s3():
    """Return set of (FAMILY, I, J) triples that already have a result
    JSON on S3 (so we skip them)."""
    r = subprocess.run(
        ['aws', 's3', 'ls',
         's3://tkf-mixdom-gpu-618647024028/balibase-runs/v11-aws-canonical/'],
        env={**os.environ, 'AWS_PROFILE': 'tkf-gpu'},
        capture_output=True, text=True, timeout=60)
    done = set()
    for line in r.stdout.splitlines():
        m = re.search(r'\s(BB\d+)_(\d+)_(\d+)\.json$', line)
        if m:
            done.add((m.group(1), int(m.group(2)), int(m.group(3))))
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
    args = ap.parse_args()

    pairs = enumerate_pairs()
    done = pairs_already_in_s3()
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
        cmd = [SKY, 'launch', '-c', name,
               '--env', f'FAMILY={fam}',
               '--env', f'PAIR_I={i}',
               '--env', f'PAIR_J={j}',
               '--down',
               '--idle-minutes-to-autostop', '20',
               '-y',
               'aws/balibase_one_pair_v2.yaml']
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
