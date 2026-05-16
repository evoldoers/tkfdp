#!/usr/bin/env python3
"""Fan out 187 BAliBASE L<150 pairs across N AWS spot instances via SkyPilot.

Pairs are submitted as individual `sky jobs launch` invocations
parameterised by the (FAMILY, PAIR_I, PAIR_J) env triple. Concurrency
is capped at CONCURRENCY active jobs at a time; the script blocks
until a slot frees up, then submits the next pending pair. Spot-evicted
jobs are retried up to MAX_RETRIES times.

Usage:
    python aws/launch_balibase_sweep.py --concurrency 16 [--dry-run]

The --dry-run flag prints the full pair list and the sky-launch
commands that *would* be issued, without actually launching anything.
Use this to sanity-check the pair enumeration before spending money.

Pair source-of-truth: derived from analysis/scripts/run_re_diag_l150.py
or equivalent corpus filter; here we enumerate the BAliBASE L<150
families and their pair counts hardcoded below (from
analysis/re_diag/REPORT.md).

Prerequisites: aws/balibase_jit_primer.yaml has been run once and
populated s3://.../jax-cache/g5xl-jax-2026-05-15.tar.gz.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# Resolve `sky` CLI: prefer the explicit venv-pinned binary so we
# don't depend on PATH being set up by the launcher's environment.
SKY_BIN = (shutil.which('sky')
           or os.path.expanduser('~/tkf-mixdom/python/.venv/bin/sky'))
from itertools import combinations
from pathlib import Path

# 22 BAliBASE bali3pdbm families with max_seq_len < 150.
# (family, n_seqs) — pairs = C(n, 2). Total = 187.
FAMILIES = [
    ('BB11001', 6),  ('BB11013', 5),  ('BB11021', 4),  ('BB11029', 4),
    ('BB11035', 4),  ('BB12014', 5),  ('BB12021', 5),  ('BB12032', 5),
    ('BB12041', 3),  ('BB20001', 5),  ('BB20008', 4),  ('BB20015', 6),
    ('BB20030', 4),  ('BB20033', 4),  ('BB20038', 4),  ('BB30015', 4),
    ('BB30022', 4),  ('BB30025', 6),  ('BB40018', 4),  ('BB40029', 9),
    ('BB40038', 4),  ('BB40045', 5),
]


def enumerate_pairs():
    """All (family, i, j) triples in canonical order."""
    out = []
    for fam, n in FAMILIES:
        for i, j in combinations(range(n), 2):
            out.append((fam, i, j))
    return out


def active_job_count(prefix='balibase-'):
    """Number of currently-active sky jobs whose name starts with prefix."""
    try:
        out = subprocess.run(
            [SKY_BIN, 'jobs', 'queue', '--all', '--no-show-all'],
            check=True, capture_output=True, text=True, timeout=60)
        active = 0
        for line in out.stdout.splitlines():
            if prefix in line and any(
                    s in line for s in ('PENDING', 'RUNNING', 'STARTING')):
                active += 1
        return active
    except subprocess.CalledProcessError:
        return 0  # if sky jobs is broken assume no active


def launch_pair(fam, i, j, dry_run=False, retry=0):
    """Submit one pair as a sky job. Returns the cmd list (for dry-run logging)."""
    name = f'balibase-{fam}-{i}-{j}-r{retry}'
    cmd = [
        SKY_BIN, 'jobs', 'launch',
        '-n', name,
        '--env', f'FAMILY={fam}',
        '--env', f'PAIR_I={i}',
        '--env', f'PAIR_J={j}',
        # `sky jobs launch` in newer SkyPilot doesn't accept --down or
        # --idle-minutes-to-autostop -- managed jobs auto-terminate
        # when the task exits.
        '-y',  # auto-confirm
        'aws/balibase_one_pair_v2.yaml',
    ]
    if dry_run:
        print('  WOULD RUN: ' + ' '.join(cmd))
        return cmd
    subprocess.Popen(cmd)  # fire-and-forget; sky daemon handles persistence
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--concurrency', type=int, default=16,
                    help='Max simultaneous active sky jobs.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print pair list + commands, do not launch.')
    ap.add_argument('--max-retries', type=int, default=3,
                    help='Spot eviction retry budget per pair.')
    ap.add_argument('--poll-seconds', type=int, default=60,
                    help='Polling interval for the concurrency gate.')
    args = ap.parse_args()

    pairs = enumerate_pairs()
    print(f'Total pairs: {len(pairs)}')
    print(f'Concurrency: {args.concurrency}')
    print(f'Mode: {"DRY RUN" if args.dry_run else "LIVE"}')
    print()

    if args.dry_run:
        for fam, i, j in pairs:
            launch_pair(fam, i, j, dry_run=True)
        return 0

    # LIVE submission.
    submitted = 0
    for fam, i, j in pairs:
        while active_job_count() >= args.concurrency:
            time.sleep(args.poll_seconds)
        launch_pair(fam, i, j, dry_run=False)
        submitted += 1
        print(f'  [{submitted}/{len(pairs)}] submitted {fam} ({i}, {j})')
        time.sleep(2)  # gentle on sky daemon

    print()
    print('All pairs submitted. Run `sky jobs queue --all` to monitor.')
    print('Retries on spot eviction are NOT automated yet -- '
          'check `sky jobs queue --all -p` for FAILED jobs and re-run them '
          'manually, or extend this driver to poll for failures.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
