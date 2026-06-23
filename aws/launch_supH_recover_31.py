#!/usr/bin/env python3
"""Re-launch the 31 supH pairs missing from supH5 (filter-trimmed)."""
import os, shutil, subprocess, time

SKY = shutil.which('sky') or os.path.expanduser('~/tkf-mixdom/python/.venv/bin/sky')
YAML = 'aws/balibase_one_pair_supH.yaml'
SUFFIX = 'supH5r2'
CONCUR = 31

MISSING = [
    ('BB12014', 0, 5), ('BB12014', 1, 5), ('BB12014', 2, 5), ('BB12014', 3, 5), ('BB12014', 4, 5),
    ('BB20008', 0, 4), ('BB20008', 0, 5), ('BB20008', 0, 6), ('BB20008', 0, 7),
    ('BB20008', 1, 4), ('BB20008', 1, 5), ('BB20008', 1, 6), ('BB20008', 1, 7),
    ('BB20008', 2, 4), ('BB20008', 2, 5), ('BB20008', 2, 6), ('BB20008', 2, 7),
    ('BB20008', 3, 4), ('BB20008', 3, 5), ('BB20008', 3, 6), ('BB20008', 3, 7),
    ('BB20008', 4, 5), ('BB20008', 4, 6), ('BB20008', 4, 7),
    ('BB20008', 5, 6), ('BB20008', 5, 7), ('BB20008', 6, 7),
    ('BB30015', 0, 4), ('BB30015', 1, 4), ('BB30015', 2, 4), ('BB30015', 3, 4),
]

print(f'Launching {len(MISSING)} pairs, suffix={SUFFIX}')
for k, (fam, i, j) in enumerate(MISSING):
    name = f'balibase-direct-{fam.lower()}-{i}-{j}-{SUFFIX}'
    cmd = [SKY, 'launch', '-c', name,
           '--env', f'FAMILY={fam}', '--env', f'PAIR_I={i}', '--env', f'PAIR_J={j}',
           '--down', '--idle-minutes-to-autostop', '20', '-y', YAML]
    log = f'/tmp/sky_direct_{name}.log'
    with open(log, 'w') as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f'  [{k+1}/{len(MISSING)}] {fam} ({i},{j}) -> {name}')
    time.sleep(1.0)
print('all launched')
