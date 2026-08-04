#!/bin/bash
cd /home/yam/tkf-dp
export PYTHONPATH=src:$PYTHONPATH OMP_NUM_THREADS=2
while pgrep -f "build_cherry_counts_trrosetta.py --out data/cherry_counts_trrosetta_rate --nshards" >/dev/null; do sleep 20; done
python3 experiments/build_cherry_counts_trrosetta.py --out data/cherry_counts_trrosetta_rate --merge > logs/tr_rate_merge.log 2>&1
echo "MERGED $(date +%T)" >> logs/tr_rate_merge.log
