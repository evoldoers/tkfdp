#!/bin/bash
# Durable driver for the K=8 mixture-component characterization (task #38).
# Runs each fit to completion in ONE process (huge wall-budget => no chunk-stall),
# resuming K=4 from its per-iter checkpoint, then the report (types + lumpability + md).
cd /home/yam/tkf-dp || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
LOG=logs/charA_driver.log
echo "=== driver start $(date) ===" >> "$LOG"
for K in 4 8; do
  echo "--- fit K=$K $(date) ---" >> "$LOG"
  python3 experiments/characterize_mixture_components.py fit --K "$K" \
      --em-iters 150 --wall-budget 9999999 >> "$LOG" 2>&1
done
echo "--- report $(date) ---" >> "$LOG"
python3 experiments/characterize_mixture_components.py report --Ks 4 8 >> "$LOG" 2>&1
echo "=== driver done $(date) ===" >> "$LOG"
