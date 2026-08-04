#!/bin/bash
# Chain the two-pass Cohn evaluation: Pass-1 (dense, single-GPU-process) then
# Pass-2 (parallel full-psi). Pass-1 frees the GPU before Pass-2's worker pool starts.
set +e
cd ~/tkf-dp || exit 1
export JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 PYTHONPATH=src:experiments:src/tkfdp/cohn_ctbn

echo "===== PASS 1 (dense) start $(date '+%F %H:%M') ====="
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda python3 experiments/cohn_pass1.py
echo "===== PASS 1 done $(date '+%F %H:%M') ====="

echo "===== PASS 2 (parallel full-psi) start $(date '+%F %H:%M') ====="
python3 experiments/cohn_pass2_parallel.py --workers 6 --budget-hours 5.5 --n2-max 400 --save-every 200
echo "===== PASS 2 done $(date '+%F %H:%M') ====="
echo "===== ALL DONE $(date '+%F %H:%M') ====="
