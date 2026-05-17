#!/usr/bin/env bash
# Launch the TASK A step 6 production replica-exchange K=4 sweep.
#
# Preconditions:
#   * Top-rung validation passed (corpus Q' F1 within 0.02 of 0.450).
#   * GPU is free (top-rung complete).
#   * Released K=4 checkpoint present at
#     results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz.
#
# Output:
#   math-paper/results/infinite_phmm_balibase_k4_replicaexchange.json
#
# Pass CUDA_VISIBLE_DEVICES=<N> on the env line; defaults to 1.
#
# Estimated wall time: 5 rungs x ~17 s/pair x 187 pairs ~= 4.5 hours on
# a single RTX 2080 Ti at the L<150 subset.
set -e

REPO=$(cd "$(dirname "$0")"/../.. && pwd)
VENV=/home/yam/tkf-mixdom/python/.venv/bin/python
CKPT="$REPO/results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz"
OUT="$REPO/math-paper/results/infinite_phmm_balibase_k4_replicaexchange.json"
LOG="/tmp/k4_re_step6.log"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$VENV" \
    analysis/scripts/sweep_infinite_phmm_balibase.py \
    --checkpoint "$CKPT" \
    --alpha-z-ladder "100,500,1e3,1e4,1e6" \
    --swap-every 10 \
    --n-sweeps 500 --n-burnin 100 \
    --max-len 150 \
    --out "$OUT" \
    > "$LOG" 2>&1 &
PID=$!
echo "K=4 RE step 6 launched on GPU $GPU: PID $PID, log $LOG, out $OUT"
