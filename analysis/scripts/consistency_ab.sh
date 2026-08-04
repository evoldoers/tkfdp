#!/bin/bash
# Consistency A/B: reuse cached pairwise posteriors (NO DP/FB/MCMC rerun),
# apply ProbCons consistency transform (iters 0=baseline,1,2) -> FSA -> SP/TC.
# 1 FSA seed (relative comparison only) for speed. Verbose per-method logging.
set -u
cd ~/tkf-dp
BALI=~/bio-datasets/data/balibase/bali3pdbm
CSV=/tmp/cons_ab.csv
echo "method,iters,SP,TC,optacc_F1,n" > "$CSV"
run() {  # $1=method $2=key
  local m="$1" key="$2"
  for it in 0 1 2; do
    local out="/tmp/ab_${m}_it${it}.json"
    echo "[$(date +%H:%M:%S)] RUN $m iters=$it ..."
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 PYTHONPATH=src timeout 2400 \
      python3 -u analysis/scripts/downstream_fsa_on_cached_qprime.py \
      --method "$m" --params-key "$key" --balibase-dir "$BALI" \
      --consistency-iters "$it" --fsa-n-seeds 1 --out "$out" \
      > "/tmp/ab_${m}_it${it}.log" 2>&1
    python3 - "$m" "$it" "$out" "$CSV" <<'PY'
import json, sys
m, it, out, csv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    c = json.load(open(out))['corpus']; n = c.get('msa_score_count', 0) or 1
    sp = round(c.get('gap1_msa_sp_sum', 0)/n, 4)
    tc = round(c.get('gap1_msa_tc_sum', 0)/n, 4)
    den = c.get('opt_acc_total_mass', 0) + c.get('gold_total', 0)
    f1 = round(2*c.get('opt_acc_e_tp', 0)/den, 4) if den else 0.0
    line = f"{m},{it},{sp},{tc},{f1},{c.get('msa_score_count',0)}"
except Exception as e:
    line = f"{m},{it},ERR,ERR,ERR,0"
open(csv, 'a').write(line + "\n")
print("  ->", line)
PY
  done
}
run infinite_phmm_mcmc_mixfragF2_K8_RE 18fb2fd3f2970765
run infinite_phmm_mcmc_K4_pdbanchor_RE 00183db783ec786d
run infinite_phmm_mcmc_K8_coupled_RE 269ba4e021784f97
run tkf92_K8_tkfdp e40f57eaa3b3f06a
echo "=== A/B DONE ==="
column -s, -t "$CSV"
