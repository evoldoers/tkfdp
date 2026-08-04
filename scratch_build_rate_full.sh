#!/bin/bash
cd /home/yam/tkf-dp
export PYTHONPATH=src:$PYTHONPATH OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
build_corpus () {
  local script=$1 out=$2
  echo "[chain] building $out ($(date +%T))"
  for s in 0 1 2 3 4; do
    nice -n 12 python3 $script --out $out --nshards 5 --shard $s --rate-het \
      > logs/$(basename $out)_shard$s.log 2>&1 &
  done
  wait
  nice -n 12 python3 $script --out $out --merge >> logs/$(basename $out)_merge.log 2>&1
  echo "[chain] $out DONE ($(date +%T))"
}
build_corpus experiments/build_cherry_counts_trrosetta.py data/cherry_counts_trrosetta_rate
build_corpus experiments/build_cherry_counts_af_full.py data/cherry_counts_af_full_rate
echo "[chain] ALL BUILDS DONE"
