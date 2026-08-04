#!/bin/bash
cd /home/yam/tkf-dp
export PYTHONPATH=src:$PYTHONPATH JAX_PLATFORMS=cpu
log(){ echo "[affull-chain $(date +%T)] $*"; }
# 1. wait for the already-running val AF download
while pgrep -f "build_af_partition.py --split val" >/dev/null; do sleep 60; done
log "val AF partitions: $(ls data/pdb_af_partition_val/*.npz 2>/dev/null | wc -l)"
# 2. test AF download (sequential, throttled)
python3 experiments/build_af_partition.py --split test --out-dir data/pdb_af_partition_test --throttle 0.25 >> logs/af_partition_test.log 2>&1
log "test AF partitions: $(ls data/pdb_af_partition_test/*.npz 2>/dev/null | wc -l)"
# 3. extract Pfam-full sub-alignments
python3 experiments/extract_pfam_full_sub.py --af-dir data/pdb_af_partition_val  --out data/pfam_full_sub_val  >> logs/extract_val.log 2>&1
python3 experiments/extract_pfam_full_sub.py --af-dir data/pdb_af_partition_test --out data/pfam_full_sub_test >> logs/extract_test.log 2>&1
log "extract done (val=$(ls data/pfam_full_sub_val/*.npz 2>/dev/null|wc -l) test=$(ls data/pfam_full_sub_test/*.npz 2>/dev/null|wc -l))"
# 4. count builds (val/test x norate/rate) + merge
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for split in val test; do
  for suf in "" "_rate"; do
    flag=""; [ "$suf" = "_rate" ] && flag="--rate-het"
    out=data/cherry_counts_af_full_${split}${suf}
    python3 experiments/build_cherry_counts_af_full.py --sub-dir data/pfam_full_sub_$split --af-dir data/pdb_af_partition_$split --out $out $flag >> logs/count_${split}${suf}.log 2>&1
    python3 experiments/build_cherry_counts_af_full.py --out $out --merge >> logs/count_${split}${suf}.log 2>&1
    log "built $out"
  done
done
log "ALL AF_FULL VAL/TEST DONE"
