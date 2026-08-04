#!/bin/bash
# 6 niced, single-threaded shards over 6000 interleaved families -> per-contact corpus.
cd /home/yam/tkf-dp
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=data/per_contact_trrosetta
mkdir -p $OUT logs
for s in 0 1 2 3 4 5; do
  if [ -f "$OUT/part_$s.npz" ]; then echo "skip shard $s (exists)"; continue; fi
  nice -n 15 python3 experiments/build_per_contact_corpus.py \
      --out $OUT --nshards 6 --shard $s --max-fam 1000 --max-seqs 256 \
      > logs/pc_build_shard_$s.log 2>&1 &
done
wait
echo "### ALL SHARDS DONE"
python3 experiments/build_per_contact_corpus.py --out $OUT --merge >> logs/pc_build_merge.log 2>&1
echo "### MERGED"
cat logs/pc_build_merge.log
