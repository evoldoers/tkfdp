#!/bin/bash
cd /home/yam/tkf-dp
export PYTHONPATH=src OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
mkdir -p results/pair_models/mixture
for K in 1 2 3 4 6 8 12 16; do
  echo "===== K=$K ====="
  nice -n 15 python3 experiments/fit_coupling_mixture.py --K $K --em-iters 20 --inner 2 \
      --min-counts 10 --seed 0 --out results/pair_models/mixture/mixture_K$K.json
done
echo "### SWEEP DONE"
python3 - <<'PY'
import json, glob
rows=[]
for f in sorted(glob.glob("results/pair_models/mixture/mixture_K*.json"), key=lambda p:int(p.split("_K")[1].split(".")[0])):
    d=json.load(open(f)); rows.append(d)
print(f"{'K':>3} {'VAL/count':>10} {'train':>10} {'w-mean MI(pi_c)':>16} {'MI(pi_c) sorted'}")
for d in rows:
    print(f"{d['K']:>3} {d['val_per_count_ll']:>10.4f} {d['train_per_count_ll']:>10.4f} "
          f"{d['mi_weighted']:>16.4f} {[round(x,3) for x in d['mi_pi'][:6]]}")
PY
