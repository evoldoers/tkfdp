#!/bin/bash
cd /home/yam/tkf-dp
export PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p results/pair_models/gammaI logs
for K in 1 2 4 8; do
  nice -n 15 python3 experiments/fit_coupling_mixture_rateI.py --K $K --gamma-cats 4 \
      --out results/pair_models/gammaI/gammaI_K$K.json > logs/gammaI_K$K.log 2>&1 &
done
wait
echo "### ALL GAMMA+I DONE"
python3 - <<'PY'
import json, glob
base={1:-2.6719,2:-2.6531,4:-2.6270,8:-2.6021}  # single-rate free-S baseline (agent-confirmed)
print(f"{'K':>2} {'single':>9} {'gammaI':>9} {'gain':>7} {'alpha':>6} {'p_inv':>6} {'rateCV':>7} {'cls<->rate NMI':>14} {'MI(pi_c)w':>10}")
for f in sorted(glob.glob("results/pair_models/gammaI/gammaI_K*.json"), key=lambda p:int(p.split('_K')[1].split('.')[0])):
    d=json.load(open(f)); K=d['K']; v=d['val_per_count_ll']; b=base.get(K,float('nan'))
    print(f"{K:>2} {b:>9.4f} {v:>9.4f} {v-b:>+7.4f} {d['alpha']:>6.3f} {d['p_inv']:>6.3f} {d['rate_cv']:>7.3f} {d['class_rate_nmi']:>14.4f} {d['mi_weighted']:>10.4f}")
PY
