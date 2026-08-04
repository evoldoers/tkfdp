#!/bin/bash
# Asym swap-pair + Gamma+I: verification gates then the two matched real fits.
set +e
cd ~/tkf-dp || exit 1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
O=results/mixture_asym_rateI; mkdir -p "$O"
PY="nice -n 10 python3 experiments/fit_coupling_mixture_asym_rateI.py"

echo "===== GATE A: single-rate twoP=4 (must reproduce AS asym_2P4 val -2.63379) $(date +%H:%M) ====="
$PY --single-rate --twoP 4 --em-iters 60 --seed 0 --out "$O/gateA_single_2P4.json" > logs/asym_ri_gateA.log 2>&1
python3 -c "import json,sys; g=json.load(open('$O/gateA_single_2P4.json')); d=abs(g['val_per_count_ll']+2.6337948752245937); print('GATE A val=%.6f diff=%.2e -> %s'%(g['val_per_count_ll'],d,'PASS' if d<1e-4 else 'FAIL')); sys.exit(0 if d<1e-4 else 1)"
if [ $? -ne 0 ]; then echo "GATE A FAILED -- aborting before the real fits"; exit 1; fi

echo "===== GATE B (force-sym twoP=4 -> sym K=2) + REAL twoP=4 + REAL twoP=8, parallel $(date +%H:%M) ====="
$PY --force-symmetric --twoP 4 --gamma-cats 4 --em-iters 100 --seed 0 --out "$O/gateB_forcesym_2P4.json" > logs/asym_ri_gateB.log 2>&1 &
$PY --twoP 4 --gamma-cats 4 --em-iters 100 --seed 0 --out "$O/asym_gammaI_2P4.json" > logs/asym_ri_real4.log 2>&1 &
$PY --twoP 8 --gamma-cats 4 --em-iters 100 --seed 0 --out "$O/asym_gammaI_2P8.json" > logs/asym_ri_real8.log 2>&1 &
wait
echo "===== ALL ASYM-RATEI FITS DONE $(date +%H:%M) ====="
