"""Pack a trained K-class TKF-DP checkpoint into a tkf92_mixture-format NPZ.

The resulting model is the no-coupling singlet ablation: per-site emission
is a mixture over the trained site classes (using empirical class
proportions and per-class equilibria), with anchor TKF92 indel kinetics
and shared LG08 exchangeability. The Potts H matrix is dropped.

Default paths target the K=8 top-8000 release checkpoint; override via
CLI for other K_c values or other anchor TKF92 params.

Usage:
    python analysis/scripts/build_k8_mixture_npz.py \\
        [--ckpt results/_preserved/K8_KH1_top8000_iter68/state.npz] \\
        [--anchor ~/tkf-mixdom/python/experiments/tkf92_fitted_params.json] \\
        [--out /tmp/tkf92_mixture_K8_tkfdp.npz]

Then feed the NPZ into tkf-mixdom's standard tkf92_mixture pipeline:
    cd ~/tkf-mixdom/python
    JAX_ENABLE_X64=1 python experiments/expected_pairwise_balibase.py \\
        --method tkf92_mixture --method-name tkf92_K8_tkfdp \\
        --anchor experiments/tkf92_fitted_params.json \\
        --params /tmp/tkf92_mixture_K8_tkfdp.npz \\
        --pair-chunk 8 --fsa-sps \\
        --balibase-dir /home/yam/bio-datasets/data/balibase/bench1.0/bali3pdbm \\
        --out /tmp/expected_balibase_tkf92_K8_tkfdp.json

The resulting JSON is directly comparable to the other expected_balibase
rows in Table 1 (TKF92-K=20, CherryML-C=20), isolating the contribution
of multi-class emission from the Potts coupling.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--ckpt',
        default='results/_preserved/K8_KH1_top8000_iter68/state.npz',
        help='Trained K-class TKF-DP state.npz (any K_c).')
    p.add_argument(
        '--anchor',
        default=str(Path.home() / 'tkf-mixdom' / 'python' / 'experiments'
                    / 'tkf92_fitted_params.json'),
        help='Anchor TKF92 params JSON (ins/del/ext rates).')
    p.add_argument(
        '--out', default='/tmp/tkf92_mixture_K8_tkfdp.npz',
        help='Output NPZ in tkf92_mixture format.')
    p.add_argument(
        '--tkfmixdom-python',
        default=str(Path.home() / 'tkf-mixdom' / 'python'),
        help='Path to tkf-mixdom/python (for tkfmixdom imports).')
    args = p.parse_args()

    sys.path.insert(0, args.tkfmixdom_python)
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    from tkfmixdom.jax.core.ctmc import build_Q_from_S_pi
    import jax.numpy as jnp

    data = np.load(args.ckpt, allow_pickle=True)
    pi_class = np.asarray(data['pi_class'], dtype=np.float64)
    K_c, A = pi_class.shape

    total = np.zeros(K_c, dtype=np.int64)
    for k in data.files:
        if re.match(r'cls_\d+$', k):
            total += np.bincount(data[k].astype(int), minlength=K_c)
    rho = total / total.sum() if total.sum() > 0 else np.ones(K_c) / K_c
    print(f'K_c={K_c}, A={A}')
    print(f'empirical rho = {rho.round(4).tolist()}  sum={rho.sum():.6f}')

    with open(args.anchor) as f:
        anchor = json.load(f)
    ins, dele, ext = anchor['ins_rate'], anchor['del_rate'], anchor['ext_rate']
    print(f'anchor TKF92: ins={ins} del={dele} ext={ext}')

    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg = np.asarray(Q_lg); pi_lg = np.asarray(pi_lg)

    # Recover symmetric LG08 exchangeability S from F81-form Q_lg.
    S_lg = np.zeros((A, A))
    for i in range(A):
        for j in range(A):
            if i != j and pi_lg[j] > 0:
                S_lg[i, j] = Q_lg[i, j] / pi_lg[j]
    S_lg = 0.5 * (S_lg + S_lg.T)

    dom_Qs = np.zeros((K_c, A, A))
    dom_S = np.zeros((K_c, A, A))
    for c in range(K_c):
        pi_c = pi_class[c]
        Q_c = np.asarray(build_Q_from_S_pi(jnp.asarray(S_lg), jnp.asarray(pi_c)))
        dom_Qs[c] = Q_c
        dom_S[c] = S_lg

    out = {
        'main_ins': np.float32(ins),
        'main_del': np.float32(dele),
        'dom_ins': np.full(K_c, ins, dtype=np.float32),
        'dom_del': np.full(K_c, dele, dtype=np.float32),
        'dom_weights': rho.astype(np.float32),
        'frag_weights': np.ones((K_c, 1), dtype=np.float32),
        'ext_rates': np.full((K_c, 1, 1), ext, dtype=np.float32),
        'dom_Qs': dom_Qs.astype(np.float32),
        'dom_pis': pi_class.astype(np.float32),
        'dom_S_exch': dom_S.astype(np.float32),
        'em_iter': np.int32(0),
        '_config': np.array(json.dumps({
            'source_ckpt': args.ckpt,
            'anchor_json': args.anchor,
            'note': ('No-coupling singlet ablation. Per-site emission is '
                     'a mixture over the trained TKF-DP site classes with '
                     'anchor TKF92 indel rates and shared LG08 exchangeability. '
                     'Drops the Potts H coupling so we can isolate the '
                     'contribution of multi-class emission alone.'),
            'rho': rho.tolist(),
            'anchor_ins': ins,
            'anchor_del': dele,
            'anchor_ext': ext,
        })),
    }
    np.savez(args.out, **out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
