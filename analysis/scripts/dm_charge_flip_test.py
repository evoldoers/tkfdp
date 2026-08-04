"""Charge-flip test on a fitted DM mixture: under the DM, draw a couple {c_i,c_j}
(a 2-draw) and ask P(the two classes are ANTIPOLAR) -- i.e. classify_pair ==
'salt_bridge' (opposite polarity dq_i*dq_j<0, the compensatory charge flip),
using the exact codebase definition (dynfield_metrics.classify_pair).

For a DM component h with class marginal p_h(c)=alpha_h[c]/A_h, a 2-draw couple
is ~ p_h(c_i)p_h(c_j) (independent for large A), so P(type T | h) = p_h^T M_T p_h
for the 0/1 type-mask M_T over the 400x400 class grid. Mixture: sum_h pi_h P(T|h)
(both classes come from the SAME drawn component). Compared against the uniform
null p(c)=1/K_c (== the flat/untrained DM).

  python analysis/scripts/dm_charge_flip_test.py --dm results/dm_fit_pdb_train/dm.npz
"""
from __future__ import annotations
import argparse, json
import numpy as np

import sys, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")           # CPU only; don't touch the GPUs
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "src"); sys.path.insert(0, "experiments"); sys.path.insert(0, "analysis/scripts")
import precompute_pairing as PP
from fit_pdb_hyperparams import load_pairs
from dynfield_metrics import archetype_charge_cys, classify_pair

TYPES = ["salt_bridge", "coflip", "disulfide", "static"]


def type_masks(charge, cys, K_a, K_c, tau=0.15, cys_t=0.15):
    """0/1 (K_c,K_c) mask per couple type via the exact classify_pair rule."""
    M = {t: np.zeros((K_c, K_c)) for t in TYPES}
    for ci in range(K_c):
        for cj in range(K_c):
            M[classify_pair(ci, cj, charge, cys, K_a, tau=tau, cys_t=cys_t)][ci, cj] = 1.0
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm", default="results/dm_fit_pdb_train/dm.npz")
    ap.add_argument("--tau", type=float, default=0.15)
    args = ap.parse_args()

    # pi_archetype (fixed LG-C20) -> per-archetype charge + cys
    byfam = load_pairs({"saltbridge"}, max_pairs=2, split="train")
    ds, _, _ = PP.build_enum400_ds(list(byfam.keys())[:1])
    pi_arch = np.asarray(ds.state.pi_archetype, float)
    K_a = pi_arch.shape[0]; K_c = K_a * K_a
    charge, cys = archetype_charge_cys(pi_arch)
    print(f"# K_a={K_a} archetypes; charge range [{charge.min():+.2f},{charge.max():+.2f}]  "
          f"acidic(<-tau): {np.where(charge<-args.tau)[0].tolist()}  basic(>tau): {np.where(charge>args.tau)[0].tolist()}")

    M = type_masks(charge, cys, K_a, K_c, tau=args.tau)
    frac_grid = {t: float(M[t].mean()) for t in TYPES}       # fraction of all class-pairs of each type

    d = np.load(args.dm)
    alpha = np.asarray(d["dm_alpha"], float); pi = np.asarray(d["dm_pi"], float)
    H = alpha.shape[0]
    p = alpha / alpha.sum(1, keepdims=True)                   # (H,K_c) component class marginals

    def type_probs(pvec):
        return {t: float(pvec @ M[t] @ pvec) for t in TYPES}

    uni = np.full(K_c, 1.0 / K_c)
    P_uni = type_probs(uni)
    # mixture: couple drawn from one component ~ pi
    P_mix = {t: float(sum(pi[h] * (p[h] @ M[t] @ p[h]) for h in range(H))) for t in TYPES}

    def line(tag, P):
        return (f"# {tag:16s} " +
                "  ".join(f"{t}={P[t]:.3f}" for t in TYPES) +
                f"   [antipolar enrich vs uniform = {P['salt_bridge']/max(P_uni['salt_bridge'],1e-9):.1f}x]")

    print(f"\n# fraction of the 400x400 class-grid that is each type (tau={args.tau}):")
    print("# " + "  ".join(f"{t}={frac_grid[t]:.3f}" for t in TYPES))
    print(f"\n# P(couple type) under a 2-draw:")
    print(line("uniform/flat DM", P_uni))
    print(line("fitted DM (mix)", P_mix))
    print(f"\n# per-component P(salt_bridge / antipolar):")
    for h in np.argsort(-pi):
        if pi[h] < 1e-3:
            continue
        Ph = type_probs(p[h])
        print(f"#   comp{h} pi={pi[h]:.3f}  salt_bridge={Ph['salt_bridge']:.3f}  "
              f"coflip={Ph['coflip']:.3f}  static={Ph['static']:.3f}  disulfide={Ph['disulfide']:.3f}")

    print(f"\n# SUMMARY: fitted DM draws an antipolar (salt-bridge) couple with prob "
          f"{P_mix['salt_bridge']:.3f} vs {P_uni['salt_bridge']:.3f} uniform "
          f"({P_mix['salt_bridge']/max(P_uni['salt_bridge'],1e-9):.1f}x enrichment).")


if __name__ == "__main__":
    main()
