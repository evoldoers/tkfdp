#!/usr/bin/env python3
r"""Figure: the Metropolis-mixture components' coupling maps.

For each fitted component c (symmetric joint stationary pi_c over the 400 amino-acid
pairs, from results/mixture_component_char/components_K{K}.npz), plot the coupling

    M_c(a,b) = log[ pi_c(a,b) / (rho_a rho_b) ]      rho = marginal of pi_c

as a 20x20 heat map, with the amino acids ordered by biophysical group
(acidic | basic | polar | special | aliphatic | aromatic) so like-with-like and
complementary blocks are visible. Red = favored pair (positive coupling), blue =
disfavored. Components are sorted by mixture weight; each panel is annotated with
its weight w and stationary mutual information MI(pi_c).

Run: PYTHONPATH=src python3 experiments/plot_mixture_components.py --K 8 \
        --out psb-paper/figures/mixture_components_K8.pdf
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

AA = "ACDEFGHIKLMNPQRSTVWY"
# biophysical grouping: acidic | basic | polar | special | aliphatic | aromatic
GROUPS = [("acidic", "DE"), ("basic", "KRH"), ("polar", "NQST"),
          ("special", "CGP"), ("aliphatic", "AVLIM"), ("aromatic", "FYW")]
ORDER = "".join(g[1] for g in GROUPS)
assert sorted(ORDER) == sorted(AA), ORDER
PERM = np.array([AA.index(a) for a in ORDER])          # ORDER[i] = AA[PERM[i]]
# group boundaries (cumulative sizes), for separator lines
BOUND = np.cumsum([len(g[1]) for g in GROUPS])[:-1]


def coupling_map(pi):
    P = np.asarray(pi, float).reshape(20, 20)
    P = 0.5 * (P + P.T)
    P = P / P.sum()
    rho = P.sum(1)
    M = np.log(np.maximum(P, 1e-300) / np.maximum(np.outer(rho, rho), 1e-300))
    return M[np.ix_(PERM, PERM)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--dir", default="results/mixture_component_char")
    ap.add_argument("--out", default="psb-paper/figures/mixture_components_K8.pdf")
    ap.add_argument("--vmax", type=float, default=1.5)
    ap.add_argument("--ncols", type=int, default=4)
    a = ap.parse_args()

    z = np.load(f"{a.dir}/components_K{a.K}.npz", allow_pickle=True)
    pis = np.asarray(z["pis"], float)
    w = np.asarray(z["weights"], float)
    mi = np.asarray(z["mi_pi"], float)
    order = np.argsort(w)[::-1]

    ncols = a.ncols
    nrows = int(np.ceil(a.K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.7 * ncols + 0.6, 1.7 * nrows + 0.3))
    axes = np.atleast_2d(axes)
    labels = list(ORDER)

    im = None
    for panel, c in enumerate(order):
        ax = axes[panel // ncols, panel % ncols]
        M = coupling_map(pis[c])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-a.vmax, vmax=a.vmax,
                       interpolation="nearest")
        for b in BOUND:                                  # group separators
            ax.axhline(b - 0.5, color="0.5", lw=0.3)
            ax.axvline(b - 0.5, color="0.5", lw=0.3)
        ax.set_title(f"$w={w[c]:.2f}$   MI$={mi[c]:.3f}$", fontsize=7, pad=3)
        ax.set_xticks(range(20)); ax.set_yticks(range(20))
        ax.set_xticklabels(labels, fontsize=3.2)
        ax.set_yticklabels(labels, fontsize=3.2)
        ax.tick_params(length=0, pad=1)
        for s in ax.spines.values():
            s.set_linewidth(0.4)

    for panel in range(a.K, nrows * ncols):              # blank any spare cells
        axes[panel // ncols, panel % ncols].axis("off")

    fig.subplots_adjust(left=0.045, right=0.9, top=0.93, bottom=0.05,
                        wspace=0.28, hspace=0.32)
    cax = fig.add_axes([0.915, 0.12, 0.016, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"coupling $\log[\pi_c(a,b)/\rho_a\rho_b]$", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}  ({nrows}x{ncols} panels, K={a.K})")


if __name__ == "__main__":
    main()
