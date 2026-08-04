#!/usr/bin/env python3
r"""Interpret the coupling-mixture components' joint stationary pi_c by CORRESPONDENCE
ANALYSIS: the principal axes of correlation/anticorrelation between the two contacting
residues, and what biophysical property each axis tracks.

For each component c (symmetric joint pi_c over 20x20 AA pairs, marginal rho):
  R(a,b) = (pi_c(a,b) - rho_a rho_b) / sqrt(rho_a rho_b)          (symmetric)
  R = sum_k sigma_k phi_k phi_k^T ;  AA axis coordinate f_k(a) = phi_k(a)/sqrt(rho_a).
The total inertia sum_k sigma_k^2 = chi^2/N measures coupling strength (monotone with MI).
Each axis k is one independent "dimension" the component couples on:
  * sign(sigma_k) > 0  => residues with the SAME-sign coordinate co-occur MORE than chance
                          (CORRELATED / assortative on that axis)
  * sign(sigma_k) < 0  => OPPOSITE-sign coordinates co-occur (ANTICORRELATED / complementary)
We name the axis by the biophysical scale its coordinate f_k best matches (rho-weighted
Pearson r over the 20 AAs): charge, hydropathy (Kyte-Doolittle), volume, aromaticity.

Run: PYTHONPATH=src python3 experiments/interpret_mixture_coupling.py --K 4 8 [--out FILE.md]
"""
from __future__ import annotations
import argparse
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AA)}
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
          L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2,
           I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8,
           Y=193.6, V=140.0)
ARO = {a: (1.0 if a in "FWYH" else 0.0) for a in AA}
SCALES = {
    "charge": np.array([CHARGE.get(a, 0.0) for a in AA]),
    "hydropathy": np.array([KD[a] for a in AA]),
    "volume": np.array([VOL[a] for a in AA]),
    "aromatic": np.array([ARO[a] for a in AA]),
}


def wpearson(x, y, w):
    w = w / w.sum()
    mx = (w * x).sum(); my = (w * y).sum()
    cxy = (w * (x - mx) * (y - my)).sum()
    vx = (w * (x - mx) ** 2).sum(); vy = (w * (y - my) ** 2).sum()
    return cxy / max(np.sqrt(vx * vy), 1e-30)


def name_axis(f, rho):
    """Best-matching biophysical scale for AA coordinate f (rho-weighted |Pearson r|)."""
    best = max(SCALES, key=lambda k: abs(wpearson(f, SCALES[k], rho)))
    return best, wpearson(f, SCALES[best], rho)


def extremes(f, rho, n=5):
    """Most extreme + and - AAs on this axis (by |coordinate|, ignoring vanishing rho)."""
    order = np.argsort(f)
    neg = [AA[i] for i in order if rho[i] > 1e-4][:n]
    pos = [AA[i] for i in order[::-1] if rho[i] > 1e-4][:n]
    return pos, neg


def analyze_component(pi, n_axes=3):
    P = np.asarray(pi, float).reshape(20, 20); P = 0.5 * (P + P.T); P = P / P.sum()
    rho = P.sum(1)
    sr = np.sqrt(np.maximum(rho, 1e-30))
    Rmat = (P - np.outer(rho, rho)) / np.outer(sr, sr)
    lam, V = np.linalg.eigh(Rmat)                          # symmetric
    order = np.argsort(np.abs(lam))[::-1]
    inertia = float((lam ** 2).sum())
    axes = []
    for k in order[:n_axes]:
        sig = float(lam[k])
        f = V[:, k] / sr                                   # AA coordinate
        f = f * np.sign(f[np.argmax(np.abs(f))])           # sign gauge: largest loading +
        nm, r = name_axis(f, rho)
        pos, neg = extremes(f, rho)
        axes.append(dict(sigma=sig, frac=float(lam[k] ** 2 / max(inertia, 1e-30)),
                         property=nm, r=float(r), pos=pos, neg=neg))
    return dict(inertia=inertia, rho=rho, axes=axes)


def washout_decomposition(pis, w):
    """Law-of-total-covariance split of the POOLED pair correlation on each biophysical
    axis into within-class (genuine per-contact coupling that survives class-averaging,
    sign-aware) + between-class (composition heterogeneity across classes, >=0).  All
    standardized to the pooled marginal (var 1), so pooled = within + between exactly.
    Returns [(axis, strength, within, between, pooled, per_class_cov)]."""
    P = np.array([0.5 * (Q + Q.T) / Q.sum() for Q in np.asarray(pis, float).reshape(-1, 20, 20)])
    w = np.asarray(w, float); w = w / w.sum()
    Pagg = np.tensordot(w, P, axes=(0, 0)); rho_g = Pagg.sum(1)
    rows = []
    for name, v in SCALES.items():
        mu = (rho_g * v).sum(); sd = np.sqrt((rho_g * (v - mu) ** 2).sum())
        vt = (v - mu) / max(sd, 1e-30)
        raw = np.array([float((Q * np.outer(vt, vt)).sum()) for Q in P])   # E[XY|c]
        mc = np.array([float((Q.sum(1) * vt).sum()) for Q in P])           # E[X|c]
        covc = raw - mc ** 2                                               # within-class cov
        within = float((w * covc).sum()); between = float((w * mc ** 2).sum())
        pooled = float((Pagg * np.outer(vt, vt)).sum())
        strength = float((w * np.abs(covc)).sum())
        rows.append((name, strength, within, between, pooled, covc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--dir", default="results/mixture_component_char")
    ap.add_argument("--n-axes", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = []

    def emit(s=""):
        print(s); lines.append(s)

    emit("# Coupling-mixture components: axes of correlation/anticorrelation")
    emit("\nCorrespondence analysis of each fitted joint stationary pi_c "
         "(results/mixture_component_char/components_K*.npz). "
         "sigma>0 = correlated (like-with-like), sigma<0 = anticorrelated (complementary). "
         "`frac` = share of this component's total coupling inertia on the axis; "
         "`r` = rho-weighted Pearson of the axis coordinate vs the named scale.")
    for K in a.K:
        z = np.load(f"{a.dir}/components_K{K}.npz", allow_pickle=True)
        pis = np.asarray(z["pis"], float); w = np.asarray(z["weights"], float)
        mi = np.asarray(z["mi_pi"], float)
        order = np.argsort(w)[::-1]
        emit(f"\n## K = {K}\n")
        emit("### Washout decomposition (what survives pooling the components)\n")
        emit("Pooled pair correlation on each axis = within-class (genuine per-contact "
             "coupling, sign-aware) + between-class (composition heterogeneity, >=0), "
             "standardized to the pooled marginal. `strength` = mean per-class |coupling|.")
        emit("")
        emit("| axis | per-class strength | within (survives) | between (composition) | pooled | per-class signs |")
        emit("|---|---:|---:|---:|---:|---|")
        for name, strength, within, between, pooled, covc in washout_decomposition(pis[order], w[order]):
            signs = "".join("-" if x < -0.005 else ("+" if x > 0.005 else "0") for x in covc)
            emit(f"| {name} | {strength:.3f} | {within:+.3f} | {between:+.3f} | {pooled:+.3f} | {signs} |")
        emit("")
        for rank, c in enumerate(order):
            res = analyze_component(pis[c], a.n_axes)
            emit(f"### component {rank} (w={w[c]:.3f}, MI={mi[c]:.3f}, "
                 f"inertia={res['inertia']:.4f})")
            emit("")
            emit("| axis | sigma | frac | property (r) | + pole | - pole | reading |")
            emit("|---:|---:|---:|---|---|---|---|")
            for ax in res["axes"]:
                sign = "corr" if ax["sigma"] > 0 else "ANTIcorr"
                if ax["sigma"] > 0:
                    reading = f"like {ax['property']} together"
                else:
                    reading = f"opposite {ax['property']} paired"
                emit(f"| {ax['property']} | {ax['sigma']:+.4f} | {ax['frac']:.2f} | "
                     f"{ax['property']} ({ax['r']:+.2f}) | {''.join(ax['pos'])} | "
                     f"{''.join(ax['neg'])} | {sign}: {reading} |")
            emit("")
    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")
        print(f"\n# wrote {a.out}")


if __name__ == "__main__":
    main()
