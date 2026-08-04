#!/usr/bin/env python3
r"""Illustrative geometry of the lumpable coupled-telegraph (n=2 pair CTMC).

The two-state pair chain, after the rate gauge, has three parameters: marginal
m=P(x=1), stationary correlation cov=Cov(x,y)=g-h, and transition synchronization
D=g+h, where g,h are the correlated (00<->11) and swap (01<->10) double-flip fluxes.
At fixed m the reversible/exchangeable/lumpable + positive generators fill a compact
region of the (cov, D) plane -- a CONE (a triangle at m=1/2) whose:
  * apex (0,0) is the independent chain (Q0, the kernel-basis reference lift);
  * lower edges D=|cov| are h=0 (right) / g=0 (left): the minimum synchronization a
    correlation forces (a double-flip flux hits 0);
  * upper edge D=Dmax is a=b=0: single-flip positivity (max synchronization).
The 1-D lumpability KERNEL is the vertical direction (raise D at fixed cov), the
single basis element K = diag(pi)^{-1} s s^T, s=(+1,-1,-1,+1) the agreement parity.
A cluster-MIXTURE (each cluster one fixed class) pools LINEARLY in (cov,D): the fitted
single chain sits at the convex combination -- correlation (x) can cancel to 0 while
synchronization (y) survives. That is the coupling washout, in one picture.

Writes psb-paper/figures/telegraph_kernel_geometry.pdf (+ .png).
Reproducible: no randomness; edit the marked component points to iterate."""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

def feasible_D(m, cov):
    """[Dmin,Dmax] of synchronization D at fixed (m,cov); None if infeasible.
    pi00=(1-m)^2+cov, c=m^2+cov; g=(D+cov)/2,h=(D-cov)/2; a=m*pi00-g, b=(1-m)c-g."""
    c = m*m + cov; pi00 = (1-m)**2 + cov; pi01 = m - c
    if min(pi00, pi01, c) < 0: return None
    Dmin = abs(cov)
    # a>=0: (D+cov)/2 <= m*pi00 ; b>=0: (D+cov)/2 <= (1-m)*c
    Dmax = 2*min(m*pi00, (1-m)*c) - cov
    return (Dmin, Dmax) if Dmax >= Dmin - 1e-12 else None

def cone(m, npts=400):
    covs = np.linspace(-m*(1-m)+1e-6, m*(1-m)-1e-6, npts)
    lo, hi, xs = [], [], []
    for cv in covs:
        r = feasible_D(m, cv)
        if r: xs.append(cv); lo.append(r[0]); hi.append(r[1])
    return np.array(xs), np.array(lo), np.array(hi)

m = 0.5
xs, lo, hi = cone(m)
# closed polygon of the feasible region
polyx = np.concatenate([xs, xs[::-1]]); polyy = np.concatenate([hi, lo[::-1]])

fig, ax = plt.subplots(figsize=(7.2, 6.2))
ax.add_patch(Polygon(np.column_stack([polyx, polyy]), closed=True,
                     facecolor="#dfe9f5", edgecolor="none", zorder=0))
# constraint edges
ax.plot(xs, hi, color="#c1440e", lw=2.2, zorder=3)                       # top: a=b=0
ax.plot(xs[xs>=0], np.abs(xs[xs>=0]), color="#1f6f3f", lw=2.2, zorder=3) # right: h=0
ax.plot(xs[xs<=0], np.abs(xs[xs<=0]), color="#2a5fb0", lw=2.2, zorder=3) # left: g=0
# apex = independent chain Q0
ax.plot(0, 0, "o", color="black", ms=8, zorder=5)
ax.annotate("independent\nchain $Q_0$", (0, 0),
            xytext=(0.016, 0.018), textcoords="data", fontsize=9.5, va="bottom")
# kernel direction (vertical), drawn low in the LEFT (blue) region to keep the centre clear
kx = -0.075
ax.add_patch(FancyArrowPatch((kx, 0.09), (kx, 0.175),
             arrowstyle="-|>", mutation_scale=15, color="#5b2a86", lw=2, zorder=6))
ax.annotate("kernel $K$\n($\\uparrow D$, fixed cov)", (kx, 0.135),
            xytext=(kx-0.012, 0.135), fontsize=9, color="#5b2a86", va="center", ha="right")
# mixture of two components (each cluster one fixed class), symmetric -> pooled on axis
A = (0.155, 0.205); B = (-0.155, 0.205); w = 0.5
P = (w*A[0]+(1-w)*B[0], w*A[1]+(1-w)*B[1])
ax.plot([A[0], B[0]], [A[1], B[1]], "--", color="#444", lw=1.4, zorder=4)
ax.plot(*A, "o", color="#111", ms=7, zorder=6)
ax.annotate("A: agree\n(cov$>$0)", A, xytext=(A[0]+0.006, A[1]+0.008), fontsize=9.5, va="bottom")
ax.plot(*B, "o", color="#111", ms=7, zorder=6)
ax.annotate("B: disagree\n(cov$<$0)", B, xytext=(B[0]-0.006, B[1]+0.008), fontsize=9.5,
            ha="right", va="bottom")
ax.plot(*P, "X", color="#c1440e", ms=13, zorder=7)
ax.annotate("pooled single-chain fit\n(convex combination):\ncorrelation cancels,\n"
            "synchronization survives", P, xytext=(0.235, 0.15), fontsize=9.5, color="#c1440e",
            va="center", ha="left", arrowprops=dict(arrowstyle="->", color="#c1440e", lw=1.3))
ax.axvline(0, color="#999", ls=":", lw=1, zorder=1)
# edge labels
ax.annotate("single-flip positivity  $a{=}b{=}0$  (max synchronization)", (0, hi.max()),
            xytext=(0, hi.max()+0.013), fontsize=9.5, color="#c1440e", ha="center")
ax.annotate("$h{=}0$: only\ncorrelated flips", (0.12, 0.12),
            xytext=(0.16, 0.075), fontsize=9, color="#1f6f3f", ha="left")
ax.annotate("$g{=}0$:\nonly swaps", (-0.185, 0.185), xytext=(-0.235, 0.16),
            fontsize=9, color="#2a5fb0", ha="right")
ax.annotate(r"$D\geq|\mathrm{cov}|$: a correlation forces synchronization",
            (0, 0.0), xytext=(0, -0.052), fontsize=10, ha="center", style="italic")
ax.text(0.5, -0.205,
        r"double flips: $g=F(00{\leftrightarrow}11)$ correlated, $h=F(01{\leftrightarrow}10)$ swap;"
        "   " r"single flips $a,b$ (one site) $\Rightarrow$ top edge $a{=}b{=}0$;"
        "   uniform marginals $m{=}1/2$",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#333")

ax.set_xlabel(r"stationary correlation   $\mathrm{cov}=\mathrm{Cov}(x,y)=g-h$", fontsize=11)
ax.set_ylabel(r"transition synchronization   $D=g+h$", fontsize=11)
ax.set_title(r"Lumpable coupled telegraphs: feasible cone, kernel, and mixture washout ($m=1/2$)",
             fontsize=11.5)
ax.set_xlim(-0.30, 0.34); ax.set_ylim(-0.075, 0.30)
ax.set_aspect("equal"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
import os; os.makedirs("psb-paper/figures", exist_ok=True)
fig.savefig("psb-paper/figures/telegraph_kernel_geometry.pdf", bbox_inches="tight")
fig.savefig("psb-paper/figures/telegraph_kernel_geometry.png", dpi=150, bbox_inches="tight")
print("wrote psb-paper/figures/telegraph_kernel_geometry.{pdf,png}")
print(f"m={m}: cov range [{xs.min():.3f},{xs.max():.3f}]  Dmax={hi.max():.3f}  "
      f"apex(0,0); pooled={P}")
