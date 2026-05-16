#!/usr/bin/env python3
"""
Visualize bridge-conditioned CTMC trajectories of amino acids over time
using the LG08 substitution-rate matrix and a custom hue-encoded colour
scheme tied to the LG08 exchangeability ordering.

Colour scheme
-------------
We sort the 20 amino acids by the first nonzero PCA component of the LG08
exchangeability matrix; the resulting permutation runs roughly hydrophobic
to charged:

    LG08 ordering = "IVLMFCWYATPGSREHKQDN"

Position lookup gives ``idx(A) in [0, 19]``.

For a single residue ``A`` we use ``hue(A) = idx(A) / 20``.

For an aligned pair ``(A, B)`` we use the symmetric-but-visually-distinct
construction

    i, j = idx(A), idx(B)
    hue(A, B) = (10 * (i + j) + max(i, j) - min(i, j)) / 400

The pair construction collapses to ``hue(A, A) = idx(A) / 20`` along the
diagonal and varies smoothly over the off-diagonal, allowing the eye to
group co-conserved positions by hue.

Background convention
---------------------
We use a WHITE background by default (better for paper figures). For each
pixel of probability mass ``q`` we plot

    H = hue(state)
    S = q / q_max
    V = 1.0

so that high probability shows as a vivid hue and low probability fades
to white. ``--background black`` swaps to ``S = 1.0, V = q / q_max`` on a
black canvas.

Panels
------
We display four panels for the example transition (a0, b0) -> (aT, bT)
[default ``(A, W) -> (I, V)``] over time ``t in [0, T]`` [default
``T = 3``]:

  1. Marginal a0 -> aT      under the singleton LG08 CTMC, bridge-conditioned.
  2. Marginal b0 -> bT      under the singleton LG08 CTMC, bridge-conditioned.
  3. Joint (a0,b0)->(aT,bT) under the product-of-independent-marginals
                            approximation.
  4. Joint (a0,b0)->(aT,bT) under the exact 400-state coevolutionary CTMC
                            with a small Potts-style coupling that biases
                            the equilibrium toward states found in the LG08
                            exchangeability matrix's principal axis.

The 400-state coevolutionary rate matrix is built as

    Q[(i,j) -> (i',j)] = LG[i,i']            for i' != i
    Q[(i,j) -> (i,j')] = LG[j,j']            for j' != j
    Q[(i,j) -> (i',j')] = 0                  for i' != i and j' != j

(no simultaneous double substitutions). A Potts-style coupling is then
applied via Metropolis-style detailed-balance reweighting:

    H[i, j] = -coupling * (s_i * s_j)

where ``s_k`` is the (centred, normalised) projection of LG state ``k`` on
the LG-PCA axis. We then symmetrise the rate matrix so that its
equilibrium becomes ``pi[i] * pi[j] * exp(-H[i,j]) / Z``. Setting
``--coupling 0`` recovers the independent-sites case (panels 3 and 4
should then look identical, which is a useful sanity check).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.linalg import expm


# Suppress jax GPU init when it isn't needed.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


ALPHA_ORDER = "ACDEFGHIKLMNPQRSTVWY"          # standard alphabetic order
LG_ORDER = "IVLMFCWYATPGSREHKQDN"             # LG08-PCA-sorted order
ALPHA_TO_LG = np.array([LG_ORDER.index(a) for a in ALPHA_ORDER])
LG_TO_ALPHA = np.array([ALPHA_ORDER.index(a) for a in LG_ORDER])


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

def hue_single(idx_lg: int) -> float:
    """LG-ordered hue for a single residue.

    Maps the 20 residues into HSV hue range [0, 0.633] (red through
    blue), avoiding the wrap-around from violet back to red that
    would visually conflate the last AA with the first.
    """
    return idx_lg / 30.0


def hue_pair(i_lg: int, j_lg: int) -> float:
    """Joint hue of an aligned pair: a random uniform 50/50 coin flip
    between hue_single(i_lg) and hue_single(j_lg).

    This realises the exchangeability/reversibility symmetry visually:
    each AA-pair cell gets the colour of one of its two component
    residues at random, so a stacked-band plot of a joint distribution
    becomes a SPECKLED mixture over the two single-site hues -- you
    randomly see one component or the other at any point.

    The coin flip is deterministic in (i_lg, j_lg) via an md5 hash, so
    renders are reproducible across runs.
    """
    import hashlib
    h = hashlib.md5(f"{i_lg},{j_lg}".encode()).digest()[0]
    return hue_single(i_lg) if (h & 1) else hue_single(j_lg)


def aa_to_lg_idx(aa: str) -> int:
    """Look up the LG-ordering index for an amino-acid letter."""
    aa = aa.upper()
    if aa not in LG_ORDER:
        raise ValueError(f"Unknown amino acid {aa!r}; expected one of {LG_ORDER}")
    return LG_ORDER.index(aa)


# ---------------------------------------------------------------------------
# LG08 rate matrix (alphabetical order, then permuted to LG order)
# ---------------------------------------------------------------------------

def load_lg_in_lg_order() -> Tuple[np.ndarray, np.ndarray]:
    """Return (Q, pi) on the LG-sorted alphabet."""
    sys.path.insert(0, os.path.expanduser("~/tkf-mixdom/python"))
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    Q_alpha, pi_alpha = rate_matrix_lg()
    Q_alpha = np.asarray(Q_alpha, dtype=np.float64)
    pi_alpha = np.asarray(pi_alpha, dtype=np.float64)
    # Reindex alphabet (ACDE...) -> LG order (IVLM...).
    Q = Q_alpha[np.ix_(LG_TO_ALPHA, LG_TO_ALPHA)]
    pi = pi_alpha[LG_TO_ALPHA]
    return Q, pi


# ---------------------------------------------------------------------------
# Bridge posteriors
# ---------------------------------------------------------------------------

def _eigen_propagator(Q: np.ndarray):
    """Return a callable t -> e^{Q t} that uses ONE eigendecomposition for
    all t. For a reversible rate matrix this is much faster than calling
    scipy.linalg.expm at each requested t (which is the previous
    implementation's bottleneck when ts is a fine grid)."""
    d, V = np.linalg.eig(Q)
    Vinv = np.linalg.inv(V)
    def prop(t: float) -> np.ndarray:
        return (V * np.exp(d * t)[None, :]) @ Vinv
    return prop, d, V, Vinv


def bridge_marginal(Q: np.ndarray, a0: int, aT: int, ts: np.ndarray, T: float) -> np.ndarray:
    """Bridge-conditioned posterior q(k; t) for a CTMC with rate Q.

    q(k; t) = P[x(t)=k | x(0)=a0, x(T)=aT]
           = (e^{Q t})_{a0,k} (e^{Q (T - t)})_{k,aT} / (e^{Q T})_{a0,aT}

    Returns array of shape (n_states, len(ts)). Uses one eigendecomposition
    of Q and reuses it for every t on the grid (much faster than re-running
    scipy.linalg.expm at each t).
    """
    n = Q.shape[0]
    prop, d, V, Vinv = _eigen_propagator(Q)
    PT = prop(T)
    Z = float(np.real_if_close(PT[a0, aT]))
    if Z <= 0:
        raise ValueError(f"Endpoint probability is {Z}; transition unreachable")
    # Precompute row-a0 of V and column-aT of Vinv (each used at every t).
    V_a0 = V[a0, :]                                   # (n,) complex
    Vinv_aT = Vinv[:, aT]                              # (n,) complex
    # Per-t: alpha = (V * exp(d t)) @ Vinv at row a0 -> V_a0 * exp(d t) @ Vinv
    # Vectorise over the t-grid.
    expdt = np.exp(np.outer(d, ts))                    # (n, n_t) complex
    # alpha[k, t] = sum_l V[a0, l] * exp(d_l t) * Vinv[l, k]
    alpha = (V_a0[:, None] * expdt).T @ Vinv           # (n_t, n)
    # beta[k, t] = sum_l V[k, l] * exp(d_l (T - t)) * Vinv[l, aT]
    expdTmt = np.exp(np.outer(d, T - ts))              # (n, n_t)
    beta = V @ (expdTmt * Vinv_aT[:, None])            # (n, n_t)
    # alpha is (n_t, n); transpose to align.
    out = (alpha.T * beta) / Z                         # (n, n_t)
    return np.real_if_close(out, tol=1e6).astype(np.float64)


def bridge_joint_independent(
    Q: np.ndarray,
    a0: int, aT: int,
    b0: int, bT: int,
    ts: np.ndarray, T: float,
) -> np.ndarray:
    """Joint posterior under independent-sites approximation.

    Returns array of shape (n_states*n_states, len(ts)) in j-major order.
    """
    qa = bridge_marginal(Q, a0, aT, ts, T)     # (n, T)
    qb = bridge_marginal(Q, b0, bT, ts, T)     # (n, T)
    n = Q.shape[0]
    nt = ts.size
    out = np.empty((n * n, nt))
    for s in range(nt):
        out[:, s] = np.outer(qa[:, s], qb[:, s]).reshape(-1)
    return out


def build_coevol_rate_matrix(
    Q: np.ndarray,
    pi: np.ndarray,
    coupling: float,
) -> np.ndarray:
    """Construct a 400-state coevolutionary rate matrix on (i, j) pairs.

    Off-diagonal: a single-site substitution at position 1 OR position 2
    (no double substitutions). Coupling biases the equilibrium toward
    co-occurrence of LG-similar residues using a Potts-style energy

        H[i, j] = -coupling * (s_i * s_j)

    where s_k is the (centred, normalised) LG-position projection in
    [-1, +1]. The target equilibrium is

        pi_pair[i, j] proportional to pi[i] * pi[j] * exp(-H[i, j])

    and the off-diagonal rate is built from the LG exchangeability S
    (Q[i, i'] = S[i, i'] * pi[i'], where S is symmetric) by

        Q_co[(i,j) -> (i',j)] = S[i, i'] * pi[i'] * sqrt(w[i', j] / w[i, j])

    with w[i, j] = pi_pair[i, j] / (pi[i] * pi[j]) the deviation of the
    joint from the product equilibrium. This satisfies detailed balance
    with pi_pair, and reduces exactly to LG when coupling=0 (so panels 3
    and 4 match in that limit). The reverse-coordinate moves use the same
    construction with i and j roles swapped.
    """
    n = Q.shape[0]
    # Centred, normalised LG-axis projection s_k in [-1, +1].
    idx = np.arange(n, dtype=np.float64)
    s = 2.0 * (idx / (n - 1)) - 1.0
    H = -coupling * np.outer(s, s)              # (n, n)

    # Build target equilibrium pi_pair[i, j].
    log_pi_pair = np.log(pi)[:, None] + np.log(pi)[None, :] - H
    log_pi_pair -= log_pi_pair.max()
    pi_pair = np.exp(log_pi_pair)
    pi_pair /= pi_pair.sum()

    # w[i, j] = pi_pair[i, j] / (pi[i] pi[j]); equals 1 at coupling=0.
    w = pi_pair / np.outer(pi, pi)

    # LG exchangeability S[i, i'] = Q[i, i'] / pi[i'] for i != i'.
    # (Symmetric since pi[i] Q[i, i'] = pi[i'] Q[i', i].)
    S = np.zeros_like(Q)
    off = ~np.eye(n, dtype=bool)
    S[off] = Q[off] / np.broadcast_to(pi[None, :], Q.shape)[off]

    N = n * n
    Q_co = np.zeros((N, N))

    def state_idx(i: int, j: int) -> int:
        return i * n + j

    for i in range(n):
        for j in range(n):
            src = state_idx(i, j)
            sqrt_w_at_j = np.sqrt(w[:, j] / w[i, j])      # (n,)
            sqrt_w_at_i = np.sqrt(w[i, :] / w[i, j])      # (n,)
            # First-coordinate moves (i -> ip, j fixed).
            row1 = S[i, :] * pi * sqrt_w_at_j
            row1[i] = 0.0
            for ip in range(n):
                if ip == i:
                    continue
                Q_co[src, state_idx(ip, j)] = row1[ip]
            # Second-coordinate moves (j -> jp, i fixed).
            row2 = S[j, :] * pi * sqrt_w_at_i
            row2[j] = 0.0
            for jp in range(n):
                if jp == j:
                    continue
                Q_co[src, state_idx(i, jp)] = row2[jp]
    # Diagonal so rows sum to 0.
    np.fill_diagonal(Q_co, 0.0)
    np.fill_diagonal(Q_co, -Q_co.sum(axis=1))
    return Q_co


def bridge_joint_exact(
    Q: np.ndarray,
    pi: np.ndarray,
    a0: int, aT: int,
    b0: int, bT: int,
    ts: np.ndarray, T: float,
    coupling: float,
) -> np.ndarray:
    """Bridge-conditioned posterior under the 400-state coevolutionary CTMC.

    Uses one eigendecomposition of Q_co (a (n^2, n^2) matrix) reused for
    every t on the grid -- much faster than calling scipy.linalg.expm at
    each t.
    """
    Q_co = build_coevol_rate_matrix(Q, pi, coupling)
    n = Q.shape[0]
    src = a0 * n + b0
    dst = aT * n + bT
    prop, d, V, Vinv = _eigen_propagator(Q_co)
    PT = prop(T)
    Z = float(np.real_if_close(PT[src, dst]))
    if Z <= 0:
        raise ValueError(f"Joint endpoint probability is {Z}; transition unreachable")
    N = Q_co.shape[0]
    out = np.empty((N, ts.size))
    V_src = V[src, :]                                  # (N,) complex
    Vinv_dst = Vinv[:, dst]                            # (N,) complex
    for s_idx, t in enumerate(ts):
        alpha = (V_src * np.exp(d * t)) @ Vinv
        beta = V @ (np.exp(d * (T - t)) * Vinv_dst)
        out[:, s_idx] = alpha * beta / Z
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def density_to_rgb(
    density: np.ndarray,
    hues: np.ndarray,
    background: str,
    gamma: float = 0.4,
    per_column: bool = False,
) -> np.ndarray:
    """Convert a (n_states, nt) density and a per-row hue array to (n_states, nt, 3) RGB.

    The intensity channel is gamma-corrected to bring out the low-density bulk
    of the bridge posterior (otherwise the spikes at t=0 and t=T saturate the
    colour range). ``gamma < 1`` lifts low values; ``gamma == 1`` is linear.

    If ``per_column`` is True, the density is normalised per time column so the
    visible intensity reflects the relative composition at each time, not the
    absolute mass (this hides the time-dependent peakedness but exposes the
    minority states).
    """
    n, nt = density.shape
    if per_column:
        col_max = density.max(axis=0, keepdims=True)
        col_max = np.where(col_max > 0, col_max, 1.0)
        norm = density / col_max
    else:
        qmax = density.max() if density.max() > 0 else 1.0
        norm = density / qmax
    norm = np.clip(norm, 0.0, 1.0)
    intensity = np.power(norm, gamma)
    H = np.broadcast_to(hues[:, None], (n, nt))
    if background == "white":
        S = intensity
        V = np.ones_like(intensity)
        bg = np.array([1.0, 1.0, 1.0])
    elif background == "black":
        S = np.ones_like(intensity)
        V = intensity
        bg = np.array([0.0, 0.0, 0.0])
    else:
        raise ValueError(f"unknown background {background!r}")
    hsv = np.stack([H, S, V], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    # Mask out exactly-zero pixels with the background colour for clean look.
    zero = (density == 0.0)
    rgb[zero] = bg
    return rgb


def plot_panel(
    ax,
    density: np.ndarray,
    hues: np.ndarray,
    ts: np.ndarray,
    title: str,
    background: str,
    yticklabels: list,
    ytickpos: list,
    gamma: float = 0.4,
    per_column: bool = False,
):
    """Render one heatmap panel."""
    rgb = density_to_rgb(density, hues, background, gamma=gamma, per_column=per_column)
    n = density.shape[0]
    ax.imshow(
        rgb,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[ts[0], ts[-1], n - 0.5, -0.5],
    )
    ax.set_xlabel("time $t$")
    ax.set_title(title, fontsize=10)
    ax.set_yticks(ytickpos)
    ax.set_yticklabels(yticklabels, fontsize=8)
    if background == "black":
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")


def make_figure(
    a0: str, aT: str, b0: str, bT: str,
    T: float, n_t: int, coupling: float,
    background: str,
    out_pdf: str, out_png: str | None,
    gamma: float = 0.4,
    per_column: bool = False,
):
    Q, pi = load_lg_in_lg_order()
    n = Q.shape[0]
    ts = np.linspace(0.0, T, n_t)

    a0_i = aa_to_lg_idx(a0)
    aT_i = aa_to_lg_idx(aT)
    b0_i = aa_to_lg_idx(b0)
    bT_i = aa_to_lg_idx(bT)

    # Marginals.
    qa = bridge_marginal(Q, a0_i, aT_i, ts, T)
    qb = bridge_marginal(Q, b0_i, bT_i, ts, T)

    # Joint under independent sites (n^2 rows in j-major: row = 20*i + j).
    qjoint_ind = bridge_joint_independent(Q, a0_i, aT_i, b0_i, bT_i, ts, T)

    # Joint under exact coevolutionary CTMC.
    qjoint_exact = bridge_joint_exact(Q, pi, a0_i, aT_i, b0_i, bT_i, ts, T, coupling)

    # Hues.
    single_hues = np.array([hue_single(k) for k in range(n)])
    pair_hues = np.array([hue_pair(i, j) for i in range(n) for j in range(n)])

    # y-tick configuration.
    single_ticklabels = list(LG_ORDER)
    single_tickpos = list(range(n))
    # For 400-state plots, mark the four corner pairs of the bridge plus a
    # uniform grid of LG i-block starts (one tick per 5 i-rows).
    pair_marks_set = set([
        (a0_i, b0_i),
        (aT_i, bT_i),
        (a0_i, bT_i),
        (aT_i, b0_i),
    ])
    for ii in range(0, n, 5):
        pair_marks_set.add((ii, 0))
    # Sort, then drop ticks that would collide visually (within 6 rows).
    pair_marks_sorted = sorted(pair_marks_set, key=lambda ij: n * ij[0] + ij[1])
    pair_marks = []
    last_pos = -1e9
    for (i, j) in pair_marks_sorted:
        pos = n * i + j
        if pos - last_pos < 6:
            continue
        pair_marks.append((i, j))
        last_pos = pos
    pair_tickpos = [n * i + j for (i, j) in pair_marks]
    pair_ticklabels = [f"({LG_ORDER[i]},{LG_ORDER[j]})" for (i, j) in pair_marks]

    fig, axes = plt.subplots(
        4, 1, figsize=(9.5, 18.0),
        gridspec_kw={"height_ratios": [1.6, 1.6, 5.0, 5.0]},
    )

    plot_panel(
        axes[0], qa, single_hues, ts,
        f"Marginal {a0} → {aT}  (singleton LG08)",
        background, single_ticklabels, single_tickpos,
        gamma=gamma, per_column=per_column,
    )
    axes[0].set_ylabel("residue (LG order)")
    # Mark the start/end residues with bold horizontal lines.
    line_kw = dict(color="black" if background == "white" else "white",
                   alpha=0.45, lw=0.6)
    axes[0].axhline(a0_i, **line_kw)
    axes[0].axhline(aT_i, **line_kw)

    plot_panel(
        axes[1], qb, single_hues, ts,
        f"Marginal {b0} → {bT}  (singleton LG08)",
        background, single_ticklabels, single_tickpos,
        gamma=gamma, per_column=per_column,
    )
    axes[1].set_ylabel("residue (LG order)")
    axes[1].axhline(b0_i, **line_kw)
    axes[1].axhline(bT_i, **line_kw)

    plot_panel(
        axes[2], qjoint_ind, pair_hues, ts,
        f"Joint ({a0},{b0}) → ({aT},{bT})  "
        f"under independent-sites product",
        background, pair_ticklabels, pair_tickpos,
        gamma=gamma, per_column=per_column,
    )
    axes[2].set_ylabel("pair (LG order, $i$-major)")
    axes[2].axhline(n * a0_i + b0_i, **line_kw)
    axes[2].axhline(n * aT_i + bT_i, **line_kw)

    plot_panel(
        axes[3], qjoint_exact, pair_hues, ts,
        f"Joint ({a0},{b0}) → ({aT},{bT})  "
        f"under coevolutionary 400-state CTMC  (coupling={coupling:g})",
        background, pair_ticklabels, pair_tickpos,
        gamma=gamma, per_column=per_column,
    )
    axes[3].set_ylabel("pair (LG order, $i$-major)")
    axes[3].axhline(n * a0_i + b0_i, **line_kw)
    axes[3].axhline(n * aT_i + bT_i, **line_kw)

    if background == "black":
        fig.patch.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {out_pdf}")
    if out_png:
        fig.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
        print(f"wrote {out_png}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-pair", default="AW",
                   help="initial pair (a0, b0) as a 2-letter string (default AW)")
    p.add_argument("--to-pair", default="IV",
                   help="final pair (aT, bT) as a 2-letter string (default IV)")
    p.add_argument("--T", type=float, default=3.0, help="total bridge time (default 3.0)")
    p.add_argument("--n-t", type=int, default=200, help="number of time samples (default 200)")
    p.add_argument("--coupling", type=float, default=0.5,
                   help="Potts-style coupling strength on the LG axis "
                        "(0 -> identical to independent panel; default 0.5)")
    p.add_argument("--background", choices=["white", "black"], default="white",
                   help="figure background convention (default white)")
    p.add_argument("--gamma", type=float, default=0.4,
                   help="gamma exponent on intensity channel (smaller -> brighten "
                        "low-density bulk; default 0.4)")
    p.add_argument("--per-column", action="store_true",
                   help="normalise each time column independently (hides the "
                        "endpoint spike, exposes the relative composition over "
                        "time)")
    p.add_argument(
        "--out-pdf",
        default=os.path.expanduser("~/tkf-dp/math-paper/figures/aa_evolution_AW_to_IV.pdf"),
        help="output PDF path",
    )
    p.add_argument(
        "--out-png",
        default=os.path.expanduser("~/tkf-dp/math-paper/figures/aa_evolution_AW_to_IV.png"),
        help="output PNG path (set to '' to skip)",
    )
    args = p.parse_args(argv)

    if len(args.from_pair) != 2 or len(args.to_pair) != 2:
        p.error("--from-pair and --to-pair must be 2-letter amino-acid strings")

    a0, b0 = args.from_pair.upper()
    aT, bT = args.to_pair.upper()
    out_png = args.out_png or None

    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)

    make_figure(
        a0=a0, aT=aT, b0=b0, bT=bT,
        T=args.T, n_t=args.n_t, coupling=args.coupling,
        background=args.background,
        out_pdf=args.out_pdf, out_png=out_png,
        gamma=args.gamma, per_column=args.per_column,
    )


if __name__ == "__main__":
    main()
