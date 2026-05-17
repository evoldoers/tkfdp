#!/usr/bin/env python3
"""K=4 atom version of the bridge-evolution plots.

Replaces the toy LG-axis Potts coupling in ``plot_aa_evolution.py``'s
``bridge_joint_exact`` with the **actual** Potts atom from a TKF-DP
checkpoint, and uses the per-class amino-acid profiles for the
single-site CTMC marginals (so the substitution-side dynamics are
class-consistent).

Five canonical examples (each plot is one PDF + PNG):

  Label              From    To     Atom h  Class-pair  Story
  CC_to_CC           (C,C)   (C,C)  2       (0, 2)      Disulfide preservation
  CH_to_CH           (C,H)   (C,H)  1       (0, 1)      Zinc-finger preservation
  DR_to_DR           (D,R)   (D,R)  3       (0, 3)      Salt-bridge preservation
  PD_to_GY           (P,D)   (G,Y)  5       (1, 2)      Beta-turn-motif coevolution (4 unique AAs!)
  AW_to_IV_negative  (A,W)   (I,V)  5       (1, 2)      Negative control

Run ``--batch`` to render all five at once; or ``--label foo --from XY --to UV
--atom h --class-pair c,c'`` to render a single custom transition.

Reproducible recipe:

    cd ~/tkf-dp
    python analysis/scripts/plot_aa_evolution_k4.py --batch

This regenerates the 5 figures at
``math-paper/figures/aa_evolution_<label>.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.linalg import expm

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_aa_evolution import (
    ALPHA_ORDER, LG_ORDER, ALPHA_TO_LG, LG_TO_ALPHA,
    hue_single, hue_pair, aa_to_lg_idx, load_lg_in_lg_order,
    bridge_marginal, bridge_joint_independent,
    density_to_rgb, plot_panel,
)


REPO = HERE.parent.parent
CHECKPOINT = (REPO / "results" / "K4-emwarm-top1000-2026-05-09"
              / "_best_chkpt" / "state.npz")
OUT_DIR = REPO / "math-paper" / "figures"


# ---------------------------------------------------------------------------
# K=4 checkpoint loading
# ---------------------------------------------------------------------------

def load_k4(checkpoint_path: Path = CHECKPOINT) -> dict:
    """Return a dict of the K=4 atoms and per-class profiles, REINDEXED to
    LG ordering for compatibility with plot_aa_evolution's hue scheme."""
    d = np.load(str(checkpoint_path), allow_pickle=True)
    pi_class_alpha = np.asarray(d["pi_class"], dtype=np.float64)   # (K_c, 20)
    atoms_alpha = np.asarray(d["potts_atoms"], dtype=np.float64)   # (K_H, 20, 20)
    assignments = np.asarray(d["potts_assignments"], dtype=np.int64)  # (K_c, K_c)
    # Reindex amino-acid axes to LG ordering.
    pi_class = pi_class_alpha[:, LG_TO_ALPHA]
    atoms = atoms_alpha[:, LG_TO_ALPHA, :][:, :, LG_TO_ALPHA]
    return {
        "pi_class": pi_class,
        "atoms": atoms,
        "assignments": assignments,
        "K_c": int(pi_class.shape[0]),
        "K_H": int(atoms.shape[0]),
    }


# ---------------------------------------------------------------------------
# Per-class single-site CTMC
# ---------------------------------------------------------------------------

def per_class_rate_matrix(Q_lg: np.ndarray, pi_lg: np.ndarray,
                           pi_class_c: np.ndarray) -> np.ndarray:
    """Felsenstein-81-style reweight of the LG rate matrix so the
    stationary distribution becomes ``pi_class_c`` instead of LG's pi.

    Off-diagonal: ``Q_class[a, b] = Q_lg[a, b] * pi_class_c[b] / pi_lg[b]``.
    Diagonal: set so each row sums to 0.

    This preserves the LG exchangeability (S = Q / pi remains the same
    symmetric matrix) but tilts the stationary toward the per-class
    profile, which is what the K=4 model's substitution dynamics
    actually use.
    """
    n = Q_lg.shape[0]
    Q_out = Q_lg.copy()
    safe_pi = np.where(pi_lg > 0, pi_lg, 1e-30)
    factor = pi_class_c / safe_pi                  # (n,)
    Q_out *= factor[None, :]                       # rescale columns
    # Re-set diagonal: row sums to zero.
    np.fill_diagonal(Q_out, 0.0)
    np.fill_diagonal(Q_out, -Q_out.sum(axis=1))
    return Q_out


# ---------------------------------------------------------------------------
# K=4 coevolutionary 400-state rate matrix
# ---------------------------------------------------------------------------

def build_k4_coevol_rate_matrix(
        Q_a: np.ndarray, Q_b: np.ndarray,
        pi_a: np.ndarray, pi_b: np.ndarray,
        H: np.ndarray) -> np.ndarray:
    """400-state coevolutionary rate matrix on (i, j) pairs whose
    stationary is

        pi_pair[i, j] ∝ pi_a[i] * pi_b[j] * exp(-H[i, j])

    and that reduces exactly to the product of (Q_a, Q_b) when H is
    constant.

    Construction (detailed-balance correction): for first-coordinate moves
    (i -> i', j fixed) we use

        Q_co[(i,j) -> (i',j)] = S_a[i, i'] * pi_a[i']
                                * sqrt(w[i', j] / w[i, j])

    where S_a = Q_a / pi_a is the per-class exchangeability (symmetric
    in the F81 case) and w[i, j] = pi_pair[i, j] / (pi_a[i] * pi_b[j]).

    The reverse-coordinate moves use the same formula with axes swapped.
    Detailed balance is satisfied with pi_pair.
    """
    n = Q_a.shape[0]
    # Joint equilibrium.
    log_pi_pair = (np.log(pi_a + 1e-30)[:, None]
                   + np.log(pi_b + 1e-30)[None, :]
                   - H)
    log_pi_pair -= log_pi_pair.max()
    pi_pair = np.exp(log_pi_pair)
    pi_pair /= pi_pair.sum()
    w = pi_pair / (np.outer(pi_a, pi_b) + 1e-300)
    # Per-class exchangeabilities (S = Q / pi; symmetric for F81).
    safe_pi_a = np.where(pi_a > 0, pi_a, 1e-30)
    safe_pi_b = np.where(pi_b > 0, pi_b, 1e-30)
    S_a = np.zeros_like(Q_a)
    S_b = np.zeros_like(Q_b)
    off = ~np.eye(n, dtype=bool)
    S_a[off] = Q_a[off] / np.broadcast_to(pi_a[None, :], Q_a.shape)[off]
    S_b[off] = Q_b[off] / np.broadcast_to(pi_b[None, :], Q_b.shape)[off]
    N = n * n
    Q_co = np.zeros((N, N))
    def idx(i, j): return i * n + j
    for i in range(n):
        for j in range(n):
            src = idx(i, j)
            sqrt_w_at_j = np.sqrt(w[:, j] / w[i, j])
            sqrt_w_at_i = np.sqrt(w[i, :] / w[i, j])
            row1 = S_a[i, :] * pi_a * sqrt_w_at_j
            row1[i] = 0.0
            for ip in range(n):
                if ip == i: continue
                Q_co[src, idx(ip, j)] = row1[ip]
            row2 = S_b[j, :] * pi_b * sqrt_w_at_i
            row2[j] = 0.0
            for jp in range(n):
                if jp == j: continue
                Q_co[src, idx(i, jp)] = row2[jp]
    np.fill_diagonal(Q_co, 0.0)
    np.fill_diagonal(Q_co, -Q_co.sum(axis=1))
    return Q_co


def bridge_joint_k4(Q_co: np.ndarray, n: int,
                     a0: int, aT: int, b0: int, bT: int,
                     ts: np.ndarray, T: float) -> np.ndarray:
    """Bridge-conditioned posterior on the joint 400-state CTMC.

    Uses one eigendecomposition of Q_co for all t (10-100x faster than
    per-t scipy.linalg.expm calls)."""
    src = a0 * n + b0
    dst = aT * n + bT
    d, V = np.linalg.eig(Q_co)
    Vinv = np.linalg.inv(V)
    PT = (V * np.exp(d * T)[None, :]) @ Vinv
    Z = float(np.real_if_close(PT[src, dst]))
    if Z <= 0:
        raise ValueError(f"Joint endpoint probability is {Z}; transition unreachable")
    N = Q_co.shape[0]
    out = np.empty((N, ts.size))
    V_src = V[src, :]
    Vinv_dst = Vinv[:, dst]
    for s_idx, t in enumerate(ts):
        alpha = (V_src * np.exp(d * t)) @ Vinv
        beta = V @ (np.exp(d * (T - t)) * Vinv_dst)
        out[:, s_idx] = np.real_if_close(alpha * beta / Z, tol=1e6)
    return out


# ---------------------------------------------------------------------------
# K=4 figure builder
# ---------------------------------------------------------------------------

CANONICAL_EXAMPLES = [
    # label, from-pair, to-pair, atom_h, class-pair (c, cp), T, story
    ("CC_to_CC", "CC", "CC", 2, (0, 2), 2.0,
        "Disulfide preservation"),
    ("CH_to_CH", "CH", "CH", 1, (0, 1), 2.0,
        "Zinc-finger preservation"),
    ("DR_to_DR", "DR", "DR", 3, (0, 3), 2.0,
        "Salt-bridge preservation"),
    ("PD_to_GY", "PD", "GY", 5, (1, 2), 2.0,
        r"$\beta$-turn-motif coevolution"),
    ("AW_to_IV_negative", "AW", "IV", 5, (1, 2), 2.0,
        "Negative control: K=4 bridge $\\approx$ independent"),
]


def make_k4_figure(label: str, from_pair: str, to_pair: str,
                    atom_h: int, class_pair: Tuple[int, int],
                    T: float = 2.0, n_t: int = 200,
                    background: str = "white",
                    gamma: float = 0.4,
                    per_column: bool = False,
                    out_dir: Path = OUT_DIR,
                    k4: dict | None = None,
                    story: str = "") -> Path:
    """Render one 4-panel K=4 figure. Returns the PDF path."""
    if k4 is None:
        k4 = load_k4()
    Q_lg, pi_lg = load_lg_in_lg_order()
    n = Q_lg.shape[0]
    ts = np.linspace(0.0, T, n_t)

    a0 = aa_to_lg_idx(from_pair[0]); aT = aa_to_lg_idx(to_pair[0])
    b0 = aa_to_lg_idx(from_pair[1]); bT = aa_to_lg_idx(to_pair[1])

    c, cp = class_pair
    pi_class = k4["pi_class"]
    H = k4["atoms"][atom_h]

    # Per-class single-site rate matrices.
    Q_a = per_class_rate_matrix(Q_lg, pi_lg, pi_class[c])
    Q_b = per_class_rate_matrix(Q_lg, pi_lg, pi_class[cp])

    # Marginals (per-class CTMC).
    qa = bridge_marginal(Q_a, a0, aT, ts, T)
    qb = bridge_marginal(Q_b, b0, bT, ts, T)

    # Joint independent: outer product of per-class marginals.
    qjoint_ind = np.empty((n * n, ts.size))
    for s in range(ts.size):
        qjoint_ind[:, s] = np.outer(qa[:, s], qb[:, s]).reshape(-1)

    # K=4 joint.
    Q_co = build_k4_coevol_rate_matrix(Q_a, Q_b, pi_class[c], pi_class[cp], H)
    qjoint_k4 = bridge_joint_k4(Q_co, n, a0, aT, b0, bT, ts, T)

    # Sanity: K=4 joint marginals integrate to the per-class single-site
    # marginals only when H is constant; with non-trivial H the joint is
    # genuinely coupled. Print a numerical diagnostic.
    mid = n_t // 2
    cell_ind = qjoint_ind.reshape(n, n, n_t)[aT, bT, mid]
    cell_k4 = qjoint_k4.reshape(n, n, n_t)[aT, bT, mid]
    print(f"  [{label}] at t=T/2: P(endpoint cell) "
          f"indep={cell_ind:.4f}, K=4={cell_k4:.4f}, "
          f"ratio={cell_k4 / max(cell_ind, 1e-30):.2f}x")

    # Hues.
    single_hues = np.array([hue_single(k) for k in range(n)])
    pair_hues = np.array([hue_pair(i, j) for i in range(n) for j in range(n)])

    single_ticklabels = list(LG_ORDER)
    single_tickpos = list(range(n))
    pair_marks = {(a0, b0), (aT, bT), (a0, bT), (aT, b0)}
    for ii in range(0, n, 5):
        pair_marks.add((ii, 0))
    pair_ticks = sorted({i * n + j for (i, j) in pair_marks})
    pair_ticklabels = [f"{LG_ORDER[k // n]}{LG_ORDER[k % n]}" for k in pair_ticks]

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 16))
    fig.suptitle(
        f"({from_pair[0]},{from_pair[1]}) "
        rf"$\to$ ({to_pair[0]},{to_pair[1]}) "
        f"under K=4 atom {atom_h} (class-pair {class_pair}) — {story}",
        fontsize=11)
    plot_panel(axes[0], qa, single_hues, ts,
                f"Marginal {from_pair[0]} $\\to$ {to_pair[0]} "
                f"(per-class {c} CTMC)",
                background, single_ticklabels, single_tickpos,
                gamma=gamma, per_column=per_column)
    axes[0].set_ylabel("state $k$ (LG-ordered)")
    plot_panel(axes[1], qb, single_hues, ts,
                f"Marginal {from_pair[1]} $\\to$ {to_pair[1]} "
                f"(per-class {cp} CTMC)",
                background, single_ticklabels, single_tickpos,
                gamma=gamma, per_column=per_column)
    axes[1].set_ylabel("state $k$ (LG-ordered)")
    plot_panel(axes[2], qjoint_ind, pair_hues, ts,
                "Joint product of per-class marginals (independent sites)",
                background, pair_ticklabels, pair_ticks,
                gamma=gamma, per_column=per_column)
    axes[2].set_ylabel("pair state $(j, k)$ (i-major)")
    plot_panel(axes[3], qjoint_k4, pair_hues, ts,
                f"Joint under K=4 coevolution (atom {atom_h})",
                background, pair_ticklabels, pair_ticks,
                gamma=gamma, per_column=per_column)
    axes[3].set_ylabel("pair state $(j, k)$ (i-major)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"aa_evolution_{label}.pdf"
    png = out_dir / f"aa_evolution_{label}.png"
    fig.savefig(pdf, dpi=150, bbox_inches="tight",
                 facecolor="white" if background == "white" else "black")
    fig.savefig(png, dpi=120, bbox_inches="tight",
                 facecolor="white" if background == "white" else "black")
    plt.close(fig)
    print(f"  wrote {pdf}")
    return pdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", action="store_true",
                   help="Render all 5 canonical examples.")
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--label", default=None)
    p.add_argument("--from-pair", default=None, dest="from_pair",
                   help="2-letter from-pair, e.g. CC")
    p.add_argument("--to-pair", default=None, dest="to_pair",
                   help="2-letter to-pair, e.g. CC")
    p.add_argument("--atom", type=int, default=None)
    p.add_argument("--class-pair", default=None, dest="class_pair",
                   help="c,c' as comma-separated ints, e.g. 0,2")
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--n-t", type=int, default=200)
    p.add_argument("--background", choices=["white", "black"], default="white")
    p.add_argument("--gamma", type=float, default=0.4)
    p.add_argument("--per-column", action="store_true")
    args = p.parse_args()

    k4 = load_k4(Path(args.checkpoint))

    if args.batch:
        for label, fp, tp, atom_h, cls, T, story in CANONICAL_EXAMPLES:
            make_k4_figure(label, fp, tp, atom_h, cls,
                           T=T, n_t=args.n_t,
                           background=args.background,
                           gamma=args.gamma, per_column=args.per_column,
                           k4=k4, story=story)
        return 0

    if not (args.label and args.from_pair and args.to_pair
            and args.atom is not None and args.class_pair):
        p.error("Either --batch, or all of --label/--from-pair/--to-pair/"
                "--atom/--class-pair must be given")
    c, cp = (int(x) for x in args.class_pair.split(","))
    make_k4_figure(args.label, args.from_pair, args.to_pair,
                   args.atom, (c, cp),
                   T=args.T, n_t=args.n_t,
                   background=args.background,
                   gamma=args.gamma, per_column=args.per_column,
                   k4=k4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
