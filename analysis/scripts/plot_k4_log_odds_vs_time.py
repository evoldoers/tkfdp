#!/usr/bin/env python3
"""K=4 emwarm checkpoint: 20x20 log_2 (joint / [marg_a * marg_b])
heat-maps at multiple branch lengths t.

Purpose
-------
The paper (main.tex line 1041) claims the stationary log-odds matrix
has a +3.64 log_2 odds C-C entry as the largest positive deviation
(disulfide-bridge signal). A previous analysis of the AA-marginal
Potts-coupling-induced boost M_solo (single_seq_edge_mcmc.py) flagged
M_solo[C, C] as the SMALLEST entry, prompting a contradiction.

This script computes the actual STATIONARY joint vs independent
log-odds at multiple t (where t = 0 is the paper's quantity, and t > 0
is the same quantity after both axes have evolved independently under
their per-class CTMC for branch length t).

Formula at t >= 0
-----------------
Let pi^{(c)}(a) = K=4 per-class amino-acid profile (pi_class[c, a]).
Let H_{cc'}(a, b) = atoms[assignments[c, c']](a, b), the Potts
coupling atom for the canonical class-pair {c, c'}.

Class-pair-conditional stationary joint at t = 0:

    Pi_{cc'}(a0, b0) = pi^{(c)}(a0) * pi^{(c')}(b0) * exp(-H_{cc'}(a0, b0)) / Z_{cc'}.

Under the per-class CTMC Q^{(c)} on each axis independently, the
class-conditional joint at time t is:

    Pi_{cc'}^{(t)}(a, b)
        = sum_{a0, b0} Pi_{cc'}(a0, b0) * P_t^{(c)}(a0 -> a) * P_t^{(c')}(b0 -> b).

Aggregated over the K^2 = 16 ordered class-pairs under a uniform prior:

    pi_joint^{(t)}(a, b) = (1/K^2) sum_{c, c'} Pi_{cc'}^{(t)}(a, b).

Reference independent baseline = product of UNIFORM-MIXTURE per-class
marginals (this matches plot_k4_effective_joint.py, which underlies
the paper claim):

    pi_marg(a) = (1/K) sum_c pi^{(c)}(a).

(This is time-invariant because per-class CTMCs are stationary at pi^{(c)};
in particular it is the marginal of pi_joint at t = +infinity, and also
the marginal of pi_joint at t = 0 *only* if the couplings H_{cc'} are
absent. With non-trivial H_{cc'}, marg-of-joint != pi_marg at t = 0, and
the difference is exactly the coupling-induced bias. Using pi_marg gives
the paper's log-odds quantity.)

The log-odds matrix at time t is

    log2 [ pi_joint^{(t)}(a, b) / (pi_marg(a) * pi_marg(b)) ]

which equals the paper's quantity at t = 0 and decays toward 0 as t grows.

Output
------
A 2x3 grid of 20x20 RdBu_r heat-maps in
   math-paper/figures/k4_log_odds_vs_time.{pdf,png}

Plus stdout: numerical value + rank of C-C among the 400 (a, b) cells
at each t.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
A = 20


def load_lg_alpha():
    """Return (Q_lg, pi_lg) in standard ACDE... order (NOT LG-PCA order).

    plot_aa_evolution.load_lg_in_lg_order() reindexes into LG_ORDER for
    its hue scheme; we want standard ACDE order to match AA_ORDER (the
    order used by plot_k4_effective_joint.py, which is the script
    underlying the paper claim).
    """
    sys.path.insert(0, os.path.expanduser("~/tkf-mixdom/python"))
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    Q_alpha, pi_alpha = rate_matrix_lg()
    return np.asarray(Q_alpha, dtype=np.float64), np.asarray(pi_alpha, dtype=np.float64)


def per_class_rate_matrix(Q_lg: np.ndarray, pi_lg: np.ndarray,
                           pi_class_c: np.ndarray) -> np.ndarray:
    """F81-style reweight so stationary becomes pi_class_c. Off-diagonals
    Q[a, b] = Q_lg[a, b] * pi_class_c[b] / pi_lg[b]; diagonal set so rows
    sum to 0. Preserves LG exchangeability S = Q / pi."""
    n = Q_lg.shape[0]
    Q_out = Q_lg.copy()
    safe_pi = np.where(pi_lg > 0, pi_lg, 1e-30)
    factor = pi_class_c / safe_pi
    Q_out *= factor[None, :]
    np.fill_diagonal(Q_out, 0.0)
    np.fill_diagonal(Q_out, -Q_out.sum(axis=1))
    return Q_out


def class_pair_joint_t0(pi_c: np.ndarray, pi_cp: np.ndarray,
                          H: np.ndarray) -> np.ndarray:
    """Class-pair-conditional stationary joint at t = 0:
       Pi[a, b] proportional to pi_c[a] * pi_cp[b] * exp(-H[a, b]).
    Normalised so that sum_{a, b} Pi[a, b] = 1.
    """
    log_pi = (np.log(pi_c + 1e-30)[:, None]
              + np.log(pi_cp + 1e-30)[None, :]
              - H)
    log_pi -= log_pi.max()
    p = np.exp(log_pi)
    p /= p.sum()
    return p


def evolve_joint_independent(Pi0: np.ndarray,
                              P_a: np.ndarray,
                              P_b: np.ndarray) -> np.ndarray:
    """Given a 2D joint Pi0[a0, b0] and per-axis substitution matrices
    P_a[a0, a] (axis 1) and P_b[b0, b] (axis 2), compute the joint at
    time t after both axes evolve independently:

        Pi_t[a, b] = sum_{a0, b0} Pi0[a0, b0] * P_a[a0, a] * P_b[b0, b].

    Vectorised via two einsums; output sums to 1 (within numerical tol).
    """
    # First evolve axis 1: Pi_t1[a, b0] = sum_{a0} Pi0[a0, b0] * P_a[a0, a]
    Pi_t1 = np.einsum("ij,ik->kj", Pi0, P_a, optimize=True)
    # Then evolve axis 2: Pi_t[a, b] = sum_{b0} Pi_t1[a, b0] * P_b[b0, b]
    Pi_t = np.einsum("kj,jl->kl", Pi_t1, P_b, optimize=True)
    return Pi_t


def compute_log_odds_at_t(pi_class: np.ndarray, atoms: np.ndarray,
                            assignments: np.ndarray,
                            P_a_per_c: np.ndarray, P_b_per_c: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate class-pair-evolved joint over the K^2 ordered class-pairs.

    Returns (joint_t, pi_marg, log_odds) where pi_marg is the uniform-
    mixture per-class marginal (the reference independence baseline used
    in the paper) and log_odds[a, b] = log_2 (joint[a, b] / (pi_marg[a] * pi_marg[b])).

    P_a_per_c: (K, A, A) per-class substitution matrix for axis 1 at time t.
    P_b_per_c: (K, A, A) per-class substitution matrix for axis 2 at time t
               (typically same content as P_a_per_c, but kept separate to
               match the symmetric class-pair indexing).
    """
    K = pi_class.shape[0]
    joint = np.zeros((A, A), dtype=np.float64)
    for c in range(K):
        for cp in range(K):
            atom_idx = int(assignments[c, cp])
            H = atoms[atom_idx]
            Pi0 = class_pair_joint_t0(pi_class[c], pi_class[cp], H)
            Pi_t = evolve_joint_independent(Pi0, P_a_per_c[c], P_b_per_c[cp])
            joint += Pi_t / (K * K)
    # Paper baseline: uniform-mixture per-class marginal (time-invariant
    # because per-class CTMC's stationary IS pi_class[c]).
    pi_marg = pi_class.mean(axis=0)
    indep = np.outer(pi_marg, pi_marg)
    indep /= indep.sum()
    # Also normalise joint (it already sums to 1 within numerical tol).
    joint = joint / joint.sum()
    log_odds = np.log2((joint + 1e-300) / (indep + 1e-300))
    return joint, pi_marg, log_odds


def make_figure(ts: list[float], log_odds_list: list[np.ndarray],
                  joint_list: list[np.ndarray], marg_list: list[np.ndarray],
                  out_dir: Path) -> Path:
    """Render a 2x3 grid of 20x20 RdBu_r heatmaps."""
    nrows, ncols = 2, 3
    assert len(ts) == nrows * ncols, f"need {nrows * ncols} t values, got {len(ts)}"

    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 10), constrained_layout=True)

    # Symmetric colour scale across all panels: anchored on the t=0 max so
    # the decay through time is visually obvious.
    vmax_global = max(float(np.abs(lo).max()) for lo in log_odds_list)

    for k, (t, lo) in enumerate(zip(ts, log_odds_list)):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        im = ax.imshow(lo, cmap="RdBu_r", vmin=-vmax_global, vmax=vmax_global,
                       aspect="equal")
        ax.set_xticks(range(A))
        ax.set_yticks(range(A))
        ax.set_xticklabels(AA_ORDER, fontsize=6, rotation=90)
        ax.set_yticklabels(AA_ORDER, fontsize=6)
        ax.set_title(rf"$t={t:g}$  ($\log_2$ joint/indep)", fontsize=10)
        # Annotate C-C value (AA_ORDER index 1 for C).
        c_idx = AA_ORDER.index("C")
        cc_val = lo[c_idx, c_idx]
        # Place text below cell, white on red / black on blue for legibility.
        text_color = "white" if abs(cc_val) > 0.5 * vmax_global else "black"
        ax.text(c_idx, c_idx, f"{cc_val:+.2f}", ha="center", va="center",
                color=text_color, fontsize=6, fontweight="bold")
        # Rank C-C among the 400 entries (1 = most positive).
        flat = lo.ravel()
        rank_desc = int(np.sum(flat > cc_val)) + 1  # number of entries strictly greater + 1
        rank_asc = int(np.sum(flat < cc_val)) + 1   # number strictly smaller + 1
        ax.set_xlabel(rf"AA $b$  (C-C rank: #{rank_desc} from top / #{rank_asc} from bottom of 400)",
                       fontsize=7)
        if c == 0:
            ax.set_ylabel("AA $a$  (standard ACDE...VWY order)", fontsize=8)

    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, location="right",
                      shrink=0.8)
    cb.set_label(r"$\log_2 \pi_{\mathrm{joint}}(a, b) / "
                  r"[\pi_{\mathrm{marg}}(a) \pi_{\mathrm{marg}}(b)]$",
                  fontsize=9)
    fig.suptitle(
        "K=4 emwarm checkpoint: log$_2$ (joint / independent) at "
        "varying branch length $t$\n"
        r"(K=4 site-class profiles + Potts atoms, uniform $K^2 = 16$ class-pair prior)",
        fontsize=11, y=1.02)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "k4_log_odds_vs_time.pdf"
    png = out_dir / "k4_log_odds_vs_time.png"
    fig.savefig(pdf, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                   default=str(Path(
                       "~/tkf-dp/results/K4-emwarm-top1000-2026-05-09/"
                       "_best_chkpt/state.npz").expanduser()))
    p.add_argument("--out-dir",
                   default=str(Path("~/tkf-dp/math-paper/figures").expanduser()))
    p.add_argument("--ts", default="0,0.1,0.3,1.0,3.0,10.0",
                   help="Comma-separated list of t values (must be 6 entries to fill 2x3 grid).")
    args = p.parse_args()

    ts = [float(x) for x in args.ts.split(",")]
    assert len(ts) == 6, "Need exactly 6 t values for the 2x3 panel."

    # Load checkpoint.
    d = np.load(args.ckpt, allow_pickle=True)
    pi_class = np.asarray(d["pi_class"], dtype=np.float64)         # (K, A) in ACDE order
    atoms = np.asarray(d["potts_atoms"], dtype=np.float64)         # (K_H, A, A)
    assignments = np.asarray(d["potts_assignments"], dtype=np.int64)  # (K, K)
    K = pi_class.shape[0]
    K_H = atoms.shape[0]
    print(f"Loaded checkpoint: K_c={K}, K_H={K_H}", flush=True)

    # Load LG rate matrix in alphabetical order to match AA_ORDER.
    Q_lg, pi_lg = load_lg_alpha()

    # Per-class rate matrices and substitution matrices P^{(c)}_t at each t.
    Q_per_c = np.stack([per_class_rate_matrix(Q_lg, pi_lg, pi_class[c])
                         for c in range(K)], axis=0)
    # Sanity check: pi_class[c] should be stationary of Q_per_c[c].
    for c in range(K):
        drift = float(np.linalg.norm(pi_class[c] @ Q_per_c[c]))
        assert drift < 1e-7, f"per_class_rate_matrix not stationary: c={c}, drift={drift}"
    print("Per-class CTMCs build OK (stationary at pi_class[c] within 1e-7).", flush=True)

    log_odds_list = []
    joint_list = []
    marg_list = []

    print("\n=== Log-odds matrix and C-C rank at each t ===\n", flush=True)
    for t in ts:
        # Per-class substitution matrices at time t.
        P_per_c = np.stack([expm(Q_per_c[c] * t) for c in range(K)], axis=0)
        joint, marg, log_odds = compute_log_odds_at_t(
            pi_class, atoms, assignments, P_per_c, P_per_c)
        log_odds_list.append(log_odds)
        joint_list.append(joint)
        marg_list.append(marg)

        # Diagnostic: joint-marg differs from pi_marg only because of H coupling.
        joint_marg = joint.sum(axis=1)
        marg_diff = float(np.linalg.norm(joint_marg - marg))
        print(f"t = {t:>6g}: ||sum_b joint(a, b) - pi_marg|| = {marg_diff:.2e} "
              f"(coupling-induced; should decay to 0 as t -> infty)")

        # C-C cell.
        c_idx = AA_ORDER.index("C")
        cc_val = log_odds[c_idx, c_idx]
        flat = [(log_odds[i, j], AA_ORDER[i], AA_ORDER[j])
                for i in range(A) for j in range(A)]
        flat_sorted_desc = sorted(flat, reverse=True)
        # Rank of C-C among all 400 entries (1 = most positive).
        rank = next(idx for idx, (_, a, b) in enumerate(flat_sorted_desc)
                     if a == "C" and b == "C") + 1
        print(f"          log_2(joint[C,C] / marg[C]^2) = {cc_val:+.4f}, "
              f"rank = {rank} / 400")
        print(f"          top 10 attractive:")
        for lo, a, b in flat_sorted_desc[:10]:
            print(f"              {a}-{b}: {lo:+.3f}")
        print(f"          bottom 10 repulsive:")
        for lo, a, b in flat_sorted_desc[-10:]:
            print(f"              {a}-{b}: {lo:+.3f}")
        print()

    # Verify C-C status across t.
    c_idx = AA_ORDER.index("C")
    cc_vals = [lo[c_idx, c_idx] for lo in log_odds_list]
    any_positive = any(v > 0 for v in cc_vals)
    is_top_at_any = False
    for k, lo in enumerate(log_odds_list):
        if lo[c_idx, c_idx] >= lo.max() - 1e-10:
            is_top_at_any = True
            print(f"C-C is THE MOST POSITIVE at t = {ts[k]}: value = "
                  f"{lo[c_idx, c_idx]:+.4f}", flush=True)
    if not is_top_at_any:
        print("C-C is NOT the single most-positive entry at any tested t", flush=True)
    print(f"C-C is positive at some tested t: {any_positive}", flush=True)

    # Render figure.
    out_dir = Path(args.out_dir)
    pdf = make_figure(ts, log_odds_list, joint_list, marg_list, out_dir)
    print(f"\nwrote {pdf}\n", flush=True)
    print(f"     {pdf.with_suffix('.png')}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
