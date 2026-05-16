"""Compact reporter for a trained Potts/site-class state.

Two entrypoints:

  format_params_summary(pi_class, potts_atoms, top_aa=5, top_pairs=10) -> str
      Pure-string formatter, no I/O. Used in the SVI loop to print a
      compact snapshot every validation step.

  inspect_param_file(path, ...)
      CLI: load pi_class.npy + potts_atoms.npy from a results/_chkpt dir,
      print the formatted summary.

Format (top_aa=5, top_pairs=10, K_c=2, K_H=1):

    pi_class:
      class 0: G:0.10  L:0.09  A:0.08  K:0.06  R:0.06   (entropy 2.81 bits)
      class 1: ...
    Potts atoms (K_H=1, ||H|| = 15.40):
      atom 0:
        diagonal favored:  RR:-3.53  CC:-2.27  KK:-1.72
        off-diag favored:  YV:-2.29  AQ:-2.12  NG:-2.00  GK:-1.93  TW:-1.71  EM:-1.49  GS:-1.43  QP:-1.37  QW:-1.36  NW:-1.34
        off-diag penalized (largest H>0): IV:+1.22  RW:+0.95  ER:+0.96  ...
        atom -> class-pairs: [(0,0), (0,1), (1,1)]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


AA = list("ARNDCQEGHILKMFPSTWYV")


def _entropy_bits(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None); p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _topk_aa(pi: np.ndarray, k: int = 5) -> str:
    idx = np.argsort(pi)[::-1][:k]
    return "  ".join(f"{AA[i]}:{pi[i]:.2f}" for i in idx)


def _topk_pairs_offdiag(H: np.ndarray, k: int = 10, sign: str = "neg") -> str:
    """Top-k off-diagonal pairs of H. sign='neg' for most-favored
    (most-negative), sign='pos' for most-penalized (most-positive)."""
    pairs = [(AA[i], AA[j], H[i, j]) for i in range(20) for j in range(i + 1, 20)]
    if sign == "neg":
        pairs.sort(key=lambda t: t[2])
    else:
        pairs.sort(key=lambda t: -t[2])
    return "  ".join(f"{a}{b}:{h:+.2f}" for a, b, h in pairs[:k])


def _topk_diag(H: np.ndarray, k: int = 5) -> str:
    diag = [(AA[i], H[i, i]) for i in range(20)]
    diag.sort(key=lambda t: t[1])
    return "  ".join(f"{a}{a}:{h:+.2f}" for a, h in diag[:k])


def _atom_class_pairs(assignments: np.ndarray, atom_idx: int) -> list[tuple[int, int]]:
    """Return list of (c, c') unordered pairs assigned to this atom.
    `assignments` is the (K_c, K_c) symmetric int matrix from PottsDPState."""
    K_c = assignments.shape[0]
    out = []
    for c in range(K_c):
        for cp in range(c, K_c):
            if int(assignments[c, cp]) == atom_idx:
                out.append((c, cp))
    return out


def format_params_summary(pi_class: np.ndarray,
                            potts_atoms: np.ndarray,
                            potts_assignments: np.ndarray | None = None,
                            top_aa: int = 5,
                            top_pairs: int = 10,
                            indent: str = "  ") -> str:
    """Compact, multi-line summary string. Pure formatting; no I/O."""
    K_c = pi_class.shape[0]
    K_H = potts_atoms.shape[0]

    lines = []
    lines.append("pi_class:")
    for c in range(K_c):
        lines.append(f"{indent}class {c}: {_topk_aa(pi_class[c], top_aa)}   "
                       f"(entropy {_entropy_bits(pi_class[c]):.2f} bits)")

    H_norms = [float(np.linalg.norm(potts_atoms[h])) for h in range(K_H)]
    lines.append(f"Potts atoms (K_H={K_H}, ||H||_F = "
                   f"[{', '.join(f'{n:.2f}' for n in H_norms)}]):")
    for h in range(K_H):
        H = potts_atoms[h]
        lines.append(f"{indent}atom {h}:")
        lines.append(f"{indent}  diag favored:    {_topk_diag(H, k=min(5, top_pairs // 2))}")
        lines.append(f"{indent}  offdiag favored: {_topk_pairs_offdiag(H, k=top_pairs, sign='neg')}")
        lines.append(f"{indent}  offdiag penal'd: {_topk_pairs_offdiag(H, k=min(5, top_pairs // 2), sign='pos')}")
        if potts_assignments is not None:
            cps = _atom_class_pairs(potts_assignments, h)
            lines.append(f"{indent}  class-pairs:     {cps}")
    return "\n".join(lines)


def inspect_param_file(run_dir: Path, top_aa: int = 5, top_pairs: int = 10) -> None:
    """CLI: load + print summary from a results dir or a _chkpt dir."""
    run_dir = Path(run_dir)
    if (run_dir / "state.npz").exists():
        # _chkpt-style: arrays packed in npz
        arrs = np.load(run_dir / "state.npz")
        pi_class = arrs["pi_class"]
        potts_atoms = arrs["potts_atoms"]
        potts_assignments = arrs["potts_assignments"]
    else:
        # results/-style: separate .npy files
        pi_class = np.load(run_dir / "pi_class.npy")
        potts_atoms = np.load(run_dir / "potts_atoms.npy")
        potts_assignments = np.load(run_dir / "potts_assignments.npy")
    print(format_params_summary(pi_class, potts_atoms, potts_assignments,
                                  top_aa=top_aa, top_pairs=top_pairs))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--top-aa", type=int, default=5)
    ap.add_argument("--top-pairs", type=int, default=10)
    args = ap.parse_args()
    inspect_param_file(args.run_dir, top_aa=args.top_aa, top_pairs=args.top_pairs)
