"""τ-binning and precomputed kernel tables (M12).

For a corpus of PaddedTrees, gather all unique per-branch τ values,
bin them to a geomspaced grid (nearest-in-log), and produce:

  - bin_centers: (n_bins,) the representative tau for each bin
  - per-tree bin_indices: (n_slots_l, 2) int32 replacing child_branch

Per Gibbs iter, given (rho, rho_chain, pi_field, S), we precompute
these tables once:

  - beta_table[bin, theta]                shape (n_bins, L)
  - W_table[bin, theta_from, theta_to]    shape (n_bins, L, L)
  - P_sub_table[bin, K_c, L, A, A]        shape (n_bins, K_c, L, A, A)

Inside the JIT forward pass, per-branch kernel lookups become gathers
from these tables via the bin index -- no per-branch matrix exp or
eigendecomp inside the JIT.

Mirrors the exp2_pfam_v2.py cherry convention:
  unique_t = np.geomspace(t_lo, t_hi, n_bins)
  bin_idx = argmin |log(tau) - log(unique_t)|
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from .tree_padded import PaddedTree


def collect_all_taus(padded_trees: 'list[PaddedTree]') -> np.ndarray:
    """Return a 1D array of all per-branch tau values across the corpus."""
    all_taus = []
    for pt in padded_trees:
        for l in range(pt.D_bucket):
            cb = pt.child_branch[l]      # (n_slots_l, 2)
            all_taus.append(cb.reshape(-1))
    if not all_taus:
        return np.array([], dtype=np.float64)
    return np.concatenate(all_taus)


def build_tau_bins(all_taus: np.ndarray,
                        n_bins: int,
                        t_lo: 'float | None' = None,
                        t_hi: 'float | None' = None,
                        ) -> np.ndarray:
    """Return `bin_centers` (n_bins,) geomspaced from [t_lo, t_hi].

    Defaults: t_lo = max(min-nonzero-tau, 1e-3), t_hi = max(2.74, max_tau).
    Matches the exp2_pfam_v2.py cherry-tau convention.
    """
    nonzero = all_taus[all_taus > 0]
    if nonzero.size == 0:
        t_lo = 0.001
        t_hi = 1.0
    else:
        if t_lo is None:
            t_lo = max(0.005, float(nonzero.min()))
        if t_hi is None:
            t_hi = max(2.74, float(nonzero.max()))
    return np.geomspace(t_lo, t_hi, n_bins).astype(np.float64)


def assign_bins(taus: np.ndarray, bin_centers: np.ndarray) -> np.ndarray:
    """Assign each tau to its nearest bin (in log space).

    Returns int array of same shape as taus. Zero taus (identity branch)
    map to bin 0 as a sentinel: caller should mask them.
    """
    taus = np.asarray(taus, dtype=np.float64)
    log_taus = np.log(np.clip(taus, 1e-9, None))
    log_centers = np.log(bin_centers)
    # nearest in log space
    diffs = np.abs(log_taus[..., None] - log_centers[None, :])
    return np.argmin(diffs, axis=-1).astype(np.int32)


def rebuild_padded_trees_with_bins(
        padded_trees: 'list[PaddedTree]',
        bin_centers: np.ndarray,
) -> 'tuple[list[PaddedTree], list[list[np.ndarray]]]':
    """For each PaddedTree, compute per-level bin-index arrays for
    child_branch. Also add an is_zero mask (for the identity/phantom
    branches with tau == 0).

    Returns (padded_trees_unchanged, bin_indices_per_tree_per_level).
    The child_branch entries themselves are left in place; bin indices
    are consumed by the binned forward pass via a parallel array.
    """
    bin_idx_per_tree: 'list[list[np.ndarray]]' = []
    for pt in padded_trees:
        per_level = []
        for l in range(pt.D_bucket):
            cb = pt.child_branch[l]              # (n_slots_l, 2)
            idx = assign_bins(cb, bin_centers)    # (n_slots_l, 2)
            per_level.append(idx)
        bin_idx_per_tree.append(per_level)
    return padded_trees, bin_idx_per_tree


def gtr_P_bins(pi_row: np.ndarray,
                   S: np.ndarray,
                   bin_centers: np.ndarray) -> np.ndarray:
    """Per-bin GTR transition matrices for ONE (class, theta) stationary.

    Returns (n_bins, A, A) with entry [b] = expm(Q * bin_centers[b]), where
    Q is the GTR generator Q_{a,b} = S[a,b] * pi_row[b] (a != b), row-sum
    normalised on the diagonal.

    This is the per-(c, theta) slice of the P_sub table. Isolating it lets
    an arch move — which changes pi_field at only one or two (c, theta)
    entries — recompute just those slices (n_bins expm) instead of the full
    K_c * L * n_bins table (see corpus_state incremental kernel updates).
    """
    Q = S * pi_row[None, :]                                   # (A, A), fresh
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return np.stack([expm(Q * float(t)) for t in bin_centers])


def field_kernel_tables(
        bin_centers: np.ndarray,
        rho: np.ndarray,
        rho_chain: float,
) -> 'tuple[np.ndarray, np.ndarray]':
    """The (beta, W) tables, which depend only on (rho, rho_chain, bins) —
    NOT on pi_field. Held invariant across arch moves.

      beta: (n_bins, L)     W: (n_bins, L, L)
    """
    n_bins = int(bin_centers.shape[0])
    L = int(rho.shape[0])
    beta = np.exp(-rho_chain * (1.0 - rho[None, :])
                     * bin_centers[:, None])                  # (n_bins, L)
    et = np.exp(-rho_chain * bin_centers)                     # (n_bins,)
    W = np.zeros((n_bins, L, L), dtype=np.float64)
    for b in range(n_bins):
        PF = et[b] * np.eye(L) + (1.0 - et[b]) * rho[None, :]
        W[b] = PF - np.diag(beta[b])
    return beta.astype(np.float64), W.astype(np.float64)


def precompute_kernel_tables(
        bin_centers: np.ndarray,
        rho: np.ndarray,
        rho_chain: float,
        pi_field: np.ndarray,
        S: np.ndarray,
) -> dict:
    """Compute per-bin kernel tables.

    Returns:
      beta:   (n_bins, L)
      W:      (n_bins, L, L)
      P_sub:  (n_bins, K_c, L, A, A)
    """
    n_bins = int(bin_centers.shape[0])
    K_c, L, A = pi_field.shape
    beta, W = field_kernel_tables(bin_centers, rho, rho_chain)
    # P_sub[bin, c, theta, :, :] = expm(Q^{c, theta} * tau_bin)
    P_sub = np.zeros((n_bins, K_c, L, A, A), dtype=np.float64)
    for c in range(K_c):
        for th in range(L):
            P_sub[:, c, th] = gtr_P_bins(pi_field[c, th], S, bin_centers)
    return {
        'beta': beta,
        'W': W,
        'P_sub': P_sub.astype(np.float64),
        'bin_centers': bin_centers,
        'pi_field': pi_field.astype(np.float64),
    }
