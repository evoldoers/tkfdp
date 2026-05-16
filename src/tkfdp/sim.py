"""Forward simulation on the 400-state joint CTMC.

For Exp 2 Layer 1: given H_true, draw cherry pair observations
(t, start_state, end_state) where start ~ pi_joint and end ~ P(t)[start, :].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .generator import build_joint_Q, joint_stationary, symmetrize_eigh, transition_matrices


def simulate_cherries(H_true: jnp.ndarray,
                      t_values: np.ndarray,
                      key: jax.Array,
                      stationary_start: bool = True):
    """Simulate (start, end) state pairs for a batch of cherry distances.

    Args:
        H_true: (20, 20) coupling matrix used to build the joint generator.
        t_values: (M,) numpy array of branch lengths (one per cherry observation).
        key: PRNG key.
        stationary_start: if True, draw start ~ pi_joint(H_true). Otherwise
            draw start ~ Uniform on the 400 states (useful for sanity checks).

    Returns:
        starts: (M,) int32 of starting joint-state indices in 0..399
        ends:   (M,) int32 of ending joint-state indices in 0..399
    """
    Q = build_joint_Q(H_true)
    pi_j = joint_stationary(H_true)
    Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)

    M = len(t_values)
    t_arr = np.asarray(t_values)

    # Quantize t to a unique grid so we can compute P only once per unique value.
    unique_t, t_idx = np.unique(t_arr, return_inverse=True)
    P_unique = np.asarray(
        transition_matrices(jnp.asarray(unique_t), Lambda, U_sym, sqrt_pij)
    )  # (K, 400, 400)
    pi_j_np = np.asarray(pi_j)

    rng = np.random.default_rng(int(np.asarray(jax.random.bits(key, (1,)))[0]))

    if stationary_start:
        starts = rng.choice(400, size=M, p=pi_j_np / pi_j_np.sum())
    else:
        starts = rng.integers(0, 400, size=M)

    ends = np.empty(M, dtype=np.int64)
    for i in range(M):
        row = P_unique[t_idx[i], starts[i], :]
        row = np.clip(row, 0.0, None)
        row = row / row.sum()
        ends[i] = rng.choice(400, p=row)

    return starts.astype(np.int64), ends.astype(np.int64)


def planted_H_chemistry(noise_scale: float = 0.15, seed: int = 7) -> np.ndarray:
    """Return a chemistry-plausible 20x20 symmetric H matrix for synthetic
    recovery tests. Conventions: alphabet ACDEFGHIKLMNPQRSTVWY (alphabetical).
    More negative = more favored pair.

    Most unplanted entries get a small N(0, noise_scale) so that Spearman
    correlation against a recovered H is meaningful (planted-only patterns
    leave too many tied zeros for a sensible rank correlation).
    """
    from .lg08 import ALPHA_ORDER
    idx = {a: i for i, a in enumerate(ALPHA_ORDER)}
    rng = np.random.default_rng(seed)
    H = rng.normal(scale=noise_scale, size=(20, 20))

    # Strongly favored: disulfide
    H[idx['C'], idx['C']] = -3.0

    # Salt bridges (favored)
    for a, b in [('K', 'E'), ('K', 'D'), ('R', 'E'), ('R', 'D')]:
        H[idx[a], idx[b]] = -1.5
        H[idx[b], idx[a]] = -1.5

    # Hydrophobic packing (favored)
    hydrophobics = ['A', 'V', 'L', 'I', 'M', 'F']
    for a in hydrophobics:
        for b in hydrophobics:
            if a != b:
                H[idx[a], idx[b]] = -0.7

    # Like-charge repulsion (disfavored)
    for a in ['K', 'R']:
        for b in ['K', 'R']:
            if a != b:
                H[idx[a], idx[b]] = 1.0
    for a in ['E', 'D']:
        for b in ['E', 'D']:
            if a != b:
                H[idx[a], idx[b]] = 1.0

    # Proline disruption (mildly disfavored with most things — keep modest)
    H[idx['P'], idx['P']] = 0.5

    H = 0.5 * (H + H.T)
    H = H - np.trace(H) / 20.0 * np.eye(20)  # zero-trace canonical form
    return H
