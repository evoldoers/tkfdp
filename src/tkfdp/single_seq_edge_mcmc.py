"""Single-sequence edge MCMC sampler.

Used as the baseline (a) / (b) panels of the Holmes 2004 Fig 13/14
composite tile: with NO cross-sequence covariation evidence (only x
provided), the sampler should be effectively flat on signal --
i.e. it cannot localise covarying-pair signal (e.g. disulfide-bonded
cysteines) from a single sequence alone.

Target distribution
-------------------

For a single sequence x of length L, the target on edge sets
E subset of {{i1, i2} | 1 <= i1 < i2 <= L} is

    P(E | x) propto eps^|E| * prod_{{i1, i2} in E} M_solo(x_{i1}, x_{i2})

with eps = 1 / alpha_z and M_solo the AA-marginal Potts-coupling-induced
boost obtained by summing out the OTHER sequence axis (Y) in the joint
M_AA tensor:

    M_solo[a, c] = (sum_{c1, c2} pi_class[c1] pi_class[c2]
                       * sum_{b, d} J[c1, c2, a, c, b, d])
                    / (pi_marg_x[a] * pi_marg_x[c])

where pi_marg_x[a] = sum_{c, b, d} pi_class[c] * J[c, c, a, a, b, d] /
A is the AA marginal under the prior class distribution. In practice we
build this marginal by summing the joint emission tensor and projecting,
following the same conventions as `aug_phmm.build_M_tensor_aa_marginal`.

H6 matching constraint: like the joint sampler, edges form a MATCHING
(no shared endpoints). Each position appears in at most one edge.

Kernel
------

Single-chain MH with:

  1. Edge add: pick a uniformly random unordered pair of currently
     unpaired distinct positions; accept by
       log H_add = log eps + log M_solo + log n_pairs_unp
                   - log(|E| + 1).

  2. Edge remove: pick an existing edge uniformly; accept by
       log H_remove = -log eps - log M_solo + log |E|
                       - log n_pairs_unp_after.

  3. Optional replica exchange on alpha_z (parallel tempering), same
     ladder semantics as `mcmc_infinite_phmm.run_replica_exchange_chain`.
     Swap MH on log alpha_swap = (|E_b| - |E_a|) * (log alpha_b - log alpha_a).

Output
------

Returns:
  - edge_pair_counts: Dict[(i1, i2) -> int], unordered-pair posterior counts.
  - edge_pos_counts: List[int] of length L+1, position marginal counts
    (each edge contributes 2: one to each endpoint).
  - triangular posterior matrix P[i, j] = count / n_recorded, symmetric.
  - SingleSeqDiagnostics: ESS-amenable traces + acceptance rates.

Verification
------------

`tests/test_single_seq_edge_mcmc.py` (alongside this module) verifies
the sampler equilibrates against direct enumeration of all edge-set
configurations on a small toy sequence (L = 5, |2^{L choose 2}| = 1024).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SingleSeqSetup:
    """Pre-computed per-sequence tables for the single-seq edge MCMC.

    Attributes:
        L:          sequence length.
        x_seq:      length-L int sequence (AA codes 0..A-1; wildcards
                    pre-clamped to A-1).
        log_M_solo: (A, A) log of the AA-marginal single-axis boost,
                    symmetric: log_M_solo[a, c] = log_M_solo[c, a].
                    Computed from the same joint_per_cp + pi_class
                    that the joint sampler uses, marginalised over the
                    OTHER sequence axis.
        alpha_z:    edge prior concentration. eps = 1 / alpha_z.
    """
    L: int
    x_seq: np.ndarray
    log_M_solo: np.ndarray
    alpha_z: float = 100.0


@dataclass
class SingleSeqDiagnostics:
    """Diagnostics for one single-seq edge MCMC chain."""
    n_sweeps: int = 0
    n_burnin: int = 0
    n_accept_add: int = 0
    n_accept_remove: int = 0
    n_propose_add: int = 0
    n_propose_remove: int = 0
    n_edges_trace: List[int] = field(default_factory=list)
    log_pi_trace: List[float] = field(default_factory=list)
    # Edge marginal posterior accumulators (post-burnin counts).
    edge_pair_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    edge_pos_counts: List[int] = field(default_factory=list)
    n_recorded_for_edges: int = 0
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Build the single-axis M_solo tensor from boost_state.
# ---------------------------------------------------------------------------


def _build_M_solo_canonical(tkf_state, pi_c: np.ndarray, t: float,
                             pair_background: str = 'lg08') -> np.ndarray:
    """Canonical (post-2026-05-15) single-axis M_solo via block_likelihoods.

    M_solo[a, c] is the boost for an "edge at coupled sites with
    observed (a, c) on one sequence" relative to the singleton-emission
    baseline, marginalising over the unobserved other sequence:

        M_solo[a, c] = [sum_{b, d} P_doublet(a, b, c, d; t)] /
                        [pi_singlet(a) * pi_singlet(c)]

    Under reversibility this reduces to
        M_solo[a, c] = pi_joint(a, c) / (pi_singlet(a) * pi_singlet(c))
    independent of t. We pass t for completeness (the doublet builder
    needs a branch length anyway); the value chosen does not affect the
    result up to numerical precision.

    Uses the canonical convention (LG08 pair-stationary background +
    empirical pi_c) consistent with the K=4 emwarm release.
    """
    from .block_likelihoods import (build_doublet_emission,
                                      build_singlet_emission)
    P_d = build_doublet_emission(tkf_state, float(t),
                                   pi_c=np.asarray(pi_c, dtype=np.float64),
                                   pair_background=pair_background)
    P_s, _, _ = build_singlet_emission(tkf_state, float(t),
                                         pi_c=np.asarray(pi_c, dtype=np.float64))
    numer = P_d.sum(axis=(1, 3))                    # (A, A)
    pi_s = P_s.sum(axis=1)                          # (A,)
    denom = pi_s[:, None] * pi_s[None, :]
    return numer / np.clip(denom, 1e-300, None)


def build_M_solo_aa_marginal(boost_state, axis: str = 'x',
                               A_alpha: int = 20) -> np.ndarray:
    """Build the (A, A) AA-marginal single-axis boost tensor.

    Two code paths:

    (1) CANONICAL (post-2026-05-15, matches the trained K=4 emwarm
        release): when ``boost_state`` carries ``tkf_state``, ``pi_c``,
        and ``branch_length`` (added by ``build_boost_state`` from
        ``coupled_annealing.py``), we compute M_solo via
        ``block_likelihoods.build_doublet_emission`` and
        ``block_likelihoods.build_singlet_emission`` under the LG08
        pair-stationary background + empirical pi_c convention. This is
        the same M-tensor convention used by the joint-chain sampler in
        ``mcmc_infinite_phmm.py``. ``axis`` is ignored in this path
        because M_solo is symmetric in X<->Y under reversibility.

    (2) RELIC (pre-2026-05-15 fallback for callers that haven't been
        upgraded): the original per-class-pair joint marginalisation
        with uniform pi_class = 1/K_c and the boost_state.denom
        normalisation. Emits a WARNING. Kept only for backward
        compatibility; results are biased relative to the canonical
        target.

    Args:
        boost_state: PairBoostState. For the canonical path needs
                     ``tkf_state``, ``pi_c``, ``branch_length`` set
                     (and optionally ``pair_background``, default
                     ``'lg08'``). The relic path needs ``joint_per_cp``,
                     ``x_seq``, ``y_seq``, ``denom``.
        axis: 'x' or 'y'. Relic-only (canonical M_solo is symmetric).
        A_alpha: alphabet size.

    Returns:
        M_solo: (A, A) boost tensor.
    """
    if (getattr(boost_state, 'tkf_state', None) is not None
            and getattr(boost_state, 'pi_c', None) is not None):
        t = float(getattr(boost_state, 'branch_length', 1.0) or 1.0)
        return _build_M_solo_canonical(
            boost_state.tkf_state,
            np.asarray(boost_state.pi_c, dtype=np.float64),
            t,
            getattr(boost_state, 'pair_background', 'lg08')
        ).astype(np.float64)

    print("[single_seq_edge_mcmc] WARNING: boost_state lacks tkf_state / "
          "pi_c; build_M_solo_aa_marginal falls back to the relic "
          "per-class-pair marginalisation with uniform pi_class. Results "
          "will be biased relative to the canonical target.", flush=True)

    J = np.asarray(boost_state.joint_per_cp)              # (K, K, A, A, A, A)
    K_c = J.shape[0]
    A_local = J.shape[2]
    assert A_local == A_alpha, \
        f"M solo: alphabet mismatch {A_local} vs {A_alpha}"
    pi_class = np.full(K_c, 1.0 / K_c, dtype=np.float64)
    # Marginal numerator: sum over (c1, c2) with prior weights, sum out
    # the orthogonal axis. J indexing: J[c1, c2, a_x_left, a_x_right,
    #                                    b_y_left, b_y_right].
    if axis == 'x':
        # Sum over y-end residues (b_y_left, b_y_right).
        # numer_x[a, c] = sum_{c1, c2} pi[c1] pi[c2] sum_{b, b'} J[c1, c2, a, c, b, b']
        numer = np.einsum('e,f,efacbd->ac',
                          pi_class, pi_class, J,
                          optimize=True)
    elif axis == 'y':
        # Sum over x-end residues (a_x_left, a_x_right).
        # numer_y[b, d] = sum_{c1, c2} pi[c1] pi[c2] sum_{a, a'} J[c1, c2, a, a', b, d]
        numer = np.einsum('e,f,efacbd->bd',
                          pi_class, pi_class, J,
                          optimize=True)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    # Denominator: average the per-cell `denom` along the orthogonal
    # axis to get a per-AA marginal under the prior class distribution.
    # boost_state.denom is (Lx, Ly); for 'x' we average over j (Ly axis),
    # for 'y' we average over i (Lx axis).
    x_seq = np.asarray(boost_state.x_seq).astype(np.int64)
    y_seq = np.asarray(boost_state.y_seq).astype(np.int64)
    denom = np.asarray(boost_state.denom)                 # (Lx, Ly)

    if axis == 'x':
        # denom_x[a] = mean over (i, j) with X_i = a of denom[i, j].
        denom_aa = np.zeros(A_local, dtype=np.float64)
        counts = np.zeros(A_local, dtype=np.int64)
        Lx, Ly = denom.shape
        for i in range(Lx):
            a = int(x_seq[i])
            if a >= A_local:
                continue
            for j in range(Ly):
                denom_aa[a] += float(denom[i, j])
                counts[a] += 1
        fallback = float(np.mean(denom)) if denom.size > 0 else 1e-10
        denom_aa = np.where(counts > 0, denom_aa / np.maximum(counts, 1),
                              fallback)
    else:
        denom_aa = np.zeros(A_local, dtype=np.float64)
        counts = np.zeros(A_local, dtype=np.int64)
        Lx, Ly = denom.shape
        for j in range(Ly):
            b = int(y_seq[j])
            if b >= A_local:
                continue
            for i in range(Lx):
                denom_aa[b] += float(denom[i, j])
                counts[b] += 1
        fallback = float(np.mean(denom)) if denom.size > 0 else 1e-10
        denom_aa = np.where(counts > 0, denom_aa / np.maximum(counts, 1),
                              fallback)

    denom_safe = np.clip(denom_aa, 1e-300, None)
    M_solo = numer / (denom_safe[:, None] * denom_safe[None, :])
    return M_solo.astype(np.float64)


# ---------------------------------------------------------------------------
# Setup builder.
# ---------------------------------------------------------------------------


def precompute_single_seq_setup(boost_state, axis: str = 'x',
                                  alpha_z: float = 100.0,
                                  A_alpha: int = 20) -> SingleSeqSetup:
    """Build a SingleSeqSetup for one sequence drawn from a boost_state.

    axis = 'x' uses boost_state.x_seq and the X-axis M_solo marginal;
    axis = 'y' uses y_seq and the Y-axis marginal.
    """
    M_solo = build_M_solo_aa_marginal(boost_state, axis=axis, A_alpha=A_alpha)
    log_M_solo = np.log(np.clip(M_solo, 1e-300, None))
    if axis == 'x':
        seq = np.asarray(boost_state.x_seq).astype(np.int32)
    else:
        seq = np.asarray(boost_state.y_seq).astype(np.int32)
    return SingleSeqSetup(
        L=int(seq.shape[0]),
        x_seq=np.minimum(seq, A_alpha - 1),
        log_M_solo=log_M_solo,
        alpha_z=float(alpha_z),
    )


def _log_M_solo_at(setup: SingleSeqSetup, i: int, j: int) -> float:
    """log M_solo(x_i, x_j) (1-based position indexing)."""
    return float(setup.log_M_solo[setup.x_seq[i - 1], setup.x_seq[j - 1]])


def _unnormalised_log_target(edges: List[Tuple[int, int]],
                               setup: SingleSeqSetup) -> float:
    """log pi(E | x) (unnormalised) for diagnostics. NOT used in MH ratios."""
    log_boost = 0.0
    for (i, j) in edges:
        log_boost += _log_M_solo_at(setup, i, j)
    eps = 1.0 / setup.alpha_z
    log_prior = len(edges) * float(np.log(eps))
    return log_boost + log_prior


def _edge_add(rng: np.random.Generator, setup: SingleSeqSetup,
              edges: List[Tuple[int, int]]
              ) -> Tuple[List[Tuple[int, int]], bool, bool]:
    """Edge add MH move.

    Returns (new_edges, proposed, accepted). proposed=False only if there
    are no valid unpaired pairs to draw from (which we count as a vacuous
    proposal -- not part of the acceptance statistics).
    """
    L = setup.L
    paired = set()
    for (i, j) in edges:
        paired.add(i); paired.add(j)
    unpaired = [k for k in range(1, L + 1) if k not in paired]
    n_unp = len(unpaired)
    if n_unp < 2:
        return edges, False, False
    n_pairs_unp = n_unp * (n_unp - 1) // 2
    # Pick unordered (a, b) uniformly from the unpaired set.
    flat = int(rng.integers(0, n_pairs_unp))
    a = 0
    while flat >= n_unp - 1 - a:
        flat -= n_unp - 1 - a
        a += 1
    b = a + 1 + flat
    p1 = unpaired[a]; p2 = unpaired[b]
    new_edge = (min(p1, p2), max(p1, p2))
    # Target ratio: eps * M_solo(x_p1, x_p2).
    log_M = _log_M_solo_at(setup, new_edge[0], new_edge[1])
    eps = 1.0 / setup.alpha_z
    log_target = float(np.log(eps)) + log_M
    # Proposal ratio: q_remove(old | new) / q_add(new | old).
    # q_add(new | old) = 1 / n_pairs_unp.
    # q_remove(old | new) = 1 / |E_new| = 1 / (|E| + 1).
    log_qratio = float(np.log(n_pairs_unp)) - float(np.log(len(edges) + 1))
    log_H = min(0.0, log_target + log_qratio)
    u = float(rng.random())
    if np.log(max(u, 1e-300)) < log_H:
        return edges + [new_edge], True, True
    return edges, True, False


def _edge_remove(rng: np.random.Generator, setup: SingleSeqSetup,
                  edges: List[Tuple[int, int]]
                  ) -> Tuple[List[Tuple[int, int]], bool, bool]:
    """Edge remove MH move."""
    if len(edges) == 0:
        return edges, False, False
    L = setup.L
    idx = int(rng.integers(0, len(edges)))
    e_to_remove = edges[idx]
    p1, p2 = e_to_remove
    # Compute n_unpaired after removal.
    paired_after = set()
    for k, e in enumerate(edges):
        if k == idx:
            continue
        paired_after.add(e[0]); paired_after.add(e[1])
    n_unp_after = sum(1 for k in range(1, L + 1) if k not in paired_after)
    n_pairs_unp_after = n_unp_after * (n_unp_after - 1) // 2
    if n_pairs_unp_after <= 0:
        # Defensive: by removing one edge we free 2 positions, so this
        # should not happen for L >= 2.
        return edges, True, False
    log_M = _log_M_solo_at(setup, p1, p2)
    eps = 1.0 / setup.alpha_z
    log_target = -(float(np.log(eps)) + log_M)
    log_qratio = float(np.log(len(edges))) - float(np.log(n_pairs_unp_after))
    log_H = min(0.0, log_target + log_qratio)
    u = float(rng.random())
    if np.log(max(u, 1e-300)) < log_H:
        new_edges = list(edges)
        new_edges.pop(idx)
        return new_edges, True, True
    return edges, True, False


# ---------------------------------------------------------------------------
# Main chain.
# ---------------------------------------------------------------------------


def run_single_seq_chain(setup: SingleSeqSetup,
                          n_sweeps: int = 5000,
                          n_burnin: int = 1000,
                          n_edge_moves_per_sweep: int = 8,
                          seed: int = 0,
                          record_every: int = 1,
                          verbose: bool = False
                          ) -> Tuple[np.ndarray, SingleSeqDiagnostics]:
    """Run a single-chain MH sampler on the single-seq edge target.

    Returns:
      P_triangular: (L+1, L+1) symmetric matrix of marginal edge-pair
        probabilities (zero on the diagonal; index 0 unused).
      SingleSeqDiagnostics.
    """
    import time
    rng = np.random.default_rng(seed)
    diag = SingleSeqDiagnostics()
    diag.n_sweeps = n_sweeps
    diag.n_burnin = n_burnin
    L = setup.L
    diag.edge_pos_counts = [0] * (L + 1)
    edges: List[Tuple[int, int]] = []

    t0 = time.time()
    for sweep in range(n_sweeps):
        # n_edge_moves_per_sweep MH steps (alternating add/remove).
        for _ in range(n_edge_moves_per_sweep):
            if rng.random() < 0.5:
                edges, proposed, accepted = _edge_add(rng, setup, edges)
                if proposed:
                    diag.n_propose_add += 1
                    if accepted:
                        diag.n_accept_add += 1
            else:
                edges, proposed, accepted = _edge_remove(rng, setup, edges)
                if proposed:
                    diag.n_propose_remove += 1
                    if accepted:
                        diag.n_accept_remove += 1
        if sweep >= n_burnin and (sweep % record_every == 0):
            diag.n_edges_trace.append(len(edges))
            diag.log_pi_trace.append(_unnormalised_log_target(edges, setup))
            diag.n_recorded_for_edges += 1
            for (p1, p2) in edges:
                key = (p1, p2)
                diag.edge_pair_counts[key] = diag.edge_pair_counts.get(key, 0) + 1
                diag.edge_pos_counts[p1] += 1
                diag.edge_pos_counts[p2] += 1
        if verbose and (sweep + 1) % 500 == 0:
            mean_E = np.mean(diag.n_edges_trace[-100:]) if diag.n_edges_trace else 0
            print(f"  single_seq sweep {sweep + 1}/{n_sweeps}: "
                  f"|E|={len(edges)} <|E|>={mean_E:.2f} "
                  f"acc_add={diag.n_accept_add / max(1, diag.n_propose_add):.2f} "
                  f"acc_rm={diag.n_accept_remove / max(1, diag.n_propose_remove):.2f}")

    diag.runtime_seconds = time.time() - t0
    # Build triangular dense matrix.
    P = np.zeros((L + 1, L + 1), dtype=np.float64)
    n_rec = max(diag.n_recorded_for_edges, 1)
    for (i1, i2), c in diag.edge_pair_counts.items():
        v = c / n_rec
        P[i1, i2] = v
        P[i2, i1] = v
    return P, diag


# ---------------------------------------------------------------------------
# Replica-exchange variant. Same kernel; ladder over alpha_z.
# ---------------------------------------------------------------------------


def run_single_seq_replica_exchange(
        setup_template: SingleSeqSetup,
        alpha_z_ladder: List[float],
        n_sweeps: int = 5000,
        n_burnin: int = 1000,
        n_edge_moves_per_sweep: int = 8,
        seed: int = 0,
        record_every: int = 1,
        swap_every: int = 10,
        verbose: bool = False,
        ) -> Tuple[np.ndarray, Dict]:
    """Replica-exchange MH on the alpha_z ladder.

    Cold rung (smallest alpha_z) is the target; hot rungs (large alpha_z)
    favour zero edges and provide escape routes for the cold chain.

    Returns:
      P_cold: cold-rung triangular posterior matrix.
      diagnostics dict: per_rung list, alpha_z_ladder, swap stats.
    """
    import time
    from dataclasses import replace as _dc_replace
    K = len(alpha_z_ladder)
    if K < 1:
        raise ValueError("alpha_z_ladder must have at least 1 entry")
    alpha_sorted = sorted(alpha_z_ladder)
    setups = [_dc_replace(setup_template, alpha_z=float(a)) for a in alpha_sorted]
    rng = np.random.default_rng(seed)
    diags: List[SingleSeqDiagnostics] = []
    for _ in range(K):
        d = SingleSeqDiagnostics()
        d.n_sweeps = n_sweeps
        d.n_burnin = n_burnin
        d.edge_pos_counts = [0] * (setup_template.L + 1)
        diags.append(d)
    states: List[List[Tuple[int, int]]] = [[] for _ in range(K)]
    swap_n_propose = [0] * max(K - 1, 0)
    swap_n_accept = [0] * max(K - 1, 0)

    t0 = time.time()
    for sweep in range(n_sweeps):
        for k in range(K):
            edges = states[k]
            for _ in range(n_edge_moves_per_sweep):
                if rng.random() < 0.5:
                    edges, proposed, accepted = _edge_add(rng, setups[k], edges)
                    if proposed:
                        diags[k].n_propose_add += 1
                        if accepted:
                            diags[k].n_accept_add += 1
                else:
                    edges, proposed, accepted = _edge_remove(rng, setups[k], edges)
                    if proposed:
                        diags[k].n_propose_remove += 1
                        if accepted:
                            diags[k].n_accept_remove += 1
            states[k] = edges
        # Swap proposal (random adjacent pair).
        if K > 1 and (sweep + 1) % swap_every == 0:
            a = int(rng.integers(0, K - 1))
            b = a + 1
            log_alpha_a = float(np.log(setups[a].alpha_z))
            log_alpha_b = float(np.log(setups[b].alpha_z))
            log_ratio = (len(states[b]) - len(states[a])) * (log_alpha_b - log_alpha_a)
            log_ratio = min(0.0, log_ratio)
            swap_n_propose[a] += 1
            u = float(rng.random())
            if np.log(max(u, 1e-300)) < log_ratio:
                swap_n_accept[a] += 1
                states[a], states[b] = states[b], states[a]
        # Cold-rung diagnostics.
        if sweep >= n_burnin and (sweep % record_every == 0):
            cold_edges = states[0]
            diags[0].n_edges_trace.append(len(cold_edges))
            diags[0].log_pi_trace.append(
                _unnormalised_log_target(cold_edges, setups[0]))
            diags[0].n_recorded_for_edges += 1
            for (p1, p2) in cold_edges:
                key = (p1, p2)
                diags[0].edge_pair_counts[key] = (
                    diags[0].edge_pair_counts.get(key, 0) + 1)
                diags[0].edge_pos_counts[p1] += 1
                diags[0].edge_pos_counts[p2] += 1
        if verbose and (sweep + 1) % 500 == 0:
            mean_E_cold = (np.mean(diags[0].n_edges_trace[-100:])
                           if diags[0].n_edges_trace else 0)
            swap_acc = [a / max(1, p) for a, p in zip(swap_n_accept, swap_n_propose)]
            print(f"  single_seq RE sweep {sweep + 1}/{n_sweeps}: "
                  f"|E|=cold:{len(states[0])} hot:{len(states[-1])} "
                  f"<|E_cold|>={mean_E_cold:.2f} "
                  f"swap_acc={[f'{x:.2f}' for x in swap_acc]}")

    runtime = time.time() - t0
    for d in diags:
        d.runtime_seconds = runtime
    L = setup_template.L
    P_cold = np.zeros((L + 1, L + 1), dtype=np.float64)
    n_rec = max(diags[0].n_recorded_for_edges, 1)
    for (i1, i2), c in diags[0].edge_pair_counts.items():
        v = c / n_rec
        P_cold[i1, i2] = v
        P_cold[i2, i1] = v
    return P_cold, {
        'per_rung': diags,
        'alpha_z_ladder': alpha_sorted,
        'swap_n_propose': swap_n_propose,
        'swap_n_accept': swap_n_accept,
        'runtime_seconds': runtime,
    }


# ---------------------------------------------------------------------------
# Exact enumeration (small L). For testing only.
# ---------------------------------------------------------------------------


def _all_matchings(L: int):
    """Enumerate every matching (subset of unordered position-pairs with
    no shared endpoint) on positions 1..L. Yields a list[tuple[int, int]].
    """
    # Recursive: pick smallest unpaired position; for each choice of
    # partner (or "leave it unpaired"), recurse on the rest.
    def helper(available: List[int]):
        if not available:
            yield []
            return
        p = available[0]
        rest = available[1:]
        # Option 1: leave p unpaired.
        yield from helper(rest)
        # Option 2: pair p with each remaining position.
        for k, q in enumerate(rest):
            new_rest = rest[:k] + rest[k + 1:]
            for sub in helper(new_rest):
                yield [(p, q)] + sub

    yield from helper(list(range(1, L + 1)))


def exact_edge_pair_posterior(setup: SingleSeqSetup) -> np.ndarray:
    """Compute the EXACT edge-pair posterior by direct enumeration over
    all matchings on positions 1..L.

    Cost grows super-exponentially in L (number of matchings on L
    positions = double factorial / sub-factorial); only viable for
    L <= ~10.

    Returns: (L+1, L+1) symmetric matrix P[i, j] = P((i, j) in E | x),
    diagonal = 0.
    """
    L = setup.L
    log_eps = float(np.log(1.0 / setup.alpha_z))
    log_weights = []
    matchings = list(_all_matchings(L))
    for M in matchings:
        log_w = len(M) * log_eps
        for (i, j) in M:
            log_w += _log_M_solo_at(setup, i, j)
        log_weights.append(log_w)
    log_weights = np.asarray(log_weights, dtype=np.float64)
    log_Z = float(np.logaddexp.reduce(log_weights))
    log_p = log_weights - log_Z
    p = np.exp(log_p)
    P = np.zeros((L + 1, L + 1), dtype=np.float64)
    for w, M in zip(p, matchings):
        for (i, j) in M:
            P[i, j] += w
            P[j, i] += w
    return P
