"""Composite (cherry) log-likelihood of an MSA under the infinite Pair HMM.

For an observed alignment A_obs between two sequences (X, Y), the marginal
log-probability under the TKF-DP infinite Pair HMM (main.tex sec:infinite-hmm)
factorises as

    log p(A_obs | X, Y) = log pi_TKF92(A_obs) + log Z_E(A_obs) - log Z_total

where Z_E(A_obs) is the *matching polynomial* of the Match-cell graph induced
by A_obs:

    Z_E(A_obs) = sum_{matchings E of Match(A_obs)} (1/alpha_z)^|E|
                 * prod_{(u, v) in E} M(u, v)

and Z_total = sum_A pi_TKF92(A) * Z_E(A) is the model-level partition.

For comparing two MSAs under the SAME model, Z_total cancels and only the
per-cherry sum

    sum_{i<j} [log pi_TKF92(A_a(i,j)) + log Z_E(A_a(i,j))]
            - [log pi_TKF92(A_b(i,j)) + log Z_E(A_b(i,j))]

matters.

The pi_TKF92 part is a single _path_log_prob call (reused from
mcmc_infinite_phmm.py). The hard part is log Z_E(A_obs), which is generally
#P-hard for N_M > ~20. We estimate it via *annealed importance sampling*
(AIS) between

  - reference: alpha_z = alpha_z_init (very large; the empty matching
    dominates and Z_E ~ 1 by construction);
  - target:    alpha_z = alpha_z_target (the user-supplied concentration,
    typically 100).

Inner kernel at each AIS rung is edge add/remove MH only (the alignment
A_obs is fixed throughout). All log-probs are in nats.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from math import comb, lgamma, log
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Tkfmixdom Pair HMM primitives (only for the per-cherry tau optimisation).
TKFMIXDOM_ROOT = Path.home() / "tkf-mixdom" / "python"
if str(TKFMIXDOM_ROOT) not in sys.path:
    sys.path.insert(0, str(TKFMIXDOM_ROOT))

from .mcmc_infinite_phmm import (  # noqa: E402
    MCMCSetup,
    _edge_add_move,
    _edge_remove_move,
    _log_M_obs,
    _match_cells_of,
    _path_log_prob,
    precompute_partial_forward,
)
from tkfmixdom.jax.core.params import M as M_STATE  # noqa: E402

# State-type constants from the upstream Pair HMM.
_M = int(M_STATE)
_I_STATE = 2
_D_STATE = 3


# ===========================================================================
# Path / cherry construction from MSAs.
# ===========================================================================


def msa_pair_to_path(
        msa_row_x: Sequence[int],
        msa_row_y: Sequence[int],
        ) -> List[Tuple[int, int, int]]:
    """Convert two aligned MSA rows into a Pair HMM path.

    Args:
      msa_row_x, msa_row_y: aligned rows; gap = -1, residue = 0..A-1
        (or 20 for wildcard). Must be of equal length.

    Returns:
      path: list of (state, i, j) triples in path order. State is one of
        {M, I, D}. Coordinates (i, j) are 1-based positions of the residues
        consumed up through the cell. Columns where BOTH rows are gap are
        skipped (they contribute no cell).
    """
    assert len(msa_row_x) == len(msa_row_y), \
        f"row length mismatch: {len(msa_row_x)} vs {len(msa_row_y)}"
    path: List[Tuple[int, int, int]] = []
    i, j = 0, 0
    for k in range(len(msa_row_x)):
        x_gap = (int(msa_row_x[k]) < 0)
        y_gap = (int(msa_row_y[k]) < 0)
        if x_gap and y_gap:
            continue
        if not x_gap and not y_gap:
            i += 1; j += 1
            path.append((_M, i, j))
        elif x_gap and not y_gap:
            j += 1
            path.append((_I_STATE, i, j))
        else:
            i += 1
            path.append((_D_STATE, i, j))
    return path


def extract_pair_seqs_from_msa(
        msa_row_x: Sequence[int],
        msa_row_y: Sequence[int],
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the unaligned (X, Y) integer sequences from two MSA rows.

    Wildcards (residues >= 20) are kept as-is so the upstream pair-HMM
    emission machinery can map them to flat-zero log-emit; the M_obs
    AA-tensor lookup separately clamps to 19.
    """
    x = np.asarray([int(c) for c in msa_row_x if int(c) >= 0],
                   dtype=np.int32)
    y = np.asarray([int(c) for c in msa_row_y if int(c) >= 0],
                   dtype=np.int32)
    return x, y


# ===========================================================================
# AIS estimator for log Z_E(A_obs).
# ===========================================================================


@dataclass
class AISDiagnostics:
    """Summary of an AIS run."""
    n_chains: int = 0
    n_ais_steps: int = 0
    n_inner_sweeps: int = 0
    alpha_z_init: float = 0.0
    alpha_z_target: float = 0.0
    log_Z_per_chain: List[float] = field(default_factory=list)
    final_n_edges_per_chain: List[int] = field(default_factory=list)
    runtime_seconds: float = 0.0
    # ESS in normal-importance-sampling sense over chains; small ESS = high
    # variance estimate.
    log_ess: float = 0.0


def _alpha_z_schedule(alpha_z_init: float, alpha_z_target: float,
                      n_steps: int) -> np.ndarray:
    """Log-linear schedule: alpha_z[k] = init * (target / init)^(k / K).

    Returns array of length n_steps + 1: schedule[0] = init, schedule[K] =
    target.
    """
    if n_steps < 1:
        return np.array([alpha_z_target], dtype=np.float64)
    ks = np.arange(n_steps + 1, dtype=np.float64) / n_steps
    log_init = np.log(alpha_z_init)
    log_target = np.log(alpha_z_target)
    return np.exp(log_init + (log_target - log_init) * ks)


def _run_ais_chain(
        setup: MCMCSetup,
        path: List[Tuple[int, int, int]],
        alpha_z_schedule: np.ndarray,
        n_inner_sweeps: int,
        edge_moves_per_sweep: int,
        rng: np.random.Generator,
        ) -> Tuple[float, int]:
    """Run one AIS chain at a fixed alignment path.

    Args:
      setup:               MCMCSetup with M_obs precomputed; setup.alpha_z
                           is mutated during the run and restored on exit.
      path:                fixed alignment (state, i, j) list. The Match
                           cells of `path` define the matching graph.
      alpha_z_schedule:    array of length n_ais_steps + 1; schedule[0] is
                           the reference (large alpha_z), schedule[-1] is
                           the target.
      n_inner_sweeps:      MH edge add/remove sweeps per rung (each sweep
                           = edge_moves_per_sweep MH attempts).
      edge_moves_per_sweep:per-sweep number of MH attempts.
      rng:                 RNG.

    Returns:
      (log_w, final_|E|): cumulative log AIS weight (= log Z_E estimate
      under the convention Z_ref = 1) and final edge count.
    """
    a0 = setup.alpha_z
    K = alpha_z_schedule.shape[0] - 1
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    log_w = 0.0
    # Initialise at the reference rung.
    setup.alpha_z = float(alpha_z_schedule[0])
    # Optionally do a few burn-in sweeps at the reference. With alpha_z huge
    # this is essentially a no-op (acceptance ~ 0 for any add).
    for _ in range(max(1, n_inner_sweeps // 2)):
        for _m in range(edge_moves_per_sweep):
            if rng.random() < 0.5:
                edges, _ = _edge_add_move(rng, setup, path, edges, k_max=-1)
            elif edges:
                edges, _ = _edge_remove_move(rng, setup, path, edges)

    # AIS loop. At each step k = 1..K:
    #   (a) Compute log_w_k = log[pi_k(E) / pi_{k-1}(E)]
    #       = (log alpha_{k-1} - log alpha_k) * |E|.
    #   (b) Update setup.alpha_z to alpha_k.
    #   (c) Do n_inner_sweeps MH sweeps at rung k.
    log_alpha_prev = float(np.log(alpha_z_schedule[0]))
    for k in range(1, K + 1):
        log_alpha_k = float(np.log(alpha_z_schedule[k]))
        log_w += (log_alpha_prev - log_alpha_k) * len(edges)
        setup.alpha_z = float(alpha_z_schedule[k])
        log_alpha_prev = log_alpha_k
        for _ in range(n_inner_sweeps):
            for _m in range(edge_moves_per_sweep):
                if rng.random() < 0.5:
                    edges, _ = _edge_add_move(
                        rng, setup, path, edges, k_max=-1)
                elif edges:
                    edges, _ = _edge_remove_move(rng, setup, path, edges)

    setup.alpha_z = a0
    return log_w, len(edges)


def estimate_log_Z_E(
        setup: MCMCSetup,
        A_obs: List[Tuple[int, int, int]],
        alpha_z: float,
        alpha_z_init: float = 1e8,
        n_ais_steps: int = 20,
        n_inner_sweeps: int = 100,
        edge_moves_per_sweep: int = 8,
        n_chains: int = 4,
        seed: int = 0,
        ) -> Tuple[float, AISDiagnostics]:
    """Estimate log Z_E(A_obs) via annealed importance sampling.

    Z_E(A_obs) = sum_{matchings E} (1/alpha_z)^|E| * prod_e M_obs(e).

    Reference: alpha_z = alpha_z_init (very large; Z_ref ~= 1 since the
    empty matching dominates by construction).
    Target:    alpha_z (user supplied; typically 100).

    Anneal log-linearly in alpha_z. Inner kernel: edge add/remove MH only.

    Multi-chain: log_Z_E = logmeanexp(per-chain log_w).

    Returns:
      (log_Z_E_estimate, diagnostics)
    """
    t0 = time.time()
    a_save = setup.alpha_z
    schedule = _alpha_z_schedule(
        alpha_z_init=float(alpha_z_init),
        alpha_z_target=float(alpha_z),
        n_steps=int(n_ais_steps),
    )
    diag = AISDiagnostics(
        n_chains=int(n_chains),
        n_ais_steps=int(n_ais_steps),
        n_inner_sweeps=int(n_inner_sweeps),
        alpha_z_init=float(alpha_z_init),
        alpha_z_target=float(alpha_z),
    )
    logws = []
    finalEs = []
    for c in range(int(n_chains)):
        rng = np.random.default_rng(int(seed) * 1000003 + c)
        logw, nE = _run_ais_chain(
            setup, A_obs, schedule, n_inner_sweeps,
            edge_moves_per_sweep, rng)
        logws.append(float(logw))
        finalEs.append(int(nE))
    diag.log_Z_per_chain = list(logws)
    diag.final_n_edges_per_chain = finalEs
    setup.alpha_z = a_save
    diag.runtime_seconds = time.time() - t0
    if not logws:
        return 0.0, diag
    arr = np.asarray(logws, dtype=np.float64)
    m = float(arr.max())
    log_mean_exp = m + float(np.log(np.mean(np.exp(arr - m))))
    # Effective sample size (in nats):
    # log_ESS = 2 log sum exp(w) - log sum exp(2w).
    if len(arr) > 1:
        w_shift = arr - m
        logsumexp = float(m + np.log(np.sum(np.exp(w_shift))))
        log_sumexp2 = float(2 * m + np.log(np.sum(np.exp(2 * w_shift))))
        diag.log_ess = float(2 * logsumexp - log_sumexp2)
    else:
        diag.log_ess = 0.0
    return float(log_mean_exp), diag


# ===========================================================================
# Closed-form / brute-force references for verification.
# ===========================================================================


def _double_factorial(n: int) -> int:
    """(n)!! for n >= -1.

    For 2k - 1: (2k - 1)!! = (2k)! / (2^k k!).
    """
    if n <= 0:
        return 1
    out = 1
    for v in range(n, 0, -2):
        out *= v
    return out


def log_Z_E_closed_form_M1(N_M: int, alpha_z: float) -> float:
    """Closed form for log Z_E(A_obs) when all M_obs(e) = 1.

    Z_E = sum_{k=0..floor(N_M/2)} C(N_M, 2k) * (2k-1)!! * (1/alpha_z)^k.
    """
    if N_M < 0:
        return float("-inf")
    eps = 1.0 / float(alpha_z)
    terms = []
    for k in range(0, N_M // 2 + 1):
        c = comb(N_M, 2 * k)
        df = _double_factorial(2 * k - 1)
        term = c * df * (eps ** k)
        terms.append(term)
    Z = float(sum(terms))
    return float(np.log(max(Z, 1e-300)))


def log_Z_E_brute_force(
        match_cells: List[Tuple[int, int]],
        log_M_lookup,
        alpha_z: float,
        ) -> float:
    """Enumerate all matchings of `match_cells` and compute log Z_E exactly.

    Args:
      match_cells: list of N_M cells (2-tuples).
      log_M_lookup: function (cell_a, cell_b) -> log M(a, b).
      alpha_z: edge concentration.

    For N_M > ~16, the matching count blows up; intended for tests at
    N_M <= 14.
    """
    N = len(match_cells)
    if N == 0:
        return 0.0
    log_eps = float(np.log(1.0 / float(alpha_z)))

    # Recursive enumeration: state is the set of unmatched cell indices.
    # We pick the lowest unmatched index and either leave it as a singleton
    # or pair it with each strictly-greater unmatched index.
    cache: Dict[int, float] = {}

    def recurse(used_mask: int) -> float:
        if used_mask in cache:
            return cache[used_mask]
        # Find lowest unmatched index.
        first = -1
        for i in range(N):
            if not (used_mask >> i) & 1:
                first = i
                break
        if first == -1:
            return 0.0  # log(1)
        # Option 1: leave first as a singleton.
        new_mask = used_mask | (1 << first)
        log_terms = [recurse(new_mask)]
        # Option 2: pair first with j > first, contributing log_eps + log M.
        for j in range(first + 1, N):
            if (used_mask >> j) & 1:
                continue
            new_mask2 = used_mask | (1 << first) | (1 << j)
            log_M_ij = float(log_M_lookup(match_cells[first], match_cells[j]))
            log_terms.append(log_eps + log_M_ij + recurse(new_mask2))
        m = max(log_terms)
        out = m + float(np.log(sum(np.exp(t - m) for t in log_terms)))
        cache[used_mask] = out
        return out

    return recurse(0)


# ===========================================================================
# Cherry / MSA-level composite log-likelihood.
# ===========================================================================


@dataclass
class CherryComposite:
    """Per-cherry composite-loglik record."""
    i: int
    j: int
    name_i: str
    name_j: str
    Lx: int
    Ly: int
    N_match: int
    log_pi_TKF92: float
    log_Z_E: float
    log_p_A: float          # log_pi_TKF92 + log_Z_E
    tau: float
    setup_seconds: float
    ais_seconds: float
    n_edges_final_chains: List[int] = field(default_factory=list)


def _build_cherry_setup(
        x_seq: np.ndarray, y_seq: np.ndarray, t: float,
        Q_lg, pi_lg, boost_state,
        ins_rate: float, del_rate: float, ext: float,
        alpha_z: float,
        ) -> MCMCSetup:
    """Wrap precompute_partial_forward."""
    return precompute_partial_forward(
        x_seq=x_seq, y_seq=y_seq, t=t,
        ins_rate=ins_rate, del_rate=del_rate, ext=ext,
        Q_lg=Q_lg, pi_lg=pi_lg, boost_state=boost_state,
        alpha_z=alpha_z)


def composite_loglik_cherry(
        msa_row_x: Sequence[int],
        msa_row_y: Sequence[int],
        x_seq: np.ndarray,
        y_seq: np.ndarray,
        t: float,
        Q_lg, pi_lg,
        boost_state,
        alpha_z: float = 100.0,
        ins_rate: float = 0.02,
        del_rate: float = 0.05,
        ext: float = 0.5,
        n_ais_steps: int = 20,
        n_inner_sweeps: int = 100,
        edge_moves_per_sweep: int = 8,
        n_chains: int = 4,
        alpha_z_init: float = 1e8,
        seed: int = 0,
        setup: Optional[MCMCSetup] = None,
        ) -> Tuple[float, float, float, AISDiagnostics, MCMCSetup]:
    """Composite log-likelihood of one cherry alignment under the model.

    Returns:
      (log_pi_TKF92, log_Z_E, log_p_A, ais_diag, setup)

    where log_p_A = log_pi_TKF92 + log_Z_E (the per-cherry contribution
    to log_L_A_obs; the global log Z_total cancels in ratios).
    """
    if setup is None:
        setup = _build_cherry_setup(
            x_seq, y_seq, t, Q_lg, pi_lg, boost_state,
            ins_rate=ins_rate, del_rate=del_rate, ext=ext,
            alpha_z=alpha_z)
    path = msa_pair_to_path(msa_row_x, msa_row_y)
    log_pi = _path_log_prob(path, setup)
    log_Z_E, diag = estimate_log_Z_E(
        setup=setup, A_obs=path, alpha_z=alpha_z,
        alpha_z_init=alpha_z_init,
        n_ais_steps=n_ais_steps, n_inner_sweeps=n_inner_sweeps,
        edge_moves_per_sweep=edge_moves_per_sweep,
        n_chains=n_chains, seed=seed)
    return float(log_pi), float(log_Z_E), float(log_pi + log_Z_E), diag, setup


@dataclass
class MSAComposite:
    """Per-MSA composite-loglik aggregate."""
    log_p_total: float
    log_pi_TKF92_total: float
    log_Z_E_total: float
    cherries: List[CherryComposite] = field(default_factory=list)
    total_seconds: float = 0.0


def _msa_extract_aligned_pair(msa: Dict[str, np.ndarray],
                              name_x: str, name_y: str
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the two aligned rows from the MSA.

    The MSA dict maps name -> (L_aln,) array with -1 for gap (matching
    `_msa_from_col_assignments` and `load_ref` conventions). Both rows
    have the same length L_aln.
    """
    rx = np.asarray(msa[name_x], dtype=np.int32)
    ry = np.asarray(msa[name_y], dtype=np.int32)
    assert rx.shape[0] == ry.shape[0], \
        f"MSA row length mismatch: {rx.shape[0]} vs {ry.shape[0]}"
    return rx, ry


def composite_loglik_msa(
        msa: Dict[str, np.ndarray],
        x_seqs: Dict[str, np.ndarray],
        state,
        alpha_z: float = 100.0,
        ins_rate: float = 0.02,
        del_rate: float = 0.05,
        ext: float = 0.5,
        pairs: Optional[List[Tuple[int, int]]] = None,
        boost_states: Optional[dict] = None,
        pair_taus: Optional[Dict[Tuple[int, int], float]] = None,
        setups_cache: Optional[Dict[Tuple[int, int], MCMCSetup]] = None,
        n_ais_steps: int = 20,
        n_inner_sweeps: int = 100,
        edge_moves_per_sweep: int = 8,
        n_chains: int = 4,
        alpha_z_init: float = 1e8,
        seed: int = 0,
        verbose: bool = False,
        ) -> MSAComposite:
    """Composite (cherry) log-likelihood of an MSA under the infinite Pair HMM.

    Sums log_p(A_obs(i, j) | X_i, X_j) over all selected pairs (i, j); each
    summand is log_pi_TKF92 + log_Z_E (the global Z_total cancels in
    pair-wise comparisons of MSAs).

    Args:
      msa:           {name: (L_aln,) int row, -1=gap}.
      x_seqs:        {name: (L_seq,) int sequence}.
      state:         trained TKF-DP minimal state (load_minimal_state output).
      alpha_z:       target edge concentration.
      pairs:         list of (i, j) name-index pairs to score. Default = all.
      boost_states:  optional precomputed dict {(i, j): PairBoostState}; if
                     None, build via build_boost_state.
      pair_taus:     optional precomputed branch lengths; if None, compute
                     via compute_pairwise_posteriors.
      setups_cache:  optional dict {(i, j): MCMCSetup} of precomputed
                     per-cherry setups. If provided, reused (and a fresh
                     entry is added for any missing (i, j)). The setup
                     depends only on (X, Y, t, model) and is independent
                     of the MSA, so it can be shared across multiple MSAs
                     for the same sequence set. This is the dominant
                     wall-time line item (O(L^4) per cherry).
      n_ais_steps,
      n_inner_sweeps,
      edge_moves_per_sweep,
      n_chains,
      alpha_z_init:  AIS hyperparameters.

    Returns:
      MSAComposite with per-cherry breakdown and totals.
    """
    from tkfmixdom.jax.core.protein import rate_matrix_lg  # noqa: WPS433
    from tkfmixdom.jax.tree.fsa_anneal import (
        compute_pairwise_posteriors, select_pairs_full,
    )
    from .coupled_annealing import build_boost_state

    t0 = time.time()
    Q_lg, pi_lg = rate_matrix_lg()
    names = list(x_seqs.keys())
    n_seqs = len(names)
    if pairs is None:
        pairs = select_pairs_full(n_seqs)

    # Precompute pair posteriors / taus / boost states (cached across all
    # cherries of the same MSA).
    if boost_states is None or pair_taus is None:
        pair_post, taus = compute_pairwise_posteriors(
            x_seqs, pairs, model='tkf92', Q=Q_lg, pi=pi_lg,
            ins_rate=ins_rate, del_rate=del_rate, ext=ext)
        if pair_taus is None:
            pair_taus = taus
        if boost_states is None:
            seqs_int = [np.asarray(x_seqs[nm]) for nm in names]
            pair_post_np = {k: np.asarray(v) for k, v in pair_post.items()}
            boost_states = build_boost_state(
                pair_post_np, pair_taus, seqs_int, state)

    cherries: List[CherryComposite] = []
    total_logp = 0.0
    total_logpi = 0.0
    total_logZ = 0.0
    if setups_cache is None:
        setups_cache = {}
    for (i, j) in pairs:
        name_i, name_j = names[i], names[j]
        rx, ry = _msa_extract_aligned_pair(msa, name_i, name_j)
        bs = boost_states[(i, j)]
        # Use the boost-state's clamped sequences (same convention as the
        # rest of the TKF-DP pipeline; the AA-tensor lookup demands 0..19).
        x_arr = np.asarray(bs.x_seq, dtype=np.int32)
        y_arr = np.asarray(bs.y_seq, dtype=np.int32)
        t = float(pair_taus[(i, j)])
        if (i, j) in setups_cache:
            setup = setups_cache[(i, j)]
            setup.alpha_z = float(alpha_z)  # ensure correct value
            setup_secs = 0.0
        else:
            t_setup = time.time()
            setup = _build_cherry_setup(
                x_arr, y_arr, t, Q_lg, pi_lg, bs,
                ins_rate=ins_rate, del_rate=del_rate, ext=ext,
                alpha_z=alpha_z)
            setups_cache[(i, j)] = setup
            setup_secs = time.time() - t_setup
        path = msa_pair_to_path(rx, ry)
        # Sanity: path-implied lengths match clamped sequences.
        N_M = sum(1 for (st, _i, _j) in path if st == _M)
        # Verify path covers all residues (i.e. consistency MSA <-> sequence).
        n_x_in_path = sum(1 for (st, _i, _j) in path if st in (_M, _D_STATE))
        n_y_in_path = sum(1 for (st, _i, _j) in path if st in (_M, _I_STATE))
        if n_x_in_path != len(x_arr) or n_y_in_path != len(y_arr):
            raise ValueError(
                f"MSA pair ({name_i}, {name_j}): path implies "
                f"{n_x_in_path}x / {n_y_in_path}y residues but sequences "
                f"have {len(x_arr)}x / {len(y_arr)}y. MSA may not be "
                f"derived from these sequences.")
        log_pi = _path_log_prob(path, setup)
        t_ais = time.time()
        log_Z_E, ais_diag = estimate_log_Z_E(
            setup=setup, A_obs=path, alpha_z=alpha_z,
            alpha_z_init=alpha_z_init,
            n_ais_steps=n_ais_steps, n_inner_sweeps=n_inner_sweeps,
            edge_moves_per_sweep=edge_moves_per_sweep,
            n_chains=n_chains, seed=seed * 7919 + i * 31 + j)
        ais_secs = time.time() - t_ais
        log_p = log_pi + log_Z_E
        cherries.append(CherryComposite(
            i=i, j=j, name_i=name_i, name_j=name_j,
            Lx=len(x_arr), Ly=len(y_arr), N_match=N_M,
            log_pi_TKF92=float(log_pi),
            log_Z_E=float(log_Z_E),
            log_p_A=float(log_p),
            tau=float(t),
            setup_seconds=float(setup_secs),
            ais_seconds=float(ais_secs),
            n_edges_final_chains=ais_diag.final_n_edges_per_chain,
        ))
        total_logp += log_p
        total_logpi += log_pi
        total_logZ += log_Z_E
        if verbose:
            print(f"  cherry ({name_i}, {name_j}): N_M={N_M:3d} "
                  f"log_pi={log_pi:8.2f}  log_Z_E={log_Z_E:7.3f}  "
                  f"|E|_final={ais_diag.final_n_edges_per_chain}  "
                  f"setup={setup_secs:.1f}s  ais={ais_secs:.1f}s")
    return MSAComposite(
        log_p_total=float(total_logp),
        log_pi_TKF92_total=float(total_logpi),
        log_Z_E_total=float(total_logZ),
        cherries=cherries,
        total_seconds=float(time.time() - t0),
    )


def compare_msa_composite_loglik(
        msa_a: Dict[str, np.ndarray],
        msa_b: Dict[str, np.ndarray],
        x_seqs: Dict[str, np.ndarray],
        state,
        alpha_z: float = 100.0,
        **kwargs,
        ) -> Tuple[float, Dict]:
    """Pairwise comparison of two MSAs under the infinite Pair HMM.

    Both MSAs MUST be derived from the same sequence set `x_seqs`. Returns
    the difference log_L_A - log_L_B (positive => A is preferred), plus a
    per-cherry breakdown.

    The model-level partition Z_total cancels here, so the difference is
    rigorous (no AIS bias from Z_total estimation).

    Returns:
      (log_L_a - log_L_b, breakdown) where breakdown is a dict with keys:
        - "msa_a": MSAComposite for A
        - "msa_b": MSAComposite for B
        - "per_cherry_diff": [(name_i, name_j, log_p_a - log_p_b), ...]
    """
    res_a = composite_loglik_msa(msa_a, x_seqs, state, alpha_z=alpha_z,
                                 **kwargs)
    res_b = composite_loglik_msa(msa_b, x_seqs, state, alpha_z=alpha_z,
                                 **kwargs)
    diffs = []
    by_pair_b = {(c.name_i, c.name_j): c for c in res_b.cherries}
    for c_a in res_a.cherries:
        c_b = by_pair_b.get((c_a.name_i, c_a.name_j))
        if c_b is None:
            continue
        diffs.append((c_a.name_i, c_a.name_j,
                      c_a.log_p_A - c_b.log_p_A))
    return (res_a.log_p_total - res_b.log_p_total,
            dict(msa_a=res_a, msa_b=res_b, per_cherry_diff=diffs))


# ===========================================================================
# CRP partition-prior count (for context; unused in Z_E).
# ===========================================================================


def crp_log_partition_norm(N: int, alpha_z: float) -> float:
    """log Gamma(alpha_z) - log Gamma(alpha_z + N) -- the Ewens normaliser.

    Not used in Z_E directly, but provided for diagnostic conversion
    between conventions: pi_CRP(empty matching | alpha, N) = alpha^N
    * (N - 1)! / [alpha (alpha + 1) ... (alpha + N - 1)]^{-1} * ...
    """
    if N <= 0:
        return 0.0
    return float(lgamma(alpha_z) - lgamma(alpha_z + N))
