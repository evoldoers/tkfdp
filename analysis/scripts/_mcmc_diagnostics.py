"""Convergence diagnostics for the infinite-PHMM MCMC sampler.

Pure-numpy: ESS via Geyer-truncated integrated-autocorrelation-time and
classical Gelman-Rubin r-hat. No arviz dependency.

Functions
---------
ess_iact(x: np.ndarray) -> float
    Effective sample size of a 1D chain via integrated autocorrelation
    time (Geyer truncation at first non-positive autocorr).

rhat_gelman_rubin(chains: list[np.ndarray]) -> float | None
    Classical Gelman-Rubin r-hat over a list of equal-length chains.
    Returns None if fewer than 2 chains.

summarise_per_chain_diags(per_chain: list[MCMCDiagnostics],
                           cold_traces_per_chain: Optional[...]) -> dict
    Bundle a per-pair diagnostics record (ESS, r-hat, acceptance rates,
    between-chain SDs) for JSON serialisation.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Optional


def _autocorr_at_lag(x: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation rho_lag of a 1D array.  Uses the
    unbiased denominator (variance computed on the full chain).  Returns
    0.0 if the variance is zero (constant chain)."""
    n = x.size
    if lag >= n or lag < 0:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    mu = float(x.mean())
    v = float(((x - mu) ** 2).mean())
    if v <= 0.0:
        return 0.0
    num = float(((x[:n - lag] - mu) * (x[lag:] - mu)).mean())
    return num / v


def ess_iact(x: Sequence[float]) -> float:
    """Effective sample size via integrated autocorrelation time.

    Geyer's initial-positive-sequence truncation: sum rho_k from k=1
    upward until the first non-positive lag.  ESS = N / (1 + 2 sum_k
    rho_k).  Returns N if the chain is too short for any non-trivial
    estimate.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return float(n)
    if float(x.var()) <= 0.0:
        return float(n)
    tau = 1.0
    # Cap lag scan at n // 4 (heuristic) to keep cost O(N^2 / 4).
    max_lag = max(1, n // 4)
    for lag in range(1, max_lag):
        rho = _autocorr_at_lag(x, lag)
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return float(n / tau)


def rhat_gelman_rubin(chains: Sequence[Sequence[float]]) -> Optional[float]:
    """Classical Gelman-Rubin r-hat for a list of equal-length chains.

    Returns None if fewer than 2 chains, or if all chain means are
    identical (in which case r-hat is exactly 1.0 by definition; we
    return 1.0 explicitly).
    """
    if chains is None or len(chains) < 2:
        return None
    # Truncate to the common shortest length so the formula is well-defined.
    lens = [len(c) for c in chains]
    n = min(lens)
    if n < 2:
        return None
    arrs = np.asarray([np.asarray(c[:n], dtype=np.float64) for c in chains])
    m = arrs.shape[0]   # number of chains
    chain_means = arrs.mean(axis=1)
    chain_vars = arrs.var(axis=1, ddof=1) if n > 1 else np.zeros(m)
    grand_mean = float(chain_means.mean())
    # Between-chain variance B = (n / (m - 1)) * sum_i (mean_i - grand)^2
    B = (n / (m - 1)) * float(((chain_means - grand_mean) ** 2).sum())
    # Within-chain variance W = mean over chains of within-chain variance
    W = float(chain_vars.mean())
    if W <= 0.0 and B <= 0.0:
        return 1.0
    if W <= 0.0:
        return float('inf')
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def geweke_z(chain: Sequence[float],
              first_frac: float = 0.1,
              last_frac: float = 0.5) -> Optional[float]:
    """Geweke (1992) within-chain z-score.

    Compares the mean of the first ``first_frac`` of the post-burn-in
    chain to the mean of the last ``last_frac`` of the chain. Under
    convergence, the two are drawn from the same distribution and
    |z| < ~1.96 holds at the 5% level.

    Returns None for chains too short to estimate (n < 10).
    """
    x = np.asarray(chain, dtype=np.float64)
    n = x.size
    if n < 10:
        return None
    n1 = max(2, int(round(n * first_frac)))
    n2 = max(2, int(round(n * last_frac)))
    a = x[:n1]
    b = x[-n2:]
    m_a, m_b = float(a.mean()), float(b.mean())
    v_a = float(a.var(ddof=1)) / n1 if n1 > 1 else 0.0
    v_b = float(b.var(ddof=1)) / n2 if n2 > 1 else 0.0
    se = (v_a + v_b) ** 0.5
    if se <= 0.0:
        return 0.0
    return float((m_a - m_b) / se)


def round_trip_stats(rung_traj, K: int) -> dict:
    """Replica-exchange round-trip statistics.

    Args:
      rung_traj: list/array of length n_sweeps, where rung_traj[s] is
        a length-K tuple/list giving the rung position of each label at
        sweep s. Label l starts at rung l. Swap acceptance permutes
        labels across rungs.
      K: number of rungs.

    A "full round trip" for a label is a cold (rung 0) -> hot
    (rung K-1) -> cold transition, or vice versa. We track each label
    independently and count alternating visits to the two endpoints.

    Returns:
      dict with n_full_round_trips, mean_round_trip_time (None if
      no completed round trips), n_round_trip_samples, and
      frac_labels_completed_full_RT.
    """
    if rung_traj is None or K < 2:
        return {
            'n_full_round_trips': 0,
            'mean_round_trip_time': None,
            'n_round_trip_samples': 0,
            'frac_labels_completed_full_RT': 0.0,
        }
    rt = np.asarray(rung_traj, dtype=np.int32)
    if rt.ndim != 2 or rt.size == 0:
        return {
            'n_full_round_trips': 0,
            'mean_round_trip_time': None,
            'n_round_trip_samples': 0,
            'frac_labels_completed_full_RT': 0.0,
        }
    n_sweeps, K_check = rt.shape
    if K_check != K:
        return {}
    n_full = 0
    sum_rtt = 0.0
    n_samples = 0
    labels_with_full_rt: set = set()
    for k in range(K):
        traj = rt[:, k]
        ends: list[tuple[int, str]] = []
        prev = None
        for s in range(n_sweeps):
            r = int(traj[s])
            if r == 0:
                if prev != 'cold':
                    ends.append((s, 'cold'))
                    prev = 'cold'
            elif r == K - 1:
                if prev != 'hot':
                    ends.append((s, 'hot'))
                    prev = 'hot'
        # A round trip = three alternating endpoint visits (cold,hot,cold)
        # or (hot,cold,hot).  Time = sweep of last - sweep of first.
        for i in range(2, len(ends)):
            if (ends[i][1] == ends[i - 2][1]
                    and ends[i][1] != ends[i - 1][1]):
                n_full += 1
                sum_rtt += float(ends[i][0] - ends[i - 2][0])
                n_samples += 1
                labels_with_full_rt.add(k)
    return {
        'n_full_round_trips': int(n_full),
        'mean_round_trip_time': (float(sum_rtt / n_samples)
                                  if n_samples > 0 else None),
        'n_round_trip_samples': int(n_samples),
        'frac_labels_completed_full_RT':
            float(len(labels_with_full_rt)) / K,
    }


def acceptance_rates(diag) -> dict:
    """Acceptance rates per move type, taken from an MCMCDiagnostics."""
    out = {}
    for name in ('seg', 'add', 'remove'):
        p = getattr(diag, f'n_propose_{name}', 0)
        a = getattr(diag, f'n_accept_{name}', 0)
        out[f'acc_{name}'] = float(a) / max(1.0, float(p))
        out[f'n_propose_{name}'] = int(p)
        out[f'n_accept_{name}'] = int(a)
    return out


def diags_to_json(per_chain, alpha_z_ladder=None,
                  swap_n_propose=None, swap_n_accept=None,
                  Q_chain_var=None, q_l1_vs_baseline=None,
                  is_replica_exchange: bool = False,
                  rung_traj=None,
                  re_replicate_traces=None) -> dict:
    """Bundle a per-pair MCMC diagnostics record.

    Args:
      per_chain: list of MCMCDiagnostics (one per chain or one per rung).
      alpha_z_ladder: list of alpha_z values for replica-exchange runs.
      swap_n_propose, swap_n_accept: per-rung-pair swap stats.
      Q_chain_var: cell-wise variance across chains; we only store its
        summary (mean, max) — the full tensor would balloon the JSON.
      q_l1_vs_baseline: scalar L1 distance between Q' and Q_baseline.
      is_replica_exchange: True if per_chain is per-rung, False if per-chain.

    Per-chain (or per-rung) entry includes:
      ess_n_match, ess_log_pi, r_hat_n_match, r_hat_log_pi,
      acc_seg/add/remove + propose counts, n_sweeps, n_burnin.
    """
    # Per-chain summaries (ESS, traces, acceptances).
    per_entries = []
    for d in per_chain:
        n_match = list(getattr(d, 'n_match_trace', []) or [])
        log_pi = list(getattr(d, 'log_pi_trace', []) or [])
        n_edges = list(getattr(d, 'n_edges_trace', []) or [])
        entry = {
            'n_sweeps': int(getattr(d, 'n_sweeps', 0)),
            'n_burnin': int(getattr(d, 'n_burnin', 0)),
            'n_recorded': int(len(n_match)),
            'runtime_seconds': float(getattr(d, 'runtime_seconds', 0.0)),
            'setup_seconds': float(getattr(d, 'setup_seconds', 0.0)),
            'setup_breakdown': dict(getattr(d, 'setup_breakdown', {}) or {}),
            'rf_seconds': float(getattr(d, 'rf_seconds', 0.0)),
            'rf_n_misses': int(getattr(d, 'rf_n_misses', 0)),
            'rf_n_hits': int(getattr(d, 'rf_n_hits', 0)),
            'mu_cache_size': int(getattr(d, 'mu_cache_size', 0)),
            'tb_seconds': float(getattr(d, 'tb_seconds', 0.0)),
            'tb_n_calls': int(getattr(d, 'tb_n_calls', 0)),
            'Lx': int(getattr(d, 'Lx', 0)),
            'Ly': int(getattr(d, 'Ly', 0)),
            'mu_cache_size_trace': list(getattr(d, 'mu_cache_size_trace', []) or []),
            # Edge marginal posterior accumulators (post-burnin counts).
            # edge_pos_x_counts[i] = #recorded sweeps where position i on
            # X is an edge endpoint (each edge contributes 2). Divide by
            # n_recorded_for_edges to get P(position has an edge | data).
            # edge_cell_counts: sparse list of [i, j, count].
            'n_recorded_for_edges': int(getattr(d, 'n_recorded_for_edges', 0)),
            'edge_pos_x_counts': [int(x) for x in (
                getattr(d, 'edge_pos_x_counts', []) or [])],
            'edge_pos_y_counts': [int(x) for x in (
                getattr(d, 'edge_pos_y_counts', []) or [])],
            'edge_cell_counts': [
                [int(i), int(j), int(c)]
                for (i, j), c in (getattr(d, 'edge_cell_counts', {}) or {}).items()
            ],
            # X-X / Y-Y unordered-pair projections of the joint sampler's
            # edges. Each row is [i1, i2, count] (with i1 <= i2). Divide
            # by n_recorded_for_edges to get the marginal probability that
            # the unordered pair (i1, i2) on X is connected by an edge.
            'edge_pair_x_counts': [
                [int(i1), int(i2), int(c)]
                for (i1, i2), c in (getattr(d, 'edge_pair_x_counts', {}) or {}).items()
            ],
            'edge_pair_y_counts': [
                [int(j1), int(j2), int(c)]
                for (j1, j2), c in (getattr(d, 'edge_pair_y_counts', {}) or {}).items()
            ],
            'ess_n_match': float(ess_iact(n_match)) if n_match else None,
            'ess_log_pi': float(ess_iact(log_pi)) if log_pi else None,
            'mean_n_edges': float(np.mean(n_edges)) if n_edges else None,
            'mean_n_match': float(np.mean(n_match)) if n_match else None,
            'mean_log_pi': float(np.mean(log_pi)) if log_pi else None,
            'std_n_match': float(np.std(n_match)) if n_match else None,
            'std_log_pi': float(np.std(log_pi)) if log_pi else None,
        }
        entry.update(acceptance_rates(d))
        # Geweke within-chain z-scores (first 10% vs last 50% of chain).
        entry['geweke_z_n_match'] = geweke_z(n_match) if n_match else None
        entry['geweke_z_log_pi'] = geweke_z(log_pi) if log_pi else None
        # Down-sample full traces if very long (to keep JSON size sane).
        # Store every-1 up to 5000 then every-K. 500 sweeps fits trivially.
        if len(log_pi) <= 5000:
            entry['log_pi_trace'] = log_pi
            entry['n_edges_trace'] = n_edges
            entry['n_match_trace'] = n_match
        else:
            k = (len(log_pi) + 4999) // 5000
            entry['log_pi_trace'] = log_pi[::k]
            entry['n_edges_trace'] = n_edges[::k]
            entry['n_match_trace'] = n_match[::k]
            entry['_trace_downsample_stride'] = int(k)
        per_entries.append(entry)

    # r-hat across chains (only meaningful for the chain ensemble).
    rhat_match = None
    rhat_log_pi = None
    if not is_replica_exchange:
        n_match_chains = [getattr(d, 'n_match_trace', []) for d in per_chain]
        log_pi_chains = [getattr(d, 'log_pi_trace', []) for d in per_chain]
        if all(len(c) >= 2 for c in n_match_chains):
            rhat_match = rhat_gelman_rubin(n_match_chains)
        if all(len(c) >= 2 for c in log_pi_chains):
            rhat_log_pi = rhat_gelman_rubin(log_pi_chains)

    # Between-chain mean spreads (Q' summary statistic).
    between_chain_n_match_sd = None
    between_chain_log_pi_sd = None
    if len(per_chain) >= 2:
        ms_n = [np.mean(getattr(d, 'n_match_trace', [])) for d in per_chain
                if len(getattr(d, 'n_match_trace', []))]
        ms_lp = [np.mean(getattr(d, 'log_pi_trace', [])) for d in per_chain
                 if len(getattr(d, 'log_pi_trace', []))]
        if len(ms_n) >= 2:
            between_chain_n_match_sd = float(np.std(ms_n, ddof=1))
        if len(ms_lp) >= 2:
            between_chain_log_pi_sd = float(np.std(ms_lp, ddof=1))

    out = {
        'is_replica_exchange': bool(is_replica_exchange),
        'n_chains': int(len(per_chain)),
        'per_chain': per_entries,
        'r_hat_n_match': rhat_match,
        'r_hat_log_pi': rhat_log_pi,
        'between_chain_n_match_sd': between_chain_n_match_sd,
        'between_chain_log_pi_sd': between_chain_log_pi_sd,
    }
    if alpha_z_ladder is not None:
        out['alpha_z_ladder'] = [float(a) for a in alpha_z_ladder]
    if swap_n_propose is not None:
        out['swap_n_propose'] = [int(x) for x in swap_n_propose]
    if swap_n_accept is not None:
        out['swap_n_accept'] = [int(x) for x in swap_n_accept]
        # Compute per-rung-pair swap acceptance rates.
        if swap_n_propose is not None:
            out['swap_acc_rates'] = [
                (float(a) / max(1.0, float(p)))
                for a, p in zip(swap_n_accept, swap_n_propose)
            ]
    if Q_chain_var is not None:
        Q_chain_var = np.asarray(Q_chain_var)
        if Q_chain_var.size > 0:
            out['Q_chain_var_mean'] = float(Q_chain_var.mean())
            out['Q_chain_var_max'] = float(Q_chain_var.max())
            out['Q_chain_sd_mean'] = float(np.sqrt(Q_chain_var.mean()))
            out['Q_chain_sd_max'] = float(np.sqrt(Q_chain_var.max()))
    if q_l1_vs_baseline is not None:
        out['q_l1_vs_baseline'] = float(q_l1_vs_baseline)
    if rung_traj is not None and alpha_z_ladder is not None:
        K = len(alpha_z_ladder)
        out['round_trip'] = round_trip_stats(rung_traj, K)
    if re_replicate_traces is not None and len(re_replicate_traces) >= 2:
        # r-hat across the cold-rung n_match_traces of independent RE chains.
        out['rhat_n_match_re_replicates'] = rhat_gelman_rubin(
            re_replicate_traces)
        out['n_re_replicates'] = int(len(re_replicate_traces))
    return out
