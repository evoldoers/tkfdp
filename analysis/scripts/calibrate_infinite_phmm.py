"""Single-pair calibration of the infinite Pair HMM (O(L^4)) sampler.

Runs the sampler on the smallest BAliBase pair and at three controlled
length budgets (L ~ 90, 120, 145), measuring wall-clock per sweep and
peak GPU memory.  Use this to verify Ada's documented limits in
analysis/mcmc_infinite_phmm.md (sections F.6 / G) before running the
full eligibility-list sweep in Tier 2d step 4.

Notes:
  * The released K=4 checkpoint (tag K4-emwarm-top1000-2026-05-09)
    lives only as a private GitHub Release artefact and is NOT
    available on this machine.  As a workaround this calibration
    constructs a STUB Potts state with K_c=1 (single class, LG08
    stationary, one zero Potts atom).  With K_c=1 the boost is
    effectively the identity, so the sampler reduces to the
    indel-only TKF92 baseline.  The TIMING and MEMORY profile is
    still informative -- the cost is dominated by the O(L^4)
    partial-Forward precompute which does not depend on the boost.
    Q_prime numbers from this calibration should NOT be interpreted
    as alignment posteriors; they are only validating the runtime
    cost profile.

Usage (GPU 1):
    CUDA_VISIBLE_DEVICES=1 python calibrate_infinite_phmm.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[2]    # ~/tkf-dp
TKFMIXDOM = Path.home() / "tkf-mixdom" / "python"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(TKFMIXDOM))


BALI_ROOT = Path.home() / "bio-datasets" / "data" / "balibase" / "bali3pdbm" / "in"

# Three calibration targets, picked from the < 150 max-len eligibility
# list.  We want enough variety in L to map out the memory profile.
CALIBRATION_FAMILIES = [
    ("BB12032",  66),   # tiniest family (5 seqs at L=66)
    ("BB11013", 101),   # mid-tier (5 seqs at L=101)
    # BB30025 (L=144) dropped from first-pass calibration -- the L=66
    # family already used 8.3 GiB of GPU memory (75% of 11 GiB) and
    # L=144 likely OOMs.  Add back manually if needed after we
    # understand why the memory is high at small L.
]


def _aa_to_int_dict():
    import string
    AA = "ACDEFGHIKLMNPQRSTVWY"
    d = {c: i for i, c in enumerate(AA)}
    for c in string.ascii_uppercase:
        d.setdefault(c, 20)
    return d


def parse_fasta(path: Path):
    AA_TO_INT = _aa_to_int_dict()
    out = []
    name = None
    seq = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    s = ''.join(seq)
                    arr = np.array([AA_TO_INT.get(c.upper(), 20) for c in s
                                    if c.isalpha()], dtype=np.int32)
                    out.append((name, arr))
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name is not None:
            s = ''.join(seq)
            arr = np.array([AA_TO_INT.get(c.upper(), 20) for c in s
                            if c.isalpha()], dtype=np.int32)
            out.append((name, arr))
    return out


@dataclass
class StubState:
    """Minimal stand-in for a trained TKF-DP state with K_c=1 and a
    single zero Potts atom.  Has the same public interface as the
    _MinimalState used by tests/test_balibase_postprocess.py.
    """
    K_c: int = 1
    A: int = 20
    pi_class: np.ndarray = field(default_factory=lambda: np.zeros((1, 20)))
    potts_dp: object = None


def build_stub_state() -> StubState:
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    from tkfdp.potts_dp import PottsDPState
    _, pi_lg = rate_matrix_lg()
    pi_lg = np.asarray(pi_lg)
    K_c = 1
    A = 20
    pi_class = pi_lg[None, :]                           # (1, A)
    atoms = np.zeros((1, A, A), dtype=np.float32)       # one zero atom
    assignments = np.zeros((K_c, K_c), dtype=np.int64)  # all -> atom 0
    counts = np.array([1], dtype=np.int64)
    potts_dp = PottsDPState(K_c=K_c, A=A, atoms=atoms,
                            assignments=assignments, counts=counts,
                            alpha_H=1.0)
    return StubState(K_c=K_c, A=A, pi_class=pi_class, potts_dp=potts_dp)


def select_first_pair(seqs):
    names = list(seqs.keys())
    return names[0], names[1]


def gpu_mem_mb():
    try:
        import jax
        # Sum of allocated bytes across all GPU devices.
        out = []
        for dev in jax.devices('gpu'):
            stats = dev.memory_stats()
            out.append({
                'device': str(dev),
                'bytes_in_use': stats.get('bytes_in_use', 0),
                'peak_bytes_in_use': stats.get('peak_bytes_in_use', 0),
                'pool_bytes': stats.get('pool_bytes', 0),
            })
        return out
    except Exception as e:
        return [{'error': str(e)}]


def run_one_pair(family: str, max_len: int, n_sweeps: int = 200,
                 n_burnin: int = 50, alpha_z: float = 100.0,
                 alpha_z_ladder=None, swap_every=10, verbose=True):
    from tkfdp.mcmc_infinite_phmm import mcmc_corrected_posterior
    from tkfdp.coupled_annealing import build_boost_state

    path = BALI_ROOT / family
    fasta = parse_fasta(path)
    print(f"=== Calibration: {family} (n={len(fasta)}, max_len={max_len})",
          flush=True)
    if len(fasta) < 2:
        raise RuntimeError(f"family {family} has < 2 sequences")
    # Pick the two longest seqs (closer to the calibration target L).
    fasta.sort(key=lambda kv: -len(kv[1]))
    (name_x, x_seq), (name_y, y_seq) = fasta[0], fasta[1]
    print(f"   pair {name_x}({len(x_seq)}) / {name_y}({len(y_seq)})",
          flush=True)

    state = build_stub_state()

    # Pre-compute the boost state for this pair.  We need a baseline
    # pair_posteriors dict for build_boost_state -- use a stand-in
    # value of 0.5 (it's overwritten by the sampler anyway).
    Lx, Ly = len(x_seq), len(y_seq)
    pair_post = {(0, 1): 0.5 * np.ones((Lx, Ly))}
    pair_taus = {(0, 1): 0.5}

    t0_boost = time.time()
    bs_all = build_boost_state(pair_post, pair_taus,
                                [np.asarray(x_seq), np.asarray(y_seq)],
                                state)
    bs = bs_all[(0, 1)]
    dt_boost = time.time() - t0_boost
    print(f"   build_boost_state: {dt_boost:.2f}s", flush=True)

    # Pick the LG Q/pi for the sampler.
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg = np.asarray(Q_lg); pi_lg = np.asarray(pi_lg)

    print(f"   GPU memory before sampler: {gpu_mem_mb()}", flush=True)

    t0_run = time.time()
    out = mcmc_corrected_posterior(
        x_seq=np.asarray(x_seq, dtype=np.int32),
        y_seq=np.asarray(y_seq, dtype=np.int32),
        t=0.5,
        ins_rate=0.02, del_rate=0.05, ext=0.5,
        Q_lg=Q_lg, pi_lg=pi_lg, boost_state=bs,
        alpha_z=alpha_z, n_sweeps=n_sweeps, n_burnin=n_burnin,
        n_chains=1, k_max=-1, seed=0,
        alpha_z_ladder=alpha_z_ladder, swap_every=swap_every,
        verbose=False,
    )
    Q_prime, _, Q_baseline, log_F0, diag = out
    dt_run = time.time() - t0_run

    peak_mem = gpu_mem_mb()
    print(f"   sampler: {dt_run:.2f}s for {n_sweeps} sweeps + {n_burnin} burn-in",
          flush=True)
    print(f"   per-sweep: {1000 * dt_run / (n_sweeps + n_burnin):.1f}ms",
          flush=True)
    print(f"   GPU memory after sampler: {peak_mem}", flush=True)

    return {
        'family': family, 'max_len_anchored': max_len,
        'name_x': name_x, 'len_x': int(Lx),
        'name_y': name_y, 'len_y': int(Ly),
        'n_sweeps': int(n_sweeps), 'n_burnin': int(n_burnin),
        'time_build_boost_s': float(dt_boost),
        'time_sampler_s': float(dt_run),
        'time_per_sweep_ms': float(1000 * dt_run / (n_sweeps + n_burnin)),
        'gpu_memory_after': peak_mem,
        'Q_prime_sum': float(np.asarray(Q_prime).sum()),
        'Q_baseline_sum': float(np.asarray(Q_baseline).sum()),
        'log_F0': float(log_F0),
        'mcmc_diag_keys': list(diag.keys()) if hasattr(diag, 'keys')
                          else None,
    }


def main():
    # First-pass calibration: single chain at alpha_z=100, no replica
    # ladder.  Replica exchange (4-rung) quadruples GPU memory at fixed
    # L and is reserved for the final sweep -- here we want a baseline
    # cost number we can extrapolate from.
    all_results = []
    for family, max_len in CALIBRATION_FAMILIES:
        try:
            res = run_one_pair(family, max_len,
                                n_sweeps=100, n_burnin=20,
                                alpha_z=100.0,
                                alpha_z_ladder=None,
                                swap_every=10)
            all_results.append(res)
            out_path = REPO / "analysis" / "calibration_infinite_phmm.json"
            out_path.write_text(json.dumps(all_results, indent=2))
            print(f"   intermediate write -> {out_path}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({
                'family': family, 'max_len_anchored': max_len,
                'error': str(e),
            })
        gc.collect()

    out_path = REPO / "analysis" / "calibration_infinite_phmm.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n=== Wrote {out_path}", flush=True)
    return all_results


if __name__ == '__main__':
    main()
