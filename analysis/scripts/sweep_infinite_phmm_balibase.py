"""Run the infinite Pair HMM (O(L^4)) MCMC sampler over the small
BAliBase pairwise families and write per-pair posterior-F1 sufficient
statistics for Table 1 in the main paper.

Eligibility filter: BAliBase ``bali3pdbm/in`` families with
max_seq_length < ``--max-len`` (default 150; 22 families fit; see
analysis/mcmc_infinite_phmm.md and calibrate_infinite_phmm.py).

This script can run the sampler in three modes:

1. Single-chain (default ``--n-chains 1``, no ladder): one chain at the
   fixed ``--alpha-z``.

2. Multi-chain (``--n-chains N``, no ladder): N independent chains at
   the fixed ``--alpha-z``. The cold-rung Q' is the across-chain mean.
   r-hat / Gelman-Rubin diagnostics are well-defined here.

3. Replica exchange (``--alpha-z-ladder a1,a2,...``): K chains, one per
   rung, with adjacent-rung swap proposals every ``--swap-every`` sweeps.
   Cold rung = min(ladder). r-hat is meaningless across rungs (different
   targets) so the per-pair record reports per-rung ESS only.

The ``--top-rung-only`` flag is the convergence-validation switch: it
forces the run to use only ``max(alpha_z_ladder)`` (or ``--alpha-z``),
at which the bounded-eps prior pins |E|=0 and the sampler reduces to
pure-Gibbs alignment resampling under TKF92. The resulting Q' should
match the baseline FB posterior to within MCMC noise.

Per-pair output records gain a new ``mcmc_diag`` sub-dict containing:
  - per-chain traces (log_pi_trace, n_edges_trace, n_match_trace)
  - per-chain ESS on n_match and log_pi (Geyer-IACT)
  - Gelman-Rubin r-hat across chains (multi-chain only)
  - acceptance rates per move type
  - q_l1_vs_baseline = ||Q' - Q_baseline||_1 / n_cells

Usage (GPU 1):

  Single-chain (legacy):
    CUDA_VISIBLE_DEVICES=1 python sweep_infinite_phmm_balibase.py

  Multi-chain:
    CUDA_VISIBLE_DEVICES=1 python sweep_infinite_phmm_balibase.py \\
        --n-chains 4 --max-len 150

  Replica exchange (default ladder 100,250,700,2000,1e4 is the v2
  ladder tuned to give cold-end swap acceptance ~0.3):
    CUDA_VISIBLE_DEVICES=1 python sweep_infinite_phmm_balibase.py \\
        --swap-every 10 --max-len 150

  Top-rung validation:
    CUDA_VISIBLE_DEVICES=1 python sweep_infinite_phmm_balibase.py \\
        --top-rung-only --alpha-z-ladder 1e6 --n-chains 4 \\
        --n-sweeps 500 --n-burnin 100 --max-len 150 \\
        --out math-paper/results/infinite_phmm_balibase_k4_top_rung_validation.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]    # ~/tkf-dp
TKFMIXDOM = Path.home() / "tkf-mixdom" / "python"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(TKFMIXDOM))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _mcmc_diagnostics

BALI_ROOT = Path.home() / "bio-datasets" / "data" / "balibase" / "bali3pdbm"

MAX_LEN = 150          # eligibility filter (raised from 120 -- the
                       # current 11 GiB GPU comfortably handles up to
                       # L<150 in dense / lex-compressed mode; see
                       # analysis/calibration_infinite_phmm.md).
DEFAULT_NSWEEPS = 8000
DEFAULT_NBURNIN = 2000
DEFAULT_ALPHAZ = 100.0
# Re-tuned ladder per analysis/re_diag/REPORT.md (commit 51c1d58): a uniform
# ~1.78x geometric ratio for the bottom four rungs gives ~30%+ swap acceptance
# at every adjacent-rung pair; the final 1000->5000 hot step (ratio 5) trades
# some swap acceptance for a meaningful hot-end. Verified on the bali3pdbm
# L<150 corpus (187 pairs, 4 replicates, 8000 sweeps) to give
# cross-replicate r-hat median 1.002, max 1.141.
DEFAULT_ALPHAZ_LADDER = '100,178,316,562,1000,5000'


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
                    out.append((name, arr, s))
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name is not None:
            s = ''.join(seq)
            arr = np.array([AA_TO_INT.get(c.upper(), 20) for c in s
                            if c.isalpha()], dtype=np.int32)
            out.append((name, arr, s))
    return out


def parse_ref(path: Path):
    """Parse a BAliBase .ref alignment.

    Returns the alignment as ``{name: gapped_str}`` so that
    ``expected_pair_f1.ref_to_pair_truth`` can consume it.  Convert dots
    to dashes for gap; preserve case (uppercase = core, lowercase =
    insert).
    """
    out = {}
    name = None
    chunks = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    out[name] = ''.join(chunks).replace('.', '-')
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            out[name] = ''.join(chunks).replace('.', '-')
    return out


def find_eligible_families(max_len: int = MAX_LEN):
    """Return list of (family_name, n_seqs, max_seq_len) for all
    BAliBase bali3pdbm families with max_seq_len < max_len AND a
    corresponding .ref file."""
    in_dir = BALI_ROOT / "in"
    ref_dir = BALI_ROOT / "ref"
    elig = []
    for fa in sorted(in_dir.iterdir()):
        if not fa.is_file():
            continue
        ref = ref_dir / fa.name
        if not ref.exists():
            continue
        # Cheap scan: count seqs + track longest.
        n_seqs = 0
        max_l = 0
        cur = 0
        with open(fa) as fh:
            for line in fh:
                if line.startswith('>'):
                    n_seqs += 1
                    if cur > max_l:
                        max_l = cur
                    cur = 0
                else:
                    cur += sum(1 for c in line.strip() if c.isalpha())
            if cur > max_l:
                max_l = cur
        if max_l < max_len and n_seqs >= 2:
            elig.append((fa.name, n_seqs, max_l))
    return elig


def build_stub_state():
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    from tkfdp.potts_dp import PottsDPState
    _, pi_lg = rate_matrix_lg()
    pi_lg = np.asarray(pi_lg)
    K_c = 1
    A = 20
    pi_class = pi_lg[None, :]
    atoms = np.zeros((1, A, A), dtype=np.float32)
    assignments = np.zeros((K_c, K_c), dtype=np.int64)
    counts = np.array([1], dtype=np.int64)
    potts_dp = PottsDPState(K_c=K_c, A=A, atoms=atoms,
                            assignments=assignments, counts=counts,
                            alpha_H=1.0)

    class _State:
        pass
    s = _State()
    s.K_c = K_c
    s.A = A
    s.pi_class = pi_class
    s.potts_dp = potts_dp
    return s


def build_k4_state(ckpt_path: str, alpha_H: float = 1.0):
    """Load the released K=4 EM-warmup checkpoint into a State suitable
    for the sampler. The training-time per-MSA partition/clique latents
    are dropped; the new family's MCMC samples its own partition from
    scratch under the loaded Potts atoms + class profiles."""
    from tkfdp.potts_dp import PottsDPState
    d = np.load(ckpt_path, allow_pickle=True)
    pi_class = np.asarray(d['pi_class'], dtype=np.float32)
    atoms = np.asarray(d['potts_atoms'], dtype=np.float32)
    assignments = np.asarray(d['potts_assignments'], dtype=np.int64)
    counts = np.asarray(d['potts_counts'], dtype=np.int64)
    K_c, A = pi_class.shape
    potts_dp = PottsDPState(K_c=K_c, A=A, atoms=atoms,
                            assignments=assignments, counts=counts,
                            alpha_H=alpha_H)

    class _State:
        pass
    s = _State()
    s.K_c = K_c
    s.A = A
    s.pi_class = pi_class
    s.potts_dp = potts_dp
    return s


def _parse_alpha_z_ladder(s):
    """Parse a comma-separated alpha_z ladder string. ``None`` or empty
    string disables RE. The default in argparse is the v2 ladder
    ``"100,250,700,2000,1e4"``."""
    if s is None or not s.strip():
        return None
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [float(p) for p in parts]


def _run_one_pair(x_seq, y_seq, t, ins_rate, del_rate, ext,
                   Q_lg, pi_lg, bs,
                   alpha_z, alpha_z_ladder, n_chains, swap_every,
                   n_sweeps, n_burnin, seed,
                   prepop_top_k=-1, prepop_chunk=256,
                   prepop_mem_budget_mib=2048.0):
    """Dispatch a single pair to the right sampler entry point.

    Returns (Q_prime, Q_baseline, log_F0, diag_obj, mode_str)
    where mode_str is one of 'replica_exchange', 'multi_chain', 'single_chain'.
    """
    from tkfdp.mcmc_infinite_phmm import mcmc_corrected_posterior

    common = dict(
        x_seq=x_seq, y_seq=y_seq, t=t,
        ins_rate=ins_rate, del_rate=del_rate, ext=ext,
        Q_lg=Q_lg, pi_lg=pi_lg, boost_state=bs,
        alpha_z=alpha_z,
        n_sweeps=n_sweeps, n_burnin=n_burnin,
        prepop_top_k=prepop_top_k, prepop_chunk=prepop_chunk,
        prepop_mem_budget_mib=prepop_mem_budget_mib,
    )

    if alpha_z_ladder is not None and len(alpha_z_ladder) > 1:
        mode = 'replica_exchange'
        Q_prime, _, Q_baseline, log_F0, diag = mcmc_corrected_posterior(
            **common,
            n_chains=n_chains, k_max=-1, seed=seed,
            alpha_z_ladder=alpha_z_ladder, swap_every=swap_every,
        )
    elif n_chains > 1:
        mode = 'multi_chain'
        Q_prime, _, Q_baseline, log_F0, diag = mcmc_corrected_posterior(
            **common,
            n_chains=n_chains, k_max=-1, seed=seed,
        )
    else:
        mode = 'single_chain'
        Q_prime, _, Q_baseline, log_F0, diag = mcmc_corrected_posterior(
            **common,
            n_chains=1, k_max=-1, seed=seed,
        )
    return Q_prime, Q_baseline, log_F0, diag, mode


def _bundle_diag(diag_obj, mode, Q_prime, Q_baseline,
                  re_replicate_traces=None):
    """Turn whatever the sampler returned (single-chain
    MCMCDiagnostics OR multi-chain dict OR replica-exchange dict) into
    a JSON-friendly diagnostics record."""
    from _mcmc_diagnostics import diags_to_json
    Q_prime = np.asarray(Q_prime)
    Q_baseline = np.asarray(Q_baseline)
    if Q_prime.size > 0 and Q_baseline.size > 0 and Q_prime.shape == Q_baseline.shape:
        q_l1 = float(np.abs(Q_prime - Q_baseline).mean())
    else:
        q_l1 = None

    if mode == 'replica_exchange':
        d = diag_obj  # dict
        return diags_to_json(
            per_chain=d.get('per_rung', []),
            alpha_z_ladder=d.get('alpha_z_ladder'),
            swap_n_propose=d.get('swap_n_propose'),
            swap_n_accept=d.get('swap_n_accept'),
            rung_traj=d.get('rung_traj'),
            Q_chain_var=None,
            q_l1_vs_baseline=q_l1,
            is_replica_exchange=True,
            re_replicate_traces=re_replicate_traces,
        )
    elif mode == 'multi_chain':
        d = diag_obj  # dict
        return diags_to_json(
            per_chain=d.get('per_chain', []),
            alpha_z_ladder=None,
            Q_chain_var=d.get('Q_chain_var'),
            q_l1_vs_baseline=q_l1,
            is_replica_exchange=False,
        )
    else:  # single_chain -> diag_obj is MCMCDiagnostics-like
        # mcmc_corrected_posterior with n_chains=1 still goes through
        # run_mcmc_multi_chain -> returns a dict; check.
        if isinstance(diag_obj, dict):
            return diags_to_json(
                per_chain=diag_obj.get('per_chain', []),
                Q_chain_var=diag_obj.get('Q_chain_var'),
                q_l1_vs_baseline=q_l1,
                is_replica_exchange=False,
            )
        return diags_to_json(
            per_chain=[diag_obj],
            q_l1_vs_baseline=q_l1,
            is_replica_exchange=False,
        )


def run_family(family: str, n_sweeps: int, n_burnin: int,
               alpha_z: float,
               n_chains: int = 1,
               alpha_z_ladder=None,
               swap_every: int = 10,
               verbose: bool = True,
               state=None,
               method_label: str = 'infinite_phmm_mcmc_K1_stub',
               ins_rate: float = 0.02,
               del_rate: float = 0.05,
               ext: float = 0.5,
               n_re_replicates: int = 1,
               cache_method_name: str | None = None,
               cache_params_key: str | None = None,
               prepop_top_k: int = -1,
               prepop_chunk: int = 256,
               prepop_mem_budget_mib: float = 2048.0,
               pair_subset=None,
               checkpoint_path: str | None = None,
               seed_base: int = 0):
    """Run the MCMC sampler on every pair of a family.  Each pair runs
    independently; we do not parallelise across pairs (the per-pair
    cache fits but the sum-across-pairs does not on a single GPU).

    Returns a per-pair list of expected-F1 sufficient stats.
    """
    sys.path.insert(0, str(TKFMIXDOM / "tkfmixdom" / "util"))
    from expected_pair_f1 import expected_pair_f1, ref_to_pair_truth
    from tkfdp.coupled_annealing import build_boost_state
    from tkfmixdom.jax.core.protein import rate_matrix_lg

    fa_path = BALI_ROOT / "in" / family
    ref_path = BALI_ROOT / "ref" / family
    fasta = parse_fasta(fa_path)
    if len(fasta) < 2:
        return {'family': family, 'skipped': 'fewer than 2 seqs'}
    ref_aln = parse_ref(ref_path)

    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg = np.asarray(Q_lg); pi_lg = np.asarray(pi_lg)

    if state is None:
        state = build_stub_state()

    rows = []
    names = [name for name, _, _ in fasta]
    seq_arrs = [arr for _, arr, _ in fasta]

    # ---- Cache-skip fast path. ----
    # If this family already has a complete Q'-cache entry under the
    # current params_key, skip the sampler entirely. We rebuild the
    # per-pair F1 sufficient stats from the loaded Q' + ref alignment.
    # No MCMC diagnostics are produced for cache-hit pairs (mcmc_mode =
    # 'cached'); downstream FSA / corpus aggregation uses the soft suff
    # stats which ARE in the row.
    if (cache_method_name is not None and cache_params_key is not None):
        try:
            from tkfmixdom.util import balibase_pair_cache as ppcache
            loaded = ppcache.load(cache_method_name, family, cache_params_key)
        except Exception as e:
            loaded = None
            print(f"   warn: cache lookup failed ({e}); running MCMC",
                  flush=True)
        if loaded is not None:
            pair_post_cached, kind, _failed = loaded
            print(f"   CACHE HIT: {len(pair_post_cached)} Q' arrays loaded; "
                  f"skipping MCMC for {family}", flush=True)
            for (i, j), Q_p in pair_post_cached.items():
                name_x = names[i]; name_y = names[j]
                if name_x not in ref_aln or name_y not in ref_aln:
                    continue
                truth = ref_to_pair_truth(
                    ref_aln, name_x, name_y, core_only=True)
                row = expected_pair_f1(np.asarray(Q_p), truth)
                row['pair'] = (i, j)
                row['name_i'] = name_x
                row['name_j'] = name_y
                row['len_i'] = int(len(seq_arrs[i]))
                row['len_j'] = int(len(seq_arrs[j]))
                row['mcmc_time_s'] = 0.0
                row['mcmc_mode'] = 'cached'
                row['method'] = method_label
                row['mcmc_diag'] = {'cached': True}
                rows.append(row)
                if verbose:
                    print(f"   pair ({name_x}, {name_y}) "
                          f"L=({len(seq_arrs[i])},{len(seq_arrs[j])}) "
                          f"CACHED eTP={row['e_tp']:.2f} "
                          f"totMass={row['total_mass']:.2f} "
                          f"gold={row['gold']}", flush=True)
            return {'family': family, 'per_pair': rows,
                    'n_seqs': len(names),
                    'cached': True}

    # Build all pair-boost-states. Use the TKF92 baseline JAX FB
    # to obtain (a) the per-pair Newton-Raphson-optimised branch
    # length tau_opt (so the FB posterior the sampler corrects
    # against matches the cached tkf92_lg08 baseline) and (b) the
    # initial pair-match posterior, which is a much better starting
    # point for the boost than the 0.5 prior.
    import jax
    import jax.numpy as jnp
    from tkfmixdom.jax.tree.fsa_anneal import _pairwise_posteriors_tkf92_jax
    from tkfmixdom.jax.dp.hmm import _pad_to_bin, _pad_seq
    pair_post = {}
    pair_taus = {}
    t0_tau = time.time()
    for i, j in combinations(range(len(names)), 2):
        x = jnp.asarray(seq_arrs[i], dtype=jnp.int32)
        y = jnp.asarray(seq_arrs[j], dtype=jnp.int32)
        Lx, Ly = int(x.shape[0]), int(y.shape[0])
        Lx_pad, Ly_pad = _pad_to_bin(Lx), _pad_to_bin(Ly)
        x_pad, y_pad = _pad_seq(x, Lx_pad), _pad_seq(y, Ly_pad)
        mp_pad, tau_opt, _ = _pairwise_posteriors_tkf92_jax(
            x_pad, y_pad, jnp.int32(Lx), jnp.int32(Ly),
            jnp.float64(ins_rate), jnp.float64(del_rate),
            jnp.float64(ext),
            jnp.asarray(Q_lg), jnp.asarray(pi_lg))
        pair_post[(i, j)] = np.asarray(mp_pad)[:Lx, :Ly].astype(np.float32)
        pair_taus[(i, j)] = float(tau_opt)
    dt_tau = time.time() - t0_tau
    tau_vals = list(pair_taus.values())
    print(f"   tau optimisation ({len(pair_taus)} pairs): {dt_tau:.2f}s, "
          f"tau range [{min(tau_vals):.3f}, {max(tau_vals):.3f}], "
          f"mean {sum(tau_vals)/len(tau_vals):.3f}", flush=True)
    t0_bs = time.time()
    # Compute / load the empirical class prior pi_c from the K=4 checkpoint.
    # The MCMC chain (block_likelihoods.build_M_tensor + build_singlet_emission)
    # uses pi_c to weight the class-mixed singleton AND class-pair-mixed
    # doublet emissions -- canonical for THIS K=4 emwarm release.
    if checkpoint_path and Path(checkpoint_path).exists():
        from tkfdp.block_likelihoods import empirical_pi_c_from_checkpoint
        try:
            pi_c_emp = empirical_pi_c_from_checkpoint(checkpoint_path)
            print(f"   empirical pi_c from {checkpoint_path}: "
                  f"{[f'{x:.3f}' for x in pi_c_emp]}", flush=True)
        except Exception as e:
            print(f"   warn: could not load empirical pi_c ({e}); "
                  f"falling back to uniform 1/K_c", flush=True)
            pi_c_emp = None
    else:
        pi_c_emp = None
    boost_states = build_boost_state(
        pair_post, pair_taus, seq_arrs, state,
        pi_c=pi_c_emp, pair_background='lg08')
    dt_bs = time.time() - t0_bs
    print(f"   build_boost_state ({len(boost_states)} pairs): {dt_bs:.2f}s",
          flush=True)

    # Optional: Holmes-Durbin optimal-accuracy indicator for the
    # second variant.
    try:
        sys.path.insert(0, str(TKFMIXDOM / "experiments"))
        from expected_pairwise_balibase import _optimal_accuracy_indicator
        have_opt_acc = True
    except Exception as e:
        print(f"   warn: cannot import _optimal_accuracy_indicator ({e}); "
              f"skipping Holmes-Durbin variant", flush=True)
        have_opt_acc = False

    # Accumulator for per-pair Q' arrays we'll write to the central
    # pair-posterior cache at end-of-family. Keyed by (i, j); the cache
    # save serialises every key as a separate ``post_<k>`` slot.
    qprime_by_pair: dict = {}

    # Optional per-pair filter for AWS-style "one job per pair" workers.
    # pair_subset is an iterable of (i, j) tuples to include; everything
    # else is skipped. None = no filter (all pairs).
    if pair_subset is not None:
        wanted = {tuple(p) for p in pair_subset}
        boost_states = {k: v for k, v in boost_states.items() if k in wanted}
        if verbose:
            print(f"   --pair-subset filter: keeping {len(boost_states)} of "
                  f"{len(wanted)} requested pairs", flush=True)
    for (i, j), bs in boost_states.items():
        name_x = names[i]; name_y = names[j]
        x_seq = seq_arrs[i]
        y_seq = seq_arrs[j]
        t = float(pair_taus[(i, j)])
        if name_x not in ref_aln or name_y not in ref_aln:
            print(f"   pair ({name_x}, {name_y}): not in ref, skip",
                  flush=True)
            continue
        t0 = time.time()
        # Loop over RE replicates with distinct seeds. n_re_replicates=1
        # is the legacy single-run path. With n_re_replicates >= 2 we
        # also collect cold-rung n_match_traces across replicates so
        # diags_to_json can compute the cross-replicate r-hat.
        n_rep = max(1, int(n_re_replicates))
        replicate_results = []
        for rep in range(n_rep):
            Q_p, Q_b, log_F0, diag_obj, mode = _run_one_pair(
                x_seq=x_seq, y_seq=y_seq, t=t,
                ins_rate=ins_rate, del_rate=del_rate, ext=ext,
                Q_lg=Q_lg, pi_lg=pi_lg, bs=bs,
                alpha_z=alpha_z, alpha_z_ladder=alpha_z_ladder,
                n_chains=n_chains, swap_every=swap_every,
                n_sweeps=n_sweeps, n_burnin=n_burnin,
                seed=int(seed_base) + rep,
                prepop_top_k=prepop_top_k, prepop_chunk=prepop_chunk,
                prepop_mem_budget_mib=prepop_mem_budget_mib,
            )
            replicate_results.append((np.asarray(Q_p), np.asarray(Q_b),
                                       diag_obj, mode))
        dt = time.time() - t0
        # Average Q' across replicates (each is the within-run cold-rung
        # mean; averaging reduces between-replicate Monte-Carlo noise).
        Q_prime_np = np.mean([r[0] for r in replicate_results], axis=0)
        Q_baseline_np = replicate_results[0][1]
        # Cold-rung n_match_traces across replicates (replica_exchange or
        # multi-chain). Used downstream by diags_to_json for r-hat.
        re_replicate_traces = None
        if n_rep >= 2:
            re_replicate_traces = []
            for _, _, dx, m in replicate_results:
                if m == 'replica_exchange':
                    pr = (dx.get('per_rung') or [None])[0]
                    if pr is not None:
                        re_replicate_traces.append(
                            list(getattr(pr, 'n_match_trace', [])))
                elif m == 'multi_chain':
                    chains = dx.get('per_chain', [])
                    if chains:
                        re_replicate_traces.append(
                            list(getattr(chains[0], 'n_match_trace', [])))
                else:
                    re_replicate_traces.append(
                        list(getattr(dx, 'n_match_trace', [])))
            if not any(re_replicate_traces):
                re_replicate_traces = None
        # Use the first replicate's diagnostics dict for the primary
        # record. The cross-replicate r-hat is added via
        # _bundle_diag(..., re_replicate_traces=...).
        diag_obj = replicate_results[0][2]
        mode = replicate_results[0][3]
        truth = ref_to_pair_truth(ref_aln, name_x, name_y, core_only=True)

        # Stash Q' for the end-of-family cache write.
        qprime_by_pair[(i, j)] = Q_prime_np.astype(np.float32, copy=False)

        # Per-pair Q' soft-F1 record.
        row = expected_pair_f1(Q_prime_np, truth)
        row['pair'] = (i, j)
        row['name_i'] = name_x
        row['name_j'] = name_y
        row['len_i'] = int(len(x_seq))
        row['len_j'] = int(len(y_seq))
        row['mcmc_time_s'] = float(dt)
        row['mcmc_mode'] = mode
        row['method'] = method_label

        # Baseline (TKF92 FB posterior) soft-F1 record for comparison.
        try:
            base_row = expected_pair_f1(Q_baseline_np, truth)
            row['baseline_e_tp'] = float(base_row['e_tp'])
            row['baseline_total_mass'] = float(base_row['total_mass'])
        except Exception as e:
            print(f"   warn: baseline_e_tp failed: {e}", flush=True)

        # Holmes-Durbin optimal-accuracy variant on Q'.
        if have_opt_acc:
            ind = _optimal_accuracy_indicator(Q_prime_np)
            opt_row = expected_pair_f1(ind, truth)
            row['opt_acc_e_tp'] = opt_row['e_tp']
            row['opt_acc_total_mass'] = opt_row['total_mass']

        # MCMC diagnostics.
        try:
            row['mcmc_diag'] = _bundle_diag(diag_obj, mode,
                                              Q_prime_np, Q_baseline_np,
                                              re_replicate_traces=re_replicate_traces)
            row['mcmc_diag']['n_re_replicates'] = n_rep
        except Exception as e:
            row['mcmc_diag'] = {'_diag_error': str(e)[:300]}

        rows.append(row)
        if verbose:
            ess_match = (row.get('mcmc_diag', {}).get('per_chain') or [{}])[0]\
                .get('ess_n_match')
            q_l1 = row.get('mcmc_diag', {}).get('q_l1_vs_baseline')
            rhat = row.get('mcmc_diag', {}).get('r_hat_n_match')
            print(f"   pair ({name_x}, {name_y}) "
                  f"L=({len(x_seq)},{len(y_seq)}) "
                  f"t={dt:.1f}s "
                  f"eTP={row['e_tp']:.2f} totMass={row['total_mass']:.2f} "
                  f"gold={row['gold']} "
                  f"mode={mode} "
                  f"ESS_M={ess_match if ess_match is None else f'{ess_match:.0f}'} "
                  f"r_hat={rhat if rhat is None else f'{rhat:.3f}'} "
                  f"|Q-Qbase|={q_l1 if q_l1 is None else f'{q_l1:.4f}'}",
                  flush=True)

    # End-of-family Q' cache write. Lands at the standard pair-posterior
    # cache path so downstream FSA / opt-acc workflows can reload these
    # via tkfmixdom.util.balibase_pair_cache.load and treat them
    # interchangeably with the cached tkf92_lg08 / mixdom_d3f1 posteriors.
    if (cache_method_name is not None and cache_params_key is not None
            and qprime_by_pair):
        try:
            from tkfmixdom.util import balibase_pair_cache as ppcache
            ppcache.save(
                cache_method_name, family, names,
                qprime_by_pair, kind='soft',
                params_key=cache_params_key)
            cache_path = ppcache.cache_paths(cache_method_name, family)[0]
            print(f"   cached {len(qprime_by_pair)} Q' arrays -> "
                  f"{cache_path}", flush=True)
        except Exception as e:
            print(f"   warn: pair-posterior cache save failed: {e}",
                  flush=True)

    return {'family': family, 'per_pair': rows,
            'n_pairs_done': len(rows),
            'n_pairs_skipped': max(0, len(boost_states) - len(rows))}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to a TKF-DP state.npz (the released K=4 '
                   'checkpoint at results/K4-emwarm-top1000-2026-05-09/'
                   '_best_chkpt/state.npz). If omitted, stub K=1 state.')
    p.add_argument('--alpha-H', type=float, default=1.0)
    p.add_argument('--n-sweeps', type=int, default=DEFAULT_NSWEEPS)
    p.add_argument('--n-burnin', type=int, default=DEFAULT_NBURNIN)
    p.add_argument('--alpha-z', type=float, default=DEFAULT_ALPHAZ,
                   help='Single-rung alpha_z (used unless --alpha-z-ladder '
                        'is given, in which case the ladder takes precedence).')
    p.add_argument('--n-chains', type=int, default=1,
                   help='Number of independent chains. With >1, the per-cell '
                        'Q\' is the across-chain mean; r-hat is computable. '
                        'Ignored when --alpha-z-ladder is given.')
    p.add_argument('--alpha-z-ladder', type=str,
                   default=DEFAULT_ALPHAZ_LADDER,
                   help='Comma-separated list of alpha_z values. Default '
                        f'"{DEFAULT_ALPHAZ_LADDER}" -- six-rung ladder '
                        'with uniform ~1.78x ratio at the bottom 4 rungs, '
                        'verified on BAliBase L<150 (187 pairs, 4 reps, '
                        '8000 sweeps) to give per-adjacent-rung swap '
                        'acceptance >=30%% (median r-hat 1.002, max 1.141). '
                        'See analysis/re_diag/REPORT.md for the validation. '
                        'Cold rung = min. Pass "" to disable RE.')
    p.add_argument('--swap-every', type=int, default=10,
                   help='Replica-exchange swap-proposal frequency in sweeps.')
    p.add_argument('--prepop-top-k', type=int, default=-1,
                   help='Pre-populate the mu_cache with restart-Forward at '
                        'the top-K Match cells by baseline F-B posterior. '
                        'Default -1 = ceil(L^{3/2}). Pass 0 to disable.')
    p.add_argument('--prepop-chunk', type=int, default=256,
                   help='vmap chunk size for the batched prepop call. '
                        'Caps peak GPU memory.')
    p.add_argument('--prepop-mem-budget-mib', type=float, default=2048.0,
                   help='Hard cap on prepop cache memory per pair (MiB). '
                        'Default 2048 = 2 GiB. With 4 procs/2 GPUs at '
                        'L=150, each cache entry ~720 KB so 2 GiB ~= 2900 '
                        'anchors > L^{3/2}~1800.')
    p.add_argument('--top-rung-only', action='store_true',
                   help='Convergence-validation switch: collapse the ladder '
                        'to a single rung at max(ladder) (or just --alpha-z '
                        'if no ladder), use --n-chains chains. At large alpha_z '
                        'the sampler reduces to pure-Gibbs alignment '
                        'resampling under TKF92 and Q\' should match the FB '
                        'baseline within MCMC noise.')
    p.add_argument('--max-len', type=int, default=MAX_LEN)
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--label', type=str, default=None,
                   help='Suffix tag for the per-pair method label. '
                   'Default infers from checkpoint vs stub.')
    p.add_argument('--ins-rate', type=float, default=None,
                   help='TKF92 insertion rate. Default reads from '
                   '--tkf92-params-json (corpus-fitted Maraschino).')
    p.add_argument('--del-rate', type=float, default=None,
                   help='TKF92 deletion rate; same default policy.')
    p.add_argument('--ext', type=float, default=None,
                   help='TKF92 fragment-extension probability r; '
                   'same default policy.')
    p.add_argument('--tkf92-params-json', type=str,
                   default=str(TKFMIXDOM / 'experiments'
                                / 'tkf92_fitted_params.json'),
                   help='Path to corpus-fitted TKF92 params JSON; used '
                   'unless --ins-rate / --del-rate / --ext override.')
    p.add_argument('--mcmc-diagnostics', action='store_true',
                   help='Enable full MCMC convergence diagnostics: '
                   'rung-trajectory recording (round-trip stats), '
                   'Geweke within-chain z-scores, swap acceptance '
                   'rates per adjacent rung, and multi-replicate RE '
                   'cross-replicate r-hat. Implies '
                   '--n-re-replicates 4 unless overridden.')
    p.add_argument('--n-re-replicates', type=int, default=None,
                   help='Number of independent sampler replicates per '
                   'pair (each at a different seed). >= 2 enables '
                   'cross-replicate r-hat. Default: 4 if '
                   '--mcmc-diagnostics, else 1.')
    p.add_argument('--seed', type=int, default=0,
                   help='Base seed for the MCMC chain. Each replicate '
                   '(see --n-re-replicates) gets seed = seed_base + rep, '
                   'so reruns at distinct --seed values give independent '
                   'chains for the same (family, pair). Default 0 '
                   'matches the original canonical-sweep convention.')
    p.add_argument('--fam-subset', type=str, default=None,
                   help='Comma-separated list of BAliBASE family IDs to '
                   'process. When set, restricts the eligible-families '
                   'list to this subset (used for multi-GPU sweep '
                   'parallelism: each worker takes a disjoint subset).')
    p.add_argument('--pair-subset', type=str, default=None,
                   help='Per-pair filter: comma-separated list of "I,J" '
                   'tuples (within the current --fam-subset family) to '
                   'process. Skips all other pairs. Used for AWS-style '
                   '"one job per pair" workers. Example: '
                   '"--fam-subset BB12041 --pair-subset 0,1" runs only '
                   'the (seq0, seq1) pair of BB12041. Multiple pairs '
                   'can be passed: "--pair-subset 0,1;0,2;1,2".')
    args = p.parse_args()
    if args.n_re_replicates is None:
        args.n_re_replicates = 4 if args.mcmc_diagnostics else 1

    if args.ins_rate is None or args.del_rate is None or args.ext is None:
        fitted = json.loads(Path(args.tkf92_params_json).read_text())
        ins_rate = (args.ins_rate if args.ins_rate is not None
                     else float(fitted['ins_rate']))
        del_rate = (args.del_rate if args.del_rate is not None
                     else float(fitted['del_rate']))
        ext = (args.ext if args.ext is not None
                else float(fitted['ext_rate']))
        print(f"TKF92 indel rates (from {args.tkf92_params_json}): "
              f"ins={ins_rate:.5f}, del={del_rate:.5f}, ext={ext:.4f}",
              flush=True)
    else:
        ins_rate, del_rate, ext = args.ins_rate, args.del_rate, args.ext
        print(f"TKF92 indel rates (CLI overrides): "
              f"ins={ins_rate:.5f}, del={del_rate:.5f}, ext={ext:.4f}",
              flush=True)

    ladder = _parse_alpha_z_ladder(args.alpha_z_ladder)
    if args.top_rung_only:
        if ladder is not None and len(ladder) > 0:
            top = max(ladder)
        else:
            top = args.alpha_z
        ladder = None              # collapse to single rung
        alpha_z = float(top)
        print(f"--top-rung-only: using single rung alpha_z={alpha_z}, "
              f"n_chains={args.n_chains}", flush=True)
    else:
        alpha_z = args.alpha_z

    elig = find_eligible_families(max_len=args.max_len)
    print(f"Eligible families ({len(elig)}, max_len<{args.max_len}): "
          f"{[f for f, _, _ in elig]}", flush=True)
    elig.sort(key=lambda r: r[2])

    # Optional family-subset filter for parallel multi-GPU runs. The
    # --fam-subset flag takes a comma-separated list of family names;
    # the launcher then ONLY processes those (in eligibility order).
    if args.fam_subset:
        wanted = set(s.strip() for s in args.fam_subset.split(',') if s.strip())
        elig = [r for r in elig if r[0] in wanted]
        print(f"--fam-subset filter: processing {len(elig)} families: "
              f"{[f for f, _, _ in elig]}", flush=True)

    if args.checkpoint:
        state = build_k4_state(args.checkpoint, alpha_H=args.alpha_H)
        default_label = f'infinite_phmm_mcmc_K{state.K_c}_coupled'
        print(f"Loaded K=4 checkpoint from {args.checkpoint}: "
              f"K_c={state.K_c}, atoms={state.potts_dp.atoms.shape}",
              flush=True)
    else:
        state = build_stub_state()
        default_label = 'infinite_phmm_mcmc_K1_stub'
        print(f"Using stub state (K_c=1, zero Potts atom).", flush=True)

    # Method label suffix encodes the sampler regime so summary scripts
    # can distinguish single-chain vs replica-exchange runs.
    suffix = ''
    if ladder is not None:
        suffix = '_RE'
    elif args.n_chains > 1:
        suffix = f'_C{args.n_chains}'
    if args.top_rung_only:
        suffix += '_topRung'
    method_label = (args.label or default_label) + suffix

    # Pair-posterior cache key for end-of-family Q' write. Hashes the
    # files we depend on (checkpoint NPZ, tkf92 fitted params) plus the
    # sampler regime so re-runs at different ladders / sweep counts
    # don't collide.
    try:
        from tkfmixdom.util import balibase_pair_cache as ppcache
    except Exception:
        ppcache = None
    cache_method_name = None
    cache_params_key = None
    if ppcache is not None:
        cache_method_name = method_label
        cache_files = []
        if args.checkpoint:
            cache_files.append(args.checkpoint)
        if args.tkf92_params_json and (args.ins_rate is None
                                         or args.del_rate is None
                                         or args.ext is None):
            cache_files.append(args.tkf92_params_json)
        ladder_str = (','.join(f'{x:.6g}' for x in ladder)
                       if ladder is not None else 'none')
        extra = (f'alpha_z={alpha_z};ladder={ladder_str};'
                  f'n_chains={args.n_chains};n_sweeps={args.n_sweeps};'
                  f'n_burnin={args.n_burnin};'
                  f'ins={ins_rate:.6g};del={del_rate:.6g};ext={ext:.6g};'
                  f'n_re_replicates={args.n_re_replicates}')
        cache_params_key = ppcache.file_params_key(*cache_files,
                                                     extra=extra)
        print(f"Pair-posterior cache: method={cache_method_name}, "
              f"params_key={cache_params_key}", flush=True)

    out_path = Path(args.out) if args.out else (
        REPO / "math-paper" / "results" /
        ("infinite_phmm_balibase_k4.json" if args.checkpoint
         else "infinite_phmm_balibase.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Mode: n_chains={args.n_chains}, ladder={ladder}, "
          f"top_rung_only={args.top_rung_only}, alpha_z={alpha_z}, "
          f"n_sweeps={args.n_sweeps}, n_burnin={args.n_burnin}",
          flush=True)

    all_results = []
    n_pairs_total = 0
    for k, (family, n_seqs, max_l) in enumerate(elig):
        t0 = time.time()
        print(f"\n=== [{k + 1}/{len(elig)}] {family} (n={n_seqs}, "
              f"max_len={max_l})",
              flush=True)
        try:
            # Parse --pair-subset 'I,J' or 'I,J;I,J;...'
            pair_subset = None
            if args.pair_subset:
                pair_subset = []
                for tok in args.pair_subset.split(';'):
                    tok = tok.strip()
                    if not tok:
                        continue
                    i_str, j_str = tok.split(',')
                    pair_subset.append((int(i_str), int(j_str)))
            res = run_family(family,
                             n_sweeps=args.n_sweeps,
                             n_burnin=args.n_burnin,
                             alpha_z=alpha_z,
                             n_chains=args.n_chains,
                             alpha_z_ladder=ladder,
                             swap_every=args.swap_every,
                             state=state,
                             method_label=method_label,
                             ins_rate=ins_rate,
                             del_rate=del_rate,
                             ext=ext,
                             n_re_replicates=args.n_re_replicates,
                             cache_method_name=cache_method_name,
                             cache_params_key=cache_params_key,
                             prepop_top_k=args.prepop_top_k,
                             prepop_chunk=args.prepop_chunk,
                             prepop_mem_budget_mib=args.prepop_mem_budget_mib,
                             pair_subset=pair_subset,
                             checkpoint_path=args.checkpoint,
                             seed_base=args.seed)
            res['wall_time_s'] = time.time() - t0
            all_results.append(res)
            n_pairs_total += res.get('n_pairs_done', 0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({'family': family, 'error': str(e),
                                'wall_time_s': time.time() - t0})
        # Annotate top-level metadata to make the run self-describing.
        meta = {
            'method_label': method_label,
            'mcmc_config': {
                'n_chains': int(args.n_chains),
                'alpha_z_ladder': ladder,
                'alpha_z': float(alpha_z),
                'swap_every': int(args.swap_every),
                'top_rung_only': bool(args.top_rung_only),
                'n_sweeps': int(args.n_sweeps),
                'n_burnin': int(args.n_burnin),
                'max_len': int(args.max_len),
                'checkpoint': args.checkpoint,
            },
            'per_family': all_results,
        }
        out_path.write_text(json.dumps(meta, indent=2))
        print(f"  wrote {out_path} after {family}", flush=True)
        gc.collect()
    print(f"\n=== Done.  Total pairs: {n_pairs_total}.  "
          f"Output: {out_path}", flush=True)


if __name__ == '__main__':
    main()
