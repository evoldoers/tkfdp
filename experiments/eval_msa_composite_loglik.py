"""Empirical demo: composite log-likelihood of four candidate MSAs on BB11001.

For each of {baseline_fsa, MUSCLE, MAFFT, BAliBASE-reference}:
  - compute the SP / TC vs the BAliBASE 3 reference
  - compute the composite (cherry) log-likelihood under our trained model

Then cross-tabulate: does the model identify the BAliBASE reference as best?
Does the model's preferred MSA (max log-L) match the SP-best MSA?

Run:
  python experiments/eval_msa_composite_loglik.py [--bench BB11001] [--alpha-z 100]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)

# Paths.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TKFMIXDOM_ROOT = Path.home() / "tkf-mixdom" / "python"
sys.path.insert(0, str(TKFMIXDOM_ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tkfmixdom.jax.evaluate.metrics import sp_score, tc_score
from tkfmixdom.jax.util.io import AA_TO_INT, INT_TO_AA, read_fasta
from tkfmixdom.jax.tree.fsa_anneal import (
    compute_pairwise_posteriors, select_pairs_full,
)
from tkfmixdom.jax.core.protein import rate_matrix_lg

from test_balibase_postprocess import (
    load_minimal_state, _msa_from_col_assignments,
)

# Use the MULTI-START sequence_annealing wrapper from eval_balibase
# (deterministic, best-of-N).
from eval_balibase import sequence_annealing, load_seqs, load_ref

from tkfdp.coupled_annealing import build_boost_state
from tkfdp.composite_partition import composite_loglik_msa


# ---------------------------------------------------------------------------
# MSA loaders / runners.
# ---------------------------------------------------------------------------

def run_baseline_fsa_msa(seqs: dict) -> dict:
    """Baseline FSA: TKF92 + sequence_annealing on shared pair_post."""
    Q_lg, pi_lg = rate_matrix_lg()
    n = len(seqs); seq_lens = [len(seqs[k]) for k in seqs]
    names = list(seqs.keys())
    pairs = select_pairs_full(n)
    pair_post, _ = compute_pairwise_posteriors(
        seqs, pairs, model='tkf92', Q=Q_lg, pi=pi_lg)
    col_assignments, msa_length = sequence_annealing(
        n, seq_lens, dict(pair_post), n_iterations=5, verbose=False)
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def _run_external(prog: str, argv_template: list, seqs: dict,
                  mafft_stdout: bool = False) -> dict:
    """Run an external aligner on a temp FASTA, return parsed MSA dict."""
    if shutil.which(prog) is None:
        raise FileNotFoundError(
            f"{prog} not on PATH. Run scripts/fetch_aligners.sh.")
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.fa"; out_path = Path(td) / "out.fa"
        with open(in_path, "w") as f:
            for name, arr in seqs.items():
                seq = "".join(INT_TO_AA.get(int(i), "X") for i in arr)
                f.write(f">{name}\n{seq}\n")
        argv = [prog] + [a.format(**{"in": str(in_path), "out": str(out_path)})
                         for a in argv_template]
        if mafft_stdout:
            with open(out_path, "w") as f:
                subprocess.run(argv, stdout=f, check=True)
        else:
            subprocess.run(argv, check=True, capture_output=True)
        out = {}
        for name, seq in read_fasta(str(out_path)):
            row = np.full(len(seq), -1, dtype=np.int32)
            for k, c in enumerate(seq):
                if c not in '.-':
                    row[k] = AA_TO_INT.get(c.upper(), 20)
            out[name] = row
    return out


def run_muscle_msa(seqs: dict) -> dict:
    return _run_external("muscle", ["-align", "{in}", "-output", "{out}"], seqs)


def run_mafft_msa(seqs: dict) -> dict:
    return _run_external("mafft", ["--auto", "--quiet", "{in}"], seqs,
                         mafft_stdout=True)


# ---------------------------------------------------------------------------
# Reference loader (specialised for our purposes: keep all residues, drop
# distinguishing information about case for the loglik calc; keep case for
# scoring purposes).
# ---------------------------------------------------------------------------

def load_ref_msa_for_loglik(path: Path) -> dict:
    """Load the BAliBASE .ref alignment as a {name: row} dict where row[k]
    is the AA index 0..19 (or 20 for unknown) at column k, or -1 for gap.

    Lowercase residues (= unaligned-but-still-residues per BAliBASE
    convention) are kept as residues. The composite_loglik_msa scoring
    treats them just like uppercase.
    """
    msa = {}
    for name, seq in read_fasta(str(path)):
        row = np.full(len(seq), -1, dtype=np.int32)
        for k, c in enumerate(seq):
            if c == '.' or c == '-':
                row[k] = -1
            else:
                row[k] = AA_TO_INT.get(c.upper(), 20)
        msa[name] = row
    return msa


def msa_consistent_with_seqs(msa: dict, seqs: dict) -> bool:
    """Verify each MSA row's non-gap residues, in order, equal seqs[name]."""
    for nm, row in msa.items():
        if nm not in seqs:
            print(f"  WARN: {nm} in MSA but not in sequences")
            return False
        non_gap = [int(c) for c in row if int(c) >= 0]
        seq = list(seqs[nm].astype(np.int32))
        if non_gap != seq:
            print(f"  MISMATCH {nm}: MSA len={len(non_gap)}, seq len={len(seq)}")
            print(f"    first diff: msa={non_gap[:20]} seq={seq[:20]}")
            return False
    return True


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="BB11001")
    ap.add_argument("--bali-root", type=Path,
                    default=Path.home() / "bio-datasets" / "data" /
                    "balibase" / "bench1.0" / "bali3")
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "results" /
                    "exp2_v2_K4_top1000_tsb_emwarm" / "_best_chkpt")
    ap.add_argument("--alpha-z", type=float, default=100.0)
    # AIS hyperparameters. The setup phase (F_partial O(L^4)) dominates
    # wall-time at typical BAliBASE sizes (L ~ 80aa); AIS itself is cheap
    # since the inner kernel just does numpy MH on a small edge set, so
    # we use a generous schedule by default.
    ap.add_argument("--n-ais-steps", type=int, default=40)
    ap.add_argument("--n-inner-sweeps", type=int, default=150)
    ap.add_argument("--n-chains", type=int, default=16)
    ap.add_argument("--alpha-z-init", type=float, default=1e8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+",
                    default=["baseline_fsa", "muscle", "mafft", "ref"])
    args = ap.parse_args()

    in_path = args.bali_root / "in" / args.bench
    ref_path = args.bali_root / "ref" / args.bench
    print(f"=== {args.bench} ===")
    print(f"  in:  {in_path}")
    print(f"  ref: {ref_path}")

    # Load sequences.
    seqs = load_seqs(in_path)
    n_seqs = len(seqs)
    print(f"  n_seqs = {n_seqs}, "
          f"L = {[len(seqs[k]) for k in seqs]}")

    # Load reference.
    ref_msa, _core = load_ref(ref_path)
    # For loglik calculation, we use the bali3 ref but drop case info; for
    # scoring we use the bali3 ref directly (load_ref already preserves
    # case via the core mask, which sp_score/tc_score don't use).

    # Load model.
    print(f"  Loading TKF-DP model from {args.checkpoint}")
    state = load_minimal_state(args.checkpoint)
    print(f"    K_c={state.K_c}, "
          f"K_H={state.potts_dp.atoms.shape[0]}")

    # Build MSAs.
    msas = {}
    method_times = {}
    for m in args.methods:
        if m == "ref":
            print(f"\n  Loading ref MSA from {ref_path}")
            t0 = time.time()
            msas[m] = load_ref_msa_for_loglik(ref_path)
            method_times[m] = time.time() - t0
        elif m == "baseline_fsa":
            print(f"\n  Building baseline_fsa MSA")
            t0 = time.time()
            msas[m] = run_baseline_fsa_msa(seqs)
            method_times[m] = time.time() - t0
        elif m == "muscle":
            print(f"\n  Running MUSCLE")
            t0 = time.time()
            msas[m] = run_muscle_msa(seqs)
            method_times[m] = time.time() - t0
        elif m == "mafft":
            print(f"\n  Running MAFFT")
            t0 = time.time()
            msas[m] = run_mafft_msa(seqs)
            method_times[m] = time.time() - t0
        else:
            raise ValueError(f"unknown method: {m}")
        print(f"    msa_length = "
              f"{len(next(iter(msas[m].values())))}, "
              f"t = {method_times[m]:.1f}s")
        # Consistency check (non-ref MSAs only; ref may have wildcards
        # mapped differently).
        if m != "ref":
            if not msa_consistent_with_seqs(msas[m], seqs):
                print(f"    WARN: {m} MSA inconsistent with sequences")

    # SP / TC vs reference.
    print(f"\n  Scoring MSAs vs BAliBASE reference (SP/TC)")
    sp_tc = {}
    for m, msa in msas.items():
        try:
            sp = float(sp_score(msa, ref_msa))
            tc = float(tc_score(msa, ref_msa))
        except Exception as e:
            sp, tc = float('nan'), float('nan')
            print(f"    {m}: scoring failed: {e}")
        sp_tc[m] = (sp, tc)
        print(f"    {m:>16s}: SP={sp:.4f}  TC={tc:.4f}")

    # Composite log-likelihood.
    # Pre-compute pair_post / pair_taus / boost_states once (shared across
    # all MSAs since they all use the same x_seqs).
    print(f"\n  Pre-computing pair_post / boost_states (shared)")
    t0 = time.time()
    Q_lg, pi_lg = rate_matrix_lg()
    pairs = select_pairs_full(n_seqs)
    pair_post, pair_taus = compute_pairwise_posteriors(
        seqs, pairs, model='tkf92', Q=Q_lg, pi=pi_lg)
    names = list(seqs.keys())
    seqs_int = [np.asarray(seqs[nm]) for nm in names]
    pair_post_np = {k: np.asarray(v) for k, v in pair_post.items()}
    boost_states = build_boost_state(pair_post_np, pair_taus, seqs_int, state)
    print(f"    boost states for {len(boost_states)} pairs in {time.time()-t0:.1f}s")

    # For each MSA, compute composite loglik. Share the per-cherry MCMCSetup
    # across MSAs since it depends only on (X, Y, model, t).
    setups_cache = {}
    print(f"\n  Computing composite log-likelihoods (alpha_z={args.alpha_z}, "
          f"AIS: {args.n_ais_steps} steps x {args.n_inner_sweeps} inner sweeps "
          f"x {args.n_chains} chains)")
    composites = {}
    for m, msa in msas.items():
        print(f"\n    --- {m} ---")
        t0 = time.time()
        try:
            res = composite_loglik_msa(
                msa=msa, x_seqs=seqs, state=state,
                alpha_z=args.alpha_z,
                pairs=pairs, boost_states=boost_states, pair_taus=pair_taus,
                setups_cache=setups_cache,
                n_ais_steps=args.n_ais_steps,
                n_inner_sweeps=args.n_inner_sweeps,
                n_chains=args.n_chains,
                alpha_z_init=args.alpha_z_init,
                seed=args.seed,
                verbose=True,
            )
            composites[m] = res
            print(f"    log_p_total = {res.log_p_total:.4f}  "
                  f"(log_pi_TKF92 = {res.log_pi_TKF92_total:.4f}, "
                  f"log_Z_E = {res.log_Z_E_total:.4f})")
            print(f"    elapsed: {time.time() - t0:.1f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            composites[m] = None
            print(f"    FAILED: {e}")

    # Summary table.
    print(f"\n=== Summary: {args.bench} ===")
    print(f"  alpha_z = {args.alpha_z}")
    print(f"  Model: K_c={state.K_c}, K_H={state.potts_dp.atoms.shape[0]}\n")
    print(f"  {'method':>14s} {'SP':>8s} {'TC':>8s} {'log_pi':>11s} "
          f"{'log_Z_E':>9s} {'log_p_total':>12s} {'wall(s)':>8s}")
    for m in args.methods:
        sp, tc = sp_tc.get(m, (float('nan'), float('nan')))
        c = composites.get(m)
        if c is None:
            print(f"  {m:>14s} {sp:8.4f} {tc:8.4f}      nan      nan          nan      nan")
        else:
            print(f"  {m:>14s} {sp:8.4f} {tc:8.4f} "
                  f"{c.log_pi_TKF92_total:11.3f} {c.log_Z_E_total:9.3f} "
                  f"{c.log_p_total:12.3f} {c.total_seconds:8.1f}")

    # Cross-tabulation of preferences.
    valid = [(m, composites[m].log_p_total)
             for m in args.methods if composites.get(m) is not None]
    if valid:
        valid.sort(key=lambda x: -x[1])
        valid_sp = [(m, sp_tc[m][0]) for m, _ in valid
                    if not np.isnan(sp_tc[m][0])]
        valid_sp.sort(key=lambda x: -x[1])
        print(f"\n  Model-preferred (max log_p): {valid[0][0]}")
        if valid_sp:
            print(f"  SP-best:                     {valid_sp[0][0]}")
        if "ref" in [m for m, _ in valid]:
            ref_rank = next(i for i, (m, _) in enumerate(valid) if m == "ref")
            print(f"  ref's rank in model-pref order: {ref_rank + 1}/{len(valid)}")

    # Pairwise comparisons (model-rigorous: same Z_total cancels).
    if len(valid) >= 2:
        print(f"\n  Pairwise log_p differences (positive => row preferred over col):")
        ms = [m for m, _ in valid]
        print(f"  {'':>14s} " + " ".join(f"{m:>14s}" for m in ms))
        for m_a in ms:
            row = [f"{composites[m_a].log_p_total - composites[m_b].log_p_total:14.3f}"
                   for m_b in ms]
            print(f"  {m_a:>14s} " + " ".join(row))

    return 0


if __name__ == "__main__":
    sys.exit(main())
