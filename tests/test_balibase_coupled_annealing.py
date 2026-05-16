"""Three-way comparison on BB11001:
    1. baseline FSA (no correction, vanilla sequence_annealing)
    2. pre-correction FSA (postprocessing.correct_pair_posterior + vanilla
       sequence_annealing)  — the existing approach
    3. coupled-annealing FSA (vanilla baseline pair posteriors +
       coupled_sequence_annealing) — the new approach

Reports SP / TC for each. The coupled-annealing variant is also swept
over the q_min / mu_min pruning thresholds.

Run: PYTHONPATH=src python3 tests/test_balibase_coupled_annealing.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# Hook into ~/tkf-mixdom for FSA + scoring (no duplication).
TKFMIXDOM_ROOT = Path.home() / "tkf-mixdom" / "python"
sys.path.insert(0, str(TKFMIXDOM_ROOT))
from tkfmixdom.jax.evaluate.metrics import sp_score, tc_score                 # noqa: E402
from tkfmixdom.jax.dp.hmm import (                                                # noqa: E402
    forward_backward_2d, pair_hmm_emissions,
)
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm                  # noqa: E402
from tkfmixdom.jax.tree.fsa_anneal import (                                       # noqa: E402
    compute_pairwise_posteriors, sequence_annealing, select_pairs_full,
)
from tkfmixdom.jax.util.io import AA_TO_INT, AMINO_ACIDS, read_fasta             # noqa: E402

from tkfdp.postprocessing import correct_pair_posterior                           # noqa: E402
from tkfdp.coupled_annealing import (                                             # noqa: E402
    build_boost_state, coupled_sequence_annealing,
)
from tkfdp.potts_dp import PottsDPState                                            # noqa: E402

# Pair HMM state codes.
S_STATE, M_STATE, I_STATE, D_STATE, E_STATE = 0, 1, 2, 3, 4

DATA_DIR = Path(__file__).parent / "data" / "balibase"
CHECKPOINT_DIR = Path(__file__).parent.parent / "results" / \
    "exp2_v2_K4_top1000_tsb_emwarm" / "_best_chkpt"


@dataclass
class _MinimalState:
    K_c: int
    A: int
    pi_class: np.ndarray
    potts_dp: PottsDPState


def load_balibase_fasta(path: Path) -> dict:
    return {
        name: np.array([AA_TO_INT.get(c, 20) for c in seq.upper()],
                          dtype=np.int32)
        for name, seq in read_fasta(str(path))
    }


def load_balibase_ref(path: Path) -> dict:
    out = {}
    for name, seq in read_fasta(str(path)):
        row = np.full(len(seq), -1, dtype=np.int32)
        for k, c in enumerate(seq):
            if c == '.' or c == '-':
                row[k] = -1
            else:
                row[k] = AA_TO_INT.get(c.upper(), 20)
        out[name] = row
    return out


def load_minimal_state(chkpt_dir: Path) -> _MinimalState:
    import json
    state_npz = np.load(chkpt_dir / "state.npz")
    meta = json.loads((chkpt_dir / "meta.json").read_text())
    K_c = int(meta["K_c"])
    A = state_npz["pi_class"].shape[1]
    h_pairs = state_npz["h_pairs"] if "h_pairs" in state_npz.files else None
    pdp = PottsDPState(
        K_c=K_c, A=A,
        atoms=state_npz["potts_atoms"],
        assignments=state_npz["potts_assignments"],
        counts=state_npz["potts_counts"],
        alpha_H=float(meta.get("alpha_H", 1.0)),
        h_pairs=h_pairs,
    )
    return _MinimalState(K_c=K_c, A=A, pi_class=state_npz["pi_class"],
                            potts_dp=pdp)


def _msa_from_col_assignments(sequences, names, col_assignments, msa_length):
    msa = {}
    for i, n in enumerate(names):
        row = np.full(msa_length, -1, dtype=np.int32)
        for k in range(len(sequences[n])):
            col = col_assignments[i][k]
            row[col] = int(sequences[n][k])
        msa[n] = row
    return msa


def _correct_pair_via_fb_rerun(x_seq, y_seq, Q_baseline, tau,
                                ins_rate, del_rate, ext, Q_lg, pi_lg,
                                state, alpha_z):
    log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
        ins_rate, del_rate, tau, ext, Q_lg, pi_lg)
    x_j = jnp.asarray(x_seq); y_j = jnp.asarray(y_seq)
    base_emit = pair_hmm_emissions(state_types, x_j, y_j, sub_matrix, pi_out)

    x_clamp = np.minimum(np.asarray(x_seq, dtype=np.int64), 19)
    y_clamp = np.minimum(np.asarray(y_seq, dtype=np.int64), 19)
    t = float(tau)
    log_boost = correct_pair_posterior(
        np.asarray(Q_baseline), x_clamp, y_clamp, t, state,
        alpha_z=alpha_z, return_boost=True,
    )

    Lx, Ly = x_seq.shape[0], y_seq.shape[0]
    is_M = jnp.asarray(state_types) == M_STATE
    boost_padded = jnp.zeros_like(base_emit)
    boost_padded = boost_padded.at[1:Lx + 1, 1:Ly + 1, :].set(
        is_M[None, None, :].astype(base_emit.dtype) *
        jnp.asarray(log_boost)[:, :, None]
    )
    emit_corr = base_emit + boost_padded

    _, posteriors, _ = forward_backward_2d(
        log_trans, jnp.asarray(state_types), x_j, y_j, sub_matrix, pi_out,
        log_emit_table=emit_corr,
    )
    Q_corr = jnp.sum(
        posteriors[1:Lx + 1, 1:Ly + 1, :] * is_M[None, None, :].astype(
            posteriors.dtype),
        axis=-1,
    )
    return np.asarray(Q_corr)


def _baseline_pair_posteriors(seqs, ins_rate, del_rate, ext):
    """Compute the baseline pair-HMM posteriors and tau estimates.

    Returns (sequences_dict, names, seq_lens, pair_post, pair_taus,
             Q_lg, pi_lg).
    """
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    Q_lg, pi_lg = rate_matrix_lg()
    names = list(seqs.keys())
    n_seqs = len(names)
    seq_lens = [len(seqs[n]) for n in names]
    pairs = select_pairs_full(n_seqs)
    pair_post, pair_taus = compute_pairwise_posteriors(
        seqs, pairs, model='tkf92',
        ins_rate=ins_rate, del_rate=del_rate, ext=ext,
        Q=Q_lg, pi=pi_lg, n_newton=5, tau_init=1.0,
    )
    return names, seq_lens, pair_post, pair_taus, Q_lg, pi_lg


def run_baseline(seqs, names, seq_lens, pair_post, n_anneal_iters=5):
    col_assignments, msa_length = sequence_annealing(
        len(names), seq_lens, dict(pair_post),
        n_iterations=n_anneal_iters, verbose=False,
    )
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def run_precorrection(seqs, names, seq_lens, pair_post, pair_taus,
                       state, ins_rate, del_rate, ext, Q_lg, pi_lg,
                       alpha_z, n_anneal_iters=5):
    pair_post_corr = {}
    for (i, j), Q in pair_post.items():
        x_arr = np.asarray(seqs[names[i]])
        y_arr = np.asarray(seqs[names[j]])
        t = float(pair_taus[(i, j)])
        Q_corr_np = _correct_pair_via_fb_rerun(
            x_arr, y_arr, np.asarray(Q), t,
            ins_rate, del_rate, ext, Q_lg, pi_lg,
            state, alpha_z=alpha_z,
        )
        pair_post_corr[(i, j)] = jnp.asarray(Q_corr_np)
    col_assignments, msa_length = sequence_annealing(
        len(names), seq_lens, pair_post_corr,
        n_iterations=n_anneal_iters, verbose=False,
    )
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def run_coupled(seqs, names, seq_lens, pair_post, pair_taus, state,
                 q_min=0.1, mu_min=0.1, max_pairs_per_anchor=32,
                 n_anneal_iters=5, verbose=False):
    seqs_int = [np.asarray(seqs[n]) for n in names]
    boost_states = build_boost_state(
        {k: np.asarray(v) for k, v in pair_post.items()},
        pair_taus, seqs_int, state,
    )
    col_assignments, msa_length = coupled_sequence_annealing(
        len(names), seq_lens,
        {k: np.asarray(v) for k, v in pair_post.items()},
        boost_states=boost_states,
        n_iterations=n_anneal_iters,
        q_min=q_min, mu_min=mu_min,
        max_pairs_per_anchor=max_pairs_per_anchor,
        verbose=verbose,
    )
    return _msa_from_col_assignments(seqs, names, col_assignments, msa_length)


def main() -> int:
    sys.setrecursionlimit(100000)
    seqs = load_balibase_fasta(DATA_DIR / "BB11001.fasta")
    ref = load_balibase_ref(DATA_DIR / "BB11001.ref")
    print(f"BB11001: {len(seqs)} sequences, lengths "
            f"{ {n: len(s) for n, s in seqs.items()} }")

    if not (CHECKPOINT_DIR / "state.npz").exists():
        print(f"WARN: no checkpoint at {CHECKPOINT_DIR}; "
                f"can't run TKF-DP variants.")
        return 1

    state = load_minimal_state(CHECKPOINT_DIR)
    print(f"Loaded TKF-DP state: K_c={state.K_c}, "
            f"K_H={state.potts_dp.atoms.shape[0]}, "
            f"side potentials={'on' if state.potts_dp.h_pairs is not None else 'off'}")

    ins_rate, del_rate, ext = 0.02, 0.05, 0.5

    print("\nComputing baseline pair-HMM posteriors...")
    names, seq_lens, pair_post, pair_taus, Q_lg, pi_lg = \
        _baseline_pair_posteriors(seqs, ins_rate, del_rate, ext)

    print(f"\n{'method':>40s}  {'SP':>8s}  {'TC':>8s}")

    # 1. baseline FSA.
    msa_b = run_baseline(seqs, names, seq_lens, pair_post)
    sp_b = sp_score(msa_b, ref); tc_b = tc_score(msa_b, ref)
    print(f"{'baseline':>40s}  {sp_b:8.4f}  {tc_b:8.4f}")

    # 2. pre-correction FSA at the alpha_z that the existing test found
    #    most informative (sweep small range).
    print(f"\n  --- pre-correction (fb_rerun) ---")
    pre_results = []
    for az in (1.0, 10.0, 100.0, 1000.0):
        msa_p = run_precorrection(
            seqs, names, seq_lens, pair_post, pair_taus, state,
            ins_rate, del_rate, ext, Q_lg, pi_lg, alpha_z=az,
        )
        sp_p = sp_score(msa_p, ref); tc_p = tc_score(msa_p, ref)
        pre_results.append((az, sp_p, tc_p))
        print(f"{f'precorr alpha_z={az:g}':>40s}  {sp_p:8.4f}  {tc_p:8.4f}")

    # 3. coupled-annealing FSA — sweep q_min / mu_min.
    print(f"\n  --- coupled annealing ---")
    coupled_results = []
    for q_min in (0.1, 0.2, 0.4, 0.6):
        for mu_min in (0.1, 0.5, 1.0, 2.0):
            try:
                msa_c = run_coupled(
                    seqs, names, seq_lens, pair_post, pair_taus, state,
                    q_min=q_min, mu_min=mu_min,
                )
                sp_c = sp_score(msa_c, ref); tc_c = tc_score(msa_c, ref)
            except Exception as e:
                sp_c = float('nan'); tc_c = float('nan')
                print(f"  ERROR at q={q_min}, mu={mu_min}: {e}")
                continue
            coupled_results.append((q_min, mu_min, sp_c, tc_c))
            tag = f"coupled q_min={q_min:g} mu_min={mu_min:g}"
            print(f"{tag:>40s}  {sp_c:8.4f}  {tc_c:8.4f}")

    # Summary line.
    best_pre = max(pre_results, key=lambda x: x[1])
    best_cpl = max(coupled_results, key=lambda x: x[2]) if \
        coupled_results else (None, None, float('nan'), float('nan'))
    print(f"\n{'best precorr':>40s}  {best_pre[1]:8.4f}  {best_pre[2]:8.4f}"
            f"   (alpha_z={best_pre[0]:g})")
    print(f"{'best coupled':>40s}  {best_cpl[2]:8.4f}  {best_cpl[3]:8.4f}"
            f"   (q_min={best_cpl[0]}, mu_min={best_cpl[1]})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
