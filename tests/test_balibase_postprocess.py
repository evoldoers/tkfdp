"""Integration test: TKF-DP postprocessing on a BAliBASE pairwise alignment.

Loads BB11001 (4 sequences ~80aa each from BAliBASE Reference 1.1), runs
the FSA pipeline from ~/tkf-mixdom both with and without the TKF-DP
postprocessing pass, and scores both alignments against the BAliBASE
reference via sum-of-pairs.

Hooks:
- Pairwise residue posteriors come from
  ``tkfmixdom.jax.tree.fsa_anneal.compute_pairwise_posteriors`` (TKF92
  Pair HMM forward--backward, GPU-accelerated).
- Multiple-alignment assembly uses the same module's
  ``sequence_annealing``.
- The TKF-DP correction applies between these two stages: each pairwise
  match-posterior matrix Q_{ij} is replaced by Q'_{ij} ∝ Q_{ij}
  exp(boost_{ij}) per ``src.tkfdp.postprocessing`` and renormalized
  (here we renormalize multiplicatively rather than re-running F/B —
  the production path would feed exp(boost) as a Match-state emission
  multiplier and re-run forward--backward).
- The trained TKF-DP state comes from the best K=4 checkpoint produced
  by ``experiments/exp2_pfam_v2.py``.

Reports SP score for baseline FSA and corrected FSA against the .ref
alignment.

Run: PYTHONPATH=src python3 tests/test_balibase_postprocess.py
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
from tkfdp.potts_dp import PottsDPState                                            # noqa: E402

# Pair HMM state codes (from tkfmixdom.jax.core.params).
S_STATE, M_STATE, I_STATE, D_STATE, E_STATE = 0, 1, 2, 3, 4


DATA_DIR = Path(__file__).parent / "data" / "balibase"
CHECKPOINT_DIR = Path(__file__).parent.parent / "results" / \
    "exp2_v2_K4_top1000_tsb_emwarm" / "_best_chkpt"


@dataclass
class _MinimalState:
    """The fields that ``src/tkfdp/postprocessing.py`` actually reads from
    a trained SVIState. Keeping this lightweight means we can load a
    checkpoint without rebuilding per-MSA state (which would require the
    original training corpus)."""
    K_c: int
    A: int
    pi_class: np.ndarray
    potts_dp: PottsDPState


# --- Loaders ---------------------------------------------------------------


def load_balibase_fasta(path: Path) -> dict:
    """Load a plain FASTA: returns {name: (L,) int array, X-mapped to 20}."""
    return {
        name: np.array([AA_TO_INT.get(c, 20) for c in seq.upper()],
                          dtype=np.int32)
        for name, seq in read_fasta(str(path))
    }


def load_balibase_ref(path: Path) -> dict:
    """Load a BAliBASE .ref alignment: returns {name: (L_aln,) int array,
    -1 = gap}. The .ref uses '.' / '-' for gaps and mixed case for
    aligned-core vs. unaligned residues; we treat both as residues
    (uppercase for scoring), gaps as -1."""
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
    """Pull just (K_c, pi_class, atoms, assignments, [h_pairs]) from the
    SVI checkpoint — bypassing per-MSA reconstruction."""
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


# --- The two FSA runs ------------------------------------------------------


def _msa_from_col_assignments(sequences, names, col_assignments,
                                  msa_length):
    msa = {}
    for i, n in enumerate(names):
        row = np.full(msa_length, -1, dtype=np.int32)
        for k in range(len(sequences[n])):
            col = col_assignments[i][k]
            row[col] = int(sequences[n][k])
        msa[n] = row
    return msa


def _correct_pair_via_fb_rerun(x_seq: np.ndarray, y_seq: np.ndarray,
                                    Q_baseline: np.ndarray, tau: float,
                                    ins_rate: float, del_rate: float,
                                    ext: float, Q_lg, pi_lg,
                                    state: _MinimalState,
                                    alpha_z: float) -> np.ndarray:
    """Apply the boost via the proper "fold into Match emission and re-run
    forward--backward" path (Section "Pairwise Alignment Postprocessing"
    of main.tex). The boost goes into the Match-state log-emission table;
    the fresh F/B then renormalizes into a valid pair-HMM posterior.

    All of x_seq, y_seq are integer arrays with wildcards mapped to 20
    (treated as zero-information by pair_hmm_emissions via its 21-padded
    log_sub / log_pi tables; the boost is applied with wildcards clamped
    to 19 since the trained Potts atoms are A=20).
    """
    log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
        ins_rate, del_rate, tau, ext, Q_lg, pi_lg)
    x_j = jnp.asarray(x_seq); y_j = jnp.asarray(y_seq)
    base_emit = pair_hmm_emissions(state_types, x_j, y_j, sub_matrix, pi_out)
    # base_emit shape: (Lx + 1, Ly + 1, ns). Match-state slice covers rows
    # i = 1..Lx and cols j = 1..Ly (entries [0, *] / [*, 0] are dummy).

    # Compute the boost using the K=20-clamped sequences (the trained Potts
    # tensors live on A = 20).
    x_clamp = np.minimum(np.asarray(x_seq, dtype=np.int64), 19)
    y_clamp = np.minimum(np.asarray(y_seq, dtype=np.int64), 19)
    t = float(tau)
    log_boost = correct_pair_posterior(
        np.asarray(Q_baseline), x_clamp, y_clamp, t, state,
        alpha_z=alpha_z, return_boost=True,
    )                                                          # (Lx, Ly)

    # Fold boost into the M-state emission table.
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


def run_fsa(sequences: dict,
             apply_correction: bool,
             state: _MinimalState | None,
             alpha_z: float,
             ins_rate: float = 0.02, del_rate: float = 0.05, ext: float = 0.5,
             n_anneal_iters: int = 5,
             correction_mode: str = "fb_rerun",
             verbose: bool = False) -> dict:
    """One FSA run, optionally with TKF-DP postprocessing.

    correction_mode:
      "fb_rerun" — fold exp(boost) into the Match-state log-emission table
        and re-run forward--backward to produce a properly renormalized
        Q'. This is the production path described in main.tex.
      "multiplicative" — Q'_{ij} ∝ Q_{ij} · exp(boost_{ij}), without
        re-running F/B. Cheap but does not enforce the pair-HMM marginal
        constraints; included for ablation.
    """
    from tkfmixdom.jax.core.protein import rate_matrix_lg

    Q_lg, pi_lg = rate_matrix_lg()
    names = list(sequences.keys())
    n_seqs = len(names)
    seq_lens = [len(sequences[n]) for n in names]

    pairs = select_pairs_full(n_seqs)
    pair_post, pair_taus = compute_pairwise_posteriors(
        sequences, pairs, model='tkf92',
        ins_rate=ins_rate, del_rate=del_rate, ext=ext,
        Q=Q_lg, pi=pi_lg, n_newton=5, tau_init=1.0, verbose=verbose,
    )

    if apply_correction:
        if state is None:
            raise ValueError("apply_correction=True requires a TKF-DP state")
        for (i, j), Q in pair_post.items():
            x_arr = np.asarray(sequences[names[i]])
            y_arr = np.asarray(sequences[names[j]])
            t = float(pair_taus[(i, j)])
            if correction_mode == "fb_rerun":
                Q_corr_np = _correct_pair_via_fb_rerun(
                    x_arr, y_arr, np.asarray(Q), t,
                    ins_rate, del_rate, ext, Q_lg, pi_lg,
                    state, alpha_z=alpha_z,
                )
            elif correction_mode == "multiplicative":
                x_clamp = np.minimum(x_arr.astype(np.int64), 19)
                y_clamp = np.minimum(y_arr.astype(np.int64), 19)
                Q_corr = correct_pair_posterior(
                    np.asarray(Q), x_clamp, y_clamp, t, state,
                    alpha_z=alpha_z, return_boost=False,
                )
                Q_corr_np = np.asarray(Q_corr).clip(min=0.0)
            else:
                raise ValueError(f"Unknown correction_mode: {correction_mode}")
            pair_post[(i, j)] = jnp.asarray(Q_corr_np)

    col_assignments, msa_length = sequence_annealing(
        n_seqs, seq_lens, pair_post,
        n_iterations=n_anneal_iters, verbose=verbose,
    )
    return _msa_from_col_assignments(sequences, names, col_assignments,
                                            msa_length)


# --- Main ------------------------------------------------------------------


def main() -> int:
    seqs = load_balibase_fasta(DATA_DIR / "BB11001.fasta")
    ref = load_balibase_ref(DATA_DIR / "BB11001.ref")
    print(f"BB11001: {len(seqs)} sequences, lengths "
            f"{ {n: len(s) for n, s in seqs.items()} }")

    if not (CHECKPOINT_DIR / "state.npz").exists():
        print(f"WARN: no checkpoint at {CHECKPOINT_DIR}; "
                f"baseline FSA only.")
        baseline = run_fsa(seqs, apply_correction=False, state=None,
                              alpha_z=100.0, verbose=False)
        sp_b = sp_score(baseline, ref)
        tc_b = tc_score(baseline, ref)
        print(f"  baseline FSA:   SP = {sp_b:.4f}   TC = {tc_b:.4f}")
        return 0

    state = load_minimal_state(CHECKPOINT_DIR)
    print(f"Loaded TKF-DP state: K_c={state.K_c}, "
            f"K_H={state.potts_dp.atoms.shape[0]}, "
            f"side potentials={'on' if state.potts_dp.h_pairs is not None else 'off'}")

    baseline = run_fsa(seqs, apply_correction=False, state=None,
                          alpha_z=100.0, verbose=False)
    sp_b = sp_score(baseline, ref)
    tc_b = tc_score(baseline, ref)

    print(f"\n{'method':>26s}  {'SP':>8s}  {'TC':>8s}")
    print(f"{'baseline':>26s}  {sp_b:8.4f}  {tc_b:8.4f}")
    for mode in ("multiplicative", "fb_rerun"):
        print(f"  --- correction mode: {mode} ---")
        for az in (1.0, 10.0, 100.0, 1000.0, 1e6):
            corr = run_fsa(seqs, apply_correction=True, state=state,
                                alpha_z=az, correction_mode=mode,
                                verbose=False)
            sp = sp_score(corr, ref)
            tc = tc_score(corr, ref)
            tag = f"{mode[:4]}. alpha_z={az:g}"
            print(f"{tag:>26s}  {sp:8.4f}  {tc:8.4f}")
    print("\nAt alpha_z -> infty the correction collapses to the identity\n"
            "(eps -> 0, log_boost -> 0); both correction modes recover\n"
            "baseline. fb_rerun is the production path (boost folded into\n"
            "Match-state log-emission table, then a fresh F/B); the\n"
            "multiplicative variant is included for ablation.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
