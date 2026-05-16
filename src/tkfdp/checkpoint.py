"""Periodic checkpoint dump + resume for SVI v2 long runs.

Writes a single rolling checkpoint to `<out_dir>/_chkpt/`:
  - state.npz       : SVIState arrays (pi_class, potts atoms, potts assignments,
                       potts counts, alpha_H) + per-MSA cls/partner/eta.
  - trace.json      : the per-outer-iter trace dict (elapsed, log_l, etc.).
  - meta.json       : iter, best_val_LL, best_iter, no-improvement counter,
                       hyperparams, family list (for resume validation), and
                       the numpy RNG state.

Resume reads `<resume_from>/state.npz` + `meta.json` + `trace.json` and
returns a populated SVIState plus the iter to start from. The caller is
responsible for asserting that the resume's family list and hyperparams
match the current run.

Atomic writes: dump to a temp file then `os.replace` so a kill mid-write
doesn't corrupt the checkpoint.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .partition_K import FamilyKState
from .potts_dp import PottsDPState
from .svi import SVIState


CHKPT_NAME = "_chkpt"
BEST_CHKPT_NAME = "_best_chkpt"


@dataclass
class EarlyStoppingState:
    """Tracks early-stopping bookkeeping across val LL evaluations."""
    best_val_LL: float = -np.inf
    best_iter: int = -1
    n_evals_since_improvement: int = 0


def save_checkpoint(state: SVIState, trace: dict, rng: np.random.Generator,
                      out_dir: Path, it: int,
                      es: EarlyStoppingState | None = None,
                      extra_meta: dict | None = None,
                      subdir: str = CHKPT_NAME) -> None:
    """Atomic checkpoint dump under `out_dir / <subdir>` (default `_chkpt`).

    Overwrites the previous checkpoint at that subdir. The atomicity gives
    crash safety: if killed mid-write, the previous checkpoint is intact.

    For best-so-far snapshots, pass `subdir=BEST_CHKPT_NAME`. The rolling
    `_chkpt` and the best-so-far `_best_chkpt` are independent.
    """
    chkpt_dir = out_dir / subdir
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. State arrays. SVIState contains numpy arrays at top level + nested
    # PottsDPState + list of FamilyKState.
    arrs: dict = dict(
        pi_class=state.pi_class.astype(np.float64),
        potts_atoms=state.potts_dp.atoms.astype(np.float64),
        potts_assignments=state.potts_dp.assignments.astype(np.int64),
        potts_counts=state.potts_dp.counts.astype(np.int64),
    )
    # Side potentials (optional). When training has the side-potential
    # extension enabled, h_pairs is a (K_c(K_c+1)/2, 2, A) tensor; old
    # checkpoints written before this branch existed simply omit the key.
    if state.potts_dp.h_pairs is not None:
        arrs["h_pairs"] = state.potts_dp.h_pairs.astype(np.float64)
    # TSB extension state (optional, similarly).
    if state.potts_dp.rho is not None:
        arrs["rho"] = state.potts_dp.rho.astype(np.float64)
    if state.potts_dp.tsb_betas is not None:
        arrs["tsb_betas"] = state.potts_dp.tsb_betas.astype(np.float64)
    for fam_idx, st in enumerate(state.states_per_msa):
        arrs[f"cls_{fam_idx}"] = st.cls.astype(np.int32)
        arrs[f"partner_{fam_idx}"] = st.partner.astype(np.int32)
        arrs[f"eta_{fam_idx}"] = state.eta_per_msa[fam_idx].astype(np.float64)

    # np.savez auto-appends ".npz" if not present, which would corrupt our
    # tmp-then-rename. Pass an open file object to avoid that.
    state_tmp = chkpt_dir / "state.tmp.npz"
    state_final = chkpt_dir / "state.npz"
    with open(state_tmp, "wb") as f:
        np.savez(f, **arrs)
    os.replace(state_tmp, state_final)

    # 2. Trace dict.
    trace_tmp = chkpt_dir / "trace.json.tmp"
    trace_final = chkpt_dir / "trace.json"
    with open(trace_tmp, "w") as f:
        json.dump(trace, f, indent=1)
    os.replace(trace_tmp, trace_final)

    # 3. Meta: iter, RNG state, hyperparams, early-stopping bookkeeping,
    # and family/MSA names for resume validation.
    meta = dict(
        iter=int(it),
        K_c=int(state.K_c),
        a_eta=float(state.a_eta), b_eta=float(state.b_eta),
        kappa_pi=float(state.kappa_pi),
        alpha_c=float(state.alpha_c),
        alpha_H=float(state.potts_dp.alpha_H),
        family_names=[st.family for st in state.states_per_msa],
        msa_lengths=[int(st.L) for st in state.states_per_msa],
        rng_state=_serialize_rng(rng),
    )
    if es is not None:
        meta["best_val_LL"] = float(es.best_val_LL)
        meta["best_iter"] = int(es.best_iter)
        meta["n_evals_since_improvement"] = int(es.n_evals_since_improvement)
    if extra_meta:
        meta.update(extra_meta)
    meta_tmp = chkpt_dir / "meta.json.tmp"
    meta_final = chkpt_dir / "meta.json"
    with open(meta_tmp, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(meta_tmp, meta_final)


def load_checkpoint(chkpt_dir: Path, per_family_data: list,
                      mu_prior: np.ndarray, tau_prior: np.ndarray
                      ) -> tuple[SVIState, dict, np.random.Generator,
                                   EarlyStoppingState, dict]:
    """Reconstruct (state, trace, rng, early_stop_state, meta) from a
    checkpoint directory. The caller validates that `meta['family_names']`
    matches the current run's family list before using the result.
    """
    chkpt_dir = Path(chkpt_dir)
    arrs = np.load(chkpt_dir / "state.npz")
    with open(chkpt_dir / "trace.json") as f:
        trace = json.load(f)
    with open(chkpt_dir / "meta.json") as f:
        meta = json.load(f)

    K_c = int(meta["K_c"])
    pi_class = np.asarray(arrs["pi_class"])
    A = pi_class.shape[1]

    # Optional fields (h_pairs / rho / tsb_betas) absent from older
    # checkpoints. np.array (not asarray) to make sure these are
    # writable -- the in-place symmetrize_h_pairs_diag projection in
    # update_potts_atoms_jit needs to mutate h_pairs.
    h_pairs = np.array(arrs["h_pairs"]) if "h_pairs" in arrs.files else None
    rho = np.array(arrs["rho"]) if "rho" in arrs.files else None
    tsb_betas = np.array(arrs["tsb_betas"]) if "tsb_betas" in arrs.files else None

    potts_dp = PottsDPState(
        K_c=K_c, A=A,
        atoms=np.asarray(arrs["potts_atoms"]),
        assignments=np.asarray(arrs["potts_assignments"]),
        counts=np.asarray(arrs["potts_counts"]),
        alpha_H=float(meta["alpha_H"]),
        mu_prior=mu_prior, tau_prior=tau_prior,
        h_pairs=h_pairs, rho=rho, tsb_betas=tsb_betas,
    )

    states_per_msa: list[FamilyKState] = []
    eta_per_msa: list[np.ndarray] = []
    for fam_idx, fd in enumerate(per_family_data):
        cls = np.asarray(arrs[f"cls_{fam_idx}"]).astype(np.int32)
        partner = np.asarray(arrs[f"partner_{fam_idx}"]).astype(np.int32)
        eta = np.asarray(arrs[f"eta_{fam_idx}"]).astype(np.float64)
        states_per_msa.append(FamilyKState(
            family=fd["family"], L=fd["L"], K=K_c,
            partner=partner, cls=cls,
        ))
        eta_per_msa.append(eta)

    state = SVIState(
        K_c=K_c, A=A, pi_class=pi_class, potts_dp=potts_dp,
        states_per_msa=states_per_msa, eta_per_msa=eta_per_msa,
        a_eta=float(meta["a_eta"]), b_eta=float(meta["b_eta"]),
        kappa_pi=float(meta["kappa_pi"]),
        alpha_c=float(meta["alpha_c"]),
        alpha_H=float(meta["alpha_H"]),
    )

    rng = _deserialize_rng(meta["rng_state"])

    es = EarlyStoppingState(
        best_val_LL=float(meta.get("best_val_LL", -np.inf)),
        best_iter=int(meta.get("best_iter", -1)),
        n_evals_since_improvement=int(
            meta.get("n_evals_since_improvement", 0)
        ),
    )
    return state, trace, rng, es, meta


def load_globals_from_checkpoint(chkpt_dir: Path, mu_prior: np.ndarray,
                                 tau_prior: np.ndarray
                                 ) -> tuple[np.ndarray, PottsDPState, dict]:
    """Load ONLY the global params (pi_class + PottsDPState) from a
    checkpoint, ignoring per-family latents. Used by --resume-globals-from
    to graft trained substitution/coupling params onto a fresh corpus.

    Returns (pi_class, potts_dp, meta). Caller is responsible for
    overlaying these onto its fresh SVIState; per-family `cls`,
    `partner`, `eta` are NOT loaded.
    """
    chkpt_dir = Path(chkpt_dir)
    arrs = np.load(chkpt_dir / "state.npz")
    with open(chkpt_dir / "meta.json") as f:
        meta = json.load(f)
    K_c = int(meta["K_c"])
    pi_class = np.asarray(arrs["pi_class"])
    A = pi_class.shape[1]
    atoms = np.asarray(arrs["potts_atoms"])
    K_H_max = int(atoms.shape[0])
    h_pairs = np.array(arrs["h_pairs"]) if "h_pairs" in arrs.files else None
    rho = np.array(arrs["rho"]) if "rho" in arrs.files else None
    tsb_betas = (np.array(arrs["tsb_betas"]) if "tsb_betas" in arrs.files
                 else None)
    # Older checkpoints (pre-h_pairs/rho/tsb_betas persistence fix) lack
    # these fields. Fall back to the uniform init that init_svi_state
    # uses for fresh runs — otherwise TSB resampling trips on a None
    # several outer iters in.
    if rho is None:
        rho = np.full(K_H_max, 1.0 / K_H_max)
    if tsb_betas is None and K_H_max > 1:
        tsb_betas = np.full(K_H_max - 1, 1.0 / K_H_max)
    potts_dp = PottsDPState(
        K_c=K_c, A=A,
        atoms=atoms,
        assignments=np.asarray(arrs["potts_assignments"]),
        counts=np.asarray(arrs["potts_counts"]),
        alpha_H=float(meta["alpha_H"]),
        mu_prior=mu_prior, tau_prior=tau_prior,
        h_pairs=h_pairs, rho=rho, tsb_betas=tsb_betas,
    )
    return pi_class, potts_dp, meta


def validate_resume(meta: dict, expected_family_names: list[str],
                       expected_K_c: int) -> None:
    """Raise ValueError if a checkpoint's family list / K_c don't match
    the current run, since the SVIState shape would mismatch."""
    if meta["K_c"] != expected_K_c:
        raise ValueError(
            f"Resume mismatch: chkpt K_c={meta['K_c']} but run K_c={expected_K_c}"
        )
    if list(meta["family_names"]) != list(expected_family_names):
        raise ValueError(
            f"Resume mismatch: chkpt families={meta['family_names']} "
            f"but run families={expected_family_names}"
        )


def _serialize_rng(rng: np.random.Generator) -> dict:
    """Capture a numpy Generator's state as a JSON-friendly dict."""
    raw = rng.bit_generator.state
    # Encode any np arrays inside as lists; recurse one level.
    out: dict = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                       for kk, vv in v.items()}
        else:
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def _deserialize_rng(state_dict: dict) -> np.random.Generator:
    rng = np.random.default_rng(0)
    raw: dict = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            raw[k] = {kk: (np.asarray(vv, dtype=np.uint64)
                            if kk == "key" or isinstance(vv, list) else vv)
                       for kk, vv in v.items()}
        else:
            raw[k] = v
    rng.bit_generator.state = raw
    return rng


def update_early_stopping(es: EarlyStoppingState, current_val_LL: float,
                             current_iter: int) -> EarlyStoppingState:
    """Mutate-in-place version of early-stopping update. Returns the same
    object for chaining. Improvement = strictly greater val LL."""
    if current_val_LL > es.best_val_LL:
        es.best_val_LL = current_val_LL
        es.best_iter = current_iter
        es.n_evals_since_improvement = 0
    else:
        es.n_evals_since_improvement += 1
    return es
