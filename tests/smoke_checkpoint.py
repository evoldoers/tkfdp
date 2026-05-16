"""Roundtrip test: save_checkpoint → load_checkpoint reconstructs the
same SVIState (arrays equal, hyperparams preserved, RNG identical)."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PARENT, "src"))

from tkfdp.checkpoint import (EarlyStoppingState, load_checkpoint,
                                save_checkpoint, update_early_stopping,
                                validate_resume)
from tkfdp.partition_K import FamilyKState
from tkfdp.potts_dp import PottsDPState
from tkfdp.svi import SVIState


def main():
    rng = np.random.default_rng(42)
    K_c = 3; A = 20; K_H = 2

    # Synthetic state.
    pi_class = np.asarray([
        [0.05] * 20, [0.04] * 20, [0.06] * 20,
    ])
    pi_class = pi_class / pi_class.sum(axis=1, keepdims=True)
    atoms = rng.standard_normal((K_H, A, A)) * 0.3
    atoms = 0.5 * (atoms + atoms.transpose(0, 2, 1))
    assignments = rng.integers(0, K_H, size=(K_c, K_c)).astype(np.int64)
    counts = np.asarray([3, 3])
    potts_dp = PottsDPState(
        K_c=K_c, A=A, atoms=atoms, assignments=assignments, counts=counts,
        alpha_H=1.5,
        mu_prior=np.zeros((A, A)), tau_prior=np.full((A, A), 4.0),
    )

    per_family_data = []
    states_per_msa = []
    eta_per_msa = []
    for fam_idx, (name, L) in enumerate([("PF00001", 30), ("PF00002", 25)]):
        per_family_data.append(dict(family=name, L=L))
        partner = -np.ones(L, dtype=np.int32)
        partner[0] = 5; partner[5] = 0
        cls = rng.integers(0, K_c, size=L).astype(np.int32)
        states_per_msa.append(FamilyKState(family=name, L=L, K=K_c,
                                              partner=partner, cls=cls))
        eta_per_msa.append(rng.uniform(0.5, 2.0, size=L))

    state = SVIState(
        K_c=K_c, A=A, pi_class=pi_class, potts_dp=potts_dp,
        states_per_msa=states_per_msa, eta_per_msa=eta_per_msa,
        a_eta=2.5, b_eta=3.5, kappa_pi=4.0, alpha_c=8.0, alpha_H=1.5,
    )
    trace = dict(elapsed=[1.0, 2.0], n_pairs=[5, 7], pi_diff=[[0.1, 0.2, 0.3]],
                   H_norm=[1.5], log_l=[-100.0, -90.0], val_LL=[[10, -200.0]])

    es = EarlyStoppingState(best_val_LL=-100.0, best_iter=20,
                              n_evals_since_improvement=2)

    # Snapshot rng state before save.
    rng_before = np.random.default_rng()
    rng_before.bit_generator.state = rng.bit_generator.state

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        save_checkpoint(state, trace, rng, out_dir, it=42, es=es)

        # Verify files exist
        chkpt = out_dir / "_chkpt"
        assert (chkpt / "state.npz").exists(), "missing state.npz"
        assert (chkpt / "trace.json").exists(), "missing trace.json"
        assert (chkpt / "meta.json").exists(), "missing meta.json"
        print(f"  Saved: state.npz ({(chkpt/'state.npz').stat().st_size} B), "
                f"trace.json, meta.json")

        # Load
        state2, trace2, rng2, es2, meta = load_checkpoint(
            chkpt, per_family_data,
            mu_prior=np.zeros((A, A)),
            tau_prior=np.full((A, A), 4.0),
        )

        # Compare
        failures = []
        if not np.allclose(state.pi_class, state2.pi_class):
            failures.append("pi_class")
        if not np.allclose(state.potts_dp.atoms, state2.potts_dp.atoms):
            failures.append("potts_atoms")
        if not np.array_equal(state.potts_dp.assignments, state2.potts_dp.assignments):
            failures.append("potts_assignments")
        if state.K_c != state2.K_c: failures.append("K_c")
        if abs(state.a_eta - state2.a_eta) > 1e-12: failures.append("a_eta")
        if abs(state.kappa_pi - state2.kappa_pi) > 1e-12: failures.append("kappa_pi")
        for fi in range(len(per_family_data)):
            if not np.array_equal(state.states_per_msa[fi].cls,
                                   state2.states_per_msa[fi].cls):
                failures.append(f"cls_{fi}")
            if not np.array_equal(state.states_per_msa[fi].partner,
                                   state2.states_per_msa[fi].partner):
                failures.append(f"partner_{fi}")
            if not np.allclose(state.eta_per_msa[fi], state2.eta_per_msa[fi]):
                failures.append(f"eta_{fi}")

        if trace["log_l"] != trace2["log_l"]:
            failures.append("trace.log_l")
        if trace["val_LL"] != trace2["val_LL"]:
            failures.append("trace.val_LL")

        if abs(es.best_val_LL - es2.best_val_LL) > 1e-12: failures.append("es.best_val_LL")
        if es.best_iter != es2.best_iter: failures.append("es.best_iter")
        if es.n_evals_since_improvement != es2.n_evals_since_improvement:
            failures.append("es.n_evals_since_improvement")

        # RNG: draw same number of samples and compare.
        a = rng_before.standard_normal(20)
        b = rng2.standard_normal(20)
        if not np.allclose(a, b):
            failures.append(f"rng_draw_after_state_capture (max_diff={np.abs(a-b).max()})")

        # Validate resume validation
        try:
            validate_resume(meta, [fd["family"] for fd in per_family_data], expected_K_c=K_c)
            print("  validate_resume: PASS (matching args)")
        except ValueError as e:
            failures.append(f"validate_resume_unexpectedly_failed: {e}")
        try:
            validate_resume(meta, ["WRONG_FAMILY"], expected_K_c=K_c)
            failures.append("validate_resume should have failed on wrong families")
        except ValueError:
            print("  validate_resume: PASS (rejects wrong families)")

        if failures:
            print(f"\nFAILURES: {failures}")
            return 1
        print("\nAll roundtrip checks PASS")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
