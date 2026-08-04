"""Soft-observation (PSWM) cluster emission: reduction to hard variant.

`cluster_emission_batched_soft` marginalises pi_field and Sigma against
per-cherry, per-site PSWM distributions. When PSWMs are delta functions
on hard residues, the soft output must equal the hard output exactly.

When PSWMs are diffuse (uniform), the marginalised factors take known
limiting values via the stochasticity identities.
"""
from __future__ import annotations

import numpy as np

from tkfdp.coupling.dynfield import emission as _em


A_ALPH = 20


def _make_pi_field(rng, K_c=3, L_max=4, A=A_ALPH):
    raw = rng.gamma(2.0, size=(K_c, L_max, A))
    return raw / raw.sum(axis=-1, keepdims=True)


def _make_rho(rng, L_max=4):
    raw = rng.gamma(2.0, size=(L_max,))
    return raw / raw.sum()


def _to_delta(hard: np.ndarray, A: int = A_ALPH) -> np.ndarray:
    """(n, m) int -> (n, m, A) one-hot floats."""
    out = np.zeros(hard.shape + (A,), dtype=np.float64)
    idx = np.indices(hard.shape)
    out[idx[0], idx[1], hard] = 1.0
    return out


def test_soft_delta_matches_hard():
    """Delta PSWMs at leaves must reproduce the hard-observation output."""
    rng = np.random.default_rng(0)
    K_c, L_max, A = 3, 4, A_ALPH
    pi_field = _make_pi_field(rng, K_c, L_max, A)
    rho = _make_rho(rng, L_max)
    n_cherries = 5
    tau = rng.uniform(0.05, 0.8, size=n_cherries)
    rho_chain = 0.6
    per_cherry = _em.precompute_cluster_emission_per_cherry(
        tau=tau, rho=rho, pi_field=pi_field, rho_chain=rho_chain,
    )

    for m in [1, 2, 3]:
        classes = rng.integers(0, K_c, size=m)
        aa_a = rng.integers(0, A, size=(n_cherries, m))
        aa_b = rng.integers(0, A, size=(n_cherries, m))

        # Hard path (no gaps).
        mask_ones = np.ones((n_cherries, m), dtype=bool)
        hard_totals = _em.cluster_emission_batched(
            classes=classes,
            X_batch=aa_a, Y_batch=aa_b,
            mask_X=mask_ones, mask_Y=mask_ones,
            per_cherry=per_cherry,
        )

        # Soft path (delta PSWMs).
        X_soft = _to_delta(aa_a, A)
        Y_soft = _to_delta(aa_b, A)
        soft_totals = _em.cluster_emission_batched_soft(
            classes=classes, X_soft=X_soft, Y_soft=Y_soft,
            per_cherry=per_cherry,
        )
        assert np.allclose(hard_totals, soft_totals, atol=1e-12), (
            f"m={m}: hard={hard_totals} soft={soft_totals} "
            f"maxdiff={np.abs(hard_totals - soft_totals).max()}")


def test_soft_uniform_pswm_matches_analytic_marginal():
    """Uniform PSWM at BOTH leaves reduces to the pi-x-pi marginal.

    Under X_soft[q, s] = pi(c_s, theta) (soft equals the pi-weighted
    distribution — but that's not fully "uniform"; we use the strict
    uniform 1/A and check via direct summation over all (A^m)^2 hard
    combinations weighted by (1/A)^{2m}.
    """
    rng = np.random.default_rng(1)
    K_c, L_max, A = 2, 3, A_ALPH
    pi_field = _make_pi_field(rng, K_c, L_max, A)
    rho = _make_rho(rng, L_max)
    n_cherries = 3
    tau = rng.uniform(0.1, 0.5, size=n_cherries)
    rho_chain = 0.4
    per_cherry = _em.precompute_cluster_emission_per_cherry(
        tau=tau, rho=rho, pi_field=pi_field, rho_chain=rho_chain,
    )

    m = 1                                       # keep enumeration small
    classes = np.array([1])
    # Uniform PSWMs.
    X_soft = np.full((n_cherries, m, A), 1.0 / A)
    Y_soft = np.full((n_cherries, m, A), 1.0 / A)
    soft = _em.cluster_emission_batched_soft(
        classes=classes, X_soft=X_soft, Y_soft=Y_soft,
        per_cherry=per_cherry,
    )
    # Analytic reference: (1/A^2) * sum_{a, b} P_model(a, b | ...) per cherry.
    # We enumerate hard (a, b) and average.
    ref = np.zeros(n_cherries, dtype=np.float64)
    mask_ones = np.ones((n_cherries, m), dtype=bool)
    for a in range(A):
        for b in range(A):
            aa_a = np.full((n_cherries, m), a, dtype=np.int64)
            aa_b = np.full((n_cherries, m), b, dtype=np.int64)
            t = _em.cluster_emission_batched(
                classes=classes, X_batch=aa_a, Y_batch=aa_b,
                mask_X=mask_ones, mask_Y=mask_ones, per_cherry=per_cherry,
            )
            ref += t / (A * A)
    assert np.allclose(soft, ref, atol=1e-12), (
        f"uniform PSWM: soft={soft} ref={ref} "
        f"maxdiff={np.abs(soft - ref).max()}")


def test_soft_intermediate_pswm_matches_enumeration():
    """A non-degenerate PSWM must equal the weighted sum over hard
    combinations."""
    rng = np.random.default_rng(2)
    K_c, L_max, A = 2, 3, A_ALPH
    pi_field = _make_pi_field(rng, K_c, L_max, A)
    rho = _make_rho(rng, L_max)
    n_cherries = 2
    tau = np.array([0.3, 0.6])
    rho_chain = 0.5
    per_cherry = _em.precompute_cluster_emission_per_cherry(
        tau=tau, rho=rho, pi_field=pi_field, rho_chain=rho_chain,
    )
    m = 1
    classes = np.array([0])
    raw_X = rng.gamma(2.0, size=(n_cherries, m, A))
    X_soft = raw_X / raw_X.sum(axis=-1, keepdims=True)
    raw_Y = rng.gamma(2.0, size=(n_cherries, m, A))
    Y_soft = raw_Y / raw_Y.sum(axis=-1, keepdims=True)

    soft = _em.cluster_emission_batched_soft(
        classes=classes, X_soft=X_soft, Y_soft=Y_soft, per_cherry=per_cherry,
    )
    # Reference: sum_{a, b} X_soft[q, 0, a] * Y_soft[q, 0, b] * P(a, b | q).
    mask_ones = np.ones((n_cherries, m), dtype=bool)
    ref = np.zeros(n_cherries, dtype=np.float64)
    for a in range(A):
        for b in range(A):
            aa_a = np.full((n_cherries, m), a, dtype=np.int64)
            aa_b = np.full((n_cherries, m), b, dtype=np.int64)
            t = _em.cluster_emission_batched(
                classes=classes, X_batch=aa_a, Y_batch=aa_b,
                mask_X=mask_ones, mask_Y=mask_ones, per_cherry=per_cherry,
            )
            for q in range(n_cherries):
                ref[q] += X_soft[q, 0, a] * Y_soft[q, 0, b] * float(t[q])
    assert np.allclose(soft, ref, atol=1e-12), (
        f"intermediate PSWM: soft={soft} ref={ref} "
        f"maxdiff={np.abs(soft - ref).max()}")


if __name__ == "__main__":
    test_soft_delta_matches_hard()
    print("delta PSWM = hard: PASS")
    test_soft_uniform_pswm_matches_analytic_marginal()
    print("uniform PSWM = mean over hard: PASS")
    test_soft_intermediate_pswm_matches_enumeration()
    print("intermediate PSWM = weighted-sum enumeration: PASS")
