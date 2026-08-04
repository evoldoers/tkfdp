"""Numerical equivalence: batched cluster emission vs per-cherry path."""
from __future__ import annotations

import numpy as np

from tkfdp.coupling.dynfield import emission as _em


def _make_pi_field(rng, K_c=3, L_max=4, A=20):
    raw = rng.gamma(2.0, size=(K_c, L_max, A))
    return raw / raw.sum(axis=-1, keepdims=True)


def _make_rho(rng, L_max=4):
    raw = rng.gamma(2.0, size=(L_max,))
    return raw / raw.sum()


def _slow_loglik(classes, X_batch, Y_batch, cherry_mask, tau, rho, pi_field,
                  rho_chain):
    """Reference implementation matching the old per-cherry loop in
    `make_cluster_loglik_fn`."""
    Q_cf = _em.per_class_field_Q(pi_field)
    ll = 0.0
    for q in range(X_batch.shape[0]):
        if not cherry_mask[q]:
            continue
        t_q = float(tau[q])
        P_half = _em.per_class_field_P_half(Q_cf, t=t_q)
        Sigma = _em.per_class_field_cherry_sigma(P_half, pi_field)
        P_per_theta_case, _ = _em.cluster_emission_per_theta(
            t=t_q, rho=rho, pi_field=pi_field,
            classes=classes,
            X_obs=X_batch[q], Y_obs=Y_batch[q],
            rho_chain=rho_chain,
            precomputed_Sigma=Sigma,
        )
        total = float(P_per_theta_case.sum())
        ll += np.log(max(total, 1e-300))
    return ll


def _batched_loglik(classes, X_batch, Y_batch, cherry_mask, tau, rho,
                     pi_field, rho_chain):
    per_cherry = _em.precompute_cluster_emission_per_cherry(
        tau=tau, rho=rho, pi_field=pi_field, rho_chain=rho_chain,
    )
    # Tests pre-2026-06-30-gap-fix used `cherry_mask` to drop cherries.
    # Translate into the new (mask_X, mask_Y) API: a cherry that passed
    # the old mask has every site observed; one that failed is replicated
    # as all-gapped (contributes 1.0 -> log 0 -> dropped from the sum).
    n, m = X_batch.shape
    mask_X = np.broadcast_to(cherry_mask[:, None], (n, m)).copy()
    mask_Y = np.broadcast_to(cherry_mask[:, None], (n, m)).copy()
    totals = _em.cluster_emission_batched(
        classes=classes,
        X_batch=X_batch, Y_batch=Y_batch,
        mask_X=mask_X, mask_Y=mask_Y,
        per_cherry=per_cherry,
    )
    return float(np.log(np.maximum(totals, 1e-300)).sum())


def test_batched_matches_per_cherry_small():
    rng = np.random.default_rng(0)
    K_c, L_max, A = 3, 4, 20
    pi_field = _make_pi_field(rng, K_c, L_max, A)
    rho = _make_rho(rng, L_max)
    n_cherries = 7
    tau = rng.uniform(0.05, 0.8, size=n_cherries)
    L_cols = 12
    aa_a = rng.integers(0, A, size=(n_cherries, L_cols))
    aa_b = rng.integers(0, A, size=(n_cherries, L_cols))
    both_aa = rng.random((n_cherries, L_cols)) > 0.1
    cls = rng.integers(0, K_c, size=L_cols)
    rho_chain = 0.6

    for m in [1, 2, 3, 5]:
        cols = rng.choice(L_cols, size=m, replace=False)
        classes = cls[cols]
        cherry_mask = both_aa[:, cols].all(axis=1)
        X_batch = aa_a[:, cols]
        Y_batch = aa_b[:, cols]
        ll_slow = _slow_loglik(
            classes, X_batch, Y_batch, cherry_mask, tau, rho, pi_field,
            rho_chain)
        ll_fast = _batched_loglik(
            classes, X_batch, Y_batch, cherry_mask, tau, rho, pi_field,
            rho_chain)
        assert np.isclose(ll_slow, ll_fast, atol=1e-9, rtol=1e-9), (
            f"m={m}: slow={ll_slow}, fast={ll_fast}, diff={ll_slow-ll_fast}")


def test_batched_handles_no_cherries():
    rng = np.random.default_rng(1)
    pi_field = _make_pi_field(rng)
    rho = _make_rho(rng)
    tau = np.array([0.3, 0.5])
    per_cherry = _em.precompute_cluster_emission_per_cherry(
        tau=tau, rho=rho, pi_field=pi_field, rho_chain=0.5)
    # All-gapped cherries should return totals=1.0 (log 0 contribution).
    X_batch = np.zeros((2, 3), dtype=np.int64)
    Y_batch = np.zeros((2, 3), dtype=np.int64)
    all_gapped = np.zeros((2, 3), dtype=bool)
    totals = _em.cluster_emission_batched(
        classes=np.array([0, 1, 2]),
        X_batch=X_batch, Y_batch=Y_batch,
        mask_X=all_gapped, mask_Y=all_gapped, per_cherry=per_cherry)
    assert totals.shape == (2,)
    assert np.allclose(totals, 1.0)


def test_batched_singleton_cluster():
    """m=1 case (the cherry_loglik_fn([s]) call in the CRP sweep)."""
    rng = np.random.default_rng(2)
    pi_field = _make_pi_field(rng)
    rho = _make_rho(rng)
    n_cherries = 5
    tau = rng.uniform(0.1, 0.5, size=n_cherries)
    cherry_mask = np.array([True, True, True, False, True])
    X_batch = rng.integers(0, 20, size=(n_cherries, 1))
    Y_batch = rng.integers(0, 20, size=(n_cherries, 1))
    classes = np.array([2])
    rho_chain = 0.8

    ll_slow = _slow_loglik(
        classes, X_batch, Y_batch, cherry_mask, tau, rho, pi_field,
        rho_chain)
    ll_fast = _batched_loglik(
        classes, X_batch, Y_batch, cherry_mask, tau, rho, pi_field,
        rho_chain)
    assert np.isclose(ll_slow, ll_fast, atol=1e-9, rtol=1e-9)


def test_gap_marginalisation_collapses_factor():
    """At a gapped site s with rho_chain=0 the per-(c, theta) factor for
    that site is 1.0 in pi_at and Sigma_at; the kernel must reproduce
    that exactly. Verify via the stochasticity identities:
    sum_a pi^(c, theta)(a) = 1, sum_{a, b} Sigma^(c, theta)(a, b; t) = 1.
    """
    rng = np.random.default_rng(7)
    pi_field = _make_pi_field(rng)
    rho = _make_rho(rng)
    K_c, L_max, A = pi_field.shape
    n = 3
    tau = np.full(n, 0.3)
    per_cherry = _em.precompute_cluster_emission_per_cherry(
        tau=tau, rho=rho, pi_field=pi_field, rho_chain=0.5)

    # Cluster size 2, classes (1, 2). Cherry 0: both sites observed.
    # Cherry 1: site 0 gapped on X only. Cherry 2: both gapped at site 0.
    classes = np.array([1, 2])
    X = np.array([[5, 7], [13, 7], [4, 11]], dtype=np.int64)
    Y = np.array([[12, 3], [12, 3], [9, 3]], dtype=np.int64)
    mask_X = np.array([[True, True], [False, True], [False, True]])
    mask_Y = np.array([[True, True], [True, True], [False, True]])
    fast = _em.cluster_emission_batched(
        classes=classes, X_batch=X, Y_batch=Y,
        mask_X=mask_X, mask_Y=mask_Y, per_cherry=per_cherry)

    # Reference cherry 1: integrate out X[1, 0] = sum over 20 X values.
    ref1 = 0.0
    for x_val in range(A):
        X_full = X.copy(); X_full[1, 0] = x_val
        m_full = mask_X.copy(); m_full[1, 0] = True
        t = _em.cluster_emission_batched(
            classes=classes, X_batch=X_full, Y_batch=Y,
            mask_X=m_full, mask_Y=mask_Y, per_cherry=per_cherry)
        ref1 += float(t[1])
    assert np.isclose(fast[1], ref1, atol=1e-12), (
        f"X-gap marginalisation: kernel={fast[1]:.10f} ref={ref1:.10f}")

    # Reference cherry 2: integrate over (X[2, 0], Y[2, 0]) = 400 values.
    ref2 = 0.0
    for x_val in range(A):
        for y_val in range(A):
            X_full = X.copy(); X_full[2, 0] = x_val
            Y_full = Y.copy(); Y_full[2, 0] = y_val
            mX_full = mask_X.copy(); mX_full[2, 0] = True
            mY_full = mask_Y.copy(); mY_full[2, 0] = True
            t = _em.cluster_emission_batched(
                classes=classes, X_batch=X_full, Y_batch=Y_full,
                mask_X=mX_full, mask_Y=mY_full, per_cherry=per_cherry)
            ref2 += float(t[2])
    assert np.isclose(fast[2], ref2, atol=1e-12), (
        f"both-gap marginalisation: kernel={fast[2]:.10f} ref={ref2:.10f}")


def test_gap_disparate_under_independence_unbiased():
    """Construct two columns under exact AA independence (pi_field
    identical across classes/atoms, single-class K_c=1). Then the merged
    log-likelihood must equal the sum of singleton log-likelihoods,
    regardless of how disparate the gap patterns are.

    This is the bias check: pre-fix code returned merged > split by
    +|discarded subset evidence| (hundreds of nats in the agent's toy);
    post-fix the difference must be 0 within fp noise.
    """
    rng = np.random.default_rng(11)
    A = 20
    # K_c=1, L_max=1: only one class and one field atom; emission factors
    # over sites by definition; no cross-site coupling.
    pi = rng.gamma(2.0, size=(A,)); pi /= pi.sum()
    pi_field = pi[None, None, :]                     # (1, 1, A)
    rho = np.array([1.0])
    n = 40
    tau = rng.uniform(0.1, 0.6, size=n)
    rho_chain = 0.5

    # Two columns; X/Y residues drawn iid from pi.
    L_cols = 2
    aa_a = rng.integers(0, A, size=(n, L_cols))
    aa_b = rng.integers(0, A, size=(n, L_cols))
    # Gap patterns: col 0 observed in 80% of cherries; col 1 observed in
    # 50%; intersection is 40% (roughly: gap-disparate).
    aa_a[rng.random((n, L_cols)) < np.array([0.2, 0.5])[None, :]] = 20
    aa_b[rng.random((n, L_cols)) < np.array([0.2, 0.5])[None, :]] = 20

    per_cherry = _em.precompute_cluster_emission_per_cherry(
        tau=tau, rho=rho, pi_field=pi_field, rho_chain=rho_chain)

    def score(cols):
        cols = np.asarray(cols, dtype=np.int64)
        X_batch = aa_a[:, cols]; Y_batch = aa_b[:, cols]
        mask_X = X_batch < A; mask_Y = Y_batch < A
        totals = _em.cluster_emission_batched(
            classes=np.zeros(cols.shape, dtype=np.int64),
            X_batch=X_batch, Y_batch=Y_batch,
            mask_X=mask_X, mask_Y=mask_Y, per_cherry=per_cherry)
        return float(np.log(np.maximum(totals, 1e-300)).sum())

    ll_merged = score([0, 1])
    ll_split = score([0]) + score([1])
    assert np.isclose(ll_merged, ll_split, atol=1e-10), (
        f"independence-unbiasedness: merged={ll_merged:.6f}, "
        f"split={ll_split:.6f}, diff={ll_merged - ll_split:.6e}")


if __name__ == "__main__":
    test_batched_matches_per_cherry_small()
    test_batched_handles_no_cherries()
    test_batched_singleton_cluster()
    test_gap_marginalisation_collapses_factor()
    test_gap_disparate_under_independence_unbiased()
    print("OK")
