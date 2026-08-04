"""JAX-batched cluster scorer regression tests."""
from __future__ import annotations

import numpy as np

from tkfdp.coupling.dynfield import emission as _em
from tkfdp.partition_K import (FamilyKState, cluster_id_from_partner,
                                  gibbs_sweep_cluster)
from tkfdp.svi import make_cluster_loglik_fn


def _make_state(rng, K_c=4, L_max=4, A=20):
    raw = rng.gamma(2.0, size=(K_c, L_max, A))
    pi_field = raw / raw.sum(axis=-1, keepdims=True)
    rho_raw = rng.gamma(2.0, size=(L_max,))
    rho = rho_raw / rho_raw.sum()
    return pi_field, rho


def _fake_msa(rng, L=30, n_cherries=8, A=20):
    aa_a = rng.integers(0, A, size=(n_cherries, L), dtype=np.int8)
    aa_b = rng.integers(0, A, size=(n_cherries, L), dtype=np.int8)
    # Insert some gaps so both_aa has variety.
    aa_a[rng.random((n_cherries, L)) < 0.1] = 20
    aa_b[rng.random((n_cherries, L)) < 0.1] = 20
    tau = rng.uniform(0.05, 0.7, size=n_cherries)
    both_aa = (aa_a < A) & (aa_b < A)
    return aa_a, aa_b, both_aa, tau


def test_batched_score_matches_numpy():
    rng = np.random.default_rng(0)
    pi_field, rho = _make_state(rng)
    aa_a, aa_b, both_aa, tau = _fake_msa(rng)
    L = aa_a.shape[1]
    cls = rng.integers(0, 4, size=L).astype(np.int32)

    per_cherry = _em.precompute_cluster_emission_per_cherry(
        tau=tau, rho=rho, pi_field=pi_field, rho_chain=0.5)
    scorer = _em.BatchedDynfieldScorer(
        aa_a=aa_a, aa_b=aa_b, both_aa=both_aa, tau=tau,
        rho=rho, pi_field=pi_field, rho_chain=0.5, max_cluster_size=8)

    candidates = []
    for _ in range(15):
        sz = int(rng.integers(1, 9))
        candidates.append(rng.choice(L, size=sz, replace=False).tolist())

    jax_scores = scorer.score_batch(candidates, cls)
    ref = []
    for c in candidates:
        cols = np.asarray(c)
        X = aa_a[:, cols]; Y = aa_b[:, cols]
        mX = X < 20; mY = Y < 20
        totals = _em.cluster_emission_batched(
            classes=cls[cols],
            X_batch=X, Y_batch=Y,
            mask_X=mX, mask_Y=mY,
            per_cherry=per_cherry)
        ref.append(float(np.log(np.maximum(totals, 1e-300)).sum()))
    ref = np.array(ref)
    assert np.allclose(jax_scores, ref, atol=1e-9, rtol=1e-9), (
        f"max diff = {np.max(np.abs(jax_scores - ref))}")


def test_gibbs_sweep_jax_matches_scalar():
    """Under the same RNG, the JAX-batched and scalar Gibbs sweeps should
    produce bit-exact cluster_id outputs (since both pipelines compute
    bit-exact log_p)."""
    rng = np.random.default_rng(0)
    pi_field, rho = _make_state(rng)
    aa_a, aa_b, both_aa, tau = _fake_msa(rng)
    L = aa_a.shape[1]
    cls = rng.integers(0, 4, size=L).astype(np.int32)

    class _DM: pass
    m = _DM(); df = _DM()
    df.pi_field = pi_field; df.rho = rho; df.rho_chain = 0.5
    m.dyn_field = df

    fn = make_cluster_loglik_fn(m, aa_a, aa_b, both_aa, tau, cls)
    scorer = _em.BatchedDynfieldScorer(
        aa_a=aa_a, aa_b=aa_b, both_aa=both_aa, tau=tau,
        rho=rho, pi_field=pi_field, rho_chain=0.5, max_cluster_size=8)
    _ = scorer.score_batch([[0]], cls)

    init_partner = np.full(L, -1, dtype=np.int32)
    init_cluster_id = cluster_id_from_partner(init_partner)

    def mk():
        return FamilyKState(family='X', L=L, K=4,
                             partner=init_partner.copy(),
                             cls=cls.copy().astype(np.int64),
                             cluster_id=init_cluster_id.copy())

    sa = mk(); rs = np.random.default_rng(42)
    gibbs_sweep_cluster(sa, fn, rs, alpha_z=1.0, max_cluster_size=8)
    sb = mk(); rb = np.random.default_rng(42)
    bfn = lambda reqs: scorer.score_batch(reqs, cls)
    gibbs_sweep_cluster(sb, fn, rb, alpha_z=1.0, max_cluster_size=8,
                        batched_score_fn=bfn)
    assert np.array_equal(sa.cluster_id, sb.cluster_id), (
        f"scalar={sa.cluster_id}\njax={sb.cluster_id}")


if __name__ == "__main__":
    test_batched_score_matches_numpy()
    test_gibbs_sweep_jax_matches_scalar()
    print("OK")
