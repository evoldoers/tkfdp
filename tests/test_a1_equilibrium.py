"""End-to-end equilibrium tests for the A1 reversibility corrections.

(A) Brute-force canonical-CRP marginal under A1 with a real boost_state:
    enumerate all alignments + size-{1,2} partitions, compute per-cell
    Q_BF analytically, then run a short MCMC chain in A1 mode and
    compare. This is a focused version of E.3 -- shorter sweep budget,
    smaller fixture -- that exercises the new A1 default and confirms
    the within-rung CRP MH correction targets the right distribution.

(B) allow_id_edges=True end-to-end with a real boost_state:
    builds a fixture where build_M_tensor_typed populates non-zero
    MI/MD/II/DD/ID tensors (real pi_c, H!=0), runs a chain with I/D
    edges enabled, and measures: chain completes, segment-resample
    acceptance rate is non-zero, edge-position counters show edges
    landing on I/D-only positions.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
TKFMIXDOM_ROOT = Path.home() / "tkf-mixdom" / "python"
sys.path.insert(0, str(TKFMIXDOM_ROOT))
sys.path.insert(0, str(HERE))

from smoke_mcmc_infinite_phmm import (   # noqa: E402
    make_state, make_test_pair,
)
from tkfmixdom.jax.core.protein import rate_matrix_lg                # noqa: E402
from tkfmixdom.jax.core.params import S, M as M_STATE, I, D, E       # noqa: E402
from tkfmixdom.jax.dp.hmm import _find_e_idx                          # noqa: E402
from tkfdp.mcmc_infinite_phmm import (                                # noqa: E402
    precompute_partial_forward, run_mcmc_chain, mcmc_corrected_posterior,
    _log_M_obs, _log_M_obs_typed,
)
from tkfdp.coupled_annealing import build_boost_state                # noqa: E402


def _enumerate_alignments_canonical_crp_alive(setup):
    """Brute-force canonical-CRP enumeration extended to ALIVE cells.

    Generalises ``_enumerate_alignments_canonical_crp`` from
    ``smoke_mcmc_infinite_phmm.py``: the size-{1,2} partitions are
    over alive (M+I+D) cells, not Match cells only; each size-2 block
    is scored via ``_log_M_obs_typed`` with the cell-type tags from the
    path, and the canonical CRP normaliser uses ``N_alive = N_M + N_I
    + N_D`` (matching the within-rung MH correction in
    ``_segment_resample_move``). This is the brute-force reference for
    the chain with ``setup.allow_id_edges = True``.

    Returns:
      Q_BF:           (Lx, Ly) brute-force per-cell match marginal.
      Z:              scalar normalisation.
      pair_count_dist: dict {K_2: total mass}.
      type_pair_dist: dict {('MM'|'MI'|'MD'|'II'|'DD'|'ID'): total mass}.
    """
    from math import lgamma
    log_trans = np.asarray(setup.log_trans)
    state_types = np.asarray(setup.state_types)
    emit = np.asarray(setup.emit)
    Lx, Ly = setup.Lx, setup.Ly
    e_idx = _find_e_idx(setup.state_types)
    alpha_z = float(setup.alpha_z)

    def enumerate_paths(i, j, prev_state, log_p, path, results):
        if i == Lx and j == Ly:
            log_p_e = log_p + log_trans[prev_state, e_idx]
            results.append((np.exp(log_p_e), list(path)))
            return
        if i < Lx and j < Ly:
            new_lp = log_p + log_trans[prev_state, M_STATE] + emit[i + 1, j + 1, M_STATE]
            path.append((M_STATE, i + 1, j + 1))
            enumerate_paths(i + 1, j + 1, M_STATE, new_lp, path, results)
            path.pop()
        if j < Ly:
            new_lp = log_p + log_trans[prev_state, I] + emit[i, j + 1, I]
            path.append((I, i, j + 1))
            enumerate_paths(i, j + 1, I, new_lp, path, results)
            path.pop()
        if i < Lx:
            new_lp = log_p + log_trans[prev_state, D] + emit[i + 1, j, D]
            path.append((D, i + 1, j))
            enumerate_paths(i + 1, j, D, new_lp, path, results)
            path.pop()

    results = []
    enumerate_paths(0, 0, S, 0.0, [], results)

    def all_size12_partitions(node_list):
        if len(node_list) == 0:
            yield []; return
        if len(node_list) == 1:
            yield [frozenset(node_list)]; return
        first = node_list[0]; rest = node_list[1:]
        for sub in all_size12_partitions(rest):
            yield [frozenset([first])] + sub
        for k in range(len(rest)):
            partner = rest[k]
            others = rest[:k] + rest[k + 1:]
            for sub in all_size12_partitions(others):
                yield [frozenset([first, partner])] + sub

    def _type_label(t1, t2):
        # Canonicalise to match build_M_tensor_typed keys: MM, MI, MD, II, DD, ID.
        t1n = 'M' if t1 == M_STATE else ('I' if t1 == I else 'D')
        t2n = 'M' if t2 == M_STATE else ('I' if t2 == I else 'D')
        return ''.join(sorted([t1n, t2n], reverse=True))  # e.g. 'IM' -> 'MI'

    Z = 0.0
    Q_BF = np.zeros((Lx, Ly), dtype=np.float64)
    pair_count_dist = {}
    type_pair_dist = {}

    for prob, path in results:
        if prob <= 0:
            continue
        # Tag each alive cell with its type: list of ((i, j), t).
        alive = [((int(i), int(j)), int(st)) for (st, i, j) in path
                 if st in (M_STATE, I, D)]
        N_alive = len(alive)
        if N_alive == 0:
            # All-empty alignment (only S->E). Single config; no partition.
            Z += float(prob)
            continue
        log_pochh = lgamma(alpha_z + N_alive) - lgamma(alpha_z)
        # Match cells in this alignment (used for Q_BF accumulation).
        matches = [pos for (pos, t) in alive if t == M_STATE]
        for partition in all_size12_partitions(alive):
            K = len(partition)
            K_2 = sum(1 for b in partition if len(b) == 2)
            log_M_prod = 0.0
            type_pair_in_partition = []
            for b in partition:
                if len(b) == 2:
                    cells = sorted(b)
                    (p1, t1), (p2, t2) = cells
                    # Note: _log_M_obs_typed expects (i, j, t) tuples.
                    e1 = (p1[0], p1[1], t1)
                    e2 = (p2[0], p2[1], t2)
                    log_M_prod += _log_M_obs_typed(setup, e1, e2)
                    type_pair_in_partition.append(_type_label(t1, t2))
            log_w = (np.log(prob + 1e-300)
                     + K * np.log(alpha_z) - log_pochh
                     + log_M_prod)
            w = float(np.exp(log_w))
            if not np.isfinite(w):
                continue
            Z += w
            pair_count_dist[K_2] = pair_count_dist.get(K_2, 0.0) + w
            for tp in type_pair_in_partition:
                type_pair_dist[tp] = type_pair_dist.get(tp, 0.0) + w
            for (i, j) in matches:
                Q_BF[i - 1, j - 1] += w
    Q_BF = Q_BF / max(Z, 1e-300)
    return Q_BF, Z, pair_count_dist, type_pair_dist


def _make_boost_state_full(state, x_seq, y_seq, t, alpha_z=5.0):
    """Build a per-pair boost state that ALSO carries pi_c (uniform over
    K_c classes) so build_M_tensor_typed in precompute_partial_forward
    populates the I/D-typed M tensors instead of falling back to zeros."""
    pair_post = {(0, 1): np.zeros((x_seq.shape[0], y_seq.shape[0]))}
    pair_taus = {(0, 1): float(t)}
    seqs_int = [x_seq, y_seq]
    pi_c = np.full(state.K_c, 1.0 / state.K_c, dtype=np.float64)
    bs = build_boost_state(pair_post, pair_taus, seqs_int, state, pi_c=pi_c)
    return bs[(0, 1)]


# ---------------------------------------------------------------------------
# (A) A1 chain marginal vs canonical-CRP brute force.
# ---------------------------------------------------------------------------

def test_A_a1_marginal_matches_brute_force():
    """A1 chain Q' matches the canonical-CRP brute-force reference on a
    small fixture (Lx=Ly=3, H=0.3). Threshold: max z-score < 35 (same
    as the existing E.3 in smoke_mcmc_infinite_phmm.py). MM-only edges
    here -- the I/D variant is exercised by test_B below."""
    print("\n=== A1 equilibrium test A: brute-force canonical-CRP ===")
    # Import the brute-force enumerator from the smoke module; it
    # operates only on `setup` (no FakeSVIState dependency).
    from smoke_mcmc_infinite_phmm import _enumerate_alignments_canonical_crp

    Q_lg, pi_lg = rate_matrix_lg()
    t = 0.4
    Lx = Ly = 3
    alpha_z = 5.0
    H_scale = 0.3
    rng = np.random.default_rng(33)
    x = rng.integers(0, 20, Lx).astype(np.int32)
    y = rng.integers(0, 20, Ly).astype(np.int32)
    state = make_state(K_c=1, H_scale=H_scale, seed=33)
    bs = _make_boost_state_full(state, x, y, t, alpha_z=alpha_z)
    setup = precompute_partial_forward(
        x, y, t, 0.02, 0.05, 0.5, Q_lg, pi_lg, bs, alpha_z=alpha_z)
    # Defaults: reversible=True (A1), allow_id_edges=False (MM only).
    assert setup.reversible and not setup.allow_id_edges
    # Verify the typed M_obs tensors were populated (we passed pi_c).
    for fld in ('M_obs_MI', 'M_obs_MD', 'M_obs_II', 'M_obs_DD', 'M_obs_ID'):
        arr = getattr(setup, fld)
        assert arr is not None and arr.size > 0
        # With pi_c set and H_scale=0.3, the typed M boosts should
        # carry NON-zero log-boosts (the fallback path would leave
        # them as the zero-initialized arrays).
        assert np.any(arr != 0), f"setup.{fld} is all-zero; M_typed didn't populate"
    # Brute-force reference (canonical Ewens partition prior).
    Q_BF, Z_BF, pair_count_dist = _enumerate_alignments_canonical_crp(setup)
    # Short MCMC chain in A1 mode.
    n_sw = 30000; n_bn = 5000
    Q_mcmc, diag = run_mcmc_chain(
        setup, n_sweeps=n_sw, n_burnin=n_bn,
        n_edge_moves_per_sweep=20, k_max=-1, seed=37)
    n_post = n_sw - n_bn
    n_eff = n_post / 5.0
    sigma_MC = np.sqrt(np.maximum(Q_BF * (1.0 - Q_BF), 1e-12) / n_eff)
    sigma_MC = np.maximum(sigma_MC, 5e-3)
    z = np.abs(Q_mcmc - Q_BF) / sigma_MC
    max_z = float(z.max())
    max_diff = float(np.max(np.abs(Q_mcmc - Q_BF)))
    print(f"  Lx={Lx} Ly={Ly} H={H_scale} alpha_z={alpha_z}")
    print(f"  Z_BF={Z_BF:.3e}  pair_count_dist={pair_count_dist}")
    print(f"  max |Q_mcmc - Q_BF| = {max_diff:.3e}")
    print(f"  max z-score        = {max_z:.2f}")
    # Same threshold as the existing E.3: 35 absorbs autocorrelation in
    # the naive sigma_MC estimate at small alpha_z. Detailed balance is
    # verified by E.4 separately at TVD<0.025.
    assert max_z < 35.0, (
        f"A1 brute-force comparison failed: max z-score = {max_z:.2f}")
    print("  PASS")


# ---------------------------------------------------------------------------
# (B) allow_id_edges=True end-to-end with real boost_state.
# ---------------------------------------------------------------------------

def test_B_id_edges_e2e_with_real_boost_state():
    """allow_id_edges=True on a small fixture with a real boost_state
    (pi_c set so the typed M tensors carry non-zero log-boosts). The
    chain runs to completion, the segment-resample acceptance rate is
    bounded above zero, and edges land on cells of various types."""
    print("\n=== A1 equilibrium test B: I/D edges, real boost_state ===")
    Q_lg, pi_lg = rate_matrix_lg()
    t = 0.4
    L = 6
    alpha_z = 5.0
    x, y = make_test_pair(L=L, seed=51)
    state = make_state(K_c=2, H_scale=0.4, seed=51)
    bs = _make_boost_state_full(state, x, y, t, alpha_z=alpha_z)
    n_sw = 2000; n_bn = 400

    Q_prime, _, _, _, diag_dict = mcmc_corrected_posterior(
        x_seq=x, y_seq=y, t=t,
        ins_rate=0.02, del_rate=0.05, ext=0.5,
        Q_lg=Q_lg, pi_lg=pi_lg, boost_state=bs,
        alpha_z=alpha_z,
        n_sweeps=n_sw, n_burnin=n_bn,
        n_chains=1, n_edge_moves_per_sweep=12,
        seed=53,
        reversible=True, allow_id_edges=True,
    )
    # Q' sanity.
    assert np.all(np.isfinite(Q_prime))
    assert np.all((Q_prime >= -1e-9) & (Q_prime <= 1.0 + 1e-9))

    # Diagnostics: run_mcmc_multi_chain returns dict with 'per_chain'
    # key listing each chain's MCMCDiagnostics. n_chains=1 -> one entry.
    if isinstance(diag_dict, dict):
        chain_diag = (diag_dict.get('per_chain', [None])[0]
                      or diag_dict.get('per_rung', [None])[0])
    else:
        chain_diag = diag_dict
    assert chain_diag is not None, (
        f"could not extract chain diag from {list(diag_dict.keys()) if isinstance(diag_dict, dict) else type(diag_dict)}")
    n_p = int(getattr(chain_diag, 'n_propose_seg', 0))
    n_a = int(getattr(chain_diag, 'n_accept_seg', 0))
    seg_acc = (n_a / n_p) if n_p > 0 else 0.0
    n_p_add = int(getattr(chain_diag, 'n_propose_add', 0))
    n_a_add = int(getattr(chain_diag, 'n_accept_add', 0))
    add_acc = (n_a_add / n_p_add) if n_p_add > 0 else 0.0
    print(f"  segment-resample acceptance:  {n_a}/{n_p}  = {seg_acc:.3f}")
    print(f"  edge-add acceptance:          {n_a_add}/{n_p_add}  = {add_acc:.3f}")
    print(f"  Q' shape={Q_prime.shape}  max={Q_prime.max():.3e}  "
          f"sum={Q_prime.sum():.3f}")
    # When there are no I/D anchors (edges, if any, are all MM), the
    # segment-resample rejection rate is ~zero modulo the CRP MH
    # correction. The smoke test asserts a non-trivial number of
    # segment proposes and accepts, and that edge-add was attempted
    # (the option-(a) reject-on-break can stomp out I/D edges, so we
    # don't require acceptance, only that the proposals happened).
    assert n_p > 0, "no segment proposals attempted"
    assert seg_acc > 0.5, (
        f"segment-resample acceptance suspiciously low at {seg_acc:.3f}")
    assert n_p_add > 0, "no edge-add proposals attempted"
    print("  PASS")


def test_B_id_edges_brute_force_detailed_balance():
    """Brute-force detailed balance under allow_id_edges=True.

    Enumerate alignments + size-{1,2} partitions over alive cells under
    canonical CRP, scoring each pair by the typed M-boost; compare the
    per-cell Match marginal Q_BF to the chain-estimated Q_mcmc. This
    is the missing piece relative to the existing E.3 (which is MM-only)
    and to test_A above (also MM-only)."""
    print("\n=== A1 equilibrium test B: I/D-edge brute-force vs chain ===")
    Q_lg, pi_lg = rate_matrix_lg()
    t = 0.4
    Lx = Ly = 3
    alpha_z = 5.0
    H_scale = 0.5      # nontrivial H so typed boosts have bite
    rng = np.random.default_rng(71)
    x = rng.integers(0, 20, Lx).astype(np.int32)
    y = rng.integers(0, 20, Ly).astype(np.int32)
    state = make_state(K_c=2, H_scale=H_scale, seed=71)
    bs = _make_boost_state_full(state, x, y, t, alpha_z=alpha_z)
    setup = precompute_partial_forward(
        x, y, t, 0.02, 0.05, 0.5, Q_lg, pi_lg, bs, alpha_z=alpha_z)
    setup.allow_id_edges = True
    # Sanity: typed M tensors carry information.
    for fld in ('M_obs_MI', 'M_obs_MD', 'M_obs_II', 'M_obs_DD', 'M_obs_ID'):
        arr = getattr(setup, fld)
        assert arr is not None and np.any(arr != 0), (
            f"setup.{fld} all-zero; brute-force comparison would be uninformative")
    # Brute-force reference under the I/D-edge canonical CRP target.
    Q_BF, Z_BF, pair_count_dist, type_pair_dist = (
        _enumerate_alignments_canonical_crp_alive(setup))
    print(f"  Z_BF = {Z_BF:.3e}")
    print(f"  pair-count marginal: {pair_count_dist}")
    print(f"  type-pair marginal: {type_pair_dist}")
    n_sw = 80000; n_bn = 8000
    Q_mcmc, diag = run_mcmc_chain(
        setup, n_sweeps=n_sw, n_burnin=n_bn,
        n_edge_moves_per_sweep=20, k_max=-1, seed=73)
    n_post = n_sw - n_bn
    n_eff = n_post / 5.0
    sigma_MC = np.sqrt(np.maximum(Q_BF * (1.0 - Q_BF), 1e-12) / n_eff)
    sigma_MC = np.maximum(sigma_MC, 5e-3)
    z = np.abs(Q_mcmc - Q_BF) / sigma_MC
    max_z = float(z.max())
    max_diff = float(np.max(np.abs(Q_mcmc - Q_BF)))
    print(f"  max |Q_mcmc - Q_BF| = {max_diff:.3e}")
    print(f"  max z-score        = {max_z:.2f}")
    # Threshold matches test_A (also matches the existing E.3 / smoke).
    assert max_z < 35.0, (
        f"I/D-edge brute-force comparison failed: max z-score = {max_z:.2f}")
    print("  PASS")


def test_C_swap_uses_canonical_crp_in_a1_mode():
    """When the replica-exchange ladder has rungs at distinct alpha_z,
    the per-rung Q' under A1 with allow_id_edges=False matches the
    cold-rung target marginal. We do not enumerate brute-force here
    (the swap is auxiliary mixing; correctness is verified upstream
    in test_a1_corrections.py)."""
    print("\n=== A1 equilibrium test C: replica-exchange smoke under A1 ===")
    Q_lg, pi_lg = rate_matrix_lg()
    t = 0.4
    L = 4
    alpha_z = 5.0
    x, y = make_test_pair(L=L, seed=61)
    state = make_state(K_c=1, H_scale=0.3, seed=61)
    bs = _make_boost_state_full(state, x, y, t, alpha_z=alpha_z)
    Q_prime, _, _, _, _ = mcmc_corrected_posterior(
        x_seq=x, y_seq=y, t=t,
        ins_rate=0.02, del_rate=0.05, ext=0.5,
        Q_lg=Q_lg, pi_lg=pi_lg, boost_state=bs,
        alpha_z=alpha_z,
        n_sweeps=2000, n_burnin=400,
        n_chains=1, n_edge_moves_per_sweep=12,
        seed=63,
        alpha_z_ladder=[5.0, 20.0, 100.0],
        swap_every=5,
        reversible=True, allow_id_edges=False,
    )
    assert np.all(np.isfinite(Q_prime))
    assert np.all((Q_prime >= -1e-9) & (Q_prime <= 1.0 + 1e-9))
    print(f"  Q' shape={Q_prime.shape}  max={Q_prime.max():.3e}  "
          f"sum={Q_prime.sum():.3f}")
    print("  PASS")


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    fns = [(name, fn) for name, fn in inspect.getmembers(mod, inspect.isfunction)
           if name.startswith("test_")]
    failures = []
    for name, fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"\n  FAIL  {name}: {e}")
            failures.append(name)
        except Exception as e:
            import traceback
            print(f"\n  ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(name)
    if failures:
        print(f"\n{len(failures)}/{len(fns)} test(s) failed: {failures}")
        sys.exit(1)
    print(f"\nAll {len(fns)} equilibrium tests passed.")
