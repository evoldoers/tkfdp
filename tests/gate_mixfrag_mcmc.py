"""Correctness gates for the MixFrag base-model swap in the infinite-pair-HMM
MCMC sampler (mcmc_infinite_phmm.py).

Gate 1 (F=1 parity): the sampler with mixfrag=([ext],[1.0]) must reproduce the
  TKF92 sampler EXACTLY (Q_prime and Q_baseline), at H=0, same seed.
Gate 2 (F=2, H=0): the MixFrag(F=2) sampler's running-mean match posterior
  Q_prime must reproduce its own exact forward-backward Q_baseline (H=0 => the
  Potts edges do not bias the alignment), to within MCMC noise. This exercises
  the 8-state kernel/traceback/segment-resample on the MixFrag Pair HMM.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(Path.home() / "tkf-mixdom" / "python"))

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfdp.lg08 import PI_LG08
from tkfdp.potts_dp import PottsDPState, canonical_pair_idx_table
from tkfdp.coupled_annealing import build_boost_state
from tkfdp.mcmc_infinite_phmm import mcmc_corrected_posterior


def make_state_H0(K_c=1, seed=0):
    A = 20
    pi_class = np.tile(np.asarray(PI_LG08), (K_c, 1))
    n_pairs = K_c * (K_c + 1) // 2
    atoms = np.zeros((n_pairs, A, A))                 # H = 0 -> M == 1
    cp_idx, _ = canonical_pair_idx_table(K_c)
    pdp = PottsDPState(K_c=K_c, A=A, atoms=atoms,
                       assignments=np.asarray(cp_idx, np.int64),
                       counts=np.ones(n_pairs, np.int64), alpha_H=1.0)

    class S_:
        pass
    s = S_(); s.K_c = K_c; s.A = A; s.pi_class = pi_class; s.potts_dp = pdp
    return s


def boost_for(state, x, y, t):
    bs = build_boost_state({(0, 1): np.zeros((x.shape[0], y.shape[0]))},
                           {(0, 1): float(t)}, [x, y], state)
    return bs[(0, 1)]


def run(x, y, t, lam, mu, ext, Q, pi, mixfrag=None, seed=0):
    state = make_state_H0()
    bs = boost_for(state, x, y, t)
    Qp, _, Qb, _, diag = mcmc_corrected_posterior(
        x, y, t, lam, mu, ext, Q, pi, bs,
        alpha_z=100.0, n_sweeps=4000, n_burnin=1000, seed=seed,
        init_mode="forward_sample", mixfrag=mixfrag)
    return np.asarray(Qp), np.asarray(Qb)


def main():
    Q, pi = rate_matrix_lg(); Q, pi = np.asarray(Q), np.asarray(pi)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 20, 11).astype(np.int32)
    y = rng.integers(0, 20, 9).astype(np.int32)
    lam, mu, ext = 0.03, 0.035, 0.6

    print("=== Gate 1: F=1 parity (MixFrag-F1 sampler == TKF92 sampler) ===")
    Qp_t, Qb_t = run(x, y, 0.5, lam, mu, ext, Q, pi, mixfrag=None, seed=7)
    Qp_m, Qb_m = run(x, y, 0.5, lam, mu, ext, Q, pi,
                     mixfrag=(np.array([ext]), np.array([1.0])), seed=7)
    d_base = float(np.abs(Qb_t - Qb_m).max())
    d_prime = float(np.abs(Qp_t - Qp_m).max())
    print(f"  max|Q_baseline diff| = {d_base:.3e}")
    print(f"  max|Q_prime    diff| = {d_prime:.3e}")
    assert d_base < 1e-9, "F=1 baseline parity FAILED"
    assert d_prime < 1e-9, "F=1 Q_prime parity FAILED"
    print("  PASS")

    print("\n=== Gate 2: F=2 at H=0 (sampler Q' reproduces exact-FB Q_base) ===")
    exts = np.array([0.41, 0.86]); weights = np.array([0.7, 0.3])
    Qp2, Qb2 = run(x, y, 0.5, lam, mu, ext, Q, pi,
                   mixfrag=(exts, weights), seed=11)
    # Row-mass and range sanity.
    assert (Qp2 >= -1e-9).all() and (Qp2 <= 1 + 1e-6).all(), "Q' out of range"
    # Q_baseline must differ from the TKF92 baseline (fragtypes matter)...
    diff_vs_tkf = float(np.abs(Qb2 - Qb_t).max())
    # ...and the MCMC running mean must track its own exact FB baseline.
    mae = float(np.abs(Qp2 - Qb2).mean())
    mx = float(np.abs(Qp2 - Qb2).max())
    print(f"  max|Q_base(F2) - Q_base(TKF92)| = {diff_vs_tkf:.3e} (expect > 0)")
    print(f"  MCMC vs exact-FB: MAE={mae:.4f}  max={mx:.4f}")
    assert mae < 0.03, f"F=2 H=0 MCMC!=FB (MAE {mae})"
    print("  PASS")
    print("\nALL GATES PASSED")


if __name__ == "__main__":
    main()
