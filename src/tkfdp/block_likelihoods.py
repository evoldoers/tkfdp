"""TKF-DP block-likelihood primitives (post-2026-05-15 unified target).

Replaces the historical first-order-correction relic
(`build_M_tensor_aa_marginal` in aug_phmm.py, gamma-weighted) and the
failed uniform-class-prior interim (`build_M_tensor_classmarg`, also in
aug_phmm.py) with a SINGLE central path that matches the model the K=4
emwarm checkpoint was trained against, modulo deliberate "marginalisation
choice" differences (no Holmes-Rubin sufficient stats; eta=1 plug-in
instead of NB integration).

The unified target distribution for a sequence pair (X, Y) at branch
length t:

    P(X, Y; t) = sum_A pi_TKF92(A | t, theta_indel)
                 * sum_E pi_Ewens-{1,2}(E | alpha_z, N_M(A))
                 * prod_{cells (i, j) in Match(A) \\ V(E)} P_singlet(x_i, y_j; t)
                 * prod_{e=((i,j),(k,l)) in E} P_doublet((x_i, x_k), (y_j, y_l); t)

The two block-likelihood primitives:

    P_singlet(a, b; t) = sum_c pi_c * pi^(c)(a) * P_c(a -> b; t)         (1)

    P_doublet((a, c), (b, d); t)
      = sum_{c1, c2} pi_c(c1) * pi_c(c2)
                  * pi_joint^{c1, c2}(a, c)
                  * P_joint^{c1, c2}((a, c) -> (b, d); t)                (2)

with:

  * pi_c[c]  = empirical class prior from training-set column counts
               (NOT uniform 1/K_c; the trained K=4 checkpoint has
               {0.196, 0.233, 0.328, 0.243}).
  * pi^(c)(a) = state.pi_class[c, a]
  * P_c(a -> b; t) = expm(Q_c * t)[a, b],
              with Q_c = (S_LG08 - diag(S_LG08)) * pi_class[c][None, :]
              (GTR(S_LG08, pi_class[c]) F81-form per-class generator).
  * pi_joint^{c1, c2}(a, c) = joint stationary at coupled cell-pair under
              Potts atom H = atoms[assignments[c1, c2]]; built with
              joint_stationary_pair(H, pi_a, pi_c, ...).
  * P_joint^{c1, c2}((a, c) -> (b, d); t) = expm(Q_pair^{c1, c2}(H) * t)
              evaluated at the pair (a*A+c) -> (b*A+d).

DOUBLET BACKGROUND CONVENTION (`pair_background` argument):

  * 'lg08' (default for THIS K=4 emwarm release): the joint generator
    Q_pair^{c1, c2} is built with pi_a = pi_b = PI_LG08_J. This MATCHES
    how the released Potts atoms were trained (multiclass.py's
    composite_log_likelihood_K calls build_joint_Q(H) with default
    pi=PI_LG08_J).
  * 'per_class': the joint generator uses pi_a = pi_class[c1], pi_b =
    pi_class[c2]. This is the "right" thing for future model variants
    where H is retrained against per-class pair background. NOT
    consistent with the released emwarm Potts atoms.

ETA HANDLING:

Two modes (per Yang 1994 discrete-gamma):

  * `n_rate_bins=1` (default): eta = 1.0 (single point estimate at the
    Gamma(a, b) mean for a = b). Matches the K=4 emwarm release which
    has all per-site eta posteriors collapsed to 1.0.
  * `n_rate_bins=K_r >= 2`: K_r representative rates from the gamma
    prior (Yang's median-of-bin method via the K_r equiprobable
    quantile bins), each weighted 1/K_r. Singleton cost scales by K_r,
    doublet cost by K_r^2. For K_r = 4 and K_c = 4 the doublet path
    runs 256 expm per t -- one precompute, then cell-wise lookup.

The Gamma prior hyperparameters MUST match the base measure used in
the MSA-conditioned MCMC training that produced the checkpoint --
otherwise inference and training are integrating against different
priors and we have a model-spec mismatch. Use
`gamma_hyperparams_from_checkpoint(ckpt_path)` to load the trained
values; only fall back to (a_eta = 2.0, b_eta = 2.0) if you've
verified the checkpoint's `meta.json` carries those exact values
(the K=4 emwarm release does).

CONSISTENCY GUARANTEE (the t = 0 identity):

At t = 0, x = y, deterministic alignment:
  P_singlet(x, x; 0) = sum_c pi_c * pi^(c)(x) = pi_marg(x)
  P_doublet((x, x'), (x, x'); 0)
    = sum_{c1, c2} pi_c(c1) * pi_c(c2) * pi_joint^{c1, c2}(x, x')
  M_boost(x, x', x, x'; 0) = P_doublet / (pi_marg(x) * pi_marg(x'))
                          = (sum class-pair joint stationary at (x, x'))
                            / (sum singleton stationaries product)

For (x, x') = (C, C), this is the +3.64 stationary log-odds verified by
plot_k4_log_odds_vs_time.py (under the LG08-pair-background convention,
since H atoms were trained that way).
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsl

# IMPORTANT: import the F81-normalised exchangeability matrix
# (mean rate = 1.0 at LG08 stationary). The unnormalised S_LG08_J has
# mean rate 0.894 -- using it would give a uniform 10.6% time-scale
# mismatch vs training (which uses S_LG08_F81_J via the same alias in
# generator.py:33 and svi.py:60). math-verifier ERROR 1, 2026-05-15.
from .lg08 import S_LG08_F81_J as S_LG08_J, PI_LG08_J
from .generator import joint_stationary_pair, build_joint_Q_pair


A = 20  # alphabet size


# ---------------------------------------------------------------------------
# Yang 1994 discrete-gamma rate-multiplier representatives
# ---------------------------------------------------------------------------

def discrete_gamma_rates(a_eta: float = 2.0, b_eta: float = 2.0,
                         n_bins: int = 1) -> np.ndarray:
    """Yang 1994 discrete-gamma representative rates, rescaled to mean 1.

    Returns a (n_bins,) array of representative rates from the Gamma(a, b)
    prior (rate parameterisation; mean = a/b). For n_bins = 1 returns
    [a/b] (the prior mean). For n_bins >= 2 returns the medians of the
    n_bins equiprobable bins, then RESCALED so their arithmetic mean is
    exactly 1.0 -- per Yang's 1994 recommendation, the median-of-bin
    method under-represents the heavy right tail (the raw average is
    ~0.955 for Gamma(2, 2) at K=4), and rescaling keeps the integrated
    branch length consistent with the prior mean. Each bin is then
    implicitly weighted 1/n_bins.

    For Gamma(2, 2) (the K=4 emwarm release) at n_bins = 4:
      raw bin medians ~ [0.305, 0.653, 1.059, 1.804] (mean ~0.955)
      after rescaling: [0.319, 0.683, 1.108, 1.889] (mean = 1.000)
    """
    if n_bins <= 1:
        return np.array([a_eta / b_eta], dtype=np.float64)
    from scipy.stats import gamma as _gamma_dist
    scale = 1.0 / b_eta
    quantiles = (np.arange(n_bins) + 0.5) / n_bins
    raw = np.asarray(_gamma_dist.ppf(quantiles, a_eta, scale=scale),
                      dtype=np.float64)
    # Yang 1994: rescale so mean = a/b (the prior mean = 1 for a = b).
    target_mean = a_eta / b_eta
    return raw * (target_mean / raw.mean())


# ---------------------------------------------------------------------------
# Empirical class prior pi_c
# ---------------------------------------------------------------------------

def gamma_hyperparams_from_checkpoint(ckpt_path: str) -> Tuple[float, float]:
    """Load the Gamma(a_eta, b_eta) hyperparameters used by the MSA-
    conditioned MCMC training that produced this checkpoint. These MUST
    be threaded through to inference (build_singlet_emission /
    build_doublet_emission with n_rate_bins >= 2) so the discrete-gamma
    integration is over the SAME prior the model was trained against.

    Looks in `<ckpt_dir>/meta.json` for `a_eta`, `b_eta` keys. Raises if
    not found -- callers should not silently fall back to defaults
    (model-spec mismatch is a real failure mode, see the 2026-05-15
    incident where the relic build_M_tensor_aa_marginal silently used
    the wrong AA encoding).
    """
    import json
    from pathlib import Path
    ckpt = Path(ckpt_path)
    meta_path = ckpt.parent / 'meta.json' if ckpt.is_file() else ckpt / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(
            f"meta.json not found at {meta_path}; cannot load gamma "
            f"hyperparameters. The trained checkpoint must record a_eta "
            f"and b_eta to ensure inference uses the same prior.")
    meta = json.loads(meta_path.read_text())
    if 'a_eta' not in meta or 'b_eta' not in meta:
        raise KeyError(
            f"{meta_path} lacks a_eta or b_eta keys. Add them to the "
            f"training-time meta.json before using rate-bin integration.")
    return float(meta['a_eta']), float(meta['b_eta'])


def empirical_pi_c_from_checkpoint(ckpt_path: str) -> np.ndarray:
    """Compute the empirical class prior pi_c from the cls_* arrays in a
    saved K=4 checkpoint. Sums the per-MSA-column class assignments across
    all training families, then normalises.

    For the released K=4-emwarm-top1000-2026-05-09 checkpoint this gives
    [0.196, 0.233, 0.328, 0.243].
    """
    d = np.load(ckpt_path, allow_pickle=True)
    K_c = int(d['pi_class'].shape[0])
    cls_keys = sorted(k for k in d.files if k.startswith('cls_'))
    totals = np.zeros(K_c, dtype=np.int64)
    for k in cls_keys:
        arr = np.asarray(d[k])
        if arr.size == 0:
            continue
        # Defensive: cls_* must be integer hard-MAP class labels; if it's
        # ever a soft posterior tensor (float, shape (L, K_c)) the
        # `(arr == c)` comparison silently returns 0 for non-exact matches
        # and the empirical prior is wrong.
        if arr.dtype.kind not in 'iu':
            raise TypeError(
                f"cls_* expected integer class labels (hard MAP); "
                f"key {k!r} has dtype {arr.dtype}. If the checkpoint "
                f"stores soft posteriors, sum them along the column axis "
                f"instead of using `==`.")
        for c in range(K_c):
            totals[c] += int((arr == c).sum())
    n = int(totals.sum())
    if n == 0:
        # Fallback: uniform prior. Should not happen for trained checkpoints.
        return np.full(K_c, 1.0 / K_c)
    return totals.astype(np.float64) / n


# ---------------------------------------------------------------------------
# Per-class single-site generator and transition matrix
# ---------------------------------------------------------------------------

def _per_class_Q(pi_class_c: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Q_c = (S - diag(S)) * pi_class[c][None, :], with diag set so rows sum
    to 0. Standard GTR(S, pi_class[c]) F81-form generator -- matches the
    SVI training convention in svi.py:hr_per_class_per_msa (line 106)."""
    S_off = S - np.diag(np.diag(S))
    Q = S_off * pi_class_c[None, :]
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def _per_class_match_transitions(state, t: float, eta: float = 1.0,
                                  S: Optional[np.ndarray] = None
                                  ) -> np.ndarray:
    """(K_c, A, A) tensor P_c(a -> b; t * eta) per class c (single rate)."""
    pi_class = np.asarray(state.pi_class, dtype=np.float64)
    K_c = pi_class.shape[0]
    S_arr = np.asarray(S_LG08_J if S is None else S, dtype=np.float64)
    out = np.zeros((K_c, A, A), dtype=np.float64)
    for c in range(K_c):
        Q_c = _per_class_Q(pi_class[c], S_arr)
        out[c] = np.asarray(jsl.expm(jnp.asarray(Q_c * t * eta)))
    return out


def _per_class_rate_marginal_transitions(state, t: float, *,
                                          rates: np.ndarray,
                                          S: Optional[np.ndarray] = None
                                          ) -> np.ndarray:
    """(K_c, A, A) tensor: P_c(a -> b; t) marginalised over the discrete-
    gamma rate bins `rates` (uniform 1/K_r weights).

    For K_r = 1, equivalent to _per_class_match_transitions with eta = rates[0].
    For K_r >= 2, sums expm(Q_c * t * r_k) over k, averaging.
    """
    pi_class = np.asarray(state.pi_class, dtype=np.float64)
    K_c = pi_class.shape[0]
    K_r = len(rates)
    S_arr = np.asarray(S_LG08_J if S is None else S, dtype=np.float64)
    out = np.zeros((K_c, A, A), dtype=np.float64)
    for c in range(K_c):
        Q_c = _per_class_Q(pi_class[c], S_arr)
        Q_j = jnp.asarray(Q_c)
        acc = np.zeros((A, A), dtype=np.float64)
        for r_k in rates:
            acc += np.asarray(jsl.expm(Q_j * (t * float(r_k))))
        out[c] = acc / K_r
    return out


# ---------------------------------------------------------------------------
# Singleton block: P_singlet(a, b; t) and the (sub_matrix_eff, pi_out_eff)
# decomposition that fits the existing TKF92 emission API.
# ---------------------------------------------------------------------------

def build_singlet_emission(state, t: float, *, eta: float = 1.0,
                            pi_c: Optional[np.ndarray] = None,
                            S: Optional[np.ndarray] = None,
                            n_rate_bins: int = 1,
                            a_eta: float = 2.0, b_eta: float = 2.0
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the TKFDP single-site joint emission tensor and its
    (sub_matrix_eff, pi_out_eff) decomposition.

    Returns:
      P_singlet: (A, A) = sum_c pi_c * pi^(c)(a) * P_c(a -> b; t * eta)
      pi_out_eff: (A,) = sum_c pi_c * pi^(c)(a) (marginal class-mixed
        stationary). Used as Insert/Delete emission AND as the marginal
        denominator of sub_matrix_eff.
      sub_matrix_eff: (A, A) = P_singlet[a, b] / pi_out_eff[a]
        (posterior-class-weighted transition; satisfies
         pi_out_eff[a] * sub_matrix_eff[a, b] == P_singlet[a, b]
         exactly, so the existing make_tkf92_pair_hmm /
         pair_hmm_emissions API consumes (sub_matrix_eff, pi_out_eff)
         unchanged).

    Args:
      state: object with .K_c, .pi_class.
      t: branch length.
      eta: rate multiplier (default 1.0 -- matches K=4 emwarm release).
      pi_c: (K_c,) class prior. Defaults to uniform 1/K_c if not given;
        callers should pass the empirical prior.
      S: exchangeability matrix (defaults to S_LG08_J).
    """
    pi_class = np.asarray(state.pi_class, dtype=np.float64)
    K_c = pi_class.shape[0]
    if pi_c is None:
        pi_c = np.full(K_c, 1.0 / K_c)
    pi_c = np.asarray(pi_c, dtype=np.float64)
    assert pi_c.shape == (K_c,), f"pi_c shape {pi_c.shape} != ({K_c},)"

    if n_rate_bins <= 1:
        # Single-rate path. eta argument is honoured (default 1.0).
        P_match = _per_class_match_transitions(state, t, eta=eta, S=S)
    else:
        # Discrete-gamma path. Ignore eta argument.
        rates = discrete_gamma_rates(a_eta=a_eta, b_eta=b_eta, n_bins=n_rate_bins)
        P_match = _per_class_rate_marginal_transitions(state, t, rates=rates, S=S)
    # P_match: (K, A, A)

    # P_singlet[a, b] = sum_c pi_c[c] * pi_class[c, a] * P_match[c, a, b]
    pi_times = pi_c[:, None, None] * pi_class[:, :, None]  # (K, A, 1)
    P_singlet = (pi_times * P_match).sum(axis=0)            # (A, A)

    # pi_out_eff[a] = sum_c pi_c[c] * pi_class[c, a]
    pi_out_eff = (pi_c[:, None] * pi_class).sum(axis=0)     # (A,)

    sub_matrix_eff = P_singlet / np.clip(pi_out_eff[:, None], 1e-300, None)
    return P_singlet, pi_out_eff, sub_matrix_eff


# ---------------------------------------------------------------------------
# Doublet block: P_doublet[(a, c), (b, d); t]
# ---------------------------------------------------------------------------

def build_doublet_emission(state, t: float, *, eta: float = 1.0,
                            pi_c: Optional[np.ndarray] = None,
                            S: Optional[np.ndarray] = None,
                            pair_background: str,
                            n_rate_bins: int = 1,
                            a_eta: float = 2.0, b_eta: float = 2.0
                            ) -> np.ndarray:
    """Build the (A, A, A, A) doublet emission tensor.

    P_doublet[a, b, c, d] = sum_{c1, c2} pi_c(c1) * pi_c(c2)
                            * pi_joint^{c1, c2}(a, c)
                            * P_joint^{c1, c2}((a, c) -> (b, d); t * eta)

    Args:
      state: K=4 model state with .K_c, .pi_class, .potts_dp.
      t: branch length.
      eta: rate multiplier (default 1.0).
      pi_c: (K_c,) class prior (defaults to uniform 1/K_c; callers
        should pass empirical).
      S: exchangeability matrix.
      pair_background: which stationary to use for the per-(c1, c2)
        joint generator and joint stationary:
          'lg08' (default; matches K=4 emwarm release training):
            pi_a = pi_b = PI_LG08_J for every (c1, c2).
          'per_class' (for future retrained models):
            pi_a = pi_class[c1], pi_b = pi_class[c2].

    Index convention (matches build_M_tensor_aa_marginal in aug_phmm.py):
      P_doublet[a, b, c, d]: left endpoint cell carries AAs (a, b) =
      (X_i, Y_j); right endpoint cell carries (c, d) = (X_{i'}, Y_{j'}).
    """
    pi_class = np.asarray(state.pi_class, dtype=np.float64)
    K_c = pi_class.shape[0]
    atoms = np.asarray(state.potts_dp.atoms, dtype=np.float64)        # (K_H, A, A)
    assignments = np.asarray(state.potts_dp.assignments, dtype=np.int64)  # (K_c, K_c)
    h_pairs = state.potts_dp.h_pairs
    use_h = h_pairs is not None
    if use_h:
        h_pairs = np.asarray(h_pairs, dtype=np.float64)
        from .potts_dp import canonical_pair_idx_table
        cp_idx_np, cp_swap_np = canonical_pair_idx_table(K_c)

    if pi_c is None:
        pi_c = np.full(K_c, 1.0 / K_c)
    pi_c = np.asarray(pi_c, dtype=np.float64)

    if pair_background not in ('lg08', 'per_class'):
        raise ValueError(f"pair_background must be 'lg08' or 'per_class', "
                          f"got {pair_background!r}")

    S_arr = np.asarray(S_LG08_J if S is None else S, dtype=np.float64)
    pi_lg08 = np.asarray(PI_LG08_J, dtype=np.float64)

    # Discrete-gamma rate bins. Build the per-site rate-pair grid
    # (r_k1, r_k2) once; weight each pair by 1/K_r^2.
    if n_rate_bins <= 1:
        rates = np.array([eta], dtype=np.float64)
    else:
        rates = discrete_gamma_rates(a_eta=a_eta, b_eta=b_eta,
                                      n_bins=n_rate_bins)
    K_r = len(rates)
    rate_pairs = [(float(r1), float(r2)) for r1 in rates for r2 in rates]
    rate_weight = 1.0 / (K_r * K_r)

    out = np.zeros((A, A, A, A), dtype=np.float64)
    for c1 in range(K_c):
        for c2 in range(K_c):
            atom_idx = int(assignments[c1, c2])
            H = jnp.asarray(atoms[atom_idx])
            if use_h:
                k_can = int(cp_idx_np[c1, c2])
                swap = int(cp_swap_np[c1, c2])
                h_a_jnp = jnp.asarray(h_pairs[k_can, swap])
                h_b_jnp = jnp.asarray(h_pairs[k_can, 1 - swap])
            else:
                h_a_jnp = h_b_jnp = None

            # Choose the stationary background for this (c1, c2) pair.
            if pair_background == 'lg08':
                pi_a_np = pi_lg08
                pi_b_np = pi_lg08
            else:  # 'per_class'
                pi_a_np = pi_class[c1]
                pi_b_np = pi_class[c2]
            pi_a_jnp = jnp.asarray(pi_a_np)
            pi_b_jnp = jnp.asarray(pi_b_np)

            # Joint stationary at coupled site -- INDEPENDENT of rate
            # multipliers (they only scale Q, not pi_joint).
            pij_flat = joint_stationary_pair(H, pi_a_jnp, pi_b_jnp,
                                              h_a=h_a_jnp, h_b=h_b_jnp)
            pij = np.asarray(pij_flat).reshape(A, A)        # (a, c)

            # Sum over (r_k1, r_k2) rate-pair bins. Each bin builds a
            # different 400x400 joint generator (per-site rates differ)
            # and exp's it.
            pij_w = pi_c[c1] * pi_c[c2] * rate_weight       # scalar
            for (r1, r2) in rate_pairs:
                Q_joint = build_joint_Q_pair(H, pi_a_jnp, pi_b_jnp,
                                              S=jnp.asarray(S_arr),
                                              eta_pair=(r1, r2),
                                              h_a=h_a_jnp, h_b=h_b_jnp)
                P_flat = jsl.expm(Q_joint * t)
                P_joint = np.asarray(P_flat).reshape(A, A, A, A)
                out += pij_w * pij[:, :, None, None] * P_joint

    # Reorder to (a, b, c, d) per the boost API: left endpoint (a, b),
    # right endpoint (c, d).
    return np.transpose(out, (0, 2, 1, 3))


# ---------------------------------------------------------------------------
# M-boost tensor: P_doublet / (P_singlet x P_singlet)
# ---------------------------------------------------------------------------

def build_M_tensor(state, t: float, *, eta: float = 1.0,
                    pi_c: Optional[np.ndarray] = None,
                    S: Optional[np.ndarray] = None,
                    pair_background: str,
                    n_rate_bins: int = 1,
                    a_eta: float = 2.0, b_eta: float = 2.0
                    ) -> np.ndarray:
    """Build the (A, A, A, A) edge-boost tensor

      M[a, b, c, d] = P_doublet[a, b, c, d] / (P_singlet[a, b] * P_singlet[c, d])

    This is the multiplicative likelihood ratio for converting two
    independent singletons at observed AAs (a, b) and (c, d) into a
    coupled doublet. Used by the MCMC chain's edge add/remove MH ratio.

    Same arguments as build_doublet_emission. See the module docstring
    for the consistency guarantee at t = 0, x = y.
    """
    P_singlet, _, _ = build_singlet_emission(
        state, t, eta=eta, pi_c=pi_c, S=S,
        n_rate_bins=n_rate_bins, a_eta=a_eta, b_eta=b_eta)
    P_doublet = build_doublet_emission(
        state, t, eta=eta, pi_c=pi_c, S=S,
        pair_background=pair_background,
        n_rate_bins=n_rate_bins, a_eta=a_eta, b_eta=b_eta)
    denom = P_singlet[:, :, None, None] * P_singlet[None, None, :, :]
    return P_doublet / np.clip(denom, 1e-300, None)
