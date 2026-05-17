"""LG08 protein exchangeabilities and equilibrium frequencies.

Numeric values (190 lower-triangle exchangeabilities + 20 frequencies)
copied verbatim from ~/tkf-mixdom/python/tkfmixdom/jax/core/protein.py,
which itself sources them from the published LG matrix in PAML format.

Exposes S, pi in *alphabetical* AA order (ACDEFGHIKLMNPQRSTVWY).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

PAML_ORDER = "ARNDCQEGHILKMFPSTWYV"
ALPHA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# LG lower-triangle exchangeabilities (190 values) in PAML order.
_LG_S_LOWER = np.array([
    0.425093,
    0.276818, 0.751878,
    0.395144, 0.123954, 5.076149,
    2.489084, 0.534551, 0.528768, 0.062556,
    0.969894, 2.807908, 1.038545, 0.363970, 0.746078,
    1.038545, 0.363970, 0.746078, 5.243870, 0.084329, 5.115644,
    2.066040, 0.390894, 1.437645, 0.554236, 0.075382, 0.594093, 2.547870,
    0.358858, 2.137150, 3.038533, 0.312261, 0.006334, 1.506500, 0.528768, 0.306475,
    0.149830, 0.109261, 0.528768, 0.042610, 0.308635, 0.126991, 0.001800, 0.021543, 0.236199,
    0.395144, 0.528768, 0.100872, 0.006613, 0.320627, 0.350230, 0.058654, 0.018625, 0.468199, 3.088510,
    0.906265, 5.351420, 3.148580, 0.569265, 0.072854, 2.006569, 1.137630, 0.336355, 0.122346, 0.068674, 0.277724,
    0.893496, 0.691268, 0.245034, 0.006613, 0.691268, 0.811614, 0.095382, 0.066236, 0.304803, 3.277830, 4.257460, 0.285078,
    0.210494, 0.145482, 0.065314, 0.003218, 0.897871, 0.089525, 0.006613, 0.062556, 0.645560, 0.829175, 2.106910, 0.046730, 1.190630,
    1.438550, 0.368739, 0.164126, 0.410886, 0.393379, 0.666506, 0.367902, 0.233397, 0.483768, 0.050644, 0.312261, 0.205711, 0.050644, 0.035454,
    4.509480, 0.887753, 3.681060, 1.169970, 2.137150, 1.003450, 0.544060, 1.595430, 0.611973, 0.131528, 0.267828, 0.665585, 0.247847, 0.364434, 1.341820,
    2.000540, 0.530324, 2.000540, 0.679371, 0.739772, 0.402941, 0.252167, 0.336355, 0.428437, 1.059470, 0.196258, 0.604070, 0.515706, 0.090855, 0.564432, 4.378020,
    0.113855, 0.869489, 0.049906, 0.006613, 0.911370, 0.247103, 0.006613, 0.167042, 0.540027, 0.157001, 0.868166, 0.035454, 0.506734, 1.289460, 0.049906, 0.306905, 0.152335,
    0.195510, 0.124630, 0.324525, 0.109261, 0.649361, 0.244157, 0.028906, 0.044265, 4.813505, 0.208836, 0.332517, 0.076701, 0.320627, 6.312580, 0.148483, 0.456190, 0.171995, 2.370130,
    2.386260, 0.186979, 0.062556, 0.068674, 1.173890, 0.117132, 0.174845, 0.188182, 0.222455, 7.821300, 1.129560, 0.137505, 2.020060, 0.569265, 0.249060, 0.582457, 2.370130, 0.268491, 0.257336,
])

_LG_PI = np.array([
    0.079066, 0.055941, 0.041977, 0.053052, 0.012937,
    0.040767, 0.071586, 0.057337, 0.022355, 0.062157,
    0.099081, 0.064600, 0.022951, 0.042302, 0.044040,
    0.061197, 0.053287, 0.012066, 0.034155, 0.069147,
])


def _lower_tri_to_matrix(s_values: np.ndarray, n: int = 20) -> np.ndarray:
    expected = n * (n - 1) // 2
    assert len(s_values) == expected
    S = np.zeros((n, n))
    idx = 0
    for i in range(1, n):
        for j in range(i):
            S[i, j] = s_values[idx]
            S[j, i] = s_values[idx]
            idx += 1
    return S


def _paml_to_alpha_perm() -> np.ndarray:
    return np.array([PAML_ORDER.index(aa) for aa in ALPHA_ORDER])


def get_lg08():
    """Return (S_alpha, pi_alpha) in alphabetical AA order (ACDEFGHIKLMNPQRSTVWY).

    S_alpha is the (20, 20) symmetric exchangeability matrix.
    pi_alpha sums to 1.

    The single-site rate matrix is built by `build_single_site_Q`.
    """
    S_paml = _lower_tri_to_matrix(_LG_S_LOWER)
    pi_paml = _LG_PI / _LG_PI.sum()
    perm = _paml_to_alpha_perm()
    S_alpha = S_paml[perm][:, perm]
    pi_alpha = pi_paml[perm]
    return S_alpha, pi_alpha


def build_single_site_Q(S: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Build the GTR rate matrix Q[i,j] = S[i,j] * pi[j] for i != j,
    diagonals set so rows sum to 0, then normalized so the mean rate
    (-sum_i pi_i Q_ii) equals 1.
    """
    Q = S * pi[None, :]
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    mean_rate = -float(np.sum(pi * np.diag(Q)))
    return Q / mean_rate


# Cached LG08 in alphabetical order, plus the singleton GTR rate matrix.
S_LG08, PI_LG08 = get_lg08()
Q_LG08 = build_single_site_Q(S_LG08, PI_LG08)

# F81 exchangeability: the symmetric matrix S' such that Q_LG08[x, y] =
# S'[x, y] * pi[y] off-diagonal, with mean rate 1 at PI_LG08. This is
# the form used in main.tex \S2 ('Q^s = eta_s * S_{xx'} * pi(x')').
# It differs from the published S_LG08 by the LG08 mean-rate normalizer
# (S_LG08_F81 = S_LG08 / mean_rate_at_pi_LG08 ~ S_LG08 / 0.894).
def _S_F81_from_Q(Q: np.ndarray, pi: np.ndarray) -> np.ndarray:
    Sf = Q / pi[None, :]
    np.fill_diagonal(Sf, 0.0)
    Sf = 0.5 * (Sf + Sf.T)   # numerically symmetrize
    return Sf

S_LG08_F81 = _S_F81_from_Q(Q_LG08, PI_LG08)

# JAX-typed exports for downstream JIT use.
S_LG08_J = jnp.asarray(S_LG08)             # published paper coefficients
S_LG08_F81_J = jnp.asarray(S_LG08_F81)     # F81 form (rate-1-normalized)
PI_LG08_J = jnp.asarray(PI_LG08)
Q_LG08_J = jnp.asarray(Q_LG08)
Q_LG08_J = jnp.asarray(Q_LG08)
