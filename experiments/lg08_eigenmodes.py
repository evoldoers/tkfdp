"""LG08 eigenmodes as modes of information loss.

Build the reversible LG08 GTR rate matrix Q (rate-1 normalised), symmetrise
it for a reversible chain, eigendecompose, and interpret each of the 19
non-equilibrium right eigenvectors psi_a as a "mode of information loss".

The math (see analysis/lg08_eigenmode_interpretation.md for prose):

  Q_ij = S_ij * pi_j (i != j),  Q_ii = -sum_{j!=i} Q_ij,
  normalised so sum_i pi_i (-Q_ii) = 1 (expected substitutions/site = 1).

For a reversible chain, Qsym = diag(sqrt pi) Q diag(1/sqrt pi) is symmetric.
Its eigenpairs (lambda_a, phi_a) give the right eigenvectors of Q as
  psi_a = phi_a / sqrt(pi),
which are functions on the 20 amino acids that decay as exp(lambda_a t):

  P(t)_xy / pi_y = sum_a exp(lambda_a t) psi_a(x) psi_a(y) pi_y  (spectral form)
  E[psi_a(X_t) | X_0 = x] = exp(lambda_a t) psi_a(x).

lambda_0 = 0 is the equilibrium mode (phi_0 = sqrt(pi), psi_0 = const 1).
The other 19 eigenvalues are negative relaxation rates. As psi_a decays, the
contrast between its positive-loading residues and its negative-loading
residues is averaged away: that averaging IS the information-loss mode.

The psi_a are orthonormal under the pi-weighted inner product
  <psi_a, psi_b>_pi = sum_i pi_i psi_a(i) psi_b(i) = delta_ab,
so every mode carries unit stationary variance. What differs is HOW FAST it
decays. The chi-squared mutual information between X_0 and X_t at stationarity
is I_chi2(t) = sum_{a>=1} exp(2 lambda_a t); its initial decay rate is
sum_a 2|lambda_a|, so mode a carries a fraction |lambda_a| / sum_b|lambda_b|
of the instantaneous information-loss rate ("MI-loss share" below).

Only numpy/scipy; reproducible (deterministic sign convention).

Run: PYTHONPATH=src python experiments/lg08_eigenmodes.py
"""

from __future__ import annotations

import numpy as np

from tkfdp.lg08 import S_LG08, PI_LG08, Q_LG08

AA = "ACDEFGHIKLMNPQRSTVWY"  # alphabetical order used by lg08.py


# --- Standard amino-acid property scales, in alphabetical order ACDEFG... ----
# Values are published literature scales (see references in the .md).
# Kyte-Doolittle hydrophobicity (1982): + hydrophobic, - hydrophilic.
KD_HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3,
}
# Residue side-chain volume (Zamyatnin 1972, cubic angstrom).
VOLUME = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1,
    "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1,
    "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0,
    "W": 227.8, "Y": 193.6,
}
# Net side-chain charge at pH 7 (H given +0.5 for partial protonation).
CHARGE = {
    "A": 0.0, "C": 0.0, "D": -1.0, "E": -1.0, "F": 0.0, "G": 0.0,
    "H": 0.5, "I": 0.0, "K": 1.0, "L": 0.0, "M": 0.0, "N": 0.0,
    "P": 0.0, "Q": 0.0, "R": 1.0, "S": 0.0, "T": 0.0, "V": 0.0,
    "W": 0.0, "Y": 0.0,
}
# Grantham polarity (1974).
POLARITY = {
    "A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3, "F": 5.2, "G": 9.0,
    "H": 10.4, "I": 5.2, "K": 11.3, "L": 4.9, "M": 5.7, "N": 11.6,
    "P": 8.0, "Q": 10.5, "R": 10.5, "S": 9.2, "T": 8.6, "V": 5.9,
    "W": 5.4, "Y": 6.2,
}
# Aromaticity indicator (F, W, Y = ring; H excluded as a weak ring).
AROMATIC = {aa: (1.0 if aa in "FWY" else 0.0) for aa in AA}
# Absolute-charge / "ionizable" indicator: charged D E K R H.
ABS_CHARGE = {aa: abs(CHARGE[aa]) for aa in AA}
# Single-residue indicators for special residues.
GLY = {aa: (1.0 if aa == "G" else 0.0) for aa in AA}
PRO = {aa: (1.0 if aa == "P" else 0.0) for aa in AA}
CYS = {aa: (1.0 if aa == "C" else 0.0) for aa in AA}
TRP = {aa: (1.0 if aa == "W" else 0.0) for aa in AA}

PROPERTIES = {
    "hydrophobicity(KD)": KD_HYDRO,
    "volume": VOLUME,
    "charge(signed)": CHARGE,
    "polarity": POLARITY,
    "aromatic(FWY)": AROMATIC,
    "|charge|(ionizable)": ABS_CHARGE,
    "Gly-flex": GLY,
    "Pro-rigid": PRO,
    "Cys": CYS,
    "Trp": TRP,
}


def _vec(d):
    """A property dict -> 20-vector in alphabetical order."""
    return np.array([d[aa] for aa in AA], dtype=float)


def eigendecompose(S, pi):
    """Return (lambdas, psi) sorted by |lambda| descending, equilibrium last.

    lambdas[k] is the relaxation rate of mode k (<= 0); lambdas[-1] ~ 0.
    psi[:, k] is the right eigenvector psi_k (function on the 20 AAs),
    orthonormal under the pi-weighted inner product and sign-fixed so its
    largest-magnitude loading is positive.
    """
    Q = Q_LG08.copy()  # already rate-1 normalised in lg08.py
    sp = np.sqrt(pi)
    Qsym = (sp[:, None]) * Q / (sp[None, :])       # diag(sqrt pi) Q diag(1/sqrt pi)
    Qsym = 0.5 * (Qsym + Qsym.T)                    # numerical symmetrisation
    lam, phi = np.linalg.eigh(Qsym)                 # ascending eigenvalues
    psi = phi / sp[:, None]                         # right eigenvectors of Q
    # order by |lambda| descending (fastest-decaying first); equilibrium last
    order = np.argsort(-np.abs(lam))
    lam, psi = lam[order], psi[:, order]
    # deterministic sign: largest |loading| entry made positive
    for k in range(psi.shape[1]):
        j = int(np.argmax(np.abs(psi[:, k])))
        if psi[j, k] < 0:
            psi[:, k] *= -1.0
    return lam, psi


def top_loadings(psi_col, n=5):
    """Top-n positive and top-n negative loadings as (aa, value) lists."""
    idx = np.argsort(-psi_col)
    pos = [(AA[i], float(psi_col[i])) for i in idx if psi_col[i] > 0][:n]
    neg_idx = np.argsort(psi_col)
    neg = [(AA[i], float(psi_col[i])) for i in neg_idx if psi_col[i] < 0][:n]
    return pos, neg


def best_property(psi_col):
    """Pearson-correlate loadings against each scale; return sorted matches."""
    out = []
    for name, d in PROPERTIES.items():
        p = _vec(d)
        if np.std(p) == 0:
            continue
        r = float(np.corrcoef(psi_col, p)[0, 1])
        out.append((name, r))
    out.sort(key=lambda t: -abs(t[1]))
    return out


def fmt_loadings(pairs):
    return ", ".join(f"{aa}{v:+.2f}" for aa, v in pairs)


def main():
    lam, psi = eigendecompose(S_LG08, PI_LG08)
    # sanity: equilibrium mode is last, ~0, psi ~ constant
    eq = lam[-1]
    total_rate = float(np.sum(np.abs(lam[:-1])))

    # pi-orthonormality check
    G = psi.T @ (PI_LG08[:, None] * psi)
    orth_err = float(np.max(np.abs(G - np.eye(20))))

    print(f"# LG08 eigenmodes (rate-1 normalised; expected subs/site = 1)")
    print(f"# equilibrium eigenvalue lambda_0 = {eq:.3e} (should be ~0)")
    print(f"# pi-orthonormality max|G - I| = {orth_err:.2e}")
    print(f"# sum of |lambda_a| over 19 modes = {total_rate:.4f}")
    print(f"# constant-vector check: psi_equil range "
          f"[{psi[:, -1].min():.4f}, {psi[:, -1].max():.4f}]")
    print()

    header = ("idx", "lambda", "rate|l|", "timescale", "MIshare",
              "bestprop", "corr", "top+", "top-")
    print("\t".join(header))
    rows = []
    for k in range(19):  # the 19 non-equilibrium modes, fastest first
        l = lam[k]
        rate = abs(l)
        tscale = 1.0 / rate
        share = rate / total_rate
        pos, neg = top_loadings(psi[:, k], n=5)
        props = best_property(psi[:, k])
        bp, br = props[0]
        rows.append({
            "idx": k + 1, "lambda": l, "rate": rate, "timescale": tscale,
            "share": share, "bestprop": bp, "corr": br,
            "props_all": props,
            "pos": pos, "neg": neg,
        })
        print(f"{k+1}\t{l:+.4f}\t{rate:.4f}\t{tscale:.3f}\t{share:.4f}\t"
              f"{bp}\t{br:+.2f}\t[{fmt_loadings(pos)}]\t[{fmt_loadings(neg)}]")
    return rows


if __name__ == "__main__":
    main()
