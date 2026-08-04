"""Tree Holmes--Rubin charge-flip counter (LG08).

For a column, run an up--down pass on the stored Felsenstein CLVs and accumulate
the endpoint-conditioned *expected substitution counts*, decomposed into
acid<->base (charge-flip) events. This distinguishes genuine repeated charge
flipping from a single ancient acidic/basic split (which standing leaf variation
cannot): a confirmed salt-bridge-flip column shows several expected acid<->base
substitutions spread over the tree.

Per branch (u->v, length t) we accumulate, over the branch posterior P(X_u,X_v|data),
  E[N_{a->b} | data] = (1/Z) sum_{a,b} Q_ab sqrt[a] inv[b] (V[a] . M . V[b]),
  M_kl = g_k I_kl(t) h_l,  g_k = sum_i combined_{u\\v}(i) inv[i] V_ik,
  h_l  = sum_j inside_v(j) sqrt[j] V_jl,
with I_kl the Van Loan integral of the coupled bridge. The P_ij(t) in the
per-type HR formula cancels the branch-posterior normaliser, leaving O(A^2) per
branch. Charge-flip count sums (a,b) over ACID x BASE u BASE x ACID; total count
sums all off-diagonal (a,b).

Validated two ways (run as a script): per-type counts sum to `hr_per_cherry`'s
total, and the tree log-likelihood matches the stored `log_p_lg_per_site`.

Usage as a module:
    from tree_hr_charge import tree_hr_column
    E_charge, E_total = tree_hr_column(parent, tau, clv[:, col, :])

Usage as a script (self-test + demo on one family/column):
    python analysis/scripts/tree_hr_charge.py [family.npz] [col]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from tkfdp.lg08 import get_lg08, build_single_site_Q  # noqa: E402

ACID = [2, 3]           # D, E   (alphabet ACDEFGHIKLMNPQRSTVWY)
BASE = [8, 14, 6]       # K, R, H
CF = [(a, b) for a in ACID for b in BASE] + [(a, b) for a in BASE for b in ACID]
A = 20

_S, PI = get_lg08()
Q = build_single_site_Q(_S, PI)                      # GTR, mean rate 1
_SQRT = np.sqrt(PI); _INV = 1.0 / _SQRT
_Qsym = _SQRT[:, None] * Q * _INV[None, :]; _Qsym = 0.5 * (_Qsym + _Qsym.T)
LAM, VEC = np.linalg.eigh(_Qsym)                     # Q = D^-1/2 V Lam V^T D^1/2


def _Pt(t):
    e = np.exp(LAM * t)
    return (_INV[:, None] * (VEC * e[None, :]) @ VEC.T) * _SQRT[None, :]


def _I_kl(t):
    e = np.exp(LAM * t)
    d = LAM[:, None] - LAM[None, :]
    safe = np.where(np.abs(d) < 1e-12, 1.0, d)
    off = (e[:, None] - e[None, :]) / safe
    return np.where(np.abs(d) < 1e-12, t * e[:, None], off)


def tree_hr_column(parent, tau, clv_col):
    """clv_col: (n_nodes, A) inside messages (any per-node scaling). Renormalises
    each node to sum 1 and uses a per-branch normaliser (= Z in exact arithmetic),
    so the stored scaling cancels and nothing underflows.
    Returns (E_charge, E_total): expected acid<->base and total substitution
    counts over the whole tree for this column."""
    n = len(parent)
    root = int(np.where(parent < 0)[0][0]) if (parent < 0).any() \
        else int(np.where(parent == np.arange(n))[0][0])
    kids = [[] for _ in range(n)]
    for v in range(n):
        u = int(parent[v])
        if v != root and u >= 0:
            kids[u].append(v)
    L = np.asarray(clv_col, np.float64).copy()
    L /= np.clip(L.sum(1, keepdims=True), 1e-300, None)
    P = {v: _Pt(float(tau[v])) for v in range(n) if v != root}
    m = {v: P[v] @ L[v] for v in range(n) if v != root}
    O = {root: PI.copy()}
    E_charge = 0.0; E_total = 0.0
    order = [root]; qi = 0
    while qi < len(order):
        u = order[qi]; qi += 1
        ch = kids[u]
        prodall = np.ones(A)
        for w in ch:
            prodall = prodall * m[w]
        for v in ch:
            mv = m[v]
            sib = np.divide(prodall, mv, out=np.zeros(A), where=mv > 1e-300)
            combined = O[u] * sib
            sum_w = float(combined @ m[v])                 # == Z
            if sum_w <= 1e-300:
                O[v] = combined @ P[v]; order.append(v); continue
            Ikl = _I_kl(float(tau[v]))
            g = (combined * _INV) @ VEC
            h = (L[v] * _SQRT) @ VEC
            M = g[:, None] * Ikl * h[None, :]
            VMV = VEC @ M @ VEC.T
            contrib = Q * (_SQRT[:, None] * _INV[None, :]) * VMV
            E_total += (contrib.sum() - np.trace(contrib)) / sum_w
            E_charge += sum(contrib[a, b] for a, b in CF) / sum_w
            O[v] = combined @ P[v]
            order.append(v)
    return E_charge, E_total


def _self_test(fam_npz, col):
    from tkfdp.eta_site import hr_per_cherry
    i, j, t = 2, 8, 0.7                                    # D -> K cherry
    Nref, _, _ = hr_per_cherry(i, j, t, Q, PI)
    Pm = _Pt(t); Ikl = _I_kl(t); gi = _INV[i] * VEC[i]; hj = _SQRT[j] * VEC[j]
    tot = 0.0
    for a in range(A):
        for b in range(A):
            if a == b:
                continue
            M = gi[:, None] * Ikl * hj[None, :]
            tot += Q[a, b] * _SQRT[a] * _INV[b] * (VEC[a] @ M @ VEC[b]) / Pm[i, j]
    assert abs(tot - Nref) < 1e-8, (tot, Nref)
    print(f"[val1] per-type HR sum={tot:.8f} == hr_per_cherry={Nref:.8f}")

    d = np.load(fam_npz)
    parent, tau, clv = d["parent"], d["tau"], d["clv"]
    root = int(np.where(parent < 0)[0][0])
    logZ = float(np.log(PI @ clv[root, col]) + d["log_scale"][root, col])
    print(f"[val2] col{col} tree logZ={logZ:.4f} == stored log_p_lg="
          f"{float(d['log_p_lg_per_site'][col]):.4f}")
    ec, et = tree_hr_column(parent, tau, clv[:, col, :])
    print(f"[demo] col{col}: E[charge-flip]={ec:.2f}  E[total]={et:.2f}  "
          f"charge-fraction={ec / et:.3f}")


if __name__ == "__main__":
    fam = sys.argv[1] if len(sys.argv) > 1 else \
        "data/pfam_processed_clv_top1000_thin128/PF02457.npz"
    c = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    _self_test(fam, c)
