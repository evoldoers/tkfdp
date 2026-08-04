#!/usr/bin/env python3
r"""Is the two-sided-Lumpable infeasibility of coupled stationaries a WASHOUT of the
pooled empirical direction, or a genuine n>=3 obstruction?

Feasibility (fixed marginal generator A reversible w.r.t. rho): a stationary pi with
marginal rho admits a reversible-exchangeable two-sided-lumpable generator with
marginal A iff b_marg(pi)_{ijk} = pi_ij A_ik lies in range(B), B the Klein-orbit
block-sum incidence.  b_marg is LINEAR in pi, so the coupling perturbations
delta (symmetric, zero row-sum: keep marginal rho) that stay feasible form a linear
subspace:

    feasible_coupling = { delta : L^T b_marg(delta) = 0 },  L = basis of left-null(B).

Part A computes dim(feasible_coupling) exactly for n=2..6 (random reversible A) and
compares to the full coupling dimension n(n-1)/2.  n=2 must give FULL (the binary
telegraph is a coupled two-sided-lumpable chain); the question is whether it drops
to 0 at n>=3 (obstruction) or stays positive (coherent couplings survive).

Part B sweeps COHERENT synthetic couplings at n=20 (empirical rho, A): diagonal
pi(a,b) ~ rho_a rho_b e^{theta 1[a=b]}, a salt-bridge charge pattern, and a rank-1
u u^T; reports range(B) residual vs MI(pi).

Run: PYTHONPATH=src python3 analysis/scripts/coherent_feasibility.py
"""
import sys
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
sys.path.insert(0, "analysis/scripts")
import fit_pair_models as fp                       # noqa: E402
import fit_lumpable_kernel as K                    # noqa: E402
from scipy.linalg import null_space                # noqa: E402
from scipy.sparse.linalg import lsqr               # noqa: E402
from lp_stationary_sweep import sym_sinkhorn, mutual_information  # noqa: E402


def reversible_A(rho, S):
    """Marginal generator A reversible w.r.t. rho from symmetric exchangeability S."""
    n = rho.shape[0]
    A = S * rho[None, :]
    np.fill_diagonal(A, 0.0)
    A[np.diag_indices(n)] = -A.sum(1)
    return A


def coupling_basis(n):
    """Basis of symmetric zero-row-sum matrices (marginal-preserving coupling
    perturbations): edge Laplacians (e_i-e_j)(e_i-e_j)^T, i<j.  dim n(n-1)/2."""
    basis = []
    for i in range(n):
        for j in range(i + 1, n):
            v = np.zeros(n); v[i] = 1.0; v[j] = -1.0
            basis.append(np.outer(v, v))
    return basis


def feasible_coupling_dim(n, seed=0):
    """Exact dim of the feasible-coupling subspace for a random reversible A."""
    rng = np.random.default_rng(seed)
    rho = rng.random(n) + 0.3; rho /= rho.sum()
    S = rng.random((n, n)) + 0.2; S = 0.5 * (S + S.T)
    A = reversible_A(rho, S)
    orbit_id, n_orbits, _ = K.build_orbits_n(n)
    B, rows = K.build_B_sparse(n, orbit_id, n_orbits)
    Bd = B.toarray()
    L = null_space(Bd.T, rcond=1e-9)               # left-null(B): (n_rows, (n-1)^2)
    ri = rows[:, 0]; rj = rows[:, 1]; rk = rows[:, 2]
    basis = coupling_basis(n)
    # constraint matrix C[:, c] = L^T b_marg(delta_c)
    C = np.zeros((L.shape[1], len(basis)))
    for c, delta in enumerate(basis):
        bm = delta[ri, rj] * A[ri, rk]             # b_marg(delta)
        C[:, c] = L.T @ bm
    d_c = len(basis)
    rank_C = int(np.linalg.matrix_rank(C, tol=1e-9)) if C.size else 0
    return dict(n=n, coupling_dof=d_c, leftnull_dim=L.shape[1],
                feasible_dim=d_c - rank_C, infeasible_rank=rank_C)


def residual_two_sided(setup, pi_vec, A, ri, rj, rk):
    b = pi_vec[ri * K.NA + rj] * A[ri, rk]
    sol = lsqr(setup["B_sparse"], b, atol=1e-12, btol=1e-12, iter_lim=2000)
    return float(np.linalg.norm(setup["B_sparse"] @ sol[0] - b)
                 / max(np.linalg.norm(b), 1e-300))


def main():
    print("# ===== Part A: exact feasible-coupling dimension (random reversible A) =====",
          flush=True)
    print(f"# {'n':>3} {'coupling_dof':>12} {'leftnull(B)':>12} "
          f"{'feasible_dim':>12}  interpretation", flush=True)
    for n in [2, 3, 4, 5, 6]:
        r = feasible_coupling_dim(n)
        if r["feasible_dim"] == r["coupling_dof"]:
            interp = "ALL couplings feasible"
        elif r["feasible_dim"] == 0:
            interp = "ONLY product feasible (obstruction)"
        else:
            interp = f"{r['feasible_dim']}/{r['coupling_dof']} coherent dirs feasible"
        print(f"# {n:>3} {r['coupling_dof']:>12} {r['leftnull_dim']:>12} "
              f"{r['feasible_dim']:>12}  {interp}", flush=True)

    print("\n# ===== Part B: coherent couplings at n=20 (empirical rho, A) =====",
          flush=True)
    corpus = "data/cherry_counts_trrosetta"
    npair, tau, _ = fp.load_parts(corpus, [0, 1, 2, 3])
    pi_emp = fp.empirical_pi(npair).reshape(K.NA, K.NA)
    rho = pi_emp.sum(1); rho /= rho.sum()
    A, rho, g = K.fit_marginal_gtr(K.marginal_counts_from_pair(npair), tau,
                                   rho=rho, iters=100)
    setup = K.pair_setup(A, rho)
    ri, rj, rk, ro = setup["rows"]
    prod = np.outer(rho, rho)
    # charge pattern (salt bridge: opposite charges attract)
    alph = "ACDEFGHIKLMNPQRSTVWY"
    q = np.array([1.0 if a in "KRH" else -1.0 if a in "DE" else 0.0 for a in alph])
    rng = np.random.default_rng(7); u = rng.normal(size=K.NA)
    directions = {
        "diagonal": np.eye(K.NA),
        "charge(saltbridge)": -np.outer(q, q),
        "rank1(uu^T)": np.outer(u, u),
    }
    print(f"# {'direction':>20} {'theta':>7} {'MI(nats)':>9} {'range(B) resid':>15}",
          flush=True)
    for name, M in directions.items():
        for theta in [0.0, 0.5, 1.0, 2.0]:
            raw = prod * np.exp(theta * M)
            pi_s = sym_sinkhorn(raw, rho)
            mi = mutual_information(pi_s, rho)
            res = residual_two_sided(setup, pi_s.reshape(-1), A, ri, rj, rk)
            print(f"# {name:>20} {theta:>7.2f} {mi:>9.4f} {res:>15.3e}", flush=True)

    print("\n# ---- verdict ----", flush=True)
    print("# See Part A: feasible_dim==coupling_dof only at n=2; the value at n>=3 "
          "decides washout (feasible_dim>0) vs obstruction (feasible_dim==0).",
          flush=True)


if __name__ == "__main__":
    main()
