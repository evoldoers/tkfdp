#!/usr/bin/env python3
r"""Does FEASIBILITY (not just likelihood) drive the two-sided Lumpable fit to a
PRODUCT stationary?

Hold the marginal generator (A, rho) fixed and sweep the stationary COUPLING:
    pi_s = symmetric-Sinkhorn_to_rho( (1-s)*(rho (x) rho) + s*pi_emp ),
s = 0 (product) .. 1 (empirical) .. >1 (extrapolated, exaggerated coupling), each
with EXACT single-site marginal rho and pi_ij = pi_ji.  For each pi_s rebuild the
two-sided lumpability RHS b_marg_ijk = pi_ij * A_ik (kernel diag(pi)^-1 too) and
solve the max-slack feasibility LP (signed t): t = max uniform rate margin of a
reversible-exchangeable generator strongly lumpable to A with stationary pi_s.

  t > 0 : a strictly-positive lumpable generator with (pi_s, A) exists.
  t = 0 : only a boundary generator (some rates pinned at 0).
  t < 0 : the positive lumpable cone is EMPTY -- no valid generator at all.

Structural anchor: the A(+)A independent lift is a valid lumpable generator ONLY at
the product stationary, so product uniquely has a guaranteed feasible point (t>=0);
this script measures how fast that advantage erodes as coupling (MI) rises.

Run: PYTHONPATH=src python3 analysis/scripts/lp_stationary_sweep.py
"""
import sys
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
import fit_pair_models as fp                      # noqa: E402
import fit_lumpable_kernel as K                   # noqa: E402

NA = K.NA


def sym_sinkhorn(M, rho, iters=500):
    """Symmetric Sinkhorn: nearest D M D (M symmetric >=0) with marginal rho."""
    M = np.maximum(M, 0.0)
    d = np.ones(NA)
    for _ in range(iters):
        r = d * (M @ d)                            # current marginal
        d = d * np.sqrt(rho / np.maximum(r, 1e-300))
    P = d[:, None] * M * d[None, :]
    P = 0.5 * (P + P.T)
    return P / P.sum()


def mutual_information(P, rho):
    """MI in nats of joint 20x20 P with marginals rho."""
    prod = np.outer(rho, rho)
    mask = P > 0
    return float((P[mask] * np.log(P[mask] / prod[mask])).sum())


def main():
    corpus = "data/cherry_counts_trrosetta"
    npair, tau, _ = fp.load_parts(corpus, [0, 1, 2, 3])
    # empirical pair stationary (symmetric, marginal = rho by construction)
    pi_emp_vec = fp.empirical_pi(npair)
    pi_emp = pi_emp_vec.reshape(NA, NA)
    rho = pi_emp.sum(1)
    rho = rho / rho.sum()
    # marginal GTR fit at THIS rho (so s=1 is exactly pi_emp, no marginal drift)
    A, rho, g = K.fit_marginal_gtr(K.marginal_counts_from_pair(npair), tau,
                                   rho=rho, iters=100)
    # structural setup (orbit_id, B_sparse, rows) -- pi/b_marg overwritten per s
    setup = K.pair_setup(A, rho)
    ri, rj, rk, ro = setup["rows"]
    prod = np.outer(rho, rho)

    print(f"# trRosetta: rho range [{rho.min():.3e},{rho.max():.3e}]; "
          f"empirical MI = {mutual_information(pi_emp, rho):.4f} nats", flush=True)
    print(f"# {'s':>5} {'MI(nats)':>9} {'t':>12} {'min_phi':>11} "
          f"{'n_pinned':>9} {'LP status':>10}", flush=True)

    rows_out = []
    for s in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        raw = (1.0 - s) * prod + s * pi_emp
        pi_s = sym_sinkhorn(raw, rho)
        mi = mutual_information(pi_s, rho)
        pi_vec = pi_s.reshape(-1)
        # rebuild the (pi_s)-dependent pieces of the setup
        setup["pi"] = pi_vec
        setup["b_marg"] = pi_vec[ri * NA + rj] * A[ri, rk]     # pi_ij * A_ik
        lp = K.lp_feasibility(setup, signed_t=True)
        t = lp["t"]
        if lp["phi"] is not None:
            phi = lp["phi"]; scale = np.abs(phi).max()
            min_phi = float(phi.min())
            n_pin = int((phi <= 1e-9 * scale).sum())
        else:
            min_phi = float("nan"); n_pin = -1
        status = lp["status"]
        print(f"# {s:>5.2f} {mi:>9.4f} {t:>12.4e} {min_phi:>11.3e} "
              f"{n_pin:>9d} {status:>10d}", flush=True)
        rows_out.append(dict(s=s, MI=mi, t=t, min_phi=min_phi, n_pinned=n_pin,
                             status=status))

    # verdict. "cone shut" = t<0 (nonneg solution needs a negative rate) OR
    # status 2 (b_marg leaves range(B): the lumpability equality has no solution).
    def shut(r):
        return (r["status"] == 2) or (np.isfinite(r["t"]) and r["t"] < 0)
    t0 = rows_out[0]["t"]
    finite_t = [r["t"] for r in rows_out if np.isfinite(r["t"])]
    shut_rows = [r for r in rows_out if shut(r)]
    print("\n# ---- interpretation ----", flush=True)
    print(f"# t(product, s=0) = {t0:.4e}", flush=True)
    if shut_rows and all((not np.isfinite(r["t"])) or r["t"] <= t0 + 1e-12
                         for r in rows_out):
        s0 = shut_rows[0]
        print(f"# t(s) is maximised at the product stationary and the positive "
              f"lumpable cone SHUTS by s={s0['s']} (MI={s0['MI']:.3f} nats).\n"
              f"# => FEASIBILITY-driven product preference: coupling squeezes the "
              f"cone closed, product uniquely retains a guaranteed feasible anchor.",
              flush=True)
    elif all(r["t"] > 0 for r in rows_out if np.isfinite(r["t"])) and not shut_rows:
        print("# t(s) stays positive through the whole sweep: coupled lumpable "
              "chains are feasible.\n# => LIKELIHOOD-driven product preference.",
              flush=True)
    else:
        print("# mixed (see table).", flush=True)
    emp = rows_out[4]
    tag = "SHUT" if shut(emp) else ("near-shut" if (np.isfinite(emp["t"]) and
          emp["t"] < 0.1 * t0) else "feasible")
    print(f"# empirical s=1: t={emp['t']:.4e}, MI={emp['MI']:.4f} nats  [{tag}]",
          flush=True)


if __name__ == "__main__":
    main()
