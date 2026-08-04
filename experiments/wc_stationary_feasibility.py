#!/usr/bin/env python3
r"""Stationary feasibility of a Watson-Crick Potts coupling under a given (HKY85) marginal.

This is a MARGINAL-LEVEL feasibility question, separate from (and prior to) the dynamical
"product trap" of analysis/lumpable_product_trap.md: before asking whether a lumpable
CHAIN can reach a WC-coupled stationary, ask whether such an exchangeable stationary
EXISTS with the given composition at all.

For an exchangeable joint pi(i,j)=pi(j,i) with marginal u supported only on WC pairs,
each A is WC-paired with a T, so pi(A,T)=pi(T,A)=:p forces u_A=u_T=p (and u_G=u_C=q):
a hard WC-Potts exchangeable stationary exists IFF u_A=u_T and u_G=u_C (Chargaff's second
parity rule).  With mismatches allowed, the composition skew caps the WC-paired mass at
    max WC mass = 2[min(u_A,u_T) + min(u_G,u_C)]   ( = 1 iff Chargaff parity ),
so a skewed marginal forces a floor of mismatches = 1 - that.  Wobble (G-T) pairs give the
excess G/T an alternative partner and raise the ceiling.  numpy/scipy only.
"""
import numpy as np
from scipy.optimize import linprog

n = 4; A, C, G, T = 0, 1, 2, 3
pairs = [(i, j) for i in range(n) for j in range(n)]
idx = {p: k for k, p in enumerate(pairs)}; NP = len(pairs)


def max_paired_mass(u, allowed):
    """Max mass an exchangeable joint (marginals=u, mismatches allowed) puts on `allowed`."""
    allow = set(allowed) | set((j, i) for (i, j) in allowed)
    c = np.array([-1.0 if (i, j) in allow else 0.0 for (i, j) in pairs])
    Aeq, beq = [], []
    for i in range(n):                                    # marginal u_i = sum_j pi_ij
        row = np.zeros(NP)
        for j in range(n): row[idx[(i, j)]] = 1
        Aeq.append(row); beq.append(u[i])
    for i in range(n):                                    # symmetry pi_ij = pi_ji
        for j in range(i + 1, n):
            row = np.zeros(NP); row[idx[(i, j)]] = 1; row[idx[(j, i)]] = -1
            Aeq.append(row); beq.append(0.0)
    r = linprog(c, A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=[(0, None)] * NP, method="highs")
    return -r.fun if r.success else None


WC = [(A, T), (G, C)]; WOBBLE = [(A, T), (G, C), (G, T)]
margs = {"symmetric        ": np.array([.25, .25, .25, .25]),
         "mild-skew A=T,C=G": np.array([.22, .28, .28, .22]),
         "uneven           ": np.array([.10, .20, .30, .40])}

if __name__ == "__main__":
    print("Max WC-paired mass of an exchangeable joint with the given marginal (mismatches allowed):")
    print(f"  {'marginal (A,C,G,T)':<19} {'hard-WC':>9} {'+wobble G-T':>12}")
    for lbl, u in margs.items():
        print(f"  {lbl:<19} {max_paired_mass(u, WC):>9.3f} {max_paired_mass(u, WOBBLE):>12.3f}   u={u}")
    print("\nClosed form (WC): 2[min(u_A,u_T)+min(u_G,u_C)] = 1  iff  u_A=u_T and u_G=u_C (Chargaff parity):")
    for lbl, u in margs.items():
        print(f"    {lbl}: {2*(min(u[A],u[T])+min(u[G],u[C])):.3f}")
