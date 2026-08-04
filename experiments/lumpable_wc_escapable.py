import numpy as np, itertools
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lumpable_tangent_rank import _modes
np.set_printoptions(suppress=True, precision=4)

# ---------- n=2 : there is only ONE relaxation mode, so no non-resonant pair can exist ----------
print("== n=2 (any marginal) ==")
u2=np.array([0.4,0.6]); r2=np.array([[0,1.0],[1.0,0]])   # single-site: one flip rate
W2,lam2,psi2=_modes(r2,u2)
print(f"  single-site eig(W)={np.round(lam2,3)}  -> exactly one nonzero mode (lam_1).")
print(f"  coupling space dim (n-1)^2 = 1 ; only mode is (1,1), resonant with itself => reachable = full.")
print(f"  ALGEBRA: pi(i,j)=u_i u_j (1 + kappa*psi_1(i)psi_1(j)); the single knob kappa is always")
print(f"           first-order accessible.  No trap is possible at n=2 (needs n>=3 for distinct rates).\n")

# ---------- n=4 : Watson-Crick coupling, escapable fraction under each marginal ----------
n=4; A,C,G,T=0,1,2,3
def dna_r(kappa):
    r=np.ones((n,n))
    for (a,b) in [(A,G),(C,T)]: r[a,b]=r[b,a]=kappa
    np.fill_diagonal(r,0); return r
# Watson-Crick coupling (favour A-T and G-C), as a zero-marginal coupling matrix
dWC=np.zeros((n,n))
for (a,b) in [(A,T),(T,A),(G,C),(C,G)]: dWC[a,b]+=1.0
dWC-=dWC.mean(0,keepdims=True); dWC-=dWC.mean(1,keepdims=True)

def escapable_fraction(r,u):
    W,lam,psi=_modes(r,u)
    # mode coefficients c_ab = sum_ij dWC[i,j] psi_a(i) psi_b(j),  a,b>=1
    C=np.zeros((n,n))
    for a in range(n):
        for b in range(n):
            C[a,b]=np.sum(dWC*np.outer(psi[:,a],psi[:,b]))
    total=0.0; reson=0.0
    for a in range(1,n):
        for b in range(1,n):
            total+=C[a,b]**2
            if abs(lam[a]-lam[b])<1e-6: reson+=C[a,b]**2
    return lam, reson/total if total>1e-12 else 0.0

cases=[("JC69            ", dna_r(1.0),          np.ones(n)/n),
       ("K80 (kappa=4)   ", dna_r(4.0),          np.ones(n)/n),
       ("HKY85 skewed    ", dna_r(4.0),          np.array([0.15,0.35,0.35,0.15])),
       ("HKY85 (uneven)  ", dna_r(4.0),          np.array([0.1,0.2,0.3,0.4])),
       ("F81 / renewal   ", np.ones((n,n))*(1-np.eye(n)), np.array([0.15,0.35,0.35,0.15]))]
print("== n=4 : fraction of the Watson-Crick coupling that is FIRST-ORDER ESCAPABLE from the product ==")
print(f"  {'marginal':<16} {'eig(W)':<26} {'escapable WC fraction'}")
for name,r,u in cases:
    lam,frac=escapable_fraction(r,u)
    print(f"  {name:<16} {str(np.round(lam,2)):<26} {frac:6.1%}")
