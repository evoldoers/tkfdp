"""HR-practicality probe: bridge-statistic E-step on the 400-state coupled pair
CTMC, against the trRosetta observed-count tensor. Tests speed + f64 stability of
the 400x400 reversible eigendecomposition + divided-difference bridge sums.
The E-step is O(A^3 * T) via eigenmode contraction (no naive O(A^4))."""
import time, sys, numpy as np
sys.path.insert(0,"src")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from tkfdp.coupling.dynfield.phylo_elbo.cluster_hr_jax import reversible_eigh_jax, _I_kl_jax
from tkfdp.lg08 import S_LG08

A2 = 400
d = np.load("data/cherry_counts_trrosetta/counts.npz", allow_pickle=True)
npair = np.asarray(d["n_pair"], np.float64)          # (20,20,20,20,32) = (i,j,k,l,t)
tc = np.asarray(d["tau_centers"], np.float64)
C = npair.reshape(A2, A2, -1)                         # (400,400,T) ancestor(ij)->desc(kl)
T = C.shape[-1]
print(f"# count tensor: {C.shape}, total {C.sum():,.0f}", flush=True)

# empirical pair stationary + a reversible single-transition (CTBN) 400-state generator
pi2 = (npair.sum((2,3,4)) + npair.sum((0,1,4))).reshape(-1); pi2 = pi2/pi2.sum()  # (400,)
pi2 = np.clip(pi2, 1e-8, None); pi2 /= pi2.sum()
S = np.asarray(S_LG08, np.float64)
# Q reversible wrt pi2: single-component transitions (i,j)->(k,j) at S[i,k]*sqrt(pi_kj/pi_ij)
idx = np.arange(A2); I,J = idx//20, idx%20
Q = np.zeros((A2,A2))
sq = np.sqrt(pi2)
for a in range(A2):
    i,j = I[a],J[a]
    for k in range(20):
        if k!=i:
            b=k*20+j; Q[a,b]=S[i,k]*sq[b]/sq[a]
        if k!=j:
            b=i*20+k; Q[a,b]=S[j,k]*sq[b]/sq[a]
Q[idx,idx] = -Q.sum(1)
print(f"# built 400-state single-transition reversible Q (nnz off-diag {int((Q>0).sum())})", flush=True)

Qj=jnp.asarray(Q); pij=jnp.asarray(pi2)
@jax.jit
def estep(Q, pi, Ct, taus):
    lam,V,sqrt,inv = reversible_eigh_jax(Q, pi)        # 400x400 reversible eigh
    def one_t(Ctt, tau):
        e = jnp.exp(lam*tau)
        P = inv[:,None]*(V*e[None,:])@V.T*sqrt[None,:] # P(t) 400x400
        G = Ctt/jnp.maximum(P,1e-300)
        Gw = inv[:,None]*G*sqrt[None,:]
        Ghat = V.T@Gw@V
        Ikl = _I_kl_jax(lam, tau)                      # divided-difference J
        VMV = V@(Ghat*Ikl)@V.T
        U = Q*(sqrt[:,None]*VMV*inv[None,:])            # expected transition counts (400x400)
        U = U - jnp.diag(jnp.diagonal(U))
        W = jnp.diagonal(VMV)                           # expected dwell (400,)
        return U, W
    Us,Ws = jax.vmap(one_t)(jnp.transpose(Ct,(2,0,1)), taus)
    return Us.sum(0), Ws.sum(0)

Cj=jnp.asarray(C); tj=jnp.asarray(tc)
U,W = estep(Qj,pij,Cj,tj); U.block_until_ready()       # compile
for it in range(3):
    t=time.time(); U,W=estep(Qj,pij,Cj,tj); U.block_until_ready(); print(f"# E-step {it}: {time.time()-t:.3f}s",flush=True)
Un=np.asarray(U); Wn=np.asarray(W)
print(f"# stability: U finite={np.isfinite(Un).all()} nan={np.isnan(Un).any()} ; W finite={np.isfinite(Wn).all()}")
print(f"# sums: sum U(expected transitions)={Un.sum():,.0f}  sum W(dwell)={Wn.sum():,.1f}  (obs counts {C.sum():,.0f})")
print(f"# sanity: U>=0? {(Un>=-1e-6).all()}  W>=0? {(Wn>=-1e-6).all()}")
