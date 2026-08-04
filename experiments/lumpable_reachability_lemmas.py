import numpy as np, itertools
np.set_printoptions(suppress=True, precision=6)
# Verify the two load-bearing lemmas of the proof, on a RANDOM reversible perturbation.
n=4; rng=np.random.RandomState(3)
states=list(itertools.product(range(n),range(n))); N=n*n
u=np.abs(rng.randn(n))+0.5; u/=u.sum()
r=np.abs(rng.randn(n,n))+0.6; r=(r+r.T)/2; np.fill_diagonal(r,0)
W=r*u[None,:]; np.fill_diagonal(W,-W.sum(1))
su=np.sqrt(u); Ws=su[:,None]*W/su[None,:]; Ws=(Ws+Ws.T)/2
lam1,V=np.linalg.eigh(Ws); psi=V/su[:,None]; o=np.argsort(-lam1); lam1=lam1[o]; psi=psi[:,o]
# pair eigenbasis e_{ab}=psi_a (x) psi_b, eigenvalue lam_a+lam_b ; pi0=u(x)u
pi0=np.array([u[i]*u[j] for (i,j) in states])
def e(a,b): return np.array([psi[i,a]*psi[j,b] for (i,j) in states])
lam=np.array([lam1[a] for a in range(n)])
Q0=np.zeros((N,N))
for x,(i,j) in enumerate(states):
    for y,(k,l) in enumerate(states):
        if x==y: continue
        if l==j: Q0[x,y]=W[i,k]
        elif k==i: Q0[x,y]=W[j,l]
    Q0[x,x]=-Q0[x].sum()

# random reversible perturbation: Q=F/pi, F=F0+eps*Phi (sym), pi=pi0+eps*delta (zero-marginal)
F0=np.zeros((N,N))
for x in range(N):
    for y in range(N):
        if x!=y: F0[x,y]=pi0[x]*Q0[x,y]           # = pi0[y]*Q0[y,x], symmetric
Phi=rng.randn(N,N); Phi=(Phi+Phi.T)/2; np.fill_diagonal(Phi,0)
dpi=rng.randn(n,n); dpi-=dpi.mean(0,keepdims=True); dpi-=dpi.mean(1,keepdims=True); delta=dpi.ravel()
# M = dQ/deps at eps=0 :  Q_xy=(F0+eps Phi)_xy/(pi0+eps delta)_x
M=np.zeros((N,N))
for x in range(N):
    for y in range(N):
        if x!=y: M[x,y]=Phi[x,y]/pi0[x] - F0[x,y]*delta[x]/pi0[x]**2
    M[x,x]=-M[x].sum()

def ip(f,g): return np.sum(pi0*f*g)                # <,>_{pi0}
def Mel(cd,ef):                                    # M_{(cd),(ef)} = <e_cd, M e_ef>_{pi0}
    return ip(e(*cd), M@e(*ef))
def D(cd,ef): return np.sum(delta*e(*cd)*e(*ef))   # <e_cd,e_ef>_delta

# Lemma (R'):  M_{cd,ef} - M_{ef,cd} == [(lam_c+lam_d)-(lam_e+lam_f)] * D_{cd,ef}
err=0
for cd in itertools.product(range(n),range(n)):
    for ef in itertools.product(range(n),range(n)):
        lhs=Mel(cd,ef)-Mel(ef,cd)
        rhs=((lam[cd[0]]+lam[cd[1]])-(lam[ef[0]]+lam[ef[1]]))*D(cd,ef)
        err=max(err,abs(lhs-rhs))
print(f"Lemma (R') reversibility identity  max|LHS-RHS| = {err:.2e}   (should be ~0)")

# Sanity: D_{(a0),(0b)} == delta_ab (the (a,b) coupling coordinate)
err2=0
for a in range(1,n):
    for b in range(1,n):
        dab=np.sum(delta*e(a,b))
        err2=max(err2, abs(D((a,0),(0,b)) - dab))
print(f"Identity  D_(a0),(0b) = delta_ab      max err = {err2:.2e}   (should be ~0)")
