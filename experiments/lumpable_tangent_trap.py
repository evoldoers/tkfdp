import numpy as np, itertools
from scipy.linalg import expm
from scipy.optimize import minimize
np.set_printoptions(suppress=True, precision=4)

def setup(n, seed=0):
    rng=np.random.RandomState(seed)
    states=list(itertools.product(range(n),range(n))); N=len(states); sidx={s:k for k,s in enumerate(states)}
    u=np.ones(n)/n; pi0=np.array([u[i]*u[j] for (i,j) in states])
    r=np.abs(rng.randn(n,n))+0.6; r=(r+r.T)/2; np.fill_diagonal(r,0)   # single-site exchangeability
    edges=[(a,b) for a in range(N) for b in range(N) if a<b]
    def kind(a,b):
        sa,sb=states[a],states[b]
        if (sa[0]!=sb[0]) and (sa[1]!=sb[1]): return 'D'
        return 'S'
    # background single-flip conductance for edge (reversible flux form Q_xy=c*pi_y)
    c0=np.zeros(len(edges))
    for m,(a,b) in enumerate(edges):
        sa,sb=states[a],states[b]
        if kind(a,b)=='S':
            if sa[0]!=sb[0]: c0[m]=r[sa[0],sb[0]]/pi0[b]*pi0[b]   # ~ r(i,k); simplify: use r
            else: c0[m]=r[sa[1],sb[1]]
            if sa[0]!=sb[0]: c0[m]=r[sa[0],sb[0]]
    return dict(n=n,states=states,sidx=sidx,u=u,pi0=pi0,r=r,edges=edges,kind=kind,N=N)

def tangent_basis(S):
    # first-order reachable stationary coupling of the reversible two-sided-lumpable family at product
    n=S['n']; states=S['states']; sidx=S['sidx']; u=S['u']; pi0=S['pi0']; edges=S['edges']; N=S['N']; r=S['r']
    # S0 (product single-flip conductances) in flux form Q_xy=Sc_xy*pi_y
    S0=np.zeros((N,N))
    for (a,b) in edges:
        sa,sb=states[a],states[b]
        if (sa[0]!=sb[0])^(sa[1]!=sb[1]):
            val = r[sa[0],sb[0]] if sa[0]!=sb[0] else r[sa[1],sb[1]]
            S0[a,b]=S0[b,a]=val
    pairs=[(a,b) for a in range(N) for b in range(N) if a<b]; pidx={}
    for t,(a,b) in enumerate(pairs): pidx[(a,b)]=t; pidx[(b,a)]=t
    npair=len(pairs); nz=npair+N
    G=np.zeros((N*(N-1),nz)); od=[(a,b) for a in range(N) for b in range(N) if a!=b]
    for t,(a,b) in enumerate(od):
        pr=(a,b) if a<b else (b,a); G[t,pidx[pr]]+=pi0[b]; G[t,npair+b]+=S0[a,b]
    def blk(a,coord,val):
        f=np.zeros(N*(N-1))
        for t,(x,y) in enumerate(od):
            if x==a and states[y][coord]==val: f[t]=1
        return f
    rows=[]
    for i in range(n):
        for k in range(n):
            if k==i: continue
            js=[sidx[(i,j)] for j in range(n)]; base=blk(js[0],0,k)
            for jj in js[1:]: rows.append(blk(jj,0,k)-base)
    for j in range(n):
        for l in range(n):
            if l==j: continue
            ks=[sidx[(i,j)] for i in range(n)]; base=blk(ks[0],1,l)
            for kk in ks[1:]: rows.append(blk(kk,1,l)-base)
    B=np.array(rows)@G; e=np.zeros(nz); e[npair:]=1; B=np.vstack([B,e])
    _,s_,vt=np.linalg.svd(B); ker=vt[np.sum(s_>1e-9):].T
    coup=[]
    for c in range(ker.shape[1]):
        DP=ker[npair:,c].reshape(n,n); am=DP.sum(1); bm=DP.sum(0)
        coup.append((DP-np.outer(am,u)-np.outer(u,bm)).ravel())
    coup=np.array(coup); _,sv,cv=np.linalg.svd(coup); rk=int(np.sum(sv>1e-8))
    return cv[:rk]   # tangent (escapable) coupling directions, n^2 vectors

def coup_space_basis(n):
    B=[]
    for a in range(1,n):
        for b in range(1,n):
            M=np.zeros((n,n)); M[0,0]=1;M[a,b]=1;M[a,0]=-1;M[0,b]=-1; B.append(M.ravel())
    Bo=np.linalg.qr(np.array(B).T)[0].T; return Bo

def metropolis_data(S,E,s,t):
    n=S['n']; states=S['states']; sidx=S['sidx']; u=S['u']; r=S['r']; N=S['N']
    logpi=np.array([np.log(u[i]*u[j])+s*E[i,j] for (i,j) in states]); pi=np.exp(logpi); pi/=pi.sum()
    Q=np.zeros((N,N))
    for a,(i,j) in enumerate(states):
        for b,(k,l) in enumerate(states):
            if a==b: continue
            if l==j and k!=i: Q[a,b]=r[i,k]*np.sqrt(pi[b]/pi[a])   # single X-flip, Metropolis-sqrt
            elif k==i and l!=j: Q[a,b]=r[j,l]*np.sqrt(pi[b]/pi[a])
    np.fill_diagonal(Q,-Q.sum(1))
    J=pi[:,None]*expm(Q*t)
    return J,pi

def mi(piv,n):
    P=piv.reshape(n,n); px=P.sum(1); py=P.sum(0)
    return float(np.sum(P*np.log((P+1e-300)/(np.outer(px,py)+1e-300))))

def fit(S,J,t,inits,lam=2e3):
    n=S['n']; N=S['N']; edges=S['edges']; states=S['states']; sidx=S['sidx']
    ne=len(edges)
    def lump_res(Q):
        res=[]
        for i in range(n):
            for k in range(n):
                if k==i: continue
                v=[sum(Q[sidx[(i,j)],sidx[(k,l)]] for l in range(n)) for j in range(n)]
                res+=[v[j]-v[0] for j in range(1,n)]
        for j in range(n):
            for l in range(n):
                if l==j: continue
                v=[sum(Q[sidx[(i,j)],sidx[(i2,l)]] for i2 in range(n)) for i in range(n)]
                res+=[v[i]-v[0] for i in range(1,n)]
        return np.array(res)
    def unpack(z):
        pi=np.exp(z[:N]-z[:N].max()); pi/=pi.sum(); c=np.exp(np.clip(z[N:],-20,8)); return pi,c
    def obj(z):
        pi,c=unpack(z); Q=np.zeros((N,N))
        for m,(a,b) in enumerate(edges): Q[a,b]=c[m]*pi[b]; Q[b,a]=c[m]*pi[a]
        np.fill_diagonal(Q,-Q.sum(1)); P=expm(Q*t); M=np.clip(pi[:,None]*P,1e-300,None)
        return -np.sum(J*np.log(M))+lam*np.sum(lump_res(Q)**2)
    best=None
    for z0 in inits:
        rr=minimize(obj,z0,method='L-BFGS-B',options={'maxiter':600})
        if best is None or rr.fun<best.fun: best=rr
    pi,c=unpack(best.x); return mi(pi,n), best.fun

def run(n, s=1.4, t=0.3):
    S=setup(n)
    Tb=tangent_basis(S)                      # escapable directions (n-1)
    Cb=coup_space_basis(n)                    # full coupling space (n-1)^2
    # projector onto tangent; complement direction
    Pt=Tb.T@Tb
    # pick delta_in (a tangent dir) and delta_out (coupling orthogonal to tangent)
    din=Tb[0].reshape(n,n)
    v=Cb[np.random.RandomState(1).randint(Cb.shape[0])]; v=v-Pt@v; v/=np.linalg.norm(v); dout=v.reshape(n,n)
    print(f"  n={n}: tangent(escapable) dim = {Tb.shape[0]}, full coupling dim = {Cb.shape[0]}")
    z_prod=np.concatenate([np.log(S['pi0']), np.log(np.array([1e-3]*len(S['edges'])))])
    rng=np.random.RandomState(0)
    rand_inits=[z_prod+np.concatenate([rng.randn(S['N'])*0.4, rng.randn(len(S['edges']))*0.8]) for _ in range(5)]
    for name,E in [('IN-tangent',din),('OUT-of-tangent',dout)]:
        J,pitrue=metropolis_data(S,E,s,t); mit=mi(pitrue.ravel(),n)
        mi_loc,_=fit(S,J,t,[z_prod])                 # local: product init only
        mi_glob,_=fit(S,J,t,[z_prod]+rand_inits)     # global: multistart
        print(f"    {name:<15} true MI={mit:.4f} | Lumpable local(from product) MI={mi_loc:.4f} | global MI={mi_glob:.4f}")

for n in (3,4):
    run(n)
