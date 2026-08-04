"""Diagnose the Cohn bound violations: harness validation + exact cross-check +
ODE-tolerance sweep. Cohn's F is provably an ELBO in exact arithmetic, so a
violation is either (a) my harness/exact, or (b) numerical F-integral error."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
from scipy.linalg import expm as scipy_expm
sys.path.insert(0, "src"); sys.path.insert(0, "experiments"); sys.path.insert(0, "src/tkfdp/cohn_ctbn")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import fit_pair_models as FP, elbo_vs_expm as EV, ctbn
NA, NS = FP.NA, FP.NS

def cohn(a, b, c, d, params, C, T, rtol, atol, mu=256):
    sm, ni, nm, *_ = ctbn.get_Markov_blankets(C)
    le, _ = ctbn.ctbn_variational_log_cond(jax.random.PRNGKey(0), jnp.array([a, b]),
        jnp.array([c, d]), sm, ni, nm, params, float(T), min_inc=1e-5,
        max_updates=mu, rtol=rtol, atol=atol)
    return float(le)

# ---------- (1) harness validation: evolsnake's own N=2 ising2, bound must hold ----------
print("=" * 70); print("(1) N=2 ising2 (evolsnake test config): bound must hold")
Js = np.log(10)/2; Jd = 0.0
C2 = jnp.array([[0., 1.], [1., 0.]])
p_is = {"S": jnp.array([[0., 1.], [1., 0.]]), "J": jnp.array([[Js, Jd], [Jd, Js]]), "h": jnp.zeros(2)}
sm, ni, nm, *_ = ctbn.get_Markov_blankets(C2)
Wis = np.asarray(ctbn.q_joint(ni, nm, ctbn.normalise_ctbn_params(p_is)))
for (x, y) in [(0*2+1, 1*2+0), (0, 0), (1*2+0, 0*2+1)]:
    a, b, c, d = x % 2, x // 2, y % 2, y // 2
    e_sci = float(np.log(scipy_expm(1.0 * Wis)[x, y]))
    le = cohn(a, b, c, d, p_is, C2, 1.0, 1e-3, 1e-6)
    print(f"  ({a},{b})->({c},{d})  exact={e_sci:+.5f}  cohn={le:+.5f}  gap={e_sci-le:+.5f}"
          f"  {'OK' if e_sci-le>=-1e-4 else 'VIOLATION'}")

# ---------- (2) N=20 exact cross-check + tolerance sweep on violating pairs ----------
print("=" * 70); print("(2) N=20 Glauber: exact cross-check + ODE-tolerance sweep")
zm = np.load("results/mixture_component_char/components_K8.npz", allow_pickle=True)
S = np.asarray(zm["S"], float); pis = np.asarray(zm["pis"], float); mi = np.asarray(zm["mi_pi"], float)
c = int(np.argsort(mi)[len(mi)//2])          # the MI=0.164 model that violated
pij = pis[c].reshape(NA, NA); pij /= pij.sum()
m1 = pij.sum(1); h = np.log(np.maximum(m1, 1e-300))
J = 0.5*(np.log(np.maximum(pij, 1e-300)) - h[:, None] - h[None, :]); Sc = S.copy(); np.fill_diagonal(Sc, 0.)
params = {"S": jnp.asarray(Sc), "J": jnp.asarray(J), "h": jnp.asarray(h)}
piv = pij.reshape(NS)
W = np.asarray(ctbn.q_joint(ni, nm, ctbn.normalise_ctbn_params(params)))
T = 0.5
eig = EV.expm_rev(W, piv, T); sci = scipy_expm(T*np.array(W, copy=True))
print(f"  exact cross-check max|expm_rev - scipy| = {np.max(np.abs(eig-sci)):.2e}  "
      f"(pij symmetric? {np.max(np.abs(pij-pij.T)):.1e})")
exact = np.log(np.maximum(eig, EV.FLOOR))

rng = np.random.default_rng(0)
# a few off-diagonal pairs (single + double substitutions)
pairs = []
for _ in range(3):
    a, b = int(rng.integers(NA)), int(rng.integers(NA)); x = a + b*NA
    w = eig[x].reshape(NA, NA, order="F")[:, b].copy(); w[a] = 0
    cc = int(np.argmax(w)); pairs.append((a, b, cc, b, "single"))
for _ in range(3):
    a, b = int(rng.integers(NA)), int(rng.integers(NA)); x = a + b*NA
    M = eig[x].reshape(NA, NA, order="F").copy(); M[a, :] = 0; M[:, b] = 0
    k = int(np.argmax(M.reshape(-1))); cc, dd = k//NA, k % NA; pairs.append((a, b, cc, dd, "double"))

print(f"  {'pair':>18} {'exact':>9} | tol 1e-3/1e-6      1e-5/1e-8      1e-7/1e-10")
for (a, b, cc, dd, reg) in pairs:
    ex = exact[a+b*NA, cc+dd*NA]
    outs = []
    for (rt, at) in [(1e-3, 1e-6), (1e-5, 1e-8), (1e-7, 1e-10)]:
        try:
            le = cohn(a, b, cc, dd, params, C2, T, rt, at)
            outs.append(f"{le:+.4f}(g{ex-le:+.4f})")
        except Exception as e:
            outs.append(f"FAIL:{type(e).__name__}")
    print(f"  {reg:>6} ({a:2d},{b:2d})->({cc:2d},{dd:2d}) {ex:>9.4f} | " + "  ".join(outs))
