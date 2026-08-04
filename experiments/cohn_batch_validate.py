"""Validate batch-native Cohn (cohn_batch.py) against: (a) the exact independent
log-likelihood at H=0, (b) the diffrax evolsnake Cohn on real coupling, (c) exact
log[e^{tW}] (same violation pattern).  CPU, small batches.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
import elbo_vs_expm as EV, fit_pair_models as FP, cohn_vs_elbo_pair as CV, ctbn
import plot_bounds_vs_cohn as P
import cohn_batch as CB
NA, NS = FP.NA, FP.NS

z = np.load("results/mixture_component_char/components_K8.npz", allow_pickle=True)
S_mix = np.asarray(z["S"], float); pis = np.asarray(z["pis"], float)
W, R, pip, pijv = P.build_model("mixture_K8_c7_mi0.290", pis, S_mix)
_, _, _, params, _ = CV.build_glauber(S_mix, pis[7].reshape(NA, NA))
Sc = np.asarray(params["S"]); hh = np.asarray(params["h"]); JJ = np.asarray(params["J"])
Sj, hj, Jj = jnp.asarray(Sc), jnp.asarray(hh), jnp.asarray(JJ)

def batch_F(S, h, J, pairs, T, n_grid, n_iter=20):
    xs = jnp.array([[a, b] for a, b, c, d in pairs]); ys = jnp.array([[c, d] for a, b, c, d in pairs])
    F, delta = CB.cohn_batch_elbo(S, h, J, xs, ys, float(T), 0.5, n_grid, n_iter)
    return np.asarray(F), np.asarray(delta)

rng = np.random.default_rng(0)
pairs = [tuple(int(v) for v in rng.integers(NA, size=4)) for _ in range(8)]
pairs = [p for p in pairs if (p[0], p[1]) != (p[2], p[3])][:6]

print("="*72)
print("(a) H=0: batch Cohn F must equal exact INDEPENDENT log-lik log[e^{tR}]_xy")
# independent generator: Glauber with J=0 -> R (per-site S exp(h)); its 2-site joint = R
T = 1.0
eR = EV.expm_rev(R, pip, T); logR = np.log(np.maximum(eR, EV.FLOOR))
Fb, _ = batch_F(Sj, hj, jnp.zeros((NA, NA)), pairs, T, n_grid=128)
for (a, b, c, d), fb in zip(pairs, Fb):
    ex = logR[a + b*NA, c + d*NA]
    print(f"  ({a:2d},{b:2d})->({c:2d},{d:2d})  batch_F(J=0)={fb:+.4f}  exact_indep={ex:+.4f}  diff={fb-ex:+.4f}")

print("="*72)
print("(b) real coupling: batch Cohn vs diffrax evolsnake Cohn (converge as n_grid grows)")
sm, ni, nm = CV.SEQ_MASK, CV.NBR_IDX, CV.NBR_MASK
exact = np.log(np.maximum(EV.expm_rev(W, pijv, T), EV.FLOOR))
for (a, b, c, d) in pairs[:4]:
    le, _ = ctbn.ctbn_variational_log_cond(jax.random.PRNGKey(0), jnp.array([a, b]), jnp.array([c, d]),
                                           sm, ni, nm, params, T, min_inc=1e-5, max_updates=256, rtol=1e-7, atol=1e-10)
    row = f"  ({a:2d},{b:2d})->({c:2d},{d:2d}) exact={exact[a+b*NA,c+d*NA]:+.4f} evolsnake={float(le):+.4f}"
    for ng in (64, 128, 256):
        fb, _ = batch_F(Sj, hj, Jj, [(a, b, c, d)], T, n_grid=ng)
        row += f"  batch(n={ng})={fb[0]:+.4f}"
    print(row)

print("="*72)
print("(c) batch Cohn vs exact over the 6 pairs, n_grid=256")
Fb, dl = batch_F(Sj, hj, Jj, pairs, T, n_grid=256)
for (a, b, c, d), fb, d_ in zip(pairs, Fb, dl):
    ex = exact[a + b*NA, c + d*NA]
    print(f"  ({a:2d},{b:2d})->({c:2d},{d:2d}) exact={ex:+.4f} batch={fb:+.4f} gap(exact-batch)={ex-fb:+.4f} converged_delta={d_:.1e}")
