"""Midpoint-node variational ELBO for the coupled 400-state (two-site Potts) pair.

A variational lower bound on log[e^{tW}]_xy that places a mean-field variational
distribution q(z) = q_i(z_i) q_j(z_j) on the *midpoint state* z of the branch.

Derivation (single midpoint; K intermediate nodes below). By Chapman-Kolmogorov
with the split at t/2,

    [e^{tW}]_xy = sum_z [e^{(t/2)W}]_xz [e^{(t/2)W}]_zy .

Introduce q(z) and apply Jensen (P_h := e^{(t/2)W}):

    log[e^{tW}]_xy = log sum_z q(z) [P_h(x,z) P_h(z,y) / q(z)]
                  >= sum_z q(z) [ log P_h(x,z) + log P_h(z,y) ] + H(q)  =: L(x,y;q).

L is tight (= exact) at q*(z) ∝ P_h(x,z)P_h(z,y). The tractable family restricts
q to the mean field over the two sites, q(z) = q_i(z_i) q_j(z_j); then the only
approximation is the lost midpoint z_i-z_j correlation. The half-branch transition
P_h is the *coupled* e^{(t/2)W}, obtained for free from the reversible eigen-
decomposition of W, so the bound sees the full coupling on each half and is NOT
slack (contrast the paper's product-family midpoint mesh nu, which is).

This generalises to K interior nodes (split into K+1 segments with exact coupled
edges e^{(t/(K+1))W}, a mean-field q at each interior node); K=0 is exact, K=1 is
the single midpoint implemented here. CAVI (coordinate ascent) is monotone and the
result is a strict lower bound.

Everything is vectorised over the whole 400x400 matrix of endpoint pairs and the
mean-field optimisation is a JIT'd JAX fori_loop (see _cavi_jax for the algorithm:
coordinate-ascent variational inference / CAVI -- alternate closed-form softmax
updates of q_i, q_j; converges in ~3-4 sweeps; no matrix is diagonalised inside
the loop). Whole-matrix cost ~50-100 ms.

Run (GPU 1, staying off GPU 0):
     CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
       PYTHONPATH=src:experiments python3 experiments/midpoint_elbo.py   # self-checks
   (drop CUDA_VISIBLE_DEVICES / set JAX_PLATFORMS=cpu to run on CPU.)
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import elbo_vs_expm as EV        # expm_rev, closed_form_L, FLOOR, load_models
import fit_pair_models as FP

NA, NS = FP.NA, FP.NS
FLOOR = EV.FLOOR


def _logexpm_half(W, pi_stat, t, k_seg):
    """log e^{(t/k_seg)W} as an (NS,NS) array via reversible eig (machine precision)."""
    P = EV.expm_rev(W, pi_stat, t / k_seg)
    return np.log(np.maximum(P, FLOOR))


def _reshape_from(logP):
    """Lfrom[x, zi, zj] = logP[x, zi + zj*NA]  (from x to midpoint pair (zi,zj))."""
    return logP.reshape(NS, NA, NA).transpose(0, 2, 1)   # [x, zj, zi] -> [x, zi, zj]


def _reshape_to(logP):
    """Lto[zi, zj, y] = logP[zi + zj*NA, y]  (from midpoint pair to y)."""
    return logP.reshape(NA, NA, NS).transpose(1, 0, 2)   # [zj, zi, y] -> [zi, zj, y]


@partial(jax.jit, static_argnums=(2,))
def _cavi_jax(Lfrom, Lto, iters):
    """JIT'd mean-field coordinate ascent (CAVI), vectorised over ALL (x,y) pairs.

    Given the two half-branch log-tables Lfrom[x,zi,zj], Lto[zi,zj,y] (both already
    computed once, outside the loop -- NO matrix is diagonalised inside), the ELBO
    over q(z)=q_i(zi)q_j(zj) is maximised by the closed-form coordinate updates

        q_i(zi) proportional to exp( sum_zj q_j(zj) [Lfrom[x,zi,zj] + Lto[zi,zj,y]] )
        q_j(zj) proportional to exp( sum_zi q_i(zi) [Lfrom[x,zi,zj] + Lto[zi,zj,y]] )

    i.e. a softmax of a linear function of the *other* 20-vector. Two factors, so
    it is a 2-block coordinate ascent; it converges in ~3-4 sweeps. Everything is a
    batched einsum over the (NS,NS,NA) arrays; XLA fuses the whole fori_loop."""
    def body(_, carry):
        qi, qj = carry
        li = jnp.einsum("xyj,xij->xyi", qj, Lfrom) + jnp.einsum("xyj,ijy->xyi", qj, Lto)
        qi = jax.nn.softmax(li, axis=-1)
        lj = jnp.einsum("xyi,xij->xyj", qi, Lfrom) + jnp.einsum("xyi,ijy->xyj", qi, Lto)
        qj = jax.nn.softmax(lj, axis=-1)
        return (qi, qj)
    q0 = jnp.full((NS, NS, NA), 1.0 / NA)
    qi, qj = jax.lax.fori_loop(0, iters, body, (q0, q0))
    E = (jnp.einsum("xyi,xyj,xij->xy", qi, qj, Lfrom)
         + jnp.einsum("xyi,xyj,ijy->xy", qi, qj, Lto))
    H = -(jnp.sum(qi * jnp.log(jnp.clip(qi, 1e-30, None)), -1)
          + jnp.sum(qj * jnp.log(jnp.clip(qj, 1e-30, None)), -1))
    return E + H


def _midpoint_cavi(Lfrom, Lto, iters=6, tol=None, fp32=False):
    """Mean-field CAVI wrapper: run the JIT'd JAX kernel and return an (NS,NS) numpy
    array. `iters` is a fixed sweep count; CAVI converges in ~3-4, and 6 gives
    <2e-7 error even at t=3 (the worst case measured). `tol` is accepted for API
    compatibility and ignored (fixed iters keeps the loop JIT/vmap-friendly).

    fp32=True runs the CAVI in single precision -- ~12x faster on this GPU (the fp64
    units are ~1/32 of fp32) but up to ~4e-3 looser at long branches; use it for
    training / ranking, keep fp64 (default) for tight bound comparisons."""
    dt = jnp.float32 if fp32 else jnp.float64
    return np.asarray(_cavi_jax(jnp.asarray(Lfrom, dtype=dt), jnp.asarray(Lto, dtype=dt), int(iters)))


def midpoint_elbo_matrix(W, pi_stat, t, iters=6, tol=None, return_full_joint=False):
    """EXACT-EDGE variant: mean-field midpoint with EXACT coupled half-transitions
    e^{(t/2)W}. This needs the eig of the full coupled W, so it does NOT scale better
    than exact -- it is a diagnostic that isolates the midpoint mean-field's cost
    (and the full-joint check that the edges are correct). For the honest, scalable
    bound use midpoint_elbo_path_edges (product-eig only)."""
    logPh = _logexpm_half(W, pi_stat, t, 2)          # log e^{(t/2)W}  (COUPLED)
    Lfrom = _reshape_from(logPh)
    Lto = _reshape_to(logPh)
    L = _midpoint_cavi(Lfrom, Lto, iters, tol)
    if return_full_joint:
        A = Lfrom[:, None, :, :] + Lto.transpose(2, 0, 1)[None, :, :, :]  # (x,y,zi,zj)
        m = A.reshape(NS, NS, -1).max(-1)
        full = np.log(np.exp(A.reshape(NS, NS, -1) - m[:, :, None]).sum(-1)) + m
        return L, full
    return L


def midpoint_elbo_path_edges(W, R, pi_prod, t, iters=6, tol=None, precomp=None,
                             fp32=False, s_grid=None):
    """HONEST/SCALABLE variant: mean-field midpoint whose half-edges are the
    closed-form PATH ELBO on each half-branch. Uses ONLY the product-bridge eig
    (never a coupled expm) and is a valid lower bound (Jensen composed with the two
    half-branch path ELBOs).

    `precomp` = closed_form_L_precompute(W,R,pi_prod) (the t-independent eig+C). Pass
    it to SHARE the eig with a full-branch path ELBO you already computed.

    `s_grid`: split fractions for the midpoint. The branch is cut into e^{s t W}
    (x->z) and e^{(1-s) t W} (z->y). For ANY s the result is a valid lower bound, so
    the per-pair MAX over the grid is also valid -- i.e. the split position is a free
    variational parameter, optimised per endpoint pair. None -> [0.5] (fixed
    midpoint, the default/original bound). Each extra s costs two half-branch kernels
    (no eig) + one CAVI."""
    if precomp is None:
        precomp = EV.closed_form_L_precompute(W, R, pi_prod)
    if s_grid is None:
        s_grid = [0.5]
    best = None
    for s in s_grid:
        Lf, _, _ = EV.closed_form_L_from_precompute(precomp, float(s) * t)         # x -> z
        Lt, _, _ = EV.closed_form_L_from_precompute(precomp, (1.0 - float(s)) * t)  # z -> y
        L = _midpoint_cavi(_reshape_from(Lf), _reshape_to(Lt), iters, tol, fp32=fp32)
        best = L if best is None else np.maximum(best, L)
    return best


# ---------------------------------------------------------------------------
def selfcheck(model, tgrid=(0.1, 0.5, 2.0)):
    W = model["W"]; pij = model["pij"].reshape(NS)
    print(f"\n[{model['name']}]")
    for t in tgrid:
        L, full = midpoint_elbo_matrix(W, pij, t, return_full_joint=True)
        exact = np.log(np.maximum(EV.expm_rev(W, pij, t), FLOOR))
        # (i) exact recovery: full-joint midpoint bound == exact
        rec = float(np.max(np.abs(full - exact)))
        # (ii) strict lower bound: exact - L >= 0
        slack = float((exact - L).min())
        # (iii) mean-field gap
        gap = exact - L
        mgap = float(gap.mean())
        print(f"  t={t:<4} full-joint==exact: {rec:.2e}   min slack (exact-L): {slack:+.2e}"
              f"   mean gap: {mgap:.4f}")


if __name__ == "__main__":
    models = EV.load_models()
    for m in models:
        selfcheck(m)
