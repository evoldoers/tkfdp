#!/usr/bin/env python3
"""FSA on BAliBASE with a per-site profile-mixture substitution model.

Wraps paper 1's BAliBASE pipeline (tkf-mixdom
``experiments/expected_pairwise_balibase.py``: same pairs, same sequence
annealing, same SP/TC scoring) and changes only the pair-HMM emissions and
the per-pair time estimate.

Substitution models (``--subst``):
    LG      paper 1's "LG08" (tkf-mixdom's table, which differs from the
            published LG in 171 of 190 exchangeabilities)
    LGp     the published LG (PAML lg.dat)
    C20p    as C20 with the published LG exchangeabilities
    C20     LG exchangeabilities with the C20 profiles, each class at mean rate 1
    C20g    as C20 with one global scale (the mixture has mean rate 1)
    CML20   the CherryML 20-class mixture of paper 1 (21-state, gap state
            dropped, weights unchanged), used per site
    CML20r  as CML20 with class weights conditioned on a residue being
            present, w_k (1 - pi^k_gap)

``--gamma ALPHA`` adds four discrete Gamma rate categories (Yang 1994 mean
rates) to every class.

A match column emits (a, b) with probability J(t)_ab = sum_k w_k pi^k_a M^k(t)_ab,
insert and delete columns emit from pibar = sum_k w_k pi^k.  Since the class is
drawn independently per column this is the exact class-marginal pair HMM, at
the same cost as a single substitution model.

Per pair:
  1. E-step: Forward-Backward under the indel model + LG at tau = 1 (as paper 1).
  2. tau_LG: 5 Newton steps on the expected complete-data log-likelihood (paper 1).
  3. ``--clock one``: one time for both processes, from Newton on the same
     objective with the LG match term replaced by sum_ab W_ab log J(tau)_ab.
     ``--clock two``: indel time tau_LG; substitution time from Newton on
     sum_ab W_ab log J(tau)_ab alone (``--newton-sub`` steps, clipped to
     ``--tmax-sub``; paper 1's values 5 and 10 are the defaults).
     ``--clock anchor``: both at tau_LG (paper 1's anchor time).
     ``--em-rounds n`` repeats the E-step (under the current model and times)
     and step 3 n-1 more times; with ``--clock two`` the later rounds refit the
     indel time from the indel term alone.
  4. Forward-Backward at those times -> match posteriors -> FSA.

With ``--subst LG --clock one --em-rounds 1`` this is paper 1's pipeline
(checked against ``_pairwise_posteriors_mixfrag_jax`` in ``--selftest``).

Per-family results are appended to ``<out>.jsonl`` as they finish (rerunning
skips families already there); the summary JSON is written at the end.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import json
from functools import partial
import sys
import time
from pathlib import Path

import numpy as np

MIXDOM_PY = Path.home() / "tkf-mixdom" / "python"
sys.path.insert(0, str(MIXDOM_PY))
sys.path.insert(0, str(MIXDOM_PY / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax
import jax.numpy as jnp

if os.environ.get("JAX_CACHE_DIR"):
    jax.config.update("jax_compilation_cache_dir", os.environ["JAX_CACHE_DIR"])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

from tkfmixdom.jax.core.params import mixfrag_trans
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.dp.hmm import _pad_to_bin, _pad_seq, forward_backward_2d
from tkfmixdom.jax.models.left_regular import make_mixfrag_pair_hmm, M as M_STATE
from tkfmixdom.util.expected_pair_f1 import aggregate_corpus

import expected_pairwise_balibase as epb
from tkfdp.progalign import lg_paml
from tkfdp.progalign.model import Subst, make_subst

A = 20
# Indel parameters exactly as recorded in paper 1's BAliBASE output JSONs
# (the TKF92 row used tkf92_fitted_params.json as it was then; the file has
# since been refit, so the values are pinned here).
INDEL = {
    "mixfrag": dict(lam=0.03392424266263374, mu=0.034690417879181354,
                    exts=[0.41396306992021775, 0.8603085081413245],
                    weights=[0.7008607354066809, 0.29913926459331913]),
    "tkf92": dict(lam=0.045805950961442965, mu=0.04679949003914898,
                  exts=[0.6834551539557077], weights=[1.0]),
}
CHERRYML_NPZ = MIXDOM_PY / "pfam" / "cherryml_mixture_C20_n5000.npz"


def subst_from_Q(Q, pi, w, name):
    """Subst for arbitrary reversible per-class chains (Q^k, pi^k), weights w."""
    Q, pi, w = np.asarray(Q, float), np.asarray(pi, float), np.asarray(w, float)
    sq = np.sqrt(pi)
    B = sq[:, :, None] * Q / sq[:, None, :]
    B = 0.5 * (B + np.swapaxes(B, 1, 2))
    lam, Psi = np.linalg.eigh(B)
    K = len(w)
    return Subst(jnp.zeros((A, A)), jnp.asarray(pi), jnp.asarray(w / w.sum()),
                 jnp.ones(K), jnp.asarray(lam), jnp.asarray(Psi), name=name)


def gamma_rates(alpha, n=4):
    """Mean rates of n equiprobable discrete Gamma(alpha, 1/alpha) categories (Yang 1994)."""
    from scipy.stats import gamma as G
    from scipy.special import gammainc
    q = G.ppf(np.arange(1, n) / n, alpha, scale=1 / alpha)
    return n * np.diff(gammainc(alpha + 1, np.concatenate([[0], q * alpha, [np.inf]])))


def with_gamma(sub, alpha, n=4):
    """Each class k becomes n classes (k, g) with weight w_k/n and rates scaled by r_g."""
    r = jnp.asarray(gamma_rates(alpha, n))
    K = sub.K
    return Subst(sub.S, jnp.repeat(sub.P, n, 0), jnp.repeat(sub.w, n) / n,
                 jnp.repeat(sub.R, n) * jnp.tile(r, K),
                 (sub.lam[:, None, :] * r[None, :, None]).reshape(K * n, A),
                 jnp.repeat(sub.Psi, n, 0), name=f"{sub.name}+G{alpha}")


def load_subst(kind, gamma=None):
    if gamma:
        return with_gamma(load_subst(kind), gamma)
    if kind == "LG":
        Q, pi = rate_matrix_lg()
        return subst_from_Q(np.asarray(Q)[None], np.asarray(pi)[None], [1.0], "LG")
    if kind in ("C20", "C20g"):
        return make_subst(kind)
    if kind in ("LGp", "C20p"):
        S, pi_lg = lg_paml.exchangeability()
        if kind == "LGp":
            P, w = pi_lg[None], np.ones(1)
        else:
            from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c20
            P, w, _ = le_gascuel_c20()
            P = np.asarray(P, float) / np.asarray(P, float).sum(1, keepdims=True)
            w = np.asarray(w, float) / np.sum(w)
        Q = S[None] * P[:, None, :]
        Q[:, np.arange(A), np.arange(A)] = -Q.sum(2)
        Q = Q / np.einsum("ka,kaa->k", P, -Q)[:, None, None]   # each class at mean rate 1
        return subst_from_Q(Q, P, w, kind)
    if kind in ("CML20", "CML20r"):
        d = np.load(CHERRYML_NPZ, allow_pickle=True)
        S = np.asarray(d["S"], float)[:, :A, :A]
        pi21 = np.asarray(d["pi"], float)
        pi21 = pi21 / pi21.sum(1, keepdims=True)
        pi = pi21[:, :A] / pi21[:, :A].sum(1, keepdims=True)
        w = np.asarray(d["weights"], float)
        if kind == "CML20r":
            w = w * (1.0 - pi21[:, A])
        Q = S * pi[:, None, :]
        Q[:, np.arange(A), np.arange(A)] = 0.0
        Q[:, np.arange(A), np.arange(A)] = -Q.sum(2)
        return subst_from_Q(Q, pi, w, kind)
    raise ValueError(kind)


def _sub_arrays(sub):
    return (sub.P, sub.w, sub.lam, sub.Psi)


def _joint(arrs, t):
    P, w, lam, Psi = arrs
    sq = jnp.sqrt(P)
    core = jnp.einsum("kxi,ki,kyi->kxy", Psi, jnp.exp(lam * t), Psi)
    Mt = core * (sq[:, None, :] / sq[:, :, None])
    return jnp.einsum("k,ka,kab->ab", w, P, Mt)


def _log(x):
    return jnp.log(jnp.maximum(x, 1e-300))


def _indel_ll(log_tau, n_trans, lam, mu, exts, weights):
    chi = mixfrag_trans(lam, mu, jnp.exp(log_tau), exts, weights)
    return jnp.sum(n_trans * _log(chi))


def _sub_ll(log_tau, W, arrs):
    return jnp.sum(W * _log(_joint(arrs, jnp.exp(log_tau))))


def _newton(f, log_tau0, n_newton, tmax=10.0):
    g, h = jax.grad(f), jax.grad(jax.grad(f))
    lt = log_tau0
    for _ in range(n_newton):
        gv, hv = g(lt), h(lt)
        safe_neg_h = jnp.where(jnp.abs(hv) > 1e-10, -hv, 1.0)
        lt = lt + jnp.clip(gv / safe_neg_h, -1.0, 1.0)
    return jnp.exp(jnp.clip(lt, jnp.log(1e-4), jnp.log(tmax)))


@jax.jit
def _hmm(t_ind, t_sub, lam, mu, exts, weights, arrs):
    """Pair-HMM parameters: MixFrag transitions at t_ind, class-marginal emissions at t_sub."""
    log_trans, st, _, _ = make_mixfrag_pair_hmm(
        lam, mu, t_ind, exts, weights, jnp.zeros((A, A)), jnp.ones(A) / A)
    J = _joint(arrs, t_sub)
    pibar = J.sum(1)
    return log_trans, st, J / pibar[:, None], pibar


@jax.jit
def _match_W(post, st, x, y):
    """Match posteriors (Lx_pad, Ly_pad) and expected match counts W_ab."""
    Lx_pad, Ly_pad = x.shape[0], y.shape[0]
    is_M = (st == M_STATE).astype(jnp.float64)
    mp = jnp.einsum("ijs,s->ij", post[1:Lx_pad + 1, 1:Ly_pad + 1, :], is_M)
    X = jax.nn.one_hot(x, A, dtype=jnp.float64)
    Y = jax.nn.one_hot(y, A, dtype=jnp.float64)
    return mp, jnp.einsum("ij,ia,jb->ab", mp, X, Y)


@partial(jax.jit, static_argnames=("mode", "n_newton", "tmax"))
def _fit_time(lt0, n_trans, W, lam, mu, exts, weights, arrs, mode, n_newton, tmax):
    """Newton on log t; mode: 'joint' (indel + substitution), 'indel', or 'sub'."""
    def f(l):
        v = 0.0
        if mode in ("joint", "indel"):
            v = v + _indel_ll(l, n_trans, lam, mu, exts, weights)
        if mode in ("joint", "sub"):
            v = v + _sub_ll(l, W, arrs)
        return v
    return _newton(f, lt0, n_newton, tmax)


def make_pair_fn(clock, em_rounds, n_newton=5, n_newton_sub=5, tmax_sub=10.0):
    """Per-pair posteriors.  Only forward_backward_2d (module-level jit in
    tkf-mixdom, as in paper 1) and _match_W depend on the sequence lengths, so
    each length bin compiles one Forward-Backward shared by every variant."""
    def pair_fn(x, y, Lx, Ly, lam, mu, exts, weights, arrs_lg, arrs):
        t_ind = t_sub = jnp.float64(1.0)
        e_arrs = arrs_lg
        for rnd in range(em_rounds):
            log_trans, st, sub, pibar = _hmm(t_ind, t_sub, lam, mu, exts, weights, e_arrs)
            _, post, n_trans = forward_backward_2d(
                log_trans, st, x, y, sub, pibar, real_Lx=Lx, real_Ly=Ly)
            _, W = _match_W(post, st, x, y)
            lt_ind0 = jnp.log(t_ind) if rnd else jnp.float64(0.0)
            lt_sub0 = jnp.log(t_sub) if rnd else jnp.float64(0.0)
            common = (n_trans, W, lam, mu, exts, weights)
            if clock == "anchor":
                t_ind = t_sub = _fit_time(lt_ind0, *common, arrs_lg, mode="joint",
                                          n_newton=n_newton, tmax=10.0)
            elif clock == "one":
                t_ind = t_sub = _fit_time(lt_ind0, *common, arrs, mode="joint",
                                          n_newton=n_newton, tmax=10.0)
            else:
                if rnd == 0:
                    # paper 1's tau under LG sets the indel clock
                    t_ind = _fit_time(lt_ind0, *common, arrs_lg, mode="joint",
                                      n_newton=n_newton, tmax=10.0)
                else:
                    t_ind = _fit_time(lt_ind0, *common, arrs, mode="indel",
                                      n_newton=n_newton, tmax=10.0)
                t_sub = _fit_time(lt_sub0, *common, arrs, mode="sub",
                                  n_newton=n_newton_sub, tmax=tmax_sub)
            e_arrs = arrs
        log_trans, st, sub, pibar = _hmm(t_ind, t_sub, lam, mu, exts, weights, arrs)
        log_prob, post, _ = forward_backward_2d(
            log_trans, st, x, y, sub, pibar, real_Lx=Lx, real_Ly=Ly)
        mp, _ = _match_W(post, st, x, y)
        return mp, t_ind, t_sub, log_prob
    return pair_fn


def make_method(indel, subst, clock, em_rounds, time_log, gamma=None,
                n_newton_sub=5, tmax_sub=10.0):
    p = INDEL[indel]
    lam, mu = jnp.float64(p["lam"]), jnp.float64(p["mu"])
    exts, weights = jnp.asarray(p["exts"]), jnp.asarray(p["weights"])
    arrs_lg = _sub_arrays(load_subst("LG"))
    arrs = _sub_arrays(load_subst(subst, gamma))
    pair_fn = make_pair_fn(clock, em_rounds, n_newton_sub=n_newton_sub,
                           tmax_sub=tmax_sub)

    def method(int_seqs, names, pairs, _raw):
        pp, failed = {}, []
        for i, j in pairs:
            xs, ys = int_seqs[names[i]], int_seqs[names[j]]
            Lx, Ly = len(xs), len(ys)
            try:
                x = _pad_seq(jnp.asarray(xs, jnp.int32), _pad_to_bin(Lx))
                y = _pad_seq(jnp.asarray(ys, jnp.int32), _pad_to_bin(Ly))
                mp, ti, ts, lp = pair_fn(x, y, jnp.int32(Lx), jnp.int32(Ly),
                                         lam, mu, exts, weights, arrs_lg, arrs)
                pp[(i, j)] = np.asarray(mp)[:Lx, :Ly]
                time_log.append((names[i], names[j], float(ti), float(ts), float(lp)))
            except Exception as e:
                jax.clear_caches()
                failed.append({"pair": [int(i), int(j)], "name_i": names[i],
                               "name_j": names[j], "Lx": Lx, "Ly": Ly,
                               "error_type": type(e).__name__,
                               "error_msg": str(e)[:500]})
        return pp, "soft", failed, {}
    return method


def selftest():
    """--subst LG --clock one --em-rounds 1 must reproduce paper 1's posteriors."""
    from tkfmixdom.jax.tree.fsa_anneal import _pairwise_posteriors_mixfrag_jax
    rng = np.random.default_rng(0)
    Q, pi = rate_matrix_lg()
    for indel in ("mixfrag", "tkf92"):
        p = INDEL[indel]
        fn = make_method(indel, "LG", "one", 1, [])
        for Lx, Ly in ((37, 52), (90, 70)):
            xs = rng.integers(0, A, Lx)
            ys = np.concatenate([xs[:Ly // 2], rng.integers(0, A, Ly - Ly // 2)])
            pp, _, failed, _ = fn({"a": xs, "b": ys}, ["a", "b"], [(0, 1)], None)
            assert not failed, failed
            x = _pad_seq(jnp.asarray(xs, jnp.int32), _pad_to_bin(Lx))
            y = _pad_seq(jnp.asarray(ys, jnp.int32), _pad_to_bin(Ly))
            ref, tau, _ = _pairwise_posteriors_mixfrag_jax(
                x, y, jnp.int32(Lx), jnp.int32(Ly), jnp.float64(p["lam"]),
                jnp.float64(p["mu"]), jnp.asarray(p["exts"]),
                jnp.asarray(p["weights"]), jnp.asarray(Q), jnp.asarray(pi))
            d = np.abs(pp[(0, 1)] - np.asarray(ref)[:Lx, :Ly]).max()
            print(f"selftest {indel} {Lx}x{Ly}: tau={float(tau):.4f} max|diff|={d:.2e}")
            assert d < 1e-8
    for kind, g in (("C20", None), ("C20g", None), ("CML20", None), ("LG", 1.5), ("C20", 0.35)):
        arrs = _sub_arrays(load_subst(kind, g))
        for t in (0.0, 0.3, 2.0):
            J = np.asarray(_joint(arrs, t))
            assert abs(J.sum() - 1) < 1e-10 and np.abs(J - J.T).max() < 1e-10, kind
            pibar = np.asarray(arrs[1] @ arrs[0])
            assert np.abs(J.sum(1) - pibar).max() < 1e-10, kind
        J0 = np.asarray(_joint(arrs, 0.0))
        assert np.abs(J0 - np.diag(np.diag(J0))).max() < 1e-12, kind
        if g:
            r = gamma_rates(g)
            assert abs(r.mean() - 1) < 1e-10
            base = _sub_arrays(load_subst(kind))
            Jg = sum(np.asarray(_joint(base, 0.7 * ri)) for ri in r) / 4
            assert np.abs(Jg - np.asarray(_joint(arrs, 0.7))).max() < 1e-12
        print(f"selftest {kind}{'+G' + str(g) if g else ''}: joint symmetric, sums to 1, margins pibar, J(0) diagonal")
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--indel", choices=list(INDEL), default="mixfrag")
    ap.add_argument("--subst", choices=["LG", "LGp", "C20", "C20p", "C20g", "CML20", "CML20r"],
                    default="C20")
    ap.add_argument("--clock", choices=["one", "two", "anchor"], default="two")
    ap.add_argument("--em-rounds", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=None,
                    help="Gamma shape for 4 discrete rate categories per class")
    ap.add_argument("--tmax-sub", type=float, default=10.0)
    ap.add_argument("--newton-sub", type=int, default=5)
    ap.add_argument("--out", help="summary JSON (per-family JSONL alongside)")
    ap.add_argument("--balibase-dir",
                    default=str(Path.home() / "bio-datasets/data/balibase/bali3pdbm"))
    ap.add_argument("--families", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    name = (f"{args.indel}_{args.subst}" + (f"G{args.gamma:g}" if args.gamma else "")
            + f"_{args.clock}" + (f"_T{args.tmax_sub:g}" if args.tmax_sub != 10 else "")
            + (f"_em{args.em_rounds}" if args.em_rounds > 1 else ""))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out.with_suffix(".jsonl")
    done = {}
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            r = json.loads(line)
            done[r["family"]] = r
    in_dir = os.path.join(args.balibase_dir, "in")
    ref_dir = os.path.join(args.balibase_dir, "ref")
    fams = sorted(f for f in os.listdir(in_dir) if os.path.exists(os.path.join(ref_dir, f)))
    if args.families:
        want = set(args.families.split(","))
        fams = [f for f in fams if f in want]
    print(f"{name}: {len(fams)} families, {len(done)} already done", flush=True)
    times = []
    method = make_method(args.indel, args.subst, args.clock, args.em_rounds, times,
                         gamma=args.gamma, n_newton_sub=args.newton_sub,
                         tmax_sub=args.tmax_sub)
    for fi, fam in enumerate(fams):
        if fam in done:
            continue
        times.clear()
        t0 = time.time()
        res = epb.process_family(fam, in_dir, ref_dir, method, fsa_mode="auto",
                                 fsa_anneal_iters=3, fsa_seed=42, fsa_sps=True,
                                 cache_method_name=None, cache_params_key=None,
                                 cache_disabled=True)
        if res is None:
            continue
        res["pair_times"] = list(times)
        res["time_total"] = time.time() - t0
        with open(jsonl, "a") as f:
            f.write(json.dumps(res) + "\n")
        done[fam] = res
        print(f"[{fi+1:>3}/{len(fams)}] {fam:<10} SP {res.get('msa_sp_g1', float('nan')):.3f} "
              f"TC {res.get('msa_tc_g1', float('nan')):.3f}  "
              f"(g0 {res.get('msa_sp_g0', float('nan')):.3f})  {res['time_total']:.0f}s",
              flush=True)

    results = [done[f] for f in fams if f in done]
    summ = {k: aggregate_corpus([{"per_pair": r[f"per_pair_{k}"]} for r in results
                                 if r.get(f"per_pair_{k}") is not None])
            for k in ("post", "hard", "opt", "fsa_sps")}
    sp1 = [r["msa_sp_g1"] for r in results if r.get("msa_sp_g1") is not None]
    tc1 = [r["msa_tc_g1"] for r in results if r.get("msa_tc_g1") is not None]
    sp0 = [r["msa_sp_g0"] for r in results if r.get("msa_sp_g0") is not None]
    tc0 = [r["msa_tc_g0"] for r in results if r.get("msa_tc_g0") is not None]
    payload = {
        "method_name": name, "indel": args.indel, "indel_params": INDEL[args.indel],
        "subst": args.subst, "gamma": args.gamma, "clock": args.clock,
        "em_rounds": args.em_rounds, "tmax_sub": args.tmax_sub,
        "newton_sub": args.newton_sub,
        "n_families": len(results),
        "mean_sp_g1": float(np.mean(sp1)) if sp1 else None,
        "mean_tc_g1": float(np.mean(tc1)) if tc1 else None,
        "mean_sp_g0": float(np.mean(sp0)) if sp0 else None,
        "mean_tc_g0": float(np.mean(tc0)) if tc0 else None,
        "n_scored_g1": len(sp1),
        **{f"corpus_{k}": v for k, v in summ.items()},
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"{name}: SP {payload['mean_sp_g1']:.4f} TC {payload['mean_tc_g1']:.4f} "
          f"(n={len(sp1)}); gap_factor=0: SP {payload['mean_sp_g0']:.4f} "
          f"TC {payload['mean_tc_g0']:.4f}", flush=True)


if __name__ == "__main__":
    main()
