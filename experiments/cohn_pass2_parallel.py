"""Pass 2 of the two-pass Cohn evaluation: the EXPENSIVE full-psi diffrax Cohn,
parallelised across worker processes on a single GPU.

Each full-psi Cohn evaluation is an endpoint-conditioned per-pair ODE solve that
runs at ~0% GPU utilisation (it is latency/dispatch-bound, not compute-bound), so
running several independent worker processes on the same GPU gives near-linear
speedup.  Workers disable XLA preallocation so many fit in GPU memory at once.

Reads the endpoint pairs and exact log-probs from cohn_pass1.json, runs full-psi
Cohn on as many of them as fit in a wall-clock budget (round-robin across cells so
an early stop still leaves every cell with balanced coverage), and writes the
merged result (Pass-1 fields + full-psi fields) to cohn_two_pass.json for plotting.

Run: JAX_ENABLE_X64=1 PYTHONPATH=src:experiments:src/tkfdp/cohn_ctbn \
     python3 experiments/cohn_pass2_parallel.py \
         [--workers 6] [--budget-hours 5.5] [--n2-max 1000] [--smoke]

Do NOT pass CUDA_VISIBLE_DEVICES on the command line: workers pin GPU 1 themselves.
"""
import os, sys, json, time, argparse
import numpy as np
import multiprocessing as mp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS1 = os.path.join(REPO, "results", "pair_models", "cohn_pass1.json")
COMPONENTS = os.path.join(REPO, "results/mixture_component_char/components_K8.npz")
OUT = os.path.join(REPO, "results", "pair_models", "cohn_two_pass.json")
MAXSTEPS = 1 << 14  # diffrax step budget; 'fail' = did not converge within it

_W = {}  # per-worker globals (populated in _init, in the child process)


def _init(components_path):
    """Runs once per worker process, BEFORE any heavy import, so we can pin the
    GPU and disable preallocation before JAX initialises CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"      # GPU 1 only (never GPU 0)
    # cuda first (default device = GPU 1) but cpu MUST be present: diffrax's max-steps
    # guard uses jax.pure_callback, which places its inputs on a CPU device. With a
    # cuda-only platform list that device is missing and the callback fails after a
    # while under many concurrent workers ("failed to find a local CPU device").
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # on-demand: ~1 GB/worker
    import jax, jax.numpy as jnp
    import fit_pair_models as FP
    import cohn_vs_elbo_pair as CV
    import ctbn
    ctbn.CTBN_MAX_STEPS = MAXSTEPS
    NA = FP.NA
    zm = np.load(components_path, allow_pickle=True)
    S_mix = np.asarray(zm["S"], float); pis = np.asarray(zm["pis"], float)
    mi = np.asarray(zm["mi_pi"], float); order = np.argsort(mi)
    models = {}
    for c in (order[len(order) // 2], order[-1]):
        _, _, _, params, _ = CV.build_glauber(S_mix, pis[c].reshape(NA, NA))
        models[f"mixture_K8_c{c}_mi{mi[c]:.3f}"] = params
    _W.update(jax=jax, jnp=jnp, CV=CV, ctbn=ctbn, NA=NA, models=models)


def _solve(task):
    """Full-psi Cohn for one pair. task = (name, t, ci, k, xstate, ystate).
    Returns (ci, k, value_or_nan, status)."""
    name, t, ci, k, xstate, ystate = task
    jax, jnp, CV, ctbn = _W["jax"], _W["jnp"], _W["CV"], _W["ctbn"]
    NA = _W["NA"]; params = _W["models"][name]
    a, b = xstate % NA, xstate // NA
    c, d = ystate % NA, ystate // NA
    try:
        le, _ = ctbn.ctbn_variational_log_cond(
            jax.random.PRNGKey(0), jnp.array([a, b]), jnp.array([c, d]),
            CV.SEQ_MASK, CV.NBR_IDX, CV.NBR_MASK, params, float(t),
            min_inc=1e-5, max_updates=128, rtol=1e-4, atol=1e-7)
        v = float(le)
        return (ci, k, v, "fail" if not np.isfinite(v) else "ok")
    except Exception:
        return (ci, k, float("nan"), "fail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget-hours", type=float, default=5.5)
    ap.add_argument("--n2-max", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.workers, a.budget_hours, a.n2_max, a.save_every = 2, 0.05, 6, 20

    d1 = json.load(open(PASS1))
    results = d1["results"]
    tgrid = d1["tgrid"]

    # flatten cells; deterministic shuffled pool order per cell
    cells = []   # (name, tkey, x[], y[], ex[])
    for name in results:
        for tk in sorted(results[name]["t"], key=float):
            cell = results[name]["t"][tk]
            cells.append((name, tk, cell["x"], cell["y"], cell["ex"]))
    rng = np.random.default_rng(1)
    order = [rng.permutation(len(c[2]))[:a.n2_max] for c in cells]
    acc = [dict(ex=[], gap=[], nfail=0, ndone=0, done_k=set()) for _ in cells]

    # resume: load already-completed (cell, k) from a previous OUT so a restart
    # continues instead of recomputing (each solve is expensive).
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))["results"]
            name_to_ci = {(name, tk): ci for ci, (name, tk, *_) in enumerate(cells)}
            for name in prev:
                for tk, pcell in prev[name].get("t", {}).items():
                    ci = name_to_ci.get((name, tk))
                    if ci is None or pcell.get("done_k") is None:
                        continue
                    A = acc[ci]
                    A["done_k"] = set(int(k) for k in pcell["done_k"])
                    A["ex"] = list(pcell.get("fullpsi_ex", []))
                    A["gap"] = list(pcell.get("gap_cohn_fullpsi", []))
                    A["ndone"] = int(pcell.get("n2_done", 0))
                    nf = pcell.get("frac_fail")
                    A["nfail"] = int(round(nf * A["ndone"])) if (nf and A["ndone"]) else 0
            nres = sum(len(A["done_k"]) for A in acc)
            if nres:
                print(f"Resuming Pass-2: {nres} solves already done, skipping them", flush=True)
        except Exception as e:
            print(f"(resume skipped: {e})", flush=True)

    # round-robin task list + (ci,k)->exact map, skipping already-done (ci,k)
    kmax = max(len(o) for o in order)
    tasks, ex_by_cik = [], {}
    for k in range(kmax):
        for ci, (name, tk, xs, ys, ex) in enumerate(cells):
            if k < len(order[ci]) and k not in acc[ci]["done_k"]:
                idx = int(order[ci][k])
                tasks.append((name, float(tk), ci, k, int(xs[idx]), int(ys[idx])))
                ex_by_cik[(ci, k)] = float(ex[idx])
    print(f"Pass-2: {len(cells)} cells, up to {kmax}/cell, {len(tasks)} candidate solves, "
          f"{a.workers} workers, budget {a.budget_hours:.2f} h", flush=True)

    def flush(tag=""):
        for ci, (name, tk, *_) in enumerate(cells):
            A = acc[ci]; cell = results[name]["t"][tk]
            cell["fullpsi_ex"] = A["ex"]
            cell["gap_cohn_fullpsi"] = A["gap"]
            cell["gap_cohn_fullpsi_mean"] = float(np.mean(A["gap"])) if A["gap"] else float("nan")
            cell["n2_done"] = A["ndone"]
            cell["frac_fail"] = (A["nfail"] / A["ndone"]) if A["ndone"] else float("nan")
            cell["done_k"] = sorted(A["done_k"])
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(tgrid=tgrid, results=results, n_grid=d1.get("n_grid"),
                           n_iter=d1.get("n_iter"), maxsteps=MAXSTEPS, workers=a.workers), f)
        os.replace(tmp, OUT)
        if tag:
            print(tag, flush=True)

    if not tasks:
        print("Nothing to do: all cells already complete.", flush=True)
        return
    t0 = time.time(); done = 0; nok = 0; stall = 0
    budget_s = a.budget_hours * 3600.0
    # maxtasksperchild recycles each worker periodically so no per-process resource
    # (GPU memory fragmentation, CPU-callback handles) accumulates over a multi-hour run.
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(COMPONENTS,),
                                        maxtasksperchild=250)
    it = pool.imap_unordered(_solve, tasks, chunksize=1)
    try:
        while True:
            try:
                ci, k, v, st = it.next(timeout=180)   # budget still fires if all workers stall
                stall = 0
            except mp.TimeoutError:
                stall += 1
                if time.time() - t0 > budget_s:
                    print("  budget reached (idle wait); stopping", flush=True); break
                if stall >= 3:
                    print("  no progress for ~9 min; assuming workers dead, stopping", flush=True); break
                print("  (waiting on slow solves...)", flush=True); continue
            except StopIteration:
                break
            A = acc[ci]; A["ndone"] += 1; A["done_k"].add(int(k))
            if st == "ok":
                exv = ex_by_cik[(ci, k)]; A["ex"].append(exv); A["gap"].append(exv - v); nok += 1
            else:
                A["nfail"] += 1
            done += 1
            if done == 40 and nok == 0:   # loud preflight: all-fail => systematic bug, not real
                print("PREFLIGHT ABORT: first 40 solves ALL failed -- likely a systematic "
                      "bug, not genuine non-convergence. Stopping before wasting the budget.", flush=True)
                break
            if done % a.save_every == 0:
                el = time.time() - t0; rate = done / el if el else 0
                eta = (len(tasks) - done) / rate if rate else 0
                flush(f"  {done}/{len(tasks)} done ({nok} ok)  {rate*60:.0f}/min  "
                      f"elapsed {el/3600:.2f}h  eta {eta/3600:.2f}h")
            if time.time() - t0 > budget_s:
                print(f"  budget reached at {done} solves; stopping", flush=True); break
    finally:
        try:
            pool.terminate(); pool.join()
        except Exception:
            pass
        flush(f"Pass-2 saved -> {OUT}  ({done} solves, {nok} ok, in {(time.time()-t0)/3600:.2f} h)")


if __name__ == "__main__":
    main()
