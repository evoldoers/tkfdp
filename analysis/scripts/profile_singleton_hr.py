"""Profile the singleton-HR path to find why it dominates a full-corpus fine-tune.
Times: (a) the 191-component swap-DM class-prior overhead in cluster_marginals,
(b) the exact JAX HR on PAIRS vs SINGLETONS, cold (compile) vs warm."""
from __future__ import annotations
import time, sys
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import precompute_pairing as PP
from tkfdp.coupling.dynfield.phylo_elbo import supervised_trainer as ST
from tkfdp.lg08 import PI_LG08
from fit_pdb_hyperparams import load_pairs

N_FAM = 60
byfam = load_pairs({"saltbridge"}, split="train")
fams = list(byfam.keys())[:N_FAM]
byfam = {f: byfam[f] for f in fams}
t0 = time.time()
ds, rates, weights = PP.build_enum400_ds(fams)
ST.set_clv_dir("data/pfam_processed_clv_top1000_thin128")
st = ds.state; Kc, Ka = st.K_c, st.K_a
w = np.asarray(weights, float)
dm = ST.make_swap_dm(Kc, Ka)
print(f"# ds built ({N_FAM} fams) in {time.time()-t0:.0f}s", flush=True)

sing_cols, n_tot, n_cov = ST.singleton_cols(ds, byfam, frac=1.0, per_fam_cap=30, seed=0)
recs_p, _, _, _ = ST.score_perbin_fast(ds, byfam, topN=8)
recs_s = ST.score_singletons_perbin(ds, sing_cols, topN=8)
print(f"# clusters: {len(recs_p)} pairs, {len(recs_s)} singletons", flush=True)

def tm(label, fn, n=1):
    t = time.time(); r = fn()
    for _ in range(n - 1):
        r = fn()
    dt = (time.time() - t) / n
    print(f"#   {label:42s} {dt:8.2f}s", flush=True)
    return r

print("\n# ---- (a) DM class-prior overhead (cluster_marginals) ----")
tm("cluster_marginals FLAT prior (None)", lambda: ST.cluster_marginals_perbin(recs_p + recs_s, None))
tm("cluster_marginals +191-comp swap DM", lambda: ST.cluster_marginals_perbin(recs_p + recs_s, dm))

print("\n# ---- (b) exact JAX HR: pairs vs singletons, cold vs warm ----")
def hr(recs):
    return ST.exact_hr_per_archetype(ds, recs, w, st.S, dm, hr_backend="jax", b_chunk=32)
tm("HR PAIRS      cold (compile+run)", lambda: hr(recs_p))
tm("HR PAIRS      warm", lambda: hr(recs_p))
tm("HR SINGLETONS cold (compile+run)", lambda: hr(recs_s))
tm("HR SINGLETONS warm", lambda: hr(recs_s))
tm("HR SINGLETONS warm (2nd)", lambda: hr(recs_s))
print(f"\n# per-singleton warm HR cost = above / {len(recs_s)} singletons")
