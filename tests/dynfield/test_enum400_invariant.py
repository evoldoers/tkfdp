"""enum400 invariant-bin fast path == generic r=0 forward, for cluster sizes
m in {1,2,3} (the general-m branch). At r=0 the field is frozen so the columns
are conditionally independent; the fast path is a rho-weighted product of
per-column static forwards. Requires the local thin128 corpus + SIFTS partitions
(skips otherwise)."""
import sys, json
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path.home() / "tkf-mixdom" / "python"))

CLV = Path("data/pfam_processed_clv_top1000_thin128")
PDIR = Path("data/pdb_partition_clv_top1000_sifts")
pytestmark = pytest.mark.skipif(not CLV.exists() or not PDIR.exists(),
                                reason="local corpus/partition data absent")


def test_invariant_fast_path_m123():
    from tkfdp.bio import load_split
    from tkfdp.lg08 import S_LG08
    from tkfdp.coupling.dynfield.phylo_elbo.corpus_state import build_corpus_state
    from tkfdp.coupling.dynfield.phylo_elbo import field_rate_trainer as frt
    from tkfdp.coupling.dynfield.phylo_elbo.rate_hetero import gamma_plus_inv_rates
    from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c20
    import jax; jax.config.update("jax_enable_x64", True)

    K_a = 20
    keep = set(load_split()["train"])
    fams = [f for f in json.loads((CLV / "index.json").read_text())["families"]
            if f in keep and (PDIR / f"{f}.npz").exists()][:1]
    pa = np.asarray(le_gascuel_c20()[0], float); pa /= pa.sum(1, keepdims=True)
    st = build_corpus_state([str(CLV / f"{f}.npz") for f in fams], K_c=K_a * K_a,
                            K_a=K_a, L_field=2, pi_archetype=pa,
                            S=np.asarray(S_LG08), rho_chain=0.15,
                            rng=np.random.default_rng(0), n_tau_bins=32,
                            max_cluster_size=4, verbose=False)     # cap>2
    st.arch_assignment = np.stack([np.arange(K_a * K_a) // K_a,
                                   np.arange(K_a * K_a) % K_a], 1).astype(np.int32)
    st.rho = np.array([0.6, 0.4]); st.refresh_pi_field()
    # size-3, size-2, size-1 clusters over the first family's columns
    clusters = [(0, np.array([0, 1, 2], np.int32)),
                (0, np.array([3, 4], np.int32)),
                (0, np.array([5], np.int32))]
    rates, weights = gamma_plus_inv_rates(3, 0.5, 0.5)
    fr = frt.build_field_rate_state(st, clusters, rates, weights)
    fr.enum400 = True; frt.init_ll(fr)

    for ci, m in [(0, 3), (1, 2), (2, 1)]:
        for pos in range(m):
            specs = [frt._spec(fr, ci, base_override=c, pos=pos)
                     for c in range(K_a * K_a)]
            generic = frt._score_specs_rates(fr, specs)[:, 0]      # r=0 column
            fast = frt.cn_invariant_column_enum400(fr, ci, pos)
            err = float(np.abs(generic - fast).max())
            assert err < 1e-8, (f"m={m}", f"pos={pos}", err)


if __name__ == "__main__":
    test_invariant_fast_path_m123()
    print("enum400 invariant fast path OK for m in {1,2,3}")
