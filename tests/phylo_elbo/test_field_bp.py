"""field_bp.field_logbp (exact BP on the L-state field-jump tree MRF) matches
brute-force enumeration of theta at every node and Delta on every edge."""
import itertools
import numpy as np
import pytest

from tkfdp.coupling.dynfield.phylo_elbo.field_bp import field_logbp
from tkfdp.coupling.dynfield.phylo_elbo.tree import make_cherry, make_balanced_binary


def _brute(tree, npot, epot, L, nd):
    nn = tree.n_nodes; root = tree.root
    Z = 0.0
    qn = [np.zeros(L) for _ in range(nn)]
    qe = {v: np.zeros((L, L, nd)) for v in epot}
    edges = [v for v in range(nn) if v != root]
    for th in itertools.product(range(L), repeat=nn):
        for dl in itertools.product(range(nd), repeat=len(edges)):
            w = np.exp(npot.get(root, np.zeros(L))[th[root]])
            for ei, v in enumerate(edges):
                u = int(tree.parent[v])
                w *= np.exp(epot[v][th[u], th[v], dl[ei]])
                w *= np.exp(npot.get(v, np.zeros(L))[th[v]])
            Z += w
            for v in range(nn):
                qn[v][th[v]] += w
            for ei, v in enumerate(edges):
                u = int(tree.parent[v])
                qe[v][th[u], th[v], dl[ei]] += w
    return (np.log(Z), [q / q.sum() for q in qn],
            {v: qe[v] / qe[v].sum() for v in qe})


@pytest.mark.parametrize("L,nd", [(2, 2), (3, 2), (3, 1)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_field_bp_matches_brute(L, nd, seed):
    rng = np.random.default_rng(seed)
    for tree in [make_cherry(0.5, np.array([[0], [0]], np.int32)),
                 make_balanced_binary(depth=2, tau=0.4,
                                      leaf_obs=np.array([[0]] * 4, np.int32))]:
        npot = {v: (rng.normal(size=L) if v == tree.root else rng.normal(size=L) * 0.3)
                for v in range(tree.n_nodes)}
        epot = {v: rng.normal(size=(L, L, nd))
                for v in range(tree.n_nodes) if v != tree.root}
        lZ, qn, qe = field_logbp(tree, npot, epot)
        bZ, bqn, bqe = _brute(tree, npot, epot, L, nd)
        assert abs(lZ - bZ) < 1e-9
        assert max(np.abs(qn[v] - bqn[v]).max() for v in range(tree.n_nodes)) < 1e-9
        assert max(np.abs(qe[v] - bqe[v]).max() for v in bqe) < 1e-9
