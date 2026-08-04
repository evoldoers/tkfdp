"""JAX scorer for the archetype-orbit permutation field (Stage B, coupled model).

The cluster field is ONE F81-on-DP chain over Theta_C = product of the distinct
orbits the columns touch (up to (s_max!)^2 = 4 for cap-2), so different-orbit
columns are COUPLED (shared jump times), not factorised. We build per-cluster
POSITIONAL tables (archetype of each column at each joint field state) and reuse
exact_cap2_jax.exact_pair_tree_ll / exact_single_tree_ll verbatim by passing
classes=[0,1] (positional) with pi=(m,L,A), P=(n_bins,m,L,A,A).

Validated == the numpy reference (orbit_scorer.score_cluster_orbit) to machine
precision across L in {1,2,4}.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from .orbit_scorer import orbit_members, orbit_perms


def build_arch_P(pi_arch, S, bin_centers) -> dict:
    """Per-archetype transition bins P_arch[b,k]=expm(Q^k tau_b), (n_bins,K,A,A)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy.linalg import expm
    from .exact_cap2 import gtr_Q
    pi_arch = np.asarray(pi_arch, float); S = np.asarray(S, float)
    bc = np.asarray(bin_centers, float); K = pi_arch.shape[0]
    P = np.stack([[expm(gtr_Q(pi_arch[k], S) * t) for k in range(K)] for t in bc])
    return {'P_arch': jnp.asarray(P), 'pi_arch': jnp.asarray(pi_arch),
            'bin_centers': bc}


def joint_field(classes, orbit_id):
    """(L, [a_col (L,) per column]) for the joint field Theta_C = product of the
    distinct orbits the columns touch. a_col[phi] = archetype of that column at
    joint field state phi."""
    orbit_id = np.asarray(orbit_id)
    members = orbit_members(orbit_id)
    classes = [int(c) for c in classes]
    touched = []
    for c in classes:
        b = int(orbit_id[c])
        if b not in touched:
            touched.append(b)
    joint = list(product(*[orbit_perms(members[b]) for b in touched]))
    L = len(joint)

    def a_of(cls):
        bi = touched.index(int(orbit_id[cls]))
        return np.asarray([joint[phi][bi][cls] for phi in range(L)], np.int64)

    return L, [a_of(c) for c in classes]


def _tree_arrays(pt, bin_centers):
    import jax.numpy as jnp
    from .tau_binning import assign_bins
    cpos, cbin, idn = [], [], []
    for l in range(pt.D_bucket):
        cp = pt.child_pos[l]; cb = pt.child_branch[l]
        cpos.append(jnp.asarray(cp))
        cbin.append(jnp.asarray(assign_bins(cb, bin_centers)))
        m = ((cp[:, 0] == cp[:, 1]) & (cb[:, 0] == 0.0) & (cb[:, 1] == 0.0))
        idn.append(jnp.asarray(m.astype(np.float64)))
    return tuple(cpos), tuple(cbin), tuple(idn)


def score_cluster(pt, classes, orbit_id, tables, rho_chain, rate) -> float:
    """One cluster (m=1 or 2 columns) on a PaddedTree under the joint orbit
    field. pt.leaf_obs is (N, m). Reuses the exact_cap2_jax forwards positionally."""
    import jax.numpy as jnp
    from .exact_cap2_jax import (exact_pair_tree_ll, exact_single_tree_ll,
                                 field_beta_J_bins)
    classes = [int(c) for c in classes]
    L, a_cols = joint_field(classes, orbit_id)
    rho = np.full(L, 1.0 / L); rce = float(rho_chain) * float(rate)
    beta, J = field_beta_J_bins(tables['bin_centers'], rho, rce)
    cpos, cbin, idn = _tree_arrays(pt, tables['bin_centers'])
    pi_arch = tables['pi_arch']; P_arch = tables['P_arch']
    pos_pi = jnp.stack([pi_arch[a] for a in a_cols], axis=0)          # (m,L,A)
    pos_P = jnp.stack([P_arch[:, a] for a in a_cols], axis=1)         # (nb,m,L,A,A)
    cls01 = jnp.arange(len(classes), dtype=jnp.int32)                 # positional
    lo = jnp.asarray(pt.leaf_obs); lm = jnp.asarray(pt.leaf_mask); rt = int(pt.root_slot)
    rho_j = jnp.asarray(rho)
    if len(classes) == 1:
        return float(exact_single_tree_ll(lo, lm, cpos, cbin, idn, rt, cls01,
                                          pos_pi, pos_P, beta, J, rho_j))
    return float(exact_pair_tree_ll(lo, lm, cpos, cbin, idn, rt, cls01,
                                    pos_pi, pos_P, beta, J, rho_j))


# ---------------------------------------------------------- batched scorer

_POS_VMAP_CACHE: dict = {}


def _pos_vmap(m, n_levels):
    """jit'd vmap of the positional exact forward over a batch of units, with
    per-unit pi (B,m,L,A) and P (B,nb,m,L,A,A); classes fixed [0..m)."""
    key = (m, n_levels)
    if key not in _POS_VMAP_CACHE:
        import jax
        import jax.numpy as jnp
        from .exact_cap2_jax import exact_pair_tree_ll, exact_single_tree_ll
        fwd = exact_pair_tree_ll if m == 2 else exact_single_tree_ll
        cls = jnp.arange(m, dtype=jnp.int32)

        def one(lo, lm, cp, cb, idn, rs, pi, P, beta, J, rho):
            return fwd(lo, lm, cp, cb, idn, rs, cls, pi, P, beta, J, rho)

        in_axes = (0, 0, tuple([0] * n_levels), tuple([0] * n_levels),
                   tuple([0] * n_levels), 0, 0, 0, None, None, None)
        _POS_VMAP_CACHE[key] = jax.jit(jax.vmap(one, in_axes=in_axes))
    return _POS_VMAP_CACHE[key]


def score_units_batched(units, tables, rho_chain, rate):
    """Score a list of units `(pt, a_cols)` (a_cols = per-column (L,) archetype
    arrays, same L within a unit) in batched form, grouped by (shape, m, L).
    Returns (len(units),) log-liks. Numerically == per-unit `score_cluster`."""
    from collections import defaultdict
    import jax.numpy as jnp
    from .tree_batch import (_bucket_shape_batch, _stack_padded_batch_binned,
                             bucket_key_from_padded)
    from .exact_cap2_jax import field_beta_J_bins
    from .tau_binning import assign_bins
    bc = tables['bin_centers']; pi_arch = tables['pi_arch']; P_arch = tables['P_arch']
    out = np.zeros(len(units))
    groups = defaultdict(list)
    for i, (pt, a_cols) in enumerate(units):
        groups[(bucket_key_from_padded(pt, None), len(a_cols),
                int(len(a_cols[0])))].append(i)
    for (bkey, m, L), idxs in groups.items():
        rho = np.full(L, 1.0 / L)
        beta, J = field_beta_J_bins(bc, rho, float(rho_chain) * float(rate))
        padded = [(units[i][0], np.zeros(m, np.int32)) for i in idxs]
        shape = _bucket_shape_batch(padded, bkey)
        binidx = [[assign_bins(units[i][0].child_branch[l], bc)
                   for l in range(units[i][0].D_bucket)] for i in idxs]
        cbin = _stack_padded_batch_binned(padded, bkey, binidx)
        pos_pi = jnp.stack([jnp.stack([pi_arch[jnp.asarray(a)] for a in units[i][1]])
                            for i in idxs])                      # (B,m,L,A)
        pos_P = jnp.stack([jnp.stack([P_arch[:, jnp.asarray(a)] for a in units[i][1]],
                                     axis=1) for i in idxs])     # (B,nb,m,L,A,A)
        fn = _pos_vmap(m, len(shape['child_pos_by_level']))
        ll = np.asarray(fn(shape['leaf_obs'], shape['leaf_mask'],
                           shape['child_pos_by_level'], cbin,
                           shape['is_identity_by_level'], shape['root_slot'],
                           pos_pi, pos_P, beta, J, jnp.asarray(rho)))
        for k, i in enumerate(idxs):
            out[i] = ll[k]
    return out
