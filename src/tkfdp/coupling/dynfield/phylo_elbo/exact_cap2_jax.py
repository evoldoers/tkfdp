"""JAX-batched exact cap-2 Felsenstein (pairs) for the dynamic-field cluster
likelihood. Drop-in replacement for the moment-matching forward: same padded
tree, same binned P_sub / pi_field tables, same field kernels -- only the
message is the full (L, A, A) joint and the branch/combine ops are exact
(exact_cap2.py, validated to machine precision vs the compound generator).

Message per node: F (L, A, A). Branch (child->parent, tau bin b, classes c1,c2):
  NJ[t]  = beta[t] * P_sub[b,c1,t] @ F[t] @ P_sub[b,c2,t]^T
  m[t]   = pi_field[c1,t] @ F[t] @ pi_field[c2,t]
  F'[t]  = NJ[t] + (J[b] @ m)[t]
Combine at an internal node = exact elementwise product. Root mass =
sum_t rho[t] * (pi_field[c1,t] @ F_root[t] @ pi_field[c2,t]).

Per-node log-scaling keeps deep 128-leaf trees numerically safe.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .tau_binning import field_kernel_tables  # beta/W; we recompute J below


def field_beta_J_bins(bin_centers, rho, rho_chain):
    """(beta (n_bins,L), J (n_bins,L,L)) with J = P_theta - beta*I, the
    >=1-jump-with-resample weight. P_theta[i,j] = g*delta + (1-g)*rho[j]."""
    import numpy as np
    bc = np.asarray(bin_centers, float)
    rho = np.asarray(rho, float)
    g = np.exp(-rho_chain * bc)                       # (n_bins,)
    L = rho.shape[0]
    P_theta = g[:, None, None] * np.eye(L)[None] + (1 - g)[:, None, None] * rho[None, None, :]
    beta = np.exp(-rho_chain * (1 - rho)[None, :] * bc[:, None])   # (n_bins,L)
    J = P_theta - beta[:, :, None] * np.eye(L)[None]
    return jnp.asarray(beta), jnp.asarray(J)


def _leaf_msg_pair(obs, L, A):
    """obs (2,) int, gap<0 -> marginalise. Returns (L,A,A)."""
    x, y = obs[0], obs[1]
    ex = jnp.where(x < 0, jnp.ones(A), jax.nn.one_hot(jnp.maximum(x, 0), A))
    ey = jnp.where(y < 0, jnp.ones(A), jax.nn.one_hot(jnp.maximum(y, 0), A))
    M = jnp.outer(ex, ey)
    return jnp.broadcast_to(M[None], (L, A, A))


def _branch_pair(F, P1, P2, pi1, pi2, beta, J):
    """F (L,A,A); P1,P2 (L,A,A); pi1,pi2 (L,A); beta (L); J (L,L)."""
    NJ = beta[:, None, None] * jnp.einsum('lab,lbc,ldc->lad', P1, F, P2)
    m = jnp.einsum('la,lab,lb->l', pi1, F, pi2)
    return NJ + (J @ m)[:, None, None]


def exact_pair_tree_ll(
    leaf_obs, leaf_mask, child_pos_by_level, child_bin_by_level,
    is_identity_by_level, root_slot, classes,
    pi_field, P_sub_bins, beta_bins, J_bins, rho,
):
    """Exact pair log-lik on one padded tree. leaf_obs (N,2) int (gap<0);
    classes (2,) int; P_sub_bins (n_bins,K_c,L,A,A); pi_field (K_c,L,A)."""
    L = rho.shape[0]; A = pi_field.shape[-1]
    c1, c2 = classes[0], classes[1]
    pi1 = pi_field[c1]; pi2 = pi_field[c2]                    # (L,A)

    # leaf level messages: (N, L, A, A) + PER-SLOT log-scale (N,).
    # A single per-level scalar is WRONG: the product combine multiplies two
    # children, so their scales compound at the parent; a global accumulator
    # credits it once and under-counts the LL by an amount growing with depth.
    msgs = jax.vmap(lambda o: _leaf_msg_pair(o, L, A))(leaf_obs)
    logscale = jnp.zeros((msgs.shape[0],), dtype=msgs.dtype)

    for lvl in range(len(child_pos_by_level)):
        cp = child_pos_by_level[lvl]; cb = child_bin_by_level[lvl]
        idn = is_identity_by_level[lvl]

        def _combine_slot(cp_i, cb_i, id_i):
            left = msgs[cp_i[0]]; right = msgs[cp_i[1]]
            lsL = logscale[cp_i[0]]; lsR = logscale[cp_i[1]]
            bL, bR = cb_i[0], cb_i[1]
            mL = _branch_pair(left, P_sub_bins[bL, c1], P_sub_bins[bL, c2],
                              pi1, pi2, beta_bins[bL], J_bins[bL])
            mR = _branch_pair(right, P_sub_bins[bR, c1], P_sub_bins[bR, c2],
                              pi1, pi2, beta_bins[bR], J_bins[bR])
            out = jnp.where(id_i > 0.5, left, mL * mR)
            out_ls = jnp.where(id_i > 0.5, lsL, lsL + lsR)
            return out, out_ls

        new, new_ls = jax.vmap(_combine_slot)(cp, cb, idn)
        mx = jnp.maximum(jnp.max(new, axis=(1, 2, 3)), 1e-300)   # per-slot
        msgs = new / mx[:, None, None, None]
        logscale = new_ls + jnp.log(mx)

    root = msgs[root_slot]
    m_root = jnp.einsum('la,lab,lb->l', pi1, root, pi2)
    mass = jnp.sum(rho * m_root)
    return jnp.log(jnp.maximum(mass, 1e-300)) + logscale[root_slot]


def bucketed_exact_ll(clusters, bin_idx_per_cluster, rho, kernel_tables,
                      rho_chain_eff, b_chunk: int = 256):
    """Exact cap-2 log-lik per (PaddedTree, classes) cluster, dispatching m=1
    (singleton) / m=2 (pair) to the exact forwards. Drop-in analogue of
    bucketed_tree_log_lik_padded_binned, reusing its shape/bin batching.

      kernel_tables: {'P_sub','pi_field','bin_centers'} (P_sub is rho_chain-
        independent; reused across field-rate bins).
      rho_chain_eff: effective field rate (rho_chain * r_g for this bin).
    """
    import numpy as np
    from collections import defaultdict
    from .tree_batch import (_bucket_shape_batch, _stack_padded_batch_binned,
                             bucket_key_from_padded)
    jax.config.update("jax_enable_x64", True)
    beta_bins, J_bins = field_beta_J_bins(
        kernel_tables['bin_centers'], rho, rho_chain_eff)
    P_sub = jnp.asarray(kernel_tables['P_sub'])
    pi_field = jnp.asarray(kernel_tables['pi_field'])
    rho_j = jnp.asarray(np.asarray(rho, dtype=np.float64))

    out = np.zeros(len(clusters), dtype=np.float64)
    by_bucket = defaultdict(list)
    for i, (pt, classes) in enumerate(clusters):
        by_bucket[bucket_key_from_padded(pt, classes)].append(i)

    for bucket, idxs in by_bucket.items():
        m_bucket = bucket[0]
        fn = exact_single_tree_ll_batch if m_bucket == 1 else exact_pair_tree_ll_batch
        for c0 in range(0, len(idxs), b_chunk):
            sub = idxs[c0:c0 + b_chunk]
            padded = [clusters[i] for i in sub]
            shape = _bucket_shape_batch(padded, bucket)
            cbin = _stack_padded_batch_binned(
                padded, bucket, [bin_idx_per_cluster[i] for i in sub])
            classes_b = jnp.asarray(np.stack(
                [np.asarray(cl, np.int32) for _, cl in padded]))
            leaf_obs = shape['leaf_obs'][:, :, :m_bucket]
            ll = fn(leaf_obs, shape['leaf_mask'], shape['child_pos_by_level'],
                    cbin, shape['is_identity_by_level'], shape['root_slot'],
                    classes_b, pi_field, P_sub, beta_bins, J_bins, rho_j)
            ll = np.asarray(ll, dtype=np.float64)
            for k, i in enumerate(sub):
                out[i] = ll[k]
    return out


def _leaf_msg_single(obs, L, A):
    x = obs[0]
    ex = jnp.where(x < 0, jnp.ones(A), jax.nn.one_hot(jnp.maximum(x, 0), A))
    return jnp.broadcast_to(ex[None], (L, A))


def _branch_single(F, P, pi, beta, J):
    """F (L,A); P (L,A,A); pi (L,A); beta (L); J (L,L)."""
    NJ = beta[:, None] * jnp.einsum('lab,lb->la', P, F)
    m = jnp.einsum('la,la->l', pi, F)
    return NJ + (J @ m)[:, None]


def exact_single_tree_ll(
    leaf_obs, leaf_mask, child_pos_by_level, child_bin_by_level,
    is_identity_by_level, root_slot, classes,
    pi_field, P_sub_bins, beta_bins, J_bins, rho,
):
    """Exact singleton (m=1) log-lik. leaf_obs (N,1) int; classes (1,)."""
    L = rho.shape[0]; A = pi_field.shape[-1]
    c = classes[0]
    pic = pi_field[c]                                        # (L,A)
    msgs = jax.vmap(lambda o: _leaf_msg_single(o, L, A))(leaf_obs)
    logscale = jnp.zeros((msgs.shape[0],), dtype=msgs.dtype)
    for lvl in range(len(child_pos_by_level)):
        cp = child_pos_by_level[lvl]; cb = child_bin_by_level[lvl]
        idn = is_identity_by_level[lvl]

        def _combine_slot(cp_i, cb_i, id_i):
            left = msgs[cp_i[0]]; right = msgs[cp_i[1]]
            lsL = logscale[cp_i[0]]; lsR = logscale[cp_i[1]]
            mL = _branch_single(left, P_sub_bins[cb_i[0], c], pic,
                                beta_bins[cb_i[0]], J_bins[cb_i[0]])
            mR = _branch_single(right, P_sub_bins[cb_i[1], c], pic,
                                beta_bins[cb_i[1]], J_bins[cb_i[1]])
            out = jnp.where(id_i > 0.5, left, mL * mR)
            out_ls = jnp.where(id_i > 0.5, lsL, lsL + lsR)
            return out, out_ls

        new, new_ls = jax.vmap(_combine_slot)(cp, cb, idn)
        mx = jnp.maximum(jnp.max(new, axis=(1, 2)), 1e-300)
        msgs = new / mx[:, None, None]; logscale = new_ls + jnp.log(mx)
    root = msgs[root_slot]
    mass = jnp.sum(rho * jnp.einsum('la,la->l', pic, root))
    return jnp.log(jnp.maximum(mass, 1e-300)) + logscale[root_slot]


def exact_single_tree_ll_batch(
    leaf_obs, leaf_mask, child_pos_by_level, child_bin_by_level,
    is_identity_by_level, root_slot, classes,
    pi_field, P_sub_bins, beta_bins, J_bins, rho,
):
    n_lvl = len(child_pos_by_level)
    in_axes = (0, 0, tuple([0] * n_lvl), tuple([0] * n_lvl), tuple([0] * n_lvl),
               0, 0, None, None, None, None, None)
    return jax.vmap(exact_single_tree_ll, in_axes=in_axes)(
        leaf_obs, leaf_mask, child_pos_by_level, child_bin_by_level,
        is_identity_by_level, root_slot, classes,
        pi_field, P_sub_bins, beta_bins, J_bins, rho)


def exact_pair_tree_ll_batch(
    leaf_obs, leaf_mask, child_pos_by_level, child_bin_by_level,
    is_identity_by_level, root_slot, classes,
    pi_field, P_sub_bins, beta_bins, J_bins, rho,
):
    """vmap exact_pair_tree_ll over a leading batch axis (per-spec tensors have
    axis 0; shared model tensors are broadcast)."""
    n_lvl = len(child_pos_by_level)
    in_axes = (0, 0, tuple([0] * n_lvl), tuple([0] * n_lvl), tuple([0] * n_lvl),
               0, 0, None, None, None, None, None)
    return jax.vmap(exact_pair_tree_ll, in_axes=in_axes)(
        leaf_obs, leaf_mask, child_pos_by_level, child_bin_by_level,
        is_identity_by_level, root_slot, classes,
        pi_field, P_sub_bins, beta_bins, J_bins, rho)
