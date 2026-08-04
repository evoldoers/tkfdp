"""Binned-tau forward pass: gather P_sub/beta/W from precomputed tables
instead of computing them per branch (M12).

Same signature as tree_log_lik_jax but takes:
  - kernel_tables: {'beta': (n_bins, L), 'W': (n_bins, L, L),
                     'P_sub': (n_bins, K_c, L, A, A)}
  - branch_bin_by_level: tuple of (n_slots_l, 2) int32 per level.

Inside JIT: per branch we do a gather from P_sub_table by (bin,
classes[n], theta) instead of eigendecomp + exp + mat-mul. On deep
trees this replaces the dominant per-branch compute with cheap
memory ops.
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from .mm_clv_jax import (
    leaf_clv_jax, mm_combine_jax, mm_edge_jax, mm_mass_one_jax)


# Module-level helpers with NO closures. Everything passed explicitly.

def _kernels_at_bin(bin_idx, beta_tab, W_tab, P_sub_tab, classes):
    """Per-branch (beta, W, P_sub) via bin index; module-level so vmap
    caches by fn identity."""
    beta = beta_tab[bin_idx]
    W = W_tab[bin_idx]
    P_by_ctheta = P_sub_tab[bin_idx]
    P_sub = P_by_ctheta[classes]
    return beta, W, P_sub


_kernels_at_bin_vmap = jax.vmap(
    _kernels_at_bin, in_axes=(0, None, None, None, None))


_mm_edge_vmap = jax.vmap(
    mm_edge_jax, in_axes=({'r': 0, 's': 0, 'A': 0}, 0, 0, 0))


_mm_combine_vmap = jax.vmap(
    mm_combine_jax, in_axes=(
        {'r': 0, 's': 0, 'A': 0}, {'r': 0, 's': 0, 'A': 0}, None))


_leaf_clv_vmap = jax.vmap(leaf_clv_jax, in_axes=(0, None))


def _lookup(clvs, idx):
    return {'r': clvs['r'][idx],
              's': clvs['s'][idx],
              'A': clvs['A'][idx]}


def _propagate_level_binned(clvs_prev: dict,
                                child_pos: jnp.ndarray,       # (n_slots, 2)
                                child_bin: jnp.ndarray,       # (n_slots, 2)
                                is_identity: jnp.ndarray,     # (n_slots,)
                                pi_arch: jnp.ndarray,         # (m, L, A)
                                beta_tab: jnp.ndarray,        # (n_bins, L)
                                W_tab: jnp.ndarray,           # (n_bins, L, L)
                                P_sub_tab: jnp.ndarray,       # (n_bins, K_c, L, A, A)
                                classes: jnp.ndarray) -> dict:
    """Level propagation using bin-indexed gathers."""
    left_idx = child_pos[:, 0]
    right_idx = child_pos[:, 1]
    left_bin = child_bin[:, 0]                                # (n_slots,)
    right_bin = child_bin[:, 1]

    left_child = _lookup(clvs_prev, left_idx)
    right_child = _lookup(clvs_prev, right_idx)

    beta_L, W_L, P_L = _kernels_at_bin_vmap(
        left_bin, beta_tab, W_tab, P_sub_tab, classes)
    beta_R, W_R, P_R = _kernels_at_bin_vmap(
        right_bin, beta_tab, W_tab, P_sub_tab, classes)

    msg_L = _mm_edge_vmap(left_child, P_L, beta_L, W_L)
    msg_R = _mm_edge_vmap(right_child, P_R, beta_R, W_R)
    combined = _mm_combine_vmap(msg_L, msg_R, pi_arch)

    is_id = is_identity[:, None]
    is_id_A = is_identity[:, None, None, None]
    return {
        'r': jnp.where(is_id > 0.5, left_child['r'], combined['r']),
        's': jnp.where(is_id > 0.5, left_child['s'], combined['s']),
        'A': jnp.where(is_id_A > 0.5, left_child['A'], combined['A']),
    }


def tree_log_lik_jax_binned(
    leaf_obs: jnp.ndarray,
    leaf_mask: jnp.ndarray,
    child_pos_by_level: 'tuple[jnp.ndarray, ...]',
    child_bin_by_level: 'tuple[jnp.ndarray, ...]',
    is_identity_by_level: 'tuple[jnp.ndarray, ...]',
    root_slot: int,
    pi_arch: jnp.ndarray,
    rho: jnp.ndarray,
    beta_tab: jnp.ndarray,
    W_tab: jnp.ndarray,
    P_sub_tab: jnp.ndarray,
    classes: jnp.ndarray,
) -> jnp.ndarray:
    """Single-tree forward log-lik using precomputed kernel tables.

    Per-level rescaling (M14) keeps r, s in a stable numeric range so
    deep trees (2315+ leaves at D_bucket ~ 46) don't underflow the
    mass to 0. After each level, we factor out
        scale_v = max(max_slot(r_v), max_slot(s_v))
    at every slot, and accumulate log_scale = sum over levels of
    log(scale). Final log-lik = log(mass_at_root) + log_scale.

    Because A stays normalised (<pi, A>_n = 1) by construction, only
    r and s absorb the scaling. Applying the same scalar to r and s
    at each slot preserves the (r * prod A + s * 1) form.
    """
    # Per-slot running log-scale (Felsenstein scaling). A single per-LEVEL
    # scalar is WRONG: the moment-matching combine multiplies the two child
    # r-masses, so a scale applied to both children compounds at the parent
    # while a per-level accumulator credits it only once -> the log-lik is
    # under-counted by an amount that grows with tree depth (correct only
    # for a lone cherry). Tracking the scale PER SLOT and summing children's
    # log-scales through the combine makes the bookkeeping exact through the
    # multiplicative combine, while still keeping r, s in a stable range.
    #
    # The level traversal is a lax.scan, NOT a Python-unrolled loop: unrolling
    # builds an HLO graph whose size grows with tree depth (~5-15s to compile
    # per shape); scanning compiles ONE level's body once (measured ~8x faster
    # compile at depth 12). To scan, every level is padded to a uniform slot
    # width N_max (static, from shapes); the extra pad slots (child_pos=0,
    # is_identity=0) read slot 0, are never referenced by root, and the rescale
    # is per-slot, so they are inert -- bit-identical to the unrolled loop.
    leaf_clvs = _leaf_clv_vmap(leaf_obs, pi_arch)
    n_levels = len(child_pos_by_level)
    if n_levels == 0:
        clvs = leaf_clvs
        logscale = jnp.zeros((leaf_clvs['r'].shape[0],), dtype=jnp.float64)
    else:
        N_max = max(leaf_clvs['r'].shape[0],
                    max(int(cp.shape[0]) for cp in child_pos_by_level))

        def _pad0(a):
            return jnp.pad(a, [(0, N_max - a.shape[0])]
                              + [(0, 0)] * (a.ndim - 1))

        leaf_p = {k: _pad0(leaf_clvs[k]) for k in leaf_clvs}
        cp_all = jnp.stack([_pad0(child_pos_by_level[l])
                               for l in range(n_levels)])
        cb_all = jnp.stack([_pad0(child_bin_by_level[l])
                               for l in range(n_levels)])
        id_all = jnp.stack([_pad0(is_identity_by_level[l])
                               for l in range(n_levels)])

        def _level_body(carry, xs):
            clvs, logscale = carry
            child_pos, child_bin, is_identity = xs
            left_idx = child_pos[:, 0]
            right_idx = child_pos[:, 1]
            # Masses multiply across subtrees -> log-scales ADD. Identity
            # pass-through carries only the left child's accumulated scale.
            child_logscale = jnp.where(
                is_identity > 0.5, logscale[left_idx],
                logscale[left_idx] + logscale[right_idx])
            clvs = _propagate_level_binned(
                clvs, child_pos, child_bin, is_identity, pi_arch,
                beta_tab, W_tab, P_sub_tab, classes)
            # Factor each slot's own max(r, s) over theta out of (r, s) and
            # fold it into that slot's running log-scale. A is left untouched
            # (stays normalised <pi, A>_n = 1), so the (r * prod A + s) form is
            # preserved and mass is homogeneous degree 1 in (r, s).
            r = clvs['r']; s = clvs['s']
            red = tuple(range(1, r.ndim))
            slot_scale = jnp.maximum(
                jnp.maximum(jnp.max(r, axis=red), jnp.max(s, axis=red)),
                1e-300)
            denom = slot_scale.reshape((-1,) + (1,) * (r.ndim - 1))
            clvs = {'r': r / denom, 's': s / denom, 'A': clvs['A']}
            return (clvs, child_logscale + jnp.log(slot_scale)), None

        logscale0 = jnp.zeros((N_max,), dtype=jnp.float64)
        (clvs, logscale), _ = jax.lax.scan(
            _level_body, (leaf_p, logscale0), (cp_all, cb_all, id_all))
    root_clv = {
        'r': jnp.take(clvs['r'], root_slot, axis=0),
        's': jnp.take(clvs['s'], root_slot, axis=0),
        'A': jnp.take(clvs['A'], root_slot, axis=0),
    }
    mass = mm_mass_one_jax(root_clv, rho)
    return (jnp.log(jnp.maximum(mass, 1e-300))
              + jnp.take(logscale, root_slot, axis=0))
