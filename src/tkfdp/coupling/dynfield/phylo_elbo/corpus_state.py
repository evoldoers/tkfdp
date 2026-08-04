"""Family-scoped corpus state + atomic-move primitives for CRP-based
phylo-ELBO Gibbs training.

Replaces the broken fixed-cluster_m chunking with a proper Ewens/CRP
prior over site partitions. Sites live in a family, are partitioned
into Potts cliques (z_s = cluster_id[s]), and every latent (c_n,
z_s, arch_assignment) is updated by atomic Gibbs moves rather than
full sweeps.

Design:
  - Per family: FULL alignment retained (no chunking). Family owns
    its PaddedTree topology + full-family leaf_obs (N_bucket, L).
    Per-site: cluster_id[s] (z_s) and classes[s] (c_n).
  - Corpus: FamilyState list + shared arch_assignment/pi_archetype/
    rho/rho_chain/S + tau bins.
  - Materialised cluster (per family, per z-block): a PaddedTree
    formed by slicing the family's leaf_obs to the block's columns
    and sqrt2-padding m up to a bucket size. Topology is shared by
    reference across all clusters of that family.
  - ll_by_cluster cache: keyed by (family_idx, frozenset(columns)).
    Invalidated in-place after every move that touches a cluster.
  - Move-type cycle and every subordinate cycle uses PermCycle
    (Fisher-Yates permutation, auto-reshuffle on exhaustion).

Ewens/CRP on z_s: Neal-3 finite-truncated Gibbs (max cluster size
capped for practicality). alpha_z fixed (default 100).

References:
  - partition_K.gibbs_sweep_cluster: the tied-theta pipeline's CRP-
    Gibbs; we mirror the Neal-3 update per site here as an atomic
    primitive.
  - math-paper/main.tex: Rung 3 (site partition CRP), Rung 4
    (single-site CRP-Gibbs + Jain-Neal split-merge).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from tkfdp.bio import GAP_INDEX
from tkfdp.pfam_data import FamilyCLV, load_clv_family

from .pfam_loader import _binarize_parent_tau
from .tree import Tree, build_tree
from .tree_padded import PaddedTree, build_padded_tree, sqrt2_bucket
from .tree_batch import (
    BucketBatchCache, bucketed_tree_log_lik_padded_binned, jit_cache_stats)


# ----------------------------------------------------- Resource probes
# Lightweight, dependency-free diagnostics so a long CRP-Gibbs run can be
# watched for the two failure modes this module is prone to: monotonic
# host-RAM growth (executable cache / allocation churn) and an ever-growing
# JIT variant set. All probes degrade to a sentinel rather than raising, so
# they are safe to call from the hot logging path.

def _rss_mb() -> float:
    """Current resident set size (host RAM) of this process, in MB."""
    try:
        with open(f"/proc/{os.getpid()}/statm") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:
        try:
            import resource
            # ru_maxrss is PEAK (KiB on Linux); a coarse fallback only.
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return float("nan")


def _jax_live_arrays() -> int:
    """Number of live JAX device arrays (-1 if unavailable)."""
    try:
        import jax
        return len(jax.live_arrays())
    except Exception:
        return -1


def _jax_device_mb() -> float:
    """Bytes-in-use on the default JAX device, in MB (nan if unavailable)."""
    try:
        import jax
        stats = jax.devices()[0].memory_stats() or {}
        b = stats.get("bytes_in_use")
        return (b / 1e6) if b is not None else float("nan")
    except Exception:
        return float("nan")
from .tau_binning import (
    build_tau_bins, collect_all_taus, gtr_P_bins, precompute_kernel_tables,
    rebuild_padded_trees_with_bins)


def _enable_x64() -> None:
    """Ensure JAX float64 is on before we materialise any jnp kernel table.

    The forward pass enables x64 on its first call, but `refresh_pi_field`
    (and the incremental updaters) create jnp float64 arrays earlier, during
    corpus build. Without x64 already on, `jnp.asarray(float64_ndarray)`
    silently downcasts to float32 — a precision bug in the kernel tables.
    Idempotent and cheap."""
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass


# -------------------------------------------------------------- Perm cycle


class PermCycle:
    """Fisher-Yates cycle: yields each element of `seq` exactly once
    per epoch (a uniformly random permutation), reshuffling when
    exhausted. Independent from other cycles; no coupled resets.

    Empty seq -> `next()` raises IndexError. Callers must handle
    length-0 (e.g. no eligible sites) themselves.
    """

    __slots__ = ('_seq', '_rng', '_perm', '_idx')

    def __init__(self, seq, rng: np.random.Generator):
        self._seq = list(seq)
        self._rng = rng
        self._perm: 'list' = []
        self._idx = 0

    def __len__(self) -> int:
        return len(self._seq)

    def next(self):
        if not self._seq:
            raise IndexError("PermCycle is empty")
        if self._idx >= len(self._perm):
            self._perm = list(self._rng.permutation(len(self._seq)))
            self._idx = 0
        v = self._seq[self._perm[self._idx]]
        self._idx += 1
        return v


# ---------------------------------------------------------- Family state


@dataclass
class FamilyState:
    """Per-family phylo-ELBO state under CRP clustering.

    Owns the family's PaddedTree topology (N_bucket, D_bucket,
    per-level child_pos/branch/slot_mask, root_slot) plus a full-
    alignment leaf_obs of shape (N_bucket, L). Per-site latents
    live here too: cluster_id (z_s) and classes (c_n).
    """
    family_id: str
    L: int
    N_bucket: int
    D_bucket: int
    n_leaves_actual: int
    depth_actual: int
    # Shared topology (referenced by all cluster PaddedTrees):
    child_pos: 'list[np.ndarray]'
    child_branch: 'list[np.ndarray]'
    slot_mask: 'list[np.ndarray]'
    root_slot: int
    leaf_mask: np.ndarray                          # (N_bucket,)
    # Full-alignment leaf residues:
    leaf_obs_full: np.ndarray                      # (N_bucket, L) int32; -1 gap
    # Latents (per-site):
    cluster_id: np.ndarray                         # (L,) int32; z_s in [0, ...)
    classes: np.ndarray                            # (L,) int32; c_n in [0, K_c)


def _family_from_clv(family_data: FamilyCLV, K_c: int,
                          rng: np.random.Generator) -> FamilyState:
    """Build a FamilyState from a FamilyCLV bundle.

    Initial state:
      - cluster_id[s] = s (all singletons).
      - classes[s] ~ Uniform[0, K_c).
    """
    L = int(family_data.L)
    N_leaves = int(family_data.n_leaves)
    # Full-alignment leaf_obs (N_leaves, L).
    leaf_full = np.where(family_data.leaf_msa == GAP_INDEX, -1,
                              family_data.leaf_msa).astype(np.int32)
    # Binarise topology, then build a family-wide PaddedTree (m = L)
    # so we get the topology + N_bucket / D_bucket / leaf_mask.
    parent_b, tau_b, _ = _binarize_parent_tau(
        family_data.parent, family_data.tau, N_leaves)
    fam_tree = build_tree(
        parent=parent_b, branch_length=tau_b, leaf_obs=leaf_full)
    fam_pt = build_padded_tree(fam_tree)                        # m == L
    return FamilyState(
        family_id=str(family_data.family),
        L=L,
        N_bucket=fam_pt.N_bucket,
        D_bucket=fam_pt.D_bucket,
        n_leaves_actual=fam_pt.n_leaves_actual,
        depth_actual=fam_pt.depth_actual,
        child_pos=fam_pt.child_pos,
        child_branch=fam_pt.child_branch,
        slot_mask=fam_pt.slot_mask,
        root_slot=fam_pt.root_slot,
        leaf_mask=fam_pt.leaf_mask,
        leaf_obs_full=fam_pt.leaf_obs,                          # (N_bucket, L)
        cluster_id=np.arange(L, dtype=np.int32),                # singletons
        classes=rng.integers(0, K_c, size=L).astype(np.int32),
    )


def load_families(clv_paths: 'list[str]', K_c: int,
                     rng: np.random.Generator) -> 'list[FamilyState]':
    return [_family_from_clv(load_clv_family(p), K_c, rng)
              for p in clv_paths]


# --------------------------------------------------- Cluster materialization


def _materialize_padded_cluster(
    fam: FamilyState, columns: np.ndarray, m_bucket: int,
) -> PaddedTree:
    """Build a PaddedTree for one cluster (family topology + given
    columns). m is padded up to m_bucket with -1 (gap) so bucket
    shape is stable across differently-sized clusters."""
    m = int(columns.shape[0])
    assert m > 0 and m <= m_bucket
    leaf_obs = np.full((fam.N_bucket, m_bucket), -1, dtype=np.int32)
    leaf_obs[:, :m] = fam.leaf_obs_full[:, columns]
    return PaddedTree(
        N_bucket=fam.N_bucket, D_bucket=fam.D_bucket,
        n_leaves_actual=fam.n_leaves_actual,
        depth_actual=fam.depth_actual,
        m=m_bucket,
        leaf_obs=leaf_obs, leaf_mask=fam.leaf_mask,
        child_pos=fam.child_pos, child_branch=fam.child_branch,
        slot_mask=fam.slot_mask, root_slot=fam.root_slot,
    )


def _cluster_classes_padded(fam: FamilyState, columns: np.ndarray,
                                 m_bucket: int) -> np.ndarray:
    """Per-site c_n for a cluster, padded with 0 to m_bucket.

    Padded slots hit gap leaves (leaf_obs = -1) so their leaf CLV is
    uniform and contributes 1 to the product — the class label at
    those slots is neutralised regardless of value."""
    out = np.zeros(m_bucket, dtype=np.int32)
    out[:columns.shape[0]] = fam.classes[columns]
    return out


def _bin_idx_padded(bin_idx_family: 'list[np.ndarray]',
                          m_bucket: int, m_actual: int
                          ) -> 'list[np.ndarray]':
    """Bin-idx-per-level arrays match a padded cluster's topology
    (bin index is per (slot, child) — independent of m). Passed
    through unchanged."""
    return bin_idx_family


def _cluster_columns_by_id(cluster_id: np.ndarray
                                  ) -> 'dict[int, np.ndarray]':
    """Group site indices by cluster label; return {cid: cols (sorted)}."""
    out: 'dict[int, list[int]]' = {}
    for s, c in enumerate(cluster_id.tolist()):
        out.setdefault(int(c), []).append(s)
    return {c: np.asarray(v, dtype=np.int32)
              for c, v in out.items()}


def _canonicalise_ids(cluster_id: np.ndarray) -> np.ndarray:
    """Relabel cluster ids to 0..K-1 in appearance order."""
    seen: 'dict[int, int]' = {}
    out = np.empty_like(cluster_id)
    for i, c in enumerate(cluster_id.tolist()):
        c = int(c)
        if c not in seen:
            seen[c] = len(seen)
        out[i] = seen[c]
    return out


# ------------------------------------------------------------- Corpus state


@dataclass
class CorpusState:
    """Corpus-level phylo-ELBO state: family list + shared parameters
    + tau bins + ll_by_cluster cache."""
    families: 'list[FamilyState]'
    K_c: int
    K_a: int
    L_field: int
    pi_archetype: np.ndarray                                    # (K_a, A)
    arch_assignment: np.ndarray                                 # (K_c, L_field)
    rho: np.ndarray                                             # (L_field,)
    rho_chain: float
    S: np.ndarray                                               # (A, A)
    bin_centers: np.ndarray
    bin_idx_by_family: 'list[list[np.ndarray]]'                 # per-family, per-level
    m_buckets: 'list[int]'                                      # sqrt2-spaced bucket sizes
    max_cluster_size: int
    alpha_z: float = 100.0
    bucket_cache: BucketBatchCache = field(default_factory=BucketBatchCache)
    ll_by_cluster: 'dict[tuple, float]' = field(default_factory=dict)
    _pi_field: 'np.ndarray | None' = None
    _kernel_tables: 'dict | None' = None

    def m_bucket_for(self, m: int) -> int:
        """sqrt2 bucket >= m, capped at max_cluster_size."""
        for b in self.m_buckets:
            if b >= m:
                return b
        return self.m_buckets[-1]

    def refresh_pi_field(self) -> None:
        """Full rebuild of pi_field + kernel_tables from current
        arch_assignment / pi_archetype. Used at init/resume and whenever a
        global param (rho, rho_chain, whole arch_assignment) changes.

        For a single-entry arch move prefer `apply_arch_slices`, which
        recomputes only the affected P_sub slices (n_bins expm) instead of
        the full K_c * L * n_bins table.

        beta/W/P_sub are stored as JAX arrays so the per-move forward's
        `jnp.asarray(...)` is a no-op (no 82 MB host->device re-upload each
        call); pi_field stays a numpy array (host-indexed when filling the
        per-cluster pi_arch batch)."""
        _enable_x64()
        import jax.numpy as jnp
        self._pi_field = np.ascontiguousarray(
            self.pi_archetype[self.arch_assignment], dtype=np.float64)
        tables = precompute_kernel_tables(
            self.bin_centers, self.rho, self.rho_chain,
            self._pi_field, self.S)
        tables['beta'] = jnp.asarray(tables['beta'])
        tables['W'] = jnp.asarray(tables['W'])
        tables['P_sub'] = jnp.asarray(tables['P_sub'])
        tables['pi_field'] = self._pi_field          # single source of truth
        self._kernel_tables = tables

    def apply_arch_slices(self, changed: 'list[tuple[int, int]]') -> None:
        """Incrementally refresh _pi_field + kernel P_sub for the (c, theta)
        entries in `changed`, AFTER arch_assignment has been mutated in
        place. Costs len(changed) * n_bins matrix exponentials, versus a
        full precompute_kernel_tables rebuild (K_c * L * n_bins). beta/W are
        untouched (they do not depend on pi_field)."""
        tables = self.kernel_tables                  # ensures built
        P_sub = tables['P_sub']
        p_by_arch = self.p_sub_by_archetype
        for (c, th) in changed:
            k = int(self.arch_assignment[c, th])
            self._pi_field[c, th] = self.pi_archetype[k]  # tables['pi_field'] aliases
            P_sub = P_sub.at[:, c, th].set(p_by_arch[k])
        tables['P_sub'] = P_sub

    def kernel_tables_with_overrides(
            self, overrides: 'dict[tuple[int, int], int]') -> dict:
        """Transient kernel_tables for a HYPOTHETICAL arch perturbation
        `overrides = {(c, theta): archetype_index k}`. Shares beta/W with the
        base tables (invariant under arch moves) and swaps in the perturbed
        P_sub slices from the per-archetype cache (see `p_sub_by_archetype`)
        plus a copied pi_field. Does NOT mutate state, does NOT recompute any
        matrix exponential (the slices are cached per archetype)."""
        base = self.kernel_tables
        P_sub = base['P_sub']
        p_by_arch = self.p_sub_by_archetype
        pi_field = np.array(base['pi_field'], copy=True)   # (K_c, L, A)
        for (c, th), k in overrides.items():
            k = int(k)
            P_sub = P_sub.at[:, c, th].set(p_by_arch[k])
            pi_field[c, th] = self.pi_archetype[k]
        out = dict(base)
        out['P_sub'] = P_sub
        out['pi_field'] = pi_field
        return out

    @property
    def pi_field(self) -> np.ndarray:
        if self._pi_field is None:
            self.refresh_pi_field()
        return self._pi_field

    @property
    def kernel_tables(self) -> dict:
        if self._kernel_tables is None:
            self.refresh_pi_field()
        return self._kernel_tables

    @property
    def p_sub_by_archetype(self):
        """Per-archetype P_sub slice cache: p[k] = gtr_P_bins(pi_archetype[k],
        S, bin_centers), shape (n_bins, A, A), as a JAX array.

        An arch move sets pi_field[c, theta] to one of the K_a fixed
        archetypes, so the corresponding P_sub slice depends ONLY on the
        archetype index -- it is invariant across moves. Computing all K_a
        slices once (K_a * n_bins expm) and indexing them turns each arch
        candidate score from `n_bins` matrix exponentials into a cached
        gather. Assumes pi_archetype is fixed during the atomic-move loop
        (it is); if archetypes are ever resampled, set `_p_sub_by_arch=None`
        to invalidate."""
        if getattr(self, '_p_sub_by_arch', None) is None:
            _enable_x64()
            import jax.numpy as jnp
            self._p_sub_by_arch = [
                jnp.asarray(gtr_P_bins(self.pi_archetype[k], self.S,
                                            self.bin_centers))
                for k in range(self.pi_archetype.shape[0])]
        return self._p_sub_by_arch


def _sqrt2_buckets_up_to(n_max: int) -> 'list[int]':
    """1, 2, 3, 4, 6, 8, 12, 16, 23, 32, ..."""
    out = [1]
    while out[-1] < n_max:
        nxt = max(out[-1] + 1, int(np.ceil(out[-1] * np.sqrt(2))))
        out.append(nxt)
    return out


def _repad_family_inplace(f: 'FamilyState', bins: 'list[np.ndarray]',
                              pad_D: int, pad_N: int) -> 'list[np.ndarray]':
    """Repad a family's tree structure + per-level bin indices to a corpus-
    global (pad_D levels, pad_N slots) shape, IN PLACE. Depth is padded with
    identity pass-through levels above the root; slots and leaves are padded
    inert (absent leaves = -1 gap -> all-ones missing-data CLV). This makes
    the compiled forward shape independent of tree depth/size, so the JIT
    variant set collapses to (m_bucket, B_pad) -- verified bit-identical
    (absent nodes are inert). Returns the repadded bin_idx list."""
    N0, D0 = f.N_bucket, f.D_bucket
    Lcols = f.leaf_obs_full.shape[1]
    lo = np.full((pad_N, Lcols), -1, np.int32); lo[:N0] = f.leaf_obs_full
    f.leaf_obs_full = lo
    lm = np.zeros(pad_N, np.float64); lm[:N0] = f.leaf_mask; f.leaf_mask = lm
    cp, cb, sm, nb = [], [], [], []
    for l in range(D0):
        a = np.zeros((pad_N, 2), np.int32); a[:f.child_pos[l].shape[0]] = f.child_pos[l]
        b = np.zeros((pad_N, 2), np.float64); b[:f.child_branch[l].shape[0]] = f.child_branch[l]
        s = np.zeros(pad_N, np.float64); s[:f.slot_mask[l].shape[0]] = f.slot_mask[l]
        bi = np.zeros((pad_N, 2), np.int32); bi[:bins[l].shape[0]] = np.asarray(bins[l])
        cp.append(a); cb.append(b); sm.append(s); nb.append(bi)
    root = f.root_slot
    for _l in range(D0, pad_D):                 # identity pass-through levels
        a = np.zeros((pad_N, 2), np.int32); a[0] = (root, root)
        s = np.zeros(pad_N, np.float64); s[0] = 1.0
        cp.append(a); cb.append(np.zeros((pad_N, 2), np.float64))
        sm.append(s); nb.append(np.zeros((pad_N, 2), np.int32))
        root = 0
    f.child_pos = cp; f.child_branch = cb; f.slot_mask = sm
    f.root_slot = root; f.N_bucket = pad_N; f.D_bucket = pad_D
    return nb


def build_corpus_state(
    clv_paths: 'list[str]',
    K_c: int, K_a: int, L_field: int,
    pi_archetype: np.ndarray, S: np.ndarray,
    rho_chain: float, rng: np.random.Generator,
    *,
    n_tau_bins: int = 32,
    max_cluster_size: int = 32,
    alpha_z: float = 100.0,
    verbose: bool = False,
) -> CorpusState:
    """Load families + build tau bins + init state."""
    families = load_families(clv_paths, K_c, rng)
    if verbose:
        n_sites = sum(f.L for f in families)
        print(f"# loaded {len(families)} families, {n_sites} sites total")

    # Collect ALL branch lengths across families for tau binning.
    # PaddedTree has child_branch per level; we need every scalar.
    all_taus: 'list[float]' = []
    for f in families:
        for lvl in f.child_branch:
            for row in lvl:
                for v in row:
                    v = float(v)
                    if v > 0:
                        all_taus.append(v)
    bin_centers = build_tau_bins(np.asarray(all_taus), n_bins=n_tau_bins)

    # Per-family bin indices per level: shape matches child_pos per level.
    bin_idx_by_family: 'list[list[np.ndarray]]' = []
    for f in families:
        per_level = []
        for lvl_branches in f.child_branch:
            # lvl_branches: (n_slots, 2) float; each entry -> nearest bin.
            n_slots = lvl_branches.shape[0]
            idx = np.zeros_like(lvl_branches, dtype=np.int32)
            for i in range(n_slots):
                for j in range(2):
                    tau = float(lvl_branches[i, j])
                    idx[i, j] = int(np.argmin(np.abs(bin_centers - tau)))
            per_level.append(idx)
        bin_idx_by_family.append(per_level)

    # Global tree padding: one compiled forward shape per (m_bucket, B_pad)
    # instead of hundreds keyed by tree depth/size. pad_D/pad_N are corpus
    # constants so the shape is stable across every scoring call.
    pad_D = max(f.D_bucket for f in families)
    pad_N = max(f.N_bucket for f in families)
    for fi, f in enumerate(families):
        bin_idx_by_family[fi] = _repad_family_inplace(
            f, bin_idx_by_family[fi], pad_D, pad_N)
    if verbose:
        print(f"# global tree pad: D={pad_D} N={pad_N} "
              f"(collapses tree-shape variety to 1)")

    m_buckets = _sqrt2_buckets_up_to(max_cluster_size)
    if verbose:
        print(f"# tau bins: {n_tau_bins}  m buckets: {m_buckets}"
              f"  alpha_z={alpha_z}  cap={max_cluster_size}")

    rho = np.full(L_field, 1.0 / L_field, dtype=np.float64)
    aa = np.tile(np.arange(min(K_c, K_a))[:, None],
                    (1, L_field)).astype(np.int32)
    if K_c > K_a:
        aa = np.concatenate(
            [aa, np.zeros((K_c - K_a, L_field), dtype=np.int32)], axis=0)

    state = CorpusState(
        families=families, K_c=K_c, K_a=K_a, L_field=L_field,
        pi_archetype=pi_archetype, arch_assignment=aa,
        rho=rho, rho_chain=rho_chain, S=S,
        bin_centers=bin_centers, bin_idx_by_family=bin_idx_by_family,
        m_buckets=m_buckets, max_cluster_size=max_cluster_size,
        alpha_z=alpha_z,
    )
    state.refresh_pi_field()
    return state


# -------------------------------------------------- ll_by_cluster utilities


def _cluster_key(fi: int, cluster_id: np.ndarray, cid: int) -> tuple:
    """Cache key: (family_idx, sorted tuple of columns)."""
    cols = tuple(int(s) for s, c in enumerate(cluster_id.tolist())
                    if int(c) == cid)
    return (fi, cols)


def _score_cluster(state: CorpusState, fi: int,
                        columns: np.ndarray, classes_padded: np.ndarray,
                        m_bucket: int) -> float:
    """Forward-only phylo-ELBO LL for one cluster."""
    fam = state.families[fi]
    pt = _materialize_padded_cluster(fam, columns, m_bucket)
    ll = bucketed_tree_log_lik_padded_binned(
        [(pt, classes_padded)],
        [state.bin_idx_by_family[fi]],
        state.rho, state.kernel_tables,
        cache=None)                          # id-based cache is unsafe for ephemeral PTs
    return float(ll[0])


# Coarse B-pad ladder: fewer distinct batch sizes => fewer distinct compiled
# shapes to warm up. Fine at the low end (small arch affected-cluster batches),
# coarse in the mid-range where the z-move's join-candidate count sweeps and
# was triggering a fresh compile almost every move. Warm forwards are ~30ms so
# the extra padding is nearly free; the set stays bounded (no JIT blowup).
_B_BUCKETS = [1, 2, 4, 8, 32, 128, 512, 4096]

# Max real specs scored per bucketed forward call. Global tree padding makes
# the forward's peak allocation scale as B * pad_N * ...; on a large-tree
# corpus an unbounded B (up to 4096 in the singleton bucket) OOMs a 48GB GPU.
# Chunking to <= _B_CHUNK bounds memory at a small throughput cost (chunks
# reuse the same power-of-2 JIT shapes). Override via env TKFDP_B_CHUNK.
_B_CHUNK = int(os.environ.get("TKFDP_B_CHUNK", "512"))


def _b_bucket(n: int) -> int:
    for b in _B_BUCKETS:
        if b >= n:
            return b
    return _B_BUCKETS[-1]


def _score_cluster_batch(
    state: CorpusState,
    specs: 'list[tuple[int, np.ndarray, np.ndarray, int]]',
    tables: 'dict | None' = None,
) -> np.ndarray:
    """Batched LL over many candidate clusters via bucketed forward.

    specs: list of (family_idx, columns, classes_padded, m_bucket)
    tables: kernel_tables to use; defaults to state.kernel_tables.

    Returns (len(specs),) array of LLs matching input order.

    Per-bucket batch size B is padded up to a fixed power-of-2
    sequence [1, 2, 4, ..., 4096] with dummy singleton specs so the
    JAX-compiled forward hits the JIT cache on subsequent moves.
    Without this, B varies with cluster count and each unique B
    triggers a ~15s compile.
    """
    if not specs:
        return np.zeros(0, dtype=np.float64)
    if tables is None:
        tables = state.kernel_tables
    # Group by bucket_key (m_bucket, N_bucket_sqrt2, D_bucket_sqrt2)
    # so each B-pad is applied per-bucket.
    from collections import defaultdict
    by_bucket: 'dict[tuple, list[int]]' = defaultdict(list)
    materialised = []                                       # (idx, pt, cls)
    for i, (fi, cols, cls_padded, mb) in enumerate(specs):
        fam = state.families[fi]
        pt = _materialize_padded_cluster(fam, cols, mb)
        materialised.append((fi, pt, cls_padded))
        key = (mb, int(pt.N_bucket), int(pt.D_bucket))   # padded == corpus const
        by_bucket[key].append(i)

    out = np.zeros(len(specs), dtype=np.float64)
    for bkey, idxs in by_bucket.items():
        # Chunk the per-bucket batch so B stays bounded: with global tree
        # padding (N_bucket/D_bucket = corpus max), a single B_pad=4096 call
        # over the singleton bucket allocates B * pad_N * ... and OOMs on a
        # large-tree corpus. Slicing into <= _B_CHUNK pieces (each padded to a
        # power of 2 for JIT-cache reuse) bounds memory; concatenation is
        # numerically identical to one big call.
        for c0 in range(0, len(idxs), _B_CHUNK):
            sub = idxs[c0:c0 + _B_CHUNK]
            n_real = len(sub)
            B_pad = _b_bucket(n_real)
            real = [(materialised[i][1], materialised[i][2]) for i in sub]
            bin_lists = [state.bin_idx_by_family[materialised[i][0]] for i in sub]
            if B_pad > n_real:
                real = real + [real[0]] * (B_pad - n_real)
                bin_lists = bin_lists + [bin_lists[0]] * (B_pad - n_real)
            lls_pad = bucketed_tree_log_lik_padded_binned(
                real, bin_lists, state.rho, tables, cache=None)
            for k, i in enumerate(sub):
                out[i] = float(lls_pad[k])
    return out


def compute_all_cluster_lls(state: CorpusState) -> None:
    """Rebuild ll_by_cluster from scratch (single batched call)."""
    specs = []
    keys = []
    for fi, fam in enumerate(state.families):
        clusters = _cluster_columns_by_id(fam.cluster_id)
        for cid, cols in clusters.items():
            m = int(cols.shape[0])
            mb = state.m_bucket_for(m)
            cls_padded = _cluster_classes_padded(fam, cols, mb)
            specs.append((fi, cols, cls_padded, mb))
            keys.append(_cluster_key(fi, fam.cluster_id, cid))
    lls = _score_cluster_batch(state, specs)
    state.ll_by_cluster = {k: float(v) for k, v in zip(keys, lls)}


def corpus_ll(state: CorpusState) -> float:
    return float(sum(state.ll_by_cluster.values()))


def cluster_size_histogram(state: CorpusState) -> np.ndarray:
    """Global histogram of cluster sizes across all families.

    Returns a 1-D int array `hist` with hist[k] = number of clusters
    of size k+1 (index 0 = singletons, 1 = doublets, ...). Trailing
    zeros trimmed.
    """
    sizes = []
    for fam in state.families:
        _, counts = np.unique(fam.cluster_id, return_counts=True)
        sizes.extend(int(c) for c in counts)
    if not sizes:
        return np.zeros(0, dtype=np.int64)
    hist = np.bincount(np.asarray(sizes, dtype=np.int64))
    return hist[1:]                                             # drop index 0


# ------------------------------------------------------------- Checkpoints


def save_checkpoint(state: CorpusState, path,
                        step: int, rng: np.random.Generator,
                        dm=None, dm_hcol=None) -> None:
    """Write compact per-family cluster_id + classes + arch_assignment
    + step + rng state to a single .npz. Everything else (topology,
    leaf_obs_full, bin_centers, tau bins) is reconstructed from CLV
    bundles on load.

    If `dm` (a DMPrior) is given, its learned mixture (alpha, pi, H, alpha_pi)
    is stored too; `dm_hcol` (a {fi: (n_col,) int array} of the per-column DM
    component, built by the caller from its own cluster-key format) persists the
    per-cluster component assignment partition-aligned, so both the frozenset
    (discovery) and int (supervised) keyings round-trip via `load_dm`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d: 'dict[str, np.ndarray]' = {
        'arch_assignment': np.asarray(state.arch_assignment, dtype=np.int32),
        'alpha_z': np.float64(state.alpha_z),
        'max_cluster_size': np.int32(state.max_cluster_size),
        'K_c': np.int32(state.K_c),
        'K_a': np.int32(state.K_a),
        'L_field': np.int32(state.L_field),
        'rho_chain': np.float64(state.rho_chain),
        'step': np.int64(step),
    }
    family_ids = []
    for fi, fam in enumerate(state.families):
        d[f'fam_{fi}_cluster_id'] = np.asarray(fam.cluster_id, np.int32)
        d[f'fam_{fi}_classes'] = np.asarray(fam.classes, np.int32)
        family_ids.append(fam.family_id)
    d['family_ids'] = np.asarray(family_ids)
    if dm is not None:
        d['dm_alpha'] = np.asarray(dm.alpha, np.float64)
        d['dm_pi'] = np.asarray(dm.pi, np.float64)
        d['dm_H'] = np.int32(dm.H)
        d['dm_alpha_pi'] = np.float64(dm.alpha_pi)
        if dm_hcol is not None:
            for fi, arr in dm_hcol.items():
                d[f'dm_hcol_fam_{fi}'] = np.asarray(arr, np.int32)
    d['rng_state_bytes'] = np.frombuffer(
        json.dumps(rng.bit_generator.state).encode(), dtype=np.uint8)
    np.savez_compressed(path, **d)


def load_dm(dm, path, state) -> 'dict | None':
    """Restore a DMPrior's mixture arrays (alpha, pi, H, alpha_pi) in place from
    a checkpoint and return {fi: (n_col,) int} of the per-column DM component (or
    None if the checkpoint predates DM persistence). The caller rebuilds `dm.h`
    from this using its own cluster-key format (frozenset(cols) for discovery,
    cluster index for supervised)."""
    d = np.load(Path(path), allow_pickle=False)
    if 'dm_alpha' not in d:
        return None
    dm.alpha = np.asarray(d['dm_alpha'], np.float64).copy()
    dm.pi = np.asarray(d['dm_pi'], np.float64).copy()
    dm.H = int(d['dm_H']); dm.alpha_pi = float(d['dm_alpha_pi'])
    hcol = {}
    for fi in range(len(state.families)):
        k = f'dm_hcol_fam_{fi}'
        if k in d:
            hcol[fi] = np.asarray(d[k], np.int32)
    return hcol


def save_checkpoint_atomic(state: CorpusState, out_dir, step: int,
                           rng: np.random.Generator,
                           name: str = "_chkpt.npz", log_stream=None,
                           t0: float = None, label: str = None,
                           dm=None, dm_hcol=None) -> 'Path':
    """Sweep-level rolling checkpoint: write to <out_dir>/<name>.tmp.npz then
    os.replace onto <out_dir>/<name>, so a crash mid-write leaves the previous
    good checkpoint intact. Persists only the canonical evolving state
    (per-family cluster_id/classes, arch_assignment, step, rng); every derived
    table is rebuilt on resume via refresh_pi_field + init_ll.

    Emits an observable one-line record (sweep, #pairs, size, wall-clock) to
    stdout and, if given, log_stream."""
    import os, time
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / name
    tmp = out_dir / (name[:-4] + "_tmp.npz")     # keep .npz so savez won't append
    save_checkpoint(state, tmp, step, rng, dm=dm, dm_hcol=dm_hcol)
    os.replace(tmp, final)
    kb = final.stat().st_size / 1024.0
    npair = sum(1 for fam in state.families
                for _, cols in _cluster_columns_by_id(fam.cluster_id).items()
                if len(cols) == 2)
    wall = "" if t0 is None else f" @ {time.time()-t0:.0f}s"
    what = label if label is not None else f"sweep={step}"
    msg = (f"#   ✓ checkpoint: {what} pairs={npair} "
           f"-> {final.name} ({kb:.1f} KB){wall}")
    print(msg, flush=True)
    if log_stream is not None:
        log_stream.write(msg + "\n"); log_stream.flush()
    return final


def load_checkpoint(state: CorpusState, path,
                        rng: np.random.Generator) -> 'tuple[int, np.random.Generator]':
    """Overwrite state.families[*].cluster_id + classes, plus
    arch_assignment, from a checkpoint. Verifies family order matches.
    Returns (step, restored_rng).
    """
    d = np.load(Path(path), allow_pickle=False)
    ckpt_ids = list(d['family_ids'].tolist())
    have_ids = [f.family_id for f in state.families]
    if ckpt_ids != have_ids:
        raise ValueError(
            f"family_ids mismatch on resume: "
            f"checkpoint has {ckpt_ids[:3]}... but corpus has "
            f"{have_ids[:3]}...")
    if int(d['K_c']) != state.K_c or int(d['K_a']) != state.K_a \
       or int(d['L_field']) != state.L_field:
        raise ValueError("K_c/K_a/L_field mismatch on resume")
    for fi, fam in enumerate(state.families):
        fam.cluster_id = np.asarray(
            d[f'fam_{fi}_cluster_id'], dtype=np.int32).copy()
        fam.classes = np.asarray(
            d[f'fam_{fi}_classes'], dtype=np.int32).copy()
    state.arch_assignment = np.asarray(
        d['arch_assignment'], dtype=np.int32).copy()
    state.alpha_z = float(d['alpha_z'])
    state.max_cluster_size = int(d['max_cluster_size'])
    state.rho_chain = float(d['rho_chain'])
    state.refresh_pi_field()
    state.ll_by_cluster = {}                                  # trigger init
    step = int(d['step'])
    rng_state = json.loads(bytes(d['rng_state_bytes']).decode())
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = rng_state
    return step, restored_rng


def apply_pdb_partition(state: CorpusState, partition_dir,
                             verbose: bool = False) -> dict:
    """Overwrite each family's cluster_id with a SUPERVISED size-2 partition
    read from <partition_dir>/<family_id>.npz (as written by
    experiments/build_pdb_partition.py). Each stored (col_i, col_j) pair
    becomes one shared cluster; every other column stays a singleton.

    Column indices in the partition npz are in the raw-alignment (== CLV)
    indexing, so they map onto cluster_id 1:1. Families with no partition
    file keep the all-singleton init. Clears the ll cache so the caller must
    recompute_all_cluster_lls afterwards.

    Returns a summary dict {n_families_matched, n_pairs, kind_totals, ...}.
    Intended to be used with a move set that EXCLUDES the 'z' move so the
    partition stays frozen for the whole run.
    """
    partition_dir = Path(partition_dir)
    n_matched = n_pairs = 0
    kind_totals: 'dict[str, int]' = {}
    n_missing = n_lmismatch = 0
    for fam in state.families:
        p = partition_dir / f"{fam.family_id}.npz"
        if not p.exists():
            n_missing += 1
            continue
        d = np.load(p, allow_pickle=True)
        if int(d['L']) != fam.L:
            # Refuse to place contacts under a different column indexing.
            n_lmismatch += 1
            if verbose:
                print(f"  ! {fam.family_id}: partition L={int(d['L'])} != "
                      f"corpus L={fam.L} -- skipped")
            continue
        pairs = np.asarray(d['pairs'], dtype=np.int32).reshape(-1, 2)
        cid = np.arange(fam.L, dtype=np.int32)                  # singletons
        for i, j in pairs:
            i, j = int(i), int(j)
            if 0 <= i < fam.L and 0 <= j < fam.L:
                cid[j] = cid[i]                                 # merge j into i
        fam.cluster_id = _canonicalise_ids(cid)
        n_matched += 1
        n_pairs += int(pairs.shape[0])
        if 'kind' in d.files:
            for k in np.asarray(d['kind']).tolist():
                kind_totals[k] = kind_totals.get(k, 0) + 1
    state.ll_by_cluster = {}                                    # force recompute
    summary = {"n_families_matched": n_matched, "n_pairs": n_pairs,
               "n_missing": n_missing, "n_lmismatch": n_lmismatch,
               "kind_totals": kind_totals}
    if verbose:
        print(f"# applied PDB partition: {n_matched} families, {n_pairs} "
              f"size-2 clusters  kinds={kind_totals}  "
              f"(missing={n_missing} Lmismatch={n_lmismatch})")
    return summary


# --------------------------------------------------------- Atomic moves


def atomic_cn_resample(state: CorpusState, fi: int, s: int,
                            rng: np.random.Generator,
                            alpha_c_log: 'np.ndarray | None' = None,
                            ) -> None:
    """K_c-way Gibbs on classes[fi][s]. Touches the one cluster
    containing site s."""
    fam = state.families[fi]
    K_c = state.K_c
    if alpha_c_log is None:
        alpha_c_log = np.zeros(K_c, dtype=np.float64)
    cid = int(fam.cluster_id[s])
    cols = _cluster_columns_by_id(fam.cluster_id)[cid]
    m = int(cols.shape[0])
    mb = state.m_bucket_for(m)
    key = _cluster_key(fi, fam.cluster_id, cid)

    # K_c candidates for classes[s]. Only the site-s slot differs;
    # everything else is fixed. Build K_c cls_padded variants, batch.
    specs = []
    for c in range(K_c):
        fam.classes[s] = c
        specs.append((fi, cols, _cluster_classes_padded(fam, cols, mb),
                        mb))
    raw_ll = _score_cluster_batch(state, specs)
    log_L = raw_ll + alpha_c_log
    log_L = log_L - log_L.max()
    p = np.exp(log_L); p /= p.sum()
    c_new = int(rng.choice(K_c, p=p))
    fam.classes[s] = c_new
    state.ll_by_cluster[key] = float(raw_ll[c_new])


def atomic_z_gibbs(state: CorpusState, fi: int, s: int,
                        rng: np.random.Generator) -> None:
    """One-site Neal-3 CRP-Gibbs update on cluster_id[fi][s].

    Candidates (with prior ratios per Ewens(alpha_z)):
      - stay in current cluster (no-op if singleton)
      - move to a fresh singleton (only if current cluster > 1)
      - join any other existing cluster (subject to max_cluster_size)

    Optimisation: all needed per-cluster scores (curr, curr_minus_s,
    singleton, and (other, other+s) for every other cluster) are
    computed in ONE batched forward call. Same-family candidates
    share tree topology so they bucket only by m -- one JIT-compiled
    vmap per m_bucket.
    """
    fam = state.families[fi]
    cid_curr = int(fam.cluster_id[s])
    clusters = _cluster_columns_by_id(fam.cluster_id)
    curr_members = clusters[cid_curr]
    n_curr = int(curr_members.shape[0])
    curr_minus = np.asarray(
        [c for c in curr_members.tolist() if c != s], dtype=np.int32)

    ll_by = state.ll_by_cluster
    key_curr = _cluster_key(fi, fam.cluster_id, cid_curr)

    # Use ll_by cache for existing clusters (curr, other). Only NEW
    # cluster configs (curr_minus, singleton, other_plus) need scoring.
    ll_curr_cached = ll_by.get(key_curr)                  # existing cluster LL
    specs: 'list[tuple[int, np.ndarray, np.ndarray, int]]' = []
    slot: 'dict[str, int]' = {}

    def _add(label, cols, mb):
        specs.append((fi, cols, _cluster_classes_padded(fam, cols, mb),
                        mb))
        slot[label] = len(specs) - 1

    if ll_curr_cached is None:
        _add('curr', curr_members, state.m_bucket_for(n_curr))
    if n_curr > 1:
        _add('curr_minus', curr_minus, state.m_bucket_for(n_curr - 1))
        s_arr = np.asarray([s], dtype=np.int32)
        _add('singleton', s_arr, state.m_bucket_for(1))

    other_cids: 'list[int]' = []
    other_ll_cached: 'dict[int, float]' = {}
    for c_other, other_members in clusters.items():
        if c_other == cid_curr:
            continue
        m_other = int(other_members.shape[0])
        if m_other + 1 > state.max_cluster_size:
            continue
        key_o = _cluster_key(fi, fam.cluster_id, c_other)
        ll_o_cached = ll_by.get(key_o)
        if ll_o_cached is None:
            _add(('other', c_other), other_members,
                  state.m_bucket_for(m_other))
        else:
            other_ll_cached[c_other] = ll_o_cached
        cols_plus = np.sort(np.concatenate(
            [other_members, np.asarray([s], dtype=np.int32)]))
        _add(('other_plus', c_other), cols_plus,
              state.m_bucket_for(m_other + 1))
        other_cids.append(c_other)

    lls = _score_cluster_batch(state, specs) if specs else np.zeros(0)

    ll_curr = ll_curr_cached if ll_curr_cached is not None \
                else float(lls[slot['curr']])
    ll_by[key_curr] = ll_curr
    if n_curr > 1:
        ll_curr_minus_s = float(lls[slot['curr_minus']])
        ll_singleton = float(lls[slot['singleton']])
    else:
        ll_curr_minus_s = 0.0
        ll_singleton = 0.0

    # Compose candidate probs.
    cand_ids = [cid_curr]
    cand_log_p = [0.0]

    if n_curr > 1:
        lik_delta = -ll_curr + ll_curr_minus_s + ll_singleton
        prior_delta = (float(np.log(state.alpha_z))
                        - float(np.log(n_curr - 1)))
        cand_ids.append(-1)                                # new singleton
        cand_log_p.append(lik_delta + prior_delta)

    other_new_ll: 'dict[int, float]' = {}
    for c_other in other_cids:
        m_other = int(clusters[c_other].shape[0])
        ll_other = (other_ll_cached[c_other]
                     if c_other in other_ll_cached
                     else float(lls[slot[('other', c_other)]]))
        ll_other_plus = float(lls[slot[('other_plus', c_other)]])
        if n_curr > 1:
            lik_delta = (-ll_curr + ll_curr_minus_s
                          - ll_other + ll_other_plus)
            prior_delta = (-float(np.log(n_curr - 1))
                            + float(np.log(m_other)))
        else:
            lik_delta = -ll_curr - ll_other + ll_other_plus
            prior_delta = (-float(np.log(state.alpha_z))
                            + float(np.log(m_other)))
        other_new_ll[c_other] = ll_other_plus
        cand_ids.append(c_other)
        cand_log_p.append(lik_delta + prior_delta)

    log_p = np.asarray(cand_log_p, dtype=np.float64)
    log_p -= log_p.max()
    p = np.exp(log_p); p /= p.sum()
    idx = int(rng.choice(len(p), p=p))
    new_cid_raw = cand_ids[idx]

    if new_cid_raw == cid_curr:
        return                                              # no change

    if new_cid_raw == -1:
        new_cid = int(fam.cluster_id.max()) + 1
        # (B): shrink current, add fresh singleton.
        fam.cluster_id[s] = new_cid
        ll_by.pop(key_curr, None)                           # old full curr
        if n_curr > 1:
            key_shrunk = _cluster_key(fi, fam.cluster_id, cid_curr)
            ll_by[key_shrunk] = ll_curr_minus_s
        key_new = _cluster_key(fi, fam.cluster_id, new_cid)
        ll_by[key_new] = ll_singleton
    else:
        # (C): shrink current, grow other. MUST pop old-target key
        # BEFORE mutating cluster_id (key uses column tuple).
        key_target_pre = _cluster_key(fi, fam.cluster_id, new_cid_raw)
        fam.cluster_id[s] = new_cid_raw
        ll_by.pop(key_curr, None)
        ll_by.pop(key_target_pre, None)
        if n_curr > 1:
            key_shrunk = _cluster_key(fi, fam.cluster_id, cid_curr)
            ll_by[key_shrunk] = ll_curr_minus_s
        key_other_new = _cluster_key(fi, fam.cluster_id, new_cid_raw)
        ll_by[key_other_new] = other_new_ll[new_cid_raw]


def _score_affected_clusters(state: CorpusState,
                                    affected_classes: 'set[int]'
                                    ) -> 'tuple[list[tuple], float]':
    """Score all clusters (across all families) that contain at least
    one site whose classes value is in affected_classes. Returns
    (affected_keys, sum_ll)."""
    affected_keys = []
    total = 0.0
    for fi, fam in enumerate(state.families):
        which = np.isin(fam.classes, list(affected_classes))
        cids = np.unique(fam.cluster_id[which])
        for cid in cids.tolist():
            cid = int(cid)
            cols = np.where(fam.cluster_id == cid)[0].astype(np.int32)
            m = int(cols.shape[0])
            mb = state.m_bucket_for(m)
            cls_padded = _cluster_classes_padded(fam, cols, mb)
            ll = _score_cluster(state, fi, cols, cls_padded, mb)
            key = _cluster_key(fi, fam.cluster_id, cid)
            state.ll_by_cluster[key] = ll
            affected_keys.append(key)
            total += ll
    return affected_keys, total


def _affected_cluster_keys_for_classes(state: CorpusState,
                                              affected_classes: 'set[int]'
                                              ) -> 'list[tuple]':
    """List cache keys of clusters containing any site with a c_n in
    affected_classes (does NOT score them; uses the existing cache)."""
    out = []
    for fi, fam in enumerate(state.families):
        which = np.isin(fam.classes, list(affected_classes))
        cids = np.unique(fam.cluster_id[which])
        for cid in cids.tolist():
            out.append(_cluster_key(fi, fam.cluster_id, int(cid)))
    return out


def _score_affected_clusters_under_pi(state: CorpusState,
                                             affected_classes: 'set[int]',
                                             pi_field: np.ndarray,
                                             ) -> 'dict[tuple, float]':
    """Score affected clusters under a HYPOTHETICAL pi_field via ONE
    batched call under transient kernel tables. Returns {key: ll}.
    Does not mutate state."""
    tables = precompute_kernel_tables(
        state.bin_centers, state.rho, state.rho_chain, pi_field, state.S)
    specs = []
    keys = []
    for fi, fam in enumerate(state.families):
        which = np.isin(fam.classes, list(affected_classes))
        cids = np.unique(fam.cluster_id[which])
        for cid in cids.tolist():
            cid = int(cid)
            cols = np.where(fam.cluster_id == cid)[0].astype(np.int32)
            m = int(cols.shape[0])
            mb = state.m_bucket_for(m)
            cls_padded = _cluster_classes_padded(fam, cols, mb)
            specs.append((fi, cols, cls_padded, mb))
            keys.append(_cluster_key(fi, fam.cluster_id, cid))
    lls = _score_cluster_batch(state, specs, tables=tables)
    return {k: float(v) for k, v in zip(keys, lls)}


def _score_affected_clusters_under_overrides(
        state: CorpusState,
        affected_classes: 'set[int]',
        overrides: 'dict[tuple[int, int], np.ndarray]',
        ) -> 'dict[tuple, float]':
    """Score affected clusters under an arch perturbation given as
    per-(c, theta) pi-row overrides. Same result as
    `_score_affected_clusters_under_pi` with the corresponding full
    pi_field, but reuses base beta/W and recomputes only the perturbed
    P_sub slices (n_bins expm per override, vs K_c * L * n_bins for a full
    rebuild). Does not mutate state."""
    tables = state.kernel_tables_with_overrides(overrides)
    specs = []
    keys = []
    for fi, fam in enumerate(state.families):
        which = np.isin(fam.classes, list(affected_classes))
        cids = np.unique(fam.cluster_id[which])
        for cid in cids.tolist():
            cid = int(cid)
            cols = np.where(fam.cluster_id == cid)[0].astype(np.int32)
            m = int(cols.shape[0])
            mb = state.m_bucket_for(m)
            cls_padded = _cluster_classes_padded(fam, cols, mb)
            specs.append((fi, cols, cls_padded, mb))
            keys.append(_cluster_key(fi, fam.cluster_id, cid))
    lls = _score_cluster_batch(state, specs, tables=tables)
    return {k: float(v) for k, v in zip(keys, lls)}


def atomic_arch_swap(state: CorpusState, theta: int, c1: int, c2: int,
                          rng: np.random.Generator) -> bool:
    """MH pairwise swap of arch_assignment[{c1,c2}, theta]. Returns
    True on accept. Uses ll_by_cluster cache; only affected clusters
    (those containing class c1 or c2) are re-evaluated."""
    aa = state.arch_assignment
    k1 = int(aa[c1, theta]); k2 = int(aa[c2, theta])
    if k1 == k2:
        return False
    affected = _affected_cluster_keys_for_classes(state, {c1, c2})
    # Swapping (c1,theta)<->(c2,theta) makes pi_field[c1,theta] use
    # archetype k2 and pi_field[c2,theta] use k1.
    overrides = {(c1, theta): k2, (c2, theta): k1}
    if not affected:
        # No cluster touches these classes -- accept trivially and update aa.
        aa[c1, theta] = k2; aa[c2, theta] = k1
        state.apply_arch_slices([(c1, theta), (c2, theta)])
        return True
    ll_curr = sum(state.ll_by_cluster[k] for k in affected)
    ll_swap_by_key = _score_affected_clusters_under_overrides(
        state, {c1, c2}, overrides)
    ll_swap = sum(ll_swap_by_key.values())
    log_ratio = ll_swap - ll_curr
    if log_ratio >= 0.0 or rng.random() < np.exp(log_ratio):
        aa[c1, theta] = k2; aa[c2, theta] = k1
        state.apply_arch_slices([(c1, theta), (c2, theta)])
        for k, v in ll_swap_by_key.items():
            state.ll_by_cluster[k] = v
        return True
    return False


def atomic_arch_gswap(state: CorpusState, theta: int, A_c: int,
                           rng: np.random.Generator) -> 'int | None':
    """Gibbs-style swap for class A_c at theta. Returns B_c chosen
    (or None if null / no-op)."""
    aa = state.arch_assignment
    K_c = state.K_c
    k_A = int(aa[A_c, theta])
    baseline_keys = _affected_cluster_keys_for_classes(state, {A_c})
    log_L_base = sum(state.ll_by_cluster[k] for k in baseline_keys)

    # For each B_c != A_c: proposed swap-state LL over affected(A_c, B_c)
    cand_ll = np.full(K_c, log_L_base, dtype=np.float64)
    cand_new_scores: 'dict[int, dict[tuple, float]]' = {}
    for B in range(K_c):
        if B == A_c:
            continue
        k_B = int(aa[B, theta])
        if k_A == k_B:
            continue
        affected = _affected_cluster_keys_for_classes(state, {A_c, B})
        if not affected:
            continue
        # Swap (A_c,theta)<->(B,theta): A_c uses k_B, B uses k_A.
        overrides = {(A_c, theta): k_B, (B, theta): k_A}
        base_affected = sum(state.ll_by_cluster[k] for k in affected)
        new_by_key = _score_affected_clusters_under_overrides(
            state, {A_c, B}, overrides)
        delta = sum(new_by_key.values()) - base_affected
        cand_ll[B] = log_L_base + delta
        cand_new_scores[B] = new_by_key

    log_L = cand_ll.copy(); log_L -= log_L.max()
    p = np.exp(log_L); p /= p.sum()
    B_new = int(rng.choice(K_c, p=p))
    if B_new == A_c or B_new not in cand_new_scores:
        return None
    k_B = int(aa[B_new, theta])
    aa[A_c, theta] = k_B; aa[B_new, theta] = k_A
    state.apply_arch_slices([(A_c, theta), (B_new, theta)])
    for k, v in cand_new_scores[B_new].items():
        state.ll_by_cluster[k] = v
    return B_new


def atomic_arch_gibbs(state: CorpusState, theta: int, c: int,
                           rho_arch: np.ndarray,
                           rng: np.random.Generator) -> int:
    """K_a-way Gibbs on arch_assignment[c, theta] (single-entry
    change; ergodic on the arch multiset). Returns new k."""
    aa = state.arch_assignment
    K_a = state.K_a
    k_curr = int(aa[c, theta])
    affected = _affected_cluster_keys_for_classes(state, {c})
    if not affected:
        # No sites in class c: pure prior draw.
        p = rho_arch / rho_arch.sum()
        k_new = int(rng.choice(K_a, p=p))
        aa[c, theta] = k_new
        if k_new != k_curr:
            state.apply_arch_slices([(c, theta)])
        return k_new
    ll_base_affected = sum(state.ll_by_cluster[k] for k in affected)
    cand_ll = np.zeros(K_a, dtype=np.float64)
    cand_new_scores: 'dict[int, dict[tuple, float]]' = {}
    for k in range(K_a):
        if k == k_curr:
            cand_ll[k] = ll_base_affected
            continue
        new_by_key = _score_affected_clusters_under_overrides(
            state, {c}, {(c, theta): k})
        cand_ll[k] = sum(new_by_key.values())
        cand_new_scores[k] = new_by_key
    log_L = cand_ll + np.log(np.maximum(rho_arch, 1e-300))
    log_L -= log_L.max()
    p = np.exp(log_L); p /= p.sum()
    k_new = int(rng.choice(K_a, p=p))
    if k_new == k_curr:
        return k_curr
    aa[c, theta] = k_new
    state.apply_arch_slices([(c, theta)])
    for k, v in cand_new_scores[k_new].items():
        state.ll_by_cluster[k] = v
    return k_new


# --------------------------------------------------------- Outer loop


def _format_size_histogram(hist: np.ndarray, top: int = 6) -> str:
    """One-line summary: 'sizes 1:N1 2:N2 ... 6:N6 (max=K)'."""
    if hist.size == 0:
        return "sizes (none)"
    parts = [f"{k+1}:{int(hist[k])}"
              for k in range(min(top, hist.size)) if hist[k] > 0]
    tail = hist[top:] if hist.size > top else np.zeros(0, dtype=hist.dtype)
    if tail.sum() > 0:
        parts.append(f">{top}:{int(tail.sum())}")
    max_size = int(hist.size)
    return "sizes " + " ".join(parts) + f" (max={max_size})"


def run_atomic_training(
    state: CorpusState,
    rng: np.random.Generator,
    n_atomic_moves: int,
    log_every: int = 1000,
    recompute_every: int = 10,      # multiples of log_every
    move_weights: 'dict[str, int] | None' = None,
    fix_theta0: bool = True,
    log_stream=None,
    checkpoint_dir: 'str | Path | None' = None,
    checkpoint_every: int = 20,     # multiples of log_every (~5% of run)
    checkpoint_keep_last: int = 3,  # rolling window of step_*.npz (0 = keep all)
    start_step: int = 0,
) -> None:
    """Run n_atomic_moves atomic Gibbs moves.

    Move types:
      'cn'         : resample classes[s] at one site
      'z'          : Neal-3 CRP-Gibbs on cluster_id[s] at one site
      'arch_swap'  : MH pair swap of arch_assignment
      'arch_gswap' : Gibbs-style swap over K_c partners+null
      'arch_gibbs' : K_a Gibbs on one arch_assignment entry

    move_weights: {name: count} defaults uniform 1 each. Cycle is the
    Fisher-Yates permutation of the count-weighted list."""
    if move_weights is None:
        move_weights = {'cn': 1, 'z': 1, 'arch_swap': 1,
                          'arch_gswap': 1, 'arch_gibbs': 1}
    move_seq = []
    for name, c in move_weights.items():
        move_seq.extend([name] * int(c))
    move_cycle = PermCycle(move_seq, rng)
    theta_cycle = PermCycle(
        list(range(1, state.L_field)) if fix_theta0
        else list(range(state.L_field)), rng)
    class_cycle = PermCycle(list(range(state.K_c)), rng)
    K_c = state.K_c
    pair_cycle = PermCycle(
        [(c1, c2) for c1 in range(K_c) for c2 in range(c1 + 1, K_c)],
        rng)
    family_cycle = PermCycle(list(range(len(state.families))), rng)
    # Per-family site cycles (site indices are family-local).
    site_cycles = [PermCycle(list(range(f.L)), rng) for f in state.families]

    rho_arch = np.full(state.K_a, 1.0 / state.K_a)

    if not state.ll_by_cluster:
        compute_all_cluster_lls(state)
    t0 = time.time()
    stats = {'cn': 0, 'z': 0, 'arch_swap': 0, 'arch_swap_acc': 0,
              'arch_gswap': 0, 'arch_gswap_moved': 0,
              'arch_gibbs': 0, 'arch_gibbs_changed': 0}

    def _log(step):
        ll = corpus_ll(state)
        elapsed = time.time() - t0
        n_clusters = sum(
            len(set(f.cluster_id.tolist())) for f in state.families)
        hist = cluster_size_histogram(state)
        # Resource probes: rss should stay roughly flat; jit variant count
        # should PLATEAU (a monotonic climb is the executable-cache leak).
        rss = _rss_mb()
        live = _jax_live_arrays()
        n_jit, n_var = jit_cache_stats()
        dev_mb = _jax_device_mb()
        res = (f"  rss={rss:.0f}MB  live={live}  jit={n_jit}/{n_var}")
        if dev_mb == dev_mb:                                # not NaN
            res += f"  dev={dev_mb:.0f}MB"
        line = (f"# step={step:6d}  ll={ll:+.4f}  n_clusters={n_clusters}"
                f"  elapsed={elapsed:.1f}s  "
                f"cn={stats['cn']}  z={stats['z']}  "
                f"swap={stats['arch_swap_acc']}/{stats['arch_swap']}  "
                f"gswap={stats['arch_gswap_moved']}/{stats['arch_gswap']}  "
                f"gibbs={stats['arch_gibbs_changed']}/{stats['arch_gibbs']}  "
                f"| {_format_size_histogram(hist)}{res}")
        print(line, flush=True)
        if log_stream is not None:
            log_stream.write(line + "\n"); log_stream.flush()

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _log(start_step)
    log_count = 0
    for step in range(start_step + 1, start_step + n_atomic_moves + 1):
        move = move_cycle.next()
        if move == 'cn':
            fi = family_cycle.next()
            s = site_cycles[fi].next()
            atomic_cn_resample(state, fi, s, rng)
            stats['cn'] += 1
        elif move == 'z':
            fi = family_cycle.next()
            s = site_cycles[fi].next()
            atomic_z_gibbs(state, fi, s, rng)
            stats['z'] += 1
        elif move == 'arch_swap':
            theta = theta_cycle.next()
            c1, c2 = pair_cycle.next()
            acc = atomic_arch_swap(state, theta, c1, c2, rng)
            stats['arch_swap'] += 1
            if acc:
                stats['arch_swap_acc'] += 1
        elif move == 'arch_gswap':
            theta = theta_cycle.next()
            A_c = class_cycle.next()
            r = atomic_arch_gswap(state, theta, A_c, rng)
            stats['arch_gswap'] += 1
            if r is not None:
                stats['arch_gswap_moved'] += 1
        elif move == 'arch_gibbs':
            theta = theta_cycle.next()
            c = class_cycle.next()
            k_curr = int(state.arch_assignment[c, theta])
            k_new = atomic_arch_gibbs(state, theta, c, rho_arch, rng)
            stats['arch_gibbs'] += 1
            if k_new != k_curr:
                stats['arch_gibbs_changed'] += 1
        if step % log_every == 0:
            log_count += 1
            if recompute_every > 0 and log_count % recompute_every == 0:
                compute_all_cluster_lls(state)              # drift reset
            _log(step)
            if (checkpoint_dir is not None
                and log_count % checkpoint_every == 0):
                latest = checkpoint_dir / "latest.npz"
                stepped = checkpoint_dir / f"step_{step:07d}.npz"
                save_checkpoint(state, stepped, step, rng)
                # Atomically replace 'latest' via rename.
                save_checkpoint(state, latest, step, rng)
                # Rolling window: drop all but the newest N stepped
                # checkpoints so a long run does not fill the disk (each
                # holds only cluster_id/classes/arch, but they accumulate).
                if checkpoint_keep_last > 0:
                    stepped_all = sorted(checkpoint_dir.glob("step_*.npz"))
                    for old in stepped_all[:-checkpoint_keep_last]:
                        try:
                            old.unlink()
                        except OSError:
                            pass

    compute_all_cluster_lls(state)
    _log(start_step + n_atomic_moves)
    if checkpoint_dir is not None:
        save_checkpoint(state, checkpoint_dir / "final.npz",
                          start_step + n_atomic_moves, rng)
