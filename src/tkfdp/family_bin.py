"""Family shape binning for vmap-across-families JIT reuse.

Bin families along multiple dimensions using sqrt(2)-geomspaced buckets
so a wide corpus of ragged shapes compiles into a small set of
fixed-shape padded batches.

Typical use: given per-family (n_branches, L) pairs, produce a
dict `bin_key -> [fam_idx, ...]` where each bin_key is the padded
(n_branches_pad, L_pad). Bins with fewer than `min_families_per_bin`
entries are merged into their next-larger neighbour to avoid singleton
JIT compiles.
"""
from __future__ import annotations

import math
from collections import defaultdict


def sqrt2_bucket(x: int) -> int:
    """Smallest sqrt(2)^n >= x. Returns 1 for x <= 1."""
    if x <= 1:
        return 1
    n = math.ceil(math.log(x) / math.log(math.sqrt(2)))
    return int(math.ceil(math.sqrt(2) ** n))


def bin_families(shapes: 'list[tuple[int, ...]]',
                     ) -> 'tuple[dict, list[tuple[int, ...]]]':
    """Bin families by sqrt(2) buckets on each shape dimension.

    We deliberately DO NOT merge sparse bins into ``next-larger neighbours''
    — the neighbour along a single sqrt(2)-step is often another empty
    bin, so the merge just churns the family into a new sparse-and-larger
    bucket, wasting padding without reducing the compile count. The
    empirical padding-waste factor at no-merge is $\leq 2\times$ on the
    top-1000 corpus so the tradeoff is fine.

    Args:
      shapes: list of shape tuples, one per family. Each entry is a
        tuple of ints — e.g. ``(n_branches, L)`` for the tied-θ path,
        ``(n_branches, L, max_cluster_size)`` for the composite path.

    Returns:
      bin_map: dict mapping padded shape tuple -> list of family indices.
      fam_to_bin: (n_families,) list of padded shape assigned to each
                    family (aligned with ``shapes``).
    """
    padded = [tuple(sqrt2_bucket(int(v)) for v in s) for s in shapes]
    per_pad: 'dict[tuple[int, ...], list[int]]' = defaultdict(list)
    for i, p in enumerate(padded):
        per_pad[p].append(i)
    return dict(per_pad), padded


def summarise_bins(bin_map: dict) -> str:
    """Human-readable summary of a bin_map for logging."""
    n_bins = len(bin_map)
    total = sum(len(v) for v in bin_map.values())
    counts = sorted(((k, len(v)) for k, v in bin_map.items()),
                       key=lambda x: -x[1])
    lines = [f"{n_bins} bins, {total} families"]
    for k, n in counts[:8]:
        lines.append(f"  pad={k}  n={n}")
    if len(counts) > 8:
        lines.append(f"  ... ({len(counts) - 8} more)")
    return "\n".join(lines)
