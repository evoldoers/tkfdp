"""Pfam-specific data prep: per-family cherries (sequence pairs at known
distances) gathered into a tensor convenient for partition MCMC + SGD on H.

Per-family layout:
    cherries: list of dicts with
        - tau: float, cherry distance (LG08 sub/site)
        - aa_a: (L,) int8, gap=20
        - aa_b: (L,) int8, gap=20
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bio import (
    AA_TO_IDX,
    GAP_INDEX,
    GAP_CHARS,
    FamilyData,
    has_family,
    load_family,
    load_split,
)
from .cherries import extract_cherries


@dataclass
class FamilyCherries:
    family: str
    L: int                              # alignment length
    n_cherries: int
    tau: np.ndarray                     # (C,) float, branch distances
    aa_a: np.ndarray                    # (C, L) int8
    aa_b: np.ndarray                    # (C, L) int8

    def both_aa_mask(self) -> np.ndarray:
        """Boolean mask (C, L): True where both ends are non-gap amino acids."""
        return (self.aa_a < 20) & (self.aa_b < 20)


def encode_seq(seq: str) -> np.ndarray:
    out = np.full(len(seq), GAP_INDEX, dtype=np.int8)
    for j, c in enumerate(seq.upper()):
        if c in GAP_CHARS:
            out[j] = GAP_INDEX
        else:
            out[j] = AA_TO_IDX.get(c, GAP_INDEX)
    return out


def family_cherries(fd: FamilyData, max_cherries: int | None = None) -> FamilyCherries:
    """Extract cherries from a family's tree and align them to MSA columns.

    The MSA seq names must match cherry leaf names. Where a cherry's seq names
    don't appear in the MSA, the cherry is skipped (FastTree may have run on
    a subset).
    """
    name_to_row = {n: i for i, n in enumerate(fd.names)}
    cherries = extract_cherries(fd.tree)
    aa_a_list = []
    aa_b_list = []
    tau_list = []
    for a, b, tau in cherries:
        if a not in name_to_row or b not in name_to_row:
            continue
        aa_a_list.append(fd.msa[name_to_row[a]])
        aa_b_list.append(fd.msa[name_to_row[b]])
        tau_list.append(tau)
        if max_cherries is not None and len(aa_a_list) >= max_cherries:
            break

    if not aa_a_list:
        return FamilyCherries(
            family=fd.family,
            L=fd.msa.shape[1] if fd.msa.size else 0,
            n_cherries=0,
            tau=np.zeros(0),
            aa_a=np.zeros((0, fd.msa.shape[1]), dtype=np.int8),
            aa_b=np.zeros((0, fd.msa.shape[1]), dtype=np.int8),
        )

    return FamilyCherries(
        family=fd.family,
        L=fd.msa.shape[1],
        n_cherries=len(aa_a_list),
        tau=np.asarray(tau_list, dtype=np.float64),
        aa_a=np.stack(aa_a_list).astype(np.int8),
        aa_b=np.stack(aa_b_list).astype(np.int8),
    )


def filter_columns(fc: FamilyCherries, min_aa_fraction: float = 0.5,
                   return_keep_mask: bool = False):
    """Drop alignment columns where < min_aa_fraction of cherries have both ends
    as amino acids. Keeps the cherry count fixed. If `return_keep_mask=True`,
    also returns the boolean (L,) mask of kept columns (in original indexing).
    """
    if fc.n_cherries == 0:
        if return_keep_mask:
            return fc, np.ones(fc.L, dtype=bool)
        return fc
    mask_col_keep = fc.both_aa_mask().mean(axis=0) >= min_aa_fraction
    fc_filt = FamilyCherries(
        family=fc.family,
        L=int(mask_col_keep.sum()),
        n_cherries=fc.n_cherries,
        tau=fc.tau,
        aa_a=fc.aa_a[:, mask_col_keep],
        aa_b=fc.aa_b[:, mask_col_keep],
    )
    if return_keep_mask:
        return fc_filt, mask_col_keep
    return fc_filt


def families_from_split(split_name: str = "train",
                        n_families: int | None = None,
                        min_cherries: int = 5,
                        min_columns: int = 30,
                        max_columns: int = 200,
                        min_aa_fraction: float = 0.5,
                        seed: int = 0) -> list[FamilyCherries]:
    """Load `n_families` families from the named split, each with a minimum
    number of cherries and columns surviving the gap filter. Useful as the
    corpus for the Pfam unsupervised fit.
    """
    sp = load_split()
    if split_name not in sp:
        raise ValueError(f"Unknown split {split_name!r}; have {list(sp.keys())}")
    families = list(sp[split_name])
    rng = np.random.default_rng(seed)
    rng.shuffle(families)

    out: list[FamilyCherries] = []
    for fam in families:
        if not has_family(fam):
            continue
        try:
            fd = load_family(fam)
        except Exception:
            continue
        fc = family_cherries(fd)
        fc = filter_columns(fc, min_aa_fraction=min_aa_fraction)
        if fc.n_cherries < min_cherries:
            continue
        if fc.L < min_columns or fc.L > max_columns:
            continue
        out.append(fc)
        if n_families is not None and len(out) >= n_families:
            break
    return out


def families_from_list(family_ids: list[str],
                       min_cherries: int = 5,
                       min_aa_fraction: float = 0.5,
                       return_keep_masks: bool = False):
    """Like `families_from_split` but loads an explicit list (e.g. for
    matched-corpus comparisons across training conditions). No min/max
    column filter and no shuffle. If `return_keep_masks=True`, also returns
    a parallel list of (L_orig,) bool masks indicating which original
    columns survived the gap filter (needed to remap PDB contacts).
    """
    out: list[FamilyCherries] = []
    masks: list[np.ndarray] = []
    for fam in family_ids:
        if not has_family(fam):
            print(f"  WARNING: {fam} not found, skipping")
            continue
        fd = load_family(fam)
        fc_full = family_cherries(fd)
        fc, keep_mask = filter_columns(fc_full, min_aa_fraction=min_aa_fraction,
                                        return_keep_mask=True)
        if fc.n_cherries < min_cherries:
            print(f"  WARNING: {fam} has only {fc.n_cherries} cherries (< {min_cherries}), skipping")
            continue
        out.append(fc)
        masks.append(keep_mask)
    if return_keep_masks:
        return out, masks
    return out
