"""PDB-derived contact maps for Pfam families: fetch the SCOP-mapped PDB
structure, align its chain sequence to one of the MSA sequences, and
return the set of MSA-column pairs whose Cα atoms are within a distance
threshold (default 8 Å). Used to seed the partition state for
exp2_pfam.py training.

Pipeline per family:
  1. Read SCOP cross-ref from `~/bio-datasets/data/pfam/seed/<fam>.sto`.
  2. Fetch the PDB to a local cache (`data/pdb_cache/<id>.pdb`).
  3. Parse the structure with biopython; extract the longest standard-AA
     chain.
  4. Pairwise-align (Bio.Align.PairwiseAligner) the chain sequence
     against the *un-gapped* sequence of every MSA row, pick the
     highest-scoring row.
  5. Walk the alignment. Each MSA column → at most one PDB residue Cα
     coordinate.
  6. For every (i, j) pair of mapped columns with d_Cα < threshold,
     record the pair.
  7. Greedy maximum matching: each column belongs to at most one pair
     in the seeded partition. Sort pairs by ascending distance, take
     pairs one at a time, skip any whose endpoints are already taken.

Returns the list of (column_i, column_j) integer pairs.

This is best-effort — Pfam columns are domain-anchored, the SCOP ref is
representative not exhaustive, and MSA columns can be insertions that
have no PDB partner. We expect ~5–30 contact pairs per family.
"""

from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bio import GAP_CHARS, PFAM_SEED_DIR

PDB_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "pdb_cache"
PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class PDBContactInfo:
    family: str
    pdb_id: str
    chain_id: str
    msa_row: str         # the MSA sequence used as alignment proxy
    n_pdb_residues: int
    n_mapped_columns: int
    contact_pairs: list[tuple[int, int]]    # (col_i, col_j), 0-based, i < j
                                            # -- post-greedy-matching: each
                                            # column at most once. Use these
                                            # when seeding a fixed size-2
                                            # partition (the "PDB-anchor /
                                            # fix" mode).
    contact_distances: list[float]          # parallel to contact_pairs.
    # Pre-greedy-matching candidate set: every (i, j) with d(Cα_i, Cα_j) <=
    # distance_threshold AND j - i > min_separation. Each column may appear
    # in multiple pairs. Use these as the SUPPORT of an allowed-pair set
    # when restricting the MCMC sampler ("PDB-restrict" mode); the sampler
    # then explores valid size-{1, 2} partitions whose pairs are a subset
    # of this candidate set.
    candidate_pairs: list[tuple[int, int]] = None      # type: ignore[assignment]
    candidate_distances: list[float] = None             # type: ignore[assignment]


def _scop_id_for_family(family: str) -> str | None:
    sto = PFAM_SEED_DIR / f"{family}.sto"
    if not sto.exists():
        return None
    with open(sto) as f:
        for line in f:
            if line.startswith("#=GF DR"):
                m = re.search(r"SCOP;\s+(\S+)", line)
                if m:
                    cand = m.group(1).strip(';')
                    if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", cand):
                        return cand.lower()
    return None


def _fetch_pdb(pdb_id: str) -> Path:
    out = PDB_CACHE_DIR / f"{pdb_id.lower()}.pdb"
    if out.exists() and out.stat().st_size > 1000:
        return out
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"  fetching {url} ...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    out.write_bytes(data)
    return out


def _extract_chains(pdb_path: Path):
    """Returns list of (chain_id, residues) where residues is a list of
    (resseq_int, ' '|'A'|..., resname, ca_xyz_or_None)."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    out = []
    for model in struct:
        for chain in model:
            residues = []
            for res in chain:
                hetflag, resseq, icode = res.id
                if hetflag != ' ':
                    continue
                if res.resname not in THREE_TO_ONE:
                    continue
                ca = res['CA'].coord if 'CA' in res else None
                if ca is None:
                    continue
                residues.append((int(resseq), THREE_TO_ONE[res.resname],
                                 np.asarray(ca, dtype=np.float64)))
            if residues:
                out.append((chain.id, residues))
        break
    return out


def _aa_align(query: str, target: str):
    """Return a Bio.Align.PairwiseAlignments alignment for query vs target."""
    from Bio.Align import PairwiseAligner, substitution_matrices
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    return aligner.align(query, target)[0], aligner.score(query, target)


def _build_msa_to_pdb_map(msa_seq_with_gaps: str, ungapped_msa_seq: str,
                          chain_seq: str, chain_residues):
    """Given the chosen MSA row's gapped sequence, its un-gapped form, the
    PDB chain's amino-acid sequence, and the chain's residue list (with
    Cα coords), return a dict { msa_column_index : Cα_coord }.

    Two-step mapping:
      MSA column -> ungapped-MSA position (if not a gap)
      ungapped-MSA position -> PDB residue index (via pairwise alignment)
    """
    aln, _score = _aa_align(ungapped_msa_seq, chain_seq)
    aln_str = str(aln)
    parts = aln_str.split('\n')
    # PairwiseAligner formats lines as (target, marks, query) — but ordering
    # depends on the version. Walk via aln.aligned which is (path) tuples.
    aligned = aln.aligned   # ((q_blocks,), (t_blocks,))
    # For each (q_start, q_end) — (t_start, t_end) pair, residues in aligned
    q_to_t: dict[int, int] = {}
    for (q0, q1), (t0, t1) in zip(aligned[0], aligned[1]):
        for k in range(q1 - q0):
            q_to_t[q0 + k] = t0 + k

    # MSA column -> ungapped MSA index (only for non-gap positions)
    col_to_ungap: dict[int, int] = {}
    pos = 0
    for col, c in enumerate(msa_seq_with_gaps):
        if c.upper() in {ch for ch in 'ACDEFGHIKLMNPQRSTVWY'}:
            col_to_ungap[col] = pos
            pos += 1

    # Compose
    col_to_ca: dict[int, np.ndarray] = {}
    for col, q_idx in col_to_ungap.items():
        if q_idx in q_to_t:
            t_idx = q_to_t[q_idx]
            if t_idx < len(chain_residues):
                col_to_ca[col] = chain_residues[t_idx][2]
    return col_to_ca


def _greedy_max_matching(pairs_with_dists: list[tuple[int, int, float]]) -> tuple[list[tuple[int, int]], list[float]]:
    """Sort pairs by distance ascending and take pairs greedily, skipping
    those that share a column with one already chosen. Yields a valid
    size-2 partition (each column used at most once)."""
    used = set()
    out_pairs = []; out_dists = []
    for i, j, d in sorted(pairs_with_dists, key=lambda x: x[2]):
        if i in used or j in used:
            continue
        used.add(i); used.add(j)
        out_pairs.append((i, j)); out_dists.append(d)
    return out_pairs, out_dists


def pdb_contacts_for_family(family: str, msa_seqs: dict[str, str],
                             distance_threshold: float = 8.0,
                             min_separation: int = 4) -> PDBContactInfo | None:
    """Compute the seeded contact set for one family.

    msa_seqs: {seq_name: aligned_seq_with_gaps} — the seed MSA sequences.
    distance_threshold: Cα-Cα cutoff in Å.
    min_separation: skip pairs |i - j| <= min_separation (local helix
                    contacts are always there; we want non-local).

    Returns None if no SCOP ref or no usable contacts.
    """
    pdb_id = _scop_id_for_family(family)
    if pdb_id is None:
        return None

    try:
        pdb_path = _fetch_pdb(pdb_id)
    except Exception as e:
        print(f"  fetch failed for {family} ({pdb_id}): {e}")
        return None

    try:
        chains = _extract_chains(pdb_path)
    except Exception as e:
        print(f"  parse failed for {family} ({pdb_id}): {e}")
        return None
    if not chains:
        return None
    chains.sort(key=lambda c: -len(c[1]))
    chain_id, chain_residues = chains[0]
    chain_seq = ''.join(r[1] for r in chain_residues)

    # Pick MSA row with best alignment to the chain sequence
    best_score = -1e30; best_name = None; best_ungap = None
    for name, gapped in msa_seqs.items():
        ungapped = ''.join(c for c in gapped.upper() if c not in GAP_CHARS and c.isalpha())
        ungapped = ''.join(c for c in ungapped if c in 'ACDEFGHIKLMNPQRSTVWY')
        if len(ungapped) < 0.1 * len(chain_seq):
            continue
        try:
            from Bio.Align import PairwiseAligner, substitution_matrices
            aligner = PairwiseAligner()
            aligner.mode = 'global'
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
            aligner.open_gap_score = -10.0
            aligner.extend_gap_score = -0.5
            sc = float(aligner.score(ungapped, chain_seq))
        except Exception:
            sc = -1e30
        if sc > best_score:
            best_score = sc; best_name = name; best_ungap = ungapped

    if best_name is None:
        return None

    col_to_ca = _build_msa_to_pdb_map(msa_seqs[best_name], best_ungap, chain_seq, chain_residues)

    cols = sorted(col_to_ca.keys())
    candidates = []
    for ai in range(len(cols)):
        for aj in range(ai + 1, len(cols)):
            ci, cj = cols[ai], cols[aj]
            if cj - ci <= min_separation:
                continue
            d = float(np.linalg.norm(col_to_ca[ci] - col_to_ca[cj]))
            if d <= distance_threshold:
                candidates.append((ci, cj, d))

    pairs, dists = _greedy_max_matching(candidates)
    cand_pairs = [(i, j) for i, j, _ in candidates]
    cand_dists = [d for _, _, d in candidates]
    return PDBContactInfo(
        family=family, pdb_id=pdb_id, chain_id=chain_id,
        msa_row=best_name,
        n_pdb_residues=len(chain_residues),
        n_mapped_columns=len(col_to_ca),
        contact_pairs=pairs, contact_distances=dists,
        candidate_pairs=cand_pairs, candidate_distances=cand_dists,
    )
