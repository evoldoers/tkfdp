"""BAliBASE-specific PDB contact extraction.

BAliBASE 3 sequence names like ``1aab_`` or ``1j46_A`` are direct PDB
references: the first 4 characters are the PDB ID, the 5th character is
the chain ID (``_`` denoting the blank-chain convention).

Adapter pipeline per BAliBASE sequence:
  1. Parse PDB ID + chain from name (e.g. ``1aab_`` -> id=``1aab``, chain
     ``_``-or-``A`` (try blank, then ``A``)).
  2. Fetch the PDB structure (cached at ``data/pdb_cache/<id>.pdb`` via
     the same ``_fetch_pdb`` helper used by ``pdb_contacts.py``).
  3. Extract Cα coordinates for the chain's standard amino-acid
     residues.
  4. Pairwise-align the BAliBASE sequence to the chain sequence
     (Bio.Align.PairwiseAligner, BLOSUM62) to map BAliBASE residue index
     -> PDB residue Cα.
  5. Return the contact pairs (i, j) of BAliBASE residue indices whose
     Cα atoms are within ``threshold`` Å (default 8.0).
"""

from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PDB_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "pdb_cache"
PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class BaliPDBMap:
    seq_name: str
    pdb_id: str
    chain_id: str
    n_pdb_residues: int
    n_seq_residues: int
    n_mapped: int
    # mapping: bali_residue_idx (0-based in ungapped BAliBASE seq) -> ca_xyz
    bali_idx_to_ca: dict[int, np.ndarray]


def parse_balibase_pdb_name(seq_name: str) -> tuple[str, str] | None:
    """Parse a BAliBASE sequence name into (pdb_id_lower, chain_candidate).

    BAliBASE 3 PDB-derived names take three forms:
      - ``XXXX`` (4 chars): PDB ID XXXX, blank chain (chain id ' ' in PDB).
      - ``XXXX_`` (5 chars): PDB ID XXXX, blank chain.
      - ``XXXX_C`` (6 chars): PDB ID XXXX, chain C.
    Returns None if the name doesn't look PDB-derived (e.g. SwissProt
    names like ``PROA_BACAA``).
    """
    m6 = re.fullmatch(r"([0-9][A-Za-z0-9]{3})_([A-Za-z0-9])", seq_name)
    if m6 is not None:
        return m6.group(1).lower(), m6.group(2)
    m5 = re.fullmatch(r"([0-9][A-Za-z0-9]{3})_", seq_name)
    if m5 is not None:
        return m5.group(1).lower(), '_'
    m4 = re.fullmatch(r"([0-9][A-Za-z0-9]{3})", seq_name)
    if m4 is not None:
        return m4.group(1).lower(), '_'
    return None


def _fetch_pdb(pdb_id: str) -> Path:
    out = PDB_CACHE_DIR / f"{pdb_id.lower()}.pdb"
    if out.exists() and out.stat().st_size > 1000:
        return out
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except Exception as e:
        raise RuntimeError(f"PDB fetch failed for {pdb_id}: {e}") from e
    out.write_bytes(data)
    return out


def _extract_chain_residues(pdb_path: Path):
    """Return {chain_id: [(resseq, one_letter, ca_xyz), ...]} for all chains
    that have at least one standard-AA residue with a Cα atom."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    out: dict[str, list] = {}
    for model in struct:
        for chain in model:
            residues = []
            for res in chain:
                hetflag, resseq, icode = res.id
                if hetflag != ' ':
                    continue
                if res.resname not in THREE_TO_ONE:
                    continue
                if 'CA' not in res:
                    continue
                ca = np.asarray(res['CA'].coord, dtype=np.float64)
                residues.append((int(resseq), THREE_TO_ONE[res.resname], ca))
            if residues:
                out[chain.id] = residues
        break  # only first model
    return out


def _align_seq_to_chain(query_seq: str, chain_seq: str):
    """Bio.Align global alignment, return (aligned_blocks, score)."""
    from Bio.Align import PairwiseAligner, substitution_matrices
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    score = aligner.score(query_seq, chain_seq)
    aln = aligner.align(query_seq, chain_seq)[0]
    return aln, float(score)


def map_balibase_seq_to_pdb(seq_name: str, ungapped_seq: str,
                              ) -> Optional[BaliPDBMap]:
    """Map an ungapped BAliBASE sequence to its source PDB chain.

    Args:
        seq_name: BAliBASE sequence name (e.g. ``1aab_`` or ``1j46_A``).
        ungapped_seq: the residue sequence as a string of one-letter codes
            (uppercase, no gaps, no insertions).

    Returns BaliPDBMap or None if name not PDB-derived / fetch / parse fails.
    """
    parsed = parse_balibase_pdb_name(seq_name)
    if parsed is None:
        return None
    pdb_id, chain_token = parsed
    try:
        pdb_path = _fetch_pdb(pdb_id)
    except Exception:
        return None
    try:
        chains = _extract_chain_residues(pdb_path)
    except Exception:
        return None
    if not chains:
        return None

    # Map the chain token to actual PDB chain id. Conventions:
    #   - '_' typically means the blank-chain (chain id ' ' in PDB).
    #     biopython exposes this as chain.id == ' '.
    # If the requested chain isn't present, fall back to the longest chain.
    candidate_chains = []
    if chain_token == '_':
        # try blank, then 'A' (older PDB files sometimes have 'A' for what
        # used to be blank in renamed files)
        for c in (' ', 'A', '_'):
            if c in chains:
                candidate_chains.append(c)
    else:
        for c in (chain_token, chain_token.upper(), chain_token.lower()):
            if c in chains and c not in candidate_chains:
                candidate_chains.append(c)
    # Last-resort fallback: longest chain.
    if not candidate_chains:
        candidate_chains = [max(chains.keys(), key=lambda k: len(chains[k]))]

    # Pick the candidate with best alignment score to ungapped_seq.
    best = None  # (score, chain_id, aln, residues)
    for cid in candidate_chains:
        residues = chains[cid]
        chain_seq = ''.join(r[1] for r in residues)
        if not chain_seq or not ungapped_seq:
            continue
        try:
            aln, sc = _align_seq_to_chain(ungapped_seq, chain_seq)
        except Exception:
            continue
        if best is None or sc > best[0]:
            best = (sc, cid, aln, residues)
    if best is None:
        return None
    _sc, chain_id, aln, residues = best

    # Walk aln.aligned to build query_idx -> chain_idx mapping.
    aligned = aln.aligned   # ((q_blocks,), (t_blocks,))
    q_to_t: dict[int, int] = {}
    for (q0, q1), (t0, t1) in zip(aligned[0], aligned[1]):
        for k in range(int(q1) - int(q0)):
            q_to_t[int(q0) + k] = int(t0) + k

    bali_idx_to_ca: dict[int, np.ndarray] = {}
    for q_idx, t_idx in q_to_t.items():
        if 0 <= t_idx < len(residues):
            # Optionally: only map if amino-acid types match (or both X).
            # For coverage, we accept all aligned residues here.
            bali_idx_to_ca[q_idx] = residues[t_idx][2]

    return BaliPDBMap(
        seq_name=seq_name, pdb_id=pdb_id, chain_id=chain_id,
        n_pdb_residues=len(residues),
        n_seq_residues=len(ungapped_seq),
        n_mapped=len(bali_idx_to_ca),
        bali_idx_to_ca=bali_idx_to_ca,
    )


def contacts_for_seq(seq_name: str, ungapped_seq: str,
                      threshold: float = 8.0,
                      min_separation: int = 4) -> Optional[set]:
    """Compute the Cα contact set as {(i, j)} of BAliBASE residue indices
    (0-based, in the ungapped sequence) with i < j, |i - j| > min_separation,
    and Cα distance <= threshold (Å).

    Returns None if mapping unavailable.
    """
    mp = map_balibase_seq_to_pdb(seq_name, ungapped_seq)
    if mp is None or len(mp.bali_idx_to_ca) < 2:
        return None
    idxs = sorted(mp.bali_idx_to_ca.keys())
    contacts = set()
    for ai in range(len(idxs)):
        ci = idxs[ai]
        ca_i = mp.bali_idx_to_ca[ci]
        for aj in range(ai + 1, len(idxs)):
            cj = idxs[aj]
            if cj - ci <= min_separation:
                continue
            d = float(np.linalg.norm(ca_i - mp.bali_idx_to_ca[cj]))
            if d <= threshold:
                contacts.add((ci, cj))
    return contacts
