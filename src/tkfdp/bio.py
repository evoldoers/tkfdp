"""Pfam loader: resolve paths via ~/bio-datasets, parse Stockholm + Newick,
load the clan-resistant train/val/test split.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lg08 import ALPHA_ORDER

BIO_DATASETS_HOME = Path(os.environ.get("BIO_DATASETS_HOME", Path.home() / "bio-datasets"))

PFAM_SEED_DIR = BIO_DATASETS_HOME / "data" / "pfam" / "seed"
PFAM_TREE_DIR = BIO_DATASETS_HOME / "data" / "pfam" / "trees"
PFAM_SPLIT_FILE = BIO_DATASETS_HOME / "fetch" / "pfam" / "splits" / "811-clan-resistant.json"

GAP_INDEX = 20  # 20 = gap; AA indices 0..19 follow ALPHA_ORDER
WILDCARD_INDEX = 20  # treat unknown residues as gap
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHA_ORDER)}
GAP_CHARS = {"-", "."}


def load_split() -> dict[str, list[str]]:
    with open(PFAM_SPLIT_FILE) as f:
        return json.load(f)


def parse_stockholm(path: Path | str) -> dict[str, str]:
    """Parse a Stockholm alignment (.sto or .sto.gz). Returns {name: aligned_seq}."""
    seqs: dict[str, str] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split(None, 1)
            if len(parts) >= 2:
                name, seq = parts[0], parts[1].replace(" ", "")
                seqs[name] = seqs.get(name, "") + seq
    return seqs


def encode_alignment(seqs: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """Encode aligned sequences as a (N, L) int8 array; gap and unknowns -> 20."""
    names = list(seqs.keys())
    if not names:
        return [], np.zeros((0, 0), dtype=np.int8)
    L = len(seqs[names[0]])
    M = np.full((len(names), L), GAP_INDEX, dtype=np.int8)
    for i, name in enumerate(names):
        s = seqs[name]
        assert len(s) == L, f"Inconsistent alignment length for {name}: {len(s)} vs {L}"
        for j, c in enumerate(s.upper()):
            if c in GAP_CHARS:
                M[i, j] = GAP_INDEX
            else:
                M[i, j] = AA_TO_IDX.get(c, GAP_INDEX)
    return names, M


# --- Newick parsing (just enough for Pfam FastTree output) ---

class Node:
    __slots__ = ("name", "branch_length", "children", "parent")

    def __init__(self, name: str = "", branch_length: float = 0.0):
        self.name = name
        self.branch_length = branch_length
        self.children: list[Node] = []
        self.parent: Node | None = None

    def is_leaf(self) -> bool:
        return not self.children


def parse_newick(s: str) -> Node:
    s = s.strip()
    if s.endswith(";"):
        s = s[:-1]
    pos = [0]

    def parse_subtree() -> Node:
        node = Node()
        if pos[0] < len(s) and s[pos[0]] == "(":
            pos[0] += 1
            while True:
                child = parse_subtree()
                child.parent = node
                node.children.append(child)
                if pos[0] >= len(s):
                    break
                if s[pos[0]] == ",":
                    pos[0] += 1
                    continue
                if s[pos[0]] == ")":
                    pos[0] += 1
                    break
        name_start = pos[0]
        while pos[0] < len(s) and s[pos[0]] not in ":,()":
            pos[0] += 1
        node.name = s[name_start:pos[0]]
        if pos[0] < len(s) and s[pos[0]] == ":":
            pos[0] += 1
            bl_start = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in ",()":
                pos[0] += 1
            try:
                node.branch_length = float(s[bl_start:pos[0]])
            except ValueError:
                node.branch_length = 0.0
        return node

    return parse_subtree()


def load_tree(path: Path | str) -> Node:
    with open(path) as f:
        return parse_newick(f.read())


@dataclass
class FamilyData:
    family: str
    names: list[str]                  # sequence names from .sto, in row order
    msa: np.ndarray                   # (N, L) int8, AA index 0..19 or 20 (gap/unknown)
    tree: Node                        # Newick root


def has_family(family: str) -> bool:
    return (PFAM_SEED_DIR / f"{family}.sto").exists() and \
           (PFAM_TREE_DIR / f"{family}.nwk").exists()


def load_family(family: str) -> FamilyData:
    seqs = parse_stockholm(PFAM_SEED_DIR / f"{family}.sto")
    names, msa = encode_alignment(seqs)
    tree = load_tree(PFAM_TREE_DIR / f"{family}.nwk")
    return FamilyData(family=family, names=names, msa=msa, tree=tree)
