"""Fast loader for preprocessed Pfam .npz files.

Reads `data/pfam_processed_top1000/*.npz` as produced by
`experiments/preprocess_pfam_topN.py`. Skips the slow Stockholm/tree
parsing path. Returns the same FamilyCherries objects as
`pfam_data.families_from_list`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pfam_data import FamilyCherries


def families_from_processed(processed_dir: Path,
                                 n_families: int | None = None,
                                 min_cherries: int = 2,
                                 ) -> list[FamilyCherries]:
    """Load preprocessed families from a directory containing per-family
    .npz files and an index.json. Returns a list of FamilyCherries.

    `n_families`: if set, take only the first N families from index.json.
    `min_cherries`: drop families with fewer than this many cherries
    (default 2 instead of 8 since the column-cache path makes small
    families essentially free to include).
    """
    processed_dir = Path(processed_dir)
    index = json.load(open(processed_dir / "index.json"))
    fam_list = index["families"]
    if n_families is not None:
        fam_list = fam_list[:n_families]
    out = []
    for fam in fam_list:
        path = processed_dir / f"{fam}.npz"
        if not path.exists():
            continue
        arrs = np.load(path, allow_pickle=False)
        n_cherries = int(arrs["n_cherries"])
        if n_cherries < min_cherries:
            continue
        out.append(FamilyCherries(
            family=fam,
            L=int(arrs["L"]),
            n_cherries=n_cherries,
            tau=arrs["tau"].astype(np.float64),
            aa_a=arrs["aa_a"].astype(np.int8),
            aa_b=arrs["aa_b"].astype(np.int8),
        ))
    return out
