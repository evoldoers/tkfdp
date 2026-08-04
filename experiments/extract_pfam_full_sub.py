#!/usr/bin/env python3
"""Pass 1 of the full-alignment corpus B: stream Pfam-A.full.gz and extract, for
each AF-partitioned family, its DEEP full alignment -- match columns only,
reservoir-subsampled to <= --max-seqs sequences -- to a compact per-family npz
(int8 MSA in our alphabet ACDEFGHIKLMNPQRSTVWY, gap=20).

Single streaming pass (Pfam-A.full is family-ordered Stockholm blocks); bounded
memory via reservoir sampling of raw sequence lines, match columns extracted from
the block's #=GC RF line at block end. Only families with an AF partition
(data/pdb_af_partition_train/<fam>.npz) are kept.

The subsampled MSAs feed pass 2 (build_cherry_counts_af_full.py), which maps the
cached AF structure onto the match columns for selective contacts and builds
pairwise-distance cherries -- combining B's family breadth with #4's depth."""
from __future__ import annotations

import argparse
import glob
import gzip
import re
import time
from pathlib import Path

import numpy as np

OUR_ALPHA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {c: i for i, c in enumerate(OUR_ALPHA)}
CODE = np.full(256, 20, np.int8)                       # char -> our index, gap=20
for _c, _i in AA_TO_IDX.items():
    CODE[ord(_c)] = _i
AC_RE = re.compile(r"^#=GF AC\s+(PF\d+)")


def encode_match(seqs):
    """(N, L_match) int8. Pfam-Stockholm marks match/insert columns by CASE (no
    #=GC RF line): uppercase and '-' are HMM match states, lowercase and '.' are
    inserts. A column is kept iff no row places a lowercase letter or a '.' there;
    kept residues map to our index (gaps/'-' -> 20)."""
    W = max(len(s) for s in seqs)
    arr = np.full((len(seqs), W), ord("."), np.uint8)          # pad with '.' (insert)
    for i, s in enumerate(seqs):
        b = np.frombuffer(s.encode("latin-1", "replace"), np.uint8)
        arr[i, :b.shape[0]] = b
    is_insert = ((arr >= 97) & (arr <= 122)) | (arr == ord("."))   # lowercase or '.'
    keep = ~is_insert.any(axis=0)
    if not keep.any():
        return np.zeros((len(seqs), 0), np.int8)
    return CODE[arr[:, keep].astype(np.int64)]                 # uppercase->idx, else 20


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", default=str(Path.home() / "bio-datasets/data/pfam/full/Pfam-A.full.gz"))
    ap.add_argument("--af-dir", default="data/pdb_af_partition_train")
    ap.add_argument("--out", default="data/pfam_full_sub")
    ap.add_argument("--max-seqs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    want = {Path(p).stem for p in glob.glob(str(Path(args.af_dir) / "*.npz"))
            if "index" not in p}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print(f"# streaming {args.full}; extracting {len(want)} AF families "
          f"(<= {args.max_seqs} seqs/fam, match cols)", flush=True)

    ac = None                     # current family AC
    keep_fam = False
    names, seqs = [], []          # reservoir (raw aligned strings)
    n_seen = 0                    # sequences seen this block (for reservoir)
    n_done = 0; n_skip = 0; t0 = time.time()

    def flush():
        nonlocal n_done, n_skip
        if not keep_fam or ac is None:
            return
        outp = out / f"{ac}.npz"
        if outp.exists():
            n_skip += 1; return
        if not seqs:
            return
        msa = encode_match(seqs)
        # drop all-gap rows
        good = (msa < 20).any(1)
        msa = msa[good]
        if msa.shape[0] < 2 or msa.shape[1] < 5:
            return
        np.savez(outp.with_suffix(".tmp"), msa=msa)       # np.savez -> <ac>.tmp.npz
        (out / f"{ac}.tmp.npz").replace(outp)
        n_done += 1
        if n_done % 500 == 0:
            print(f"  [{n_done} extracted, {n_skip} cached] last={ac} "
                  f"depth={msa.shape[0]} Lm={msa.shape[1]} t={time.time()-t0:.0f}s",
                  flush=True)

    opn = gzip.open(args.full, "rt", encoding="latin-1")
    for line in opn:
        if line.startswith("# STOCKHOLM"):
            ac = None; keep_fam = False; names = []; seqs = []; n_seen = 0
            continue
        if line.startswith("#=GF AC"):
            m = AC_RE.match(line)
            if m:
                ac = m.group(1); keep_fam = ac in want
            continue
        if line.startswith("//"):
            flush()
            ac = None; keep_fam = False; names = []; seqs = []
            continue
        if line.startswith("#") or not line.strip():
            continue
        if not keep_fam:
            continue
        # sequence line: "name aligned_seq"
        parts = line.rstrip("\n").split(None, 1)
        if len(parts) != 2:
            continue
        nm, sq = parts
        n_seen += 1
        if len(seqs) < args.max_seqs:               # reservoir sampling
            names.append(nm); seqs.append(sq)
        else:
            j = int(rng.integers(0, n_seen))
            if j < args.max_seqs:
                names[j] = nm; seqs[j] = sq
    opn.close()
    flush()
    print(f"# DONE: {n_done} families extracted, {n_skip} cached, "
          f"{time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
