#!/usr/bin/env python3
"""Parse a Pfam FULL Stockholm alignment (.full.gz) into a clean 20-state MSA for
Potts-model fitting: keep match columns, drop very-gappy columns/sequences, impute the
few residual gaps from each column's amino-acid marginal.  Caches to <acc>_msa.npz."""
import sys, gzip, argparse
import numpy as np

AAORD = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a: i for i, a in enumerate(AAORD)}
GAP = 20


def parse_stockholm(path):
    seqs, order = {}, []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if not line.strip() or line[0] == "#" or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            name, seq = parts
            if name not in seqs:
                seqs[name] = []; order.append(name)
            seqs[name].append(seq)
    rows = ["".join(seqs[n]) for n in order]
    Ls = {len(r) for r in rows}
    if len(Ls) != 1:
        # keep only the modal length (defensive; Pfam full should be rectangular)
        from collections import Counter
        modeL = Counter(len(r) for r in rows).most_common(1)[0][0]
        rows = [r for r in rows if len(r) == modeL]
    return rows


def to_matrix(rows, gap_col=0.4, gap_seq=0.4, seed=0):
    M = np.array([list(r) for r in rows])                 # (N, Lfull) char
    # match columns = no lowercase / '.' anywhere in the column (Stockholm insert marker)
    is_insert = np.array([np.any([(c.islower() or c == ".") for c in M[:, j]])
                          for j in range(M.shape[1])])
    Mm = M[:, ~is_insert]                                 # match columns only
    # map to ints: 20 AAs -> 0..19, everything else (-, X, B, Z, ...) -> GAP
    lut = np.full(128, GAP, int)
    for a, i in AA2I.items():
        lut[ord(a)] = i
    codes = np.array([[lut[ord(c)] if ord(c) < 128 else GAP for c in row] for row in Mm])
    # drop gappy columns then gappy sequences
    colgap = (codes == GAP).mean(0)
    codes = codes[:, colgap <= gap_col]
    seqgap = (codes == GAP).mean(1)
    codes = codes[seqgap <= gap_seq]
    # impute residual gaps from each column's AA marginal
    rng = np.random.default_rng(seed)
    for j in range(codes.shape[1]):
        col = codes[:, j]; g = col == GAP
        if g.any():
            obs = col[~g]
            if len(obs) == 0:
                col[g] = rng.integers(0, 20, g.sum())
            else:
                cnt = np.bincount(obs, minlength=20).astype(float); cnt /= cnt.sum()
                col[g] = rng.choice(20, size=g.sum(), p=cnt)
            codes[:, j] = col
    return codes.astype(np.int8)


def meff(msa, theta=0.8, chunk=1000):
    N = msa.shape[0]; w = np.zeros(N)
    for a in range(0, N, chunk):
        blk = msa[a:a + chunk]
        idn = (blk[:, None, :] == msa[None, :, :]).mean(2)    # (chunk, N)
        w[a:a + chunk] = 1.0 / (idn >= theta).sum(1)
    return w, float(w.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="PF00186")
    ap.add_argument("--in", dest="inp", default=None)
    args = ap.parse_args()
    path = args.inp or f"data/pfam_full/{args.acc}.full.gz"
    rows = parse_stockholm(path)
    msa = to_matrix(rows)
    N, L = msa.shape
    w, Meff = meff(msa) if N <= 20000 else (None, float("nan"))
    colgapmean = float((msa == GAP).mean()) if (msa == GAP).any() else 0.0
    out = f"data/pfam_full/{args.acc}_msa.npz"
    np.savez(out, msa=msa, weights=(w if w is not None else np.ones(N)))
    print(f"# {args.acc}: parsed {len(rows)} rows -> clean MSA N={N} L={L}  "
          f"(residual gap frac {colgapmean:.4f})")
    print(f"# M_eff@0.8 = {Meff:.0f}  (N/M_eff = {N/Meff:.1f})" if w is not None else "# M_eff skipped")
    print(f"# saved -> {out}")
