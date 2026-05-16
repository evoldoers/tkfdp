#!/usr/bin/env python3
"""Pick a CLOSE Cys-rich PF00014 pair satisfying the sharpening-sweep
constraints:

  - sequence identity >= 65% (over aligned non-gap columns)
  - tau in [0.1, 0.5]   (puts C-C log2 in [+2.5, +3.3] under K=4 emwarm)
  - shared conserved Cys-Cys positions >= 4 (both sequences have C in
    >= 4 of the family's Cys columns)

Tau is computed with the SAME helper used everywhere else in tkf-dp
(_pairwise_posteriors_tkf92_jax, which fits tau via Newton internally
and returns it as tau_opt). We only iterate pairs that pass the
cheaper sequence-identity + shared-Cys filters first; tau is the
expensive one.

Usage:
    python analysis/scripts/select_close_cys_pair.py \
        --pfam-sto ~/bio-datasets/data/pfam/random100/PF00014.sto \
        --tkf92-params ~/tkf-mixdom/python/experiments/tkf92_fitted_params.json \
        --out /tmp/select_close_cys_pair_PF00014.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def _aa_to_int_dict():
    # The K=4 model lives on the ACDE (alphabetical) AA alphabet --
    # this matches the rate matrix returned by
    # tkfmixdom.jax.core.protein.rate_matrix_lg(), which is permuted
    # from PAML to alphabetical order (see _paml_to_alphabetical_perm).
    # The pre-2026-05-15 encoding used PAML order (ARND...) and thus
    # silently misidentified all AAs by their PAML index, destroying
    # the C-C coupling signal.
    import string
    aa = "ACDEFGHIKLMNPQRSTVWY"
    d = {c: i for i, c in enumerate(aa)}
    for c in string.ascii_uppercase:
        d.setdefault(c, 20)
    return d


def parse_stockholm(path: Path):
    names: list[str] = []
    seqs: dict[str, list[str]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name, frag = parts[0], parts[1].strip()
            if name not in seqs:
                seqs[name] = []
                names.append(name)
            seqs[name].append(frag)
    return names, ["".join(seqs[n]) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfam-sto", type=Path,
                    default=Path.home() / "bio-datasets" / "data" / "pfam"
                    / "random100" / "PF00014.sto")
    ap.add_argument("--tkf92-params", type=Path,
                    default=Path.home() / "tkf-mixdom" / "python" /
                    "experiments" / "tkf92_fitted_params.json")
    ap.add_argument("--id-min", type=float, default=0.65)
    ap.add_argument("--tau-min", type=float, default=0.10)
    ap.add_argument("--tau-max", type=float, default=0.50)
    ap.add_argument("--cys-min", type=int, default=4,
                    help="Min shared conserved Cys positions")
    ap.add_argument("--cys-frac", type=float, default=0.5,
                    help="Column is a 'Cys column' if Cys frac >= this")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON to write results to")
    args = ap.parse_args()

    names, alignments = parse_stockholm(args.pfam_sto)
    aa_grid = np.array([[c.upper() for c in a] for a in alignments])
    c_frac = (aa_grid == "C").mean(axis=0)
    c_cols_1based = [i + 1 for i, f in enumerate(c_frac) if f >= args.cys_frac]
    print(f"[select-close] N seqs = {len(names)}, Cys cols (1-based) = "
          f"{c_cols_1based}", flush=True)

    # First filter: identity + shared-Cys.
    AA = _aa_to_int_dict()
    N = len(names)
    candidates = []   # list of (i, j, id_frac, shared_cys)
    for i in range(N):
        a = alignments[i]
        for j in range(i + 1, N):
            b = alignments[j]
            m = nc = 0
            for ca, cb in zip(a, b):
                if ca.isalpha() and cb.isalpha():
                    nc += 1
                    if ca.upper() == cb.upper():
                        m += 1
            if nc == 0:
                continue
            idf = m / nc
            if idf < args.id_min:
                continue
            # Shared Cys: both have 'C' at one of the Cys columns
            shared = 0
            for col in c_cols_1based:
                ca = a[col - 1].upper()
                cb = b[col - 1].upper()
                if ca == "C" and cb == "C":
                    shared += 1
            if shared < args.cys_min:
                continue
            candidates.append((i, j, idf, shared, nc))
    print(f"[select-close] {len(candidates)} pairs pass id+cys filter",
          flush=True)

    if not candidates:
        print("[select-close] NO candidates pass id+cys filter. "
              "Loosen --id-min or --cys-min.")
        return

    # Now: for each candidate, fit tau via the standard helper.
    # We do this lazily — sort candidates by descending shared-cys then by
    # descending identity, fit tau, take first that falls in window.
    candidates.sort(key=lambda x: (-x[3], -x[2]))   # prefer high shared cys + id

    # Load TKF92 indel params + LG matrix.
    fitted = json.loads(args.tkf92_params.read_text())
    ins_rate = float(fitted["ins_rate"])
    del_rate = float(fitted["del_rate"])
    ext = float(fitted["ext_rate"])

    from tkfmixdom.jax.core.protein import rate_matrix_lg
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg = np.asarray(Q_lg)
    pi_lg = np.asarray(pi_lg)

    import jax.numpy as jnp
    from tkfmixdom.jax.dp.hmm import _pad_to_bin, _pad_seq
    from tkfmixdom.jax.tree.fsa_anneal import _pairwise_posteriors_tkf92_jax

    def to_int_arr(s: str):
        raw = "".join(c for c in s if c.isalpha())
        return np.array([AA.get(c.upper(), 20) for c in raw], dtype=np.int32), raw

    rows = []
    for k, (i, j, idf, shared, nc) in enumerate(candidates):
        x_seq, raw_x = to_int_arr(alignments[i])
        y_seq, raw_y = to_int_arr(alignments[j])
        Lx, Ly = int(x_seq.shape[0]), int(y_seq.shape[0])
        Lx_pad, Ly_pad = _pad_to_bin(Lx), _pad_to_bin(Ly)
        x_pad = _pad_seq(jnp.asarray(x_seq, dtype=jnp.int32), Lx_pad)
        y_pad = _pad_seq(jnp.asarray(y_seq, dtype=jnp.int32), Ly_pad)
        t0 = time.time()
        _, tau_opt, _ = _pairwise_posteriors_tkf92_jax(
            x_pad, y_pad, jnp.int32(Lx), jnp.int32(Ly),
            jnp.float64(ins_rate), jnp.float64(del_rate), jnp.float64(ext),
            jnp.asarray(Q_lg), jnp.asarray(pi_lg),
        )
        tau = float(tau_opt)
        dt = time.time() - t0
        in_window = args.tau_min <= tau <= args.tau_max
        marker = "**" if in_window else "  "
        print(f"  {marker} [{k:>3}/{len(candidates)}] "
              f"i={i:>2} j={j:>2} ({names[i][:30]} / {names[j][:30]}) "
              f"id={idf*100:5.1f}%  Lx,Ly=({Lx},{Ly})  "
              f"shared_cys={shared}  tau={tau:.3f}  [{dt:.1f}s]",
              flush=True)
        rows.append({
            "pair": [i, j], "names": [names[i], names[j]],
            "id_frac": idf, "id_compared": nc,
            "shared_cys": shared, "tau": tau,
            "Lx": Lx, "Ly": Ly,
            "in_tau_window": in_window,
        })

    # First in-window candidate (already sorted by quality).
    in_window = [r for r in rows if r["in_tau_window"]]
    print()
    if not in_window:
        print(f"[select-close] No pair has tau in [{args.tau_min}, "
              f"{args.tau_max}]. Best (in terms of shared-cys+id) was:")
        print(json.dumps(rows[0], indent=2))
    else:
        chosen = in_window[0]
        print("[select-close] CHOSEN:")
        print(json.dumps(chosen, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "cys_cols_1based": c_cols_1based,
            "rows": rows,
            "chosen": in_window[0] if in_window else None,
        }, indent=2))
        print(f"[select-close] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
