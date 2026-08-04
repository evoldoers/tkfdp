"""Build a supervised size-2 cluster partition from PDB structures, in the
column indexing of the full-tree CLV corpus (``data/pfam_processed_clv_top1000``).

For each PDB-anchored family in the CLV corpus this writes a per-family
``.npz`` holding a size-2 partition (each column in at most one pair),
formed from several structural coevolution signals and labelled by ``kind``:

  saltbridge  -- acidic (D/E) <-> basic (K/R/H) with charged-group heavy
                 atoms within ``sb_max_dist`` (Barlow & Thornton's 4 A
                 cutoff). The canonical acid-base "field flip". Ca-only
                 contacts miss these (backbone 9-11 A apart while side
                 chains bridge), so we detect from side-chain atoms.
  cation_pi   -- Lys/Arg cationic N <-> Phe/Tyr/Trp aromatic-ring centroid
                 within ``cp_max_dist``. Complementary/swappable, directly
                 expressible by the field chain; often trades with a nearby
                 salt bridge.
  disulfide   -- Cys-Cys SG-SG within ``ss_max_dist``. Highest-precision
                 structural pair; a co-conservation anchor (both-or-neither),
                 and a negative control for the flip diagnostic.
  volume      -- generic reciprocal 3D-NN Ca contact (nonlocal, Ca-Ca <=
                 ``nn_max_dist``) between two CORE-HYDROPHOBIC residues.
                 HEURISTIC LABEL ONLY: true size/volume compensation is a
                 covariation signal, not a fixed-distance interaction, so
                 this just tags core-packing contacts for post-hoc
                 stratification.
  nn          -- any other reciprocal 3D-NN Ca contact (the generic net;
                 also picks up aromatic stacking and mixed packing).

Priority in greedy maximum matching (each column in <= 1 pair): disulfide,
then saltbridge, then cation_pi, then the reciprocal-NN Ca contacts
(volume / nn share the NN geometry, split only by residue identity).

Column indices are in the RAW alignment indexing of ``load_family(fam).msa``,
which is exactly the CLV corpus ``L`` indexing (the CLV builder applies no
column filtering -- see ``preprocess_pfam_pswm.py``), so pairs drop onto CLV
``cluster_id`` columns 1:1.

Output ``<out-dir>/<fam>.npz``:
    pairs   (P, 2) int32   -- (col_i, col_j), i < j
    dist    (P,)   float32 -- defining distance for the pair's kind (A)
    kind    (P,)   <U10    -- one of the labels above
    pdb_id, chain_id, family, L  -- provenance scalars
plus ``<out-dir>/index.json`` with per-family + corpus-wide kind counts.

Usage:
    python experiments/build_pdb_partition.py \
        --clv-dir data/pfam_processed_clv_top1000 \
        --out-dir data/pdb_partition_clv_top1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tkfdp.bio import parse_stockholm  # noqa: E402
from tkfdp.pdb_contacts import (  # noqa: E402
    _scop_id_for_family, _fetch_pdb, THREE_TO_ONE, PFAM_SEED_DIR,
    PDB_CACHE_DIR,
)

ACIDIC = {"D", "E"}
BASIC = {"K", "R", "H"}
CATIONIC = {"K", "R"}            # for cation-pi (His is a weak/variable case)
AROMATIC = {"F", "Y", "W"}
CORE_HYDROPHOBIC = {"A", "V", "L", "I", "M", "F", "W", "Y", "C"}

# Charged-group heavy atoms (salt bridge / basic-N end of cation-pi).
CHARGED_ATOMS = {
    "ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"),
    "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"), "HIS": ("ND1", "NE2"),
}
# Aromatic ring atoms whose mean is the ring centroid.
RING_ATOMS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}


def extract_chains_rich(pdb_path: Path):
    """Longest-first list of (chain_id, residues). Each residue is a dict:
    aa (one-letter), ca (xyz), charged ((M,3) or empty), ring ((3,) centroid
    or None), sg ((3,) or None). Standard-AA residues with a Ca only, same
    ordering as pdb_contacts._extract_chains so the residue index used by the
    MSA->residue map is consistent."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    out = []
    for model in struct:
        for chain in model:
            residues = []
            for res in chain:
                hetflag, resseq, _ = res.id
                if hetflag != " " or res.resname not in THREE_TO_ONE:
                    continue
                if "CA" not in res:
                    continue
                aa = THREE_TO_ONE[res.resname]
                charged = [np.asarray(res[a].coord, dtype=np.float64)
                           for a in CHARGED_ATOMS.get(res.resname, ())
                           if a in res]
                ring_pts = [np.asarray(res[a].coord, dtype=np.float64)
                            for a in RING_ATOMS.get(res.resname, ())
                            if a in res]
                residues.append({
                    "aa": aa,
                    "ca": np.asarray(res["CA"].coord, dtype=np.float64),
                    # Cbeta for the standard Cb<8A contact map (Gly has no CB ->
                    # fall back to CA). Additive: existing consumers ignore it.
                    "cb": np.asarray(res["CB"].coord, dtype=np.float64)
                          if "CB" in res
                          else np.asarray(res["CA"].coord, dtype=np.float64),
                    "charged": (np.stack(charged) if charged
                                else np.zeros((0, 3), dtype=np.float64)),
                    "ring": (np.mean(ring_pts, axis=0) if ring_pts else None),
                    "sg": (np.asarray(res["SG"].coord, dtype=np.float64)
                           if aa == "C" and "SG" in res else None),
                })
            if residues:
                out.append((chain.id, residues))
        break
    return out


def _aa_align(query: str, target: str):
    from Bio.Align import PairwiseAligner, substitution_matrices
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    return aligner.align(query, target)[0]


def build_msa_to_residue_map(msa_seq_with_gaps: str, ungapped_msa_seq: str,
                             chain_seq: str) -> dict[int, int]:
    """{msa_column -> residue index into chain_residues}, same two-step logic
    as pdb_contacts._build_msa_to_pdb_map but returning the residue index."""
    aln = _aa_align(ungapped_msa_seq, chain_seq)
    q_to_t: dict[int, int] = {}
    for (q0, q1), (t0, t1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(q1 - q0):
            q_to_t[q0 + k] = t0 + k
    aa_set = set("ACDEFGHIKLMNPQRSTVWY")
    col_to_ungap, pos = {}, 0
    for col, c in enumerate(msa_seq_with_gaps):
        if c.upper() in aa_set:
            col_to_ungap[col] = pos
            pos += 1
    return {col: q_to_t[q] for col, q in col_to_ungap.items() if q in q_to_t}


def reciprocal_nn_pairs(col_to_ca, max_dist, min_separation):
    """(i, j, dist) for reciprocal-nearest-neighbour Ca pairs, nonlocal."""
    cols = sorted(col_to_ca)
    n = len(cols)
    if n < 2:
        return []
    coords = np.array([col_to_ca[c] for c in cols])
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    sep = np.abs(np.array(cols)[:, None] - np.array(cols)[None, :])
    d = np.where(sep > min_separation, d, np.inf)
    np.fill_diagonal(d, np.inf)
    nn = np.argmin(d, axis=1)
    nn_dist = d[np.arange(n), nn]
    pairs, seen = [], set()
    for ai in range(n):
        aj = nn[ai]
        if nn[aj] == ai and nn_dist[ai] < max_dist:
            i, j = sorted((cols[ai], cols[aj]))
            if (i, j) not in seen:
                seen.add((i, j))
                pairs.append((i, j, float(nn_dist[ai])))
    return pairs


def _min_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min())


def sidechain_pairs(col_to_res, residues, sb_max_dist, cp_max_dist,
                    ss_max_dist, min_separation):
    """Scan mapped column pairs for salt bridges, cation-pi, disulfides.
    Returns dict kind -> list of (i, j, dist)."""
    cols = sorted(col_to_res)
    hits = {"saltbridge": [], "cation_pi": [], "disulfide": []}
    for a in range(len(cols)):
        ci = cols[a]
        ri = residues[col_to_res[ci]]
        for b in range(a + 1, len(cols)):
            cj = cols[b]
            if cj - ci <= min_separation:
                continue
            rj = residues[col_to_res[cj]]
            ai_, aj_ = ri["aa"], rj["aa"]
            # salt bridge: acidic<->basic charged groups.
            if (({ai_, aj_} & ACIDIC) and ({ai_, aj_} & BASIC)
                    and ri["charged"].shape[0] and rj["charged"].shape[0]):
                d = _min_dist(ri["charged"], rj["charged"])
                if d <= sb_max_dist:
                    hits["saltbridge"].append((ci, cj, d))
                    continue
            # cation-pi: K/R cation N <-> F/Y/W ring centroid.
            cat, aro = None, None
            if ai_ in CATIONIC and aj_ in AROMATIC:
                cat, aro = ri, rj
            elif aj_ in CATIONIC and ai_ in AROMATIC:
                cat, aro = rj, ri
            if cat is not None and cat["charged"].shape[0] and aro["ring"] is not None:
                d = _min_dist(cat["charged"], aro["ring"][None, :])
                if d <= cp_max_dist:
                    hits["cation_pi"].append((ci, cj, d))
                    continue
            # disulfide: Cys SG <-> Cys SG.
            if ai_ == "C" and aj_ == "C" and ri["sg"] is not None and rj["sg"] is not None:
                d = float(np.linalg.norm(ri["sg"] - rj["sg"]))
                if d <= ss_max_dist:
                    hits["disulfide"].append((ci, cj, d))
    return hits


def greedy_match(hits, nn_contacts, col_to_res, residues):
    """Priority: disulfide, saltbridge, cation_pi, then reciprocal-NN Ca
    (labelled volume if both residues core-hydrophobic, else nn). Each column
    used at most once."""
    used = set()
    pairs, dists, kinds = [], [], []

    def take(group, kind_fn):
        for i, j, d in sorted(group, key=lambda x: x[2]):
            if i in used or j in used:
                continue
            used.add(i); used.add(j)
            pairs.append((i, j)); dists.append(d); kinds.append(kind_fn(i, j))

    take(hits["disulfide"], lambda i, j: "disulfide")
    take(hits["saltbridge"], lambda i, j: "saltbridge")
    take(hits["cation_pi"], lambda i, j: "cation_pi")

    def nn_kind(i, j):
        ai_ = residues[col_to_res[i]]["aa"]
        aj_ = residues[col_to_res[j]]["aa"]
        both_core = ai_ in CORE_HYDROPHOBIC and aj_ in CORE_HYDROPHOBIC
        return "volume" if both_core else "nn"

    take(nn_contacts, nn_kind)
    return pairs, dists, kinds


def _best_msa_row(msa_seqs, chain_seq):
    from Bio.Align import PairwiseAligner, substitution_matrices
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aa_set = set("ACDEFGHIKLMNPQRSTVWY")
    best_score, best_name, best_ungap = -1e30, None, None
    for name, gapped in msa_seqs.items():
        ungapped = "".join(c for c in gapped.upper() if c in aa_set)
        if len(ungapped) < 0.1 * len(chain_seq):
            continue
        try:
            sc = float(aligner.score(ungapped, chain_seq))
        except Exception:
            continue
        if sc > best_score:
            best_score, best_name, best_ungap = sc, name, ungapped
    return best_name, best_ungap


def load_sifts(path: Path, families: set) -> dict:
    """Build {family -> [(pdb, chain, coverage)]} from SIFTS pdb_chain_pfam."""
    import gzip
    fam2pdb: dict = {}
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("PDB\t"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            pdb, chain, _sp, pf = p[0], p[1], p[2], p[3]
            cov = float(p[4]) if len(p) > 4 and p[4].replace('.', '', 1).isdigit() else 0.0
            if pf in families:
                fam2pdb.setdefault(pf, []).append((pdb, chain, cov))
    return fam2pdb


def _order_candidates(cands, max_try):
    """Cached PDBs first, then by SIFTS coverage desc; cap at max_try."""
    def key(c):
        pdb, chain, cov = c
        cached = (PDB_CACHE_DIR / f"{pdb.lower()}.pdb").exists()
        return (0 if cached else 1, -cov)
    seen = set()
    out = []
    for c in sorted(cands, key=key):
        k = (c[0].lower(), c[1])
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= max_try:
            break
    return out


def _map_pdb_chain(pdb_id, chain_hint, msa_seqs, L):
    """Fetch+parse one PDB, pick the chain (chain_hint, else longest), align to
    the best MSA row, return (col_to_res, residues, chain_id) or None."""
    try:
        pdb_path = _fetch_pdb(pdb_id)
        chains = extract_chains_rich(pdb_path)
    except Exception:
        return None
    if not chains:
        return None
    pick = None
    if chain_hint:
        for cid, res in chains:
            if str(cid) == str(chain_hint):
                pick = (cid, res)
                break
    if pick is None:
        pick = max(chains, key=lambda c: len(c[1]))
    chain_id, residues = pick
    chain_seq = "".join(r["aa"] for r in residues)
    best_name, best_ungap = _best_msa_row(msa_seqs, chain_seq)
    if best_name is None:
        return None
    col_to_res = build_msa_to_residue_map(
        msa_seqs[best_name], best_ungap, chain_seq)
    if not col_to_res:
        return None
    return col_to_res, residues, chain_id


def partition_for_family(family, expected_L, nn_max_dist, sb_max_dist,
                         cp_max_dist, ss_max_dist, min_separation,
                         candidates=None, max_pdb_try=6):
    """Return (pairs, dists, kinds, meta) or None / a status dict.

    candidates: optional [(pdb_id, chain_hint, coverage)] from SIFTS; the one
    mapping the MOST columns is used. If None, falls back to the single SCOP id.
    """
    sto = PFAM_SEED_DIR / f"{family}.sto"
    if not sto.exists():
        return None
    msa_seqs = parse_stockholm(sto)
    if not msa_seqs:
        return None
    L = len(next(iter(msa_seqs.values())))
    if expected_L is not None and L != expected_L:
        return {"__mismatch__": (L, expected_L)}

    if candidates is None:
        pdb_id = _scop_id_for_family(family)
        if pdb_id is None:
            return None
        cand_list = [(pdb_id, None, 1.0)]
    else:
        cand_list = _order_candidates(candidates, max_pdb_try)
        if not cand_list:
            return None

    # Try candidates; keep the one mapping the most columns.
    best = None
    best_pdb = None
    for pdb_id, chain_hint, _cov in cand_list:
        m = _map_pdb_chain(pdb_id, chain_hint, msa_seqs, L)
        if m is None:
            continue
        col_to_res, residues, chain_id = m
        if best is None or len(col_to_res) > len(best[0]):
            best = (col_to_res, residues, chain_id)
            best_pdb = pdb_id
    if best is None:
        return None
    col_to_res, residues, chain_id = best
    col_to_ca = {c: residues[t]["ca"] for c, t in col_to_res.items()}

    hits = sidechain_pairs(col_to_res, residues, sb_max_dist, cp_max_dist,
                           ss_max_dist, min_separation)
    nn = reciprocal_nn_pairs(col_to_ca, nn_max_dist, min_separation)
    pairs, dists, kinds = greedy_match(hits, nn, col_to_res, residues)
    keep = [k for k, (i, j) in enumerate(pairs) if max(i, j) < L]
    pairs = [pairs[k] for k in keep]
    dists = [dists[k] for k in keep]
    kinds = [kinds[k] for k in keep]
    if not pairs:
        return None
    counts = {k: int(kinds.count(k)) for k in set(kinds)}
    meta = {"pdb_id": best_pdb, "chain_id": str(chain_id), "L": L,
            "n_pairs": len(pairs), "kinds": counts, "n_mapped_cols": len(col_to_res),
            "cand": {k: len(v) for k, v in hits.items()} | {"nn": len(nn)}}
    return pairs, dists, kinds, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clv-dir", default="data/pfam_processed_clv_top1000")
    ap.add_argument("--out-dir", default="data/pdb_partition_clv_top1000")
    ap.add_argument("--nn-max-dist", type=float, default=8.0,
                    help="Ca-Ca cutoff for reciprocal-NN contacts (A).")
    ap.add_argument("--sb-max-dist", type=float, default=4.0,
                    help="charged-atom cutoff for salt bridges (A).")
    ap.add_argument("--cp-max-dist", type=float, default=6.0,
                    help="cation-N to ring-centroid cutoff for cation-pi (A).")
    ap.add_argument("--ss-max-dist", type=float, default=2.5,
                    help="SG-SG cutoff for disulfides (A).")
    ap.add_argument("--min-separation", type=int, default=4,
                    help="skip pairs with |i - j| <= this (nonlocal only).")
    ap.add_argument("--max-families", type=int, default=0,
                    help="0 = all families in the CLV corpus.")
    ap.add_argument("--sifts", type=str, default="",
                    help="SIFTS pdb_chain_pfam.tsv[.gz]; resolves families to "
                         "PDB via SIFTS (broad) instead of the sparse seed-file "
                         "SCOP ref. Empty = SCOP-only (legacy).")
    ap.add_argument("--max-pdb-try", type=int, default=6,
                    help="Max candidate PDB chains to try per family (SIFTS "
                         "mode); best column coverage wins.")
    args = ap.parse_args()

    clv_dir, out_dir = Path(args.clv_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = json.loads((clv_dir / "index.json").read_text())
    fam_L = {}
    for fam in index["families"]:
        p = clv_dir / f"{fam}.npz"
        if p.exists():
            fam_L[fam] = int(np.load(p)["L"])
    families = list(fam_L)
    if args.max_families:
        families = families[: args.max_families]

    fam2pdb = {}
    if args.sifts:
        fam2pdb = load_sifts(Path(args.sifts), set(families))
        print(f"# SIFTS: {len(fam2pdb)}/{len(families)} families have PDB "
              f"candidates (max-pdb-try={args.max_pdb_try})")
    print(f"# scanning {len(families)} CLV families for PDB partitions")

    summary = {}
    n_ok = n_nopdb = n_mismatch = n_err = 0
    tot = {"saltbridge": 0, "cation_pi": 0, "disulfide": 0, "volume": 0, "nn": 0}
    tot_pairs = 0
    for fi, fam in enumerate(families):
        cands = fam2pdb.get(fam) if args.sifts else None
        res = partition_for_family(
            fam, fam_L[fam], args.nn_max_dist, args.sb_max_dist,
            args.cp_max_dist, args.ss_max_dist, args.min_separation,
            candidates=cands, max_pdb_try=args.max_pdb_try)
        if res is None:
            n_nopdb += 1
        elif isinstance(res, dict) and "__mismatch__" in res:
            n_mismatch += 1
            print(f"  ! {fam}: L mismatch {res['__mismatch__']} -- skipped")
        elif isinstance(res, dict) and "__error__" in res:
            n_err += 1
        else:
            pairs, dists, kinds, meta = res
            np.savez(
                out_dir / f"{fam}.npz",
                pairs=np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
                dist=np.asarray(dists, dtype=np.float32),
                kind=np.asarray(kinds, dtype="<U10"),
                pdb_id=meta["pdb_id"], chain_id=meta["chain_id"],
                family=fam, L=meta["L"])
            summary[fam] = meta
            n_ok += 1
            tot_pairs += meta["n_pairs"]
            for k, v in meta["kinds"].items():
                tot[k] = tot.get(k, 0) + v
        if (fi + 1) % 100 == 0:
            print(f"  [{fi+1}/{len(families)}] ok={n_ok} nopdb={n_nopdb} "
                  f"mism={n_mismatch} err={n_err} pairs={tot_pairs} "
                  f"sb={tot['saltbridge']} cp={tot['cation_pi']} "
                  f"ss={tot['disulfide']} vol={tot['volume']}", flush=True)

    (out_dir / "index.json").write_text(json.dumps({
        "clv_dir": str(clv_dir),
        "nn_max_dist": args.nn_max_dist, "sb_max_dist": args.sb_max_dist,
        "cp_max_dist": args.cp_max_dist, "ss_max_dist": args.ss_max_dist,
        "min_separation": args.min_separation,
        "n_families_with_partition": n_ok, "n_no_pdb": n_nopdb,
        "n_mismatch": n_mismatch, "n_error": n_err,
        "total_pairs": tot_pairs, "kind_totals": tot,
        "families": summary,
    }, indent=2))
    print(f"\n# done: {n_ok} families, {tot_pairs} pairs  "
          f"[saltbridge={tot['saltbridge']} cation_pi={tot['cation_pi']} "
          f"disulfide={tot['disulfide']} volume={tot['volume']} nn={tot['nn']}]"
          f"\n# nopdb={n_nopdb} mismatch={n_mismatch} err={n_err}"
          f"\n# wrote {out_dir}")


if __name__ == "__main__":
    main()
