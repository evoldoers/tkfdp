#!/usr/bin/env python3
"""Pick a BAliBASE pair that maximises TKF-DP-relevant coevolution signal.

Score per (family, i, j):
  score = n_qualifying * f_tau(tau)
  n_qualifying = # of column-pairs (a, b), a<b, with:
                    - a and b are both PDB-Ca contacts (<= threshold A)
                      in BOTH sequences,
                    - the AA-pair (X[a], Y[b]) or (X[b], Y[a]) has high
                      K=4 log_2(joint/indep) at THIS pair's tau.
  f_tau(tau) = 1 if tau in [tau_lo, tau_hi]
             = exp(-((tau - tau_mid)^2) / sigma^2) otherwise.

The K=4 log-odds matrix is computed at the pair's own tau by evolving the
class-pair stationary joint independently under the per-class CTMC
(matches plot_k4_log_odds_vs_time.py).

The K=4 model alphabet is ACDE order (per training pipeline, see
tkfdp.lg08; rate_matrix_lg() in tkfmixdom.jax.core.protein is also ACDE
despite its mislabeled AA_ORDER constant).

Identity is computed over the BAliBASE-ref-aligned columns (uppercase
core + lowercase insertion both contribute), restricted to columns where
both sequences have an amino acid.

Tau is estimated via the standard _pairwise_posteriors_tkf92_jax helper,
which optimises tau by Newton-Raphson at the same time as it computes the
pair posterior.

Usage:
    /home/yam/tkf-mixdom/python/.venv/bin/python \\
        analysis/scripts/select_pdb_edge_cov_pair.py \\
        --bali-root ~/bio-datasets/data/balibase/bali3pdbm \\
        --ckpt results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz \\
        --tkf92-params ~/tkf-mixdom/python/experiments/tkf92_fitted_params.json \\
        --contact-threshold 10.0 \\
        --log-odds-min 1.0 \\
        --id-min 0.50 --id-max 0.65 \\
        --tau-lo 0.5 --tau-hi 1.0 \\
        --out /tmp/pdb_edge_cov_pair.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path.home() / "tkf-mixdom" / "python"))

# ACDE order is the actual alphabet used by the K=4 model (and the LG
# matrix returned by rate_matrix_lg(), despite its mislabeled
# constant). See note in plot_k4_log_odds_vs_time.py and lg08.py.
AA_ACDE = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {c: i for i, c in enumerate(AA_ACDE)}


def parse_fasta(path: Path):
    """Return list[(name, raw_uppercase_seq)] from a BAliBASE `in/<family>` file."""
    out = []
    name = None
    seq = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out.append((name, "".join(seq).upper()))
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name is not None:
            out.append((name, "".join(seq).upper()))
    return out


def parse_ref(path: Path):
    """Return dict name -> aligned BAliBASE ref string (mixed case)."""
    out = {}
    name = None
    seq = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(seq)
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name is not None:
            out[name] = "".join(seq)
    return out


def encode_acde(raw: str) -> np.ndarray:
    """Encode raw uppercase AA string -> int32 indices in ACDE order; X -> 20."""
    return np.array([AA_TO_IDX.get(c, 20) for c in raw if c.isalpha()],
                     dtype=np.int32)


def aligned_identity(ref_x: str, ref_y: str):
    """Identity fraction over aligned columns where both have an amino acid."""
    n_match = n_compared = 0
    for ca, cb in zip(ref_x, ref_y):
        if ca.isalpha() and cb.isalpha():
            n_compared += 1
            if ca.upper() == cb.upper():
                n_match += 1
    return n_match, n_compared, (n_match / n_compared if n_compared else 0.0)


def per_class_rate_matrix(Q_lg: np.ndarray, pi_lg: np.ndarray,
                           pi_class_c: np.ndarray) -> np.ndarray:
    """F81 reweight so stationary becomes pi_class_c."""
    Q_out = Q_lg.copy()
    safe_pi = np.where(pi_lg > 0, pi_lg, 1e-30)
    factor = pi_class_c / safe_pi
    Q_out *= factor[None, :]
    np.fill_diagonal(Q_out, 0.0)
    np.fill_diagonal(Q_out, -Q_out.sum(axis=1))
    return Q_out


def class_pair_joint_t0(pi_c: np.ndarray, pi_cp: np.ndarray,
                          H: np.ndarray) -> np.ndarray:
    log_pi = (np.log(pi_c + 1e-30)[:, None]
              + np.log(pi_cp + 1e-30)[None, :]
              - H)
    log_pi -= log_pi.max()
    p = np.exp(log_pi)
    p /= p.sum()
    return p


def evolve_joint_independent(Pi0, P_a, P_b):
    Pi_t1 = np.einsum("ij,ik->kj", Pi0, P_a, optimize=True)
    Pi_t = np.einsum("kj,jl->kl", Pi_t1, P_b, optimize=True)
    return Pi_t


def k4_log_odds_at_tau(pi_class, atoms, assignments, Q_lg, pi_lg,
                        tau: float) -> np.ndarray:
    """Compute K=4 log_2(joint(a, b) / [pi_marg(a) * pi_marg(b)]) at the
    given branch length tau (using same code path as plot_k4_log_odds_vs_time.py).
    """
    from scipy.linalg import expm
    K = pi_class.shape[0]
    A = pi_class.shape[1]
    Q_per_c = np.stack([per_class_rate_matrix(Q_lg, pi_lg, pi_class[c])
                         for c in range(K)], axis=0)
    P_per_c = np.stack([expm(Q_per_c[c] * tau) for c in range(K)], axis=0)
    joint = np.zeros((A, A), dtype=np.float64)
    for c in range(K):
        for cp in range(K):
            atom_idx = int(assignments[c, cp])
            H = atoms[atom_idx]
            Pi0 = class_pair_joint_t0(pi_class[c], pi_class[cp], H)
            Pi_t = evolve_joint_independent(Pi0, P_per_c[c], P_per_c[cp])
            joint += Pi_t / (K * K)
    pi_marg = pi_class.mean(axis=0)
    indep = np.outer(pi_marg, pi_marg)
    indep /= indep.sum()
    joint /= joint.sum()
    log_odds = np.log2((joint + 1e-300) / (indep + 1e-300))
    return log_odds


def tau_score(tau: float, lo: float, hi: float, sigma: float) -> float:
    if lo <= tau <= hi:
        return 1.0
    mid = 0.5 * (lo + hi)
    return float(np.exp(-((tau - mid) ** 2) / (sigma ** 2)))


def ungapped_to_aligned_map(aligned: str) -> list[int]:
    """Return aligned-column index for each ungapped residue position.

    out[k] = the 0-based column in `aligned` corresponding to the k-th
    amino-acid in the ungapped sequence.
    """
    out = []
    for col, c in enumerate(aligned):
        if c.isalpha():
            out.append(col)
    return out


def aligned_to_ungapped_idx(aligned: str) -> dict[int, int]:
    """Return aligned_col_idx -> ungapped_idx mapping (0-based)."""
    out = {}
    k = 0
    for col, c in enumerate(aligned):
        if c.isalpha():
            out[col] = k
            k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bali-root", type=Path,
                    default=Path.home() / "bio-datasets" / "data" /
                    "balibase" / "bali3pdbm")
    ap.add_argument("--ckpt", type=Path,
                    default=REPO / "results" / "K4-emwarm-top1000-2026-05-09" /
                    "_best_chkpt" / "state.npz")
    ap.add_argument("--tkf92-params", type=Path,
                    default=Path.home() / "tkf-mixdom" / "python" /
                    "experiments" / "tkf92_fitted_params.json")
    ap.add_argument("--contact-threshold", type=float, default=10.0)
    ap.add_argument("--min-separation", type=int, default=4)
    ap.add_argument("--log-odds-min", type=float, default=1.0,
                    help="Min K=4 log_2(joint/indep) to qualify a contact pair.")
    ap.add_argument("--id-min", type=float, default=0.50)
    ap.add_argument("--id-max", type=float, default=0.65)
    ap.add_argument("--tau-lo", type=float, default=0.5)
    ap.add_argument("--tau-hi", type=float, default=1.0)
    ap.add_argument("--tau-sigma", type=float, default=0.4)
    ap.add_argument("--qualifying-min", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=10000,
                    help="Skip pairs where max(Lx, Ly) exceeds this (MCMC cost).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--family-filter", default="",
                    help="Only families whose name contains this substring.")
    ap.add_argument("--limit-families", type=int, default=120)
    args = ap.parse_args()

    print(f"[select-pdb-edge-cov] start: contact={args.contact_threshold}A, "
          f"id={args.id_min}-{args.id_max}, tau=[{args.tau_lo},{args.tau_hi}], "
          f"log-odds-min={args.log_odds_min}, qualifying-min={args.qualifying_min}",
          flush=True)

    # ---- Load K=4 checkpoint ----
    d = np.load(args.ckpt, allow_pickle=True)
    pi_class = np.asarray(d["pi_class"], dtype=np.float64)
    atoms = np.asarray(d["potts_atoms"], dtype=np.float64)
    assignments = np.asarray(d["potts_assignments"], dtype=np.int64)
    print(f"[select-pdb-edge-cov] K={pi_class.shape[0]}, A={pi_class.shape[1]}, "
          f"K_H={atoms.shape[0]}", flush=True)

    # ---- LG matrix ----
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg = np.asarray(Q_lg, dtype=np.float64)
    pi_lg = np.asarray(pi_lg, dtype=np.float64)

    # ---- TKF92 indel params ----
    fitted = json.loads(args.tkf92_params.read_text())
    ins_rate = float(fitted["ins_rate"])
    del_rate = float(fitted["del_rate"])
    ext = float(fitted["ext_rate"])

    # ---- BAliBASE iteration ----
    from tkfdp.balibase_pdb_contacts import contacts_for_seq

    in_dir = args.bali_root / "in"
    ref_dir = args.bali_root / "ref"
    families = sorted([p.name for p in in_dir.iterdir()
                       if (ref_dir / p.name).exists()])
    if args.family_filter:
        families = [f for f in families if args.family_filter in f]
    families = families[: args.limit_families]
    print(f"[select-pdb-edge-cov] scanning {len(families)} families", flush=True)

    # Cache for contact sets per (seq_name, raw_seq).
    contact_cache: dict[str, set] = {}

    def get_contacts(seq_name: str, raw_seq: str):
        key = f"{seq_name}::{len(raw_seq)}"
        if key in contact_cache:
            return contact_cache[key]
        cs = contacts_for_seq(seq_name, raw_seq,
                              threshold=args.contact_threshold,
                              min_separation=args.min_separation)
        contact_cache[key] = cs
        return cs

    # We need JAX bits for tau estimation.
    import jax.numpy as jnp
    from tkfmixdom.jax.dp.hmm import _pad_to_bin, _pad_seq
    from tkfmixdom.jax.tree.fsa_anneal import _pairwise_posteriors_tkf92_jax

    candidates = []  # list of dicts
    t0_total = time.time()

    for fam in families:
        fa_path = in_dir / fam
        ref_path = ref_dir / fam
        try:
            fasta = parse_fasta(fa_path)
            ref = parse_ref(ref_path)
        except Exception as e:
            print(f"[select-pdb-edge-cov] {fam}: parse error {e}", flush=True)
            continue
        n = len(fasta)
        if n < 2:
            continue
        # For each pair: check (a) identity in 50-65% range, (b) PDB
        # contacts available for both, (c) tau.
        for i in range(n):
            name_i, raw_i = fasta[i]
            ref_i = ref.get(name_i)
            if ref_i is None:
                continue
            for j in range(i + 1, n):
                name_j, raw_j = fasta[j]
                ref_j = ref.get(name_j)
                if ref_j is None:
                    continue
                m, ncmp, idf = aligned_identity(ref_i, ref_j)
                if not (args.id_min <= idf <= args.id_max):
                    continue
                # Try PDB contacts.
                c_i = get_contacts(name_i, raw_i)
                c_j = get_contacts(name_j, raw_j)
                if c_i is None or c_j is None:
                    continue
                if not c_i or not c_j:
                    continue
                # Estimate tau.
                x = encode_acde(raw_i)
                y = encode_acde(raw_j)
                Lx, Ly = int(x.shape[0]), int(y.shape[0])
                if Lx < 10 or Ly < 10:
                    continue
                if max(Lx, Ly) > args.max_len:
                    continue
                Lx_pad, Ly_pad = _pad_to_bin(Lx), _pad_to_bin(Ly)
                x_pad = _pad_seq(jnp.asarray(x), Lx_pad)
                y_pad = _pad_seq(jnp.asarray(y), Ly_pad)
                _, tau_opt, _ = _pairwise_posteriors_tkf92_jax(
                    x_pad, y_pad, jnp.int32(Lx), jnp.int32(Ly),
                    jnp.float64(ins_rate), jnp.float64(del_rate),
                    jnp.float64(ext),
                    jnp.asarray(Q_lg), jnp.asarray(pi_lg),
                )
                tau = float(tau_opt)
                fts = tau_score(tau, args.tau_lo, args.tau_hi, args.tau_sigma)
                if fts < 0.1:  # skip pairs with disastrous tau
                    continue
                # Compute K=4 log-odds at this tau.
                log_odds = k4_log_odds_at_tau(
                    pi_class, atoms, assignments, Q_lg, pi_lg, tau)
                # For each (a, b) where both a and b are contact-positions
                # in both sequences, check if the AA-pair lookups give
                # high log-odds. We require the column-pair to be a contact
                # in BOTH sequences (intersection).
                # contacts_for_seq returns (i, j) tuples of 0-based ungapped
                # residue indices (which IS the ungapped residue position).
                # So contacts are directly on the ungapped sequence.
                #
                # We want column-pair (a, b) such that (a, b) in c_i AND
                # (a, b) in c_j. This requires that the residue at position
                # a in seq i and position a in seq j correspond to the
                # same ungapped column; we do this via the aligned-column
                # mapping (ref-alignment).
                ref_i_a2u = aligned_to_ungapped_idx(ref_i)
                ref_j_a2u = aligned_to_ungapped_idx(ref_j)
                # Compose: for each aligned column where both have a residue,
                # (ungapped_pos_i, ungapped_pos_j) gives the per-seq indices.
                shared_cols = []
                for col in range(min(len(ref_i), len(ref_j))):
                    if col in ref_i_a2u and col in ref_j_a2u:
                        shared_cols.append((col, ref_i_a2u[col], ref_j_a2u[col]))
                # We want column-pair (col_a, col_b) such that (u_i_a, u_i_b)
                # is in c_i AND (u_j_a, u_j_b) is in c_j.
                qualifying = []
                for k_a in range(len(shared_cols)):
                    col_a, u_i_a, u_j_a = shared_cols[k_a]
                    for k_b in range(k_a + 1, len(shared_cols)):
                        col_b, u_i_b, u_j_b = shared_cols[k_b]
                        # contacts are sorted (i < j) tuples
                        pair_i = (u_i_a, u_i_b) if u_i_a < u_i_b else (u_i_b, u_i_a)
                        pair_j = (u_j_a, u_j_b) if u_j_a < u_j_b else (u_j_b, u_j_a)
                        if pair_i not in c_i or pair_j not in c_j:
                            continue
                        # AA-pair on X and Y.
                        aa_x_a = raw_i[u_i_a]
                        aa_x_b = raw_i[u_i_b]
                        aa_y_a = raw_j[u_j_a]
                        aa_y_b = raw_j[u_j_b]
                        # Check if any of the four AA-pair lookups is high.
                        idx_xa = AA_TO_IDX.get(aa_x_a, -1)
                        idx_xb = AA_TO_IDX.get(aa_x_b, -1)
                        idx_ya = AA_TO_IDX.get(aa_y_a, -1)
                        idx_yb = AA_TO_IDX.get(aa_y_b, -1)
                        if min(idx_xa, idx_xb, idx_ya, idx_yb) < 0:
                            continue
                        # The X-X edge connects position u_i_a and u_i_b
                        # in seq X; the AA-pair on that edge is (aa_x_a, aa_x_b).
                        # The Y-Y edge connects position u_j_a and u_j_b
                        # in seq Y; AA-pair (aa_y_a, aa_y_b).
                        # We score based on min of the two log-odds.
                        lo_x = float(log_odds[idx_xa, idx_xb])
                        lo_y = float(log_odds[idx_ya, idx_yb])
                        lo_min = min(lo_x, lo_y)
                        if lo_min >= args.log_odds_min:
                            qualifying.append({
                                "col_a": int(col_a),
                                "col_b": int(col_b),
                                "u_i_a": int(u_i_a),
                                "u_i_b": int(u_i_b),
                                "u_j_a": int(u_j_a),
                                "u_j_b": int(u_j_b),
                                "aa_x": aa_x_a + aa_x_b,
                                "aa_y": aa_y_a + aa_y_b,
                                "lo_x": lo_x,
                                "lo_y": lo_y,
                                "lo_min": lo_min,
                            })
                score = len(qualifying) * fts
                if len(qualifying) >= args.qualifying_min:
                    print(f"  {fam} pair=({i},{j}) ({name_i}/{name_j}) "
                          f"id={idf*100:5.1f}% tau={tau:.3f} fts={fts:.2f} "
                          f"|c_i|={len(c_i)} |c_j|={len(c_j)} "
                          f"qual={len(qualifying)} score={score:.2f}", flush=True)
                candidates.append({
                    "family": fam,
                    "pair": [i, j],
                    "names": [name_i, name_j],
                    "Lx": Lx, "Ly": Ly,
                    "id_frac": idf,
                    "id_compared": ncmp,
                    "tau": tau,
                    "tau_score": fts,
                    "n_contacts_x": len(c_i),
                    "n_contacts_y": len(c_j),
                    "n_qualifying": len(qualifying),
                    "score": score,
                    "qualifying": qualifying,
                })

    print(f"[select-pdb-edge-cov] total elapsed: {time.time()-t0_total:.1f}s, "
          f"{len(candidates)} candidates", flush=True)

    # Filter to those with enough qualifying contacts.
    good = [c for c in candidates if c["n_qualifying"] >= args.qualifying_min]
    good.sort(key=lambda r: -r["score"])
    print(f"[select-pdb-edge-cov] {len(good)} candidates pass "
          f"qualifying-min={args.qualifying_min}", flush=True)
    if not good:
        # Fall back to top by qualifying anyway.
        print("[select-pdb-edge-cov] no pair passes qualifying-min; "
              "showing top by score across all candidates", flush=True)
        candidates.sort(key=lambda r: -r["score"])
        good = candidates[:10]

    print()
    print("=== Top 10 by score ===")
    for r in good[:10]:
        print(f"  {r['family']} pair=({r['pair'][0]},{r['pair'][1]}) "
              f"id={r['id_frac']*100:.1f}% tau={r['tau']:.3f} "
              f"qual={r['n_qualifying']} score={r['score']:.2f}")
        for q in r["qualifying"][:6]:
            print(f"     contact col=({q['col_a']:>3},{q['col_b']:>3}) "
                  f"X[{q['u_i_a']:>3},{q['u_i_b']:>3}]={q['aa_x']} "
                  f"Y[{q['u_j_a']:>3},{q['u_j_b']:>3}]={q['aa_y']} "
                  f"lo_x={q['lo_x']:+.2f} lo_y={q['lo_y']:+.2f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
            "n_candidates": len(candidates),
            "top": good[:20],
            "winner": good[0] if good else None,
        }, indent=2))
        print(f"[select-pdb-edge-cov] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
