"""Build supervised-coupling counts from PDB-derived reciprocal-NN contacts.

For each PDB-anchored Pfam family in the training split:
  1. Pull the SCOP-anchored PDB, parse, align to a representative MSA row.
  2. Build the {MSA_column -> Cα coord} map.
  3. Find all (i, j) reciprocal-nearest-neighbor pairs of mapped columns
     with Cα-Cα distance < 10 Å (each column's NN is the other; reciprocal).
  4. For each contact pair, gather (anc_i, anc_j, des_i, des_j, tau) tuples
     across all cherries from the preprocessed cherry npz.

Output: a single .npz with concatenated per-pair cherry observations and a
parallel pair-index array, plus discretized tau bins.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def reciprocal_nn_pairs(col_to_ca, max_dist=10.0, min_separation=2):
    """Return list of (col_i, col_j, dist) for reciprocal-NN pairs."""
    cols = sorted(col_to_ca.keys())
    n = len(cols)
    if n < 2:
        return []

    coords = np.array([col_to_ca[c] for c in cols])
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    sep = np.abs(np.array(cols)[:, None] - np.array(cols)[None, :])
    dists = np.where(sep > min_separation, dists, np.inf)
    np.fill_diagonal(dists, np.inf)

    nn = np.argmin(dists, axis=1)
    nn_dist = dists[np.arange(n), nn]

    pairs = []
    seen = set()
    for ai in range(n):
        aj = nn[ai]
        if nn[aj] == ai and nn_dist[ai] < max_dist:
            i, j = sorted((cols[ai], cols[aj]))
            if (i, j) in seen:
                continue
            seen.add((i, j))
            pairs.append((i, j, float(nn_dist[ai])))
    return pairs


def parse_msa_from_sto(sto_path: Path) -> dict[str, str]:
    """Parse a Stockholm MSA into {name: gapped_aligned_seq}."""
    seqs: dict[str, str] = {}
    with open(sto_path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('#') or not line or line.startswith('//'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                name, seq = parts
                seqs[name] = seqs.get(name, '') + seq
    return seqs


def discretize_tau(tau_values, n_bins=8):
    """Return tau bin index for each value + the bin centers."""
    tau_arr = np.asarray(tau_values)
    log_tau = np.log(np.clip(tau_arr, 1e-3, None))
    edges = np.quantile(log_tau, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    bin_idx = np.digitize(log_tau, edges[1:-1])
    bin_centers = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.any():
            bin_centers[b] = np.exp(log_tau[mask].mean())
        else:
            bin_centers[b] = np.exp(0.5 * (edges[b] + edges[b+1]))
    return bin_idx, bin_centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-dir',
                    default='/home/yam/tkf-dp/data/pfam_processed_top1000_pdb',
                    help='Per-family cherry npz directory.')
    ap.add_argument('--pfam-seed-dir',
                    default='/home/yam/bio-datasets/data/pfam/seed',
                    help='Stockholm seed MSA directory.')
    ap.add_argument('--out', default='/tmp/supervised_contacts_K8.npz')
    ap.add_argument('--max-dist', type=float, default=10.0,
                    help='Cα-Cα distance threshold (Å).')
    ap.add_argument('--min-separation', type=int, default=2,
                    help='Skip column pairs with |i - j| <= this.')
    ap.add_argument('--n-time-bins', type=int, default=8,
                    help='Number of branch-length bins.')
    ap.add_argument('--max-families', type=int, default=0,
                    help='0 = all (default).')
    args = ap.parse_args()

    sys.path.insert(0, '/home/yam/tkf-dp/src')
    from tkfdp.pdb_contacts import (
        _scop_id_for_family, _fetch_pdb, _extract_chains,
        _build_msa_to_pdb_map, GAP_CHARS,
    )

    corpus = Path(args.corpus_dir)
    seed_dir = Path(args.pfam_seed_dir)
    families = sorted(p.stem for p in corpus.glob('*.npz'))
    if args.max_families:
        families = families[: args.max_families]
    print(f'Scanning {len(families)} families ...')

    all_anc1, all_anc2 = [], []
    all_des1, all_des2 = [], []
    all_tau = []
    all_pair_idx = []     # idx into pair_meta
    pair_meta = []        # list of (family, col_i, col_j, dist)
    fam_stats = []

    n_ok = n_nopdb = n_nocontact = n_nocherry = 0
    for fi, fam in enumerate(families):
        sto = seed_dir / f'{fam}.sto'
        if not sto.exists():
            n_nopdb += 1
            continue
        pdb_id = _scop_id_for_family(fam)
        if pdb_id is None:
            n_nopdb += 1
            continue
        try:
            pdb_path = _fetch_pdb(pdb_id)
            chains = _extract_chains(pdb_path)
            if not chains:
                n_nopdb += 1; continue
            chains.sort(key=lambda c: -len(c[1]))
            chain_id, chain_residues = chains[0]
            chain_seq = ''.join(r[1] for r in chain_residues)
        except Exception as e:
            n_nopdb += 1; continue

        msa_seqs = parse_msa_from_sto(sto)
        if not msa_seqs:
            n_nopdb += 1; continue

        # Pick best MSA row for alignment
        best_score = -1e30; best_name = None; best_ungap = None
        try:
            from Bio.Align import PairwiseAligner, substitution_matrices
            aligner = PairwiseAligner()
            aligner.mode = 'global'
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
            aligner.open_gap_score = -10.0
            aligner.extend_gap_score = -0.5
        except Exception:
            n_nopdb += 1; continue

        for name, gapped in msa_seqs.items():
            ungapped = ''.join(c for c in gapped.upper() if c not in GAP_CHARS and c.isalpha())
            ungapped = ''.join(c for c in ungapped if c in 'ACDEFGHIKLMNPQRSTVWY')
            if len(ungapped) < 0.1 * len(chain_seq):
                continue
            try:
                sc = float(aligner.score(ungapped, chain_seq))
            except Exception:
                continue
            if sc > best_score:
                best_score = sc; best_name = name; best_ungap = ungapped

        if best_name is None:
            n_nopdb += 1; continue

        try:
            col_to_ca = _build_msa_to_pdb_map(
                msa_seqs[best_name], best_ungap, chain_seq, chain_residues)
        except Exception:
            n_nopdb += 1; continue

        pairs = reciprocal_nn_pairs(
            col_to_ca, max_dist=args.max_dist, min_separation=args.min_separation)
        if not pairs:
            n_nocontact += 1; continue

        cherry = np.load(corpus / f'{fam}.npz', allow_pickle=True)
        aa_a, aa_b, tau = cherry['aa_a'], cherry['aa_b'], cherry['tau']
        if len(tau) == 0:
            n_nocherry += 1; continue

        fam_pair_count = 0
        for (i, j, d) in pairs:
            if i >= aa_a.shape[1] or j >= aa_a.shape[1]:
                continue
            anc_i = aa_a[:, i]; anc_j = aa_a[:, j]
            des_i = aa_b[:, i]; des_j = aa_b[:, j]
            valid = (
                (anc_i >= 0) & (anc_i < 20) & (anc_j >= 0) & (anc_j < 20)
                & (des_i >= 0) & (des_i < 20) & (des_j >= 0) & (des_j < 20)
            )
            if not valid.any():
                continue
            ai = anc_i[valid]; aj = anc_j[valid]
            bi = des_i[valid]; bj = des_j[valid]
            tau_v = tau[valid]
            pidx = len(pair_meta)
            pair_meta.append((fam, int(i), int(j), float(d)))
            all_anc1.append(ai.astype(np.int8))
            all_anc2.append(aj.astype(np.int8))
            all_des1.append(bi.astype(np.int8))
            all_des2.append(bj.astype(np.int8))
            all_tau.append(tau_v.astype(np.float32))
            all_pair_idx.append(np.full(len(ai), pidx, dtype=np.int32))
            fam_pair_count += 1

        if fam_pair_count > 0:
            n_ok += 1
            fam_stats.append((fam, fam_pair_count, len(pairs)))
        else:
            n_nocherry += 1

        if (fi + 1) % 100 == 0:
            print(f'  [{fi+1}/{len(families)}] ok={n_ok} nopdb={n_nopdb} '
                  f'nocontact={n_nocontact} nocherry={n_nocherry}  '
                  f'(running cherries={sum(len(x) for x in all_tau)})')

    print(f'\nFinal: ok={n_ok}, nopdb={n_nopdb}, nocontact={n_nocontact}, nocherry={n_nocherry}')
    if not all_tau:
        print('No contacts found; nothing to write.')
        return

    anc1 = np.concatenate(all_anc1); anc2 = np.concatenate(all_anc2)
    des1 = np.concatenate(all_des1); des2 = np.concatenate(all_des2)
    tau = np.concatenate(all_tau)
    pair_idx = np.concatenate(all_pair_idx)
    print(f'Total cherry-observations: {len(tau):,} across '
          f'{len(pair_meta):,} column-pairs from {n_ok} families.')

    tau_bin, tau_centers = discretize_tau(tau, n_bins=args.n_time_bins)
    print(f'Tau bin centers: {tau_centers.round(3).tolist()}')

    np.savez_compressed(
        args.out,
        anc1=anc1, anc2=anc2, des1=des1, des2=des2,
        tau=tau, tau_bin=tau_bin.astype(np.int16),
        tau_centers=tau_centers.astype(np.float32),
        pair_idx=pair_idx,
        pair_meta=np.array(pair_meta, dtype=object),
        n_pairs=len(pair_meta),
        n_time_bins=args.n_time_bins,
        max_dist=args.max_dist,
        min_separation=args.min_separation,
    )
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
