#!/usr/bin/env python3
"""Composite-likelihood MSA-column edge posterior triangle (Holmes 2004 Fig 14).

Runs the joint Infinite Pair HMM sampler on a handful of sequence-pairs
from a Pfam family's MSA, projects each pair's edges from sequence
coordinates to MSA-column coordinates, and aggregates a composite
edge-pair posterior over MSA columns. Plots:

  TOP    : triangular heatmap P(MSA-cols (k1, k2) connected | MSA)
  BOTTOM : the MSA itself (colored by AA chemistry; gaps blank).

Default family: PF00014 (Kunitz/BPTI -- 99 sequences, 80 columns, three
classical disulfide pairs: C2-C47, C15-C75, C55-C79).

The composite likelihood is

    sum over chosen sequence-pairs (s_a, s_b) of
        log P(s_a, s_b, E_{ab} | TKF-DP, alpha_z),

where E_{ab} is the per-pair edge set (each pair has its own sampler
chain; the edge sets are NOT shared across pairs). Each pair's edges
are then mapped to MSA columns using the per-sequence column-position
maps.

Usage
-----

    python analysis/scripts/plot_holmes_msa_triangle.py \
        --pfam-sto ~/bio-datasets/data/pfam/random100/PF00014.sto \
        --ckpt results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz \
        --n-pairs 8 \
        --n-sweeps 1500 --n-burnin 300 \
        --out math-paper/figures/holmes_msa_triangle_PF00014
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


def parse_stockholm(path: Path):
    """Parse a Pfam Stockholm file. Returns (names, aligned_seqs)
    where aligned_seqs[k] is the original gapped string."""
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


def _aa_to_int():
    # The K=4 model lives on the **ACDE** (alphabetical) AA alphabet,
    # matching the training pipeline (tkfdp.lg08 PI_LG08_J / S_LG08_F81_J
    # are ACDE-ordered) and the rate matrix returned by
    # tkfmixdom.jax.core.protein.rate_matrix_lg() (which is ACDE-ordered
    # despite that module's mislabeled AA_ORDER constant; the matrix
    # itself is ACDE -- verify pi[1] = 0.0129 = C).
    # Pre-2026-05-15 this used the ARND order, which silently mapped
    # C -> int 4 = F (phenylalanine in the model's ACDE alphabet),
    # destroying the C-C coupling signal in MCMC runs.
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return {c: i for i, c in enumerate(aa)}


def aligned_to_seq_and_map(aligned: str, A2I: dict[str, int]):
    """Strip gaps from an aligned sequence; return (seq_int, col_to_pos).

    col_to_pos[k] is the 1-based sequence position of column k (1-based), or
    0 if column k is a gap. Conversely pos_to_col[1..L] is the MSA column.
    """
    col_to_pos = [0]  # 1-based; index 0 unused
    seq_int = []
    pos = 0
    for col, c in enumerate(aligned, start=1):
        if c.isalpha():
            pos += 1
            seq_int.append(A2I.get(c.upper(), 20))
            col_to_pos.append(pos)
        elif c in "-.":
            col_to_pos.append(0)
        else:
            col_to_pos.append(0)
    pos_to_col = [0] * (pos + 1)
    for col, p in enumerate(col_to_pos):
        if p > 0:
            pos_to_col[p] = col
    return (np.asarray(seq_int, dtype=np.int32),
            np.asarray(col_to_pos, dtype=np.int32),
            np.asarray(pos_to_col, dtype=np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfam-sto", type=Path,
                    default=Path.home() / "bio-datasets" / "data" / "pfam" /
                    "random100" / "PF00014.sto",
                    help="Stockholm-format Pfam family MSA.")
    ap.add_argument("--ckpt", type=Path,
                    default=REPO / "results" / "K4-emwarm-top1000-2026-05-09" /
                    "_best_chkpt" / "state.npz")
    ap.add_argument("--tkf92-params",
                    default=Path.home() / "tkf-mixdom" / "python" /
                    "experiments" / "tkf92_fitted_params.json")
    ap.add_argument("--alpha-z-ladder", default="100,250,700,2000,1e4")
    ap.add_argument("--no-condition-on-edges", dest="condition_on_edges",
                    action="store_false", default=True,
                    help="Disable conditioning the column-pair edge "
                    "marginal on |E| >= 1 (default: ON). With conditioning, "
                    "the triangle is renormalised by P(|E| >= 1 | data) "
                    "estimated as 1 - exp(-E[|E|]) (Poisson), giving a "
                    "scale comparable across pairs.")
    ap.add_argument("--n-pairs", type=int, default=6,
                    help="Number of randomly chosen sequence-pairs from the MSA.")
    ap.add_argument("--n-sweeps", type=int, default=1500)
    ap.add_argument("--n-burnin", type=int, default=300)
    ap.add_argument("--swap-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="math-paper/figures/holmes_msa_triangle_PF00014",
                    type=Path)
    ap.add_argument("--cache-json", type=Path, default=None)
    ap.add_argument("--seq-subset", type=str, default=None,
                    help="Comma-separated list of zero-based sequence indices to "
                    "use (defaults to a random subset). Useful for repeatable runs.")
    ap.add_argument("--cmap", default="magma")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--family-label", default=None,
                    help="Family label for the title (defaults to filename stem).")
    ap.add_argument("--gappy-threshold", type=float, default=0.5,
                    help="Gap-fraction threshold above which a column is 'gappy' "
                    "and masked in the triangle (still shown in MSA). Default 0.5.")
    ap.add_argument("--top-k", type=int, default=0,
                    help="If > 0, fixed top-K cells. If 0 (default), "
                    "dynamically pick N = the minimum prefix length such "
                    "that all Cys-Cys pairs are included.")
    ap.add_argument("--__padding", type=int, default=0,
                    help="Number of brightest non-gappy cells to annotate "
                    "with circles + arcs. Default 15.")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Parse MSA ------------------------------------------------------
    names, alignments = parse_stockholm(args.pfam_sto)
    N_seqs = len(names)
    if N_seqs == 0:
        raise RuntimeError(f"No sequences parsed from {args.pfam_sto}")
    L_msa = len(alignments[0])
    for nm, aln in zip(names, alignments):
        if len(aln) != L_msa:
            raise RuntimeError(f"Inconsistent MSA column count at {nm}: "
                                 f"{len(aln)} vs {L_msa}")
    A2I = _aa_to_int()
    seq_ints = []
    col_to_pos_list = []
    raw_strs = []
    for aln in alignments:
        s_int, c2p, _ = aligned_to_seq_and_map(aln, A2I)
        seq_ints.append(s_int)
        col_to_pos_list.append(c2p)
        raw_strs.append(aln.replace("-", "").replace(".", ""))
    print(f"[holmes-msa] {args.pfam_sto.name}: N={N_seqs}, L_msa={L_msa}",
          flush=True)

    # ---- 2. Select sequence-pairs ----------------------------------------
    rng = np.random.default_rng(args.seed)
    if args.seq_subset is not None:
        chosen = [int(x) for x in args.seq_subset.split(",")]
    else:
        # Random subset; pick MIN(2 * n_pairs, N_seqs) sequences and form
        # pairs from them so each sequence appears at most ceil(2 * n_pairs / N) times.
        n_to_pick = min(2 * args.n_pairs, N_seqs)
        chosen = rng.choice(N_seqs, size=n_to_pick, replace=False).tolist()
    print(f"[holmes-msa]   chosen sequences ({len(chosen)}): {chosen}", flush=True)
    # Build pair list of size args.n_pairs by combinations within chosen
    # set. Adjacent pairs from the chosen list, wrapping around so all
    # chosen sequences appear.
    pair_list = []
    for k in range(args.n_pairs):
        s_a = chosen[(2 * k) % len(chosen)]
        s_b = chosen[(2 * k + 1) % len(chosen)]
        if s_a == s_b:
            s_b = chosen[(2 * k + 2) % len(chosen)]
        pair_list.append((s_a, s_b))
    print(f"[holmes-msa]   pair_list: {pair_list}", flush=True)

    # ---- 3. Set up TKF params + K4 state ---------------------------------
    fitted = json.loads(Path(args.tkf92_params).read_text())
    ins_rate = float(fitted["ins_rate"])
    del_rate = float(fitted["del_rate"])
    ext = float(fitted["ext_rate"])
    print(f"[holmes-msa] TKF92 indel: ins={ins_rate:.5f} del={del_rate:.5f} "
          f"ext={ext:.4f}", flush=True)

    from tkfmixdom.jax.core.protein import rate_matrix_lg

    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg = np.asarray(Q_lg); pi_lg = np.asarray(pi_lg)

    # Re-use the K4 loader from plot_holmes_tile.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_holmes_tile import _build_k4_state, _build_boost_state
    from tkfdp.block_likelihoods import empirical_pi_c_from_checkpoint
    state = _build_k4_state(args.ckpt)
    pi_c = empirical_pi_c_from_checkpoint(args.ckpt)
    pair_background = 'lg08'
    print(f"[holmes-msa] K4 ckpt loaded: K_c={state.K_c}, A={state.A}",
          flush=True)
    print(f"[holmes-msa] empirical pi_c = "
          f"{[f'{x:.3f}' for x in pi_c]}", flush=True)

    # ---- 4. Cache check --------------------------------------------------
    cache_data = None
    if args.cache_json is not None and args.cache_json.exists():
        try:
            cache_data = json.loads(args.cache_json.read_text())
            if (cache_data.get("pfam_path") == str(args.pfam_sto)
                    and tuple(tuple(p) for p in cache_data["pair_list"]) ==
                    tuple(pair_list)):
                print(f"[holmes-msa] cache hit: {args.cache_json}", flush=True)
            else:
                cache_data = None
        except Exception as e:
            print(f"[holmes-msa] cache load failed: {e}", flush=True)
            cache_data = None

    # ---- 5. Per-pair joint sampler ---------------------------------------
    ladder = [float(x) for x in args.alpha_z_ladder.split(",")]
    if cache_data is None:
        from tkfdp.mcmc_infinite_phmm import mcmc_corrected_posterior

        # Composite-likelihood MSA-column edge-pair counters.
        col_pair_counts: dict[tuple[int, int], int] = {}
        # ALSO save per-pair MSA-column edge-pair counts (needed for Figure A
        # pair-selection by which pair contributes most to top cells).
        per_pair_col_pair_counts: list[list[list[int]]] = []
        total_recorded = 0
        per_pair_summary = []
        for k, (s_a, s_b) in enumerate(pair_list):
            x_seq = seq_ints[s_a]
            y_seq = seq_ints[s_b]
            Lx, Ly = int(x_seq.shape[0]), int(y_seq.shape[0])
            print(f"[holmes-msa] pair {k+1}/{len(pair_list)}: "
                  f"({names[s_a]}:{Lx}, {names[s_b]}:{Ly}) ", flush=True)
            bs, tau = _build_boost_state(
                x_seq, y_seq, ins_rate, del_rate, ext, Q_lg, pi_lg, state,
                pi_c=pi_c, pair_background=pair_background)
            t0 = time.time()
            Q_prime, _, Q_baseline, log_F0, joint_diag = mcmc_corrected_posterior(
                x_seq=x_seq, y_seq=y_seq, t=tau,
                ins_rate=ins_rate, del_rate=del_rate, ext=ext,
                Q_lg=Q_lg, pi_lg=pi_lg, boost_state=bs,
                alpha_z=ladder[0],
                alpha_z_ladder=ladder, swap_every=args.swap_every,
                n_sweeps=args.n_sweeps, n_burnin=args.n_burnin,
                n_chains=1, k_max=-1, seed=args.seed + k,
                init_mode="viterbi", verbose=False,
            )
            cold = joint_diag["per_rung"][0]
            nrec = int(cold.n_recorded_for_edges)
            total_recorded += nrec
            dt = time.time() - t0
            print(f"[holmes-msa]   tau={tau:.3f} done in {dt:.0f}s, "
                  f"n_recorded={nrec}", flush=True)
            # Lift to MSA columns: each edge ((ai, aj), (bi, bj)) has
            # X-endpoint pair (ai, bi) and Y-endpoint pair (aj, bj). We map
            # those onto MSA columns via col_to_pos inversion. Edge contributes
            # +1 to col_pair_counts[(col(s_a, ai), col(s_a, bi))]
            # AND +1 to col_pair_counts[(col(s_b, aj), col(s_b, bj))].
            # (Each edge generates TWO column-pair observations: one per
            # sequence axis, summed over pairs.)
            c2p_a = col_to_pos_list[s_a]
            c2p_b = col_to_pos_list[s_b]
            # Build inverse: pos_to_col_a[1..Lx] = MSA col, 0 if mismatch.
            pos_to_col_a = np.zeros(Lx + 1, dtype=np.int32)
            for col_idx in range(c2p_a.shape[0]):
                p = int(c2p_a[col_idx])
                if 1 <= p <= Lx:
                    pos_to_col_a[p] = col_idx
            pos_to_col_b = np.zeros(Ly + 1, dtype=np.int32)
            for col_idx in range(c2p_b.shape[0]):
                p = int(c2p_b[col_idx])
                if 1 <= p <= Ly:
                    pos_to_col_b[p] = col_idx
            this_pair_counts: dict[tuple[int, int], int] = {}
            for (i1, i2), c in cold.edge_pair_x_counts.items():
                col1 = int(pos_to_col_a[i1])
                col2 = int(pos_to_col_a[i2])
                if col1 == 0 or col2 == 0 or col1 == col2:
                    continue
                key = (min(col1, col2), max(col1, col2))
                col_pair_counts[key] = col_pair_counts.get(key, 0) + c
                this_pair_counts[key] = this_pair_counts.get(key, 0) + c
            for (j1, j2), c in cold.edge_pair_y_counts.items():
                col1 = int(pos_to_col_b[j1])
                col2 = int(pos_to_col_b[j2])
                if col1 == 0 or col2 == 0 or col1 == col2:
                    continue
                key = (min(col1, col2), max(col1, col2))
                col_pair_counts[key] = col_pair_counts.get(key, 0) + c
                this_pair_counts[key] = this_pair_counts.get(key, 0) + c
            per_pair_col_pair_counts.append(
                [[int(k_[0]), int(k_[1]), int(v)] for k_, v in this_pair_counts.items()]
            )
            per_pair_summary.append({
                "pair": [int(s_a), int(s_b)],
                "names": [names[s_a], names[s_b]],
                "Lx": Lx, "Ly": Ly,
                "tau": float(tau),
                "n_recorded": nrec,
                "wall_time_s": float(dt),
            })

        # Build column-pair posterior matrix. Normaliser: total over all pairs
        # of n_recorded * 2 (each edge contributes 2 axis observations).
        n_axis_obs = 2 * total_recorded
        col_pair_post = np.zeros((L_msa + 1, L_msa + 1), dtype=np.float64)
        for (c1, c2), c in col_pair_counts.items():
            v = c / max(n_axis_obs, 1)
            col_pair_post[c1, c2] = v
            col_pair_post[c2, c1] = v

        cache_data = {
            "pfam_path": str(args.pfam_sto),
            "L_msa": L_msa, "N_seqs": N_seqs,
            "pair_list": [list(p) for p in pair_list],
            "names_used": list(set([names[i] for p in pair_list for i in p])),
            "col_pair_post": col_pair_post.tolist(),
            "col_pair_counts": [[int(k[0]), int(k[1]), int(v)]
                                  for k, v in col_pair_counts.items()],
            "per_pair_col_pair_counts": per_pair_col_pair_counts,
            "total_recorded": int(total_recorded),
            "per_pair": per_pair_summary,
            "alpha_z_ladder": ladder,
            "n_sweeps": args.n_sweeps, "n_burnin": args.n_burnin,
        }
        if args.cache_json is not None:
            args.cache_json.parent.mkdir(parents=True, exist_ok=True)
            args.cache_json.write_text(json.dumps(cache_data))
            print(f"[holmes-msa] cache written to {args.cache_json}", flush=True)
    else:
        col_pair_post = np.asarray(cache_data["col_pair_post"])

    # ---- 5b. Optional renormalisation: condition on |E| >= 1 -----------------
    # See the matching block in plot_holmes_tile.py. Default ON;
    # disable with --no-condition-on-edges. Poisson estimate of
    # P(|E| >= 1 | data) ~= 1 - exp(-E[|E|]) where E[|E|] = sum / 2.
    if args.condition_on_edges:
        mean_E = float(col_pair_post.sum()) / 2.0  # expected edges/sample
        if mean_E > 0.0:
            Z = 1.0 - np.exp(-mean_E)  # Poisson estimate of P(|E|>=1)
            if Z < 1.0 - 1e-9:
                col_pair_post = col_pair_post / Z
                print(f"[holmes-msa] conditioned on |E| >= 1: "
                      f"mean_E={mean_E:.3f}, Z=P(|E|>=1)~={Z:.3f}, "
                      f"scale=1/Z={1.0/Z:.3f}", flush=True)

    # ---- 6. Define gappy / Cys / top-K --------------------------------------
    aa_grid = np.array([[c.upper() for c in aln] for aln in alignments])
    c_frac = (aa_grid == "C").mean(axis=0)
    c_cols = [i + 1 for i, f in enumerate(c_frac) if f >= 0.5]
    # Gappy criterion: gap fraction > gappy_threshold (any non-letter).
    is_letter = np.vectorize(lambda ch: ch.isalpha())(aa_grid)
    gap_frac = 1.0 - is_letter.mean(axis=0)
    gappy_cols = [i + 1 for i, f in enumerate(gap_frac) if f > args.gappy_threshold]
    # Per-column AA frequencies (over non-gap residues), plus column
    # information content = log2(20) - H(column) in bits. Used both
    # for legacy consensus annotation AND the sequence-logo render.
    import collections as _coll
    consensus_aa = []
    consensus_freq = []
    column_freqs = []     # list of dict[AA -> freq] over non-gap entries
    column_info = []      # bits, log2(20) - H(column); 0 if all-gap
    LOG2_20 = float(np.log2(20))
    for col_idx in range(aa_grid.shape[1]):
        non_gap = [aa_grid[r, col_idx] for r in range(aa_grid.shape[0])
                    if aa_grid[r, col_idx].isalpha()]
        if not non_gap:
            consensus_aa.append("-")
            consensus_freq.append(0.0)
            column_freqs.append({})
            column_info.append(0.0)
            continue
        cnt = _coll.Counter(non_gap)
        total = sum(cnt.values())
        freqs = {a: c / total for a, c in cnt.items()}
        top_aa, top_n = cnt.most_common(1)[0]
        consensus_aa.append(top_aa)
        consensus_freq.append(top_n / total)
        column_freqs.append(freqs)
        H = -sum(p * np.log2(p) for p in freqs.values() if p > 0)
        column_info.append(max(LOG2_20 - H, 0.0))
    print(f"[holmes-msa] cys columns (frac >= 0.5 are C): {c_cols}", flush=True)
    print(f"[holmes-msa] gappy columns (gap_frac > {args.gappy_threshold}): "
          f"{gappy_cols}", flush=True)

    cset = set(c_cols)
    gset = set(gappy_cols)

    # Top-K cells from saved triangle (NOT from a stale report). Skip cells
    # whose row or column is gappy (those are masked NaN in the triangle
    # display, so the bright squares should never appear there).
    trip = []
    for i in range(1, L_msa + 1):
        if i in gset:
            continue
        for j in range(i + 1, L_msa + 1):
            if j in gset:
                continue
            v = float(col_pair_post[i, j])
            if v > 0:
                trip.append((v, i, j))
    trip.sort(reverse=True)

    # Dynamic top-N: choose N = smallest prefix length such that the
    # top-N contains every Cys-Cys pair (i, j with both i and j in cset).
    # Provides a clean "model recovers all disulfide-candidate cells"
    # statement: N is small when the chain ranks every Cys-Cys cell
    # above background; N is large when some C-C cells are buried in
    # the bulk.
    if args.top_k > 0:
        top_k = trip[: args.top_k]
    else:
        n_cc_total = sum(1 for v, i, j in trip
                          if i in cset and j in cset)
        if n_cc_total == 0:
            top_k = trip[: 15]    # fallback: no Cys in this family
        else:
            n_cc_seen = 0
            cutoff = None
            for k, (v, i, j) in enumerate(trip):
                if i in cset and j in cset:
                    n_cc_seen += 1
                if n_cc_seen == n_cc_total:
                    cutoff = k + 1
                    break
            top_k = trip[: cutoff]
            print(f"[holmes-msa] dynamic top-N: N={cutoff} (smallest "
                  f"prefix that includes all {n_cc_total} Cys-Cys pairs)",
                  flush=True)

    def _cat(i, j):
        if i in cset and j in cset:
            return "C-C"
        if i in cset or j in cset:
            return "mixed"
        return "none"

    cat_color = {"C-C": "red", "mixed": "orange", "none": "grey"}
    print(f"[holmes-msa] top-{len(top_k)} (gappy-filtered) MSA-column-pair posteriors:",
          flush=True)
    n_cc = n_mix = n_none = 0
    for p, i, j in top_k:
        cat = _cat(i, j)
        if cat == "C-C":
            n_cc += 1
        elif cat == "mixed":
            n_mix += 1
        else:
            n_none += 1
        ic = "C" if i in cset else " "
        jc = "C" if j in cset else " "
        print(f"  P={p:.5f}  ({i:>2}{ic}, {j:>2}{jc})  {cat}", flush=True)
    n_total = len(top_k)
    n_total_safe = max(n_total, 1)
    print(f"[holmes-msa] top-{n_total}: C-C={n_cc} ({100*n_cc/n_total_safe:.0f}%), "
          f"mixed={n_mix} ({100*n_mix/n_total_safe:.0f}%), "
          f"none={n_none} ({100*n_none/n_total_safe:.0f}%)", flush=True)

    cys_pairs = [(c1, c2) for c1 in c_cols for c2 in c_cols if c1 < c2]
    if cys_pairs:
        cys_mass = np.mean([col_pair_post[c1, c2] for c1, c2 in cys_pairs])
        bg_all = []
        for i in range(1, L_msa + 1):
            for j in range(i + 1, L_msa + 1):
                if i in gset or j in gset:
                    continue
                if i not in cset and j not in cset:
                    bg_all.append(col_pair_post[i, j])
        bg_mass = float(np.mean(bg_all)) if bg_all else 0.0
        print(f"[holmes-msa] mean P at C-C MSA-column pairs: {cys_mass:.5f} "
              f"(vs non-C-C non-gappy bg: {bg_mass:.5f}; "
              f"ratio {cys_mass / max(bg_mass, 1e-12):.2f}x)", flush=True)

    if args.no_plot:
        return

    # ---- 7. Plot ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from matplotlib.colors import PowerNorm
    from matplotlib.gridspec import GridSpec

    # ----- AA colour mapping: LG_ORDER (sec. plot_aa_evolution.py) -----
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_aa_evolution import LG_ORDER, hue_single

    def _aa_colour(aa: str):
        idx = LG_ORDER.index(aa)
        H = hue_single(idx)
        # Saturated, full-value colour for the MSA panel.
        return mcolors.hsv_to_rgb(np.array([H, 0.85, 0.95]))

    aa_to_color = {a: tuple(_aa_colour(a)) for a in LG_ORDER}
    gap_rgb = np.array([0.92, 0.92, 0.92])

    family_label = args.family_label or args.pfam_sto.stem
    fig_w = max(9.5, 0.13 * L_msa + 2.5)
    fig_h = max(8.0, 0.13 * L_msa + 0.05 * N_seqs + 4.0)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Layout: 4 rows = [triangle, arc-gap, MSA, AA legend].
    # The arc-gap row holds the Cys "C" markers and the top-K arcs linking
    # column pairs; it provides the visual gap requested in Item 2.
    # Heights chosen so the arc panel is tall enough to show full arcs.
    gs = GridSpec(
        4, 2,
        height_ratios=[L_msa * 1.0, 18.0, max(N_seqs, 1) * 0.55, 5.0],
        width_ratios=[40, 1],
        wspace=0.05, hspace=0.06,
    )
    ax_tri = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    ax_arc = fig.add_subplot(gs[1, 0], sharex=ax_tri)
    ax_msa = fig.add_subplot(gs[2, 0], sharex=ax_tri)
    ax_legend = fig.add_subplot(gs[3, 0])

    # ----- Triangle: mask gappy columns to NaN (light grey via cmap.set_bad)
    import numpy.ma as ma
    P_show = np.full((L_msa, L_msa), np.nan, dtype=np.float64)
    for i in range(1, L_msa + 1):
        for j in range(i + 1, L_msa + 1):
            if i in gset or j in gset:
                continue  # leave NaN
            P_show[i - 1, j - 1] = col_pair_post[i, j]
    masked = ma.masked_invalid(P_show)
    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad("lightgrey")
    vmax = float(max(col_pair_post.max(), 1e-4))
    im = ax_tri.imshow(
        masked, cmap=cmap, norm=PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax),
        origin="upper", interpolation="nearest", aspect="auto",
    )
    ax_tri.set_xlim(-0.5, L_msa - 0.5)
    ax_tri.set_ylim(L_msa - 0.5, -0.5)
    ax_tri.set_yticks([])
    # Hide x tick labels on triangle and arc panel (MSA below carries them).
    ax_tri.tick_params(axis="x", labelbottom=False, labeltop=False,
                        bottom=False, top=False)
    # Vertical red dotted lines at conserved Cys columns through the triangle.
    for c in c_cols:
        ax_tri.axvline(c - 1, color="red", linestyle=":", lw=0.5, alpha=0.6)

    ax_tri.set_title(
        f"Composite-likelihood edge-pair posterior over MSA columns: "
        f"{family_label}\n"
        f"{len(cache_data['pair_list'])} sequence-pairs, "
        f"n_sweeps={cache_data['n_sweeps']}, "
        f"alpha_z={cache_data['alpha_z_ladder'][0]}; "
        f"gappy (gap_frac > {args.gappy_threshold}) shown light grey",
        fontsize=10,
    )
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("P(MSA-column-pair connected | family MSA)", fontsize=9)

    # ----- ARC panel between triangle and MSA --------------------------
    # Each top-K cell gets an arc connecting col_i to col_j (the "circuit
    # diagram" of putative covarying-column pairs), plus a circle on the
    # triangle at (col_i, col_j). Color-coded by C-C/mixed/none.
    # Cys "C" markers go in this panel near the bottom (just above MSA).
    # The arc panel spans y in [0, 1]: 0 = top of MSA, 1 = bottom of triangle.
    # Arcs are half-ellipses below the top (apex pointing UP, away from MSA).
    ax_arc.set_xlim(-0.5, L_msa - 0.5)
    ax_arc.set_ylim(0.0, 1.0)
    ax_arc.set_yticks([])
    ax_arc.tick_params(axis="x", labelbottom=False, labeltop=False,
                        bottom=False, top=False)
    ax_arc.set_facecolor("none")
    for spine in ax_arc.spines.values():
        spine.set_visible(False)
    # Proper sequence logo at the bottom of the arc panel, just above
    # the MSA. For each non-gappy column, render a Schneider-style
    # stack of AA letters: total stack height = info content (bits)
    # = log2(20) - H(column), with each letter's height = p_aa * I_c.
    # Letters coloured by AA hue. Letters are stretched independently
    # in x (one column wide) and y (proportional to p * I_c) via
    # matplotlib's TextPath + Affine2D, the standard logo idiom.
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Affine2D
    LOGO_Y_MAX = 0.22  # logo confined to [0.0, 0.22] of the arc panel;
                       # full-info column (I_c = log2(20)) fills this height
    logo_font = FontProperties(family="DejaVu Sans", weight="bold")

    # Reference glyph used to normalise vertical extent. We measure 'M'
    # because it's tall and has no descenders -- gives a consistent
    # scale factor across letters.
    ref_path = TextPath((0, 0), "M", size=1.0, prop=logo_font)
    ref_bbox = ref_path.get_extents()
    ref_h = ref_bbox.height
    ref_w = ref_bbox.width

    for ci in range(L_msa):
        if (ci + 1) in gset:
            continue
        freqs = column_freqs[ci]
        if not freqs:
            continue
        I_c = column_info[ci]
        if I_c <= 1e-6:
            continue
        # Total stack height in axes units.
        h_stack = LOGO_Y_MAX * (I_c / LOG2_20)
        # Sort AAs in stack: most-frequent at top (Schneider convention),
        # so we stack bottom-up by descending frequency.
        sorted_aas = sorted(freqs.items(), key=lambda x: x[1])
        y_cursor = 0.0
        for aa, p in sorted_aas:
            if p <= 0:
                continue
            letter_h = h_stack * p
            if letter_h < 1e-4:
                continue
            if aa not in aa_to_color:
                continue
            letter_color = aa_to_color[aa]
            # Render the letter via TextPath, then scale + translate.
            tp = TextPath((0, 0), aa, size=1.0, prop=logo_font)
            bbox = tp.get_extents()
            # Move glyph so its bbox starts at (0, 0).
            t = (Affine2D()
                 .translate(-bbox.x0, -bbox.y0)
                 .scale(0.92 / max(bbox.width, 1e-6),
                         letter_h / max(bbox.height, 1e-6))
                 .translate(ci - 0.46, y_cursor))
            patch = mpatches.PathPatch(t.transform_path(tp),
                                       facecolor=letter_color,
                                       edgecolor="none", lw=0.0,
                                       zorder=3)
            ax_arc.add_patch(patch)
            y_cursor += letter_h
    for c in c_cols:
        ax_arc.axvline(c - 1, color="red", linestyle=":", lw=0.5, alpha=0.4)

    # Arcs + circles for top-K, coloured by posterior probability via
    # the same colormap as the triangle heatmap (so a brighter arc
    # = a higher-P column-pair). Stagger arc heights by rank so they
    # don't all overlap visually.
    y_base = 0.10
    max_arc_top = 0.92
    # Sort top-K by mid-x to stagger heights left-to-right.
    ranked = sorted(top_k, key=lambda t: (t[1] + t[2]))
    n = len(ranked)
    # P-to-colour mapping: re-use the triangle's vmin/vmax range so
    # arc colour matches the heatmap colour at the same cell.
    p_norm = PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax)
    p_cmap = matplotlib.colormaps.get_cmap(args.cmap)
    for rank, (p, i, j) in enumerate(ranked):
        col = p_cmap(p_norm(p))
        # Circle on triangle at (j-1, i-1): note imshow has x=column, y=row.
        circ = mpatches.Circle((j - 1, i - 1), radius=1.5,
                               fill=False, edgecolor=col, lw=1.2, alpha=0.95)
        ax_tri.add_patch(circ)
        x_mid = (i + j) / 2 - 1
        width = (j - i)
        h_frac = 0.30 + 0.55 * (rank / max(n - 1, 1))
        h_total = (max_arc_top - y_base) * h_frac
        arc = mpatches.Arc((x_mid, y_base), width, 2 * h_total,
                           angle=0.0, theta1=0.0, theta2=180.0,
                           edgecolor=col, lw=0.9, alpha=0.95)
        ax_arc.add_patch(arc)

    # ----- MSA panel: use LG_ORDER colour mapping ----------------------
    msa_rgb = np.full((N_seqs, L_msa, 3), 1.0)
    for r, aln in enumerate(alignments):
        for c, ch in enumerate(aln):
            cu = ch.upper()
            if cu.isalpha() and cu in aa_to_color:
                msa_rgb[r, c, :] = aa_to_color[cu]
            elif cu == "X" or cu == "B" or cu == "Z" or cu == "J" or cu == "U" or cu == "O":
                msa_rgb[r, c, :] = (0.5, 0.5, 0.5)
            else:
                msa_rgb[r, c, :] = gap_rgb
    ax_msa.imshow(msa_rgb, aspect="auto", interpolation="nearest")
    ax_msa.set_xlabel("MSA column", fontsize=9)
    ax_msa.set_ylabel(f"{N_seqs} sequences", fontsize=9)
    ax_msa.set_yticks([])
    # Red dotted column lines at conserved Cys columns.
    for c in c_cols:
        ax_msa.axvline(c - 0.5, color="red", linestyle=":", lw=0.5, alpha=0.5)
    # Ticks on the column axis (bottom only).
    step = max(L_msa // 10, 1)
    xticks = list(range(0, L_msa, step))
    ax_msa.set_xticks(xticks)
    ax_msa.set_xticklabels([str(t + 1) for t in xticks], fontsize=8)
    ax_msa.tick_params(axis="x", labelbottom=True, labeltop=False,
                        bottom=True, top=False)

    # ----- AA legend panel: 20 swatches + letters in LG_ORDER ----------
    ax_legend.set_xlim(0, 20)
    ax_legend.set_ylim(0, 1)
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    for spine in ax_legend.spines.values():
        spine.set_visible(False)
    for k_idx, aa in enumerate(LG_ORDER):
        col = aa_to_color[aa]
        rect = mpatches.Rectangle((k_idx + 0.05, 0.30), 0.9, 0.45,
                                  facecolor=col, edgecolor="black", lw=0.4)
        ax_legend.add_patch(rect)
        # Letters centred vertically in the swatch (y = 0.525 = midpoint
        # of [0.30, 0.75]), not at the bottom as before.
        ax_legend.text(k_idx + 0.5, 0.525, aa, ha="center", va="center",
                       fontsize=8, weight="bold")
    ax_legend.text(10.0, 0.05,
                   f"AA color key (LG08-PCA order; gap shown light grey)",
                   ha="center", va="bottom", fontsize=8)

    # Save -- skip tight_layout because we manually control spacing via
    # gridspec hspace (Item 2 needs a clean gap between triangle and MSA).
    pdf_path = Path(str(args.out) + ".pdf")
    png_path = Path(str(args.out) + ".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[holmes-msa] wrote {pdf_path} and {png_path}", flush=True)


if __name__ == "__main__":
    main()
