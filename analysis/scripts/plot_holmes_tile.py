#!/usr/bin/env python3
"""Generate a (Lx+Ly) x (Lx+Ly) Holmes-tile composite figure on a BAliBase pair.

Modeled on Figure 13 of Holmes 2004 ("A probabilistic model for the
evolution of RNA structure", BMC Bioinformatics 5:166).

Layout (rows/cols = X positions 1..Lx then Y positions 1..Ly):

  (a) X-X lower triangle    : single-X edge MCMC marginal P(edge | x only)
  (b) Y-Y lower triangle    : single-Y edge MCMC marginal P(edge | y only)
  (c) Y-X below-diag block  : TKF92-LG08 alignment marginal (no edges)
                              -- column j of (c) is the X-axis at sequence
                              position 1..Lx, row i is the Y-axis position
  (d) X-X upper triangle    : Infinite Pair HMM X-X edge marginal --
                              project joint sampler edges onto X positions
  (e) Y-Y upper triangle    : Infinite Pair HMM Y-Y edge marginal --
                              project joint sampler edges onto Y positions
  (f) X-Y above-diag block  : Infinite Pair HMM alignment marginal Q'

Grid lines at i = Lx and j = Lx separate the X and Y blocks.
Cys positions on each sequence are marked with red dotted gridlines.

Defaults assume the BAliBase BB12032 (snake 3-finger toxin) family pair
0 = (1drs_, 1cb9_A) and the released K=4 checkpoint at
``results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz``.

Usage
-----

    python analysis/scripts/plot_holmes_tile.py \
        --family BB12032 --pair-i 0 --pair-j 1 \
        --ckpt results/K4-emwarm-top1000-2026-05-09/_best_chkpt/state.npz \
        --out math-paper/figures/holmes_tile_BB12032

Outputs both .pdf and .png at the chosen prefix.
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
sys.path.insert(0, str(REPO / "analysis" / "scripts"))


# ---------------------------------------------------------------------------
# Helpers (copied / adapted from sweep_infinite_phmm_balibase.py).
# ---------------------------------------------------------------------------

import string


# The K=4 model lives on the **ACDE** (alphabetical) AA alphabet --
# this matches the training pipeline (tkfdp.lg08 PI_LG08_J / S_LG08_F81_J
# are ACDE-ordered) and the rate matrix returned by
# tkfmixdom.jax.core.protein.rate_matrix_lg() (which is ACDE-ordered
# despite that module's mislabeled AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
# constant; the matrix itself is ACDE -- verify with pi[1]=0.0129=C).
# Encoding pre-2026-05-15 used ARND order, which silently treated
# cysteines as F (phe) and thus largely destroyed the C-C coupling
# signal.
def _aa_to_int_dict():
    aa = "ACDEFGHIKLMNPQRSTVWY"
    d = {c: i for i, c in enumerate(aa)}
    for c in string.ascii_uppercase:
        d.setdefault(c, 20)
    return d


def parse_fasta(path: Path):
    AA_TO_INT = _aa_to_int_dict()
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
                    s = "".join(seq)
                    arr = np.array(
                        [AA_TO_INT.get(c.upper(), 20) for c in s if c.isalpha()],
                        dtype=np.int32,
                    )
                    out.append((name, arr, s))
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name is not None:
            s = "".join(seq)
            arr = np.array(
                [AA_TO_INT.get(c.upper(), 20) for c in s if c.isalpha()],
                dtype=np.int32,
            )
            out.append((name, arr, s))
    return out


def _build_k4_state(ckpt_path: Path, alpha_H: float = 1.0):
    """Reload the released K=4 checkpoint as a minimal State for the sampler."""
    from tkfdp.potts_dp import PottsDPState

    d = np.load(ckpt_path, allow_pickle=True)
    pi_class = np.asarray(d["pi_class"], dtype=np.float32)
    atoms = np.asarray(d["potts_atoms"], dtype=np.float32)
    assignments = np.asarray(d["potts_assignments"], dtype=np.int64)
    counts = np.asarray(d["potts_counts"], dtype=np.int64)
    K_c, A = pi_class.shape
    potts_dp = PottsDPState(
        K_c=K_c, A=A, atoms=atoms,
        assignments=assignments, counts=counts, alpha_H=alpha_H,
    )

    class _State:
        pass

    s = _State()
    s.K_c = K_c
    s.A = A
    s.pi_class = pi_class
    s.potts_dp = potts_dp
    return s


def _build_boost_state(x_seq, y_seq, ins_rate, del_rate, ext, Q_lg, pi_lg,
                        state, *, pi_c=None, pair_background='lg08'):
    """Build the PairBoostState for a single (x, y) pair.

    Pass pi_c (empirical class prior from training counts) and
    pair_background ('lg08' or 'per_class') so the MCMC chain's
    block_likelihoods.build_singlet_emission + build_M_tensor use the
    canonical convention. Without these the chain falls back to plain
    LG08 baseline + gamma-weighted M (relic), which biases results.
    """
    import jax
    import jax.numpy as jnp

    from tkfdp.coupled_annealing import build_boost_state
    from tkfmixdom.jax.dp.hmm import _pad_to_bin, _pad_seq
    from tkfmixdom.jax.tree.fsa_anneal import _pairwise_posteriors_tkf92_jax

    x = jnp.asarray(x_seq, dtype=jnp.int32)
    y = jnp.asarray(y_seq, dtype=jnp.int32)
    Lx, Ly = int(x.shape[0]), int(y.shape[0])
    Lx_pad, Ly_pad = _pad_to_bin(Lx), _pad_to_bin(Ly)
    x_pad, y_pad = _pad_seq(x, Lx_pad), _pad_seq(y, Ly_pad)
    mp_pad, tau_opt, _ = _pairwise_posteriors_tkf92_jax(
        x_pad, y_pad, jnp.int32(Lx), jnp.int32(Ly),
        jnp.float64(ins_rate), jnp.float64(del_rate), jnp.float64(ext),
        jnp.asarray(Q_lg), jnp.asarray(pi_lg),
    )
    pair_post = {(0, 1): np.asarray(mp_pad)[:Lx, :Ly].astype(np.float32)}
    pair_taus = {(0, 1): float(tau_opt)}
    boost_states = build_boost_state(
        pair_post, pair_taus, [x_seq, y_seq], state,
        pi_c=pi_c, pair_background=pair_background)
    return boost_states[(0, 1)], float(tau_opt)


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------


def parse_stockholm(path: Path):
    """Parse a Pfam Stockholm file. Returns a list of (name, seq_int, raw_str)
    matching the parse_fasta output convention. Raw_str has gaps stripped."""
    AA_TO_INT = _aa_to_int_dict()
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
    out = []
    for n in names:
        aligned = "".join(seqs[n])
        raw = "".join(c for c in aligned if c.isalpha())
        arr = np.array([AA_TO_INT.get(c.upper(), 20) for c in raw],
                        dtype=np.int32)
        out.append((n, arr, raw))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="BB12032",
                    help="BAliBase bali3pdbm family name (used unless --pfam-sto).")
    ap.add_argument("--pfam-sto", type=Path, default=None,
                    help="Optional Stockholm-format Pfam family MSA. If "
                    "supplied, the script loads sequences from here (instead "
                    "of BAliBase) and the --pair-name-{x,y} arguments select "
                    "which sequences to use as the pair.")
    ap.add_argument("--pair-name-x", default=None,
                    help="Sequence name for X (only used with --pfam-sto).")
    ap.add_argument("--pair-name-y", default=None,
                    help="Sequence name for Y (only used with --pfam-sto).")
    ap.add_argument("--pair-i", type=int, default=0)
    ap.add_argument("--pair-j", type=int, default=1)
    ap.add_argument("--ckpt", type=Path,
                    default=REPO / "results" / "K4-emwarm-top1000-2026-05-09" /
                    "_best_chkpt" / "state.npz",
                    help="Released K=4 checkpoint state.npz")
    ap.add_argument("--balibase-root", type=Path,
                    default=Path.home() / "bio-datasets" / "data" /
                    "balibase" / "bali3pdbm")
    ap.add_argument("--tkf92-params",
                    default=Path.home() / "tkf-mixdom" / "python" /
                    "experiments" / "tkf92_fitted_params.json")
    ap.add_argument("--alpha-z-ladder", default="100,250,700,2000,1e4",
                    help="Replica-exchange ladder for the joint sampler.")
    ap.add_argument("--alpha-z-single", type=float, default=100.0,
                    help="alpha_z for the single-seq baseline samplers.")
    ap.add_argument("--alpha-z-single-ladder", default="100,500,2000,1e4",
                    help="Replica-exchange ladder for the single-seq sampler.")
    ap.add_argument("--n-sweeps", type=int, default=2000)
    ap.add_argument("--n-burnin", type=int, default=500)
    ap.add_argument("--n-sweeps-single", type=int, default=10000,
                    help="Single-seq sampler is much cheaper; run longer.")
    ap.add_argument("--n-burnin-single", type=int, default=1000)
    ap.add_argument("--swap-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="math-paper/figures/holmes_tile_BB12032",
                    type=Path,
                    help="Output prefix; both .pdf and .png are written.")
    ap.add_argument("--cache-json", type=Path, default=None,
                    help="Optional JSON cache path for the heavy sampler "
                    "outputs. If it exists, load instead of re-running.")
    ap.add_argument("--cmap", default="magma",
                    help="Matplotlib colormap (default: magma).")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip the plot; just compute and write the cache.")
    ap.add_argument("--no-condition-on-edges", dest="condition_on_edges",
                    action="store_false", default=True,
                    help="Disable conditioning the edge-marginal panels on "
                    "|E| >= 1. Default: ON (recommended). With conditioning, "
                    "each edge panel is renormalised by P(|E| >= 1 | data) "
                    "estimated as 1 - exp(-E[|E|]) (Poisson), so cell values "
                    "have comparable scale across pairs and prior-strength "
                    "settings. Use --no-condition-on-edges to recover the "
                    "raw marginal P(cell is endpoint) used by older figures.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-panel renormalisation diagnostics.")
    ap.add_argument("--disulfide-x", type=str, default=None,
                    help="Comma-separated 1-based seq-pos pairs of known "
                    "disulfide bonds in X, e.g. '1:15,3:25,28:37,40:55'. "
                    "Each bond is drawn as a cyan circle in panels (a) and "
                    "(d).")
    ap.add_argument("--disulfide-y", type=str, default=None,
                    help="Same as --disulfide-x but for Y. Drawn in "
                    "panels (b) and (e).")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load sequences -------------------------------------------------
    if args.pfam_sto is not None:
        # Pfam mode: load Stockholm, optionally pick by sequence name.
        fasta_all = parse_stockholm(args.pfam_sto)
        name_to_idx = {n: idx for idx, (n, _, _) in enumerate(fasta_all)}
        if args.pair_name_x is not None and args.pair_name_y is not None:
            try:
                args.pair_i = name_to_idx[args.pair_name_x]
                args.pair_j = name_to_idx[args.pair_name_y]
            except KeyError as e:
                raise RuntimeError(
                    f"Sequence {e!r} not in Stockholm file {args.pfam_sto}")
        fasta = fasta_all
    else:
        fa_path = args.balibase_root / "in" / args.family
        fasta = parse_fasta(fa_path)
    if max(args.pair_i, args.pair_j) >= len(fasta):
        raise RuntimeError(
            f"Pair indices ({args.pair_i}, {args.pair_j}) exceed family size "
            f"{len(fasta)} for {args.family}")
    name_x, x_seq, raw_x = fasta[args.pair_i]
    name_y, y_seq, raw_y = fasta[args.pair_j]
    Lx, Ly = int(x_seq.shape[0]), int(y_seq.shape[0])
    print(f"[holmes-tile] {args.family} pair=({args.pair_i},{args.pair_j}) "
          f"= ({name_x}:{Lx}, {name_y}:{Ly})", flush=True)

    cys_x = [i + 1 for i, a in enumerate(raw_x) if a.upper() == "C"]
    cys_y = [i + 1 for i, a in enumerate(raw_y) if a.upper() == "C"]
    print(f"[holmes-tile]   cys_x={cys_x}\n[holmes-tile]   cys_y={cys_y}",
          flush=True)

    # ---- 2. TKF92 params + checkpoint state --------------------------------
    fitted = json.loads(Path(args.tkf92_params).read_text())
    ins_rate = float(fitted["ins_rate"])
    del_rate = float(fitted["del_rate"])
    ext = float(fitted["ext_rate"])
    print(f"[holmes-tile] TKF92 indel: ins={ins_rate:.5f} "
          f"del={del_rate:.5f} ext={ext:.4f}", flush=True)

    from tkfmixdom.jax.core.protein import rate_matrix_lg

    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg = np.asarray(Q_lg)
    pi_lg = np.asarray(pi_lg)
    state = _build_k4_state(args.ckpt)
    print(f"[holmes-tile] K4 ckpt loaded: K_c={state.K_c}, A={state.A}",
          flush=True)

    # Empirical class prior (canonical for this checkpoint).
    from tkfdp.block_likelihoods import empirical_pi_c_from_checkpoint
    try:
        pi_c_emp = empirical_pi_c_from_checkpoint(str(args.ckpt))
        print(f"[holmes-tile] empirical pi_c = "
              f"{[f'{x:.3f}' for x in pi_c_emp]}", flush=True)
    except Exception as e:
        print(f"[holmes-tile] warn: could not load empirical pi_c ({e}); "
              f"falling back to uniform 1/K_c", flush=True)
        pi_c_emp = None

    # ---- 3. Build boost state for the pair --------------------------------
    bs, tau = _build_boost_state(
        x_seq, y_seq, ins_rate, del_rate, ext, Q_lg, pi_lg, state,
        pi_c=pi_c_emp, pair_background='lg08')
    print(f"[holmes-tile] tau_opt={tau:.4f}", flush=True)

    # ---- 4. Cache check ----------------------------------------------------
    cache_data = None
    if args.cache_json is not None and args.cache_json.exists():
        try:
            cache_data = json.loads(args.cache_json.read_text())
            if (cache_data["Lx"] == Lx and cache_data["Ly"] == Ly
                    and cache_data["family"] == args.family
                    and tuple(cache_data["pair"]) == (args.pair_i, args.pair_j)):
                print(f"[holmes-tile] cache hit: {args.cache_json}", flush=True)
            else:
                cache_data = None
        except Exception as e:
            print(f"[holmes-tile] cache load failed: {e}", flush=True)
            cache_data = None

    if cache_data is None:
        # ---- 5. Run joint sampler (RE) ------------------------------------
        ladder = [float(x) for x in args.alpha_z_ladder.split(",")]
        from tkfdp.mcmc_infinite_phmm import mcmc_corrected_posterior

        t0 = time.time()
        print(f"[holmes-tile] joint sampler (RE ladder={ladder}, "
              f"n_sweeps={args.n_sweeps}, n_burnin={args.n_burnin}) ...",
              flush=True)
        Q_prime, _, Q_baseline, log_F0, joint_diag = mcmc_corrected_posterior(
            x_seq=x_seq, y_seq=y_seq, t=tau,
            ins_rate=ins_rate, del_rate=del_rate, ext=ext,
            Q_lg=Q_lg, pi_lg=pi_lg, boost_state=bs,
            alpha_z=ladder[0],
            alpha_z_ladder=ladder, swap_every=args.swap_every,
            n_sweeps=args.n_sweeps, n_burnin=args.n_burnin,
            n_chains=1, k_max=-1, seed=args.seed,
            init_mode="viterbi",
            verbose=True,
        )
        joint_dt = time.time() - t0
        print(f"[holmes-tile]   joint sampler done in {joint_dt:.1f}s",
              flush=True)
        cold_diag = joint_diag["per_rung"][0]
        nrec = int(cold_diag.n_recorded_for_edges)
        print(f"[holmes-tile]   cold-rung n_recorded={nrec}", flush=True)

        # XX projection: build symmetric (Lx+1, Lx+1) dense matrix
        Pxx_joint = np.zeros((Lx + 1, Lx + 1), dtype=np.float64)
        for (i1, i2), c in cold_diag.edge_pair_x_counts.items():
            v = c / max(nrec, 1)
            Pxx_joint[i1, i2] = v
            Pxx_joint[i2, i1] = v
        Pyy_joint = np.zeros((Ly + 1, Ly + 1), dtype=np.float64)
        for (j1, j2), c in cold_diag.edge_pair_y_counts.items():
            v = c / max(nrec, 1)
            Pyy_joint[j1, j2] = v
            Pyy_joint[j2, j1] = v

        # ---- 6. Run single-seq edge MCMC on X --------------------------
        from tkfdp.single_seq_edge_mcmc import (
            precompute_single_seq_setup,
            run_single_seq_replica_exchange,
        )

        ladder_s = [float(x) for x in args.alpha_z_single_ladder.split(",")]
        print(f"[holmes-tile] single-seq X sampler (RE ladder={ladder_s}, "
              f"n_sweeps={args.n_sweeps_single}, n_burnin={args.n_burnin_single})",
              flush=True)
        setup_x = precompute_single_seq_setup(bs, axis="x",
                                               alpha_z=ladder_s[0])
        t0 = time.time()
        Pxx_single, single_x_diag = run_single_seq_replica_exchange(
            setup_x, alpha_z_ladder=ladder_s,
            n_sweeps=args.n_sweeps_single,
            n_burnin=args.n_burnin_single,
            n_edge_moves_per_sweep=8, seed=args.seed + 1,
            swap_every=args.swap_every, verbose=False,
        )
        print(f"[holmes-tile]   single X done in {time.time() - t0:.1f}s",
              flush=True)
        # ---- 7. Run single-seq edge MCMC on Y --------------------------
        print(f"[holmes-tile] single-seq Y sampler (RE)", flush=True)
        setup_y = precompute_single_seq_setup(bs, axis="y",
                                               alpha_z=ladder_s[0])
        t0 = time.time()
        Pyy_single, single_y_diag = run_single_seq_replica_exchange(
            setup_y, alpha_z_ladder=ladder_s,
            n_sweeps=args.n_sweeps_single,
            n_burnin=args.n_burnin_single,
            n_edge_moves_per_sweep=8, seed=args.seed + 2,
            swap_every=args.swap_every, verbose=False,
        )
        print(f"[holmes-tile]   single Y done in {time.time() - t0:.1f}s",
              flush=True)

        cache_data = {
            "family": args.family,
            "pair": [args.pair_i, args.pair_j],
            "name_x": name_x, "name_y": name_y,
            "raw_x": raw_x, "raw_y": raw_y,
            "Lx": Lx, "Ly": Ly,
            "cys_x": cys_x, "cys_y": cys_y,
            "tau": tau,
            "Q_baseline": np.asarray(Q_baseline).tolist(),
            "Q_prime": np.asarray(Q_prime).tolist(),
            "Pxx_joint": Pxx_joint.tolist(),
            "Pyy_joint": Pyy_joint.tolist(),
            "Pxx_single": Pxx_single.tolist(),
            "Pyy_single": Pyy_single.tolist(),
            "joint_nrec": nrec,
            "single_x_nrec": single_x_diag["per_rung"][0].n_recorded_for_edges,
            "single_y_nrec": single_y_diag["per_rung"][0].n_recorded_for_edges,
            "alpha_z_ladder": ladder,
            "alpha_z_single_ladder": ladder_s,
            "n_sweeps": args.n_sweeps, "n_burnin": args.n_burnin,
            "n_sweeps_single": args.n_sweeps_single,
            "n_burnin_single": args.n_burnin_single,
        }
        if args.cache_json is not None:
            args.cache_json.parent.mkdir(parents=True, exist_ok=True)
            args.cache_json.write_text(json.dumps(cache_data))
            print(f"[holmes-tile] wrote cache to {args.cache_json}", flush=True)
    else:
        # Reload arrays from cache_data.
        Q_baseline = np.asarray(cache_data["Q_baseline"])
        Q_prime = np.asarray(cache_data["Q_prime"])
        Pxx_joint = np.asarray(cache_data["Pxx_joint"])
        Pyy_joint = np.asarray(cache_data["Pyy_joint"])
        Pxx_single = np.asarray(cache_data["Pxx_single"])
        Pyy_single = np.asarray(cache_data["Pyy_single"])

    if args.no_plot:
        return

    # ---- Optional renormalisation: condition on |E| >= 1 -----------------
    # P(cell is edge endpoint | data, |E| >= 1)
    #   = P(cell is edge endpoint | data) / P(|E| >= 1 | data).
    #
    # Cleanly defined; renormalises every edge-marginal panel by a single
    # scalar Z = P(|E| >= 1 | data). With existing caches that don't store
    # the empirical n_samples_with_edges, we fall back to the Poisson
    # approximation Z ~= 1 - exp(-lambda) where lambda = E[|E|] is the
    # average number of edges per sample, recovered from the per-cell
    # array as `array.sum() / 2` (every edge contributes 2 endpoints on
    # each axis). For sparse-edge regimes this is a tight approximation;
    # for higher densities it slightly under-estimates Z (and so over-
    # estimates the conditional posterior). When proper n_samples_with_edges
    # is available it should be preferred.
    def _condition_on_edges(arr: np.ndarray, label: str) -> np.ndarray:
        # arr is a (L+1, L+1) matrix of marginal P(cell is endpoint).
        # mean_E = E[|E|] (expected number of edges per MCMC sample).
        # Each edge contributes 2 endpoints on the X (or Y) axis, so
        # mean_E = arr.sum() / 2 by linearity of expectation.
        mean_E = float(arr.sum()) / 2.0
        if mean_E <= 0.0:
            return arr  # nothing to condition on
        # Poisson approximation: P(|E| >= 1) ~= 1 - exp(-mean_E).
        Z = 1.0 - np.exp(-mean_E)
        if Z <= 0.0 or Z >= 1.0 - 1e-9:
            # Already saturated; conditional ~= unconditional.
            return arr
        scaled = arr / Z
        if args.verbose:
            print(f"[holmes-tile] {label}: mean_E={mean_E:.3f}, "
                  f"Z=P(|E|>=1)~={Z:.3f}, scale=1/Z={1.0/Z:.3f}",
                  flush=True)
        return scaled

    if args.condition_on_edges:
        Pxx_joint  = _condition_on_edges(Pxx_joint,  "Pxx_joint")
        Pyy_joint  = _condition_on_edges(Pyy_joint,  "Pyy_joint")
        Pxx_single = _condition_on_edges(Pxx_single, "Pxx_single")
        Pyy_single = _condition_on_edges(Pyy_single, "Pyy_single")

    # ---- 8. Assemble the composite tile -----------------------------------
    # Layout: rows = X(1..Lx) then Y(1..Ly); cols = X(1..Lx) then Y(1..Ly).
    # Index convention: 0-based numpy. Position p_x in 1..Lx maps to row p_x - 1.
    # Position p_y in 1..Ly maps to row Lx + p_y - 1.
    L = Lx + Ly
    tile = np.zeros((L, L), dtype=np.float64)

    # X-X block: rows/cols 0..Lx-1. Split into upper (d, joint) and lower (a, single).
    # Pxx_joint[i, j] with i, j in 1..Lx -- use full matrix (symmetric).
    # For tiling: we want the LOWER triangle (i > j) of the X-X block to be (a)
    # = single, and UPPER (i < j) to be (d) = joint.
    for i in range(1, Lx + 1):
        for j in range(1, Lx + 1):
            if i == j:
                tile[i - 1, j - 1] = 0.0  # diagonal
            elif i > j:  # lower triangle -> (a) single-X
                tile[i - 1, j - 1] = Pxx_single[i, j]
            else:        # upper triangle -> (d) joint X-X
                tile[i - 1, j - 1] = Pxx_joint[i, j]

    # Y-Y block: rows/cols Lx..Lx+Ly-1. Lower (b) = single-Y, upper (e) = joint Y-Y.
    for i in range(1, Ly + 1):
        for j in range(1, Ly + 1):
            row = Lx + i - 1
            col = Lx + j - 1
            if i == j:
                tile[row, col] = 0.0
            elif i > j:
                tile[row, col] = Pyy_single[i, j]
            else:
                tile[row, col] = Pyy_joint[i, j]

    # (c) Y-X block: rows Lx..Lx+Ly-1, cols 0..Lx-1. This is the alignment
    # marginal Q_baseline with TKF92-LG08 (no edges). Q_baseline has shape
    # (Lx, Ly); the (i, j) entry is P(X[i] aligned to Y[j]). Map to (row, col)
    # = (Lx + j - 1, i - 1). 1-based to 0-based: tile[Lx + jp, ip] = Q_baseline[ip, jp].
    for ip in range(Lx):
        for jp in range(Ly):
            tile[Lx + jp, ip] = Q_baseline[ip, jp]

    # (f) X-Y block: rows 0..Lx-1, cols Lx..Lx+Ly-1. Joint sampler alignment
    # marginal Q_prime (with edges). Q_prime shape (Lx, Ly).
    for ip in range(Lx):
        for jp in range(Ly):
            tile[ip, Lx + jp] = Q_prime[ip, jp]

    # ---- 9. Plot ----------------------------------------------------------
    # Split the composite tile into two scale groups so the rare strong
    # alignment cells in (c)/(f) don't suppress the much weaker edge
    # posterior signal in (a),(b),(d),(e):
    #   tile_align : (c) + (f), uses vmax = 1.0 (alignment posterior).
    #   tile_edges : (a) + (b) + (d) + (e), uses vmax = edge_max.
    # We render TWO imshow layers using masked arrays.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy.ma as ma
    from matplotlib.colors import PowerNorm

    align_mask = np.zeros((L, L), dtype=bool)
    align_mask[Lx:, :Lx] = True   # (c)
    align_mask[:Lx, Lx:] = True   # (f)
    edge_mask = ~align_mask
    # Diagonal blocks for edge_mask: include all cells of X-X and Y-Y
    # blocks. The actual diagonals are zero so the colormap absorbs them.
    # Edge max from the edge panels only.
    edge_panel = tile.copy()
    edge_panel[align_mask] = 0.0
    edge_max = float(max(edge_panel.max(), 1e-4))
    edge_max_round = max(edge_max, 0.05)  # at least 5% for color range

    tile_align = ma.array(tile, mask=~align_mask)
    tile_edges = ma.array(tile, mask=align_mask)

    fig, ax = plt.subplots(figsize=(11.5, 10.5))
    # PowerNorm with gamma < 1 brightens low values.
    norm_align = PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)
    norm_edges = PowerNorm(gamma=0.6, vmin=0.0, vmax=edge_max_round)
    im_align = ax.imshow(tile_align, cmap=args.cmap, norm=norm_align,
                          origin="upper", interpolation="nearest")
    im_edges = ax.imshow(tile_edges, cmap=args.cmap, norm=norm_edges,
                          origin="upper", interpolation="nearest")

    # Major separator gridlines at i=Lx and j=Lx (between X- and Y-blocks).
    ax.axhline(Lx - 0.5, color="white", lw=1.5, alpha=0.7)
    ax.axvline(Lx - 0.5, color="white", lw=1.5, alpha=0.7)

    # Cys markers: red dotted gridlines through every Cys position on each axis.
    cys_lines_x = [c - 1 for c in cys_x]               # X positions
    cys_lines_y = [Lx + c - 1 for c in cys_y]          # Y positions in tile coords
    for r in cys_lines_x + cys_lines_y:
        ax.axhline(r, color="red", linestyle=":", lw=0.5, alpha=0.6)
        ax.axvline(r, color="red", linestyle=":", lw=0.5, alpha=0.6)

    # ------------------------------------------------------------------
    # AA-letter sequence labels along all four borders, coloured by AA hue.
    # ------------------------------------------------------------------
    # The plot's pixel coordinate system: row i, col j (origin upper).
    # X-block: rows 0..Lx-1, cols 0..Lx-1; X letter at position p (1-based)
    #   in seq lives at row/col p - 1.
    # Y-block: rows Lx..Lx+Ly-1, cols Lx..Lx+Ly-1; Y letter at position p
    #   lives at row/col Lx + p - 1.
    #
    # For each AA letter (LG-PCA ordering for the hue palette), we plot a
    # bold letter just outside the relevant border at the cell centre. The
    # hue is hue_single(idx_lg) from plot_aa_evolution.py.
    #
    # Numeric ticks (every 10 positions) replace the prior X:N/Y:N markers.
    sys.path.insert(0, str(REPO / "analysis" / "scripts"))
    from plot_aa_evolution import hue_single, aa_to_lg_idx
    import matplotlib.colors as mcolors

    def _aa_color(aa: str) -> str:
        try:
            return mcolors.hsv_to_rgb((hue_single(aa_to_lg_idx(aa)), 0.85, 0.75))
        except ValueError:
            return (0.4, 0.4, 0.4)

    # Border offsets in data coordinates (in pixels).
    # We pick offsets that put the letters just outside the heatmap.
    # The default tight_layout will adjust the figure to fit.
    L_total = Lx + Ly
    # Per-letter font size: roughly the pixel size of one cell in the figure.
    # The figure is ~11.5x10.5 inches; the heatmap takes most of that.
    # With L_total pixels, each cell is roughly 0.9 / L_total inches; one
    # font point is ~1/72 inch, so font_pt ~ 70 * 0.9 / L_total.
    letter_pt = max(3.0, min(8.0, 600.0 / max(L_total, 1)))
    # Offset (in data coordinates) of letter from heatmap edge.
    off = max(1.0, L_total / 80.0)

    def _draw_letter(ax, x_data, y_data, letter, rotation=0,
                      ha="center", va="center"):
        col = _aa_color(letter)
        ax.text(x_data, y_data, letter, color=col,
                 fontsize=letter_pt, ha=ha, va=va,
                 rotation=rotation, family="monospace", weight="bold",
                 clip_on=False)

    # LEFT border: write X letters down rows 0..Lx-1, Y letters down rows
    # Lx..Lx+Ly-1. Each letter sits at x = -off, y = position.
    for i, c in enumerate(raw_x):
        _draw_letter(ax, -off, i, c, rotation=0, ha="right", va="center")
    for j, c in enumerate(raw_y):
        _draw_letter(ax, -off, Lx + j, c, rotation=0, ha="right", va="center")

    # RIGHT border: x = L_total + off.
    for i, c in enumerate(raw_x):
        _draw_letter(ax, L_total - 0.5 + off, i, c,
                     rotation=0, ha="left", va="center")
    for j, c in enumerate(raw_y):
        _draw_letter(ax, L_total - 0.5 + off, Lx + j, c,
                     rotation=0, ha="left", va="center")

    # TOP border: y = -off, X letters along cols 0..Lx-1 then Y letters
    # along cols Lx..L_total-1.
    for i, c in enumerate(raw_x):
        _draw_letter(ax, i, -off, c, rotation=90, ha="center", va="bottom")
    for j, c in enumerate(raw_y):
        _draw_letter(ax, Lx + j, -off, c, rotation=90, ha="center", va="bottom")

    # BOTTOM border: y = L_total + off.
    for i, c in enumerate(raw_x):
        _draw_letter(ax, i, L_total - 0.5 + off, c,
                     rotation=90, ha="center", va="top")
    for j, c in enumerate(raw_y):
        _draw_letter(ax, Lx + j, L_total - 0.5 + off, c,
                     rotation=90, ha="center", va="top")

    # Numeric position markers every 10 positions, faint, off the inner
    # heatmap so they don't fight the letter colouring.
    num_ticks_x = list(range(0, Lx, 10))
    num_ticks_y = list(range(0, Ly, 10))
    tick_positions = num_ticks_x + [Lx + j for j in num_ticks_y]
    tick_labels = [f"X{p+1}" for p in num_ticks_x] + [f"Y{p+1}" for p in num_ticks_y]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, ha="center",
                       fontsize=max(4.0, letter_pt * 0.7), color="0.5")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=max(4.0, letter_pt * 0.7),
                       color="0.5")
    # Push tick labels further out to leave room for the letter border.
    ax.tick_params(axis="x", which="both", pad=letter_pt * 2.5)
    ax.tick_params(axis="y", which="both", pad=letter_pt * 2.5)

    # Block-region labels in the corners of their segments. Below-diagonal
    # captions (a, b, c) go to the bottom-left of their region; above-
    # diagonal captions (d, e, f) go to the top-right. With origin='upper'
    # the visual bottom of a region is at the LARGER y coordinate.
    pad = 1.0
    # ~1 em offset in data units for the stationary/covariant labels:
    # roughly 1 em ~ fontsize-pt / 72 in. -> for an (Lx+Ly)-cell axis on
    # a 10.5 in figure that is (Lx+Ly)/10.5 cells per inch.
    em = max(2.0, (Lx + Ly) * 11 / 72 / 10.5)
    bl_kwargs = dict(color="white", fontsize=11, ha="left", va="bottom",
                     weight="bold", alpha=0.85)
    tr_kwargs = dict(color="white", fontsize=11, ha="right", va="top",
                     weight="bold", alpha=0.85)
    # Per user direction: stationary labels ((a), (b)) move UP+RIGHT by 1 em
    # toward the center of their block; covariant labels ((d), (e)) move
    # DOWN+LEFT by 1 em toward the center of theirs. With origin='upper'
    # "up" = smaller y, "down" = larger y.
    # (a) X-X lower triangle -> bottom-left of X-X block, shifted +x / -y.
    ax.text(pad + em, Lx - pad - em, "(a) X (stationary)", **bl_kwargs)
    # (d) X-X upper triangle -> top-right of X-X block, shifted -x / +y.
    ax.text(Lx - pad - em, pad + em, "(d) X (covariant)", **tr_kwargs)
    # (b) Y-Y lower triangle -> bottom-left of Y-Y block, shifted +x / -y.
    ax.text(Lx + pad + em, Lx + Ly - pad - em, "(b) Y (stationary)",
            **bl_kwargs)
    # (e) Y-Y upper triangle -> top-right of Y-Y block, shifted -x / +y.
    ax.text(Lx + Ly - pad - em, Lx + pad + em, "(e) Y (covariant)",
            **tr_kwargs)
    # (c) Y-X below-diag block -> bottom-left.
    ax.text(pad, Lx + Ly - pad, "(c) TKF92 alignment", **bl_kwargs)
    # (f) X-Y above-diag block -> top-right.
    ax.text(Lx + Ly - pad, pad, "(f) TKFDP alignment", **tr_kwargs)

    # Compute ungapped sequence identity over the aligned columns (for Pfam
    # mode; for BAliBase we don't have a single MSA, so skip).
    id_str = ""
    if args.pfam_sto is not None:
        # Use the Pfam stockholm aligned columns to compute pair identity.
        names_all = [t[0] for t in parse_stockholm(args.pfam_sto)]
        with open(args.pfam_sto) as fh:
            seqs: dict[str, list[str]] = {}
            order: list[str] = []
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                nm, frag = parts[0], parts[1].strip()
                if nm not in seqs:
                    seqs[nm] = []
                    order.append(nm)
                seqs[nm].append(frag)
        ax_x = "".join(seqs[name_x]) if name_x in seqs else None
        ax_y = "".join(seqs[name_y]) if name_y in seqs else None
        if ax_x is not None and ax_y is not None:
            n_matches = n_comp = 0
            for ca, cb in zip(ax_x, ax_y):
                if ca.isalpha() and cb.isalpha():
                    n_comp += 1
                    if ca.upper() == cb.upper():
                        n_matches += 1
            id_frac = n_matches / max(n_comp, 1)
            id_str = (f", id={100*id_frac:.1f}% over {n_comp} aligned cols")

    # Title dropped per user direction (the metadata is in the caption).

    # Optional cyan disulfide-bond circles in panels (a),(b),(d),(e).
    def _parse_bonds(s):
        if not s:
            return []
        out = []
        for tok in s.split(","):
            i, j = tok.split(":")
            i, j = int(i), int(j)
            if i > j:
                i, j = j, i
            out.append((i, j))
        return out
    bonds_x = _parse_bonds(args.disulfide_x)
    bonds_y = _parse_bonds(args.disulfide_y)
    # In data coords (origin='upper': larger y = downward). Each bond
    # (i, j) is at (col, row) = (j-1, i-1) in the upper triangle
    # (= covariant panel) and (col, row) = (i-1, j-1) in the lower
    # triangle (= stationary panel). Cyan, non-heatmap.
    ring_kw = dict(facecolor='none', edgecolor='cyan',
                    linewidth=1.5, s=200, zorder=4)
    for (i, j) in bonds_x:
        # (d) X-X covariant: upper triangle, plot at (col=j-1, row=i-1).
        ax.scatter([j - 1], [i - 1], **ring_kw)
        # (a) X-X stationary: lower triangle, plot at (col=i-1, row=j-1).
        ax.scatter([i - 1], [j - 1], **ring_kw)
    for (i, j) in bonds_y:
        # (e) Y-Y covariant: upper triangle in the (Lx..) shifted block.
        ax.scatter([Lx + j - 1], [Lx + i - 1], **ring_kw)
        # (b) Y-Y stationary: lower triangle in the same block.
        ax.scatter([Lx + i - 1], [Lx + j - 1], **ring_kw)

    # Two colorbars stacked vertically and pinned to the right of the
    # Y-half columns: TOP one matches the (f) TKFDP-alignment block
    # (rows 0..Lx; alignment posterior); BOTTOM one matches the (e)
    # Y-covariant block (rows Lx..Lx+Ly; edge-pair posterior).
    plt.tight_layout()
    fig.canvas.draw()    # force layout so ax.get_position() is real
    ax_pos = ax.get_position()
    cb_w = 0.022
    cb_pad = 0.012
    cb_left = ax_pos.x1 + cb_pad
    cb_bottom_total = ax_pos.y0
    cb_total_h = ax_pos.height
    # Vertical split between top (alignment, rows 0..Lx) and bottom
    # (edges, rows Lx..Lx+Ly). With origin='upper' the TOP of the image
    # corresponds to LARGER figure-y, so the TOP colorbar starts at
    # bottom + total_h * Ly/(Lx+Ly).
    frac_top = Lx / (Lx + Ly)
    h_top = cb_total_h * frac_top
    h_bot = cb_total_h * (1.0 - frac_top)
    cb_align_ax = fig.add_axes(
        [cb_left, cb_bottom_total + cb_total_h - h_top, cb_w, h_top])
    cb_edges_ax = fig.add_axes(
        [cb_left, cb_bottom_total, cb_w, h_bot])
    cb_align = fig.colorbar(im_align, cax=cb_align_ax)
    cb_align.set_label("(c)/(f) alignment posterior", fontsize=9)
    cb_edges = fig.colorbar(im_edges, cax=cb_edges_ax)
    cb_edges.set_label("(a),(b),(d),(e) edge-pair posterior", fontsize=9)

    pdf_path = Path(str(args.out) + ".pdf")
    png_path = Path(str(args.out) + ".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"[holmes-tile] wrote {pdf_path} and {png_path}", flush=True)

    # ---- 10. One-line interpretation ------------------------------------
    # Compare Cys-Cys mass between joint and single panels.
    def _avg_at_cys_pairs(P, cys, L_axis):
        if len(cys) < 2:
            return 0.0
        total = 0.0
        count = 0
        for a in range(len(cys)):
            for b in range(a + 1, len(cys)):
                if 1 <= cys[a] <= L_axis and 1 <= cys[b] <= L_axis:
                    total += P[cys[a], cys[b]]
                    count += 1
        return total / max(count, 1)

    def _avg_at_random_pairs(P, L_axis):
        # Off-diagonal entries 1..L_axis.
        vals = []
        for i in range(1, L_axis + 1):
            for j in range(i + 1, L_axis + 1):
                vals.append(P[i, j])
        return float(np.mean(vals)) if vals else 0.0

    cys_xx_joint = _avg_at_cys_pairs(Pxx_joint, cys_x, Lx)
    cys_xx_single = _avg_at_cys_pairs(Pxx_single, cys_x, Lx)
    bg_xx_joint = _avg_at_random_pairs(Pxx_joint, Lx)
    bg_xx_single = _avg_at_random_pairs(Pxx_single, Lx)
    cys_yy_joint = _avg_at_cys_pairs(Pyy_joint, cys_y, Ly)
    cys_yy_single = _avg_at_cys_pairs(Pyy_single, cys_y, Ly)
    bg_yy_joint = _avg_at_random_pairs(Pyy_joint, Ly)
    bg_yy_single = _avg_at_random_pairs(Pyy_single, Ly)
    print(f"[holmes-tile] cys-cys vs background pair posterior:")
    print(f"  X-X joint:  {cys_xx_joint:.4f} (cys) vs {bg_xx_joint:.4f} (bg)")
    print(f"  X-X single: {cys_xx_single:.4f} (cys) vs {bg_xx_single:.4f} (bg)")
    print(f"  Y-Y joint:  {cys_yy_joint:.4f} (cys) vs {bg_yy_joint:.4f} (bg)")
    print(f"  Y-Y single: {cys_yy_single:.4f} (cys) vs {bg_yy_single:.4f} (bg)")


if __name__ == "__main__":
    main()
