"""Experiment 2 Layer 2: Pfam unsupervised H + per-MSA partition MCMC.

Per the user's framing of pfam_evaluation.md (Sept 2026 revision):

- Use Pfam families from ~/bio-datasets/data/pfam (Stockholm + Newick).
- Each family contributes cherries (sister-leaf pairs) at FastTree-LG distances.
- All cherries within an MSA share the same per-MSA partition over columns
  (size-2 cap), but the *global* H is shared across all families.
- The partition is *unsupervised*: per-family matchings are sampled by
  single-column Gibbs against the joint composite likelihood (no PDB contact
  maps, no precomputed plmDCA).

Outer alternation:
  - SGD on H with current partitions (composite log-L over all cherry x edge
    observations).
  - Per-family Gibbs sweep on the partition.

Outputs:
  - H_pooled.npy
  - convergence trace (log L, ||H||_F, total #pairs)
  - heatmap, residue dendrogram, h(a) physchem correlation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tkfdp.composite import composite_log_likelihood, project_to_symmetric_zero_trace
from tkfdp.generator import (
    A,
    A2,
    build_joint_Q,
    joint_stationary,
    log_transition_matrices,
    symmetrize_eigh,
    transition_matrices,
)
from tkfdp.lg08 import ALPHA_ORDER, Q_LG08_J
from tkfdp.partition import (
    FamilyPartitionState,
    edges_of,
    gibbs_sweep,
    init_all_singletons,
    n_pairs_in,
)
from tkfdp.pfam_data import FamilyCherries, families_from_split


@jax.jit
def log_P_unique_jit(H_, unique_t_):
    Q = build_joint_Q(H_)
    pi_j = joint_stationary(H_)
    Lambda, U_sym, sqrt_pij = symmetrize_eigh(Q, pi_j)
    return log_transition_matrices(unique_t_, Lambda, U_sym, sqrt_pij)


@jax.jit
def log_P_singleton_jit(unique_t_):
    """log P_LG08(t) for each unique t. Returns (K, A, A)."""
    def one(t):
        return jnp.log(jnp.clip(jsl.expm(Q_LG08_J * t), 1e-300, 1.0))
    return jax.vmap(one)(unique_t_)


def build_corpus(families: list[FamilyCherries]):
    """Pool cherries across families, mapping each to a global unique-t index.

    Returns:
        unique_t: (K,) numpy array
        per_family: list of dicts with
            'tau_idx': (C,) int32 into unique_t
            'aa_a', 'aa_b': (C, L) int8
            'both_aa': (C, L) bool
    """
    all_t = np.concatenate([fc.tau for fc in families])
    unique_t, inv = np.unique(all_t, return_inverse=True)

    per_family = []
    cursor = 0
    for fc in families:
        C = fc.n_cherries
        tau_idx = inv[cursor: cursor + C].astype(np.int32)
        cursor += C
        per_family.append(dict(
            family=fc.family,
            L=fc.L,
            tau_idx=tau_idx,
            aa_a=fc.aa_a,
            aa_b=fc.aa_b,
            both_aa=fc.both_aa_mask(),
            n_cherries=C,
        ))
    return unique_t, per_family


def gather_pair_loglik(family_data: dict,
                       log_P_unique: np.ndarray,
                       s: int) -> np.ndarray:
    """Vectorized: pair_loglik(s, t) for all t in the family.

    Returns (L,) numpy array. Entries where the pair is degenerate
    (e.g., t == s, or every cherry has a gap somewhere in {s, t}) are 0.
    """
    aa_a = family_data["aa_a"]      # (C, L)
    aa_b = family_data["aa_b"]      # (C, L)
    both_aa = family_data["both_aa"]  # (C, L)
    tau_idx = family_data["tau_idx"]  # (C,)
    L = family_data["L"]
    C = family_data["n_cherries"]

    # AA at column s for each cherry. (C,)
    a_s = aa_a[:, s].astype(np.int64)
    b_s = aa_b[:, s].astype(np.int64)
    valid_s = both_aa[:, s]   # (C,)

    # For all t: start_idx[c, t] = a_s[c] * 20 + aa_a[c, t]; end_idx[c, t] = b_s[c]*20 + aa_b[c, t]
    aa_a_64 = aa_a.astype(np.int64)
    aa_b_64 = aa_b.astype(np.int64)
    start_idx = a_s[:, None] * 20 + aa_a_64  # (C, L)
    end_idx = b_s[:, None] * 20 + aa_b_64    # (C, L)
    valid = valid_s[:, None] & both_aa       # (C, L)

    # log_P_per_cherry: (C, 400, 400) — too big for some L
    # Better: gather one (cherry, start, end) at a time, but we want pair_loglik(s, t) for all t.
    # We can compute log_P[t_idx[c]] which is (400, 400), then fancy-index by start_idx[c, t], end_idx[c, t].
    # Vectorized over t:
    log_p_per_obs = np.zeros((C, L), dtype=np.float64)
    for c in range(C):
        if not valid_s[c]:
            continue
        P = log_P_unique[tau_idx[c]]  # (400, 400)
        # gather P[start_idx[c, :], end_idx[c, :]] over all t
        # only for t with both_aa[c, t]
        v = both_aa[c, :]
        if v.any():
            si = start_idx[c, v]
            ei = end_idx[c, v]
            log_p_per_obs[c, v] = P[si, ei]

    pair_loglik = log_p_per_obs.sum(axis=0)  # (L,)
    return pair_loglik


def precompute_singleton_loglik(family_data: dict,
                                log_P_LG08_unique: np.ndarray) -> np.ndarray:
    """Precompute single_loglik[s] for all columns s in the family."""
    aa_a = family_data["aa_a"]; aa_b = family_data["aa_b"]
    both_aa = family_data["both_aa"]; tau_idx = family_data["tau_idx"]
    L = family_data["L"]; C = family_data["n_cherries"]
    out = np.zeros(L, dtype=np.float64)
    for c in range(C):
        v = both_aa[c, :]
        if not v.any():
            continue
        a = aa_a[c, v].astype(np.int64)
        b = aa_b[c, v].astype(np.int64)
        out[v] += log_P_LG08_unique[tau_idx[c]][a, b]
    return out


def build_obs_from_partitions(per_family: list[dict],
                              states: list[FamilyPartitionState]) -> np.ndarray:
    """Flatten all (cherry, edge) pair observations across families into
    a single (M, 3) int64 array (t_idx, start_state, end_state) usable
    by composite_log_likelihood.
    """
    rows = []
    for fd, st in zip(per_family, states):
        for s in range(fd["L"]):
            t = int(st.partner[s])
            if t <= s:
                continue  # only count each edge once and skip singletons
            valid = fd["both_aa"][:, s] & fd["both_aa"][:, t]
            if not valid.any():
                continue
            tau_idx_c = fd["tau_idx"][valid]
            a_s = fd["aa_a"][valid, s].astype(np.int64)
            a_t = fd["aa_a"][valid, t].astype(np.int64)
            b_s = fd["aa_b"][valid, s].astype(np.int64)
            b_t = fd["aa_b"][valid, t].astype(np.int64)
            start = a_s * A + a_t
            end = b_s * A + b_t
            rows.append(np.column_stack([tau_idx_c, start, end]))
    if rows:
        return np.concatenate(rows, axis=0)
    return np.zeros((0, 3), dtype=np.int64)


def heatmap(H: np.ndarray, title: str, path: Path, vmax=None):
    if vmax is None:
        vmax = float(np.abs(H).max())
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(H, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(20)); ax.set_yticks(range(20))
    ax.set_xticklabels(list(ALPHA_ORDER), fontsize=8)
    ax.set_yticklabels(list(ALPHA_ORDER), fontsize=8)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-families", type=int, default=10)
    ap.add_argument("--min-cherries", type=int, default=8)
    ap.add_argument("--min-cols", type=int, default=30)
    ap.add_argument("--max-cols", type=int, default=120)
    ap.add_argument("--n-outer", type=int, default=30)
    ap.add_argument("--n-sgd-per-outer", type=int, default=100)
    ap.add_argument("--n-gibbs-per-outer", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--l2", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--init-H-scale", type=float, default=0.3,
                    help="Initial H ~ N(0, init_H_scale); breaks degeneracy and gives Gibbs signal")
    ap.add_argument("--init-pair-fraction", type=float, default=0.4,
                    help="Fraction of columns initialized to random pairs (singletons otherwise)")
    ap.add_argument("--init", type=str, default="singletons", choices=["singletons"])
    ap.add_argument("--dp-alpha", type=float, default=1.0,
                    help="DP concentration α. Pairs cost α^2 vs two singletons (each costs α). "
                         "Larger α => more singletons, less selection feedback.")
    ap.add_argument("--n-partition-samples", type=int, default=1,
                    help="Number of independent partition samples per SGD step. "
                         "Each step's gradient is averaged over these samples (Monte Carlo "
                         "marginalisation of the partition latent).")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--families", type=str, default=None,
                    help="Comma-separated list of explicit Pfam IDs (overrides --split/--n-families)")
    ap.add_argument("--seed-from-pdb", action="store_true",
                    help="Initialize each family's partition from PDB contacts (Cα <8 Å) "
                         "rather than random pairs. Requires SCOP cross-ref in the family's .sto.")
    ap.add_argument("--pdb-contact-threshold", type=float, default=8.0)
    ap.add_argument("--pdb-min-separation", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results" / "exp2_pfam")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    keep_masks = None  # only populated if explicit list, used by PDB seeding
    if args.families:
        family_ids = [f.strip() for f in args.families.split(",") if f.strip()]
        print(f"Loading {len(family_ids)} explicit Pfam families ...")
        from tkfdp.pfam_data import families_from_list
        families, keep_masks = families_from_list(
            family_ids, min_cherries=args.min_cherries, return_keep_masks=True,
        )
    else:
        print(f"Loading {args.n_families} Pfam families from split={args.split} ...")
        families = families_from_split(
            split_name=args.split, n_families=args.n_families,
            min_cherries=args.min_cherries, min_columns=args.min_cols,
            max_columns=args.max_cols, seed=args.seed,
        )
    print(f"  Loaded {len(families)} families:")
    for fc in families:
        print(f"    {fc.family}: {fc.L} cols, {fc.n_cherries} cherries")
    if len(families) == 0:
        print("ERROR: no families pass the filters; widen them.")
        sys.exit(2)

    # If PDB seeding requested, compute per-family contact pairs (in filtered
    # column indexing) up front; reused for every replica's init.
    pdb_pairs_per_family: list[list[tuple[int, int]] | None] = [None] * len(families)
    if args.seed_from_pdb:
        if keep_masks is None:
            print("ERROR: --seed-from-pdb requires --families (need stable column indexing).")
            sys.exit(2)
        from tkfdp.bio import parse_stockholm, PFAM_SEED_DIR
        from tkfdp.pdb_contacts import pdb_contacts_for_family
        print("\nFetching PDB contacts for seeding ...")
        for k, fc in enumerate(families):
            seqs = parse_stockholm(PFAM_SEED_DIR / f"{fc.family}.sto")
            info = pdb_contacts_for_family(
                fc.family, seqs,
                distance_threshold=args.pdb_contact_threshold,
                min_separation=args.pdb_min_separation,
            )
            if info is None:
                print(f"  {fc.family}: no PDB contacts (will fall back to random)")
                pdb_pairs_per_family[k] = []
                continue
            # Remap original column indices -> filtered column indices via keep_mask
            mask = keep_masks[k]
            orig_to_filt = -np.ones(len(mask), dtype=np.int64)
            orig_to_filt[mask] = np.arange(mask.sum())
            remapped = []
            for (i_o, j_o), d in zip(info.contact_pairs, info.contact_distances):
                if i_o < len(mask) and j_o < len(mask):
                    i_f = orig_to_filt[i_o]; j_f = orig_to_filt[j_o]
                    if i_f >= 0 and j_f >= 0:
                        remapped.append((int(i_f), int(j_f)))
            pdb_pairs_per_family[k] = remapped
            print(f"  {fc.family}: PDB={info.pdb_id} {len(info.contact_pairs)} raw pairs -> {len(remapped)} after column filter")

    unique_t, per_family = build_corpus(families)
    print(f"  Unique cherry distances: {len(unique_t)}, range [{unique_t.min():.3f}, {unique_t.max():.3f}]")

    # Precompute LG08 single-site log P for each unique t, then per-family single_loglik tables.
    log_P_LG08 = np.asarray(log_P_singleton_jit(jnp.asarray(unique_t)))  # (K, A, A)
    single_loglik_per_family = [precompute_singleton_loglik(fd, log_P_LG08) for fd in per_family]

    # Initialize n_partition_samples parallel partition replicas
    rng = np.random.default_rng(args.seed)
    from tkfdp.partition import init_random_pairs, init_from_pairs
    K_part = max(1, args.n_partition_samples)
    state_replicas = []
    for k in range(K_part):
        rep = []
        for fam_idx, fd in enumerate(per_family):
            if args.seed_from_pdb and pdb_pairs_per_family[fam_idx]:
                rep.append(init_from_pairs(fd["family"], fd["L"], pdb_pairs_per_family[fam_idx]))
            else:
                n_pairs_init = int(fd["L"] * args.init_pair_fraction / 2)
                rep.append(init_random_pairs(fd["family"], fd["L"], n_pairs_init, rng))
        state_replicas.append(rep)
    print(f"\nReplica 0 initial pair counts: "
          f"{[int((s.partner >= 0).sum() // 2) for s in state_replicas[0]]}")

    # Initialize H with non-trivial random scale: at H=0 the joint Q is the
    # tensor sum of two LG08 chains so its spectrum is degenerate (eigh JVP NaN);
    # at H=0 the partition Gibbs is also signal-free.
    H_init = rng.normal(scale=args.init_H_scale, size=(A, A))
    H_init = 0.5 * (H_init + H_init.T)
    H_init = H_init - np.trace(H_init) / A * np.eye(A)
    H = jnp.asarray(H_init)
    import optax
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(H)

    l2 = args.l2

    @jax.jit
    def neg_penalized_log_l(H_, t_, o_):
        ll = composite_log_likelihood(H_, t_, o_)
        off = H_ - jnp.diag(jnp.diag(H_))
        return -ll + 0.5 * l2 * jnp.sum(off * off)

    grad_fn = jax.jit(jax.grad(neg_penalized_log_l))

    @jax.jit
    def loss_fn(H_, t_, o_):
        return composite_log_likelihood(H_, t_, o_)

    unique_t_j = jnp.asarray(unique_t)

    # Trace
    log_l_history = []
    H_norm_history = []
    n_pairs_history = []
    iter_times = []

    log_prior_offset = -2.0 * np.log(args.dp_alpha)

    t0_total = time.time()
    for it in range(args.n_outer):
        t_outer_start = time.time()

        # === Build observations from each partition replica; SGD on H using
        # the gradient averaged over the K_part replicas (Monte Carlo
        # marginalisation of the partition latent). ===
        obs_per_rep = [build_obs_from_partitions(per_family, rep) for rep in state_replicas]
        total_obs = sum(o.shape[0] for o in obs_per_rep)
        if total_obs == 0:
            print(f"[outer {it}] no pair observations yet; H stays at init.")
            ll_now = float('nan')
        else:
            obs_jax_per_rep = [jnp.asarray(o) for o in obs_per_rep if o.shape[0] > 0]
            for sgd_step in range(args.n_sgd_per_outer):
                # Average the gradient across replicas (Monte Carlo estimator)
                g_sum = jnp.zeros((A, A))
                for o_j in obs_jax_per_rep:
                    g_sum = g_sum + grad_fn(H, unique_t_j, o_j)
                g = g_sum / max(1, len(obs_jax_per_rep))
                g = (g + g.T) / 2
                g = g - jnp.trace(g) / A * jnp.eye(A)
                updates, opt_state = optimizer.update(g, opt_state)
                H = optax.apply_updates(H, updates)
                H = project_to_symmetric_zero_trace(H)
            # Report log L on the first replica's obs (just for monitoring)
            ll_now = float(loss_fn(H, unique_t_j, obs_jax_per_rep[0]))

        # === Compute log_P_unique for partition Gibbs ===
        log_P_unique = np.asarray(log_P_unique_jit(H, unique_t_j))  # (K, 400, 400)

        # === Per-replica, per-family Gibbs sweeps ===
        n_pairs_total = 0
        for rep in state_replicas:
            for fd, st, sll in zip(per_family, rep, single_loglik_per_family):
                def pair_fn(s):
                    return gather_pair_loglik(fd, log_P_unique, s)
                for _ in range(args.n_gibbs_per_outer):
                    gibbs_sweep(st, pair_fn, sll, rng,
                                  temperature=args.temperature,
                                  log_pair_prior_offset=log_prior_offset)
                n_pairs_total += n_pairs_in(st)
        # Average pair count across replicas for reporting
        n_pairs_total = n_pairs_total / max(1, len(state_replicas))

        H_norm = float(jnp.linalg.norm(H))
        iter_time = time.time() - t_outer_start
        log_l_history.append(ll_now)
        H_norm_history.append(H_norm)
        n_pairs_history.append(n_pairs_total)
        iter_times.append(iter_time)
        print(f"[outer {it+1:3d}/{args.n_outer}] log L={ll_now:11.2f}  ||H||_F={H_norm:.3f}  pairs={n_pairs_total}  ({iter_time:.1f}s)")

    elapsed_total = time.time() - t0_total
    H_np = np.asarray(H)
    print(f"\nTotal elapsed: {elapsed_total:.1f}s ({elapsed_total/args.n_outer:.1f}s / outer iter)")
    print(f"Final ||H||_F = {np.linalg.norm(H_np):.3f}")
    print(f"Final pair count: {n_pairs_history[-1]}")

    # Save artifacts
    np.save(args.out_dir / "H_pooled.npy", H_np)
    # Final partitions per family (replica 0): list of (i, j) pairs (i < j)
    final_partitions = {
        per_family[fam_idx]["family"]: [(int(s), int(state_replicas[0][fam_idx].partner[s]))
                                         for s in range(per_family[fam_idx]["L"])
                                         if state_replicas[0][fam_idx].partner[s] > s]
        for fam_idx in range(len(per_family))
    }
    with open(args.out_dir / "final_partitions.json", "w") as f:
        json.dump(final_partitions, f, indent=2)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(args.out_dir / "trace.json", "w") as f:
        json.dump(dict(
            log_l=log_l_history,
            H_norm=H_norm_history,
            n_pairs=n_pairs_history,
            iter_times=iter_times,
            elapsed_total=elapsed_total,
            families=[fc.family for fc in families],
        ), f, indent=2)

    # Heatmap with biochemistry-class ordering for readability
    chem_order = "ILMVAFW" + "STNQGHC" + "DERKP" + "Y"
    chem_perm = np.array([ALPHA_ORDER.index(c) for c in chem_order])
    H_chem = H_np[chem_perm][:, chem_perm]
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = float(np.abs(H_np).max())
    im = ax.imshow(H_chem, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(20)); ax.set_yticks(range(20))
    ax.set_xticklabels(list(chem_order), fontsize=9)
    ax.set_yticklabels(list(chem_order), fontsize=9)
    ax.set_title(f"H_pooled (Pfam unsupervised, {len(families)} families)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(args.out_dir / "H_pooled_heatmap.png", dpi=120); plt.close(fig)

    # Convergence plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(range(1, len(log_l_history) + 1), log_l_history); axes[0].set_xlabel("outer iter"); axes[0].set_ylabel("composite log L")
    axes[1].plot(range(1, len(H_norm_history) + 1), H_norm_history); axes[1].set_xlabel("outer iter"); axes[1].set_ylabel("||H||_F")
    axes[2].plot(range(1, len(n_pairs_history) + 1), n_pairs_history); axes[2].set_xlabel("outer iter"); axes[2].set_ylabel("# pairs (corpus)")
    plt.tight_layout(); plt.savefig(args.out_dir / "convergence.png", dpi=120); plt.close(fig)

    # Top-10 most negative off-diag entries (favored pairs)
    iu = np.triu_indices(20, k=1)
    off = H_np[iu]
    order = np.argsort(off)
    top = order[:15]
    print("\nTop-15 most negative off-diagonal entries (favored pairs):")
    for k in top:
        a, b = iu[0][k], iu[1][k]
        print(f"  {ALPHA_ORDER[a]}-{ALPHA_ORDER[b]}: {off[k]:+.3f}")

    print(f"\nArtifacts in: {args.out_dir}")


if __name__ == "__main__":
    main()
