"""Split-merge moves on the archetype set.

At training time the archetype trainer periodically applies split-merge
moves to re-allocate the fixed budget of K_a archetype slots. Two
motivations:

- Under truncated stick-breaking (TSB), dead archetypes (those with
  negligible rho_arch) never revive on their own -- they represent
  wasted capacity.
- Overloaded archetypes (those with high rho_arch and diffuse
  pi_arch) may be aggregating heterogeneous positions that would
  benefit from being split into two.

The moves here are hill-climbing under the multinomial marginal
log-likelihood (they accept iff the SS-based LL strictly improves),
NOT a full reversible Jain-Neal Metropolis-Hastings. This matches the
surrounding stochastic-EM training regime -- we are moving pi_arch
and arch_assignment towards the SS-based MAP, not sampling from a
posterior. The move template is Jain & Neal (2004) split-merge for
Dirichlet mixture models; the acceptance criterion here is
deterministic (LL improvement) rather than the full MH ratio.

Two move types are proposed:

SPLIT: pick a dead slot k_dead (rho_arch[k_dead] < eps) and an
overloaded slot k_over (highest rho_arch * entropy(pi_arch) product).
Do 2-means clustering on the per-(c, theta) residue-count evidence
of the (c, theta) pairs currently assigned to k_over; the two centroids
become the new pi_arch[k_over] and pi_arch[k_dead], and the (c, theta)
assignments are redistributed accordingly.

MERGE: pick a pair (k1, k2) with the smallest total-variation distance
between their pi_arch rows and non-negligible rho_arch. Merge them by
taking the rho_arch-weighted mean of pi_arch[k1] and pi_arch[k2] into
k1, redirect all (c, theta) previously pointing at k2 to k1, and free
k2 (uniform pi_arch, zero rho_arch mass).

After any accepted move, rho_arch and tsb_betas_arch are recomputed
from the updated arch_assignment via the existing TSB update.
"""
from __future__ import annotations

import numpy as np


def dead_archetypes(rho_arch: np.ndarray,
                     threshold: float = 0.01,
                     pi_arch: 'np.ndarray | None' = None,
                     entropy_threshold_bits: float = 0.5,
                     arch_assignment: 'np.ndarray | None' = None) -> list:
    """Return the archetype indices considered "dead" for revival.

    Three failure modes count as dead:
      1. Low weight: rho_arch[k] < threshold. Under an updated TSB
         this correlates with no (c, θ) slot being routed to k.
      2. Collapsed emission: entropy(pi_arch[k]) < entropy_threshold_bits.
         The slot has been driven to near-zero mass on most of the
         alphabet by SS starvation, so residues at those entries
         produce huge log-losses whenever observed.
         (Requires pi_arch to be passed; skipped if pi_arch is None.)
      3. Zero usage: no (c, θ) slot in arch_assignment points at k.
         When the TSB is frozen at uniform (e.g., --update-arch-tsb
         off in train_dynfield.py), rho_arch stays at 1/K_a and
         detector (1) never fires. This usage-based detector is the
         natural fallback: an archetype that no one uses IS dead by
         definition. (Requires arch_assignment to be passed; skipped
         otherwise.)
    """
    r = np.asarray(rho_arch, dtype=np.float64)
    low_usage = np.where(r < float(threshold))[0].tolist()
    out = list(low_usage)
    seen = set(low_usage)
    if pi_arch is not None:
        p = np.asarray(pi_arch, dtype=np.float64)
        ent = -(p * np.log2(np.maximum(p, 1e-300))).sum(axis=1)
        collapsed = np.where(ent < float(entropy_threshold_bits))[0].tolist()
        for k in collapsed:
            if k not in seen:
                out.append(int(k)); seen.add(k)
    if arch_assignment is not None:
        K_a = int(r.shape[0])
        counts = np.bincount(
            np.asarray(arch_assignment, dtype=np.int64).reshape(-1),
            minlength=K_a)
        no_slot = np.where(counts == 0)[0].tolist()
        for k in no_slot:
            if k not in seen:
                out.append(int(k)); seen.add(k)
    return out


def overloaded_archetypes(pi_arch: np.ndarray, rho_arch: np.ndarray,
                            top_k: int = 1) -> list:
    """Rank archetypes by rho_arch[k] * entropy(pi_arch[k]) and return
    the top-k indices. High rho AND high entropy = heavy AND diffuse =
    good split candidate."""
    p = np.asarray(pi_arch, dtype=np.float64)
    r = np.asarray(rho_arch, dtype=np.float64)
    ent = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1)
    score = r * ent
    idx = np.argsort(score)[::-1][:int(top_k)]
    return idx.tolist()


def similar_archetype_pairs(pi_arch: np.ndarray, rho_arch: np.ndarray,
                              tvd_threshold: float = 0.05,
                              rho_min: float = 0.01) -> list:
    """Return sorted list of (k1, k2, tvd) pairs with total-variation
    distance below `tvd_threshold` and both rho_arch entries above
    `rho_min`. Sorted ascending by TVD (most-similar first)."""
    p = np.asarray(pi_arch, dtype=np.float64)
    r = np.asarray(rho_arch, dtype=np.float64)
    K_a = p.shape[0]
    pairs = []
    for k1 in range(K_a):
        for k2 in range(k1 + 1, K_a):
            if r[k1] < rho_min or r[k2] < rho_min:
                continue
            tvd = 0.5 * float(np.abs(p[k1] - p[k2]).sum())
            if tvd < tvd_threshold:
                pairs.append((int(k1), int(k2), tvd))
    return sorted(pairs, key=lambda x: x[2])


def _two_means_on_rows(rows: np.ndarray, weights: np.ndarray,
                          rng: np.random.Generator,
                          n_iters: int = 20) -> tuple:
    """Weighted 2-means on n rows in the A-simplex using L1 (TVD) distance.

    rows: (n, A) simplex points.
    weights: (n,) row masses (used as weight for the centroid mean).

    Returns (labels, centers) with labels in {0, 1} and centers shape (2, A).
    """
    n, A = rows.shape
    if n < 2:
        return np.zeros(n, dtype=np.int32), np.stack([rows.mean(axis=0)] * 2)
    # Init centers by 2 random rows (weight-biased). Guard against
    # non-finite weights (upstream V/W/U SS can pick up NaN when the
    # M-step is in a bad regime); fall back to uniform sampling.
    w_arr = np.asarray(weights, dtype=np.float64)
    w_arr = np.where(np.isfinite(w_arr) & (w_arr > 0), w_arr, 0.0)
    total_w = float(w_arr.sum())
    if total_w <= 0.0:
        w_p = np.full(n, 1.0 / n)
    else:
        w_p = w_arr / total_w
    ids = rng.choice(n, size=2, replace=False, p=w_p)
    centers = rows[ids].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(int(n_iters)):
        # Assign each row to nearest center by TVD.
        d0 = 0.5 * np.abs(rows - centers[0]).sum(axis=1)
        d1 = 0.5 * np.abs(rows - centers[1]).sum(axis=1)
        new_labels = (d1 < d0).astype(np.int32)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        # Update centers: weighted mean of assigned rows.
        for j in range(2):
            m = (labels == j)
            if m.sum() == 0:
                # Empty cluster: reinit with the farthest-from-current row.
                d_other = 0.5 * np.abs(rows - centers[1 - j]).sum(axis=1)
                centers[j] = rows[int(np.argmax(d_other))]
            else:
                w = weights[m]
                total = float(w.sum())
                if total < 1e-300:
                    centers[j] = rows[m].mean(axis=0)
                else:
                    centers[j] = (rows[m] * w[:, None]).sum(axis=0) / total
            # Normalise so centers stay on the simplex.
            centers[j] = np.maximum(centers[j], 1e-30)
            centers[j] = centers[j] / centers[j].sum()
    return labels, centers


def propose_split(pi_arch: np.ndarray, arch_assignment: np.ndarray,
                    N_ctheta: np.ndarray, k_source: int, k_target: int,
                    rng: np.random.Generator) -> tuple:
    """Split archetype k_source into (k_source, k_target).

    Uses weighted 2-means on the per-(c, theta) residue-count vectors of
    the (c, theta) pairs currently pointing at k_source. The two
    centroids become the new pi_arch[k_source] and pi_arch[k_target];
    each (c, theta) is reassigned to the nearer centroid.

    Args:
      pi_arch: (K_a, A) current archetype distributions.
      arch_assignment: (K_c, L_max) int current per-(c, theta) archetype
        indices.
      N_ctheta: (K_c, L_max, A) per-(c, theta) residue-count evidence.
      k_source: heavy archetype to split.
      k_target: dead slot to receive one of the split centroids.
      rng: numpy random generator.

    Returns (pi_new, arch_new) proposals; or None if there are fewer
    than 2 (c, theta) members assigned to k_source (nothing to split).
    """
    K_a, A = pi_arch.shape
    mask = (arch_assignment == int(k_source))                    # (K_c, L_max)
    ct = np.argwhere(mask)                                        # (n, 2)
    n = int(ct.shape[0])
    if n < 2:
        return None
    # Collect (c, theta) residue-count rows.
    rows = np.zeros((n, A), dtype=np.float64)
    for i in range(n):
        c, th = int(ct[i, 0]), int(ct[i, 1])
        rows[i] = N_ctheta[c, th]
    row_mass = rows.sum(axis=1)
    total_mass = float(row_mass.sum())
    if total_mass < 1e-9:
        return None
    # Normalise for 2-means input; keep row_mass as the weight. Guard
    # against non-finite rows (SS noise) — refuse the split rather than
    # feeding NaN into 2-means.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        rows_norm = rows / np.maximum(row_mass[:, None], 1e-300)
    if not np.all(np.isfinite(rows_norm)):
        return None
    labels, centers = _two_means_on_rows(rows_norm, row_mass, rng)

    # Refuse degenerate splits (one side empty).
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    if n0 == 0 or n1 == 0:
        return None

    pi_new = pi_arch.copy()
    pi_new[int(k_source)] = np.maximum(centers[0], 1e-30)
    pi_new[int(k_source)] /= pi_new[int(k_source)].sum()
    pi_new[int(k_target)] = np.maximum(centers[1], 1e-30)
    pi_new[int(k_target)] /= pi_new[int(k_target)].sum()

    arch_new = arch_assignment.copy()
    for i in range(n):
        c, th = int(ct[i, 0]), int(ct[i, 1])
        arch_new[c, th] = int(k_source) if labels[i] == 0 else int(k_target)
    return pi_new, arch_new


def propose_merge(pi_arch: np.ndarray, arch_assignment: np.ndarray,
                    rho_arch: np.ndarray, k1: int, k2: int) -> tuple:
    """Merge archetypes k1 and k2 into k1.

    pi_arch[k1] <- weighted mean of pi_arch[k1] and pi_arch[k2] with
    weights rho_arch[k1] and rho_arch[k2]. pi_arch[k2] becomes a
    uniform slot (unused, awaiting future revival). All (c, theta)
    pointing at k2 are redirected to k1.

    Returns (pi_new, arch_new) proposals.
    """
    A = pi_arch.shape[1]
    pi_new = pi_arch.copy()
    w1, w2 = float(rho_arch[int(k1)]), float(rho_arch[int(k2)])
    w_tot = w1 + w2
    if w_tot < 1e-12:
        w1, w2 = 0.5, 0.5
        w_tot = 1.0
    pi_new[int(k1)] = (w1 * pi_arch[int(k1)] + w2 * pi_arch[int(k2)]) / w_tot
    pi_new[int(k2)] = 1.0 / float(A)
    arch_new = arch_assignment.copy()
    arch_new[arch_assignment == int(k2)] = int(k1)
    return pi_new, arch_new


def multinomial_log_lik(pi_arch: np.ndarray,
                          arch_assignment: np.ndarray,
                          N_ctheta: np.ndarray) -> float:
    """Sum log-likelihood under the multinomial emission model:

        LL = sum_{(c, theta), a} N_ctheta[c, theta, a]
                                * log pi_arch[arch_assignment[c, theta], a]

    Used as the SS-based hill-climbing acceptance criterion for split
    and merge proposals. Deliberately excludes the substitution-side
    contribution (which involves W and the exchangeability S); the
    multinomial LL is a fast proxy for the archetype-emission fit that
    dominates the response to a split/merge move.
    """
    log_pi = np.log(np.maximum(pi_arch, 1e-300))
    K_c, L_max, A = N_ctheta.shape
    ll = 0.0
    for c in range(K_c):
        for th in range(L_max):
            k = int(arch_assignment[c, th])
            ll += float(np.dot(N_ctheta[c, th], log_pi[k]))
    return ll


def split_merge_step(state, N_ctheta: np.ndarray,
                       rng: np.random.Generator, *,
                       max_moves_per_call: int = 2,
                       dead_threshold: float = 0.01,
                       dead_entropy_threshold_bits: float = 0.5,
                       merge_tvd_threshold: float = 0.05,
                       ll_improvement_threshold: float = 1.0,
                       merge_ll_tolerance_relative: float = 1e-4,
                       verbose: bool = False) -> dict:
    """Hill-climb split-merge on state.dyn_field's archetype block.

    Applies up to `max_moves_per_call` accepted split/merge moves. Each
    proposal is accepted iff the multinomial LL under the SS strictly
    improves by at least `ll_improvement_threshold` (splits) or is
    within `merge_ll_tolerance` of the current LL (merges — merges are
    allowed a small tolerance since they reduce parameter count, which
    is beneficial under an implicit MDL / Ockham penalty not captured
    by the plain LL).

    Rebuilds rho_arch and tsb_betas_arch from the new arch_assignment
    using the existing TSB update. Materialises pi_field after any
    accepted move so downstream consumers see the update.

    Returns a diagnostic dict:
      {'splits': [{'k_source': .., 'k_target': .., 'll_delta': ..}, ...],
       'merges': [{'k1': .., 'k2': .., 'tvd': .., 'll_delta': ..}, ...],
       'll_before': float, 'll_after': float,
       'n_moves_accepted': int}
    """
    from .archetypes import _update_rho_arch_tsb_expected
    dyn = state.dyn_field
    pi_arch = np.asarray(dyn.pi_archetype, dtype=np.float64).copy()
    arch_assignment = np.asarray(
        dyn.arch_assignment, dtype=np.int32).copy()
    rho_arch = np.asarray(dyn.rho_arch, dtype=np.float64).copy()
    K_a = int(pi_arch.shape[0])
    alpha_arch = float(dyn.alpha_arch)

    def _rebuild_rho(arch_asg):
        n_k = np.bincount(arch_asg.reshape(-1).astype(np.int64),
                            minlength=K_a).astype(np.float64)
        tsb_new, rho_new = _update_rho_arch_tsb_expected(
            n_k, K_a, alpha_arch=alpha_arch)
        return tsb_new, rho_new

    info = {'splits': [], 'merges': [],
              'll_before': None, 'll_after': None,
              'n_moves_accepted': 0}
    ll_cur = multinomial_log_lik(pi_arch, arch_assignment, N_ctheta)
    info['ll_before'] = float(ll_cur)

    if verbose:
        # Always emit an entry-point line so we can tell from logs
        # whether the step is being called at all.
        _deads_now = dead_archetypes(
            rho_arch, threshold=dead_threshold,
            pi_arch=pi_arch,
            entropy_threshold_bits=dead_entropy_threshold_bits,
            arch_assignment=arch_assignment)
        _pairs_now = similar_archetype_pairs(
            pi_arch, rho_arch, tvd_threshold=merge_tvd_threshold)
        _ent_now = -(pi_arch * np.log2(np.maximum(pi_arch, 1e-300))
                          ).sum(axis=1)
        print(f"    [split-merge] enter: ll_cur={ll_cur:.1f}  "
                f"deads={_deads_now}  merge_pairs={_pairs_now[:3]}  "
                f"entropies={[float(f'{e:.2f}') for e in _ent_now]}",
                flush=True)

    for _ in range(int(max_moves_per_call)):
        move_accepted_this_round = False

        # ----- SPLIT proposal -----
        # Extended dead detector: low rho OR collapsed emission (entropy
        # under threshold). A rho-heavy but entropy-collapsed archetype
        # is broken (residues at its near-zero entries produce
        # catastrophic log-loss); reviving it via a split is strictly
        # better than leaving it in place.
        deads = dead_archetypes(
            rho_arch, threshold=dead_threshold,
            pi_arch=pi_arch,
            entropy_threshold_bits=dead_entropy_threshold_bits,
            arch_assignment=arch_assignment)
        overs = [k for k in overloaded_archetypes(pi_arch, rho_arch, top_k=3)
                    if k not in deads]
        if deads and overs:
            k_dead = int(deads[0])
            k_over = int(overs[0])
            proposal = propose_split(
                pi_arch, arch_assignment, N_ctheta,
                k_over, k_dead, rng)
            if proposal is not None:
                pi_prop, arch_prop = proposal
                ll_prop = multinomial_log_lik(pi_prop, arch_prop, N_ctheta)
                if ll_prop > ll_cur + ll_improvement_threshold:
                    pi_arch = pi_prop
                    arch_assignment = arch_prop
                    tsb_new, rho_arch = _rebuild_rho(arch_assignment)
                    dyn.tsb_betas_arch = tsb_new
                    info['splits'].append({
                        'k_source': k_over,
                        'k_target': k_dead,
                        'll_delta': float(ll_prop - ll_cur),
                    })
                    ll_cur = ll_prop
                    info['n_moves_accepted'] += 1
                    move_accepted_this_round = True
                    if verbose:
                        print(f"    [split-merge] SPLIT k={k_over} "
                                f"-> ({k_over}, {k_dead}); "
                                f"dLL={ll_prop - info['ll_before']:+.1f}",
                                flush=True)

        # ----- MERGE proposal -----
        # Tolerance is relative to the current LL scale: merging near-
        # identical archetypes should always be accepted even if the
        # multinomial LL nudges down by numerical noise, because merging
        # reduces model capacity and frees the slot for future split
        # revivals. Using an absolute tolerance would be brittle across
        # runs with different corpus sizes.
        merge_ll_tolerance = abs(ll_cur) * float(merge_ll_tolerance_relative)
        pairs = similar_archetype_pairs(
            pi_arch, rho_arch,
            tvd_threshold=merge_tvd_threshold)
        if pairs:
            k1, k2, tvd = pairs[0]
            pi_prop, arch_prop = propose_merge(
                pi_arch, arch_assignment, rho_arch, k1, k2)
            ll_prop = multinomial_log_lik(pi_prop, arch_prop, N_ctheta)
            if ll_prop >= ll_cur - merge_ll_tolerance:
                pi_arch = pi_prop
                arch_assignment = arch_prop
                tsb_new, rho_arch = _rebuild_rho(arch_assignment)
                dyn.tsb_betas_arch = tsb_new
                info['merges'].append({
                    'k1': int(k1), 'k2': int(k2),
                    'tvd': float(tvd),
                    'll_delta': float(ll_prop - ll_cur),
                })
                ll_cur = ll_prop
                info['n_moves_accepted'] += 1
                move_accepted_this_round = True
                if verbose:
                    print(f"    [split-merge] MERGE ({k1}, {k2}) -> {k1}; "
                            f"tvd={tvd:.3f}  dLL={ll_prop - info['ll_before']:+.1f}",
                            flush=True)

        if not move_accepted_this_round:
            break

    if info['n_moves_accepted'] > 0:
        dyn.pi_archetype = pi_arch
        dyn.arch_assignment = arch_assignment
        dyn.rho_arch = rho_arch
        dyn.materialise_pi_field()
    elif verbose:
        print(f"    [split-merge] no moves accepted; ll={ll_cur:.1f}",
                flush=True)

    info['ll_after'] = float(ll_cur)
    return info
