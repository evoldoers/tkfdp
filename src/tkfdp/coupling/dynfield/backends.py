"""Training-backend abstraction for dynfield archetype training.

The training loop consumes two E-step outputs (per outer iter):

  1. Soft sufficient statistics: `soft_stats(model, per_family_data)`
     returning a dict with `N` (per-(c, theta, a) residue-occupancy
     tensor), `r` (per-theta cluster-count vector), `log_lik` (data log-
     likelihood under the current model), `n_clust` (number of scoring
     units). Consumed by:
       - The rho TSB posterior-mean update (via `r`);
       - soft_arch_posterior for archetype assignments (via `N`);
       - Split-merge acceptance / multinomial LL (via `N`);
       - LL trajectory reporting (via `log_lik`).

  2. Trajectory sufficient statistics for the HR M-step:
     `hr_stats(model, per_family_data, ...)` returning
     {`V`, `U`, `W`, `N_theta_sum`, `T_sum`, `log_lik`, `n_clust`}
     (and optionally `q_inv_mean`, `q_bin_mean` under +Gamma+I). These
     drive:
       - The pi_arch Newton MAP update (eq:arch-Mstep-pi);
       - The rho_chain closed-form Gamma posterior (eq:arch-Mstep-rho);
       - The p_inv Beta-conjugate update (par:arch-gamma-plus-I).

Downstream M-steps are agnostic to the *source* of these tensors. This
module defines a `DynfieldStatsBackend` protocol so an alternative
tree-based training regime (par:arch-phylo-elbo) can supply the same
tensors by variational message-passing on a tree, then be selected at
CLI time without touching the training loop.

Two implementations:

  - `CompositeBackend`: the existing cherry-pair training regime.
    Cherry-independent per-branch HR passes, marginalising theta_X
    against the prior rho[theta] independently per cherry. Wraps
    `updates.accumulate_cluster_stats_soft` and
    `hr_jax.accumulate_cluster_stats_hr_jax[_gammaI]`.

  - `PhyloELBOBackend`: NOT YET IMPLEMENTED. Placeholder + design
    docstring; raises `NotImplementedError` with the concrete plan.
    Would use variational message-passing on a tree (loaded from a
    preprocessing cache) with the moment-matching family of
    par:arch-phylo-elbo, then extract per-branch HR statistics from
    the converged variational posterior (par:arch-phylo-elbo-hr).

CLI dispatch is via `--training-mode {composite, phylo-elbo}` on
`experiments/train_dynfield.py`. Composite is the default.
"""
from __future__ import annotations

from typing import Optional, Protocol


class DynfieldStatsBackend(Protocol):
    """E-step interface consumed by the archetype trainer."""

    name: str  # "composite" or "phylo-elbo"

    def soft_stats(self, model, cluster_observations: list) -> dict:
        """Return {'N': (K_c, L_max, A), 'r': (L_max,), 'log_lik': float,
        'n_clust': int}. Used by rho TSB update, arch soft posterior,
        split-merge, LL reporting."""
        ...

    def hr_stats(self, model, cluster_observations: list,
                    bins_per_cluster: 'list | None' = None,
                    bin_means_full_site: 'list | None' = None,
                    use_gammaI: bool = False) -> dict:
        """Return {'V': (K_c, L_max, A), 'U': (K_c, L_max, A, A),
        'W': (K_c, L_max, A), 'N_theta_sum': float, 'T_sum': float,
        'log_lik': float, 'n_clust': int}. Optionally 'q_inv_mean',
        'q_bin_mean' under +Gamma+I. Consumed by pi_arch Newton,
        rho_chain gamma, p_inv Beta-conjugate M-steps."""
        ...


class CompositeBackend:
    """Existing cherry-pair regime. Delegates to
    `updates.accumulate_cluster_stats_soft` and
    `hr_jax.accumulate_cluster_stats_hr_jax[_gammaI]`.

    This wrapper keeps the current training loop's semantics unchanged
    while exposing the backend-protocol interface. Every marginalisation
    over theta_X is done independently per cherry against the prior
    rho[theta]; consequently a per-MSA composite-likelihood freedom
    (par:arch-phylo-elbo, motivational paragraph) is inherent to this
    backend --- diagnosable via analysis/archetype_biophysics.py step 6.
    """

    name = "composite"

    def soft_stats(self, model, cluster_observations):
        from .updates import accumulate_cluster_stats_soft
        import numpy as np
        c_obs = [(cls, X, Y) for (cls, X, Y, _) in cluster_observations]
        c_t = np.asarray([t for (_, _, _, t) in cluster_observations],
                            dtype=np.float64)
        return accumulate_cluster_stats_soft(model, c_obs, t_per_cluster=c_t)

    def hr_stats(self, model, cluster_observations,
                    bins_per_cluster=None, bin_means_full_site=None,
                    use_gammaI=False):
        import numpy as np
        c_obs = [(cls, X, Y) for (cls, X, Y, _) in cluster_observations]
        c_t = np.asarray([t for (_, _, _, t) in cluster_observations],
                            dtype=np.float64)
        if use_gammaI:
            from .hr_jax import accumulate_cluster_stats_hr_jax_gammaI
            return accumulate_cluster_stats_hr_jax_gammaI(
                model, c_obs, t_per_cluster=c_t,
                bins_per_cluster=bins_per_cluster,
                bin_means_full_site=bin_means_full_site)
        else:
            from .hr_jax import accumulate_cluster_stats_hr_jax
            return accumulate_cluster_stats_hr_jax(
                model, c_obs, t_per_cluster=c_t,
                bins_per_cluster=bins_per_cluster,
                bin_means_full=bin_means_full_site)


class PhyloELBOBackend:
    """NOT YET IMPLEMENTED. Tree-ELBO variational training backend.

    See appendix par:arch-phylo-elbo and par:arch-phylo-elbo-hr for the
    full derivation. Concrete implementation plan:

      1. Preprocessing (offline, cached):
         For each Pfam family, build a guide tree from the aligned
         sequences via FastTree or RAxML. Store as JSON alongside the
         family's cherry npz:
             {'topology': [(parent_idx, child_idx, tau), ...],
              'leaf_seqs': [(leaf_idx, aa_sequence_int32), ...],
              'internal_nodes': [node_idx, ...]}
         Attach class assignments and site rate bins from the current
         SVI state per iter (not baked into the tree cache; those are
         model state that changes).

      2. Variational family (per node v, per cluster):
         q_v(theta_v) in Delta^{L-1}
         p_{n, v}(x_v_n | theta_v) in Delta^{A-1}   for each site n
         lambda_v(theta_v) in [0, 1]
         Total per-node storage O(L * m * A + 2L).

      3. Message-passing sweeps:
         - Post-order (leaves -> root): edge propagation preserves the
           (rank-1 + scalar) form; internal-node combines project via
           moment matching (q_v, per-site marginals p_{n,v}, lambda_v
           all closed form).
         - Pre-order (root -> leaves): backward messages update parents'
           messages into each child's variational marginal.
         - Loop until ELBO change < tolerance (~10-50 sweeps typical).

      4. HR sufficient statistics from converged q:
         Per branch v -> u of length tau_vu, per (theta_v, theta_u):
             w_vu(theta_v, theta_u) = q_v(theta_v)
                                       * q_u(theta_u | theta_v; tau_vu)
             (V_n, U_n, W_n) = standard whole-branch HR primitives
                                (par:arch-hr) at fixed (theta_v, theta_u)
                                using per-site pair distributions
                                factored out of q_v x q_u | branch.
         Aggregate as
             V[k, a] += sum_{vu, n, theta_v, theta_u} w * [arch=k] * V_n
         and similarly for U, W. Field-jump counts and time sums use
             sum_q N_theta += sum_{vu, theta_v, theta_u} w
                                * E[N_theta | theta_v, theta_u, tau_vu]
             sum_q T       += sum_vu tau_vu.

      5. Backend wiring:
         `soft_stats` and `hr_stats` both return dicts with the same
         keys as CompositeBackend, but the tensors are computed by
         summing the variational per-branch expectations rather than
         by cherry-independent HR passes. The rest of the trainer is
         untouched.

    Notes on complexity:
      Per cluster per iter: O(edges * L^2 * m * A^2) per sweep,
      * O(10-50) sweeps for ELBO convergence. For a ~50-leaf per-family
      subtree and m ~ 5 columns per cluster this is ~L^2 * 250 * A^2 ~
      1e7 flops per cluster per iter; manageable in JAX.

      For scale: sample a ~50-leaf subtree per family per iter rather
      than propagating the full ~1000-leaf tree, since the ELBO
      approximation degrades gracefully with depth and the M-step is
      driven by aggregated stats across the corpus.

    Trigger: composite training's step 6 diagnostic reports an artifact
    fraction above threshold, or the user's downstream evaluation calls
    for tree-based ancestor reconstruction.

    Author's note: the ~600-1200 LOC + tests for a full implementation
    are out of scope for the scaffold commit; this class raises
    NotImplementedError until then. See task #109.
    """

    name = "phylo-elbo"

    def soft_stats(self, model, cluster_observations):
        raise NotImplementedError(
            "PhyloELBOBackend not yet implemented. See "
            "src/tkfdp/coupling/dynfield/backends.py for the plan. "
            "Composite training (--training-mode composite) is the "
            "operational default and reflects the training regime "
            "reported in the paper. See task #109 for progress.")

    def hr_stats(self, model, cluster_observations,
                    bins_per_cluster=None, bin_means_full_site=None,
                    use_gammaI=False):
        raise NotImplementedError(
            "PhyloELBOBackend not yet implemented. See "
            "src/tkfdp/coupling/dynfield/backends.py for the plan. "
            "Composite training (--training-mode composite) is the "
            "operational default and reflects the training regime "
            "reported in the paper. See task #109 for progress.")


def make_backend(training_mode: str) -> DynfieldStatsBackend:
    """Factory: dispatch a backend by CLI name."""
    training_mode = training_mode.strip().lower()
    if training_mode == "composite":
        return CompositeBackend()
    if training_mode == "phylo-elbo":
        return PhyloELBOBackend()
    raise ValueError(
        f"unknown training_mode={training_mode!r}; expected "
        f"'composite' or 'phylo-elbo'")
