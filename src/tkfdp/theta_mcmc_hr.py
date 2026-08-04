"""HR sufficient-statistic accumulator under fully-observed (X, theta, M).

Given the tied-theta MCMC sampler in theta_mcmc.py, we have per
(family, cluster) the sampled theta_v at every node and a binary jump
indicator M_v per branch. This module converts those samples into
corpus-level HR sufficient statistics (V, U, W, N_theta_sum, T_sum)
that plug directly into the existing pi_arch / rho_chain M-step.

Attribution:
  - Branches with M_v = 0 (theta_p = theta_c = theta): standard GTR
    bridge HR SS at archetype k = arch_assignment[c_s, theta] using
    the existing gtr_bridge_hr_jax primitive. Contribution to V at
    (k, X_p^s), W at (k, :), U at (k, :, :).
  - Branches with M_v = 1: per-site case-jump SS attribution using
    the case_jump primitive in theta_mcmc_hr_case1.py — pre-jump
    segment under k_p = arch[c_s, theta_p], middle segments across
    all L archetypes (via E_arrivals), post-jump segment under
    k_c = arch[c_s, theta_c]. Plus E_N_jump per M=1 branch summed
    into N_theta_sum (not just the count of M=1 branches). Matches
    the closed forms of par:arch-hr in appendix-tkfdp.tex, extracted
    from _case_jump in coupling/dynfield/hr_jax.py with theta_X,
    theta_Y treated as fixed (sampled) instead of marginalised.
  - T_sum: sum tau_v over ALL branches with observations.

Output format matches accumulate_cluster_stats_hr_jax so downstream
M-step (update_pi_archetype_gtr, update_rho_chain_gamma) is unchanged.
"""
from __future__ import annotations

import numpy as np


A_ALPH = 20


def _build_batched_bridge_kernel():
    """Return a jit-compiled vmap of gtr_bridge_hr_jax over (k, X, Y, t,
    m) per-call arrays, gathering eigendecomps from per-archetype tensors.
    m is the per-site Γ+I rate multiplier; passed as m·ξ into
    gtr_bridge_hr_jax, and used to rescale U back to physical units
    (Q^{k, m}·W) after the primitive returns U/m from the ξ_scaled
    denominator in J."""
    import jax
    import jax.numpy as jnp
    from tkfdp.coupling.dynfield.hr_jax import gtr_bridge_hr_jax

    def _one(k_i, x_p, x_c, tau_v, m_site, pi_arch_all, xi_all, U_arch_all,
             D_arch_all, S):
        pi_k = pi_arch_all[k_i]
        xi_scaled = m_site * xi_all[k_i]
        U_k = U_arch_all[k_i]
        D_k = D_arch_all[k_i]
        P, W_i, U_ij = gtr_bridge_hr_jax(
            pi_k, xi_scaled, U_k, D_k, S, x_p, x_c, tau_v)
        # Defensive: for invariant sites (m=0), gtr_bridge_hr_jax
        # returns W_i and U_ij normalised by max(P_XY, 1e-300); when
        # X_p != X_c under m=0 the true P_XY is 0, and the 1e-300
        # floor inflates outputs to ~1e300. Downstream V/W/U scatter
        # then poisons the archetype M-step. Detect m=0 + X_p != X_c
        # and zero the emission contribution — physically an
        # impossible branch that should contribute nothing to SS.
        invariant_impossible = (m_site == 0.0) & (x_p != x_c)
        W_i_safe = jnp.where(invariant_impossible, jnp.zeros_like(W_i), W_i)
        U_ij_safe = jnp.where(invariant_impossible, jnp.zeros_like(U_ij),
                                    m_site * U_ij)
        return P, W_i_safe, U_ij_safe
    batched = jax.vmap(_one, in_axes=(0, 0, 0, 0, 0, None, None, None,
                                        None, None))
    return jax.jit(batched)


_batched_bridge_kernel = None


def _get_batched_kernel():
    global _batched_bridge_kernel
    if _batched_bridge_kernel is None:
        _batched_bridge_kernel = _build_batched_bridge_kernel()
    return _batched_bridge_kernel


def accumulate_hr_ss_tied_theta(state,
                                        per_family_data: list,
                                        theta_samples: dict,
                                        M_samples: dict,
                                        cluster_columns_by_fam_cid: dict,
                                        cluster_classes_by_fam_cid: dict,
                                        cluster_branches_by_fam_cid: dict,
                                        S: np.ndarray,
                                        bin_means_full: 'np.ndarray | None' = None,
                                        batch_size: int = 524288,
                                        case_jump_batch_size: int = 65536,
                                        ) -> dict:
    """Corpus-level HR SS under sampled (theta, M).

    Vectorised: M=0 bridge SS and M=1 case-jump SS calls are each
    collected into flat arrays and dispatched through jit-compiled
    vmap kernels. Scatter into V, W, U tensors is done with
    per-unique-key bucketing on the host.

    case_jump_batch_size is smaller than batch_size because each
    M=1 call returns (L, A) V/W and (L, A, A) U tensors (~14 KB at
    L=4, A=20) versus (A,) W and (A, A) U (~3.4 KB) for M=0. 65k
    calls * 14 KB = ~900 MB per batch on GPU, well within budget.
    """
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from tkfdp.theta_mcmc_hr_case1 import (
        get_case_jump_kernel, get_E_N_jump_kernel)

    dyn = state.dyn_field
    pi_arch = np.asarray(dyn.pi_archetype, dtype=np.float64)
    arch_assignment = np.asarray(dyn.arch_assignment, dtype=np.int32)
    rho_np = np.asarray(dyn.rho, dtype=np.float64)
    rho_chain_f = float(dyn.rho_chain)
    K_a, A = pi_arch.shape
    K_c, L_theta = arch_assignment.shape

    xi_all = np.zeros((K_a, A), dtype=np.float64)
    U_arch_all = np.zeros((K_a, A, A), dtype=np.float64)
    D_arch_all = np.zeros((K_a, A), dtype=np.float64)
    for k in range(K_a):
        pi_k = pi_arch[k]
        Q_off = S * pi_k[None, :]
        Q_off = Q_off - np.diag(np.diag(Q_off))
        Q_diag = -Q_off.sum(axis=1)
        Q_mat = Q_off + np.diag(Q_diag)
        D_half = np.sqrt(np.maximum(pi_k, 1e-300))
        Q_sym = (D_half[:, None] * Q_mat) / D_half[None, :]
        Q_sym = 0.5 * (Q_sym + Q_sym.T)
        xi, U_mat = np.linalg.eigh(Q_sym)
        xi_all[k] = xi
        U_arch_all[k] = U_mat
        D_arch_all[k] = D_half
    S_j = jnp.asarray(S, dtype=jnp.float64)
    pi_arch_j = jnp.asarray(pi_arch, dtype=jnp.float64)
    xi_all_j = jnp.asarray(xi_all, dtype=jnp.float64)
    U_arch_all_j = jnp.asarray(U_arch_all, dtype=jnp.float64)
    D_arch_all_j = jnp.asarray(D_arch_all, dtype=jnp.float64)
    rho_j = jnp.asarray(rho_np, dtype=jnp.float64)
    rho_chain_j = jnp.float64(rho_chain_f)
    arch_j = jnp.asarray(arch_assignment, dtype=jnp.int32)

    # M=0 call buffers.
    M0_calls_k: 'list[int]' = []
    M0_calls_xp: 'list[int]' = []
    M0_calls_xc: 'list[int]' = []
    M0_calls_tau: 'list[float]' = []
    M0_calls_cs: 'list[int]' = []
    M0_calls_theta: 'list[int]' = []
    M0_calls_m: 'list[float]' = []
    # M=1 per-site call buffers.
    M1_calls_cs: 'list[int]' = []
    M1_calls_thp: 'list[int]' = []
    M1_calls_thc: 'list[int]' = []
    M1_calls_tau: 'list[float]' = []
    M1_calls_xp: 'list[int]' = []
    M1_calls_xc: 'list[int]' = []
    M1_calls_m: 'list[float]' = []
    # M=1 per-branch buffers (for E_N_jump).
    M1_br_thp: 'list[int]' = []
    M1_br_thc: 'list[int]' = []
    M1_br_tau: 'list[float]' = []

    T_sum = 0.0
    n_clust = 0

    V_total = np.zeros((K_c, L_theta, A), dtype=np.float64)
    U_total = np.zeros((K_c, L_theta, A, A), dtype=np.float64)
    W_total = np.zeros((K_c, L_theta, A), dtype=np.float64)

    for fam_idx, fd in enumerate(per_family_data):
        theta_by_cid = theta_samples.get(fam_idx, {})
        if not theta_by_cid:
            continue
        aa_a_full = fd['aa_a']
        aa_b_full = fd['aa_b']
        tau_branch = fd['tau']
        both_aa = fd['both_aa']

        cc_map = cluster_columns_by_fam_cid.get(fam_idx, {})
        cl_map = cluster_classes_by_fam_cid.get(fam_idx, {})
        br_map = cluster_branches_by_fam_cid.get(fam_idx, {})

        # Per-site rate multiplier m from Γ+I bins (if enabled). Default
        # m=1 for every site — physically "no rate heterogeneity".
        st_fam = state.states_per_msa[fam_idx]
        _srb = getattr(st_fam, 'site_rate_bin', None)
        if _srb is not None and bin_means_full is not None:
            m_per_col_fam = np.asarray(
                bin_means_full, dtype=np.float64)[
                np.asarray(_srb, dtype=np.int64)]
        else:
            n_cols = int(both_aa.shape[1])
            m_per_col_fam = np.ones(n_cols, dtype=np.float64)

        for cid, theta_sampled in theta_by_cid.items():
            M_per_branch = M_samples[fam_idx][cid]
            cluster_columns = cc_map[cid]
            classes_c = cl_map[cid]
            branches = br_map[cid]
            m = int(cluster_columns.shape[0])
            n_branches = int(branches.shape[0])
            n_clust += 1

            child_ids = branches[:, 1]
            theta_p_v = theta_sampled[branches[:, 0]]
            theta_c_v = theta_sampled[child_ids]
            M_v = M_per_branch[child_ids]
            tau_v = tau_branch
            obs_matrix = both_aa[:, cluster_columns]
            has_obs = obs_matrix.any(axis=1)

            T_sum += float(np.sum(tau_v[has_obs]))

            # --- M=0 branches ---
            mask_M0 = (M_v == 0) & has_obs
            if mask_M0.any():
                active_branches = np.flatnonzero(mask_M0)
                X_p_at = aa_a_full[active_branches][:, cluster_columns]
                Y_c_at = aa_b_full[active_branches][:, cluster_columns]
                obs_at = obs_matrix[active_branches]
                theta_at = theta_p_v[active_branches]
                tau_at = tau_v[active_branches]
                n_active = int(active_branches.shape[0])
                m_col_cluster = m_per_col_fam[cluster_columns]  # (m,)
                classes_bcast = np.broadcast_to(
                    classes_c[None, :], (n_active, m))
                theta_bcast = np.broadcast_to(
                    theta_at[:, None], (n_active, m))
                tau_bcast = np.broadcast_to(
                    tau_at[:, None], (n_active, m))
                m_bcast = np.broadcast_to(
                    m_col_cluster[None, :], (n_active, m))
                valid_mask = obs_at & (X_p_at < 20) & (Y_c_at < 20)
                flat_valid = valid_mask.ravel()
                if flat_valid.any():
                    idx_valid = np.flatnonzero(flat_valid)
                    cs_flat = classes_bcast.ravel()[idx_valid]
                    th_flat = theta_bcast.ravel()[idx_valid]
                    tau_flat = tau_bcast.ravel()[idx_valid]
                    m_flat = m_bcast.ravel()[idx_valid]
                    xp_flat = X_p_at.ravel()[idx_valid]
                    yc_flat = Y_c_at.ravel()[idx_valid]
                    k_flat = arch_assignment[cs_flat, th_flat]

                    np.add.at(V_total, (cs_flat, th_flat, xp_flat), 1.0)

                    M0_calls_k.extend(k_flat.tolist())
                    M0_calls_xp.extend(xp_flat.tolist())
                    M0_calls_xc.extend(yc_flat.tolist())
                    M0_calls_tau.extend(tau_flat.tolist())
                    M0_calls_cs.extend(cs_flat.tolist())
                    M0_calls_theta.extend(th_flat.tolist())
                    M0_calls_m.extend(m_flat.tolist())

            # --- M=1 branches ---
            mask_M1 = (M_v == 1) & has_obs
            if mask_M1.any():
                active_M1 = np.flatnonzero(mask_M1)
                # Per-branch collection (for E_N_jump).
                M1_br_thp.extend(theta_p_v[active_M1].tolist())
                M1_br_thc.extend(theta_c_v[active_M1].tolist())
                M1_br_tau.extend(tau_v[active_M1].tolist())
                # Per-site collection (for V/W/U).
                X_p_at1 = aa_a_full[active_M1][:, cluster_columns]
                Y_c_at1 = aa_b_full[active_M1][:, cluster_columns]
                obs_at1 = obs_matrix[active_M1]
                theta_p_at1 = theta_p_v[active_M1]
                theta_c_at1 = theta_c_v[active_M1]
                tau_at1 = tau_v[active_M1]
                n_active_M1 = int(active_M1.shape[0])
                m_col_cluster1 = m_per_col_fam[cluster_columns]
                classes_bcast1 = np.broadcast_to(
                    classes_c[None, :], (n_active_M1, m))
                theta_p_bcast1 = np.broadcast_to(
                    theta_p_at1[:, None], (n_active_M1, m))
                theta_c_bcast1 = np.broadcast_to(
                    theta_c_at1[:, None], (n_active_M1, m))
                tau_bcast1 = np.broadcast_to(
                    tau_at1[:, None], (n_active_M1, m))
                m_bcast1 = np.broadcast_to(
                    m_col_cluster1[None, :], (n_active_M1, m))
                valid_mask1 = obs_at1 & (X_p_at1 < 20) & (Y_c_at1 < 20)
                flat_valid1 = valid_mask1.ravel()
                if flat_valid1.any():
                    idx_valid1 = np.flatnonzero(flat_valid1)
                    M1_calls_cs.extend(
                        classes_bcast1.ravel()[idx_valid1].tolist())
                    M1_calls_thp.extend(
                        theta_p_bcast1.ravel()[idx_valid1].tolist())
                    M1_calls_thc.extend(
                        theta_c_bcast1.ravel()[idx_valid1].tolist())
                    M1_calls_tau.extend(
                        tau_bcast1.ravel()[idx_valid1].tolist())
                    M1_calls_xp.extend(
                        X_p_at1.ravel()[idx_valid1].tolist())
                    M1_calls_xc.extend(
                        Y_c_at1.ravel()[idx_valid1].tolist())
                    M1_calls_m.extend(
                        m_bcast1.ravel()[idx_valid1].tolist())

    # ---- Dispatch: E_N_jump for N_theta_sum ----
    N_theta_sum = 0.0
    if M1_br_thp:
        E_N_kernel = get_E_N_jump_kernel()
        thp_arr = jnp.asarray(M1_br_thp, dtype=jnp.int32)
        thc_arr = jnp.asarray(M1_br_thc, dtype=jnp.int32)
        tau_arr = jnp.asarray(M1_br_tau, dtype=jnp.float64)
        E_N_v = np.asarray(
            E_N_kernel(rho_j, rho_chain_j, thp_arr, thc_arr, tau_arr))
        E_N_v = np.maximum(E_N_v, 0.0)
        N_theta_sum = float(E_N_v.sum())

    # ---- Dispatch: M=0 bridge SS ----
    n_calls_M0 = len(M0_calls_k)
    if n_calls_M0 > 0:
        kernel = _get_batched_kernel()
        k_arr = np.asarray(M0_calls_k, dtype=np.int32)
        xp_arr = np.asarray(M0_calls_xp, dtype=np.int32)
        xc_arr = np.asarray(M0_calls_xc, dtype=np.int32)
        tau_arr0 = np.asarray(M0_calls_tau, dtype=np.float64)
        m_arr0 = np.asarray(M0_calls_m, dtype=np.float64)
        cs_arr = np.asarray(M0_calls_cs, dtype=np.int64)
        th_arr = np.asarray(M0_calls_theta, dtype=np.int64)

        for start in range(0, n_calls_M0, batch_size):
            end = min(n_calls_M0, start + batch_size)
            k_c = jnp.asarray(k_arr[start:end])
            xp_c = jnp.asarray(xp_arr[start:end])
            xc_c = jnp.asarray(xc_arr[start:end])
            tau_c = jnp.asarray(tau_arr0[start:end])
            m_c = jnp.asarray(m_arr0[start:end])
            _, W_i, U_ij = kernel(
                k_c, xp_c, xc_c, tau_c, m_c,
                pi_arch_j, xi_all_j, U_arch_all_j, D_arch_all_j, S_j)
            W_np = np.asarray(W_i, dtype=np.float64)
            U_np = np.asarray(U_ij, dtype=np.float64)
            cs_c = cs_arr[start:end]
            th_c = th_arr[start:end]
            unique_pairs = np.unique(
                np.stack([cs_c, th_c], axis=1), axis=0)
            for pair in unique_pairs:
                cs_v, th_v = int(pair[0]), int(pair[1])
                mask = (cs_c == cs_v) & (th_c == th_v)
                if mask.any():
                    W_total[cs_v, th_v] += W_np[mask].sum(axis=0)
                    U_total[cs_v, th_v] += U_np[mask].sum(axis=0)

    # ---- Dispatch: M=1 case-jump SS ----
    n_calls_M1 = len(M1_calls_cs)
    if n_calls_M1 > 0:
        case_kernel = get_case_jump_kernel()
        cs_arr_M1 = np.asarray(M1_calls_cs, dtype=np.int32)
        thp_arr_M1 = np.asarray(M1_calls_thp, dtype=np.int32)
        thc_arr_M1 = np.asarray(M1_calls_thc, dtype=np.int32)
        tau_arr_M1 = np.asarray(M1_calls_tau, dtype=np.float64)
        xp_arr_M1 = np.asarray(M1_calls_xp, dtype=np.int32)
        xc_arr_M1 = np.asarray(M1_calls_xc, dtype=np.int32)
        m_arr_M1 = np.asarray(M1_calls_m, dtype=np.float64)
        for start in range(0, n_calls_M1, case_jump_batch_size):
            end = min(n_calls_M1, start + case_jump_batch_size)
            cs_c = jnp.asarray(cs_arr_M1[start:end])
            thp_c = jnp.asarray(thp_arr_M1[start:end])
            thc_c = jnp.asarray(thc_arr_M1[start:end])
            tau_c = jnp.asarray(tau_arr_M1[start:end])
            xp_c = jnp.asarray(xp_arr_M1[start:end])
            xc_c = jnp.asarray(xc_arr_M1[start:end])
            m_c = jnp.asarray(m_arr_M1[start:end])
            V_LA_j, W_LA_j, U_LAA_j, _EN_persite = case_kernel(
                pi_arch_j, xi_all_j, U_arch_all_j, D_arch_all_j, S_j,
                rho_j, rho_chain_j, arch_j,
                cs_c, thp_c, thc_c, xp_c, xc_c, tau_c, m_c)
            V_LA = np.asarray(V_LA_j, dtype=np.float64)
            W_LA = np.asarray(W_LA_j, dtype=np.float64)
            U_LAA = np.asarray(U_LAA_j, dtype=np.float64)
            cs_np = np.asarray(cs_arr_M1[start:end], dtype=np.int64)
            unique_cs = np.unique(cs_np)
            for cs_v in unique_cs:
                mask = cs_np == cs_v
                if mask.any():
                    V_total[cs_v] += V_LA[mask].sum(axis=0)
                    W_total[cs_v] += W_LA[mask].sum(axis=0)
                    U_total[cs_v] += U_LAA[mask].sum(axis=0)

    return dict(
        V=V_total, U=U_total, W=W_total,
        N_theta_sum=float(N_theta_sum),
        T_sum=float(T_sum),
        n_clust=int(n_clust),
        log_lik=0.0,
    )
