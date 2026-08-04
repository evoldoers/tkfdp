"""Smoke-run of the phylo-ELBO Gibbs training loop on a synthetic corpus.

Milestone 7 of docs/phylo_elbo_gibbs_jax_plan.md.

Runs the full Gibbs sweep loop (arch_assignment + c_n + rho_chain
M-step) on a small synthetic corpus and reports:
  - corpus log-likelihood per iter (should trend upward)
  - fraction of arch_assignment entries matching a known "true"
    assignment (planted in the simulation)
  - fraction of c_n entries matching a known "true" per-site class

Since the phylo-ELBO Gibbs infrastructure isn't wired into the main
tied-theta trainer yet (that requires per-family FastTree guide-tree
preprocessing and cluster-subtree extraction, which is a larger
follow-up), this script serves as the M7 sanity check.

Usage:
    python experiments/phylo_elbo_gibbs_smoke.py [--n-iters 5]
                                                  [--n-clusters 20]
                                                  [--seed 42]
"""
from __future__ import annotations

import argparse

import numpy as np

from tkfdp.coupling.dynfield.phylo_elbo.gibbs_arch import (
    gibbs_arch_assignment_sweep)
from tkfdp.coupling.dynfield.phylo_elbo.gibbs_cn import (
    gibbs_cn_sweep)
from tkfdp.coupling.dynfield.phylo_elbo.tree import (
    make_balanced_binary, make_cherry)
from tkfdp.coupling.dynfield.phylo_elbo.tree_batch import (
    bucketed_tree_log_lik)


def _synthesize_corpus(rng, n_clusters, K_c, K_a, L, A, rho, rho_chain,
                          pi_archetype, true_arch, m_range=(1, 4)):
    """Synthesize `n_clusters` clusters (cherries + depth-2 mix) with
    "true" per-cluster c_n labels. Leaf observations drawn to be
    highly informative under the true (c, arch) model."""
    clusters = []
    true_classes_list = []
    for _ in range(n_clusters):
        shape = rng.choice(['cherry', 'depth2'])
        m = int(rng.integers(m_range[0], m_range[1] + 1))
        true_classes = rng.integers(0, K_c, size=m).astype(np.int32)
        n_leaves = 2 if shape == 'cherry' else 4
        # Draw theta_root; site residues drawn from pi_archetype[
        # true_arch[true_c_n, theta_root]] and replicated across leaves
        # (no substitution -- makes signal strong for testing).
        theta_root = int(rng.choice(L, p=rho))
        leaf_obs = np.zeros((n_leaves, m), dtype=np.int32)
        for n in range(m):
            c = int(true_classes[n])
            k = int(true_arch[c, theta_root])
            a = int(rng.choice(A, p=pi_archetype[k]))
            leaf_obs[:, n] = a
        if shape == 'cherry':
            tree = make_cherry(tau=0.1, leaf_obs=leaf_obs)
        else:
            tree = make_balanced_binary(
                depth=2, tau=0.1, leaf_obs=leaf_obs)
        # Init classes RANDOMLY (not true).
        init_classes = rng.integers(0, K_c, size=m).astype(np.int32)
        clusters.append((tree, init_classes))
        true_classes_list.append(true_classes)
    return clusters, true_classes_list


def _corpus_log_lik(clusters, pi_field, rho, S, rho_chain):
    """Total corpus log-likelihood via the bucketed pipeline."""
    lls = bucketed_tree_log_lik(clusters, rho, pi_field, S, rho_chain)
    return float(lls.sum())


def _frac_correct_arch(aa, true_arch, fix_theta0=True):
    L = aa.shape[1]
    theta_range = np.arange(1, L) if fix_theta0 else np.arange(L)
    correct = int((aa[:, theta_range] == true_arch[:, theta_range]).sum())
    total = int(aa.shape[0] * len(theta_range))
    return correct / total


def _frac_correct_classes(clusters, true_classes_list):
    total = 0; correct = 0
    for i, (_, classes) in enumerate(clusters):
        m = int(classes.shape[0])
        total += m
        correct += int((classes == true_classes_list[i]).sum())
    return correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument("--n-iters", type=int, default=5)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fix-theta0", action='store_true', default=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    L = 3; K_c = 3; K_a = 3; A = 4
    rho_chain = 0.05
    rho = np.full(L, 1.0 / L)
    rho_arch = np.full(K_a, 1.0 / K_a)
    S = np.ones((A, A)) - np.eye(A)

    # Sharp K_a=3 archetypes.
    pi_archetype = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
    ])
    # True arch_assignment: cyclic non-identity at theta != 0.
    true_arch = np.array([
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ], dtype=np.int32)

    print(f"# Synthesizing corpus (n_clusters={args.n_clusters}, "
            f"K_c={K_c}, K_a={K_a}, L={L}, A={A}) ...")
    clusters, true_classes_list = _synthesize_corpus(
        rng, args.n_clusters, K_c, K_a, L, A, rho, rho_chain,
        pi_archetype, true_arch)

    # Initial: identity arch_assignment; random c_n (already random in
    # synthesize).
    aa = np.tile(np.arange(K_c)[:, None], (1, L)).astype(np.int32)

    def eval_now():
        pi_field = pi_archetype[aa]  # (K_c, L, A)
        ll = _corpus_log_lik(clusters, pi_field, rho, S, rho_chain)
        aa_acc = _frac_correct_arch(aa, true_arch, args.fix_theta0)
        cn_acc = _frac_correct_classes(clusters, true_classes_list)
        return ll, aa_acc, cn_acc

    ll, aa_acc, cn_acc = eval_now()
    print(f"# init:   corpus LL = {ll:+.4f}  arch_correct={aa_acc:.2f}"
            f"  cn_correct={cn_acc:.2f}")

    for it in range(args.n_iters):
        # arch_assignment Gibbs.
        aa, ai = gibbs_arch_assignment_sweep(
            clusters, pi_archetype, aa, rho_arch, rho, S, rho_chain,
            rng, fix_theta0=args.fix_theta0)
        # Derive pi_field from updated aa.
        pi_field = pi_archetype[aa]

        # c_n Gibbs.
        new_cls, cnfo = gibbs_cn_sweep(
            clusters, pi_field, rho, S, rho_chain, K_c, rng)
        clusters = [(clusters[i][0], new_cls[i])
                     for i in range(len(clusters))]

        ll, aa_acc, cn_acc = eval_now()
        print(f"# iter {it + 1}: corpus LL = {ll:+.4f}  "
                f"arch_correct={aa_acc:.2f}  cn_correct={cn_acc:.2f}  "
                f"aa_changed={ai['n_changed']}  cn_changed="
                f"{cnfo['n_changed']}/{cnfo['n_sites_total']}")

    print("\n# Final arch_assignment:")
    print(aa)
    print("\n# True arch_assignment:")
    print(true_arch)


if __name__ == "__main__":
    main()
