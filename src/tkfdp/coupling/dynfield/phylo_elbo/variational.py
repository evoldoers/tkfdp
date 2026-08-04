"""Per-node variational state for the moment-matching family.

At every tree node v, restrict the joint (x_v, theta_v) marginal to

  q_v(x_v, theta_v) = q_v(theta_v) * [
    (1 - lambda_v(theta_v)) * prod_n p_n_v(x_v_n | theta_v)
      + lambda_v(theta_v) * prod_n pi^{arch(c_n, theta_v)}(x_v_n)]

with parameters at v:

  q_v[theta]           (L,) probabilities over field states
  p_v[n, theta, aa]    (m, L, A) per-site per-theta residue marginal on
                        the "tracking" component
  lambda_v[theta]      (L,) mixture weight of the "regenerated"
                        component (prob at least one field event has
                        occurred on the path from v to its last
                        "tracking" ancestor)

Storage per node: O(L*m*A + 2L). Across a tree of N nodes: O(N*L*m*A).

This module defines the state container and construction helpers only.
The moment-matching update primitives (per-edge propagation, internal-
node combining) live in `edge_propagate.py` and `moment_match.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NodeState:
    """Variational parameters at a single node for a single cluster."""
    q: np.ndarray                        # (L,) probabilities
    p: np.ndarray                        # (m, L, A) residue marginals
    lam: np.ndarray                      # (L,) mixture weights in [0, 1]

    def L(self) -> int:
        return int(self.q.shape[0])

    def m(self) -> int:
        return int(self.p.shape[0])

    def A(self) -> int:
        return int(self.p.shape[2])

    def check(self) -> None:
        L, m, A = self.L(), self.m(), self.A()
        assert self.q.shape == (L,)
        assert self.p.shape == (m, L, A)
        assert self.lam.shape == (L,)
        # Simplex checks.
        assert np.isfinite(self.q).all()
        assert self.q.min() >= -1e-9, f"q has negative entries: {self.q.min()}"
        assert abs(self.q.sum() - 1.0) < 1e-6, f"q sums to {self.q.sum()}"
        assert (self.lam >= -1e-9).all() and (self.lam <= 1 + 1e-9).all()
        # p is (m, L, A); each slice p[n, theta, :] is a simplex.
        row_sums = self.p.sum(axis=2)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"per-site p rows do not sum to 1: min={row_sums.min()}, "
            f"max={row_sums.max()}")


@dataclass
class VariationalState:
    """All node states for a single cluster on a single tree.

    Attributes:
      states: list of NodeState, indexed by node id 0..n_nodes-1.
      L, m, A: dimensions inherited from any NodeState.
    """
    states: 'list[NodeState]'

    def __getitem__(self, v: int) -> NodeState:
        return self.states[int(v)]

    def __len__(self) -> int:
        return len(self.states)

    @property
    def L(self) -> int:
        return self.states[0].L()

    @property
    def m(self) -> int:
        return self.states[0].m()

    @property
    def A(self) -> int:
        return self.states[0].A()

    def check(self) -> None:
        for v, s in enumerate(self.states):
            s.check()
            assert s.L() == self.L
            assert s.m() == self.m
            assert s.A() == self.A


# ---------------------------------------------------------------------------
# Initialisation helpers.
# ---------------------------------------------------------------------------


def init_leaf(leaf_obs_row: np.ndarray,
                pi_arch: np.ndarray,
                arch_assignment: np.ndarray,
                cls_of_cluster: np.ndarray,
                rho: np.ndarray) -> NodeState:
    """Initialise a leaf variational state from observations.

    At a leaf the residue is *observed* (or gapped). We initialise:
      q(theta) = rho[theta]                    (weak prior over theta)
      p[n, theta, :] = delta at obs residue    (for observed positions)
                     = uniform                (for gapped positions;
                                                  1/A ~ uninformative)
      lam(theta) = 0                           (fully tracking at leaf)

    Args:
      leaf_obs_row: (m,) int32; residue at each cluster column at this
        leaf; -1 marks gapped.
      pi_arch: (K_a, A) archetype simplex points (only used to know A).
      arch_assignment: (K_c, L) int; unused at leaf init but signature
        symmetry with internal-node init.
      cls_of_cluster: (m,) int; per-column class assignment; unused at
        leaf init.
      rho: (L,) TSB weights over field states.

    Returns a NodeState.
    """
    m = int(leaf_obs_row.shape[0])
    L = int(rho.shape[0])
    A = int(pi_arch.shape[1])
    q = np.asarray(rho, dtype=np.float64).copy()
    q = q / max(q.sum(), 1e-300)
    p = np.full((m, L, A), 1.0 / A, dtype=np.float64)
    for n in range(m):
        x = int(leaf_obs_row[n])
        if x >= 0:
            p[n, :, :] = 0.0
            p[n, :, x] = 1.0
    lam = np.zeros(L, dtype=np.float64)
    st = NodeState(q=q, p=p, lam=lam)
    st.check()
    return st


def init_internal_uniform(m: int, L: int, A: int, rho: np.ndarray
                              ) -> NodeState:
    """Cold-start init for an internal node: q = rho, p = uniform,
    lambda = 0.5 (a priori we don't know)."""
    q = np.asarray(rho, dtype=np.float64).copy()
    q = q / max(q.sum(), 1e-300)
    p = np.full((m, L, A), 1.0 / A, dtype=np.float64)
    lam = np.full(L, 0.5, dtype=np.float64)
    st = NodeState(q=q, p=p, lam=lam)
    st.check()
    return st


def init_variational_state(tree,
                              pi_arch: np.ndarray,
                              arch_assignment: np.ndarray,
                              cls_of_cluster: np.ndarray,
                              rho: np.ndarray) -> VariationalState:
    """Initialise VariationalState across all nodes of the tree.

    Leaves are initialised from observations; internals get the
    uniform-uninformative cold start.
    """
    L = int(rho.shape[0])
    A = int(pi_arch.shape[1])
    m = int(tree.m)
    states: 'list[NodeState]' = []
    for v in range(tree.n_nodes):
        if tree.is_leaf(v):
            states.append(init_leaf(tree.leaf_obs[v], pi_arch,
                                        arch_assignment, cls_of_cluster,
                                        rho))
        else:
            states.append(init_internal_uniform(m, L, A, rho))
    vs = VariationalState(states=states)
    vs.check()
    return vs
