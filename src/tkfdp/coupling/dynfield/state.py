"""DynamicFieldState: the dynamic-latent-field analogue of PottsDPState.

Mirrors the shape contract of `PottsDPState` (see src/tkfdp/potts_dp.py) so
the SVI loop, checkpoint writer, and downstream tooling can dispatch by
`coupling_variant` and otherwise treat both variants uniformly.

Fields (the substantive variant-specific state):

  pi_field: (K_c, L_max, A) per-(class, field-atom) stationary tensor.
    The conditional residue stationary at site class c when the latent
    field is at atom theta. Each (K_c, L_max) row is a Dirichlet
    distribution over the alphabet.
  rho: (L_max,) DP stick-breaking weights for the field selector
    (the F81-on-DP atom mass; sums to 1).
  tsb_betas: (L_max - 1,) Beta(1, alpha_field) latent stick-breaking
    draws. `rho = stick_breaking_to_rho(tsb_betas)`.
  alpha_field: float, DP concentration parameter. Subject to
    Escobar-West auxiliary-variable Gibbs / Brent MAP updates in the
    SVI loop (D.1).
  rho_chain: float, the F81-on-DP rate multiplier from the
    supplement (sec:dynamic-suppl): `Q(theta -> theta') = rho_chain *
    rho_{theta'}` for theta != theta'. Controls how fast the field
    chain mixes:
      - rho_chain = 1.0 (default): standard F81 rate. No-jump probability
        over a half-edge of length t/2 is exp(-rho_chain * t / 2) under
        the carrier-vs-stationary mixture interpretation (see notes
        below on Interp 1 vs Interp 2).
      - rho_chain -> 0: field never jumps; dynfield reduces to per-class
        GTR with the initial-field stationary pi_field[c, theta_0].
        This is the correct "Potts limit" of the model.
      - rho_chain -> infty: field jumps continuously; residue
        immediately equilibrates to the field-marginal stationary
        pi_class[c]. Behaviour approaches a static-mixture model.
    NOTE on interpretation: the cap-2 cherry doublet closed form
    (docs/dynfield_math.md §3) uses a "carrier-vs-stationary mixture"
    formulation (Interp 1) where one Bernoulli event per edge governs
    whether the residue is carried (parent-evolved) or resampled at
    the new field's stationary. This corresponds to a F81-on-DP chain
    in which self-jumps are counted as resampling events. The
    "no-self-jump" CTMC interpretation (Interp 2) gives a
    state-dependent no-jump probability exp(-rho_chain * (1 - rho_theta) * t/2),
    which differs from Interp 1's uniform exp(-rho_chain * t / 2). Both
    have the same field marginal P[theta -> theta'; t] but different
    (residue, field) joint dynamics. The Phase A derivation and Phase C
    implementation use Interp 1; the paper's prose is consistent with
    either. See dynfield_math.md "Interpretation notes".
  pi_class: (K_c, A) shared per-class marginal stationary,
    equal to `(rho[None, :, None] * pi_field).sum(axis=1)` at training
    convergence. Carried explicitly to match the `state.pi_class`
    convention used by the shared SVI scaffold.

Auxiliary fields (training scratch, not part of the trained model):

  per_cluster_field_assignment: dict mapping cluster_key to int in
    [0, L_max). Field atom index currently assigned to each cluster
    on the partition. Refreshed by the SVI's per-cluster Gibbs sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DynamicFieldState:
    """State container for the dynamic-latent-field coupling variant.

    See module docstring for field semantics.
    """

    K_c: int
    A: int
    pi_field: np.ndarray              # (K_c, L_max, A)
    rho: np.ndarray                   # (L_max,)
    tsb_betas: Optional[np.ndarray]   # (L_max - 1,) or None for direct rho
    alpha_field: float
    pi_class: np.ndarray              # (K_c, A) -- shared scaffold needs this

    # F81-on-DP rate multiplier; controls field-chain mixing speed.
    # rho_chain = 0 -> per-class GTR limit (no field jumps).
    rho_chain: float = 1.0

    # Hierarchical archetype vocabulary. `pi_field` is a DERIVED quantity:
    # pi_field[c, theta, :] = pi_archetype[arch_assignment[c, theta], :].
    # The archetype vocabulary is drawn from a truncated stick-breaking
    # (TSB) prior with concentration alpha_arch and base measure
    # Dirichlet(alpha_prior); assignments arch_assignment map each
    # (c, theta) to one of K_a archetypes. This restricts pi_field to a
    # discrete set of shared simplex points -- a site "class" is a
    # pattern of archetype selections across theta rather than an
    # unconstrained per-atom Dirichlet. See dynfield.archetypes.
    #
    # Semantics:
    #   pi_archetype: (K_a, A) -- learnt archetype distributions.
    #   arch_assignment: (K_c, L_max) int32 -- per-(c, theta) archetype index.
    #   rho_arch: (K_a,) -- TSB weights on archetypes (analogous to rho).
    #   tsb_betas_arch: (K_a - 1,) -- Beta(1, alpha_arch) latent stick
    #     draws such that rho_arch = stick_breaking(tsb_betas_arch).
    #   alpha_arch: float -- TSB concentration on archetype vocabulary.
    pi_archetype: np.ndarray = None  # (K_a, A); set by construction
    arch_assignment: np.ndarray = None  # (K_c, L_max) int32
    rho_arch: np.ndarray = None  # (K_a,)
    tsb_betas_arch: Optional[np.ndarray] = None  # (K_a - 1,) or None
    alpha_arch: float = 1.0

    # Rate heterogeneity (Yang 1994 style Gamma+I; see
    # math-paper/appendix-tkfdp.tex par:arch-gamma-plus-I):
    #   K_rate_bins: int -- number of Gamma quantile bins (0 disables).
    #   alpha_gamma: float -- Gamma(alpha, alpha) shape; mean rate 1,
    #                variance 1/alpha across bins.
    #   p_inv: float -- proportion of clusters in the invariant
    #                (zero-rate) bin.
    # When K_rate_bins == 0 or all three are None, the base model with a
    # single rho_chain applies.
    K_rate_bins: Optional[int] = None
    alpha_gamma: Optional[float] = None
    p_inv: Optional[float] = None

    # Per-SITE Gamma+I rate heterogeneity on the substitution rate; see
    # math-paper/appendix-tkfdp.tex par:arch-gamma-plus-I-persite:
    #   K_rate_bins_site: int -- number of per-site Gamma bins.
    #   alpha_gamma_site: float -- Gamma(alpha,alpha) shape parameter.
    #   p_inv_site: float -- prior mixing weight on the invariant bin.
    # Sites' bin assignments {b_s} live in FamilyKState.site_rate_bin
    # (persistent per (family, column) with values in {0..K_rate_bins_site})
    # and are Gibbs-refreshed every `site_resample_every` outer iters
    # (default 5) via a Rao-Blackwellised conditional over the field
    # trajectory. Between resamples the bins are held fixed and the HR
    # pass applies m_{b_s} to xi^(k) per site via eigenvalue scaling.
    K_rate_bins_site: Optional[int] = None
    alpha_gamma_site: Optional[float] = None
    p_inv_site: Optional[float] = None

    # Training scratch:
    per_cluster_field_assignment: dict = field(default_factory=dict)

    @property
    def K_a(self) -> int:
        return int(self.pi_archetype.shape[0])

    def materialise_pi_field(self) -> None:
        """Rewrite pi_field to pi_archetype[arch_assignment, :]. Call after
        any change to pi_archetype or arch_assignment."""
        self.pi_field = self.pi_archetype[self.arch_assignment]

    @property
    def L_max(self) -> int:
        return int(self.pi_field.shape[1])

    def __post_init__(self):
        K_c = int(self.K_c)
        A = int(self.A)
        L_max = int(self.pi_field.shape[1])
        assert self.pi_field.shape == (K_c, L_max, A), (
            f"pi_field shape {self.pi_field.shape} != ({K_c}, {L_max}, {A})")
        assert self.rho.shape == (L_max,), (
            f"rho shape {self.rho.shape} != ({L_max},)")
        if self.tsb_betas is not None:
            assert self.tsb_betas.shape == (L_max - 1,), (
                f"tsb_betas shape {self.tsb_betas.shape} != ({L_max - 1},)")
        assert self.pi_class.shape == (K_c, A), (
            f"pi_class shape {self.pi_class.shape} != ({K_c}, {A})")
