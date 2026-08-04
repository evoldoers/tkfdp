"""Coupling-model abstraction layer.

Both the Potts variant (cap-2 Sinkhorn-corrected joint, current default)
and the dynamic latent-field variant (F81-on-DP field selector, in
development) implement the `CouplingModel` protocol below. Code that
needs variant-specific math (cluster emission, M-tensor for the IPHMM,
atom/field updates, indel-seam joint stationary) calls through the
protocol; everything else (TKF92 indels, partition Gibbs, EM warmup,
SVI scaffold, IPHMM kernel, checkpoint frame) is variant-agnostic.

Design and math derivations live in:
  - docs/dynfield_design.md   (architecture + 9-phase implementation plan)
  - docs/dynfield_math.md     (per-(c, theta) emission, hierarchical
                                Felsenstein recursion, no-Sinkhorn proof)

This package is a strictly-additive refactor. The legacy Potts call
sites in svi.py / block_likelihoods.py / mcmc_infinite_phmm.py keep
working byte-identical; the new path
(`state.coupling.build_M_tensor(...)` etc.) dispatches to the same
underlying math via the PottsCouplingModel adapter. Migration to the
new path is incremental.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class CouplingModel(Protocol):
    """The coupling-specific surface area.

    Implementations:
      - PottsCouplingModel (coupling.potts) -- cap-2 Sinkhorn-corrected
        Potts joint with per-class pi and shared Potts atom H.
      - DynamicFieldCouplingModel (coupling.dynfield) -- F81-on-DP field
        selector with per-(class, field) stationary. Not yet implemented;
        see docs/dynfield_design.md Phase C.
    """

    variant: str   # 'potts' | 'dynamic_field'
    K_c: int
    A: int

    # ---- emission / boost builders -----------------------------------------

    def build_singlet_emission(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """K_c-marginalized singleton joint emission tensor.

        Returns (P_singlet, pi_out_eff, sub_matrix_eff); shapes per
        block_likelihoods.build_singlet_emission. Variant-shared in
        practice (per-class GTR singlets ignore the coupling object), but
        kept on the protocol for uniformity."""

    def build_doublet_emission(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            pair_background: str,
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            reversible: bool = True,
            ) -> np.ndarray:
        """(A, A, A, A) doublet emission tensor for a coupled column pair.
        Potts variant: Sinkhorn-corrected pi_joint joint generator expm.
        Dynamic-field variant: L^3 sum over (parent, leaf-X, leaf-Y)
        field states."""

    def build_M_tensor(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            pair_background: str,
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            reversible: bool = True,
            ) -> np.ndarray:
        """(A, A, A, A) edge-boost tensor M = P_doublet / (P_singlet*P_singlet)."""

    def build_M_tensor_typed(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            pair_background: str,
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            reversible: bool = True,
            ) -> dict:
        """Dict {'MM','MI','MD','II','DD','ID'} of M-boost tensors keyed
        by edge-endpoint-type pair, for allow_id_edges=True in the IPHMM."""

    # ---- serialization (Phase I) ------------------------------------------

    def to_npz(self) -> dict:
        """Variant-specific keys to serialize into the checkpoint state.npz."""

    @classmethod
    def from_npz(cls, arrs: dict, meta: dict) -> 'CouplingModel':
        """Reconstruct from a checkpoint."""


# Variant registry. Populated by the variant modules at import time.
VARIANTS: dict = {}


def register(variant_name: str):
    """Decorator: register a CouplingModel implementation."""
    def deco(cls):
        VARIANTS[variant_name] = cls
        return cls
    return deco


def get(variant_name: str):
    """Look up a registered variant class. Raises KeyError if unknown."""
    if variant_name not in VARIANTS:
        # Lazy import the standard implementations so callers don't have to.
        from . import potts as _potts        # noqa: F401
        from . import dynfield as _dynfield  # noqa: F401
    if variant_name not in VARIANTS:
        raise KeyError(
            f"unknown coupling variant {variant_name!r}; "
            f"registered: {sorted(VARIANTS)}")
    return VARIANTS[variant_name]
