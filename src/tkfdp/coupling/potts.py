"""Potts variant of the CouplingModel.

Adapter around the existing block_likelihoods machinery. Holds
references to the SVI state's pi_class and potts_dp; methods delegate
to the existing `block_likelihoods.build_*` functions, which keep
working byte-identical on the legacy call path (svi.py,
mcmc_infinite_phmm.py.precompute_partial_forward, composite_partition,
val_loglik_v2). This is the Phase B "no behaviour change" adapter; the
math is unchanged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .. import block_likelihoods as _bl
from ..potts_dp import PottsDPState
from . import register


@register('potts')
class PottsCouplingModel:
    """Cap-2 Sinkhorn-corrected Potts coupling.

    Parameters:
      K_c:        number of site classes.
      A:          alphabet size (20 for amino acids).
      pi_class:   (K_c, A) per-class stationary.
      potts_dp:   PottsDPState (atoms, assignments, counts, rho, tsb_betas).

    The class is a thin shim around the existing block_likelihoods
    functions; all heavy lifting (Sinkhorn iteration, expm, etc.) is
    delegated there. A1 (reversible=True) is the default everywhere.
    """

    variant = 'potts'

    def __init__(self, K_c: int, A: int,
                 pi_class: np.ndarray, potts_dp: PottsDPState):
        self.K_c = int(K_c)
        self.A = int(A)
        self.pi_class = pi_class
        self.potts_dp = potts_dp

    @classmethod
    def from_svi_state(cls, state) -> 'PottsCouplingModel':
        """Construct from an SVIState (or any object exposing K_c, A,
        pi_class, potts_dp). Used by the SVI loop / val LL / IPHMM
        precompute pipelines."""
        return cls(K_c=state.K_c, A=state.A,
                   pi_class=np.asarray(state.pi_class),
                   potts_dp=state.potts_dp)

    # _StateShim adapter: the existing block_likelihoods functions
    # expect a `state` with .K_c, .A, .pi_class, .potts_dp. We satisfy
    # that directly; no shim needed because PottsCouplingModel itself has
    # those attributes.

    def build_singlet_emission(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            ):
        return _bl.build_singlet_emission(
            self, t, eta=eta, pi_c=pi_c, S=S,
            n_rate_bins=n_rate_bins, a_eta=a_eta, b_eta=b_eta)

    def build_doublet_emission(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            pair_background: str,
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            reversible: bool = True,
            ) -> np.ndarray:
        return _bl.build_doublet_emission(
            self, t, eta=eta, pi_c=pi_c, S=S,
            pair_background=pair_background,
            n_rate_bins=n_rate_bins, a_eta=a_eta, b_eta=b_eta,
            reversible=reversible)

    def build_M_tensor(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            pair_background: str,
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            reversible: bool = True,
            ) -> np.ndarray:
        return _bl.build_M_tensor(
            self, t, eta=eta, pi_c=pi_c, S=S,
            pair_background=pair_background,
            n_rate_bins=n_rate_bins, a_eta=a_eta, b_eta=b_eta,
            reversible=reversible)

    def build_M_tensor_typed(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            pair_background: str,
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            reversible: bool = True,
            ) -> dict:
        return _bl.build_M_tensor_typed(
            self, t, eta=eta, pi_c=pi_c, S=S,
            pair_background=pair_background,
            n_rate_bins=n_rate_bins, a_eta=a_eta, b_eta=b_eta,
            reversible=reversible)

    # ---- Serialization ----------------------------------------------------

    def to_npz(self) -> dict:
        """Variant-specific arrays under the standard Potts keys. The
        SVI loop's checkpoint writer merges these into the per-MSA state
        npz."""
        out = {
            'pi_class':            np.asarray(self.pi_class).astype(np.float64),
            'potts_atoms':         np.asarray(self.potts_dp.atoms).astype(np.float64),
            'potts_assignments':   np.asarray(self.potts_dp.assignments).astype(np.int64),
            'potts_counts':        np.asarray(self.potts_dp.counts).astype(np.int64),
        }
        if self.potts_dp.rho is not None:
            out['rho'] = np.asarray(self.potts_dp.rho).astype(np.float64)
        if self.potts_dp.tsb_betas is not None:
            out['tsb_betas'] = np.asarray(self.potts_dp.tsb_betas).astype(np.float64)
        return out

    @classmethod
    def from_npz(cls, arrs, meta) -> 'PottsCouplingModel':
        """Adapter around checkpoint.load_globals_from_checkpoint. Used
        when loading a checkpoint without instantiating an SVIState (e.g.
        for inference-only paths). For training resume use
        checkpoint.load_checkpoint(...).coupling instead."""
        K_c = int(meta['K_c'])
        pi_class = np.asarray(arrs['pi_class'])
        A = pi_class.shape[1]
        atoms = np.asarray(arrs['potts_atoms'])
        rho = (np.asarray(arrs['rho'])
               if hasattr(arrs, 'files') and 'rho' in arrs.files
               or (isinstance(arrs, dict) and 'rho' in arrs)
               else None)
        tsb_betas = (np.asarray(arrs['tsb_betas'])
                     if hasattr(arrs, 'files') and 'tsb_betas' in arrs.files
                     or (isinstance(arrs, dict) and 'tsb_betas' in arrs)
                     else None)
        if rho is None:
            K_H_max = int(atoms.shape[0])
            rho = np.full(K_H_max, 1.0 / K_H_max)
        if tsb_betas is None and atoms.shape[0] > 1:
            tsb_betas = np.full(atoms.shape[0] - 1, 1.0 / atoms.shape[0])
        potts_dp = PottsDPState(
            K_c=K_c, A=A,
            atoms=atoms,
            assignments=np.asarray(arrs['potts_assignments']),
            counts=np.asarray(arrs['potts_counts']),
            alpha_H=float(meta.get('alpha_H', 1.0)),
            mu_prior=np.zeros(A * (A + 1) // 2),
            tau_prior=np.ones(A * (A + 1) // 2),
            rho=rho, tsb_betas=tsb_betas,
        )
        return cls(K_c=K_c, A=A, pi_class=pi_class, potts_dp=potts_dp)
