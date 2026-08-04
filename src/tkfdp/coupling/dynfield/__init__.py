"""DynamicFieldCouplingModel: implementation of the CouplingModel protocol
for the dynamic-latent-field variant.

The adapter is thin: it delegates the per-(c, theta) generator construction
and the 4-case cluster joint emission to `coupling.dynfield.emission`, and
keeps its persistent state (pi_field, rho, tsb_betas, alpha_field, pi_class,
per-cluster field assignments) in a `DynamicFieldState` dataclass.

Unlike the Potts variant, the dynamic-field variant does NOT need a Sinkhorn
correction at the indel seam: under instant re-equilibration the cluster-of-2
joint stationary `J^(c1, c2)(a, b) = sum_theta rho[theta] *
pi_field[c1, theta, a] * pi_field[c2, theta, b]` is marginal-consistent by
construction (its row marginal equals `pi_lone^(c1)` by construction). So
`reversible=True` is the no-op natural case (the protocol's default); the
`reversible` flag is accepted on the API for protocol compatibility but does
not change the math here.

The `pair_background` argument from the protocol is likewise vestigial: under
dynamic-field there is no separate per-pair background -- `pi_field` IS the
background. Accepted for compatibility; passing `'per_class'` is the natural
interpretation, and `'lg08'` raises since LG08 is shared per-class background
not the per-pair joint dynfield uses.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .. import register
from . import emission as _em
from .state import DynamicFieldState


@register('dynamic_field')
class DynamicFieldCouplingModel:
    """Dynamic-latent-field cap-2 cluster joint emission.

    Parameters:
      K_c:     number of site classes.
      A:       alphabet size (20 for amino acids).
      pi_class: (K_c, A) shared per-class marginal stationary. Should
        equal `(rho[None, :, None] * dyn_field.pi_field).sum(axis=1)` at
        training convergence. Carried for compatibility with the SVI
        scaffold that expects `state.pi_class`.
      dyn_field: DynamicFieldState carrying pi_field, rho, etc.

    See `docs/dynfield_math.md` (Phase A) for the math and
    `tests/dynfield/test_math_precompute.py` for verification against a
    faithful trajectory MC.
    """

    variant = 'dynamic_field'

    def __init__(self, K_c: int, A: int,
                 pi_class: np.ndarray, dyn_field: DynamicFieldState):
        self.K_c = int(K_c)
        self.A = int(A)
        self.pi_class = np.asarray(pi_class, dtype=np.float64)
        self.dyn_field = dyn_field

    @classmethod
    def from_svi_state(cls, state) -> 'DynamicFieldCouplingModel':
        """Construct from an SVIState that has been extended to carry
        `state.dyn_field: DynamicFieldState` (the dynfield analogue of
        `state.potts_dp`)."""
        return cls(K_c=state.K_c, A=state.A,
                   pi_class=np.asarray(state.pi_class),
                   dyn_field=state.dyn_field)

    # ---- emission / boost builders -----------------------------------------

    def build_singlet_emission(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if pi_c is None:
            pi_c = np.full(self.K_c, 1.0 / self.K_c)
        if n_rate_bins > 1:
            raise NotImplementedError(
                "dynamic-field variant: n_rate_bins > 1 not yet wired "
                "(needs Yang discrete-gamma marginalisation on top of the "
                "4-case sum; planned)")
        return _em.class_marginal_singlet(
            t, pi_c=np.asarray(pi_c), rho=self.dyn_field.rho,
            pi_field=self.dyn_field.pi_field, S=S, eta=eta,
            rho_chain=float(self.dyn_field.rho_chain))

    def build_doublet_emission(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            pair_background: str = 'per_class',
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            reversible: bool = True,
            ) -> np.ndarray:
        if pi_c is None:
            pi_c = np.full(self.K_c, 1.0 / self.K_c)
        if n_rate_bins > 1:
            raise NotImplementedError(
                "dynamic-field variant: n_rate_bins > 1 not yet wired")
        # pair_background is vestigial under dynamic-field; pi_field IS
        # the background, so any value passed is treated as the natural
        # 'per_class' interpretation. We accept the Potts default 'lg08'
        # as a no-op so generic call sites (e.g. the IPHMM precompute,
        # which defaults to 'lg08' for backward compat with released
        # Potts checkpoints) work uniformly across variants.
        # reversible is also vestigial: J^(c1, c2) is marginal-consistent
        # by construction; no Sinkhorn step exists to disable.
        return _em.class_marginal_doublet(
            t, pi_c=np.asarray(pi_c), rho=self.dyn_field.rho,
            pi_field=self.dyn_field.pi_field, S=S, eta=eta,
            rho_chain=float(self.dyn_field.rho_chain))

    def build_M_tensor(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            pair_background: str = 'per_class',
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            reversible: bool = True,
            ) -> np.ndarray:
        P_singlet, _, _ = self.build_singlet_emission(
            t, eta=eta, pi_c=pi_c, S=S, n_rate_bins=n_rate_bins,
            a_eta=a_eta, b_eta=b_eta)
        P_doublet = self.build_doublet_emission(
            t, eta=eta, pi_c=pi_c, S=S,
            pair_background=pair_background,
            n_rate_bins=n_rate_bins, a_eta=a_eta, b_eta=b_eta,
            reversible=reversible)
        denom = P_singlet[:, :, None, None] * P_singlet[None, None, :, :]
        return P_doublet / np.clip(denom, 1e-300, None)

    def build_M_tensor_typed(
            self, t: float, *, eta: float = 1.0,
            pi_c: Optional[np.ndarray] = None,
            S: Optional[np.ndarray] = None,
            pair_background: str = 'per_class',
            n_rate_bins: int = 1,
            a_eta: float = 2.0, b_eta: float = 2.0,
            reversible: bool = True,
            ) -> dict:
        """Edge-type-aware M tensors. Same axis-sum recipe as the Potts
        variant; only the underlying P_singlet / P_doublet differ. See
        block_likelihoods.build_M_tensor_typed for the index conventions."""
        P_singlet, pi_eff, _ = self.build_singlet_emission(
            t, eta=eta, pi_c=pi_c, S=S, n_rate_bins=n_rate_bins,
            a_eta=a_eta, b_eta=b_eta)
        P_doublet = self.build_doublet_emission(
            t, eta=eta, pi_c=pi_c, S=S,
            pair_background=pair_background,
            n_rate_bins=n_rate_bins, a_eta=a_eta, b_eta=b_eta,
            reversible=reversible)
        denom_MM = P_singlet[:, :, None, None] * P_singlet[None, None, :, :]
        M_MM = P_doublet / np.clip(denom_MM, 1e-300, None)
        P_MI = P_doublet.sum(axis=2)
        denom_MI = P_singlet[:, :, None] * pi_eff[None, None, :]
        M_MI = P_MI / np.clip(denom_MI, 1e-300, None)
        P_MD = P_doublet.sum(axis=3)
        denom_MD = P_singlet[:, :, None] * pi_eff[None, None, :]
        M_MD = P_MD / np.clip(denom_MD, 1e-300, None)
        P_II = P_doublet.sum(axis=(0, 2))
        denom_II = pi_eff[:, None] * pi_eff[None, :]
        M_II = P_II / np.clip(denom_II, 1e-300, None)
        P_DD = P_doublet.sum(axis=(1, 3))
        denom_DD = pi_eff[:, None] * pi_eff[None, :]
        M_DD = P_DD / np.clip(denom_DD, 1e-300, None)
        P_ID = P_doublet.sum(axis=(0, 3))
        denom_ID = pi_eff[:, None] * pi_eff[None, :]
        M_ID = P_ID / np.clip(denom_ID, 1e-300, None)
        return {'MM': M_MM, 'MI': M_MI, 'MD': M_MD,
                'II': M_II, 'DD': M_DD, 'ID': M_ID}

    # ---- Serialization ----------------------------------------------------

    def to_npz(self) -> dict:
        """Variant-specific arrays under the standard dynfield keys. The
        SVI loop's checkpoint writer merges these into the per-MSA state
        npz; `meta.json` should carry `coupling_variant='dynamic_field'`
        and `rho_chain` (the F81-on-DP rate multiplier)."""
        out = {
            'pi_class':       np.asarray(self.pi_class).astype(np.float64),
            'pi_field':       np.asarray(self.dyn_field.pi_field).astype(np.float64),
            'rho':            np.asarray(self.dyn_field.rho).astype(np.float64),
        }
        if self.dyn_field.tsb_betas is not None:
            out['tsb_betas'] = np.asarray(
                self.dyn_field.tsb_betas).astype(np.float64)
        # Hierarchical archetype state (when enabled).
        if self.dyn_field.pi_archetype is not None:
            out['pi_archetype'] = np.asarray(
                self.dyn_field.pi_archetype).astype(np.float64)
            out['arch_assignment'] = np.asarray(
                self.dyn_field.arch_assignment).astype(np.int32)
            out['rho_arch'] = np.asarray(
                self.dyn_field.rho_arch).astype(np.float64)
            if self.dyn_field.tsb_betas_arch is not None:
                out['tsb_betas_arch'] = np.asarray(
                    self.dyn_field.tsb_betas_arch).astype(np.float64)
        return out

    @classmethod
    def from_npz(cls, arrs, meta) -> 'DynamicFieldCouplingModel':
        K_c = int(meta['K_c'])
        pi_class = np.asarray(arrs['pi_class'])
        A = pi_class.shape[1]
        pi_field = np.asarray(arrs['pi_field'])
        rho = np.asarray(arrs['rho'])
        tsb_betas = (np.asarray(arrs['tsb_betas'])
                     if (hasattr(arrs, 'files') and 'tsb_betas' in arrs.files)
                     or (isinstance(arrs, dict) and 'tsb_betas' in arrs)
                     else None)
        dyn = DynamicFieldState(
            K_c=K_c, A=A,
            pi_field=pi_field,
            rho=rho,
            tsb_betas=tsb_betas,
            alpha_field=float(meta.get('alpha_field', 1.0)),
            pi_class=pi_class,
            rho_chain=float(meta.get('rho_chain', 1.0)),
        )
        # Restore hierarchical archetype state if present in the npz.
        def _has(k):
            return ((hasattr(arrs, 'files') and k in arrs.files)
                     or (isinstance(arrs, dict) and k in arrs))
        if _has('pi_archetype') and _has('arch_assignment'):
            dyn.pi_archetype = np.asarray(arrs['pi_archetype'])
            dyn.arch_assignment = np.asarray(arrs['arch_assignment'])
            dyn.rho_arch = (np.asarray(arrs['rho_arch'])
                             if _has('rho_arch') else None)
            dyn.tsb_betas_arch = (np.asarray(arrs['tsb_betas_arch'])
                                    if _has('tsb_betas_arch') else None)
            dyn.alpha_arch = float(meta.get('alpha_arch', 1.0))
        return cls(K_c=K_c, A=A, pi_class=pi_class, dyn_field=dyn)
