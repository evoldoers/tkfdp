"""Collapsed discovery sampler.

The site class is a per-column nuisance; the precompute (Stage B/C) makes its
marginal tractable, so we Rao-Blackwellize it -- marginalize the class
analytically and sample ONLY the partition (which columns pair). The cn-move is
gone. Every pairing decision reads the class-marginalized evidence from the
PairingCache; the DM mixture enters as a reweight of that cached evidence and is
fit from class POSTERIORS, not sampled labels.

Partition prior: size-{1,2} Ewens, whose per-pair cost -log(alpha_z) is already
folded into PairingCache.logodds. Only shortlist columns (those appearing in some
cached pair) can pair; every other column is a permanent singleton.

z-move (Neal-3 CRP, cap-2): for column s, detach it from its current partner,
then sample its new state from {singleton} u {pair with a currently-singleton
shortlist neighbour u}, with log-weights {0} u {logodds(s,u)} -- the cached
class-marginalized (optionally DM-reweighted) pairing log-odds.
"""
from __future__ import annotations

import numpy as np

from .pairing_cache import PairingCache, _lse


class CollapsedState:
    def __init__(self, cache_dir, fams, alpha_z=100.0):
        self.pc = PairingCache(cache_dir)
        self.alpha_z = float(alpha_z)
        self.fams = [f for f in fams if self.pc.has(f)]
        self.neigh: dict = {}                       # (fam,col) -> [neighbour cols]
        self.partner: dict = {}                     # (fam,col) -> partner col | None
        self.cols: dict = {}                        # fam -> [shortlist cols]
        for fam in self.fams:
            rec = self.pc._family(fam)
            if rec is None:
                continue
            d, idx, _ = rec
            cs = set()
            for (a, b) in idx:
                self.neigh.setdefault((fam, a), []).append(b)
                self.neigh.setdefault((fam, b), []).append(a)
                cs.add(a); cs.add(b)
            self.cols[fam] = sorted(cs)
            for c in cs:
                self.partner[(fam, c)] = None

    def pairs(self):
        """Current discovered pairs as {fam: set((i,j))}."""
        out = {}
        for (fam, c), t in self.partner.items():
            if t is not None and c < t:
                out.setdefault(fam, set()).add((c, t))
        return out


def _mix_logodds(pc, fam, i, j, alpha_z, dm=None):
    """Pairing log-odds, DM mixture marginalized over components (or flat if
    dm is None). Returns None on a cache miss."""
    if dm is None:
        return pc.logodds(fam, i, j, alpha_z)
    # logsumexp_h [ log pi_h + (pair_marg_h - sing_i_h - sing_j_h) ] - log alpha_z,
    # via the identity logodds_h = pair_marg_h - sing_i_h - sing_j_h - log alpha_z
    lo_h = []
    for h in range(dm.H):
        lo = pc.logodds(fam, i, j, alpha_z,
                        dm_logp=pc.dm_logprior_grid(fam, i, j, dm, h),
                        dm_single=pc.dm_single_logp(dm, h))
        if lo is None:
            return None
        lo_h.append(np.log(dm.pi[h] + 1e-300) + lo)
    return _lse(np.array(lo_h))


def z_sweep(state: CollapsedState, rng, dm=None):
    """One collapsed CRP sweep over all shortlist columns.

    The pairing log-odds of merging (i,j) is partition-INDEPENDENT under the
    cap-2 collapsed model (each candidate pair's cached evidence + fixed-DM prior
    is self-contained), so it is memoized on `state` across sweeps -- essential
    when dm has H>1 components (H logodds evaluations per candidate). The memo is
    keyed by (fam,i,j) and assumes a FIXED dm for the run's lifetime; reset
    state._lo_memo if dm changes."""
    memo = state.__dict__.setdefault("_lo_memo", {})
    n_pair_moves = 0
    for fam in state.fams:
        cols = state.cols.get(fam, [])
        for s in rng.permutation(cols):
            s = int(s)
            t_old = state.partner[(fam, s)]
            if t_old is not None:                    # detach s
                state.partner[(fam, s)] = None
                state.partner[(fam, t_old)] = None
            cands, logw = [None], [0.0]
            for u in state.neigh.get((fam, s), []):
                if u != s and state.partner[(fam, u)] is None:
                    mk = (fam, min(s, u), max(s, u))
                    if mk in memo:
                        lo = memo[mk]
                    else:
                        lo = _mix_logodds(state.pc, fam, s, u, state.alpha_z, dm)
                        memo[mk] = lo
                    if lo is not None:
                        cands.append(u); logw.append(lo)
            lw = np.array(logw); lw -= lw.max(); p = np.exp(lw); p /= p.sum()
            choice = cands[int(rng.choice(len(cands), p=p))]
            if choice is not None:
                state.partner[(fam, s)] = choice
                state.partner[(fam, choice)] = s
                n_pair_moves += 1
    return n_pair_moves
