"""Stage C of the precompute pairing pipeline
(analysis/precompute_pairing_pipeline.md): the cache-consumption layer the
z-move reads instead of scoring pairs live.

The precompute (experiments/precompute_pairing.py) writes, per family, a
`<fam>.pairev.npz` holding for each shortlisted pair (i,j):
  top_i,top_j (topN,)   the top-N class indices per column (by singleton LL)
  LL_pair (topN,topN)   the UNSUMMED, rate-marginalized pair log-lik over that
                        class grid  (moment-matching forward -- the sampler's
                        own scorer; see precompute_pairing for the mm-vs-exact
                        rationale)
  logZ_s_i, logZ_s_j    the full-K_c singleton marginals (rate-marginalized)

The pairing log-odds for merging columns i,j into one cluster is
  logsumexp_{a,b}( LL_pair[a,b] + dm_logp[a,b] ) - logZ_s_i - logZ_s_j - log alpha_z
where dm_logp is an optional (topN,topN) DM log-prior over the class grid,
recomputed cheaply at lookup time so the DM mixture stays live. Because
LL_pair is stored UNSUMMED, a DM update never touches the cache.

DESIGN NOTE (fixed-class vs class-marginalized). The cached evidence marginalizes
the per-column class over the top-N grid; the *current* z-move instead scores at
each column's fixed current class. Wiring this cache into the z-move therefore
turns the pairing decision into a class-MARGINALIZED one (which is also what the
DM reweight wants) -- a cleaner joint move, but a deliberate change from the
fixed-class z-move. Pairs absent from the shortlist (miz below z*) are cache
misses: the z-move treats them as non-candidates (won't-pair).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _lse(v):
    v = np.asarray(v, float); m = v.max()
    return float(np.log(np.sum(np.exp(v - m))) + m) if np.isfinite(m) else -np.inf


class PairingCache:
    """Lazy per-family loader for the pairing-evidence cache. Thread the same
    instance through a run; families load on first touch and stay resident."""

    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self._fam: dict = {}                     # fam -> (npz-dict, {(i,j):row}) | None

    def _family(self, fam):
        if fam not in self._fam:
            p = self.dir / f"{fam}.pairev.npz"
            if not p.exists():
                self._fam[fam] = None
            else:
                d = np.load(p)
                dd = {k: d[k] for k in d.files}
                ij = dd["ij"].reshape(-1, 2)
                idx = {(int(a), int(b)): k for k, (a, b) in enumerate(ij)}
                scol = {int(c): r for r, c in enumerate(dd["sing_cols"])} \
                    if "sing_cols" in dd and dd["sing_cols"].size else {}
                self._fam[fam] = (dd, idx, scol)
        return self._fam[fam]

    def has(self, fam) -> bool:
        """Is a cache present for this family (even if empty)?"""
        return self._family(fam) is not None

    def get(self, fam, i, j):
        """Cached record for pair (i,j) as a dict, or None on a miss (family not
        cached, or pair not shortlisted). Order-insensitive in (i,j)."""
        rec = self._family(fam)
        if rec is None:
            return None
        d, idx, _ = rec
        key = (int(min(i, j)), int(max(i, j)))
        k = idx.get(key)
        if k is None:
            return None
        return {"top_i": d["top_i"][k], "top_j": d["top_j"][k],
                "LL_pair": d["LL_pair"][k],
                "logZ_s_i": float(d["logZ_s_i"][k]),
                "logZ_s_j": float(d["logZ_s_j"][k])}

    def sing_ll(self, fam, col):
        """(K_c,) per-column singleton LL over all classes, or None if the family
        is uncached or predates the enriched (sing_ll) format."""
        rec = self._family(fam)
        if rec is None:
            return None
        d, _, scol = rec
        r = scol.get(int(col))
        return None if r is None else d["sing_ll"][r]

    def singleton_logmarg(self, fam, col, dm_single=None):
        """Class-marginalized singleton log-evidence for `col`:
        logsumexp(sing_ll[col] + dm_single). dm_single: optional (K_c,) DM
        single-class log-prior (flat if None). None on a miss."""
        s = self.sing_ll(fam, col)
        if s is None:
            return None
        return _lse(s if dm_single is None else s + np.asarray(dm_single))

    def pair_logmarg(self, fam, i, j, dm_logp=None):
        """Class-marginalized pair log-evidence: logsumexp(LL_pair [+ dm_logp])."""
        r = self.get(fam, i, j)
        if r is None:
            return None
        grid = r["LL_pair"] if dm_logp is None else r["LL_pair"] + np.asarray(dm_logp)
        return _lse(np.asarray(grid).ravel())

    def logodds(self, fam, i, j, alpha_z, dm_logp=None, dm_single=None):
        """Class-marginalized pairing log-odds for merging (i,j):
        pair_logmarg - singleton_logmarg(i) - singleton_logmarg(j) - log alpha_z.
        Uses the enriched per-column singleton vectors when present (DM-exact),
        else the flat stored logZ_s. dm_logp: (topN,topN) grid prior; dm_single:
        (K_c,) single-class prior. None on a cache miss."""
        pm = self.pair_logmarg(fam, i, j, dm_logp=dm_logp)
        if pm is None:
            return None
        si = self.singleton_logmarg(fam, i, dm_single=dm_single)
        sj = self.singleton_logmarg(fam, j, dm_single=dm_single)
        if si is None or sj is None:                         # old-format fallback
            r = self.get(fam, i, j)
            si = r["logZ_s_i"] if si is None else si
            sj = r["logZ_s_j"] if sj is None else sj
        return pm - si - sj - float(np.log(alpha_z))

    def dm_logprior_grid(self, fam, i, j, dm, component):
        """(topN,topN) DM log-prior log P_DM(top_i[a], top_j[b] | component) over
        the cached class grid, from a DMPrior's per-component alpha. Lets the
        z-move reweight the cached evidence by the live DM mixture without
        recomputing any likelihood. Returns None on a cache miss."""
        r = self.get(fam, i, j)
        if r is None:
            return None
        a = np.asarray(dm.alpha[component], float)           # (K_c,)
        A = a.sum()
        li = np.log(a[r["top_i"].astype(int)])               # (topN,)
        lj = np.log(a[r["top_j"].astype(int)])
        # DM predictive of the 2-class multiset {c_i,c_j}: proportional (up to the
        # column-independent gammaln(A)-gammaln(A+2)) to alpha_{c_i} * alpha_{c_j}
        # for distinct classes; the shared constant cancels in the logsumexp ratio.
        return li[:, None] + lj[None, :] - 2.0 * np.log(A)

    def dm_single_logp(self, dm, component):
        """(K_c,) DM single-class log-prior log(alpha_{h,c}/A_h) for one column."""
        a = np.asarray(dm.alpha[component], float)
        return np.log(a) - np.log(a.sum())
