"""Mixture-of-Dirichlet-multinomials prior on per-cluster class composition
(appendix sec:dm-mixture). A cluster's class multiset ~ sum_h pi_h DM(n_C|alpha_h);
exchangeable in the columns -> depends only on counts -> cap>2-safe, no pairwise
term. Provides the cn predictive log-prior (into cn's alpha_c_log slot) and the
h_C / alpha_h / pi updates. Learned components = coevolutionary cluster types.

Key-based so it serves both fixed clusters (supervised: key = cluster index) and
dynamic clusters (discovery: key = frozenset(columns)).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, digamma


class DMPrior:
    def __init__(self, K_c, H=20, alpha_pi=5.0, alpha0=1.0):
        self.K_c = int(K_c); self.H = int(H); self.alpha_pi = float(alpha_pi)
        self.alpha = np.full((self.H, self.K_c), float(alpha0))
        self.pi = np.full(self.H, 1.0 / self.H)
        self.h: dict = {}                                  # cluster key -> component

    def _counts(self, state, fi, cols):
        n = np.zeros(self.K_c, np.float64)
        for c in state.families[fi].classes[np.asarray(cols)]:
            n[int(c)] += 1.0
        return n

    def logdm(self, n, h):
        a = self.alpha[h]; A = a.sum(); m = n.sum(); nz = n > 0
        return float(gammaln(A) - gammaln(A + m)
                     + (gammaln(a[nz] + n[nz]) - gammaln(a[nz])).sum())

    def sample_h(self, state, fi, cols, key, rng):
        n = self._counts(state, fi, cols)
        logp = np.log(self.pi + 1e-300) + np.array(
            [self.logdm(n, h) for h in range(self.H)])
        logp -= logp.max(); p = np.exp(logp); p /= p.sum()
        self.h[key] = int(rng.choice(self.H, p=p))

    def cn_logprior(self, state, fi, cols, s, key):
        """(K_c,) DM predictive log-prior for cn: log(alpha_{h,k} + n^{-s}_k)."""
        n = self._counts(state, fi, cols)
        n[int(state.families[fi].classes[s])] -= 1.0          # exclude column s
        return np.log(self.alpha[self.h.get(key, 0)] + n)

    def update_alpha(self, clusters, iters=3, floor=1e-3):
        """Minka fixed-point per component. clusters: iterable of (key, fi, cols,
        state); pools by current h[key]."""
        comp = {h: [] for h in range(self.H)}
        for key, fi, cols, state in clusters:
            comp[self.h.get(key, 0)].append(self._counts(state, fi, cols))
        for h in range(self.H):
            if not comp[h]:
                continue
            N = np.stack(comp[h]); m = N.sum(1)
            for _ in range(iters):
                a = self.alpha[h]; A = a.sum()
                num = (digamma(N + a) - digamma(a)).sum(0)
                den = (digamma(m + A) - digamma(A)).sum()
                self.alpha[h] = np.maximum(a * num / max(den, 1e-12), floor)

    def update_pi(self, keys, rng):
        nh = np.bincount([self.h.get(k, 0) for k in keys],
                         minlength=self.H).astype(float)
        v = np.empty(self.H)
        for h in range(self.H - 1):
            v[h] = rng.beta(1.0 + nh[h], self.alpha_pi + nh[h + 1:].sum())
        v[-1] = 1.0
        pi = np.empty(self.H); rem = 1.0
        for h in range(self.H):
            pi[h] = v[h] * rem; rem *= (1.0 - v[h])
        self.pi = pi / pi.sum()

    def component_top_classes(self, k=4):
        nh = {}
        for key, h in self.h.items():
            nh[h] = nh.get(h, 0) + 1
        out = {}
        for h in sorted(nh, key=lambda h: -nh[h]):
            top = np.argsort(-self.alpha[h])[:k]
            out[h] = {"n_clusters": nh[h],
                      "top": [(int(c), round(float(self.alpha[h, c]), 3)) for c in top]}
        return out
