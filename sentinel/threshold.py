"""Streaming extreme-value calibration (SPOT-style, Siffer et al., KDD 2017).

Scores above an initial high quantile `t` are "peaks". Their excesses are
modelled with a Generalized Pareto Distribution, giving a tail probability

    P(S > s) ~= (N_t / n) * (1 + gamma * (s - t) / sigma) ** (-1 / gamma)

The GPD is refit online by method of moments from running sums, so every
update is O(1). Scores that triggered an alarm are winsorized to the alarm
level before being added: they still count as exceedances (so a pattern that
keeps recurring gradually becomes "normal") but one huge anomaly cannot blow
up the fitted scale and mask the next one.

Shape shrinkage (`shrink` = n0 > 0): with only a dozen peaks after warm-up the
moment estimate of the shape gamma is very noisy, and a spuriously heavy tail
(gamma ~ 0.4) pushes the alarm level out to z ~ 200, so obvious anomalies go
unflagged. The shape is therefore shrunk toward the exponential tail (gamma = 0)
with weight nt / (nt + n0): a heavy tail is only believed once enough peaks
have been seen. The scale is then set so the mean excess is preserved.
"""
from __future__ import annotations

import math

import numpy as np


class TailModel:
    __slots__ = ("t", "n", "nt", "s1", "s2", "gamma", "sigma", "shrink")

    def __init__(self, scores, init_q: float = 0.98, shrink: float = 0.0):
        self.shrink = shrink
        scores = np.asarray(scores, dtype=float)
        self.t = float(np.quantile(scores, init_q))
        exc = scores[scores > self.t] - self.t
        self.n = len(scores)
        self.nt = len(exc)
        self.s1 = float(exc.sum())
        self.s2 = float((exc ** 2).sum())
        self._fit()

    def _fit(self) -> None:
        if self.nt < 3:
            self.gamma, self.sigma = 0.0, max(self.s1 / max(self.nt, 1), 1e-9)
            return
        mean = self.s1 / self.nt
        var = max(self.s2 / self.nt - mean * mean, 1e-12)
        r = mean * mean / var
        # gamma < 0 implies a hard upper bound: any score past it gets p=0 and
        # a "certain" alarm. Fall back to the exponential tail instead.
        self.gamma = min(max(0.5 * (1.0 - r), 0.0), 0.8)
        self.sigma = max(0.5 * mean * (r + 1.0), 1e-9)
        if self.shrink > 0 and self.gamma > 0:     # only heavy-tail estimates are shrunk
            self.gamma *= self.nt / (self.nt + self.shrink)
            self.sigma = max(mean * (1.0 - self.gamma), 1e-9)   # GPD mean excess = sigma / (1 - gamma)

    def tail_prob(self, s: float) -> float:
        base = (self.nt + 1) / (self.n + 1)     # pseudo-count: never exactly 0
        if s <= self.t:
            if self.t <= 0:
                return 1.0
            # linear ramp between "typical" and the peaks threshold, for display
            frac = s / self.t if self.t > 0 else 1.0
            return min(1.0, base + (1.0 - base) * (1.0 - max(frac, 0.0)))
        y = (s - self.t) / self.sigma
        if abs(self.gamma) < 1e-6:
            return base * math.exp(-y)
        inner = 1.0 + self.gamma * y
        if inner <= 0:
            return 1e-300
        return base * inner ** (-1.0 / self.gamma)

    def level_for(self, p: float) -> float:
        """Inverse of tail_prob above t: the score whose tail probability is p."""
        base = (self.nt + 1) / (self.n + 1)
        if p >= base:
            return self.t
        r = p / base
        if abs(self.gamma) < 1e-6:
            return self.t - self.sigma * math.log(r)
        return self.t + self.sigma / self.gamma * (r ** (-self.gamma) - 1.0)

    def update(self, s: float, cap: float | None = None) -> None:
        self.n += 1
        if s > self.t:
            if cap is not None and s > cap:
                s = cap
            e = s - self.t
            self.nt += 1
            self.s1 += e
            self.s2 += e * e
            self._fit()
