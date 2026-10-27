"""Streaming experts. Each one sees one value at a time with O(1) state.

Protocol:
    z = expert.score(x)      # signed surprise of x BEFORE learning from it
    expert.update(x)         # absorb x into the baseline
    expert.describe(x)       # dict of human-readable evidence for x
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _nearest_window(buf, n, m, lo, x):
    """Nearest past window to the query (buf[n-m+1:n] followed by x).

    Candidates start in [lo, n-2m+2) so they end >= m points before the query.
    Early abandoning: stop summing a candidate once it exceeds the best so far.
    """
    q0 = n - m + 1
    best, arg = np.inf, -1
    for s in range(lo, n - 2 * m + 2):
        acc = 0.0
        for k in range(m):
            qk = x if k == m - 1 else buf[q0 + k]
            d = buf[s + k] - qk
            acc += d * d
            if acc >= best:
                break
        if acc < best:
            best, arg = acc, s
    return best, arg


class RobustEW:
    """Exponentially weighted location/scale with Huber-style clipping.

    Deviations are clipped at `clip` scales before updating, so a single
    outlier cannot drag the baseline; a sustained shift is absorbed slowly.
    """

    __slots__ = ("alpha", "clip", "floor", "m", "s")

    def __init__(self, alpha: float, clip: float = 3.0, floor: float = 1e-9):
        self.alpha, self.clip, self.floor = alpha, clip, floor
        self.m = 0.0
        self.s = 0.0

    def init(self, m: float, s: float) -> None:
        self.m, self.s = m, s

    @property
    def scale(self) -> float:
        return self.s if self.s > self.floor else self.floor

    def z(self, x: float) -> float:
        return (x - self.m) / self.scale

    def update(self, x: float) -> None:
        lim = self.clip * self.scale
        d = x - self.m
        a = abs(d)
        self.m += self.alpha * (lim if d > lim else (-lim if d < -lim else d))
        # the scale may grow faster than the mean moves, so a collapsed scale
        # (e.g. MAD = 0 on mostly-constant data) can still recover
        self.s += self.alpha * ((a if a < 2 * lim else 2 * lim) - self.s)


def _median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _mad(v, med):
    """MAD, falling back to mean absolute deviation when most values are equal."""
    dev = [abs(a - med) for a in v]
    mad = _median(dev)
    return mad if mad > 0 else sum(dev) / len(dev)


class LevelExpert:
    """Point outliers: robust z-score of the value against its baseline."""

    name = "level"

    def __init__(self, alpha: float, floor: float):
        self.ew = RobustEW(alpha, floor=floor)

    def init(self, buf):
        med = _median(buf)
        self.ew.init(med, _mad(buf, med))

    def score(self, x):
        return self.ew.z(x)

    def update(self, x):
        self.ew.update(x)

    def describe(self, x):
        lo, hi = self.ew.m - 3 * self.ew.scale, self.ew.m + 3 * self.ew.scale
        return {"expected": self.ew.m, "band": (lo, hi)}


class DiffExpert:
    """Spikes / abrupt jumps: robust z-score of the step change x_t - x_{t-1}."""

    name = "jump"

    def __init__(self, alpha: float, floor: float):
        self.ew = RobustEW(alpha, floor=floor)
        self.prev = None

    def init(self, buf):
        d = [b - a for a, b in zip(buf[:-1], buf[1:])]
        med = _median(d)
        self.ew.init(med, _mad(d, med))
        self.prev = buf[-1]

    def score(self, x):
        return self.ew.z(x - self.prev)

    def update(self, x):
        self.ew.update(x - self.prev)
        self.prev = x

    def describe(self, x):
        return {"step": x - self.prev, "typical_step": self.ew.scale}


class SeasonalExpert:
    """Breaks in a repeating pattern: residual against the value one period ago."""

    name = "seasonal"

    def __init__(self, period: int, alpha: float, floor: float):
        self.period = period
        self.ew = RobustEW(alpha, floor=floor)
        self.ring = []
        self.pos = 0

    def init(self, buf):
        p = self.period
        r = [buf[i] - buf[i - p] for i in range(p, len(buf))]
        med = _median(r)
        self.ew.init(med, _mad(r, med))
        self.ring = list(buf[-p:])
        self.pos = 0

    def _lagged(self):
        return self.ring[self.pos]

    def score(self, x):
        return self.ew.z(x - self._lagged())

    def update(self, x):
        self.ew.update(x - self._lagged())
        self.ring[self.pos] = x
        self.pos = (self.pos + 1) % self.period

    def describe(self, x):
        return {"same_time_last_period": self._lagged(), "period_pts": self.period}


class ShiftExpert:
    """Sustained level shifts: fast mean drifting away from the slow baseline."""

    name = "shift"

    def __init__(self, alpha_fast: float, alpha_slow: float, floor: float):
        self.af = alpha_fast
        self.slow = RobustEW(alpha_slow, floor=floor)
        self.fast = 0.0

    def init(self, buf):
        med = _median(buf)
        self.slow.init(med, _mad(buf, med))
        self.fast = med

    def score(self, x):
        f = self.fast + self.af * (x - self.fast)
        return (f - self.slow.m) / self.slow.scale

    def update(self, x):
        self.fast += self.af * (x - self.fast)
        self.slow.update(x)

    def describe(self, x):
        return {"recent_mean": self.fast + self.af * (x - self.fast),
                "baseline_mean": self.slow.m}


class VolatilityExpert:
    """Changes in noise level (bursts or flat-lining), on step changes."""

    name = "volatility"

    def __init__(self, alpha_fast: float, alpha_slow: float, floor: float):
        self.af, self.asl, self.floor = alpha_fast, alpha_slow, floor
        self.fast = self.slow = floor
        self.prev = None

    def init(self, buf):
        d = [abs(b - a) for a, b in zip(buf[:-1], buf[1:])]
        m = max(sum(d) / len(d), self.floor)
        self.fast = self.slow = m
        self.prev = buf[-1]

    def _fast_next(self, x):
        return self.fast + self.af * (abs(x - self.prev) - self.fast)

    def score(self, x):
        f = max(self._fast_next(x), self.floor)
        return math.log(f / max(self.slow, self.floor))

    def update(self, x):
        a = abs(x - self.prev)
        self.fast += self.af * (a - self.fast)
        lim = 5 * self.slow
        self.slow += self.asl * (min(a, lim) - self.slow)
        self.prev = x

    def describe(self, x):
        return {"recent_volatility": self._fast_next(x), "baseline_volatility": self.slow}


class DiscordExpert:
    """Novelty: distance from the latest m-point window to its nearest
    neighbour among the past `horizon` windows (a streaming left matrix
    profile). Patterns that have occurred before score low, so a recurring
    burst stops alarming after its first appearance.
    """

    name = "novelty"

    def __init__(self, m: int, horizon: int):
        self.m, self.h = m, horizon
        self.cap = 2 * (horizon + 2 * m)
        self.buf = np.zeros(self.cap)
        self.n = 0
        self._last = (0.0, -1)

    def init(self, buf):
        for x in buf:
            self.update(x)

    def _nn(self, x):
        m = self.m
        if self.n < 3 * m:
            return 0.0, -1
        # past windows must end at least m points before the query (no trivial match)
        lo = max(0, self.n - self.h - m)
        best, j = _nearest_window(self.buf, self.n, m, lo, x)
        return math.sqrt(best / m), j

    def score(self, x):
        self._last = self._nn(x)
        return self._last[0]

    def update(self, x):
        if self.n == self.cap:
            keep = self.h + 2 * self.m
            self.buf[:keep] = self.buf[self.n - keep:self.n]
            self.n = keep
        self.buf[self.n] = x
        self.n += 1

    def describe(self, x):
        d, j = self._last
        return {"nn_distance": d, "nn_age": (self.n - self.m + 1 - j) if j >= 0 else None,
                "window": self.m}
