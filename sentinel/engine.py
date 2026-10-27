"""Sentinel streaming engine: experts -> EVT calibration -> fusion -> alarms.

Usage:
    sen = Sentinel(SentinelParams(), step_seconds=300)
    for ts, x in stream:
        out = sen.update(ts, x)      # out.label in {"NORMAL", "ANOMALOUS"}
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .experts import (DiffExpert, DiscordExpert, LevelExpert, SeasonalExpert,
                      ShiftExpert, VolatilityExpert)
from .threshold import TailModel


@dataclass
class SentinelParams:
    warmup: int = 300            # points used to learn the baseline (no alarms)
    risk: float = 1e-5           # target tail probability for an alarm
    hysteresis: float = 1.0      # alarm stays on while score > threshold - this
    max_event_len: int = 30      # cap on one ANOMALOUS run (points)
    cooldown: int = 300          # min points between two alarm onsets
    init_q: float = 0.95         # EVT peaks-over-threshold initial quantile
    tail_shrink: float = 200.0   # pseudo-peaks shrinking the GPD shape toward 0 (0 = off)
    alpha_level: float = 0.02
    alpha_diff: float = 0.02
    alpha_seasonal: float = 0.02
    alpha_fast: float = 0.1
    alpha_slow: float = 0.005
    novelty_gate: float = 0.0    # min novelty surprise (-log10 p) to allow an alarm (0 = off)
    cusum_k: float = 2.0         # evidence accumulation slack (0 = off)
    relearn: bool = True         # alarm still on when the cooldown ends -> re-learn baseline, no new alarm
    acf_min: float = 0.4         # autocorrelation needed to enable seasonal expert
    discord_m: int = 8           # novelty expert window length
    discord_h: int = 2000        # novelty expert memory (past points searched)
    experts: tuple = ("level", "seasonal", "shift", "volatility", "novelty")
    history: int = 6000          # bounded raw history (for period detection)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["experts"] = "+".join(self.experts)
        return d


@dataclass
class StepOutput:
    score: float
    threshold: float
    label: str
    expert: str | None = None
    evidence: dict | None = field(default=None)   # set on alarm onset only


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    den = float((x * x).sum())
    return float((x[lag:] * x[:-lag]).sum() / den) if den > 0 else 0.0


class Sentinel:
    def __init__(self, params: SentinelParams | None = None, step_seconds: float = 300.0):
        self.p = params or SentinelParams()
        self.step_seconds = step_seconds
        self.threshold = -math.log10(self.p.risk)
        self.hist: deque = deque(maxlen=self.p.history)
        self.t = 0
        self.experts = []
        self.tails = []
        self.seasonal_checked_at = -1
        self.in_event = False
        self.event_len = 0
        self.acc = 0.0
        self.nov_idx = None
        self.last_onset = -10 ** 9
        self.events: list[dict] = []
        self.relearns: list[dict] = []    # regime changes absorbed by re-learning

    # ---------------------------------------------------------------- setup
    def _floor(self, buf) -> float:
        a = np.asarray(buf)
        d = np.abs(np.diff(a))
        d = d[d > 0]
        quantum = float(d.min()) if len(d) else 0.0
        return max(0.5 * quantum, 1e-6 * float(np.median(np.abs(a))), 1e-9)

    def _make(self, name, floor, period=None):
        p = self.p
        if name == "level":
            return LevelExpert(p.alpha_level, floor)
        if name == "jump":
            return DiffExpert(p.alpha_diff, floor)
        if name == "seasonal":
            return SeasonalExpert(period, p.alpha_seasonal, floor)
        if name == "shift":
            return ShiftExpert(p.alpha_fast, p.alpha_slow, floor)
        if name == "volatility":
            return VolatilityExpert(p.alpha_fast, p.alpha_slow, floor)
        if name == "novelty":
            return DiscordExpert(p.discord_m, p.discord_h)
        raise ValueError(name)

    def _calibrate(self, expert, buf):
        """Init on the head of `buf`, replay the rest to build the tail."""
        buf = list(buf)
        k = max(len(buf) // 4, getattr(expert, "period", 0) + 10)
        expert.init(buf[:k])
        scores = []
        for x in buf[k:]:
            scores.append(abs(expert.score(x)))
            expert.update(x)
        return TailModel(scores, self.p.init_q, self.p.tail_shrink)

    def _periods(self):
        day = round(86400 / self.step_seconds)
        return [day, 7 * day] if day >= 4 else []

    def _try_seasonal(self):
        if "seasonal" not in self.p.experts or any(e.name == "seasonal" for e in self.experts):
            return
        buf = np.asarray(self.hist)
        for period in self._periods():
            if len(buf) >= 3 * period and _acf(buf, period) >= self.p.acf_min:
                e = self._make("seasonal", self.floor, period)
                self.tails.append(self._calibrate(e, self.hist))
                self.experts.append(e)
                return

    def _setup(self):
        self.floor = self._floor(self.hist)
        for name in self.p.experts:
            if name == "seasonal":
                continue
            e = self._make(name, self.floor)
            self.tails.append(self._calibrate(e, self.hist))
            self.experts.append(e)
        names = [e.name for e in self.experts]
        self.nov_idx = names.index("novelty") if "novelty" in names else None
        self._try_seasonal()

    def _relearn(self, ts):
        """The alarm condition outlasted a whole cooldown: this is a new normal,
        not a new incident. Rebuild every expert and its tail from recent data
        (the last `warmup` points; novelty and seasonal keep their long memory)."""
        recent = list(self.hist)[-self.p.warmup:]
        self.floor = self._floor(recent)
        for j, e in enumerate(self.experts):
            buf = self.hist if e.name in ("novelty", "seasonal") else recent
            ne = self._make(e.name, self.floor, getattr(e, "period", None))
            self.tails[j] = self._calibrate(ne, buf)
            self.experts[j] = ne
        self.acc = 0.0
        self.last_onset = self.t
        self.relearns.append({"t": self.t, "timestamp": str(ts)})

    # ---------------------------------------------------------------- stream
    def update(self, ts, x: float) -> StepOutput:
        self.t += 1
        if x != x:   # NaN: hold state, report normal
            return StepOutput(0.0, self.threshold, "NORMAL")
        if self.t <= self.p.warmup:
            self.hist.append(x)
            if self.t == self.p.warmup:
                self._setup()
            return StepOutput(0.0, self.threshold, "NORMAL")

        zs = [e.score(x) for e in self.experts]
        ps = [tl.tail_prob(abs(z)) for tl, z in zip(self.tails, zs)]
        self.last_zs, self.last_ps = zs, ps      # exposed for plots / debugging
        k = min(range(len(ps)), key=ps.__getitem__)
        p_comb = min(1.0, ps[k] * len(ps))          # Bonferroni fusion
        point_score = -math.log10(max(p_comb, 1e-300))
        # novelty gate: other experts may only alarm on a pattern that is also
        # somewhat new; otherwise the score is capped at the novelty evidence
        if self.p.novelty_gate > 0 and self.nov_idx is not None:
            nov = -math.log10(max(ps[self.nov_idx], 1e-300))
            if nov < self.p.novelty_gate:
                point_score = min(point_score, nov)
        # CUSUM on surprise: sustained moderate evidence adds up to an alarm
        if self.p.cusum_k > 0:
            self.acc = max(0.0, self.acc + point_score - self.p.cusum_k)
            score = max(point_score, self.acc)
        else:
            score = point_score

        onset = relearn = False
        if self.in_event:
            self.event_len += 1
            if score < self.threshold - self.p.hysteresis or self.event_len >= self.p.max_event_len:
                self.in_event = False
        elif score >= self.threshold and self.t - self.last_onset > self.p.cooldown:
            if self.p.relearn and self.t - self.last_onset == self.p.cooldown + 1:
                relearn = True
            else:
                self.in_event, self.event_len, self.last_onset, onset = True, 0, self.t, True
                self.acc = 0.0

        out = StepOutput(score, self.threshold,
                         "ANOMALOUS" if self.in_event else "NORMAL",
                         self.experts[k].name if score >= self.threshold else None)
        if onset:
            out.evidence = self._evidence(ts, x, zs, ps, k, score)
            self.events.append(out.evidence)

        # winsorize at the per-expert level that would have fired an alarm
        p_alarm = self.p.risk / len(self.experts)
        for e, tl, z in zip(self.experts, self.tails, zs):
            a = abs(z)
            tl.update(a, cap=tl.level_for(p_alarm) if a > tl.t else None)
            e.update(x)

        self.hist.append(x)
        if relearn:
            self._relearn(ts)
        if len(self.experts) < len(self.p.experts) and self.t % 500 == 0:
            self._try_seasonal()
        return out

    # -------------------------------------------------------------- evidence
    def _evidence(self, ts, x, zs, ps, k, score):
        from .evidence import explain
        detail = {e.name: {"z": round(z, 2), "p": float(f"{p:.3g}")}
                  for e, z, p in zip(self.experts, zs, ps)}
        top = self.experts[k]
        info = top.describe(x)
        return {
            "timestamp": str(ts), "value": x, "score": round(score, 2),
            "threshold": round(self.threshold, 2), "expert": top.name,
            "z": round(zs[k], 2), "experts": detail, "info": info,
            "explanation": explain(top.name, x, zs[k], info),
        }


def run_series(values, timestamps, params: SentinelParams | None = None):
    """Run a full series through a fresh engine. Returns arrays + events."""
    ts = np.asarray(timestamps)
    step = float(np.median(np.diff(ts)) / np.timedelta64(1, "s")) if len(ts) > 1 else 300.0
    sen = Sentinel(params, step_seconds=step)
    n = len(values)
    scores = np.zeros(n)
    labels = np.zeros(n, dtype=bool)
    for i in range(n):
        o = sen.update(ts[i], float(values[i]))
        scores[i] = o.score
        labels[i] = o.label == "ANOMALOUS"
    return scores, labels, sen.events
