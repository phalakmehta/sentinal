"""Event-based evaluation: precision, recall, F1 and detection delay.

Definitions (applied per series, then micro-aggregated across series):
  * predicted event  = a maximal run of consecutive ANOMALOUS points
  * true positive    = ground-truth window overlapped by >= 1 predicted event
  * false positive   = predicted event that overlaps no ground-truth window
  * precision        = predicted events hitting a window / all predicted events
  * recall           = detected windows / all windows
  * delay            = first ANOMALOUS timestamp inside a window - window start
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def label_runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True runs."""
    lab = np.asarray(labels, dtype=bool)
    if not lab.any():
        return []
    d = np.diff(np.concatenate([[0], lab.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def window_indices(timestamps: np.ndarray, windows) -> list[tuple[int, int]]:
    ts = np.asarray(timestamps)
    return [(int(np.searchsorted(ts, a, "left")), int(np.searchsorted(ts, b, "right") - 1))
            for a, b in windows]


@dataclass
class EventResult:
    n_windows: int = 0
    n_detected: int = 0
    n_pred: int = 0
    n_pred_hit: int = 0
    delays_pts: list = field(default_factory=list)
    delays_frac: list = field(default_factory=list)   # delay / window length
    delays_sec: list = field(default_factory=list)

    @property
    def fp(self) -> int:
        return self.n_pred - self.n_pred_hit

    @property
    def precision(self) -> float:
        return self.n_pred_hit / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.n_detected / self.n_windows if self.n_windows else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def __iadd__(self, other: "EventResult") -> "EventResult":
        for k in ("n_windows", "n_detected", "n_pred", "n_pred_hit"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.delays_pts += other.delays_pts
        self.delays_frac += other.delays_frac
        self.delays_sec += other.delays_sec
        return self

    def summary(self) -> dict:
        med = lambda v: round(float(np.median(v)), 3) if v else None
        return {
            "f1": round(self.f1, 4), "precision": round(self.precision, 4),
            "recall": round(self.recall, 4), "windows": self.n_windows,
            "detected": self.n_detected, "alarms": self.n_pred, "fp": self.fp,
            "median_delay_pts": med(self.delays_pts),
            "median_delay_frac": med(self.delays_frac),
        }


def evaluate_series(timestamps: np.ndarray, labels: np.ndarray, windows) -> EventResult:
    ts = np.asarray(timestamps)
    runs = label_runs(labels)
    res = EventResult(n_windows=len(windows), n_pred=len(runs))
    hit = np.zeros(len(runs), dtype=bool)
    for ws, we in window_indices(ts, windows):
        first = None
        for k, (rs, re) in enumerate(runs):
            if rs <= we and re >= ws:
                hit[k] = True
                onset = max(rs, ws)
                first = onset if first is None else min(first, onset)
        if first is not None:
            res.n_detected += 1
            res.delays_pts.append(first - ws)
            res.delays_frac.append((first - ws) / max(we - ws + 1, 1))
            res.delays_sec.append(float((ts[first] - ts[ws]) / np.timedelta64(1, "s")))
    res.n_pred_hit = int(hit.sum())
    return res
