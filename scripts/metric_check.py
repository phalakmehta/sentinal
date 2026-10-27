"""Why we report event-F1 and not point-adjusted F1, from saved outputs only (uses 0 runs).

    python scripts/metric_check.py      # -> results/metric_check.csv

Scores the final labels (predictions/predictions.csv) three ways:
  * event-F1       : the challenge metric (sentinel/metrics.py)
  * point-wise F1  : every labelled window point is a positive, scored point by point
  * point-adjusted : as point-wise, but once any point of a window fires, the whole window counts
                     as found (the "point adjustment" used in many papers)
and compares them with detectors that know nothing: random alarms, and flagging every point.
Random rows are the mean of 5 seeds.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sentinel.data import load_nab
from sentinel.metrics import EventResult, evaluate_series, window_indices

SEEDS = range(5)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def score(series, preds):
    """preds: one boolean array per series. Returns event-F1, point-wise F1, point-adjusted F1."""
    ev = EventResult()
    pw = np.zeros(3)   # tp, fp, fn
    pa = np.zeros(3)
    for s, pred in zip(series, preds):
        idx = window_indices(s.timestamps, s.windows)
        truth = np.zeros(len(s), bool)
        adj = pred.copy()
        for a, b in idx:
            truth[a:b + 1] = True
            if pred[a:b + 1].any():
                adj[a:b + 1] = True
        ev += evaluate_series(s.timestamps, pred, s.windows)
        for acc, p in ((pw, pred), (pa, adj)):
            acc += [(p & truth).sum(), (p & ~truth).sum(), (~p & truth).sum()]
    return ev.f1, f1(*pw), f1(*pa), ev.n_pred


def main():
    series = load_nab()
    df = pd.read_csv(ROOT / "predictions" / "predictions.csv", usecols=["series", "label"])
    by = dict(tuple(df.groupby("series", sort=False)))
    ours = [(by[s.name]["label"] == "ANOMALOUS").to_numpy() for s in series]
    assert all(len(p) == len(s) for p, s in zip(ours, series))
    rate = np.concatenate(ours).mean()

    def random(p, seed):
        rng = np.random.default_rng(seed)
        return [rng.random(len(s)) < p for s in series]

    rows = [("Sentinel (run 6)", score(series, ours))]
    for name, p in ((f"random, {rate:.2%} of points (our alarm rate)", rate),
                    ("random, 1% of points", 0.01)):
        rows.append((name, tuple(np.mean([score(series, random(p, k)) for k in SEEDS], axis=0))))
    rows.append(("flag every point", score(series, [np.ones(len(s), bool) for s in series])))

    out = pd.DataFrame([(n, *v) for n, v in rows],
                       columns=["detector", "event_f1", "pointwise_f1", "point_adjusted_f1", "events"])
    out[["event_f1", "pointwise_f1", "point_adjusted_f1"]] = out[
        ["event_f1", "pointwise_f1", "point_adjusted_f1"]].round(3)
    out["events"] = out["events"].round().astype(int)
    out.to_csv(ROOT / "results" / "metric_check.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
