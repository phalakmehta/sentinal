"""Run Sentinel on a folder of time series and write the submission files.

    python predict.py                                   # NAB corpus in data/nab
    python predict.py --data path/to/csvs --labels windows.json --out predictions

Input : every *.csv under --data with columns `timestamp,value` (any depth).
Output (in --out):
  predictions.csv  one row per point:
      series, timestamp, value, anomaly_score (0-1), score (-log10 p),
      threshold, label (NORMAL/ANOMALOUS), expert, event_id, explanation
  events.json      one record per alarm with the full evidence payload
  metrics.json     event-F1 / precision / recall / delay (only with labels)
  runtime.json     per-point streaming latency statistics
Each series is processed strictly as a stream: the output at time t only
depends on values up to t.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.data import NAB_DIR, load_series
from sentinel.engine import Sentinel, SentinelParams
from sentinel.metrics import EventResult, evaluate_series

WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))


EXPERTS = ("level", "jump", "seasonal", "shift", "volatility", "novelty")


def process(args):
    """Stream one series. With diag=True also return each expert's surprise
    (-log10 tail probability) per point and its final tail shape; this is
    bookkeeping only and never changes a decision."""
    s, params, *rest = args
    diag = bool(rest and rest[0])
    ts = s.timestamps
    step = float(np.median(np.diff(ts)) / np.timedelta64(1, "s")) if len(ts) > 1 else 300.0
    sen = Sentinel(params, step_seconds=step)
    n = len(s)
    score = np.zeros(n)
    label = np.empty(n, dtype=object)
    expert = np.empty(n, dtype=object)
    event_id = np.full(n, -1)
    expl = np.empty(n, dtype=object)
    lat = np.zeros(n)
    sup = {e: np.full(n, np.nan, dtype=np.float32) for e in EXPERTS} if diag else None
    ev = -1
    for i in range(n):
        t0 = time.perf_counter_ns()
        o = sen.update(ts[i], float(s.values[i]))
        lat[i] = time.perf_counter_ns() - t0
        if diag and i >= params.warmup:
            for e, p in zip(sen.experts, sen.last_ps):
                sup[e.name][i] = -math.log10(max(p, 1e-300))
        score[i], label[i], expert[i] = o.score, o.label, o.expert or ""
        if o.evidence is not None:
            ev += 1
            o.evidence["event_id"] = ev
            o.evidence["series"] = s.name
            expl[i] = o.evidence["explanation"]
        if o.label == "ANOMALOUS":
            event_id[i] = ev
    thr = sen.threshold
    df = pd.DataFrame({
        "series": s.name, "timestamp": pd.to_datetime(ts), "value": s.values,
        "anomaly_score": np.round(score / (score + thr), 4),
        "score": np.round(score, 3), "threshold": round(thr, 3), "label": label,
        "expert": expert, "event_id": event_id, "explanation": expl,
    })
    res = evaluate_series(ts, label == "ANOMALOUS", s.windows) if s.windows is not None else None
    post = lat[params.warmup:]  # steady-state streaming latency
    if not diag:
        return df, sen.events, res, post
    for e in EXPERTS:
        if e in params.experts:
            df["surprise_" + e] = np.round(sup[e], 3)
    tails = {e.name: {"gamma": round(tl.gamma, 3), "sigma": round(tl.sigma, 4), "peaks": tl.nt,
                      "seen": tl.n} for e, tl in zip(sen.experts, sen.tails)}
    return df, sen.events, res, post, tails


def discover(data: Path, labels: dict):
    out = []
    for f in sorted(data.rglob("*.csv")):
        name = f.relative_to(data).as_posix()
        out.append(load_series(f, name, labels.get(name, []) if labels is not None else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=NAB_DIR / "series")
    ap.add_argument("--labels", type=Path, default=NAB_DIR / "combined_windows.json")
    ap.add_argument("--out", type=Path, default=Path("predictions"))
    ap.add_argument("--no-labels", action="store_true", help="skip metrics")
    a = ap.parse_args()
    labels = None
    if not a.no_labels and a.labels and a.labels.exists():
        labels = json.loads(a.labels.read_text())
    series = discover(a.data, labels)
    if labels is None:
        for s in series:
            s.windows = None
    params = SentinelParams()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with Pool(WORKERS) as pool:
        results = pool.map(process, [(s, params) for s in series])
    wall = time.perf_counter() - t0

    pd.concat([r[0] for r in results]).to_csv(a.out / "predictions.csv", index=False)
    events = [e for r in results for e in r[1]]
    (a.out / "events.json").write_text(json.dumps(events, indent=1, default=str))

    lat = np.concatenate([r[3] for r in results]) / 1e3
    runtime = {"points": int(sum(len(s) for s in series)), "series": len(series),
               "wall_seconds": round(wall, 2), "workers": WORKERS,
               "latency_us_p50": round(float(np.percentile(lat, 50)), 1),
               "latency_us_p99": round(float(np.percentile(lat, 99)), 1),
               "latency_us_mean": round(float(lat.mean()), 1),
               "throughput_points_per_sec_per_core": int(1e6 / lat.mean())}
    (a.out / "runtime.json").write_text(json.dumps(runtime, indent=1))
    print("runtime:", json.dumps(runtime))

    if labels is not None:
        tot, per = EventResult(), {}
        for s, r in zip(series, results):
            tot += r[2]
            per.setdefault(s.category, EventResult()).__iadd__(r[2])
        metrics = {"overall": tot.summary(),
                   "median_delay_minutes": round(float(np.median(tot.delays_sec)) / 60, 1) if tot.delays_sec else None,
                   "per_category": {c: r.summary() for c, r in sorted(per.items())},
                   "params": params.to_dict()}
        (a.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        print("metrics:", json.dumps(metrics["overall"]))
    print(f"wrote {a.out}/predictions.csv ({runtime['points']} rows), {len(events)} events")


if __name__ == "__main__":
    main()
