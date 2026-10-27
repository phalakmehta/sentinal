"""Budgeted experiment runner: at most 15 runs, each one logged before and after.

One run = one full fit-and-validation pass of one configuration over every
series (stream all 58 NAB series, score against the labelled windows).
EDA, plotting and reading the saved outputs of a finished run do not count.

    python scripts/experiment.py --run 1 --tag baseline --why "reproduce v5"
    python scripts/experiment.py --run 2 --tag shrink50 --why "..." --set tail_shrink=50
    python scripts/experiment.py --decide 2 keep "F1 up, 4/6 categories up"
    python scripts/experiment.py --status

Every run writes runs/run_NN/ (params, metrics, per-window, per-alarm, events,
per-point predictions with per-expert surprise) so no run ever needs repeating,
and appends one row to runs/run_log.csv. The script refuses run 16.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from benchmark import _parse_sets
from predict import WORKERS, discover, process
from sentinel.data import NAB_DIR, load_nab
from sentinel.engine import SentinelParams
from sentinel.metrics import EventResult, label_runs, window_indices

BUDGET = 15
RUNS = ROOT / "runs"
LOG = RUNS / "run_log.csv"
COUNTS = ("n_windows", "n_detected", "n_pred", "n_pred_hit")


def read_log():
    return list(csv.DictReader(LOG.open(newline=""))) if LOG.exists() else []


def write_log(rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with LOG.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def code_hash():
    h = hashlib.sha1()
    for p in sorted((ROOT / "sentinel").glob("*.py")) + [ROOT / "predict.py"]:
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


def counts(r: EventResult) -> dict:
    return {k: getattr(r, k) for k in COUNTS}


def from_counts(c: dict) -> EventResult:
    return EventResult(**{k: int(c[k]) for k in COUNTS})


def loco(run_cats):
    """run_cats: {run: {category: counts}}. For each category pick the run that
    is best on the other categories, then score that run on the held-out one."""
    cats = sorted({c for v in run_cats.values() for c in v})
    held, picks = EventResult(), {}
    for c in cats:
        def train_f1(run):
            tot = EventResult()
            for k, v in run_cats[run].items():
                if k != c:
                    tot += from_counts(v)
            return tot.f1
        best = max(run_cats, key=train_f1)
        held += from_counts(run_cats[best].get(c, dict.fromkeys(COUNTS, 0)))
        picks[c] = best
    return held, picks


def analyse(series, results, label_points):
    """Per-window and per-alarm tables from the streamed outputs."""
    win_rows, alarm_rows = [], []
    for s, (df, events, res, _, tails) in zip(series, results):
        lab = (df.label == "ANOMALOUS").to_numpy()
        score = df.score.to_numpy()
        sup = {c[9:]: df[c].to_numpy() for c in df.columns if c.startswith("surprise_")}
        runs = label_runs(lab)
        widx = window_indices(s.timestamps, s.windows or [])
        lpts = [int(np.searchsorted(s.timestamps, np.datetime64(pd.Timestamp(t))))
                for t in label_points.get(s.name, [])]
        for k, (a, b) in enumerate(widx):
            onsets = [max(rs, a) for rs, re in runs if rs <= b and re >= a]
            first = min(onsets) if onsets else None
            lp = next((p for p in lpts if a <= p <= b), None)
            row = {"series": s.name, "category": s.category, "window": k, "start": a, "end": b,
                   "label_idx": lp, "hit": first is not None, "first_alarm": first,
                   "delay_from_start": None if first is None else first - a,
                   "delay_from_label": None if first is None or lp is None else first - lp,
                   "max_score": round(float(score[a:b + 1].max()), 2) if b >= a else 0.0}
            for e, v in sup.items():
                seg = v[a:b + 1]
                row["max_" + e] = round(float(np.nanmax(seg)), 2) if np.isfinite(seg).any() else None
            win_rows.append(row)
        for rs, re in runs:
            hit = any(rs <= b and re >= a for a, b in widx)
            dist = min((0 if a <= rs <= b else min(abs(rs - a), abs(rs - b)) for a, b in widx), default=None)
            row = {"series": s.name, "category": s.category, "idx": rs, "timestamp": str(df.timestamp.iloc[rs]),
                   "length": re - rs + 1, "hit": hit, "dist_to_window": dist, "expert": df.expert.iloc[rs],
                   "score": float(df.score.iloc[rs]), "explanation": df.explanation.iloc[rs]}
            for e, v in sup.items():
                row["surprise_" + e] = round(float(v[rs]), 2) if np.isfinite(v[rs]) else None
            alarm_rows.append(row)
    return pd.DataFrame(win_rows), pd.DataFrame(alarm_rows)


def status():
    rows = read_log()
    print(f"runs used: {len(rows)} / {BUDGET}")
    if rows:
        cols = ["run", "tag", "f1", "precision", "recall", "alarms", "fp", "loco_f1", "decision"]
        print(pd.DataFrame(rows)[cols].to_string(index=False))


def decide(run, verdict, note):
    rows = read_log()
    for r in rows:
        if r["run"] == str(run):
            r["decision"], r["decision_note"] = verdict, note
            write_log(rows)
            print(f"run {run}: {verdict} ({note})")
            return
    sys.exit(f"run {run} not in log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--why", help="hypothesis, written before the run")
    ap.add_argument("--set", nargs="*", help="param overrides key=value")
    ap.add_argument("--decide", nargs=3, metavar=("RUN", "VERDICT", "NOTE"))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry", type=Path, help="smoke test on a non-NAB folder (needs --data); not counted")
    ap.add_argument("--data", type=Path)
    ap.add_argument("--labels", type=Path)
    a = ap.parse_args()
    if a.status:
        return status()
    if a.decide:
        return decide(*a.decide)
    if a.run is None or not a.why:
        sys.exit("--run and --why are required")

    if a.dry:
        if a.data is None or a.data.resolve() == (NAB_DIR / "series").resolve():
            sys.exit("--dry needs --data pointing at a synthetic (non-NAB) folder")
        out_dir, log_rows = a.dry / f"run_{a.run:02d}", None
        labels = json.loads(a.labels.read_text()) if a.labels else {}
        series, label_points = discover(a.data, labels), {}
    else:
        log_rows = read_log()
        if len(log_rows) >= BUDGET:
            sys.exit(f"budget exhausted: {len(log_rows)}/{BUDGET} runs used")
        if a.run != len(log_rows) + 1:
            sys.exit(f"next run number must be {len(log_rows) + 1}")
        out_dir = RUNS / f"run_{a.run:02d}"
        series = load_nab()
        label_points = json.loads((NAB_DIR / "combined_labels.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = _parse_sets(a.set)
    params = SentinelParams(**overrides)

    t0 = time.perf_counter()
    with Pool(WORKERS) as pool:
        results = pool.map(process, [(s, params, True) for s in series])
    wall = time.perf_counter() - t0

    total, per_cat = EventResult(), {}
    for s, r in zip(series, results):
        total += r[2]
        per_cat.setdefault(s.category, EventResult()).__iadd__(r[2])
    wins, alarms = analyse(series, results, label_points)
    lat = np.concatenate([r[3] for r in results]) / 1e3
    d_lab = wins.delay_from_label.dropna()
    summ = {**total.summary(),
            "median_delay_from_label": float(d_lab.median()) if len(d_lab) else None,
            "pct_within_10_of_label": round(100 * float((d_lab.abs() <= 10).mean()), 1) if len(d_lab) else None,
            "us_per_point_p50": round(float(np.percentile(lat, 50)), 1),
            "us_per_point_p99": round(float(np.percentile(lat, 99)), 1), "wall_seconds": round(wall, 1)}

    pd.concat([r[0] for r in results]).to_parquet(out_dir / "predictions.parquet", index=False)
    wins.to_csv(out_dir / "windows.csv", index=False)
    alarms.to_csv(out_dir / "alarms.csv", index=False)
    (out_dir / "events.json").write_text(json.dumps([e for r in results for e in r[1]], indent=1, default=str))
    pd.DataFrame([{"series": s.name, "category": s.category, **counts(r[2]),
                   **{f"{k}_{e}": v for e, t in r[4].items() for k, v in t.items()}}
                  for s, r in zip(series, results)]).to_csv(out_dir / "per_series.csv", index=False)
    metrics = {"run": a.run, "tag": a.tag, "hypothesis": a.why, "overrides": a.set or [],
               "code": code_hash(), "overall": summ,
               "per_category": {c: {**r.summary(), **counts(r)} for c, r in sorted(per_cat.items())},
               "params": params.to_dict()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))

    print(pd.DataFrame([{"category": c, **r.summary()} for c, r in sorted(per_cat.items())]).to_string(index=False))
    print("\nRUN", a.run, json.dumps(summ))
    if log_rows is None:
        print("dry run: nothing logged, budget untouched ->", out_dir)
        return

    run_cats = {}
    for r in log_rows + [{"run": str(a.run)}]:
        m = json.loads((RUNS / f"run_{int(r['run']):02d}" / "metrics.json").read_text())
        run_cats[int(r["run"])] = {c: {k: v[k] for k in COUNTS} for c, v in m["per_category"].items() if v["n_windows"]}
    held, picks = loco(run_cats)
    row = {"run": a.run, "time": datetime.now().isoformat(timespec="seconds"), "tag": a.tag,
           "hypothesis": a.why, "changes": " ".join(a.set or []) or "(none)",
           **{k: summ[k] for k in ("f1", "precision", "recall", "detected", "windows", "alarms", "fp",
                                   "median_delay_from_label", "pct_within_10_of_label", "us_per_point_p50")},
           "loco_f1": round(held.f1, 4), "code": metrics["code"],
           **{"f1_" + c.replace("real", "").replace("artificial", "art"): round(r.f1, 3)
              for c, r in sorted(per_cat.items()) if r.n_windows},
           "decision": "", "decision_note": ""}
    write_log(log_rows + [row])
    print(f"LOCO held-out F1 over runs 1..{a.run}: {held.f1:.4f}  picks: {picks}")
    print(f"budget: {len(log_rows) + 1}/{BUDGET} runs used")


if __name__ == "__main__":
    main()
