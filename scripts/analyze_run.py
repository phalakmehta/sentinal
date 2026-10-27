"""Error analysis of a finished run, from its saved outputs only (uses 0 runs).

    python scripts/analyze_run.py 1            # one run
    python scripts/analyze_run.py 2 --vs 1     # what changed between two runs

Joins the run's per-window results with the EDA anomaly types
(results/eda/anomaly_windows.csv, from scripts/eda.py).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import numpy as np
import pandas as pd

RUNS = ROOT / "runs"
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 60)


def load(run):
    d = RUNS / f"run_{run:02d}"
    win = pd.read_csv(d / "windows.csv")
    types = pd.read_csv(ROOT / "results" / "eda" / "anomaly_windows.csv")
    win = win.merge(types[["series", "window", "type", "record_ratio"]], on=["series", "window"], how="left")
    alarms = pd.read_csv(d / "alarms.csv")
    per = pd.read_csv(d / "per_series.csv")
    m = json.loads((d / "metrics.json").read_text())
    return win, alarms, per, m


def report(run):
    win, alarms, per, m = load(run)
    o = m["overall"]
    print(f"== run {run} ({m['tag']}): F1 {o['f1']}  P {o['precision']}  R {o['recall']}  "
          f"alarms {o['alarms']}  FP {o['fp']}")
    print("\ndetection rate by anomaly type:")
    print(win.groupby("type").hit.agg(hits="sum", windows="count", rate="mean").round(2).to_string())
    miss = win[~win.hit]
    ex = [c for c in win.columns if c.startswith("max_") and c != "max_score"]
    closest = miss[ex].fillna(-1).idxmax(axis=1).str[4:].where(miss[ex].notna().any(axis=1), "warm-up")
    print("\nmissed windows: expert that came closest:", closest.value_counts().to_dict())
    print("missed windows: best fused score buckets:",
          pd.cut(miss.max_score, [-1, 2, 3, 4, 5, 1e9]).value_counts().sort_index().to_dict())
    fp = alarms[~alarms.hit]
    print("\nfalse alarms by expert:", fp.expert.value_counts().to_dict())
    print("hits by expert       :", alarms[alarms.hit].expert.value_counts().to_dict())
    print("false alarms by category:", fp.category.value_counts().to_dict())
    g = [c for c in per.columns if c.startswith("gamma_")]
    print("\nfinal tail shape gamma per expert (median / share > 0.3):")
    for c in g:
        v = per[c].dropna()
        print(f"  {c[6:]:10s} median {v.median():.2f}   >0.3 in {int((v > 0.3).sum())}/{len(v)} series")
    return win, alarms


def compare(run, base):
    w1, a1, _, _ = load(run)
    w0, a0, _, _ = load(base)
    j = w1.merge(w0[["series", "window", "hit"]], on=["series", "window"], suffixes=("", "_base"))
    print(f"\n== run {run} vs run {base}")
    print("newly detected:")
    print(j[j.hit & ~j.hit_base][["series", "window", "type", "first_alarm", "delay_from_label"]].to_string(index=False))
    print("newly missed:")
    print(j[~j.hit & j.hit_base][["series", "window", "type", "max_score"]].to_string(index=False))
    key = lambda a: set(zip(a.series, a.idx))
    f1, f0 = a1[~a1.hit], a0[~a0.hit]
    new = f1[[k not in key(f0) for k in zip(f1.series, f1.idx)]]
    gone = f0[[k not in key(f1) for k in zip(f0.series, f0.idx)]]
    print(f"false alarms: {len(f0)} -> {len(f1)}  (new {len(new)}, removed {len(gone)})")
    print("new false alarms by expert:", new.expert.value_counts().to_dict(),
          " by category:", new.category.value_counts().to_dict())
    print(new[["series", "idx", "expert", "score", "dist_to_window"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=int)
    ap.add_argument("--vs", type=int)
    a = ap.parse_args()
    report(a.run)
    if a.vs:
        compare(a.run, a.vs)


if __name__ == "__main__":
    main()
