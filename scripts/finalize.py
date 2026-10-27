"""Turn a logged run into the submission files, without spending another run.

    python predict.py --no-labels          # inference only: predictions.csv, events.json, runtime.json
    python scripts/finalize.py 6           # check + write predictions/metrics.json from run 6

Checks that predict.py's current defaults equal the run's parameters and that
its labels are identical to the run's saved labels (the detector is
deterministic), then copies the run's metrics into predict.py's metrics.json
format. No metric is recomputed here.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from sentinel.engine import SentinelParams


def main():
    run = int(sys.argv[1])
    out = ROOT / "predictions"
    m = json.loads((ROOT / "runs" / f"run_{run:02d}" / "metrics.json").read_text())
    if SentinelParams().to_dict() != m["params"]:
        sys.exit("SentinelParams defaults differ from the run's parameters")
    saved = pd.read_parquet(ROOT / "runs" / f"run_{run:02d}" / "predictions.parquet", columns=["series", "label"])
    now = pd.read_csv(out / "predictions.csv", usecols=["series", "label"])
    # predict.py and the runner list series in a different order; compare series by series
    saved, now = (d.sort_values("series", kind="stable").reset_index(drop=True) for d in (saved, now))
    if not (saved.series.equals(now.series) and saved.label.equals(now.label)):
        sys.exit("predictions.csv labels differ from the run's saved labels")
    log = pd.read_csv(ROOT / "runs" / "run_log.csv")
    o = m["overall"]
    metrics = {"overall": {k: o[k] for k in ("f1", "precision", "recall", "windows", "detected", "alarms", "fp",
                                             "median_delay_pts", "median_delay_frac")},
               "median_delay_from_label_pts": o["median_delay_from_label"],
               "pct_detections_within_10_pts_of_label": o["pct_within_10_of_label"],
               "loco_held_out_f1": float(log.loc[log.run == run, "loco_f1"].iloc[0]),
               "source_run": run, "runs_used": int(len(log)), "run_budget": 15,
               "per_category": {c: {k: v[k] for k in metrics_keys(v)} for c, v in m["per_category"].items()},
               "params": m["params"]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"labels identical to run {run} ({len(now)} rows); wrote predictions/metrics.json:",
          json.dumps(metrics["overall"]))


def metrics_keys(v):
    return [k for k in ("f1", "precision", "recall", "windows", "detected", "alarms", "fp",
                        "median_delay_pts", "median_delay_frac") if k in v]


if __name__ == "__main__":
    main()
