"""Data for the web deck -> web/data.js (window.SENTINEL = {...}).

Reads saved outputs only (runs/, results/, predictions/), so no experiment run is spent.
Series are shrunk with min-max buckets so every spike survives, then scaled to 0..1.

    python scripts/web_deck_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from deck_figures import three_sigma_events
from sentinel.data import load_nab
from sentinel.metrics import window_indices

OUT = ROOT / "web" / "data.js"
SERIES = {s.name: s for s in load_nab()}
ALARMS = {r: pd.read_csv(ROOT / "runs" / f"run_{r:02d}" / "alarms.csv") for r in (2, 6)}


def minmax(values, n=700):
    """Indices of the min and max in each of n buckets, in time order."""
    edges = np.linspace(0, len(values), n + 1).astype(int)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            seg = values[a:b]
            idx.extend(sorted({a + int(np.argmin(seg)), a + int(np.argmax(seg))}))
    return np.array(idx)


def trace(name, run=6, n=700, extra=None):
    s = SERIES[name]
    v = s.values
    lo, hi = float(v.min()), float(v.max())
    norm = lambda x: round((float(x) - lo) / (hi - lo), 4)
    L = len(v) - 1
    idx = minmax(v, n)
    a = ALARMS[run]
    a = a[a.series == name]
    out = {
        "name": name,
        "n": len(v),
        "pts": [[round(i / L, 4), norm(v[i])] for i in idx],
        "windows": [[round(p / L, 4), round(q / L, 4)] for p, q in window_indices(s.timestamps, s.windows)],
        "alarms": [{"x": round(int(r.idx) / L, 4), "y": norm(v[int(r.idx)]), "hit": bool(r.hit),
                    "expert": r.expert, "score": round(float(r.score), 1), "why": r.explanation,
                    "t": str(r.timestamp)[:16]} for _, r in a.iterrows()],
        "start": str(s.timestamps[0])[:10],
        "end": str(s.timestamps[-1])[:10],
    }
    if extra:
        out.update(extra(s, L))
    return out


def sigma(s, L):
    ev = three_sigma_events(s.values)
    return {"sigma": [round(int(i) / L, 4) for i in ev]}


def score_trace(name, n=700):
    p = pd.read_csv(ROOT / "predictions" / "predictions.csv", usecols=["series", "score"])
    sc = p.loc[p.series == name, "score"].to_numpy()
    edges = np.linspace(0, len(sc), n + 1).astype(int)
    L = len(sc) - 1
    return [[round(a / L, 4), round(float(sc[a:b].max()), 2)] for a, b in zip(edges[:-1], edges[1:]) if b > a]


def main():
    m = json.loads((ROOT / "predictions" / "metrics.json").read_text())
    log = pd.read_csv(ROOT / "runs" / "run_log.csv")
    base = pd.read_csv(ROOT / "results" / "baselines.csv")
    delay = pd.read_csv(ROOT / "results" / "detection_delay.csv").pts.clip(-60, 60)
    hist, _ = np.histogram(delay, bins=np.arange(-62.5, 63, 5))

    t = pd.read_csv(ROOT / "results" / "eda" / "anomaly_windows.csv")[["series", "window", "type"]]
    t["type"] = t.type.replace({"volatility": "subtle"})
    w6 = pd.read_csv(ROOT / "runs" / "run_06" / "windows.csv").merge(t, on=["series", "window"])
    types = {k: {"n": int(len(g)), "caught": int(g.hit.sum())} for k, g in w6.groupby("type")}

    clean = "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"
    data = {
        "overall": m["overall"],
        "delay": {"median": m["median_delay_from_label_pts"], "within10": m["pct_detections_within_10_pts_of_label"],
                  "hist": hist.tolist()},
        "loco": m["loco_held_out_f1"],
        "categories": {k: v for k, v in m["per_category"].items() if v["windows"]},
        "baselines": base.to_dict("records"),
        "runs": [{"run": int(r.run), "f1": round(float(r.f1), 3), "decision": r.decision} for r in log.itertuples()],
        "types": types,
        "hero": trace("realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv", n=900),
        "machine": trace("realKnownCause/machine_temperature_system_failure.csv"),
        "fb": trace("realTweets/Twitter_volume_FB.csv", extra=sigma),
        "clean": {**trace(clean), "score": score_trace(clean)},
        "relearn_before": trace("realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv", run=2),
        "relearn_after": trace("realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv", run=6),
        "ibm": trace("realTweets/Twitter_volume_IBM.csv"),
        "cvs": trace("realTweets/Twitter_volume_CVS.csv"),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("window.SENTINEL = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    for k in ("hero", "machine", "fb", "clean", "relearn_before", "relearn_after", "ibm", "cvs"):
        a = data[k]["alarms"]
        print(f"  {k:15s} alarms {len(a):3d}  hits {sum(x['hit'] for x in a):3d}  windows {len(data[k]['windows'])}"
              + (f"  sigma {len(data[k]['sigma'])}" if "sigma" in data[k] else ""))
    print("  types", types)


if __name__ == "__main__":
    main()
