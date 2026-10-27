"""Run Sentinel over the corpus, print event metrics and log the experiment.

    python scripts/benchmark.py --tag v1
    python scripts/benchmark.py --tag lowrisk --set risk=1e-5 cooldown=500
"""
import os

# One BLAS/OpenMP thread per worker process: N workers x N threads each
# oversubscribes the CPU and can starve everything else on the machine.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))
import argparse
import csv
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

from sentinel.data import load_nab
from sentinel.engine import SentinelParams, run_series
from sentinel.metrics import EventResult, evaluate_series

LOG = ROOT / "experiments.csv"


def log_rows(rows):
    """Append rows to experiments.csv, widening the header if params were added."""
    old = list(csv.DictReader(LOG.open(newline=""))) if LOG.exists() else []
    fields = list(old[0].keys()) if old else []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with LOG.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(old + rows)


def _parse_sets(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k == "experts":
            out[k] = tuple(v.split("+"))
        else:
            out[k] = json.loads(v)
    return out


def _job(args):
    s, params = args
    t0 = time.perf_counter()
    _, labels, events = run_series(s.values, s.timestamps, params)
    dt = time.perf_counter() - t0
    return s.name, evaluate_series(s.timestamps, labels, s.windows), dt, len(s), events


def run(params, series, workers=WORKERS):
    with Pool(workers) as pool:
        return pool.map(_job, [(s, params) for s in series])


def report(results, verbose=True):
    total, per_cat = EventResult(), {}
    secs = pts = 0
    for name, r, dt, n, _ in results:
        total += r
        per_cat.setdefault(name.split("/")[0], EventResult()).__iadd__(r)
        secs += dt
        pts += n
    if verbose:
        rows = [{"category": c, **r.summary()} for c, r in sorted(per_cat.items())]
        print(pd.DataFrame(rows).to_string(index=False))
    summ = total.summary()
    summ["us_per_point"] = round(1e6 * secs / pts, 2)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--set", nargs="*", help="param overrides key=value")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    params = SentinelParams(**_parse_sets(a.set))
    series = load_nab()
    res = run(params, series)
    summ = report(res)
    print("\nOVERALL", json.dumps(summ))
    if not a.no_log:
        row = {"time": datetime.now().isoformat(timespec="seconds"), "tag": a.tag,
               **summ, **params.to_dict()}
        log_rows([row])


if __name__ == "__main__":
    main()
