"""Evaluate the baselines with the same event metrics as Sentinel.

Each baseline gets its global threshold k swept and its *best* k reported,
which is an optimistic (oracle-tuned) number for the baseline. The ensemble
combines the standardised scores of all six other methods, by mean and by max,
and only the better of the two is reported (again in the baselines' favour).
"""
import os

# One BLAS/OpenMP thread per worker process: N workers x N threads each
# oversubscribes the CPU and can starve everything else on the machine.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sentinel.baselines import BASELINES, alarms_from_scores, standardise
from sentinel.data import load_nab
from sentinel.metrics import EventResult, evaluate_series

WARMUP = 300
KS = [3, 5, 8, 12, 20, 30, 50, 80]


def _job(args):
    s, name = args
    t0 = time.perf_counter()
    z = standardise(BASELINES[name](s.values, WARMUP), WARMUP)
    return name, z, time.perf_counter() - t0


def best_k(series, zs):
    best = None
    for k in KS:
        tot = EventResult()
        for s, z in zip(series, zs):
            tot += evaluate_series(s.timestamps, alarms_from_scores(z, k), s.windows)
        if best is None or tot.f1 > best[1].f1:
            best = (k, tot)
    return best


def main():
    series = load_nab()
    pts = sum(len(s) for s in series)
    z, secs = {n: [] for n in BASELINES}, {n: 0.0 for n in BASELINES}
    with Pool(WORKERS) as pool:
        for name in BASELINES:
            for n, zi, dt in pool.map(_job, [(s, name) for s in series]):
                z[n].append(zi)
                secs[n] += dt
    rows = []
    for name in BASELINES:
        k, tot = best_k(series, z[name])
        rows.append({"method": name, "best_k": k, **tot.summary(), "us_per_point": round(1e6 * secs[name] / pts, 2)})
        print(json.dumps(rows[-1]), flush=True)

    ens = None
    for how, f in (("mean", np.mean), ("max", np.max)):
        zs = [f(np.vstack([z[n][i] for n in BASELINES]), axis=0) for i in range(len(series))]
        k, tot = best_k(series, zs)
        print(f"ensemble ({how}): F1 {tot.f1:.4f} at k={k}", flush=True)
        if ens is None or tot.f1 > ens[2].f1:
            ens = (how, k, tot)
    rows.append({"method": f"ensemble_{ens[0]}", "best_k": ens[1], **ens[2].summary(),
                 "us_per_point": round(1e6 * sum(secs.values()) / pts, 2)})
    print(json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "baselines.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    (ROOT / "results").mkdir(exist_ok=True)
    main()
