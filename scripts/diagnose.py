"""Per-series / per-expert breakdown of alarms (hits vs false positives)."""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from benchmark import _parse_sets, run
from sentinel.data import load_nab
from sentinel.engine import SentinelParams
from sentinel.metrics import window_indices


def main():
    params = SentinelParams(**_parse_sets(sys.argv[1:]))
    series = load_nab()
    res = run(params, series)
    hits, fps = Counter(), Counter()
    rows = []
    for s, (name, r, _, _, events) in zip(series, res):
        widx = window_indices(s.timestamps, s.windows)
        ts = s.timestamps
        for ev in events:
            i = int(np.searchsorted(ts, np.datetime64(pd.Timestamp(ev["timestamp"]))))
            hit = any(a <= i <= b for a, b in widx)
            (hits if hit else fps)[ev["expert"]] += 1
        rows.append({"series": name, "win": r.n_windows, "det": r.n_detected,
                     "alarms": r.n_pred, "fp": r.fp,
                     "experts": ",".join(f"{e['expert'][:3]}@{np.searchsorted(ts, np.datetime64(pd.Timestamp(e['timestamp'])))}" for e in events)[:110]})
    pd.set_option("display.width", 250, "display.max_colwidth", 110)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nhits by expert:", dict(hits))
    print("fps  by expert:", dict(fps))


if __name__ == "__main__":
    main()
