"""Profile the NAB corpus: lengths, sampling, windows, probation overlap."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from sentinel.data import load_nab
from sentinel.metrics import window_indices

rows = []
for s in load_nab():
    n = len(s)
    step = pd.Series(np.diff(s.timestamps)).mode()[0] / np.timedelta64(1, "m")
    widx = window_indices(s.timestamps, s.windows)
    wlen = [b - a + 1 for a, b in widx]
    probation = int(0.15 * n)
    rows.append({
        "series": s.name, "n": n, "step_min": step,
        "gaps": int((np.diff(s.timestamps) != np.median(np.diff(s.timestamps))).sum()),
        "nan": int(np.isnan(s.values).sum()), "windows": len(s.windows),
        "win_len_mean": np.mean(wlen) if wlen else 0,
        "win_frac": sum(wlen) / n,
        "first_win_start_frac": round(widx[0][0] / n, 3) if widx else None,
        "wins_in_probation": sum(a < probation for a, _ in widx),
    })
df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_rows", 100)
print(df.to_string(index=False))
print("\nseries:", len(df), "points:", df.n.sum(), "windows:", df.windows.sum(),
      "no-window series:", (df.windows == 0).sum(),
      "windows starting in first 15%:", df.wins_in_probation.sum())
print("mean fraction of points inside windows:", round(df.win_frac.mean(), 3))
