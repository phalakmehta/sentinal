"""Data for the web demo -> web/demo/data/ (the 58 NAB CSVs + manifest.json).

The manifest holds, per series: labelled windows as index pairs (same window_indices the
scorer uses) and the alarms of the submitted run (runs/run_06), so the page can show
"identical to our submitted predictions" after a replay. No experiment run is spent.

    python scripts/web_demo_data.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from sentinel.data import NAB_DIR, load_nab
from sentinel.metrics import window_indices

OUT = ROOT / "web" / "demo" / "data"


def main():
    alarms = pd.read_csv(ROOT / "runs" / "run_06" / "alarms.csv")
    series = []
    for s in load_nab():
        dst = OUT / "series" / s.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(NAB_DIR / "series" / s.name, dst)
        a = alarms[alarms.series == s.name]
        series.append({
            "name": s.name,
            "n": len(s),
            "windows": [list(w) for w in window_indices(s.timestamps, s.windows)],
            "ref": {"alarms": a.idx.astype(int).tolist(), "hits": a.hit.astype(bool).tolist()},
        })
    shutil.copyfile(NAB_DIR / "NAB_LICENSE.txt", OUT / "NAB_LICENSE.txt")
    (OUT / "manifest.json").write_text(json.dumps({"series": series}, separators=(",", ":")), encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {len(series)} series + manifest to {OUT.relative_to(ROOT)} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
