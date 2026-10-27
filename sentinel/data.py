"""Dataset loading for NAB-format time series.

A dataset is a folder of CSVs (columns: timestamp, value) plus a JSON file
mapping each relative CSV path to a list of [start, end] anomaly windows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NAB_DIR = ROOT / "data" / "nab"


@dataclass
class Series:
    name: str                      # e.g. "realKnownCause/nyc_taxi.csv"
    timestamps: np.ndarray         # datetime64[ns]
    values: np.ndarray             # float64
    windows: list = field(default_factory=list)   # [(start, end)] as datetime64
    labels: list = field(default_factory=list)    # anomaly point timestamps

    @property
    def category(self) -> str:
        return self.name.split("/")[0]

    def __len__(self) -> int:
        return len(self.values)


def load_series(csv_path: Path, name: str, windows=None, labels=None) -> Series:
    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df["timestamp"]).to_numpy(dtype="datetime64[ns]")
    vals = df["value"].to_numpy(dtype=np.float64)
    wins = [(np.datetime64(pd.Timestamp(a)), np.datetime64(pd.Timestamp(b)))
            for a, b in (windows or [])]
    labs = [np.datetime64(pd.Timestamp(t)) for t in (labels or [])]
    return Series(name, ts, vals, wins, labs)


def load_nab(root: Path = NAB_DIR, categories=None) -> list[Series]:
    windows = json.loads((root / "combined_windows.json").read_text())
    labels_path = root / "combined_labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    out = []
    for name in sorted(windows):
        if categories and name.split("/")[0] not in categories:
            continue
        out.append(load_series(root / "series" / name, name,
                               windows[name], labels.get(name)))
    return out
