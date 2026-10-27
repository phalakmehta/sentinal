"""Grid search over a few global knobs with leave-one-category-out (LOCO) scoring.

Every configuration is logged to experiments.csv. The LOCO score picks the best
config on 6 of the 7 NAB categories and evaluates it on the held-out one, which
estimates how a config chosen this way generalises to unseen series.

    python scripts/tune.py --tag sweep1 --grid risk=1e-3,1e-4,1e-5 cooldown=150,300
"""
import argparse
import csv
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import _parse_sets, log_rows, report, run
from sentinel.data import load_nab
from sentinel.engine import SentinelParams
from sentinel.metrics import EventResult


def _values(spec):
    k, v = spec.split("=", 1)
    if k == "experts":
        return k, [tuple(x.split("+")) for x in v.split(",")]
    return k, [json.loads(x) for x in v.split(",")]


def by_category(results):
    cats = {}
    for name, r, *_ in results:
        cats.setdefault(name.split("/")[0], EventResult()).__iadd__(r)
    return cats


def total(cats):
    tot = EventResult()
    for r in cats.values():
        tot += r
    return tot


def loco(configs):
    """configs: list of (label, {category: EventResult})."""
    cats = sorted(configs[0][1])
    held = EventResult()
    picks = []
    for c in cats:
        def train_f1(cfg):
            tot = EventResult()
            for k, r in cfg[1].items():
                if k != c:
                    tot += r
            return tot.f1
        best = max(configs, key=train_f1)
        held += best[1][c]
        picks.append((c, best[0]))
    return held, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--grid", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", help="fixed overrides")
    a = ap.parse_args()
    fixed = _parse_sets(a.set)
    keys, vals = zip(*[_values(g) for g in a.grid])
    series = load_nab()
    configs, rows = [], []
    for combo in itertools.product(*vals):
        over = dict(zip(keys, combo))
        params = SentinelParams(**{**fixed, **over})
        res = run(params, series)
        summ = report(res, verbose=False)
        label = " ".join(f"{k}={'+'.join(v) if isinstance(v, tuple) else v}" for k, v in over.items())
        print(f"{label:60s} f1={summ['f1']:.4f} p={summ['precision']:.3f} "
              f"r={summ['recall']:.3f} alarms={summ['alarms']}", flush=True)
        configs.append((label, by_category(res)))
        rows.append({"time": datetime.now().isoformat(timespec="seconds"),
                     "tag": a.tag, **summ, **params.to_dict()})
    log_rows(rows)
    held, picks = loco(configs)
    best = max(configs, key=lambda c: total(c[1]).f1)
    print("\nbest in-sample:", best[0])
    print("LOCO held-out:", json.dumps(held.summary()))
    for c, lab in picks:
        print(f"  held-out {c:22s} <- picked {lab}")


if __name__ == "__main__":
    main()
