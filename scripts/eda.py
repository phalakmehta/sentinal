"""Exploratory data analysis of the NAB corpus (no detector involved, uses 0 runs).

Writes:
  results/eda/series_profile.csv     one row per series: size, sampling, gaps, value traits
  results/eda/category_summary.csv   the same, summarised per category
  results/eda/anomaly_windows.csv    one row per labelled window: what kind of anomaly it is
  results/figures/eda_*.png          slide figures

    python scripts/eda.py

Anomaly types (first rule that matches, computed offline, so it may look at the
whole window; the detector itself never does this):
  level shift      the median after the labelled point moved >= 4 noise units
  new extreme      a spike/dip >= 5 noise units that is the largest seen so far
  repeated extreme a spike/dip >= 5 noise units, but an equal or larger one
                   already happened earlier in the series without a label
  volatility       the step-to-step noise doubled or halved inside the window
  subtle           none of the above (pattern / context changes)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from plotstyle import AQUA, BLUE, GREY, INK2, MUTED, ORANGE, SURF, WIN, plt, save
from sentinel.data import load_nab
from sentinel.metrics import window_indices

EDA = ROOT / "results" / "eda"
EDA.mkdir(parents=True, exist_ok=True)
WARMUP = 300
LABELS = json.loads((ROOT / "data" / "nab" / "combined_labels.json").read_text())
TYPES = ["level shift", "new extreme", "repeated extreme", "volatility", "subtle"]
TYPE_COLORS = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4"]   # categorical slots 1-5, fixed order
SHORT = {"artificialNoAnomaly": "Art. (no anomaly)", "artificialWithAnomaly": "Art. (with anomaly)",
         "realAWSCloudwatch": "AWS Cloudwatch", "realAdExchange": "Ad Exchange",
         "realKnownCause": "Known cause", "realTraffic": "Traffic", "realTweets": "Tweets"}


def robust_scale(v):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return 1e-9
    mad = 1.4826 * np.median(np.abs(v - np.median(v)))
    return mad if mad > 0 else max(float(np.std(v)), 1e-9)


def acf(x, lag):
    x = x[~np.isnan(x)]
    if lag < 1 or len(x) < 3 * lag:
        return np.nan
    x = x - x.mean()
    den = (x * x).sum()
    return float((x[lag:] * x[:-lag]).sum() / den) if den > 0 else np.nan


def longest_flat_run(x):
    same = np.concatenate([[False], np.diff(x) == 0])
    best = cur = 0
    for s in same:
        cur = cur + 1 if s else 0
        best = max(best, cur)
    return best + 1


def profile(s):
    x, ts = s.values, s.timestamps
    d = np.diff(ts) / np.timedelta64(1, "m")
    step = float(np.median(d))
    z = (x - np.nanmedian(x)) / robust_scale(x)
    day = round(1440 / step)
    widx = window_indices(ts, s.windows)
    wlen = [b - a + 1 for a, b in widx]
    return {
        "series": s.name, "category": s.category, "n": len(s), "days": round(float((ts[-1] - ts[0]) / np.timedelta64(1, "D")), 1),
        "step_min": step, "irregular_pct": round(100 * float((d != step).mean()), 2),
        "max_gap_steps": round(float(d.max() / step), 1), "unsorted_or_dup": int((d <= 0).sum()),
        "nan": int(np.isnan(x).sum()), "min": float(np.nanmin(x)), "median": float(np.nanmedian(x)),
        "max": float(np.nanmax(x)), "zero_pct": round(100 * float((x == 0).mean()), 1),
        "distinct_pct": round(100 * len(np.unique(x)) / len(x), 1),
        "flat_run_max": longest_flat_run(x),
        "tail_p999": round(float(np.nanpercentile(np.abs(z), 99.9)), 1),   # Gaussian noise gives 3.3
        "skew": round(float(pd.Series(x).skew()), 2),
        "acf_day": round(acf(x, day), 2), "acf_week": round(acf(x, 7 * day), 2),
        "drift": round(float((np.nanmedian(x[-len(x) // 5:]) - np.nanmedian(x[:len(x) // 5])) / robust_scale(x)), 2),
        "windows": len(widx), "win_len_mean": round(float(np.mean(wlen)), 1) if wlen else 0,
        "labelled_pct": round(100 * sum(wlen) / len(x), 1),
        "windows_in_warmup": sum(a < WARMUP for a, _ in widx),
    }


def classify(s):
    """One row per labelled window: offline description of the anomaly inside it."""
    x, ts = s.values, s.timestamps
    resid = x - pd.Series(x).rolling(51, center=True, min_periods=1).median().to_numpy()
    rows = []
    labs = [int(np.searchsorted(ts, np.datetime64(pd.Timestamp(t)))) for t in LABELS.get(s.name, [])]
    for k, (a, b) in enumerate(window_indices(ts, s.windows)):
        lab = next((l for l in labs if a <= l <= b), (a + b) // 2)
        row = {"series": s.name, "category": s.category, "window": k, "start": a, "end": b,
               "label_idx": lab, "len": b - a + 1, "pos_frac": round(a / len(x), 3),
               "starts_in_warmup": a < WARMUP}
        if a < 30:
            rows.append({**row, "type": "subtle", "note": "no history before window"})
            continue
        w = max(20, (b - a) // 2)
        base = x[max(0, a - 1000):a]
        rs = robust_scale(resid[max(0, a - 1000):a])
        peak = float(np.nanmax(np.abs(resid[a:b + 1]))) / rs
        prior_peak = float(np.nanmax(np.abs(resid[:a]))) / rs
        before, after = x[max(0, a - w):a], x[lab:min(len(x), lab + w)]
        shift = abs(float(np.nanmedian(after) - np.nanmedian(before))) / robust_scale(base)
        dv = np.nanstd(np.diff(x[a:b + 1])) / max(np.nanstd(np.diff(x[max(0, a - 2 * w):a])), 1e-12)
        if shift >= 4:
            typ = "level shift"
        elif peak >= 5 and peak > prior_peak:
            typ = "new extreme"
        elif peak >= 5:
            typ = "repeated extreme"
        elif dv >= 2 or dv <= 0.5:
            typ = "volatility"
        else:
            typ = "subtle"
        rows.append({**row, "peak_z": round(peak, 1), "prior_peak_z": round(prior_peak, 1),
                     "record_ratio": round(peak / max(prior_peak, 1e-9), 2),
                     "shift_z": round(shift, 1), "vol_ratio": round(float(dv), 2), "type": typ, "note": ""})
    return rows


# ------------------------------------------------------------------ figures
def cat_axis(ax, cats):
    ax.set_yticks(range(len(cats)), [SHORT[c] for c in cats])
    ax.invert_yaxis()
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")


def jitter(n, seed=0):
    return np.random.default_rng(seed).uniform(-0.18, 0.18, n)


def fig_dataset(prof):
    cats = sorted(prof.category.unique())
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"wspace": 0.08})
    g = prof.groupby("category")
    pts = g.n.sum().reindex(cats) / 1000
    axs[0].barh(range(len(cats)), pts, color=BLUE, height=0.55)
    for i, c in enumerate(cats):
        axs[0].text(pts[c] + 2, i, f"{pts[c]:.0f}k pts · {g.size()[c]} series · {int(g.windows.sum()[c])} windows",
                    va="center", fontsize=9, color=INK2)
    axs[0].set_xlim(0, pts.max() * 1.9)
    axs[0].set_xlabel("points (thousands)")
    axs[0].set_title("Size per category", loc="left")
    cat_axis(axs[0], cats)
    for i, c in enumerate(cats):
        v = prof[prof.category == c]
        axs[1].scatter(v.n, i + jitter(len(v), i), s=26, color=BLUE, edgecolor=SURF, lw=1, zorder=3)
        axs[2].scatter(v.step_min, i + jitter(len(v), i + 9), s=26, color=BLUE, edgecolor=SURF, lw=1, zorder=3)
    axs[1].set_xscale("log")
    axs[1].axvline(WARMUP, color=ORANGE, lw=1.4, ls="--")
    axs[1].text(WARMUP * 1.08, len(cats) - 0.55, "300-point\nwarm-up", color=INK2, fontsize=9)
    axs[1].set_xlabel("series length (points, log)")
    axs[1].set_title("Length of each series", loc="left")
    axs[2].set_xscale("log")
    axs[2].set_xticks([5, 10, 30, 60], ["5 min", "10 min", "30 min", "1 h"])
    axs[2].set_xlabel("time between points")
    axs[2].set_title("Sampling interval", loc="left")
    for ax in axs[1:]:
        cat_axis(ax, cats)
        ax.set_yticklabels([])
    save(fig, "extra/eda_dataset.png")


def fig_labels(win, prof):
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    axs[0].hist(win["len"], bins=np.arange(0, 600, 25), color=BLUE, edgecolor=SURF, lw=2)
    axs[0].set_xlabel("window length (points)")
    axs[0].set_ylabel("windows")
    axs[0].set_title("Windows are long: median %d points" % win["len"].median(), loc="left")
    axs[1].hist(win.pos_frac, bins=np.linspace(0, 1, 21), color=BLUE, edgecolor=SURF, lw=2)
    axs[1].set_xlabel("where the window starts (0 = series start, 1 = end)")
    axs[1].set_title("Anomalies appear anywhere in time", loc="left")
    lab = prof[prof.windows > 0].labelled_pct
    axs[2].hist(lab, bins=np.arange(0, 22, 1), color=BLUE, edgecolor=SURF, lw=2)
    axs[2].set_xlabel("% of the series inside a window")
    axs[2].set_ylabel("series")
    axs[2].set_title("About 10% of each series is labelled", loc="left")
    save(fig, "extra/eda_labels.png")


def fig_traits(prof):
    cats = sorted(prof.category.unique())
    specs = [("tail_p999", "Heavy tails: 99.9th pct of |robust z|", True, 3.3, "Gaussian = 3.3"),
             ("acf_day", "Daily rhythm: autocorrelation at 1 day", False, 0.4, "seasonal expert on"),
             ("irregular_pct", "Irregular sampling: % uneven gaps", False, None, None),
             ("distinct_pct", "Discreteness: % distinct values", False, None, None)]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw={"wspace": 0.1})
    for j, (ax, (col, title, logx, ref, reftxt)) in enumerate(zip(axs, specs)):
        for i, c in enumerate(cats):
            v = prof[prof.category == c][col].dropna()
            ax.scatter(v, i + jitter(len(v), i + 31 * j), s=26, color=BLUE, edgecolor=SURF, lw=1, zorder=3)
        if logx:
            ax.set_xscale("log")
        if ref is not None:
            ax.axvline(ref, color=ORANGE, lw=1.4, ls="--")
            ax.text(ref * 1.15 if logx else ref + 0.02, len(cats) - 0.62, reftxt, color=INK2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        cat_axis(ax, cats)
        if j:
            ax.set_yticklabels([])
    save(fig, "extra/eda_traits.png")


def fig_types(win):
    cats = [c for c in sorted(win.category.unique())]
    tab = pd.crosstab(win.category, win.type).reindex(index=cats, columns=TYPES, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 3.9))
    left = np.zeros(len(cats))
    for t, col in zip(TYPES, TYPE_COLORS):
        v = tab[t].to_numpy()
        ax.barh(range(len(cats)), v, left=left, color=col, height=0.6, edgecolor=SURF, lw=2, label=t)
        for i, (l, n) in enumerate(zip(left, v)):
            if n >= 2:
                ax.text(l + n / 2, i, str(n), ha="center", va="center", fontsize=9, color="#0b0b0b")
        left += v
    tot = tab.sum()
    ax.legend([f"{t} ({tot[t]})" for t in TYPES], ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9.5)
    ax.set_xlabel("labelled windows")
    ax.set_title("What kind of anomaly is inside each labelled window?", loc="left")
    cat_axis(ax, cats)
    save(fig, "extra/eda_anomaly_types_by_category.png")


# one plain example per anomaly type: the slide-3 figure
TYPE_EXAMPLES = [
    ("Biggest spike ever seen", "new extreme", "realAWSCloudwatch/ec2_network_in_257a54.csv", 0, (900, 2300),
     "62x bigger than anything before it"),
    ("Big spike, but seen before", "repeated extreme", "realAdExchange/exchange-4_cpc_results.csv", 1, (300, 900),
     "an earlier, unlabelled spike was 10x bigger"),
    ("Level shift", "level shift", "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv", 1, (1900, 3300),
     "jumps from 17 to 28 and stays there"),
    ("Subtle change", "subtle", "realKnownCause/nyc_taxi.csv", 1, (6400, 7900),
     "Thanksgiving: taxi demand peaks are lower"),
]


def fig_type_examples(series, win):
    by_name = {s.name: s for s in series}
    counts = win.type.replace({"volatility": "subtle"}).value_counts()
    fig, axs = plt.subplots(2, 2, figsize=(14, 6.4))
    for ax, (title, typ, name, k, (lo, hi), note) in zip(axs.flat, TYPE_EXAMPLES):
        s = by_name[name]
        a, b = window_indices(s.timestamps, s.windows)[k]
        hi = min(hi, len(s))
        ax.axvspan(a, b, color=WIN, lw=0)
        ax.plot(np.arange(lo, hi), s.values[lo:hi], color=BLUE, lw=0.9)
        ax.set_title(f"{title}  ·  {counts.get(typ, 0)} of {len(win)} windows", loc="left", pad=20)
        ax.text(0, 1.015, note, transform=ax.transAxes, va="bottom", color=INK2, fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])     # the shape is the message; the note gives the numbers
        ax.grid(False)
        ax.tick_params(labelsize=9)
    fig.text(0.01, -0.01, "Grey band = labelled anomaly window. x-axis = time step. "
             "Types are computed from the data alone (rules in scripts/eda.py).", color=INK2, fontsize=10)
    fig.tight_layout()
    save(fig, "eda_anomaly_types.png")


def main():
    series = load_nab()
    prof = pd.DataFrame([profile(s) for s in series])
    win = pd.DataFrame([r for s in series for r in classify(s)])
    prof.to_csv(EDA / "series_profile.csv", index=False)
    win.to_csv(EDA / "anomaly_windows.csv", index=False)
    g = prof.groupby("category")
    summ = pd.DataFrame({
        "series": g.size(), "points": g.n.sum(), "median_len": g.n.median(),
        "step_min": g.step_min.median(), "series_with_gaps": g.irregular_pct.apply(lambda v: int((v > 0).sum())),
        "nan": g.nan.sum(), "windows": g.windows.sum(), "median_win_len": win.groupby("category")["len"].median(),
        "labelled_pct": g.labelled_pct.mean().round(1), "tail_p999_median": g.tail_p999.median(),
        "acf_day_median": g.acf_day.median(), "windows_in_warmup": g.windows_in_warmup.sum(),
    }).fillna(0)
    summ.to_csv(EDA / "category_summary.csv")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summ.to_string())
    print("\nseries:", len(prof), "points:", prof.n.sum(), "windows:", len(win),
          "| unsorted/duplicate timestamps:", prof.unsorted_or_dup.sum(), "| NaNs:", prof.nan.sum())
    print("series with heavy tails (p99.9 |z| > 6):", int((prof.tail_p999 > 6).sum()), "of", len(prof))
    print("series with daily rhythm (acf_day >= 0.4):", int((prof.acf_day >= 0.4).sum()))
    print("\nanomaly types:\n", win.type.value_counts().to_string())
    print("\ntypes by category:\n", pd.crosstab(win.category, win.type).to_string())
    print("\nwindows starting in warm-up:", int(win.starts_in_warmup.sum()))
    fig_dataset(prof)
    fig_labels(win, prof)
    fig_traits(prof)
    fig_types(win)
    fig_type_examples(series, win)


if __name__ == "__main__":
    main()
