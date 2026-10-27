"""Slide-sized figures for the LaTeX deck -> deck/figures/*.pdf (vector, Fira Sans).

Every figure is drawn at exactly the size it has on the 16 x 9 cm slide and is
included at natural size, so figure text matches the slide text (7 pt).
Inputs are saved outputs only (runs/, results/); the rarity-curve illustration
replays one series without labels, so no experiment run is spent.

    python scripts/deck_figures.py
"""
import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from eda import TYPE_EXAMPLES          # same four examples as the EDA figure
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sentinel.data import load_nab
from sentinel.engine import Sentinel, SentinelParams
from sentinel.metrics import window_indices

OUT = ROOT / "deck" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#d03b3b"
INK, INK2, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#b5b3ab"
WIN = (137 / 255, 135 / 255, 129 / 255, 0.22)


def use_fira():
    try:
        reg = subprocess.run(["kpsewhich", "FiraSans-Regular.otf"], capture_output=True, text=True).stdout.strip()
    except OSError:
        reg = ""
    for f in glob.glob(str(Path(reg).parent / "FiraSans-*.otf")) if reg else []:
        fm.fontManager.addfont(f)
    return "Fira Sans" if reg else "DejaVu Sans"


plt.rcParams.update({
    "font.family": use_fira(), "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "axes.titleweight": "medium", "axes.titlecolor": INK, "axes.labelcolor": INK2,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GREY, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 6.5, "legend.labelcolor": INK2,
    "savefig.transparent": True, "figure.dpi": 150,
    "savefig.bbox": "standard",     # plotstyle (imported via eda) sets "tight", which pads every figure
    "axes.facecolor": "none", "figure.facecolor": "none",
})

SERIES = {s.name: s for s in load_nab()}
ALARMS = {r: pd.read_csv(ROOT / "runs" / f"run_{r:02d}" / "alarms.csv") for r in (1, 2, 6)}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")       # fixed size: no tight bbox, the layout is set per figure
    plt.close(fig)
    print("wrote", name)


def bands(ax, s):
    for a, b in window_indices(s.timestamps, s.windows):
        ax.axvspan(a, b, color=WIN, lw=0)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def alarms(ax, s, run, hit_color=INK2, size=20):
    a = ALARMS[run]
    a = a[a.series == s.name]
    for _, r in a.iterrows():
        ax.scatter([r.idx], [s.values[r.idx]], marker="v", s=size, color=hit_color if r.hit else RED,
                   edgecolor="white", lw=0.6, zorder=5)
    return a


def fig(w, h, **kw):
    """Figure of exactly w x h cm with the axes inset by the given margins (cm)."""
    f = plt.figure(figsize=(w * CM, h * CM))
    l, r, b, t = kw.get("l", 0.1), kw.get("r", 0.1), kw.get("b", 0.1), kw.get("t", 0.1)
    return f, f.add_axes([l / w, b / h, 1 - (l + r) / w, 1 - (b + t) / h])


# ---------------------------------------------------------------- slide 1
def hero():
    s = SERIES["realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv"]
    f, ax = fig(14.8, 2.1, l=0, r=0, b=0, t=0.15)
    bands(ax, s)
    ax.plot(s.values, color=BLUE, lw=0.5)
    alarms(ax, s, 6, hit_color=RED, size=30)
    ax.set_xlim(0, len(s))
    bare(ax)
    save(f, "hero")


# ---------------------------------------------------------------- slide 2
def machine():
    s = SERIES["realKnownCause/machine_temperature_system_failure.csv"]
    f, ax = fig(7.5, 4.4, l=1.0, b=0.5, t=0.1, r=0.1)
    t = pd.to_datetime(s.timestamps)
    for a, b in window_indices(s.timestamps, s.windows):
        ax.axvspan(t[a], t[b], color=WIN, lw=0)
    ax.plot(t, s.values, color=BLUE, lw=0.45)
    ax.set_ylabel("temperature")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(t[0], t[-1])
    save(f, "machine")


# ---------------------------------------------------------------- slide 3
def three_sigma_events(x):
    x = pd.Series(x)
    m = x.rolling(288, min_periods=100).mean().shift(1)
    sd = x.rolling(288, min_periods=100).std().shift(1)
    idx = np.flatnonzero(((x - m).abs() > 3 * sd).to_numpy())
    return idx[np.concatenate([[True], np.diff(idx) > 12])] if len(idx) else idx


def sigma_rule():
    s = SERIES["realTweets/Twitter_volume_FB.csv"]
    ev = three_sigma_events(s.values)
    f, ax = fig(8.9, 2.9, l=0, r=0, b=0.25, t=0.45)
    bands(ax, s)
    ax.plot(s.values, color=BLUE, lw=0.45)
    top = s.values.max()
    ax.scatter(ev, np.full(len(ev), -0.1 * top), marker="|", s=22, color=ORANGE, lw=0.8, clip_on=False)
    mine = alarms(ax, s, 6, hit_color=RED, size=26)
    ax.set_ylim(-0.04 * top, top * 1.03)
    ax.set_xlim(0, len(s))
    bare(ax)
    ax.text(0, 1.12, "|", color=ORANGE, transform=ax.transAxes, fontsize=8, fontweight="bold", va="center")
    ax.text(0.012, 1.12, f"textbook 3σ rule: {len(ev)} alarms", color=INK2, transform=ax.transAxes, va="center")
    ax.scatter([0.40], [1.12], marker="v", s=22, color=RED, transform=ax.transAxes, clip_on=False)
    ax.text(0.415, 1.12, f"Sentinel: {len(mine)} alarms, {int(mine.hit.sum())} correct", color=INK2,
            transform=ax.transAxes, va="center")
    save(f, "sigma_rule")


def types_strip():
    win = pd.read_csv(ROOT / "results" / "eda" / "anomaly_windows.csv")
    counts = win.type.replace({"volatility": "subtle"}).value_counts()
    short = {"new extreme": "biggest ever", "repeated extreme": "seen before", "level shift": "level shift",
             "subtle": "subtle"}
    f = plt.figure(figsize=(8.9 * CM, 2.1 * CM))
    for k, (_, typ, name, j, (lo, hi), _) in enumerate(TYPE_EXAMPLES):
        ax = f.add_axes([k * 0.255, 0.0, 0.235, 0.74])
        s = SERIES[name]
        a, b = window_indices(s.timestamps, s.windows)[j]
        hi = min(hi, len(s))
        ax.axvspan(a, b, color=WIN, lw=0)
        ax.plot(np.arange(lo, hi), s.values[lo:hi], color=BLUE, lw=0.5)
        ax.set_xlim(lo, hi)
        bare(ax)
        ax.text(0, 1.06, f"{counts[typ]}", transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK)
        ax.text(0.2 if counts[typ] >= 10 else 0.12, 1.06, short[typ], transform=ax.transAxes, color=INK2)
    save(f, "types_strip")


# ---------------------------------------------------------------- slide 5
def rarity_curves():
    """Level-expert tail on exchange-4_cpm at the moment it scores the spike at index 1276."""
    s = SERIES["realAdExchange/exchange-4_cpm_results.csv"]
    i0 = 1276
    fits = {}
    for key, p in (("old", SentinelParams(tail_shrink=0, relearn=False)), ("new", SentinelParams())):
        sen = Sentinel(p, 3600.0)
        zs = []
        for i in range(i0):
            sen.update(s.timestamps[i], float(s.values[i]))
            if i >= p.warmup:
                zs.append(abs(sen.last_zs[[e.name for e in sen.experts].index("level")]))
        j = [e.name for e in sen.experts].index("level")
        fits[key] = (sen.tails[j], abs(sen.experts[j].score(float(s.values[i0]))), p.risk / len(sen.experts), zs)
    z_spike, p_alarm, zs = fits["new"][1], fits["new"][2], np.sort(fits["new"][3])
    f, ax = fig(7.0, 4.3, l=1.15, b=0.85, t=0.1, r=0.15)
    surv = 1 - np.arange(len(zs)) / len(zs)
    keep = zs > 0.3
    ax.loglog(zs[keep], surv[keep], lw=0, marker="o", ms=1.5, color=GREY)
    ax.text(14, 0.07, "seen so far", color=MUTED, fontsize=6.3)
    grid = np.geomspace(1, 3000, 500)
    levels = {}
    for key, col, lw, ls in (("old", MUTED, 1.1, "--"), ("new", BLUE, 1.5, "-")):
        tl = fits[key][0]
        levels[key] = tl.level_for(p_alarm)
        ax.loglog(grid, [tl.tail_prob(g) for g in grid], color=col, lw=lw, ls=ls)
    ax.axhline(p_alarm, color=INK2, lw=0.6, ls=":")
    ax.text(1.1, p_alarm * 0.25, "alarm level", color=INK2, fontsize=6.3)
    ax.axvline(z_spike, color=RED, lw=0.8)
    ax.text(z_spike * 0.92, 0.6, f"the spike: {z_spike:.0f}", color=RED, fontsize=6.5, ha="right", va="top")
    to, tn = fits["old"][0], fits["new"][0]
    ax.text(levels["old"] * 0.75, 2.5e-5, f"old fit, γ = {to.gamma:.2f}\nalarm needs {levels['old']:.0f}",
            color=INK2, fontsize=6.3, ha="center")
    ax.text(1.4, 6e-5, f"with shrinkage, γ = {tn.gamma:.2f}\nalarm needs {levels['new']:.0f}",
            color=BLUE, fontsize=6.3)
    ax.set_ylim(1.5e-7, 1.5)
    ax.set_xlim(1, 3000)
    ax.set_xlabel("how far from normal (noise units)")
    ax.set_ylabel("chance of seeing it")
    save(f, "rarity_curves")
    print(f"  spike z={z_spike:.1f}  old gamma={to.gamma:.2f} needs {levels['old']:.0f}  "
          f"new gamma={tn.gamma:.2f} needs {levels['new']:.0f}  p_alarm={p_alarm:.1e}")


def relearn_strip(name="relearn_strip", w=6.9, h=1.9):
    s = SERIES["realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv"]
    f = plt.figure(figsize=(w * CM, h * CM))
    for k, (run, lab) in enumerate(((2, "before"), (6, "with re-learning"))):
        ax = f.add_axes([0, 0.52 - 0.52 * k, 1, 0.33])
        bands(ax, s)
        ax.plot(s.values, color=BLUE, lw=0.4)
        a = alarms(ax, s, run, size=16)
        n_fp = int((~a.hit).sum())
        ax.text(0, 1.1, f"{lab}: {n_fp} false alarm{'s' * (n_fp != 1)}", transform=ax.transAxes, color=INK2,
                fontsize=6.5)
        ax.set_xlim(0, len(s))
        bare(ax)
    save(f, name)


def score_chart():
    """A clean server: the signal on top, Sentinel's saved score below on a '1 in N' axis."""
    s = SERIES["realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"]
    p = pd.read_csv(ROOT / "predictions" / "predictions.csv", usecols=["series", "score"])
    sc = p.loc[p.series == s.name, "score"].to_numpy()
    f = plt.figure(figsize=(7.6 * CM, 4.0 * CM))
    a1 = f.add_axes([0.25, 0.68, 0.74, 0.28])
    a2 = f.add_axes([0.25, 0.05, 0.74, 0.56])
    bands(a1, s)
    a1.plot(s.values, color=BLUE, lw=0.4)
    alarms(a1, s, 6, hit_color=RED, size=18)
    a1.set_xlim(0, len(s))
    bare(a1)
    a1.text(-0.02, 0.5, "the signal", transform=a1.transAxes, ha="right", va="center", color=INK2)
    bands(a2, s)
    y = np.minimum(sc, 10)
    a2.plot(y, color=BLUE, lw=0.45)
    a2.fill_between(np.arange(len(s)), 5, y, where=y >= 5, color=ORANGE, alpha=0.45, lw=0)
    a2.axhline(5, color=ORANGE, lw=0.9, ls="--")
    a2.text(len(s), 5.35, "alarm line", color=ORANGE, ha="right", fontsize=6.5, fontweight="bold")
    a2.set_yticks([1, 3, 5, 7, 9], ["1 in 10", "1 in 1,000", "1 in 100,000", "1 in 10 million", "1 in a billion"])
    a2.get_yticklabels()[2].set_color(ORANGE)
    a2.set_ylim(0, 10.3)
    a2.set_xlim(0, len(s))
    a2.set_xticks([])
    a2.spines["bottom"].set_visible(False)
    save(f, "score_chart")


# ---------------------------------------------------------------- slide 6
def trajectory():
    log = pd.read_csv(ROOT / "runs" / "run_log.csv").set_index("run")
    p1 = [("v1", 0.343), ("v2", 0.473), ("v3", 0.500), ("v4", 0.535), ("v5", 0.551)]
    p2 = [(f"R{r}", log.f1[r], log.decision[r]) for r in range(2, len(log) + 1)]
    f, ax = fig(7.2, 4.4, l=0.85, b=0.45, t=0.65, r=0.1)
    xs1 = np.arange(len(p1))
    xs2 = np.arange(len(p1), len(p1) + len(p2))
    vals = [v for _, v in p1] + [v for _, v, _ in p2]
    best = np.maximum.accumulate(vals)
    ax.step(np.concatenate([xs1, xs2]), best, where="post", color=INK, lw=0.9, zorder=2)
    ax.scatter(xs1, [v for _, v in p1], s=16, color=GREY, zorder=3)
    for x, (_, v, d) in zip(xs2, p2):
        kept = d == "keep"
        ax.scatter([x], [v], s=18, color=BLUE if kept else "white", edgecolor=BLUE if kept else GREY,
                   lw=0.9, zorder=3)
    ax.axvline(len(p1) - 0.5, color=GRID, lw=0.8)
    ax.text(2, 1.05, "phase 1 · development\n124 logged runs", ha="center", va="bottom", color=INK2,
            fontsize=6.3, transform=ax.get_xaxis_transform())
    ax.text(len(p1) + (len(p2) - 1) / 2, 1.05, "phase 2 · 15-run budget\n7 used · filled = kept", ha="center",
            va="bottom", color=INK2, fontsize=6.3, transform=ax.get_xaxis_transform())
    i_best = len(p1) + int(np.argmax([v for _, v, _ in p2]))
    ax.text(i_best, best[-1] + 0.012, f"{best[-1]:.3f}", ha="center", va="bottom", color=BLUE, fontsize=7,
            fontweight="bold")
    ax.set_xticks(np.concatenate([xs1, xs2]), [n for n, _ in p1] + [n for n, _, _ in p2])
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_ylim(0.3, 0.64)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6])
    ax.set_ylabel("event-F1, all 58 series")
    ax.grid(True, axis="y")
    save(f, "trajectory")


# ---------------------------------------------------------------- slide 7
def delay_hist():
    d = pd.read_csv(ROOT / "results" / "detection_delay.csv").pts.clip(-62, 62)
    f, ax = fig(6.8, 2.2, l=0.1, b=0.8, t=0.05, r=0.25)
    ax.hist(d, bins=np.arange(-62.5, 63, 5), color=BLUE, edgecolor="white", lw=0.4)
    ax.axvline(0, color=INK2, lw=0.6, ls=":")
    ax.set_xticks([-60, -30, 0, 30, 60], ["≤ −60", "−30", "0", "30", "≥ 60"])
    ax.set_xlabel("first alarm − labelled anomaly point (steps)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    save(f, "delay_hist")


# ---------------------------------------------------------------- slide 8
BASELINE_NAMES = {"rolling_z": "Rolling z-score", "ocsvm": "One-Class SVM", "isolation_forest": "Isolation Forest",
                  "lof": "Local Outlier Factor", "autoencoder": "Autoencoder",
                  "temporal_clustering": "Temporal clustering", "ensemble_mean": "Ensemble of all 6"}


def f1_bars():
    """Every method from the problem statement's list, worst at the bottom, Sentinel on top."""
    b = pd.read_csv(ROOT / "results" / "baselines.csv").sort_values("f1")
    rows = [(BASELINE_NAMES[m], v) for m, v in zip(b.method, b.f1)]
    rows.append(("Sentinel (ours)", pd.read_json(ROOT / "predictions" / "metrics.json").loc["f1", "overall"]))
    top = len(rows) - 1
    f, ax = fig(6.0, 3.9, l=2.15, b=0.8, t=0.05, r=0.15)
    y = np.arange(len(rows))
    ax.barh(y, [v for _, v in rows], color=[GREY] * top + [BLUE], height=0.62)
    for yy, (_, v) in zip(y, rows):
        # round half up, like the tables (0.2345 -> 0.235)
        ax.text(v + 0.012, yy, f"{v + 1e-9:.3f}", va="center", fontsize=6.8, color=INK if yy == top else INK2,
                fontweight="bold" if yy == top else "normal")
    ax.set_yticks(y, [n for n, _ in rows])
    ax.get_yticklabels()[-1].set_color(INK)
    ax.set_xlim(0, 0.7)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("event-F1")
    ax.tick_params(axis="y", length=0)
    save(f, "f1_bars")


# ---------------------------------------------------------------- slide 10
def fp_example():
    s = SERIES["realTweets/Twitter_volume_IBM.csv"]
    f, ax = fig(7.2, 2.0, l=0, r=0, b=0, t=0.42)
    bands(ax, s)
    ax.plot(s.values, color=BLUE, lw=0.35)
    a = alarms(ax, s, 6, size=18)
    ax.set_xlim(0, len(s))
    bare(ax)
    for x0, col, lab in ((0.0, RED, f"false alarm ({int((~a.hit).sum())})"),
                         (0.3, INK2, f"correct alarm ({int(a.hit.sum())})")):
        ax.scatter([x0 + 0.01], [1.16], marker="v", s=18, color=col, transform=ax.transAxes, clip_on=False)
        ax.text(x0 + 0.03, 1.16, lab, color=INK2, transform=ax.transAxes, va="center")
    save(f, "fp_example")


def type_rates(name="type_rates", runs=(6,), w=7.2, h=2.4):
    t = pd.read_csv(ROOT / "results" / "eda" / "anomaly_windows.csv")[["series", "window", "type"]]
    t["type"] = t.type.replace({"volatility": "subtle"})
    order = ["level shift", "new extreme", "subtle", "repeated extreme"]
    label = {"level shift": "level shift", "new extreme": "biggest ever", "subtle": "subtle",
             "repeated extreme": "seen before"}
    f, ax = fig(w, h, l=1.55, b=0.75, t=0.05, r=0.4)
    y = np.arange(len(order))
    hgt = 0.62 / len(runs)
    for k, run in enumerate(runs):
        wdf = pd.read_csv(ROOT / "runs" / f"run_{run:02d}" / "windows.csv").merge(t, on=["series", "window"])
        r = wdf.groupby("type").hit.agg(["sum", "count"]).reindex(order)
        off = (k - (len(runs) - 1) / 2) * hgt
        ax.barh(y + off, r["sum"] / r["count"], height=hgt * 0.9, color=BLUE if run == runs[-1] else GREY,
                label=f"run {run}" if len(runs) > 1 else None)
        for yy, (h_, c_) in zip(y, r.itertuples(index=False)):
            ax.text(h_ / c_ + 0.015, yy + off, f"{h_}/{c_}", va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(y, [label[o] for o in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("labelled windows detected")
    if len(runs) > 1:
        ax.legend(loc="lower right")
    save(f, name)


# ---------------------------------------------------------------- backup
def shrink_backup():
    p1 = pd.read_csv(ROOT / "runs" / "run_01" / "per_series.csv")
    p6 = pd.read_csv(ROOT / "runs" / "run_06" / "per_series.csv")
    f1 = pd.read_csv(ROOT / "runs" / "run_log.csv").set_index("run").f1
    fg = plt.figure(figsize=(14.6 * CM, 4.6 * CM))
    a1 = fg.add_axes([0.07, 0.2, 0.38, 0.76])
    a2 = fg.add_axes([0.58, 0.2, 0.38, 0.76])
    bins = np.linspace(0, 0.6, 13)
    mid, w = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
    for k, (p, col, lab) in enumerate(((p1, GREY, "before (run 1)"), (p6, BLUE, "after (run 6)"))):
        hh, _ = np.histogram(p.gamma_level.dropna().clip(upper=0.599), bins=bins)
        a1.bar(mid + (k - 0.5) * w * 0.42, hh, width=w * 0.4, color=col, label=lab)
    a1.set_xlabel("tail shape γ of the level expert (higher = heavier)")
    a1.set_ylabel("series")
    a1.legend()
    for pts, col, lab in (([(0, f1[1]), (100, f1[2]), (400, f1[3])], GREY, "without re-learning"),
                          ([(100, f1[4]), (200, f1[6]), (400, f1[7])], BLUE, "with re-learning")):
        a2.plot(*zip(*pts), color=col, marker="o", ms=3.5, lw=1.2)
        for xx, yy in pts:
            a2.text(xx + 8, yy + 0.002, f"{yy:.3f}", fontsize=6.3, color=INK2)
    a2.set_xlabel("shrinkage strength n₀")
    a2.set_ylabel("event-F1")
    a2.set_ylim(0.535, 0.605)
    a2.set_xlim(-20, 470)
    a2.text(300, 0.5895, "with re-learning", color=BLUE, fontsize=6.5, ha="left")
    a2.text(120, 0.539, "without re-learning", color=MUTED, fontsize=6.5, ha="left")
    a2.grid(True, axis="y")
    save(fg, "b_shrinkage")


def fp_by_category():
    a = ALARMS[6]
    names = {"realTweets": "Tweets", "realKnownCause": "Known failures", "realAWSCloudwatch": "Servers (AWS)",
             "realTraffic": "Road traffic", "realAdExchange": "Ads (hourly)"}
    fp = a[~a.hit].category.map(names).value_counts()
    f, ax = fig(6.6, 3.0, l=2.1, b=0.8, t=0.05, r=0.3)
    ax.barh(np.arange(len(fp)), fp.values, color=RED, height=0.6)
    for yy, v in enumerate(fp.values):
        ax.text(v + 0.3, yy, str(v), va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(np.arange(len(fp)), fp.index)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("false alarms (39 in total)")
    save(f, "b_fp_by_category")


if __name__ == "__main__":
    hero()
    machine()
    sigma_rule()
    types_strip()
    rarity_curves()
    relearn_strip()
    trajectory()
    delay_hist()
    f1_bars()
    fp_example()
    type_rates()
    shrink_backup()
    relearn_strip("b_relearn", w=14.6, h=4.2)
    type_rates("b_type_rates", runs=(1, 6), w=7.2, h=3.4)
    fp_by_category()
