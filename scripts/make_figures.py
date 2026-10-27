"""Render every figure used in the slide deck into results/figures/.

Inputs: experiments.csv, results/baselines.csv, predictions/metrics.json and
fresh Sentinel runs on a few showcase series.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from plotstyle import (AQUA, BASE, BLUE, GREY, GRID, INK, INK2, MUTED, OUT, ORANGE, RED,
                       SURF, WIN, plt, save)
from sentinel.data import load_nab
from sentinel.engine import Sentinel, SentinelParams
from sentinel.metrics import window_indices

SERIES = {s.name: s for s in load_nab()}
LABELS = json.loads((ROOT / "data" / "nab" / "combined_labels.json").read_text())


def stream(name, params=None, collect=None):
    s = SERIES[name]
    step = float(np.median(np.diff(s.timestamps)) / np.timedelta64(1, "s"))
    sen = Sentinel(params or SentinelParams(), step_seconds=step)
    n = len(s)
    score, lab = np.zeros(n), np.zeros(n, bool)
    onsets, zs = [], []
    for i in range(n):
        o = sen.update(s.timestamps[i], float(s.values[i]))
        score[i], lab[i] = o.score, o.label == "ANOMALOUS"
        if o.evidence:
            onsets.append((i, o.evidence))
        if collect is not None and hasattr(sen, "last_zs") and i >= sen.p.warmup:
            names = [e.name for e in sen.experts]
            if collect in names:
                zs.append(abs(sen.last_zs[names.index(collect)]))
    return s, sen, score, lab, onsets, np.array(zs)


def plot_series(ax, s, onsets=None, title=None, warmup=None):
    x = np.arange(len(s))
    for a, b in window_indices(s.timestamps, s.windows):
        ax.axvspan(a, b, color=WIN, lw=0)
    ax.plot(x, s.values, color=BLUE, lw=0.9)
    if warmup:
        ax.axvspan(0, warmup, color=(0.5, 0.5, 0.5, 0.07), lw=0)
    if onsets:
        idx = [i for i, _ in onsets]
        ax.scatter(idx, s.values[idx], marker="v", s=70, color=RED, edgecolor=SURF, lw=1.5, zorder=5)
    ax.set_title(title or s.name, loc="left")
    ax.grid(False)
    ax.set_xlim(0, len(s))


# 1. EDA: one example per category --------------------------------------------
def fig_eda():
    picks = {"artificialWithAnomaly/art_daily_jumpsup.csv": "Artificial: daily pattern, then a jump",
             "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv": "Server CPU (AWS, every 5 min)",
             "realAdExchange/exchange-4_cpm_results.csv": "Online ad price (every hour)",
             "realKnownCause/nyc_taxi.csv": "New York taxi demand (every 30 min)",
             "realTraffic/occupancy_6005.csv": "Road sensor occupancy (traffic)",
             "realTweets/Twitter_volume_GOOG.csv": "Tweets mentioning Google (every 5 min)"}
    fig, axes = plt.subplots(3, 2, figsize=(13, 7.5))
    for ax, (n, title) in zip(axes.flat, picks.items()):
        plot_series(ax, SERIES[n], title=title)
        ax.tick_params(labelsize=9)
    fig.text(0.01, -0.01, "Grey bands = labelled anomaly windows (ground truth). x-axis = time step.",
             color=INK2, fontsize=10)
    fig.tight_layout()
    save(fig, "eda_examples.png")


# 2. tuning trajectory ---------------------------------------------------------
def fig_trajectory():
    log = pd.read_csv(ROOT / "experiments.csv")
    miles = [("v1\n5 experts +\nEVT threshold", 0.3433), ("v2\nfix tail &\nscale collapse", 0.4727),
             ("v3\nCUSUM evidence\naccumulation", 0.4997), ("v4\n+ novelty\n(matrix profile)", 0.5348),
             ("v5\nrisk 1e-5,\ndrop jump", 0.5514), ("v6 (run 4)\ntail shrinkage\n+ re-learning", 0.5838),
             ("v7 (run 6)\nshrinkage\nn0 = 200", 0.5941)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    xs = np.arange(len(miles))
    a1.bar(xs, [m[1] for m in miles], color=[GREY] * 5 + [BLUE] * 2, width=0.55)
    a1.text(2, 0.66, "phase 1: development", ha="center", color=INK2, fontsize=9.5)
    a1.text(5.5, 0.66, "phase 2: 15-run budget", ha="center", color=INK2, fontsize=9.5)
    for x, (_, v) in zip(xs, miles):
        a1.text(x, v + 0.01, f"{v:.3f}", ha="center", color=INK, fontsize=11, fontweight="semibold")
    a1.set_xticks(xs, [m[0] for m in miles], fontsize=9.5, color=INK2)
    a1.set_ylim(0, 0.7)
    a1.set_ylabel("event-F1 (all 58 series)")
    a1.set_title("Milestones: each change kept only if event-F1 improved", loc="left")
    # all runs: precision vs recall with iso-F1 curves
    r = np.linspace(0.05, 1, 200)
    for f in (0.3, 0.4, 0.5, 0.6):
        p = f * r / (2 * r - f)
        ok = (p > 0) & (p <= 1)
        a2.plot(r[ok], p[ok], color=GRID, lw=1)
        a2.text(0.93, f * 0.93 / (2 * 0.93 - f) + 0.01, f"F1={f}", color=MUTED, fontsize=8.5)
    a2.scatter(log.recall, log.precision, s=22, color=BLUE, alpha=0.45, edgecolor="none")
    best = log.loc[log.f1.idxmax()]
    a2.scatter([best.recall], [best.precision], s=90, color=BLUE, edgecolor=INK, lw=1.2, zorder=4)
    a2.annotate(f"final  F1={best.f1:.3f}", (best.recall, best.precision), xytext=(12, 8),
                textcoords="offset points", color=INK, fontsize=10)
    a2.set_xlim(0, 1); a2.set_ylim(0, 1)
    a2.set_xlabel("recall (windows detected)"); a2.set_ylabel("precision (alarms that hit)")
    a2.set_title(f"Phase 1: all {len(log)} development runs", loc="left")
    a2.grid(True, axis="both")
    fig.tight_layout()
    save(fig, "tuning_trajectory.png")


# 3. baselines -----------------------------------------------------------------
def fig_baselines():
    b = pd.read_csv(ROOT / "results" / "baselines.csv")
    m = json.loads((ROOT / "predictions" / "metrics.json").read_text())["overall"]
    names = {"rolling_z": "Rolling z-score", "isolation_forest": "Isolation Forest",
             "lof": "Local Outlier Factor", "ocsvm": "One-Class SVM"}
    rows = [(names[r.method], r.f1, r.precision, r.recall, r.fp) for r in b.itertuples()]
    rows.append(("Sentinel (ours)", m["f1"], m["precision"], m["recall"], m["fp"]))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[GREY] * (len(rows) - 1) + [BLUE], height=0.55)
    for yy, r in zip(y, rows):
        ax.text(r[1] + 0.008, yy, f"{r[1]:.3f}   (P {r[2]:.2f} · R {r[3]:.2f} · {r[4]} false alarms)",
                va="center", color=INK2, fontsize=10)
    ax.set_yticks(y, [r[0] for r in rows], color=INK)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("event-F1 on all 58 NAB series")
    ax.grid(True, axis="x"); ax.grid(False, axis="y")
    ax.set_title("Sentinel vs permitted baselines (baselines get their best threshold, oracle-tuned)",
                 loc="left", fontsize=11)
    save(fig, "baselines.png")


# 4. per-category --------------------------------------------------------------
def fig_categories():
    m = json.loads((ROOT / "predictions" / "metrics.json").read_text())["per_category"]
    cats = [c for c in m if m[c]["windows"]]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(cats))
    w = 0.26
    for k, (key, col, lab) in enumerate([("precision", BLUE, "Precision"), ("recall", ORANGE, "Recall"),
                                         ("f1", AQUA, "Event-F1")]):
        ax.bar(x + (k - 1) * w, [m[c][key] for c in cats], width=w - 0.03, color=col, label=lab)
    ax.set_xticks(x, [f"{c}\n({m[c]['windows']} windows)" for c in cats], fontsize=9.5, color=INK2)
    ax.set_ylim(0, 1.1)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13))
    noan = m.get("artificialNoAnomaly", {})
    ax.set_title(f"Per-category metrics  ·  artificialNoAnomaly (no anomalies): {noan.get('alarms', 0)} false alarms",
                 loc="left", y=1.12)
    save(fig, "per_category.png")


# 5. detection delay relative to the labelled anomaly point ---------------------
def fig_delay():
    ev = json.loads((ROOT / "predictions" / "events.json").read_text())
    rows = []
    for name, s in SERIES.items():
        widx = window_indices(s.timestamps, s.windows)
        lab_idx = [int(np.searchsorted(s.timestamps, np.datetime64(pd.Timestamp(t)))) for t in LABELS.get(name, [])]
        al = sorted(int(np.searchsorted(s.timestamps, np.datetime64(pd.Timestamp(e["timestamp"]))))
                    for e in ev if e["series"] == name)
        step_h = float(np.median(np.diff(s.timestamps)) / np.timedelta64(1, "h"))
        for a, b in widx:
            inside = [li for li in lab_idx if a <= li <= b]
            li = inside[0] if inside else (a + b) // 2
            hit = [i for i in al if a <= i <= b]
            if hit:
                rows.append({"pts": hit[0] - li, "hours": (hit[0] - li) * step_h, "frac": (hit[0] - a) / (b - a + 1)})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "detection_delay.csv", index=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    a1.hist(d.pts.clip(-300, 300), bins=30, color=BLUE, edgecolor=SURF, lw=1.5)
    a1.axvline(0, color=INK2, ls="--", lw=1.2)
    a1.text(4, a1.get_ylim()[1] * 0.9, "labelled anomaly onset", color=INK2, fontsize=9.5)
    a1.set_xlabel("first alarm − labelled anomaly point (time steps)")
    a1.set_ylabel("detected windows")
    a1.set_title(f"Detection delay · median {d.pts.median():+.0f} steps ({d.hours.median():+.1f} h)", loc="left")
    a2.hist(d.frac, bins=20, range=(0, 1), color=BLUE, edgecolor=SURF, lw=1.5)
    a2.set_xlabel("position of first alarm inside the window (0 = window start, 1 = end)")
    a2.set_title("Where inside the window we fire", loc="left")
    fig.tight_layout()
    save(fig, "detection_delay.png")
    return d


# 6. showcase detections + false positive analysis ------------------------------
def fig_detections():
    picks = ["realTweets/Twitter_volume_CVS.csv", "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
             "realTraffic/speed_6005.csv"]
    fig, axes = plt.subplots(len(picks), 1, figsize=(13, 8))
    for ax, n in zip(axes, picks):
        s, sen, _, _, on, _ = stream(n)
        plot_series(ax, s, on, title=n.replace("/", " / "), warmup=sen.p.warmup)
        for i, e in on:
            ax.annotate(e["expert"], (i, s.values[i]), xytext=(4, 8), textcoords="offset points",
                        color=INK2, fontsize=9)
    fig.text(0.01, -0.01, "▼ red = Sentinel alarm (labelled with the expert that fired) · grey band = labelled "
             "anomaly window · light shading at left = warm-up", color=INK2, fontsize=10)
    fig.tight_layout()
    save(fig, "detections.png")


def fig_false_positives():
    picks = ["realTweets/Twitter_volume_IBM.csv", "realKnownCause/rogue_agent_key_updown.csv"]
    fig, axes = plt.subplots(len(picks), 1, figsize=(13, 5.4))
    for ax, n in zip(axes, picks):
        s, sen, _, _, on, _ = stream(n)
        plot_series(ax, s, on, title=n.replace("/", " / "), warmup=sen.p.warmup)
        widx = window_indices(s.timestamps, s.windows)
        for i, e in on:
            fp = not any(a <= i <= b for a, b in widx)
            ax.annotate(("FP · " if fp else "") + e["expert"], (i, s.values[i]), xytext=(4, 8),
                        textcoords="offset points", color=RED if fp else INK2, fontsize=9)
    fig.tight_layout()
    save(fig, "false_positives.png")


# 7. ablation ------------------------------------------------------------------
def fig_ablation():
    log = pd.read_csv(ROOT / "experiments.csv")
    a = log[log.tag == "ablation-final"].copy()
    full = set("level+jump+seasonal+shift+volatility+novelty".split("+"))
    a["removed"] = a.experts.map(lambda e: ", ".join(sorted(full - set(e.split("+")))) or "none (all 6)")
    a = a.sort_values("f1")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    cols = [BLUE if r == "jump" else GREY for r in a.removed]
    ax.barh(a.removed, a.f1, color=cols, height=0.55)
    for y, v in enumerate(a.f1):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", color=INK2, fontsize=10)
    ax.set_xlim(0.45, 0.58)
    ax.set_xlabel("event-F1 with this expert removed")
    ax.grid(True, axis="x"); ax.grid(False, axis="y")
    ax.set_title("Ablation: removing novelty hurts most; removing jump helps (adopted)", loc="left", fontsize=11)
    save(fig, "ablation.png")


# 8. threshold design: EVT tail fit + adaptive threshold trace ------------------
def fig_threshold():
    n = "realTweets/Twitter_volume_CVS.csv"
    s, sen, score, lab, on, zs = stream(n, collect="novelty")
    names = [e.name for e in sen.experts]
    tl = sen.tails[names.index("novelty")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    z = np.sort(zs)
    surv = 1.0 - np.arange(len(z)) / len(z)
    keep = z > 0
    a1.loglog(z[keep], surv[keep], color=MUTED, lw=0, marker="o", ms=2.5, label="empirical tail of novelty score")
    lvl = tl.level_for(sen.p.risk / len(sen.experts))
    grid = np.geomspace(tl.t, lvl * 1.2, 200)
    a1.loglog(grid, [tl.tail_prob(g) for g in grid], color=BLUE, lw=2, label="fitted GPD tail (updated online)")
    a1.set_ylim(min(1e-6, tl.tail_prob(lvl) / 3), 1.5)
    a1.axvline(lvl, color=RED, ls="--", lw=1.3)
    a1.text(lvl * 0.93, 0.2, "alarm level\nfor target risk", color=RED, fontsize=9.5, ha="right")
    a1.axvline(tl.t, color=INK2, ls=":", lw=1)
    a1.text(tl.t * 1.08, 0.5, "peaks\nthreshold t", color=INK2, fontsize=9)
    a1.set_xlabel("novelty score (nearest-neighbour distance)")
    a1.set_ylabel("P(score > x)")
    a1.legend(loc="lower left", fontsize=9.5)
    a1.set_title("Extreme-value calibration: threshold from a risk, not a constant", loc="left", fontsize=11)
    a1.grid(True, axis="both")
    x = np.arange(len(s))
    a2.plot(x, np.minimum(score, 30), color=BLUE, lw=0.8)
    a2.axhline(sen.threshold, color=INK2, ls="--", lw=1.2)
    a2.text(len(s) * 0.01, sen.threshold + 0.6, f"alarm threshold  −log₁₀(risk) = {sen.threshold:.0f}",
            color=INK2, fontsize=9.5)
    for a, b in window_indices(s.timestamps, s.windows):
        a2.axvspan(a, b, color=WIN, lw=0)
    a2.set_ylim(0, 30)
    a2.set_xlabel("time step"); a2.set_ylabel("fused score (−log₁₀ p)")
    a2.set_title(f"Fused surprise on {n.split('/')[1]}", loc="left", fontsize=11)
    fig.tight_layout()
    save(fig, "threshold_design.png")


# 9. phase 2: the budgeted runs (from runs/) ------------------------------------
RUN_LABELS = {1: "baseline\n(v5)", 2: "tail\nshrink 100", 3: "tail\nshrink 400", 4: "shrink 100\n+ re-learn",
              5: "warm-up\n200", 6: "shrink 200\n+ re-learn", 7: "shrink 400\n+ re-learn"}
DECISION_COLORS = {"keep": BLUE, "drop": GREY, "reference": MUTED}


def fig_simple_rule():
    """Why simple thresholds fail: a textbook 3-sigma rule vs Sentinel on one series,
    plus the corpus-wide alarm count. Sentinel's alarms come from the saved run 6."""
    def three_sigma_events(x):
        x = pd.Series(x)
        m = x.rolling(288, min_periods=100).mean().shift(1)
        sd = x.rolling(288, min_periods=100).std().shift(1)
        idx = np.flatnonzero(((x - m).abs() > 3 * sd).to_numpy())
        if len(idx) == 0:
            return idx
        return idx[np.concatenate([[True], np.diff(idx) > 12])]   # crossings < 12 steps apart = one alarm
    total = sum(len(three_sigma_events(s.values)) for s in SERIES.values())
    al = pd.read_csv(ROOT / "runs" / "run_06" / "alarms.csv")
    n = "realTweets/Twitter_volume_FB.csv"
    s = SERIES[n]
    ev = three_sigma_events(s.values)
    mine = al[al.series == n]
    fig, ax = plt.subplots(figsize=(13, 4))
    plot_series(ax, s)
    top = s.values.max()
    ax.scatter(ev, np.full(len(ev), -0.06 * top), marker="|", s=120, color=ORANGE, lw=1.4, clip_on=False,
               label=f"simple 3-sigma rule: {len(ev)} alarms")
    ax.scatter(mine.idx, s.values[mine.idx], marker="v", s=90, color=RED, edgecolor=SURF, lw=1.5, zorder=5,
               label=f"Sentinel: {len(mine)} alarms, {int(mine.hit.sum())} correct")
    ax.set_ylim(-0.1 * top, top * 1.08)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(f"Tweets mentioning Facebook: {len(s.windows)} real anomalies (grey bands)", loc="left")
    fig.text(0.01, -0.02, f"3-sigma rule = alarm when a value is more than 3 standard deviations from the average "
             f"of the previous 288 points. Across all 58 series it raises {total:,} alarms for 116 real anomalies; "
             f"Sentinel raises {len(al)}.", color=INK2, fontsize=10)
    save(fig, "eda_simple_rule_fails.png")
    print("3-sigma alarms across corpus:", total)


def fig_budget():
    log = pd.read_csv(ROOT / "runs" / "run_log.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    x = log.run.to_numpy()
    a1.bar(x, log.f1, color=[DECISION_COLORS.get(d, GREY) for d in log.decision], width=0.6)
    a1.step(x, log.f1.cummax(), where="mid", color=INK, lw=1.4)
    for xx, v, d in zip(x, log.f1, log.decision):
        a1.text(xx, v + 0.006, f"{v:.3f}", ha="center", fontsize=10, color=INK,
                fontweight="semibold" if d == "keep" else "normal")
    a1.set_xticks(x, [RUN_LABELS.get(r, str(r)) for r in x], fontsize=9, color=INK2)
    a1.set_xlim(0.4, len(log) + 2.6)
    a1.set_ylim(0.45, 0.63)
    a1.axvspan(len(log) + 0.5, len(log) + 2.6, color=(0.5, 0.5, 0.5, 0.07), lw=0)
    a1.text(len(log) + 1.55, 0.47, f"runs {len(log) + 1}–15\nkept in\nreserve", ha="center", color=INK2, fontsize=10)
    a1.set_ylabel("event-F1 (all 58 series)")
    a1.set_title("15-run budget: one hypothesis per run, kept only if F1 rises (blue = kept)", loc="left")
    r = np.linspace(0.05, 1, 200)
    for f in (0.5, 0.55, 0.6):
        p = f * r / (2 * r - f)
        ok = (p > 0) & (p <= 1)
        a2.plot(r[ok], p[ok], color=GRID, lw=1)
        lx, ly = 0.665, f * 0.665 / (2 * 0.665 - f)   # label at the right edge, or at the bottom edge
        if ly < 0.51:
            ly = 0.51
            lx = f * ly / (2 * ly - f)
        a2.text(lx + 0.002, ly + 0.002, f"F1={f}", color=MUTED, fontsize=8.5)
    for _, row in log.iterrows():
        a2.scatter(row.recall, row.precision, s=70, color=DECISION_COLORS.get(row.decision, GREY),
                   edgecolor=SURF, lw=1.5, zorder=3)
        a2.annotate(str(row.run), (row.recall, row.precision), xytext=(6, 4), textcoords="offset points",
                    fontsize=9.5, color=INK2)
    a2.set_xlim(0.48, 0.68)
    a2.set_ylim(0.5, 0.68)
    a2.set_xlabel("recall (windows detected)")
    a2.set_ylabel("precision (alarms that hit)")
    a2.set_title("Precision vs recall of each run", loc="left")
    a2.grid(True, axis="both")
    fig.tight_layout()
    save(fig, "budget_runs.png")


def fig_shrinkage():
    p1 = pd.read_csv(ROOT / "runs" / "run_01" / "per_series.csv")
    p6 = pd.read_csv(ROOT / "runs" / "run_06" / "per_series.csv")
    f1 = pd.read_csv(ROOT / "runs" / "run_log.csv").set_index("run").f1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    bins = np.linspace(0, 0.6, 13)
    mid, w = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
    for k, (p, col, lab) in enumerate(((p1, GREY, "before (run 1)"), (p6, BLUE, "after (run 6)"))):
        h, _ = np.histogram(p.gamma_level.dropna().clip(upper=0.599), bins=bins)
        a1.bar(mid + (k - 0.5) * w * 0.42, h, width=w * 0.4, color=col, label=lab)
    a1.set_xlabel("tail shape γ of the level expert (0 = exponential, higher = heavier)")
    a1.set_ylabel("series")
    a1.legend()
    a1.set_title("Shrinkage stops spurious heavy tails", loc="left")
    without = [(0, f1[1]), (100, f1[2]), (400, f1[3])]
    with_rl = [(100, f1[4]), (200, f1[6]), (400, f1[7])]
    for pts, col, lab in ((without, GREY, "without re-learning"), (with_rl, BLUE, "with re-learning")):
        a2.plot(*zip(*pts), color=col, marker="o", ms=8, lw=2, label=lab)
        for xx, yy in pts:
            a2.text(xx + 8, yy + 0.002, f"{yy:.3f}", fontsize=9, color=INK2)
    a2.set_xlabel("shrinkage strength n0 (pseudo tail samples)")
    a2.set_ylabel("event-F1")
    a2.set_ylim(0.53, 0.605)
    a2.legend(loc="lower right")
    a2.set_title("Dose-response: best at n0 = 200 with re-learning", loc="left")
    fig.tight_layout()
    save(fig, "tail_shrinkage.png")


def fig_relearn():
    n = "realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv"
    s = SERIES[n]
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.2), sharex=True)
    for ax, run, title in ((axes[0], 2, "Before (run 2): re-fires each time the 300-point cooldown ends"),
                           (axes[1], 6, "After (run 6): the new regime is re-learned, not re-alarmed")):
        al = pd.read_csv(ROOT / "runs" / f"run_{run:02d}" / "alarms.csv")
        al = al[al.series == n]
        plot_series(ax, s, warmup=300)
        for _, a in al.iterrows():
            ax.scatter([a.idx], [s.values[a.idx]], marker="v", s=70, color=INK2 if a.hit else RED,
                       edgecolor=SURF, lw=1.5, zorder=5)
        n_fp = int((~al.hit).sum())
        ax.set_title(f"{title}  ·  {n_fp} false alarm{'s' * (n_fp != 1)}", loc="left", fontsize=11)
    fig.text(0.01, -0.01, f"{n}  ·  ▼ red = false alarm, ▼ dark = hit · grey band = labelled window",
             color=INK2, fontsize=10)
    fig.tight_layout()
    save(fig, "relearn_example.png")


def fig_error_types():
    t = pd.read_csv(ROOT / "results" / "eda" / "anomaly_windows.csv")[["series", "window", "type"]]
    order = ["level shift", "new extreme", "repeated extreme", "subtle"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    y = np.arange(len(order))
    for k, (run, col, lab) in enumerate(((1, GREY, "before (run 1)"), (6, BLUE, "after (run 6)"))):
        w = pd.read_csv(ROOT / "runs" / f"run_{run:02d}" / "windows.csv").merge(t, on=["series", "window"])
        r = w.groupby("type").hit.agg(["sum", "count"]).reindex(order)
        off = (k - 0.5) * 0.36
        ax.barh(y + off, r["sum"] / r["count"], height=0.32, color=col, label=lab)
        for yy, (h, c) in zip(y, r.itertuples(index=False)):
            ax.text(h / c + 0.01, yy + off, f"{h}/{c}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of labelled windows detected")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(loc="lower right")
    ax.set_title("Detection rate by anomaly type (types from the EDA)", loc="left")
    save(fig, "error_by_type.png")


if __name__ == "__main__":
    fig_eda()
    fig_trajectory()
    fig_baselines()
    fig_categories()
    d = fig_delay()
    print("delay summary:", d.describe().round(2).to_dict())
    fig_detections()
    fig_false_positives()
    fig_ablation()
    fig_threshold()
    fig_simple_rule()
    fig_budget()
    fig_shrinkage()
    fig_relearn()
    fig_error_types()
