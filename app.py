"""Sentinel live demo: replay a NAB series as a stream and watch alarms fire.

    streamlit run app.py
"""
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sentinel.data import load_nab
from sentinel.engine import Sentinel, SentinelParams
from sentinel.metrics import evaluate_series, window_indices

# reference palette (dataviz skill): one series hue, status-critical for alarms
BLUE, RED = "#2a78d6", "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE, WINDOW = "#e1e0d9", "#c3c2b7", "#fcfcfb", "rgba(137,135,129,0.18)"
EXPERT_LABEL = {"level": "Level", "jump": "Jump", "seasonal": "Seasonal", "shift": "Level shift",
                "volatility": "Volatility", "novelty": "Novelty"}

st.set_page_config(page_title="Sentinel", page_icon="🛰️", layout="wide")


@st.cache_data
def corpus():
    return {s.name: s for s in load_nab()}


def step_seconds(ts):
    return float(np.median(np.diff(ts)) / np.timedelta64(1, "s"))


def reset(name, params):
    s = corpus()[name]
    st.session_state.update(
        name=name, params=params, idx=0, playing=False,
        sen=Sentinel(params, step_seconds=step_seconds(s.timestamps)),
        scores=np.zeros(len(s)), labels=np.zeros(len(s), dtype=bool),
        events=[], lat_ns=0, lat_n=0,
    )


def advance(n):
    s = corpus()[st.session_state.name]
    sen, i0 = st.session_state.sen, st.session_state.idx
    i1 = min(i0 + n, len(s))
    sc, lab = st.session_state.scores, st.session_state.labels
    t0 = time.perf_counter_ns()
    for i in range(i0, i1):
        o = sen.update(s.timestamps[i], float(s.values[i]))
        sc[i], lab[i] = o.score, o.label == "ANOMALOUS"
        if o.evidence is not None:
            o.evidence["index"] = i
            st.session_state.events.append(o.evidence)
    if i1 > sen.p.warmup and i1 > i0:
        st.session_state.lat_ns += time.perf_counter_ns() - t0
        st.session_state.lat_n += i1 - i0
    st.session_state.idx = i1
    if i1 >= len(s):
        st.session_state.playing = False


def base_layout(fig, height, ytitle):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE, font=dict(family="system-ui, Segoe UI, sans-serif", color=INK2, size=12),
        hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=INK2)),
    )
    fig.update_xaxes(showgrid=False, linecolor=BASE, tickfont=dict(color=MUTED))
    fig.update_yaxes(title_text=ytitle, gridcolor=GRID, zeroline=False, linecolor=BASE,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK2))
    return fig


def signal_chart(s, lo, hi, windows):
    ts, v = s.timestamps[lo:hi], s.values[lo:hi]
    fig = go.Figure()
    shown = False
    # pd.Timestamp, not numpy datetime64: plotly turns datetime64 list items into integers,
    # which switches the whole x-axis to numbers and hides the signal
    stamp = lambda i: pd.Timestamp(s.timestamps[i])
    for a, b in windows:
        if b < lo or a >= hi:
            continue
        a2, b2 = max(a, lo), min(b, hi - 1)
        fig.add_trace(go.Scatter(
            x=[stamp(a2), stamp(a2), stamp(b2), stamp(b2)],
            y=[v.min(), v.max(), v.max(), v.min()], fill="toself", fillcolor=WINDOW,
            line=dict(width=0), mode="lines", hoverinfo="skip",
            name="Labeled anomaly window", legendgroup="win", showlegend=not shown))
        shown = True
    fig.add_trace(go.Scatter(x=ts, y=v, mode="lines", line=dict(color=BLUE, width=1.5),
                             name="Signal", hovertemplate="%{y:,.2f}<extra>value</extra>"))
    ev = [e for e in st.session_state.events if lo <= e["index"] < hi]
    if ev:
        fig.add_trace(go.Scatter(
            x=[stamp(e["index"]) for e in ev], y=[e["value"] for e in ev],
            mode="markers", name="Sentinel alarm",
            marker=dict(symbol="triangle-down", size=13, color=RED, line=dict(color=SURFACE, width=2)),
            customdata=[EXPERT_LABEL.get(e["expert"], e["expert"]) for e in ev],
            hovertemplate="ALARM · %{customdata}<extra></extra>"))
    return base_layout(fig, 300, "value")


def score_chart(s, lo, hi, thr):
    ts, sc = s.timestamps[lo:hi], st.session_state.scores[lo:hi]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=sc, mode="lines", line=dict(color=BLUE, width=1.5),
                             name="Anomaly score (−log₁₀ p)",
                             hovertemplate="%{y:.2f}<extra>score</extra>"))
    fig.add_hline(y=thr, line=dict(color=INK2, width=1.5, dash="dash"),
                  annotation_text=f"adaptive threshold (risk {st.session_state.params.risk:g})",
                  annotation_position="top left", annotation_font_color=INK2)
    fig.update_yaxes(range=[0, max(thr * 2, min(float(sc.max()) * 1.1 if len(sc) else 1, 40))])
    return base_layout(fig, 220, "score")


def expert_bars(ev):
    names = list(ev["experts"])
    vals = [min(-np.log10(max(ev["experts"][n]["p"], 1e-300)), 40) for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=[EXPERT_LABEL.get(n, n) for n in names], orientation="h",
        marker=dict(color=BLUE, cornerradius=4), hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        text=[f"{x:.1f}" for x in vals], textposition="outside", textfont=dict(color=INK2)))
    fig.add_vline(x=ev["threshold"], line=dict(color=INK2, dash="dash", width=1))
    fig.update_layout(height=190, margin=dict(l=4, r=24, t=4, b=4), showlegend=False,
                      plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
                      font=dict(family="system-ui, Segoe UI, sans-serif", color=INK2, size=11))
    fig.update_xaxes(title_text="surprise per expert (−log₁₀ p)", gridcolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK2))
    return fig


# ---------------------------------------------------------------- sidebar
data = corpus()
# deep link: ?series=realTweets/Twitter_volume_CVS.csv&end=1 opens that series already replayed
link = st.query_params.get("series")
link = link if link in data else None
with st.sidebar:
    st.title("🛰️ Sentinel")
    st.caption("Adaptive streaming anomaly detection · Utkrishta 2026, Challenge 2")
    cats = sorted({n.split("/")[0] for n in data})
    cat = st.selectbox("Category", cats, index=cats.index(link.split("/")[0] if link else "realTweets"))
    names = [n for n in data if n.startswith(cat + "/")]
    default = link or "realTweets/Twitter_volume_CVS.csv"
    name = st.selectbox("Series", names, index=names.index(default) if default in names else 0,
                        format_func=lambda n: n.split("/")[1])
    with st.expander("Detector settings"):
        risk = st.select_slider("Alarm risk (tail probability)", [1e-3, 1e-4, 1e-5, 1e-6], value=1e-5,
                                format_func=lambda r: f"{r:g}")
        cooldown = st.slider("Cooldown between alarms (points)", 50, 1000, 300, 50)
        gate = st.slider("Novelty gate (0 = off)", 0.0, 3.0, 0.0, 0.5)
    params = SentinelParams(risk=risk, cooldown=cooldown, novelty_gate=gate)
    speed = st.select_slider("Replay speed (points per tick)", [10, 25, 50, 100, 250, 500, 1000], value=100)
    widths = [500, 1000, 2000, 5000, "all"]
    qview = st.query_params.get("view")     # deep link, e.g. &view=all
    qview = "all" if qview == "all" else int(qview) if qview and qview.isdigit() and int(qview) in widths else 2000
    view = st.select_slider("Live view width (points)", widths, value=qview)

if st.session_state.get("name") != name or st.session_state.get("params") != params:
    reset(name, params)
if link and st.query_params.get("end") == "1" and not st.session_state.get("deep_linked"):
    st.session_state.deep_linked = True
    advance(10 ** 9)

c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Play" if not st.session_state.playing else "⏸ Pause", use_container_width=True):
    st.session_state.playing = not st.session_state.playing
if c2.button("⏭ End", use_container_width=True):
    advance(10 ** 9)
if c3.button("↺ Reset", use_container_width=True):
    reset(name, params)


# ---------------------------------------------------------------- main view
@st.fragment(run_every=0.4)
def live():
    if st.session_state.playing:
        advance(speed)
    s = data[st.session_state.name]
    idx = st.session_state.idx
    sen = st.session_state.sen
    widx = window_indices(s.timestamps, s.windows)
    started = [(a, b) for a, b in widx if a < idx]

    k = st.columns(5)
    k[0].metric(f"Points streamed (of {len(s):,})", f"{idx:,}",
                help="Warm-up (first %d points) learns the baseline; no alarms there." % sen.p.warmup)
    k[1].metric("Alarms raised", len(st.session_state.events))
    if idx:
        r = evaluate_series(s.timestamps[:idx], st.session_state.labels[:idx],
                            [s.windows[i] for i, (a, _) in enumerate(widx) if a < idx])
        k[2].metric("Windows detected", f"{r.n_detected} / {len(started)}")
        k[3].metric("Precision so far", f"{r.precision:.0%}" if r.n_pred else "–")
    else:
        k[2].metric("Windows detected", "–")
        k[3].metric("Precision so far", "–")
    lat = st.session_state.lat_ns / max(st.session_state.lat_n, 1) / 1e3
    k[4].metric("Latency per point", f"{lat:.0f} µs" if st.session_state.lat_n else "–")

    if idx <= sen.p.warmup:
        st.info(f"Learning the baseline: {idx}/{sen.p.warmup} warm-up points. Press ▶ Play.")
    hi = max(idx, 2)
    lo = 0 if view == "all" else max(0, hi - int(view))
    st.plotly_chart(signal_chart(s, lo, hi, widx), use_container_width=True, key="sig")
    st.plotly_chart(score_chart(s, lo, hi, sen.threshold), use_container_width=True, key="score")

    st.subheader("Alarm evidence")
    if not st.session_state.events:
        st.caption("No alarms yet. Every alarm shows which expert fired, how surprising it was, and why.")
    for j, ev in enumerate(reversed(st.session_state.events[-4:])):
        inside = any(a <= ev["index"] <= b for a, b in widx)
        with st.container(border=True):
            left, right = st.columns([3, 2])
            with left:
                verdict = "✅ inside a labeled anomaly window" if inside else "⚠️ outside labeled windows (false positive)"
                st.markdown(f"**🚨 ANOMALOUS** · `{ev['timestamp'][:19]}` · expert **{EXPERT_LABEL.get(ev['expert'], ev['expert'])}**")
                st.markdown(ev["explanation"])
                st.caption(f"score {ev['score']:.1f} vs threshold {ev['threshold']:.1f} · {verdict}")
                with st.expander("Raw evidence JSON"):
                    st.code(json.dumps({k2: v for k2, v in ev.items() if k2 != "index"}, indent=1, default=str),
                            language="json")
            with right:
                st.plotly_chart(expert_bars(ev), use_container_width=True, key=f"bars{j}")


live()
