"""Plain-English explanations for alarms."""
from __future__ import annotations


def _f(v: float) -> str:
    a = abs(v)
    if a >= 1e5 or (0 < a < 1e-2):
        return f"{v:.3g}"
    return f"{v:,.2f}"


def explain(expert: str, x: float, z: float, info: dict) -> str:
    side = "above" if z > 0 else "below"
    if expert == "level":
        lo, hi = info["band"]
        return (f"Value {_f(x)} is {abs(z):.1f} scales {side} the learned baseline "
                f"{_f(info['expected'])} (normal band {_f(lo)} to {_f(hi)}).")
    if expert == "jump":
        ratio = abs(info["step"]) / max(info["typical_step"], 1e-12)
        kind = "rise" if info["step"] > 0 else "drop"
        return (f"Sudden {kind} of {_f(abs(info['step']))} in one step, "
                f"{ratio:.1f}x the typical step size.")
    if expert == "seasonal":
        return (f"Value {_f(x)} breaks the repeating pattern: the same point one period "
                f"earlier ({info['period_pts']} steps) was {_f(info['same_time_last_period'])}.")
    if expert == "shift":
        return (f"Sustained level shift: recent mean {_f(info['recent_mean'])} vs baseline "
                f"{_f(info['baseline_mean'])} ({abs(z):.1f} scales {side}).")
    if expert == "volatility":
        ratio = info["recent_volatility"] / max(info["baseline_volatility"], 1e-12)
        if ratio >= 1000:
            return "Noise burst: step-to-step volatility jumped over 1,000x from a nearly flat baseline."
        if ratio >= 1:
            return f"Noise burst: step-to-step volatility is {ratio:.1f}x its baseline."
        return f"Signal went unusually flat: volatility is {ratio:.2f}x its baseline."
    if expert == "novelty":
        age = info.get("nn_age")
        when = f"{age} steps ago" if age is not None else "in memory"
        return (f"The last {info['window']} points form a pattern not seen before: the closest "
                f"past match ({when}) differs by {_f(info['nn_distance'])} per point.")
    return f"Score exceeded the adaptive threshold (expert {expert}, z={z:.1f})."
