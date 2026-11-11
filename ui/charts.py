"""Plotly figures for the dashboard (analytics sub-module)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from intellihire.scoring.features import FEATURE_LABELS

ACCENT = "#0f766e"
PALETTE = ["#0f766e", "#2563eb", "#b7791f", "#7c3aed", "#db2777", "#475569"]
STATUS_COLORS = {"new": "#94a3b8", "shortlisted": "#2563eb", "interview": "#7c3aed", "hired": "#0f766e", "rejected": "#b42318"}


def _layout(fig: go.Figure, height: int = 320, **kw) -> go.Figure:
    layout = dict(
        height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(size=12), legend=dict(orientation="h", y=-0.15),
    )
    layout.update(kw)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "", "font": {"size": 40}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": ACCENT, "thickness": 0.3},
               "steps": [{"range": [0, 55], "color": "rgba(180,35,24,0.10)"},
                         {"range": [55, 75], "color": "rgba(183,121,31,0.12)"},
                         {"range": [75, 100], "color": "rgba(15,118,110,0.12)"}]},
    ))
    return _layout(fig, 220, margin=dict(l=20, r=20, t=20, b=0))


def contribution_waterfall(contributions: dict[str, float], weights: dict[str, float]) -> go.Figure:
    keys = [k for k in FEATURE_LABELS if k in contributions]
    labels = [f"{FEATURE_LABELS[k]}<br><sub>weight {weights.get(k, 0):.0%}</sub>" for k in keys]
    values = [contributions[k] for k in keys]
    fig = go.Figure(go.Waterfall(
        x=labels + ["Final score"], y=values + [0], measure=["relative"] * len(values) + ["total"],
        text=[f"+{v:.1f}" for v in values] + [f"{sum(values):.1f}"], textposition="outside",
        connector={"line": {"color": "rgba(128,128,128,0.4)"}},
        increasing={"marker": {"color": ACCENT}}, totals={"marker": {"color": "#1f2937"}},
    ))
    fig.update_yaxes(range=[0, 110], title="points (out of 100)")
    return _layout(fig, 340, title="How the final score is built", showlegend=False)


def radar(candidate: dict[str, float | None], benchmark: dict[str, float] | None, name: str) -> go.Figure:
    keys = [k for k in FEATURE_LABELS if candidate.get(k) is not None]
    if len(keys) < 3:
        keys = list(FEATURE_LABELS)
    theta = [FEATURE_LABELS[k] for k in keys] + [FEATURE_LABELS[keys[0]]]
    fig = go.Figure()
    if benchmark:
        r = [benchmark.get(k) or 0 for k in keys]
        fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=theta, name="Top-5 average", line=dict(color="#94a3b8", dash="dot")))
    r = [candidate.get(k) or 0 for k in keys]
    fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=theta, name=name, fill="toself",
                                  line=dict(color=ACCENT), fillcolor="rgba(15,118,110,0.18)"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False), bgcolor="rgba(0,0,0,0)"))
    return _layout(fig, 340, title="Sub-scores vs. top-5 average")


def score_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(x=df["final"], nbinsx=20, marker_color=ACCENT, opacity=0.85))
    fig.update_xaxes(title="final score", range=[0, 100])
    fig.update_yaxes(title="candidates")
    return _layout(fig, title="Score distribution", bargap=0.05)


def subscore_box(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, key in enumerate(["semantic", "skill", "experience", "video", "communication"]):
        series = df[key].dropna()
        if len(series):
            fig.add_trace(go.Box(y=series, name=FEATURE_LABELS[key], marker_color=PALETTE[i % len(PALETTE)], boxmean=True))
    fig.update_yaxes(range=[0, 105], title="score")
    return _layout(fig, title="Sub-score spread across the pool", showlegend=False)


def missing_skills_bar(counts: dict[str, int], total: int) -> go.Figure:
    items = sorted(counts.items(), key=lambda kv: kv[1])[-15:]
    fig = go.Figure(go.Bar(
        x=[v / total * 100 if total else 0 for _, v in items], y=[k for k, _ in items], orientation="h",
        marker_color="#b42318", text=[f"{v} ({v / total:.0%})" if total else str(v) for _, v in items],
        textposition="outside",
    ))
    fig.update_xaxes(title="% of applicants missing the skill", range=[0, 115])
    return _layout(fig, max(260, 28 * len(items) + 80), title="Missing required skills across the pool", showlegend=False)


def coverage_bar(coverage: dict[str, float], required: list[str]) -> go.Figure:
    items = sorted(coverage.items(), key=lambda kv: kv[1])
    colors = [ACCENT if k in required else "#b7791f" for k, _ in items]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker_color=colors,
                           text=[f"{v:.0f}%" for _, v in items], textposition="outside"))
    fig.update_xaxes(title="% of applicants with the skill (teal = required, amber = preferred)", range=[0, 115])
    return _layout(fig, max(260, 26 * len(items) + 80), title="Skill coverage in the applicant pool", showlegend=False)


def experience_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for status, group in df.groupby("status"):
        fig.add_trace(go.Scatter(
            x=group["years"], y=group["final"], mode="markers", name=status,
            marker=dict(size=10, color=STATUS_COLORS.get(status, "#64748b"), line=dict(width=1, color="white")),
            text=group["display"], hovertemplate="%{text}<br>%{x} yrs · score %{y:.1f}<extra></extra>",
        ))
    fig.update_xaxes(title="years of experience")
    fig.update_yaxes(title="final score", range=[0, 105])
    return _layout(fig, title="Experience vs. score")


def status_funnel(counts: dict[str, int]) -> go.Figure:
    order = ["new", "shortlisted", "interview", "hired", "rejected"]
    labels = [s for s in order if counts.get(s)]
    fig = go.Figure(go.Bar(x=labels, y=[counts[s] for s in labels], marker_color=[STATUS_COLORS[s] for s in labels],
                           text=[counts[s] for s in labels], textposition="outside"))
    fig.update_yaxes(title="candidates")
    return _layout(fig, title="Pipeline status", showlegend=False)


def processing_timeline(created: list) -> go.Figure:
    if not created:
        return _layout(go.Figure(), title="Processing queue")
    s = pd.Series(1, index=pd.to_datetime(created)).sort_index()
    hourly = s.resample("h").sum()
    hourly = hourly[hourly > 0]
    fig = go.Figure(go.Bar(x=hourly.index, y=hourly.values, marker_color="#2563eb"))
    fig.update_yaxes(title="resumes processed")
    return _layout(fig, title="Resumes processed over time", showlegend=False)


def importance_bar(importance: dict[str, float]) -> go.Figure:
    items = sorted(importance.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v * 100 for _, v in items], y=[FEATURE_LABELS.get(k, k.replace("_", " ")) for k, _ in items],
                           orientation="h", marker_color="#7c3aed", text=[f"{v:.0%}" for _, v in items],
                           textposition="outside"))
    fig.update_xaxes(title="relative importance (%)", range=[0, 115])
    return _layout(fig, 300, title="What drives your shortlisting decisions", showlegend=False)
