"""Shared Streamlit helpers: session/auth state, job picker, styling, formatting."""
from __future__ import annotations

import streamlit as st

from intellihire.services.pipeline import list_jobs
from intellihire.services.users import get_active_user

CSS = """
<style>
:root { --ih-accent: #0f766e; --ih-muted: #6b6a63; --ih-line: #e4e2d9; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.ih-kpi { border: 1px solid var(--ih-line); border-radius: 12px; padding: 14px 16px; background: #fff; }
.ih-kpi .label { font-size: 0.78rem; color: var(--ih-muted); text-transform: uppercase; letter-spacing: .05em; }
.ih-kpi .value { font-size: 1.7rem; font-weight: 650; font-variant-numeric: tabular-nums; margin-top: 2px; }
.ih-kpi .sub { font-size: 0.8rem; color: var(--ih-muted); }
.ih-chip { display: inline-block; padding: 2px 9px; margin: 2px 4px 2px 0; border-radius: 999px; font-size: 0.8rem;
           border: 1px solid var(--ih-line); background: #fff; white-space: nowrap; }
.ih-chip.ok { background: #e7f5ef; border-color: #b9e2cf; color: #0b5e43; }
.ih-chip.miss { background: #fdecec; border-color: #f5c2c2; color: #9b1c1c; }
.ih-chip.pref { background: #f3f0e6; border-color: #e1dbc6; color: #6b5a1f; }
.ih-chip.video { background: #ecf2fd; border-color: #c4d5f5; color: #1d4691; }
.ih-card { border: 1px solid var(--ih-line); border-radius: 12px; padding: 14px 16px; background: #fff; height: 100%; }
.ih-card .rank { font-size: .78rem; color: var(--ih-muted); }
.ih-card .name { font-weight: 650; font-size: 1.02rem; margin: 2px 0 6px; }
.ih-card .score { font-size: 1.5rem; font-weight: 700; color: var(--ih-accent); font-variant-numeric: tabular-nums; }
.ih-muted { color: var(--ih-muted); font-size: 0.88rem; }
.ih-quote { border-left: 3px solid var(--ih-line); padding: 2px 10px; color: #3d3c37; font-size: 0.88rem; margin: 4px 0; }
@media (prefers-color-scheme: dark) {
  .ih-kpi, .ih-card, .ih-chip { background: transparent; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def current_user() -> dict | None:
    """The signed-in user, re-read from the database on every call so a role change,
    deactivation or deletion takes effect on the next rerun instead of at logout."""
    user = st.session_state.get("user")
    if not user:
        return None
    fresh = get_active_user(user.get("id"))
    if fresh is None:
        st.session_state.pop("user", None)
        return None
    st.session_state["user"] = fresh
    return fresh


def is_admin() -> bool:
    user = current_user()
    return bool(user and user["role"] == "admin")


def blind_mode() -> bool:
    return bool(st.session_state.get("blind_mode", False))


def display_name(cid: int, name: str) -> str:
    return f"Candidate C{cid}" if blind_mode() else name


def kpi(label: str, value, sub: str = "") -> None:
    st.markdown(
        f'<div class="ih-kpi"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def chips(items: list[str], kind: str = "") -> str:
    from html import escape

    return "".join(f'<span class="ih-chip {kind}">{escape(str(i))}</span>' for i in items) or '<span class="ih-muted">none</span>'


def job_picker(key: str = "job_picker", label: str = "Job", include_closed: bool = True) -> dict | None:
    """Select box bound to a shared `job_id` so every page opens on the same job."""
    jobs = list_jobs(include_closed)
    if not jobs:
        if not include_closed and list_jobs(True):
            st.info("All jobs are closed - reopen one or create a new job on the **Jobs** page first.")
        else:
            st.info("No jobs yet - create one on the **Jobs** page first.")
        return None
    ids = [j["id"] for j in jobs]
    current = st.session_state.get("job_id")
    index = ids.index(current) if current in ids else 0
    labels = {j["id"]: f"{j['title']}  ·  {j['candidates']} candidates" + ("  (closed)" if j["status"] != "open" else "")
              for j in jobs}
    chosen = st.selectbox(label, ids, index=index, format_func=lambda i: labels[i], key=key)
    st.session_state["job_id"] = chosen
    return next(j for j in jobs if j["id"] == chosen)


def score_color(value: float | None) -> str:
    if value is None:
        return "#9b9a92"
    if value >= 75:
        return "#0f766e"
    if value >= 55:
        return "#b7791f"
    return "#b42318"


def fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"
