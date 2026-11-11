"""Page registry. Pages are (re)built on every script run, as Streamlit expects;
`page(name)` returns an equivalent object for st.switch_page()."""
from __future__ import annotations

import streamlit as st

from ui.views import admin, analytics, assistant, candidate, dashboard, jobs, login, rankings, upload

_SPECS = {
    "login": (login.render, "Sign in", "🔐"),
    "dashboard": (dashboard.render, "Dashboard", "🏠"),
    "jobs": (jobs.render, "Jobs", "📋"),
    "upload": (upload.render, "Upload candidates", "📤"),
    "rankings": (rankings.render, "Rankings", "🏆"),
    "candidate": (candidate.render, "Candidate profile", "👤"),
    "assistant": (assistant.render, "AI assistant", "💬"),
    "analytics": (analytics.render, "Analytics", "📊"),
    "admin": (admin.render, "Admin", "🛠️"),
}


def page(name: str):
    fn, title, icon = _SPECS[name]
    return st.Page(fn, title=title, icon=icon, url_path=name, default=name in {"dashboard", "login"})
