from __future__ import annotations

import pandas as pd
import streamlit as st

from intellihire.services.insights import dashboard_stats
from intellihire.services.pipeline import list_jobs
from intellihire.services.system import component_statuses
from ui.common import current_user, kpi


@st.cache_data(ttl=60, show_spinner=False)
def _degraded_components() -> list[tuple[str, str]]:
    return [(c.name, c.detail) for c in component_statuses() if c.degraded or not c.ok]


def render() -> None:
    user = current_user()
    st.title(f"Welcome back, {user['full_name'].split()[0]}")
    st.caption("IntelliHire ranks candidates transparently and explains every score. Final decisions stay with you.")

    stats = dashboard_stats()
    cols = st.columns(5)
    with cols[0]:
        kpi("Open jobs", stats["jobs_open"], f"{stats['jobs_total']} total")
    with cols[1]:
        kpi("Candidates", stats["candidates"], "resumes processed")
    with cols[2]:
        kpi("Video intros", stats["videos"], "transcribed")
    with cols[3]:
        kpi("Shortlisted", stats["shortlisted"], "incl. interview / hired")
    with cols[4]:
        kpi("Avg. score", stats["avg_score"], f"{stats['flagged']} flagged resumes")

    degraded = _degraded_components()
    if degraded:
        with st.expander(f"⚠️ {len(degraded)} component(s) running in fallback mode", expanded=False):
            for name, detail in degraded:
                st.markdown(f"- **{name}:** {detail}")
            st.caption("The app works in fallback mode, but install the full stack (see README) for best accuracy.")

    st.subheader("Jobs")
    jobs = list_jobs()
    if not jobs:
        st.info("Start by creating a job on the **Jobs** page, then upload resumes on **Upload candidates**.")
        _how_it_works()
        return
    df = pd.DataFrame(jobs)[["title", "department", "status", "candidates", "videos", "created_at"]]
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%d %b %Y")
    st.dataframe(df, hide_index=True, column_config={
        "title": "Job", "department": "Department", "status": "Status", "candidates": "Candidates",
        "videos": "Videos", "created_at": "Created"})
    _how_it_works()


def _how_it_works() -> None:
    with st.expander("How IntelliHire scores candidates"):
        st.markdown(
            "1. **Ingestion & validation** - PDF/DOCX/TXT resumes and MP4/MOV/AVI videos are checked (type, size, "
            "signature) before parsing.\n"
            "2. **Extraction** - PyMuPDF / python-docx extract visible text; hidden text is quarantined. Whisper "
            "transcribes speech only - video frames are never analysed.\n"
            "3. **NLP** - spaCy matches 140+ skills with aliases, parses experience and education.\n"
            "4. **Scoring** - `final = Σ wᵢ·Sᵢ` over semantic match (MiniLM embeddings), skill validation, "
            "experience fit, video relevance and communication. Missing optional inputs are re-weighted, not penalised.\n"
            "5. **Explain & assist** - every score has a breakdown; the RAG assistant answers with cited evidence; "
            "a learned ranker (Scikit-learn/XGBoost) can model your own shortlisting decisions."
        )
