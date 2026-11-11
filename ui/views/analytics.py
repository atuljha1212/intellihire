from __future__ import annotations

import pandas as pd
import streamlit as st

from intellihire.scoring.ranker import NotEnoughLabels
from intellihire.services.insights import job_analytics, latest_ranker, train_job_ranker
from ui import charts
from ui.common import current_user, display_name, job_picker, kpi


def render() -> None:
    st.title("Analytics")
    job = job_picker("analytics_job")
    if not job:
        return
    data = job_analytics(job["id"])
    if data["n"] == 0:
        st.info("No candidates yet.")
        return
    df = pd.DataFrame(data["rows"])
    df["display"] = [display_name(i, n) for i, n in zip(df["id"], df["name"])]

    cols = st.columns(5)
    with cols[0]:
        kpi("Applicants", data["n"])
    with cols[1]:
        kpi("Median score", f"{df['final'].median():.1f}")
    with cols[2]:
        kpi("Score ≥ 70", int((df["final"] >= 70).sum()))
    with cols[3]:
        kpi("With video", data["with_video"], f"{data['with_video'] / data['n']:.0%} of pool")
    with cols[4]:
        kpi("Flagged", data["flagged"], "possible injection")

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.score_histogram(df), config={"displayModeBar": False}, key="hist")
    c2.plotly_chart(charts.subscore_box(df), config={"displayModeBar": False}, key="box")
    c3, c4 = st.columns(2)
    if data["missing_counts"]:
        c3.plotly_chart(charts.missing_skills_bar(data["missing_counts"], data["n"]), config={"displayModeBar": False},
                        key="missing")
    else:
        c3.success("Every applicant has all required skills.")
    if data["coverage"]:
        c4.plotly_chart(charts.coverage_bar(data["coverage"], data["required"]), config={"displayModeBar": False},
                        key="coverage")
    c5, c6 = st.columns(2)
    c5.plotly_chart(charts.experience_scatter(df), config={"displayModeBar": False}, key="scatter")
    c6.plotly_chart(charts.status_funnel(data["status_counts"]), config={"displayModeBar": False}, key="funnel")
    st.plotly_chart(charts.processing_timeline(data["created"]), config={"displayModeBar": False}, key="timeline")

    st.divider()
    _ranker_section(job["id"], data["status_counts"])


def _ranker_section(job_id: int, status_counts: dict[str, int]) -> None:
    st.subheader("Learned ranker (Scikit-learn + XGBoost)")
    positives = sum(status_counts.get(s, 0) for s in ("shortlisted", "interview", "hired"))
    negatives = status_counts.get("rejected", 0)
    st.caption("Trains on your own decisions: shortlisted / interview / hired = positive, rejected = negative. "
               "Logistic Regression, Random Forest and XGBoost are cross-validated on the five sub-scores and the best "
               f"model (by ROC-AUC) is kept. Labelled so far: **{positives} positive**, **{negatives} negative** "
               "(need ≥ 6 with ≥ 2 of each).")
    if st.button("Train / retrain ranker", type="primary"):
        try:
            with st.spinner("Cross-validating models…"):
                report = train_job_ranker(current_user()["id"], job_id)
            st.success(f"Best model: **{report.best_model}** - ML scores added to the rankings.")
            for note in report.notes:
                st.caption(note)
        except NotEnoughLabels as exc:
            st.warning(str(exc))
    model = latest_ranker(job_id)
    if not model:
        return
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown(f"**Current model:** {model['model']} · trained on {model['n']} decisions · "
                    f"{pd.to_datetime(model['trained_at']):%d %b %Y %H:%M}")
        auc = pd.DataFrame([{"model": k, "CV ROC-AUC": v} for k, v in model["metrics"].items()])
        st.dataframe(auc, hide_index=True, column_config={"CV ROC-AUC": st.column_config.NumberColumn(format="%.3f")})
    with c2:
        st.plotly_chart(charts.importance_bar(model["importance"]), config={"displayModeBar": False}, key="importance")
