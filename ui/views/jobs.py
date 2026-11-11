from __future__ import annotations

import streamlit as st

from intellihire.config import DEFAULT_WEIGHTS
from intellihire.nlp.jd_parser import parse_job_description
from intellihire.nlp.skills import canonical_skill_names
from intellihire.scoring.features import FEATURE_LABELS
from intellihire.services.analysis import AnalysisError, read_job_document
from intellihire.services.pipeline import delete_job, get_job, save_job, update_job
from ui.common import chips, current_user, job_picker


def render() -> None:
    st.title("Jobs")
    st.caption("Create a job from a description; IntelliHire extracts the reference profile, which you can correct.")
    tab_new, tab_manage = st.tabs(["➕ Create job", "✏️ Manage jobs"])
    with tab_new:
        _create()
    with tab_manage:
        _manage()


def _parse_custom(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()][:40]


def _weights_editor(prefix: str, current: dict[str, float]) -> dict[str, float]:
    st.markdown("**Scoring weights** - relative importance of each feature (normalised automatically)")
    cols = st.columns(5)
    weights = {}
    for col, key in zip(cols, DEFAULT_WEIGHTS):
        with col:
            weights[key] = st.slider(FEATURE_LABELS[key], 0.0, 1.0, float(current.get(key, DEFAULT_WEIGHTS[key])), 0.05,
                                     key=f"{prefix}_w_{key}")
    total = sum(weights.values())
    if total == 0:
        st.warning("At least one weight must be above zero - defaults will be used.")
        return dict(DEFAULT_WEIGHTS)
    st.caption("Effective: " + " · ".join(f"{FEATURE_LABELS[k]} {v / total:.0%}" for k, v in weights.items()))
    return weights


def _experience_inputs(prefix: str, min_y: float | None, max_y: float | None) -> tuple[float | None, float | None]:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        specified = st.checkbox("Experience required", value=min_y is not None, key=f"{prefix}_exp_on")
    if not specified:
        return None, None
    with c2:
        lo = st.number_input("Min years", 0.0, 40.0, float(min_y or 0.0), 0.5, key=f"{prefix}_min")
    with c3:
        hi = st.number_input("Max years (0 = no max)", 0.0, 50.0, float(max_y or 0.0), 0.5, key=f"{prefix}_max")
    return lo, (hi if hi and hi >= lo else None)


def _create() -> None:
    with st.form("jd_input"):
        c1, c2, c3 = st.columns([2, 1, 1])
        title = c1.text_input("Job title (leave blank to detect)")
        department = c2.text_input("Department")
        location = c3.text_input("Location")
        source = st.radio("Job description", ["Paste text", "Upload file (PDF / DOCX / TXT)"], horizontal=True)
        text = st.text_area("Paste the job description", height=240) if source == "Paste text" else ""
        upload = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"]) if source != "Paste text" else None
        analyse = st.form_submit_button("Analyse description", type="primary")

    if analyse:
        try:
            raw = read_job_document(upload.name, upload.getvalue()) if upload is not None else text
            if len(raw.strip()) < 30:
                raise AnalysisError("Please provide a job description of at least 30 characters.")
            st.session_state["jd_draft"] = {
                "profile": parse_job_description(raw, title),
                "department": department, "location": location,
            }
        except AnalysisError as exc:
            st.error(str(exc))

    draft = st.session_state.get("jd_draft")
    if not draft:
        return
    profile = draft["profile"]
    st.divider()
    st.subheader("Review the extracted profile")
    st.caption("Correct anything the parser missed - these lists drive skill scoring and explanations.")
    title_final = st.text_input("Title", profile.title, key="new_title")
    custom_raw = st.text_input("Custom skills not in the taxonomy (comma separated)", key="new_custom",
                               help="E.g. an internal tool name. They become selectable below and are searched for in resumes.")
    custom = _parse_custom(custom_raw)
    options = sorted(set(canonical_skill_names()) | set(profile.required_skills) | set(profile.preferred_skills) | set(custom))
    required = st.multiselect("Required skills", options, default=profile.required_skills, key="new_required")
    preferred = st.multiselect("Preferred skills", [o for o in options if o not in required],
                               default=[p for p in profile.preferred_skills if p not in required], key="new_preferred")
    min_y, max_y = _experience_inputs("new", profile.min_years, profile.max_years)
    weights = _weights_editor("new", DEFAULT_WEIGHTS)
    with st.expander("Detected responsibilities (used for video relevance)"):
        st.text(profile.responsibilities[:3000])
    if profile.key_phrases:
        st.markdown("**Key phrases:** " + chips(profile.key_phrases), unsafe_allow_html=True)

    c1, c2 = st.columns([1, 5])
    if c1.button("Save job", type="primary"):
        if not required and not preferred:
            st.warning("Select at least one skill so the skill score can be computed.")
            return
        with st.spinner("Embedding the job description…"):
            job_id = save_job(current_user()["id"], profile, title=title_final, department=draft["department"],
                              location=draft["location"], required=required, preferred=preferred,
                              min_years=min_y, max_years=max_y, weights=weights, custom_skills=custom)
        st.session_state["job_id"] = job_id
        st.session_state.pop("jd_draft", None)
        st.success("Job saved. Next: upload resumes on the **Upload candidates** page.")
    if c2.button("Discard draft"):
        st.session_state.pop("jd_draft", None)
        st.rerun()


def _manage() -> None:
    job_meta = job_picker("manage_job")
    if not job_meta:
        return
    job = get_job(job_meta["id"])
    jid = job["id"]
    st.markdown(f"**{job['title']}** · {job['department'] or 'no department'} · {job['location'] or 'no location'} · "
                f"status **{job['status']}** · {job_meta['candidates']} candidates")
    with st.expander("Job description"):
        st.text(job["description"][:6000])

    custom = _parse_custom(st.text_input("Custom skills (comma separated)", ", ".join(job["custom_skills"]),
                                         key=f"m_custom_{jid}"))
    options = sorted(set(canonical_skill_names()) | set(job["required_skills"]) | set(job["preferred_skills"]) | set(custom))
    required = st.multiselect("Required skills", options, default=job["required_skills"], key=f"m_req_{jid}")
    preferred = st.multiselect("Preferred skills", [o for o in options if o not in required],
                               default=[p for p in job["preferred_skills"] if p not in required], key=f"m_pref_{jid}")
    min_y, max_y = _experience_inputs(f"m_{jid}", job["min_years"], job["max_years"])
    weights = _weights_editor(f"m_{jid}", job["weights"])
    status = st.radio("Status", ["open", "closed"], index=0 if job["status"] == "open" else 1, horizontal=True,
                      key=f"m_status_{jid}")

    c1, c2, _ = st.columns([1.3, 1, 3])
    if c1.button("Save & re-score", type="primary", key=f"m_save_{jid}"):
        with st.spinner("Re-scoring all candidates…"):
            update_job(current_user()["id"], jid, required_skills=required, preferred_skills=preferred,
                       custom_skills=custom, min_years=min_y, max_years=max_y, weights=weights, status=status)
        st.success("Saved - rankings updated.")
    with c2.popover("Delete job"):
        st.warning("Deletes the job, its candidates, transcripts and vectors. This cannot be undone.")
        if st.button("Confirm delete", type="primary", key=f"m_del_{jid}"):
            delete_job(current_user()["id"], jid)
            st.session_state.pop("job_id", None)
            st.rerun()
