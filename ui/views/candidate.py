from __future__ import annotations

from html import escape

import streamlit as st

from intellihire.agent.agent import RecruiterAgent
from intellihire.services.insights import ranking_rows
from intellihire.services.pipeline import (
    STATUSES,
    attach_video,
    candidate_options,
    delete_candidate,
    get_candidate,
    save_notes,
    save_summary,
    set_status,
)
from ui import charts
from ui.common import blind_mode, chips, current_user, display_name, fmt_pct, job_picker


def _redact(c: dict, text: str) -> str:
    """Blind mode: remove name parts (case-insensitive) and contact details from free text."""
    if not blind_mode() or not text:
        return text
    from intellihire.rag.guard import redact_pii
    from intellihire.rag.retriever import redact_name

    return redact_name(redact_pii(text), c.get("name") or "")


def _flash() -> None:
    msg = st.session_state.pop("_flash_candidate", None)
    if msg:
        st.toast(msg)


def render() -> None:
    _flash()
    st.title("Candidate profile")
    job = job_picker("cand_job")
    if not job:
        return
    options = candidate_options(job["id"])
    if not options:
        st.info("No candidates for this job yet.")
        return
    ids = [o[0] for o in options]
    labels = {cid: (f"{lbl.split('  ')[0]}  {display_name(cid, '')}".strip() if blind_mode() else lbl) for cid, lbl in options}
    preselect = st.session_state.get("candidate_id")
    cid = st.selectbox("Candidate", ids, index=ids.index(preselect) if preselect in ids else 0,
                       format_func=lambda i: labels[i])
    st.session_state["candidate_id"] = cid
    c = get_candidate(cid)
    if c is None:
        st.error("Candidate not found.")
        return

    _header(c)
    tabs = st.tabs(["📈 Score explanation", "🧩 Skills & experience", "🎥 Video", "🤖 AI insights", "🛡️ Security", "📄 Resume"])
    with tabs[0]:
        _score_tab(c, job["id"])
    with tabs[1]:
        _skills_tab(c)
    with tabs[2]:
        _video_tab(c)
    with tabs[3]:
        _ai_tab(c)
    with tabs[4]:
        _security_tab(c)
    with tabs[5]:
        _resume_tab(c)


def _header(c: dict) -> None:
    sc = c["score"] or {}
    name = display_name(c["id"], c["name"])
    left, mid, right = st.columns([2.2, 1.2, 1.6])
    with left:
        st.subheader(name)
        if not blind_mode():
            contact = " · ".join(x for x in [c["email"], c["phone"]] if x)
            st.markdown(f"<span class='ih-muted'>{escape(contact) or 'no contact details found'}</span>",
                        unsafe_allow_html=True)
            for link in c["links"][:3]:
                st.markdown(f"<span class='ih-muted'>{escape(link)}</span>", unsafe_allow_html=True)
        st.markdown(f"Rank **#{sc.get('rank', '—')}** for *{escape(c['job_title'])}* · "
                    f"{c['experience_years']:g} yrs experience"
                    + ("" if blind_mode() else f" · file `{escape(c['resume_filename'])}`"))
        if c["injection_flags"]:
            st.warning(f"⚠ {len(c['injection_flags'])} security flag(s) - see the Security tab.")
    with mid:
        st.plotly_chart(charts.gauge(sc.get("final", 0.0)), config={"displayModeBar": False}, key="gauge")
        if sc.get("ml") is not None:
            st.caption(f"Learned-ranker score: **{sc['ml']:.0f}**/100")
    with right:
        status = st.selectbox("Status", STATUSES, index=STATUSES.index(c["status"]), key=f"status_{c['id']}")
        notes = st.text_area("Recruiter notes", c["notes"], height=110, key=f"notes_{c['id']}")
        if st.button("Save", type="primary", key=f"save_{c['id']}"):
            if status != c["status"]:
                set_status(current_user()["id"], [c["id"]], status)
            save_notes(current_user()["id"], c["id"], notes)
            st.session_state["_flash_candidate"] = "Saved"
            st.rerun()


def _score_tab(c: dict, job_id: int) -> None:
    sc = c["score"]
    if not sc:
        st.info("Not scored yet.")
        return
    d = sc["details"]
    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(charts.contribution_waterfall(d["contributions"], d["weights_used"]),
                        config={"displayModeBar": False}, key="waterfall")
    with right:
        top = [r for r in ranking_rows(job_id)[:5]]
        bench = {k: (sum((r[k] or 0) for r in top) / len(top)) for k in
                 ["semantic", "skill", "experience", "video", "communication"]} if top else None
        mine = {k: (None if sc[k] is None else sc[k] * 100) for k in
                ["semantic", "skill", "experience", "video", "communication"]}
        st.plotly_chart(charts.radar(mine, bench, display_name(c["id"], c["name"])),
                        config={"displayModeBar": False}, key="radar")
    st.markdown("**Why this score**")
    for reason in d.get("reasons", []):
        st.markdown(f"- {reason}")
    cols = st.columns(5)
    labels = [("Semantic", "semantic"), ("Skills", "skill"), ("Experience", "experience"), ("Video", "video"),
              ("Communication", "communication")]
    for col, (label, key) in zip(cols, labels):
        weight = d["weights_used"].get(key)
        col.metric(label, fmt_pct(None if sc[key] is None else sc[key] * 100),
                   help=f"Weight used: {weight:.0%}" if weight is not None else "Not applied (input missing)")
    st.caption(f"Raw cosine similarity resume↔JD: {d['raw_cosine']:.3f}"
               + (f" · transcript↔responsibilities: {d['raw_video_cosine']:.3f}" if d.get("raw_video_cosine") is not None else ""))


def _skills_tab(c: dict) -> None:
    d = (c["score"] or {}).get("details", {})
    st.markdown("**Required skills present**<br>" + chips(d.get("matched_required", []), "ok"), unsafe_allow_html=True)
    st.markdown("**Required skills missing**<br>" + chips(d.get("missing_required", []), "miss"), unsafe_allow_html=True)
    st.markdown("**Preferred skills present**<br>" + chips(d.get("matched_preferred", []), "pref"), unsafe_allow_html=True)
    if d.get("video_only_skills"):
        st.markdown("**Mentioned only in the video (half credit)**<br>" + chips(d["video_only_skills"], "video"),
                    unsafe_allow_html=True)
    other = sorted(set(c["skills"]) - set(d.get("matched_required", [])) - set(d.get("matched_preferred", [])))
    st.markdown("**Other skills detected**<br>" + chips(other), unsafe_allow_html=True)

    with st.expander("Where each skill was found (evidence)"):
        for skill, lines in sorted(c["skills"].items()):
            quotes = "".join(f"<div class='ih-quote'>{escape(_redact(c, line))}</div>" for line in lines[:2])
            st.markdown(f"**{escape(skill)}**{quotes}", unsafe_allow_html=True)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Experience:** {c['experience_years']:g} years · basis: *{c['experience_method']}*")
        for line in c["experience_evidence"][:6]:
            st.markdown(f"<div class='ih-quote'>{escape(_redact(c, line))}</div>", unsafe_allow_html=True)
    with right:
        st.markdown("**Education**")
        for line in c["education"] or ["—"]:
            st.markdown(f"- {_redact(c, line)}")
        st.markdown("**Certifications**")
        for line in c["certifications"] or ["—"]:
            st.markdown(f"- {_redact(c, line)}")
    if c["key_phrases"]:
        st.markdown("**Key phrases** " + chips(c["key_phrases"]), unsafe_allow_html=True)


def _video_tab(c: dict) -> None:
    v = c["video"]
    st.caption("Only the audio is used: Whisper transcribes speech and the scoring uses the transcript. "
               "Video frames (face, appearance, background) are never analysed.")
    if v is None:
        st.info("No video introduction for this candidate.")
    else:
        cols = st.columns(5)
        cols[0].metric("Duration", f"{v['duration']:.0f}s")
        cols[1].metric("Words", v["word_count"])
        cols[2].metric("Pace", f"{v['wpm']:.0f} wpm")
        cols[3].metric("Diversity (MATTR)", f"{v['mattr']:.2f}")
        cols[4].metric("Fillers", f"{v['filler_rate'] * 100:.1f}%")
        for note in (v["comm_details"] or {}).get("notes", []):
            st.markdown(f"- {note}")
        if v["video_skills"]:
            st.markdown("**Skills mentioned:** " + chips(v["video_skills"], "video"), unsafe_allow_html=True)
        st.markdown(f"**Transcript** <span class='ih-muted'>({v['engine']}, language: {v['language'] or '?'})</span>",
                    unsafe_allow_html=True)
        query = st.text_input("Search the transcript", key=f"tsearch_{c['id']}")
        for seg in v["segments"]:
            if query and query.lower() not in _redact(c, seg["text"]).lower():
                continue
            st.markdown(f"`{seg['start']:>6.1f}s` {escape(_redact(c, seg['text']))}", unsafe_allow_html=True)
    st.divider()
    upload = st.file_uploader("Upload / replace video introduction", type=["mp4", "mov", "avi"], key=f"vid_{c['id']}")
    if upload is not None and st.button("Transcribe & re-score", key=f"vbtn_{c['id']}"):
        with st.spinner("Extracting audio and transcribing…"):
            result = attach_video(current_user()["id"], c["id"], upload.name, upload.getvalue())
        if result.status == "error":
            st.error(result.message)
        else:
            st.session_state["_flash_candidate"] = result.message
            st.rerun()


def _ai_tab(c: dict) -> None:
    agent = RecruiterAgent(c["job_id"], blind=blind_mode())
    st.caption("Generated from the candidate's own documents with cited evidence [E#]. "
               "Candidate text is sanitised before it reaches the model.")
    left, right = st.columns(2)
    if left.button("Generate evidence-based summary", type="primary", key=f"sum_{c['id']}"):
        with st.spinner("Retrieving evidence and summarising…"):
            result = agent.summarize(c["id"])
        save_summary(c["id"], result.text)
        st.session_state[f"summary_evidence_{c['id']}"] = result.evidence
        c["ai_summary"] = result.text
    focus = right.text_input("Interview focus (optional)", key=f"focus_{c['id']}")
    if right.button("Generate interview questions", key=f"iq_{c['id']}"):
        with st.spinner("Planning questions…"):
            result = agent.interview_questions(c["id"], focus or None)
        st.session_state[f"questions_{c['id']}"] = (result.text, result.evidence)

    if c["ai_summary"]:
        st.markdown("#### Summary")
        st.markdown(_redact(c, c["ai_summary"]))
        _evidence_list(c, st.session_state.get(f"summary_evidence_{c['id']}", []))
    if st.session_state.get(f"questions_{c['id']}"):
        text, evidence = st.session_state[f"questions_{c['id']}"]
        st.markdown("#### Interview questions")
        st.markdown(_redact(c, text))
        _evidence_list(c, evidence)


def _evidence_list(c: dict, evidence: list[dict]) -> None:
    if not evidence:
        return
    with st.expander(f"Evidence used ({len(evidence)})"):
        for e in evidence:
            where = e["source"] + (f", {e['start_sec']:.0f}s" if e.get("start_sec") is not None else "")
            st.markdown(f"**[{e['ref']}]** <span class='ih-muted'>{where}</span><div class='ih-quote'>{escape(_redact(c, e['text'][:500]))}</div>",
                        unsafe_allow_html=True)


def _security_tab(c: dict) -> None:
    st.markdown("IntelliHire scores **visible text only**. Hidden text (white, microscopic or off-page) is quarantined, "
                "and instruction-like sentences are stripped before any text reaches the LLM.")
    if not c["injection_flags"]:
        st.success("No hidden text or injection patterns detected.")
    for f in c["injection_flags"]:
        st.markdown(f"- **{f['kind']}** in *{f['location']}*")
        st.markdown(f"<div class='ih-quote'>{escape(_redact(c, f['snippet']))}</div>", unsafe_allow_html=True)
    if c["hidden_text"]:
        with st.expander("Hidden text found in the file"):
            st.text(_redact(c, c["hidden_text"][:3000]))
    if c["warnings"]:
        st.markdown("**Parser warnings:** " + _redact(c, "; ".join(c["warnings"])))
    st.caption(f"Parsed with: {c['parser']}")
    with st.popover("Delete candidate"):
        st.warning("Removes the candidate, transcript, scores and vectors.")
        if st.button("Confirm delete", type="primary", key=f"del_{c['id']}"):
            delete_candidate(current_user()["id"], c["id"])
            st.session_state.pop("candidate_id", None)
            st.rerun()


def _resume_tab(c: dict) -> None:
    if blind_mode():
        st.info("Resume text is hidden in blind mode.")
        return
    text = c["resume_text"]
    st.text_area("Extracted resume text (visible text only)", text, height=500, disabled=True)
