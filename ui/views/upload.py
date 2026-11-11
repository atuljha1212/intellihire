from __future__ import annotations

import pandas as pd
import streamlit as st

from intellihire.config import get_settings
from intellihire.ingestion.validation import file_stem_key, validate_batch_size
from intellihire.media.transcriber import transcriber_status
from intellihire.services.pipeline import add_resumes, attach_video, candidate_options
from ui.common import blind_mode, current_user, display_name, job_picker


def pair_videos(resume_names: list[str], video_names: list[str]) -> dict[str, str]:
    """Pair resumes with videos by exact stem key; empty keys never match and a video is used once."""
    by_stem: dict[str, str] = {}
    for v in video_names:
        key = file_stem_key(v)
        if key and key not in by_stem:
            by_stem[key] = v
    pairs: dict[str, str] = {}
    used: set[str] = set()
    for r in resume_names:
        key = file_stem_key(r)
        match = by_stem.get(key) if key else None
        if match and match not in used:
            pairs[r] = match
            used.add(match)
    return pairs


def render() -> None:
    st.title("Upload candidates")
    settings = get_settings()
    job = job_picker("upload_job", include_closed=False)
    if not job:
        return
    st.caption(f"Up to {settings.max_batch_files} resumes per batch (PDF, DOCX, TXT · max {settings.max_resume_mb:g} MB each). "
               f"Optional intro videos (MP4, MOV, AVI · max {settings.max_video_mb:g} MB, {settings.max_video_seconds} s) "
               "are paired with resumes by file name, e.g. `jane_doe.pdf` + `jane_doe.mp4`.")

    nonce = st.session_state.setdefault("upload_nonce", 0)
    resumes = st.file_uploader("Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True, key=f"res_{nonce}")
    videos = st.file_uploader("Video introductions (optional)", type=["mp4", "mov", "avi"], accept_multiple_files=True,
                              key=f"vid_{nonce}")

    pairs: dict[str, str] = {}
    if resumes and videos:
        pairs = pair_videos([r.name for r in resumes], [v.name for v in videos])
        unmatched = sorted({v.name for v in videos} - set(pairs.values()))
        shown = [f"video #{i}" for i, _ in enumerate(unmatched, start=1)] if blind_mode() else unmatched
        st.markdown(f"**Pairing:** {len(pairs)} video(s) matched to resumes."
                    + (f" Unpaired videos: {', '.join(shown)} - attach them below after processing." if unmatched else ""))

    if resumes:
        problem = validate_batch_size(len(resumes))
        if problem:
            st.error(problem)
            return
        if videos and "unavailable" in transcriber_status():
            st.warning(f"Speech-to-text: {transcriber_status()}. Videos will be skipped.")
        if st.button(f"Process {len(resumes)} resume(s)" + (f" + {len(pairs)} video(s)" if pairs else ""), type="primary"):
            _process(job["id"], resumes, videos or [], pairs)

    if st.session_state.get("last_upload_results"):
        _show_results(st.session_state["last_upload_results"])

    st.divider()
    st.subheader("Attach a video to an existing candidate")
    options = candidate_options(job["id"])
    if not options:
        st.caption("No candidates yet.")
        return
    labels = {cid: (f"{lbl.split('  ')[0]}  {display_name(cid, '')}".strip() if blind_mode() else lbl) for cid, lbl in options}
    cid = st.selectbox("Candidate", [o[0] for o in options], format_func=lambda i: labels[i], key="attach_cid")
    single = st.file_uploader("Video", type=["mp4", "mov", "avi"], key=f"single_vid_{nonce}")
    if single is not None and st.button("Transcribe & re-score"):
        with st.spinner("Extracting audio and transcribing with Whisper…"):
            result = attach_video(current_user()["id"], cid, single.name, single.getvalue())
        (st.success if result.status != "error" else st.error)(
            f"{'video' if blind_mode() else result.filename}: {result.message}")


def _process(job_id: int, resumes, videos, pairs: dict[str, str]) -> None:
    user_id = current_user()["id"]
    bar = st.progress(0.0, text="Starting…")

    def progress(i: int, n: int, name: str) -> None:
        bar.progress((i - 1) / n, text=f"Parsing resume {i}/{n}: {name}")

    files = [(r.name, r.getvalue()) for r in resumes]
    results = add_resumes(user_id, job_id, files, progress)
    rows = [{"file": r.filename, "candidate": r.name, "candidate_id": r.candidate_id, "status": r.status, "details": r.message,
             "flags": r.flags, "warnings": "; ".join(r.warnings)} for r in results]

    video_by_name = {v.name: v for v in videos}
    to_transcribe = [(r, video_by_name[pairs[r.filename]]) for r in results
                     if r.filename in pairs and r.candidate_id and r.status in {"added", "duplicate"}]
    for i, (res, video) in enumerate(to_transcribe, start=1):
        bar.progress(i / max(len(to_transcribe), 1), text=f"Transcribing video {i}/{len(to_transcribe)}: {video.name}")
        vres = attach_video(user_id, res.candidate_id, video.name, video.getvalue())
        rows.append({"file": video.name, "candidate": res.name, "candidate_id": res.candidate_id, "status": vres.status, "details": vres.message,
                     "flags": 0, "warnings": ""})
    bar.progress(1.0, text="Done")
    st.session_state["last_upload_results"] = rows
    st.session_state["upload_nonce"] += 1  # clears the uploaders
    st.rerun()


def _show_results(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if blind_mode():
        df["candidate"] = [display_name(cid, "") if cid else "-" for cid in df.get("candidate_id", [None] * len(df))]
        df["file"] = [f"file #{i}" for i in range(1, len(df) + 1)]
    df = df.drop(columns=["candidate_id"], errors="ignore")
    added = int((df["status"] == "added").sum())
    dupes = int((df["status"] == "duplicate").sum())
    errors = int((df["status"] == "error").sum())
    vids = int((df["status"] == "video added").sum())
    st.success(f"{added} added · {vids} videos transcribed · {dupes} duplicates skipped · {errors} rejected")
    if int(df["flags"].sum()):
        st.warning("Some resumes contain hidden or instruction-like text (possible prompt injection). "
                   "They are scored on visible text only - review them on the candidate page.")
    st.dataframe(df, hide_index=True, column_config={"flags": st.column_config.NumberColumn("⚠ flags")})
    if st.button("Clear results"):
        st.session_state.pop("last_upload_results", None)
        st.rerun()
