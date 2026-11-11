"""Regression tests for UI QA findings (T18, T19, T49-T53)."""
from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("plotly")

from pathlib import Path  # noqa: E402

from intellihire.db.session import init_db  # noqa: E402
from intellihire.ingestion.documents import extract_document  # noqa: E402
from intellihire.nlp.jd_parser import parse_job_description  # noqa: E402
from intellihire.services import pipeline, users  # noqa: E402
from tests.test_ui import page  # noqa: E402
from ui.views.upload import pair_videos  # noqa: E402


def test_pairing_exact_only():
    assert pair_videos(["ram.pdf", "ramesh.pdf"], ["ramesh.mp4"]) == {"ramesh.pdf": "ramesh.mp4"}


def test_pairing_empty_keys_and_reuse():
    assert pair_videos(["resume.pdf"], ["video.mp4"]) == {}
    assert pair_videos(["a_resume.pdf", "a_cv.pdf"], ["a.mp4"]) == {"a_resume.pdf": "a.mp4"}


@pytest.fixture
def ui_job(samples):
    init_db()
    admin = users.authenticate("admin", "admin123")
    jd = extract_document("jd.txt", (samples / "jobs" / "ml_engineer_jd.txt").read_bytes()).text
    profile = parse_job_description(jd)
    job_id = pipeline.save_job(admin["id"], profile, title="QA UI job", required=profile.required_skills,
                               preferred=profile.preferred_skills, min_years=profile.min_years,
                               max_years=profile.max_years)
    files = [(p.name, p.read_bytes()) for p in sorted((samples / "resumes" / "ml_engineer").iterdir())][:3]
    pipeline.add_resumes(admin["id"], job_id, files)
    yield admin, job_id
    pipeline.delete_job(admin["id"], job_id)


def _texts(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [t.value for t in at.text_area]
    return "\n".join(str(p) for p in parts)


def test_blind_candidate_page_hides_filename_and_resume(ui_job):
    admin, job_id = ui_job
    at = page("candidate", admin, job_id)
    at.session_state["blind_mode"] = True
    at.run()
    assert not at.exception
    text = _texts(at)
    assert ".pdf" not in text and ".docx" not in text and ".txt" not in text
    assert not at.text_area or all("Extracted resume text" not in t.label for t in at.text_area if t.label)


def test_upload_blind_and_admin_caption(ui_job):
    admin, job_id = ui_job
    at = page("upload", admin, job_id)
    at.session_state["blind_mode"] = True
    at.run()
    assert not at.exception
    sel = [s for s in at.selectbox if s.label == "Candidate"]
    assert sel and all(o.startswith("#") is False or "Candidate C" in o for o in sel[0].options)
    at = page("admin", admin, job_id).run()
    assert any("Quick start (local)" in c.value for c in at.caption)
