"""Headless UI tests with Streamlit's AppTest: every page renders without
exceptions and the main interactive flows work end to end."""
from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("plotly")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from intellihire.db.session import init_db  # noqa: E402
from intellihire.ingestion.documents import extract_document  # noqa: E402
from intellihire.nlp.jd_parser import parse_job_description  # noqa: E402
from intellihire.services import insights, pipeline, users  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

VIEWS = ["dashboard", "jobs", "upload", "rankings", "candidate", "assistant", "analytics", "admin"]


@pytest.fixture(scope="module")
def ui_job(samples):
    init_db()
    admin = users.authenticate("admin", "admin123")
    jd = extract_document("jd.txt", (samples / "jobs" / "ml_engineer_jd.txt").read_bytes()).text
    profile = parse_job_description(jd)
    job_id = pipeline.save_job(admin["id"], profile, title="UI test job", required=profile.required_skills,
                               preferred=profile.preferred_skills, min_years=profile.min_years,
                               max_years=profile.max_years)
    files = [(p.name, p.read_bytes()) for p in sorted((samples / "resumes" / "ml_engineer").iterdir())]
    pipeline.add_resumes(admin["id"], job_id, files)
    for name, status in {"Priya": "shortlisted", "Meera": "interview", "Rahul": "shortlisted",
                         "Karan": "rejected", "Sneha": "rejected", "Vikram": "rejected"}.items():
        pipeline.set_status(admin["id"], pipeline.find_candidate_ids(job_id, name)[:1], status)
    return admin, job_id


def page(view: str, user: dict, job_id: int) -> AppTest:
    at = AppTest.from_string(
        f"import sys\nsys.path.insert(0, {str(ROOT)!r})\nfrom ui.views import {view}\n{view}.render()\n",
        default_timeout=120,
    )
    at.session_state["user"] = user
    at.session_state["job_id"] = job_id
    return at


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


@pytest.mark.parametrize("view", VIEWS)
def test_every_page_renders(ui_job, view):
    admin, job_id = ui_job
    at = page(view, admin, job_id).run()
    assert not at.exception, [e.message for e in at.exception]


def test_login_flow():
    init_db()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    at.text_input[0].input("admin")
    at.text_input[1].input("nope")
    at.button[0].click().run()
    assert "Invalid" in at.error[0].value
    at.text_input[0].input("admin")
    at.text_input[1].input("admin123")
    at.button[0].click().run()
    assert at.session_state["user"]["username"] == "admin"
    assert not at.exception


def test_create_job_flow(ui_job):
    admin, job_id = ui_job
    before = len(pipeline.list_jobs())
    at = page("jobs", admin, job_id).run()
    at.text_area[0].input("Data Analyst\nRequirements:\n- 2+ years SQL, Excel, Tableau and Python\nNice to have:\n- Power BI")
    at.button[0].click().run()
    required = next(m for m in at.multiselect if m.label == "Required skills")
    assert {"SQL", "Excel", "Tableau", "Python"} <= set(required.value)
    button(at, "Save job").click().run()
    assert len(pipeline.list_jobs()) == before + 1


def test_assistant_quick_action(ui_job):
    admin, job_id = ui_job
    at = page("assistant", admin, job_id).run()
    button(at, "Who are the top 5 candidates and why?").click().run()
    assert any("Top 5" in m.value for m in at.markdown)
    assert insights.load_chat(admin["id"], job_id)


def test_train_ranker_and_ml_rank_mode(ui_job):
    admin, job_id = ui_job
    at = page("analytics", admin, job_id).run()
    button(at, "Train / retrain ranker").click().run()
    assert any("Best model" in s.value for s in at.success)
    at = page("rankings", admin, job_id).run()
    rank_by = next(s for s in at.selectbox if s.label == "Rank by")
    assert len(rank_by.options) == 3
    rank_by.select("Blend (50/50)").run()
    assert not at.exception


def test_candidate_ai_and_status(ui_job):
    admin, job_id = ui_job
    at = page("candidate", admin, job_id).run()
    button(at, "Generate evidence-based summary").click().run()
    button(at, "Generate interview questions").click().run()
    texts = " ".join(m.value for m in at.markdown)
    assert "Skill gap" in texts and "Interview questions" in texts
    status_box = next(s for s in at.selectbox if s.key and s.key.startswith("status_"))
    status_box.select("hired")
    button(at, "Save").click().run()
    assert pipeline.get_candidate(at.session_state["candidate_id"])["status"] == "hired"


def test_blind_mode_hides_names(ui_job):
    admin, job_id = ui_job
    at = page("candidate", admin, job_id)
    at.session_state["blind_mode"] = True
    at.run()
    assert all(h.value.startswith("Candidate C") for h in at.subheader)
