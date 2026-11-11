"""Full pipeline through the database: job -> resumes -> scores/ranks -> RAG
retrieval -> offline agent -> learned ranker -> maintenance -> deletion."""
from __future__ import annotations

import json
import subprocess

import pytest

pytest.importorskip("sqlalchemy")

from intellihire.agent.agent import RecruiterAgent  # noqa: E402
from intellihire.db.session import init_db  # noqa: E402
from intellihire.ingestion.documents import extract_document  # noqa: E402
from intellihire.nlp.jd_parser import parse_job_description  # noqa: E402
from intellihire.rag.retriever import retrieve  # noqa: E402
from intellihire.services import insights, pipeline, users  # noqa: E402
from conftest import FFMPEG  # noqa: E402


@pytest.fixture(scope="module")
def seeded(samples):
    init_db()
    admin = users.authenticate("admin", "admin123")
    assert admin and admin["role"] == "admin"
    jd = extract_document("jd.txt", (samples / "jobs" / "ml_engineer_jd.txt").read_bytes()).text
    profile = parse_job_description(jd)
    job_id = pipeline.save_job(admin["id"], profile, title=profile.title, required=profile.required_skills,
                               preferred=profile.preferred_skills, min_years=profile.min_years,
                               max_years=profile.max_years)
    files = [(p.name, p.read_bytes()) for p in sorted((samples / "resumes" / "ml_engineer").iterdir())]
    results = pipeline.add_resumes(admin["id"], job_id, files)
    return admin, job_id, results


def test_auth_and_user_management(seeded):
    admin, _, _ = seeded
    assert users.authenticate("admin", "wrong") is None
    created = users.create_user(admin["id"], "recruiter1", "Secret123", "recruiter", "Rita Recruiter")
    assert users.authenticate("recruiter1", "Secret123")["role"] == "recruiter"
    with pytest.raises(users.UserError):
        users.create_user(admin["id"], "recruiter1", "Secret123", "recruiter")
    with pytest.raises(users.UserError):
        users.create_user(admin["id"], "weak", "short", "recruiter")
    users.set_user_active(admin["id"], created["id"], False)
    assert users.authenticate("recruiter1", "Secret123") is None
    assert any(r["action"] == "login_failed" for r in users.recent_audit())


def test_resumes_processed_and_ranked(seeded):
    _, job_id, results = seeded
    assert all(r.status == "added" for r in results), [(r.filename, r.message) for r in results if r.status != "added"]
    rows = insights.ranking_rows(job_id)
    assert len(rows) == len(results)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert rows[0]["final"] >= rows[-1]["final"]
    top_names = [r["name"] for r in rows[:5]]
    assert "Priya Sharma" in top_names
    bottom = [r["name"] for r in rows[-4:]]
    assert {"Karan Mehta", "Sneha Patel"} & set(bottom)


def test_duplicate_upload_is_skipped(seeded, samples):
    admin, job_id, _ = seeded
    path = samples / "resumes" / "ml_engineer" / "priya_sharma.pdf"
    again = pipeline.add_resume(admin["id"], job_id, path.name, path.read_bytes())
    assert again.status == "duplicate"


def test_rejected_files_report_reason(seeded):
    admin, job_id, _ = seeded
    res = pipeline.add_resume(admin["id"], job_id, "evil.pdf", b"MZ not a pdf")
    assert res.status == "error" and "signature" in res.message


def test_injection_resumes_flagged(seeded):
    _, job_id, _ = seeded
    rows = {r["name"]: r for r in insights.ranking_rows(job_id)}
    assert rows["Neha Gupta"]["flags"] > 0
    assert rows["Rohan Das"]["flags"] > 0
    cid = pipeline.find_candidate_ids(job_id, "Rohan Das")[0]
    detail = pipeline.get_candidate(cid)
    if "pymupdf" in detail["parser"]:
        # hidden keyword stuffing must not count as skills
        assert "PyTorch" not in detail["skills"] and "Kubernetes" not in detail["skills"]
        assert rows["Rohan Das"]["rank"] > len(rows) // 2


def test_candidate_detail_has_explanations(seeded):
    _, job_id, _ = seeded
    cid = pipeline.find_candidate_ids(job_id, "#1")[0]
    c = pipeline.get_candidate(cid)
    d = c["score"]["details"]
    assert d["reasons"] and abs(sum(d["contributions"].values()) - c["score"]["final"]) < 0.1
    assert c["skills"] and all(isinstance(v, list) for v in c["skills"].values())


def test_retrieval_is_scoped_and_sanitised(seeded):
    _, job_id, _ = seeded
    neha = pipeline.find_candidate_ids(job_id, "Neha")[0]
    hits = retrieve(job_id, "machine learning NLP experience", [neha], k=3)
    assert hits and all(h["candidate_id"] == neha for h in hits)
    joined = " ".join(h["text"] for h in hits)
    assert "ignore all previous instructions" not in joined.lower()
    assert "@example.com" not in joined
    blind = retrieve(job_id, "python", k=3, blind=True)
    assert all(h["candidate"].startswith("Candidate C") for h in blind)


@pytest.mark.parametrize("query, tool", [
    ("Who are the top 5 candidates and why?", "list_top_candidates"),
    ("Compare the top 2 candidates", "compare_candidates"),
    ("What are the skill gaps of #1?", "get_skill_gap"),
    ("Write interview questions for Priya", "interview_question_plan"),
    ("Who has Kubernetes experience?", "find_candidates_by_skill"),
    ("Tell me about Meera Krishnan", "get_candidate_profile"),
    ("What are the job requirements?", "get_job_requirements"),
    ("Which applicant built semantic search systems in production?", "search_evidence"),
])
def test_offline_agent_routes_to_tools(seeded, query, tool):
    _, job_id, _ = seeded
    result = RecruiterAgent(job_id).run(query)
    assert result.mode == "offline"
    assert result.trace and result.trace[0]["tool"] == tool
    assert "error" not in json.loads(result.trace[0]["result"])
    assert len(result.text) > 40


def test_agent_tools_handle_bad_input(seeded):
    _, job_id, _ = seeded
    agent = RecruiterAgent(job_id)
    assert "error" in json.loads(agent.tools.execute("get_candidate_profile", {"candidate": "Nobody Atall"}))
    assert "error" in json.loads(agent.tools.execute("drop_tables", {}))
    assert "error" in json.loads(agent.tools.execute("list_top_candidates", {"bogus": 1}))


def test_summary_and_questions_offline(seeded):
    _, job_id, _ = seeded
    cid = pipeline.find_candidate_ids(job_id, "#1")[0]
    agent = RecruiterAgent(job_id)
    assert "Skill gap" in agent.summarize(cid).text
    assert "Interview questions" in agent.interview_questions(cid).text


def test_job_update_rescores(seeded):
    admin, job_id, _ = seeded
    before = {r["id"]: r["final"] for r in insights.ranking_rows(job_id)}
    pipeline.update_job(admin["id"], job_id, weights={"semantic": 0, "skill": 1, "experience": 0, "video": 0,
                                                      "communication": 0})
    after = insights.ranking_rows(job_id)
    assert any(abs(before[r["id"]] - r["final"]) > 0.5 for r in after)
    for r in after:  # with only the skill weight, final == skill score
        if r["skill"] is not None:
            assert r["final"] == pytest.approx(r["skill"], abs=0.1)
    pipeline.update_job(admin["id"], job_id, weights={"semantic": 0.3, "skill": 0.35, "experience": 0.2,
                                                      "video": 0.1, "communication": 0.05},
                        custom_skills=["Agentic AI"])


def test_learned_ranker(seeded):
    admin, job_id, _ = seeded
    decisions = {"Priya Sharma": "shortlisted", "Meera Krishnan": "interview", "Rahul Verma": "shortlisted",
                 "Fatima Khan": "shortlisted", "Karan Mehta": "rejected", "Sneha Patel": "rejected",
                 "Vikram Singh": "rejected", "Arjun Nair": "rejected"}
    for name, status in decisions.items():
        pipeline.set_status(admin["id"], pipeline.find_candidate_ids(job_id, name)[:1], status)
    report = insights.train_job_ranker(admin["id"], job_id)
    assert report.n_samples == 8 and report.feature_importance
    rows = insights.ranking_rows(job_id)
    assert all(r["ml"] is not None for r in rows)
    assert insights.latest_ranker(job_id)["model"] == report.best_model


def test_analytics_and_chat_log(seeded):
    admin, job_id, _ = seeded
    data = insights.job_analytics(job_id)
    assert data["n"] > 0 and data["coverage"] and data["status_counts"]
    insights.save_chat(admin["id"], job_id, "user", "hello")
    insights.save_chat(admin["id"], job_id, "assistant", "hi", [{"tool": "x"}])
    assert [m["role"] for m in insights.load_chat(admin["id"], job_id)] == ["user", "assistant"]
    insights.clear_chat(admin["id"], job_id)
    assert insights.load_chat(admin["id"], job_id) == []
    assert insights.dashboard_stats()["candidates"] >= data["n"]


def test_rebuild_index(seeded):
    _, job_id, _ = seeded
    n = pipeline.rebuild_vector_index(reembed=False)
    assert n > 0
    assert retrieve(job_id, "python", k=2)
    assert pipeline.rebuild_vector_index(reembed=True) == n


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_video_without_speech_is_reported(seeded, tmp_path):
    admin, job_id, _ = seeded
    video = tmp_path / "intro.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=gray:s=160x120:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    cid = pipeline.find_candidate_ids(job_id, "#1")[0]
    res = pipeline.attach_video(admin["id"], cid, "intro.mp4", video.read_bytes())
    assert res.status == "error" and "audio" in res.message


def test_delete_candidate_and_job(seeded):
    admin, job_id, _ = seeded
    count = len(insights.ranking_rows(job_id))
    cid = pipeline.find_candidate_ids(job_id, "Karan")[0]
    pipeline.delete_candidate(admin["id"], cid)
    rows = insights.ranking_rows(job_id)
    assert len(rows) == count - 1 and [r["rank"] for r in rows] == list(range(1, count))
    pipeline.delete_job(admin["id"], job_id)
    assert pipeline.get_job(job_id) is None
    assert insights.ranking_rows(job_id) == []
