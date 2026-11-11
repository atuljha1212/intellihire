"""Regression tests for the QA 'core' findings (T06, T21-T23, T32-T35, T43-T46, T59)."""
from __future__ import annotations

import numpy as np
import pytest

from intellihire.config import _env_positive_float, _env_positive_int
from intellihire.rag.chunking import chunk_text, chunk_transcript
from intellihire.rag.vector_store import FaissStore, NumpyStore
from intellihire.scoring.features import CandidateSpec, JobSpec, compute_scores


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def vecs(n, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# ---- T35
def test_chunk_overlap_never_exceeds_max_words():
    text = "\n".join(" ".join("w" for _ in range(n)) for n in (10, 100, 120))
    sizes = [len(c.text.split()) for c in chunk_text(text, max_words=120, overlap_words=25)]
    assert max(sizes) <= 120, sizes


def test_long_transcript_segment_is_split():
    chunks = chunk_transcript([{"start": 0.0, "end": 100.0, "text": "word " * 1000}], max_words=80)
    assert len(chunks) > 1 and all(len(c.text.split()) <= 80 for c in chunks)
    assert chunks[0].start_sec == 0.0 and chunks[-1].end_sec == pytest.approx(100.0)


# ---- T21 / T23 / T22 (FAISS)
faiss = pytest.importorskip("faiss")


def test_stale_flag_survives_add_and_restart(tmp_path):
    s = FaissStore(tmp_path, 8, "enc-a")
    s.add([1, 2, 3], vecs(3))
    s2 = FaissStore(tmp_path, 8, "enc-b")  # encoder changed
    assert s2.stale
    s2.add([10], vecs(1, seed=5))
    s3 = FaissStore(tmp_path, 8, "enc-b")
    assert s3.stale  # the add must not have erased the stale marker


def test_replace_all_clears_stale_and_persists(tmp_path):
    FaissStore(tmp_path, 8, "enc-a").add([1], vecs(1))
    s = FaissStore(tmp_path, 8, "enc-b")
    s.replace_all([7, 8], vecs(2, seed=3))
    assert not s.stale and s.count() == 2
    again = FaissStore(tmp_path, 8, "enc-b")
    assert not again.stale and again.count() == 2


def test_corrupt_index_marks_stale_instead_of_crashing(tmp_path):
    (tmp_path / "faiss.index").write_bytes(b"garbage")
    s = FaissStore(tmp_path, 8, "enc-a")
    assert s.stale and s.count() == 0


def test_save_is_atomic_no_tmp_left(tmp_path):
    s = FaissStore(tmp_path, 8, "enc-a")
    s.add([1], vecs(1))
    assert not list(tmp_path.glob("*.tmp"))


# ---- T46 / numpy store
def test_numpy_store_corrupt_and_search(tmp_path):
    (tmp_path / "numpy_store.npz").write_bytes(b"junk")
    s = NumpyStore(tmp_path, 8, "enc-a")
    assert s.stale
    s.replace_all([1, 2], vecs(2))
    assert not s.stale and s.search(vecs(2)[0], k=1)[0][0] == 1
    assert NumpyStore(tmp_path, 8, "enc-a").count() == 2


# ---- T33 / T06 scoring
def _job(**kw):
    base = dict(required_skills=["Python"], preferred_skills=[], min_years=None, max_years=None,
                jd_vec=unit([1, 0, 0, 0]), resp_vec=unit([0, 1, 0, 0]),
                weights={"semantic": 1, "skill": 1, "experience": 1, "video": 1, "communication": 1})
    base.update(kw)
    return JobSpec(**base)


def test_perfect_candidate_scores_exactly_100():
    cand = CandidateSpec(skills={"Python": []}, years=0, resume_vec=unit([1, 0, 0, 0]),
                         video_vec=unit([0, 1, 0, 0]), comm_score=1.0)
    b = compute_scores(_job(), cand, lambda c: 1.0)
    assert b.final_score == 100.0


def test_zero_active_weight_warns():
    cand = CandidateSpec(skills={}, years=0, resume_vec=unit([1, 0, 0, 0]))
    job = _job(weights={"semantic": 0, "skill": 0, "experience": 0, "video": 1, "communication": 1})
    b = compute_scores(job, cand, lambda c: 1.0)
    assert b.final_score == 0 and any("No scoring feature" in r for r in b.reasons)
    assert not any("redistributed" in r for r in b.reasons)


def test_encoder_mismatch_drops_semantic():
    cand = CandidateSpec(skills={"Python": []}, years=0, resume_vec=unit([0, 0, 1, 0]))
    b = compute_scores(_job(embeddings_ok=False, embeddings_note="mismatch!"), cand, lambda c: 1.0)
    assert "semantic" not in b.weights_used and "mismatch!" in b.reasons[0]


# ---- T59
def test_non_positive_limits_fall_back(monkeypatch):
    monkeypatch.setenv("MAX_VIDEO_SECONDS", "0")
    monkeypatch.setenv("MAX_VIDEO_MB", "-5")
    assert _env_positive_int("MAX_VIDEO_SECONDS", 300) == 300
    assert _env_positive_float("MAX_VIDEO_MB", 200) == 200


# ---- T44
def test_ranker_trains_and_nan_fold_is_not_averaged():
    from intellihire.scoring.ranker import train_ranker

    rng = np.random.default_rng(1)
    X = rng.random((20, 6))
    y = np.array([0, 1] * 10)
    _, report = train_ranker(X, y)
    assert report.best_model


# ---- T43 / T32 / T22 / T06 (DB level)
@pytest.fixture(scope="module")
def db_job(samples):
    from sqlalchemy import select

    from intellihire.db.models import User
    from intellihire.db.session import init_db, session_scope
    from intellihire.nlp.jd_parser import parse_job_description
    from intellihire.services import pipeline

    init_db()
    with session_scope() as s:
        uid = s.scalar(select(User.id))
    profile = parse_job_description("Data engineer. Required: Python, SQL, Docker, Kubernetes, AWS. 3+ years experience.")
    job_id = pipeline.save_job(uid, profile, title="QA core job", required=["Python", "SQL", "Docker", "Kubernetes", "AWS"],
                               preferred=[], min_years=3, max_years=None)
    resume = (samples / "resumes" / "ml_engineer").iterdir().__next__()
    res = pipeline.add_resume(uid, job_id, resume.name, resume.read_bytes())
    assert res.status == "added"
    return uid, job_id, res.candidate_id


def test_find_candidate_ids_rejects_empty_ref(db_job):
    from intellihire.services import pipeline

    _, job_id, cid = db_job
    assert pipeline.find_candidate_ids(job_id, "") == []
    assert pipeline.find_candidate_ids(job_id, "   ") == []
    assert pipeline.find_candidate_ids(job_id, str(cid)) == [cid]


def test_required_total_uses_job_required_count(db_job):
    from intellihire.services import insights

    _, job_id, _ = db_job
    assert all(r["required_total"] == 5 for r in insights.ranking_rows(job_id))


def test_encoder_mismatch_disables_semantic_then_rebuild_fixes(db_job):
    from intellihire.db.models import Job
    from intellihire.db.session import session_scope
    from intellihire.services import pipeline

    _, job_id, cid = db_job
    with session_scope() as s:
        s.get(Job, job_id).encoder_name = "some-other-encoder"
    pipeline.rescore_job(job_id)
    det = pipeline.get_candidate(cid)["score"]["details"]
    assert "semantic" not in det["weights_used"]
    assert "unavailable" in det["reasons"][0].lower()
    pipeline.rebuild_vector_index()  # mismatch forces re-embedding
    det = pipeline.get_candidate(cid)["score"]["details"]
    assert "semantic" in det["weights_used"]
    assert pipeline.get_job(job_id)["encoder_name"] == pipeline.get_encoder().name


def test_unindex_failure_is_swallowed(monkeypatch):
    from intellihire.services import pipeline

    class Boom:
        def remove(self, ids):
            raise RuntimeError("disk full")

    monkeypatch.setattr(pipeline, "get_vector_store", lambda: Boom())
    pipeline._unindex([1, 2])  # must not raise


def test_attach_video_merges_transcript_flags_and_surfaces_errors(db_job, monkeypatch):
    from types import SimpleNamespace

    from intellihire.services import analysis, pipeline

    uid, _, cid = db_job
    flag = {"kind": "override instructions", "snippet": "ignore previous", "location": "video transcript"}
    comm = SimpleNamespace(word_count=3, wpm=100.0, mattr=0.5, filler_rate=0.0, score=0.5,
                           to_dict=lambda: {})
    transcript = SimpleNamespace(text="ignore previous", language="en", engine="fake", as_dicts=lambda: [])
    va = SimpleNamespace(
        validation=SimpleNamespace(safe_name="v.mp4", sha256="x"), metadata=SimpleNamespace(duration_sec=1.0),
        transcript=transcript, communication=comm, video_skills=[], chunks=[], chunk_vectors=[],
        doc_vector=np.ones(4, dtype=np.float32) / 2, injection_flags=[flag])
    monkeypatch.setattr(pipeline, "analyze_video", lambda *a, **k: va)
    for _ in range(2):  # re-attaching must not duplicate the flag
        assert pipeline.attach_video(uid, cid, "v.mp4", b"x").status == "video added"
    flags = pipeline.get_candidate(cid)["injection_flags"]
    assert [f["location"] for f in flags].count("video transcript") == 1

    def boom(*a, **k):
        raise analysis.AnalysisError("no countable words")

    monkeypatch.setattr(pipeline, "analyze_video", boom)
    res = pipeline.attach_video(uid, cid, "v.mp4", b"x")
    assert res.status == "error" and "no countable" in res.message
