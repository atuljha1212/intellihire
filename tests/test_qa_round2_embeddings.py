"""Round 2 (O1/O2): per-candidate encoder tracking, migration and fallback-safe rebuild."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# Optional: path to a pre-round-2 copy of the codebase used to build a legacy database.
_BACKUP_ENV = os.environ.get("INTELLIHIRE_ROUND2_BACKUP", "")
BACKUP = Path(_BACKUP_ENV) if _BACKUP_ENV else None


@pytest.fixture(scope="module")
def ctx(samples):
    from sqlalchemy import select

    from intellihire.db.models import User
    from intellihire.db.session import init_db, session_scope
    from intellihire.nlp.jd_parser import parse_job_description
    from intellihire.services import pipeline

    init_db()
    with session_scope() as s:
        uid = s.scalar(select(User.id))
    profile = parse_job_description("Data engineer. Required: Python, SQL, Docker. 3+ years experience.")
    job_id = pipeline.save_job(uid, profile, title="R2 emb job", required=["Python", "SQL", "Docker"],
                               preferred=[], min_years=3, max_years=None)
    resume = next((samples / "resumes" / "ml_engineer").iterdir())
    res = pipeline.add_resume(uid, job_id, resume.name, resume.read_bytes())
    assert res.status == "added"
    return uid, job_id, res.candidate_id


def _set(model, ident, **kw):
    from intellihire.db.session import session_scope

    with session_scope() as s:
        obj = s.get(model, ident)
        for k, v in kw.items():
            setattr(obj, k, v)


def test_new_resume_records_active_encoder(ctx):
    from intellihire.db.models import Candidate
    from intellihire.db.session import session_scope
    from intellihire.embeddings.encoder import get_encoder

    _, _, cid = ctx
    with session_scope() as s:
        assert s.get(Candidate, cid).encoder_name == get_encoder().name


def test_candidate_encoder_mismatch_drops_semantic_only_for_that_candidate(ctx):
    from intellihire.db.models import Candidate
    from intellihire.services import pipeline

    _, job_id, cid = ctx
    _set(Candidate, cid, encoder_name="hashing-fallback-384")
    pipeline.rescore_job(job_id)
    det = pipeline.get_candidate(cid)["score"]["details"]
    assert "semantic" not in det["weights_used"]
    assert "re-embed" in det["reasons"][0].lower()
    _set(Candidate, cid, encoder_name=None)  # legacy NULL: assumed to match the job
    pipeline.rescore_job(job_id)
    assert "semantic" in pipeline.get_candidate(cid)["score"]["details"]["weights_used"]
    _set(Candidate, cid, encoder_name=pipeline.get_encoder().name)


def test_health_counts_candidate_mismatch(ctx):
    from intellihire.db.models import Candidate
    from intellihire.services import pipeline
    from intellihire.services.system import component_statuses

    _, _, cid = ctx
    _set(Candidate, cid, encoder_name="other-enc")
    try:
        rows = {r.name: r for r in component_statuses()}
        assert "1 candidate(s)" in rows["Candidate embeddings"].detail
    finally:
        _set(Candidate, cid, encoder_name=pipeline.get_encoder().name)


def test_rebuild_refused_when_fallback_active(ctx, monkeypatch):
    from intellihire.db.models import Candidate
    from intellihire.embeddings import encoder as enc_mod
    from intellihire.services import pipeline

    _, job_id, cid = ctx
    real = pipeline.get_encoder().name
    before = pipeline.get_candidate(cid)["score"]["final"]
    monkeypatch.setattr(pipeline, "get_encoder", lambda: enc_mod.HashingEncoder())
    for reembed in (True, False):
        with pytest.raises(pipeline.EmbeddingModelUnavailable, match="fallback active"):
            pipeline.rebuild_vector_index(reembed=reembed)
    monkeypatch.undo()
    from intellihire.db.session import session_scope

    with session_scope() as s:
        assert s.get(Candidate, cid).encoder_name == real
    assert pipeline.get_candidate(cid)["score"]["final"] == before


def test_reembed_updates_candidate_encoder(ctx):
    from intellihire.db.models import Candidate, VideoTranscript
    from intellihire.db.session import session_scope
    from intellihire.services import pipeline

    _, job_id, cid = ctx
    _set(Candidate, cid, encoder_name="other-enc")
    pipeline.rebuild_vector_index()  # candidate mismatch forces re-embedding
    with session_scope() as s:
        assert s.get(Candidate, cid).encoder_name == pipeline.get_encoder().name
    assert "semantic" in pipeline.get_candidate(cid)["score"]["details"]["weights_used"]
    assert VideoTranscript.encoder_name is not None


def test_migration_adds_columns_to_round2_database(tmp_path):
    if BACKUP is None or not BACKUP.exists():
        pytest.skip("round-2 backup not available")
    data = tmp_path / "data"
    db = tmp_path / "old.db"
    env = {**os.environ, "INTELLIHIRE_DATA_DIR": str(data), "DATABASE_URL": f"sqlite:///{db.as_posix()}",
           "LLM_PROVIDER": "none"}
    old = textwrap.dedent(f"""
        import sys; sys.path.insert(0, r"{BACKUP}")
        from intellihire.db.session import init_db
        init_db()
        from sqlalchemy import inspect
        from intellihire.db.session import get_engine
        cols = [c["name"] for c in inspect(get_engine()).get_columns("candidates")]
        assert "encoder_name" not in cols
        from sqlalchemy import select
        from intellihire.db.models import User
        from intellihire.db.session import session_scope
        from intellihire.nlp.jd_parser import parse_job_description
        from intellihire.services import pipeline
        with session_scope() as s:
            uid = s.scalar(select(User.id))
        p = parse_job_description("Data engineer. Required: Python, SQL. 3+ years experience.")
        jid = pipeline.save_job(uid, p, title="old", required=["Python","SQL"], preferred=[], min_years=3, max_years=None)
        f = next((__import__("pathlib").Path(r"{ROOT}") / "sample_data" / "resumes" / "ml_engineer").iterdir())
        assert pipeline.add_resume(uid, jid, f.name, f.read_bytes()).status == "added"
        print("OLD_OK")
    """)
    r = subprocess.run([sys.executable, "-c", old], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert "OLD_OK" in r.stdout, r.stderr[-2000:]
    new = textwrap.dedent(f"""
        import sys; sys.path.insert(0, r"{ROOT}")
        from intellihire.db.session import init_db, get_engine, session_scope
        init_db(); init_db()  # idempotent
        from sqlalchemy import inspect
        insp = inspect(get_engine())
        for t in ("candidates", "video_transcripts"):
            assert "encoder_name" in [c["name"] for c in insp.get_columns(t)], t
        from intellihire.services import pipeline
        from intellihire.services.system import component_statuses
        jobs = pipeline.list_jobs()
        assert pipeline.rescore_job(jobs[0]["id"]) == 1
        cand = pipeline.get_candidate(pipeline.find_candidate_ids(jobs[0]["id"], "#1")[0])
        assert "semantic" in cand["score"]["details"]["weights_used"]  # legacy NULL assumed to match
        rows = {{r.name: r for r in component_statuses()}}
        assert "unknown encoder" in rows["Embedding provenance"].detail
        print("NEW_OK")
    """)
    r = subprocess.run([sys.executable, "-c", new], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert "NEW_OK" in r.stdout, r.stderr[-2000:]
