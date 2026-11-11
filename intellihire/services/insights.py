"""Read models for the dashboard: ranking tables, analytics, ML training, chat log."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sqlalchemy import delete, func, select

from intellihire.config import get_settings
from intellihire.db.models import Candidate, ChatMessage, Job, RankerModel, Score, VideoTranscript
from intellihire.db.session import session_scope
from intellihire.scoring.ranker import TrainingReport, status_to_label, train_ranker
from intellihire.services.pipeline import rescore_job
from intellihire.services.users import audit


def ranking_rows(job_id: int) -> list[dict]:
    with session_scope() as s:
        rows = s.execute(
            select(Candidate, Score, VideoTranscript.id)
            .join(Score, Score.candidate_id == Candidate.id, isouter=True)
            .join(VideoTranscript, VideoTranscript.candidate_id == Candidate.id, isouter=True)
            .where(Candidate.job_id == job_id)
            .order_by(Score.rank.is_(None), Score.rank)
        ).all()
        job = s.get(Job, job_id)
        n_required = len(job.required_skills) if job else None
        out = []
        for cand, sc, video_id in rows:
            details = (sc.details if sc else {}) or {}
            out.append({
                "id": cand.id,
                "rank": sc.rank if sc else None,
                "name": cand.name,
                "email": cand.email,
                "final": sc.final_score if sc else 0.0,
                "ml": sc.ml_score if sc else None,
                "semantic": (sc.s_semantic or 0) * 100 if sc else 0,
                "skill": None if not sc or sc.s_skill is None else sc.s_skill * 100,
                "experience": (sc.s_experience or 0) * 100 if sc else 0,
                "video": None if not sc or sc.s_video is None else sc.s_video * 100,
                "communication": None if not sc or sc.s_comm is None else sc.s_comm * 100,
                "years": cand.experience_years,
                "matched": len(details.get("matched_required", [])),
                "required_total": n_required if n_required is not None else (
                    len(details.get("matched_required", [])) + len(details.get("missing_required", []))),
                "missing": details.get("missing_required", []),
                "has_video": video_id is not None,
                "flags": len(cand.injection_flags or []),
                "status": cand.status,
                "skills": sorted((cand.skills or {}).keys()),
            })
        return out


def dashboard_stats() -> dict:
    with session_scope() as s:
        return {
            "jobs_open": s.scalar(select(func.count(Job.id)).where(Job.status == "open")) or 0,
            "jobs_total": s.scalar(select(func.count(Job.id))) or 0,
            "candidates": s.scalar(select(func.count(Candidate.id))) or 0,
            "videos": s.scalar(select(func.count(VideoTranscript.id))) or 0,
            "shortlisted": s.scalar(
                select(func.count(Candidate.id)).where(Candidate.status.in_(["shortlisted", "interview", "hired"]))
            ) or 0,
            "flagged": sum(1 for f in s.scalars(select(Candidate.injection_flags)) if f),
            "avg_score": round(float(s.scalar(select(func.avg(Score.final_score))) or 0.0), 1),
        }


def job_analytics(job_id: int) -> dict:
    rows = ranking_rows(job_id)
    with session_scope() as s:
        job = s.get(Job, job_id)
        required = list(job.required_skills) if job else []
        preferred = list(job.preferred_skills) if job else []
        created = [c for (c,) in s.execute(select(Candidate.created_at).where(Candidate.job_id == job_id))]
    n = len(rows)
    missing = Counter(skill for r in rows for skill in r["missing"])
    coverage = {
        skill: sum(1 for r in rows if skill in r["skills"]) / n * 100 if n else 0.0
        for skill in required + preferred
    }
    return {
        "rows": rows,
        "n": n,
        "missing_counts": dict(missing.most_common()),
        "coverage": coverage,
        "required": required,
        "preferred": preferred,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "created": created,
        "with_video": sum(1 for r in rows if r["has_video"]),
        "flagged": sum(1 for r in rows if r["flags"]),
    }


def train_job_ranker(user_id: int, job_id: int) -> TrainingReport:
    with session_scope() as s:
        rows = s.execute(
            select(Candidate.status, Score.details).join(Score, Score.candidate_id == Candidate.id)
            .where(Candidate.job_id == job_id)
        ).all()
    X, y = [], []
    for status, details in rows:
        label = status_to_label(status)
        if label is not None and details and "features" in details:
            X.append(details["features"])
            y.append(label)
    path = get_settings().model_dir / "rankers" / f"job_{job_id}.joblib"
    _, report = train_ranker(np.asarray(X), np.asarray(y), save_path=path)
    with session_scope() as s:
        s.add(RankerModel(job_id=job_id, model_type=report.best_model, metrics=report.cv_auc,
                          feature_importance=report.feature_importance, n_samples=report.n_samples, path=str(path)))
        audit(s, user_id, "ranker_trained", "job", job_id, model=report.best_model, n=report.n_samples)
    rescore_job(job_id)  # fills ml_score for every candidate
    return report


def latest_ranker(job_id: int) -> dict | None:
    with session_scope() as s:
        m = s.scalar(select(RankerModel).where(RankerModel.job_id == job_id).order_by(RankerModel.id.desc()))
        if m is None:
            return None
        return {"model": m.model_type, "metrics": m.metrics, "importance": m.feature_importance,
                "n": m.n_samples, "trained_at": m.trained_at}


# ------------------------------------------------------------------- chat log
def load_chat(user_id: int, job_id: int, limit: int = 50) -> list[dict]:
    with session_scope() as s:
        rows = list(s.scalars(
            select(ChatMessage).where(ChatMessage.user_id == user_id, ChatMessage.job_id == job_id)
            .order_by(ChatMessage.id.desc()).limit(limit)
        ))
    return [{"role": r.role, "content": r.content, "trace": r.trace} for r in reversed(rows)]


def save_chat(user_id: int, job_id: int, role: str, content: str, trace: list | None = None) -> None:
    with session_scope() as s:
        s.add(ChatMessage(user_id=user_id, job_id=job_id, role=role, content=content, trace=trace or []))


def clear_chat(user_id: int, job_id: int) -> None:
    with session_scope() as s:
        s.execute(delete(ChatMessage).where(ChatMessage.user_id == user_id, ChatMessage.job_id == job_id))
