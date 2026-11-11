"""End-to-end pipeline with persistence: jobs, resumes, videos, scores, vectors.

Ordering rule for the two stores: the relational transaction commits first,
then vectors are written to FAISS. Chunk embeddings are also kept in MySQL, so
the vector index can always be rebuilt from the database (`rebuild_vector_index`).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from intellihire.config import DEFAULT_WEIGHTS, get_settings
from intellihire.db.models import Candidate, Job, RankerModel, Score, VectorChunk, VideoTranscript
from intellihire.db.session import session_scope
from intellihire.embeddings.encoder import from_bytes, get_encoder, to_bytes
from intellihire.media.transcriber import TranscriptionUnavailable
from intellihire.nlp.jd_parser import JobProfile, parse_job_description
from intellihire.nlp.skills import extract_skills
from intellihire.rag.vector_store import get_vector_store
from intellihire.scoring.features import CandidateSpec, JobSpec, compute_scores, feature_row
from intellihire.scoring.ranker import load_ranker, predict_scores
from intellihire.services.analysis import AnalysisError, analyze_resume, analyze_video, embed_job
from intellihire.services.users import audit

log = logging.getLogger(__name__)

STATUSES = ["new", "shortlisted", "interview", "rejected", "hired"]


class PipelineError(Exception):
    pass


@dataclass
class ProcessResult:
    filename: str
    status: str  # "added" | "duplicate" | "error" | "video added"
    message: str = ""
    candidate_id: int | None = None
    name: str = ""
    flags: int = 0
    warnings: list[str] = field(default_factory=list)


# ============================================================================ jobs
def save_job(user_id: int, profile: JobProfile, *, title: str, department: str = "", location: str = "",
             required: list[str], preferred: list[str], min_years: float | None, max_years: float | None,
             weights: dict[str, float] | None = None, custom_skills: list[str] | None = None) -> int:
    encoder = get_encoder()
    jd_vec, resp_vec = embed_job(profile, encoder)
    with session_scope() as s:
        job = Job(
            title=title.strip()[:200] or profile.title,
            department=department.strip()[:120],
            location=location.strip()[:120],
            description=profile.text,
            required_skills=list(dict.fromkeys(required)),
            preferred_skills=[p for p in dict.fromkeys(preferred) if p not in required],
            custom_skills=list(custom_skills or []),
            min_years=min_years,
            max_years=max_years,
            responsibilities=profile.responsibilities,
            education_req=profile.education,
            key_phrases=profile.key_phrases,
            weights=dict(weights or DEFAULT_WEIGHTS),
            jd_embedding=to_bytes(jd_vec),
            resp_embedding=to_bytes(resp_vec),
            encoder_name=encoder.name,
            created_by=user_id,
        )
        s.add(job)
        s.flush()
        audit(s, user_id, "job_created", "job", job.id, title=job.title)
        return job.id


def update_job(user_id: int, job_id: int, **fields) -> None:
    """Update editable job fields; re-embeds when the description changes, then re-scores."""
    allowed = {"title", "department", "location", "required_skills", "preferred_skills", "custom_skills",
               "min_years", "max_years", "weights", "status", "description"}
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise PipelineError("Job not found")
        for key, value in fields.items():
            if key in allowed:
                setattr(job, key, value)
        if "description" in fields:
            profile = parse_job_description(fields["description"], job.title, job.custom_skills)
            job.description = profile.text
            job.responsibilities = profile.responsibilities
            job.key_phrases = profile.key_phrases
            if _job_encoder_mismatch(job):
                # candidates are still in the old embedding space; re-embedding only the job would mix spaces.
                log.warning("Job %s keeps its old embeddings until the index is rebuilt with re-embedding", job_id)
            else:
                encoder = get_encoder()  # one instance for the vectors and the recorded name
                jd_vec, resp_vec = embed_job(profile, encoder)
                job.jd_embedding, job.resp_embedding = to_bytes(jd_vec), to_bytes(resp_vec)
                job.encoder_name = encoder.name
        job.preferred_skills = [p for p in job.preferred_skills if p not in job.required_skills]
        if "custom_skills" in fields:
            # new custom skills must be searched for in already-uploaded resumes and transcripts
            custom = tuple(job.custom_skills or ())
            for cand in s.scalars(select(Candidate).where(Candidate.job_id == job_id)):
                cand.skills = extract_skills(cand.resume_text, custom)
                if cand.video is not None:
                    cand.video.video_skills = sorted(extract_skills(cand.video.transcript, custom))
        audit(s, user_id, "job_updated", "job", job_id, fields=sorted(fields))
    rescore_job(job_id)


def delete_job(user_id: int, job_id: int) -> None:
    with session_scope() as s:
        chunk_ids = list(s.scalars(select(VectorChunk.id).where(VectorChunk.job_id == job_id)))
        job = s.get(Job, job_id)
        if job is None:
            return
        s.execute(delete(VectorChunk).where(VectorChunk.job_id == job_id))
        s.delete(job)
        audit(s, user_id, "job_deleted", "job", job_id, title=job.title)
    _unindex(chunk_ids)


def list_jobs(include_closed: bool = True) -> list[dict]:
    with session_scope() as s:
        counts = dict(s.execute(select(Candidate.job_id, func.count(Candidate.id)).group_by(Candidate.job_id)).all())
        videos = dict(
            s.execute(
                select(Candidate.job_id, func.count(VideoTranscript.id))
                .join(VideoTranscript, VideoTranscript.candidate_id == Candidate.id)
                .group_by(Candidate.job_id)
            ).all()
        )
        query = select(Job).order_by(Job.created_at.desc())
        if not include_closed:
            query = query.where(Job.status == "open")
        return [
            {
                "id": j.id, "title": j.title, "department": j.department, "location": j.location,
                "status": j.status, "created_at": j.created_at, "candidates": counts.get(j.id, 0),
                "videos": videos.get(j.id, 0), "required_skills": j.required_skills,
            }
            for j in s.scalars(query)
        ]


def get_job(job_id: int) -> dict | None:
    with session_scope() as s:
        j = s.get(Job, job_id)
        if j is None:
            return None
        return {
            "id": j.id, "title": j.title, "department": j.department, "location": j.location,
            "description": j.description, "required_skills": list(j.required_skills),
            "preferred_skills": list(j.preferred_skills), "custom_skills": list(j.custom_skills or []),
            "min_years": j.min_years, "max_years": j.max_years, "responsibilities": j.responsibilities,
            "education_req": j.education_req, "key_phrases": j.key_phrases,
            "weights": {**DEFAULT_WEIGHTS, **(j.weights or {})}, "status": j.status,
            "encoder_name": j.encoder_name, "created_at": j.created_at,
        }


def _job_encoder_mismatch(job: Job) -> bool:
    """True when the job's stored vectors come from another encoder than the active one."""
    return bool(job.encoder_name) and job.encoder_name != get_encoder().name


def _job_spec(job: Job) -> JobSpec:
    ok = not _job_encoder_mismatch(job)
    return JobSpec(
        required_skills=list(job.required_skills),
        preferred_skills=list(job.preferred_skills),
        min_years=job.min_years,
        max_years=job.max_years,
        jd_vec=from_bytes(job.jd_embedding),
        resp_vec=from_bytes(job.resp_embedding),
        weights={**DEFAULT_WEIGHTS, **(job.weights or {})},
        embeddings_ok=ok,
        encoder_name=job.encoder_name or get_encoder().name,
        embeddings_note="" if ok else (
            f"Semantic scores unavailable: job embeddings were built with '{job.encoder_name}' but the active "
            f"encoder is '{get_encoder().name}'. Run Admin > Rebuild vector index with re-embedding."),
    )


def _vec_matches(stored: str | None, spec: JobSpec) -> bool:
    """A stored vector may be compared with the job's only if the same encoder made both and it is the
    active one. NULL (legacy rows) is assumed to match the job."""
    return (stored or spec.encoder_name) == spec.encoder_name == get_encoder().name


# ====================================================================== scoring
def _score_candidate(s: Session, cand: Candidate, spec: JobSpec) -> Score:
    video = cand.video
    cspec = CandidateSpec(
        skills=cand.skills or {},
        years=cand.experience_years or 0.0,
        resume_vec=from_bytes(cand.resume_embedding),
        video_skills=list(video.video_skills) if video else [],
        video_vec=from_bytes(video.embedding) if video else None,
        comm_score=video.comm_score if video else None,
        comm_reliable=bool(video and video.word_count >= 40),
        resume_ok=_vec_matches(cand.encoder_name, spec),
        video_ok=bool(video) and _vec_matches(video.encoder_name, spec),
    )
    b = compute_scores(spec, cspec, get_encoder().calibrate)
    score = cand.score or Score(candidate_id=cand.id, job_id=cand.job_id)
    score.s_semantic, score.raw_cosine = b.s_semantic, b.raw_cosine
    score.s_skill, score.s_experience = b.s_skill, b.s_experience
    score.s_video, score.s_comm, score.final_score = b.s_video, b.s_comm, b.final_score
    score.matched_skills = b.matched_required + b.matched_preferred
    score.missing_skills = b.missing_required
    details = b.to_dict()
    details["features"] = feature_row(b, video is not None)
    score.details = details
    if cand.score is None:
        s.add(score)
        cand.score = score
    return score


def _update_ranks(s: Session, job_id: int) -> None:
    scores = list(s.scalars(select(Score).where(Score.job_id == job_id).order_by(Score.final_score.desc(), Score.id)))
    for i, sc in enumerate(scores, start=1):
        sc.rank = i
    model_row = s.scalar(select(RankerModel).where(RankerModel.job_id == job_id).order_by(RankerModel.id.desc()))
    model = load_ranker(get_settings().model_dir / "rankers" / f"job_{job_id}.joblib") if model_row else None
    if model is not None and scores:
        X = np.asarray([sc.details.get("features", [0] * 6) for sc in scores], dtype=float)
        for sc, p in zip(scores, predict_scores(model, X)):
            sc.ml_score = round(float(p) * 100, 2)


def rescore_job(job_id: int) -> int:
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            return 0
        spec = _job_spec(job)
        cands = list(s.scalars(select(Candidate).where(Candidate.job_id == job_id)))
        for cand in cands:
            _score_candidate(s, cand, spec)
        s.flush()
        _update_ranks(s, job_id)
        return len(cands)


# ===================================================================== resumes
def _store_chunks(s: Session, cand: Candidate, chunks, vectors) -> list[tuple[int, np.ndarray]]:
    rows = []
    for chunk, vec in zip(chunks, vectors):
        row = VectorChunk(
            candidate_id=cand.id, job_id=cand.job_id, source=chunk.source, chunk_index=chunk.index,
            text=chunk.text, start_sec=chunk.start_sec, end_sec=chunk.end_sec, embedding=to_bytes(vec),
        )
        s.add(row)
        rows.append((row, vec))
    s.flush()
    return [(row.id, vec) for row, vec in rows]


def _unindex(ids: list[int]) -> None:
    """Remove vectors after the DB commit; a failure must not abort the caller's follow-up work."""
    if not ids:
        return
    try:
        get_vector_store().remove(ids)
    except Exception as exc:  # leftover vectors are harmless (chunk rows are gone) and a rebuild clears them
        log.error("Vector removal failed (rebuild from Admin page): %s", exc)


def _index(pairs: list[tuple[int, np.ndarray]]) -> None:
    if not pairs:
        return
    try:
        get_vector_store().add([p[0] for p in pairs], np.vstack([p[1] for p in pairs]))
    except Exception as exc:  # the DB copy allows a later rebuild
        log.error("Vector indexing failed (rebuild from Admin page): %s", exc)


def add_resume(user_id: int, job_id: int, filename: str, data: bytes, rank_now: bool = True) -> ProcessResult:
    sha = hashlib.sha256(data).hexdigest()
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            return ProcessResult(filename, "error", "Job not found")
        existing = s.scalar(select(Candidate).where(Candidate.job_id == job_id, Candidate.file_sha256 == sha))
        if existing:
            return ProcessResult(filename, "duplicate", "Same file already uploaded for this job",
                                 existing.id, existing.name)
        custom = list(job.custom_skills or [])

    try:
        ra = analyze_resume(filename, data, custom)
    except AnalysisError as exc:
        return ProcessResult(filename, "error", str(exc))

    p = ra.profile
    with session_scope() as s:
        job = s.get(Job, job_id)
        cand = Candidate(
            job_id=job_id, name=p.name[:160], email=p.email[:255], phone=p.phone[:40], links=p.links,
            resume_filename=ra.validation.safe_name, file_sha256=sha, resume_text=ra.document.text,
            hidden_text=ra.document.hidden_text[:20000], parser=ra.document.parser,
            skills=p.skills, experience_years=p.experience.years, experience_method=p.experience.method,
            experience_evidence=p.experience.evidence, education=p.education, certifications=p.certifications,
            key_phrases=p.key_phrases, injection_flags=ra.injection_flags, warnings=ra.warnings,
            resume_embedding=to_bytes(ra.doc_vector),
            encoder_name=getattr(ra, "encoder_name", "") or get_encoder().name,
        )
        s.add(cand)
        s.flush()
        pairs = _store_chunks(s, cand, ra.chunks, ra.chunk_vectors)
        _score_candidate(s, cand, _job_spec(job))
        s.flush()
        if rank_now:
            _update_ranks(s, job_id)
        audit(s, user_id, "resume_added", "candidate", cand.id, file=ra.validation.safe_name,
              flags=len(ra.injection_flags))
        result = ProcessResult(filename, "added", "Processed", cand.id, cand.name, len(ra.injection_flags), ra.warnings)
    _index(pairs)
    return result


def add_resumes(user_id: int, job_id: int, files: list[tuple[str, bytes]],
                progress: Callable[[int, int, str], None] | None = None) -> list[ProcessResult]:
    results = []
    for i, (name, data) in enumerate(files, start=1):
        if progress:
            progress(i, len(files), name)
        try:
            results.append(add_resume(user_id, job_id, name, data, rank_now=False))
        except Exception as exc:  # keep the batch going
            log.exception("Failed to process %s", name)
            results.append(ProcessResult(name, "error", f"Unexpected error: {exc}"))
    with session_scope() as s:
        _update_ranks(s, job_id)
    return results


# ====================================================================== videos
def attach_video(user_id: int, candidate_id: int, filename: str, data: bytes) -> ProcessResult:
    with session_scope() as s:
        cand = s.get(Candidate, candidate_id)
        if cand is None:
            return ProcessResult(filename, "error", "Candidate not found")
        custom = list(cand.job.custom_skills or [])
        name = cand.name
    try:
        va = analyze_video(filename, data, custom)
    except (AnalysisError, TranscriptionUnavailable) as exc:
        return ProcessResult(filename, "error", str(exc), candidate_id, name)

    with session_scope() as s:
        cand = s.get(Candidate, candidate_id)
        old_ids = list(s.scalars(select(VectorChunk.id).where(
            VectorChunk.candidate_id == candidate_id, VectorChunk.source == "video")))
        s.execute(delete(VectorChunk).where(VectorChunk.candidate_id == candidate_id, VectorChunk.source == "video"))
        if cand.video is not None:
            s.delete(cand.video)
            s.flush()
        c = va.communication
        kept = [f for f in (cand.injection_flags or []) if f.get("location") != "video transcript"]
        cand.injection_flags = kept + list(getattr(va, "injection_flags", []) or [])
        cand.video = VideoTranscript(
            candidate_id=candidate_id, filename=va.validation.safe_name, file_sha256=va.validation.sha256,
            duration_sec=va.metadata.duration_sec, transcript=va.transcript.text, segments=va.transcript.as_dicts(),
            language=va.transcript.language, engine=va.transcript.engine, word_count=c.word_count, wpm=c.wpm,
            mattr=c.mattr, filler_rate=c.filler_rate, comm_score=c.score, comm_details=c.to_dict(),
            video_skills=va.video_skills, embedding=to_bytes(va.doc_vector),
            encoder_name=getattr(va, "encoder_name", "") or get_encoder().name,
        )
        s.flush()
        pairs = _store_chunks(s, cand, va.chunks, va.chunk_vectors)
        _score_candidate(s, cand, _job_spec(cand.job))
        s.flush()
        _update_ranks(s, cand.job_id)
        audit(s, user_id, "video_added", "candidate", candidate_id, file=va.validation.safe_name,
              seconds=va.metadata.duration_sec)
    _unindex(old_ids)
    _index(pairs)
    return ProcessResult(filename, "video added", f"{c.word_count} words transcribed", candidate_id, name)


# ================================================================== candidates
def set_status(user_id: int, candidate_ids: list[int], status: str) -> None:
    if status not in STATUSES:
        raise PipelineError(f"Unknown status {status}")
    with session_scope() as s:
        for cid in candidate_ids:
            cand = s.get(Candidate, cid)
            if cand is not None and cand.status != status:
                audit(s, user_id, "status_changed", "candidate", cid, old=cand.status, new=status)
                cand.status = status


def save_notes(user_id: int, candidate_id: int, notes: str) -> None:
    with session_scope() as s:
        cand = s.get(Candidate, candidate_id)
        if cand is not None:
            cand.recruiter_notes = notes[:10000]
            audit(s, user_id, "notes_saved", "candidate", candidate_id)


def save_summary(candidate_id: int, summary: str) -> None:
    with session_scope() as s:
        cand = s.get(Candidate, candidate_id)
        if cand is not None:
            cand.ai_summary = summary


def delete_candidate(user_id: int, candidate_id: int) -> None:
    with session_scope() as s:
        cand = s.get(Candidate, candidate_id)
        if cand is None:
            return
        job_id = cand.job_id
        chunk_ids = list(s.scalars(select(VectorChunk.id).where(VectorChunk.candidate_id == candidate_id)))
        s.delete(cand)
        s.flush()
        _update_ranks(s, job_id)
        audit(s, user_id, "candidate_deleted", "candidate", candidate_id)
    _unindex(chunk_ids)


def candidate_options(job_id: int) -> list[tuple[int, str]]:
    """(id, label) pairs ordered by rank, for select boxes."""
    with session_scope() as s:
        rows = s.execute(
            select(Candidate.id, Candidate.name, Score.rank, Score.final_score)
            .join(Score, Score.candidate_id == Candidate.id, isouter=True)
            .where(Candidate.job_id == job_id)
            .order_by(Score.rank.is_(None), Score.rank, Candidate.id)
        ).all()
    return [(r.id, f"#{r.rank or '-'}  {r.name}  ({(r.final_score or 0):.1f})") for r in rows]


def get_candidate(candidate_id: int) -> dict | None:
    with session_scope() as s:
        c = s.get(Candidate, candidate_id)
        if c is None:
            return None
        sc, v = c.score, c.video
        return {
            "id": c.id, "job_id": c.job_id, "job_title": c.job.title, "name": c.name, "email": c.email,
            "phone": c.phone, "links": c.links, "resume_filename": c.resume_filename, "resume_text": c.resume_text,
            "hidden_text": c.hidden_text, "parser": c.parser, "skills": c.skills or {},
            "experience_years": c.experience_years, "experience_method": c.experience_method,
            "experience_evidence": c.experience_evidence, "education": c.education,
            "certifications": c.certifications, "key_phrases": c.key_phrases,
            "injection_flags": c.injection_flags or [], "warnings": c.warnings or [], "status": c.status,
            "notes": c.recruiter_notes, "ai_summary": c.ai_summary, "created_at": c.created_at,
            "score": None if sc is None else {
                "final": sc.final_score, "rank": sc.rank, "ml": sc.ml_score, "semantic": sc.s_semantic,
                "skill": sc.s_skill, "experience": sc.s_experience, "video": sc.s_video,
                "communication": sc.s_comm, "details": sc.details,
            },
            "video": None if v is None else {
                "filename": v.filename, "duration": v.duration_sec, "transcript": v.transcript,
                "segments": v.segments, "language": v.language, "engine": v.engine, "wpm": v.wpm,
                "mattr": v.mattr, "filler_rate": v.filler_rate, "comm_score": v.comm_score,
                "comm_details": v.comm_details, "video_skills": v.video_skills, "word_count": v.word_count,
            },
        }


def find_candidate_ids(job_id: int, name_or_id: str) -> list[int]:
    """Resolve a reference like '12', '#3' (rank) or a (partial) name within a job."""
    import difflib

    ref = str(name_or_id).strip()
    if not ref:  # "" is a substring of every name: never match
        return []
    with session_scope() as s:
        rows = s.execute(
            select(Candidate.id, Candidate.name, Score.rank)
            .join(Score, Score.candidate_id == Candidate.id, isouter=True)
            .where(Candidate.job_id == job_id)
        ).all()
    if ref.startswith("#") and ref[1:].isdigit():
        return [r.id for r in rows if r.rank == int(ref[1:])]
    if ref.isdigit():
        hit = [r.id for r in rows if r.id == int(ref)]
        if hit:
            return hit
    lowered = ref.lower()
    exact = [r.id for r in rows if r.name.lower() == lowered]
    if exact:
        return exact
    partial = [r.id for r in rows if lowered in r.name.lower()]
    if partial:
        return partial
    names = {r.name.lower(): r.id for r in rows}
    return [names[m] for m in difflib.get_close_matches(lowered, list(names), n=2, cutoff=0.6)]


# ============================================================ maintenance
FALLBACK_REFUSAL = ("The real embedding model is not loaded (fallback active); re-embedding is disabled until it "
                    "loads. Existing embeddings were kept unchanged.")


class EmbeddingModelUnavailable(PipelineError):
    """Re-embedding/rebuilding was refused because only the hashing fallback encoder is active."""


def _is_null_vector(blob: bytes | None) -> bool:
    vec = from_bytes(blob)
    return vec is None or not np.any(vec)


def _stored_encoder_names(s: Session) -> set[str]:
    """Distinct non-empty encoder names recorded on jobs, candidates and transcripts."""
    names: set[str] = set()
    for col in (Job.encoder_name, Candidate.encoder_name, VideoTranscript.encoder_name):
        names.update(n for n in s.scalars(select(col).distinct()) if n)
    return names


def rebuild_vector_index(reembed: bool = False, progress: Callable[[int, int, str], None] | None = None) -> int:
    """Rebuild FAISS from stored chunk embeddings (or re-encode every text first).

    The new index is built off to the side and swapped in atomically at the end. If any job was
    embedded with a different encoder than the active one, re-embedding is forced: copying old-space
    vectors into an index labelled with the new encoder would be silently wrong.
    """
    encoder = get_encoder()
    store = get_vector_store()
    with session_scope() as s:
        foreign = any(n != encoder.name for n in _stored_encoder_names(s))
        if encoder.is_fallback and foreign:
            # re-embedding with hashing would overwrite good embeddings; copying them would mix spaces
            raise EmbeddingModelUnavailable(FALLBACK_REFUSAL)
        if not reembed and foreign:
            log.warning("Encoder changed since embeddings were stored; forcing re-embedding")
            reembed = True
        if reembed:
            jobs = list(s.scalars(select(Job)))
            for job in jobs:
                profile = parse_job_description(job.description, job.title, job.custom_skills)
                jd_vec, resp_vec = embed_job(profile, encoder)
                job.jd_embedding, job.resp_embedding, job.encoder_name = to_bytes(jd_vec), to_bytes(resp_vec), encoder.name
        chunks = list(s.scalars(select(VectorChunk).order_by(VectorChunk.candidate_id, VectorChunk.id)))
        total = len(chunks)
        batch: list[tuple[int, np.ndarray]] = []
        for start in range(0, total, 256):
            part = chunks[start : start + 256]
            if progress:
                progress(min(start + 256, total), total, "chunks")
            if reembed:
                vecs = encoder.encode([c.text for c in part])
                for c, v in zip(part, vecs):
                    c.embedding = to_bytes(v)
            batch.extend((c.id, from_bytes(c.embedding)) for c in part if c.embedding)
        if reembed:
            by_cand: dict[int, list[np.ndarray]] = {}
            by_video: dict[int, list[np.ndarray]] = {}
            for c in chunks:
                (by_video if c.source == "video" else by_cand).setdefault(c.candidate_id, []).append(from_bytes(c.embedding))
            for cand in s.scalars(select(Candidate)):
                for target, attr, pool in ((cand, "resume_embedding", by_cand), (cand.video, "embedding", by_video)):
                    vecs = pool.get(cand.id)
                    if target is not None and vecs:
                        mean = np.mean(vecs, axis=0)
                        setattr(target, attr, to_bytes(mean / (np.linalg.norm(mean) or 1.0)))
                        target.encoder_name = encoder.name
                    elif target is not None and _is_null_vector(getattr(target, attr)):
                        # no chunks: the stored vector is empty/zero (same in every space), so stamping
                        # it keeps it from looking "foreign" and forcing a re-embed on every rebuild
                        target.encoder_name = encoder.name
    # DB committed: now swap the index in one step (old index stays intact if building failed above)
    store.replace_all([p[0] for p in batch], np.vstack([p[1] for p in batch]) if batch else None)
    if reembed:
        for job in list_jobs():
            rescore_job(job["id"])
    return len(batch)
