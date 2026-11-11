"""Relational schema (MySQL via SQLAlchemy ORM; SQLite works for local demos).

All queries go through the ORM, so every value is bound as a parameter - no
string-built SQL anywhere (protection against SQL injection).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import LONGTEXT, MEDIUMBLOB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

LongText = Text().with_variant(LONGTEXT(), "mysql")
Blob = LargeBinary().with_variant(MEDIUMBLOB(), "mysql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128), default="")
    email: Mapped[str] = mapped_column(String(255), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="recruiter")  # admin | recruiter
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    department: Mapped[str] = mapped_column(String(120), default="")
    location: Mapped[str] = mapped_column(String(120), default="")
    description: Mapped[str] = mapped_column(LongText)
    required_skills: Mapped[list] = mapped_column(JSON, default=list)
    preferred_skills: Mapped[list] = mapped_column(JSON, default=list)
    custom_skills: Mapped[list] = mapped_column(JSON, default=list)
    min_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    responsibilities: Mapped[str] = mapped_column(LongText, default="")
    education_req: Mapped[str] = mapped_column(String(200), default="")
    key_phrases: Mapped[list] = mapped_column(JSON, default=list)
    weights: Mapped[dict] = mapped_column(JSON, default=dict)
    jd_embedding: Mapped[bytes | None] = mapped_column(Blob, nullable=True)
    resp_embedding: Mapped[bytes | None] = mapped_column(Blob, nullable=True)
    encoder_name: Mapped[str] = mapped_column(String(120), default="")
    status: Mapped[str] = mapped_column(String(16), default="open")  # open | closed
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    candidates: Mapped[list[Candidate]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True
    )


class Candidate(Base):
    __tablename__ = "candidates"
    __table_args__ = (UniqueConstraint("job_id", "file_sha256", name="uq_candidate_job_file"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    email: Mapped[str] = mapped_column(String(255), default="")
    phone: Mapped[str] = mapped_column(String(40), default="")
    links: Mapped[list] = mapped_column(JSON, default=list)
    resume_filename: Mapped[str] = mapped_column(String(200))
    file_sha256: Mapped[str] = mapped_column(String(64))
    resume_text: Mapped[str] = mapped_column(LongText)
    hidden_text: Mapped[str] = mapped_column(LongText, default="")
    parser: Mapped[str] = mapped_column(String(80), default="")
    skills: Mapped[dict] = mapped_column(JSON, default=dict)  # skill -> evidence lines
    experience_years: Mapped[float] = mapped_column(Float, default=0.0)
    experience_method: Mapped[str] = mapped_column(String(40), default="")
    experience_evidence: Mapped[list] = mapped_column(JSON, default=list)
    education: Mapped[list] = mapped_column(JSON, default=list)
    certifications: Mapped[list] = mapped_column(JSON, default=list)
    key_phrases: Mapped[list] = mapped_column(JSON, default=list)
    injection_flags: Mapped[list] = mapped_column(JSON, default=list)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    resume_embedding: Mapped[bytes | None] = mapped_column(Blob, nullable=True)
    encoder_name: Mapped[str | None] = mapped_column(String(128), nullable=True)  # NULL = legacy/unknown
    status: Mapped[str] = mapped_column(String(16), default="new", index=True)
    recruiter_notes: Mapped[str] = mapped_column(Text, default="")
    ai_summary: Mapped[str] = mapped_column(LongText, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    job: Mapped[Job] = relationship(back_populates="candidates")
    video: Mapped[VideoTranscript | None] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", uselist=False, passive_deletes=True
    )
    score: Mapped[Score | None] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", uselist=False, passive_deletes=True
    )
    chunks: Mapped[list[VectorChunk]] = relationship(cascade="all, delete-orphan", passive_deletes=True)


class VideoTranscript(Base):
    __tablename__ = "video_transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), unique=True)
    filename: Mapped[str] = mapped_column(String(200))
    file_sha256: Mapped[str] = mapped_column(String(64), default="")
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    transcript: Mapped[str] = mapped_column(LongText)
    segments: Mapped[list] = mapped_column(JSON, default=list)
    language: Mapped[str] = mapped_column(String(16), default="")
    engine: Mapped[str] = mapped_column(String(60), default="")
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    wpm: Mapped[float] = mapped_column(Float, default=0.0)
    mattr: Mapped[float] = mapped_column(Float, default=0.0)
    filler_rate: Mapped[float] = mapped_column(Float, default=0.0)
    comm_score: Mapped[float] = mapped_column(Float, default=0.0)
    comm_details: Mapped[dict] = mapped_column(JSON, default=dict)
    video_skills: Mapped[list] = mapped_column(JSON, default=list)
    embedding: Mapped[bytes | None] = mapped_column(Blob, nullable=True)
    encoder_name: Mapped[str | None] = mapped_column(String(128), nullable=True)  # NULL = legacy/unknown
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    candidate: Mapped[Candidate] = relationship(back_populates="video")


class Score(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), unique=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    s_semantic: Mapped[float] = mapped_column(Float, default=0.0)
    raw_cosine: Mapped[float] = mapped_column(Float, default=0.0)
    s_skill: Mapped[float | None] = mapped_column(Float, nullable=True)
    s_experience: Mapped[float] = mapped_column(Float, default=0.0)
    s_video: Mapped[float | None] = mapped_column(Float, nullable=True)
    s_comm: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    ml_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    matched_skills: Mapped[list] = mapped_column(JSON, default=list)
    missing_skills: Mapped[list] = mapped_column(JSON, default=list)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    candidate: Mapped[Candidate] = relationship(back_populates="score")


class VectorChunk(Base):
    """Text of each embedded chunk; its primary key is the id stored in FAISS."""

    __tablename__ = "vector_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[int] = mapped_column(Integer, index=True)
    source: Mapped[str] = mapped_column(String(16))  # resume | video
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    start_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(Blob, nullable=True)  # lets the index be rebuilt


class RankerModel(Base):
    __tablename__ = "ranker_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    model_type: Mapped[str] = mapped_column(String(80))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    feature_importance: Mapped[dict] = mapped_column(JSON, default=dict)
    n_samples: Mapped[int] = mapped_column(Integer, default=0)
    path: Mapped[str] = mapped_column(String(400))
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(LongText)
    trace: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(60))
    entity: Mapped[str] = mapped_column(String(40), default="")
    entity_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
