"""Database-free processing: file bytes in, structured + vectorised profiles out.

`pipeline.py` persists these results; tests and the stress test call them directly.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from intellihire.config import get_settings
from intellihire.embeddings.encoder import BaseEncoder, embed_document, get_encoder
from intellihire.ingestion.documents import DocumentParseError, ExtractedDocument, extract_document
from intellihire.ingestion.validation import ValidationResult, validate_document, validate_video
from intellihire.media.communication import CommunicationMetrics, communication_metrics
from intellihire.media.transcriber import Transcript, transcribe
from intellihire.media.video import VideoMetadata, VideoProcessingError, extract_audio, probe_video, wav_duration
from intellihire.nlp.jd_parser import JobProfile, parse_job_description
from intellihire.nlp.resume_parser import ResumeProfile, parse_resume
from intellihire.nlp.skills import extract_skills
from intellihire.rag.chunking import Chunk, chunk_text, chunk_transcript
from intellihire.rag.guard import scan_document, scan_for_injection


class AnalysisError(Exception):
    """A user-facing processing failure (validation, parsing, media)."""


@dataclass
class JobAnalysis:
    profile: JobProfile
    jd_vec: np.ndarray
    resp_vec: np.ndarray
    encoder_name: str


@dataclass
class ResumeAnalysis:
    validation: ValidationResult
    document: ExtractedDocument
    profile: ResumeProfile
    injection_flags: list[dict]
    chunks: list[Chunk]
    chunk_vectors: np.ndarray
    doc_vector: np.ndarray
    warnings: list[str] = field(default_factory=list)
    encoder_name: str = ""  # encoder that produced chunk_vectors/doc_vector


@dataclass
class VideoAnalysis:
    validation: ValidationResult
    metadata: VideoMetadata
    transcript: Transcript
    communication: CommunicationMetrics
    video_skills: list[str]
    chunks: list[Chunk]
    chunk_vectors: np.ndarray
    doc_vector: np.ndarray | None
    injection_flags: list[dict] = field(default_factory=list)
    encoder_name: str = ""  # encoder that produced chunk_vectors/doc_vector


def read_job_document(filename: str, data: bytes) -> str:
    validation = validate_document(filename, data, max_mb=get_settings().max_jd_mb)
    if not validation.ok:
        raise AnalysisError(validation.error_message)
    try:
        return extract_document(validation.safe_name, data).text
    except DocumentParseError as exc:
        raise AnalysisError(str(exc)) from exc


def embed_job(profile: JobProfile, encoder: BaseEncoder | None = None) -> tuple[np.ndarray, np.ndarray]:
    encoder = encoder or get_encoder()
    jd_vec = embed_document([c.text for c in chunk_text(profile.text)], encoder)
    resp_vec = embed_document([c.text for c in chunk_text(profile.responsibilities)], encoder)
    return jd_vec, resp_vec


def analyze_job(text: str, title: str = "", extra_skills: list[str] | None = None) -> JobAnalysis:
    if len(text.strip()) < 30:
        raise AnalysisError("The job description is too short to analyse (min. 30 characters).")
    encoder = get_encoder()
    profile = parse_job_description(text, title, extra_skills)
    jd_vec, resp_vec = embed_job(profile, encoder)
    return JobAnalysis(profile, jd_vec, resp_vec, encoder.name)


def analyze_resume(filename: str, data: bytes, extra_skills: list[str] | None = None) -> ResumeAnalysis:
    validation = validate_document(filename, data)
    if not validation.ok:
        raise AnalysisError(validation.error_message)
    try:
        document = extract_document(validation.safe_name, data)
    except DocumentParseError as exc:
        raise AnalysisError(str(exc)) from exc

    # Only visible text is parsed and scored; hidden text is kept for the audit only.
    profile = parse_resume(document.text, validation.safe_name, extra_skills)
    flags = scan_document(document.text, document.hidden_text)
    encoder = get_encoder()
    chunks = chunk_text(document.text, "resume")
    vectors = encoder.encode([c.text for c in chunks]) if chunks else np.zeros((0, encoder.dim), np.float32)
    doc_vec = (vectors.mean(axis=0) if len(vectors) else np.zeros(encoder.dim, np.float32)).astype(np.float32)
    norm = np.linalg.norm(doc_vec)
    if norm > 0:
        doc_vec = doc_vec / norm
    return ResumeAnalysis(validation, document, profile, flags, chunks, vectors, doc_vec, list(document.warnings),
                          encoder.name)


def analyze_video(filename: str, data: bytes, extra_skills: list[str] | None = None) -> VideoAnalysis:
    settings = get_settings()
    validation = validate_video(filename, data)
    if not validation.ok:
        raise AnalysisError(validation.error_message)
    with tempfile.TemporaryDirectory(dir=settings.upload_dir, ignore_cleanup_errors=True) as tmp:
        video_path = Path(tmp) / validation.safe_name
        video_path.write_bytes(data)
        try:
            meta = probe_video(video_path, settings.max_video_seconds)
            wav = extract_audio(video_path, Path(tmp) / "audio.wav", settings.max_video_seconds)
            if not meta.duration_sec:
                # OpenCV could not report the length: measure it from the extracted audio instead.
                # extract_audio keeps one second beyond the limit, so anything longer is over it.
                meta.duration_sec = round(wav_duration(wav), 2)
                if meta.duration_sec > settings.max_video_seconds + 0.5:
                    raise VideoProcessingError(
                        f"video is longer than {settings.max_video_seconds}s; the limit is {settings.max_video_seconds}s"
                    )
        except VideoProcessingError as exc:
            raise AnalysisError(str(exc)) from exc
        transcript = transcribe(wav, meta.duration_sec)  # TranscriptionUnavailable propagates
    if not transcript.text:
        raise AnalysisError("No speech was detected in the video.")

    segments = transcript.as_dicts()
    comm = communication_metrics(transcript.text, segments, meta.duration_sec)
    if comm.score is None:
        # The pipeline stores a non-null communication score; a transcript with no countable words is no speech.
        raise AnalysisError("No measurable speech was found in the video.")
    encoder = get_encoder()
    chunks = chunk_transcript(segments)
    vectors = encoder.encode([c.text for c in chunks]) if chunks else np.zeros((0, encoder.dim), np.float32)
    doc_vec = embed_document([c.text for c in chunks], encoder) if chunks else None
    video_skills = sorted(extract_skills(transcript.text, tuple(extra_skills or ())))
    # Spoken words are untrusted input for the LLM, so they get the same injection scan as resumes.
    injection_flags = [f.to_dict() for f in scan_for_injection(transcript.text, "video transcript")]
    return VideoAnalysis(validation, meta, transcript, comm, video_skills, chunks, vectors, doc_vec, injection_flags,
                         encoder.name)
