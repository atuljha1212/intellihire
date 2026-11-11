"""Component health for the Admin page and dashboard warnings."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComponentStatus:
    name: str
    detail: str
    ok: bool
    degraded: bool = False


def component_statuses() -> list[ComponentStatus]:
    from intellihire.config import get_settings
    from intellihire.db.session import database_label
    from intellihire.embeddings.encoder import get_encoder
    from intellihire.ingestion.documents import HAS_PYMUPDF
    from intellihire.media.transcriber import transcriber_status
    from intellihire.nlp.spacy_pipeline import has_trained_pipeline, nlp_status
    from intellihire.rag.llm import llm_status
    from intellihire.rag.vector_store import get_vector_store
    from intellihire.scoring.ranker import HAS_XGBOOST

    settings = get_settings()
    out: list[ComponentStatus] = []
    out.append(ComponentStatus("Database", database_label(), True, settings.is_sqlite))
    enc = get_encoder()
    out.append(ComponentStatus("Embeddings", f"{enc.name} ({enc.dim}-d)", True, enc.is_fallback))
    store = get_vector_store()
    stale = getattr(store, "stale", False)
    out.append(ComponentStatus("Vector store", f"{store.backend} - {store.count()} vectors" + (" (rebuild needed)" if stale else ""),
                               True, store.backend == "numpy" or stale))
    try:
        from sqlalchemy import select

        from intellihire.db.models import Candidate, Job, VideoTranscript
        from intellihire.db.session import session_scope

        with session_scope() as s:
            mismatched = sum(1 for n in s.scalars(select(Job.encoder_name)) if n and n != enc.name)
            cand_names = list(s.scalars(select(Candidate.encoder_name)))
            vid_names = list(s.scalars(select(VideoTranscript.encoder_name)))
        cand_bad = sum(1 for n in cand_names if n and n != enc.name)
        vid_bad = sum(1 for n in vid_names if n and n != enc.name)
        unknown = sum(1 for n in cand_names + vid_names if not n)
        if cand_bad or vid_bad:
            out.append(ComponentStatus(
                "Candidate embeddings", f"{cand_bad} candidate(s) and {vid_bad} video(s) embedded with a different "
                f"encoder than the active '{enc.name}' - those semantic/video scores are disabled; re-embed", True, True))
        if unknown:
            out.append(ComponentStatus(
                "Embedding provenance", f"{unknown} stored embedding(s) have an unknown encoder (legacy rows, "
                "assumed to match their job)", True, False))
        if mismatched:
            out.append(ComponentStatus(
                "Embedding consistency", f"{mismatched} job(s) embedded with a different encoder - semantic scores "
                "are disabled for them; run 'Rebuild vector index' with re-embedding", True, True))
    except Exception:  # health page must never crash
        pass
    out.append(ComponentStatus("spaCy", nlp_status(), True, not has_trained_pipeline()))
    out.append(ComponentStatus("PDF parser", "PyMuPDF" if HAS_PYMUPDF else "pypdf fallback", True, not HAS_PYMUPDF))
    whisper = transcriber_status()
    out.append(ComponentStatus("Speech-to-text", whisper, not whisper.startswith("unavailable"), "missing" in whisper))
    try:
        from intellihire.media.video import ffmpeg_binary

        ffmpeg = ffmpeg_binary()
        out.append(ComponentStatus("FFmpeg", ffmpeg, True))
    except Exception as exc:
        out.append(ComponentStatus("FFmpeg", str(exc), False))
    out.append(ComponentStatus("ML ranker", "XGBoost + Scikit-learn" if HAS_XGBOOST else "Scikit-learn (XGBoost missing)",
                               True, not HAS_XGBOOST))
    llm = llm_status()
    out.append(ComponentStatus("LLM", llm, True, llm.startswith("offline")))
    return out
