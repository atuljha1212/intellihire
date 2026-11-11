"""Grounded retrieval: recruiter question -> query vector -> FAISS search scoped
to one job (optionally to specific candidates) -> sanitised evidence chunks."""
from __future__ import annotations

import re

import numpy as np
from sqlalchemy import select

from intellihire.db.models import Candidate, VectorChunk
from intellihire.db.session import session_scope
from intellihire.embeddings.encoder import from_bytes, get_encoder
from intellihire.rag.guard import safe_name, sanitize_for_llm
from intellihire.rag.vector_store import get_vector_store


def candidate_label(cid: int, name: str, blind: bool) -> str:
    # The name can be applicant-controlled (e.g. derived from the upload filename),
    # so it is scanned and sanitised before it reaches a prompt or tool result.
    return f"Candidate C{cid}" if blind else f"{safe_name(name)} (C{cid})"


def redact_name(text: str, name: str) -> str:
    """Blind mode: remove the candidate's full name and each name part from text."""
    parts = [p for p in re.split(r"\s+", (name or "").strip()) if len(p) > 1]
    if not parts:
        return text
    for target in [" ".join(parts)] + sorted(parts, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(target)}(?!\w)", "[name]", text, flags=re.I)
    return re.sub(r"\[name\](?:\s+\[name\])+", "[name]", text)


def retrieve(job_id: int, query: str, candidate_ids: list[int] | None = None, k: int = 6,
             blind: bool = False, source: str | None = None) -> list[dict]:
    with session_scope() as s:
        q = select(VectorChunk.id).where(VectorChunk.job_id == job_id)
        if candidate_ids:
            q = q.where(VectorChunk.candidate_id.in_(candidate_ids))
        if source:
            q = q.where(VectorChunk.source == source)
        allowed = list(s.scalars(q))
    if not allowed:
        return []

    qvec = get_encoder().encode([query])[0]
    hits = get_vector_store().search(qvec, k=k, allowed_ids=allowed)
    if not hits:  # index empty or stale: fall back to the embeddings stored in the DB
        with session_scope() as s:
            rows = s.execute(select(VectorChunk.id, VectorChunk.embedding).where(VectorChunk.id.in_(allowed))).all()
        rows = [(i, from_bytes(e)) for i, e in rows if e]
        if rows:
            ids = np.asarray([r[0] for r in rows])
            scores = np.vstack([r[1] for r in rows]) @ qvec
            order = np.argsort(-scores)[:k]
            hits = [(int(ids[i]), float(scores[i])) for i in order]

    score_by_id = dict(hits)
    with session_scope() as s:
        rows = s.execute(
            select(VectorChunk, Candidate.name)
            .join(Candidate, Candidate.id == VectorChunk.candidate_id)
            .where(VectorChunk.id.in_(list(score_by_id)))
        ).all()
    evidence = []
    for chunk, name in sorted(rows, key=lambda r: -score_by_id[r[0].id]):
        evidence.append({
            "chunk_id": chunk.id,
            "candidate_id": chunk.candidate_id,
            "candidate": candidate_label(chunk.candidate_id, name, blind),
            "source": chunk.source,
            "text": redact_name(sanitize_for_llm(chunk.text), name) if blind else sanitize_for_llm(chunk.text),
            "score": round(score_by_id[chunk.id], 4),
            "start_sec": chunk.start_sec,
            "end_sec": chunk.end_sec,
        })
    for i, e in enumerate(evidence, start=1):
        e["ref"] = f"E{i}"
    return evidence
