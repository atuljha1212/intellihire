"""Semantic Embedding Service: Sentence Transformers all-MiniLM-L6-v2 (384-d).

MiniLM reads at most 256 word-pieces, so long documents are embedded chunk by
chunk and mean-pooled (`embed_document`) instead of being silently truncated.

If the model cannot be loaded (e.g. first run without internet) and
ALLOW_FALLBACKS is on, a deterministic 384-d hashing encoder keeps the app
working; the dashboard shows a warning while it is active.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from intellihire.config import get_settings

log = logging.getLogger(__name__)

EMBEDDING_DIM = 384


class BaseEncoder:
    name: str = "base"
    dim: int = EMBEDDING_DIM
    is_fallback: bool = False
    # Typical cosine range for (unrelated, strongly related) texts, used to
    # calibrate raw cosine similarity into a 0-1 score.
    calibration: tuple[float, float] = (0.15, 0.70)

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def calibrate(self, cosine: float) -> float:
        lo, hi = self.calibration
        return float(min(1.0, max(0.0, (cosine - lo) / (hi - lo))))


class SentenceTransformerEncoder(BaseEncoder):
    calibration = (0.15, 0.70)

    def __init__(self, model_name: str, cache_dir: str):
        from sentence_transformers import SentenceTransformer

        try:  # prefer the local cache: no Hugging Face Hub round-trip on every start
            self._model = SentenceTransformer(model_name, cache_folder=cache_dir, device="cpu", local_files_only=True)
        except Exception as exc:
            log.info("Local-only load of %s failed (%s); trying online", model_name, exc)
            self._model = SentenceTransformer(model_name, cache_folder=cache_dir, device="cpu")
        self._lock = threading.Lock()
        self.name = model_name.split("/")[-1]
        # sentence-transformers >= 5 renamed the method; use whichever exists (no FutureWarning)
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(dim_fn())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            vectors = self._model.encode(
                texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )
        return vectors.astype(np.float32)


class HashingEncoder(BaseEncoder):
    """Offline fallback: hashed lemma uni/bi-grams projected to 384 dims (no semantics)."""

    name = "hashing-fallback-384"
    is_fallback = True
    calibration = (0.10, 0.65)

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            n_features=EMBEDDING_DIM, ngram_range=(1, 2), alternate_sign=False, norm="l2", stop_words="english"
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._vec.transform(texts).toarray().astype(np.float32)


_FALLBACK_RETRY_SECONDS = 300.0
_encoder_lock = threading.Lock()
_encoder: BaseEncoder | None = None
_fallback_since: float = 0.0


def get_encoder() -> BaseEncoder:
    """Real model once loaded (cached); a fallback is only cached for a few minutes
    so a transient load error does not pin the hashing encoder for the process life."""
    global _encoder, _fallback_since
    with _encoder_lock:
        if _encoder is not None and (
            not _encoder.is_fallback or time.monotonic() - _fallback_since < _FALLBACK_RETRY_SECONDS
        ):
            return _encoder
        settings = get_settings()
        try:
            _encoder = SentenceTransformerEncoder(
                settings.embedding_model, str(settings.model_dir / "sentence-transformers")
            )
            log.info("Loaded embedding model %s", settings.embedding_model)
        except Exception as exc:
            if not settings.allow_fallbacks:
                raise
            log.warning("Sentence Transformer unavailable (%s); using hashing fallback encoder", exc)
            if _encoder is None or not _encoder.is_fallback:
                _encoder = HashingEncoder()
            _fallback_since = time.monotonic()
        return _encoder


def _reset_encoder_cache() -> None:
    global _encoder
    with _encoder_lock:
        _encoder = None


get_encoder.cache_clear = _reset_encoder_cache  # type: ignore[attr-defined]  # lru_cache-compatible


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


def embed_document(chunks: list[str], encoder: BaseEncoder | None = None) -> np.ndarray:
    """Mean-pool chunk embeddings into one unit-length document vector."""
    encoder = encoder or get_encoder()
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        return np.zeros(encoder.dim, dtype=np.float32)
    return normalize(encoder.encode(chunks).mean(axis=0))


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def to_bytes(vec: np.ndarray | None) -> bytes | None:
    return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()


def from_bytes(blob: bytes | None) -> np.ndarray | None:
    return None if not blob else np.frombuffer(blob, dtype=np.float32).copy()
