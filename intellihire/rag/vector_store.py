"""Vector Database Service for RAG.

Default backend is a persisted FAISS `IndexIDMap2(IndexFlatIP)` - exact inner
product search on unit vectors, i.e. cosine similarity. Vector ids are the
primary keys of `vector_chunks` rows in MySQL, so the chunk text and metadata
live in the relational store and FAISS holds only numbers.

`VECTOR_BACKEND=chroma` switches to a persistent ChromaDB collection.
If neither library is installed, a NumPy store with the same interface is used.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import numpy as np

from intellihire.config import get_settings

log = logging.getLogger(__name__)


class VectorStore:
    backend = "base"

    def add(self, ids: list[int], vectors: np.ndarray) -> None: ...
    def remove(self, ids: list[int]) -> None: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...

    def replace_all(self, ids: list[int], vectors: np.ndarray | None) -> None:
        """Atomically replace the whole index content (used by rebuilds); clears `stale`."""
        self.reset()
        if ids:
            self.add(ids, vectors)

    def search(self, query: np.ndarray, k: int = 5, allowed_ids: list[int] | None = None) -> list[tuple[int, float]]:
        ...


def _top_k(ids: np.ndarray, scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    order = np.argsort(-scores)[:k]
    return [(int(ids[i]), float(scores[i])) for i in order]


class FaissStore(VectorStore):
    backend = "faiss"

    def __init__(self, directory: Path, dim: int, encoder_name: str):
        import faiss

        self._faiss = faiss
        self._path = directory / "faiss.index"
        self._meta_path = directory / "faiss.meta.json"
        self._lock = threading.Lock()
        self.dim = dim
        self.encoder_name = encoder_name
        self.stale = False
        self._index = self._new_index()
        if self._path.exists():
            try:
                index = faiss.read_index(str(self._path))
                meta = json.loads(self._meta_path.read_text()) if self._meta_path.exists() else {}
                if index.d != dim or meta.get("encoder") != encoder_name or meta.get("stale"):
                    log.warning("FAISS index was built with a different encoder; it must be rebuilt")
                    self.stale = True
                else:
                    self._index = index
            except Exception as exc:  # corrupt / partially written index: rebuild from the database
                log.error("FAISS index unreadable (%s); marked stale - rebuild it from the Admin page", exc)
                self.stale = True

    def _new_index(self):
        return self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(self.dim))

    def _save(self) -> None:
        if self.stale:  # never persist new data over an index that still needs a rebuild
            return
        self._write(self._index)

    def _write(self, index) -> None:
        tmp, meta_tmp = self._path.with_name(self._path.name + ".tmp"), self._meta_path.with_name(self._meta_path.name + ".tmp")
        self._faiss.write_index(index, str(tmp))
        meta_tmp.write_text(json.dumps({"encoder": self.encoder_name, "dim": self.dim}))
        os.replace(tmp, self._path)
        os.replace(meta_tmp, self._meta_path)

    def replace_all(self, ids, vectors):
        new = self._new_index()
        if len(ids):
            new.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), np.asarray(ids, dtype=np.int64))
        with self._lock:
            self._write(new)  # atomic on disk first; memory swaps only if that succeeded
            self._index = new
            self.stale = False

    def add(self, ids, vectors):
        if not len(ids):
            return
        with self._lock:
            id_arr = np.asarray(ids, dtype=np.int64)
            self._index.remove_ids(id_arr)  # make re-adds idempotent
            self._index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), id_arr)
            self._save()

    def remove(self, ids):
        if not ids:
            return
        with self._lock:
            self._index.remove_ids(np.asarray(ids, dtype=np.int64))
            self._save()

    def count(self):
        return int(self._index.ntotal)

    def reset(self):
        with self._lock:
            self._index = self._new_index()
            self.stale = False
            self._save()

    def search(self, query, k=5, allowed_ids=None):
        if self._index.ntotal == 0:
            return []
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        with self._lock:
            if allowed_ids is not None:
                # Exact scoped search: reconstruct only the allowed vectors.
                present, vectors = [], []
                for vid in allowed_ids:
                    try:
                        vectors.append(self._index.reconstruct(int(vid)))
                        present.append(vid)
                    except RuntimeError:
                        continue
                if not present:
                    return []
                scores = np.vstack(vectors) @ q[0]
                return _top_k(np.asarray(present), scores, k)
            scores, ids = self._index.search(q, min(k, int(self._index.ntotal)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


class ChromaStore(VectorStore):  # pragma: no cover - optional backend
    backend = "chroma"

    def __init__(self, directory: Path, dim: int, encoder_name: str):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(directory / "chroma"))
        self._name = f"intellihire_{encoder_name.replace('/', '_').replace('.', '_')}"[:60]
        self._col = self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})
        self.dim = dim
        self.stale = False

    def add(self, ids, vectors):
        if len(ids):
            self._col.upsert(ids=[str(i) for i in ids], embeddings=np.asarray(vectors, dtype=np.float32).tolist())

    def remove(self, ids):
        if ids:
            self._col.delete(ids=[str(i) for i in ids])

    def count(self):
        return int(self._col.count())

    def reset(self):
        self._client.delete_collection(self._name)
        self._col = self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})

    def search(self, query, k=5, allowed_ids=None):
        if self.count() == 0:
            return []
        if allowed_ids is not None:
            got = self._col.get(ids=[str(i) for i in allowed_ids], include=["embeddings"])
            if not got["ids"]:
                return []
            scores = np.asarray(got["embeddings"], dtype=np.float32) @ query.astype(np.float32)
            return _top_k(np.asarray([int(i) for i in got["ids"]]), scores, k)
        res = self._col.query(query_embeddings=[query.tolist()], n_results=min(k, self.count()))
        return [(int(i), 1.0 - float(d)) for i, d in zip(res["ids"][0], res["distances"][0])]


class NumpyStore(VectorStore):
    """Dependency-free fallback with the same behaviour as the FAISS flat index."""

    backend = "numpy"

    def __init__(self, directory: Path | None, dim: int, encoder_name: str):
        self._path = directory / "numpy_store.npz" if directory else None
        self._lock = threading.Lock()
        self.dim = dim
        self.encoder_name = encoder_name
        self.stale = False
        self._vectors: dict[int, np.ndarray] = {}
        if self._path and self._path.exists():
            try:
                data = np.load(self._path, allow_pickle=False)
                if str(data["encoder"]) == encoder_name and data["vectors"].shape[-1] == dim:
                    self._vectors = {int(i): v for i, v in zip(data["ids"], data["vectors"])}
                else:
                    self.stale = True
            except Exception as exc:
                log.error("NumPy vector store unreadable (%s); marked stale - rebuild it from the Admin page", exc)
                self.stale = True

    def _save(self):
        if not self._path or self.stale:
            return
        ids = np.asarray(list(self._vectors), dtype=np.int64)
        vecs = np.vstack(list(self._vectors.values())) if self._vectors else np.zeros((0, self.dim), np.float32)
        tmp = self._path.with_name("numpy_store.tmp.npz")
        np.savez(tmp, ids=ids, vectors=vecs, encoder=np.asarray(self.encoder_name))
        os.replace(tmp, self._path)

    def replace_all(self, ids, vectors):
        with self._lock:
            old = (self._vectors, self.stale)
            self._vectors = {int(i): v for i, v in zip(ids, np.asarray(vectors, dtype=np.float32))} if len(ids) else {}
            self.stale = False
            try:
                self._save()
            except Exception:
                self._vectors, self.stale = old
                raise

    def add(self, ids, vectors):
        with self._lock:
            for i, v in zip(ids, np.asarray(vectors, dtype=np.float32)):
                self._vectors[int(i)] = v
            self._save()

    def remove(self, ids):
        with self._lock:
            for i in ids:
                self._vectors.pop(int(i), None)
            self._save()

    def count(self):
        return len(self._vectors)

    def reset(self):
        with self._lock:
            self._vectors = {}
            self.stale = False
            self._save()

    def search(self, query, k=5, allowed_ids=None):
        with self._lock:
            pool = dict(self._vectors) if allowed_ids is None else {
                i: self._vectors[i] for i in allowed_ids if i in self._vectors}
        if not pool:
            return []
        ids = np.asarray(list(pool))
        scores = np.vstack(list(pool.values())) @ query.astype(np.float32)
        return _top_k(ids, scores, k)


_store_lock = threading.Lock()
_store: VectorStore | None = None
_store_key: tuple | None = None


def get_vector_store() -> VectorStore:
    """Cached per (backend, encoder): if the encoder changes (e.g. fallback -> real model)
    the store is reopened and the on-disk meta check marks it stale."""
    global _store, _store_key
    from intellihire.embeddings.encoder import get_encoder

    settings = get_settings()
    encoder = get_encoder()
    key = (settings.vector_backend, str(settings.vector_dir), encoder.name, encoder.dim)
    with _store_lock:
        if _store is not None and _store_key == key:
            return _store
        _store, _store_key = _open_store(settings, encoder), key
        return _store


def _open_store(settings, encoder) -> VectorStore:
    if settings.vector_backend == "chroma":
        try:
            return ChromaStore(settings.vector_dir, encoder.dim, encoder.name)
        except ImportError:
            log.warning("chromadb not installed; falling back to FAISS")
    try:
        return FaissStore(settings.vector_dir, encoder.dim, encoder.name)
    except ImportError:
        log.warning("faiss not installed; using NumPy vector store")
        return NumpyStore(settings.vector_dir, encoder.dim, encoder.name)


def _reset_store_cache() -> None:
    global _store, _store_key
    with _store_lock:
        _store, _store_key = None, None


get_vector_store.cache_clear = _reset_store_cache  # type: ignore[attr-defined]
