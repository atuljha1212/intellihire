"""Download and cache every model locally so the app never waits on the network
at request time (Module 3 milestone): MiniLM, the spaCy pipeline, Whisper.

Usage:  python scripts/download_models.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the status marks below.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intellihire.config import get_settings  # noqa: E402


def main() -> int:
    s = get_settings()
    ok = True
    print(f"Models directory: {s.model_dir}")

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(s.embedding_model, cache_folder=str(s.model_dir / "sentence-transformers"))
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        print(f"✓ Sentence Transformer {s.embedding_model} ({get_dim()}-d)")
    except Exception as exc:
        ok = False
        print(f"✗ Sentence Transformer: {exc}")

    try:
        import spacy

        try:
            spacy.load(s.spacy_model)
            print(f"✓ spaCy {s.spacy_model} already installed")
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", s.spacy_model], check=True)
            print(f"✓ spaCy {s.spacy_model} downloaded")
    except Exception as exc:
        ok = False
        print(f"✗ spaCy: {exc}")

    if s.whisper_mode == "local":
        try:
            import whisper

            whisper.load_model(s.whisper_model, device="cpu", download_root=str(s.model_dir / "whisper"))
            print(f"✓ Whisper '{s.whisper_model}'")
        except Exception as exc:
            ok = False
            print(f"✗ Whisper: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
