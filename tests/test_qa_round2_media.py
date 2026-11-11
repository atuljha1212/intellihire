from __future__ import annotations

import io
import subprocess

import pytest

from intellihire.ingestion import documents
from intellihire.ingestion.documents import DocumentParseError, extract_docx
from intellihire.media import transcriber, video
from intellihire.media.communication import communication_metrics
from intellihire.media.video import VideoProcessingError, probe_video

from conftest import FFMPEG  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def _pdf(pages: int = 3) -> bytes:
    doc = documents.fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"Page {i} experience with Python and SQL")
    return doc.tobytes()


def test_t41_pdf_deadline(monkeypatch):
    monkeypatch.setattr(documents, "PDF_PROCESSING_SECONDS", -1.0)
    with pytest.raises(DocumentParseError, match="too long"):
        documents.extract_pdf(_pdf())


def test_t41_pdf_deadline_pypdf(monkeypatch):
    pytest.importorskip("pypdf")
    monkeypatch.setattr(documents, "PDF_PROCESSING_SECONDS", -1.0)
    with pytest.raises(DocumentParseError, match="too long"):
        documents._extract_pdf_pypdf(_pdf())


def _docx(paragraphs: list[str]) -> bytes:
    import docx

    d = docx.Document()
    for t in paragraphs:
        d.add_paragraph(t)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_t41_docx_paragraph_cap(monkeypatch):
    monkeypatch.setattr(documents, "MAX_DOCX_PARAGRAPHS", 5)
    with pytest.raises(DocumentParseError, match="paragraphs"):
        extract_docx(_docx(["x"] * 10))
    assert extract_docx(_docx(["hello world"] * 3)).text


def test_t41_docx_char_cap(monkeypatch):
    monkeypatch.setattr(documents, "MAX_DOCX_CHARS", 100)
    with pytest.raises(DocumentParseError, match="characters"):
        extract_docx(_docx(["a" * 60, "b" * 60]))


def test_t41_analysis_catches_parse_error():
    import intellihire.services.analysis as an

    assert an.DocumentParseError is DocumentParseError


@needs_ffmpeg
def test_t47_ffmpeg_fallback_when_opencv_fails(tmp_path, monkeypatch):
    v = tmp_path / "a.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=160x120:d=3",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(v)], check=True)
    mute = tmp_path / "m.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=160x120:d=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mute)], check=True)

    class Dead:
        def __init__(self, *_): pass
        def isOpened(self): return False
        def release(self): pass

    monkeypatch.setattr(video.cv2, "VideoCapture", Dead)
    meta = probe_video(v, 60)
    assert meta.duration_sec == pytest.approx(3, abs=0.3) and meta.width == 160
    with pytest.raises(VideoProcessingError, match="limit"):
        probe_video(v, 1)
    with pytest.raises(VideoProcessingError, match="no audio"):
        probe_video(mute, 60)
    junk = tmp_path / "j.mp4"
    junk.write_bytes(b"not a video at all" * 10)
    with pytest.raises(VideoProcessingError, match="could not be read"):
        probe_video(junk, 60)


def test_t61_initial_prompt_local(monkeypatch):
    seen = {}

    class FakeModel:
        def transcribe(self, audio, **kw):
            seen.update(kw)
            return {"segments": [], "language": "en"}

    monkeypatch.setattr(transcriber, "_local_model", lambda: FakeModel())
    monkeypatch.setattr(transcriber, "load_wav_16k", lambda p: [0.0])
    monkeypatch.delenv("WHISPER_INITIAL_PROMPT", raising=False)
    transcriber._transcribe_local("x.wav")
    assert "Umm" in seen["initial_prompt"]
    seen.clear()
    monkeypatch.setenv("WHISPER_INITIAL_PROMPT", "")
    transcriber._transcribe_local("x.wav")
    assert "initial_prompt" not in seen


def test_t61_short_transcript_weights():
    text = "I built search systems and ranking models for recruiting teams every day"  # 12 words
    m = communication_metrics(text, [{"start": 0, "end": 5, "text": text}], 5)
    assert not m.diversity_reliable
    assert m.score == pytest.approx(round(0.5 * m.pace_score + 0.5 * m.diversity_score, 3), abs=0.002)
    assert any("low-reliability" in n for n in m.notes)


def test_t61_long_transcript_keeps_filler_weight():
    words = " ".join(f"word{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(60))
    m = communication_metrics(words, [{"start": 0, "end": 25, "text": ""}], 25)
    assert m.diversity_reliable
    assert m.score == pytest.approx(0.4 * m.pace_score + 0.4 * m.diversity_score + 0.2 * m.filler_score, abs=0.002)
