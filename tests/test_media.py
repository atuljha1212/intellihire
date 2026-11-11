from __future__ import annotations

import subprocess

import pytest

from intellihire.media.communication import communication_metrics, mattr
from intellihire.media.transcriber import Segment, clean_segments, load_wav_16k
from intellihire.media.video import VideoProcessingError, extract_audio, probe_video

from conftest import FFMPEG  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def _make_video(path, audio: bool, seconds: int = 3):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
           f"color=c=gray:s=160x120:d={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


@needs_ffmpeg
def test_probe_and_extract_audio(tmp_path):
    video = tmp_path / "intro.mp4"
    _make_video(video, audio=True)
    meta = probe_video(video, max_seconds=60)
    assert meta.duration_sec == pytest.approx(3, abs=0.2) and meta.width == 160
    wav = extract_audio(video, tmp_path / "a.wav", max_seconds=60)
    audio = load_wav_16k(wav)
    assert len(audio) == pytest.approx(3 * 16000, rel=0.05)
    assert audio.dtype.name == "float32" and abs(audio).max() > 0.05


@needs_ffmpeg
def test_silent_video_and_length_limit(tmp_path):
    video = tmp_path / "mute.mp4"
    _make_video(video, audio=False)
    with pytest.raises(VideoProcessingError, match="no audio"):
        extract_audio(video, tmp_path / "a.wav", max_seconds=60)
    with pytest.raises(VideoProcessingError, match="limit"):
        probe_video(video, max_seconds=1)


def test_clean_segments_removes_noise_and_loops():
    segs = [Segment(0, 1, " [Music] "), Segment(1, 2, "Hello, I am Jane."), Segment(2, 3, "hello, i am jane."),
            Segment(3, 4, "I build ML systems.")]
    cleaned = clean_segments(segs)
    assert [s.text for s in cleaned] == ["Hello, I am Jane.", "I build ML systems."]


def test_communication_metrics():
    text = ("Hi, I am a machine learning engineer. Over four years I built search systems, ranking models and data "
            "pipelines for recruiting products, and I enjoy explaining complex models to non technical teams. "
            "Recently I deployed transformer embeddings with vector indexes, reducing latency and improving relevance.")
    words = len(text.split())
    seconds = words / 140 * 60  # 140 wpm: inside the benchmark
    m = communication_metrics(text, [{"start": 0, "end": seconds, "text": text}], seconds)
    assert m.wpm == pytest.approx(140, abs=3)
    assert m.pace_score == 1.0 and m.filler_rate == 0 and m.reliable
    slow = communication_metrics("um uh so um I um like uh code", [{"start": 0, "end": 20, "text": ""}], 20)
    assert slow.filler_rate > 0.3 and slow.score < m.score and not slow.reliable


def test_mattr_is_length_robust():
    assert mattr("a b c d".split()) == 1.0
    assert mattr(["word"] * 100) == pytest.approx(1 / 40)
