"""Multimodal file ingest: OpenCV validates the container and reads its metadata;
the audio track is exported to 16 kHz mono WAV for Whisper.

Ethics by design: no video frame is ever decoded for analysis. Faces,
appearance and background are never seen by the scoring pipeline - only speech.

Note on tooling: OpenCV's VideoCapture exposes only the image stream, so the
audio track is demuxed with FFmpeg (the same binary Whisper itself requires).
The `imageio-ffmpeg` wheel ships a static FFmpeg, so no system install is needed.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Raised for unreadable, too long or silent videos."""


@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


def ffmpeg_binary() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depends on environment
        raise VideoProcessingError("FFmpeg not found. Install it or `pip install imageio-ffmpeg`.") from exc


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_RE = re.compile(r"Stream #.*Video:.*?,\s*(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"([\d.]+)\s*fps")


def _probe_with_ffmpeg(path: str | Path, timeout: int = 30) -> VideoMetadata:
    """Fallback probe when OpenCV cannot open the file: ask FFmpeg (it supports far more codecs)."""
    try:
        proc = subprocess.run(
            [ffmpeg_binary(), "-hide_banner", "-nostdin", "-i", str(path)],
            capture_output=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoProcessingError("the video could not be read (probe timed out)") from exc
    except OSError as exc:
        raise VideoProcessingError(f"the video could not be read: {exc}") from exc
    err = proc.stderr or ""  # ffmpeg exits non-zero here (no output file) but still prints stream info
    m = _DURATION_RE.search(err)
    if not m:
        raise VideoProcessingError("the video could not be read (corrupted or unsupported format)")
    if not re.search(r"Stream #.*Audio:", err):
        raise VideoProcessingError("the video has no audio track")
    duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    vm, fm = _VIDEO_RE.search(err), _FPS_RE.search(err)
    fps = float(fm.group(1)) if fm else 0.0
    width, height = (int(vm.group(1)), int(vm.group(2))) if vm else (0, 0)
    return VideoMetadata(fps, int(duration * fps) if fps else 0, width, height, round(duration, 2))


def probe_video(path: str | Path, max_seconds: int) -> VideoMetadata:
    """Read container metadata only (no frames): OpenCV first, FFmpeg when OpenCV cannot open the file."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            meta = _probe_with_ffmpeg(path)
            if meta.duration_sec > max_seconds:
                raise VideoProcessingError(f"video is {meta.duration_sec:.0f}s long; the limit is {max_seconds}s")
            return meta
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    if duration > max_seconds:
        raise VideoProcessingError(f"video is {duration:.0f}s long; the limit is {max_seconds}s")
    return VideoMetadata(fps, frames, width, height, round(duration, 2))


def extract_audio(video_path: str | Path, wav_path: str | Path, max_seconds: int, timeout: int = 300) -> Path:
    """Drop the video stream and write 16 kHz mono PCM WAV (Whisper's native format)."""
    wav_path = Path(wav_path)
    cmd = [
        ffmpeg_binary(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-vn",  # discard all video frames
        "-map", "0:a:0?",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        # One second beyond the limit: a longer video is then detectable from the WAV duration.
        "-t", str(max_seconds + 1),
        str(wav_path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoProcessingError("audio extraction timed out") from exc
    if proc.returncode != 0:
        if "does not contain any stream" in proc.stderr:
            raise VideoProcessingError("the video has no audio track")
        raise VideoProcessingError(f"FFmpeg could not read the audio: {proc.stderr.strip()[:300]}")
    if not wav_path.exists() or wav_path.stat().st_size <= 44:  # 44 bytes = empty WAV header
        raise VideoProcessingError("the video has no audio track")
    return wav_path


def wav_duration(wav_path: str | Path) -> float:
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate() or 1)
