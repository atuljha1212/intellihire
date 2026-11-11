"""Speech-to-text with OpenAI Whisper - a local model by default, or the hosted
Whisper API (WHISPER_MODE=api). Produces a cleaned, time-stamped transcript."""
from __future__ import annotations

import logging
import os
import re
import wave
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from intellihire.config import get_settings

log = logging.getLogger(__name__)


class TranscriptionUnavailable(Exception):
    """Raised when no speech-to-text backend is configured or installed."""


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    duration_sec: float = 0.0
    engine: str = ""

    def as_dicts(self) -> list[dict]:
        return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text} for s in self.segments]


# Whisper normally drops "um"/"uh". Priming it with disfluent text keeps them in the transcript (documented
# technique), which the filler-rate metric needs. Set WHISPER_INITIAL_PROMPT to override; empty disables it.
DEFAULT_INITIAL_PROMPT = "Umm, let me think, like, hmm... Okay, so, uh, I mean, you know."


def _initial_prompt() -> str | None:
    value = os.environ.get("WHISPER_INITIAL_PROMPT")
    value = DEFAULT_INITIAL_PROMPT if value is None else value.strip()
    return value or None


_NOISE = re.compile(r"\[(?:music|applause|laughter|silence|noise|inaudible|blank_audio)\]|\((?:music|laughs?)\)", re.I)


def clean_segments(segments: list[Segment]) -> list[Segment]:
    """Remove noise tags, empty segments and Whisper's repeated-phrase loops."""
    cleaned: list[Segment] = []
    for seg in segments:
        text = _NOISE.sub("", seg.text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if cleaned and text.lower() == cleaned[-1].text.lower():
            continue  # hallucinated repetition
        cleaned.append(Segment(seg.start, seg.end, text))
    return cleaned


_FILLER_WORDS = {
    "um", "umm", "uh", "uhh", "er", "erm", "hmm", "hm", "mm", "mmm", "ah", "like", "so", "okay", "ok",
    "i", "mean", "you", "know", "let", "me", "think", "well", "yeah",
}


def _norm_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z']+", text.lower()))


def drop_prompt_echo(segments: list[Segment], prompt: str | None = None) -> list[Segment]:
    """Drop segments that merely echo Whisper's disfluency initial_prompt (a known leak on near-silent
    audio) or consist solely of filler words, so they cannot inflate the filler rate."""
    norm_prompt = _norm_words(DEFAULT_INITIAL_PROMPT if prompt is None else prompt)
    kept: list[Segment] = []
    for seg in segments:
        norm = _norm_words(seg.text)
        words = norm.split()
        if not words:
            continue
        if norm_prompt and (norm in norm_prompt or SequenceMatcher(None, norm, norm_prompt).ratio() >= 0.8):
            continue
        if len(words) >= 3 and all(w in _FILLER_WORDS for w in words):
            continue
        kept.append(seg)
    return kept


def load_wav_16k(path: str | Path) -> np.ndarray:
    """Read a 16-bit mono 16 kHz WAV as float32 in [-1, 1] (Whisper's input)."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM WAV")
        frames = wf.readframes(wf.getnframes())
        channels = wf.getnchannels()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


@lru_cache(maxsize=1)
def _local_model():
    try:
        import whisper
    except ImportError as exc:
        raise TranscriptionUnavailable(
            "openai-whisper is not installed (pip install openai-whisper) - or set WHISPER_MODE=api"
        ) from exc
    settings = get_settings()
    return whisper.load_model(settings.whisper_model, device="cpu", download_root=str(settings.model_dir / "whisper"))


def _is_hallucination(raw: dict) -> bool:
    """Standard Whisper heuristic: a segment the model itself rates as probably silence (high no_speech_prob)
    and decoded with low confidence (low avg_logprob) is usually a hallucination on quiet audio."""
    no_speech = raw.get("no_speech_prob")
    avg_logprob = raw.get("avg_logprob")
    if no_speech is None or avg_logprob is None:
        return False
    return float(no_speech) > 0.6 and float(avg_logprob) < -1.0


def _transcribe_local(wav_path: Path) -> Transcript:
    model = _local_model()
    # Passing the array (not the path) means Whisper does not need ffmpeg on PATH.
    kwargs = {"initial_prompt": _initial_prompt()} if _initial_prompt() else {}
    result = model.transcribe(
        load_wav_16k(wav_path), fp16=False, verbose=None, condition_on_previous_text=False, **kwargs
    )
    segments = [
        Segment(float(s["start"]), float(s["end"]), s["text"])
        for s in result.get("segments", [])
        if not _is_hallucination(s)
    ]
    return Transcript(
        text="",
        segments=segments,
        language=result.get("language", ""),
        engine=f"whisper-local:{get_settings().whisper_model}",
    )


def _transcribe_api(wav_path: Path) -> Transcript:
    settings = get_settings()
    if not settings.openai_api_key:
        raise TranscriptionUnavailable("WHISPER_MODE=api requires OPENAI_API_KEY")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise TranscriptionUnavailable("the openai package is not installed") from exc
    client = OpenAI(api_key=settings.openai_api_key)
    kwargs = {"prompt": _initial_prompt()} if _initial_prompt() else {}
    with open(wav_path, "rb") as fh:
        resp = client.audio.transcriptions.create(
            model="whisper-1", file=fh, response_format="verbose_json", timestamp_granularities=["segment"],
            **kwargs,
        )
    raw_segments = getattr(resp, "segments", None) or []
    segments = [
        Segment(float(getattr(s, "start", 0.0)), float(getattr(s, "end", 0.0)), getattr(s, "text", ""))
        for s in raw_segments
    ]
    if not segments and getattr(resp, "text", ""):
        segments = [Segment(0.0, float(getattr(resp, "duration", 0.0) or 0.0), resp.text)]
    return Transcript(text="", segments=segments, language=getattr(resp, "language", "") or "", engine="whisper-api")


def transcribe(wav_path: str | Path, duration_sec: float = 0.0) -> Transcript:
    settings = get_settings()
    wav_path = Path(wav_path)
    transcript = _transcribe_api(wav_path) if settings.whisper_mode == "api" else _transcribe_local(wav_path)
    transcript.segments = clean_segments(transcript.segments)
    transcript.segments = drop_prompt_echo(transcript.segments, _initial_prompt() or DEFAULT_INITIAL_PROMPT)
    transcript.text = " ".join(s.text for s in transcript.segments).strip()
    transcript.duration_sec = duration_sec or (transcript.segments[-1].end if transcript.segments else 0.0)
    return transcript


def transcriber_status() -> str:
    settings = get_settings()
    if settings.whisper_mode == "api":
        return "Whisper API" if settings.openai_api_key else "Whisper API (missing OPENAI_API_KEY)"
    try:
        import whisper  # noqa: F401

        return f"Local Whisper ({settings.whisper_model})"
    except ImportError:
        return "unavailable (pip install openai-whisper)"
