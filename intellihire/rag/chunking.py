"""Split resumes and transcripts into overlapping, line-aligned chunks for
embedding and retrieval."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str  # "resume" | "video"
    index: int
    start_sec: float | None = None
    end_sec: float | None = None


def chunk_text(text: str, source: str = "resume", max_words: int = 120, overlap_words: int = 25) -> list[Chunk]:
    """Greedy line packing: lines are kept whole where possible; long lines are split."""
    pieces: list[str] = []
    for line in text.split("\n"):
        words = line.split()
        while len(words) > max_words:
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words - overlap_words :]
        if words:
            pieces.append(" ".join(words))

    chunks: list[Chunk] = []
    current: list[str] = []
    count = 0
    carried = 0  # how many leading entries of `current` are overlap from the previous chunk
    for piece in pieces:
        n = len(piece.split())
        if current and count + n > max_words:
            if len(current) == carried:  # only overlap left: drop it rather than emit a duplicate
                current, count, carried = [], 0, 0
                current.append(piece)
                count += n
                continue
            chunks.append(Chunk("\n".join(current), source, len(chunks)))
            # carry (at most `overlap_words` of) the tail forward as overlap
            tail: list[str] = []
            tail_count = 0
            for prev in reversed(current):
                room = overlap_words - tail_count
                if room <= 0:
                    break
                prev_words = prev.split()
                if len(prev_words) > room:
                    if tail:  # whole pieces only; trim just when a single piece is longer than the overlap
                        break
                    prev, prev_words = " ".join(prev_words[-room:]), prev_words[-room:]
                tail.insert(0, prev)
                tail_count += len(prev_words)
            while tail and tail_count + n > max_words:  # never let overlap push the chunk past the cap
                tail_count -= len(tail.pop(0).split())
            current, count = tail, tail_count
            carried = len(current)
        current.append(piece)
        count += n
    if current and len(current) > carried:
        chunks.append(Chunk("\n".join(current), source, len(chunks)))
    return chunks


def _split_segments(segments: list[dict], max_words: int) -> list[dict]:
    """Split segments longer than max_words, spreading their time span proportionally."""
    out: list[dict] = []
    for seg in segments:
        words = seg["text"].split()
        if len(words) <= max_words:
            out.append(seg)
            continue
        start, end = seg["start"], seg["end"]
        parts = [words[i : i + max_words] for i in range(0, len(words), max_words)]
        for k, part in enumerate(parts):
            out.append({**seg, "text": " ".join(part),
                        "start": start + (end - start) * k / len(parts),
                        "end": start + (end - start) * (k + 1) / len(parts)})
    return out


def chunk_transcript(segments: list[dict], max_words: int = 80) -> list[Chunk]:
    """Group time-stamped Whisper segments into chunks that keep their timing."""
    chunks: list[Chunk] = []
    buf: list[dict] = []
    count = 0
    for seg in _split_segments(segments, max_words):
        n = len(seg["text"].split())
        if buf and count + n > max_words:
            chunks.append(Chunk(" ".join(s["text"] for s in buf), "video", len(chunks), buf[0]["start"], buf[-1]["end"]))
            buf, count = [], 0
        buf.append(seg)
        count += n
    if buf:
        chunks.append(Chunk(" ".join(s["text"] for s in buf), "video", len(chunks), buf[0]["start"], buf[-1]["end"]))
    return chunks
