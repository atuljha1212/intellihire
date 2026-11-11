"""Communication signal (S_COMM) from structural transcript metrics only:
speaking pace (words per minute), vocabulary diversity (moving-average
type-token ratio) and filler-word rate. Nothing about voice, accent or
appearance is measured."""
from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass

PACE_IDEAL = (120.0, 165.0)  # comfortable presentation pace, wpm
PACE_LIMITS = (60.0, 230.0)  # score reaches 0 at these extremes
MATTR_RANGE = (0.55, 0.80)  # maps diversity linearly into [0, 1]
MATTR_WINDOW = 40
FILLER_CAP = 0.06  # 6 % fillers -> filler score 0
MIN_WORDS_RELIABLE = 40
MATTR_MIN_WORDS = 50  # below this the diversity figure is too noisy; pace carries the score

_FILLERS_SINGLE = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm"}
_FILLERS_MULTI = ["you know", "i mean", "sort of", "kind of", "like i said", "basically"]


@dataclass
class CommunicationMetrics:
    word_count: int
    speaking_seconds: float
    wpm: float
    mattr: float
    filler_rate: float
    pace_score: float
    diversity_score: float
    filler_score: float
    score: float | None  # None when the signal is unavailable (no countable words or no pace)
    reliable: bool
    notes: list[str]
    diversity_reliable: bool = True  # False when too few words for a meaningful MATTR

    def to_dict(self) -> dict:
        return asdict(self)


def _words(text: str) -> list[str]:
    """Unicode-aware tokeniser: letters and combining marks form words (so accented and
    Devanagari words count), apostrophes inside a word are kept, digits and punctuation split."""
    words: list[str] = []
    current: list[str] = []

    def flush() -> None:
        word = "".join(current).rstrip("'’")
        if word:
            words.append(word)
        current.clear()

    for ch in text.lower():
        if unicodedata.category(ch)[0] in ("L", "M") or (ch in "'’" and current):
            current.append(ch)
        elif current:
            flush()
    if current:
        flush()
    return words


def mattr(words: list[str], window: int = MATTR_WINDOW) -> float:
    """Moving-average type-token ratio (length-robust lexical diversity)."""
    if not words:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)
    ratios = [len(set(words[i : i + window])) / window for i in range(len(words) - window + 1)]
    return sum(ratios) / len(ratios)


def _pace_score(wpm: float) -> float:
    lo, hi = PACE_IDEAL
    min_l, max_l = PACE_LIMITS
    if lo <= wpm <= hi:
        return 1.0
    if wpm < lo:
        return max(0.0, (wpm - min_l) / (lo - min_l))
    return max(0.0, (max_l - wpm) / (max_l - hi))


def communication_metrics(text: str, segments: list[dict] | None, duration_sec: float) -> CommunicationMetrics:
    words = _words(text)
    n = len(words)
    speaking = 0.0
    if segments:
        speaking = sum(max(0.0, float(s["end"]) - float(s["start"])) for s in segments)
    if speaking <= 0:
        speaking = duration_sec
    wpm = n / (speaking / 60.0) if speaking > 0 else 0.0

    lowered = " " + " ".join(words) + " "
    fillers = sum(1 for w in words if w in _FILLERS_SINGLE)
    fillers += sum(lowered.count(f" {p} ") for p in _FILLERS_MULTI)
    filler_rate = fillers / n if n else 0.0

    diversity = mattr(words)
    pace_s = _pace_score(wpm)
    div_s = min(1.0, max(0.0, (diversity - MATTR_RANGE[0]) / (MATTR_RANGE[1] - MATTR_RANGE[0])))
    fill_s = max(0.0, 1.0 - filler_rate / FILLER_CAP)
    diversity_reliable = n >= MATTR_MIN_WORDS
    if diversity_reliable:
        score: float | None = 0.4 * pace_s + 0.4 * div_s + 0.2 * fill_s
    else:  # too short for MATTR and for a meaningful filler rate: pace/vocab only
        score = 0.5 * pace_s + 0.5 * div_s

    notes = []
    if n == 0 or wpm <= 0:
        # No countable words (e.g. non-letter text) or no measurable speaking time: nothing to score.
        score = None
        notes.append("Communication signal unavailable - no countable words or speaking time in the transcript")
    elif wpm < PACE_IDEAL[0]:
        notes.append(f"Pace of {wpm:.0f} wpm is slower than the {PACE_IDEAL[0]:.0f}-{PACE_IDEAL[1]:.0f} wpm benchmark")
    elif wpm > PACE_IDEAL[1]:
        notes.append(f"Pace of {wpm:.0f} wpm is faster than the {PACE_IDEAL[0]:.0f}-{PACE_IDEAL[1]:.0f} wpm benchmark")
    else:
        notes.append(f"Pace of {wpm:.0f} wpm is within the benchmark range")
    notes.append(f"Vocabulary diversity (MATTR) {diversity:.2f}")
    notes.append(f"Filler words: {filler_rate * 100:.1f}% of words")
    if n and not diversity_reliable:
        notes.append(
            f"Only {n} words (fewer than {MATTR_MIN_WORDS}): vocabulary diversity is low-reliability and the "
            "filler rate is excluded from the score"
        )
    reliable = score is not None and n >= MIN_WORDS_RELIABLE
    if score is not None and not reliable:
        notes.append(f"Only {n} words spoken - signal has low confidence")

    return CommunicationMetrics(
        word_count=n,
        speaking_seconds=round(speaking, 1),
        wpm=round(wpm, 1),
        mattr=round(diversity, 3),
        filler_rate=round(filler_rate, 4),
        pace_score=round(pace_s, 3),
        diversity_score=round(div_s, 3),
        filler_score=round(fill_s, 3),
        score=round(score, 3) if score is not None else None,
        reliable=reliable,
        notes=notes,
        diversity_reliable=diversity_reliable,
    )
