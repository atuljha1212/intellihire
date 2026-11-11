"""Text normalisation (Stage 2): remove non-printable characters, layout
artifacts, broken hyphenation and excess whitespace while keeping line breaks,
which the section detectors rely on."""
from __future__ import annotations

import re
import unicodedata

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ", "​": "",
    "‌": "", "‍": "", "﻿": "", "­": "",
}
_BULLETS = re.compile(r"^[\s]*[•●▪■‣⁃∙◦➢➔►*]+\s*", re.M)

_HYPHEN_BREAK = re.compile(r"([^\W\d_])-[ ]*\n[ ]*(?=([^\W\d_]))")


def _rejoin(match: re.Match) -> str:
    if match.group(2).islower():
        return match.group(1)
    return match.group(0)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    # Drop control / non-printable characters (keep letters of every script).
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")
    text = _BULLETS.sub("- ", text)
    # Re-join words broken across lines: "develop-\n  ment" -> "development". Only a
    # letter followed by a lowercase letter, so "2019-\n2021" and "Python-\nDjango" stay apart.
    text = _HYPHEN_BREAK.sub(_rejoin, text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))
