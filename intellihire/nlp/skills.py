"""Skill extraction with spaCy's PhraseMatcher (case-insensitive, alias aware).

Aliases containing symbols ("C++", "CI/CD", ".NET") and case-sensitive short
names ("Go", "R") are matched by a boundary-aware regex, because tokenisers
split such strings inconsistently. Every hit keeps the line it came from, so the
dashboard can show *where* a skill was found (explainability).
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from functools import lru_cache

from intellihire.nlp.spacy_pipeline import get_nlp
from intellihire.nlp.taxonomy import SKILLS

MAX_EVIDENCE_PER_SKILL = 3
_WORD_ONLY = re.compile(r"^[a-z0-9 ]+$")
# Left boundary: not preceded by a word char, '+', '#' or '.'.
# Right boundary: not followed by a word char, '+', '#', or a '.' that continues a word.
_LEFT = r"(?<![\w+#.])"
_RIGHT = r"(?![\w+#]|\.\w)"
# Case-sensitive short names ("C", "Go", "R"): "R&D" / "Go-getter" are not skills, "R." / "C." at the end
# of a sentence are.
_RIGHT_CS = r"(?![\w+#&-]|\.\w)"
_CONTEXT_BEFORE = re.compile(
    r"(?:plan|vitamin|grade|type|class|section|option|category|part|appendix|hepatitis|level|tier|team|group|"
    r"exhibit|figure|fig|chapter|point|item|schedule|rule|clause|article|zone|phase|variant|model|class)\.?\s+$",
    re.IGNORECASE,
)
_GO_NOT_SKILL_AFTER = re.compile(
    r"\s+(?:to|for|the|a|an|with|ahead|back|live|through|beyond|above|getter|big|forward|on|by|at|over|out|into|about|it|"
    r"and\s+(?:get|see|do|find))\b"
)


def canonical_skill_names() -> list[str]:
    return sorted(SKILLS)


def skill_category(skill: str) -> str:
    return SKILLS.get(skill, {}).get("category", "Custom")


@dataclass
class _Matchers:
    alias_to_skill: dict[str, str]
    regex_ci: re.Pattern | None
    regex_cs: re.Pattern | None
    phrase_matcher: object | None
    nlp: object | None
    cs_alias_to_skill: dict[str, str] = field(default_factory=dict)


def _alternation(aliases: list[str]) -> str:
    return "|".join(re.escape(a) for a in sorted(set(aliases), key=len, reverse=True))


@lru_cache(maxsize=32)
def _build(extra: tuple[str, ...]) -> _Matchers:
    alias_to_skill: dict[str, str] = {}
    case_sensitive: dict[str, str] = {}
    for name, info in SKILLS.items():
        for alias in info.get("cs_aliases", []):
            case_sensitive[alias] = name
        if info.get("case_sensitive"):
            case_sensitive[name] = name
            for alias in info.get("aliases", []):
                alias_to_skill[alias.lower()] = name
        else:
            alias_to_skill[name.lower()] = name
            for alias in info.get("aliases", []):
                alias_to_skill[alias.lower()] = name
    known = {n.lower() for n in SKILLS} | set(alias_to_skill) | {a.lower() for a in case_sensitive}
    for custom in extra:
        custom = (custom or "").strip()
        if custom and custom.lower() not in known:  # "python" must not override canonical "Python"
            alias_to_skill[custom.lower()] = custom

    nlp, _ = get_nlp()
    phrase_matcher = None
    regex_aliases = list(alias_to_skill)
    if nlp is not None:
        from spacy.matcher import PhraseMatcher

        phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        word_only = [a for a in alias_to_skill if _WORD_ONLY.match(a)]
        for alias in word_only:
            phrase_matcher.add(alias_to_skill[alias], [nlp.make_doc(alias)])
        regex_aliases = [a for a in alias_to_skill if not _WORD_ONLY.match(a)]

    regex_ci = (
        re.compile(_LEFT + "(" + _alternation(regex_aliases) + ")" + _RIGHT, re.IGNORECASE)
        if regex_aliases
        else None
    )
    regex_cs = re.compile(_LEFT + "(" + _alternation(list(case_sensitive)) + ")" + _RIGHT_CS) if case_sensitive else None
    return _Matchers(alias_to_skill, regex_ci, regex_cs, phrase_matcher, nlp, case_sensitive)


def _line_index(text: str) -> tuple[list[int], list[str]]:
    starts, lines, pos = [], [], 0
    for line in text.split("\n"):
        starts.append(pos)
        lines.append(line)
        pos += len(line) + 1
    return starts, lines


# "in Go with gRPC", "using Go for services": after these words "Go" is the language, whatever follows.
_GO_LANGUAGE_BEFORE = re.compile(r"\b(?:in|using|with|written in|of|languages?\s*:)\s+$", re.IGNORECASE)


def _plausible_short_skill(text: str, match: re.Match) -> bool:
    """Reject one/two-letter case-sensitive hits that are clearly ordinary text ("Plan C", "Go to market")."""
    token = match.group(1)
    if _CONTEXT_BEFORE.search(text[max(0, match.start() - 14):match.start()]):
        return False
    if (token == "Go" and _GO_NOT_SKILL_AFTER.match(text, match.end())
            and not _GO_LANGUAGE_BEFORE.search(text[max(0, match.start() - 20):match.start()])):
        return False
    return True


def extract_skills(text: str, extra_skills: list[str] | tuple[str, ...] = ()) -> dict[str, list[str]]:
    """Return {canonical skill: [evidence lines]} found in `text`."""
    if not text:
        return {}
    m = _build(tuple(sorted(set(extra_skills))))
    starts, lines = _line_index(text)
    found: dict[str, list[str]] = {}

    def record(skill: str, char_pos: int) -> None:
        line_no = max(0, bisect.bisect_right(starts, char_pos) - 1)
        evidence = lines[line_no].strip()[:220]
        bucket = found.setdefault(skill, [])
        if evidence and evidence not in bucket and len(bucket) < MAX_EVIDENCE_PER_SKILL:
            bucket.append(evidence)

    if m.phrase_matcher is not None:
        doc = m.nlp.make_doc(text[:200_000])
        for match_id, start, end in m.phrase_matcher(doc):
            record(m.nlp.vocab.strings[match_id], doc[start:end].start_char)
    if m.regex_ci is not None:
        for match in m.regex_ci.finditer(text):
            skill = m.alias_to_skill.get(match.group(1).lower())
            if skill:
                record(skill, match.start())
    if m.regex_cs is not None:
        for match in m.regex_cs.finditer(text):
            token = match.group(1)
            if len(token) <= 2 and not _plausible_short_skill(text, match):
                continue
            record(m.cs_alias_to_skill.get(token, token), match.start())
    return found


def split_skill_matches(
    required: list[str], preferred: list[str], candidate_skills: dict[str, list[str]]
) -> dict[str, list[str]]:
    have = set(candidate_skills)
    return {
        "matched_required": [s for s in required if s in have],
        "missing_required": [s for s in required if s not in have],
        "matched_preferred": [s for s in preferred if s in have],
        "missing_preferred": [s for s in preferred if s not in have],
    }
