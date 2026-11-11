"""Job Description Processor: turns raw requirements into a reference profile
(required / preferred skills, experience range, responsibilities, education)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from intellihire.ingestion.normalize import normalize_text
from intellihire.nlp.experience import extract_requirement
from intellihire.nlp.skills import extract_skills
from intellihire.nlp.spacy_pipeline import key_phrases

_HEADING_TYPES: list[tuple[str, re.Pattern]] = [
    ("preferred", re.compile(
        r"^\s*(?:preferred|nice[- ]to[- ]have|good[- ]to[- ]have|bonus(?: points)?|desired|desirable|"
        r"preferred (?:qualifications|skills)|plus|additional skills|it would be great if)\b.*:?\s*$", re.I)),
    ("required", re.compile(
        r"^\s*(?:requirements?|required (?:skills|qualifications)|qualifications|must[- ]haves?|"
        r"what you(?:'ll)? (?:need|bring)|what we(?:'re)? looking for|skills(?: required| & experience| and experience)?|"
        r"key skills|technical skills|eligibility|who you are|basic qualifications|minimum qualifications)\b.*:?\s*$", re.I)),
    ("responsibilities", re.compile(
        r"^\s*(?:responsibilities|key responsibilities|roles? (?:and|&) responsibilities|what you(?:'ll)? do|"
        r"duties|the role|your role|job description|about the role|day[- ]to[- ]day|you will)\b.*:?\s*$", re.I)),
    ("other", re.compile(
        r"^\s*(?:about us|about the company|who we are|benefits|perks|what we offer|compensation|salary|"
        r"location|how to apply|equal opportunity)\b.*:?\s*$", re.I)),
]
_EDU_RE = re.compile(
    r"\b(ph\.?d|doctorate|master'?s|m\.?tech|mba|bachelor'?s|b\.?tech|b\.?sc|m\.?sc|degree in [a-z ]+)"
    # short forms are case-sensitive: "be" / "MS Office" are not degrees
    r"|(?-i:\b(?:M\.\s?S|B\.\s?E|M\.\s?E|B\.\s?S)\b\.?|\b(?:MS|BE|ME|BS)\b(?=\s+(?:in|of)\b|\s*/\s*(?:PhD|Ph\.D)|\s*\())",
    re.IGNORECASE,
)


@dataclass
class JobProfile:
    title: str
    text: str
    required_skills: list[str]
    preferred_skills: list[str]
    min_years: float | None
    max_years: float | None
    responsibilities: str
    education: str
    key_phrases: list[str] = field(default_factory=list)
    skill_evidence: dict[str, list[str]] = field(default_factory=dict)


def _is_heading(line: str) -> bool:
    return 0 < len(line) <= 60 and (line.endswith(":") or line.isupper() or len(line.split()) <= 6)


def split_sections(text: str) -> dict[str, str]:
    buckets: dict[str, list[str]] = {"intro": [], "required": [], "preferred": [], "responsibilities": [], "other": []}
    current = "intro"
    for line in text.split("\n"):
        stripped = line.strip().strip("#*").strip()
        if _is_heading(stripped):
            matched = next((name for name, pat in _HEADING_TYPES if pat.match(stripped)), None)
            if matched:
                current = matched
                _, _, rest = stripped.partition(":")
                if rest.strip():  # "Requirements: Python, SQL" keeps its inline content
                    buckets[current].append(rest.strip())
                elif ":" not in stripped and not stripped.isupper() and extract_skills(stripped):
                    # "Bonus points for Kubernetes": a short content line that starts like a heading
                    buckets[current].append(stripped)
                continue
        buckets[current].append(line)
    return {k: "\n".join(v).strip() for k, v in buckets.items()}


def parse_job_description(text: str, title: str = "", extra_skills: list[str] | None = None) -> JobProfile:
    text = normalize_text(text)
    extra = tuple(extra_skills or ())
    sections = split_sections(text)
    all_skills = extract_skills(text, extra)
    preferred_found = extract_skills(sections["preferred"], extra) if sections["preferred"] else {}
    required_found = extract_skills(sections["required"], extra) if sections["required"] else {}

    # Required = skills in a "requirements" section, plus any skill mentioned
    # outside the "preferred" section. Preferred = only in the preferred section.
    preferred = [s for s in preferred_found if s not in required_found]
    required = [s for s in all_skills if s in required_found or s not in preferred_found]
    if not sections["required"] and not sections["preferred"]:
        required, preferred = list(all_skills), []

    min_years, max_years = extract_requirement(text)
    edu = _EDU_RE.search(text)
    if not title:
        first = next((l.strip() for l in text.split("\n") if l.strip()), "Untitled role")
        title = re.sub(r"^(job title|position|role)\s*:\s*", "", first, flags=re.I)[:120]

    return JobProfile(
        title=title,
        text=text,
        required_skills=sorted(required),
        preferred_skills=sorted(preferred),
        min_years=min_years,
        max_years=max_years,
        responsibilities=sections["responsibilities"] or text,
        education=edu.group(0) if edu else "",
        key_phrases=key_phrases(text, 12),
        skill_evidence=all_skills,
    )
