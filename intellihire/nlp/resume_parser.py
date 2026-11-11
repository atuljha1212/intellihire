"""Resume Parser: maps normalised resume text to a structured candidate profile."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePath

from intellihire.nlp.experience import ExperienceInfo, extract_experience
from intellihire.nlp.skills import extract_skills
from intellihire.nlp.spacy_pipeline import key_phrases, person_names
from intellihire.nlp.taxonomy import CERTIFICATIONS, DEGREE_PATTERNS

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<![\w+])\+?\(?\d[\d\s().-]{8,18}\d(?!\d)")
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com/in|github\.com|gitlab\.com)/[A-Za-z0-9_./-]+", re.I)
_DEGREE_RE = re.compile("|".join(DEGREE_PATTERNS), re.IGNORECASE)
def _cert_alternation(certs: list[str]) -> str:
    return "|".join(re.escape(c) for c in sorted(certs, key=len, reverse=True))


# Short acronyms (CKA, CSM, CFA ...) are matched in upper case only and as whole words, so
# "hackathon" / "packages" / "CFAs" do not produce certifications.
_ACRONYMS = [c for c in CERTIFICATIONS if c == c.upper() and len(c) <= 6]
_CERT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:(?-i:" + _cert_alternation(_ACRONYMS) + ")|"
    + _cert_alternation([c for c in CERTIFICATIONS if c not in _ACRONYMS]) + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_CERT_LINE = re.compile(r"\b(certified|certification|certificate)\b", re.IGNORECASE)
_NAME_BLOCKLIST = re.compile(
    r"\b(resume|curriculum|vitae|cv|profile|summary|objective|contact|email|phone|address|engineer|developer|"
    r"scientist|analyst|manager|intern|student|linkedin|github|experience|skills?|education|projects?|details|"
    r"information|personal|technical|work|certifications?|achievements|references|software|senior|junior|lead|"
    r"consultant|specialist|architect|designer|officer|executive|director|tester|administrator|ignore|instructions?|"
    r"rank|previous|candidate|page|document)\b",
    re.IGNORECASE,
)
_LOCATION_WORDS = {
    "india", "delhi", "mumbai", "bangalore", "bengaluru", "hyderabad", "chennai", "pune", "kolkata", "noida", "gurgaon",
    "gurugram", "ahmedabad", "jaipur", "kerala", "karnataka", "maharashtra", "usa", "uk", "london", "york", "new",
    "california", "texas", "singapore", "dubai", "canada", "toronto", "germany", "berlin", "paris", "sydney", "australia",
    "san", "francisco", "seattle", "boston", "street", "road", "nagar", "city", "state", "remote",
}
_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z.'-]*")
_NAME_PARTICLES = {"van", "von", "de", "del", "da", "bin", "bint", "al", "el", "la", "le", "di", "du", "ibn"}


@dataclass
class ResumeProfile:
    name: str
    email: str
    phone: str
    links: list[str]
    skills: dict[str, list[str]]
    experience: ExperienceInfo
    education: list[str]
    certifications: list[str]
    key_phrases: list[str] = field(default_factory=list)


def _clean_name(raw: str) -> str | None:
    """Return a normalised person name if `raw` looks like one (single line, 2-4 alphabetic tokens), else None."""
    line = re.sub(r"^(name)\s*:\s*", "", raw.strip(), flags=re.I)
    line = re.sub(r"\s+", " ", line)
    words = line.split(" ")
    if not 2 <= len(words) <= 4 or "@" in line or any(ch.isdigit() for ch in line):
        return None
    if not all(_NAME_TOKEN.fullmatch(w) for w in words):
        return None
    if _NAME_BLOCKLIST.search(line) or any(w.lower().strip(".,") in _LOCATION_WORDS for w in words):
        return None
    if sum(1 for w in words if len(w.strip(".")) >= 2) < 2:
        return None
    if line.isupper() or line.islower():
        line = line.title() if line.isupper() else line
    # ordinary capitalised words only (allow particles like "van"/"de" in lowercase)
    if not all(w[0].isupper() or w.lower() in _NAME_PARTICLES for w in line.split(" ")):
        return None
    if extract_skills(line):  # "Excel VBA", "Machine Learning" ...
        return None
    return line


def _filename_name(filename: str) -> str:
    stem = PurePath(filename).stem
    tokens = [t for t in re.split(r"[\s_.\-]+", stem) if t]
    tokens = [t for t in tokens if not t.isdigit() and t.lower() not in {"resume", "cv", "final", "updated", "new", "latest", "copy"}]
    tokens = [re.sub(r"\d+", "", t) for t in tokens]  # "Jane2" -> "Jane"
    tokens = [t for t in tokens if t]
    if (
        1 <= len(tokens) <= 4
        and all(t.isalpha() and len(t) <= 20 for t in tokens)
        and not _NAME_BLOCKLIST.search(" ".join(tokens))
    ):
        return " ".join(tokens).title()
    return "Unknown Candidate"


def guess_name(text: str, filename: str) -> str:
    # 1. a name-like line among the first lines (handles ALL-CAPS headers)
    for line in [l.strip() for l in text.split("\n") if l.strip()][:6]:
        name = _clean_name(line)
        if name:
            return name
    # 2. spaCy PERSON entities, validated; entities may span lines, so check each piece
    for entity in person_names(text):
        for piece in entity.splitlines():
            name = _clean_name(piece)
            if name:
                return name
    # 3. filename (only a short alphabetic name; never a sentence)
    return _filename_name(filename)


def _phone(text: str) -> str:
    for m in PHONE_RE.finditer(text[:1500]):
        digits = re.sub(r"\D", "", m.group(0))
        groups = re.findall(r"\d+", m.group(0))
        looks_like_years = all(len(g) == 4 and g[:2] in {"19", "20"} for g in groups)
        if 10 <= len(digits) <= 13 and not looks_like_years:
            return m.group(0).strip()
    return ""


def _dedupe_contained(items: set[str]) -> list[str]:
    """Drop entries that are contained in a longer entry ("AWS Certified X" vs the full line)."""
    ordered = sorted(items, key=len, reverse=True)
    kept: list[str] = []
    for item in ordered:
        if not any(item.lower() in k.lower() for k in kept):
            kept.append(item)
    return sorted(kept)


def parse_resume(text: str, filename: str, extra_skills: list[str] | None = None) -> ResumeProfile:
    email = EMAIL_RE.search(text)
    education = []
    for line in text.split("\n"):
        if _DEGREE_RE.search(line) and line.strip() not in education:
            education.append(line.strip()[:160])
    certs = {m.group(0) for m in _CERT_RE.finditer(text)}
    for line in text.split("\n"):
        if _CERT_LINE.search(line) and len(line) < 160:
            certs.add(line.strip(" -"))
    return ResumeProfile(
        name=guess_name(text, filename),
        email=email.group(0) if email else "",
        phone=_phone(text),
        links=sorted({m.group(0) for m in URL_RE.finditer(text)})[:5],
        skills=extract_skills(text, tuple(extra_skills or ())),
        experience=extract_experience(text),
        education=education[:5],
        certifications=_dedupe_contained(certs)[:10],
        key_phrases=key_phrases(text, 12),
    )
