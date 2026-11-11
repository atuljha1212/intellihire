"""Experience parsing: total years from employment date ranges (overlaps merged)
or explicit statements, and the experience requirement of a job description."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from intellihire.nlp.skills import extract_skills
from intellihire.nlp.taxonomy import DEGREE_PATTERNS

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_ISO = r"\d{4}[/-](?:0[1-9]|1[0-2])(?!\d)"  # 2020-01
_DATE = (
    rf"(?:{_MONTH_RE}\.?\s*,?\s*\d{{4}}(?!\d)|{_ISO}|\d{{1,2}}\s*[/.-]\s*\d{{4}}(?!\d)"
    rf"|(?:0?[1-9]|1[0-2])\s*/\s*\d{{2}}(?!\d)|{_MONTH_RE}\.?\s*,?\s*'?\d{{2}}(?!\d)|\d{{4}}(?!\d))"
)
_PRESENT = r"(?:present|current(?:ly)?|now|today|till date|to date|ongoing)\b"
_RANGE_RE = re.compile(
    rf"(?<![\w/.-])({_DATE})\s*(?:-|to\b|until\b|till\b|–|—)\s*({_DATE}|{_PRESENT})", re.IGNORECASE
)
_EXPLICIT_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\.?(?:\s+and\s+\d+\s+months?)?\s+(?:of\s+)?"
    r"(?:\w+\s+){0,3}?experience",
    re.IGNORECASE,
)
_EXPERIENCE_HEADINGS = re.compile(
    r"^\s*(?:work\s+|professional\s+|employment\s+|relevant\s+|industry\s+)?"
    r"(?:experience|employment(?: history)?|work history|internships?|career history)\s*:?\s*$",
    re.IGNORECASE,
)
_OTHER_HEADINGS = re.compile(
    r"^\s*(?:education|academics?|academic background|qualifications?|skills|technical skills|projects?|"
    r"academic projects|certifications?|achievements|awards|publications|languages|interests|hobbies|"
    r"summary|profile|objective|references|extra[- ]curricular(?: activities)?|positions of responsibility)\s*:?\s*$",
    re.IGNORECASE,
)
_SECTION_END = re.compile(
    r"^\s*(?:education|academics?|academic background|qualifications?|skills|technical skills|key skills|core skills|"
    r"projects?|academic projects|personal projects|certifications?|achievements|awards|publications|languages|"
    r"interests|hobbies|summary|profile|objective|references|extra[- ]curricular|positions of responsibility|"
    r"volunteer(?:ing)?|training|courses)\b[^.\d]{0,30}$",
    re.IGNORECASE,
)
# An actual degree (shared with resume_parser) or a grade/class marker; these always mark an education line.
_DEGREE_RE = re.compile(
    "|".join(DEGREE_PATTERNS) + r"|\bassociate'?s?\s+(?:degree|of|in)\b|\bcgpa\b|\bgpa\b|\bclass\s+xii?\b"
    # "Bachelor's degree in ...", "Masters in ...", "Degree in ...": a degree even when the line also
    # names "Systems"/"Technologies"/"at" (which would otherwise read as job context)
    r"|\b(?:bachelor|master)(?:'s|’s|s)?\s+(?:degree|of|in)\b|\bdegree\s+in\b",
    re.IGNORECASE,
)
# Plain education words. They mark an education line only when the line has no job or company context,
# so "360 Degree Solutions" or "University of Phoenix - IT Support Specialist" stay work entries.
_PLAIN_EDU_RE = re.compile(
    r"\b(?:university|college|school|degree|diploma|bachelor'?s?|masters?)\b", re.IGNORECASE
)
_JOB_CONTEXT_RE = re.compile(
    r"\b(?:engineer|developer|analyst|specialist|manager|consultant|intern|architect|administrator|officer|"
    r"executive|coordinator|technician|designer|scientist|assistant|director|lead|supervisor|programmer|trainee|"
    r"researcher|tester|professor|lecturer|teacher|associate|founder|head|representative|clerk|staff|support|"
    r"inc|ltd|llc|corp|pvt|plc|solutions|technologies|systems|company|group|at)\b|@",
    re.IGNORECASE,
)
_EDU_HEADING = re.compile(
    r"^\s*(?:education|academics?|academic background)(?:\s*(?:&|and)\s*training)?\s*:?\s*$", re.IGNORECASE
)


@dataclass
class ExperienceInfo:
    years: float
    method: str  # "date ranges" | "explicit statement" | "none"
    evidence: list[str] = field(default_factory=list)


def _two_digit_year(yy: int) -> int:
    return 2000 + yy if yy <= 69 else 1900 + yy


def _parse_date(token: str, is_end: bool, today: date) -> tuple[int, int] | None:
    token = token.strip().lower().rstrip(".")
    if re.fullmatch(_PRESENT, token, re.IGNORECASE):
        return today.year, today.month
    m = re.fullmatch(rf"({_MONTH_RE})\.?\s*,?\s*(\d{{4}})", token, re.IGNORECASE)
    if m:
        return int(m.group(2)), _MONTHS[m.group(1).lower()[:3]]
    m = re.fullmatch(r"(\d{4})[/-](0[1-9]|1[0-2])", token)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d{1,2})\s*[/.-]\s*(\d{4})", token)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))
    m = re.fullmatch(r"(\d{1,2})\s*/\s*(\d{2})", token)
    if m and 1 <= int(m.group(1)) <= 12:
        return _two_digit_year(int(m.group(2))), int(m.group(1))
    m = re.fullmatch(rf"({_MONTH_RE})\.?\s*,?\s*'?(\d{{2}})", token, re.IGNORECASE)
    if m:
        return _two_digit_year(int(m.group(2))), _MONTHS[m.group(1).lower()[:3]]
    if re.fullmatch(r"\d{4}", token):
        # year only: a start is read as July and an end as June, so "2019 - 2021" counts 2 years, not 3
        return int(token), 6 if is_end else 7
    return None


def experience_section(text: str) -> str | None:
    """Text under an EXPERIENCE-style heading, up to the next known heading."""
    lines = text.split("\n")
    collecting, out = False, []
    for line in lines:
        if _EXPERIENCE_HEADINGS.match(line):
            collecting = True
            continue
        if collecting and (_OTHER_HEADINGS.match(line) or _SECTION_END.match(line)):
            collecting = False
            continue
        if collecting:
            out.append(line)
    return "\n".join(out) if out else None


def _merge_months(intervals: list[tuple[int, int]]) -> int:
    total, cur_start, cur_end = 0, None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def extract_experience(text: str, today: date | None = None) -> ExperienceInfo:
    today = today or date.today()
    section = experience_section(text)
    scope = section if section else text
    intervals: list[tuple[int, int]] = []
    evidence: list[str] = []
    scope_lines = scope.split("\n")
    today_idx = today.year * 12 + today.month
    in_edu_section = False  # only tracked without an experience section (the whole text is scanned)
    for i, line in enumerate(scope_lines):
        if section is None:
            if _EDU_HEADING.match(line):
                in_edu_section = True
                continue
            if _OTHER_HEADINGS.match(line) or _EXPERIENCE_HEADINGS.match(line):
                in_edu_section = False
            if in_edu_section:
                continue  # dates under an Education heading are education
        has_job_context = bool(_JOB_CONTEXT_RE.search(line))
        if section is None and not has_job_context and _PLAIN_EDU_RE.search(line):
            continue  # without a clear section, ignore education date ranges
        if _DEGREE_RE.search(line):
            continue
        # a date line right after a degree line ("B.Tech in CS / IIT Delhi / 2018 - 2022") is education,
        # unless the line itself carries job context ("TCS, Engineer, 2018 - 2020")
        nearby = [l for l in scope_lines[max(0, i - 2):i] if l.strip()]
        if not has_job_context and any(_DEGREE_RE.search(l) and not _RANGE_RE.search(l) for l in nearby):
            continue
        for m in _RANGE_RE.finditer(line):
            start = _parse_date(m.group(1), False, today)
            end = _parse_date(m.group(2), True, today)
            if not start or not end:
                continue
            s_idx, e_idx = start[0] * 12 + start[1] - 1, end[0] * 12 + end[1]
            if start[0] == end[0] and re.fullmatch(r"\d{4}", m.group(1).strip()) and re.fullmatch(r"\d{4}", m.group(2).strip()):
                s_idx, e_idx = start[0] * 12, start[0] * 12 + 6  # "2022 - 2022": about half a year
            e_idx = min(e_idx, today_idx)  # an end date in the future means "until now"
            if 1970 <= start[0] <= today.year and s_idx < e_idx:
                intervals.append((s_idx, e_idx))
                evidence.append(line.strip()[:160])

    explicit = [float(m.group(1)) for m in _EXPLICIT_RE.finditer(text) if float(m.group(1)) <= 45]
    explicit_lines = [m.group(0) for m in _EXPLICIT_RE.finditer(text)][:2]

    if intervals:
        years = round(_merge_months(intervals) / 12, 1)
        return ExperienceInfo(years, "date ranges", evidence[:6])
    if explicit:
        return ExperienceInfo(max(explicit), "explicit statement", explicit_lines)
    return ExperienceInfo(0.0, "none", [])


# ------------------------------------------------------------------ JD requirement
_JD_RANGE = re.compile(r"(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE)
_JD_MIN = re.compile(
    r"(?:minimum|min\.?|at\s+least|atleast|over|more\s+than)\s*(?:of\s*)?(\d{1,2})\s*\+?\s*(?:years?|yrs?)",
    re.IGNORECASE,
)
_JD_PLUS = re.compile(r"(\d{1,2})\s*\+\s*(?:years?|yrs?)", re.IGNORECASE)
_JD_PLAIN = re.compile(r"(\d{1,2})\s*(?:years?|yrs?)(?:\s+of)?(?:\s+\w+){0,3}\s+experience", re.IGNORECASE)
_FRESHER = re.compile(r"\b(fresher|freshers|entry[- ]level|new grad(?:uate)?s?|0\s*-\s*1 years?)\b", re.IGNORECASE)


_REQ_CONTEXT = re.compile(
    r"\b(?:experience|exp|require[sd]?|requirements?|must|need(?:s|ed)?|minimum|at least|should have|looking for|expertise|"
    r"proficien\w*|hands[- ]on|background)\b",
    re.IGNORECASE,
)
_SENTENCE_BREAK = re.compile(r"[\n;]|(?<!\d)\.(?=\s|$)|\.(?=\s+[A-Z])")


def _is_requirement_context(text: str, m: re.Match) -> bool:
    """True when the years figure sits in a sentence about experience or requirements, not "10+ years in business"."""
    start = max((b.end() for b in _SENTENCE_BREAK.finditer(text, 0, m.start())), default=0)
    end_m = _SENTENCE_BREAK.search(text, m.end())
    sentence = text[start:end_m.start() if end_m else len(text)]
    if _REQ_CONTEXT.search(sentence):
        return True
    return bool(extract_skills(text[m.end():m.end() + 40].split("\n")[0]))  # "5+ years of Python"


def extract_requirement(jd_text: str) -> tuple[float | None, float | None]:
    """Return (min_years, max_years) required by a JD; (None, None) if unspecified."""
    for m in _JD_RANGE.finditer(jd_text):
        if _is_requirement_context(jd_text, m):
            lo, hi = sorted((float(m.group(1)), float(m.group(2))))
            return lo, hi
    for pattern in (_JD_MIN, _JD_PLUS, _JD_PLAIN):
        for m in pattern.finditer(jd_text):
            if _is_requirement_context(jd_text, m):
                return float(m.group(1)), None
    if _FRESHER.search(jd_text):
        return 0.0, 1.0
    return None, None


def experience_fit(years: float, min_years: float | None, max_years: float | None) -> tuple[float, str]:
    """Deterministic experience-tier score in [0, 1] with a human-readable reason."""
    if min_years is None or min_years <= 0:
        if max_years is not None and years > max_years + 4:
            return 0.85, f"{years:g} yrs is well above the {max_years:g}-yr upper bound for this role"
        return 1.0, "The job description sets no minimum experience"
    if years >= min_years:
        if max_years is not None and years > max_years + 4:
            return 0.85, f"{years:g} yrs exceeds the {min_years:g}-{max_years:g} yr range (may be over-qualified)"
        return 1.0, f"{years:g} yrs meets the {min_years:g}+ yr requirement"
    score = max(0.0, years / min_years)
    return round(score, 3), f"{years:g} yrs is below the {min_years:g} yr minimum"
