"""Stage 1 - multi-format ingestion with a security validation constraint.

Every uploaded file is checked with the standard library *before* any parser
touches it: extension allow-list, size limits, magic-byte signature, zip-bomb
guard for DOCX, filename sanitisation and a SHA-256 fingerprint for de-duplication.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePath

from intellihire.config import get_settings

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}

# A DOCX that expands beyond this is treated as a zip bomb.
MAX_DOCX_UNCOMPRESSED = 50 * 1024 * 1024
MAX_DOCX_MEMBERS = 2_000
MAX_FILENAME_LENGTH = 150


@dataclass
class ValidationResult:
    ok: bool
    kind: str  # "pdf" | "docx" | "txt" | "video"
    safe_name: str
    size_bytes: int
    sha256: str
    errors: list[str] = field(default_factory=list)

    @property
    def error_message(self) -> str:
        return "; ".join(self.errors)


def sanitize_filename(name: str) -> str:
    """Strip any path components and characters that are unsafe on disk."""
    base = PurePath(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    if len(base) > MAX_FILENAME_LENGTH:
        # Truncate the stem, never the extension (the extension decides the file type).
        suffix = PurePath(base).suffix
        if not suffix or len(suffix) > 16:
            suffix = ""
        stem = base[: len(base) - len(suffix)] if suffix else base
        stem = stem[: MAX_FILENAME_LENGTH - len(suffix)].rstrip(" .")
        base = (stem + suffix) if stem else base[:MAX_FILENAME_LENGTH]
    return base or "upload"


# Words that only describe the file, removed when they appear as whole tokens.
_STEM_NOISE_TOKENS = {"resume", "cv", "video", "intro", "introduction"}


def file_stem_key(name: str) -> str:
    """Normalised stem used to pair a resume with its companion video.

    Descriptive words are removed only as whole separator-delimited tokens
    ("Jane_Doe_Resume" -> "janedoe", but "Cvijanovic" stays intact). The key is
    never empty: if nothing but noise words remain, the full sanitised stem is
    used so that e.g. "resume.pdf" and "video.mp4" do not pair with each other.
    """
    stem = PurePath(sanitize_filename(name)).stem.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t]
    key = "".join(t for t in tokens if t not in _STEM_NOISE_TOKENS)
    return key or "".join(tokens) or stem or "upload"


def _looks_like_pdf(data: bytes) -> bool:
    return data[:1024].lstrip().startswith(b"%PDF-")


def _check_docx(data: bytes) -> str | None:
    if not data.startswith(b"PK\x03\x04"):
        return "not a valid DOCX (zip) container"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) > MAX_DOCX_MEMBERS:
                return "DOCX contains too many internal files"
            if sum(m.file_size for m in members) > MAX_DOCX_UNCOMPRESSED:
                return "DOCX expands to an unsafe size (possible zip bomb)"
            if "word/document.xml" not in {m.filename for m in members}:
                return "zip file is not a Word document"
    except zipfile.BadZipFile:
        return "corrupted DOCX file"
    return None


def _looks_like_text(data: bytes) -> bool:
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        # Allow latin-1 style text with few control characters.
        controls = sum(1 for b in sample if b < 9 or 13 < b < 32)
        return controls < len(sample) * 0.01


def _looks_like_video(data: bytes, ext: str) -> bool:
    head = data[:64]
    if ext in {".mp4", ".mov"}:
        # ISO base media: box size (4 bytes) + box type.
        return head[4:8] in {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}
    if ext == ".avi":
        return head[:4] == b"RIFF" and head[8:12] == b"AVI "
    return False


def validate_document(name: str, data: bytes, max_mb: float | None = None) -> ValidationResult:
    """Validate a resume or job-description document (.pdf, .docx, .txt)."""
    settings = get_settings()
    limit_mb = max_mb if max_mb is not None else settings.max_resume_mb
    safe = sanitize_filename(name)
    ext = PurePath(safe).suffix.lower()
    result = ValidationResult(
        ok=True,
        kind=ext.lstrip("."),
        safe_name=safe,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    if ext not in DOCUMENT_EXTENSIONS:
        result.errors.append(f"unsupported file type '{ext or 'none'}' (allowed: PDF, DOCX, TXT)")
    if len(data) == 0:
        result.errors.append("file is empty")
    elif len(data) > limit_mb * 1024 * 1024:
        result.errors.append(f"file is larger than {limit_mb:g} MB")

    if not result.errors:
        if ext == ".pdf" and not _looks_like_pdf(data):
            result.errors.append("file content is not a PDF (signature mismatch)")
        elif ext == ".docx":
            problem = _check_docx(data)
            if problem:
                result.errors.append(problem)
        elif ext == ".txt" and not _looks_like_text(data):
            result.errors.append("file does not look like plain text")

    result.ok = not result.errors
    return result


def validate_video(name: str, data: bytes) -> ValidationResult:
    """Validate a candidate introduction video (.mp4, .mov, .avi)."""
    settings = get_settings()
    safe = sanitize_filename(name)
    ext = PurePath(safe).suffix.lower()
    result = ValidationResult(
        ok=True,
        kind="video",
        safe_name=safe,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    if ext not in VIDEO_EXTENSIONS:
        result.errors.append(f"unsupported video type '{ext or 'none'}' (allowed: MP4, MOV, AVI)")
    if len(data) == 0:
        result.errors.append("file is empty")
    elif len(data) > settings.max_video_mb * 1024 * 1024:
        result.errors.append(f"video is larger than {settings.max_video_mb:g} MB")
    if not result.errors and not _looks_like_video(data, ext):
        result.errors.append("file content does not match its video extension")
    result.ok = not result.errors
    return result


def validate_batch_size(count: int) -> str | None:
    limit = get_settings().max_batch_files
    if count > limit:
        return f"You can upload at most {limit} resumes per batch ({count} selected)."
    return None
