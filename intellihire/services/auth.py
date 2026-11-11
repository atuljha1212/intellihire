"""Authentication and user management (bcrypt password hashing, roles)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re

try:
    import bcrypt

    HAS_BCRYPT = True
except ImportError:  # pragma: no cover - depends on environment
    bcrypt = None
    HAS_BCRYPT = False

ROLES = ("admin", "recruiter")
_PBKDF2_ROUNDS = 390_000


def hash_password(password: str) -> str:
    if HAS_BCRYPT:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return "pbkdf2$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    if not stored:
        return False
    if stored.startswith("pbkdf2$"):
        _, salt_b64, digest_b64 = stored.split("$", 2)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt_b64), _PBKDF2_ROUNDS)
        return hmac.compare_digest(digest, base64.b64decode(digest_b64))
    if HAS_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
        except ValueError:
            return False
    return False


def password_problems(password: str) -> list[str]:
    problems = []
    if len(password) < 8:
        problems.append("at least 8 characters")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        problems.append("letters and digits")
    if len(password.encode("utf-8")) > 72:  # bcrypt's input limit; bcrypt>=5 raises instead of truncating
        problems.append("at most 72 bytes")
    return problems


def valid_username(username: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{3,64}", username))
