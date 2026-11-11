"""User accounts, login and the audit trail."""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from intellihire.config import get_settings
from intellihire.db.models import AuditLog, User, utcnow
from intellihire.db.session import session_scope
from intellihire.services.auth import ROLES, hash_password, password_problems, valid_username, verify_password


class UserError(Exception):
    pass


# Server-side login throttling, counted from the audit log (so it holds across
# browser sessions and server restarts). Every failed login row records the
# username and the client IP (when known). Within LOGIN_WINDOW:
# - (username, IP) pair: MAX_FAILED_LOGINS failures (and no successful login of
#   that user from that IP since) lock only that pair, so a stranger cannot lock
#   the real user out from another address.
# - per IP, across all usernames: MAX_FAILED_PER_IP failures stop password spraying.
# - per username, across all IPs: past MAX_FAILED_PER_USERNAME failures the account
#   accepts one attempt per BACKSTOP_DELAY (a delay, not a lock) against a
#   distributed brute force. IPs this user has signed in from recently are exempt.
# When the client IP is unknown (None), the pair rule falls back to the username
# alone (all failures for it, from any IP).
# The pair limit, the per-IP limit (0 = off, e.g. behind a NAT that hides client IPs)
# and the window come from LOGIN_MAX_FAILURES / LOGIN_MAX_FAILURES_PER_IP /
# LOGIN_WINDOW_MINUTES (defaults 5 / 20 / 15).
_LIMITS = get_settings()
MAX_FAILED_LOGINS = _LIMITS.login_max_failures
MAX_FAILED_PER_IP = _LIMITS.login_max_failures_per_ip
MAX_FAILED_PER_USERNAME = 50
BACKSTOP_DELAY = timedelta(seconds=30)
KNOWN_IP_WINDOW = timedelta(days=30)
LOGIN_WINDOW = timedelta(minutes=_LIMITS.login_window_minutes)

# The shipped bootstrap password (config.py default). An admin account still using
# it is flagged so the UI can force a password change.
DEFAULT_ADMIN_PASSWORD = "admin123"
_DEFAULT_PW_CACHE: dict[str, bool] = {}


# Verified against when the username does not exist, so a failed login takes the
# same time either way and response timing does not reveal which usernames exist.
_DUMMY_HASH: str | None = None


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("intellihire-timing-equaliser")
    return _DUMMY_HASH


def audit(session: Session, user_id: int | None, action: str, entity: str = "", entity_id: int | None = None,
          **detail) -> None:
    session.add(AuditLog(user_id=user_id, action=action, entity=entity, entity_id=entity_id, detail=detail))


def _public(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "full_name": user.full_name or user.username,
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at,
        "last_login": user.last_login,
    }


def _clean_ip(ip: str | None) -> str | None:
    ip = str(ip or "").strip()[:64]
    return ip or None


def _recent_failures(s: Session, username: str, user_id: int | None, ip: str | None = None) -> list:
    """Timestamps of failed logins for `username` inside LOGIN_WINDOW since its last
    successful login. With `ip`, only failures from that IP count, and only a
    successful login from that IP resets the count."""
    cutoff = utcnow() - LOGIN_WINDOW
    rows = s.scalars(
        select(AuditLog)
        .where(AuditLog.action.in_(("login_failed", "login")), AuditLog.created_at >= cutoff)
        .order_by(AuditLog.id.desc())
    )
    failures = []
    for r in rows:
        detail = r.detail or {}
        if ip is not None and detail.get("ip") != ip:
            continue
        if r.action == "login":
            if user_id is not None and r.user_id == user_id:
                break
            continue
        if str(detail.get("detail_username", "")).strip() == username:
            failures.append(r.created_at)
    return failures


def _failures_where(s: Session, key: str, value: str) -> list:
    """Timestamps of all failed logins inside LOGIN_WINDOW whose audit detail[key] == value."""
    cutoff = utcnow() - LOGIN_WINDOW
    rows = s.scalars(select(AuditLog).where(AuditLog.action == "login_failed", AuditLog.created_at >= cutoff))
    return [r.created_at for r in rows if str((r.detail or {}).get(key, "")).strip() == value]


def _known_ip(s: Session, user_id: int | None, ip: str | None) -> bool:
    """True if `user_id` signed in successfully from `ip` within KNOWN_IP_WINDOW."""
    if user_id is None or ip is None:
        return False
    cutoff = utcnow() - KNOWN_IP_WINDOW
    rows = s.scalars(select(AuditLog).where(AuditLog.action == "login", AuditLog.user_id == user_id,
                                            AuditLog.created_at >= cutoff))
    return any((r.detail or {}).get("ip") == ip for r in rows)


def _wait_after(failures: list, limit: int) -> float:
    """Seconds until fewer than `limit` of `failures` remain inside LOGIN_WINDOW (0 = not limited)."""
    if len(failures) < limit:
        return 0.0
    unlock = sorted(failures, reverse=True)[limit - 1] + LOGIN_WINDOW
    return (unlock - utcnow()).total_seconds()


def _throttle(s: Session, username: str, user: User | None, ip: str | None) -> tuple[int, str]:
    """(seconds to wait, which rule) for a login attempt; (0, "") when allowed."""
    user_id = user.id if user else None
    waits = [(_wait_after(_recent_failures(s, username, user_id, ip), MAX_FAILED_LOGINS),
              "user+ip" if ip else "user")]
    if ip is not None:
        if MAX_FAILED_PER_IP > 0:
            waits.append((_wait_after(_failures_where(s, "ip", ip), MAX_FAILED_PER_IP), "ip"))
        if not _known_ip(s, user_id, ip):
            everywhere = _failures_where(s, "detail_username", username)
            if len(everywhere) >= MAX_FAILED_PER_USERNAME:
                waits.append(((max(everywhere) + BACKSTOP_DELAY - utcnow()).total_seconds(), "user backstop"))
    wait, rule = max(waits)
    return (max(1, int(wait) + 1), rule) if wait > 0 else (0, "")


def login_lockout_seconds(username: str, ip: str | None = None) -> int:
    """Seconds until `username` may try to sign in again from `ip` (0 = not locked).
    The reason is deliberately not returned so the UI cannot reveal which limit hit."""
    username = (username or "").strip()[:64]
    if not username:
        return 0
    with session_scope() as s:
        user = s.scalar(select(User).where(User.username == username))
        return _throttle(s, username, user, _clean_ip(ip))[0]


def authenticate(username: str, password: str, ip: str | None = None) -> dict | None:
    username = (username or "").strip()[:64]
    ip = _clean_ip(ip)
    with session_scope() as s:
        user = s.scalar(select(User).where(User.username == username))
        wait, rule = _throttle(s, username, user, ip)
        if wait:
            verify_password(password, _dummy_hash())  # same timing as a normal attempt
            audit(s, user.id if user else None, "login_throttled", "user", detail_username=username, ip=ip,
                  rule=rule)
            return None
        # Always verify the password first so an inactive account answers in the
        # same time as an active one (no timing oracle on account status).
        ok = verify_password(password, user.password_hash if user else _dummy_hash())
        if user is None or not ok or not user.is_active:
            audit(s, user.id if user else None, "login_failed", "user", detail_username=username, ip=ip)
            return None
        user.last_login = utcnow()
        audit(s, user.id, "login", "user", user.id, ip=ip)
        return _public(user)


def uses_default_password(user_id: int) -> bool:
    """True if this account's password is still the shipped default ('admin123')."""
    with session_scope() as s:
        user = s.get(User, user_id)
        stored = user.password_hash if user else ""
    if not stored:
        return False
    if stored not in _DEFAULT_PW_CACHE:  # bcrypt is slow; the answer only changes with the hash
        _DEFAULT_PW_CACHE[stored] = verify_password(DEFAULT_ADMIN_PASSWORD, stored)
    return _DEFAULT_PW_CACHE[stored]


def get_active_user(user_id: int | None) -> dict | None:
    """Fresh copy of a signed-in user from the DB, or None if it no longer exists or is inactive."""
    if user_id is None:
        return None
    with session_scope() as s:
        user = s.get(User, user_id)
        return _public(user) if user is not None and user.is_active else None


def _require_admin(s: Session, actor_id: int | None) -> User:
    actor = s.get(User, actor_id) if actor_id is not None else None
    if actor is None or not actor.is_active or actor.role != "admin":
        raise UserError("Only an active admin can manage users.")
    return actor


def _other_active_admins(s: Session, user_id: int) -> int:
    return len(list(s.scalars(select(User.id).where(User.role == "admin", User.is_active.is_(True),
                                                      User.id != user_id))))


def list_users() -> list[dict]:
    with session_scope() as s:
        return [_public(u) for u in s.scalars(select(User).order_by(User.id))]


def create_user(actor_id: int, username: str, password: str, role: str, full_name: str = "", email: str = "") -> dict:
    if not valid_username(username):
        raise UserError("Username must be 3-64 characters: letters, digits, '.', '_' or '-'.")
    if role not in ROLES:
        raise UserError(f"Role must be one of {ROLES}.")
    problems = password_problems(password)
    if problems:
        raise UserError("Password needs " + " and ".join(problems) + ".")
    with session_scope() as s:
        _require_admin(s, actor_id)
        if s.scalar(select(User).where(User.username == username)):
            raise UserError("That username is already taken.")
        user = User(username=username, password_hash=hash_password(password), role=role,
                    full_name=full_name.strip(), email=email.strip())
        s.add(user)
        s.flush()
        audit(s, actor_id, "user_created", "user", user.id, role=role)
        return _public(user)


def set_user_active(actor_id: int, user_id: int, active: bool) -> None:
    if actor_id == user_id and not active:
        raise UserError("You cannot deactivate your own account.")
    with session_scope() as s:
        _require_admin(s, actor_id)
        user = s.get(User, user_id)
        if user is None:
            raise UserError("User not found.")
        if not active and user.role == "admin" and user.is_active and _other_active_admins(s, user_id) == 0:
            raise UserError("You cannot deactivate the last active admin.")
        user.is_active = active
        audit(s, actor_id, "user_activated" if active else "user_deactivated", "user", user_id)


def change_password(actor_id: int, user_id: int, new_password: str) -> None:
    problems = password_problems(new_password)
    if problems:
        raise UserError("Password needs " + " and ".join(problems) + ".")
    if new_password == DEFAULT_ADMIN_PASSWORD:
        raise UserError("Choose a password other than the default one.")
    with session_scope() as s:
        if actor_id != user_id:  # users may change their own password; anyone else's needs an admin
            _require_admin(s, actor_id)
        else:
            actor = s.get(User, actor_id)
            if actor is None or not actor.is_active:
                raise UserError("Only an active user can change a password.")
        user = s.get(User, user_id)
        if user is None:
            raise UserError("User not found.")
        user.password_hash = hash_password(new_password)
        audit(s, actor_id, "password_changed", "user", user_id)


def recent_audit(limit: int = 200) -> list[dict]:
    with session_scope() as s:
        names = {u.id: u.username for u in s.scalars(select(User))}
        rows = s.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(limit))
        return [
            {"time": r.created_at, "user": names.get(r.user_id, "-"), "action": r.action,
             "entity": f"{r.entity} #{r.entity_id}" if r.entity_id else r.entity, "detail": r.detail}
            for r in rows
        ]
