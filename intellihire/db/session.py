"""Engine / session factory and schema bootstrap."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from collections.abc import Iterator

from sqlalchemy import create_engine, event, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from intellihire.config import get_settings
from intellihire.db.models import Base, User

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    url = settings.database_url
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(pool_size=5, max_overflow=10, pool_recycle=1800)
        if url.startswith("mysql") and "charset=" not in url:
            url += ("&" if "?" in url else "?") + "charset=utf8mb4"
    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # enforce ON DELETE CASCADE
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error."""
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# (table, column, DDL type) added after the first release; create_all never alters existing tables.
_ADDED_COLUMNS = [
    ("candidates", "encoder_name", "VARCHAR(128)"),
    ("video_transcripts", "encoder_name", "VARCHAR(128)"),
]


def _add_missing_columns(engine: Engine) -> None:
    """Idempotent, additive-only migration (SQLite and MySQL)."""
    insp = inspect(engine)
    for table, column, ddl in _ADDED_COLUMNS:
        if not insp.has_table(table):
            continue
        if column in {c["name"] for c in insp.get_columns(table)}:
            continue
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            log.info("Migrated: added %s.%s", table, column)
        except Exception:  # another process may have added it concurrently
            if column not in {c["name"] for c in inspect(engine).get_columns(table)}:
                raise


def init_db() -> None:
    """Create tables (idempotent) and the bootstrap admin account."""
    from intellihire.services.auth import hash_password

    engine = get_engine()
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    settings = get_settings()
    with session_scope() as s:
        if s.scalar(select(User).limit(1)) is None:
            s.add(
                User(
                    username=settings.admin_username,
                    full_name="Administrator",
                    password_hash=hash_password(settings.admin_password),
                    role="admin",
                )
            )
            log.info("Created bootstrap admin user '%s'", settings.admin_username)
            if settings.admin_password == "admin123":
                log.warning("The bootstrap admin uses the default password; the app will require a change at "
                            "first sign-in. Set ADMIN_PASSWORD in .env to choose your own.")


def database_label() -> str:
    engine = get_engine()
    url = engine.url
    if url.get_backend_name() == "sqlite":
        return f"SQLite ({url.database})"
    return f"{url.get_backend_name()} @ {url.host}:{url.port or ''}/{url.database}"


def table_names() -> list[str]:
    return inspect(get_engine()).get_table_names()
