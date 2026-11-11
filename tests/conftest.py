"""Test configuration: isolated data directory, SQLite DB and offline LLM mode.

Environment variables are set before any `intellihire` module is imported, so
the cached settings pick them up."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="intellihire_test_")
os.environ["INTELLIHIRE_DATA_DIR"] = _TMP
# Set explicitly (not just popped): config.py calls load_dotenv(), which would
# otherwise re-import a DATABASE_URL from .env and point the tests at a real DB.
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMP, 'test.db').as_posix()}"
os.environ["LLM_PROVIDER"] = "none"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""
os.environ.setdefault("INTELLIHIRE_MODEL_DIR", str(ROOT / "models"))

import pytest  # noqa: E402

SAMPLES = ROOT / "sample_data"


def find_ffmpeg() -> str | None:
    """Same lookup order as the app: system FFmpeg, then the imageio-ffmpeg wheel."""
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = find_ffmpeg()


@pytest.fixture(scope="session")
def samples() -> Path:
    if not (SAMPLES / "jobs" / "ml_engineer_jd.txt").exists():
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_samples.py")], check=True)
    return SAMPLES


@pytest.fixture
def tmp_dir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_login_throttle():
    """Clear failed/throttled login audit rows after each test, so one test's bad
    passwords never lock `admin` (or anyone) out in a later test. Only runs once
    the DB layer has been imported by some test; never imports it itself."""
    yield
    if "intellihire.db.session" not in sys.modules:
        return
    try:
        from sqlalchemy import delete

        from intellihire.db.models import AuditLog
        from intellihire.db.session import session_scope

        with session_scope() as s:
            s.execute(delete(AuditLog).where(AuditLog.action.in_(("login_failed", "login_throttled"))))
    except Exception:  # tables not created yet, or DB unavailable: nothing to reset
        pass
