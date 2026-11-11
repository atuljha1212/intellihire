"""Central configuration, read from environment variables (or a .env file).

On Streamlit Cloud, top-level keys in `secrets.toml` are exported as environment
variables automatically, so the same names work there.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # optional: load a local .env during development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() != "" else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_positive_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value > 0 else default


def _env_non_negative_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value >= 0 else default


def _env_positive_float(name: str, default: float) -> float:
    value = _env_float(name, default)
    return value if value > 0 and value == value and value != float("inf") else default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


# Default weights of the scoring equation (Stage 4 of the report).
DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic": 0.30,
    "skill": 0.35,
    "experience": 0.20,
    "video": 0.10,
    "communication": 0.05,
}


@dataclass(frozen=True)
class Settings:
    # Storage
    database_url: str
    data_dir: Path
    upload_dir: Path
    vector_dir: Path
    model_dir: Path

    # NLP / ML
    embedding_model: str
    spacy_model: str
    vector_backend: str  # "faiss" | "chroma"

    # Speech-to-text
    whisper_mode: str  # "local" | "api"
    whisper_model: str

    # Generative AI
    llm_provider: str  # "anthropic" | "openai" | "none"
    llm_model: str
    anthropic_api_key: str
    openai_api_key: str
    openai_base_url: str
    llm_max_tokens: int
    agent_max_steps: int

    # Ingestion limits (security validation constraint, Stage 1)
    max_resume_mb: float
    max_jd_mb: float
    max_video_mb: float
    max_video_seconds: int
    max_batch_files: int

    # Bootstrap admin account
    admin_username: str
    admin_password: str

    allow_fallbacks: bool = True

    # Login throttle (services/users.py). Behind a reverse proxy, set trusted_proxy_hops
    # to the number of proxies in front of the app so the client IP is read from
    # X-Forwarded-For; 0 never trusts that header. login_max_failures_per_ip=0 disables
    # the per-IP cap.
    trusted_proxy_hops: int = 0
    login_max_failures: int = 5
    login_max_failures_per_ip: int = 20
    login_window_minutes: int = 15

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def llm_enabled(self) -> bool:
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        if self.llm_provider == "openai":
            # Local OpenAI-compatible servers (Ollama, LM Studio) need no real key.
            return bool(self.openai_api_key or self.openai_base_url)
        return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(_env("INTELLIHIRE_DATA_DIR", str(PROJECT_ROOT / "data")))
    model_dir = Path(_env("INTELLIHIRE_MODEL_DIR", str(PROJECT_ROOT / "models")))
    default_db = f"sqlite:///{(data_dir / 'intellihire.db').as_posix()}"

    provider = _env("LLM_PROVIDER", "").lower()
    if not provider:
        if _env("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif _env("OPENAI_API_KEY") or _env("OPENAI_BASE_URL"):
            provider = "openai"
        else:
            provider = "none"
    default_model = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}.get(provider, "")

    settings = Settings(
        database_url=_env("DATABASE_URL", default_db),
        data_dir=data_dir,
        upload_dir=data_dir / "uploads",
        vector_dir=data_dir / "vectors",
        model_dir=model_dir,
        embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        spacy_model=_env("SPACY_MODEL", "en_core_web_sm"),
        vector_backend=_env("VECTOR_BACKEND", "faiss").lower(),
        whisper_mode=_env("WHISPER_MODE", "local").lower(),
        whisper_model=_env("WHISPER_MODEL", "base"),
        llm_provider=provider,
        llm_model=_env("LLM_MODEL", default_model),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_base_url=_env("OPENAI_BASE_URL"),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 1500),
        agent_max_steps=_env_int("AGENT_MAX_STEPS", 6),
        max_resume_mb=_env_positive_float("MAX_RESUME_MB", 5),
        max_jd_mb=_env_positive_float("MAX_JD_MB", 5),
        max_video_mb=_env_positive_float("MAX_VIDEO_MB", 200),
        max_video_seconds=_env_positive_int("MAX_VIDEO_SECONDS", 300),
        max_batch_files=_env_positive_int("MAX_BATCH_FILES", 100),
        admin_username=_env("ADMIN_USERNAME", "admin"),
        admin_password=_env("ADMIN_PASSWORD", "admin123"),
        allow_fallbacks=_env_bool("ALLOW_FALLBACKS", True),
        trusted_proxy_hops=_env_non_negative_int("TRUSTED_PROXY_HOPS", 0),
        login_max_failures=_env_positive_int("LOGIN_MAX_FAILURES", 5),
        login_max_failures_per_ip=_env_non_negative_int("LOGIN_MAX_FAILURES_PER_IP", 20),
        login_window_minutes=_env_positive_int("LOGIN_WINDOW_MINUTES", 15),
    )
    for directory in (settings.data_dir, settings.upload_dir, settings.vector_dir, settings.model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return settings
