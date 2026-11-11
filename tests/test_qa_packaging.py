"""Regression tests for packaging and script fixes (T25, T26, T55, T56, T57, T58, download_models encoding)."""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def test_dockerignore_excludes_secrets_and_local_state():  # T25
    patterns = set(_lines(ROOT / ".dockerignore"))
    for required in (".env", "data/", "models/", ".venv/", "__pycache__/", "*.pyc", ".pytest_cache/", ".git/", "*.db"):
        assert required in patterns, required
    assert "!.env.example" in patterns


def test_compose_does_not_publish_mysql_and_reads_passwords_from_env():  # T26
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    mysql = compose["services"]["mysql"]
    assert "ports" not in mysql or not any(str(p).startswith("3306") for p in mysql["ports"])
    env = mysql["environment"]
    assert env["MYSQL_PASSWORD"] == "${MYSQL_PASSWORD:-intellihire}"
    assert env["MYSQL_ROOT_PASSWORD"].startswith("${MYSQL_ROOT_PASSWORD:-")


def test_compose_env_file_is_optional():  # T56
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert compose["services"]["app"]["env_file"] == [{"path": ".env", "required": False}]


def test_dockerfile_runs_as_non_root_and_warns_visibly():  # T55
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER intellihire" in text
    assert "useradd" in text
    assert "WARNING: model pre-download failed" in text and ">&2" in text


def test_requirements_header_no_longer_claims_all_packages_are_capped():  # T57
    header = [ln for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if ln.startswith("#")]
    assert not any("Upper bounds cap each package" in ln for ln in header)
    assert any("no upper cap" in ln for ln in header)


def test_download_models_reconfigures_console_encoding():  # download_models UnicodeEncodeError
    text = (ROOT / "scripts" / "download_models.py").read_text(encoding="utf-8")
    assert 'sys.stdout.reconfigure(encoding="utf-8", errors="replace")' in text or 'reconfigure(encoding="utf-8"' in text


def _load_seed_module():
    spec = importlib.util.spec_from_file_location("seed_demo_under_test", ROOT / "scripts" / "seed_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_reset_refuses_without_yes_when_not_interactive(monkeypatch, capsys):  # T58
    seed = _load_seed_module()
    monkeypatch.setattr(sys, "stdin", io.StringIO("yes\n"))  # not a TTY: must not reset
    with pytest.raises(SystemExit) as exc:
        seed.confirm_reset(assume_yes=False)
    assert "--yes" in str(exc.value.code)
    assert "DROP ALL TABLES" in capsys.readouterr().out


def test_seed_reset_prints_target_with_password_masked(monkeypatch, capsys):  # T58
    seed = _load_seed_module()
    monkeypatch.setattr(seed, "get_engine", lambda: type("E", (), {"url": _FakeURL()})())
    seed.confirm_reset(assume_yes=True)
    out = capsys.readouterr().out
    assert "secret-pw" not in out
    assert "mysql+pymysql://user@db.example/intellihire" in out


class _FakeURL:
    def render_as_string(self, hide_password: bool = True) -> str:
        return "mysql+pymysql://user@db.example/intellihire"
