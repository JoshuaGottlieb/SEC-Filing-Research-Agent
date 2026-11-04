from __future__ import annotations

import pytest

from sec_agent.config import ConfigError, Settings


def test_defaults_use_local_disk_and_agreed_models(monkeypatch):
    # Clear the overrides first, so a developer's own .env can't change the result.
    for name in ("SEC_AGENT_DATA_DIR", "SEC_AGENT_MODEL_STRONG", "SEC_AGENT_MODEL_WORKER"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings()
    assert "~" not in str(settings.data_dir)
    assert settings.model_strong == "claude-opus-5-5"
    assert settings.model_worker == "claude-haiku-5-5"


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("SEC_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEC_AGENT_MAX_USD_PER_RUN", "2.5")
    monkeypatch.setenv("SEC_AGENT_LLM_CACHE", "off")
    settings = Settings()
    assert settings.data_dir == tmp_path
    assert settings.max_usd_per_run == 2.5
    assert settings.llm_cache is False


def test_ensure_dirs_creates_storage(settings):
    settings.ensure_dirs()
    assert settings.raw_filings_dir.is_dir()
    assert settings.chroma_dir.is_dir()
    assert settings.runs_dir.is_dir()


def test_missing_api_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        Settings(data_dir=tmp_path).require_api_key()


def test_sec_user_agent_needs_contact_email(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "No Email Here")
    with pytest.raises(ConfigError, match="contact email"):
        Settings().require_sec_user_agent()


def test_api_key_is_hidden_when_printed(settings):
    # repr() ends up in logs and tracebacks, so the key must not appear there.
    assert "test-key" not in repr(settings)
