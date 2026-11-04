"""Runtime settings, loaded from the environment and the repo's .env file.

Every other module gets its configuration from get_settings() at the bottom of this file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# This file is src/sec_agent/config.py, so parents[2] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Exported to the process environment, not just read into Settings, because
# libraries read some of these directly (TOKENIZERS_PARALLELISM, for example).
# Real environment variables take precedence over the file.
load_dotenv(REPO_ROOT / ".env", override=False)


class ConfigError(RuntimeError):
    """A required setting is missing or invalid."""


class Settings(BaseSettings):
    # Fields without an alias read SEC_AGENT_<NAME> (data_dir <- SEC_AGENT_DATA_DIR).
    # Unknown variables in .env are ignored.
    model_config = SettingsConfigDict(env_prefix="SEC_AGENT_", extra="ignore")

    # Optional so commands that don't need them still start; the require_* methods below
    # fail with a clear message where one is needed. SecretStr keeps the key out of logs.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    sec_user_agent: str | None = Field(default=None, validation_alias="SEC_USER_AGENT")

    # Defaults to local disk: DuckDB, Chroma, and SQLite are unreliable on
    # VirtualBox shared folders, where the repo lives.
    data_dir: Path = Path("~/.local/share/sec-agent")
    model_strong: str = "claude-opus-5-5"
    model_worker: str = "claude-haiku-5-5"
    # Hard stop for one run's LLM spend. gt=0 so the cap can't be switched off with a zero.
    max_usd_per_run: float = Field(default=10.0, gt=0)
    llm_cache: bool = True

    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, value: Path) -> Path:
        return value.expanduser()  # Path does not expand "~" on its own

    @property
    def raw_filings_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_filings_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "sec_agent.duckdb"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def tree_db_path(self) -> Path:
        return self.data_dir / "tot_tree.sqlite"

    @property
    def checkpoint_db_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def llm_cache_path(self) -> Path:
        return self.data_dir / "llm_cache.sqlite"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def ensure_dirs(self) -> None:
        # Directories only; database files are created by the code that owns them.
        for path in (
            self.data_dir,
            self.raw_filings_dir,
            self.parsed_filings_dir,
            self.chroma_dir,
            self.runs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def require_api_key(self) -> str:
        if self.anthropic_api_key is None or not self.anthropic_api_key.get_secret_value():
            raise ConfigError("ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example).")
        return self.anthropic_api_key.get_secret_value()

    def require_sec_user_agent(self) -> str:
        # The "@" test only catches a missing email; it doesn't validate one.
        if not self.sec_user_agent or "@" not in self.sec_user_agent:
            raise ConfigError(
                "SEC_USER_AGENT must include a name and contact email, as EDGAR requires. "
                "Add it to .env (see .env.example)."
            )
        return self.sec_user_agent


@lru_cache
def get_settings() -> Settings:
    # Read once per process; tests that change the environment call get_settings.cache_clear().
    return Settings()
