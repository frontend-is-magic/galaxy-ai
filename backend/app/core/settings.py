from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GALAXY_AI_", extra="ignore")

    service_name: str = "galaxy-ai-backend"
    version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 8000
    repo_root: Path = Field(default_factory=_default_repo_root)
    home: Path | None = None

    @property
    def local_only(self) -> bool:
        return self.host == "127.0.0.1"

    @property
    def data_home(self) -> Path:
        return self.home or self.repo_root / ".galaxy-ai"

    @property
    def database_path(self) -> Path:
        return self.data_home / "galaxy-ai.sqlite3"

    @property
    def fallback_model_directory(self) -> Path:
        return self.repo_root / "models"

    @property
    def fallback_output_directory(self) -> Path:
        return self.repo_root / "outputs"

    @property
    def fallback_dataset_directory(self) -> Path:
        return self.repo_root / "datasets"

    @property
    def fallback_checkpoint_directory(self) -> Path:
        return self.repo_root / "checkpoints"

    @property
    def fallback_working_directory(self) -> Path:
        return self.repo_root


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
