import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.paths import normalize_local_path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _env_file_path() -> Path:
    repo_root = os.environ.get("GALAXY_AI_REPO_ROOT")
    if repo_root:
        return Path(repo_root).expanduser().resolve() / ".env"
    return _default_repo_root() / ".env"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GALAXY_AI_", extra="ignore")

    service_name: str = "galaxy-ai-backend"
    version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 8000
    repo_root: Path = Field(default_factory=_default_repo_root)
    home: Path | None = None
    working_directory: Path | None = None
    model_directory: Path | None = None
    classification_dataset_directory: Path | None = None
    classification_output_directory: Path | None = None
    training_dataset_directory: Path | None = None
    training_output_directory: Path | None = None
    device: str = "auto"

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
        return self.fallback_working_directory / "models"

    @property
    def fallback_output_directory(self) -> Path:
        return self.fallback_working_directory / "outputs"

    @property
    def fallback_dataset_directory(self) -> Path:
        return self.fallback_working_directory / "datasets"

    @property
    def fallback_checkpoint_directory(self) -> Path:
        return self.fallback_working_directory / "checkpoints"

    @property
    def fallback_working_directory(self) -> Path:
        return Path.home() / "Documents" / "galaxy-ai"

    @property
    def image_classification_working_directory(self) -> Path:
        return self._configured_path(
            self.working_directory,
            self.fallback_working_directory / "nebula-sorter",
        )

    @property
    def image_classification_model_directory(self) -> Path:
        return self._configured_path(
            self.model_directory,
            self.image_classification_working_directory / "models",
        )

    @property
    def image_classification_classification_dataset_directory(self) -> Path:
        return self._configured_path(
            self.classification_dataset_directory,
            self.image_classification_working_directory / "classification" / "datasets",
        )

    @property
    def image_classification_classification_output_directory(self) -> Path:
        return self._configured_path(
            self.classification_output_directory,
            self.image_classification_working_directory / "classification" / "outputs",
        )

    @property
    def image_classification_training_dataset_directory(self) -> Path:
        return self._configured_path(
            self.training_dataset_directory,
            self.image_classification_working_directory / "training" / "datasets",
        )

    @property
    def image_classification_training_output_directory(self) -> Path:
        return self._configured_path(
            self.training_output_directory,
            self.image_classification_working_directory / "training" / "outputs",
        )

    @property
    def image_classification_training_checkpoint_directory(self) -> Path:
        return self.image_classification_working_directory / "training" / "checkpoints"

    def _configured_path(self, configured: Path | None, fallback: Path) -> Path:
        path = configured or fallback
        return normalize_local_path(path, self.repo_root)


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings(_env_file=_env_file_path())
