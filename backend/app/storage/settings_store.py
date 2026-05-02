from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.core.paths import stringify_path
from app.core.settings import AppSettings
from app.storage.database import initialize_database


class DirectorySettings(BaseModel):
    model_directory: str
    output_directory: str
    dataset_directory: str
    checkpoint_directory: str
    working_directory: str
    classification_dataset_directory: str
    classification_output_directory: str
    training_dataset_directory: str
    training_output_directory: str
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    database_path: str


class DirectorySettingsUpdate(BaseModel):
    working_directory: str
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


SETTINGS_KEYS = ("working_directory", "device")


def _stringify(path: Path) -> str:
    return stringify_path(path)


def _directory_settings_from_working_directory(working_directory: Path) -> dict[str, str]:
    return {
        "model_directory": _stringify(working_directory / "models"),
        "output_directory": _stringify(working_directory / "outputs"),
        "dataset_directory": _stringify(working_directory / "datasets"),
        "checkpoint_directory": _stringify(working_directory / "checkpoints"),
        "working_directory": _stringify(working_directory),
        "classification_dataset_directory": _stringify(
            working_directory / "classification" / "datasets"
        ),
        "classification_output_directory": _stringify(
            working_directory / "classification" / "outputs"
        ),
        "training_dataset_directory": _stringify(working_directory / "training" / "datasets"),
        "training_output_directory": _stringify(working_directory / "training" / "outputs"),
        "device": "auto",
    }


def _ensure_directory_settings(values: dict[str, str]) -> None:
    Path(values["working_directory"]).expanduser().mkdir(parents=True, exist_ok=True)
    for key in (
        "model_directory",
        "output_directory",
        "dataset_directory",
        "classification_dataset_directory",
        "classification_output_directory",
        "training_dataset_directory",
        "training_output_directory",
        "checkpoint_directory",
    ):
        Path(values[key]).expanduser().mkdir(parents=True, exist_ok=True)


def fallback_directory_settings(settings: AppSettings) -> dict[str, str]:
    return {
        "model_directory": _stringify(settings.image_classification_model_directory),
        "output_directory": _stringify(
            settings.image_classification_classification_output_directory
        ),
        "dataset_directory": _stringify(
            settings.image_classification_classification_dataset_directory
        ),
        "checkpoint_directory": _stringify(
            settings.image_classification_training_checkpoint_directory
        ),
        "working_directory": _stringify(settings.image_classification_working_directory),
        "classification_dataset_directory": _stringify(
            settings.image_classification_classification_dataset_directory
        ),
        "classification_output_directory": _stringify(
            settings.image_classification_classification_output_directory
        ),
        "training_dataset_directory": _stringify(
            settings.image_classification_training_dataset_directory
        ),
        "training_output_directory": _stringify(
            settings.image_classification_training_output_directory
        ),
        "device": settings.device,
    }


async def read_directory_settings(settings: AppSettings) -> DirectorySettings:
    await initialize_database(settings.database_path)
    values = fallback_directory_settings(settings)

    _ensure_directory_settings(values)
    return DirectorySettings(**values, database_path=_stringify(settings.database_path))


async def write_directory_settings(
    settings: AppSettings,
    update: DirectorySettingsUpdate,
) -> DirectorySettings:
    del settings, update
    raise ValueError("Directory settings are configured by the root .env file.")
