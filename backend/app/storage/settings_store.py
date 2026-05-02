from pathlib import Path
from typing import Literal

import aiosqlite
from pydantic import BaseModel

from app.core.settings import AppSettings
from app.storage.database import initialize_database


class DirectorySettings(BaseModel):
    model_directory: str
    output_directory: str
    dataset_directory: str
    checkpoint_directory: str
    working_directory: str
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    database_path: str


class DirectorySettingsUpdate(BaseModel):
    working_directory: str
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


SETTINGS_KEYS = ("device",)


def _stringify(path: Path) -> str:
    return str(path.expanduser())


def _directory_settings_from_working_directory(working_directory: Path) -> dict[str, str]:
    return {
        "model_directory": _stringify(working_directory / "models"),
        "output_directory": _stringify(working_directory / "outputs"),
        "dataset_directory": _stringify(working_directory / "datasets"),
        "checkpoint_directory": _stringify(working_directory / "checkpoints"),
        "working_directory": _stringify(working_directory),
        "device": "auto",
    }


def _ensure_directory_settings(values: dict[str, str]) -> None:
    Path(values["working_directory"]).expanduser().mkdir(parents=True, exist_ok=True)
    for key in (
        "model_directory",
        "output_directory",
        "dataset_directory",
        "checkpoint_directory",
    ):
        Path(values[key]).expanduser().mkdir(parents=True, exist_ok=True)


def fallback_directory_settings(settings: AppSettings) -> dict[str, str]:
    return _directory_settings_from_working_directory(settings.fallback_working_directory)


async def read_directory_settings(settings: AppSettings) -> DirectorySettings:
    await initialize_database(settings.database_path)
    values = fallback_directory_settings(settings)

    async with aiosqlite.connect(settings.database_path) as connection:
        async with connection.execute("SELECT key, value FROM directory_settings") as cursor:
            rows = await cursor.fetchall()

    for key, value in rows:
        if key == "device":
            values["device"] = value

    _ensure_directory_settings(values)
    return DirectorySettings(**values, database_path=_stringify(settings.database_path))


async def write_directory_settings(
    settings: AppSettings,
    update: DirectorySettingsUpdate,
) -> DirectorySettings:
    await initialize_database(settings.database_path)
    values = update.model_dump()
    _ensure_directory_settings(fallback_directory_settings(settings))

    async with aiosqlite.connect(settings.database_path) as connection:
        await connection.executemany(
            """
            INSERT INTO directory_settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            [(key, str(values[key])) for key in SETTINGS_KEYS],
        )
        await connection.commit()

    return await read_directory_settings(settings)
