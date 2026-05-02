from pathlib import Path

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
    database_path: str


class DirectorySettingsUpdate(BaseModel):
    model_directory: str
    output_directory: str
    dataset_directory: str
    checkpoint_directory: str
    working_directory: str


DIRECTORY_KEYS = (
    "model_directory",
    "output_directory",
    "dataset_directory",
    "checkpoint_directory",
    "working_directory",
)


def _stringify(path: Path) -> str:
    return str(path.expanduser())


def fallback_directory_settings(settings: AppSettings) -> dict[str, str]:
    return {
        "model_directory": _stringify(settings.fallback_model_directory),
        "output_directory": _stringify(settings.fallback_output_directory),
        "dataset_directory": _stringify(settings.fallback_dataset_directory),
        "checkpoint_directory": _stringify(settings.fallback_checkpoint_directory),
        "working_directory": _stringify(settings.fallback_working_directory),
    }


async def read_directory_settings(settings: AppSettings) -> DirectorySettings:
    await initialize_database(settings.database_path)
    values = fallback_directory_settings(settings)

    async with aiosqlite.connect(settings.database_path) as connection:
        async with connection.execute("SELECT key, value FROM directory_settings") as cursor:
            rows = await cursor.fetchall()

    for key, value in rows:
        if key in values:
            values[key] = value

    return DirectorySettings(**values, database_path=_stringify(settings.database_path))


async def write_directory_settings(
    settings: AppSettings,
    update: DirectorySettingsUpdate,
) -> DirectorySettings:
    await initialize_database(settings.database_path)
    values = update.model_dump()

    async with aiosqlite.connect(settings.database_path) as connection:
        await connection.executemany(
            """
            INSERT INTO directory_settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            [(key, values[key]) for key in DIRECTORY_KEYS],
        )
        await connection.commit()

    return await read_directory_settings(settings)
