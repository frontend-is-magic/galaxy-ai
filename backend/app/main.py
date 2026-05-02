from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.settings import AppSettings, get_settings
from app.runtime.hardware import detect_hardware
from app.storage.database import initialize_database
from app.storage.settings_store import (
    DirectorySettings,
    DirectorySettingsUpdate,
    read_directory_settings,
    write_directory_settings,
)
from app.tasks.registry import TaskDefinition, list_tasks


def _settings() -> AppSettings:
    get_settings.cache_clear()
    return get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = _settings()
    await initialize_database(settings.database_path)
    yield


def create_app() -> FastAPI:
    api = FastAPI(title="Galaxy AI Backend", version="0.1.0", lifespan=lifespan)

    @api.get("/health")
    async def health() -> dict[str, object]:
        settings = _settings()
        return {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.version,
            "bind": {
                "host": settings.host,
                "port": settings.port,
                "local_only": settings.local_only,
            },
        }

    @api.get("/runtime/hardware")
    async def hardware() -> dict[str, object]:
        return detect_hardware()

    @api.get("/settings", response_model=DirectorySettings)
    async def get_directory_settings() -> DirectorySettings:
        return await read_directory_settings(_settings())

    @api.put("/settings", response_model=DirectorySettings)
    async def put_directory_settings(update: DirectorySettingsUpdate) -> DirectorySettings:
        return await write_directory_settings(_settings(), update)

    @api.get("/tasks")
    async def tasks() -> dict[str, list[TaskDefinition]]:
        return {"tasks": list_tasks()}

    return api


app = create_app()
