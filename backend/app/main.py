from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.settings import AppSettings, get_settings
from app.image_classification.device import select_device
from app.image_classification.model_resolver import resolve_model_ref
from app.image_classification.scanner import scan_images
from app.image_classification.schemas import (
    BatchInferenceRequest,
    RunCreateResponse,
    TrainingRequest,
)
from app.image_classification.service import validate_imagefolder_dataset
from app.runs.executor import RunExecutor
from app.runtime.hardware import detect_hardware
from app.storage.database import initialize_database
from app.storage.runs_store import (
    RunCreate,
    RunRecord,
    append_run_log,
    create_run,
    get_run,
    list_run_logs,
    mark_unfinished_runs_interrupted,
)
from app.storage.settings_store import (
    DirectorySettings,
    DirectorySettingsUpdate,
    read_directory_settings,
    write_directory_settings,
)
from app.tasks.registry import TaskDefinition, list_tasks


class ModelOption(BaseModel):
    label: str
    path: str
    source: str


class ModelOptionsResponse(BaseModel):
    local_models: list[ModelOption]
    recommended_hf_models: list[ModelOption]


RECOMMENDED_HF_MODELS = (
    ModelOption(
        label="Microsoft ResNet-50",
        path="microsoft/resnet-50",
        source="huggingface",
    ),
    ModelOption(
        label="Google ViT Base",
        path="google/vit-base-patch16-224-in21k",
        source="huggingface",
    ),
    ModelOption(
        label="Facebook ConvNeXT Tiny",
        path="facebook/convnext-tiny-224",
        source="huggingface",
    ),
)


def list_local_model_options(model_directory: str) -> list[ModelOption]:
    root = Path(model_directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    options: list[ModelOption] = []

    for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if child.is_dir() and (child / "config.json").is_file():
            options.append(
                ModelOption(
                    label=child.name,
                    path=str(child),
                    source="local",
                )
            )

    return options


def _settings() -> AppSettings:
    get_settings.cache_clear()
    return get_settings()


@asynccontextmanager
async def lifespan(api: FastAPI) -> AsyncIterator[None]:
    settings = _settings()
    await initialize_database(settings.database_path)
    await mark_unfinished_runs_interrupted(settings.database_path)
    api.state.run_executor = RunExecutor(settings.database_path)
    try:
        yield
    finally:
        api.state.run_executor.shutdown()


def create_app() -> FastAPI:
    api = FastAPI(title="Galaxy AI Backend", version="0.1.0", lifespan=lifespan)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://localhost:5173",
            "http://localhost:5174",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    @api.get("/models/options", response_model=ModelOptionsResponse)
    async def model_options(model_directory: str) -> ModelOptionsResponse:
        return ModelOptionsResponse(
            local_models=list_local_model_options(model_directory),
            recommended_hf_models=list(RECOMMENDED_HF_MODELS),
        )

    @api.get("/tasks")
    async def tasks() -> dict[str, list[TaskDefinition]]:
        return {"tasks": list_tasks()}

    @api.post(
        "/image-classification/inference",
        response_model=RunCreateResponse,
        status_code=202,
    )
    async def create_image_classification_inference(
        request: Request,
        inference_request: BatchInferenceRequest,
    ) -> RunCreateResponse:
        settings = _settings()
        directory_settings = await read_directory_settings(settings)
        device = select_device(inference_request.device)
        model_directory = Path(
            inference_request.model_directory or directory_settings.model_directory
        )

        try:
            image_paths = scan_images(
                inference_request.input_directory,
                inference_request.input_paths,
                inference_request.recursive,
            )
            resolved = resolve_model_ref(
                inference_request.model_ref,
                inference_request.allow_download,
                model_directory,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        output_base = Path(
            inference_request.output_directory or directory_settings.output_directory
        )
        run = await create_run(
            settings.database_path,
            RunCreate(
                run_type="image_classification_inference",
                request=inference_request.model_dump(),
                hardware_backend=device,
                model_ref=inference_request.model_ref,
                input_path=inference_request.input_directory
                or ",".join(inference_request.input_paths or []),
                output_path=str(output_base),
                total_items=len(image_paths),
            ),
        )
        await append_run_log(
            settings.database_path,
            run.run_id,
            (
                f"Model source resolved as {resolved.source}; "
                f"allow_download={inference_request.allow_download}."
            ),
        )
        request.app.state.run_executor.submit(
            run.run_id,
            __import__(
                "app.image_classification.service",
                fromlist=["run_batch_inference"],
            ).run_batch_inference,
            {
                "model_ref": inference_request.model_ref,
                "allow_download": inference_request.allow_download,
                "model_directory": model_directory,
                "image_paths": image_paths,
                "output_directory": output_base / run.run_id,
                "batch_size": inference_request.batch_size,
                "top_k": inference_request.top_k,
                "device": device,
            },
        )
        return RunCreateResponse(run_id=run.run_id, status=run.status)

    @api.post(
        "/image-classification/training",
        response_model=RunCreateResponse,
        status_code=202,
    )
    async def create_image_classification_training(
        request: Request,
        training_request: TrainingRequest,
    ) -> RunCreateResponse:
        settings = _settings()
        directory_settings = await read_directory_settings(settings)
        device = select_device(training_request.device)
        model_directory = Path(
            training_request.model_directory or directory_settings.model_directory
        )

        try:
            labels = validate_imagefolder_dataset(training_request.dataset_directory)
            resolved = resolve_model_ref(
                training_request.base_model_ref,
                training_request.allow_download,
                model_directory,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        output_base = Path(training_request.output_directory or directory_settings.output_directory)
        checkpoint_base = Path(
            training_request.checkpoint_directory or directory_settings.checkpoint_directory
        )
        run = await create_run(
            settings.database_path,
            RunCreate(
                run_type="image_classification_training",
                request=training_request.model_dump(),
                hardware_backend=device,
                model_ref=training_request.base_model_ref,
                input_path=training_request.dataset_directory,
                output_path=str(output_base),
                total_items=len(labels),
            ),
        )
        await append_run_log(
            settings.database_path,
            run.run_id,
            (
                f"Model source resolved as {resolved.source}; "
                f"allow_download={training_request.allow_download}."
            ),
        )
        request.app.state.run_executor.submit(
            run.run_id,
            __import__(
                "app.image_classification.service",
                fromlist=["run_imagefolder_training"],
            ).run_imagefolder_training,
            {
                "base_model_ref": training_request.base_model_ref,
                "allow_download": training_request.allow_download,
                "model_directory": model_directory,
                "dataset_directory": Path(training_request.dataset_directory),
                "output_directory": output_base / run.run_id,
                "checkpoint_directory": checkpoint_base / run.run_id,
                "epochs": training_request.epochs,
                "batch_size": training_request.batch_size,
                "learning_rate": training_request.learning_rate,
                "seed": training_request.seed if training_request.use_seed else None,
                "device": device,
            },
        )
        return RunCreateResponse(run_id=run.run_id, status=run.status)

    @api.get("/runs/{run_id}", response_model=RunRecord)
    async def read_run(run_id: str) -> RunRecord:
        run = await get_run(_settings().database_path, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return run

    @api.get("/runs/{run_id}/logs")
    async def read_run_logs(run_id: str) -> dict[str, list[str]]:
        run = await get_run(_settings().database_path, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return {"logs": await list_run_logs(_settings().database_path, run_id)}

    @api.post("/runs/{run_id}/cancel")
    async def cancel_run(request: Request, run_id: str) -> dict[str, str]:
        try:
            status = await request.app.state.run_executor.cancel(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"run_id": run_id, "status": status}

    return api


app = create_app()
