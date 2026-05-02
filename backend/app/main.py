import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from app.core.paths import normalize_local_path
from app.core.settings import AppSettings, get_settings
from app.image_classification.datasets import (
    DatasetClearResponse,
    DatasetImportResponse,
    DatasetMode,
    DatasetPreviewResponse,
    DatasetUpload,
    clear_dataset,
    dataset_image_path,
    import_dataset,
    list_dataset,
)
from app.image_classification.device import select_device
from app.image_classification.model_compatibility import (
    image_classification_compatibility_error,
    image_classification_inference_error,
    training_base_model_compatibility_error,
)
from app.image_classification.model_resolver import ResolvedModel, resolve_model_ref
from app.image_classification.scanner import scan_images
from app.image_classification.schemas import (
    BatchInferenceRequest,
    RunCreateResponse,
    TrainingRequest,
)
from app.image_classification.service import (
    count_imagefolder_training_images,
    training_final_model_directory,
    validate_imagefolder_dataset,
)
from app.image_classification.workspace import image_classification_workspace
from app.runtime.hardware import detect_hardware
from app.runtime.run_executor import RunExecutor
from app.storage.database import initialize_database
from app.storage.runs_store import (
    ActiveRunExistsError,
    RunCreate,
    RunRecord,
    append_run_log,
    create_run_with_single_active_lock,
    get_active_run,
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
    compatible: bool
    compatibility_error: str | None = None
    requires_download: bool = False


class ModelOptionsResponse(BaseModel):
    local_models: list[ModelOption]
    recommended_hf_models: list[ModelOption]


class ActiveRunResponse(BaseModel):
    run: RunRecord | None


class OpenRunOutputResponse(BaseModel):
    run_id: str
    output_path: str
    opened: bool


RECOMMENDED_HF_MODELS = (
    ModelOption(
        label="Microsoft ResNet-50",
        path="microsoft/resnet-50",
        source="huggingface",
        compatible=True,
        requires_download=True,
    ),
    ModelOption(
        label="Google ViT Base",
        path="google/vit-base-patch16-224-in21k",
        source="huggingface",
        compatible=True,
        requires_download=True,
    ),
    ModelOption(
        label="Facebook ConvNeXT Tiny",
        path="facebook/convnext-tiny-224",
        source="huggingface",
        compatible=True,
        requires_download=True,
    ),
)


def list_local_model_options(model_directory: str) -> list[ModelOption]:
    root = normalize_local_path(model_directory)
    root.mkdir(parents=True, exist_ok=True)
    options: list[ModelOption] = []
    candidates = _local_model_candidates(root)

    for label, model_path in candidates:
        compatibility_error = image_classification_compatibility_error(model_path)
        options.append(
            ModelOption(
                label=label,
                path=str(model_path),
                source="local",
                compatible=compatibility_error is None,
                compatibility_error=compatibility_error,
                requires_download=False,
            )
        )

    return options


def _local_model_candidates(root: Path) -> list[tuple[str, Path]]:
    seen: set[Path] = set()
    candidates: list[tuple[str, Path]] = []

    def add_candidate(label: str, model_path: Path) -> None:
        resolved_path = model_path.resolve()
        if resolved_path in seen:
            return
        seen.add(resolved_path)
        candidates.append((label, model_path))

    if (root / "config.json").is_file():
        add_candidate(root.name, root)

    for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if child.is_dir() and (child / "config.json").is_file():
            add_candidate(child.name, child)

    for cache_root in sorted(root.glob("models--*"), key=lambda path: path.name.lower()):
        snapshots_root = cache_root / "snapshots"
        if not snapshots_root.is_dir():
            continue
        label = _label_for_hugging_face_cache(cache_root)
        for snapshot in sorted(snapshots_root.iterdir(), key=lambda path: path.name.lower()):
            if snapshot.is_dir() and (snapshot / "config.json").is_file():
                add_candidate(label, snapshot)

    return sorted(candidates, key=lambda item: (item[0].lower(), str(item[1]).lower()))


def _label_for_hugging_face_cache(cache_root: Path) -> str:
    if not cache_root.name.startswith("models--"):
        return cache_root.name
    return cache_root.name.removeprefix("models--").replace("--", "/")


def ensure_image_classification_model(resolved: ResolvedModel) -> None:
    if resolved.source != "local_path":
        return

    compatibility_error = image_classification_inference_error(Path(resolved.load_ref))
    if compatibility_error is not None:
        raise ValueError(compatibility_error)


def ensure_training_base_model(resolved: ResolvedModel) -> None:
    if resolved.source != "local_path":
        return

    compatibility_error = training_base_model_compatibility_error(Path(resolved.load_ref))
    if compatibility_error is not None:
        raise ValueError(compatibility_error)


def normalized_model_ref(value: str) -> str:
    path = Path(value).expanduser()
    if value.startswith((".", "/", "~")) or path.exists():
        return str(normalize_local_path(value))
    return value


def open_output_directory(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return

    command = ["open", str(path)] if sys.platform == "darwin" else ["xdg-open", str(path)]
    subprocess.run(command, check=True)


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
        try:
            return await write_directory_settings(_settings(), update)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/models/options", response_model=ModelOptionsResponse)
    async def model_options(model_directory: str) -> ModelOptionsResponse:
        return ModelOptionsResponse(
            local_models=list_local_model_options(model_directory),
            recommended_hf_models=list(RECOMMENDED_HF_MODELS),
        )

    @api.get("/tasks")
    async def tasks() -> dict[str, list[TaskDefinition]]:
        return {"tasks": list_tasks()}

    @api.get(
        "/image-classification/datasets",
        response_model=DatasetPreviewResponse,
    )
    async def image_classification_dataset(mode: DatasetMode) -> DatasetPreviewResponse:
        workspace = image_classification_workspace(_settings()).ensure()
        return list_dataset(mode, workspace)

    @api.post(
        "/image-classification/datasets/import",
        response_model=DatasetImportResponse,
    )
    async def import_image_classification_dataset(request: Request) -> DatasetImportResponse:
        form = await request.form()
        mode = str(form.get("mode") or "")
        if mode not in {"classification", "training"}:
            raise HTTPException(
                status_code=400,
                detail="Dataset import mode must be classification or training.",
            )
        label = form.get("label")
        relative_paths = [
            str(value)
            for value in (form.getlist("relative_paths[]") or form.getlist("relative_paths"))
        ]
        file_values = [value for value in form.getlist("files") if isinstance(value, UploadFile)]
        uploads: list[DatasetUpload] = []
        for index, upload_file in enumerate(file_values):
            content = await upload_file.read()
            relative_path = relative_paths[index] if index < len(relative_paths) else None
            uploads.append(
                DatasetUpload(
                    file_name=upload_file.filename or f"image-{index + 1}",
                    content=content,
                    relative_path=relative_path,
                )
            )

        try:
            return import_dataset(
                mode=mode,
                label=str(label) if label is not None else None,
                uploads=uploads,
                workspace=image_classification_workspace(_settings()).ensure(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/image-classification/datasets/image")
    async def image_classification_dataset_image(
        mode: DatasetMode,
        relative_path: str,
        label: str | None = None,
    ) -> FileResponse:
        try:
            path = dataset_image_path(
                mode=mode,
                relative_path=relative_path,
                label=label,
                workspace=image_classification_workspace(_settings()).ensure(),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset image not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(path)

    @api.delete(
        "/image-classification/datasets",
        response_model=DatasetClearResponse,
    )
    async def clear_image_classification_dataset(mode: DatasetMode) -> DatasetClearResponse:
        workspace = image_classification_workspace(_settings()).ensure()
        return clear_dataset(mode, workspace)

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
        workspace = image_classification_workspace(settings).ensure()
        device = select_device(inference_request.device)
        model_directory = normalize_local_path(
            inference_request.model_directory or workspace.model_directory
        )
        input_directory = (
            None
            if inference_request.input_paths
            else str(
                normalize_local_path(
                    inference_request.input_directory or workspace.classification_dataset_directory
                )
            )
        )
        input_paths = (
            [str(normalize_local_path(path)) for path in inference_request.input_paths]
            if inference_request.input_paths
            else None
        )

        try:
            image_paths = scan_images(
                input_directory,
                input_paths,
                inference_request.recursive,
            )
            resolved = resolve_model_ref(
                inference_request.model_ref,
                inference_request.allow_download,
                model_directory,
            )
            ensure_image_classification_model(resolved)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        output_base = normalize_local_path(
            inference_request.output_directory or workspace.classification_output_directory
        )
        normalized_inference_request = inference_request.model_dump()
        normalized_inference_request.update(
            {
                "model_ref": normalized_model_ref(inference_request.model_ref),
                "model_directory": str(model_directory),
                "input_directory": input_directory,
                "input_paths": input_paths,
                "output_directory": str(output_base),
            }
        )
        try:
            run = await create_run_with_single_active_lock(
                settings.database_path,
                RunCreate(
                    run_type="image_classification_inference",
                    request=normalized_inference_request,
                    hardware_backend=device,
                    model_ref=normalized_inference_request["model_ref"],
                    input_path=input_directory or ",".join(input_paths or []),
                    output_path=str(output_base),
                    total_items=len(image_paths),
                ),
            )
        except ActiveRunExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
                "input_root": normalize_local_path(input_directory) if input_directory else None,
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
        workspace = image_classification_workspace(settings).ensure()
        device = select_device(training_request.device)
        model_directory = normalize_local_path(
            training_request.model_directory or workspace.model_directory
        )
        dataset_directory = str(
            normalize_local_path(
                training_request.dataset_directory or workspace.training_dataset_directory
            )
        )

        try:
            validate_imagefolder_dataset(dataset_directory)
            training_image_count = count_imagefolder_training_images(dataset_directory)
            resolved = resolve_model_ref(
                training_request.base_model_ref,
                training_request.allow_download,
                model_directory,
            )
            ensure_training_base_model(resolved)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        output_base = normalize_local_path(
            training_request.output_directory or workspace.training_output_directory
        )
        final_model_directory = training_final_model_directory(
            model_directory,
            training_request.base_model_ref,
        )
        normalized_training_request = training_request.model_dump()
        normalized_training_request.update(
            {
                "base_model_ref": normalized_model_ref(training_request.base_model_ref),
                "model_directory": str(model_directory),
                "dataset_directory": dataset_directory,
                "output_directory": str(output_base),
            }
        )
        try:
            run = await create_run_with_single_active_lock(
                settings.database_path,
                RunCreate(
                    run_type="image_classification_training",
                    request=normalized_training_request,
                    hardware_backend=device,
                    model_ref=normalized_training_request["base_model_ref"],
                    input_path=dataset_directory,
                    output_path=str(final_model_directory),
                    total_items=training_image_count,
                ),
            )
        except ActiveRunExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        temporary_model_directory = model_directory / f".training-{run.run_id}"
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
                "dataset_directory": Path(dataset_directory),
                "output_directory": output_base / run.run_id,
                "temporary_model_directory": temporary_model_directory,
                "final_model_directory": final_model_directory,
                "epochs": training_request.epochs,
                "batch_size": training_request.batch_size,
                "learning_rate": training_request.learning_rate,
                "seed": training_request.seed if training_request.use_seed else None,
                "device": device,
            },
        )
        return RunCreateResponse(run_id=run.run_id, status=run.status)

    @api.get("/runs/active", response_model=ActiveRunResponse)
    async def read_active_run() -> ActiveRunResponse:
        return ActiveRunResponse(run=await get_active_run(_settings().database_path))

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

    @api.post("/runs/{run_id}/open-output", response_model=OpenRunOutputResponse)
    async def open_run_output(run_id: str) -> OpenRunOutputResponse:
        run = await get_run(_settings().database_path, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        if run.status != "completed":
            raise HTTPException(
                status_code=400,
                detail="Run output can only be opened after the run completes.",
            )
        if not run.output_path:
            raise HTTPException(status_code=400, detail="Run output path is empty.")

        output_path = normalize_local_path(run.output_path)
        if not output_path.exists() or not output_path.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"Run output directory not found: {output_path}",
            )

        try:
            open_output_directory(output_path)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return OpenRunOutputResponse(
            run_id=run.run_id,
            output_path=str(output_path),
            opened=True,
        )

    @api.post("/runs/{run_id}/cancel")
    async def cancel_run(request: Request, run_id: str) -> dict[str, str]:
        try:
            status = await request.app.state.run_executor.cancel(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"run_id": run_id, "status": status}

    return api


app = create_app()
