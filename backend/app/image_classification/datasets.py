from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.image_classification.scanner import SUPPORTED_IMAGE_EXTENSIONS
from app.image_classification.service import safe_label_directory_name
from app.image_classification.workspace import ImageClassificationWorkspace

DatasetMode = Literal["classification", "training"]


class DatasetImageItem(BaseModel):
    file_name: str
    relative_path: str
    size: int


class DatasetLabelGroup(BaseModel):
    label: str
    count: int
    items: list[DatasetImageItem]


class DatasetPreviewResponse(BaseModel):
    mode: DatasetMode
    count: int
    items: list[DatasetImageItem]
    labels: list[DatasetLabelGroup]


class DatasetImportResponse(BaseModel):
    mode: DatasetMode
    imported_count: int


class DatasetClearResponse(BaseModel):
    mode: DatasetMode
    deleted_count: int


@dataclass(frozen=True)
class DatasetUpload:
    file_name: str
    content: bytes
    relative_path: str | None = None


def list_dataset(
    mode: DatasetMode, workspace: ImageClassificationWorkspace
) -> DatasetPreviewResponse:
    workspace.ensure()
    if mode == "classification":
        items = _list_images(workspace.classification_dataset_directory)
        return DatasetPreviewResponse(
            mode=mode,
            count=len(items),
            items=items,
            labels=[],
        )

    train_root = workspace.training_dataset_train_directory
    labels: list[DatasetLabelGroup] = []
    for label_directory in sorted(train_root.iterdir(), key=lambda path: path.name.lower()):
        if not label_directory.is_dir():
            continue
        items = _list_images(label_directory)
        if not items:
            continue
        labels.append(
            DatasetLabelGroup(
                label=label_directory.name,
                count=len(items),
                items=items,
            )
        )

    return DatasetPreviewResponse(
        mode=mode,
        count=sum(label.count for label in labels),
        items=[],
        labels=labels,
    )


def import_dataset(
    *,
    mode: DatasetMode,
    label: str | None,
    uploads: list[DatasetUpload],
    workspace: ImageClassificationWorkspace,
) -> DatasetImportResponse:
    workspace.ensure()
    if mode == "training":
        if not label or not label.strip():
            raise ValueError("Training dataset import requires a label.")
        base_directory = workspace.training_dataset_train_directory / safe_label_directory_name(
            label
        )
    else:
        base_directory = workspace.classification_dataset_directory

    base_directory.mkdir(parents=True, exist_ok=True)
    used_destinations: set[Path] = set()
    imported_count = 0
    for upload in uploads:
        source_name = upload.relative_path or upload.file_name
        if not _is_supported_image_name(source_name):
            continue
        destination = _safe_destination(base_directory, source_name)
        destination = _unique_destination(destination, used_destinations)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(upload.content)
        imported_count += 1

    return DatasetImportResponse(mode=mode, imported_count=imported_count)


def clear_dataset(
    mode: DatasetMode,
    workspace: ImageClassificationWorkspace,
) -> DatasetClearResponse:
    workspace.ensure()
    target = (
        workspace.classification_dataset_directory
        if mode == "classification"
        else workspace.training_dataset_train_directory
    )
    deleted_count = len(_list_images(target)) if target.exists() else 0
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return DatasetClearResponse(mode=mode, deleted_count=deleted_count)


def dataset_image_path(
    *,
    mode: DatasetMode,
    relative_path: str,
    label: str | None,
    workspace: ImageClassificationWorkspace,
) -> Path:
    workspace.ensure()
    if mode == "training":
        if not label or not label.strip():
            raise ValueError("Training dataset image preview requires a label.")
        base_directory = workspace.training_dataset_train_directory / safe_label_directory_name(
            label
        )
    else:
        base_directory = workspace.classification_dataset_directory

    path = Path(relative_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Invalid dataset image path.")
    if not _is_supported_image_name(path.name):
        raise ValueError("Unsupported dataset image type.")

    candidate = (base_directory / path).resolve()
    try:
        candidate.relative_to(base_directory.resolve())
    except ValueError as exc:
        raise ValueError("Invalid dataset image path.") from exc

    if not candidate.is_file():
        raise FileNotFoundError(str(candidate))
    return candidate


def _list_images(root: Path) -> list[DatasetImageItem]:
    if not root.exists():
        return []

    items: list[DatasetImageItem] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root)).lower()):
        if not path.is_file() or not _is_supported_image_name(path.name):
            continue
        items.append(
            DatasetImageItem(
                file_name=path.name,
                relative_path=path.relative_to(root).as_posix(),
                size=path.stat().st_size,
            )
        )
    return items


def _is_supported_image_name(name: str) -> bool:
    return Path(name).suffix.lower().lstrip(".") in SUPPORTED_IMAGE_EXTENSIONS


def _safe_destination(base_directory: Path, relative_path: str) -> Path:
    raw_parts = Path(relative_path).parts
    parts = [_safe_path_part(part) for part in raw_parts if part not in {"", ".", ".."}]
    parts = [part for part in parts if part]
    if not parts:
        parts = ["image"]
    if not _is_supported_image_name(parts[-1]):
        parts[-1] = f"{parts[-1]}.jpg"
    return base_directory.joinpath(*parts)


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip(" ._")
    return cleaned or "item"


def _unique_destination(path: Path, used_destinations: set[Path]) -> Path:
    candidate = path
    suffix = 2
    while candidate.exists() or candidate in used_destinations:
        candidate = path.with_name(f"{path.stem}__{suffix}{path.suffix}")
        suffix += 1
    used_destinations.add(candidate)
    return candidate
