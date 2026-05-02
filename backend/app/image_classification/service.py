from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any, Protocol

from app.core.paths import normalize_local_path
from app.image_classification.model_compatibility import (
    image_classification_inference_error,
    training_base_model_compatibility_error,
)
from app.image_classification.model_resolver import materialize_model_ref, resolve_model_ref
from app.image_classification.scanner import SUPPORTED_IMAGE_EXTENSIONS


class LogCallback(Protocol):
    def __call__(self, message: str) -> None: ...


class ProgressCallback(Protocol):
    def __call__(self, processed_items: int, total_items: int | None = None) -> None: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


def training_timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_label_directory_name(label: str) -> str:
    safe_label = re.sub(r"[^\w.-]+", "_", label.strip())
    safe_label = re.sub(r"_+", "_", safe_label).strip("_")
    if safe_label in {"", ".", ".."}:
        return "unknown"
    return safe_label


def safe_model_directory_name(model_ref: str) -> str:
    path = Path(model_ref).expanduser()
    if model_ref.startswith((".", "/", "~")) or path.exists():
        return safe_label_directory_name(path.name)
    return safe_label_directory_name(model_ref.replace("/", "--"))


def _unique_model_directory(path: Path) -> Path:
    candidate = path
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}__{suffix}")
        suffix += 1
    return candidate


def training_final_model_directory(model_directory: Path, base_model_ref: str) -> Path:
    name = f"{safe_model_directory_name(base_model_ref)}-{training_timestamp_for_filename()}"
    return _unique_model_directory(normalize_local_path(model_directory) / name)


def training_processed_items_from_steps(
    *,
    global_step: int,
    max_steps: int,
    total_items: int,
) -> int:
    if max_steps <= 0 or total_items <= 0:
        return 0
    progress = round((max(0, global_step) / max_steps) * total_items)
    return max(0, min(total_items, progress))


def remove_temporary_training_directory(path: Path) -> None:
    if path.name.startswith(".training-") and path.exists():
        shutil.rmtree(path)


def _unique_destination(path: Path, used_destinations: set[Path]) -> Path:
    candidate = path
    suffix = 2
    while candidate in used_destinations or candidate.exists():
        candidate = path.with_name(f"{path.stem}__{suffix}{path.suffix}")
        suffix += 1
    used_destinations.add(candidate)
    return candidate


def copy_classified_image(
    *,
    image_path: Path,
    assigned_label: str,
    output_directory: Path,
    input_root: Path | None,
    used_destinations: set[Path],
) -> Path:
    output_directory = normalize_local_path(output_directory)
    image_path = normalize_local_path(image_path)
    input_root = normalize_local_path(input_root) if input_root is not None else None
    label_directory = output_directory / safe_label_directory_name(assigned_label)
    try:
        relative_path = (
            image_path.resolve().relative_to(input_root.resolve()) if input_root else None
        )
    except ValueError:
        relative_path = None

    destination = label_directory / (relative_path or image_path.name)
    destination = _unique_destination(destination, used_destinations)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination)
    return destination


def validate_imagefolder_dataset(dataset_directory: str) -> list[str]:
    root = Path(dataset_directory).expanduser()
    train_root = root / "train"
    if not train_root.exists() or not train_root.is_dir():
        raise ValueError("ImageFolder dataset must contain train/<label>/ images.")

    labels: list[str] = []
    for child in sorted(train_root.iterdir()):
        if not child.is_dir():
            continue
        has_images = any(
            path.is_file() and path.suffix.lower().lstrip(".") in SUPPORTED_IMAGE_EXTENSIONS
            for path in child.rglob("*")
        )
        if has_images:
            labels.append(child.name)

    if len(labels) < 2:
        raise ValueError("ImageFolder training requires at least two labels with images.")
    return labels


def count_imagefolder_training_images(dataset_directory: str) -> int:
    train_root = Path(dataset_directory).expanduser() / "train"
    return sum(
        1
        for path in train_root.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in SUPPORTED_IMAGE_EXTENSIONS
    )


def run_batch_inference(
    *,
    model_ref: str,
    allow_download: bool,
    model_directory: Path,
    image_paths: list[Path],
    output_directory: Path,
    batch_size: int,
    top_k: int,
    device: str,
    cancel_event: Event,
    log_callback: LogCallback,
    progress_callback: ProgressCallback | None = None,
    input_root: Path | None = None,
) -> dict[str, Any]:
    model_directory = normalize_local_path(model_directory)
    output_directory = normalize_local_path(output_directory)
    input_root = normalize_local_path(input_root) if input_root is not None else None
    image_paths = [normalize_local_path(path) for path in image_paths]

    torch = __import__("torch")
    image_module = __import__("PIL.Image", fromlist=["Image"])
    transformers = __import__(
        "transformers",
        fromlist=["AutoImageProcessor", "AutoModelForImageClassification"],
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    organized_root = output_directory
    used_destinations: set[Path] = set()
    label_directories: dict[str, dict[str, str]] = {}
    organized_items = 0

    resolved = resolve_model_ref(model_ref, allow_download, model_directory)
    if resolved.source == "hf_repo" and allow_download:
        log_callback(f"Explicit Hugging Face download allowed for model: {model_ref}")
    resolved = materialize_model_ref(resolved, log_callback)
    if cancel_event.is_set():
        return {
            "output_path": str(output_directory),
            "total_items": len(image_paths),
            "processed_items": 0,
            "metadata": {},
        }
    compatibility_error = image_classification_inference_error(Path(resolved.load_ref))
    if compatibility_error is not None:
        raise ValueError(compatibility_error)

    image_processor = transformers.AutoImageProcessor.from_pretrained(
        resolved.load_ref,
        local_files_only=resolved.local_files_only,
        cache_dir=str(resolved.cache_dir) if resolved.cache_dir else None,
    )
    model = transformers.AutoModelForImageClassification.from_pretrained(
        resolved.load_ref,
        local_files_only=resolved.local_files_only,
        cache_dir=str(resolved.cache_dir) if resolved.cache_dir else None,
    )
    if cancel_event.is_set():
        return {
            "output_path": str(output_directory),
            "total_items": len(image_paths),
            "processed_items": 0,
            "metadata": {},
        }
    model.to(device)
    model.eval()

    processed = 0
    failed = 0
    for start in range(0, len(image_paths), batch_size):
        if cancel_event.is_set():
            break

        batch_paths = image_paths[start : start + batch_size]
        batch_images = []
        valid_paths = []
        for path in batch_paths:
            try:
                batch_images.append(image_module.open(path).convert("RGB"))
                valid_paths.append(path)
            except Exception as exc:
                failed += 1
                processed += 1
                log_callback(f"Failed to read image {path}: {exc}")
                if progress_callback is not None:
                    progress_callback(processed)
                if cancel_event.is_set():
                    break

        if cancel_event.is_set() or not batch_images:
            continue

        inputs = image_processor(batch_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            top_scores, top_indices = torch.topk(
                probabilities,
                k=min(top_k, probabilities.shape[-1]),
                dim=-1,
            )

        for image_path, scores, indices in zip(valid_paths, top_scores, top_indices, strict=True):
            if cancel_event.is_set():
                break
            predictions = []
            for score, index in zip(scores.tolist(), indices.tolist(), strict=True):
                label = model.config.id2label.get(index) or model.config.id2label.get(str(index))
                predictions.append({"label": str(label or index), "score": float(score)})
            assigned_label = predictions[0]["label"]
            organized_path = copy_classified_image(
                image_path=image_path,
                assigned_label=assigned_label,
                output_directory=output_directory,
                input_root=input_root,
                used_destinations=used_destinations,
            )
            safe_label = safe_label_directory_name(assigned_label)
            label_directories[safe_label] = {
                "label": assigned_label,
                "path": str(output_directory / safe_label),
            }
            organized_items += 1
            log_callback(f"Classified {image_path} as {assigned_label}: {organized_path}")
            processed += 1
            if progress_callback is not None:
                progress_callback(processed)
            if cancel_event.is_set():
                break

    metadata = {
        "model_ref": model_ref,
        "model_source": resolved.source,
        "hardware_backend": device,
        "started_at": started_at,
        "completed_at": _now(),
        "total_items": len(image_paths),
        "processed_items": processed,
        "successful_items": processed - failed,
        "failed_items": failed,
        "organized_root": str(organized_root),
        "organized_items": organized_items,
        "label_directories": label_directories,
    }

    return {
        "output_path": str(output_directory),
        "total_items": len(image_paths),
        "processed_items": processed,
        "metadata": metadata,
    }


def run_imagefolder_training(
    *,
    base_model_ref: str,
    allow_download: bool,
    model_directory: Path,
    dataset_directory: Path,
    output_directory: Path,
    temporary_model_directory: Path,
    final_model_directory: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int | None,
    device: str,
    cancel_event: Event,
    log_callback: LogCallback,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    model_directory = normalize_local_path(model_directory)
    dataset_directory = normalize_local_path(dataset_directory)
    output_directory = normalize_local_path(output_directory)
    temporary_model_directory = normalize_local_path(temporary_model_directory)
    final_model_directory = normalize_local_path(final_model_directory)

    datasets = __import__("datasets", fromlist=["load_dataset"])
    transformers = __import__(
        "transformers",
        fromlist=[
            "AutoImageProcessor",
            "AutoModelForImageClassification",
            "Trainer",
            "TrainerCallback",
            "TrainingArguments",
        ],
    )

    labels = validate_imagefolder_dataset(str(dataset_directory))
    image_count = count_imagefolder_training_images(str(dataset_directory))
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = output_directory / "training_metadata.json"
    started_at = _now()

    resolved = resolve_model_ref(base_model_ref, allow_download, model_directory)
    if resolved.source == "hf_repo" and allow_download:
        log_callback(f"Explicit Hugging Face download allowed for model: {base_model_ref}")
    resolved = materialize_model_ref(resolved, log_callback)
    compatibility_error = training_base_model_compatibility_error(Path(resolved.load_ref))
    if compatibility_error is not None:
        raise ValueError(compatibility_error)

    dataset = datasets.load_dataset("imagefolder", data_dir=str(dataset_directory))
    label_names = list(getattr(dataset["train"].features["label"], "names", labels))
    id2label = {index: label for index, label in enumerate(label_names)}
    label2id = {label: index for index, label in id2label.items()}

    image_processor = transformers.AutoImageProcessor.from_pretrained(
        resolved.load_ref,
        local_files_only=resolved.local_files_only,
        cache_dir=str(resolved.cache_dir) if resolved.cache_dir else None,
    )
    model = transformers.AutoModelForImageClassification.from_pretrained(
        resolved.load_ref,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=resolved.local_files_only,
        cache_dir=str(resolved.cache_dir) if resolved.cache_dir else None,
    )

    def transform(batch: dict[str, Any]) -> dict[str, Any]:
        images = [image.convert("RGB") for image in batch["image"]]
        inputs = image_processor(images, return_tensors="pt")
        inputs["labels"] = batch["label"]
        return inputs

    dataset = dataset.with_transform(transform)
    eval_dataset = dataset.get("validation") or dataset.get("val")

    training_args_kwargs = {
        "output_dir": str(output_directory / "trainer"),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "remove_unused_columns": False,
        "push_to_hub": False,
        "save_strategy": "no",
        "eval_strategy": "epoch" if eval_dataset is not None else "no",
        "report_to": [],
    }
    if seed is not None:
        training_args_kwargs["seed"] = seed

    training_args = transformers.TrainingArguments(**training_args_kwargs)

    if cancel_event.is_set():
        raise RuntimeError("Training was cancelled before start.")

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        processing_class=image_processor,
        callbacks=[
            _training_progress_callback(
                transformers.TrainerCallback,
                image_count,
                progress_callback,
                cancel_event,
            )
        ],
    )
    trainer.train()
    if cancel_event.is_set():
        remove_temporary_training_directory(temporary_model_directory)
        raise RuntimeError("Training was cancelled.")
    try:
        temporary_model_directory.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(temporary_model_directory))
        image_processor.save_pretrained(str(temporary_model_directory))
        normalize_image_classification_model_config(temporary_model_directory, id2label, label2id)
        final_model_directory.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temporary_model_directory), str(final_model_directory))
    except Exception:
        remove_temporary_training_directory(temporary_model_directory)
        raise
    if progress_callback is not None:
        progress_callback(image_count)

    metadata = {
        "base_model_ref": base_model_ref,
        "model_source": resolved.source,
        "dataset_directory": str(dataset_directory),
        "labels": label_names,
        "label2id": label2id,
        "id2label": id2label,
        "hardware_backend": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "final_model_path": str(final_model_directory),
        "started_at": started_at,
        "completed_at": _now(),
    }
    _write_json(metadata_path, metadata)

    return {
        "output_path": str(final_model_directory),
        "total_items": image_count,
        "processed_items": image_count,
        "metadata": metadata,
    }


def normalize_image_classification_model_config(
    model_directory: Path,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> None:
    config_path = model_directory / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}

    config["galaxy_ai_task"] = "image_classification"
    config["id2label"] = {str(index): label for index, label in id2label.items()}
    config["label2id"] = label2id

    _write_json(config_path, config)


def _training_progress_callback(
    trainer_callback_class,
    total_items: int,
    progress_callback: ProgressCallback | None,
    cancel_event: Event,
):
    class GalaxyTrainingProgressCallback(trainer_callback_class):
        def on_step_begin(self, args, state, control, **kwargs):  # noqa: ANN001
            return self._stop_if_cancelled(control)

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            control = self._stop_if_cancelled(control)
            if progress_callback is None:
                return control
            processed_items = training_processed_items_from_steps(
                global_step=int(getattr(state, "global_step", 0) or 0),
                max_steps=int(getattr(state, "max_steps", 0) or 0),
                total_items=total_items,
            )
            progress_callback(processed_items, total_items)
            return control

        def _stop_if_cancelled(self, control):  # noqa: ANN001
            if cancel_event.is_set():
                control.should_training_stop = True
                control.should_epoch_stop = True
            return control

    return GalaxyTrainingProgressCallback()
