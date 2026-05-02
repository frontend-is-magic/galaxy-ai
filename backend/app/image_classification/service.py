from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any, Protocol

from app.image_classification.model_resolver import resolve_model_ref
from app.image_classification.scanner import SUPPORTED_IMAGE_EXTENSIONS


class LogCallback(Protocol):
    def __call__(self, message: str) -> None: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_label_directory_name(label: str) -> str:
    safe_label = re.sub(r"[^\w.-]+", "_", label.strip())
    safe_label = re.sub(r"_+", "_", safe_label).strip("_")
    if safe_label in {"", ".", ".."}:
        return "unknown"
    return safe_label


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
    label_directory = output_directory / "by_label" / safe_label_directory_name(assigned_label)
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
    input_root: Path | None = None,
) -> dict[str, Any]:
    torch = __import__("torch")
    image_module = __import__("PIL.Image", fromlist=["Image"])
    transformers = __import__(
        "transformers",
        fromlist=["AutoImageProcessor", "AutoModelForImageClassification"],
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = output_directory / "classification_results.jsonl"
    metadata_path = output_directory / "metadata.json"
    started_at = _now()
    organized_root = output_directory / "by_label"
    used_destinations: set[Path] = set()
    label_directories: dict[str, dict[str, str]] = {}
    organized_items = 0

    resolved = resolve_model_ref(model_ref, allow_download, model_directory)
    if resolved.source == "hf_repo" and allow_download:
        log_callback(f"Explicit Hugging Face download allowed for model: {model_ref}")

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
    model.to(device)
    model.eval()

    processed = 0
    failed = 0
    with results_path.open("w", encoding="utf-8") as output:
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
                    output.write(json.dumps({"image_path": str(path), "error": str(exc)}) + "\n")

            if not batch_images:
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

            for image_path, scores, indices in zip(
                valid_paths, top_scores, top_indices, strict=True
            ):
                predictions = []
                for score, index in zip(scores.tolist(), indices.tolist(), strict=True):
                    label = model.config.id2label.get(index) or model.config.id2label.get(
                        str(index)
                    )
                    predictions.append({"label": str(label or index), "score": float(score)})
                assigned_label = predictions[0]["label"]
                assigned_score = predictions[0]["score"]
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
                    "path": str(output_directory / "by_label" / safe_label),
                }
                organized_items += 1
                output.write(
                    json.dumps(
                        {
                            "image_path": str(image_path),
                            "predictions": predictions,
                            "assigned_label": assigned_label,
                            "assigned_score": assigned_score,
                            "organized_path": str(organized_path),
                        }
                    )
                    + "\n"
                )
                processed += 1

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
        "results_path": str(results_path),
        "organized_root": str(organized_root),
        "organized_items": organized_items,
        "label_directories": label_directories,
    }
    _write_json(metadata_path, metadata)

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
    checkpoint_directory: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int | None,
    device: str,
    cancel_event: Event,
    log_callback: LogCallback,
) -> dict[str, Any]:
    datasets = __import__("datasets", fromlist=["load_dataset"])
    transformers = __import__(
        "transformers",
        fromlist=[
            "AutoImageProcessor",
            "AutoModelForImageClassification",
            "Trainer",
            "TrainingArguments",
        ],
    )

    labels = validate_imagefolder_dataset(str(dataset_directory))
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    final_model_directory = output_directory / "final_model"
    metadata_path = output_directory / "training_metadata.json"
    started_at = _now()

    resolved = resolve_model_ref(base_model_ref, allow_download, model_directory)
    if resolved.source == "hf_repo" and allow_download:
        log_callback(f"Explicit Hugging Face download allowed for model: {base_model_ref}")

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
        "output_dir": str(checkpoint_directory),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "remove_unused_columns": False,
        "push_to_hub": False,
        "save_strategy": "epoch",
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
    )
    trainer.train()
    trainer.save_model(str(final_model_directory))
    image_processor.save_pretrained(str(final_model_directory))

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
        "checkpoint_path": str(checkpoint_directory),
        "final_model_path": str(final_model_directory),
        "started_at": started_at,
        "completed_at": _now(),
    }
    _write_json(metadata_path, metadata)

    return {
        "output_path": str(final_model_directory),
        "total_items": len(label_names),
        "processed_items": len(label_names),
        "metadata": metadata,
    }
