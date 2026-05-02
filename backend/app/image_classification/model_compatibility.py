import json
from pathlib import Path
from typing import Any

IMAGE_CLASSIFICATION_ARCHITECTURE_SUFFIX = "ForImageClassification"
VISION_BASE_MODEL_INFERENCE_ERROR = (
    "Model is a vision base model, not a trained image classification model. "
    "Train it first or select a trained image classification model."
)


def image_classification_compatibility_error(model_path: Path) -> str | None:
    config, error = _read_model_config(model_path)
    if error is not None:
        return error

    architectures = config.get("architectures")
    if _has_image_classification_architecture(architectures):
        return None

    if config.get("galaxy_ai_task") == "image_classification" and _has_label_mapping(config):
        return None

    if _has_label_mapping(config) and (
        _looks_like_vision_model(config) or _has_image_processor_config(model_path)
    ):
        return None

    return "Model config is not for image classification."


def image_classification_inference_error(model_path: Path) -> str | None:
    compatibility_error = image_classification_compatibility_error(model_path)
    if compatibility_error is None:
        return None

    if training_base_model_compatibility_error(model_path) is None:
        return VISION_BASE_MODEL_INFERENCE_ERROR

    return compatibility_error


def training_base_model_compatibility_error(model_path: Path) -> str | None:
    config, error = _read_model_config(model_path)
    if error is not None:
        return error

    if image_classification_compatibility_error(model_path) is None:
        return None

    if _looks_like_vision_model(config) and _has_image_processor_config(model_path):
        return None

    return "Model config is not a supported vision model for image classification training."


def _read_model_config(model_path: Path) -> tuple[dict[str, Any], str | None]:
    config_path = model_path.expanduser() / "config.json"
    if not config_path.is_file():
        return {}, "Model config.json is missing."

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, "Model config.json is not valid JSON."
    return config, None


def _has_image_classification_architecture(value: Any) -> bool:
    return isinstance(value, list) and any(
        isinstance(item, str) and item.endswith(IMAGE_CLASSIFICATION_ARCHITECTURE_SUFFIX)
        for item in value
    )


def _has_label_mapping(config: dict[str, Any]) -> bool:
    return isinstance(config.get("id2label"), dict) and isinstance(config.get("label2id"), dict)


def _looks_like_vision_model(config: dict[str, Any]) -> bool:
    model_type = config.get("model_type")
    return isinstance(model_type, str) and model_type.lower() in {
        "beit",
        "convnext",
        "deit",
        "dinov2",
        "efficientnet",
        "mobilevit",
        "resnet",
        "swin",
        "vit",
    }


def _has_image_processor_config(model_path: Path) -> bool:
    return any(
        (model_path / filename).is_file()
        for filename in ("preprocessor_config.json", "image_processor_config.json")
    )
