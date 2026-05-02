import json
from pathlib import Path
from typing import Any

IMAGE_CLASSIFICATION_ARCHITECTURE_SUFFIX = "ForImageClassification"


def image_classification_compatibility_error(model_path: Path) -> str | None:
    config_path = model_path.expanduser() / "config.json"
    if not config_path.is_file():
        return "Model config.json is missing."

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "Model config.json is not valid JSON."

    architectures = config.get("architectures")
    if _has_image_classification_architecture(architectures):
        return None

    if _has_label_mapping(config) and _looks_like_vision_model(config):
        return None

    return "Model config is not for image classification."


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
