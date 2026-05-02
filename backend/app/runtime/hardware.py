import importlib
from typing import Any


def _backend(name: str, available: bool, label: str) -> dict[str, Any]:
    return {
        "name": name,
        "available": available,
        "label": label,
    }


def detect_hardware() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return {
            "active_backend": "cpu",
            "torch_available": False,
            "backends": {
                "cuda": _backend("cuda", False, "NVIDIA CUDA"),
                "mps": _backend("mps", False, "Apple Metal Performance Shaders"),
                "cpu": _backend("cpu", True, "CPU"),
            },
        }

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
    )

    active_backend = "cpu"
    if cuda_available:
        active_backend = "cuda"
    elif mps_available:
        active_backend = "mps"

    return {
        "active_backend": active_backend,
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", None),
        "backends": {
            "cuda": _backend("cuda", cuda_available, "NVIDIA CUDA"),
            "mps": _backend("mps", mps_available, "Apple Metal Performance Shaders"),
            "cpu": _backend("cpu", True, "CPU"),
        },
    }
