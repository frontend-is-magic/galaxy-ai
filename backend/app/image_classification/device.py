from fastapi import HTTPException

from app.runtime.hardware import detect_hardware

VALID_DEVICES = {"auto", "cuda", "mps", "cpu"}


def select_device(requested: str) -> str:
    if requested not in VALID_DEVICES:
        raise HTTPException(status_code=400, detail=f"Unsupported device: {requested}")

    hardware = detect_hardware()
    if requested == "auto":
        return str(hardware["active_backend"])

    backend = hardware["backends"].get(requested)
    if backend is None or not backend["available"]:
        raise HTTPException(
            status_code=400, detail=f"Requested device is not available: {requested}"
        )

    return requested
