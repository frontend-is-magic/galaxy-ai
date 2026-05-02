from pathlib import Path
from urllib.parse import urlparse

SUPPORTED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}


def _is_remote(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _is_supported(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in SUPPORTED_IMAGE_EXTENSIONS


def scan_images(
    input_directory: str | None,
    input_paths: list[str] | None,
    recursive: bool,
) -> list[Path]:
    has_directory = input_directory is not None
    has_paths = bool(input_paths)
    if has_directory == has_paths:
        raise ValueError("Provide exactly one of input_directory or input_paths.")

    if input_directory is not None:
        if _is_remote(input_directory):
            raise ValueError("Remote image URLs are not supported.")
        directory = Path(input_directory).expanduser()
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Input directory does not exist: {directory}")

        pattern = "**/*" if recursive else "*"
        images = sorted(
            path for path in directory.glob(pattern) if path.is_file() and _is_supported(path)
        )
    else:
        images = []
        for item in input_paths or []:
            if _is_remote(item):
                raise ValueError("Remote image URLs are not supported.")
            path = Path(item).expanduser()
            if not path.exists() or not path.is_file():
                raise ValueError(f"Input image does not exist: {path}")
            if _is_supported(path):
                images.append(path)

    if not images:
        raise ValueError("No supported images found.")
    return images
