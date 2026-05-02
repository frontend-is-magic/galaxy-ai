from __future__ import annotations

from pathlib import Path


def normalize_local_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base.expanduser() / path
    return path.resolve()


def normalize_optional_local_path(
    value: str | Path | None,
    base: Path | None = None,
) -> Path | None:
    if value is None:
        return None
    return normalize_local_path(value, base)


def stringify_path(value: str | Path, base: Path | None = None) -> str:
    return str(normalize_local_path(value, base))
