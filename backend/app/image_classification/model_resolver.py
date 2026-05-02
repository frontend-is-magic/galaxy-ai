from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolvedModel:
    load_ref: str
    source: str
    local_files_only: bool
    cache_dir: Path | None


def _looks_like_local_path(model_ref: str) -> bool:
    path = Path(model_ref).expanduser()
    return (
        model_ref.startswith(".")
        or model_ref.startswith("/")
        or model_ref.startswith("~")
        or path.suffix != ""
    )


def resolve_model_ref(
    model_ref: str,
    allow_download: bool,
    model_directory: Path,
) -> ResolvedModel:
    path = Path(model_ref).expanduser()
    if path.exists():
        return ResolvedModel(
            load_ref=str(path),
            source="local_path",
            local_files_only=True,
            cache_dir=None,
        )

    if _looks_like_local_path(model_ref):
        raise ValueError(f"Local model path does not exist: {path}")

    return ResolvedModel(
        load_ref=model_ref,
        source="hf_repo",
        local_files_only=not allow_download,
        cache_dir=model_directory if allow_download else None,
    )
