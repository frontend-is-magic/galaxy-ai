import re
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from app.core.paths import normalize_local_path


@dataclass(frozen=True)
class ResolvedModel:
    load_ref: str
    source: str
    local_files_only: bool
    cache_dir: Path | None
    local_dir: Path | None = None
    repo_id: str | None = None


def _looks_like_local_path(model_ref: str) -> bool:
    path = Path(model_ref).expanduser()
    return (
        model_ref.startswith(".")
        or model_ref.startswith("/")
        or model_ref.startswith("~")
        or path.suffix != ""
    )


def local_model_directory_for_repo(model_ref: str, model_directory: Path) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "--", model_ref.strip()).strip("-")
    return normalize_local_path(model_directory) / safe_name


def _is_readable_model_directory(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def resolve_model_ref(
    model_ref: str,
    allow_download: bool,
    model_directory: Path,
) -> ResolvedModel:
    path = normalize_local_path(model_ref)
    if path.exists():
        return ResolvedModel(
            load_ref=str(path),
            source="local_path",
            local_files_only=True,
            cache_dir=None,
            local_dir=path,
        )

    if _looks_like_local_path(model_ref):
        raise ValueError(f"Local model path does not exist: {path}")

    local_dir = local_model_directory_for_repo(model_ref, model_directory)
    if _is_readable_model_directory(local_dir):
        return ResolvedModel(
            load_ref=str(local_dir),
            source="local_path",
            local_files_only=True,
            cache_dir=None,
            local_dir=local_dir,
            repo_id=model_ref,
        )

    if not allow_download:
        raise ValueError(
            "Model is not available locally: "
            f"{model_ref}. Set allow_download=true to download it into {local_dir}."
        )

    return ResolvedModel(
        load_ref=model_ref,
        source="hf_repo",
        local_files_only=False,
        cache_dir=None,
        local_dir=local_dir,
        repo_id=model_ref,
    )


def materialize_model_ref(
    resolved: ResolvedModel,
    log_callback,
) -> ResolvedModel:
    if resolved.source != "hf_repo":
        return resolved
    if resolved.repo_id is None or resolved.local_dir is None:
        raise ValueError("Hugging Face model resolution is missing a local target directory.")

    resolved.local_dir.parent.mkdir(parents=True, exist_ok=True)
    log_callback(f"Downloading Hugging Face model {resolved.repo_id} into {resolved.local_dir}.")
    downloaded_path = Path(
        snapshot_download(repo_id=resolved.repo_id, local_dir=str(resolved.local_dir))
    )
    return ResolvedModel(
        load_ref=str(downloaded_path),
        source="local_path",
        local_files_only=True,
        cache_dir=None,
        local_dir=downloaded_path,
        repo_id=resolved.repo_id,
    )
