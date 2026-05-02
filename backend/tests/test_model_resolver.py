import pytest

from app.image_classification.model_resolver import resolve_model_ref


def test_resolve_model_ref_uses_existing_local_path(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    resolved = resolve_model_ref(
        model_ref=str(model_dir),
        allow_download=False,
        model_directory=tmp_path / "models",
    )

    assert resolved.source == "local_path"
    assert resolved.load_ref == str(model_dir)
    assert resolved.local_files_only is True
    assert resolved.cache_dir is None


def test_resolve_model_ref_rejects_missing_local_path_without_download(tmp_path):
    with pytest.raises(ValueError, match="Local model path does not exist"):
        resolve_model_ref(
            model_ref=str(tmp_path / "missing-model"),
            allow_download=False,
            model_directory=tmp_path / "models",
        )


def test_resolve_model_ref_keeps_hf_repo_local_only_by_default(tmp_path):
    resolved = resolve_model_ref(
        model_ref="google/vit-base-patch16-224-in21k",
        allow_download=False,
        model_directory=tmp_path / "models",
    )

    assert resolved.source == "hf_repo"
    assert resolved.load_ref == "google/vit-base-patch16-224-in21k"
    assert resolved.local_files_only is True
    assert resolved.cache_dir is None


def test_resolve_model_ref_allows_explicit_download_to_model_directory(tmp_path):
    resolved = resolve_model_ref(
        model_ref="google/vit-base-patch16-224-in21k",
        allow_download=True,
        model_directory=tmp_path / "models",
    )

    assert resolved.source == "hf_repo"
    assert resolved.local_files_only is False
    assert resolved.cache_dir == tmp_path / "models"
