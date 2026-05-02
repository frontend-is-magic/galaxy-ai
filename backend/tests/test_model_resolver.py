import pytest

from app.image_classification.model_resolver import (
    local_model_directory_for_repo,
    materialize_model_ref,
    resolve_model_ref,
)


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


def test_hf_repo_maps_to_models_directory(tmp_path):
    assert (
        local_model_directory_for_repo(
            "google/vit-base-patch16-224-in21k",
            tmp_path / "models",
        )
        == tmp_path / "models" / "google--vit-base-patch16-224-in21k"
    )


def test_resolve_model_ref_uses_materialized_hf_repo_from_models_directory(tmp_path):
    model_dir = tmp_path / "models" / "google--vit-base-patch16-224-in21k"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        '{"architectures": ["ViTForImageClassification"]}',
        encoding="utf-8",
    )

    resolved = resolve_model_ref(
        model_ref="google/vit-base-patch16-224-in21k",
        allow_download=False,
        model_directory=tmp_path / "models",
    )

    assert resolved.source == "local_path"
    assert resolved.load_ref == str(model_dir)
    assert resolved.local_files_only is True
    assert resolved.cache_dir is None


def test_resolve_model_ref_rejects_hf_repo_without_local_model_or_download(tmp_path):
    with pytest.raises(ValueError, match="Model is not available locally"):
        resolve_model_ref(
            model_ref="google/vit-base-patch16-224-in21k",
            allow_download=False,
            model_directory=tmp_path / "models",
        )


def test_resolve_model_ref_plans_explicit_download_to_materialized_directory(tmp_path):
    resolved = resolve_model_ref(
        model_ref="google/vit-base-patch16-224-in21k",
        allow_download=True,
        model_directory=tmp_path / "models",
    )

    assert resolved.source == "hf_repo"
    assert resolved.local_files_only is False
    assert resolved.cache_dir is None
    assert resolved.load_ref == "google/vit-base-patch16-224-in21k"
    assert resolved.local_dir == tmp_path / "models" / "google--vit-base-patch16-224-in21k"


def test_materialize_model_ref_downloads_hf_repo_into_models_directory(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(repo_id, local_dir):
        calls.append((repo_id, local_dir))
        materialized = tmp_path / "models" / "google--vit-base-patch16-224-in21k"
        materialized.mkdir(parents=True)
        (materialized / "config.json").write_text(
            '{"architectures": ["ViTForImageClassification"]}',
            encoding="utf-8",
        )
        return str(materialized)

    monkeypatch.setattr(
        "app.image_classification.model_resolver.snapshot_download",
        fake_snapshot_download,
    )
    resolved = resolve_model_ref(
        model_ref="google/vit-base-patch16-224-in21k",
        allow_download=True,
        model_directory=tmp_path / "models",
    )

    materialized = materialize_model_ref(resolved, log_callback=lambda _message: None)

    assert calls == [
        (
            "google/vit-base-patch16-224-in21k",
            str(tmp_path / "models" / "google--vit-base-patch16-224-in21k"),
        )
    ]
    assert materialized.source == "local_path"
    assert materialized.load_ref == str(tmp_path / "models" / "google--vit-base-patch16-224-in21k")
    assert materialized.local_files_only is True
