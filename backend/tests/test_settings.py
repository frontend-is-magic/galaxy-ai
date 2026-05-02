from fastapi.testclient import TestClient

from app.main import create_app


def test_settings_return_fallback_directories(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    working_directory = home / "Documents" / "galaxy-ai" / "nebula-sorter"

    with TestClient(create_app()) as client:
        response = client.get("/settings")

    assert response.status_code == 200
    assert response.json() == {
        "model_directory": str(working_directory / "models"),
        "output_directory": str(working_directory / "classification" / "outputs"),
        "dataset_directory": str(working_directory / "classification" / "datasets"),
        "checkpoint_directory": str(working_directory / "training" / "checkpoints"),
        "working_directory": str(working_directory),
        "classification_dataset_directory": str(working_directory / "classification" / "datasets"),
        "classification_output_directory": str(working_directory / "classification" / "outputs"),
        "training_dataset_directory": str(working_directory / "training" / "datasets"),
        "training_output_directory": str(working_directory / "training" / "outputs"),
        "device": "auto",
        "database_path": str(tmp_path / ".galaxy-ai" / "galaxy-ai.sqlite3"),
    }
    assert (working_directory / "models").is_dir()
    assert (working_directory / "classification" / "outputs").is_dir()
    assert (working_directory / "classification" / "datasets").is_dir()
    assert (working_directory / "training" / "outputs").is_dir()
    assert (working_directory / "training" / "datasets").is_dir()
    assert (working_directory / "training" / "checkpoints").is_dir()
    assert not (working_directory / "checkpoints").exists()


def test_settings_return_env_directories_with_expanded_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("GALAXY_AI_WORKING_DIRECTORY", "~/Documents/galaxy-ai/nebula-sorter")
    monkeypatch.setenv("GALAXY_AI_MODEL_DIRECTORY", "~/Documents/galaxy-ai/nebula-sorter/models")
    monkeypatch.setenv(
        "GALAXY_AI_TRAINING_DATASET_DIRECTORY",
        "~/Documents/galaxy-ai/nebula-sorter/training/datasets",
    )
    monkeypatch.setenv("GALAXY_AI_DEVICE", "mps")

    root = home / "Documents" / "galaxy-ai" / "nebula-sorter"

    with TestClient(create_app()) as client:
        response = client.get("/settings")

    assert response.status_code == 200
    assert response.json() == {
        "model_directory": str(root / "models"),
        "output_directory": str(root / "classification" / "outputs"),
        "dataset_directory": str(root / "classification" / "datasets"),
        "checkpoint_directory": str(root / "training" / "checkpoints"),
        "working_directory": str(root),
        "classification_dataset_directory": str(root / "classification" / "datasets"),
        "classification_output_directory": str(root / "classification" / "outputs"),
        "training_dataset_directory": str(root / "training" / "datasets"),
        "training_output_directory": str(root / "training" / "outputs"),
        "device": "mps",
        "database_path": str(tmp_path / ".galaxy-ai" / "galaxy-ai.sqlite3"),
    }
    assert (root / "models").is_dir()
    assert (root / "classification" / "datasets").is_dir()
    assert (root / "classification" / "outputs").is_dir()
    assert (root / "training" / "datasets").is_dir()
    assert (root / "training" / "outputs").is_dir()


def test_settings_reject_directory_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))

    with TestClient(create_app()) as client:
        response = client.put(
            "/settings",
            json={"working_directory": str(tmp_path / "custom-workspace"), "device": "mps"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Directory settings are configured by the root .env file."


def test_settings_reject_invalid_device(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))

    with TestClient(create_app()) as client:
        response = client.put(
            "/settings",
            json={"working_directory": str(tmp_path / "workspace"), "device": "metal"},
        )

    assert response.status_code == 422


def test_model_options_scan_local_models_and_recommend_hf_models(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_directory = tmp_path / "models"
    local_model = model_directory / "vit-local"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text(
        '{"architectures": ["ViTForImageClassification"]}',
        encoding="utf-8",
    )
    (model_directory / "notes.txt").write_text("ignore me", encoding="utf-8")

    with TestClient(create_app()) as client:
        response = client.get(
            "/models/options",
            params={"model_directory": str(model_directory)},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["local_models"] == [
        {
            "label": "vit-local",
            "path": str(local_model),
            "source": "local",
            "compatible": True,
            "compatibility_error": None,
            "requires_download": False,
        }
    ]
    hf_paths = {model["path"] for model in body["recommended_hf_models"]}
    assert {
        "microsoft/resnet-50",
        "google/vit-base-patch16-224-in21k",
        "facebook/convnext-tiny-224",
    }.issubset(hf_paths)
    hf_models = response.json()["recommended_hf_models"]
    assert all(model["compatible"] is True for model in hf_models)
    assert all(model["requires_download"] is True for model in hf_models)


def test_model_options_scan_model_directory_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_directory = tmp_path / "vit-direct"
    model_directory.mkdir()
    (model_directory / "config.json").write_text(
        '{"architectures": ["ViTForImageClassification"]}',
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        response = client.get(
            "/models/options",
            params={"model_directory": str(model_directory)},
        )

    assert response.status_code == 200
    assert response.json()["local_models"] == [
        {
            "label": "vit-direct",
            "path": str(model_directory),
            "source": "local",
            "compatible": True,
            "compatibility_error": None,
            "requires_download": False,
        }
    ]


def test_model_options_expands_tilde_model_directory(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_directory = home / "models"
    local_model = model_directory / "vit-local"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text(
        '{"architectures": ["ViTForImageClassification"]}',
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        response = client.get(
            "/models/options",
            params={"model_directory": "~/models"},
        )

    assert response.status_code == 200
    assert response.json()["local_models"][0]["path"] == str(local_model)


def test_model_options_scan_hugging_face_snapshot_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_directory = tmp_path / "models"
    snapshot_model = model_directory / "models--microsoft--resnet-50" / "snapshots" / "abc123"
    snapshot_model.mkdir(parents=True)
    (snapshot_model / "config.json").write_text(
        '{"architectures": ["ResNetForImageClassification"]}',
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        response = client.get(
            "/models/options",
            params={"model_directory": str(model_directory)},
        )

    assert response.status_code == 200
    assert response.json()["local_models"] == [
        {
            "label": "microsoft/resnet-50",
            "path": str(snapshot_model),
            "source": "local",
            "compatible": True,
            "compatibility_error": None,
            "requires_download": False,
        }
    ]


def test_model_options_deduplicates_local_model_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_directory = tmp_path / "vit-direct"
    model_directory.mkdir()
    (model_directory / "config.json").write_text(
        '{"architectures": ["ViTForImageClassification"]}',
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        response = client.get(
            "/models/options",
            params={"model_directory": str(model_directory)},
        )

    assert response.status_code == 200
    assert len(response.json()["local_models"]) == 1


def test_model_options_scans_trained_model_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_directory = tmp_path / "models"
    trained_model = model_directory / "trained-123"
    trained_model.mkdir(parents=True)
    (trained_model / "config.json").write_text(
        '{"id2label": {"0": "cat"}, "label2id": {"cat": 0}}',
        encoding="utf-8",
    )
    (trained_model / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    with TestClient(create_app()) as client:
        response = client.get(
            "/models/options",
            params={"model_directory": str(model_directory)},
        )

    assert response.status_code == 200
    assert response.json()["local_models"] == [
        {
            "label": "trained-123",
            "path": str(trained_model),
            "source": "local",
            "compatible": True,
            "compatibility_error": None,
            "requires_download": False,
        }
    ]


def test_model_options_marks_non_image_classification_models_incompatible(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_directory = tmp_path / "models"
    language_model = model_directory / "gpt-local"
    language_model.mkdir(parents=True)
    (language_model / "config.json").write_text(
        '{"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}',
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        response = client.get(
            "/models/options",
            params={"model_directory": str(model_directory)},
        )

    assert response.status_code == 200
    assert response.json()["local_models"] == [
        {
            "label": "gpt-local",
            "path": str(language_model),
            "source": "local",
            "compatible": False,
            "compatibility_error": "Model config is not for image classification.",
            "requires_download": False,
        }
    ]


def test_backend_does_not_open_directory_picker(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))

    with TestClient(create_app()) as client:
        response = client.post("/settings/select-directory", json={})

    assert response.status_code == 404
