from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def test_settings_return_fallback_directories(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    working_directory = home / "Documents" / "galaxy-ai"

    with TestClient(create_app()) as client:
        response = client.get("/settings")

    assert response.status_code == 200
    assert response.json() == {
        "model_directory": str(working_directory / "models"),
        "output_directory": str(working_directory / "outputs"),
        "dataset_directory": str(working_directory / "datasets"),
        "checkpoint_directory": str(working_directory / "checkpoints"),
        "working_directory": str(working_directory),
        "device": "auto",
        "database_path": str(tmp_path / ".galaxy-ai" / "galaxy-ai.sqlite3"),
    }
    assert (working_directory / "models").is_dir()
    assert (working_directory / "outputs").is_dir()
    assert (working_directory / "datasets").is_dir()
    assert (working_directory / "checkpoints").is_dir()


def test_settings_persist_directory_updates(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    working_directory = home / "Documents" / "galaxy-ai"
    payload = {
        "working_directory": str(tmp_path / "custom-workspace"),
        "device": "mps",
    }
    expected = {
        "model_directory": str(working_directory / "models"),
        "output_directory": str(working_directory / "outputs"),
        "dataset_directory": str(working_directory / "datasets"),
        "checkpoint_directory": str(working_directory / "checkpoints"),
        "working_directory": str(working_directory),
        "device": "mps",
        "database_path": str(tmp_path / ".galaxy-ai" / "galaxy-ai.sqlite3"),
    }

    with TestClient(create_app()) as client:
        update_response = client.put("/settings", json=payload)
        read_response = client.get("/settings")

    assert update_response.status_code == 200
    assert update_response.json() == expected
    assert read_response.json() == update_response.json()
    assert Path(tmp_path / ".galaxy-ai" / "galaxy-ai.sqlite3").exists()
    assert working_directory.is_dir()
    assert not (tmp_path / "custom-workspace").exists()
    assert (working_directory / "models").is_dir()
    assert (working_directory / "datasets").is_dir()
    assert (working_directory / "outputs").is_dir()
    assert (working_directory / "checkpoints").is_dir()


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
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    (model_directory / "notes.txt").write_text("ignore me", encoding="utf-8")

    with TestClient(create_app()) as client:
        response = client.get(
            "/models/options",
            params={"model_directory": str(model_directory)},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["local_models"] == [
        {"label": "vit-local", "path": str(local_model), "source": "local"}
    ]
    hf_paths = {model["path"] for model in body["recommended_hf_models"]}
    assert {
        "microsoft/resnet-50",
        "google/vit-base-patch16-224-in21k",
        "facebook/convnext-tiny-224",
    }.issubset(hf_paths)


def test_backend_does_not_open_directory_picker(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))

    with TestClient(create_app()) as client:
        response = client.post("/settings/select-directory", json={})

    assert response.status_code == 404
