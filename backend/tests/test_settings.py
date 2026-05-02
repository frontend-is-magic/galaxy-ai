from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def test_settings_return_fallback_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))

    with TestClient(create_app()) as client:
        response = client.get("/settings")

    assert response.status_code == 200
    assert response.json() == {
        "model_directory": str(tmp_path / "models"),
        "output_directory": str(tmp_path / "outputs"),
        "dataset_directory": str(tmp_path / "datasets"),
        "checkpoint_directory": str(tmp_path / "checkpoints"),
        "working_directory": str(tmp_path),
        "database_path": str(tmp_path / ".galaxy-ai" / "galaxy-ai.sqlite3"),
    }


def test_settings_persist_directory_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    payload = {
        "model_directory": str(tmp_path / "custom-models"),
        "output_directory": str(tmp_path / "custom-outputs"),
        "dataset_directory": str(tmp_path / "custom-datasets"),
        "checkpoint_directory": str(tmp_path / "custom-checkpoints"),
        "working_directory": str(tmp_path / "workspace"),
    }

    with TestClient(create_app()) as client:
        update_response = client.put("/settings", json=payload)
        read_response = client.get("/settings")

    assert update_response.status_code == 200
    assert update_response.json() == payload | {
        "database_path": str(tmp_path / ".galaxy-ai" / "galaxy-ai.sqlite3"),
    }
    assert read_response.json() == update_response.json()
    assert Path(tmp_path / ".galaxy-ai" / "galaxy-ai.sqlite3").exists()
