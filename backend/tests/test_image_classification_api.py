import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.storage.runs_store import RunCreate, create_run, update_run_status


def write_image_classification_config(model_dir: Path) -> None:
    (model_dir / "config.json").write_text(
        '{"architectures": ["ViTForImageClassification"]}',
        encoding="utf-8",
    )


def database_path_for(tmp_path: Path) -> Path:
    return tmp_path / ".galaxy-ai" / "galaxy-ai.sqlite3"


def create_direct_run(
    tmp_path: Path,
    run_type: str,
    status: str = "queued",
):
    return asyncio.run(
        create_run(
            database_path_for(tmp_path),
            RunCreate(
                run_type=run_type,
                status=status,
                request={},
                hardware_backend="cpu",
                model_ref="local-model",
                input_path="input",
                output_path="output",
            ),
        )
    )


def create_valid_inference_inputs(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    image_dir = tmp_path / "images"
    model_dir.mkdir(exist_ok=True)
    write_image_classification_config(model_dir)
    image_dir.mkdir(exist_ok=True)
    (image_dir / "cat.jpg").write_bytes(b"fake image")
    return model_dir, image_dir


def create_valid_training_inputs(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    dataset_class = tmp_path / "dataset" / "train" / "cat"
    second_dataset_class = tmp_path / "dataset" / "train" / "dog"
    model_dir.mkdir(exist_ok=True)
    write_image_classification_config(model_dir)
    dataset_class.mkdir(parents=True, exist_ok=True)
    second_dataset_class.mkdir(parents=True, exist_ok=True)
    (dataset_class / "cat.jpg").write_bytes(b"fake image")
    (second_dataset_class / "dog.jpg").write_bytes(b"fake image")
    return model_dir, tmp_path / "dataset"


def test_image_classification_inference_creates_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir = tmp_path / "model"
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "outputs"
    model_dir.mkdir()
    write_image_classification_config(model_dir)
    image_dir.mkdir()
    (image_dir / "cat.jpg").write_bytes(b"fake image")

    def fake_inference(**kwargs):
        Path(kwargs["output_directory"]).mkdir(parents=True, exist_ok=True)
        return {
            "output_path": str(kwargs["output_directory"]),
            "total_items": len(kwargs["image_paths"]),
            "processed_items": len(kwargs["image_paths"]),
            "metadata": {"successful_items": len(kwargs["image_paths"]), "failed_items": 0},
        }

    monkeypatch.setattr("app.image_classification.service.run_batch_inference", fake_inference)

    with TestClient(create_app()) as client:
        create_response = client.post(
            "/image-classification/inference",
            json={
                "model_ref": str(model_dir),
                "input_directory": str(image_dir),
                "output_directory": str(output_dir),
                "device": "cpu",
            },
        )
        run_id = create_response.json()["run_id"]
        status_response = client.get(f"/runs/{run_id}")

    assert create_response.status_code == 202
    assert status_response.status_code == 200
    assert status_response.json()["run_type"] == "image_classification_inference"
    assert status_response.json()["status"] in {"queued", "running", "completed"}


def test_image_classification_training_creates_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir = tmp_path / "model"
    dataset_class = tmp_path / "dataset" / "train" / "cat"
    second_dataset_class = tmp_path / "dataset" / "train" / "dog"
    model_dir.mkdir()
    write_image_classification_config(model_dir)
    dataset_class.mkdir(parents=True)
    second_dataset_class.mkdir(parents=True)
    (dataset_class / "cat.jpg").write_bytes(b"fake image")
    (second_dataset_class / "dog.jpg").write_bytes(b"fake image")

    def fake_training(**kwargs):
        final_model = Path(kwargs["output_directory"]) / "final_model"
        final_model.mkdir(parents=True, exist_ok=True)
        return {
            "output_path": str(final_model),
            "total_items": 1,
            "processed_items": 1,
            "metadata": {"labels": ["cat"]},
        }

    monkeypatch.setattr("app.image_classification.service.run_imagefolder_training", fake_training)

    with TestClient(create_app()) as client:
        create_response = client.post(
            "/image-classification/training",
            json={
                "base_model_ref": str(model_dir),
                "dataset_directory": str(tmp_path / "dataset"),
                "device": "cpu",
            },
        )
        run_id = create_response.json()["run_id"]
        status_response = client.get(f"/runs/{run_id}")

    assert create_response.status_code == 202
    assert status_response.status_code == 200
    assert status_response.json()["run_type"] == "image_classification_training"


def test_image_classification_training_uses_request_model_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir = tmp_path / "model"
    task_model_dir = tmp_path / "task" / "models"
    dataset_class = tmp_path / "dataset" / "train" / "cat"
    second_dataset_class = tmp_path / "dataset" / "train" / "dog"
    model_dir.mkdir()
    write_image_classification_config(model_dir)
    task_model_dir.mkdir(parents=True)
    dataset_class.mkdir(parents=True)
    second_dataset_class.mkdir(parents=True)
    (dataset_class / "cat.jpg").write_bytes(b"fake image")
    (second_dataset_class / "dog.jpg").write_bytes(b"fake image")
    seen_model_directories: list[Path] = []

    def fake_resolver(model_ref, allow_download, model_directory):
        seen_model_directories.append(model_directory)

        class Resolved:
            source = "local_path"
            load_ref = str(model_dir)

        return Resolved()

    def fake_training(**kwargs):
        seen_model_directories.append(kwargs["model_directory"])
        return {
            "output_path": str(kwargs["output_directory"]),
            "total_items": 1,
            "processed_items": 1,
            "metadata": {"labels": ["cat"]},
        }

    monkeypatch.setattr("app.main.resolve_model_ref", fake_resolver)
    monkeypatch.setattr("app.image_classification.service.run_imagefolder_training", fake_training)

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/training",
            json={
                "base_model_ref": str(model_dir),
                "model_directory": str(task_model_dir),
                "dataset_directory": str(tmp_path / "dataset"),
                "device": "cpu",
            },
        )

    assert response.status_code == 202
    assert task_model_dir in seen_model_directories


def test_image_classification_rejects_unavailable_requested_device(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir = tmp_path / "model"
    image_dir = tmp_path / "images"
    model_dir.mkdir()
    image_dir.mkdir()
    (image_dir / "cat.jpg").write_bytes(b"fake image")

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/inference",
            json={
                "model_ref": str(model_dir),
                "input_directory": str(image_dir),
                "device": "cuda",
            },
        )

    assert response.status_code == 400
    assert "Requested device is not available" in response.json()["detail"]


def test_image_classification_rejects_incompatible_local_model(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir = tmp_path / "model"
    dataset_class = tmp_path / "dataset" / "train" / "cat"
    second_dataset_class = tmp_path / "dataset" / "train" / "dog"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}',
        encoding="utf-8",
    )
    dataset_class.mkdir(parents=True)
    second_dataset_class.mkdir(parents=True)
    (dataset_class / "cat.jpg").write_bytes(b"fake image")
    (second_dataset_class / "dog.jpg").write_bytes(b"fake image")

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/training",
            json={
                "base_model_ref": str(model_dir),
                "dataset_directory": str(tmp_path / "dataset"),
                "device": "cpu",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Model config is not for image classification."


def test_active_run_endpoint_returns_null_when_no_image_classification_run(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))

    with TestClient(create_app()) as client:
        response = client.get("/runs/active")

    assert response.status_code == 200
    assert response.json() == {"run": None}


def test_image_classification_inference_rejects_when_training_run_is_active(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir, image_dir = create_valid_inference_inputs(tmp_path)

    with TestClient(create_app()) as client:
        active_run = create_direct_run(tmp_path, "image_classification_training")
        response = client.post(
            "/image-classification/inference",
            json={
                "model_ref": str(model_dir),
                "input_directory": str(image_dir),
                "device": "cpu",
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Another image classification task is already active: "
        f"{active_run.run_id} (image_classification_training). "
        "Cancel it or wait for it to finish."
    )


def test_image_classification_training_rejects_when_inference_run_is_active(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir, dataset_dir = create_valid_training_inputs(tmp_path)

    with TestClient(create_app()) as client:
        active_run = create_direct_run(tmp_path, "image_classification_inference")
        response = client.post(
            "/image-classification/training",
            json={
                "base_model_ref": str(model_dir),
                "dataset_directory": str(dataset_dir),
                "device": "cpu",
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Another image classification task is already active: "
        f"{active_run.run_id} (image_classification_inference). "
        "Cancel it or wait for it to finish."
    )


def test_image_classification_lock_releases_after_active_run_is_terminal(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir, image_dir = create_valid_inference_inputs(tmp_path)

    def fake_inference(**kwargs):
        return {
            "output_path": str(kwargs["output_directory"]),
            "total_items": 1,
            "processed_items": 1,
            "metadata": {},
        }

    monkeypatch.setattr("app.image_classification.service.run_batch_inference", fake_inference)

    with TestClient(create_app()) as client:
        active_run = create_direct_run(tmp_path, "image_classification_training")
        asyncio.run(update_run_status(database_path_for(tmp_path), active_run.run_id, "cancelled"))
        response = client.post(
            "/image-classification/inference",
            json={
                "model_ref": str(model_dir),
                "input_directory": str(image_dir),
                "device": "cpu",
            },
        )

    assert response.status_code == 202


def test_cancel_untracked_unfinished_training_run_marks_cancelled_and_releases_lock(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir, image_dir = create_valid_inference_inputs(tmp_path)

    def fake_inference(**kwargs):
        return {
            "output_path": str(kwargs["output_directory"]),
            "total_items": 1,
            "processed_items": 1,
            "metadata": {},
        }

    monkeypatch.setattr("app.image_classification.service.run_batch_inference", fake_inference)

    with TestClient(create_app()) as client:
        active_run = create_direct_run(tmp_path, "image_classification_training")
        cancel_response = client.post(f"/runs/{active_run.run_id}/cancel")
        logs_response = client.get(f"/runs/{active_run.run_id}/logs")
        create_response = client.post(
            "/image-classification/inference",
            json={
                "model_ref": str(model_dir),
                "input_directory": str(image_dir),
                "device": "cpu",
            },
        )

    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"
    assert "Cancellation requested." in logs_response.json()["logs"]
    assert create_response.status_code == 202


def test_cancel_run_marks_run_cancelling(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir = tmp_path / "model"
    image_dir = tmp_path / "images"
    model_dir.mkdir()
    write_image_classification_config(model_dir)
    image_dir.mkdir()
    (image_dir / "cat.jpg").write_bytes(b"fake image")

    def fake_inference(**kwargs):
        return {
            "output_path": str(kwargs["output_directory"]),
            "total_items": 1,
            "processed_items": 1,
            "metadata": {},
        }

    monkeypatch.setattr("app.image_classification.service.run_batch_inference", fake_inference)

    with TestClient(create_app()) as client:
        create_response = client.post(
            "/image-classification/inference",
            json={
                "model_ref": str(model_dir),
                "input_directory": str(image_dir),
                "device": "cpu",
            },
        )
        response = client.post(f"/runs/{create_response.json()['run_id']}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] in {"cancelling", "cancelled", "completed"}
