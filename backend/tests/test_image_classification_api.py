import asyncio
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.storage.runs_store import RunCreate, create_run, update_run_status


def write_image_classification_config(model_dir: Path) -> None:
    (model_dir / "config.json").write_text(
        '{"architectures": ["ViTForImageClassification"]}',
        encoding="utf-8",
    )


def write_vit_base_config(model_dir: Path) -> None:
    (model_dir / "config.json").write_text('{"model_type": "vit"}', encoding="utf-8")
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")


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


def wait_for_run_to_finish(client: TestClient, run_id: str) -> None:
    for _ in range(50):
        status = client.get(f"/runs/{run_id}").json()["status"]
        if status in {"completed", "error", "cancelled"}:
            return
        time.sleep(0.01)


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
    model_library_dir = tmp_path / "models"
    dataset_class = tmp_path / "dataset" / "train" / "cat"
    second_dataset_class = tmp_path / "dataset" / "train" / "dog"
    model_dir.mkdir()
    write_image_classification_config(model_dir)
    dataset_class.mkdir(parents=True)
    second_dataset_class.mkdir(parents=True)
    (dataset_class / "cat.jpg").write_bytes(b"fake image")
    (dataset_class / "cat-2.jpg").write_bytes(b"fake image")
    (second_dataset_class / "dog.jpg").write_bytes(b"fake image")

    with TestClient(create_app()) as client:
        create_response = client.post(
            "/image-classification/training",
            json={
                "base_model_ref": str(model_dir),
                "model_directory": str(model_library_dir),
                "dataset_directory": str(tmp_path / "dataset"),
                "device": "cpu",
            },
        )
        run_id = create_response.json()["run_id"]
        status_response = client.get(f"/runs/{run_id}")

    assert create_response.status_code == 202
    assert status_response.status_code == 200
    assert status_response.json()["run_type"] == "image_classification_training"
    assert status_response.json()["total_items"] == 3
    assert Path(status_response.json()["output_path"]).name.startswith("model-")
    assert ".training-" not in status_response.json()["output_path"]


def test_image_classification_training_expands_configured_tilde_directories(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir = home / "models"
    local_model = model_dir / "vit-local"
    dataset_class = home / "datasets" / "train" / "cat"
    second_dataset_class = home / "datasets" / "train" / "dog"
    local_model.mkdir(parents=True)
    write_image_classification_config(local_model)
    dataset_class.mkdir(parents=True)
    second_dataset_class.mkdir(parents=True)
    (dataset_class / "cat.jpg").write_bytes(b"fake image")
    (second_dataset_class / "dog.jpg").write_bytes(b"fake image")

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/training",
            json={
                "base_model_ref": "~/models/vit-local",
                "model_directory": "~/models",
                "dataset_directory": "~/datasets",
                "output_directory": "~/outputs",
                "device": "cpu",
            },
        )
        run = client.get(f"/runs/{response.json()['run_id']}").json()

    assert response.status_code == 202
    assert run["input_path"] == str(home / "datasets")
    assert Path(run["output_path"]).parent == model_dir
    assert "~" not in run["output_path"]
    assert run["request"]["model_directory"] == str(model_dir)
    assert run["request"]["dataset_directory"] == str(home / "datasets")
    assert run["request"]["output_directory"] == str(home / "outputs")


def test_image_classification_inference_expands_request_tilde_directories(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir = home / "models" / "vit-local"
    image_dir = home / "datasets"
    output_dir = home / "outputs"
    model_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    write_image_classification_config(model_dir)
    (image_dir / "cat.jpg").write_bytes(b"fake image")

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/inference",
            json={
                "model_ref": "~/models/vit-local",
                "model_directory": "~/models",
                "input_directory": "~/datasets",
                "output_directory": "~/outputs",
                "device": "cpu",
            },
        )
        run = client.get(f"/runs/{response.json()['run_id']}").json()

    assert response.status_code == 202
    assert run["input_path"] == str(image_dir)
    assert run["output_path"] == str(output_dir)
    assert run["model_ref"] == str(model_dir)
    assert run["request"]["model_directory"] == str(home / "models")
    assert run["request"]["input_directory"] == str(image_dir)
    assert run["request"]["output_directory"] == str(output_dir)
    assert "~" not in str(run["request"])


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
        seen_model_directories.append(kwargs["temporary_model_directory"].parent)
        return {
            "output_path": str(kwargs["temporary_model_directory"]),
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


def test_image_classification_training_accepts_local_vit_base_model(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir = tmp_path / "vit-base"
    dataset_class = tmp_path / "dataset" / "train" / "cat"
    second_dataset_class = tmp_path / "dataset" / "train" / "dog"
    model_dir.mkdir()
    write_vit_base_config(model_dir)
    dataset_class.mkdir(parents=True)
    second_dataset_class.mkdir(parents=True)
    (dataset_class / "cat.jpg").write_bytes(b"fake image")
    (second_dataset_class / "dog.jpg").write_bytes(b"fake image")

    def fake_training(**kwargs):
        return {
            "output_path": str(kwargs["temporary_model_directory"]),
            "total_items": 2,
            "processed_items": 2,
            "metadata": {},
        }

    monkeypatch.setattr("app.image_classification.service.run_imagefolder_training", fake_training)

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/training",
            json={
                "base_model_ref": str(model_dir),
                "dataset_directory": str(tmp_path / "dataset"),
                "device": "cpu",
            },
        )

    assert response.status_code == 202


def test_image_classification_inference_allows_explicit_hf_download(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    image_dir = tmp_path / "images"
    model_dir = tmp_path / "models"
    image_dir.mkdir()
    (image_dir / "cat.jpg").write_bytes(b"fake image")

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/inference",
            json={
                "model_ref": "microsoft/resnet-50",
                "model_directory": str(model_dir),
                "input_directory": str(image_dir),
                "allow_download": True,
                "device": "cpu",
            },
        )
        assert response.status_code == 202
        run = client.get(f"/runs/{response.json()['run_id']}").json()

    assert run["request"]["model_directory"] == str(model_dir)


def test_image_classification_training_allows_explicit_hf_download(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir = tmp_path / "models"
    dataset_class = tmp_path / "dataset" / "train" / "cat"
    second_dataset_class = tmp_path / "dataset" / "train" / "dog"
    dataset_class.mkdir(parents=True)
    second_dataset_class.mkdir(parents=True)
    (dataset_class / "cat.jpg").write_bytes(b"fake image")
    (second_dataset_class / "dog.jpg").write_bytes(b"fake image")

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/training",
            json={
                "base_model_ref": "microsoft/resnet-50",
                "model_directory": str(model_dir),
                "dataset_directory": str(tmp_path / "dataset"),
                "allow_download": True,
                "device": "cpu",
            },
        )
        assert response.status_code == 202
        run = client.get(f"/runs/{response.json()['run_id']}").json()

    assert run["request"]["model_directory"] == str(model_dir)


def test_image_classification_inference_accepts_trained_model_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    trained_model = tmp_path / "models" / "trained-run"
    image_dir = tmp_path / "images"
    trained_model.mkdir(parents=True)
    image_dir.mkdir()
    (trained_model / "config.json").write_text(
        '{"id2label": {"0": "cat"}, "label2id": {"cat": 0}}',
        encoding="utf-8",
    )
    (trained_model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
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
        response = client.post(
            "/image-classification/inference",
            json={
                "model_ref": str(trained_model),
                "input_directory": str(image_dir),
                "device": "cpu",
            },
        )

    assert response.status_code == 202


def test_image_classification_inference_rejects_vit_base_model_with_clear_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    model_dir = tmp_path / "vit-base"
    image_dir = tmp_path / "images"
    model_dir.mkdir()
    image_dir.mkdir()
    write_vit_base_config(model_dir)
    (image_dir / "cat.jpg").write_bytes(b"fake image")

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/inference",
            json={
                "model_ref": str(model_dir),
                "input_directory": str(image_dir),
                "device": "cpu",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Model is a vision base model, not a trained image classification model. "
        "Train it first or select a trained image classification model."
    )


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
    assert (
        response.json()["detail"]
        == "Model config is not a supported vision model for image classification training."
    )


def test_image_classification_inference_rejects_uncached_hf_model_without_download(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "cat.jpg").write_bytes(b"fake image")

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/inference",
            json={
                "model_ref": "microsoft/resnet-50",
                "model_directory": str(tmp_path / "models"),
                "input_directory": str(image_dir),
                "allow_download": False,
                "device": "cpu",
            },
        )

    assert response.status_code == 400
    assert "Model is not available locally" in response.json()["detail"]


def test_image_classification_training_rejects_uncached_hf_model_without_download(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    dataset_class = tmp_path / "dataset" / "train" / "cat"
    second_dataset_class = tmp_path / "dataset" / "train" / "dog"
    dataset_class.mkdir(parents=True)
    second_dataset_class.mkdir(parents=True)
    (dataset_class / "cat.jpg").write_bytes(b"fake image")
    (second_dataset_class / "dog.jpg").write_bytes(b"fake image")

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/training",
            json={
                "base_model_ref": "microsoft/resnet-50",
                "model_directory": str(tmp_path / "models"),
                "dataset_directory": str(tmp_path / "dataset"),
                "allow_download": False,
                "device": "cpu",
            },
        )

    assert response.status_code == 400
    assert "Model is not available locally" in response.json()["detail"]


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


def test_open_run_output_opens_completed_run_output_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    output_dir = tmp_path / "outputs" / "run-1"
    output_dir.mkdir(parents=True)
    opened_paths: list[Path] = []

    def fake_open_output_directory(path: Path) -> None:
        opened_paths.append(path)

    monkeypatch.setattr("app.main.open_output_directory", fake_open_output_directory)

    with TestClient(create_app()) as client:
        run = asyncio.run(
            create_run(
                database_path_for(tmp_path),
                RunCreate(
                    run_type="image_classification_inference",
                    status="completed",
                    request={},
                    hardware_backend="cpu",
                    output_path=str(output_dir),
                ),
            )
        )
        response = client.post(f"/runs/{run.run_id}/open-output")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": run.run_id,
        "output_path": str(output_dir),
        "opened": True,
    }
    assert opened_paths == [output_dir]


def test_open_run_output_rejects_missing_unfinished_and_missing_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "app.main.open_output_directory",
        lambda _path: (_ for _ in ()).throw(AssertionError("should not open")),
    )

    with TestClient(create_app()) as client:
        queued_run = asyncio.run(
            create_run(
                database_path_for(tmp_path),
                RunCreate(
                    run_type="image_classification_inference",
                    status="queued",
                    request={},
                    hardware_backend="cpu",
                    output_path=str(tmp_path / "queued"),
                ),
            )
        )
        empty_output_run = asyncio.run(
            create_run(
                database_path_for(tmp_path),
                RunCreate(
                    run_type="image_classification_inference",
                    status="completed",
                    request={},
                    hardware_backend="cpu",
                    output_path=None,
                ),
            )
        )
        missing_dir_run = asyncio.run(
            create_run(
                database_path_for(tmp_path),
                RunCreate(
                    run_type="image_classification_inference",
                    status="completed",
                    request={},
                    hardware_backend="cpu",
                    output_path=str(tmp_path / "missing"),
                ),
            )
        )

        missing_response = client.post("/runs/missing-run/open-output")
        queued_response = client.post(f"/runs/{queued_run.run_id}/open-output")
        empty_output_response = client.post(f"/runs/{empty_output_run.run_id}/open-output")
        missing_dir_response = client.post(f"/runs/{missing_dir_run.run_id}/open-output")

    assert missing_response.status_code == 404
    assert queued_response.status_code == 400
    assert empty_output_response.status_code == 400
    assert missing_dir_response.status_code == 404


def test_open_run_output_reports_system_open_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    output_dir = tmp_path / "outputs" / "run-1"
    output_dir.mkdir(parents=True)

    def fake_open_output_directory(_path: Path) -> None:
        raise RuntimeError("open failed")

    monkeypatch.setattr("app.main.open_output_directory", fake_open_output_directory)

    with TestClient(create_app()) as client:
        run = asyncio.run(
            create_run(
                database_path_for(tmp_path),
                RunCreate(
                    run_type="image_classification_inference",
                    status="completed",
                    request={},
                    hardware_backend="cpu",
                    output_path=str(output_dir),
                ),
            )
        )
        response = client.post(f"/runs/{run.run_id}/open-output")

    assert response.status_code == 500
    assert response.json()["detail"] == "open failed"
