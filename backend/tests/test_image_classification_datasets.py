import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.settings import get_settings
from app.image_classification.workspace import image_classification_workspace
from app.main import create_app


def wait_for_run_to_finish(client: TestClient, run_id: str) -> None:
    for _ in range(50):
        status = client.get(f"/runs/{run_id}").json()["status"]
        if status in {"completed", "error", "cancelled"}:
            return
        time.sleep(0.01)


def test_fixed_image_classification_workspace_creates_mode_directories(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    get_settings.cache_clear()

    workspace = image_classification_workspace(get_settings())
    workspace.ensure()

    root = home / "Documents" / "galaxy-ai" / "nebula-sorter"
    assert workspace.working_directory == root
    assert workspace.model_directory == root / "models"
    assert workspace.classification_dataset_directory == root / "classification" / "datasets"
    assert workspace.classification_output_directory == root / "classification" / "outputs"
    assert workspace.training_dataset_directory == root / "training" / "datasets"
    assert workspace.training_output_directory == root / "training" / "outputs"
    assert workspace.training_checkpoint_directory == root / "training" / "checkpoints"
    assert workspace.model_directory.is_dir()
    assert workspace.classification_dataset_directory.is_dir()
    assert workspace.training_dataset_train_directory.is_dir()


def test_dataset_preview_lists_classification_images(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    root = home / "Documents" / "galaxy-ai" / "nebula-sorter"
    image = root / "classification" / "datasets" / "nested" / "cat.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"cat")

    with TestClient(create_app()) as client:
        response = client.get(
            "/image-classification/datasets",
            params={"mode": "classification"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "classification"
    assert body["count"] == 1
    assert body["items"][0]["file_name"] == "cat.jpg"
    assert body["items"][0]["relative_path"] == "nested/cat.jpg"
    assert body["items"][0]["size"] == 3
    assert body["labels"] == []


def test_dataset_preview_groups_training_images_by_label(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    root = home / "Documents" / "galaxy-ai" / "nebula-sorter"
    cat = root / "training" / "datasets" / "train" / "cat" / "cat.jpg"
    dog = root / "training" / "datasets" / "train" / "dog" / "dog.png"
    cat.parent.mkdir(parents=True)
    dog.parent.mkdir(parents=True)
    cat.write_bytes(b"cat")
    dog.write_bytes(b"dog")

    with TestClient(create_app()) as client:
        response = client.get(
            "/image-classification/datasets",
            params={"mode": "training"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "training"
    assert body["count"] == 2
    assert [(group["label"], group["count"]) for group in body["labels"]] == [
        ("cat", 1),
        ("dog", 1),
    ]
    assert body["items"] == []


def test_import_classification_dataset_copies_supported_images(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    root = home / "Documents" / "galaxy-ai" / "nebula-sorter"

    with TestClient(create_app()) as client:
        response = client.post(
            "/image-classification/datasets/import",
            files=[
                ("mode", (None, "classification")),
                ("relative_paths[]", (None, "cats/cat.jpg")),
                ("relative_paths[]", (None, "notes/readme.txt")),
                ("files", ("cat.jpg", b"cat", "image/jpeg")),
                ("files", ("readme.txt", b"ignore", "text/plain")),
            ],
        )
        preview = client.get(
            "/image-classification/datasets",
            params={"mode": "classification"},
        )

    assert response.status_code == 200
    assert response.json()["imported_count"] == 1
    assert (root / "classification" / "datasets" / "cats" / "cat.jpg").read_bytes() == b"cat"
    assert preview.json()["count"] == 1


def test_import_training_dataset_requires_label_and_uses_label_directory(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    root = home / "Documents" / "galaxy-ai" / "nebula-sorter"

    with TestClient(create_app()) as client:
        missing_label = client.post(
            "/image-classification/datasets/import",
            files=[
                ("mode", (None, "training")),
                ("files", ("cat.jpg", b"cat", "image/jpeg")),
            ],
        )
        imported = client.post(
            "/image-classification/datasets/import",
            files=[
                ("mode", (None, "training")),
                ("label", (None, "cat")),
                ("relative_paths[]", (None, "nested/cat.jpg")),
                ("files", ("cat.jpg", b"cat", "image/jpeg")),
            ],
        )
        preview = client.get(
            "/image-classification/datasets",
            params={"mode": "training"},
        )

    assert missing_label.status_code == 400
    assert missing_label.json()["detail"] == "Training dataset import requires a label."
    assert imported.status_code == 200
    assert imported.json()["imported_count"] == 1
    assert (
        root / "training" / "datasets" / "train" / "cat" / "nested" / "cat.jpg"
    ).read_bytes() == b"cat"
    assert preview.json()["labels"][0]["label"] == "cat"


def test_clear_dataset_removes_only_selected_mode(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    root = home / "Documents" / "galaxy-ai" / "nebula-sorter"
    classification_image = root / "classification" / "datasets" / "cat.jpg"
    training_image = root / "training" / "datasets" / "train" / "cat" / "cat.jpg"
    model_file = root / "models" / "vit-local" / "config.json"
    classification_image.parent.mkdir(parents=True)
    training_image.parent.mkdir(parents=True)
    model_file.parent.mkdir(parents=True)
    classification_image.write_bytes(b"cat")
    training_image.write_bytes(b"cat")
    model_file.write_text("{}", encoding="utf-8")

    with TestClient(create_app()) as client:
        response = client.delete(
            "/image-classification/datasets",
            params={"mode": "classification"},
        )

    assert response.status_code == 200
    assert response.json()["deleted_count"] == 1
    assert not classification_image.exists()
    assert training_image.exists()
    assert model_file.exists()


def test_dataset_image_preview_returns_classification_image(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    image = (
        home
        / "Documents"
        / "galaxy-ai"
        / "nebula-sorter"
        / "classification"
        / "datasets"
        / "cats"
        / "cat.jpg"
    )
    image.parent.mkdir(parents=True)
    image.write_bytes(b"cat-image")

    with TestClient(create_app()) as client:
        response = client.get(
            "/image-classification/datasets/image",
            params={"mode": "classification", "relative_path": "cats/cat.jpg"},
        )

    assert response.status_code == 200
    assert response.content == b"cat-image"


def test_dataset_image_preview_returns_training_image_for_label(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    image = (
        home
        / "Documents"
        / "galaxy-ai"
        / "nebula-sorter"
        / "training"
        / "datasets"
        / "train"
        / "cat"
        / "cat.jpg"
    )
    image.parent.mkdir(parents=True)
    image.write_bytes(b"cat-train")

    with TestClient(create_app()) as client:
        response = client.get(
            "/image-classification/datasets/image",
            params={
                "mode": "training",
                "label": "cat",
                "relative_path": "cat.jpg",
            },
        )

    assert response.status_code == 200
    assert response.content == b"cat-train"


def test_dataset_image_preview_rejects_invalid_paths_and_missing_training_label(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))

    with TestClient(create_app()) as client:
        missing_label = client.get(
            "/image-classification/datasets/image",
            params={"mode": "training", "relative_path": "cat.jpg"},
        )
        invalid_path = client.get(
            "/image-classification/datasets/image",
            params={"mode": "classification", "relative_path": "../cat.jpg"},
        )
        missing_file = client.get(
            "/image-classification/datasets/image",
            params={"mode": "classification", "relative_path": "missing.jpg"},
        )

    assert missing_label.status_code == 400
    assert missing_label.json()["detail"] == ("Training dataset image preview requires a label.")
    assert invalid_path.status_code == 400
    assert invalid_path.json()["detail"] == "Invalid dataset image path."
    assert missing_file.status_code == 404


def test_default_runs_use_fixed_mode_directories(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GALAXY_AI_HOME", str(tmp_path / ".galaxy-ai"))
    monkeypatch.setenv("GALAXY_AI_REPO_ROOT", str(tmp_path))
    root = home / "Documents" / "galaxy-ai" / "nebula-sorter"
    model = root / "models" / "vit-local"
    image = root / "classification" / "datasets" / "cat.jpg"
    train_cat = root / "training" / "datasets" / "train" / "cat" / "cat.jpg"
    train_dog = root / "training" / "datasets" / "train" / "dog" / "dog.jpg"
    model.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    train_cat.parent.mkdir(parents=True)
    train_dog.parent.mkdir(parents=True)
    (model / "config.json").write_text(
        '{"architectures": ["ViTForImageClassification"]}',
        encoding="utf-8",
    )
    image.write_bytes(b"cat")
    train_cat.write_bytes(b"cat")
    train_dog.write_bytes(b"dog")

    with TestClient(create_app()) as client:
        inference = client.post(
            "/image-classification/inference",
            json={"model_ref": str(model), "device": "cpu"},
        )
        inference_run_id = inference.json()["run_id"]
        inference_run = client.get(f"/runs/{inference_run_id}").json()
        client.post(f"/runs/{inference_run_id}/cancel")
        wait_for_run_to_finish(client, inference_run_id)
        training = client.post(
            "/image-classification/training",
            json={"base_model_ref": str(model), "device": "cpu"},
        )
        training_run = client.get(f"/runs/{training.json()['run_id']}").json()

    assert inference.status_code == 202
    assert training.status_code == 202
    assert inference_run["input_path"] == str(root / "classification" / "datasets")
    assert inference_run["output_path"] == str(root / "classification" / "outputs")
    assert training_run["input_path"] == str(root / "training" / "datasets")
    assert Path(training_run["output_path"]).parent == root / "models"
    assert inference_run["request"]["model_directory"] == str(root / "models")
    assert inference_run["request"]["input_directory"] == str(root / "classification" / "datasets")
    assert inference_run["request"]["output_directory"] == str(root / "classification" / "outputs")
    assert training_run["request"]["model_directory"] == str(root / "models")
    assert training_run["request"]["dataset_directory"] == str(root / "training" / "datasets")
    assert training_run["request"]["output_directory"] == str(root / "training" / "outputs")
