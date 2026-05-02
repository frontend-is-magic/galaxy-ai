import builtins
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from app.image_classification import service


def test_classified_image_copy_sanitizes_label_and_preserves_relative_path(tmp_path):
    input_root = tmp_path / "inputs"
    image_path = input_root / "nested" / "cat.jpg"
    output_root = tmp_path / "outputs"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"cat")

    copied_path = service.copy_classified_image(
        image_path=image_path,
        assigned_label=" Persian/Cat : 1\t",
        output_directory=output_root,
        input_root=input_root,
        used_destinations=set(),
    )

    assert copied_path == output_root / "Persian_Cat_1" / "nested" / "cat.jpg"
    assert copied_path.read_bytes() == b"cat"
    assert image_path.exists()


def test_classified_image_copy_avoids_collisions_for_explicit_paths(tmp_path):
    first = tmp_path / "first" / "same.jpg"
    second = tmp_path / "second" / "same.jpg"
    output_root = tmp_path / "outputs"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    used_destinations: set[Path] = set()

    first_copy = service.copy_classified_image(
        image_path=first,
        assigned_label="cat",
        output_directory=output_root,
        input_root=None,
        used_destinations=used_destinations,
    )
    second_copy = service.copy_classified_image(
        image_path=second,
        assigned_label="cat",
        output_directory=output_root,
        input_root=None,
        used_destinations=used_destinations,
    )

    assert first_copy == output_root / "cat" / "same.jpg"
    assert second_copy == output_root / "cat" / "same__2.jpg"
    assert first_copy.read_bytes() == b"first"
    assert second_copy.read_bytes() == b"second"


def test_classified_image_copy_expands_tilde_output_directory(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    image_path = tmp_path / "cat.jpg"
    image_path.write_bytes(b"cat")

    copied_path = service.copy_classified_image(
        image_path=image_path,
        assigned_label="cat",
        output_directory=Path("~/outputs"),
        input_root=None,
        used_destinations=set(),
    )

    assert copied_path == home / "outputs" / "cat" / "cat.jpg"
    assert copied_path.read_bytes() == b"cat"
    assert not (Path.cwd() / "~").exists()


def test_normalize_image_classification_model_config_writes_reloadable_metadata(tmp_path):
    model_dir = tmp_path / "trained-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"architectures": ["ViTForImageClassification"], "model_type": "vit"}',
        encoding="utf-8",
    )

    service.normalize_image_classification_model_config(
        model_dir,
        id2label={0: "cat", 1: "dog"},
        label2id={"cat": 0, "dog": 1},
    )

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert config["id2label"] == {"0": "cat", "1": "dog"}
    assert config["label2id"] == {"cat": 0, "dog": 1}
    assert config["galaxy_ai_task"] == "image_classification"
    assert config["architectures"] == ["ViTForImageClassification"]


def test_training_model_name_uses_base_model_and_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "training_timestamp_for_filename", lambda: "20260502-121314")

    model_path = service.training_final_model_directory(
        model_directory=tmp_path / "models",
        base_model_ref="google/vit-base-patch16-224-in21k",
    )

    assert model_path == tmp_path / "models" / "google--vit-base-patch16-224-in21k-20260502-121314"


def test_training_processed_items_from_steps_maps_to_total_images():
    assert (
        service.training_processed_items_from_steps(
            global_step=2,
            max_steps=4,
            total_items=10,
        )
        == 5
    )
    assert (
        service.training_processed_items_from_steps(
            global_step=0,
            max_steps=0,
            total_items=10,
        )
        == 0
    )


def test_training_progress_callback_reports_processed_items():
    class FakeTrainerCallback:
        pass

    updates: list[tuple[int, int | None]] = []
    callback = service._training_progress_callback(
        FakeTrainerCallback,
        total_items=10,
        progress_callback=lambda current, total=None: updates.append((current, total)),
        cancel_event=Event(),
    )

    control = object()
    result = callback.on_step_end(
        args=None,
        state=SimpleNamespace(global_step=2, max_steps=4),
        control=control,
    )

    assert result is control
    assert updates == [(5, 10)]


def test_training_progress_callback_requests_stop_when_cancelled():
    class FakeTrainerCallback:
        pass

    cancel_event = Event()
    cancel_event.set()
    callback = service._training_progress_callback(
        FakeTrainerCallback,
        total_items=10,
        progress_callback=None,
        cancel_event=cancel_event,
    )
    control = SimpleNamespace(should_training_stop=False, should_epoch_stop=False)

    result = callback.on_step_begin(args=None, state=SimpleNamespace(), control=control)

    assert result is control
    assert control.should_training_stop is True
    assert control.should_epoch_stop is True


def test_remove_temporary_training_directory_only_removes_training_dirs(tmp_path):
    temporary_dir = tmp_path / ".training-run"
    normal_dir = tmp_path / "model"
    temporary_dir.mkdir()
    normal_dir.mkdir()

    service.remove_temporary_training_directory(temporary_dir)
    service.remove_temporary_training_directory(normal_dir)

    assert not temporary_dir.exists()
    assert normal_dir.exists()


def test_run_imagefolder_training_disables_checkpoints(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "vit-base"
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "outputs" / "run"
    temporary_model = tmp_path / "models" / ".training-run"
    final_model = tmp_path / "models" / "vit-base-20260502-121314"
    (dataset_root / "train" / "cat").mkdir(parents=True)
    (dataset_root / "train" / "dog").mkdir(parents=True)
    (dataset_root / "train" / "cat" / "cat.jpg").write_bytes(b"cat")
    (dataset_root / "train" / "dog" / "dog.jpg").write_bytes(b"dog")
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "vit"}', encoding="utf-8")
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    training_args_seen: dict[str, object] = {}

    class FakeLabelFeature:
        names = ["cat", "dog"]

    class FakeTrainDataset:
        features = {"label": FakeLabelFeature()}

    class FakeDataset(dict):
        def __init__(self):
            super().__init__(train=FakeTrainDataset())

        def with_transform(self, _transform):
            return self

    class FakeImageProcessor:
        def __call__(self, _images, return_tensors):
            assert return_tensors == "pt"
            return {}

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    class FakeModel:
        pass

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            training_args_seen.update(kwargs)

    class FakeTrainer:
        def __init__(self, **_kwargs):
            pass

        def train(self):
            return None

        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text('{"model_type": "vit"}', encoding="utf-8")

    fake_datasets = SimpleNamespace(load_dataset=lambda *_args, **_kwargs: FakeDataset())
    fake_transformers = SimpleNamespace(
        AutoImageProcessor=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: FakeImageProcessor()
        ),
        AutoModelForImageClassification=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: FakeModel()
        ),
        Trainer=FakeTrainer,
        TrainerCallback=type("FakeTrainerCallback", (), {}),
        TrainingArguments=FakeTrainingArguments,
    )
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "datasets":
            return fake_datasets
        if name == "transformers":
            return fake_transformers
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = service.run_imagefolder_training(
        base_model_ref=str(model_dir),
        allow_download=False,
        model_directory=tmp_path / "models",
        dataset_directory=dataset_root,
        output_directory=output_dir,
        temporary_model_directory=temporary_model,
        final_model_directory=final_model,
        epochs=3,
        batch_size=2,
        learning_rate=5e-5,
        seed=None,
        device="cpu",
        cancel_event=Event(),
        log_callback=lambda _message: None,
    )

    metadata = json.loads((output_dir / "training_metadata.json").read_text(encoding="utf-8"))
    assert training_args_seen["save_strategy"] == "no"
    assert training_args_seen["output_dir"] == str(output_dir / "trainer")
    assert "checkpoint_path" not in metadata
    assert result["output_path"] == str(final_model)


def test_run_batch_inference_writes_only_class_directories(
    tmp_path,
    monkeypatch,
):
    input_root = tmp_path / "inputs"
    first = input_root / "batch-a" / "one.jpg"
    second = input_root / "batch-b" / "two.jpg"
    output_root = tmp_path / "outputs"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"architectures": ["ViTForImageClassification"]}',
        encoding="utf-8",
    )

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeInput:
        def __init__(self, images):
            self.images = images

        def to(self, _device):
            return self

    class FakeProcessor:
        def __call__(self, images, return_tensors):
            assert return_tensors == "pt"
            return {"pixel_values": FakeInput(images)}

    class FakeModel:
        config = SimpleNamespace(id2label={0: "cat", 1: "dog"})

        def to(self, _device):
            return self

        def eval(self):
            return None

        def __call__(self, **inputs):
            item_count = len(inputs["pixel_values"].images)
            return SimpleNamespace(logits=FakeMatrix([[0.8, 0.2], [0.1, 0.9]][:item_count]))

    class FakeMatrix:
        def __init__(self, rows):
            self.rows = rows
            self.shape = (len(rows), len(rows[0]))

    class FakeRow(list):
        def tolist(self):
            return list(self)

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return None

    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(functional=SimpleNamespace(softmax=lambda logits, dim: logits)),
        no_grad=lambda: FakeNoGrad(),
    )

    def fake_topk(probabilities, k, dim):
        assert dim == -1
        scores = []
        indices = []
        for row in probabilities.rows:
            ranked = sorted(enumerate(row), key=lambda item: item[1], reverse=True)[:k]
            indices.append(FakeRow(index for index, _score in ranked))
            scores.append(FakeRow(score for _index, score in ranked))
        return scores, indices

    fake_torch.topk = fake_topk
    fake_image_module = SimpleNamespace(open=lambda _path: FakeImage())
    fake_transformers = SimpleNamespace(
        AutoImageProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeProcessor()),
        AutoModelForImageClassification=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: FakeModel()
        ),
    )
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            return fake_torch
        if name == "PIL.Image":
            return fake_image_module
        if name == "transformers":
            return fake_transformers
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        service,
        "resolve_model_ref",
        lambda *_args, **_kwargs: SimpleNamespace(
            source="local_path",
            load_ref=str(model_dir),
            local_files_only=True,
            cache_dir=None,
        ),
    )
    progress_updates: list[int] = []

    result = service.run_batch_inference(
        model_ref=str(model_dir),
        allow_download=False,
        model_directory=tmp_path / "models",
        image_paths=[first, second],
        output_directory=output_root,
        batch_size=2,
        top_k=2,
        device="cpu",
        cancel_event=Event(),
        log_callback=lambda _message: None,
        progress_callback=progress_updates.append,
        input_root=input_root,
    )

    assert (output_root / "cat" / "batch-a" / "one.jpg").read_bytes() == b"first"
    assert (output_root / "dog" / "batch-b" / "two.jpg").read_bytes() == b"second"
    assert not (output_root / "by_label").exists()
    assert not (output_root / "classification_results.jsonl").exists()
    assert not (output_root / "metadata.json").exists()
    assert result["metadata"]["organized_items"] == 2
    assert progress_updates == [1, 2]


def test_run_batch_inference_stops_after_cancel_event_is_set(
    tmp_path,
    monkeypatch,
):
    input_root = tmp_path / "inputs"
    first = input_root / "one.jpg"
    second = input_root / "two.jpg"
    output_root = tmp_path / "outputs"
    input_root.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"architectures": ["ViTForImageClassification"]}',
        encoding="utf-8",
    )

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeInput:
        def __init__(self, images):
            self.images = images

        def to(self, _device):
            return self

    class FakeProcessor:
        def __call__(self, images, return_tensors):
            assert return_tensors == "pt"
            return {"pixel_values": FakeInput(images)}

    class FakeModel:
        config = SimpleNamespace(id2label={0: "cat"})

        def to(self, _device):
            return self

        def eval(self):
            return None

        def __call__(self, **inputs):
            return SimpleNamespace(
                logits=FakeMatrix([[1.0], [1.0]][: len(inputs["pixel_values"].images)])
            )

    class FakeMatrix:
        def __init__(self, rows):
            self.rows = rows
            self.shape = (len(rows), len(rows[0]))

    class FakeRow(list):
        def tolist(self):
            return list(self)

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return None

    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(functional=SimpleNamespace(softmax=lambda logits, dim: logits)),
        no_grad=lambda: FakeNoGrad(),
    )

    def fake_topk(probabilities, k, dim):
        assert dim == -1
        return (
            [FakeRow([row[0]]) for row in probabilities.rows],
            [FakeRow([0]) for _row in probabilities.rows],
        )

    fake_torch.topk = fake_topk
    fake_transformers = SimpleNamespace(
        AutoImageProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeProcessor()),
        AutoModelForImageClassification=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: FakeModel()
        ),
    )
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            return fake_torch
        if name == "PIL.Image":
            return SimpleNamespace(open=lambda _path: FakeImage())
        if name == "transformers":
            return fake_transformers
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        service,
        "resolve_model_ref",
        lambda *_args, **_kwargs: SimpleNamespace(
            source="local_path",
            load_ref=str(model_dir),
            local_files_only=True,
            cache_dir=None,
        ),
    )
    cancel_event = Event()

    def cancel_after_first_progress(processed_items, _total_items=None):
        if processed_items == 1:
            cancel_event.set()

    result = service.run_batch_inference(
        model_ref=str(model_dir),
        allow_download=False,
        model_directory=tmp_path / "models",
        image_paths=[first, second],
        output_directory=output_root,
        batch_size=2,
        top_k=1,
        device="cpu",
        cancel_event=cancel_event,
        log_callback=lambda _message: None,
        progress_callback=cancel_after_first_progress,
        input_root=input_root,
    )

    assert (output_root / "cat" / "one.jpg").exists()
    assert not (output_root / "cat" / "two.jpg").exists()
    assert result["processed_items"] == 1
