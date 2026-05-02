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

    assert copied_path == output_root / "by_label" / "Persian_Cat_1" / "nested" / "cat.jpg"
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

    assert first_copy == output_root / "by_label" / "cat" / "same.jpg"
    assert second_copy == output_root / "by_label" / "cat" / "same__2.jpg"
    assert first_copy.read_bytes() == b"first"
    assert second_copy.read_bytes() == b"second"


def test_run_batch_inference_writes_class_directories_and_result_metadata(
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
            load_ref=str(tmp_path / "model"),
            local_files_only=True,
            cache_dir=None,
        ),
    )

    result = service.run_batch_inference(
        model_ref=str(tmp_path / "model"),
        allow_download=False,
        model_directory=tmp_path / "models",
        image_paths=[first, second],
        output_directory=output_root,
        batch_size=2,
        top_k=2,
        device="cpu",
        cancel_event=Event(),
        log_callback=lambda _message: None,
        input_root=input_root,
    )

    assert (output_root / "by_label" / "cat" / "batch-a" / "one.jpg").read_bytes() == b"first"
    assert (output_root / "by_label" / "dog" / "batch-b" / "two.jpg").read_bytes() == b"second"
    result_rows = [
        json.loads(line)
        for line in (output_root / "classification_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result_rows[0]["assigned_label"] == "cat"
    assert result_rows[0]["assigned_score"] == 0.8
    assert result_rows[0]["organized_path"].endswith("by_label/cat/batch-a/one.jpg")
    metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["organized_root"] == str(output_root / "by_label")
    assert metadata["organized_items"] == 2
    assert metadata["label_directories"]["cat"]["path"] == str(output_root / "by_label" / "cat")
    assert result["metadata"]["organized_items"] == 2
