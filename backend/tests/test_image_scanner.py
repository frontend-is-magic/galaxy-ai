import pytest

from app.image_classification.scanner import scan_images


def test_scan_images_reads_nested_images_for_recursive_directories(tmp_path):
    (tmp_path / "cat.jpg").write_bytes(b"image")
    (tmp_path / "notes.txt").write_text("skip")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "dog.PNG").write_bytes(b"image")

    images = scan_images(input_directory=str(tmp_path), input_paths=None, recursive=True)

    assert images == [tmp_path / "cat.jpg", nested / "dog.PNG"]


def test_scan_images_reads_only_top_level_images_for_non_recursive_directories(
    tmp_path,
):
    (tmp_path / "cat.jpg").write_bytes(b"image")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "dog.PNG").write_bytes(b"image")

    images = scan_images(input_directory=str(tmp_path), input_paths=None, recursive=False)

    assert images == [tmp_path / "cat.jpg"]


def test_scan_images_rejects_remote_urls():
    with pytest.raises(ValueError, match="Remote image URLs are not supported"):
        scan_images(
            input_directory=None, input_paths=["https://example.com/cat.jpg"], recursive=True
        )


def test_scan_images_requires_exactly_one_input_source(tmp_path):
    with pytest.raises(ValueError, match="Provide exactly one"):
        scan_images(
            input_directory=str(tmp_path), input_paths=[str(tmp_path / "cat.jpg")], recursive=True
        )


def test_scan_images_rejects_empty_directories(tmp_path):
    with pytest.raises(ValueError, match="No supported images"):
        scan_images(input_directory=str(tmp_path), input_paths=None, recursive=True)


def test_scan_images_returns_explicit_paths_in_input_order(tmp_path):
    first = tmp_path / "first.webp"
    second = tmp_path / "second.jpeg"
    first.write_bytes(b"image")
    second.write_bytes(b"image")

    images = scan_images(
        input_directory=None,
        input_paths=[str(second), str(first)],
        recursive=True,
    )

    assert images == [second, first]
