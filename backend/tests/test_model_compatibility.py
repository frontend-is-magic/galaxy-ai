from app.image_classification.model_compatibility import (
    image_classification_compatibility_error,
    image_classification_inference_error,
    training_base_model_compatibility_error,
)


def test_training_artifact_with_labels_and_image_processor_is_compatible(tmp_path):
    model_dir = tmp_path / "trained-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"id2label": {"0": "cat"}, "label2id": {"cat": 0}}',
        encoding="utf-8",
    )
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    assert image_classification_compatibility_error(model_dir) is None


def test_galaxy_training_artifact_marker_is_compatible_without_architecture(tmp_path):
    model_dir = tmp_path / "trained-vit"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        (
            '{"galaxy_ai_task": "image_classification", '
            '"id2label": {"0": "cat"}, "label2id": {"cat": 0}}'
        ),
        encoding="utf-8",
    )

    assert image_classification_compatibility_error(model_dir) is None


def test_label_mapping_without_image_processor_is_not_compatible(tmp_path):
    model_dir = tmp_path / "text-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"id2label": {"0": "positive"}, "label2id": {"positive": 0}}',
        encoding="utf-8",
    )

    assert (
        image_classification_compatibility_error(model_dir)
        == "Model config is not for image classification."
    )


def test_vit_base_model_with_processor_is_valid_for_training(tmp_path):
    model_dir = tmp_path / "vit-base"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "vit"}', encoding="utf-8")
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    assert training_base_model_compatibility_error(model_dir) is None


def test_vit_base_model_has_clear_inference_error(tmp_path):
    model_dir = tmp_path / "vit-base"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "vit"}', encoding="utf-8")
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    assert (
        image_classification_inference_error(model_dir)
        == "Model is a vision base model, not a trained image classification model. "
        "Train it first or select a trained image classification model."
    )


def test_text_model_is_not_valid_for_image_classification_training(tmp_path):
    model_dir = tmp_path / "text-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "bert", "architectures": ["BertForSequenceClassification"]}',
        encoding="utf-8",
    )

    assert (
        training_base_model_compatibility_error(model_dir)
        == "Model config is not a supported vision model for image classification training."
    )
