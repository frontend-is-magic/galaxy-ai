from dataclasses import dataclass
from pathlib import Path

from app.core.settings import AppSettings


@dataclass(frozen=True)
class ImageClassificationWorkspace:
    working_directory: Path
    model_directory: Path
    classification_dataset_directory: Path
    classification_output_directory: Path
    training_dataset_directory: Path
    training_output_directory: Path
    training_checkpoint_directory: Path

    @property
    def training_dataset_train_directory(self) -> Path:
        return self.training_dataset_directory / "train"

    def ensure(self) -> "ImageClassificationWorkspace":
        for directory in (
            self.working_directory,
            self.model_directory,
            self.classification_dataset_directory,
            self.classification_output_directory,
            self.training_dataset_train_directory,
            self.training_output_directory,
            self.training_checkpoint_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def image_classification_workspace(settings: AppSettings) -> ImageClassificationWorkspace:
    return ImageClassificationWorkspace(
        working_directory=settings.image_classification_working_directory,
        model_directory=settings.image_classification_model_directory,
        classification_dataset_directory=(
            settings.image_classification_classification_dataset_directory
        ),
        classification_output_directory=(
            settings.image_classification_classification_output_directory
        ),
        training_dataset_directory=settings.image_classification_training_dataset_directory,
        training_output_directory=settings.image_classification_training_output_directory,
        training_checkpoint_directory=settings.image_classification_training_checkpoint_directory,
    )
