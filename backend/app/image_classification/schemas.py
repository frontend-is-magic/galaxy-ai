from pydantic import BaseModel, Field, model_validator


class RunCreateResponse(BaseModel):
    run_id: str
    status: str


class Prediction(BaseModel):
    label: str
    score: float


class BatchInferenceRequest(BaseModel):
    model_ref: str
    model_directory: str | None = None
    allow_download: bool = False
    input_directory: str | None = None
    input_paths: list[str] | None = None
    output_directory: str | None = None
    recursive: bool = True
    batch_size: int = Field(default=8, ge=1, le=256)
    top_k: int = Field(default=5, ge=1, le=100)
    device: str = "auto"

    @model_validator(mode="after")
    def validate_input_source(self) -> "BatchInferenceRequest":
        if (self.input_directory is None) == (not self.input_paths):
            raise ValueError("Provide exactly one of input_directory or input_paths.")
        return self


class TrainingRequest(BaseModel):
    base_model_ref: str
    model_directory: str | None = None
    allow_download: bool = False
    dataset_directory: str
    output_directory: str | None = None
    checkpoint_directory: str | None = None
    epochs: int = Field(default=50, ge=1, le=1000)
    batch_size: int = Field(default=8, ge=1, le=256)
    learning_rate: float = Field(default=5e-5, gt=0)
    seed: int = 42
    use_seed: bool = False
    device: str = "auto"
