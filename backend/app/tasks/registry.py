from pydantic import BaseModel


class TaskDefinition(BaseModel):
    id: str
    name: str
    status: str
    local_execution: bool
    description: str


TASKS = [
    TaskDefinition(
        id="text-generation",
        name="Text generation",
        status="planned",
        local_execution=True,
        description="Generate text with a local language model.",
    ),
    TaskDefinition(
        id="image-generation",
        name="Image generation",
        status="planned",
        local_execution=True,
        description="Generate images with a local diffusion model.",
    ),
    TaskDefinition(
        id="vision-understanding",
        name="Vision understanding",
        status="planned",
        local_execution=True,
        description="Analyze local images with a local vision model.",
    ),
    TaskDefinition(
        id="image-classification",
        name="Batch image classification",
        status="available",
        local_execution=True,
        description="Classify local image batches and fine-tune local image classifiers.",
    ),
    TaskDefinition(
        id="speech-transcription",
        name="Speech transcription",
        status="planned",
        local_execution=True,
        description="Transcribe local audio with a local speech model.",
    ),
    TaskDefinition(
        id="training",
        name="Training and fine-tuning",
        status="planned",
        local_execution=True,
        description="Train or fine-tune local models with local datasets.",
    ),
]


def list_tasks() -> list[TaskDefinition]:
    return TASKS
