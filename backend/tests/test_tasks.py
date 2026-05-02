from fastapi.testclient import TestClient

from app.main import app


def test_tasks_returns_initial_task_registry():
    with TestClient(app) as client:
        response = client.get("/tasks")

    assert response.status_code == 200
    assert response.json() == {
        "tasks": [
            {
                "id": "text-generation",
                "name": "Text generation",
                "status": "planned",
                "local_execution": True,
                "description": "Generate text with a local language model.",
            },
            {
                "id": "image-generation",
                "name": "Image generation",
                "status": "planned",
                "local_execution": True,
                "description": "Generate images with a local diffusion model.",
            },
            {
                "id": "vision-understanding",
                "name": "Vision understanding",
                "status": "planned",
                "local_execution": True,
                "description": "Analyze local images with a local vision model.",
            },
            {
                "id": "image-classification",
                "name": "Batch image classification",
                "status": "available",
                "local_execution": True,
                "description": (
                    "Classify local image batches and fine-tune local image classifiers."
                ),
            },
            {
                "id": "speech-transcription",
                "name": "Speech transcription",
                "status": "planned",
                "local_execution": True,
                "description": "Transcribe local audio with a local speech model.",
            },
            {
                "id": "training",
                "name": "Training and fine-tuning",
                "status": "planned",
                "local_execution": True,
                "description": "Train or fine-tune local models with local datasets.",
            },
        ]
    }
