[中文说明](README_zh.md)

# Galaxy AI

Galaxy AI is a local-first AI lab for running, training, and exploring AI tasks on your own machine. It is designed to use local compute such as CUDA, Apple GPU through MPS, or CPU, without depending on cloud inference.

## Vision

The app will provide a visual galaxy interface where each planet represents an AI task. The initial planets are planned for:

- Text generation
- Image generation
- Vision understanding
- Speech transcription

Model training and fine-tuning are planned as an extensible task area so users can continue training Hugging Face pretrained models on their own datasets and persist trained models locally.

## Planned Stack

- Frontend: Vite, React, TypeScript, Tailwind CSS
- 3D GUI: React Three Fiber and Three.js
- Backend: Python and FastAPI
- AI runtime: PyTorch with CUDA, MPS, and CPU fallback
- AI ecosystem: Hugging Face `transformers`, `diffusers`, and `safetensors`
- Storage: SQLite metadata plus local filesystem artifacts
- Dependency tools: `npm` for frontend, `uv` for Python

## Local-First Safety

- The app runs locally by default.
- Backend services bind to `127.0.0.1` by default.
- Telemetry is not allowed.
- Online inference APIs are not allowed.
- Hugging Face downloads must be explicit user actions.
- Inference, training, evaluation, checkpoints, and trained model persistence happen locally.

`models/`, `outputs/`, `datasets/`, and `checkpoints/` are fallback directories only. The actual working directories must be configurable from the app Settings UI.

## Local Startup

Run initialization after cloning the repository or when dependencies change:

```bash
./init.sh
```

Start the local backend and frontend after initialization:

```bash
./start.sh
```

Windows PowerShell users can run `./init.ps1` first, then `./start.ps1`.

## Repository Status

This repository currently contains the initial project rules and metadata. The React and FastAPI scaffold will be created later on the `develop` branch.

## Branch Workflow

- `main`: stable repository rules, releases, and reviewed integration points
- `develop`: active development branch
