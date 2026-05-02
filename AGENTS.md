# Galaxy AI Repository Rules

## Product Direction

Galaxy AI is a local-first AI lab. It should use local compute, including CUDA, Apple GPU through MPS, or CPU, to run AI tasks without depending on cloud inference.

- The home screen is an interactive galaxy.
- Each planet represents one AI task.
- Initial task planets are text generation, image generation, vision understanding, and speech transcription.
- Model training and fine-tuning should be supported as an extensible task area.
- The UI must show the active hardware backend, model path, working directory, task status, errors, and output location where relevant.

## Safety and Local-First Rules

- The app must run fully locally by default.
- Backend services bind to `127.0.0.1` by default.
- Do not add telemetry.
- Do not call online inference APIs.
- Any network access must be explicit, user-triggered, and visible in UI or documentation.
- Hugging Face may be used for explicit model downloads, but inference, training, and evaluation must run locally after download.
- User datasets, checkpoints, generated outputs, and locally trained models must not be uploaded by default.
- Secrets, local paths, and machine-specific configuration must stay out of git.

## Technology Choices

- Frontend: Vite, React, TypeScript, Tailwind CSS.
- 3D GUI: React Three Fiber and Three.js.
- Backend: Python and FastAPI.
- AI runtime: PyTorch first, with automatic CUDA, MPS, then CPU selection.
- AI ecosystem: Hugging Face first, especially `transformers`, `diffusers`, and `safetensors`.
- Frontend dependencies: `npm`.
- Python dependencies: `uv`.
- Storage: SQLite for metadata and task history, filesystem directories for large files.

## Model, Dataset, and Artifact Directories

- `models/`, `outputs/`, `datasets/`, and `checkpoints/` are fallback directories only.
- The actual model directory, output directory, dataset directory, checkpoint directory, and working directory must be configurable from Settings in the top-right UI.
- Do not hard-code fallback directories as the only valid locations in business logic.
- Directory settings should persist in a local ignored config file or local SQLite database.
- When configuration is missing, the app may fall back to repository-local ignored directories.
- Trained model artifacts must be persisted locally with enough metadata to reload them later.
- Training records must include base model, dataset source, training config, hardware backend, checkpoint path, final model path, and error logs.

## Project Layout

Use this layout as the default direction when the scaffold is created:

```text
frontend/       React GUI
backend/        FastAPI app, AI runtime, task registry, workers, storage, training workflows
docs/           Architecture notes and decision records
init.sh         macOS/Linux initialization entry
init.ps1        Windows PowerShell initialization entry
start.sh        macOS/Linux startup entry
start.ps1       Windows PowerShell startup entry
dev.sh          Optional macOS/Linux development entry
dev.ps1         Optional Windows PowerShell development entry
```

- Startup scripts belong in the repository root, not in `scripts/`.
- Do not commit local runtime directories such as `models/`, `outputs/`, `datasets/`, `checkpoints/`, SQLite runtime files, virtual environments, local config, or generated AI artifacts.

## Initialization and Startup Script Rules

- Root initialization entries must be cross-platform: `init.sh` for macOS/Linux and `init.ps1` for Windows PowerShell.
- Root startup entries must be cross-platform: `start.sh` for macOS/Linux and `start.ps1` for Windows PowerShell.
- Initialization scripts must run preflight/bootstrap checks for Node.js, npm, Python, uv, frontend dependencies, and the backend virtual environment.
- Missing project-level dependencies must be fixed in place by initialization scripts:
  - Install frontend dependencies with `npm ci` when a lockfile exists, otherwise `npm install`.
  - Create or synchronize the backend `.venv` with `uv sync` or an equivalent `uv` workflow.
- Startup scripts must only start already initialized services.
- Startup scripts must fail fast with clear guidance to run the matching initialization script when `frontend/node_modules` or `backend/.venv` is missing.
- If `uv` is missing, scripts should guide installation using the official installer.
- If fixing a missing tool requires modifying global PATH, shell profiles, a system package manager, or other machine-level state, the script must explain the action and ask for confirmation.
- If Node.js or Python is missing, scripts must provide platform-specific installation guidance. They may call a platform package manager only after explicit confirmation.
- Initialization scripts must be idempotent: repeated runs should skip or refresh already satisfied project dependencies without requiring manual cleanup.
- Failures must report the missing item, suggested command, and any log location.

## Frontend Code Rules

- Use TypeScript strict mode.
- Organize frontend code feature-first and keep files close to the feature that owns them.
- Promote code to shared/common only after it is reused across more than two features or forms a clear cross-feature contract.
- Do not create a hand-written API adapter, DTO translation layer, or field mapping layer.
- The frontend may directly use real fields exposed by the backend.
- Backend Pydantic schemas and OpenAPI are the source of truth for response fields.
- Field renames or response shape changes are full-stack contract changes.
- A lightweight HTTP helper is allowed for base URL, request cancellation, JSON parsing, and common error handling. It must not translate business fields.
- TypeScript types should match backend response fields. Prefer future OpenAPI generation over duplicate hand-written models.
- Keep React components small and clear. Move complex state to hooks or feature state modules.

## Backend Code Rules

- Keep FastAPI routes thin. Routes validate input, schedule work, and return status.
- Do not put heavy AI inference or training directly in route handlers.
- Split backend responsibilities into routes, services/domain logic, task registry, runtime adapters, training adapters, storage, and workers.
- Long-running tasks must expose status, errors, logs, output paths, and cancellation semantics.
- Runtime adapters should hide hardware and library-specific details behind clear local interfaces.
- Use Python type hints for new code.
- Prefer small, correct implementations. Add abstraction only when duplication or boundaries justify it.

## Training Rules

- The app may continue training, fine-tune, or adapt Hugging Face pretrained models with user-provided datasets.
- Training, evaluation, checkpointing, and final model export must run locally.
- Training outputs must support local persistence, version records, and later loading.
- Dataset paths and trained model paths must be configurable, not hard-coded.
- Track model source, license, dataset source, hardware requirement, and recommended runtime settings where possible.

## Formatting, Tests, and Quality Gates

- Frontend formatting and checks: Prettier, ESLint, TypeScript check, and Vitest.
- Backend formatting and checks: `ruff format`, `ruff check`, and `pytest`.
- Add `.editorconfig`, lint scripts, format scripts, and check commands when the actual scaffold is created.
- For GUI work, add browser or screenshot verification once the app has runnable UI.
- Do not present unverified results as confirmed working.

## Dependency Rules

- Every new external dependency must have a clear purpose.
- Avoid adding heavy dependencies unless they directly support the AI lab, local runtime, GUI, or quality gates.
- Do not introduce online services for functionality that must remain local.

## Git and Branch Rules

- `main` is for stable repository rules, releases, and reviewed integration points.
- `develop` is the active development branch.
- If a remote branch already has history, do not force push. Fetch and reconcile first.
- Keep commits focused and describe the behavior or rule being changed.
