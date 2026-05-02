export type DevicePreference = "auto" | "cpu" | "cuda" | "mps";

export type TrainingRequest = {
  base_model_ref: string;
  model_directory: string | null;
  allow_download: boolean;
  dataset_directory: string;
  output_directory: string | null;
  checkpoint_directory: string | null;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  seed: number;
  use_seed: boolean;
  device: DevicePreference;
};

export type BatchInferenceRequest = {
  model_ref: string;
  model_directory: string | null;
  allow_download: boolean;
  input_directory: string;
  output_directory: string | null;
  recursive: boolean;
  batch_size: number;
  top_k: number;
  device: DevicePreference;
};

export type DatasetMode = "classification" | "training";

export type DatasetImageItem = {
  file_name: string;
  relative_path: string;
  size: number;
};

export type DatasetLabelGroup = {
  label: string;
  count: number;
  items: DatasetImageItem[];
};

export type DatasetPreviewResponse = {
  mode: DatasetMode;
  count: number;
  items: DatasetImageItem[];
  labels: DatasetLabelGroup[];
};

export type DatasetImportFile = {
  file: File;
  relativePath: string;
};

export type DatasetImportResponse = {
  mode: DatasetMode;
  imported_count: number;
};

export type DatasetClearResponse = {
  mode: DatasetMode;
  deleted_count: number;
};

export type DirectorySettings = {
  model_directory: string;
  output_directory: string;
  dataset_directory: string;
  checkpoint_directory: string;
  working_directory: string;
  classification_dataset_directory?: string;
  classification_output_directory?: string;
  training_dataset_directory?: string;
  training_output_directory?: string;
  device: DevicePreference;
  database_path: string;
};

export type DirectorySettingsUpdate = {
  working_directory: string;
  device: DevicePreference;
};

export type ModelOption = {
  label: string;
  path: string;
  source: "local" | "huggingface";
  compatible: boolean;
  compatibility_error: string | null;
  requires_download: boolean;
};

export type ModelOptionsResponse = {
  local_models: ModelOption[];
  recommended_hf_models: ModelOption[];
};

export type RunCreateResponse = {
  run_id: string;
  status: RunStatus;
};

export type RunStatus =
  | "queued"
  | "running"
  | "completed"
  | "error"
  | "cancelling"
  | "cancelled"
  | "interrupted";

export type RunRecord = {
  run_id: string;
  run_type: string;
  status: RunStatus;
  request: Record<string, unknown>;
  hardware_backend: string;
  model_ref: string | null;
  input_path: string | null;
  output_path: string | null;
  total_items: number;
  processed_items: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  progress_context?: RunProgressContext | null;
};

export type RunProgressContext = {
  scope?: "run" | "epoch";
  epoch_key?: string | null;
  current_epoch?: number | null;
  total_epochs?: number | null;
  current?: number | null;
  total?: number | null;
  label?: string | null;
};

export type RunLogsResponse = {
  logs: string[];
};

export type OpenRunOutputResponse = {
  run_id: string;
  output_path: string;
  opened: boolean;
};

export type RuntimeHardwareResponse = {
  active_backend: string;
  torch_available: boolean;
  backends: Record<string, { available: boolean; label?: string }>;
};

const apiHost = import.meta.env.GALAXY_AI_HOST || "127.0.0.1";
const apiPort = import.meta.env.GALAXY_AI_PORT || "8000";
const apiBaseUrl =
  import.meta.env.VITE_GALAXY_API_BASE_URL || `http://${apiHost}:${apiPort}`;

export async function createImageClassificationTraining(
  request: TrainingRequest,
): Promise<RunCreateResponse> {
  return apiRequest<RunCreateResponse>("/image-classification/training", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function createImageClassificationInference(
  request: BatchInferenceRequest,
): Promise<RunCreateResponse> {
  return apiRequest<RunCreateResponse>("/image-classification/inference", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function getRun(runId: string): Promise<RunRecord> {
  return apiRequest<RunRecord>(`/runs/${runId}`);
}

export async function getRunLogs(runId: string): Promise<RunLogsResponse> {
  return apiRequest<RunLogsResponse>(`/runs/${runId}/logs`);
}

export async function cancelRun(runId: string): Promise<RunCreateResponse> {
  return apiRequest<RunCreateResponse>(`/runs/${runId}/cancel`, {
    method: "POST",
  });
}

export async function openRunOutput(runId: string): Promise<OpenRunOutputResponse> {
  return apiRequest<OpenRunOutputResponse>(`/runs/${runId}/open-output`, {
    method: "POST",
  });
}

export async function getSettings(): Promise<DirectorySettings> {
  return apiRequest<DirectorySettings>("/settings");
}

export async function updateSettings(
  request: DirectorySettingsUpdate,
): Promise<DirectorySettings> {
  return apiRequest<DirectorySettings>("/settings", {
    method: "PUT",
    body: JSON.stringify(request),
  });
}

export async function getModelOptions(
  modelDirectory: string,
): Promise<ModelOptionsResponse> {
  const params = new URLSearchParams({ model_directory: modelDirectory });
  return apiRequest<ModelOptionsResponse>(`/models/options?${params.toString()}`);
}

export async function getRuntimeHardware(): Promise<RuntimeHardwareResponse> {
  return apiRequest<RuntimeHardwareResponse>("/runtime/hardware");
}

export async function getImageClassificationDataset(
  mode: DatasetMode,
): Promise<DatasetPreviewResponse> {
  const params = new URLSearchParams({ mode });
  return apiRequest<DatasetPreviewResponse>(
    `/image-classification/datasets?${params.toString()}`,
  );
}

export async function importImageClassificationDataset({
  mode,
  label,
  files,
}: {
  mode: DatasetMode;
  label?: string;
  files: DatasetImportFile[];
}): Promise<DatasetImportResponse> {
  const body = new FormData();
  body.append("mode", mode);
  if (label) {
    body.append("label", label);
  }
  for (const item of files) {
    body.append("relative_paths[]", item.relativePath);
    body.append("files", item.file, item.file.name);
  }

  return apiRequest<DatasetImportResponse>("/image-classification/datasets/import", {
    method: "POST",
    body,
  });
}

export async function clearImageClassificationDataset(
  mode: DatasetMode,
): Promise<DatasetClearResponse> {
  const params = new URLSearchParams({ mode });
  return apiRequest<DatasetClearResponse>(
    `/image-classification/datasets?${params.toString()}`,
    { method: "DELETE" },
  );
}

export function datasetImageUrl({
  mode,
  relativePath,
  label,
}: {
  mode: DatasetMode;
  relativePath: string;
  label?: string;
}): string {
  const params = new URLSearchParams({
    mode,
    relative_path: relativePath,
  });
  if (label) {
    params.set("label", label);
  }
  return `${apiBaseUrl}/image-classification/datasets/image?${params.toString()}`;
}

async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const isFormData = init.body instanceof FormData;
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers: isFormData
      ? init.headers
      : {
          "Content-Type": "application/json",
          ...init.headers,
        },
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return (await response.json()) as T;
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") {
      return body.detail;
    }
  } catch {
    // Fall through to the status-based message when the backend returned no JSON.
  }

  return `Backend request failed with status ${response.status}.`;
}
