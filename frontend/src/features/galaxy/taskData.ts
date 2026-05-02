import type { BatchImageClassificationTask } from "./types";
import type { TrainingRequest } from "./api";
import type { TaskCapabilityMode, TaskDirectories } from "./types";

function envString(key: keyof ImportMetaEnv, fallback: string): string {
  return import.meta.env[key] || fallback;
}

export const defaultNebulaWorkspaceDirectory = envString(
  "GALAXY_AI_WORKING_DIRECTORY",
  "~/Documents/galaxy-ai/nebula-sorter",
);

export const defaultModelDirectory = envString(
  "GALAXY_AI_MODEL_DIRECTORY",
  `${defaultNebulaWorkspaceDirectory}/models`,
);

export const defaultClassificationDatasetDirectory = envString(
  "GALAXY_AI_CLASSIFICATION_DATASET_DIRECTORY",
  `${defaultNebulaWorkspaceDirectory}/classification/datasets`,
);

export const defaultClassificationOutputDirectory = envString(
  "GALAXY_AI_CLASSIFICATION_OUTPUT_DIRECTORY",
  `${defaultNebulaWorkspaceDirectory}/classification/outputs`,
);

export const defaultTrainingDatasetDirectory = envString(
  "GALAXY_AI_TRAINING_DATASET_DIRECTORY",
  `${defaultNebulaWorkspaceDirectory}/training/datasets`,
);

export const defaultTrainingOutputDirectory = envString(
  "GALAXY_AI_TRAINING_OUTPUT_DIRECTORY",
  `${defaultNebulaWorkspaceDirectory}/training/outputs`,
);

export const defaultTrainingCheckpointDirectory = `${defaultNebulaWorkspaceDirectory}/training/checkpoints`;

export function defaultTaskDirectoriesForMode(
  mode: TaskCapabilityMode,
): TaskDirectories {
  return {
    working_directory: defaultNebulaWorkspaceDirectory,
    model_directory: defaultModelDirectory,
    dataset_directory:
      mode === "training"
        ? defaultTrainingDatasetDirectory
        : defaultClassificationDatasetDirectory,
    output_directory:
      mode === "training"
        ? defaultTrainingOutputDirectory
        : defaultClassificationOutputDirectory,
    checkpoint_directory: defaultTrainingCheckpointDirectory,
  };
}

export const nebulaSorterTask: BatchImageClassificationTask = {
  id: "nebula-sorter",
  directory_name: "nebula-sorter",
  name: "星云分拣站",
  task_type: "批量图片分类",
  status: "idle",
  orbit_radius: 4.8,
  hardware_backend: {
    kind: "mps",
    label: "Apple GPU / MPS",
    device_name: "本机图形处理器",
  },
  model_path: "",
  working_directory: defaultNebulaWorkspaceDirectory,
  model_directory: defaultModelDirectory,
  input_directory: defaultClassificationDatasetDirectory,
  output_directory: defaultClassificationOutputDirectory,
  supported_formats: ["jpg", "jpeg", "png", "bmp", "webp"],
  total_images: 12842,
  processed_images: 0,
  error_message: null,
  status_message: "当前无错误",
};

export const defaultTrainingRequest: TrainingRequest = {
  base_model_ref: "",
  model_directory: defaultModelDirectory,
  allow_download: false,
  dataset_directory: defaultTrainingDatasetDirectory,
  output_directory: defaultTrainingOutputDirectory,
  checkpoint_directory: defaultTrainingCheckpointDirectory,
  epochs: 50,
  batch_size: 8,
  learning_rate: 0.00005,
  seed: 42,
  use_seed: false,
  device: "auto",
};
