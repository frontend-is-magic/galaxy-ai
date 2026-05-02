import type { BatchImageClassificationTask } from "./types";
import type { TrainingRequest } from "./api";

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
  working_directory: "~/Projects/NebulaSorter",
  model_directory: "~/Projects/NebulaSorter/models",
  input_directory: "~/Projects/NebulaSorter/input",
  output_directory: "~/Projects/NebulaSorter/output",
  supported_formats: ["jpg", "jpeg", "png", "bmp", "webp"],
  total_images: 12842,
  processed_images: 0,
  error_message: null,
  status_message: "当前无错误",
};

export const defaultTrainingRequest: TrainingRequest = {
  base_model_ref: "",
  model_directory: "~/Projects/NebulaSorter/models",
  allow_download: false,
  dataset_directory: "~/Datasets/NebulaSorter",
  output_directory: "~/Projects/NebulaSorter/output",
  checkpoint_directory: "~/Projects/NebulaSorter/checkpoints",
  epochs: 50,
  batch_size: 8,
  learning_rate: 0.00005,
  seed: 42,
  use_seed: false,
  device: "auto",
};
