export type HardwareBackendKind = "cuda" | "mps" | "cpu";

export type HardwareBackend = {
  kind: HardwareBackendKind;
  label: string;
  device_name: string;
};

export type TaskStatus = "idle" | "running" | "completed" | "error";

export type TaskViewMode = "starmap" | "professional";

export type TaskCapabilityMode = "classification" | "training";

export type TaskPlanet = {
  id: string;
  name: string;
  task_type: string;
  status: TaskStatus;
  orbit_radius: number;
};

export type BatchImageClassificationTask = TaskPlanet & {
  hardware_backend: HardwareBackend;
  model_path: string;
  working_directory: string;
  input_directory: string;
  output_directory: string;
  supported_formats: string[];
  total_images: number;
  processed_images: number;
  error_message: string | null;
  status_message: string;
};
