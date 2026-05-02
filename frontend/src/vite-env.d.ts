/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_GALAXY_API_BASE_URL?: string;
  readonly GALAXY_AI_HOST?: string;
  readonly GALAXY_AI_PORT?: string;
  readonly GALAXY_AI_WORKING_DIRECTORY?: string;
  readonly GALAXY_AI_MODEL_DIRECTORY?: string;
  readonly GALAXY_AI_CLASSIFICATION_DATASET_DIRECTORY?: string;
  readonly GALAXY_AI_CLASSIFICATION_OUTPUT_DIRECTORY?: string;
  readonly GALAXY_AI_TRAINING_DATASET_DIRECTORY?: string;
  readonly GALAXY_AI_TRAINING_OUTPUT_DIRECTORY?: string;
  readonly GALAXY_AI_DEVICE?: string;
}

type BrowserDirectoryHandle = {
  kind?: "directory";
  name: string;
  path?: string;
  entries?: () => AsyncIterableIterator<
    [string, BrowserDirectoryHandle | BrowserFileHandle]
  >;
  values?: () => AsyncIterableIterator<BrowserDirectoryHandle | BrowserFileHandle>;
};

type BrowserFileHandle = {
  kind: "file";
  name: string;
  getFile: () => Promise<File>;
};

interface Window {
  showDirectoryPicker?: (options?: {
    mode?: "read" | "readwrite";
  }) => Promise<BrowserDirectoryHandle>;
}
