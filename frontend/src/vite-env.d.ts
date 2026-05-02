/// <reference types="vite/client" />

type BrowserDirectoryHandle = {
  name: string;
  path?: string;
};

interface Window {
  showDirectoryPicker?: (options?: {
    mode?: "read" | "readwrite";
  }) => Promise<BrowserDirectoryHandle>;
}
