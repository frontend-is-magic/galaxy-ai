import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./features/galaxy/GalaxyCanvas", () => ({
  GalaxyCanvas: () => <section aria-label="Galaxy AI 星系任务画布" />,
}));

import { App } from "./App";

const defaultSettings = {
  model_directory: "/Users/test/Documents/galaxy-ai/models",
  output_directory: "/Users/test/Documents/galaxy-ai/outputs",
  dataset_directory: "/Users/test/Documents/galaxy-ai/datasets",
  checkpoint_directory: "/Users/test/Documents/galaxy-ai/checkpoints",
  working_directory: "/Users/test/Documents/galaxy-ai",
  device: "auto",
  database_path: "/Users/test/.galaxy-ai/galaxy-ai.sqlite3",
};

const defaultNebulaDirectory = "/Users/test/Documents/galaxy-ai/nebula-sorter";

const modelOptions = {
  local_models: [
    {
      label: "vit-local",
      path: `${defaultNebulaDirectory}/models/vit-local`,
      source: "local",
      compatible: true,
      compatibility_error: null,
      requires_download: false,
    },
  ],
  recommended_hf_models: [
    {
      label: "Microsoft ResNet-50",
      path: "microsoft/resnet-50",
      source: "huggingface",
      compatible: true,
      compatibility_error: null,
      requires_download: true,
    },
    {
      label: "Google ViT Base",
      path: "google/vit-base-patch16-224-in21k",
      source: "huggingface",
      compatible: true,
      compatibility_error: null,
      requires_download: true,
    },
  ],
};

const modelOptionsWithoutLocal = {
  local_models: [],
  recommended_hf_models: modelOptions.recommended_hf_models,
};

const runtimeHardware = {
  active_backend: "mps",
  torch_available: true,
  backends: {
    cpu: { available: true, label: "CPU" },
    cuda: { available: false, label: "CUDA" },
    mps: { available: true, label: "Apple GPU / MPS" },
  },
};

const runningTrainingRun = {
  run_id: "run-training-1",
  run_type: "image_classification_training",
  status: "running",
  request: {
    base_model_ref: "microsoft/resnet-50",
    model_directory: `${defaultNebulaDirectory}/models`,
    dataset_directory: `${defaultNebulaDirectory}/datasets`,
    output_directory: `${defaultNebulaDirectory}/outputs`,
    checkpoint_directory: `${defaultNebulaDirectory}/checkpoints`,
    epochs: 50,
    batch_size: 8,
    learning_rate: 0.00005,
    seed: 42,
    use_seed: false,
    device: "auto",
    allow_download: false,
  },
  hardware_backend: "auto",
  model_ref: "microsoft/resnet-50",
  input_path: "~/Datasets/NebulaSorter",
  output_path: "~/Projects/NebulaSorter/output/run-training-1",
  total_items: 2,
  processed_items: 1,
  error_message: null,
  created_at: "2026-05-01T10:00:00Z",
  updated_at: "2026-05-01T10:00:03Z",
  started_at: "2026-05-01T10:00:01Z",
  completed_at: null,
};

const runningClassificationRun = {
  run_id: "run-classification-1",
  run_type: "image_classification_inference",
  status: "running",
  request: {
    model_ref: `${defaultNebulaDirectory}/models/vit-local`,
    model_directory: `${defaultNebulaDirectory}/models`,
    input_directory: `${defaultNebulaDirectory}/datasets`,
    output_directory: `${defaultNebulaDirectory}/outputs`,
    recursive: true,
    batch_size: 32,
    top_k: 5,
    device: "auto",
    allow_download: false,
  },
  hardware_backend: "auto",
  model_ref: `${defaultNebulaDirectory}/models/vit-local`,
  input_path: `${defaultNebulaDirectory}/datasets`,
  output_path: `${defaultNebulaDirectory}/outputs/run-classification-1`,
  total_items: 10,
  processed_items: 3,
  error_message: null,
  created_at: "2026-05-01T10:00:00Z",
  updated_at: "2026-05-01T10:00:03Z",
  started_at: "2026-05-01T10:00:01Z",
  completed_at: null,
};

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(defaultFetchMock));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Galaxy AI home", () => {
  it("selects the Nebula Sorter task planet by default", async () => {
    render(<App />);

    expect(screen.getAllByText("星云分拣站")[0]).toBeInTheDocument();
    expect(screen.getAllByText("批量图片分类")[0]).toBeInTheDocument();
    expect(await screen.findByText("本地运行")).toBeInTheDocument();
  });

  it("initializes task directories from global settings", async () => {
    render(<App />);

    expect(await screen.findByDisplayValue(defaultNebulaDirectory)).toBeInTheDocument();
    expect(
      screen.getByDisplayValue(`${defaultNebulaDirectory}/datasets`),
    ).toBeInTheDocument();
    expect(
      screen.getByDisplayValue(`${defaultNebulaDirectory}/outputs`),
    ).toBeInTheDocument();
  });

  it("auto-selects the first HF model when no local model is available", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);

        if (url.startsWith("http://127.0.0.1:8000/models/options")) {
          return Promise.resolve(jsonResponse(modelOptionsWithoutLocal));
        }

        return defaultFetchMock(input);
      }),
    );

    render(<App />);

    const modelSelect = await screen.findByRole("combobox", { name: "基础模型" });
    await waitFor(() => {
      expect(modelSelect).toHaveValue("microsoft/resnet-50");
    });
    expect(screen.queryByText("~/Models/resnet50")).not.toBeInTheDocument();
  });

  it("falls back to auto when persisted settings do not include a device", async () => {
    const legacySettings = { ...defaultSettings };
    delete (legacySettings as Partial<typeof defaultSettings>).device;
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);

        if (url === "http://127.0.0.1:8000/settings") {
          return Promise.resolve(jsonResponse(legacySettings));
        }

        return defaultFetchMock(input);
      }),
    );

    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByDisplayValue(defaultNebulaDirectory)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "设置" }));

    expect(await screen.findByRole("combobox", { name: "硬件设备" })).toHaveValue(
      "auto",
    );
  });

  it("removes display mode tabs and keeps more settings collapsed by default", async () => {
    render(<App />);

    expect(await screen.findByDisplayValue(defaultNebulaDirectory)).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "星图模式" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "专业模式" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tablist", { name: "展示模式" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /更多设置/ })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    expect(screen.getByRole("combobox", { name: "基础模型" })).toBeInTheDocument();
    expect(screen.queryByText("模型目录")).not.toBeInTheDocument();
  });

  it("switches between classification and training modes", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "训练模式" }));

    expect(screen.getByText("训练摘要")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "训练模式" })).toHaveAttribute(
      "aria-selected",
      "true",
    );

    await user.click(screen.getByRole("tab", { name: "分类模式" }));

    expect(screen.getByText(/快速启动批量图片分类任务/)).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "分类模式" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("posts classification requests to the backend and renders run status", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (
        url === "http://127.0.0.1:8000/image-classification/inference" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(
          jsonResponse({ run_id: "run-classification-1", status: "queued" }, 202),
        );
      }

      if (url === "http://127.0.0.1:8000/runs/run-classification-1/logs") {
        return Promise.resolve(jsonResponse({ logs: ["分类任务已排队"] }));
      }

      if (url === "http://127.0.0.1:8000/runs/run-classification-1") {
        return Promise.resolve(jsonResponse(runningClassificationRun));
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: "基础模型" })).toHaveValue(
        `${defaultNebulaDirectory}/models/vit-local`,
      );
    });
    await user.click(screen.getByRole("button", { name: /开始分类/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/image-classification/inference",
        expect.objectContaining({ method: "POST" }),
      );
    });
    const classificationCall = fetchMock.mock.calls.find(
      ([url]) => url === "http://127.0.0.1:8000/image-classification/inference",
    );
    const classificationRequest = JSON.parse(
      String((classificationCall?.[1] as RequestInit).body),
    );
    expect(classificationRequest).toEqual({
      model_ref: `${defaultNebulaDirectory}/models/vit-local`,
      model_directory: `${defaultNebulaDirectory}/models`,
      allow_download: false,
      input_directory: `${defaultNebulaDirectory}/datasets`,
      output_directory: `${defaultNebulaDirectory}/outputs`,
      recursive: true,
      batch_size: 32,
      top_k: 5,
      device: "auto",
    });
    expect(await screen.findByText("run-classification-1")).toBeInTheDocument();
    expect(screen.getByText("分类任务已排队")).toBeInTheDocument();
  });

  it("locks training while classification is running and confirms cancellation from the action button", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (
        url === "http://127.0.0.1:8000/image-classification/inference" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(
          jsonResponse({ run_id: "run-classification-1", status: "queued" }, 202),
        );
      }

      if (
        url === "http://127.0.0.1:8000/runs/run-classification-1/cancel" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(
          jsonResponse({ run_id: "run-classification-1", status: "cancelling" }),
        );
      }

      if (url === "http://127.0.0.1:8000/runs/run-classification-1/logs") {
        return Promise.resolve(jsonResponse({ logs: ["分类运行中"] }));
      }

      if (url === "http://127.0.0.1:8000/runs/run-classification-1") {
        return Promise.resolve(jsonResponse(runningClassificationRun));
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: "基础模型" })).toHaveValue(
        `${defaultNebulaDirectory}/models/vit-local`,
      );
    });
    await user.click(screen.getByRole("button", { name: "开始分类" }));

    expect(await screen.findByRole("button", { name: "终止分类" })).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "分类进度" })).toHaveAttribute(
      "aria-valuenow",
      "30",
    );
    expect(screen.getByText("分类运行中 · 3 / 10")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "训练模式" }));

    expect(screen.queryByRole("button", { name: "开始训练" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "终止分类" })).toBeInTheDocument();
    expect(screen.getByText("分类运行中，训练已锁定")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "终止分类" }));
    expect(
      screen.getByRole("dialog", { name: "确认终止分类任务" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "继续运行" }));
    expect(fetchMock).not.toHaveBeenCalledWith(
      "http://127.0.0.1:8000/runs/run-classification-1/cancel",
      expect.anything(),
    );

    await user.click(screen.getByRole("button", { name: "终止分类" }));
    await user.click(screen.getByRole("button", { name: "确认终止" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/runs/run-classification-1/cancel",
        expect.objectContaining({ method: "POST" }),
      );
    });
  });

  it("opens global settings for the default working directory only", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "设置" }));

    expect(
      await screen.findByRole("dialog", { name: "本地默认设置" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("默认工作目录")).toHaveValue(
      "/Users/test/Documents/galaxy-ai",
    );
    expect(screen.getByLabelText("默认工作目录")).toBeDisabled();
    expect(
      screen.queryByRole("button", { name: "选择默认工作目录" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "硬件设备" })).toHaveValue("auto");
    expect(screen.queryByText("默认模型")).not.toBeInTheDocument();
  });

  it("renders interactive summary controls without opening more settings", async () => {
    render(<App />);

    const modelSelect = await screen.findByRole("combobox", { name: "基础模型" });
    expect(modelSelect).toBeInTheDocument();
    expect(
      await screen.findByRole("option", { name: "vit-local" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: "Microsoft ResNet-50" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "选择工作目录" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "选择数据集目录" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "选择输出目录" })).toBeInTheDocument();
  });

  it("saves the working directory and derives task directories from it", async () => {
    const customSettings = {
      ...defaultSettings,
      model_directory: "/Users/test/custom-lab/models",
      output_directory: "/Users/test/custom-lab/outputs",
      dataset_directory: "/Users/test/custom-lab/datasets",
      checkpoint_directory: "/Users/test/custom-lab/checkpoints",
      working_directory: "/Users/test/custom-lab",
      device: "cpu",
    };
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (url === "http://127.0.0.1:8000/settings" && init?.method !== "PUT") {
        return Promise.resolve(jsonResponse(defaultSettings));
      }

      if (url === "http://127.0.0.1:8000/settings" && init?.method === "PUT") {
        return Promise.resolve(jsonResponse(customSettings));
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);
    const directoryPicker = vi.fn();
    vi.stubGlobal("showDirectoryPicker", directoryPicker);

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "设置" }));
    await user.selectOptions(screen.getByRole("combobox", { name: "硬件设备" }), "cpu");
    await user.click(screen.getByRole("button", { name: "保存设置" }));

    await waitFor(() => {
      expect(
        screen.queryByRole("dialog", { name: "本地默认设置" }),
      ).not.toBeInTheDocument();
    });
    expect(
      await screen.findByDisplayValue("/Users/test/custom-lab/nebula-sorter"),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /更多设置/ }));
    expect(
      screen.getByDisplayValue("/Users/test/custom-lab/nebula-sorter/models"),
    ).toBeInTheDocument();
    expect(directoryPicker).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/settings",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({
          working_directory: "/Users/test/Documents/galaxy-ai",
          device: "cpu",
        }),
      }),
    );
    expect(screen.getByText("全局设备：CPU")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "设置" }));
    expect(screen.getByRole("combobox", { name: "硬件设备" })).toHaveValue("cpu");
    expect(fetchMock).not.toHaveBeenCalledWith(
      "http://127.0.0.1:8000/settings/select-directory",
      expect.anything(),
    );
  });

  it("blocks task-local directory changes with a toast", async () => {
    const directoryPicker = vi.fn();
    const fetchMock = vi.fn(defaultFetchMock);
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("showDirectoryPicker", directoryPicker);

    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "选择工作目录" }));

    expect(await screen.findByText("当前版本不允许更改目录")).toBeInTheDocument();
    expect(screen.getByDisplayValue(defaultNebulaDirectory)).toBeDisabled();
    expect(directoryPicker).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: /更多设置/ }));
    expect(screen.getByDisplayValue(`${defaultNebulaDirectory}/models`)).toBeDisabled();
    expect(fetchMock).not.toHaveBeenCalledWith(
      "http://127.0.0.1:8000/settings/select-directory",
      expect.anything(),
    );
  });

  it("posts training requests to the backend and renders run status with logs", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (url === "http://127.0.0.1:8000/settings") {
        return Promise.resolve(jsonResponse(defaultSettings));
      }

      if (
        url === "http://127.0.0.1:8000/image-classification/training" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(
          jsonResponse({ run_id: "run-training-1", status: "queued" }, 202),
        );
      }

      if (url === "http://127.0.0.1:8000/runs/run-training-1/logs") {
        return Promise.resolve(
          jsonResponse({ logs: ["训练任务已排队", "正在加载数据集"] }),
        );
      }

      if (url === "http://127.0.0.1:8000/runs/run-training-1") {
        return Promise.resolve(jsonResponse(runningTrainingRun));
      }

      if (url.startsWith("http://127.0.0.1:8000/models/options")) {
        return Promise.resolve(jsonResponse(modelOptions));
      }

      return Promise.resolve(jsonResponse({}, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "训练模式" }));
    expect(screen.queryByRole("button", { name: /更多设置/ })).not.toBeInTheDocument();
    expect(screen.getByText("训练设置")).toBeInTheDocument();
    expect(screen.getByText("训练轮次")).toBeInTheDocument();
    expect(screen.getByText("批大小")).toBeInTheDocument();
    expect(screen.getByText("学习率")).toBeInTheDocument();
    expect(screen.getByText("随机种子")).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "训练轮次" })).toHaveValue(50);
    const seedInput = screen.getByRole("spinbutton", { name: "随机种子" });
    expect(seedInput).toHaveValue(42);
    expect(seedInput).toBeDisabled();
    await user.click(screen.getByRole("checkbox", { name: "启用随机种子" }));
    expect(seedInput).toBeEnabled();
    expect(
      await screen.findByDisplayValue(`${defaultNebulaDirectory}/outputs`),
    ).toBeInTheDocument();
    expect(screen.getByDisplayValue(`${defaultNebulaDirectory}/models`)).toBeDisabled();
    expect(
      screen.getByDisplayValue(`${defaultNebulaDirectory}/checkpoints`),
    ).toBeDisabled();
    const batchSizeSelect = screen.getByRole("combobox", { name: "批大小" });
    expect(batchSizeSelect).toHaveValue("8");
    expect(screen.getByRole("option", { name: "8" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "16" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "32" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "64" })).toBeInTheDocument();
    await user.selectOptions(batchSizeSelect, "64");
    await user.click(screen.getByRole("button", { name: "开始训练" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/image-classification/training",
        expect.objectContaining({ method: "POST" }),
      );
    });

    const trainingCall = fetchMock.mock.calls.find(
      ([url]) => url === "http://127.0.0.1:8000/image-classification/training",
    );
    const trainingRequest = JSON.parse(String((trainingCall?.[1] as RequestInit).body));
    expect(trainingRequest).toEqual({
      base_model_ref: `${defaultNebulaDirectory}/models/vit-local`,
      model_directory: `${defaultNebulaDirectory}/models`,
      dataset_directory: `${defaultNebulaDirectory}/datasets`,
      output_directory: `${defaultNebulaDirectory}/outputs`,
      checkpoint_directory: `${defaultNebulaDirectory}/checkpoints`,
      epochs: 50,
      batch_size: 64,
      learning_rate: 0.00005,
      seed: 42,
      use_seed: true,
      device: "auto",
      allow_download: false,
    });

    expect(await screen.findByText("run-training-1")).toBeInTheDocument();
    expect(screen.getByText("正在加载数据集")).toBeInTheDocument();
    expect(screen.getAllByText("运行中")[0]).toBeInTheDocument();
  });

  it("blocks HF training until explicit download is enabled", async () => {
    const fetchMock = vi.fn(defaultFetchMock);
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "训练模式" }));
    await user.selectOptions(
      await screen.findByRole("combobox", { name: "基础模型" }),
      "microsoft/resnet-50",
    );

    expect(
      await screen.findByText(/需勾选允许显式下载模型，或选择本地已缓存模型/),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始训练" })).toBeDisabled();
    expect(fetchMock).not.toHaveBeenCalledWith(
      "http://127.0.0.1:8000/image-classification/training",
      expect.anything(),
    );
  });

  it("allows HF training after explicit download is enabled", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (
        url === "http://127.0.0.1:8000/image-classification/training" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(
          jsonResponse({ run_id: "run-training-1", status: "queued" }, 202),
        );
      }

      if (url === "http://127.0.0.1:8000/runs/run-training-1/logs") {
        return Promise.resolve(jsonResponse({ logs: ["允许下载模型"] }));
      }

      if (url === "http://127.0.0.1:8000/runs/run-training-1") {
        return Promise.resolve(jsonResponse(runningTrainingRun));
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "训练模式" }));
    await user.selectOptions(
      await screen.findByRole("combobox", { name: "基础模型" }),
      "microsoft/resnet-50",
    );
    await user.click(screen.getByRole("checkbox", { name: "允许显式下载模型" }));
    await user.click(screen.getByRole("button", { name: "开始训练" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/image-classification/training",
        expect.objectContaining({ method: "POST" }),
      );
    });
    const trainingCall = fetchMock.mock.calls.find(
      ([url]) => url === "http://127.0.0.1:8000/image-classification/training",
    );
    const trainingRequest = JSON.parse(String((trainingCall?.[1] as RequestInit).body));
    expect(trainingRequest.allow_download).toBe(true);
  });

  it("keeps more settings in classification and uses fixed batch size options", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByDisplayValue(defaultNebulaDirectory)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /更多设置/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /更多设置/ }));

    const batchSizeSelect = screen.getByRole("combobox", { name: "批大小" });
    expect(batchSizeSelect).toHaveValue("32");
    expect(screen.getByRole("option", { name: "8" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "16" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "32" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "64" })).toBeInTheDocument();

    await user.selectOptions(batchSizeSelect, "16");

    expect(batchSizeSelect).toHaveValue("16");
  });

  it("keeps hardware settings out of the task inspector", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByDisplayValue(defaultNebulaDirectory)).toBeInTheDocument();
    expect(screen.queryByText("硬件后端")).not.toBeInTheDocument();
    expect(screen.queryByText("计算后端")).not.toBeInTheDocument();
    expect(screen.queryByText("设备")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /更多设置/ }));

    expect(screen.queryByText("硬件后端")).not.toBeInTheDocument();
    expect(screen.queryByText("计算后端")).not.toBeInTheDocument();
    expect(screen.queryByText("设备")).not.toBeInTheDocument();
  });

  it("locks classification while training is running and confirms cancellation from the action button", async () => {
    let runFetchCount = 0;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (url === "http://127.0.0.1:8000/settings") {
        return Promise.resolve(jsonResponse(defaultSettings));
      }

      if (
        url === "http://127.0.0.1:8000/image-classification/training" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(
          jsonResponse({ run_id: "run-training-1", status: "queued" }, 202),
        );
      }

      if (
        url === "http://127.0.0.1:8000/runs/run-training-1/cancel" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(
          jsonResponse({ run_id: "run-training-1", status: "cancelling" }),
        );
      }

      if (url === "http://127.0.0.1:8000/runs/run-training-1/logs") {
        return Promise.resolve(
          jsonResponse({
            logs:
              runFetchCount > 1
                ? ["正在训练分类头", "正在取消运行"]
                : ["正在训练分类头"],
          }),
        );
      }

      if (url === "http://127.0.0.1:8000/runs/run-training-1") {
        runFetchCount += 1;
        return Promise.resolve(
          jsonResponse(
            runFetchCount > 1
              ? { ...runningTrainingRun, status: "cancelling" }
              : runningTrainingRun,
          ),
        );
      }

      if (url.startsWith("http://127.0.0.1:8000/models/options")) {
        return Promise.resolve(jsonResponse(modelOptions));
      }

      return Promise.resolve(jsonResponse({}, 404));
    });
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "训练模式" }));
    await user.click(screen.getByRole("button", { name: "开始训练" }));
    await screen.findByText("run-training-1");

    expect(screen.getByRole("button", { name: "终止训练" })).toBeInTheDocument();
    expect(
      screen.getByRole("progressbar", { name: "训练当前轮次进度" }),
    ).toHaveAttribute("aria-valuenow", "50");
    expect(screen.getByText("当前训练轮次进度 · 1 / 2")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "分类模式" }));

    expect(screen.queryByRole("button", { name: "开始分类" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "终止训练" })).toBeInTheDocument();
    expect(screen.getByText("训练运行中，分类已锁定")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "终止训练" }));
    expect(
      screen.getByRole("dialog", { name: "确认终止训练任务" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "确认终止" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/runs/run-training-1/cancel",
        expect.objectContaining({ method: "POST" }),
      );
    });
    expect(
      screen.queryByRole("dialog", { name: "确认终止训练任务" }),
    ).not.toBeInTheDocument();
  });

  it("removes task status and inline status hint areas from the panel body", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByDisplayValue(defaultNebulaDirectory)).toBeInTheDocument();
    expect(screen.queryByText("任务状态")).not.toBeInTheDocument();
    expect(screen.queryByText("错误信息")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /更多设置/ }));

    expect(screen.queryByText("任务状态")).not.toBeInTheDocument();
    expect(screen.queryByText("错误信息")).not.toBeInTheDocument();
  });
});

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function defaultFetchMock(input: RequestInfo | URL): Promise<Response> {
  const url = String(input);

  if (url.startsWith("http://127.0.0.1:8000/models/options")) {
    return Promise.resolve(jsonResponse(modelOptions));
  }

  if (url === "http://127.0.0.1:8000/settings") {
    return Promise.resolve(jsonResponse(defaultSettings));
  }

  if (url === "http://127.0.0.1:8000/runtime/hardware") {
    return Promise.resolve(jsonResponse(runtimeHardware));
  }

  if (url === "http://127.0.0.1:8000/settings/select-directory") {
    return Promise.resolve(jsonResponse({ selected_directory: null }));
  }

  return Promise.resolve(jsonResponse({}, 404));
}
