import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./features/galaxy/GalaxyCanvas", () => ({
  GalaxyCanvas: () => <section aria-label="Galaxy AI 星系任务画布" />,
}));

import { App } from "./App";

const defaultNebulaDirectory = "/Users/test/Documents/galaxy-ai/nebula-sorter";
const classificationDatasetDirectory = `${defaultNebulaDirectory}/classification/datasets`;
const classificationOutputDirectory = `${defaultNebulaDirectory}/classification/outputs`;
const trainingDatasetDirectory = `${defaultNebulaDirectory}/training/datasets`;
const trainingOutputDirectory = `${defaultNebulaDirectory}/training/outputs`;
const trainingCheckpointDirectory = `${defaultNebulaDirectory}/training/checkpoints`;

const defaultSettings = {
  model_directory: `${defaultNebulaDirectory}/models`,
  output_directory: classificationOutputDirectory,
  dataset_directory: classificationDatasetDirectory,
  checkpoint_directory: trainingCheckpointDirectory,
  working_directory: defaultNebulaDirectory,
  classification_dataset_directory: classificationDatasetDirectory,
  classification_output_directory: classificationOutputDirectory,
  training_dataset_directory: trainingDatasetDirectory,
  training_output_directory: trainingOutputDirectory,
  device: "auto",
  database_path: "/Users/test/.galaxy-ai/galaxy-ai.sqlite3",
};

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

const emptyClassificationDataset = {
  mode: "classification",
  count: 0,
  items: [],
  labels: [],
};

const emptyTrainingDataset = {
  mode: "training",
  count: 0,
  items: [],
  labels: [],
};

const runningTrainingRun = {
  run_id: "run-training-1",
  run_type: "image_classification_training",
  status: "running",
  request: {
    base_model_ref: "microsoft/resnet-50",
    model_directory: `${defaultNebulaDirectory}/models`,
    dataset_directory: trainingDatasetDirectory,
    output_directory: trainingOutputDirectory,
    checkpoint_directory: trainingCheckpointDirectory,
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
  total_items: 128,
  processed_items: 32,
  progress_context: {
    scope: "epoch",
    current: 4,
    total: 8,
    current_epoch: 3,
    total_epochs: 50,
    label: "当前训练轮次",
  },
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
    input_directory: classificationDatasetDirectory,
    output_directory: classificationOutputDirectory,
    recursive: true,
    batch_size: 32,
    top_k: 5,
    device: "auto",
    allow_download: false,
  },
  hardware_backend: "auto",
  model_ref: `${defaultNebulaDirectory}/models/vit-local`,
  input_path: classificationDatasetDirectory,
  output_path: `${classificationOutputDirectory}/run-classification-1`,
  total_items: 10,
  processed_items: 3,
  error_message: null,
  created_at: "2026-05-01T10:00:00Z",
  updated_at: "2026-05-01T10:00:03Z",
  started_at: "2026-05-01T10:00:01Z",
  completed_at: null,
};

const completedClassificationRun = {
  ...runningClassificationRun,
  status: "completed",
  processed_items: 10,
  completed_at: "2026-05-01T10:00:05Z",
};

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(defaultFetchMock));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Galaxy AI home", () => {
  it("selects the Nebula Sorter task planet by default", async () => {
    render(<App />);

    expect(screen.getAllByText("星云分拣站")[0]).toBeInTheDocument();
    expect(screen.getAllByText("批量图片分类")[0]).toBeInTheDocument();
    expect(await screen.findByText("本地运行")).toBeInTheDocument();
  });

  it("shows the fixed workspace in the header", async () => {
    render(<App />);

    expect(await screen.findByText("工作目录：")).toBeInTheDocument();
    expect(await screen.findByText(defaultNebulaDirectory)).toBeInTheDocument();
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

  it("refreshes base model options when local models change", async () => {
    let allowRefreshedOptions = false;
    const refreshedModelOptions = {
      ...modelOptions,
      local_models: [
        ...modelOptions.local_models,
        {
          label: "new-local-model",
          path: `${defaultNebulaDirectory}/models/new-local-model`,
          source: "local",
          compatible: true,
          compatibility_error: null,
          requires_download: false,
        },
      ],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);

        if (url.startsWith("http://127.0.0.1:8000/models/options")) {
          return Promise.resolve(
            jsonResponse(allowRefreshedOptions ? refreshedModelOptions : modelOptions),
          );
        }

        return defaultFetchMock(input);
      }),
    );

    render(<App />);

    expect(
      await screen.findByRole("option", { name: "vit-local" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("option", { name: "new-local-model" }),
    ).not.toBeInTheDocument();

    allowRefreshedOptions = true;
    act(() => {
      window.dispatchEvent(new Event("focus"));
    });

    expect(
      await screen.findByRole("option", { name: "new-local-model" }),
    ).toBeInTheDocument();
  });

  it("allows classification with an HF model without explicit download controls", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (url.startsWith("http://127.0.0.1:8000/models/options")) {
        return Promise.resolve(jsonResponse(modelOptionsWithoutLocal));
      }

      if (
        url === "http://127.0.0.1:8000/image-classification/inference" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(
          jsonResponse({ run_id: "run-classification-1", status: "queued" }, 202),
        );
      }

      if (url === "http://127.0.0.1:8000/runs/run-classification-1/logs") {
        return Promise.resolve(jsonResponse({ logs: ["允许下载模型"] }));
      }

      if (url === "http://127.0.0.1:8000/runs/run-classification-1") {
        return Promise.resolve(
          jsonResponse({
            ...runningClassificationRun,
            model_ref: "microsoft/resnet-50",
            request: {
              ...runningClassificationRun.request,
              model_ref: "microsoft/resnet-50",
              allow_download: true,
            },
          }),
        );
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);

    const modelSelect = await screen.findByRole("combobox", { name: "基础模型" });
    await waitFor(() => {
      expect(modelSelect).toHaveValue("microsoft/resnet-50");
    });
    expect(
      screen.queryByText(/需勾选允许显式下载模型，或选择本地已缓存模型/),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始分类" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /更多设置/ }));
    expect(
      screen.queryByRole("checkbox", { name: "允许显式下载模型" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始分类" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "开始分类" }));

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
    expect(classificationRequest).toEqual(
      expect.objectContaining({
        model_ref: "microsoft/resnet-50",
        allow_download: true,
      }),
    );
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

    expect(await screen.findByText(defaultNebulaDirectory)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "设置" }));

    expect(await screen.findByRole("combobox", { name: "硬件设备" })).toHaveValue(
      "auto",
    );
  });

  it("removes display mode tabs and keeps more settings collapsed by default", async () => {
    render(<App />);

    expect(await screen.findByText(defaultNebulaDirectory)).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "星图模式" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "专业模式" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tablist", { name: "展示模式" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /更多设置/ })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    expect(screen.getByRole("combobox", { name: "基础模型" })).toBeInTheDocument();
    expect(screen.queryByText("模型目录")).not.toBeInTheDocument();
    expect(screen.queryByText("工作目录")).not.toBeInTheDocument();
    expect(screen.queryByText("数据集目录")).not.toBeInTheDocument();
    expect(screen.queryByText("输出目录")).not.toBeInTheDocument();
    expect(screen.queryByText("Checkpoint目录")).not.toBeInTheDocument();
    expect(screen.queryByText("输出路径")).not.toBeInTheDocument();
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
      input_directory: classificationDatasetDirectory,
      output_directory: classificationOutputDirectory,
      recursive: true,
      batch_size: 32,
      top_k: 5,
      device: "auto",
    });
    expect(await screen.findByText("run-classification-1")).toBeInTheDocument();
    expect(screen.getByText("分类任务已排队")).toBeInTheDocument();
  });

  it("shows a result button for completed classification runs and opens the output directory", async () => {
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
        return Promise.resolve(jsonResponse({ logs: ["分类完成"] }));
      }

      if (url === "http://127.0.0.1:8000/runs/run-classification-1") {
        return Promise.resolve(jsonResponse(completedClassificationRun));
      }

      if (
        url === "http://127.0.0.1:8000/runs/run-classification-1/open-output" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(
          jsonResponse({
            run_id: "run-classification-1",
            output_path: completedClassificationRun.output_path,
            opened: true,
          }),
        );
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
    const resultButton = await screen.findByRole("button", { name: "查看结果" });

    await user.click(resultButton);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/runs/run-classification-1/open-output",
        expect.objectContaining({ method: "POST" }),
      );
    });
  });

  it("shows the open-output error when opening classification results fails", async () => {
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
        return Promise.resolve(jsonResponse({ logs: ["分类完成"] }));
      }

      if (url === "http://127.0.0.1:8000/runs/run-classification-1") {
        return Promise.resolve(jsonResponse(completedClassificationRun));
      }

      if (
        url === "http://127.0.0.1:8000/runs/run-classification-1/open-output" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(jsonResponse({ detail: "无法打开结果目录。" }, 500));
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
    await user.click(await screen.findByRole("button", { name: "查看结果" }));

    expect(await screen.findByText("无法打开结果目录。")).toBeInTheDocument();
  });

  it("does not show the classification result button while training mode is selected", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);

      if (url === "http://127.0.0.1:8000/runs/run-classification-1") {
        return Promise.resolve(jsonResponse(completedClassificationRun));
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "训练模式" }));

    expect(screen.queryByRole("button", { name: "查看结果" })).not.toBeInTheDocument();
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

    const classificationCancelButton = await screen.findByRole("button", {
      name: /终止分类\s+3\/10/,
    });
    expect(classificationCancelButton).toBeInTheDocument();
    expect(classificationCancelButton).toHaveClass("justify-center");
    expect(classificationCancelButton).not.toHaveClass("grid-cols-[1fr_auto_1fr]");
    expect(classificationCancelButton).not.toHaveTextContent("30%");
    expect(classificationCancelButton.querySelector("svg")).toHaveClass("fill-current");
    expect(
      screen.queryByRole("progressbar", { name: "分类进度" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("分类运行中 · 3 / 10")).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "训练模式" }));

    expect(screen.queryByRole("button", { name: "开始训练" })).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /终止分类\s+3\/10/ }),
    ).toBeInTheDocument();
    expect(screen.queryByText("分类运行中，训练已锁定")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /终止分类\s+3\/10/ }));
    expect(
      screen.getByRole("dialog", { name: "确认终止分类任务" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "继续运行" }));
    expect(fetchMock).not.toHaveBeenCalledWith(
      "http://127.0.0.1:8000/runs/run-classification-1/cancel",
      expect.anything(),
    );

    await user.click(screen.getByRole("button", { name: /终止分类\s+3\/10/ }));
    await user.click(screen.getByRole("button", { name: "确认终止" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/runs/run-classification-1/cancel",
        expect.objectContaining({ method: "POST" }),
      );
    });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /终止分类\s+3\/10/ })).toBeDisabled();
    });
  });

  it("opens global settings for the default working directory only", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "设置" }));

    expect(
      await screen.findByRole("dialog", { name: "本地默认设置" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("默认工作目录")).toHaveValue(defaultNebulaDirectory);
    expect(screen.getByLabelText("默认工作目录")).toBeDisabled();
    expect(
      screen.queryByRole("button", { name: "选择默认工作目录" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "硬件设备" })).toHaveValue("auto");
    expect(screen.getByRole("combobox", { name: "硬件设备" })).toBeDisabled();
    expect(screen.queryByText("默认模型")).not.toBeInTheDocument();
  });

  it("renders model and dataset controls without panel directory fields", async () => {
    render(<App />);

    const modelSelect = await screen.findByRole("combobox", { name: "基础模型" });
    expect(modelSelect).toBeInTheDocument();
    expect(
      await screen.findByRole("option", { name: "vit-local" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: "Microsoft ResNet-50" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "添加分类图片" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "删除分类数据集" })).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "选择工作目录" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "选择数据集目录" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "选择输出目录" }),
    ).not.toBeInTheDocument();
  });

  it("renders all classification dataset images in a scrollable thumbnail preview", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);

        if (
          url ===
          "http://127.0.0.1:8000/image-classification/datasets?mode=classification"
        ) {
          return Promise.resolve(
            jsonResponse({
              ...emptyClassificationDataset,
              count: 7,
              items: Array.from({ length: 7 }, (_, index) => ({
                file_name: `cat-${index + 1}.jpg`,
                relative_path: `cats/cat-${index + 1}.jpg`,
                size: 2048 + index,
              })),
            }),
          );
        }

        return defaultFetchMock(input);
      }),
    );

    render(<App />);

    expect(await screen.findByRole("img", { name: "cat-1.jpg" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "cat-7.jpg" })).toBeInTheDocument();
    expect(screen.getByTestId("dataset-preview-scroll")).toBeInTheDocument();
    expect(screen.getAllByRole("img")).toHaveLength(7);
    expect(screen.getByRole("img", { name: "cat-1.jpg" })).toHaveAttribute(
      "src",
      expect.stringContaining(
        "/image-classification/datasets/image?mode=classification",
      ),
    );
    expect(screen.getByRole("img", { name: "cat-1.jpg" })).toHaveAttribute(
      "src",
      expect.stringContaining("relative_path=cats%2Fcat-1.jpg"),
    );
  });

  it("switches the training preview when selecting a different label", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);

        if (
          url === "http://127.0.0.1:8000/image-classification/datasets?mode=training"
        ) {
          return Promise.resolve(
            jsonResponse({
              ...emptyTrainingDataset,
              count: 3,
              labels: [
                {
                  label: "cat",
                  count: 2,
                  items: [
                    { file_name: "cat-1.jpg", relative_path: "cat-1.jpg", size: 3 },
                    { file_name: "cat-2.jpg", relative_path: "cat-2.jpg", size: 4 },
                  ],
                },
                {
                  label: "dog",
                  count: 1,
                  items: [
                    { file_name: "dog-1.jpg", relative_path: "dog-1.jpg", size: 5 },
                  ],
                },
              ],
            }),
          );
        }

        return defaultFetchMock(input);
      }),
    );

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "训练模式" }));

    const labelList = await screen.findByLabelText("训练 label 列表");
    expect(within(labelList).getByRole("button", { name: "cat" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(within(labelList).getByRole("button", { name: "dog" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(screen.getByRole("img", { name: "cat-1.jpg" })).toHaveAttribute(
      "src",
      expect.stringContaining("mode=training"),
    );
    expect(screen.getByRole("img", { name: "cat-1.jpg" })).toHaveAttribute(
      "src",
      expect.stringContaining("label=cat"),
    );
    expect(screen.queryByRole("img", { name: "dog-1.jpg" })).not.toBeInTheDocument();

    await user.click(within(labelList).getByRole("button", { name: "dog" }));
    expect(within(labelList).getByRole("button", { name: "dog" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.queryByRole("img", { name: "cat-1.jpg" })).not.toBeInTheDocument();
    expect(screen.getByRole("img", { name: "dog-1.jpg" })).toHaveAttribute(
      "src",
      expect.stringContaining("label=dog"),
    );
  });

  it("imports classification images from a browser-selected directory", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (
        url ===
        "http://127.0.0.1:8000/image-classification/datasets?mode=classification"
      ) {
        return Promise.resolve(
          jsonResponse({
            ...emptyClassificationDataset,
            count: init?.method === "DELETE" ? 0 : 1,
            items:
              init?.method === "DELETE"
                ? []
                : [{ file_name: "cat.jpg", relative_path: "cats/cat.jpg", size: 3 }],
          }),
        );
      }

      if (
        url ===
          "http://127.0.0.1:8000/image-classification/datasets?mode=classification" &&
        init?.method === "DELETE"
      ) {
        return Promise.resolve(
          jsonResponse({ mode: "classification", deleted_count: 1 }),
        );
      }

      if (
        url === "http://127.0.0.1:8000/image-classification/datasets/import" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(
          jsonResponse({ mode: "classification", imported_count: 1 }),
        );
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal(
      "showDirectoryPicker",
      vi.fn(() =>
        Promise.resolve(
          directoryHandle("selected", [
            [
              "cats",
              directoryHandle("cats", [
                ["cat.jpg", fileHandle("cat.jpg", "cat")],
                ["readme.txt", fileHandle("readme.txt", "skip")],
              ]),
            ],
          ]),
        ),
      ),
    );

    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "添加分类图片" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/image-classification/datasets/import",
        expect.objectContaining({ method: "POST" }),
      );
    });
    const importCall = fetchMock.mock.calls.find(
      ([url]) => url === "http://127.0.0.1:8000/image-classification/datasets/import",
    );
    const body = (importCall?.[1] as RequestInit).body as FormData;
    expect(body.get("mode")).toBe("classification");
    expect(body.getAll("relative_paths[]")).toEqual(["cats/cat.jpg"]);
    expect(body.getAll("files")).toHaveLength(1);
    expect(await screen.findByText("cat.jpg")).toBeInTheDocument();
  });

  it("imports training images with a manually entered label", async () => {
    let hasImported = false;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (url === "http://127.0.0.1:8000/image-classification/datasets?mode=training") {
        return Promise.resolve(
          jsonResponse(
            hasImported
              ? {
                  ...emptyTrainingDataset,
                  count: 1,
                  labels: [
                    {
                      label: "cat",
                      count: 1,
                      items: [
                        { file_name: "cat.jpg", relative_path: "cat.jpg", size: 3 },
                      ],
                    },
                  ],
                }
              : emptyTrainingDataset,
          ),
        );
      }

      if (
        url === "http://127.0.0.1:8000/image-classification/datasets/import" &&
        init?.method === "POST"
      ) {
        hasImported = true;
        return Promise.resolve(jsonResponse({ mode: "training", imported_count: 1 }));
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal(
      "showDirectoryPicker",
      vi.fn(() =>
        Promise.resolve(
          directoryHandle("selected", [["cat.jpg", fileHandle("cat.jpg", "cat")]]),
        ),
      ),
    );

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "训练模式" }));
    const importButton = await screen.findByRole("button", { name: "添加训练图片" });
    expect(importButton).toBeDisabled();
    expect(screen.getByText("先添加或选择 label 后再导入图片。")).toBeInTheDocument();

    await user.type(screen.getByRole("textbox", { name: "新增 label" }), "cat");
    await user.click(screen.getByRole("button", { name: "添加 label" }));
    const labelList = screen.getByLabelText("训练 label 列表");
    expect(within(labelList).getByRole("button", { name: "cat" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByText("cat 暂无图片。")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "添加训练图片" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/image-classification/datasets/import",
        expect.objectContaining({ method: "POST" }),
      );
    });
    const importCall = fetchMock.mock.calls.find(
      ([url]) => url === "http://127.0.0.1:8000/image-classification/datasets/import",
    );
    const body = (importCall?.[1] as RequestInit).body as FormData;
    expect(body.get("mode")).toBe("training");
    expect(body.get("label")).toBe("cat");
    expect(body.getAll("relative_paths[]")).toEqual(["cat.jpg"]);
    expect(await screen.findByRole("img", { name: "cat.jpg" })).toBeInTheDocument();
  });

  it("clears all training label tabs after successfully deleting the training dataset", async () => {
    let hasCleared = false;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (url === "http://127.0.0.1:8000/image-classification/datasets?mode=training") {
        if (init?.method === "DELETE") {
          hasCleared = true;
          return Promise.resolve(jsonResponse({ mode: "training", deleted_count: 3 }));
        }

        return Promise.resolve(
          jsonResponse(
            hasCleared
              ? emptyTrainingDataset
              : {
                  ...emptyTrainingDataset,
                  count: 3,
                  labels: [
                    {
                      label: "cat",
                      count: 2,
                      items: [
                        { file_name: "cat-1.jpg", relative_path: "cat-1.jpg", size: 3 },
                      ],
                    },
                    {
                      label: "dog",
                      count: 1,
                      items: [
                        { file_name: "dog-1.jpg", relative_path: "dog-1.jpg", size: 5 },
                      ],
                    },
                  ],
                },
          ),
        );
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "训练模式" }));
    const labelList = await screen.findByLabelText("训练 label 列表");
    expect(within(labelList).getByRole("button", { name: "cat" })).toBeInTheDocument();
    expect(within(labelList).getByRole("button", { name: "dog" })).toBeInTheDocument();

    await user.type(screen.getByRole("textbox", { name: "新增 label" }), "bird");
    await user.click(screen.getByRole("button", { name: "添加 label" }));
    expect(within(labelList).getByRole("button", { name: "bird" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "删除训练数据集" }));
    await user.click(screen.getByRole("button", { name: "确认删除" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/image-classification/datasets?mode=training",
        expect.objectContaining({ method: "DELETE" }),
      );
    });
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "cat" })).not.toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: "cat" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "dog" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "bird" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "添加训练图片" })).toBeDisabled();
    expect(screen.getByText("训练数据集为空。")).toBeInTheDocument();
  });

  it("keeps training label tabs when deleting the training dataset fails", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (
        url === "http://127.0.0.1:8000/image-classification/datasets?mode=training" &&
        init?.method === "DELETE"
      ) {
        return Promise.resolve(jsonResponse({ detail: "delete failed" }, 500));
      }

      if (url === "http://127.0.0.1:8000/image-classification/datasets?mode=training") {
        return Promise.resolve(
          jsonResponse({
            ...emptyTrainingDataset,
            count: 1,
            labels: [
              {
                label: "cat",
                count: 1,
                items: [
                  { file_name: "cat-1.jpg", relative_path: "cat-1.jpg", size: 3 },
                ],
              },
            ],
          }),
        );
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "训练模式" }));
    const labelList = await screen.findByLabelText("训练 label 列表");
    await user.type(screen.getByRole("textbox", { name: "新增 label" }), "bird");
    await user.click(screen.getByRole("button", { name: "添加 label" }));

    await user.click(screen.getByRole("button", { name: "删除训练数据集" }));
    await user.click(screen.getByRole("button", { name: "确认删除" }));

    expect(await screen.findByText("delete failed")).toBeInTheDocument();
    expect(within(labelList).getByRole("button", { name: "cat" })).toBeInTheDocument();
    expect(within(labelList).getByRole("button", { name: "bird" })).toBeInTheDocument();
  });

  it("clears the selected dataset after confirmation", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (
        url ===
          "http://127.0.0.1:8000/image-classification/datasets?mode=classification" &&
        init?.method === "DELETE"
      ) {
        return Promise.resolve(
          jsonResponse({ mode: "classification", deleted_count: 1 }),
        );
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "删除分类数据集" }));
    expect(
      screen.getByRole("dialog", { name: "确认删除分类数据集" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "确认删除" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/image-classification/datasets?mode=classification",
        expect.objectContaining({ method: "DELETE" }),
      );
    });
  });

  it("shows settings from backend without saving directory overrides", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      if (url === "http://127.0.0.1:8000/settings" && init?.method !== "PUT") {
        return Promise.resolve(jsonResponse({ ...defaultSettings, device: "cpu" }));
      }

      return defaultFetchMock(input);
    });
    vi.stubGlobal("fetch", fetchMock);
    const directoryPicker = vi.fn();
    vi.stubGlobal("showDirectoryPicker", directoryPicker);

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "设置" }));
    expect(screen.getByRole("combobox", { name: "硬件设备" })).toHaveValue("cpu");
    expect(screen.getByRole("combobox", { name: "硬件设备" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "关闭" }));

    expect(await screen.findByText(defaultNebulaDirectory)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /更多设置/ }));
    expect(screen.queryByText("模型目录")).not.toBeInTheDocument();
    expect(directoryPicker).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalledWith(
      "http://127.0.0.1:8000/settings",
      expect.objectContaining({ method: "PUT" }),
    );
    expect(screen.getByText("全局设备：CPU")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalledWith(
      "http://127.0.0.1:8000/settings/select-directory",
      expect.anything(),
    );
  });

  it("keeps task-local directory controls out of the panel", async () => {
    render(<App />);

    expect(await screen.findByText(defaultNebulaDirectory)).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "选择工作目录" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "选择数据集目录" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "选择输出目录" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("工作目录")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /更多设置/ }));

    expect(screen.queryByText("模型目录")).not.toBeInTheDocument();
    expect(screen.queryByText("Checkpoint目录")).not.toBeInTheDocument();
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
    expect(screen.getByRole("button", { name: "添加训练图片" })).toBeInTheDocument();
    expect(screen.queryByText("输出目录")).not.toBeInTheDocument();
    expect(screen.queryByText("模型目录")).not.toBeInTheDocument();
    expect(screen.queryByText("Checkpoint目录")).not.toBeInTheDocument();
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
      dataset_directory: trainingDatasetDirectory,
      output_directory: trainingOutputDirectory,
      checkpoint_directory: trainingCheckpointDirectory,
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

  it("allows HF training without explicit download controls", async () => {
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
    expect(
      screen.queryByRole("checkbox", { name: "允许显式下载模型" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始训练" })).toBeEnabled();
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

    expect(await screen.findByText(defaultNebulaDirectory)).toBeInTheDocument();
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

    expect(await screen.findByText(defaultNebulaDirectory)).toBeInTheDocument();
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

    expect(screen.getByText("训练轮次：3 / 50")).toBeInTheDocument();
    const trainingCancelButton = screen.getByRole("button", {
      name: /终止训练\s+32\/128/,
    });
    expect(trainingCancelButton).toBeInTheDocument();
    expect(trainingCancelButton).toHaveClass("justify-center");
    expect(trainingCancelButton).not.toHaveClass("grid-cols-[1fr_auto_1fr]");
    expect(trainingCancelButton).not.toHaveTextContent("50%");
    expect(trainingCancelButton).not.toHaveTextContent("4/8");
    expect(trainingCancelButton.querySelector("svg")).toHaveClass("fill-current");
    expect(
      screen.queryByRole("progressbar", { name: "训练当前轮次进度" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("当前训练轮次进度 · 1 / 2")).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "分类模式" }));

    expect(screen.queryByRole("button", { name: "开始分类" })).not.toBeInTheDocument();
    expect(screen.getByText("训练轮次：3 / 50")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /终止训练\s+32\/128/ }),
    ).toBeInTheDocument();
    expect(screen.queryByText("训练运行中，分类已锁定")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /终止训练\s+32\/128/ }));
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
    expect(screen.getByRole("button", { name: /终止训练\s+32\/128/ })).toBeDisabled();
    expect(
      screen.queryByRole("dialog", { name: "确认终止训练任务" }),
    ).not.toBeInTheDocument();
  });

  it("removes task status and inline status hint areas from the panel body", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText(defaultNebulaDirectory)).toBeInTheDocument();
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

  if (
    url === "http://127.0.0.1:8000/image-classification/datasets?mode=classification"
  ) {
    return Promise.resolve(jsonResponse(emptyClassificationDataset));
  }

  if (url === "http://127.0.0.1:8000/image-classification/datasets?mode=training") {
    return Promise.resolve(jsonResponse(emptyTrainingDataset));
  }

  if (url === "http://127.0.0.1:8000/settings/select-directory") {
    return Promise.resolve(jsonResponse({ selected_directory: null }));
  }

  return Promise.resolve(jsonResponse({}, 404));
}

function directoryHandle(
  name: string,
  entries: Array<[string, BrowserDirectoryHandle | BrowserFileHandle]>,
): BrowserDirectoryHandle {
  return {
    kind: "directory",
    name,
    entries: async function* () {
      for (const entry of entries) {
        yield entry;
      }
    },
  };
}

function fileHandle(name: string, content: string): BrowserFileHandle {
  return {
    kind: "file",
    name,
    getFile: async () => new File([content], name, { type: "image/jpeg" }),
  };
}
