import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("./features/galaxy/GalaxyCanvas", () => ({
  GalaxyCanvas: () => <section aria-label="Galaxy AI 星系任务画布" />,
}));

import { App } from "./App";

const runningTrainingRun = {
  run_id: "run-training-1",
  run_type: "image_classification_training",
  status: "running",
  request: {
    base_model_ref: "~/Models/resnet50",
    dataset_directory: "~/Datasets/NebulaSorter",
    output_directory: "~/Projects/NebulaSorter/output",
    checkpoint_directory: "~/Projects/NebulaSorter/checkpoints",
    epochs: 3,
    batch_size: 8,
    learning_rate: 0.00005,
    seed: 42,
    device: "auto",
    allow_download: false,
  },
  hardware_backend: "auto",
  model_ref: "~/Models/resnet50",
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

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Galaxy AI home", () => {
  it("selects the Nebula Sorter task planet by default", () => {
    render(<App />);

    expect(screen.getAllByText("星云分拣站")[0]).toBeInTheDocument();
    expect(screen.getAllByText("批量图片分类")[0]).toBeInTheDocument();
    expect(screen.getByText("本地运行")).toBeInTheDocument();
  });

  it("switches between task-local starmap and professional modes", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(screen.getByText(/快速启动批量图片分类任务/)).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "专业模式" }));

    expect(screen.getByText("分类参数")).toBeInTheDocument();
    expect(screen.getByText("运行日志")).toBeInTheDocument();
  });

  it("keeps view mode tabs independent from classification and training tabs", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "训练模式" }));

    expect(screen.getByText("训练摘要")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "星图模式" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("tab", { name: "训练模式" })).toHaveAttribute(
      "aria-selected",
      "true",
    );

    await user.click(screen.getByRole("tab", { name: "专业模式" }));

    expect(screen.getByText("本地训练配置")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "训练模式" })).toHaveAttribute(
      "aria-selected",
      "true",
    );

    await user.click(screen.getByRole("tab", { name: "分类模式" }));

    expect(screen.getByText("分类参数")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "专业模式" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("updates local task status when starting classification", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /开始分类/i }));

    expect(screen.getAllByText("运行中")[0]).toBeInTheDocument();
    expect(screen.getByText("正在扫描输入文件夹")).toBeInTheDocument();
  });

  it("posts training requests to the backend and renders run status with logs", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ run_id: "run-training-1", status: "queued" }, 202),
      )
      .mockResolvedValueOnce(jsonResponse(runningTrainingRun))
      .mockResolvedValueOnce(
        jsonResponse({ logs: ["训练任务已排队", "正在加载数据集"] }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "专业模式" }));
    await user.click(screen.getByRole("tab", { name: "训练模式" }));
    await user.click(screen.getByRole("button", { name: "开始训练" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/image-classification/training",
        expect.objectContaining({ method: "POST" }),
      );
    });

    const trainingRequest = JSON.parse(
      String((fetchMock.mock.calls[0][1] as RequestInit).body),
    );
    expect(trainingRequest).toEqual({
      base_model_ref: "~/Models/resnet50",
      dataset_directory: "~/Datasets/NebulaSorter",
      output_directory: "~/Projects/NebulaSorter/output",
      checkpoint_directory: "~/Projects/NebulaSorter/checkpoints",
      epochs: 3,
      batch_size: 8,
      learning_rate: 0.00005,
      seed: 42,
      device: "auto",
      allow_download: false,
    });

    expect(await screen.findByText("run-training-1")).toBeInTheDocument();
    expect(screen.getByText("正在加载数据集")).toBeInTheDocument();
    expect(screen.getAllByText("运行中")[0]).toBeInTheDocument();
  });

  it("cancels an active training run through the backend", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({ run_id: "run-training-1", status: "queued" }, 202),
      )
      .mockResolvedValueOnce(jsonResponse(runningTrainingRun))
      .mockResolvedValueOnce(jsonResponse({ logs: ["正在训练分类头"] }))
      .mockResolvedValueOnce(
        jsonResponse({ run_id: "run-training-1", status: "cancelling" }),
      )
      .mockResolvedValueOnce(
        jsonResponse({ ...runningTrainingRun, status: "cancelling" }),
      )
      .mockResolvedValueOnce(
        jsonResponse({ logs: ["正在训练分类头", "正在取消运行"] }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("tab", { name: "专业模式" }));
    await user.click(screen.getByRole("tab", { name: "训练模式" }));
    await user.click(screen.getByRole("button", { name: "开始训练" }));
    await screen.findByText("run-training-1");

    await user.click(screen.getByRole("button", { name: "取消运行" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/runs/run-training-1/cancel",
        expect.objectContaining({ method: "POST" }),
      );
    });
    expect(await screen.findByText("正在取消运行")).toBeInTheDocument();
  });
});

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}
