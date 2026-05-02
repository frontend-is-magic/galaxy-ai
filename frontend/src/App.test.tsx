import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

vi.mock("./features/galaxy/GalaxyCanvas", () => ({
  GalaxyCanvas: () => <section aria-label="Galaxy AI 星系任务画布" />,
}));

import { App } from "./App";

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

  it("updates local task status when starting classification", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /开始分类/i }));

    expect(screen.getAllByText("运行中")[0]).toBeInTheDocument();
    expect(screen.getByText("正在扫描输入文件夹")).toBeInTheDocument();
  });
});
