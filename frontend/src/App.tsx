import { Settings } from "lucide-react";
import { useState } from "react";

import { GalaxyCanvas } from "./features/galaxy/GalaxyCanvas";
import { TaskInspector } from "./features/galaxy/TaskInspector";
import { nebulaSorterTask } from "./features/galaxy/taskData";
import type { TaskMode, TaskStatus } from "./features/galaxy/types";

export function App() {
  const [mode, setMode] = useState<TaskMode>("starmap");
  const [status, setStatus] = useState<TaskStatus>(nebulaSorterTask.status);
  const [statusMessage, setStatusMessage] = useState("当前无错误");

  const task = {
    ...nebulaSorterTask,
    status,
    error_message: status === "error" ? "输入目录无法访问" : null,
    status_message: statusMessage,
  };

  function startClassification() {
    setStatus("running");
    setStatusMessage("正在扫描输入文件夹");
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[#03060d] text-slate-100">
      <div className="relative min-h-screen">
        <GalaxyCanvas task={task} />

        <header className="pointer-events-none fixed inset-x-0 top-0 z-30 flex items-center justify-between border-b border-cyan-200/10 bg-black/35 px-4 py-3 backdrop-blur-xl sm:px-7">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid size-9 place-items-center rounded-full border border-cyan-300/25 bg-cyan-300/10 shadow-[0_0_30px_rgba(34,211,238,0.25)]">
              <span className="size-4 rounded-full bg-[conic-gradient(from_210deg,#22d3ee,#6d5dfc,#22c55e,#22d3ee)]" />
            </div>
            <h1 className="text-[1.35rem] font-semibold tracking-normal text-white sm:text-[1.7rem]">
              Galaxy AI
            </h1>
            <div className="hidden h-8 w-px bg-slate-500/35 sm:block" />
            <div className="hidden items-center gap-2 rounded-md border border-emerald-300/20 bg-emerald-400/10 px-3 py-2 text-sm font-medium text-emerald-300 shadow-[0_0_24px_rgba(16,185,129,0.12)] md:flex">
              <span className="size-2 rounded-full bg-emerald-400" />
              本地运行
            </div>
            <div className="hidden rounded-md border border-white/10 bg-white/[0.045] px-3 py-2 text-sm text-slate-300 lg:block">
              {task.hardware_backend.label}
            </div>
          </div>

          <button className="pointer-events-auto inline-flex items-center gap-2 rounded-md border border-white/12 bg-white/[0.055] px-3 py-2 text-sm font-medium text-slate-100 transition hover:border-cyan-300/40 hover:bg-cyan-300/10 focus:outline-none focus:ring-2 focus:ring-cyan-300/50">
            <Settings className="size-4" aria-hidden="true" />
            设置
          </button>
        </header>

        <TaskInspector
          mode={mode}
          task={task}
          onModeChange={setMode}
          onStart={startClassification}
        />

        <div className="fixed bottom-5 left-4 z-20 hidden items-center gap-2 rounded-lg border border-white/10 bg-black/45 p-2 shadow-2xl shadow-cyan-950/40 backdrop-blur-xl md:flex">
          <button className="toolbar-button" aria-label="平移视图">
            <span className="text-lg">⌁</span>
          </button>
          <button className="toolbar-button" aria-label="定位星球">
            <span className="size-3 rounded-full border border-cyan-200" />
          </button>
          <div className="h-8 w-px bg-white/10" />
          <button className="toolbar-button" aria-label="缩小">
            −
          </button>
          <div className="min-w-16 rounded-md bg-white/[0.06] px-3 py-2 text-center text-sm text-slate-200">
            100%
          </div>
          <button className="toolbar-button" aria-label="放大">
            +
          </button>
        </div>
      </div>
    </main>
  );
}
