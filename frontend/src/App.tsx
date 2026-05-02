import { Settings } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { GalaxyCanvas } from "./features/galaxy/GalaxyCanvas";
import { TaskInspector } from "./features/galaxy/TaskInspector";
import {
  cancelRun,
  createImageClassificationTraining,
  getRun,
  getRunLogs,
} from "./features/galaxy/api";
import type { RunRecord, RunStatus, TrainingRequest } from "./features/galaxy/api";
import { defaultTrainingRequest, nebulaSorterTask } from "./features/galaxy/taskData";
import type {
  TaskCapabilityMode,
  TaskStatus,
  TaskViewMode,
} from "./features/galaxy/types";

const unfinishedRunStatuses = new Set<RunStatus>(["queued", "running", "cancelling"]);

export function App() {
  const [viewMode, setViewMode] = useState<TaskViewMode>("starmap");
  const [capabilityMode, setCapabilityMode] =
    useState<TaskCapabilityMode>("classification");
  const [status, setStatus] = useState<TaskStatus>(nebulaSorterTask.status);
  const [statusMessage, setStatusMessage] = useState("当前无错误");
  const [trainingRequest, setTrainingRequest] =
    useState<TrainingRequest>(defaultTrainingRequest);
  const [trainingRun, setTrainingRun] = useState<RunRecord | null>(null);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [isTrainingSubmitting, setIsTrainingSubmitting] = useState(false);

  const selectedStatus = useMemo(() => {
    if (capabilityMode === "training" && trainingRun) {
      return mapRunStatusToTaskStatus(trainingRun.status);
    }

    return status;
  }, [capabilityMode, status, trainingRun]);

  const task = {
    ...nebulaSorterTask,
    status: selectedStatus,
    error_message:
      capabilityMode === "training"
        ? trainingError || trainingRun?.error_message || null
        : status === "error"
          ? "输入目录无法访问"
          : null,
    status_message:
      capabilityMode === "training"
        ? trainingError || trainingRun?.status || "训练准备就绪"
        : statusMessage,
  };

  function startClassification() {
    setStatus("running");
    setStatusMessage("正在扫描输入文件夹");
  }

  function updateTrainingRequest(update: Partial<TrainingRequest>) {
    setTrainingRequest((current) => ({ ...current, ...update }));
  }

  const refreshTrainingRun = useCallback(async (runId: string) => {
    const [run, logs] = await Promise.all([getRun(runId), getRunLogs(runId)]);
    setTrainingRun(run);
    setTrainingLogs(logs.logs);
  }, []);

  async function startTraining() {
    setIsTrainingSubmitting(true);
    setTrainingError(null);

    try {
      const created = await createImageClassificationTraining(trainingRequest);
      setTrainingRun(createPendingRun(created.run_id, created.status, trainingRequest));
      setTrainingLogs([]);
      await refreshTrainingRun(created.run_id);
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "训练请求失败。");
    } finally {
      setIsTrainingSubmitting(false);
    }
  }

  async function cancelTraining() {
    if (!trainingRun) {
      return;
    }

    setTrainingError(null);

    try {
      await cancelRun(trainingRun.run_id);
      await refreshTrainingRun(trainingRun.run_id);
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "取消运行失败。");
    }
  }

  useEffect(() => {
    if (!trainingRun || !unfinishedRunStatuses.has(trainingRun.status)) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      void refreshTrainingRun(trainingRun.run_id).catch((error: unknown) => {
        setTrainingError(error instanceof Error ? error.message : "刷新训练状态失败。");
      });
    }, 2500);

    return () => window.clearInterval(intervalId);
  }, [refreshTrainingRun, trainingRun]);

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
          viewMode={viewMode}
          capabilityMode={capabilityMode}
          task={task}
          trainingRequest={trainingRequest}
          trainingRun={trainingRun}
          trainingLogs={trainingLogs}
          trainingError={trainingError}
          isTrainingSubmitting={isTrainingSubmitting}
          onViewModeChange={setViewMode}
          onCapabilityModeChange={setCapabilityMode}
          onTrainingRequestChange={updateTrainingRequest}
          onStartClassification={startClassification}
          onStartTraining={startTraining}
          onCancelTraining={cancelTraining}
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

function mapRunStatusToTaskStatus(status: RunStatus): TaskStatus {
  if (status === "completed") {
    return "completed";
  }
  if (status === "error" || status === "cancelled" || status === "interrupted") {
    return "error";
  }
  return "running";
}

function createPendingRun(
  runId: string,
  status: RunStatus,
  request: TrainingRequest,
): RunRecord {
  const now = new Date().toISOString();
  return {
    run_id: runId,
    run_type: "image_classification_training",
    status,
    request,
    hardware_backend: request.device,
    model_ref: request.base_model_ref,
    input_path: request.dataset_directory,
    output_path: request.output_directory,
    total_items: 0,
    processed_items: 0,
    error_message: null,
    created_at: now,
    updated_at: now,
    started_at: null,
    completed_at: null,
  };
}
