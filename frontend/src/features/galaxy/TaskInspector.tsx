import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Folder,
  Gauge,
  Play,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import type { ReactNode } from "react";

import type { BatchImageClassificationTask, TaskMode, TaskStatus } from "./types";

type TaskInspectorProps = {
  task: BatchImageClassificationTask;
  mode: TaskMode;
  onModeChange: (mode: TaskMode) => void;
  onStart: () => void;
};

const statusLabel: Record<TaskStatus, string> = {
  idle: "就绪",
  running: "运行中",
  completed: "已完成",
  error: "错误",
};

export function TaskInspector({
  task,
  mode,
  onModeChange,
  onStart,
}: TaskInspectorProps) {
  const progress =
    task.total_images === 0
      ? 0
      : Math.round((task.processed_images / task.total_images) * 100);

  return (
    <aside className="task-inspector fixed inset-x-4 bottom-4 top-20 z-20 max-w-[calc(100vw-2rem)] overflow-hidden rounded-2xl border border-white/12 bg-[#07111c]/88 shadow-[0_24px_80px_rgba(0,0,0,0.58)] backdrop-blur-2xl md:inset-x-auto md:left-auto md:right-5 md:w-[31rem] md:max-w-none lg:right-6">
      <div className="flex h-full flex-col">
        <div className="border-b border-white/10 p-5 sm:p-6">
          <div className="flex items-start justify-between gap-4">
            <div className="flex min-w-0 items-center gap-4">
              <div className="grid size-14 shrink-0 place-items-center rounded-full border border-cyan-300/35 bg-[radial-gradient(circle_at_35%_30%,#67e8f9,#4338ca_45%,#12061f_78%)] shadow-[0_0_30px_rgba(34,211,238,0.34)]">
                <Sparkles className="size-6 text-cyan-100" aria-hidden="true" />
              </div>
              <div className="min-w-0">
                <h2 className="truncate text-2xl font-semibold text-white">
                  {task.name}
                </h2>
                <p className="mt-1 text-sm text-slate-300">{task.task_type}</p>
              </div>
            </div>
            <StatusPill status={task.status} />
          </div>

          <div
            className="mt-5 grid grid-cols-2 rounded-lg border border-white/10 bg-black/20 p-1"
            role="tablist"
            aria-label="星云分拣站模式"
          >
            <ModeTab
              active={mode === "starmap"}
              icon={<Sparkles className="size-4" aria-hidden="true" />}
              label="星图模式"
              onClick={() => onModeChange("starmap")}
            />
            <ModeTab
              active={mode === "professional"}
              icon={<SlidersHorizontal className="size-4" aria-hidden="true" />}
              label="专业模式"
              onClick={() => onModeChange("professional")}
            />
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-5 sm:p-6">
          {mode === "starmap" ? (
            <StarmapMode task={task} progress={progress} />
          ) : (
            <ProfessionalMode task={task} progress={progress} />
          )}
        </div>

        <div className="border-t border-white/10 p-5 sm:p-6">
          <button
            className="inline-flex w-full items-center justify-center gap-3 rounded-lg bg-cyan-400 px-5 py-4 text-base font-semibold text-cyan-950 shadow-[0_0_34px_rgba(34,211,238,0.32)] transition hover:bg-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-100"
            type="button"
            onClick={onStart}
          >
            <Play className="size-5 fill-current" aria-hidden="true" />
            开始分类
          </button>
        </div>
      </div>
    </aside>
  );
}

function StarmapMode({
  task,
  progress,
}: {
  task: BatchImageClassificationTask;
  progress: number;
}) {
  return (
    <div className="space-y-5">
      <p className="text-sm leading-6 text-slate-300">
        快速启动批量图片分类任务。路径、模型和输出位置全部保留在本机。
      </p>

      <div className="space-y-3">
        <InfoRow
          icon={<Cpu className="size-4" aria-hidden="true" />}
          label="计算后端"
          value={task.hardware_backend.label}
          status={statusLabel[task.status]}
        />
        <InfoRow label="模型" value={task.model_path} />
        <InfoRow
          icon={<Folder className="size-4" aria-hidden="true" />}
          label="工作目录"
          value={task.working_directory}
        />
        <InfoRow
          icon={<Folder className="size-4" aria-hidden="true" />}
          label="输入文件夹"
          value={task.input_directory}
        />
        <InfoRow
          icon={<Folder className="size-4" aria-hidden="true" />}
          label="输出位置"
          value={task.output_directory}
        />
      </div>

      <TaskProgress task={task} progress={progress} />
      <StatusLine task={task} />
    </div>
  );
}

function ProfessionalMode({
  task,
  progress,
}: {
  task: BatchImageClassificationTask;
  progress: number;
}) {
  return (
    <div className="space-y-5">
      <section className="panel-section">
        <div className="section-title">
          <Gauge className="size-4 text-cyan-300" aria-hidden="true" />
          分类参数
        </div>
        <div className="mt-4 grid gap-3 text-sm">
          <InfoRow label="批大小" value="32 images" />
          <InfoRow label="置信度阈值" value="0.72" />
          <InfoRow label="支持格式" value={task.supported_formats.join(", ")} />
        </div>
      </section>

      <section className="panel-section">
        <div className="section-title">
          <Cpu className="size-4 text-cyan-300" aria-hidden="true" />
          本地运行配置
        </div>
        <div className="mt-4 grid gap-3 text-sm">
          <InfoRow label="硬件后端" value={task.hardware_backend.label} />
          <InfoRow label="设备" value={task.hardware_backend.device_name} />
          <InfoRow label="模型路径" value={task.model_path} />
          <InfoRow label="工作目录" value={task.working_directory} />
          <InfoRow label="输出位置" value={task.output_directory} />
        </div>
      </section>

      <section className="panel-section">
        <div className="section-title">
          <AlertCircle className="size-4 text-cyan-300" aria-hidden="true" />
          运行日志
        </div>
        <div className="mt-4 space-y-3 text-sm text-slate-300">
          <TaskProgress task={task} progress={progress} />
          <StatusLine task={task} />
          <div className="rounded-lg border border-white/10 bg-black/24 p-3 text-xs leading-5 text-slate-400">
            [{new Date().toLocaleTimeString("zh-CN", { hour12: false })}]{" "}
            {task.status_message}
          </div>
        </div>
      </section>
    </div>
  );
}

function ModeTab({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      className={`inline-flex min-w-0 items-center justify-center gap-2 rounded-md px-2 py-3 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-cyan-300/45 sm:px-3 ${
        active
          ? "border border-cyan-300/40 bg-cyan-400/12 text-cyan-200 shadow-[0_0_24px_rgba(34,211,238,0.12)]"
          : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-100"
      }`}
      role="tab"
      aria-selected={active}
      type="button"
      onClick={onClick}
    >
      {icon}
      {label}
    </button>
  );
}

function StatusPill({ status }: { status: TaskStatus }) {
  const isError = status === "error";
  return (
    <div
      className={`inline-flex shrink-0 items-center gap-2 rounded-md px-2.5 py-1.5 text-sm ${
        isError
          ? "bg-amber-400/10 text-amber-300"
          : "bg-emerald-400/10 text-emerald-300"
      }`}
    >
      <span
        className={`size-2 rounded-full ${isError ? "bg-amber-300" : "bg-emerald-400"}`}
      />
      {statusLabel[status]}
    </div>
  );
}

function InfoRow({
  icon,
  label,
  value,
  status,
}: {
  icon?: ReactNode;
  label: string;
  value: string;
  status?: string;
}) {
  return (
    <div className="grid grid-cols-[4.9rem_minmax(0,1fr)] items-center gap-3 text-sm sm:grid-cols-[5.25rem_minmax(0,1fr)]">
      <span className="text-slate-400">{label}</span>
      <div className="flex min-w-0 items-center gap-2 rounded-lg border border-white/10 bg-white/[0.045] px-3 py-2 text-slate-200">
        {icon ? <span className="text-slate-400">{icon}</span> : null}
        <span className="min-w-0 flex-1 truncate">{value}</span>
        {status ? (
          <span className="inline-flex items-center gap-1.5 text-emerald-300">
            <span className="size-1.5 rounded-full bg-emerald-400" />
            {status}
          </span>
        ) : null}
      </div>
    </div>
  );
}

function TaskProgress({
  task,
  progress,
}: {
  task: BatchImageClassificationTask;
  progress: number;
}) {
  return (
    <div className="space-y-3 border-t border-white/10 pt-5">
      <div className="flex items-center justify-between text-sm">
        <span className="text-slate-400">任务状态</span>
        <span className="font-medium text-slate-100">{statusLabel[task.status]}</span>
      </div>
      <div className="grid grid-cols-[auto_1fr_auto] items-center gap-3 text-sm text-slate-300">
        <span>
          {task.processed_images.toLocaleString("en-US")} /{" "}
          {task.total_images.toLocaleString("en-US")}
        </span>
        <div className="h-2 overflow-hidden rounded-full bg-slate-700/70">
          <div
            className="h-full rounded-full bg-cyan-300 transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>
        <span>{progress}%</span>
      </div>
    </div>
  );
}

function StatusLine({ task }: { task: BatchImageClassificationTask }) {
  const hasError = Boolean(task.error_message);
  return (
    <div className="flex items-center gap-2 border-t border-white/10 pt-5 text-sm">
      {hasError ? (
        <AlertCircle className="size-4 text-amber-300" aria-hidden="true" />
      ) : (
        <CheckCircle2 className="size-4 text-emerald-300" aria-hidden="true" />
      )}
      <span className="text-slate-400">错误信息</span>
      <span className={hasError ? "text-amber-300" : "text-slate-200"}>
        {task.error_message ?? task.status_message}
      </span>
    </div>
  );
}
