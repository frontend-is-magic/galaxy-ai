import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Database,
  Folder,
  Gauge,
  Play,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Square,
} from "lucide-react";
import type { ReactNode } from "react";

import type { RunRecord, RunStatus, TrainingRequest } from "./api";
import type {
  BatchImageClassificationTask,
  TaskCapabilityMode,
  TaskStatus,
  TaskViewMode,
} from "./types";

type TaskInspectorProps = {
  task: BatchImageClassificationTask;
  viewMode: TaskViewMode;
  capabilityMode: TaskCapabilityMode;
  trainingRequest: TrainingRequest;
  trainingRun: RunRecord | null;
  trainingLogs: string[];
  trainingError: string | null;
  isTrainingSubmitting: boolean;
  onViewModeChange: (mode: TaskViewMode) => void;
  onCapabilityModeChange: (mode: TaskCapabilityMode) => void;
  onTrainingRequestChange: (update: Partial<TrainingRequest>) => void;
  onStartClassification: () => void;
  onStartTraining: () => void;
  onCancelTraining: () => void;
};

const taskStatusLabel: Record<TaskStatus, string> = {
  idle: "就绪",
  running: "运行中",
  completed: "已完成",
  error: "错误",
};

const runStatusLabel: Record<RunStatus, string> = {
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  error: "错误",
  cancelling: "取消中",
  cancelled: "已取消",
  interrupted: "已中断",
};

const unfinishedRunStatuses = new Set<RunStatus>(["queued", "running", "cancelling"]);

export function TaskInspector({
  task,
  viewMode,
  capabilityMode,
  trainingRequest,
  trainingRun,
  trainingLogs,
  trainingError,
  isTrainingSubmitting,
  onViewModeChange,
  onCapabilityModeChange,
  onTrainingRequestChange,
  onStartClassification,
  onStartTraining,
  onCancelTraining,
}: TaskInspectorProps) {
  const progress =
    task.total_images === 0
      ? 0
      : Math.round((task.processed_images / task.total_images) * 100);
  const isTraining = capabilityMode === "training";
  const canCancelTraining =
    Boolean(trainingRun) &&
    trainingRun !== null &&
    unfinishedRunStatuses.has(trainingRun.status);
  const isTrainingStartDisabled =
    isTrainingSubmitting || (isTraining && canCancelTraining);

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

          <SegmentedTabs
            className="mt-5"
            label="展示模式"
            tabs={[
              {
                id: "starmap",
                label: "星图模式",
                icon: <Sparkles className="size-4" aria-hidden="true" />,
                active: viewMode === "starmap",
                onClick: () => onViewModeChange("starmap"),
              },
              {
                id: "professional",
                label: "专业模式",
                icon: <SlidersHorizontal className="size-4" aria-hidden="true" />,
                active: viewMode === "professional",
                onClick: () => onViewModeChange("professional"),
              },
            ]}
          />

          <SegmentedTabs
            className="mt-3"
            label="任务能力"
            tabs={[
              {
                id: "classification",
                label: "分类模式",
                icon: <Gauge className="size-4" aria-hidden="true" />,
                active: capabilityMode === "classification",
                onClick: () => onCapabilityModeChange("classification"),
              },
              {
                id: "training",
                label: "训练模式",
                icon: <Database className="size-4" aria-hidden="true" />,
                active: capabilityMode === "training",
                onClick: () => onCapabilityModeChange("training"),
              },
            ]}
          />
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-5 sm:p-6">
          {viewMode === "starmap" && capabilityMode === "classification" ? (
            <ClassificationStarmap task={task} progress={progress} />
          ) : null}

          {viewMode === "starmap" && capabilityMode === "training" ? (
            <TrainingStarmap
              trainingRequest={trainingRequest}
              trainingRun={trainingRun}
              trainingLogs={trainingLogs}
              trainingError={trainingError}
            />
          ) : null}

          {viewMode === "professional" && capabilityMode === "classification" ? (
            <ClassificationProfessional task={task} progress={progress} />
          ) : null}

          {viewMode === "professional" && capabilityMode === "training" ? (
            <TrainingProfessional
              trainingRequest={trainingRequest}
              trainingRun={trainingRun}
              trainingLogs={trainingLogs}
              trainingError={trainingError}
              onTrainingRequestChange={onTrainingRequestChange}
              onCancelTraining={onCancelTraining}
              canCancelTraining={canCancelTraining}
            />
          ) : null}
        </div>

        <div className="border-t border-white/10 p-5 sm:p-6">
          <button
            className="inline-flex w-full items-center justify-center gap-3 rounded-lg bg-cyan-400 px-5 py-4 text-base font-semibold text-cyan-950 shadow-[0_0_34px_rgba(34,211,238,0.32)] transition hover:bg-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-100 disabled:cursor-not-allowed disabled:opacity-60"
            type="button"
            onClick={isTraining ? onStartTraining : onStartClassification}
            disabled={isTrainingStartDisabled}
          >
            <Play className="size-5 fill-current" aria-hidden="true" />
            {isTraining ? "开始训练" : "开始分类"}
          </button>
        </div>
      </div>
    </aside>
  );
}

function ClassificationStarmap({
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

      <TaskBasics task={task} />
      <TaskProgress task={task} progress={progress} />
      <StatusLine task={task} />
    </div>
  );
}

function TrainingStarmap({
  trainingRequest,
  trainingRun,
  trainingLogs,
  trainingError,
}: {
  trainingRequest: TrainingRequest;
  trainingRun: RunRecord | null;
  trainingLogs: string[];
  trainingError: string | null;
}) {
  return (
    <div className="space-y-5">
      <section className="panel-section">
        <div className="section-title">
          <Database className="size-4 text-cyan-300" aria-hidden="true" />
          训练摘要
        </div>
        <p className="mt-3 text-sm leading-6 text-slate-300">
          本地微调图片分类模型。完整参数在专业模式中配置，数据集不上传。
          切换到专业模式可调整路径、超参和下载开关。
        </p>
        <div className="mt-4 grid gap-3 text-sm">
          <InfoRow label="基础模型" value={trainingRequest.base_model_ref} />
          <InfoRow label="数据集" value={trainingRequest.dataset_directory} />
          <InfoRow label="轮次" value={`${trainingRequest.epochs} epochs`} />
        </div>
      </section>
      <RunStatusPanel
        run={trainingRun}
        logs={trainingLogs}
        error={trainingError}
        compact
      />
    </div>
  );
}

function ClassificationProfessional({
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

function TrainingProfessional({
  trainingRequest,
  trainingRun,
  trainingLogs,
  trainingError,
  onTrainingRequestChange,
  onCancelTraining,
  canCancelTraining,
}: {
  trainingRequest: TrainingRequest;
  trainingRun: RunRecord | null;
  trainingLogs: string[];
  trainingError: string | null;
  onTrainingRequestChange: (update: Partial<TrainingRequest>) => void;
  onCancelTraining: () => void;
  canCancelTraining: boolean;
}) {
  return (
    <div className="space-y-5">
      <section className="panel-section">
        <div className="flex items-center justify-between gap-3">
          <div className="section-title">
            <Database className="size-4 text-cyan-300" aria-hidden="true" />
            本地训练配置
          </div>
          <div className="inline-flex items-center gap-2 text-xs font-medium text-emerald-300">
            <ShieldCheck className="size-4" aria-hidden="true" />
            数据集不上传
          </div>
        </div>

        <p className="mt-3 text-xs leading-5 text-slate-400">
          ImageFolder 数据集结构：train/&lt;label&gt;/image files。
        </p>

        <div className="mt-4 grid gap-3">
          <TextField
            label="基础模型"
            value={trainingRequest.base_model_ref}
            onChange={(value) => onTrainingRequestChange({ base_model_ref: value })}
          />
          <TextField
            label="数据集目录"
            value={trainingRequest.dataset_directory}
            onChange={(value) => onTrainingRequestChange({ dataset_directory: value })}
          />
          <TextField
            label="输出目录"
            value={trainingRequest.output_directory ?? ""}
            onChange={(value) =>
              onTrainingRequestChange({ output_directory: value || null })
            }
          />
          <TextField
            label="Checkpoint目录"
            value={trainingRequest.checkpoint_directory ?? ""}
            onChange={(value) =>
              onTrainingRequestChange({ checkpoint_directory: value || null })
            }
          />
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          <NumberField
            label="Epochs"
            min={1}
            max={1000}
            value={trainingRequest.epochs}
            onChange={(value) => onTrainingRequestChange({ epochs: value })}
          />
          <NumberField
            label="Batch size"
            min={1}
            max={256}
            value={trainingRequest.batch_size}
            onChange={(value) => onTrainingRequestChange({ batch_size: value })}
          />
          <NumberField
            label="Learning rate"
            min={0.000001}
            step={0.00001}
            value={trainingRequest.learning_rate}
            onChange={(value) => onTrainingRequestChange({ learning_rate: value })}
          />
          <NumberField
            label="Seed"
            value={trainingRequest.seed}
            onChange={(value) => onTrainingRequestChange({ seed: value })}
          />
        </div>

        <div className="mt-4 space-y-3">
          <DeviceSelector
            value={trainingRequest.device}
            onChange={(device) => onTrainingRequestChange({ device })}
          />
          <label className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-sm">
            <span>
              <span className="block font-medium text-slate-100">允许显式下载模型</span>
              <span className="text-xs text-slate-400">
                关闭时只使用本地已有模型路径。
              </span>
            </span>
            <input
              aria-label="允许显式下载模型"
              checked={trainingRequest.allow_download}
              className="size-4 accent-cyan-300"
              type="checkbox"
              onChange={(event) =>
                onTrainingRequestChange({
                  allow_download: event.currentTarget.checked,
                })
              }
            />
          </label>
        </div>
      </section>

      <RunStatusPanel
        run={trainingRun}
        logs={trainingLogs}
        error={trainingError}
        onCancel={onCancelTraining}
        canCancel={canCancelTraining}
      />
    </div>
  );
}

function TaskBasics({ task }: { task: BatchImageClassificationTask }) {
  return (
    <div className="space-y-3">
      <InfoRow
        icon={<Cpu className="size-4" aria-hidden="true" />}
        label="计算后端"
        value={task.hardware_backend.label}
        status={taskStatusLabel[task.status]}
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
  );
}

function RunStatusPanel({
  run,
  logs,
  error,
  compact = false,
  onCancel,
  canCancel = false,
}: {
  run: RunRecord | null;
  logs: string[];
  error: string | null;
  compact?: boolean;
  onCancel?: () => void;
  canCancel?: boolean;
}) {
  const progress =
    run && run.total_items > 0
      ? Math.round((run.processed_items / run.total_items) * 100)
      : 0;

  return (
    <section className="panel-section">
      <div className="flex items-center justify-between gap-3">
        <div className="section-title">
          <RotateCcw className="size-4 text-cyan-300" aria-hidden="true" />
          训练运行状态
        </div>
        {canCancel ? (
          <button
            className="inline-flex items-center gap-2 rounded-md border border-amber-300/30 bg-amber-300/10 px-3 py-1.5 text-xs font-semibold text-amber-200 transition hover:bg-amber-300/15"
            type="button"
            onClick={onCancel}
          >
            <Square className="size-3" aria-hidden="true" />
            取消运行
          </button>
        ) : null}
      </div>

      <div className="mt-4 grid gap-3 text-sm">
        <InfoRow label="run_id" value={run?.run_id ?? "尚未创建"} />
        <InfoRow label="状态" value={run ? runStatusLabel[run.status] : "就绪"} />
        <InfoRow label="输出路径" value={run?.output_path ?? "--"} />
        {!compact ? (
          <InfoRow
            label="Checkpoint"
            value={String(run?.request.checkpoint_directory ?? "--")}
          />
        ) : null}
      </div>

      <div className="mt-4 grid grid-cols-[auto_1fr_auto] items-center gap-3 text-sm text-slate-300">
        <span>
          {run?.processed_items ?? 0} / {run?.total_items ?? 0}
        </span>
        <div className="h-2 overflow-hidden rounded-full bg-slate-700/70">
          <div
            className="h-full rounded-full bg-cyan-300 transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>
        <span>{progress}%</span>
      </div>

      {error || run?.error_message ? (
        <div className="mt-4 flex items-center gap-2 rounded-lg border border-amber-300/20 bg-amber-300/10 px-3 py-2 text-sm text-amber-200">
          <AlertCircle className="size-4" aria-hidden="true" />
          {error ?? run?.error_message}
        </div>
      ) : null}

      <div className="mt-4 rounded-lg border border-white/10 bg-black/24 p-3">
        <div className="mb-2 text-xs font-semibold text-slate-300">运行日志</div>
        <div className="space-y-1 text-xs leading-5 text-slate-400">
          {logs.length > 0 ? (
            logs.slice(-4).map((line) => <p key={line}>{line}</p>)
          ) : (
            <p>等待训练运行。</p>
          )}
        </div>
      </div>
    </section>
  );
}

function SegmentedTabs({
  className,
  label,
  tabs,
}: {
  className?: string;
  label: string;
  tabs: Array<{
    id: string;
    label: string;
    icon: ReactNode;
    active: boolean;
    onClick: () => void;
  }>;
}) {
  return (
    <div
      className={`${className ?? ""} grid grid-cols-2 rounded-lg border border-white/10 bg-black/20 p-1`}
      role="tablist"
      aria-label={label}
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={`inline-flex min-w-0 items-center justify-center gap-2 rounded-md px-2 py-3 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-cyan-300/45 sm:px-3 ${
            tab.active
              ? "border border-cyan-300/40 bg-cyan-400/12 text-cyan-200 shadow-[0_0_24px_rgba(34,211,238,0.12)]"
              : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-100"
          }`}
          role="tab"
          aria-selected={tab.active}
          type="button"
          onClick={tab.onClick}
        >
          {tab.icon}
          {tab.label}
        </button>
      ))}
    </div>
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
      {taskStatusLabel[status]}
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
    <div className="grid grid-cols-[5.6rem_minmax(0,1fr)] items-center gap-3 text-sm sm:grid-cols-[6.25rem_minmax(0,1fr)]">
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

function TextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="grid grid-cols-[5.6rem_minmax(0,1fr)] items-center gap-3 text-sm sm:grid-cols-[6.25rem_minmax(0,1fr)]">
      <span className="text-slate-400">{label}</span>
      <input
        className="min-w-0 rounded-lg border border-white/10 bg-white/[0.045] px-3 py-2 text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-cyan-300/45 focus:ring-2 focus:ring-cyan-300/15"
        value={value}
        onChange={(event) => onChange(event.currentTarget.value)}
      />
    </label>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="grid gap-1.5 text-sm">
      <span className="text-slate-400">{label}</span>
      <input
        className="min-w-0 rounded-lg border border-white/10 bg-white/[0.045] px-3 py-2 text-slate-100 outline-none transition focus:border-cyan-300/45 focus:ring-2 focus:ring-cyan-300/15"
        max={max}
        min={min}
        step={step}
        type="number"
        value={value}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
    </label>
  );
}

function DeviceSelector({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="grid gap-2 text-sm">
      <span className="text-slate-400">设备</span>
      <div className="grid grid-cols-4 gap-2">
        {["auto", "cpu", "cuda", "mps"].map((device) => (
          <button
            key={device}
            className={`rounded-md border px-2 py-2 text-xs font-semibold uppercase transition ${
              value === device
                ? "border-cyan-300/45 bg-cyan-300/12 text-cyan-200"
                : "border-white/10 bg-white/[0.04] text-slate-400 hover:text-slate-100"
            }`}
            type="button"
            aria-pressed={value === device}
            onClick={() => onChange(device)}
          >
            {device}
          </button>
        ))}
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
        <span className="font-medium text-slate-100">
          {taskStatusLabel[task.status]}
        </span>
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
