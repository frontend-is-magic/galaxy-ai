import {
  AlertCircle,
  Database,
  FolderOpen,
  FolderPlus,
  Gauge,
  Play,
  RotateCcw,
  Trash2,
  SlidersHorizontal,
  Sparkles,
  Square,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";

import type {
  DatasetMode,
  DatasetPreviewResponse,
  ModelOptionsResponse,
  RunRecord,
  RunStatus,
  TrainingRequest,
} from "./api";
import { datasetImageUrl } from "./api";
import type {
  BatchImageClassificationTask,
  TaskCapabilityMode,
  TaskStatus,
} from "./types";

type TaskInspectorProps = {
  task: BatchImageClassificationTask;
  capabilityMode: TaskCapabilityMode;
  isMoreSettingsOpen: boolean;
  trainingRequest: TrainingRequest;
  classificationRun: RunRecord | null;
  classificationLogs: string[];
  classificationError: string | null;
  trainingRun: RunRecord | null;
  trainingLogs: string[];
  trainingError: string | null;
  modelOptions: ModelOptionsResponse | null;
  modelOptionsError: string | null;
  classificationDataset: DatasetPreviewResponse | null;
  trainingDataset: DatasetPreviewResponse | null;
  datasetError: string | null;
  isImportingDataset: boolean;
  canStartTask: boolean;
  startBlockedReason: string | null;
  activeTaskMode: TaskCapabilityMode | null;
  onCapabilityModeChange: (mode: TaskCapabilityMode) => void;
  onMoreSettingsToggle: () => void;
  onTrainingRequestChange: (update: Partial<TrainingRequest>) => void;
  onTaskModelChange: (modelRef: string) => void;
  onStartClassification: (batchSize: number) => void;
  onStartTraining: () => void;
  onCancelClassification: () => void;
  onCancelTraining: () => void;
  onOpenClassificationOutput: (runId: string) => void;
  onImportDataset: (mode: DatasetMode, label?: string) => void;
  onClearDataset: (mode: DatasetMode) => Promise<boolean>;
};

const taskStatusLabel: Record<TaskStatus, string> = {
  idle: "就绪",
  running: "运行中",
  completed: "已完成",
  error: "错误",
};

const cancellableRunStatuses = new Set<RunStatus>(["queued", "running"]);
const batchSizeOptions = [8, 16, 32, 64];

export function TaskInspector({
  task,
  capabilityMode,
  isMoreSettingsOpen,
  trainingRequest,
  classificationRun,
  classificationLogs,
  classificationError,
  trainingRun,
  trainingLogs,
  trainingError,
  modelOptions,
  modelOptionsError,
  classificationDataset,
  trainingDataset,
  datasetError,
  isImportingDataset,
  canStartTask,
  startBlockedReason,
  activeTaskMode,
  onCapabilityModeChange,
  onMoreSettingsToggle,
  onTrainingRequestChange,
  onTaskModelChange,
  onStartClassification,
  onStartTraining,
  onCancelClassification,
  onCancelTraining,
  onOpenClassificationOutput,
  onImportDataset,
  onClearDataset,
}: TaskInspectorProps) {
  const isTraining = capabilityMode === "training";
  const [classificationBatchSize, setClassificationBatchSize] = useState(32);
  const [classificationConfidence, setClassificationConfidence] = useState(0.72);
  const [pendingCancelMode, setPendingCancelMode] = useState<TaskCapabilityMode | null>(
    null,
  );
  const activeRun =
    activeTaskMode === "classification"
      ? classificationRun
      : activeTaskMode === "training"
        ? trainingRun
        : null;

  function confirmCancel() {
    if (pendingCancelMode === "training") {
      onCancelTraining();
    }
    if (pendingCancelMode === "classification") {
      onCancelClassification();
    }
    setPendingCancelMode(null);
  }

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
          {capabilityMode === "classification" ? (
            <ClassificationSummary
              modelOptions={modelOptions}
              modelOptionsError={modelOptionsError}
              selectedModel={trainingRequest.base_model_ref}
              classificationRun={classificationRun}
              classificationLogs={classificationLogs}
              classificationError={classificationError}
              dataset={classificationDataset}
              datasetError={datasetError}
              isImportingDataset={isImportingDataset}
              startBlockedReason={startBlockedReason}
              onTaskModelChange={onTaskModelChange}
              onOpenOutput={onOpenClassificationOutput}
              onImportDataset={() => onImportDataset("classification")}
              onClearDataset={() => onClearDataset("classification")}
            />
          ) : null}

          {capabilityMode === "training" ? (
            <TrainingSummary
              trainingRequest={trainingRequest}
              trainingRun={trainingRun}
              trainingLogs={trainingLogs}
              trainingError={trainingError}
              modelOptions={modelOptions}
              modelOptionsError={modelOptionsError}
              dataset={trainingDataset}
              datasetError={datasetError}
              isImportingDataset={isImportingDataset}
              onTaskModelChange={onTaskModelChange}
              onTrainingRequestChange={onTrainingRequestChange}
              onImportDataset={(label) => onImportDataset("training", label)}
              onClearDataset={() => onClearDataset("training")}
              startBlockedReason={startBlockedReason}
            />
          ) : null}

          {capabilityMode === "classification" ? (
            <MoreSettingsAccordion
              isOpen={isMoreSettingsOpen}
              onToggle={onMoreSettingsToggle}
            >
              <ClassificationAdvanced
                task={task}
                batchSize={classificationBatchSize}
                confidence={classificationConfidence}
                onBatchSizeChange={setClassificationBatchSize}
                onConfidenceChange={setClassificationConfidence}
              />
            </MoreSettingsAccordion>
          ) : null}
        </div>

        <div className="border-t border-white/10 p-5 sm:p-6">
          <TaskActionButton
            activeMode={activeTaskMode}
            currentMode={capabilityMode}
            disabled={!canStartTask}
            run={activeRun}
            onStart={
              isTraining
                ? onStartTraining
                : () => onStartClassification(classificationBatchSize)
            }
            onRequestCancel={(mode) => setPendingCancelMode(mode)}
          />
        </div>
      </div>
      {pendingCancelMode ? (
        <CancelRunDialog
          mode={pendingCancelMode}
          onClose={() => setPendingCancelMode(null)}
          onConfirm={confirmCancel}
        />
      ) : null}
    </aside>
  );
}

function ClassificationSummary({
  modelOptions,
  modelOptionsError,
  selectedModel,
  classificationRun,
  classificationLogs,
  classificationError,
  dataset,
  datasetError,
  isImportingDataset,
  startBlockedReason,
  onTaskModelChange,
  onOpenOutput,
  onImportDataset,
  onClearDataset,
}: {
  modelOptions: ModelOptionsResponse | null;
  modelOptionsError: string | null;
  selectedModel: string;
  classificationRun: RunRecord | null;
  classificationLogs: string[];
  classificationError: string | null;
  dataset: DatasetPreviewResponse | null;
  datasetError: string | null;
  isImportingDataset: boolean;
  startBlockedReason: string | null;
  onTaskModelChange: (modelRef: string) => void;
  onOpenOutput: (runId: string) => void;
  onImportDataset: () => void;
  onClearDataset: () => Promise<boolean>;
}) {
  const canOpenOutput =
    classificationRun?.status === "completed" && Boolean(classificationRun.output_path);

  return (
    <div className="space-y-5">
      <p className="text-sm leading-6 text-slate-300">
        快速启动批量图片分类任务。模型选择、数据集预览和运行日志全部保留在本机。
      </p>

      <TaskSummaryControls
        modelOptions={modelOptions}
        modelOptionsError={modelOptionsError}
        selectedModel={selectedModel}
        onModelChange={onTaskModelChange}
      />
      {startBlockedReason ? (
        <div className="rounded-lg border border-amber-300/20 bg-amber-300/10 px-3 py-2 text-sm text-amber-200">
          {startBlockedReason}
        </div>
      ) : null}
      <DatasetPreviewPanel
        mode="classification"
        dataset={dataset}
        error={datasetError}
        isImporting={isImportingDataset}
        onImport={onImportDataset}
        onClear={onClearDataset}
      />
      <RunStatusPanel
        title="分类运行日志"
        emptyLogMessage="等待分类运行。"
        run={classificationRun}
        logs={classificationLogs}
        error={classificationError}
        action={
          canOpenOutput && classificationRun ? (
            <button
              className="inline-flex items-center gap-2 rounded-lg border border-cyan-300/25 bg-cyan-300/10 px-3 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-300/16"
              type="button"
              onClick={() => onOpenOutput(classificationRun.run_id)}
            >
              <FolderOpen className="size-4" aria-hidden="true" />
              查看结果
            </button>
          ) : null
        }
      />
    </div>
  );
}

function TrainingSummary({
  trainingRequest,
  trainingRun,
  trainingLogs,
  trainingError,
  modelOptions,
  modelOptionsError,
  dataset,
  datasetError,
  isImportingDataset,
  onTaskModelChange,
  onTrainingRequestChange,
  onImportDataset,
  onClearDataset,
  startBlockedReason,
}: {
  trainingRequest: TrainingRequest;
  trainingRun: RunRecord | null;
  trainingLogs: string[];
  trainingError: string | null;
  modelOptions: ModelOptionsResponse | null;
  modelOptionsError: string | null;
  dataset: DatasetPreviewResponse | null;
  datasetError: string | null;
  isImportingDataset: boolean;
  onTaskModelChange: (modelRef: string) => void;
  onTrainingRequestChange: (update: Partial<TrainingRequest>) => void;
  onImportDataset: (label: string) => void;
  onClearDataset: () => Promise<boolean>;
  startBlockedReason: string | null;
}) {
  const datasetLabels = dataset?.labels.map((group) => group.label) ?? [];
  const datasetLabelKey = datasetLabels.join("\u0000");
  const [localLabels, setLocalLabels] = useState<string[]>([]);
  const [selectedLabel, setSelectedLabel] = useState("");
  const [newLabel, setNewLabel] = useState("");
  const selectableLabels = useMemo(
    () => [
      ...datasetLabels,
      ...localLabels.filter((label) => !datasetLabels.includes(label)),
    ],
    // datasetLabelKey keeps this memo stable across equivalent backend refreshes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [datasetLabelKey, localLabels],
  );
  const activeLabel =
    selectedLabel && selectableLabels.includes(selectedLabel)
      ? selectedLabel
      : selectableLabels[0] || "";

  useEffect(() => {
    setLocalLabels((current) =>
      current.filter((label) => !datasetLabels.includes(label)),
    );
    if (selectedLabel && !selectableLabels.includes(selectedLabel)) {
      setSelectedLabel("");
    }
    // datasetLabelKey intentionally represents the backend label set.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetLabelKey]);

  function addLabel() {
    const normalizedLabel = newLabel.trim();
    if (!normalizedLabel) {
      return;
    }
    setLocalLabels((current) =>
      selectableLabels.includes(normalizedLabel)
        ? current
        : [...current, normalizedLabel],
    );
    setSelectedLabel(normalizedLabel);
    setNewLabel("");
  }

  async function clearTrainingDataset() {
    const didClear = await onClearDataset();
    if (didClear) {
      setLocalLabels([]);
      setSelectedLabel("");
      setNewLabel("");
    }
    return didClear;
  }

  return (
    <div className="space-y-5">
      <section className="panel-section">
        <div className="section-title">
          <Database className="size-4 text-cyan-300" aria-hidden="true" />
          训练摘要
        </div>
        <p className="mt-3 text-sm leading-6 text-slate-300">
          本地微调图片分类模型。超参、数据集、日志和取消控制都集中在这里。
          数据集不会上传。
        </p>
        <div className="mt-4 grid gap-3 text-sm">
          <TaskSummaryControls
            modelOptions={modelOptions}
            modelOptionsError={modelOptionsError}
            selectedModel={trainingRequest.base_model_ref}
            onModelChange={onTaskModelChange}
          />
        </div>
      </section>
      <DatasetPreviewPanel
        mode="training"
        dataset={dataset}
        error={datasetError}
        isImporting={isImportingDataset}
        selectedLabel={activeLabel}
        labelOptions={selectableLabels}
        newLabel={newLabel}
        onNewLabelChange={setNewLabel}
        onAddLabel={addLabel}
        onSelectedLabelChange={setSelectedLabel}
        onImport={() => onImportDataset(activeLabel)}
        onClear={clearTrainingDataset}
      />
      <section className="panel-section">
        <div className="section-title">
          <SlidersHorizontal className="size-4 text-cyan-300" aria-hidden="true" />
          训练设置
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <NumberField
            label="训练轮次"
            min={1}
            max={1000}
            value={trainingRequest.epochs}
            onChange={(value) => onTrainingRequestChange({ epochs: value })}
          />
          <BatchSizeSelect
            label="批大小"
            value={trainingRequest.batch_size}
            onChange={(value) => onTrainingRequestChange({ batch_size: value })}
          />
          <NumberField
            label="学习率"
            min={0.000001}
            step={0.00001}
            value={trainingRequest.learning_rate}
            onChange={(value) => onTrainingRequestChange({ learning_rate: value })}
          />
          <SeedField
            label="随机种子"
            value={trainingRequest.seed}
            enabled={trainingRequest.use_seed}
            onChange={(value) => onTrainingRequestChange({ seed: value })}
            onEnabledChange={(enabled) =>
              onTrainingRequestChange({ use_seed: enabled })
            }
          />
        </div>
        {startBlockedReason ? (
          <div className="mt-4 rounded-lg border border-amber-300/20 bg-amber-300/10 px-3 py-2 text-sm text-amber-200">
            {startBlockedReason}
          </div>
        ) : null}
      </section>
      <RunStatusPanel
        title="训练运行日志"
        emptyLogMessage="等待训练运行。"
        run={trainingRun}
        logs={trainingLogs}
        error={trainingError}
      />
    </div>
  );
}

function ClassificationAdvanced({
  task,
  batchSize,
  confidence,
  onBatchSizeChange,
  onConfidenceChange,
}: {
  task: BatchImageClassificationTask;
  batchSize: number;
  confidence: number;
  onBatchSizeChange: (value: number) => void;
  onConfidenceChange: (value: number) => void;
}) {
  return (
    <div className="grid gap-4 text-sm">
      <BatchSizeSelect label="批大小" value={batchSize} onChange={onBatchSizeChange} />
      <NumberField
        label="置信度阈值"
        min={0}
        max={1}
        step={0.01}
        value={confidence}
        onChange={onConfidenceChange}
      />
      <InfoRow label="支持格式" value={task.supported_formats.join(", ")} />
      <div className="rounded-lg border border-white/10 bg-black/24 p-3 text-xs leading-5 text-slate-400">
        [{new Date().toLocaleTimeString("zh-CN", { hour12: false })}]{" "}
        {task.status_message}
      </div>
    </div>
  );
}

function DatasetPreviewPanel({
  mode,
  dataset,
  error,
  isImporting,
  selectedLabel,
  labelOptions,
  newLabel,
  onNewLabelChange,
  onAddLabel,
  onSelectedLabelChange,
  onImport,
  onClear,
}: {
  mode: DatasetMode;
  dataset: DatasetPreviewResponse | null;
  error: string | null;
  isImporting: boolean;
  selectedLabel?: string;
  labelOptions?: string[];
  newLabel?: string;
  onNewLabelChange?: (value: string) => void;
  onAddLabel?: () => void;
  onSelectedLabelChange?: (label: string) => void;
  onImport: () => void;
  onClear: () => void | Promise<boolean>;
}) {
  const [isConfirmingClear, setIsConfirmingClear] = useState(false);
  const title = mode === "training" ? "训练数据集" : "分类数据集";
  const importLabel = mode === "training" ? "添加训练图片" : "添加分类图片";
  const clearLabel = mode === "training" ? "删除训练数据集" : "删除分类数据集";
  const datasetLabels = dataset?.labels.map((group) => group.label) ?? [];
  const selectableLabels = labelOptions ?? datasetLabels;
  const selectedLabelGroup = dataset?.labels.find(
    (group) => group.label === selectedLabel,
  );
  const canImport = mode !== "training" || Boolean(selectedLabel);

  return (
    <section className="panel-section">
      <div className="flex items-center justify-between gap-3">
        <div className="section-title">
          <Database className="size-4 text-cyan-300" aria-hidden="true" />
          {title}
        </div>
        <span className="text-xs font-medium text-slate-400">
          {dataset?.count ?? 0} 张图片
        </span>
      </div>

      {mode === "training" ? (
        <div className="mt-4 grid gap-3">
          <div className="flex flex-wrap gap-2" aria-label="训练 label 列表">
            {selectableLabels.length > 0 ? (
              selectableLabels.map((label) => (
                <button
                  key={label}
                  aria-pressed={selectedLabel === label}
                  className={`rounded-full border px-3 py-1.5 text-sm font-semibold transition ${
                    selectedLabel === label
                      ? "border-cyan-300/45 bg-cyan-300/16 text-cyan-100"
                      : "border-white/10 bg-white/[0.045] text-slate-300 hover:bg-white/[0.08]"
                  }`}
                  type="button"
                  onClick={() => onSelectedLabelChange?.(label)}
                >
                  {label}
                </button>
              ))
            ) : (
              <span className="text-sm text-slate-400">
                先添加一个 label，再导入训练图片。
              </span>
            )}
          </div>
          <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto]">
            <label className="grid gap-1.5 text-sm">
              <span className="text-slate-400">新增 label</span>
              <input
                aria-label="新增 label"
                className="min-w-0 rounded-lg border border-white/10 bg-white/[0.045] px-3 py-2 text-slate-100 outline-none transition focus:border-cyan-300/45 focus:ring-2 focus:ring-cyan-300/15"
                placeholder="例如 cat"
                value={newLabel ?? ""}
                onChange={(event) => onNewLabelChange?.(event.currentTarget.value)}
              />
            </label>
            <button
              className="self-end rounded-lg border border-white/10 bg-white/[0.055] px-3 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.09] disabled:cursor-not-allowed disabled:opacity-60"
              type="button"
              disabled={!newLabel?.trim()}
              onClick={onAddLabel}
            >
              添加 label
            </button>
          </div>
        </div>
      ) : null}

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          className="inline-flex items-center gap-2 rounded-lg border border-cyan-300/25 bg-cyan-300/10 px-3 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-300/16 disabled:cursor-not-allowed disabled:opacity-60"
          type="button"
          disabled={isImporting || !canImport}
          onClick={onImport}
        >
          <FolderPlus className="size-4" aria-hidden="true" />
          {isImporting ? "导入中" : importLabel}
        </button>
        <button
          className="inline-flex items-center gap-2 rounded-lg border border-amber-300/25 bg-amber-300/10 px-3 py-2 text-sm font-semibold text-amber-100 transition hover:bg-amber-300/16"
          type="button"
          onClick={() => setIsConfirmingClear(true)}
        >
          <Trash2 className="size-4" aria-hidden="true" />
          {clearLabel}
        </button>
      </div>

      {error ? (
        <div className="mt-4 rounded-lg border border-amber-300/20 bg-amber-300/10 px-3 py-2 text-sm text-amber-200">
          {error}
        </div>
      ) : null}
      {mode === "training" && !canImport ? (
        <p className="mt-3 text-sm text-slate-400">先添加或选择 label 后再导入图片。</p>
      ) : null}

      <div
        className="mt-4 max-h-80 overflow-y-auto rounded-lg border border-white/10 bg-black/24 p-3"
        data-testid="dataset-preview-scroll"
      >
        {mode === "classification" ? (
          <ImageListPreview
            emptyMessage="分类数据集为空。"
            items={dataset?.items ?? []}
            mode="classification"
          />
        ) : (
          <ImageListPreview
            emptyMessage={
              selectedLabel ? `${selectedLabel} 暂无图片。` : "训练数据集为空。"
            }
            items={selectedLabelGroup?.items ?? []}
            label={selectedLabel}
            mode="training"
          />
        )}
      </div>

      {isConfirmingClear ? (
        <ClearDatasetDialog
          mode={mode}
          onClose={() => setIsConfirmingClear(false)}
          onConfirm={() => {
            setIsConfirmingClear(false);
            void onClear();
          }}
        />
      ) : null}
    </section>
  );
}

function ImageListPreview({
  mode,
  label,
  items,
  emptyMessage,
}: {
  mode: DatasetMode;
  label?: string;
  items: DatasetPreviewResponse["items"];
  emptyMessage: string;
}) {
  if (items.length === 0) {
    return <p className="text-sm text-slate-400">{emptyMessage}</p>;
  }

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {items.map((item) => (
        <DatasetThumbnail
          key={item.relative_path}
          item={item}
          label={label}
          mode={mode}
        />
      ))}
    </div>
  );
}

function DatasetThumbnail({
  mode,
  label,
  item,
}: {
  mode: DatasetMode;
  label?: string;
  item: DatasetPreviewResponse["items"][number];
}) {
  const [hasImageError, setHasImageError] = useState(false);

  return (
    <div className="min-w-0 rounded-lg border border-white/10 bg-white/[0.035] p-2">
      {hasImageError ? (
        <div className="grid aspect-square w-full place-items-center rounded-md border border-white/10 bg-black/24 text-xs text-slate-500">
          预览不可用
        </div>
      ) : (
        <img
          alt={item.file_name}
          className="aspect-square w-full rounded-md object-cover"
          loading="lazy"
          src={datasetImageUrl({
            mode,
            label,
            relativePath: item.relative_path,
          })}
          onError={() => setHasImageError(true)}
        />
      )}
      <div className="mt-2 min-w-0">
        <div className="truncate text-xs font-medium text-slate-200">
          {item.file_name}
        </div>
        <div className="mt-1 text-[11px] text-slate-500">{formatBytes(item.size)}</div>
      </div>
    </div>
  );
}

function ClearDatasetDialog({
  mode,
  onClose,
  onConfirm,
}: {
  mode: DatasetMode;
  onClose: () => void;
  onConfirm: () => void;
}) {
  const title = mode === "training" ? "确认删除训练数据集" : "确认删除分类数据集";

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/55 px-4 backdrop-blur-sm">
      <div
        aria-labelledby="clear-dataset-title"
        className="w-full max-w-sm rounded-xl border border-white/12 bg-[#07111c] p-5 shadow-[0_24px_80px_rgba(0,0,0,0.6)]"
        role="dialog"
      >
        <h3 id="clear-dataset-title" className="text-lg font-semibold text-white">
          {title}
        </h3>
        <p className="mt-3 text-sm leading-6 text-slate-300">
          这会清空当前模式下已导入的全部图片。
        </p>
        <div className="mt-5 grid grid-cols-2 gap-3">
          <button
            className="rounded-lg border border-white/10 bg-white/[0.055] px-4 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.09]"
            type="button"
            onClick={onClose}
          >
            取消
          </button>
          <button
            className="rounded-lg border border-amber-300/30 bg-amber-300/12 px-4 py-3 text-sm font-semibold text-amber-100 transition hover:bg-amber-300/18"
            type="button"
            onClick={onConfirm}
          >
            确认删除
          </button>
        </div>
      </div>
    </div>
  );
}

function formatBytes(size: number): string {
  if (size < 1024) {
    return `${size} B`;
  }
  return `${(size / 1024).toFixed(1)} KB`;
}

function MoreSettingsAccordion({
  isOpen,
  onToggle,
  children,
}: {
  isOpen: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  return (
    <section className="mt-5 rounded-xl border border-white/10 bg-black/18">
      <button
        aria-expanded={isOpen}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left text-sm font-semibold text-slate-100 transition hover:bg-white/[0.04] focus:outline-none focus:ring-2 focus:ring-cyan-300/35"
        type="button"
        onClick={onToggle}
      >
        <span className="inline-flex items-center gap-2">
          <SlidersHorizontal className="size-4 text-cyan-300" aria-hidden="true" />
          更多设置
        </span>
        <span className="text-xs text-slate-400">{isOpen ? "收起" : "展开"}</span>
      </button>
      {isOpen ? (
        <div className="space-y-5 border-t border-white/10 p-4">{children}</div>
      ) : null}
    </section>
  );
}

function TaskSummaryControls({
  modelOptions,
  modelOptionsError,
  selectedModel,
  onModelChange,
}: {
  modelOptions: ModelOptionsResponse | null;
  modelOptionsError: string | null;
  selectedModel: string;
  onModelChange: (modelRef: string) => void;
}) {
  return (
    <div className="grid gap-3">
      <label className="grid gap-2 text-sm">
        <span className="text-slate-400">基础模型</span>
        <select
          aria-label="基础模型"
          className="min-w-0 rounded-lg border border-white/10 bg-[#121827] px-3 py-2 text-slate-100 outline-none transition focus:border-cyan-300/45 focus:ring-2 focus:ring-cyan-300/15"
          value={selectedModel}
          onChange={(event) => onModelChange(event.currentTarget.value)}
        >
          {!selectedModel ? (
            <option disabled value="">
              暂无可用模型
            </option>
          ) : null}
          {selectedModel && !hasModelOption(modelOptions, selectedModel) ? (
            <option value={selectedModel}>{selectedModel}</option>
          ) : null}
          <optgroup label="本地模型">
            {(modelOptions?.local_models?.length ?? 0) > 0 ? (
              modelOptions?.local_models.map((model) => (
                <option key={model.path} value={model.path}>
                  {model.label}
                </option>
              ))
            ) : (
              <option disabled value="__empty_local__">
                未发现本地模型
              </option>
            )}
          </optgroup>
          <option disabled value="__separator__">
            ────────
          </option>
          <optgroup label="HF 推荐模型">
            {modelOptions?.recommended_hf_models?.map((model) => (
              <option key={model.path} value={model.path}>
                {model.label}
              </option>
            ))}
          </optgroup>
        </select>
      </label>

      {modelOptionsError ? (
        <div className="mt-3 rounded-lg border border-amber-300/20 bg-amber-300/10 px-3 py-2 text-sm text-amber-200">
          {modelOptionsError}
        </div>
      ) : null}
    </div>
  );
}

function hasModelOption(
  modelOptions: ModelOptionsResponse | null,
  selectedModel: string,
): boolean {
  return Boolean(
    modelOptions &&
      [...modelOptions.local_models, ...modelOptions.recommended_hf_models].some(
        (model) => model.path === selectedModel,
      ),
  );
}

function RunStatusPanel({
  title,
  emptyLogMessage,
  run,
  logs,
  error,
  action,
}: {
  title: string;
  emptyLogMessage: string;
  run: RunRecord | null;
  logs: string[];
  error: string | null;
  action?: ReactNode;
}) {
  return (
    <section className="panel-section">
      <div className="flex items-center justify-between gap-3">
        <div className="section-title">
          <RotateCcw className="size-4 text-cyan-300" aria-hidden="true" />
          {title}
        </div>
        {action}
      </div>

      <div className="mt-4 grid gap-3 text-sm">
        <InfoRow label="run_id" value={run?.run_id ?? "尚未创建"} />
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
            <p>{emptyLogMessage}</p>
          )}
        </div>
      </div>
    </section>
  );
}

function TaskActionButton({
  activeMode,
  currentMode,
  disabled,
  run,
  onStart,
  onRequestCancel,
}: {
  activeMode: TaskCapabilityMode | null;
  currentMode: TaskCapabilityMode;
  disabled: boolean;
  run: RunRecord | null;
  onStart: () => void;
  onRequestCancel: (mode: TaskCapabilityMode) => void;
}) {
  if (activeMode) {
    const progress = getRunProgress(run, activeMode);
    const canRequestCancel =
      Boolean(run) && run !== null && cancellableRunStatuses.has(run.status);
    const actionLabel = activeMode === "training" ? "终止训练" : "终止分类";
    const progressText = `${progress.current}/${progress.total}`;

    return (
      <div className="grid gap-2">
        {activeMode === "training" ? (
          <div className="text-center text-xs font-medium text-slate-400">
            {formatTrainingEpochLabel(run)}
          </div>
        ) : null}
        <button
          className="inline-flex w-full items-center justify-center gap-3 rounded-lg bg-red-500 px-5 py-4 text-base font-semibold text-red-50 shadow-[0_0_34px_rgba(248,113,113,0.28)] transition hover:bg-red-400 focus:outline-none focus:ring-2 focus:ring-red-100 disabled:cursor-not-allowed disabled:opacity-70"
          type="button"
          disabled={!canRequestCancel}
          onClick={() => onRequestCancel(activeMode)}
        >
          <Square className="size-5 fill-current" aria-hidden="true" />
          <span>{actionLabel}</span>
          <span className="font-mono text-sm font-semibold text-red-50/90">
            {progressText}
          </span>
        </button>
      </div>
    );
  }

  const idleLabel = currentMode === "training" ? "开始训练" : "开始分类";

  return (
    <button
      aria-label={idleLabel}
      className="inline-flex w-full items-center justify-center gap-3 rounded-lg bg-cyan-400 px-5 py-4 text-base font-semibold text-cyan-950 shadow-[0_0_34px_rgba(34,211,238,0.32)] transition hover:bg-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-100 disabled:cursor-not-allowed disabled:opacity-60"
      type="button"
      disabled={disabled}
      onClick={onStart}
    >
      <Play className="size-5 fill-current" aria-hidden="true" />
      {idleLabel}
    </button>
  );
}

function getRunProgress(
  run: RunRecord | null,
  mode: TaskCapabilityMode,
): { current: number; total: number; percent: number } {
  void mode;
  const current = run?.processed_items ?? 0;
  const total = run?.total_items ?? 0;
  const percent = total > 0 ? Math.round((current / total) * 100) : 0;

  return {
    current,
    total,
    percent: Math.max(0, Math.min(100, percent)),
  };
}

function formatTrainingEpochLabel(run: RunRecord | null): string {
  const context = run?.progress_context;
  const currentEpoch =
    typeof context?.current_epoch === "number" ? context.current_epoch : null;
  const totalEpochs =
    typeof context?.total_epochs === "number"
      ? context.total_epochs
      : getNumericRequestValue(run, "epochs");

  return `训练轮次：${currentEpoch ?? "--"} / ${totalEpochs ?? "--"}`;
}

function getNumericRequestValue(run: RunRecord | null, key: string): number | null {
  const value = run?.request?.[key];
  return typeof value === "number" ? value : null;
}

function CancelRunDialog({
  mode,
  onClose,
  onConfirm,
}: {
  mode: TaskCapabilityMode;
  onClose: () => void;
  onConfirm: () => void;
}) {
  const title = mode === "training" ? "确认终止训练任务" : "确认终止分类任务";

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/55 px-4 backdrop-blur-sm">
      <div
        aria-labelledby="cancel-run-title"
        className="w-full max-w-sm rounded-xl border border-white/12 bg-[#07111c] p-5 shadow-[0_24px_80px_rgba(0,0,0,0.6)]"
        role="dialog"
      >
        <h3 id="cancel-run-title" className="text-lg font-semibold text-white">
          {title}
        </h3>
        <p className="mt-3 text-sm leading-6 text-slate-300">
          已提交的本地运行会收到取消请求，已经写入的日志会保留。
        </p>
        <div className="mt-5 grid grid-cols-2 gap-3">
          <button
            className="rounded-lg border border-white/10 bg-white/[0.055] px-4 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.09]"
            type="button"
            onClick={onClose}
          >
            继续运行
          </button>
          <button
            className="rounded-lg border border-amber-300/30 bg-amber-300/12 px-4 py-3 text-sm font-semibold text-amber-100 transition hover:bg-amber-300/18"
            type="button"
            onClick={onConfirm}
          >
            确认终止
          </button>
        </div>
      </div>
    </div>
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

function BatchSizeSelect({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="grid gap-1.5 text-sm">
      <span className="text-slate-400">{label}</span>
      <select
        aria-label={label}
        className="min-w-0 rounded-lg border border-white/10 bg-[#121827] px-3 py-2 text-slate-100 outline-none transition focus:border-cyan-300/45 focus:ring-2 focus:ring-cyan-300/15"
        value={String(value)}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      >
        {batchSizeOptions.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function SeedField({
  label,
  value,
  enabled,
  onChange,
  onEnabledChange,
}: {
  label: string;
  value: number;
  enabled: boolean;
  onChange: (value: number) => void;
  onEnabledChange: (enabled: boolean) => void;
}) {
  return (
    <label className="grid gap-1.5 text-sm">
      <span className="flex items-center justify-between gap-3 text-slate-400">
        <span>{label}</span>
        <span className="inline-flex items-center gap-1.5 text-xs text-slate-300">
          <input
            aria-label="启用随机种子"
            checked={enabled}
            className="size-3.5 accent-cyan-300"
            type="checkbox"
            onChange={(event) => onEnabledChange(event.currentTarget.checked)}
          />
          启用
        </span>
      </span>
      <input
        aria-label={label}
        className="min-w-0 rounded-lg border border-white/10 bg-white/[0.045] px-3 py-2 text-slate-100 outline-none transition disabled:cursor-not-allowed disabled:border-white/[0.06] disabled:bg-white/[0.025] disabled:text-slate-500 focus:border-cyan-300/45 focus:ring-2 focus:ring-cyan-300/15"
        disabled={!enabled}
        type="number"
        value={value}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
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
