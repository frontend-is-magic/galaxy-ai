import { Settings, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { GalaxyCanvas } from "./features/galaxy/GalaxyCanvas";
import { TaskInspector } from "./features/galaxy/TaskInspector";
import {
  cancelRun,
  clearImageClassificationDataset,
  createImageClassificationInference,
  createImageClassificationTraining,
  getImageClassificationDataset,
  getModelOptions,
  getRun,
  getRunLogs,
  getRuntimeHardware,
  getSettings,
  importImageClassificationDataset,
  openRunOutput,
} from "./features/galaxy/api";
import type {
  BatchInferenceRequest,
  DatasetImportFile,
  DatasetPreviewResponse,
  DatasetMode,
  DevicePreference,
  DirectorySettings,
  ModelOption,
  ModelOptionsResponse,
  RunRecord,
  RunStatus,
  RuntimeHardwareResponse,
  TrainingRequest,
} from "./features/galaxy/api";
import {
  defaultTaskDirectoriesForMode,
  defaultTrainingRequest,
  nebulaSorterTask,
} from "./features/galaxy/taskData";
import type { TaskCapabilityMode, TaskStatus } from "./features/galaxy/types";
import type { TaskDirectories } from "./features/galaxy/types";

const unfinishedRunStatuses = new Set<RunStatus>(["queued", "running", "cancelling"]);
const cancellableRunStatuses = new Set<RunStatus>(["queued", "running"]);
const selectableDevices: DevicePreference[] = ["auto", "cpu", "cuda", "mps"];
const modelOptionsRefreshIntervalMs = 2500;
const supportedImportExtensions = new Set(["jpg", "jpeg", "png", "bmp", "webp"]);

export function App() {
  const [capabilityMode, setCapabilityMode] =
    useState<TaskCapabilityMode>("classification");
  const [isMoreSettingsOpen, setIsMoreSettingsOpen] = useState(false);
  const [status, setStatus] = useState<TaskStatus>(nebulaSorterTask.status);
  const [statusMessage, setStatusMessage] = useState("当前无错误");
  const [trainingRequest, setTrainingRequest] =
    useState<TrainingRequest>(defaultTrainingRequest);
  const [classificationRun, setClassificationRun] = useState<RunRecord | null>(null);
  const [classificationLogs, setClassificationLogs] = useState<string[]>([]);
  const [classificationError, setClassificationError] = useState<string | null>(null);
  const [isClassificationSubmitting, setIsClassificationSubmitting] = useState(false);
  const [trainingRun, setTrainingRun] = useState<RunRecord | null>(null);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [isTrainingSubmitting, setIsTrainingSubmitting] = useState(false);
  const [modelOptions, setModelOptions] = useState<ModelOptionsResponse | null>(null);
  const [modelOptionsError, setModelOptionsError] = useState<string | null>(null);
  const [classificationDataset, setClassificationDataset] =
    useState<DatasetPreviewResponse | null>(null);
  const [trainingDataset, setTrainingDataset] = useState<DatasetPreviewResponse | null>(
    null,
  );
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [isImportingDataset, setIsImportingDataset] = useState(false);
  const [globalDevice, setGlobalDevice] = useState<DevicePreference>("auto");
  const [runtimeHardware, setRuntimeHardware] =
    useState<RuntimeHardwareResponse | null>(null);
  const [settingsDraftWorkingDirectory, setSettingsDraftWorkingDirectory] =
    useState("");
  const [settingsDraftDevice, setSettingsDraftDevice] =
    useState<DevicePreference>("auto");
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [taskDirectorySettings, setTaskDirectorySettings] = useState<
    Record<TaskCapabilityMode, TaskDirectories>
  >(() => ({
    classification: defaultTaskDirectoriesForMode("classification"),
    training: defaultTaskDirectoriesForMode("training"),
  }));
  const taskDirectories = useMemo(
    () => taskDirectorySettings[capabilityMode],
    [capabilityMode, taskDirectorySettings],
  );

  const selectedStatus = useMemo(() => {
    if (capabilityMode === "training" && trainingRun) {
      return mapRunStatusToTaskStatus(trainingRun.status);
    }
    if (capabilityMode === "classification" && classificationRun) {
      return mapRunStatusToTaskStatus(classificationRun.status);
    }

    return status;
  }, [capabilityMode, classificationRun, status, trainingRun]);

  const selectedModelOption = useMemo(
    () => findModelOption(modelOptions, trainingRequest.base_model_ref),
    [modelOptions, trainingRequest.base_model_ref],
  );
  const selectedModelAllowsDownload = selectedModelOption?.source === "huggingface";
  const selectedModelCompatibilityError =
    selectedModelOption?.compatible === false
      ? selectedModelOption.compatibility_error ||
        "当前模型不是图片分类模型，请选择兼容模型。"
      : null;
  const startBlockedReason = selectedModelCompatibilityError;
  const activeTaskMode = getActiveTaskMode({
    classificationRun,
    trainingRun,
    isClassificationSubmitting,
    isTrainingSubmitting,
  });

  const task = {
    ...nebulaSorterTask,
    model_path: trainingRequest.base_model_ref || "暂无可用模型",
    working_directory: taskDirectories.working_directory,
    model_directory: taskDirectories.model_directory,
    input_directory: taskDirectories.dataset_directory,
    output_directory: taskDirectories.output_directory,
    total_images:
      capabilityMode === "classification" && classificationRun
        ? classificationRun.total_items
        : nebulaSorterTask.total_images,
    processed_images:
      capabilityMode === "classification" && classificationRun
        ? classificationRun.processed_items
        : nebulaSorterTask.processed_images,
    status: selectedStatus,
    error_message:
      capabilityMode === "training"
        ? trainingError || trainingRun?.error_message || null
        : classificationError || classificationRun?.error_message || null,
    status_message:
      capabilityMode === "training"
        ? trainingError || trainingRun?.status || "训练准备就绪"
        : classificationError || classificationRun?.status || statusMessage,
  };

  async function startClassification(batchSize: number) {
    if (activeTaskMode) {
      return;
    }

    setIsClassificationSubmitting(true);
    setClassificationError(null);

    try {
      const classificationDirectories = taskDirectorySettings.classification;
      const request: BatchInferenceRequest = {
        model_ref: trainingRequest.base_model_ref,
        model_directory: classificationDirectories.model_directory,
        allow_download: selectedModelAllowsDownload,
        input_directory: classificationDirectories.dataset_directory,
        output_directory: classificationDirectories.output_directory,
        recursive: true,
        batch_size: batchSize,
        top_k: 5,
        device: globalDevice,
      };
      const created = await createImageClassificationInference(request);
      setStatus("running");
      setStatusMessage("分类任务已提交");
      setClassificationRun(
        createPendingRun(
          created.run_id,
          created.status,
          "image_classification_inference",
          request,
          request.model_ref,
          request.input_directory,
          request.output_directory,
          request.device,
        ),
      );
      setClassificationLogs([]);
      await refreshClassificationRun(created.run_id);
    } catch (error) {
      setStatus("error");
      setClassificationError(error instanceof Error ? error.message : "分类请求失败。");
    } finally {
      setIsClassificationSubmitting(false);
    }
  }

  function updateTrainingRequest(update: Partial<TrainingRequest>) {
    setTrainingRequest((current) => ({ ...current, ...update }));
  }

  function updateTaskModel(modelRef: string) {
    setTrainingRequest((current) => ({
      ...current,
      base_model_ref: modelRef,
    }));
  }

  const selectedModelAvailable = useMemo(
    () => hasModelOption(modelOptions, trainingRequest.base_model_ref),
    [modelOptions, trainingRequest.base_model_ref],
  );

  const canStartTask =
    Boolean(trainingRequest.base_model_ref) &&
    (modelOptions === null || selectedModelAvailable) &&
    !startBlockedReason;

  const openSettings = useCallback(async () => {
    setIsSettingsOpen(true);
    setSettingsError(null);

    try {
      const loaded = await getSettings();
      const loadedDevice = normalizeDevice(loaded.device);
      setSettingsDraftWorkingDirectory(loaded.working_directory);
      setSettingsDraftDevice(loadedDevice);
      setTaskDirectorySettings(directoriesFromSettings(loaded));
      setGlobalDevice(loadedDevice);
      updateTrainingRequest({ device: loadedDevice });
    } catch (error) {
      setSettingsError(error instanceof Error ? error.message : "加载设置失败。");
    }
  }, []);

  useEffect(() => {
    let isCurrent = true;

    void getSettings()
      .then((loaded) => {
        if (isCurrent) {
          const loadedDevice = normalizeDevice(loaded.device);
          setSettingsDraftWorkingDirectory(loaded.working_directory);
          setSettingsDraftDevice(loadedDevice);
          setTaskDirectorySettings(directoriesFromSettings(loaded));
          setGlobalDevice(loadedDevice);
          updateTrainingRequest({ device: loadedDevice });
        }
      })
      .catch(() => {
        // Keep the repository-local demo defaults when the backend is not ready.
      });

    return () => {
      isCurrent = false;
    };
  }, []);

  const refreshTrainingRun = useCallback(async (runId: string) => {
    const [run, logs] = await Promise.all([getRun(runId), getRunLogs(runId)]);
    setTrainingRun((current) => mergeRunRefresh(current, run));
    setTrainingLogs(logs.logs);
  }, []);

  const refreshClassificationRun = useCallback(async (runId: string) => {
    const [run, logs] = await Promise.all([getRun(runId), getRunLogs(runId)]);
    setClassificationRun((current) => mergeRunRefresh(current, run));
    setClassificationLogs(logs.logs);
  }, []);

  const refreshDatasets = useCallback(async () => {
    const [classification, training] = await Promise.all([
      getImageClassificationDataset("classification"),
      getImageClassificationDataset("training"),
    ]);
    setClassificationDataset(classification);
    setTrainingDataset(training);
  }, []);

  useEffect(() => {
    void refreshDatasets().catch((error: unknown) => {
      setDatasetError(error instanceof Error ? error.message : "加载数据集预览失败。");
    });
  }, [refreshDatasets]);

  useEffect(() => {
    let isCurrent = true;

    function refreshModelOptions() {
      setModelOptionsError(null);

      void getModelOptions(taskDirectorySettings.classification.model_directory)
        .then((options) => {
          if (isCurrent) {
            setModelOptions(options);
          }
        })
        .catch((error: unknown) => {
          if (isCurrent) {
            setModelOptionsError(
              error instanceof Error ? error.message : "加载模型列表失败。",
            );
          }
        });
    }

    function refreshVisibleModelOptions() {
      if (document.visibilityState === "visible") {
        refreshModelOptions();
      }
    }

    refreshModelOptions();
    const intervalId = window.setInterval(
      refreshModelOptions,
      modelOptionsRefreshIntervalMs,
    );
    window.addEventListener("focus", refreshModelOptions);
    document.addEventListener("visibilitychange", refreshVisibleModelOptions);

    return () => {
      isCurrent = false;
      window.clearInterval(intervalId);
      window.removeEventListener("focus", refreshModelOptions);
      document.removeEventListener("visibilitychange", refreshVisibleModelOptions);
    };
  }, [taskDirectorySettings.classification.model_directory]);

  useEffect(() => {
    const fallbackModel = pickFallbackModel(modelOptions);
    if (!fallbackModel) {
      return;
    }

    if (!hasModelOption(modelOptions, trainingRequest.base_model_ref)) {
      updateTaskModel(fallbackModel);
    }
  }, [modelOptions, trainingRequest.base_model_ref]);

  useEffect(() => {
    void getRuntimeHardware()
      .then(setRuntimeHardware)
      .catch(() => {
        // Settings stays usable when the backend cannot report hardware details.
      });
  }, []);

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

  useEffect(() => {
    if (!classificationRun || !unfinishedRunStatuses.has(classificationRun.status)) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      void refreshClassificationRun(classificationRun.run_id).catch(
        (error: unknown) => {
          setClassificationError(
            error instanceof Error ? error.message : "刷新分类状态失败。",
          );
        },
      );
    }, 2500);

    return () => window.clearInterval(intervalId);
  }, [classificationRun, refreshClassificationRun]);

  async function startTraining() {
    if (activeTaskMode) {
      return;
    }

    setIsTrainingSubmitting(true);
    setTrainingError(null);

    try {
      const trainingDirectories = taskDirectorySettings.training;
      const request = {
        ...trainingRequest,
        model_directory: trainingDirectories.model_directory,
        dataset_directory: trainingDirectories.dataset_directory,
        output_directory: trainingDirectories.output_directory,
        checkpoint_directory: trainingDirectories.checkpoint_directory,
        allow_download: selectedModelAllowsDownload,
        device: globalDevice,
      };
      const created = await createImageClassificationTraining(request);
      setTrainingRun(
        createPendingRun(
          created.run_id,
          created.status,
          "image_classification_training",
          request,
          request.base_model_ref,
          request.dataset_directory,
          request.output_directory,
          request.device,
        ),
      );
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
    setTrainingRun((current) => markRunCancelling(current, trainingRun.run_id));

    try {
      const cancelled = await cancelRun(trainingRun.run_id);
      setTrainingRun((current) =>
        current && current.run_id === cancelled.run_id
          ? { ...current, status: cancelled.status }
          : current,
      );
      setTrainingLogs((current) =>
        current.includes("正在取消运行") ? current : [...current, "正在取消运行"],
      );
      await refreshTrainingRun(trainingRun.run_id);
      setTrainingLogs((current) =>
        current.includes("正在取消运行") ? current : [...current, "正在取消运行"],
      );
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "取消运行失败。");
    }
  }

  async function cancelClassification() {
    if (!classificationRun) {
      return;
    }

    setClassificationError(null);
    setClassificationRun((current) =>
      markRunCancelling(current, classificationRun.run_id),
    );

    try {
      const cancelled = await cancelRun(classificationRun.run_id);
      setClassificationRun((current) =>
        current && current.run_id === cancelled.run_id
          ? { ...current, status: cancelled.status }
          : current,
      );
      setClassificationLogs((current) =>
        current.includes("正在取消运行") ? current : [...current, "正在取消运行"],
      );
      await refreshClassificationRun(classificationRun.run_id);
      setClassificationLogs((current) =>
        current.includes("正在取消运行") ? current : [...current, "正在取消运行"],
      );
    } catch (error) {
      setClassificationError(error instanceof Error ? error.message : "取消运行失败。");
    }
  }

  async function openClassificationOutput(runId: string) {
    setClassificationError(null);

    try {
      await openRunOutput(runId);
    } catch (error) {
      setClassificationError(
        error instanceof Error ? error.message : "无法打开结果目录。",
      );
    }
  }

  async function importDataset(mode: DatasetMode, label?: string) {
    if (!window.showDirectoryPicker) {
      setDatasetError("当前浏览器不支持目录导入。");
      return;
    }

    const normalizedLabel = label?.trim();
    if (mode === "training" && !normalizedLabel) {
      setDatasetError("请先添加或选择训练 label。");
      return;
    }

    setDatasetError(null);
    setIsImportingDataset(true);

    try {
      const directory = await window.showDirectoryPicker({ mode: "read" });
      const files = await collectImageFiles(directory);
      if (files.length === 0) {
        setDatasetError("所选目录中未发现支持格式图片。");
        return;
      }
      await importImageClassificationDataset({
        mode,
        label: normalizedLabel,
        files,
      });
      await refreshDatasets();
    } catch (error) {
      setDatasetError(error instanceof Error ? error.message : "导入数据集失败。");
    } finally {
      setIsImportingDataset(false);
    }
  }

  async function clearDataset(mode: DatasetMode): Promise<boolean> {
    setDatasetError(null);

    try {
      await clearImageClassificationDataset(mode);
      await refreshDatasets();
      return true;
    } catch (error) {
      setDatasetError(error instanceof Error ? error.message : "清空数据集失败。");
      return false;
    }
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
              全局设备：{globalDevice.toUpperCase()}
            </div>
            <div
              className="hidden max-w-[32rem] rounded-md border border-white/10 bg-white/[0.045] px-3 py-2 text-sm text-slate-300 xl:block"
              title={taskDirectories.working_directory}
            >
              <span className="text-slate-500">工作目录：</span>
              <span className="inline-block max-w-[24rem] truncate align-bottom font-mono text-cyan-100">
                {taskDirectories.working_directory}
              </span>
            </div>
          </div>

          <button
            className="pointer-events-auto inline-flex items-center gap-2 rounded-md border border-white/12 bg-white/[0.055] px-3 py-2 text-sm font-medium text-slate-100 transition hover:border-cyan-300/40 hover:bg-cyan-300/10 focus:outline-none focus:ring-2 focus:ring-cyan-300/50"
            type="button"
            onClick={() => void openSettings()}
          >
            <Settings className="size-4" aria-hidden="true" />
            设置
          </button>
        </header>

        <TaskInspector
          capabilityMode={capabilityMode}
          isMoreSettingsOpen={isMoreSettingsOpen}
          task={task}
          trainingRequest={trainingRequest}
          classificationRun={classificationRun}
          classificationLogs={classificationLogs}
          classificationError={classificationError}
          trainingRun={trainingRun}
          trainingLogs={trainingLogs}
          trainingError={trainingError}
          modelOptions={modelOptions}
          modelOptionsError={modelOptionsError}
          classificationDataset={classificationDataset}
          trainingDataset={trainingDataset}
          datasetError={datasetError}
          isImportingDataset={isImportingDataset}
          canStartTask={canStartTask}
          startBlockedReason={startBlockedReason}
          activeTaskMode={activeTaskMode}
          onCapabilityModeChange={setCapabilityMode}
          onMoreSettingsToggle={() => setIsMoreSettingsOpen((current) => !current)}
          onTrainingRequestChange={updateTrainingRequest}
          onTaskModelChange={updateTaskModel}
          onStartClassification={startClassification}
          onStartTraining={startTraining}
          onCancelClassification={cancelClassification}
          onCancelTraining={cancelTraining}
          onOpenClassificationOutput={openClassificationOutput}
          onImportDataset={importDataset}
          onClearDataset={clearDataset}
        />

        {isSettingsOpen ? (
          <SettingsDialog
            draftWorkingDirectory={settingsDraftWorkingDirectory}
            draftDevice={settingsDraftDevice}
            runtimeHardware={runtimeHardware}
            error={settingsError}
            onClose={() => setIsSettingsOpen(false)}
            onDraftDeviceChange={setSettingsDraftDevice}
          />
        ) : null}

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

function getActiveTaskMode({
  classificationRun,
  trainingRun,
  isClassificationSubmitting,
  isTrainingSubmitting,
}: {
  classificationRun: RunRecord | null;
  trainingRun: RunRecord | null;
  isClassificationSubmitting: boolean;
  isTrainingSubmitting: boolean;
}): TaskCapabilityMode | null {
  if (
    isClassificationSubmitting ||
    (classificationRun && unfinishedRunStatuses.has(classificationRun.status))
  ) {
    return "classification";
  }

  if (
    isTrainingSubmitting ||
    (trainingRun && unfinishedRunStatuses.has(trainingRun.status))
  ) {
    return "training";
  }

  return null;
}

function markRunCancelling(run: RunRecord | null, runId: string): RunRecord | null {
  if (!run || run.run_id !== runId || !cancellableRunStatuses.has(run.status)) {
    return run;
  }
  return { ...run, status: "cancelling" };
}

function mergeRunRefresh(current: RunRecord | null, refreshed: RunRecord): RunRecord {
  if (
    current?.run_id === refreshed.run_id &&
    current.status === "cancelling" &&
    cancellableRunStatuses.has(refreshed.status)
  ) {
    return {
      ...refreshed,
      status: "cancelling",
      error_message: current.error_message,
    };
  }

  return refreshed;
}

function createPendingRun(
  runId: string,
  status: RunStatus,
  runType: string,
  request: TrainingRequest | BatchInferenceRequest,
  modelRef: string,
  inputPath: string,
  outputPath: string | null,
  hardwareBackend: string,
): RunRecord {
  const now = new Date().toISOString();
  return {
    run_id: runId,
    run_type: runType,
    status,
    request,
    hardware_backend: hardwareBackend,
    model_ref: modelRef,
    input_path: inputPath,
    output_path: outputPath,
    total_items: 0,
    processed_items: 0,
    error_message: null,
    created_at: now,
    updated_at: now,
    started_at: null,
    completed_at: null,
  };
}

function directoriesFromSettings(
  settings: DirectorySettings,
): Record<TaskCapabilityMode, TaskDirectories> {
  const classificationDatasetDirectory =
    settings.classification_dataset_directory ?? settings.dataset_directory;
  const classificationOutputDirectory =
    settings.classification_output_directory ?? settings.output_directory;
  const trainingDatasetDirectory =
    settings.training_dataset_directory ?? settings.dataset_directory;
  const trainingOutputDirectory =
    settings.training_output_directory ?? settings.output_directory;

  return {
    classification: {
      working_directory: settings.working_directory,
      model_directory: settings.model_directory,
      dataset_directory: classificationDatasetDirectory,
      output_directory: classificationOutputDirectory,
      checkpoint_directory: settings.checkpoint_directory,
    },
    training: {
      working_directory: settings.working_directory,
      model_directory: settings.model_directory,
      dataset_directory: trainingDatasetDirectory,
      output_directory: trainingOutputDirectory,
      checkpoint_directory: settings.checkpoint_directory,
    },
  };
}

function findModelOption(
  modelOptions: ModelOptionsResponse | null,
  selectedModel: string,
): ModelOption | null {
  if (!modelOptions || !selectedModel) {
    return null;
  }

  return (
    [...modelOptions.local_models, ...modelOptions.recommended_hf_models].find(
      (model) => model.path === selectedModel,
    ) ?? null
  );
}

function hasModelOption(
  modelOptions: ModelOptionsResponse | null,
  selectedModel: string,
): boolean {
  if (!modelOptions || !selectedModel) {
    return false;
  }

  return [...modelOptions.local_models, ...modelOptions.recommended_hf_models].some(
    (model) => model.path === selectedModel,
  );
}

function pickFallbackModel(modelOptions: ModelOptionsResponse | null): string {
  return (
    modelOptions?.local_models[0]?.path ??
    modelOptions?.recommended_hf_models[0]?.path ??
    ""
  );
}

function normalizeDevice(device: unknown): DevicePreference {
  return selectableDevices.includes(device as DevicePreference)
    ? (device as DevicePreference)
    : "auto";
}

async function collectImageFiles(
  directory: BrowserDirectoryHandle,
): Promise<DatasetImportFile[]> {
  const files: DatasetImportFile[] = [];

  async function visit(handle: BrowserDirectoryHandle, prefix: string) {
    const entries = handle.entries
      ? handle.entries()
      : handle.values
        ? mapValuesToEntries(handle.values())
        : null;
    if (!entries) {
      return;
    }

    for await (const [name, entry] of entries) {
      const relativePath = prefix ? `${prefix}/${name}` : name;
      if (entry.kind === "file") {
        const file = await entry.getFile();
        if (isSupportedImageName(file.name || name)) {
          files.push({ file, relativePath });
        }
      } else {
        await visit(entry, relativePath);
      }
    }
  }

  await visit(directory, "");
  return files;
}

async function* mapValuesToEntries(
  values: AsyncIterableIterator<BrowserDirectoryHandle | BrowserFileHandle>,
): AsyncIterableIterator<[string, BrowserDirectoryHandle | BrowserFileHandle]> {
  for await (const entry of values) {
    yield [entry.name, entry];
  }
}

function isSupportedImageName(name: string): boolean {
  const extension = name.split(".").pop()?.toLowerCase() ?? "";
  return supportedImportExtensions.has(extension);
}

function SettingsDialog({
  draftWorkingDirectory,
  draftDevice,
  runtimeHardware,
  error,
  onClose,
  onDraftDeviceChange,
}: {
  draftWorkingDirectory: string;
  draftDevice: DevicePreference;
  runtimeHardware: RuntimeHardwareResponse | null;
  error: string | null;
  onClose: () => void;
  onDraftDeviceChange: (value: DevicePreference) => void;
}) {
  return (
    <div
      className="fixed inset-0 z-40 grid place-items-center bg-black/58 px-4 backdrop-blur-sm"
      role="presentation"
    >
      <section
        aria-label="本地默认设置"
        className="w-full max-w-xl rounded-2xl border border-white/12 bg-[#07111c] p-5 shadow-[0_24px_80px_rgba(0,0,0,0.62)] sm:p-6"
        role="dialog"
      >
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold text-white">本地默认设置</h2>
            <p className="mt-1 text-sm text-slate-400">
              目录和设备由仓库根目录 .env 配置；这里仅展示当前生效值。
            </p>
          </div>
          <button
            aria-label="关闭设置"
            className="grid size-9 place-items-center rounded-md border border-white/10 bg-white/[0.04] text-slate-300 transition hover:text-white"
            type="button"
            onClick={onClose}
          >
            <X className="size-4" aria-hidden="true" />
          </button>
        </div>

        <div className="mt-5 grid gap-3">
          <label className="grid gap-2 text-sm">
            <span className="font-medium text-slate-200">默认工作目录</span>
            <input
              className="min-w-0 rounded-lg border border-white/10 bg-white/[0.035] px-3 py-2 text-slate-400 outline-none"
              disabled
              value={draftWorkingDirectory}
            />
          </label>
          <label className="grid gap-2 text-sm">
            <span className="font-medium text-slate-200">硬件设备</span>
            <select
              aria-label="硬件设备"
              className="min-w-0 rounded-lg border border-white/10 bg-[#121827] px-3 py-2 text-slate-100 outline-none transition focus:border-cyan-300/45 focus:ring-2 focus:ring-cyan-300/15"
              disabled
              value={draftDevice}
              onChange={(event) =>
                onDraftDeviceChange(event.currentTarget.value as DevicePreference)
              }
            >
              {selectableDevices.map((device) => {
                const available =
                  device === "auto" ||
                  runtimeHardware?.backends?.[device]?.available !== false;
                const label =
                  device === "auto"
                    ? "Auto"
                    : (runtimeHardware?.backends?.[device]?.label ??
                      device.toUpperCase());

                return (
                  <option key={device} disabled={!available} value={device}>
                    {available ? label : `${label}（不可用）`}
                  </option>
                );
              })}
            </select>
          </label>
        </div>

        {error ? (
          <div className="mt-4 rounded-lg border border-amber-300/20 bg-amber-300/10 px-3 py-2 text-sm text-amber-200">
            {error}
          </div>
        ) : null}

        <div className="mt-6 flex justify-end gap-3">
          <button
            className="rounded-lg border border-white/10 bg-white/[0.04] px-4 py-2 text-sm font-semibold text-slate-200 transition hover:bg-white/[0.08]"
            type="button"
            onClick={onClose}
          >
            关闭
          </button>
        </div>
      </section>
    </div>
  );
}
