import { Save, Settings, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { GalaxyCanvas } from "./features/galaxy/GalaxyCanvas";
import { TaskInspector } from "./features/galaxy/TaskInspector";
import {
  cancelRun,
  createImageClassificationInference,
  createImageClassificationTraining,
  getModelOptions,
  getRun,
  getRunLogs,
  getRuntimeHardware,
  getSettings,
  updateSettings,
} from "./features/galaxy/api";
import type {
  BatchInferenceRequest,
  DevicePreference,
  ModelOption,
  ModelOptionsResponse,
  RunRecord,
  RunStatus,
  RuntimeHardwareResponse,
  TrainingRequest,
} from "./features/galaxy/api";
import { defaultTrainingRequest, nebulaSorterTask } from "./features/galaxy/taskData";
import type {
  TaskCapabilityMode,
  TaskDirectories,
  TaskStatus,
} from "./features/galaxy/types";

const unfinishedRunStatuses = new Set<RunStatus>(["queued", "running", "cancelling"]);
const selectableDevices: DevicePreference[] = ["auto", "cpu", "cuda", "mps"];

const defaultTaskDirectories: TaskDirectories = {
  working_directory: nebulaSorterTask.working_directory,
  model_directory: nebulaSorterTask.model_directory,
  dataset_directory: defaultTrainingRequest.dataset_directory,
  output_directory: nebulaSorterTask.output_directory,
  checkpoint_directory: defaultTrainingRequest.checkpoint_directory ?? "",
};

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
  const [taskDirectories, setTaskDirectories] =
    useState<TaskDirectories>(defaultTaskDirectories);
  const [hasCustomTaskDirectories, setHasCustomTaskDirectories] = useState(false);
  const [modelOptions, setModelOptions] = useState<ModelOptionsResponse | null>(null);
  const [modelOptionsError, setModelOptionsError] = useState<string | null>(null);
  const [globalDevice, setGlobalDevice] = useState<DevicePreference>("auto");
  const [locallySavedDevice, setLocallySavedDevice] = useState<DevicePreference | null>(
    null,
  );
  const [runtimeHardware, setRuntimeHardware] =
    useState<RuntimeHardwareResponse | null>(null);
  const [settingsDraftWorkingDirectory, setSettingsDraftWorkingDirectory] =
    useState("");
  const [settingsDraftDevice, setSettingsDraftDevice] =
    useState<DevicePreference>("auto");
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSettingsSaving, setIsSettingsSaving] = useState(false);

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
  const selectedModelRequiresDownload = Boolean(
    selectedModelOption?.source === "huggingface" &&
      selectedModelOption.requires_download &&
      !trainingRequest.allow_download,
  );
  const selectedModelCompatibilityError =
    selectedModelOption?.compatible === false
      ? selectedModelOption.compatibility_error ||
        "当前模型不是图片分类模型，请选择兼容模型。"
      : null;
  const startBlockedReason = selectedModelCompatibilityError
    ? selectedModelCompatibilityError
    : selectedModelRequiresDownload
      ? "需勾选允许显式下载模型，或选择本地已缓存模型。"
      : null;
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
      const request: BatchInferenceRequest = {
        model_ref: trainingRequest.base_model_ref,
        model_directory: taskDirectories.model_directory,
        allow_download: trainingRequest.allow_download,
        input_directory: taskDirectories.dataset_directory,
        output_directory: taskDirectories.output_directory,
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

  function applyTaskDirectories(nextDirectories: TaskDirectories, isCustom = true) {
    setTaskDirectories(nextDirectories);
    setHasCustomTaskDirectories(isCustom);
    setTrainingRequest((current) => ({
      ...current,
      model_directory: nextDirectories.model_directory,
      dataset_directory: nextDirectories.dataset_directory,
      output_directory: nextDirectories.output_directory,
      checkpoint_directory: nextDirectories.checkpoint_directory,
    }));
  }

  function updateTaskDirectory() {
    showDirectoryLockedToast();
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
      const effectiveDevice = locallySavedDevice ?? loadedDevice;
      setSettingsDraftWorkingDirectory(loaded.working_directory);
      setSettingsDraftDevice(effectiveDevice);
      setGlobalDevice(effectiveDevice);
      updateTrainingRequest({ device: effectiveDevice });

      if (!hasCustomTaskDirectories) {
        applyTaskDirectories(
          deriveTaskDirectories(
            loaded.working_directory,
            nebulaSorterTask.directory_name,
          ),
          false,
        );
      }
    } catch (error) {
      setSettingsError(error instanceof Error ? error.message : "加载设置失败。");
    }
  }, [hasCustomTaskDirectories, locallySavedDevice]);

  useEffect(() => {
    let isCurrent = true;

    void getSettings()
      .then((loaded) => {
        if (isCurrent && !hasCustomTaskDirectories) {
          const loadedDevice = normalizeDevice(loaded.device);
          setSettingsDraftWorkingDirectory(loaded.working_directory);
          setSettingsDraftDevice(loadedDevice);
          setGlobalDevice(loadedDevice);
          updateTrainingRequest({ device: loadedDevice });
          applyTaskDirectories(
            deriveTaskDirectories(
              loaded.working_directory,
              nebulaSorterTask.directory_name,
            ),
            false,
          );
        }
      })
      .catch(() => {
        // Keep the repository-local demo defaults when the backend is not ready.
      });

    return () => {
      isCurrent = false;
    };
  }, [hasCustomTaskDirectories]);

  async function saveSettings() {
    setIsSettingsSaving(true);
    setSettingsError(null);

    try {
      const requestedDevice = settingsDraftDevice;
      const saved = await updateSettings({
        working_directory: settingsDraftWorkingDirectory,
        device: requestedDevice,
      });
      const savedDevice = normalizeDevice(requestedDevice);
      setLocallySavedDevice(savedDevice);
      setSettingsDraftWorkingDirectory(saved.working_directory);
      setSettingsDraftDevice(savedDevice);
      setGlobalDevice(savedDevice);
      updateTrainingRequest({ device: savedDevice });

      if (!hasCustomTaskDirectories) {
        applyTaskDirectories(
          deriveTaskDirectories(
            saved.working_directory,
            nebulaSorterTask.directory_name,
          ),
          false,
        );
      }
      setIsSettingsOpen(false);
    } catch (error) {
      setSettingsError(error instanceof Error ? error.message : "保存设置失败。");
    } finally {
      setIsSettingsSaving(false);
    }
  }

  function showDirectoryLockedToast() {
    setToastMessage("当前版本不允许更改目录");
    window.setTimeout(() => {
      setToastMessage(null);
    }, 2600);
  }

  const refreshTrainingRun = useCallback(async (runId: string) => {
    const [run, logs] = await Promise.all([getRun(runId), getRunLogs(runId)]);
    setTrainingRun(run);
    setTrainingLogs(logs.logs);
  }, []);

  const refreshClassificationRun = useCallback(async (runId: string) => {
    const [run, logs] = await Promise.all([getRun(runId), getRunLogs(runId)]);
    setClassificationRun(run);
    setClassificationLogs(logs.logs);
  }, []);

  useEffect(() => {
    let isCurrent = true;
    setModelOptionsError(null);

    void getModelOptions(taskDirectories.model_directory)
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

    return () => {
      isCurrent = false;
    };
  }, [taskDirectories.model_directory]);

  async function startTraining() {
    if (activeTaskMode) {
      return;
    }

    setIsTrainingSubmitting(true);
    setTrainingError(null);

    try {
      const request = { ...trainingRequest, device: globalDevice };
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

    try {
      await cancelRun(trainingRun.run_id);
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

    try {
      await cancelRun(classificationRun.run_id);
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

  useEffect(() => {
    void getRuntimeHardware()
      .then(setRuntimeHardware)
      .catch(() => {
        // Settings stays usable when the backend cannot report hardware details.
      });
  }, []);

  useEffect(() => {
    const fallbackModel = pickFallbackModel(modelOptions);
    if (!fallbackModel) {
      return;
    }

    if (!hasModelOption(modelOptions, trainingRequest.base_model_ref)) {
      updateTaskModel(fallbackModel);
    }
  }, [modelOptions, trainingRequest.base_model_ref]);

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
          taskDirectories={taskDirectories}
          isTrainingSubmitting={isTrainingSubmitting}
          isClassificationSubmitting={isClassificationSubmitting}
          canStartTask={canStartTask}
          startBlockedReason={startBlockedReason}
          activeTaskMode={activeTaskMode}
          onCapabilityModeChange={setCapabilityMode}
          onMoreSettingsToggle={() => setIsMoreSettingsOpen((current) => !current)}
          onTrainingRequestChange={updateTrainingRequest}
          onTaskDirectoryChange={updateTaskDirectory}
          onTaskModelChange={updateTaskModel}
          onSelectTaskDirectory={showDirectoryLockedToast}
          onStartClassification={startClassification}
          onStartTraining={startTraining}
          onCancelClassification={cancelClassification}
          onCancelTraining={cancelTraining}
        />

        {isSettingsOpen ? (
          <SettingsDialog
            draftWorkingDirectory={settingsDraftWorkingDirectory}
            draftDevice={settingsDraftDevice}
            runtimeHardware={runtimeHardware}
            error={settingsError}
            isSaving={isSettingsSaving}
            onClose={() => setIsSettingsOpen(false)}
            onDraftDeviceChange={setSettingsDraftDevice}
            onSave={saveSettings}
          />
        ) : null}

        {toastMessage ? (
          <div
            className="fixed bottom-5 left-1/2 z-50 -translate-x-1/2 rounded-lg border border-cyan-300/25 bg-[#07111c]/95 px-4 py-3 text-sm font-medium text-cyan-100 shadow-[0_18px_50px_rgba(0,0,0,0.45)] backdrop-blur-xl"
            role="status"
          >
            {toastMessage}
          </div>
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

function deriveTaskDirectories(
  workingDirectory: string,
  childDirectoryName?: string,
): TaskDirectories {
  const normalized = trimTrailingSlash(workingDirectory);
  const taskWorkingDirectory = childDirectoryName
    ? `${normalized}/${childDirectoryName}`
    : normalized;

  return {
    working_directory: taskWorkingDirectory,
    model_directory: `${taskWorkingDirectory}/models`,
    dataset_directory: `${taskWorkingDirectory}/datasets`,
    output_directory: `${taskWorkingDirectory}/outputs`,
    checkpoint_directory: `${taskWorkingDirectory}/checkpoints`,
  };
}

function trimTrailingSlash(path: string): string {
  return path.replace(/\/+$/, "") || path;
}

function SettingsDialog({
  draftWorkingDirectory,
  draftDevice,
  runtimeHardware,
  error,
  isSaving,
  onClose,
  onDraftDeviceChange,
  onSave,
}: {
  draftWorkingDirectory: string;
  draftDevice: DevicePreference;
  runtimeHardware: RuntimeHardwareResponse | null;
  error: string | null;
  isSaving: boolean;
  onClose: () => void;
  onDraftDeviceChange: (value: DevicePreference) => void;
  onSave: () => void;
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
              设置全局默认工作目录；模型由各个子应用单独选择。
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
            取消
          </button>
          <button
            className="inline-flex items-center gap-2 rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-cyan-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
            type="button"
            disabled={isSaving}
            onClick={onSave}
          >
            <Save className="size-4" aria-hidden="true" />
            保存设置
          </button>
        </div>
      </section>
    </div>
  );
}
