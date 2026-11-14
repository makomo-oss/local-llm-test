import {
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  ModelRegistry,
  TextStreamer,
  env,
  type DataType,
  type DeviceType,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type PretrainedModelOptions,
  type ProgressInfo,
} from '@huggingface/transformers';
import {
  getModelById,
  getResolvedDownloadBytes,
  parseThinkingOutput,
  resolveEffectiveDtype,
} from '../models/catalog.ts';
import { detectMobileDeviceContext } from '../telemetry/systemProbe.ts';
import type {
  DeviceOption,
  DownloadFileProgress,
  ExecutionBackend,
  InferenceSpeedMetrics,
  ModelLoadProgress,
  ModelLoadStage,
  ModelSpec,
  QuantDtype,
  ResolvedDtype,
  WorkerDeviceTarget,
  WorkerRequest,
  WorkerResponse,
} from '../types.ts';
import { TRANSFORMERS_CACHE_NAME, clearModelCache, inspectModelCaches } from './cacheManager.ts';
import {
  acquireGpuDeviceForPreference,
  getDeviceRuntimeDetails,
  probeComputeDevices,
  type AcquiredGpuDevice,
} from './deviceProbe.ts';
import {
  createSerializedCacheAdapter,
  filterCausalLmModelFiles,
  isOptionalModelFile,
  precacheModelFilesSequentially,
  resolveExternalDataAugmentedConfig,
} from './modelPrecacher.ts';
import { resolveOpfsRootDirectory, type OpfsDirectoryLike } from './opfsCacheStore.ts';
import { SessionLoadCoordinator, withSerializedInferenceSessions } from './sessionSerializer.ts';

const TRAILING_SPECIAL_DELIMITER_REGEX =
  /(?:(?:<turn\|>|<\|im_end\|>|<\|eot_id\|>|<\|end_of_text\|>|<end_of_turn>|<eos>|<\|endoftext\|>|<\/s>)\s*)+$/;

/**
 * Strip trailing special stop / turn delimiters emitted by chat models without touching
 * thinking delimiters (`<think>`, `</think>`, `<|channel>`, `<channel|>`).
 */
export function stripTrailingSpecialDelimiters(text: string): string {
  return text.replace(TRAILING_SPECIAL_DELIMITER_REGEX, '');
}

/**
 * Compute token throughput and latency breakdown metrics from per-token timestamps.
 */
export function computeSpeedMetrics(
  promptTokens: number,
  tokenTimestampsMs: readonly number[],
  startMs: number,
  endMs: number
): InferenceSpeedMetrics {
  const safePromptTokens =
    typeof promptTokens === 'number' && Number.isFinite(promptTokens) && promptTokens > 0
      ? Math.floor(promptTokens)
      : 0;
  const safeStartMs = typeof startMs === 'number' && Number.isFinite(startMs) ? startMs : 0;
  const safeEndMs = typeof endMs === 'number' && Number.isFinite(endMs) ? endMs : safeStartMs;
  const generatedTokens = tokenTimestampsMs.length;

  if (generatedTokens === 0) {
    return {
      ttftMs: 0,
      prefillMs: 0,
      decodeMs: 0,
      totalMs: Math.max(0, safeEndMs - safeStartMs),
      promptTokens: safePromptTokens,
      generatedTokens: 0,
      tokensPerSec: 0,
      decodeTokensPerSec: 0,
      interTokenLatencyAvgMs: 0,
      interTokenLatencyP95Ms: 0,
    };
  }

  const firstTokenMs = Math.max(safeStartMs, tokenTimestampsMs[0] ?? safeStartMs);
  const lastTokenMs = Math.max(firstTokenMs, tokenTimestampsMs[generatedTokens - 1] ?? firstTokenMs);
  const effectiveEndMs = Math.max(lastTokenMs, safeEndMs);

  const totalMs = Math.max(0, effectiveEndMs - safeStartMs);
  const ttftMs = Math.max(0, firstTokenMs - safeStartMs);
  const prefillMs = ttftMs;
  const decodeMs = Math.max(0, totalMs - ttftMs);

  const tokensPerSec = totalMs > 0 ? generatedTokens / (totalMs / 1000) : 0;
  const decodeTokensPerSec =
    generatedTokens > 1 && decodeMs > 0 ? (generatedTokens - 1) / (decodeMs / 1000) : 0;

  let interTokenLatencyAvgMs = 0;
  let interTokenLatencyP95Ms = 0;

  if (generatedTokens > 1) {
    const deltas: number[] = [];
    for (let i = 1; i < generatedTokens; i += 1) {
      const prev = tokenTimestampsMs[i - 1] ?? firstTokenMs;
      const curr = tokenTimestampsMs[i] ?? prev;
      deltas.push(Math.max(0, curr - prev));
    }

    const totalDelta = deltas.reduce((acc, delta) => acc + delta, 0);
    interTokenLatencyAvgMs = totalDelta / deltas.length;

    const sorted = [...deltas].sort((a, b) => a - b);
    const p95Index = Math.min(
      sorted.length - 1,
      Math.max(0, Math.ceil(0.95 * sorted.length) - 1)
    );
    interTokenLatencyP95Ms = sorted[p95Index] ?? 0;
  }

  return {
    ttftMs,
    prefillMs,
    decodeMs,
    totalMs,
    promptTokens: safePromptTokens,
    generatedTokens,
    tokensPerSec,
    decodeTokensPerSec,
    interTokenLatencyAvgMs,
    interTokenLatencyP95Ms,
  };
}

interface ProgressAggregator {
  handle(info: ProgressInfo): ModelLoadProgress | null;
  buildStageProgress(stage: ModelLoadStage, message: string, pctOverride?: number): ModelLoadProgress;
  registerExpectedFiles(files: readonly string[]): void;
}

/**
 * Aggregate per-file Transformers.js `progress_callback` events into normalized
 * `ModelLoadProgress` snapshots while ignoring `progress_total` multimodal HEAD overestimates.
 */
export function createDownloadProgressAggregator(
  model: ModelSpec,
  effectiveDtype: ResolvedDtype | QuantDtype
): ProgressAggregator {
  const fileMap = new Map<string, DownloadFileProgress>();
  const catalogExpectedBytes = getResolvedDownloadBytes(model, effectiveDtype, true);
  let lastEmitAtMs = -Infinity;
  let lastEmittedPct = -1;

  const snapshot = (
    stage: ModelLoadStage,
    message: string,
    pctOverride?: number
  ): ModelLoadProgress => {
    const files = Array.from(fileMap.values()).sort((a, b) => a.file.localeCompare(b.file));
    const loadedBytes = files.reduce((sum, item) => sum + item.loadedBytes, 0);
    const observedTotalBytes = files.reduce((sum, item) => sum + item.totalBytes, 0);
    const expectedTotalBytes = Math.max(catalogExpectedBytes, observedTotalBytes);

    let overallProgressPct: number;
    if (typeof pctOverride === 'number') {
      overallProgressPct = pctOverride;
    } else if (expectedTotalBytes > 0) {
      const rawPct = (loadedBytes / expectedTotalBytes) * 100;
      overallProgressPct = Math.min(99, Math.max(0, Math.round(rawPct * 10) / 10));
    } else {
      overallProgressPct = 0;
    }

    return {
      modelId: model.id,
      stage,
      message,
      overallProgressPct,
      loadedBytes,
      expectedTotalBytes,
      files,
    };
  };

  return {
    registerExpectedFiles(files) {
      for (const rawFile of files) {
        const file = rawFile.trim();
        if (file && !fileMap.has(file)) {
          fileMap.set(file, {
            file,
            status: 'initiate',
            loadedBytes: 0,
            totalBytes: 0,
            progressPct: 0,
          });
        }
      }
    },
    buildStageProgress(stage, message, pctOverride) {
      return snapshot(stage, message, pctOverride);
    },
    handle(info) {
      if (info.status === 'progress_total' || info.status === 'ready') {
        return null;
      }

      const file = info.file;
      if (typeof file !== 'string' || file.length === 0) {
        return null;
      }

      const existing = fileMap.get(file);
      let isTransition = false;

      if (info.status === 'initiate') {
        if (!existing) {
          fileMap.set(file, {
            file,
            status: 'initiate',
            loadedBytes: 0,
            totalBytes: 0,
            progressPct: 0,
          });
          isTransition = true;
        }
      } else if (info.status === 'download') {
        if (!existing || existing.status === 'initiate') {
          fileMap.set(file, {
            file,
            status: 'download',
            loadedBytes: existing?.loadedBytes ?? 0,
            totalBytes: existing?.totalBytes ?? 0,
            progressPct: existing?.progressPct ?? 0,
          });
          isTransition = true;
        }
      } else if (info.status === 'progress') {
        const loaded =
          typeof info.loaded === 'number' && Number.isFinite(info.loaded) && info.loaded >= 0
            ? info.loaded
            : (existing?.loadedBytes ?? 0);
        const total =
          typeof info.total === 'number' && Number.isFinite(info.total) && info.total > 0
            ? info.total
            : (existing?.totalBytes ?? 0);
        const pct =
          typeof info.progress === 'number' && Number.isFinite(info.progress)
            ? Math.min(100, Math.max(0, info.progress))
            : total > 0
              ? Math.min(100, Math.max(0, (loaded / total) * 100))
              : (existing?.progressPct ?? 0);

        fileMap.set(file, {
          file,
          status: 'progress',
          loadedBytes: Math.max(existing?.loadedBytes ?? 0, loaded),
          totalBytes: Math.max(existing?.totalBytes ?? 0, total),
          progressPct: Math.max(existing?.progressPct ?? 0, pct),
        });
      } else if (info.status === 'done') {
        const totalBytes = existing?.totalBytes ?? 0;
        fileMap.set(file, {
          file,
          status: 'done',
          loadedBytes: totalBytes > 0 ? totalBytes : (existing?.loadedBytes ?? 0),
          totalBytes,
          progressPct: 100,
        });
        isTransition = true;
      }

      const entries = Array.from(fileMap.values());
      const hasWeightsDone = entries.some(
        (entry) => entry.status === 'done' && /\.onnx(?:_data)?/i.test(entry.file)
      );
      const allDone = entries.length > 0 && entries.every((entry) => entry.status === 'done');

      if (info.status === 'done' && hasWeightsDone && allDone) {
        return snapshot('initializing-session', 'ONNXセッション・シェーダーを初期化中...', 99);
      }

      const progress = snapshot('downloading', `ダウンロード中: ${file}`);
      const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const pctDelta = Math.abs(progress.overallProgressPct - lastEmittedPct);

      if (!isTransition && now - lastEmitAtMs < 60 && pctDelta < 1) {
        return null;
      }

      lastEmitAtMs = now;
      lastEmittedPct = progress.overallProgressPct;
      return progress;
    },
  };
}

interface DedicatedWorkerScopeLike {
  crossOriginIsolated?: boolean;
  navigator?: Navigator;
  postMessage(message: WorkerResponse): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<WorkerRequest>) => void): void;
}

const workerScope: DedicatedWorkerScopeLike | null =
  typeof window === 'undefined' &&
  typeof self !== 'undefined' &&
  typeof (self as unknown as { postMessage?: unknown }).postMessage === 'function'
    ? (self as unknown as DedicatedWorkerScopeLike)
    : null;

function isIosOrMobileWebKit(): boolean {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : '';
  return /\b(?:iPhone|iPad|iPod|CriOS|FxiOS|EdgiOS|OPiOS)\b/i.test(ua);
}

interface ConfigureTransformersOptions {
  requestedWasmThreads?: number;
  backend?: ExecutionBackend;
  isConstrainedMobile?: boolean;
  opfsRoot?: OpfsDirectoryLike | null;
  coordinator?: SessionLoadCoordinator;
  signal?: AbortSignal;
}

function configureTransformersEnvironment(options: ConfigureTransformersOptions = {}): number {
  const runtimeDetails = getDeviceRuntimeDetails();
  const maxThreads = options.isConstrainedMobile ? 2 : 4;
  const targetThreads =
    runtimeDetails.crossOriginIsolated &&
    typeof options.requestedWasmThreads === 'number' &&
    Number.isFinite(options.requestedWasmThreads) &&
    options.requestedWasmThreads >= 1
      ? Math.min(maxThreads, Math.floor(options.requestedWasmThreads))
      : Math.min(maxThreads, runtimeDetails.numThreads);
  const activeThreads = runtimeDetails.crossOriginIsolated ? targetThreads : 1;

  env.allowLocalModels = false;
  env.useBrowserCache = true;
  env.useWasmCache = true;

  const serializedCache = createSerializedCacheAdapter(
    env.cacheKey || TRANSFORMERS_CACHE_NAME,
    undefined,
    {
      opfsRoot: options.opfsRoot,
      coordinator: options.coordinator,
      signal: options.signal,
    }
  );
  if (serializedCache) {
    env.customCache = serializedCache;
    env.useCustomCache = true;
  } else {
    env.useCustomCache = false;
    env.customCache = null;
  }

  const wasmEnv = env.backends?.onnx?.wasm as
    | {
        proxy?: boolean;
        simd?: boolean;
        numThreads?: number;
        wasmPaths?: string | { mjs?: string; wasm?: string };
        wasmBinary?: Uint8Array;
      }
    | undefined;
  if (wasmEnv) {
    wasmEnv.proxy = false;
    wasmEnv.simd = true;
    wasmEnv.numThreads = activeThreads;

    // On CPU (WASM) execution or iOS WebKit browsers (Safari/Chrome/Firefox/Edge on iOS),
    // avoid the 26.8 MiB asyncify WASM binary unless WebGPU actually requires it.
    const hasWebGpuApi = typeof navigator !== 'undefined' && 'gpu' in navigator;
    const preferNonAsyncify =
      options.backend === 'wasm' || (isIosOrMobileWebKit() && !hasWebGpuApi);
    if (preferNonAsyncify) {
      const onnxVersion = (
        env.backends?.onnx as { versions?: { web?: string } } | undefined
      )?.versions?.web;
      if (wasmEnv.wasmPaths && typeof wasmEnv.wasmPaths === 'object') {
        if (typeof wasmEnv.wasmPaths.mjs === 'string') {
          wasmEnv.wasmPaths.mjs = wasmEnv.wasmPaths.mjs.replace(/\.(?:asyncify|jsep)\./, '.');
        }
        if (typeof wasmEnv.wasmPaths.wasm === 'string') {
          wasmEnv.wasmPaths.wasm = wasmEnv.wasmPaths.wasm.replace(/\.(?:asyncify|jsep)\./, '.');
        }
      } else if (onnxVersion) {
        const prefix = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${onnxVersion}/dist/`;
        wasmEnv.wasmPaths = {
          mjs: `${prefix}ort-wasm-simd-threaded.mjs`,
          wasm: `${prefix}ort-wasm-simd-threaded.wasm`,
        };
      }
    }
  }

  return activeThreads;
}

function yieldToEventLoop(delayMs = 0): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, delayMs);
  });
}

function postWorkerMessage(message: WorkerResponse): void {
  workerScope?.postMessage(message);
}

function formatWorkerError(err: unknown): string {
  if (err instanceof Error) {
    return err.message || err.name;
  }
  if (typeof err === 'string') {
    return err;
  }
  try {
    return JSON.stringify(err);
  } catch {
    return '不明なワーカーエラーが発生しました。';
  }
}

export function classifyModelLoadError(err: unknown): { code: string; message: string } {
  const rawMessage = formatWorkerError(err);
  const explicitCode =
    err && typeof err === 'object' ? (err as { code?: unknown }).code : undefined;
  const errName = err instanceof Error ? err.name : '';

  if (
    explicitCode === 'STORAGE_QUOTA_EXCEEDED' ||
    errName === 'QuotaExceededError' ||
    errName === 'NS_ERROR_DOM_QUOTA_REACHED' ||
    /quota|storage limit/i.test(rawMessage)
  ) {
    return {
      code: 'STORAGE_QUOTA_EXCEEDED',
      message: `ブラウザの保存容量 (OPFS / CacheStorage) が不足したためモデルを保存できませんでした: ${rawMessage}。不要なキャッシュを削除して空き容量を確保してください。`,
    };
  }

  if (explicitCode === 'STORAGE_WRITE_FAILED') {
    return {
      code: 'STORAGE_WRITE_FAILED',
      message: `モデルファイルの保存 (OPFS / CacheStorage) に失敗しました: ${rawMessage}。通常ブラウザタブで空き容量を確保のうえ再試行してください。`,
    };
  }

  if (
    explicitCode === 'WEBGPU_DEVICE_LOST' ||
    /WebGPU デバイスが失われました|device is lost|device lost/i.test(rawMessage)
  ) {
    return {
      code: 'WEBGPU_DEVICE_LOST',
      message: `GPU ドライバまたはメモリ制限により WebGPU デバイスがリセットされました (${rawMessage})。CPU (WASM SIMD) またはより軽量なモデルをお試しください。`,
    };
  }

  if (
    /out of memory|bad_alloc|failed to allocate|cannot allocate|allocation failed|memory access out of bounds|invalid array length|oom/i.test(
      rawMessage
    )
  ) {
    return {
      code: 'OUT_OF_MEMORY',
      message: `メモリ不足のためモデルの初期化に失敗しました (${rawMessage})。スマホ・低メモリ環境では Qwen 3.5 0.8B (WebGPU) またはより小さい量子化をお試しください。`,
    };
  }

  return {
    code: 'MODEL_LOAD_FAILED',
    message: `モデルのロードに失敗しました: ${rawMessage}`,
  };
}

function buildWorkerDeviceOption(
  target: WorkerDeviceTarget,
  acquired: AcquiredGpuDevice | null,
  activeThreads: number
): DeviceOption {
  const runtimeDetails = getDeviceRuntimeDetails();
  if (target.backend === 'webgpu' && acquired) {
    const features =
      acquired.grantedFeatures.length > 0 ? acquired.grantedFeatures : acquired.features;
    const isFallback = acquired.adapterInfo.isFallbackAdapter || target.gpuPreference === 'fallback';
    const gpuName =
      acquired.adapterInfo.description ||
      [acquired.adapterInfo.vendor, acquired.adapterInfo.architecture, acquired.adapterInfo.device]
        .filter((part) => part.length > 0)
        .join(' ') ||
      (isFallback ? 'Software Adapter' : 'WebGPU Adapter');

    const label = isFallback
      ? `WebGPU Fallback (ソフトウェア): ${gpuName}`
      : target.gpuPreference === 'high-performance'
        ? `WebGPU (High Performance): ${gpuName}`
        : target.gpuPreference === 'low-power'
          ? `WebGPU (Low Power): ${gpuName}`
          : `WebGPU: ${gpuName}`;

    return {
      id: `webgpu-${acquired.gpuPreference}`,
      label,
      backend: 'webgpu',
      gpuPreference: acquired.gpuPreference,
      fingerprint: acquired.fingerprint,
      adapterInfo: acquired.adapterInfo,
      features,
      limits: acquired.limits,
      supportsFp16: acquired.supportsFp16,
      recommendedDtype: acquired.supportsFp16 ? 'q4f16' : 'q4',
      available: true,
      runtimeDetails,
    };
  }

  const features: string[] = [];
  if (runtimeDetails.simdSupported) {
    features.push('wasm-simd');
  }
  if (activeThreads > 1) {
    features.push('threads');
  }

  const simdLabel = runtimeDetails.simdSupported ? ' SIMD' : '';
  const coiSuffix = runtimeDetails.crossOriginIsolated ? '' : ' / 非COI';
  return {
    id: 'wasm-cpu',
    label: `CPU (WASM${simdLabel}, ${activeThreads}スレッド${coiSuffix})`,
    backend: 'wasm',
    features,
    limits: {},
    supportsFp16: false,
    recommendedDtype: 'q4',
    available: true,
    runtimeDetails: {
      ...runtimeDetails,
      numThreads: activeThreads,
    },
  };
}

function sampleWorkerJsHeap(): { usedBytes?: number; totalBytes?: number } {
  const perf = (
    globalThis as {
      performance?: {
        memory?: {
          usedJSHeapSize?: number;
          totalJSHeapSize?: number;
        };
      };
    }
  ).performance;
  const mem = perf?.memory;
  if (!mem) {
    return {};
  }
  const usedBytes =
    typeof mem.usedJSHeapSize === 'number' && Number.isFinite(mem.usedJSHeapSize)
      ? mem.usedJSHeapSize
      : undefined;
  const totalBytes =
    typeof mem.totalJSHeapSize === 'number' && Number.isFinite(mem.totalJSHeapSize)
      ? mem.totalJSHeapSize
      : undefined;
  return {
    ...(usedBytes !== undefined ? { usedBytes } : {}),
    ...(totalBytes !== undefined ? { totalBytes } : {}),
  };
}

interface WorkerRuntimeState {
  model: PreTrainedModel | null;
  tokenizer: PreTrainedTokenizer | null;
  modelSpec: ModelSpec | null;
  deviceOption: DeviceOption | null;
  effectiveDtype: QuantDtype | null;
  gpuDevice: GPUDevice | null;
  activeThreads: number;
  busyOperation: 'load-model' | 'generate' | 'unload-model' | null;
  activeGenerationRequestId: string | null;
  activeStopper: InterruptableStoppingCriteria | null;
  abortRequested: boolean;
  deviceLostError: string | null;
}

const state: WorkerRuntimeState = {
  model: null,
  tokenizer: null,
  modelSpec: null,
  deviceOption: null,
  effectiveDtype: null,
  gpuDevice: null,
  activeThreads: 1,
  busyOperation: null,
  activeGenerationRequestId: null,
  activeStopper: null,
  abortRequested: false,
  deviceLostError: null,
};

const intentionallyDestroyedGpuDevices = new WeakSet<GPUDevice>();

async function disposeLoadedRuntime(): Promise<void> {
  const prevModel = state.model;
  const prevGpuDevice = state.gpuDevice;

  state.model = null;
  state.tokenizer = null;
  state.modelSpec = null;
  state.deviceOption = null;
  state.effectiveDtype = null;
  state.gpuDevice = null;
  state.deviceLostError = null;

  if (prevModel) {
    try {
      await prevModel.dispose();
    } catch {
      // Ignore model disposal error when tearing down previous sessions.
    }
  }

  if (prevGpuDevice) {
    intentionallyDestroyedGpuDevices.add(prevGpuDevice);
    try {
      prevGpuDevice.destroy();
    } catch {
      // Ignore GPUDevice destroy error if already lost/destroyed.
    }
  }
}

async function handleProbeDevices(requestId: string): Promise<void> {
  try {
    state.activeThreads = configureTransformersEnvironment();
    const devices = await probeComputeDevices();
    postWorkerMessage({
      type: 'devices-probed',
      requestId,
      devices,
    });
  } catch (err) {
    postWorkerMessage({
      type: 'error',
      requestId,
      message: `デバイス検出に失敗しました: ${formatWorkerError(err)}`,
      code: 'PROBE_FAILED',
      recoverable: true,
    });
  }
}

async function handleLoadModel(
  req: Extract<WorkerRequest, { type: 'load-model' }>
): Promise<void> {
  if (state.busyOperation) {
    postWorkerMessage({
      type: 'error',
      requestId: req.requestId,
      message: `別の処理 (${state.busyOperation}) が実行中のためモデルをロードできません。`,
      code: 'CONCURRENT_OPERATION',
      recoverable: true,
    });
    return;
  }

  const modelSpec = getModelById(req.modelId);
  if (!modelSpec) {
    postWorkerMessage({
      type: 'error',
      requestId: req.requestId,
      message: `未知のモデルIDです: ${req.modelId}`,
      code: 'UNKNOWN_MODEL',
      recoverable: true,
    });
    return;
  }

  state.busyOperation = 'load-model';
  const loadStartMs = performance.now();
  let acquiredGpu: AcquiredGpuDevice | null = null;
  let loadedModel: PreTrainedModel | null = null;
  const sessionCoordinator = new SessionLoadCoordinator();
  const loadAbortController = new AbortController();

  try {
    await disposeLoadedRuntime();

    const isConstrainedMobile =
      req.mobileContext?.isMobile ?? detectMobileDeviceContext().isMobile;
    const opfsRoot = await resolveOpfsRootDirectory();
    const activeThreads = configureTransformersEnvironment({
      requestedWasmThreads:
        req.deviceTarget.backend === 'wasm' ? req.deviceTarget.numThreads : undefined,
      backend: req.deviceTarget.backend,
      isConstrainedMobile,
      opfsRoot,
      coordinator: sessionCoordinator,
      signal: loadAbortController.signal,
    });
    state.activeThreads = activeThreads;

    postWorkerMessage({
      type: 'load-progress',
      requestId: req.requestId,
      progress: {
        modelId: modelSpec.id,
        stage: 'probing-device',
        message:
          req.deviceTarget.backend === 'webgpu'
            ? 'WebGPU アダプターと GPUDevice を取得中...'
            : 'WASM (CPU) ランタイムを初期化中...',
        overallProgressPct: 0,
        loadedBytes: 0,
        expectedTotalBytes: 0,
        files: [],
      },
    });

    const notices: string[] = [];
    if (req.deviceTarget.backend === 'webgpu') {
      acquiredGpu = await acquireGpuDeviceForPreference(req.deviceTarget.gpuPreference, {
        expectedDeviceFingerprint: req.deviceTarget.expectedDeviceFingerprint,
      });

      const boundDevice = acquiredGpu.device;
      state.gpuDevice = boundDevice;

      const webgpuEnv = (env.backends?.onnx as { webgpu?: Record<string, unknown> } | undefined)
        ?.webgpu;
      if (webgpuEnv && typeof webgpuEnv === 'object') {
        webgpuEnv.adapter = acquiredGpu.adapter;
        webgpuEnv.device = acquiredGpu.device;
        webgpuEnv.powerPreference = acquiredGpu.gpuPreference;
      }

      if (boundDevice.lost && typeof boundDevice.lost.then === 'function') {
        void boundDevice.lost.then((lostInfo) => {
          if (intentionallyDestroyedGpuDevices.has(boundDevice)) {
            return;
          }
          const reason = lostInfo?.reason ? ` (${lostInfo.reason})` : '';
          const detail = lostInfo?.message ? `: ${lostInfo.message}` : '';
          const lossMessage = `WebGPU デバイスが失われました${reason}${detail}`;
          state.deviceLostError = lossMessage;
          loadAbortController.abort(new Error(lossMessage));

          if (state.activeGenerationRequestId && state.activeStopper) {
            state.activeStopper.interrupt();
          }

          if (state.gpuDevice === boundDevice) {
            postWorkerMessage({
              type: 'error',
              requestId: state.activeGenerationRequestId ?? 'device-lost',
              message: lossMessage,
              code: 'WEBGPU_DEVICE_LOST',
              recoverable: true,
            });
          }
        });
      }

      if (!acquiredGpu.fingerprintMatched) {
        notices.push(
          '選択された WebGPU アダプターと異なるデバイスが割り当てられました（環境により優先アダプターが自動選択されました）。'
        );
      }
    }

    const deviceOption = buildWorkerDeviceOption(req.deviceTarget, acquiredGpu, activeThreads);
    const resolved = resolveEffectiveDtype(modelSpec, deviceOption, req.dtype ?? 'auto', {
      useExternalDataFormat: req.useExternalDataFormat,
    });

    if (resolved.adjusted && resolved.reason) {
      notices.push(resolved.reason);
    }

    const aggregator = createDownloadProgressAggregator(modelSpec, resolved);
    let sessionStageEmitted = false;

    const progressCallback = (info: ProgressInfo): void => {
      const progress = aggregator.handle(info);
      if (!progress) {
        return;
      }
      if (progress.stage === 'initializing-session') {
        sessionStageEmitted = true;
      }
      postWorkerMessage({
        type: 'load-progress',
        requestId: req.requestId,
        progress,
      });
    };

    postWorkerMessage({
      type: 'load-progress',
      requestId: req.requestId,
      progress: aggregator.buildStageProgress(
        'downloading',
        `${modelSpec.name} (${resolved.dtype}) のトークナイザーとモデルを取得中...`,
        0
      ),
    });

    // Yield briefly after disposing previous GPU/WASM resources so buffers can be reclaimed before download.
    await yieldToEventLoop(40);

    const tokenizer = await AutoTokenizer.from_pretrained(modelSpec.repoId, {
      progress_callback: progressCallback,
    });

    // Download uncached ONNX files strictly one-by-one into persistent OPFS / CacheStorage
    // so mobile browsers never buffer parallel multi-hundred-MiB weights in RAM.
    let precachedInStorage = false;
    {
      const config = (await AutoConfig.from_pretrained(modelSpec.repoId)) as unknown as Record<
        string,
        unknown
      >;
      const effectiveConfig = resolveExternalDataAugmentedConfig(
        config,
        resolved.useExternalDataFormat === true
      );
      const effectiveDtypeArg = (resolved.modelDtype ?? resolved.dtype) as
        | DataType
        | Record<string, DataType>;
      const rawModelFiles = await ModelRegistry.get_model_files(modelSpec.repoId, {
        config: effectiveConfig as never,
        dtype: effectiveDtypeArg,
        device: deviceOption.backend as DeviceType,
      });
      const modelFiles = filterCausalLmModelFiles(rawModelFiles, modelSpec.architecture);
      aggregator.registerExpectedFiles(modelFiles.filter((file) => !isOptionalModelFile(file)));
      const precacheResult = await precacheModelFilesSequentially({
        repoId: modelSpec.repoId,
        files: modelFiles,
        cacheName: env.cacheKey || TRANSFORMERS_CACHE_NAME,
        remoteHost: env.remoteHost,
        remotePathTemplate: env.remotePathTemplate,
        opfsRoot,
        signal: loadAbortController.signal,
        onFileProgress: (info) => {
          progressCallback(info as ProgressInfo);
        },
      });
      precachedInStorage = precacheResult.usedBrowserCache;
    }

    if (precachedInStorage) {
      await yieldToEventLoop(40);
      sessionStageEmitted = true;
      postWorkerMessage({
        type: 'load-progress',
        requestId: req.requestId,
        progress: aggregator.buildStageProgress(
          'initializing-session',
          'ダウンロード完了・推論エンジン (ONNX) を初期化中...',
          99
        ),
      });
      await yieldToEventLoop(0);
    }

    const baseSessionOptions: Record<string, unknown> = isConstrainedMobile
      ? { enableCpuMemArena: false, enableMemPattern: false }
      : {};

    const modelOptions: PretrainedModelOptions = {
      device: deviceOption.backend,
      dtype: (resolved.modelDtype ?? resolved.dtype) as PretrainedModelOptions['dtype'],
      // When precached, omit progress_callback so Transformers.js skips DefaultProgressCallback,
      // which would otherwise pin every resolved Uint8Array weight buffer in .loads until load completes.
      ...(precachedInStorage ? {} : { progress_callback: progressCallback }),
      ...(resolved.useExternalDataFormat === true ? { use_external_data_format: true } : {}),
      ...(deviceOption.backend === 'webgpu' && acquiredGpu
        ? {
            session_options: {
              ...baseSessionOptions,
              executionProviders: [
                {
                  name: 'webgpu',
                  device: acquiredGpu.device,
                },
              ],
            },
          }
        : isConstrainedMobile
          ? { session_options: baseSessionOptions }
          : {}),
    };

    loadedModel = await withSerializedInferenceSessions(sessionCoordinator, () =>
      AutoModelForCausalLM.from_pretrained(modelSpec.repoId, modelOptions)
    );

    // Release cached WASM Uint8Array binary once WebAssembly module has been compiled/instantiated.
    const wasmEnvAfterLoad = env.backends?.onnx?.wasm as { wasmBinary?: Uint8Array } | undefined;
    if (wasmEnvAfterLoad && wasmEnvAfterLoad.wasmBinary) {
      wasmEnvAfterLoad.wasmBinary = undefined;
    }

    if (state.deviceLostError) {
      throw new Error(state.deviceLostError);
    }

    if (!sessionStageEmitted) {
      postWorkerMessage({
        type: 'load-progress',
        requestId: req.requestId,
        progress: aggregator.buildStageProgress(
          'initializing-session',
          'ONNX推論セッションを準備中...',
          99
        ),
      });
    }

    state.model = loadedModel;
    state.tokenizer = tokenizer;
    state.modelSpec = modelSpec;
    state.deviceOption = deviceOption;
    state.effectiveDtype = resolved.dtype;

    postWorkerMessage({
      type: 'load-progress',
      requestId: req.requestId,
      progress: aggregator.buildStageProgress('ready', `${modelSpec.name} の準備が完了しました。`, 100),
    });

    const fallbackNotice = notices.length > 0 ? notices.join(' ') : undefined;
    postWorkerMessage({
      type: 'model-loaded',
      requestId: req.requestId,
      modelId: modelSpec.id,
      device: deviceOption,
      effectiveDtype: resolved.dtype,
      useExternalDataFormat: resolved.useExternalDataFormat === true,
      loadTimeMs: Math.max(0, performance.now() - loadStartMs),
      ...(fallbackNotice ? { fallbackNotice } : {}),
    });
  } catch (err) {
    loadAbortController.abort(err);
    sessionCoordinator.abortAll(err);
    if (loadedModel) {
      try {
        await loadedModel.dispose();
      } catch {
        // Ignore cleanup failure on partial load.
      }
    }
    await disposeLoadedRuntime();

    const classified = classifyModelLoadError(err);
    postWorkerMessage({
      type: 'error',
      requestId: req.requestId,
      message: classified.message,
      code: classified.code,
      recoverable: true,
    });
  } finally {
    state.busyOperation = null;
  }
}

async function handleGenerate(req: Extract<WorkerRequest, { type: 'generate' }>): Promise<void> {
  if (state.busyOperation) {
    postWorkerMessage({
      type: 'error',
      requestId: req.requestId,
      message: `別の処理 (${state.busyOperation}) が実行中のため生成を開始できません。`,
      code: 'CONCURRENT_OPERATION',
      recoverable: true,
    });
    return;
  }

  if (!state.model || !state.tokenizer || !state.modelSpec || !state.deviceOption || !state.effectiveDtype) {
    postWorkerMessage({
      type: 'error',
      requestId: req.requestId,
      message: 'モデルがロードされていません。先にモデルをロードしてください。',
      code: 'MODEL_NOT_LOADED',
      recoverable: true,
    });
    return;
  }

  if (state.deviceLostError) {
    postWorkerMessage({
      type: 'error',
      requestId: req.requestId,
      message: state.deviceLostError,
      code: 'WEBGPU_DEVICE_LOST',
      recoverable: true,
    });
    return;
  }

  const model = state.model;
  const tokenizer = state.tokenizer;
  const modelSpec = state.modelSpec;
  const backend: ExecutionBackend = state.deviceOption.backend;
  const dtype: QuantDtype = state.effectiveDtype;

  state.busyOperation = 'generate';
  state.activeGenerationRequestId = req.requestId;
  state.abortRequested = false;

  const stopper = new InterruptableStoppingCriteria();
  state.activeStopper = stopper;

  let tokenized: {
    input_ids: { dims?: number[]; dispose?: () => void };
    attention_mask?: { dispose?: () => void };
    [key: string]: unknown;
  } | null = null;
  let output: unknown = null;

  try {
    const templateOptions: NonNullable<Parameters<PreTrainedTokenizer['apply_chat_template']>[1]> & {
      tokenize?: true;
      return_tensor?: true;
      return_dict?: true;
      enable_thinking?: boolean;
    } = {
      add_generation_prompt: true,
      tokenize: true,
      return_tensor: true,
      return_dict: true,
      ...(modelSpec.supportsThinking ? { enable_thinking: req.config.enableThinking } : {}),
    };

    let thinkingExpectedInPrompt = false;
    if (modelSpec.family === 'qwen3_5' && req.config.enableThinking) {
      try {
        const renderedPrompt = tokenizer.apply_chat_template(req.messages, {
          ...templateOptions,
          tokenize: false,
        });
        thinkingExpectedInPrompt =
          typeof renderedPrompt === 'string' && /<think>\s*$/.test(renderedPrompt);
      } catch {
        thinkingExpectedInPrompt = false;
      }
    }

    const encoded = tokenizer.apply_chat_template<true, true, true>(req.messages, templateOptions);
    tokenized = encoded as unknown as typeof tokenized;
    const promptTokens = Number(encoded.input_ids?.dims?.at(-1) ?? 0);

    postWorkerMessage({
      type: 'generation-started',
      requestId: req.requestId,
      promptTokens,
    });

    const tokenTimestampsMs: number[] = [];
    let rawAccumulatedText = '';
    let emittedCleanText = '';
    let firstTokenEventSent = false;
    let putActive = false;
    let putTokenRecorded = false;
    let putChunkBuffer = '';

    // Keep Gemma 4 special channel tokens (`<|channel>`, `<channel|>`) intact for thinking split;
    // strip trailing stop/turn delimiters via `stripTrailingSpecialDelimiters` in both modes.
    const skipSpecialTokens = modelSpec.family !== 'gemma4';

    const flushCleanChunk = (chunkText: string, tokenTimestampMs: number): boolean => {
      rawAccumulatedText += chunkText;
      const cleanFullText = stripTrailingSpecialDelimiters(rawAccumulatedText);
      const deltaText =
        cleanFullText.length > emittedCleanText.length
          ? cleanFullText.slice(emittedCleanText.length)
          : '';
      emittedCleanText = cleanFullText;

      if (deltaText.length === 0) {
        return false;
      }

      const isFirstToken = !firstTokenEventSent;
      firstTokenEventSent = true;
      postWorkerMessage({
        type: 'token',
        requestId: req.requestId,
        deltaText,
        fullText: emittedCleanText,
        tokenCount: tokenTimestampsMs.length,
        timestampMs: tokenTimestampMs,
        isFirstToken,
      });
      return true;
    };

    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: skipSpecialTokens,
      token_callback_function: () => {
        const now = performance.now();
        tokenTimestampsMs.push(now);
        putTokenRecorded = true;
      },
      callback_function: (textChunk: string) => {
        if (putActive) {
          putChunkBuffer += textChunk;
          return;
        }
        const ts = tokenTimestampsMs[tokenTimestampsMs.length - 1] ?? performance.now();
        flushCleanChunk(textChunk, ts);
      },
    });

    const origPut = streamer.put.bind(streamer);
    streamer.put = (value: bigint[][]): void => {
      putActive = true;
      putTokenRecorded = false;
      putChunkBuffer = '';
      try {
        origPut(value);
      } finally {
        putActive = false;
      }

      if (putTokenRecorded) {
        const latestTimestamp = tokenTimestampsMs[tokenTimestampsMs.length - 1] ?? performance.now();
        const emitted =
          putChunkBuffer.length > 0 ? flushCleanChunk(putChunkBuffer, latestTimestamp) : false;

        // Emit immediate token event on buffered subwords (e.g. token #1 before first space)
        // so live TTFT and token counters update without waiting for word boundaries.
        if (!emitted) {
          const isFirstToken = !firstTokenEventSent;
          firstTokenEventSent = true;
          postWorkerMessage({
            type: 'token',
            requestId: req.requestId,
            deltaText: '',
            fullText: emittedCleanText,
            tokenCount: tokenTimestampsMs.length,
            timestampMs: latestTimestamp,
            isFirstToken,
          });
        }
      }
      putChunkBuffer = '';
    };

    const startMs = performance.now();
    output = await model.generate({
      ...encoded,
      max_new_tokens: req.config.maxNewTokens,
      do_sample: req.config.doSample,
      ...(req.config.doSample
        ? {
            temperature: req.config.temperature,
            top_p: req.config.topP,
            top_k: req.config.topK,
          }
        : {}),
      streamer,
      stopping_criteria: stopper,
      return_dict_in_generate: true,
    });
    const endMs = performance.now();

    if (state.deviceLostError) {
      throw new Error(state.deviceLostError);
    }

    const fullText = stripTrailingSpecialDelimiters(rawAccumulatedText);
    const parsed = parseThinkingOutput(fullText, modelSpec.family, {
      thinkingExpectedInPrompt,
    });
    const speed = computeSpeedMetrics(promptTokens, tokenTimestampsMs, startMs, endMs);
    const heap = sampleWorkerJsHeap();

    postWorkerMessage({
      type: 'generation-complete',
      requestId: req.requestId,
      fullText,
      thinking: parsed.thinking,
      content: parsed.content,
      aborted: stopper.interrupted || state.abortRequested,
      speed,
      ...(heap.usedBytes !== undefined ? { workerJsHeapUsedBytes: heap.usedBytes } : {}),
      ...(heap.totalBytes !== undefined ? { workerJsHeapTotalBytes: heap.totalBytes } : {}),
      activeThreads: state.activeThreads,
      backend,
      dtype,
    });
  } catch (err) {
    postWorkerMessage({
      type: 'error',
      requestId: req.requestId,
      message: state.deviceLostError ?? `推論の実行中にエラーが発生しました: ${formatWorkerError(err)}`,
      code: state.deviceLostError ? 'WEBGPU_DEVICE_LOST' : 'GENERATION_FAILED',
      recoverable: true,
    });
  } finally {
    if (output && typeof output === 'object' && 'past_key_values' in output) {
      const pastKv = (
        output as { past_key_values?: { dispose?: () => Promise<void> | void } | null }
      ).past_key_values;
      try {
        await pastKv?.dispose?.();
      } catch {
        // Ignore KV cache disposal failure during cleanup.
      }
    }

    if (tokenized && typeof tokenized === 'object') {
      const tensorInputs = tokenized as {
        input_ids?: { dispose?: () => void };
        attention_mask?: { dispose?: () => void };
      };
      try {
        tensorInputs.input_ids?.dispose?.();
        tensorInputs.attention_mask?.dispose?.();
      } catch {
        // Ignore input tensor disposal failure.
      }
    }

    state.activeStopper = null;
    state.activeGenerationRequestId = null;
    state.abortRequested = false;
    state.busyOperation = null;
  }
}

async function handleUnloadModel(requestId: string): Promise<void> {
  if (state.busyOperation) {
    postWorkerMessage({
      type: 'error',
      requestId,
      message: `別の処理 (${state.busyOperation}) が実行中のためアンロードできません。`,
      code: 'CONCURRENT_OPERATION',
      recoverable: true,
    });
    return;
  }

  state.busyOperation = 'unload-model';
  try {
    await disposeLoadedRuntime();
    postWorkerMessage({
      type: 'model-unloaded',
      requestId,
    });
  } catch (err) {
    postWorkerMessage({
      type: 'error',
      requestId,
      message: `モデルのアンロードに失敗しました: ${formatWorkerError(err)}`,
      code: 'UNLOAD_FAILED',
      recoverable: true,
    });
  } finally {
    state.busyOperation = null;
  }
}

async function handleInspectCache(
  req: Extract<WorkerRequest, { type: 'inspect-cache' }>
): Promise<void> {
  try {
    const models = req.modelIds
      ? req.modelIds
          .map((id) => getModelById(id))
          .filter((spec): spec is ModelSpec => spec !== undefined)
      : undefined;
    const caches = await inspectModelCaches(models);
    postWorkerMessage({
      type: 'cache-info',
      requestId: req.requestId,
      caches,
    });
  } catch (err) {
    postWorkerMessage({
      type: 'error',
      requestId: req.requestId,
      message: `キャッシュの確認に失敗しました: ${formatWorkerError(err)}`,
      code: 'CACHE_INSPECT_FAILED',
      recoverable: true,
    });
  }
}

async function handleClearCache(
  req: Extract<WorkerRequest, { type: 'clear-cache' }>
): Promise<void> {
  try {
    const result = await clearModelCache(req.modelId);
    postWorkerMessage({
      type: 'cache-cleared',
      requestId: req.requestId,
      ...(req.modelId !== undefined ? { modelId: req.modelId } : {}),
      clearedBytes: result.clearedBytes,
      deletedEntries: result.deletedEntries,
    });
  } catch (err) {
    postWorkerMessage({
      type: 'error',
      requestId: req.requestId,
      message: `キャッシュの削除に失敗しました: ${formatWorkerError(err)}`,
      code: 'CACHE_CLEAR_FAILED',
      recoverable: true,
    });
  }
}

function handleAbort(req: Extract<WorkerRequest, { type: 'abort' }>): void {
  if (!state.activeGenerationRequestId) {
    return;
  }
  if (!req.targetRequestId || req.targetRequestId === state.activeGenerationRequestId) {
    state.abortRequested = true;
    state.activeStopper?.interrupt();
  }
}

async function dispatchWorkerRequest(req: WorkerRequest): Promise<void> {
  switch (req.type) {
    case 'abort':
      handleAbort(req);
      return;
    case 'probe-devices':
      await handleProbeDevices(req.requestId);
      return;
    case 'load-model':
      await handleLoadModel(req);
      return;
    case 'generate':
      await handleGenerate(req);
      return;
    case 'unload-model':
      await handleUnloadModel(req.requestId);
      return;
    case 'inspect-cache':
      await handleInspectCache(req);
      return;
    case 'clear-cache':
      await handleClearCache(req);
      return;
  }
}

if (workerScope) {
  configureTransformersEnvironment();
  workerScope.addEventListener('message', (event) => {
    const request = event?.data;
    if (!request || typeof request !== 'object' || typeof request.type !== 'string') {
      return;
    }
    void dispatchWorkerRequest(request);
  });
}
