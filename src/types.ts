/**
 * Supported local LLM identifiers in the application catalog.
 */
export type ModelId =
  | 'gemma-4-e4b'
  | 'gemma-4-e2b'
  | 'qwen-3.5-4b'
  | 'qwen-3.5-2b'
  | 'qwen-3.5-0.8b'
  | 'llama-3.2-3b';

/**
 * Architectural model family used for chat templating and thinking tag parsing.
 */
export type ModelFamily = 'gemma4' | 'qwen3_5' | 'llama';

/**
 * ONNX weight quantization / precision dtype options supported by Transformers.js v4.
 */
export type QuantDtype = 'q4f16' | 'q4' | 'q8' | 'fp16';

/**
 * Runtime ONNX session dtypes accepted by Transformers.js `from_pretrained`,
 * including `fp32` used for per-session overrides on CPU (WASM).
 */
export type TransformersDtype = QuantDtype | 'fp32';

/**
 * Execution backends available in the browser.
 */
export type ExecutionBackend = 'webgpu' | 'wasm';

/**
 * Probe preference passed to navigator.gpu.requestAdapter.
 */
export type GpuPreference = 'high-performance' | 'low-power' | 'default' | 'fallback';

/**
 * Serializable snapshot of WebGPU GPUAdapterInfo.
 */
export interface AdapterInfoSnapshot {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallbackAdapter: boolean;
  subgroupMinSize?: number;
  subgroupMaxSize?: number;
}

/**
 * Browser / worker runtime capability flags relevant to WASM and multi-threading.
 */
export interface DeviceRuntimeDetails {
  crossOriginIsolated: boolean;
  hardwareConcurrency: number;
  simdSupported: boolean;
  numThreads: number;
}

/**
 * Selectable compute device discovered during probing (WebGPU GPU or CPU WASM).
 */
export interface DeviceOption {
  id: string;
  label: string;
  backend: ExecutionBackend;
  gpuPreference?: GpuPreference;
  fingerprint?: string;
  adapterInfo?: AdapterInfoSnapshot;
  features: string[];
  limits: Record<string, number>;
  supportsFp16: boolean;
  recommendedDtype: QuantDtype;
  available: boolean;
  unavailableReason?: string;
  runtimeDetails: DeviceRuntimeDetails;
}

/**
 * Catalog entry describing an unauthenticated Hugging Face ONNX model.
 */
export interface ModelSpec {
  id: ModelId;
  name: string;
  family: ModelFamily;
  repoId: string;
  parameterLabel: string;
  architecture: string;
  causalLmClass: string;
  contextLength: number;
  license: string;
  supportsThinking: boolean;
  defaultThinkingEnabled: boolean;
  /**
   * True when the repo's config.json omits `model_quantized.onnx` from
   * `transformers.js_config.use_external_data_format`, requiring explicit
   * `use_external_data_format: true` to load `q8`.
   */
  requiresExplicitExternalDataForQ8: boolean;
  availableDtypes: readonly QuantDtype[];
  tokenizerBytes: number;
  textOnlyBytesByDtype: Record<QuantDtype, number>;
  fullMultimodalBytesByDtype?: Partial<Record<QuantDtype, number>>;
  kvCacheFp16OnQ4f16: boolean;
  descriptionJa: string;
  tags: readonly string[];
}

/**
 * Effective quantization resolution result for a (model, device, requestedDtype) tuple.
 */
export interface ResolvedDtype {
  dtype: QuantDtype;
  /**
   * Actual value passed to Transformers.js `from_pretrained({ dtype })`.
   * May be a per-session dtype map (e.g. `{ embed_tokens: 'fp16', decoder_model_merged: 'q4' }`)
   * when a backend requires avoiding quantized ops such as `GatherBlockQuantized` on WASM.
   * Defaults to `dtype` when omitted.
   */
  modelDtype?: TransformersDtype | Record<string, TransformersDtype>;
  /**
   * Total weight bytes corresponding to `modelDtype` when per-session overrides alter download size.
   * Falls back to `model.textOnlyBytesByDtype[dtype]` when omitted.
   */
  effectiveWeightsBytes?: number;
  requestedDtype?: QuantDtype | 'auto';
  /**
   * Defined as `true` only when `from_pretrained` must override `config.json`
   * (e.g., Llama 3.2 `q8`). Must remain `undefined` otherwise so `config.json`'s
   * per-file external data chunk map is preserved.
   */
  useExternalDataFormat?: true;
  adjusted: boolean;
  reason?: string;
}

/**
 * Parsed separation of reasoning trace and final user-facing answer.
 */
export interface ParsedThinkingOutput {
  thinking: string;
  content: string;
  hasThinking: boolean;
  isThinkingInProgress: boolean;
}

export type ChatRole = 'system' | 'user' | 'assistant';

/**
 * Minimal prompt turn passed to tokenizer.apply_chat_template inside the worker.
 */
export interface ChatPromptMessage {
  role: ChatRole;
  content: string;
}

/**
 * Full chat message stored in UI history and benchmark transcripts.
 */
export interface ChatMessage extends ChatPromptMessage {
  id: string;
  thinking?: string;
  createdAt: number;
  modelId?: ModelId;
  deviceId?: string;
  dtype?: QuantDtype;
  metrics?: InferenceMetrics;
  isStreaming?: boolean;
  isThinkingInProgress?: boolean;
  aborted?: boolean;
  error?: string;
}

/**
 * Text generation hyperparameters passed to Transformers.js generate().
 */
export interface GenerationConfig {
  maxNewTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  doSample: boolean;
  enableThinking: boolean;
}

export type DownloadFileStatus = 'initiate' | 'download' | 'progress' | 'done';

/**
 * Per-file download progress reported while fetching ONNX shards/tokenizer.
 */
export interface DownloadFileProgress {
  file: string;
  status: DownloadFileStatus;
  loadedBytes: number;
  totalBytes: number;
  progressPct: number;
}

export type ModelLoadStage =
  | 'probing-device'
  | 'downloading'
  | 'compiling-shaders'
  | 'initializing-session'
  | 'ready';

/**
 * Aggregated model loading progress across all required text-only artifacts.
 */
export interface ModelLoadProgress {
  modelId: ModelId;
  stage: ModelLoadStage;
  message: string;
  overallProgressPct: number;
  loadedBytes: number;
  expectedTotalBytes: number;
  files: DownloadFileProgress[];
}

/**
 * Individual cached file entry in CacheStorage (`transformers-cache`).
 */
export interface ModelCacheEntry {
  url: string;
  file: string;
  bytes: number;
}

/**
 * Browser CacheStorage inspection summary for a model repository.
 */
export interface ModelCacheInfo {
  modelId: ModelId;
  repoId: string;
  cached: boolean;
  cachedBytes: number;
  fileCount: number;
  files: ModelCacheEntry[];
  lastCheckedAt: number;
}

/**
 * Token throughput and latency breakdown for one generation turn.
 */
export interface InferenceSpeedMetrics {
  ttftMs: number;
  prefillMs: number;
  decodeMs: number;
  totalMs: number;
  promptTokens: number;
  generatedTokens: number;
  tokensPerSec: number;
  decodeTokensPerSec: number;
  interTokenLatencyAvgMs: number;
  interTokenLatencyP95Ms: number;
}

/**
 * Memory, main-thread responsiveness, and backend resource metrics.
 */
export interface InferenceLoadMetrics {
  jsHeapUsedBytes?: number;
  jsHeapTotalBytes?: number;
  uaMemoryBytes?: number;
  estimatedVramBytes: number;
  mainThreadAvgFrameMs: number;
  mainThreadCpuBusyRatio: number;
  droppedFrames: number;
  longTaskCount: number;
  backend: ExecutionBackend;
  dtype: QuantDtype;
  activeThreads: number;
}

export type PowerEstimationMethod = 'battery-api+utilization' | 'utilization-model';

/**
 * Battery and estimated power/energy efficiency metrics for one generation turn.
 */
export interface InferencePowerMetrics {
  batterySupported: boolean;
  batteryLevelStart?: number;
  batteryLevelEnd?: number;
  batteryCharging?: boolean;
  batteryDischargingTimeSec?: number;
  batteryDeltaPct?: number;
  estimatedPowerWatts: number;
  estimatedEnergyWh: number;
  tokensPerWattHour?: number;
  estimationMethod: PowerEstimationMethod;
}

/**
 * Combined inference telemetry (speed, resource load, and power estimation).
 */
export interface InferenceMetrics {
  speed: InferenceSpeedMetrics;
  load: InferenceLoadMetrics;
  power: InferencePowerMetrics;
}

export type BenchmarkRunStatus = 'completed' | 'aborted' | 'error';

/**
 * Saved benchmark comparison entry across models, devices, and dtypes.
 */
export interface BenchmarkRecord {
  id: string;
  timestamp: number;
  promptId?: string;
  promptLabel: string;
  promptText: string;
  modelId: ModelId;
  modelName: string;
  deviceId: string;
  deviceLabel: string;
  backend: ExecutionBackend;
  gpuPreference?: GpuPreference;
  dtype: QuantDtype;
  generationConfig: GenerationConfig;
  status: BenchmarkRunStatus;
  outputPreview: string;
  thinkingPreview?: string;
  metrics: InferenceMetrics;
  error?: string;
}

/**
 * Serializable device selection passed from main thread to Web Worker.
 * GPUDevice cannot be transferred via postMessage, so the worker re-acquires
 * the adapter and GPUDevice directly via navigator.gpu using gpuPreference.
 */
export type WorkerDeviceTarget =
  | {
      backend: 'webgpu';
      gpuPreference: GpuPreference;
      expectedDeviceFingerprint?: string;
    }
  | {
      backend: 'wasm';
      numThreads?: number;
      simdSupported?: boolean;
    };

/**
 * Discriminated union of messages sent from the main thread to the LLM Web Worker.
 */
export type WorkerRequest =
  | {
      type: 'probe-devices';
      requestId: string;
    }
  | {
      type: 'load-model';
      requestId: string;
      modelId: ModelId;
      deviceTarget: WorkerDeviceTarget;
      dtype?: QuantDtype;
      useExternalDataFormat?: boolean;
    }
  | {
      type: 'generate';
      requestId: string;
      messages: ChatPromptMessage[];
      config: GenerationConfig;
    }
  | {
      type: 'abort';
      requestId: string;
      targetRequestId?: string;
    }
  | {
      type: 'unload-model';
      requestId: string;
    }
  | {
      type: 'inspect-cache';
      requestId: string;
      modelIds?: ModelId[];
    }
  | {
      type: 'clear-cache';
      requestId: string;
      modelId?: ModelId;
    };

/**
 * Discriminated union of messages emitted from the LLM Web Worker to the main thread.
 */
export type WorkerResponse =
  | {
      type: 'devices-probed';
      requestId: string;
      devices: DeviceOption[];
    }
  | {
      type: 'load-progress';
      requestId: string;
      progress: ModelLoadProgress;
    }
  | {
      type: 'model-loaded';
      requestId: string;
      modelId: ModelId;
      device: DeviceOption;
      effectiveDtype: QuantDtype;
      useExternalDataFormat: boolean;
      loadTimeMs: number;
      fallbackNotice?: string;
    }
  | {
      type: 'generation-started';
      requestId: string;
      promptTokens: number;
    }
  | {
      type: 'token';
      requestId: string;
      deltaText: string;
      fullText: string;
      tokenCount: number;
      timestampMs: number;
      isFirstToken: boolean;
    }
  | {
      type: 'generation-complete';
      requestId: string;
      fullText: string;
      thinking: string;
      content: string;
      aborted: boolean;
      speed: InferenceSpeedMetrics;
      workerJsHeapUsedBytes?: number;
      workerJsHeapTotalBytes?: number;
      activeThreads: number;
      backend: ExecutionBackend;
      dtype: QuantDtype;
    }
  | {
      type: 'model-unloaded';
      requestId: string;
    }
  | {
      type: 'cache-info';
      requestId: string;
      caches: ModelCacheInfo[];
    }
  | {
      type: 'cache-cleared';
      requestId: string;
      modelId?: ModelId;
      clearedBytes: number;
      deletedEntries: number;
    }
  | {
      type: 'error';
      requestId: string;
      message: string;
      code?: string;
      recoverable?: boolean;
    };
