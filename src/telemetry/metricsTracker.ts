import type {
  DeviceOption,
  ExecutionBackend,
  InferenceLoadMetrics,
  InferenceMetrics,
  InferencePowerMetrics,
  InferenceSpeedMetrics,
  ModelSpec,
  PowerEstimationMethod,
  QuantDtype,
} from '../types.ts';
import {
  estimateModelMemoryBytes,
  sampleBatteryStatus,
  sampleMemorySnapshot,
  sampleMemorySnapshotAsync,
  type BatterySnapshot,
  type MemorySnapshot,
  type SystemProbeEnvironment,
} from './systemProbe.ts';

export type PowerDeviceClass = 'discrete-gpu' | 'integrated-gpu' | 'software-gpu' | 'cpu-wasm';

export interface EstimatePowerMetricsParams {
  backend: ExecutionBackend;
  device?: DeviceOption;
  model: ModelSpec;
  dtype: QuantDtype;
  totalMs: number;
  generatedTokens: number;
  cpuBusyRatio: number;
  activeThreads?: number;
  batteryStart?: BatterySnapshot;
  batteryEnd?: BatterySnapshot;
}

export interface MainThreadLoadSnapshot {
  mainThreadAvgFrameMs: number;
  mainThreadCpuBusyRatio: number;
  droppedFrames: number;
  longTaskCount: number;
  elapsedMs: number;
  sampleCount: number;
}

export interface PerformanceEntryLike {
  duration?: number;
  entryType?: string;
}

export interface PerformanceObserverEntryListLike {
  getEntries(): PerformanceEntryLike[];
}

export interface PerformanceObserverLike {
  observe(options: { entryTypes?: string[]; type?: string; buffered?: boolean }): void;
  disconnect(): void;
}

export interface PerformanceObserverConstructorLike {
  new (callback: (list: PerformanceObserverEntryListLike) => void): PerformanceObserverLike;
  supportedEntryTypes?: readonly string[];
}

export interface MainThreadMonitorEnvironment {
  requestAnimationFrame?: ((callback: (timestamp: number) => void) => number) | null;
  cancelAnimationFrame?: ((handle: number) => void) | null;
  PerformanceObserver?: PerformanceObserverConstructorLike | null;
  now?: () => number;
}

export interface TelemetryEnvironment extends SystemProbeEnvironment, MainThreadMonitorEnvironment {}

export interface LiveSpeedSnapshot {
  elapsedMs: number;
  ttftMs?: number;
  generatedTokens: number;
  promptTokens: number;
  overallTokensPerSec: number;
  decodeTokensPerSec: number;
  instantTokensPerSec: number;
  rollingTokensPerSec: number;
  interTokenLatencyAvgMs: number;
  interTokenLatencyP95Ms: number;
}

export interface LiveTelemetrySnapshot {
  speed: LiveSpeedSnapshot;
  memory: MemorySnapshot;
  estimatedVramBytes: number;
  load: MainThreadLoadSnapshot;
  power: InferencePowerMetrics;
  battery?: BatterySnapshot;
}

export interface LiveTelemetrySessionOptions {
  model: ModelSpec;
  device: DeviceOption;
  dtype: QuantDtype;
  promptTokens?: number;
  contextTokens?: number;
  activeThreads?: number;
  env?: TelemetryEnvironment;
}

export interface FinalizeTelemetryInput {
  speed?: InferenceSpeedMetrics;
  workerJsHeapUsedBytes?: number;
  workerJsHeapTotalBytes?: number;
  activeThreads?: number;
  backend?: ExecutionBackend;
  dtype?: QuantDtype;
  finishedAtMs?: number;
}

const TARGET_FRAME_BUDGET_MS = 16.67;
const DROPPED_FRAME_THRESHOLD_MS = 25;
const ROLLING_WINDOW_MS = 1000;
const REFERENCE_LAPTOP_BATTERY_WH = 55;

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

function roundTo(value: number, decimals: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function defaultNow(): number {
  try {
    if (typeof globalThis.performance?.now === 'function') {
      return globalThis.performance.now();
    }
  } catch {
    // Fallback to Date.now when performance is unavailable.
  }
  return Date.now();
}

function parseModelBillionScale(model: ModelSpec): number {
  const label = model.parameterLabel.trim().toUpperCase();
  const match = label.match(/([0-9]+(?:\.[0-9]+)?)/);
  const raw = match && match[1] ? Number.parseFloat(match[1]) : 2.5;
  if (!Number.isFinite(raw) || raw <= 0) {
    return 1.0;
  }
  // Effective active parameter multiplier for power envelope scaling (0.78 for 0.8B .. 1.18 for E4B/4B).
  const normalized = 0.72 + Math.min(4.5, raw) * 0.1;
  return label.startsWith('E') ? normalized * 0.96 : normalized;
}

function getDtypePowerMultiplier(dtype: QuantDtype): number {
  switch (dtype) {
    case 'q4f16':
      return 0.94;
    case 'q4':
      return 1.0;
    case 'q8':
      return 1.12;
    case 'fp16':
      return 1.22;
  }
}

/**
 * Classify a compute backend and WebGPU adapter into a power-estimation tier.
 */
export function classifyPowerDevice(
  backend: ExecutionBackend,
  device?: DeviceOption
): PowerDeviceClass {
  if (backend === 'wasm') {
    return 'cpu-wasm';
  }

  const info = device?.adapterInfo;
  const vendor = (info?.vendor ?? '').toLowerCase();
  const description = (info?.description ?? '').toLowerCase();
  const deviceName = (info?.device ?? '').toLowerCase();
  const architecture = (info?.architecture ?? '').toLowerCase();
  const label = (device?.label ?? '').toLowerCase();
  const combined = `${vendor} ${description} ${deviceName} ${architecture} ${label}`;

  if (
    info?.isFallbackAdapter ||
    device?.gpuPreference === 'fallback' ||
    /\b(swiftshader|llvmpipe|softpipe|warp)\b|microsoft basic render|software/.test(combined)
  ) {
    return 'software-gpu';
  }

  if (device?.gpuPreference === 'low-power') {
    return 'integrated-gpu';
  }

  // Discrete GPU cues take precedence over generic vendor names (e.g., Intel Arc, AMD Radeon RX).
  if (
    vendor.includes('nvidia') ||
    /\b(nvidia|geforce|rtx|gtx|quadro|tesla|radeon rx|radeon pro|arc a\d+|arc b\d+)\b/.test(
      combined
    )
  ) {
    return 'discrete-gpu';
  }

  // Apple Silicon, Intel iGPU, AMD APU/iGPU, and mobile SoC GPUs.
  if (
    vendor.includes('apple') ||
    vendor.includes('intel') ||
    vendor.includes('qualcomm') ||
    vendor.includes('arm') ||
    /\b(apple|uhd|iris|radeon graphics|vega|780m|680m|adreno|mali)\b/.test(combined)
  ) {
    return 'integrated-gpu';
  }

  if (device?.gpuPreference === 'high-performance') {
    return 'discrete-gpu';
  }

  return 'integrated-gpu';
}

/**
 * Compute transparent, honest power & energy efficiency metrics for a single inference run.
 *
 * - Distinguishes discrete GPU (`high-performance`), integrated/Apple GPU (`low-power` / Apple /
 *   Intel / AMD iGPU), software fallback GPU, and CPU WASM (`activeThreads` scaling).
 * - Uses `'battery-api+utilization'` when Battery API confirms the host is running on battery
 *   discharge (`batteryCharging === false`), and `'utilization-model'` otherwise.
 */
export function estimatePowerMetrics(params: EstimatePowerMetricsParams): InferencePowerMetrics {
  const totalMs =
    typeof params.totalMs === 'number' && Number.isFinite(params.totalMs) && params.totalMs > 0
      ? params.totalMs
      : 0;
  const generatedTokens =
    typeof params.generatedTokens === 'number' &&
    Number.isFinite(params.generatedTokens) &&
    params.generatedTokens > 0
      ? Math.round(params.generatedTokens)
      : 0;
  const cpuBusyRatio = clamp(params.cpuBusyRatio, 0, 1);

  const deviceClass = classifyPowerDevice(params.backend, params.device);
  const modelFactor = parseModelBillionScale(params.model) * getDtypePowerMultiplier(params.dtype);
  const activeThreads = clamp(
    Math.round(params.activeThreads ?? params.device?.runtimeDetails.numThreads ?? 1),
    1,
    16
  );

  // Whether inference compute itself is active during this run.
  const isComputing = totalMs > 0 || generatedTokens > 0;
  const computeActivity = isComputing ? 1 : 0.15;

  let baseModelWatts: number;
  switch (deviceClass) {
    case 'discrete-gpu':
      // Laptop/desktop dGPU incremental compute envelope (~34W..72W depending on model/utilization).
      baseModelWatts = (26 + 24 * modelFactor) * computeActivity + 8 * cpuBusyRatio;
      break;
    case 'integrated-gpu':
      // Apple Silicon / Intel / AMD iGPU unified memory envelope (~10W..23W).
      baseModelWatts = (7.5 + 8.5 * modelFactor) * computeActivity + 3.5 * cpuBusyRatio;
      break;
    case 'software-gpu':
      // Software rasterizer/compute runs on host CPU threads (~15W..28W).
      baseModelWatts = (10 + 10 * modelFactor) * computeActivity + 4.5 * cpuBusyRatio;
      break;
    case 'cpu-wasm': {
      // Sublinear thread scaling reflects turbo & memory bandwidth saturation across cores.
      const threadScale = Math.pow(activeThreads, 0.72);
      baseModelWatts = (6 + 4.2 * threadScale * modelFactor) * computeActivity + 2.5 * cpuBusyRatio;
      break;
    }
  }

  const batteryStart = params.batteryStart;
  const batteryEnd = params.batteryEnd;
  const batterySupported = Boolean(batteryStart?.supported || batteryEnd?.supported);

  const batteryLevelStart =
    batteryStart?.supported && typeof batteryStart.level === 'number'
      ? batteryStart.level
      : undefined;
  const batteryLevelEnd =
    batteryEnd?.supported && typeof batteryEnd.level === 'number' ? batteryEnd.level : undefined;

  const batteryCharging =
    batteryEnd?.supported && typeof batteryEnd.charging === 'boolean'
      ? batteryEnd.charging
      : batteryStart?.supported && typeof batteryStart.charging === 'boolean'
        ? batteryStart.charging
        : undefined;

  const batteryDischargingTimeSec =
    batteryEnd?.supported && typeof batteryEnd.dischargingTimeSec === 'number'
      ? batteryEnd.dischargingTimeSec
      : batteryStart?.supported && typeof batteryStart.dischargingTimeSec === 'number'
        ? batteryStart.dischargingTimeSec
        : undefined;

  const batteryDeltaPct =
    batteryLevelStart !== undefined && batteryLevelEnd !== undefined
      ? roundTo(batteryLevelEnd - batteryLevelStart, 4)
      : undefined;

  const onBatteryDischarge = batterySupported && batteryCharging === false;
  const estimationMethod: PowerEstimationMethod = onBatteryDischarge
    ? 'battery-api+utilization'
    : 'utilization-model';

  let estimatedPowerWatts = baseModelWatts;

  if (onBatteryDischarge) {
    // When actual battery percentage dropped over a meaningful interval (>= 5s),
    // gently calibrate the component utilization model while clamping noisy OS steps.
    if (batteryDeltaPct !== undefined && batteryDeltaPct < 0 && totalMs >= 5000) {
      const hours = totalMs / 3_600_000;
      const systemObservedWatts = ((-batteryDeltaPct / 100) * REFERENCE_LAPTOP_BATTERY_WH) / hours;
      const inferenceShareWatts = clamp(
        systemObservedWatts * 0.65,
        baseModelWatts * 0.65,
        baseModelWatts * 1.5
      );
      estimatedPowerWatts = baseModelWatts * 0.75 + inferenceShareWatts * 0.25;
    } else if (
      batteryDischargingTimeSec !== undefined &&
      batteryDischargingTimeSec > 0 &&
      (batteryLevelEnd ?? batteryLevelStart ?? 0) > 0
    ) {
      const currentPct = batteryLevelEnd ?? batteryLevelStart ?? 50;
      const remainingWh = REFERENCE_LAPTOP_BATTERY_WH * (currentPct / 100);
      const systemDischargeWatts = remainingWh / (batteryDischargingTimeSec / 3600);
      const boundedHint = clamp(
        systemDischargeWatts * 0.55,
        baseModelWatts * 0.75,
        baseModelWatts * 1.35
      );
      estimatedPowerWatts = baseModelWatts * 0.85 + boundedHint * 0.15;
    }
  }

  estimatedPowerWatts = roundTo(Math.max(1, estimatedPowerWatts), 2);
  const hours = totalMs / 3_600_000;
  const estimatedEnergyWh = roundTo(estimatedPowerWatts * hours, 6);
  const tokensPerWattHour =
    estimatedEnergyWh > 0 && generatedTokens > 0
      ? roundTo(generatedTokens / estimatedEnergyWh, 2)
      : undefined;

  return {
    batterySupported,
    ...(batteryLevelStart !== undefined ? { batteryLevelStart } : {}),
    ...(batteryLevelEnd !== undefined ? { batteryLevelEnd } : {}),
    ...(batteryCharging !== undefined ? { batteryCharging } : {}),
    ...(batteryDischargingTimeSec !== undefined ? { batteryDischargingTimeSec } : {}),
    ...(batteryDeltaPct !== undefined ? { batteryDeltaPct } : {}),
    estimatedPowerWatts,
    estimatedEnergyWh,
    ...(tokensPerWattHour !== undefined ? { tokensPerWattHour } : {}),
    estimationMethod,
  };
}

/**
 * Monitors main-thread frame pacing (`requestAnimationFrame` deltas) and
 * long tasks (`PerformanceObserver('longtask')`) safely across browser and test runtimes.
 */
export class MainThreadLoadMonitor {
  private readonly env: MainThreadMonitorEnvironment;
  private running = false;
  private startedAt = 0;
  private stoppedAt: number | undefined;
  private lastFrameAt: number | undefined;
  private rafId: number | undefined;
  private observer: PerformanceObserverLike | undefined;
  private frameDeltas: number[] = [];
  private longTaskCount = 0;
  private longTaskTotalMs = 0;

  constructor(env: MainThreadMonitorEnvironment = {}) {
    this.env = env;
  }

  private now(): number {
    return this.env.now ? this.env.now() : defaultNow();
  }

  private getRaf(): ((callback: (timestamp: number) => void) => number) | undefined {
    if ('requestAnimationFrame' in this.env) {
      return this.env.requestAnimationFrame ?? undefined;
    }
    try {
      return typeof globalThis.requestAnimationFrame === 'function'
        ? globalThis.requestAnimationFrame.bind(globalThis)
        : undefined;
    } catch {
      return undefined;
    }
  }

  private getCancelRaf(): ((handle: number) => void) | undefined {
    if ('cancelAnimationFrame' in this.env) {
      return this.env.cancelAnimationFrame ?? undefined;
    }
    try {
      return typeof globalThis.cancelAnimationFrame === 'function'
        ? globalThis.cancelAnimationFrame.bind(globalThis)
        : undefined;
    } catch {
      return undefined;
    }
  }

  private getObserverCtor(): PerformanceObserverConstructorLike | undefined {
    if ('PerformanceObserver' in this.env) {
      return this.env.PerformanceObserver ?? undefined;
    }
    try {
      const ctor = (globalThis as { PerformanceObserver?: PerformanceObserverConstructorLike })
        .PerformanceObserver;
      return typeof ctor === 'function' ? ctor : undefined;
    } catch {
      return undefined;
    }
  }

  start(): void {
    this.cleanupMonitors();

    this.running = true;
    this.startedAt = this.now();
    this.stoppedAt = undefined;
    this.lastFrameAt = undefined;
    this.frameDeltas = [];
    this.longTaskCount = 0;
    this.longTaskTotalMs = 0;

    this.startLongTaskObserver();
    this.scheduleNextFrame();
  }

  private startLongTaskObserver(): void {
    const ObserverCtor = this.getObserverCtor();
    if (!ObserverCtor) {
      return;
    }

    try {
      const supported = ObserverCtor.supportedEntryTypes;
      if (Array.isArray(supported) && !supported.includes('longtask')) {
        return;
      }
      const observer = new ObserverCtor((list) => {
        if (!this.running) {
          return;
        }
        try {
          const entries = list.getEntries();
          for (const entry of entries) {
            this.longTaskCount += 1;
            if (typeof entry.duration === 'number' && Number.isFinite(entry.duration) && entry.duration > 0) {
              this.longTaskTotalMs += entry.duration;
            } else {
              this.longTaskTotalMs += 50;
            }
          }
        } catch {
          // Ignore malformed PerformanceObserver entries.
        }
      });
      observer.observe({ entryTypes: ['longtask'] });
      this.observer = observer;
    } catch {
      // Long Task API is unsupported or threw in restricted context.
      this.observer = undefined;
    }
  }

  private scheduleNextFrame(): void {
    const raf = this.getRaf();
    if (!raf || !this.running) {
      return;
    }
    try {
      this.rafId = raf((timestamp) => {
        this.onFrame(timestamp);
      });
    } catch {
      this.rafId = undefined;
    }
  }

  private onFrame(timestamp?: number): void {
    if (!this.running) {
      return;
    }
    const frameTime =
      typeof timestamp === 'number' && Number.isFinite(timestamp) ? timestamp : this.now();

    if (this.lastFrameAt !== undefined) {
      const delta = Math.max(0, frameTime - this.lastFrameAt);
      this.frameDeltas.push(delta);
    }
    this.lastFrameAt = frameTime;
    this.scheduleNextFrame();
  }

  snapshot(): MainThreadLoadSnapshot {
    const endTime = this.running ? this.now() : (this.stoppedAt ?? this.now());
    const elapsedMs = Math.max(0, endTime - this.startedAt);
    const sampleCount = this.frameDeltas.length;

    let totalFrameMs = 0;
    let excessFrameMs = 0;
    let droppedFrames = 0;

    for (const delta of this.frameDeltas) {
      totalFrameMs += delta;
      if (delta > DROPPED_FRAME_THRESHOLD_MS) {
        droppedFrames += 1;
      }
      if (delta > TARGET_FRAME_BUDGET_MS) {
        excessFrameMs += delta - TARGET_FRAME_BUDGET_MS;
      }
    }

    const mainThreadAvgFrameMs = sampleCount > 0 ? roundTo(totalFrameMs / sampleCount, 2) : 0;
    const rafBusyRatio = totalFrameMs > 0 ? clamp(excessFrameMs / totalFrameMs, 0, 1) : 0;
    const longTaskBusyRatio =
      elapsedMs > 0 && this.longTaskTotalMs > 0
        ? clamp(this.longTaskTotalMs / elapsedMs, 0, 1)
        : 0;
    const mainThreadCpuBusyRatio = roundTo(Math.max(rafBusyRatio, longTaskBusyRatio), 4);

    return {
      mainThreadAvgFrameMs,
      mainThreadCpuBusyRatio,
      droppedFrames,
      longTaskCount: this.longTaskCount,
      elapsedMs: roundTo(elapsedMs, 2),
      sampleCount,
    };
  }

  stop(): MainThreadLoadSnapshot {
    if (this.running) {
      this.running = false;
      this.stoppedAt = this.now();
      this.cleanupMonitors();
    }
    return this.snapshot();
  }

  private cleanupMonitors(): void {
    if (this.rafId !== undefined) {
      const cancelRaf = this.getCancelRaf();
      try {
        cancelRaf?.(this.rafId);
      } catch {
        // Ignore cleanup errors.
      }
      this.rafId = undefined;
    }

    if (this.observer) {
      try {
        this.observer.disconnect();
      } catch {
        // Ignore cleanup errors.
      }
      this.observer = undefined;
    }
  }
}

function computePercentile(sortedValues: readonly number[], percentile: number): number {
  if (sortedValues.length === 0) {
    return 0;
  }
  const index = Math.min(
    sortedValues.length - 1,
    Math.max(0, Math.ceil((percentile / 100) * sortedValues.length) - 1)
  );
  return sortedValues[index] ?? 0;
}

interface TokenSamplePoint {
  atMs: number;
  count: number;
}

/**
 * Coordinates battery snapshots, main-thread responsiveness, memory snapshots,
 * live streaming throughput, and final `InferenceMetrics` assembly for one generation turn.
 */
export class LiveTelemetrySession {
  private readonly model: ModelSpec;
  private readonly device: DeviceOption;
  private readonly dtype: QuantDtype;
  private readonly contextTokensOverride?: number;
  private readonly activeThreadsOverride?: number;
  private readonly env: TelemetryEnvironment;
  private readonly loadMonitor: MainThreadLoadMonitor;

  private promptTokens: number;
  private generatedTokens = 0;
  private startedAtMs: number | undefined;
  private firstTokenAtMs: number | undefined;
  private lastTokenAtMs: number | undefined;
  private finishedAtMs: number | undefined;

  private tokenSamples: TokenSamplePoint[] = [];
  private interTokenDeltasMs: number[] = [];
  private batteryStart: BatterySnapshot | undefined;
  private batteryLatest: BatterySnapshot | undefined;

  constructor(options: LiveTelemetrySessionOptions) {
    this.model = options.model;
    this.device = options.device;
    this.dtype = options.dtype;
    this.promptTokens = Math.max(0, Math.round(options.promptTokens ?? 0));
    this.contextTokensOverride = options.contextTokens;
    this.activeThreadsOverride = options.activeThreads;
    this.env = options.env ?? {};
    this.loadMonitor = new MainThreadLoadMonitor(this.env);
  }

  private now(): number {
    return this.env.now ? this.env.now() : defaultNow();
  }

  async start(promptTokens?: number): Promise<LiveTelemetrySnapshot> {
    if (typeof promptTokens === 'number' && Number.isFinite(promptTokens)) {
      this.promptTokens = Math.max(0, Math.round(promptTokens));
    }

    this.generatedTokens = 0;
    this.startedAtMs = this.now();
    this.firstTokenAtMs = undefined;
    this.lastTokenAtMs = undefined;
    this.finishedAtMs = undefined;
    this.tokenSamples = [];
    this.interTokenDeltasMs = [];

    this.loadMonitor.start();
    this.batteryStart = await sampleBatteryStatus(this.env);
    this.batteryLatest = this.batteryStart;

    // Warm UA memory cache asynchronously when crossOriginIsolated without blocking streaming.
    void sampleMemorySnapshotAsync(this.env).catch(() => undefined);

    return this.getSnapshot(this.startedAtMs);
  }

  onGenerationStarted(promptTokens: number): void {
    if (Number.isFinite(promptTokens) && promptTokens >= 0) {
      this.promptTokens = Math.round(promptTokens);
    }
  }

  onToken(tokenCount?: number, atMs?: number): LiveTelemetrySnapshot {
    const ts = typeof atMs === 'number' && Number.isFinite(atMs) ? atMs : this.now();
    if (this.startedAtMs === undefined) {
      this.startedAtMs = ts;
    }

    const prevCount = this.generatedTokens;
    const nextCount =
      typeof tokenCount === 'number' && Number.isFinite(tokenCount) && tokenCount >= 0
        ? Math.round(tokenCount)
        : prevCount + 1;

    if (nextCount > 0 && this.firstTokenAtMs === undefined) {
      this.firstTokenAtMs = ts;
    }

    if (this.lastTokenAtMs !== undefined && nextCount > prevCount) {
      const steps = nextCount - prevCount;
      const perTokenDelta = Math.max(0, ts - this.lastTokenAtMs) / steps;
      for (let i = 0; i < steps; i += 1) {
        this.interTokenDeltasMs.push(perTokenDelta);
      }
    }

    if (nextCount > prevCount) {
      this.tokenSamples.push({ atMs: ts, count: nextCount });
      this.lastTokenAtMs = ts;
    }

    this.generatedTokens = Math.max(prevCount, nextCount);

    return this.getSnapshot(ts);
  }

  private computeLiveSpeed(referenceNowMs: number): LiveSpeedSnapshot {
    const startedAt = this.startedAtMs ?? referenceNowMs;
    const elapsedMs = Math.max(0, referenceNowMs - startedAt);
    const ttftMs =
      this.firstTokenAtMs !== undefined ? Math.max(0, this.firstTokenAtMs - startedAt) : undefined;
    const decodeMs =
      this.firstTokenAtMs !== undefined ? Math.max(0, referenceNowMs - this.firstTokenAtMs) : 0;

    const overallTokensPerSec =
      elapsedMs > 0 && this.generatedTokens > 0
        ? roundTo((this.generatedTokens * 1000) / elapsedMs, 2)
        : 0;

    const decodeTokensPerSec =
      decodeMs > 0 && this.generatedTokens > 1
        ? roundTo(((this.generatedTokens - 1) * 1000) / decodeMs, 2)
        : overallTokensPerSec;

    const lastDelta =
      this.interTokenDeltasMs.length > 0
        ? this.interTokenDeltasMs[this.interTokenDeltasMs.length - 1]
        : undefined;
    const instantTokensPerSec =
      lastDelta !== undefined && lastDelta > 0
        ? roundTo(1000 / lastDelta, 2)
        : decodeTokensPerSec;

    let rollingTokensPerSec = decodeTokensPerSec;
    if (this.tokenSamples.length >= 2) {
      const latest = this.tokenSamples[this.tokenSamples.length - 1]!;
      const windowStart = latest.atMs - ROLLING_WINDOW_MS;
      let anchor = this.tokenSamples[0]!;
      for (let i = this.tokenSamples.length - 2; i >= 0; i -= 1) {
        const candidate = this.tokenSamples[i]!;
        anchor = candidate;
        if (candidate.atMs <= windowStart) {
          break;
        }
      }
      const deltaMs = latest.atMs - anchor.atMs;
      const deltaTokens = latest.count - anchor.count;
      if (deltaMs > 0 && deltaTokens > 0) {
        rollingTokensPerSec = roundTo((deltaTokens * 1000) / deltaMs, 2);
      }
    }

    const sortedDeltas = [...this.interTokenDeltasMs].sort((a, b) => a - b);
    const avgItl =
      sortedDeltas.length > 0
        ? sortedDeltas.reduce((acc, v) => acc + v, 0) / sortedDeltas.length
        : 0;
    const p95Itl = computePercentile(sortedDeltas, 95);

    return {
      elapsedMs: roundTo(elapsedMs, 2),
      ...(ttftMs !== undefined ? { ttftMs: roundTo(ttftMs, 2) } : {}),
      generatedTokens: this.generatedTokens,
      promptTokens: this.promptTokens,
      overallTokensPerSec,
      decodeTokensPerSec,
      instantTokensPerSec,
      rollingTokensPerSec,
      interTokenLatencyAvgMs: roundTo(avgItl, 2),
      interTokenLatencyP95Ms: roundTo(p95Itl, 2),
    };
  }

  getSnapshot(atMs?: number): LiveTelemetrySnapshot {
    const refNow = atMs ?? this.finishedAtMs ?? this.now();
    const speed = this.computeLiveSpeed(refNow);
    const load = this.loadMonitor.snapshot();
    const memory = sampleMemorySnapshot(this.env);
    const contextTokens =
      this.contextTokensOverride ?? this.promptTokens + this.generatedTokens;
    const estimatedVramBytes = estimateModelMemoryBytes(this.model, this.dtype, contextTokens);
    const activeThreads = this.activeThreadsOverride ?? this.device.runtimeDetails.numThreads;

    const power = estimatePowerMetrics({
      backend: this.device.backend,
      device: this.device,
      model: this.model,
      dtype: this.dtype,
      totalMs: speed.elapsedMs,
      generatedTokens: speed.generatedTokens,
      cpuBusyRatio: load.mainThreadCpuBusyRatio,
      activeThreads,
      batteryStart: this.batteryStart,
      batteryEnd: this.batteryLatest,
    });

    return {
      speed,
      memory,
      estimatedVramBytes,
      load,
      power,
      ...(this.batteryLatest ? { battery: this.batteryLatest } : {}),
    };
  }

  async finish(workerData?: FinalizeTelemetryInput): Promise<InferenceMetrics> {
    const finishedAt = workerData?.finishedAtMs ?? this.now();
    this.finishedAtMs = finishedAt;

    const loadSnap = this.loadMonitor.stop();
    const batteryEnd = await sampleBatteryStatus(this.env);
    this.batteryLatest = batteryEnd;
    const memoryEnd = await sampleMemorySnapshotAsync(this.env);

    const liveSpeed = this.computeLiveSpeed(finishedAt);
    const speed: InferenceSpeedMetrics = workerData?.speed ?? {
      ttftMs: liveSpeed.ttftMs ?? liveSpeed.elapsedMs,
      prefillMs: liveSpeed.ttftMs ?? liveSpeed.elapsedMs,
      decodeMs:
        liveSpeed.ttftMs !== undefined
          ? roundTo(Math.max(0, liveSpeed.elapsedMs - liveSpeed.ttftMs), 2)
          : 0,
      totalMs: liveSpeed.elapsedMs,
      promptTokens: liveSpeed.promptTokens,
      generatedTokens: liveSpeed.generatedTokens,
      tokensPerSec: liveSpeed.overallTokensPerSec,
      decodeTokensPerSec: liveSpeed.decodeTokensPerSec,
      interTokenLatencyAvgMs: liveSpeed.interTokenLatencyAvgMs,
      interTokenLatencyP95Ms: liveSpeed.interTokenLatencyP95Ms,
    };

    const effectiveBackend = workerData?.backend ?? this.device.backend;
    const effectiveDtype = workerData?.dtype ?? this.dtype;
    const effectiveThreads =
      workerData?.activeThreads ??
      this.activeThreadsOverride ??
      this.device.runtimeDetails.numThreads;

    const estimatedVramBytes = estimateModelMemoryBytes(
      this.model,
      effectiveDtype,
      this.contextTokensOverride ?? speed.promptTokens + speed.generatedTokens
    );

    const jsHeapUsedBytes = workerData?.workerJsHeapUsedBytes ?? memoryEnd.jsHeapUsedBytes;
    const jsHeapTotalBytes = workerData?.workerJsHeapTotalBytes ?? memoryEnd.jsHeapTotalBytes;

    const load: InferenceLoadMetrics = {
      ...(jsHeapUsedBytes !== undefined ? { jsHeapUsedBytes } : {}),
      ...(jsHeapTotalBytes !== undefined ? { jsHeapTotalBytes } : {}),
      ...(memoryEnd.uaMemoryBytes !== undefined ? { uaMemoryBytes: memoryEnd.uaMemoryBytes } : {}),
      estimatedVramBytes,
      mainThreadAvgFrameMs: loadSnap.mainThreadAvgFrameMs,
      mainThreadCpuBusyRatio: loadSnap.mainThreadCpuBusyRatio,
      droppedFrames: loadSnap.droppedFrames,
      longTaskCount: loadSnap.longTaskCount,
      backend: effectiveBackend,
      dtype: effectiveDtype,
      activeThreads: effectiveThreads,
    };

    const power = estimatePowerMetrics({
      backend: effectiveBackend,
      device: this.device,
      model: this.model,
      dtype: effectiveDtype,
      totalMs: speed.totalMs,
      generatedTokens: speed.generatedTokens,
      cpuBusyRatio: loadSnap.mainThreadCpuBusyRatio,
      activeThreads: effectiveThreads,
      batteryStart: this.batteryStart,
      batteryEnd,
    });

    return {
      speed,
      load,
      power,
    };
  }

  abort(): void {
    this.finishedAtMs = this.now();
    this.loadMonitor.stop();
  }
}
