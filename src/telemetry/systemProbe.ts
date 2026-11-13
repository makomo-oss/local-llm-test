import type { ModelId, ModelSpec, QuantDtype } from '../types.ts';

/**
 * Snapshot of host battery telemetry obtained from the Web Battery Status API.
 */
export interface BatterySnapshot {
  supported: boolean;
  /** Battery charge percentage in [0, 100]. */
  level?: number;
  charging?: boolean;
  /** Remaining discharge time in seconds when finite and >= 0. */
  dischargingTimeSec?: number;
  /** Remaining charge time in seconds when finite and >= 0. */
  chargingTimeSec?: number;
  sampledAt: number;
}

/**
 * Snapshot of browser memory measurements available to the current context.
 */
export interface MemorySnapshot {
  jsHeapUsedBytes?: number;
  jsHeapTotalBytes?: number;
  jsHeapSizeLimitBytes?: number;
  uaMemoryBytes?: number;
  crossOriginIsolated: boolean;
  sampledAt: number;
}

/**
 * Minimal Web BatteryManager contract for safe probing and testing.
 */
export interface BatteryManagerLike {
  level: number;
  charging: boolean;
  dischargingTime: number;
  chargingTime: number;
  addEventListener?: (type: string, listener: EventListenerOrEventListenerObject) => void;
  removeEventListener?: (type: string, listener: EventListenerOrEventListenerObject) => void;
}

export interface PerformanceMemoryLike {
  usedJSHeapSize?: number;
  totalJSHeapSize?: number;
  jsHeapSizeLimit?: number;
}

export interface PerformanceWithMemoryLike {
  memory?: PerformanceMemoryLike;
  measureUserAgentSpecificMemory?: () => Promise<{ bytes?: number }>;
  now?: () => number;
}

export interface NavigatorWithBatteryLike {
  getBattery?: () => Promise<BatteryManagerLike>;
}

/**
 * Optional environment dependencies injected for deterministic testing.
 */
export interface SystemProbeEnvironment {
  navigator?: NavigatorWithBatteryLike | null;
  performance?: PerformanceWithMemoryLike | null;
  crossOriginIsolated?: boolean;
  nowEpochMs?: () => number;
}

const DEFAULT_CONTEXT_TOKENS = 2048;
const MIN_RUNTIME_OVERHEAD_BYTES = 64 * 1024 * 1024;
const ACTIVATION_WEIGHT_RATIO = 0.12;

/**
 * Verified KV cache element counts per token (`2 * layers * kvHeads * headDim`)
 * for models in `MODEL_CATALOG`.
 */
const KV_ELEMENTS_PER_TOKEN_BY_MODEL: Readonly<Record<ModelId, number>> = {
  'gemma-4-e4b': 2 * 42 * 2 * 256,
  'gemma-4-e2b': 2 * 35 * 1 * 256,
  'qwen-3.5-4b': 2 * 32 * 4 * 256,
  'qwen-3.5-2b': 2 * 24 * 2 * 256,
  'qwen-3.5-0.8b': 2 * 24 * 2 * 256,
  'llama-3.2-3b': 2 * 28 * 8 * 128,
};

/**
 * Per-performance-object cache of `measureUserAgentSpecificMemory()` bytes so
 * synchronous `sampleMemorySnapshot()` calls can surface the latest isolated reading
 * without mixing state across test mocks.
 */
const uaMemoryCache = new WeakMap<object, number>();

function getEpochMs(env?: SystemProbeEnvironment): number {
  return env?.nowEpochMs ? env.nowEpochMs() : Date.now();
}

function getNavigator(env?: SystemProbeEnvironment): NavigatorWithBatteryLike | undefined {
  if (env && 'navigator' in env) {
    return env.navigator ?? undefined;
  }
  try {
    return (globalThis as { navigator?: NavigatorWithBatteryLike }).navigator;
  } catch {
    return undefined;
  }
}

function getPerformance(env?: SystemProbeEnvironment): PerformanceWithMemoryLike | undefined {
  if (env && 'performance' in env) {
    return env.performance ?? undefined;
  }
  try {
    return (globalThis as { performance?: PerformanceWithMemoryLike }).performance;
  } catch {
    return undefined;
  }
}

function isCrossOriginIsolated(env?: SystemProbeEnvironment): boolean {
  if (env && typeof env.crossOriginIsolated === 'boolean') {
    return env.crossOriginIsolated;
  }
  try {
    return Boolean((globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated);
  } catch {
    return false;
  }
}

function toFiniteNonNegativeInt(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return undefined;
  }
  return Math.round(value);
}

function toBatterySnapshot(manager: BatteryManagerLike, sampledAt: number): BatterySnapshot {
  const rawLevel = manager.level;
  const level =
    typeof rawLevel === 'number' && Number.isFinite(rawLevel)
      ? Math.round(Math.min(100, Math.max(0, rawLevel * 100)) * 100) / 100
      : undefined;

  const charging = typeof manager.charging === 'boolean' ? manager.charging : undefined;

  const dischargingTimeSec =
    typeof manager.dischargingTime === 'number' &&
    Number.isFinite(manager.dischargingTime) &&
    manager.dischargingTime >= 0
      ? Math.round(manager.dischargingTime)
      : undefined;

  const chargingTimeSec =
    typeof manager.chargingTime === 'number' &&
    Number.isFinite(manager.chargingTime) &&
    manager.chargingTime >= 0
      ? Math.round(manager.chargingTime)
      : undefined;

  return {
    supported: true,
    ...(level !== undefined ? { level } : {}),
    ...(charging !== undefined ? { charging } : {}),
    ...(dischargingTimeSec !== undefined ? { dischargingTimeSec } : {}),
    ...(chargingTimeSec !== undefined ? { chargingTimeSec } : {}),
    sampledAt,
  };
}

async function resolveBatteryManager(
  env?: SystemProbeEnvironment
): Promise<BatteryManagerLike | undefined> {
  const nav = getNavigator(env);
  if (!nav || typeof nav.getBattery !== 'function') {
    return undefined;
  }
  try {
    const manager = await nav.getBattery();
    if (!manager || typeof manager !== 'object') {
      return undefined;
    }
    return manager;
  } catch {
    return undefined;
  }
}

/**
 * Safely query the Battery Status API (`navigator.getBattery?.()`).
 * Returns `{ supported: false }` without throwing in unsupported browsers
 * (e.g., Safari/Firefox) or restricted execution contexts.
 */
export async function sampleBatteryStatus(
  env?: SystemProbeEnvironment
): Promise<BatterySnapshot> {
  const sampledAt = getEpochMs(env);
  const manager = await resolveBatteryManager(env);
  if (!manager) {
    return {
      supported: false,
      sampledAt,
    };
  }
  try {
    return toBatterySnapshot(manager, sampledAt);
  } catch {
    return {
      supported: false,
      sampledAt,
    };
  }
}

/**
 * Subscribe to `levelchange` and `chargingchange` events on the Battery Status API.
 * Returns an idempotent cleanup function; resolves to a no-op if unsupported.
 */
export async function observeBatteryChanges(
  onChange: (snap: BatterySnapshot) => void,
  env?: SystemProbeEnvironment
): Promise<() => void> {
  const manager = await resolveBatteryManager(env);
  if (!manager || typeof manager.addEventListener !== 'function') {
    return () => {};
  }

  let disposed = false;
  const handleBatteryEvent: EventListener = () => {
    if (disposed) {
      return;
    }
    try {
      const snap = toBatterySnapshot(manager, getEpochMs(env));
      onChange(snap);
    } catch {
      // Never let a listener error break battery event dispatch.
    }
  };

  try {
    manager.addEventListener('levelchange', handleBatteryEvent);
    manager.addEventListener('chargingchange', handleBatteryEvent);
  } catch {
    return () => {};
  }

  return () => {
    if (disposed) {
      return;
    }
    disposed = true;
    try {
      manager.removeEventListener?.('levelchange', handleBatteryEvent);
      manager.removeEventListener?.('chargingchange', handleBatteryEvent);
    } catch {
      // Ignore cleanup errors on restricted targets.
    }
  };
}

/**
 * Synchronously sample JS heap memory (`performance.memory` when available) and
 * include cached `measureUserAgentSpecificMemory()` bytes when cross-origin isolated.
 */
export function sampleMemorySnapshot(env?: SystemProbeEnvironment): MemorySnapshot {
  const sampledAt = getEpochMs(env);
  const isolated = isCrossOriginIsolated(env);
  const perf = getPerformance(env);

  let jsHeapUsedBytes: number | undefined;
  let jsHeapTotalBytes: number | undefined;
  let jsHeapSizeLimitBytes: number | undefined;

  try {
    const mem = perf?.memory;
    if (mem && typeof mem === 'object') {
      jsHeapUsedBytes = toFiniteNonNegativeInt(mem.usedJSHeapSize);
      jsHeapTotalBytes = toFiniteNonNegativeInt(mem.totalJSHeapSize);
      jsHeapSizeLimitBytes = toFiniteNonNegativeInt(mem.jsHeapSizeLimit);
    }
  } catch {
    // Some contexts restrict reading performance.memory.
  }

  const cachedUaBytes =
    isolated && perf && typeof perf === 'object' ? uaMemoryCache.get(perf) : undefined;

  return {
    ...(jsHeapUsedBytes !== undefined ? { jsHeapUsedBytes } : {}),
    ...(jsHeapTotalBytes !== undefined ? { jsHeapTotalBytes } : {}),
    ...(jsHeapSizeLimitBytes !== undefined ? { jsHeapSizeLimitBytes } : {}),
    ...(cachedUaBytes !== undefined ? { uaMemoryBytes: cachedUaBytes } : {}),
    crossOriginIsolated: isolated,
    sampledAt,
  };
}

/**
 * Measure total user-agent memory asynchronously via `performance.measureUserAgentSpecificMemory()`
 * when `crossOriginIsolated` is true, caching the reading for synchronous snapshots.
 */
export async function measureUserAgentMemoryBytes(
  env?: SystemProbeEnvironment
): Promise<number | undefined> {
  if (!isCrossOriginIsolated(env)) {
    return undefined;
  }
  const perf = getPerformance(env);
  if (!perf || typeof perf.measureUserAgentSpecificMemory !== 'function') {
    return undefined;
  }
  try {
    const measurement = await perf.measureUserAgentSpecificMemory();
    const bytes = toFiniteNonNegativeInt(measurement?.bytes);
    if (bytes !== undefined) {
      uaMemoryCache.set(perf, bytes);
    }
    return bytes;
  } catch {
    return undefined;
  }
}

/**
 * Sample memory asynchronously, refreshing `measureUserAgentSpecificMemory()` when
 * `crossOriginIsolated` is enabled.
 */
export async function sampleMemorySnapshotAsync(
  env?: SystemProbeEnvironment
): Promise<MemorySnapshot> {
  await measureUserAgentMemoryBytes(env);
  return sampleMemorySnapshot(env);
}

function parseParameterCountBillions(parameterLabel: string): number {
  const match = parameterLabel.match(/([0-9]+(?:\.[0-9]+)?)\s*B/i);
  if (!match || !match[1]) {
    return 2;
  }
  const parsed = Number.parseFloat(match[1]);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 2;
}

/**
 * Estimate total runtime memory/VRAM footprint (in bytes) for text-only inference:
 * model ONNX weights + tokenizer + KV cache for `contextTokens` + runtime activation buffers.
 */
export function estimateModelMemoryBytes(
  model: ModelSpec,
  dtype: QuantDtype,
  contextTokens = DEFAULT_CONTEXT_TOKENS
): number {
  const weightsBytes = model.textOnlyBytesByDtype[dtype] ?? 0;
  const safeTokens =
    typeof contextTokens === 'number' && Number.isFinite(contextTokens)
      ? Math.min(model.contextLength, Math.max(0, Math.round(contextTokens)))
      : DEFAULT_CONTEXT_TOKENS;

  const kvElementsPerToken =
    KV_ELEMENTS_PER_TOKEN_BY_MODEL[model.id] ??
    Math.round(16_384 * Math.max(0.5, parseParameterCountBillions(model.parameterLabel)));

  const usesFp16Kv = dtype === 'fp16' || (dtype === 'q4f16' && model.kvCacheFp16OnQ4f16);
  const kvBytesPerElement = usesFp16Kv ? 2 : 4;
  const kvCacheBytes = safeTokens * kvElementsPerToken * kvBytesPerElement;

  const activationAndRuntimeBytes =
    Math.max(MIN_RUNTIME_OVERHEAD_BYTES, Math.round(weightsBytes * ACTIVATION_WEIGHT_RATIO)) +
    Math.max(0, model.tokenizerBytes);

  return Math.round(Math.max(0, weightsBytes) + kvCacheBytes + activationAndRuntimeBytes);
}
