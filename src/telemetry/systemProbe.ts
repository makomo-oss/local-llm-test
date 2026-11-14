import { formatBytes } from '../models/catalog.ts';
import type { DeviceOption, ModelId, ModelSpec, QuantDtype, ResolvedDtype } from '../types.ts';

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

export type MemoryRiskLevel = 'ok' | 'caution' | 'high';

export interface MemoryRiskEnvironment {
  userAgent?: string;
  maxTouchPoints?: number;
  deviceMemoryGb?: number;
  jsHeapSizeLimitBytes?: number;
  coarsePointer?: boolean;
}

export interface MobileDeviceContext {
  isMobile: boolean;
  deviceMemoryGb?: number;
}

export interface MemoryRiskAssessment {
  level: MemoryRiskLevel;
  isMobile: boolean;
  estimatedRuntimeBytes: number;
  estimatedLoadPeakBytes: number;
  estimatedTabBudgetBytes?: number;
  headline: string;
  detail?: string;
}

const BYTES_PER_GIB = 1024 ** 3;
const MOBILE_DEFAULT_TAB_BUDGET_BYTES = Math.round(1.5 * BYTES_PER_GIB);
const WASM32_SAFE_RUNTIME_LIMIT_BYTES = Math.round(3.2 * BYTES_PER_GIB);

/**
 * Estimate total runtime memory/VRAM footprint (in bytes) for text-only inference:
 * model ONNX weights + tokenizer + KV cache for `contextTokens` + runtime activation buffers.
 */
export function estimateModelMemoryBytes(
  model: ModelSpec,
  dtype: QuantDtype,
  contextTokens = DEFAULT_CONTEXT_TOKENS,
  overrideWeightsBytes?: number
): number {
  const weightsBytes = overrideWeightsBytes ?? model.textOnlyBytesByDtype[dtype] ?? 0;
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

/**
 * Detect smartphone/tablet browser contexts (iOS Safari, iPadOS Safari, Android browsers)
 * and read `navigator.deviceMemory` without misclassifying desktop touchscreens.
 */
export function detectMobileDeviceContext(
  env?: MemoryRiskEnvironment
): MobileDeviceContext {
  const nav =
    typeof navigator !== 'undefined'
      ? (navigator as Navigator & { deviceMemory?: unknown; maxTouchPoints?: number })
      : undefined;

  const ua = env?.userAgent ?? nav?.userAgent ?? '';
  const maxTouchPoints =
    env?.maxTouchPoints ?? (typeof nav?.maxTouchPoints === 'number' ? nav.maxTouchPoints : 0);

  const rawDeviceMem =
    env && 'deviceMemoryGb' in env
      ? env.deviceMemoryGb
      : typeof nav?.deviceMemory === 'number'
        ? nav.deviceMemory
        : undefined;
  const deviceMemoryGb =
    typeof rawDeviceMem === 'number' && Number.isFinite(rawDeviceMem) && rawDeviceMem > 0
      ? rawDeviceMem
      : undefined;

  const hasCoarsePointer =
    env?.coarsePointer ??
    (typeof window !== 'undefined' && typeof window.matchMedia === 'function'
      ? window.matchMedia('(pointer: coarse)').matches
      : false);

  const uaLooksMobile = /Android|iPhone|iPad|iPod|Mobile|webOS|BlackBerry|IEMobile|Opera Mini/i.test(
    ua
  );
  const isIpadDesktopUa = /Macintosh/i.test(ua) && maxTouchPoints > 1;
  const isNonDesktopCoarse = hasCoarsePointer && !/Windows NT|Linux x86_64|X11/i.test(ua) && ua === '';

  return {
    isMobile: uaLooksMobile || isIpadDesktopUa || isNonDesktopCoarse,
    ...(deviceMemoryGb !== undefined ? { deviceMemoryGb } : {}),
  };
}

/**
 * Assess end-of-download ONNX initialization memory & GPU buffer risk so smartphones,
 * low-RAM browsers, and WebGPU devices warn cleanly before hitting Jetsam / OOM /
 * wasm32 / `maxStorageBufferBindingSize` ceilings.
 */
export function assessModelMemoryRisk(
  model: ModelSpec,
  resolved: Pick<
    ResolvedDtype,
    | 'dtype'
    | 'effectiveWeightsBytes'
    | 'activeRuntimeWeightsBytes'
    | 'onDemandEmbedTokens'
    | 'maxSingleTensorBytes'
  >,
  device: Pick<DeviceOption, 'backend' | 'limits'> | undefined,
  env?: MemoryRiskEnvironment
): MemoryRiskAssessment {
  const ctx = detectMobileDeviceContext(env);
  const fullEffectiveWeightsBytes =
    resolved.effectiveWeightsBytes ?? model.textOnlyBytesByDtype[resolved.dtype] ?? 0;
  const hasStreamingRuntime =
    Boolean(resolved.onDemandEmbedTokens) ||
    resolved.activeRuntimeWeightsBytes !== undefined;
  const activeWeightsBytes =
    resolved.activeRuntimeWeightsBytes ?? fullEffectiveWeightsBytes;

  const estimatedRuntimeBytes = estimateModelMemoryBytes(
    model,
    resolved.dtype,
    DEFAULT_CONTEXT_TOKENS,
    activeWeightsBytes
  );
  const estimatedLoadPeakBytes = hasStreamingRuntime
    ? Math.round(
        activeWeightsBytes * 1.18 +
          Math.max(
            resolved.maxSingleTensorBytes ?? 0,
            MIN_RUNTIME_OVERHEAD_BYTES,
            activeWeightsBytes * 0.15
          ) +
          Math.max(0, model.tokenizerBytes)
      )
    : Math.round(
        fullEffectiveWeightsBytes * 1.55 +
          Math.max(MIN_RUNTIME_OVERHEAD_BYTES, fullEffectiveWeightsBytes * 0.15) +
          Math.max(0, model.tokenizerBytes)
      );

  // 0. WebGPU single-tensor buffer size ceiling (`maxStorageBufferBindingSize` / `maxBufferSize`)
  if (
    device?.backend === 'webgpu' &&
    typeof resolved.maxSingleTensorBytes === 'number' &&
    Number.isFinite(resolved.maxSingleTensorBytes) &&
    resolved.maxSingleTensorBytes > 0
  ) {
    const candidateLimits = [
      device.limits?.maxStorageBufferBindingSize,
      device.limits?.maxBufferSize,
    ].filter((val): val is number => typeof val === 'number' && Number.isFinite(val) && val > 0);
    if (candidateLimits.length > 0) {
      const effectiveGpuLimit = Math.min(...candidateLimits);
      if (resolved.maxSingleTensorBytes > effectiveGpuLimit) {
        return {
          level: 'high',
          isMobile: ctx.isMobile,
          estimatedRuntimeBytes,
          estimatedLoadPeakBytes,
          estimatedTabBudgetBytes: effectiveGpuLimit,
          headline: `選択中の WebGPU デバイスの単一バッファ上限 (${formatBytes(effectiveGpuLimit)}) が ${model.name} (${resolved.dtype}) の最大テンソル (${formatBytes(resolved.maxSingleTensorBytes)}) より小さいためロードに失敗する可能性が高いです。`,
          detail:
            'CPU (WASM) デバイスへ切り替えるか、単一テンソルが小さい「Qwen 3.5 0.8B」(約 121.3 MiB) や「Gemma 4 E2B」(約 192 MiB) を選択してください。',
        };
      }
    }
  }

  // 1. CPU/WASM 32-bit linear address space ceiling (~3.2–4 GiB regardless of desktop/mobile)
  if (device?.backend === 'wasm' && estimatedRuntimeBytes >= WASM32_SAFE_RUNTIME_LIMIT_BYTES) {
    return {
      level: 'high',
      isMobile: ctx.isMobile,
      estimatedRuntimeBytes,
      estimatedLoadPeakBytes,
      estimatedTabBudgetBytes: WASM32_SAFE_RUNTIME_LIMIT_BYTES,
      headline: `CPU (WASM) で ${model.name} (${resolved.dtype}) を読み込むと推定 ${formatBytes(estimatedRuntimeBytes)} 以上となり、WASM のメモリ上限を超えてクラッシュ・再読込する可能性が高いです。`,
      detail: 'WebGPU デバイスへの切替、または Qwen 3.5 0.8B などの軽量モデルをおすすめします。',
    };
  }

  // 2. Smartphone / tablet per-tab memory limits (WebKit Jetsam / Android LMKD)
  if (ctx.isMobile) {
    const budgetBytes =
      ctx.deviceMemoryGb !== undefined
        ? Math.round(Math.min(ctx.deviceMemoryGb * 0.42, 2.2) * BYTES_PER_GIB)
        : MOBILE_DEFAULT_TAB_BUDGET_BYTES;

    if (estimatedLoadPeakBytes > budgetBytes * 1.15) {
      return {
        level: 'high',
        isMobile: true,
        estimatedRuntimeBytes,
        estimatedLoadPeakBytes,
        estimatedTabBudgetBytes: budgetBytes,
        headline: `スマホ/モバイル環境では ${model.name} (${resolved.dtype}) の読み込み終盤メモリが推定 ${formatBytes(estimatedLoadPeakBytes)} に達し、タブの上限 (目安 ${formatBytes(budgetBytes)}) を超えて強制終了・再読込される可能性が高いです。`,
        detail:
          'スマホでは超軽量な「Qwen 3.5 0.8B」+ WebGPU (または他のタブを閉じた状態) の利用を強く推奨します。',
      };
    }

    if (estimatedLoadPeakBytes > budgetBytes * 0.78) {
      return {
        level: 'caution',
        isMobile: true,
        estimatedRuntimeBytes,
        estimatedLoadPeakBytes,
        estimatedTabBudgetBytes: budgetBytes,
        headline: `スマホ環境での実行はメモリ余裕が少なめです (終盤ピーク推定 ${formatBytes(estimatedLoadPeakBytes)} / 目安 ${formatBytes(budgetBytes)})。`,
        detail:
          device?.backend === 'wasm'
            ? 'スマホでは CPU (WASM) より WebGPU (利用可能時) + Qwen 3.5 0.8B の方が省メモリで安定します。他のタブを閉じて実行してください。'
            : '他のアプリやブラウザタブを閉じてからロードすると安定します。',
      };
    }

    return {
      level: 'ok',
      isMobile: true,
      estimatedRuntimeBytes,
      estimatedLoadPeakBytes,
      estimatedTabBudgetBytes: budgetBytes,
      headline: 'モバイル環境でも実行可能な軽量サイズです。',
    };
  }

  // 3. Low-RAM Desktop / Chromebook (< 8 GiB deviceMemory; note Chrome caps navigator.deviceMemory at 8)
  if (ctx.deviceMemoryGb !== undefined && ctx.deviceMemoryGb < 8) {
    const budgetBytes = Math.round(ctx.deviceMemoryGb * 0.85 * BYTES_PER_GIB);
    if (estimatedLoadPeakBytes > budgetBytes * 1.15) {
      return {
        level: 'high',
        isMobile: false,
        estimatedRuntimeBytes,
        estimatedLoadPeakBytes,
        estimatedTabBudgetBytes: budgetBytes,
        headline: `端末の申告メモリ (${ctx.deviceMemoryGb} GiB) に対して ${model.name} の初期化メモリ (推定 ${formatBytes(estimatedLoadPeakBytes)}) が大きく、ブラウザがクラッシュする可能性があります。`,
        detail: '軽量な Qwen 3.5 0.8B や 2B モデル、またはより小さい量子化 (q4f16 / q4) をおすすめします。',
      };
    }
    if (estimatedLoadPeakBytes > budgetBytes * 0.8) {
      return {
        level: 'caution',
        isMobile: false,
        estimatedRuntimeBytes,
        estimatedLoadPeakBytes,
        estimatedTabBudgetBytes: budgetBytes,
        headline: `端末メモリ (${ctx.deviceMemoryGb} GiB) に対して比較的大きいモデルです。他タブを閉じて実行してください。`,
      };
    }
  }

  return {
    level: 'ok',
    isMobile: false,
    estimatedRuntimeBytes,
    estimatedLoadPeakBytes,
    headline: 'メモリ要件は許容範囲内です。',
  };
}
