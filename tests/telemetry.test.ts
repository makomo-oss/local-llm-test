import { describe, expect, it, vi } from 'vitest';
import { getModelById, MODEL_CATALOG } from '../src/models/catalog.ts';
import {
  BENCHMARK_STORAGE_KEY,
  BenchmarkStore,
  computeBenchmarkComparison,
  escapeCsvCell,
  type StorageLike,
} from '../src/telemetry/benchmarkStore.ts';
import {
  classifyPowerDevice,
  estimatePowerMetrics,
  LiveTelemetrySession,
  MainThreadLoadMonitor,
  type PerformanceObserverEntryListLike,
  type PerformanceObserverLike,
} from '../src/telemetry/metricsTracker.ts';
import {
  BENCHMARK_PRESETS,
  createDefaultGenerationConfig,
  getPresetById,
} from '../src/telemetry/presets.ts';
import {
  assessModelMemoryRisk,
  detectMobileDeviceContext,
  estimateModelMemoryBytes,
  measureUserAgentMemoryBytes,
  observeBatteryChanges,
  sampleBatteryStatus,
  sampleMemorySnapshot,
  sampleMemorySnapshotAsync,
  type BatteryManagerLike,
} from '../src/telemetry/systemProbe.ts';
import type { BenchmarkRecord, DeviceOption, InferenceMetrics } from '../src/types.ts';

function makeWebGpuDevice(overrides: Partial<DeviceOption> = {}): DeviceOption {
  return {
    id: overrides.id ?? 'webgpu-high-performance',
    label: overrides.label ?? 'WebGPU (High Performance): Mock GPU',
    backend: 'webgpu',
    gpuPreference: overrides.gpuPreference ?? 'high-performance',
    adapterInfo: overrides.adapterInfo ?? {
      vendor: 'nvidia',
      architecture: 'ampere',
      device: 'rtx-3070',
      description: 'NVIDIA GeForce RTX 3070',
      isFallbackAdapter: false,
    },
    features: overrides.features ?? ['shader-f16', 'subgroups'],
    limits: overrides.limits ?? { maxStorageBufferBindingSize: 1024 * 1024 * 1024 },
    supportsFp16: overrides.supportsFp16 ?? true,
    recommendedDtype: overrides.recommendedDtype ?? 'q4f16',
    available: true,
    runtimeDetails: overrides.runtimeDetails ?? {
      crossOriginIsolated: false,
      hardwareConcurrency: 8,
      simdSupported: true,
      numThreads: 1,
    },
  };
}

function makeWasmDevice(numThreads = 1): DeviceOption {
  return {
    id: 'wasm-cpu',
    label: `CPU (WASM, ${numThreads}スレッド)`,
    backend: 'wasm',
    features: ['wasm-simd'],
    limits: {},
    supportsFp16: false,
    recommendedDtype: 'q4',
    available: true,
    runtimeDetails: {
      crossOriginIsolated: numThreads > 1,
      hardwareConcurrency: 8,
      simdSupported: true,
      numThreads,
    },
  };
}

function makeMetrics(overrides: {
  tokensPerSec?: number;
  decodeTokensPerSec?: number;
  ttftMs?: number;
  estimatedVramBytes?: number;
  tokensPerWattHour?: number;
  generatedTokens?: number;
  totalMs?: number;
  powerWatts?: number;
} = {}): InferenceMetrics {
  const generatedTokens = overrides.generatedTokens ?? 100;
  const totalMs = overrides.totalMs ?? 4000;
  const ttftMs = overrides.ttftMs ?? 300;
  const tokensPerSec = overrides.tokensPerSec ?? 25;
  const decodeTokensPerSec = overrides.decodeTokensPerSec ?? 27;
  const estimatedPowerWatts = overrides.powerWatts ?? 30;
  const estimatedEnergyWh = (estimatedPowerWatts * totalMs) / 3_600_000;

  return {
    speed: {
      ttftMs,
      prefillMs: ttftMs,
      decodeMs: Math.max(0, totalMs - ttftMs),
      totalMs,
      promptTokens: 40,
      generatedTokens,
      tokensPerSec,
      decodeTokensPerSec,
      interTokenLatencyAvgMs: 37,
      interTokenLatencyP95Ms: 52,
    },
    load: {
      estimatedVramBytes: overrides.estimatedVramBytes ?? 1_500_000_000,
      mainThreadAvgFrameMs: 16.7,
      mainThreadCpuBusyRatio: 0.04,
      droppedFrames: 0,
      longTaskCount: 0,
      backend: 'webgpu',
      dtype: 'q4f16',
      activeThreads: 1,
    },
    power: {
      batterySupported: false,
      estimatedPowerWatts,
      estimatedEnergyWh,
      tokensPerWattHour:
        overrides.tokensPerWattHour ?? (generatedTokens > 0 ? generatedTokens / estimatedEnergyWh : undefined),
      estimationMethod: 'utilization-model',
    },
  };
}

function makeBenchmarkRecord(
  id: string,
  partial: Partial<BenchmarkRecord> = {},
  metricOverrides: Parameters<typeof makeMetrics>[0] = {}
): BenchmarkRecord {
  const model = getModelById(partial.modelId ?? 'qwen-3.5-2b');
  return {
    id,
    timestamp: partial.timestamp ?? 1_700_000_000_000,
    promptId: partial.promptId ?? 'standard-comparison',
    promptLabel: partial.promptLabel ?? 'Standard Benchmark',
    promptText: partial.promptText ?? 'Explain quantized KV caches.',
    modelId: model.id,
    modelName: partial.modelName ?? model.name,
    deviceId: partial.deviceId ?? 'webgpu-high-performance',
    deviceLabel: partial.deviceLabel ?? 'WebGPU RTX',
    backend: partial.backend ?? 'webgpu',
    gpuPreference: partial.gpuPreference ?? 'high-performance',
    dtype: partial.dtype ?? 'q4f16',
    generationConfig: partial.generationConfig ?? createDefaultGenerationConfig(model),
    status: partial.status ?? 'completed',
    outputPreview: partial.outputPreview ?? 'KV cache quantization reduces bandwidth...',
    metrics: partial.metrics ?? makeMetrics(metricOverrides),
    ...(partial.error ? { error: partial.error } : {}),
  };
}

function createMemoryStorage(initial: Record<string, string> = {}): StorageLike {
  const map = new Map<string, string>(Object.entries(initial));
  return {
    getItem: (key) => map.get(key) ?? null,
    setItem: (key, value) => {
      map.set(key, value);
    },
    removeItem: (key) => {
      map.delete(key);
    },
  };
}

describe('systemProbe', () => {
  it('samples supported Battery Status API accurately including IEEE-754 level rounding', async () => {
    const snap = await sampleBatteryStatus({
      navigator: {
        getBattery: async () => ({
          level: 0.57,
          charging: false,
          dischargingTime: 7200,
          chargingTime: Number.POSITIVE_INFINITY,
        }),
      },
      nowEpochMs: () => 123456,
    });

    expect(snap).toEqual({
      supported: true,
      level: 57,
      charging: false,
      dischargingTimeSec: 7200,
      sampledAt: 123456,
    });
  });

  it('falls back safely without throwing when getBattery is unavailable or rejects', async () => {
    const unsupported = await sampleBatteryStatus({
      navigator: {},
      nowEpochMs: () => 100,
    });
    expect(unsupported).toEqual({ supported: false, sampledAt: 100 });

    const rejected = await sampleBatteryStatus({
      navigator: {
        getBattery: async () => {
          throw new DOMException('Blocked by Permissions-Policy');
        },
      },
      nowEpochMs: () => 200,
    });
    expect(rejected).toEqual({ supported: false, sampledAt: 200 });

    const nullNav = await sampleBatteryStatus({
      navigator: null,
      nowEpochMs: () => 300,
    });
    expect(nullNav).toEqual({ supported: false, sampledAt: 300 });
  });

  it('observes battery levelchange/chargingchange and cleans up listeners cleanly', async () => {
    const listeners = new Map<string, Set<EventListener>>();
    const manager: BatteryManagerLike = {
      level: 0.8,
      charging: false,
      dischargingTime: 5400,
      chargingTime: Number.POSITIVE_INFINITY,
      addEventListener: (type, listener) => {
        const set = listeners.get(type) ?? new Set();
        set.add(listener as EventListener);
        listeners.set(type, set);
      },
      removeEventListener: (type, listener) => {
        listeners.get(type)?.delete(listener as EventListener);
      },
    };

    const events: Array<{ level?: number; charging?: boolean }> = [];
    const cleanup = await observeBatteryChanges(
      (snap) => events.push({ level: snap.level, charging: snap.charging }),
      { navigator: { getBattery: async () => manager }, nowEpochMs: () => 999 }
    );

    expect(listeners.get('levelchange')?.size).toBe(1);
    expect(listeners.get('chargingchange')?.size).toBe(1);

    manager.level = 0.78;
    for (const fn of listeners.get('levelchange') ?? []) {
      fn(new Event('levelchange'));
    }
    expect(events).toEqual([{ level: 78, charging: false }]);

    manager.charging = true;
    for (const fn of listeners.get('chargingchange') ?? []) {
      fn(new Event('chargingchange'));
    }
    expect(events).toHaveLength(2);
    expect(events[1]).toEqual({ level: 78, charging: true });

    cleanup();
    cleanup(); // idempotent
    expect(listeners.get('levelchange')?.size).toBe(0);
    expect(listeners.get('chargingchange')?.size).toBe(0);

    const noopCleanup = await observeBatteryChanges(() => {}, { navigator: {} });
    expect(() => noopCleanup()).not.toThrow();
  });

  it('samples performance.memory safely and caches measureUserAgentSpecificMemory when crossOriginIsolated', async () => {
    const fallbackSnap = sampleMemorySnapshot({
      performance: {},
      crossOriginIsolated: false,
      nowEpochMs: () => 50,
    });
    expect(fallbackSnap).toEqual({
      crossOriginIsolated: false,
      sampledAt: 50,
    });

    const measureMock = vi.fn(async () => ({ bytes: 420_000_000 }));
    const mockPerf = {
      memory: {
        usedJSHeapSize: 48_000_000,
        totalJSHeapSize: 64_000_000,
        jsHeapSizeLimit: 2_000_000_000,
      },
      measureUserAgentSpecificMemory: measureMock,
    };

    // Does not call measureUserAgentSpecificMemory when not crossOriginIsolated
    const nonIsolatedBytes = await measureUserAgentMemoryBytes({
      performance: mockPerf,
      crossOriginIsolated: false,
    });
    expect(nonIsolatedBytes).toBeUndefined();
    expect(measureMock).not.toHaveBeenCalled();

    // Calls and caches when crossOriginIsolated is true
    const asyncSnap = await sampleMemorySnapshotAsync({
      performance: mockPerf,
      crossOriginIsolated: true,
      nowEpochMs: () => 500,
    });
    expect(measureMock).toHaveBeenCalledTimes(1);
    expect(asyncSnap).toEqual({
      jsHeapUsedBytes: 48_000_000,
      jsHeapTotalBytes: 64_000_000,
      jsHeapSizeLimitBytes: 2_000_000_000,
      uaMemoryBytes: 420_000_000,
      crossOriginIsolated: true,
      sampledAt: 500,
    });

    // Subsequent synchronous snapshot reuses cached UA memory
    const syncSnap = sampleMemorySnapshot({
      performance: mockPerf,
      crossOriginIsolated: true,
      nowEpochMs: () => 600,
    });
    expect(syncSnap.uaMemoryBytes).toBe(420_000_000);
  });

  it('estimates model runtime memory footprint monotonically across dtypes and context lengths', () => {
    const gemma = getModelById('gemma-4-e2b');
    const memQ4f16_0 = estimateModelMemoryBytes(gemma, 'q4f16', 0);
    const memQ4f16_2048 = estimateModelMemoryBytes(gemma, 'q4f16', 2048);
    const memQ4f16_8192 = estimateModelMemoryBytes(gemma, 'q4f16', 8192);
    const memFp16_2048 = estimateModelMemoryBytes(gemma, 'fp16', 2048);

    expect(memQ4f16_0).toBeGreaterThan(gemma.textOnlyBytesByDtype.q4f16);
    expect(memQ4f16_2048).toBeGreaterThan(memQ4f16_0);
    expect(memQ4f16_8192).toBeGreaterThan(memQ4f16_2048);
    expect(memFp16_2048).toBeGreaterThan(memQ4f16_2048);

    // Clamps negative and oversized context tokens cleanly
    expect(estimateModelMemoryBytes(gemma, 'q4f16', -50)).toBe(memQ4f16_0);
    expect(estimateModelMemoryBytes(gemma, 'q4f16', gemma.contextLength * 10)).toBe(
      estimateModelMemoryBytes(gemma, 'q4f16', gemma.contextLength)
    );
  });
});

describe('estimatePowerMetrics', () => {
  const model = getModelById('qwen-3.5-2b');

  it('distinguishes discrete GPU, integrated/Apple GPU, fallback software GPU, and CPU WASM thread scaling', () => {
    const dGpu = makeWebGpuDevice({
      gpuPreference: 'high-performance',
      adapterInfo: {
        vendor: 'nvidia',
        architecture: 'ada',
        device: 'rtx-4070',
        description: 'NVIDIA GeForce RTX 4070',
        isFallbackAdapter: false,
      },
    });
    const appleGpu = makeWebGpuDevice({
      id: 'webgpu-apple',
      gpuPreference: 'high-performance',
      adapterInfo: {
        vendor: 'apple',
        architecture: 'common-3',
        device: 'apple-m3',
        description: 'Apple M3 GPU',
        isFallbackAdapter: false,
      },
    });
    const intelIGpu = makeWebGpuDevice({
      id: 'webgpu-low-power',
      gpuPreference: 'low-power',
      adapterInfo: {
        vendor: 'intel',
        architecture: 'gen-12lp',
        device: 'iris-xe',
        description: 'Intel Iris Xe Graphics',
        isFallbackAdapter: false,
      },
    });
    const fallbackGpu = makeWebGpuDevice({
      id: 'webgpu-fallback',
      gpuPreference: 'fallback',
      adapterInfo: {
        vendor: 'google',
        architecture: 'swiftshader',
        device: '',
        description: 'SwiftShader',
        isFallbackAdapter: true,
      },
    });
    const wasm1 = makeWasmDevice(1);
    const wasm4 = makeWasmDevice(4);

    expect(classifyPowerDevice('webgpu', dGpu)).toBe('discrete-gpu');
    expect(classifyPowerDevice('webgpu', appleGpu)).toBe('integrated-gpu');
    expect(classifyPowerDevice('webgpu', intelIGpu)).toBe('integrated-gpu');
    expect(classifyPowerDevice('webgpu', fallbackGpu)).toBe('software-gpu');
    expect(classifyPowerDevice('wasm', wasm4)).toBe('cpu-wasm');

    const pDGpu = estimatePowerMetrics({
      backend: 'webgpu',
      device: dGpu,
      model,
      dtype: 'q4f16',
      totalMs: 10_000,
      generatedTokens: 250,
      cpuBusyRatio: 0.1,
    });
    const pApple = estimatePowerMetrics({
      backend: 'webgpu',
      device: appleGpu,
      model,
      dtype: 'q4f16',
      totalMs: 10_000,
      generatedTokens: 250,
      cpuBusyRatio: 0.1,
    });
    const pWasm1 = estimatePowerMetrics({
      backend: 'wasm',
      device: wasm1,
      model,
      dtype: 'q4',
      totalMs: 10_000,
      generatedTokens: 40,
      cpuBusyRatio: 0.15,
    });
    const pWasm4 = estimatePowerMetrics({
      backend: 'wasm',
      device: wasm4,
      model,
      dtype: 'q4',
      totalMs: 10_000,
      generatedTokens: 90,
      cpuBusyRatio: 0.15,
    });

    expect(pDGpu.estimatedPowerWatts).toBeGreaterThan(pApple.estimatedPowerWatts);
    expect(pWasm4.estimatedPowerWatts).toBeGreaterThan(pWasm1.estimatedPowerWatts);
    expect(pDGpu.estimationMethod).toBe('utilization-model');
    expect(pDGpu.estimatedEnergyWh).toBeGreaterThan(0);
    expect(pDGpu.tokensPerWattHour).toBeGreaterThan(0);
  });

  it('uses battery-api+utilization on battery discharge and utilization-model when charging or unsupported', () => {
    const iGpu = makeWebGpuDevice({
      gpuPreference: 'low-power',
      adapterInfo: {
        vendor: 'amd',
        architecture: 'rdna3',
        device: '780m',
        description: 'AMD Radeon 780M Graphics',
        isFallbackAdapter: false,
      },
    });

    const discharging = estimatePowerMetrics({
      backend: 'webgpu',
      device: iGpu,
      model,
      dtype: 'q4f16',
      totalMs: 30_000,
      generatedTokens: 600,
      cpuBusyRatio: 0.12,
      batteryStart: { supported: true, level: 82, charging: false, dischargingTimeSec: 10800, sampledAt: 1 },
      batteryEnd: { supported: true, level: 81.5, charging: false, dischargingTimeSec: 10400, sampledAt: 30001 },
    });

    expect(discharging.batterySupported).toBe(true);
    expect(discharging.estimationMethod).toBe('battery-api+utilization');
    expect(discharging.batteryLevelStart).toBe(82);
    expect(discharging.batteryLevelEnd).toBe(81.5);
    expect(discharging.batteryDeltaPct).toBe(-0.5);
    expect(discharging.batteryCharging).toBe(false);
    expect(discharging.batteryDischargingTimeSec).toBe(10400);
    expect(discharging.tokensPerWattHour).toBeGreaterThan(0);

    const pluggedIn = estimatePowerMetrics({
      backend: 'webgpu',
      device: iGpu,
      model,
      dtype: 'q4f16',
      totalMs: 30_000,
      generatedTokens: 600,
      cpuBusyRatio: 0.12,
      batteryStart: { supported: true, level: 82, charging: true, sampledAt: 1 },
      batteryEnd: { supported: true, level: 83, charging: true, sampledAt: 30001 },
    });

    expect(pluggedIn.batterySupported).toBe(true);
    expect(pluggedIn.estimationMethod).toBe('utilization-model');
    expect(pluggedIn.batteryDeltaPct).toBe(1);
    expect(pluggedIn.batteryCharging).toBe(true);

    const zeroTokens = estimatePowerMetrics({
      backend: 'wasm',
      model,
      dtype: 'q4',
      totalMs: 0,
      generatedTokens: 0,
      cpuBusyRatio: 0,
    });
    expect(zeroTokens.estimatedEnergyWh).toBe(0);
    expect(zeroTokens.tokensPerWattHour).toBeUndefined();
    expect(Number.isFinite(zeroTokens.estimatedPowerWatts)).toBe(true);
  });
});

describe('MainThreadLoadMonitor', () => {
  it('operates safely when requestAnimationFrame and PerformanceObserver are missing', () => {
    let clock = 1000;
    const monitor = new MainThreadLoadMonitor({
      requestAnimationFrame: null,
      cancelAnimationFrame: null,
      PerformanceObserver: null,
      now: () => clock,
    });

    monitor.start();
    clock = 1500;
    const mid = monitor.snapshot();
    expect(mid).toEqual({
      mainThreadAvgFrameMs: 0,
      mainThreadCpuBusyRatio: 0,
      droppedFrames: 0,
      longTaskCount: 0,
      elapsedMs: 500,
      sampleCount: 0,
    });

    clock = 1750;
    const end = monitor.stop();
    expect(end.elapsedMs).toBe(750);
    // Calling stop again returns frozen snapshot
    clock = 2500;
    expect(monitor.stop().elapsedMs).toBe(750);
  });

  it('tracks rAF dropped frames (>25ms) and PerformanceObserver longtask entries safely', () => {
    let clock = 0;
    let nextId = 1;
    const pendingCallbacks = new Map<number, (ts: number) => void>();
    const cancelSpy = vi.fn((id: number) => {
      pendingCallbacks.delete(id);
    });

    let observerCb: ((list: PerformanceObserverEntryListLike) => void) | undefined;
    const disconnectSpy = vi.fn();

    class FakePerformanceObserver implements PerformanceObserverLike {
      static supportedEntryTypes = ['longtask'];
      constructor(cb: (list: PerformanceObserverEntryListLike) => void) {
        observerCb = cb;
      }
      observe(): void {}
      disconnect(): void {
        disconnectSpy();
      }
    }

    const monitor = new MainThreadLoadMonitor({
      requestAnimationFrame: (cb) => {
        const id = nextId++;
        pendingCallbacks.set(id, cb);
        return id;
      },
      cancelAnimationFrame: cancelSpy,
      PerformanceObserver: FakePerformanceObserver,
      now: () => clock,
    });

    monitor.start();

    const stepRaf = (ts: number) => {
      clock = ts;
      const entries = [...pendingCallbacks.entries()];
      pendingCallbacks.clear();
      for (const [, cb] of entries) {
        cb(ts);
      }
    };

    // Frame deltas: 16ms, 40ms (dropped!), 16ms -> avg = 24ms, dropped = 1
    stepRaf(0);
    stepRaf(16);
    stepRaf(56);
    stepRaf(72);

    observerCb?.({
      getEntries: () => [{ entryType: 'longtask', duration: 60 }],
    });

    clock = 100;
    const finalSnap = monitor.stop();
    expect(finalSnap.sampleCount).toBe(3);
    expect(finalSnap.droppedFrames).toBe(1);
    expect(finalSnap.mainThreadAvgFrameMs).toBe(24);
    expect(finalSnap.longTaskCount).toBe(1);
    expect(finalSnap.mainThreadCpuBusyRatio).toBeCloseTo(0.6, 2);
    expect(cancelSpy).toHaveBeenCalledTimes(1);
    expect(disconnectSpy).toHaveBeenCalledTimes(1);
  });

  it('swallows PerformanceObserver exceptions in restricted browsers', () => {
    class ThrowingObserver implements PerformanceObserverLike {
      constructor() {
        throw new Error('Not supported');
      }
      observe(): void {}
      disconnect(): void {}
    }

    const monitor = new MainThreadLoadMonitor({
      requestAnimationFrame: null,
      PerformanceObserver: ThrowingObserver,
      now: () => 10,
    });
    expect(() => monitor.start()).not.toThrow();
    expect(monitor.stop().longTaskCount).toBe(0);
  });
});

describe('LiveTelemetrySession', () => {
  it('coordinates battery, main-thread monitor, live streaming metrics, and final InferenceMetrics', async () => {
    let clock = 1000;
    let batteryLevel = 0.75;
    const model = getModelById('qwen-3.5-0.8b');
    const device = makeWebGpuDevice({ gpuPreference: 'low-power' });

    const session = new LiveTelemetrySession({
      model,
      device,
      dtype: 'q4f16',
      env: {
        now: () => clock,
        nowEpochMs: () => 1_700_000_000_000 + clock,
        requestAnimationFrame: null,
        PerformanceObserver: null,
        navigator: {
          getBattery: async () => ({
            level: batteryLevel,
            charging: false,
            dischargingTime: 9000,
            chargingTime: Number.POSITIVE_INFINITY,
          }),
        },
        performance: {
          memory: {
            usedJSHeapSize: 60_000_000,
            totalJSHeapSize: 90_000_000,
          },
        },
      },
    });

    const initial = await session.start(32);
    expect(initial.speed.generatedTokens).toBe(0);
    expect(initial.speed.ttftMs).toBeUndefined();
    expect(initial.memory.jsHeapUsedBytes).toBe(60_000_000);

    session.onGenerationStarted(48);

    clock = 1200; // TTFT = 200ms
    const tok1 = session.onToken(1, clock);
    expect(tok1.speed.ttftMs).toBe(200);
    expect(tok1.speed.generatedTokens).toBe(1);

    clock = 1240; // +40ms (25 tok/s instant)
    session.onToken(2, clock);
    clock = 1280; // +40ms
    const tok3 = session.onToken(3, clock);
    expect(tok3.speed.instantTokensPerSec).toBe(25);
    expect(tok3.speed.rollingTokensPerSec).toBe(25);
    expect(tok3.speed.decodeTokensPerSec).toBe(25);
    expect(tok3.estimatedVramBytes).toBeGreaterThan(model.textOnlyBytesByDtype.q4f16);

    clock = 1400;
    batteryLevel = 0.749;
    const metrics = await session.finish({
      workerJsHeapUsedBytes: 120_000_000,
      workerJsHeapTotalBytes: 180_000_000,
    });

    expect(metrics.speed.ttftMs).toBe(200);
    expect(metrics.speed.totalMs).toBe(400);
    expect(metrics.speed.generatedTokens).toBe(3);
    expect(metrics.speed.promptTokens).toBe(48);
    expect(metrics.load.jsHeapUsedBytes).toBe(120_000_000);
    expect(metrics.load.backend).toBe('webgpu');
    expect(metrics.load.dtype).toBe('q4f16');
    expect(metrics.power.batterySupported).toBe(true);
    expect(metrics.power.batteryDeltaPct).toBe(-0.1);
    expect(metrics.power.estimationMethod).toBe('battery-api+utilization');
  });
});

describe('BenchmarkStore', () => {
  it('persists under local-llm-test:benchmarks:v1, reloads, summarizes, and removes/clears safely', () => {
    const storage = createMemoryStorage();
    const store = new BenchmarkStore({ storage });

    const r1 = makeBenchmarkRecord(
      'run-1',
      { timestamp: 1000, modelId: 'qwen-3.5-0.8b', deviceId: 'wasm-cpu', backend: 'wasm' },
      { tokensPerSec: 12, ttftMs: 900, estimatedVramBytes: 750_000_000, tokensPerWattHour: 2200 }
    );
    const r2 = makeBenchmarkRecord(
      'run-2',
      { timestamp: 2000, modelId: 'qwen-3.5-2b', deviceId: 'webgpu-hp', backend: 'webgpu' },
      { tokensPerSec: 42, ttftMs: 250, estimatedVramBytes: 1_800_000_000, tokensPerWattHour: 5400 }
    );

    store.add(r1);
    store.add(r2);

    expect(storage.getItem(BENCHMARK_STORAGE_KEY)).toBeTruthy();

    // Reload from same storage key
    const reloaded = new BenchmarkStore({ storage });
    const list = reloaded.list();
    expect(list.map((r) => r.id)).toEqual(['run-2', 'run-1']);

    const stats = reloaded.getSummaryStats();
    expect(stats.totalRuns).toBe(2);
    expect(stats.completedRuns).toBe(2);
    expect(stats.uniqueModels).toBe(2);
    expect(stats.uniqueDevices).toBe(2);
    expect(stats.maxTokensPerSec).toBe(42);
    expect(stats.minTtftMs).toBe(250);

    expect(reloaded.remove('run-1')).toBe(true);
    expect(reloaded.remove('missing')).toBe(false);
    expect(reloaded.list()).toHaveLength(1);

    reloaded.clear();
    expect(reloaded.list()).toHaveLength(0);
    expect(storage.getItem(BENCHMARK_STORAGE_KEY)).toBeNull();
  });

  it('handles QuotaExceededError, corrupted JSON, and missing storage gracefully', () => {
    const fullStorage: StorageLike = {
      getItem: () => '{bad json',
      setItem: () => {
        throw new DOMException('Quota exceeded', 'QuotaExceededError');
      },
      removeItem: () => {
        throw new Error('Blocked');
      },
    };

    const store = new BenchmarkStore({ storage: fullStorage });
    expect(store.list()).toEqual([]);
    expect(store.getStorageError()).toBeTruthy();

    const r1 = makeBenchmarkRecord('run-quota');
    expect(() => store.add(r1)).not.toThrow();
    expect(store.list()).toHaveLength(1);
    expect(() => store.clear()).not.toThrow();

    const memoryOnly = new BenchmarkStore({ storage: null });
    memoryOnly.add(r1);
    expect(memoryOnly.list()).toHaveLength(1);
  });

  it('exports valid JSON and RFC 4180 CSV escaping commas, quotes, and newlines', () => {
    expect(escapeCsvCell('plain')).toBe('plain');
    expect(escapeCsvCell('hello, world')).toBe('"hello, world"');
    expect(escapeCsvCell('say "hi"')).toBe('"say ""hi"""');
    expect(escapeCsvCell('line1\nline2')).toBe('"line1\nline2"');

    const store = new BenchmarkStore({ storage: null });
    store.add(
      makeBenchmarkRecord('csv-1', {
        promptLabel: 'Test, "Quoted"\nPrompt',
        outputPreview: 'Result with, comma and "quotes"\nand newline',
      })
    );

    const json = JSON.parse(store.exportJson()) as {
      version: number;
      records: BenchmarkRecord[];
      summary: { totalRuns: number };
    };
    expect(json.version).toBe(1);
    expect(json.summary.totalRuns).toBe(1);
    expect(json.records[0]?.id).toBe('csv-1');

    const csv = store.exportCsv();
    expect(csv).toContain('"Test, ""Quoted""\nPrompt"');
    expect(csv).toContain('"Result with, comma and ""quotes""\nand newline"');
  });

  it('computes comparison rankings across completed runs while excluding aborted/error runs', () => {
    const fastest = makeBenchmarkRecord(
      'fastest',
      { modelId: 'qwen-3.5-0.8b', timestamp: 10 },
      { tokensPerSec: 65, ttftMs: 180, estimatedVramBytes: 900_000_000, tokensPerWattHour: 4000 }
    );
    const lightest = makeBenchmarkRecord(
      'lightest',
      { modelId: 'qwen-3.5-0.8b', timestamp: 20 },
      { tokensPerSec: 30, ttftMs: 240, estimatedVramBytes: 700_000_000, tokensPerWattHour: 8500 }
    );
    const erroredZeroTtft = makeBenchmarkRecord(
      'errored',
      { status: 'error', error: 'OOM', timestamp: 30 },
      { tokensPerSec: 0, ttftMs: 0, generatedTokens: 0, estimatedVramBytes: 100 }
    );

    const comparison = computeBenchmarkComparison([fastest, lightest, erroredZeroTtft]);
    expect(comparison.totalRecords).toBe(3);
    expect(comparison.comparableCount).toBe(2);
    expect(comparison.excludedCount).toBe(1);

    expect(comparison.winners.fastestThroughput?.record.id).toBe('fastest');
    expect(comparison.winners.lowestTtft?.record.id).toBe('fastest');
    expect(comparison.winners.lowestMemory?.record.id).toBe('lightest');
    expect(comparison.winners.bestEfficiency?.record.id).toBe('lightest');
    expect(comparison.fastestThroughput.map((e) => e.record.id)).toEqual(['fastest', 'lightest']);
  });
});

describe('presets', () => {
  it('provides curated Japanese and English benchmark presets with unique IDs', () => {
    expect(BENCHMARK_PRESETS.length).toBeGreaterThanOrEqual(5);
    const ids = new Set(BENCHMARK_PRESETS.map((p) => p.id));
    expect(ids.size).toBe(BENCHMARK_PRESETS.length);

    expect(getPresetById('quick-qa-ja')?.language).toBe('ja');
    expect(getPresetById('ja-summarization')?.category).toBe('summarization');
    expect(getPresetById('logic-reasoning')?.recommendedEnableThinking).toBe(true);
    expect(getPresetById('ts-codegen')?.category).toBe('coding');
    expect(getPresetById('standard-comparison')?.language).toBe('en');
    expect(getPresetById('non-existent')).toBeUndefined();
  });

  it('creates safe default GenerationConfig aligned with model thinking capability', () => {
    for (const model of MODEL_CATALOG) {
      const cfg = createDefaultGenerationConfig(model);
      expect(cfg.maxNewTokens).toBeGreaterThan(0);
      expect(cfg.temperature).toBeGreaterThan(0);
      expect(cfg.topP).toBeGreaterThan(0);
      expect(cfg.topP).toBeLessThanOrEqual(1);
      expect(cfg.topK).toBeGreaterThanOrEqual(1);
      expect(cfg.enableThinking).toBe(model.defaultThinkingEnabled);
    }

    const qwen4b = getModelById('qwen-3.5-4b');
    expect(createDefaultGenerationConfig(qwen4b).enableThinking).toBe(true);
    expect(createDefaultGenerationConfig(qwen4b).maxNewTokens).toBe(1024);

    // Llama 3.2 3B does not support thinking even if override asks for it
    const llama = getModelById('llama-3.2-3b');
    expect(createDefaultGenerationConfig(llama, { enableThinking: true }).enableThinking).toBe(
      false
    );
  });
});

describe('mobile memory risk assessment', () => {
  it('accurately identifies smartphones/iPadOS vs Windows touchscreens', () => {
    expect(
      detectMobileDeviceContext({
        userAgent:
          'Mozilla/5.0 (iPhone; CPU iPhone OS 18_3 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1',
      }).isMobile
    ).toBe(true);

    expect(
      detectMobileDeviceContext({
        userAgent:
          'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)',
        maxTouchPoints: 5,
      }).isMobile
    ).toBe(true);

    expect(
      detectMobileDeviceContext({
        userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/128.0 Mobile',
        deviceMemoryGb: 8,
      })
    ).toEqual({ isMobile: true, deviceMemoryGb: 8 });

    expect(
      detectMobileDeviceContext({
        userAgent:
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36',
        maxTouchPoints: 10,
        deviceMemoryGb: 8,
      })
    ).toEqual({ isMobile: false, deviceMemoryGb: 8 });
  });

  it('flags oversized mobile models and WASM >3.2 GiB loads as high risk without false-alarming desktop WebGPU', () => {
    const webgpuDev = makeWebGpuDevice();
    const wasmDev = makeWasmDevice(1);

    const qwen08 = getModelById('qwen-3.5-0.8b');
    const qwen2 = getModelById('qwen-3.5-2b');
    const gemmaE4b = getModelById('gemma-4-e4b');

    const iphoneEnv = {
      userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Mobile/15E148',
    };
    expect(
      assessModelMemoryRisk(qwen08, { dtype: 'q4f16' }, webgpuDev, iphoneEnv).level
    ).toBe('ok');
    expect(
      assessModelMemoryRisk(
        qwen08,
        { dtype: 'q4', effectiveWeightsBytes: 508_559_997 + 486_301_511 },
        wasmDev,
        iphoneEnv
      ).level
    ).toBe('caution');
    expect(
      assessModelMemoryRisk(qwen2, { dtype: 'q4f16' }, webgpuDev, iphoneEnv).level
    ).toBe('high');
    expect(
      assessModelMemoryRisk(gemmaE4b, { dtype: 'q4f16' }, webgpuDev, iphoneEnv).level
    ).toBe('high');

    // Desktop Chrome caps deviceMemory at 8 even with >32GB RAM — do not warn on desktop WebGPU E4B
    const desktopEnv = {
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0',
      deviceMemoryGb: 8,
    };
    expect(
      assessModelMemoryRisk(gemmaE4b, { dtype: 'q4f16' }, webgpuDev, desktopEnv).level
    ).toBe('ok');

    // Desktop WASM Gemma 4 E4B (>17 GB) still exceeds 32-bit WASM ceiling -> high
    expect(
      assessModelMemoryRisk(
        gemmaE4b,
        { dtype: 'q4', effectiveWeightsBytes: 13_958_647_953 + 3_380_723_437 },
        wasmDev,
        desktopEnv
      ).level
    ).toBe('high');
  });
});
