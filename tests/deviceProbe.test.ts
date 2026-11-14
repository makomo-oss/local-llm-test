import { describe, expect, it, vi } from 'vitest';
import {
  acquireGpuDeviceForPreference,
  getDeviceRuntimeDetails,
  pickDefaultDevice,
  probeComputeDevices,
  toWorkerDeviceTarget,
} from '../src/engine/deviceProbe.ts';

interface MockAdapterSpec {
  info: Partial<GPUAdapterInfo>;
  features?: string[];
  limits?: Record<string, number>;
  requestDevice?: (descriptor?: GPUDeviceDescriptor) => Promise<GPUDevice>;
}

function createMockDevice(grantedFeatures: string[] = []): GPUDevice {
  return {
    features: new Set(grantedFeatures) as unknown as GPUSupportedFeatures,
  } as unknown as GPUDevice;
}

function createMockAdapter(spec: MockAdapterSpec): GPUAdapter {
  const featureSet = new Set(spec.features ?? []);
  const defaultLimits: Record<string, number> = {
    maxBindGroups: 4,
    maxComputeWorkgroupStorageSize: 32768,
    maxComputeWorkgroupsPerDimension: 65535,
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 1073741824,
    maxBufferSize: 2147483648,
    maxComputeInvocationsPerWorkgroup: 1024,
    maxComputeWorkgroupSizeX: 1024,
    maxComputeWorkgroupSizeY: 1024,
    maxComputeWorkgroupSizeZ: 64,
    ...spec.limits,
  };

  return {
    info: {
      vendor: spec.info.vendor ?? '',
      architecture: spec.info.architecture ?? '',
      device: spec.info.device ?? '',
      description: spec.info.description ?? '',
      isFallbackAdapter: spec.info.isFallbackAdapter ?? false,
      subgroupMinSize: spec.info.subgroupMinSize ?? 4,
      subgroupMaxSize: spec.info.subgroupMaxSize ?? 64,
    } as GPUAdapterInfo,
    features: featureSet as unknown as GPUSupportedFeatures,
    limits: defaultLimits as unknown as GPUSupportedLimits,
    requestDevice:
      spec.requestDevice ??
      vi.fn(async (descriptor?: GPUDeviceDescriptor) =>
        createMockDevice(
          descriptor?.requiredFeatures ? [...descriptor.requiredFeatures] : [...featureSet]
        )
      ),
  } as unknown as GPUAdapter;
}

describe('getDeviceRuntimeDetails', () => {
  it('forces 1 thread when crossOriginIsolated is false and scales up to 4 when isolated', () => {
    expect(
      getDeviceRuntimeDetails({
        crossOriginIsolated: false,
        hardwareConcurrency: 16,
        simdSupported: true,
        sharedArrayBufferAvailable: true,
      })
    ).toEqual({
      crossOriginIsolated: false,
      hardwareConcurrency: 16,
      simdSupported: true,
      numThreads: 1,
    });

    expect(
      getDeviceRuntimeDetails({
        crossOriginIsolated: true,
        hardwareConcurrency: 16,
        simdSupported: true,
        sharedArrayBufferAvailable: true,
      })
    ).toEqual({
      crossOriginIsolated: true,
      hardwareConcurrency: 16,
      simdSupported: true,
      numThreads: 4,
    });

    expect(
      getDeviceRuntimeDetails({
        crossOriginIsolated: true,
        hardwareConcurrency: 3,
        simdSupported: true,
        sharedArrayBufferAvailable: true,
      }).numThreads
    ).toBe(2);

    expect(
      getDeviceRuntimeDetails({
        crossOriginIsolated: true,
        hardwareConcurrency: 8,
        simdSupported: false,
        sharedArrayBufferAvailable: false,
      }).numThreads
    ).toBe(1);
  });
});

describe('probeComputeDevices', () => {
  it('deduplicates identical GPU adapters on single-GPU machines and appends WASM CPU', async () => {
    const singleGpu = createMockAdapter({
      info: {
        vendor: 'apple',
        architecture: 'common-3',
        device: 'apple-m3',
        description: 'Apple M3 Pro',
        isFallbackAdapter: false,
      },
      features: ['shader-f16', 'subgroups', 'timestamp-query'],
    });

    const mockGpu = {
      requestAdapter: vi.fn(async (opts?: GPURequestAdapterOptions) => {
        if (opts?.forceFallbackAdapter) {
          return null;
        }
        return singleGpu;
      }),
    } as unknown as GPU;

    const devices = await probeComputeDevices({
      gpu: mockGpu,
      crossOriginIsolated: false,
      hardwareConcurrency: 12,
      simdSupported: true,
    });

    expect(devices).toHaveLength(2);
    expect(devices[0]?.id).toBe('webgpu-high-performance');
    expect(devices[0]?.available).toBe(true);
    expect(devices[0]?.supportsFp16).toBe(true);
    expect(devices[0]?.recommendedDtype).toBe('q4f16');
    expect(devices[0]?.label).toContain('Apple M3 Pro');

    expect(devices[1]?.id).toBe('wasm-cpu');
    expect(devices[1]?.backend).toBe('wasm');
    expect(devices[1]?.available).toBe(true);
    expect(devices[1]?.recommendedDtype).toBe('q4');
    expect(devices[1]?.runtimeDetails.numThreads).toBe(1);

    expect(pickDefaultDevice(devices)?.id).toBe('webgpu-high-performance');
  });

  it('keeps distinct dGPU, iGPU, and software fallback adapters separate', async () => {
    const dGpu = createMockAdapter({
      info: {
        vendor: 'nvidia',
        architecture: 'ada-lovelace',
        device: '0x2704',
        description: 'NVIDIA GeForce RTX 4080',
        isFallbackAdapter: false,
      },
      features: ['shader-f16', 'subgroups', 'subgroup-size-control'],
      limits: { maxBufferSize: 4_294_967_296 },
    });

    const iGpu = createMockAdapter({
      info: {
        vendor: 'intel',
        architecture: 'gen-12lp',
        device: '0x46a6',
        description: 'Intel Iris Xe Graphics',
        isFallbackAdapter: false,
      },
      features: ['subgroups'],
      limits: { maxBufferSize: 1_073_741_824 },
    });

    const swAdapter = createMockAdapter({
      info: {
        vendor: 'google',
        architecture: 'swiftshader',
        device: '0x0000',
        description: 'Google SwiftShader',
        isFallbackAdapter: true,
      },
      features: [],
      limits: { maxBufferSize: 268_435_456 },
    });

    const mockGpu = {
      requestAdapter: vi.fn(async (opts?: GPURequestAdapterOptions) => {
        if (opts?.forceFallbackAdapter) return swAdapter;
        if (opts?.powerPreference === 'low-power') return iGpu;
        return dGpu; // both high-performance and default return dGpu
      }),
    } as unknown as GPU;

    const devices = await probeComputeDevices({
      gpu: mockGpu,
      crossOriginIsolated: true,
      hardwareConcurrency: 16,
      simdSupported: true,
      sharedArrayBufferAvailable: true,
    });

    expect(devices.map((d) => d.id)).toEqual([
      'webgpu-high-performance',
      'webgpu-low-power',
      'webgpu-fallback',
      'wasm-cpu',
    ]);
    expect(devices[0]?.recommendedDtype).toBe('q4f16');
    expect(devices[1]?.supportsFp16).toBe(false);
    expect(devices[1]?.recommendedDtype).toBe('q4');
    expect(devices[2]?.adapterInfo?.isFallbackAdapter).toBe(true);
    expect(devices[2]?.label).toContain('Fallback');
    expect(devices[3]?.runtimeDetails.numThreads).toBe(4);
  });

  it('distinguishes adapters with redacted info by limits while deduplicating identical redacted ones', async () => {
    const redactedHigh = createMockAdapter({
      info: {},
      features: ['shader-f16'],
      limits: { maxBufferSize: 4_294_967_296 },
    });
    const redactedLow = createMockAdapter({
      info: {},
      features: [],
      limits: { maxBufferSize: 1_073_741_824 },
    });

    const mockGpu = {
      requestAdapter: vi.fn(async (opts?: GPURequestAdapterOptions) => {
        if (opts?.forceFallbackAdapter) return null;
        if (opts?.powerPreference === 'low-power') return redactedLow;
        return redactedHigh;
      }),
    } as unknown as GPU;

    const devices = await probeComputeDevices({ gpu: mockGpu });
    expect(devices.map((d) => d.id)).toEqual([
      'webgpu-high-performance',
      'webgpu-low-power',
      'wasm-cpu',
    ]);
  });

  it('handles software-fallback-only systems cleanly', async () => {
    const swAdapter = createMockAdapter({
      info: {
        vendor: 'google',
        architecture: 'swiftshader',
        description: 'SwiftShader Device',
        isFallbackAdapter: true,
      },
      features: [],
    });

    const mockGpu = {
      requestAdapter: vi.fn(async (opts?: GPURequestAdapterOptions) => {
        if (opts?.powerPreference === 'high-performance' || opts?.powerPreference === 'low-power') {
          return null;
        }
        return swAdapter;
      }),
    } as unknown as GPU;

    const devices = await probeComputeDevices({ gpu: mockGpu });
    expect(devices.map((d) => d.id)).toEqual(['webgpu-default', 'wasm-cpu']);
    expect(devices[0]?.adapterInfo?.isFallbackAdapter).toBe(true);
    expect(devices[0]?.label).toContain('Fallback');
    expect(pickDefaultDevice(devices)?.id).toBe('webgpu-default');
  });

  it('falls back to WASM CPU when WebGPU is unavailable or adapters fail', async () => {
    const noGpuDevices = await probeComputeDevices({
      gpu: null,
      crossOriginIsolated: false,
      hardwareConcurrency: 4,
      simdSupported: true,
    });

    expect(noGpuDevices).toHaveLength(2);
    expect(noGpuDevices[0]?.available).toBe(false);
    expect(noGpuDevices[0]?.unavailableReason).toBeTruthy();
    expect(noGpuDevices[1]?.id).toBe('wasm-cpu');
    expect(noGpuDevices[1]?.available).toBe(true);
    expect(pickDefaultDevice(noGpuDevices)?.id).toBe('wasm-cpu');

    // Also when navigator.gpu exists but all adapter requests reject
    const failingGpu = {
      requestAdapter: vi.fn(async () => {
        throw new Error('GPU blocklisted');
      }),
    } as unknown as GPU;

    const failedDevices = await probeComputeDevices({ gpu: failingGpu });
    expect(failedDevices[0]?.available).toBe(false);
    expect(failedDevices[1]?.id).toBe('wasm-cpu');
    expect(pickDefaultDevice(failedDevices)?.id).toBe('wasm-cpu');
  });

  it('builds serializable WorkerDeviceTarget descriptors for WebGPU and WASM', async () => {
    const gpuAdapter = createMockAdapter({
      info: { vendor: 'amd', description: 'Radeon RX 7800 XT' },
      features: ['shader-f16'],
    });
    const mockGpu = {
      requestAdapter: vi.fn(async (opts?: GPURequestAdapterOptions) =>
        opts?.forceFallbackAdapter ? null : gpuAdapter
      ),
    } as unknown as GPU;

    const devices = await probeComputeDevices({ gpu: mockGpu });
    const gpuTarget = toWorkerDeviceTarget(devices[0]!);
    expect(gpuTarget).toEqual({
      backend: 'webgpu',
      gpuPreference: 'high-performance',
      expectedDeviceFingerprint: 'amd|||radeon rx 7800 xt|hw',
    });

    const wasmTarget = toWorkerDeviceTarget(devices[1]!);
    expect(wasmTarget.backend).toBe('wasm');
  });
});

describe('acquireGpuDeviceForPreference', () => {
  it('requests ORT features and clamped limits directly from matching adapter (worker-compatible)', async () => {
    const requestDeviceSpy = vi.fn(async (descriptor?: GPUDeviceDescriptor) =>
      createMockDevice((descriptor?.requiredFeatures as string[]) ?? [])
    );
    const adapter = createMockAdapter({
      info: {
        vendor: 'nvidia',
        architecture: 'ampere',
        device: '0x2204',
        description: 'RTX 3090',
      },
      features: [
        'shader-f16',
        'subgroups',
        'subgroup-size-control',
        'timestamp-query',
        'texture-compression-bc',
      ],
      limits: {
        maxStorageBufferBindingSize: 2_147_483_648,
        maxBufferSize: 4_294_967_296,
      },
      requestDevice: requestDeviceSpy,
    });

    const mockGpu = {
      requestAdapter: vi.fn(async () => adapter),
    } as unknown as GPU;

    const acquired = await acquireGpuDeviceForPreference('high-performance', {
      gpu: mockGpu,
      expectedDeviceFingerprint: 'nvidia|ampere|0x2204|rtx 3090|hw',
    });

    expect(mockGpu.requestAdapter).toHaveBeenCalledWith({
      powerPreference: 'high-performance',
    });
    expect(requestDeviceSpy).toHaveBeenCalledTimes(1);
    const descriptor = requestDeviceSpy.mock.calls[0]?.[0];
    expect(descriptor?.requiredFeatures).toEqual([
      'shader-f16',
      'subgroups',
      'subgroup-size-control',
      'timestamp-query',
    ]);
    expect(descriptor?.requiredLimits?.maxStorageBufferBindingSize).toBe(2_147_483_648);
    expect(descriptor?.requiredLimits?.maxBufferSize).toBe(4_294_967_296);

    expect(acquired.appliedCustomLimits).toBe(true);
    expect(acquired.supportsFp16).toBe(true);
    expect(acquired.fingerprintMatched).toBe(true);
  });

  it('safely falls back when driver rejects custom requiredLimits or experimental features', async () => {
    const requestDeviceSpy = vi
      .fn<NonNullable<MockAdapterSpec['requestDevice']>>()
      .mockRejectedValueOnce(new Error('Limits exceed driver allowance'))
      .mockRejectedValueOnce(new Error('Unsupported experimental feature'))
      .mockResolvedValueOnce(createMockDevice(['shader-f16']));

    const adapter = createMockAdapter({
      info: { vendor: 'intel', description: 'Intel Arc' },
      features: ['shader-f16', 'subgroups', 'subgroup-size-control'],
      requestDevice: requestDeviceSpy,
    });

    const mockGpu = {
      requestAdapter: vi.fn(async () => adapter),
    } as unknown as GPU;

    const acquired = await acquireGpuDeviceForPreference('low-power', {
      gpu: mockGpu,
      expectedDeviceFingerprint: 'different|fingerprint|0||hw',
    });

    expect(requestDeviceSpy).toHaveBeenCalledTimes(3);
    expect(acquired.appliedCustomLimits).toBe(false);
    expect(acquired.supportsFp16).toBe(true);
    expect(acquired.grantedFeatures).toEqual(['shader-f16']);
    expect(acquired.fingerprintMatched).toBe(false);
  });

  it('works via global navigator.gpu in a Web Worker-like environment without window', async () => {
    const adapter = createMockAdapter({
      info: { vendor: 'qualcomm', description: 'Adreno' },
      features: [],
    });
    vi.stubGlobal('navigator', {
      hardwareConcurrency: 8,
      gpu: {
        requestAdapter: vi.fn(async () => adapter),
      },
    });

    try {
      const devices = await probeComputeDevices();
      expect(devices[0]?.backend).toBe('webgpu');

      const acquired = await acquireGpuDeviceForPreference('default');
      expect(acquired.supportsFp16).toBe(false);
      expect(acquired.adapterInfo.vendor).toBe('qualcomm');
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('throws clear error when WebGPU or matching adapter is unavailable', async () => {
    await expect(
      acquireGpuDeviceForPreference('high-performance', { gpu: null })
    ).rejects.toThrow(/WebGPU/);

    const emptyGpu = {
      requestAdapter: vi.fn(async () => null),
    } as unknown as GPU;
    await expect(
      acquireGpuDeviceForPreference('fallback', { gpu: emptyGpu })
    ).rejects.toThrow(/fallback/);
  });

  it('preserves essential ONNX buffer limits and shader-f16 on tier-2 fallback when experimental features fail', async () => {
    const requestDeviceSpy = vi
      .fn<NonNullable<MockAdapterSpec['requestDevice']>>()
      .mockRejectedValueOnce(new Error('subgroups feature unsupported'))
      .mockResolvedValueOnce(createMockDevice(['shader-f16']));

    const adapter = createMockAdapter({
      info: { vendor: 'qualcomm', architecture: 'adreno', device: '750', description: 'Adreno 750' },
      features: ['shader-f16', 'subgroups', 'subgroup-size-control', 'timestamp-query'],
      limits: {
        maxStorageBufferBindingSize: 1_073_741_824,
        maxBufferSize: 2_147_483_648,
        maxComputeWorkgroupStorageSize: 32_768,
        maxStorageBuffersPerShaderStage: 8,
        maxComputeInvocationsPerWorkgroup: 1024,
        maxUniformBufferBindingSize: 65_536,
      },
      requestDevice: requestDeviceSpy,
    });

    const mockGpu = {
      requestAdapter: vi.fn(async () => adapter),
    } as unknown as GPU;

    const acquired = await acquireGpuDeviceForPreference('high-performance', { gpu: mockGpu });

    expect(requestDeviceSpy).toHaveBeenCalledTimes(2);
    const tier2Descriptor = requestDeviceSpy.mock.calls[1]?.[0];
    expect(tier2Descriptor?.requiredFeatures).toEqual(['shader-f16']);
    expect(tier2Descriptor?.requiredLimits).toEqual({
      maxStorageBufferBindingSize: 1_073_741_824,
      maxBufferSize: 2_147_483_648,
      maxComputeWorkgroupStorageSize: 32_768,
      maxStorageBuffersPerShaderStage: 8,
      maxComputeInvocationsPerWorkgroup: 1024,
    });
    expect(acquired.appliedCustomLimits).toBe(true);
    expect(acquired.supportsFp16).toBe(true);
    expect(acquired.grantedFeatures).toEqual(['shader-f16']);
  });

  it('falls back all the way to bare requestDevice() when even shader-f16 fails', async () => {
    const requestDeviceSpy = vi
      .fn<NonNullable<MockAdapterSpec['requestDevice']>>()
      .mockRejectedValueOnce(new Error('tier 1 failed'))
      .mockRejectedValueOnce(new Error('tier 2 failed'))
      .mockRejectedValueOnce(new Error('shader-f16 rejected'))
      .mockResolvedValueOnce(createMockDevice([]));

    const adapter = createMockAdapter({
      info: { vendor: 'arm', description: 'Mali' },
      features: ['shader-f16', 'subgroups'],
      limits: {
        maxStorageBufferBindingSize: 536_870_912,
        maxBufferSize: 1_073_741_824,
      },
      requestDevice: requestDeviceSpy,
    });

    const acquired = await acquireGpuDeviceForPreference('default', {
      gpu: { requestAdapter: vi.fn(async () => adapter) } as unknown as GPU,
    });

    expect(requestDeviceSpy).toHaveBeenCalledTimes(4);
    expect(requestDeviceSpy.mock.calls[3]?.[0]).toBeUndefined();
    expect(acquired.supportsFp16).toBe(false);
    expect(acquired.appliedCustomLimits).toBe(false);
    expect(acquired.grantedFeatures).toEqual([]);
  });
});
