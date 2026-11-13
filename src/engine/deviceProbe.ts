import type {
  AdapterInfoSnapshot,
  DeviceOption,
  DeviceRuntimeDetails,
  GpuPreference,
  WorkerDeviceTarget,
} from '../types.ts';

/**
 * Minimal valid WebAssembly v128 SIMD module used to feature-detect fixed-width SIMD
 * synchronously in both Window and Web Worker contexts.
 */
const WASM_SIMD_PROBE_BYTES = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, // magic
  0x01, 0x00, 0x00, 0x00, // version
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, // type: () -> v128
  0x03, 0x02, 0x01, 0x00, // function section
  0x0a, 0x0a, 0x01, 0x08, 0x00, 0x41, 0x00, 0xfd, 0x0f, 0xfd, 0x62, 0x0b, // v128.const + i8x16.bitmask
]);

/**
 * WebGPU optional features leveraged by ONNX Runtime Web (`ort.webgpu`).
 */
export const ORT_WEBGPU_FEATURES = [
  'shader-f16',
  'subgroups',
  'subgroup-size-control',
  'timestamp-query',
] as const;

/**
 * Compute limits forwarded when requesting a GPUDevice so ORT does not get
 * restricted to WebGPU's 128 MiB default `maxStorageBufferBindingSize`.
 */
const ORT_REQUESTED_LIMIT_KEYS = [
  'maxBindGroups',
  'maxComputeWorkgroupStorageSize',
  'maxComputeWorkgroupsPerDimension',
  'maxStorageBuffersPerShaderStage',
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
] as const;

/**
 * Complete list of GPUSupportedLimits numeric properties (prototype getters in browsers).
 */
const ALL_ADAPTER_LIMIT_KEYS = [
  'maxTextureDimension1D',
  'maxTextureDimension2D',
  'maxTextureDimension3D',
  'maxTextureArrayLayers',
  'maxBindGroups',
  'maxBindGroupsPlusVertexBuffers',
  'maxBindingsPerBindGroup',
  'maxDynamicUniformBuffersPerPipelineLayout',
  'maxDynamicStorageBuffersPerPipelineLayout',
  'maxSampledTexturesPerShaderStage',
  'maxSamplersPerShaderStage',
  'maxStorageBuffersPerShaderStage',
  'maxStorageTexturesPerShaderStage',
  'maxUniformBuffersPerShaderStage',
  'maxUniformBufferBindingSize',
  'maxStorageBufferBindingSize',
  'minUniformBufferOffsetAlignment',
  'minStorageBufferOffsetAlignment',
  'maxVertexBuffers',
  'maxBufferSize',
  'maxVertexAttributes',
  'maxVertexBufferArrayStride',
  'maxInterStageShaderVariables',
  'maxColorAttachments',
  'maxColorAttachmentBytesPerSample',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
] as const;

const VENDOR_DISPLAY_NAMES: Record<string, string> = {
  nvidia: 'NVIDIA',
  amd: 'AMD',
  intel: 'Intel',
  apple: 'Apple',
  qualcomm: 'Qualcomm',
  arm: 'ARM',
  google: 'Google',
  microsoft: 'Microsoft',
  mesa: 'Mesa',
};

const PROBE_ORDER: readonly GpuPreference[] = [
  'high-performance',
  'low-power',
  'default',
  'fallback',
];

export interface ProbeEnvironmentOverrides {
  gpu?: GPU | null;
  crossOriginIsolated?: boolean;
  hardwareConcurrency?: number;
  simdSupported?: boolean;
  sharedArrayBufferAvailable?: boolean;
}

export interface AcquireGpuDeviceOptions {
  gpu?: GPU | null;
  expectedDeviceFingerprint?: string;
}

export interface AcquiredGpuDevice {
  adapter: GPUAdapter;
  device: GPUDevice;
  gpuPreference: GpuPreference;
  adapterInfo: AdapterInfoSnapshot;
  fingerprint: string;
  fingerprintMatched: boolean;
  features: string[];
  grantedFeatures: string[];
  limits: Record<string, number>;
  supportsFp16: boolean;
  appliedCustomLimits: boolean;
}

/**
 * Feature-detect WebAssembly fixed-width 128-bit SIMD support.
 */
export function detectWasmSimdSupport(): boolean {
  try {
    return typeof WebAssembly?.validate === 'function' && WebAssembly.validate(WASM_SIMD_PROBE_BYTES);
  } catch {
    return false;
  }
}

/**
 * Collect runtime capabilities relevant to ONNX Runtime WASM execution.
 * Works identically in Window and Dedicated Web Worker scopes.
 */
export function getDeviceRuntimeDetails(
  overrides?: Pick<
    ProbeEnvironmentOverrides,
    'crossOriginIsolated' | 'hardwareConcurrency' | 'simdSupported' | 'sharedArrayBufferAvailable'
  >
): DeviceRuntimeDetails {
  const globalScope = globalThis as {
    crossOriginIsolated?: boolean;
    navigator?: { hardwareConcurrency?: number };
  };

  const crossOriginIsolated =
    overrides?.crossOriginIsolated ?? Boolean(globalScope.crossOriginIsolated);

  const rawCores = overrides?.hardwareConcurrency ?? globalScope.navigator?.hardwareConcurrency;
  const hardwareConcurrency =
    typeof rawCores === 'number' && Number.isFinite(rawCores) && rawCores >= 1
      ? Math.floor(rawCores)
      : 1;

  const simdSupported = overrides?.simdSupported ?? detectWasmSimdSupport();
  const hasSab = overrides?.sharedArrayBufferAvailable ?? typeof SharedArrayBuffer !== 'undefined';

  // ORT Web enforces numThreads = 1 when !crossOriginIsolated (e.g., GitHub Pages without COOP/COEP).
  const numThreads =
    crossOriginIsolated && hasSab
      ? Math.min(4, Math.max(1, Math.ceil(hardwareConcurrency / 2)))
      : 1;

  return {
    crossOriginIsolated,
    hardwareConcurrency,
    simdSupported,
    numThreads,
  };
}

/**
 * Map internal preference enum to WebGPU `navigator.gpu.requestAdapter` options.
 */
export function toRequestAdapterOptions(gpuPreference: GpuPreference): GPURequestAdapterOptions {
  switch (gpuPreference) {
    case 'high-performance':
      return { powerPreference: 'high-performance' };
    case 'low-power':
      return { powerPreference: 'low-power' };
    case 'fallback':
      return { forceFallbackAdapter: true };
    case 'default':
    default:
      return {};
  }
}

function readGpuFromEnvironment(override?: GPU | null): GPU | undefined {
  if (override === null) {
    return undefined;
  }
  if (override !== undefined) {
    return override;
  }
  const nav = (globalThis as { navigator?: { gpu?: GPU } }).navigator;
  return nav?.gpu;
}

function extractAdapterInfoSnapshot(adapter: GPUAdapter): AdapterInfoSnapshot {
  const rawInfo = (adapter as { info?: Partial<GPUAdapterInfo> }).info;
  const vendor = typeof rawInfo?.vendor === 'string' ? rawInfo.vendor.trim() : '';
  const architecture =
    typeof rawInfo?.architecture === 'string' ? rawInfo.architecture.trim() : '';
  const device = typeof rawInfo?.device === 'string' ? rawInfo.device.trim() : '';
  const description = typeof rawInfo?.description === 'string' ? rawInfo.description.trim() : '';
  const isFallbackAdapter = Boolean(rawInfo?.isFallbackAdapter);
  const subgroupMinSize =
    typeof rawInfo?.subgroupMinSize === 'number' && Number.isFinite(rawInfo.subgroupMinSize)
      ? rawInfo.subgroupMinSize
      : undefined;
  const subgroupMaxSize =
    typeof rawInfo?.subgroupMaxSize === 'number' && Number.isFinite(rawInfo.subgroupMaxSize)
      ? rawInfo.subgroupMaxSize
      : undefined;

  return {
    vendor,
    architecture,
    device,
    description,
    isFallbackAdapter,
    ...(subgroupMinSize !== undefined ? { subgroupMinSize } : {}),
    ...(subgroupMaxSize !== undefined ? { subgroupMaxSize } : {}),
  };
}

function extractFeatureNames(features: GPUSupportedFeatures | Iterable<string> | undefined): string[] {
  if (!features) {
    return [];
  }
  try {
    return Array.from(features as Iterable<string>).sort();
  } catch {
    return [];
  }
}

function extractLimitsSnapshot(limits: GPUSupportedLimits | Record<string, unknown> | undefined): Record<string, number> {
  const out: Record<string, number> = {};
  if (!limits) {
    return out;
  }
  const record = limits as unknown as Record<string, unknown>;
  for (const key of ALL_ADAPTER_LIMIT_KEYS) {
    const val = record[key];
    if (typeof val === 'number' && Number.isFinite(val)) {
      out[key] = val;
    }
  }
  return out;
}

/**
 * Build a stable adapter identity fingerprint for matching a selected GPU across
 * Window -> Web Worker boundaries.
 */
export function buildAdapterFingerprint(info: AdapterInfoSnapshot): string {
  return [
    info.vendor.toLowerCase(),
    info.architecture.toLowerCase(),
    info.device.toLowerCase(),
    info.description.toLowerCase(),
    info.isFallbackAdapter ? 'fallback' : 'hw',
  ].join('|');
}

function buildDedupeKey(
  info: AdapterInfoSnapshot,
  features: readonly string[],
  limits: Readonly<Record<string, number>>
): string {
  const fingerprint = buildAdapterFingerprint(info);
  const hasAnyString =
    info.vendor.length > 0 ||
    info.architecture.length > 0 ||
    info.device.length > 0 ||
    info.description.length > 0;
  const keyLimits = `${limits.maxBufferSize ?? ''}|${limits.maxStorageBufferBindingSize ?? ''}|${limits.maxComputeWorkgroupStorageSize ?? ''}`;
  if (hasAnyString) {
    return `${fingerprint}|${keyLimits}`;
  }
  return `${fingerprint}|${keyLimits}|${features.join(',')}`;
}

function formatVendorName(vendor: string): string {
  if (!vendor) return '';
  const mapped = VENDOR_DISPLAY_NAMES[vendor.toLowerCase()];
  if (mapped) return mapped;
  return vendor.charAt(0).toUpperCase() + vendor.slice(1);
}

function formatGpuDisplayName(info: AdapterInfoSnapshot): string {
  if (info.description.length > 0) {
    return info.description;
  }
  const vendor = formatVendorName(info.vendor);
  const parts = [vendor, info.architecture, info.device].filter((part) => part.length > 0);
  if (parts.length > 0) {
    return parts.join(' ');
  }
  return info.isFallbackAdapter ? 'Software Adapter' : 'WebGPU Adapter';
}

function formatWebGpuLabel(
  gpuPreference: GpuPreference,
  info: AdapterInfoSnapshot
): string {
  const displayName = formatGpuDisplayName(info);
  if (info.isFallbackAdapter || gpuPreference === 'fallback') {
    return `WebGPU Fallback (ソフトウェア): ${displayName}`;
  }
  if (gpuPreference === 'high-performance') {
    return `WebGPU (High Performance): ${displayName}`;
  }
  if (gpuPreference === 'low-power') {
    return `WebGPU (Low Power): ${displayName}`;
  }
  return `WebGPU: ${displayName}`;
}

function createWasmCpuOption(runtimeDetails: DeviceRuntimeDetails): DeviceOption {
  const features: string[] = [];
  if (runtimeDetails.simdSupported) {
    features.push('wasm-simd');
  }
  if (runtimeDetails.numThreads > 1) {
    features.push('threads');
  }

  const simdLabel = runtimeDetails.simdSupported ? ' SIMD' : '';
  const coiSuffix = runtimeDetails.crossOriginIsolated ? '' : ' / 非COI';
  return {
    id: 'wasm-cpu',
    label: `CPU (WASM${simdLabel}, ${runtimeDetails.numThreads}スレッド${coiSuffix})`,
    backend: 'wasm',
    features,
    limits: {},
    supportsFp16: false,
    recommendedDtype: 'q4',
    available: true,
    runtimeDetails,
  };
}

/**
 * Probe all selectable WebGPU compute adapters (`high-performance`, `low-power`,
 * `default`, `fallback`) with deduplication, plus the always-available CPU (`wasm`)
 * backend. Safe to invoke in both Window and Web Worker scopes.
 */
export async function probeComputeDevices(
  overrides?: ProbeEnvironmentOverrides
): Promise<DeviceOption[]> {
  const runtimeDetails = getDeviceRuntimeDetails(overrides);
  const gpu = readGpuFromEnvironment(overrides?.gpu);
  const devices: DeviceOption[] = [];

  if (!gpu || typeof gpu.requestAdapter !== 'function') {
    devices.push({
      id: 'webgpu-unavailable',
      label: 'WebGPU (利用不可)',
      backend: 'webgpu',
      features: [],
      limits: {},
      supportsFp16: false,
      recommendedDtype: 'q4',
      available: false,
      unavailableReason:
        'このブラウザまたは実行コンテキストでは WebGPU (navigator.gpu) が利用できません。',
      runtimeDetails,
    });
    devices.push(createWasmCpuOption(runtimeDetails));
    return devices;
  }

  const seenKeys = new Set<string>();
  const hardwareOptions: DeviceOption[] = [];
  const fallbackOptions: DeviceOption[] = [];
  let probeFailureCount = 0;

  for (const gpuPreference of PROBE_ORDER) {
    let adapter: GPUAdapter | null = null;
    try {
      adapter = await gpu.requestAdapter(toRequestAdapterOptions(gpuPreference));
    } catch {
      probeFailureCount += 1;
      continue;
    }

    if (!adapter) {
      continue;
    }

    const adapterInfo = extractAdapterInfoSnapshot(adapter);
    const features = extractFeatureNames(adapter.features);
    const limits = extractLimitsSnapshot(adapter.limits);
    const dedupeKey = buildDedupeKey(adapterInfo, features, limits);

    if (seenKeys.has(dedupeKey)) {
      continue;
    }
    seenKeys.add(dedupeKey);

    const supportsFp16 = features.includes('shader-f16');
    const isFallback = adapterInfo.isFallbackAdapter || gpuPreference === 'fallback';
    const option: DeviceOption = {
      id: `webgpu-${gpuPreference}`,
      label: formatWebGpuLabel(gpuPreference, adapterInfo),
      backend: 'webgpu',
      gpuPreference,
      fingerprint: buildAdapterFingerprint(adapterInfo),
      adapterInfo,
      features,
      limits,
      supportsFp16,
      recommendedDtype: supportsFp16 ? 'q4f16' : 'q4',
      available: true,
      runtimeDetails,
    };

    if (isFallback) {
      fallbackOptions.push(option);
    } else {
      hardwareOptions.push(option);
    }
  }

  devices.push(...hardwareOptions, ...fallbackOptions);

  if (devices.length === 0) {
    devices.push({
      id: 'webgpu-unavailable',
      label: 'WebGPU (利用不可)',
      backend: 'webgpu',
      features: [],
      limits: {},
      supportsFp16: false,
      recommendedDtype: 'q4',
      available: false,
      unavailableReason:
        probeFailureCount === PROBE_ORDER.length
          ? 'WebGPU アダプターの取得中にエラーが発生しました。'
          : '利用可能な WebGPU アダプターが検出されませんでした。',
      runtimeDetails,
    });
  }

  devices.push(createWasmCpuOption(runtimeDetails));
  return devices;
}

/**
 * Pick the highest-priority available compute device:
 * hardware WebGPU -> fallback WebGPU -> CPU (WASM).
 */
export function pickDefaultDevice(devices: readonly DeviceOption[]): DeviceOption | undefined {
  const availableHardwareGpu = devices.find(
    (d) =>
      d.available &&
      d.backend === 'webgpu' &&
      !d.adapterInfo?.isFallbackAdapter &&
      d.gpuPreference !== 'fallback'
  );
  if (availableHardwareGpu) {
    return availableHardwareGpu;
  }

  const availableAnyGpu = devices.find((d) => d.available && d.backend === 'webgpu');
  if (availableAnyGpu) {
    return availableAnyGpu;
  }

  return devices.find((d) => d.available && d.backend === 'wasm') ?? devices.find((d) => d.available);
}

/**
 * Convert a probed `DeviceOption` into a serializable `WorkerDeviceTarget`
 * for `WorkerRequest` (`load-model`).
 */
export function toWorkerDeviceTarget(device: DeviceOption): WorkerDeviceTarget {
  if (device.backend === 'webgpu') {
    return {
      backend: 'webgpu',
      gpuPreference: device.gpuPreference ?? 'default',
      ...(device.fingerprint ? { expectedDeviceFingerprint: device.fingerprint } : {}),
    };
  }
  return {
    backend: 'wasm',
    numThreads: device.runtimeDetails.numThreads,
    simdSupported: device.runtimeDetails.simdSupported,
  };
}

function buildClampedRequiredLimits(
  limits: Readonly<Record<string, number>>
): Record<string, number> {
  const requiredLimits: Record<string, number> = {};
  for (const key of ORT_REQUESTED_LIMIT_KEYS) {
    const val = limits[key];
    if (typeof val === 'number' && Number.isFinite(val) && val > 0) {
      requiredLimits[key] = val;
    }
  }
  return requiredLimits;
}

/**
 * Acquire a `GPUAdapter` and `GPUDevice` matching `gpuPreference` directly in the
 * current execution context (Window or Web Worker), requesting ORT-supported features
 * and adapter limits with safe fallbacks if the driver rejects custom descriptors.
 */
export async function acquireGpuDeviceForPreference(
  gpuPreference: GpuPreference,
  options?: AcquireGpuDeviceOptions
): Promise<AcquiredGpuDevice> {
  const gpu = readGpuFromEnvironment(options?.gpu);
  if (!gpu || typeof gpu.requestAdapter !== 'function') {
    throw new Error('この実行コンテキストでは WebGPU (navigator.gpu) が利用できません。');
  }

  const adapter = await gpu.requestAdapter(toRequestAdapterOptions(gpuPreference));
  if (!adapter) {
    throw new Error(
      `指定された WebGPU アダプター (${gpuPreference}) を取得できませんでした。`
    );
  }

  const adapterInfo = extractAdapterInfoSnapshot(adapter);
  const features = extractFeatureNames(adapter.features);
  const limits = extractLimitsSnapshot(adapter.limits);
  const fingerprint = buildAdapterFingerprint(adapterInfo);
  const fingerprintMatched =
    !options?.expectedDeviceFingerprint || options.expectedDeviceFingerprint === fingerprint;

  const featureSet = new Set(features);
  const requiredFeatures = ORT_WEBGPU_FEATURES.filter((feature) =>
    featureSet.has(feature)
  ) as unknown as GPUFeatureName[];
  const requiredLimits = buildClampedRequiredLimits(limits);
  const hasCustomLimits = Object.keys(requiredLimits).length > 0;

  let device: GPUDevice;
  let appliedCustomLimits = false;
  let requestedFeaturesForDevice: string[] = [...requiredFeatures];

  try {
    device = await adapter.requestDevice({
      ...(requiredFeatures.length > 0 ? { requiredFeatures } : {}),
      ...(hasCustomLimits ? { requiredLimits } : {}),
    });
    appliedCustomLimits = hasCustomLimits;
  } catch {
    try {
      device = await adapter.requestDevice(
        requiredFeatures.length > 0 ? { requiredFeatures } : undefined
      );
    } catch {
      const minimalFeatures = (featureSet.has('shader-f16') ? ['shader-f16'] : []) as GPUFeatureName[];
      try {
        requestedFeaturesForDevice = [...minimalFeatures];
        device = await adapter.requestDevice(
          minimalFeatures.length > 0 ? { requiredFeatures: minimalFeatures } : undefined
        );
      } catch {
        requestedFeaturesForDevice = [];
        device = await adapter.requestDevice();
      }
    }
  }

  const grantedFeatures = device.features
    ? extractFeatureNames(device.features)
    : requestedFeaturesForDevice;
  const supportsFp16 = grantedFeatures.includes('shader-f16');

  return {
    adapter,
    device,
    gpuPreference,
    adapterInfo,
    fingerprint,
    fingerprintMatched,
    features,
    grantedFeatures,
    limits,
    supportsFp16,
    appliedCustomLimits,
  };
}
