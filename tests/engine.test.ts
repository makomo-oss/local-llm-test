import { afterEach, describe, expect, it, vi } from 'vitest';
import { ModelRegistry } from '@huggingface/transformers';
import {
  TRANSFORMERS_CACHE_NAME,
  clearModelCache,
  extractModelFileFromCacheUrl,
  getBrowserStorageEstimate,
  inspectModelCaches,
  measureCachedResponseBytes,
} from '../src/engine/cacheManager.ts';
import {
  classifyModelLoadError,
  computeSpeedMetrics,
  createDownloadProgressAggregator,
  stripTrailingSpecialDelimiters,
} from '../src/engine/llm.worker.ts';
import {
  buildPrecachedResponseHeaders,
  buildRemoteModelFileUrl,
  createSerializedCacheAdapter,
  filterCausalLmModelFiles,
  isOptionalModelFile,
  precacheModelFilesSequentially,
  probeStreamingCachePutSupport,
  resolveExpectedContentLength,
  resolveExternalDataAugmentedConfig,
} from '../src/engine/modelPrecacher.ts';
import {
  LlmRuntimeClient,
  LlmRuntimeError,
  type WorkerLike,
} from '../src/engine/runtimeClient.ts';
import {
  getModelById,
  getModelDownloadBytes,
  getResolvedDownloadBytes,
  resolveEffectiveDtype,
} from '../src/models/catalog.ts';
import type {
  DeviceOption,
  GenerationConfig,
  ModelLoadProgress,
  WorkerRequest,
  WorkerResponse,
} from '../src/types.ts';

interface MockCacheEntrySpec {
  contentLength?: string;
  xLinkedSize?: string;
  contentEncoding?: string;
  blobBytes?: number;
  throwOnBlob?: boolean;
  hasClone?: boolean;
}

function createMockCacheStorage(
  initialBuckets: Record<string, Record<string, MockCacheEntrySpec>> = {}
) {
  const buckets = new Map<string, Map<string, MockCacheEntrySpec>>();
  for (const [bucketName, entries] of Object.entries(initialBuckets)) {
    buckets.set(bucketName, new Map(Object.entries(entries)));
  }

  const blobSpies = new Map<string, ReturnType<typeof vi.fn>>();

  const makeCacheObj = (bucket: Map<string, MockCacheEntrySpec>): Cache =>
    ({
      keys: vi.fn(async () =>
        Array.from(bucket.keys()).map((url) => ({ url }) as unknown as Request)
      ),
      match: vi.fn(async (req: RequestInfo | URL) => {
        const url = typeof req === 'string' ? req : req instanceof URL ? req.href : req.url;
        const spec = bucket.get(url);
        if (!spec) {
          return undefined;
        }
        const blobSpy = vi.fn(async () => {
          if (spec.throwOnBlob) {
            throw new Error('Failed to inspect cached blob');
          }
          return new Blob([new Uint8Array(spec.blobBytes ?? 0)]);
        });
        blobSpies.set(url, blobSpy);

        const headerEntries: Record<string, string> = {};
        if (spec.contentLength !== undefined) {
          headerEntries['content-length'] = spec.contentLength;
        }
        if (spec.xLinkedSize !== undefined) {
          headerEntries['x-linked-size'] = spec.xLinkedSize;
        }
        if (spec.contentEncoding !== undefined) {
          headerEntries['content-encoding'] = spec.contentEncoding;
        }

        const responseMock: Partial<Response> = {
          headers: new Headers(headerEntries),
          blob: blobSpy,
        };
        if (spec.hasClone !== false) {
          responseMock.clone = () => responseMock as Response;
        }

        return responseMock as Response;
      }),
      delete: vi.fn(async (req: RequestInfo | URL) => {
        const url = typeof req === 'string' ? req : req instanceof URL ? req.href : req.url;
        return bucket.delete(url);
      }),
    }) as unknown as Cache;

  const openSpy = vi.fn(async (name: string) => {
    let bucket = buckets.get(name);
    if (!bucket) {
      bucket = new Map();
      buckets.set(name, bucket);
    }
    return makeCacheObj(bucket);
  });

  const hasSpy = vi.fn(async (name: string) => buckets.has(name));
  const deleteBucketSpy = vi.fn(async (name: string) => buckets.delete(name));

  const storage = {
    open: openSpy,
    has: hasSpy,
    delete: deleteBucketSpy,
  };

  return { storage, buckets, blobSpies, openSpy, hasSpy, deleteBucketSpy };
}

class MockWorker implements WorkerLike {
  readonly postedMessages: WorkerRequest[] = [];
  terminated = false;
  private readonly messageListeners = new Set<(event: MessageEvent<WorkerResponse>) => void>();
  private readonly errorListeners = new Set<(event: ErrorEvent) => void>();
  private readonly messageErrorListeners = new Set<(event: MessageEvent) => void>();

  postMessage(message: unknown): void {
    this.postedMessages.push(message as WorkerRequest);
  }

  terminate(): void {
    this.terminated = true;
  }

  addEventListener(type: string, listener: (...args: any[]) => void): void {
    if (type === 'message') {
      this.messageListeners.add(listener);
    } else if (type === 'error') {
      this.errorListeners.add(listener);
    } else if (type === 'messageerror') {
      this.messageErrorListeners.add(listener);
    }
  }

  removeEventListener(type: string, listener: (...args: any[]) => void): void {
    if (type === 'message') {
      this.messageListeners.delete(listener);
    } else if (type === 'error') {
      this.errorListeners.delete(listener);
    } else if (type === 'messageerror') {
      this.messageErrorListeners.delete(listener);
    }
  }

  emitMessage(data: WorkerResponse): void {
    const event = { data } as MessageEvent<WorkerResponse>;
    for (const listener of [...this.messageListeners]) {
      listener(event);
    }
  }

  emitError(message: string): void {
    const event = { message } as ErrorEvent;
    for (const listener of [...this.errorListeners]) {
      listener(event);
    }
  }

  emitMessageError(): void {
    const event = {} as MessageEvent;
    for (const listener of [...this.messageErrorListeners]) {
      listener(event);
    }
  }

  listenerCount(): number {
    return (
      this.messageListeners.size +
      this.errorListeners.size +
      this.messageErrorListeners.size
    );
  }
}

const SAMPLE_DEVICE: DeviceOption = {
  id: 'webgpu-high-performance',
  label: 'WebGPU (High Performance): Mock GPU',
  backend: 'webgpu',
  gpuPreference: 'high-performance',
  fingerprint: 'nvidia|ada|0x1234|mock gpu|hw',
  features: ['shader-f16', 'subgroups'],
  limits: { maxBufferSize: 2_147_483_648 },
  supportsFp16: true,
  recommendedDtype: 'q4f16',
  available: true,
  runtimeDetails: {
    crossOriginIsolated: false,
    hardwareConcurrency: 8,
    simdSupported: true,
    numThreads: 1,
  },
};

const SAMPLE_GEN_CONFIG: GenerationConfig = {
  maxNewTokens: 256,
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  doSample: true,
  enableThinking: true,
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('cacheManager', () => {
  it('returns empty uncached summaries safely when CacheStorage is unavailable or cache is absent', async () => {
    vi.stubGlobal('caches', undefined);
    const unavailable = await inspectModelCaches();
    expect(unavailable).toHaveLength(6);
    expect(unavailable.every((entry) => !entry.cached && entry.cachedBytes === 0)).toBe(true);

    const mock = createMockCacheStorage({});
    vi.stubGlobal('caches', mock.storage);
    const absent = await inspectModelCaches();
    expect(mock.hasSpy).toHaveBeenCalledWith(TRANSFORMERS_CACHE_NAME);
    expect(mock.openSpy).not.toHaveBeenCalled();
    expect(absent.every((entry) => !entry.cached && entry.fileCount === 0)).toBe(true);
  });

  it('extracts repo files without prefix collisions and measures sizes via content-length first', async () => {
    expect(
      extractModelFileFromCacheUrl(
        'https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX-OPT/resolve/main/tokenizer.json',
        'onnx-community/Qwen3.5-4B-ONNX'
      )
    ).toBeNull();

    expect(
      extractModelFileFromCacheUrl(
        'https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data',
        'onnx-community/Qwen3.5-4B-ONNX'
      )
    ).toBe('onnx/decoder_model_merged_q4f16.onnx_data');

    const qwenWeightUrl =
      'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data';
    const qwenLinkedWeightUrl =
      'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/main/onnx/decoder_model_merged_q4.onnx_data';
    const qwenMissingHeaderWeightUrl =
      'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/main/onnx/embed_tokens_q4f16.onnx';
    const qwenMissingHeaderDataUrl =
      'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/main/onnx/embed_tokens_fp16.onnx_data';
    const qwenConfigUrl =
      'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/main/config.json';
    const gemmaTokenizerUrl =
      'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/tokenizer.json';
    const wasmRuntimeUrl =
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0/dist/ort-wasm-simd-threaded.asyncify.wasm';

    const mock = createMockCacheStorage({
      [TRANSFORMERS_CACHE_NAME]: {
        [qwenWeightUrl]: { contentLength: '584000000', throwOnBlob: true },
        [qwenLinkedWeightUrl]: { xLinkedSize: '285000000', throwOnBlob: true },
        [qwenMissingHeaderWeightUrl]: { blobBytes: 440_700 },
        [qwenMissingHeaderDataUrl]: { throwOnBlob: true, hasClone: false },
        [qwenConfigUrl]: { blobBytes: 1200 },
        [gemmaTokenizerUrl]: { contentLength: '19439251', throwOnBlob: true },
        [wasmRuntimeUrl]: { contentLength: '26861770', throwOnBlob: true },
      },
    });
    vi.stubGlobal('caches', mock.storage);

    const results = await inspectModelCaches([
      getModelById('qwen-3.5-0.8b'),
      getModelById('gemma-4-e2b'),
      getModelById('llama-3.2-3b'),
    ]);

    expect(mock.blobSpies.get(qwenWeightUrl)).not.toHaveBeenCalled();
    expect(mock.blobSpies.get(qwenLinkedWeightUrl)).not.toHaveBeenCalled();
    expect(mock.blobSpies.get(qwenMissingHeaderWeightUrl)).toHaveBeenCalledTimes(1);
    expect(mock.blobSpies.get(qwenMissingHeaderDataUrl)).toHaveBeenCalledTimes(1);
    expect(mock.blobSpies.get(qwenConfigUrl)).toHaveBeenCalledTimes(1);

    expect(results[0]).toMatchObject({
      modelId: 'qwen-3.5-0.8b',
      cached: true,
      cachedBytes: 584_000_000 + 285_000_000 + 440_700 + 1_200,
      fileCount: 5,
    });
    expect(results[1]).toMatchObject({
      modelId: 'gemma-4-e2b',
      cached: true,
      cachedBytes: 19_439_251,
      fileCount: 1,
    });
    expect(results[2]).toMatchObject({
      modelId: 'llama-3.2-3b',
      cached: false,
      cachedBytes: 0,
      fileCount: 0,
    });

    // Compressed assets (e.g. CDN .wasm with content-encoding: br) must measure uncompressed blob size
    const compressedWasmResponse = {
      headers: new Headers({
        'content-encoding': 'br',
        'content-length': '4704500',
      }),
      clone() {
        return this;
      },
      blob: async () => ({ size: 26_861_777 }) as Blob,
    } as unknown as Response;
    expect(
      await measureCachedResponseBytes(compressedWasmResponse, 'ort-wasm-simd-threaded.jsep.wasm')
    ).toBe(26_861_777);
  });

  it('clears a single model cache or deletes the entire transformers-cache bucket', async () => {
    const qwenUrl =
      'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/main/onnx/model_q4f16.onnx';
    const llamaUrl =
      'https://huggingface.co/onnx-community/Llama-3.2-3B-Instruct-ONNX/resolve/main/tokenizer.json';
    const wasmUrl =
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0/dist/ort-wasm-simd-threaded.asyncify.wasm';

    const mock = createMockCacheStorage({
      [TRANSFORMERS_CACHE_NAME]: {
        [qwenUrl]: { contentLength: '500000' },
        [llamaUrl]: { contentLength: '11574638' },
        [wasmUrl]: { contentLength: '26861770' },
      },
    });
    vi.stubGlobal('caches', mock.storage);

    const singleResult = await clearModelCache('qwen-3.5-0.8b');
    expect(singleResult).toEqual({
      clearedBytes: 500_000,
      deletedEntries: 1,
    });
    expect(mock.deleteBucketSpy).not.toHaveBeenCalled();
    expect(mock.buckets.get(TRANSFORMERS_CACHE_NAME)?.has(qwenUrl)).toBe(false);
    expect(mock.buckets.get(TRANSFORMERS_CACHE_NAME)?.has(llamaUrl)).toBe(true);

    const fullResult = await clearModelCache();
    expect(fullResult).toEqual({
      clearedBytes: 11_574_638 + 26_861_770,
      deletedEntries: 2,
    });
    expect(mock.deleteBucketSpy).toHaveBeenCalledWith(TRANSFORMERS_CACHE_NAME);
    expect(mock.buckets.has(TRANSFORMERS_CACHE_NAME)).toBe(false);
  });

  it('safely queries navigator.storage.estimate() and handles unsupported/failing environments', async () => {
    vi.stubGlobal('navigator', {
      storage: {
        estimate: vi.fn(async () => ({
          usage: 2_000_000_000,
          quota: 10_000_000_000,
        })),
      },
    });
    expect(await getBrowserStorageEstimate()).toEqual({
      usageBytes: 2_000_000_000,
      quotaBytes: 10_000_000_000,
      availableBytes: 8_000_000_000,
    });

    vi.stubGlobal('navigator', {});
    expect(await getBrowserStorageEstimate()).toEqual({});

    vi.stubGlobal('navigator', {
      storage: {
        estimate: vi.fn(async () => {
          throw new Error('Blocked');
        }),
      },
    });
    expect(await getBrowserStorageEstimate()).toEqual({});
  });
});

describe('llm.worker pure helpers', () => {
  it('computes speed metrics accurately for zero, single, and multi-token runs including P95 ITL', () => {
    expect(computeSpeedMetrics(64, [], 1000, 1250)).toEqual({
      ttftMs: 0,
      prefillMs: 0,
      decodeMs: 0,
      totalMs: 250,
      promptTokens: 64,
      generatedTokens: 0,
      tokensPerSec: 0,
      decodeTokensPerSec: 0,
      interTokenLatencyAvgMs: 0,
      interTokenLatencyP95Ms: 0,
    });

    const singleToken = computeSpeedMetrics(32, [1120], 1000, 1125);
    expect(singleToken).toEqual({
      ttftMs: 120,
      prefillMs: 120,
      decodeMs: 5,
      totalMs: 125,
      promptTokens: 32,
      generatedTokens: 1,
      tokensPerSec: 8,
      decodeTokensPerSec: 0,
      interTokenLatencyAvgMs: 0,
      interTokenLatencyP95Ms: 0,
    });

    // 21 tokens => 20 intervals: nineteen 20ms intervals + one 200ms tail spike
    const timestamps: number[] = [1100];
    for (let i = 1; i <= 19; i += 1) {
      timestamps.push(1100 + i * 20);
    }
    timestamps.push(timestamps[timestamps.length - 1]! + 200);

    const multi = computeSpeedMetrics(100, timestamps, 1000, 1680);
    expect(multi.ttftMs).toBe(100);
    expect(multi.prefillMs).toBe(100);
    expect(multi.decodeMs).toBe(580);
    expect(multi.totalMs).toBe(680);
    expect(multi.prefillMs + multi.decodeMs).toBe(multi.totalMs);
    expect(multi.generatedTokens).toBe(21);
    expect(multi.tokensPerSec).toBeCloseTo(21 / 0.68, 6);
    expect(multi.decodeTokensPerSec).toBeCloseTo(20 / 0.58, 6);
    expect(multi.interTokenLatencyAvgMs).toBeCloseTo((19 * 20 + 200) / 20, 6);
    expect(multi.interTokenLatencyP95Ms).toBe(20);

    // Add one more token with 200ms gap -> 21 intervals => ceil(0.95*21)-1 = 19 (2nd highest = 200ms)
    const skewed = computeSpeedMetrics(100, [...timestamps, timestamps[timestamps.length - 1]! + 200], 1000, 1880);
    expect(skewed.interTokenLatencyP95Ms).toBe(200);
  });

  it('strips trailing special stop/turn delimiters while preserving thinking tags', () => {
    expect(stripTrailingSpecialDelimiters('Hello world<|im_end|>')).toBe('Hello world');
    expect(stripTrailingSpecialDelimiters('Answer<turn|>\n')).toBe('Answer');
    expect(stripTrailingSpecialDelimiters('Llama reply<|eot_id|><|end_of_text|>')).toBe(
      'Llama reply'
    );
    expect(
      stripTrailingSpecialDelimiters('<|channel>thought\nReasoning<channel|>Final answer<turn|><eos>')
    ).toBe('<|channel>thought\nReasoning<channel|>Final answer');
    expect(stripTrailingSpecialDelimiters('<think>\nStep 1\n</think>\nOk<|im_end|>')).toBe(
      '<think>\nStep 1\n</think>\nOk'
    );
    expect(stripTrailingSpecialDelimiters('No delimiters here')).toBe('No delimiters here');
  });

  it('normalizes download progress against text-only bytes and ignores multimodal progress_total', () => {
    const gemma = getModelById('gemma-4-e2b');
    const aggregator = createDownloadProgressAggregator(gemma, 'q4f16');
    const expectedTextBytes = getModelDownloadBytes(gemma, 'q4f16', true);

    expect(
      aggregator.handle({
        status: 'progress_total',
        name: gemma.repoId,
        progress: 50,
        loaded: 100,
        total: gemma.fullMultimodalBytesByDtype!.q4f16!,
        files: {},
      })
    ).toBeNull();

    const init = aggregator.handle({
      status: 'initiate',
      name: gemma.repoId,
      file: 'onnx/decoder_model_merged_q4f16.onnx_data',
    });
    expect(init?.stage).toBe('downloading');
    expect(init?.expectedTotalBytes).toBe(expectedTextBytes);

    const half = aggregator.handle({
      status: 'progress',
      name: gemma.repoId,
      file: 'onnx/decoder_model_merged_q4f16.onnx_data',
      loaded: 1_500_000_000,
      total: 3_000_000_000,
      progress: 50,
    });
    expect(half?.loadedBytes).toBe(1_500_000_000);
    expect(half?.overallProgressPct).toBeGreaterThan(40);
    expect(half?.overallProgressPct).toBeLessThan(60);

    const done = aggregator.handle({
      status: 'done',
      name: gemma.repoId,
      file: 'onnx/decoder_model_merged_q4f16.onnx_data',
    });
    expect(done?.stage).toBe('initializing-session');
    expect(done?.overallProgressPct).toBe(99);
  });

  it('uses ResolvedDtype effectiveWeightsBytes for WASM split embed_tokens downloads', () => {
    const qwen = getModelById('qwen-3.5-0.8b');
    const resolved = resolveEffectiveDtype(
      qwen,
      { backend: 'wasm', supportsFp16: false, features: [] },
      'q4'
    );
    const aggregator = createDownloadProgressAggregator(qwen, resolved);
    const expectedWasmBytes = getResolvedDownloadBytes(qwen, resolved, true);

    expect(expectedWasmBytes).toBe(994_861_508 + qwen.tokenizerBytes);
    const init = aggregator.handle({
      status: 'initiate',
      name: qwen.repoId,
      file: 'onnx/embed_tokens_fp16.onnx_data',
    });
    expect(init?.expectedTotalBytes).toBe(expectedWasmBytes);
  });
});

describe('LlmRuntimeClient', () => {
  it('probes devices, loads model with progress updates, and streams tokens cleanly', async () => {
    const worker = new MockWorker();
    const client = new LlmRuntimeClient({ workerFactory: () => worker });

    const probePromise = client.probeDevices();
    const probeReq = worker.postedMessages[0]!;
    expect(probeReq.type).toBe('probe-devices');
    worker.emitMessage({
      type: 'devices-probed',
      requestId: probeReq.requestId,
      devices: [SAMPLE_DEVICE],
    });
    expect(await probePromise).toEqual([SAMPLE_DEVICE]);

    const progressEvents: ModelLoadProgress[] = [];
    const loadPromise = client.loadModel(
      {
        modelId: 'qwen-3.5-0.8b',
        deviceTarget: {
          backend: 'webgpu',
          gpuPreference: 'high-performance',
          expectedDeviceFingerprint: SAMPLE_DEVICE.fingerprint,
        },
        dtype: 'q4f16',
      },
      (p) => progressEvents.push(p)
    );

    const loadReq = worker.postedMessages[1]!;
    expect(loadReq.type).toBe('load-model');
    worker.emitMessage({
      type: 'load-progress',
      requestId: loadReq.requestId,
      progress: {
        modelId: 'qwen-3.5-0.8b',
        stage: 'downloading',
        message: 'Downloading weights',
        overallProgressPct: 45,
        loadedBytes: 270_000_000,
        expectedTotalBytes: 603_931_785,
        files: [],
      },
    });
    worker.emitMessage({
      type: 'model-loaded',
      requestId: loadReq.requestId,
      modelId: 'qwen-3.5-0.8b',
      device: SAMPLE_DEVICE,
      effectiveDtype: 'q4',
      useExternalDataFormat: false,
      loadTimeMs: 1234,
      fallbackNotice: 'Fallback to q4',
    });

    const loaded = await loadPromise;
    expect(progressEvents).toHaveLength(1);
    expect(progressEvents[0]?.overallProgressPct).toBe(45);
    expect(loaded).toMatchObject({
      modelId: 'qwen-3.5-0.8b',
      effectiveDtype: 'q4',
      fallbackNotice: 'Fallback to q4',
    });

    const starts: number[] = [];
    const tokens: string[] = [];
    const genPromise = client.generate(
      [{ role: 'user', content: 'Hello' }],
      SAMPLE_GEN_CONFIG,
      {
        onStart: ({ promptTokens }) => starts.push(promptTokens),
        onToken: (ev) => tokens.push(ev.deltaText),
      }
    );

    const genReq = worker.postedMessages[2]!;
    expect(genReq.type).toBe('generate');
    worker.emitMessage({
      type: 'generation-started',
      requestId: genReq.requestId,
      promptTokens: 18,
    });
    worker.emitMessage({
      type: 'token',
      requestId: genReq.requestId,
      deltaText: 'Hi ',
      fullText: 'Hi ',
      tokenCount: 1,
      timestampMs: 1050,
      isFirstToken: true,
    });
    worker.emitMessage({
      type: 'token',
      requestId: genReq.requestId,
      deltaText: 'there!',
      fullText: 'Hi there!',
      tokenCount: 2,
      timestampMs: 1080,
      isFirstToken: false,
    });
    worker.emitMessage({
      type: 'generation-complete',
      requestId: genReq.requestId,
      fullText: 'Hi there!',
      thinking: '',
      content: 'Hi there!',
      aborted: false,
      speed: computeSpeedMetrics(18, [1050, 1080], 1000, 1085),
      workerJsHeapUsedBytes: 50_000_000,
      activeThreads: 1,
      backend: 'webgpu',
      dtype: 'q4',
    });

    const genResult = await genPromise;
    expect(starts).toEqual([18]);
    expect(tokens).toEqual(['Hi ', 'there!']);
    expect(genResult.content).toBe('Hi there!');
    expect(genResult.aborted).toBe(false);
    expect(genResult.workerJsHeapUsedBytes).toBe(50_000_000);
  });

  it('aborts active generation, guards concurrent exclusive ops, and supports cache/unload', async () => {
    const worker = new MockWorker();
    const client = new LlmRuntimeClient({ workerFactory: () => worker });

    // No-op abort when idle
    client.abort();
    expect(worker.postedMessages).toHaveLength(0);

    const genPromise = client.generate([{ role: 'user', content: 'Count' }], SAMPLE_GEN_CONFIG);
    const genReq = worker.postedMessages[0]!;

    // Concurrent generate / loadModel / unloadModel / clearCache rejected while generating
    await expect(
      client.generate([{ role: 'user', content: 'Again' }], SAMPLE_GEN_CONFIG)
    ).rejects.toMatchObject({ name: 'LlmRuntimeError', code: 'CONCURRENT_OPERATION' });
    await expect(
      client.loadModel({
        modelId: 'qwen-3.5-0.8b',
        deviceTarget: { backend: 'wasm' },
      })
    ).rejects.toMatchObject({ code: 'CONCURRENT_OPERATION' });
    await expect(client.unloadModel()).rejects.toMatchObject({ code: 'CONCURRENT_OPERATION' });

    client.abort();
    const abortReq = worker.postedMessages[1]!;
    expect(abortReq).toMatchObject({
      type: 'abort',
      targetRequestId: genReq.requestId,
    });

    worker.emitMessage({
      type: 'generation-complete',
      requestId: genReq.requestId,
      fullText: '1, 2',
      thinking: '',
      content: '1, 2',
      aborted: true,
      speed: computeSpeedMetrics(10, [100, 140], 50, 145),
      activeThreads: 1,
      backend: 'wasm',
      dtype: 'q4',
    });
    const abortedRes = await genPromise;
    expect(abortedRes.aborted).toBe(true);
    expect(client.currentOperation).toBeNull();

    // Unload, inspectCache, clearCache work once previous operation settled
    const unloadPromise = client.unloadModel();
    const unloadReq = worker.postedMessages[2]!;
    worker.emitMessage({ type: 'model-unloaded', requestId: unloadReq.requestId });
    await expect(unloadPromise).resolves.toBeUndefined();

    const inspectPromise = client.inspectCache(['qwen-3.5-0.8b']);
    const inspectReq = worker.postedMessages[3]!;
    worker.emitMessage({
      type: 'cache-info',
      requestId: inspectReq.requestId,
      caches: [],
    });
    expect(await inspectPromise).toEqual([]);

    const clearPromise = client.clearCache('qwen-3.5-0.8b');
    const clearReq = worker.postedMessages[4]!;
    worker.emitMessage({
      type: 'cache-cleared',
      requestId: clearReq.requestId,
      modelId: 'qwen-3.5-0.8b',
      clearedBytes: 1024,
      deletedEntries: 1,
    });
    expect(await clearPromise).toEqual({
      modelId: 'qwen-3.5-0.8b',
      clearedBytes: 1024,
      deletedEntries: 1,
    });
  });

  it('propagates worker errors, recovers from worker crashes, and cleans up on dispose', async () => {
    const workers: MockWorker[] = [];
    const unexpectedErrors: LlmRuntimeError[] = [];
    const client = new LlmRuntimeClient({
      workerFactory: () => {
        const w = new MockWorker();
        workers.push(w);
        return w;
      },
      onUnexpectedError: (err) => unexpectedErrors.push(err),
    });

    // Request-scoped worker error
    const loadPromise = client.loadModel({
      modelId: 'gemma-4-e2b',
      deviceTarget: { backend: 'webgpu', gpuPreference: 'high-performance' },
    });
    const loadReq = workers[0]!.postedMessages[0]!;
    workers[0]!.emitMessage({
      type: 'error',
      requestId: loadReq.requestId,
      message: 'Out of memory',
      code: 'MODEL_LOAD_FAILED',
      recoverable: true,
    });
    await expect(loadPromise).rejects.toMatchObject({
      message: 'Out of memory',
      code: 'MODEL_LOAD_FAILED',
      recoverable: true,
    });

    // Unsolicited device-lost notification
    workers[0]!.emitMessage({
      type: 'error',
      requestId: 'device-lost',
      message: 'GPU reset',
      code: 'WEBGPU_DEVICE_LOST',
      recoverable: true,
    });
    expect(unexpectedErrors).toHaveLength(1);
    expect(unexpectedErrors[0]?.code).toBe('WEBGPU_DEVICE_LOST');

    // Crash recovery: worker emits error event -> rejects pending + terminates crashed worker
    const pendingBeforeCrash = client.probeDevices();
    workers[0]!.emitError('Worker out of memory');
    await expect(pendingBeforeCrash).rejects.toMatchObject({
      code: 'WORKER_CRASHED',
      recoverable: true,
    });
    expect(workers[0]!.terminated).toBe(true);
    expect(workers[0]!.listenerCount()).toBe(0);

    // Next request spawns a fresh worker seamlessly
    const recoveredPromise = client.probeDevices();
    expect(workers).toHaveLength(2);
    const recoveredReq = workers[1]!.postedMessages[0]!;
    workers[1]!.emitMessage({
      type: 'devices-probed',
      requestId: recoveredReq.requestId,
      devices: [SAMPLE_DEVICE],
    });
    expect(await recoveredPromise).toEqual([SAMPLE_DEVICE]);

    // Dispose terminates active worker, rejects in-flight requests, and blocks further use
    const inFlight = client.inspectCache();
    client.dispose();
    await expect(inFlight).rejects.toMatchObject({ code: 'CLIENT_DISPOSED' });
    expect(client.isDisposed).toBe(true);
    expect(workers[1]!.terminated).toBe(true);
    expect(workers[1]!.listenerCount()).toBe(0);
    await expect(client.probeDevices()).rejects.toMatchObject({ code: 'CLIENT_DISPOSED' });
  });
});

describe('modelPrecacher & low-memory worker safeguards', () => {
  it('constructs exact Hugging Face resolve URLs and sanitizes streamed response headers', () => {
    expect(
      buildRemoteModelFileUrl(
        'onnx-community/Qwen3.5-0.8B-ONNX',
        '/onnx/decoder_model_merged_q4f16.onnx_data'
      )
    ).toBe(
      'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data'
    );

    const compressedHeaders = new Headers({
      'content-encoding': 'br',
      'content-length': '120',
      'content-type': 'application/json',
      etag: '"abc"',
    });
    expect(resolveExpectedContentLength(compressedHeaders)).toBeUndefined();
    const cleaned = buildPrecachedResponseHeaders(compressedHeaders);
    expect(cleaned.get('content-encoding')).toBeNull();
    expect(cleaned.get('content-length')).toBeNull();
    expect(cleaned.get('content-type')).toBe('application/json');

    const lfsHeaders = new Headers({
      'x-linked-size': '584705674',
      'content-type': 'application/octet-stream',
    });
    expect(resolveExpectedContentLength(lfsHeaders)).toBe(584_705_674);
    const preserved = buildPrecachedResponseHeaders(lfsHeaders);
    expect(preserved.get('content-length')).toBe('584705674');

    expect(isOptionalModelFile('generation_config.json')).toBe(true);
    expect(isOptionalModelFile('config.json')).toBe(false);
    expect(isOptionalModelFile('onnx/model_q4.onnx')).toBe(false);
  });

  it('augments config with use_external_data_format without mutating source and discovers ONNX + external data via ModelRegistry', async () => {
    const baseLlamaConfig: Record<string, unknown> = {
      model_type: 'llama',
      'transformers.js_config': {
        use_external_data_format: { 'model_q4f16.onnx': 1 },
      },
    };
    const augmented = resolveExternalDataAugmentedConfig(baseLlamaConfig, true);
    expect(
      (baseLlamaConfig['transformers.js_config'] as Record<string, unknown>).use_external_data_format
    ).toEqual({ 'model_q4f16.onnx': 1 });
    expect(
      (augmented['transformers.js_config'] as Record<string, unknown>).use_external_data_format
    ).toBe(true);

    const qwenConfig = {
      architectures: ['Qwen3_5ForConditionalGeneration'],
      model_type: 'qwen3_5',
      'transformers.js_config': {
        use_external_data_format: true,
      },
    };
    const qwenRawFiles = await ModelRegistry.get_model_files('onnx-community/Qwen3.5-0.8B-ONNX', {
      config: qwenConfig as never,
      dtype: 'q4f16',
      device: 'webgpu',
    });
    const qwenFiles = filterCausalLmModelFiles(qwenRawFiles, 'Qwen3_5ForConditionalGeneration');
    expect(qwenFiles).toContain('onnx/embed_tokens_q4f16.onnx');
    expect(qwenFiles).toContain('onnx/embed_tokens_q4f16.onnx_data');
    expect(qwenFiles).toContain('onnx/decoder_model_merged_q4f16.onnx');
    expect(qwenFiles).toContain('onnx/decoder_model_merged_q4f16.onnx_data');
    expect(qwenFiles.some((f) => f.includes('vision_encoder'))).toBe(false);

    const gemmaConfig = {
      architectures: ['Gemma4ForConditionalGeneration'],
      model_type: 'gemma4',
      text_config: {
        model_type: 'gemma4_text',
      },
      'transformers.js_config': {
        use_external_data_format: true,
      },
    };
    const gemmaRawFiles = await ModelRegistry.get_model_files(
      'onnx-community/gemma-4-E2B-it-ONNX',
      {
        config: gemmaConfig as never,
        dtype: 'q4f16',
        device: 'webgpu',
      }
    );
    const gemmaFiles = filterCausalLmModelFiles(gemmaRawFiles, 'Gemma4ForConditionalGeneration');
    expect(gemmaFiles).toContain('onnx/embed_tokens_q4f16.onnx');
    expect(gemmaFiles).toContain('onnx/decoder_model_merged_q4f16.onnx');
    expect(gemmaFiles.some((f) => f.includes('vision_encoder') || f.includes('audio_encoder'))).toBe(
      false
    );

    const llamaQ8Files = await ModelRegistry.get_model_files(
      'onnx-community/Llama-3.2-3B-Instruct-ONNX',
      {
        config: augmented as never,
        dtype: 'q8',
        device: 'webgpu',
      }
    );
    expect(llamaQ8Files).toContain('onnx/model_quantized.onnx');
    expect(llamaQ8Files).toContain('onnx/model_quantized.onnx_data');
  });

  it('downloads files strictly one-by-one into CacheStorage via stream, skips valid cached entries, and handles 404/aborts/empty binaries', async () => {
    const timeline: string[] = [];
    const stored = new Map<string, { bytes: number; headers: Headers }>();

    const cachedEmbedUrl = buildRemoteModelFileUrl(
      'onnx-community/Qwen3.5-0.8B-ONNX',
      'onnx/embed_tokens_q4f16.onnx_data'
    );
    const corruptEmptyDecoderUrl = buildRemoteModelFileUrl(
      'onnx-community/Qwen3.5-0.8B-ONNX',
      'onnx/decoder_model_merged_q4f16.onnx_data'
    );

    stored.set(cachedEmbedUrl, {
      bytes: 1234,
      headers: new Headers({ 'content-length': '1234' }),
    });
    // Simulate a corrupt 0-byte WebKit cache entry from a previous failed attempt
    stored.set(corruptEmptyDecoderUrl, {
      bytes: 0,
      headers: new Headers({ 'content-length': '0' }),
    });

    const mockCache: Cache = {
      match: vi.fn(async (req: RequestInfo | URL) => {
        const key = typeof req === 'string' ? req : req instanceof URL ? req.href : req.url;
        const entry = stored.get(key);
        if (!entry) return undefined;
        return new Response(new Uint8Array(entry.bytes), { status: 200, headers: entry.headers });
      }),
      put: vi.fn(async (req: RequestInfo | URL, response: Response) => {
        const key = typeof req === 'string' ? req : req instanceof URL ? req.href : req.url;
        const buf = await response.arrayBuffer();
        timeline.push(`put:${key.split('/').pop()}`);
        stored.set(key, { bytes: buf.byteLength, headers: new Headers(response.headers) });
      }),
      delete: vi.fn(async () => true),
      keys: vi.fn(async () => []),
      add: vi.fn(),
      addAll: vi.fn(),
      matchAll: vi.fn(),
    } as unknown as Cache;

    const mockStorage = {
      open: vi.fn(async () => mockCache),
    } as unknown as CacheStorage;

    const fetchFn = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const name = url.split('/').pop()!;
      timeline.push(`fetch:${name}`);

      if (name === 'generation_config.json') {
        return new Response(null, { status: 404, statusText: 'Not Found' });
      }

      const chunks =
        name === 'decoder_model_merged_q4f16.onnx_data'
          ? [new Uint8Array(64).fill(1), new Uint8Array(64).fill(2)]
          : [new Uint8Array(32).fill(3)];
      const total = chunks.reduce((s, c) => s + c.byteLength, 0);

      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(chunk);
          }
          controller.close();
        },
      });
      return new Response(body, {
        status: 200,
        headers: { 'content-length': String(total), 'content-type': 'application/octet-stream' },
      });
    }) as unknown as typeof fetch;

    const progressEvents: string[] = [];
    const result = await precacheModelFilesSequentially({
      repoId: 'onnx-community/Qwen3.5-0.8B-ONNX',
      files: [
        'onnx/embed_tokens_q4f16.onnx',
        'onnx/embed_tokens_q4f16.onnx_data',
        'onnx/decoder_model_merged_q4f16.onnx_data',
        'generation_config.json',
      ],
      cacheStorage: mockStorage,
      fetchFn,
      supportsStreamingPut: true,
      minEmitIntervalMs: 0,
      onFileProgress: (ev) => {
        progressEvents.push(`${ev.file}:${ev.status}:${ev.loaded ?? 0}/${ev.total ?? 0}`);
      },
    });

    expect(result).toEqual({
      usedBrowserCache: true,
      totalFiles: 4,
      downloadedFiles: 2,
      skippedCachedFiles: 1,
      downloadedBytes: 32 + 128,
    });

    // Strict sequential guarantee: file 1 finishes cache.put BEFORE file 2 starts fetching!
    expect(timeline).toEqual([
      'fetch:embed_tokens_q4f16.onnx',
      'put:embed_tokens_q4f16.onnx',
      'fetch:decoder_model_merged_q4f16.onnx_data',
      'put:decoder_model_merged_q4f16.onnx_data',
      'fetch:generation_config.json',
    ]);
    expect(stored.get(corruptEmptyDecoderUrl)?.bytes).toBe(128);
    expect(progressEvents).toContain('onnx/embed_tokens_q4f16.onnx_data:done:1234/1234');
    expect(progressEvents).toContain('onnx/decoder_model_merged_q4f16.onnx_data:done:128/128');

    // Required missing binary must throw loudly
    const failFetch = vi.fn(
      async () => new Response(null, { status: 404, statusText: 'Not Found' })
    ) as unknown as typeof fetch;
    await expect(
      precacheModelFilesSequentially({
        repoId: 'onnx-community/Qwen3.5-0.8B-ONNX',
        files: ['onnx/missing_required.onnx'],
        cacheStorage: mockStorage,
        fetchFn: failFetch,
        supportsStreamingPut: true,
      })
    ).rejects.toThrow('onnx/missing_required.onnx');
  });

  it('spools via OPFS sync access handle when WebKit fails stream-backed Cache.put, cleans temp OPFS file, and maps QuotaExceededError', async () => {
    const webkitBrokenCache: Cache = {
      match: vi.fn(async () => new Response(new Uint8Array(0), { status: 200 })),
      put: vi.fn(async () => undefined),
      delete: vi.fn(async () => true),
    } as unknown as Cache;
    expect(await probeStreamingCachePutSupport(webkitBrokenCache)).toBe(false);

    const writtenChunks: number[] = [];
    const removeEntrySpy = vi.fn(async () => undefined);
    const closeSpy = vi.fn();
    const mockOpfs = {
      getFileHandle: vi.fn(async () => ({
        createSyncAccessHandle: vi.fn(async () => ({
          write: (buf: BufferSource) => {
            writtenChunks.push((buf as Uint8Array).byteLength);
            return (buf as Uint8Array).byteLength;
          },
          flush: vi.fn(),
          close: closeSpy,
        })),
        getFile: vi.fn(async () => new Blob([new Uint8Array(256).fill(7)])),
      })),
      removeEntry: removeEntrySpy,
    };

    const storedBodies = new Map<string, number>();
    const backingCache: Cache = {
      match: vi.fn(async (req: RequestInfo | URL) => {
        const key = String(req);
        if (key.includes('__stream_cache_probe__')) {
          return new Response(new Uint8Array(0), { status: 200 });
        }
        return undefined;
      }),
      put: vi.fn(async (req: RequestInfo | URL, res: Response) => {
        const key = String(req);
        if (key.includes('__stream_cache_probe__')) return;
        const ab = await res.arrayBuffer();
        storedBodies.set(key, ab.byteLength);
      }),
      delete: vi.fn(async () => true),
    } as unknown as Cache;

    const storage = { open: vi.fn(async () => backingCache) } as unknown as CacheStorage;
    const fetchFn = vi.fn(async () => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new Uint8Array(128).fill(7));
          controller.enqueue(new Uint8Array(128).fill(7));
          controller.close();
        },
      });
      return new Response(body, {
        status: 200,
        headers: { 'content-length': '256' },
      });
    }) as unknown as typeof fetch;

    const res = await precacheModelFilesSequentially({
      repoId: 'onnx-community/Qwen3.5-0.8B-ONNX',
      files: ['onnx/embed_tokens_q4f16.onnx_data'],
      cacheStorage: storage,
      fetchFn,
      opfsRoot: mockOpfs,
      minEmitIntervalMs: 0,
    });
    expect(res.downloadedBytes).toBe(256);
    expect(writtenChunks).toEqual([128, 128]);
    expect(closeSpy).toHaveBeenCalledTimes(1);
    expect(removeEntrySpy).toHaveBeenCalledTimes(1);
    expect(
      storedBodies.get(
        buildRemoteModelFileUrl(
          'onnx-community/Qwen3.5-0.8B-ONNX',
          'onnx/embed_tokens_q4f16.onnx_data'
        )
      )
    ).toBe(256);

    // QuotaExceededError propagates with STORAGE_QUOTA_EXCEEDED
    const quotaCache: Cache = {
      match: vi.fn(async () => undefined),
      put: vi.fn(async () => {
        const err = new Error('Quota exceeded');
        err.name = 'QuotaExceededError';
        throw err;
      }),
      delete: vi.fn(async () => true),
    } as unknown as Cache;
    await expect(
      precacheModelFilesSequentially({
        repoId: 'onnx-community/Qwen3.5-0.8B-ONNX',
        files: ['onnx/decoder_model_merged_q4f16.onnx_data'],
        cacheStorage: { open: vi.fn(async () => quotaCache) } as unknown as CacheStorage,
        fetchFn,
        supportsStreamingPut: true,
      })
    ).rejects.toMatchObject({ code: 'STORAGE_QUOTA_EXCEEDED' });
  });

  it('serializes concurrent customCache reads via FIFO queue and classifies worker OOM/quota errors', async () => {
    let activeMatches = 0;
    let peakConcurrent = 0;
    const order: string[] = [];

    const slowCache: Cache = {
      match: vi.fn(async (req: RequestInfo | URL) => {
        activeMatches += 1;
        peakConcurrent = Math.max(peakConcurrent, activeMatches);
        order.push(`start:${String(req)}`);
        await new Promise((r) => setTimeout(r, 15));
        order.push(`end:${String(req)}`);
        activeMatches -= 1;
        return new Response('ok');
      }),
      put: vi.fn(async () => undefined),
    } as unknown as Cache;

    const adapter = createSerializedCacheAdapter(TRANSFORMERS_CACHE_NAME, {
      open: vi.fn(async () => slowCache),
    } as unknown as CacheStorage)!;

    await Promise.all([adapter.match('a.onnx_data'), adapter.match('b.onnx_data')]);
    expect(peakConcurrent).toBe(1);
    expect(order).toEqual([
      'start:a.onnx_data',
      'end:a.onnx_data',
      'start:b.onnx_data',
      'end:b.onnx_data',
    ]);

    expect(
      classifyModelLoadError(
        new Error("Can't create a session. failed to allocate a buffer of size 1200000000.")
      )
    ).toMatchObject({ code: 'OUT_OF_MEMORY' });

    const quotaErr = new Error('Quota exceeded');
    quotaErr.name = 'QuotaExceededError';
    expect(classifyModelLoadError(quotaErr)).toMatchObject({ code: 'STORAGE_QUOTA_EXCEEDED' });
  });
});
