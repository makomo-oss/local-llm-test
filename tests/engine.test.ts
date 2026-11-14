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
  OPFS_MODEL_CACHE_DIR,
  OPFS_PERSISTENT_THRESHOLD_BYTES,
  buildOpfsEntryStem,
  clearOpfsModelEntries,
  inspectOpfsModelEntries,
  readOpfsCachedResponse,
  shouldPreferPersistentOpfs,
  writeResponseToOpfsStore,
  writeStreamToOpfsStore,
  type OpfsDirectoryLike,
} from '../src/engine/opfsCacheStore.ts';
import {
  SessionLoadCoordinator,
  parseOnnxSessionGroup,
  releaseSessionExternalData,
  withSerializedInferenceSessions,
} from '../src/engine/sessionSerializer.ts';
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

    expect(expectedWasmBytes).toBe(633_308_015 + qwen.tokenizerBytes);
    const init = aggregator.handle({
      status: 'initiate',
      name: qwen.repoId,
      file: 'onnx/embed_tokens_q4f16.onnx_data',
    });
    expect(init?.expectedTotalBytes).toBe(expectedWasmBytes);

    const legacyResolved = resolveEffectiveDtype(
      qwen,
      { backend: 'wasm', supportsFp16: false, features: [] },
      'q4',
      { onDemandEmbedTokens: false }
    );
    expect(getResolvedDownloadBytes(qwen, legacyResolved, true)).toBe(
      994_861_508 + qwen.tokenizerBytes
    );
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

    const writeErr = Object.assign(new Error('Unexpected internal error'), {
      code: 'STORAGE_WRITE_FAILED',
    });
    expect(classifyModelLoadError(writeErr)).toMatchObject({ code: 'STORAGE_WRITE_FAILED' });

    expect(
      classifyModelLoadError(new Error('WebGPU validation failed: device is lost'))
    ).toMatchObject({ code: 'WEBGPU_DEVICE_LOST' });

    expect(classifyModelLoadError(new Error('network timeout'))).toMatchObject({
      code: 'MODEL_LOAD_FAILED',
    });
  });
});

interface MockOpfsNode {
  files: Map<string, Uint8Array>;
  dirs: Map<string, MockOpfsNode>;
}

function createInMemoryOpfsRoot(
  options: { supportsSyncAccess?: boolean } = {}
): { root: OpfsDirectoryLike; tree: MockOpfsNode } {
  const supportsSync = options.supportsSyncAccess !== false;
  const tree: MockOpfsNode = { files: new Map(), dirs: new Map() };

  const wrapDir = (node: MockOpfsNode): OpfsDirectoryLike => ({
    async getDirectoryHandle(name, opts) {
      let child = node.dirs.get(name);
      if (!child) {
        if (!opts?.create) {
          throw new DOMException('NotFoundError', 'NotFoundError');
        }
        child = { files: new Map(), dirs: new Map() };
        node.dirs.set(name, child);
      }
      return wrapDir(child);
    },
    async getFileHandle(name, opts) {
      if (!node.files.has(name)) {
        if (!opts?.create) {
          throw new DOMException('NotFoundError', 'NotFoundError');
        }
        node.files.set(name, new Uint8Array(0));
      }
      const handle: ReturnType<OpfsDirectoryLike['getFileHandle']> extends Promise<infer H>
        ? H
        : never = {
        async getFile() {
          const data = node.files.get(name) ?? new Uint8Array(0);
          return new Blob([Uint8Array.from(data)]);
        },
        ...(supportsSync
          ? {
              async createSyncAccessHandle() {
                let buf = Uint8Array.from(node.files.get(name) ?? new Uint8Array(0));
                let closed = false;
                return {
                  truncate(newSize: number) {
                    const next = new Uint8Array(newSize);
                    next.set(buf.subarray(0, Math.min(buf.length, newSize)));
                    buf = next;
                  },
                  write(view, writeOpts) {
                    if (closed) throw new Error('SyncAccessHandle closed');
                    const src =
                      view instanceof ArrayBuffer
                        ? new Uint8Array(view)
                        : new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
                    const at = writeOpts?.at ?? buf.length;
                    const end = at + src.byteLength;
                    if (end > buf.length) {
                      const grown = new Uint8Array(end);
                      grown.set(buf);
                      buf = grown;
                    }
                    buf.set(src, at);
                    return src.byteLength;
                  },
                  flush() {
                    node.files.set(name, Uint8Array.from(buf));
                  },
                  close() {
                    if (!closed) {
                      closed = true;
                      node.files.set(name, Uint8Array.from(buf));
                    }
                  },
                };
              },
            }
          : {}),
        async createWritable() {
          const chunks: Uint8Array[] = [];
          let aborted = false;
          return {
            async write(data) {
              const src =
                data instanceof ArrayBuffer ? new Uint8Array(data) : Uint8Array.from(data);
              chunks.push(src);
            },
            async close() {
              if (aborted) return;
              const total = chunks.reduce((s, c) => s + c.byteLength, 0);
              const out = new Uint8Array(total);
              let off = 0;
              for (const c of chunks) {
                out.set(c, off);
                off += c.byteLength;
              }
              node.files.set(name, out);
            },
            async abort() {
              aborted = true;
            },
          };
        },
      };
      return handle;
    },
    async removeEntry(name, opts) {
      if (node.files.delete(name)) return;
      if (node.dirs.has(name)) {
        const sub = node.dirs.get(name)!;
        if (!opts?.recursive && (sub.files.size > 0 || sub.dirs.size > 0)) {
          throw new DOMException('InvalidModificationError', 'InvalidModificationError');
        }
        node.dirs.delete(name);
      }
    },
    async *entries() {
      for (const name of node.files.keys()) {
        yield [name, { kind: 'file', name }];
      }
      for (const name of node.dirs.keys()) {
        yield [name, { kind: 'directory', name }];
      }
    },
  });

  return { root: wrapDir(tree), tree };
}

function makeChunkedStream(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  let idx = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (idx >= chunks.length) {
        controller.close();
        return;
      }
      controller.enqueue(chunks[idx++]!);
    },
  });
}

describe('Mobile OPFS persistence, session serialization & worker lifecycle', () => {
  it('enforces shouldPreferPersistentOpfs rules at format and 4 MiB boundaries', () => {
    expect(shouldPreferPersistentOpfs('onnx/decoder_model_merged_q4.onnx')).toBe(true);
    expect(shouldPreferPersistentOpfs('https://huggingface.co/a/b/resolve/main/onnx/model.onnx_data?download=1')).toBe(true);
    expect(shouldPreferPersistentOpfs('onnx/model.onnx_data_1')).toBe(true);
    expect(shouldPreferPersistentOpfs('tokenizer.json', OPFS_PERSISTENT_THRESHOLD_BYTES)).toBe(true);
    expect(shouldPreferPersistentOpfs('tokenizer.json', OPFS_PERSISTENT_THRESHOLD_BYTES - 1)).toBe(false);
    expect(shouldPreferPersistentOpfs('config.json', 1024)).toBe(false);
  });

  it('streams chunks via OPFS SyncAccessHandle and createWritable fallback, validates length, and cleans up corruption or aborts', async () => {
    const { root, tree } = createInMemoryOpfsRoot({ supportsSyncAccess: true });
    const url = 'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/main/onnx/embed_tokens_fp16.onnx_data';
    const c1 = new Uint8Array([10, 20, 30, 40]);
    const c2 = new Uint8Array([50, 60, 70]);

    const written = await writeStreamToOpfsStore({
      opfsRoot: root,
      url,
      stream: makeChunkedStream([c1, c2]),
      headers: new Headers({ 'content-encoding': 'gzip', 'content-type': 'application/octet-stream' }),
      expectedTotalBytes: 7,
    });
    expect(written).toBe(7);

    const hit = await readOpfsCachedResponse(root, url);
    expect(hit).toBeDefined();
    expect(hit?.headers.get('content-length')).toBe('7');
    expect(hit?.headers.get('content-encoding')).toBeNull();
    expect(Array.from(new Uint8Array(await hit!.arrayBuffer()))).toEqual([10, 20, 30, 40, 50, 60, 70]);

    // Corrupt truncated binary in OPFS should self-heal to undefined and delete corrupt entries
    const cacheDirNode = tree.dirs.get(OPFS_MODEL_CACHE_DIR)!;
    const stem = buildOpfsEntryStem(url);
    cacheDirNode.files.set(`${stem}.bin`, new Uint8Array([1, 2]));
    expect(await readOpfsCachedResponse(root, url)).toBeUndefined();
    expect(cacheDirNode.files.has(`${stem}.meta.json`)).toBe(false);
    expect(cacheDirNode.files.has(`${stem}.bin`)).toBe(false);

    // Size mismatch & empty stream must reject and remove partial files
    await expect(
      writeStreamToOpfsStore({
        opfsRoot: root,
        url,
        stream: makeChunkedStream([new Uint8Array([1, 2])]),
        headers: new Headers(),
        expectedTotalBytes: 99,
      })
    ).rejects.toThrow(/size mismatch/i);
    expect(cacheDirNode.files.size).toBe(0);

    await expect(
      writeStreamToOpfsStore({
        opfsRoot: root,
        url,
        stream: makeChunkedStream([]),
        headers: new Headers(),
      })
    ).rejects.toThrow(/empty binary/i);
    expect(cacheDirNode.files.size).toBe(0);

    // Mid-stream abort must clean up partial entry and propagate AbortError
    const ac = new AbortController();
    let pulled = 0;
    const abortingStream = new ReadableStream<Uint8Array>({
      pull(controller) {
        pulled += 1;
        if (pulled === 1) {
          controller.enqueue(new Uint8Array([1, 2, 3]));
          ac.abort();
        } else {
          controller.enqueue(new Uint8Array([4, 5]));
        }
      },
    });
    await expect(
      writeStreamToOpfsStore({
        opfsRoot: root,
        url,
        stream: abortingStream,
        headers: new Headers(),
        signal: ac.signal,
      })
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(cacheDirNode.files.size).toBe(0);

    // createWritable fallback without createSyncAccessHandle
    const { root: writableRoot } = createInMemoryOpfsRoot({ supportsSyncAccess: false });
    const writableWritten = await writeResponseToOpfsStore(
      writableRoot,
      url,
      new Response(new Uint8Array([9, 8, 7, 6]), {
        status: 200,
        headers: { 'content-length': '4' },
      })
    );
    expect(writableWritten).toBe(4);
    const writableHit = await readOpfsCachedResponse(writableRoot, url);
    expect(Array.from(new Uint8Array(await writableHit!.arrayBuffer()))).toEqual([9, 8, 7, 6]);
  });

  it('inspects and clears model caches seamlessly across both OPFS and CacheStorage', async () => {
    const qwen = getModelById('qwen-3.5-0.8b');
    const llama = getModelById('llama-3.2-3b');
    const { root: opfsRoot, tree } = createInMemoryOpfsRoot();

    const qwenOnnxUrl = buildRemoteModelFileUrl(qwen.repoId, 'onnx/decoder_model_merged_q4.onnx_data');
    const qwenTokUrl = buildRemoteModelFileUrl(qwen.repoId, 'tokenizer.json');
    const llamaOnnxUrl = buildRemoteModelFileUrl(llama.repoId, 'onnx/model_q4f16.onnx_data');

    await writeStreamToOpfsStore({
      opfsRoot,
      url: qwenOnnxUrl,
      stream: makeChunkedStream([new Uint8Array(12)]),
      headers: new Headers(),
      expectedTotalBytes: 12,
    });
    await writeStreamToOpfsStore({
      opfsRoot,
      url: llamaOnnxUrl,
      stream: makeChunkedStream([new Uint8Array(20)]),
      headers: new Headers(),
      expectedTotalBytes: 20,
    });
    // Legacy temp spool file should also be removed when clearing everything
    tree.files.set('__local_llm_precache_tmp__.bin', new Uint8Array([1, 2, 3]));

    const mockStorage = createMockCacheStorage({
      [TRANSFORMERS_CACHE_NAME]: {
        [qwenTokUrl]: { contentLength: '5', blobBytes: 5 },
      },
    });
    vi.stubGlobal('caches', mockStorage.storage);

    try {
      const inspected = await inspectModelCaches([qwen, llama], { opfsRoot });
      expect(inspected[0]).toMatchObject({ modelId: qwen.id, cached: true, cachedBytes: 17, fileCount: 2 });
      expect(inspected[1]).toMatchObject({ modelId: llama.id, cached: true, cachedBytes: 20, fileCount: 1 });

      const clearedQwen = await clearModelCache(qwen.id, { opfsRoot });
      expect(clearedQwen).toEqual({ clearedBytes: 17, deletedEntries: 2 });

      const afterQwenClear = await inspectModelCaches([qwen, llama], { opfsRoot });
      expect(afterQwenClear[0]?.cached).toBe(false);
      expect(afterQwenClear[1]?.cachedBytes).toBe(20);

      const clearedAll = await clearModelCache(undefined, { opfsRoot });
      expect(clearedAll.clearedBytes).toBe(20);
      expect(tree.files.has('__local_llm_precache_tmp__.bin')).toBe(false);
      expect(await inspectOpfsModelEntries(opfsRoot)).toEqual([]);
      expect((await clearOpfsModelEntries(null)).deletedEntries).toBe(0);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('serializes multi-session weight loading across constructSessions, frees externalData immediately, and aborts waiters on error', async () => {
    expect(parseOnnxSessionGroup('https://hf.co/r/resolve/main/onnx/embed_tokens_fp16.onnx_data')).toBe('embed_tokens');
    expect(parseOnnxSessionGroup('onnx/decoder_model_merged_q4.onnx')).toBe('decoder_model_merged');
    expect(parseOnnxSessionGroup('onnx/vision_encoder_q4.onnx_data_1')).toBe('vision_encoder');
    expect(parseOnnxSessionGroup('onnx/audio_encoder.onnx')).toBe('audio_encoder');
    expect(parseOnnxSessionGroup('onnx/model_q4f16.onnx')).toBe('model');
    expect(parseOnnxSessionGroup('tokenizer.json')).toBeNull();

    const { root: opfsRoot } = createInMemoryOpfsRoot();
    const embedUrl = 'https://hf.co/repo/resolve/main/onnx/embed_tokens_fp16.onnx_data';
    const decoderUrl = 'https://hf.co/repo/resolve/main/onnx/decoder_model_merged_q4.onnx_data';
    await writeResponseToOpfsStore(opfsRoot, embedUrl, new Response(new Uint8Array(10)));
    await writeResponseToOpfsStore(opfsRoot, decoderUrl, new Response(new Uint8Array(10)));

    const coordinator = new SessionLoadCoordinator();
    const ac = new AbortController();
    const adapter = createSerializedCacheAdapter(TRANSFORMERS_CACHE_NAME, undefined, {
      opfsRoot,
      coordinator,
      signal: ac.signal,
    })!;

    const timeline: string[] = [];
    let concurrentSessions = 0;
    let peakConcurrent = 0;
    const origTargetCreate = vi.fn(async (buf: unknown, opts?: { externalData?: Array<{ data: Uint8Array }> }) => {
      expect(opts?.externalData?.[0]?.data.byteLength).toBe(10);
      timeline.push(`create:${String(buf)}`);
      concurrentSessions += 1;
      peakConcurrent = Math.max(peakConcurrent, concurrentSessions);
      await new Promise((r) => setTimeout(r, 15));
      concurrentSessions -= 1;
      timeline.push(`created:${String(buf)}`);
      return { sessionId: buf };
    });
    const fakeSessionTarget = {
      create: origTargetCreate as unknown as (
        buf: unknown,
        opts?: { externalData?: Array<{ path: string; data: Uint8Array }> }
      ) => Promise<unknown>,
    };

    await withSerializedInferenceSessions(
      coordinator,
      async () => {
        await Promise.all(
          [
            { group: 'embed_tokens', url: embedUrl },
            { group: 'decoder_model_merged', url: decoderUrl },
          ].map(async ({ group, url }) => {
            const res = await adapter.match(url);
            const buf = new Uint8Array(await res!.arrayBuffer());
            timeline.push(`read:${group}`);
            const externalData = [{ path: `${group}.onnx_data`, data: buf }];
            await fakeSessionTarget.create(group, { externalData });
            // After wrapped InferenceSession.create resolves, externalData must already be cleared
            expect(externalData.length).toBe(0);
          })
        );
      },
      { target: fakeSessionTarget as never, yieldAfterCreateMs: 5 }
    );

    expect(peakConcurrent).toBe(1);
    expect(timeline.indexOf('created:embed_tokens')).toBeLessThan(timeline.indexOf('read:decoder_model_merged'));
    // Original create function must be restored after exit
    expect(fakeSessionTarget.create).toBe(origTargetCreate);

    // releaseSessionExternalData clears array entries safely even with empty/non-object input
    const ext = [{ data: new Uint8Array(8) }];
    releaseSessionExternalData({ externalData: ext });
    expect(ext.length).toBe(0);

    // Error propagation must abort queued session waiters without hanging and restore target.create
    const errCoordinator = new SessionLoadCoordinator();
    const errAdapter = createSerializedCacheAdapter(TRANSFORMERS_CACHE_NAME, undefined, {
      opfsRoot,
      coordinator: errCoordinator,
    })!;
    await errAdapter.match(embedUrl);
    const pendingDecoder = errAdapter.match(decoderUrl);
    const failingCreate = { create: vi.fn(async () => {}) as never };
    await expect(
      withSerializedInferenceSessions(
        errCoordinator,
        async () => {
          throw new Error('session creation exploded');
        },
        { target: failingCreate, yieldAfterCreateMs: 0 }
      )
    ).rejects.toThrow('session creation exploded');
    await expect(pendingDecoder).rejects.toThrow('session creation exploded');
  });

  it('recycles cache puts to OPFS when Cache.put fails and tags unrecoverable storage failures cleanly', async () => {
    const { root: opfsRoot } = createInMemoryOpfsRoot();
    const smallUrl = 'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/main/tokenizer.json';
    const throwingCache: Cache = {
      match: vi.fn(async () => undefined),
      put: vi.fn(async () => {
        throw new DOMException('Unexpected internal error', 'UnknownError');
      }),
      delete: vi.fn(async () => false),
      keys: vi.fn(async () => []),
      add: vi.fn(),
      addAll: vi.fn(),
      matchAll: vi.fn(async () => []),
    } as unknown as Cache;

    const adapter = createSerializedCacheAdapter(
      TRANSFORMERS_CACHE_NAME,
      {
        open: async () => throwingCache,
        has: async () => true,
        delete: async () => true,
        keys: async () => [],
        match: async () => undefined,
      } as unknown as CacheStorage,
      { opfsRoot }
    )!;

    await adapter.put(smallUrl, new Response(new Uint8Array([1, 2, 3, 4]), { status: 200 }));
    const recovered = await adapter.match(smallUrl);
    expect(Array.from(new Uint8Array(await recovered!.arrayBuffer()))).toEqual([1, 2, 3, 4]);

    // When both OPFS is unavailable and Cache.put throws UnknownError during precache, label STORAGE_WRITE_FAILED
    await expect(
      precacheModelFilesSequentially({
        repoId: 'onnx-community/Qwen3.5-0.8B-ONNX',
        files: ['onnx/decoder.onnx_data'],
        opfsRoot: null,
        cacheStorage: {
          open: async () => throwingCache,
          has: async () => true,
          delete: async () => true,
          keys: async () => [],
          match: async () => undefined,
        } as unknown as CacheStorage,
        supportsStreamingPut: true,
        fetchFn: vi.fn(
          async () =>
            new Response(makeChunkedStream([new Uint8Array([1, 2, 3])]), {
              status: 200,
              headers: { 'content-length': '3' },
            })
        ),
      })
    ).rejects.toMatchObject({ code: 'STORAGE_WRITE_FAILED' });
  });

  it('recycles Web Worker on repeat loadModel() and post-load unloadModel() while preserving idle workers', async () => {
    const spawned: MockWorker[] = [];
    const client = new LlmRuntimeClient({
      workerFactory: () => {
        const w = new MockWorker();
        spawned.push(w);
        return w;
      },
    });

    // Unload before any model was loaded should NOT terminate the idle worker
    const idleUnload = client.unloadModel();
    expect(spawned).toHaveLength(1);
    const firstWorker = spawned[0]!;
    firstWorker.emitMessage({
      type: 'model-unloaded',
      requestId: firstWorker.postedMessages[0]!.requestId,
    });
    await idleUnload;
    expect(firstWorker.terminated).toBe(false);

    // First model load sets worker runtime state
    const load1 = client.loadModel({
      modelId: 'qwen-3.5-0.8b',
      deviceTarget: { backend: 'wasm', numThreads: 2 },
      mobileContext: { isMobile: true, deviceMemoryGb: 4 },
    });
    const load1Req = firstWorker.postedMessages.at(-1)!;
    expect(load1Req.type).toBe('load-model');
    if (load1Req.type === 'load-model') {
      expect(load1Req.mobileContext).toEqual({ isMobile: true, deviceMemoryGb: 4 });
    }
    const wasmDevice: DeviceOption = {
      id: 'wasm-cpu',
      label: 'CPU',
      backend: 'wasm',
      features: [],
      limits: {},
      supportsFp16: false,
      recommendedDtype: 'q4',
      available: true,
      runtimeDetails: SAMPLE_DEVICE.runtimeDetails,
    };

    firstWorker.emitMessage({
      type: 'model-loaded',
      requestId: load1Req.requestId,
      modelId: 'qwen-3.5-0.8b',
      device: wasmDevice,
      effectiveDtype: 'q4',
      useExternalDataFormat: true,
      loadTimeMs: 120,
    });
    await load1;
    expect(firstWorker.terminated).toBe(false);

    // Subsequent loadModel must terminate previous worker to free WebAssembly.Memory before spawning next
    const load2 = client.loadModel({
      modelId: 'qwen-3.5-0.8b',
      deviceTarget: { backend: 'wasm', numThreads: 1 },
    });
    expect(firstWorker.terminated).toBe(true);
    expect(spawned).toHaveLength(2);
    const secondWorker = spawned[1]!;
    secondWorker.emitMessage({
      type: 'model-loaded',
      requestId: secondWorker.postedMessages[0]!.requestId,
      modelId: 'qwen-3.5-0.8b',
      device: wasmDevice,
      effectiveDtype: 'q4',
      useExternalDataFormat: true,
      loadTimeMs: 80,
    });
    await load2;

    // unloadModel after a model was loaded must terminate worker after model-unloaded to reclaim linear memory
    const unloadAfterLoad = client.unloadModel();
    const unloadReq = secondWorker.postedMessages.at(-1)!;
    secondWorker.emitMessage({ type: 'model-unloaded', requestId: unloadReq.requestId });
    await unloadAfterLoad;
    expect(secondWorker.terminated).toBe(true);

    client.dispose();
  });

  it('prevents premature 99% initializing-session flash across sequential multi-file downloads when files are pre-registered', () => {
    const qwen = getModelById('qwen-3.5-0.8b');
    const resolved = resolveEffectiveDtype(
      qwen,
      { backend: 'wasm', supportsFp16: false, features: [] },
      'q4'
    );
    const aggregator = createDownloadProgressAggregator(qwen, resolved);
    aggregator.registerExpectedFiles([
      'onnx/embed_tokens_fp16.onnx',
      'onnx/embed_tokens_fp16.onnx_data',
      'onnx/decoder_model_merged_q4.onnx',
      'onnx/decoder_model_merged_q4.onnx_data',
    ]);

    aggregator.handle({ status: 'done', name: qwen.repoId, file: 'onnx/embed_tokens_fp16.onnx' });
    const mid = aggregator.handle({
      status: 'done',
      name: qwen.repoId,
      file: 'onnx/embed_tokens_fp16.onnx_data',
    });
    expect(mid?.stage).toBe('downloading');
    expect(mid?.overallProgressPct).toBeLessThan(99);

    aggregator.handle({
      status: 'done',
      name: qwen.repoId,
      file: 'onnx/decoder_model_merged_q4.onnx',
    });
    const readyToInit = aggregator.handle({
      status: 'done',
      name: qwen.repoId,
      file: 'onnx/decoder_model_merged_q4.onnx_data',
    });
    expect(readyToInit?.stage).toBe('initializing-session');
    expect(readyToInit?.overallProgressPct).toBe(99);
  });
});
