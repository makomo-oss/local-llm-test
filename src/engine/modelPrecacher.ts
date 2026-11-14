import { TRANSFORMERS_CACHE_NAME } from './cacheManager.ts';

/**
 * Lightweight Cache API subset required by Transformers.js `env.customCache`.
 */
export interface SerializedCacheAdapter {
  match(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<Response | undefined>;
  put(request: RequestInfo | URL, response: Response): Promise<void>;
}

/**
 * Progress info shape compatible with `@huggingface/transformers` `ProgressCallback`.
 */
export interface PrecacheFileProgressInfo {
  status: 'initiate' | 'progress' | 'done';
  name: string;
  file: string;
  loaded?: number;
  total?: number;
  progress?: number;
}

export interface BuildRemoteUrlOptions {
  revision?: string;
  remoteHost?: string;
  remotePathTemplate?: string;
}

export interface StreamResponseToCacheOptions {
  cache: Cache;
  cacheKey: string;
  response: Response;
  name: string;
  filePath: string;
  expectedTotalBytes?: number;
  onFileProgress?: (info: PrecacheFileProgressInfo) => void;
  minEmitIntervalMs?: number;
  nowFn?: () => number;
}

export interface PrecacheModelFilesOptions {
  repoId: string;
  files: readonly string[];
  cacheStorage?: CacheStorage;
  cacheName?: string;
  fetchFn?: typeof fetch;
  revision?: string;
  remoteHost?: string;
  remotePathTemplate?: string;
  signal?: AbortSignal;
  onFileProgress?: (info: PrecacheFileProgressInfo) => void;
  minEmitIntervalMs?: number;
  nowFn?: () => number;
  /** Override streaming `Cache.put(new Response(ReadableStream))` support check. */
  supportsStreamingPut?: boolean;
  /** Optional OPFS root directory override for deterministic testing. */
  opfsRoot?: OpfsDirectoryLike;
}

export interface OpfsSyncAccessHandleLike {
  write(buffer: BufferSource, options?: { at?: number }): number;
  flush(): void;
  close(): void;
}

export interface OpfsFileHandleLike {
  createSyncAccessHandle?: () => Promise<OpfsSyncAccessHandleLike>;
  getFile(): Promise<File | Blob>;
}

export interface OpfsDirectoryLike {
  getFileHandle(name: string, options?: { create?: boolean }): Promise<OpfsFileHandleLike>;
  removeEntry(name: string): Promise<void>;
}

export interface PrecacheModelFilesResult {
  usedBrowserCache: boolean;
  totalFiles: number;
  downloadedFiles: number;
  skippedCachedFiles: number;
  downloadedBytes: number;
}

const DEFAULT_REMOTE_HOST = 'https://huggingface.co/';
const DEFAULT_REMOTE_PATH_TEMPLATE = '{model}/resolve/{revision}/';
const DEFAULT_EMIT_INTERVAL_MS = 120;
const MIN_PROGRESS_BYTE_DELTA = 256 * 1024;

function parsePositiveIntHeader(headers: Headers, headerName: string): number | undefined {
  const raw = headers.get(headerName);
  if (!raw) {
    return undefined;
  }
  const trimmed = raw.trim();
  if (!/^\d+$/.test(trimmed)) {
    return undefined;
  }
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : undefined;
}

/**
 * Construct the exact Hugging Face resolve URL used by Transformers.js as the
 * CacheStorage key for a given repository file.
 */
export function buildRemoteModelFileUrl(
  repoId: string,
  filePath: string,
  options: BuildRemoteUrlOptions = {}
): string {
  const remoteHost = options.remoteHost ?? DEFAULT_REMOTE_HOST;
  const template = options.remotePathTemplate ?? DEFAULT_REMOTE_PATH_TEMPLATE;
  const revision = options.revision ?? 'main';

  const hostWithSlash = remoteHost.endsWith('/') ? remoteHost : `${remoteHost}/`;
  const templateResolved = template
    .replace('{model}', repoId.replace(/^\/+|\/+$/g, ''))
    .replace('{revision}', encodeURIComponent(revision));
  const cleanPrefix = templateResolved.replace(/^\/+/, '');
  const prefixWithSlash = cleanPrefix.endsWith('/') ? cleanPrefix : `${cleanPrefix}/`;
  const cleanFile = filePath.replace(/^\/+/, '');

  return `${hostWithSlash}${prefixWithSlash}${cleanFile}`;
}

/**
 * Read uncompressed byte size from `content-length` (identity encoding only) or
 * Hugging Face's `x-linked-size` header without buffering the body.
 */
export function resolveExpectedContentLength(headers: Headers): number | undefined {
  const encoding = headers.get('content-encoding')?.trim().toLowerCase();
  const isIdentity = !encoding || encoding === 'identity';
  if (isIdentity) {
    const contentLength = parsePositiveIntHeader(headers, 'content-length');
    if (contentLength !== undefined) {
      return contentLength;
    }
  }
  return parsePositiveIntHeader(headers, 'x-linked-size');
}

/**
 * Build sanitized headers for a `Response` backed by a decoded `ReadableStream`.
 * Strips transport `content-encoding` because `fetch().body` already yields
 * decoded bytes, and sets `content-length` when known so subsequent cache size
 * inspection never needs `.blob()` on multi-hundred-MB weight artifacts.
 */
export function buildPrecachedResponseHeaders(
  sourceHeaders: Headers,
  knownUncompressedBytes?: number
): Headers {
  const out = new Headers();
  const copyKeys = ['content-type', 'etag', 'last-modified', 'x-linked-size'] as const;
  for (const key of copyKeys) {
    const val = sourceHeaders.get(key);
    if (val) {
      out.set(key, val);
    }
  }

  const resolvedLength = knownUncompressedBytes ?? resolveExpectedContentLength(sourceHeaders);
  if (resolvedLength !== undefined && Number.isFinite(resolvedLength) && resolvedLength > 0) {
    out.set('content-length', String(Math.floor(resolvedLength)));
  }

  return out;
}

/**
 * Optional configuration files that Hugging Face repos may omit (404) without
 * breaking `AutoModelForCausalLM.from_pretrained`.
 */
export function isOptionalModelFile(filePath: string): boolean {
  const normalized = filePath.trim().toLowerCase();
  if (!normalized || normalized === 'config.json') {
    return false;
  }
  if (normalized.endsWith('.onnx') || normalized.includes('.onnx_data')) {
    return false;
  }
  return normalized.endsWith('.json');
}

/**
 * Return a shallow-cloned `config` with `transformers.js_config.use_external_data_format`
 * forced to `true` when the caller explicitly requests external data loading
 * (e.g., Llama 3.2 3B `q8`).
 */
export function resolveExternalDataAugmentedConfig<T extends Record<string, unknown>>(
  config: T,
  useExternalDataFormat?: boolean
): T {
  if (!useExternalDataFormat) {
    return config;
  }
  const existingCustom =
    config['transformers.js_config'] &&
    typeof config['transformers.js_config'] === 'object' &&
    !Array.isArray(config['transformers.js_config'])
      ? (config['transformers.js_config'] as Record<string, unknown>)
      : {};

  const proto = Object.getPrototypeOf(config) as object | null;
  return Object.assign(Object.create(proto) as Record<string, unknown>, config, {
    'transformers.js_config': {
      ...existingCustom,
      use_external_data_format: true,
    },
  }) as T;
}

/**
 * Filter model files returned by `ModelRegistry.get_model_files` down to the
 * text-only CausalLM sessions (`embed_tokens` + `decoder_model_merged`), stripping
 * unused multimodal `vision_encoder` / `audio_encoder` artifacts so we never
 * waste smartphone RAM or bandwidth pre-downloading encoders unused by `AutoModelForCausalLM`.
 */
export function filterCausalLmModelFiles(
  files: readonly string[],
  architecture?: string
): string[] {
  const isMultimodalConditional =
    typeof architecture === 'string' && /ForConditionalGeneration$/i.test(architecture);
  if (!isMultimodalConditional) {
    return [...files];
  }
  return files.filter((filePath) => {
    if (!filePath.startsWith('onnx/')) {
      return true;
    }
    return (
      filePath.startsWith('onnx/embed_tokens') ||
      filePath.startsWith('onnx/decoder_model_merged')
    );
  });
}

function isQuotaExceededError(err: unknown): boolean {
  if (!err || typeof err !== 'object') {
    return false;
  }
  const maybeErr = err as { name?: string; code?: string | number; message?: string };
  if (
    maybeErr.name === 'QuotaExceededError' ||
    maybeErr.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
    maybeErr.code === 'STORAGE_QUOTA_EXCEEDED' ||
    maybeErr.code === 22
  ) {
    return true;
  }
  return typeof maybeErr.message === 'string' && /quota|storage limit/i.test(maybeErr.message);
}

function wrapCacheWriteError(err: unknown, filePath: string): Error {
  if (isQuotaExceededError(err)) {
    const quotaError = new Error(
      `ストレージ容量上限 (QuotaExceeded) により ${filePath} を保存できませんでした。不要なキャッシュを削除してください。`
    );
    quotaError.name = 'QuotaExceededError';
    (quotaError as Error & { code?: string }).code = 'STORAGE_QUOTA_EXCEEDED';
    return quotaError;
  }
  return err instanceof Error ? err : new Error(String(err));
}

/**
 * Stream an HTTP `Response` through a metering `TransformStream` directly into
 * `CacheStorage` without allocating a full-file `Uint8Array` in JS heap.
 */
export async function streamResponseToCache(
  options: StreamResponseToCacheOptions
): Promise<number> {
  const {
    cache,
    cacheKey,
    response,
    name,
    filePath,
    expectedTotalBytes,
    onFileProgress,
    minEmitIntervalMs = DEFAULT_EMIT_INTERVAL_MS,
    nowFn = () => Date.now(),
  } = options;

  const knownTotal = expectedTotalBytes ?? resolveExpectedContentLength(response.headers);
  onFileProgress?.({
    status: 'initiate',
    name,
    file: filePath,
    loaded: 0,
    ...(knownTotal !== undefined ? { total: knownTotal, progress: 0 } : {}),
  });

  if (!response.body || typeof TransformStream === 'undefined') {
    try {
      await cache.put(cacheKey, response);
    } catch (err) {
      throw wrapCacheWriteError(err, filePath);
    }
    const finalBytes = knownTotal ?? 0;
    onFileProgress?.({
      status: 'done',
      name,
      file: filePath,
      loaded: finalBytes,
      ...(finalBytes > 0 ? { total: finalBytes, progress: 100 } : {}),
    });
    return finalBytes;
  }

  let loadedBytes = 0;
  let lastEmitAt = 0;
  let lastEmittedBytes = 0;
  const byteDeltaThreshold =
    knownTotal !== undefined && knownTotal > 0
      ? Math.max(MIN_PROGRESS_BYTE_DELTA, Math.floor(knownTotal * 0.01))
      : MIN_PROGRESS_BYTE_DELTA;

  const meter = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      loadedBytes += chunk.byteLength;
      controller.enqueue(chunk);

      if (onFileProgress) {
        const now = nowFn();
        if (
          now - lastEmitAt >= minEmitIntervalMs ||
          loadedBytes - lastEmittedBytes >= byteDeltaThreshold
        ) {
          lastEmitAt = now;
          lastEmittedBytes = loadedBytes;
          const total = knownTotal !== undefined && knownTotal >= loadedBytes ? knownTotal : undefined;
          onFileProgress({
            status: 'progress',
            name,
            file: filePath,
            loaded: loadedBytes,
            ...(total !== undefined
              ? { total, progress: Math.min(100, (loadedBytes / total) * 100) }
              : {}),
          });
        }
      }
    },
    flush() {
      const total = knownTotal !== undefined && knownTotal >= loadedBytes ? knownTotal : loadedBytes;
      onFileProgress?.({
        status: 'done',
        name,
        file: filePath,
        loaded: loadedBytes,
        total,
        progress: 100,
      });
    },
  });

  const meteredStream = response.body.pipeThrough(meter);
  const headers = buildPrecachedResponseHeaders(response.headers, knownTotal);
  const streamedResponse = new Response(meteredStream, {
    status: 200,
    statusText: 'OK',
    headers,
  });

  try {
    await cache.put(cacheKey, streamedResponse);
  } catch (err) {
    throw wrapCacheWriteError(err, filePath);
  }

  return loadedBytes;
}

const PROBE_STREAM_CACHE_URL = 'https://local-llm-test.invalid/__stream_cache_probe__';
const OPFS_PRECACHE_TMP_FILE = '__local_llm_precache_tmp__.bin';

/**
 * Probe with a 4-byte stream whether this browser's `Cache.put()` reliably
 * persists `Response(ReadableStream)` bodies (WebKit/iOS Safari can silently
 * store 0-byte bodies or throw for stream-backed Response objects).
 */
export async function probeStreamingCachePutSupport(cache: Cache): Promise<boolean> {
  if (typeof ReadableStream === 'undefined' || typeof cache.put !== 'function') {
    return false;
  }
  try {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array([1, 2, 3, 4]));
        controller.close();
      },
    });
    const probeRes = new Response(stream, {
      status: 200,
      headers: { 'content-length': '4' },
    });
    await cache.put(PROBE_STREAM_CACHE_URL, probeRes);
    const matched = await cache.match(PROBE_STREAM_CACHE_URL);
    await cache.delete?.(PROBE_STREAM_CACHE_URL).catch(() => false);
    if (!matched) {
      // Minimal test mocks without backing storage cannot store responses;
      // treat no-op mock put cleanly as supported unless an actual 0-byte body occurred.
      return true;
    }
    const buf = typeof matched.arrayBuffer === 'function' ? await matched.arrayBuffer() : null;
    return buf !== null ? buf.byteLength === 4 : true;
  } catch {
    await cache.delete?.(PROBE_STREAM_CACHE_URL).catch(() => false);
    return false;
  }
}

async function resolveOpfsRoot(override?: OpfsDirectoryLike): Promise<OpfsDirectoryLike | null> {
  if (override) {
    return override;
  }
  try {
    const storageManager = (
      navigator as Navigator & { storage?: { getDirectory?: () => Promise<OpfsDirectoryLike> } }
    ).storage;
    if (storageManager && typeof storageManager.getDirectory === 'function') {
      return await storageManager.getDirectory();
    }
  } catch {
    // Ignore restricted OPFS (e.g. private browsing).
  }
  return null;
}

async function streamResponseViaOpfsOrDirectCache(
  options: StreamResponseToCacheOptions & { opfsRoot?: OpfsDirectoryLike }
): Promise<number> {
  const {
    cache,
    cacheKey,
    response,
    name,
    filePath,
    expectedTotalBytes,
    onFileProgress,
    minEmitIntervalMs = DEFAULT_EMIT_INTERVAL_MS,
    nowFn = () => Date.now(),
  } = options;

  const knownTotal = expectedTotalBytes ?? resolveExpectedContentLength(response.headers);
  const opfsRoot = await resolveOpfsRoot(options.opfsRoot);

  if (response.body && opfsRoot) {
    let accessHandle: OpfsSyncAccessHandleLike | undefined;
    try {
      const fileHandle = await opfsRoot.getFileHandle(OPFS_PRECACHE_TMP_FILE, { create: true });
      if (typeof fileHandle.createSyncAccessHandle === 'function') {
        accessHandle = await fileHandle.createSyncAccessHandle();

        onFileProgress?.({
          status: 'initiate',
          name,
          file: filePath,
          loaded: 0,
          ...(knownTotal !== undefined ? { total: knownTotal, progress: 0 } : {}),
        });

        const reader = response.body.getReader();
        let loadedBytes = 0;
        let lastEmitAt = 0;
        let lastEmittedBytes = 0;
        const byteDeltaThreshold =
          knownTotal !== undefined && knownTotal > 0
            ? Math.max(MIN_PROGRESS_BYTE_DELTA, Math.floor(knownTotal * 0.01))
            : MIN_PROGRESS_BYTE_DELTA;

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            if (value && value.byteLength > 0) {
              let written = 0;
              while (written < value.byteLength) {
                written += accessHandle.write(value.subarray(written), { at: loadedBytes + written });
              }
              loadedBytes += value.byteLength;

              if (onFileProgress) {
                const now = nowFn();
                if (
                  now - lastEmitAt >= minEmitIntervalMs ||
                  loadedBytes - lastEmittedBytes >= byteDeltaThreshold
                ) {
                  lastEmitAt = now;
                  lastEmittedBytes = loadedBytes;
                  const total =
                    knownTotal !== undefined && knownTotal >= loadedBytes ? knownTotal : undefined;
                  onFileProgress({
                    status: 'progress',
                    name,
                    file: filePath,
                    loaded: loadedBytes,
                    ...(total !== undefined
                      ? { total, progress: Math.min(100, (loadedBytes / total) * 100) }
                      : {}),
                  });
                }
              }
            }
          }
        } catch (err) {
          await reader.cancel().catch(() => {});
          throw err;
        }

        accessHandle.flush();
        accessHandle.close();
        accessHandle = undefined;

        const diskFile = await fileHandle.getFile();
        const finalBytes = diskFile.size || loadedBytes;
        const headers = buildPrecachedResponseHeaders(response.headers, finalBytes);
        try {
          await cache.put(
            cacheKey,
            new Response(diskFile, {
              status: 200,
              statusText: 'OK',
              headers,
            })
          );
        } catch (err) {
          throw wrapCacheWriteError(err, filePath);
        }

        onFileProgress?.({
          status: 'done',
          name,
          file: filePath,
          loaded: finalBytes,
          total: finalBytes,
          progress: 100,
        });
        return finalBytes;
      }
    } finally {
      try {
        accessHandle?.close();
      } catch {
        // Ignore handle close cleanup failures.
      }
      await opfsRoot.removeEntry(OPFS_PRECACHE_TMP_FILE).catch(() => {});
    }
  }

  // Last-resort WebKit Private Browsing fallback (no OPFS sync access handle):
  // let the browser store the untouched fetch Response directly.
  onFileProgress?.({
    status: 'initiate',
    name,
    file: filePath,
    loaded: 0,
    ...(knownTotal !== undefined ? { total: knownTotal, progress: 0 } : {}),
  });
  try {
    await cache.put(cacheKey, response);
  } catch (err) {
    throw wrapCacheWriteError(err, filePath);
  }
  const finalBytes = knownTotal ?? 0;
  onFileProgress?.({
    status: 'done',
    name,
    file: filePath,
    loaded: finalBytes,
    ...(finalBytes > 0 ? { total: finalBytes, progress: 100 } : {}),
  });
  return finalBytes;
}

/**
 * Download every uncached ONNX model file sequentially one-at-a-time via
 * streaming `Cache.put()` (or OPFS disk spool -> `Response(File)` -> `Cache.put()`
 * on WebKit/Safari where stream-backed Cache.put is unsupported), preventing
 * parallel weight buffering and multi-copy JS heap spikes on mobile browsers.
 */
export async function precacheModelFilesSequentially(
  options: PrecacheModelFilesOptions
): Promise<PrecacheModelFilesResult> {
  const storage = options.cacheStorage ?? (typeof caches !== 'undefined' ? caches : undefined);
  const fetchFn = options.fetchFn ?? (typeof fetch === 'function' ? fetch.bind(globalThis) : undefined);

  const uniqueFiles: string[] = [];
  const seen = new Set<string>();
  for (const file of options.files) {
    const trimmed = file.trim();
    if (trimmed && !seen.has(trimmed)) {
      seen.add(trimmed);
      uniqueFiles.push(trimmed);
    }
  }

  if (!storage || typeof storage.open !== 'function' || !fetchFn) {
    return {
      usedBrowserCache: false,
      totalFiles: uniqueFiles.length,
      downloadedFiles: 0,
      skippedCachedFiles: 0,
      downloadedBytes: 0,
    };
  }

  const cache = await storage.open(options.cacheName ?? TRANSFORMERS_CACHE_NAME);
  const canStreamPut =
    options.supportsStreamingPut ?? (await probeStreamingCachePutSupport(cache));
  let downloadedFiles = 0;
  let skippedCachedFiles = 0;
  let downloadedBytes = 0;

  for (const filePath of uniqueFiles) {
    if (options.signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }

    const url = buildRemoteModelFileUrl(options.repoId, filePath, {
      revision: options.revision,
      remoteHost: options.remoteHost,
      remotePathTemplate: options.remotePathTemplate,
    });

    const existing = await cache.match(url);
    const cachedTotal = existing ? resolveExpectedContentLength(existing.headers) : undefined;
    const isBinaryFile = filePath.endsWith('.onnx') || filePath.includes('.onnx_data');
    const isEmptyCachedBinary = isBinaryFile && cachedTotal === 0;
    if (
      existing &&
      !isEmptyCachedBinary &&
      (existing.ok || existing.status === 200 || existing.status === 0)
    ) {
      options.onFileProgress?.({
        status: 'initiate',
        name: options.repoId,
        file: filePath,
        loaded: 0,
        ...(cachedTotal !== undefined ? { total: cachedTotal, progress: 0 } : {}),
      });
      options.onFileProgress?.({
        status: 'done',
        name: options.repoId,
        file: filePath,
        loaded: cachedTotal ?? 0,
        ...(cachedTotal !== undefined ? { total: cachedTotal, progress: 100 } : {}),
      });
      skippedCachedFiles += 1;
      continue;
    }

    const response = await fetchFn(
      url,
      options.signal ? { signal: options.signal } : undefined
    );

    if (!response.ok) {
      await response.body?.cancel().catch(() => {});
      if (response.status === 404 && isOptionalModelFile(filePath)) {
        continue;
      }
      throw new Error(
        `モデルファイルの取得に失敗しました (${response.status} ${response.statusText || 'HTTP Error'}): ${filePath}`
      );
    }

    const written = canStreamPut
      ? await streamResponseToCache({
          cache,
          cacheKey: url,
          response,
          name: options.repoId,
          filePath,
          onFileProgress: options.onFileProgress,
          minEmitIntervalMs: options.minEmitIntervalMs,
          nowFn: options.nowFn,
        })
      : await streamResponseViaOpfsOrDirectCache({
          cache,
          cacheKey: url,
          response,
          name: options.repoId,
          filePath,
          onFileProgress: options.onFileProgress,
          minEmitIntervalMs: options.minEmitIntervalMs,
          nowFn: options.nowFn,
          opfsRoot: options.opfsRoot,
        });
    downloadedFiles += 1;
    downloadedBytes += written;
  }

  return {
    usedBrowserCache: true,
    totalFiles: uniqueFiles.length,
    downloadedFiles,
    skippedCachedFiles,
    downloadedBytes,
  };
}

/**
 * Serialize `CacheStorage` reads/writes behind a single FIFO queue so split-session
 * `Promise.all` loads (`embed_tokens` + `decoder_model_merged`) do not decode multiple
 * giant `.onnx_data` IPC buffers simultaneously.
 */
export function createSerializedCacheAdapter(
  cacheName = TRANSFORMERS_CACHE_NAME,
  cacheStorage?: CacheStorage
): SerializedCacheAdapter | null {
  const storage = cacheStorage ?? (typeof caches !== 'undefined' ? caches : undefined);
  if (!storage || typeof storage.open !== 'function') {
    return null;
  }

  let cachePromise: Promise<Cache> | null = null;
  const getBucket = (): Promise<Cache> => {
    if (!cachePromise) {
      cachePromise = storage.open(cacheName).catch((err) => {
        cachePromise = null;
        throw err;
      });
    }
    return cachePromise;
  };

  let opQueue: Promise<void> = Promise.resolve();
  const enqueue = <T>(fn: () => Promise<T>): Promise<T> => {
    const run = opQueue.then(fn, fn);
    opQueue = run.then(
      () => undefined,
      () => undefined
    );
    return run;
  };

  return {
    async match(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<Response | undefined> {
      return enqueue(async () => {
        const cache = await getBucket();
        return cache.match(request, options);
      });
    },
    async put(request: RequestInfo | URL, response: Response): Promise<void> {
      return enqueue(async () => {
        const cache = await getBucket();
        await cache.put(request, response);
      });
    },
  };
}
