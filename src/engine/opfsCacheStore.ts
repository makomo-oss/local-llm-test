import { extractModelFileFromCacheUrl } from './cacheManager.ts';

export const OPFS_MODEL_CACHE_DIR = 'local-llm-cache';
export const OPFS_PERSISTENT_THRESHOLD_BYTES = 4 * 1024 * 1024; // 4 MiB

export interface OpfsSyncAccessHandleLike {
  write(buffer: ArrayBufferView | ArrayBuffer, options?: { at?: number }): number;
  read?(buffer: ArrayBufferView | ArrayBuffer, options?: { at?: number }): number;
  getSize?(): number;
  flush(): void;
  close(): void;
  truncate?(newSize: number): void;
}

export interface OpfsWritableStreamLike {
  write(data: Uint8Array | ArrayBuffer): Promise<void>;
  close(): Promise<void>;
  abort?(reason?: unknown): Promise<void>;
}

export interface OpfsFileHandleLike {
  createSyncAccessHandle?: () => Promise<OpfsSyncAccessHandleLike>;
  createWritable?: () => Promise<OpfsWritableStreamLike>;
  getFile: () => Promise<File | Blob>;
}

export interface OpfsDirectoryLike {
  getFileHandle(name: string, options?: { create?: boolean }): Promise<OpfsFileHandleLike>;
  getDirectoryHandle?(name: string, options?: { create?: boolean }): Promise<OpfsDirectoryLike>;
  removeEntry(name: string, options?: { recursive?: boolean }): Promise<void>;
  entries?: () => AsyncIterable<[string, { kind: 'file' | 'directory'; name: string }]>;
  values?: () => AsyncIterable<{ kind: 'file' | 'directory'; name: string }>;
}

export interface OpfsStoredEntryMeta {
  url: string;
  byteLength: number;
  headers: Array<[string, string]>;
  savedAt: number;
}

export interface OpfsPartialEntryMeta {
  url: string;
  bytes: number;
  totalBytes?: number;
  etag?: string;
  headers: Array<[string, string]>;
  updatedAt: number;
}

export interface OpfsResumableEntry {
  byteOffset: number;
  totalBytes?: number;
  etag?: string;
}

export interface ReadStreamChunksOptions {
  signal?: AbortSignal;
  stagingBytes?: number;
  yieldEveryBytes?: number;
  onChunk: (chunk: Uint8Array) => Promise<void> | void;
}

export interface OpfsInspectedEntry {
  url: string;
  file: string;
  bytes: number;
}

export interface OpfsClearResult {
  clearedBytes: number;
  deletedEntries: number;
}

export const STREAM_READ_BUFFER_BYTES = 512 * 1024; // 512 KiB reusable staging buffer
const DEFAULT_STREAM_YIELD_INTERVAL_BYTES = 16 * 1024 * 1024;
const META_SUFFIX = '.meta.json';
const PARTIAL_SUFFIX = '.partial.json';
const BIN_SUFFIX = '.bin';

export function isTransientStreamError(err: unknown): boolean {
  return Boolean(
    err &&
      typeof err === 'object' &&
      (err as { isTransientStreamError?: unknown }).isTransientStreamError === true
  );
}

function markTransientStreamError(err: unknown): unknown {
  if (err && typeof err === 'object' && !(err instanceof DOMException && err.name === 'AbortError')) {
    try {
      Object.defineProperty(err, 'isTransientStreamError', {
        value: true,
        configurable: true,
        enumerable: false,
        writable: true,
      });
    } catch {
      // Ignore frozen error objects
    }
  }
  return err;
}

/**
 * Resolve OPFS root directory safely when available in the current execution context.
 */
export async function resolveOpfsRootDirectory(
  storageManager: Pick<StorageManager, 'getDirectory'> | undefined =
    typeof navigator !== 'undefined' ? navigator.storage : undefined
): Promise<OpfsDirectoryLike | null> {
  if (!storageManager || typeof storageManager.getDirectory !== 'function') {
    return null;
  }
  try {
    return (await storageManager.getDirectory()) as unknown as OpfsDirectoryLike;
  } catch {
    return null;
  }
}

export function isPersistentOpfsCapable(
  opfsRoot: OpfsDirectoryLike | null | undefined
): opfsRoot is OpfsDirectoryLike & Required<Pick<OpfsDirectoryLike, 'getDirectoryHandle'>> {
  return Boolean(opfsRoot && typeof opfsRoot.getDirectoryHandle === 'function');
}

/**
 * Determine whether an artifact should be stored directly in persistent OPFS rather than
 * CacheStorage (avoids Chromium incognito/mobile >200 MiB UnknownError and WebKit IPC copies).
 */
export function shouldPreferPersistentOpfs(
  fileOrUrl: string,
  expectedBytes?: number
): boolean {
  const normalized = fileOrUrl.split('#')[0]?.split('?')[0] ?? fileOrUrl;
  if (/\.onnx(?:_data(?:_\d+)?)?$/i.test(normalized)) {
    return true;
  }
  if (typeof expectedBytes === 'number' && expectedBytes >= OPFS_PERSISTENT_THRESHOLD_BYTES) {
    return true;
  }
  return false;
}

/**
 * FNV-1a 64-bit hex + sanitized basename (bounded <= 96 chars) safe on all filesystems.
 */
export function buildOpfsEntryStem(url: string): string {
  let h1 = 0xcbf29ce4n;
  let h2 = 0x84222325n;
  const prime = 0x100000001b3n;
  const mask64 = 0xffffffffffffffffn;
  let hash = (h1 << 32n) | h2;
  for (let i = 0; i < url.length; i += 1) {
    hash ^= BigInt(url.charCodeAt(i));
    hash = (hash * prime) & mask64;
  }
  const hex = hash.toString(16).padStart(16, '0');
  const rawBase = (url.split('?')[0]?.split('/').pop() ?? 'file').replace(/[^a-zA-Z0-9._-]/g, '_');
  const shortBase = rawBase.slice(-48) || 'artifact';
  return `${hex}_${shortBase}`;
}

async function getPersistentCacheDir(
  opfsRoot: OpfsDirectoryLike,
  create: boolean
): Promise<OpfsDirectoryLike | null> {
  if (typeof opfsRoot.getDirectoryHandle !== 'function') {
    return null;
  }
  try {
    return await opfsRoot.getDirectoryHandle(OPFS_MODEL_CACHE_DIR, { create });
  } catch {
    return null;
  }
}

async function writeSmallTextFile(
  dir: OpfsDirectoryLike,
  fileName: string,
  text: string
): Promise<void> {
  const bytes = new TextEncoder().encode(text);
  const handle = await dir.getFileHandle(fileName, { create: true });
  if (typeof handle.createSyncAccessHandle === 'function') {
    let access: OpfsSyncAccessHandleLike | null = null;
    try {
      access = await handle.createSyncAccessHandle();
      if (typeof access.write !== 'function') {
        throw new Error(`OPFS SyncAccessHandle does not support write() for ${fileName}`);
      }
      access.truncate?.(0);
      let written = 0;
      while (written < bytes.byteLength) {
        const n = access.write(bytes.subarray(written), { at: written });
        if (n <= 0) {
          throw new Error(`OPFS write returned 0 bytes for ${fileName}`);
        }
        written += n;
      }
      access.flush();
      return;
    } catch (syncErr) {
      if (isQuotaOrAbortOpfsError(syncErr) || typeof handle.createWritable !== 'function') {
        throw syncErr;
      }
      // Fall through to WritableStream when SyncAccessHandle rejects (e.g. NotSupportedError).
    } finally {
      if (access) {
        try {
          access.close();
        } catch {
          // Ignore close cleanup failure
        }
      }
    }
  }
  if (typeof handle.createWritable === 'function') {
    const writable = await handle.createWritable();
    try {
      await writable.write(bytes);
      await writable.close();
    } catch (err) {
      await writable.abort?.(err).catch(() => {});
      throw err;
    }
    return;
  }
  throw new Error('OPFS file handle does not support SyncAccessHandle or WritableStream.');
}

async function removeEntryQuietly(dir: OpfsDirectoryLike, name: string): Promise<void> {
  await dir.removeEntry(name).catch(() => {});
}

async function readOpfsMetaFile(
  dir: OpfsDirectoryLike,
  metaFileName: string
): Promise<OpfsStoredEntryMeta | null> {
  try {
    const handle = await dir.getFileHandle(metaFileName, { create: false });
    const file = await handle.getFile();
    const raw = await file.text();
    const parsed = JSON.parse(raw) as Partial<OpfsStoredEntryMeta>;
    if (
      typeof parsed?.url !== 'string' ||
      typeof parsed.byteLength !== 'number' ||
      !Number.isFinite(parsed.byteLength) ||
      parsed.byteLength <= 0 ||
      !Array.isArray(parsed.headers)
    ) {
      return null;
    }
    return {
      url: parsed.url,
      byteLength: parsed.byteLength,
      headers: parsed.headers.filter(
        (pair): pair is [string, string] =>
          Array.isArray(pair) && typeof pair[0] === 'string' && typeof pair[1] === 'string'
      ),
      savedAt: typeof parsed.savedAt === 'number' ? parsed.savedAt : Date.now(),
    };
  } catch {
    return null;
  }
}

async function readOpfsPartialMetaFile(
  dir: OpfsDirectoryLike,
  partialFileName: string
): Promise<OpfsPartialEntryMeta | null> {
  try {
    const handle = await dir.getFileHandle(partialFileName, { create: false });
    const file = await handle.getFile();
    const raw = await file.text();
    const parsed = JSON.parse(raw) as Partial<OpfsPartialEntryMeta>;
    if (
      typeof parsed?.url !== 'string' ||
      typeof parsed.bytes !== 'number' ||
      !Number.isFinite(parsed.bytes) ||
      parsed.bytes <= 0 ||
      !Array.isArray(parsed.headers)
    ) {
      return null;
    }
    const totalBytes =
      typeof parsed.totalBytes === 'number' &&
      Number.isFinite(parsed.totalBytes) &&
      parsed.totalBytes > 0
        ? parsed.totalBytes
        : undefined;
    return {
      url: parsed.url,
      bytes: Math.floor(parsed.bytes),
      ...(totalBytes !== undefined ? { totalBytes } : {}),
      ...(typeof parsed.etag === 'string' && parsed.etag ? { etag: parsed.etag } : {}),
      headers: parsed.headers.filter(
        (pair): pair is [string, string] =>
          Array.isArray(pair) && typeof pair[0] === 'string' && typeof pair[1] === 'string'
      ),
      updatedAt: typeof parsed.updatedAt === 'number' ? parsed.updatedAt : Date.now(),
    };
  } catch {
    return null;
  }
}

function isDetachedArrayBuffer(buffer: ArrayBuffer | null | undefined): boolean {
  if (!buffer || buffer.byteLength === 0) {
    return true;
  }
  return Boolean((buffer as { detached?: boolean }).detached);
}

/**
 * Consume a `ReadableStream<Uint8Array>` using a BYOB reader backed by a single reusable
 * `ArrayBuffer` (falling back cleanly to default `getReader()` on non-byte streams) and yield
 * periodically so mobile Workers reclaim external ArrayBuffers without GC sawtooth OOMs.
 */
export async function readStreamChunks(
  stream: ReadableStream<Uint8Array>,
  options: ReadStreamChunksOptions
): Promise<void> {
  const bufferBytes = Math.max(4096, Math.floor(options.stagingBytes ?? STREAM_READ_BUFFER_BYTES));
  const yieldEveryBytes = Math.max(
    bufferBytes,
    Math.floor(options.yieldEveryBytes ?? DEFAULT_STREAM_YIELD_INTERVAL_BYTES)
  );
  const { signal, onChunk } = options;

  let byobReader: ReadableStreamBYOBReader | null = null;
  try {
    byobReader = stream.getReader({ mode: 'byob' });
  } catch {
    byobReader = null;
  }

  let bytesSinceYield = 0;

  if (byobReader) {
    let staging = new ArrayBuffer(bufferBytes);
    try {
      while (true) {
        if (signal?.aborted) {
          throw new DOMException('Aborted', 'AbortError');
        }
        if (isDetachedArrayBuffer(staging)) {
          staging = new ArrayBuffer(bufferBytes);
        }
        const view = new Uint8Array(staging);
        let readResult: ReadableStreamReadResult<Uint8Array>;
        try {
          readResult = await byobReader.read(view);
        } catch (readErr) {
          throw markTransientStreamError(readErr);
        }
        const { done, value } = readResult;
        if (done) {
          break;
        }
        if (value && value.byteLength > 0) {
          await onChunk(value);
          bytesSinceYield += value.byteLength;
          if (bytesSinceYield >= yieldEveryBytes) {
            bytesSinceYield = 0;
            await new Promise<void>((resolve) => setTimeout(resolve, 0));
          }
        }
        const returnedBuf = value?.buffer as ArrayBuffer | undefined;
        staging =
          returnedBuf && !isDetachedArrayBuffer(returnedBuf) && returnedBuf.byteLength >= 4096
            ? returnedBuf
            : new ArrayBuffer(0);
      }
    } catch (err) {
      await byobReader.cancel(err).catch(() => {});
      throw err;
    } finally {
      staging = new ArrayBuffer(0);
      try {
        byobReader.releaseLock();
      } catch {
        // Ignore release errors after cancel
      }
    }
    return;
  }

  const reader = stream.getReader();
  try {
    while (true) {
      if (signal?.aborted) {
        throw new DOMException('Aborted', 'AbortError');
      }
      let readResult: ReadableStreamReadResult<Uint8Array>;
      try {
        readResult = await reader.read();
      } catch (readErr) {
        throw markTransientStreamError(readErr);
      }
      const { done, value } = readResult;
      if (done) {
        break;
      }
      if (value && value.byteLength > 0) {
        await onChunk(value);
        bytesSinceYield += value.byteLength;
        if (bytesSinceYield >= yieldEveryBytes) {
          bytesSinceYield = 0;
          await new Promise<void>((resolve) => setTimeout(resolve, 0));
        }
      }
    }
  } catch (err) {
    await reader.cancel(err).catch(() => {});
    throw err;
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // Ignore release errors after cancel
    }
  }
}

/**
 * Check whether an incomplete persistent OPFS `.bin` + `.partial.json` exists and can be
 * resumed via HTTP `Range: bytes=<byteOffset>-`. Cleans up corrupt/mismatched partials.
 */
export async function readOpfsResumableEntry(
  opfsRoot: OpfsDirectoryLike | null | undefined,
  url: string
): Promise<OpfsResumableEntry | null> {
  if (!isPersistentOpfsCapable(opfsRoot)) {
    return null;
  }
  const dir = await getPersistentCacheDir(opfsRoot, false);
  if (!dir) {
    return null;
  }

  const stem = buildOpfsEntryStem(url);
  const partialName = `${stem}${PARTIAL_SUFFIX}`;
  const binName = `${stem}${BIN_SUFFIX}`;
  const partialMeta = await readOpfsPartialMetaFile(dir, partialName);
  if (!partialMeta || partialMeta.url !== url) {
    return null;
  }

  try {
    const fileHandle = await dir.getFileHandle(binName, { create: false });
    const binFile = await fileHandle.getFile();
    if (
      binFile &&
      binFile.size > 0 &&
      binFile.size === partialMeta.bytes &&
      (!partialMeta.totalBytes || binFile.size < partialMeta.totalBytes)
    ) {
      return {
        byteOffset: binFile.size,
        ...(partialMeta.totalBytes !== undefined ? { totalBytes: partialMeta.totalBytes } : {}),
        ...(partialMeta.etag ? { etag: partialMeta.etag } : {}),
      };
    }
  } catch {
    // Missing or unreadable partial binary
  }

  await removeEntryQuietly(dir, partialName);
  await removeEntryQuietly(dir, binName);
  return null;
}

/**
 * Remove any incomplete partial download metadata & binary for `url`.
 */
export async function clearOpfsResumableEntry(
  opfsRoot: OpfsDirectoryLike | null | undefined,
  url: string
): Promise<void> {
  if (!isPersistentOpfsCapable(opfsRoot)) {
    return;
  }
  const dir = await getPersistentCacheDir(opfsRoot, false);
  if (!dir) {
    return;
  }
  const stem = buildOpfsEntryStem(url);
  await removeEntryQuietly(dir, `${stem}${PARTIAL_SUFFIX}`);
  await removeEntryQuietly(dir, `${stem}${BIN_SUFFIX}`);
}

export interface OpfsValidatedCachedFile {
  dir: OpfsDirectoryLike;
  meta: OpfsStoredEntryMeta;
  fileHandle: OpfsFileHandleLike;
  binFile: File | Blob;
}

export interface OpfsSyncReadHandle {
  byteLength: number;
  readAt(dest: Uint8Array, offset: number): number;
  close(): void;
}

interface WorkerFileReaderSyncLike {
  readAsArrayBuffer(blob: Blob): ArrayBuffer;
}

function getFileReaderSyncConstructor(): (new () => WorkerFileReaderSyncLike) | undefined {
  const scope = globalThis as { FileReaderSync?: new () => WorkerFileReaderSyncLike };
  return typeof scope.FileReaderSync === 'function' ? scope.FileReaderSync : undefined;
}

function isQuotaOrAbortOpfsError(err: unknown): boolean {
  if (!err || typeof err !== 'object') {
    return false;
  }
  const maybeErr = err as { name?: unknown; code?: unknown };
  return (
    maybeErr.name === 'QuotaExceededError' ||
    maybeErr.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
    maybeErr.name === 'AbortError' ||
    maybeErr.code === 'STORAGE_QUOTA_EXCEEDED' ||
    maybeErr.code === 22
  );
}

/**
 * Create a synchronous `OpfsSyncReadHandle` backed by `FileReaderSync.readAsArrayBuffer(blob.slice(...))`
 * inside Dedicated Web Workers. Avoids buffering multi-GiB files in RAM when `createSyncAccessHandle`
 * is unavailable or throws (`NotSupportedError` / lock contention on Safari / WebViews).
 */
export function createBlobSyncReadHandle(
  blob: Pick<Blob, 'size' | 'slice'>,
  url: string,
  expectedByteLength?: number
): OpfsSyncReadHandle | undefined {
  const ReaderSyncCtor = getFileReaderSyncConstructor();
  if (!ReaderSyncCtor || typeof blob.slice !== 'function') {
    return undefined;
  }
  const totalSize = blob.size;
  if (!Number.isFinite(totalSize) || totalSize <= 0) {
    return undefined;
  }
  if (
    typeof expectedByteLength === 'number' &&
    Number.isFinite(expectedByteLength) &&
    totalSize !== expectedByteLength
  ) {
    return undefined;
  }

  const reader = new ReaderSyncCtor();
  let closed = false;

  return {
    byteLength: totalSize,
    readAt(dest: Uint8Array, offset: number): number {
      if (closed) {
        throw new Error(`OPFS FileReaderSync handle is already closed for ${url}`);
      }
      const safeOffset = Number.isFinite(offset) ? Math.max(0, Math.trunc(offset)) : 0;
      if (dest.byteLength <= 0 || safeOffset >= totalSize) {
        return 0;
      }
      const end = Math.min(totalSize, safeOffset + dest.byteLength);
      if (end <= safeOffset) {
        return 0;
      }
      const ab = reader.readAsArrayBuffer(blob.slice(safeOffset, end));
      const chunk = new Uint8Array(ab);
      const copyLen = Math.min(dest.byteLength, chunk.byteLength);
      if (copyLen <= 0) {
        return 0;
      }
      dest.set(copyLen === chunk.byteLength ? chunk : chunk.subarray(0, copyLen), 0);
      return copyLen;
    },
    close(): void {
      closed = true;
    },
  };
}

async function resolveValidatedOpfsEntry(
  opfsRoot: OpfsDirectoryLike | null | undefined,
  url: string
): Promise<OpfsValidatedCachedFile | undefined> {
  if (!isPersistentOpfsCapable(opfsRoot)) {
    return undefined;
  }
  const dir = await getPersistentCacheDir(opfsRoot, false);
  if (!dir) {
    return undefined;
  }

  const stem = buildOpfsEntryStem(url);
  const metaName = `${stem}${META_SUFFIX}`;
  const binName = `${stem}${BIN_SUFFIX}`;
  const meta = await readOpfsMetaFile(dir, metaName);
  if (!meta || meta.url !== url) {
    return undefined;
  }

  try {
    const fileHandle = await dir.getFileHandle(binName, { create: false });
    const binFile = await fileHandle.getFile();
    if (!binFile || binFile.size <= 0 || binFile.size !== meta.byteLength) {
      await removeEntryQuietly(dir, metaName);
      await removeEntryQuietly(dir, binName);
      return undefined;
    }
    return { dir, meta, fileHandle, binFile };
  } catch {
    await removeEntryQuietly(dir, metaName);
    return undefined;
  }
}

/**
 * Open a validated persistent OPFS entry without wrapping it in a `Response`.
 * Useful for on-demand byte-range `Blob.slice` reads.
 */
export async function openOpfsCachedBlob(
  opfsRoot: OpfsDirectoryLike | null | undefined,
  url: string
): Promise<Blob | undefined> {
  const entry = await resolveValidatedOpfsEntry(opfsRoot, url);
  return entry?.binFile;
}

/**
 * Open a synchronous OPFS `FileSystemSyncAccessHandle` reader inside Dedicated Workers.
 * Returns `undefined` when unavailable (e.g. non-worker contexts, missing entry, or no `.read`).
 */
export async function openOpfsSyncReadHandle(
  opfsRoot: OpfsDirectoryLike | null | undefined,
  url: string
): Promise<OpfsSyncReadHandle | undefined> {
  const entry = await resolveValidatedOpfsEntry(opfsRoot, url);
  if (!entry) {
    return undefined;
  }

  if (typeof entry.fileHandle.createSyncAccessHandle === 'function') {
    let accessHandle: OpfsSyncAccessHandleLike | null = null;
    try {
      accessHandle = await entry.fileHandle.createSyncAccessHandle();
      const totalSize =
        typeof accessHandle.getSize === 'function' ? accessHandle.getSize() : entry.meta.byteLength;
      if (
        typeof accessHandle.read === 'function' &&
        Number.isFinite(totalSize) &&
        totalSize === entry.meta.byteLength
      ) {
        const readFn = accessHandle.read.bind(accessHandle);
        let closed = false;
        return {
          byteLength: totalSize,
          readAt(dest: Uint8Array, offset: number): number {
            if (closed) {
              throw new Error(`OPFS SyncAccessHandle is already closed for ${url}`);
            }
            return readFn(dest, { at: offset });
          },
          close(): void {
            if (closed || !accessHandle) {
              return;
            }
            closed = true;
            try {
              accessHandle.close();
            } catch {
              // Ignore close errors during teardown
            }
          },
        };
      }
      accessHandle.close();
      accessHandle = null;
    } catch {
      if (accessHandle) {
        try {
          accessHandle.close();
        } catch {
          // Ignore close errors
        }
      }
    }
  }

  // Fallback for Safari/WebViews or locked SyncAccessHandles inside Web Workers:
  // stream slices synchronously from OPFS File without buffering the full binary in RAM.
  if (typeof entry.fileHandle.getFile === 'function') {
    try {
      const file = await entry.fileHandle.getFile();
      const blobHandle = createBlobSyncReadHandle(file, url, entry.meta.byteLength);
      if (blobHandle) {
        return blobHandle;
      }
    } catch {
      // Ignore and let callers fall back gracefully.
    }
  }

  return undefined;
}

/**
 * Look up a validated OPFS entry and return a lazy Blob/File-backed `Response`.
 */
export async function readOpfsCachedResponse(
  opfsRoot: OpfsDirectoryLike | null | undefined,
  url: string
): Promise<Response | undefined> {
  const entry = await resolveValidatedOpfsEntry(opfsRoot, url);
  if (!entry) {
    return undefined;
  }
  const headers = new Headers(entry.meta.headers);
  if (!headers.has('content-length')) {
    headers.set('content-length', String(entry.meta.byteLength));
  }
  return new Response(entry.binFile, {
    status: 200,
    statusText: 'OK',
    headers,
  });
}

/**
 * Stream a ReadableStream directly into a persistent OPFS binary file with zero heap growth,
 * verifying exact byte count and committing metadata last so crashes never leave corrupt hits.
 * Supports optional byte-offset resumption and `.partial.json` persistence on network failure.
 */
export async function writeStreamToOpfsStore(options: {
  opfsRoot: OpfsDirectoryLike;
  url: string;
  stream: ReadableStream<Uint8Array>;
  headers: Headers;
  expectedTotalBytes?: number;
  resumeFromBytes?: number;
  keepPartialOnFailure?: boolean;
  etag?: string;
  signal?: AbortSignal;
  onProgress?: (loadedBytes: number, totalBytes?: number) => void;
}): Promise<number> {
  const {
    opfsRoot,
    url,
    stream,
    headers,
    expectedTotalBytes,
    resumeFromBytes = 0,
    keepPartialOnFailure = false,
    etag,
    signal,
    onProgress,
  } = options;
  const dir = await getPersistentCacheDir(opfsRoot, true);
  if (!dir) {
    throw new Error('Persistent OPFS directory is unavailable.');
  }

  const stem = buildOpfsEntryStem(url);
  const binName = `${stem}${BIN_SUFFIX}`;
  const metaName = `${stem}${META_SUFFIX}`;
  const partialName = `${stem}${PARTIAL_SUFFIX}`;
  // Invalidate any stale meta before touching .bin so an interrupted download is never treated as complete.
  await removeEntryQuietly(dir, metaName);

  const requestedResume =
    Number.isFinite(resumeFromBytes) && resumeFromBytes > 0 ? Math.floor(resumeFromBytes) : 0;

  const fileHandle = await dir.getFileHandle(binName, { create: true });
  let accessHandle: OpfsSyncAccessHandleLike | null = null;
  let writable: OpfsWritableStreamLike | null = null;
  let usedSyncAccess = false;
  let loadedBytes = 0;

  try {
    if (typeof fileHandle.createSyncAccessHandle === 'function') {
      try {
        accessHandle = await fileHandle.createSyncAccessHandle();
        if (typeof accessHandle.write !== 'function') {
          throw new Error('OPFS SyncAccessHandle does not support write().');
        }
        usedSyncAccess = true;
        if (requestedResume > 0) {
          const existingSize =
            typeof accessHandle.getSize === 'function'
              ? accessHandle.getSize()
              : (await fileHandle.getFile()).size;
          if (existingSize !== requestedResume) {
            throw new Error(
              `OPFS resume offset mismatch for ${url}: disk=${existingSize}, requested=${requestedResume}.`
            );
          }
          loadedBytes = requestedResume;
        } else {
          accessHandle.truncate?.(0);
        }
      } catch (syncErr) {
        if (accessHandle) {
          try {
            accessHandle.close();
          } catch {
            // Ignore close cleanup failure
          }
          accessHandle = null;
        }
        if (
          requestedResume === 0 &&
          !isQuotaOrAbortOpfsError(syncErr) &&
          typeof fileHandle.createWritable === 'function'
        ) {
          writable = await fileHandle.createWritable();
        } else {
          throw syncErr;
        }
      }
    } else if (requestedResume === 0 && typeof fileHandle.createWritable === 'function') {
      writable = await fileHandle.createWritable();
    } else if (requestedResume > 0) {
      throw new Error(`OPFS resume requires SyncAccessHandle for ${url}.`);
    } else {
      throw new Error('OPFS file handle does not support SyncAccessHandle or WritableStream.');
    }

    await readStreamChunks(stream, {
      signal,
      onChunk: async (value) => {
        if (accessHandle) {
          let offset = 0;
          while (offset < value.byteLength) {
            const chunk = offset === 0 ? value : value.subarray(offset);
            const written = accessHandle.write(chunk, { at: loadedBytes });
            if (written <= 0) {
              throw new Error(`OPFS write failed at offset ${loadedBytes} for ${url}`);
            }
            loadedBytes += written;
            offset += written;
          }
        } else if (writable) {
          await writable.write(value);
          loadedBytes += value.byteLength;
        }

        onProgress?.(loadedBytes, expectedTotalBytes);
      },
    });

    if (accessHandle) {
      accessHandle.flush();
      accessHandle.close();
      accessHandle = null;
    } else if (writable) {
      await writable.close();
      writable = null;
    }

    const diskFile = await fileHandle.getFile();
    const finalBytes = diskFile.size;
    if (finalBytes <= 0 || loadedBytes <= 0) {
      throw new Error(`OPFS write produced an empty binary (${finalBytes} bytes) for ${url}`);
    }
    if (
      typeof expectedTotalBytes === 'number' &&
      expectedTotalBytes > 0 &&
      finalBytes !== expectedTotalBytes
    ) {
      const sizeErr = new Error(
        `OPFS write size mismatch for ${url}: wrote ${finalBytes} bytes, expected ${expectedTotalBytes}.`
      );
      if (keepPartialOnFailure && usedSyncAccess && finalBytes < expectedTotalBytes) {
        // Premature clean EOF during network transfer can safely resume from finalBytes.
        throw markTransientStreamError(sizeErr);
      }
      throw sizeErr;
    }

    await removeEntryQuietly(dir, partialName);
    const finalHeaders = new Headers(headers);
    finalHeaders.delete('content-encoding');
    finalHeaders.delete('content-range');
    finalHeaders.set('content-length', String(finalBytes));
    const meta: OpfsStoredEntryMeta = {
      url,
      byteLength: finalBytes,
      headers: [...finalHeaders.entries()],
      savedAt: Date.now(),
    };
    await writeSmallTextFile(dir, metaName, JSON.stringify(meta));
    return finalBytes;
  } catch (err) {
    const isAbort =
      signal?.aborted || (err instanceof DOMException && err.name === 'AbortError');
    const isTransient = isTransientStreamError(err);
    const isIntegrityErr =
      !isAbort &&
      !isTransient &&
      err instanceof Error &&
      (err.message.includes('size mismatch') ||
        err.message.includes('empty binary') ||
        err.message.includes('OPFS resume'));

    const isTruncationAfterClose =
      keepPartialOnFailure &&
      usedSyncAccess &&
      isTransient &&
      accessHandle === null &&
      loadedBytes > 0;

    if (
      keepPartialOnFailure &&
      (Boolean(accessHandle) || isTruncationAfterClose) &&
      loadedBytes > 0 &&
      !isAbort &&
      !isQuotaOrAbortOpfsError(err) &&
      !isIntegrityErr
    ) {
      try {
        const pendingHandle = accessHandle as OpfsSyncAccessHandleLike | null;
        if (pendingHandle) {
          pendingHandle.flush();
          pendingHandle.close();
          accessHandle = null;
        }
        const diskFile = await fileHandle.getFile();
        if (diskFile.size === loadedBytes) {
          const partialHeaders = new Headers(headers);
          partialHeaders.delete('content-encoding');
          partialHeaders.delete('content-range');
          const partialMeta: OpfsPartialEntryMeta = {
            url,
            bytes: loadedBytes,
            ...(typeof expectedTotalBytes === 'number' && expectedTotalBytes > 0
              ? { totalBytes: expectedTotalBytes }
              : {}),
            ...(etag ? { etag } : {}),
            headers: [...partialHeaders.entries()],
            updatedAt: Date.now(),
          };
          await writeSmallTextFile(dir, partialName, JSON.stringify(partialMeta));
          throw err;
        }
      } catch (saveErr) {
        if (saveErr === err) {
          throw err;
        }
        // Fall through to full cleanup if saving partial state itself failed
      }
    }

    const danglingHandle = accessHandle as OpfsSyncAccessHandleLike | null;
    if (danglingHandle) {
      try {
        danglingHandle.close();
      } catch {
        // Ignore close errors
      }
      accessHandle = null;
    }
    if (writable) {
      await writable.abort?.(err).catch(() => {});
    }
    await removeEntryQuietly(dir, metaName);
    await removeEntryQuietly(dir, partialName);
    await removeEntryQuietly(dir, binName);
    throw err;
  }
}

/**
 * Persist an existing `Response` (e.g. from `Cache.put` fallback) into OPFS.
 */
export async function writeResponseToOpfsStore(
  opfsRoot: OpfsDirectoryLike,
  url: string,
  response: Response
): Promise<number> {
  if (response.body) {
    const rawLen = Number(response.headers.get('content-length') ?? '0');
    return writeStreamToOpfsStore({
      opfsRoot,
      url,
      stream: response.body,
      headers: new Headers(response.headers),
      expectedTotalBytes: Number.isFinite(rawLen) && rawLen > 0 ? rawLen : undefined,
    });
  }
  const ab = await response.arrayBuffer();
  const singleChunkStream = new ReadableStream<Uint8Array>({
    start(controller) {
      if (ab.byteLength > 0) {
        controller.enqueue(new Uint8Array(ab));
      }
      controller.close();
    },
  });
  return writeStreamToOpfsStore({
    opfsRoot,
    url,
    stream: singleChunkStream,
    headers: new Headers(response.headers),
    expectedTotalBytes: ab.byteLength,
  });
}

async function listDirectoryEntryNames(dir: OpfsDirectoryLike): Promise<string[]> {
  const names: string[] = [];
  if (typeof dir.entries === 'function') {
    for await (const [name, handle] of dir.entries()) {
      if (handle?.kind === 'file' || typeof name === 'string') {
        names.push(name);
      }
    }
    return names;
  }
  if (typeof dir.values === 'function') {
    for await (const handle of dir.values()) {
      if (handle?.name) {
        names.push(handle.name);
      }
    }
  }
  return names;
}

/**
 * Inspect validated persistent OPFS cache entries for one repository (or all entries).
 */
export async function inspectOpfsModelEntries(
  opfsRoot: OpfsDirectoryLike | null | undefined,
  repoId?: string
): Promise<OpfsInspectedEntry[]> {
  if (!isPersistentOpfsCapable(opfsRoot)) {
    return [];
  }
  const dir = await getPersistentCacheDir(opfsRoot, false);
  if (!dir) {
    return [];
  }

  const entryNames = await listDirectoryEntryNames(dir);
  const nameSet = new Set(entryNames);
  const results: OpfsInspectedEntry[] = [];

  for (const name of entryNames) {
    if (!name.endsWith(META_SUFFIX)) {
      continue;
    }
    const stem = name.slice(0, -META_SUFFIX.length);
    const binName = `${stem}${BIN_SUFFIX}`;
    if (!nameSet.has(binName)) {
      continue;
    }
    const meta = await readOpfsMetaFile(dir, name);
    if (!meta) {
      continue;
    }
    const file = repoId ? extractModelFileFromCacheUrl(meta.url, repoId) : meta.url;
    if (repoId && !file) {
      continue;
    }
    try {
      const binHandle = await dir.getFileHandle(binName, { create: false });
      const binFile = await binHandle.getFile();
      if (binFile.size > 0 && binFile.size === meta.byteLength) {
        results.push({
          url: meta.url,
          file: file ?? meta.url,
          bytes: binFile.size,
        });
      }
    } catch {
      // Ignore unreadable files
    }
  }

  return results;
}

/**
 * Remove persistent OPFS model entries (for a single repo, or entire directory if repoId omitted).
 */
export async function clearOpfsModelEntries(
  opfsRoot: OpfsDirectoryLike | null | undefined,
  repoId?: string
): Promise<OpfsClearResult> {
  if (!isPersistentOpfsCapable(opfsRoot)) {
    return { clearedBytes: 0, deletedEntries: 0 };
  }

  let clearedBytes = 0;
  let deletedEntries = 0;

  if (!repoId) {
    // Also remove legacy temporary spool file if left behind by an earlier crash
    await removeEntryQuietly(opfsRoot, '__local_llm_precache_tmp__.bin');
    const dir = await getPersistentCacheDir(opfsRoot, false);
    if (!dir) {
      return { clearedBytes: 0, deletedEntries: 0 };
    }
    const names = await listDirectoryEntryNames(dir);
    for (const name of names) {
      if (name.endsWith(BIN_SUFFIX)) {
        try {
          const fh = await dir.getFileHandle(name, { create: false });
          const f = await fh.getFile();
          clearedBytes += f.size;
          deletedEntries += 1;
        } catch {
          // Ignore size inspection error
        }
      }
      await removeEntryQuietly(dir, name);
    }
    await opfsRoot.removeEntry(OPFS_MODEL_CACHE_DIR, { recursive: true }).catch(() => {});
    return { clearedBytes, deletedEntries };
  }

  const dir = await getPersistentCacheDir(opfsRoot, false);
  if (!dir) {
    return { clearedBytes: 0, deletedEntries: 0 };
  }

  const entryNames = await listDirectoryEntryNames(dir);
  const handledStems = new Set<string>();
  for (const name of entryNames) {
    if (!name.endsWith(META_SUFFIX)) {
      continue;
    }
    const meta = await readOpfsMetaFile(dir, name);
    if (!meta || extractModelFileFromCacheUrl(meta.url, repoId) === null) {
      continue;
    }
    const stem = name.slice(0, -META_SUFFIX.length);
    handledStems.add(stem);
    const binName = `${stem}${BIN_SUFFIX}`;
    clearedBytes += meta.byteLength;
    deletedEntries += 1;
    await removeEntryQuietly(dir, name);
    await removeEntryQuietly(dir, `${stem}${PARTIAL_SUFFIX}`);
    await removeEntryQuietly(dir, binName);
  }

  for (const name of entryNames) {
    if (!name.endsWith(PARTIAL_SUFFIX)) {
      continue;
    }
    const stem = name.slice(0, -PARTIAL_SUFFIX.length);
    if (handledStems.has(stem)) {
      continue;
    }
    const partial = await readOpfsPartialMetaFile(dir, name);
    if (!partial || extractModelFileFromCacheUrl(partial.url, repoId) === null) {
      continue;
    }
    const binName = `${stem}${BIN_SUFFIX}`;
    clearedBytes += partial.bytes;
    deletedEntries += 1;
    await removeEntryQuietly(dir, name);
    await removeEntryQuietly(dir, binName);
  }

  return { clearedBytes, deletedEntries };
}
