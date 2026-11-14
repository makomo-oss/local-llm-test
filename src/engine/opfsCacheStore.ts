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

export interface OpfsInspectedEntry {
  url: string;
  file: string;
  bytes: number;
}

export interface OpfsClearResult {
  clearedBytes: number;
  deletedEntries: number;
}

const META_SUFFIX = '.meta.json';
const BIN_SUFFIX = '.bin';

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
    const access = await handle.createSyncAccessHandle();
    try {
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
    } finally {
      try {
        access.close();
      } catch {
        // Ignore close cleanup failure
      }
    }
    return;
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
  if (!entry || typeof entry.fileHandle.createSyncAccessHandle !== 'function') {
    return undefined;
  }

  let accessHandle: OpfsSyncAccessHandleLike | null = null;
  try {
    accessHandle = await entry.fileHandle.createSyncAccessHandle();
    if (typeof accessHandle.read !== 'function') {
      accessHandle.close();
      return undefined;
    }
    const readFn = accessHandle.read.bind(accessHandle);
    const totalSize =
      typeof accessHandle.getSize === 'function' ? accessHandle.getSize() : entry.meta.byteLength;
    if (!Number.isFinite(totalSize) || totalSize !== entry.meta.byteLength) {
      accessHandle.close();
      return undefined;
    }

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
  } catch {
    if (accessHandle) {
      try {
        accessHandle.close();
      } catch {
        // Ignore close errors
      }
    }
    return undefined;
  }
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
 */
export async function writeStreamToOpfsStore(options: {
  opfsRoot: OpfsDirectoryLike;
  url: string;
  stream: ReadableStream<Uint8Array>;
  headers: Headers;
  expectedTotalBytes?: number;
  signal?: AbortSignal;
  onProgress?: (loadedBytes: number, totalBytes?: number) => void;
}): Promise<number> {
  const { opfsRoot, url, stream, headers, expectedTotalBytes, signal, onProgress } = options;
  const dir = await getPersistentCacheDir(opfsRoot, true);
  if (!dir) {
    throw new Error('Persistent OPFS directory is unavailable.');
  }

  const stem = buildOpfsEntryStem(url);
  const binName = `${stem}${BIN_SUFFIX}`;
  const metaName = `${stem}${META_SUFFIX}`;
  // Invalidate any stale meta before touching .bin so an interrupted download is never treated as complete.
  await removeEntryQuietly(dir, metaName);

  const fileHandle = await dir.getFileHandle(binName, { create: true });
  const reader = stream.getReader();
  let accessHandle: OpfsSyncAccessHandleLike | null = null;
  let writable: OpfsWritableStreamLike | null = null;
  let loadedBytes = 0;

  try {
    if (typeof fileHandle.createSyncAccessHandle === 'function') {
      accessHandle = await fileHandle.createSyncAccessHandle();
      accessHandle.truncate?.(0);
    } else if (typeof fileHandle.createWritable === 'function') {
      writable = await fileHandle.createWritable();
    } else {
      throw new Error('OPFS file handle does not support SyncAccessHandle or WritableStream.');
    }

    while (true) {
      if (signal?.aborted) {
        throw new DOMException('Aborted', 'AbortError');
      }
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (!value || value.byteLength === 0) {
        continue;
      }

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
    }

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
      throw new Error(
        `OPFS write size mismatch for ${url}: wrote ${finalBytes} bytes, expected ${expectedTotalBytes}.`
      );
    }

    const finalHeaders = new Headers(headers);
    finalHeaders.delete('content-encoding');
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
    try {
      await reader.cancel(err);
    } catch {
      // Ignore cancel errors
    }
    if (accessHandle) {
      try {
        accessHandle.close();
      } catch {
        // Ignore close errors
      }
    }
    if (writable) {
      await writable.abort?.(err).catch(() => {});
    }
    await removeEntryQuietly(dir, metaName);
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
  for (const name of entryNames) {
    if (!name.endsWith(META_SUFFIX)) {
      continue;
    }
    const meta = await readOpfsMetaFile(dir, name);
    if (!meta || extractModelFileFromCacheUrl(meta.url, repoId) === null) {
      continue;
    }
    const stem = name.slice(0, -META_SUFFIX.length);
    const binName = `${stem}${BIN_SUFFIX}`;
    clearedBytes += meta.byteLength;
    deletedEntries += 1;
    await removeEntryQuietly(dir, name);
    await removeEntryQuietly(dir, binName);
  }

  return { clearedBytes, deletedEntries };
}
