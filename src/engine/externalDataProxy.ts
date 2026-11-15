import { migrateCacheResponseToOpfs } from './modelPrecacher.ts';
import {
  createBlobSyncReadHandle,
  isPersistentOpfsCapable,
  openOpfsSyncReadHandle,
  probeSyncReadHandle,
  readOpfsCachedResponse,
  type OpfsDirectoryLike,
  type OpfsSyncReadHandle,
} from './opfsCacheStore.ts';
import { createSyncHttpRangeReadHandle } from './syncHttpRangeReader.ts';

/**
 * Reusable single-buffer scratch pool for synchronous ORT `externalData.subarray()`
 * slice reads during serialized `InferenceSession.create()` calls. Reuses one growing
 * `ArrayBuffer` (released immediately after session init) instead of allocating ~900+
 * multi-MiB temporary `Uint8Array`s totaling 1.5-2.9 GiB of JS heap churn.
 */
export const DEFAULT_RETAINED_SCRATCH_MAX_BYTES = 18 * 1024 * 1024;

function alignUp16(byteLength: number): number {
  return Math.ceil(byteLength / 16) * 16;
}

export class ExternalDataScratchPool {
  private buffer: ArrayBuffer | null = null;
  private pendingHandoff: {
    buffer: ArrayBuffer;
    byteLength: number;
    alignedSize: number;
  } | null = null;

  acquire(byteLength: number): Uint8Array {
    this.pendingHandoff = null;
    if (!Number.isFinite(byteLength) || byteLength <= 0) {
      return new Uint8Array(0);
    }
    const len = Math.floor(byteLength);
    if (!this.buffer || this.buffer.byteLength < len) {
      // Clear previous buffer reference before allocating a larger capacity so they never overlap.
      this.buffer = null;
      this.buffer = new ArrayBuffer(len);
    }
    return new Uint8Array(this.buffer, 0, len);
  }

  /**
   * Register the just-filled external-data slice so the WebGPU upload optimizer can
   * share the same backing ArrayBuffer without allocating a second 192 MiB staging copy.
   */
  recordHandoff(buffer: ArrayBuffer, byteLength: number, alignedSize: number): void {
    if (this.buffer === buffer && alignedSize > 0 && byteLength > 0 && alignedSize >= byteLength) {
      this.pendingHandoff = { buffer, byteLength, alignedSize };
    }
  }

  /**
   * Consume the last recorded zero-copy handoff only when `alignedSize` matches exactly.
   */
  consumeHandoff(alignedSize: number): ArrayBuffer | null {
    const handoff = this.pendingHandoff;
    this.pendingHandoff = null;
    if (!handoff || handoff.alignedSize !== alignedSize || handoff.buffer !== this.buffer) {
      return null;
    }
    return handoff.buffer;
  }

  /**
   * Release oversized scratch slabs (> 18 MiB, e.g. 192 MiB lm_head) immediately after upload
   * while preserving <= 18 MiB slab reuse for normal weight tensors.
   */
  trimIfLargerThan(maxRetainedBytes = DEFAULT_RETAINED_SCRATCH_MAX_BYTES): void {
    if (this.buffer && this.buffer.byteLength > maxRetainedBytes) {
      this.buffer = null;
      this.pendingHandoff = null;
    }
  }

  release(): void {
    this.pendingHandoff = null;
    this.buffer = null;
  }
}

export interface SyncAccessExternalDataProxyOptions {
  scratchPool?: ExternalDataScratchPool;
  onReadError?: (info: { offset: number; length: number; error: unknown }) => void;
}

/**
 * Create a lightweight `Uint8Array`-prototype slice reader backed synchronously by an OPFS
 * `FileSystemSyncAccessHandle` (or custom `readAt` callback).
 *
 * ORT's `loadFile` preserves values where `value instanceof Uint8Array` without copying, and
 * ORT's WASM/WebGPU C++ external-data bridge (`mountExternalData`) accesses only `.byteLength`
 * and `.subarray(begin, end)` tensor-by-tensor — avoiding multi-GiB full-file JS `ArrayBuffer` copies.
 */
export function createSyncAccessExternalDataProxy(
  byteLength: number,
  readAt: (dest: Uint8Array, offset: number) => number,
  options?: SyncAccessExternalDataProxyOptions
): Uint8Array {
  const safeLength = Math.max(0, Math.floor(byteLength));
  const proxy = Object.create(Uint8Array.prototype) as Uint8Array;
  const scratchPool = options?.scratchPool;
  const onReadError = options?.onReadError;

  Object.defineProperties(proxy, {
    BYTES_PER_ELEMENT: {
      value: 1,
      enumerable: false,
      configurable: true,
    },
    byteLength: {
      value: safeLength,
      enumerable: true,
      configurable: true,
    },
    length: {
      value: safeLength,
      enumerable: false,
      configurable: true,
    },
    byteOffset: {
      value: 0,
      enumerable: false,
      configurable: true,
    },
    subarray: {
      value(begin = 0, end = safeLength): Uint8Array {
        const rawStart = Number.isFinite(begin) ? Math.trunc(begin) : 0;
        const rawEnd = end === undefined ? safeLength : Number.isFinite(end) ? Math.trunc(end) : safeLength;
        const start =
          rawStart < 0 ? Math.max(0, safeLength + rawStart) : Math.min(safeLength, rawStart);
        const stop = rawEnd < 0 ? Math.max(0, safeLength + rawEnd) : Math.min(safeLength, rawEnd);
        const sliceLen = Math.max(0, stop - start);
        if (sliceLen === 0) {
          return new Uint8Array(0);
        }

        try {
          const alignedSize = alignUp16(sliceLen);
          const backing = scratchPool ? scratchPool.acquire(alignedSize) : new Uint8Array(sliceLen);
          const out = backing.length === sliceLen ? backing : backing.subarray(0, sliceLen);
          let filled = 0;
          while (filled < sliceLen) {
            const target = filled === 0 ? out : out.subarray(filled);
            const bytesRead = readAt(target, start + filled);
            if (!Number.isFinite(bytesRead) || bytesRead <= 0) {
              throw new Error(
                `Short synchronous external-data read at offset ${start + filled} (read ${bytesRead}, needed ${sliceLen - filled}).`
              );
            }
            filled += bytesRead;
          }
          const result = filled === sliceLen ? out : out.subarray(0, filled);
          if (scratchPool && filled === sliceLen) {
            if (alignedSize > sliceLen) {
              // Zero only trailing 16-byte WebGPU alignment bytes without touching [0 .. sliceLen).
              backing.fill(0, sliceLen, alignedSize);
            }
            if (backing.buffer instanceof ArrayBuffer) {
              scratchPool.recordHandoff(backing.buffer, sliceLen, alignedSize);
            }
          }
          return result;
        } catch (err) {
          onReadError?.({ offset: start, length: sliceLen, error: err });
          throw err;
        }
      },
      enumerable: false,
      configurable: true,
    },
  });

  return proxy;
}

function normalizeExternalFileKey(pathOrFileName: string): string {
  const withoutQuery = (pathOrFileName.split('#')[0]?.split('?')[0] ?? pathOrFileName).trim();
  const clean = withoutQuery.replace(/^\.\//, '');
  return clean.split('/').pop() ?? clean;
}

export interface ExternalDataResolveDeps {
  opfsRoot?: OpfsDirectoryLike | null;
  cacheName: string;
  cacheStorage?: Pick<CacheStorage, 'open'> | null;
  fetchFn?: typeof fetch;
}

/**
 * Coordinates virtualized `.onnx_data*` URLs between `createSerializedCacheAdapter` and
 * `InferenceSession.create()` so non-embedding sessions stream external data from OPFS on
 * demand and close all `SyncAccessHandle` descriptors immediately after session creation.
 */
export class ExternalDataSessionBridge {
  private readonly urlByKey = new Map<string, string>();
  private readonly activeHandles = new Set<OpfsSyncReadHandle>();
  private readonly scratchPool = new ExternalDataScratchPool();

  getScratchPool(): ExternalDataScratchPool {
    return this.scratchPool;
  }

  registerUrl(fileNameOrPath: string, url: string): void {
    const clean = fileNameOrPath.replace(/^\.\//, '');
    if (clean) {
      this.urlByKey.set(clean, url);
    }
    const baseName = normalizeExternalFileKey(clean);
    if (baseName) {
      this.urlByKey.set(baseName, url);
    }
  }

  resolveUrl(fileNameOrPath: string): string | undefined {
    const clean = fileNameOrPath.replace(/^\.\//, '');
    return this.urlByKey.get(clean) ?? this.urlByKey.get(normalizeExternalFileKey(clean));
  }

  /**
   * Resolve any bare path external-data entries in `sessionOptions.externalData` into
   * streaming OPFS `SyncAccessHandle` proxies (or buffered fallback if sync handles are
   * unavailable), returning a cleanup callback that closes opened handles immediately.
   */
  async prepareSessionExternalData(
    sessionOptions: Record<string, unknown>,
    deps: ExternalDataResolveDeps
  ): Promise<() => void> {
    const rawEntries = sessionOptions.externalData;
    if (!Array.isArray(rawEntries) || rawEntries.length === 0) {
      return () => {};
    }

    const sessionHandles: OpfsSyncReadHandle[] = [];
    const resolvedEntries: Array<{ path: string; data: Uint8Array }> = [];
    let modified = false;

    try {
      for (const entry of rawEntries) {
        if (typeof entry === 'string') {
          const url = this.resolveUrl(entry);
          if (!url) {
            // Preserve non-virtualized string entry untouched
            resolvedEntries.push(entry as unknown as { path: string; data: Uint8Array });
            continue;
          }
          const resolvedData = await this.openOrLoadExternalBytes(url, deps, sessionHandles);
          resolvedEntries.push({ path: entry, data: resolvedData });
          modified = true;
          continue;
        }

        if (entry && typeof entry === 'object') {
          const obj = entry as { path?: unknown; data?: unknown };
          if (typeof obj.path === 'string' && obj.data instanceof Uint8Array) {
            resolvedEntries.push({ path: obj.path, data: obj.data });
            continue;
          }
          if (
            typeof obj.path === 'string' &&
            (typeof obj.data === 'string' || obj.data === undefined)
          ) {
            const url =
              (typeof obj.data === 'string' ? this.resolveUrl(obj.data) : undefined) ??
              this.resolveUrl(obj.path);
            if (url) {
              const resolvedData = await this.openOrLoadExternalBytes(url, deps, sessionHandles);
              resolvedEntries.push({ path: obj.path, data: resolvedData });
              modified = true;
              continue;
            }
          }
        }

        resolvedEntries.push(entry as { path: string; data: Uint8Array });
      }
    } catch (err) {
      this.closeHandles(sessionHandles);
      throw err;
    }

    if (modified) {
      sessionOptions.externalData = resolvedEntries;
    }

    return () => {
      this.closeHandles(sessionHandles);
    };
  }

  closeAll(): void {
    for (const handle of this.activeHandles) {
      try {
        handle.close();
      } catch {
        // Ignore teardown errors
      }
    }
    this.activeHandles.clear();
    this.scratchPool.release();
  }

  private lastReadError: {
    url: string;
    offset: number;
    length: number;
    error: unknown;
  } | null = null;

  consumeLastReadError(): {
    url: string;
    offset: number;
    length: number;
    error: unknown;
  } | null {
    const err = this.lastReadError;
    this.lastReadError = null;
    return err;
  }

  clearLastReadError(): void {
    this.lastReadError = null;
  }

  private wrapVerifiedHandle(
    handle: OpfsSyncReadHandle,
    url: string,
    sessionHandles: OpfsSyncReadHandle[]
  ): Uint8Array {
    try {
      probeSyncReadHandle(handle, url);
    } catch (err) {
      try {
        handle.close();
      } catch {
        // Ignore cleanup failure after failed readiness probe
      }
      throw err;
    }

    this.activeHandles.add(handle);
    sessionHandles.push(handle);
    return createSyncAccessExternalDataProxy(handle.byteLength, handle.readAt, {
      scratchPool: this.scratchPool,
      onReadError: ({ offset, length, error }) => {
        this.lastReadError = { url, offset, length, error };
      },
    });
  }

  private closeHandles(handles: OpfsSyncReadHandle[]): void {
    for (const handle of handles) {
      try {
        handle.close();
      } catch {
        // Ignore teardown errors
      }
      this.activeHandles.delete(handle);
    }
    if (this.activeHandles.size === 0) {
      this.scratchPool.release();
    }
  }

  private async openOrLoadExternalBytes(
    url: string,
    deps: ExternalDataResolveDeps,
    sessionHandles: OpfsSyncReadHandle[]
  ): Promise<Uint8Array> {
    const MAX_IN_MEMORY_EXTERNAL_BYTES = 64 * 1024 * 1024;
    let lastLocalReadError: unknown = null;
    let knownByteLength: number | undefined;

    if (deps.opfsRoot) {
      let syncHandle = await openOpfsSyncReadHandle(deps.opfsRoot, url);
      if (syncHandle?.source === 'blob-file-reader') {
        try {
          // Brief retry in case a preceding SyncAccessHandle close was still settling on mobile WebKit/Chrome.
          knownByteLength = syncHandle.byteLength;
          syncHandle.close();
        } catch {
          // Ignore close error before retry
        }
        await new Promise<void>((resolve) => setTimeout(resolve, 20));
        syncHandle = await openOpfsSyncReadHandle(deps.opfsRoot, url);
      }

      if (syncHandle) {
        knownByteLength = syncHandle.byteLength;
        try {
          return this.wrapVerifiedHandle(syncHandle, url, sessionHandles);
        } catch (err) {
          lastLocalReadError = err;
          this.lastReadError = {
            url,
            offset: 0,
            length: Math.min(64, syncHandle.byteLength),
            error: err,
          };
        }
      }

      if (!lastLocalReadError) {
        const opfsResp = await readOpfsCachedResponse(deps.opfsRoot, url);
        if (opfsResp) {
          const contentLen = Number(opfsResp.headers.get('content-length') ?? '0');
          if (Number.isFinite(contentLen) && contentLen > 0) {
            knownByteLength = contentLen;
          }
          if (!Number.isFinite(contentLen) || contentLen <= MAX_IN_MEMORY_EXTERNAL_BYTES) {
            return new Uint8Array(await opfsResp.arrayBuffer());
          }
        }
      }
    }

    const storage =
      deps.cacheStorage === undefined
        ? (globalThis as { caches?: Pick<CacheStorage, 'open'> }).caches
        : deps.cacheStorage;
    if (storage && typeof storage.open === 'function') {
      try {
        const cache = await storage.open(deps.cacheName);
        const hit = await cache.match(url);
        if (hit) {
          const headerLen = Number(hit.headers?.get?.('content-length') ?? '0');
          if (Number.isFinite(headerLen) && headerLen > 0) {
            knownByteLength = headerLen;
          }

          const cleanUrl = url.split('#')[0]?.split('?')[0] ?? url;
          const isExternalBinary = /\.onnx_data(?:_\d+)?$/i.test(cleanUrl);
          if (
            isExternalBinary &&
            isPersistentOpfsCapable(deps.opfsRoot) &&
            typeof hit.clone === 'function'
          ) {
            const migrated = await migrateCacheResponseToOpfs({
              opfsRoot: deps.opfsRoot,
              url,
              cachedResponse: hit.clone(),
              cache: typeof (cache as { delete?: unknown }).delete === 'function' ? cache : undefined,
            });
            if (migrated) {
              const migratedHandle = await openOpfsSyncReadHandle(deps.opfsRoot, url);
              if (migratedHandle) {
                knownByteLength = migratedHandle.byteLength;
                try {
                  return this.wrapVerifiedHandle(migratedHandle, url, sessionHandles);
                } catch (err) {
                  lastLocalReadError = err;
                  this.lastReadError = {
                    url,
                    offset: 0,
                    length: Math.min(64, migratedHandle.byteLength),
                    error: err,
                  };
                }
              }
            }
          }

          const hasWorkerFileReaderSync =
            typeof (globalThis as { FileReaderSync?: unknown }).FileReaderSync === 'function';
          if (hasWorkerFileReaderSync && typeof hit.blob === 'function') {
            const blob = await hit.blob();
            if (blob.size > 0) {
              knownByteLength = blob.size;
            }
            const blobHandle = createBlobSyncReadHandle(blob, url);
            if (blobHandle) {
              try {
                return this.wrapVerifiedHandle(blobHandle, url, sessionHandles);
              } catch (err) {
                lastLocalReadError = err;
                this.lastReadError = {
                  url,
                  offset: 0,
                  length: Math.min(64, blob.size),
                  error: err,
                };
              }
            } else if (blob.size <= MAX_IN_MEMORY_EXTERNAL_BYTES) {
              return new Uint8Array(await blob.arrayBuffer());
            }
          } else if (
            !lastLocalReadError &&
            (!Number.isFinite(headerLen) || headerLen <= MAX_IN_MEMORY_EXTERNAL_BYTES)
          ) {
            return new Uint8Array(await hit.arrayBuffer());
          }
        }
      } catch (err) {
        if (!lastLocalReadError) {
          lastLocalReadError = err;
        }
      }
    }

    // Last-resort fallback for mobile browsers where SyncAccessHandle is unavailable AND
    // FileReaderSync throws NotReadableError on multi-hundred-MiB Blobs: stream via sync HTTP Range.
    if (lastLocalReadError || (knownByteLength !== undefined && knownByteLength > MAX_IN_MEMORY_EXTERNAL_BYTES)) {
      const rangeHandle = createSyncHttpRangeReadHandle(url, {
        expectedByteLength: knownByteLength,
      });
      if (rangeHandle) {
        try {
          return this.wrapVerifiedHandle(rangeHandle, url, sessionHandles);
        } catch (rangeErr) {
          lastLocalReadError = rangeErr;
        }
      }
      if (lastLocalReadError) {
        throw lastLocalReadError;
      }
    }

    const fetchImpl = deps.fetchFn ?? globalThis.fetch;
    if (typeof fetchImpl !== 'function') {
      throw new Error(`Unable to resolve external ONNX data for ${url}`);
    }
    const resp = await fetchImpl(url);
    if (!resp.ok) {
      throw new Error(`Failed to fetch external ONNX data (${resp.status}) for ${url}`);
    }
    const respLen = Number(resp.headers?.get?.('content-length') ?? '0');
    if (Number.isFinite(respLen) && respLen > MAX_IN_MEMORY_EXTERNAL_BYTES) {
      await resp.body?.cancel?.().catch(() => {});
      const rangeHandle = createSyncHttpRangeReadHandle(url, {
        expectedByteLength: respLen,
      });
      if (rangeHandle) {
        return this.wrapVerifiedHandle(rangeHandle, url, sessionHandles);
      }
    }
    return new Uint8Array(await resp.arrayBuffer());
  }
}
