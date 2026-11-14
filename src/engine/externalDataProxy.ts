import { openOpfsSyncReadHandle, readOpfsCachedResponse, type OpfsDirectoryLike, type OpfsSyncReadHandle } from './opfsCacheStore.ts';

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
  readAt: (dest: Uint8Array, offset: number) => number
): Uint8Array {
  const safeLength = Math.max(0, Math.floor(byteLength));
  const proxy = Object.create(Uint8Array.prototype) as Uint8Array;

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

        const out = new Uint8Array(sliceLen);
        let filled = 0;
        while (filled < sliceLen) {
          const target = filled === 0 ? out : out.subarray(filled);
          const bytesRead = readAt(target, start + filled);
          if (!Number.isFinite(bytesRead) || bytesRead <= 0) {
            throw new Error(
              `Short synchronous OPFS external-data read at offset ${start + filled} (read ${bytesRead}, needed ${sliceLen - filled}).`
            );
          }
          filled += bytesRead;
        }
        return out;
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
  }

  private async openOrLoadExternalBytes(
    url: string,
    deps: ExternalDataResolveDeps,
    sessionHandles: OpfsSyncReadHandle[]
  ): Promise<Uint8Array> {
    if (deps.opfsRoot) {
      const syncHandle = await openOpfsSyncReadHandle(deps.opfsRoot, url);
      if (syncHandle) {
        this.activeHandles.add(syncHandle);
        sessionHandles.push(syncHandle);
        return createSyncAccessExternalDataProxy(syncHandle.byteLength, syncHandle.readAt);
      }

      const opfsResp = await readOpfsCachedResponse(deps.opfsRoot, url);
      if (opfsResp) {
        return new Uint8Array(await opfsResp.arrayBuffer());
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
          return new Uint8Array(await hit.arrayBuffer());
        }
      } catch {
        // Fall through to fetch fallback
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
    return new Uint8Array(await resp.arrayBuffer());
  }
}
