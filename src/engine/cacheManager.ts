import { MODEL_CATALOG, getModelById } from '../models/catalog.ts';
import type { ModelCacheEntry, ModelCacheInfo, ModelId, ModelSpec } from '../types.ts';

/**
 * Default Cache API bucket used by `@huggingface/transformers` for model files and ORT WASM assets.
 */
export const TRANSFORMERS_CACHE_NAME = 'transformers-cache';

const LARGE_BINARY_FILE_REGEX = /\.(?:onnx(?:_data(?:_\d+)?)?|bin|safetensors|wasm)$/i;

export interface ClearModelCacheResult {
  clearedBytes: number;
  deletedEntries: number;
}

export interface BrowserStorageEstimate {
  usageBytes?: number;
  quotaBytes?: number;
  availableBytes?: number;
}

interface CacheStorageLike {
  open(cacheName: string): Promise<Cache>;
  delete(cacheName: string): Promise<boolean>;
  has?(cacheName: string): Promise<boolean>;
}

function getCacheStorage(): CacheStorageLike | undefined {
  const scope = globalThis as { caches?: CacheStorageLike };
  const storage = scope.caches;
  if (!storage || typeof storage.open !== 'function') {
    return undefined;
  }
  return storage;
}

async function openExistingTransformersCache(): Promise<Cache | null> {
  const cacheStorage = getCacheStorage();
  if (!cacheStorage) {
    return null;
  }

  try {
    if (typeof cacheStorage.has === 'function') {
      const exists = await cacheStorage.has(TRANSFORMERS_CACHE_NAME);
      if (!exists) {
        return null;
      }
    }
    return await cacheStorage.open(TRANSFORMERS_CACHE_NAME);
  } catch {
    return null;
  }
}

/**
 * Extract the repository-relative file path when `url` belongs to `repoId`,
 * returning `null` when the request belongs to a different repository or asset.
 */
export function extractModelFileFromCacheUrl(url: string, repoId: string): string | null {
  if (!url || !repoId) {
    return null;
  }

  let path = url;
  try {
    path = decodeURIComponent(new URL(url, 'https://huggingface.co').pathname);
  } catch {
    // Fall back to raw string when URL parsing fails on custom cache keys.
  }

  const normalizedPath = path.replace(/^\/+/, '');
  const normalizedRepo = repoId.replace(/^\/+|\/+$/g, '');

  if (!normalizedPath.startsWith(`${normalizedRepo}/`)) {
    const nestedPrefix = `/${normalizedRepo}/`;
    const nestedIdx = path.indexOf(nestedPrefix);
    if (nestedIdx === -1) {
      return null;
    }
    const tail = path.slice(nestedIdx + nestedPrefix.length);
    return normalizeRepoTailToFile(tail);
  }

  const tail = normalizedPath.slice(normalizedRepo.length + 1);
  return normalizeRepoTailToFile(tail);
}

function normalizeRepoTailToFile(tail: string): string | null {
  if (!tail) {
    return null;
  }
  const resolveMatch = tail.match(/^resolve\/[^/]+\/(.+)$/);
  if (resolveMatch?.[1]) {
    return resolveMatch[1];
  }
  return tail;
}

/**
 * Determine cached response byte size without reading gigabyte weight files into RAM.
 * Always inspects `content-length` header first, falling back to `blob().size` only for
 * small non-weight files when the header is missing.
 */
export async function measureCachedResponseBytes(
  response: Response | undefined,
  fileHint: string
): Promise<number> {
  if (!response) {
    return 0;
  }

  const rawLength = response.headers?.get?.('content-length');
  if (typeof rawLength === 'string' && rawLength.trim().length > 0) {
    const parsed = Number(rawLength.trim());
    if (Number.isFinite(parsed) && parsed >= 0) {
      return Math.floor(parsed);
    }
  }

  if (LARGE_BINARY_FILE_REGEX.test(fileHint)) {
    return 0;
  }

  if (typeof response.blob === 'function') {
    try {
      const blob = await response.blob();
      if (typeof blob?.size === 'number' && Number.isFinite(blob.size) && blob.size >= 0) {
        return Math.floor(blob.size);
      }
    } catch {
      return 0;
    }
  }

  return 0;
}

function buildEmptyCacheInfo(model: ModelSpec, checkedAt: number): ModelCacheInfo {
  return {
    modelId: model.id,
    repoId: model.repoId,
    cached: false,
    cachedBytes: 0,
    fileCount: 0,
    files: [],
    lastCheckedAt: checkedAt,
  };
}

/**
 * Inspect `CacheStorage` (`transformers-cache`) entries for each requested catalog model
 * without loading large model weight bodies into memory.
 */
export async function inspectModelCaches(
  models: readonly ModelSpec[] = MODEL_CATALOG
): Promise<ModelCacheInfo[]> {
  const checkedAt = Date.now();
  const cache = await openExistingTransformersCache();
  if (!cache) {
    return models.map((model) => buildEmptyCacheInfo(model, checkedAt));
  }

  let requests: readonly Request[];
  try {
    requests = await cache.keys();
  } catch {
    return models.map((model) => buildEmptyCacheInfo(model, checkedAt));
  }

  const entriesByModelId = new Map<ModelId, ModelCacheEntry[]>();
  for (const model of models) {
    entriesByModelId.set(model.id, []);
  }

  for (const request of requests) {
    const url = request.url;
    let matchedModel: ModelSpec | undefined;
    let matchedFile: string | null = null;

    for (const model of models) {
      const file = extractModelFileFromCacheUrl(url, model.repoId);
      if (file !== null) {
        matchedModel = model;
        matchedFile = file;
        break;
      }
    }

    if (!matchedModel || matchedFile === null) {
      continue;
    }

    let bytes = 0;
    try {
      const response = await cache.match(request);
      bytes = await measureCachedResponseBytes(response, matchedFile);
    } catch {
      bytes = 0;
    }

    entriesByModelId.get(matchedModel.id)?.push({
      url,
      file: matchedFile,
      bytes,
    });
  }

  return models.map((model) => {
    const files = (entriesByModelId.get(model.id) ?? []).sort((a, b) =>
      a.file.localeCompare(b.file)
    );
    const cachedBytes = files.reduce((sum, entry) => sum + entry.bytes, 0);
    return {
      modelId: model.id,
      repoId: model.repoId,
      cached: files.length > 0,
      cachedBytes,
      fileCount: files.length,
      files,
      lastCheckedAt: checkedAt,
    };
  });
}

/**
 * Clear cached entries for a specific model (`repoId` match) or delete the entire
 * `transformers-cache` bucket when `modelId` is omitted.
 */
export async function clearModelCache(modelId?: ModelId): Promise<ClearModelCacheResult> {
  const cacheStorage = getCacheStorage();
  if (!cacheStorage) {
    return { clearedBytes: 0, deletedEntries: 0 };
  }

  if (modelId === undefined) {
    let clearedBytes = 0;
    let deletedEntries = 0;

    const existingCache = await openExistingTransformersCache();
    if (existingCache) {
      try {
        const requests = await existingCache.keys();
        for (const request of requests) {
          deletedEntries += 1;
          try {
            const response = await existingCache.match(request);
            clearedBytes += await measureCachedResponseBytes(response, request.url);
          } catch {
            // Keep counting entry even if size measurement fails.
          }
        }
      } catch {
        // Proceed with full cache deletion even if key enumeration failed.
      }
    }

    try {
      const deleted = await cacheStorage.delete(TRANSFORMERS_CACHE_NAME);
      if (!deleted && deletedEntries === 0) {
        return { clearedBytes: 0, deletedEntries: 0 };
      }
      return { clearedBytes, deletedEntries };
    } catch {
      return { clearedBytes: 0, deletedEntries: 0 };
    }
  }

  const model = getModelById(modelId);
  if (!model) {
    return { clearedBytes: 0, deletedEntries: 0 };
  }

  const cache = await openExistingTransformersCache();
  if (!cache) {
    return { clearedBytes: 0, deletedEntries: 0 };
  }

  let requests: readonly Request[];
  try {
    requests = await cache.keys();
  } catch {
    return { clearedBytes: 0, deletedEntries: 0 };
  }

  let clearedBytes = 0;
  let deletedEntries = 0;

  for (const request of requests) {
    const file = extractModelFileFromCacheUrl(request.url, model.repoId);
    if (file === null) {
      continue;
    }

    let entryBytes = 0;
    try {
      const response = await cache.match(request);
      entryBytes = await measureCachedResponseBytes(response, file);
    } catch {
      entryBytes = 0;
    }

    try {
      const deleted = await cache.delete(request);
      if (deleted) {
        deletedEntries += 1;
        clearedBytes += entryBytes;
      }
    } catch {
      // Ignore single-entry delete failure and continue clearing remaining entries.
    }
  }

  return { clearedBytes, deletedEntries };
}

/**
 * Safely query the browser's `navigator.storage.estimate()` API.
 */
export async function getBrowserStorageEstimate(): Promise<BrowserStorageEstimate> {
  const nav = (
    globalThis as {
      navigator?: {
        storage?: {
          estimate?: () => Promise<{ usage?: number; quota?: number }>;
        };
      };
    }
  ).navigator;

  if (typeof nav?.storage?.estimate !== 'function') {
    return {};
  }

  try {
    const estimate = await nav.storage.estimate();
    const usageBytes =
      typeof estimate?.usage === 'number' && Number.isFinite(estimate.usage) && estimate.usage >= 0
        ? estimate.usage
        : undefined;
    const quotaBytes =
      typeof estimate?.quota === 'number' && Number.isFinite(estimate.quota) && estimate.quota >= 0
        ? estimate.quota
        : undefined;
    const availableBytes =
      usageBytes !== undefined && quotaBytes !== undefined
        ? Math.max(0, quotaBytes - usageBytes)
        : undefined;

    return {
      ...(usageBytes !== undefined ? { usageBytes } : {}),
      ...(quotaBytes !== undefined ? { quotaBytes } : {}),
      ...(availableBytes !== undefined ? { availableBytes } : {}),
    };
  } catch {
    return {};
  }
}
