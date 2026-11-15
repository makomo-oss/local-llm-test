import type { OpfsSyncReadHandle } from './opfsCacheStore.ts';

export const DEFAULT_SYNC_RANGE_CHUNK_BYTES = 4 * 1024 * 1024; // 4 MiB per Range XHR
const MAX_TRANSIENT_XHR_RETRIES = 2;

export interface SyncXhrLike {
  open(method: string, url: string, async: boolean): void;
  setRequestHeader(name: string, value: string): void;
  send(body?: unknown): void;
  responseType: XMLHttpRequestResponseType;
  status: number;
  response: unknown;
  getResponseHeader(name: string): string | null;
}

export interface CreateSyncHttpRangeReadHandleOptions {
  expectedByteLength?: number;
  maxChunkBytes?: number;
  xhrFactory?: () => SyncXhrLike;
}

function isHttpUrl(url: string): boolean {
  return /^https?:\/\//i.test(url.trim());
}

function parseContentRangeTotal(headerValue: string | null): number | undefined {
  if (!headerValue) {
    return undefined;
  }
  const match = /^\s*bytes\s+\d+-\d+\/(\d+|\*)\s*$/i.exec(headerValue);
  if (!match || match[1] === '*') {
    return undefined;
  }
  const total = Number.parseInt(match[1]!, 10);
  return Number.isSafeInteger(total) && total > 0 ? total : undefined;
}

function parseIdentityContentLength(xhr: SyncXhrLike): number | undefined {
  const encoding = xhr.getResponseHeader('content-encoding')?.trim().toLowerCase();
  const isIdentity = !encoding || encoding === 'identity';
  const candidate = isIdentity
    ? (xhr.getResponseHeader('content-length') ?? xhr.getResponseHeader('x-linked-size'))
    : xhr.getResponseHeader('x-linked-size');
  if (!candidate) {
    return undefined;
  }
  const trimmed = candidate.trim();
  if (!/^\d+$/.test(trimmed)) {
    return undefined;
  }
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function isTransientStatus(status: number): boolean {
  return status === 0 || status === 408 || status === 429 || (status >= 500 && status <= 599);
}

function coerceResponseBytes(response: unknown): Uint8Array {
  if (response instanceof ArrayBuffer) {
    return new Uint8Array(response);
  }
  if (ArrayBuffer.isView(response)) {
    return new Uint8Array(response.buffer, response.byteOffset, response.byteLength);
  }
  throw new Error('Synchronous Range XHR did not return an ArrayBuffer payload');
}

function probeRemoteByteLength(url: string, createXhr: () => SyncXhrLike): number | undefined {
  try {
    const headXhr = createXhr();
    headXhr.open('HEAD', url, false);
    headXhr.send();
    if (headXhr.status >= 200 && headXhr.status < 300) {
      const len = parseIdentityContentLength(headXhr);
      if (len !== undefined) {
        return len;
      }
    }
  } catch {
    // Fall through to 1-byte Range probe
  }

  try {
    const probeXhr = createXhr();
    probeXhr.open('GET', url, false);
    probeXhr.responseType = 'arraybuffer';
    probeXhr.setRequestHeader('Range', 'bytes=0-0');
    probeXhr.send();
    if (probeXhr.status === 206) {
      return parseContentRangeTotal(probeXhr.getResponseHeader('content-range'));
    }
    if (probeXhr.status === 200) {
      return parseIdentityContentLength(probeXhr);
    }
  } catch {
    return undefined;
  }
  return undefined;
}

/**
 * Create a Dedicated-Worker synchronous `OpfsSyncReadHandle` backed by bounded
 * synchronous HTTP `Range` `XMLHttpRequest` calls. Used strictly as a resilient
 * last-resort fallback when OPFS `SyncAccessHandle` is unavailable and mobile
 * `FileReaderSync` throws `NotReadableError` on large cached Blobs.
 */
export function createSyncHttpRangeReadHandle(
  url: string,
  options: CreateSyncHttpRangeReadHandleOptions = {}
): OpfsSyncReadHandle | undefined {
  const createXhr =
    options.xhrFactory ??
    (typeof XMLHttpRequest === 'function' ? () => new XMLHttpRequest() : undefined);
  if (!createXhr || !isHttpUrl(url)) {
    return undefined;
  }

  const totalSize =
    typeof options.expectedByteLength === 'number' &&
    Number.isFinite(options.expectedByteLength) &&
    options.expectedByteLength > 0
      ? Math.trunc(options.expectedByteLength)
      : probeRemoteByteLength(url, createXhr);

  if (totalSize === undefined || totalSize <= 0) {
    return undefined;
  }

  const maxChunkBytes = Math.max(
    1,
    Math.trunc(options.maxChunkBytes ?? DEFAULT_SYNC_RANGE_CHUNK_BYTES)
  );
  let closed = false;

  const tagRangeFailure = (err: unknown): Error & { code: string } => {
    const base = err instanceof Error ? err : new Error(String(err));
    return Object.assign(base, { code: 'SYNC_HTTP_RANGE_FAILED' });
  };

  const fetchRangeChunk = (start: number, length: number): Uint8Array => {
    const endInclusive = start + length - 1;
    let lastError: unknown;
    for (let attempt = 0; attempt <= MAX_TRANSIENT_XHR_RETRIES; attempt += 1) {
      let terminalError: (Error & { code: string }) | null = null;
      try {
        const xhr = createXhr();
        xhr.open('GET', url, false);
        xhr.responseType = 'arraybuffer';
        xhr.setRequestHeader('Range', `bytes=${start}-${endInclusive}`);
        xhr.send();

        if (xhr.status === 206) {
          const bytes = coerceResponseBytes(xhr.response);
          if (bytes.byteLength < length) {
            terminalError = tagRangeFailure(
              `Short HTTP 206 Range response for ${url} (${bytes.byteLength} < ${length} bytes)`
            );
          } else {
            return bytes.byteLength === length ? bytes : bytes.subarray(0, length);
          }
        } else if (xhr.status === 200) {
          const bytes = coerceResponseBytes(xhr.response);
          if (bytes.byteLength >= start + length) {
            return bytes.subarray(start, start + length);
          }
          terminalError = tagRangeFailure(
            `HTTP 200 fallback response too short for ${url} (${bytes.byteLength} < ${start + length})`
          );
        } else if (attempt < MAX_TRANSIENT_XHR_RETRIES && isTransientStatus(xhr.status)) {
          lastError = tagRangeFailure(`HTTP ${xhr.status} reading Range bytes=${start}-${endInclusive}`);
          continue;
        } else {
          terminalError = tagRangeFailure(
            `HTTP ${xhr.status} failed reading Range bytes=${start}-${endInclusive} from ${url}`
          );
        }
      } catch (err) {
        lastError = err;
        if (attempt >= MAX_TRANSIENT_XHR_RETRIES) {
          throw tagRangeFailure(err);
        }
        continue;
      }
      if (terminalError) {
        throw terminalError;
      }
    }
    throw tagRangeFailure(lastError);
  };

  return {
    byteLength: totalSize,
    source: 'remote-range',
    readAt(dest: Uint8Array, offset: number): number {
      if (closed) {
        throw new Error(`Sync HTTP Range handle is already closed for ${url}`);
      }
      if (!Number.isFinite(offset) || offset < 0 || offset > totalSize) {
        throw new RangeError(`Out-of-range offset ${offset} for ${url} (size=${totalSize})`);
      }
      const safeOffset = Math.trunc(offset);
      if (dest.byteLength <= 0 || safeOffset === totalSize) {
        return 0;
      }
      const targetLen = Math.min(dest.byteLength, totalSize - safeOffset);
      let copied = 0;
      while (copied < targetLen) {
        const chunkLen = Math.min(maxChunkBytes, targetLen - copied);
        const chunk = fetchRangeChunk(safeOffset + copied, chunkLen);
        dest.set(chunk, copied);
        copied += chunk.byteLength;
      }
      return copied;
    },
    close(): void {
      closed = true;
    },
  };
}
