import { InferenceSession } from 'onnxruntime-web/webgpu';

/**
 * Derive a logical ONNX session group key from a model artifact URL or file path so
 * multi-session models (`embed_tokens`, `decoder_model_merged`, etc.) load strictly one at a time.
 * Returns `null` for non-ONNX assets (`config.json`, `tokenizer.json`, `.wasm`, etc.).
 */
export function parseOnnxSessionGroup(urlOrPath: string): string | null {
  const clean = (urlOrPath.split('#')[0]?.split('?')[0] ?? urlOrPath).trim();
  if (!/\.onnx(?:_data(?:_\d+)?)?$/i.test(clean)) {
    return null;
  }

  const fileName = clean.split('/').pop() ?? clean;
  if (/^embed_tokens(?:_|\.)/i.test(fileName)) {
    return 'embed_tokens';
  }
  if (/^decoder_model_merged(?:_|\.)/i.test(fileName)) {
    return 'decoder_model_merged';
  }
  if (/^decoder_model(?:_|\.)/i.test(fileName)) {
    return 'decoder_model';
  }
  if (/^decoder(?:_|\.)/i.test(fileName)) {
    return 'decoder';
  }
  if (/^vision_encoder(?:_|\.)/i.test(fileName)) {
    return 'vision_encoder';
  }
  if (/^audio_encoder(?:_|\.)/i.test(fileName)) {
    return 'audio_encoder';
  }
  if (/^model(?:_|\.)/i.test(fileName)) {
    return 'model';
  }

  const stripped = fileName
    .replace(/\.onnx(?:_data(?:_\d+)?)?$/i, '')
    .replace(/_(?:fp16|fp32|q4f16|q4|quantized|int8|uint8|bnb4)$/i, '');
  return stripped || 'model';
}

interface SessionWaiter {
  group: string;
  resolve: () => void;
  reject: (err: unknown) => void;
}

/**
 * Coordinates cache reads and `InferenceSession.create()` calls across Transformers.js's
 * parallel `constructSessions()` (`Promise.all`) so two sessions never buffer hundreds of MiB
 * of JS `Uint8Array` weights at the same instant.
 */
export class SessionLoadCoordinator {
  private activeGroup: string | null = null;
  private readonly waiters: SessionWaiter[] = [];
  private disposedError: Error | null = null;

  get currentGroup(): string | null {
    return this.activeGroup;
  }

  get waitingCount(): number {
    return this.waiters.length;
  }

  /**
   * Wait until `group` holds the exclusive session loading slot.
   * Re-entrant for calls belonging to the currently active group (e.g. `.onnx` + `.onnx_data`).
   */
  async enterGroup(group: string, signal?: AbortSignal): Promise<void> {
    if (this.disposedError) {
      throw this.disposedError;
    }
    if (signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }
    if (this.activeGroup === null) {
      this.activeGroup = group;
      return;
    }
    if (this.activeGroup === group) {
      return;
    }

    await new Promise<void>((resolve, reject) => {
      const waiter: SessionWaiter = {
        group,
        resolve: () => {
          signal?.removeEventListener('abort', onAbort);
          resolve();
        },
        reject: (err) => {
          signal?.removeEventListener('abort', onAbort);
          reject(err);
        },
      };

      const onAbort = () => {
        const idx = this.waiters.indexOf(waiter);
        if (idx >= 0) {
          this.waiters.splice(idx, 1);
        }
        waiter.reject(new DOMException('Aborted', 'AbortError'));
      };

      signal?.addEventListener('abort', onAbort, { once: true });
      this.waiters.push(waiter);
    });
  }

  /**
   * Release the active session group once its ORT `InferenceSession.create()` has finished and
   * JS external data buffers have been cleared, waking the next queued session group (if any).
   */
  completeActiveGroup(): void {
    this.activeGroup = null;
    if (this.waiters.length === 0) {
      return;
    }

    const next = this.waiters.shift()!;
    this.activeGroup = next.group;
    next.resolve();

    // Also wake any peers waiting on that same newly activated group.
    for (let i = this.waiters.length - 1; i >= 0; i -= 1) {
      const candidate = this.waiters[i];
      if (candidate && candidate.group === next.group) {
        this.waiters.splice(i, 1);
        candidate.resolve();
      }
    }
  }

  /**
   * Abort all pending session waiters (called on load cancellation, errors, or disposal).
   */
  abortAll(err?: unknown): void {
    const error =
      err instanceof Error ? err : new Error(typeof err === 'string' ? err : 'Session load aborted');
    this.disposedError = error;
    this.activeGroup = null;
    while (this.waiters.length > 0) {
      const waiter = this.waiters.shift()!;
      waiter.reject(error);
    }
  }
}

/**
 * Release JS references to external ONNX weight buffers (`externalData`) immediately after
 * `InferenceSession.create` returns (ORT has already copied/mounted them into WASM/GPU).
 */
export function releaseSessionExternalData(options: unknown): void {
  if (!options || typeof options !== 'object') {
    return;
  }
  const opts = options as { externalData?: unknown };
  if (!Array.isArray(opts.externalData)) {
    return;
  }
  for (let i = 0; i < opts.externalData.length; i += 1) {
    const entry = opts.externalData[i];
    if (entry && typeof entry === 'object' && Object.isExtensible(entry)) {
      try {
        (entry as { data?: Uint8Array }).data = new Uint8Array(0);
      } catch {
        // Ignore frozen/read-only descriptors
      }
    }
  }
  try {
    opts.externalData.length = 0;
  } catch {
    // Ignore read-only arrays
  }
}

export interface SerializedSessionCreateHooks {
  target?: {
    create: typeof InferenceSession.create;
  };
  yieldAfterCreateMs?: number;
  onSessionCreated?: () => void;
}

/**
 * Wrap `InferenceSession.create` while running `fn` so multi-session Transformers.js loads
 * initialize one session completely, drop its JS ArrayBuffers, yield for GC, and only then
 * permit the next queued session to read weights from OPFS/CacheStorage.
 */
export async function withSerializedInferenceSessions<T>(
  coordinator: SessionLoadCoordinator,
  fn: () => Promise<T>,
  hooks: SerializedSessionCreateHooks = {}
): Promise<T> {
  const target =
    hooks.target ??
    (InferenceSession as unknown as {
      create: typeof InferenceSession.create;
    });
  const yieldMs = hooks.yieldAfterCreateMs ?? 50;
  const origCreate = target.create;

  let createTail: Promise<void> = Promise.resolve();

  const wrappedCreate = function (
    this: unknown,
    arg0: unknown,
    arg1?: unknown,
    arg2?: unknown,
    arg3?: unknown
  ) {
    const run = async () => {
      // ORT InferenceSession.create supports (path|buffer, options?) or (buffer, byteOffset, byteLength, options?)
      const sessionOptions = arg2 === undefined && arg3 === undefined ? arg1 : arg3;
      try {
        const session = await (
          origCreate as unknown as (...args: unknown[]) => Promise<InferenceSession>
        ).apply(this, [
          arg0,
          ...(arg1 !== undefined ? [arg1] : []),
          ...(arg2 !== undefined ? [arg2] : []),
          ...(arg3 !== undefined ? [arg3] : []),
        ]);
        hooks.onSessionCreated?.();
        return session;
      } finally {
        releaseSessionExternalData(sessionOptions);
        if (yieldMs > 0) {
          await new Promise<void>((resolve) => setTimeout(resolve, yieldMs));
        }
        coordinator.completeActiveGroup();
      }
    };

    const next = createTail.then(run, run);
    createTail = next.then(
      () => undefined,
      () => undefined
    );
    return next;
  } as unknown as typeof InferenceSession.create;

  target.create = wrappedCreate;
  try {
    return await fn();
  } catch (err) {
    coordinator.abortAll(err);
    throw err;
  } finally {
    target.create = origCreate;
    coordinator.abortAll();
  }
}
