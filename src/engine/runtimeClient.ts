import type {
  ChatPromptMessage,
  DeviceOption,
  ExecutionBackend,
  GenerationConfig,
  InferenceSpeedMetrics,
  ModelCacheInfo,
  ModelId,
  ModelLoadProgress,
  QuantDtype,
  WorkerDeviceTarget,
  WorkerRequest,
  WorkerResponse,
} from '../types.ts';

/**
 * Minimal Web Worker interface required by `LlmRuntimeClient`, enabling clean
 * dependency injection in unit tests without a browser environment.
 */
export interface WorkerLike {
  postMessage(message: unknown): void;
  terminate(): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<WorkerResponse>) => void): void;
  addEventListener(type: 'error', listener: (event: ErrorEvent) => void): void;
  addEventListener(type: 'messageerror', listener: (event: MessageEvent) => void): void;
  removeEventListener(type: 'message', listener: (event: MessageEvent<WorkerResponse>) => void): void;
  removeEventListener(type: 'error', listener: (event: ErrorEvent) => void): void;
  removeEventListener(type: 'messageerror', listener: (event: MessageEvent) => void): void;
}

export interface LlmRuntimeClientOptions {
  /** Custom worker factory (defaults to dedicated ES module `llm.worker.ts`). */
  workerFactory?: () => WorkerLike;
  /** Optional handler for unsolicited worker errors (e.g., idle WebGPU device loss). */
  onUnexpectedError?: (error: LlmRuntimeError) => void;
}

export interface LoadModelParams {
  modelId: ModelId;
  deviceTarget: WorkerDeviceTarget;
  dtype?: QuantDtype;
  useExternalDataFormat?: boolean;
}

export interface ModelLoadedResult {
  modelId: ModelId;
  device: DeviceOption;
  effectiveDtype: QuantDtype;
  useExternalDataFormat: boolean;
  loadTimeMs: number;
  fallbackNotice?: string;
}

export interface TokenStreamEvent {
  deltaText: string;
  fullText: string;
  tokenCount: number;
  timestampMs: number;
  isFirstToken: boolean;
}

export interface GenerateCallbacks {
  onStart?: (info: { promptTokens: number }) => void;
  onToken?: (event: TokenStreamEvent) => void;
}

export interface GenerateResult {
  fullText: string;
  thinking: string;
  content: string;
  aborted: boolean;
  speed: InferenceSpeedMetrics;
  workerJsHeapUsedBytes?: number;
  workerJsHeapTotalBytes?: number;
  activeThreads: number;
  backend: ExecutionBackend;
  dtype: QuantDtype;
}

export interface ClearCacheResult {
  modelId?: ModelId;
  clearedBytes: number;
  deletedEntries: number;
}

/**
 * Structured error surfaced from the LLM Web Worker or runtime client guards.
 */
export class LlmRuntimeError extends Error {
  readonly code?: string;
  readonly recoverable: boolean;

  constructor(message: string, code?: string, recoverable = true) {
    super(message);
    this.name = 'LlmRuntimeError';
    if (code !== undefined) {
      this.code = code;
    }
    this.recoverable = recoverable;
  }
}

type ExclusiveOperation = 'load-model' | 'generate' | 'unload-model';

interface PendingRequest {
  requestType: WorkerRequest['type'];
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  onProgress?: (progress: ModelLoadProgress) => void;
  onStart?: (info: { promptTokens: number }) => void;
  onToken?: (event: TokenStreamEvent) => void;
  onSettled?: () => void;
}

function defaultWorkerFactory(): WorkerLike {
  return new Worker(new URL('./llm.worker.ts', import.meta.url), {
    type: 'module',
  }) as unknown as WorkerLike;
}

/**
 * Main-thread async client wrapping the dedicated LLM Web Worker protocol.
 * Supports request correlation, streaming callbacks, exclusive operation guarding,
 * generation abort, automatic worker crash recovery, and clean disposal.
 */
export class LlmRuntimeClient {
  private readonly workerFactory: () => WorkerLike;
  private readonly onUnexpectedError?: (error: LlmRuntimeError) => void;
  private worker: WorkerLike | null = null;
  private requestCounter = 0;
  private disposed = false;
  private activeExclusiveOp: ExclusiveOperation | null = null;
  private activeGenerateRequestId: string | null = null;
  private readonly pending = new Map<string, PendingRequest>();

  private readonly handleMessage = (event: MessageEvent<WorkerResponse>): void => {
    const response = event?.data;
    if (!response || typeof response !== 'object' || typeof response.type !== 'string') {
      return;
    }

    const requestId = 'requestId' in response ? response.requestId : undefined;
    const entry = requestId ? this.pending.get(requestId) : undefined;

    if (!entry) {
      if (response.type === 'error') {
        this.onUnexpectedError?.(
          new LlmRuntimeError(
            response.message,
            response.code,
            response.recoverable ?? true
          )
        );
      }
      return;
    }

    switch (response.type) {
      case 'load-progress':
        entry.onProgress?.(response.progress);
        return;

      case 'generation-started':
        entry.onStart?.({ promptTokens: response.promptTokens });
        return;

      case 'token':
        entry.onToken?.({
          deltaText: response.deltaText,
          fullText: response.fullText,
          tokenCount: response.tokenCount,
          timestampMs: response.timestampMs,
          isFirstToken: response.isFirstToken,
        });
        return;

      case 'devices-probed':
        this.settlePending(requestId!, entry, () => entry.resolve(response.devices));
        return;

      case 'model-loaded': {
        const result: ModelLoadedResult = {
          modelId: response.modelId,
          device: response.device,
          effectiveDtype: response.effectiveDtype,
          useExternalDataFormat: response.useExternalDataFormat,
          loadTimeMs: response.loadTimeMs,
          ...(response.fallbackNotice !== undefined
            ? { fallbackNotice: response.fallbackNotice }
            : {}),
        };
        this.settlePending(requestId!, entry, () => entry.resolve(result));
        return;
      }

      case 'generation-complete': {
        const result: GenerateResult = {
          fullText: response.fullText,
          thinking: response.thinking,
          content: response.content,
          aborted: response.aborted,
          speed: response.speed,
          ...(response.workerJsHeapUsedBytes !== undefined
            ? { workerJsHeapUsedBytes: response.workerJsHeapUsedBytes }
            : {}),
          ...(response.workerJsHeapTotalBytes !== undefined
            ? { workerJsHeapTotalBytes: response.workerJsHeapTotalBytes }
            : {}),
          activeThreads: response.activeThreads,
          backend: response.backend,
          dtype: response.dtype,
        };
        this.settlePending(requestId!, entry, () => entry.resolve(result));
        return;
      }

      case 'model-unloaded':
        this.settlePending(requestId!, entry, () => entry.resolve(undefined));
        return;

      case 'cache-info':
        this.settlePending(requestId!, entry, () => entry.resolve(response.caches));
        return;

      case 'cache-cleared': {
        const result: ClearCacheResult = {
          ...(response.modelId !== undefined ? { modelId: response.modelId } : {}),
          clearedBytes: response.clearedBytes,
          deletedEntries: response.deletedEntries,
        };
        this.settlePending(requestId!, entry, () => entry.resolve(result));
        return;
      }

      case 'error':
        this.settlePending(requestId!, entry, () =>
          entry.reject(
            new LlmRuntimeError(
              response.message,
              response.code,
              response.recoverable ?? true
            )
          )
        );
        return;
    }
  };

  private readonly handleWorkerError = (event: ErrorEvent): void => {
    const message =
      event?.message && event.message.trim().length > 0
        ? `Web Worker がクラッシュしました: ${event.message}`
        : 'Web Worker で予期しないエラーが発生しました。次回の操作で自動復旧します。';
    const error = new LlmRuntimeError(message, 'WORKER_CRASHED', true);
    this.teardownWorkerAndFailPending(error);
  };

  private readonly handleMessageError = (): void => {
    const error = new LlmRuntimeError(
      'Web Worker とのメッセージシリアライズに失敗しました。',
      'WORKER_MESSAGE_ERROR',
      true
    );
    this.teardownWorkerAndFailPending(error);
  };

  constructor(options?: LlmRuntimeClientOptions) {
    this.workerFactory = options?.workerFactory ?? defaultWorkerFactory;
    this.onUnexpectedError = options?.onUnexpectedError;
  }

  /**
   * Whether `dispose()` has been called on this client instance.
   */
  get isDisposed(): boolean {
    return this.disposed;
  }

  /**
   * Currently active exclusive operation (`'load-model' | 'generate' | 'unload-model'`), if any.
   */
  get currentOperation(): ExclusiveOperation | null {
    return this.activeExclusiveOp;
  }

  /**
   * Probe available WebGPU adapters and CPU WASM capabilities inside the Worker.
   */
  probeDevices(): Promise<DeviceOption[]> {
    return this.sendRequest<DeviceOption[]>({
      type: 'probe-devices',
      requestId: this.nextRequestId('probe'),
    });
  }

  /**
   * Load the specified model & tokenizer onto the target WebGPU or WASM device.
   */
  loadModel(
    params: LoadModelParams,
    onProgress?: (progress: ModelLoadProgress) => void
  ): Promise<ModelLoadedResult> {
    return this.sendExclusiveRequest<ModelLoadedResult>(
      'load-model',
      {
        type: 'load-model',
        requestId: this.nextRequestId('load'),
        modelId: params.modelId,
        deviceTarget: params.deviceTarget,
        ...(params.dtype !== undefined ? { dtype: params.dtype } : {}),
        ...(params.useExternalDataFormat !== undefined
          ? { useExternalDataFormat: params.useExternalDataFormat }
          : {}),
      },
      onProgress ? { onProgress } : undefined
    );
  }

  /**
   * Stream chat completion tokens and compute inference speed metrics in the Worker.
   */
  generate(
    messages: ChatPromptMessage[],
    config: GenerationConfig,
    callbacks?: GenerateCallbacks
  ): Promise<GenerateResult> {
    const requestId = this.nextRequestId('gen');
    return this.sendExclusiveRequest<GenerateResult>(
      'generate',
      {
        type: 'generate',
        requestId,
        messages,
        config,
      },
      {
        ...(callbacks?.onStart ? { onStart: callbacks.onStart } : {}),
        ...(callbacks?.onToken ? { onToken: callbacks.onToken } : {}),
      }
    );
  }

  /**
   * Request interruption of the active text generation turn (if any).
   */
  abort(): void {
    if (this.disposed || !this.worker || !this.activeGenerateRequestId) {
      return;
    }

    const request: WorkerRequest = {
      type: 'abort',
      requestId: this.nextRequestId('abort'),
      targetRequestId: this.activeGenerateRequestId,
    };

    try {
      this.worker.postMessage(request);
    } catch {
      // Ignore postMessage failure if worker already crashed/terminated.
    }
  }

  /**
   * Unload the current model sessions and destroy any active WebGPU device.
   */
  unloadModel(): Promise<void> {
    return this.sendExclusiveRequest<void>('unload-model', {
      type: 'unload-model',
      requestId: this.nextRequestId('unload'),
    });
  }

  /**
   * Inspect Browser CacheStorage (`transformers-cache`) entries for catalog models.
   */
  inspectCache(modelIds?: ModelId[]): Promise<ModelCacheInfo[]> {
    return this.sendRequest<ModelCacheInfo[]>({
      type: 'inspect-cache',
      requestId: this.nextRequestId('cache-info'),
      ...(modelIds !== undefined ? { modelIds } : {}),
    });
  }

  /**
   * Clear Browser CacheStorage entries for a specific model or the entire cache.
   */
  clearCache(modelId?: ModelId): Promise<ClearCacheResult> {
    if (this.activeExclusiveOp === 'load-model') {
      return Promise.reject(
        new LlmRuntimeError(
          'モデルロード中にキャッシュを削除することはできません。ロード完了後または中断後に再実行してください。',
          'CONCURRENT_OPERATION',
          true
        )
      );
    }
    return this.sendRequest<ClearCacheResult>({
      type: 'clear-cache',
      requestId: this.nextRequestId('cache-clear'),
      ...(modelId !== undefined ? { modelId } : {}),
    });
  }

  /**
   * Terminate the underlying Web Worker, remove event listeners, and reject pending calls.
   */
  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.teardownWorkerAndFailPending(
      new LlmRuntimeError('LlmRuntimeClient は破棄されました。', 'CLIENT_DISPOSED', false)
    );
  }

  private nextRequestId(prefix: string): string {
    this.requestCounter += 1;
    return `${prefix}-${this.requestCounter}`;
  }

  private ensureWorker(): WorkerLike {
    if (this.disposed) {
      throw new LlmRuntimeError(
        '破棄済みの LlmRuntimeClient は使用できません。',
        'CLIENT_DISPOSED',
        false
      );
    }

    if (this.worker) {
      return this.worker;
    }

    const created = this.workerFactory();
    created.addEventListener('message', this.handleMessage);
    created.addEventListener('error', this.handleWorkerError);
    created.addEventListener('messageerror', this.handleMessageError);
    this.worker = created;
    return created;
  }

  private detachAndTerminateWorker(): void {
    const activeWorker = this.worker;
    if (!activeWorker) {
      return;
    }
    this.worker = null;
    try {
      activeWorker.removeEventListener('message', this.handleMessage);
      activeWorker.removeEventListener('error', this.handleWorkerError);
      activeWorker.removeEventListener('messageerror', this.handleMessageError);
    } catch {
      // Ignore listener removal errors on mock or terminated workers.
    }
    try {
      activeWorker.terminate();
    } catch {
      // Ignore termination errors.
    }
  }

  private teardownWorkerAndFailPending(error: LlmRuntimeError): void {
    this.detachAndTerminateWorker();
    this.activeExclusiveOp = null;
    this.activeGenerateRequestId = null;

    const entries = Array.from(this.pending.entries());
    this.pending.clear();
    for (const [, entry] of entries) {
      try {
        entry.onSettled?.();
      } catch {
        // Ignore cleanup callback errors.
      }
      entry.reject(error);
    }
  }

  private settlePending(requestId: string, entry: PendingRequest, finish: () => void): void {
    this.pending.delete(requestId);
    try {
      entry.onSettled?.();
    } finally {
      finish();
    }
  }

  private sendExclusiveRequest<T>(
    operation: ExclusiveOperation,
    request: WorkerRequest,
    callbacks?: Pick<PendingRequest, 'onProgress' | 'onStart' | 'onToken'>
  ): Promise<T> {
    if (this.disposed) {
      return Promise.reject(
        new LlmRuntimeError(
          '破棄済みの LlmRuntimeClient は使用できません。',
          'CLIENT_DISPOSED',
          false
        )
      );
    }

    if (this.activeExclusiveOp !== null) {
      return Promise.reject(
        new LlmRuntimeError(
          `別の処理 (${this.activeExclusiveOp}) が実行中のため ${operation} を開始できません。`,
          'CONCURRENT_OPERATION',
          true
        )
      );
    }

    this.activeExclusiveOp = operation;
    if (operation === 'generate') {
      this.activeGenerateRequestId = request.requestId;
    }

    return this.sendRequest<T>(request, {
      ...callbacks,
      onSettled: () => {
        if (this.activeExclusiveOp === operation) {
          this.activeExclusiveOp = null;
        }
        if (operation === 'generate' && this.activeGenerateRequestId === request.requestId) {
          this.activeGenerateRequestId = null;
        }
      },
    });
  }

  private sendRequest<T>(
    request: WorkerRequest,
    hooks?: Pick<PendingRequest, 'onProgress' | 'onStart' | 'onToken' | 'onSettled'>
  ): Promise<T> {
    if (this.disposed) {
      hooks?.onSettled?.();
      return Promise.reject(
        new LlmRuntimeError(
          '破棄済みの LlmRuntimeClient は使用できません。',
          'CLIENT_DISPOSED',
          false
        )
      );
    }

    let worker: WorkerLike;
    try {
      worker = this.ensureWorker();
    } catch (err) {
      hooks?.onSettled?.();
      return Promise.reject(err);
    }

    return new Promise<T>((resolve, reject) => {
      const pendingEntry: PendingRequest = {
        requestType: request.type,
        resolve: resolve as (value: unknown) => void,
        reject,
        ...(hooks?.onProgress ? { onProgress: hooks.onProgress } : {}),
        ...(hooks?.onStart ? { onStart: hooks.onStart } : {}),
        ...(hooks?.onToken ? { onToken: hooks.onToken } : {}),
        ...(hooks?.onSettled ? { onSettled: hooks.onSettled } : {}),
      };

      this.pending.set(request.requestId, pendingEntry);

      try {
        worker.postMessage(request);
      } catch (err) {
        this.pending.delete(request.requestId);
        hooks?.onSettled?.();
        const message =
          err instanceof Error ? err.message : 'Web Worker へのメッセージ送信に失敗しました。';
        reject(new LlmRuntimeError(message, 'WORKER_POST_FAILED', true));
      }
    });
  }
}
