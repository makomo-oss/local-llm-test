/**
 * Scoped WebGPU weight-upload optimizer for ONNX Runtime Web sessions.
 *
 * ORT WebGPU (`g.pe` in `ort-wasm`) uploads external tensor initializers
 * synchronously by calling `device.createBuffer({ mappedAtCreation: true,
 * usage: MAP_WRITE | COPY_SRC })` (~900+ times for Gemma 4 E2B, ~1.45 GiB total)
 * before draining or submitting. On mobile unified-memory GPUs this creates a
 * massive Dawn shared-memory transfer spike that can crash tabs or lose the
 * GPU device.
 *
 * Intercepting ONLY ORT's unlabeled `g.pe` staging buffers while
 * `InferenceSession.create()` runs lets us back them with a single reusable CPU
 * `ArrayBuffer` and stream each tensor into the real `COPY_DST` destination
 * buffer in bounded `4 MiB` `GPUQueue.writeBuffer()` chunks, slashing peak GPU
 * process memory without affecting any labeled C++ WebGPU buffers or runtime
 * inference dispatches.
 */

import {
  DEFAULT_RETAINED_SCRATCH_MAX_BYTES,
  type ExternalDataScratchPool,
} from './externalDataProxy.ts';

const WEBGPU_MAP_WRITE_COPY_SRC = 6; // GPUBufferUsage.MAP_WRITE (2) | GPUBufferUsage.COPY_SRC (4)
const DEFAULT_WRITE_CHUNK_BYTES = 4 * 1024 * 1024;
const MIN_ZERO_COPY_HANDOFF_BYTES = 4096;

export interface WebGpuUploadOptimizerStats {
  interceptCount: number;
  chunkWriteCount: number;
  bytesUploaded: number;
  peakScratchBytes: number;
}

export interface WebGpuUploadOptimizerHandle {
  dispose(): void;
  getStats(): WebGpuUploadOptimizerStats;
}

export interface WebGpuUploadOptimizerOptions {
  /** Maximum chunk size per `queue.writeBuffer` call (aligned down to a multiple of 4). */
  chunkBytes?: number;
  /** Optional external-data scratch pool to share without double 192 MiB copies. */
  sharedScratchPool?: ExternalDataScratchPool | null;
  /** Threshold above which temporary upload scratch slabs are released right after copy. */
  maxRetainedScratchBytes?: number;
}

interface OptimizableCommandEncoder {
  copyBufferToBuffer(
    source: unknown,
    sourceOffset: number,
    destination: unknown,
    destinationOffset: number,
    size: number
  ): void;
  finish?(descriptor?: unknown): unknown;
}

interface OptimizableQueue {
  writeBuffer(
    buffer: unknown,
    bufferOffset: number,
    data: ArrayBufferView | ArrayBuffer,
    dataOffset?: number,
    size?: number
  ): void;
  submit?(commandBuffers: Iterable<unknown>): void;
}

export interface OptimizableGpuDevice {
  createBuffer(descriptor: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
    label?: string;
  }): unknown;
  createCommandEncoder(descriptor?: unknown): OptimizableCommandEncoder;
  queue: OptimizableQueue;
}

interface ActiveVirtualStaging {
  token: Record<string, unknown>;
  size: number;
  mapped: boolean;
  unmapped: boolean;
  handoffBuffer: ArrayBuffer | null;
  handoffView: Uint8Array | null;
}

export function installWebGpuUploadOptimizer(
  device: OptimizableGpuDevice | GPUDevice | null | undefined,
  options: WebGpuUploadOptimizerOptions = {}
): WebGpuUploadOptimizerHandle {
  const stats: WebGpuUploadOptimizerStats = {
    interceptCount: 0,
    chunkWriteCount: 0,
    bytesUploaded: 0,
    peakScratchBytes: 0,
  };

  if (
    !device ||
    typeof device.createBuffer !== 'function' ||
    typeof device.createCommandEncoder !== 'function' ||
    !device.queue ||
    typeof device.queue.writeBuffer !== 'function'
  ) {
    return {
      dispose(): void {},
      getStats(): WebGpuUploadOptimizerStats {
        return { ...stats };
      },
    };
  }

  const target = device as OptimizableGpuDevice;
  const rawChunkBytes = options.chunkBytes ?? DEFAULT_WRITE_CHUNK_BYTES;
  const chunkBytes = Math.max(4, Math.floor(rawChunkBytes / 4) * 4);
  const sharedScratchPool = options.sharedScratchPool ?? null;
  const maxRetainedScratchBytes = Math.max(
    4096,
    options.maxRetainedScratchBytes ?? DEFAULT_RETAINED_SCRATCH_MAX_BYTES
  );

  const origCreateBuffer = target.createBuffer;
  const origCreateCommandEncoder = target.createCommandEncoder;
  const origWriteBuffer = target.queue.writeBuffer;
  const origSubmit =
    typeof target.queue.submit === 'function' ? target.queue.submit : null;
  const noopCommandBuffers = new WeakSet<object>();

  let scratchBuffer: ArrayBuffer | null = null;
  let scratchView: Uint8Array | null = null;
  let activeStaging: ActiveVirtualStaging | null = null;
  let disposed = false;

  const trimOversizedScratch = (byteLength: number): void => {
    if (byteLength <= maxRetainedScratchBytes) {
      return;
    }
    sharedScratchPool?.trimIfLargerThan(maxRetainedScratchBytes);
    if (scratchBuffer && scratchBuffer.byteLength > maxRetainedScratchBytes) {
      scratchView = null;
      scratchBuffer = null;
    }
  };

  if (origSubmit) {
    target.queue.submit = function patchedSubmit(
      this: unknown,
      commandBuffers: Iterable<unknown>
    ): void {
      if (disposed) {
        origSubmit.call(this, commandBuffers);
        return;
      }
      const realCommandBuffers: unknown[] = [];
      for (const cmdBuf of commandBuffers) {
        if (!cmdBuf || typeof cmdBuf !== 'object' || !noopCommandBuffers.has(cmdBuf)) {
          realCommandBuffers.push(cmdBuf);
        }
      }
      if (realCommandBuffers.length === 0) {
        return;
      }
      origSubmit.call(this, realCommandBuffers);
    };
  }

  const ensureScratch = (requiredBytes: number): Uint8Array => {
    if (!scratchBuffer || scratchBuffer.byteLength < requiredBytes) {
      // Release old buffer reference before allocating a larger one so both do not overlap.
      scratchBuffer = null;
      scratchView = null;
      scratchBuffer = new ArrayBuffer(requiredBytes);
      scratchView = new Uint8Array(scratchBuffer);
      if (requiredBytes > stats.peakScratchBytes) {
        stats.peakScratchBytes = requiredBytes;
      }
    }
    return scratchView!;
  };

  target.createBuffer = function patchedCreateBuffer(
    this: unknown,
    descriptor: {
      size: number;
      usage: number;
      mappedAtCreation?: boolean;
      label?: string;
    }
  ): unknown {
    // Intercept strictly single-at-a-time unlabeled mappedAtCreation(MAP_WRITE | COPY_SRC)
    // staging buffers emitted by ORT's EM_JS `g.pe` helper. Labeled buffers and any
    // concurrent/overlapping staging requests pass untouched to the native device.
    if (
      !disposed &&
      activeStaging === null &&
      descriptor &&
      typeof descriptor === 'object' &&
      descriptor.mappedAtCreation === true &&
      descriptor.usage === WEBGPU_MAP_WRITE_COPY_SRC &&
      !Object.prototype.hasOwnProperty.call(descriptor, 'label') &&
      typeof descriptor.size === 'number' &&
      Number.isFinite(descriptor.size) &&
      descriptor.size > 0 &&
      descriptor.size % 4 === 0
    ) {
      const alignedSize = descriptor.size;
      stats.interceptCount += 1;

      const stagingState: ActiveVirtualStaging = {
        token: {},
        size: alignedSize,
        mapped: false,
        unmapped: false,
        handoffBuffer: null,
        handoffView: null,
      };

      stagingState.token = {
        size: alignedSize,
        usage: WEBGPU_MAP_WRITE_COPY_SRC,
        mapState: 'mapped',
        label: '',
        getMappedRange(offset = 0, rangeSize?: number): ArrayBuffer {
          if (disposed || activeStaging !== stagingState || stagingState.unmapped) {
            throw new Error('Virtual WebGPU staging buffer is no longer mapped.');
          }
          if (offset !== 0 || (rangeSize !== undefined && (rangeSize <= 0 || rangeSize > alignedSize))) {
            throw new RangeError(
              `Unsupported getMappedRange(${offset}, ${String(rangeSize)}) on ${alignedSize}-byte virtual staging buffer.`
            );
          }
          if (stagingState.handoffBuffer) {
            stagingState.mapped = true;
            return stagingState.handoffBuffer;
          }
          if (
            sharedScratchPool &&
            (rangeSize === undefined || rangeSize === alignedSize) &&
            alignedSize > MIN_ZERO_COPY_HANDOFF_BYTES
          ) {
            const handoffBuf = sharedScratchPool.consumeHandoff(alignedSize);
            if (handoffBuf) {
              stagingState.handoffBuffer = handoffBuf;
              stagingState.handoffView = new Uint8Array(handoffBuf, 0, alignedSize);
              stagingState.mapped = true;
              if (alignedSize > stats.peakScratchBytes) {
                stats.peakScratchBytes = alignedSize;
              }
              return handoffBuf;
            }
          }

          const view = ensureScratch(alignedSize);
          // Zero trailing 16-byte alignment padding so stale bytes from a previous larger
          // tensor cannot leak into `[byteLength .. alignedSize)`.
          if (alignedSize <= 4096) {
            view.fill(0, 0, alignedSize);
          } else {
            view.fill(0, alignedSize - 16, alignedSize);
          }
          stagingState.mapped = true;
          return scratchBuffer!;
        },
        unmap(): void {
          if (activeStaging === stagingState) {
            stagingState.unmapped = true;
            stagingState.token.mapState = 'unmapped';
          }
        },
        destroy(): void {
          stagingState.token.mapState = 'destroyed';
          if (activeStaging === stagingState) {
            activeStaging = null;
          }
          stagingState.handoffView = null;
          stagingState.handoffBuffer = null;
          trimOversizedScratch(stagingState.size);
        },
      };

      activeStaging = stagingState;
      return stagingState.token;
    }

    return origCreateBuffer.call(this, descriptor);
  };

  target.createCommandEncoder = function patchedCreateCommandEncoder(
    this: unknown,
    descriptor?: unknown
  ): OptimizableCommandEncoder {
    if (disposed || !activeStaging) {
      return origCreateCommandEncoder.call(this, descriptor);
    }

    let nativeEncoder: (OptimizableCommandEncoder & Record<string, unknown>) | null = null;
    const ensureNativeEncoder = (): OptimizableCommandEncoder & Record<string, unknown> => {
      if (!nativeEncoder) {
        nativeEncoder = origCreateCommandEncoder.call(target, descriptor) as
          OptimizableCommandEncoder & Record<string, unknown>;
      }
      return nativeEncoder;
    };

    const virtualEncoder = {
      copyBufferToBuffer(
        source: unknown,
        sourceOffset: number,
        destination: unknown,
        destinationOffset: number,
        size: number
      ): void {
        if (activeStaging && source === activeStaging.token) {
          const staging = activeStaging;
          if (!staging.unmapped) {
            throw new Error(
              'Virtual WebGPU staging buffer must be unmapped before copyBufferToBuffer.'
            );
          }
          if (
            sourceOffset < 0 ||
            size < 0 ||
            sourceOffset + size > staging.size ||
            sourceOffset % 4 !== 0 ||
            destinationOffset % 4 !== 0 ||
            size % 4 !== 0
          ) {
            throw new RangeError(
              `Invalid virtual copyBufferToBuffer bounds: srcOffset=${sourceOffset}, dstOffset=${destinationOffset}, size=${size}, bufSize=${staging.size}.`
            );
          }

          const view =
            staging.handoffView ??
            (staging.mapped && scratchView ? scratchView : ensureScratch(staging.size));
          for (let off = 0; off < size; off += chunkBytes) {
            const len = Math.min(chunkBytes, size - off);
            origWriteBuffer.call(
              target.queue,
              destination,
              destinationOffset + off,
              view,
              sourceOffset + off,
              len
            );
            stats.chunkWriteCount += 1;
            stats.bytesUploaded += len;
          }
          staging.handoffView = null;
          staging.handoffBuffer = null;
          activeStaging = null;
          trimOversizedScratch(staging.size);
          return;
        }

        const native = ensureNativeEncoder();
        native.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size);
      },
      finish(finishDescriptor?: unknown): unknown {
        if (!nativeEncoder) {
          const noopToken = {
            label:
              descriptor && typeof descriptor === 'object' && 'label' in descriptor
                ? String((descriptor as { label?: unknown }).label ?? '')
                : '',
          };
          noopCommandBuffers.add(noopToken);
          return noopToken;
        }
        return typeof nativeEncoder.finish === 'function'
          ? nativeEncoder.finish(finishDescriptor)
          : undefined;
      },
    };

    return new Proxy(virtualEncoder, {
      get(proxyTarget, prop, receiver) {
        if (prop in proxyTarget) {
          return Reflect.get(proxyTarget, prop, receiver);
        }
        const native = ensureNativeEncoder();
        const val = Reflect.get(native, prop, native);
        return typeof val === 'function' ? val.bind(native) : val;
      },
      set(_proxyTarget, prop, value) {
        const native = ensureNativeEncoder();
        return Reflect.set(native, prop, value, native);
      },
    }) as unknown as OptimizableCommandEncoder;
  };

  return {
    dispose(): void {
      if (disposed) {
        return;
      }
      disposed = true;
      target.createBuffer = origCreateBuffer;
      target.createCommandEncoder = origCreateCommandEncoder;
      if (origSubmit) {
        target.queue.submit = origSubmit;
      }
      activeStaging = null;
      scratchView = null;
      scratchBuffer = null;
    },
    getStats(): WebGpuUploadOptimizerStats {
      return { ...stats };
    },
  };
}
