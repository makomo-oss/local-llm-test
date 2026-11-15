import { describe, expect, it, vi } from 'vitest';
import {
  installWebGpuUploadOptimizer,
  type OptimizableGpuDevice,
} from '../src/engine/webgpuUploadOptimizer.ts';

interface RecordedCopy {
  source: unknown;
  sourceOffset: number;
  destination: unknown;
  destinationOffset: number;
  size: number;
}

interface RecordedWrite {
  buffer: unknown;
  bufferOffset: number;
  bytes: number[];
}

function createMockGpuDevice(options: { failWrite?: boolean } = {}) {
  const nativeBuffers: Array<Record<string, unknown>> = [];
  const nativeCopies: RecordedCopy[] = [];
  const queueWrites: RecordedWrite[] = [];

  const createBufferFn = vi.fn((descriptor: { size: number; usage: number; mappedAtCreation?: boolean; label?: string }) => {
    const buf: Record<string, unknown> = {
      kind: 'native-buffer',
      size: descriptor.size,
      usage: descriptor.usage,
      label: descriptor.label ?? '',
      mappedAtCreation: Boolean(descriptor.mappedAtCreation),
      getMappedRange: () => new ArrayBuffer(descriptor.size),
      unmap: vi.fn(),
      destroy: vi.fn(),
    };
    nativeBuffers.push(buf);
    return buf;
  });

  const createCommandEncoderFn = vi.fn(() => ({
    copyBufferToBuffer(
      source: unknown,
      sourceOffset: number,
      destination: unknown,
      destinationOffset: number,
      size: number
    ): void {
      nativeCopies.push({ source, sourceOffset, destination, destinationOffset, size });
    },
    finish: vi.fn(() => ({ kind: 'cmd-buf' })),
  }));

  const writeBufferFn = vi.fn(
    (
      buffer: unknown,
      bufferOffset: number,
      data: ArrayBufferView | ArrayBuffer,
      dataOffset = 0,
      size?: number
    ) => {
      if (options.failWrite) {
        throw new Error('writeBuffer failure');
      }
      const raw =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const start = data instanceof ArrayBuffer ? dataOffset : dataOffset;
      const len = size ?? raw.byteLength - start;
      queueWrites.push({
        buffer,
        bufferOffset,
        bytes: Array.from(raw.subarray(start, start + len)),
      });
    }
  );

  const device: OptimizableGpuDevice = {
    createBuffer: createBufferFn,
    createCommandEncoder: createCommandEncoderFn,
    queue: {
      writeBuffer: writeBufferFn,
    },
  };

  return {
    device,
    createBufferFn,
    createCommandEncoderFn,
    writeBufferFn,
    nativeBuffers,
    nativeCopies,
    queueWrites,
  };
}

describe('webgpuUploadOptimizer', () => {
  it('returns safe no-op handle when device is null or missing WebGPU methods', () => {
    const handle1 = installWebGpuUploadOptimizer(null);
    expect(handle1.getStats()).toEqual({
      interceptCount: 0,
      chunkWriteCount: 0,
      bytesUploaded: 0,
      peakScratchBytes: 0,
    });
    handle1.dispose();

    const partial = { createBuffer: vi.fn() } as unknown as OptimizableGpuDevice;
    const handle2 = installWebGpuUploadOptimizer(partial);
    expect(handle2.getStats().interceptCount).toBe(0);
    handle2.dispose();
  });

  it('intercepts unlabeled g.pe staging buffers, streams chunked writeBuffer uploads, and restores on dispose', () => {
    const mock = createMockGpuDevice();
    const handle = installWebGpuUploadOptimizer(mock.device, { chunkBytes: 8 });

    // Real destination buffer (labeled + usage != 6)
    const dstBuf = mock.device.createBuffer({
      size: 20,
      usage: 0x80 | 0x08,
      label: 'tensor-dst',
    });
    expect(mock.nativeBuffers).toHaveLength(1);

    // Unlabeled g.pe staging buffer of 20 bytes -> chunked into 8 + 8 + 4 bytes
    const staging = mock.device.createBuffer({
      mappedAtCreation: true,
      size: 20,
      usage: 6,
    }) as {
      getMappedRange(): ArrayBuffer;
      unmap(): void;
      destroy(): void;
    };
    const ab1 = staging.getMappedRange();
    const u8_1 = new Uint8Array(ab1);
    for (let i = 0; i < 18; i += 1) {
      u8_1[i] = i + 1;
    }
    staging.unmap();

    const enc1 = mock.device.createCommandEncoder();
    enc1.copyBufferToBuffer(staging, 0, dstBuf, 4, 20);
    staging.destroy();

    expect(mock.nativeBuffers).toHaveLength(1);
    expect(mock.nativeCopies).toHaveLength(0);
    expect(mock.queueWrites).toEqual([
      { buffer: dstBuf, bufferOffset: 4, bytes: [1, 2, 3, 4, 5, 6, 7, 8] },
      { buffer: dstBuf, bufferOffset: 12, bytes: [9, 10, 11, 12, 13, 14, 15, 16] },
      { buffer: dstBuf, bufferOffset: 20, bytes: [17, 18, 0, 0] },
    ]);

    // Second smaller 12-byte upload reuses same backing ArrayBuffer without re-allocating or leaking tail bytes
    const staging2 = mock.device.createBuffer({
      mappedAtCreation: true,
      size: 12,
      usage: 6,
    }) as {
      getMappedRange(): ArrayBuffer;
      unmap(): void;
      destroy(): void;
    };
    const ab2 = staging2.getMappedRange();
    expect(ab2).toBe(ab1);
    new Uint8Array(ab2).set([99, 98, 97, 96, 95, 94, 93, 92, 91], 0);
    staging2.unmap();
    mock.device.createCommandEncoder().copyBufferToBuffer(staging2, 0, dstBuf, 0, 12);
    staging2.destroy();

    expect(mock.queueWrites.slice(3)).toEqual([
      { buffer: dstBuf, bufferOffset: 0, bytes: [99, 98, 97, 96, 95, 94, 93, 92] },
      { buffer: dstBuf, bufferOffset: 8, bytes: [91, 0, 0, 0] },
    ]);

    expect(handle.getStats()).toEqual({
      interceptCount: 2,
      chunkWriteCount: 5,
      bytesUploaded: 32,
      peakScratchBytes: 20,
    });

    handle.dispose();
    // Idempotent second dispose
    handle.dispose();
    expect(mock.device.createBuffer).toBe(mock.createBufferFn);
    expect(mock.device.createCommandEncoder).toBe(mock.createCommandEncoderFn);
  });

  it('passes through labeled, non-staging, 0-byte, unaligned, or overlapping concurrent buffers to native device', () => {
    const mock = createMockGpuDevice();
    const handle = installWebGpuUploadOptimizer(mock.device, { chunkBytes: 8 });

    // Labeled mappedAtCreation buffer -> native
    mock.device.createBuffer({ mappedAtCreation: true, size: 16, usage: 6, label: 'cpp-buf' });
    // 0-byte or unaligned -> native
    mock.device.createBuffer({ mappedAtCreation: true, size: 0, usage: 6 });
    mock.device.createBuffer({ mappedAtCreation: true, size: 6, usage: 6 });
    // Different usage -> native
    mock.device.createBuffer({ mappedAtCreation: true, size: 16, usage: 10 });
    expect(mock.nativeBuffers).toHaveLength(4);
    expect(handle.getStats().interceptCount).toBe(0);

    // Concurrent second virtual request while first is active falls back safely to native
    const virtual1 = mock.device.createBuffer({ mappedAtCreation: true, size: 16, usage: 6 }) as {
      getMappedRange(): ArrayBuffer;
      unmap(): void;
      destroy(): void;
    };
    const nativeOverlap = mock.device.createBuffer({ mappedAtCreation: true, size: 16, usage: 6 });
    expect(mock.nativeBuffers).toHaveLength(5);
    expect((nativeOverlap as { kind?: string }).kind).toBe('native-buffer');

    // Copying a native buffer still invokes native copyBufferToBuffer
    const enc = mock.device.createCommandEncoder();
    const dst = mock.device.createBuffer({ size: 16, usage: 8, label: 'dst' });
    enc.copyBufferToBuffer(nativeOverlap, 0, dst, 0, 16);
    expect(mock.nativeCopies).toHaveLength(1);

    // Destroying virtual1 without copying releases slot for future uploads
    virtual1.destroy();
    const virtual2 = mock.device.createBuffer({ mappedAtCreation: true, size: 8, usage: 6 }) as {
      getMappedRange(): ArrayBuffer;
      unmap(): void;
      destroy(): void;
    };
    expect(handle.getStats().interceptCount).toBe(2);
    virtual2.destroy();
    handle.dispose();
  });

  it('validates unmapped copy, bad alignment, and out-of-range getMappedRange cleanly', () => {
    const mock = createMockGpuDevice();
    const handle = installWebGpuUploadOptimizer(mock.device);
    const dst = mock.device.createBuffer({ size: 32, usage: 8, label: 'dst' });

    const staging = mock.device.createBuffer({ mappedAtCreation: true, size: 16, usage: 6 }) as {
      getMappedRange(offset?: number, size?: number): ArrayBuffer;
      unmap(): void;
      destroy(): void;
    };
    expect(() => staging.getMappedRange(4, 8)).toThrow(/Unsupported getMappedRange/);
    staging.getMappedRange();

    const enc = mock.device.createCommandEncoder();
    expect(() => enc.copyBufferToBuffer(staging, 0, dst, 0, 16)).toThrow(/must be unmapped/);

    staging.unmap();
    expect(() => enc.copyBufferToBuffer(staging, 2, dst, 0, 8)).toThrow(/copyBufferToBuffer bounds/);
    expect(() => enc.copyBufferToBuffer(staging, 0, dst, 0, 32)).toThrow(/copyBufferToBuffer bounds/);

    staging.destroy();
    handle.dispose();
  });

  it('captures synchronous writeBuffer, submit, and createBuffer errors via getLastError and onUploadError', () => {
    const errors: unknown[] = [];
    const mock = createMockGpuDevice({ failWrite: true });
    const submitMock = vi.fn(() => {
      throw new Error('submit failure: device is lost');
    });
    const origCreateBuffer = mock.device.createBuffer;
    let throwOnNativeCreate = false;
    mock.device.queue.submit = submitMock;
    mock.device.createBuffer = (desc) => {
      if (throwOnNativeCreate) {
        throw new Error('native createBuffer OOM');
      }
      return origCreateBuffer.call(mock.device, desc);
    };

    const handle = installWebGpuUploadOptimizer(mock.device, {
      onUploadError: (err) => errors.push(err),
    });
    expect(handle.getLastError()).toBeNull();
    expect(handle.getStats().lastError).toBeUndefined();

    // 1. writeBuffer error captured during virtual copyBufferToBuffer
    const dst = mock.device.createBuffer({ size: 16, usage: 8, label: 'dst' });
    const staging = mock.device.createBuffer({ mappedAtCreation: true, size: 16, usage: 6 }) as {
      getMappedRange(): ArrayBuffer;
      unmap(): void;
      destroy(): void;
    };
    staging.getMappedRange();
    staging.unmap();
    const enc = mock.device.createCommandEncoder();
    expect(() => enc.copyBufferToBuffer!(staging, 0, dst, 0, 16)).toThrow('writeBuffer failure');
    expect((handle.getLastError() as Error).message).toBe('writeBuffer failure');
    expect(handle.getStats().lastError).toContain('writeBuffer failure');
    staging.destroy();

    // 2. Native submit error captured
    expect(() => mock.device.queue.submit!([{ kind: 'real-cmd' }])).toThrow(
      'submit failure: device is lost'
    );
    expect((handle.getLastError() as Error).message).toContain('device is lost');

    // 3. Native createBuffer error captured
    throwOnNativeCreate = true;
    expect(() => mock.device.createBuffer({ size: 64, usage: 8, label: 'weights' })).toThrow(
      'native createBuffer OOM'
    );
    expect((handle.getLastError() as Error).message).toBe('native createBuffer OOM');
    expect(errors).toHaveLength(3);

    handle.dispose();
  });

  it('supports elideEmptyCommandEncoders: false without Proxy/submit mutation and captures virtual staging errors', () => {
    const mock = createMockGpuDevice();
    const realFinishSpy = vi.fn(() => ({ kind: 'native-empty-cmd' }));
    const submitSpy = vi.fn();
    mock.device.createCommandEncoder = vi.fn(() => ({
      copyBufferToBuffer: vi.fn(),
      finish: realFinishSpy,
    }));
    mock.device.queue.submit = submitSpy;

    const handle = installWebGpuUploadOptimizer(mock.device, {
      chunkBytes: 8,
      elideEmptyCommandEncoders: false,
    });
    const dst = mock.device.createBuffer({ size: 16, usage: 8, label: 'dst' });
    const staging = mock.device.createBuffer({ mappedAtCreation: true, size: 16, usage: 6 }) as {
      getMappedRange(offset?: number, size?: number): ArrayBuffer;
      unmap(): void;
      destroy(): void;
    };
    const mapped = staging.getMappedRange();
    new Uint8Array(mapped).set([1, 2, 3, 4]);
    staging.unmap();
    const enc = mock.device.createCommandEncoder();
    enc.copyBufferToBuffer(staging, 0, dst, 0, 16);
    const cmd = (enc as unknown as { finish(): unknown }).finish();
    expect(realFinishSpy).toHaveBeenCalledTimes(1);
    expect(cmd).toEqual({ kind: 'native-empty-cmd' });
    mock.device.queue.submit([cmd]);
    expect(submitSpy).toHaveBeenCalledWith([{ kind: 'native-empty-cmd' }]);
    staging.destroy();

    // Virtual validation errors (unmapped copy / out-of-range getMappedRange) must also be recorded
    const badStaging = mock.device.createBuffer({ mappedAtCreation: true, size: 16, usage: 6 }) as {
      getMappedRange(offset?: number, size?: number): ArrayBuffer;
      unmap(): void;
      destroy(): void;
    };
    expect(() => badStaging.getMappedRange(4, 8)).toThrow(/Unsupported getMappedRange/);
    expect((handle.getLastError() as Error).message).toContain('Unsupported getMappedRange');

    badStaging.getMappedRange();
    const badEnc = mock.device.createCommandEncoder();
    expect(() => badEnc.copyBufferToBuffer(badStaging, 0, dst, 0, 16)).toThrow(/must be unmapped/);
    expect((handle.getLastError() as Error).message).toContain('must be unmapped');
    badStaging.destroy();

    handle.dispose();
  });

  it('supports useVirtualStaging: false passthrough diagnostics and non-writable queue.submit hosts', () => {
    const mock = createMockGpuDevice();
    let failNativeMap = false;
    const origCreate = mock.device.createBuffer;
    mock.device.createBuffer = (desc) => {
      const buf = origCreate.call(mock.device, desc) as Record<string, unknown>;
      if (desc.mappedAtCreation) {
        buf.getMappedRange = () => {
          if (failNativeMap) {
            throw new Error('native Dawn OOM on getMappedRange');
          }
          return new ArrayBuffer(desc.size);
        };
      }
      return buf;
    };
    // Simulate a frozen / non-writable queue.submit property
    Object.defineProperty(mock.device.queue, 'submit', {
      value: vi.fn(),
      writable: false,
      configurable: false,
    });

    const handle = installWebGpuUploadOptimizer(mock.device, {
      useVirtualStaging: false,
      elideEmptyCommandEncoders: false,
    });
    const dst = mock.device.createBuffer({ size: 16, usage: 8, label: 'dst' });
    const nativeStaging = mock.device.createBuffer({
      mappedAtCreation: true,
      size: 16,
      usage: 6,
    }) as { getMappedRange(): ArrayBuffer; unmap(): void; destroy(): void };
    expect(handle.getStats().interceptCount).toBe(0);
    failNativeMap = true;
    expect(() => nativeStaging.getMappedRange()).toThrow('native Dawn OOM on getMappedRange');
    expect((handle.getLastError() as Error).message).toContain('native Dawn OOM on getMappedRange');
    mock.device.createCommandEncoder().copyBufferToBuffer(nativeStaging, 0, dst, 0, 16);
    expect(mock.nativeCopies).toHaveLength(1);
    nativeStaging.destroy();
    handle.dispose();
  });
});
