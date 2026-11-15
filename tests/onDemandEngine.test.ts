import { Tensor as OrtTensor } from 'onnxruntime-web/webgpu';
import { describe, expect, it, vi } from 'vitest';
import {
  createExternalBlobResolver,
  createOnDemandEmbedTokensSession,
  createRemoteRangeBlob,
  decodeFp16Bits,
  parseHttpRangeTotalBytes,
  parseOnnxEmbedTokensPlan,
} from '../src/engine/embedTokensSession.ts';
import {
  DEFAULT_RETAINED_SCRATCH_MAX_BYTES,
  ExternalDataScratchPool,
  ExternalDataSessionBridge,
  createSyncAccessExternalDataProxy,
} from '../src/engine/externalDataProxy.ts';
import {
  assertStorageQuotaForDownload,
  assertWebGpuSingleTensorLimit,
  classifyModelLoadError,
  enrichModelLoadError,
  resolveQuotaAwareEmbedPlan,
} from '../src/engine/llm.worker.ts';
import {
  createSerializedCacheAdapter,
  migrateCacheResponseToOpfs,
  precacheModelFilesSequentially,
} from '../src/engine/modelPrecacher.ts';
import {
  createBlobSyncReadHandle,
  openOpfsSyncReadHandle,
  probeSyncReadHandle,
  writeResponseToOpfsStore,
  writeStreamToOpfsStore,
  type OpfsDirectoryLike,
  type OpfsFileHandleLike,
  type OpfsSyncAccessHandleLike,
  type OpfsSyncReadHandle,
  type OpfsWritableStreamLike,
} from '../src/engine/opfsCacheStore.ts';
import {
  SessionLoadCoordinator,
  withSerializedInferenceSessions,
} from '../src/engine/sessionSerializer.ts';
import { createSyncHttpRangeReadHandle } from '../src/engine/syncHttpRangeReader.ts';
import { installWebGpuUploadOptimizer } from '../src/engine/webgpuUploadOptimizer.ts';
import { getModelById, resolveEffectiveDtype } from '../src/models/catalog.ts';
import { assessModelMemoryRisk } from '../src/telemetry/systemProbe.ts';
import type { DeviceOption } from '../src/types.ts';

// ---------------------------------------------------------------------------
// Minimal ONNX ModelProto test builder (valid standard protobuf wire format)
// ---------------------------------------------------------------------------

const ONNX_DT = {
  FLOAT: 1,
  UINT8: 2,
  INT64: 7,
  FLOAT16: 10,
  INT4: 22,
} as const;

const UTF8_ENC = new TextEncoder();

function encodeVarint(val: bigint | number): Uint8Array {
  let v = typeof val === 'bigint' ? BigInt.asUintN(64, val) : BigInt.asUintN(64, BigInt(val));
  const out: number[] = [];
  while (v >= 0x80n) {
    out.push(Number((v & 0x7fn) | 0x80n));
    v >>= 7n;
  }
  out.push(Number(v));
  return new Uint8Array(out);
}

function concatBytes(...chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((s, c) => s + c.byteLength, 0);
  const out = new Uint8Array(total);
  let pos = 0;
  for (const c of chunks) {
    out.set(c, pos);
    pos += c.byteLength;
  }
  return out;
}

function pbTag(fn: number, wt: number): Uint8Array {
  return encodeVarint((BigInt(fn) << 3n) | BigInt(wt));
}

function pbVarint(fn: number, val: bigint | number): Uint8Array {
  return concatBytes(pbTag(fn, 0), encodeVarint(val));
}

function pbBytes(fn: number, data: Uint8Array | string): Uint8Array {
  const bytes = typeof data === 'string' ? UTF8_ENC.encode(data) : data;
  return concatBytes(pbTag(fn, 2), encodeVarint(bytes.byteLength), bytes);
}

function pbValueInfo(name: string): Uint8Array {
  return pbBytes(1, name);
}

function pbAttrInt(name: string, val: number | bigint): Uint8Array {
  return concatBytes(pbBytes(1, name), pbVarint(3, val));
}

function pbNode(
  opType: string,
  inputs: string[],
  outputs: string[],
  attrs: Uint8Array[] = []
): Uint8Array {
  return concatBytes(
    ...inputs.map((i) => pbBytes(1, i)),
    ...outputs.map((o) => pbBytes(2, o)),
    pbBytes(4, opType),
    ...attrs.map((a) => pbBytes(5, a))
  );
}

function pbStringStringEntry(key: string, val: string): Uint8Array {
  return concatBytes(pbBytes(1, key), pbBytes(2, val));
}

function pbTensorInlineInt64(name: string, dims: number[], values: bigint[]): Uint8Array {
  const raw = new Uint8Array(values.length * 8);
  const dv = new DataView(raw.buffer);
  for (let i = 0; i < values.length; i += 1) {
    dv.setBigInt64(i * 8, values[i]!, true);
  }
  return concatBytes(
    ...dims.map((d) => pbVarint(1, d)),
    pbVarint(2, ONNX_DT.INT64),
    pbBytes(8, name),
    pbBytes(9, raw)
  );
}

function pbTensorInlineFloat16Scalar(name: string, f16Bits: number): Uint8Array {
  const raw = new Uint8Array(2);
  new DataView(raw.buffer).setUint16(0, f16Bits, true);
  return concatBytes(pbVarint(2, ONNX_DT.FLOAT16), pbBytes(8, name), pbBytes(9, raw));
}

function pbTensorInlineFloat32Scalar(name: string, val: number): Uint8Array {
  const raw = new Uint8Array(4);
  new DataView(raw.buffer).setFloat32(0, val, true);
  return concatBytes(pbVarint(2, ONNX_DT.FLOAT), pbBytes(8, name), pbBytes(9, raw));
}

function pbTensorExternal(
  name: string,
  dims: number[],
  dataType: number,
  location: string,
  offset: number,
  length: number
): Uint8Array {
  return concatBytes(
    ...dims.map((d) => pbVarint(1, d)),
    pbVarint(2, dataType),
    pbVarint(14, 1), // data_location = EXTERNAL
    pbBytes(8, name),
    pbBytes(13, pbStringStringEntry('location', location)),
    pbBytes(13, pbStringStringEntry('offset', String(offset))),
    pbBytes(13, pbStringStringEntry('length', String(length)))
  );
}

function pbGraphModel(
  nodes: Uint8Array[],
  initializers: Uint8Array[],
  inputs: string[],
  outputs: string[]
): Uint8Array {
  const graph = concatBytes(
    ...nodes.map((n) => pbBytes(1, n)),
    ...initializers.map((init) => pbBytes(5, init)),
    ...inputs.map((i) => pbBytes(11, pbValueInfo(i))),
    ...outputs.map((o) => pbBytes(12, pbValueInfo(o)))
  );
  return concatBytes(pbVarint(1, 10), pbBytes(7, graph));
}

function encodeFp16Approx(val: number): number {
  if (val === 0) return 0;
  const f32 = new Float32Array([val]);
  const bits = new Uint32Array(f32.buffer)[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exp = ((bits >>> 23) & 0xff) - 127 + 15;
  const mant = (bits >>> 13) & 0x03ff;
  if (exp <= 0) return sign;
  if (exp >= 0x1f) return sign | 0x7c00;
  return sign | (exp << 10) | mant;
}

// ---------------------------------------------------------------------------
// In-memory OPFS mock with SyncAccessHandle read/write support
// ---------------------------------------------------------------------------

function createMockOpfsRoot() {
  const files = new Map<string, Uint8Array>();

  const makeDir = (prefix = ''): OpfsDirectoryLike => ({
    async getDirectoryHandle(name: string) {
      return makeDir(prefix ? `${prefix}/${name}` : name);
    },
    async getFileHandle(name: string, options?: { create?: boolean }): Promise<OpfsFileHandleLike> {
      const fullPath = prefix ? `${prefix}/${name}` : name;
      if (!files.has(fullPath)) {
        if (!options?.create) {
          throw Object.assign(new Error('NotFoundError'), { name: 'NotFoundError' });
        }
        files.set(fullPath, new Uint8Array(0));
      }
      return {
        async getFile() {
          const data = files.get(fullPath) ?? new Uint8Array(0);
          return new Blob([data as BlobPart]);
        },
        async createWritable(): Promise<OpfsWritableStreamLike> {
          const chunks: Uint8Array[] = [];
          return {
            async write(chunk: Uint8Array) {
              chunks.push(new Uint8Array(chunk));
            },
            async close() {
              files.set(fullPath, concatBytes(...chunks));
            },
            async abort() {
              chunks.length = 0;
            },
          };
        },
        async createSyncAccessHandle(): Promise<OpfsSyncAccessHandleLike> {
          let closed = false;
          const prev = files.get(fullPath);
          let buffer: Uint8Array = prev ? prev.slice() : new Uint8Array(0);
          return {
            getSize() {
              if (closed) throw new Error('Handle closed');
              return buffer.byteLength;
            },
            read(target: Uint8Array, options?: { at?: number }) {
              if (closed) throw new Error('Handle closed');
              const at = options?.at ?? 0;
              if (at >= buffer.byteLength) return 0;
              const n = Math.min(target.byteLength, buffer.byteLength - at);
              target.set(buffer.subarray(at, at + n), 0);
              return n;
            },
            write(data: Uint8Array) {
              if (closed) throw new Error('Handle closed');
              buffer = concatBytes(buffer, data);
              return data.byteLength;
            },
            flush() {
              files.set(fullPath, buffer);
            },
            close() {
              files.set(fullPath, buffer);
              closed = true;
            },
          };
        },
      };
    },
    async removeEntry(name: string) {
      const fullPath = prefix ? `${prefix}/${name}` : name;
      files.delete(fullPath);
    },
    async *entries() {
      for (const key of files.keys()) {
        if (prefix ? key.startsWith(`${prefix}/`) : !key.includes('/')) {
          const base = prefix ? key.slice(prefix.length + 1) : key;
          yield [base, { kind: 'file', name: base }] as [string, { kind: 'file'; name: string }];
        }
      }
    },
  });

  return {
    root: makeDir(),
    files,
  };
}

function createMockCacheStorage(initialEntries: Record<string, Uint8Array> = {}): CacheStorage {
  const store = new Map<string, Uint8Array>();
  for (const [k, v] of Object.entries(initialEntries)) {
    store.set(k, v);
  }
  const cacheObj = {
    async match(req: RequestInfo | URL): Promise<Response | undefined> {
      const key = typeof req === 'string' ? req : req instanceof URL ? req.href : req.url;
      const bytes = store.get(key);
      return bytes ? new Response(bytes as BodyInit) : undefined;
    },
    async put(req: RequestInfo | URL, resp: Response): Promise<void> {
      const key = typeof req === 'string' ? req : req instanceof URL ? req.href : req.url;
      store.set(key, new Uint8Array(await resp.arrayBuffer()));
    },
    async delete(req: RequestInfo | URL): Promise<boolean> {
      const key = typeof req === 'string' ? req : req instanceof URL ? req.href : req.url;
      return store.delete(key);
    },
  };
  return {
    async open() {
      return cacheObj as unknown as Cache;
    },
  } as unknown as CacheStorage;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('embedTokensSession (ONNX protobuf parser & on-demand streamer)', () => {
  it('decodes multi-branch Gemma 4 GatherBlockQuantized + sentinel guards + Mul + Reshape + Cast accurately', async () => {
    // vocabSize = 4, rowWidth = 32, blockSize = 32 -> 1 block per row
    const vocabSize = 4;
    const blockSize = 32;
    const packedCols = 16; // 32 * 4-bit = 16 bytes
    const numBlocks = 1;

    // Main table (inputs_embeds): unsigned UINT4 + zp + float16 scale + mulScale(4.0)
    const mainData = new Uint8Array(vocabSize * packedCols);
    const mainScales = new Uint8Array(vocabSize * numBlocks * 2);
    const mainZp = new Uint8Array(vocabSize * 1);
    const mainScalesView = new DataView(mainScales.buffer);

    for (let r = 0; r < vocabSize; r += 1) {
      mainScalesView.setUint16(r * 2, encodeFp16Approx(0.25 * (r + 1)), true);
      mainZp[r] = 0x05; // zp = 5
      for (let c = 0; c < packedCols; c += 1) {
        // low nibble = (r + 5) & 0xf, high nibble = (r + 7) & 0xf
        const lo = (r + 5) & 0x0f;
        const hi = (r + 7) & 0x0f;
        mainData[r * packedCols + c] = lo | (hi << 4);
      }
    }

    // Per-layer table: signed INT4 packed nibbles, reshaped to [0, 0, 2, 16]
    const pleData = new Uint8Array(vocabSize * packedCols);
    const pleScales = new Uint8Array(vocabSize * 2);
    const pleZp = new Uint8Array(vocabSize);
    const pleScalesView = new DataView(pleScales.buffer);
    for (let r = 0; r < vocabSize; r += 1) {
      pleScalesView.setUint16(r * 2, encodeFp16Approx(0.5), true);
      pleZp[r] = 0x0f; // signed 4-bit: 0xf -> -1
      // low nibble = 0xe (-2 signed), high nibble = 0x03 (+3 signed)
      pleData.fill(0x3e, r * packedCols, (r + 1) * packedCols);
      // Differentiate row r col 0 low nibble -> r (positive for r=0..3)
      pleData[r * packedCols] = (r & 0x0f) | 0x30;
    }

    const extBytes = concatBytes(mainData, mainScales, mainZp, pleData, pleScales, pleZp);
    const loc = 'embed_tokens_q4f16.onnx_data';
    let off = 0;
    const initMainData = pbTensorExternal('w_q', [vocabSize, packedCols], ONNX_DT.UINT8, loc, off, mainData.byteLength);
    off += mainData.byteLength;
    const initMainScales = pbTensorExternal('w_s', [vocabSize, numBlocks], ONNX_DT.FLOAT16, loc, off, mainScales.byteLength);
    off += mainScales.byteLength;
    const initMainZp = pbTensorExternal('w_z', [vocabSize, 1], ONNX_DT.UINT8, loc, off, mainZp.byteLength);
    off += mainZp.byteLength;
    const initPleData = pbTensorExternal('p_q', [vocabSize, packedCols], ONNX_DT.INT4, loc, off, pleData.byteLength);
    off += pleData.byteLength;
    const initPleScales = pbTensorExternal('p_s', [vocabSize, numBlocks], ONNX_DT.FLOAT16, loc, off, pleScales.byteLength);
    off += pleScales.byteLength;
    const initPleZp = pbTensorExternal('p_z', [vocabSize, 1], ONNX_DT.INT4, loc, off, pleZp.byteLength);

    const mulMainBits = encodeFp16Approx(4.0);
    const mulPleBits = encodeFp16Approx(2.0);
    const onnxBytes = pbGraphModel(
      [
        // Branch 1: inputs_embeds
        pbNode('GatherBlockQuantized', ['w_q', 'input_ids', 'w_s', 'w_z'], ['g1'], [
          pbAttrInt('bits', 4),
          pbAttrInt('block_size', blockSize),
          pbAttrInt('gather_axis', 0),
          pbAttrInt('quantize_axis', 1),
        ]),
        pbNode('Mul', ['g1', 'c_mul1'], ['s1']),
        pbNode('Cast', ['s1'], ['inputs_embeds'], [pbAttrInt('to', ONNX_DT.FLOAT)]),
        // Branch 2: Gemma 4 sentinel guard (id < 4 && id != 2 && id != 3 ? id : 0)
        pbNode('Less', ['input_ids', 'c_4'], ['cond_lt']),
        pbNode('Equal', ['input_ids', 'c_2'], ['eq2']),
        pbNode('Not', ['eq2'], ['n2']),
        pbNode('And', ['cond_lt', 'n2'], ['a1']),
        pbNode('Equal', ['input_ids', 'c_3'], ['eq3']),
        pbNode('Not', ['eq3'], ['n3']),
        pbNode('And', ['a1', 'n3'], ['cond_final']),
        pbNode('Where', ['cond_final', 'input_ids', 'c_0'], ['ple_ids']),
        pbNode('GatherBlockQuantized', ['p_q', 'ple_ids', 'p_s', 'p_z'], ['g2'], [
          pbAttrInt('bits', 4),
          pbAttrInt('block_size', blockSize),
          pbAttrInt('gather_axis', 0),
          pbAttrInt('quantize_axis', 1),
        ]),
        pbNode('Mul', ['g2', 'c_mul2'], ['s2']),
        pbNode('Reshape', ['s2', 'c_shape'], ['r2']),
        pbNode('Cast', ['r2'], ['per_layer_inputs'], [pbAttrInt('to', ONNX_DT.FLOAT)]),
      ],
      [
        initMainData,
        initMainScales,
        initMainZp,
        initPleData,
        initPleScales,
        initPleZp,
        pbTensorInlineFloat16Scalar('c_mul1', mulMainBits),
        pbTensorInlineFloat16Scalar('c_mul2', mulPleBits),
        pbTensorInlineInt64('c_4', [], [4n]),
        pbTensorInlineInt64('c_2', [], [2n]),
        pbTensorInlineInt64('c_3', [], [3n]),
        pbTensorInlineInt64('c_0', [], [0n]),
        pbTensorInlineInt64('c_shape', [4], [0n, 0n, 2n, 16n]),
      ],
      ['input_ids'],
      ['inputs_embeds', 'per_layer_inputs']
    );

    const plan = parseOnnxEmbedTokensPlan(onnxBytes);
    expect(plan).not.toBeNull();
    expect(plan!.outputNames).toEqual(['inputs_embeds', 'per_layer_inputs']);
    expect(plan!.externalLocations).toEqual([loc]);
    expect(plan!.branches[0]?.mulScale).toBeCloseTo(decodeFp16Bits(mulMainBits), 5);
    expect(plan!.branches[1]?.reshapeTarget).toEqual([0, 0, 2, 16]);

    // Track actual slice requests against external Blob
    let sliceCount = 0;
    const backingBlob = new Blob([extBytes as BlobPart]);
    const trackedBlob = {
      get size() {
        return backingBlob.size;
      },
      slice(start?: number, end?: number) {
        sliceCount += 1;
        return backingBlob.slice(start, end);
      },
    } as unknown as Blob;

    const session = await createOnDemandEmbedTokensSession(plan!, async (name) =>
      name === loc ? trackedBlob : null
    );
    expect(session.inputNames).toEqual(['input_ids']);
    expect(session.outputNames).toEqual(['inputs_embeds', 'per_layer_inputs']);

    // Query [1, 2, -1]:
    // - row 1 unguarded -> scale = 0.5, mulScale = 4 => factor = 2.0, col 0 q=6 zp=5 => 2.0, col 1 q=8 zp=5 => 6.0
    // - row 2 IS sentinel (== 2) on inputs_embeds -> row 2 unguarded (factor=3.0, col0=(7-5)*3=6.0);
    //   on per_layer_inputs -> guarded to row 0 (col0 q=0, zp=-1, scale=0.5, mul=2 => (0 - (-1))*1 = 1.0)
    // - row -1 => row 3 on inputs_embeds (factor=4.0, col0=(8-5)*4=12.0),
    //   on per_layer_inputs guarded because -1 < 4 && -1 != 2 && -1 != 3 -> guardedWait: -1 < 4! -1+4 = 3 -> row 3!
    const inputTensor = new OrtTensor('int64', BigInt64Array.from([1n, 2n, -1n]), [1, 3]);
    const outputs = await session.run({ input_ids: inputTensor });
    const inputsEmbeds = outputs.inputs_embeds as OrtTensor;
    const pleOut = outputs.per_layer_inputs as OrtTensor;

    expect(inputsEmbeds.dims).toEqual([1, 3, 32]);
    expect(pleOut.dims).toEqual([1, 3, 2, 16]);

    const emb = inputsEmbeds.data as Float32Array;
    expect(emb[0]).toBeCloseTo(2.0, 5); // tok 1 col 0
    expect(emb[1]).toBeCloseTo(6.0, 5); // tok 1 col 1
    expect(emb[32]).toBeCloseTo(6.0, 5); // tok 2 col 0
    expect(emb[64]).toBeCloseTo(12.0, 5); // tok -1 (=3) col 0

    const ple = pleOut.data as Float32Array;
    expect(ple[0]).toBeCloseTo((1 - -1) * 1, 5); // row 1 col 0 = 2.0
    expect(ple[1]).toBeCloseTo((3 - -1) * 1, 5); // row 1 col 1 = 4.0
    expect(ple[32]).toBeCloseTo((0 - -1) * 1, 5); // sentinel tok 2 -> guarded to row 0 => 1.0
    expect(ple[64]).toBeCloseTo((3 - -1) * 1, 5); // tok -1 (=3) col 0 => 4.0

    const slicesBeforeRepeat = sliceCount;
    expect(slicesBeforeRepeat).toBeGreaterThan(0);

    // Repeating same token IDs must hit 8 MiB row cache with zero extra Blob reads
    await session.run({
      input_ids: new OrtTensor('int64', BigInt64Array.from([1n, 2n]), [1, 2]),
    });
    expect(sliceCount).toBe(slicesBeforeRepeat);

    await session.release();
    await expect(
      session.run({ input_ids: new OrtTensor('int64', BigInt64Array.from([0n]), [1, 1]) })
    ).rejects.toThrow(/released/);
  });

  it('supports Gather (fp16) with inline weights, float32 scales, bounded LRU eviction, and error checks', async () => {
    // Test unquantized Gather(fp16) + float32 scalar Mul
    const tableRaw = new Uint8Array(3 * 4 * 2); // 3 rows x 4 cols x fp16
    const tv = new DataView(tableRaw.buffer);
    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 4; c += 1) {
        tv.setUint16((r * 4 + c) * 2, encodeFp16Approx(r * 10 + c), true);
      }
    }
    const loc = 'embed_tokens_fp16.onnx_data';
    const onnxGather = pbGraphModel(
      [
        pbNode('Gather', ['embed_w', 'input_ids'], ['g'], [pbAttrInt('axis', 0)]),
        pbNode('Mul', ['g', 'c_scale'], ['inputs_embeds']),
      ],
      [
        pbTensorExternal('embed_w', [3, 4], ONNX_DT.FLOAT16, loc, 0, tableRaw.byteLength),
        pbTensorInlineFloat32Scalar('c_scale', 0.5),
      ],
      ['input_ids'],
      ['inputs_embeds']
    );

    const plan = parseOnnxEmbedTokensPlan(onnxGather);
    expect(plan).not.toBeNull();

    let reads = 0;
    const blob = {
      size: tableRaw.byteLength,
      slice(start?: number, end?: number) {
        reads += 1;
        return new Blob([tableRaw as BlobPart]).slice(start, end);
      },
    } as unknown as Blob;

    // LRU cap = 32 bytes (2 rows * 4 floats * 4 bytes = 32 bytes => holds at most 2 rows)
    const session = await createOnDemandEmbedTokensSession(plan!, async () => blob, {
      maxRowCacheBytes: 32,
    });

    const out = await session.run({
      input_ids: new OrtTensor('int64', BigInt64Array.from([0n, 1n, 0n]), [1, 3]),
    });
    expect(reads).toBe(2); // row 0 deduped, row 1 read once
    const f32 = (out.inputs_embeds as OrtTensor).data as Float32Array;
    expect([...f32.slice(0, 4)]).toEqual([0, 0.5, 1, 1.5]);
    expect([...f32.slice(4, 8)]).toEqual([5, 5.5, 6, 6.5]);

    // Reading row 2 evicts LRU row 1 (because row 0 was touched after 1 during set order? Wait: set order [0, 1])
    await session.run({ input_ids: new OrtTensor('int64', BigInt64Array.from([2n]), [1, 1]) });
    expect(reads).toBe(3);

    // Querying row 0 (evicted as oldest from [0, 1] -> [1, 2]) causes a miss; row 2 hits cache
    const beforeZero = reads;
    await session.run({ input_ids: new OrtTensor('int64', BigInt64Array.from([2n, 0n]), [1, 2]) });
    expect(reads).toBe(beforeZero + 1);

    // Out-of-bounds index throws RangeError
    await expect(
      session.run({ input_ids: new OrtTensor('int64', BigInt64Array.from([99n]), [1, 1]) })
    ).rejects.toThrow(RangeError);

    // Non-embedding graph (e.g. decoder with >1 input or MatMul) safely returns null
    const nonEmbedGraph = pbGraphModel(
      [pbNode('MatMul', ['input_ids', 'w'], ['logits'])],
      [],
      ['input_ids', 'attention_mask'],
      ['logits']
    );
    expect(parseOnnxEmbedTokensPlan(nonEmbedGraph)).toBeNull();

    // Missing blob rejects cleanly
    await expect(createOnDemandEmbedTokensSession(plan!, async () => null)).rejects.toThrow(
      /Missing external embedding/
    );
  });

  it('createExternalBlobResolver resolves OPFS -> CacheStorage -> fetch in priority order', async () => {
    const { root } = createMockOpfsRoot();
    const opfsUrl = 'https://huggingface.co/org/model/resolve/main/onnx/embed_opfs.onnx_data';
    const cacheUrl = 'https://huggingface.co/org/model/resolve/main/onnx/embed_cache.onnx_data';
    const fetchUrl = 'https://huggingface.co/org/model/resolve/main/onnx/embed_fetch.onnx_data';
    const missingUrl = 'https://huggingface.co/org/model/resolve/main/onnx/missing.onnx_data';

    await writeResponseToOpfsStore(root, opfsUrl, new Response(new Uint8Array([1, 2, 3, 4])));
    const cacheStorage = createMockCacheStorage({ [cacheUrl]: new Uint8Array([5, 6, 7]) });
    const fetchFn = vi.fn(async (url: RequestInfo | URL) => {
      if (String(url) === fetchUrl) {
        return new Response(new Uint8Array([8, 9]));
      }
      return new Response(null, { status: 404 });
    }) as unknown as typeof fetch;

    const resolveBlob = createExternalBlobResolver({
      urlByLocation: new Map([
        ['embed_opfs.onnx_data', opfsUrl],
        ['embed_cache.onnx_data', cacheUrl],
        ['embed_fetch.onnx_data', fetchUrl],
        ['missing.onnx_data', missingUrl],
      ]),
      opfsRoot: root,
      cacheName: 'test-cache',
      cacheStorage,
      fetchFn,
    });

    const b1 = await resolveBlob('embed_opfs.onnx_data');
    expect(new Uint8Array(await b1!.arrayBuffer())).toEqual(new Uint8Array([1, 2, 3, 4]));
    expect(fetchFn).not.toHaveBeenCalled();

    const b2 = await resolveBlob('embed_cache.onnx_data');
    expect(new Uint8Array(await b2!.arrayBuffer())).toEqual(new Uint8Array([5, 6, 7]));

    const b3 = await resolveBlob('embed_fetch.onnx_data');
    expect(new Uint8Array(await b3!.arrayBuffer())).toEqual(new Uint8Array([8, 9]));

    expect(await resolveBlob('missing.onnx_data')).toBeNull();
    expect(await resolveBlob('unregistered.onnx_data')).toBeNull();
  });
});

describe('externalDataProxy, sessionSerializer hooks & modelPrecacher virtualization', () => {
  it('streams SyncAccessHandle slices lazily without copying full file and handles negative/short bounds', () => {
    const backing = new Uint8Array(100);
    for (let i = 0; i < backing.length; i += 1) {
      backing[i] = i & 0xff;
    }

    // Force short reads (at most 7 bytes per readAt call)
    const proxy = createSyncAccessExternalDataProxy(backing.byteLength, (dest, offset) => {
      const len = Math.min(7, dest.byteLength, backing.byteLength - offset);
      if (len <= 0) return 0;
      dest.set(backing.subarray(offset, offset + len), 0);
      return len;
    });

    expect(proxy instanceof Uint8Array).toBe(true);
    expect(proxy.byteLength).toBe(100);
    expect(proxy.length).toBe(100);
    expect(proxy.subarray(10, 30)).toEqual(backing.subarray(10, 30));
    expect(proxy.subarray(-5)).toEqual(backing.subarray(95));
    expect(proxy.subarray(50, 50)).toEqual(new Uint8Array(0));

    const brokenProxy = createSyncAccessExternalDataProxy(16, () => 0);
    expect(() => brokenProxy.subarray(0, 8)).toThrow(/Short synchronous (?:OPFS|external-data) read/);
  });

  it('ExternalDataSessionBridge opens SyncAccessHandle proxies and closes handles cleanly on success and failure', async () => {
    const { root } = createMockOpfsRoot();
    const decoderUrl =
      'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data';
    const payload = new Uint8Array([10, 20, 30, 40, 50, 60]);
    await writeResponseToOpfsStore(root, decoderUrl, new Response(payload));

    const bridge = new ExternalDataSessionBridge();
    bridge.registerUrl('decoder_model_merged_q4f16.onnx_data', decoderUrl);

    const closedUrls: string[] = [];
    const origGetDir = root.getDirectoryHandle!.bind(root);
    const wrappedRoot: OpfsDirectoryLike = {
      ...root,
      async getDirectoryHandle(name, opts) {
        const sub = await origGetDir(name, opts);
        return {
          ...sub,
          async getFileHandle(fName, fOpts) {
            const fh = await sub.getFileHandle(fName, fOpts);
            return {
              ...fh,
              async createSyncAccessHandle() {
                const sh = await fh.createSyncAccessHandle!();
                return {
                  ...sh,
                  close() {
                    closedUrls.push(fName);
                    sh.close();
                  },
                };
              },
            };
          },
        };
      },
    };

    const existingBuf = new Uint8Array([1, 2]);
    const sessionOpts: Record<string, unknown> = {
      externalData: [
        { path: './decoder_model_merged_q4f16.onnx_data', data: decoderUrl },
        { path: './decoder_model_merged_q4f16.onnx_data' },
        { path: './already_inline.onnx_data', data: existingBuf },
      ],
    };

    const cleanup = await bridge.prepareSessionExternalData(sessionOpts, {
      opfsRoot: wrappedRoot,
      cacheName: 'test-cache',
    });

    const entries = sessionOpts.externalData as Array<{ path: string; data: Uint8Array }>;
    expect(entries[0]!.data instanceof Uint8Array).toBe(true);
    expect(entries[0]!.data.byteLength).toBe(6);
    expect(entries[0]!.data.subarray(2, 5)).toEqual(new Uint8Array([30, 40, 50]));
    expect(entries[1]!.data instanceof Uint8Array).toBe(true);
    expect(entries[1]!.data.subarray(0, 2)).toEqual(new Uint8Array([10, 20]));
    expect(entries[2]!.data).toBe(existingBuf);
    expect(closedUrls).toHaveLength(0);

    cleanup();
    expect(closedUrls.length).toBe(2);
    bridge.closeAll();
  });

  it('withSerializedInferenceSessions routes interceptCreate vs prepareExternalData and always cleans up', async () => {
    const coordinator = new SessionLoadCoordinator();
    const events: string[] = [];
    const fakeSession = { inputNames: ['input_ids'], outputNames: ['inputs_embeds'] };
    const ortTarget = {
      async create(_data: unknown, opts?: Record<string, unknown>) {
        events.push(`ort.create:${JSON.stringify(opts?.externalData)}`);
        return fakeSession as never;
      },
    };

    const fakeSessionCreate = ortTarget.create.bind(ortTarget);
    ortTarget.create = fakeSessionCreate;

    // 1. Intercepted path skips prepareExternalData & ORT create, emits onSessionCreated
    const customSession = { inputNames: ['input_ids'], outputNames: ['inputs_embeds'], custom: true };
    const intercepted = await withSerializedInferenceSessions(
      coordinator,
      async () => ortTarget.create(new Uint8Array([1, 2]), { externalData: ['foo'] }),
      {
        target: ortTarget as never,
        interceptCreate: async () => customSession,
        prepareExternalData: async () => {
          events.push('prepare-should-not-run');
        },
        onSessionCreated: () => events.push('created'),
      }
    );
    expect(intercepted).toBe(customSession);
    expect(events).toEqual(['created']);

    // 2. Non-intercepted path runs prepareExternalData -> ORT create -> cleanup even on error
    events.length = 0;
    const errorTarget = {
      async create(_data?: unknown, _opts?: Record<string, unknown>) {
        events.push('ort.create');
        throw new Error('boom');
      },
    };
    const extOpts: Record<string, unknown> = {
      externalData: [{ path: 'a.onnx_data', data: new Uint8Array([9]) }],
    };
    await expect(
      withSerializedInferenceSessions(
        coordinator,
        async () => errorTarget.create(new Uint8Array([1]), extOpts),
        {
          target: errorTarget as never,
          interceptCreate: async () => null,
          prepareExternalData: async () => {
            events.push('prepare');
            return () => {
              events.push('cleanup');
            };
          },
        }
      )
    ).rejects.toThrow('boom');
    expect(events).toEqual(['prepare', 'ort.create', 'cleanup']);
    expect(extOpts.externalData).toEqual([]);
  });

  it('createSerializedCacheAdapter virtualizes OPFS external data URLs without buffering into RAM', async () => {
    const { root } = createMockOpfsRoot();
    const embedDataUrl =
      'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/embed_tokens_q4f16.onnx_data';
    const decoderDataUrl =
      'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data';
    const onnxHeaderUrl =
      'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/embed_tokens_q4f16.onnx';
    const cacheOnlyEmbedUrl =
      'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/main/onnx/embed_tokens_q4f16.onnx_data';
    const cacheOnlyDecoderUrl =
      'https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data';

    await writeResponseToOpfsStore(root, embedDataUrl, new Response(new Uint8Array([1, 2, 3])));
    await writeResponseToOpfsStore(root, decoderDataUrl, new Response(new Uint8Array([4, 5, 6])));
    await writeResponseToOpfsStore(root, onnxHeaderUrl, new Response(new Uint8Array([7, 8])));

    const cacheStorage = createMockCacheStorage({
      [cacheOnlyEmbedUrl]: new Uint8Array([11, 12]),
      [cacheOnlyDecoderUrl]: new Uint8Array([13, 14]),
    });

    const matched: Array<[string, string]> = [];
    const adapter = createSerializedCacheAdapter('test-cache', cacheStorage, {
      opfsRoot: root,
      virtualizeExternalData: {
        allowCacheStorage: (url) => /embed_tokens[^/]*\.onnx_data/.test(url),
        onMatched: (fileName, url) => matched.push([fileName, url]),
      },
    })!;

    // OPFS .onnx_data* returns URL string directly
    const hitDecoder = await adapter.match(decoderDataUrl);
    expect(hitDecoder as unknown).toBe(decoderDataUrl);

    // CacheStorage embed_tokens .onnx_data returns URL string because allowCacheStorage matched
    const hitCacheEmbed = await adapter.match(cacheOnlyEmbedUrl);
    expect(hitCacheEmbed as unknown).toBe(cacheOnlyEmbedUrl);

    // CacheStorage decoder .onnx_data returns real Response (not virtualized)
    const hitCacheDecoder = await adapter.match(cacheOnlyDecoderUrl);
    expect(hitCacheDecoder instanceof Response).toBe(true);
    expect(new Uint8Array(await hitCacheDecoder!.arrayBuffer())).toEqual(new Uint8Array([13, 14]));

    // Non-.onnx_data* (.onnx graph) still returns a real Response
    const hitOnnx = await adapter.match(onnxHeaderUrl);
    expect(hitOnnx instanceof Response).toBe(true);
    expect(new Uint8Array(await hitOnnx!.arrayBuffer())).toEqual(new Uint8Array([7, 8]));

    expect(matched).toEqual([
      ['decoder_model_merged_q4f16.onnx_data', decoderDataUrl],
      ['embed_tokens_q4f16.onnx_data', cacheOnlyEmbedUrl],
    ]);
  });
});

describe('WebGPU single-tensor limits, streaming risk & WEBGPU_BUFFER_LIMIT error classification', () => {
  const makeWebGpuDevice = (limits: Record<string, number>): DeviceOption => ({
    id: 'webgpu-test',
    label: 'WebGPU Test',
    backend: 'webgpu',
    features: ['shader-f16'],
    limits,
    supportsFp16: true,
    recommendedDtype: 'q4f16',
    available: true,
    runtimeDetails: {
      crossOriginIsolated: false,
      hardwareConcurrency: 8,
      simdSupported: true,
      numThreads: 1,
    },
  });

  it('checks maxStorageBufferBindingSize and maxBufferSize against maxSingleTensorBytes', () => {
    const gemmaE2b = getModelById('gemma-4-e2b');
    const llama3b = getModelById('llama-3.2-3b');
    const desktopEnv = { userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', deviceMemoryGb: 8 };
    const oneGibAdapter = makeWebGpuDevice({
      maxStorageBufferBindingSize: 1024 * 1024 * 1024,
      maxBufferSize: 1024 * 1024 * 1024,
    });

    // With default on-demand embed_tokens streaming, Gemma 4 E2B max GPU tensor is 192 MiB (fits 1 GiB!)
    const streamingResolved = resolveEffectiveDtype(gemmaE2b, oneGibAdapter, 'q4f16');
    const streamingRisk = assessModelMemoryRisk(gemmaE2b, streamingResolved, oneGibAdapter, desktopEnv);
    expect(streamingRisk.level).toBe('ok');

    // On 8 GB Android, Gemma 4 E2B q4f16 peak drops from high (5+ GB) to caution (~1.90 GiB)
    const android8GbRisk = assessModelMemoryRisk(gemmaE2b, streamingResolved, oneGibAdapter, {
      userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Mobile Safari/537.36',
      deviceMemoryGb: 8,
    });
    expect(android8GbRisk.level).toBe('caution');

    // With legacy opt-out (embed_tokens uploaded to GPU = 1,120 MiB), 1 GiB WebGPU adapter flags high!
    const legacyResolved = resolveEffectiveDtype(gemmaE2b, oneGibAdapter, 'q4f16', {
      onDemandEmbedTokens: false,
    });
    const legacyRisk = assessModelMemoryRisk(gemmaE2b, legacyResolved, oneGibAdapter, desktopEnv);
    expect(legacyRisk.level).toBe('high');
    expect(legacyRisk.headline).toContain('単一バッファ上限');

    // Monolithic Llama 3.2 3B q4 has a 1.47 GiB single tensor -> flagged on 1 GiB adapter, OK on 2 GiB
    const llamaQ4 = resolveEffectiveDtype(llama3b, oneGibAdapter, 'q4');
    expect(assessModelMemoryRisk(llama3b, llamaQ4, oneGibAdapter, desktopEnv).level).toBe('high');

    const exactBoundaryAdapter = makeWebGpuDevice({
      maxStorageBufferBindingSize: llamaQ4.maxSingleTensorBytes!,
      maxBufferSize: llamaQ4.maxSingleTensorBytes!,
    });
    expect(
      assessModelMemoryRisk(llama3b, llamaQ4, exactBoundaryAdapter, desktopEnv).level
    ).toBe('ok');

    const minusOneAdapter = makeWebGpuDevice({
      maxBufferSize: llamaQ4.maxSingleTensorBytes! - 1,
    });
    expect(
      assessModelMemoryRisk(llama3b, llamaQ4, minusOneAdapter, desktopEnv).level
    ).toBe('high');

    // Gemma 4 E2B on Desktop WASM fits well under the 3.2 GiB wasm32 ceiling (~1.86 GB weights)
    const wasmCpu: DeviceOption = {
      id: 'wasm-cpu',
      label: 'CPU (WASM)',
      backend: 'wasm',
      features: ['wasm-simd'],
      limits: {},
      supportsFp16: false,
      recommendedDtype: 'q4',
      available: true,
      runtimeDetails: {
        crossOriginIsolated: false,
        hardwareConcurrency: 8,
        simdSupported: true,
        numThreads: 1,
      },
    };
    const gemmaWasm = resolveEffectiveDtype(gemmaE2b, wasmCpu, 'q4');
    expect(assessModelMemoryRisk(gemmaE2b, gemmaWasm, wasmCpu, desktopEnv).level).toBe('ok');
  });

  it('classifies WebGPU single-buffer limit validation errors cleanly as WEBGPU_BUFFER_LIMIT', () => {
    const res1 = classifyModelLoadError(
      new Error('Binding size (1174405120) is larger than maxStorageBufferBindingSize (1073741824).')
    );
    expect(res1.code).toBe('WEBGPU_BUFFER_LIMIT');

    const res2 = classifyModelLoadError(
      new Error('GPUBuffer size exceeds maxBufferSize limit')
    );
    expect(res2.code).toBe('WEBGPU_BUFFER_LIMIT');

    const tagged = Object.assign(new Error('limit check'), { code: 'WEBGPU_BUFFER_LIMIT' });
    expect(classifyModelLoadError(tagged).code).toBe('WEBGPU_BUFFER_LIMIT');
  });

  it('allows mobile 128 MiB maxStorage + 256 MiB maxBuffer GPUs for split Gemma 4 E2B and validates preflight limits/quota', () => {
    const gemmaE2b = getModelById('gemma-4-e2b');
    const gemmaE4b = getModelById('gemma-4-e4b');
    const qwen08 = getModelById('qwen-3.5-0.8b');
    const llama3b = getModelById('llama-3.2-3b');

    const mobileSpecGpu = makeWebGpuDevice({
      maxStorageBufferBindingSize: 134_217_728, // 128 MiB
      maxBufferSize: 268_435_456, // 256 MiB
    });
    const android8gEnv = {
      userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) Mobile Safari/537.36',
      deviceMemoryGb: 8,
    };

    // Gemma 4 E2B (192 MiB MatMulNBits) fits 256 MiB maxBuffer with split storage bindings (no false high block)
    const e2bResolved = resolveEffectiveDtype(gemmaE2b, mobileSpecGpu, 'q4f16');
    const e2bRisk = assessModelMemoryRisk(gemmaE2b, e2bResolved, mobileSpecGpu, android8gEnv);
    expect(e2bRisk.level).toBe('caution');
    expect(e2bRisk.headline).not.toContain('単一バッファ上限');
    expect(() => assertWebGpuSingleTensorLimit(gemmaE2b, mobileSpecGpu, e2bResolved)).not.toThrow();

    // Gemma 4 E4B (320 MiB) exceeds 256 MiB maxBuffer -> high risk, alternatives must NOT suggest E4B itself
    const e4bResolved = resolveEffectiveDtype(gemmaE4b, mobileSpecGpu, 'q4f16');
    const e4bRisk = assessModelMemoryRisk(gemmaE4b, e4bResolved, mobileSpecGpu, android8gEnv);
    expect(e4bRisk.level).toBe('high');
    expect(e4bRisk.detail).toContain('Gemma 4 E2B');
    expect(e4bRisk.detail).not.toContain('Gemma 4 E4B');
    expect(() => assertWebGpuSingleTensorLimit(gemmaE4b, mobileSpecGpu, e4bResolved)).toThrow(
      expect.objectContaining({ code: 'WEBGPU_BUFFER_LIMIT' })
    );

    // Llama 3.2 3B monolithic Gather requires min(maxStorage, maxBuffer)
    const llamaResolved = resolveEffectiveDtype(llama3b, mobileSpecGpu, 'q4f16');
    const hugeBufferTinyStorage = makeWebGpuDevice({
      maxStorageBufferBindingSize: 134_217_728,
      maxBufferSize: 2_147_483_648,
    });
    expect(() =>
      assertWebGpuSingleTensorLimit(llama3b, hugeBufferTinyStorage, llamaResolved)
    ).toThrow(expect.objectContaining({ code: 'WEBGPU_BUFFER_LIMIT' }));

    // When maxBufferSize is 64 MiB (< even Qwen 0.8B 121.3 MiB), detail must advise CPU without broken suggestions
    const tinyGpu = makeWebGpuDevice({
      maxStorageBufferBindingSize: 64 * 1024 * 1024,
      maxBufferSize: 64 * 1024 * 1024,
    });
    const tinyQwenRisk = assessModelMemoryRisk(
      qwen08,
      resolveEffectiveDtype(qwen08, tinyGpu, 'q4f16'),
      tinyGpu,
      android8gEnv
    );
    expect(tinyQwenRisk.level).toBe('high');
    expect(tinyQwenRisk.detail).toBe('CPU (WASM) デバイスへ切り替えてください。');

    // Storage quota preflight fails early with QuotaExceededError naming cached models
    expect(() =>
      assertStorageQuotaForDownload(
        gemmaE2b,
        e2bResolved,
        600 * 1024 * 1024,
        [
          {
            modelId: 'qwen-3.5-0.8b',
            repoId: qwen08.repoId,
            cached: true,
            cachedBytes: 580 * 1024 * 1024,
            fileCount: 4,
            files: [],
            lastCheckedAt: Date.now(),
          },
        ]
      )
    ).toThrow(expect.objectContaining({ name: 'QuotaExceededError', code: 'STORAGE_QUOTA_EXCEEDED' }));

    // Passes when model is already mostly cached or estimate unavailable
    expect(() =>
      assertStorageQuotaForDownload(gemmaE2b, e2bResolved, undefined, [])
    ).not.toThrow();
    expect(() =>
      assertStorageQuotaForDownload(
        gemmaE2b,
        e2bResolved,
        600 * 1024 * 1024,
        [
          {
            modelId: 'gemma-4-e2b',
            repoId: gemmaE2b.repoId,
            cached: true,
            cachedBytes: 3_150_000_000,
            fileCount: 6,
            files: [],
            lastCheckedAt: Date.now(),
          },
        ]
      )
    ).not.toThrow();
  });

  it('falls back cleanly to createWritable and FileReaderSync when createSyncAccessHandle throws NotSupportedError', async () => {
    const stored = new Map<string, Uint8Array>();
    const syncErrRoot: OpfsDirectoryLike = {
      async getDirectoryHandle() {
        return syncErrRoot;
      },
      async getFileHandle(name: string, options?: { create?: boolean }): Promise<OpfsFileHandleLike> {
        if (!stored.has(name) && !options?.create) {
          throw Object.assign(new Error('NotFound'), { name: 'NotFoundError' });
        }
        return {
          async createSyncAccessHandle(): Promise<OpfsSyncAccessHandleLike> {
            throw Object.assign(new Error('SyncAccessHandle not supported'), {
              name: 'NotSupportedError',
            });
          },
          async createWritable(): Promise<OpfsWritableStreamLike> {
            const chunks: Uint8Array[] = [];
            return {
              async write(chunk: Uint8Array) {
                chunks.push(new Uint8Array(chunk));
              },
              async close() {
                stored.set(name, concatBytes(...chunks));
              },
              async abort() {
                chunks.length = 0;
              },
            };
          },
          async getFile() {
            const bytes = stored.get(name) ?? new Uint8Array(0);
            return new Blob([bytes as BlobPart]);
          },
        };
      },
      async removeEntry(name: string) {
        stored.delete(name);
      },
    };

    const url =
      'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data';
    const payload = new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]);
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(payload.subarray(0, 5));
        c.enqueue(payload.subarray(5));
        c.close();
      },
    });

    const written = await writeStreamToOpfsStore({
      opfsRoot: syncErrRoot,
      url,
      stream,
      headers: new Headers({ 'content-length': '8' }),
      expectedTotalBytes: 8,
    });
    expect(written).toBe(8);

    // Without FileReaderSync (e.g. main thread), openOpfsSyncReadHandle returns undefined safely
    expect(await openOpfsSyncReadHandle(syncErrRoot, url)).toBeUndefined();

    // With Web Worker FileReaderSync available, opens slice reader seamlessly
    class MockFileReaderSync {
      readAsArrayBuffer(blob: Blob): ArrayBuffer {
        const idx = Number((blob as unknown as { __start?: number }).__start ?? 0);
        const end = Number((blob as unknown as { __end?: number }).__end ?? idx);
        const data = (blob as unknown as { __bytes?: Uint8Array }).__bytes;
        return data ? data.slice(idx, end).buffer : new ArrayBuffer(0);
      }
    }
    vi.stubGlobal('FileReaderSync', MockFileReaderSync);
    try {
      const patchFileSlice: OpfsDirectoryLike = {
        ...syncErrRoot,
        async getDirectoryHandle() {
          return patchFileSlice;
        },
        async getFileHandle(name: string, opts?: { create?: boolean }) {
          const fh = await syncErrRoot.getFileHandle(name, opts);
          return {
            ...fh,
            async getFile() {
              const bytes = stored.get(name) ?? new Uint8Array(0);
              return {
                size: bytes.byteLength,
                slice(start = 0, end = bytes.byteLength) {
                  return {
                    size: Math.max(0, end - start),
                    __start: start,
                    __end: end,
                    __bytes: bytes,
                  } as unknown as Blob;
                },
                arrayBuffer: async () => bytes.slice().buffer,
                text: async () => new TextDecoder().decode(bytes),
              } as unknown as File;
            },
          };
        },
      } as OpfsDirectoryLike;

      const handle = await openOpfsSyncReadHandle(patchFileSlice, url);
      expect(handle?.byteLength).toBe(8);
      const pool = new ExternalDataScratchPool();
      const proxy = createSyncAccessExternalDataProxy(handle!.byteLength, handle!.readAt, {
        scratchPool: pool,
      });
      const s1 = proxy.subarray(2, 6);
      expect(Array.from(s1)).toEqual([30, 40, 50, 60]);
      const s2 = proxy.subarray(0, 3);
      expect(Array.from(s2)).toEqual([10, 20, 30]);
      // Shared buffer reused without leaking prior tail
      expect(s2.buffer).toBe(s1.buffer);
      pool.release();
      handle?.close();
      expect(() => handle?.readAt(new Uint8Array(2), 0)).toThrow(/closed/i);

      // ExternalDataSessionBridge uses CacheStorage Blob + FileReaderSync without calling Response.arrayBuffer()
      const rawData = new Uint8Array([11, 22, 33, 44, 55, 66]);
      const arrayBufferSpy = vi.fn(async () => rawData.slice().buffer);
      const mockCacheStorage: Pick<CacheStorage, 'open'> = {
        async open() {
          return {
            async match() {
              return {
                blob: async () =>
                  ({
                    size: rawData.byteLength,
                    slice(start = 0, end = rawData.byteLength) {
                      return {
                        size: Math.max(0, end - start),
                        __start: start,
                        __end: end,
                        __bytes: rawData,
                      } as unknown as Blob;
                    },
                  }) as unknown as Blob,
                arrayBuffer: arrayBufferSpy,
              } as unknown as Response;
            },
          } as unknown as Cache;
        },
      };
      const bridge = new ExternalDataSessionBridge();
      const cacheUrl =
        'https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data_1';
      bridge.registerUrl('decoder_model_merged_q4f16.onnx_data_1', cacheUrl);
      const sessionOptions: Record<string, unknown> = {
        externalData: ['decoder_model_merged_q4f16.onnx_data_1'],
      };
      const cleanup = await bridge.prepareSessionExternalData(sessionOptions, {
        opfsRoot: null,
        cacheName: 'test-cache',
        cacheStorage: mockCacheStorage,
      });
      expect(arrayBufferSpy).not.toHaveBeenCalled();
      const entries = sessionOptions.externalData as Array<{ path: string; data: Uint8Array }>;
      expect(entries[0]?.path).toBe('decoder_model_merged_q4f16.onnx_data_1');
      expect(Array.from(entries[0]!.data.subarray(1, 5))).toEqual([22, 33, 44, 55]);
      cleanup();
      bridge.closeAll();

      // QuotaExceededError on createSyncAccessHandle must NOT fall back to createWritable and must clean partials
      const writableSpy = vi.fn();
      const quotaStored = new Set<string>();
      const quotaRoot: OpfsDirectoryLike = {
        async getDirectoryHandle() {
          return quotaRoot;
        },
        async getFileHandle(name: string): Promise<OpfsFileHandleLike> {
          quotaStored.add(name);
          return {
            async createSyncAccessHandle(): Promise<OpfsSyncAccessHandleLike> {
              throw Object.assign(new Error('No space left'), { name: 'QuotaExceededError' });
            },
            createWritable: writableSpy as unknown as OpfsFileHandleLike['createWritable'],
            getFile: async () => new Blob([]),
          };
        },
        async removeEntry(name: string) {
          quotaStored.delete(name);
        },
      };
      await expect(
        writeStreamToOpfsStore({
          opfsRoot: quotaRoot,
          url,
          stream: new ReadableStream<Uint8Array>({
            start(c) {
              c.enqueue(new Uint8Array([1, 2]));
              c.close();
            },
          }),
          headers: new Headers(),
        })
      ).rejects.toMatchObject({ name: 'QuotaExceededError' });
      expect(writableSpy).not.toHaveBeenCalled();
      expect(quotaStored.size).toBe(0);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('shares aligned scratch buffer without overwriting trailing 16 weight bytes, trims slabs > threshold, and elides empty GPU submits', () => {
    const scratchPool = new ExternalDataScratchPool();
    // Seed large dirty pool buffer (> 8192 bytes) to test both non-zero dirt & post-upload trim
    const dirtySeed = scratchPool.acquire(16384);
    dirtySeed.fill(0xaa);

    const exactAlignedData = new Uint8Array(4112); // > 4096 and % 16 === 0
    for (let i = 0; i < exactAlignedData.length; i += 1) {
      exactAlignedData[i] = ((i * 7 + 3) & 0xff) || 1;
    }

    const proxy = createSyncAccessExternalDataProxy(
      exactAlignedData.byteLength,
      (dest, offset) => {
        const n = Math.min(dest.byteLength, exactAlignedData.byteLength - offset);
        dest.set(exactAlignedData.subarray(offset, offset + n), 0);
        return n;
      },
      { scratchPool }
    );

    let createEncoderCalls = 0;
    let submitCalls = 0;
    const submittedBatches: unknown[][] = [];
    const uploadedBytes: number[] = [];

    const device = {
      createBuffer: vi.fn((desc: { size: number; usage: number; label?: string }) => ({
        size: desc.size,
        usage: desc.usage,
        label: desc.label ?? 'dst',
      })),
      createCommandEncoder: vi.fn(() => {
        createEncoderCalls += 1;
        return {
          copyBufferToBuffer: vi.fn(),
          finish: vi.fn(() => ({ kind: 'native-cmd' })),
        };
      }),
      queue: {
        writeBuffer: vi.fn(
          (_dst: unknown, _dstOff: number, data: ArrayBufferView | ArrayBuffer, dataOff = 0, len?: number) => {
            const view =
              data instanceof ArrayBuffer
                ? new Uint8Array(data, dataOff, len)
                : new Uint8Array(data.buffer, data.byteOffset + dataOff, len ?? data.byteLength);
            for (const b of view) {
              uploadedBytes.push(b);
            }
          }
        ),
        submit: vi.fn((cmds: Iterable<unknown>) => {
          submitCalls += 1;
          submittedBatches.push([...cmds]);
        }),
      },
    };

    const opt = installWebGpuUploadOptimizer(device, {
      chunkBytes: 2048,
      sharedScratchPool: scratchPool,
      maxRetainedScratchBytes: 8192,
    });

    // 1. Aligned > 4096 tensor: getMappedRange shares exact ArrayBuffer and preserves last 16 bytes
    const slice = proxy.subarray(0, 4112);
    const staging = device.createBuffer({
      size: 4112,
      usage: 6,
      mappedAtCreation: true,
    } as never) as unknown as {
      getMappedRange(): ArrayBuffer;
      unmap(): void;
      destroy(): void;
    };
    const mapped = staging.getMappedRange();
    expect(mapped).toBe(slice.buffer);
    new Uint8Array(mapped, 0, slice.byteLength).set(slice);
    staging.unmap();
    const enc = device.createCommandEncoder();
    const dst = device.createBuffer({ size: 4112, usage: 8, label: 'w1' });
    enc.copyBufferToBuffer(staging, 0, dst, 0, 4112);
    const cmd = (enc as unknown as { finish(): unknown }).finish();
    device.queue.submit([cmd]);
    staging.destroy();

    expect(Uint8Array.from(uploadedBytes)).toEqual(exactAlignedData);
    // Neither native createCommandEncoder nor native queue.submit was invoked for virtual copy!
    expect(createEncoderCalls).toBe(0);
    expect(submitCalls).toBe(0);

    // 2. Unaligned tensor > 8192 triggers immediate trim of the previous 16384-byte slab after upload
    uploadedBytes.length = 0;
    const unalignedData = new Uint8Array(8200);
    for (let i = 0; i < unalignedData.length; i += 1) {
      unalignedData[i] = (i & 0xfe) | 1;
    }
    const uProxy = createSyncAccessExternalDataProxy(
      unalignedData.byteLength,
      (dest, offset) => {
        dest.set(unalignedData.subarray(offset, offset + dest.byteLength), 0);
        return dest.byteLength;
      },
      { scratchPool }
    );
    const uSlice = uProxy.subarray(0, 8200);
    const uStaging = device.createBuffer({
      size: 8208,
      usage: 6,
      mappedAtCreation: true,
    } as never) as unknown as typeof staging;
    const uMapped = uStaging.getMappedRange();
    expect(uMapped).toBe(uSlice.buffer);
    new Uint8Array(uMapped, 0, 8200).set(uSlice);
    uStaging.unmap();
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(uStaging, 0, dst, 0, 8208);
    device.queue.submit([(enc2 as unknown as { finish(): unknown }).finish()]);
    expect(uploadedBytes.slice(0, 8200)).toEqual(Array.from(unalignedData));
    expect(uploadedBytes.slice(8200)).toEqual(new Array(8).fill(0));

    // After > 8192 tensor finished uploading, slab must have been trimmed (subsequent 64-byte acquire gets a new small buffer)
    const afterTrim = scratchPool.acquire(64);
    expect(afterTrim.buffer).not.toBe(uMapped);
    expect(DEFAULT_RETAINED_SCRATCH_MAX_BYTES).toBe(18 * 1024 * 1024);

    // 3. Mixed batch with a real native encoder command materializes encoder and preserves real submit only
    const encReal = device.createCommandEncoder();
    const realCmd = (encReal as unknown as { finish(): unknown }).finish();
    device.queue.submit([cmd, realCmd]);
    expect(submitCalls).toBe(1);
    expect(submittedBatches[0]).toEqual([realCmd]);

    opt.dispose();
  });

  it('streams remote slices with createRemoteRangeBlob LRU cache and selects quota-aware embed fallback accurately', async () => {
    expect(parseHttpRangeTotalBytes('bytes 0-0/1590689792')).toBe(1590689792);
    expect(parseHttpRangeTotalBytes('invalid')).toBeUndefined();

    const fullRemote = new Uint8Array(256);
    for (let i = 0; i < fullRemote.length; i += 1) {
      fullRemote[i] = i;
    }

    const fetchedRanges: string[] = [];
    const rangeFetch = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      const hdr = new Headers(init?.headers).get('Range') ?? '';
      fetchedRanges.push(hdr);
      const m = /^bytes=(\d+)-(\d+)$/.exec(hdr);
      if (!m) {
        return new Response(null, { status: 400 });
      }
      const start = Number(m[1]);
      const end = Math.min(fullRemote.length - 1, Number(m[2]));
      return new Response(fullRemote.subarray(start, end + 1), {
        status: 206,
        headers: { 'content-range': `bytes ${start}-${end}/${fullRemote.length}` },
      });
    }) as unknown as typeof fetch;

    // Resolver detects HTTP 206 support and returns a Range Blob without downloading all 256 bytes at once
    const resolver = createExternalBlobResolver({
      urlByLocation: new Map([
        ['embed.onnx_data', 'https://huggingface.co/test/resolve/main/onnx/embed.onnx_data'],
      ]),
      cacheName: 'empty-cache',
      cacheStorage: null,
      fetchFn: rangeFetch,
    });
    const resolvedBlob = await resolver('embed.onnx_data');
    expect(resolvedBlob?.size).toBe(256);

    // Bounded LRU test: 4096-byte pages with 4096-byte cap evicts page 0 when page 1 is loaded
    const bigRemote = new Uint8Array(8192);
    for (let i = 0; i < bigRemote.length; i += 1) {
      bigRemote[i] = (i * 3) & 0xff;
    }
    const lruRanges: string[] = [];
    const lruFetch = vi.fn(async (_u: RequestInfo | URL, init?: RequestInit) => {
      const hdr = new Headers(init?.headers).get('Range') ?? '';
      lruRanges.push(hdr);
      const m = /^bytes=(\d+)-(\d+)$/.exec(hdr)!;
      const start = Number(m[1]);
      const end = Math.min(bigRemote.length - 1, Number(m[2]));
      return new Response(bigRemote.subarray(start, end + 1), {
        status: 206,
        headers: { 'content-range': `bytes ${start}-${end}/${bigRemote.length}` },
      });
    }) as unknown as typeof fetch;
    const lruBlob = createRemoteRangeBlob(
      'https://huggingface.co/test/resolve/main/onnx/embed_lru.onnx_data',
      bigRemote.length,
      lruFetch,
      { blockBytes: 4096, maxCacheBytes: 4096 }
    );
    expect(new Uint8Array(await lruBlob.slice(0, 16).arrayBuffer())).toEqual(
      bigRemote.subarray(0, 16)
    );
    // Same page 0 hit should NOT fire another HTTP request
    expect(new Uint8Array(await lruBlob.slice(100, 116).arrayBuffer())).toEqual(
      bigRemote.subarray(100, 116)
    );
    expect(lruRanges).toEqual(['bytes=0-4095']);
    // Reading page 1 evicts page 0, so reading page 0 again fetches page 0 once more
    expect(new Uint8Array(await lruBlob.slice(4096, 4112).arrayBuffer())).toEqual(
      bigRemote.subarray(4096, 4112)
    );
    expect(new Uint8Array(await lruBlob.slice(0, 16).arrayBuffer())).toEqual(
      bigRemote.subarray(0, 16)
    );
    expect(lruRanges).toEqual(['bytes=0-4095', 'bytes=4096-8191', 'bytes=0-4095']);

    fetchedRanges.length = 0;
    const directRemote = createRemoteRangeBlob(
      'https://huggingface.co/test/resolve/main/onnx/embed.onnx_data',
      256,
      rangeFetch,
      { blockBytes: 4096, maxCacheBytes: 0 }
    );
    const s1 = new Uint8Array(await directRemote.slice(10, 20).arrayBuffer());
    const sNeg = new Uint8Array(await directRemote.slice(-4).arrayBuffer());
    const sEmpty = new Uint8Array(await directRemote.slice(50, 50).arrayBuffer());
    expect(s1).toEqual(fullRemote.subarray(10, 20));
    expect(sNeg).toEqual(fullRemote.subarray(252));
    expect(sEmpty.byteLength).toBe(0);
    expect(fetchedRanges).toEqual(['bytes=10-19', 'bytes=252-255']);

    // Guard against accidental full multi-GiB buffering on remote Range Blob
    const hugeRemote = createRemoteRangeBlob('https://example.com/huge.bin', 64 * 1024 * 1024, rangeFetch);
    await expect(hugeRemote.arrayBuffer()).rejects.toThrow(/Refusing to buffer/);

    // Virtualization allowRemoteFallback matches only designated embed_tokens URLs when missing from disk
    const remoteAdapter = createSerializedCacheAdapter('empty', undefined, {
      opfsRoot: null,
      virtualizeExternalData: {
        allowRemoteFallback: (u) => /embed_tokens[^/]*\.onnx_data/.test(u),
      },
    })!;
    const embedUrl =
      'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/embed_tokens_q4f16.onnx_data';
    const decoderUrl =
      'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data';
    expect((await remoteAdapter.match(embedUrl)) as unknown).toBe(embedUrl);
    expect(await remoteAdapter.match(decoderUrl)).toBeUndefined();

    // Quota-aware plan falls back to remote Range embed_tokens when quota fits decoder (~2.0 GB) but not embed+decoder (~3.1 GB)
    const gemmaE2b = getModelById('gemma-4-e2b');
    const e2bResolved = resolveEffectiveDtype(
      gemmaE2b,
      { backend: 'webgpu', supportsFp16: true, features: ['shader-f16'] },
      'q4f16'
    );
    const modelFiles = [
      'onnx/embed_tokens_q4f16.onnx',
      'onnx/embed_tokens_q4f16.onnx_data',
      'onnx/decoder_model_merged_q4f16.onnx',
      'onnx/decoder_model_merged_q4f16.onnx_data',
      'tokenizer.json',
    ];
    const tightPlan = resolveQuotaAwareEmbedPlan({
      modelSpec: gemmaE2b,
      resolved: e2bResolved,
      modelFiles,
      availableBytes: 2_000_000_000,
      cacheInfos: [],
    });
    expect(tightPlan.useRemoteEmbedRangeFallback).toBe(true);
    expect(tightPlan.filesToPrecache).toEqual([
      'onnx/embed_tokens_q4f16.onnx',
      'onnx/decoder_model_merged_q4f16.onnx',
      'onnx/decoder_model_merged_q4f16.onnx_data',
      'tokenizer.json',
    ]);
    expect(tightPlan.notice).toContain('HTTP Range');

    // Too small even for decoder -> throws STORAGE_QUOTA_EXCEEDED
    expect(() =>
      resolveQuotaAwareEmbedPlan({
        modelSpec: gemmaE2b,
        resolved: e2bResolved,
        modelFiles,
        availableBytes: 500_000_000,
        cacheInfos: [],
      })
    ).toThrow(expect.objectContaining({ code: 'STORAGE_QUOTA_EXCEEDED' }));
  });
});

describe('Sync external-data reliability, Cache->OPFS migration & Range fallback', () => {
  it('probeSyncReadHandle validates 0-byte, 1-byte, multi-chunk handles and throws preserving NotReadableError or 0-byte EOF', () => {
    const emptyHandle: OpfsSyncReadHandle = {
      byteLength: 0,
      readAt: () => 0,
      close: vi.fn(),
    };
    expect(() => probeSyncReadHandle(emptyHandle, 'empty.onnx_data')).not.toThrow();

    const tinyBytes = new Uint8Array([42]);
    const tinyOffsets: number[] = [];
    const tinyHandle: OpfsSyncReadHandle = {
      byteLength: 1,
      readAt: (dest, offset) => {
        tinyOffsets.push(offset);
        dest.set(tinyBytes.subarray(offset, offset + dest.byteLength), 0);
        return dest.byteLength;
      },
      close: vi.fn(),
    };
    probeSyncReadHandle(tinyHandle, 'tiny.onnx_data');
    expect(tinyOffsets).toEqual([0]);

    const midOffsets: number[] = [];
    const largeHandle: OpfsSyncReadHandle = {
      byteLength: 4096,
      readAt: (dest, offset) => {
        midOffsets.push(offset);
        return dest.byteLength;
      },
      close: vi.fn(),
    };
    probeSyncReadHandle(largeHandle, 'large.onnx_data');
    expect(midOffsets).toEqual([0, 2040, 4032]);

    const tailFailingHandle: OpfsSyncReadHandle = {
      byteLength: 4096,
      readAt: (dest, offset) => {
        if (offset >= 4000) {
          throw new DOMException('Tail slice unreadable', 'NotReadableError');
        }
        return dest.byteLength;
      },
      close: vi.fn(),
    };
    expect(() => probeSyncReadHandle(tailFailingHandle, 'broken.onnx_data')).toThrow(
      expect.objectContaining({ name: 'NotReadableError' })
    );

    const eofHandle: OpfsSyncReadHandle = {
      byteLength: 1024,
      readAt: () => 0,
      close: vi.fn(),
    };
    expect(() => probeSyncReadHandle(eofHandle, 'eof.onnx_data')).toThrow(
      /Sync probe returned 0 bytes/
    );
  });

  it('tags OPFS sync vs blob sources, retries FileReaderSync once, and wraps permanent NotReadableError', async () => {
    const { root } = createMockOpfsRoot();
    const url = 'https://example.com/model/decoder.onnx_data';
    const data = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    await writeResponseToOpfsStore(root, url, new Response(data));

    const opfsHandle = await openOpfsSyncReadHandle(root, url);
    expect(opfsHandle?.source).toBe('opfs-sync-access');
    opfsHandle?.close();

    // Without createSyncAccessHandle at any OPFS directory depth, openOpfsSyncReadHandle falls back to blob-file-reader
    const wrapDirWithoutSync = (dir: OpfsDirectoryLike): OpfsDirectoryLike => ({
      async getDirectoryHandle(name, opts) {
        return wrapDirWithoutSync(await dir.getDirectoryHandle!(name, opts));
      },
      async getFileHandle(fileName, fileOpts) {
        const fh = await dir.getFileHandle!(fileName, fileOpts);
        return {
          getFile: fh.getFile,
          createWritable: fh.createWritable,
        };
      },
      async removeEntry(name, opts) {
        return dir.removeEntry(name, opts);
      },
    });
    const blobOnlyRoot = wrapDirWithoutSync(root);

    const g = globalThis as { FileReaderSync?: unknown };
    const origFrs = g.FileReaderSync;
    let transientCalls = 0;
    class RetryingMockFileReaderSync {
      readAsArrayBuffer(blobSlice: Blob): ArrayBuffer {
        transientCalls += 1;
        if (transientCalls === 1) {
          throw new DOMException('Transient read lock', 'NotReadableError');
        }
        return data.subarray(0, blobSlice.size).slice().buffer;
      }
    }
    try {
      g.FileReaderSync = RetryingMockFileReaderSync;
      const fallbackHandle = await openOpfsSyncReadHandle(blobOnlyRoot, url);
      expect(fallbackHandle?.source).toBe('blob-file-reader');
      const dst = new Uint8Array(4);
      expect(fallbackHandle!.readAt(dst, 0)).toBe(4);
      expect(transientCalls).toBe(2);
      expect(dst).toEqual(new Uint8Array([1, 2, 3, 4]));
      fallbackHandle?.close();

      class AlwaysFailingFileReaderSync {
        readAsArrayBuffer(): ArrayBuffer {
          throw new DOMException('The requested file could not be read', 'NotReadableError');
        }
      }
      g.FileReaderSync = AlwaysFailingFileReaderSync;
      const brokenBlobHandle = createBlobSyncReadHandle(new Blob([data]), url, data.byteLength);
      expect(() => brokenBlobHandle!.readAt(new Uint8Array(4), 0)).toThrow(
        expect.objectContaining({
          name: 'NotReadableError',
        })
      );
      expect(() => brokenBlobHandle!.readAt(new Uint8Array(4), 0)).toThrow(
        /FileReaderSync failed reading .* \[0\.\.4\/8\]: NotReadableError/
      );
    } finally {
      g.FileReaderSync = origFrs;
    }
  });

  it('writeResponseToOpfsStore streams null-body CacheStorage responses via blob.stream() or sliced blob without arrayBuffer()', async () => {
    const payload = new Uint8Array(1024 * 1024 + 321);
    for (let i = 0; i < payload.length; i += 1) {
      payload[i] = (i * 13) & 0xff;
    }

    // 1. Response with body === null and blob.stream() present
    const { root: root1 } = createMockOpfsRoot();
    const arrayBufferSpy1 = vi.fn(async () => {
      throw new Error('Must not call arrayBuffer()');
    });
    const fakeBlob1 = new Blob([payload]);
    const respViaBlobStream = {
      ok: true,
      status: 200,
      headers: new Headers({ 'content-length': String(payload.byteLength) }),
      body: null,
      blob: async () => fakeBlob1,
      arrayBuffer: arrayBufferSpy1,
    } as unknown as Response;

    const url1 = 'https://example.com/m/stream_from_blob.onnx_data';
    const written1 = await writeResponseToOpfsStore(root1, url1, respViaBlobStream);
    expect(written1).toBe(payload.byteLength);
    expect(arrayBufferSpy1).not.toHaveBeenCalled();

    const handle1 = await openOpfsSyncReadHandle(root1, url1);
    const out1 = new Uint8Array(payload.byteLength);
    expect(handle1!.readAt(out1, 0)).toBe(payload.byteLength);
    expect(out1).toEqual(payload);
    handle1?.close();

    // 2. Response with body === null AND no blob.stream() -> falls back to sliced blob chunks
    const { root: root2 } = createMockOpfsRoot();
    const sliceCalls: Array<[number, number]> = [];
    const blobWithoutStream = {
      size: payload.byteLength,
      slice(start?: number, end?: number) {
        const s = start ?? 0;
        const e = end ?? payload.byteLength;
        sliceCalls.push([s, e]);
        return {
          async arrayBuffer() {
            return payload.subarray(s, e).slice().buffer;
          },
        };
      },
    } as unknown as Blob;
    const respViaSlicedBlob = {
      ok: true,
      status: 200,
      headers: new Headers({ 'content-length': String(payload.byteLength) }),
      body: null,
      blob: async () => blobWithoutStream,
      arrayBuffer: vi.fn(async () => {
        throw new Error('Must not call full arrayBuffer()');
      }),
    } as unknown as Response;

    const url2 = 'https://example.com/m/sliced_blob.onnx_data';
    const written2 = await writeResponseToOpfsStore(root2, url2, respViaSlicedBlob);
    expect(written2).toBe(payload.byteLength);
    expect(sliceCalls).toEqual([[0, payload.byteLength]]);

    const handle2 = await openOpfsSyncReadHandle(root2, url2);
    const out2 = new Uint8Array(payload.byteLength);
    expect(handle2!.readAt(out2, 0)).toBe(payload.byteLength);
    expect(out2).toEqual(payload);
    handle2?.close();
  });

  it('migrateCacheResponseToOpfs and precacheModelFilesSequentially migrate CacheStorage .onnx_data and preserve cache on QuotaExceededError', async () => {
    const url =
      'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data';
    const bytes = new Uint8Array([9, 8, 7, 6, 5, 4, 3, 2]);

    const cacheStorage = createMockCacheStorage({ [url]: bytes });
    const cache = await cacheStorage.open('transformers-cache');
    const { root } = createMockOpfsRoot();

    // QuotaExceededError during OPFS write must return false and preserve CacheStorage entry
    const quotaRoot: OpfsDirectoryLike = {
      async getDirectoryHandle() {
        throw new DOMException('OPFS quota exceeded', 'QuotaExceededError');
      },
      async getFileHandle() {
        throw new DOMException('OPFS quota exceeded', 'QuotaExceededError');
      },
      async removeEntry() {},
    };
    const quotaOk = await migrateCacheResponseToOpfs({
      opfsRoot: quotaRoot,
      url,
      cachedResponse: (await cache.match(url))!,
      cache,
    });
    expect(quotaOk).toBe(false);
    expect(await cache.match(url)).toBeDefined();

    // precacheModelFilesSequentially migrates legacy CacheStorage hit -> OPFS without fetching
    const fetchSpy = vi.fn(async () => {
      throw new Error('Should not fetch cached file');
    }) as unknown as typeof fetch;
    const precacheRes = await precacheModelFilesSequentially({
      repoId: 'onnx-community/gemma-4-E2B-it-ONNX',
      files: ['onnx/decoder_model_merged_q4f16.onnx_data'],
      cacheStorage,
      opfsRoot: root,
      fetchFn: fetchSpy,
    });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(precacheRes.skippedCachedFiles).toBe(1);
    expect(await cache.match(url)).toBeUndefined();

    const opfsHandle = await openOpfsSyncReadHandle(root, url);
    expect(opfsHandle?.source).toBe('opfs-sync-access');
    const verified = new Uint8Array(bytes.byteLength);
    expect(opfsHandle!.readAt(verified, 0)).toBe(bytes.byteLength);
    expect(verified).toEqual(bytes);
    opfsHandle?.close();
  });

  it('createSyncHttpRangeReadHandle discovers length via HEAD or bytes=0-0, stitches multi-chunk reads, retries 503, and rejects OOB/404/short responses', () => {
    const backing = new Uint8Array(25);
    for (let i = 0; i < backing.length; i += 1) {
      backing[i] = i + 10;
    }

    const requests: Array<{ method: string; range: string | null }> = [];
    let forceTransient503Once = true;
    const makeXhr = (omitHeadLength: boolean, force200OnGet = false) => () => {
      let method = 'GET';
      let rangeHeader: string | null = null;
      const xhr = {
        status: 0,
        responseType: '' as XMLHttpRequestResponseType,
        response: null as unknown,
        open(m: string) {
          method = m;
        },
        setRequestHeader(name: string, value: string) {
          if (name.toLowerCase() === 'range') {
            rangeHeader = value;
          }
        },
        getResponseHeader(name: string) {
          const lower = name.toLowerCase();
          if (method === 'HEAD' && lower === 'content-length') {
            return omitHeadLength ? null : String(backing.byteLength);
          }
          if (lower === 'content-range' && rangeHeader === 'bytes=0-0') {
            return `bytes 0-0/${backing.byteLength}`;
          }
          return null;
        },
        send() {
          requests.push({ method, range: rangeHeader });
          if (method === 'HEAD') {
            xhr.status = 200;
            return;
          }
          if (rangeHeader === 'bytes=0-0') {
            xhr.status = 206;
            xhr.response = backing.subarray(0, 1).slice().buffer;
            return;
          }
          if (forceTransient503Once && rangeHeader === 'bytes=8-13') {
            forceTransient503Once = false;
            xhr.status = 503;
            return;
          }
          if (force200OnGet) {
            xhr.status = 200;
            xhr.response = backing.slice().buffer;
            return;
          }
          const match = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader ?? '');
          if (!match) {
            xhr.status = 400;
            return;
          }
          const start = Number(match[1]);
          const end = Math.min(backing.byteLength - 1, Number(match[2]));
          xhr.status = 206;
          xhr.response = backing.subarray(start, end + 1).slice().buffer;
        },
      };
      return xhr;
    };

    // HEAD missing Content-Length -> falls back to Range: bytes=0-0 Content-Range probe
    const handle = createSyncHttpRangeReadHandle('https://example.com/weights.onnx_data', {
      maxChunkBytes: 6,
      xhrFactory: makeXhr(true, false),
    });
    expect(handle?.source).toBe('remote-range');
    expect(handle?.byteLength).toBe(25);

    // Multi-chunk read across 3 HTTP Range requests + transient 503 retry
    const dest = new Uint8Array(15);
    expect(handle!.readAt(dest, 2)).toBe(15);
    expect(dest).toEqual(backing.subarray(2, 17));
    expect(requests.filter((r) => r.range === 'bytes=8-13')).toHaveLength(2);

    // Boundary: out-of-range offset throws, 0-length succeeds, closed handle throws
    expect(() => handle!.readAt(new Uint8Array(1), -1)).toThrow(/Out-of-range/);
    expect(() => handle!.readAt(new Uint8Array(1), 26)).toThrow(/Out-of-range/);
    expect(handle!.readAt(new Uint8Array(0), 25)).toBe(0);
    handle!.close();
    expect(() => handle!.readAt(new Uint8Array(4), 0)).toThrow(/already closed/);

    // HTTP 200 full-response slicing still yields requested window
    const h200 = createSyncHttpRangeReadHandle('https://example.com/full200.onnx_data', {
      expectedByteLength: 25,
      xhrFactory: makeXhr(false, true),
    });
    const out200 = new Uint8Array(5);
    expect(h200!.readAt(out200, 10)).toBe(5);
    expect(out200).toEqual(backing.subarray(10, 15));
    h200!.close();

    // Permanent 404 or short 206 throws SYNC_HTTP_RANGE_FAILED
    const failing404 = createSyncHttpRangeReadHandle('https://example.com/missing.onnx_data', {
      expectedByteLength: 10,
      xhrFactory: () => {
        const x = {
          status: 404,
          responseType: '' as XMLHttpRequestResponseType,
          response: null,
          open: vi.fn(),
          setRequestHeader: vi.fn(),
          getResponseHeader: () => null,
          send: vi.fn(),
        };
        return x;
      },
    });
    expect(() => failing404!.readAt(new Uint8Array(4), 0)).toThrow(
      expect.objectContaining({ code: 'SYNC_HTTP_RANGE_FAILED' })
    );
  });

  it('ExternalDataSessionBridge migrates CacheStorage -> OPFS, retries transient lock, falls back to Range on NotReadableError without 1.4GB RAM copy, and retains errors across closeAll()', async () => {
    const decoderUrl =
      'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data';
    const payload = new Uint8Array([11, 22, 33, 44, 55, 66, 77, 88, 99, 100]);

    // Case 1: CacheStorage hit migrates into OPFS SyncAccessHandle so FileReaderSync is never touched
    const { root } = createMockOpfsRoot();
    const cacheStorage = createMockCacheStorage({ [decoderUrl]: payload });
    const bridge1 = new ExternalDataSessionBridge();
    bridge1.registerUrl('decoder_model_merged_q4f16.onnx_data', decoderUrl);

    const g = globalThis as { FileReaderSync?: unknown; caches?: unknown; XMLHttpRequest?: unknown };
    const origFrs = g.FileReaderSync;
    const origCaches = g.caches;
    const origXhr = g.XMLHttpRequest;
    const frsCtorSpy = vi.fn();
    g.FileReaderSync = class FailIfUsedFileReaderSync {
      constructor() {
        frsCtorSpy();
      }
      readAsArrayBuffer(): ArrayBuffer {
        throw new Error('Should have migrated to OPFS SyncAccessHandle');
      }
    };
    g.caches = cacheStorage;

    try {
      const sessionOpts1: { externalData: Array<string | { path: string; data: Uint8Array }> } = {
        externalData: ['decoder_model_merged_q4f16.onnx_data'],
      };
      const release1 = await bridge1.prepareSessionExternalData(sessionOpts1, {
        opfsRoot: root,
        cacheName: 'transformers-cache',
      });
      const entryData = (sessionOpts1.externalData[0] as { path: string; data: Uint8Array }).data;
      expect(entryData.subarray(2, 6)).toEqual(payload.subarray(2, 6));
      expect(frsCtorSpy).not.toHaveBeenCalled();
      const cacheAfter = await cacheStorage.open('transformers-cache');
      expect(await cacheAfter.match(decoderUrl)).toBeUndefined();
      release1();
      bridge1.closeAll();

      // Case 1b: Transient SyncAccessHandle contention ('blob-file-reader' first -> 'opfs-sync-access' on 20ms retry)
      let binHandleRequests = 0;
      const wrapTransientSyncLock = (dir: OpfsDirectoryLike): OpfsDirectoryLike => ({
        async getDirectoryHandle(name, opts) {
          return wrapTransientSyncLock(await dir.getDirectoryHandle!(name, opts));
        },
        async getFileHandle(fileName, fileOpts) {
          const fh = await dir.getFileHandle!(fileName, fileOpts);
          if (!fileName.endsWith('.bin') || fileOpts?.create) {
            return fh;
          }
          binHandleRequests += 1;
          const failSyncOnce = binHandleRequests === 1;
          return {
            getFile: fh.getFile,
            createWritable: fh.createWritable,
            async createSyncAccessHandle() {
              if (failSyncOnce) {
                throw new DOMException('NoModificationAllowedError lock held briefly', 'NoModificationAllowedError');
              }
              return fh.createSyncAccessHandle!();
            },
          };
        },
        async removeEntry(name, opts) {
          return dir.removeEntry(name, opts);
        },
      });
      const retryBridge = new ExternalDataSessionBridge();
      retryBridge.registerUrl('decoder_model_merged_q4f16.onnx_data', decoderUrl);
      const retryOpts: { externalData: Array<string | { path: string; data: Uint8Array }> } = {
        externalData: ['decoder_model_merged_q4f16.onnx_data'],
      };
      await retryBridge.prepareSessionExternalData(retryOpts, {
        opfsRoot: wrapTransientSyncLock(root),
        cacheName: 'transformers-cache',
      });
      expect(binHandleRequests).toBe(2);
      expect(
        (retryOpts.externalData[0] as { path: string; data: Uint8Array }).data.subarray(0, 4)
      ).toEqual(payload.subarray(0, 4));
      expect(frsCtorSpy).not.toHaveBeenCalled();
      retryBridge.closeAll();

      // Case 2: CacheStorage-only FileReaderSync fails NotReadableError -> falls back cleanly to sync Range without arrayBuffer() OOM
      const bigBytesHeaderOnly = new Headers({
        'content-length': String(128 * 1024 * 1024),
      });
      const arrayBufferOomTrap = vi.fn(async () => {
        throw new Error('arrayBuffer() must NEVER be called on >64MiB external data');
      });
      const fakeBigCache: Cache = {
        match: vi.fn(async () => ({
          ok: true,
          status: 200,
          headers: bigBytesHeaderOnly,
          blob: async () => ({ size: 128 * 1024 * 1024, slice: () => new Blob([new Uint8Array(64)]) }),
          arrayBuffer: arrayBufferOomTrap,
          clone() {
            return this;
          },
        })) as unknown as Cache['match'],
        delete: vi.fn(async () => true),
      } as unknown as Cache;
      g.caches = { open: vi.fn(async () => fakeBigCache) };
      g.FileReaderSync = class NotReadableFrs {
        readAsArrayBuffer(): ArrayBuffer {
          throw new DOMException('The requested file could not be read', 'NotReadableError');
        }
      };
      g.XMLHttpRequest = class MockSyncRangeXhr {
        status = 0;
        responseType = '';
        response: unknown = null;
        private rangeHeader: string | null = null;
        private method = 'GET';
        open(m: string) {
          this.method = m;
        }
        setRequestHeader(k: string, v: string) {
          if (k.toLowerCase() === 'range') {
            this.rangeHeader = v;
          }
        }
        getResponseHeader(name: string) {
          if (this.method === 'HEAD' && name.toLowerCase() === 'content-length') {
            return String(128 * 1024 * 1024);
          }
          return null;
        }
        send() {
          if (this.method === 'HEAD') {
            this.status = 200;
            return;
          }
          const m = /^bytes=(\d+)-(\d+)$/.exec(this.rangeHeader ?? '');
          const start = m ? Number(m[1]) : 0;
          const end = m ? Number(m[2]) : 0;
          const slice = new Uint8Array(end - start + 1);
          for (let i = 0; i < slice.length; i += 1) {
            slice[i] = (start + i) & 0xff;
          }
          this.status = 206;
          this.response = slice.buffer;
        }
      };

      const bridge2 = new ExternalDataSessionBridge();
      bridge2.registerUrl('decoder_model_merged_q4f16.onnx_data', decoderUrl);
      const rangeSessionOpts: { externalData: Array<string | { path: string; data: Uint8Array }> } = {
        externalData: ['decoder_model_merged_q4f16.onnx_data'],
      };
      await bridge2.prepareSessionExternalData(rangeSessionOpts, {
        opfsRoot: null,
        cacheName: 'transformers-cache',
      });
      const rangeProxy = (rangeSessionOpts.externalData[0] as { path: string; data: Uint8Array }).data;
      expect(rangeProxy.byteLength).toBe(128 * 1024 * 1024);
      expect(rangeProxy.subarray(200, 204)).toEqual(new Uint8Array([200, 201, 202, 203]));
      expect(arrayBufferOomTrap).not.toHaveBeenCalled();

      // Case 3: Mid-session proxy error persists across closeAll() until consumeLastReadError()
      bridge2.closeAll();
      const postCloseProxy = (rangeSessionOpts.externalData[0] as { path: string; data: Uint8Array })
        .data;
      expect(() => postCloseProxy.subarray(0, 4)).toThrow(/already closed/);
      const consumed = bridge2.consumeLastReadError();
      expect(consumed).not.toBeNull();
      expect(consumed!.url).toBe(decoderUrl);
      expect(consumed!.offset).toBe(0);
      expect(consumed!.length).toBe(4);
      expect(bridge2.consumeLastReadError()).toBeNull();

      // Case 4: Both local sync NotReadableError & Range unavailable -> throws original NotReadableError
      g.XMLHttpRequest = undefined;
      const bridge3 = new ExternalDataSessionBridge();
      bridge3.registerUrl('decoder_model_merged_q4f16.onnx_data', decoderUrl);
      await expect(
        bridge3.prepareSessionExternalData(
          { externalData: ['decoder_model_merged_q4f16.onnx_data'] },
          { opfsRoot: null, cacheName: 'transformers-cache' }
        )
      ).rejects.toMatchObject({ name: 'NotReadableError' });
      bridge3.closeAll();
    } finally {
      g.FileReaderSync = origFrs;
      g.caches = origCaches;
      g.XMLHttpRequest = origXhr;
    }
  });

  it('enrichModelLoadError + classifyModelLoadError surface swallowed memory-copy root causes cleanly', () => {
    const ortMemCopyErr = new Error(
      "Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: Deserialize tensor model_layers_0_attn_k_proj_MatMul_weight_zp failed.Failed to load external data file \"decoder_model_merged_q4f16.onnx_data\", error: Unknown error occurred in memory copy."
    );
    const notReadable = new DOMException('The requested file could not be read', 'NotReadableError');
    const enrichedRead = enrichModelLoadError(
      ortMemCopyErr,
      {
        url: 'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data',
        offset: 211876864,
        length: 6144,
        error: notReadable,
      },
      null
    );
    const classifiedRead = classifyModelLoadError(enrichedRead);
    expect(classifiedRead.code).toBe('EXTERNAL_DATA_READ_FAILED');
    expect(classifiedRead.message).toContain('NotReadableError');
    expect(classifiedRead.message).toContain('211876864+6144');

    // Enriched WebGPU device lost or limit error retains high-priority specific classification
    const enrichedLost = enrichModelLoadError(
      ortMemCopyErr,
      null,
      new Error('Queue submit failed: WebGPU device is lost')
    );
    expect(classifyModelLoadError(enrichedLost).code).toBe('WEBGPU_DEVICE_LOST');

    // Un-enriched ORT memory copy error also classifies as EXTERNAL_DATA_READ_FAILED
    expect(classifyModelLoadError(ortMemCopyErr).code).toBe('EXTERNAL_DATA_READ_FAILED');
  });
});
