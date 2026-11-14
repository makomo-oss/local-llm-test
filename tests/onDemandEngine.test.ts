import { Tensor as OrtTensor } from 'onnxruntime-web/webgpu';
import { describe, expect, it, vi } from 'vitest';
import {
  createExternalBlobResolver,
  createOnDemandEmbedTokensSession,
  decodeFp16Bits,
  parseOnnxEmbedTokensPlan,
} from '../src/engine/embedTokensSession.ts';
import {
  ExternalDataSessionBridge,
  createSyncAccessExternalDataProxy,
} from '../src/engine/externalDataProxy.ts';
import { classifyModelLoadError } from '../src/engine/llm.worker.ts';
import { createSerializedCacheAdapter } from '../src/engine/modelPrecacher.ts';
import {
  writeResponseToOpfsStore,
  type OpfsDirectoryLike,
  type OpfsFileHandleLike,
  type OpfsSyncAccessHandleLike,
  type OpfsWritableStreamLike,
} from '../src/engine/opfsCacheStore.ts';
import {
  SessionLoadCoordinator,
  withSerializedInferenceSessions,
} from '../src/engine/sessionSerializer.ts';
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
    expect(() => brokenProxy.subarray(0, 8)).toThrow(/Short synchronous OPFS/);
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
  });
});
