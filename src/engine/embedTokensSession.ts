import { Tensor as OrtTensor } from 'onnxruntime-web/webgpu';
import { openOpfsCachedBlob, type OpfsDirectoryLike } from './opfsCacheStore.ts';

const ONNX_DT = {
  FLOAT: 1,
  UINT8: 2,
  INT32: 6,
  INT64: 7,
  FLOAT16: 10,
  UINT4: 21,
  INT4: 22,
} as const;

const MAX_EMBED_ROW_CACHE_BYTES = 8 * 1024 * 1024; // 8 MiB bounded row cache

const UTF8_DECODER = new TextDecoder();

/**
 * 65536-entry IEEE-754 binary16 -> binary32 lookup table built once on demand (~256 KiB).
 */
let fp16ToF32Lut: Float32Array | null = null;

function getFp16Lut(): Float32Array {
  if (fp16ToF32Lut) {
    return fp16ToF32Lut;
  }
  const lut = new Float32Array(65536);
  for (let h = 0; h < 65536; h += 1) {
    lut[h] = decodeFp16Bits(h);
  }
  fp16ToF32Lut = lut;
  return lut;
}

export function decodeFp16Bits(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x03ff;
  if (exp === 0) {
    return mant === 0 ? (sign < 0 ? -0 : 0) : sign * Math.pow(2, -14) * (mant / 1024);
  }
  if (exp === 0x1f) {
    return mant === 0 ? sign * Infinity : NaN;
  }
  return sign * Math.pow(2, exp - 15) * (1 + mant / 1024);
}

interface ProtoField {
  fn: number;
  wt: number;
  varint?: bigint;
  bytes?: Uint8Array;
}

function readVarint(buf: Uint8Array, pos: number): [bigint, number] {
  let result = 0n;
  let shift = 0n;
  let p = pos;
  while (p < buf.length) {
    const byte = buf[p]!;
    p += 1;
    result |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      return [result, p];
    }
    shift += 7n;
    if (shift > 70n) {
      throw new Error('Malformed protobuf varint');
    }
  }
  throw new Error('Truncated protobuf varint');
}

function asSignedInt64(v: bigint): bigint {
  // Varints in protobuf represent 64-bit signed values in two's complement unsigned encoding.
  return BigInt.asIntN(64, v);
}

function decodeProtoFields(buf: Uint8Array): ProtoField[] {
  const fields: ProtoField[] = [];
  let p = 0;
  while (p < buf.length) {
    const [tag, afterTag] = readVarint(buf, p);
    p = afterTag;
    const fn = Number(tag >> 3n);
    const wt = Number(tag & 7n);
    if (wt === 0) {
      const [varint, next] = readVarint(buf, p);
      fields.push({ fn, wt, varint });
      p = next;
    } else if (wt === 2) {
      const [lenBig, afterLen] = readVarint(buf, p);
      const len = Number(lenBig);
      if (!Number.isSafeInteger(len) || len < 0 || afterLen + len > buf.length) {
        throw new Error('Truncated protobuf length-delimited field');
      }
      fields.push({ fn, wt, bytes: buf.subarray(afterLen, afterLen + len) });
      p = afterLen + len;
    } else if (wt === 1) {
      if (p + 8 > buf.length) throw new Error('Truncated protobuf 64-bit field');
      fields.push({ fn, wt, bytes: buf.subarray(p, p + 8) });
      p += 8;
    } else if (wt === 5) {
      if (p + 4 > buf.length) throw new Error('Truncated protobuf 32-bit field');
      fields.push({ fn, wt, bytes: buf.subarray(p, p + 4) });
      p += 4;
    } else {
      throw new Error(`Unsupported protobuf wire type ${wt}`);
    }
  }
  return fields;
}

interface ParsedTensor {
  name: string;
  dims: number[];
  dataType: number;
  rawData?: Uint8Array;
  floatData: number[];
  int64Data: bigint[];
  external?: {
    location: string;
    offset: number;
    length: number;
  };
}

interface ParsedAttr {
  name: string;
  i?: bigint;
  f?: number;
}

interface ParsedNode {
  opType: string;
  domain: string;
  inputs: string[];
  outputs: string[];
  attrs: Record<string, ParsedAttr>;
}

function parseTensorProto(buf: Uint8Array): ParsedTensor {
  let name = '';
  const dims: number[] = [];
  let dataType = 0;
  let rawData: Uint8Array | undefined;
  const floatData: number[] = [];
  const int64Data: bigint[] = [];
  const ext: Record<string, string> = {};

  for (const field of decodeProtoFields(buf)) {
    if (field.fn === 1) {
      if (field.wt === 0 && field.varint !== undefined) {
        dims.push(Number(asSignedInt64(field.varint)));
      } else if (field.wt === 2 && field.bytes) {
        let p = 0;
        while (p < field.bytes.length) {
          const [v, next] = readVarint(field.bytes, p);
          dims.push(Number(asSignedInt64(v)));
          p = next;
        }
      }
    } else if (field.fn === 2 && field.varint !== undefined) {
      dataType = Number(field.varint);
    } else if (field.fn === 4 && field.bytes) {
      const view = new DataView(field.bytes.buffer, field.bytes.byteOffset, field.bytes.byteLength);
      for (let o = 0; o + 4 <= field.bytes.byteLength; o += 4) {
        floatData.push(view.getFloat32(o, true));
      }
    } else if (field.fn === 7) {
      if (field.wt === 0 && field.varint !== undefined) {
        int64Data.push(asSignedInt64(field.varint));
      } else if (field.wt === 2 && field.bytes) {
        let p = 0;
        while (p < field.bytes.length) {
          const [v, next] = readVarint(field.bytes, p);
          int64Data.push(asSignedInt64(v));
          p = next;
        }
      }
    } else if (field.fn === 8 && field.bytes) {
      name = UTF8_DECODER.decode(field.bytes);
    } else if (field.fn === 9 && field.bytes) {
      rawData = field.bytes;
    } else if (field.fn === 13 && field.bytes) {
      let key = '';
      let value = '';
      for (const kv of decodeProtoFields(field.bytes)) {
        if (kv.fn === 1 && kv.bytes) key = UTF8_DECODER.decode(kv.bytes);
        if (kv.fn === 2 && kv.bytes) value = UTF8_DECODER.decode(kv.bytes);
      }
      if (key) {
        ext[key] = value;
      }
    }
  }

  let external: ParsedTensor['external'];
  if (ext.location) {
    const offset = ext.offset ? Number.parseInt(ext.offset, 10) : 0;
    const length = ext.length ? Number.parseInt(ext.length, 10) : -1;
    external = {
      location: ext.location,
      offset: Number.isFinite(offset) && offset >= 0 ? offset : 0,
      length: Number.isFinite(length) && length >= 0 ? length : -1,
    };
  }

  return {
    name,
    dims,
    dataType,
    rawData,
    floatData,
    int64Data,
    external,
  };
}

function parseAttrProto(buf: Uint8Array): ParsedAttr {
  let name = '';
  let i: bigint | undefined;
  let f: number | undefined;
  for (const field of decodeProtoFields(buf)) {
    if (field.fn === 1 && field.bytes) {
      name = UTF8_DECODER.decode(field.bytes);
    } else if (field.fn === 2 && field.bytes && field.bytes.byteLength === 4) {
      f = new DataView(field.bytes.buffer, field.bytes.byteOffset, 4).getFloat32(0, true);
    } else if (field.fn === 3 && field.varint !== undefined) {
      i = asSignedInt64(field.varint);
    }
  }
  return { name, i, f };
}

function parseNodeProto(buf: Uint8Array): ParsedNode {
  const inputs: string[] = [];
  const outputs: string[] = [];
  let opType = '';
  let domain = '';
  const attrs: Record<string, ParsedAttr> = {};

  for (const field of decodeProtoFields(buf)) {
    if (field.fn === 1 && field.bytes) {
      inputs.push(UTF8_DECODER.decode(field.bytes));
    } else if (field.fn === 2 && field.bytes) {
      outputs.push(UTF8_DECODER.decode(field.bytes));
    } else if (field.fn === 4 && field.bytes) {
      opType = UTF8_DECODER.decode(field.bytes);
    } else if (field.fn === 5 && field.bytes) {
      const attr = parseAttrProto(field.bytes);
      if (attr.name) {
        attrs[attr.name] = attr;
      }
    } else if (field.fn === 7 && field.bytes) {
      domain = UTF8_DECODER.decode(field.bytes);
    }
  }

  return { opType, domain, inputs, outputs, attrs };
}

function parseValueInfoName(buf: Uint8Array): string {
  for (const field of decodeProtoFields(buf)) {
    if (field.fn === 1 && field.bytes) {
      return UTF8_DECODER.decode(field.bytes);
    }
  }
  return '';
}

function extractInlineInt64Array(tensor: ParsedTensor): bigint[] | null {
  if (tensor.int64Data.length > 0) {
    return tensor.int64Data;
  }
  if (tensor.rawData && tensor.dataType === ONNX_DT.INT64) {
    const view = new DataView(
      tensor.rawData.buffer,
      tensor.rawData.byteOffset,
      tensor.rawData.byteLength
    );
    const out: bigint[] = [];
    for (let o = 0; o + 8 <= tensor.rawData.byteLength; o += 8) {
      out.push(view.getBigInt64(o, true));
    }
    return out;
  }
  if (tensor.rawData && tensor.dataType === ONNX_DT.INT32) {
    const view = new DataView(
      tensor.rawData.buffer,
      tensor.rawData.byteOffset,
      tensor.rawData.byteLength
    );
    const out: bigint[] = [];
    for (let o = 0; o + 4 <= tensor.rawData.byteLength; o += 4) {
      out.push(BigInt(view.getInt32(o, true)));
    }
    return out;
  }
  return null;
}

function extractInlineFloatScalar(tensor: ParsedTensor): number | null {
  const numElements = tensor.dims.reduce((a, b) => a * b, 1);
  if (numElements !== 1) {
    return null;
  }
  if (tensor.floatData.length === 1) {
    return tensor.floatData[0]!;
  }
  if (tensor.rawData) {
    const view = new DataView(
      tensor.rawData.buffer,
      tensor.rawData.byteOffset,
      tensor.rawData.byteLength
    );
    if (tensor.dataType === ONNX_DT.FLOAT && tensor.rawData.byteLength === 4) {
      return view.getFloat32(0, true);
    }
    if (tensor.dataType === ONNX_DT.FLOAT16 && tensor.rawData.byteLength === 2) {
      return decodeFp16Bits(view.getUint16(0, true));
    }
  }
  return null;
}

export interface TensorSliceRef {
  dataType: number;
  dims: number[];
  location: string | null;
  offset: number;
  byteLength: number;
  inlineData?: Uint8Array;
}

export type IndexGuardEvaluator = (tokenId: bigint) => bigint;

export interface GatherBlockQuantizedBranch {
  kind: 'gather-block-quantized';
  vocabSize: number;
  rowWidth: number;
  blockSize: number;
  isSigned: boolean;
  scaleDataType: number;
  mulScale: number;
  reshapeTarget?: number[];
  outputName: string;
  outputDataType: number;
  evaluateIndex: IndexGuardEvaluator;
  data: TensorSliceRef;
  scales: TensorSliceRef;
  zeroPoint?: TensorSliceRef;
}

export interface GatherFloatBranch {
  kind: 'gather';
  vocabSize: number;
  rowWidth: number;
  mulScale: number;
  reshapeTarget?: number[];
  outputName: string;
  outputDataType: number;
  evaluateIndex: IndexGuardEvaluator;
  table: TensorSliceRef;
}

export type EmbedOutputBranch = GatherBlockQuantizedBranch | GatherFloatBranch;

export interface EmbedTokensGraphPlan {
  inputName: string;
  outputNames: string[];
  branches: EmbedOutputBranch[];
  externalLocations: string[];
}

function toSliceRef(tensor: ParsedTensor): TensorSliceRef | null {
  if (tensor.external && tensor.external.location && tensor.external.length >= 0) {
    return {
      dataType: tensor.dataType,
      dims: tensor.dims,
      location: tensor.external.location,
      offset: tensor.external.offset,
      byteLength: tensor.external.length,
    };
  }
  if (tensor.rawData) {
    return {
      dataType: tensor.dataType,
      dims: tensor.dims,
      location: null,
      offset: 0,
      byteLength: tensor.rawData.byteLength,
      inlineData: tensor.rawData,
    };
  }
  return null;
}

/**
 * Parse an `.onnx` protobuf byte buffer and compile an execution plan if it matches the
 * supported pure embedding-lookup structure (`Gather` / `com.microsoft:GatherBlockQuantized` +
 * optional index sentinel guards / scalar `Mul` / `Reshape` / `Cast`).
 * Returns `null` for any unsupported op or graph pattern so callers safely fall back to ORT.
 */
export function parseOnnxEmbedTokensPlan(onnxBytes: Uint8Array): EmbedTokensGraphPlan | null {
  try {
    const modelFields = decodeProtoFields(onnxBytes);
    const graphField = modelFields.find((f) => f.fn === 7 && f.bytes);
    if (!graphField?.bytes) {
      return null;
    }

    const inputNames: string[] = [];
    const outputNames: string[] = [];
    const initializers = new Map<string, ParsedTensor>();
    const nodeByOutput = new Map<string, ParsedNode>();

    for (const gf of decodeProtoFields(graphField.bytes)) {
      if (gf.fn === 1 && gf.bytes) {
        const node = parseNodeProto(gf.bytes);
        for (const out of node.outputs) {
          if (out) {
            nodeByOutput.set(out, node);
          }
        }
      } else if (gf.fn === 5 && gf.bytes) {
        const tensor = parseTensorProto(gf.bytes);
        if (tensor.name) {
          initializers.set(tensor.name, tensor);
        }
      } else if (gf.fn === 11 && gf.bytes) {
        const name = parseValueInfoName(gf.bytes);
        if (name) inputNames.push(name);
      } else if (gf.fn === 12 && gf.bytes) {
        const name = parseValueInfoName(gf.bytes);
        if (name) outputNames.push(name);
      }
    }

    if (inputNames.length !== 1 || outputNames.length === 0) {
      return null;
    }
    const inputName = inputNames[0]!;

    const buildBoolExpr = (tensorName: string, depth = 0): ((id: bigint) => boolean) | null => {
      if (depth > 32) return null;
      const node = nodeByOutput.get(tensorName);
      if (!node) return null;

      if (node.opType === 'Not' && node.inputs.length === 1) {
        const inner = buildBoolExpr(node.inputs[0]!, depth + 1);
        return inner ? (id) => !inner(id) : null;
      }
      if (node.opType === 'And' && node.inputs.length === 2) {
        const a = buildBoolExpr(node.inputs[0]!, depth + 1);
        const b = buildBoolExpr(node.inputs[1]!, depth + 1);
        return a && b ? (id) => a(id) && b(id) : null;
      }
      if (node.opType === 'Or' && node.inputs.length === 2) {
        const a = buildBoolExpr(node.inputs[0]!, depth + 1);
        const b = buildBoolExpr(node.inputs[1]!, depth + 1);
        return a && b ? (id) => a(id) || b(id) : null;
      }
      if (
        (node.opType === 'Less' ||
          node.opType === 'LessOrEqual' ||
          node.opType === 'Equal' ||
          node.opType === 'Greater' ||
          node.opType === 'GreaterOrEqual') &&
        node.inputs.length === 2
      ) {
        const lhs = buildIndexExpr(node.inputs[0]!, depth + 1);
        const rhs = buildIndexExpr(node.inputs[1]!, depth + 1);
        if (!lhs || !rhs) return null;
        switch (node.opType) {
          case 'Less':
            return (id) => lhs(id) < rhs(id);
          case 'LessOrEqual':
            return (id) => lhs(id) <= rhs(id);
          case 'Equal':
            return (id) => lhs(id) === rhs(id);
          case 'Greater':
            return (id) => lhs(id) > rhs(id);
          case 'GreaterOrEqual':
            return (id) => lhs(id) >= rhs(id);
        }
      }
      return null;
    };

    const buildIndexExpr = (tensorName: string, depth = 0): IndexGuardEvaluator | null => {
      if (depth > 32) return null;
      if (tensorName === inputName) {
        return (id) => id;
      }
      const init = initializers.get(tensorName);
      if (init) {
        const arr = extractInlineInt64Array(init);
        if (arr && arr.length === 1) {
          const val = arr[0]!;
          return () => val;
        }
        return null;
      }
      const node = nodeByOutput.get(tensorName);
      if (!node) return null;
      if (node.opType === 'Where' && node.inputs.length === 3) {
        const cond = buildBoolExpr(node.inputs[0]!, depth + 1);
        const onTrue = buildIndexExpr(node.inputs[1]!, depth + 1);
        const onFalse = buildIndexExpr(node.inputs[2]!, depth + 1);
        if (!cond || !onTrue || !onFalse) return null;
        return (id) => (cond(id) ? onTrue(id) : onFalse(id));
      }
      return null;
    };

    const externalLocations = new Set<string>();
    const branches: EmbedOutputBranch[] = [];

    for (const outName of outputNames) {
      let curName = outName;
      let outputDataType: number = ONNX_DT.FLOAT;
      let reshapeTarget: number[] | undefined;
      let mulScale = 1;

      // Walk post-processing nodes backwards from output to Gather / GatherBlockQuantized
      let guard = 0;
      let baseNode: ParsedNode | null = null;
      while (guard < 16) {
        guard += 1;
        const node = nodeByOutput.get(curName);
        if (!node) {
          return null;
        }
        if (node.opType === 'Cast') {
          const toVal = Number(node.attrs.to?.i ?? ONNX_DT.FLOAT);
          if (toVal !== ONNX_DT.FLOAT) {
            return null;
          }
          outputDataType = toVal;
          curName = node.inputs[0] ?? '';
        } else if (node.opType === 'Reshape' && node.inputs.length >= 2) {
          const shapeInit = initializers.get(node.inputs[1]!);
          const shapeBigInts = shapeInit ? extractInlineInt64Array(shapeInit) : null;
          if (!shapeBigInts || shapeBigInts.length === 0) {
            return null;
          }
          reshapeTarget = shapeBigInts.map((v) => Number(v));
          curName = node.inputs[0] ?? '';
        } else if (node.opType === 'Mul' && node.inputs.length === 2) {
          const initA = initializers.get(node.inputs[0]!);
          const initB = initializers.get(node.inputs[1]!);
          const scalarA = initA ? extractInlineFloatScalar(initA) : null;
          const scalarB = initB ? extractInlineFloatScalar(initB) : null;
          if (scalarA !== null && Number.isFinite(scalarA) && !initB) {
            mulScale *= scalarA;
            curName = node.inputs[1]!;
          } else if (scalarB !== null && Number.isFinite(scalarB) && !initA) {
            mulScale *= scalarB;
            curName = node.inputs[0]!;
          } else {
            return null;
          }
        } else if (node.opType === 'GatherBlockQuantized' || node.opType === 'Gather') {
          baseNode = node;
          break;
        } else {
          return null;
        }
      }

      if (!baseNode) {
        return null;
      }

      if (baseNode.opType === 'GatherBlockQuantized') {
        if (baseNode.inputs.length < 3 || baseNode.inputs.length > 4) {
          return null;
        }
        const bits = Number(baseNode.attrs.bits?.i ?? 4);
        const blockSize = Number(baseNode.attrs.block_size?.i ?? 32);
        const gatherAxis = Number(baseNode.attrs.gather_axis?.i ?? 0);
        const rawQuantAxis = Number(baseNode.attrs.quantize_axis?.i ?? 1);

        const dataInit = initializers.get(baseNode.inputs[0]!);
        const scalesInit = initializers.get(baseNode.inputs[2]!);
        const zpInit = baseNode.inputs[3] ? initializers.get(baseNode.inputs[3]) : undefined;
        if (!dataInit || !scalesInit) {
          return null;
        }
        if (dataInit.dims.length !== 2 || scalesInit.dims.length !== 2) {
          return null;
        }
        const quantizeAxis = rawQuantAxis < 0 ? dataInit.dims.length + rawQuantAxis : rawQuantAxis;
        if (bits !== 4 || gatherAxis !== 0 || quantizeAxis !== 1 || blockSize <= 0) {
          return null;
        }
        if (
          dataInit.dataType !== ONNX_DT.UINT8 &&
          dataInit.dataType !== ONNX_DT.UINT4 &&
          dataInit.dataType !== ONNX_DT.INT4
        ) {
          return null;
        }
        if (scalesInit.dataType !== ONNX_DT.FLOAT16 && scalesInit.dataType !== ONNX_DT.FLOAT) {
          return null;
        }

        const vocabSize = dataInit.dims[0]!;
        const packedCols = dataInit.dims[1]!;
        const numBlocks = scalesInit.dims[1]!;
        if (scalesInit.dims[0] !== vocabSize || packedCols <= 0 || numBlocks <= 0) {
          return null;
        }

        const rowWidth = numBlocks * blockSize;
        if (packedCols * 2 < rowWidth) {
          return null;
        }

        const dataRef = toSliceRef(dataInit);
        const scalesRef = toSliceRef(scalesInit);
        const zpRef = zpInit ? toSliceRef(zpInit) : undefined;
        if (!dataRef || !scalesRef || (zpInit && !zpRef)) {
          return null;
        }
        if (dataRef.location) externalLocations.add(dataRef.location);
        if (scalesRef.location) externalLocations.add(scalesRef.location);
        if (zpRef?.location) externalLocations.add(zpRef.location);

        const evaluateIndex = buildIndexExpr(baseNode.inputs[1]!);
        if (!evaluateIndex) {
          return null;
        }

        branches.push({
          kind: 'gather-block-quantized',
          vocabSize,
          rowWidth,
          blockSize,
          isSigned: dataInit.dataType === ONNX_DT.INT4,
          scaleDataType: scalesInit.dataType,
          mulScale,
          reshapeTarget,
          outputName: outName,
          outputDataType,
          evaluateIndex,
          data: dataRef,
          scales: scalesRef,
          zeroPoint: zpRef ?? undefined,
        });
      } else {
        // Unquantized Gather(table, indices)
        const axis = Number(baseNode.attrs.axis?.i ?? 0);
        const tableInit = baseNode.inputs[0] ? initializers.get(baseNode.inputs[0]) : undefined;
        if (axis !== 0 || !tableInit || tableInit.dims.length !== 2) {
          return null;
        }
        if (tableInit.dataType !== ONNX_DT.FLOAT16 && tableInit.dataType !== ONNX_DT.FLOAT) {
          return null;
        }
        const tableRef = toSliceRef(tableInit);
        const evaluateIndex = baseNode.inputs[1] ? buildIndexExpr(baseNode.inputs[1]) : null;
        if (!tableRef || !evaluateIndex) {
          return null;
        }
        if (tableRef.location) externalLocations.add(tableRef.location);

        branches.push({
          kind: 'gather',
          vocabSize: tableInit.dims[0]!,
          rowWidth: tableInit.dims[1]!,
          mulScale,
          reshapeTarget,
          outputName: outName,
          outputDataType,
          evaluateIndex,
          table: tableRef,
        });
      }
    }

    return {
      inputName,
      outputNames,
      branches,
      externalLocations: [...externalLocations],
    };
  } catch {
    return null;
  }
}

export type ResolveExternalBlobFn = (location: string) => Promise<Blob | null>;

/**
 * Create a browser `Blob` resolver for external `.onnx_data*` locations using OPFS first, then
 * CacheStorage, and finally HTTP fetch if needed.
 */
export function createExternalBlobResolver(options: {
  urlByLocation: ReadonlyMap<string, string>;
  opfsRoot?: OpfsDirectoryLike | null;
  cacheName: string;
  cacheStorage?: Pick<CacheStorage, 'open'> | null;
  fetchFn?: typeof fetch;
}): ResolveExternalBlobFn {
  const pending = new Map<string, Promise<Blob | null>>();

  const resolveSingle = async (location: string): Promise<Blob | null> => {
    const clean = location.replace(/^\.\//, '');
    const base = clean.split('/').pop() ?? clean;
    const url =
      options.urlByLocation.get(location) ??
      options.urlByLocation.get(clean) ??
      options.urlByLocation.get(base);
    if (!url) {
      return null;
    }

    if (options.opfsRoot) {
      const opfsBlob = await openOpfsCachedBlob(options.opfsRoot, url);
      if (opfsBlob) {
        return opfsBlob;
      }
    }

    const storage =
      options.cacheStorage === undefined
        ? (globalThis as { caches?: Pick<CacheStorage, 'open'> }).caches
        : options.cacheStorage;
    if (storage && typeof storage.open === 'function') {
      try {
        const cache = await storage.open(options.cacheName);
        const hit = await cache.match(url);
        if (hit) {
          return await hit.blob();
        }
      } catch {
        // Ignore cache error and fall through to fetch if permitted
      }
    }

    const fetchImpl = options.fetchFn ?? globalThis.fetch;
    if (typeof fetchImpl === 'function') {
      try {
        const resp = await fetchImpl(url);
        if (resp.ok) {
          return await resp.blob();
        }
      } catch {
        return null;
      }
    }
    return null;
  };

  return (location: string): Promise<Blob | null> => {
    const existing = pending.get(location);
    if (existing) {
      return existing;
    }
    const task = resolveSingle(location).catch((err) => {
      pending.delete(location);
      throw err;
    });
    pending.set(location, task);
    return task;
  };
}

export interface OnDemandEmbedTokensSessionOptions {
  tensorCtor?: typeof OrtTensor;
  maxRowCacheBytes?: number;
}

export interface OnDemandEmbedTokensSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>;
  release(): Promise<void>;
}

function resolveReshapedDims(
  baseDims: readonly number[],
  reshapeTarget?: readonly number[]
): number[] {
  if (!reshapeTarget || reshapeTarget.length === 0) {
    return [...baseDims];
  }
  const totalElements = baseDims.reduce((a, b) => a * b, 1);
  const out = reshapeTarget.map((targetDim, idx) =>
    targetDim === 0 ? (baseDims[idx] ?? 0) : targetDim
  );
  const negIdx = out.indexOf(-1);
  if (negIdx >= 0) {
    let knownProd = 1;
    for (let i = 0; i < out.length; i += 1) {
      if (i !== negIdx) {
        knownProd *= out[i]!;
      }
    }
    out[negIdx] = knownProd > 0 ? Math.floor(totalElements / knownProd) : 0;
  }
  return out;
}

function readTensorIds(inputTensor: unknown): { dims: number[]; ids: BigInt64Array | Int32Array | number[] } {
  if (!inputTensor || typeof inputTensor !== 'object') {
    throw new Error('OnDemandEmbedTokensSession requires an input tensor.');
  }
  const tensorObj = inputTensor as {
    dims?: readonly number[];
    data?: BigInt64Array | Int32Array | Uint32Array | number[];
    cpuData?: BigInt64Array | Int32Array | Uint32Array | number[];
  };
  const dims = Array.isArray(tensorObj.dims) ? [...tensorObj.dims] : [];
  const data = tensorObj.data ?? tensorObj.cpuData;
  if (!data) {
    throw new Error('Input tensor to OnDemandEmbedTokensSession is missing CPU data.');
  }
  return { dims, ids: data as BigInt64Array | Int32Array | number[] };
}

/**
 * Create a lightweight duck-typed ORT session executing `embed_tokens` directly via
 * on-demand `Blob.slice` row reads without loading GB-scale embedding tables into RAM/VRAM.
 */
export async function createOnDemandEmbedTokensSession(
  plan: EmbedTokensGraphPlan,
  resolveBlob: ResolveExternalBlobFn,
  options: OnDemandEmbedTokensSessionOptions = {}
): Promise<OnDemandEmbedTokensSession> {
  const blobs = new Map<string, Blob>();
  for (const location of plan.externalLocations) {
    const blob = await resolveBlob(location);
    if (!blob) {
      throw new Error(`Missing external embedding weight blob "${location}".`);
    }
    blobs.set(location, blob);
  }

  const TensorCtor = options.tensorCtor ?? OrtTensor;
  const maxCacheBytes = Math.max(0, options.maxRowCacheBytes ?? MAX_EMBED_ROW_CACHE_BYTES);
  const rowCache = new Map<string, Float32Array>();
  let cachedBytes = 0;
  let released = false;

  const getCachedRow = (key: string): Float32Array | undefined => {
    const hit = rowCache.get(key);
    if (!hit) {
      return undefined;
    }
    rowCache.delete(key);
    rowCache.set(key, hit);
    return hit;
  };

  const putCachedRow = (key: string, row: Float32Array): void => {
    if (maxCacheBytes <= 0 || row.byteLength > maxCacheBytes) {
      return;
    }
    const existing = rowCache.get(key);
    if (existing) {
      cachedBytes -= existing.byteLength;
      rowCache.delete(key);
    }
    while (cachedBytes + row.byteLength > maxCacheBytes && rowCache.size > 0) {
      const oldestKey = rowCache.keys().next().value;
      if (oldestKey === undefined) break;
      const oldest = rowCache.get(oldestKey);
      if (oldest) {
        cachedBytes -= oldest.byteLength;
      }
      rowCache.delete(oldestKey);
    }
    rowCache.set(key, row);
    cachedBytes += row.byteLength;
  };

  const readTensorSlice = async (
    sliceRef: TensorSliceRef,
    relByteOffset: number,
    byteLength: number
  ): Promise<Uint8Array> => {
    if (byteLength === 0) {
      return new Uint8Array(0);
    }
    if (sliceRef.inlineData) {
      const start = sliceRef.offset + relByteOffset;
      return sliceRef.inlineData.subarray(start, start + byteLength);
    }
    if (!sliceRef.location) {
      throw new Error('TensorSliceRef has neither location nor inlineData.');
    }
    const blob = blobs.get(sliceRef.location);
    if (!blob) {
      throw new Error(`External embedding blob "${sliceRef.location}" is not available.`);
    }
    const start = sliceRef.offset + relByteOffset;
    const end = start + byteLength;
    if (end > blob.size) {
      throw new Error(
        `Out-of-bounds embedding slice [${start}, ${end}) on "${sliceRef.location}" (${blob.size} bytes).`
      );
    }
    const ab = await blob.slice(start, end).arrayBuffer();
    return new Uint8Array(ab);
  };

  const lut = getFp16Lut();

  const loadQuantizedRow = async (
    branch: GatherBlockQuantizedBranch,
    rowIdx: number
  ): Promise<Float32Array> => {
    const qRowBytes = branch.data.dims[1]!;
    const scaleElemBytes = branch.scaleDataType === ONNX_DT.FLOAT16 ? 2 : 4;
    const numBlocks = branch.scales.dims[1]!;
    const sRowBytes = numBlocks * scaleElemBytes;
    const zpRowBytes = branch.zeroPoint ? branch.zeroPoint.dims[1]! : 0;

    const [qBytes, sBytes, zpBytes] = await Promise.all([
      readTensorSlice(branch.data, rowIdx * qRowBytes, qRowBytes),
      readTensorSlice(branch.scales, rowIdx * sRowBytes, sRowBytes),
      branch.zeroPoint
        ? readTensorSlice(branch.zeroPoint, rowIdx * zpRowBytes, zpRowBytes)
        : Promise.resolve(null),
    ]);

    const out = new Float32Array(branch.rowWidth);
    const sView = new DataView(sBytes.buffer, sBytes.byteOffset, sBytes.byteLength);
    const scaleIsFp16 = branch.scaleDataType === ONNX_DT.FLOAT16;

    for (let block = 0; block < numBlocks; block += 1) {
      const rawScale = scaleIsFp16
        ? lut[sView.getUint16(block * 2, true)]!
        : sView.getFloat32(block * 4, true);

      let zp = 0;
      if (zpBytes) {
        const zb = zpBytes[block >> 1]!;
        const zNibble = (block & 1) === 0 ? zb & 0x0f : (zb >>> 4) & 0x0f;
        zp = branch.isSigned && zNibble >= 8 ? zNibble - 16 : zNibble;
      }

      const scaled = rawScale * branch.mulScale;
      const colStart = block * branch.blockSize;
      const colEnd = Math.min(branch.rowWidth, colStart + branch.blockSize);
      for (let col = colStart; col < colEnd; col += 1) {
        const qb = qBytes[col >> 1]!;
        const qNibble = (col & 1) === 0 ? qb & 0x0f : (qb >>> 4) & 0x0f;
        const qVal = branch.isSigned && qNibble >= 8 ? qNibble - 16 : qNibble;
        out[col] = (qVal - zp) * scaled;
      }
    }

    return out;
  };

  const loadFloatGatherRow = async (branch: GatherFloatBranch, rowIdx: number): Promise<Float32Array> => {
    const isFp16 = branch.table.dataType === ONNX_DT.FLOAT16;
    const elemBytes = isFp16 ? 2 : 4;
    const rowBytes = branch.rowWidth * elemBytes;
    const raw = await readTensorSlice(branch.table, rowIdx * rowBytes, rowBytes);
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const out = new Float32Array(branch.rowWidth);

    if (isFp16) {
      for (let col = 0; col < branch.rowWidth; col += 1) {
        out[col] = lut[view.getUint16(col * 2, true)]! * branch.mulScale;
      }
    } else {
      for (let col = 0; col < branch.rowWidth; col += 1) {
        out[col] = view.getFloat32(col * 4, true) * branch.mulScale;
      }
    }
    return out;
  };

  const fetchBranchRow = async (
    branchIdx: number,
    branch: EmbedOutputBranch,
    rowIdx: number
  ): Promise<Float32Array> => {
    const cacheKey = `${branchIdx}:${rowIdx}`;
    const cached = getCachedRow(cacheKey);
    if (cached) {
      return cached;
    }
    const row =
      branch.kind === 'gather-block-quantized'
        ? await loadQuantizedRow(branch, rowIdx)
        : await loadFloatGatherRow(branch, rowIdx);
    putCachedRow(cacheKey, row);
    return row;
  };

  return {
    inputNames: [plan.inputName],
    outputNames: [...plan.outputNames],

    async run(feeds: Record<string, unknown>): Promise<Record<string, unknown>> {
      if (released) {
        throw new Error('OnDemandEmbedTokensSession has been released.');
      }

      const rawInput = feeds[plan.inputName];
      const { dims, ids } = readTensorIds(rawInput);
      const tokenCount = ids.length;
      const outputs: Record<string, unknown> = {};

      await Promise.all(
        plan.branches.map(async (branch, branchIdx) => {
          const rowIndices = new Int32Array(tokenCount);
          const uniqueRows = new Set<number>();

          for (let i = 0; i < tokenCount; i += 1) {
            const rawId = ids[i]!;
            const guarded = branch.evaluateIndex(typeof rawId === 'bigint' ? rawId : BigInt(rawId));
            let rowIdx = Number(guarded);
            if (rowIdx < 0) {
              rowIdx += branch.vocabSize;
            }
            if (!Number.isInteger(rowIdx) || rowIdx < 0 || rowIdx >= branch.vocabSize) {
              throw new RangeError(
                `Embedding index ${guarded.toString()} out of bounds for vocabSize=${branch.vocabSize}`
              );
            }
            rowIndices[i] = rowIdx;
            uniqueRows.add(rowIdx);
          }

          const uniqueList = [...uniqueRows];
          const fetchedRows = await Promise.all(
            uniqueList.map((rowIdx) => fetchBranchRow(branchIdx, branch, rowIdx))
          );
          const rowMap = new Map<number, Float32Array>();
          for (let j = 0; j < uniqueList.length; j += 1) {
            rowMap.set(uniqueList[j]!, fetchedRows[j]!);
          }

          const flatOut = new Float32Array(tokenCount * branch.rowWidth);
          for (let i = 0; i < tokenCount; i += 1) {
            const srcRow = rowMap.get(rowIndices[i]!)!;
            flatOut.set(srcRow, i * branch.rowWidth);
          }

          const finalDims = resolveReshapedDims([...dims, branch.rowWidth], branch.reshapeTarget);
          outputs[branch.outputName] = new TensorCtor('float32', flatOut, finalDims);
        })
      );

      return outputs;
    },

    async release(): Promise<void> {
      released = true;
      rowCache.clear();
      cachedBytes = 0;
      blobs.clear();
    },
  };
}
