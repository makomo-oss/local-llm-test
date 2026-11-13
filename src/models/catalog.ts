import type {
  DeviceOption,
  ModelFamily,
  ModelId,
  ModelSpec,
  ParsedThinkingOutput,
  QuantDtype,
  ResolvedDtype,
  TransformersDtype,
} from '../types.ts';

/**
 * Verified catalog of unauthenticated public ONNX models from Hugging Face
 * (`onnx-community`) for Transformers.js v4 (`AutoTokenizer` + `AutoModelForCausalLM`).
 *
 * Byte sizes reflect text-only inference (`embed_tokens` + `decoder_model_merged` / `model`)
 * plus separate `tokenizer.json` sizes verified from Hugging Face repo metadata.
 */
export const MODEL_CATALOG: readonly ModelSpec[] = [
  {
    id: 'gemma-4-e4b',
    name: 'Gemma 4 E4B',
    family: 'gemma4',
    repoId: 'onnx-community/gemma-4-E4B-it-ONNX',
    parameterLabel: 'E4B',
    architecture: 'Gemma4ForConditionalGeneration',
    causalLmClass: 'Gemma4ForCausalLM',
    contextLength: 131_072,
    license: 'Apache-2.0',
    supportsThinking: true,
    defaultThinkingEnabled: false,
    requiresExplicitExternalDataForQ8: false,
    availableDtypes: ['q4f16', 'q4', 'q8', 'fp16'],
    tokenizerBytes: 19_439_251,
    textOnlyBytesByDtype: {
      q4f16: 4_905_482_405,
      q4: 5_616_292_603,
      q8: 9_810_159_379,
      fp16: 16_470_207_217,
    },
    fullMultimodalBytesByDtype: {
      q4f16: 5_178_861_705,
      q4: 5_923_096_506,
      q8: 10_355_845_813,
    },
    kvCacheFp16OnQ4f16: true,
    descriptionJa:
      'Google製 Gemma 4 の高性能モデル（テキスト推論時）。128Kコンテキストと思考モードに対応し、高品質な日本語・コード生成が可能です。',
    tags: ['Google', '128K Context', 'Thinking', 'High Quality'],
  },
  {
    id: 'gemma-4-e2b',
    name: 'Gemma 4 E2B',
    family: 'gemma4',
    repoId: 'onnx-community/gemma-4-E2B-it-ONNX',
    parameterLabel: 'E2B',
    architecture: 'Gemma4ForConditionalGeneration',
    causalLmClass: 'Gemma4ForCausalLM',
    contextLength: 131_072,
    license: 'Apache-2.0',
    supportsThinking: true,
    defaultThinkingEnabled: false,
    requiresExplicitExternalDataForQ8: false,
    availableDtypes: ['q4f16', 'q4', 'q8', 'fp16'],
    tokenizerBytes: 19_439_251,
    textOnlyBytesByDtype: {
      q4f16: 3_111_069_636,
      q4: 3_627_411_909,
      q8: 6_221_495_832,
      fp16: 10_264_308_077,
    },
    fullMultimodalBytesByDtype: {
      q4f16: 3_381_966_758,
      q4: 3_930_062_458,
      q8: 6_762_217_909,
    },
    kvCacheFp16OnQ4f16: true,
    descriptionJa:
      'Google製 Gemma 4 の中量級モデル（テキスト推論時）。E4Bより省メモリで動作しつつ、思考モードと128Kコンテキストに対応します。',
    tags: ['Google', '128K Context', 'Thinking', 'Balanced'],
  },
  {
    id: 'qwen-3.5-4b',
    name: 'Qwen 3.5 4B',
    family: 'qwen3_5',
    repoId: 'onnx-community/Qwen3.5-4B-ONNX',
    parameterLabel: '4B',
    architecture: 'Qwen3_5ForConditionalGeneration',
    causalLmClass: 'Qwen3_5ForCausalLM',
    contextLength: 262_144,
    license: 'Apache-2.0',
    supportsThinking: true,
    defaultThinkingEnabled: true,
    requiresExplicitExternalDataForQ8: false,
    availableDtypes: ['q4f16', 'q4', 'q8', 'fp16'],
    tokenizerBytes: 19_226_111,
    textOnlyBytesByDtype: {
      q4f16: 2_803_657_555,
      q4: 3_109_119_274,
      q8: 5_605_003_321,
      fp16: 9_686_105_308,
    },
    fullMultimodalBytesByDtype: {
      q4f16: 3_002_210_633,
      q4: 3_328_755_825,
      q8: 6_000_091_758,
    },
    kvCacheFp16OnQ4f16: true,
    descriptionJa:
      'Alibaba製 Qwen 3.5 の4Bモデル（テキスト推論時）。256Kの長大コンテキストと <think> 思考トレースに対応し、推論品質と速度のバランスに優れます。',
    tags: ['Qwen', '256K Context', 'Thinking', 'Balanced'],
  },
  {
    id: 'qwen-3.5-2b',
    name: 'Qwen 3.5 2B',
    family: 'qwen3_5',
    repoId: 'onnx-community/Qwen3.5-2B-ONNX',
    parameterLabel: '2B',
    architecture: 'Qwen3_5ForConditionalGeneration',
    causalLmClass: 'Qwen3_5ForCausalLM',
    contextLength: 262_144,
    license: 'Apache-2.0',
    supportsThinking: true,
    defaultThinkingEnabled: false,
    requiresExplicitExternalDataForQ8: false,
    availableDtypes: ['q4f16', 'q4', 'q8', 'fp16'],
    tokenizerBytes: 19_226_111,
    textOnlyBytesByDtype: {
      q4f16: 1_384_836_046,
      q4: 1_535_809_591,
      q8: 2_768_076_697,
      fp16: 4_782_795_205,
    },
    fullMultimodalBytesByDtype: {
      q4f16: 1_582_175_684,
      q4: 1_754_100_605,
      q8: 3_160_738_252,
    },
    kvCacheFp16OnQ4f16: true,
    descriptionJa:
      'Alibaba製 Qwen 3.5 の軽量2Bモデル（テキスト推論時）。q4f16で約1.29 GiBと軽く、WebGPU・CPUベンチマークの基準として扱いやすいモデルです。',
    tags: ['Qwen', '256K Context', 'Thinking', 'Fast'],
  },
  {
    id: 'qwen-3.5-0.8b',
    name: 'Qwen 3.5 0.8B',
    family: 'qwen3_5',
    repoId: 'onnx-community/Qwen3.5-0.8B-ONNX',
    parameterLabel: '0.8B',
    architecture: 'Qwen3_5ForConditionalGeneration',
    causalLmClass: 'Qwen3_5ForCausalLM',
    contextLength: 262_144,
    license: 'Apache-2.0',
    supportsThinking: true,
    defaultThinkingEnabled: false,
    requiresExplicitExternalDataForQ8: false,
    availableDtypes: ['q4f16', 'q4', 'q8', 'fp16'],
    tokenizerBytes: 19_226_111,
    textOnlyBytesByDtype: {
      q4f16: 584_705_674,
      q4: 649_200_288,
      q8: 1_168_013_485,
      fp16: 2_015_271_040,
    },
    fullMultimodalBytesByDtype: {
      q4f16: 646_838_112,
      q4: 717_652_150,
      q8: 1_291_007_603,
    },
    kvCacheFp16OnQ4f16: true,
    descriptionJa:
      'Alibaba製 Qwen 3.5 の超軽量0.8Bモデル（テキスト推論時）。約558 MiB（584.7 MB）で初回ダウンロードが最も速く、CPU (WASM) や省電力GPUでの動作確認に最適です。',
    tags: ['Qwen', '256K Context', 'Thinking', 'Ultra-Light'],
  },
  {
    id: 'llama-3.2-3b',
    name: 'Llama 3.2 3B Instruct',
    family: 'llama',
    repoId: 'onnx-community/Llama-3.2-3B-Instruct-ONNX',
    parameterLabel: '3B',
    architecture: 'LlamaForCausalLM',
    causalLmClass: 'LlamaForCausalLM',
    contextLength: 131_072,
    license: 'llama3.2',
    supportsThinking: false,
    defaultThinkingEnabled: false,
    requiresExplicitExternalDataForQ8: true,
    availableDtypes: ['q4f16', 'q4', 'q8', 'fp16'],
    tokenizerBytes: 11_574_638,
    textOnlyBytesByDtype: {
      q4f16: 2_407_617_315,
      q4: 3_405_686_568,
      q8: 3_214_399_024,
      fp16: 6_459_242_427,
    },
    kvCacheFp16OnQ4f16: true,
    descriptionJa:
      'Meta製 Llama 3.2 の3B対話向けモデル（非思考型・128K対応）。WebGPU / CPU 間の純粋なデコード性能比較に適しています。',
    tags: ['Meta', '128K Context', 'Non-Thinking', 'Standard'],
  },
] as const;

const MODEL_BY_ID = new Map<string, ModelSpec>(MODEL_CATALOG.map((model) => [model.id, model]));

/**
 * Look up a model specification by its catalog ID.
 */
export function getModelById(id: ModelId): ModelSpec;
export function getModelById(id: string): ModelSpec | undefined;
export function getModelById(id: string): ModelSpec | undefined {
  return MODEL_BY_ID.get(id);
}

/**
 * Compute expected download size in bytes for text-only execution (`AutoModelForCausalLM`).
 */
export function getModelDownloadBytes(
  model: ModelSpec,
  dtype: QuantDtype,
  includeTokenizer = true
): number {
  const weightsBytes = model.textOnlyBytesByDtype[dtype];
  return weightsBytes + (includeTokenizer ? model.tokenizerBytes : 0);
}

/**
 * Compute expected download size in bytes for text-only execution (`AutoModelForCausalLM`),
 * factoring in any per-session override tracked on `ResolvedDtype`.
 */
export function getResolvedDownloadBytes(
  model: ModelSpec,
  resolvedOrDtype: ResolvedDtype | QuantDtype,
  includeTokenizer = true
): number {
  const weightsBytes =
    typeof resolvedOrDtype === 'string'
      ? model.textOnlyBytesByDtype[resolvedOrDtype]
      : (resolvedOrDtype.effectiveWeightsBytes ??
        model.textOnlyBytesByDtype[resolvedOrDtype.dtype]);
  return weightsBytes + (includeTokenizer ? model.tokenizerBytes : 0);
}

interface WasmSplitModelWeightSpec {
  embedDtype: TransformersDtype;
  embedBytes: number;
  decoderBytesByDtype: {
    q4: number;
    q8: number;
  };
}

/**
 * Per-session ONNX overrides for split causal-LM models on CPU (WASM).
 *
 * Qwen 3.5 and Gemma 4 emit `GatherBlockQuantized(1)` inside `embed_tokens_q4.onnx`
 * and `embed_tokens_quantized.onnx`, which ONNX Runtime WASM CPU does not implement
 * (`ERROR_CODE: 9`). `decoder_model_merged` uses `MatMulNBits` + `Gather` cleanly.
 * - Qwen 3.5 uses `embed_tokens_fp16.onnx` (`Gather` + `Cast` -> float32), halving RAM/disk vs `fp32`.
 * - Gemma 4 uses `embed_tokens.onnx` (`fp32`) to avoid FP16 `Mul` ops inside `embed_tokens_fp16.onnx`.
 */
const WASM_SPLIT_MODEL_WEIGHTS: Partial<Record<ModelId, WasmSplitModelWeightSpec>> = {
  'qwen-3.5-0.8b': {
    embedDtype: 'fp16',
    embedBytes: 508_559_997,
    decoderBytesByDtype: {
      q4: 486_301_511,
      q8: 874_001_726,
    },
  },
  'qwen-3.5-2b': {
    embedDtype: 'fp16',
    embedBytes: 1_017_119_358,
    decoderBytesByDtype: {
      q4: 1_210_012_894,
      q8: 2_180_054_057,
    },
  },
  'qwen-3.5-4b': {
    embedDtype: 'fp16',
    embedBytes: 1_271_399_038,
    decoderBytesByDtype: {
      q4: 2_701_873_617,
      q8: 4_869_975_241,
    },
  },
  'gemma-4-e2b': {
    embedDtype: 'fp32',
    embedBytes: 11_005_857_939,
    decoderBytesByDtype: {
      q4: 1_864_750_511,
      q8: 3_040_111_074,
    },
  },
  'gemma-4-e4b': {
    embedDtype: 'fp32',
    embedBytes: 13_958_647_953,
    decoderBytesByDtype: {
      q4: 3_380_723_437,
      q8: 5_775_233_760,
    },
  },
};

function finalizeWasmDtype(
  model: ModelSpec,
  decoderDtype: 'q4' | 'q8',
  requestedDtype: QuantDtype | 'auto' | undefined,
  fallbackReason?: string,
  useExternalDataFormat?: true
): ResolvedDtype {
  const splitSpec = WASM_SPLIT_MODEL_WEIGHTS[model.id];
  if (!splitSpec) {
    return {
      dtype: decoderDtype,
      requestedDtype,
      ...(useExternalDataFormat ? { useExternalDataFormat } : {}),
      adjusted: Boolean(fallbackReason),
      ...(fallbackReason ? { reason: fallbackReason } : {}),
    };
  }

  const effectiveWeightsBytes =
    splitSpec.embedBytes + splitSpec.decoderBytesByDtype[decoderDtype];
  const splitReason = `WASM (CPU) では embed_tokens の GatherBlockQuantized 非対応を回避するため embed_tokens=${splitSpec.embedDtype}・decoder_model_merged=${decoderDtype} を使用します。`;

  return {
    dtype: decoderDtype,
    modelDtype: {
      embed_tokens: splitSpec.embedDtype,
      decoder_model_merged: decoderDtype,
    },
    effectiveWeightsBytes,
    requestedDtype,
    adjusted: true,
    reason: fallbackReason ? `${fallbackReason} ${splitReason}` : splitReason,
  };
}

export interface ResolveDtypeOptions {
  /**
   * Explicitly allow `use_external_data_format: true` when `q8` is requested on
   * models whose `config.json` omits `model_quantized.onnx` (Llama 3.2 3B).
   */
  useExternalDataFormat?: boolean;
}

/**
 * Resolve the safest effective ONNX quantization dtype for a model + compute device.
 *
 * Key runtime rules enforced here:
 * 1. `q4f16` / `fp16` require WebGPU `shader-f16`; otherwise fall back to `q4`.
 * 2. CPU/WASM defaults to explicit `'q4'` (instead of Transformers.js's implicit `'q8'`)
 *    to avoid Llama's external-data omission trap and cut WASM memory roughly in half.
 * 3. Llama 3.2 3B `q8` omits `model_quantized.onnx` from `config.json`'s
 *    `use_external_data_format` map; it either sets `useExternalDataFormat: true`
 *    when opted-in or falls back safely (`q4f16` / `q4`).
 * 4. On CPU/WASM, split-session models whose quantized `embed_tokens` contains
 *    `GatherBlockQuantized` (Qwen 3.5, Gemma 4) use a per-session `modelDtype` map
 *    (`embed_tokens: 'fp16' | 'fp32'`, `decoder_model_merged: 'q4' | 'q8'`).
 */
export function resolveEffectiveDtype(
  model: ModelSpec,
  device: Pick<DeviceOption, 'backend' | 'supportsFp16' | 'features'>,
  requestedDtype?: QuantDtype | 'auto',
  options?: ResolveDtypeOptions
): ResolvedDtype {
  const hasShaderF16 =
    device.backend === 'webgpu' &&
    (device.supportsFp16 || device.features.includes('shader-f16'));
  const webgpuDefault: QuantDtype = hasShaderF16 ? 'q4f16' : 'q4';

  if (!requestedDtype || requestedDtype === 'auto') {
    if (device.backend === 'wasm') {
      return finalizeWasmDtype(model, 'q4', requestedDtype);
    }
    return {
      dtype: webgpuDefault,
      requestedDtype,
      adjusted: false,
    };
  }

  const candidate: QuantDtype = requestedDtype;
  if (!model.availableDtypes.includes(candidate)) {
    const fallback = device.backend === 'wasm' ? 'q4' : webgpuDefault;
    const unsupportedReason = `${model.name} は dtype "${requestedDtype}" に未対応のため ${fallback} にフォールバックしました。`;
    if (device.backend === 'wasm') {
      return finalizeWasmDtype(model, 'q4', requestedDtype, unsupportedReason);
    }
    return {
      dtype: fallback,
      requestedDtype,
      adjusted: true,
      reason: unsupportedReason,
    };
  }

  if (device.backend === 'wasm') {
    if (candidate === 'q4f16' || candidate === 'fp16') {
      return finalizeWasmDtype(
        model,
        'q4',
        requestedDtype,
        `WASM (CPU) バックエンドでは ${candidate} を使用できないため q4 にフォールバックしました。`
      );
    }

    if (candidate === 'q8' && model.requiresExplicitExternalDataForQ8) {
      if (options?.useExternalDataFormat) {
        return finalizeWasmDtype(model, 'q8', requestedDtype, undefined, true);
      }
      return finalizeWasmDtype(
        model,
        'q4',
        requestedDtype,
        `${model.name} の q8 は use_external_data_format: true 指定が必要なため安全な q4 にフォールバックしました。`
      );
    }

    return finalizeWasmDtype(model, candidate, requestedDtype);
  }

  // WebGPU backend
  if ((candidate === 'q4f16' || candidate === 'fp16') && !hasShaderF16) {
    return {
      dtype: 'q4',
      requestedDtype,
      adjusted: true,
      reason: `選択された WebGPU デバイスは shader-f16 に非対応のため ${candidate} から q4 にフォールバックしました。`,
    };
  }

  if (candidate === 'q8' && model.requiresExplicitExternalDataForQ8) {
    if (options?.useExternalDataFormat) {
      return {
        dtype: 'q8',
        requestedDtype,
        useExternalDataFormat: true,
        adjusted: false,
      };
    }
    return {
      dtype: webgpuDefault,
      requestedDtype,
      adjusted: true,
      reason: `${model.name} の q8 は use_external_data_format: true 指定が必要なため ${webgpuDefault} にフォールバックしました。`,
    };
  }

  return {
    dtype: candidate,
    requestedDtype,
    adjusted: false,
  };
}

export interface FormatBytesOptions {
  /** Use IEC binary units (1024 / KiB, MiB, GiB) by default; set false for SI (1000 / KB, MB, GB). */
  binary?: boolean;
  /** Override fixed decimal precision. */
  decimals?: number;
}

/**
 * Format a byte count into a human-readable string (e.g. `4.57 GiB`, `557.6 MiB`).
 */
export function formatBytes(bytes: number, options?: FormatBytesOptions): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 B';
  }

  const binary = options?.binary ?? true;
  const base = binary ? 1024 : 1000;
  const units = binary
    ? (['B', 'KiB', 'MiB', 'GiB', 'TiB'] as const)
    : (['B', 'KB', 'MB', 'GB', 'TB'] as const);

  let unitIndex = 0;
  let value = bytes;
  while (value >= base && unitIndex < units.length - 1) {
    value /= base;
    unitIndex += 1;
  }

  const defaultDecimals = unitIndex === 0 ? 0 : unitIndex >= 3 ? 2 : 1;
  const decimals = options?.decimals ?? defaultDecimals;
  const formatted = unitIndex === 0 ? Math.round(value).toString() : value.toFixed(decimals);
  return `${formatted} ${units[unitIndex] ?? 'B'}`;
}

export interface ParseThinkingOptions {
  /**
   * Set true during streaming for Qwen 3.5 when `enable_thinking: true` placed
   * `<think>\n` inside the prompt and `TextStreamer({ skip_prompt: true })`
   * stripped the opening `<think>` tag from generated text.
   */
  thinkingExpectedInPrompt?: boolean;
}

const SPECIAL_STOP_TOKEN_REGEX =
  /(?:<turn\|>|<\|im_end\|>|<\|eot_id\|>|<eos>|<\|endoftext\|>)+$/g;
const LEADING_GEMMA_TURN_REGEX = /^<\|turn>model\r?\n?/;
const LEADING_QWEN_ASSISTANT_REGEX = /^<\|im_start\|>assistant\r?\n?/;

function cleanModelControlArtifacts(rawText: string): string {
  return rawText
    .replace(LEADING_GEMMA_TURN_REGEX, '')
    .replace(LEADING_QWEN_ASSISTANT_REGEX, '')
    .replace(SPECIAL_STOP_TOKEN_REGEX, '');
}

/**
 * Parse reasoning traces and final assistant response text across model families,
 * supporting both completed responses and partially streamed tokens.
 */
export function parseThinkingOutput(
  rawText: string,
  family: ModelFamily,
  options?: ParseThinkingOptions
): ParsedThinkingOutput {
  const cleaned = cleanModelControlArtifacts(rawText);

  if (family === 'llama') {
    const content = cleaned.trim();
    return {
      thinking: '',
      content,
      hasThinking: false,
      isThinkingInProgress: false,
    };
  }

  if (family === 'qwen3_5') {
    return parseQwenThinking(cleaned, options?.thinkingExpectedInPrompt === true);
  }

  return parseGemma4Thinking(cleaned);
}

function parseQwenThinking(text: string, thinkingExpectedInPrompt: boolean): ParsedThinkingOutput {
  const OPEN_TAG = '<think>';
  const CLOSE_TAG = '</think>';

  const hasOpen = text.includes(OPEN_TAG);
  const hasClose = text.includes(CLOSE_TAG);

  if (!hasOpen && !hasClose) {
    const trimmed = text.trim();
    if (thinkingExpectedInPrompt) {
      return {
        thinking: trimmed,
        content: '',
        hasThinking: trimmed.length > 0,
        isThinkingInProgress: true,
      };
    }
    return {
      thinking: '',
      content: trimmed,
      hasThinking: false,
      isThinkingInProgress: false,
    };
  }

  // Handle prompt-opened `<think>` stripped by `skip_prompt: true`, where only `</think>` appears.
  if (!hasOpen && hasClose) {
    const closeIdx = text.indexOf(CLOSE_TAG);
    const thinking = text.slice(0, closeIdx).trim();
    const rest = text.slice(closeIdx + CLOSE_TAG.length);
    const parsedRest = rest.includes(OPEN_TAG) ? parseQwenThinking(rest, false) : null;
    const combinedThinking = [thinking, parsedRest?.thinking ?? '']
      .filter((part) => part.length > 0)
      .join('\n\n');
    const content = (parsedRest ? parsedRest.content : rest).trim();
    return {
      thinking: combinedThinking,
      content,
      hasThinking: combinedThinking.length > 0,
      isThinkingInProgress: parsedRest?.isThinkingInProgress ?? false,
    };
  }

  const thinkingParts: string[] = [];
  const contentParts: string[] = [];
  let cursor = 0;
  let isThinkingInProgress = false;

  while (cursor < text.length) {
    const openIdx = text.indexOf(OPEN_TAG, cursor);
    if (openIdx === -1) {
      contentParts.push(text.slice(cursor));
      break;
    }

    if (openIdx > cursor) {
      contentParts.push(text.slice(cursor, openIdx));
    }

    const innerStart = openIdx + OPEN_TAG.length;
    const closeIdx = text.indexOf(CLOSE_TAG, innerStart);
    if (closeIdx === -1) {
      const partialThinking = text.slice(innerStart).trim();
      if (partialThinking.length > 0) {
        thinkingParts.push(partialThinking);
      }
      isThinkingInProgress = true;
      break;
    }

    const closedThinking = text.slice(innerStart, closeIdx).trim();
    if (closedThinking.length > 0) {
      thinkingParts.push(closedThinking);
    }
    cursor = closeIdx + CLOSE_TAG.length;
  }

  const thinking = thinkingParts.join('\n\n').trim();
  const content = contentParts.join('').trim();

  return {
    thinking,
    content,
    hasThinking: thinking.length > 0 || isThinkingInProgress,
    isThinkingInProgress,
  };
}

function parseGemma4Thinking(text: string): ParsedThinkingOutput {
  const CHANNEL_OPEN = '<|channel>';
  const CHANNEL_CLOSE = '<channel|>';

  if (text.includes(CHANNEL_OPEN)) {
    const thinkingParts: string[] = [];
    const contentParts: string[] = [];
    let cursor = 0;
    let isThinkingInProgress = false;

    while (cursor < text.length) {
      const openIdx = text.indexOf(CHANNEL_OPEN, cursor);
      if (openIdx === -1) {
        contentParts.push(text.slice(cursor));
        break;
      }

      if (openIdx > cursor) {
        contentParts.push(text.slice(cursor, openIdx));
      }

      const afterOpen = openIdx + CHANNEL_OPEN.length;
      const closeIdx = text.indexOf(CHANNEL_CLOSE, afterOpen);
      const rawChannelBody =
        closeIdx === -1 ? text.slice(afterOpen) : text.slice(afterOpen, closeIdx);

      // Gemma 4 thought channel starts with `thought\n` (or partial `thought` during token streaming).
      const thoughtPrefixMatch = rawChannelBody.match(/^thought(?:\r?\n|$)/);
      if (thoughtPrefixMatch) {
        const thoughtText = rawChannelBody.slice(thoughtPrefixMatch[0].length).trim();
        if (thoughtText.length > 0) {
          thinkingParts.push(thoughtText);
        }
        if (closeIdx === -1) {
          isThinkingInProgress = true;
          break;
        }
      } else if (closeIdx === -1 && 'thought'.startsWith(rawChannelBody.trim())) {
        // Partial header `<|channel>thou...` while streaming.
        isThinkingInProgress = true;
        break;
      } else {
        // Non-thought channel block: keep body in content if closed.
        const body = rawChannelBody.trim();
        if (body.length > 0) {
          contentParts.push(body);
        }
        if (closeIdx === -1) {
          break;
        }
      }

      cursor = closeIdx + CHANNEL_CLOSE.length;
    }

    const thinking = thinkingParts.join('\n\n').trim();
    const content = contentParts.join('').trim();

    return {
      thinking,
      content,
      hasThinking: thinking.length > 0 || isThinkingInProgress,
      isThinkingInProgress,
    };
  }

  // Fallback when `skip_special_tokens: true` stripped `<|channel>` (and possibly `<channel|>`),
  // leaving a raw `thought\n` prefix.
  const strippedPrefixMatch = text.match(/^thought\r?\n/);
  if (strippedPrefixMatch) {
    const rest = text.slice(strippedPrefixMatch[0].length);
    const closeIdx = rest.indexOf(CHANNEL_CLOSE);
    if (closeIdx !== -1) {
      const thinking = rest.slice(0, closeIdx).trim();
      const content = rest.slice(closeIdx + CHANNEL_CLOSE.length).trim();
      return {
        thinking,
        content,
        hasThinking: thinking.length > 0,
        isThinkingInProgress: false,
      };
    }

    const thinking = rest.trim();
    return {
      thinking,
      content: '',
      hasThinking: true,
      isThinkingInProgress: true,
    };
  }

  const content = text.trim();
  return {
    thinking: '',
    content,
    hasThinking: false,
    isThinkingInProgress: false,
  };
}
