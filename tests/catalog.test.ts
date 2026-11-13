import { describe, expect, it } from 'vitest';
import {
  MODEL_CATALOG,
  formatBytes,
  getModelById,
  getModelDownloadBytes,
  parseThinkingOutput,
  resolveEffectiveDtype,
} from '../src/models/catalog.ts';
import type { DeviceOption, ModelId } from '../src/types.ts';

function makeWebGpuDevice(supportsFp16: boolean): DeviceOption {
  return {
    id: supportsFp16 ? 'webgpu-high-performance' : 'webgpu-low-power',
    label: 'Mock WebGPU Device',
    backend: 'webgpu',
    gpuPreference: supportsFp16 ? 'high-performance' : 'low-power',
    features: supportsFp16 ? ['shader-f16', 'subgroups'] : ['subgroups'],
    limits: { maxStorageBufferBindingSize: 1024 * 1024 * 1024 },
    supportsFp16,
    recommendedDtype: supportsFp16 ? 'q4f16' : 'q4',
    available: true,
    runtimeDetails: {
      crossOriginIsolated: false,
      hardwareConcurrency: 8,
      simdSupported: true,
      numThreads: 1,
    },
  };
}

function makeWasmDevice(crossOriginIsolated = false): DeviceOption {
  return {
    id: 'wasm-cpu',
    label: 'CPU (WASM)',
    backend: 'wasm',
    features: ['wasm-simd'],
    limits: {},
    supportsFp16: false,
    recommendedDtype: 'q4',
    available: true,
    runtimeDetails: {
      crossOriginIsolated,
      hardwareConcurrency: 8,
      simdSupported: true,
      numThreads: crossOriginIsolated ? 4 : 1,
    },
  };
}

describe('MODEL_CATALOG', () => {
  it('contains all 6 verified public Hugging Face ONNX models with exact sizes', () => {
    expect(MODEL_CATALOG).toHaveLength(6);

    const expected: Record<
      ModelId,
      {
        repoId: string;
        q4f16: number;
        q4: number;
        q8: number;
        fp16: number;
        tokenizerBytes: number;
        contextLength: number;
        license: string;
        supportsThinking: boolean;
      }
    > = {
      'gemma-4-e4b': {
        repoId: 'onnx-community/gemma-4-E4B-it-ONNX',
        q4f16: 4_905_482_405,
        q4: 5_616_292_603,
        q8: 9_810_159_379,
        fp16: 16_470_207_217,
        tokenizerBytes: 19_439_251,
        contextLength: 131_072,
        license: 'Apache-2.0',
        supportsThinking: true,
      },
      'gemma-4-e2b': {
        repoId: 'onnx-community/gemma-4-E2B-it-ONNX',
        q4f16: 3_111_069_636,
        q4: 3_627_411_909,
        q8: 6_221_495_832,
        fp16: 10_264_308_077,
        tokenizerBytes: 19_439_251,
        contextLength: 131_072,
        license: 'Apache-2.0',
        supportsThinking: true,
      },
      'qwen-3.5-4b': {
        repoId: 'onnx-community/Qwen3.5-4B-ONNX',
        q4f16: 2_803_657_555,
        q4: 3_109_119_274,
        q8: 5_605_003_321,
        fp16: 9_686_105_308,
        tokenizerBytes: 19_226_111,
        contextLength: 262_144,
        license: 'Apache-2.0',
        supportsThinking: true,
      },
      'qwen-3.5-2b': {
        repoId: 'onnx-community/Qwen3.5-2B-ONNX',
        q4f16: 1_384_836_046,
        q4: 1_535_809_591,
        q8: 2_768_076_697,
        fp16: 4_782_795_205,
        tokenizerBytes: 19_226_111,
        contextLength: 262_144,
        license: 'Apache-2.0',
        supportsThinking: true,
      },
      'qwen-3.5-0.8b': {
        repoId: 'onnx-community/Qwen3.5-0.8B-ONNX',
        q4f16: 584_705_674,
        q4: 649_200_288,
        q8: 1_168_013_485,
        fp16: 2_015_271_040,
        tokenizerBytes: 19_226_111,
        contextLength: 262_144,
        license: 'Apache-2.0',
        supportsThinking: true,
      },
      'llama-3.2-3b': {
        repoId: 'onnx-community/Llama-3.2-3B-Instruct-ONNX',
        q4f16: 2_407_617_315,
        q4: 3_405_686_568,
        q8: 3_214_399_024,
        fp16: 6_459_242_427,
        tokenizerBytes: 11_574_638,
        contextLength: 131_072,
        license: 'llama3.2',
        supportsThinking: false,
      },
    };

    for (const [id, spec] of Object.entries(expected) as [ModelId, (typeof expected)[ModelId]][]) {
      const model = getModelById(id);
      expect(model.repoId).toBe(spec.repoId);
      expect(model.textOnlyBytesByDtype.q4f16).toBe(spec.q4f16);
      expect(model.textOnlyBytesByDtype.q4).toBe(spec.q4);
      expect(model.textOnlyBytesByDtype.q8).toBe(spec.q8);
      expect(model.textOnlyBytesByDtype.fp16).toBe(spec.fp16);
      expect(model.tokenizerBytes).toBe(spec.tokenizerBytes);
      expect(model.contextLength).toBe(spec.contextLength);
      expect(model.license).toBe(spec.license);
      expect(model.supportsThinking).toBe(spec.supportsThinking);
      expect(model.descriptionJa.length).toBeGreaterThan(10);
      expect(model.tags.length).toBeGreaterThan(0);
    }

    expect(getModelById('non-existent-model')).toBeUndefined();
    expect(getModelDownloadBytes(getModelById('qwen-3.5-0.8b'), 'q4f16')).toBe(
      584_705_674 + 19_226_111
    );
    expect(getModelDownloadBytes(getModelById('qwen-3.5-0.8b'), 'q4f16', false)).toBe(584_705_674);
  });
});

describe('resolveEffectiveDtype', () => {
  const gpuFp16 = makeWebGpuDevice(true);
  const gpuNoFp16 = makeWebGpuDevice(false);
  const wasmCpu = makeWasmDevice();

  it('picks q4f16 when WebGPU supports shader-f16 and falls back to q4 otherwise', () => {
    const gemma = getModelById('gemma-4-e2b');

    const withFp16 = resolveEffectiveDtype(gemma, gpuFp16);
    expect(withFp16).toEqual({
      dtype: 'q4f16',
      requestedDtype: undefined,
      adjusted: false,
    });
    expect(withFp16.useExternalDataFormat).toBeUndefined();

    const withoutFp16 = resolveEffectiveDtype(gemma, gpuNoFp16);
    expect(withoutFp16).toEqual({
      dtype: 'q4',
      requestedDtype: undefined,
      adjusted: false,
    });

    const explicitFallback = resolveEffectiveDtype(gemma, gpuNoFp16, 'q4f16');
    expect(explicitFallback.dtype).toBe('q4');
    expect(explicitFallback.adjusted).toBe(true);
    expect(explicitFallback.reason).toContain('shader-f16');

    const fp16Fallback = resolveEffectiveDtype(gemma, gpuNoFp16, 'fp16');
    expect(fp16Fallback.dtype).toBe('q4');
    expect(fp16Fallback.adjusted).toBe(true);
  });

  it('defaults CPU/WASM to q4 and avoids fp16/q4f16', () => {
    const qwen = getModelById('qwen-3.5-0.8b');

    expect(resolveEffectiveDtype(qwen, wasmCpu)).toEqual({
      dtype: 'q4',
      requestedDtype: undefined,
      adjusted: false,
    });

    const q4f16OnWasm = resolveEffectiveDtype(qwen, wasmCpu, 'q4f16');
    expect(q4f16OnWasm.dtype).toBe('q4');
    expect(q4f16OnWasm.adjusted).toBe(true);

    const q8OnQwenWasm = resolveEffectiveDtype(qwen, wasmCpu, 'q8');
    expect(q8OnQwenWasm).toEqual({
      dtype: 'q8',
      requestedDtype: 'q8',
      adjusted: false,
    });
  });

  it('guards against the Llama 3.2 3B q8 external-data trap unless explicitly opted in', () => {
    const llama = getModelById('llama-3.2-3b');
    expect(llama.requiresExplicitExternalDataForQ8).toBe(true);

    // Default WASM picks q4 safely
    expect(resolveEffectiveDtype(llama, wasmCpu).dtype).toBe('q4');

    // Explicit q8 on WASM without opt-in falls back to q4
    const llamaWasmQ8 = resolveEffectiveDtype(llama, wasmCpu, 'q8');
    expect(llamaWasmQ8.dtype).toBe('q4');
    expect(llamaWasmQ8.adjusted).toBe(true);
    expect(llamaWasmQ8.useExternalDataFormat).toBeUndefined();

    // Explicit q8 on WebGPU without opt-in falls back to q4f16 (with fp16) or q4
    const llamaGpuQ8 = resolveEffectiveDtype(llama, gpuFp16, 'q8');
    expect(llamaGpuQ8.dtype).toBe('q4f16');
    expect(llamaGpuQ8.adjusted).toBe(true);

    // Explicit q8 WITH useExternalDataFormat: true preserves q8 and signals override
    const llamaQ8OptIn = resolveEffectiveDtype(llama, wasmCpu, 'q8', {
      useExternalDataFormat: true,
    });
    expect(llamaQ8OptIn).toEqual({
      dtype: 'q8',
      requestedDtype: 'q8',
      useExternalDataFormat: true,
      adjusted: false,
    });

    // Other dtypes must keep useExternalDataFormat undefined so config.json chunk map stays intact
    expect(resolveEffectiveDtype(llama, gpuFp16, 'q4f16').useExternalDataFormat).toBeUndefined();
    expect(resolveEffectiveDtype(llama, wasmCpu, 'q4').useExternalDataFormat).toBeUndefined();
  });
});

describe('formatBytes', () => {
  it('formats zero, small, MiB, GiB, and SI values accurately', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(-100)).toBe('0 B');
    expect(formatBytes(Number.NaN)).toBe('0 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1536)).toBe('1.5 KiB');
    expect(formatBytes(584_705_674)).toBe('557.6 MiB');
    expect(formatBytes(584_705_674, { binary: false })).toBe('584.7 MB');
    expect(formatBytes(4_905_482_405)).toBe('4.57 GiB');
  });
});

describe('parseThinkingOutput', () => {
  it('parses Qwen 3.5 complete, unclosed streaming, and prompt-opened thinking tags', () => {
    // Complete <think>...</think>
    const complete = parseThinkingOutput(
      '<think>\nStep 1: 2 + 2 = 4\n</think>\n\nThe answer is 4.<|im_end|>',
      'qwen3_5'
    );
    expect(complete).toEqual({
      thinking: 'Step 1: 2 + 2 = 4',
      content: 'The answer is 4.',
      hasThinking: true,
      isThinkingInProgress: false,
    });

    // Unclosed streaming <think>...
    const streaming = parseThinkingOutput('<think>\nAnalyzing prompt...', 'qwen3_5');
    expect(streaming).toEqual({
      thinking: 'Analyzing prompt...',
      content: '',
      hasThinking: true,
      isThinkingInProgress: true,
    });

    // Empty disabled-thinking prefix `<think>\n\n</think>\n\n`
    const emptyThink = parseThinkingOutput('<think>\n\n</think>\n\nHello world', 'qwen3_5');
    expect(emptyThink).toEqual({
      thinking: '',
      content: 'Hello world',
      hasThinking: false,
      isThinkingInProgress: false,
    });

    // Prompt-pre-opened `<think>\n` stripped by TextStreamer(skip_prompt: true)
    const promptOpenedStreaming = parseThinkingOutput('Reasoning in progress', 'qwen3_5', {
      thinkingExpectedInPrompt: true,
    });
    expect(promptOpenedStreaming).toEqual({
      thinking: 'Reasoning in progress',
      content: '',
      hasThinking: true,
      isThinkingInProgress: true,
    });

    const promptOpenedComplete = parseThinkingOutput(
      'Reasoning done\n</think>\n\nFinal answer',
      'qwen3_5',
      { thinkingExpectedInPrompt: true }
    );
    expect(promptOpenedComplete).toEqual({
      thinking: 'Reasoning done',
      content: 'Final answer',
      hasThinking: true,
      isThinkingInProgress: false,
    });
  });

  it('parses Gemma 4 channel tags and stripped-special-token thought fallback', () => {
    // Complete `<|channel>thought\n...<channel|>`
    const complete = parseThinkingOutput(
      '<|channel>thought\nChecking constraints.\n<channel|>Here is the answer.<turn|>',
      'gemma4'
    );
    expect(complete).toEqual({
      thinking: 'Checking constraints.',
      content: 'Here is the answer.',
      hasThinking: true,
      isThinkingInProgress: false,
    });

    // Unclosed streaming `<|channel>thought\n...`
    const streaming = parseThinkingOutput('<|channel>thought\nThinking still...', 'gemma4');
    expect(streaming).toEqual({
      thinking: 'Thinking still...',
      content: '',
      hasThinking: true,
      isThinkingInProgress: true,
    });

    // Partial `<|channel>thou` header while streaming
    const partialTag = parseThinkingOutput('<|channel>thou', 'gemma4');
    expect(partialTag).toEqual({
      thinking: '',
      content: '',
      hasThinking: true,
      isThinkingInProgress: true,
    });

    // Fallback when skip_special_tokens stripped `<|channel>` and `<channel|>`
    const strippedStreaming = parseThinkingOutput('thought\nStep A\nStep B', 'gemma4');
    expect(strippedStreaming).toEqual({
      thinking: 'Step A\nStep B',
      content: '',
      hasThinking: true,
      isThinkingInProgress: true,
    });

    // Plain output without thinking tags
    const plain = parseThinkingOutput('Direct response<turn|>', 'gemma4');
    expect(plain).toEqual({
      thinking: '',
      content: 'Direct response',
      hasThinking: false,
      isThinkingInProgress: false,
    });
  });

  it('keeps Llama output as plain content and strips EOS artifacts', () => {
    const out = parseThinkingOutput('Llama response<|eot_id|>', 'llama');
    expect(out).toEqual({
      thinking: '',
      content: 'Llama response',
      hasThinking: false,
      isThinkingInProgress: false,
    });
  });
});
